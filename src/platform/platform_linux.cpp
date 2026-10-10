// Linux platform layer.
//
// Hotkey backends, in order of preference:
//   1. XRecord   (observes the chord without stealing it from other applications)
//   2. XGrabKey  (core Xlib passive grab; takes the chord away from other clients)
//   3. evdev     (/dev/input/event*; used when there is no X server at all)
//
// Everything X11-related is compiled only when CMake found X11 (ASR_HAVE_X11);
// XTest/XRecord use their own macros so a machine with a partial X11 install
// still builds.
#if defined(__linux__)

#  include "platform/platform.h"

#  include <algorithm>
#  include <atomic>
#  include <cctype>
#  include <cerrno>
#  include <chrono>
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#  include <chrono>
#  include <condition_variable>
#  include <dlfcn.h>
#  include <fcntl.h>
#  include <filesystem>
#  include <fstream>
#  include <future>
#  include <memory>
#  include <mutex>
#  include <poll.h>
#  include <pthread.h>
#  include <signal.h>
#  include <string>
#  include <sys/ioctl.h>
#  include <sys/wait.h>
#  include <thread>
#  include <unistd.h>
#  include <utility>
#  include <vector>

#  include <linux/input.h>

#  if defined(ASR_HAVE_X11)
#    include <X11/XKBlib.h>
#    include <X11/Xatom.h>
#    include <X11/Xlib.h>
#    include <X11/keysym.h>
#    if defined(ASR_HAVE_XRECORD)
#      include <X11/extensions/record.h>
#    endif
#    if defined(ASR_HAVE_XTEST)
#      include <X11/extensions/XTest.h>
#    endif
#  endif

namespace asr::platform {
namespace {

// ------------------------------------------------------------- small utils

long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string ascii_lower(const std::string & s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return r;
}

std::string read_first_line(const char * path) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line)) return std::string();
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    return line;
}

#  if defined(ASR_HAVE_X11)
// Our own XTest Ctrl+V shows up in the X11 backends as normal key events, so the
// listeners ignore keyboard input for a short window while a paste is in flight.
std::atomic<long long> g_inject_suppress_until{0};

bool injection_suppressed() {
    return now_ms() < g_inject_suppress_until.load();
}
#  endif

bool write_all(int fd, const char * data, size_t size) {
    while (size > 0) {
        const ssize_t n = write(fd, data, size);
        if (n > 0) {
            data += n;
            size -= static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

// wl-copy reads the text from stdin.  SIGPIPE is blocked around the write: the
// child may already be gone (no wl-copy installed), which must not kill us.
bool wl_copy(const std::string & text) {
    int fds[2];
    if (pipe(fds) != 0) return false;
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    if (pid == 0) {
        close(fds[1]);
        if (dup2(fds[0], STDIN_FILENO) < 0) _exit(127);
        close(fds[0]);
        execlp("wl-copy", "wl-copy", static_cast<char *>(nullptr));
        _exit(127);
    }
    close(fds[0]);

    sigset_t block;
    sigset_t old;
    sigemptyset(&block);
    sigaddset(&block, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &block, &old);
    const bool wrote = write_all(fds[1], text.data(), text.size());
    close(fds[1]);
    struct timespec zero;
    zero.tv_sec = 0;
    zero.tv_nsec = 0;
    sigtimedwait(&block, nullptr, &zero);  // drop a pending SIGPIPE, if any
    pthread_sigmask(SIG_SETMASK, &old, nullptr);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    return wrote && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

bool wl_paste(std::string * out) {
    int fds[2];
    if (pipe(fds) != 0) return false;
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    if (pid == 0) {
        close(fds[0]);
        if (dup2(fds[1], STDOUT_FILENO) < 0) _exit(127);
        close(fds[1]);
        execlp("wl-paste", "wl-paste", "--no-newline", static_cast<char *>(nullptr));
        _exit(127);
    }
    close(fds[1]);

    std::string data;
    char buf[4096];
    const long long deadline = now_ms() + 1500;
    bool timed_out = false;
    for (;;) {
        struct pollfd pfd;
        pfd.fd = fds[0];
        pfd.events = POLLIN;
        pfd.revents = 0;
        const long long left = deadline - now_ms();
        const int n = poll(&pfd, 1, left > 0 ? static_cast<int>(left) : 0);
        if (n == 0) {
            timed_out = true;
            break;
        }
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        const ssize_t r = read(fds[0], buf, sizeof(buf));
        if (r > 0) {
            data.append(buf, static_cast<size_t>(r));
            continue;
        }
        if (r < 0 && errno == EINTR) continue;
        break;  // EOF (child done) or read error
    }
    close(fds[0]);
    int status = 0;
    if (timed_out) kill(pid, SIGKILL);
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (timed_out) return false;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return false;
    *out = std::move(data);
    return true;
}

// ---------------------------------------------------------------- X11 bits

#  if defined(ASR_HAVE_X11)

// X errors are asynchronous; a fatal (default) handler would kill the process on
// a failed XGrabKey, so the handler only records the last error code.
std::atomic<int> g_last_x_error{0};

int x11_error_handler(Display *, XErrorEvent * event) {
    g_last_x_error.store(event->error_code);
    return 0;  // keep going
}

// XInitThreads must run before the first Xlib call; the clipboard owner and the
// listeners each use their own Display from their own thread.
void x11_global_init() {
    static std::once_flag once;
    std::call_once(once, [] {
        XInitThreads();
        XSetErrorHandler(x11_error_handler);
    });
}

KeySym keysym_for_key(const std::string & key) {
    if (key.size() == 1) {
        const char c = key[0];
        if (c >= 'A' && c <= 'Z') return XK_a + (c - 'A');
        if (c >= '0' && c <= '9') return XK_0 + (c - '0');
        return NoSymbol;
    }
    if (key.size() >= 2 && key[0] == 'F') {
        const int n = std::atoi(key.c_str() + 1);
        if (n >= 1 && n <= 24) return XK_F1 + (n - 1);  // XK_F1..XK_F24 are contiguous
        return NoSymbol;
    }
    if (key == "Space") return XK_space;
    if (key == "ScrollLock") return XK_Scroll_Lock;
    if (key == "Pause") return XK_Pause;
    if (key == "Insert") return XK_Insert;
    if (key == "Home") return XK_Home;
    if (key == "End") return XK_End;
    if (key == "PageUp") return XK_Page_Up;
    if (key == "PageDown") return XK_Page_Down;
    if (key == "RightCtrl") return XK_Control_R;
    if (key == "RightAlt") return XK_Alt_R;
    if (key == "RightShift") return XK_Shift_R;
    return NoSymbol;
}

// The chord resolved against the current keyboard mapping.
struct ChordKeys {
    unsigned int main_code = 0;
    KeySym       main_sym = NoSymbol;
    std::vector<unsigned int> ctrl_codes;
    std::vector<unsigned int> alt_codes;
    std::vector<unsigned int> shift_codes;
    std::vector<unsigned int> win_codes;

    bool is_modifier_sym(KeySym ks) const {
        if (ctrl_has() && (ks == XK_Control_L || ks == XK_Control_R)) return true;
        if (alt_has() && (ks == XK_Alt_L || ks == XK_Alt_R)) return true;
        if (shift_has() && (ks == XK_Shift_L || ks == XK_Shift_R)) return true;
        if (win_has() && (ks == XK_Super_L || ks == XK_Super_R)) return true;
        return false;
    }
    bool is_modifier_code(unsigned int code) const {
        return contains(ctrl_codes, code) || contains(alt_codes, code) ||
               contains(shift_codes, code) || contains(win_codes, code);
    }
    bool ctrl_has() const { return !ctrl_codes.empty(); }
    bool alt_has() const { return !alt_codes.empty(); }
    bool shift_has() const { return !shift_codes.empty(); }
    bool win_has() const { return !win_codes.empty(); }

    static bool contains(const std::vector<unsigned int> & v, unsigned int code) {
        return std::find(v.begin(), v.end(), code) != v.end();
    }
};

// Which modifier keys of the chord are currently held, tracked by keycode.
struct ModState {
    std::vector<unsigned int> held;

    void set(unsigned int code, bool down) {
        const auto it = std::find(held.begin(), held.end(), code);
        if (down) {
            if (it == held.end()) held.push_back(code);
        } else if (it != held.end()) {
            held.erase(it);
        }
    }
    bool any(const std::vector<unsigned int> & codes) const {
        for (unsigned int code : codes) {
            if (std::find(held.begin(), held.end(), code) != held.end()) return true;
        }
        return false;
    }
};

// Physical X keycode (evdev code + 8) for keys whose keysym moves between layout
// groups: with a Cyrillic group active XKeysymToKeycode(XK_a) returns 0, yet the
// user means the physical key.  Only used when the keysym lookup found nothing.
unsigned int evdev_keycode_for_key(const std::string & key) {
    static const unsigned int kLetters[26] = {38, 56, 54, 40, 26, 41, 42, 43, 31, 44, 45, 46, 58,
                                              57, 32, 33, 24, 27, 39, 28, 30, 55, 25, 53, 29, 52};
    static const unsigned int kDigits[10] = {19, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    if (key.size() == 1) {
        const char c = key[0];
        if (c >= 'A' && c <= 'Z') return kLetters[c - 'A'];
        if (c >= '0' && c <= '9') return kDigits[c - '0'];
    }
    return 0;
}

void resolve_chord(Display * dpy, const Chord & chord, ChordKeys * out) {
    out->main_sym = keysym_for_key(chord.key);
    out->main_code = static_cast<unsigned int>(XKeysymToKeycode(dpy, out->main_sym));
    if (out->main_code == 0) out->main_code = evdev_keycode_for_key(chord.key);
    auto add = [&](KeySym ks, std::vector<unsigned int> * v) {
        const unsigned int code = static_cast<unsigned int>(XKeysymToKeycode(dpy, ks));
        if (code != 0) v->push_back(code);
    };
    if (chord.ctrl) {
        add(XK_Control_L, &out->ctrl_codes);
        add(XK_Control_R, &out->ctrl_codes);
    }
    if (chord.alt) {
        add(XK_Alt_L, &out->alt_codes);
        add(XK_Alt_R, &out->alt_codes);
    }
    if (chord.shift) {
        add(XK_Shift_L, &out->shift_codes);
        add(XK_Shift_R, &out->shift_codes);
    }
    if (chord.win) {
        add(XK_Super_L, &out->win_codes);
        add(XK_Super_R, &out->win_codes);
    }
}

struct KeyTracker {
    ChordKeys keys;
    ModState  mods;
    bool      main_down = false;  // physically down as far as we know
    bool      pressed = false;    // on_press was delivered, release still owed

    bool modifiers_held(const Chord & chord) const {
        if (chord.ctrl && !mods.any(keys.ctrl_codes)) return false;
        if (chord.alt && !mods.any(keys.alt_codes)) return false;
        if (chord.shift && !mods.any(keys.shift_codes)) return false;
        if (chord.win && !mods.any(keys.win_codes)) return false;
        return true;
    }
};

// ------------------------------------------------- 1. XRecord (no stealing)

#    if defined(ASR_HAVE_XRECORD)

class X11RecordListener final : public HotkeyListener {
public:
    X11RecordListener(const Chord & chord, std::function<void()> on_press,
                      std::function<void()> on_release)
        : chord_(chord), on_press_(std::move(on_press)), on_release_(std::move(on_release)) {}

    ~X11RecordListener() override { stop(); }

    bool start(std::string * err) override {
        // A previous run may have finished without being joined yet.
        if (thread_.joinable() && !started_.load()) {
            if (thread_.get_id() == std::this_thread::get_id()) return false;
            thread_.join();
        }
        if (started_.load()) return true;
        x11_global_init();
        const bool trace = std::getenv("ASR_HOTKEY_DEBUG") != nullptr;
        auto dbg = [&](const char * what) {
            if (trace) {
                std::fprintf(stderr, "[hotkey/xrecord] %s\n", what);
                std::fflush(stderr);
            }
        };

        dbg("XOpenDisplay(query)");
        Display * query = XOpenDisplay(nullptr);
        if (query == nullptr) {
            if (err != nullptr) *err = "cannot open X11 display";
            return false;
        }
        dbg("XOpenDisplay(record)");
        Display * record = XOpenDisplay(nullptr);
        if (record == nullptr) {
            XCloseDisplay(query);
            if (err != nullptr) *err = "cannot open X11 display";
            return false;
        }

        resolve_chord(query, chord_, &tracker_.keys);
        dbg("resolve_chord done");
        if (tracker_.keys.main_sym == NoSymbol) {
            close_plain(record);
            close_plain(query);
            if (err != nullptr) *err = "unsupported hotkey key: " + chord_.key;
            return false;
        }
        if (tracker_.keys.main_code == 0) {
            close_plain(record);
            close_plain(query);
            if (err != nullptr) *err = "hotkey key not present on this layout: " + chord_.key;
            return false;
        }

        int major = 0;
        int minor = 0;
        g_last_x_error.store(0);
        if (XRecordQueryVersion(record, &major, &minor) == 0) {
            close_plain(record);
            close_plain(query);
            if (err != nullptr) *err = "XRecord extension not available";
            return false;
        }

        XRecordRange * range = XRecordAllocRange();
        if (range == nullptr) {
            close_plain(record);
            close_plain(query);
            if (err != nullptr) *err = "XRecordAllocRange failed";
            return false;
        }
        range->device_events.first = KeyPress;
        range->device_events.last = KeyRelease;
        XRecordClientSpec clients = XRecordAllClients;
        const XRecordContext context =
            XRecordCreateContext(record, 0, &clients, 1, &range, 1);
        XFree(range);
        if (context == 0) {
            close_plain(record);
            close_plain(query);
            if (err != nullptr) *err = "XRecordCreateContext failed";
            return false;
        }
        // Only sync while the connection is still usable for normal requests.
        XSync(record, False);
        if (g_last_x_error.load() != 0) {
            const int code = g_last_x_error.load();
            drop_record(record);
            close_plain(query);
            if (err != nullptr) {
                *err = "XRecordCreateContext failed (X error " + std::to_string(code) + ")";
            }
            return false;
        }
        // From here on the record display may only be used for XRecord calls.
        if (XRecordEnableContextAsync(record, context, &X11RecordListener::intercept,
                                      reinterpret_cast<XPointer>(this)) == 0) {
            drop_record(record);
            close_plain(query);
            if (err != nullptr) *err = "XRecordEnableContextAsync failed";
            return false;
        }

        query_ = query;
        record_ = record;
        context_ = context;
        stop_.store(false);
        started_.store(true);
        thread_ = std::thread([this] { run(); });
        return true;
    }

    void stop() override {
        stop_.store(true);
        if (thread_.joinable()) {
            if (thread_.get_id() == std::this_thread::get_id()) {
                // Called from the intercept callback: let the loop unwind itself.
                return;
            }
            thread_.join();
        }
        // run() released the X resources on its way out.
    }

    const char * backend_name() const override { return "xrecord"; }

private:
    // XCloseDisplay() deadlocks on a connection that carries (or once carried) an
    // enabled record context: libXtst's close hook waits for a server reply that
    // never arrives once record data has been in flight.  Verified on
    // WSLg/Xwayland - XRecordDisableContext right after enabling hangs as well.
    // Dropping the socket is enough: the server destroys the context when the
    // client disconnects.
    static void drop_record(Display * dpy) {
        if (dpy == nullptr) return;
        close(ConnectionNumber(dpy));
    }

    // Safe only while no record context exists on that connection.
    static void close_plain(Display * dpy) {
        if (dpy != nullptr) XCloseDisplay(dpy);
    }

    static void intercept(XPointer closure, XRecordInterceptData * data) {
        // XPointer is char*, so the listener pointer round-trips through it.
        reinterpret_cast<X11RecordListener *>(closure)->on_record(*data);
        XRecordFreeData(data);  // mandatory, the server hands the buffer to us
    }

    void on_record(const XRecordInterceptData & data) {
        // Core device events arrive as the raw 32 byte wire event; type and
        // keycode ("detail") are plain bytes, so no byte-order handling is needed.
        if (data.category != XRecordFromServer || data.data == nullptr) return;
        if (data.data_len < 8) return;
        const unsigned char type = data.data[0];
        const unsigned int keycode = data.data[1];
        if (type == KeyPress) handle_key(keycode, true);
        else if (type == KeyRelease) handle_key(keycode, false);
    }

    void handle_key(unsigned int keycode, bool press) {
        if (injection_suppressed()) return;
        const KeySym ks = XkbKeycodeToKeysym(query_, keycode, 0, 0);
        const bool is_main =
            keycode == tracker_.keys.main_code ||
            (ks != NoSymbol && ks == tracker_.keys.main_sym);

        if (press) {
            if (is_main) {
                // "Ctrl+RightCtrl" style chords: the main key is also a modifier.
                if (tracker_.keys.is_modifier_code(keycode)) {
                    tracker_.mods.set(keycode, true);
                }
                if (!tracker_.main_down) {
                    tracker_.main_down = true;
                    if (tracker_.modifiers_held(chord_)) {
                        tracker_.pressed = true;
                        if (on_press_) on_press_();
                    }
                }
                return;
            }
            if (tracker_.keys.is_modifier_sym(ks)) tracker_.mods.set(keycode, true);
            return;
        }

        if (is_main) {
            if (tracker_.keys.is_modifier_code(keycode)) tracker_.mods.set(keycode, false);
            // XKB detectable auto-repeat (enabled process-wide, e.g. by GLFW)
            // reports a held key as a KeyRelease/KeyPress pair.  The key is still
            // down on the server, so only a release the keymap confirms counts.
            if (keycode_down(keycode)) return;
            tracker_.main_down = false;
            if (tracker_.pressed) {
                tracker_.pressed = false;
                if (on_release_) on_release_();
            }
            return;
        }
        if (tracker_.keys.is_modifier_sym(ks)) {
            tracker_.mods.set(keycode, false);
            if (tracker_.pressed && !tracker_.modifiers_held(chord_)) {
                tracker_.pressed = false;
                tracker_.main_down = false;
                if (on_release_) on_release_();
            }
        }
    }

    bool keycode_down(unsigned int keycode) {
        char keys[32];
        std::memset(keys, 0, sizeof(keys));
        XQueryKeymap(query_, keys);
        const unsigned int index = keycode / 8;
        if (index >= sizeof(keys)) return false;
        return (static_cast<unsigned char>(keys[index]) & (1u << (keycode % 8))) != 0;
    }

    void run() {
        while (!stop_.load()) {
            struct pollfd pfd;
            pfd.fd = ConnectionNumber(record_);
            pfd.events = POLLIN;
            pfd.revents = 0;
            const int n = poll(&pfd, 1, 50);
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (n == 0) continue;
            if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) break;
            if ((pfd.revents & POLLIN) != 0) {
                XRecordProcessReplies(record_);  // invokes intercept() per event
            }
        }
        // Release the X resources here, on the thread that owns them: no blocking
        // request is issued, the connection is simply dropped (see drop_record()).
        for (int i = 0; i < 100; ++i) {
            struct pollfd pfd;
            pfd.fd = ConnectionNumber(record_);
            pfd.events = POLLIN;
            pfd.revents = 0;
            if (poll(&pfd, 1, 0) <= 0 || (pfd.revents & POLLIN) == 0) break;
            XRecordProcessReplies(record_);
        }
        drop_record(record_);
        record_ = nullptr;
        context_ = 0;
        close_plain(query_);
        query_ = nullptr;
        started_.store(false);
    }

    Chord                 chord_;
    std::function<void()> on_press_;
    std::function<void()> on_release_;

    KeyTracker          tracker_;
    Display *           query_ = nullptr;   // keymap lookups (record thread only)
    Display *           record_ = nullptr;  // owned by the record extension
    XRecordContext      context_ = 0;
    std::atomic<bool>   stop_{false};
    std::atomic<bool>   started_{false};
    std::thread         thread_;
};

#    endif  // ASR_HAVE_XRECORD

// ------------------------------------------- 2. XGrabKey (steals the chord)

class X11GrabListener final : public HotkeyListener {
public:
    X11GrabListener(const Chord & chord, std::function<void()> on_press,
                    std::function<void()> on_release)
        : chord_(chord), on_press_(std::move(on_press)), on_release_(std::move(on_release)) {}

    ~X11GrabListener() override { stop(); }

    bool start(std::string * err) override {
        if (started_.load()) return true;
        x11_global_init();

        Display * dpy = XOpenDisplay(nullptr);
        if (dpy == nullptr) {
            if (err != nullptr) *err = "cannot open X11 display";
            return false;
        }
        resolve_chord(dpy, chord_, &tracker_.keys);
        if (tracker_.keys.main_sym == NoSymbol) {
            XCloseDisplay(dpy);
            if (err != nullptr) *err = "unsupported hotkey key: " + chord_.key;
            return false;
        }
        if (tracker_.keys.main_code == 0) {
            XCloseDisplay(dpy);
            if (err != nullptr) *err = "hotkey key not present on this layout: " + chord_.key;
            return false;
        }

        unsigned int mods = 0;
        if (chord_.ctrl) mods |= ControlMask;
        if (chord_.alt) mods |= Mod1Mask;
        if (chord_.shift) mods |= ShiftMask;
        if (chord_.win) mods |= Mod4Mask;
        const Window root = DefaultRootWindow(dpy);
        const unsigned int locks[4] = {0u, LockMask, Mod2Mask, LockMask | Mod2Mask};

        g_last_x_error.store(0);
        for (unsigned int lock : locks) {
            XGrabKey(dpy, static_cast<int>(tracker_.keys.main_code), mods | lock, root, False,
                     GrabModeAsync, GrabModeAsync);
        }
        XSync(dpy, False);
        if (g_last_x_error.load() != 0) {
            const int code = g_last_x_error.load();
            for (unsigned int lock : locks) {
                XUngrabKey(dpy, static_cast<int>(tracker_.keys.main_code), mods | lock, root);
            }
            XSync(dpy, False);
            XCloseDisplay(dpy);
            if (err != nullptr) {
                *err = "cannot grab hotkey (X error " + std::to_string(code) +
                       "); it is probably already grabbed by another application";
            }
            return false;
        }

        XSelectInput(dpy, root, KeyPressMask | KeyReleaseMask);
        Bool supported = False;
        XkbSetDetectableAutoRepeat(dpy, True, &supported);

        dpy_ = dpy;
        root_ = root;
        mods_ = mods;
        stop_.store(false);
        started_.store(true);
        thread_ = std::thread([this] { run(); });
        return true;
    }

    void stop() override {
        stop_.store(true);
        if (thread_.joinable()) {
            if (thread_.get_id() == std::this_thread::get_id()) return;  // called from a callback
            thread_.join();
        }
        if (!started_.exchange(false)) return;
        if (dpy_ != nullptr) {
            const unsigned int locks[4] = {0u, LockMask, Mod2Mask, LockMask | Mod2Mask};
            for (unsigned int lock : locks) {
                XUngrabKey(dpy_, static_cast<int>(tracker_.keys.main_code), mods_ | lock, root_);
            }
            XSync(dpy_, False);
            XCloseDisplay(dpy_);
            dpy_ = nullptr;
        }
    }

    const char * backend_name() const override { return "x11-grab"; }

private:
    void run() {
        int idle = 0;
        while (!stop_.load()) {
            bool had_event = false;
            while (XPending(dpy_) > 0) {
                XEvent ev;
                XNextEvent(dpy_, &ev);
                handle_event(ev);
                had_event = true;
            }
            if (had_event) {
                idle = 0;
            } else if (++idle >= 10) {
                // Every ~500 ms: a broken grab (e.g. the modifier was released
                // first) can swallow the main key release, so resync with the
                // server's keymap instead of staying pressed forever.
                idle = 0;
                if (tracker_.main_down || tracker_.pressed) resync();
            }
            struct pollfd pfd;
            pfd.fd = ConnectionNumber(dpy_);
            pfd.events = POLLIN;
            pfd.revents = 0;
            const int n = poll(&pfd, 1, 50);
            if (n < 0 && errno != EINTR) break;
        }
    }

    void handle_event(const XEvent & ev) {
        if (ev.type != KeyPress && ev.type != KeyRelease) return;
        if (injection_suppressed()) return;  // ignore our own Ctrl+V
        const unsigned int keycode = ev.xkey.keycode;
        const KeySym ks = XkbKeycodeToKeysym(dpy_, keycode, 0, 0);
        const bool is_main =
            keycode == tracker_.keys.main_code ||
            (ks != NoSymbol && ks == tracker_.keys.main_sym);

        if (ev.type == KeyPress) {
            if (is_main) {
                if (tracker_.keys.is_modifier_code(keycode)) tracker_.mods.set(keycode, true);
                if (!tracker_.main_down) {
                    tracker_.main_down = true;
                    if (tracker_.modifiers_held(chord_)) {
                        tracker_.pressed = true;
                        if (on_press_) on_press_();
                    }
                }
                return;
            }
            if (tracker_.keys.is_modifier_sym(ks)) tracker_.mods.set(keycode, true);
            return;
        }

        if (is_main) {
            if (tracker_.keys.is_modifier_code(keycode)) tracker_.mods.set(keycode, false);
            tracker_.main_down = false;
            if (tracker_.pressed) {
                tracker_.pressed = false;
                if (on_release_) on_release_();
            }
            return;
        }
        if (tracker_.keys.is_modifier_sym(ks)) {
            tracker_.mods.set(keycode, false);
            if (tracker_.pressed && !tracker_.modifiers_held(chord_)) {
                tracker_.pressed = false;
                tracker_.main_down = false;
                if (on_release_) on_release_();
            }
        }
    }

    void resync() {
        char keys[32];
        std::memset(keys, 0, sizeof(keys));
        XQueryKeymap(dpy_, keys);
        auto down = [&keys](unsigned int code) {
            return (static_cast<unsigned char>(keys[code / 8]) & (1u << (code % 8))) != 0;
        };
        const bool main_down =
            tracker_.keys.main_code != 0 && down(tracker_.keys.main_code);
        bool mods_held = true;
        if (chord_.ctrl && !any_down(down, tracker_.keys.ctrl_codes)) mods_held = false;
        if (chord_.alt && !any_down(down, tracker_.keys.alt_codes)) mods_held = false;
        if (chord_.shift && !any_down(down, tracker_.keys.shift_codes)) mods_held = false;
        if (chord_.win && !any_down(down, tracker_.keys.win_codes)) mods_held = false;
        if (tracker_.pressed && (!main_down || !mods_held)) {
            tracker_.pressed = false;
            tracker_.main_down = false;
            if (on_release_) on_release_();
        } else if (tracker_.main_down && !main_down) {
            tracker_.main_down = false;
        }
    }

    template <typename DownFn>
    bool any_down(DownFn down, const std::vector<unsigned int> & codes) {
        for (unsigned int code : codes) {
            if (down(code)) return true;
        }
        return false;
    }

    Chord                 chord_;
    std::function<void()> on_press_;
    std::function<void()> on_release_;

    KeyTracker        tracker_;
    Display *         dpy_ = nullptr;
    Window            root_ = 0;
    unsigned int      mods_ = 0;
    std::atomic<bool> stop_{false};
    std::atomic<bool> started_{false};
    std::thread       thread_;
};

#  endif  // ASR_HAVE_X11

// ------------------------------------------------------- 3. evdev fallback

// evdev key codes; the interesting ones are the fixed KEY_* constants from
// <linux/input.h> (KEY_LEFTCTRL=29, KEY_LEFTALT=56, KEY_LEFTSHIFT=42,
// KEY_LEFTMETA=125, right hand variants 97/100/54/126).
int evdev_key_for(const std::string & key) {
    if (key.size() == 1) {
        const char c = key[0];
        if (c >= 'A' && c <= 'Z') {
            static const int kLetters[26] = {
                KEY_A, KEY_B, KEY_C, KEY_D, KEY_E, KEY_F, KEY_G, KEY_H, KEY_I,
                KEY_J, KEY_K, KEY_L, KEY_M, KEY_N, KEY_O, KEY_P, KEY_Q, KEY_R,
                KEY_S, KEY_T, KEY_U, KEY_V, KEY_W, KEY_X, KEY_Y, KEY_Z};
            return kLetters[c - 'A'];
        }
        if (c >= '0' && c <= '9') {
            static const int kDigits[10] = {KEY_0, KEY_1, KEY_2, KEY_3, KEY_4,
                                            KEY_5, KEY_6, KEY_7, KEY_8, KEY_9};
            return kDigits[c - '0'];
        }
        return 0;
    }
    if (key.size() >= 2 && key[0] == 'F') {
        const int n = std::atoi(key.c_str() + 1);
        if (n >= 1 && n <= 12) return KEY_F1 + (n - 1);
        if (n >= 13 && n <= 24) return KEY_F13 + (n - 13);
        return 0;
    }
    if (key == "Space") return KEY_SPACE;
    if (key == "ScrollLock") return KEY_SCROLLLOCK;
    if (key == "Pause") return KEY_PAUSE;
    if (key == "Insert") return KEY_INSERT;
    if (key == "Home") return KEY_HOME;
    if (key == "End") return KEY_END;
    if (key == "PageUp") return KEY_PAGEUP;
    if (key == "PageDown") return KEY_PAGEDOWN;
    if (key == "RightCtrl") return KEY_RIGHTCTRL;
    if (key == "RightAlt") return KEY_RIGHTALT;
    if (key == "RightShift") return KEY_RIGHTSHIFT;
    return 0;
}

class EvdevListener final : public HotkeyListener {
public:
    EvdevListener(const Chord & chord, std::function<void()> on_press,
                  std::function<void()> on_release)
        : chord_(chord), on_press_(std::move(on_press)), on_release_(std::move(on_release)) {}

    ~EvdevListener() override { stop(); }

    bool start(std::string * err) override {
        if (started_.load()) return true;
        main_code_ = evdev_key_for(chord_.key);
        if (main_code_ == 0) {
            if (err != nullptr) *err = "unsupported hotkey key: " + chord_.key;
            return false;
        }
        if (chord_.ctrl) add_codes(&ctrl_codes_, KEY_LEFTCTRL, KEY_RIGHTCTRL);
        if (chord_.alt) add_codes(&alt_codes_, KEY_LEFTALT, KEY_RIGHTALT);
        if (chord_.shift) add_codes(&shift_codes_, KEY_LEFTSHIFT, KEY_RIGHTSHIFT);
        if (chord_.win) add_codes(&win_codes_, KEY_LEFTMETA, KEY_RIGHTMETA);

        for (int i = 0; i < 32 && fd_ < 0; ++i) {
            char path[32];
            std::snprintf(path, sizeof(path), "/dev/input/event%d", i);
            const int fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
            if (fd < 0) continue;
            if (!looks_like_keyboard(fd)) {
                close(fd);
                continue;
            }
            fd_ = fd;
        }
        if (fd_ < 0) {
            if (err != nullptr) {
                *err = "cannot grab hotkey: no X11 display and no readable /dev/input/event* "
                       "(add the user to the input group)";
            }
            return false;
        }

        key_down_.assign(KEY_MAX + 1, 0);
        // Prime the state so a modifier that is already held when we start counts.
        unsigned char state[(KEY_MAX / 8) + 1];
        std::memset(state, 0, sizeof(state));
        if (ioctl(fd_, EVIOCGKEY(sizeof(state)), state) >= 0) {
            for (int code = 0; code <= KEY_MAX; ++code) {
                key_down_[static_cast<size_t>(code)] =
                    (state[code / 8] & (1 << (code % 8))) != 0 ? 1 : 0;
            }
        }

        stop_.store(false);
        started_.store(true);
        thread_ = std::thread([this] { run(); });
        return true;
    }

    void stop() override {
        stop_.store(true);
        if (thread_.joinable()) {
            if (thread_.get_id() == std::this_thread::get_id()) return;
            thread_.join();
        }
        if (!started_.exchange(false)) return;
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    const char * backend_name() const override { return "evdev"; }

private:
    static void add_codes(std::vector<int> * v, int a, int b) {
        v->push_back(a);
        v->push_back(b);
    }

    bool looks_like_keyboard(int fd) const {
        unsigned long ev_bits = 0;
        if (ioctl(fd, EVIOCGBIT(0, sizeof(ev_bits)), &ev_bits) < 0) return false;
        if ((ev_bits & (1UL << EV_KEY)) == 0) return false;
        unsigned char key_bits[(KEY_MAX / 8) + 1];
        std::memset(key_bits, 0, sizeof(key_bits));
        if (ioctl(fd, EVIOCGBIT(EV_KEY, sizeof(key_bits)), key_bits) < 0) return false;
        auto has = [&key_bits](int code) {
            return (key_bits[code / 8] & (1 << (code % 8))) != 0;
        };
        if (!has(KEY_A) && !has(KEY_SPACE)) return false;
        char name[256];
        std::memset(name, 0, sizeof(name));
        if (ioctl(fd, EVIOCGNAME(sizeof(name)), name) >= 0) {
            const std::string lowered = ascii_lower(name);
            static const char * kSkip[] = {"video bus", "power button", "sleep button",
                                           "pc speaker", "wireless radio"};
            for (const char * skip : kSkip) {
                if (lowered.find(skip) != std::string::npos) return false;
            }
        }
        return true;
    }

    bool modifiers_down() const {
        if (chord_.ctrl && !any_down(ctrl_codes_)) return false;
        if (chord_.alt && !any_down(alt_codes_)) return false;
        if (chord_.shift && !any_down(shift_codes_)) return false;
        if (chord_.win && !any_down(win_codes_)) return false;
        return true;
    }

    bool any_down(const std::vector<int> & codes) const {
        for (int code : codes) {
            if (code >= 0 && code <= KEY_MAX && key_down_[static_cast<size_t>(code)] != 0) {
                return true;
            }
        }
        return false;
    }

    bool is_modifier(int code) const {
        return std::find(ctrl_codes_.begin(), ctrl_codes_.end(), code) != ctrl_codes_.end() ||
               std::find(alt_codes_.begin(), alt_codes_.end(), code) != alt_codes_.end() ||
               std::find(shift_codes_.begin(), shift_codes_.end(), code) != shift_codes_.end() ||
               std::find(win_codes_.begin(), win_codes_.end(), code) != win_codes_.end();
    }

    // value: 0 = release, 1 = press, 2 = auto-repeat.  The device is not grabbed,
    // so the key still reaches the focused window.
    void handle_key(int code, int value) {
        if (code < 0 || code > KEY_MAX || value == 2) return;
        const bool down = value == 1;
        key_down_[static_cast<size_t>(code)] = down ? 1 : 0;

        if (code == main_code_) {
            if (down) {
                if (!main_down_) {
                    main_down_ = true;
                    if (modifiers_down()) {
                        pressed_ = true;
                        if (on_press_) on_press_();
                    }
                }
            } else {
                main_down_ = false;
                if (pressed_) {
                    pressed_ = false;
                    if (on_release_) on_release_();
                }
            }
            return;
        }
        if (!down && is_modifier(code) && pressed_) {
            pressed_ = false;
            main_down_ = false;
            if (on_release_) on_release_();
        }
    }

    void run() {
        struct input_event ev;
        for (;;) {
            if (stop_.load()) break;
            struct pollfd pfd;
            pfd.fd = fd_;
            pfd.events = POLLIN;
            pfd.revents = 0;
            const int n = poll(&pfd, 1, 50);
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (n == 0) continue;
            if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) break;
            bool device_gone = false;
            for (;;) {
                const ssize_t r = read(fd_, &ev, sizeof(ev));
                if (r == static_cast<ssize_t>(sizeof(ev))) {
                    if (ev.type == EV_KEY) handle_key(ev.code, ev.value);
                    continue;
                }
                if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
                if (r < 0 && errno == EINTR) continue;
                device_gone = true;  // 0 or a real error: the device vanished
                break;
            }
            if (device_gone) break;
        }
    }

    Chord                 chord_;
    std::function<void()> on_press_;
    std::function<void()> on_release_;

    int               fd_ = -1;
    int               main_code_ = 0;
    std::vector<int>  ctrl_codes_;
    std::vector<int>  alt_codes_;
    std::vector<int>  shift_codes_;
    std::vector<int>  win_codes_;
    std::vector<char> key_down_;
    bool              main_down_ = false;
    bool              pressed_ = false;
    std::atomic<bool> stop_{false};
    std::atomic<bool> started_{false};
    std::thread       thread_;
};

// ------------------------------------------------------ backend selection

const char * default_backend_name() {
#  if defined(ASR_HAVE_XRECORD)
    return "xrecord";
#  elif defined(ASR_HAVE_X11)
    return "x11-grab";
#  else
    return "evdev";
#  endif
}

// Tries the backends in order of preference and keeps the first one that starts.
class AutoHotkeyListener final : public HotkeyListener {
public:
    AutoHotkeyListener(const Chord & chord, std::function<void()> on_press,
                       std::function<void()> on_release)
        : chord_(chord), on_press_(std::move(on_press)), on_release_(std::move(on_release)) {}

    bool start(std::string * err) override {
        // Still running: start() is idempotent and the backend reports success.
        if (inner_ != nullptr) return inner_->start(err);
        std::string reason;
#  if defined(ASR_HAVE_XRECORD)
        if (try_backend(std::make_unique<X11RecordListener>(chord_, on_press_, on_release_),
                        &reason)) {
            return true;
        }
#  endif
#  if defined(ASR_HAVE_X11)
        if (try_backend(std::make_unique<X11GrabListener>(chord_, on_press_, on_release_),
                        &reason)) {
            return true;
        }
#  endif
        if (try_backend(std::make_unique<EvdevListener>(chord_, on_press_, on_release_), &reason)) {
            return true;
        }
        if (err != nullptr) {
            *err = "cannot grab hotkey: no X11 display and no readable /dev/input/event* "
                   "(add the user to the input group)";
            // Keep the most specific reason (e.g. an unmapped key) visible.
            if (!reason.empty()) *err += " [" + reason + "]";
        }
        return false;
    }

    void stop() override {
        if (inner_ != nullptr) {
            inner_->stop();
            // Drop the backend so a later start() picks and initialises one again.
            inner_.reset();
        }
    }

    const char * backend_name() const override {
        return inner_ != nullptr ? inner_->backend_name() : default_backend_name();
    }

private:
    // Starts the candidate with a bounded wait.  XOpenDisplay() can block for a
    // long time when DISPLAY points at a server that is gone (a sandbox, a stale
    // session), and that must never hang the application: after
    // `kStartTimeoutMs` the candidate is parked (kept alive, never used) and the
    // next backend is tried.
    bool try_backend(std::unique_ptr<HotkeyListener> candidate, std::string * reason) {
        struct State {
            std::mutex mutex;
            std::condition_variable cv;
            bool done = false;
            bool ok = false;
            std::string err;
        };
        auto state = std::make_shared<State>();
        HotkeyListener * raw = candidate.get();
        std::thread starter([raw, state]() {
            const bool ok = raw->start(&state->err);
            {
                std::lock_guard<std::mutex> lk(state->mutex);
                state->ok = ok;
                state->done = true;
            }
            state->cv.notify_all();
        });

        bool timed_out = false;
        {
            std::unique_lock<std::mutex> lk(state->mutex);
            if (!state->cv.wait_for(lk, std::chrono::milliseconds(kStartTimeoutMs),
                                    [&] { return state->done; })) {
                timed_out = true;
            }
        }
        if (timed_out) {
            // The thread is stuck inside a blocking Xlib call: detach it, keep
            // the object alive forever and move on to the next backend.
            starter.detach();
            parked_.push_back(std::move(candidate));
            if (reason->empty()) *reason = "timed out after " + std::to_string(kStartTimeoutMs) + " ms";
            return false;
        }
        starter.join();
        if (!state->ok) {
            if (reason->empty()) *reason = state->err;
            return false;
        }
        inner_ = std::move(candidate);
        return true;
    }

    static constexpr int kStartTimeoutMs = 3000;
    Chord                 chord_;
    std::function<void()> on_press_;
    std::function<void()> on_release_;
    std::unique_ptr<HotkeyListener> inner_;
    // backends that never returned from start() - leaked on purpose
    std::vector<std::unique_ptr<HotkeyListener>> parked_;
};

}  // namespace

std::unique_ptr<HotkeyListener> create_hotkey_listener(const Chord & chord,
                                                       std::function<void()> on_press,
                                                       std::function<void()> on_release) {
    return std::make_unique<AutoHotkeyListener>(chord, std::move(on_press), std::move(on_release));
}

// ---------------------------------------------------------------- clipboard

namespace {

// ---------------------------------------------------- Wayland helper tools

#  if defined(ASR_HAVE_X11)

// Clipboard served by a detached owner thread with its own X11 connection.
struct ClipboardState {
    std::mutex         mutex;
    std::string        text;     // latest UTF-8 text set by the application
    unsigned long long gen = 0;  // bumped on every set; the owner watches this
    bool               started = false;
};

ClipboardState & clipboard_state() {
    static ClipboardState state;
    return state;
}

std::string clipboard_snapshot(unsigned long long * gen) {
    ClipboardState & state = clipboard_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    *gen = state.gen;
    return state.text;
}

void serve_selection_request(Display * dpy, const XSelectionRequestEvent & req, Atom utf8,
                             Atom targets, Atom text_atom, const std::string & text) {
    XSelectionEvent reply{};
    reply.type = SelectionNotify;
    reply.display = req.display;
    reply.requestor = req.requestor;
    reply.selection = req.selection;
    reply.target = req.target;
    reply.property = None;
    reply.time = req.time;

    if (req.property != None) {
        if (req.target == targets) {
            Atom offered[3] = {utf8, XA_STRING, text_atom};
            XChangeProperty(dpy, req.requestor, req.property, XA_ATOM, 32, PropModeReplace,
                            reinterpret_cast<const unsigned char *>(offered), 3);
            reply.property = req.property;
        } else if (req.target == utf8 || req.target == XA_STRING || req.target == text_atom) {
            const Atom type = (req.target == utf8) ? utf8 : XA_STRING;
            // No INCR protocol: clipboard payloads here are short transcripts.
            XChangeProperty(dpy, req.requestor, req.property, type, 8, PropModeReplace,
                            reinterpret_cast<const unsigned char *>(text.data()),
                            static_cast<int>(text.size()));
            reply.property = req.property;
        }
    }
    XSendEvent(dpy, req.requestor, False, 0, reinterpret_cast<XEvent *>(&reply));
    XFlush(dpy);
}

// Never returns: the process exit tears it down.  The thread owns this Display
// exclusively, which keeps Xlib usage single threaded per connection.
void clipboard_owner_thread(const std::shared_ptr<std::promise<bool>> & ready) {
    x11_global_init();
    Display * dpy = XOpenDisplay(nullptr);
    if (dpy == nullptr) {
        ready->set_value(false);
        return;
    }
    const Window root = DefaultRootWindow(dpy);
    const Window win = XCreateSimpleWindow(dpy, root, -10, -10, 1, 1, 0, 0, 0);
    if (win == None) {
        XCloseDisplay(dpy);
        ready->set_value(false);
        return;
    }
    const Atom clip = XInternAtom(dpy, "CLIPBOARD", False);
    const Atom utf8 = XInternAtom(dpy, "UTF8_STRING", False);
    const Atom targets = XInternAtom(dpy, "TARGETS", False);
    const Atom text_atom = XInternAtom(dpy, "TEXT", False);

    unsigned long long served = 0;
    std::string current = clipboard_snapshot(&served);
    XSetSelectionOwner(dpy, clip, win, CurrentTime);
    XFlush(dpy);
    ready->set_value(true);

    for (;;) {
        while (XPending(dpy) > 0) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == SelectionRequest) {
                serve_selection_request(dpy, ev.xselectionrequest, utf8, targets, text_atom,
                                        current);
            }
            // SelectionClear means another client took over; ownership is retaken
            // below as soon as the application sets new text.
        }
        unsigned long long gen = 0;
        std::string latest = clipboard_snapshot(&gen);
        if (gen != served) {
            served = gen;
            current = std::move(latest);
            XSetSelectionOwner(dpy, clip, win, CurrentTime);
            XFlush(dpy);
        }
        struct pollfd pfd;
        pfd.fd = ConnectionNumber(dpy);
        pfd.events = POLLIN;
        pfd.revents = 0;
        poll(&pfd, 1, 50);  // short wait so a new generation is picked up promptly
    }
}

struct SelectionData {
    Atom                type = None;
    int                 format = 0;
    unsigned long       nitems = 0;
    std::string         bytes;  // raw property bytes as Xlib returns them
};

bool read_property(Display * dpy, Window win, Atom prop, SelectionData * out) {
    Atom type = None;
    int format = 0;
    unsigned long nitems = 0;
    unsigned long bytes_after = 0;
    unsigned char * data = nullptr;
    // 4 MiB in 32-bit units is vastly more than any transcript needs.
    const int status = XGetWindowProperty(dpy, win, prop, 0, 1 << 20, True, AnyPropertyType,
                                          &type, &format, &nitems, &bytes_after, &data);
    if (status != Success) return false;
    bool ok = false;
    if (data != nullptr && type != None) {
        size_t item_size = 1;
        if (format == 32) item_size = sizeof(long);  // Xlib widens 32 bit data to long
        else if (format == 16) item_size = sizeof(short);
        out->bytes.assign(reinterpret_cast<const char *>(data), nitems * item_size);
        ok = true;
    }
    if (data != nullptr) XFree(data);
    out->type = type;
    out->format = format;
    out->nitems = nitems;
    return ok;
}

// Asks the owner for `target` and waits (bounded) for the SelectionNotify.
bool convert_selection(Display * dpy, Window win, Atom selection, Atom target, Atom prop,
                       SelectionData * out, int timeout_ms) {
    XDeleteProperty(dpy, win, prop);
    XConvertSelection(dpy, selection, target, prop, win, CurrentTime);
    XFlush(dpy);

    const long long deadline = now_ms() + timeout_ms;
    while (now_ms() < deadline) {
        while (XPending(dpy) > 0) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type != SelectionNotify) continue;
            if (ev.xselection.requestor != win) continue;
            if (ev.xselection.property == None) return false;  // target refused
            return read_property(dpy, win, ev.xselection.property, out);
        }
        struct pollfd pfd;
        pfd.fd = ConnectionNumber(dpy);
        pfd.events = POLLIN;
        pfd.revents = 0;
        const long long left = deadline - now_ms();
        if (poll(&pfd, 1, left > 0 ? static_cast<int>(left) : 0) <= 0) break;
    }
    return false;
}

bool target_offered(const SelectionData & data, Atom want) {
    if (data.format != 32) return false;
    for (unsigned long i = 0; i < data.nitems; ++i) {
        if ((i + 1) * sizeof(long) > data.bytes.size()) break;
        long value = 0;
        std::memcpy(&value, data.bytes.data() + i * sizeof(long), sizeof(long));
        if (static_cast<Atom>(value) == want) return true;
    }
    return false;
}

bool x11_clipboard_get(std::string * out) {
    x11_global_init();
    Display * dpy = XOpenDisplay(nullptr);
    if (dpy == nullptr) return false;
    const Window win = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), -10, -10, 1, 1, 0, 0, 0);
    bool ok = false;
    if (win != None) {
        const Atom clip = XInternAtom(dpy, "CLIPBOARD", False);
        const Atom prop = XInternAtom(dpy, "ASR_CLIPBOARD_DATA", False);
        const Atom utf8 = XInternAtom(dpy, "UTF8_STRING", False);
        const Atom targets = XInternAtom(dpy, "TARGETS", False);

        // ICCCM TARGETS negotiation: ask what the owner offers, then request the
        // best flavour instead of guessing.
        Atom want = utf8;
        SelectionData data;
        if (convert_selection(dpy, win, clip, targets, prop, &data, 300) && data.format == 32) {
            if (target_offered(data, utf8)) want = utf8;
            else if (target_offered(data, XA_STRING)) want = XA_STRING;
        }
        auto fetch = [&](Atom target, std::string * text) {
            SelectionData got;
            if (!convert_selection(dpy, win, clip, target, prop, &got, 800)) return false;
            if (got.format != 8 || got.type == None) return false;
            text->assign(got.bytes);
            return true;
        };
        // Retry once with the legacy target when the owner refused UTF8_STRING.
        ok = fetch(want, out) || (want != XA_STRING && fetch(XA_STRING, out));
        XDestroyWindow(dpy, win);
    }
    XCloseDisplay(dpy);
    return ok;
}

#  endif  // ASR_HAVE_X11

}  // namespace

bool clipboard_set_text(const std::string & utf8) {
#  if defined(ASR_HAVE_X11)
    const char * display = std::getenv("DISPLAY");
    if (display != nullptr && *display != '\0') {
        ClipboardState & state = clipboard_state();
        bool start_owner = false;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.text = utf8;
            ++state.gen;
            if (state.started) return true;  // owner thread re-reads on the new gen
            state.started = true;
            start_owner = true;
        }
        if (start_owner) {
            auto ready = std::make_shared<std::promise<bool>>();
            std::future<bool> done = ready->get_future();
            // Detached on purpose: clipboard_set_text must not block on serving
            // the selection.
            std::thread([ready] { clipboard_owner_thread(ready); }).detach();
            if (done.wait_for(std::chrono::seconds(3)) == std::future_status::ready &&
                done.get()) {
                return true;
            }
            std::lock_guard<std::mutex> lock(state.mutex);
            state.started = false;  // DISPLAY existed but was unusable: retry later
        }
    }
#  endif
    // No usable X11 (pure Wayland): best effort through wl-copy.
    return wl_copy(utf8);
}

bool clipboard_get_text(std::string * out) {
    if (out == nullptr) return false;
#  if defined(ASR_HAVE_X11)
    if (x11_clipboard_get(out)) return true;
#  endif
    return wl_paste(out);
}

#  if defined(ASR_HAVE_X11) && !defined(ASR_HAVE_XTEST)
// When the XTest headers are missing, resolve the single entry point at runtime
// instead of adding a hard link dependency.  Never called through this
// declaration directly (only through the pointer below).
extern "C" int XTestFakeKeyEvent(void * display, unsigned int keycode, int is_press,
                                 unsigned long delay);
using XTestFakeKeyEventFn = decltype(&XTestFakeKeyEvent);

namespace {

XTestFakeKeyEventFn load_xtst() {
    static std::once_flag once;
    static XTestFakeKeyEventFn fn = nullptr;
    std::call_once(once, [] {
        void * lib = dlopen("libXtst.so.6", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) lib = dlopen("libXtst.so", RTLD_NOW | RTLD_LOCAL);
        if (lib != nullptr) {
            fn = reinterpret_cast<XTestFakeKeyEventFn>(dlsym(lib, "XTestFakeKeyEvent"));
        }
        // The library is intentionally never dlclosed: Xlib keeps extension
        // pointers that would dangle afterwards.
    });
    return fn;
}

}  // namespace
#  endif

#  if defined(ASR_HAVE_X11)
namespace {

// XKeysymToKeycode() returns 0 while the server has not pushed a real keyboard
// layout yet (observed on WSLg's Xwayland before any window took focus).
// 37 and 55 are the fixed evdev X keycodes of LeftCtrl and V.
KeyCode paste_keycode(Display * dpy, KeySym ks, KeyCode evdev_fallback) {
    const KeyCode code = XKeysymToKeycode(dpy, ks);
    return code != 0 ? code : evdev_fallback;
}

}  // namespace
#  endif

bool paste_into_focused_window() {
#  if defined(ASR_HAVE_XTEST)
    x11_global_init();
    Display * dpy = XOpenDisplay(nullptr);
    if (dpy == nullptr) return false;
    const KeyCode ctrl = paste_keycode(dpy, XK_Control_L, 37);
    const KeyCode v = paste_keycode(dpy, XK_v, 55);
    bool ok = ctrl != 0 && v != 0;
    if (ok) {
        // Keep the listeners from treating our own Ctrl+V as a hotkey chord.
        g_inject_suppress_until.store(now_ms() + 250);
        ok = XTestFakeKeyEvent(dpy, ctrl, True, 0) != 0;
        XFlush(dpy);
        usleep(20000);
        ok = ok && XTestFakeKeyEvent(dpy, v, True, 0) != 0;
        XFlush(dpy);
        usleep(20000);
        XTestFakeKeyEvent(dpy, v, False, 0);
        XTestFakeKeyEvent(dpy, ctrl, False, 0);
        XFlush(dpy);
        g_inject_suppress_until.store(now_ms() + 250);
    }
    XCloseDisplay(dpy);
    return ok;
#  elif defined(ASR_HAVE_X11)
    XTestFakeKeyEventFn fake = load_xtst();
    if (fake == nullptr) return false;  // no XTest: the app asks for a manual paste

    x11_global_init();
    Display * dpy = XOpenDisplay(nullptr);
    if (dpy == nullptr) return false;
    const KeyCode ctrl = paste_keycode(dpy, XK_Control_L, 37);
    const KeyCode v = paste_keycode(dpy, XK_v, 55);
    bool ok = ctrl != 0 && v != 0;
    if (ok) {
        g_inject_suppress_until.store(now_ms() + 250);
        ok = fake(dpy, ctrl, True, 0) != 0;
        XFlush(dpy);
        usleep(20000);
        ok = ok && fake(dpy, v, True, 0) != 0;
        XFlush(dpy);
        usleep(20000);
        fake(dpy, v, False, 0);
        fake(dpy, ctrl, False, 0);
        XFlush(dpy);
        g_inject_suppress_until.store(now_ms() + 250);
    }
    XCloseDisplay(dpy);
    return ok;
#  else
    return false;  // built without X11: inject_text() cannot work
#  endif
}

// -------------------------------------------------------------------- misc

std::string user_data_dir() {
    std::string base;
    const char * xdg = std::getenv("XDG_DATA_HOME");
    if (xdg != nullptr && *xdg != '\0') {
        base = xdg;
    } else {
        const std::string home = home_dir();
        if (home.empty()) return std::string();
        base = home + "/.local/share";
    }
    const std::filesystem::path dir = std::filesystem::path(base) / "npa_asr";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir.string();
}

namespace {

bool on_path(const char * name) {
    const char * path = std::getenv("PATH");
    if (path == nullptr) return false;
    const std::string all(path);
    size_t start = 0;
    for (;;) {
        const size_t colon = all.find(':', start);
        const std::string dir =
            all.substr(start, colon == std::string::npos ? std::string::npos : colon - start);
        if (!dir.empty() && access((dir + "/" + name).c_str(), X_OK) == 0) return true;
        if (colon == std::string::npos) break;
        start = colon + 1;
    }
    return false;
}

// Double fork so the opener is reparented to init; the pipe reports exec()
// failures (EOF on success, thanks to FD_CLOEXEC).
bool spawn_detached(const char * tool, const std::string & arg) {
    int fds[2];
    if (pipe(fds) != 0) return false;
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    if (pid == 0) {
        close(fds[0]);
        fcntl(fds[1], F_SETFD, FD_CLOEXEC);
        setsid();
        if (fork() != 0) _exit(0);
        const int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            if (devnull > STDERR_FILENO) close(devnull);
        }
        execlp(tool, tool, arg.c_str(), static_cast<char *>(nullptr));
        const int exec_errno = errno;
        const ssize_t ignored = write(fds[1], &exec_errno, sizeof(exec_errno));
        (void)ignored;
        _exit(127);
    }
    close(fds[1]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    int exec_errno = 0;
    ssize_t n = 0;
    do {
        n = read(fds[0], &exec_errno, sizeof(exec_errno));
    } while (n < 0 && errno == EINTR);
    close(fds[0]);
    return n == 0;
}

}  // namespace

bool open_in_browser(const std::string & url) {
    if (url.empty()) return false;
    // Under WSL the Windows-side handler is preferable when it is installed.
    if (on_path("wslview")) return spawn_detached("wslview", url);
    return spawn_detached("xdg-open", url);
}

std::string cpu_brand() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, 10, "model name") != 0) continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const size_t begin = line.find_first_not_of(" \t", colon + 1);
        if (begin == std::string::npos) continue;
        std::string brand = line.substr(begin);
        while (!brand.empty() && (brand.back() == '\r' || brand.back() == ' ')) brand.pop_back();
        return brand;
    }
    return std::string();
}

std::string os_name() {
    const std::string release = read_first_line("/proc/sys/kernel/osrelease");
    if (release.empty()) return "Linux";
    // WSL1 and WSL2 both mark the kernel version as Microsoft's.
    const std::string version = ascii_lower(read_first_line("/proc/version"));
    const bool wsl = version.find("microsoft") != std::string::npos;
    return (wsl ? std::string("WSL2 Linux ") : std::string("Linux ")) + release;
}

}  // namespace asr::platform

#endif  // __linux__
