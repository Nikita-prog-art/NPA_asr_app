// Windows platform layer: global push-to-talk hotkey (low-level keyboard hook
// on a dedicated message-pumping thread), clipboard and text injection.
//
// The whole file is empty on non-Windows platforms.
#if defined(_WIN32)

#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <shellapi.h>  // ShellExecuteA (WIN32_LEAN_AND_MEAN hides it)

#  include "platform/platform.h"

#  include <atomic>
#  include <chrono>
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#  include <filesystem>
#  include <future>
#  include <memory>
#  include <mutex>
#  include <string>
#  include <thread>
#  include <utility>
#  include <vector>

#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif

namespace asr::platform {
namespace {

// ------------------------------------------------------------- conversions

std::string utf16_to_utf8(const std::wstring & w) {
    if (w.empty()) return std::string();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                                      nullptr, 0, nullptr, nullptr);
    if (n <= 0) return std::string();
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                        out.data(), n, nullptr, nullptr);
    return out;
}

std::wstring utf8_to_utf16(const std::string & s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                                      nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

// ------------------------------------------------------------- virtual keys

// Canonical key name (see platform.h) -> VK_* code, 0 when unsupported.
int vk_for_key(const std::string & key) {
    if (key.size() == 1) {
        const char c = key[0];
        if (c >= 'A' && c <= 'Z') return c;  // VK codes for letters are ASCII
        if (c >= '0' && c <= '9') return c;
    }
    if (key.size() >= 2 && key[0] == 'F') {
        const int n = std::atoi(key.c_str() + 1);
        if (n >= 1 && n <= 24) return VK_F1 + (n - 1);
    }
    if (key == "Space") return VK_SPACE;
    if (key == "ScrollLock") return VK_SCROLL;
    if (key == "Pause") return VK_PAUSE;
    if (key == "Insert") return VK_INSERT;
    if (key == "Home") return VK_HOME;
    if (key == "End") return VK_END;
    if (key == "PageUp") return VK_PRIOR;
    if (key == "PageDown") return VK_NEXT;
    // Low-level hooks report the side-specific VK codes for these keys.
    if (key == "RightCtrl") return VK_RCONTROL;
    if (key == "RightAlt") return VK_RMENU;
    if (key == "RightShift") return VK_RSHIFT;
    return 0;
}

bool modifier_down(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

bool chord_modifiers_down(const Chord & c) {
    if (c.ctrl && !modifier_down(VK_CONTROL)) return false;
    if (c.alt && !modifier_down(VK_MENU)) return false;
    if (c.shift && !modifier_down(VK_SHIFT)) return false;
    if (c.win && !(modifier_down(VK_LWIN) || modifier_down(VK_RWIN))) return false;
    return true;
}

}  // namespace

// A low-level keyboard hook must be installed and pumped by the same thread, so
// the listener owns a dedicated thread with its own GetMessage loop.  The hook
// callback runs on that thread and therefore needs a way back to the listener.
class WinHotkeyListener final : public HotkeyListener {
public:
    WinHotkeyListener(const Chord & chord, std::function<void()> on_press,
                      std::function<void()> on_release)
        : chord_(chord), on_press_(std::move(on_press)), on_release_(std::move(on_release)) {}

    ~WinHotkeyListener() override { stop(); }

    bool start(std::string * err) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (thread_.joinable()) return true;  // already running
        main_vk_ = vk_for_key(chord_.key);
        if (main_vk_ == 0) {
            if (err != nullptr) *err = "unsupported hotkey key: " + chord_.key;
            return false;
        }
        auto ready = std::make_shared<std::promise<bool>>();
        std::future<bool> done = ready->get_future();
        start_err_.clear();
        thread_ = std::thread([this, ready] { thread_main(ready); });
        if (done.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            if (err != nullptr) *err = "keyboard hook thread did not become ready";
            return false;
        }
        const bool ok = done.get();
        if (!ok) {
            if (thread_.joinable()) thread_.join();
            if (err != nullptr) *err = start_err_.empty() ? "keyboard hook install failed" : start_err_;
        }
        return ok;
    }

    void stop() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!thread_.joinable()) return;  // never started, or already stopped
        if (thread_.get_id() == std::this_thread::get_id()) {
            // Called from inside a callback: ask the loop to quit without self-joining.
            PostQuitMessage(0);
            return;
        }
        const DWORD tid = thread_id_.load();
        if (tid != 0) {
            // The queue exists before start() returns, so a single post is enough;
            // retry briefly in case the thread is still winding up.
            for (int i = 0; i < 50; ++i) {
                if (PostThreadMessageW(tid, WM_QUIT, 0, 0) != 0) break;
                Sleep(10);
            }
        }
        thread_.join();
        thread_id_.store(0);
    }

    const char * backend_name() const override { return "win-hook"; }

private:
    static LRESULT CALLBACK hook_proc(int nCode, WPARAM wParam, LPARAM lParam) {
        if (nCode == HC_ACTION && t_current_ != nullptr) {
            t_current_->handle_event(wParam, reinterpret_cast<const KBDLLHOOKSTRUCT *>(lParam));
        }
        return CallNextHookEx(nullptr, nCode, wParam, lParam);
    }

    void handle_event(WPARAM wParam, const KBDLLHOOKSTRUCT * kb) {
        // Ignore everything this process (or another application) injected, e.g.
        // the Ctrl+V that paste_into_focused_window() sends.
        if ((kb->flags & LLKHF_INJECTED) != 0) return;
        const DWORD vk = kb->vkCode;
        const bool down = (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN);
        const bool up = (wParam == WM_KEYUP || wParam == WM_SYSKEYUP);
        if (!down && !up) return;

        if (down && vk == static_cast<DWORD>(main_vk_)) {
            if (!main_key_down_) {
                main_key_down_ = true;
                if (chord_modifiers_down(chord_)) {
                    pressed_ = true;
                    if (on_press_) on_press_();
                }
            }
        } else if (up && vk == static_cast<DWORD>(main_vk_)) {
            main_key_down_ = false;
            if (pressed_) {
                pressed_ = false;
                if (on_release_) on_release_();
            }
        }

        // Releasing a modifier before the main key ends the chord as well; this is
        // re-checked on every event so the release is never missed.
        if (pressed_ && !chord_modifiers_down(chord_)) {
            pressed_ = false;
            if (on_release_) on_release_();
        }
    }

    void thread_main(const std::shared_ptr<std::promise<bool>> & ready) {
        // GetMessageW creates the thread's message queue; create it explicitly so
        // thread_id_ is usable by the first stop() call, then publish it.
        MSG msg;
        PeekMessageW(&msg, nullptr, WM_USER, WM_USER, PM_NOREMOVE);
        thread_id_.store(GetCurrentThreadId());
        t_current_ = this;

        const HMODULE mod = GetModuleHandleW(nullptr);
        hook_ = SetWindowsHookExW(WH_KEYBOARD_LL, &WinHotkeyListener::hook_proc, mod, 0);
        if (hook_ == nullptr) {
            start_err_ = "SetWindowsHookExW(WH_KEYBOARD_LL) failed (error " +
                         std::to_string(GetLastError()) + ")";
            t_current_ = nullptr;
            ready->set_value(false);
            return;
        }
        ready->set_value(true);

        // Low-level hook callbacks are delivered while this thread pumps messages.
        while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        UnhookWindowsHookEx(hook_);
        hook_ = nullptr;
        t_current_ = nullptr;
    }

    // The hook callback runs on the installing thread, so a thread_local pointer
    // keeps several listeners (or several threads) from clobbering each other.
    static thread_local WinHotkeyListener * t_current_;

    Chord                 chord_;
    std::function<void()> on_press_;
    std::function<void()> on_release_;

    std::mutex        mutex_;
    std::thread       thread_;
    std::atomic<DWORD> thread_id_{0};
    std::string       start_err_;
    HHOOK             hook_ = nullptr;  // hook thread only
    int               main_vk_ = 0;
    bool              main_key_down_ = false;  // hook thread only
    bool              pressed_ = false;        // hook thread only
};

thread_local WinHotkeyListener * WinHotkeyListener::t_current_ = nullptr;

std::unique_ptr<HotkeyListener> create_hotkey_listener(const Chord & chord,
                                                       std::function<void()> on_press,
                                                       std::function<void()> on_release) {
    return std::make_unique<WinHotkeyListener>(chord, std::move(on_press), std::move(on_release));
}

// ---------------------------------------------------------------- clipboard

namespace {

// The clipboard is a shared resource: OpenClipboard fails while another process
// owns it, so retry briefly instead of losing the text.
bool open_clipboard_retry(HWND owner) {
    for (int i = 0; i < 10; ++i) {
        if (OpenClipboard(owner) != 0) return true;
        ::Sleep(10);
    }
    return false;
}

}  // namespace

bool clipboard_set_text(const std::string & utf8) {
    if (!open_clipboard_retry(nullptr)) return false;
    bool ok = false;
    if (EmptyClipboard() != 0) {
        const std::wstring w = utf8_to_utf16(utf8);
        const SIZE_T bytes = (w.size() + 1) * sizeof(wchar_t);
        HGLOBAL mem = GlobalAlloc(GMEM_MOVEABLE, bytes);
        if (mem != nullptr) {
            void * dst = GlobalLock(mem);
            if (dst != nullptr) {
                std::memcpy(dst, w.c_str(), bytes);
                GlobalUnlock(mem);
                // Ownership moves to the clipboard on success, so only free on failure.
                if (SetClipboardData(CF_UNICODETEXT, mem) != nullptr) ok = true;
                else GlobalFree(mem);
            } else {
                GlobalFree(mem);
            }
        }
    }
    CloseClipboard();
    return ok;
}

bool clipboard_get_text(std::string * out) {
    if (out == nullptr) return false;
    if (!open_clipboard_retry(nullptr)) return false;
    bool ok = false;
    HANDLE handle = GetClipboardData(CF_UNICODETEXT);
    if (handle != nullptr) {
        const wchar_t * p = static_cast<const wchar_t *>(GlobalLock(handle));
        if (p != nullptr) {
            *out = utf16_to_utf8(std::wstring(p));
            GlobalUnlock(handle);
            ok = true;
        }
    }
    CloseClipboard();
    return ok;
}

bool paste_into_focused_window() {
    // Scan codes are used so the chord is layout independent enough for Ctrl+V.
    INPUT in[4] = {};
    const UINT ctrl_scan = MapVirtualKeyW(VK_CONTROL, MAPVK_VK_TO_VSC);
    const UINT v_scan = MapVirtualKeyW('V', MAPVK_VK_TO_VSC);
    if (ctrl_scan == 0 || v_scan == 0) return false;
    const UINT scans[4] = {ctrl_scan, v_scan, v_scan, ctrl_scan};
    const bool ups[4] = {false, false, true, true};
    for (int i = 0; i < 4; ++i) {
        in[i].type = INPUT_KEYBOARD;
        in[i].ki.wVk = 0;
        in[i].ki.wScan = static_cast<WORD>(scans[i]);
        in[i].ki.dwFlags = static_cast<DWORD>(KEYEVENTF_SCANCODE) |
                           (ups[i] ? static_cast<DWORD>(KEYEVENTF_KEYUP) : 0u);
        in[i].ki.time = 0;
        in[i].ki.dwExtraInfo = 0;
    }
    // SendInput inserts all events or none; 0 means another process blocked us.
    const UINT sent = SendInput(4, in, sizeof(INPUT));
    return sent != 0;
}

// -------------------------------------------------------------------- misc

std::string user_data_dir() {
    // Use the wide environment API so non-ASCII profile paths stay correct.
    std::wstring base;
    const DWORD need = GetEnvironmentVariableW(L"LOCALAPPDATA", nullptr, 0);
    if (need > 0) {
        std::wstring buf(static_cast<size_t>(need), L'\0');
        const DWORD len = GetEnvironmentVariableW(L"LOCALAPPDATA", buf.data(), need);
        if (len > 0 && len < need) {
            buf.resize(len);
            base = buf;
        }
    }
    std::filesystem::path dir;
    if (!base.empty()) dir = std::filesystem::path(base) / L"npa_asr";
    else {
        const std::string home = home_dir();
        if (home.empty()) return std::string();
        dir = std::filesystem::path(utf8_to_utf16(home)) / L"npa_asr";
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir.u8string();
}

bool open_in_browser(const std::string & url) {
    if (url.empty()) return false;
    const HINSTANCE r = ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    return reinterpret_cast<INT_PTR>(r) > 32;
}

std::string cpu_brand() {
    auto cpuid_leaf = [](unsigned leaf, int out[4]) {
#  if defined(_MSC_VER)
        __cpuid(out, static_cast<int>(leaf));
#  else
        unsigned a = 0, b = 0, c = 0, d = 0;
        __cpuid(static_cast<unsigned>(leaf), a, b, c, d);
        out[0] = static_cast<int>(a);
        out[1] = static_cast<int>(b);
        out[2] = static_cast<int>(c);
        out[3] = static_cast<int>(d);
#  endif
    };
    int regs[4] = {0, 0, 0, 0};
    cpuid_leaf(0x80000000u, regs);
    if (static_cast<unsigned>(regs[0]) < 0x80000004u) return std::string();
    char brand[49] = {0};
    for (unsigned leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf) {
        cpuid_leaf(leaf, regs);
        std::memcpy(brand + static_cast<size_t>(leaf - 0x80000002u) * 16u, regs, 16);
    }
    std::string s(brand);
    const size_t b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return std::string();
    const size_t e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

std::string os_name() {
    // RtlGetVersion reports the real build (GetVersionEx is manifest-lied to).
    struct OsVersionInfoW {
        ULONG dwOSVersionInfoSize;
        ULONG dwMajorVersion;
        ULONG dwMinorVersion;
        ULONG dwBuildNumber;
        ULONG dwPlatformId;
        WCHAR szCSDVersion[128];
    };
    using RtlGetVersionFn = LONG(WINAPI *)(OsVersionInfoW *);
    OsVersionInfoW vi{};
    vi.dwOSVersionInfoSize = sizeof(vi);
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll != nullptr) {
        // Function-pointer cast via void*: keeps -Wcast-function-type quiet.
        auto fn = reinterpret_cast<RtlGetVersionFn>(
            reinterpret_cast<void *>(GetProcAddress(ntdll, "RtlGetVersion")));
        if (fn != nullptr && fn(&vi) == 0) {
            char buf[64];
            if (vi.dwMajorVersion == 10 && vi.dwBuildNumber >= 22000) {
                std::snprintf(buf, sizeof(buf), "Windows 11 (build %lu)",
                              static_cast<unsigned long>(vi.dwBuildNumber));
            } else if (vi.dwMajorVersion == 10) {
                std::snprintf(buf, sizeof(buf), "Windows 10 (build %lu)",
                              static_cast<unsigned long>(vi.dwBuildNumber));
            } else {
                std::snprintf(buf, sizeof(buf), "Windows %lu.%lu (build %lu)",
                              static_cast<unsigned long>(vi.dwMajorVersion),
                              static_cast<unsigned long>(vi.dwMinorVersion),
                              static_cast<unsigned long>(vi.dwBuildNumber));
            }
            return buf;
        }
    }
    return "Windows";
}

}  // namespace asr::platform

#endif  // _WIN32
