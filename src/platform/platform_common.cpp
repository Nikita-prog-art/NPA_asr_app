// OS-independent part of the platform layer (see platform.h): chord parsing /
// formatting, inject_text, sleep_ms and the path helpers that need nothing but
// the C standard library (plus one Win32 call for the executable path).
//
// This translation unit is compiled on every platform, so every OS-specific
// include stays behind the matching #if guard.
#include "platform/platform.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX  // otherwise windows.h defines min/max as macros
#  endif
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach-o/dyld.h>
#  include <unistd.h>
#else
#  include <unistd.h>
#endif

namespace asr::platform {
namespace {

std::string ascii_lower(const std::string & s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return r;
}

std::string trim_ascii(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) --e;
    return s.substr(b, e - b);
}

// Canonicalises one non-modifier token ("space" -> "Space", "f12" -> "F12").
// `tok` must already be lower-cased and trimmed.
bool parse_key_token(const std::string & tok, std::string * canonical) {
    if (tok.size() == 1) {
        const char c = tok[0];
        if (c >= 'a' && c <= 'z') {
            *canonical = std::string(1, static_cast<char>(c - 'a' + 'A'));
            return true;
        }
        if (c >= '0' && c <= '9') {
            *canonical = tok;
            return true;
        }
        return false;
    }
    if (tok.size() >= 2 && tok[0] == 'f') {
        bool digits = true;
        for (size_t i = 1; i < tok.size(); ++i) {
            if (std::isdigit(static_cast<unsigned char>(tok[i])) == 0) digits = false;
        }
        if (digits) {
            const int n = std::atoi(tok.c_str() + 1);
            if (n >= 1 && n <= 24) {
                *canonical = "F" + std::to_string(n);
                return true;
            }
            return false;
        }
    }
    static const struct { const char * name; const char * canonical; } kNamed[] = {
        {"space", "Space"},           {"scrolllock", "ScrollLock"}, {"pause", "Pause"},
        {"insert", "Insert"},         {"home", "Home"},             {"end", "End"},
        {"pageup", "PageUp"},         {"pagedown", "PageDown"},     {"rightctrl", "RightCtrl"},
        {"rightalt", "RightAlt"},     {"rightshift", "RightShift"},
    };
    for (const auto & n : kNamed) {
        if (tok == n.name) {
            *canonical = n.canonical;
            return true;
        }
    }
    return false;
}

#if defined(_WIN32)
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
#endif

}  // namespace

// ------------------------------------------------------------------- chord

std::string Chord::to_string() const {
    std::string s;
    if (ctrl) s += "Ctrl+";
    if (alt) s += "Alt+";
    if (shift) s += "Shift+";
    if (win) s += "Win+";
    s += key;
    return s;
}

bool Chord::parse(const std::string & text, Chord * out, std::string * err) {
    if (out == nullptr) {
        if (err != nullptr) *err = "no output chord given";
        return false;
    }

    Chord chord;
    std::string key;
    bool have_key = false;

    size_t start = 0;
    for (;;) {
        const size_t plus = text.find('+', start);
        const std::string tok =
            trim_ascii(ascii_lower(text.substr(start, plus == std::string::npos
                                                       ? std::string::npos
                                                       : plus - start)));
        if (tok.empty()) {
            if (err != nullptr) *err = "empty token in chord '" + text + "'";
            return false;
        }
        if (tok == "ctrl" || tok == "control") {
            chord.ctrl = true;
        } else if (tok == "alt") {
            chord.alt = true;
        } else if (tok == "shift") {
            chord.shift = true;
        } else if (tok == "win" || tok == "super" || tok == "meta" || tok == "cmd") {
            chord.win = true;
        } else {
            std::string canonical;
            if (!parse_key_token(tok, &canonical)) {
                if (err != nullptr) *err = "unknown key '" + tok + "'";
                return false;
            }
            if (have_key) {
                if (err != nullptr) *err = "more than one main key in chord '" + text + "'";
                return false;
            }
            key = canonical;
            have_key = true;
        }
        if (plus == std::string::npos) break;
        start = plus + 1;
    }

    if (!have_key) {
        if (err != nullptr) *err = "chord '" + text + "' has no main key";
        return false;
    }
    chord.key = key;
    *out = chord;
    return true;
}

// -------------------------------------------------------------------- text

bool inject_text(const std::string & utf8) {
    return clipboard_set_text(utf8) && paste_into_focused_window();
}

// -------------------------------------------------------------------- misc

void sleep_ms(int ms) {
    if (ms <= 0) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

std::string home_dir() {
#if defined(_WIN32)
    const char * profile = std::getenv("USERPROFILE");
    if (profile != nullptr && *profile != '\0') return profile;
    const char * drive = std::getenv("HOMEDRIVE");
    const char * rest = std::getenv("HOMEPATH");
    if (drive != nullptr && *drive != '\0' && rest != nullptr && *rest != '\0') {
        return std::string(drive) + rest;
    }
    return std::string();
#else
    const char * home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') return home;
    return std::string();
#endif
}

std::string exe_path() {
#if defined(_WIN32)
    // GetModuleFileNameW truncates instead of returning the needed size; retry
    // with a bigger buffer while the long-path support is uncertain.
    std::vector<wchar_t> buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(),
                                           static_cast<DWORD>(buf.size()));
        if (n == 0) return std::string();
        if (n < buf.size()) return utf16_to_utf8(std::wstring(buf.data(), n));
        buf.resize(buf.size() * 2);
    }
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);  // fails on purpose, sets the needed size
    std::vector<char> buf(static_cast<size_t>(size) + 1, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return std::string();
    return std::string(buf.data());
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return std::string();
    return std::string(buf, static_cast<size_t>(n));
#endif
}

std::string exe_dir() {
    const std::string p = exe_path();
    if (p.empty()) return std::string();
    const size_t slash = p.find_last_of("\\/");
    return slash == std::string::npos ? std::string() : p.substr(0, slash);
}

// -------------------------------------------------------------------- tray

#if !defined(_WIN32)
// Only Windows has a real notification-area icon for now; on X11 the desktop
// panels are not a standard protocol and an icon that never appears would be
// worse than none, so the application keeps "close" meaning "quit" there.
// (platform_win.cpp provides the Windows implementation.)
namespace {

class UnavailableTrayIcon final : public TrayIcon {
public:
    bool add(const std::string &, std::string * err) override {
        if (err != nullptr) *err = "системный трей не поддерживается на этой платформе";
        return false;
    }
    void remove() override {}
    TrayAction take_action() override { return TrayAction::None; }
    const char * backend_name() const override { return "none"; }
};

}  // namespace

std::unique_ptr<TrayIcon> create_tray_icon() { return std::make_unique<UnavailableTrayIcon>(); }
#endif

// -------------------------------------------------------------------- icon

// The application icon: a rounded square in the accent colour of the GUI with
// three white bars (a level meter) on it.  Drawn here instead of shipping an
// .ico/.png so the repository stays free of binary assets and every size is
// rendered at its native resolution (16 px in the notification area, 32 px on
// the taskbar, any size on HiDPI).
bool app_icon_rgba(int size, std::vector<unsigned char> * out) {
    if (out == nullptr || size < 8 || size > 1024) return false;
    const float s = static_cast<float>(size);
    const float radius = std::max(1.0f, s * 0.22f);
    // centre x and height, in fractions of the icon
    const float bars[3][2] = {{0.30f, 0.34f}, {0.50f, 0.64f}, {0.70f, 0.44f}};
    const float bar_w = std::max(1.0f, s * 0.125f);
    const float accent[3] = {66.0f, 150.0f, 250.0f};

    out->assign(static_cast<size_t>(size) * static_cast<size_t>(size) * 4, 0);
    unsigned char * px = out->data();
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const float cx = static_cast<float>(x) + 0.5f;
            const float cy = static_cast<float>(y) + 0.5f;
            // signed distance to the rounded square (negative inside)
            const float qx = std::fabs(cx - s * 0.5f) - (s * 0.5f - radius);
            const float qy = std::fabs(cy - s * 0.5f) - (s * 0.5f - radius);
            const float ax = std::max(qx, 0.0f);
            const float ay = std::max(qy, 0.0f);
            const float d = std::sqrt(ax * ax + ay * ay) + std::min(std::max(qx, qy), 0.0f) - radius;
            const float bg = std::clamp(0.5f - d, 0.0f, 1.0f);  // 1 px antialiased edge
            float glyph = 0.0f;
            for (const auto & b : bars) {
                const float bx = std::fabs(cx - s * b[0]) - bar_w * 0.5f;
                const float by = std::fabs(cy - s * 0.5f) - s * b[1] * 0.5f;
                glyph = std::max(glyph, std::clamp(0.5f - std::max(bx, by), 0.0f, 1.0f));
            }
            unsigned char * p = px + (static_cast<size_t>(y) * static_cast<size_t>(size) +
                                      static_cast<size_t>(x)) * 4;
            for (int c = 0; c < 3; ++c) {
                const float value = accent[c] * (1.0f - glyph) + 255.0f * glyph;
                p[c] = static_cast<unsigned char>(std::lround(value));
            }
            p[3] = static_cast<unsigned char>(std::lround(255.0f * bg));
        }
    }
    return true;
}

}  // namespace asr::platform
