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
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
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

std::string exe_dir() {
#if defined(_WIN32)
    // GetModuleFileNameW truncates instead of returning the needed size; retry
    // with a bigger buffer while the long-path support is uncertain.
    std::vector<wchar_t> buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(),
                                           static_cast<DWORD>(buf.size()));
        if (n == 0) return std::string();
        if (n < buf.size()) {
            std::wstring w(buf.data(), n);
            const size_t slash = w.find_last_of(L"\\/");
            if (slash != std::wstring::npos) w.resize(slash);
            return utf16_to_utf8(w);
        }
        buf.resize(buf.size() * 2);
    }
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);  // fails on purpose, sets the needed size
    std::vector<char> buf(static_cast<size_t>(size) + 1, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return std::string();
    std::string p(buf.data());
    const size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? std::string() : p.substr(0, slash);
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return std::string();
    const std::string p(buf, static_cast<size_t>(n));
    const size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? std::string() : p.substr(0, slash);
#endif
}

}  // namespace asr::platform
