// Per-user installation and autostart management (see install.h).
//
// "Installing" is a single-file copy of the running executable into a per-user
// directory plus an optional autostart entry.  The copy is written to a
// temporary name in the target directory and then renamed over the destination,
// so a failure never leaves a half-written executable behind.  Settings and
// models live in platform::user_data_dir() and are never touched here.
#include "app/install.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

#include "json.hpp"
#include "platform/platform.h"
#include "util/sha256.h"

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <combaseapi.h>  // CoTaskMemFree (win32 lean doesn't pull it in)
#  include <shlobj.h>      // SHGetKnownFolderPath / FOLDERID_LocalAppData
#else
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace asr::app {
namespace {

namespace fs = std::filesystem;

#if defined(_WIN32)
const char * const kExeName = "npa_asr.exe";
const wchar_t * const kRunKey = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";
const wchar_t * const kRunValue = L"NPA ASR";
// Human readable location of the autostart entry, used only for display.
const char * const kAutostartTarget = "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\NPA ASR";
#else
const char * const kExeName = "npa_asr";
const char * const kAutostartFileName = "npa-asr.desktop";
#endif

// ---------------------------------------------------------------- win32 bits

#if defined(_WIN32)

std::wstring to_wide(const std::string & s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                                      nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

fs::path local_appdata_dir() {
    PWSTR raw = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_DEFAULT, nullptr, &raw)) &&
        raw != nullptr) {
        const fs::path dir{std::wstring(raw)};
        CoTaskMemFree(raw);
        return dir;
    }
    if (raw != nullptr) CoTaskMemFree(raw);
    // Fallback for restricted environments where the known folder API fails.
    const DWORD need = GetEnvironmentVariableW(L"LOCALAPPDATA", nullptr, 0);
    if (need > 0) {
        std::wstring buf(static_cast<size_t>(need), L'\0');
        const DWORD len = GetEnvironmentVariableW(L"LOCALAPPDATA", buf.data(), need);
        if (len > 0 && len < need) {
            buf.resize(len);
            return fs::path(buf);
        }
    }
    return fs::path();
}

#endif  // _WIN32

// ------------------------------------------------------------------- paths

unsigned long current_pid() {
#if defined(_WIN32)
    return static_cast<unsigned long>(GetCurrentProcessId());
#else
    return static_cast<unsigned long>(::getpid());
#endif
}

// UTF-8 representation for user-facing strings (no-op on POSIX, where
// std::filesystem already stores native narrow bytes).
std::string path_str(const fs::path & p) { return p.u8string(); }

// Builds a filesystem path from a UTF-8 string.  On Windows the native narrow
// encoding is not UTF-8, so the string must be widened first, otherwise
// non-ASCII user names produce an unusable path.
fs::path path_from_utf8(const std::string & s) {
#if defined(_WIN32)
    return fs::path(to_wide(s));
#else
    return fs::path(s);
#endif
}

// Lowercase hex digest of a file, "" on failure.  Reading through
// std::filesystem/std::ifstream takes a native path, so this also works for
// non-ASCII Windows paths (unlike util::sha256_file, which takes a narrow
// std::string).
std::string file_digest(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::string();
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (f.bad()) return std::string();
    return util::sha256_hex(data);
}

fs::path install_dir_path() {
#if defined(_WIN32)
    fs::path base = local_appdata_dir();
    if (base.empty()) {
        const std::string home = platform::home_dir();
        if (home.empty()) return fs::path();
        base = fs::path(to_wide(home)) / L"AppData" / L"Local";
    }
    return base / L"Programs" / L"NPA ASR";
#else
    const std::string home = platform::home_dir();
    if (home.empty()) return fs::path();
    return fs::path(home) / ".local" / "bin";
#endif
}

fs::path install_exe_path() {
    const fs::path dir = install_dir_path();
    if (dir.empty()) return fs::path();
    return dir / kExeName;
}

#if !defined(_WIN32)

// $XDG_CONFIG_HOME or ~/.config
fs::path config_dir_path() {
    const char * xdg = std::getenv("XDG_CONFIG_HOME");
    if (xdg != nullptr && *xdg != '\0') return fs::path(xdg);
    const std::string home = platform::home_dir();
    if (home.empty()) return fs::path();
    return fs::path(home) / ".config";
}

// $XDG_DATA_HOME or ~/.local/share
fs::path data_dir_path() {
    const char * xdg = std::getenv("XDG_DATA_HOME");
    if (xdg != nullptr && *xdg != '\0') return fs::path(xdg);
    const std::string home = platform::home_dir();
    if (home.empty()) return fs::path();
    return fs::path(home) / ".local" / "share";
}

fs::path autostart_desktop_path() {
    const fs::path dir = config_dir_path();
    if (dir.empty()) return fs::path();
    return dir / "autostart" / kAutostartFileName;
}

fs::path menu_desktop_path() {
    const fs::path dir = data_dir_path();
    if (dir.empty()) return fs::path();
    return dir / "applications" / kAutostartFileName;
}

#endif  // !_WIN32

// ------------------------------------------------------------------ helpers

void set_error(std::string * err, const std::string & text) {
    if (err != nullptr) *err = text;
}

bool file_exists(const fs::path & p) {
    if (p.empty()) return false;
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

#if defined(_WIN32)
std::string ascii_lower(std::string s) {
    for (char & c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}
#endif

// True when both paths refer to the same file.  Prefers the inode/file-id
// comparison and falls back to canonical string comparison.
bool same_file(const fs::path & a, const fs::path & b) {
    if (a.empty() || b.empty()) return false;
    std::error_code ec;
    if (fs::exists(a, ec) && fs::exists(b, ec)) {
        ec.clear();
        if (fs::equivalent(a, b, ec) && !ec) return true;
    }
    ec.clear();
    const std::string ca = fs::weakly_canonical(a, ec).u8string();
    ec.clear();
    const std::string cb = fs::weakly_canonical(b, ec).u8string();
    if (ca.empty() || cb.empty()) return false;
#if defined(_WIN32)
    return ascii_lower(ca) == ascii_lower(cb);
#else
    return ca == cb;
#endif
}

bool rename_replace(const fs::path & from, const fs::path & to) {
#if defined(_WIN32)
    return MoveFileExW(from.wstring().c_str(), to.wstring().c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return ::rename(from.c_str(), to.c_str()) == 0;
#endif
}

// Copy `src` over `dst` through a temporary file in dst's directory.  On
// Windows the destination executable cannot be replaced while it is running;
// that case is detected before the copy and reported by the caller.
bool copy_atomic(const fs::path & src, const fs::path & dst, std::string * err) {
    std::error_code ec;
    fs::create_directories(dst.parent_path(), ec);
    if (ec) {
        set_error(err, "не удалось создать каталог " + path_str(dst.parent_path()) + ": " +
                           ec.message());
        return false;
    }
    const fs::path tmp =
        dst.parent_path() / (dst.filename().u8string() + ".tmp." + std::to_string(current_pid()));
    fs::remove(tmp, ec);
    ec.clear();
    fs::copy_file(src, tmp, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        fs::remove(tmp, ec);
        set_error(err, "не удалось скопировать " + path_str(src) + " в " + path_str(tmp) + ": " +
                           ec.message());
        return false;
    }
#if !defined(_WIN32)
    ::chmod(tmp.c_str(), 0755);
#endif
    if (!rename_replace(tmp, dst)) {
#if defined(_WIN32)
        const DWORD code = GetLastError();
        set_error(err, "не удалось заменить " + path_str(dst) +
                           (code == ERROR_ACCESS_DENIED || code == ERROR_SHARING_VIOLATION
                                ? " (файл занят или уже запущен)"
                                : "") +
                           " [код " + std::to_string(static_cast<unsigned long>(code)) + "]");
#else
        set_error(err, "не удалось заменить " + path_str(dst) + ": " + std::strerror(errno));
#endif
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

#if !defined(_WIN32)
bool write_text_file(const fs::path & p, const std::string & text, std::string * err) {
    if (p.empty()) {
        set_error(err, "не удалось определить путь для записи");
        return false;
    }
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    if (ec) {
        set_error(err, "не удалось создать каталог " + path_str(p.parent_path()) + ": " +
                           ec.message());
        return false;
    }
    const fs::path tmp = p.parent_path() / (p.filename().u8string() + ".tmp." +
                                            std::to_string(current_pid()));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            set_error(err, "не удалось записать " + path_str(tmp));
            fs::remove(tmp, ec);
            return false;
        }
        f << text;
        f.flush();
        if (!f) {
            set_error(err, "не удалось записать " + path_str(tmp));
            fs::remove(tmp, ec);
            return false;
        }
    }
    if (!rename_replace(tmp, p)) {
        set_error(err, "не удалось заменить " + path_str(p));
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}
#endif  // !_WIN32

// --- install.json (the "offer once" flag) ---------------------------------

fs::path install_json_path() {
    const std::string data = platform::user_data_dir();
    if (data.empty()) return fs::path();
    return path_from_utf8(data) / "install.json";
}

bool read_install_json(nlohmann::json * out) {
    const fs::path p = install_json_path();
    if (p.empty() || !file_exists(p)) return false;
    try {
        std::ifstream f(p);
        if (!f) return false;
        f >> *out;
        // Anything the parser accepts (null, [], "x", 0) is not a usable state
        // file: report "no flag yet" so the prompt is offered instead of being
        // suppressed forever by a type_error in the callers.
        return out->is_object();
    } catch (const std::exception &) {
        return false;
    }
}

void write_install_json(const nlohmann::json & j) {
    const fs::path p = install_json_path();
    if (p.empty()) return;
    try {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
        std::ofstream f(p);
        if (f) f << j.dump(2) << "\n";
    } catch (const std::exception &) {
        // Best effort: a missing flag only means the prompt may show again.
    }
}

// --------------------------------------------------------- autostart (Linux)

#if !defined(_WIN32)

// freedesktop Exec quoting: the value may contain spaces, and the spec asks for
// the double quote, backtick, dollar sign and backslash to be escaped inside a
// quoted argument.  "%" is the field-code escape character.
std::string exec_quote(const std::string & s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '\\' || c == '"' || c == '$' || c == '`') out.push_back('\\');
        if (c == '%') {
            out += "%%";  // field-code escape
            continue;
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// Defined after the platform blocks; shared by both platforms.
std::string exec_command(bool minimized);

std::string desktop_entry(bool autostart) {
    std::string s;
    s += "[Desktop Entry]\n";
    s += "Type=Application\n";
    s += "Name=NPA ASR\n";
    // Only the autostart entry starts minimized; launching from the menu should
    // show the window.
    s += "Exec=" + exec_command(autostart) + "\n";
    if (autostart) {
        s += "X-GNOME-Autostart-enabled=true\n";
        s += "Terminal=false\n";
    } else {
        s += "Terminal=false\n";
        s += "Categories=AudioVideo;Utility;\n";
    }
    return s;
}

bool is_autostart_enabled() {
    return file_exists(autostart_desktop_path());
}

bool dir_on_path(const std::string & dir) {
    const char * path = std::getenv("PATH");
    if (path == nullptr) return false;
    const std::string home = platform::home_dir();
    std::error_code want_ec;
    const fs::path want = fs::weakly_canonical(path_from_utf8(dir), want_ec);
    if (want_ec || want.empty()) return false;

    const std::string all(path);
    size_t start = 0;
    for (;;) {
        const size_t colon = all.find(':', start);
        std::string e = all.substr(start, colon == std::string::npos ? std::string::npos : colon - start);
        if (!e.empty()) {
            // expand a leading ~ or $HOME the way a shell would
            if (e[0] == '~' && (e.size() == 1 || e[1] == '/')) {
                e = home + e.substr(1);
            } else if (e.rfind("$HOME", 0) == 0) {
                e = home + e.substr(5);
            }
            std::error_code ec;
            const fs::path cand = fs::weakly_canonical(path_from_utf8(e), ec);
            if (!ec && !cand.empty() && cand == want) return true;
        }
        if (colon == std::string::npos) break;
        start = colon + 1;
    }
    return false;
}

#endif  // !_WIN32

// ------------------------------------------------------- autostart (Windows)

#if defined(_WIN32)

bool is_autostart_enabled() {
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, kRunKey, 0, KEY_QUERY_VALUE, &key) != ERROR_SUCCESS) {
        return false;
    }
    DWORD type = 0;
    DWORD size = 0;
    const LSTATUS st = RegQueryValueExW(key, kRunValue, nullptr, &type, nullptr, &size);
    RegCloseKey(key);
    // Only a non-empty string value is a usable autostart command; a leftover of
    // another type must not be reported as "autostart is on".
    const bool string_value = type == REG_SZ || type == REG_EXPAND_SZ;
    return st == ERROR_SUCCESS && string_value && size > sizeof(wchar_t);
}

#endif  // _WIN32

// Command line that launches the installed copy.  The autostart entry starts it
// minimized; the menu entry shows the window.
std::string exec_command(bool minimized) {
    const std::string exe = path_str(install_exe_path());
#if defined(_WIN32)
    std::string cmd = "\"" + exe + "\"";
#else
    std::string cmd = exec_quote(exe);
#endif
    if (minimized) cmd += " --minimized";
    return cmd;
}

}  // namespace

// ---------------------------------------------------------------- public API

std::string autostart_command() {
    return exec_command(true);
}

InstallStatus install_status() {
    InstallStatus st;
    try {
        st.install_dir = path_str(install_dir_path());
        st.install_exe = path_str(install_exe_path());
        st.exe_path = platform::exe_path();
#if defined(_WIN32)
        st.autostart_target = kAutostartTarget;
#else
        st.autostart_target = path_str(autostart_desktop_path());
#endif

        st.installed = file_exists(install_exe_path());
        const fs::path self = path_from_utf8(st.exe_path);
        st.running_installed = st.installed && same_file(self, install_exe_path());
        st.autostart = is_autostart_enabled();

        if (st.installed) {
            std::error_code ec;
            const uintmax_t a = fs::file_size(install_exe_path(), ec);
            ec.clear();
            const uintmax_t b = fs::file_size(self, ec);
            if (!ec && a == b) {
                // An unreadable file yields an empty digest: never treat that as
                // "the same build".
                const std::string da = file_digest(self);
                const std::string db = file_digest(install_exe_path());
                st.up_to_date = !da.empty() && da == db;
            }
        }

        // Human readable notes, most important first.
        if (st.running_installed) {
            st.detail = "приложение запущено из каталога установки";
        } else if (st.installed && !st.up_to_date) {
            st.detail = "установленная копия устарела — выполните --install";
        }
#if !defined(_WIN32)
        if (!st.running_installed && !st.install_dir.empty() && !dir_on_path(st.install_dir)) {
            if (!st.detail.empty()) st.detail += "; ";
            st.detail += st.install_dir + " отсутствует в PATH";
        }
#endif
    } catch (const std::exception & e) {
        st = InstallStatus{};
        st.detail = std::string("не удалось определить состояние: ") + e.what();
    } catch (...) {
        st = InstallStatus{};
        st.detail = "не удалось определить состояние установки";
    }
    return st;
}

bool install_app(bool autostart, bool autostart_only, std::string * err) {
    try {
        if (!autostart_only) {
            const std::string src_s = platform::exe_path();
            if (src_s.empty()) {
                set_error(err, "не удалось определить путь к исполняемому файлу");
                return false;
            }
            const fs::path src = path_from_utf8(src_s);
            const fs::path dir = install_dir_path();
            const fs::path dst = install_exe_path();
            if (dir.empty() || dst.empty()) {
                set_error(err, "не удалось определить каталог установки (пуст HOME/LOCALAPPDATA)");
                return false;
            }
            if (same_file(src, dst)) {
                set_error(err, "приложение уже запущено из каталога установки (" + path_str(dst) +
                                   "); скопировать файл поверх работающего процесса нельзя");
                return false;
            }
            if (!copy_atomic(src, dst, err)) return false;
#if !defined(_WIN32)
            // A menu entry makes the installed copy launchable from the desktop.
            const fs::path menu = menu_desktop_path();
            if (!menu.empty() && !write_text_file(menu, desktop_entry(false), err)) return false;
#endif
        }
        if (autostart) return set_autostart(true, err);
        return true;
    } catch (const std::exception & e) {
        set_error(err, e.what());
        return false;
    } catch (...) {
        set_error(err, "неизвестная ошибка при установке");
        return false;
    }
}

bool uninstall_app(bool keep_files, std::string * err) {
    try {
        bool ok = true;
        std::string first_err;

        // Uninstalling always unregisters the application (autostart entry and,
        // on Linux, the menu entry); keep_files only decides whether the
        // installed executable itself is deleted.
        std::string aerr;
        if (!set_autostart(false, &aerr)) {
            ok = false;
            first_err = aerr;
        }

        std::error_code ec;
#if !defined(_WIN32)
        const fs::path menu = menu_desktop_path();
        if (!menu.empty()) {
            fs::remove(menu, ec);
            if (ec) {
                ok = false;
                if (first_err.empty()) {
                    first_err = "не удалось удалить " + path_str(menu) + ": " + ec.message();
                }
            }
        }
#endif

        if (!keep_files) {
            const fs::path dst = install_exe_path();
            if (!dst.empty()) {
                ec.clear();
                fs::remove(dst, ec);
                if (ec) {
                    ok = false;
                    if (first_err.empty()) {
                        first_err = "не удалось удалить " + path_str(dst) + ": " + ec.message();
#if defined(_WIN32)
                        if (same_file(path_from_utf8(platform::exe_path()), dst)) {
                            first_err += " (приложение запущено из этой копии: закройте его и удалите файл вручную)";
                        }
#endif
                    }
                }
            }
        }

        if (!ok) {
            set_error(err, first_err.empty() ? std::string("не удалось удалить автозапуск") : first_err);
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        set_error(err, e.what());
        return false;
    } catch (...) {
        set_error(err, "неизвестная ошибка при удалении");
        return false;
    }
}

bool set_autostart(bool on, std::string * err) {
    try {
        if (on) {
            // An autostart entry that points at nothing is worse than no entry:
            // it silently does nothing on the next login.
            const fs::path target = install_exe_path();
            if (target.empty() || !file_exists(target)) {
                set_error(err, "сначала установите приложение (npa_asr --install): " +
                                   (target.empty() ? std::string("путь установки не определён")
                                                   : path_str(target) + " не найден"));
                return false;
            }
        }
#if defined(_WIN32)
        if (on) {
            const std::wstring data = to_wide(autostart_command());
            HKEY key = nullptr;
            const LSTATUS open = RegCreateKeyExW(HKEY_CURRENT_USER, kRunKey, 0, nullptr, 0,
                                                 KEY_SET_VALUE, nullptr, &key, nullptr);
            if (open != ERROR_SUCCESS) {
                set_error(err, "не удалось открыть раздел реестра автозапуска [код " +
                                   std::to_string(static_cast<unsigned long>(open)) + "]");
                return false;
            }
            const LSTATUS st =
                RegSetValueExW(key, kRunValue, 0, REG_SZ,
                               reinterpret_cast<const BYTE *>(data.c_str()),
                               static_cast<DWORD>((data.size() + 1) * sizeof(wchar_t)));
            RegCloseKey(key);
            if (st != ERROR_SUCCESS) {
                set_error(err, "не удалось записать автозапуск в реестр [код " +
                                   std::to_string(static_cast<unsigned long>(st)) + "]");
                return false;
            }
            return true;
        }
        HKEY key = nullptr;
        if (RegOpenKeyExW(HKEY_CURRENT_USER, kRunKey, 0, KEY_SET_VALUE, &key) != ERROR_SUCCESS) {
            return true;  // the key does not exist: autostart is already off
        }
        const LSTATUS del = RegDeleteValueW(key, kRunValue);
        RegCloseKey(key);
        if (del != ERROR_SUCCESS && del != ERROR_FILE_NOT_FOUND) {
            set_error(err, "не удалось удалить автозапуск из реестра [код " +
                               std::to_string(static_cast<unsigned long>(del)) + "]");
            return false;
        }
        return true;
#else
        const fs::path p = autostart_desktop_path();
        if (p.empty()) {
            set_error(err, "не удалось определить путь автозапуска");
            return false;
        }
        if (on) return write_text_file(p, desktop_entry(true), err);
        std::error_code ec;
        fs::remove(p, ec);
        if (ec) {
            set_error(err, "не удалось удалить " + path_str(p) + ": " + ec.message());
            return false;
        }
        return true;
#endif
    } catch (const std::exception & e) {
        set_error(err, e.what());
        return false;
    } catch (...) {
        set_error(err, "неизвестная ошибка при настройке автозапуска");
        return false;
    }
}

bool should_offer_install() {
    try {
        if (install_status().running_installed) return false;
        nlohmann::json j;
        if (!read_install_json(&j)) return true;  // no flag file yet: never asked
        return !j.value("prompt_shown", false);
    } catch (...) {
        return false;
    }
}

void mark_install_prompt_shown() {
    try {
        nlohmann::json j;
        read_install_json(&j);  // preserves any other keys already stored
        j["prompt_shown"] = true;
        write_install_json(j);
    } catch (...) {
        // Best effort.
    }
}

}  // namespace asr::app
