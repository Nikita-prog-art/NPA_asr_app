// Windows downloader built on WinHTTP.  WinHTTP is part of the OS, so this
// needs no third party library (winhttp.lib is linked by CMake).
#include "net/net.h"

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winhttp.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "util/sha256.h"

#ifndef ASR_VERSION
#  define ASR_VERSION "0.1.0"
#endif

namespace asr::net {
namespace {

// GitHub rejects some default agents, so always identify ourselves.
const char * kUserAgentUtf8 = "npa_asr/" ASR_VERSION;

template <typename T, size_t N>
constexpr DWORD buf_chars(T (&)[N]) {
    return static_cast<DWORD>(N);
}

std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) return std::wstring();
    const int n = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                                        nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w(static_cast<size_t>(n), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::string wide_to_utf8(const std::wstring & w) {
    if (w.empty()) return std::string();
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                                        nullptr, 0, nullptr, nullptr);
    if (n <= 0) return std::string();
    std::string s(static_cast<size_t>(n), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr,
                          nullptr);
    return s;
}

std::string win_error(DWORD code) {
    if (code == 0) return "unknown error";
    LPWSTR buf = nullptr;
    const DWORD n = ::FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                         FORMAT_MESSAGE_IGNORE_INSERTS,
                                     nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                     reinterpret_cast<LPWSTR>(&buf), 0, nullptr);
    std::string msg;
    if (n != 0 && buf != nullptr) {
        msg = wide_to_utf8(std::wstring(buf, n));
        while (!msg.empty() && (msg.back() == '\r' || msg.back() == '\n' || msg.back() == ' ')) {
            msg.pop_back();
        }
    }
    if (buf != nullptr) ::LocalFree(buf);
    if (msg.empty()) msg = "error " + std::to_string(code);
    return msg;
}

struct WinHttpHandle {
    HINTERNET h = nullptr;

    WinHttpHandle() = default;
    explicit WinHttpHandle(HINTERNET p) : h(p) {}
    ~WinHttpHandle() { reset(); }
    WinHttpHandle(const WinHttpHandle &) = delete;
    WinHttpHandle & operator=(const WinHttpHandle &) = delete;
    WinHttpHandle(WinHttpHandle && o) noexcept : h(o.h) { o.h = nullptr; }
    WinHttpHandle & operator=(WinHttpHandle && o) noexcept {
        if (this != &o) {
            reset();
            h = o.h;
            o.h = nullptr;
        }
        return *this;
    }
    void reset() {
        if (h != nullptr) ::WinHttpCloseHandle(h);
        h = nullptr;
    }
    HINTERNET get() const { return h; }
    explicit operator bool() const { return h != nullptr; }
};

struct FileHandle {
    HANDLE h = INVALID_HANDLE_VALUE;

    FileHandle() = default;
    explicit FileHandle(HANDLE p) : h(p) {}
    ~FileHandle() { reset(); }
    FileHandle(const FileHandle &) = delete;
    FileHandle & operator=(const FileHandle &) = delete;
    FileHandle(FileHandle && o) noexcept : h(o.h) { o.h = INVALID_HANDLE_VALUE; }
    FileHandle & operator=(FileHandle && o) noexcept {
        if (this != &o) {
            reset();
            h = o.h;
            o.h = INVALID_HANDLE_VALUE;
        }
        return *this;
    }
    void reset() {
        if (h != INVALID_HANDLE_VALUE) ::CloseHandle(h);
        h = INVALID_HANDLE_VALUE;
    }
    HANDLE get() const { return h; }
    explicit operator bool() const { return h != INVALID_HANDLE_VALUE; }
};

// Size of an existing file, 0 when it does not exist.
uint64_t file_size_utf8(const std::string & path) {
    WIN32_FILE_ATTRIBUTE_DATA fad {};
    if (!::GetFileAttributesExW(utf8_to_wide(path).c_str(), GetFileExInfoStandard, &fad)) return 0;
    return (static_cast<uint64_t>(fad.nFileSizeHigh) << 32) | fad.nFileSizeLow;
}

bool remove_file(const std::string & path) {
    const std::wstring w = utf8_to_wide(path);
    return ::DeleteFileW(w.c_str()) != 0 || ::GetLastError() == ERROR_FILE_NOT_FOUND;
}

// Verifies the sha256 (when one was published) and only then moves
// dest.part -> dest, so a crash never leaves a half written model in place.
bool finish_download(const std::string & part, const std::string & dest,
                     const std::string & expected_sha256, Progress * progress, std::string * err) {
    if (!expected_sha256.empty()) {
        std::string want = expected_sha256;
        for (char & c : want) {
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        }
        const std::string got = util::sha256_file(part);
        if (got.empty()) {
            *err = "cannot read the downloaded file " + part;
            return false;
        }
        if (got != want) {
            *err = "sha256 mismatch: expected " + want + ", got " + got;
            remove_file(part);  // corrupt payload: never resume from it
            return false;
        }
    }

    // MOVEFILE_REPLACE_EXISTING gives POSIX-like rename semantics on Win32.
    const std::wstring wpart = utf8_to_wide(part);
    const std::wstring wdest = utf8_to_wide(dest);
    if (!::MoveFileExW(wpart.c_str(), wdest.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED)) {
        *err = "cannot move " + part + " to " + dest + ": " + win_error(::GetLastError());
        return false;
    }
    if (progress != nullptr) progress->ok.store(true);
    return true;
}

struct Request {
    WinHttpHandle session;
    WinHttpHandle connect;
    WinHttpHandle request;
};

struct UrlParts {
    std::wstring scheme;
    std::wstring host;
    std::wstring path;  // path + query
    INTERNET_PORT port = 0;
    bool secure = false;
};

bool crack_url(const std::string & url, UrlParts * out, std::string * err) {
    const std::wstring wurl = utf8_to_wide(url);
    if (wurl.empty()) {
        *err = "empty or non-UTF-8 url";
        return false;
    }
    wchar_t scheme[16] = {};
    wchar_t host[256] = {};
    wchar_t path[4096] = {};
    wchar_t extra[4096] = {};

    URL_COMPONENTS uc {};
    uc.dwStructSize = sizeof(uc);
    uc.lpszScheme = scheme;
    uc.dwSchemeLength = buf_chars(scheme);
    uc.lpszHostName = host;
    uc.dwHostNameLength = buf_chars(host);
    uc.lpszUrlPath = path;
    uc.dwUrlPathLength = buf_chars(path);
    uc.lpszExtraInfo = extra;
    uc.dwExtraInfoLength = buf_chars(extra);

    if (!::WinHttpCrackUrl(wurl.c_str(), static_cast<DWORD>(wurl.size()), 0, &uc)) {
        *err = "cannot parse url " + url + ": " + win_error(::GetLastError());
        return false;
    }
    // On success the dw*Length members hold the component lengths (no NUL).
    out->scheme.assign(uc.lpszScheme, uc.dwSchemeLength);
    out->host.assign(uc.lpszHostName, uc.dwHostNameLength);
    out->path.assign(uc.lpszUrlPath, uc.dwUrlPathLength);
    out->path.append(uc.lpszExtraInfo, uc.dwExtraInfoLength);
    if (out->path.empty()) out->path = L"/";
    out->port = uc.nPort;
    out->secure = ::_wcsicmp(out->scheme.c_str(), L"https") == 0;
    return true;
}

// Opens the connection and issues the GET, including the optional Range header.
// `status` receives the HTTP status code; 4xx/5xx are reported by the caller so
// that the response body is never mistaken for a model file.
bool open_get_request(const std::string & url, uint64_t resume_offset, int timeout_ms,
                      Request * out, DWORD * status, std::string * err) {
    UrlParts u;
    if (!crack_url(url, &u, err)) return false;
    if (!u.secure) {
        *err = "refusing a non-https download: " + url;
        return false;
    }

    Request req;
    req.session = WinHttpHandle(::WinHttpOpen(utf8_to_wide(kUserAgentUtf8).c_str(),
                                              WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                              WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (!req.session) {
        *err = "WinHttpOpen failed: " + win_error(::GetLastError());
        return false;
    }
    // resolve / connect / send / receive, all in milliseconds.
    ::WinHttpSetTimeouts(req.session.get(), timeout_ms, timeout_ms, timeout_ms, timeout_ms);

    req.connect = WinHttpHandle(::WinHttpConnect(req.session.get(), u.host.c_str(), u.port, 0));
    if (!req.connect) {
        *err = "cannot connect to " + wide_to_utf8(u.host) + ": " + win_error(::GetLastError());
        return false;
    }

    // Redirects are followed automatically (the default policy forbids
    // https -> http downgrades), which is what GitHub release assets need.
    const DWORD flags = u.secure ? WINHTTP_FLAG_SECURE : 0;
    req.request = WinHttpHandle(::WinHttpOpenRequest(req.connect.get(), L"GET", u.path.c_str(),
                                                     nullptr, WINHTTP_NO_REFERER,
                                                     WINHTTP_DEFAULT_ACCEPT_TYPES, flags));
    if (!req.request) {
        *err = "WinHttpOpenRequest failed: " + win_error(::GetLastError());
        return false;
    }

    if (resume_offset > 0) {
        const std::wstring range =
            L"Range: bytes=" + std::to_wstring(resume_offset) + L"-";
        if (!::WinHttpAddRequestHeaders(req.request.get(), range.c_str(),
                                        static_cast<DWORD>(-1),
                                        WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE)) {
            *err = "cannot set the Range header: " + win_error(::GetLastError());
            return false;
        }
    }

    if (!::WinHttpSendRequest(req.request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                              WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        *err = "cannot send the request: " + win_error(::GetLastError());
        return false;
    }
    if (!::WinHttpReceiveResponse(req.request.get(), nullptr)) {
        *err = "no response from the server: " + win_error(::GetLastError());
        return false;
    }

    DWORD code = 0;
    DWORD code_len = sizeof(code);
    if (!::WinHttpQueryHeaders(req.request.get(),
                               WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                               WINHTTP_HEADER_NAME_BY_INDEX, &code, &code_len,
                               WINHTTP_NO_HEADER_INDEX)) {
        *err = "cannot read the HTTP status: " + win_error(::GetLastError());
        return false;
    }
    *status = code;

    *out = std::move(req);
    return true;
}

// Content-Length is read as text because the DWORD form of the query truncates
// at 4 GiB, and GGUF models are bigger than that.
uint64_t content_length_of(HINTERNET request) {
    wchar_t buf[64] = {};
    DWORD len = buf_chars(buf);
    if (!::WinHttpQueryHeaders(request, WINHTTP_QUERY_CONTENT_LENGTH,
                               WINHTTP_HEADER_NAME_BY_INDEX, buf, &len, WINHTTP_NO_HEADER_INDEX)) {
        return 0;
    }
    return static_cast<uint64_t>(::_wcstoui64(buf, nullptr, 10));
}

bool write_all(HANDLE h, const char * data, DWORD n, std::string * err) {
    DWORD off = 0;
    while (off < n) {
        DWORD written = 0;
        if (!::WriteFile(h, data + off, n - off, &written, nullptr) || written == 0) {
            *err = "cannot write to the .part file: " + win_error(::GetLastError());
            return false;
        }
        off += written;
    }
    return true;
}

struct GetOutcome {
    DWORD status = 0;
    uint64_t written = 0;
    bool range_ignored = false;     // resumed, but the server answered 200
    bool already_complete = false;  // resumed, server answered 416
};

// Streams the body into `file` (no size limit) or into `text` (capped).
bool winhttp_get(const std::string & url, uint64_t resume_offset, int timeout_ms, HANDLE file,
                 std::string * text, Progress * progress, const std::atomic<bool> * cancel,
                 const std::function<void()> * on_tick, GetOutcome * outcome, std::string * err) {
    Request req;
    DWORD status = 0;
    if (!open_get_request(url, resume_offset, timeout_ms, &req, &status, err)) return false;
    outcome->status = status;

    if (resume_offset > 0 && status == 200) {
        // The server ignored our Range header; report it before any body byte is
        // written so the caller can truncate the .part file and start over.
        outcome->range_ignored = true;
        return true;
    }
    if (resume_offset > 0 && status == 416) {
        // Requested range starts past EOF: the .part file is already complete.
        outcome->already_complete = true;
        return true;
    }
    if (status >= 400) {
        *err = "HTTP " + std::to_string(status) + " for " + url;
        return false;
    }
    if (status != 200 && status != 206) {
        *err = "unexpected HTTP status " + std::to_string(status) + " for " + url;
        return false;
    }

    if (progress != nullptr) {
        uint64_t total = content_length_of(req.request.get());
        if (total > 0 && resume_offset > 0 && status == 206) total += resume_offset;
        progress->total.store(total);
    }

    constexpr DWORD kChunk = 64 * 1024;
    std::vector<char> buf(kChunk);
    for (;;) {
        if (cancel != nullptr && cancel->load()) {
            *err = "cancelled";  // the .part file is kept so we can resume later
            return false;
        }
        DWORD avail = 0;
        if (!::WinHttpQueryDataAvailable(req.request.get(), &avail)) {
            *err = "connection lost while downloading: " + win_error(::GetLastError());
            return false;
        }
        if (avail == 0) break;  // end of body
        const DWORD want = avail > kChunk ? kChunk : avail;
        DWORD got = 0;
        if (!::WinHttpReadData(req.request.get(), buf.data(), want, &got)) {
            *err = "read from the server failed: " + win_error(::GetLastError());
            return false;
        }
        if (got == 0) break;

        if (file != INVALID_HANDLE_VALUE) {
            if (!write_all(file, buf.data(), got, err)) return false;
        } else if (text != nullptr) {
            constexpr size_t kMaxText = 64u * 1024u * 1024u;  // manifests are tiny
            if (text->size() + got > kMaxText) {
                *err = "response too large";
                return false;
            }
            text->append(buf.data(), got);
        }

        outcome->written += got;
        if (progress != nullptr) progress->done.fetch_add(got);
        if (on_tick != nullptr && *on_tick) (*on_tick)();
    }
    return true;
}

// Opens dest.part: truncated for a fresh download, positioned at the end when
// resuming (the Range offset always equals the current file size).
bool open_part_file(const std::string & part, uint64_t offset, FileHandle * out,
                    std::string * err) {
    const std::wstring wpart = utf8_to_wide(part);
    FileHandle f(::CreateFileW(wpart.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                               OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                               nullptr));
    if (!f) {
        *err = "cannot open " + part + " for writing: " + win_error(::GetLastError());
        return false;
    }
    LARGE_INTEGER pos {};
    pos.QuadPart = static_cast<LONGLONG>(offset);
    if (!::SetFilePointerEx(f.get(), pos, nullptr, FILE_BEGIN)) {
        *err = "cannot seek in " + part + ": " + win_error(::GetLastError());
        return false;
    }
    if (offset == 0 && !::SetEndOfFile(f.get())) {
        *err = "cannot truncate " + part + ": " + win_error(::GetLastError());
        return false;
    }
    *out = std::move(f);
    return true;
}

bool do_download(const std::string & url, const std::string & dest_path,
                 const std::string & expected_sha256, Progress * progress,
                 const std::atomic<bool> * cancel, std::string * err,
                 const std::function<void()> & on_tick) {
    const std::string part = dest_path + ".part";
    uint64_t offset = file_size_utf8(part);
    if (progress != nullptr) progress->done.store(offset);

    std::string last_err;
    // Two attempts at most: the second one restarts from zero when the server
    // ignored our Range request or the partial file was corrupt.
    for (int attempt = 0; attempt < 2; ++attempt) {
        FileHandle file;
        if (!open_part_file(part, offset, &file, err)) return false;

        GetOutcome outcome;
        if (!winhttp_get(url, offset, 30000, file.get(), nullptr, progress, cancel, &on_tick,
                         &outcome, err)) {
            return false;  // keeps the .part file (unless it was "cancelled")
        }
        file.reset();

        if (outcome.already_complete) {
            if (finish_download(part, dest_path, expected_sha256, progress, err)) return true;
            last_err = *err;
            offset = file_size_utf8(part);  // 0 after a hash mismatch removed it
            continue;
        }
        if (outcome.range_ignored) {
            // Nothing was written, so drop the stale .part and fetch it again.
            if (!remove_file(part)) {
                *err = "cannot remove " + part + ": " + win_error(::GetLastError());
                return false;
            }
            offset = 0;
            if (progress != nullptr) progress->done.store(0);
            continue;
        }

        if (progress != nullptr && progress->total.load() > 0 &&
            progress->done.load() != progress->total.load()) {
            *err = "download ended early: got " + std::to_string(progress->done.load()) + " of " +
                   std::to_string(progress->total.load()) + " bytes";
            return false;
        }
        return finish_download(part, dest_path, expected_sha256, progress, err);
    }

    *err = last_err.empty() ? std::string("download failed") : last_err;
    return false;
}

}  // namespace

bool download(const std::string & url, const std::string & dest_path,
              const std::string & expected_sha256, Progress * progress,
              const std::atomic<bool> * cancel, std::string * err,
              const std::function<void()> & on_tick) {
    std::string local_err;
    if (err == nullptr) err = &local_err;
    err->clear();
    if (url.empty() || dest_path.empty()) {
        *err = "download: empty url or destination";
        return false;
    }
    if (progress != nullptr) {
        progress->done.store(0);
        progress->total.store(0);
        progress->ok.store(false);
    }
    const bool ok = do_download(url, dest_path, expected_sha256, progress, cancel, err, on_tick);
    if (ok && progress != nullptr) progress->ok.store(true);
    return ok;
}

bool fetch_text(const std::string & url, std::string * out, std::string * err) {
    std::string local_err;
    if (err == nullptr) err = &local_err;
    err->clear();
    if (out == nullptr) {
        *err = "fetch_text: null output";
        return false;
    }
    if (url.empty()) {
        *err = "fetch_text: empty url";
        return false;
    }

    GetOutcome outcome;
    std::string body;
    if (!winhttp_get(url, 0, 30000, INVALID_HANDLE_VALUE, &body, nullptr, nullptr, nullptr,
                     &outcome, err)) {
        return false;
    }
    *out = std::move(body);
    return true;
}

}  // namespace asr::net

#endif  // _WIN32
