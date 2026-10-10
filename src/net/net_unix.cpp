// Linux/macOS downloader.
//
// Two implementations behind one API:
//  * libcurl (preferred) - used when CMake found curl and therefore defined
//    ASR_HAVE_LIBCURL and linked CURL::libcurl.
//  * spawning the `curl` (or `wget`) command line tool - the fallback for
//    builds without libcurl headers, so the file always compiles.
#include "net/net.h"

#if defined(_WIN32)
// Windows builds compile net_win.cpp; this translation unit stays empty there.
#else

#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/stat.h>  // stat() for the .part file size
#include <sys/types.h>
#include <vector>

#if defined(ASR_HAVE_LIBCURL) && defined(__has_include)
#  if __has_include(<curl/curl.h>)
#    include <curl/curl.h>
#    define ASR_NET_LIBCURL 1
#  endif
#endif
#ifndef ASR_NET_LIBCURL
#  define ASR_NET_LIBCURL 0
#endif

#if ASR_NET_LIBCURL
#  include <mutex>
#else
#  include <fcntl.h>
#  include <signal.h>
#  include <sys/wait.h>
#  include <time.h>
#  include <unistd.h>
#endif

#include "util/sha256.h"

#ifndef ASR_VERSION
#  define ASR_VERSION "0.1.0"
#endif

namespace asr::net {
namespace {

namespace fs = std::filesystem;

// GitHub rejects some default agents, so always identify ourselves.
const char * kUserAgent = "npa_asr/" ASR_VERSION;

std::string lower_ascii(std::string s) {
    for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

uint64_t file_size(const std::string & path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return 0;
    if (!S_ISREG(st.st_mode)) return 0;
    return static_cast<uint64_t>(st.st_size);
}

// Verifies the sha256 (when one was published) and only then moves
// dest.part -> dest, so a crash never leaves a half written model in place.
bool finish_download(const std::string & part, const std::string & dest,
                     const std::string & expected_sha256, Progress * progress,
                     std::string * err) {
    if (!expected_sha256.empty()) {
        const std::string want = lower_ascii(expected_sha256);
        const std::string got = util::sha256_file(part);
        if (got.empty()) {
            *err = "cannot read the downloaded file " + part;
            return false;
        }
        if (got != want) {
            *err = "sha256 mismatch: expected " + want + ", got " + got;
            std::error_code ec;
            fs::remove(part, ec);  // corrupt payload: never resume from it
            return false;
        }
    }

    std::error_code ec;
    fs::rename(part, dest, ec);
    if (ec) {
        fs::remove(dest, ec);  // replace an existing model if the fs refuses
        ec.clear();
        fs::rename(part, dest, ec);
        if (ec) {
            *err = "cannot move " + part + " to " + dest + ": " + ec.message();
            return false;
        }
    }
    if (progress != nullptr) progress->ok.store(true);
    return true;
}

#if ASR_NET_LIBCURL

// ---------------------------------------------------------------- libcurl
void curl_global_init_once() {
    // curl_global_init is not thread safe; downloads may run off the UI thread.
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

bool starts_with_ci(const std::string & s, const char * prefix) {
    const size_t n = std::strlen(prefix);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        if (std::tolower(static_cast<unsigned char>(s[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

struct Transfer {
    FILE * fp = nullptr;
    Progress * progress = nullptr;
    const std::atomic<bool> * cancel = nullptr;
    const std::function<void()> * tick = nullptr;
    uint64_t resumed_from = 0;      // bytes already in the .part file
    uint64_t content_length = 0;    // last Content-Length seen
    long status = 0;                // last HTTP status (redirects emit several)
    bool range_ignored = false;     // resumed, but the server answered 200
    bool already_complete = false;  // resumed, server answered 416
    bool cancelled = false;
    std::string io_error;
};

size_t write_cb(char * ptr, size_t size, size_t nmemb, void * userdata) {
    Transfer * t = static_cast<Transfer *>(userdata);
    const size_t n = size * nmemb;
    if (n == 0) return 0;
    if (t->fp == nullptr || std::fwrite(ptr, 1, n, t->fp) != n) {
        t->io_error = "cannot write to the .part file";
        return 0;  // short count -> CURLE_WRITE_ERROR, transfer aborted
    }
    if (t->progress != nullptr) t->progress->done.fetch_add(static_cast<uint64_t>(n));
    if (t->tick != nullptr && *t->tick) (*t->tick)();
    return n;
}

// Returning non-zero aborts the transfer with CURLE_ABORTED_BY_CALLBACK; this
// is the only place a cancel request can interrupt a blocking easy_perform.
int xferinfo_cb(void * userdata, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    Transfer * t = static_cast<Transfer *>(userdata);
    if (t->cancel != nullptr && t->cancel->load()) {
        t->cancelled = true;
        return 1;
    }
    return 0;
}

size_t header_cb(char * buffer, size_t size, size_t nitems, void * userdata) {
    Transfer * t = static_cast<Transfer *>(userdata);
    const size_t n = size * nitems;
    const std::string line(buffer, n);

    if (line.compare(0, 5, "HTTP/") == 0) {
        // A redirect chain emits several status lines; the last one wins.
        const size_t sp = line.find(' ');
        t->status = sp == std::string::npos ? 0 : std::strtol(line.c_str() + sp + 1, nullptr, 10);
        if (t->resumed_from > 0 && t->status == 200) t->range_ignored = true;
        if (t->resumed_from > 0 && t->status == 416) t->already_complete = true;
        // Abort in the header phase so nothing is appended to the .part file.
        if (t->range_ignored || t->already_complete) return 0;
    } else if (starts_with_ci(line, "content-length:")) {
        t->content_length = std::strtoull(line.c_str() + 15, nullptr, 10);
        if (t->progress != nullptr) {
            uint64_t total = t->content_length;
            if (t->resumed_from > 0 && t->status == 206) total += t->resumed_from;
            t->progress->total.store(total);
        }
    } else if (starts_with_ci(line, "content-range:")) {
        // "bytes <first>-<last>/<total>"
        const size_t slash = line.find('/');
        if (slash != std::string::npos) {
            const uint64_t total = std::strtoull(line.c_str() + slash + 1, nullptr, 10);
            if (total > 0 && t->progress != nullptr) t->progress->total.store(total);
        }
    }
    return n;
}

// One GET.  The caller owns `t` and the open file.  Returns the CURLcode.
CURLcode perform_transfer(const std::string & url, Transfer * t) {
    CURL * h = curl_easy_init();
    if (h == nullptr) return CURLE_FAILED_INIT;

    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(h, CURLOPT_FAILONERROR, 1L);  // 4xx/5xx -> error, no body
    curl_easy_setopt(h, CURLOPT_USERAGENT, kUserAgent);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, t);
    curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, header_cb);
    curl_easy_setopt(h, CURLOPT_HEADERDATA, t);
    curl_easy_setopt(h, CURLOPT_XFERINFOFUNCTION, xferinfo_cb);
    curl_easy_setopt(h, CURLOPT_XFERINFODATA, t);
    curl_easy_setopt(h, CURLOPT_NOPROGRESS, 0L);
    if (t->resumed_from > 0) {
        curl_easy_setopt(h, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(t->resumed_from));
    }
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, 60L);  // give up on stalled links

    const CURLcode rc = curl_easy_perform(h);
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &t->status);
    curl_easy_cleanup(h);
    return rc;
}

bool libcurl_download(const std::string & url, const std::string & dest_path,
                      const std::string & expected_sha256, Progress * progress,
                      const std::atomic<bool> * cancel, std::string * err,
                      const std::function<void()> & on_tick) {
    curl_global_init_once();

    const std::string part = dest_path + ".part";
    uint64_t offset = file_size(part);
    std::string last_err;

    // At most two attempts: the second one starts from zero when the server
    // refused/ignored our Range request.
    for (int attempt = 0; attempt < 2; ++attempt) {
        Transfer t;
        t.progress = progress;
        t.cancel = cancel;
        t.tick = &on_tick;
        t.resumed_from = offset;
        if (progress != nullptr) {
            progress->done.store(offset);
            progress->total.store(0);
        }

        FILE * fp = std::fopen(part.c_str(), offset > 0 ? "ab" : "wb");
        if (fp == nullptr) {
            *err = "cannot open " + part + " for writing: " + std::strerror(errno);
            return false;
        }
        t.fp = fp;
        const CURLcode rc = perform_transfer(url, &t);
        std::fclose(fp);
        t.fp = nullptr;

        if (t.cancelled || rc == CURLE_ABORTED_BY_CALLBACK) {
            // The .part file is kept so the download can resume later.
            *err = "cancelled";
            return false;
        }
        if (t.already_complete) {
            // The range starts past EOF: the .part already holds the whole
            // file, so just verify and move it into place.
            if (finish_download(part, dest_path, expected_sha256, progress, err)) return true;
            last_err = *err;
            offset = file_size(part);  // 0 after a hash mismatch removed it
            continue;
        }
        if (t.range_ignored) {
            // Server answered 200 to our Range request; nothing was written,
            // so drop the partial file and fetch everything again.
            std::error_code ec;
            fs::remove(part, ec);
            offset = 0;
            continue;
        }
        if (rc != CURLE_OK) {
            last_err = std::string("download failed: ") + curl_easy_strerror(rc);
            if (!t.io_error.empty()) last_err += " (" + t.io_error + ")";
            *err = last_err;
            return false;  // keep the partial file for a later resume
        }
        if (progress != nullptr && progress->total.load() > 0 &&
            progress->done.load() != progress->total.load()) {
            *err = "download ended early: got " + std::to_string(progress->done.load()) +
                   " of " + std::to_string(progress->total.load()) + " bytes";
            return false;
        }
        return finish_download(part, dest_path, expected_sha256, progress, err);
    }

    *err = last_err.empty() ? std::string("download failed") : last_err;
    return false;
}

size_t text_write_cb(char * ptr, size_t size, size_t nmemb, void * userdata) {
    std::string * out = static_cast<std::string *>(userdata);
    const size_t n = size * nmemb;
    constexpr size_t kMaxText = 64u * 1024u * 1024u;  // manifests are tiny
    if (out->size() + n > kMaxText) return 0;
    out->append(ptr, n);
    return n;
}

bool libcurl_fetch_text(const std::string & url, std::string * out, std::string * err) {
    curl_global_init_once();

    CURL * h = curl_easy_init();
    if (h == nullptr) {
        *err = "curl_easy_init failed";
        return false;
    }
    std::string body;
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(h, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(h, CURLOPT_USERAGENT, kUserAgent);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(h, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(h, CURLOPT_MAXFILESIZE_LARGE, static_cast<curl_off_t>(64) * 1024 * 1024);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, text_write_cb);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &body);

    const CURLcode rc = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(h);

    if (rc != CURLE_OK) {
        *err = std::string("GET ") + url + " failed: " + curl_easy_strerror(rc);
        return false;
    }
    *out = std::move(body);
    return true;
}

#else  // !ASR_NET_LIBCURL

// ------------------------------------------------- curl/wget child process
void sleep_ms(long ms) {
    struct timespec ts {};
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    while (::nanosleep(&ts, &ts) != 0 && errno == EINTR) {
    }
}

// Looks the tool up in PATH ourselves so we can report "not installed" instead
// of a bare exit code 127 from the forked child.
std::string find_tool(const char * name) {
    const char * path = ::getenv("PATH");
    if (path == nullptr) return {};
    const std::string dirs(path);
    size_t begin = 0;
    while (begin <= dirs.size()) {
        const size_t end = dirs.find(':', begin);
        const std::string dir =
            dirs.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!dir.empty()) {
            const std::string candidate = dir + "/" + name;
            if (::access(candidate.c_str(), X_OK) == 0) return candidate;
        }
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return {};
}

std::string tail_of(const std::string & s, size_t max) {
    if (s.size() <= max) return s;
    return s.substr(s.size() - max);
}

std::string trim_ws(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == '\n' || s[i] == '\r' || s[i] == ' ')) ++i;
    return s.substr(i);
}

struct ChildOutcome {
    bool started = false;
    bool cancelled = false;
    int exit_code = -1;
    std::string err_text;  // tail of the child's stderr
};

void drain_fd(int fd, std::string * sink, size_t keep) {
    char buf[4096];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n > 0) {
            sink->append(buf, static_cast<size_t>(n));
            if (sink->size() > keep * 4) *sink = tail_of(*sink, keep);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        break;  // 0 = EOF, EAGAIN = nothing buffered right now
    }
}

// Runs argv and polls the size of `part_path` for progress.  stderr goes to a
// pipe that we drain with a non-blocking read so the child can never block on a
// full pipe while we are sleeping.
ChildOutcome run_child_poll(const std::vector<std::string> & argv, const std::string & part_path,
                            Progress * progress, const std::atomic<bool> * cancel,
                            const std::function<void()> & on_tick) {
    ChildOutcome out;
    int pipefd[2] = {-1, -1};
    if (::pipe(pipefd) != 0) return out;

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        return out;
    }

    if (pid == 0) {
        // Child: stderr -> pipe, stdin -> /dev/null (never steal the terminal).
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);
        const int devnull = ::open("/dev/null", O_RDONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDIN_FILENO);
            ::close(devnull);
        }
        std::vector<char *> args;
        args.reserve(argv.size() + 1);
        for (const std::string & a : argv) args.push_back(const_cast<char *>(a.c_str()));
        args.push_back(nullptr);
        ::execv(args[0], args.data());  // only returns on failure
        std::fprintf(stderr, "cannot execute %s: %s\n", args[0], std::strerror(errno));
        ::_exit(127);
    }

    // Parent.
    out.started = true;
    ::close(pipefd[1]);
    ::fcntl(pipefd[0], F_SETFL, O_NONBLOCK);

    std::string seen_err;
    int status = 0;
    bool cancel_sent = false;
    int polls_since_cancel = 0;

    for (;;) {
        if (!cancel_sent && cancel != nullptr && cancel->load()) {
            ::kill(pid, SIGTERM);  // keeps curl's -C - state in the .part file
            cancel_sent = true;
        }
        if (cancel_sent && ++polls_since_cancel > 60) ::kill(pid, SIGKILL);  // stuck

        const pid_t w = ::waitpid(pid, &status, WNOHANG);
        if (w == pid) break;
        if (w < 0 && errno != EINTR) break;  // ECHILD: nothing left to wait for

        drain_fd(pipefd[0], &seen_err, 2048);
        if (progress != nullptr) progress->done.store(file_size(part_path));  // total unknown
        if (on_tick) on_tick();
        sleep_ms(100);
    }

    drain_fd(pipefd[0], &seen_err, 2048);
    ::close(pipefd[0]);
    out.err_text = tail_of(trim_ws(seen_err), 2048);

    if (cancel_sent) {
        out.cancelled = true;
        return out;
    }
    if (WIFEXITED(status)) {
        out.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        out.exit_code = 128 + WTERMSIG(status);
    }
    return out;
}

// Reads the child's stdout into `out` (blocking read loop; the child writes
// nothing else, so there is no deadlock risk).  Returns the exit code or -1.
int run_child_capture(const std::vector<std::string> & argv, std::string * out,
                      std::string * err_text) {
    int pipefd[2] = {-1, -1};
    if (::pipe(pipefd) != 0) return -1;

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        return -1;
    }
    if (pid == 0) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::close(pipefd[1]);
        const int devnull = ::open("/dev/null", O_RDONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDIN_FILENO);
            ::close(devnull);
        }
        std::vector<char *> args;
        args.reserve(argv.size() + 1);
        for (const std::string & a : argv) args.push_back(const_cast<char *>(a.c_str()));
        args.push_back(nullptr);
        ::execv(args[0], args.data());
        std::fprintf(stderr, "cannot execute %s: %s\n", args[0], std::strerror(errno));
        ::_exit(127);
    }

    ::close(pipefd[1]);
    constexpr size_t kMaxText = 64u * 1024u * 1024u;
    bool too_large = false;
    char buf[64 * 1024];
    for (;;) {
        const ssize_t n = ::read(pipefd[0], buf, sizeof(buf));
        if (n > 0) {
            if (out->size() + static_cast<size_t>(n) > kMaxText) {
                too_large = true;
                ::kill(pid, SIGKILL);  // SIGKILL guarantees the drain below ends
                break;
            }
            out->append(buf, static_cast<size_t>(n));
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
    // Drain whatever is left; after SIGKILL the child closes the pipe.
    while (::read(pipefd[0], buf, sizeof(buf)) > 0) {
    }
    ::close(pipefd[0]);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) break;
    }
    if (too_large) {
        *err_text = "response too large";
        out->clear();
        return -1;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

std::vector<std::string> curl_download_argv(const std::string & tool, const std::string & url,
                                            const std::string & part) {
    // --continue-at - resumes from the size of the existing output file.
    return {tool,      "--fail",        "--location",     "--silent", "--show-error",
            "--retry", "3",            "--retry-delay",  "1",        "--connect-timeout",
            "30",      "--user-agent", kUserAgent,       "--continue-at",
            "-",       "--output",     part,             url};
}

std::vector<std::string> wget_download_argv(const std::string & tool, const std::string & url,
                                            const std::string & part) {
    // No --continue here on purpose: GNU wget with -c -O seeks to the resume
    // offset and, when the server answers 200 instead of 206, writes the body
    // there anyway - a silently corrupt file with a correct total size and exit
    // code 0.  wget only restarts safely when it owns the output filename, so
    // the wget fallback downloads from scratch; curl and libcurl resume.
    return {tool, "--quiet", "--tries=3", "--timeout=30",
            "--output-document=" + part, url};
}

bool spawn_download(const std::string & url, const std::string & dest_path,
                    const std::string & expected_sha256, Progress * progress,
                    const std::atomic<bool> * cancel, std::string * err,
                    const std::function<void()> & on_tick) {
    const std::string curl = find_tool("curl");
    const std::string wget = curl.empty() ? find_tool("wget") : std::string();
    if (curl.empty() && wget.empty()) {
        *err = "neither curl nor wget was found in PATH";
        return false;
    }
    const bool use_curl = !curl.empty();
    const std::string tool = use_curl ? curl : wget;

    const std::string part = dest_path + ".part";
    uint64_t offset = file_size(part);
    if (!use_curl && offset > 0) {
        // The wget fallback cannot resume safely (see wget_download_argv), so
        // start over instead of appending to a file we do not trust.
        std::error_code ec;
        fs::remove(part, ec);
        offset = 0;
    }
    std::string last_err;

    for (int attempt = 0; attempt < 2; ++attempt) {
        if (progress != nullptr) {
            progress->done.store(offset);
            progress->total.store(0);  // the CLI gives us no size up front
        }
        const std::vector<std::string> argv =
            use_curl ? curl_download_argv(tool, url, part) : wget_download_argv(tool, url, part);
        const ChildOutcome outcome = run_child_poll(argv, part, progress, cancel, on_tick);
        if (!outcome.started) {
            *err = "cannot start " + tool;
            return false;
        }
        if (outcome.cancelled) {
            *err = "cancelled";  // the .part file is kept so curl can resume it
            return false;
        }
        if (outcome.exit_code == 0) {
            return finish_download(part, dest_path, expected_sha256, progress, err);
        }

        last_err = tool + " exited with code " + std::to_string(outcome.exit_code);
        if (!outcome.err_text.empty()) last_err += ": " + outcome.err_text;

        const uint64_t after = file_size(part);
        if (offset > 0 && after <= offset) {
            // Nothing was appended: the server most likely ignored our Range
            // request (curl exits 33).  Drop the file and fetch from scratch.
            std::error_code ec;
            fs::remove(part, ec);
            offset = 0;
            continue;
        }
        // Real progress was made but the transfer still broke: the curl path
        // keeps the .part file so a later call can resume it.
        *err = last_err;
        return false;
    }

    *err = last_err.empty() ? std::string("download failed") : last_err;
    return false;
}

bool spawn_fetch_text(const std::string & url, std::string * out, std::string * err) {
    const std::string curl = find_tool("curl");
    const std::string wget = curl.empty() ? find_tool("wget") : std::string();
    if (curl.empty() && wget.empty()) {
        *err = "neither curl nor wget was found in PATH";
        return false;
    }

    std::vector<std::string> argv;
    if (!curl.empty()) {
        argv = {curl, "--fail",     "--location",         "--silent",
                "--show-error",     "--max-time", "30", "--user-agent",
                kUserAgent,         url};
    } else {
        argv = {wget, "--quiet", "--tries=3", "--timeout=30", "--output-document=-", url};
    }

    std::string body;
    std::string child_err;
    const int code = run_child_capture(argv, &body, &child_err);
    if (code != 0) {
        *err = "GET " + url + " failed";
        if (code > 0) *err += " (exit code " + std::to_string(code) + ")";
        if (!child_err.empty()) *err += ": " + child_err;
        return false;
    }
    *out = std::move(body);
    return true;
}

#endif  // ASR_NET_LIBCURL

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

    const bool ok = [&] {
#if ASR_NET_LIBCURL
        return libcurl_download(url, dest_path, expected_sha256, progress, cancel, err, on_tick);
#else
        return spawn_download(url, dest_path, expected_sha256, progress, cancel, err, on_tick);
#endif
    }();

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
#if ASR_NET_LIBCURL
    return libcurl_fetch_text(url, out, err);
#else
    return spawn_fetch_text(url, out, err);
#endif
}

}  // namespace asr::net

#endif  // !_WIN32
