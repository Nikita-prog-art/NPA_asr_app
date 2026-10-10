// Model/release downloads.  Implemented twice: net_win.cpp (WinHTTP, no external
// dependency) and net_unix.cpp (libcurl, falling back to spawning curl/wget).
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace asr::net {

struct Progress {
    std::atomic<uint64_t> done{0};
    std::atomic<uint64_t> total{0};   // 0 when the server does not report a size
    std::atomic<bool>     ok{false};
};

// Blocking download of `url` to `dest_path`.
//  * writes to `dest_path`.part and renames on success
//  * resumes with a Range request when the .part file exists
//  * verifies sha256 (lower-case hex) when expected_sha256 is non-empty
//  * calls `on_tick` (optional) after every chunk, from this thread
// Returns false and fills *err on failure.  If cancel is set the function
// returns false with err == "cancelled".
bool download(const std::string & url,
              const std::string & dest_path,
              const std::string & expected_sha256,
              Progress * progress,
              const std::atomic<bool> * cancel,
              std::string * err,
              const std::function<void()> & on_tick = {});

// Blocking GET of a small resource (manifest JSON) into a string.
bool fetch_text(const std::string & url, std::string * out, std::string * err);

}  // namespace asr::net
