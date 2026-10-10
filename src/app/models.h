// Local model discovery and background downloads of published GGUF files.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "net/manifest.h"
#include "net/net.h"

namespace asr::app {

struct LocalModel {
    std::string path;
    std::string name;   // file name
    std::string quant;  // parsed from the file name when possible
    uint64_t size = 0;
};

// <user data dir>/models - created on demand.
std::string default_models_dir();
std::vector<LocalModel> scan_models_dir(const std::string & dir);

// ------------------------------------------------------------------ downloads
struct DownloadJob {
    std::atomic<bool> running{false};
    std::atomic<bool> finished{false};
    std::atomic<bool> cancel{false};
    asr::net::Progress progress;
    std::string url, dest, error;
    bool ok = false;
    std::thread thread;

    ~DownloadJob();
    void stop();
    // 0..1, or -1 when the total size is unknown.
    double fraction() const {
        const uint64_t total = progress.total.load();
        return total ? static_cast<double>(progress.done.load()) / static_cast<double>(total) : -1.0;
    }
};

std::shared_ptr<DownloadJob> start_download(const std::string & url, const std::string & dest,
                                            const std::string & sha256);

// Blocking; run it on a worker thread.  Returns false with *err when neither the
// remote manifest nor the compiled-in one is usable.
bool fetch_manifest(asr::net::Manifest * out, std::string * err);

}  // namespace asr::app
