#include "app/models.h"

#include <algorithm>
#include <filesystem>

#include "platform/platform.h"

namespace fs = std::filesystem;

namespace asr::app {

std::string default_models_dir() {
    const fs::path p = fs::path(platform::user_data_dir()) / "models";
    std::error_code ec;
    fs::create_directories(p, ec);
    return p.string();
}

std::vector<LocalModel> scan_models_dir(const std::string & dir) {
    std::vector<LocalModel> out;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return out;
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) continue;
        const std::string name = e.path().filename().string();
        if (e.path().extension() != ".gguf") continue;
        LocalModel m;
        m.path = e.path().string();
        m.name = name;
        m.size = static_cast<uint64_t>(e.file_size(ec));
        // "gigaam-v3-e2e_rnnt-q8_0.gguf" -> "q8_0"
        const std::string stem = e.path().stem().string();
        const size_t dash = stem.rfind('-');
        if (dash != std::string::npos) m.quant = stem.substr(dash + 1);
        out.push_back(std::move(m));
    }
    std::sort(out.begin(), out.end(), [](const LocalModel & a, const LocalModel & b) {
        return a.name < b.name;
    });
    return out;
}

DownloadJob::~DownloadJob() { stop(); }

void DownloadJob::stop() {
    cancel.store(true);
    if (thread.joinable()) thread.join();
}

std::shared_ptr<DownloadJob> start_download(const std::string & url, const std::string & dest,
                                            const std::string & sha256) {
    auto job = std::make_shared<DownloadJob>();
    job->url = url;
    job->dest = dest;
    job->running.store(true);
    std::weak_ptr<DownloadJob> weak = job;
    job->thread = std::thread([weak, url, dest, sha256]() {
        auto self = weak.lock();
        if (!self) return;
        std::string err;
        const bool ok = asr::net::download(url, dest, sha256, &self->progress, &self->cancel, &err);
        self->ok = ok;
        self->error = err;
        self->running.store(false);
        self->finished.store(true);
    });
    return job;
}

bool fetch_manifest(asr::net::Manifest * out, std::string * err) {
    return asr::net::manifest_fetch(out, err);
}

}  // namespace asr::app
