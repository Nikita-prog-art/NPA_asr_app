#include "app/controller.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>

namespace asr::app {

namespace {
double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}
constexpr size_t kMaxLog = 200;
}  // namespace

Controller::Controller() = default;

Controller::~Controller() { shutdown(); }

void Controller::log(const std::string & msg) {
    // own mutex: callers often already hold mutex_
    std::lock_guard<std::mutex> lk(log_mutex_);
    log_.push_back(msg);
    if (log_.size() > kMaxLog) log_.erase(log_.begin(), log_.begin() + (log_.size() - kMaxLog));
}

std::vector<std::string> Controller::take_log() {
    std::lock_guard<std::mutex> lk(log_mutex_);
    std::vector<std::string> out;
    out.swap(log_);
    return out;
}

bool Controller::init(const Settings & s, std::string * err) {
    settings_ = s;
    if (settings_.n_threads <= 0) settings_.n_threads = default_n_threads();

    {
        std::lock_guard<std::mutex> lk(mutex_);
        status_.n_threads = settings_.n_threads;
        status_.cpu = platform::cpu_brand();
        status_.hotkey = settings_.hotkey;
    }

    // ---- hotkey
    platform::Chord chord;
    std::string cerr;
    if (!platform::Chord::parse(settings_.hotkey, &chord, &cerr)) {
        log("hotkey: " + cerr + " (falling back to Ctrl+Alt+Space)");
        settings_.hotkey = "Ctrl+Alt+Space";
        platform::Chord::parse(settings_.hotkey, &chord, &cerr);
    }
    hotkey_ = platform::create_hotkey_listener(
        chord, [this]() { press_ptt(); }, [this]() { release_ptt(); });
    if (hotkey_->start(&cerr)) {
        std::lock_guard<std::mutex> lk(mutex_);
        status_.listening = true;
        status_.hotkey_backend = hotkey_->backend_name();
        log("hotkey " + chord.to_string() + " active via " + hotkey_->backend_name());
    } else {
        std::lock_guard<std::mutex> lk(mutex_);
        status_.listening = false;
        status_.error = cerr;
        log("hotkey failed: " + cerr);
    }

    // ---- audio
    refresh_devices();
    preroll_max_ = static_cast<size_t>(0.3 * 16000);
    const int rate = 16000;
    std::string aerr;
    if (!capture_.start(settings_.audio_device, rate,
                        [this](const float * pcm, int n) { ring_.write(pcm, static_cast<size_t>(n)); },
                        &aerr)) {
        log("audio: " + aerr);
        std::lock_guard<std::mutex> lk(mutex_);
        if (status_.error.empty()) status_.error = aerr;
    } else {
        std::lock_guard<std::mutex> lk(mutex_);
        status_.audio_device = capture_.device_name();
        log("capture device: " + capture_.device_name());
    }

    // ---- worker
    worker_ = std::thread([this]() { worker_loop(); });

    // ---- model
    if (!settings_.model_path.empty() && std::filesystem::exists(settings_.model_path)) {
        request_model_load(settings_.model_path);
    } else if (err && !settings_.model_path.empty()) {
        *err = "model file not found: " + settings_.model_path;
    }
    return true;
}

void Controller::refresh_devices() {
    devices_ = audio::list_capture_devices();
}

void Controller::shutdown() {
    quit_.store(true);
    if (worker_.joinable()) worker_.join();
    if (hotkey_) hotkey_->stop();
    capture_.stop();
}

// ------------------------------------------------------------------ model

void Controller::request_model_load(const std::string & path) {
    {
        std::lock_guard<std::mutex> lk(model_mutex_);
        pending_model_path_ = path;
    }
    {
        std::lock_guard<std::mutex> lk(mutex_);
        status_.model_loading = true;
        status_.busy = true;
        status_.error.clear();
    }
    model_load_pending_.store(true);
}

bool Controller::reload_model_sync(const std::string & path, std::string * err) {
    auto rec = std::make_unique<Recognizer>();
    RecognizerConfig cfg;
    cfg.model_path = path;
    cfg.n_threads = settings_.n_threads;
    cfg.step_sec = settings_.step_sec;
    cfg.max_window_sec = settings_.max_window_sec;
    if (!rec->init(cfg, err)) {
        std::lock_guard<std::mutex> lk(mutex_);
        status_.model_loading = false;
        status_.busy = false;
        status_.error = err ? *err : "model load failed";
        return false;
    }
    recognizer_ = std::move(rec);
    settings_.model_path = path;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        status_.model_loaded = true;
        status_.model_loading = false;
        status_.busy = false;
        status_.model_path = path;
        status_.model_name = recognizer_->hp().model_name;
        status_.n_threads = settings_.n_threads;
        status_.error.clear();
    }
    log("model loaded: " + path);
    return true;
}

void Controller::handle_model_load() {
    std::string path;
    {
        std::lock_guard<std::mutex> lk(model_mutex_);
        path = pending_model_path_;
    }
    std::string err;
    if (!reload_model_sync(path, &err)) log("model load failed: " + err);
}

// ------------------------------------------------------------------ settings

void Controller::apply_settings(const Settings & s) {
    const bool hotkey_changed = s.hotkey != settings_.hotkey;
    const bool device_changed = s.audio_device != settings_.audio_device;
    const bool threads_changed = s.n_threads != settings_.n_threads;
    settings_ = s;
    if (settings_.n_threads <= 0) settings_.n_threads = default_n_threads();
    if (recognizer_ && (threads_changed || s.step_sec != settings_.step_sec)) {
        recognizer_->set_n_threads(settings_.n_threads);
        RecognizerConfig cfg;
        cfg.model_path = settings_.model_path;
        cfg.n_threads = settings_.n_threads;
        cfg.step_sec = settings_.step_sec;
        cfg.max_window_sec = settings_.max_window_sec;
        // window parameters are read at begin(); nothing else to do here
        (void) cfg;
    }
    if (hotkey_changed && hotkey_) {
        hotkey_->stop();
        platform::Chord chord;
        std::string cerr;
        if (platform::Chord::parse(settings_.hotkey, &chord, &cerr)) {
            hotkey_ = platform::create_hotkey_listener(
                chord, [this]() { press_ptt(); }, [this]() { release_ptt(); });
            if (hotkey_->start(&cerr)) {
                std::lock_guard<std::mutex> lk(mutex_);
                status_.hotkey = settings_.hotkey;
                status_.hotkey_backend = hotkey_->backend_name();
                status_.listening = true;
                status_.error.clear();
                log("hotkey switched to " + chord.to_string() + " via " + hotkey_->backend_name());
            } else {
                std::lock_guard<std::mutex> lk(mutex_);
                status_.listening = false;
                status_.error = cerr;
                log("hotkey failed: " + cerr);
            }
        }
    }
    if (device_changed) {
        std::string aerr;
        capture_.stop();
        if (!capture_.start(settings_.audio_device, 16000,
                            [this](const float * pcm, int n) {
                                ring_.write(pcm, static_cast<size_t>(n));
                            },
                            &aerr)) {
            log("audio: " + aerr);
        } else {
            std::lock_guard<std::mutex> lk(mutex_);
            status_.audio_device = capture_.device_name();
            log("capture device: " + capture_.device_name());
        }
    }
    {
        std::lock_guard<std::mutex> lk(mutex_);
        status_.n_threads = settings_.n_threads;
        status_.hotkey = settings_.hotkey;
    }
}

Settings Controller::settings() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return settings_;
}

AppStatus Controller::status() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return status_;
}

// ------------------------------------------------------------------ push-to-talk

void Controller::press_ptt() {
    pending_start_.store(true);
    pending_stop_.store(false);
}

void Controller::release_ptt() {
    pending_stop_.store(true);
}

void Controller::deliver_text(const std::string & text) {
    if (text.empty()) return;
    switch (settings_.output_mode) {
        case OutputMode::WindowOnly:
            break;
        case OutputMode::ClipboardOnly:
            if (!platform::clipboard_set_text(text)) log("clipboard: failed");
            break;
        case OutputMode::InsertAndClipboard:
        default:
            if (!platform::inject_text(text)) log("insert: failed (text is on the clipboard)");
            break;
    }
}

void Controller::worker_loop() {
    std::vector<float> chunk(4096);
    std::vector<float> drain;
    while (!quit_.load()) {
        if (model_load_pending_.exchange(false)) handle_model_load();

        // keep the ring drained while idle, remembering a short pre-roll so the
        // first word is not clipped
        if (!recording_.load()) {
            size_t n;
            while ((n = ring_.read(chunk.data(), chunk.size())) > 0) {
                preroll_.insert(preroll_.end(), chunk.begin(), chunk.begin() + static_cast<long>(n));
                if (preroll_.size() > preroll_max_) {
                    preroll_.erase(preroll_.begin(),
                                   preroll_.begin() + static_cast<long>(preroll_.size() - preroll_max_));
                }
            }
        }

        if (pending_start_.exchange(false) && recognizer_ && !recording_.load()) {
            recognizer_->begin();
            if (!preroll_.empty()) recognizer_->feed(preroll_.data(), static_cast<int>(preroll_.size()));
            preroll_.clear();
            recording_.store(true);
            std::lock_guard<std::mutex> lk(mutex_);
            status_.recording = true;
            status_.final_text.clear();
            status_.committed.clear();
            status_.provisional.clear();
            status_.audio_sec = 0;
        }

        if (recording_.load()) {
            size_t n;
            while ((n = ring_.read(chunk.data(), chunk.size())) > 0) {
                recognizer_->feed(chunk.data(), static_cast<int>(n));
            }
            {
                std::string c, p;
                recognizer_->snapshot(&c, &p);
                std::lock_guard<std::mutex> lk(mutex_);
                status_.committed = c;
                status_.provisional = p;
                status_.audio_sec = recognizer_->audio_seconds();
            }
        }

        if (pending_stop_.exchange(false) && recording_.load()) {
            // drain whatever is left, then do the final pass
            size_t n;
            while ((n = ring_.read(chunk.data(), chunk.size())) > 0) {
                recognizer_->feed(chunk.data(), static_cast<int>(n));
            }
            recording_.store(false);
            const double t0 = now_ms();
            Transcription tr;
            std::string err;
            if (recognizer_->finish(&tr, &err)) {
                deliver_text(tr.text);
                std::lock_guard<std::mutex> lk(mutex_);
                status_.final_text = tr.text;
                status_.committed = tr.text;
                status_.provisional.clear();
                status_.last_infer_ms = tr.infer_ms;
                status_.last_total_ms = now_ms() - t0;
                status_.n_windows = tr.n_windows;
                status_.audio_sec = tr.audio_sec;
                log("transcribed " + std::to_string(tr.audio_sec) + "s in " +
                    std::to_string(static_cast<int>(tr.infer_ms)) + " ms");
            } else {
                log("transcription failed: " + err);
                std::lock_guard<std::mutex> lk(mutex_);
                status_.error = err;
            }
            std::lock_guard<std::mutex> lk(mutex_);
            status_.recording = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(recording_.load() ? 10 : 50));
    }
}

}  // namespace asr::app
