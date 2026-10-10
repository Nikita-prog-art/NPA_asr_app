// Application controller: hotkey -> microphone -> recognizer -> text output.
//
// Threading:
//   * the hotkey backend calls press_ptt()/release_ptt() from its own thread and
//     only flips atomic flags;
//   * the audio callback pushes samples into a lock-free ring buffer;
//   * one worker thread drains the ring, drives the streaming recognizer, and
//     delivers the final text;
//   * the GUI polls status(), which is a mutex-protected snapshot.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "app/settings.h"
#include "audio/capture.h"
#include "core/recognizer.h"
#include "platform/platform.h"

namespace asr::app {

struct AppStatus {
    bool model_loaded = false;
    bool model_loading = false;
    bool listening = false;   // hotkey backend active
    bool recording = false;
    bool busy = false;        // finishing / loading
    std::string model_name;
    std::string model_path;
    std::string error;
    std::string hotkey;
    std::string hotkey_backend;
    std::string audio_device;
    std::string committed;
    std::string provisional;
    std::string final_text;
    double audio_sec = 0.0;
    double last_infer_ms = 0.0;
    double last_total_ms = 0.0;
    int n_windows = 0;
    int n_threads = 0;
    std::string cpu;
};

class Controller {
public:
    Controller();
    ~Controller();
    Controller(const Controller &) = delete;
    Controller & operator=(const Controller &) = delete;

    // Starts the worker thread and the audio device; loads the model if the
    // settings point at an existing file.
    bool init(const Settings & s, std::string * err);
    void shutdown();

    // Asynchronous model load; progress is visible through status().
    void request_model_load(const std::string & path);
    bool reload_model_sync(const std::string & path, std::string * err);

    void apply_settings(const Settings & s);

    AppStatus status() const;
    Settings settings() const;

    // Push-to-talk, safe to call from any thread.
    void press_ptt();
    void release_ptt();
    bool recording() const { return recording_.load(); }

    // UI log (ring of the last N messages).
    void log(const std::string & msg);
    std::vector<std::string> take_log();

    const std::vector<audio::DeviceInfo> & devices() const { return devices_; }
    void refresh_devices();

private:
    void worker_loop();
    void handle_model_load();
    void deliver_text(const std::string & text);

    Settings settings_;
    mutable std::mutex mutex_;
    AppStatus status_;
    std::vector<std::string> log_;
    std::vector<audio::DeviceInfo> devices_;

    std::unique_ptr<platform::HotkeyListener> hotkey_;
    std::unique_ptr<Recognizer> recognizer_;
    audio::Capture capture_;
    audio::RingBuffer ring_{1 << 20};
    std::vector<float> preroll_;
    size_t preroll_max_ = 0;

    std::thread worker_;
    std::atomic<bool> quit_{false};
    std::atomic<bool> recording_{false};
    std::atomic<bool> pending_start_{false};
    std::atomic<bool> pending_stop_{false};
    std::atomic<bool> model_load_pending_{false};
    std::string pending_model_path_;
    std::mutex model_mutex_;
    // The log has its own mutex: log() is called while mutex_ is already held
    // (init(), apply_settings(), the worker thread), so it must never touch it.
    mutable std::mutex log_mutex_;
};

}  // namespace asr::app
