// Persisted application settings (JSON in the per-user data directory).
#pragma once

#include <string>

namespace asr::app {

enum class OutputMode {
    InsertAndClipboard = 0,  // paste into the focused window and copy
    ClipboardOnly = 1,
    WindowOnly = 2,
};

const char * output_mode_name(OutputMode m);
OutputMode output_mode_from_name(const std::string & s);

struct Settings {
    std::string model_path;              // path to the .gguf
    int n_threads = 0;                   // 0 = hardware concurrency
    std::string hotkey = "Ctrl+Alt+Space";
    bool dark_theme = true;
    int audio_device = -1;               // -1 = system default
    OutputMode output_mode = OutputMode::InsertAndClipboard;
    float step_sec = 1.0f;               // streaming re-encode interval
    float max_window_sec = 20.0f;        // sliding window length
    // Off by default: a window that floats above everything else never gets out
    // of the way when the user switches to the application they dictate into.
    bool always_on_top = false;
    float ui_scale = 0.0f;               // 0 = follow the monitor DPI
    bool sound_on_done = false;

    // Loads from <user_data_dir>/settings.json, falling back to the defaults.
    static Settings load();
    bool save(std::string * err = nullptr) const;

    static std::string settings_path();
};

}  // namespace asr::app
