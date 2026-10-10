#include "app/settings.h"

#include <filesystem>
#include <fstream>

#include "json.hpp"
#include "platform/platform.h"

namespace asr::app {

const char * output_mode_name(OutputMode m) {
    switch (m) {
        case OutputMode::ClipboardOnly: return "clipboard";
        case OutputMode::WindowOnly: return "window";
        default: return "insert+clipboard";
    }
}

OutputMode output_mode_from_name(const std::string & s) {
    if (s == "clipboard") return OutputMode::ClipboardOnly;
    if (s == "window") return OutputMode::WindowOnly;
    return OutputMode::InsertAndClipboard;
}

std::string Settings::settings_path() {
    return (std::filesystem::path(platform::user_data_dir()) / "settings.json").string();
}

Settings Settings::load() {
    Settings s;
    std::ifstream f(settings_path());
    if (!f) return s;
    try {
        nlohmann::json j;
        f >> j;
        s.model_path = j.value("model_path", s.model_path);
        s.n_threads = j.value("n_threads", s.n_threads);
        s.hotkey = j.value("hotkey", s.hotkey);
        s.dark_theme = j.value("dark_theme", s.dark_theme);
        s.audio_device = j.value("audio_device", s.audio_device);
        s.output_mode = output_mode_from_name(j.value("output_mode", std::string("insert+clipboard")));
        s.step_sec = j.value("step_sec", s.step_sec);
        s.max_window_sec = j.value("max_window_sec", s.max_window_sec);
        s.ui_scale = j.value("ui_scale", s.ui_scale);
        // v1 kept the window above every other window by default, which made it
        // impossible to get out of the way; files written before v2 keep the new
        // default instead of the value they were saved with.
        const int version = j.value("version", 1);
        s.always_on_top = version >= 2 ? j.value("always_on_top", s.always_on_top)
                                       : s.always_on_top;
        s.sound_on_done = j.value("sound_on_done", s.sound_on_done);
    } catch (const std::exception &) {
        // corrupted settings: keep the defaults
    }
    return s;
}

bool Settings::save(std::string * err) const {
    try {
        nlohmann::json j;
        j["version"] = 2;
        j["model_path"] = model_path;
        j["n_threads"] = n_threads;
        j["hotkey"] = hotkey;
        j["dark_theme"] = dark_theme;
        j["audio_device"] = audio_device;
        j["output_mode"] = output_mode_name(output_mode);
        j["step_sec"] = step_sec;
        j["max_window_sec"] = max_window_sec;
        j["always_on_top"] = always_on_top;
        j["sound_on_done"] = sound_on_done;
        const std::string p = settings_path();
        std::filesystem::create_directories(std::filesystem::path(p).parent_path());
        std::ofstream f(p);
        if (!f) {
            if (err) *err = "cannot write " + p;
            return false;
        }
        f << j.dump(2) << "\n";
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}

}  // namespace asr::app
