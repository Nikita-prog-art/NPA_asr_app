// Fallback platform layer for systems that have no dedicated implementation
// (neither Windows, Linux nor Apple).  Everything degrades gracefully so the
// core library still links; the hotkey simply never fires.
#if !defined(_WIN32) && !defined(__linux__) && !defined(__APPLE__)

#  include "platform/platform.h"

#  include <memory>
#  include <string>
#  include <utility>

namespace asr::platform {
namespace {

class NullHotkeyListener final : public HotkeyListener {
public:
    bool start(std::string * err) override {
        if (err != nullptr) *err = "unsupported platform";
        return false;
    }
    void stop() override {}
    const char * backend_name() const override { return "null"; }
};

}  // namespace

std::unique_ptr<HotkeyListener> create_hotkey_listener(const Chord &,
                                                       std::function<void()>,
                                                       std::function<void()>) {
    return std::make_unique<NullHotkeyListener>();
}

bool clipboard_set_text(const std::string &) { return false; }

bool clipboard_get_text(std::string *) { return false; }

bool paste_into_focused_window() { return false; }

std::string user_data_dir() { return std::string(); }

bool open_in_browser(const std::string &) { return false; }

std::string cpu_brand() { return std::string(); }

std::string os_name() { return std::string(); }

}  // namespace asr::platform

#endif  // !_WIN32 && !__linux__ && !__APPLE__
