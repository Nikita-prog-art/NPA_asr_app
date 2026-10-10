// Cross-platform services: global push-to-talk hotkey, clipboard, text
// injection, a few filesystem helpers.
//
// Implementations: platform_win.cpp (Windows), platform_linux.cpp (X11/evdev),
// platform_null.cpp (fallback so the core always links).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace asr::platform {

// ------------------------------------------------------------------ hotkey

// A push-to-talk chord: modifiers + one "main" key.
struct Chord {
    bool ctrl  = false;
    bool alt   = false;
    bool shift = false;
    bool win   = false;  // Windows key / Super
    // Canonical key name.  Supported: "Space", "F1".."F24", "A".."Z",
    // "0".."9", "ScrollLock", "Pause", "Insert", "Home", "End",
    // "PageUp", "PageDown", "RightCtrl", "RightAlt", "RightShift".
    std::string key = "Space";

    std::string to_string() const;                     // "Ctrl+Alt+Space"
    static bool parse(const std::string & text, Chord * out, std::string * err);
};

// Observes the chord globally and reports press/release transitions.
// Callbacks may be invoked from an internal thread and must be cheap and
// thread-safe; they must not block.  Press is delivered once per physical
// press (auto-repeat must be filtered), release once.
class HotkeyListener {
public:
    virtual ~HotkeyListener() = default;
    virtual bool start(std::string * err) = 0;
    virtual void stop() = 0;
    virtual const char * backend_name() const = 0;   // "win-hook", "x11-record", "evdev"
};

// Never returns nullptr.  `on_press_fail` reports errors after start().
std::unique_ptr<HotkeyListener> create_hotkey_listener(
    const Chord & chord,
    std::function<void()> on_press,
    std::function<void()> on_release);

// ------------------------------------------------------------------ text

bool clipboard_set_text(const std::string & utf8);
bool clipboard_get_text(std::string * out);

// Pastes the current clipboard content into the focused window of another
// application (Ctrl+V / Cmd+V at the OS level).  The caller is expected to
// have put the text on the clipboard beforehand (or to use inject_text).
bool paste_into_focused_window();

// Convenience: clipboard_set_text(text) followed by paste_into_focused_window().
bool inject_text(const std::string & utf8);

// ------------------------------------------------------------------ tray

// What the user picked in the notification area ("трей").
enum class TrayAction {
    None,
    Show,  // restore the main window
    Quit,  // the menu's exit item
};

// Notification-area icon.  The window close button hides the window while this
// icon keeps the application reachable; the icon restores the window or quits.
//
// The backend runs its own thread (the Windows shell wants a message loop for
// its callbacks), so nothing here may touch the GUI: the user's choice is
// recorded and picked up by the UI thread through take_action().
class TrayIcon {
public:
    virtual ~TrayIcon() = default;
    // Registers the icon.  Returns false (and fills *err) when this desktop has
    // no notification area - the caller then keeps "close" meaning "quit".
    virtual bool add(const std::string & tooltip, std::string * err) = 0;
    virtual void remove() = 0;
    // Pending action, cleared by the call.  UI-thread only.
    virtual TrayAction take_action() = 0;
    // Short description for the log and --info ("windows shell", "none").
    virtual const char * backend_name() const = 0;
};

// Never returns nullptr; it may however report that it cannot add an icon.
std::unique_ptr<TrayIcon> create_tray_icon();

// RGBA pixels (size * size * 4, top-down) of the application icon.  Used for
// the window/taskbar icon by the GUI and by the tray backends, so the icon is
// drawn in code and the repository carries no binary asset.
bool app_icon_rgba(int size, std::vector<unsigned char> * out);

// ------------------------------------------------------------------ misc

std::string exe_dir();
// Full path of the running executable ("" when it cannot be determined).
std::string exe_path();
std::string home_dir();
// Per-user writable directory for models/settings
// (Linux: $XDG_DATA_HOME/npa_asr or ~/.local/share/npa_asr,
//  Windows: %LOCALAPPDATA%\npa_asr).
std::string user_data_dir();
bool        open_in_browser(const std::string & url);
std::string cpu_brand();          // human readable CPU name, may be empty
std::string os_name();            // "Windows 11 (26100.1234)" / "Linux 6.6.87"
void        sleep_ms(int ms);

}  // namespace asr::platform
