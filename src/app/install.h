// Per-user "installation" of the application and autostart management.
//
// The executable is a single self-contained file, so installing means copying
// that one file into a per-user directory and (optionally) registering it for
// autostart:
//
//   Windows: %LOCALAPPDATA%\Programs\NPA ASR\npa_asr.exe
//            HKCU\Software\Microsoft\Windows\CurrentVersion\Run  ("NPA ASR")
//   Linux:   ~/.local/bin/npa_asr
//            ~/.config/autostart/npa-asr.desktop
//            ~/.local/share/applications/npa-asr.desktop   (menu entry)
//
// Settings and models always stay in platform::user_data_dir(), so installing,
// uninstalling or running a different copy never loses them.
#pragma once

#include <string>

namespace asr::app {

struct InstallStatus {
    bool installed = false;     // a copy of the executable exists in install_dir
    bool up_to_date = false;    // ... and it is the same build as this process
    bool autostart = false;     // the autostart entry is registered
    bool running_installed = false;  // this process *is* the installed copy
    std::string install_dir;         // directory the app installs itself into
    std::string install_exe;         // <install_dir>/npa_asr[.exe]
    std::string exe_path;            // executable of the running process
    std::string autostart_target;    // registry value / .desktop file path
    std::string detail;              // human readable extra info (may be empty)
};

// Never fails; an unreadable state simply reports false flags.
InstallStatus install_status();

// Copies the running executable into install_dir (replacing an older copy) and,
// when `autostart` is true, registers the autostart entry.  When
// `autostart_only` is true the copy step is skipped.  Returns false and fills
// *err on failure.
bool install_app(bool autostart, bool autostart_only, std::string * err);

// Removes the autostart entry and, unless `keep_files`, the installed copy.
// Settings and models are never touched.
bool uninstall_app(bool keep_files, std::string * err);

// Registers/unregisters autostart without touching the installed copy.
bool set_autostart(bool on, std::string * err);

// The command line the autostart entry uses (exposed for tests/diagnostics).
std::string autostart_command();

// The application should offer the "install" prompt exactly once: the flag is
// stored in <user_data_dir>/install.json, and the prompt is not offered at all
// when the process already runs from the install directory.
bool should_offer_install();
void mark_install_prompt_shown();

}  // namespace asr::app
