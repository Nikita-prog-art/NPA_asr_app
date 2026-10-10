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
    // Source revision of the installed copy, read out of that file (see
    // build_info.h).  Empty when there is no copy or it carries no marker - for
    // instance when it was installed by a build that predates the marker.
    std::string installed_build_id;
    std::string detail;              // human readable extra info (may be empty)
};

// True when the installed copy comes from a different source revision than this
// process: exactly the case the application offers an update for.  A rebuilt
// copy of the *same* revision is not an update (only its bytes differ).
bool install_update_pending(const InstallStatus & st);

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

// The install prompt is offered when this build has not answered it yet: once
// when the application is not installed at all, and once per new source
// revision afterwards (so a new build - but not a rebuild of the same commit -
// offers to update the installed copy).  The answer is remembered per build in
// <user_data_dir>/install.json, together with the revision that gave it.
bool should_offer_install();
void mark_install_prompt_shown();

}  // namespace asr::app
