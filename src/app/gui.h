#pragma once

namespace asr::app {

// Desktop GUI (ImGui + GLFW).  `minimized` starts the window iconified, which is
// what the autostart entry uses so logging in does not pop a window up.
// Returns the process exit code.
int run_gui(bool minimized = false);

}  // namespace asr::app
