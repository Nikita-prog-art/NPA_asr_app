#pragma once

#include <string>
#include <vector>

namespace asr::app {

// Command line mode.  Returns the process exit code.
int run_cli(const std::vector<std::string> & args);

}  // namespace asr::app
