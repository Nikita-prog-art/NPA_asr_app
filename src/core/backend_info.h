// Human readable summary of the ggml backends/devices that were found at
// runtime (used by `npa_asr --info` and the GUI's About panel).
#pragma once

#include <string>

namespace asr {

// Loads all ggml backends and describes the available devices, e.g.
//   "CPU: ggml-cpu-alderlake | Intel(R) Core(TM) i9-13900HK"
std::string backend_summary();

}  // namespace asr
