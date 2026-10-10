// Human readable summary of the ggml backends/devices that were found at
// runtime (used by `npa_asr --info` and the GUI's About panel).
#pragma once

#include <string>

namespace asr {

// Describes the available devices, e.g.
//   "CPU: Intel(R) Core(TM) i9-13900HK | baseline: x86-64-v3 (AVX2)"
std::string backend_summary();

// The instruction set baseline this executable was compiled for (the value of
// the ASR_CPU_ARCH CMake option).
std::string cpu_baseline();

// True when the CPU (and the OS) can execute the baseline above.  A statically
// linked binary cannot fall back to another CPU backend at run time, so this is
// what turns an inevitable SIGILL into a readable error message.
bool cpu_supports_baseline();

}  // namespace asr
