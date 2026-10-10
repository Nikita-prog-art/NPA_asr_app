#include "core/backend_info.h"

#include <cstdio>
#include <string>

#include "ggml-backend.h"
#include "platform/platform.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#  include <intrin.h>
#endif

#ifndef ASR_CPU_BASELINE
#  define ASR_CPU_BASELINE "x86-64-v3"
#endif

namespace asr {

namespace {

#if defined(__x86_64__) || defined(_M_X64)
bool x86_cpu_has_avx2() {
#  if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuidex(regs, 1, 0);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx     = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;  // XMM and YMM state enabled
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#  else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#  endif
}
#else
bool x86_cpu_has_avx2() { return true; }  // non-x86 builds need no check
#endif

}  // namespace

std::string cpu_baseline() {
    std::string out = ASR_CPU_BASELINE;
    const std::string base = ASR_CPU_BASELINE;
    if (base == "x86-64-v3") out += " (AVX2/FMA/F16C/BMI2)";
    else if (base == "x86-64-v2") out += " (SSE4.2/AVX)";
    else if (base == "x86-64") out += " (baseline x86-64)";
    else if (base == "native") out += " (tuned for the build machine)";
    return out;
}

bool cpu_supports_baseline() {
    const std::string base = ASR_CPU_BASELINE;
    if (base == "native") return true;  // only meant to run where it was built
    if (base == "x86-64-v3") return x86_cpu_has_avx2();
    return true;
}

std::string backend_summary() {
    ggml_backend_load_all();
    std::string out;
    const size_t nreg = ggml_backend_reg_count();
    for (size_t i = 0; i < nreg; ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (!reg) continue;
        const char * rname = ggml_backend_reg_name(reg);
        const size_t ndev = ggml_backend_reg_dev_count(reg);
        for (size_t d = 0; d < ndev; ++d) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, d);
            if (!dev) continue;
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(dev, &props);
            if (!out.empty()) out += " | ";
            out += rname ? rname : "?";
            if (props.description && *props.description) {
                out += " [";
                out += props.description;
                out += "]";
            }
        }
    }
    if (out.empty()) out = "no ggml backends";

    out += " | compiled for: ";
    out += cpu_baseline();
    return out;
}

}  // namespace asr
