#include "core/backend_info.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "platform/platform.h"

namespace asr {

namespace {

// The runtime-dispatched CPU backends are separate modules (ggml-cpu-alderlake,
// ggml-cpu-haswell, ...) sitting next to the executable.  ggml scores them at
// startup and registers only the best one as "CPU", so listing the modules is
// what shows the fallback chain that was shipped.
std::vector<std::string> cpu_variant_modules() {
    std::vector<std::string> out;
    std::error_code ec;
    const std::filesystem::path dir(platform::exe_dir());
    if (!std::filesystem::is_directory(dir, ec)) return out;
    for (const auto & e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) continue;
        const std::string name = e.path().filename().string();
        // "ggml-cpu-alderlake.dll" on Windows, "libggml-cpu-alderlake.so" on Linux
        const size_t pos = name.rfind("ggml-cpu-", 0) == 0 ? 0 : (name.rfind("libggml-cpu-", 0) == 0 ? 3 : std::string::npos);
        if (pos == std::string::npos) continue;
        const std::string ext = e.path().extension().string();
        if (ext != ".so" && ext != ".dll" && ext != ".dylib") continue;
        out.push_back(e.path().stem().string().substr(pos));
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

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

    const std::vector<std::string> variants = cpu_variant_modules();
    if (!variants.empty()) {
        out += " | CPU variants: ";
        for (size_t i = 0; i < variants.size(); ++i) {
            if (i) out += ", ";
            out += variants[i];
        }
    }
    return out;
}

}  // namespace asr
