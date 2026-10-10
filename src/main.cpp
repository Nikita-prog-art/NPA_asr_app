// npa_asr - entry point.
//   (no args | --gui | --minimized) : desktop GUI
//   otherwise                       : CLI (see src/app/cli.cpp)
#include <cstdio>
#include <string>
#include <vector>

#include "app/cli.h"
#include "app/gui.h"
#include "core/backend_info.h"

int main(int argc, char ** argv) {
    // A statically linked binary carries exactly one CPU backend, so a machine
    // without the required instructions would die with SIGILL deep inside ggml.
    // Say so up front, together with the way to get a binary that runs here.
    if (!asr::cpu_supports_baseline()) {
        std::fprintf(stderr, "npa_asr: this executable was compiled for %s, which this CPU lacks.\n",
                     asr::cpu_baseline().c_str());
        std::fprintf(stderr, "npa_asr: rebuild with -DASR_CPU_ARCH=x86-64-v2 (or x86-64) for this machine.\n");
        return 2;
    }

    std::vector<std::string> args(argv + 1, argv + argc);

    bool gui = args.empty();
    bool minimized = false;
    for (const auto & a : args) {
        if (a == "--gui") gui = true;
        if (a == "--minimized") {  // used by the autostart entry
            gui = true;
            minimized = true;
        }
    }

    if (gui) {
        return asr::app::run_gui(minimized);
    }
    return asr::app::run_cli(args);
}
