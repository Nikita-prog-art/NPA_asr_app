// npa_asr - entry point.
//   (no args | --gui) : desktop GUI
//   otherwise         : CLI (see src/app/cli.cpp)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "app/cli.h"
#include "app/gui.h"

int main(int argc, char ** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    bool gui = args.empty();
    for (const auto & a : args) {
        if (a == "--gui") gui = true;
    }

    if (gui) {
        return asr::app::run_gui();
    }
    return asr::app::run_cli(args);
}
