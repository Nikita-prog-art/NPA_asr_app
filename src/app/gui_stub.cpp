#include "app/gui.h"

#include <cstdio>

namespace asr::app {

#if !ASR_HAVE_GUI
int run_gui(bool minimized) {
    (void) minimized;
    std::fprintf(stderr, "npa_asr: built without GUI support (-DASR_GUI=OFF)\n");
    return 2;
}
#endif

}  // namespace asr::app
