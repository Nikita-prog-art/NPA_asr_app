// The application icon, drawn in code (see platform.h).
//
// It lives in its own translation unit so that the icon generator built for
// Windows (tools/make_icon.cpp) can link it without dragging in the rest of the
// platform layer, and it is compiled into the application through the usual
// src/platform/*.cpp glob.
#include "platform/platform.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace asr::platform {

// The application icon: a rounded square in the accent colour of the GUI with
// three white bars (a level meter) on it.  Drawn here instead of shipping an
// .ico/.png so the repository stays free of binary assets and every size is
// rendered at its native resolution (16 px in the notification area, 32 px on
// the taskbar, any size on HiDPI).  The Windows executable icon is produced
// from this function at build time by tools/make_icon.cpp.
bool app_icon_rgba(int size, std::vector<unsigned char> * out) {
    if (out == nullptr || size < 8 || size > 1024) return false;
    const float s = static_cast<float>(size);
    const float radius = std::max(1.0f, s * 0.22f);
    // centre x and height, in fractions of the icon
    const float bars[3][2] = {{0.30f, 0.34f}, {0.50f, 0.64f}, {0.70f, 0.44f}};
    const float bar_w = std::max(1.0f, s * 0.125f);
    const float accent[3] = {66.0f, 150.0f, 250.0f};

    out->assign(static_cast<size_t>(size) * static_cast<size_t>(size) * 4, 0);
    unsigned char * px = out->data();
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const float cx = static_cast<float>(x) + 0.5f;
            const float cy = static_cast<float>(y) + 0.5f;
            // signed distance to the rounded square (negative inside)
            const float qx = std::fabs(cx - s * 0.5f) - (s * 0.5f - radius);
            const float qy = std::fabs(cy - s * 0.5f) - (s * 0.5f - radius);
            const float ax = std::max(qx, 0.0f);
            const float ay = std::max(qy, 0.0f);
            const float d = std::sqrt(ax * ax + ay * ay) + std::min(std::max(qx, qy), 0.0f) - radius;
            const float bg = std::clamp(0.5f - d, 0.0f, 1.0f);  // 1 px antialiased edge
            float glyph = 0.0f;
            for (const auto & b : bars) {
                const float bx = std::fabs(cx - s * b[0]) - bar_w * 0.5f;
                const float by = std::fabs(cy - s * 0.5f) - s * b[1] * 0.5f;
                glyph = std::max(glyph, std::clamp(0.5f - std::max(bx, by), 0.0f, 1.0f));
            }
            unsigned char * p = px + (static_cast<size_t>(y) * static_cast<size_t>(size) +
                                      static_cast<size_t>(x)) * 4;
            for (int c = 0; c < 3; ++c) {
                const float value = accent[c] * (1.0f - glyph) + 255.0f * glyph;
                p[c] = static_cast<unsigned char>(std::lround(value));
            }
            p[3] = static_cast<unsigned char>(std::lround(255.0f * bg));
        }
    }
    return true;
}

}  // namespace asr::platform
