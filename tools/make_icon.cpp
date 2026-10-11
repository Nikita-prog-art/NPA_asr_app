// Writes the multi-size Windows icon for the executable's icon resource.
//
// Explorer, the taskbar and Alt-Tab take the application icon from a resource
// inside the .exe, and a resource has to be a file - so the icon that the
// application draws at run time (platform::app_icon_rgba, see
// src/platform/app_icon.cpp) is materialised into an .ico at build time instead
// of being checked in as a binary asset.  The two cannot drift apart.
//
// Usage: npa_icon <output.ico>
#include "platform/platform.h"

#include <cstdio>
#include <vector>

namespace {

// Every size the shell asks for: 16 px in the notification area and the title
// bar, 32/48 px in the taskbar and Explorer's icon views, up to 256 px (which
// the .ico format stores as 0) for extra-large icons.
const int kSizes[] = {16, 24, 32, 48, 64, 128, 256};

void put_u16(std::vector<unsigned char> & out, unsigned value) {
    out.push_back(static_cast<unsigned char>(value & 0xFFu));
    out.push_back(static_cast<unsigned char>((value >> 8) & 0xFFu));
}

void put_u32(std::vector<unsigned char> & out, unsigned value) {
    put_u16(out, value & 0xFFFFu);
    put_u16(out, (value >> 16) & 0xFFFFu);
}

// One image: a BITMAPINFOHEADER, the BGRA pixels bottom-up as Windows wants
// them, and the 1-bit AND mask (all zero - the alpha channel decides).
std::vector<unsigned char> make_image(int size) {
    std::vector<unsigned char> rgba;
    if (!asr::platform::app_icon_rgba(size, &rgba)) return {};

    const size_t pixels = static_cast<size_t>(size) * static_cast<size_t>(size);
    const int mask_stride = ((size + 31) / 32) * 4;  // 1 bpp rows padded to 4 bytes

    std::vector<unsigned char> image;
    image.reserve(40 + pixels * 4 + static_cast<size_t>(mask_stride) * size);

    put_u32(image, 40);                                         // biSize
    put_u32(image, static_cast<unsigned>(size));                // biWidth
    put_u32(image, static_cast<unsigned>(size) * 2u);           // biHeight: XOR + AND
    put_u16(image, 1);                                          // biPlanes
    put_u16(image, 32);                                         // biBitCount
    put_u32(image, 0);                                          // biCompression: BI_RGB
    put_u32(image, static_cast<unsigned>(pixels * 4 +
                                         static_cast<size_t>(mask_stride) * size));  // biSizeImage
    put_u32(image, 0);                                          // biXPelsPerMeter
    put_u32(image, 0);                                          // biYPelsPerMeter
    put_u32(image, 0);                                          // biClrUsed
    put_u32(image, 0);                                          // biClrImportant

    for (int y = size - 1; y >= 0; --y) {
        const unsigned char * row = rgba.data() + static_cast<size_t>(y) * size * 4;
        for (int x = 0; x < size; ++x) {
            image.push_back(row[x * 4 + 2]);  // blue
            image.push_back(row[x * 4 + 1]);  // green
            image.push_back(row[x * 4 + 0]);  // red
            image.push_back(row[x * 4 + 3]);  // alpha
        }
    }
    image.insert(image.end(), static_cast<size_t>(mask_stride) * size, 0);
    return image;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: npa_icon <output.ico>\n");
        return 2;
    }

    std::vector<std::vector<unsigned char>> images;
    for (int size : kSizes) {
        std::vector<unsigned char> image = make_image(size);
        if (image.empty()) {
            std::fprintf(stderr, "npa_icon: cannot render the %d px icon\n", size);
            return 1;
        }
        images.push_back(std::move(image));
    }

    std::vector<unsigned char> ico;
    put_u16(ico, 0);  // reserved
    put_u16(ico, 1);  // resource type: icon
    put_u16(ico, static_cast<unsigned>(images.size()));

    unsigned offset = 6 + 16 * static_cast<unsigned>(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        const int size = kSizes[i];
        const unsigned char dimension = size >= 256 ? 0 : static_cast<unsigned char>(size);
        ico.push_back(dimension);                                  // width (0 = 256)
        ico.push_back(dimension);                                  // height
        ico.push_back(0);                                          // palette entries
        ico.push_back(0);                                          // reserved
        put_u16(ico, 1);                                           // colour planes
        put_u16(ico, 32);                                          // bits per pixel
        put_u32(ico, static_cast<unsigned>(images[i].size()));      // bytes of image data
        put_u32(ico, offset);                                      // offset of image data
        offset += static_cast<unsigned>(images[i].size());
    }
    for (const std::vector<unsigned char> & image : images) {
        ico.insert(ico.end(), image.begin(), image.end());
    }

    std::FILE * file = std::fopen(argv[1], "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "npa_icon: cannot open %s for writing\n", argv[1]);
        return 1;
    }
    const size_t written = std::fwrite(ico.data(), 1, ico.size(), file);
    const bool closed = std::fclose(file) == 0;
    if (written != ico.size() || !closed) {
        std::fprintf(stderr, "npa_icon: cannot write %s\n", argv[1]);
        return 1;
    }

    std::printf("npa_icon: %s (%zu bytes, sizes", argv[1], ico.size());
    for (int size : kSizes) std::printf(" %d", size);
    std::printf(")\n");
    return 0;
}
