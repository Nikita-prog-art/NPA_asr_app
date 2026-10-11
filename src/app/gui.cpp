// Desktop GUI for npa_asr (Dear ImGui + GLFW + OpenGL3).
//
// The window is a thin shell around asr::app::Controller: the controller owns
// every piece of application logic (hotkey, audio, recogniser, model loading)
// and this file renders a per-frame snapshot of its state plus the configuration
// that is persisted through asr::app::Settings.
//
// Threading rules honoured here:
//   * status()/devices()/take_log() are UI-thread calls, once per frame;
//   * press_ptt()/release_ptt() are safe from the UI thread;
//   * network I/O (manifest, downloads, model loads) never blocks the UI thread.
#include "app/gui.h"

#if ASR_HAVE_GUI

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include "app/build_info.h"
#include "app/controller.h"
#include "app/install.h"
#include "app/models.h"
#include "app/settings.h"
#include "core/backend_info.h"
#include "net/manifest.h"
#include "platform/platform.h"

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace asr::app {

namespace {

namespace fs = std::filesystem;

// ------------------------------------------------------------------ constants

// GLFW's always-on-top window attribute (GLFW_ALWAYS_ON_TOP in the task
// description).  GLFW_FLOATING maps to the topmost window flag on Windows and to
// _NET_WM_STATE_ABOVE on X11; window managers that do not know it simply ignore
// the hint, which keeps this harmless on every platform.
constexpr int kAlwaysOnTopAttrib = GLFW_FLOATING;

const ImVec4 kRecordingRed(1.00f, 0.36f, 0.36f, 1.00f);
const ImVec4 kRecordingBg(0.34f, 0.09f, 0.10f, 1.00f);
const ImVec4 kWarnYellow(0.95f, 0.75f, 0.25f, 1.00f);
const ImVec4 kDimText(0.62f, 0.65f, 0.70f, 1.00f);

// ------------------------------------------------------------------ helpers

// The application manifest asks Windows for Per-Monitor-V2 DPI awareness, but
// the request is repeated here: it is a no-op when the manifest already did it,
// and it keeps the window crisp on toolchains that do not compile the resource
// file.  Without it Windows bitmap-stretches the whole window at 125% and the
// content scale GLFW reports stays 1.0.
void ensure_dpi_awareness() {
#if defined(_WIN32)
    using set_ctx_fn = BOOL(WINAPI *)(HANDLE);
    using set_aware_fn = BOOL(WINAPI *)(void);
    HMODULE user32 = ::GetModuleHandleW(L"user32.dll");
    if (user32 == nullptr) return;
    auto set_ctx = reinterpret_cast<set_ctx_fn>(
        reinterpret_cast<void *>(::GetProcAddress(user32, "SetProcessDpiAwarenessContext")));
    if (set_ctx != nullptr) {
        // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (HANDLE) -4
        set_ctx(reinterpret_cast<HANDLE>(static_cast<intptr_t>(-4)));
        return;
    }
    auto set_aware = reinterpret_cast<set_aware_fn>(
        reinterpret_cast<void *>(::GetProcAddress(user32, "SetProcessDPIAware")));
    if (set_aware != nullptr) set_aware();
#endif
}

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

float clamp_scale(float scale) {
    // 2x is already a very large interface; beyond that the fixed two-panel
    // layout no longer fits on any screen.
    return std::max(0.75f, std::min(2.0f, scale));
}

// The scale the window's monitor asks for (1.25 at "125%" on Windows, usually
// 1.0 on X11 unless Xft.dpi says otherwise).
float monitor_scale(GLFWwindow * window) {
    float sx = 1.0f;
    float sy = 1.0f;
    if (window != nullptr) {
        glfwGetWindowContentScale(window, &sx, &sy);
    } else if (GLFWmonitor * primary = glfwGetPrimaryMonitor()) {
        glfwGetMonitorContentScale(primary, &sx, &sy);
    }
    if (!(sx > 0.0f)) sx = 1.0f;  // some accessibility tools report 0.0
    return clamp_scale(sx);
}

std::string lower_copy(std::string s) {
    for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool has_extension(const std::string & path, const char * ext) {
    const std::string e = lower_copy(ext);
    if (path.size() < e.size()) return false;
    return lower_copy(path.substr(path.size() - e.size())) == e;
}

std::string file_name(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string format_mb(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f МБ", static_cast<double>(bytes) / 1e6);
    return buf;
}

// ------------------------------------------------------------------ fonts

#if defined(_WIN32)
const char * const kRegularFonts[] = {
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
};
const char * const kItalicFonts[] = {
    "C:/Windows/Fonts/segoeuii.ttf",
    "C:/Windows/Fonts/ariali.ttf",
};
const char * const kMonoFonts[] = {
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
};
#else
const char * const kRegularFonts[] = {
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
};
const char * const kItalicFonts[] = {
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf",
};
const char * const kMonoFonts[] = {
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
};
#endif

// Returns the first candidate that exists on disk, or an empty string.
std::string first_existing_font(const char * const * paths, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        std::error_code ec;
        if (fs::is_regular_file(paths[i], ec) && !ec) return paths[i];
    }
    return std::string();
}

// Latin + Cyrillic (the labels and the recognised text are Russian) plus the
// punctuation and symbols the interface uses: ImGui renders a missing glyph as
// '?', so the ranges have to cover what the code actually prints.
const ImWchar * ui_glyph_ranges() {
    static const ImWchar ranges[] = {
        0x0020, 0x00FF,  // Basic Latin + Latin Supplement
        0x0400, 0x052F,  // Cyrillic + Cyrillic Supplement
        0x2010, 0x205E,  // General Punctuation (dashes, quotes, ellipsis)
        0x2116, 0x2117,  // №
        0x2190, 0x21FF,  // Arrows
        0x25A0, 0x25FF,  // Geometric Shapes (● ○ ■)
        0x20AC, 0x20AC,  // €
        0,
    };
    return ranges;
}

// Adds the first available font of a candidate list.  Returns nullptr when none
// of the files can be loaded (the caller then keeps the previous font).
ImFont * add_cyrillic_font(ImFontAtlas * atlas, const char * const * paths, size_t count,
                           float size_px) {
    const std::string path = first_existing_font(paths, count);
    if (path.empty()) return nullptr;
    return atlas->AddFontFromFileTTF(path.c_str(), size_px, nullptr, ui_glyph_ranges());
}

// ------------------------------------------------------------------ theme

ImVec4 accent_color(bool dark) {
    return dark ? ImVec4(0.26f, 0.59f, 0.98f, 1.00f) : ImVec4(0.11f, 0.42f, 0.85f, 1.00f);
}

// Real dark / light themes with a shared accent colour and slightly rounded
// frames.  Called once on the first frame and whenever the checkbox toggles.
void apply_theme(bool dark) {
    ImGuiStyle & s = ImGui::GetStyle();
    if (dark) {
        ImGui::StyleColorsDark();
    } else {
        ImGui::StyleColorsLight();
    }

    s.WindowRounding = 6.0f;
    s.ChildRounding = 6.0f;
    s.FrameRounding = 5.0f;
    s.PopupRounding = 6.0f;
    s.GrabRounding = 5.0f;
    s.ScrollbarRounding = 8.0f;
    s.TabRounding = 5.0f;
    s.WindowBorderSize = 1.0f;
    s.FrameBorderSize = dark ? 0.0f : 1.0f;
    s.WindowPadding = ImVec2(10.0f, 10.0f);
    s.FramePadding = ImVec2(8.0f, 5.0f);
    s.ItemSpacing = ImVec2(8.0f, 6.0f);

    const ImVec4 a = accent_color(dark);
    const ImVec4 a_hover(a.x + 0.10f, a.y + 0.08f, a.z + 0.04f, 1.00f);
    const ImVec4 a_soft(a.x, a.y, a.z, dark ? 0.55f : 0.32f);
    const ImVec4 a_soft_hover(a.x, a.y, a.z, dark ? 0.80f : 0.52f);

    s.Colors[ImGuiCol_Button] = a_soft;
    s.Colors[ImGuiCol_ButtonHovered] = a_soft_hover;
    s.Colors[ImGuiCol_ButtonActive] = a;
    s.Colors[ImGuiCol_Header] = a_soft;
    s.Colors[ImGuiCol_HeaderHovered] = a_soft_hover;
    s.Colors[ImGuiCol_HeaderActive] = a;
    s.Colors[ImGuiCol_CheckMark] = a;
    s.Colors[ImGuiCol_CheckboxSelectedBg] = a_soft;
    s.Colors[ImGuiCol_SliderGrab] = a;
    s.Colors[ImGuiCol_SliderGrabActive] = a_hover;
    s.Colors[ImGuiCol_TextLink] = a;
    s.Colors[ImGuiCol_TextSelectedBg] = ImVec4(a.x, a.y, a.z, 0.35f);
    s.Colors[ImGuiCol_PlotHistogram] = a;
    s.Colors[ImGuiCol_Separator] = ImVec4(a.x, a.y, a.z, dark ? 0.35f : 0.28f);
    s.Colors[ImGuiCol_TabHovered] = a_soft_hover;
    s.Colors[ImGuiCol_TabSelected] = ImVec4(a.x, a.y, a.z, dark ? 0.42f : 0.26f);
    s.Colors[ImGuiCol_TabSelectedOverline] = a;
    s.Colors[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(a.x, a.y, a.z, 0.45f);
    s.Colors[ImGuiCol_ResizeGrip] = a_soft;
    s.Colors[ImGuiCol_ResizeGripHovered] = a_soft_hover;
    s.Colors[ImGuiCol_ResizeGripActive] = a;
    if (dark) {
        s.Colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.11f, 0.13f, 1.00f);
        s.Colors[ImGuiCol_ChildBg] = ImVec4(0.13f, 0.14f, 0.16f, 1.00f);
        s.Colors[ImGuiCol_PopupBg] = ImVec4(0.13f, 0.14f, 0.16f, 0.98f);
        s.Colors[ImGuiCol_Border] = ImVec4(0.28f, 0.30f, 0.34f, 0.70f);
    } else {
        s.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.95f, 1.00f);
        s.Colors[ImGuiCol_ChildBg] = ImVec4(0.98f, 0.98f, 0.99f, 1.00f);
        s.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.98f);
        s.Colors[ImGuiCol_Border] = ImVec4(0.65f, 0.66f, 0.70f, 0.70f);
    }
}

// Applies the theme and then the interface scale.
//
// ScaleAllSizes() multiplies the style *in place*, and apply_theme() only
// restores the colours plus a handful of sizes, so the style has to be reset to
// the library defaults first: scaling an already scaled style compounds and the
// interface grows on every theme toggle or Ctrl +/- press.
void apply_style(bool dark, float scale) {
    ImGuiStyle & s = ImGui::GetStyle();
    s = ImGuiStyle{};
    apply_theme(dark);
    if (!(scale > 0.0f)) return;
    s.ScaleAllSizes(scale);
    // ImGui 1.92 rasterises fonts on demand at FontSizeBase * FontScaleDpi, so
    // the glyphs stay crisp instead of being scaled up as a texture.
    s.FontScaleDpi = scale;
}

// Keeps the layout space constant when the interface scale changes: the window
// grows (or shrinks) with it, up to the work area of its monitor.  Without this
// the scaled widgets do not fit into the window they were laid out for.
void resize_window_for_scale(GLFWwindow * window, float from, float to) {
    if (window == nullptr || !(from > 0.0f) || !(to > 0.0f)) return;
    int w = 0;
    int h = 0;
    glfwGetWindowSize(window, &w, &h);
    if (w <= 0 || h <= 0) return;

    const double f = static_cast<double>(to) / static_cast<double>(from);
    int nw = static_cast<int>(std::lround(w * f));
    int nh = static_cast<int>(std::lround(h * f));

    GLFWmonitor * mon = glfwGetWindowMonitor(window);
    if (mon == nullptr) mon = glfwGetPrimaryMonitor();
    int mx = 0, my = 0, mw = 0, mh = 0;
    if (mon != nullptr) {
        glfwGetMonitorWorkarea(mon, &mx, &my, &mw, &mh);
        if (mw > 0 && nw > mw) nw = mw;
        if (mh > 0 && nh > mh) nh = mh;
    }
    glfwSetWindowSize(window, nw, nh);
    // the minimum size is in screen coordinates, i.e. it has to follow the scale
    glfwSetWindowSizeLimits(window, static_cast<int>(std::lround(720.0f * to)),
                            static_cast<int>(std::lround(480.0f * to)),
                            GLFW_DONT_CARE, GLFW_DONT_CARE);
}

// ------------------------------------------------------------------ state

// Result of a background fetch_manifest() call.  Kept alive by shared_ptr so the
// worker never touches GUI memory directly.
struct ManifestFetch {
    std::mutex mutex;
    bool done = false;
    bool ok = false;
    std::string error;
    net::Manifest manifest;
};

struct GuiState {
    // fonts
    ImFont * font_regular = nullptr;
    ImFont * font_italic = nullptr;
    ImFont * font_mono = nullptr;

    // settings mirror (the controller gets copies through apply_settings())
    Settings settings;
    bool settings_dirty = false;
    double settings_changed_at = 0.0;
    bool always_on_top = true;
    bool theme_applied = false;
    bool ptt_mouse_held = false;

    // interface scale: the value actually applied, the monitor's suggestion and
    // a pending change requested by Ctrl +/- or the settings tab
    float ui_scale = 1.0f;
    float monitor_scale_now = 1.0f;
    bool scale_pending = false;
    std::string scale_toast;
    double scale_toast_until = 0.0;

    // per-user install / autostart
    InstallStatus install;
    bool install_checked = false;
    bool install_offer = false;      // first-run prompt
    bool install_autostart = true;   // checkbox in the prompt
    bool install_popup_opened = false;
    std::string install_message;
    std::string install_error;

    // hotkey capture
    bool capturing = false;
    char hotkey_buf[64] = {};
    std::string hotkey_status;

    // notification area: with an icon the close button hides the window and the
    // application keeps listening for the hotkey
    std::unique_ptr<platform::TrayIcon> tray;
    bool tray_active = false;
    bool quit_requested = false;
    std::string tray_status;

    // transcript
    std::string transcript;
    std::string last_final;
    std::string copy_status;
    std::vector<std::string> log_lines;

    // local models
    std::string models_dir;
    char path_buf[1024] = {};
    std::string models_status;
    std::vector<LocalModel> local_models;
    bool models_need_scan = true;

    // built-in file browser
    bool browser_open = false;
    bool browser_refresh = false;
    std::string browser_dir;
    std::string browser_selected;
    std::string browser_error;
    std::vector<std::string> browser_dirs;
    std::vector<std::string> browser_files;

    // model catalogue (fetch_manifest on a worker thread)
    bool manifest_loading = false;
    bool manifest_loaded = false;
    std::string manifest_error;
    net::Manifest manifest;
    std::shared_ptr<ManifestFetch> fetch;
    std::thread fetch_thread;

    // a .gguf dropped onto the window, loaded on the next frame
    std::string pending_load;

    // download
    std::shared_ptr<DownloadJob> job;
    std::string job_label;

    // cached about-panel information (backend_summary() is not free)
    bool info_cached = false;
    std::string backend_info;
    std::string cpu_info;
    std::string os_info;
};

void touch_settings(GuiState & ui) {
    ui.settings_dirty = true;
    ui.settings_changed_at = now_seconds();
}

// ------------------------------------------------------------------ hotkey

// Translates a GLFW key + modifier mask into the platform chord spelling
// ("Ctrl+Alt+Space").  Returns false for keys that cannot carry a chord.
bool glfw_key_to_chord(int key, int mods, std::string * out) {
    std::string name;
    bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
    bool alt = (mods & GLFW_MOD_ALT) != 0;
    bool shift = (mods & GLFW_MOD_SHIFT) != 0;
    bool super = (mods & GLFW_MOD_SUPER) != 0;

    if (key == GLFW_KEY_RIGHT_CONTROL) {
        name = "RightCtrl";
        ctrl = false;
    } else if (key == GLFW_KEY_RIGHT_ALT) {
        name = "RightAlt";
        alt = false;
    } else if (key == GLFW_KEY_RIGHT_SHIFT) {
        name = "RightShift";
        shift = false;
    } else if (key == GLFW_KEY_SPACE) {
        name = "Space";
    } else if (key == GLFW_KEY_SCROLL_LOCK) {
        name = "ScrollLock";
    } else if (key == GLFW_KEY_PAUSE) {
        name = "Pause";
    } else if (key == GLFW_KEY_INSERT) {
        name = "Insert";
    } else if (key == GLFW_KEY_HOME) {
        name = "Home";
    } else if (key == GLFW_KEY_END) {
        name = "End";
    } else if (key == GLFW_KEY_PAGE_UP) {
        name = "PageUp";
    } else if (key == GLFW_KEY_PAGE_DOWN) {
        name = "PageDown";
    } else if (key >= GLFW_KEY_F1 && key <= GLFW_KEY_F24) {
        name = "F" + std::to_string(key - GLFW_KEY_F1 + 1);
    } else if (key >= GLFW_KEY_A && key <= GLFW_KEY_Z) {
        name = std::string(1, static_cast<char>('A' + (key - GLFW_KEY_A)));
    } else if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9) {
        name = std::string(1, static_cast<char>('0' + (key - GLFW_KEY_0)));
    } else {
        return false;  // modifiers alone, Escape, punctuation: unusable
    }

    platform::Chord chord;
    chord.ctrl = ctrl;
    chord.alt = alt;
    chord.shift = shift;
    chord.win = super;
    chord.key = name;
    *out = chord.to_string();
    return true;
}

// Ctrl + / Ctrl - / Ctrl 0 change the interface scale.  This is deliberately
// handled before the hotkey capture so the shortcuts keep working while a text
// field has focus and while a new chord is being recorded.
void handle_scale_shortcut(GuiState & ui, int key) {
    bool reset = false;
    float step = 0.0f;
    if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
        step = 0.05f;
    } else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
        step = -0.05f;
    } else if (key == GLFW_KEY_0 || key == GLFW_KEY_KP_0) {
        reset = true;
    } else {
        return;
    }

    if (reset) {
        ui.settings.ui_scale = 0.0f;  // back to the monitor's own scale
    } else {
        const float base = ui.settings.ui_scale > 0.0f ? ui.settings.ui_scale : ui.ui_scale;
        const float next = clamp_scale(base + step);
        ui.settings.ui_scale = std::round(next * 20.0f) / 20.0f;  // 5% steps
    }
    ui.scale_pending = true;
    touch_settings(ui);
}

// Installed before ImGui_ImplGlfw_InitForOpenGL() so that the backend chains
// into it (it stores and calls the previous GLFW key callback).
void glfw_key_callback(GLFWwindow * window, int key, int scancode, int action, int mods) {
    (void) scancode;
    GuiState * ui = static_cast<GuiState *>(glfwGetWindowUserPointer(window));
    if (ui == nullptr) return;

    if (action == GLFW_PRESS && (mods & GLFW_MOD_CONTROL) != 0) {
        if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD || key == GLFW_KEY_MINUS ||
            key == GLFW_KEY_KP_SUBTRACT || key == GLFW_KEY_0 || key == GLFW_KEY_KP_0) {
            handle_scale_shortcut(*ui, key);
            return;
        }
    }

    if (!ui->capturing) return;
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_ESCAPE) {
        ui->capturing = false;
        ui->hotkey_status = "запись сочетания отменена";
        return;
    }

    std::string chord;
    if (!glfw_key_to_chord(key, mods, &chord)) return;  // wait for the main key

    std::snprintf(ui->hotkey_buf, sizeof(ui->hotkey_buf), "%s", chord.c_str());
    ui->settings.hotkey = chord;
    ui->capturing = false;
    ui->hotkey_status = "новое сочетание: " + chord;
    touch_settings(*ui);
}

// Drag & drop: loads the first dropped .gguf (processed on the next frame).
void glfw_drop_callback(GLFWwindow * window, int count, const char ** paths) {
    GuiState * ui = static_cast<GuiState *>(glfwGetWindowUserPointer(window));
    if (ui == nullptr) return;
    for (int i = 0; i < count; ++i) {
        const std::string path = paths[i] != nullptr ? paths[i] : std::string();
        if (path.empty()) continue;
        if (!has_extension(path, ".gguf")) {
            ui->models_status = "перетащите файл .gguf: " + file_name(path);
            continue;
        }
        std::snprintf(ui->path_buf, sizeof(ui->path_buf), "%s", path.c_str());
        ui->settings.model_path = path;
        ui->models_status = "загрузка перетащенной модели: " + file_name(path);
        ui->pending_load = path;  // picked up by poll_background() on the next frame
    }
}

// ------------------------------------------------------------------ backend

void glfw_error_callback(int code, const char * description) {
    std::fprintf(stderr, "npa_asr: GLFW error %d: %s\n", code, description != nullptr ? description : "?");
}

void set_always_on_top(GLFWwindow * window, bool on) {
    if (window == nullptr) return;
    const int wanted = on ? GLFW_TRUE : GLFW_FALSE;
    if (glfwGetWindowAttrib(window, kAlwaysOnTopAttrib) == wanted) return;
    glfwSetWindowAttrib(window, kAlwaysOnTopAttrib, wanted);
}

// ------------------------------------------------------------------ tray

// Brings the window back from the notification area.  Works for both states the
// user can leave it in: hidden ("крестик") and minimised (taskbar button).
void show_main_window(GLFWwindow * window) {
    if (window == nullptr) return;
    if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) glfwRestoreWindow(window);
    glfwShowWindow(window);
    glfwFocusWindow(window);
}

// The close button: hide instead of quit while the tray icon is there.  The
// controller keeps running, so the push-to-talk hotkey still works and the tray
// icon is the way back.
void glfw_close_callback(GLFWwindow * window) {
    auto * ui = static_cast<GuiState *>(glfwGetWindowUserPointer(window));
    if (ui == nullptr || !ui->tray_active) return;  // no tray: really close
    glfwSetWindowShouldClose(window, GLFW_FALSE);
    glfwHideWindow(window);
    ui->tray_status = "приложение свёрнуто в трей — значок рядом с часами";
}

// Picks up what the user chose in the notification area.  Called once per frame
// from the UI thread; the tray backend only records the choice.
void poll_tray(GuiState & ui, GLFWwindow * window) {
    if (!ui.tray) return;
    switch (ui.tray->take_action()) {
        case platform::TrayAction::Show:
            show_main_window(window);
            ui.tray_status.clear();
            break;
        case platform::TrayAction::Quit:
            ui.quit_requested = true;
            break;
        default:
            break;
    }
}

// ------------------------------------------------------------------ model load

void request_load(GuiState & ui, Controller & controller, const std::string & path) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) {
        ui.models_status = "файл не найден: " + path;
        return;
    }
    ui.settings.model_path = path;
    std::snprintf(ui.path_buf, sizeof(ui.path_buf), "%s", path.c_str());
    ui.models_status = "загрузка модели: " + file_name(path);
    controller.request_model_load(path);  // asynchronous, progress via status()
    touch_settings(ui);
}

void begin_download(GuiState & ui, const std::string & url, const std::string & sha256,
                    const std::string & label) {
    std::error_code ec;
    fs::create_directories(ui.models_dir, ec);
    const std::string dest = (fs::path(ui.models_dir) / file_name(url)).string();
    ui.job = start_download(url, dest, sha256);
    ui.job_label = label + " → " + file_name(url);
    ui.models_status = "загрузка: " + ui.job_label;
    ui.models_need_scan = true;
}

void start_manifest_fetch(GuiState & ui) {
    if (ui.manifest_loading) return;
    if (ui.fetch_thread.joinable()) ui.fetch_thread.join();
    ui.fetch = std::make_shared<ManifestFetch>();
    const std::shared_ptr<ManifestFetch> result = ui.fetch;
    ui.manifest_loading = true;
    ui.manifest_error.clear();
    // fetch_manifest() does blocking network I/O: keep it off the UI thread and
    // hand the parsed catalogue back through the shared result + mutex.
    ui.fetch_thread = std::thread([result]() {
        net::Manifest manifest;
        std::string err;
        const bool ok = fetch_manifest(&manifest, &err);
        std::lock_guard<std::mutex> lock(result->mutex);
        result->ok = ok;
        result->error = err;
        if (ok) result->manifest = std::move(manifest);
        result->done = true;  // published last, read under the same mutex
    });
}

// Moves a finished background fetch into the GUI state.
void poll_manifest(GuiState & ui) {
    if (!ui.manifest_loading || !ui.fetch) return;
    bool done = false;
    {
        std::lock_guard<std::mutex> lock(ui.fetch->mutex);
        done = ui.fetch->done;
    }
    if (!done) return;
    if (ui.fetch_thread.joinable()) ui.fetch_thread.join();
    ui.manifest_loading = false;
    std::lock_guard<std::mutex> lock(ui.fetch->mutex);
    if (ui.fetch->ok) {
        ui.manifest = ui.fetch->manifest;
        ui.manifest_loaded = true;
        ui.manifest_error.clear();
    } else {
        ui.manifest_error = ui.fetch->error.empty() ? "не удалось получить каталог моделей"
                                                    : ui.fetch->error;
    }
}

// ------------------------------------------------------------------ file browser

void refresh_browser(GuiState & ui) {
    ui.browser_dirs.clear();
    ui.browser_files.clear();
    ui.browser_error.clear();
    ui.browser_refresh = false;

    std::error_code ec;
    const fs::path dir(ui.browser_dir);
    if (ui.browser_dir.empty() || !fs::is_directory(dir, ec)) {
        ui.browser_error = "каталог недоступен: " + ui.browser_dir;
        return;
    }
    for (fs::directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec)) {
        const bool is_dir = it->is_directory(ec);
        if (ec) {
            ec.clear();
            continue;
        }
        const std::string name = it->path().filename().string();
        if (!name.empty() && name[0] == '.') continue;  // hide dot files
        if (is_dir) {
            ui.browser_dirs.push_back(it->path().string());
        } else {
            const bool is_file = it->is_regular_file(ec);
            if (ec) ec.clear();
            if (is_file && has_extension(name, ".gguf")) ui.browser_files.push_back(it->path().string());
        }
    }
    std::sort(ui.browser_dirs.begin(), ui.browser_dirs.end());
    std::sort(ui.browser_files.begin(), ui.browser_files.end());
}

void draw_file_browser(GuiState & ui) {
    if (ui.browser_open) {
        ui.browser_open = false;
        ui.browser_refresh = true;
        ImGui::OpenPopup("Выбор файла модели");
    }
    ImGui::SetNextWindowSize(ImVec2(700.0f, 480.0f), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("Выбор файла модели", nullptr, ImGuiWindowFlags_NoSavedSettings)) return;

    if (ui.browser_refresh) refresh_browser(ui);

    if (ImGui::Button("Вверх")) {
        const fs::path current(ui.browser_dir);
        if (current.has_parent_path() && current.parent_path() != current) {
            ui.browser_dir = current.parent_path().string();
            ui.browser_refresh = true;
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", ui.browser_dir.c_str());
    if (!ui.browser_error.empty()) ImGui::TextColored(kWarnYellow, "%s", ui.browser_error.c_str());
    ImGui::Separator();

    const float footer = ImGui::GetFrameHeightWithSpacing() + 6.0f;
    ImGui::BeginChild("browser_list", ImVec2(0.0f, -footer), ImGuiChildFlags_Borders);
    for (const std::string & dir : ui.browser_dirs) {
        const std::string label = "[каталог]  " + file_name(dir);
        ImGui::PushID(dir.c_str());
        ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick);
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            ui.browser_dir = dir;
            ui.browser_refresh = true;
        }
        ImGui::PopID();
    }
    for (const std::string & file : ui.browser_files) {
        const bool selected = (ui.browser_selected == file);
        ImGui::PushID(file.c_str());
        ImGui::Selectable(file_name(file).c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick);
        if (ImGui::IsItemClicked()) ui.browser_selected = file;
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            ui.browser_selected = file;
            std::snprintf(ui.path_buf, sizeof(ui.path_buf), "%s", file.c_str());
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopID();
    }
    if (ui.browser_dirs.empty() && ui.browser_files.empty() && ui.browser_error.empty()) {
        ImGui::TextDisabled("нет подкаталогов и файлов .gguf");
    }
    ImGui::EndChild();

    ImGui::BeginDisabled(ui.browser_selected.empty());
    if (ImGui::Button("Выбрать")) {
        std::snprintf(ui.path_buf, sizeof(ui.path_buf), "%s", ui.browser_selected.c_str());
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Отмена")) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

// ------------------------------------------------------------------ panels

void draw_main_panel(GuiState & ui, Controller & controller, GLFWwindow * window,
                     const AppStatus & st) {
    (void) window;

    ImGui::TextColored(accent_color(ui.settings.dark_theme), "NPA ASR");
    ImGui::SameLine();
    ImGui::TextDisabled("%s · %s", ASR_VERSION, build_id());
    ImGui::Separator();

    // ---- status line
    if (st.model_loaded) {
        ImGui::Text("Модель: %s", st.model_name.empty() ? file_name(st.model_path).c_str()
                                                        : st.model_name.c_str());
    } else if (st.model_loading) {
        ImGui::TextColored(kWarnYellow, "Модель: загрузка…");
    } else {
        ImGui::TextColored(kDimText, "Модель: модель не загружена");
    }
    ImGui::Text("Сочетание: %s", st.hotkey.empty() ? ui.settings.hotkey.c_str() : st.hotkey.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("(%s)", st.hotkey_backend.empty()
                                    ? (st.listening ? "активно" : "недоступно")
                                    : st.hotkey_backend.c_str());
    ImGui::Text("Микрофон: %s", st.audio_device.empty() ? "системный по умолчанию"
                                                        : st.audio_device.c_str());
    ImGui::Text("Потоки: %d", st.n_threads);

    if (!st.error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, kRecordingRed);
        ImGui::TextWrapped("Ошибка: %s", st.error.c_str());
        ImGui::PopStyleColor();
    }
    if (!st.model_loaded && !st.model_loading) {
        ImGui::TextColored(kWarnYellow, "Загрузите модель на вкладке «Модели».");
    }
    if (!st.listening && !st.hotkey_backend.empty()) {
        ImGui::TextDisabled("Глобальное сочетание недоступно — используйте кнопку ниже.");
    }

    ImGui::Spacing();

    // ---- push-to-talk button
    // "Удерживайте для записи (Ctrl+Alt+Space)"
    char ptt_label[192];
    if (st.recording) {
        std::snprintf(ptt_label, sizeof(ptt_label), "● Идёт запись  %.1fs###ptt", st.audio_sec);
        ImGui::PushStyleColor(ImGuiCol_Text, kRecordingRed);
        ImGui::PushStyleColor(ImGuiCol_Button, kRecordingBg);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kRecordingBg);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, kRecordingBg);
    } else {
        std::snprintf(ptt_label, sizeof(ptt_label), "Удерживайте для записи (%s)###ptt",
                      ui.settings.hotkey.c_str());
    }
    ImGui::Button(ptt_label, ImVec2(-1.0f, 76.0f));
    if (st.recording) ImGui::PopStyleColor(4);

    // mouse-down -> press, mouse-up -> release (the ###ptt suffix keeps the ID
    // stable while the visible label changes with the recording state)
    if (ImGui::IsItemActivated()) {
        ui.ptt_mouse_held = true;
        controller.press_ptt();
    }
    if (ImGui::IsItemDeactivated()) {
        ui.ptt_mouse_held = false;
        controller.release_ptt();
    }
    if (ui.ptt_mouse_held && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ui.ptt_mouse_held = false;
        controller.release_ptt();
    }
    if (st.recording) {
        ImGui::TextColored(kRecordingRed, "● Идёт запись  %.1fs", st.audio_sec);
        ImGui::SameLine();
        ImGui::TextDisabled("отпустите кнопку или сочетание, чтобы распознать");
    } else if (st.busy) {
        ImGui::TextDisabled("Обработка…");
    } else {
        ImGui::TextDisabled("Удерживайте кнопку или %s во время речи.", ui.settings.hotkey.c_str());
    }

    // ---- actions
    if (ImGui::Button("Скопировать последний текст")) {
        const std::string & text = ui.last_final.empty() ? ui.transcript : ui.last_final;
        if (text.empty()) {
            ui.copy_status = "нечего копировать";
        } else if (platform::clipboard_set_text(text)) {
            ui.copy_status = "текст скопирован в буфер обмена";
        } else {
            ui.copy_status = "не удалось обратиться к буферу обмена";
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Стереть")) {
        ui.transcript.clear();
        ui.last_final.clear();
        ui.copy_status.clear();
    }
    if (!ui.copy_status.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", ui.copy_status.c_str());
    }
    if (!ui.last_final.empty()) {
        ImGui::SameLine();
        if (ImGui::SmallButton("Копировать")) {
            if (platform::clipboard_set_text(ui.last_final)) ui.copy_status = "текст скопирован";
        }
        ImGui::SameLine();
        ImGui::TextDisabled("последний результат: %.0f мс", st.last_infer_ms);
    }

    // ---- live text
    ImGui::SeparatorText("Распознанный текст");

    std::string tail;
    if (st.recording) {
        tail = st.committed + st.provisional;
    } else if (!st.committed.empty() && st.committed != ui.last_final) {
        tail = st.committed;  // final pass in progress
    }
    // a finished utterance is appended to the scrollback and stays visible
    if (!st.final_text.empty() && st.final_text != ui.last_final) {
        if (!ui.transcript.empty()) ui.transcript += "\n";
        ui.transcript += st.final_text;
        ui.last_final = st.final_text;
    }

    ImGui::BeginChild("live_text", ImVec2(0.0f, ImGui::GetContentRegionAvail().y),
                      ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
    const bool at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f;
    if (ui.transcript.empty() && tail.empty()) {
        ImGui::TextDisabled("Здесь появится распознанный текст…");
    }
    if (!ui.transcript.empty()) ImGui::TextWrapped("%s", ui.transcript.c_str());
    if (!tail.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, kDimText);
        ImGui::PushFont(ui.font_italic, 0.0f);  // nullptr keeps the current font
        ImGui::TextWrapped("%s", tail.c_str());
        ImGui::PopFont();
        ImGui::PopStyleColor();
    }
    if (at_bottom || st.recording) ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
}

// Defined with the rest of the install UI below; the settings tab shows it at
// the bottom (installing is a per-user setting, not a separate screen).
void draw_install_section(GuiState & ui);
void text_dim_wrapped(const std::string & text);

void draw_settings_tab(GuiState & ui, Controller & controller, GLFWwindow * window,
                       const AppStatus & st) {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw < 1) hw = 1;

    ImGui::SeparatorText("Распознавание");

    bool auto_threads = ui.settings.n_threads == 0;
    if (ImGui::Checkbox("авто", &auto_threads)) {
        ui.settings.n_threads = auto_threads ? 0 : default_n_threads();
        touch_settings(ui);
    }
    ImGui::SameLine();
    int threads = ui.settings.n_threads > 0 ? ui.settings.n_threads : default_n_threads();
    ImGui::BeginDisabled(auto_threads);
    if (ImGui::SliderInt("Потоки", &threads, 1, hw)) {
        ui.settings.n_threads = threads;
        touch_settings(ui);
    }
    ImGui::EndDisabled();
    if (auto_threads) {
        ImGui::TextDisabled("используется: %d из %d (авто: половина логических ядер)", st.n_threads, hw);
    } else {
        ImGui::TextDisabled("используется: %d из %d", st.n_threads, hw);
    }

    if (ImGui::SliderFloat("Шаг потока, с", &ui.settings.step_sec, 0.2f, 3.0f, "%.2f")) touch_settings(ui);
    if (ImGui::SliderFloat("Окно, с", &ui.settings.max_window_sec, 5.0f, 30.0f, "%.1f")) touch_settings(ui);

    static const char * const kModes[] = {
        "вставлять в окно + буфер",
        "только буфер",
        "только в окне",
    };
    int mode = static_cast<int>(ui.settings.output_mode);
    if (ImGui::Combo("Вывод текста", &mode, kModes, IM_ARRAYSIZE(kModes))) {
        ui.settings.output_mode = static_cast<OutputMode>(mode);
        touch_settings(ui);
    }

    ImGui::SeparatorText("Управление");

    ImGui::TextUnformatted("Сочетание клавиш (push-to-talk)");
    if (ImGui::InputText("##hotkey", ui.hotkey_buf, sizeof(ui.hotkey_buf))) {
        platform::Chord chord;
        std::string err;
        if (platform::Chord::parse(ui.hotkey_buf, &chord, &err)) {
            ui.settings.hotkey = chord.to_string();
            ui.hotkey_status.clear();
            touch_settings(ui);
        } else {
            ui.hotkey_status = "неверное сочетание: " + err;
        }
    }
    if (ImGui::Button(ui.capturing ? "Нажмите сочетание… (Esc — отмена)" : "Записать сочетание")) {
        ui.capturing = true;
        ui.hotkey_status = "нажмите нужную комбинацию клавиш…";
    }
    if (ui.capturing) {
        ImGui::SameLine();
        ImGui::TextColored(kWarnYellow, "● запись сочетания");
    }
    if (!ui.hotkey_status.empty()) ImGui::TextDisabled("%s", ui.hotkey_status.c_str());
    ImGui::TextDisabled("текущее: %s", ui.settings.hotkey.c_str());

    ImGui::SeparatorText("Устройства");

    const std::vector<audio::DeviceInfo> & devices = controller.devices();
    std::string preview = "системный по умолчанию";
    for (const auto & d : devices) {
        if (d.index == ui.settings.audio_device) {
            preview = d.name;
            break;
        }
    }
    ImGui::TextUnformatted("Микрофон");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-110.0f);  // leave room for the refresh button
    if (ImGui::BeginCombo("##microphone", preview.c_str())) {
        const bool default_selected = ui.settings.audio_device < 0;
        if (ImGui::Selectable("системный по умолчанию", default_selected)) {
            ui.settings.audio_device = -1;
            touch_settings(ui);
        }
        if (default_selected) ImGui::SetItemDefaultFocus();
        for (const auto & d : devices) {
            const bool selected = d.index == ui.settings.audio_device;
            const std::string label = d.name + (d.is_default ? "  [по умолчанию]" : "");
            if (ImGui::Selectable(label.c_str(), selected)) {
                ui.settings.audio_device = d.index;
                touch_settings(ui);
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("Обновить")) controller.refresh_devices();

    ImGui::SeparatorText("Окно и оформление");

    if (ImGui::Checkbox("Тёмная тема", &ui.settings.dark_theme)) {
        ui.scale_pending = true;  // apply_style() rebuilds theme + scale
        touch_settings(ui);
    }
    if (ImGui::Checkbox("Поверх остальных окон", &ui.always_on_top)) {
        ui.settings.always_on_top = ui.always_on_top;
        set_always_on_top(window, ui.always_on_top);
        touch_settings(ui);
    }

    // ---- interface scale: Ctrl +/-, or the buttons, or automatic (monitor DPI)
    ImGui::TextUnformatted("Масштаб интерфейса");
    ImGui::SameLine();
    if (ImGui::SmallButton("-##scale_down")) handle_scale_shortcut(ui, GLFW_KEY_MINUS);
    ImGui::SameLine();
    if (ImGui::SmallButton("+##scale_up")) handle_scale_shortcut(ui, GLFW_KEY_EQUAL);
    ImGui::SameLine();
    ImGui::Text("%d %%", static_cast<int>(std::lround(ui.ui_scale * 100.0f)));
    ImGui::SameLine();
    bool auto_scale = ui.settings.ui_scale <= 0.0f;
    if (ImGui::Checkbox("авто##scale_auto", &auto_scale)) {
        ui.settings.ui_scale = auto_scale ? 0.0f : ui.ui_scale;
        ui.scale_pending = true;
        touch_settings(ui);
    }
    if (auto_scale) {
        ImGui::TextDisabled("по масштабу монитора: %d %%",
                            static_cast<int>(std::lround(ui.monitor_scale_now * 100.0f)));
    }
    ImGui::TextDisabled("горячие клавиши: Ctrl + / Ctrl - / Ctrl 0 (шаг 5 %%)");

    ImGui::Separator();
    if (ImGui::Button("Сохранить настройки")) {
        controller.apply_settings(ui.settings);
        std::string err;
        if (ui.settings.save(&err)) {
            ui.settings_dirty = false;
            ui.hotkey_status = "настройки сохранены в " + Settings::settings_path();
        } else {
            ui.hotkey_status = "ошибка сохранения: " + err;
        }
    }
    ImGui::SameLine();
    if (ui.settings_dirty) {
        ImGui::TextColored(kWarnYellow, "есть неприменённые изменения");
    } else {
        ImGui::TextDisabled("применяются автоматически");
    }

    // ---- install / autostart: the settings the application keeps for this user
    ImGui::SeparatorText("Установка приложения");
    draw_install_section(ui);
    text_dim_wrapped("Установка — это копирование этого файла в личный каталог (права "
                     "администратора не нужны); настройки и модели остаются на месте.");
    text_dim_wrapped("Командная строка: npa_asr --install | --autostart on|off | "
                     "--install-status | --uninstall");

    ImGui::SeparatorText("Завершение работы");
    if (ui.tray_active) {
        text_dim_wrapped("Крестик сворачивает приложение в трей: распознавание и сочетание "
                         "клавиш продолжают работать, вернуться можно щелчком по значку рядом "
                         "с часами (там же «Выход»).");
        if (ImGui::Button("Свернуть в трей")) {
            glfwHideWindow(window);
            ui.tray_status = "приложение свёрнуто в трей — значок рядом с часами";
        }
        ImGui::SameLine();
    } else {
        text_dim_wrapped("Системный трей недоступен, поэтому крестик закрывает приложение.");
    }
    if (ImGui::Button("Выход")) ui.quit_requested = true;
    if (!ui.tray_status.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", ui.tray_status.c_str());
    }
}

void draw_models_tab(GuiState & ui, Controller & controller) {
    if (ui.models_need_scan) {
        ui.local_models = scan_models_dir(ui.models_dir);
        ui.models_need_scan = false;
    }

    ImGui::SeparatorText("Локальные модели");
    if (ImGui::Button("Обновить")) ui.models_need_scan = true;
    ImGui::SameLine();
    ImGui::TextDisabled("%s", ui.models_dir.c_str());

    if (ui.local_models.empty()) {
        ImGui::TextDisabled("В каталоге нет файлов .gguf.");
    } else if (ImGui::BeginTable("local_models", 4,
                                 ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Файл", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Квант", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Размер", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableHeadersRow();
        for (const LocalModel & m : ui.local_models) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(m.name.c_str());
            if (!m.path.empty() && m.path == ui.settings.model_path) {
                ImGui::SameLine();
                ImGui::TextDisabled("[текущая]");
            }
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(m.quant.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(format_mb(m.size).c_str());
            ImGui::TableNextColumn();
            ImGui::PushID(m.path.c_str());
            if (ImGui::Button("Загрузить")) request_load(ui, controller, m.path);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    ImGui::SeparatorText("Путь к модели");
    ImGui::SetNextItemWidth(-215.0f);  // room for the two buttons on this line
    ImGui::InputText("##model_path", ui.path_buf, sizeof(ui.path_buf));
    ImGui::SameLine();
    if (ImGui::Button("Загрузить##path")) request_load(ui, controller, ui.path_buf);
    ImGui::SameLine();
    if (ImGui::Button("Обзор…")) {
        if (ui.browser_dir.empty()) {
            if (!ui.settings.model_path.empty()) {
                ui.browser_dir = fs::path(ui.settings.model_path).parent_path().string();
            }
            if (ui.browser_dir.empty() || !fs::is_directory(fs::path(ui.browser_dir))) {
                ui.browser_dir = platform::home_dir();
            }
        }
        ui.browser_open = true;
    }
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
    ImGui::TextWrapped("Совет: файл .gguf можно просто перетащить в окно приложения.");
    ImGui::PopStyleColor();

    ImGui::SeparatorText("Каталог моделей");
    if (!ui.manifest_loaded && !ui.manifest_loading && ui.manifest_error.empty()) {
        start_manifest_fetch(ui);  // first visit to the tab: fetch in background
    }
    if (ImGui::Button("Обновить каталог")) start_manifest_fetch(ui);
    ImGui::SameLine();
    if (ui.manifest_loading) {
        ImGui::TextDisabled("получение каталога…");
    } else if (ui.manifest_loaded) {
        ImGui::TextDisabled("моделей: %d", static_cast<int>(ui.manifest.models.size()));
    }

    if (!ui.manifest_error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, kRecordingRed);
        ImGui::TextWrapped("Каталог недоступен: %s", ui.manifest_error.c_str());
        ImGui::PopStyleColor();
        ImGui::TextWrapped(
            "Если сети нет, сконвертируйте модель локально с помощью "
            "tools/convert_gigaam_to_gguf.py и загрузите полученный файл с диска.");
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
        for (const std::string & url : net::manifest_urls()) {
            ImGui::TextWrapped("  %s", url.c_str());
        }
        ImGui::PopStyleColor();
    }

    for (size_t mi = 0; mi < ui.manifest.models.size(); ++mi) {
        const net::ModelEntry & entry = ui.manifest.models[mi];
        ImGui::SeparatorText(entry.name.empty() ? entry.id.c_str() : entry.name.c_str());
        if (!entry.license.empty()) ImGui::TextDisabled("Лицензия: %s", entry.license.c_str());
        // The suggested quantization goes first; the rest stays by size, so the
        // list reads "what to take" before "the other options".
        std::vector<size_t> order(entry.files.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const net::ModelFile & fa = entry.files[a];
            const net::ModelFile & fb = entry.files[b];
            if (fa.recommended != fb.recommended) return fa.recommended;
            return fa.size > fb.size;
        });
        for (size_t fi : order) {
            const net::ModelFile & file = entry.files[fi];
            ImGui::PushID(static_cast<int>(mi * 1000 + fi));
            ImGui::Text("%-10s %s", file.quant.c_str(), format_mb(file.size).c_str());
            if (file.recommended) {
                ImGui::SameLine();
                ImGui::TextDisabled("рекомендуется");
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(ui.job != nullptr);
            if (ImGui::Button("Скачать")) {
                begin_download(ui, file.url, file.sha256,
                               entry.name.empty() ? entry.id : entry.name);
            }
            ImGui::EndDisabled();
            ImGui::PopID();
        }
    }

    // ---- download progress
    if (ui.job) {
        const double fraction = ui.job->fraction();
        const double done = static_cast<double>(ui.job->progress.done.load());
        const double total = static_cast<double>(ui.job->progress.total.load());
        char overlay[128];
        if (fraction >= 0.0) {
            std::snprintf(overlay, sizeof(overlay), "%.0f%%  %.1f / %.1f МБ", fraction * 100.0,
                          done / 1e6, total / 1e6);
        } else {
            std::snprintf(overlay, sizeof(overlay), "%.1f МБ", done / 1e6);
        }
        ImGui::ProgressBar(static_cast<float>(fraction < 0.0 ? 0.0 : fraction), ImVec2(-1.0f, 0.0f),
                           overlay);
        ImGui::TextDisabled("%s", ui.job_label.c_str());
        if (ImGui::Button("Отмена")) ui.job->cancel.store(true);
    }

    if (!ui.models_status.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", ui.models_status.c_str());
    }

    draw_file_browser(ui);
}

void draw_log_tab(GuiState & ui) {
    if (ImGui::Button("Очистить")) ui.log_lines.clear();
    ImGui::SameLine();
    ImGui::TextDisabled("сообщений: %d", static_cast<int>(ui.log_lines.size()));

    ImGui::BeginChild("log_child", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
    const bool at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f;
    // PushFont(font, 0.0f) means "keep the current size" in ImGui 1.92, which
    // would render the monospace font at the regular font's size; passing the
    // font's own LegacySize is the documented way to get the size it was loaded
    // with.
    ImGui::PushFont(ui.font_mono, ui.font_mono != nullptr ? ui.font_mono->LegacySize : 0.0f);
    ImGui::PushTextWrapPos(0.0f);  // long paths and URLs must wrap, not scroll
    for (const std::string & line : ui.log_lines) ImGui::TextUnformatted(line.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopFont();
    if (ui.log_lines.empty()) ImGui::TextDisabled("Журнал пуст.");
    if (at_bottom) ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
}

// ------------------------------------------------------------------ install

void refresh_install_status(GuiState & ui) {
    ui.install = install_status();
    ui.install_checked = true;
}

void run_install(GuiState & ui, bool autostart, bool autostart_only) {
    std::string err;
    if (install_app(autostart, autostart_only, &err)) {
        refresh_install_status(ui);
        ui.install_error.clear();
        ui.install_message = autostart_only ? "автозагрузка включена"
                                            : "установлено: " + ui.install.install_exe +
                                                  " (сборка " + build_id() + ")";
    } else {
        ui.install_error = err;
    }
    mark_install_prompt_shown();
}

void run_uninstall(GuiState & ui) {
    std::string err;
    if (uninstall_app(false, &err)) {
        refresh_install_status(ui);
        ui.install_error.clear();
        ui.install_message = "установленная копия и автозагрузка удалены";
    } else {
        ui.install_error = err;
    }
}

void set_autostart_checked(GuiState & ui, bool on) {
    std::string err;
    if (set_autostart(on, &err)) {
        refresh_install_status(ui);
        ui.install_error.clear();
        ui.install_message = on ? "автозагрузка включена" : "автозагрузка выключена";
    } else {
        ui.install_error = err;
    }
}

// Dimmed text that wraps: install paths are long and the panel is narrow.
void text_dim_wrapped(const std::string & text) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
    ImGui::TextWrapped("%s", text.c_str());
    ImGui::PopStyleColor();
}

void draw_install_section(GuiState & ui) {
    if (!ui.install_checked) refresh_install_status(ui);
    const InstallStatus & st = ui.install;
    const bool update = install_update_pending(st);

    if (st.running_installed) {
        ImGui::TextWrapped("Приложение запущено из установленной копии (сборка %s).", build_id());
    } else if (st.installed && st.up_to_date) {
        ImGui::TextWrapped("Установленная копия совпадает с запущенной (сборка %s).", build_id());
    } else if (update) {
        ImGui::TextWrapped("Установлена копия другой сборки (%s), запущена %s — её можно обновить.",
                           st.installed_build_id.empty() ? "без метки" : st.installed_build_id.c_str(),
                           build_id());
    } else if (st.installed) {
        ImGui::TextWrapped("Установленная копия — та же сборка (%s), но файл отличается "
                           "(пересборка без изменений в коде).",
                           build_id());
    } else {
        ImGui::TextWrapped("Приложение не установлено, запускается из текущего файла (сборка %s).",
                           build_id());
    }
    text_dim_wrapped("Каталог установки: " + st.install_dir);
    if (!st.detail.empty()) text_dim_wrapped(st.detail);

    bool autostart = st.autostart;
    if (ImGui::Checkbox("Запускать при входе в систему (свёрнутым)", &autostart)) {
        set_autostart_checked(ui, autostart);
    }
    text_dim_wrapped("Автозапуск: " + st.autostart_target);

    // The primary action owns its line: with the longer labels ("Обновить
    // установленную копию") the buttons do not fit side by side in the panel.
    if (!st.running_installed) {
        const char * label = !st.installed ? "Установить в систему"
                                           : (update ? "Обновить установленную копию"
                                                     : "Переустановить");
        if (ImGui::Button(label)) run_install(ui, false, false);
        ImGui::SameLine();
    }
    if (ImGui::Button("Проверить")) refresh_install_status(ui);
    if (st.installed && !st.running_installed) {
        if (ImGui::Button("Удалить копию")) run_uninstall(ui);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Удалить автозапуск и установленный файл;\nнастройки и модели останутся");
        }
    }

    if (!ui.install_message.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, accent_color(ui.settings.dark_theme));
        ImGui::TextWrapped("%s", ui.install_message.c_str());
        ImGui::PopStyleColor();
    }
    if (!ui.install_error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, kRecordingRed);
        ImGui::TextWrapped("%s", ui.install_error.c_str());
        ImGui::PopStyleColor();
    }
}

// Offered on the first run of a copy that is not installed yet, and once for
// every new source revision afterwards.
void draw_install_popup(GuiState & ui) {
    if (!ui.install_offer) return;

    if (!ui.install_popup_opened) {
        refresh_install_status(ui);
        ui.install_autostart = ui.install.autostart || !ui.install.installed;
        ImGui::OpenPopup("Установка приложения");
        ui.install_popup_opened = true;
    }

    const bool update = install_update_pending(ui.install);
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(560.0f * ui.ui_scale, 0.0f), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("Установка приложения", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }

    if (update) {
        ImGui::TextWrapped("Установлена копия сборки %s. Заменить её этой сборкой (%s)?",
                           ui.install.installed_build_id.empty()
                               ? "без метки"
                               : ui.install.installed_build_id.c_str(),
                           build_id());
    } else {
        ImGui::TextWrapped("Скопировать приложение в личную папку пользователя и запускать его при входе в систему?");
    }
    ImGui::Spacing();
    ImGui::BulletText("Программа: %s", ui.install.install_exe.c_str());
    ImGui::BulletText("Настройки и модели остаются в %s", platform::user_data_dir().c_str());
    ImGui::BulletText("Установка не требует прав администратора");
    ImGui::Spacing();

    bool autostart = ui.install_autostart;
    if (ImGui::Checkbox("Запускать при входе в систему (свёрнутым)", &autostart)) {
        ui.install_autostart = autostart;
    }

    ImGui::Spacing();
    if (ImGui::Button(update ? "Обновить" : "Установить", ImVec2(160.0f * ui.ui_scale, 0.0f))) {
        run_install(ui, ui.install_autostart, false);
        ui.install_offer = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button(update ? "Позже" : "Не сейчас", ImVec2(120.0f * ui.ui_scale, 0.0f))) {
        mark_install_prompt_shown();
        ui.install_offer = false;
        ImGui::CloseCurrentPopup();
    }
    if (!ui.install_error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, kRecordingRed);
        ImGui::TextWrapped("%s", ui.install_error.c_str());
        ImGui::PopStyleColor();
    }
    ImGui::EndPopup();
}

void draw_about_tab(GuiState & ui) {
    if (!ui.info_cached) {
        ui.backend_info = backend_summary();
        ui.cpu_info = platform::cpu_brand();
        ui.os_info = platform::os_name();
        ui.info_cached = true;
    }

    ImGui::SeparatorText("NPA ASR");
    ImGui::Text("Версия: %s   ·   сборка %s", ASR_VERSION, build_id());
    ImGui::TextWrapped(
        "Офлайн-распознавание русской речи (GigaAM-v3 e2e_rnct на ggml): "
        "push-to-talk, потоковое распознавание и вставка текста в активное окно.");

    ImGui::SeparatorText("Система");
    ImGui::TextWrapped("ОС: %s", ui.os_info.empty() ? "неизвестно" : ui.os_info.c_str());
    ImGui::TextWrapped("Процессор: %s", ui.cpu_info.empty() ? "неизвестно" : ui.cpu_info.c_str());
    ImGui::TextWrapped("Вычислительные бэкенды: %s",
                       ui.backend_info.empty() ? "неизвестно" : ui.backend_info.c_str());

    ImGui::SeparatorText("Файлы");
    ImGui::TextWrapped("Настройки: %s", Settings::settings_path().c_str());
    ImGui::TextWrapped("Каталог моделей: %s", ui.models_dir.c_str());

    ImGui::SeparatorText("Модель и лицензия");
    if (ImGui::Button("Открыть страницу GigaAM-v3")) {
        platform::open_in_browser("https://huggingface.co/ai-sage/GigaAM-v3");
    }
    ImGui::TextWrapped(
        "Лицензия: MIT. Приложение распространяется по лицензии MIT; модель GigaAM-v3 "
        "опубликована ai-sage на Hugging Face под собственной лицензией — проверьте её "
        "перед коммерческим использованием.");
}

// ------------------------------------------------------------------ frame

void poll_background(GuiState & ui, Controller & controller) {
    // ---- model catalogue (fetch_manifest runs on its own thread)
    poll_manifest(ui);

    // ---- downloads (start_download runs its own thread)
    if (ui.job && ui.job->finished.load()) {
        const bool ok = ui.job->ok;
        const std::string dest = ui.job->dest;
        const std::string error = ui.job->error;
        ui.job.reset();  // joins the finished worker in ~DownloadJob
        ui.models_need_scan = true;
        if (ok) {
            ui.models_status = "загружено: " + dest;
            request_load(ui, controller, dest);
        } else if (error == "cancelled") {
            ui.models_status = "загрузка отменена";
        } else {
            ui.models_status = "ошибка загрузки: " + error;
        }
    }

    // ---- a .gguf dropped onto the window (collected in the GLFW drop callback)
    if (!ui.pending_load.empty()) {
        const std::string path = ui.pending_load;
        ui.pending_load.clear();
        request_load(ui, controller, path);
    }
}

// Applies the theme/scale and, when the user changed the scale explicitly, the
// matching window size.  Must run *before* ImGui::NewFrame() so that the frame
// is laid out with the new style.
void update_interface_scale(GuiState & ui, GLFWwindow * window) {
    const float monitor = monitor_scale(window);
    const bool monitor_changed = std::fabs(monitor - ui.monitor_scale_now) > 0.001f;
    ui.monitor_scale_now = monitor;

    if (ui.theme_applied && !ui.scale_pending && !monitor_changed) return;

    const bool first = !ui.theme_applied;
    const float previous = ui.ui_scale;
    const float wanted =
        ui.settings.ui_scale > 0.0f ? clamp_scale(ui.settings.ui_scale) : monitor;

    if (!first && std::fabs(wanted - previous) > 0.001f) {
        resize_window_for_scale(window, previous, wanted);
    }

    apply_style(ui.settings.dark_theme, wanted);

    if (!first && std::fabs(wanted - previous) > 0.001f) {
        ui.scale_toast = "масштаб интерфейса: " +
                         std::to_string(static_cast<int>(std::lround(wanted * 100.0f))) + " %";
        ui.scale_toast_until = now_seconds() + 1.5;
    }
    ui.ui_scale = wanted;
    ui.scale_pending = false;
    if (first) set_always_on_top(window, ui.always_on_top);
    ui.theme_applied = true;
}

// Each tab scrolls on its own.  Without this the whole panel scrolls - the tab
// strip included - as soon as a tab is taller than the window, and the tabs
// disappear from view while scrolling.
void begin_tab_body(const char * id) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::BeginChild(id, ImVec2(0.0f, 0.0f));
    ImGui::PopStyleVar();
}

void end_tab_body() { ImGui::EndChild(); }

void draw_frame(GuiState & ui, Controller & controller, GLFWwindow * window) {

    const AppStatus st = controller.status();  // one snapshot per frame

    for (const std::string & message : controller.take_log()) ui.log_lines.push_back(message);

    // The install state is read once (it hashes two executables) and the
    // settings tab works off this snapshot; the actions refresh it.
    if (!ui.install_checked) refresh_install_status(ui);

    // debounced apply_settings(): sliders and combos may change every frame
    if (ui.settings_dirty && (now_seconds() - ui.settings_changed_at) > 0.4) {
        controller.apply_settings(ui.settings);
        ui.settings_dirty = false;
    }

    const ImGuiViewport * viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 10.0f));
    const ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoSavedSettings;
    ImGui::Begin("NPA ASR", nullptr, window_flags);
    ImGui::PopStyleVar(3);

    const float available_w = ImGui::GetContentRegionAvail().x;
    // The panel minimums are in logical pixels but the widgets inside them are
    // scaled, so they have to follow the interface scale.
    const float panel_scale = ui.ui_scale > 0.0f ? ui.ui_scale : 1.0f;
    float right_w = available_w * 0.44f;
    if (right_w < 380.0f * panel_scale) right_w = 380.0f * panel_scale;
    if (right_w > available_w - 320.0f * panel_scale) right_w = available_w - 320.0f * panel_scale;
    if (right_w < 320.0f * panel_scale) right_w = 320.0f * panel_scale;
    const float left_w = available_w - right_w - ImGui::GetStyle().ItemSpacing.x;

    ImGui::BeginChild("main_panel", ImVec2(left_w, 0.0f), ImGuiChildFlags_Borders);
    draw_main_panel(ui, controller, window, st);
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("tabs_panel", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
    if (ImGui::BeginTabBar("asr_tabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Настройки")) {
            begin_tab_body("settings_body");
            draw_settings_tab(ui, controller, window, st);
            end_tab_body();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Модели")) {
            begin_tab_body("models_body");
            draw_models_tab(ui, controller);
            end_tab_body();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Журнал")) {
            begin_tab_body("log_body");
            draw_log_tab(ui);
            end_tab_body();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("О сборке")) {
            begin_tab_body("about_body");
            draw_about_tab(ui);
            end_tab_body();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();

    ImGui::End();

    // ---- transient scale indicator (Ctrl +/-, or a monitor change)
    if (!ui.scale_toast.empty()) {
        if (now_seconds() < ui.scale_toast_until) {
            const ImGuiViewport * vp = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x - 12.0f * ui.ui_scale,
                                           vp->WorkPos.y + vp->WorkSize.y - 12.0f * ui.ui_scale),
                                    ImGuiCond_Always, ImVec2(1.0f, 1.0f));
            const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                           ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize |
                                           ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
            ImGui::SetNextWindowBgAlpha(0.85f);
            if (ImGui::Begin("##scale_toast", nullptr, flags)) {
                ImGui::TextUnformatted(ui.scale_toast.c_str());
            }
            ImGui::End();
        } else {
            ui.scale_toast.clear();
        }
    }

    draw_install_popup(ui);
}

}  // namespace

// ------------------------------------------------------------------ entry point

int run_gui(bool minimized) {
    ensure_dpi_awareness();
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::fprintf(stderr, "npa_asr: glfwInit() failed\n");
        return 1;
    }

#if defined(__APPLE__)
    const char * glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#else
    const char * glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    // Created hidden and shown further down, once the icon is in place: the
    // taskbar button is built when the window first becomes visible, so a window
    // that glfwCreateWindow() shows by itself would keep the generic GLFW icon
    // until it is hidden and shown again (which is what "the icon only appears
    // after the first trip to the tray" was).
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    // The revision in the title bar makes it obvious which build is on screen.
    const std::string window_title =
        std::string("NPA ASR ") + ASR_VERSION + " (" + build_id() + ")";
    GLFWwindow * window = glfwCreateWindow(1040, 720, window_title.c_str(), nullptr, nullptr);
    if (window == nullptr) {
        const char * description = nullptr;
        glfwGetError(&description);
        std::fprintf(stderr, "npa_asr: glfwCreateWindow() failed: %s\n",
                     description != nullptr ? description : "unknown error");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // vsync: the main loop runs at the display rate (~60 FPS)
    {
        // The same icon the tray uses, so the taskbar/Alt-Tab entry is not the
        // generic GLFW one.  Both shell sizes are supplied: 16 px is the title
        // bar and the notification area, 32 px the taskbar and Alt-Tab.
        std::vector<unsigned char> icon_16;
        std::vector<unsigned char> icon_32;
        GLFWimage images[2];
        int image_count = 0;
        if (platform::app_icon_rgba(16, &icon_16)) {
            images[image_count].width = 16;
            images[image_count].height = 16;
            images[image_count].pixels = icon_16.data();
            ++image_count;
        }
        if (platform::app_icon_rgba(32, &icon_32)) {
            images[image_count].width = 32;
            images[image_count].height = 32;
            images[image_count].pixels = icon_32.data();
            ++image_count;
        }
        if (image_count > 0) glfwSetWindowIcon(window, image_count, images);
    }
    // Size limits are in screen coordinates, i.e. physical pixels on a scaled
    // display, so they have to be scaled as well.
    const float start_scale = monitor_scale(nullptr);
    glfwSetWindowSizeLimits(window, static_cast<int>(std::lround(720.0f * start_scale)),
                            static_cast<int>(std::lround(480.0f * start_scale)),
                            GLFW_DONT_CARE, GLFW_DONT_CARE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO & io = ImGui::GetIO();
    io.IniFilename = nullptr;  // the layout is fixed, do not litter imgui.ini
    io.LogFilename = nullptr;

    // Cyrillic glyphs are mandatory: both the labels and the recognised text are
    // Russian, and the built-in ImGui font has no Cyrillic coverage.
    GuiState ui;
    ui.font_regular =
        add_cyrillic_font(io.Fonts, kRegularFonts, IM_ARRAYSIZE(kRegularFonts), 17.0f);
    if (ui.font_regular == nullptr) {
        std::fprintf(stderr,
                     "npa_asr: warning: no TrueType font with Cyrillic glyphs found, "
                     "falling back to the ImGui default font\n");
        io.Fonts->AddFontDefault();
        ui.font_regular = io.Fonts->Fonts.empty() ? nullptr : io.Fonts->Fonts[0];
    }
    ui.font_italic = add_cyrillic_font(io.Fonts, kItalicFonts, IM_ARRAYSIZE(kItalicFonts), 17.0f);
    ui.font_mono = add_cyrillic_font(io.Fonts, kMonoFonts, IM_ARRAYSIZE(kMonoFonts), 15.0f);

    glfwSetWindowUserPointer(window, &ui);
    // installed *before* ImGui initialises its callbacks so the backend chains
    // into them instead of overwriting them
    glfwSetKeyCallback(window, glfw_key_callback);
    glfwSetDropCallback(window, glfw_drop_callback);
    // The ImGui GLFW backend installs no close callback of its own.
    glfwSetWindowCloseCallback(window, glfw_close_callback);

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
        std::fprintf(stderr, "npa_asr: ImGui_ImplGlfw_InitForOpenGL() failed\n");
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    if (!ImGui_ImplOpenGL3_Init(glsl_version)) {
        std::fprintf(stderr, "npa_asr: ImGui_ImplOpenGL3_Init() failed\n");
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    ui.models_dir = default_models_dir();

    Controller controller;
    Settings settings = Settings::load();
    std::string init_error;
    if (!controller.init(settings, &init_error)) {
        std::fprintf(stderr, "npa_asr: controller: %s\n", init_error.c_str());
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    ui.settings = settings;
    ui.always_on_top = settings.always_on_top;
    std::snprintf(ui.path_buf, sizeof(ui.path_buf), "%s", ui.settings.model_path.c_str());
    std::snprintf(ui.hotkey_buf, sizeof(ui.hotkey_buf), "%s", ui.settings.hotkey.c_str());

    // Offer to install (or, for a build with a different source revision, to
    // update) the per-user copy once - see should_offer_install().
    ui.install_offer = should_offer_install();

    // ---- notification area: the close button hides the window while the icon
    // is there (the hotkey keeps working), and "Выход" in its menu quits.
    ui.tray = platform::create_tray_icon();
    {
        std::string tray_error;
        const std::string tip = "NPA ASR " + std::string(ASR_VERSION) + " — удерживайте " +
                                ui.settings.hotkey + ", чтобы записать";
        if (ui.tray->add(tip, &tray_error)) {
            ui.tray_active = true;
            ui.log_lines.push_back("трей: значок добавлен (" +
                                   std::string(ui.tray->backend_name()) + ")");
        } else {
            ui.log_lines.push_back("трей: недоступен, крестик закрывает приложение (" +
                                   tray_error + ")");
        }
    }

    // First show, with the window icon already registered (see the
    // GLFW_VISIBLE hint above).  The autostart entry starts minimised.
    glfwShowWindow(window);
    if (minimized) glfwIconifyWindow(window);

    while (!glfwWindowShouldClose(window) && !ui.quit_requested) {
        const bool visible = glfwGetWindowAttrib(window, GLFW_VISIBLE) != GLFW_FALSE;
        if (visible) {
            glfwPollEvents();
        } else {
            // In the tray: nothing is drawn, so wait for the next event instead
            // of spinning.  The tray backend uses its own thread, hence the
            // short timeout rather than an indefinite wait.
            glfwWaitEventsTimeout(0.05);
        }
        poll_tray(ui, window);
        poll_background(ui, controller);
        if (!visible) continue;  // hidden: no frames, no ImGui state to update

        // before NewFrame: the theme/scale and the window size they need
        update_interface_scale(ui, window);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        draw_frame(ui, controller, window);

        ImGui::Render();

        int display_w = 0;
        int display_h = 0;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        const ImVec4 clear = ui.settings.dark_theme ? ImVec4(0.08f, 0.09f, 0.10f, 1.00f)
                                                    : ImVec4(0.90f, 0.91f, 0.93f, 1.00f);
        glClearColor(clear.x, clear.y, clear.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // ---- shutdown: persist settings, stop the controller, tear GLFW down
    if (ui.tray) ui.tray->remove();
    {
        std::string save_error;
        if (!ui.settings.save(&save_error)) {
            std::fprintf(stderr, "npa_asr: settings: %s\n", save_error.c_str());
        }
    }
    controller.shutdown();

    if (ui.fetch_thread.joinable()) {
        // give the network worker a short grace period, then let it finish
        // detached: it only owns its own shared result, never the GUI state
        for (int i = 0; i < 30; ++i) {
            bool done = false;
            if (ui.fetch) {
                std::lock_guard<std::mutex> lock(ui.fetch->mutex);
                done = ui.fetch->done;
            }
            if (done) break;
            platform::sleep_ms(10);
        }
        if (ui.fetch_thread.joinable()) ui.fetch_thread.detach();
    }
    ui.job.reset();  // cancel and join a running download

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

}  // namespace asr::app

#endif  // ASR_HAVE_GUI
