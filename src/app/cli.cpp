#include "app/cli.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include "app/build_info.h"
#include "app/controller.h"
#include "app/install.h"
#include "app/models.h"
#include "app/settings.h"
#include "audio/audio_file.h"
#include "audio/capture.h"
#include "core/backend_info.h"
#include "core/fft.h"
#include "core/recognizer.h"
#include "platform/platform.h"

namespace asr::app {

namespace {

void usage() {
    std::printf(
        "npa_asr %s - GigaAM-v3 e2e_rnnt speech recognition on ggml\n"
        "\n"
        "usage:\n"
        "  npa_asr                          launch the GUI\n"
        "  npa_asr --gui                    launch the GUI\n"
        "  npa_asr -m MODEL.gguf -i FILE    transcribe an audio file\n"
        "  npa_asr -m MODEL.gguf --stream FILE   streaming simulation (chunked input)\n"
        "  npa_asr -m MODEL.gguf --mic      headless push-to-talk (hotkey)\n"
        "  npa_asr --list-devices           list capture devices\n"
        "  npa_asr --list-models            list local .gguf models\n"
        "  npa_asr --download ID --quant Q  download a published model\n"
        "  npa_asr -m MODEL.gguf --record 5   record 5 s from the microphone and transcribe\n"
        "  npa_asr --manifest               show the model catalogue\n"
        "  npa_asr --info                   backend / CPU information\n"
        "  npa_asr --install-status         show the install / autostart state\n"
        "\n"
        "options:\n"
        "  -m, --model PATH     path to the .gguf model\n"
        "  -i, --input PATH     audio file (wav/mp3/flac/ogg)\n"
        "  -t, --threads N      worker threads (default: half of the logical cores)\n"
        "      --step SEC       streaming re-encode interval (default 1.0)\n"
        "      --window SEC     sliding window length (default 20)\n"
        "      --models-dir DIR model directory (default: user data dir/models)\n"
        "      --record SEC     record SEC seconds from the microphone\n"
        "      --save-wav PATH  write the recorded audio to a wav file\n"
        "      --dump-tokens    print token ids\n"
        "      --install        установить копию приложения в профиль и включить автозапуск\n"
        "      --install-only   установить копию приложения без автозапуска\n"
        "      --uninstall      удалить автозапуск и установленные файлы\n"
        "      --uninstall-keep-files  удалить автозапуск, файлы оставить\n"
        "      --autostart on|off      включить или выключить автозапуск\n"
        "      --install-status показать состояние установки и автозапуска\n",
        ASR_VERSION);
}

struct Args {
    std::string model, input, download, quant, models_dir;
    int threads = 0;
    float step = 1.0f, window = 20.0f;
    bool stream = false, mic = false, dump_tokens = false;
    bool list_devices = false, list_models = false, manifest = false, info = false;
    bool install = false, install_only = false;
    bool uninstall = false, uninstall_keep_files = false, install_status = false;
    int autostart = -1;  // -1 = not requested, 0 = off, 1 = on
    bool help = false;
    float record_sec = 0.0f;
    std::string save_wav;
};

bool parse_args(const std::vector<std::string> & args, Args * a) {
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string & s = args[i];
        auto next = [&](std::string * out) {
            if (i + 1 < args.size()) { *out = args[++i]; return true; }
            return false;
        };
        if (s == "-m" || s == "--model") { if (!next(&a->model)) return false; }
        else if (s == "-i" || s == "--input") { if (!next(&a->input)) return false; }
        else if (s == "-t" || s == "--threads") { std::string v; if (!next(&v)) return false; a->threads = std::atoi(v.c_str()); }
        else if (s == "--step") { std::string v; if (!next(&v)) return false; a->step = std::stof(v); }
        else if (s == "--window") { std::string v; if (!next(&v)) return false; a->window = std::stof(v); }
        else if (s == "--models-dir") { if (!next(&a->models_dir)) return false; }
        else if (s == "--download") { if (!next(&a->download)) return false; }
        else if (s == "--quant") { if (!next(&a->quant)) return false; }
        else if (s == "--stream") a->stream = true;
        else if (s == "--mic") a->mic = true;
        else if (s == "--dump-tokens") a->dump_tokens = true;
        else if (s == "--list-devices") a->list_devices = true;
        else if (s == "--list-models") a->list_models = true;
        else if (s == "--manifest") a->manifest = true;
        else if (s == "--info") a->info = true;
        else if (s == "--install") a->install = true;
        else if (s == "--install-only") a->install_only = true;
        else if (s == "--uninstall") a->uninstall = true;
        else if (s == "--uninstall-keep-files") a->uninstall_keep_files = true;
        else if (s == "--install-status") a->install_status = true;
        else if (s == "--autostart") {
            std::string v;
            if (!next(&v)) return false;
            if (v == "on") a->autostart = 1;
            else if (v == "off") a->autostart = 0;
            else { std::fprintf(stderr, "--autostart: ожидается on или off\n"); return false; }
        }
        else if (s == "--record") { std::string v; if (!next(&v)) return false; a->record_sec = std::stof(v); }
        else if (s == "--save-wav") { if (!next(&a->save_wav)) return false; }
        else if (s == "-h" || s == "--help") a->help = true;
        else { std::fprintf(stderr, "unknown option: %s\n", s.c_str()); return false; }
    }
    return true;
}

int cmd_info() {
    std::printf("npa_asr %s\n", ASR_VERSION);
    std::printf("build   : %s\n", build_id());
    std::printf("os      : %s\n", platform::os_name().c_str());
    std::printf("cpu     : %s\n", platform::cpu_brand().c_str());
    std::printf("threads : %u logical, auto uses %d\n", std::thread::hardware_concurrency(),
                default_n_threads());
    std::printf("data dir: %s\n", platform::user_data_dir().c_str());
    std::printf("models  : %s\n", default_models_dir().c_str());
    std::printf("ggml devices: %s\n", backend_summary().c_str());
    std::printf("mel FFT : %s\n", fft_backend_name().c_str());
    {
        const std::unique_ptr<platform::TrayIcon> tray = platform::create_tray_icon();
        std::printf("tray    : %s\n", tray->backend_name());
    }
    return 0;
}

int cmd_list_devices() {
    const auto devs = audio::list_capture_devices();
    if (devs.empty()) {
        std::printf("no capture devices found\n");
        return 1;
    }
    for (const auto & d : devs) {
        std::printf("%3d  %s%s\n", d.index, d.name.c_str(), d.is_default ? "  [default]" : "");
    }
    return 0;
}

int cmd_list_models(const std::string & dir) {
    const auto models = scan_models_dir(dir);
    if (models.empty()) {
        std::printf("no .gguf models in %s\n", dir.c_str());
        return 1;
    }
    for (const auto & m : models) {
        std::printf("%10.1f MB  %-12s %s\n", m.size / 1e6, m.quant.c_str(), m.path.c_str());
    }
    return 0;
}

int cmd_manifest() {
    asr::net::Manifest m;
    std::string err;
    if (!fetch_manifest(&m, &err)) {
        std::printf("manifest unavailable: %s\n", err.c_str());
        return 1;
    }
    for (const auto & e : m.models) {
        std::printf("%s  (%s)\n", e.name.c_str(), e.id.c_str());
        for (const auto & f : e.files) {
            std::printf("   %-8s %8.1f MB  %s\n", f.quant.c_str(), f.size / 1e6, f.url.c_str());
        }
    }
    return 0;
}

int cmd_download(const Args & a) {
    asr::net::Manifest m;
    std::string err;
    if (!fetch_manifest(&m, &err)) {
        std::fprintf(stderr, "manifest unavailable: %s\n", err.c_str());
        return 1;
    }
    const std::string id = a.download.empty() ? "gigaam-v3-e2e_rnnt" : a.download;
    const asr::net::ModelEntry * e = m.model(id);
    if (!e) {
        std::fprintf(stderr, "unknown model id: %s\n", id.c_str());
        return 1;
    }
    const std::string quant = a.quant.empty() ? "f16" : a.quant;
    const asr::net::ModelFile * f = e->file(quant);
    if (!f) {
        std::fprintf(stderr, "quant %s is not published for %s\n", quant.c_str(), id.c_str());
        return 1;
    }
    const std::string dir = a.models_dir.empty() ? default_models_dir() : a.models_dir;
    std::filesystem::create_directories(dir);
    std::string name = f->url;
    const size_t slash = name.rfind('/');
    if (slash != std::string::npos) name = name.substr(slash + 1);
    const std::string dest = (std::filesystem::path(dir) / name).string();

    std::printf("downloading %s (%s, %.1f MB) -> %s\n", id.c_str(), quant.c_str(), f->size / 1e6,
                dest.c_str());
    auto job = start_download(f->url, dest, f->sha256);
    int last_pct = -1;
    while (!job->finished.load()) {
        const double fr = job->fraction();
        const int pct = fr < 0 ? -1 : static_cast<int>(fr * 100);
        if (pct != last_pct) {
            std::printf("\r%3d%%  %.1f / %.1f MB", pct < 0 ? 0 : pct, job->progress.done.load() / 1e6,
                        job->progress.total.load() / 1e6);
            std::fflush(stdout);
            last_pct = pct;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::printf("\n");
    if (!job->ok) {
        std::fprintf(stderr, "download failed: %s\n", job->error.c_str());
        return 1;
    }
    std::printf("saved to %s\n", dest.c_str());
    return 0;
}

int cmd_transcribe(const Args & a) {
    std::vector<float> pcm;
    std::string err;
    if (!audio::load_file(a.input, 16000, pcm, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    RecognizerConfig cfg;
    cfg.model_path = a.model;
    cfg.n_threads = a.threads > 0 ? a.threads : default_n_threads();
    cfg.step_sec = a.step;
    cfg.max_window_sec = a.window;
    Recognizer rec;
    if (!rec.init(cfg, &err)) {
        std::fprintf(stderr, "model: %s\n", err.c_str());
        return 1;
    }
    if (!a.stream) {
        Transcription tr;
        if (!rec.transcribe(pcm.data(), static_cast<int>(pcm.size()), &tr, &err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        std::printf("%s\n", tr.text.c_str());
        std::fprintf(stderr, "[%.2fs audio, %.0f ms, %.1fx realtime, %d threads]\n", tr.audio_sec,
                     tr.infer_ms, tr.audio_sec * 1000.0 / std::max(tr.infer_ms, 0.001), rec.n_threads());
        if (a.dump_tokens) {
            std::fprintf(stderr, "tokens:");
            for (int t : tr.tokens) std::fprintf(stderr, " %d", t);
            std::fprintf(stderr, "\n");
        }
        return 0;
    }
    // streaming simulation: feed 100 ms chunks like the microphone would
    rec.begin();
    const int chunk = 1600;
    for (size_t off = 0; off < pcm.size(); off += chunk) {
        const int n = static_cast<int>(std::min<size_t>(chunk, pcm.size() - off));
        rec.feed(pcm.data() + off, n);
    }
    Transcription tr;
    if (!rec.finish(&tr, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::printf("%s\n", tr.text.c_str());
    std::fprintf(stderr, "[streamed, %.0f ms final pass, %d windows]\n", tr.infer_ms, tr.n_windows);
    return 0;
}

int cmd_mic(const Args & a) {
    Settings s = Settings::load();
    s.model_path = a.model;
    if (a.threads > 0) s.n_threads = a.threads;
    s.step_sec = a.step;
    s.max_window_sec = a.window;

    Controller ctl;
    std::string err;
    if (!ctl.init(s, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::printf("listening: hold %s to record, Ctrl+C to quit\n", s.hotkey.c_str());
    // wait for the model to load
    for (int i = 0; i < 600; ++i) {
        if (ctl.status().model_loaded) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!ctl.status().model_loaded) {
        std::fprintf(stderr, "model failed to load: %s\n", ctl.status().error.c_str());
        return 1;
    }
    std::printf("model ready (%s)\n", ctl.status().model_name.c_str());
    std::string last;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const AppStatus st = ctl.status();
        if (st.recording && st.provisional != last) {
            last = st.provisional;
            std::printf("\r[rec] %s%s", st.committed.c_str(), st.provisional.c_str());
            std::fflush(stdout);
        }
        if (!st.recording && !st.final_text.empty() && st.final_text != last) {
            last = st.final_text;
            std::printf("\r%s\n", st.final_text.c_str());
        }
    }
    return 0;
}

int cmd_install(const Args & a) {
    // An explicit --autostart always wins, so "--install --autostart off" cannot
    // silently enable autostart.
    bool want_autostart = a.install && !a.install_only;
    if (a.autostart >= 0) want_autostart = a.autostart == 1;
    std::string err;
    if (!install_app(want_autostart, false, &err)) {
        std::fprintf(stderr, "установка не удалась: %s\n", err.c_str());
        return 1;
    }
    const InstallStatus st = install_status();
    std::printf("установлено: %s (сборка %s)\n", st.install_exe.c_str(), build_id());
    if (want_autostart) {
        std::printf("автозапуск включён: %s\n", st.autostart_target.c_str());
    } else {
        std::printf("автозапуск не включён (включить: npa_asr --autostart on)\n");
    }
    return 0;
}

int cmd_uninstall(const Args & a) {
    std::string err;
    if (!uninstall_app(a.uninstall_keep_files, &err)) {
        std::fprintf(stderr, "удаление не удалось: %s\n", err.c_str());
        return 1;
    }
    if (a.uninstall_keep_files) {
        std::printf("автозапуск удалён, установленные файлы сохранены\n");
    } else {
        std::printf("автозапуск и установленные файлы удалены\n");
    }
    return 0;
}

int cmd_set_autostart(bool on) {
    std::string err;
    if (!set_autostart(on, &err)) {
        std::fprintf(stderr, "не удалось изменить автозапуск: %s\n", err.c_str());
        return 1;
    }
    std::printf(on ? "автозапуск включён\n" : "автозапуск выключен\n");
    return 0;
}

int cmd_install_status() {
    const InstallStatus st = install_status();
    std::printf("исполняемый файл      : %s\n", st.exe_path.c_str());
    std::printf("сборка запущенного    : %s\n", build_id());
    std::printf("каталог установки     : %s\n", st.install_dir.c_str());
    std::printf("установлено           : %s\n", st.installed ? "да" : "нет");
    std::printf("сборка установленной  : %s\n",
                !st.installed ? "-"
                              : (st.installed_build_id.empty() ? "(без метки)"
                                                               : st.installed_build_id.c_str()));
    std::printf("копия актуальна       : %s\n",
                !st.installed ? "-" : (st.up_to_date ? "да" : "нет"));
    std::printf("нужно обновление      : %s\n", install_update_pending(st) ? "да" : "нет");
    std::printf("предложить установку  : %s\n", should_offer_install() ? "да" : "нет");
    std::printf("автозапуск            : %s\n", st.autostart ? "включён" : "выключен");
    std::printf("запущено из установки : %s\n", st.running_installed ? "да" : "нет");
    std::printf("точка автозапуска     : %s\n", st.autostart_target.c_str());
    if (!st.detail.empty()) std::printf("примечание            : %s\n", st.detail.c_str());
    return 0;
}

}  // namespace

// Records a fixed number of seconds from the microphone and transcribes it.
// Useful for checking the audio path without a global hotkey (and on machines
// where grabbing one is not possible).
int cmd_record(const Args & a) {
    RecognizerConfig cfg;
    cfg.model_path = a.model;
    cfg.n_threads = a.threads > 0 ? a.threads : default_n_threads();
    cfg.step_sec = a.step;
    cfg.max_window_sec = a.window;
    Recognizer rec;
    std::string err;
    if (!rec.init(cfg, &err)) {
        std::fprintf(stderr, "model: %s\n", err.c_str());
        return 1;
    }

    audio::RingBuffer ring(1 << 21);
    audio::Capture cap;
    if (!cap.start(-1, 16000,
                   [&](const float * pcm, int n) { ring.write(pcm, static_cast<size_t>(n)); },
                   &err)) {
        std::fprintf(stderr, "audio: %s\n", err.c_str());
        return 1;
    }
    std::printf("recording %.1f s from \"%s\" - speak now\n", a.record_sec, cap.device_name().c_str());
    std::fflush(stdout);

    rec.begin();
    std::vector<float> chunk(4096);
    std::vector<float> all;
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        size_t n = ring.read(chunk.data(), chunk.size());
        if (n) {
            rec.feed(chunk.data(), static_cast<int>(n));
            all.insert(all.end(), chunk.begin(), chunk.begin() + static_cast<long>(n));
        }
        if (el >= a.record_sec) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    size_t n;
    while ((n = ring.read(chunk.data(), chunk.size())) > 0) {
        rec.feed(chunk.data(), static_cast<int>(n));
        all.insert(all.end(), chunk.begin(), chunk.begin() + static_cast<long>(n));
    }
    cap.stop();

    if (!a.save_wav.empty() && !all.empty()) {
        if (audio::write_wav16(a.save_wav, all.data(), all.size(), 16000)) {
            std::fprintf(stderr, "[saved %s: %zu samples]\n", a.save_wav.c_str(), all.size());
        }
    }

    Transcription tr;
    if (!rec.finish(&tr, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::printf("%s\n", tr.text.c_str());
    std::fprintf(stderr, "[%.2f s audio, %.0f ms final pass, %d windows]\n", tr.audio_sec, tr.infer_ms,
                 tr.n_windows);
    return 0;
}

int run_cli(const std::vector<std::string> & args) {
    Args a;
    if (!parse_args(args, &a)) {
        usage();
        return 2;
    }
    if (a.help) {
        usage();
        return 0;
    }
    if (a.install_status) return cmd_install_status();
    if (a.install || a.install_only) return cmd_install(a);
    if (a.uninstall || a.uninstall_keep_files) return cmd_uninstall(a);
    if (a.autostart >= 0) return cmd_set_autostart(a.autostart == 1);
    if (a.info) return cmd_info();
    if (a.list_devices) return cmd_list_devices();
    if (a.list_models) {
        return cmd_list_models(a.models_dir.empty() ? default_models_dir() : a.models_dir);
    }
    if (a.manifest) return cmd_manifest();
    if (!a.download.empty() || !a.quant.empty()) return cmd_download(a);
    if (!a.model.empty() && a.mic) return cmd_mic(a);
    if (!a.model.empty() && a.record_sec > 0) return cmd_record(a);
    if (!a.model.empty() && !a.input.empty()) return cmd_transcribe(a);

    usage();
    return 2;
}

}  // namespace asr::app
