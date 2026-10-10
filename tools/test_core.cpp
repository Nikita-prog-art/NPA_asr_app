#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <thread>
#include <string>
#include <vector>

#include "audio/audio_file.h"
#include "core/encoder.h"
#include "core/fft.h"
#include "core/recognizer.h"
#include "core/mel.h"
#include "core/model.h"
#include "golden.h"

using namespace asr;

static int usage() {
    std::fprintf(stderr,
                 "usage: asr_test mel <model.gguf> <golden.bin>\n"
                 "       asr_test enc <model.gguf> <golden.bin> [threads]\n"
                 "       asr_test fft <N>\n"
                 "       asr_test conv1d\n"
                 "       asr_test run <model.gguf> <golden.bin> [threads]\n"
                 "       asr_test batch <model.gguf> <list.txt> [threads]\n"
                 "       asr_test bench <model.gguf> <audio.wav> [threads] [iters]\n");
    return 2;
}

struct Diff {
    double max_abs = 0;
    double mean_abs = 0;
    int64_t idx = -1;
    double got = 0, want = 0;
};

[[maybe_unused]] static void stats(const char * tag, const std::vector<float> & got,
                                   const float * want, size_t n) {
    double gmin = 1e30, gmax = -1e30, gsum = 0, wmin = 1e30, wmax = -1e30, wsum = 0;
    size_t gmaxi = 0, wmaxi = 0;
    for (size_t i = 0; i < n; ++i) {
        gsum += got[i]; wsum += want[i];
        if (got[i] < gmin) gmin = got[i];
        if (got[i] > gmax) { gmax = got[i]; gmaxi = i; }
        if (want[i] < wmin) wmin = want[i];
        if (want[i] > wmax) { wmax = want[i]; wmaxi = i; }
    }
    std::printf("  [%s] got min/max/mean %.4f/%.4f/%.4f (argmax %zu) | want %.4f/%.4f/%.4f (argmax %zu)\n",
                tag, gmin, gmax, gsum / n, gmaxi, wmin, wmax, wsum / n, wmaxi);
}

static Diff compare(const std::vector<float> & got, const float * want, size_t n) {
    Diff d;
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs(double(got[i]) - double(want[i]));
        sum += e;
        if (e > d.max_abs) {
            d.max_abs = e;
            d.idx = static_cast<int64_t>(i);
            d.got = got[i];
            d.want = want[i];
        }
    }
    d.mean_abs = sum / static_cast<double>(n);
    return d;
}

// Compares the C++ log-mel front-end against the PyTorch golden.
static int test_mel(const std::string & model_path, const std::string & gold_path) {
    test::Goldens g;
    if (!g.load(gold_path)) {
        std::fprintf(stderr, "cannot read goldens %s\n", gold_path.c_str());
        return 1;
    }
    Model model;
    std::string err;
    if (!model.load(model_path, &err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const HParams & hp = model.hp();
    std::printf("model: %s  d_model=%d layers=%d vocab=%zu\n", hp.model_name.c_str(),
                hp.enc_d_model, hp.enc_n_layers, hp.vocab.size());

    MelFrontend mel;
    MelConfig mc;
    mc.sample_rate = hp.sample_rate;
    mc.n_mels = hp.n_mels;
    mc.n_fft = hp.n_fft;
    mc.win_length = hp.win_length;
    mc.hop_length = hp.hop_length;
    mc.center = hp.center;
    const float * fb = model.f32("mel.fb");
    const float * win = model.f32("mel.window");
    if (!fb || !win) { std::fprintf(stderr, "mel.fb / mel.window missing\n"); return 1; }
    if (!mel.init(mc, fb, win, &err)) { std::fprintf(stderr, "mel init: %s\n", err.c_str()); return 1; }

    const auto * gwav = g.get("wav");
    const auto * gmel = g.get("mel");
    if (!gwav || !gmel) { std::fprintf(stderr, "goldens lack wav/mel\n"); return 1; }
    const float * wav = gwav->as_f32();
    const int n = static_cast<int>(gwav->numel());

    std::vector<float> out;
    int T = 0;
    mel.compute(wav, n, out, &T);

    const int gT = static_cast<int>(gmel->dims[2]);
    const int gM = static_cast<int>(gmel->dims[1]);
    std::printf("mel: got T=%d M=%d, golden T=%d M=%d\n", T, mc.n_mels, gT, gM);
    if (T != gT || mc.n_mels != gM) { std::fprintf(stderr, "shape mismatch\n"); return 1; }

    // golden is [1, n_mels, T] (torch layout)
    const float * gm = gmel->as_f32();
    double max_abs = 0.0, sum_abs = 0.0;
    int worst_t = -1, worst_m = -1;
    for (int t = 0; t < T; ++t) {
        for (int m = 0; m < gM; ++m) {
            const float a = out[static_cast<size_t>(t) * gM + m];
            const float b = gm[static_cast<size_t>(m) * T + t];
            const double d = std::fabs(double(a) - double(b));
            sum_abs += d;
            if (d > max_abs) { max_abs = d; worst_t = t; worst_m = m; }
        }
    }
    std::printf("mel diff: max_abs=%.3e mean_abs=%.3e (worst t=%d m=%d)\n", max_abs,
                sum_abs / (double(T) * gM), worst_t, worst_m);
    std::printf("mel sample got=%.6f golden=%.6f\n",
                out[static_cast<size_t>(worst_t) * gM + worst_m],
                gm[static_cast<size_t>(worst_m) * T + worst_t]);
    return max_abs < 2e-4 ? 0 : 1;
}

static int test_enc(const std::string & model_path, const std::string & gold_path, int threads) {
    test::Goldens g;
    if (!g.load(gold_path)) {
        std::fprintf(stderr, "cannot read goldens %s\n", gold_path.c_str());
        return 1;
    }
    Model model;
    std::string err;
    if (!model.load(model_path, &err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    if (threads > 0) model.set_n_threads(threads);

    Encoder enc;
    if (!enc.init(model, &err)) {
        std::fprintf(stderr, "encoder init: %s\n", err.c_str());
        return 1;
    }

    const auto * gmel = g.get("mel");
    const auto * genc = g.get("enc");
    if (!gmel || !genc) { std::fprintf(stderr, "goldens lack mel/enc\n"); return 1; }
    const int gT = static_cast<int>(gmel->dims[2]);
    const int gM = static_cast<int>(gmel->dims[1]);
    const int gTe = static_cast<int>(genc->dims[2]);
    const int gD = static_cast<int>(genc->dims[1]);

    // golden mel is [1, n_mels, T]; the encoder wants time-major [T, n_mels]
    std::vector<float> mel(static_cast<size_t>(gT) * gM);
    const float * gm = gmel->as_f32();
    for (int t = 0; t < gT; ++t)
        for (int m = 0; m < gM; ++m) mel[static_cast<size_t>(t) * gM + m] = gm[static_cast<size_t>(m) * gT + t];

    std::vector<std::pair<std::string, std::vector<float>>> dumps;
    std::vector<float> out;
    int T = 0;
    const bool want_dumps = g.has("layer0");
    if (!enc.run(mel.data(), gT, out, &T, &err, want_dumps ? &dumps : nullptr)) {
        std::fprintf(stderr, "encoder run: %s\n", err.c_str());
        return 1;
    }
    if (T != gTe || gD != model.hp().enc_d_model) {
        std::fprintf(stderr, "encoder shape mismatch: T=%d (want %d)\n", T, gTe);
        return 1;
    }
    // the golden `enc` is [1, D, T] (torch layout), `out` is [D, T] with D
    // fastest: reorder the golden into the ggml layout before comparing
    std::vector<float> enc_want(out.size());
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < gD; ++c)
            enc_want[static_cast<size_t>(c) + static_cast<size_t>(t) * gD] =
                genc->as_f32()[static_cast<size_t>(c) * T + t];
    Diff d = compare(out, enc_want.data(), out.size());
    std::printf("enc  : T=%d D=%d  max_abs=%.3e mean_abs=%.3e at %lld (got %.5f want %.5f)\n", T, gD,
                d.max_abs, d.mean_abs, (long long) d.idx, d.got, d.want);
    int rc = d.max_abs < 2e-2 ? 0 : 1;

    if (want_dumps) {
        for (auto & [name, buf] : dumps) {
            const auto * gl = g.get(name);
            if (!gl) continue;
            if (gl->numel() != buf.size()) {
                std::printf("  %-8s SIZE MISMATCH got %zu want %zu\n", name.c_str(), buf.size(), gl->numel());
                rc = 1;
                continue;
            }
            Diff dl = compare(buf, gl->as_f32(), buf.size());
            std::printf("  %-8s max_abs=%.3e mean_abs=%.3e at %lld (got %.5f want %.5f)%s\n", name.c_str(),
                        dl.max_abs, dl.mean_abs, (long long) dl.idx, dl.got, dl.want,
                        dl.max_abs > 2e-2 ? "   <-- MISMATCH" : "");
            if (dl.max_abs > 2e-2) rc = 1;
        }
    }
    return rc;
}

// Minimal ggml conv1d check against a naive implementation.
static int test_conv1d() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    const int K = 3, IC = 2, OC = 3, T = 7, S = 1, P = 1;
    ggml_init_params ip{};
    ip.mem_size = 4 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, IC, OC);
    ggml_tensor * in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, IC, 1);
    ggml_tensor * y = ggml_conv_1d(ctx, w, in, S, P, 1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { std::fprintf(stderr, "alloc failed\n"); return 1; }

    std::vector<float> wv(K * IC * OC), iv(T * IC);
    for (size_t i = 0; i < wv.size(); ++i) wv[i] = 0.1f * float(i + 1);
    for (size_t i = 0; i < iv.size(); ++i) iv[i] = 0.5f * float(i % 5) - 1.0f;
    ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * 4);
    ggml_backend_tensor_set(in, iv.data(), 0, iv.size() * 4);
    ggml_backend_graph_compute(be, gf);

    const int OW = (T + 2 * P - K) / S + 1;
    std::vector<float> got(OW * OC);
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * 4);
    std::printf("conv1d out ne=(%lld,%lld,%lld) OW=%d\n", (long long) y->ne[0], (long long) y->ne[1],
                (long long) y->ne[2], OW);

    double maxe = 0;
    for (int ow = 0; ow < OW; ++ow) {
        for (int oc = 0; oc < OC; ++oc) {
            float acc = 0;
            for (int k = 0; k < K; ++k) {
                const int t = ow * S + k - P;
                if (t < 0 || t >= T) continue;
                for (int ic = 0; ic < IC; ++ic) {
                    acc += wv[k + ic * K + oc * K * IC] * iv[t + ic * T];
                }
            }
            const float g = got[ow + oc * OW];
            const double e = std::fabs(g - acc);
            maxe = std::max(maxe, e);
            if (ow < 2) std::printf("  ow=%d oc=%d got=%.5f naive=%.5f\n", ow, oc, g, acc);
        }
    }
    std::printf("conv1d max err = %.3e %s (ggml conv_1d uses an f16 im2col)\n", maxe, maxe < 5e-3 ? "OK" : "MISMATCH");

    // ---- depthwise conv
    {
        const int KD = 3, CD = 4, TD = 6, PD = 1;
        ggml_context * c2 = ggml_init(ip);
        ggml_tensor * wd = ggml_new_tensor_3d(c2, GGML_TYPE_F32, KD, 1, CD);
        ggml_tensor * ind = ggml_new_tensor_3d(c2, GGML_TYPE_F32, TD, CD, 1);
        ggml_tensor * yd = ggml_conv_1d_dw(c2, wd, ind, 1, PD, 1);
        ggml_cgraph * g2 = ggml_new_graph(c2);
        ggml_build_forward_expand(g2, yd);
        ggml_gallocr_t a2 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(a2, g2);
        std::vector<float> wdv(KD * CD), idv(TD * CD);
        for (size_t i = 0; i < wdv.size(); ++i) wdv[i] = 0.2f * float(i + 1);
        for (size_t i = 0; i < idv.size(); ++i) idv[i] = 0.5f * float(i % 7) - 1.5f;
        ggml_backend_tensor_set(wd, wdv.data(), 0, wdv.size() * 4);
        ggml_backend_tensor_set(ind, idv.data(), 0, idv.size() * 4);
        ggml_backend_graph_compute(be, g2);
        const int OWD = (TD + 2 * PD - KD) + 1;
        std::vector<float> gotd(OWD * CD);
        ggml_backend_tensor_get(yd, gotd.data(), 0, gotd.size() * 4);
        std::printf("conv1d_dw out ne=(%lld,%lld,%lld)\n", (long long) yd->ne[0], (long long) yd->ne[1],
                    (long long) yd->ne[2]);
        double emax = 0;
        for (int ow = 0; ow < OWD; ++ow) {
            for (int ch = 0; ch < CD; ++ch) {
                float acc = 0;
                for (int k = 0; k < KD; ++k) {
                    const int t = ow + k - PD;
                    if (t < 0 || t >= TD) continue;
                    acc += wdv[k + ch * KD] * idv[t + ch * TD];
                }
                const float gv = gotd[ow + ch * OWD];
                if (ow < 2) std::printf("  dw ow=%d ch=%d got=%.5f naive=%.5f\n", ow, ch, gv, acc);
                emax = std::max(emax, std::fabs(double(gv) - acc));
            }
        }
        std::printf("conv1d_dw max err = %.3e %s\n", emax, emax < 5e-3 ? "OK" : "MISMATCH");
        ggml_gallocr_free(a2);
        ggml_free(c2);
        if (emax >= 5e-3) maxe = 1e9;
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(be);
    return maxe < 1e-4 ? 0 : 1;
}

// Full pipeline: mel -> encoder -> RNN-T -> text, offline and streaming.
static int test_run(const std::string & model_path, const std::string & gold_path, int threads) {
    test::Goldens g;
    if (!g.load(gold_path)) {
        std::fprintf(stderr, "cannot read goldens %s\n", gold_path.c_str());
        return 1;
    }
    const auto * gwav = g.get("wav");
    const auto * gtok = g.get("tokens");
    if (!gwav || !gtok) { std::fprintf(stderr, "goldens lack wav/tokens\n"); return 1; }
    const float * wav = gwav->as_f32();
    const int n = static_cast<int>(gwav->numel());

    RecognizerConfig cfg;
    cfg.model_path = model_path;
    cfg.n_threads = threads;
    Recognizer rec;
    std::string err;
    if (!rec.init(cfg, &err)) {
        std::fprintf(stderr, "recognizer init: %s\n", err.c_str());
        return 1;
    }
    const HParams & hp = rec.hp();
    std::printf("model %s: %d threads, %zu vocab\n", hp.model_name.c_str(), rec.n_threads(),
                rec.tokenizer().size());

    int rc = 0;
    Transcription off;
    if (!rec.transcribe(wav, n, &off, &err)) {
        std::fprintf(stderr, "transcribe: %s\n", err.c_str());
        return 1;
    }
    std::printf("offline : %.2fs audio, %.1f ms (%.1fx realtime) -> \"%s\"\n", off.audio_sec,
                off.infer_ms, off.audio_sec * 1000.0 / std::max(off.infer_ms, 0.001), off.text.c_str());

    if (const auto * gg = g.get("g0")) {
        const std::vector<float> & gv = rec.rnnt().first_step_g();
        if (gv.size() == gg->numel()) {
            Diff dg = compare(gv, gg->as_f32(), gv.size());
            std::printf("g0      : max_abs=%.3e mean_abs=%.3e\n", dg.max_abs, dg.mean_abs);
        }
    }
    auto cmp_vec = [&](const char * name, const std::vector<float> & v) {
        const auto * gv = g.get(name);
        if (!gv || gv->numel() != v.size()) return;
        Diff dd = compare(v, gv->as_f32(), v.size());
        std::printf("%-9s: max_abs=%.3e mean_abs=%.3e\n", name, dd.max_abs, dd.mean_abs);
    };
    cmp_vec("encproj0", rec.rnnt().first_enc_proj());
    cmp_vec("predproj0", rec.rnnt().first_pred_proj());
    cmp_vec("jointin0", rec.rnnt().first_joint_in());
    if (const auto * gl = g.get("logits0_raw")) {
        const std::vector<float> & lg = rec.rnnt().first_step_logits();
        if (lg.size() == gl->numel()) {
            Diff dl = compare(lg, gl->as_f32(), lg.size());
            int am = 0;
            for (size_t i = 1; i < lg.size(); ++i) if (lg[i] > lg[am]) am = (int) i;
            int wm = 0;
            for (size_t i = 1; i < gl->numel(); ++i) if (gl->as_f32()[i] > gl->as_f32()[wm]) wm = (int) i;
            std::printf("logits0 : max_abs=%.3e argmax got=%d want=%d  (got[%d]=%.3f want=%.3f)\n",
                        dl.max_abs, am, wm, am, lg[am], gl->as_f32()[am]);
            std::vector<float> gv(lg), wv(gl->as_f32(), gl->as_f32() + gl->numel());
            for (int k = 0; k < 5; ++k) {
                int bi = 0, wi = 0;
                for (size_t i = 1; i < gv.size(); ++i) if (gv[i] > gv[bi]) bi = (int) i;
                for (size_t i = 1; i < wv.size(); ++i) if (wv[i] > wv[wi]) wi = (int) i;
                std::printf("   top%d got #%d=%.3f | want #%d=%.3f\n", k, bi, gv[bi], wi, wv[wi]);
                gv[bi] = -1e30f;
                wv[wi] = -1e30f;
            }
        }
    }

    const int64_t * want = gtok->as_i64();
    const size_t nw = gtok->numel();
    const bool tokens_ok = off.tokens.size() == nw &&
                           std::equal(off.tokens.begin(), off.tokens.end(), want);
    std::printf("tokens  : got %zu, want %zu -> %s\n", off.tokens.size(), nw,
                tokens_ok ? "IDENTICAL" : "DIFFERENT");
    if (!tokens_ok) {
        std::printf("  got : ");
        for (size_t i = 0; i < off.tokens.size() && i < 40; ++i) std::printf("%d ", off.tokens[i]);
        std::printf("\n  want: ");
        for (size_t i = 0; i < nw && i < 40; ++i) std::printf("%lld ", (long long) want[i]);
        std::printf("\n");
        rc = 1;
    }

    // streaming: feed 100 ms chunks like the microphone would
    RecognizerConfig scfg = cfg;
    scfg.step_sec = 0.5f;
    scfg.min_window_sec = 0.4f;
    Recognizer srec;
    if (!srec.init(scfg, &err)) {
        std::fprintf(stderr, "stream init: %s\n", err.c_str());
        return 1;
    }
    srec.begin();
    const int chunk = hp.sample_rate / 10;
    std::string last;
    int updates = 0;
    for (int off2 = 0; off2 < n; off2 += chunk) {
        const int m = std::min(chunk, n - off2);
        srec.feed(wav + off2, m);
        const std::string cur = srec.current_text();
        if (cur != last) { last = cur; ++updates; }
    }
    Transcription st;
    if (!srec.finish(&st, &err)) {
        std::fprintf(stderr, "stream finish: %s\n", err.c_str());
        return 1;
    }
    std::printf("stream  : %.1f ms final pass, %d text updates, text \"%s\"\n", st.infer_ms, updates,
                st.text.c_str());
    if (st.text != off.text) {
        std::printf("  NOTE: streaming text differs from the offline text\n");
    }
    return rc;
}

static int test_fft(int n) {
    Rfft fft(n);
    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) in[i] = std::sin(0.1 * i) + 0.3 * std::cos(0.37 * i) + 0.01 * i;
    std::vector<std::complex<float>> out(fft.n_bins());
    fft.forward(in.data(), out.data());
    std::vector<double> ind(in.begin(), in.end());
    auto ref = dft_naive(ind);
    double maxe = 0;
    for (int k = 0; k < fft.n_bins(); ++k) {
        maxe = std::max(maxe, std::fabs(double(out[k].real()) - ref[k].real()));
        maxe = std::max(maxe, std::fabs(double(out[k].imag()) - ref[k].imag()));
    }
    std::printf("fft n=%d (%s): max abs err vs naive DFT = %.3e\n", n, fft_backend_name().c_str(),
                maxe);
    return maxe < 1e-3 ? 0 : 1;
}

// Transcribes a list of audio files with one loaded model and prints
// "<path>\t<text>" per line.  Loading the model costs more than a short
// utterance, so a quality comparison across models has to batch: this is what
// the quantization study uses, and it keeps the listening test to one load.
static int test_batch(const std::string & model_path, const std::string & list_path, int threads) {
    std::ifstream list(list_path);
    if (!list) {
        std::fprintf(stderr, "cannot read list %s\n", list_path.c_str());
        return 1;
    }
    RecognizerConfig cfg;
    cfg.model_path = model_path;
    cfg.n_threads = threads;
    Recognizer rec;
    std::string err;
    if (!rec.init(cfg, &err)) {
        std::fprintf(stderr, "recognizer init: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "batch: %s, %d threads\n", rec.hp().model_name.c_str(), rec.n_threads());

    std::string line;
    int n_files = 0;
    int n_failed = 0;
    double audio_sec = 0.0;
    double infer_ms = 0.0;
    while (std::getline(list, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::vector<float> pcm;
        if (!audio::load_file(line, 16000, pcm, &err)) {
            std::fprintf(stderr, "skip %s: %s\n", line.c_str(), err.c_str());
            ++n_failed;
            continue;
        }
        Transcription tr;
        if (!rec.transcribe(pcm.data(), static_cast<int>(pcm.size()), &tr, &err)) {
            std::fprintf(stderr, "skip %s: %s\n", line.c_str(), err.c_str());
            ++n_failed;
            continue;
        }
        std::string text;
        for (char c : tr.text) text += (c == '\t' || c == '\n') ? ' ' : c;
        std::printf("%s\t%s\n", line.c_str(), text.c_str());
        audio_sec += tr.audio_sec;
        infer_ms += tr.infer_ms;
        ++n_files;
    }
    std::fprintf(stderr, "batch: %d files (%.1f s audio) in %.1f s, %d failed\n", n_files, audio_sec,
                 infer_ms / 1000.0, n_failed);
    return n_files > 0 ? 0 : 1;
}

// Timing benchmark: mel front-end, encoder and the whole pipeline, for every
// available FFT back-end.
static int test_bench(const std::string & model_path, const std::string & wav_path, int threads,
                      int iters) {
    std::vector<float> pcm;
    std::string err;
    if (!audio::load_file(wav_path, 16000, pcm, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const double audio_sec = double(pcm.size()) / 16000.0;
    std::printf("bench: %.2f s of audio, %d threads, %d iterations\n", audio_sec,
                threads > 0 ? threads : default_n_threads(), iters);

    // A single FFT implementation now (the built-in Bluestein): this used to
    // loop over "oneMKL" and "internal".
    {
        RecognizerConfig cfg;
        cfg.model_path = model_path;
        cfg.n_threads = threads;
        Recognizer rec;
        if (!rec.init(cfg, &err)) {
            std::fprintf(stderr, "init: %s\n", err.c_str());
            return 1;
        }
        // mel front-end alone
        const HParams & hp = rec.hp();
        MelFrontend mel;
        MelConfig mc;
        mc.sample_rate = hp.sample_rate; mc.n_mels = hp.n_mels; mc.n_fft = hp.n_fft;
        mc.win_length = hp.win_length; mc.hop_length = hp.hop_length; mc.center = hp.center;
        Model model;
        if (!model.load(model_path, &err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
        if (!mel.init(mc, model.f32("mel.fb"), model.f32("mel.window"), &err)) return 1;
        std::vector<float> mout;
        int mframes = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) mel.compute(pcm.data(), (int) pcm.size(), mout, &mframes);
        auto t1 = std::chrono::steady_clock::now();
        const double mel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        Encoder enc;
        if (!enc.init(model, &err)) return 1;
        enc.set_n_threads(cfg.n_threads);
        std::vector<float> eout;
        int eT = 0;
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) enc.run(mout.data(), mframes, eout, &eT, &err);
        t1 = std::chrono::steady_clock::now();
        const double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        Transcription tr;
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) rec.transcribe(pcm.data(), (int) pcm.size(), &tr, &err);
        t1 = std::chrono::steady_clock::now();
        const double all_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        std::printf("  fft=%-18s mel %6.1f ms (%4.1f%%)  enc %7.1f ms (%4.1f%%)  total %7.1f ms  %5.1fx RT\n",
                    fft_backend_name().c_str(), mel_ms, 100.0 * mel_ms / all_ms, enc_ms,
                    100.0 * enc_ms / all_ms, all_ms, audio_sec * 1000.0 / all_ms);
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    if (cmd == "mel" && argc >= 4) return test_mel(argv[2], argv[3]);
    if (cmd == "enc" && argc >= 4) return test_enc(argv[2], argv[3], argc >= 5 ? std::atoi(argv[4]) : 0);
    if (cmd == "fft" && argc >= 3) return test_fft(std::atoi(argv[2]));
    if (cmd == "conv1d") return test_conv1d();
    if (cmd == "run" && argc >= 4) return test_run(argv[2], argv[3], argc >= 5 ? std::atoi(argv[4]) : 0);
    if (cmd == "batch" && argc >= 4)
        return test_batch(argv[2], argv[3], argc >= 5 ? std::atoi(argv[4]) : 0);
    if (cmd == "bench" && argc >= 4)
        return test_bench(argv[2], argv[3], argc >= 5 ? std::atoi(argv[4]) : 0,
                          argc >= 6 ? std::atoi(argv[5]) : 5);
    return usage();
}
