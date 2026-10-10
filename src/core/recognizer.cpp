#include "core/recognizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

namespace asr {

namespace {
double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// longest common prefix length
size_t lcp(const std::vector<int> & a, const std::vector<int> & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}
}  // namespace

Recognizer::Recognizer() = default;
Recognizer::~Recognizer() = default;

bool Recognizer::init(const RecognizerConfig & cfg, std::string * err) {
    cfg_ = cfg;
    if (cfg_.n_threads <= 0) {
        cfg_.n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (cfg_.n_threads <= 0) cfg_.n_threads = 4;
    }
    if (!model_.load(cfg_.model_path, err)) return false;
    model_.set_n_threads(cfg_.n_threads);

    const HParams & hp = model_.hp();
    const float * fb = model_.f32("mel.fb");
    const float * win = model_.f32("mel.window");
    if (!fb || !win) {
        if (err) *err = "model is missing the mel filterbank / window";
        return false;
    }
    MelConfig mc;
    mc.sample_rate = hp.sample_rate;
    mc.n_mels = hp.n_mels;
    mc.n_fft = hp.n_fft;
    mc.win_length = hp.win_length;
    mc.hop_length = hp.hop_length;
    mc.center = hp.center;
    if (!mel_.init(mc, fb, win, err)) return false;
    if (!encoder_.init(model_, err)) return false;
    if (!rnnt_.init(model_, err)) return false;
    tokenizer_ = Tokenizer(hp.vocab);
    loaded_ = true;
    return true;
}

void Recognizer::set_n_threads(int n) {
    if (n <= 0) return;
    cfg_.n_threads = n;
    model_.set_n_threads(n);
}

// ---------------------------------------------------------------- mel cache

void Recognizer::update_mel() {
    const int hop = mel_.config().hop_length;
    const int win = mel_.config().win_length;
    const int n = static_cast<int>(buffer_.size());
    // only frames fully covered by the available audio
    const int frames = n < win ? 0 : (n - win) / hop + 1;
    if (frames <= mel_frames_) return;
    const int first = mel_frames_;
    const int count = frames - first;
    mel_cache_.resize(static_cast<size_t>(frames) * mel_.config().n_mels);
    mel_.compute_frames(buffer_.data(), n, first, frames, mel_cache_.data() + static_cast<size_t>(first) * mel_.config().n_mels);
    mel_frames_ = frames;
    mel_covered_samples_ = n;
}

// ---------------------------------------------------------------- one window

bool Recognizer::run_window(int f0, int f1, std::vector<int> * tokens, std::vector<int> * frames,
                            double * ms, std::string * err) {
    const HParams & hp = model_.hp();
    if (f1 - f0 < hp.enc_subs_kernel * 2) {
        if (err) *err = "not enough audio for the encoder";
        return false;
    }
    const double t0 = now_ms();
    std::vector<float> enc;
    int T = 0;
    const float * mel_slice = mel_cache_.data() + static_cast<size_t>(f0) * hp.n_mels;
    if (!encoder_.run(mel_slice, f1 - f0, enc, &T, err)) return false;

    rnnt_.reset();
    tokens->clear();
    std::vector<int> local_frames;
    rnnt_.decode_frames(enc.data(), hp.enc_d_model, T, 0, T, *tokens, &local_frames);
    if (frames) {
        frames->clear();
        frames->reserve(local_frames.size());
        const int factor = hp.enc_subs_factor;
        for (int t : local_frames) frames->push_back(f0 + t * factor);
    }
    if (ms) *ms = now_ms() - t0;
    return true;
}

// ---------------------------------------------------------------- offline

bool Recognizer::transcribe(const float * pcm, int n_samples, Transcription * out, std::string * err) {
    if (!loaded_) {
        if (err) *err = "model not loaded";
        return false;
    }
    buffer_.assign(pcm, pcm + n_samples);
    mel_cache_.clear();
    mel_frames_ = 0;
    update_mel();
    if (mel_frames_ <= 0) {
        if (err) *err = "audio too short";
        return false;
    }
    std::vector<int> tokens, frames;
    double ms = 0;
    if (!run_window(0, mel_frames_, &tokens, &frames, &ms, err)) return false;

    if (out) {
        out->tokens = tokens;
        out->text = tokenizer_.decode(tokens);
        out->audio_sec = static_cast<double>(n_samples) / hp().sample_rate;
        out->infer_ms = ms;
        out->n_windows = 1;
    }
    return true;
}

// ---------------------------------------------------------------- streaming

void Recognizer::begin() {
    buffer_.clear();
    mel_cache_.clear();
    mel_frames_ = 0;
    mel_covered_samples_ = 0;
    last_run_samples_ = 0;
    window_f0_ = 0;
    committed_.clear();
    hypothesis_.clear();
    prev_tokens_.clear();
    n_windows_ = 0;
    last_infer_ms_ = 0.0;
    recording_ = true;
}

void Recognizer::feed(const float * pcm, int n_samples) {
    if (!recording_ || n_samples <= 0) return;
    buffer_.insert(buffer_.end(), pcm, pcm + n_samples);
    update_mel();

    const int rate = hp().sample_rate;
    const double new_sec = static_cast<double>(static_cast<int>(buffer_.size()) - last_run_samples_) / rate;
    if (audio_seconds() < cfg_.min_window_sec) return;
    if (new_sec >= cfg_.step_sec) streaming_step(false);
}

void Recognizer::streaming_step(bool force) {
    const HParams & hp = model_.hp();
    const int rate = hp.sample_rate;
    const int hop = mel_.config().hop_length;

    const int total_samples = static_cast<int>(buffer_.size());
    const int win_samples = static_cast<int>(cfg_.max_window_sec * rate);
    const int s0 = std::max(0, total_samples - win_samples);
    const int f0 = s0 / hop;
    const int f1 = mel_frames_;
    if (f1 - f0 < hp.enc_subs_kernel * 2) return;

    std::vector<int> tokens, frames;
    double ms = 0;
    std::string err;
    if (!run_window(f0, f1, &tokens, &frames, &ms, &err)) return;
    n_windows_++;
    last_infer_ms_ = ms;
    last_run_samples_ = total_samples;

    // absolute mel frame per token
    std::vector<TokFrame> cur;
    cur.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        cur.push_back(TokFrame{frames[i], tokens[i]});
    }

    // If the window is about to slide, everything before the new window start
    // must already be committed, otherwise it would be lost.
    const int next_f0 = std::max(0, (total_samples + static_cast<int>(cfg_.step_sec * rate) - win_samples)) / hop;
    const int guard = static_cast<int>(cfg_.commit_guard_sec * rate) / hop;

    // LocalAgreement-2: commit the prefix that agrees with the previous window
    std::vector<int> prev_vals, cur_vals;
    for (const auto & t : hypothesis_) prev_vals.push_back(t.token);
    for (const auto & t : cur) cur_vals.push_back(t.token);
    size_t agree = lcp(prev_vals, cur_vals);

    // tokens that the sliding window is about to drop must be committed now
    size_t must = 0;
    while (must < cur.size() && cur[must].frame < next_f0 + guard) ++must;
    const size_t commit_n = std::max(agree, must);

    if (commit_n > committed_.size() && commit_n <= cur.size()) {
        committed_.assign(cur.begin(), cur.begin() + static_cast<long>(commit_n));
    }
    // drop committed/hypothesis tokens that fall out of the next window
    if (next_f0 > window_f0_) {
        auto drop_before = [&](std::vector<TokFrame> & v) {
            v.erase(std::remove_if(v.begin(), v.end(),
                                   [&](const TokFrame & t) { return t.frame < next_f0; }),
                    v.end());
        };
        drop_before(committed_);
        drop_before(cur);
        window_f0_ = next_f0;
    }
    hypothesis_ = cur;
    (void) force;
}

std::string Recognizer::decode_tokens(const std::vector<TokFrame> & t) const {
    std::vector<int> ids;
    ids.reserve(t.size());
    for (const auto & x : t) ids.push_back(x.token);
    return tokenizer_.decode(ids);
}

void Recognizer::snapshot(std::string * committed, std::string * provisional) const {
    if (committed) *committed = decode_tokens(committed_);
    if (provisional) {
        std::vector<TokFrame> rest;
        if (hypothesis_.size() > committed_.size()) {
            rest.assign(hypothesis_.begin() + static_cast<long>(committed_.size()), hypothesis_.end());
        }
        *provisional = decode_tokens(rest);
    }
}

std::string Recognizer::current_text() const {
    std::string a, b;
    snapshot(&a, &b);
    return a + b;
}

bool Recognizer::finish(Transcription * out, std::string * err) {
    recording_ = false;
    const HParams & hp = model_.hp();
    if (buffer_.empty()) {
        if (out) *out = Transcription{};
        return true;
    }
    update_mel();
    const int total_samples = static_cast<int>(buffer_.size());
    const int win_samples = static_cast<int>(cfg_.max_window_sec * hp.sample_rate);
    const int s0 = std::max(0, total_samples - win_samples);
    const int f0 = s0 / mel_.config().hop_length;
    const int f1 = mel_frames_;
    if (f1 - f0 < hp.enc_subs_kernel * 2) {
        if (err) *err = "audio too short";
        if (out) *out = Transcription{};
        return false;
    }

    std::vector<int> tokens, frames;
    double ms = 0;
    if (!run_window(f0, f1, &tokens, &frames, &ms, err)) return false;

    // When the final window covers the whole recording (the usual push-to-talk
    // case) the exact full-context decode is used as is.  For very long
    // dictations the already committed prefix is kept and the final window
    // supplies the tail.
    std::vector<int> all;
    if (f0 == 0) {
        all = tokens;
    } else {
        for (const auto & t : committed_) {
            if (t.frame < f0) all.push_back(t.token);
        }
        all.insert(all.end(), tokens.begin(), tokens.end());
    }

    if (out) {
        out->tokens = all;
        out->text = tokenizer_.decode(all);
        out->audio_sec = static_cast<double>(total_samples) / hp.sample_rate;
        out->infer_ms = ms;
        out->n_windows = n_windows_ + 1;
    }
    return true;
}

}  // namespace asr
