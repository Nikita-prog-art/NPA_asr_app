#include "core/mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace asr {

MelFrontend::MelFrontend() = default;
MelFrontend::~MelFrontend() = default;

bool MelFrontend::init(const MelConfig & cfg, const float * fb, const float * window, std::string * err) {
    cfg_ = cfg;
    if (cfg_.n_fft < 2 || cfg_.n_mels <= 0 || cfg_.hop_length <= 0 || cfg_.win_length <= 0) {
        if (err) *err = "invalid mel configuration";
        return false;
    }
    if (cfg_.win_length > cfg_.n_fft) {
        if (err) *err = "win_length > n_fft is not supported";
        return false;
    }
    const int nfreq = n_freq();
    fb_.assign(fb, fb + static_cast<size_t>(nfreq) * cfg_.n_mels);

    // torchaudio pads the window to n_fft centred when win_length < n_fft
    window_.assign(cfg_.n_fft, 0.0f);
    const int off = (cfg_.n_fft - cfg_.win_length) / 2;
    std::copy(window, window + cfg_.win_length, window_.begin() + off);

    try {
        fft_ = std::make_unique<Rfft>(cfg_.n_fft);
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
    frame_buf_.resize(cfg_.n_fft);
    spec_.resize(nfreq);
    return true;
}

int MelFrontend::n_frames(int n) const {
    if (cfg_.center) return n / cfg_.hop_length + 1;
    if (n < cfg_.win_length) return 0;
    return (n - cfg_.win_length) / cfg_.hop_length + 1;
}

void MelFrontend::compute_frames(const float * audio, int n_samples, int first_frame, int last_frame,
                                 float * out) const {
    const int nfreq = n_freq();
    const int n_fft = cfg_.n_fft;
    const int hop = cfg_.hop_length;

    for (int t = first_frame; t < last_frame; ++t) {
        const int start = t * hop;
        // fill the frame, zero outside the signal (torch pads with zeros when
        // center=False and the frame crosses the end)
        for (int i = 0; i < n_fft; ++i) {
            // torch.stft extracts n_fft-long frames starting at t*hop (no extra
            // offset); when win_length < n_fft the window is zero padded on both
            // sides, which the prepared window_ already accounts for.
            const int idx = start + i;
            const float s = (idx >= 0 && idx < n_samples) ? audio[idx] : 0.0f;
            frame_buf_[i] = s * window_[i];
        }
        fft_->forward(frame_buf_.data(), spec_.data());
        float * dst = out + static_cast<size_t>(t - first_frame) * cfg_.n_mels;
        for (int m = 0; m < cfg_.n_mels; ++m) {
            float acc = 0.0f;
            const float * fbm = &fb_[static_cast<size_t>(m) * nfreq];
            for (int k = 0; k < nfreq; ++k) {
                const float re = spec_[k].real();
                const float im = spec_[k].imag();
                acc += fbm[k] * (re * re + im * im);
            }
            acc = std::min(std::max(acc, 1e-9f), 1e9f);
            dst[m] = std::log(acc);
        }
    }
}

void MelFrontend::compute(const float * audio, int n_samples, std::vector<float> & out, int * n_out) const {
    const int T = n_frames(n_samples);
    out.assign(static_cast<size_t>(std::max(T, 0)) * cfg_.n_mels, 0.0f);
    if (T > 0) compute_frames(audio, n_samples, 0, T, out.data());
    if (n_out) *n_out = T;
}

}  // namespace asr
