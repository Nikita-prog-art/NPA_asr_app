// log-mel front-end, numerically matching
//   torchaudio.transforms.MelSpectrogram(...) + SpecScaler (log clamp)
// as used by GigaAM-v3's FeatureExtractor.
//
// Output layout is time-major: out[t * n_mels + m], ready to be handed to a
// ggml tensor with ne = (n_mels, n_frames, 1) ... (ne0 = n_mels).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/fft.h"

namespace asr {

struct MelConfig {
    int sample_rate = 16000;
    int n_mels      = 64;
    int n_fft       = 320;
    int win_length  = 320;
    int hop_length  = 160;
    bool center     = false;
};

class MelFrontend {
public:
    MelFrontend();
    ~MelFrontend();

    // `fb` points at n_freq * n_mels floats laid out ggml-style:
    // element (freq, mel) at fb[freq + mel * n_freq].  `window` holds
    // win_length floats.
    bool init(const MelConfig & cfg, const float * fb, const float * window, std::string * err);

    const MelConfig & config() const { return cfg_; }
    int n_freq() const { return cfg_.n_fft / 2 + 1; }

    // Number of frames produced for `n` input samples.
    int n_frames(int n) const;

    // Computes frames [first_frame, last_frame) of the signal `audio`.
    // out must hold (last_frame - first_frame) * n_mels floats.
    void compute_frames(const float * audio, int n_samples, int first_frame, int last_frame,
                        float * out) const;

    // Convenience: all frames.
    void compute(const float * audio, int n_samples, std::vector<float> & out, int * n_out) const;

private:
    MelConfig cfg_;
    std::vector<float> fb_;      // n_freq * n_mels, ggml layout
    std::vector<float> window_;  // n_fft, zero padded
    std::unique_ptr<Rfft> fft_;
    // scratch
    mutable std::vector<float> frame_buf_;
    mutable std::vector<std::complex<float>> spec_;
};

}  // namespace asr
