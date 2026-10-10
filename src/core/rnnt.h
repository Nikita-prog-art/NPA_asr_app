// RNN-Transducer head: prediction network (embedding + LSTM) and joint network,
// plus the frame-synchronous greedy decoder of the reference implementation.
//
// The head is tiny (a few hundred thousand weights), so it runs in plain C++
// instead of a ggml graph: one graph per emitted symbol would be far slower.
// All weights are read from the GGUF in row-major [out][in] order.
#pragma once

#include <string>
#include <vector>

#include "core/model.h"

namespace asr {

class RnntDecoder {
public:
    RnntDecoder() = default;

    bool init(const Model & model, std::string * err);

    // Clears the prediction-network state (as if decoding started from scratch).
    void reset();

    // Greedy-decodes encoder frames [t0, t1) of `enc` ([d_model, T], channel
    // major) appending emitted token ids to `tokens`.
    // `token_frames`, when given, receives the encoder frame index of every
    // emitted token (used for streaming local agreement).
    void decode_frames(const float * enc, int d_model, int T, int t0, int t1,
                       std::vector<int> & tokens, std::vector<int> * token_frames = nullptr);

    int blank_id() const { return blank_; }
    int num_classes() const { return classes_; }
    int max_symbols() const { return max_symbols_; }
    int pred_hidden() const { return hidden_; }

    // Debug: joint logits (num_classes) of the very first decoding step.
    const std::vector<float> & first_step_logits() const { return first_logits_; }
    const std::vector<float> & first_step_g() const { return first_g_; }
    const std::vector<float> & first_enc_proj() const { return first_enc_proj_; }
    const std::vector<float> & first_pred_proj() const { return first_pred_proj_; }
    const std::vector<float> & first_joint_in() const { return first_joint_in_; }

private:
    void lstm_step(int label, std::vector<float> & g);
    void matvec(const std::vector<float> & w, const float * x, int stride, int n_in, int n_out,
                const std::vector<float> & bias, float * y) const;

    std::vector<float> embed_, w_ih_, w_hh_, b_ih_, b_hh_;
    std::vector<float> enc_w_, enc_b_, pred_w_, pred_b_, out_w_, out_b_;
    int hidden_ = 320, classes_ = 1025, blank_ = 1024, max_symbols_ = 10;
    int d_model_ = 768;

    // state
    std::vector<float> h_, c_, h_new_, c_new_;
    std::vector<float> g_, enc_proj_, pred_proj_, joint_in_, logits_;
    int last_label_ = -1;
    bool g_valid_ = false;
    bool first_step_done_ = false;
    std::vector<float> first_logits_;
    std::vector<float> first_g_, first_enc_proj_, first_pred_proj_, first_joint_in_;
};

}  // namespace asr
