#include "core/rnnt.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace asr {

bool RnntDecoder::init(const Model & model, std::string * err) {
    const HParams & hp = model.hp();
    hidden_ = hp.dec_pred_hidden;
    classes_ = hp.dec_num_classes;
    blank_ = hp.dec_blank_id;
    max_symbols_ = hp.dec_max_symbols;
    d_model_ = hp.enc_d_model;

    struct Req {
        const char * name;
        std::vector<float> * dst;
    };
    const Req reqs[] = {
        {"dec.embed.weight", &embed_},   {"dec.lstm.w_ih", &w_ih_},  {"dec.lstm.w_hh", &w_hh_},
        {"dec.lstm.b_ih", &b_ih_},       {"dec.lstm.b_hh", &b_hh_},  {"joint.enc.weight", &enc_w_},
        {"joint.enc.bias", &enc_b_},     {"joint.pred.weight", &pred_w_}, {"joint.pred.bias", &pred_b_},
        {"joint.out.weight", &out_w_},   {"joint.out.bias", &out_b_},
    };
    for (const auto & r : reqs) {
        if (!model.read_f32(r.name, *r.dst)) {
            if (err) *err = std::string("missing tensor: ") + r.name;
            return false;
        }
    }
    if (static_cast<int>(embed_.size()) != classes_ * hidden_ ||
        static_cast<int>(w_ih_.size()) != 4 * hidden_ * hidden_ ||
        static_cast<int>(out_w_.size()) != classes_ * hp.dec_joint_hidden) {
        if (err) *err = "RNN-T head tensor shapes do not match the hyper-parameters";
        return false;
    }

    h_.assign(hidden_, 0.0f);
    c_.assign(hidden_, 0.0f);
    h_new_.assign(hidden_, 0.0f);
    c_new_.assign(hidden_, 0.0f);
    g_.assign(hidden_, 0.0f);
    enc_proj_.assign(hp.dec_joint_hidden, 0.0f);
    pred_proj_.assign(hp.dec_joint_hidden, 0.0f);
    joint_in_.assign(hp.dec_joint_hidden, 0.0f);
    logits_.assign(classes_, 0.0f);
    reset();
    return true;
}

void RnntDecoder::reset() {
    std::fill(h_.begin(), h_.end(), 0.0f);
    std::fill(c_.begin(), c_.end(), 0.0f);
    last_label_ = -1;
    g_valid_ = false;
    first_step_done_ = false;
    first_logits_.clear();
}

// y = W x + b, with W row-major [n_out][n_in] and x sampled with `stride`
void RnntDecoder::matvec(const std::vector<float> & w, const float * x, int stride, int n_in,
                         int n_out, const std::vector<float> & bias, float * y) const {
    for (int o = 0; o < n_out; ++o) {
        const float * row = w.data() + static_cast<size_t>(o) * n_in;
        float acc = bias.empty() ? 0.0f : bias[static_cast<size_t>(o)];
        for (int i = 0; i < n_in; ++i) acc += row[i] * x[static_cast<size_t>(i) * stride];
        y[o] = acc;
    }
}

void RnntDecoder::lstm_step(int label, std::vector<float> & g) {
    // input vector: embedding of `label`, or zeros for the initial step
    // (the reference calls predict(None, None) first, which feeds zeros)
    const float * emb = nullptr;
    std::vector<float> zeros;
    if (label < 0) {
        zeros.assign(hidden_, 0.0f);
        emb = zeros.data();
    } else {
        emb = embed_.data() + static_cast<size_t>(label) * hidden_;
    }

    // gates = W_ih x + b_ih + W_hh h + b_hh   (order i, f, g, o)
    std::vector<float> gates(4 * hidden_);
    for (int o = 0; o < 4 * hidden_; ++o) {
        const float * ri = w_ih_.data() + static_cast<size_t>(o) * hidden_;
        const float * rh = w_hh_.data() + static_cast<size_t>(o) * hidden_;
        float acc = b_ih_[static_cast<size_t>(o)] + b_hh_[static_cast<size_t>(o)];
        for (int i = 0; i < hidden_; ++i) acc += ri[i] * emb[i];
        for (int i = 0; i < hidden_; ++i) acc += rh[i] * h_[static_cast<size_t>(i)];
        gates[static_cast<size_t>(o)] = acc;
    }
    for (int i = 0; i < hidden_; ++i) {
        const float ii = 1.0f / (1.0f + std::exp(-gates[static_cast<size_t>(i)]));
        const float ff = 1.0f / (1.0f + std::exp(-gates[static_cast<size_t>(hidden_ + i)]));
        const float gg = std::tanh(gates[static_cast<size_t>(2 * hidden_ + i)]);
        const float oo = 1.0f / (1.0f + std::exp(-gates[static_cast<size_t>(3 * hidden_ + i)]));
        c_new_[static_cast<size_t>(i)] = ff * c_[static_cast<size_t>(i)] + ii * gg;
        h_new_[static_cast<size_t>(i)] = oo * std::tanh(c_new_[static_cast<size_t>(i)]);
    }
    g.assign(h_new_.begin(), h_new_.end());
}

void RnntDecoder::decode_frames(const float * enc, int d_model, int T, int t0, int t1,
                                std::vector<int> & tokens, std::vector<int> * token_frames) {
    const int jh = static_cast<int>(enc_proj_.size());
    const int n_in = d_model;

    for (int t = std::max(0, t0); t < std::min(T, t1); ++t) {
        // `enc` is [d_model, T] channel-major: frame t starts at enc + t*d_model
        matvec(enc_w_, enc + static_cast<size_t>(t) * d_model, 1, n_in, jh, enc_b_, enc_proj_.data());

        int new_symbols = 0;
        for (;;) {
            if (!g_valid_) {
                lstm_step(last_label_, g_);
                matvec(pred_w_, g_.data(), 1, hidden_, jh, pred_b_, pred_proj_.data());
                g_valid_ = true;
            }
            for (int i = 0; i < jh; ++i) {
                const float v = enc_proj_[static_cast<size_t>(i)] + pred_proj_[static_cast<size_t>(i)];
                joint_in_[static_cast<size_t>(i)] = v > 0.0f ? v : 0.0f;  // ReLU
            }
            matvec(out_w_, joint_in_.data(), 1, jh, classes_, out_b_, logits_.data());
            if (!first_step_done_) {
                first_logits_ = logits_;
                first_g_ = g_;
                first_enc_proj_ = enc_proj_;
                first_pred_proj_ = pred_proj_;
                first_joint_in_ = joint_in_;
                first_step_done_ = true;
            }

            int best = 0;
            float best_v = logits_[0];
            for (int c = 1; c < classes_; ++c) {
                if (logits_[static_cast<size_t>(c)] > best_v) {
                    best_v = logits_[static_cast<size_t>(c)];
                    best = c;
                }
            }
            if (best == blank_) break;

            tokens.push_back(best);
            if (token_frames) token_frames->push_back(t);
            // the state only advances when a symbol is emitted
            h_ = h_new_;
            c_ = c_new_;
            last_label_ = best;
            g_valid_ = false;
            if (++new_symbols >= max_symbols_) break;
        }
    }
}

}  // namespace asr
