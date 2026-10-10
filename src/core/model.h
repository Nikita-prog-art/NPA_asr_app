// GGUF model container for GigaAM-v3 (ggml backend).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace asr {

struct HParams {
    std::string model_name;
    // front-end
    int  sample_rate = 16000;
    int  n_mels      = 64;
    int  n_fft       = 320;
    int  win_length  = 320;
    int  hop_length  = 160;
    bool center      = false;
    // encoder
    int   enc_n_layers       = 16;
    int   enc_d_model        = 768;
    int   enc_n_heads        = 16;
    int   enc_d_ff           = 3072;
    int   enc_conv_kernel    = 5;
    int   enc_subs_kernel    = 5;
    int   enc_subs_factor    = 4;
    int   enc_pos_emb_max_len = 5000;
    float enc_norm_eps       = 1e-5f;
    // decoder / joint
    int dec_pred_hidden   = 320;
    int dec_pred_layers   = 1;
    int dec_num_classes   = 1025;
    int dec_blank_id      = 1024;
    int dec_joint_hidden  = 320;
    int dec_max_symbols   = 10;
    // tokenizer
    std::vector<std::string> vocab;
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    bool load(const std::string & path, std::string * err);

    const HParams & hp() const { return hp_; }

    ggml_backend_t backend() const { return backend_; }
    void set_n_threads(int n);

    // Raw tensor (may be f16/q8_0/...). Returns nullptr when missing.
    ggml_tensor * tensor(const std::string & name) const;

    // Reads a (small) tensor into a float32 host buffer; cached.  Returns
    // nullptr when the tensor is missing.
    const float * f32(const std::string & name) const;

    // Reads a tensor into `out` (converted to f32), returns false when missing.
    bool read_f32(const std::string & name, std::vector<float> & out) const;

    const std::string & path() const { return path_; }
    size_t n_tensors() const { return tensors_.size(); }

private:
    HParams hp_;
    std::string path_;
    ggml_context * ctx_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
    mutable std::unordered_map<std::string, std::vector<float>> f32_cache_;
    int n_threads_ = 0;
};

}  // namespace asr
