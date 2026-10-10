// GigaAM-v3 Conformer encoder on top of ggml.
//
// Ports (verbatim, in ggml terms) the reference implementation:
//   StridingSubsampling (2x conv1d k=5 s=2 p=2 + ReLU)
//   16 x ConformerLayer: FFN/2, rotary MHA (RoPE applied to the *input* of the
//   q/k projections, NEOX pairing), depthwise conv module with LayerNorm
//   (conv_norm_type=layer_norm), FFN/2, LayerNorm out.
//
// Activations use ggml's channel-first layout [d_model, T] (ne0 = features), the
// layout ggml_norm and ggml_mul_mat expect; the depthwise part of the conv
// module is evaluated in the time-major layout [T, d_model] because
// ggml_conv_1d_dw wants it, with cheap explicit transposes around it.
#pragma once

#include <memory>
#include <utility>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/model.h"

namespace asr {

class Encoder {
public:
    Encoder();
    ~Encoder();
    Encoder(const Encoder &) = delete;
    Encoder & operator=(const Encoder &) = delete;

    bool init(const Model & model, std::string * err);

    // `mel` holds T_mel * n_mels floats, time-major (frame t at t*n_mels).
    // On success `out` holds T_enc * d_model floats, channel-major
    // (channel c at c*T_enc), matching the reference [1, d_model, T] tensor.
    // When `dumps` is non-null it receives named [T_enc, d_model] buffers
    // ("subs", "layer0".., plus layer-0 sub-stages) for numerical debugging.
    bool run(const float * mel, int T_mel, std::vector<float> & out, int * T_enc, std::string * err,
             std::vector<std::pair<std::string, std::vector<float>>> * dumps = nullptr);

    void set_n_threads(int n);

    // Number of encoder frames for a given number of mel frames.
    int enc_frames(int T_mel) const;

    // Subsampling-conv output length for one conv layer.
    static int conv_out_len(int n, int k, int s, int p);

private:
    ggml_tensor * w32(const std::string & name);          // 1-D f32 copy
    ggml_tensor * w32_2d(const std::string & name, int d0, int d1);
    ggml_tensor * w32_2d_auto(const std::string & name);
    void register_f32(const std::string & name);

    const Model * model_ = nullptr;
    ggml_context * wctx_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> w32_;
    ggml_gallocr_t galloc_ = nullptr;
    int n_threads_ = 0;
};

}  // namespace asr
