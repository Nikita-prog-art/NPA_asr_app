#include "core/encoder.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "ggml-alloc.h"

namespace asr {

namespace {

struct BuildError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Small helpers ------------------------------------------------------------

// ggml_add with a diagnostic instead of a bare GGML_ASSERT
ggml_tensor * add_bc(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, const char * tag) {
    if (!ggml_can_repeat(b, a)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "broadcast failed in %s: a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)", tag,
                      (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                      (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
        throw BuildError(buf);
    }
    return ggml_add(ctx, a, b);
}

// ggml_mul_mat with a diagnostic
ggml_tensor * mm(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, const char * tag) {
    const bool ok = (a->ne[0] == b->ne[0]) && (b->ne[2] % a->ne[2] == 0) && (b->ne[3] % a->ne[3] == 0) &&
                    (a->nb[0] <= a->nb[1]);
    if (!ok) {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "mul_mat %s: a=(%lld,%lld,%lld,%lld) nb0=%zu nb1=%zu b=(%lld,%lld,%lld,%lld)", tag,
                      (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                      (size_t) a->nb[0], (size_t) a->nb[1],
                      (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
        throw BuildError(buf);
    }
    return ggml_mul_mat(ctx, a, b);
}

// 1-D convolution with a float32 im2col.  ggml_conv_1d hardcodes an f16
// im2col, which costs real precision on the large activation values this model
// produces; here the (f16) kernel stays src0 and the f32 activations are src1.
ggml_tensor * conv1d_f32(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, int s0, int p0,
                         int d0) {
    ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F32);
    ggml_tensor * a2 = ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2]);
    ggml_tensor * i2 = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    ggml_tensor * r = ggml_mul_mat(ctx, a2, i2);          // [OC, N*OW]
    r = ggml_cont(ctx, ggml_transpose(ctx, r));           // [N*OW, OC]
    return ggml_reshape_3d(ctx, r, im2col->ne[1], a->ne[2], im2col->ne[2]);
}

// Depthwise 1-D convolution with a float32 im2col (kernel [K,1,C]).
ggml_tensor * conv1d_dw_f32(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, int s0, int p0,
                            int d0) {
    ggml_tensor * nb = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);
    ggml_tensor * im2col = ggml_im2col(ctx, a, nb, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F32);
    ggml_tensor * r = ggml_mul_mat(ctx, a, im2col);       // [1, OW, C]
    return ggml_reshape_3d(ctx, r, r->ne[1], r->ne[2], 1);
}

// y = W x + b   (W has ggml shape [in, out], x is [in, T])
ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = mm(ctx, w, x, "linear");
    if (b) y = add_bc(ctx, y, b, "linear");
    return y;
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b,
                         float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    y = add_bc(ctx, y, b, "layer_norm");
    return y;
}

}  // namespace

Encoder::Encoder() = default;

Encoder::~Encoder() {
    if (galloc_) ggml_gallocr_free(galloc_);
    if (wctx_) ggml_free(wctx_);
}

int Encoder::conv_out_len(int n, int k, int s, int p) {
    return (n + 2 * p - k) / s + 1;
}

int Encoder::enc_frames(int T_mel) const {
    const HParams & hp = model_->hp();
    const int k = hp.enc_subs_kernel;
    const int p = (k - 1) / 2;
    const int nsub = static_cast<int>(std::log2(hp.enc_subs_factor) + 0.5);
    int n = T_mel;
    for (int i = 0; i < nsub; ++i) n = conv_out_len(n, k, 2, p);
    return n;
}

void Encoder::register_f32(const std::string & name) {
    if (w32_.count(name)) return;
    const float * src = model_->f32(name);
    if (!src) throw BuildError("missing f32 tensor: " + name);
    ggml_tensor * mt = model_->tensor(name);
    if (!mt) throw BuildError("missing tensor: " + name);
    const int64_t n = ggml_nelements(mt);
    ggml_tensor * t = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, n);
    std::memcpy(t->data, src, static_cast<size_t>(n) * sizeof(float));
    w32_[name] = t;
}

ggml_tensor * Encoder::w32(const std::string & name) {
    auto it = w32_.find(name);
    if (it != w32_.end()) return it->second;
    register_f32(name);
    return w32_[name];
}

ggml_tensor * Encoder::w32_2d_auto(const std::string & name) {
    ggml_tensor * mt = model_->tensor(name);
    if (!mt) throw BuildError("missing tensor: " + name);
    return w32_2d(name, 1, static_cast<int>(ggml_nelements(mt)));
}

ggml_tensor * Encoder::w32_2d(const std::string & name, int d0, int d1) {
    const std::string key = name + "@2d";
    auto it = w32_.find(key);
    if (it != w32_.end()) return it->second;
    const float * src = model_->f32(name);
    if (!src) throw BuildError("missing f32 tensor: " + name);
    ggml_tensor * t = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, d0, d1);
    std::memcpy(t->data, src, static_cast<size_t>(d0) * d1 * sizeof(float));
    w32_[key] = t;
    return t;
}

bool Encoder::init(const Model & model, std::string * err) {
    model_ = &model;
    // 8 MiB is plenty for the f32 copies of every bias/norm/mel tensor
    ggml_init_params ip{};
    ip.mem_size = 8u * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    wctx_ = ggml_init(ip);
    if (!wctx_) {
        if (err) *err = "ggml_init failed for the encoder weight context";
        return false;
    }
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend()));
    if (!galloc_) {
        if (err) *err = "ggml_gallocr_new failed";
        return false;
    }
    return true;
}

void Encoder::set_n_threads(int n) {
    n_threads_ = n;
    if (model_) const_cast<Model *>(model_)->set_n_threads(n);
}

bool Encoder::run(const float * mel, int T_mel, std::vector<float> & out, int * T_enc_out,
                  std::string * err, std::vector<std::pair<std::string, std::vector<float>>> * dumps_out) {
    const HParams & hp = model_->hp();
    const int D = hp.enc_d_model;
    const int H = hp.enc_n_heads;
    const int dk = D / H;
    const int K = hp.enc_conv_kernel;
    const int ksub = hp.enc_subs_kernel;
    const float eps = hp.enc_norm_eps;

    if (T_mel < ksub) {
        if (err) *err = "audio too short for the encoder";
        return false;
    }
    const int T = enc_frames(T_mel);
    if (T <= 0) {
        if (err) *err = "no encoder frames";
        return false;
    }

    ggml_context * ctx = nullptr;
    try {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        if (!ctx) throw BuildError("ggml_init failed for the graph context");

        // ---- inputs
        ggml_tensor * mel_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_mel, hp.n_mels, 1);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);

        // ---- striding subsampling (conv1d, stride 2, pad (k-1)/2, ReLU)
        ggml_tensor * x = mel_in;
        const int nsub = static_cast<int>(std::log2(hp.enc_subs_factor) + 0.5);
        for (int i = 0; i < nsub; ++i) {
            const std::string idx = std::to_string(2 * i);
            ggml_tensor * w = model_->tensor("subs.conv" + idx + ".weight");
            if (!w) throw BuildError("missing subs.conv" + idx + ".weight");
            x = conv1d_f32(ctx, w, x, 2, (ksub - 1) / 2, 1);
            x = add_bc(ctx, x, w32_2d_auto("subs.conv" + idx + ".bias"), "subs.bias");
            x = ggml_relu(ctx, x);
        }
        // [T, D, 1] -> [D, T]
        x = ggml_reshape_2d(ctx, x, T, D);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));

        const float rope_base = static_cast<float>(hp.enc_pos_emb_max_len);
        std::vector<std::pair<std::string, ggml_tensor *>> dumps;
        if (dumps_out) {
            ggml_set_output(x);
            dumps.emplace_back("subs", x);
        }

        for (int il = 0; il < hp.enc_n_layers; ++il) {
            const std::string p = "blk." + std::to_string(il) + ".";
            auto W = [&](const char * suffix) {
                ggml_tensor * t = model_->tensor(p + suffix);
                if (!t) throw BuildError("missing tensor: " + p + suffix);
                return t;
            };
            // linear biases are 1-D [out] (broadcast along ne0); conv biases need
            // the explicit [1, out] shape because their activation is [T, out]
            auto B1 = [&](const char * suffix) { return w32(p + suffix); };
            auto B2 = [&](const char * suffix) { return w32_2d_auto(p + suffix); };

            // ---- FFN 1
            ggml_tensor * residual = x;
            ggml_tensor * n = layer_norm(ctx, x, w32(p + "norm_ff1.weight"), w32(p + "norm_ff1.bias"), eps);
            ggml_tensor * h = linear(ctx, W("ff1.linear1.weight"), B1("ff1.linear1.bias"), n);
            h = ggml_silu(ctx, h);
            h = linear(ctx, W("ff1.linear2.weight"), B1("ff1.linear2.bias"), h);
            x = add_bc(ctx, residual, ggml_scale(ctx, h, 0.5f), "ffn_residual");
            if (dumps_out && il == 0) { ggml_set_output(x); dumps.emplace_back("l0_ff1", x); }

            // ---- self attention (rotary applied to the *input* of q/k)
            residual = x;
            n = layer_norm(ctx, x, w32(p + "norm_att.weight"), w32(p + "norm_att.bias"), eps);
            ggml_tensor * nr = ggml_reshape_3d(ctx, n, dk, H, T);
            nr = ggml_rope_ext(ctx, nr, pos, nullptr, dk, GGML_ROPE_TYPE_NEOX, T, rope_base,
                               1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            nr = ggml_reshape_2d(ctx, nr, D, T);

            ggml_tensor * q = linear(ctx, W("attn.q.weight"), B1("attn.q.bias"), nr);
            ggml_tensor * k = linear(ctx, W("attn.k.weight"), B1("attn.k.bias"), nr);
            ggml_tensor * v = linear(ctx, W("attn.v.weight"), B1("attn.v.bias"), n);

            ggml_tensor * q3 = ggml_permute(ctx, ggml_reshape_3d(ctx, q, dk, H, T), 0, 2, 1, 3);
            ggml_tensor * k3 = ggml_permute(ctx, ggml_reshape_3d(ctx, k, dk, H, T), 0, 2, 1, 3);
            ggml_tensor * kq = mm(ctx, k3, q3, "attn_scores");  // [T_k, T_q, H]
            ggml_tensor * pr = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / std::sqrt(static_cast<float>(dk)), 0.0f);

            ggml_tensor * v3 = ggml_permute(ctx, ggml_reshape_3d(ctx, v, dk, H, T), 1, 2, 0, 3);
            v3 = ggml_cont(ctx, v3);                       // [T_k, dk, H]
            ggml_tensor * attn = mm(ctx, v3, pr, "attn_values");  // [dk, T_q, H]
            attn = ggml_permute(ctx, attn, 0, 2, 1, 3);      // [dk, H, T_q]
            attn = ggml_cont(ctx, attn);
            attn = ggml_reshape_2d(ctx, attn, D, T);
            attn = linear(ctx, W("attn.out.weight"), B1("attn.out.bias"), attn);
            x = add_bc(ctx, residual, attn, "attn_residual");
            if (dumps_out && il == 0) { ggml_set_output(x); dumps.emplace_back("l0_attn", x); }

            // ---- convolution module (evaluated in [T, D] layout)
            residual = x;
            n = layer_norm(ctx, x, w32(p + "norm_conv.weight"), w32(p + "norm_conv.bias"), eps);
            ggml_tensor * nt = ggml_cont(ctx, ggml_transpose(ctx, n));  // [T, D]
            nt = ggml_reshape_3d(ctx, nt, T, D, 1);

            ggml_tensor * c = conv1d_f32(ctx, W("conv.pw1.weight"), nt, 1, 0, 1);  // [T, 2D, 1]
            c = add_bc(ctx, c, w32_2d_auto(p + "conv.pw1.bias"), "pw1_bias");
            // GLU over the channel dimension (ne1)
            ggml_tensor * ga = ggml_view_3d(ctx, c, T, D, 1, c->nb[1], c->nb[2], 0);
            ggml_tensor * gb = ggml_view_3d(ctx, c, T, D, 1, c->nb[1], c->nb[2],
                                            static_cast<size_t>(D) * c->nb[1]);
            ggml_tensor * g = ggml_mul(ctx, ga, ggml_sigmoid(ctx, gb));
            if (dumps_out && il == 0) {
                ggml_tensor * d = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, g, T, D)));
                ggml_set_output(d);
                dumps.emplace_back("l0_glu", d);
            }

            ggml_tensor * dw = conv1d_dw_f32(ctx, W("conv.dw.weight"), g, 1, (K - 1) / 2, 1);
            dw = add_bc(ctx, dw, w32_2d_auto(p + "conv.dw.bias"), "dw_bias");

            // LayerNorm over the channel dim: [T,D] -> [D,T]
            ggml_tensor * dwt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, dw, T, D)));
            if (dumps_out && il == 0) {
                ggml_set_output(dwt);
                dumps.emplace_back("l0_dw", dwt);
            }
            dwt = layer_norm(ctx, dwt, w32(p + "conv.norm.weight"), w32(p + "conv.norm.bias"), eps);
            if (dumps_out && il == 0) {
                ggml_set_output(dwt);
                dumps.emplace_back("l0_dwnorm", dwt);
            }
            dwt = ggml_silu(ctx, dwt);
            dwt = ggml_cont(ctx, ggml_transpose(ctx, dwt));  // [T, D]
            dwt = ggml_reshape_3d(ctx, dwt, T, D, 1);
            ggml_tensor * c2 = conv1d_f32(ctx, W("conv.pw2.weight"), dwt, 1, 0, 1);
            c2 = add_bc(ctx, c2, w32_2d_auto(p + "conv.pw2.bias"), "pw2_bias");
            if (dumps_out && il == 0) {
                ggml_tensor * d = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, c2, T, D)));
                ggml_set_output(d);
                dumps.emplace_back("l0_pw2", d);
            }
            c2 = ggml_reshape_2d(ctx, c2, T, D);
            c2 = ggml_cont(ctx, ggml_transpose(ctx, c2));  // [D, T]
            x = add_bc(ctx, residual, c2, "conv_residual");
            if (dumps_out && il == 0) { ggml_set_output(x); dumps.emplace_back("l0_conv", x); }

            // ---- FFN 2
            residual = x;
            n = layer_norm(ctx, x, w32(p + "norm_ff2.weight"), w32(p + "norm_ff2.bias"), eps);
            h = linear(ctx, W("ff2.linear1.weight"), B1("ff2.linear1.bias"), n);
            h = ggml_silu(ctx, h);
            h = linear(ctx, W("ff2.linear2.weight"), B1("ff2.linear2.bias"), h);
            x = add_bc(ctx, residual, ggml_scale(ctx, h, 0.5f), "ffn_residual");
            if (dumps_out && il == 0) { ggml_set_output(x); dumps.emplace_back("l0_ff2", x); }

            // ---- output norm
            x = layer_norm(ctx, x, w32(p + "norm_out.weight"), w32(p + "norm_out.bias"), eps);

            if (dumps_out) {
                ggml_set_output(x);
                dumps.emplace_back("layer" + std::to_string(il), x);
            }
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(gf, x);
        // debug tensors are dead branches: expand them too so the allocator
        // gives them a buffer
        for (auto & [name, dt] : dumps) ggml_build_forward_expand(gf, dt);

        if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
            throw BuildError("ggml_gallocr_alloc_graph failed (out of memory?)");
        }

        // `mel` is time-major ([t][m]); ggml's ne=(T, n_mels) layout stores time
        // fastest, so transpose while uploading.
        {
            std::vector<float> mel_ggml(static_cast<size_t>(T_mel) * hp.n_mels);
            for (int t = 0; t < T_mel; ++t)
                for (int m = 0; m < hp.n_mels; ++m)
                    mel_ggml[static_cast<size_t>(t) + static_cast<size_t>(m) * T_mel] =
                        mel[static_cast<size_t>(t) * hp.n_mels + m];
            ggml_backend_tensor_set(mel_in, mel_ggml.data(), 0, mel_ggml.size() * sizeof(float));
        }
        std::vector<int32_t> positions(T);
        for (int i = 0; i < T; ++i) positions[i] = i;
        ggml_backend_tensor_set(pos, positions.data(), 0, static_cast<size_t>(T) * sizeof(int32_t));

        const ggml_status st = ggml_backend_graph_compute(model_->backend(), gf);
        if (st != GGML_STATUS_SUCCESS) {
            throw BuildError("graph compute failed: " + std::to_string(static_cast<int>(st)));
        }

        out.resize(static_cast<size_t>(D) * T);
        ggml_backend_tensor_get(x, out.data(), 0, out.size() * sizeof(float));
        if (T_enc_out) *T_enc_out = T;

        if (dumps_out) {
            dumps_out->clear();
            std::vector<float> tmp(static_cast<size_t>(D) * T);
            for (auto & [name, dt] : dumps) {
                if (dt->buffer == nullptr) {
                    std::fprintf(stderr, "[encoder] dump '%s' has no buffer, skipped\n", name.c_str());
                    continue;
                }
                ggml_backend_tensor_get(dt, tmp.data(), 0, tmp.size() * sizeof(float));
                std::vector<float> td(tmp.size());
                for (int t = 0; t < T; ++t)
                    for (int c = 0; c < D; ++c)
                        td[static_cast<size_t>(t) * D + c] = tmp[static_cast<size_t>(c) + static_cast<size_t>(t) * D];
                dumps_out->emplace_back(name, std::move(td));
            }
        }
        ggml_free(ctx);
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        if (ctx) ggml_free(ctx);
        return false;
    }
}

}  // namespace asr
