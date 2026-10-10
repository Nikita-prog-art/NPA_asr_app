#include "core/model.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "gguf.h"

namespace asr {

namespace {

std::string kv_str(const gguf_context * g, const char * key, const std::string & def = {}) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(g, id);
}

int64_t kv_int(const gguf_context * g, const char * key, int64_t def = 0) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(g, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(g, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(g, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(g, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(g, id));
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(g, id);
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(g, id) ? 1 : 0;
        default: return def;
    }
}

float kv_f32(const gguf_context * g, const char * key, float def = 0.0f) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(g, id);
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT64) return static_cast<float>(gguf_get_val_f64(g, id));
    return def;
}

bool kv_bool(const gguf_context * g, const char * key, bool def = false) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_BOOL) return gguf_get_val_bool(g, id);
    return def;
}

}  // namespace

Model::Model() = default;

Model::~Model() {
    f32_cache_.clear();
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

bool Model::load(const std::string & path, std::string * err) {
    path_ = path;
    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        if (err) *err = "no CPU backend available";
        return false;
    }
    backend_ = ggml_backend_dev_init(dev, nullptr);
    if (!backend_) {
        if (err) *err = "failed to init CPU backend";
        return false;
    }

    // The GGUF file is opened with no_alloc=true, then a backend buffer is
    // allocated for the tensors and the raw data is streamed from the file into
    // it.  This is the same order llama.cpp uses and it works for every backend
    // (tensors always end up with a valid buffer, which
    // ggml_backend_tensor_get/set require).
    ggml_context * ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &ctx;

    gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        if (err) *err = "cannot read GGUF file: " + path;
        return false;
    }
    ctx_ = ctx;

    buffer_ = ggml_backend_alloc_ctx_tensors(ctx, backend_);
    if (!buffer_) {
        if (err) *err = "failed to allocate the model buffer";
        gguf_free(g);
        return false;
    }

    hp_.model_name = kv_str(g, "gigaam.model_name", "gigaam");
    hp_.sample_rate = static_cast<int>(kv_int(g, "gigaam.sample_rate", 16000));
    hp_.n_mels      = static_cast<int>(kv_int(g, "gigaam.n_mels", 64));
    hp_.n_fft       = static_cast<int>(kv_int(g, "gigaam.n_fft", 320));
    hp_.win_length  = static_cast<int>(kv_int(g, "gigaam.win_length", 320));
    hp_.hop_length  = static_cast<int>(kv_int(g, "gigaam.hop_length", 160));
    hp_.center      = kv_bool(g, "gigaam.center", false);

    hp_.enc_n_layers        = static_cast<int>(kv_int(g, "gigaam.encoder.n_layers", 16));
    hp_.enc_d_model         = static_cast<int>(kv_int(g, "gigaam.encoder.d_model", 768));
    hp_.enc_n_heads         = static_cast<int>(kv_int(g, "gigaam.encoder.n_heads", 16));
    hp_.enc_d_ff            = static_cast<int>(kv_int(g, "gigaam.encoder.d_ff", 3072));
    hp_.enc_conv_kernel     = static_cast<int>(kv_int(g, "gigaam.encoder.conv_kernel", 5));
    hp_.enc_subs_kernel     = static_cast<int>(kv_int(g, "gigaam.encoder.subs_kernel", 5));
    hp_.enc_subs_factor     = static_cast<int>(kv_int(g, "gigaam.encoder.subsampling_factor", 4));
    hp_.enc_pos_emb_max_len = static_cast<int>(kv_int(g, "gigaam.encoder.pos_emb_max_len", 5000));
    hp_.enc_norm_eps        = kv_f32(g, "gigaam.encoder.norm_eps", 1e-5f);

    hp_.dec_pred_hidden  = static_cast<int>(kv_int(g, "gigaam.decoder.pred_hidden", 320));
    hp_.dec_pred_layers  = static_cast<int>(kv_int(g, "gigaam.decoder.pred_layers", 1));
    hp_.dec_num_classes  = static_cast<int>(kv_int(g, "gigaam.decoder.num_classes", 1025));
    hp_.dec_blank_id     = static_cast<int>(kv_int(g, "gigaam.decoder.blank_id", 1024));
    hp_.dec_joint_hidden = static_cast<int>(kv_int(g, "gigaam.decoder.joint_hidden", 320));
    hp_.dec_max_symbols  = static_cast<int>(kv_int(g, "gigaam.decoder.max_symbols_per_step", 10));

    const int64_t vocab_id = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (vocab_id >= 0 && gguf_get_kv_type(g, vocab_id) == GGUF_TYPE_ARRAY) {
        const int64_t n = gguf_get_arr_n(g, vocab_id);
        hp_.vocab.reserve(n);
        for (int64_t i = 0; i < n; ++i) hp_.vocab.emplace_back(gguf_get_arr_str(g, vocab_id, i));
    }

    const int64_t n_tensors = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(ctx_, name);
        if (!t) continue;
        tensors_.emplace(name, t);
    }

    // ---- stream the weights into the backend buffer
    const size_t data_offset = gguf_get_data_offset(g);
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot reopen GGUF file for reading";
        gguf_free(g);
        return false;
    }
    std::vector<uint8_t> chunk;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(ctx_, name);
        if (!t) continue;
        const size_t nbytes = ggml_nbytes(t);
        const size_t off = data_offset + gguf_get_tensor_offset(g, i);
        if (std::fseek(f, static_cast<long>(off), SEEK_SET) != 0) {
            if (err) *err = std::string("seek failed for tensor ") + name;
            std::fclose(f);
            gguf_free(g);
            return false;
        }
        chunk.resize(nbytes);
        if (std::fread(chunk.data(), 1, nbytes, f) != nbytes) {
            if (err) *err = std::string("short read for tensor ") + name;
            std::fclose(f);
            gguf_free(g);
            return false;
        }
        ggml_backend_tensor_set(t, chunk.data(), 0, nbytes);
    }
    std::fclose(f);

    gguf_free(g);

    if (backend_) set_n_threads(n_threads_ > 0 ? n_threads_ : 0);
    return true;
}

void Model::set_n_threads(int n) {
    n_threads_ = n;
    if (!backend_) return;
    // The CPU backend may live in a dynamically loaded module, so go through the
    // backend registry instead of calling ggml_backend_cpu_set_n_threads()
    // directly.
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (set_threads) {
            set_threads(backend_, n);
            return;
        }
    }
}

ggml_tensor * Model::tensor(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second;
}

bool Model::read_f32(const std::string & name, std::vector<float> & out) const {
    ggml_tensor * t = tensor(name);
    if (!t) return false;
    const int64_t n = ggml_nelements(t);
    out.resize(static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(static_cast<size_t>(n));
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(tmp[i]);
        return true;
    }
    return false;
}

const float * Model::f32(const std::string & name) const {
    auto it = f32_cache_.find(name);
    if (it != f32_cache_.end()) return it->second.data();
    std::vector<float> tmp;
    if (!read_f32(name, tmp)) return nullptr;
    auto res = f32_cache_.emplace(name, std::move(tmp));
    return res.first->second.data();
}

}  // namespace asr
