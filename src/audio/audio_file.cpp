#include "audio/audio_file.h"

#include <cstdio>
#include <vector>

#include "miniaudio.h"

namespace asr::audio {

bool load_file(const std::string & path, int target_rate, std::vector<float> & out, std::string * err) {
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, static_cast<ma_uint32>(target_rate));
    ma_decoder decoder;
    if (ma_decoder_init_file(path.c_str(), &cfg, &decoder) != MA_SUCCESS) {
        if (err) *err = "cannot open audio file: " + path;
        return false;
    }
    out.clear();
    std::vector<float> chunk(1 << 15);
    for (;;) {
        ma_uint64 got = 0;
        if (ma_decoder_read_pcm_frames(&decoder, chunk.data(), chunk.size(), &got) != MA_SUCCESS) break;
        if (got == 0) break;
        out.insert(out.end(), chunk.begin(), chunk.begin() + static_cast<long>(got));
    }
    ma_decoder_uninit(&decoder);
    if (out.empty()) {
        if (err) *err = "audio file is empty or undecodable: " + path;
        return false;
    }
    return true;
}

bool write_wav16(const std::string & path, const float * samples, size_t n, int sample_rate) {
    ma_encoder_config cfg = ma_encoder_config_init(ma_encoding_format_wav, ma_format_s16, 1,
                                                   static_cast<ma_uint32>(sample_rate));
    ma_encoder enc;
    if (ma_encoder_init_file(path.c_str(), &cfg, &enc) != MA_SUCCESS) return false;
    std::vector<int16_t> pcm(n);
    for (size_t i = 0; i < n; ++i) {
        float v = samples[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm[i] = static_cast<int16_t>(v * 32767.0f);
    }
    ma_uint64 written = 0;
    const ma_result r = ma_encoder_write_pcm_frames(&enc, pcm.data(), n, &written);
    ma_encoder_uninit(&enc);
    return r == MA_SUCCESS && written == n;
}

}  // namespace asr::audio
