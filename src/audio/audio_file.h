// Audio file loading (wav/mp3/flac/ogg via miniaudio) into 16 kHz mono float.
#pragma once

#include <string>
#include <vector>

namespace asr::audio {

// Decodes `path` to mono float32 at `target_rate` Hz (resampled by miniaudio).
// Returns false and fills *err on failure.
bool load_file(const std::string & path, int target_rate, std::vector<float> & out, std::string * err);

// Writes a mono float32 buffer as a 16-bit PCM wav file.
bool write_wav16(const std::string & path, const float * samples, size_t n, int sample_rate);

}  // namespace asr::audio
