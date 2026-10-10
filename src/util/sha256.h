// Minimal SHA-256 (FIPS 180-4) used to verify downloaded model files.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace asr::util {

std::string sha256_hex(const void * data, size_t len);
std::string sha256_hex(const std::string & data);
// Returns an empty string if the file cannot be read.
std::string sha256_file(const std::string & path);

}  // namespace asr::util
