// SHA-256 (FIPS 180-4), dependency-free.  Used to verify downloaded models.
#include "sha256.h"

#include <cstring>
#include <fstream>

namespace asr::util {
namespace {

constexpr uint32_t kK[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

// Initial state: first 32 bits of the fractional parts of the square roots of
// the first eight primes.
constexpr uint32_t kInit[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                               0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

struct Ctx {
    uint32_t state[8];
    uint8_t  buf[64];
    uint64_t total_len;  // message length in bytes (not bits) so far
    size_t   buf_len;
};

void ctx_init(Ctx * c) {
    std::memcpy(c->state, kInit, sizeof(kInit));
    c->total_len = 0;
    c->buf_len = 0;
    std::memset(c->buf, 0, sizeof(c->buf));
}

void transform(Ctx * c, const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = c->state[0], b = c->state[1], cc = c->state[2], d = c->state[3];
    uint32_t e = c->state[4], f = c->state[5], g = c->state[6], h = c->state[7];

    for (int i = 0; i < 64; ++i) {
        const uint32_t big_s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + big_s1 + ch + kK[i] + w[i];
        const uint32_t big_s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & cc) ^ (b & cc);
        const uint32_t t2 = big_s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = cc;
        cc = b;
        b = a;
        a = t1 + t2;
    }

    c->state[0] += a;
    c->state[1] += b;
    c->state[2] += cc;
    c->state[3] += d;
    c->state[4] += e;
    c->state[5] += f;
    c->state[6] += g;
    c->state[7] += h;
}

void update(Ctx * c, const uint8_t * data, size_t len) {
    if (data == nullptr || len == 0) return;
    c->total_len += len;

    // Top up a partially filled block first.
    if (c->buf_len > 0) {
        const size_t want = 64 - c->buf_len;
        const size_t take = len < want ? len : want;
        std::memcpy(c->buf + c->buf_len, data, take);
        c->buf_len += take;
        data += take;
        len -= take;
        if (c->buf_len == 64) {
            transform(c, c->buf);
            c->buf_len = 0;
        }
    }

    while (len >= 64) {
        transform(c, data);
        data += 64;
        len -= 64;
    }

    if (len > 0) {
        std::memcpy(c->buf, data, len);
        c->buf_len = len;
    }
}

void final(Ctx * c, uint8_t out[32]) {
    const uint64_t bit_len = c->total_len * 8u;
    // Append 0x80, pad with zeros, then the 64-bit big-endian bit length.
    const uint8_t pad_byte = 0x80;
    update(c, &pad_byte, 1);
    const uint8_t zero = 0x00;
    while (c->buf_len != 56) update(c, &zero, 1);
    uint8_t len_be[8];
    for (int i = 0; i < 8; ++i) len_be[i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));
    update(c, len_be, 8);

    for (int i = 0; i < 8; ++i) {
        out[i * 4] = static_cast<uint8_t>(c->state[i] >> 24);
        out[i * 4 + 1] = static_cast<uint8_t>(c->state[i] >> 16);
        out[i * 4 + 2] = static_cast<uint8_t>(c->state[i] >> 8);
        out[i * 4 + 3] = static_cast<uint8_t>(c->state[i]);
    }
}

std::string to_hex(const uint8_t digest[32]) {
    static const char kHex[] = "0123456789abcdef";
    std::string s;
    s.resize(64);
    for (size_t i = 0; i < 32; ++i) {
        s[i * 2] = kHex[digest[i] >> 4];
        s[i * 2 + 1] = kHex[digest[i] & 0x0f];
    }
    return s;
}

}  // namespace

std::string sha256_hex(const void * data, size_t len) {
    Ctx c;
    ctx_init(&c);
    update(&c, static_cast<const uint8_t *>(data), len);
    uint8_t digest[32];
    final(&c, digest);
    return to_hex(digest);
}

std::string sha256_hex(const std::string & data) {
    return sha256_hex(data.data(), data.size());
}

std::string sha256_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";

    Ctx c;
    ctx_init(&c);
    char buf[64 * 1024];  // 64 KiB chunks keep memory flat for multi-GB models
    while (in) {
        in.read(buf, sizeof(buf));
        const std::streamsize n = in.gcount();
        if (n > 0) update(&c, reinterpret_cast<const uint8_t *>(buf), static_cast<size_t>(n));
        if (n <= 0) break;
    }
    if (in.bad()) return "";  // mid-file I/O error

    uint8_t digest[32];
    final(&c, digest);
    return to_hex(digest);
}

}  // namespace asr::util
