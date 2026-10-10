#include "core/fft.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846  // MSVC does not define it without _USE_MATH_DEFINES
#endif

namespace asr {

std::string fft_backend_name() {
    return "internal Bluestein";
}

namespace {
// ---------------------------------------------------------------- radix-2 FFT

void fft_radix2(std::complex<double> * a, int n, bool inverse) {
    // bit reversal
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * M_PI / len * (inverse ? 1.0 : -1.0);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < n; ++i) a[i] /= n;
    }
}

int next_pow2(int n) {
    int m = 1;
    while (m < n) m <<= 1;
    return m;
}

}  // namespace

Rfft::Rfft(int n) : n_(n) {
    if (n < 2) throw std::invalid_argument("Rfft: n must be >= 2");

    // Bluestein tables.  n_fft = 320 = 2^6 * 5 is not a power of two, so the
    // chirp-z transform is the cheapest way to a plain radix-2 kernel.
    m_ = next_pow2(2 * n - 1);
    chirp_.resize(n);
    std::vector<std::complex<double>> b(m_, {0.0, 0.0});
    for (int k = 0; k < n; ++k) {
        const double ph = M_PI * static_cast<double>(k) * k / n;
        chirp_[k] = std::complex<double>(std::cos(ph), -std::sin(ph));
        const std::complex<double> bb(std::cos(ph), std::sin(ph));
        b[k] = bb;
        if (k) b[m_ - k] = bb;
    }
    fft_radix2(b.data(), m_, false);
    kernel_ = std::move(b);
}

Rfft::~Rfft() = default;

void Rfft::forward_internal(const float * in, std::complex<float> * out,
                            std::complex<double> * scratch) const {
    std::complex<double> * a = scratch;
    std::fill(a, a + m_, std::complex<double>(0.0, 0.0));
    for (int k = 0; k < n_; ++k) a[k] = static_cast<double>(in[k]) * chirp_[k];
    fft_radix2(a, m_, false);
    for (int k = 0; k < m_; ++k) a[k] *= kernel_[k];
    fft_radix2(a, m_, true);
    for (int k = 0; k <= n_ / 2; ++k) {
        const std::complex<double> v = a[k] * chirp_[k];
        out[k] = std::complex<float>(static_cast<float>(v.real()), static_cast<float>(v.imag()));
    }
}

void Rfft::forward_batch_internal(const float * in, std::complex<float> * out, int n_frames) const {
    // One scratch buffer for the whole batch: no member state is touched, so
    // several threads may share one Rfft instance.
    std::vector<std::complex<double>> scratch(static_cast<size_t>(m_));
    for (int f = 0; f < n_frames; ++f) {
        forward_internal(in + static_cast<size_t>(f) * n_, out + static_cast<size_t>(f) * n_bins(),
                         scratch.data());
    }
}

void Rfft::forward(const float * in, std::complex<float> * out) const {
    std::vector<std::complex<double>> scratch(static_cast<size_t>(m_));
    forward_internal(in, out, scratch.data());
}

void Rfft::forward_batch(const float * in, std::complex<float> * out, int n_frames) const {
    if (n_frames <= 0) return;
    forward_batch_internal(in, out, n_frames);
}

std::vector<std::complex<double>> dft_naive(const std::vector<double> & in) {
    const int n = static_cast<int>(in.size());
    std::vector<std::complex<double>> out(n / 2 + 1);
    for (int k = 0; k <= n / 2; ++k) {
        std::complex<double> acc(0.0, 0.0);
        for (int t = 0; t < n; ++t) {
            const double ang = -2.0 * M_PI * k * t / n;
            acc += in[t] * std::complex<double>(std::cos(ang), std::sin(ang));
        }
        out[k] = acc;
    }
    return out;
}

}  // namespace asr
