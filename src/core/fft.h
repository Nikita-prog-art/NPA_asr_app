// Real-input FFT of arbitrary (non power of two) length.
//
// Self-contained Bluestein implementation on top of a radix-2 FFT: gigaam uses
// n_fft = 320 = 2^6 * 5, which is not a power of two, so a plain radix-2 FFT
// cannot be used directly.  A constructed Rfft is immutable and safe to share
// between threads (every call allocates its own scratch buffer).
#pragma once

#include <complex>
#include <string>
#include <vector>

namespace asr {

// Kept for compatibility with existing command lines and settings: every name
// now selects the built-in implementation.  Set before creating any Rfft.
void set_fft_backend(const std::string & name);
std::string fft_backend_name();

class Rfft {
public:
    // n must be >= 2.
    explicit Rfft(int n);
    ~Rfft();
    Rfft(const Rfft &) = delete;
    Rfft & operator=(const Rfft &) = delete;

    int size() const { return n_; }
    int n_bins() const { return n_ / 2 + 1; }  // onesided rfft size
    bool using_mkl() const;                    // always false, kept for tests

    // out must hold n_bins() complex values.
    void forward(const float * in, std::complex<float> * out) const;

    // Batched: in = n_frames * n floats, out = n_frames * n_bins complex values.
    void forward_batch(const float * in, std::complex<float> * out, int n_frames) const;

private:
    void forward_internal(const float * in, std::complex<float> * out,
                          std::complex<double> * scratch) const;
    void forward_batch_internal(const float * in, std::complex<float> * out, int n_frames) const;

    int n_;
    // Bluestein tables (m_ = power of two >= 2n-1)
    int m_ = 0;
    std::vector<std::complex<double>> chirp_;   // w[k] = exp(-i*pi*k^2/n)
    std::vector<std::complex<double>> kernel_;  // FFT of b
};

// Small helper used by tests: naive O(n^2) DFT for verification.
std::vector<std::complex<double>> dft_naive(const std::vector<double> & in);

}  // namespace asr
