// Real-input FFT of arbitrary (non power of two) length.
//
// Two back-ends:
//   * Intel oneMKL DFTI (when the build enabled ASR_HAVE_MKL_FFT and oneMKL is
//     found at configure time) - used for batched transforms;
//   * a portable Bluestein implementation on top of a radix-2 FFT, used as the
//     fallback and for verification (gigaam n_fft = 320 = 2^6 * 5, which is not
//     a power of two, so a plain radix-2 FFT cannot be used directly).
#pragma once

#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace asr {

// Which FFT implementation to use.  "auto" (default) prefers oneMKL when the
// build enabled it, "internal" forces the portable Bluestein path, "mkl" forces
// oneMKL (ignored when the build has no MKL).  Set before creating any Rfft.
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
    bool using_mkl() const;

    // out must hold n_bins() complex values.
    void forward(const float * in, std::complex<float> * out) const;

    // Batched: in = n_frames * n floats, out = n_frames * n_bins complex values.
    void forward_batch(const float * in, std::complex<float> * out, int n_frames) const;

private:
    void forward_internal(const float * in, std::complex<float> * out) const;
    void forward_batch_internal(const float * in, std::complex<float> * out, int n_frames) const;

    int n_;
    // Bluestein tables (m_ = power of two >= 2n-1)
    int m_ = 0;
    std::vector<std::complex<double>> chirp_;   // w[k] = exp(-i*pi*k^2/n)
    std::vector<std::complex<double>> kernel_;  // FFT of b
    std::vector<std::complex<double>> work_;    // scratch (not thread safe)

#if defined(ASR_HAVE_MKL_FFT)
    struct Mkl;
    std::unique_ptr<Mkl> mkl_;
#endif
};

// Small helper used by tests: naive O(n^2) DFT for verification.
std::vector<std::complex<double>> dft_naive(const std::vector<double> & in);

}  // namespace asr
