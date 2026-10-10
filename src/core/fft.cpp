#include "core/fft.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846  // MSVC does not define it without _USE_MATH_DEFINES
#endif

#if defined(ASR_HAVE_MKL_FFT)
#include <mkl_dfti.h>
#endif

namespace asr {

namespace {
enum FftBackend { kFftAuto = 0, kFftInternal = 1, kFftMkl = 2 };
int g_fft_backend = kFftAuto;
}  // namespace

void set_fft_backend(const std::string & name) {
    if (name == "internal" || name == "builtin" || name == "bluestein") {
        g_fft_backend = kFftInternal;
    } else if (name == "mkl" || name == "onemkl") {
        g_fft_backend = kFftMkl;
    } else {
        g_fft_backend = kFftAuto;
    }
}

std::string fft_backend_name() {
#if defined(ASR_HAVE_MKL_FFT)
    if (g_fft_backend == kFftMkl) return "oneMKL (forced)";
    if (g_fft_backend == kFftAuto) return "oneMKL";
#else
    if (g_fft_backend == kFftMkl) return "internal Bluestein (oneMKL not compiled in)";
#endif
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

// ---------------------------------------------------------------- MKL

#if defined(ASR_HAVE_MKL_FFT)
struct Rfft::Mkl {
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;
    bool batch_ready = false;
    int batch_n = 0;
    ~Mkl() {
        if (handle) DftiFreeDescriptor(&handle);
    }
};
#endif

Rfft::Rfft(int n) : n_(n) {
    if (n < 2) throw std::invalid_argument("Rfft: n must be >= 2");

#if defined(ASR_HAVE_MKL_FFT)
    if (g_fft_backend != kFftInternal) {
        mkl_ = std::make_unique<Mkl>();
        const MKL_LONG dims[1] = {n};
        if (DftiCreateDescriptor(&mkl_->handle, DFTI_SINGLE, DFTI_REAL, 1, dims) != DFTI_NO_ERROR) {
            mkl_.reset();
        } else {
            DftiSetValue(mkl_->handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            DftiSetValue(mkl_->handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            DftiSetValue(mkl_->handle, DFTI_PACKED_FORMAT, DFTI_CCS_FORMAT);
            if (DftiCommitDescriptor(mkl_->handle) != DFTI_NO_ERROR) mkl_.reset();
        }
    }
#endif

    // Bluestein tables - also prepared when MKL is available so that the
    // portable path can be selected for verification.
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
    work_.resize(m_);
}

Rfft::~Rfft() = default;

bool Rfft::using_mkl() const {
#if defined(ASR_HAVE_MKL_FFT)
    return g_fft_backend != kFftInternal && mkl_ && mkl_->handle != nullptr;
#else
    return false;
#endif
}

void Rfft::forward_internal(const float * in, std::complex<float> * out) const {
    std::vector<std::complex<double>> & a = const_cast<Rfft *>(this)->work_;
    std::fill(a.begin(), a.end(), std::complex<double>(0.0, 0.0));
    for (int k = 0; k < n_; ++k) a[k] = static_cast<double>(in[k]) * chirp_[k];
    fft_radix2(a.data(), m_, false);
    for (int k = 0; k < m_; ++k) a[k] *= kernel_[k];
    fft_radix2(a.data(), m_, true);
    for (int k = 0; k <= n_ / 2; ++k) {
        const std::complex<double> v = a[k] * chirp_[k];
        out[k] = std::complex<float>(static_cast<float>(v.real()), static_cast<float>(v.imag()));
    }
}

void Rfft::forward_batch_internal(const float * in, std::complex<float> * out, int n_frames) const {
    for (int f = 0; f < n_frames; ++f) {
        forward_internal(in + static_cast<size_t>(f) * n_, out + static_cast<size_t>(f) * n_bins());
    }
}

void Rfft::forward(const float * in, std::complex<float> * out) const {
#if defined(ASR_HAVE_MKL_FFT)
    if (using_mkl()) {
        const MKL_LONG status = DftiComputeForward(mkl_->handle, const_cast<float *>(in),
                                                  reinterpret_cast<float *>(out));
        if (status == DFTI_NO_ERROR) return;
    }
#endif
    forward_internal(in, out);
}

void Rfft::forward_batch(const float * in, std::complex<float> * out, int n_frames) const {
    if (n_frames <= 0) return;
#if defined(ASR_HAVE_MKL_FFT)
    if (using_mkl() && n_frames > 1) {
        if (!mkl_->batch_ready || mkl_->batch_n != n_frames) {
            MKL_LONG strides_in[2] = {0, static_cast<MKL_LONG>(n_)};
            MKL_LONG strides_out[2] = {0, static_cast<MKL_LONG>(n_bins()) * 2};
            DftiSetValue(mkl_->handle, DFTI_NUMBER_OF_TRANSFORMS, n_frames);
            DftiSetValue(mkl_->handle, DFTI_INPUT_DISTANCE, n_);
            DftiSetValue(mkl_->handle, DFTI_OUTPUT_DISTANCE, n_bins() * 2);
            DftiSetValue(mkl_->handle, DFTI_INPUT_STRIDES, strides_in);
            DftiSetValue(mkl_->handle, DFTI_OUTPUT_STRIDES, strides_out);
            if (DftiCommitDescriptor(mkl_->handle) != DFTI_NO_ERROR) {
                forward_batch_internal(in, out, n_frames);
                return;
            }
            mkl_->batch_ready = true;
            mkl_->batch_n = n_frames;
        }
        if (DftiComputeForward(mkl_->handle, const_cast<float *>(in),
                               reinterpret_cast<float *>(out)) == DFTI_NO_ERROR) {
            return;
        }
        mkl_->batch_ready = false;
    }
#endif
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
