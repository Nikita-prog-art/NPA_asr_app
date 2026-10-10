// Microphone capture (miniaudio) and a lock-free single-producer ring buffer.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace asr::audio {

// ---------------------------------------------------------------- ring buffer
// Single producer (audio callback) / single consumer (worker thread).
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity = 1 << 20) : buf_(capacity), cap_(capacity) {}

    size_t write(const float * src, size_t n) {
        const size_t w = w_.load(std::memory_order_relaxed);
        const size_t r = r_.load(std::memory_order_acquire);
        const size_t free = cap_ - (w - r);
        if (n > free) n = free;
        for (size_t i = 0; i < n; ++i) buf_[(w + i) % cap_] = src[i];
        w_.store(w + n, std::memory_order_release);
        return n;
    }

    size_t read(float * dst, size_t n) {
        const size_t r = r_.load(std::memory_order_relaxed);
        const size_t w = w_.load(std::memory_order_acquire);
        const size_t avail = w - r;
        if (n > avail) n = avail;
        for (size_t i = 0; i < n; ++i) dst[i] = buf_[(r + i) % cap_];
        r_.store(r + n, std::memory_order_release);
        return n;
    }

    size_t available() const {
        return w_.load(std::memory_order_acquire) - r_.load(std::memory_order_acquire);
    }
    void clear() {
        r_.store(w_.load(std::memory_order_acquire), std::memory_order_release);
    }

private:
    std::vector<float> buf_;
    size_t cap_;
    std::atomic<size_t> w_{0};
    std::atomic<size_t> r_{0};
};

// ---------------------------------------------------------------- devices
struct DeviceInfo {
    int index = -1;
    std::string name;
    bool is_default = false;
};

std::vector<DeviceInfo> list_capture_devices();

// ---------------------------------------------------------------- capture
class Capture {
public:
    Capture();
    ~Capture();
    Capture(const Capture &) = delete;
    Capture & operator=(const Capture &) = delete;

    // Starts capturing mono float32 at `sample_rate`.  `cb` is invoked on the
    // audio thread and must not block (push into a RingBuffer).
    bool start(int device_index, int sample_rate, std::function<void(const float *, int)> cb,
               std::string * err);
    void stop();
    bool running() const { return running_; }
    const std::string & device_name() const { return device_name_; }

public:
    struct Impl;  // defined in capture.cpp

private:
    Impl * impl_ = nullptr;
    bool running_ = false;
    std::string device_name_;
};

}  // namespace asr::audio
