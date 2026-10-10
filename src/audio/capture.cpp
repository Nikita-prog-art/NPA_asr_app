#include "audio/capture.h"

#include <cstring>

#include "miniaudio.h"

namespace asr::audio {

std::vector<DeviceInfo> list_capture_devices() {
    std::vector<DeviceInfo> out;
    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) return out;

    ma_device_info * playback = nullptr;
    ma_device_info * capture = nullptr;
    ma_uint32 n_playback = 0, n_capture = 0;
    if (ma_context_get_devices(&ctx, &playback, &n_playback, &capture, &n_capture) == MA_SUCCESS) {
        for (ma_uint32 i = 0; i < n_capture; ++i) {
            DeviceInfo d;
            d.index = static_cast<int>(i);
            d.name = capture[i].name;
            d.is_default = capture[i].isDefault != 0;
            out.push_back(d);
        }
    }
    ma_context_uninit(&ctx);
    return out;
}

struct Capture::Impl {
    ma_device device{};
    bool initialized = false;
    std::function<void(const float *, int)> cb;
    std::string name;
};

namespace {
void data_callback(ma_device * device, void * /*output*/, const void * input, ma_uint32 frame_count) {
    auto * impl = static_cast<Capture::Impl *>(device->pUserData);
    if (!impl || !impl->cb || !input) return;
    impl->cb(static_cast<const float *>(input), static_cast<int>(frame_count));
}
}  // namespace

Capture::Capture() : impl_(new Impl()) {}

Capture::~Capture() {
    stop();
    delete impl_;
}

bool Capture::start(int device_index, int sample_rate, std::function<void(const float *, int)> cb,
                    std::string * err) {
    stop();
    impl_->cb = std::move(cb);

    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format = ma_format_f32;
    cfg.capture.channels = 1;
    cfg.sampleRate = static_cast<ma_uint32>(sample_rate);
    cfg.dataCallback = data_callback;
    cfg.pUserData = impl_;

    ma_context ctx;
    ma_device_info * playback = nullptr;
    ma_device_info * capture_devs = nullptr;
    ma_uint32 n_playback = 0, n_capture = 0;
    ma_device_id * want_id = nullptr;
    ma_context * ctx_ptr = nullptr;
    if (device_index >= 0 && ma_context_init(nullptr, 0, nullptr, &ctx) == MA_SUCCESS) {
        if (ma_context_get_devices(&ctx, &playback, &n_playback, &capture_devs, &n_capture) == MA_SUCCESS &&
            device_index < static_cast<int>(n_capture)) {
            want_id = &capture_devs[device_index].id;
            impl_->name = capture_devs[device_index].name;
            ctx_ptr = &ctx;
        }
    }

    ma_result r = ma_device_init(ctx_ptr, &cfg, &impl_->device);
    if (ctx_ptr) ma_context_uninit(&ctx);
    if (r != MA_SUCCESS) {
        if (err) *err = std::string("cannot open capture device: ") + ma_result_description(r);
        return false;
    }
    impl_->initialized = true;
    r = ma_device_start(&impl_->device);
    if (r != MA_SUCCESS) {
        if (err) *err = std::string("cannot start capture: ") + ma_result_description(r);
        ma_device_uninit(&impl_->device);
        impl_->initialized = false;
        return false;
    }
    if (impl_->name.empty()) impl_->name = "default";
    device_name_ = impl_->name;
    running_ = true;
    return true;
}

void Capture::stop() {
    if (impl_->initialized) {
        ma_device_stop(&impl_->device);
        ma_device_uninit(&impl_->device);
        impl_->initialized = false;
    }
    running_ = false;
}

}  // namespace asr::audio
