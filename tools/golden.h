// Reader for the flat golden container produced by tools/dump_goldens.py.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace asr::test {

struct GoldArray {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t dtype = 0;  // 0=f32 1=f64 2=i64 3=i32
    std::vector<uint8_t> data;

    size_t numel() const {
        size_t n = 1;
        for (auto d : dims) n *= static_cast<size_t>(d);
        return n;
    }
    const float * as_f32() const { return reinterpret_cast<const float *>(data.data()); }
    const int64_t * as_i64() const { return reinterpret_cast<const int64_t *>(data.data()); }
};

class Goldens {
public:
    bool load(const std::string & path) {
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        char magic[8];
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "ASRGOLD1", 8) != 0) {
            std::fclose(f);
            return false;
        }
        uint32_t n = 0;
        if (std::fread(&n, 4, 1, f) != 1) { std::fclose(f); return false; }
        for (uint32_t i = 0; i < n; ++i) {
            GoldArray a;
            uint32_t name_len = 0;
            if (std::fread(&name_len, 4, 1, f) != 1) { std::fclose(f); return false; }
            a.name.resize(name_len);
            if (std::fread(a.name.data(), 1, name_len, f) != name_len) { std::fclose(f); return false; }
            uint32_t ndim = 0;
            if (std::fread(&ndim, 4, 1, f) != 1) { std::fclose(f); return false; }
            a.dims.resize(ndim);
            for (uint32_t d = 0; d < ndim; ++d) {
                if (std::fread(&a.dims[d], 8, 1, f) != 1) { std::fclose(f); return false; }
            }
            if (std::fread(&a.dtype, 4, 1, f) != 1) { std::fclose(f); return false; }
            uint64_t nbytes = 0;
            if (std::fread(&nbytes, 8, 1, f) != 1) { std::fclose(f); return false; }
            a.data.resize(nbytes);
            if (nbytes && std::fread(a.data.data(), 1, nbytes, f) != nbytes) { std::fclose(f); return false; }
            map_[a.name] = std::move(a);
        }
        std::fclose(f);
        return true;
    }

    const GoldArray * get(const std::string & name) const {
        auto it = map_.find(name);
        return it == map_.end() ? nullptr : &it->second;
    }

    bool has(const std::string & name) const { return map_.count(name) > 0; }

private:
    std::unordered_map<std::string, GoldArray> map_;
};

}  // namespace asr::test
