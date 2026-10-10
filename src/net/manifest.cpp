// Manifest parsing + fetching.  Parsing is deliberately defensive: a broken or
// partially rolled-out manifest must produce a readable error, never a crash.
#include "net/manifest.h"

#include "net/manifest_builtin_data.h"

#include <cctype>
#include <utility>

#include "json.hpp"
#include "net/net.h"

namespace asr::net {
namespace {

using nlohmann::json;

std::string where_of(const std::string & model_id, size_t index) {
    if (!model_id.empty()) return "model \"" + model_id + "\"";
    return "model #" + std::to_string(index);
}

// Reads an optional string.  JSON null counts as "absent" so that a workflow
// can emit null placeholders without invalidating the whole catalogue.
bool opt_string(const json & obj, const char * key, std::string * out,
                const std::string & where, std::string * err) {
    if (!obj.contains(key) || obj[key].is_null()) return true;
    const json & v = obj[key];
    if (!v.is_string()) {
        *err = where + ": field \"" + key + "\" must be a string";
        return false;
    }
    *out = v.get<std::string>();
    return true;
}

// Reads an optional non-negative integer.  Integral floats are accepted because
// some JSON emitters write sizes as 445089120.0.
bool opt_u64(const json & obj, const char * key, uint64_t * out,
             const std::string & where, std::string * err) {
    if (!obj.contains(key) || obj[key].is_null()) return true;
    const json & v = obj[key];
    if (v.is_number_unsigned()) {
        *out = v.get<uint64_t>();
        return true;
    }
    if (v.is_number_integer()) {
        const int64_t s = v.get<int64_t>();
        if (s < 0) {
            *err = where + ": field \"" + key + "\" must not be negative";
            return false;
        }
        *out = static_cast<uint64_t>(s);
        return true;
    }
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (d < 0.0 || d != static_cast<double>(static_cast<uint64_t>(d))) {
            *err = where + ": field \"" + key + "\" must be a non-negative integer";
            return false;
        }
        *out = static_cast<uint64_t>(d);
        return true;
    }
    *err = where + ": field \"" + key + "\" must be a number";
    return false;
}

bool is_hex64(const std::string & s) {
    for (char c : s) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

// Normalises "sha256:<hex>" / upper-case hex to lower-case bare hex and
// validates the length, so downloads can compare with a plain ==.
bool normalise_sha256(std::string in, std::string * out, const std::string & where,
                      std::string * err) {
    if (in.empty()) return true;  // no hash published -> skip verification
    if (in.size() > 7 && in.compare(0, 7, "sha256:") == 0) in.erase(0, 7);
    for (char & c : in) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (in.size() != 64 || !is_hex64(in)) {
        *err = where + ": field \"sha256\" must be 64 hex characters";
        return false;
    }
    *out = std::move(in);
    return true;
}

bool parse_file(const json & f, ModelFile * out, const std::string & where, std::string * err) {
    if (!f.is_object()) {
        *err = where + ": every entry of \"files\" must be an object";
        return false;
    }
    if (!opt_string(f, "quant", &out->quant, where, err)) return false;
    if (!opt_string(f, "url", &out->url, where, err)) return false;
    if (out->quant.empty()) {
        *err = where + ": file entry is missing \"quant\"";
        return false;
    }
    if (out->url.empty()) {
        *err = where + ": file entry \"" + out->quant + "\" is missing \"url\"";
        return false;
    }
    if (out->url.compare(0, 7, "http://") != 0 && out->url.compare(0, 8, "https://") != 0) {
        *err = where + ": file entry \"" + out->quant + "\" has a non-http(s) url";
        return false;
    }
    if (!opt_string(f, "sha256", &out->sha256, where, err)) return false;
    if (!normalise_sha256(out->sha256, &out->sha256, where, err)) return false;
    if (!opt_u64(f, "size", &out->size, where, err)) return false;
    if (!opt_u64(f, "params", &out->params, where, err)) return false;
    return true;
}

// Adds everything from `src` that `dst` does not have yet (first wins).
void merge_into(Manifest * dst, const Manifest & src) {
    for (const ModelEntry & m : src.models) {
        ModelEntry * existing = nullptr;
        for (ModelEntry & d : dst->models) {
            if (d.id == m.id) {
                existing = &d;
                break;
            }
        }
        if (existing == nullptr) {
            dst->models.push_back(m);
            continue;
        }
        for (const ModelFile & f : m.files) {
            if (existing->file(f.quant) == nullptr) existing->files.push_back(f);
        }
    }
}

}  // namespace

const ModelFile * ModelEntry::file(const std::string & quant) const {
    for (const ModelFile & f : files) {
        if (f.quant == quant) return &f;
    }
    return nullptr;
}

const ModelEntry * Manifest::model(const std::string & id) const {
    for (const ModelEntry & m : models) {
        if (m.id == id) return &m;
    }
    return nullptr;
}

bool manifest_parse(const std::string & text, Manifest * out, std::string * err) {
    std::string local_err;
    if (err == nullptr) err = &local_err;
    err->clear();
    if (out == nullptr) {
        *err = "manifest_parse: null output";
        return false;
    }

    try {
        const json root = json::parse(text);
        if (!root.is_object()) {
            *err = "manifest: top level value must be a JSON object";
            return false;
        }

        Manifest parsed;
        if (root.contains("schema") && !root["schema"].is_null()) {
            if (!root["schema"].is_number_integer() && !root["schema"].is_number_unsigned()) {
                *err = "manifest: \"schema\" must be an integer";
                return false;
            }
            parsed.schema = root["schema"].get<int>();
            if (parsed.schema != 1) {
                *err = "manifest: unsupported schema " + std::to_string(parsed.schema) +
                       " (this build understands schema 1)";
                return false;
            }
        }

        if (!root.contains("models") || !root["models"].is_array()) {
            *err = "manifest: \"models\" must be an array";
            return false;
        }

        size_t index = 0;
        for (const json & m : root["models"]) {
            const std::string where = where_of(m.is_object() && m.contains("id") && m["id"].is_string()
                                                   ? m["id"].get<std::string>()
                                                   : std::string(),
                                               index);
            ++index;
            if (!m.is_object()) {
                *err = where + ": every entry of \"models\" must be an object";
                return false;
            }
            ModelEntry entry;
            if (!opt_string(m, "id", &entry.id, where, err)) return false;
            if (entry.id.empty()) {
                *err = where + ": missing \"id\"";
                return false;
            }
            const std::string w = where_of(entry.id, 0);
            if (!opt_string(m, "name", &entry.name, w, err)) return false;
            if (!opt_string(m, "repo", &entry.repo, w, err)) return false;
            if (!opt_string(m, "revision", &entry.revision, w, err)) return false;
            if (!opt_string(m, "license", &entry.license, w, err)) return false;

            if (!m.contains("files") || !m["files"].is_array()) {
                *err = w + ": \"files\" must be an array";
                return false;
            }
            if (m["files"].empty()) {
                *err = w + ": \"files\" is empty, nothing to download";
                return false;
            }
            for (const json & f : m["files"]) {
                ModelFile file;
                if (!parse_file(f, &file, w, err)) return false;
                if (entry.file(file.quant) != nullptr) {
                    *err = w + ": duplicate quant \"" + file.quant + "\"";
                    return false;
                }
                entry.files.push_back(std::move(file));
            }
            parsed.models.push_back(std::move(entry));
        }

        *out = std::move(parsed);
        return true;
    } catch (const std::exception & e) {
        // nlohmann throws json::parse_error on malformed text; nothing escapes.
        *err = std::string("manifest: ") + e.what();
        return false;
    } catch (...) {
        *err = "manifest: unknown error while parsing";
        return false;
    }
}

std::vector<std::string> manifest_urls() {
    // Both are produced by the conversion workflow: the checked-in copy is
    // updated on the main branch, the release asset is attached to the release.
    return {
        "https://raw.githubusercontent.com/Nikita-prog-art/NPA_asr_app/main/models/manifest.json",
        "https://github.com/Nikita-prog-art/NPA_asr_app/releases/latest/download/manifest.json",
    };
}

const Manifest & manifest_builtin() {
    // Generated from models/manifest.json by tools/make_manifest_header.py (the
    // conversion workflow refreshes it).  Empty until the models are published.
    static const Manifest kBuiltin = [] {
        Manifest m;
        std::string err;
        if (!manifest_parse(kBuiltinManifestJson, &m, &err)) {
            m = Manifest{};
            m.schema = 1;
        }
        return m;
    }();
    return kBuiltin;
}

bool manifest_fetch(Manifest * out, std::string * err) {
    std::string local_err;
    if (err == nullptr) err = &local_err;
    err->clear();
    if (out == nullptr) {
        *err = "manifest_fetch: null output";
        return false;
    }

    Manifest merged;
    merged.schema = 1;
    bool got_remote = false;
    std::string problems;

    for (const std::string & url : manifest_urls()) {
        std::string body;
        std::string fetch_err;
        if (!fetch_text(url, &body, &fetch_err)) {
            problems += "[" + url + "] " + fetch_err + "\n";
            continue;
        }
        Manifest one;
        std::string parse_err;
        if (!manifest_parse(body, &one, &parse_err)) {
            problems += "[" + url + "] " + parse_err + "\n";
            continue;
        }
        merge_into(&merged, one);  // earlier sources win on conflicts
        got_remote = true;
    }

    if (got_remote) {
        merge_into(&merged, manifest_builtin());  // fill in known ids we did not fetch
        *out = std::move(merged);
        return true;
    }

    const Manifest & builtin = manifest_builtin();
    if (!builtin.models.empty()) {
        *out = builtin;
        return true;  // offline, but the compiled-in catalogue is usable
    }
    *err = problems.empty() ? std::string("manifest: no sources configured") : problems;
    return false;
}

}  // namespace asr::net
