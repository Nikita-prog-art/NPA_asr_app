// Model catalogue: parsed from the manifest JSON published by the conversion
// workflow (release asset + the copy checked in on the main branch).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace asr::net {

struct ModelFile {
    std::string quant;
    std::string url;
    std::string sha256;   // lower-case hex, may be empty
    uint64_t size = 0;    // bytes, 0 when unknown
    uint64_t params = 0;  // parameter count, 0 when unknown
    // The quantization the catalogue suggests when the user has no preference;
    // the conversion workflow marks one file per model.
    bool recommended = false;
};

struct ModelEntry {
    std::string id, name, repo, revision, license;
    std::vector<ModelFile> files;

    // Returns nullptr when the quant is not published.
    const ModelFile * file(const std::string & quant) const;
};

struct Manifest {
    int schema = 1;
    std::vector<ModelEntry> models;

    // Returns nullptr when the id is unknown.
    const ModelEntry * model(const std::string & id) const;
};

// Parses manifest JSON.  Never throws; on failure returns false and, when `err`
// is non-null, stores a human readable reason.
bool manifest_parse(const std::string & json, Manifest * out, std::string * err);

// Remote manifests, in order of preference.  The first is the checked-in
// manifest (updated by the conversion workflow), the second is the release
// asset.  Both are plain HTTPS GETs.
std::vector<std::string> manifest_urls();

// Compiled-in fallback, may be empty.
const Manifest & manifest_builtin();

// Convenience: fetch manifest_urls() in order, parse, and merge; returns the
// first successfully parsed manifest, falling back to manifest_builtin().
bool manifest_fetch(Manifest * out, std::string * err);

}  // namespace asr::net
