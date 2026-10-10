// See build_info.h.  This is the only translation unit that knows the exact
// revision, so a new commit only recompiles this file.
#include "app/build_info.h"

// cmake/build_id.h.in -> <build>/generated/npa_asr_build_id.h defines
// ASR_BUILD_ID.  A missing header must not break the build: an unknown revision
// only means the install prompt falls back to "ask once per run".
#if defined(__has_include)
#  if __has_include("npa_asr_build_id.h")
#    include "npa_asr_build_id.h"
#  endif
#endif

#ifndef ASR_BUILD_ID
#  define ASR_BUILD_ID "unknown"
#endif

namespace asr::app {
namespace {

constexpr char kMarkerPrefix[] = "npa-asr-build-id:";

// One string constant holding the prefix *and* the revision: the two halves
// must be adjacent in the image, otherwise another copy of the application
// cannot recover the revision from this executable (see install.cpp).
constexpr char kMarker[] = "npa-asr-build-id:" ASR_BUILD_ID;

}  // namespace

const char * build_id() { return kMarker + sizeof(kMarkerPrefix) - 1; }

const char * build_marker() { return kMarker; }

const char * build_marker_prefix() { return kMarkerPrefix; }

}  // namespace asr::app
