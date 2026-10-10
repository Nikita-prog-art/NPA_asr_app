// Identifies the source revision a binary was built from.
//
// The value is a git revision ("351de61", with a "-dirty" suffix when the
// working tree had local modifications) and is baked in at configure time (see
// ASR_BUILD_ID in CMakeLists.txt).  Two things use it:
//
//   * the user interface and --info show it, so it is obvious which build is
//     running;
//   * the "should I offer to install?" decision compares it with the revision
//     of the installed copy, so the application offers to update an installed
//     copy exactly when the source revision changed - and stays quiet when the
//     same revision is simply rebuilt (the bytes differ, the code does not).
//
// The revision of an installed copy is read straight out of that file: every
// executable contains the marker "npa-asr-build-id:<revision>" as one string
// constant (build_marker()), which install.cpp searches for.  A side-car state
// file could not do that - it would go stale when the installed executable is
// replaced by anything other than this application.
#pragma once

namespace asr::app {

// Human readable revision of the running build, or "unknown" when it could not
// be determined at configure time.
const char * build_id();

// "npa-asr-build-id:<revision>": the string that is present verbatim in the
// executable image.  Returned for tests and diagnostics.
const char * build_marker();

// The prefix of build_marker(), used to find a *foreign* marker.
const char * build_marker_prefix();

}  // namespace asr::app
