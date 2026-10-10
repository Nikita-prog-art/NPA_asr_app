#!/usr/bin/env bash
# Build npa_asr on Linux.
#
#   scripts/build_linux.sh [extra cmake args...]
#
# Environment:
#   ASR_COMPILER  gcc | clang | icx   (default: gcc; icx measured ~equal, see README)
#   ASR_MKL_FFT   ON to link oneMKL DFTI for the mel FFT (default: OFF - it is
#                 ~0.2% of the runtime and maps ~100 MB of MKL, see README)
#   ASR_NATIVE    ON to build a single-arch binary for this machine only
#   BUILD_DIR     build directory (default: <repo>/build)
#   ONEAPI_ROOT   only used when ASR_COMPILER=icx or ASR_MKL_FFT=ON
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
ONEAPI_ROOT="${ONEAPI_ROOT:-$HOME/dists/intel/oneapi}"

COMPILER="${ASR_COMPILER:-gcc}"
case "$COMPILER" in
    icx)   CC_BIN=icx;   CXX_BIN=icpx ;;
    clang) CC_BIN=clang; CXX_BIN=clang++ ;;
    *)     CC_BIN=gcc;   CXX_BIN=g++ ;;
esac

# oneAPI is optional: it is only sourced when the Intel compiler or oneMKL is
# actually requested, so the default build has no oneAPI runtime dependency.
MKL_ARG=()
if [ "${ASR_MKL_FFT:-OFF}" != "OFF" ]; then
    MKL_ARG+=(-DASR_MKL_FFT=ON)
fi
if [ "$COMPILER" = icx ] || [ "${ASR_MKL_FFT:-OFF}" != "OFF" ]; then
    if [ -f "$ONEAPI_ROOT/setvars.sh" ]; then
        echo "[build] sourcing $ONEAPI_ROOT/setvars.sh"
        set +u
        # shellcheck disable=SC1090
        source "$ONEAPI_ROOT/setvars.sh" --force >/dev/null 2>&1 || true
        set -u
    else
        echo "[build] oneAPI not found at $ONEAPI_ROOT"
    fi
fi
echo "[build] compiler: $CC_BIN / $CXX_BIN"

NATIVE_ARG=()
if [ "${ASR_NATIVE:-OFF}" != "OFF" ]; then
    NATIVE_ARG+=(-DASR_NATIVE=ON)
fi

# ggml builds the CPU backend once per ISA variant, and the big translation units
# (ggml-cpu/ops.cpp with AVX512) need ~2 GB of RAM each, so an unconstrained
# -j$(nproc) can exhaust a 16 GB WSL VM and freeze the whole distro.  Budget
# ~2 GB per job and let ASR_JOBS override it.
JOBS="${ASR_JOBS:-}"
if [ -z "$JOBS" ]; then
    mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    JOBS=$(( mem_gb / 2 ))
    [ "$JOBS" -gt "$(nproc)" ] && JOBS="$(nproc)"
    [ "$JOBS" -lt 1 ] && JOBS=1
fi
echo "[build] parallel jobs: $JOBS (ASR_JOBS to override)"

cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DCMAKE_C_COMPILER="$CC_BIN" \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    "${NATIVE_ARG[@]}" "${MKL_ARG[@]}" "$@"

cmake --build "$BUILD_DIR" -j "$JOBS"

echo
echo "[build] done: $BUILD_DIR/bin/npa_asr"
echo "[build] run the GUI:  $BUILD_DIR/bin/npa_asr"
echo "[build] transcribe:   $BUILD_DIR/bin/npa_asr -m model.gguf -i audio.wav"
