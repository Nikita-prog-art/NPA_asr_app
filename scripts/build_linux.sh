#!/usr/bin/env bash
# Build npa_asr on Linux.
#
#   scripts/build_linux.sh [extra cmake args...]
#
# Environment:
#   ASR_CPU_ARCH  x86-64-v3 (default) | native | x86-64-v2 | x86-64
#                 native = tuned for this machine only, no fallback
#   ASR_NATIVE    ON is a shorthand for ASR_CPU_ARCH=native
#   ASR_COMPILER  gcc | clang            (default: gcc)
#   ASR_JOBS      parallel build jobs    (default: derived from the RAM size)
#   BUILD_DIR     build directory        (default: <repo>/build)
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"

COMPILER="${ASR_COMPILER:-gcc}"
case "$COMPILER" in
    clang) CC_BIN=clang; CXX_BIN=clang++ ;;
    *)     CC_BIN=gcc;   CXX_BIN=g++ ;;
esac
echo "[build] compiler: $CC_BIN / $CXX_BIN"

ARCH_ARG=()
if [ -n "${ASR_CPU_ARCH:-}" ]; then
    ARCH_ARG+=(-DASR_CPU_ARCH="$ASR_CPU_ARCH")
fi
if [ "${ASR_NATIVE:-OFF}" != "OFF" ]; then
    ARCH_ARG+=(-DASR_NATIVE=ON)
fi

# One ggml CPU backend is compiled for a fixed ISA baseline (~1 GB per job);
# budget 2 GB per job anyway so a memory-limited WSL VM is not swapped to death.
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
    "${ARCH_ARG[@]}" "$@"

cmake --build "$BUILD_DIR" -j "$JOBS"

echo
echo "[build] done: $BUILD_DIR/bin/npa_asr (self-contained, no .so next to it)"
echo "[build] run the GUI:  $BUILD_DIR/bin/npa_asr"
echo "[build] transcribe:   $BUILD_DIR/bin/npa_asr -m model.gguf -i audio.wav"
