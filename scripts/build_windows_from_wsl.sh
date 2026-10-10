#!/usr/bin/env bash
# Build npa_asr for Windows *from inside WSL*.
#
#   scripts/build_windows_from_wsl.sh [extra cmake args...]
#
# MSVC cannot compile from a UNC path (\\wsl.localhost\...), and the sandbox /
# WSL filesystem is slow to walk file by file, so this script
#   1. packs the sources into one tarball (fast, single file over the 9p bridge),
#   2. copies it to a Windows directory and extracts it there,
#   3. runs the normal Windows build (scripts/build_windows.cmd) on the host.
#
# Environment:
#   WIN_BUILD_DIR  Windows target directory (default: C:\dists\builds\npa_asr_build)
#   ASR_JOBS       parallel jobs passed to the Windows build
#   WSL_DISTRO     distro name used for the \\wsl.localhost path (default: Debian)
#   WIN_SRC        repository path as seen from Windows (default: current dir)
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WIN_BUILD_DIR="${WIN_BUILD_DIR:-C:\\dists\\builds\\npa_asr_build}"
WSL_DISTRO="${WSL_DISTRO:-Debian}"
PS="${PS_EXE:-/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe}"
TMP_TGZ="$ROOT/.win_src.tgz"

echo "[win] packing sources"
tar czf "$TMP_TGZ" -C "$ROOT" \
    --exclude='third_party/ggml/.git' --exclude='third_party/whisper.cpp' \
    --exclude='third_party/_fetched' --exclude='*.gguf' --exclude='.ref' \
    CMakeLists.txt cmake src tools scripts .github README.md third_party

# UNC path of the tarball for PowerShell
UNC_TGZ="\\\\wsl.localhost\\${WSL_DISTRO}\\$(echo "$ROOT" | sed 's|^/||; s|/|\\\\|g')\\.win_src.tgz"

"$PS" -NoProfile -Command "
  \$dst = '${WIN_BUILD_DIR}';
  New-Item -ItemType Directory -Force -Path \$dst | Out-Null;
  Copy-Item '${UNC_TGZ}' (Join-Path \$dst 'src.tgz') -Force;
  tar -xzf (Join-Path \$dst 'src.tgz') -C \$dst;
  Remove-Item (Join-Path \$dst 'src.tgz');
  Write-Output ('[win] extracted to ' + \$dst)
"
rm -f "$TMP_TGZ"

echo "[win] building on the host"
cd /mnt/c/Windows/System32
./cmd.exe /c "${WIN_BUILD_DIR}\\scripts\\build_windows.cmd $*" 2>&1 | tr -d '\r'

echo
echo "[win] done: ${WIN_BUILD_DIR}\\build-windows\\bin\\npa_asr.exe"
