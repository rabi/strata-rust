#!/usr/bin/env bash
# Build the C++ shim (and cppdump) into target/shim/. With CUDA available set
# STRATA_SHIM_CUDA=1 to compile the device slots too.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
strata="${STRATA_REPO:-$root/../Strata}"
out="$root/target/shim"
mkdir -p "$out"

CXXFLAGS=(-std=c++20 -fPIC -O2 -I "$here" -I "$root/crates/strata-device/include" -I "$strata/include")
if [[ "${STRATA_SHIM_CUDA:-0}" == 1 ]]; then
  CXXFLAGS+=(-DSTRATA_SHIM_CUDA -I "${CUDA_HOME:-/usr/local/cuda}/include")
  LDFLAGS=(-L "${CUDA_HOME:-/usr/local/cuda}/lib64" -lcudart)
else
  LDFLAGS=()
fi

"${CXX:-g++}" -shared "${CXXFLAGS[@]}" "$here/strata_shim.cpp" "$strata/src/platform/direct_file.cpp" \
  -pthread "${LDFLAGS[@]}" -o "$out/libstrata_kernels.so"
echo "built $out/libstrata_kernels.so"

"${CXX:-g++}" "${CXXFLAGS[@]}" -O2 "$root/tools/cppdump.cpp" -o "$out/cppdump"
echo "built $out/cppdump"
