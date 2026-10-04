#!/usr/bin/env bash
# Build the C++ shim (and cppdump) into target/shim/. With CUDA available set
# STRATA_SHIM_CUDA=1 to compile the device slots too.
#
# The shim needs two pieces of real Strata C++: the DirectFile reader (linked
# into the shim) and gguf_reader.hpp (used by cppdump). Both come from a Strata
# checkout if one is reachable - $STRATA_REPO, else ../Strata - and from the
# unmodified copies in shim/vendor/ otherwise (see shim/vendor/VENDORED.md).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
out="$root/target/shim"
mkdir -p "$out"

strata="${STRATA_REPO:-$root/../Strata}"
if [[ -d "$strata/include/strata/platform" ]]; then
  inc="$strata/include"
  direct_cpp="$strata/src/platform/direct_file.cpp"
  echo "using Strata checkout at $strata"
elif [[ -n "${STRATA_REPO:-}" ]]; then
  echo "STRATA_REPO=$STRATA_REPO set but has no include/strata/platform - refusing" >&2
  exit 1
else
  inc="$here/vendor"
  direct_cpp="$here/vendor/strata/platform/direct_file.cpp"
  echo "using vendored Strata sources (shim/vendor)"
fi

CXXFLAGS=(-std=c++20 -fPIC -O2 -I "$here" -I "$root/crates/strata-device/include" -I "$inc")
if [[ "${STRATA_SHIM_CUDA:-0}" == 1 ]]; then
  CXXFLAGS+=(-DSTRATA_SHIM_CUDA -I "${CUDA_HOME:-/usr/local/cuda}/include")
  LDFLAGS=(-L "${CUDA_HOME:-/usr/local/cuda}/lib64" -lcudart)
else
  LDFLAGS=()
fi

"${CXX:-g++}" -shared "${CXXFLAGS[@]}" "$here/strata_shim.cpp" "$direct_cpp" \
  -pthread "${LDFLAGS[@]}" -o "$out/libstrata_kernels.so"
echo "built $out/libstrata_kernels.so"

"${CXX:-g++}" "${CXXFLAGS[@]}" -O2 "$root/tools/cppdump.cpp" -o "$out/cppdump"
echo "built $out/cppdump"
