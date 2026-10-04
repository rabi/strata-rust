# Vendored from Strata (unmodified copies)

| File | Upstream md5 |
|---|---|
| `strata/platform/direct_file.cpp` | `06e75b780696a6df9f323c28e20ee185` |
| `strata/platform/direct_file.hpp` | `390283684b249076009f1350a471c277` |
| `strata/artifact/gguf_reader.hpp` | `33e11f406f792c8326ef5e28b623feac` |
| `strata/artifact/gguf_split.hpp` | `bef400d67b30696b1d4d7d278bda302b` |

Upstream: Strata @ `99f3dbd0b21d1401b3769e0c0d963913607f380b`.
Verify a copy is still pristine:

    cd <Strata> && md5sum src/platform/direct_file.cpp include/strata/platform/direct_file.hpp \
        include/strata/artifact/gguf_reader.hpp include/strata/artifact/gguf_split.hpp

They compile against nothing but the C++20 standard library (checked: their only
`#include "..."` lines are each other). `shim/build.sh` prefers a real Strata
checkout (`STRATA_REPO=..`) and falls back to this directory; the probe tests
cross-validate the Rust GGUF reader against the vendored reader, so drift shows
up as a hash mismatch, not a silent divergence. Refresh the copies and the
md5 table together when upstream changes.
