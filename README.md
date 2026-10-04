# strata-rust

Rust engine core for [Strata](../Strata), with the CUDA/HIP kernels left in C++
behind a C ABI. Plan and measured line counts: **[docs/MIGRATION.md](docs/MIGRATION.md)**.

## What runs today

The ABI is not a paper contract any more — a C++ shim implements it and a Rust
binary drives real files through it:

```
./shim/build.sh                       # -> target/shim/libstrata_kernels.so + cppdump
STRATA_KERNELS_LIB=$PWD/target/shim/libstrata_kernels.so \
  cargo test --offline                # 58 tests incl. live-ABI round trips
STRATA_KERNELS_LIB=$PWD/target/shim/libstrata_kernels.so \
  cargo run --release -p strata-probe -- <file.gguf> target/shim/cppdump
```

`shim/build.sh` needs two pieces of real Strata C++ (the `DirectFile` reader,
`gguf_reader.hpp`). It uses a Strata checkout when reachable
(`$STRATA_REPO`, else `../Strata`) and otherwise the unmodified copies in
`shim/vendor/` — verified: a tree with no Strata checkout builds, tests 58/58,
and the probe+cppdump produce the identical whole-model hash below.

The probe opens a GGUF through the *real engine reader*
(`strata::platform::DirectFile`, O_DIRECT, worker-thread completions) over the
vtable, streams every tensor payload through one pinned 1 MiB buffer in aligned
chunks, and cross-checks the result — header, per-tensor table, and a whole-file
hash — against the independent C++ mmap reader (`tools/cppdump.cpp`, which uses
the engine's own `gguf_reader.hpp`). Measured on
`Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf`:
47 tensors, 3 metadata keys, 481280 bytes, two readers, identical fnv1a64
(`3b7b3afb6fe0711d`). The CPU shim fills 8 of 28 slots (file IO + pinned
memory); unset `STRATA_KERNELS_LIB` and the live tests skip, so `cargo test`
stays green on boxes with no shim build.

| Crate | What it is | C++ source |
|---|---|---|
| `strata-spec` | speculation control: `Controller`, `DraftPolicy`, `SuffixDrafter` | `src/spec`, `include/strata/spec` |
| `strata-core` | host-side sampler math: `penalty_rows`, coupled-draft cell/ring | `include/strata/core/coupled_draft.hpp`, `include/strata/kernels/sampler.hpp` |
| `strata-artifact` | GGUF v3 reader + split-shard resolver | `include/strata/artifact/gguf_reader.hpp`, `gguf_split.hpp` |
| `strata-device` | the kernel/device ABI as `repr(C)` types, header, `dlopen` loader (`shim.rs`) | the boundary between `src/core` and `src/kernels` |
| `strata-probe` | binary: drives the shim end-to-end, cross-validates against C++ | exercises `include/strata_kernels.h` |

Status: **Phase 0** (pure logic) done for the four library crates; the IO half
of the **Phase 2 boundary** is now executable and tested on both sides of the
ABI. `unsafe_code` is denied workspace-wide; `unsafe` lives only in
`strata-device`'s loader (one `#![allow]` module) and one `#[cfg(test)]` block.

## GPU host

`STRATA_SHIM_CUDA=1 ./shim/build.sh` compiles the device slots
(`cudaMalloc`/streams/graph capture/sampler + the appended device-memory and
async-DMA slots; 28 slots total). Then the GPU test suite:

```
STRATA_SHIM_CUDA=1 CUDA_HOME=/usr/local/cuda ./shim/build.sh
STRATA_KERNELS_LIB=$PWD/target/shim/libstrata_kernels.so \
  cargo test -p strata-device --test gpu_abi -- --nocapture
```

`gpu_abi` (6 tests) proves, through the ABI only: the runtime reports the GPUs
with sane numbers; pinned allocations are page-aligned and writable at 32 MiB;
a 1 MiB h2d/d2h round trip preserves bytes and a *changed* source changes the
device copy (a silent no-op memcpy cannot pass); a captured graph replays 100x
and each replay moves the bytes written *after* capture — the property real
graph replay needs; four concurrent streams copy independently without
bleeding into each other; and a CPU shim answers 0 devices and refuses
stream/alloc calls instead of faking them. On a CPU-only shim build every test
prints a skip and the suite stays green, so `cargo test` never needs a GPU.

The probe prints `device_count` and per-GPU name/arch/memory through the ABI's
`device_info` slot — first proof the device half of the vtable answers.

## Layout pinning

`crates/strata-device/include/strata_kernels.h` and
`crates/strata-device/src/abi.rs` describe the same structs. The header carries
`static_assert`s on `sizeof`/`offsetof`; the Rust tests assert the same numbers.
The shim is compiled against the same header, so all three sides are checked at
build time. Verified to compile as both C11 and C++17:

```sh
gcc -std=c11  -fsyntax-only -I crates/strata-device/include -x c  <(echo '#include <strata_kernels.h>')
g++ -std=c++17 -fsyntax-only -I crates/strata-device/include -x c++ <(echo '#include <strata_kernels.h>')
```
