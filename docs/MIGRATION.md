# Migrating the Strata engine core from C++ to Rust

Status: Phase 0 in progress. This is the plan the code in this workspace follows;
every number below was measured on this checkout (`/home/ramishra/work/LLM/Strata`,
engine 0.1.26) on 2026-07-29, and the code cited as evidence is cited by path.

## 1. What the migration is for

Strata runs a 24,576-expert MoE on one consumer GPU plus system RAM. Two bug
classes have shipped in it, and both are memory-lifetime bugs that the Rust
borrow checker rejects at compile time:

- **issue #29** (fixed in 0.1.12): a race in the CPU expert pool on big-VRAM cards.
  A request stopped moving with the GPU at 100% and one CPU core busy; the user saw
  a hung answer. `docs/DETAILS.md:874`.
- **issue #31** (fixed in 0.1.14): with the IQ packs the host could wait forever
  inside the NVIDIA driver while copying experts during a verify window. The fix
  moved the copy into a GPU kernel; `--pcie-mode dma` restores the old path.
  `docs/DETAILS.md:874`.

Both are host-side orchestration code — the part that decides *which* expert
buffer is where, and *when*. Neither is a kernel bug. That is the argument for this
migration: rewrite the orchestration where the compiler enforces the lifetime
rules, and leave the numerics in the language they were tuned in.

## 2. Line counts, and which lines are in scope

Counted over `src/` and `include/` (`*.cpp *.hpp *.h *.cu *.cuh *.inc`):
**74,682 lines**, and the two groups below account for every one of them.

| Group | Lines | What it is | Rust? |
|---|---:|---|---|
| **A — stays C++/CUDA** | | | |
| `src/kernels/cuda` | 17,560 | every device kernel (nvcc) | no |
| `src/kernels/cpu` | 3,558 | AVX-512/AVX2 expert kernels, ggml i-quants | no |
| `src/kernels/*_parity.cpp`, `*_test.cpp`, `*.inc` | 13,855 | 34 parity harnesses + oracle vectors | no (driven from Rust) |
| `src/kernels/*.cpp` (other) | 611 | `ngram.cpp` and friends | no |
| `include/strata/kernels` | 4,009 | kernel decls → replaced by `strata_kernels.h` | no |
| `src/prefill/*.cu` | 3,830 | ggml CUDA host paths, MoE MMQ / fused-IQ | no |
| `src/prefill` (non-`.cu`) | 2,847 | prefill host orchestration + ggml CPU refs | no |
| `include/strata/prefill` | 483 | prefill decls → become ABI decls | no |
| `src/core/*.cu` | 1,035 | `device.cu`, `pinned.cu`: CUDA graphs + pinned arena | no |
| `src/platform` + header | 633 | `direct_file` (IO_uring/overlapped IO), pinned alloc | no |
| `include/strata/hip_compat` | 285 | HIP→CUDA shim | no |
| **A total** | **48,706** | | |
| **B — the Rust rewrite target** | | | |
| `src/core` host code | 10,815 | session, layer, weights, verify, expert_source, MTP, conversation memory | yes |
| `src/core/*_test.cpp`, `*_parity.cpp` | 977 | their tests | re-expressed as Rust tests |
| `include/strata/core` | 3,994 | the same, declared | yes |
| `src/program` + header | 7,053 | `generate.cpp` (6,913): CLI, pipe protocol, integration point | yes |
| `src/spec` + header | 718 | controller, draft policy, suffix drafter | **done** |
| `src/artifact` + header | 1,233 | GGUF reader, split, dequant refs | **reader done** |
| `src/ngram` + header | 945 | the 28.8 GB SSD n-gram table | yes |
| `src/plan` + header | 241 | | yes |
| **B total** | **25,976** | | |
| **C** | | | |
| `serve/*.py` | 8,961 | OpenAI/Anthropic API, web app | Rust later (axum), separate from A/B |

Group B is 35% of the codebase and 26,000-odd lines; group A is 65%. Group A is why
this is a *hybrid* migration and not a rewrite: `rustc` cannot compile a `.cu`
file, and 17.5k lines of hand-tuned kernels come with 34 oracle-backed parity
harnesses proving what they do. Those are an asset, not debt.

Evidence for the boundary being drawable where I draw it:

- **The CUDA API surface the core actually uses is small enough to put behind a
  vtable.** `grep -o '\bcuda[A-Za-z0-9_]*'` over `src/core/*.cpp` and
  `src/program/*.cpp` finds **93 distinct CUDA symbols**; the top of that list is
  the whole story: `cudaSuccess` 287, `cudaMalloc` 83, `cudaGetErrorString` 70,
  `cudaMemcpy*` ~130 across the three directions, `cudaStreamSynchronize` 45,
  `cudaEventRecord` 27, `cudaGraphLaunch` 20. Allocation, copies, streams, events,
  graph launch — the handful of verbs `strata_kernels.h` already exposes. It is not
  a call-per-kernel boundary.
- **But the boundary does not exist yet.** 12 core/program `.cpp` files include
  `<cuda_runtime.h>` *directly* (`expert_source.cpp:15`, `session.cpp:13`,
  `layer.cpp:36`, `weights.cpp:6`, `expert_cache.cpp:4`, `native_dense.cpp:6`,
  `native_head.cpp:6`, `concurrent_main.cpp:19`, `conversation_snapshot.cpp:4`,
  `overlap_main.cpp:15`, `generate.cpp:75`, plus the snapshot test) and 7 core
  headers do (`on_device.hpp:6`, `session.hpp:20`, `verify.hpp:31`, `mtp.hpp:26`,
  `graph.hpp:41`, `peer_experts.hpp:21`, `remote_experts.hpp:6`). So **Phase 2 is
  gated on a decoupling pass**: those CUDA calls move behind the vtable before the
  module can be ported, module by module. That is real work and the plan says so.
- **ggml** (`third_party/ggml`) is referenced by 33 files under `include/` and 61
  under `src/`, including `src/core/layer.cpp` and `src/program/generate.cpp`. So
  ggml sits inside group A (the prefill accelerator) with two group-B callers, which
  reach it through the ABI. It stays C++: it is a prefill speedup, not the product.
- **`windows.h`** appears in 16 files. The group-B ones are
  `conversation_memory.cpp`, `device_main.cpp`, `expert_source.cpp`,
  `generate.cpp` and the two artifact headers. Those become `windows-sys` bindings
  or move behind the platform shim.

## 3. The ABI

`crates/strata-device/include/strata_kernels.h` +
`crates/strata-device/src/abi.rs`.

**A vtable, not loose `extern "C"` symbols.** The C++ layer fills one
`StrataKernels` struct of function pointers at load time and returns it from
`strata_kernels_load(abi_version, out, err, err_len)`. Three reasons:

1. The Rust core **links with no CUDA present**. `StrataKernels::none()` fills
   every slot with `None`, and the pure-Rust crates build and test on a machine
   with no GPU and no toolkit — which is how this workspace is verified today.
2. **Version skew degrades instead of failing to load.** A shim that predates a slot
   leaves it null; the core checks and falls back (e.g. `sample_greedy_cluster`
   returns 0 → the caller uses the one-block argmax). A missing symbol would kill
   the process at load.
3. It gives one place to put `abi_version`, so the C++ side can refuse a core built
   against a header it does not implement.

**Slot order is append-only.** Never reorder or delete; that is what makes an old
`.so` next to a new binary still work. Four device-memory slots
(`device_alloc`/`device_free`/`memcpy_h2d_async`/`memcpy_d2h_async`) were appended
within ABI v1 — legal because the caller pre-zeroes `out` (`StrataKernels::none()`),
so an old shim whose `memset` covers only its own smaller `sizeof` cannot leave
garbage in the new tail. `abi.rs` pins the vtable at 30 slots (the
snapshot seam added `memcpy_default` and `device_sync`); the header says
the same with a `static_assert`. The claim is tested, not asserted: a 24-slot
shim (its `sizeof` = 192, its own header copy) loaded by the 30-slot Rust core
still drives the probe end-to-end and produces the identical whole-model hash
(`3b7b3afb6fe0711d`) as the current shim — bytes crossed the boundary through a
struct 32 bytes smaller than the core's own.

**Layout is pinned from both sides.** The header has `static_assert`s on
`sizeof`/`offsetof` (`SamplerParams` 64 bytes, `DeviceInfo` 232, `IoCompletion` 16);
`abi.rs` asserts the same numbers in `#[test]`. Verified: the header compiles clean
under `g++ -std=c++17` and `gcc -std=c11`.

Two ABI rules taken from the C++ code's own constraints, not invented:

- **`*_dev` vs `*_host` in the name, always.** Passing host memory where a device
  pointer is expected faults inside the kernel with an illegal memory access
  reported by a *later, unrelated* synchronising call. The naming is the only cheap
  defence, so it is part of the contract.
- **Anything that changes per round under graph capture is data in a device buffer,
  never a kernel argument.** `SamplerParams` is staged to device memory
  (`coupled_stage`) rather than passed by value into a captured kernel. This is why
  `coupled_draft_sample` takes `params_dev` and not `params`.

**What stays C++ behind the ABI on purpose:** CUDA graph capture and the
stream/event plumbing (`core/graph.hpp`, `core/device.cu`). CUDA graph tooling in
Rust is thin enough that rewriting it would be a bigger project than the win. Note
`stream_query_done`: the C++ side polls an event rather than blocking on a sync,
because a thread that only reads memory never makes the driver flush — measured, and
recorded in `core/graph.hpp`. The ABI exposes the poll, so the mistake cannot be
re-introduced from Rust.

**Not yet in the ABI:** the expert-pager's read path end-to-end (`file_submit` /
`file_wait` are here and now exercised end-to-end — see §4 Phase 0.5 — but the
`expert_source` policy that decides *what* to read is group B and not ported), the
n-gram table, and the prefill/ggml entry points. Those get added as their module
is ported, append-only.

**The ABI has a live implementation and a consumer.** `shim/strata_shim.cpp`
implements the header for real: CPU build links the engine's own
`src/platform/direct_file.cpp` (O_DIRECT worker-thread reader, unmodified) and
`posix_memalign` pinned memory, filling 8 of 30 slots; `STRATA_SHIM_CUDA=1`
adds device/stream/graph slots. `crates/strata-device/src/shim.rs` is the Rust
`dlopen`/`dlsym` loader (the one `#![allow(unsafe_code)]` module in the tree),
and `crates/strata-probe` is a binary that drives it. The handshake itself was
cross-checked with a standalone C shim: C reads the requested ABI number (1 and
2 both arrive correctly) and fills the struct Rust reads, field for field.

## 4. Phase order

**Phase 0 — pure logic, no device, no I/O. Done for these crates.**
`strata-spec` (controller, draft policy, suffix drafter), `strata-core`
(`penalty_rows`, coupled-draft cell/ring math), `strata-artifact` (GGUF v3 reader +
split resolver), `strata-device` (the ABI contract as types). 0 clippy warnings,
no `unsafe` outside `strata-device`'s loader module and one `#[cfg(test)]` block,
no external crates; the workspace now carries 58 tests (counted with the CPU shim
built; the 6 device tests skip without a CUDA shim).

Why first: these are the modules with the most logic per line and the least device
coupling, and every one ships with a C++ test whose assertions can be transcribed.
`SuffixDrafter` (open-addressed trigram table, 4-way, Murmur3 finalizer) is exactly
the kind of pointer-arithmetic table that issue #29-style bugs live in.

**Phase 0.5 — the IO half of the boundary, executable. Done.**
`./shim/build.sh` produces `libstrata_kernels.so` (CPU: 8/24 slots — file IO via
the engine's real `DirectFile`, pinned memory) and `cppdump` (dumps a GGUF
through the engine's real `gguf_reader.hpp`). `strata-probe` then reads a model
tensor-by-tensor over the vtable — submit/wait, completion tags, one reused
pinned 1 MiB buffer, aligned-window chunking that tolerates 32-aligned GGUFs and
short reads at EOF — and compares against `cppdump`. Measured on
`Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf`:
47 tensors, 3 metadata keys, 481280 bytes of payload, two independent readers
(O_DIRECT-through-ABI vs mmap-in-C++), identical fnv1a64 `3b7b3afb6fe0711d`.
Same checks run as `tests/shim_live.rs` (3 tests; they skip without
`STRATA_KERNELS_LIB`, so no-shim boxes stay green). Two real findings came out of
it: the engine's fnv1a64 seed is Strata's own constant (1469598103934665603), not
the FNV basis — both readers must copy the same one; and the probe's first draft
assumed 4096-aligned tensors, which real GGUFs (this one: align 32) are not.

With device-memory slots appended, `tests/gpu_abi.rs` (6 tests) runs the device
half of the vtable on a CUDA host: runtime report sanity, pinned-alignment at
32 MiB, a 1 MiB h2d/d2h round trip with a negative control (changed source must
change the device copy — a no-op memcpy returning OK fails), 100 graph replays
that must each move the bytes written *after* capture (the live-memory property
real replay needs), 4 concurrent streams without cross-contamination, and the
CPU shim answering 0/refusing instead of faking. Every test skips with a printed
reason when the shim has no device slots, so the workspace still tests green
with no GPU — as verified on the dev box (58 tests, 6 skips printed).

Measured on a GPU host (`giant18`): NVIDIA L4, cc 8.9, 58 SMs, 22.1 GiB, driver
reported as 13030, CUDA shim built with the vendored sources — all 6 device
tests pass through the vtable in 1.27 s, and the CPU-shim-refusal test's
self-check correctly detected the CUDA build and skipped itself.

`mkgguf` output is machine-independent, which is the point of hashing it: the
512 MB file hashes `30c489fd66fe95ad` identically on the dev box and on
`giant18`, and on both machines writer, Rust buffered reader, the engine's C++
mmap reader and the C++ O_DIRECT reader (536,871,408 bytes across 4,781 tensors)
produce that one value. A synthetic file that reproduces bit-exactly across
machines is reusable as a regression fixture.

**Phase 1 — Rust replaces `serve/`.** Easiest real win, and the only phase with no
GPU in the loop. `serve/server.py:338` already launches the engine as
`subprocess.Popen([exe, "--serve", *args], stdin=PIPE, stdout=PIPE)`. Because that
is a *pipe protocol*, not a library call, an axum server talking the same protocol
to the *existing C++ engine* is a complete, shippable replacement with the engine
untouched. `serve/winjob.py`'s Windows job-object handling and the OpenAI +
Anthropic SSE framing are the two things that actually need care.

**Phase 2 — Rust owns the loop over the ABI.** Port `core/weights`,
`core/expert_source` (the pager policy — the issue #29/#31 code), `core/session`,
`core/layer`, `core/verify`, `core/mtp` one module at a time, each behind the
existing parity harness. This is where Rust earns its keep: the expert-buffer
lifetime that took two releases to fix becomes a borrow-checker error at compile time.

Phase 2 goes smallest-CUDA-coupling first, so the vtable grows gradually and each
step is reviewable. Uses of `cuda*`/`hip*` symbols per file (excluding the header
name itself), measured:

| Module | CUDA uses | Distinct symbols | Lines | Order |
|---|---:|---:|---:|---|
| `conversation_memory.cpp` | 0 | 0 | 51 | 1 |
| `load_main.cpp` | 0 | 0 | 122 | 1 |
| `layout.cpp` | 0 | 0 | 173 | 1 |
| `conversation_state.cpp` | 7 | 5 | 305 | 2 |
| `conversation_snapshot.cpp` | 8 | 7 | 268 | 2 |
| `native_dense.cpp` | 12 | 7 | 195 | 2 |
| `weights.cpp` | 32 | 7 | 479 | 2 |
| `expert_cache.cpp` | 51 | 17 | 447 | 3 |
| `expert_source.cpp` | 58 | 22 | 2,707 | 3 |
| `layer.cpp` | 115 | 21 | 1,321 | 4 |
| `verify.cpp` | 147 | **51** | 1,373 | 4 |
| `mtp.cpp` | 170 | 39 | 895 | 4 |
| `session.cpp` | 220 | 36 | 970 | 5 |
| `generate.cpp` | 340 | 50 | 6,913 | 6 (Phase 3) |

`verify.cpp` (51 distinct symbols) and `mtp.cpp` (39, most of them graph capture)
are the two that need the widest vtable, which is why they come after
`weights`/`expert_source` have already forced most slots into existence. The
zero-coupling files (`conversation_memory`, `load_main`, `layout`) port with no ABI
work at all and are the right way to start Phase 2.

**Phase 2, module 1 — done: `conversation_memory` -> `strata-core::host_memory`.**
The conversation cache's physical-RAM admission gate, 51 C++ lines with zero CUDA
symbols. `mem_available` takes any `BufRead` (the C++ took an `istream`),
`memory_admit` keeps the ordering that is the overflow defense (`available >=
floor` first, then `allocation <= available - floor`), and unknown telemetry
still fails closed — a machine that will not report free memory does not park
caches. All 23 boolean checks from `conversation_memory_test.cpp` are
transcribed 1:1 (11 malformed-telemetry cases, the boundaries, the u64 overflow
probes, the torn-stream variant of the badbit check); the C++ test was built and
run independently here first (23/23). The Windows provider branch
(`GlobalMemoryStatusEx`) type-checked by compiling the cfg-flipped module with
`--emit=metadata`, since the Windows rustup target would not install on this
box; the layout assert caught a real error in the mirror struct (72 -> 64 bytes)
before it shipped. The port adds one check the C++ could not make portably: on
Linux the provider is not allowed to say unknown, because /proc/meminfo is
always there. 67 workspace tests, 0 clippy, fmt clean.

**Phase 2, module 2 — done: `layout.cpp` -> `strata-core::layout`.** 173 C++
lines, zero CUDA symbols, and no C++ test to transcribe, so the gate is a spec
harness built for the purpose (`tools/layout_corpus.cpp`): it runs the real
`layout.cpp` and prints the byte plan for every shape the port claims to
reproduce, and `tests/layout_corpus.rs` replays the same shapes and compares. 14
golden lines, checked in. The same pattern — build a harness when no test exists —
is what every later module without a C++ test will use.

**Phase 2, module 3 — done: `conversation_state.cpp` + `conversation_snapshot.cpp`
-> `strata-core::{conversation, conversation_kv, conversation_state}`.** 573 C++
lines, 15 CUDA calls between them, all of them behind one seam: the `Device` trait
(read/write for K/V blocks, read/write for the running state, sync, the two
residency hooks). The core stays pointer-free — device memory is an opaque address
the core selects and never dereferences — so the same code runs against the shim on
a GPU and against named byte buffers on the host. The gate is
`tools/conversation_corpus.cpp`: it compiles the real `.cpp` files against stub CUDA
symbols, runs 24 fixtures (every format x expert-count x zero-QSA x ple combination)
through the byte estimates, the validation rejections, the fault-injected transfer
sequences and the read-back verification, and prints 2,750 lines. The Rust replay
matches all of them byte for byte — including the error strings, the exact copy
sequence (buffer, offset, length), and the fnv1a64 of every buffer after every phase.
Two things the replay caught that a unit test would not: the running-state copies
address each layer's OWN buffer at offset 0 while the checkpoint's vector is the
concatenated one, and libstdc++ copy-construction allocates exactly `size`, which is
a different capacity from growing to it — and `bytes()` counts capacity, so the
difference is a real admission number. 75 workspace tests, 0 clippy, fmt clean.

Module 3's device half then got a real implementation: `strata-device::snapshot::
SnapshotDevice` resolves `(layer, pool-set, region)` to an address once and moves
every byte through two new append-only slots, `memcpy_default` and `device_sync`
(28 -> 30). The core never sees an address; the seam never sees a format branch.
It refuses what it cannot serve at construction — a streamed or ring layer needs
`kv_stream_reset`/`kv_ring_restore`, which have no slots yet, so its image could
never be verified readable — and it range-checks every transfer before the DMA.

The seam is gated twice. On the dev box, `tests/snapshot_device.rs` injects a fake
vtable over host memory so every branch of the seam runs without a GPU: the save
moves the golden's 10,440 bytes and reproduces its fingerprints (K
`6404351448661373315`, pooled `9014609365494796707`, gdn `7415105493412125315`);
the restore replays the golden's 17-copy sequence with the same sizes in the same
order (10,504 bytes - the 64-byte difference is the two moving pooled rows a
restore rebuilds and a save never reads); the first copy of a save is the gdn row,
so a failure there reports the running-state prefix, exactly as the golden's
`SAVE_INCREMENTAL|copy_failure_ok=0` line does. On `giant18`,
`tests/gpu_snapshot.rs` runs the same fixture over real `cudaMalloc`,
`cudaMemcpy(..., cudaMemcpyDefault)` and `cudaDeviceSynchronize`: save reads the
golden bytes over real DMA, the buffers are overwritten with a different pattern,
restore puts the originals back in VRAM. Measured: 2/2 pass in 0.26 s on the L4
with a 25-of-30-slot CUDA shim, and the 6 earlier ABI tests still pass in 1.13 s.
82 workspace tests, 0 clippy, fmt clean.

**Phase 3 — `generate.cpp` last.** It is 6,913 lines, it is the integration
point (CLI, pipe protocol, prefill orchestration, image handling), and it is the
single most CUDA-coupled file in group B (340 uses, 50 distinct symbols). Porting
it early means porting against moving targets.

## 5. Port method, and why it is trustworthy

Each ported module carries a `#[cfg(test)]` module transcribed from the matching
C++ test file, and the C++ test is *additionally* built and run against the Rust
numbers where the C++ test prints them rather than asserting them.

Example: `src/spec/controller_test.cpp` prints its decisions for a human to read
(`controller_test.cpp:55-70`). `crates/strata-spec/src/parity.rs` re-runs the same
scenarios and asserts the printed values — the full 7-point acceptance sweep
(`p=0.50 → none k=0, 48.8 tok/s` … `p=0.95 → mtp k=5, 84.6 tok/s`) to 0.05 tok/s,
using the *same* LCG as the C++ loop (`r = 2654435761*(i+1)`, then
`r = r*1103515245+12345`, accept while `(r>>8)%1000 < p*1000`) so both
implementations consume the identical acceptance stream. That test passes.

Where a C++ test is boolean-only (`draft_policy_test.cpp`, `suffix_drafter_test.cpp`
check conditions and print ok/FAIL), the conditions are ported as-is — 20 spec tests
green, and the C++ binaries pass independently (`controller tests: OK`, `PASS`,
`suffix_drafter unit tests: OK`).

The 34 `*_parity.cpp` kernel harnesses stay C++ and become the Phase-2 gate: a
ported module is done when the harness it feeds still passes, driven from
`cargo test` as an external binary.

## 6. Constraints actually hit

**crates.io was blocked while Phase 0 was written** (CONNECT tunnel failed, 403
through the proxy), so every Phase-0 crate is **pure `std`, zero dependencies** —
and that stays true now that the registry is allowlisted, because the zero-dep
constraint is what forced the reader design below.

- The GGUF reader uses `File` + `seek` + bounded reads through a `Cursor` with an
  explicit end, not `mmap`. Consequence and reason: **`GgufFile` deliberately has no
  `tensor_data()` accessor returning a pointer into the file.** With mmap the C++
  version hands out `const uint8_t*` whose validity is nobody's contract; the Rust
  version returns offsets only, and tensor payload bytes are read through
  `strata-device`'s direct-IO path (`file_submit`), which is where the alignment
  requirement (`file_alignment()`, `platform/direct_file.hpp`) actually applies.
  This is *better* than the C++ API, and it is a direct result of not being able to
  pull `memmap2`. When the registry is reachable, `memmap2` can be added behind
  `Cursor` without changing a caller.
- No `thiserror`: errors are `String`. Ugly, and honest for Phase 0.
- No `bindgen`/`cc`: the header is hand-mirrored and pinned by static asserts on
  both sides. Once bindgen is available, the same struct definitions should generate
  the header instead of being checked against it.

**`unsafe_code` is `deny`-ed at the workspace level** (`[workspace.lints.rust]`). It
is `deny`, not `forbid`, because `forbid` cannot be relaxed even by a test, and
calling through a vtable slot is inherently `unsafe` — the one exception is
`strata-device`'s `#[cfg(test)] mod tests`. Everywhere else in the tree, and in all
non-test code including `strata-device`'s own, there is no `unsafe` block: the crate
is a data contract that never calls through a pointer. Checked with
`grep -rn 'unsafe {' crates/*/src` → one hit, inside that test module.

**Design doc decisions that are load-bearing, not taste:**

- ggml stays C++ (see §2). Rewriting i-quants/ggml in Rust is a separate project
  with a large perf risk and no safety upside — the safety bugs were in the host
  orchestration.
- CUDA graphs stay C++ (§3).
- The reader never mmaps (§6).
- Phase 1 is the Python server because the pipe protocol is already a process
  boundary — the one place a whole subsystem can be swapped with zero engine risk.

**Phase 2, module 4 — done: `native_dense.cpp` + `weights.cpp` ->
`strata-core::{native_dense, weights, native_mm}`.** 674 C++ lines, 44 CUDA calls
between them, 7 distinct symbols each — all of them slots the Phase-0 vtable already
has, so this module added no ABI. It is the first ported module that touches the
GGUF reader (`native_dense.cpp` includes `gguf_reader.hpp`), so `strata-core` now
depends on `strata-artifact`, which is the direction the C++ already went and which
still leaves `strata-artifact` dependency-free.

Neither file had a C++ test to transcribe, so the gate was built for the purpose:
`tools/native_dense_corpus.cpp` compiles the real `.cpp` files on the host with the
CUDA calls `--wrap`'ped onto one slab, takes the block sizes from `ggml-common.h`
(`GGML_COMMON_DECL_C`, so no CUDA headers are needed), and runs five scenarios —
the byte tables, the two pack fixtures, every tensor kind the loader can meet, every
refusal message, the split arbitration, the layer range, and the span checks — and
prints 426 lines. The fixture files it writes are embedded in the golden as hex, so
`tests/native_dense_corpus.rs` rebuilds them and replays the whole thing through the
Rust port. The two line streams are identical.

Three things the replay caught that a unit test would not, all of them in the Rust:

- `sscanf` reports the number of conversions it performed, so a row with trailing
  fields reports the format's own width (19), not its field count. The Rust counted
  fields, which turned every well-formed row of the real index format into a refusal.
- `atoi(name + 4)` reads the leading digits and stops, so `blk.2.attn_q.weight` is
  layer 2. The Rust parsed the whole tail, failed, and fell back to 0 — which put a
  layer-2 tensor inside a `(0, 1)` range.
- The exact fp16 -> f32 widening computed `ex - 15 + 127` unsigned and underflowed for
  every exponent below 15. It is `ex + 112`.

One divergence is deliberate and is not a bug: the C++ threads one `err` string
through `WeightTable::load` and `NativeDense::load`, and `check_architecture` assigns
its result to it, so a load that reaches a valid architecture CLEARS a message an
earlier call left there. The Rust returns `Result` instead; the harness models the
clearing where the C++ did it, and the golden line records the behaviour.

84 workspace tests, 0 clippy warnings, fmt clean.

**The vtable grew from 30 slots to 50 in one append, before the modules that need
it.** The remaining Phase-2 files use 45 distinct CUDA functions between them; 16
were already covered, so 20 went in (`get_device`, `get_last_error`,
`peek_last_error`, `mem_get_info`, `host_register`/`unregister`/`get_device_pointer`,
`memset`/`memset_async`/`memcpy2d_async`, `stream_create_with_flags`/`stream_sync`/
`stream_wait_event`, `event_create`/`destroy`/`record`/`sync`/`query_done`/
`elapsed_ms`, `graph_upload`). Each signature was taken from a real call site, not
from the CUDA docs — `graph_upload` uploads the instantiated exec because that is
what `mtp.cpp` passes, and `event_create(out, 0, ..)` is `cudaEventCreate` so the
flags-0 and flags-N callers share one slot.

`cudaGetLastError` and `cudaPeekAtLastError` are both slots because they are not
interchangeable and the engine uses both — one clears the sticky error, one does
not, and `tests/gpu_slots.rs` tests exactly that difference.

Four functions the remaining modules use are deliberately **not** slots:
`cudaGraphGetNodes`, `cudaGraphNodeGetType`, `cudaGraphKernelNodeGetParams`,
`cudaLaunchHostFunc` (all `verify.cpp` only) and `cudaGetDriverEntryPoint[ByVersion]`
(`expert_cache.cpp` only). Those are the shape problem: "inspect a captured graph
node" and "load a driver symbol by name" cannot go behind an opaque handle without
mirroring the API they wrap, which is the decision `verify.cpp` needs before it
ports. The other five files — `pinned`, `expert_source`, `layer`, `mtp`, `session`
— are covered by the 50 slots as they stand.

`tests/gpu_slots.rs` exercises every new slot on real hardware (one pitched 2D copy
with a checked destination grid, a memset round trip, an event-timed stream, a
gated second stream, an upload-then-replay that must move live bytes). It skips on
a CPU shim build, so it needs one run on the GPU box.

**Phase 2, module 5 — done: `expert_source.cpp`'s CPU-only policy ->
`strata-core::expert_plan`.** 241 C++ lines, and unlike every module before it,
zero CUDA calls are reachable from any of them — the header says the planner is
"kept CPU-only so selection and byte accounting can be tested without initializing
a GPU", so this port adds no ABI and no device work at all. What moved: the host-RAM
gate (`host_available_memory`, `cgroup_available_bytes`), the cache-complement
planner, the complement-or-mapped resolution, the adaptive-tier swap, and the
resident keep decision.

The gate is `tools/expert_plan_corpus.cpp`: it compiles the real
`expert_source.cpp` on the host (CUDA symbols stubbed to abort-if-called, the
kernel/platform symbols left unresolved because nothing under test reaches them),
runs 56 observations over all five functions, and prints one line each. The fake
`/proc` and cgroup tree it writes is embedded in the golden as hex, so
`tests/expert_plan_corpus.rs` rebuilds it and replays every case. The two line
streams are identical.

What the corpus pins down that reading the code would have let me guess wrong:

- `root / "/foo"` *replaces* the root — appending an absolute path to a path is an
  assignment in `std::filesystem`. So a cgroup group only resolves when its path is
  literally under the mount root, and the containment check that follows is a
  **string** prefix test, not a component test. A process in a nested cgroup whose
  path is not under `/sys/fs/cgroup` fails the whole read.
- `memory.max` reading `max` skips the accounting for that group and carries on up;
  the root group is allowed to have no `memory.max` at all, but only when
  `cgroup.controllers` is there to prove it is a real cgroup root. Remove that file
  and the same case returns false.
- `>>` into a `uint64_t` accepts a leading `-` and wraps; `stoull` throws. Same
  bytes, different verdicts, and the two paths sit three lines apart.
- `/proc/meminfo` here takes the LAST well-formed line and skips malformed ones. The
  `conversation_memory.cpp` parser ported in module 1 rejects duplicates and fails
  closed. Two parsers, two files, two behaviours — both ported as they are, not
  unified.
- `mark_pairs` short-circuits: a primary-tier refusal means the additional tier is
  never examined, so the duplicate-primary error wins over a duplicate in the other
  list.

`pinned`'s deterministic core (the read plan, the checksums, the cap and slice
arithmetic — ~150 of its 725 lines) still folds in behind an `Arena` trait; that is
the same module's second half and has not moved yet.

96 workspace tests (95 before this module), 0 clippy warnings, fmt clean.

**Phase 2, module 5b — done: `pinned.cu`'s deterministic core ->
`strata-core::pinned`.** The gate worked because `pinned.cu` has no `__global__`
and no `__device__` in it — it is host code that *calls* the runtime — so g++
compiles it against the stub headers with `-x c++` and the harness calls the real
functions. What moved: the pin cap, the slice bounds, the read plan with its
per-layer FNV-1a, and the unbuffered gate. What did not: mmap, madvise, hugepages,
the working-set lock, the timed probes — the platform boundary itself.

The `Arena` trait this was going to be folded in behind turned out not to be
needed. `load_experts_ranges` takes its destination buffer from the caller, so the
seam is a `&mut [u8]`, not a trait. The trait belongs to the arena *lifecycle*,
which has not moved.

`tools/pinned_corpus.cpp` prints 25 observations; the timing fields are excluded
because they measure the machine. The fixture files are embedded in the golden as
hex and `tests/pinned_corpus.rs` rebuilds them.

What the corpus caught, all of it in the Rust:

- The checksum vector is indexed by layer, not appended to. The first version
  pushed, and every success case came back with double the entries.
- The checksums are only moved on the **success** path — a refused load reports
  none — while `bytes` and `layers` are filled before the read starts and so
  survive the failure. Two fields zeroed on error, two not, and the difference is
  the whole contract with the caller.
- `atoi("abc")` is `0`, and `0` is not negative, so a garbage
  `STRATA_ARENA_PIN_GIB` pins zero GiB. Only an empty value means unset.
- Seeking past EOF on a regular file *succeeds*. The read is what reports it, so
  the "seek failed" message is unreachable on that path — the refusal that fires is
  the short read, naming the offset the seek landed on.
- The plan is thread-count independent: the one-thread and four-thread cases hash
  identically. That is what lets the port run it serially without changing an answer
  — the parallelism stays with whatever owns the destination buffer, and `threads`
  is kept in the signature because the C++ has it.

The `#ifdef _WIN32` branches (`sliced_pin_limit`, the unbuffered `ReadFile` path,
the DXGI budget) are compiled out here and are **not** covered by this golden.

99 workspace tests (96 before this module), 0 clippy warnings, fmt clean.
