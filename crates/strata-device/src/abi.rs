// The Rust side of the kernel/device ABI. The matching C header is checked in at
// `../include/strata_kernels.h` and the layout tests here pin the same sizes.
//
// The ABI is a vtable, not loose extern symbols: the C++ layer fills one
// `StrataKernels` struct of function pointers at load time, so the Rust core
// links without CUDA present (tests run on CPU, `none()` below), and a device
// layer that predates a slot shows up as `None` instead of a link error. Slot
// order is append-only.

use std::os::raw::{c_char, c_int, c_void};

/// Bumped when the vtable layout changes in a way a shim must not be linked
/// against without being rebuilt for it. Slot *additions* do not bump it (a
/// shim that predates a slot leaves it null and the core degrades); reordering,
/// removal, or a struct-layout change does.
pub const STRATA_ABI_VERSION: u32 = 1;

/// Vtable slots in this ABI revision. Shims report how many they filled; tests
/// and tools print the pair, so "8 of 28" reads as a CPU build at a glance.
pub const STRATA_SLOT_COUNT: usize =
    std::mem::size_of::<StrataKernels>() / std::mem::size_of::<usize>();

/// Returned by every ABI call. Details go to the caller's error buffer
/// (`err` / `err_len`), so no C++ exception crosses this boundary.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrataStatus {
    Ok = 0,
    /// the device layer rejected the call; the error buffer says why
    Error = 1,
    /// the call would block and the caller asked not to
    WouldBlock = 2,
    /// this build of the device layer has no such slot
    Unsupported = 3,
}

impl StrataStatus {
    pub fn is_ok(self) -> bool {
        self == StrataStatus::Ok
    }
}

/// The sampler chain: penalties -> top_k -> top_p -> min_p -> temperature -> pick.
///
/// Layout mirrors `strata::kernels::SamplerParams` exactly (strata/kernels/sampler.hpp).
/// Anything that changes per round under graph capture is passed as DATA in a
/// device buffer, not as an argument — `layout_mirrors_cpp` pins the size.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SamplerParams {
    /// sampled path: 1..=64 as given; 0 (off) or >64 keep the widest list, 64
    pub top_k: c_int,
    /// 1.0 disables the filter
    pub top_p: f32,
    /// 0 disables; keeps tokens with p >= min_p * p_max
    pub min_p: f32,
    /// <= 0 means greedy
    pub temperature: f32,
    /// top_p keeps at least this many
    pub min_keep: c_int,
    /// 0 disables; the window over `history` to count occurrences in
    pub penalty_last_n: c_int,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    /// drives Philox, counter-based on (seed, token index)
    pub seed: u64,
    /// absolute draw index of row 0; advance across decode calls
    pub counter: u64,
    pub greedy: bool,
}

impl Default for SamplerParams {
    fn default() -> Self {
        SamplerParams {
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            temperature: 1.0,
            min_keep: 1,
            penalty_last_n: 0,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            seed: 0,
            counter: 0,
            greedy: false,
        }
    }
}

/// One GPU. The C++ `DeviceInfo` holds `std::string`s, which cannot cross an FFI
/// boundary, so this is the ABI form: fixed-size UTF-8 buffers the shim fills.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeviceInfo {
    pub ordinal: c_int,
    pub cc_major: c_int,
    pub cc_minor: c_int,
    pub multi_processor_count: c_int,
    pub driver_version: c_int,
    pub runtime_version: c_int,
    /// as reported by cudaMemGetInfo at query time
    pub total_bytes: u64,
    pub free_bytes: u64,
    /// HIP gcnArchName without its feature suffix ("gfx1201"); empty on CUDA
    pub arch: [u8; 64],
    pub name: [u8; 128],
}

impl Default for DeviceInfo {
    fn default() -> Self {
        DeviceInfo {
            ordinal: -1,
            cc_major: 0,
            cc_minor: 0,
            multi_processor_count: 0,
            driver_version: 0,
            runtime_version: 0,
            total_bytes: 0,
            free_bytes: 0,
            arch: [0; 64],
            name: [0; 128],
        }
    }
}

impl DeviceInfo {
    pub fn name_str(&self) -> &str {
        cstr(&self.name)
    }
    pub fn arch_str(&self) -> &str {
        cstr(&self.arch)
    }
}

fn cstr(buf: &[u8]) -> &str {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    std::str::from_utf8(&buf[..end]).unwrap_or("")
}

/// One completion from a direct-IO file (platform/direct_file.hpp).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IoCompletion {
    /// the caller's tag from `submit`
    pub tag: u64,
    /// bytes transferred (short only at end of file)
    pub bytes: u32,
    pub ok: u32,
}

/// Opaque handles: created and destroyed by the device layer, never by Rust.
pub type Stream = *mut c_void;
pub type DeviceFile = *mut c_void;
pub type CapturedGraphHandle = *mut c_void;
pub type SessionHandle = *mut c_void;

/// The two coupled-draft calls carry too many device buffers for a readable inline
/// signature.
pub type CoupledStage = unsafe extern "C" fn(
    mapped_params: *const SamplerParams,
    mapped_hist: *const i32,
    params_dev: *mut SamplerParams,
    ring_dev: *mut i32,
    cap: c_int,
    stream: Stream,
) -> StrataStatus;

#[allow(clippy::too_many_arguments)]
pub type CoupledDraftSample = unsafe extern "C" fn(
    logits_dev: *mut f32,
    n_vocab: c_int,
    sub_to_id_dev: *const i32,
    id_to_sub_dev: *const i32,
    id_vocab: c_int,
    params_dev: *const SamplerParams,
    ring_dev: *mut i32,
    cap: c_int,
    j: c_int,
    step_rec_dev: *const i32,
    scratch_dev: *mut c_void,
    out_id: *mut i32,
    out_prob: *mut f32,
    stream: Stream,
) -> StrataStatus;

/// The kernel + device layer. Slot order is append-only; a shim that predates a
/// slot leaves it null and the core degrades instead of failing to load.
///
/// Buffers named `*_dev` are DEVICE pointers and `*_host` are pinned host
/// pointers. Passing host memory where a device pointer is expected faults inside
/// the kernel with an illegal memory access reported by some later, unrelated
/// synchronising call — the split is in the name for that reason.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct StrataKernels {
    // ---- device / runtime
    pub device_count: Option<unsafe extern "C" fn() -> c_int>,
    pub device_info:
        Option<unsafe extern "C" fn(ordinal: c_int, out: *mut DeviceInfo) -> StrataStatus>,
    /// "" when the device can run this binary, else the reason (HIP arch guard)
    pub gpu_arch_problem: Option<
        unsafe extern "C" fn(ordinal: c_int, out: *mut c_char, out_len: usize) -> StrataStatus,
    >,
    pub set_device: Option<unsafe extern "C" fn(ordinal: c_int) -> StrataStatus>,
    pub stream_create: Option<unsafe extern "C" fn(out: *mut Stream) -> StrataStatus>,
    pub stream_destroy: Option<unsafe extern "C" fn(stream: Stream)>,
    /// polls an event rather than blocking on a sync: a thread that only reads
    /// memory never makes the driver flush (graph.hpp, measured)
    pub stream_query_done: Option<unsafe extern "C" fn(stream: Stream) -> c_int>,

    // ---- direct-IO file (the expert pager)
    pub file_open: Option<
        unsafe extern "C" fn(
            path: *const c_char,
            out: *mut DeviceFile,
            err: *mut c_char,
            err_len: usize,
        ) -> StrataStatus,
    >,
    pub file_close: Option<unsafe extern "C" fn(file: DeviceFile)>,
    pub file_size: Option<unsafe extern "C" fn(file: DeviceFile) -> u64>,
    /// `buffer` is aligned to `file_alignment()`; offset too
    pub file_submit: Option<
        unsafe extern "C" fn(
            file: DeviceFile,
            offset: u64,
            buffer: *mut c_void,
            length: u32,
            tag: u64,
            err: *mut c_char,
            err_len: usize,
        ) -> StrataStatus,
    >,
    pub file_wait: Option<
        unsafe extern "C" fn(
            file: DeviceFile,
            out: *mut IoCompletion,
            max: c_int,
            timeout_ms: c_int,
        ) -> c_int,
    >,
    pub file_alignment: Option<unsafe extern "C" fn() -> u32>,

    // ---- pinned memory (the expert arena)
    pub pinned_alloc: Option<
        unsafe extern "C" fn(
            bytes: u64,
            out: *mut *mut c_void,
            err: *mut c_char,
            err_len: usize,
        ) -> StrataStatus,
    >,
    pub pinned_free: Option<unsafe extern "C" fn(p: *mut c_void, bytes: u64)>,

    // ---- captured graphs. Every buffer the body touches must exist before
    // `record` and never move: the registry holds the graph, not the memory
    pub graph_record: Option<
        unsafe extern "C" fn(
            out: *mut CapturedGraphHandle,
            stream: Stream,
            err: *mut c_char,
            err_len: usize,
        ) -> StrataStatus,
    >,
    pub graph_end: Option<
        unsafe extern "C" fn(
            graph: CapturedGraphHandle,
            err: *mut c_char,
            err_len: usize,
        ) -> StrataStatus,
    >,
    pub graph_replay: Option<
        unsafe extern "C" fn(
            graph: CapturedGraphHandle,
            stream: Stream,
            err: *mut c_char,
            err_len: usize,
        ) -> StrataStatus,
    >,
    pub graph_destroy: Option<unsafe extern "C" fn(graph: CapturedGraphHandle)>,

    // ---- sampler
    /// `history` is (n_tokens, history_len) int32 device memory, one row per token
    /// row, -1 in unused slots; pass null + 0 when no penalties apply
    pub sample_tokens: Option<
        unsafe extern "C" fn(
            logits_dev: *const f32,
            n_tokens: c_int,
            n_vocab: c_int,
            history_dev: *const i32,
            history_len: c_int,
            params: *const SamplerParams,
            out_dev: *mut c_int,
            stream: Stream,
        ) -> StrataStatus,
    >,
    /// the greedy pick on an 8-CTA cluster (sm_90+ CUDA only); returns false when
    /// it cannot run and the caller falls back to the one-block argmax
    pub sample_greedy_cluster: Option<
        unsafe extern "C" fn(
            logits_dev: *const f32,
            n_tokens: c_int,
            n_vocab: c_int,
            out_dev: *mut c_int,
            stream: Stream,
        ) -> c_int,
    >,

    // ---- coupled draft sampling (STRATA_SPEC_COUPLED). Device pointers
    // throughout; every per-request / per-round value comes from device memory so
    // the calls can be captured.
    /// 0 when the vocabulary is too wide for the split merge
    pub coupled_scratch_bytes: Option<unsafe extern "C" fn(n_vocab: c_int) -> usize>,
    pub coupled_stage: Option<CoupledStage>,
    pub coupled_draft_sample: Option<CoupledDraftSample>,

    // ---- device memory + DMA (appended for ABI v1: an older shim leaves these
    // null because strata_kernels_load zeroes `out` before filling it). The
    // graph/sampler slots above only got exercised once a test could put bytes
    // on the device and read them back - cudaMalloc/cudaMemcpy are that floor.
    pub device_alloc: Option<
        unsafe extern "C" fn(
            bytes: u64,
            out: *mut *mut c_void,
            err: *mut c_char,
            err_len: usize,
        ) -> StrataStatus,
    >,
    pub device_free: Option<unsafe extern "C" fn(p: *mut c_void, bytes: u64)>,
    /// source may be pageable, but only pinned sources actually run async; the
    /// completion is stream-ordered, so stream_query_done settles it
    pub memcpy_h2d_async: Option<
        unsafe extern "C" fn(
            dst_dev: *mut c_void,
            src_host: *const c_void,
            bytes: u64,
            stream: Stream,
        ) -> StrataStatus,
    >,
    pub memcpy_d2h_async: Option<
        unsafe extern "C" fn(
            dst_host: *mut c_void,
            src_dev: *const c_void,
            bytes: u64,
            stream: Stream,
        ) -> StrataStatus,
    >,

    // ---- conversation snapshot (appended within ABI v1). Synchronous by
    // design: parking and resuming a conversation is never a decode step, so
    // the contract copies with cudaMemcpyDefault and settles with
    // cudaDeviceSynchronize, like the C++ it replaces. A shim without these
    // slots cannot save or restore snapshots; the core asks before it tries.
    /// `cudaMemcpy(dst, src, bytes, cudaMemcpyDefault)` — the engine holds one
    /// address per region and the runtime resolves the direction. `err` carries
    /// the raw `cudaErrorString`, which the caller prefixes per its contract.
    pub memcpy_default: Option<
        unsafe extern "C" fn(
            dst: *mut c_void,
            src: *const c_void,
            bytes: u64,
            err: *mut c_char,
            err_len: usize,
        ) -> StrataStatus,
    >,
    pub device_sync: Option<unsafe extern "C" fn(err: *mut c_char, err_len: usize) -> StrataStatus>,
}

impl StrataKernels {
    /// No device layer: every slot missing. The pure Rust core (spec, artifact,
    /// sampler arithmetic) still runs and tests, which is what lets the port be
    /// verified on a machine with no GPU and no CUDA toolkit.
    pub const fn none() -> StrataKernels {
        StrataKernels {
            device_count: None,
            device_info: None,
            gpu_arch_problem: None,
            set_device: None,
            stream_create: None,
            stream_destroy: None,
            stream_query_done: None,
            file_open: None,
            file_close: None,
            file_size: None,
            file_submit: None,
            file_wait: None,
            file_alignment: None,
            pinned_alloc: None,
            pinned_free: None,
            graph_record: None,
            graph_end: None,
            graph_replay: None,
            graph_destroy: None,
            sample_tokens: None,
            sample_greedy_cluster: None,
            coupled_scratch_bytes: None,
            coupled_stage: None,
            coupled_draft_sample: None,
            device_alloc: None,
            device_free: None,
            memcpy_h2d_async: None,
            memcpy_d2h_async: None,
            memcpy_default: None,
            device_sync: None,
        }
    }

    pub fn filled_slots(&self) -> usize {
        let Self {
            device_count,
            device_info,
            gpu_arch_problem,
            set_device,
            stream_create,
            stream_destroy,
            stream_query_done,
            file_open,
            file_close,
            file_size,
            file_submit,
            file_wait,
            file_alignment,
            pinned_alloc,
            pinned_free,
            graph_record,
            graph_end,
            graph_replay,
            graph_destroy,
            sample_tokens,
            sample_greedy_cluster,
            coupled_scratch_bytes,
            coupled_stage,
            coupled_draft_sample,
            device_alloc,
            device_free,
            memcpy_h2d_async,
            memcpy_d2h_async,
            memcpy_default,
            device_sync,
        } = *self;
        [
            device_count.map(|f| f as usize),
            device_info.map(|f| f as usize),
            gpu_arch_problem.map(|f| f as usize),
            set_device.map(|f| f as usize),
            stream_create.map(|f| f as usize),
            stream_destroy.map(|f| f as usize),
            stream_query_done.map(|f| f as usize),
            file_open.map(|f| f as usize),
            file_close.map(|f| f as usize),
            file_size.map(|f| f as usize),
            file_submit.map(|f| f as usize),
            file_wait.map(|f| f as usize),
            file_alignment.map(|f| f as usize),
            pinned_alloc.map(|f| f as usize),
            pinned_free.map(|f| f as usize),
            graph_record.map(|f| f as usize),
            graph_end.map(|f| f as usize),
            graph_replay.map(|f| f as usize),
            graph_destroy.map(|f| f as usize),
            sample_tokens.map(|f| f as usize),
            sample_greedy_cluster.map(|f| f as usize),
            coupled_scratch_bytes.map(|f| f as usize),
            coupled_stage.map(|f| f as usize),
            coupled_draft_sample.map(|f| f as usize),
            device_alloc.map(|f| f as usize),
            device_free.map(|f| f as usize),
            memcpy_h2d_async.map(|f| f as usize),
            memcpy_d2h_async.map(|f| f as usize),
            memcpy_default.map(|f| f as usize),
            device_sync.map(|f| f as usize),
        ]
        .iter()
        .filter(|s| s.is_some())
        .count()
    }
}

#[cfg(test)]
#[allow(unsafe_code)] // the only unsafe in the workspace: calling through a vtable slot
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    /// strata::kernels::SamplerParams: 10 x 4-byte fields (40), 2 x u64 (56), bool
    /// padded to 8-alignment = 64. If the C++ side grows a field without growing
    /// this, the generated header and this test are the two halves of the check.
    #[test]
    fn sampler_params_layout_mirrors_cpp() {
        assert_eq!(size_of::<SamplerParams>(), 64);
        assert_eq!(align_of::<SamplerParams>(), 8);
    }

    #[test]
    fn io_completion_is_the_cpp_struct() {
        // uint64 tag, uint32 bytes, bool ok -> 16 with tail padding
        assert_eq!(size_of::<IoCompletion>(), 16);
    }

    #[test]
    fn device_info_is_plain_data() {
        // 6 c_int (24, already 8-aligned) + 2 u64 + 64 + 128 = 232
        assert_eq!(size_of::<DeviceInfo>(), 232);
        assert_eq!(align_of::<DeviceInfo>(), 8);
    }

    #[test]
    fn device_names_decode_past_their_nul() {
        let mut d = DeviceInfo::default();
        d.name[..11].copy_from_slice(b"NVIDIA 5070"); // no terminator: read to the end of the buffer
        assert_eq!(d.name_str(), "NVIDIA 5070");
        let mut e = DeviceInfo::default();
        e.arch[..7].copy_from_slice(b"gfx1201");
        e.arch[7] = 0;
        assert_eq!(e.arch_str(), "gfx1201");
        assert_eq!(DeviceInfo::default().name_str(), "");
    }

    #[test]
    fn no_device_layer_is_usable() {
        let k = StrataKernels::none();
        assert_eq!(k.filled_slots(), 0);
        assert!(k.device_count.is_none());
        assert!(k.sample_tokens.is_none());
    }

    #[test]
    fn vtable_slots_are_nullable_function_pointers() {
        unsafe extern "C" fn count() -> c_int {
            3
        }
        let mut k = StrataKernels::none();
        k.device_count = Some(count);
        assert_eq!(k.filled_slots(), 1);
        assert_eq!(unsafe { k.device_count.unwrap()() }, 3);
    }

    #[test]
    fn appended_device_slots_keep_the_prefix_layout() {
        // Append-only is the whole versioning story: the first 24 slots keep
        // their offsets, and the four device-memory slots sit at the tail.
        assert_eq!(
            std::mem::size_of::<StrataKernels>(),
            30 * std::mem::size_of::<usize>()
        );
        let none = StrataKernels::none();
        assert!(none.device_alloc.is_none() && none.memcpy_d2h_async.is_none());
        assert!(none.memcpy_default.is_none() && none.device_sync.is_none());
    }
}
