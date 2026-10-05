//! `SnapshotDevice` on a fake vtable: real addresses, real `memcpy_default`
//! calls, host memory behind them. The corpus replay injects a scripted backend
//! and never sees an address; this test injects the *vtable*, so every branch of
//! `snapshot.rs` — address arithmetic, the range checks, the error-buffer path,
//! the residency refusal — runs before the code ever sees a GPU.
//!
//! The expectations are golden C++ numbers for fixture 0 (fp16 K/V, two QSA
//! layers plus the drafter's ring): `SAVE_FULL copies=10440`, and the region
//! fingerprints `SAVE_FULL_KV`/`SAVE_FULL_STATE` print. A seam that selected the
//! wrong buffer or offset would move different bytes and stop matching them.

#![allow(unsafe_code)] // fake vtable slots: the same kind of unsafe as shim.rs

use std::cell::Cell;
use std::os::raw::{c_char, c_void};

use strata_core::conv_buffer::ConversationBuffer;
use strata_core::conv_vec::{CppVec, SIZEOF_CHECKPOINT};
use strata_core::conversation::{
    geometry_key, ConversationCheckpoint, ConversationImageKey, ConversationKvReuse,
    ConversationRestore, ConversationView, SavedConversation, SessionState,
};
use strata_core::conversation_kv::{self as kv, Device, PoolSet, Region, Running, RunningPart};
use strata_core::conversation_state as cs;
use strata_core::fnv;
use strata_core::layout::ModelGeometry;
use strata_device::abi::{StrataKernels, StrataStatus};
use strata_device::snapshot::{
    DeviceRegion, IndexerBuffers, LayerKv, LayerState, Residency, RunningTarget, SessionDevice,
    SnapshotDevice,
};

thread_local! {
    static COPIES: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
    static SYNCED: Cell<usize> = const { Cell::new(0) };
    static FAIL_AT: Cell<usize> = const { Cell::new(usize::MAX) };
    static LOG: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}
fn log_clear() {
    LOG.with(|l| l.borrow_mut().clear());
}
fn log() -> Vec<u64> {
    LOG.with(|l| l.borrow().clone())
}
fn copies() -> usize {
    COPIES.with(|c| c.get())
}
fn bytes() -> u64 {
    BYTES.with(|c| c.get())
}
fn synced() -> usize {
    SYNCED.with(|c| c.get())
}

extern "C" fn fake_alloc(
    bytes: u64,
    out: *mut *mut c_void,
    err: *mut c_char,
    err_len: usize,
) -> StrataStatus {
    if bytes == 0 || out.is_null() {
        set_err(err, err_len, "fake alloc: zero bytes");
        return StrataStatus::Error;
    }
    let buf = vec![0u8; bytes as usize].into_boxed_slice();
    let p = Box::into_raw(buf) as *mut c_void;
    unsafe { *out = p };
    StrataStatus::Ok
}

extern "C" fn fake_free(p: *mut c_void, bytes: u64) {
    if p.is_null() {
        return;
    }
    // SAFETY: p came from fake_alloc with exactly this size.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            p as *mut u8,
            bytes as usize,
        )))
    };
}

extern "C" fn fake_memcpy(
    dst: *mut c_void,
    src: *const c_void,
    bytes: u64,
    err: *mut c_char,
    err_len: usize,
) -> StrataStatus {
    let n = COPIES.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if n == FAIL_AT.with(|c| c.get()) {
        set_err(err, err_len, "fake cudaMemcpy");
        return StrataStatus::Error;
    }
    BYTES.with(|c| c.set(c.get() + bytes));
    LOG.with(|l| l.borrow_mut().push(bytes));
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes as usize) };
    StrataStatus::Ok
}

extern "C" fn fake_sync(_err: *mut c_char, _err_len: usize) -> StrataStatus {
    SYNCED.with(|c| c.set(c.get() + 1));
    StrataStatus::Ok
}

fn set_err(err: *mut c_char, err_len: usize, msg: &str) {
    if err.is_null() || err_len == 0 {
        return;
    }
    let n = std::cmp::min(msg.len(), err_len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(msg.as_ptr() as *const c_char, err, n);
        *err.add(n) = 0;
    }
}

fn kernels() -> StrataKernels {
    let mut k = StrataKernels::none();
    k.device_alloc = Some(fake_alloc);
    k.device_free = Some(fake_free);
    k.memcpy_default = Some(fake_memcpy);
    k.device_sync = Some(fake_sync);
    k
}

/// The fixture-0 geometry: the corpus harness's shapes (8 layers, 256 experts,
/// 1 KV head, head_dim 64, idx_dim 8), so the golden numbers apply.
fn geometry() -> ModelGeometry {
    ModelGeometry {
        n_layers: 8,
        n_expert: 256,
        ssm_state_size: 2,
        ssm_v_heads: 2,
        ssm_conv_channels: 8,
        n_head_kv: 1,
        head_dim: 64,
        idx_key_dim: 8,
        ..ModelGeometry::default()
    }
}

const KV_BYTES: usize = 12 * 64 * 2; // cells * head_dim * 2 (fp16), 1 KV head
const IDX_DIM: usize = 8;

/// Device buffers allocated through the vtable — addresses chosen by the fake
/// allocator, exactly the condition the C++ engine lived in. The last
/// allocation is a sentinel region nothing references, so any out-of-range write
/// lands outside a buffer the seam is allowed to touch.
struct Pools {
    l0: LayerKv,
    l1: LayerKv,
    draft: LayerKv,
    idx: [IndexerBuffers; 2],
    gdn: DeviceRegion,
    keep: Vec<(usize, usize)>,
}

impl Pools {
    fn new(k: &StrataKernels) -> Pools {
        let mut p = Pools {
            l0: LayerKv::default(),
            l1: LayerKv::default(),
            draft: LayerKv::default(),
            idx: [IndexerBuffers::default(), IndexerBuffers::default()],
            gdn: DeviceRegion::new(std::ptr::dangling::<c_void>(), 0).unwrap(),
            keep: Vec::new(),
        };
        let (l0, i0) = p.layer(k, true);
        let (l1, i1) = p.layer(k, true);
        let (dr, _) = p.layer(k, false);
        p.gdn = p.take(k, 768);
        p.take(k, 4096); // the sentinel
        p.l0 = l0;
        p.l1 = l1;
        p.draft = dr;
        p.idx = [i0, i1];
        p
    }

    fn take(&mut self, k: &StrataKernels, bytes: usize) -> DeviceRegion {
        let mut p = std::ptr::null_mut();
        let mut err = [0 as c_char; 128];
        let f = k.device_alloc.unwrap();
        let st = unsafe { f(bytes as u64, &mut p, err.as_mut_ptr(), err.len()) };
        assert!(st.is_ok(), "alloc {bytes}");
        unsafe { std::ptr::write_bytes(p as *mut u8, 0xa5, bytes) };
        self.keep.push((p as usize, bytes));
        DeviceRegion::new(p, bytes).unwrap()
    }

    /// One layer: k/v pools, the indexer's spare-row array when indexed, and the
    /// three per-layer indexer arrays. fp16, so no scale pools — null, like the
    /// corpus fixture's, which is how a missing buffer is meant to look.
    fn layer(&mut self, k: &StrataKernels, index: bool) -> (LayerKv, IndexerBuffers) {
        let k_pool = self.take(k, KV_BYTES);
        let v_pool = self.take(k, KV_BYTES);
        let pooled = index.then(|| self.take(k, 26 * IDX_DIM * 4));
        let idx = IndexerBuffers {
            tail: Some(self.take(k, 3 * IDX_DIM * 4)),
            dead: Some(self.take(k, IDX_DIM * 4)),
            block_pos: Some(self.take(k, 4)),
            pooled,
        };
        let mut regions: [Option<DeviceRegion>; 5] = Default::default();
        regions[0] = Some(k_pool);
        regions[1] = Some(v_pool);
        regions[4] = pooled;
        (LayerKv::regions(regions), idx)
    }
}

impl Drop for Pools {
    fn drop(&mut self) {
        for &(p, bytes) in &self.keep {
            fake_free(p as *mut c_void, bytes as u64);
        }
    }
}

fn state(pooled_rows: i64) -> kv::QsaState {
    kv::QsaState {
        mode: 0,
        max_cells: 96,
        n_pages: 24,
        n_slots: 24,
        idx_pooled_rows: pooled_rows,
        idx_pooled: pooled_rows != 0,
        idx_tail: true,
        idx_dead: true,
        idx_block_pos: true,
        slots: kv::Pools {
            k_pool: true,
            v_pool: true,
            ..kv::Pools::default()
        },
        ..kv::QsaState::default()
    }
}

fn session_state(g: &ModelGeometry) -> SessionState {
    SessionState {
        max_cells: 96,
        gdn_state: true,
        layer_hi: g.n_layers,
        gdn_alloc: g.n_gdn_layers(),
        qsa_alloc: g.n_qsa_layers(),
        qsa_states: vec![state(26), state(26), state(0)],
        ..Default::default()
    }
}

/// The fixture's live prefix: nine tokens, one image at token 1, one retained
/// checkpoint at five tokens — the same view the corpus harness saves through.
fn fixture_view() -> (
    CppVec<i32>,
    CppVec<ConversationImageKey>,
    CppVec<ConversationCheckpoint>,
) {
    let mut ids = CppVec::with_unit(4);
    ids.resize(9, &0);
    for (i, slot) in ids.as_mut_slice().iter_mut().enumerate() {
        *slot = i as i32 + 1;
    }
    let mut imgs = CppVec::with_unit(16);
    imgs.push(ConversationImageKey { start: 1, hash: 44 });
    let mut cps = CppVec::with_unit(SIZEOF_CHECKPOINT);
    cps.push(checkpoint(5));
    (ids, imgs, cps)
}

/// A checkpoint with the whole-session running-state payload: the golden's SIZES
/// line (gdn=768, tail=96, dead=32, block_pos=4), fill 13.
fn checkpoint(tokens: usize) -> ConversationCheckpoint {
    let mut c = ConversationCheckpoint::default();
    c.ids.resize(tokens, &0);
    for (i, slot) in c.ids.as_mut_slice().iter_mut().enumerate() {
        *slot = i as i32 + 1;
    }
    c.imgs.push(ConversationImageKey { start: 1, hash: 44 });
    c.gdn.resize(768, &13);
    c.tails.resize(2 * 96, &13);
    c.dead.resize(2 * 32, &13);
    c.block_pos.resize(2 * 4, &13);
    c
}

fn reset_counters() {
    COPIES.with(|c| c.set(0));
    BYTES.with(|c| c.set(0));
    SYNCED.with(|c| c.set(0));
    FAIL_AT.with(|c| c.set(usize::MAX));
    log_clear();
}

/// The device bytes of a buffer, through the contract's own reader.
fn buffer_bytes(b: &ConversationBuffer) -> Vec<u8> {
    let mut v = vec![0u8; b.size()];
    let n = v.len();
    assert!(b.read(&mut v, 0, n));
    v
}

#[test]
fn a_save_through_the_seam_moves_exactly_the_bytes_the_cxx_moved() {
    let k = kernels();
    reset_counters();
    let g = geometry();
    let pools = Pools::new(&k);
    let device = SnapshotDevice::new(
        &k,
        SessionDevice {
            draft: LayerState::resident(pools.draft, pools.draft),
            qsa: vec![
                LayerState::resident(pools.l0, pools.l0),
                LayerState::resident(pools.l1, pools.l1),
            ],
            running: RunningTarget {
                gdn: Some(pools.gdn),
                ple: None,
                indexer: pools.idx.to_vec(),
            },
        },
    )
    .unwrap();
    let mut device = device;
    let ss = session_state(&g);
    let draft = state(0);
    let (ids, imgs, cps) = fixture_view();
    let view = ConversationView {
        ids: &ids,
        images: &imgs,
        checkpoints: &cps,
        cvec: true,
    };
    let mut image = SavedConversation::default();
    let mut reuse = ConversationKvReuse::default();
    let mut err = String::new();
    let ok = cs::snapshot_save(
        &mut image,
        &view,
        &ss,
        &g,
        &draft,
        &mut device,
        &mut err,
        &mut reuse,
        None,
    );
    assert!(ok, "save through the seam: {err}");

    // golden: SAVE_FULL|copies=10440|bytes=12600
    assert_eq!(bytes(), 10440, "bytes through cudaMemcpyDefault");
    assert_eq!(image.bytes(), 12600, "image bytes (the admission count)");

    // golden SAVE_FULL_KV: k=v=6404351448661373315 on every layer,
    // pooled=9014609365494796707 on the two indexed layers, scale regions empty
    // (fp16). The empty-region fingerprint is the FNV offset basis.
    let golden_k = 6404351448661373315u64;
    let golden_pooled = 9014609365494796707u64;
    for (layer, item) in image.kv.iter().enumerate() {
        assert_eq!(fnv(&buffer_bytes(&item.k)), golden_k, "layer {layer} K");
        assert_eq!(fnv(&buffer_bytes(&item.v)), golden_k, "layer {layer} V");
        assert!(
            buffer_bytes(&item.k_scale).is_empty(),
            "fp16 has no K scales"
        );
        let want = if layer < 2 {
            golden_pooled
        } else {
            strata_core::FNV_OFFSET
        };
        assert_eq!(
            fnv(&buffer_bytes(&item.pooled)),
            want,
            "layer {layer} pooled"
        );
    }
    // the save's copy order, as the golden's failure lines show it starts with:
    // the gdn row first (the running-state helper runs before the K/V one)
    assert_eq!(
        log().first(),
        Some(&768u64),
        "the save starts at the gdn row"
    );

    // golden SAVE_FULL_STATE: gdn=7415105493412125315, ple empty (no ple buffer)
    assert_eq!(
        fnv(image.live.gdn.as_slice()),
        7415105493412125315,
        "gdn payload"
    );
    assert!(image.live.ple.as_slice().is_empty());

    // a restore through the same seam moves the same bytes back
    image.geometry = geometry_key(&g);
    image.layer_hi = g.n_layers;
    BYTES.with(|c| c.set(0));
    log_clear();
    let mut ss2 = ss.clone();
    let r = cs::snapshot_restore(&image, &mut ss2, &g, &draft, &mut device, &mut err);
    assert_eq!(r, ConversationRestore::Restored, "restore: {err}");
    // the golden's success-restore sequence (17 copies) totals 10504: the
    // 10440 the save read, plus the 2 x 32-byte moving spare rows at offset 64
    // that a restore reconstructs and a save never reads
    assert_eq!(
        bytes(),
        10504,
        "restore moves the golden's 17-copy sequence"
    );
    assert!(synced() >= 2, "the contract settles with device_sync");
    // the golden's success-restore COPY lines, in order — the seam's sequence
    // must be identical, which pins every (layer, region, offset) choice
    assert_eq!(
        log(),
        vec![
            1536, 1536, 96, // L0 k, v, idx_pooled
            1536, 1536, 96, // L1
            1536, 1536, // DRAFT k, v
            768,  // gdn
            96, 32, 4, 32, // L0 tail, dead, block_pos, pooled row @64
            96, 32, 4, 32, // L1
        ],
        "the restore's 17-copy sequence, in the golden's order"
    );
}

#[test]
fn a_failing_copy_reports_the_prefixed_error_and_stops_the_save() {
    let k = kernels();
    reset_counters();
    let g = geometry();
    let pools = Pools::new(&k);
    let mut device = SnapshotDevice::new(
        &k,
        SessionDevice {
            draft: LayerState::resident(pools.draft, pools.draft),
            qsa: vec![
                LayerState::resident(pools.l0, pools.l0),
                LayerState::resident(pools.l1, pools.l1),
            ],
            running: RunningTarget {
                gdn: Some(pools.gdn),
                ple: None,
                indexer: pools.idx.to_vec(),
            },
        },
    )
    .unwrap();
    let ss = session_state(&g);
    let draft = state(0);
    let (ids, imgs, cps) = fixture_view();
    let view = ConversationView {
        ids: &ids,
        images: &imgs,
        checkpoints: &cps,
        cvec: true,
    };
    FAIL_AT.with(|c| c.set(1)); // the first cudaMemcpyDefault fails
    let mut image = SavedConversation::default();
    let mut reuse = ConversationKvReuse::default();
    let mut err = String::new();
    let ok = cs::snapshot_save(
        &mut image,
        &view,
        &ss,
        &g,
        &draft,
        &mut device,
        &mut err,
        &mut reuse,
        None,
    );
    assert!(!ok, "the save must report the failure");
    // golden SAVE_INCREMENTAL|copy_failure_ok=0|err=conversation snapshot
    // running-state copy: stub — the save's first copy is the gdn row, so the
    // running-state helper's prefix, then the shim's raw string
    assert_eq!(
        err, "conversation snapshot running-state copy: fake cudaMemcpy",
        "{err}"
    );
    FAIL_AT.with(|c| c.set(usize::MAX));
    log_clear();
}

#[test]
fn a_state_needing_residency_is_refused_at_construction() {
    let k = kernels();
    reset_counters();
    for r in [Residency::Streamed, Residency::Ring] {
        let session = SessionDevice {
            draft: LayerState::resident(LayerKv::default(), LayerKv::default()),
            qsa: vec![LayerState {
                residency: r,
                slots: LayerKv::default(),
                host: LayerKv::default(),
            }],
            running: RunningTarget::empty(),
        };
        let e = SnapshotDevice::new(&k, session).unwrap_err();
        assert!(
            e.contains("cannot reset residency") || e.contains("restore a ring"),
            "{e}"
        );
    }
}

#[test]
fn a_vtable_without_the_slots_serves_nothing() {
    let bare = StrataKernels::none();
    assert!(!SnapshotDevice::can_serve(&bare));
    let e = SnapshotDevice::new(
        &bare,
        SessionDevice {
            draft: LayerState::resident(LayerKv::default(), LayerKv::default()),
            qsa: Vec::new(),
            running: RunningTarget::empty(),
        },
    )
    .unwrap_err();
    assert!(e.contains("memcpy_default"), "{e}");
}

#[test]
fn a_transfer_past_the_end_of_a_buffer_never_reaches_the_dma() {
    let k = kernels();
    reset_counters();
    let pools = Pools::new(&k);
    let mut device = SnapshotDevice::new(
        &k,
        SessionDevice {
            draft: LayerState::resident(pools.draft, pools.draft),
            qsa: Vec::new(),
            running: RunningTarget {
                gdn: Some(pools.gdn),
                ple: None,
                indexer: pools.idx.to_vec(),
            },
        },
    )
    .unwrap();
    // one byte too many for the K pool: refused before the vtable sees anything
    let mut too_big = vec![0u8; KV_BYTES + 8];
    let e = device
        .read(kv::Layer::Draft, PoolSet::Slots, Region::K, 0, &mut too_big)
        .unwrap_err();
    assert!(e.contains("runs past"), "{e}");
    assert_eq!(copies(), 0, "rejected before the DMA");

    // the exact-size read at offset 0 is the first copy the fake sees
    let mut exact = vec![0u8; KV_BYTES];
    device
        .read(kv::Layer::Draft, PoolSet::Slots, Region::K, 0, &mut exact)
        .unwrap();
    assert_eq!(copies(), 1);
    assert!(
        exact.iter().all(|&b| b == 0xa5),
        "the copy moved the device fill"
    );

    // the running-state path resolves its own targets, and a name the session
    // doesn't have comes back as an error the call site prefixes
    let mut gdn = vec![0u8; 768];
    device.read_running(Running::Gdn, 0, &mut gdn).unwrap();
    assert!(gdn.iter().all(|&b| b == 0xa5));
    let e = device
        .read_running(
            Running::Indexer {
                ordinal: 7,
                part: RunningPart::Tail,
            },
            0,
            &mut gdn,
        )
        .unwrap_err();
    assert!(e.contains("no buffer"), "{e}");

    // a write is the same path in reverse — and an offset read lands where the
    // offset says: the dead array (8 bytes) followed by the sentinel fill is
    // indistinguishable, so check by round-trip instead of by address
    let payload = vec![0x5au8; 3 * IDX_DIM * 4];
    device
        .write_running(
            Running::Indexer {
                ordinal: 1,
                part: RunningPart::Tail,
            },
            0,
            &payload,
        )
        .unwrap();
    let mut back = vec![0u8; 3 * IDX_DIM * 4];
    device
        .read_running(
            Running::Indexer {
                ordinal: 1,
                part: RunningPart::Tail,
            },
            0,
            &mut back,
        )
        .unwrap();
    assert_eq!(
        back, payload,
        "the layer-1 tail round-trips to its own buffer"
    );
    // and the layer-0 tail was not touched
    let mut other = vec![0u8; 3 * IDX_DIM * 4];
    device
        .read_running(
            Running::Indexer {
                ordinal: 0,
                part: RunningPart::Tail,
            },
            0,
            &mut other,
        )
        .unwrap();
    assert!(other.iter().all(|&b| b == 0xa5), "sibling buffer untouched");
}
