//! The conversation-snapshot seam over a REAL CUDA shim: `cudaMalloc` buffers,
//! `cudaMemcpy(..., cudaMemcpyDefault)` transfers, `cudaDeviceSynchronize`.
//! Same fixture and same golden numbers as the CPU rehearsal
//! (`snapshot_device.rs`), so a difference between the two is a difference in
//! the driver path, not the contract.
//!
//! With no shim configured, or a CPU-only shim build, every test prints a skip
//! and passes, so `cargo test` stays green everywhere. On a CUDA build:
//!
//!   STRATA_SHIM_CUDA=1 CUDA_HOME=/usr/local/cuda ./shim/build.sh
//!   STRATA_KERNELS_LIB=$PWD/target/shim/libstrata_kernels.so \
//!     cargo test -p strata-device --test gpu_snapshot -- --nocapture

use strata_core::conv_vec::{CppVec, SIZEOF_CHECKPOINT};
use strata_core::conversation::{
    geometry_key, ConversationCheckpoint, ConversationImageKey, ConversationKvReuse,
    ConversationRestore, ConversationView, SavedConversation, SessionState,
};
use strata_core::conversation_kv::{self as kv, Device};
use strata_core::conversation_state as cs;
use strata_core::fnv;
use strata_core::layout::ModelGeometry;
use strata_device::snapshot::{
    DeviceRegion, IndexerBuffers, LayerKv, LayerState, RunningTarget, SessionDevice, SnapshotDevice,
};
use strata_device::{DeviceBuf, Shim};

const KV_BYTES: usize = 12 * 64 * 2; // cells * head_dim * 2 (fp16), 1 KV head
const IDX_DIM: usize = 8;

fn shim() -> Option<Shim> {
    Shim::try_load().map(|r| r.expect("STRATA_KERNELS_LIB set but failed to load"))
}

/// Skip when this shim build has no CUDA, or sees no device.
macro_rules! need_cuda {
    ($shim:expr) => {
        if $shim.kernels().memcpy_default.is_none()
            || $shim.kernels().device_sync.is_none()
            || $shim.device_count() == 0
        {
            eprintln!(
                "skipped: shim has {} of {} slots, {} device(s)",
                $shim.filled_slots(),
                strata_device::STRATA_SLOT_COUNT,
                $shim.device_count()
            );
            return;
        }
    };
}

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
/// checkpoint at five (golden CARVE sizes: tail=96, dead=32, block_pos=4).
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
    let mut cp = ConversationCheckpoint::default();
    cp.ids.resize(5, &0);
    for (i, slot) in cp.ids.as_mut_slice().iter_mut().enumerate() {
        *slot = i as i32 + 1;
    }
    cp.imgs.push(ConversationImageKey { start: 1, hash: 44 });
    cp.gdn.resize(768, &13);
    cp.tails.resize(2 * 96, &13);
    cp.dead.resize(2 * 32, &13);
    cp.block_pos.resize(2 * 4, &13);
    let mut cps = CppVec::with_unit(SIZEOF_CHECKPOINT);
    cps.push(cp);
    (ids, imgs, cps)
}

/// Every device buffer of the fixture, allocated by the shim's `cudaMalloc`,
/// filled through the seam itself (a write through `memcpy_default` is the same
/// DMA the save/restore uses — no side channel).
struct Gpu {
    kv: Vec<DeviceBuf<'static>>,
    idx: Vec<DeviceBuf<'static>>,
    gdn: DeviceBuf<'static>,
}

fn region(b: &DeviceBuf) -> Option<DeviceRegion> {
    DeviceRegion::new(b.dev_ptr(), b.len())
}

/// Leak the shim for the fixture's lifetime in test scope only; the buffers
/// outlive the session exactly like the engine's arena does.
fn gpu(shim: &'static Shim) -> Gpu {
    let k = shim.kernels();
    let mut kv = Vec::new();
    let mut idx = Vec::new();
    // three K/V layers (two QSA + the drafter), each k/v and, for the two QSA
    // layers, the pooled array; then each layer's tail/dead/block_pos
    for layer in 0..3 {
        kv.push(DeviceBuf::alloc(k, KV_BYTES).unwrap());
        kv.push(DeviceBuf::alloc(k, KV_BYTES).unwrap());
        if layer < 2 {
            kv.push(DeviceBuf::alloc(k, 26 * IDX_DIM * 4).unwrap());
        }
    }
    for _ in 0..2 {
        idx.push(DeviceBuf::alloc(k, 3 * IDX_DIM * 4).unwrap());
        idx.push(DeviceBuf::alloc(k, IDX_DIM * 4).unwrap());
        idx.push(DeviceBuf::alloc(k, 4).unwrap());
    }
    let gdn = DeviceBuf::alloc(k, 768).unwrap();
    Gpu { kv, idx, gdn }
}

fn layer_kv(g: &Gpu, base: usize, index: bool) -> LayerKv {
    let mut regions: [Option<DeviceRegion>; 5] = Default::default();
    regions[0] = region(&g.kv[base]);
    regions[1] = region(&g.kv[base + 1]);
    if index {
        regions[4] = region(&g.kv[base + 2]);
    }
    LayerKv::regions(regions)
}

fn indexer(g: &Gpu, which: usize, index: bool) -> IndexerBuffers {
    IndexerBuffers {
        tail: if index {
            region(&g.idx[which * 3])
        } else {
            None
        },
        dead: if index {
            region(&g.idx[which * 3 + 1])
        } else {
            None
        },
        block_pos: if index {
            region(&g.idx[which * 3 + 2])
        } else {
            None
        },
        pooled: if index {
            region(&g.kv[which * 3 + 2])
        } else {
            None
        },
    }
}

/// Fill every buffer with `pattern` through the seam's own write paths.
fn fill(device: &mut SnapshotDevice, pattern: u8) {
    let payload = vec![pattern; KV_BYTES];
    for base in (0..9).step_by(3).take(3) {
        let set = if base / 3 == 2 {
            // the drafter's regions are addressed the same way
            kv::Layer::Draft
        } else {
            kv::Layer::Qsa { ordinal: base / 3 }
        };
        device
            .write(set, kv::PoolSet::Slots, kv::Region::K, 0, &payload)
            .unwrap();
        device
            .write(set, kv::PoolSet::Slots, kv::Region::V, 0, &payload)
            .unwrap();
        if base / 3 < 2 {
            device
                .write(
                    set,
                    kv::PoolSet::Slots,
                    kv::Region::Pooled,
                    0,
                    &vec![pattern; 26 * IDX_DIM * 4],
                )
                .unwrap();
        }
    }
    device
        .write_running(kv::Running::Gdn, 0, &vec![pattern; 768])
        .unwrap();
    for ordinal in 0..2 {
        device
            .write_running(
                kv::Running::Indexer {
                    ordinal,
                    part: kv::RunningPart::Tail,
                },
                0,
                &[pattern; 3 * IDX_DIM * 4],
            )
            .unwrap();
        device
            .write_running(
                kv::Running::Indexer {
                    ordinal,
                    part: kv::RunningPart::Dead,
                },
                0,
                &[pattern; IDX_DIM * 4],
            )
            .unwrap();
        device
            .write_running(
                kv::Running::Indexer {
                    ordinal,
                    part: kv::RunningPart::BlockPos,
                },
                0,
                &[pattern; 4],
            )
            .unwrap();
    }
    device.sync().unwrap();
}

fn save_image(device: &mut SnapshotDevice) -> SavedConversation {
    let g = geometry();
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
    assert!(
        cs::snapshot_save(&mut image, &view, &ss, &g, &draft, device, &mut err, &mut reuse, None),
        "save over the real driver: {err}"
    );
    image
}

#[test]
fn a_save_over_the_real_driver_reads_the_golden_bytes() {
    let Some(shim) = shim() else { return };
    need_cuda!(shim);
    // SAFETY: leaked for test scope so DeviceBuf<'static> borrows stay valid;
    // the process exits after the test suite.
    let shim = Box::leak(Box::new(shim));
    let g = geometry();
    let gpub = gpu(shim);
    let session = SessionDevice {
        draft: LayerState::resident(layer_kv(&gpub, 6, false), layer_kv(&gpub, 6, false)),
        qsa: vec![
            LayerState::resident(layer_kv(&gpub, 0, true), layer_kv(&gpub, 0, true)),
            LayerState::resident(layer_kv(&gpub, 3, true), layer_kv(&gpub, 3, true)),
        ],
        running: RunningTarget {
            gdn: region(&gpub.gdn),
            ple: None,
            indexer: vec![indexer(&gpub, 0, true), indexer(&gpub, 1, true)],
        },
    };
    let mut device = SnapshotDevice::new(shim.kernels(), session).unwrap();
    fill(&mut device, 0xa5);

    let image = save_image(&mut device);
    // golden SAVE_FULL: bytes=12600, k/v=6404351448661373315, pooled QSA=9014609365494796707
    assert_eq!(image.bytes(), 12600, "image bytes");
    for (layer, item) in image.kv.iter().enumerate() {
        let mut v = vec![0u8; item.k.size()];
        let n = v.len();
        assert!(item.k.read(&mut v, 0, n));
        assert_eq!(
            fnv(&v),
            6404351448661373315,
            "layer {layer} K came over real DMA"
        );
        assert!(v.iter().all(|&b| b == 0xa5), "layer {layer} K bytes");
    }
    assert_eq!(
        fnv(image.live.gdn.as_slice()),
        7415105493412125315,
        "gdn over real DMA"
    );

    // overwrite every buffer with a different pattern, restore the image, and
    // the ORIGINAL bytes must come back through real DMA in both directions
    fill(&mut device, 0x33);
    let mut image = image;
    image.geometry = geometry_key(&g);
    image.layer_hi = g.n_layers;
    let mut ss = session_state(&g);
    let mut err = String::new();
    let r = cs::snapshot_restore(&image, &mut ss, &g, &state(0), &mut device, &mut err);
    assert_eq!(
        r,
        ConversationRestore::Restored,
        "restore over the real driver: {err}"
    );
    let mut k0 = vec![0u8; KV_BYTES];
    device
        .read(
            kv::Layer::Qsa { ordinal: 0 },
            kv::PoolSet::Slots,
            kv::Region::K,
            0,
            &mut k0,
        )
        .unwrap();
    device.sync().unwrap();
    assert!(
        k0.iter().all(|&b| b == 0xa5),
        "the restore's bytes survived in VRAM"
    );
    let mut gdnb = vec![0u8; 768];
    device.read_running(kv::Running::Gdn, 0, &mut gdnb).unwrap();
    device.sync().unwrap();
    assert!(
        gdnb.iter().all(|&b| b == 0xa5),
        "the gdn row round-tripped through VRAM"
    );
}

#[test]
fn the_cpu_only_paths_stay_honest() {
    // can_serve is the gate the engine will use to pick this seam over any
    // fallback; a CPU-only shim must say no
    let Some(shim) = shim() else { return };
    let serves = SnapshotDevice::can_serve(shim.kernels());
    let cuda = shim.kernels().memcpy_default.is_some() && shim.kernels().device_sync.is_some();
    assert_eq!(serves, cuda);
    eprintln!(
        "shim {} of {} slots, snapshot slots {}",
        shim.filled_slots(),
        strata_device::STRATA_SLOT_COUNT,
        if serves { "present" } else { "absent" }
    );
}
