//! Port of `src/core/conversation_snapshot.cpp` — the K/V half of the
//! conversation snapshot contract.
//!
//! A parked conversation is the running state plus one `ConversationKv` per QSA
//! layer and the drafter's ring. This module owns the K/V half: how many bytes it
//! is, what
//! makes an image restorable, and the transfer sequence that saves, restores and
//! reads it back.
//!
//! Two properties are load-bearing, and they are why this is a module rather than
//! a loop:
//!
//! * An invalid image is rejected **before a single byte is written**. The C++
//!   test asserts it by fingerprinting every pool after every rejection; the
//!   replay does the same (`CORPUS|REJECT|…|fingerprint=…`). A restore that wrote
//!   half an image and then failed would leave a session that looks alive and
//!   answers from two conversations at once.
//! * The byte estimate **is** the admission decision. [`capture_bytes`] counts
//!   retained capacity plus the segment directory a resize must allocate next to
//!   the old one, so RAM admission sees the peak, not the payload.
//!
//! Device access goes through [`SessionRegions`], so this crate stays
//! pointer-free, and the same code path is exercised on the host by the corpus
//! replay and on the GPU by the shim-backed implementation.

use crate::conv_buffer::{add, product, ConversationBuffer};
pub use crate::conv_vec::{SIZEOF_CHECKPOINT, SIZEOF_IMAGE_KEY, SIZEOF_KV, SIZEOF_KV_REUSE};
use crate::layout::ModelGeometry;

/// One KV page is one indexer block: the granule KV streaming keeps resident.
/// `qsa_real_shapes().page_size` — the artifact's geometry, not a tuning knob.
pub const PAGE_SIZE: i64 = 4;
/// The indexer pools one key per this many cells (`qsa_real_shapes().idx_block`).
pub const IDX_BLOCK: i64 = 4;
/// Elements per `block_q4_0`.
pub const QK4_0: i64 = 32;
/// `sizeof(block_q4_0)`: an fp16 scale plus 16 bytes of packed codes.
pub const Q4_BLOCK_BYTES: u64 = 18;
/// INT8 K/V stores one fp16 scale per this many values.
pub const INT8_GROUP: i64 = 64;
/// `std::array<uint8_t, 65536>`: the read-back workspace of [`verify`].
pub const VERIFY_WINDOW: usize = 65536;

/// `KvFormat`, plus the snapshot-only variants the block movers must never see.
pub type KvFormat = i32;
pub const KV_F16: KvFormat = 0;
pub const KV_INT8: KvFormat = 1;
pub const KV_Q4: KvFormat = 2;
/// Hybrid K8V4 identifies a snapshot layout only.
pub const KV_HYBRID: KvFormat = 3;
/// Rotated INT8 K/V (#293): those bytes mean nothing to a state that does not
/// rotate, so a rotated snapshot never restores into an unrotated state. Q4_0 is
/// always rotated, so it needs no marker.
pub const KV_ROTATED: KvFormat = 16;

/// `kv_q4_bytes_per_head`: `block_q4_0` blocks per cell per head.
pub fn q4_bytes_per_head(head_dim: i64) -> u64 {
    (head_dim / QK4_0) as u64 * Q4_BLOCK_BYTES
}

/// `qsa_kv_format`: the format for the block movers. A hybrid state must never
/// reach them — the C++ stops the process there, and every caller here checks
/// hybridity first, so the case is unreachable rather than merely unhandled.
pub fn kv_format(st: &QsaState) -> KvFormat {
    if st.hybrid {
        panic!("strata: qsa_kv_format: a hybrid K8V4 state must never reach the block movers");
    }
    if st.q4 {
        KV_Q4
    } else if st.int8 {
        KV_INT8
    } else {
        KV_F16
    }
}

/// The pool pointers of one set — `KvHostPools`, or the pool fields of
/// `QsaState` — as presence bits. The C++ hands the transfers raw pointers; all
/// this contract does with them is choose one by format and check it is not null,
/// so null-ness is the whole observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pools {
    pub k_pool: bool,
    pub v_pool: bool,
    pub k_q: bool,
    pub v_q: bool,
    pub k_scale: bool,
    pub v_scale: bool,
    pub k_q4: bool,
    pub v_q4: bool,
    /// `idx_pooled` belongs to the state, not to a set: only the state's copy has it.
    pub idx_pooled: bool,
}

impl Pools {
    /// `KvHostPools::present()`: any one of the three code layouts is enough.
    pub fn present(&self) -> bool {
        self.k_pool || self.k_q || self.k_q4
    }

    /// The five region pointers of this set, in `REGIONS` order. `idx_pooled` is
    /// supplied by the state, which is what `owned` carries here for it.
    fn region_ptrs(&self, idx_pooled: bool) -> [bool; 5] {
        [
            self.k_pool,
            self.v_pool,
            self.k_scale,
            self.v_scale,
            idx_pooled,
        ]
    }
}

/// The streaming residency map, as far as this contract checks it: that every
/// array exists and its extents match the state's. It never reads a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamMap {
    pub page_table: bool,
    pub slot_block: bool,
    pub slot_stamp: bool,
    pub slot_ref: bool,
    pub ctl: bool,
    pub miss_block: bool,
    pub miss_slot: bool,
    pub n_blocks: i64,
    pub n_slots: i64,
}

impl StreamMap {
    fn complete(&self) -> bool {
        self.page_table
            && self.slot_block
            && self.slot_stamp
            && self.slot_ref
            && self.ctl
            && self.miss_block
            && self.miss_slot
    }
}

/// The `QsaState` fields this contract reads. The layout flags and extents are
/// values; the state's device memory is presence bits, because the contract only
/// ever selects among pointers and checks them for null.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QsaState {
    /// 0: every page resident (identity table). 1: streamed, so `host` is
    /// authoritative. 2: a ring — the MTP drafter's window.
    pub mode: i32,
    pub q4: bool,
    pub int8: bool,
    pub hybrid: bool,
    pub rot: bool,
    pub max_cells: i64,
    pub n_pages: i64,
    pub n_slots: i64,
    pub idx_pooled_rows: i64,
    /// The indexer's own device memory, checked by the running-state half.
    pub idx_pooled: bool,
    pub idx_tail: bool,
    pub idx_dead: bool,
    pub idx_block_pos: bool,
    pub page_table: bool,
    /// The VRAM slots the readers see.
    pub slots: Pools,
    /// The authoritative host copy. A state with no host set carries `Pools` with
    /// every field false.
    pub host: Pools,
    pub map: Option<StreamMap>,
}

impl QsaState {
    /// `pools(st, resident)`: the five pointers a transfer addresses, by format
    /// and residency. A streamed or ring state is authoritative in its host
    /// pools; the slots are replaceable. Hybrid is mode 0 only, so it has no host
    /// set to choose between.
    pub fn pool_ptrs(&self, resident: bool) -> [bool; 5] {
        let host = self.mode != 0 && !resident;
        let src = if host { &self.host } else { &self.slots };
        if self.hybrid {
            // K in INT8, V in the rotated Q4_0 pool: only the slots exist.
            [
                self.slots.k_q,
                self.slots.v_q4,
                self.slots.k_scale,
                false,
                self.idx_pooled,
            ]
        } else if self.q4 {
            [src.k_q4, src.v_q4, false, false, self.idx_pooled]
        } else if self.int8 {
            [src.k_q, src.v_q, src.k_scale, src.v_scale, self.idx_pooled]
        } else {
            src.region_ptrs(self.idx_pooled)
        }
    }
}

/// The five byte regions of one layer's K/V, in the C++ `std::array` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    K,
    V,
    KScale,
    VScale,
    Pooled,
}

pub const REGIONS: [Region; 5] = [
    Region::K,
    Region::V,
    Region::KScale,
    Region::VScale,
    Region::Pooled,
];

/// Which layer's pools a transfer addresses. `Qsa` carries the GLOBAL ordinal, as
/// `qsa_states[qsa_ord0 + j]` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    Qsa { ordinal: usize },
    Draft,
}

/// Which pool set: the authoritative one (the host pools when streamed) or the
/// VRAM slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolSet {
    Authoritative,
    Slots,
}

/// One destination of a transfer: which layer, which set, which region.
#[derive(Debug, Clone, Copy)]
struct Endpoint {
    layer: Layer,
    set: PoolSet,
    region: Region,
    /// Whether that pool pointer exists; a missing one is refused before the copy.
    present: bool,
}

/// The running-state buffers a snapshot carries, as the transfer seam names them.
/// `ordinal` is the QSA layer's GLOBAL ordinal (the state's index in
/// `qsa_states`), which is what the C++ uses to reach it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Running {
    Gdn,
    Ple,
    Indexer { ordinal: usize, part: RunningPart },
}

/// Which indexer buffer of one QSA layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunningPart {
    Tail,
    Dead,
    BlockPos,
    /// The moving spare row: `idx_pooled + row * idx_key_dim * 4`.
    PooledRow,
}

/// Byte-level access to the session's device-side state — the seam between this
/// contract and the CUDA/HIP layer.
///
/// Two pairs of operations, matching the C++'s two copy helpers, because they
/// report failures with different prefixes and the message a caller sees is part
/// of the contract:
///
/// * [`Device::read`] / [`Device::write`] move K/V blocks — `cudaMemcpy(...,
///   cudaMemcpyDefault)`, reported as `"conversation snapshot copy: ..."`.
/// * [`Device::read_running`] / [`Device::write_running`] move the running state
///   (`conversation_state.cpp`'s `copy()`), reported as `"conversation snapshot
///   running-state copy: ..."`.
///
/// Implementations return the raw `cudaErrorString`; the call site adds the prefix,
/// so a shim-backed implementation and the host replay report the same message for
/// the same failure. `at` is a byte offset within the target — sizes come from
/// [`layout`], so this never has to ask how big a region is. Null targets are the
/// caller's check: the contract refuses a missing pool before it calls here,
/// exactly as the C++ checks its pointers before `cudaMemcpy`.
pub trait Device {
    /// Take one region's bytes OUT of the state (a save, a verify read-back).
    fn read(
        &mut self,
        layer: Layer,
        set: PoolSet,
        region: Region,
        at: usize,
        host: &mut [u8],
    ) -> Result<(), String>;

    /// Put one region's bytes INTO the state (a restore).
    fn write(
        &mut self,
        layer: Layer,
        set: PoolSet,
        region: Region,
        at: usize,
        host: &[u8],
    ) -> Result<(), String>;

    /// The running state, out (a checkpoint save).
    fn read_running(&mut self, target: Running, at: usize, host: &mut [u8]) -> Result<(), String>;

    /// The running state, in (a checkpoint restore).
    fn write_running(&mut self, target: Running, at: usize, host: &[u8]) -> Result<(), String>;

    /// `cudaDeviceSynchronize`; the C++ prefixes a failure with
    /// `"conversation snapshot synchronize: "`.
    fn sync(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// After a restore: make every evicted block unreadable so the next resolve
    /// refills the slots from the restored authoritative pools
    /// (`kv_stream_reset`). Called only for `mode == 1`.
    fn reset_residency(&mut self, _layer: Layer) -> Result<(), String> {
        Ok(())
    }

    /// Refill the drafter's ring window (`kv_ring_restore`). Called only for
    /// `mode == 2` with a nonzero extent.
    fn restore_ring(&mut self, _layer: Layer, _upto: i64) -> Result<(), String> {
        Ok(())
    }

    /// `cudaGetLastError()` after the residency work: a launch failure from the
    /// two above surfaces here, not at the transfer.
    fn last_error(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// One layer's identity-layout K/V and completed indexer rows. For streamed
/// layers the bytes came from the authoritative host pool, not the replaceable
/// VRAM slots.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConversationKv {
    pub format: KvFormat,
    pub cells: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub page_size: i64,
    pub pooled_rows: i64,
    pub idx_dim: i64,
    pub k: ConversationBuffer,
    pub v: ConversationBuffer,
    pub k_scale: ConversationBuffer,
    pub v_scale: ConversationBuffer,
    pub pooled: ConversationBuffer,
}

impl ConversationKv {
    pub fn region(&self, r: Region) -> &ConversationBuffer {
        match r {
            Region::K => &self.k,
            Region::V => &self.v,
            Region::KScale => &self.k_scale,
            Region::VScale => &self.v_scale,
            Region::Pooled => &self.pooled,
        }
    }

    pub fn region_mut(&mut self, r: Region) -> &mut ConversationBuffer {
        match r {
            Region::K => &mut self.k,
            Region::V => &mut self.v,
            Region::KScale => &mut self.k_scale,
            Region::VScale => &mut self.v_scale,
            Region::Pooled => &mut self.pooled,
        }
    }

    /// `{k, v, k_scale, v_scale, pooled}` sizes, in `REGIONS` order.
    pub fn region_sizes(&self) -> [usize; 5] {
        [
            self.k.size(),
            self.v.size(),
            self.k_scale.size(),
            self.v_scale.size(),
            self.pooled.size(),
        ]
    }

    /// Bytes held, counting each segment's capacity and every directory. This is
    /// the retained term of the capture estimate: the part already resident, which
    /// needs no new admission.
    pub fn bytes(&self) -> usize {
        self.k.bytes()
            + self.v.bytes()
            + self.k_scale.bytes()
            + self.v_scale.bytes()
            + self.pooled.bytes()
    }
}

/// Retained K/V from a previous capture, plus what may be reused from it.
///
/// `stages` is the layer-split case (each later stage keeps its own, at the same
/// extents); the replay leaves it empty and [`ConversationKvReuse::bytes`] counts
/// it either way.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConversationKvReuse {
    pub kv: crate::conv_vec::CppVec<ConversationKv>,
    /// The extent the image was validated at, and the earliest subsequent rewrite.
    pub captured_tokens: i64,
    pub unchanged_tokens: i64,
    pub stages: crate::conv_vec::CppVec<ConversationKvReuse>,
}

impl ConversationKvReuse {
    /// `bytes()`: the directories retained plus every layer's own bytes. The
    /// `kv.capacity()` term is why a capture that grew its directory transiently
    /// keeps costing RAM it no longer uses — hence the modelled capacity.
    pub fn bytes(&self) -> usize {
        let mut n = self.kv.bytes() + self.stages.bytes();
        for layer in self.kv.iter() {
            n += layer.bytes();
        }
        for stage in self.stages.iter() {
            n += stage.bytes();
        }
        n
    }
}

/// `struct Layout`: the byte plan of one layer at one extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Layout {
    pub format: KvFormat,
    pub cells: i64,
    pub pooled_rows: i64,
    pub page_size: i64,
    pub data: usize,
    pub scales: usize,
    pub pooled: usize,
    pub value_data: usize,
    pub value_scales: usize,
}

impl Layout {
    /// The five region sizes, in `REGIONS` order.
    pub fn region_sizes(&self) -> [usize; 5] {
        [
            self.data,
            self.value_data,
            self.scales,
            self.value_scales,
            self.pooled,
        ]
    }
}

/// One K/V operation's target: which state, which geometry, how far, and whether
/// the indexer's pooled rows are part of it (`false` for the draft layer, whose
/// attention has no indexer). Grouped because every entry point takes all four,
/// exactly as the C++ signature does.
#[derive(Debug, Clone, Copy)]
pub struct Extent<'a> {
    pub st: &'a QsaState,
    pub g: &'a ModelGeometry,
    pub upto: i64,
    pub index: bool,
    /// Which layer's pools this names. `save`/`restore`/`verify` address the state
    /// through it, so it travels with the extent instead of adding an argument to
    /// every entry point.
    pub layer_hint: Layer,
}

impl<'a> Extent<'a> {
    pub fn new(
        layer: Layer,
        st: &'a QsaState,
        g: &'a ModelGeometry,
        upto: i64,
        index: bool,
    ) -> Self {
        Self {
            st,
            g,
            upto,
            index,
            layer_hint: layer,
        }
    }

    /// `layout()` for this extent.
    pub fn layout(&self, l: &mut Layout, error: &mut String) -> bool {
        layout(self.st, self.g, self.upto, self.index, l, error)
    }
}

const PREFIX: &str = "conversation snapshot: ";
const MISSING_BUFFER: &str = "conversation snapshot: missing state buffer";
const COPY_PREFIX: &str = "conversation snapshot copy: ";

fn fail(error: &mut String, message: &str) -> bool {
    *error = format!("{PREFIX}{message}");
    false
}

/// `valid_extent`: the extent fits the arena, and rounding it up to a whole page
/// cannot overflow.
pub fn valid_extent(st: &QsaState, upto: i64, error: &mut String) -> bool {
    if upto < 0 || upto > st.max_cells || upto > i64::MAX - (PAGE_SIZE - 1) {
        return fail(error, "invalid K/V extent");
    }
    true
}

/// `layout`: the byte plan for one layer at one extent, or the reason there is
/// none.
pub fn layout(
    st: &QsaState,
    g: &ModelGeometry,
    upto: i64,
    index: bool,
    out: &mut Layout,
    error: &mut String,
) -> bool {
    if !valid_extent(st, upto, error) {
        return false;
    }
    if st.hybrid && (st.mode != 0 || st.q4 || st.int8) {
        return fail(
            error,
            "hybrid K8V4 requires an identity layout and distinct format flags",
        );
    }
    let int8_keys = st.int8 || st.hybrid;
    if g.n_head_kv <= 0
        || g.head_dim <= 0
        || g.head_dim > i64::from(i32::MAX)
        || g.idx_key_dim <= 0
        || (st.q4 && g.head_dim % QK4_0 != 0)
        || (int8_keys && !st.q4 && g.head_dim % INT8_GROUP != 0)
    {
        return fail(error, "invalid K/V geometry");
    }
    let cells = (upto + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE;
    let bytes_per_head = if st.q4 {
        q4_bytes_per_head(g.head_dim)
    } else if int8_keys {
        g.head_dim as u64
    } else {
        g.head_dim as u64 * 2
    };
    // Include the moving spare row, not only completed blocks: the checkpoint
    // restore reconstructs that row when it rewinds to an earlier prefix.
    let pooled_rows = if index && upto > 0 {
        upto / IDX_BLOCK + 1
    } else {
        0
    };
    // Format 3 identifies a snapshot K8V4 only; +16 marks rotated INT8 (#293).
    let rotated = if st.rot && !st.q4 && !st.hybrid {
        KV_ROTATED
    } else {
        0
    };
    let mut l = Layout {
        format: (if st.hybrid { KV_HYBRID } else { kv_format(st) }) + rotated,
        cells,
        pooled_rows,
        page_size: PAGE_SIZE,
        data: 0,
        scales: 0,
        pooled: 0,
        value_data: 0,
        value_scales: 0,
    };
    let scale_per_head = if int8_keys && !st.q4 {
        (g.head_dim / INT8_GROUP) as u64 * 2
    } else {
        0
    };
    if !product(
        &mut l.data,
        &[cells as u64, g.n_head_kv as u64, bytes_per_head],
    ) || !product(
        &mut l.scales,
        &[cells as u64, g.n_head_kv as u64, scale_per_head],
    ) || !product(
        &mut l.pooled,
        &[
            pooled_rows as u64,
            g.idx_key_dim as u64,
            std::mem::size_of::<f32>() as u64,
        ],
    ) {
        return fail(error, "K/V byte count overflow");
    }
    l.value_data = l.data;
    l.value_scales = l.scales;
    if st.hybrid {
        // K in INT8, V in rotated Q4_0: same cells, different bytes per head.
        if !product(
            &mut l.value_data,
            &[
                cells as u64,
                g.n_head_kv as u64,
                q4_bytes_per_head(g.head_dim),
            ],
        ) {
            return fail(error, "hybrid V byte count overflow");
        }
        l.value_scales = 0;
    }
    *out = l;
    true
}

/// `valid`: does this state own the extents `layout` assumed? No device calls, no
/// writes — the gate a restore must pass before its first transfer.
pub fn valid(st: &QsaState, l: &Layout, upto: i64, error: &mut String) -> bool {
    if upto < 0
        || upto > st.max_cells
        || st.n_pages < 0
        || l.cells / l.page_size > st.n_pages
        || st.mode < 0
        || st.mode > 2
        || st.n_slots <= 0
        || (st.mode == 0 && st.n_slots < st.n_pages)
        || l.pooled_rows > st.idx_pooled_rows
        || (st.mode != 0 && !st.host.present())
    {
        return fail(
            error,
            "invalid K/V extent or missing authoritative host pool",
        );
    }
    if st.mode == 1
        && st
            .map
            .is_none_or(|m| !m.complete() || m.n_blocks != st.n_pages || m.n_slots != st.n_slots)
    {
        return fail(error, "invalid streaming map");
    }
    if st.mode == 2 {
        // The ring's own slots are what a reader sees, so those are the pointers
        // a ring restore needs, with scales only in INT8 (Q4_0 carries its own).
        let slots = st.pool_ptrs(true);
        let scales_ok = !st.int8 || (slots[2] && slots[3]);
        if !st.page_table || !slots[0] || !slots[1] || !scales_ok {
            return fail(error, "invalid draft ring");
        }
    }
    true
}

/// The C++ `transfer()`: a zero-length copy never reaches the device, a missing
/// pool is refused before it, and a failed copy carries the C++ message.
fn transfer<R: Device + ?Sized>(
    regions: &mut R,
    to: Endpoint,
    at: usize,
    host: &mut [u8],
    to_state: bool,
    failure: &mut Option<String>,
) -> bool {
    if host.is_empty() {
        return true;
    }
    if !to.present {
        *failure = Some(MISSING_BUFFER.to_string());
        return false;
    }
    let result = if to_state {
        regions.write(to.layer, to.set, to.region, at, host)
    } else {
        regions.read(to.layer, to.set, to.region, at, host)
    };
    match result {
        Ok(()) => true,
        Err(e) => {
            *failure = Some(format!("{COPY_PREFIX}{e}"));
            false
        }
    }
}

/// The five endpoints of one extent's transfers: the state's authoritative set
/// (host pools when streamed), plus the resident slot set a ring verification also
/// reads.
fn endpoints(e: &Extent<'_>, resident: bool) -> [Endpoint; 5] {
    let ptrs = e.st.pool_ptrs(resident);
    let set = if resident {
        PoolSet::Slots
    } else {
        PoolSet::Authoritative
    };
    let mut out = [Endpoint {
        layer: Layer::Draft,
        set,
        region: Region::K,
        present: false,
    }; 5];
    for (i, &r) in REGIONS.iter().enumerate() {
        out[i] = Endpoint {
            layer: e.layer(),
            set,
            region: r,
            present: ptrs[i],
        };
    }
    out
}

impl Extent<'_> {
    pub fn layer(&self) -> Layer {
        self.layer_hint
    }
}

/// `conversation_kv_capture_bytes`: peak bytes held to keep one layer at `upto`,
/// including retained capacity and the directories a resize allocates. `image`
/// may be empty, for a pure estimate.
pub fn kv_capture_bytes(
    image: &ConversationKv,
    e: &Extent<'_>,
    bytes: &mut usize,
    error: &mut String,
) -> bool {
    let mut l = Layout::default();
    if !e.layout(&mut l, error) || !valid(e.st, &l, e.upto, error) {
        return false;
    }
    let sizes = l.region_sizes();
    *bytes = 0;
    for (i, &r) in REGIONS.iter().enumerate() {
        let n = image.region(r).allocation_peak(sizes[i]);
        if n == usize::MAX || !add(bytes, n) {
            return fail(error, "segmented K/V allocation overflow");
        }
    }
    true
}

/// `conversation_kv_bytes`: the estimate for a layer with nothing retained yet;
/// `0` when no legal layout exists at this extent.
pub fn kv_bytes(e: &Extent<'_>) -> usize {
    let mut bytes = 0usize;
    let mut error = String::new();
    if kv_capture_bytes(&ConversationKv::default(), e, &mut bytes, &mut error) {
        bytes
    } else {
        0
    }
}

/// `conversation_kv_save`. `unchanged_tokens` is a prefix of retained storage
/// already known good — valid ONLY for storage actually restored into this
/// session, never for equal token ids. `reused_bytes` accumulates bytes whose
/// transfer was skipped, which is what makes an incremental capture cheap.
pub fn kv_save<R: Device + ?Sized>(
    image: &mut ConversationKv,
    regions: &mut R,
    e: &Extent<'_>,
    unchanged_tokens: i64,
    reused_bytes: Option<&mut usize>,
    error: &mut String,
) -> bool {
    let mut l = Layout::default();
    if !e.layout(&mut l, error) || !valid(e.st, &l, e.upto, error) {
        return false;
    }
    if unchanged_tokens < 0 || unchanged_tokens > e.upto || unchanged_tokens > image.cells {
        return fail(error, "invalid unchanged prefix");
    }
    let whole_cells = unchanged_tokens / l.page_size * l.page_size;
    image.format = l.format;
    image.cells = l.cells;
    image.heads = e.g.n_head_kv;
    image.head_dim = e.g.head_dim;
    image.page_size = l.page_size;
    image.pooled_rows = l.pooled_rows;
    image.idx_dim = e.g.idx_key_dim;

    // The source is the state's authoritative set — for a streamed layer the host
    // pools, which is the entire point of the save.
    let src = endpoints(e, false);
    let sizes = l.region_sizes();
    let mut reused = reused_bytes;
    for (i, &r) in REGIONS.iter().enumerate() {
        // Recopy the partial page and the indexer's moving spare row: completed
        // pages/rows strictly before the first rewritten token remain identical.
        let keep = if i == 4 {
            if e.index {
                (unchanged_tokens / IDX_BLOCK) as usize
                    * e.g.idx_key_dim as usize
                    * std::mem::size_of::<f32>()
            } else {
                0
            }
        } else if l.cells != 0 {
            (sizes[i] / l.cells as usize) * whole_cells as usize
        } else {
            0
        };
        if keep > image.region(r).size() {
            return fail(error, "missing reusable prefix");
        }
        image.region_mut(r).resize(sizes[i], 0);
        let mut failure: Option<String> = None;
        // The segment run is the DESTINATION: bytes move out of the state, which
        // is why the harness reports these as `dst=image`.
        let ok = image.region_mut(r).visit(keep, sizes[i] - keep, |dst, at| {
            transfer(regions, src[i], at, dst, false, &mut failure)
        });
        if !ok {
            *error = failure.unwrap_or_else(|| MISSING_BUFFER.to_string());
            return false;
        }
        if let Some(reused) = reused.as_mut() {
            **reused += keep;
        }
    }
    true
}

/// `conversation_kv_validate`: is this image restorable into this state? No
/// device calls, no destination writes — the whole-session prevalidation gate.
pub fn kv_validate(image: &ConversationKv, e: &Extent<'_>, error: &mut String) -> bool {
    let mut l = Layout::default();
    if !e.layout(&mut l, error) || !valid(e.st, &l, e.upto, error) {
        return false;
    }
    if image.format != l.format
        || image.cells != l.cells
        || image.heads != e.g.n_head_kv
        || image.head_dim != e.g.head_dim
        || image.page_size != l.page_size
        || image.pooled_rows != l.pooled_rows
        || image.idx_dim != e.g.idx_key_dim
    {
        return fail(error, "incompatible K/V geometry");
    }
    if image.region_sizes() != l.region_sizes() {
        return fail(error, "invalid K/V payload size");
    }
    for to in endpoints(e, false) {
        if !to.present
            && l.region_sizes()[REGIONS.iter().position(|&r| r == to.region).unwrap()] != 0
        {
            return fail(error, "missing target state buffer");
        }
    }
    true
}

/// `conversation_kv_restore`. The image validates first, so a failure past that
/// point can leave partial state and the caller MUST NOT continue inference.
pub fn kv_restore<R: Device + ?Sized>(
    image: &ConversationKv,
    regions: &mut R,
    e: &Extent<'_>,
    error: &mut String,
) -> bool {
    if !kv_validate(image, e, error) {
        return false;
    }
    let dst = endpoints(e, false);
    let sizes = image.region_sizes();
    for (i, &r) in REGIONS.iter().enumerate() {
        let mut failure: Option<String> = None;
        // The run is borrowed immutably and copied into the state, so it needs a
        // staging buffer: `transfer` takes `&mut [u8]` either direction because
        // the C++ takes a `void*` destination either way.
        let mut staging: Vec<u8> = Vec::new();
        let ok = image.region(r).visit_const(0, sizes[i], |src, at| {
            staging.clear();
            staging.extend_from_slice(src);
            transfer(regions, dst[i], at, &mut staging, true, &mut failure)
        });
        if !ok {
            *error = failure.unwrap_or_else(|| MISSING_BUFFER.to_string());
            return false;
        }
    }
    // The VRAM slots still hold the outgoing conversation: resolve must refill
    // them from the restored authoritative pools before any attention reads.
    if e.st.mode == 1 {
        if let Err(err) = regions.reset_residency(e.layer()) {
            *error = format!("{COPY_PREFIX}{err}");
            return false;
        }
    }
    if e.st.mode == 2 && e.upto > 0 {
        if let Err(err) = regions.restore_ring(e.layer(), e.upto) {
            *error = format!("{COPY_PREFIX}{err}");
            return false;
        }
    }
    if let Err(err) = regions.last_error() {
        *error = format!("conversation snapshot residency restore: {err}");
        return false;
    }
    true
}

/// `conversation_kv_verify`: diagnostic read-back after a synchronized restore.
/// Compares the authoritative bytes and the resident draft-ring pages, and
/// fingerprints the authoritative payload only. Never changes model state.
///
/// The walk follows the image's OWN segment runs, so how many device reads a
/// verification performs is a property of the image's segmentation — exactly as in
/// the C++, where the same `visit` drives the same 64 KiB read-back buffer.
pub fn kv_verify<R: Device + ?Sized>(
    image: &ConversationKv,
    regions: &mut R,
    e: &Extent<'_>,
    fingerprint: &mut u64,
    error: &mut String,
) -> bool {
    if !kv_validate(image, e, error) {
        return false;
    }
    let mut hasher = FnvHash::default();
    let authoritative = endpoints(e, false);
    for (i, &r) in REGIONS.iter().enumerate() {
        let mut failure: Option<String> = None;
        let ok = image
            .region(r)
            .visit_const(0, image.region(r).size(), |expected, at| {
                compare_run(
                    regions,
                    authoritative[i],
                    at,
                    expected,
                    true,
                    &mut hasher,
                    &mut failure,
                )
            });
        if !ok {
            *error = failure.unwrap_or_else(|| MISSING_BUFFER.to_string());
            return false;
        }
    }

    // The draft ring's resident pages must match too. They are a copy of the
    // authoritative bytes, so they are compared and not hashed.
    if e.st.mode == 2 && image.cells > 0 {
        let resident = endpoints(e, true);
        let end = image.cells / image.page_size;
        let begin = (end - e.st.n_slots).max(0);
        for i in 0..4 {
            let r = REGIONS[i];
            if image.region(r).is_empty() {
                continue;
            }
            let page_bytes = image.region(r).size() / end as usize;
            for page in begin..end {
                let base = page as usize * page_bytes;
                // Page `p` lives in slot `p % n_slots`, so the ring offset of a
                // byte at `at` is that slot's page start plus its position in the
                // page — the C++'s own arithmetic.
                let slot_base = page as usize % e.st.n_slots as usize * page_bytes;
                let mut failure: Option<String> = None;
                let ok = image
                    .region(r)
                    .visit_const(base, page_bytes, |expected, at| {
                        compare_run(
                            regions,
                            resident[i],
                            slot_base + at - base,
                            expected,
                            false,
                            &mut hasher,
                            &mut failure,
                        )
                    });
                if !ok {
                    *error = failure.unwrap_or_else(|| MISSING_BUFFER.to_string());
                    return false;
                }
            }
        }
    }
    *fingerprint = hasher.finish();
    true
}

/// Compare one `visit` run against the device in [`VERIFY_WINDOW`] windows,
/// hashing it when it is authoritative payload. The run's bytes are the EXPECTED
/// side: they came out of the image, and the device is read to check it agrees.
fn compare_run<R: Device + ?Sized>(
    regions: &mut R,
    to: Endpoint,
    at: usize,
    expected: &[u8],
    include_hash: bool,
    hasher: &mut FnvHash,
    failure: &mut Option<String>,
) -> bool {
    let mut window = [0u8; VERIFY_WINDOW];
    let mut chunk = 0usize;
    while chunk < expected.len() {
        let n = VERIFY_WINDOW.min(expected.len() - chunk);
        if !transfer(regions, to, at + chunk, &mut window[..n], false, failure) {
            return false;
        }
        if window[..n] != expected[chunk..chunk + n] {
            *failure = Some(format!("{PREFIX}restored K/V bytes differ"));
            return false;
        }
        if include_hash {
            hasher.update(&window[..n]);
        }
        chunk += n;
    }
    true
}

/// fnv1a64, accumulated over the read-back windows.
#[derive(Debug, Clone, Copy)]
struct FnvHash {
    hash: u64,
}

impl Default for FnvHash {
    fn default() -> Self {
        Self {
            hash: crate::FNV_OFFSET,
        }
    }
}

impl FnvHash {
    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.hash ^= u64::from(b);
            self.hash = self.hash.wrapping_mul(crate::FNV_PRIME);
        }
    }

    fn finish(self) -> u64 {
        self.hash
    }
}
