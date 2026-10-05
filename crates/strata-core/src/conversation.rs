//! The snapshot's data model — the structs of `conversation_cache.hpp` and
//! `conversation_snapshot.hpp` that the RAM accounting and the validation rules
//! are written against.
//!
//! Every `bytes()` here is an admission decision, not an introspection helper, so
//! the containers are [`CppVec`] replicas with libstdc++'s growth policy and the
//! C++ `sizeof` as the unit: the same snapshot must cost the same bytes in the
//! port as it does in the engine.

use crate::conv_vec::{
    CppVec, SIZEOF_CHECKPOINT, SIZEOF_IMAGE_KEY, SIZEOF_KV, SIZEOF_KV_REUSE, SIZEOF_SAVED,
};
use crate::conversation_kv::ConversationKv;
use crate::layout::ModelGeometry;

/// `sizeof` of the geometry key's elements: the runtime-compatibility fingerprint
/// a snapshot carries, as the C++ `std::array<int64_t, 18>`.
pub const GEOMETRY_KEY_LEN: usize = 18;

/// `geometry_key(g)`: the 18 fields, in the C++'s order. Two runtimes share a
/// snapshot only when this matches exactly.
pub fn geometry_key(g: &ModelGeometry) -> [i64; GEOMETRY_KEY_LEN] {
    [
        g.n_embd,
        g.n_layers,
        g.qsa_interval,
        g.ssm_state_size,
        g.ssm_k_heads,
        g.ssm_v_heads,
        g.ssm_d_conv,
        g.ssm_conv_channels,
        g.ssm_value_dim,
        g.n_head,
        g.n_head_kv,
        g.head_dim,
        g.idx_q_heads,
        g.idx_key_dim,
        g.hc,
        g.hc_lr,
        g.n_expert,
        g.n_ff,
    ]
}

/// One image's identity within a conversation: where its tokens start, and the
/// fingerprint of its pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConversationImageKey {
    pub start: i64,
    pub hash: u64,
}

/// A parked prefix: its tokens, the images it contains, and the running state as
/// it was at that point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationCheckpoint {
    pub ids: CppVec<i32>,
    pub imgs: CppVec<ConversationImageKey>,
    pub gdn: CppVec<u8>,
    pub ple: CppVec<u8>,
    pub tails: CppVec<u8>,
    pub dead: CppVec<u8>,
    pub block_pos: CppVec<u8>,
    /// Upstream root-pinned/LRU retention stamp; the port carries it, never reads it.
    pub used: u64,
    /// Layer-split parking: one part per later stage. A whole-session image must
    /// have none — the engine rejects them there.
    pub stage_parts: CppVec<ConversationCheckpoint>,
}

impl Default for ConversationCheckpoint {
    fn default() -> Self {
        Self {
            ids: CppVec::new(),
            imgs: CppVec::new(),
            gdn: CppVec::new(),
            ple: CppVec::new(),
            tails: CppVec::new(),
            dead: CppVec::new(),
            block_pos: CppVec::new(),
            used: 0,
            stage_parts: CppVec::with_unit(SIZEOF_CHECKPOINT),
        }
    }
}

impl ConversationCheckpoint {
    /// Bytes held, exactly as the C++ counts them: every vector's capacity at the
    /// C++ `sizeof`, plus the parts' own bytes.
    pub fn bytes(&self) -> usize {
        self.ids.capacity() * std::mem::size_of::<i32>()
            + self.imgs.capacity() * SIZEOF_IMAGE_KEY
            + self.gdn.capacity()
            + self.ple.capacity()
            + self.tails.capacity()
            + self.dead.capacity()
            + self.block_pos.capacity()
            + self.stage_parts.capacity() * SIZEOF_CHECKPOINT
            + self.stage_parts.iter().map(|p| p.bytes()).sum::<usize>()
    }
}

/// Retained K/V from a previous capture, plus what may be reused from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationKvReuse {
    pub kv: CppVec<ConversationKv>,
    /// The extent the image was validated at, and the earliest subsequent rewrite.
    pub captured_tokens: i64,
    pub unchanged_tokens: i64,
    /// With a layer split: the later stages' own retained K/V, one per stage.
    pub stages: CppVec<ConversationKvReuse>,
}

impl Default for ConversationKvReuse {
    fn default() -> Self {
        Self {
            kv: CppVec::with_unit(SIZEOF_KV),
            captured_tokens: 0,
            unchanged_tokens: 0,
            stages: CppVec::with_unit(SIZEOF_KV_REUSE),
        }
    }
}

impl ConversationKvReuse {
    pub fn bytes(&self) -> usize {
        self.kv.capacity() * SIZEOF_KV
            + self.stages.capacity() * SIZEOF_KV_REUSE
            + self.kv.iter().map(|l| l.bytes()).sum::<usize>()
            + self.stages.iter().map(|s| s.bytes()).sum::<usize>()
    }
}

/// A parked conversation: one image per layer-split stage, each holding that
/// stage's running state and K/V.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedConversation {
    /// Runtime compatibility only; not a model/weights identity or a disk schema.
    pub geometry: [i64; GEOMETRY_KEY_LEN],
    /// The session layer range the image was captured from; restore needs the same.
    pub layer_lo: i64,
    pub layer_hi: i64,
    pub live: ConversationCheckpoint,
    pub checkpoints: CppVec<ConversationCheckpoint>,
    /// The session's own QSA layers, then the draft layer.
    pub kv: CppVec<ConversationKv>,
    pub cvec: bool,
    pub stage_images: CppVec<SavedConversation>,
}

impl Default for SavedConversation {
    fn default() -> Self {
        Self {
            geometry: [0; GEOMETRY_KEY_LEN],
            layer_lo: 0,
            layer_hi: 0,
            live: ConversationCheckpoint::default(),
            checkpoints: CppVec::with_unit(SIZEOF_CHECKPOINT),
            kv: CppVec::with_unit(SIZEOF_KV),
            cvec: true,
            stage_images: CppVec::with_unit(SIZEOF_SAVED),
        }
    }
}

impl SavedConversation {
    pub fn bytes(&self) -> usize {
        self.live.bytes()
            + self.checkpoints.capacity() * SIZEOF_CHECKPOINT
            + self.kv.capacity() * SIZEOF_KV
            + self.stage_images.iter().map(|s| s.bytes()).sum::<usize>()
            + self.checkpoints.iter().map(|c| c.bytes()).sum::<usize>()
            + self.kv.iter().map(|k| k.bytes()).sum::<usize>()
    }
}

/// What the caller wants parked or restored: the live token prefix, its images,
/// the checkpoint chain, and the steering mode.
#[derive(Debug, Clone, Copy)]
pub struct ConversationView<'a> {
    pub ids: &'a CppVec<i32>,
    pub images: &'a CppVec<ConversationImageKey>,
    pub checkpoints: &'a CppVec<ConversationCheckpoint>,
    pub cvec: bool,
}

/// `ConversationStateSizes`: whole-model running-state bytes. `gdn` covers every
/// GDN layer; the indexer sizes are per QSA layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConversationStateSizes {
    pub gdn: usize,
    pub ple: usize,
    pub tail: usize,
    pub dead: usize,
    pub block_pos: usize,
}

/// The session fields this contract reads — the view of `SessionState` that the
/// snapshot rules need, not the whole session object. `qsa_states` holds the
/// owned ones at `[qsa_ord0, qsa_ord0 + qsa_alloc)`; a session with none has an
/// empty slice, which is what a null `qsa_states` means to the C++.
#[derive(Debug, Clone, Default)]
pub struct SessionState {
    pub max_cells: i64,
    /// Whether the session owns the GDN state and the ple history.
    pub gdn_state: bool,
    pub ple_hist: bool,
    /// The session's layer carve: `[layer_lo, layer_hi)`, all of it on one GPU.
    pub layer_lo: i64,
    pub layer_hi: i64,
    /// How many GDN rows / QSA states this session owns, and where its QSA states
    /// start in the global array.
    pub gdn_alloc: i64,
    pub qsa_ord0: i64,
    pub qsa_alloc: i64,
    pub qsa_states: Vec<crate::conversation_kv::QsaState>,
    pub ple_prev: [i32; 2],
}

impl SessionState {
    /// `owned_qsa(ss)`: how many QSA states this session owns.
    pub fn owned_qsa(&self) -> usize {
        std::cmp::max(self.qsa_alloc, 0) as usize
    }

    /// `owned(ss, j)`: the j-th QSA state this session owns, by global ordinal.
    pub fn owned(&self, j: usize) -> Option<&crate::conversation_kv::QsaState> {
        self.qsa_states.get(self.qsa_ord0 as usize + j)
    }
}

/// `ConversationRestore`: the outcome of a restore, as the C++ reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationRestore {
    Restored,
    Invalid,
    TransferFailed,
}

impl ConversationRestore {
    /// The harness's name for the outcome, used in the golden corpus.
    pub fn corpus_name(self) -> &'static str {
        match self {
            Self::Restored => "restored",
            Self::Invalid => "invalid",
            Self::TransferFailed => "transfer_failed",
        }
    }
}
