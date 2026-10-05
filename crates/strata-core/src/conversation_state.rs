//! Port of `src/core/conversation_state.cpp` — the running-state half of the
//! conversation snapshot contract.
//!
//! The K/V half is [`crate::conversation_kv`]; this module is the other kind of
//! state a snapshot carries: the GDN recurrence, the ple history, and each QSA
//! layer's indexer (tail, dead set, block positions, pooled rows). Same rules as
//! the K/V half: an invalid image is rejected before a single byte is written, and
//! the byte estimate is the admission decision.
//!
//! The session's layer carve (#216) is why the sizes are a function of their own:
//! `qsa_states` keeps GLOBAL ordinals and holds the owned ones
//! `[qsa_ord0, qsa_ord0 + qsa_alloc)`, and `gdn_state` holds `gdn_alloc` rows. A
//! whole-model session owns everything; a stage of a layer split owns a slice, and
//! saves and restores only what it owns.

use crate::conv_buffer::{add, product};
use crate::conversation::{
    geometry_key, ConversationCheckpoint, ConversationImageKey, ConversationKvReuse,
    ConversationRestore, ConversationStateSizes, ConversationView, SavedConversation, SessionState,
};
use crate::conversation_kv::{self as kv, Device, Running, RunningPart};
use crate::layout::ModelGeometry;

/// `NG_HIST` / `NG_HC_DIM` (`ngram.hpp`): the ple history's shape, in rows and
/// elements per row. Spelled out as the kernel header spells them, because the
/// byte estimate is the product of exactly these.
pub const NG_HIST: u64 = 9;
pub const NG_HC_DIM: u64 = 10240;

/// `qsa_real_shapes().idx_block`: the indexer's granule. The checkpoint restore
/// reconstructs the moving spare row at `tokens / idx_block`, so the byte plan
/// counts completed blocks plus that one row.
pub const IDX_BLOCK: i64 = 4;

const PREFIX: &str = "conversation snapshot: ";
const RUNNING_PREFIX: &str = "conversation snapshot running-state copy: ";
const SYNC_PREFIX: &str = "conversation snapshot synchronize: ";

fn fail(error: &mut String, message: &str) -> bool {
    *error = format!("{PREFIX}{message}");
    false
}

/// `sync(error)`: `cudaDeviceSynchronize` with the C++ message.
fn sync(device: &mut dyn Device, error: &mut String) -> bool {
    match device.sync() {
        Ok(()) => true,
        Err(e) => {
            *error = format!("{SYNC_PREFIX}{e}");
            false
        }
    }
}

/// `copy(dst, src, bytes, error)`: the running-state mover, in the direction a save
/// needs — the state out into the checkpoint's payload. A zero-byte copy is a no-op
/// and a failed copy carries the C++ message; `at` is a byte offset in the target.
fn copy_running_out(
    device: &mut dyn Device,
    target: Running,
    at: usize,
    payload: &mut [u8],
    error: &mut String,
) -> bool {
    if payload.is_empty() {
        return true;
    }
    match device.read_running(target, at, payload) {
        Ok(()) => true,
        Err(e) => {
            *error = format!("{RUNNING_PREFIX}{e}");
            false
        }
    }
}

/// The same mover, in the direction a restore needs: the checkpoint's payload into
/// the state.
fn copy_running_in(
    device: &mut dyn Device,
    target: Running,
    at: usize,
    payload: &[u8],
    error: &mut String,
) -> bool {
    if payload.is_empty() {
        return true;
    }
    match device.write_running(target, at, payload) {
        Ok(()) => true,
        Err(e) => {
            *error = format!("{RUNNING_PREFIX}{e}");
            false
        }
    }
}

/// `image_keys(images, tokens)`: strictly increasing starts, each inside the live
/// token range. A checkpoint's image list must be a prefix of its conversation's.
fn image_keys(images: &[ConversationImageKey], tokens: usize) -> bool {
    let mut previous = -1i64;
    for image in images {
        if image.start <= previous || image.start < 0 || (image.start as u64) >= tokens as u64 {
            return false;
        }
        previous = image.start;
    }
    true
}

/// `checkpoint_targets`: does this session own the running state a checkpoint of
/// `tokens` tokens claims, and does every owned QSA layer own the indexer buffers
/// for it?
fn checkpoint_targets(
    ss: &SessionState,
    g: &ModelGeometry,
    tokens: usize,
    z: &mut ConversationStateSizes,
    error: &mut String,
) -> bool {
    if !conversation_session_sizes(g, ss, z, error) {
        return false;
    }
    if ss.max_cells < 0
        || (tokens as u64) > ss.max_cells as u64
        || (z.gdn != 0 && !ss.gdn_state)
        || (ss.owned_qsa() > 0 && ss.qsa_states.is_empty())
    {
        return fail(error, "invalid session running-state targets");
    }
    for j in 0..ss.owned_qsa() {
        let Some(st) = ss.owned(j) else {
            return fail(error, "invalid session running-state targets");
        };
        let mut pooled_bytes = 0usize;
        if !st.idx_tail
            || !st.idx_dead
            || !st.idx_block_pos
            || !st.idx_pooled
            || st.max_cells < 0
            || (tokens as u64) > st.max_cells as u64
            || (tokens != 0
                && (tokens as u64 / IDX_BLOCK as u64)
                    >= std::cmp::max(st.idx_pooled_rows, 0) as u64)
            || !product(
                &mut pooled_bytes,
                &[
                    tokens as u64 / IDX_BLOCK as u64 + 1,
                    g.idx_key_dim as u64,
                    std::mem::size_of::<f32>() as u64,
                ],
            )
        {
            return fail(error, "invalid indexer running-state target");
        }
    }
    true
}

/// `view_validate`: the live prefix, its images and the checkpoint chain, against
/// what the session owns.
fn view_validate(
    view: &ConversationView<'_>,
    ss: &SessionState,
    g: &ModelGeometry,
    error: &mut String,
) -> bool {
    let mut z = ConversationStateSizes::default();
    if !checkpoint_targets(ss, g, view.ids.len(), &mut z, error) {
        return false;
    }
    if view.ids.is_empty()
        || !image_keys(view.images.as_slice(), view.ids.len())
        || view.ids.as_slice().iter().any(|&id| id < 0)
    {
        return fail(error, "invalid live token/image metadata");
    }
    for c in view.checkpoints.iter() {
        if !c.stage_parts.is_empty()
            || c.ids.is_empty()
            || c.ids.len() > view.ids.len()
            || c.ids.as_slice() != &view.ids.as_slice()[..c.ids.len()]
        {
            return fail(error, "checkpoint is not a live token prefix");
        }
        if !checkpoint_validate(c, ss, g, error) {
            return false;
        }
        let mut image = 0usize;
        for key in view.images.iter() {
            if (key.start as u64) >= c.ids.len() as u64 {
                break;
            }
            if image >= c.imgs.len() || c.imgs.get(image) != Some(key) {
                return fail(error, "checkpoint image identity differs");
            }
            image += 1;
        }
        if image != c.imgs.len() {
            return fail(error, "checkpoint image prefix differs");
        }
    }
    true
}

/// `metadata_bytes(c, total)`: a checkpoint's own payload, counted by SIZE (not
/// capacity — this one is the payload, not the container).
fn metadata_bytes(c: &ConversationCheckpoint, total: &mut usize) -> bool {
    let mut ids = 0usize;
    let mut images = 0usize;
    if !product(
        &mut ids,
        &[c.ids.len() as u64, std::mem::size_of::<i32>() as u64],
    ) || !product(
        &mut images,
        &[
            c.imgs.len() as u64,
            crate::conv_vec::SIZEOF_IMAGE_KEY as u64,
        ],
    ) {
        return false;
    }
    for n in [
        ids,
        images,
        c.gdn.len(),
        c.ple.len(),
        c.tails.len(),
        c.dead.len(),
        c.block_pos.len(),
    ] {
        if !add(total, n) {
            return false;
        }
    }
    true
}

/// `conversation_state_sizes`: whole-model running-state bytes.
pub fn conversation_state_sizes(
    g: &ModelGeometry,
    z: &mut ConversationStateSizes,
    error: &mut String,
) -> bool {
    *z = ConversationStateSizes::default();
    let key = geometry_key(g);
    for (i, &v) in key.iter().enumerate() {
        if v < 0 || (i != 1 && v == 0) {
            return fail(error, "invalid model geometry");
        }
    }
    let mut recurrence = 0usize;
    let mut convolution = 0usize;
    if !product(
        &mut recurrence,
        &[
            g.ssm_state_size as u64,
            g.ssm_v_heads as u64,
            g.ssm_state_size as u64,
        ],
    ) || !product(
        &mut convolution,
        &[g.ssm_conv_channels as u64, (g.ssm_d_conv - 1) as u64],
    ) || !add(&mut recurrence, convolution)
        || !product(
            &mut z.gdn,
            &[
                g.n_gdn_layers() as u64,
                recurrence as u64,
                std::mem::size_of::<f32>() as u64,
            ],
        )
        || !product(
            &mut z.ple,
            &[NG_HIST, NG_HC_DIM, std::mem::size_of::<f32>() as u64],
        )
        || !product(
            &mut z.tail,
            &[
                (IDX_BLOCK - 1) as u64,
                g.idx_key_dim as u64,
                std::mem::size_of::<f32>() as u64,
            ],
        )
        || !product(
            &mut z.dead,
            &[g.idx_key_dim as u64, std::mem::size_of::<f32>() as u64],
        )
    {
        return fail(error, "running-state byte count overflow");
    }
    z.block_pos = std::mem::size_of::<i32>();
    // The totals are overflow probes: the per-layer bytes are counted per layer by
    // the caller, and only the overflow here is a rejection.
    let mut total = 0usize;
    if !product(&mut total, &[g.n_qsa_layers() as u64, z.tail as u64])
        || !product(&mut total, &[g.n_qsa_layers() as u64, z.dead as u64])
        || !product(&mut total, &[g.n_qsa_layers() as u64, z.block_pos as u64])
    {
        return fail(error, "indexer byte count overflow");
    }
    true
}

/// `conversation_session_sizes`: the same for one session's layer carve. `gdn`
/// covers its `gdn_alloc` rows; the per-QSA-layer sizes apply to each owned state.
pub fn conversation_session_sizes(
    g: &ModelGeometry,
    ss: &SessionState,
    z: &mut ConversationStateSizes,
    error: &mut String,
) -> bool {
    if !conversation_state_sizes(g, z, error) {
        return false;
    }
    let n_gdn = g.n_gdn_layers();
    let n_qsa = g.n_qsa_layers();
    if ss.gdn_alloc < 0
        || ss.gdn_alloc > n_gdn
        || ss.qsa_ord0 < 0
        || ss.qsa_alloc < 0
        || ss.qsa_ord0 > n_qsa
        || ss.qsa_alloc > n_qsa - ss.qsa_ord0
    {
        return fail(error, "invalid session layer carve");
    }
    z.gdn = if n_gdn > 0 {
        z.gdn / n_gdn as usize * ss.gdn_alloc as usize
    } else {
        0
    };
    true
}

/// `conversation_checkpoint_validate`.
pub fn checkpoint_validate(
    c: &ConversationCheckpoint,
    ss: &SessionState,
    g: &ModelGeometry,
    error: &mut String,
) -> bool {
    let mut z = ConversationStateSizes::default();
    if !checkpoint_targets(ss, g, c.ids.len(), &mut z, error) {
        return false;
    }
    let layers = ss.owned_qsa();
    if c.gdn.len() != z.gdn
        || c.ple.len() != (if ss.ple_hist { z.ple } else { 0 })
        || c.tails.len() != layers * z.tail
        || c.dead.len() != layers * z.dead
        || c.block_pos.len() != layers * z.block_pos
        || !image_keys(c.imgs.as_slice(), c.ids.len())
    {
        return fail(error, "invalid checkpoint running-state payload");
    }
    true
}

/// `conversation_checkpoint_save`: read the running state OUT of the session into
/// the checkpoint's payload vectors.
pub fn checkpoint_save(
    c: &mut ConversationCheckpoint,
    ss: &SessionState,
    g: &ModelGeometry,
    device: &mut dyn Device,
    error: &mut String,
) -> bool {
    let mut z = ConversationStateSizes::default();
    if !checkpoint_targets(ss, g, c.ids.len(), &mut z, error) {
        return false;
    }
    let layers = ss.owned_qsa();
    c.gdn.resize(z.gdn, &0);
    c.ple.resize(if ss.ple_hist { z.ple } else { 0 }, &0);
    c.tails.resize(layers * z.tail, &0);
    c.dead.resize(layers * z.dead, &0);
    c.block_pos.resize(layers * z.block_pos, &0);
    if !copy_running_out(device, Running::Gdn, 0, c.gdn.as_mut_slice(), error)
        || !copy_running_out(device, Running::Ple, 0, c.ple.as_mut_slice(), error)
    {
        return false;
    }
    for j in 0..layers {
        let ordinal = ss.qsa_ord0 as usize + j;
        let part = |p| Running::Indexer { ordinal, part: p };
        // Each layer owns its own tail/dead/positions buffer, so the device side is
        // always at offset 0; the checkpoint's vector is the concatenated one.
        if !copy_running_out(
            device,
            part(RunningPart::Tail),
            0,
            &mut c.tails.as_mut_slice()[j * z.tail..(j + 1) * z.tail],
            error,
        ) || !copy_running_out(
            device,
            part(RunningPart::Dead),
            0,
            &mut c.dead.as_mut_slice()[j * z.dead..(j + 1) * z.dead],
            error,
        ) || !copy_running_out(
            device,
            part(RunningPart::BlockPos),
            0,
            &mut c.block_pos.as_mut_slice()[j * z.block_pos..(j + 1) * z.block_pos],
            error,
        ) {
            return false;
        }
    }
    true
}

/// `conversation_checkpoint_restore`. Validates first; past that point a failure
/// can leave partial state. The final `sync` is the C++'s own.
pub fn checkpoint_restore(
    c: &ConversationCheckpoint,
    ss: &mut SessionState,
    g: &ModelGeometry,
    device: &mut dyn Device,
    error: &mut String,
) -> bool {
    if !checkpoint_validate(c, ss, g, error) {
        return false;
    }
    let mut z = ConversationStateSizes::default();
    if !conversation_session_sizes(g, ss, &mut z, error) {
        return false;
    }
    if !copy_running_in(device, Running::Gdn, 0, c.gdn.as_slice(), error)
        || !copy_running_in(device, Running::Ple, 0, c.ple.as_slice(), error)
    {
        return false;
    }
    for j in 0..ss.owned_qsa() {
        let ordinal = ss.qsa_ord0 as usize + j;
        let part = |p| Running::Indexer { ordinal, part: p };
        // The device side is the layer's own buffer, so offset 0; the checkpoint's
        // vector is the concatenated one.
        if !copy_running_in(
            device,
            part(RunningPart::Tail),
            0,
            &c.tails.as_slice()[j * z.tail..(j + 1) * z.tail],
            error,
        ) || !copy_running_in(
            device,
            part(RunningPart::Dead),
            0,
            &c.dead.as_slice()[j * z.dead..(j + 1) * z.dead],
            error,
        ) || !copy_running_in(
            device,
            part(RunningPart::BlockPos),
            0,
            &c.block_pos.as_slice()[j * z.block_pos..(j + 1) * z.block_pos],
            error,
        ) {
            return false;
        }
        if !c.ids.is_empty() {
            let row = c.ids.len() / IDX_BLOCK as usize;
            let at = row * g.idx_key_dim as usize * std::mem::size_of::<f32>();
            if !copy_running_in(
                device,
                part(RunningPart::PooledRow),
                at,
                &c.dead.as_slice()[j * z.dead..(j + 1) * z.dead],
                error,
            ) {
                return false;
            }
        }
    }
    let tokens = c.ids.len();
    ss.ple_prev[0] = if tokens >= 2 {
        c.ids.as_slice()[tokens - 2]
    } else {
        -1
    };
    ss.ple_prev[1] = if tokens >= 1 {
        c.ids.as_slice()[tokens - 1]
    } else {
        -1
    };
    sync(device, error)
}

/// `conversation_snapshot_bytes`: the whole-session byte estimate — the number
/// admission is decided on.
pub fn conversation_snapshot_bytes(
    view: &ConversationView<'_>,
    ss: &SessionState,
    g: &ModelGeometry,
    draft: Option<&kv::QsaState>,
    bytes: &mut usize,
    error: &mut String,
) -> bool {
    *bytes = 0;
    if !view_validate(view, ss, g, error) {
        return false;
    }
    let mut z = ConversationStateSizes::default();
    if !conversation_session_sizes(g, ss, &mut z, error) {
        return false;
    }
    let mut ids = 0usize;
    let mut images = 0usize;
    let mut checkpoints = 0usize;
    let mut layers = 0usize;
    let mut tails = 0usize;
    let mut dead = 0usize;
    let mut positions = 0usize;
    let qsa = ss.owned_qsa() as u64;
    let draft_layers = u64::from(draft.is_some());
    if !product(
        &mut ids,
        &[view.ids.len() as u64, std::mem::size_of::<i32>() as u64],
    ) || !product(
        &mut images,
        &[
            view.images.len() as u64,
            crate::conv_vec::SIZEOF_IMAGE_KEY as u64,
        ],
    ) || !product(
        &mut checkpoints,
        &[
            view.checkpoints.len() as u64,
            crate::conv_vec::SIZEOF_CHECKPOINT as u64,
        ],
    ) || !product(
        &mut layers,
        &[qsa + draft_layers, crate::conv_vec::SIZEOF_KV as u64],
    ) || !product(&mut tails, &[qsa, z.tail as u64])
        || !product(&mut dead, &[qsa, z.dead as u64])
        || !product(&mut positions, &[qsa, z.block_pos as u64])
    {
        return fail(error, "snapshot metadata byte count overflow");
    }
    for n in [
        ids,
        images,
        checkpoints,
        layers,
        tails,
        dead,
        positions,
        z.gdn,
        if ss.ple_hist { z.ple } else { 0 },
    ] {
        if !add(bytes, n) {
            return fail(error, "snapshot byte count overflow");
        }
    }
    for c in view.checkpoints.iter() {
        if !metadata_bytes(c, bytes) {
            return fail(error, "checkpoint byte count overflow");
        }
    }
    // view_validate bounds this by the signed max_cells, so the cast is safe.
    let upto = view.ids.len() as i64;
    for i in 0..qsa + draft_layers {
        let e = match draft {
            Some(draft) if i == qsa => kv::Extent::new(kv::Layer::Draft, draft, g, upto, false),
            _ => kv::Extent::new(
                kv::Layer::Qsa {
                    ordinal: i as usize,
                },
                ss.owned(i as usize).expect("validated owned state"),
                g,
                upto,
                true,
            ),
        };
        let n = kv::kv_bytes(&e);
        if n == 0 || !add(bytes, n) {
            return fail(error, "invalid or overflowing K/V byte estimate");
        }
    }
    true
}

/// `conversation_snapshot_capture_bytes`: the estimate with retained capacity.
/// `reuse` is consumed by the capture that follows it, including on failure.
pub fn conversation_snapshot_capture_bytes(
    reuse: &ConversationKvReuse,
    view: &ConversationView<'_>,
    ss: &SessionState,
    g: &ModelGeometry,
    draft: Option<&kv::QsaState>,
    bytes: &mut usize,
    error: &mut String,
) -> bool {
    if !conversation_snapshot_bytes(view, ss, g, draft, bytes, error) {
        return false;
    }
    if reuse.kv.is_empty() {
        return true;
    }
    let layers = ss.owned_qsa() + usize::from(draft.is_some());
    if reuse.kv.len() != layers
        || reuse.unchanged_tokens < 0
        || reuse.unchanged_tokens > reuse.captured_tokens
        || reuse.unchanged_tokens > view.ids.len() as i64
    {
        return fail(error, "invalid retained K/V prefix");
    }
    for i in 0..layers {
        let index = i < ss.owned_qsa();
        let layer = if index {
            kv::Layer::Qsa { ordinal: i }
        } else {
            kv::Layer::Draft
        };
        let st = if index {
            ss.owned(i).expect("validated owned state")
        } else {
            draft.expect("counted draft layer")
        };
        let image = reuse.kv.get(i).expect("validated layer count");
        if !kv::kv_validate(
            image,
            &kv::Extent::new(layer, st, g, reuse.captured_tokens, index),
            error,
        ) {
            return false;
        }
        let fresh = kv::kv_bytes(&kv::Extent::new(layer, st, g, view.ids.len() as i64, index));
        let mut retained = 0usize;
        if !kv::kv_capture_bytes(
            image,
            &kv::Extent::new(layer, st, g, view.ids.len() as i64, index),
            &mut retained,
            error,
        ) {
            return false;
        }
        *bytes -= fresh;
        if !add(bytes, retained) {
            return fail(error, "retained K/V allocation overflow");
        }
    }
    let mut directory = 0usize;
    if !product(
        &mut directory,
        &[
            (reuse.kv.capacity() - layers) as u64,
            crate::conv_vec::SIZEOF_KV as u64,
        ],
    ) || !add(bytes, directory)
    {
        return fail(error, "retained K/V directory overflow");
    }
    true
}

/// `conversation_snapshot_save`. Builds into a new object so a failure cannot
/// publish a partial snapshot; the active session is never modified.
#[allow(clippy::too_many_arguments)]
pub fn conversation_snapshot_save(
    image: &mut SavedConversation,
    view: &ConversationView<'_>,
    ss: &SessionState,
    g: &ModelGeometry,
    draft: Option<&kv::QsaState>,
    error: &mut String,
    device: &mut dyn Device,
    reuse: &mut ConversationKvReuse,
    mut reused_bytes: Option<&mut usize>,
) -> bool {
    let mut estimate = 0usize;
    if !conversation_snapshot_capture_bytes(reuse, view, ss, g, draft, &mut estimate, error)
        || !sync(device, error)
    {
        return false;
    }
    let mut captured = SavedConversation {
        geometry: geometry_key(g),
        layer_lo: ss.layer_lo,
        layer_hi: ss.layer_hi,
        cvec: view.cvec,
        ..SavedConversation::default()
    };
    captured.live.ids = view.ids.clone();
    captured.live.imgs = view.images.clone();
    captured.checkpoints = view.checkpoints.clone();
    let unchanged = if reuse.kv.is_empty() {
        0
    } else {
        reuse.unchanged_tokens
    };
    captured.kv = std::mem::take(&mut reuse.kv);
    let layers = ss.owned_qsa();
    captured
        .kv
        .resize(layers + usize::from(draft.is_some()), &Default::default());
    if !checkpoint_save(&mut captured.live, ss, g, device, error) {
        return false;
    }
    let upto = view.ids.len() as i64;
    for j in 0..layers {
        let st = ss.owned(j).expect("validated owned state");
        if !kv::kv_save(
            captured.kv.get_mut(j).expect("resized"),
            device,
            &kv::Extent::new(kv::Layer::Qsa { ordinal: j }, st, g, upto, true),
            unchanged,
            reused_bytes.as_deref_mut(),
            error,
        ) {
            return false;
        }
    }
    // The draft's final cell may not have been computed when the output cap was
    // reached: refresh that page even when the main prefix continued unchanged.
    if let Some(draft) = draft {
        let last = captured.kv.len() - 1;
        if !kv::kv_save(
            captured.kv.get_mut(last).expect("resized"),
            device,
            &kv::Extent::new(kv::Layer::Draft, draft, g, upto, false),
            std::cmp::max(0, unchanged - 1),
            reused_bytes,
            error,
        ) {
            return false;
        }
    }
    *image = captured;
    true
}

/// `conversation_snapshot_validate`: is this image restorable into this session?
pub fn conversation_snapshot_validate(
    image: &SavedConversation,
    ss: &SessionState,
    g: &ModelGeometry,
    draft: Option<&kv::QsaState>,
    error: &mut String,
) -> bool {
    if !image.live.stage_parts.is_empty() {
        return fail(
            error,
            "a running checkpoint's stage parts belong in the stage images, not in live.stage_parts",
        );
    }
    if image.geometry != geometry_key(g) {
        return fail(error, "incompatible runtime geometry");
    }
    // An image holds exactly one carve's running state and K/V: same range or nothing.
    if image.layer_lo != ss.layer_lo || image.layer_hi != ss.layer_hi {
        return fail(error, "snapshot from another session layer range");
    }
    let view = ConversationView {
        ids: &image.live.ids,
        images: &image.live.imgs,
        checkpoints: &image.checkpoints,
        cvec: image.cvec,
    };
    if !view_validate(&view, ss, g, error) || !checkpoint_validate(&image.live, ss, g, error) {
        return false;
    }
    let layers = ss.owned_qsa();
    if image.kv.len() != layers + usize::from(draft.is_some()) {
        return fail(error, "invalid K/V layer count");
    }
    let upto = image.live.ids.len() as i64;
    for j in 0..layers {
        let st = ss.owned(j).expect("validated owned state");
        if !kv::kv_validate(
            image.kv.get(j).expect("validated layer count"),
            &kv::Extent::new(kv::Layer::Qsa { ordinal: j }, st, g, upto, true),
            error,
        ) {
            return false;
        }
    }
    match draft {
        Some(draft) => kv::kv_validate(
            image.kv.get(image.kv.len() - 1).expect("counted"),
            &kv::Extent::new(kv::Layer::Draft, draft, g, upto, false),
            error,
        ),
        None => true,
    }
}

/// `conversation_snapshot_restore`. Invalid images are rejected before a single
/// byte is written; a transfer failure may leave partial state, and the caller
/// MUST NOT continue inference from that session.
pub fn conversation_snapshot_restore(
    image: &SavedConversation,
    ss: &mut SessionState,
    g: &ModelGeometry,
    draft: Option<&kv::QsaState>,
    device: &mut dyn Device,
    error: &mut String,
) -> ConversationRestore {
    if !conversation_snapshot_validate(image, ss, g, draft, error) {
        return ConversationRestore::Invalid;
    }
    if !sync(device, error) {
        return ConversationRestore::TransferFailed;
    }
    let upto = image.live.ids.len() as i64;
    for j in 0..ss.owned_qsa() {
        let st = ss.owned(j).expect("validated owned state");
        if !kv::kv_restore(
            image.kv.get(j).expect("validated layer count"),
            device,
            &kv::Extent::new(kv::Layer::Qsa { ordinal: j }, st, g, upto, true),
            error,
        ) {
            return ConversationRestore::TransferFailed;
        }
    }
    if let Some(draft) = draft {
        let last = image.kv.len() - 1;
        if !kv::kv_restore(
            image.kv.get(last).expect("counted"),
            device,
            &kv::Extent::new(kv::Layer::Draft, draft, g, upto, false),
            error,
        ) {
            return ConversationRestore::TransferFailed;
        }
    }
    if !checkpoint_restore(&image.live, ss, g, device, error) {
        return ConversationRestore::TransferFailed;
    }
    ConversationRestore::Restored
}

/// The whole-session form of the entry points, where `draft` is a reference rather
/// than a pointer: an image WITHOUT the draft layer's K/V passes `None`, and a
/// layer split's stage image is restored stage by stage, never through here.
pub fn snapshot_bytes(
    view: &ConversationView<'_>,
    ss: &SessionState,
    g: &ModelGeometry,
    draft: &kv::QsaState,
    bytes: &mut usize,
    error: &mut String,
) -> bool {
    conversation_snapshot_bytes(view, ss, g, Some(draft), bytes, error)
}

/// The whole-session form: a layer split's image is rejected here, because it is
/// restored stage by stage.
pub fn snapshot_validate(
    image: &SavedConversation,
    ss: &SessionState,
    g: &ModelGeometry,
    draft: &kv::QsaState,
    error: &mut String,
) -> bool {
    if !image.stage_images.is_empty() {
        return fail(error, "a layer split's image restored as a single session");
    }
    conversation_snapshot_validate(image, ss, g, Some(draft), error)
}

/// The whole-session form of [`conversation_snapshot_restore`].
pub fn snapshot_restore(
    image: &SavedConversation,
    ss: &mut SessionState,
    g: &ModelGeometry,
    draft: &kv::QsaState,
    device: &mut dyn Device,
    error: &mut String,
) -> ConversationRestore {
    conversation_snapshot_restore(image, ss, g, Some(draft), device, error)
}

/// The whole-session form of [`conversation_snapshot_capture_bytes`].
pub fn snapshot_capture_bytes(
    reuse: &ConversationKvReuse,
    view: &ConversationView<'_>,
    ss: &SessionState,
    g: &ModelGeometry,
    draft: &kv::QsaState,
    bytes: &mut usize,
    error: &mut String,
) -> bool {
    conversation_snapshot_capture_bytes(reuse, view, ss, g, Some(draft), bytes, error)
}

/// The whole-session form of [`conversation_snapshot_save`].
#[allow(clippy::too_many_arguments)]
pub fn snapshot_save(
    image: &mut SavedConversation,
    view: &ConversationView<'_>,
    ss: &SessionState,
    g: &ModelGeometry,
    draft: &kv::QsaState,
    device: &mut dyn Device,
    error: &mut String,
    reuse: &mut ConversationKvReuse,
    reused_bytes: Option<&mut usize>,
) -> bool {
    conversation_snapshot_save(
        image,
        view,
        ss,
        g,
        Some(draft),
        error,
        device,
        reuse,
        reused_bytes,
    )
}
