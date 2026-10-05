//! Byte-parity replay of the C++ conversation-snapshot contract.
//!
//! `tools/conversation_corpus.cpp` compiles the real
//! `src/core/conversation_state.cpp` + `src/core/conversation_snapshot.cpp` from
//! a Strata checkout against stub CUDA symbols and prints one `CORPUS|…` line per
//! observation: byte estimates, error strings, rejection verdicts, the exact
//! sequence of device transfers (buffer, offset, byte count), and an fnv1a64 of
//! every state buffer after every phase. The golden file is that output, checked
//! in.
//!
//! This test replays the same programs against the Rust port and compares the
//! lines. The device seam is a host-test backend: named byte buffers standing in
//! for the pools, with the same fault injection (`fail the Nth copy`, `fail the
//! Nth sync`) so the failure paths are replayed, not assumed.

use std::collections::BTreeMap;

use strata_core::conv_buffer::ConversationBuffer as Buf;
use strata_core::conv_vec::{CppVec, SIZEOF_CHECKPOINT, SIZEOF_KV, SIZEOF_KV_REUSE, SIZEOF_SAVED};
use strata_core::conversation::{
    ConversationCheckpoint, ConversationImageKey, ConversationKvReuse, ConversationRestore,
    ConversationStateSizes, ConversationView, SavedConversation, SessionState,
};
use strata_core::conversation_kv::{
    self as kv, ConversationKv, Device, Layer, PoolSet, QsaState, Region, Running, RunningPart,
};
use strata_core::conversation_state as cs;
use strata_core::layout::ModelGeometry;

const FNV_OFFSET: u64 = 1_469_598_103_934_665_603;
const FNV_PRIME: u64 = 1_099_511_628_211;

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn golden() -> Vec<String> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden_conversation_corpus.txt"
    );
    let text = std::fs::read_to_string(path).expect("golden corpus checked in");
    text.lines()
        .filter(|l| l.starts_with("CORPUS|"))
        .map(str::to_string)
        .collect()
}

fn esc(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '|' => o.push_str("%7C"),
            '\n' => o.push_str("%0A"),
            _ => o.push(c),
        }
    }
    o
}

fn yesno(b: bool) -> &'static str {
    if b {
        "1"
    } else {
        "0"
    }
}

// ============================================================ the host-test device
/// The device seam, backed by named byte buffers. `read`/`write` move bytes
/// between the caller and one named buffer exactly as `cudaMemcpy(..., Default)`
/// does for the C++, and every call is logged the way the harness logs it — the
/// log IS the transfer plan the port claims to reproduce.
struct Backend {
    buffers: BTreeMap<String, Vec<u8>>,
    /// The buffers in the harness's fingerprint order.
    order: Vec<String>,
    ple_prev: [i32; 2],
    copy_calls: usize,
    sync_calls: usize,
    fail_copy: usize,
    fail_sync: usize,
    copied_bytes: usize,
    log: Vec<String>,
}

impl Backend {
    fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
            order: Vec::new(),
            ple_prev: [-1, -1],
            copy_calls: 0,
            sync_calls: 0,
            fail_copy: 0,
            fail_sync: 0,
            copied_bytes: 0,
            log: Vec::new(),
        }
    }

    /// `reg()`: an empty buffer owns no bytes, so it is not registered and
    /// contributes nothing to the fingerprint — the harness's own rule.
    fn reg_ordered(&mut self, name: &str, bytes: Vec<u8>) {
        let empty = bytes.is_empty();
        self.order.push(name.to_string());
        if !empty {
            self.buffers.insert(name.to_string(), bytes);
        }
    }

    fn reset_state(&mut self) {
        for b in self.buffers.values_mut() {
            for v in b.iter_mut() {
                *v = 0xa5;
            }
        }
        self.ple_prev = [-1, -1];
        self.copy_calls = 0;
        self.sync_calls = 0;
        self.fail_copy = 0;
        self.fail_sync = 0;
        self.copied_bytes = 0;
    }

    fn state_fingerprint(&self) -> u64 {
        let mut h = FNV_OFFSET;
        for name in &self.order {
            if let Some(b) = self.buffers.get(name) {
                for &byte in b {
                    h ^= u64::from(byte);
                    h = h.wrapping_mul(FNV_PRIME);
                }
            }
        }
        for &p in &self.ple_prev {
            for &byte in &p.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
        h
    }

    /// One `cudaMemcpy(..., Default)`: log it, then move the bytes unless this is
    /// the injected failure — which is reported the way `cudaMemcpy` reports it, as
    /// an error the caller turns into the C++ message. `device` names the device
    /// side; the image side is the harness's `"image"`.
    fn memcpy(
        &mut self,
        device: &str,
        at: usize,
        buffer: &mut [u8],
        to_state: bool,
    ) -> Result<(), String> {
        self.copy_calls += 1;
        let failed = self.fail_copy != 0 && self.copy_calls == self.fail_copy;
        let named = |d: bool| {
            if d {
                format!("{device}:{at}")
            } else {
                "image".to_string()
            }
        };
        self.log.push(format!(
            "CORPUS|COPY|n={}|bytes={}|dst={}|src={}{}",
            self.copy_calls,
            buffer.len(),
            named(to_state),
            named(!to_state),
            if failed { "|FAILED" } else { "" }
        ));
        if failed {
            return Err("stub".to_string());
        }
        self.copied_bytes += buffer.len();
        let region = self
            .buffers
            .get_mut(device)
            .expect("device buffer registered");
        let range = at..at + buffer.len();
        if to_state {
            region[range].copy_from_slice(buffer);
        } else {
            buffer.copy_from_slice(&region[range]);
        }
        Ok(())
    }
}

impl Device for Backend {
    fn read(
        &mut self,
        layer: Layer,
        _set: PoolSet,
        region: Region,
        at: usize,
        host: &mut [u8],
    ) -> Result<(), String> {
        self.memcpy(&device_name(layer, Some(region), None), at, host, false)
    }

    fn write(
        &mut self,
        layer: Layer,
        _set: PoolSet,
        region: Region,
        at: usize,
        host: &[u8],
    ) -> Result<(), String> {
        let mut buf = host.to_vec();
        self.memcpy(&device_name(layer, Some(region), None), at, &mut buf, true)
    }

    fn read_running(&mut self, target: Running, at: usize, host: &mut [u8]) -> Result<(), String> {
        self.memcpy(
            &device_name(Layer::Draft, None, Some(target)),
            at,
            host,
            false,
        )
    }

    fn write_running(&mut self, target: Running, at: usize, host: &[u8]) -> Result<(), String> {
        let mut buf = host.to_vec();
        self.memcpy(
            &device_name(Layer::Draft, None, Some(target)),
            at,
            &mut buf,
            true,
        )
    }

    fn sync(&mut self) -> Result<(), String> {
        self.sync_calls += 1;
        if self.fail_sync != 0 && self.sync_calls == self.fail_sync {
            return Err("stub".to_string());
        }
        Ok(())
    }
}

fn region_name(region: &Region) -> &'static str {
    match region {
        Region::K => "k",
        Region::V => "v",
        Region::KScale => "k_scale",
        Region::VScale => "v_scale",
        Region::Pooled => "idx_pooled",
    }
}

fn layer_name(layer: &Layer) -> String {
    match layer {
        Layer::Draft => "DRAFT".to_string(),
        Layer::Qsa { ordinal } => format!("L{ordinal}"),
    }
}

/// The harness's region names: `L0:k`, `DRAFT:idx_pooled`, `gdn`, `ple`.
fn device_name(layer: Layer, region: Option<Region>, running: Option<Running>) -> String {
    if let Some(r) = running {
        return match r {
            Running::Gdn => "gdn".to_string(),
            Running::Ple => "ple".to_string(),
            Running::Indexer { ordinal, part } => {
                let part = match part {
                    RunningPart::Tail => "idx_tail",
                    RunningPart::Dead => "idx_dead",
                    RunningPart::BlockPos => "idx_block_pos",
                    RunningPart::PooledRow => "idx_pooled",
                };
                format!("L{ordinal}:{part}")
            }
        };
    }
    format!("{}:{}", layer_name(&layer), region_name(&region.unwrap()))
}

// ============================================================ the fixtures
/// The fixture's three pool sets, exactly as `conversation_corpus.cpp` builds
/// them: the same buffers, the same fill, the same presence.
struct Pools {
    st: QsaState,
    data: [Vec<u8>; 8],
}

impl Pools {
    fn new(g: &ModelGeometry, format: i32) -> Self {
        let per: i64 = if format == 2 {
            (g.head_dim / 32) * 18
        } else {
            g.head_dim * if format == 1 { 1 } else { 2 }
        };
        let mut data = [const { Vec::new() }; 8];
        data[0] = vec![0xa5; (96 * g.n_head_kv * per) as usize];
        data[1] = data[0].clone();
        data[2] = if format == 1 {
            vec![0xa5; (96 * g.n_head_kv * (g.head_dim / 64) * 2) as usize]
        } else {
            Vec::new()
        };
        data[3] = data[2].clone();
        data[4] = vec![0xa5; (26 * g.idx_key_dim * 4) as usize];
        data[5] = vec![0xa5; (3 * g.idx_key_dim * 4) as usize];
        data[6] = vec![0xa5; (g.idx_key_dim * 4) as usize];
        data[7] = vec![0xa5; 4];

        let mut slots = strata_core::conversation_kv::Pools::default();
        if format == 2 {
            slots.k_q4 = !data[0].is_empty();
            slots.v_q4 = !data[1].is_empty();
        } else if format == 1 {
            slots.k_q = !data[0].is_empty();
            slots.v_q = !data[1].is_empty();
            slots.k_scale = !data[2].is_empty();
            slots.v_scale = !data[3].is_empty();
        } else {
            slots.k_pool = !data[0].is_empty();
            slots.v_pool = !data[1].is_empty();
        }
        let st = QsaState {
            max_cells: 96,
            n_pages: 24,
            n_slots: 24,
            idx_pooled_rows: 26,
            int8: format == 1,
            q4: format == 2,
            idx_pooled: !data[4].is_empty(),
            idx_tail: !data[5].is_empty(),
            idx_dead: !data[6].is_empty(),
            idx_block_pos: !data[7].is_empty(),
            slots,
            ..QsaState::default()
        };
        Self { st, data }
    }

    fn image(&self, g: &ModelGeometry, index: bool) -> ConversationKv {
        let mut k = ConversationKv {
            format: kv::kv_format(&self.st),
            cells: 12,
            page_size: 4,
            heads: g.n_head_kv,
            head_dim: g.head_dim,
            idx_dim: g.idx_key_dim,
            pooled_rows: if index { 3 } else { 0 },
            ..ConversationKv::default()
        };
        k.k.resize(self.data[0].len() / 8, 13);
        k.v = k.k.clone();
        k.k_scale.resize(self.data[2].len() / 8, 13);
        k.v_scale = k.k_scale.clone();
        k.pooled
            .resize((k.pooled_rows * g.idx_key_dim * 4) as usize, 13);
        k
    }
}

fn fixture_geometry(format: i32, experts: i64, zero_qsa: bool, _ple: bool) -> ModelGeometry {
    let _ = format;
    ModelGeometry {
        n_layers: if zero_qsa { 3 } else { 8 },
        n_expert: experts,
        ssm_state_size: 2,
        ssm_v_heads: 2,
        ssm_conv_channels: 8,
        n_head_kv: 1,
        head_dim: 64,
        idx_key_dim: 8,
        ..ModelGeometry::default()
    }
}

/// One fixture run: the same program, the same order, the same lines.
struct World {
    g: ModelGeometry,
    format: i32,
    experts: i64,
    zero_qsa: bool,
    ple: bool,
    z: ConversationStateSizes,
    first: Pools,
    last: Pools,
    draft: Pools,
    layers: Vec<QsaState>,
    backend: Backend,
    image: SavedConversation,
    out: Vec<String>,
}

impl World {
    fn new(format: i32, experts: i64, zero_qsa: bool, ple: bool) -> Self {
        let g = fixture_geometry(format, experts, zero_qsa, ple);
        let mut w = Self {
            g,
            format,
            experts,
            zero_qsa,
            ple,
            z: ConversationStateSizes::default(),
            first: Pools::new(&g, format),
            last: Pools::new(&g, format),
            draft: Pools::new(&g, format),
            layers: Vec::new(),
            backend: Backend::new(),
            image: SavedConversation::default(),
            out: Vec::new(),
        };
        w.layers = vec![w.first.st, w.last.st];
        w
    }

    fn session(&self, backend: &Backend) -> SessionState {
        SessionState {
            max_cells: 96,
            gdn_state: backend.buffers.contains_key("gdn"),
            ple_hist: self.ple && backend.buffers.contains_key("ple"),
            layer_hi: self.g.n_layers,
            gdn_alloc: self.g.n_gdn_layers(),
            qsa_alloc: self.g.n_qsa_layers(),
            ple_prev: backend.ple_prev,
            qsa_states: if self.zero_qsa {
                Vec::new()
            } else {
                self.layers.clone()
            },
            ..SessionState::default()
        }
    }

    fn checkpoint(&self, ss: &SessionState, tokens: usize) -> ConversationCheckpoint {
        let mut c = ConversationCheckpoint::default();
        c.ids.resize(tokens, &0);
        for i in 0..tokens {
            c.ids.as_mut_slice()[i] = i as i32 + 1;
        }
        c.imgs.push(ConversationImageKey { start: 1, hash: 44 });
        c.gdn.resize(self.z.gdn, &13);
        c.ple.resize(if self.ple { self.z.ple } else { 0 }, &13);
        c.tails
            .resize(self.g.n_qsa_layers() as usize * self.z.tail, &13);
        c.dead
            .resize(self.g.n_qsa_layers() as usize * self.z.dead, &13);
        c.block_pos
            .resize(self.g.n_qsa_layers() as usize * self.z.block_pos, &13);
        let _ = ss;
        c
    }

    /// One observation line, after any transfer lines the harness would have
    /// printed first: the device log is interleaved, exactly as the C++ prints it.
    /// The restore writes `SessionState::ple_prev`; the fingerprint hashes it, so
    /// the test mirrors the session field into the backend after every call that
    /// can reach it.
    fn sync_ple_prev(&mut self, ss: &SessionState) {
        self.backend.ple_prev = ss.ple_prev;
    }

    fn line(&mut self, s: String) {
        self.out.extend(std::mem::take(&mut self.backend.log));
        self.out.push(format!("CORPUS|{s}"));
    }

    fn sizes(&mut self, geo: &ModelGeometry, tag: &str) {
        let mut z = ConversationStateSizes::default();
        let mut err = String::new();
        let ok = cs::conversation_state_sizes(geo, &mut z, &mut err);
        self.line(format!(
            "SIZES|tag={tag}|ok={}|err={}|gdn={}|ple={}|tail={}|dead={}|block_pos={}",
            yesno(ok),
            esc(&err),
            z.gdn,
            z.ple,
            z.tail,
            z.dead,
            z.block_pos
        ));
        let ss = self.session(&self.backend);
        let mut z3 = ConversationStateSizes::default();
        let mut err2 = String::new();
        let ok2 = cs::conversation_session_sizes(geo, &ss, &mut z3, &mut err2);
        self.line(format!(
            "SESSION_SIZES|tag={tag}|ok={}|err={}|gdn={}|ple={}|tail={}|dead={}|block_pos={}",
            yesno(ok2),
            esc(&err2),
            z3.gdn,
            z3.ple,
            z3.tail,
            z3.dead,
            z3.block_pos
        ));
    }

    fn reject(&mut self, label: &str, mutate: impl FnOnce(&mut SavedConversation)) {
        let mut bad = self.image.clone();
        mutate(&mut bad);
        let mut ss = self.session(&self.backend);
        let draft = self.draft.st;
        let mut err = String::new();
        let r = cs::snapshot_restore(&bad, &mut ss, &self.g, &draft, &mut self.backend, &mut err);
        self.sync_ple_prev(&ss);
        let fp = self.backend.state_fingerprint();
        self.line(format!(
            "REJECT|label={label}|restore={}|err={}|fingerprint={}",
            r.corpus_name(),
            esc(&err),
            fp
        ));
    }

    fn run(&mut self) {
        self.line(format!(
            "FIX|format={}|experts={}|zero_qsa={}|ple={}",
            self.format,
            self.experts,
            yesno(self.zero_qsa),
            yesno(self.ple)
        ));
        let mut err = String::new();
        let ok = cs::conversation_state_sizes(&self.g, &mut self.z, &mut err);
        self.line(format!(
            "SIZES|tag=fixture|ok={}|err={}|gdn={}|ple={}|tail={}|dead={}|block_pos={}",
            yesno(ok),
            esc(&err),
            self.z.gdn,
            self.z.ple,
            self.z.tail,
            self.z.dead,
            self.z.block_pos
        ));

        // the fixture's buffers, registered in the harness's fingerprint order
        self.backend.reg_ordered("gdn", vec![0xa5; self.z.gdn]);
        self.backend
            .reg_ordered("ple", vec![0xa5; if self.ple { self.z.ple } else { 0 }]);
        let names = [
            "k",
            "v",
            "k_scale",
            "v_scale",
            "idx_pooled",
            "idx_tail",
            "idx_dead",
            "idx_block_pos",
        ];
        for (i, pool) in [&self.first, &self.last, &self.draft].iter().enumerate() {
            let layer = ["L0", "L1", "DRAFT"][i];
            for (j, n) in names.iter().enumerate() {
                let label = format!("{layer}:{n}");
                self.backend.order.push(label.clone());
                if !pool.data[j].is_empty() {
                    self.backend.buffers.insert(label, pool.data[j].clone());
                }
            }
        }

        let ss = self.session(&self.backend);
        self.image = SavedConversation {
            geometry: strata_core::conversation::geometry_key(&self.g),
            layer_hi: self.g.n_layers,
            ..SavedConversation::default()
        };
        self.image.live = self.checkpoint(&ss, 9);
        self.image.checkpoints.push(self.checkpoint(&ss, 5));
        self.image.kv.reserve(self.g.n_qsa_layers() as usize + 1);
        if !self.zero_qsa {
            self.image.kv.push(self.first.image(&self.g, true));
            self.image.kv.push(self.last.image(&self.g, true));
        }
        self.image.kv.push(self.draft.image(&self.g, false));

        let ok = cs::snapshot_validate(&self.image, &ss, &self.g, &self.draft.st, &mut err);
        self.line(format!(
            "VALIDATE|ok={}|err={}|image_bytes={}",
            yesno(ok),
            esc(&err),
            self.image.bytes()
        ));
        let ids = self.image.live.ids.clone();
        let imgs = self.image.live.imgs.clone();
        let checkpoints = self.image.checkpoints.clone();
        let view = ConversationView {
            ids: &ids,
            images: &imgs,
            checkpoints: &checkpoints,
            cvec: true,
        };
        let mut estimate = 0usize;
        let ok = cs::snapshot_bytes(&view, &ss, &self.g, &self.draft.st, &mut estimate, &mut err);
        self.line(format!(
            "ESTIMATE|ok={}|err={}|bytes={}|image_bytes={}",
            yesno(ok),
            esc(&err),
            estimate,
            self.image.bytes()
        ));
        let mut capture_estimate = 0usize;
        let ok = cs::snapshot_capture_bytes(
            &ConversationKvReuse::default(),
            &view,
            &ss,
            &self.g,
            &self.draft.st,
            &mut capture_estimate,
            &mut err,
        );
        self.line(format!(
            "CAPTURE_ESTIMATE|ok={}|err={}|bytes={}",
            yesno(ok),
            esc(&err),
            capture_estimate
        ));

        let ok = cs::checkpoint_validate(&self.image.live, &ss, &self.g, &mut err);
        self.line(format!(
            "CHECKPOINT_VALIDATE|ok={}|err={}",
            yesno(ok),
            esc(&err)
        ));

        if !self.zero_qsa {
            let stage = SessionState {
                layer_lo: 4,
                gdn_alloc: 3,
                qsa_ord0: 1,
                qsa_alloc: 1,
                ..ss.clone()
            };
            let mut zs = ConversationStateSizes::default();
            let ok = cs::conversation_session_sizes(&self.g, &stage, &mut zs, &mut err);
            self.line(format!(
                "CARVE|ok={}|err={}|gdn={}|tail={}|dead={}|block_pos={}",
                yesno(ok),
                esc(&err),
                zs.gdn,
                zs.tail,
                zs.dead,
                zs.block_pos
            ));
            let ok = cs::checkpoint_validate(&self.image.live, &stage, &self.g, &mut err);
            self.line(format!(
                "CARVE_CHECKPOINT|whole_rejected={}|err={}",
                yesno(!ok),
                esc(&err)
            ));
            let mut part = self.image.live.clone();
            part.gdn.resize(zs.gdn, &0);
            part.tails.resize(zs.tail, &0);
            part.dead.resize(zs.dead, &0);
            part.block_pos.resize(zs.block_pos, &0);
            let ok = cs::checkpoint_validate(&part, &stage, &self.g, &mut err);
            self.line(format!(
                "CARVE_CHECKPOINT|part_ok={}|err={}",
                yesno(ok),
                esc(&err)
            ));
            let ok = cs::snapshot_validate(&self.image, &stage, &self.g, &self.draft.st, &mut err);
            self.line(format!(
                "CARVE_SNAPSHOT|rejected={}|err={}",
                yesno(!ok),
                esc(&err)
            ));
            let stage2 = SessionState {
                qsa_alloc: 2,
                ..stage.clone()
            };
            let mut junk = ConversationStateSizes::default();
            let ok = cs::conversation_session_sizes(&self.g, &stage2, &mut junk, &mut err);
            self.line(format!("CARVE|qsa_over_ok={}|err={}", yesno(ok), esc(&err)));
            let stage3 = SessionState {
                qsa_alloc: 1,
                gdn_alloc: 7,
                ..stage.clone()
            };
            let ok = cs::conversation_session_sizes(&self.g, &stage3, &mut junk, &mut err);
            self.line(format!("CARVE|gdn_over_ok={}|err={}", yesno(ok), esc(&err)));
        }

        self.reject("late_draft_corruption", |s| {
            s.kv.last_mut().unwrap().k.pop_back()
        });
        self.reject("expert_geometry", |s| s.geometry[16] = 128);
        self.reject("layer_split_live", |s| {
            s.live.stage_parts.push(Default::default())
        });
        self.reject("layer_split_checkpoint", |s| {
            s.checkpoints.as_mut_slice()[0]
                .stage_parts
                .push(Default::default())
        });
        self.reject("missing_kv_layer", |s| s.kv.pop());
        self.reject("bad_live_recurrence", |s| s.live.gdn.pop());
        self.reject("bad_retained_checkpoint", |s| {
            s.checkpoints.last_mut().unwrap().gdn.pop()
        });
        self.reject("foreign_checkpoint_prefix", |s| {
            s.checkpoints.as_mut_slice()[0].ids.as_mut_slice()[0] = 99
        });
        self.reject("foreign_checkpoint_image", |s| {
            s.checkpoints.as_mut_slice()[0].imgs.as_mut_slice()[0].hash += 1
        });
        self.reject("negative_image_position", |s| {
            s.live.imgs.as_mut_slice()[0].start = -1
        });
        self.reject("negative_live_token", |s| s.live.ids.as_mut_slice()[0] = -1);
        self.reject("ple_mismatch", |s| s.live.ple.push(0));
        self.reject("tail_mismatch", |s| s.live.tails.push(0));
        if !self.zero_qsa {
            self.reject("indexer_spare_corruption", |s| {
                s.kv.as_mut_slice()[1].pooled.pop_back()
            });
            self.reject("missing_spare_key", |s| s.live.dead.pop());
            self.reject("missing_indexer_metadata", |s| s.live.block_pos.pop());
            let ss2 = SessionState {
                qsa_states: Vec::new(),
                ..self.session(&self.backend)
            };
            let mut e2 = String::new();
            let ok = cs::snapshot_validate(&self.image, &ss2, &self.g, &self.draft.st, &mut e2);
            self.line(format!(
                "MISSING_QSA|rejected={}|err={}",
                yesno(!ok),
                esc(&e2)
            ));
            self.reject("other_layer_range", |s| s.layer_lo = 4);
        }

        // geometry arithmetic guards
        let bad = ModelGeometry {
            ssm_state_size: i64::MAX,
            ..self.g
        };
        self.sizes(&bad, "overflow");
        let bad = ModelGeometry {
            qsa_interval: 0,
            ..self.g
        };
        self.sizes(&bad, "zero_interval");
        let bad = ModelGeometry {
            n_head_kv: i64::MAX,
            ..self.g
        };
        self.line(format!(
            "KV_BYTES|overflow={}",
            kv::kv_bytes(&kv::Extent::new(
                Layer::Draft,
                &self.draft.st,
                &bad,
                9,
                false
            ))
        ));
        let mut bad = self.g;
        bad.head_dim = i64::MAX;
        self.line(format!(
            "KV_BYTES|head_dim_overflow={}",
            kv::kv_bytes(&kv::Extent::new(
                Layer::Draft,
                &self.draft.st,
                &bad,
                9,
                false
            ))
        ));

        self.transfers();
        self.verify_section();
        self.save_section();
    }

    fn transfers(&mut self) {
        self.xfer("fail_sync_pre", 0, 1);
        self.xfer("fail_copy_first", 1, 0);
        self.xfer("fail_copy_partial", 2, 0);
        self.xfer("fail_sync_post", 0, 2);
        self.backend.reset_state();
        let mut ss = self.session(&self.backend);
        let draft = self.draft.st;
        let mut err = String::new();
        let r = cs::snapshot_restore(
            &self.image,
            &mut ss,
            &self.g,
            &draft,
            &mut self.backend,
            &mut err,
        );
        self.sync_ple_prev(&ss);
        let gdn = self.backend.buffers.get("gdn").cloned().unwrap_or_default();
        let history = self.backend.buffers.get("ple").cloned().unwrap_or_default();
        let gdn_match = gdn == self.image.live.gdn.as_slice();
        let ple_match = history == self.image.live.ple.as_slice();
        let fp = self.backend.state_fingerprint();
        self.line(format!(
            "XFER|tag=success|restore={}|err={}|copies={}|gdn_match={}|ple_match={}|ple_prev={},{}|fingerprint={}",
            r.corpus_name(),
            esc(&err),
            self.backend.copy_calls,
            yesno(gdn_match),
            yesno(ple_match),
            self.backend.ple_prev[0],
            self.backend.ple_prev[1],
            fp
        ));
    }

    fn xfer(&mut self, tag: &str, fc: usize, fs: usize) {
        self.backend.reset_state();
        self.backend.fail_copy = fc;
        self.backend.fail_sync = fs;
        let mut ss = self.session(&self.backend);
        let draft = self.draft.st;
        let mut err = String::new();
        let r = cs::snapshot_restore(
            &self.image,
            &mut ss,
            &self.g,
            &draft,
            &mut self.backend,
            &mut err,
        );
        self.sync_ple_prev(&ss);
        let fp = self.backend.state_fingerprint();
        self.line(format!(
            "XFER|tag={tag}|restore={}|err={}|copies={}|syncs={}|fingerprint={}",
            r.corpus_name(),
            esc(&err),
            self.backend.copy_calls,
            self.backend.sync_calls,
            fp
        ));
    }

    fn verify_section(&mut self) {
        self.backend.reset_state();
        let ss = self.session(&self.backend);
        let draft = self.draft.st;
        let mut err = String::new();
        let rr = cs::snapshot_restore(
            &self.image,
            &mut { ss },
            &self.g,
            &draft,
            &mut self.backend,
            &mut err,
        );
        if rr != ConversationRestore::Restored {
            self.line(format!("VERIFY|restore_failed={}", esc(&err)));
        }
        let last = self.image.kv.len() - 1;
        let image = self.image.kv.get(last).expect("draft image").clone();
        let mut fp = 0u64;
        let ok = kv::kv_verify(
            &image,
            &mut self.backend,
            &kv::Extent::new(Layer::Draft, &self.draft.st, &self.g, 9, false),
            &mut fp,
            &mut err,
        );
        self.line(format!(
            "VERIFY|tag=draft|ok={}|err={}|fingerprint={}",
            yesno(ok),
            esc(&err),
            fp
        ));
        self.backend.buffers.get_mut("DRAFT:k").expect("registered")[0] ^= 1;
        let mut fp2 = 0u64;
        let ok = kv::kv_verify(
            &image,
            &mut self.backend,
            &kv::Extent::new(Layer::Draft, &self.draft.st, &self.g, 9, false),
            &mut fp2,
            &mut err,
        );
        self.line(format!(
            "VERIFY|tag=corrupt|ok={}|err={}|fingerprint={}",
            yesno(ok),
            esc(&err),
            fp2
        ));
        self.backend.buffers.get_mut("DRAFT:k").expect("registered")[0] ^= 1;
        self.backend.fail_copy = self.backend.copy_calls + 1;
        let mut fp3 = 0u64;
        let ok = kv::kv_verify(
            &image,
            &mut self.backend,
            &kv::Extent::new(Layer::Draft, &self.draft.st, &self.g, 9, false),
            &mut fp3,
            &mut err,
        );
        self.line(format!(
            "VERIFY|tag=copy_failed|ok={}|err={}|fingerprint={}",
            yesno(ok),
            esc(&err),
            fp3
        ));
    }

    fn save_section(&mut self) {
        self.backend.reset_state();
        let ids = self.image.live.ids.clone();
        let imgs = self.image.live.imgs.clone();
        let checkpoints = self.image.checkpoints.clone();
        let ss = self.session(&self.backend);
        let draft = self.draft.st;
        let mut err = String::new();
        let mut full = SavedConversation::default();
        let mut reuse = ConversationKvReuse::default();
        let saved = cs::snapshot_save(
            &mut full,
            &ConversationView {
                ids: &ids,
                images: &imgs,
                checkpoints: &checkpoints,
                cvec: true,
            },
            &ss,
            &self.g,
            &draft,
            &mut self.backend,
            &mut err,
            &mut reuse,
            None,
        );
        let full_copies = self.backend.copied_bytes;
        self.line(format!(
            "SAVE_FULL|ok={}|err={}|copies={}|bytes={}",
            yesno(saved),
            esc(&err),
            full_copies,
            full.bytes()
        ));
        for i in 0..full.kv.len() {
            let layer = full.kv.get(i).expect("layer");
            self.line(format!(
                "SAVE_FULL_KV|layer={i}|format={}|cells={}|heads={}|head_dim={}|page_size={}|pooled_rows={}|idx_dim={}|k={}|v={}|ks={}|vs={}|pooled={}",
                layer.format,
                layer.cells,
                layer.heads,
                layer.head_dim,
                layer.page_size,
                layer.pooled_rows,
                layer.idx_dim,
                fnv(&flat(&layer.k)),
                fnv(&flat(&layer.v)),
                fnv(&flat(&layer.k_scale)),
                fnv(&flat(&layer.v_scale)),
                fnv(&flat(&layer.pooled))
            ));
        }
        self.line(format!(
            "SAVE_FULL_STATE|gdn={}|ple={}|tails={}|dead={}|block_pos={}",
            fnv(full.live.gdn.as_slice()),
            fnv(full.live.ple.as_slice()),
            fnv(full.live.tails.as_slice()),
            fnv(full.live.dead.as_slice()),
            fnv(full.live.block_pos.as_slice())
        ));

        let mut reuse = ConversationKvReuse {
            kv: clone_with(&full.kv, SIZEOF_KV),
            captured_tokens: 9,
            unchanged_tokens: 9,
            stages: CppVec::with_unit(SIZEOF_KV_REUSE),
        };
        let mut peak = 0usize;
        let view = ConversationView {
            ids: &ids,
            images: &imgs,
            checkpoints: &checkpoints,
            cvec: true,
        };
        let admitted =
            cs::snapshot_capture_bytes(&reuse, &view, &ss, &self.g, &draft, &mut peak, &mut err);
        self.line(format!(
            "REUSE_ADMISSION|ok={}|err={}|peak={}|reuse_bytes={}",
            yesno(admitted),
            esc(&err),
            peak,
            reuse.bytes()
        ));

        self.backend.copied_bytes = 0;
        let mut reused = 0usize;
        let mut incremental = SavedConversation::default();
        let saved_inc = cs::snapshot_save(
            &mut incremental,
            &view,
            &ss,
            &self.g,
            &draft,
            &mut self.backend,
            &mut err,
            &mut reuse,
            Some(&mut reused),
        );
        self.line(format!(
            "SAVE_INCREMENTAL|ok={}|err={}|reused={}|copies={}|sum={}|full_copies={}|within_peak={}|bytes={}",
            yesno(saved_inc),
            esc(&err),
            reused,
            self.backend.copied_bytes,
            self.backend.copied_bytes + reused,
            full_copies,
            yesno(incremental.bytes() <= peak),
            incremental.bytes()
        ));
        let mut agree = incremental.kv.len() == full.kv.len();
        let mut i = 0;
        while agree && i < full.kv.len() {
            let a = incremental.kv.get(i).expect("layer");
            let b = full.kv.get(i).expect("layer");
            agree = a.k.same_contents(&b.k)
                && a.v.same_contents(&b.v)
                && a.k_scale.same_contents(&b.k_scale)
                && a.v_scale.same_contents(&b.v_scale)
                && a.pooled.same_contents(&b.pooled);
            i += 1;
        }
        self.line(format!(
            "SAVE_INCREMENTAL|agree={}|gdn_equal={}",
            yesno(agree),
            yesno(incremental.live.gdn.as_slice() == full.live.gdn.as_slice())
        ));

        let mut bad_reuse = ConversationKvReuse {
            kv: clone_with(&full.kv, SIZEOF_KV),
            captured_tokens: 9,
            unchanged_tokens: 9,
            stages: CppVec::with_unit(SIZEOF_KV_REUSE),
        };
        bad_reuse.kv.last_mut().unwrap().k.pop_back();
        self.backend.copy_calls = 0;
        self.backend.sync_calls = 0;
        let mut peak2 = 0usize;
        let admitted = cs::snapshot_capture_bytes(
            &bad_reuse, &view, &ss, &self.g, &draft, &mut peak2, &mut err,
        );
        self.line(format!(
            "REUSE_ADMISSION|malformed_ok={}|err={}|copies={}|syncs={}",
            yesno(admitted),
            esc(&err),
            self.backend.copy_calls,
            self.backend.sync_calls
        ));

        let mut fail_reuse = ConversationKvReuse {
            kv: clone_with(&full.kv, SIZEOF_KV),
            captured_tokens: 9,
            unchanged_tokens: 9,
            stages: CppVec::with_unit(SIZEOF_KV_REUSE),
        };
        self.backend.fail_copy = self.backend.copy_calls + 1;
        let mut unpublished = SavedConversation::default();
        let saved_fail = cs::snapshot_save(
            &mut unpublished,
            &view,
            &ss,
            &self.g,
            &draft,
            &mut self.backend,
            &mut err,
            &mut fail_reuse,
            None,
        );
        self.line(format!(
            "SAVE_INCREMENTAL|copy_failure_ok={}|err={}|kv_empty={}|ids_empty={}",
            yesno(saved_fail),
            esc(&err),
            yesno(unpublished.kv.is_empty()),
            yesno(unpublished.live.ids.is_empty())
        ));
    }
}

fn flat(b: &Buf) -> Vec<u8> {
    let mut v = vec![0u8; b.size()];
    assert!(b.read(&mut v, 0, b.size()));
    v
}

/// `ConversationKvReuse reuse{full.kv, ...}`: copy-construction allocates exactly
/// the source's SIZE, which is a different capacity from what growing to it gives —
/// and `bytes()` counts the capacity, so the difference is a real admission number.
fn clone_with(src: &CppVec<ConversationKv>, unit: usize) -> CppVec<ConversationKv> {
    let mut out = CppVec::with_unit(unit);
    out.reserve(src.len());
    for item in src.iter() {
        out.push(item.clone());
    }
    out
}

#[test]
fn conversation_corpus_replay() {
    let want = golden();
    let mut got: Vec<String> = Vec::new();
    for &format in &[0i32, 1, 2] {
        for &experts in &[256i64, 512] {
            for &zero_qsa in &[false, true] {
                for &ple in &[false, true] {
                    let mut w = World::new(format, experts, zero_qsa, ple);
                    let lines = {
                        w.run();
                        std::mem::take(&mut w.out)
                    };
                    got.extend(lines);
                }
            }
        }
    }
    // the BUFFER section is covered by its own test; compare everything else
    let want: Vec<String> = want
        .into_iter()
        .filter(|l| !l.starts_with("CORPUS|BUFFER|"))
        .collect();
    assert_eq!(got.len(), want.len(), "replay line count");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "replay line {i}");
    }
}

#[test]
fn buffer_segmentation_matches_the_cxx_corpus() {
    const SEG: usize = Buf::SEGMENT_BYTES;
    let want: Vec<String> = golden()
        .into_iter()
        .filter(|l| l.starts_with("CORPUS|BUFFER|"))
        .collect();
    let mut got: Vec<String> = Vec::new();
    let probe = |tag: &str, b: &Buf, next: usize, got: &mut Vec<String>| {
        got.push(format!(
            "CORPUS|BUFFER|tag={tag}|size={}|bytes={}|peak={}|next={}|fill={}",
            b.size(),
            b.bytes(),
            b.allocation_peak(next),
            next,
            fnv(&flat(b))
        ));
    };

    // a fresh capture allocates exactly its payload
    for &n in &[1usize, 1024, 65536, SEG - 1, SEG, SEG + 1, 2 * SEG + 7] {
        let mut fresh = Buf::new();
        fresh.resize(n, 0x5a);
        probe("fresh", &fresh, n, &mut got);
    }
    // later appends grow the last segment geometrically, bounded by 16 MiB
    {
        let mut g = Buf::new();
        g.resize(4096, 0x11);
        probe("append_start", &g, 4096, &mut got);
        for _ in 0..12 {
            let next = g.size() * 2 + 1000;
            g.resize(next, 0x22);
            probe("append", &g, g.size(), &mut got);
        }
        // and shrinking drops whole segments before trimming the last
        for &n in &[g.size() - 1, g.size() / 3, 1usize, 0usize, 20 * SEG] {
            g.resize(n, 0x33);
            probe("shrink", &g, n, &mut got);
        }
    }
    // a failed admission must saturate, not wrap
    {
        let mut h = Buf::new();
        h.resize(16, 0x44);
        got.push(format!(
            "CORPUS|BUFFER|tag=saturate|peak_max={}",
            u8::from(h.allocation_peak(usize::MAX) == usize::MAX)
        ));
        let mut window = [0u8; 32];
        got.push(format!(
            "CORPUS|BUFFER|tag=visit|in_range={}|past_end={}|zero={}",
            u8::from(h.read(&mut window, 4, 8)),
            u8::from(h.read(&mut window, 12, 8)),
            u8::from(h.read(&mut window, 0, 0)),
        ));
        let mut split = Buf::new();
        split.resize(SEG, 0x66);
        split.resize(SEG + 4096, 0x66);
        let mut single = Buf::new();
        single.resize(SEG + 4096, 0x66);
        got.push(format!(
            "CORPUS|BUFFER|tag=segmentation|equal={}|bytes_differ={}",
            u8::from(split.same_contents(&single)),
            u8::from(split.bytes() != single.bytes()),
        ));
        let mut shorter = Buf::new();
        shorter.resize(SEG + 4095, 0x66);
        got.push(format!(
            "CORPUS|BUFFER|tag=segmentation|shorter_equal={}",
            u8::from(split.same_contents(&shorter))
        ));
        let mut pop = Buf::new();
        pop.resize(SEG + 1, 0x77);
        pop.pop_back();
        probe("pop_to_boundary", &pop, pop.size(), &mut got);
    }
    assert_eq!(got.len(), want.len(), "BUFFER line count");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "BUFFER line {i}");
    }
}

#[test]
fn segment_directory_units_match() {
    assert_eq!(std::mem::size_of::<Vec<u8>>(), 24);
    let mut b = Buf::new();
    b.resize(1, 0);
    assert_eq!(b.bytes(), 1 + 24);
    assert_eq!(SIZEOF_CHECKPOINT, 200);
    assert_eq!(SIZEOF_KV, 216);
    assert_eq!(SIZEOF_KV_REUSE, 64);
    assert_eq!(SIZEOF_SAVED, 440);
}
