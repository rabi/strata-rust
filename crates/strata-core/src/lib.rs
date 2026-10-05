//! Engine-core pieces ported to Rust ahead of the FFI work: the host-side
//! arithmetic of coupled draft sampling and the verify-window penalty histories.
//! These are the pure functions the GPU paths must agree with; here they are the
//! contract, not a mirror of one.
// `unsafe_code` is denied workspace-wide ([workspace.lints.rust] in Cargo.toml).

pub mod conv_buffer;
pub mod conv_vec;
pub mod conversation;
pub mod conversation_kv;
pub mod conversation_state;
pub mod coupled_draft;
pub mod expert_plan;
pub mod host_memory;
pub mod layout;
pub mod native_dense;
pub mod native_mm;
pub mod pinned;
pub mod sampler;
pub mod weights;

/// Strata's fingerprint seed and prime (`pinned.cu`'s fnv1a64). Shared by the
/// snapshot read-back check and every corpus fingerprint.
pub const FNV_OFFSET: u64 = 1_469_598_103_934_665_603;
pub const FNV_PRIME: u64 = 1_099_511_628_211;

/// fnv1a64 over a byte range, the fingerprint the snapshot contract reports.
pub fn fnv(bytes: &[u8]) -> u64 {
    fnv_seeded(bytes, FNV_OFFSET)
}

/// The same with an explicit seed - `pinned.cu`'s `fnv1a64(p, n, seed)`, which the
/// read plan chains per chunk.
#[must_use]
pub fn fnv_seeded(bytes: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}
