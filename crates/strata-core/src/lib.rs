//! Engine-core pieces ported to Rust ahead of the FFI work: the host-side
//! arithmetic of coupled draft sampling and the verify-window penalty histories.
//! These are the pure functions the GPU paths must agree with; here they are the
//! contract, not a mirror of one.
// `unsafe_code` is denied workspace-wide ([workspace.lints.rust] in Cargo.toml).

pub mod coupled_draft;
pub mod sampler;
