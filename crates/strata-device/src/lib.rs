//! The FFI boundary between the Rust engine core and the C++/CUDA layer.
//!
//! This crate owns the *contract*: the type layouts and the function list the C++
//! `libstrata_kernels` must satisfy. The matching C header is checked in at
//! `include/strata_kernels.h`; the C++ shim includes it, and this crate's layout
//! tests pin the same sizes, so the two sides cannot drift (a mismatch is a
//! compile error on one side or a failing test on the other).
//!
//! Rules for this boundary (from the migration design):
//!   * plain `#[repr(C)]` data only; no C++ types cross it
//!   * pointers are non-owning `*const/_mut c_void` buffers + lengths, never
//!     `Vec`/`String`
//!   * every call returns an `i32` status (`StrataStatus`); details go to a
//!     caller-supplied error buffer, so no C++ exception crosses the ABI
//!   * streams/handles are opaque `*mut c_void` created by the device layer
// `unsafe_code` is denied workspace-wide ([workspace.lints.rust] in Cargo.toml).
// `abi` is a pure data contract and stays clean; `shim` re-allows it locally —
// dlopen and every call through a vtable slot are inherently unsafe — and is the
// workspace's non-test unsafe lives in those two files. The types they hand
// out are safe to use.

pub mod abi;
pub mod shim;
// the vtable calls through the snapshot slots are unsafe, like shim's
#[allow(unsafe_code)]
pub mod snapshot;

pub use abi::*;
pub use shim::{
    CapturedGraph, DeviceBuf, DeviceFile, Event, Pinned, ReplayableGraph, Shim, Stream,
};
pub use snapshot::{
    DeviceRegion, IndexerBuffers, LayerKv, LayerState, Residency, RunningTarget, SessionDevice,
    SnapshotDevice,
};
