//! Speculation control for the Strata engine: which drafter proposes this step
//! ([`Controller`]), whether a lookup window beats the MTP's ([`DraftPolicy`]),
//! and the suffix-lookup drafter itself ([`SuffixDrafter`]).
//!
//! Ported 1:1 from the C++ in `Strata/src/spec` and `Strata/include/strata/spec`.
//! The C++ unit tests are ported as `#[cfg(test)]` modules and must agree bit-for-bit
//! on the decisions.
// `unsafe_code` is denied workspace-wide ([workspace.lints.rust] in Cargo.toml).

pub mod controller;
pub mod draft_policy;
pub mod suffix_drafter;

#[cfg(test)]
mod parity;

pub use controller::{Choice, Controller, CostModel, Source, K_MAX};
pub use draft_policy::DraftPolicy;
pub use suffix_drafter::SuffixDrafter;
