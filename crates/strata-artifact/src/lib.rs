//! GGUF v3 reader: header, metadata KV (all value types incl. arrays), tensor
//! directory, shard resolution, block geometry, and the qwen4exp architecture
//! guard. Ported from the header-only C++ reader in
//! `Strata/include/strata/artifact/`.
//!
//! Deliberate difference from the C++: the C++ mmaps the file and hands out
//! `tensor_data()` pointers. This reader parses the header with bounded
//! sequential reads and does not map the data section — expert/tensor payload
//! reads stay with the direct-IO loader (`strata-device`), which is how the
//! engine already fetches experts anyway. `data_start()` and `offset` give the
//! file offsets that loader needs.
//! `unsafe_code` is denied workspace-wide (see [workspace.lints.rust]).

pub mod gguf;
pub mod split;

pub use gguf::{
    block_geometry, check_architecture, ggml_type_name, tensor_payload_bytes, GgufFile, GgufModel,
    MetaType, MetaValue, Qwen4ExpGuard, TensorInfo,
};
pub use split::gguf_split_paths;
