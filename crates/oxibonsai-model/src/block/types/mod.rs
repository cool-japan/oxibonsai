//! Phase 33 split: `block::types` was a 1853-line monolith holding the
//! `TransformerBlock` definition, the three `forward*` entry points, all
//! GPU full-layer dispatch helpers, ~84 per-quant-family weight accessors,
//! and the private `ScratchBuffers` helper.  The new layout groups closely
//! related code into focused sub-modules while preserving the original
//! public surface (re-exported via `pub use types::*;` from `block/mod.rs`).
//!
//! Sub-modules:
//!   * `layer_stats`   — `LayerStats` struct + impl.
//!   * `block_def`     — `TransformerBlock<'a>` struct definition + `new`.
//!   * `upload`        — `upload_to_gpu` (one method, GPU memory mgmt).
//!   * `forward`       — single-token `forward` entry point.
//!   * `forward_stats` — single-token `forward_with_stats` entry point.
//!   * `forward_sw`    — sliding-window `forward_with_sliding_window`.
//!   * `helpers`       — `layer_idx` + (cfg-gated) full-layer GPU dispatchers.
//!   * `accessors`     — every public weight/handle/block accessor.
//!   * `scratch`       — private `ScratchBuffers` shared by the forward
//!     entry points.
//!
//! Each sub-file contributes its own `impl<'a> TransformerBlock<'a> { ... }`
//! block (Rust supports multiple inherent impl blocks for the same type
//! across separate files).

mod accessors;
mod block_def;
mod forward;
mod forward_stats;
mod forward_sw;
mod helpers;
mod layer_stats;
mod scratch;
mod upload;

pub use block_def::TransformerBlock;
pub use layer_stats::LayerStats;
