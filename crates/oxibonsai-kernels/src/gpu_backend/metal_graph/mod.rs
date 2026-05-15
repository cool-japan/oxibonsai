//! Direct Metal dispatch engine for OxiBonsai FFN pipeline.
//!
//! Bypasses scirs2-core's abstraction layer and encodes all FFN operations
//! into a single command buffer with a single compute encoder, following the
//! llama.cpp architecture pattern.
//!
//! # Architecture
//!
//! - Single `metal::Device` (system default, singleton)
//! - Dedicated `metal::CommandQueue` per graph
//! - Pre-compiled compute pipeline states from concatenated MSL sources
//! - Lazily pre-allocated intermediate GPU buffers (shared mode + hazard tracking)
//!
//! # Buffer hazard tracking
//!
//! All CPU-accessible buffers use `MTLResourceOptions::StorageModeShared` with default
//! (tracked) hazard tracking mode.  With a non-concurrent compute encoder,
//! Metal automatically inserts memory barriers for read-after-write
//! dependencies, so explicit `memory_barrier_with_resources` calls are
//! not required.
//!
//! # Module structure
//!
//! Phase 30A split the monolithic `metal_graph.rs` (1948 lines) into focused
//! sub-modules; all external `super::metal_graph::*` access paths are
//! preserved through the re-exports below.
//!
//!   - [`error`]: [`MetalGraphError`] enum and [`MetalWeightHandle`] handle type.
//!   - [`reformat`]: Q1/TQ2 weight block AoS→SoA reformatters.
//!   - [`pipelines`]: MSL compilation, metallib caching, and `MetalPipelines`.
//!   - [`buffers`]: Intermediate buffer set plus crate-shared allocation,
//!     upload/download, and dispatch helpers.
//!   - [`graph`]: [`MetalGraph`] struct, weight cache, single GEMV dispatch,
//!     and the fused FFN phase.
//!   - [`tests`]: Compile- and runtime correctness tests (no-op on non-Metal hosts).

#![cfg(all(feature = "metal", target_os = "macos"))]

mod buffers;
mod error;
mod graph;
mod pipelines;
mod reformat;

pub use error::{MetalGraphError, MetalWeightHandle};
pub use graph::MetalGraph;

// Crate-internal helpers used by sibling modules
// (`metal_dispatch`, `metal_full_layer`, `metal_prefill`, `metal_fp8_*`).
pub(crate) use buffers::{alloc_buf, div_ceil, download_f32, set_scalar, upload_f32};

#[cfg(test)]
mod tests;
