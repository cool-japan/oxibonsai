//! Direct Metal dispatch engine for OxiBonsai FFN pipeline.
//!
//! Bypasses scirs2-core's abstraction layer and encodes all FFN operations
//! into a single command buffer with a single compute encoder, following the
//! llama.cpp architecture pattern.
//!
//! # Architecture
//!
//! - One process-shared [`MetalDevice`]: the system default `metal::Device`,
//!   the compiled pipelines and the weight cache (`MET-08`)
//! - One [`MetalGraph`] per **session**, each with its own
//!   `metal::CommandQueue` and lazily allocated workspace; a thread selects
//!   its session with [`MetalGraph::with_session`] / [`SessionScope`]
//! - Pre-compiled compute pipeline states from concatenated MSL sources — one
//!   combined metallib (plus its best-effort bf16 sidecar) that the kernel
//!   families resolve their pipelines from by name (`MET-10`: the K-quant,
//!   standard-quant and FP8 families no longer compile private libraries);
//!   the batched prefill-attention library (`metal_prefill/attention.rs`) is
//!   the one library still compiled separately
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
//!   - `error`: [`MetalGraphError`] enum and [`MetalWeightHandle`] handle type.
//!   - `reformat`: Q1/TQ2 weight block AoS→SoA reformatters.
//!   - `pipelines`: MSL compilation, metallib caching, and `MetalPipelines`.
//!   - `buffers`: Intermediate buffer set plus crate-shared allocation,
//!     upload/download, and dispatch helpers.
//!   - `graph`: [`MetalGraph`] struct, weight cache, single GEMV dispatch,
//!     and the fused FFN phase.
//!   - `tests` (+ `tests_gemv_tq2`, `tests_gemm_tq2`, `tests_gemm_f32`,
//!     `tests_vae`, `tests_dit_attention`): Compile- and runtime correctness
//!     tests, grouped by concern (no-op on non-Metal hosts).

#![cfg(all(feature = "metal", target_os = "macos"))]

mod buffers;
mod error;
mod graph;
mod pipelines;
mod reformat;
mod vae;

pub use error::{MetalGraphError, MetalWeightHandle};
pub use graph::{MetalDevice, MetalGraph, SessionScope};

// Crate-internal helpers used by sibling modules
// (`metal_dispatch`, `metal_full_layer`, `metal_prefill`, `metal_fp8_*`).
pub(crate) use buffers::{
    alloc_buf, commit_and_wait, div_ceil, download_f32, set_scalar, upload_f32,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_dit_attention;
#[cfg(test)]
mod tests_gemm_f32;
#[cfg(test)]
mod tests_gemm_tq2;
#[cfg(test)]
mod tests_gemv_tq2;
#[cfg(test)]
mod tests_no_private_library;
#[cfg(test)]
mod tests_vae;
