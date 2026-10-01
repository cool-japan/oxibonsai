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
//! - Objective-C autorelease pools around command-buffer lifetimes (see
//!   [`with_autorelease_pool`])
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
pub use graph::{
    default_prefill_budget, MetalDevice, MetalGraph, MetalPrefillPolicy, PrefillCostSnapshot,
    PrefillDeadlineScope, PrefillDecision, PrefillRoute, PrefillWorkShape, SessionScope,
    FUSED_BUDGET_FACTOR, FUSED_BUDGET_SLACK,
};

/// The head-free hidden-state prefill (the dense embedding pass, MET-05) and
/// the logits prefill's micro-batch size, nameable from outside the crate
/// through this public module (`metal_prefill` itself is private).
pub use crate::gpu_backend::metal_prefill::{
    try_metal_full_forward_prefill_hidden, try_metal_full_forward_prefill_hidden_cached,
    try_metal_full_forward_prefill_hidden_ternary, try_metal_full_forward_prefill_q1_cached,
    HiddenPrefillInput, HiddenPrefillShape, HIDDEN_PREFILL_MICRO_BATCH, PREFILL_LOGITS_MICRO_BATCH,
};

/// Run `f` inside a fresh Objective-C autorelease pool, drained when `f`
/// returns (or unwinds).
///
/// Metal methods outside the `new`/`alloc`/`copy` families that return an
/// object — `-[MTLCommandQueue commandBuffer]`,
/// `-[MTLCommandBuffer computeCommandEncoder]`, `-[MTLCommandBuffer error]` —
/// hand back autoreleased objects, as do the `NSString`s the `metal` crate
/// builds for names and labels, and a thread with no pool of its own keeps
/// them until it exits: about 1.8 KiB per command buffer and encoder, i.e.
/// per decoded token on a long-lived server thread. [`MetalGraph`]'s own
/// GEMV / GEMM / attention / FFN dispatches and the batched prefill runner
/// drain a pool per command buffer; a caller that drives whole forwards
/// through the Metal backend (a model's fused decode, prefill and
/// hidden-state passes) wraps each call in this as well, so nothing a
/// forward autoreleases outlives it whichever entry point it reaches.
pub fn with_autorelease_pool<T>(f: impl FnOnce() -> T) -> T {
    metal::objc::rc::autoreleasepool(f)
}

// Crate-internal helpers used by sibling modules
// (`metal_dispatch`, `metal_full_layer`, `metal_prefill`, `metal_fp8_*`).
pub(crate) use buffers::{
    alloc_buf, commit_and_wait, commit_and_wait_bounded, div_ceil, download_f32, set_scalar,
    upload_f32, wait_bounded,
};
pub(crate) use graph::effective_prefill_deadline;

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
mod tests_hidden;
#[cfg(test)]
mod tests_no_private_library;
#[cfg(test)]
mod tests_vae;
