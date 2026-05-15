//! Prefill (batch) GPU dispatch for OxiBonsai — CUDA backend.
//!
//! Mirrors [`metal_prefill`] for Linux/Windows.  Handles batch processing of
//! multiple tokens during prompt prefill using GEMM kernels.
//!
//! # Architecture
//!
//! - [`CudaPrefillBuffers`]: Pre-allocated GPU buffers sized for `batch_size` tokens.
//! - [`CudaPrefillModules`]: Compiled CUDA functions for the 5 prefill kernels.
//! - `encode_prefill_ffn_phase`: Batched FFN pipeline (RMSNorm → gate+up+SwiGLU → down).
//! - `encode_prefill_layer`: One full prefill transformer layer.
//! - [`try_cuda_prefill`]: Public entry point mirroring `try_metal_full_forward_prefill`.
//!
//! # Module structure
//!
//! Phase 29 split the monolithic `cuda_prefill.rs` (1989 lines) into focused
//! sub-modules; all external `super::cuda_prefill::*` access paths are
//! preserved through the re-exports below.
//!
//!   - [`state`]: types ([`CudaPrefillBuffers`], [`CudaPrefillModules`]),
//!     singleton state, [`init_prefill_modules`], and buffer-acquisition
//!     helpers.
//!   - [`launchers`]: thin `unsafe fn` wrappers around `launch_builder()`
//!     for the 7 prefill kernels (4 Q1 + 3 TQ2).
//!   - [`encode_q1`]: Q1 (1-bit) FFN + full-layer encoders.
//!   - [`encode_ternary`]: TQ2 (ternary) FFN + full-layer encoders.
//!   - [`try_apis`]: public [`try_cuda_prefill`] / [`try_cuda_prefill_ternary`]
//!     entry points.
//!
//! # Batch tensor layout
//!
//! All batched buffers use **column-major** layout: `buf[col * dim + element]`
//! where `col` is the batch/token index.  This matches the Metal MSL kernels.
//!
//! # Attention in the prefill path
//!
//! We do not have a batched attention kernel; attention is processed sequentially
//! per token using the existing single-token attention kernels from `cuda_full_layer`.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

mod encode_q1;
mod encode_ternary;
mod launchers;
mod state;
mod try_apis;

pub use state::{init_prefill_modules, CudaPrefillBuffers, CudaPrefillModules};
pub use try_apis::{try_cuda_prefill, try_cuda_prefill_ternary};

#[cfg(test)]
mod tests;
