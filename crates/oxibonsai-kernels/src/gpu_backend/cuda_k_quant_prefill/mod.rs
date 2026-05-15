//! Batch prefill (GEMM) dispatch for OxiBonsai — K-quant CUDA backend.
//!
//! This module provides the batch prefill path for K-quant quantised models
//! (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K). It mirrors the architecture of
//! [`cuda_q_std_prefill`] for Q4_0/Q8_0, but dispatches across 6 formats
//! via the [`KQuantFormat`] enum.
//!
//! # Architecture
//!
//! - [`CudaKQuantPrefillModules`]: Compiled CUDA functions for the 18 batch GEMM kernels.
//! - [`CudaKQuantPrefillLayerParams`]: Per-layer weight handles and raw AoS bytes.
//! - [`try_cuda_prefill_k_quant`]: Public entry point for K-quant batch prefill.
//!
//! # Module structure
//!
//! Phase 29 split the monolithic `cuda_k_quant_prefill.rs` (1941 lines) into
//! focused sub-modules. All external access paths via
//! `super::cuda_k_quant_prefill::*` continue to work via the re-exports below.
//!
//!   - [`state`]: types, singleton state, init, buffer/KV-cache/logits acquisition.
//!   - [`launchers`]: 18 `unsafe fn` launchers (gemm + gemm_residual +
//!     fused_gate_up_swiglu × 6 K-quant formats).
//!   - [`encode`]: [`encode_k_quant_ffn_phase`] + full-layer encoder with
//!     [`KQuantFormat`] dispatch.
//!   - [`try_api`]: public [`try_cuda_prefill_k_quant`] entry point.
//!
//! # Batch tensor layout
//!
//! All batched buffers use **column-major** layout: `buf[col * dim + element]`
//! where `col` is the batch/token index.
//!
//! # Weight layout
//!
//! K-quant weights stay in AoS layout as stored in GGUF.  QK_K = 256 weights
//! per super-block; `hidden_size` must be a multiple of 256.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

mod encode;
mod launchers;
mod state;
mod try_api;

pub use state::{
    init_k_quant_prefill_modules, CudaKQuantPrefillLayerParams, CudaKQuantPrefillModules,
    KQuantFormat,
};
pub use try_api::try_cuda_prefill_k_quant;

#[cfg(test)]
mod tests;
