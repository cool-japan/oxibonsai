//! CUDA GPU forward-pass methods for `BonsaiModel`.
//!
//! Phase 32 split the monolithic `forward_cuda.rs` (1801 lines) into focused
//! sub-modules grouped by quant family.  Each sub-module contributes its own
//! `impl<'a> BonsaiModel<'a> { ... }` block (Rust supports multiple inherent
//! impl blocks across files for the same type).
//!
//! Sub-modules:
//!   - [`byte_helpers`]: free `#[repr(C)]` block → `&[u8]` zero-copy casts for
//!     every K-quant and standard-quant block format.
//!   - [`q1`]: 1-bit (Q1) helper builders plus all top-level dispatch entry
//!     points (`try_cuda_full_forward_inner`, `try_cuda_full_forward_with_lm_head`,
//!     `try_cuda_prefill_with_lm_head`, `try_cuda_prefill_verify`).  The
//!     dispatchers route to ternary / Q-std / FP8 / K-quant paths as needed.
//!   - [`ternary`]: TQ2 ternary helper builders and the dedicated ternary
//!     batch-prefill methods.
//!   - [`q_std`]: Q4_0 / Q8_0 helper builders and the dedicated Q-std
//!     batch-prefill methods.
//!   - [`k_quant`]: Q2K / Q3K / Q4K / Q5K / Q6K / Q8K helper builders and the
//!     K-quant batch-prefill methods.
//!
//! FP8 batch-prefill methods live in the sibling `forward_cuda_fp8` module
//! (`crates/oxibonsai-model/src/model/types/forward_cuda_fp8.rs`).

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

mod byte_helpers;
mod k_quant;
mod q1;
mod q_std;
mod ternary;
