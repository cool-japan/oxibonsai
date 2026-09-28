//! `TransformerBlock<'a>` struct definition and `new` constructor.
//!
//! Other inherent methods on `TransformerBlock` live in sibling sub-modules
//! (Rust supports multiple `impl` blocks for the same type across files).

use crate::layers::linear::LinearLayer;
use crate::layers::rms_norm::RmsNorm;
use oxibonsai_kernels::GpuWeightHandle;
use std::sync::Mutex;

use super::scratch::ScratchBuffers;

/// A single Qwen3 Transformer block.
///
/// Holds references to weight data (zero-copy from GGUF mmap).
pub struct TransformerBlock<'a> {
    /// Layer index (0-based).
    pub(super) layer_idx: usize,
    /// Pre-attention RMSNorm.
    pub(super) attn_norm: RmsNorm,
    /// Q projection: [hidden_size → num_heads * head_dim].
    pub(super) attn_q: LinearLayer<'a>,
    /// K projection: [hidden_size → num_kv_heads * head_dim].
    pub(super) attn_k: LinearLayer<'a>,
    /// V projection: [hidden_size → num_kv_heads * head_dim].
    pub(super) attn_v: LinearLayer<'a>,
    /// Output projection: [num_heads * head_dim → hidden_size].
    pub(super) attn_output: LinearLayer<'a>,
    /// Per-head QK-norm on Q vectors (shape=[head_dim], shared across all Q heads).
    pub(super) attn_q_norm: RmsNorm,
    /// Per-head QK-norm on K vectors (shape=[head_dim], shared across all KV heads).
    pub(super) attn_k_norm: RmsNorm,
    /// Pre-FFN RMSNorm.
    pub(super) ffn_norm: RmsNorm,
    /// Gate projection: [hidden_size → intermediate_size].
    pub(super) ffn_gate: LinearLayer<'a>,
    /// Up projection: [hidden_size → intermediate_size].
    pub(super) ffn_up: LinearLayer<'a>,
    /// Down projection: [intermediate_size → hidden_size].
    pub(super) ffn_down: LinearLayer<'a>,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) hidden_size: usize,
    /// Fused Q+K+V weight handle (single GPU dispatch) — **1-bit only**.
    ///
    /// Every consumer decodes this buffer as `Q1_0_g128`
    /// (`OneBitKernel::gemv_cached`, `try_metal_qkv`, `try_cuda_qkv`, the Q1
    /// full-layer path), so it is never populated for another format; a
    /// ternary block's fused handle lives in
    /// [`Self::fused_qkv_handle_ternary`] instead (`M-21`).
    pub(super) fused_qkv_handle: Option<GpuWeightHandle>,
    /// Fused gate+up weight handle (single GPU dispatch) — **1-bit only**, for
    /// the same reason as `fused_qkv_handle`.
    pub(super) fused_gate_up_handle: Option<GpuWeightHandle>,
    /// Fused Q‖K‖V handle of a **ternary** (`TQ2_0_g128`) block (`M-21`).
    ///
    /// Built by `upload_to_gpu` through `TernaryKernel::upload_weights_ternary`
    /// over the concatenated Q, K and V blocks, and consumed by `forward`'s
    /// ternary fused-QKV arms, whose GPU slot is this handle's id (see
    /// `TransformerBlock::ternary_fused_qkv_slot`). Kept apart from
    /// `fused_qkv_handle` on purpose: putting a TQ2 handle there would divert
    /// the ternary block into the 1-bit fused branch (which then falls back to
    /// three separate projections) and away from the working ternary fused
    /// arm.
    pub(super) fused_qkv_handle_ternary: Option<GpuWeightHandle>,
    /// Fused gate‖up handle of a **ternary** block (`M-21`): one GEMV for both
    /// FFN input projections instead of two, keyed on this handle's id (see
    /// `TransformerBlock::ternary_fused_gate_up_slot`).
    pub(super) fused_gate_up_handle_ternary: Option<GpuWeightHandle>,
    /// Pre-allocated scratch buffers (Mutex for Sync safety; uncontended in practice).
    pub(super) scratch: Mutex<ScratchBuffers>,
}

impl<'a> TransformerBlock<'a> {
    /// Create a new Transformer block from loaded weights.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layer_idx: usize,
        attn_norm: RmsNorm,
        attn_q: LinearLayer<'a>,
        attn_k: LinearLayer<'a>,
        attn_v: LinearLayer<'a>,
        attn_output: LinearLayer<'a>,
        attn_q_norm: RmsNorm,
        attn_k_norm: RmsNorm,
        ffn_norm: RmsNorm,
        ffn_gate: LinearLayer<'a>,
        ffn_up: LinearLayer<'a>,
        ffn_down: LinearLayer<'a>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_size: usize,
    ) -> Self {
        let inter = ffn_gate.out_features();
        let scratch = Mutex::new(ScratchBuffers::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            inter,
        ));
        Self {
            layer_idx,
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_q_norm,
            attn_k_norm,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            fused_qkv_handle: None,
            fused_gate_up_handle: None,
            fused_qkv_handle_ternary: None,
            fused_gate_up_handle_ternary: None,
            scratch,
        }
    }
}
