//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use metal::{Buffer, MTLResourceOptions};
use super::super::metal_graph::{alloc_buf, MetalGraphError};


/// Intermediate buffers for batch prefill (processing multiple tokens at once).
///
/// Batched buffers use column-major layout: `buf[col * dim + element]` where
/// `col` is the batch (token) index and `dim` is the vector dimension.
///
/// Single-token buffers are used inside the sequential attention loop where
/// we process one query position at a time with the existing attention kernels.
pub(crate) struct PrefillBuffers {
    /// Batched hidden states: `[batch_size × hidden_size]` f32 (column-major).
    pub hidden_buf: Buffer,
    /// Batched normed hidden: `[batch_size × hidden_size]` f32 (column-major).
    pub normed_buf: Buffer,
    /// Batched QKV projection output: `[batch_size × qkv_dim]` f32 (column-major).
    pub qkv_buf: Buffer,
    /// Batched attention output: `[batch_size × nq*head_dim]` f32 (column-major).
    pub attn_out_buf: Buffer,
    /// Batched SwiGLU output: `[batch_size × intermediate_size]` f32 (column-major).
    pub swiglu_buf: Buffer,
    /// Batched concatenated gate+up GEMM output: `[batch_size × 2·intermediate_size]`
    /// f32 (column-major). Used by the ternary prefill path which lacks a
    /// fused gate+up+SwiGLU GEMM kernel — the TQ2 GEMM writes here, then
    /// `dispatch_batched_swiglu` reduces it to `swiglu_buf`.
    /// The Q1 prefill path's fused `fused_gate_up_swiglu_gemm_q1` kernel
    /// writes directly into `swiglu_buf` and ignores this buffer.
    pub gate_up_buf: Buffer,
    /// Single-token Q after norm: `[nq * head_dim]` f32.
    pub q_normed_buf: Buffer,
    /// Single-token K after norm: `[nkv * head_dim]` f32.
    pub k_normed_buf: Buffer,
    /// Single-token Q after RoPE: `[nq * head_dim]` f32.
    pub q_rope_buf: Buffer,
    /// Single-token K after RoPE: `[nkv * head_dim]` f32.
    pub k_rope_buf: Buffer,
    /// Single-token attention scores: `[nq × max_seq]` f32.
    pub scores_buf: Buffer,
    /// RoPE cos table: `[batch_size × half_dim]` f32.
    pub cos_buf: Buffer,
    /// RoPE sin table: `[batch_size × half_dim]` f32.
    pub sin_buf: Buffer,
    /// Cached dimensions.
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
}
impl PrefillBuffers {
    /// Allocate all prefill buffers for the given dimensions and batch size.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        device: &metal::Device,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self, MetalGraphError> {
        let f = std::mem::size_of::<f32>();
        let shared = MTLResourceOptions::StorageModeShared;
        let private = MTLResourceOptions::StorageModePrivate;
        let qkv_dim = nq * head_dim + 2 * nkv * head_dim;
        Ok(Self {
            hidden_buf: alloc_buf(
                device,
                (batch_size * hidden_size * f) as u64,
                shared,
            )?,
            normed_buf: alloc_buf(
                device,
                (batch_size * hidden_size * f) as u64,
                private,
            )?,
            qkv_buf: alloc_buf(device, (batch_size * qkv_dim * f) as u64, private)?,
            attn_out_buf: alloc_buf(
                device,
                (batch_size * nq * head_dim * f) as u64,
                private,
            )?,
            swiglu_buf: alloc_buf(
                device,
                (batch_size * intermediate_size * f) as u64,
                private,
            )?,
            gate_up_buf: alloc_buf(
                device,
                (batch_size * 2 * intermediate_size * f) as u64,
                private,
            )?,
            q_normed_buf: alloc_buf(device, (nq * head_dim * f) as u64, private)?,
            k_normed_buf: alloc_buf(device, (nkv * head_dim * f) as u64, private)?,
            q_rope_buf: alloc_buf(device, (nq * head_dim * f) as u64, private)?,
            k_rope_buf: alloc_buf(device, (nkv * head_dim * f) as u64, private)?,
            scores_buf: alloc_buf(device, (nq * max_seq * f) as u64, private)?,
            cos_buf: alloc_buf(
                device,
                (batch_size * (head_dim / 2) * f) as u64,
                shared,
            )?,
            sin_buf: alloc_buf(
                device,
                (batch_size * (head_dim / 2) * f) as u64,
                shared,
            )?,
            batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq,
        })
    }
    /// Check whether existing buffers match the requested dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn matches(
        &self,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> bool {
        self.batch_size == batch_size && self.hidden_size == hidden_size
            && self.intermediate_size == intermediate_size && self.nq == nq
            && self.nkv == nkv && self.head_dim == head_dim && self.max_seq == max_seq
    }
}
/// Model configuration for a single transformer layer.
pub(crate) struct LayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub max_seq_len: usize,
}
/// References to a single layer's weight buffers for prefill encoding.
pub(crate) struct LayerWeightRefs<'a> {
    pub attn_norm: &'a Buffer,
    pub qkv: &'a Buffer,
    pub q_norm: &'a Buffer,
    pub k_norm: &'a Buffer,
    pub output_proj: &'a Buffer,
    pub ffn_norm: &'a Buffer,
    pub gate_up: &'a Buffer,
    pub down: &'a Buffer,
}
