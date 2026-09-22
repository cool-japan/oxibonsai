//! The two decoder-layer kinds of a `qwen35` stack (M-04 / M-16, design
//! §3.2), and the per-layer scratch their forwards run in.
//!
//! Layer `i` is **full attention** iff `(i + 1) % full_attention_interval ==
//! 0` (27B: `{3, 7, …, 63}`, 16 of 64); every other layer is a recurrent
//! Gated-DeltaNet ("linear attention") layer.
//!
//! # Slots, not layer indices
//!
//! A full layer carries a `kv_slot` into a **16**-entry KV cache and a
//! linear layer a `rec_slot` into a **48**-entry recurrent cache. Indexing
//! either cache by `layer_idx` would allocate 4× the memory it needs and
//! leave three quarters of it permanently zero.
//!
//! # Scratch
//!
//! Each block owns its activation buffers behind a `Mutex`, allocated once
//! from the config rather than per token (`forward` used to allocate three
//! `Vec`s per token on the dense path; the 27B would allocate fourteen).
//! They live here, with the block, because B2-11's `block_full.rs` /
//! `block_linear.rs` own the *forward bodies* but not this file — a forward
//! must not have to widen the type it runs on.
//!
//! `q_all` is deliberately `2 * n_heads * head_dim` wide and `gate` is a
//! **separate** buffer: `attn_q [5120, 12288]` is `[q(256) | gate(256)]`
//! interleaved per head, and the de-interleave has to happen before
//! `compute_gqa_attention`, which parallelises over `head_dim`-sized chunks
//! of its output and would otherwise stride wrongly (M-16's correction).

use std::sync::{Arc, Mutex, MutexGuard};

use oxibonsai_core::config_hybrid::HybridConfig;

use crate::error::{ModelError, ModelResult};
use crate::hybrid::weights::{Bf16Matrix, GdnGateWeights};
use crate::layers::linear::LinearLayer;
use crate::layers::rms_norm::RmsNorm;

/// One decoder layer of a Qwen3.5 hybrid stack.
#[derive(Debug)]
pub enum HybridBlock<'a> {
    /// `(layer_idx + 1) % full_attention_interval == 0`.
    Full(FullAttnBlock<'a>),
    /// Gated-DeltaNet linear-attention layer.
    Linear(LinearAttnBlock<'a>),
}

impl<'a> HybridBlock<'a> {
    /// This layer's index in the stack (`0..block_count`).
    #[must_use]
    pub fn layer_idx(&self) -> usize {
        match self {
            Self::Full(b) => b.layer_idx,
            Self::Linear(b) => b.layer_idx,
        }
    }

    /// `true` for a full-attention layer.
    #[must_use]
    pub fn is_full(&self) -> bool {
        matches!(self, Self::Full(_))
    }

    /// This layer's KV-cache slot, or `None` for a linear layer (which has
    /// no KV cache at all — its history lives in the recurrent state).
    #[must_use]
    pub fn kv_slot(&self) -> Option<usize> {
        match self {
            Self::Full(b) => Some(b.kv_slot),
            Self::Linear(_) => None,
        }
    }

    /// This layer's recurrent-cache slot, or `None` for a full layer.
    #[must_use]
    pub fn rec_slot(&self) -> Option<usize> {
        match self {
            Self::Full(_) => None,
            Self::Linear(b) => Some(b.rec_slot),
        }
    }

    /// The full-attention layer, if this is one.
    #[must_use]
    pub fn as_full(&self) -> Option<&FullAttnBlock<'a>> {
        match self {
            Self::Full(b) => Some(b),
            Self::Linear(_) => None,
        }
    }

    /// The Gated-DeltaNet layer, if this is one.
    #[must_use]
    pub fn as_linear(&self) -> Option<&LinearAttnBlock<'a>> {
        match self {
            Self::Full(_) => None,
            Self::Linear(b) => Some(b),
        }
    }
}

/// A full-attention layer: GQA with `head_dim` 256, a `q|gate` interleave
/// inside one projection, per-head q/k RMSNorm and a sigmoid output gate.
#[derive(Debug)]
pub struct FullAttnBlock<'a> {
    /// Index in the stack.
    pub(crate) layer_idx: usize,
    /// Pre-attention RMSNorm, `[hidden]`.
    pub(crate) attn_norm: RmsNorm,
    /// Pre-FFN RMSNorm (GGUF `post_attention_norm`), `[hidden]`.
    pub(crate) post_attn_norm: RmsNorm,
    /// `[q | gate]` projection, `hidden -> 2 * n_heads * head_dim`,
    /// interleaved per head.
    pub(crate) attn_q: LinearLayer<'a>,
    /// Key projection, `hidden -> n_kv_heads * head_dim`.
    pub(crate) attn_k: LinearLayer<'a>,
    /// Value projection, `hidden -> n_kv_heads * head_dim`.
    pub(crate) attn_v: LinearLayer<'a>,
    /// Output projection, `n_heads * head_dim -> hidden`.
    pub(crate) attn_output: LinearLayer<'a>,
    /// Per-head query RMSNorm, `[head_dim]`.
    pub(crate) attn_q_norm: RmsNorm,
    /// Per-head key RMSNorm, `[head_dim]`.
    pub(crate) attn_k_norm: RmsNorm,
    /// SwiGLU gate projection.
    pub(crate) ffn_gate: LinearLayer<'a>,
    /// SwiGLU up projection.
    pub(crate) ffn_up: LinearLayer<'a>,
    /// SwiGLU down projection.
    pub(crate) ffn_down: LinearLayer<'a>,
    /// Index into the 16-entry KV cache — **not** `layer_idx`.
    pub(crate) kv_slot: usize,
    /// Activation buffers, allocated once per layer.
    pub(crate) scratch: Mutex<FullScratch>,
}

impl<'a> FullAttnBlock<'a> {
    /// Index in the stack.
    #[inline]
    #[must_use]
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Index into the KV cache (`0..num_full_layers`).
    #[inline]
    #[must_use]
    pub fn kv_slot(&self) -> usize {
        self.kv_slot
    }

    /// Pre-attention RMSNorm.
    #[inline]
    #[must_use]
    pub fn attn_norm(&self) -> &RmsNorm {
        &self.attn_norm
    }

    /// Pre-FFN RMSNorm.
    #[inline]
    #[must_use]
    pub fn post_attn_norm(&self) -> &RmsNorm {
        &self.post_attn_norm
    }

    /// The fused `[q | gate]` projection.
    #[inline]
    #[must_use]
    pub fn attn_q(&self) -> &LinearLayer<'a> {
        &self.attn_q
    }

    /// The key projection.
    #[inline]
    #[must_use]
    pub fn attn_k(&self) -> &LinearLayer<'a> {
        &self.attn_k
    }

    /// The value projection.
    #[inline]
    #[must_use]
    pub fn attn_v(&self) -> &LinearLayer<'a> {
        &self.attn_v
    }

    /// The attention output projection.
    #[inline]
    #[must_use]
    pub fn attn_output(&self) -> &LinearLayer<'a> {
        &self.attn_output
    }

    /// Per-head query RMSNorm.
    #[inline]
    #[must_use]
    pub fn attn_q_norm(&self) -> &RmsNorm {
        &self.attn_q_norm
    }

    /// Per-head key RMSNorm.
    #[inline]
    #[must_use]
    pub fn attn_k_norm(&self) -> &RmsNorm {
        &self.attn_k_norm
    }

    /// The SwiGLU gate projection.
    #[inline]
    #[must_use]
    pub fn ffn_gate(&self) -> &LinearLayer<'a> {
        &self.ffn_gate
    }

    /// The SwiGLU up projection.
    #[inline]
    #[must_use]
    pub fn ffn_up(&self) -> &LinearLayer<'a> {
        &self.ffn_up
    }

    /// The SwiGLU down projection.
    #[inline]
    #[must_use]
    pub fn ffn_down(&self) -> &LinearLayer<'a> {
        &self.ffn_down
    }

    /// Lock this layer's activation buffers.
    ///
    /// # Errors
    ///
    /// [`ModelError::Internal`] when the mutex is poisoned (a previous
    /// forward panicked while holding it).
    pub fn scratch(&self) -> ModelResult<MutexGuard<'_, FullScratch>> {
        self.scratch
            .lock()
            .map_err(|_| ModelError::Internal(format!("layer {} scratch poisoned", self.layer_idx)))
    }
}

/// A Gated-DeltaNet (linear-attention) layer.
#[derive(Debug)]
pub struct LinearAttnBlock<'a> {
    /// Index in the stack.
    pub(crate) layer_idx: usize,
    /// Pre-attention RMSNorm, `[hidden]`.
    pub(crate) attn_norm: RmsNorm,
    /// Pre-FFN RMSNorm (GGUF `post_attention_norm`), `[hidden]`.
    pub(crate) post_attn_norm: RmsNorm,
    /// `[q | k | v]` projection, `hidden -> conv_dim`.
    pub(crate) attn_qkv: LinearLayer<'a>,
    /// Output gate `z`, `hidden -> ssm_inner_size` (tiled v order).
    pub(crate) attn_gate: LinearLayer<'a>,
    /// Raw decay projection, BF16, `hidden -> n_v_heads`. **Not folded** —
    /// it consumes the *un-rotated* activation.
    pub(crate) ssm_alpha: Bf16Matrix<'a>,
    /// Raw β projection, BF16, `hidden -> n_v_heads`. **Not folded**.
    pub(crate) ssm_beta: Bf16Matrix<'a>,
    /// Depthwise causal conv weights, `[conv_dim][conv_kernel]`
    /// channel-major — the layout `causal_conv1d_k4_decode` expects.
    pub(crate) ssm_conv1d: Arc<[f32]>,
    /// `ssm_a` (`A = -exp(A_log)`) and `ssm_dt.bias`, bound by name.
    pub(crate) gates: GdnGateWeights,
    /// Gated RMSNorm weight, `[head_v_dim]`, shared across v-heads.
    pub(crate) ssm_norm: RmsNorm,
    /// Output projection, `ssm_inner_size -> hidden`, folded, with its
    /// columns in **grouped** v-head order.
    pub(crate) ssm_out: LinearLayer<'a>,
    /// SwiGLU gate projection.
    pub(crate) ffn_gate: LinearLayer<'a>,
    /// SwiGLU up projection.
    pub(crate) ffn_up: LinearLayer<'a>,
    /// SwiGLU down projection.
    pub(crate) ffn_down: LinearLayer<'a>,
    /// Index into the 48-entry recurrent cache — **not** `layer_idx`.
    pub(crate) rec_slot: usize,
    /// Activation buffers, allocated once per layer.
    pub(crate) scratch: Mutex<LinearScratch>,
}

impl<'a> LinearAttnBlock<'a> {
    /// Index in the stack.
    #[inline]
    #[must_use]
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Index into the recurrent cache (`0..num_linear_layers`).
    #[inline]
    #[must_use]
    pub fn rec_slot(&self) -> usize {
        self.rec_slot
    }

    /// Pre-attention RMSNorm.
    #[inline]
    #[must_use]
    pub fn attn_norm(&self) -> &RmsNorm {
        &self.attn_norm
    }

    /// Pre-FFN RMSNorm.
    #[inline]
    #[must_use]
    pub fn post_attn_norm(&self) -> &RmsNorm {
        &self.post_attn_norm
    }

    /// The fused `[q | k | v]` projection.
    #[inline]
    #[must_use]
    pub fn attn_qkv(&self) -> &LinearLayer<'a> {
        &self.attn_qkv
    }

    /// The `z` gate projection.
    #[inline]
    #[must_use]
    pub fn attn_gate(&self) -> &LinearLayer<'a> {
        &self.attn_gate
    }

    /// The raw decay projection (un-rotated input).
    #[inline]
    #[must_use]
    pub fn ssm_alpha(&self) -> &Bf16Matrix<'a> {
        &self.ssm_alpha
    }

    /// The raw β projection (un-rotated input).
    #[inline]
    #[must_use]
    pub fn ssm_beta(&self) -> &Bf16Matrix<'a> {
        &self.ssm_beta
    }

    /// Depthwise conv weights, channel-major.
    #[inline]
    #[must_use]
    pub fn ssm_conv1d(&self) -> &[f32] {
        &self.ssm_conv1d
    }

    /// The Gated-DeltaNet gate pair, bound by name.
    #[inline]
    #[must_use]
    pub fn gates(&self) -> &GdnGateWeights {
        &self.gates
    }

    /// The gated RMSNorm weight.
    #[inline]
    #[must_use]
    pub fn ssm_norm(&self) -> &RmsNorm {
        &self.ssm_norm
    }

    /// The Gated-DeltaNet output projection.
    #[inline]
    #[must_use]
    pub fn ssm_out(&self) -> &LinearLayer<'a> {
        &self.ssm_out
    }

    /// The SwiGLU gate projection.
    #[inline]
    #[must_use]
    pub fn ffn_gate(&self) -> &LinearLayer<'a> {
        &self.ffn_gate
    }

    /// The SwiGLU up projection.
    #[inline]
    #[must_use]
    pub fn ffn_up(&self) -> &LinearLayer<'a> {
        &self.ffn_up
    }

    /// The SwiGLU down projection.
    #[inline]
    #[must_use]
    pub fn ffn_down(&self) -> &LinearLayer<'a> {
        &self.ffn_down
    }

    /// Lock this layer's activation buffers.
    ///
    /// # Errors
    ///
    /// [`ModelError::Internal`] when the mutex is poisoned.
    pub fn scratch(&self) -> ModelResult<MutexGuard<'_, LinearScratch>> {
        self.scratch
            .lock()
            .map_err(|_| ModelError::Internal(format!("layer {} scratch poisoned", self.layer_idx)))
    }
}

/// Activation buffers of one full-attention layer, allocated once.
#[derive(Debug, Clone)]
pub struct FullScratch {
    /// RMSNorm output, `[hidden]`.
    pub normed: Vec<f32>,
    /// `attn_q`'s raw output, `[2 * n_heads * head_dim]` — `q` and `gate`
    /// still interleaved per head.
    pub q_all: Vec<f32>,
    /// De-interleaved queries, `[n_heads * head_dim]`.
    pub q: Vec<f32>,
    /// De-interleaved output gate, `[n_heads * head_dim]`.
    pub gate: Vec<f32>,
    /// Keys, `[n_kv_heads * head_dim]`.
    pub k: Vec<f32>,
    /// Values, `[n_kv_heads * head_dim]`.
    pub v: Vec<f32>,
    /// Attention output before the gate, `[n_heads * head_dim]`.
    pub attn_out: Vec<f32>,
    /// SwiGLU gate branch, `[intermediate]`.
    pub ffn_gate: Vec<f32>,
    /// SwiGLU up branch, `[intermediate]`.
    pub ffn_up: Vec<f32>,
}

impl FullScratch {
    /// Allocate every buffer from `config`.
    #[must_use]
    pub fn new(config: &HybridConfig) -> Self {
        let heads_width = config.base.num_attention_heads * config.base.head_dim;
        let kv_width = config.base.num_kv_heads * config.base.head_dim;
        Self {
            normed: vec![0.0; config.base.hidden_size],
            q_all: vec![0.0; heads_width * 2],
            q: vec![0.0; heads_width],
            gate: vec![0.0; heads_width],
            k: vec![0.0; kv_width],
            v: vec![0.0; kv_width],
            attn_out: vec![0.0; heads_width],
            ffn_gate: vec![0.0; config.base.intermediate_size],
            ffn_up: vec![0.0; config.base.intermediate_size],
        }
    }

    /// Total bytes held.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let floats = self.normed.len()
            + self.q_all.len()
            + self.q.len()
            + self.gate.len()
            + self.k.len()
            + self.v.len()
            + self.attn_out.len()
            + self.ffn_gate.len()
            + self.ffn_up.len();
        floats * core::mem::size_of::<f32>()
    }
}

/// Activation buffers of one Gated-DeltaNet layer, allocated once.
///
/// The `_grouped` buffers are where [`crate::hybrid::VHeadMap`] lands: the
/// GGUF hands out `v` and `z` in tiled v-head order and the kernel consumes
/// grouped order, so exactly two gathers per token happen here and nothing
/// is permuted on disk.
#[derive(Debug, Clone)]
pub struct LinearScratch {
    /// RMSNorm output, `[hidden]`.
    pub normed: Vec<f32>,
    /// `attn_qkv`'s output, `[conv_dim]` — `[q | k | v]`, v in tiled order.
    pub qkv: Vec<f32>,
    /// Post-conv `[q | k | v]`, `[conv_dim]`.
    pub conv_out: Vec<f32>,
    /// L2-normalised queries, `[n_k_heads * head_k_dim]`.
    pub q: Vec<f32>,
    /// L2-normalised keys, `[n_k_heads * head_k_dim]`.
    pub k: Vec<f32>,
    /// Values re-indexed into grouped v-head order, `[ssm_inner_size]`.
    pub v_grouped: Vec<f32>,
    /// `z` gate in raw (tiled) order, `[ssm_inner_size]`.
    pub z_tiled: Vec<f32>,
    /// `z` gate re-indexed into grouped order, `[ssm_inner_size]`.
    pub z_grouped: Vec<f32>,
    /// Raw `ssm_alpha` output in tiled order, `[n_v_heads]`.
    pub alpha_tiled: Vec<f32>,
    /// Raw `ssm_beta` output in tiled order, `[n_v_heads]`.
    pub beta_tiled: Vec<f32>,
    /// `ssm_alpha` re-indexed into grouped order, `[n_v_heads]`.
    pub alpha_grouped: Vec<f32>,
    /// `ssm_beta` re-indexed into grouped order, `[n_v_heads]`.
    pub beta_grouped: Vec<f32>,
    /// Gated-DeltaNet output in grouped order, `[ssm_inner_size]`.
    pub gdn_out: Vec<f32>,
    /// SwiGLU gate branch, `[intermediate]`.
    pub ffn_gate: Vec<f32>,
    /// SwiGLU up branch, `[intermediate]`.
    pub ffn_up: Vec<f32>,
}

impl LinearScratch {
    /// Allocate every buffer from `config`.
    #[must_use]
    pub fn new(config: &HybridConfig) -> Self {
        let inner = config.ssm_inner_size;
        let qk = config.n_k_heads() * config.head_k_dim();
        let heads = config.n_v_heads();
        Self {
            normed: vec![0.0; config.base.hidden_size],
            qkv: vec![0.0; config.conv_dim()],
            conv_out: vec![0.0; config.conv_dim()],
            q: vec![0.0; qk],
            k: vec![0.0; qk],
            v_grouped: vec![0.0; inner],
            z_tiled: vec![0.0; inner],
            z_grouped: vec![0.0; inner],
            alpha_tiled: vec![0.0; heads],
            beta_tiled: vec![0.0; heads],
            alpha_grouped: vec![0.0; heads],
            beta_grouped: vec![0.0; heads],
            gdn_out: vec![0.0; inner],
            ffn_gate: vec![0.0; config.base.intermediate_size],
            ffn_up: vec![0.0; config.base.intermediate_size],
        }
    }

    /// Total bytes held.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let floats = self.normed.len()
            + self.qkv.len()
            + self.conv_out.len()
            + self.q.len()
            + self.k.len()
            + self.v_grouped.len()
            + self.z_tiled.len()
            + self.z_grouped.len()
            + self.alpha_tiled.len()
            + self.beta_tiled.len()
            + self.alpha_grouped.len()
            + self.beta_grouped.len()
            + self.gdn_out.len()
            + self.ffn_gate.len()
            + self.ffn_up.len();
        floats * core::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::tests_support::bonsai2_config;

    #[test]
    fn full_scratch_sizes_match_the_27b_geometry() {
        let config = bonsai2_config();
        let scratch = FullScratch::new(&config);
        assert_eq!(scratch.normed.len(), 5120);
        // `attn_q` is [5120, 12288] = 24 heads x 256 x 2 — q and gate.
        assert_eq!(scratch.q_all.len(), 12_288);
        assert_eq!(scratch.q.len(), 6144);
        assert_eq!(scratch.gate.len(), 6144);
        assert_eq!(scratch.q.len() + scratch.gate.len(), scratch.q_all.len());
        // GQA: 4 kv heads of 256.
        assert_eq!(scratch.k.len(), 1024);
        assert_eq!(scratch.v.len(), 1024);
        assert_eq!(scratch.attn_out.len(), 6144);
        assert_eq!(scratch.ffn_gate.len(), 17_408);
        assert_eq!(scratch.ffn_up.len(), 17_408);
        // ~230 KB per full layer, 16 layers: a rounding error beside the
        // weights, and zero per-token allocation.
        assert_eq!(
            scratch.memory_bytes(),
            4 * (5120 + 12_288 + 6144 * 3 + 1024 * 2 + 17_408 * 2)
        );
    }

    #[test]
    fn linear_scratch_sizes_match_the_27b_geometry() {
        let config = bonsai2_config();
        let scratch = LinearScratch::new(&config);
        assert_eq!(scratch.normed.len(), 5120);
        // `attn_qkv` is [5120, 10240] = q 2048 | k 2048 | v 6144.
        assert_eq!(scratch.qkv.len(), 10_240);
        assert_eq!(scratch.conv_out.len(), 10_240);
        assert_eq!(scratch.q.len(), 2048);
        assert_eq!(scratch.k.len(), 2048);
        for buffer in [
            &scratch.v_grouped,
            &scratch.z_tiled,
            &scratch.z_grouped,
            &scratch.gdn_out,
        ] {
            assert_eq!(buffer.len(), 6144);
        }
        for buffer in [
            &scratch.alpha_tiled,
            &scratch.beta_tiled,
            &scratch.alpha_grouped,
            &scratch.beta_grouped,
        ] {
            assert_eq!(buffer.len(), 48);
        }
        assert_eq!(scratch.ffn_gate.len(), 17_408);
        assert!(scratch.memory_bytes() > 0);
    }

    #[test]
    fn scratch_is_allocated_per_layer_kind_not_per_token() {
        let config = bonsai2_config();
        let full = FullScratch::new(&config);
        let linear = LinearScratch::new(&config);
        // Both fit comfortably under a megabyte, so holding one per layer
        // for all 64 layers is ~13 MB — the point of pre-allocating.
        assert!(full.memory_bytes() < 1 << 20, "{}", full.memory_bytes());
        assert!(linear.memory_bytes() < 1 << 20, "{}", linear.memory_bytes());
    }
}
