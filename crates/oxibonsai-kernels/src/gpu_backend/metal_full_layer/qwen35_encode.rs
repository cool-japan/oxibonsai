//! Per-layer command encoding of the Qwen3.5 hybrid encoder (a child
//! module of `metal_full_layer::qwen35`): the stage list of each layer kind,
//! the kernels every stage dispatches and the attention / LM-head tails.
//!
//! Encoding is fallible: a layer whose KV or recurrent slot is missing from
//! the dense slot index, or a folded width without a bound sign vector, is
//! an [`MetalGraphError::EncodingFailed`] rather than a dispatch against
//! another layer's state. The callers close the encoder on every path and
//! drop the command buffer uncommitted on an error.

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use super::{
    encode_gemv, rotation_scale, set_f32, set_u32, set_u64, LayerKind, Qwen35GpuModel, GDN_THREADS,
    NORM_THREADS, SCORES_BATCH_STRIDE,
};
use crate::gated_delta_net::GdnDims;
use crate::gpu_backend::kernel_sources::ATTENTION_SCORES_V2_THREADS;
use crate::gpu_backend::metal_graph::MetalGraphError;

// ═══════════════════════════════════════════════════════════════════════════
// Per-layer encoding
// ═══════════════════════════════════════════════════════════════════════════

/// One kernel group of a layer, in execution order. A forward encodes every
/// stage of every layer into one encoder; [`Qwen35GpuModel::trace_layer`]
/// commits after each so it can read the stage's outputs back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    AttnNorm,
    LinearProj,
    Conv,
    Gdn,
    GatedNorm,
    SsmOut,
    FullProj,
    QkNormRope,
    Attention,
    SigmoidGate,
    AttnOut,
    FfnNorm,
    FfnGateUp,
    FfnAct,
    FfnDown,
}

const LINEAR_STAGES: [Stage; 10] = [
    Stage::AttnNorm,
    Stage::LinearProj,
    Stage::Conv,
    Stage::Gdn,
    Stage::GatedNorm,
    Stage::SsmOut,
    Stage::FfnNorm,
    Stage::FfnGateUp,
    Stage::FfnAct,
    Stage::FfnDown,
];

const FULL_STAGES: [Stage; 10] = [
    Stage::AttnNorm,
    Stage::FullProj,
    Stage::QkNormRope,
    Stage::Attention,
    Stage::SigmoidGate,
    Stage::AttnOut,
    Stage::FfnNorm,
    Stage::FfnGateUp,
    Stage::FfnAct,
    Stage::FfnDown,
];

/// An encoding precondition that does not hold.
fn encoding_failed(what: String) -> MetalGraphError {
    MetalGraphError::EncodingFailed(format!("qwen35 GPU: {what}"))
}

impl Qwen35GpuModel<'_> {
    pub(super) fn layer_stages(&self, layer: usize) -> &'static [Stage] {
        match self.layers.get(layer) {
            Some(LayerKind::FullAttention(_)) => &FULL_STAGES,
            Some(LayerKind::LinearAttention(_)) => &LINEAR_STAGES,
            None => &[],
        }
    }

    /// The buffers a stage writes, with their per-token widths, for
    /// [`Self::trace_layer`].
    pub(super) fn stage_outputs(&self, stage: Stage) -> Vec<(&'static str, &Buffer, usize)> {
        let s = &self.scratch;
        let c = &self.cfg;
        let folded = c.hadamard_block.is_some();
        match stage {
            Stage::AttnNorm | Stage::FfnNorm => {
                let (normed, rotated) = if stage == Stage::AttnNorm {
                    ("attn_norm", "attn_norm_rotated")
                } else {
                    ("ffn_norm", "ffn_norm_rotated")
                };
                let mut out = vec![(normed, &s.normed, c.hidden)];
                if folded {
                    out.push((rotated, &s.rotated, c.hidden));
                }
                out
            }
            Stage::LinearProj => vec![
                ("attn_qkv", &s.qkv, c.conv_dim()),
                ("attn_gate", &s.z, c.inner()),
                ("ssm_alpha_beta", &s.ab, 2 * c.n_v_heads),
            ],
            Stage::Conv => vec![("conv_silu", &s.conv_out, c.conv_dim())],
            Stage::Gdn => vec![("gdn", &s.gdn_out, c.inner())],
            Stage::GatedNorm => vec![("gated_norm_rotated", &s.gdn_rot, c.inner())],
            Stage::SsmOut | Stage::AttnOut => vec![("attn_residual", &s.resid, c.hidden)],
            Stage::FullProj => vec![
                ("attn_q", &s.q_all, 2 * c.heads_width()),
                ("attn_k", &s.k, c.n_kv_heads * c.head_dim),
                ("attn_v", &s.v, c.n_kv_heads * c.head_dim),
            ],
            Stage::QkNormRope => vec![
                ("q_rope", &s.q_rope, c.heads_width()),
                ("k_rope", &s.k_rope, c.n_kv_heads * c.head_dim),
            ],
            Stage::Attention => vec![("attention", &s.attn, c.heads_width())],
            Stage::SigmoidGate => vec![("attn_gated_rotated", &s.attn_rot, c.heads_width())],
            Stage::FfnGateUp => vec![
                ("ffn_gate", &s.ffn_gate, c.intermediate),
                ("ffn_up", &s.ffn_up, c.intermediate),
            ],
            Stage::FfnAct => vec![("ffn_act_rotated", &s.ffn_act, c.intermediate)],
            Stage::FfnDown => vec![("residual", &s.resid, c.hidden)],
        }
    }

    pub(super) fn encode_layer(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize,
        t_len: usize,
        start_pos: usize,
    ) -> Result<(), MetalGraphError> {
        if layer >= self.layers.len() {
            return Err(encoding_failed(format!(
                "layer {layer} of {}",
                self.layers.len()
            )));
        }
        for &stage in self.layer_stages(layer) {
            self.encode_stage(enc, layer, stage, t_len, start_pos)?;
        }
        Ok(())
    }

    /// The KV slot of full-attention layer `layer`, from the dense index.
    fn kv_slot(&self, layer: usize) -> Result<u64, MetalGraphError> {
        self.layer_kv_slot
            .get(layer)
            .copied()
            .flatten()
            .map(u64::from)
            .ok_or_else(|| encoding_failed(format!("layer {layer} has no KV-cache slot")))
    }

    /// The recurrent-state slot of Gated-DeltaNet layer `layer`, from the
    /// dense index.
    fn rec_slot(&self, layer: usize) -> Result<u64, MetalGraphError> {
        self.layer_rec_slot
            .get(layer)
            .copied()
            .flatten()
            .map(u64::from)
            .ok_or_else(|| encoding_failed(format!("layer {layer} has no recurrent-state slot")))
    }

    /// The activation every folded projection of the current stage reads:
    /// the rotated copy when the checkpoint is folded, the plain one
    /// otherwise.
    fn folded_input(&self) -> &Buffer {
        if self.cfg.hadamard_block.is_some() {
            &self.scratch.rotated
        } else {
            &self.scratch.normed
        }
    }

    /// Sign vector for `width`. A folded model must have one bound (the
    /// constructor uploads every rotated width); an unfolded model's
    /// kernels never read the binding, so any live buffer stands in.
    fn signs_for(&self, width: usize) -> Result<&Buffer, MetalGraphError> {
        match (self.cfg.hadamard_block, self.signs.get(&width)) {
            (Some(_), Some(signs)) => Ok(signs),
            (Some(block), None) => Err(encoding_failed(format!(
                "no Hadamard sign vector is bound for width {width} (block {block})"
            ))),
            (None, _) => Ok(&self.scores),
        }
    }

    fn block_u32(&self) -> u32 {
        self.cfg.hadamard_block.map_or(0, |b| b as u32)
    }

    fn scale(&self) -> f32 {
        self.cfg.hadamard_block.map_or(1.0, rotation_scale)
    }

    fn encode_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &Buffer,
        rows: usize,
        x_offset: u64,
    ) -> Result<(), MetalGraphError> {
        let s = &self.scratch;
        let signs = self.signs_for(self.cfg.hidden)?;
        enc.set_compute_pipeline_state(&self.pipes.rmsnorm_rotate);
        enc.set_buffer(0, Some(&s.resid), x_offset);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(&s.normed), 0);
        enc.set_buffer(3, Some(&s.rotated), 0);
        enc.set_buffer(4, Some(signs), 0);
        set_u32(enc, 5, self.cfg.hidden as u32);
        set_f32(enc, 6, self.cfg.rms_eps);
        set_u32(enc, 7, self.block_u32());
        set_f32(enc, 8, self.scale());
        enc.dispatch_thread_groups(
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(NORM_THREADS, 1, 1),
        );
        Ok(())
    }

    pub(super) fn encode_stage(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize,
        stage: Stage,
        t_len: usize,
        start_pos: usize,
    ) -> Result<(), MetalGraphError> {
        let kind = self
            .layers
            .get(layer)
            .ok_or_else(|| encoding_failed(format!("layer {layer} of {}", self.layers.len())))?;
        let s = &self.scratch;
        let c = &self.cfg;
        let ffn = match kind {
            LayerKind::FullAttention(l) => &l.ffn,
            LayerKind::LinearAttention(l) => &l.ffn,
        };
        match (kind, stage) {
            (LayerKind::FullAttention(l), Stage::AttnNorm) => {
                self.encode_norm(enc, &l.attn_norm, t_len, 0)?;
            }
            (LayerKind::LinearAttention(l), Stage::AttnNorm) => {
                self.encode_norm(enc, &l.attn_norm, t_len, 0)?;
            }
            (LayerKind::FullAttention(l), Stage::FfnNorm) => {
                self.encode_norm(enc, &l.post_attention_norm, t_len, 0)?;
            }
            (LayerKind::LinearAttention(l), Stage::FfnNorm) => {
                self.encode_norm(enc, &l.post_attention_norm, t_len, 0)?;
            }
            (LayerKind::LinearAttention(l), Stage::LinearProj) => {
                let x = self.folded_input();
                encode_gemv(enc, &l.attn_qkv, (x, 0), (&s.qkv, 0), t_len, false);
                encode_gemv(enc, &l.attn_gate, (x, 0), (&s.z, 0), t_len, false);
                // `ssm_alpha` / `ssm_beta` are not folded: the un-rotated input.
                encode_gemv(
                    enc,
                    &l.ssm_alpha_beta,
                    (&s.normed, 0),
                    (&s.ab, 0),
                    t_len,
                    false,
                );
            }
            (LayerKind::LinearAttention(l), Stage::Conv) => {
                let conv_off = self.rec_slot(layer)? * (c.conv_dim() * 3 * 4) as u64;
                enc.set_compute_pipeline_state(&self.pipes.conv1d_silu);
                enc.set_buffer(0, Some(&s.qkv), 0);
                enc.set_buffer(1, Some(&l.ssm_conv1d), 0);
                enc.set_buffer(2, Some(&self.conv_state), conv_off);
                enc.set_buffer(3, Some(&s.conv_out), 0);
                set_u32(enc, 4, c.conv_dim() as u32);
                set_u32(enc, 5, t_len as u32);
                enc.dispatch_thread_groups(
                    MTLSize::new(c.conv_dim().div_ceil(256) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            (LayerKind::LinearAttention(l), Stage::Gdn) => {
                let per_slot = (c.n_v_heads * c.head_v_dim * c.head_k_dim * 4) as u64;
                let state_off = self.rec_slot(layer)? * per_slot;
                // The CPU recurrence's own output scale, `1/sqrt(head_v_dim)`;
                // `validate` admits only square states, where it is also the
                // reference implementation's `1/sqrt(S_k)`.
                let out_scale =
                    GdnDims::new(c.n_k_heads, c.n_v_heads, c.head_k_dim, c.head_v_dim).out_scale();
                enc.set_compute_pipeline_state(&self.pipes.gdn);
                enc.set_buffer(0, Some(&s.conv_out), 0);
                enc.set_buffer(1, Some(&s.ab), 0);
                enc.set_buffer(2, Some(&l.a_neg), 0);
                enc.set_buffer(3, Some(&l.dt_bias), 0);
                enc.set_buffer(4, Some(&self.ssm_state), state_off);
                enc.set_buffer(5, Some(&s.gdn_out), 0);
                set_u32(enc, 6, c.n_k_heads as u32);
                set_u32(enc, 7, c.n_v_heads as u32);
                set_u32(enc, 8, c.head_k_dim as u32);
                set_u32(enc, 9, c.head_v_dim as u32);
                set_u32(enc, 10, c.conv_dim() as u32);
                set_u32(enc, 11, t_len as u32);
                set_f32(enc, 12, c.rms_eps);
                set_f32(enc, 13, out_scale);
                enc.dispatch_thread_groups(
                    MTLSize::new(c.n_v_heads as u64, 1, 1),
                    MTLSize::new(GDN_THREADS, 1, 1),
                );
            }
            (LayerKind::LinearAttention(l), Stage::GatedNorm) => {
                let span = c.gated_norm_span();
                let signs = self.signs_for(c.inner())?;
                enc.set_compute_pipeline_state(&self.pipes.gated_norm_rotate);
                enc.set_buffer(0, Some(&s.gdn_out), 0);
                enc.set_buffer(1, Some(&s.z), 0);
                enc.set_buffer(2, Some(&l.ssm_norm), 0);
                enc.set_buffer(3, Some(&s.gdn_rot), 0);
                enc.set_buffer(4, Some(signs), 0);
                set_u32(enc, 5, c.inner() as u32);
                set_u32(enc, 6, c.head_v_dim as u32);
                set_u32(enc, 7, c.n_k_heads as u32);
                set_u32(enc, 8, (c.n_v_heads / c.n_k_heads) as u32);
                set_f32(enc, 9, c.rms_eps);
                set_u32(enc, 10, self.block_u32());
                set_f32(enc, 11, self.scale());
                set_u32(enc, 12, span as u32);
                enc.dispatch_thread_groups(
                    MTLSize::new((c.inner() / span) as u64, t_len as u64, 1),
                    MTLSize::new(span as u64, 1, 1),
                );
            }
            (LayerKind::LinearAttention(l), Stage::SsmOut) => {
                encode_gemv(enc, &l.ssm_out, (&s.gdn_rot, 0), (&s.resid, 0), t_len, true);
            }
            (LayerKind::FullAttention(l), Stage::FullProj) => {
                let x = self.folded_input();
                encode_gemv(enc, &l.attn_q, (x, 0), (&s.q_all, 0), t_len, false);
                encode_gemv(enc, &l.attn_k, (x, 0), (&s.k, 0), t_len, false);
                encode_gemv(enc, &l.attn_v, (x, 0), (&s.v, 0), t_len, false);
            }
            (LayerKind::FullAttention(l), Stage::QkNormRope) => {
                let half = (c.n_rot / 2) as u64;
                let rope_off = start_pos as u64 * half * 4;
                enc.set_compute_pipeline_state(&self.pipes.qk_norm_rope);
                enc.set_buffer(0, Some(&s.q_all), 0);
                enc.set_buffer(1, Some(&s.k), 0);
                enc.set_buffer(2, Some(&s.q_rope), 0);
                enc.set_buffer(3, Some(&s.k_rope), 0);
                enc.set_buffer(4, Some(&l.attn_q_norm), 0);
                enc.set_buffer(5, Some(&l.attn_k_norm), 0);
                enc.set_buffer(6, Some(&self.rope_cos), rope_off);
                enc.set_buffer(7, Some(&self.rope_sin), rope_off);
                set_u32(enc, 8, c.n_heads as u32);
                set_u32(enc, 9, c.n_kv_heads as u32);
                set_u32(enc, 10, c.head_dim as u32);
                set_f32(enc, 11, c.rms_eps);
                set_u32(enc, 12, c.n_rot as u32);
                set_u32(enc, 13, (2 * c.head_dim) as u32);
                set_u32(enc, 14, (2 * c.heads_width()) as u32);
                set_u32(enc, 15, (c.n_kv_heads * c.head_dim) as u32);
                enc.dispatch_thread_groups(
                    MTLSize::new((c.n_heads + c.n_kv_heads) as u64, t_len as u64, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            (LayerKind::FullAttention(_), Stage::Attention) => {
                self.encode_attention(enc, layer, t_len, start_pos)?;
            }
            (LayerKind::FullAttention(_), Stage::SigmoidGate) => {
                let span = c.span();
                let signs = self.signs_for(c.heads_width())?;
                enc.set_compute_pipeline_state(&self.pipes.sigmoid_gate_rotate);
                enc.set_buffer(0, Some(&s.attn), 0);
                enc.set_buffer(1, Some(&s.q_all), 0);
                enc.set_buffer(2, Some(&s.attn_rot), 0);
                enc.set_buffer(3, Some(signs), 0);
                set_u32(enc, 4, c.heads_width() as u32);
                set_u32(enc, 5, c.head_dim as u32);
                set_u32(enc, 6, self.block_u32());
                set_f32(enc, 7, self.scale());
                set_u32(enc, 8, span as u32);
                enc.dispatch_thread_groups(
                    MTLSize::new((c.heads_width() / span) as u64, t_len as u64, 1),
                    MTLSize::new(span as u64, 1, 1),
                );
            }
            (LayerKind::FullAttention(l), Stage::AttnOut) => {
                encode_gemv(
                    enc,
                    &l.attn_output,
                    (&s.attn_rot, 0),
                    (&s.resid, 0),
                    t_len,
                    true,
                );
            }
            (_, Stage::FfnGateUp) => {
                let x = self.folded_input();
                encode_gemv(enc, &ffn.gate, (x, 0), (&s.ffn_gate, 0), t_len, false);
                encode_gemv(enc, &ffn.up, (x, 0), (&s.ffn_up, 0), t_len, false);
            }
            (_, Stage::FfnAct) => {
                let span = c.span();
                let signs = self.signs_for(c.intermediate)?;
                enc.set_compute_pipeline_state(&self.pipes.swiglu_rotate);
                enc.set_buffer(0, Some(&s.ffn_gate), 0);
                enc.set_buffer(1, Some(&s.ffn_up), 0);
                enc.set_buffer(2, Some(&s.ffn_act), 0);
                enc.set_buffer(3, Some(signs), 0);
                set_u32(enc, 4, c.intermediate as u32);
                set_u32(enc, 5, self.block_u32());
                set_f32(enc, 6, self.scale());
                set_u32(enc, 7, span as u32);
                enc.dispatch_thread_groups(
                    MTLSize::new((c.intermediate / span) as u64, t_len as u64, 1),
                    MTLSize::new(span as u64, 1, 1),
                );
            }
            (_, Stage::FfnDown) => {
                encode_gemv(enc, &ffn.down, (&s.ffn_act, 0), (&s.resid, 0), t_len, true);
            }
            // [`Self::layer_stages`] never pairs a stage with the other
            // layer kind; a caller that does is refused, not skipped.
            (LayerKind::FullAttention(_), other) => {
                return Err(encoding_failed(format!(
                    "stage {other:?} does not belong to full-attention layer {layer}"
                )));
            }
            (LayerKind::LinearAttention(_), other) => {
                return Err(encoding_failed(format!(
                    "stage {other:?} does not belong to Gated-DeltaNet layer {layer}"
                )));
            }
        }
        Ok(())
    }

    /// Store the chunk's keys/values in the layer's KV slot, then for every
    /// token score its query against positions `0..=start_pos + t`, softmax
    /// and take the weighted sum of values (GQA, `f16` cache).
    fn encode_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize,
        t_len: usize,
        start_pos: usize,
    ) -> Result<(), MetalGraphError> {
        let s = &self.scratch;
        let c = &self.cfg;
        let slot_offset = self.kv_slot(layer)? * (c.n_kv_heads * c.max_seq_len * c.head_dim) as u64;
        let (nq, nkv, hd) = (c.n_heads as u32, c.n_kv_heads as u32, c.head_dim as u32);
        let heads_per_group = nq / nkv;
        let max_seq = c.max_seq_len as u32;
        let inv_sqrt = 1.0f32 / (c.head_dim as f32).sqrt();

        enc.set_compute_pipeline_state(&self.pipes.kv_store);
        enc.set_buffer(0, Some(&s.k_rope), 0);
        enc.set_buffer(1, Some(&s.v), 0);
        enc.set_buffer(2, Some(&self.k_cache), 0);
        enc.set_buffer(3, Some(&self.v_cache), 0);
        set_u32(enc, 4, hd);
        set_u32(enc, 5, nkv);
        set_u32(enc, 6, max_seq);
        set_u32(enc, 7, start_pos as u32);
        set_u64(enc, 8, slot_offset);
        enc.dispatch_thread_groups(
            MTLSize::new(c.head_dim.div_ceil(64) as u64, nkv as u64, t_len as u64),
            MTLSize::new(64, 1, 1),
        );

        let q_row_bytes = (c.heads_width() * 4) as u64;
        for t in 0..t_len {
            let seq_len = (start_pos + t + 1) as u32;
            enc.set_compute_pipeline_state(&self.pipes.scores);
            enc.set_buffer(0, Some(&s.q_rope), t as u64 * q_row_bytes);
            enc.set_buffer(1, Some(&self.k_cache), 0);
            enc.set_buffer(2, Some(&self.scores), 0);
            set_u32(enc, 3, hd);
            set_u32(enc, 4, nq);
            set_u32(enc, 5, nkv);
            set_u32(enc, 6, heads_per_group);
            set_u32(enc, 7, max_seq);
            set_u32(enc, 8, seq_len);
            set_f32(enc, 9, inv_sqrt);
            set_u64(enc, 10, slot_offset);
            set_u32(enc, 11, SCORES_BATCH_STRIDE);
            enc.dispatch_thread_groups(
                MTLSize::new(nq as u64, seq_len.div_ceil(SCORES_BATCH_STRIDE) as u64, 1),
                MTLSize::new(ATTENTION_SCORES_V2_THREADS, 1, 1),
            );

            enc.set_compute_pipeline_state(&self.pipes.softmax);
            enc.set_buffer(0, Some(&self.scores), 0);
            set_u32(enc, 1, nq);
            set_u32(enc, 2, max_seq);
            set_u32(enc, 3, seq_len);
            enc.dispatch_thread_groups(MTLSize::new(nq as u64, 1, 1), MTLSize::new(256, 1, 1));

            enc.set_compute_pipeline_state(&self.pipes.weighted_sum);
            enc.set_buffer(0, Some(&self.scores), 0);
            enc.set_buffer(1, Some(&self.v_cache), 0);
            enc.set_buffer(2, Some(&s.attn), t as u64 * q_row_bytes);
            set_u32(enc, 3, hd);
            set_u32(enc, 4, nq);
            set_u32(enc, 5, nkv);
            set_u32(enc, 6, heads_per_group);
            set_u32(enc, 7, max_seq);
            set_u32(enc, 8, seq_len);
            set_u64(enc, 9, slot_offset);
            enc.dispatch_thread_groups(
                MTLSize::new(c.head_dim.div_ceil(64) as u64, nq as u64, 1),
                MTLSize::new(64, 1, 1),
            );
        }
        Ok(())
    }

    /// Final RMSNorm (+ rotation) of the last token and the LM head.
    pub(super) fn encode_head(
        &self,
        enc: &ComputeCommandEncoderRef,
        t_len: usize,
    ) -> Result<(), MetalGraphError> {
        let last = (t_len.saturating_sub(1) * self.cfg.hidden * 4) as u64;
        self.encode_norm(enc, &self.output_norm, 1, last)?;
        encode_gemv(
            enc,
            &self.lm_head,
            (self.folded_input(), 0),
            (&self.logits, 0),
            1,
            false,
        );
        Ok(())
    }
}
