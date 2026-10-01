//! Head-free Metal prefill for dense embeddings: the GPU route of
//! [`BonsaiModel::forward_hidden`].
//!
//! [`BonsaiModel::try_metal_forward_hidden`] gathers the prompt's embedding
//! rows, builds their RoPE tables and hands both to
//! `oxibonsai_kernels::gpu_backend::try_metal_full_forward_prefill_hidden_cached`
//! together with the model's **cached** fused weight set — the same GPU
//! buffers the fused decode and the batched logits prefill bind. The kernels
//! run every layer over 128-row micro-batches and return the `output_norm`
//! output of every row, i.e. exactly what the per-token path feeds to the LM
//! head, without ever running the LM head.
//!
//! # No device KV state is touched (MET-05)
//!
//! The pass runs in a request-scoped Metal session with its own device KV
//! cache sized to the input and freed on return; the KV cache of the session
//! this thread decodes in — the process-default one, for most callers — is
//! neither read nor written. Correspondingly this path never calls
//! `note_device_kv_used`: the model's MET-05 latch still describes whatever
//! generation state the model had before the embedding.
//!
//! Like the model's fused decode and prefill forwards (`forward_metal.rs`),
//! the pass runs inside an autorelease pool of its own, so an embedding
//! server thread keeps none of the request's autoreleased Metal objects once
//! the call returns.

use super::{BonsaiModel, OutputWeight};
use crate::error::{ModelError, ModelResult};
use oxibonsai_kernels::gpu_backend::{
    try_metal_full_forward_prefill_hidden_cached, with_autorelease_pool, HiddenPrefillInput,
    HiddenPrefillShape,
};
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::CachedModelWeights;

impl BonsaiModel<'_> {
    /// Final-normed hidden state of every token through the head-free Metal
    /// batch prefill, `[n_tokens × hidden_size]` row-major — or `Ok(None)`
    /// when this model or input is not served by that path:
    ///
    /// * `kernel` is not a GPU-accelerated dispatcher;
    /// * the model declares a sliding attention window (the batched kernels
    ///   compute full causal attention, M-17);
    /// * the LM head is neither 1-bit nor ternary, or the model has no
    ///   blocks;
    /// * the fused weight cache (`get_or_create_gpu_cache`) is not resident,
    ///   or holds the other format.
    ///
    /// Positions are `0..tokens.len()`. The caller validates the input (see
    /// `check_embedding_input`); this path checks only what it needs.
    ///
    /// # Errors
    ///
    /// A token past the vocabulary ([`ModelError::MissingTensor`]), a
    /// position past the RoPE table, or any Metal failure
    /// ([`ModelError::Internal`] naming it) — `forward_hidden` answers every
    /// error by running the batched CPU pass instead.
    pub fn try_metal_forward_hidden(
        &self,
        tokens: &[u32],
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Option<Vec<f32>>> {
        if tokens.is_empty()
            || !kernel.is_gpu_accelerated()
            || self.config.sliding_window.is_some()
            || self.blocks.is_empty()
        {
            return Ok(None);
        }
        let wants_q1 = match &self.output_weight {
            OutputWeight::OneBit(_) => true,
            OutputWeight::Ternary(_) => false,
            _ => return Ok(None),
        };
        let guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| ModelError::Internal(format!("gpu_weight_cache lock: {e}")))?;
        let cached = match guard.as_ref() {
            Some(cached @ CachedModelWeights::Q1(_)) if wants_q1 => cached,
            Some(cached @ CachedModelWeights::Ternary(_)) if !wants_q1 => cached,
            _ => return Ok(None),
        };
        let h = self.config.hidden_size;
        let half_dim = self.config.head_dim / 2;
        let n = tokens.len();
        let len = n.checked_mul(h).ok_or_else(|| ModelError::ShapeInvariant {
            tensor: "BonsaiModel::try_metal_forward_hidden".to_string(),
            expected: "n_tokens * hidden_size representable as usize".to_string(),
            actual: format!("{n} x {h}"),
        })?;
        let mut hidden_batch = vec![0.0f32; len];
        self.token_embd.copy_rows(tokens, &mut hidden_batch)?;
        let mut cos_table = vec![0.0f32; n * half_dim];
        let mut sin_table = vec![0.0f32; n * half_dim];
        for pos in 0..n {
            cos_table[pos * half_dim..(pos + 1) * half_dim]
                .copy_from_slice(self.rope.cos_at_checked(pos)?);
            sin_table[pos * half_dim..(pos + 1) * half_dim]
                .copy_from_slice(self.rope.sin_at_checked(pos)?);
        }
        let shape = HiddenPrefillShape {
            hidden_size: h,
            intermediate_size: self.config.intermediate_size,
            nq: self.config.num_attention_heads,
            nkv: self.config.num_kv_heads,
            head_dim: self.config.head_dim,
            eps: self.blocks[0].attn_norm_eps(),
            final_norm_eps: self.output_norm.eps(),
        };
        let mut rows = Vec::with_capacity(len);
        with_autorelease_pool(|| {
            try_metal_full_forward_prefill_hidden_cached(
                &HiddenPrefillInput {
                    hidden_batch: &hidden_batch,
                    cos_table: &cos_table,
                    sin_table: &sin_table,
                    batch_size: n,
                },
                cached,
                self.output_norm.weight(),
                &shape,
                &mut rows,
            )
        })
        .map_err(|e| ModelError::Internal(format!("Metal hidden-state prefill: {e}")))?;
        if rows.len() != len {
            return Err(ModelError::ShapeInvariant {
                tensor: "BonsaiModel::try_metal_forward_hidden rows".to_string(),
                expected: format!("{len} floats"),
                actual: format!("{} floats", rows.len()),
            });
        }
        Ok(Some(rows))
    }
}
