//! TQ2 ternary CUDA path helpers and dedicated batch-prefill methods.

use super::super::{BonsaiModel, OutputWeight};
use crate::block::blocks_as_bytes_ternary;

impl<'a> BonsaiModel<'a> {
    /// Build per-layer ternary QKV byte concatenations for the CUDA ternary path.
    ///
    /// Each layer's Q, K, V TQ2 block bytes are concatenated in that order.
    /// Built fresh on each call (no caching needed — the GPU weight cache handles
    /// upload deduplication via handle IDs).
    pub(super) fn build_cuda_ternary_qkv_concats(
        &self,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let n_layers = self.blocks.len();
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes = blocks_as_bytes_ternary(
                block
                    .attn_q_blocks_ternary()
                    .ok_or("attn_q: not a ternary layer")?,
            );
            let k_bytes = blocks_as_bytes_ternary(
                block
                    .attn_k_blocks_ternary()
                    .ok_or("attn_k: not a ternary layer")?,
            );
            let v_bytes = blocks_as_bytes_ternary(
                block
                    .attn_v_blocks_ternary()
                    .ok_or("attn_v: not a ternary layer")?,
            );
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        Ok(qkv_concats)
    }

    /// Build per-layer `CudaFullForwardLayerParamsTernary` for the CUDA ternary path.
    ///
    /// Handle namespaces (distinct from Q1 CUDA ranges 1M–4M and CUDA ternary norms 5M):
    ///   norm    handles: `5_000_000 + layer * 10 + offset`
    ///   weight  handles: `6_000_000 + layer * 10 + offset`
    pub(super) fn build_cuda_ternary_layer_params<'b>(
        &'b self,
        qkv_concats: &'b [Vec<u8>],
    ) -> Result<
        Vec<oxibonsai_kernels::CudaFullForwardLayerParamsTernary<'b>>,
        Box<dyn std::error::Error>,
    > {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let mut layer_params: Vec<oxibonsai_kernels::CudaFullForwardLayerParamsTernary<'b>> =
            Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 5_000_000u64 + (block.layer_index() as u64) * 10;
            let weight_handle_base = 6_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(oxibonsai_kernels::CudaFullForwardLayerParamsTernary {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: weight_handle_base,
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: weight_handle_base + 1,
                attn_proj_bytes: blocks_as_bytes_ternary(
                    block
                        .attn_output_blocks_ternary()
                        .ok_or("attn_output: not a ternary layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: weight_handle_base + 2,
                gate_bytes: blocks_as_bytes_ternary(
                    block
                        .ffn_gate_blocks_ternary()
                        .ok_or("ffn_gate: not a ternary layer")?,
                ),
                up_bytes: blocks_as_bytes_ternary(
                    block
                        .ffn_up_blocks_ternary()
                        .ok_or("ffn_up: not a ternary layer")?,
                ),
                down_handle: weight_handle_base + 3,
                down_bytes: blocks_as_bytes_ternary(
                    block
                        .ffn_down_blocks_ternary()
                        .ok_or("ffn_down: not a ternary layer")?,
                ),
            });
        }
        Ok(layer_params)
    }

    /// GPU batch prefill for TQ2 ternary models (CUDA): all layers + final norm + LM head.
    ///
    /// Returns the last token's logits.  Mirrors `try_cuda_prefill_with_lm_head` but
    /// uses TQ2 GEMM/GEMV kernels throughout.
    pub(super) fn try_cuda_prefill_with_lm_head_ternary(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_ternary = match &self.output_weight {
            OutputWeight::Ternary(ref t) => t,
            _ => return Err("try_cuda_prefill_with_lm_head_ternary: not a ternary model".into()),
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
        let heads_per_group = nq.checked_div(nkv).unwrap_or(1);
        let max_seq_len = self.kv_cache.max_seq_len();

        let mut hidden_batch = vec![0.0f32; batch_size * h];
        for (t, &token_id) in token_ids.iter().enumerate() {
            let embd_start = token_id as usize * h;
            let embd_end = embd_start + h;
            if embd_end > self.token_embd.len() {
                return Err(format!(
                    "token_id {} out of range (vocab={})",
                    token_id,
                    self.token_embd.len() / h
                )
                .into());
            }
            hidden_batch[t * h..(t + 1) * h]
                .copy_from_slice(&self.token_embd[embd_start..embd_end]);
        }

        let mut cos_table = vec![0.0f32; batch_size * half_dim];
        let mut sin_table = vec![0.0f32; batch_size * half_dim];
        for t in 0..batch_size {
            let pos = pos_start + t;
            let cos_vals = self.rope.cos_at(pos);
            let sin_vals = self.rope.sin_at(pos);
            cos_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(cos_vals);
            sin_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(sin_vals);
        }

        let final_norm_handle = 5_900_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 7_000_000u64;
        let lm_head_bytes = blocks_as_bytes_ternary(lm_head_ternary.blocks());
        let lm_head_out_features = lm_head_ternary.out_features();

        let qkv_concats = self.build_cuda_ternary_qkv_concats()?;
        let layer_params = self.build_cuda_ternary_layer_params(&qkv_concats)?;

        let mut logits = vec![0.0f32; lm_head_out_features];
        oxibonsai_kernels::try_cuda_prefill_ternary(
            &hidden_batch,
            batch_size,
            pos_start,
            n_layers,
            &layer_params,
            &cos_table,
            &sin_table,
            h,
            inter,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            Some(&mut logits),
            None,
        )
        .map_err(|e| {
            tracing::warn!(error = %e, "CUDA ternary batch prefill failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(logits)
    }

    /// GPU batch prefill verify for TQ2 ternary models (CUDA): greedy argmax per position.
    ///
    /// Returns the greedy argmax token ID for each input position.
    pub(super) fn try_cuda_prefill_verify_ternary(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_ternary = match &self.output_weight {
            OutputWeight::Ternary(ref t) => t,
            _ => return Err("try_cuda_prefill_verify_ternary: not a ternary model".into()),
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let heads_per_group = nq.checked_div(nkv).unwrap_or(1);
        let max_seq_len = self.kv_cache.max_seq_len();

        let final_norm_handle = 5_900_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 7_000_000u64;
        let lm_head_bytes = blocks_as_bytes_ternary(lm_head_ternary.blocks());
        let lm_head_out_features = lm_head_ternary.out_features();

        let qkv_concats = self.build_cuda_ternary_qkv_concats()?;
        let layer_params = self.build_cuda_ternary_layer_params(&qkv_concats)?;

        let mut token_ids_out: Vec<u32> = Vec::with_capacity(batch_size);
        for (t, &tok_id) in token_ids.iter().enumerate().take(batch_size) {
            let embd_start = tok_id as usize * h;
            if embd_start + h > self.token_embd.len() {
                return Err(format!(
                    "token_id {} out of range (vocab={})",
                    tok_id,
                    self.token_embd.len() / h
                )
                .into());
            }
            let single_hidden = self.token_embd[embd_start..embd_start + h].to_vec();
            let pos = pos_start + t;
            let cos_single = self.rope.cos_at(pos);
            let sin_single = self.rope.sin_at(pos);

            let mut greedy_id: u32 = 0;
            oxibonsai_kernels::try_cuda_prefill_ternary(
                &single_hidden,
                1,
                pos,
                n_layers,
                &layer_params,
                cos_single,
                sin_single,
                h,
                inter,
                nq,
                nkv,
                hd,
                heads_per_group,
                eps,
                max_seq_len,
                Some(final_norm_handle),
                Some(final_norm_bytes),
                final_norm_eps,
                Some(lm_head_handle),
                Some(lm_head_bytes),
                lm_head_out_features,
                None,
                Some(&mut greedy_id),
            )
            .map_err(|e| {
                tracing::warn!(error = %e, "CUDA ternary prefill verify at pos {pos}: {e}");
                Box::new(e) as Box<dyn std::error::Error>
            })?;
            token_ids_out.push(greedy_id);
        }
        Ok(token_ids_out)
    }
}
