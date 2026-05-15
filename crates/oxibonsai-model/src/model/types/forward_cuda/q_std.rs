//! Q4_0 / Q8_0 standard-quant CUDA batch-prefill helpers and entry points.

use super::byte_helpers::{blocks_q4_0_as_bytes, blocks_q8_0_as_bytes};
use super::super::{BonsaiModel, OutputWeight};

impl<'a> BonsaiModel<'a> {
    /// Convert a Q4_0 block slice to raw bytes (zero-copy, file-level fn below).
    ///
    /// GPU batch prefill for Q4_0/Q8_0 models (CUDA): all layers + final norm + LM head.
    ///
    /// Returns the last token's logits.  Uses `try_cuda_prefill_q_std` (Phase 24A).
    pub(super) fn build_cuda_q_std_qkv_concats(
        &self,
        q4_0: bool,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let n_layers = self.blocks.len();
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let (q_bytes, k_bytes, v_bytes) = if q4_0 {
                (
                    blocks_q4_0_as_bytes(block.attn_q_blocks_q4_0().ok_or("attn_q: not Q4_0")?),
                    blocks_q4_0_as_bytes(block.attn_k_blocks_q4_0().ok_or("attn_k: not Q4_0")?),
                    blocks_q4_0_as_bytes(block.attn_v_blocks_q4_0().ok_or("attn_v: not Q4_0")?),
                )
            } else {
                (
                    blocks_q8_0_as_bytes(block.attn_q_blocks_q8_0().ok_or("attn_q: not Q8_0")?),
                    blocks_q8_0_as_bytes(block.attn_k_blocks_q8_0().ok_or("attn_k: not Q8_0")?),
                    blocks_q8_0_as_bytes(block.attn_v_blocks_q8_0().ok_or("attn_v: not Q8_0")?),
                )
            };
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        Ok(qkv_concats)
    }

    /// Build per-layer `CudaQStdPrefillLayerParams` for the Q4_0/Q8_0 CUDA path.
    ///
    /// Handle namespaces (distinct from all existing ranges 1M–7M):
    ///   Q4_0 norm handles:   `8_000_000 + layer * 10 + offset`
    ///   Q4_0 weight handles: `9_000_000 + layer * 10 + offset`
    ///   Q8_0 norm handles:   `10_000_000 + layer * 10 + offset`
    ///   Q8_0 weight handles: `11_000_000 + layer * 10 + offset`
    ///
    /// Per-layer offsets: +0=attn_norm, +1=q_norm, +2=k_norm, +3=ffn_norm
    ///                    +0=fused_qkv, +1=attn_proj, +2=gate_up, +3=down
    pub(super) fn build_cuda_q_std_layer_params<'b>(
        &'b self,
        qkv_concats: &'b [Vec<u8>],
        q4_0: bool,
    ) -> Result<Vec<oxibonsai_kernels::CudaQStdPrefillLayerParams<'b>>, Box<dyn std::error::Error>>
    {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let norm_base_offset = if q4_0 { 8_000_000u64 } else { 10_000_000u64 };
        let weight_base_offset = if q4_0 { 9_000_000u64 } else { 11_000_000u64 };
        let mut layer_params: Vec<oxibonsai_kernels::CudaQStdPrefillLayerParams<'b>> =
            Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = norm_base_offset + (block.layer_index() as u64) * 10;
            let weight_handle_base = weight_base_offset + (block.layer_index() as u64) * 10;
            let (attn_proj_bytes, gate_bytes, up_bytes, down_bytes) = if q4_0 {
                (
                    blocks_q4_0_as_bytes(
                        block
                            .attn_output_blocks_q4_0()
                            .ok_or("attn_output: not Q4_0")?,
                    ),
                    blocks_q4_0_as_bytes(block.ffn_gate_blocks_q4_0().ok_or("ffn_gate: not Q4_0")?),
                    blocks_q4_0_as_bytes(block.ffn_up_blocks_q4_0().ok_or("ffn_up: not Q4_0")?),
                    blocks_q4_0_as_bytes(block.ffn_down_blocks_q4_0().ok_or("ffn_down: not Q4_0")?),
                )
            } else {
                (
                    blocks_q8_0_as_bytes(
                        block
                            .attn_output_blocks_q8_0()
                            .ok_or("attn_output: not Q8_0")?,
                    ),
                    blocks_q8_0_as_bytes(block.ffn_gate_blocks_q8_0().ok_or("ffn_gate: not Q8_0")?),
                    blocks_q8_0_as_bytes(block.ffn_up_blocks_q8_0().ok_or("ffn_up: not Q8_0")?),
                    blocks_q8_0_as_bytes(block.ffn_down_blocks_q8_0().ok_or("ffn_down: not Q8_0")?),
                )
            };
            layer_params.push(oxibonsai_kernels::CudaQStdPrefillLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: weight_handle_base,
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: weight_handle_base + 1,
                attn_proj_bytes,
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: weight_handle_base + 2,
                gate_bytes,
                up_bytes,
                down_handle: weight_handle_base + 3,
                down_bytes,
                q4_0,
            });
        }
        Ok(layer_params)
    }

    /// GPU batch prefill for Q4_0/Q8_0 models (CUDA): all layers + final norm + LM head.
    ///
    /// Returns the last token's logits.  Dispatches to `try_cuda_prefill_q_std` (Phase 24A).
    pub(in super::super) fn try_cuda_prefill_with_lm_head_q_std(
        &self,
        token_ids: &[u32],
        pos_start: usize,
        q4_0: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
        let heads_per_group = nq.checked_div(nkv).unwrap_or(1);
        let max_seq_len = self.kv_cache.max_seq_len();

        // Build flattened hidden_batch from token embeddings
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

        // Build RoPE tables for the full batch
        let mut cos_table = vec![0.0f32; batch_size * half_dim];
        let mut sin_table = vec![0.0f32; batch_size * half_dim];
        for t in 0..batch_size {
            let pos = pos_start + t;
            let cos_vals = self.rope.cos_at(pos);
            let sin_vals = self.rope.sin_at(pos);
            cos_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(cos_vals);
            sin_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(sin_vals);
        }

        // Handle namespaces for final norm + LM head
        let final_norm_handle = if q4_0 { 8_900_000u64 } else { 10_900_000u64 };
        let lm_head_handle = if q4_0 { 9_900_000u64 } else { 11_900_000u64 };
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();

        let (lm_head_bytes, lm_head_out_features) = match &self.output_weight {
            OutputWeight::Q4_0(ref linear) if q4_0 => {
                (blocks_q4_0_as_bytes(linear.blocks()), linear.out_features())
            }
            OutputWeight::Q8_0(ref linear) if !q4_0 => {
                (blocks_q8_0_as_bytes(linear.blocks()), linear.out_features())
            }
            _ => {
                return Err(format!(
                    "try_cuda_prefill_with_lm_head_q_std: LM head quant mismatch (q4_0={})",
                    q4_0
                )
                .into())
            }
        };

        let qkv_concats = self.build_cuda_q_std_qkv_concats(q4_0)?;
        let layer_params = self.build_cuda_q_std_layer_params(&qkv_concats, q4_0)?;

        let mut logits = vec![0.0f32; lm_head_out_features];
        oxibonsai_kernels::try_cuda_prefill_q_std(
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
            q4_0,
            Some(&mut logits),
            None,
        )
        .map_err(|e| {
            tracing::warn!(error = %e, "CUDA Q4_0/Q8_0 batch prefill dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(logits)
    }

    /// GPU batch prefill verify for Q4_0/Q8_0 models (CUDA): greedy argmax per position.
    ///
    /// Returns the greedy argmax token ID for each input position.
    /// Dispatches to `try_cuda_prefill_q_std` with `greedy_token_id_out` set (Phase 24A).
    pub(super) fn try_cuda_prefill_verify_q_std(
        &self,
        token_ids: &[u32],
        pos_start: usize,
        q4_0: bool,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let heads_per_group = nq.checked_div(nkv).unwrap_or(1);
        let max_seq_len = self.kv_cache.max_seq_len();

        let final_norm_handle = if q4_0 { 8_900_000u64 } else { 10_900_000u64 };
        let lm_head_handle = if q4_0 { 9_900_000u64 } else { 11_900_000u64 };
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();

        let (lm_head_bytes, lm_head_out_features) = match &self.output_weight {
            OutputWeight::Q4_0(ref linear) if q4_0 => {
                (blocks_q4_0_as_bytes(linear.blocks()), linear.out_features())
            }
            OutputWeight::Q8_0(ref linear) if !q4_0 => {
                (blocks_q8_0_as_bytes(linear.blocks()), linear.out_features())
            }
            _ => {
                return Err(format!(
                    "try_cuda_prefill_verify_q_std: LM head quant mismatch (q4_0={})",
                    q4_0
                )
                .into())
            }
        };

        let qkv_concats = self.build_cuda_q_std_qkv_concats(q4_0)?;
        let layer_params = self.build_cuda_q_std_layer_params(&qkv_concats, q4_0)?;

        let mut token_ids_out: Vec<u32> = Vec::with_capacity(batch_size);
        for (t, &tok_id) in token_ids.iter().enumerate() {
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
            oxibonsai_kernels::try_cuda_prefill_q_std(
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
                q4_0,
                None,
                Some(&mut greedy_id),
            )
            .map_err(|e| {
                tracing::warn!(
                    error = %e,
                    "CUDA Q4_0/Q8_0 prefill verify dispatch failed at pos {pos}"
                );
                Box::new(e) as Box<dyn std::error::Error>
            })?;
            token_ids_out.push(greedy_id);
        }
        Ok(token_ids_out)
    }
}
