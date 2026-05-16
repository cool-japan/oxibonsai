//! K-quant (Q2K / Q3K / Q4K / Q5K / Q6K / Q8K) CUDA batch-prefill helpers and
//! entry points (Phase 25).

use super::super::{BonsaiModel, OutputWeight};
use super::byte_helpers::{
    blocks_q2k_as_bytes, blocks_q3k_as_bytes, blocks_q4k_as_bytes, blocks_q5k_as_bytes,
    blocks_q6k_as_bytes, blocks_q8k_as_bytes,
};

impl<'a> BonsaiModel<'a> {
    // ── Phase 25: K-quant batch GEMM prefill ─────────────────────────────────

    /// GPU batch prefill for K-quant models: all layers + final norm + LM head.
    ///
    /// Returns the last token's logits.  Dispatches to `try_cuda_prefill_k_quant` (Phase 25).
    pub(in super::super) fn try_cuda_prefill_with_lm_head_k_quant(
        &self,
        token_ids: &[u32],
        pos_start: usize,
        fmt: oxibonsai_kernels::KQuantFormat,
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

        // Handle namespaces for final norm + LM head (Phase 25 ranges)
        let format_offset: u64 = match fmt {
            oxibonsai_kernels::KQuantFormat::Q2K => 0,
            oxibonsai_kernels::KQuantFormat::Q3K => 1_000_000,
            oxibonsai_kernels::KQuantFormat::Q4K => 2_000_000,
            oxibonsai_kernels::KQuantFormat::Q5K => 3_000_000,
            oxibonsai_kernels::KQuantFormat::Q6K => 4_000_000,
            oxibonsai_kernels::KQuantFormat::Q8K => 5_000_000,
        };
        let final_norm_handle = 24_000_000u64 + format_offset;
        let lm_head_handle = 25_000_000u64 + format_offset;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();

        let (lm_head_bytes, lm_head_out_features): (&[u8], usize) = match &self.output_weight {
            OutputWeight::Q2K(lm) => (blocks_q2k_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::Q3K(lm) => (blocks_q3k_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::Q4K(lm) => (blocks_q4k_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::Q5K(lm) => (blocks_q5k_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::Q6K(lm) => (blocks_q6k_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::Q8K(lm) => (blocks_q8k_as_bytes(lm.blocks()), lm.out_features()),
            _ => return Err("K-quant LM head expected".into()),
        };

        let qkv_concats = self.build_cuda_k_quant_qkv_concats(fmt)?;
        let layer_params = self.build_cuda_k_quant_layer_params(&qkv_concats, fmt)?;

        let mut logits = vec![0.0f32; lm_head_out_features];
        oxibonsai_kernels::try_cuda_prefill_k_quant(
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
            fmt,
            Some(&mut logits),
            None,
        )
        .map_err(|e| {
            tracing::warn!(error = %e, "CUDA K-quant batch prefill dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(logits)
    }

    /// GPU batch prefill verify for K-quant models (CUDA): greedy argmax per position.
    ///
    /// Returns the greedy argmax token ID for each input position.
    /// Dispatches to `try_cuda_prefill_k_quant` with `greedy_token_id_out` set (Phase 25).
    pub(in super::super) fn try_cuda_prefill_verify_k_quant(
        &self,
        token_ids: &[u32],
        pos_start: usize,
        fmt: oxibonsai_kernels::KQuantFormat,
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

        let format_offset: u64 = match fmt {
            oxibonsai_kernels::KQuantFormat::Q2K => 0,
            oxibonsai_kernels::KQuantFormat::Q3K => 1_000_000,
            oxibonsai_kernels::KQuantFormat::Q4K => 2_000_000,
            oxibonsai_kernels::KQuantFormat::Q5K => 3_000_000,
            oxibonsai_kernels::KQuantFormat::Q6K => 4_000_000,
            oxibonsai_kernels::KQuantFormat::Q8K => 5_000_000,
        };
        let final_norm_handle = 24_000_000u64 + format_offset;
        let lm_head_handle = 25_000_000u64 + format_offset;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();

        let (lm_head_bytes, lm_head_out_features): (&[u8], usize) = match &self.output_weight {
            OutputWeight::Q2K(lm) => (blocks_q2k_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::Q3K(lm) => (blocks_q3k_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::Q4K(lm) => (blocks_q4k_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::Q5K(lm) => (blocks_q5k_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::Q6K(lm) => (blocks_q6k_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::Q8K(lm) => (blocks_q8k_as_bytes(lm.blocks()), lm.out_features()),
            _ => return Err("K-quant LM head expected".into()),
        };

        let qkv_concats = self.build_cuda_k_quant_qkv_concats(fmt)?;
        let layer_params = self.build_cuda_k_quant_layer_params(&qkv_concats, fmt)?;

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
            oxibonsai_kernels::try_cuda_prefill_k_quant(
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
                fmt,
                None,
                Some(&mut greedy_id),
            )
            .map_err(|e| {
                tracing::warn!(
                    error = %e,
                    "CUDA K-quant prefill verify dispatch failed at pos {pos}"
                );
                Box::new(e) as Box<dyn std::error::Error>
            })?;
            token_ids_out.push(greedy_id);
        }
        Ok(token_ids_out)
    }

    /// Build per-layer QKV byte concatenations for the K-quant CUDA path.
    ///
    /// Each layer's Q, K, V block bytes are concatenated in that order.
    pub(super) fn build_cuda_k_quant_qkv_concats(
        &self,
        fmt: oxibonsai_kernels::KQuantFormat,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let n_layers = self.blocks.len();
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let (q_bytes, k_bytes, v_bytes): (&[u8], &[u8], &[u8]) = match fmt {
                oxibonsai_kernels::KQuantFormat::Q2K => (
                    blocks_q2k_as_bytes(block.attn_q_blocks_q2k().ok_or("attn_q: not Q2_K")?),
                    blocks_q2k_as_bytes(block.attn_k_blocks_q2k().ok_or("attn_k: not Q2_K")?),
                    blocks_q2k_as_bytes(block.attn_v_blocks_q2k().ok_or("attn_v: not Q2_K")?),
                ),
                oxibonsai_kernels::KQuantFormat::Q3K => (
                    blocks_q3k_as_bytes(block.attn_q_blocks_q3k().ok_or("attn_q: not Q3_K")?),
                    blocks_q3k_as_bytes(block.attn_k_blocks_q3k().ok_or("attn_k: not Q3_K")?),
                    blocks_q3k_as_bytes(block.attn_v_blocks_q3k().ok_or("attn_v: not Q3_K")?),
                ),
                oxibonsai_kernels::KQuantFormat::Q4K => (
                    blocks_q4k_as_bytes(block.attn_q_blocks_q4k().ok_or("attn_q: not Q4_K")?),
                    blocks_q4k_as_bytes(block.attn_k_blocks_q4k().ok_or("attn_k: not Q4_K")?),
                    blocks_q4k_as_bytes(block.attn_v_blocks_q4k().ok_or("attn_v: not Q4_K")?),
                ),
                oxibonsai_kernels::KQuantFormat::Q5K => (
                    blocks_q5k_as_bytes(block.attn_q_blocks_q5k().ok_or("attn_q: not Q5_K")?),
                    blocks_q5k_as_bytes(block.attn_k_blocks_q5k().ok_or("attn_k: not Q5_K")?),
                    blocks_q5k_as_bytes(block.attn_v_blocks_q5k().ok_or("attn_v: not Q5_K")?),
                ),
                oxibonsai_kernels::KQuantFormat::Q6K => (
                    blocks_q6k_as_bytes(block.attn_q_blocks_q6k().ok_or("attn_q: not Q6_K")?),
                    blocks_q6k_as_bytes(block.attn_k_blocks_q6k().ok_or("attn_k: not Q6_K")?),
                    blocks_q6k_as_bytes(block.attn_v_blocks_q6k().ok_or("attn_v: not Q6_K")?),
                ),
                oxibonsai_kernels::KQuantFormat::Q8K => (
                    blocks_q8k_as_bytes(block.attn_q_blocks_q8k().ok_or("attn_q: not Q8_K")?),
                    blocks_q8k_as_bytes(block.attn_k_blocks_q8k().ok_or("attn_k: not Q8_K")?),
                    blocks_q8k_as_bytes(block.attn_v_blocks_q8k().ok_or("attn_v: not Q8_K")?),
                ),
            };
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        Ok(qkv_concats)
    }

    /// Build per-layer `CudaKQuantPrefillLayerParams` for the K-quant CUDA path.
    ///
    /// Handle namespaces (non-overlapping with all existing ranges 1M–23M):
    ///   Q2K norms: 12_000_000 + layer*10,  weights: 13_000_000 + layer*10
    ///   Q3K norms: 14_000_000 + layer*10,  weights: 15_000_000 + layer*10
    ///   Q4K norms: 16_000_000 + layer*10,  weights: 17_000_000 + layer*10
    ///   Q5K norms: 18_000_000 + layer*10,  weights: 19_000_000 + layer*10
    ///   Q6K norms: 20_000_000 + layer*10,  weights: 21_000_000 + layer*10
    ///   Q8K norms: 22_000_000 + layer*10,  weights: 23_000_000 + layer*10
    pub(super) fn build_cuda_k_quant_layer_params<'b>(
        &'b self,
        qkv_concats: &'b [Vec<u8>],
        fmt: oxibonsai_kernels::KQuantFormat,
    ) -> Result<Vec<oxibonsai_kernels::CudaKQuantPrefillLayerParams<'b>>, Box<dyn std::error::Error>>
    {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let (norm_base, weight_base): (u64, u64) = match fmt {
            oxibonsai_kernels::KQuantFormat::Q2K => (12_000_000, 13_000_000),
            oxibonsai_kernels::KQuantFormat::Q3K => (14_000_000, 15_000_000),
            oxibonsai_kernels::KQuantFormat::Q4K => (16_000_000, 17_000_000),
            oxibonsai_kernels::KQuantFormat::Q5K => (18_000_000, 19_000_000),
            oxibonsai_kernels::KQuantFormat::Q6K => (20_000_000, 21_000_000),
            oxibonsai_kernels::KQuantFormat::Q8K => (22_000_000, 23_000_000),
        };
        let mut layer_params: Vec<oxibonsai_kernels::CudaKQuantPrefillLayerParams<'b>> =
            Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = norm_base + (block.layer_index() as u64) * 10;
            let weight_handle_base = weight_base + (block.layer_index() as u64) * 10;
            let (attn_proj_bytes, gate_bytes, up_bytes, down_bytes): (&[u8], &[u8], &[u8], &[u8]) =
                match fmt {
                    oxibonsai_kernels::KQuantFormat::Q2K => (
                        blocks_q2k_as_bytes(
                            block
                                .attn_output_blocks_q2k()
                                .ok_or("attn_output: not Q2_K")?,
                        ),
                        blocks_q2k_as_bytes(
                            block.ffn_gate_blocks_q2k().ok_or("ffn_gate: not Q2_K")?,
                        ),
                        blocks_q2k_as_bytes(block.ffn_up_blocks_q2k().ok_or("ffn_up: not Q2_K")?),
                        blocks_q2k_as_bytes(
                            block.ffn_down_blocks_q2k().ok_or("ffn_down: not Q2_K")?,
                        ),
                    ),
                    oxibonsai_kernels::KQuantFormat::Q3K => (
                        blocks_q3k_as_bytes(
                            block
                                .attn_output_blocks_q3k()
                                .ok_or("attn_output: not Q3_K")?,
                        ),
                        blocks_q3k_as_bytes(
                            block.ffn_gate_blocks_q3k().ok_or("ffn_gate: not Q3_K")?,
                        ),
                        blocks_q3k_as_bytes(block.ffn_up_blocks_q3k().ok_or("ffn_up: not Q3_K")?),
                        blocks_q3k_as_bytes(
                            block.ffn_down_blocks_q3k().ok_or("ffn_down: not Q3_K")?,
                        ),
                    ),
                    oxibonsai_kernels::KQuantFormat::Q4K => (
                        blocks_q4k_as_bytes(
                            block
                                .attn_output_blocks_q4k()
                                .ok_or("attn_output: not Q4_K")?,
                        ),
                        blocks_q4k_as_bytes(
                            block.ffn_gate_blocks_q4k().ok_or("ffn_gate: not Q4_K")?,
                        ),
                        blocks_q4k_as_bytes(block.ffn_up_blocks_q4k().ok_or("ffn_up: not Q4_K")?),
                        blocks_q4k_as_bytes(
                            block.ffn_down_blocks_q4k().ok_or("ffn_down: not Q4_K")?,
                        ),
                    ),
                    oxibonsai_kernels::KQuantFormat::Q5K => (
                        blocks_q5k_as_bytes(
                            block
                                .attn_output_blocks_q5k()
                                .ok_or("attn_output: not Q5_K")?,
                        ),
                        blocks_q5k_as_bytes(
                            block.ffn_gate_blocks_q5k().ok_or("ffn_gate: not Q5_K")?,
                        ),
                        blocks_q5k_as_bytes(block.ffn_up_blocks_q5k().ok_or("ffn_up: not Q5_K")?),
                        blocks_q5k_as_bytes(
                            block.ffn_down_blocks_q5k().ok_or("ffn_down: not Q5_K")?,
                        ),
                    ),
                    oxibonsai_kernels::KQuantFormat::Q6K => (
                        blocks_q6k_as_bytes(
                            block
                                .attn_output_blocks_q6k()
                                .ok_or("attn_output: not Q6_K")?,
                        ),
                        blocks_q6k_as_bytes(
                            block.ffn_gate_blocks_q6k().ok_or("ffn_gate: not Q6_K")?,
                        ),
                        blocks_q6k_as_bytes(block.ffn_up_blocks_q6k().ok_or("ffn_up: not Q6_K")?),
                        blocks_q6k_as_bytes(
                            block.ffn_down_blocks_q6k().ok_or("ffn_down: not Q6_K")?,
                        ),
                    ),
                    oxibonsai_kernels::KQuantFormat::Q8K => (
                        blocks_q8k_as_bytes(
                            block
                                .attn_output_blocks_q8k()
                                .ok_or("attn_output: not Q8_K")?,
                        ),
                        blocks_q8k_as_bytes(
                            block.ffn_gate_blocks_q8k().ok_or("ffn_gate: not Q8_K")?,
                        ),
                        blocks_q8k_as_bytes(block.ffn_up_blocks_q8k().ok_or("ffn_up: not Q8_K")?),
                        blocks_q8k_as_bytes(
                            block.ffn_down_blocks_q8k().ok_or("ffn_down: not Q8_K")?,
                        ),
                    ),
                };
            layer_params.push(oxibonsai_kernels::CudaKQuantPrefillLayerParams {
                format: fmt,
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
            });
        }
        Ok(layer_params)
    }
}
