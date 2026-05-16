//! CUDA FP8 E4M3/E5M2 batch prefill methods for `BonsaiModel` (Phase 26).

use super::{BonsaiModel, OutputWeight};

// ── Byte-cast helpers ────────────────────────────────────────────────────────

fn fp8_e4m3_as_bytes(blocks: &[oxibonsai_core::BlockFP8E4M3]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
        )
    }
}

fn fp8_e5m2_as_bytes(blocks: &[oxibonsai_core::BlockFP8E5M2]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
        )
    }
}

impl<'a> BonsaiModel<'a> {
    /// Build Q+K+V FP8 byte concatenations per layer.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn build_fp8_qkv_concats(
        &self,
        is_e4m3: bool,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let mut concats = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (qb, kb, vb) = if is_e4m3 {
                let q = fp8_e4m3_as_bytes(
                    block
                        .attn_q_blocks_fp8e4m3()
                        .ok_or("attn_q: not FP8 E4M3")?,
                );
                let k = fp8_e4m3_as_bytes(
                    block
                        .attn_k_blocks_fp8e4m3()
                        .ok_or("attn_k: not FP8 E4M3")?,
                );
                let v = fp8_e4m3_as_bytes(
                    block
                        .attn_v_blocks_fp8e4m3()
                        .ok_or("attn_v: not FP8 E4M3")?,
                );
                (q, k, v)
            } else {
                let q = fp8_e5m2_as_bytes(
                    block
                        .attn_q_blocks_fp8e5m2()
                        .ok_or("attn_q: not FP8 E5M2")?,
                );
                let k = fp8_e5m2_as_bytes(
                    block
                        .attn_k_blocks_fp8e5m2()
                        .ok_or("attn_k: not FP8 E5M2")?,
                );
                let v = fp8_e5m2_as_bytes(
                    block
                        .attn_v_blocks_fp8e5m2()
                        .ok_or("attn_v: not FP8 E5M2")?,
                );
                (q, k, v)
            };
            let mut concat = Vec::with_capacity(qb.len() + kb.len() + vb.len());
            concat.extend_from_slice(qb);
            concat.extend_from_slice(kb);
            concat.extend_from_slice(vb);
            concats.push(concat);
        }
        Ok(concats)
    }

    /// Build per-layer gate+up concatenation buffers (owned).
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn build_fp8_gate_up_owners(
        &self,
        is_e4m3: bool,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let mut owners = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (gb, ub) = if is_e4m3 {
                let g = fp8_e4m3_as_bytes(
                    block
                        .ffn_gate_blocks_fp8e4m3()
                        .ok_or("ffn_gate: not FP8 E4M3")?,
                );
                let u = fp8_e4m3_as_bytes(
                    block
                        .ffn_up_blocks_fp8e4m3()
                        .ok_or("ffn_up: not FP8 E4M3")?,
                );
                (g, u)
            } else {
                let g = fp8_e5m2_as_bytes(
                    block
                        .ffn_gate_blocks_fp8e5m2()
                        .ok_or("ffn_gate: not FP8 E5M2")?,
                );
                let u = fp8_e5m2_as_bytes(
                    block
                        .ffn_up_blocks_fp8e5m2()
                        .ok_or("ffn_up: not FP8 E5M2")?,
                );
                (g, u)
            };
            let mut combined = Vec::with_capacity(gb.len() + ub.len());
            combined.extend_from_slice(gb);
            combined.extend_from_slice(ub);
            owners.push(combined);
        }
        Ok(owners)
    }

    /// GPU batch prefill for FP8 E4M3/E5M2 models: all layers + final norm + LM head.
    ///
    /// Returns the last token's logits.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    pub(super) fn try_cuda_prefill_with_lm_head_fp8(
        &self,
        token_ids: &[u32],
        pos_start: usize,
        is_e4m3: bool,
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

        // Build hidden batch from token embeddings.
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

        // Build RoPE tables.
        let mut cos_table = vec![0.0f32; batch_size * half_dim];
        let mut sin_table = vec![0.0f32; batch_size * half_dim];
        for t in 0..batch_size {
            let pos = pos_start + t;
            cos_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(self.rope.cos_at(pos));
            sin_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(self.rope.sin_at(pos));
        }

        // Handle namespaces (Phase 26):
        //   E4M3 final_norm=30_000_000, lm_head=31_000_000
        //   E5M2 final_norm=32_000_000, lm_head=33_000_000
        let (final_norm_handle, lm_head_handle): (u64, u64) = if is_e4m3 {
            (30_000_000, 31_000_000)
        } else {
            (32_000_000, 33_000_000)
        };
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();

        let (lm_head_bytes, lm_head_out_features): (&[u8], usize) = match &self.output_weight {
            OutputWeight::FP8E4M3(lm) => (fp8_e4m3_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::FP8E5M2(lm) => (fp8_e5m2_as_bytes(lm.blocks()), lm.out_features()),
            _ => return Err("FP8 LM head expected".into()),
        };

        // Build weight byte buffers (owned: gate_up per layer, qkv concat per layer).
        let qkv_concats = self.build_fp8_qkv_concats(is_e4m3)?;
        let gate_up_owners = self.build_fp8_gate_up_owners(is_e4m3)?;

        // Build layer params — gate_up_owners and qkv_concats must outlive this vec.
        let (norm_base, weight_base): (u64, u64) = if is_e4m3 {
            (26_000_000, 27_000_000)
        } else {
            (28_000_000, 29_000_000)
        };
        let mut layer_params: Vec<oxibonsai_kernels::CudaFP8PrefillLayerParams<'_>> =
            Vec::with_capacity(n_layers);
        for (idx, block) in self.blocks.iter().enumerate() {
            let li = block.layer_index() as u64;
            let nh = norm_base + li * 10;
            let wh = weight_base + li * 10;
            let (attn_proj_bytes, down_bytes) = if is_e4m3 {
                let ap = fp8_e4m3_as_bytes(
                    block
                        .attn_output_blocks_fp8e4m3()
                        .ok_or("attn_output: not FP8 E4M3")?,
                );
                let d = fp8_e4m3_as_bytes(
                    block
                        .ffn_down_blocks_fp8e4m3()
                        .ok_or("ffn_down: not FP8 E4M3")?,
                );
                (ap, d)
            } else {
                let ap = fp8_e5m2_as_bytes(
                    block
                        .attn_output_blocks_fp8e5m2()
                        .ok_or("attn_output: not FP8 E5M2")?,
                );
                let d = fp8_e5m2_as_bytes(
                    block
                        .ffn_down_blocks_fp8e5m2()
                        .ok_or("ffn_down: not FP8 E5M2")?,
                );
                (ap, d)
            };
            layer_params.push(oxibonsai_kernels::CudaFP8PrefillLayerParams {
                attn_norm_handle: nh,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: wh + 1,
                fused_qkv_bytes: &qkv_concats[idx],
                q_norm_handle: nh + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: nh + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: wh + 2,
                attn_proj_bytes,
                ffn_norm_handle: nh + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: wh + 3,
                gate_up_bytes: &gate_up_owners[idx],
                ffn_down_handle: wh + 4,
                ffn_down_bytes: down_bytes,
            });
        }

        let mut logits = vec![0.0f32; lm_head_out_features];
        oxibonsai_kernels::try_cuda_prefill_fp8(
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
            is_e4m3,
            Some(&mut logits),
            None,
        )
        .map_err(|e| {
            tracing::warn!(error = %e, "CUDA FP8 batch prefill dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(logits)
    }

    /// GPU batch prefill verify for FP8 models (CUDA): greedy argmax per position.
    ///
    /// Returns the greedy argmax token ID for each input position.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    pub(super) fn try_cuda_prefill_verify_fp8(
        &self,
        token_ids: &[u32],
        pos_start: usize,
        is_e4m3: bool,
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
        let half_dim = hd / 2;
        let heads_per_group = nq.checked_div(nkv).unwrap_or(1);
        let max_seq_len = self.kv_cache.max_seq_len();

        let (final_norm_handle, lm_head_handle): (u64, u64) = if is_e4m3 {
            (30_000_000, 31_000_000)
        } else {
            (32_000_000, 33_000_000)
        };
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();

        let (lm_head_bytes, lm_head_out_features): (&[u8], usize) = match &self.output_weight {
            OutputWeight::FP8E4M3(lm) => (fp8_e4m3_as_bytes(lm.blocks()), lm.out_features()),
            OutputWeight::FP8E5M2(lm) => (fp8_e5m2_as_bytes(lm.blocks()), lm.out_features()),
            _ => return Err("FP8 LM head expected".into()),
        };

        let qkv_concats = self.build_fp8_qkv_concats(is_e4m3)?;
        let gate_up_owners = self.build_fp8_gate_up_owners(is_e4m3)?;

        let (norm_base, weight_base): (u64, u64) = if is_e4m3 {
            (26_000_000, 27_000_000)
        } else {
            (28_000_000, 29_000_000)
        };
        let mut layer_params: Vec<oxibonsai_kernels::CudaFP8PrefillLayerParams<'_>> =
            Vec::with_capacity(n_layers);
        for (idx, block) in self.blocks.iter().enumerate() {
            let li = block.layer_index() as u64;
            let nh = norm_base + li * 10;
            let wh = weight_base + li * 10;
            let (attn_proj_bytes, down_bytes) = if is_e4m3 {
                let ap = fp8_e4m3_as_bytes(
                    block
                        .attn_output_blocks_fp8e4m3()
                        .ok_or("attn_output: not FP8 E4M3")?,
                );
                let d = fp8_e4m3_as_bytes(
                    block
                        .ffn_down_blocks_fp8e4m3()
                        .ok_or("ffn_down: not FP8 E4M3")?,
                );
                (ap, d)
            } else {
                let ap = fp8_e5m2_as_bytes(
                    block
                        .attn_output_blocks_fp8e5m2()
                        .ok_or("attn_output: not FP8 E5M2")?,
                );
                let d = fp8_e5m2_as_bytes(
                    block
                        .ffn_down_blocks_fp8e5m2()
                        .ok_or("ffn_down: not FP8 E5M2")?,
                );
                (ap, d)
            };
            layer_params.push(oxibonsai_kernels::CudaFP8PrefillLayerParams {
                attn_norm_handle: nh,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: wh + 1,
                fused_qkv_bytes: &qkv_concats[idx],
                q_norm_handle: nh + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: nh + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: wh + 2,
                attn_proj_bytes,
                ffn_norm_handle: nh + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: wh + 3,
                gate_up_bytes: &gate_up_owners[idx],
                ffn_down_handle: wh + 4,
                ffn_down_bytes: down_bytes,
            });
        }

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
            let cos_single: Vec<f32> = self.rope.cos_at(pos).to_vec();
            let sin_single: Vec<f32> = self.rope.sin_at(pos).to_vec();
            let _ = half_dim;

            let mut greedy_id: u32 = 0;
            oxibonsai_kernels::try_cuda_prefill_fp8(
                &single_hidden,
                1,
                pos,
                n_layers,
                &layer_params,
                &cos_single,
                &sin_single,
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
                is_e4m3,
                None,
                Some(&mut greedy_id),
            )
            .map_err(|e| {
                tracing::warn!(error = %e, "CUDA FP8 prefill verify at pos {pos}: {e}");
                Box::new(e) as Box<dyn std::error::Error>
            })?;
            token_ids_out.push(greedy_id);
        }
        Ok(token_ids_out)
    }
}
