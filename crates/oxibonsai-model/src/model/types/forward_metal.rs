//! Metal GPU forward-pass methods for `BonsaiModel`.

use super::{BonsaiModel, OutputWeight};
use crate::block::{blocks_as_bytes, blocks_as_bytes_ternary};

impl<'a> BonsaiModel<'a> {
    /// Attempt to run all transformer layers in a single Metal command buffer.
    ///
    /// On success, `hidden` is updated in-place through all layers. The GPU
    /// manages its own KV cache. Returns `Err` if any precondition is not
    /// met or the dispatch fails.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn try_metal_full_forward_inner(
        &self,
        hidden: &mut [f32],
        pos: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
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
        let max_seq_len = self.kv_cache.max_seq_len();
        for block in &self.blocks {
            if block.fused_qkv_gpu_handle().is_none()
                || block.attn_output_gpu_handle().is_none()
                || block.fused_gate_up_gpu_handle().is_none()
                || block.ffn_down_gpu_handle().is_none()
            {
                return Err("missing GPU handle".into());
            }
        }
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParams<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block.fused_qkv_gpu_handle().map(|h| h.id()).unwrap_or(0),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block.attn_output_gpu_handle().map(|h| h.id()).unwrap_or(0),
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|h| h.id())
                    .unwrap_or(0),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block.ffn_down_gpu_handle().map(|h| h.id()).unwrap_or(0),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        oxibonsai_kernels::try_metal_full_forward(
            hidden,
            pos,
            n_layers,
            &layer_params,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            None,
            None,
            eps,
            None,
            None,
            0,
            None,
            None,
        )
        .map_err(|e| {
            tracing::warn!(
                error = % e, "full-forward GPU dispatch failed, falling back"
            );
            Box::new(e) as Box<dyn std::error::Error>
        })
    }

    /// Attempt to run every transformer layer on Metal for a ternary
    /// (TQ2_0_g128) model, encoding all layers into a single command buffer.
    ///
    /// Mirrors [`try_metal_full_forward_inner`] but uses the TQ2 GEMV kernel
    /// and ternary block slices. Returns `Err` if any block is not ternary or
    /// the Metal dispatch fails — in which case the caller should fall back
    /// to the CPU per-layer path.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn try_metal_full_forward_ternary_inner(
        &self,
        hidden: &mut [f32],
        pos: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParamsTernary;
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
        let max_seq_len = self.kv_cache.max_seq_len();
        for block in &self.blocks {
            if block.attn_q_blocks_ternary().is_none()
                || block.attn_k_blocks_ternary().is_none()
                || block.attn_v_blocks_ternary().is_none()
                || block.attn_output_blocks_ternary().is_none()
                || block.ffn_gate_blocks_ternary().is_none()
                || block.ffn_up_blocks_ternary().is_none()
                || block.ffn_down_blocks_ternary().is_none()
            {
                return Err("non-ternary block on ternary full-forward path".into());
            }
        }
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
        let mut layer_params: Vec<FullForwardLayerParamsTernary<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 2_000_000u64 + (block.layer_index() as u64) * 10;
            let weight_handle_base = 3_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParamsTernary {
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
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        oxibonsai_kernels::try_metal_full_forward_ternary(
            hidden,
            pos,
            n_layers,
            &layer_params,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            None,
            None,
            eps,
            None,
            None,
            0,
            None,
            None,
        )
        .map_err(|e| {
            tracing::warn!(
                error = % e, "ternary full-forward GPU dispatch failed, falling back"
            );
            Box::new(e) as Box<dyn std::error::Error>
        })
    }

    /// Attempt to run all transformer layers + final RMSNorm + LM head GEMV
    /// in a single Metal command buffer.
    ///
    /// On success, `logits` is filled with the output logits and `hidden` is
    /// NOT updated (the GPU handles everything end-to-end). Returns `Err` if
    /// any precondition is not met (missing GPU handles, FP32 LM head, etc.).
    ///
    /// For ternary models the function delegates to the ternary GPU path using
    /// pre-cached byte slices; `get_or_create_gpu_cache` is called first.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn try_metal_full_forward_with_lm_head(
        &self,
        hidden: &mut [f32],
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }

        // ── Ternary path ──────────────────────────────────────────────────────
        if matches!(&self.output_weight, OutputWeight::Ternary(_)) {
            return self.try_metal_full_forward_with_lm_head_ternary(hidden, pos, logits);
        }

        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => unreachable!("handled above"),
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on fused GPU path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        for block in &self.blocks {
            if block.fused_qkv_gpu_handle().is_none()
                || block.attn_output_gpu_handle().is_none()
                || block.fused_gate_up_gpu_handle().is_none()
                || block.ffn_down_gpu_handle().is_none()
            {
                return Err("missing GPU handle".into());
            }
        }
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParams<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block.fused_qkv_gpu_handle().map(|h| h.id()).unwrap_or(0),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block.attn_output_gpu_handle().map(|h| h.id()).unwrap_or(0),
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|h| h.id())
                    .unwrap_or(0),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block.ffn_down_gpu_handle().map(|h| h.id()).unwrap_or(0),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        oxibonsai_kernels::try_metal_full_forward(
            hidden,
            pos,
            n_layers,
            &layer_params,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            Some(logits),
            None,
        )
        .map_err(|e| {
            tracing::warn!(
                error = % e, "full-forward+lm_head GPU dispatch failed, falling back"
            );
            Box::new(e) as Box<dyn std::error::Error>
        })
    }

    /// GPU batch prefill implementation: all layers + final norm + LM head.
    ///
    /// Both 1-bit and ternary (TQ2_0_g128) LM-head models are supported. The
    /// ternary path delegates to a separate helper that builds
    /// `FullForwardLayerParamsTernary` and dispatches the new TQ2 batched
    /// GEMM kernel across all layers.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn try_metal_prefill_with_lm_head(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                // Ternary batch prefill (TQ2_0_g128 weights end-to-end).
                return self.try_metal_prefill_with_lm_head_ternary(token_ids, pos_start);
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on GPU prefill path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
        let max_seq_len = self.kv_cache.max_seq_len();
        for block in &self.blocks {
            if block.fused_qkv_gpu_handle().is_none()
                || block.attn_output_gpu_handle().is_none()
                || block.fused_gate_up_gpu_handle().is_none()
                || block.ffn_down_gpu_handle().is_none()
            {
                return Err("missing GPU handle".into());
            }
        }
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
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParams<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block
                    .fused_qkv_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block
                    .attn_output_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block.ffn_down_gpu_handle().map(|hnd| hnd.id()).unwrap_or(0),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        let mut logits = vec![0.0f32; lm_head_out_features];
        oxibonsai_kernels::try_metal_full_forward_prefill(
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
            tracing::warn!(error = % e, "batch prefill GPU dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(logits)
    }

    /// GPU batch prefill verify: all layers + final norm + LM head + per-position argmax.
    ///
    /// Both 1-bit and ternary (TQ2_0_g128) LM-head models are supported. The
    /// ternary path delegates to
    /// [`Self::try_metal_prefill_verify_ternary_path`] which dispatches the
    /// new TQ2 batched GEMM kernel and per-position TQ2 LM-head GEMV.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn try_metal_prefill_verify(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                // Ternary batch prefill verify (TQ2_0_g128 weights end-to-end).
                return self.try_metal_prefill_verify_ternary_path(token_ids, pos_start);
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on GPU prefill verify path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
        let max_seq_len = self.kv_cache.max_seq_len();
        for block in &self.blocks {
            if block.fused_qkv_gpu_handle().is_none()
                || block.attn_output_gpu_handle().is_none()
                || block.fused_gate_up_gpu_handle().is_none()
                || block.ffn_down_gpu_handle().is_none()
            {
                return Err("missing GPU handle".into());
            }
        }
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
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParams<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block
                    .fused_qkv_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block
                    .attn_output_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(0),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block.ffn_down_gpu_handle().map(|hnd| hnd.id()).unwrap_or(0),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        let mut batch_token_ids: Vec<u32> = Vec::with_capacity(batch_size);
        oxibonsai_kernels::try_metal_full_forward_prefill_verify(
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
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            &mut batch_token_ids,
        )
        .map_err(|e| {
            tracing::warn!(error = % e, "batch prefill verify GPU dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(batch_token_ids)
    }

    /// Greedy forward pass: runs all layers + LM head + argmax entirely on GPU.
    ///
    /// Instead of downloading the full logits vector (~607KB), this performs
    /// argmax on the GPU and downloads only the resulting token ID (4 bytes).
    /// This eliminates ~607KB of GPU→CPU bandwidth per token and removes
    /// CPU-side sampling overhead for greedy (temperature=0) decoding.
    ///
    /// On the first call, all weight handles are cached in `gpu_weight_cache`.
    /// Subsequent calls skip ALL byte concatenation, weight upload, and
    /// HashMap lookups — passing pre-cached handles directly to the GPU.
    ///
    /// Supports both Q1 (1-bit) and ternary (TQ2_0_g128) models. FP32 LM head
    /// is not supported and returns `Err`.
    ///
    /// Returns the token ID directly, or `Err` if the GPU path is not available.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn forward_greedy_gpu(
        &self,
        token_id: u32,
        pos: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        let eps = if self.blocks.is_empty() {
            return Err("no blocks".into());
        } else {
            self.blocks[0].attn_norm_eps()
        };
        let final_norm_eps = self.output_norm.eps();

        // Ternary models use a separate cached path that builds
        // FullForwardLayerParamsTernary from the cached byte slices.
        if matches!(&self.output_weight, OutputWeight::Ternary(_)) {
            return self.forward_greedy_gpu_ternary(token_id, pos);
        }

        let lm_head_out_features = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear.out_features(),
            OutputWeight::Ternary(_) => unreachable!("handled above"),
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on greedy GPU path".into());
            }
        };
        self.get_or_create_gpu_cache()?;
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
        let mut hidden = self.token_embd[embd_start..embd_end].to_vec();
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        let mut greedy_token_id: u32 = 0;
        let guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        let cached = guard.as_ref().ok_or("GPU weight cache not populated")?;
        oxibonsai_kernels::try_metal_full_forward_cached(
            &mut hidden,
            pos,
            cached,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            final_norm_eps,
            lm_head_out_features,
            None,
            Some(&mut greedy_token_id),
        )
        .map_err(|e| {
            tracing::warn!(error = % e, "cached greedy GPU forward failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(greedy_token_id)
    }

    /// Ternary-model greedy decode: all transformer layers + ternary LM head + GPU argmax.
    ///
    /// Uses the pre-cached byte slices from `gpu_weight_cache` to build
    /// `FullForwardLayerParamsTernary` and dispatch through the TQ2 Metal kernel.
    /// On the first call the cache is populated (byte copies + GPU uploads);
    /// subsequent calls reuse the kernel-side weight cache.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn forward_greedy_gpu_ternary(
        &self,
        token_id: u32,
        pos: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParamsTernary;
        let n_layers = self.blocks.len();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        let eps = if self.blocks.is_empty() {
            return Err("no blocks".into());
        } else {
            self.blocks[0].attn_norm_eps()
        };
        let final_norm_eps = self.output_norm.eps();

        self.get_or_create_gpu_cache()?;

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
        let mut hidden = self.token_embd[embd_start..embd_end].to_vec();
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        let mut greedy_token_id: u32 = 0;

        let guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        let cached = guard.as_ref().ok_or("GPU weight cache not populated")?;

        if cached.ternary_qkv_concats.len() != n_layers {
            return Err(format!(
                "ternary cache layer count mismatch: expected {n_layers}, got {}",
                cached.ternary_qkv_concats.len()
            )
            .into());
        }

        // Build FullForwardLayerParamsTernary referencing cached byte slices.
        // These are cheap struct literals — no allocation beyond the Vec header.
        let mut layer_params: Vec<FullForwardLayerParamsTernary<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 5_000_000u64 + (block.layer_index() as u64) * 10;
            let weight_handle_base = 6_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParamsTernary {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: weight_handle_base,
                fused_qkv_bytes: &cached.ternary_qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: weight_handle_base + 1,
                attn_proj_bytes: &cached.ternary_attn_proj_bytes[i],
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: weight_handle_base + 2,
                gate_bytes: &cached.ternary_gate_bytes[i],
                up_bytes: &cached.ternary_up_bytes[i],
                down_handle: weight_handle_base + 3,
                down_bytes: &cached.ternary_down_bytes[i],
            });
        }
        let final_norm_handle = 5_900_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let lm_head_handle = 7_000_000u64;
        let lm_head_bytes = &cached.ternary_lm_head_bytes;
        let lm_head_out_features = cached.ternary_lm_head_out_features;

        oxibonsai_kernels::try_metal_forward_greedy_ternary(
            &mut hidden,
            pos,
            n_layers,
            &layer_params,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            &mut greedy_token_id,
        )
        .map_err(|e| {
            tracing::warn!(error = % e, "ternary greedy GPU forward failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(greedy_token_id)
    }

    /// Ternary fused forward + LM head (single token, non-greedy sampling).
    ///
    /// Routes the decode hot-path for ternary models through the GPU TQ2 kernel.
    /// Uses the same cached byte slices as `forward_greedy_gpu_ternary`.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn try_metal_full_forward_with_lm_head_ternary(
        &self,
        hidden: &mut [f32],
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParamsTernary;
        let n_layers = self.blocks.len();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        let eps = if self.blocks.is_empty() {
            return Err("no blocks".into());
        } else {
            self.blocks[0].attn_norm_eps()
        };
        let final_norm_eps = self.output_norm.eps();

        self.get_or_create_gpu_cache()?;

        let guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        let cached = guard.as_ref().ok_or("GPU weight cache not populated")?;

        if cached.ternary_qkv_concats.len() != n_layers {
            return Err(format!(
                "ternary cache layer count mismatch: expected {n_layers}, got {}",
                cached.ternary_qkv_concats.len()
            )
            .into());
        }

        let mut layer_params: Vec<FullForwardLayerParamsTernary<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 5_000_000u64 + (block.layer_index() as u64) * 10;
            let weight_handle_base = 6_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParamsTernary {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: weight_handle_base,
                fused_qkv_bytes: &cached.ternary_qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: weight_handle_base + 1,
                attn_proj_bytes: &cached.ternary_attn_proj_bytes[i],
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: weight_handle_base + 2,
                gate_bytes: &cached.ternary_gate_bytes[i],
                up_bytes: &cached.ternary_up_bytes[i],
                down_handle: weight_handle_base + 3,
                down_bytes: &cached.ternary_down_bytes[i],
            });
        }
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        let final_norm_handle = 5_900_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let lm_head_handle = 7_000_000u64;
        let lm_head_bytes = &cached.ternary_lm_head_bytes;
        let lm_head_out_features = cached.ternary_lm_head_out_features;

        oxibonsai_kernels::try_metal_prefill_ternary(
            hidden,
            pos,
            n_layers,
            &layer_params,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            logits,
        )
        .map_err(|e| {
            tracing::warn!(error = % e, "ternary fused GPU forward failed");
            Box::new(e) as Box<dyn std::error::Error>
        })
    }

    /// GPU batch prefill — ternary (TQ2_0_g128) variant.
    ///
    /// Mirror of [`Self::try_metal_prefill_with_lm_head`] for ternary
    /// LM-head models. Builds `FullForwardLayerParamsTernary` from the
    /// model's per-layer ternary blocks and dispatches the new TQ2 batched
    /// prefill kernel via [`oxibonsai_kernels::try_metal_full_forward_prefill_ternary`].
    /// Only the last token's logits are returned.
    ///
    /// Marked `pub` so parity tests can invoke this **strict** path
    /// directly, bypassing the silent fallback in
    /// [`Self::forward_prefill`] that masks GPU dispatch failures.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn try_metal_prefill_with_lm_head_ternary(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParamsTernary;
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::Ternary(linear) => linear,
            _ => return Err("ternary prefill called on non-ternary model".into()),
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
        let max_seq_len = self.kv_cache.max_seq_len();

        // Embed prompt tokens into `[batch × hidden]` column-major layout.
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

        // Pre-compute RoPE cos/sin tables for every position in the batch.
        let mut cos_table = vec![0.0f32; batch_size * half_dim];
        let mut sin_table = vec![0.0f32; batch_size * half_dim];
        for t in 0..batch_size {
            let pos = pos_start + t;
            let cos_vals = self.rope.cos_at(pos);
            let sin_vals = self.rope.sin_at(pos);
            cos_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(cos_vals);
            sin_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(sin_vals);
        }

        // Build per-layer ternary parameters: concatenate Q+K+V byte slices
        // per layer (mirrors what the per-position ternary forward does).
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut attn_proj_bytes_per_layer: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut gate_bytes_per_layer: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut up_bytes_per_layer: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut down_bytes_per_layer: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
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
            attn_proj_bytes_per_layer.push(
                blocks_as_bytes_ternary(
                    block
                        .attn_output_blocks_ternary()
                        .ok_or("attn_output: not a ternary layer")?,
                )
                .to_vec(),
            );
            gate_bytes_per_layer.push(
                blocks_as_bytes_ternary(
                    block
                        .ffn_gate_blocks_ternary()
                        .ok_or("ffn_gate: not a ternary layer")?,
                )
                .to_vec(),
            );
            up_bytes_per_layer.push(
                blocks_as_bytes_ternary(
                    block
                        .ffn_up_blocks_ternary()
                        .ok_or("ffn_up: not a ternary layer")?,
                )
                .to_vec(),
            );
            down_bytes_per_layer.push(
                blocks_as_bytes_ternary(
                    block
                        .ffn_down_blocks_ternary()
                        .ok_or("ffn_down: not a ternary layer")?,
                )
                .to_vec(),
            );
        }

        let mut layer_params: Vec<FullForwardLayerParamsTernary<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 5_000_000u64 + (block.layer_index() as u64) * 10;
            let weight_handle_base = 6_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParamsTernary {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: weight_handle_base,
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: weight_handle_base + 1,
                attn_proj_bytes: &attn_proj_bytes_per_layer[i],
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: weight_handle_base + 2,
                gate_bytes: &gate_bytes_per_layer[i],
                up_bytes: &up_bytes_per_layer[i],
                down_handle: weight_handle_base + 3,
                down_bytes: &down_bytes_per_layer[i],
            });
        }

        let final_norm_handle = 5_900_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 7_000_000u64;
        let lm_head_bytes_vec = blocks_as_bytes_ternary(lm_head_linear.blocks()).to_vec();
        let lm_head_bytes: &[u8] = &lm_head_bytes_vec;
        let lm_head_out_features = lm_head_linear.out_features();

        let mut logits = vec![0.0f32; lm_head_out_features];
        oxibonsai_kernels::try_metal_full_forward_prefill_ternary(
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
            tracing::warn!(error = % e, "ternary batch prefill GPU dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(logits)
    }

    /// GPU batch prefill verify — ternary (TQ2_0_g128) variant.
    ///
    /// Mirror of [`Self::try_metal_prefill_verify`] for ternary LM-head
    /// models. Returns the per-position greedy argmax token IDs.
    ///
    /// Marked `pub` so parity tests can invoke this **strict** path
    /// directly, bypassing the silent fallback in
    /// [`Self::forward_prefill_verify`] that masks GPU dispatch failures.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn try_metal_prefill_verify_ternary_path(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParamsTernary;
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::Ternary(linear) => linear,
            _ => return Err("ternary prefill verify called on non-ternary model".into()),
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
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

        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut attn_proj_bytes_per_layer: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut gate_bytes_per_layer: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut up_bytes_per_layer: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut down_bytes_per_layer: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
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
            attn_proj_bytes_per_layer.push(
                blocks_as_bytes_ternary(
                    block
                        .attn_output_blocks_ternary()
                        .ok_or("attn_output: not a ternary layer")?,
                )
                .to_vec(),
            );
            gate_bytes_per_layer.push(
                blocks_as_bytes_ternary(
                    block
                        .ffn_gate_blocks_ternary()
                        .ok_or("ffn_gate: not a ternary layer")?,
                )
                .to_vec(),
            );
            up_bytes_per_layer.push(
                blocks_as_bytes_ternary(
                    block
                        .ffn_up_blocks_ternary()
                        .ok_or("ffn_up: not a ternary layer")?,
                )
                .to_vec(),
            );
            down_bytes_per_layer.push(
                blocks_as_bytes_ternary(
                    block
                        .ffn_down_blocks_ternary()
                        .ok_or("ffn_down: not a ternary layer")?,
                )
                .to_vec(),
            );
        }

        let mut layer_params: Vec<FullForwardLayerParamsTernary<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 5_000_000u64 + (block.layer_index() as u64) * 10;
            let weight_handle_base = 6_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParamsTernary {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: weight_handle_base,
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: weight_handle_base + 1,
                attn_proj_bytes: &attn_proj_bytes_per_layer[i],
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: weight_handle_base + 2,
                gate_bytes: &gate_bytes_per_layer[i],
                up_bytes: &up_bytes_per_layer[i],
                down_handle: weight_handle_base + 3,
                down_bytes: &down_bytes_per_layer[i],
            });
        }

        let final_norm_handle = 5_900_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 7_000_000u64;
        let lm_head_bytes_vec = blocks_as_bytes_ternary(lm_head_linear.blocks()).to_vec();
        let lm_head_bytes: &[u8] = &lm_head_bytes_vec;
        let lm_head_out_features = lm_head_linear.out_features();

        let mut batch_token_ids: Vec<u32> = Vec::with_capacity(batch_size);
        oxibonsai_kernels::try_metal_full_forward_prefill_verify_ternary(
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
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            &mut batch_token_ids,
        )
        .map_err(|e| {
            tracing::warn!(error = % e, "ternary batch prefill verify GPU dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(batch_token_ids)
    }
}
