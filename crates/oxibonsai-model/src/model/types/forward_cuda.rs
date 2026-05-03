//! CUDA GPU forward-pass methods for `BonsaiModel`.

use super::{BonsaiModel, OutputWeight};
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
use crate::block::{blocks_as_bytes, blocks_as_bytes_ternary};

impl<'a> BonsaiModel<'a> {
    /// Get or build the cached per-layer QKV byte concatenations for the CUDA path.
    ///
    /// On first call the vectors are built and stored in `cuda_qkv_cache`.
    /// On subsequent calls the cached version is returned immediately.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn get_or_build_cuda_qkv_cache(
        &self,
    ) -> Result<std::sync::Arc<Vec<Vec<u8>>>, Box<dyn std::error::Error>> {
        let guard = self
            .cuda_qkv_cache
            .lock()
            .map_err(|e| format!("cuda_qkv_cache lock: {e}"))?;
        if let Some(ref cache) = *guard {
            return Ok(std::sync::Arc::clone(cache));
        }
        drop(guard);
        let n_layers = self.blocks.len();
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
        let mut guard = self
            .cuda_qkv_cache
            .lock()
            .map_err(|e| format!("cuda_qkv_cache lock: {e}"))?;
        let arc = std::sync::Arc::new(qkv_concats);
        *guard = Some(std::sync::Arc::clone(&arc));
        Ok(arc)
    }

    /// Build per-layer `CudaFullForwardLayerParams` using cached QKV bytes.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn build_cuda_layer_params<'b>(
        &'b self,
        qkv_concats: &'b [Vec<u8>],
    ) -> Result<Vec<oxibonsai_kernels::CudaFullForwardLayerParams<'b>>, Box<dyn std::error::Error>>
    {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let mut layer_params: Vec<oxibonsai_kernels::CudaFullForwardLayerParams<'b>> =
            Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            let weight_handle_base = 2_000_000u64 + (block.layer_index() as u64) * 4;
            layer_params.push(oxibonsai_kernels::CudaFullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block
                    .fused_qkv_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(weight_handle_base),
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block
                    .attn_output_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(weight_handle_base + 1),
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
                    .unwrap_or(weight_handle_base + 2),
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block
                    .ffn_down_gpu_handle()
                    .map(|hnd| hnd.id())
                    .unwrap_or(weight_handle_base + 3),
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        Ok(layer_params)
    }

    /// Build per-layer ternary QKV byte concatenations for the CUDA ternary path.
    ///
    /// Each layer's Q, K, V TQ2 block bytes are concatenated in that order.
    /// Built fresh on each call (no caching needed — the GPU weight cache handles
    /// upload deduplication via handle IDs).
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn build_cuda_ternary_qkv_concats(&self) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
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
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn build_cuda_ternary_layer_params<'b>(
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

    /// Attempt to run all transformer layers (layers only, no LM head) on CUDA GPU.
    ///
    /// On success, returns the post-layers hidden state as a `Vec<f32>` which the
    /// caller should use to replace the CPU `hidden` buffer.  Returns `Err` if CUDA
    /// is unavailable or any precondition is not met.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    pub(super) fn try_cuda_full_forward_inner(
        &self,
        hidden: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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
        let qkv_concats = self.get_or_build_cuda_qkv_cache()?;
        let layer_params = self.build_cuda_layer_params(&qkv_concats)?;
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        oxibonsai_kernels::try_cuda_full_forward(
            hidden,
            &layer_params,
            rope_cos,
            rope_sin,
            pos,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            h,
            inter,
            max_seq_len,
            None,
            0,
        )
        .ok_or_else(|| {
            tracing::warn!("CUDA full-forward (layers only) returned None, falling back");
            Box::<dyn std::error::Error>::from("CUDA layers-only forward returned None")
        })
    }

    /// Attempt to run all transformer layers + final RMSNorm + LM head on CUDA GPU.
    ///
    /// On success, returns the output logits vector directly (no intermediate allocation).
    /// Returns `Err` if any precondition is not met (no CUDA device, FP32 LM head,
    /// missing GPU handles, etc.).
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    pub(super) fn try_cuda_full_forward_with_lm_head(
        &self,
        hidden: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }

        // ── Ternary path ──────────────────────────────────────────────────────
        if let OutputWeight::Ternary(ref lm_head_ternary) = self.output_weight {
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
            let lm_head_handle = 7_000_000u64;
            let lm_head_bytes = blocks_as_bytes_ternary(lm_head_ternary.blocks());
            let vocab_size = lm_head_ternary.out_features();
            let qkv_concats = self.build_cuda_ternary_qkv_concats()?;
            let layer_params = self.build_cuda_ternary_layer_params(&qkv_concats)?;
            let rope_cos = self.rope.cos_at(pos);
            let rope_sin = self.rope.sin_at(pos);
            return match oxibonsai_kernels::try_cuda_full_forward_ternary_with_gpu_lm_head(
                hidden,
                &layer_params,
                rope_cos,
                rope_sin,
                pos,
                nq,
                nkv,
                hd,
                heads_per_group,
                eps,
                h,
                inter,
                max_seq_len,
                Some(final_norm_bytes),
                final_norm_handle,
                lm_head_handle,
                lm_head_bytes,
                vocab_size,
            ) {
                Some(gpu_logits) => Ok(gpu_logits),
                None => {
                    tracing::warn!(
                        "CUDA ternary full-forward+gpu_lm_head returned None, falling back"
                    );
                    Err("CUDA ternary full-forward+gpu_lm_head returned None".into())
                }
            };
        }

        // ── Q1 path ───────────────────────────────────────────────────────────
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => unreachable!("handled above"),
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on CUDA fused GPU path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let heads_per_group = nq.checked_div(nkv).unwrap_or(1);
        let max_seq_len = self.kv_cache.max_seq_len();
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let lm_head_handle = 4_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let vocab_size = lm_head_linear.out_features();
        let qkv_concats = self.get_or_build_cuda_qkv_cache()?;
        let layer_params = self.build_cuda_layer_params(&qkv_concats)?;
        let rope_cos = self.rope.cos_at(pos);
        let rope_sin = self.rope.sin_at(pos);
        match oxibonsai_kernels::try_cuda_full_forward_with_gpu_lm_head(
            hidden,
            &layer_params,
            rope_cos,
            rope_sin,
            pos,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            h,
            inter,
            max_seq_len,
            Some(final_norm_bytes),
            final_norm_handle,
            lm_head_handle,
            lm_head_bytes,
            vocab_size,
        ) {
            Some(gpu_logits) => Ok(gpu_logits),
            None => {
                tracing::warn!("CUDA full-forward+gpu_lm_head returned None, falling back");
                Err("CUDA full-forward+gpu_lm_head returned None".into())
            }
        }
    }

    /// GPU batch prefill implementation (CUDA): all layers + final norm + LM head.
    ///
    /// Returns the last token's logits.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    pub(super) fn try_cuda_prefill_with_lm_head(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        // Ternary batch prefill: no CUDA batch ternary prefill kernel exists yet.
        // Fall back to sequential single-token ternary forward (same approach as Metal).
        if matches!(&self.output_weight, OutputWeight::Ternary(_)) {
            return Err(
                "ternary CUDA batch prefill not yet implemented; using sequential GPU path".into(),
            );
        }

        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => unreachable!("handled above"),
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on CUDA prefill path".into());
            }
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
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        let qkv_concats = self.get_or_build_cuda_qkv_cache()?;
        let layer_params = self.build_cuda_layer_params(&qkv_concats)?;
        let mut logits = vec![0.0f32; lm_head_out_features];
        oxibonsai_kernels::try_cuda_prefill(
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
            tracing::warn!(error = % e, "CUDA batch prefill dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(logits)
    }

    /// GPU batch prefill verify (CUDA): all layers + final norm + LM head + argmax.
    ///
    /// Returns the greedy argmax token ID for each input position.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    pub(super) fn try_cuda_prefill_verify(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        // Ternary batch prefill verify: no CUDA batch ternary prefill-verify kernel exists yet.
        // Fall back to sequential single-token ternary forward (same approach as Metal).
        if matches!(&self.output_weight, OutputWeight::Ternary(_)) {
            return Err(
                "ternary CUDA batch prefill verify not yet implemented; using sequential GPU path"
                    .into(),
            );
        }

        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => unreachable!("handled above"),
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on CUDA prefill verify path".into());
            }
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
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        let qkv_concats = self.get_or_build_cuda_qkv_cache()?;
        let layer_params = self.build_cuda_layer_params(&qkv_concats)?;
        let mut token_ids_out: Vec<u32> = Vec::with_capacity(batch_size);
        for t in 0..batch_size {
            let single_embd_start = token_ids[t] as usize * h;
            let single_hidden = self.token_embd[single_embd_start..single_embd_start + h].to_vec();
            let pos = pos_start + t;
            let t_half = half_dim;
            let cos_single = &cos_table[t * t_half..(t + 1) * t_half];
            let sin_single = &sin_table[t * t_half..(t + 1) * t_half];
            let mut greedy_id: u32 = 0;
            oxibonsai_kernels::try_cuda_prefill(
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
                tracing::warn!(
                    error = % e, "CUDA prefill verify dispatch failed at pos {pos}"
                );
                Box::new(e) as Box<dyn std::error::Error>
            })?;
            token_ids_out.push(greedy_id);
        }
        Ok(token_ids_out)
    }
}
