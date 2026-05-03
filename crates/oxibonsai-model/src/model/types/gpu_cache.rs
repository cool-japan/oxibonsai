//! GPU weight cache management for the Metal zero-overhead decode path.

use super::{BonsaiModel, OutputWeight};
use crate::block::{blocks_as_bytes, blocks_as_bytes_ternary};

impl<'a> BonsaiModel<'a> {
    /// Build and cache GPU weight handles on first call; no-op on subsequent calls.
    ///
    /// For Q1 models, the cache contains pre-uploaded `CachedLayerWeights` handles
    /// plus a pre-uploaded LM-head handle — used by `try_metal_full_forward_cached`.
    ///
    /// For ternary (TQ2_0_g128) models, the cache stores raw byte copies for each
    /// layer's weight matrices. Callers rebuild `FullForwardLayerParamsTernary` from
    /// these bytes each decode step (cheap struct literals; no `Box::leak`).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn get_or_create_gpu_cache(&self) -> Result<(), Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
        {
            let guard = self
                .gpu_weight_cache
                .lock()
                .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
            if guard.is_some() {
                return Ok(());
            }
        }
        let n_layers = self.blocks.len();

        // ── Ternary path ──────────────────────────────────────────────────────
        if let OutputWeight::Ternary(ref lm_head_ternary) = self.output_weight {
            return self.build_ternary_gpu_cache(n_layers, lm_head_ternary);
        }

        // ── Q1 path ───────────────────────────────────────────────────────────
        if self.blocks.iter().any(|b| b.attn_q_blocks().is_none()) {
            return Ok(());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(ref linear) => linear,
            OutputWeight::Fp32 { .. } => return Err("FP32 LM head not supported".into()),
            OutputWeight::Ternary(_) => unreachable!("handled above"),
        };
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
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let cached = oxibonsai_kernels::build_cached_weights(
            &layer_params,
            final_norm_handle,
            final_norm_bytes,
            lm_head_handle,
            lm_head_bytes,
        )
        .map_err(|e| format!("build_cached_weights: {e}"))?;
        let mut guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        *guard = Some(cached);
        tracing::info!("GPU weight cache populated (Q1; all subsequent tokens use cached handles)");
        Ok(())
    }

    /// Build and store the ternary (TQ2_0_g128) GPU weight cache.
    ///
    /// Raw byte copies are stored so that `FullForwardLayerParamsTernary` can be
    /// rebuilt cheaply on every decode step without `Box::leak` or self-referential
    /// gymnastics.
    ///
    /// Handle ID namespaces (distinct from the Q1 ranges 1M–3M):
    ///   norm    handles: 5_000_000 + layer * 10 + offset
    ///   weight  handles: 6_000_000 + layer * 10 + offset
    ///   lm_head handle : 7_000_000
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn build_ternary_gpu_cache(
        &self,
        n_layers: usize,
        lm_head_ternary: &crate::layers::linear::LinearTernary<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut ternary_qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut ternary_attn_proj_bytes: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut ternary_gate_bytes: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut ternary_up_bytes: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        let mut ternary_down_bytes: Vec<Vec<u8>> = Vec::with_capacity(n_layers);

        for block in &self.blocks {
            // QKV concat
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
            let mut qkv = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            qkv.extend_from_slice(q_bytes);
            qkv.extend_from_slice(k_bytes);
            qkv.extend_from_slice(v_bytes);
            ternary_qkv_concats.push(qkv);

            // Attention projection (output)
            let proj_bytes = blocks_as_bytes_ternary(
                block
                    .attn_output_blocks_ternary()
                    .ok_or("attn_output: not a ternary layer")?,
            );
            ternary_attn_proj_bytes.push(proj_bytes.to_vec());

            // Gate projection (stored separately; kernel concatenates with up lazily)
            let gate = blocks_as_bytes_ternary(
                block
                    .ffn_gate_blocks_ternary()
                    .ok_or("ffn_gate: not a ternary layer")?,
            );
            ternary_gate_bytes.push(gate.to_vec());

            // Up projection
            let up = blocks_as_bytes_ternary(
                block
                    .ffn_up_blocks_ternary()
                    .ok_or("ffn_up: not a ternary layer")?,
            );
            ternary_up_bytes.push(up.to_vec());

            // Down projection
            let down = blocks_as_bytes_ternary(
                block
                    .ffn_down_blocks_ternary()
                    .ok_or("ffn_down: not a ternary layer")?,
            );
            ternary_down_bytes.push(down.to_vec());
        }

        let ternary_lm_head_bytes = blocks_as_bytes_ternary(lm_head_ternary.blocks()).to_vec();
        let ternary_lm_head_out_features = lm_head_ternary.out_features();

        // Use the dedicated ternary-only builder, which avoids any Q1 block-size
        // validation (18-byte alignment) that `build_cached_weights` imposes on
        // `lm_head_bytes`. The resulting `CachedModelWeights` has trivial f32
        // placeholders for the Q1 `final_norm` / `lm_head` handles; those fields
        // are never accessed on the ternary forward path.
        let cached = oxibonsai_kernels::build_cached_weights_ternary_only(
            ternary_qkv_concats,
            ternary_attn_proj_bytes,
            ternary_gate_bytes,
            ternary_up_bytes,
            ternary_down_bytes,
            ternary_lm_head_bytes,
            ternary_lm_head_out_features,
        )
        .map_err(|e| format!("build_cached_weights_ternary_only: {e}"))?;

        let mut guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        *guard = Some(cached);
        tracing::info!(
            "GPU weight cache populated (ternary; {} layers, {} lm-head output features)",
            n_layers,
            ternary_lm_head_out_features,
        );
        Ok(())
    }
}
