//! Miscellaneous `TransformerBlock` methods: `layer_idx` plus the cfg-gated
//! full-layer GPU dispatch helpers (`try_full_layer_gpu` / `try_full_layer_cuda`).
//!
//! Both GPU dispatchers are private (`pub(super)` so they can be called from
//! the `forward` sub-module).

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
use crate::error::ModelResult;
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
use crate::kv_cache::KvCache;
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
use crate::layers::rope::RopeTable;

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
use crate::block::functions::blocks_as_bytes;

use super::block_def::TransformerBlock;

impl<'a> TransformerBlock<'a> {
    /// Get the layer index.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Attempt full-layer GPU dispatch (attention + FFN in a single command buffer).
    ///
    /// Returns:
    /// - `Some(Ok(()))` if the full layer was successfully computed on GPU.
    /// - `Some(Err(..))` if GPU dispatch was attempted but failed.
    /// - `None` if preconditions are not met (handles not available).
    ///
    /// On `Some(Ok(()))`, `hidden` is modified in-place and the caller should
    /// return early, skipping the CPU path entirely. The GPU manages its own
    /// KV cache internally.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn try_full_layer_gpu(
        &self,
        hidden: &mut [f32],
        pos: usize,
        rope: &RopeTable,
        kv_cache: &KvCache,
    ) -> Option<ModelResult<()>> {
        let fused_qkv_handle = self.fused_qkv_handle?;
        let attn_proj_handle = self.attn_output.gpu_handle()?;
        let fused_gate_up_handle = self.fused_gate_up_handle?;
        let down_handle = self.ffn_down.gpu_handle()?;
        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let inter = self.ffn_gate.out_features();
        let eps = self.attn_norm.eps();
        let n_layers = kv_cache.num_layers();
        let max_seq_len = kv_cache.max_seq_len();
        let norm_handle_base = 1_000_000u64 + (self.layer_idx as u64) * 10;
        let attn_norm_handle_id = norm_handle_base;
        let q_norm_handle_id = norm_handle_base + 1;
        let k_norm_handle_id = norm_handle_base + 2;
        let ffn_norm_handle_id = norm_handle_base + 3;
        let (q_blk, k_blk, v_blk, out_blk, gate_blk, up_blk, dn_blk) = match (
            self.attn_q.blocks_1bit(),
            self.attn_k.blocks_1bit(),
            self.attn_v.blocks_1bit(),
            self.attn_output.blocks_1bit(),
            self.ffn_gate.blocks_1bit(),
            self.ffn_up.blocks_1bit(),
            self.ffn_down.blocks_1bit(),
        ) {
            (Some(q), Some(k), Some(v), Some(o), Some(g), Some(u), Some(d)) => {
                (q, k, v, o, g, u, d)
            }
            _ => return None,
        };
        let fused_qkv_bytes = blocks_as_bytes(q_blk);
        let fused_qkv_k_bytes = blocks_as_bytes(k_blk);
        let fused_qkv_v_bytes = blocks_as_bytes(v_blk);
        let mut qkv_concat = Vec::with_capacity(
            fused_qkv_bytes.len() + fused_qkv_k_bytes.len() + fused_qkv_v_bytes.len(),
        );
        qkv_concat.extend_from_slice(fused_qkv_bytes);
        qkv_concat.extend_from_slice(fused_qkv_k_bytes);
        qkv_concat.extend_from_slice(fused_qkv_v_bytes);
        let attn_proj_bytes = blocks_as_bytes(out_blk);
        let gate_bytes = blocks_as_bytes(gate_blk);
        let up_bytes = blocks_as_bytes(up_blk);
        let down_bytes = blocks_as_bytes(dn_blk);
        let rope_cos = rope.cos_at(pos);
        let rope_sin = rope.sin_at(pos);
        let result = oxibonsai_kernels::try_metal_full_layer(
            hidden,
            pos,
            self.layer_idx,
            attn_norm_handle_id,
            self.attn_norm.weight(),
            fused_qkv_handle.id(),
            &qkv_concat,
            q_norm_handle_id,
            self.attn_q_norm.weight(),
            k_norm_handle_id,
            self.attn_k_norm.weight(),
            attn_proj_handle.id(),
            attn_proj_bytes,
            ffn_norm_handle_id,
            self.ffn_norm.weight(),
            fused_gate_up_handle.id(),
            gate_bytes,
            up_bytes,
            down_handle.id(),
            down_bytes,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            n_layers,
        );
        match result {
            Ok(()) => {
                tracing::debug!(
                    target : "block_profile", "L{layer}: full-layer GPU dispatch OK",
                    layer = self.layer_idx,
                );
                Some(Ok(()))
            }
            Err(e) => {
                tracing::warn!(
                    layer = self.layer_idx, error = % e,
                    "full-layer GPU dispatch failed, falling back to CPU path"
                );
                Some(Err(crate::error::ModelError::Internal(format!(
                    "Metal full-layer dispatch failed: {e}"
                ))))
            }
        }
    }

    /// Attempt full-layer CUDA GPU dispatch (attention + FFN, no intermediate
    /// CPU round-trips between the two sublayers).
    ///
    /// Returns:
    /// - `Some(Ok(()))` if the full layer was successfully computed on GPU.
    /// - `Some(Err(..))` if GPU dispatch was attempted but failed.
    /// - `None` if preconditions are not met (handles not available).
    #[cfg(all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    ))]
    pub(super) fn try_full_layer_cuda(
        &self,
        hidden: &mut [f32],
        pos: usize,
        rope: &RopeTable,
        kv_cache: &KvCache,
    ) -> Option<ModelResult<()>> {
        let fused_qkv_handle = self.fused_qkv_handle?;
        let attn_proj_handle = self.attn_output.gpu_handle()?;
        let fused_gate_up_handle = self.fused_gate_up_handle?;
        let down_handle = self.ffn_down.gpu_handle()?;
        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nq / nkv;
        let inter = self.ffn_gate.out_features();
        let eps = self.attn_norm.eps();
        let n_layers = kv_cache.num_layers();
        let max_seq_len = kv_cache.max_seq_len();
        let norm_handle_base = 2_000_000u64 + (self.layer_idx as u64) * 10;
        let attn_norm_handle_id = norm_handle_base;
        let q_norm_handle_id = norm_handle_base + 1;
        let k_norm_handle_id = norm_handle_base + 2;
        let ffn_norm_handle_id = norm_handle_base + 3;
        let (q_blk, k_blk, v_blk, out_blk, gate_blk, up_blk, dn_blk) = match (
            self.attn_q.blocks_1bit(),
            self.attn_k.blocks_1bit(),
            self.attn_v.blocks_1bit(),
            self.attn_output.blocks_1bit(),
            self.ffn_gate.blocks_1bit(),
            self.ffn_up.blocks_1bit(),
            self.ffn_down.blocks_1bit(),
        ) {
            (Some(q), Some(k), Some(v), Some(o), Some(g), Some(u), Some(d)) => {
                (q, k, v, o, g, u, d)
            }
            _ => return None,
        };
        let fused_qkv_bytes = blocks_as_bytes(q_blk);
        let fused_qkv_k_bytes = blocks_as_bytes(k_blk);
        let fused_qkv_v_bytes = blocks_as_bytes(v_blk);
        let mut qkv_concat = Vec::with_capacity(
            fused_qkv_bytes.len() + fused_qkv_k_bytes.len() + fused_qkv_v_bytes.len(),
        );
        qkv_concat.extend_from_slice(fused_qkv_bytes);
        qkv_concat.extend_from_slice(fused_qkv_k_bytes);
        qkv_concat.extend_from_slice(fused_qkv_v_bytes);
        let attn_proj_bytes = blocks_as_bytes(out_blk);
        let gate_bytes = blocks_as_bytes(gate_blk);
        let up_bytes = blocks_as_bytes(up_blk);
        let down_bytes = blocks_as_bytes(dn_blk);
        let rope_cos = rope.cos_at(pos);
        let rope_sin = rope.sin_at(pos);
        let result = oxibonsai_kernels::try_cuda_full_layer(
            hidden,
            pos,
            self.layer_idx,
            attn_norm_handle_id,
            self.attn_norm.weight(),
            fused_qkv_handle.id(),
            &qkv_concat,
            attn_proj_handle.id(),
            attn_proj_bytes,
            q_norm_handle_id,
            self.attn_q_norm.weight(),
            k_norm_handle_id,
            self.attn_k_norm.weight(),
            ffn_norm_handle_id,
            self.ffn_norm.weight(),
            fused_gate_up_handle.id(),
            gate_bytes,
            up_bytes,
            down_handle.id(),
            down_bytes,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            max_seq_len,
            n_layers,
        );
        match result {
            Ok(()) => {
                tracing::debug!(
                    target : "block_profile", "L{layer}: CUDA full-layer dispatch OK",
                    layer = self.layer_idx,
                );
                Some(Ok(()))
            }
            Err(e) => {
                tracing::warn!(
                    layer = self.layer_idx, error = % e,
                    "CUDA full-layer dispatch failed, falling back to CPU path"
                );
                Some(Err(crate::error::ModelError::Internal(format!(
                    "CUDA full-layer dispatch failed: {e}"
                ))))
            }
        }
    }
}
