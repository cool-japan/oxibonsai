//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::Arc;

use super::super::metal_full_layer::FullForwardLayerParams;
use super::super::metal_graph::{MetalGraph, MetalGraphError, MetalWeightHandle};

/// Attempt to run batch prefill (ALL transformer layers + LM head) in a
/// single Metal command buffer for multiple prompt tokens.
///
/// Like `try_metal_full_forward`, but processes `batch_size` tokens at once
/// using GEMM instead of GEMV for projections, with sequential per-token
/// attention within each layer. Only the last token's logits are returned.
///
/// Returns `Ok(())` on success. Returns `Err(...)` if Metal is unavailable
/// or any dispatch step fails.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParams<'_>],
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_handle: Option<u64>,
    final_norm_bytes: Option<&[f32]>,
    final_norm_eps: f32,
    lm_head_handle: Option<u64>,
    lm_head_bytes: Option<&[u8]>,
    lm_head_out_features: usize,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), MetalGraphError> {
    if layer_params.len() != n_layers {
        return Err(
            MetalGraphError::EncodingFailed(
                format!(
                    "layer_params length mismatch: need {n_layers}, got {}", layer_params
                    .len()
                ),
            ),
        );
    }
    let graph = MetalGraph::global()?;
    #[allow(clippy::type_complexity)]
    let mut layer_weights: Vec<
        (
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
        ),
    > = Vec::with_capacity(n_layers);
    for lp in layer_params {
        let attn_norm_w = graph
            .get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph
            .get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph
            .get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph
            .get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
        let fused_qkv_w = graph
            .get_or_upload_q1_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w = graph
            .get_or_upload_q1_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph
            .get_or_upload_q1_weight_soa_lazy(
                lp.gate_up_handle,
                || {
                    let mut fused = Vec::with_capacity(
                        gate_bytes.len() + up_bytes.len(),
                    );
                    fused.extend_from_slice(gate_bytes);
                    fused.extend_from_slice(up_bytes);
                    fused
                },
            )?;
        let down_w = graph.get_or_upload_q1_weight_soa(lp.down_handle, lp.down_bytes)?;
        layer_weights
            .push((
                attn_norm_w,
                fused_qkv_w,
                q_norm_w,
                k_norm_w,
                attn_proj_w,
                ffn_norm_w,
                gate_up_w,
                down_w,
            ));
    }
    let weight_refs: Vec<_> = layer_weights
        .iter()
        .map(|(a, b, c, d, e, f, g, h)| (a, b, c, d, e, f, g, h))
        .collect();
    let final_norm_cached = match (final_norm_handle, final_norm_bytes) {
        (Some(handle), Some(bytes)) => {
            Some(graph.get_or_upload_f32_weight(handle, bytes)?)
        }
        _ => None,
    };
    let lm_head_cached = match (lm_head_handle, lm_head_bytes) {
        (Some(handle), Some(bytes)) => {
            Some(graph.get_or_upload_q1_weight_soa(handle, bytes)?)
        }
        _ => None,
    };
    graph
        .encode_full_forward_prefill(
            hidden_batch,
            pos_start,
            batch_size,
            n_layers,
            &weight_refs,
            cos_table,
            sin_table,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            eps,
            max_seq_len,
            final_norm_cached.as_ref(),
            final_norm_eps,
            lm_head_cached.as_ref(),
            lm_head_out_features,
            logits_out,
            greedy_token_id_out,
        )
}
/// Full-forward prefill for **verification** (speculative decoding).
///
/// Runs all transformer layers then final-norm + LM-head on **every** batch
/// position and returns per-position argmax token IDs.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill_verify(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParams<'_>],
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_handle: Option<u64>,
    final_norm_bytes: Option<&[f32]>,
    final_norm_eps: f32,
    lm_head_handle: Option<u64>,
    lm_head_bytes: Option<&[u8]>,
    lm_head_out_features: usize,
    batch_token_ids_out: &mut Vec<u32>,
) -> Result<(), MetalGraphError> {
    if layer_params.len() != n_layers {
        return Err(
            MetalGraphError::EncodingFailed(
                format!(
                    "layer_params length mismatch: need {n_layers}, got {}", layer_params
                    .len()
                ),
            ),
        );
    }
    let graph = MetalGraph::global()?;
    #[allow(clippy::type_complexity)]
    let mut layer_weights: Vec<
        (
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
        ),
    > = Vec::with_capacity(n_layers);
    for lp in layer_params {
        let attn_norm_w = graph
            .get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph
            .get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph
            .get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph
            .get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
        let fused_qkv_w = graph
            .get_or_upload_q1_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w = graph
            .get_or_upload_q1_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph
            .get_or_upload_q1_weight_soa_lazy(
                lp.gate_up_handle,
                || {
                    let mut fused = Vec::with_capacity(
                        gate_bytes.len() + up_bytes.len(),
                    );
                    fused.extend_from_slice(gate_bytes);
                    fused.extend_from_slice(up_bytes);
                    fused
                },
            )?;
        let down_w = graph.get_or_upload_q1_weight_soa(lp.down_handle, lp.down_bytes)?;
        layer_weights
            .push((
                attn_norm_w,
                fused_qkv_w,
                q_norm_w,
                k_norm_w,
                attn_proj_w,
                ffn_norm_w,
                gate_up_w,
                down_w,
            ));
    }
    let weight_refs: Vec<_> = layer_weights
        .iter()
        .map(|(a, b, c, d, e, f, g, h)| (a, b, c, d, e, f, g, h))
        .collect();
    let final_norm_cached = match (final_norm_handle, final_norm_bytes) {
        (Some(handle), Some(bytes)) => {
            Some(graph.get_or_upload_f32_weight(handle, bytes)?)
        }
        _ => None,
    };
    let lm_head_cached = match (lm_head_handle, lm_head_bytes) {
        (Some(handle), Some(bytes)) => {
            Some(graph.get_or_upload_q1_weight_soa(handle, bytes)?)
        }
        _ => None,
    };
    graph
        .encode_full_forward_prefill_verify(
            hidden_batch,
            pos_start,
            batch_size,
            n_layers,
            &weight_refs,
            cos_table,
            sin_table,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            eps,
            max_seq_len,
            final_norm_cached.as_ref(),
            final_norm_eps,
            lm_head_cached.as_ref(),
            lm_head_out_features,
            batch_token_ids_out,
        )
}
/// Attempt to run **ternary** batch prefill (ALL transformer layers + LM head)
/// in a single Metal command buffer for multiple prompt tokens.
///
/// Mirror of [`try_metal_full_forward_prefill`] but every weight projection
/// dispatches through the TQ2_0_g128 batched GEMM kernel. The final RMSNorm
/// runs on the last token only and the LM head is dispatched via the TQ2
/// GEMV. Only the last token's logits (or its greedy argmax) are returned.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill_ternary(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[super::super::metal_full_layer::FullForwardLayerParamsTernary<'_>],
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_handle: Option<u64>,
    final_norm_bytes: Option<&[f32]>,
    final_norm_eps: f32,
    lm_head_handle: Option<u64>,
    lm_head_bytes: Option<&[u8]>,
    lm_head_out_features: usize,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), MetalGraphError> {
    if layer_params.len() != n_layers {
        return Err(
            MetalGraphError::EncodingFailed(
                format!(
                    "layer_params length mismatch: need {n_layers}, got {}", layer_params
                    .len()
                ),
            ),
        );
    }
    let graph = MetalGraph::global()?;
    #[allow(clippy::type_complexity)]
    let mut layer_weights: Vec<
        (
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
        ),
    > = Vec::with_capacity(n_layers);
    for lp in layer_params {
        let attn_norm_w = graph
            .get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph
            .get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph
            .get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph
            .get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
        let fused_qkv_w = graph
            .get_or_upload_tq2_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w = graph
            .get_or_upload_tq2_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph
            .get_or_upload_tq2_weight_soa_lazy(
                lp.gate_up_handle,
                || {
                    let mut fused = Vec::with_capacity(
                        gate_bytes.len() + up_bytes.len(),
                    );
                    fused.extend_from_slice(gate_bytes);
                    fused.extend_from_slice(up_bytes);
                    fused
                },
            )?;
        let down_w = graph.get_or_upload_tq2_weight_soa(lp.down_handle, lp.down_bytes)?;
        layer_weights
            .push((
                attn_norm_w,
                fused_qkv_w,
                q_norm_w,
                k_norm_w,
                attn_proj_w,
                ffn_norm_w,
                gate_up_w,
                down_w,
            ));
    }
    let weight_refs: Vec<_> = layer_weights
        .iter()
        .map(|(a, b, c, d, e, f, g, h)| (a, b, c, d, e, f, g, h))
        .collect();
    let final_norm_cached = match (final_norm_handle, final_norm_bytes) {
        (Some(handle), Some(bytes)) => {
            Some(graph.get_or_upload_f32_weight(handle, bytes)?)
        }
        _ => None,
    };
    let lm_head_cached = match (lm_head_handle, lm_head_bytes) {
        (Some(handle), Some(bytes)) => {
            Some(graph.get_or_upload_tq2_weight_soa(handle, bytes)?)
        }
        _ => None,
    };
    graph
        .encode_full_forward_prefill_ternary(
            hidden_batch,
            pos_start,
            batch_size,
            n_layers,
            &weight_refs,
            cos_table,
            sin_table,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            eps,
            max_seq_len,
            final_norm_cached.as_ref(),
            final_norm_eps,
            lm_head_cached.as_ref(),
            lm_head_out_features,
            logits_out,
            greedy_token_id_out,
        )
}
/// Ternary batch prefill **verify** (speculative decoding).
///
/// Runs all layers + final norm + TQ2 LM head on every batch position and
/// returns the per-position greedy argmax token IDs.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill_verify_ternary(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[super::super::metal_full_layer::FullForwardLayerParamsTernary<'_>],
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_handle: Option<u64>,
    final_norm_bytes: Option<&[f32]>,
    final_norm_eps: f32,
    lm_head_handle: Option<u64>,
    lm_head_bytes: Option<&[u8]>,
    lm_head_out_features: usize,
    batch_token_ids_out: &mut Vec<u32>,
) -> Result<(), MetalGraphError> {
    if layer_params.len() != n_layers {
        return Err(
            MetalGraphError::EncodingFailed(
                format!(
                    "layer_params length mismatch: need {n_layers}, got {}", layer_params
                    .len()
                ),
            ),
        );
    }
    let graph = MetalGraph::global()?;
    #[allow(clippy::type_complexity)]
    let mut layer_weights: Vec<
        (
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
            Arc<MetalWeightHandle>,
        ),
    > = Vec::with_capacity(n_layers);
    for lp in layer_params {
        let attn_norm_w = graph
            .get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph
            .get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph
            .get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph
            .get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
        let fused_qkv_w = graph
            .get_or_upload_tq2_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w = graph
            .get_or_upload_tq2_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph
            .get_or_upload_tq2_weight_soa_lazy(
                lp.gate_up_handle,
                || {
                    let mut fused = Vec::with_capacity(
                        gate_bytes.len() + up_bytes.len(),
                    );
                    fused.extend_from_slice(gate_bytes);
                    fused.extend_from_slice(up_bytes);
                    fused
                },
            )?;
        let down_w = graph.get_or_upload_tq2_weight_soa(lp.down_handle, lp.down_bytes)?;
        layer_weights
            .push((
                attn_norm_w,
                fused_qkv_w,
                q_norm_w,
                k_norm_w,
                attn_proj_w,
                ffn_norm_w,
                gate_up_w,
                down_w,
            ));
    }
    let weight_refs: Vec<_> = layer_weights
        .iter()
        .map(|(a, b, c, d, e, f, g, h)| (a, b, c, d, e, f, g, h))
        .collect();
    let final_norm_cached = match (final_norm_handle, final_norm_bytes) {
        (Some(handle), Some(bytes)) => {
            Some(graph.get_or_upload_f32_weight(handle, bytes)?)
        }
        _ => None,
    };
    let lm_head_cached = match (lm_head_handle, lm_head_bytes) {
        (Some(handle), Some(bytes)) => {
            Some(graph.get_or_upload_tq2_weight_soa(handle, bytes)?)
        }
        _ => None,
    };
    graph
        .encode_full_forward_prefill_verify_ternary(
            hidden_batch,
            pos_start,
            batch_size,
            n_layers,
            &weight_refs,
            cos_table,
            sin_table,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            eps,
            max_seq_len,
            final_norm_cached.as_ref(),
            final_norm_eps,
            lm_head_cached.as_ref(),
            lm_head_out_features,
            batch_token_ids_out,
        )
}
