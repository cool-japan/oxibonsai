//! Public batched-prefill entry points (`try_metal_full_forward_prefill*`).
//!
//! Every weight lookup here goes through the shared resolvers of
//! `metal_full_layer::functions_3`, so the batched prefill binds exactly the
//! buffers the single-token paths of the same model bind (`MET-02`):
//!
//! - **ternary** — all eight per-layer buffers and the tail are keyed
//!   `WeightKey::new(lp.model_epoch, kind, slot)` ([`resolve_ternary_layer`] /
//!   [`resolve_ternary_tail`]); a slice mixing epochs is rejected;
//! - **Q1** — the norms and the tail are keyed under the layers' shared
//!   `model_epoch`, the projections by their upload-handle ids
//!   ([`resolve_q1_layer`] / [`resolve_q1_tail`]).
//!
//! These lookups used to be spelled out per function against the legacy
//! epoch, which is why a model keyed under its own epoch missed all of its
//! resident buffers on the batched prefill and fell back to the CPU.
//!
//! 🤖 Originally generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::metal_full_layer::functions_3::{
    q1_layer_refs, resolve_q1_layer, resolve_q1_tail, resolve_ternary_layer, resolve_ternary_tail,
    shared_model_epoch, shared_q1_model_epoch, tail_part, ternary_layer_refs,
};
use super::super::metal_full_layer::{FullForwardLayerParams, FullForwardLayerParamsTernary};
use super::super::metal_graph::{MetalGraph, MetalGraphError};

/// Reject a `layer_params` slice whose length disagrees with `n_layers`.
fn check_layer_count(n_layers: usize, given: usize) -> Result<(), MetalGraphError> {
    if given != n_layers {
        return Err(MetalGraphError::EncodingFailed(format!(
            "layer_params length mismatch: need {n_layers}, got {given}"
        )));
    }
    Ok(())
}

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
    check_layer_count(n_layers, layer_params.len())?;
    let epoch = shared_q1_model_epoch(layer_params)?;
    let graph = MetalGraph::global()?;
    let layers = layer_params
        .iter()
        .map(|lp| resolve_q1_layer(&graph, lp))
        .collect::<Result<Vec<_>, _>>()?;
    let weight_refs = q1_layer_refs(&layers);
    let (final_norm_cached, lm_head_cached) = resolve_q1_tail(
        &graph,
        epoch,
        tail_part(final_norm_handle, final_norm_bytes),
        tail_part(lm_head_handle, lm_head_bytes),
    )?;
    graph.encode_full_forward_prefill(
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
    check_layer_count(n_layers, layer_params.len())?;
    let epoch = shared_q1_model_epoch(layer_params)?;
    let graph = MetalGraph::global()?;
    let layers = layer_params
        .iter()
        .map(|lp| resolve_q1_layer(&graph, lp))
        .collect::<Result<Vec<_>, _>>()?;
    let weight_refs = q1_layer_refs(&layers);
    let (final_norm_cached, lm_head_cached) = resolve_q1_tail(
        &graph,
        epoch,
        tail_part(final_norm_handle, final_norm_bytes),
        tail_part(lm_head_handle, lm_head_bytes),
    )?;
    graph.encode_full_forward_prefill_verify(
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
///
/// Every lookup — eight per layer plus the tail — is keyed under the layers'
/// shared `model_epoch` (`MET-02`); a slice mixing epochs is rejected. Callers
/// that hold a `CachedTernaryWeights` use
/// `try_metal_full_forward_prefill_ternary_cached`, which performs no lookup
/// at all (`MET-03`).
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill_ternary(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParamsTernary<'_>],
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
    check_layer_count(n_layers, layer_params.len())?;
    let epoch = shared_model_epoch(layer_params)?;
    let graph = MetalGraph::global()?;
    let layers = layer_params
        .iter()
        .map(|lp| resolve_ternary_layer(&graph, lp))
        .collect::<Result<Vec<_>, _>>()?;
    let weight_refs = ternary_layer_refs(&layers);
    let (final_norm_w, lm_head_w) = resolve_ternary_tail(
        &graph,
        epoch,
        tail_part(final_norm_handle, final_norm_bytes),
        tail_part(lm_head_handle, lm_head_bytes),
    )?;
    graph.encode_full_forward_prefill_ternary(
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
        final_norm_w.as_ref(),
        final_norm_eps,
        lm_head_w.as_ref(),
        lm_head_out_features,
        logits_out,
        greedy_token_id_out,
    )
}
/// Ternary batch prefill **verify** (speculative decoding).
///
/// Runs all layers + final norm + TQ2 LM head on every batch position and
/// returns the per-position greedy argmax token IDs. Keyed exactly like
/// [`try_metal_full_forward_prefill_ternary`].
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill_verify_ternary(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParamsTernary<'_>],
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
    check_layer_count(n_layers, layer_params.len())?;
    let epoch = shared_model_epoch(layer_params)?;
    let graph = MetalGraph::global()?;
    let layers = layer_params
        .iter()
        .map(|lp| resolve_ternary_layer(&graph, lp))
        .collect::<Result<Vec<_>, _>>()?;
    let weight_refs = ternary_layer_refs(&layers);
    let (final_norm_w, lm_head_w) = resolve_ternary_tail(
        &graph,
        epoch,
        tail_part(final_norm_handle, final_norm_bytes),
        tail_part(lm_head_handle, lm_head_bytes),
    )?;
    graph.encode_full_forward_prefill_verify_ternary(
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
        final_norm_w.as_ref(),
        final_norm_eps,
        lm_head_w.as_ref(),
        lm_head_out_features,
        batch_token_ids_out,
    )
}
