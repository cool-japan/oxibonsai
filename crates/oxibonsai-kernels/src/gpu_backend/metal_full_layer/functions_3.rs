//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::metal_graph::{MetalGraph, MetalGraphError, MetalWeightHandle};
use std::sync::Arc;

use super::types::{
    CachedLayerWeights, CachedModelWeights, FullForwardLayerParams, FullForwardLayerParamsTernary,
};

/// Attempt to run the FFN phase via direct Metal dispatch.
///
/// This is the main entry point for `block.rs`. It:
/// 1. Gets the global `MetalGraph` singleton
/// 2. Uploads/caches weights lazily (first call per layer uploads, subsequent calls reuse)
/// 3. Encodes the full 7-op FFN pipeline in one command buffer
///
/// Returns `Ok(())` if the Metal dispatch succeeded.
/// Returns `Err(...)` if Metal is not available or dispatch failed.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_ffn(
    hidden: &mut [f32],
    attn_out: &[f32],
    norm_weight: &[f32],
    eps: f32,
    attn_proj_handle_id: u64,
    attn_proj_bytes: &[u8],
    gate_up_handle_id: u64,
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_handle_id: u64,
    down_bytes: &[u8],
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<(), MetalGraphError> {
    let graph = MetalGraph::global()?;
    let attn_proj_w = graph.get_or_upload_q1_weight_soa(attn_proj_handle_id, attn_proj_bytes)?;
    let gate_up_w = graph.get_or_upload_q1_weight_soa_lazy(gate_up_handle_id, || {
        let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        fused.extend_from_slice(gate_bytes);
        fused.extend_from_slice(up_bytes);
        fused
    })?;
    let down_w = graph.get_or_upload_q1_weight_soa(down_handle_id, down_bytes)?;
    graph.encode_ffn_phase(
        hidden,
        attn_out,
        norm_weight,
        &attn_proj_w,
        &gate_up_w,
        &down_w,
        hidden_size,
        intermediate_size,
        eps,
    )
}
/// Attempt to run a fused QKV projection via direct Metal dispatch.
///
/// This is the main entry point for `block.rs` QKV acceleration. It:
/// 1. Gets the global `MetalGraph` singleton
/// 2. Uploads/caches the fused Q+K+V weight lazily (first call concatenates and uploads)
/// 3. Encodes a single GEMV dispatch in one command buffer
///
/// Returns `Ok(())` if the Metal dispatch succeeded.
/// Returns `Err(...)` if Metal is not available or dispatch failed.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_qkv(
    input: &[f32],
    output: &mut [f32],
    weight_handle_id: u64,
    q_bytes: &[u8],
    k_bytes: &[u8],
    v_bytes: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let graph = MetalGraph::global()?;
    let weight = graph.get_or_upload_q1_weight_soa_lazy(weight_handle_id, || {
        let mut fused = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
        fused.extend_from_slice(q_bytes);
        fused.extend_from_slice(k_bytes);
        fused.extend_from_slice(v_bytes);
        fused
    })?;
    graph.encode_qkv_phase(input, output, &weight, n_rows, k)
}
/// Attempt to run a full transformer layer via direct Metal dispatch.
///
/// This encodes the complete attention + FFN pipeline for one transformer
/// layer into a single Metal command buffer, eliminating per-kernel
/// CPU→GPU synchronisation overhead.
///
/// Returns `Ok(())` on success. Returns `Err(...)` if Metal is unavailable
/// or any dispatch step fails.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_layer(
    hidden: &mut [f32],
    pos: usize,
    layer_idx: usize,
    attn_norm_handle_id: u64,
    attn_norm_bytes: &[f32],
    fused_qkv_handle_id: u64,
    fused_qkv_bytes: &[u8],
    q_norm_handle_id: u64,
    q_norm_bytes: &[f32],
    k_norm_handle_id: u64,
    k_norm_bytes: &[f32],
    attn_proj_handle_id: u64,
    attn_proj_bytes: &[u8],
    ffn_norm_handle_id: u64,
    ffn_norm_bytes: &[f32],
    gate_up_handle_id: u64,
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_handle_id: u64,
    down_bytes: &[u8],
    rope_cos: &[f32],
    rope_sin: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    n_layers: usize,
) -> Result<(), MetalGraphError> {
    let graph = MetalGraph::global()?;
    let attn_norm_w = graph.get_or_upload_f32_weight(attn_norm_handle_id, attn_norm_bytes)?;
    let q_norm_w = graph.get_or_upload_f32_weight(q_norm_handle_id, q_norm_bytes)?;
    let k_norm_w = graph.get_or_upload_f32_weight(k_norm_handle_id, k_norm_bytes)?;
    let ffn_norm_w = graph.get_or_upload_f32_weight(ffn_norm_handle_id, ffn_norm_bytes)?;
    let fused_qkv_w = graph.get_or_upload_q1_weight_soa(fused_qkv_handle_id, fused_qkv_bytes)?;
    let attn_proj_w = graph.get_or_upload_q1_weight_soa(attn_proj_handle_id, attn_proj_bytes)?;
    let gate_up_w = graph.get_or_upload_q1_weight_soa_lazy(gate_up_handle_id, || {
        let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        fused.extend_from_slice(gate_bytes);
        fused.extend_from_slice(up_bytes);
        fused
    })?;
    let down_w = graph.get_or_upload_q1_weight_soa(down_handle_id, down_bytes)?;
    graph.encode_full_layer(
        hidden,
        pos,
        layer_idx,
        &attn_norm_w,
        &fused_qkv_w,
        &q_norm_w,
        &k_norm_w,
        &attn_proj_w,
        &ffn_norm_w,
        &gate_up_w,
        &down_w,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        n_layers,
    )
}
/// Attempt to run ALL transformer layers in a single Metal command buffer.
///
/// This encodes the complete attention + FFN pipeline for all `n_layers`
/// layers into one command buffer, eliminating N-1 GPU scheduling events
/// compared to the per-layer path.
///
/// When `final_norm_handle` and `lm_head_handle` are both `Some`, the final
/// RMSNorm and LM head GEMV are appended to the same command buffer,
/// eliminating an additional CPU→GPU round trip.  In that case, logits are
/// written to `logits_out` and `hidden` is NOT updated.
///
/// When `greedy_token_id_out` is `Some`, argmax is performed on the GPU after
/// the LM head GEMV and only the resulting token ID (4 bytes) is downloaded
/// instead of the full logits vector (~607KB), dramatically reducing PCIe/
/// memory bandwidth overhead for greedy (temperature=0) decoding.
///
/// Returns `Ok(())` on success. Returns `Err(...)` if Metal is unavailable
/// or any dispatch step fails.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward(
    hidden: &mut [f32],
    pos: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParams<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
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
        return Err(MetalGraphError::EncodingFailed(format!(
            "layer_params length mismatch: need {n_layers}, got {}",
            layer_params.len()
        )));
    }
    let graph = MetalGraph::global()?;
    #[allow(clippy::type_complexity)]
    let mut layer_weights: Vec<(
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
    )> = Vec::with_capacity(n_layers);
    for lp in layer_params {
        let attn_norm_w =
            graph.get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph.get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph.get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph.get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
        let fused_qkv_w =
            graph.get_or_upload_q1_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w =
            graph.get_or_upload_q1_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph.get_or_upload_q1_weight_soa_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;
        let down_w = graph.get_or_upload_q1_weight_soa(lp.down_handle, lp.down_bytes)?;
        layer_weights.push((
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
        (Some(handle), Some(bytes)) => Some(graph.get_or_upload_f32_weight(handle, bytes)?),
        _ => None,
    };
    let lm_head_cached = match (lm_head_handle, lm_head_bytes) {
        (Some(handle), Some(bytes)) => Some(graph.get_or_upload_q1_weight_soa(handle, bytes)?),
        _ => None,
    };
    graph.encode_full_forward(
        hidden,
        pos,
        n_layers,
        &weight_refs,
        rope_cos,
        rope_sin,
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
/// Ternary twin of [`try_metal_full_forward`] — every attention/FFN GEMV
/// dispatches through the TQ2 Metal kernel and the whole forward pass is
/// encoded into a single command buffer.
///
/// The final-norm / LM-head tail uses `encode_tail_and_commit_ternary`,
/// dispatching the LM head via `dispatch_gemv_tq2` (TQ2_0_g128 ternary weight).
/// Pass `None` for both `final_norm_*` and `lm_head_*` parameters to skip
/// the tail — the caller then runs the CPU final-norm + LM-head path.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_ternary(
    hidden: &mut [f32],
    pos: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParamsTernary<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
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
        return Err(MetalGraphError::EncodingFailed(format!(
            "layer_params length mismatch: need {n_layers}, got {}",
            layer_params.len()
        )));
    }
    let graph = MetalGraph::global()?;
    #[allow(clippy::type_complexity)]
    let mut layer_weights: Vec<(
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
    )> = Vec::with_capacity(n_layers);
    for lp in layer_params {
        let attn_norm_w =
            graph.get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph.get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph.get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph.get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
        let fused_qkv_w =
            graph.get_or_upload_tq2_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w =
            graph.get_or_upload_tq2_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph.get_or_upload_tq2_weight_soa_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;
        let down_w = graph.get_or_upload_tq2_weight_soa(lp.down_handle, lp.down_bytes)?;
        layer_weights.push((
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
        (Some(handle), Some(bytes)) => Some(graph.get_or_upload_f32_weight(handle, bytes)?),
        _ => None,
    };
    let lm_head_cached = match (lm_head_handle, lm_head_bytes) {
        (Some(handle), Some(bytes)) => Some(graph.get_or_upload_tq2_weight_soa(handle, bytes)?),
        _ => None,
    };
    graph.encode_full_forward_ternary(
        hidden,
        pos,
        n_layers,
        &weight_refs,
        rope_cos,
        rope_sin,
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
/// Build the cached weight handles from layer params (called once on first token).
/// This does all the QKV concatenation, AoS→SoA conversion, and GPU upload.
pub fn build_cached_weights(
    layer_params: &[FullForwardLayerParams<'_>],
    final_norm_handle: u64,
    final_norm_bytes: &[f32],
    lm_head_handle: u64,
    lm_head_bytes: &[u8],
) -> Result<CachedModelWeights, MetalGraphError> {
    let graph = MetalGraph::global()?;
    let mut layers = Vec::with_capacity(layer_params.len());
    for lp in layer_params {
        let attn_norm = graph.get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm = graph.get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm = graph.get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm = graph.get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
        let fused_qkv =
            graph.get_or_upload_q1_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj =
            graph.get_or_upload_q1_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up = graph.get_or_upload_q1_weight_soa_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;
        let down = graph.get_or_upload_q1_weight_soa(lp.down_handle, lp.down_bytes)?;
        layers.push(CachedLayerWeights {
            attn_norm,
            fused_qkv,
            q_norm,
            k_norm,
            attn_proj,
            ffn_norm,
            gate_up,
            down,
        });
    }
    let final_norm = graph.get_or_upload_f32_weight(final_norm_handle, final_norm_bytes)?;
    let lm_head = graph.get_or_upload_q1_weight_soa(lm_head_handle, lm_head_bytes)?;
    Ok(CachedModelWeights {
        layers,
        final_norm,
        lm_head,
        // Q1 path: ternary fields are unused.
        ternary_qkv_concats: Vec::new(),
        ternary_attn_proj_bytes: Vec::new(),
        ternary_gate_bytes: Vec::new(),
        ternary_up_bytes: Vec::new(),
        ternary_down_bytes: Vec::new(),
        ternary_lm_head_bytes: Vec::new(),
        ternary_lm_head_out_features: 0,
    })
}
/// Build a `CachedModelWeights` shell for ternary (TQ2_0_g128) models.
///
/// Ternary models do **not** use the Q1 `layers` / `final_norm` / `lm_head`
/// handles at runtime — those fields exist on the struct only because it is
/// shared with the Q1 path.  This constructor avoids calling
/// `get_or_upload_q1_weight_soa` (which enforces 18-byte block alignment) by
/// uploading trivial f32 placeholders under dedicated handle IDs in the
/// ternary-reserved namespace (`4_000_000` / `4_000_001`).
///
/// The real ternary weights are stored in the `ternary_*` Vec fields and
/// rebuilt into `FullForwardLayerParamsTernary` structs on every decode call.
pub fn build_cached_weights_ternary_only(
    ternary_qkv_concats: Vec<Vec<u8>>,
    ternary_attn_proj_bytes: Vec<Vec<u8>>,
    ternary_gate_bytes: Vec<Vec<u8>>,
    ternary_up_bytes: Vec<Vec<u8>>,
    ternary_down_bytes: Vec<Vec<u8>>,
    ternary_lm_head_bytes: Vec<u8>,
    ternary_lm_head_out_features: usize,
) -> Result<CachedModelWeights, MetalGraphError> {
    let graph = MetalGraph::global()?;
    // Trivial f32 placeholder uploads — no block-size constraint, never used
    // for actual inference on the ternary path.
    let dummy_f32 = [0.0_f32];
    let final_norm = graph.get_or_upload_f32_weight(4_000_000u64, &dummy_f32)?;
    let lm_head_placeholder = graph.get_or_upload_f32_weight(4_000_001u64, &dummy_f32)?;
    Ok(CachedModelWeights {
        layers: Vec::new(),
        final_norm,
        lm_head: lm_head_placeholder,
        ternary_qkv_concats,
        ternary_attn_proj_bytes,
        ternary_gate_bytes,
        ternary_up_bytes,
        ternary_down_bytes,
        ternary_lm_head_bytes,
        ternary_lm_head_out_features,
    })
}
/// Like `try_metal_full_forward`, but uses pre-cached GPU weight handles.
/// Eliminates ALL per-token weight lookup, upload, and allocation overhead.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_cached(
    hidden: &mut [f32],
    pos: usize,
    cached: &CachedModelWeights,
    rope_cos: &[f32],
    rope_sin: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_eps: f32,
    lm_head_out_features: usize,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), MetalGraphError> {
    let n_layers = cached.layers.len();
    let graph = MetalGraph::global()?;
    let weight_refs: Vec<_> = cached
        .layers
        .iter()
        .map(|lw| {
            (
                &lw.attn_norm,
                &lw.fused_qkv,
                &lw.q_norm,
                &lw.k_norm,
                &lw.attn_proj,
                &lw.ffn_norm,
                &lw.gate_up,
                &lw.down,
            )
        })
        .collect();
    graph.encode_full_forward(
        hidden,
        pos,
        n_layers,
        &weight_refs,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        Some(&cached.final_norm),
        final_norm_eps,
        Some(&cached.lm_head),
        lm_head_out_features,
        logits_out,
        greedy_token_id_out,
    )
}
/// Ternary prefill wrapper: runs all layers + final norm + TQ2 LM head, returning the logits.
///
/// This is a thin convenience wrapper around [`try_metal_full_forward_ternary`] that
/// presets the output mode for prefill (logits returned, no greedy argmax). The caller
/// receives the full `lm_head_out_features`-length logits vector in `logits_out`.
///
/// Use when the model needs sampling (top-p / top-k) after the forward pass.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_prefill_ternary(
    hidden: &mut [f32],
    pos: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParamsTernary<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
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
    logits_out: &mut Vec<f32>,
) -> Result<(), MetalGraphError> {
    try_metal_full_forward_ternary(
        hidden,
        pos,
        n_layers,
        layer_params,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        final_norm_handle,
        final_norm_bytes,
        final_norm_eps,
        lm_head_handle,
        lm_head_bytes,
        lm_head_out_features,
        Some(logits_out),
        None,
    )
}
/// Ternary prefill-verify wrapper: runs all layers + final norm + TQ2 LM head + GPU argmax.
///
/// Thin convenience wrapper around [`try_metal_full_forward_ternary`] that presets the
/// output mode for speculative-decoding verification: greedy argmax is performed on the
/// GPU and only the 4-byte token ID is downloaded, rather than the full logits vector.
///
/// Use when verifying a speculative draft token — only the winning token ID is needed.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_prefill_verify_ternary(
    hidden: &mut [f32],
    pos: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParamsTernary<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
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
    greedy_token_id_out: &mut u32,
) -> Result<(), MetalGraphError> {
    try_metal_full_forward_ternary(
        hidden,
        pos,
        n_layers,
        layer_params,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        final_norm_handle,
        final_norm_bytes,
        final_norm_eps,
        lm_head_handle,
        lm_head_bytes,
        lm_head_out_features,
        None,
        Some(greedy_token_id_out),
    )
}
/// Ternary greedy-decoding wrapper: runs all layers + final norm + TQ2 LM head + GPU argmax.
///
/// Thin convenience wrapper around [`try_metal_full_forward_ternary`] that presets the
/// output mode for greedy (temperature = 0) autoregressive decoding: argmax is performed
/// on the GPU and only the winning token ID (4 bytes) is downloaded, dramatically reducing
/// PCIe / memory-bandwidth overhead compared to downloading the full logits vector.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_forward_greedy_ternary(
    hidden: &mut [f32],
    pos: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParamsTernary<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
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
    greedy_token_id_out: &mut u32,
) -> Result<(), MetalGraphError> {
    try_metal_full_forward_ternary(
        hidden,
        pos,
        n_layers,
        layer_params,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        final_norm_handle,
        final_norm_bytes,
        final_norm_eps,
        lm_head_handle,
        lm_head_bytes,
        lm_head_out_features,
        None,
        Some(greedy_token_id_out),
    )
}
