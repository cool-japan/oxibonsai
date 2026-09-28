//! Public full-layer / full-forward Metal entry points.
//!
//! Q1 (1-bit) and ternary (TQ2_0_g128) twins of every entry point live here.
//! The ternary half is keyed per model epoch (`MET-02`: every lookup is
//! `WeightKey::new(lp.model_epoch, kind, slot)`) and has a **cached** shape
//! (`MET-03`): [`build_cached_weights_ternary_only`] resolves every layer's
//! eight buffers once into a [`CachedTernaryWeights`], and the
//! `try_metal_*_ternary_cached` entry points bind those handles directly
//! instead of re-running 8 cache lookups per layer per token.
//!
//! 🤖 Originally generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::metal_graph::{MetalGraph, MetalGraphError, MetalWeightHandle};
use std::sync::Arc;

use super::types::{
    CachedLayerWeights, CachedModelWeights, CachedQ1Weights, CachedTernaryLayerWeights,
    CachedTernaryWeights, FullForwardLayerParams, FullForwardLayerParamsTernary,
    LEGACY_MODEL_EPOCH,
};

/// Bytes per `TQ2_0_g128` block (32 bytes of 2-bit codes + an f16 scale).
const TQ2_BLOCK_BYTES: usize = 34;
/// Weights per `TQ2_0_g128` block.
const TQ2_BLOCK_WEIGHTS: usize = 128;

/// The eight per-layer handles in the order every `encode_*` entry point takes
/// them: `(attn_norm, fused_qkv, q_norm, k_norm, attn_proj, ffn_norm,
/// gate_up, down)`.
pub(crate) type LayerWeightRefs<'w> = (
    &'w Arc<MetalWeightHandle>,
    &'w Arc<MetalWeightHandle>,
    &'w Arc<MetalWeightHandle>,
    &'w Arc<MetalWeightHandle>,
    &'w Arc<MetalWeightHandle>,
    &'w Arc<MetalWeightHandle>,
    &'w Arc<MetalWeightHandle>,
    &'w Arc<MetalWeightHandle>,
);

/// Borrow a ternary layer cache in the tuple shape the encoders take.
pub(crate) fn ternary_layer_refs(layers: &[CachedTernaryLayerWeights]) -> Vec<LayerWeightRefs<'_>> {
    layers
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
        .collect()
}

/// The one weight-cache epoch shared by every layer of a forward (`MET-02`).
///
/// One forward binds one model's weights, so a slice whose layers disagree is
/// a caller bug — rejected rather than resolved against two models at once.
/// An empty slice has no layers to disagree and resolves to
/// [`LEGACY_MODEL_EPOCH`] (only its tail, if any, is looked up).
pub(crate) fn shared_model_epoch(
    layer_params: &[FullForwardLayerParamsTernary<'_>],
) -> Result<u64, MetalGraphError> {
    let Some(first) = layer_params.first() else {
        return Ok(LEGACY_MODEL_EPOCH);
    };
    for (i, lp) in layer_params.iter().enumerate() {
        if lp.model_epoch != first.model_epoch {
            return Err(MetalGraphError::EncodingFailed(format!(
                "ternary layer {i} is keyed under model epoch {} but layer 0 under {}: one \
                 forward pass must bind a single model's weights",
                lp.model_epoch, first.model_epoch
            )));
        }
    }
    Ok(first.model_epoch)
}

/// Resolve (uploading on a miss) one ternary layer's eight buffers, every one
/// keyed `WeightKey::new(lp.model_epoch, kind, slot)` (`MET-02`).
///
/// The gate‖up concatenation is built inside the lazy closure, so a resident
/// slot never materialises it.
pub(crate) fn resolve_ternary_layer(
    graph: &MetalGraph,
    lp: &FullForwardLayerParamsTernary<'_>,
) -> Result<CachedTernaryLayerWeights, MetalGraphError> {
    let epoch = lp.model_epoch;
    let attn_norm =
        graph.get_or_upload_f32_weight_for_epoch(epoch, lp.attn_norm_handle, lp.attn_norm_bytes)?;
    let q_norm =
        graph.get_or_upload_f32_weight_for_epoch(epoch, lp.q_norm_handle, lp.q_norm_bytes)?;
    let k_norm =
        graph.get_or_upload_f32_weight_for_epoch(epoch, lp.k_norm_handle, lp.k_norm_bytes)?;
    let ffn_norm =
        graph.get_or_upload_f32_weight_for_epoch(epoch, lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
    let fused_qkv = graph.get_or_upload_tq2_weight_soa_for_epoch(
        epoch,
        lp.fused_qkv_handle,
        lp.fused_qkv_bytes,
    )?;
    let attn_proj = graph.get_or_upload_tq2_weight_soa_for_epoch(
        epoch,
        lp.attn_proj_handle,
        lp.attn_proj_bytes,
    )?;
    let (gate_bytes, up_bytes) = (lp.gate_bytes, lp.up_bytes);
    let gate_up =
        graph.get_or_upload_tq2_weight_soa_lazy_for_epoch(epoch, lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;
    let down =
        graph.get_or_upload_tq2_weight_soa_for_epoch(epoch, lp.down_handle, lp.down_bytes)?;
    Ok(CachedTernaryLayerWeights {
        attn_norm,
        fused_qkv,
        q_norm,
        k_norm,
        attn_proj,
        ffn_norm,
        gate_up,
        down,
    })
}

/// Resolve the final-norm → LM-head tail under `epoch`, or `None` when the
/// caller passed no tail. Half a tail (one of the two handles without the
/// other) is `None` too, matching the encoders, which run the tail only when
/// both weights are present.
#[allow(clippy::type_complexity)]
pub(crate) fn resolve_ternary_tail(
    graph: &MetalGraph,
    epoch: u64,
    final_norm: Option<(u64, &[f32])>,
    lm_head: Option<(u64, &[u8])>,
) -> Result<
    (
        Option<Arc<MetalWeightHandle>>,
        Option<Arc<MetalWeightHandle>>,
    ),
    MetalGraphError,
> {
    let final_norm = match final_norm {
        Some((slot, bytes)) => Some(graph.get_or_upload_f32_weight_for_epoch(epoch, slot, bytes)?),
        None => None,
    };
    let lm_head = match lm_head {
        Some((slot, bytes)) => {
            Some(graph.get_or_upload_tq2_weight_soa_for_epoch(epoch, slot, bytes)?)
        }
        None => None,
    };
    Ok((final_norm, lm_head))
}

/// Pair an optional slot with its optional bytes (both must be present).
pub(crate) fn tail_part<T: ?Sized>(slot: Option<u64>, bytes: Option<&T>) -> Option<(u64, &T)> {
    match (slot, bytes) {
        (Some(slot), Some(bytes)) => Some((slot, bytes)),
        _ => None,
    }
}

/// Attempt to run the FFN phase via direct Metal dispatch.
///
/// This is the main entry point for `block.rs`. It:
/// 1. Resolves this thread's `MetalGraph` session (`MetalGraph::global()`)
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
/// 1. Resolves this thread's `MetalGraph` session (`MetalGraph::global()`)
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
///
/// Every weight is looked up under the layers' shared `model_epoch`
/// (`MET-02`), the tail included; a slice mixing epochs is rejected. Callers
/// that decode many tokens should build a [`CachedTernaryWeights`] once with
/// [`build_cached_weights_ternary_only`] and use
/// [`try_metal_full_forward_ternary_cached`], which skips these per-call
/// lookups entirely (`MET-03`).
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
    let epoch = shared_model_epoch(layer_params)?;
    let graph = MetalGraph::global()?;
    let layers = layer_params
        .iter()
        .map(|lp| resolve_ternary_layer(&graph, lp))
        .collect::<Result<Vec<_>, _>>()?;
    let weight_refs = ternary_layer_refs(&layers);
    let (final_norm_cached, lm_head_cached) = resolve_ternary_tail(
        &graph,
        epoch,
        tail_part(final_norm_handle, final_norm_bytes),
        tail_part(lm_head_handle, lm_head_bytes),
    )?;
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
    Ok(CachedModelWeights::Q1(CachedQ1Weights {
        layers,
        final_norm,
        lm_head,
    }))
}
/// Build a [`CachedModelWeights::Ternary`] cache for a ternary (TQ2_0_g128)
/// model (`MET-03`).
///
/// The ternary mirror of [`build_cached_weights`]: every layer's eight buffers
/// and the optional final-norm → LM-head tail are resolved **once** — found
/// resident, or uploaded on a miss — under the layers' shared `model_epoch`
/// (`MET-02`), and the returned cache holds only the reference-counted GPU
/// handles. [`try_metal_full_forward_ternary_cached`] and its wrappers then
/// dispatch straight from those handles, with no per-token lookup and no host
/// copy of any weight.
///
/// `final_norm` / `lm_head` are `(slot, bytes)` pairs and must be passed
/// together (a model whose LM head is not ternary passes neither, and its
/// cache has no tail). `lm_head_out_features` is the LM head's row count and
/// must be non-zero exactly when the tail is present.
///
/// # Errors
///
/// [`MetalGraphError::EncodingFailed`] for a half tail, a tail/row-count
/// disagreement or layers keyed under different epochs; otherwise whatever an
/// upload or cache lookup returns (allocation failure, a TQ2 validation
/// failure, a kind mismatch on a slot).
pub fn build_cached_weights_ternary_only(
    layer_params: &[FullForwardLayerParamsTernary<'_>],
    final_norm: Option<(u64, &[f32])>,
    lm_head: Option<(u64, &[u8])>,
    lm_head_out_features: usize,
) -> Result<CachedModelWeights, MetalGraphError> {
    if final_norm.is_some() != lm_head.is_some() {
        return Err(MetalGraphError::EncodingFailed(
            "build_cached_weights_ternary_only: the final norm and the LM head form one tail — \
             pass both or neither"
                .into(),
        ));
    }
    if lm_head.is_some() != (lm_head_out_features > 0) {
        return Err(MetalGraphError::EncodingFailed(format!(
            "build_cached_weights_ternary_only: lm_head_out_features = {lm_head_out_features} \
             disagrees with the tail being {}",
            if lm_head.is_some() {
                "present"
            } else {
                "absent"
            }
        )));
    }
    let model_epoch = shared_model_epoch(layer_params)?;
    let graph = MetalGraph::global()?;
    let layers = layer_params
        .iter()
        .map(|lp| resolve_ternary_layer(&graph, lp))
        .collect::<Result<Vec<_>, _>>()?;
    let (final_norm, lm_head) = resolve_ternary_tail(&graph, model_epoch, final_norm, lm_head)?;
    Ok(CachedModelWeights::Ternary(CachedTernaryWeights {
        model_epoch,
        layers,
        final_norm,
        lm_head,
        lm_head_out_features,
    }))
}

/// Borrow the ternary variant of a model cache, or explain which one it was.
fn ternary_cache<'c>(
    cached: &'c CachedModelWeights,
    what: &str,
) -> Result<&'c CachedTernaryWeights, MetalGraphError> {
    match cached {
        CachedModelWeights::Ternary(tern) => Ok(tern),
        CachedModelWeights::Q1(_) => Err(MetalGraphError::EncodingFailed(format!(
            "{what} invoked with a Q1 weight cache; use the Q1 forward path"
        ))),
    }
}

/// The cached tail to run, given whether the caller asked for an output that
/// needs one (`logits` / greedy token), or `None` for a hidden-state-only
/// forward. Asking for logits from a cache without a tail is an error, not a
/// silent hidden-state download.
#[allow(clippy::type_complexity)]
fn cached_tail<'c>(
    tern: &'c CachedTernaryWeights,
    wants_tail: bool,
    what: &str,
) -> Result<
    (
        Option<&'c Arc<MetalWeightHandle>>,
        Option<&'c Arc<MetalWeightHandle>>,
    ),
    MetalGraphError,
> {
    if !wants_tail {
        return Ok((None, None));
    }
    match (tern.final_norm.as_ref(), tern.lm_head.as_ref()) {
        (Some(norm), Some(head)) if tern.lm_head_out_features > 0 => Ok((Some(norm), Some(head))),
        _ => Err(MetalGraphError::EncodingFailed(format!(
            "{what}: logits / a greedy token were requested but this ternary weight cache has \
             no LM-head tail (the model's LM head is not ternary)"
        ))),
    }
}

/// Cached twin of [`try_metal_full_forward_ternary`] (`MET-03`): binds the
/// handles of a [`CachedTernaryWeights`] built by
/// [`build_cached_weights_ternary_only`] instead of looking every weight up
/// again, and encodes the whole forward into one command buffer.
///
/// The final-norm → LM-head tail runs iff `logits_out` or
/// `greedy_token_id_out` is `Some` (and then the cache must have a tail);
/// with both `None`, the post-layer hidden state is written back to `hidden`
/// for a caller-side tail — the shape `try_metal_full_forward_ternary_inner`
/// uses.
///
/// Because the cache holds the very buffers the non-cached path resolves (same
/// epoch, same slots), the two produce bit-identical results.
///
/// # Errors
///
/// A Q1 cache, a requested tail the cache does not have, or any encode /
/// command-buffer failure.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_ternary_cached(
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
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), MetalGraphError> {
    const WHAT: &str = "try_metal_full_forward_ternary_cached";
    let tern = ternary_cache(cached, WHAT)?;
    let wants_tail = logits_out.is_some() || greedy_token_id_out.is_some();
    let (final_norm_w, lm_head_w) = cached_tail(tern, wants_tail, WHAT)?;
    let graph = MetalGraph::global()?;
    let weight_refs = ternary_layer_refs(&tern.layers);
    graph.encode_full_forward_ternary(
        hidden,
        pos,
        tern.layers.len(),
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
        final_norm_w,
        final_norm_eps,
        lm_head_w,
        tern.lm_head_out_features,
        logits_out,
        greedy_token_id_out,
    )
}

/// Cached twin of [`try_metal_prefill_ternary`]: single-token forward that
/// returns the logits (`MET-03`).
///
/// # Errors
///
/// As [`try_metal_full_forward_ternary_cached`].
#[allow(clippy::too_many_arguments)]
pub fn try_metal_prefill_ternary_cached(
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
    logits_out: &mut Vec<f32>,
) -> Result<(), MetalGraphError> {
    try_metal_full_forward_ternary_cached(
        hidden,
        pos,
        cached,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        final_norm_eps,
        Some(logits_out),
        None,
    )
}

/// Cached twin of [`try_metal_forward_greedy_ternary`]: single-token forward
/// with the argmax on the GPU, downloading only the 4-byte token id
/// (`MET-03`).
///
/// # Errors
///
/// As [`try_metal_full_forward_ternary_cached`].
#[allow(clippy::too_many_arguments)]
pub fn try_metal_forward_greedy_ternary_cached(
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
    greedy_token_id_out: &mut u32,
) -> Result<(), MetalGraphError> {
    try_metal_full_forward_ternary_cached(
        hidden,
        pos,
        cached,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        final_norm_eps,
        None,
        Some(greedy_token_id_out),
    )
}

/// Cached twin of [`try_metal_prefill_verify_ternary`] (single-token
/// speculative verify: GPU argmax only). Identical dispatch to
/// [`try_metal_forward_greedy_ternary_cached`]; kept as its own name so a
/// caller's intent survives in the call site, exactly like the non-cached pair.
///
/// # Errors
///
/// As [`try_metal_full_forward_ternary_cached`].
#[allow(clippy::too_many_arguments)]
pub fn try_metal_prefill_verify_ternary_cached(
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
    greedy_token_id_out: &mut u32,
) -> Result<(), MetalGraphError> {
    try_metal_forward_greedy_ternary_cached(
        hidden,
        pos,
        cached,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        final_norm_eps,
        greedy_token_id_out,
    )
}

/// Cached twin of `try_metal_full_forward_prefill_ternary` — **batched**
/// ternary prefill over `batch_size` prompt tokens in one command buffer,
/// binding a [`CachedTernaryWeights`] instead of looking every weight up per
/// call (`MET-03`).
///
/// Only the last token's logits (or its greedy argmax) are produced, exactly
/// like the non-cached path. Because every weight comes from the cache, this
/// path performs **no** weight-cache lookup at all — which is also what takes
/// the batched prefill off the legacy-keyed lookups the non-cached
/// `metal_prefill` entry still makes (see `MET-02`).
///
/// # Errors
///
/// A Q1 cache, a cache without an LM-head tail, or any encode /
/// command-buffer failure.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill_ternary_cached(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    cached: &CachedModelWeights,
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_eps: f32,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), MetalGraphError> {
    const WHAT: &str = "try_metal_full_forward_prefill_ternary_cached";
    let tern = ternary_cache(cached, WHAT)?;
    let (final_norm_w, lm_head_w) = cached_tail(tern, true, WHAT)?;
    let graph = MetalGraph::global()?;
    let weight_refs = ternary_layer_refs(&tern.layers);
    graph.encode_full_forward_prefill_ternary(
        hidden_batch,
        pos_start,
        batch_size,
        tern.layers.len(),
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
        final_norm_w,
        final_norm_eps,
        lm_head_w,
        tern.lm_head_out_features,
        logits_out,
        greedy_token_id_out,
    )
}

/// Cached twin of `try_metal_full_forward_prefill_verify_ternary` — batched
/// speculative-verify prefill returning every position's greedy argmax
/// (`MET-03`).
///
/// # Errors
///
/// As [`try_metal_full_forward_prefill_ternary_cached`].
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill_verify_ternary_cached(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    cached: &CachedModelWeights,
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_eps: f32,
    batch_token_ids_out: &mut Vec<u32>,
) -> Result<(), MetalGraphError> {
    const WHAT: &str = "try_metal_full_forward_prefill_verify_ternary_cached";
    let tern = ternary_cache(cached, WHAT)?;
    let (final_norm_w, lm_head_w) = cached_tail(tern, true, WHAT)?;
    let graph = MetalGraph::global()?;
    let weight_refs = ternary_layer_refs(&tern.layers);
    graph.encode_full_forward_prefill_verify_ternary(
        hidden_batch,
        pos_start,
        batch_size,
        tern.layers.len(),
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
        final_norm_w,
        final_norm_eps,
        lm_head_w,
        tern.lm_head_out_features,
        batch_token_ids_out,
    )
}

/// Fused ternary GEMV over the **row-wise concatenation** of `aos_parts`
/// (`M-21`): one dispatch for Q‖K‖V (3 parts), gate‖up (2 parts), or any
/// other stack of `TQ2_0_g128` matrices sharing the input width `k`.
///
/// The concatenation is built only on a cache miss and uploaded under
/// `WeightKey::new(model_epoch, WeightKind::Tq2Soa, slot)`; later calls bind
/// the resident buffer. `output[..n_rows]` receives the parts' outputs in
/// order.
///
/// Unlike a bare `get_or_upload` + `encode_gemv_tq2`, this **refuses** a
/// resident buffer whose byte length is not exactly `n_rows × (k / 128) × 34`:
/// a slot another producer filled (a Q-only upload, say) would otherwise be
/// read past its end by the GEMV with `n_rows` recomputing every offset — the
/// defect the CUDA twin was caught with (`M-21` blocking fix). The parts'
/// total length is checked the same way before anything is uploaded.
///
/// # Errors
///
/// [`MetalGraphError::InvalidDimensions`] for `k` not a positive multiple of
/// 128, parts that do not add up to `n_rows` rows, or a resident buffer of the
/// wrong size; otherwise the upload's / dispatch's error.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_gemv_tq2_fused(
    input: &[f32],
    output: &mut [f32],
    model_epoch: u64,
    slot: u64,
    aos_parts: &[&[u8]],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    if k == 0 || !k.is_multiple_of(TQ2_BLOCK_WEIGHTS) {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "try_metal_gemv_tq2_fused: k = {k} must be a positive multiple of {TQ2_BLOCK_WEIGHTS}"
        )));
    }
    let expected_bytes = n_rows
        .checked_mul(k / TQ2_BLOCK_WEIGHTS)
        .and_then(|blocks| blocks.checked_mul(TQ2_BLOCK_BYTES))
        .ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "try_metal_gemv_tq2_fused: n_rows = {n_rows} x k = {k} overflows"
            ))
        })?;
    let parts_bytes: usize = aos_parts.iter().map(|part| part.len()).sum();
    if parts_bytes != expected_bytes {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "try_metal_gemv_tq2_fused: the {} parts hold {parts_bytes} bytes but {n_rows} rows \
             x k = {k} need {expected_bytes} ({TQ2_BLOCK_BYTES} B per {TQ2_BLOCK_WEIGHTS}-weight \
             block)",
            aos_parts.len()
        )));
    }
    let graph = MetalGraph::global()?;
    let weight = graph.get_or_upload_tq2_weight_soa_lazy_for_epoch(model_epoch, slot, || {
        let mut fused = Vec::with_capacity(parts_bytes);
        for part in aos_parts {
            fused.extend_from_slice(part);
        }
        fused
    })?;
    if weight.byte_len() != expected_bytes {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "try_metal_gemv_tq2_fused: slot {slot:#x} (epoch {model_epoch}) holds {} bytes, \
             expected {expected_bytes} for {n_rows} rows x k = {k}; refusing to run the GEMV \
             over a buffer this call did not build",
            weight.byte_len()
        )));
    }
    graph.encode_gemv_tq2(&weight, input, output, n_rows, k)
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
    let q1 = match cached {
        CachedModelWeights::Q1(q1) => q1,
        CachedModelWeights::Ternary(_) => {
            return Err(MetalGraphError::EncodingFailed(
                "try_metal_full_forward_cached invoked with a ternary weight cache; \
                 use the ternary forward path"
                    .to_string(),
            ));
        }
    };
    let n_layers = q1.layers.len();
    let graph = MetalGraph::global()?;
    let weight_refs: Vec<_> = q1
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
        Some(&q1.final_norm),
        final_norm_eps,
        Some(&q1.lm_head),
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

#[cfg(test)]
#[path = "ternary_cache_tests.rs"]
mod ternary_cache_tests;
