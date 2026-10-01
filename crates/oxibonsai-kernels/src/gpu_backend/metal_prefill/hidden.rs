//! Head-free batched Metal prefill: the final-normed hidden state of every
//! prompt row, for the dense embedding pass (`BonsaiModel::forward_hidden`).
//!
//! Every other batched prefill entry point ends in the LM head over the last
//! row and writes the device KV cache of the session the calling thread
//! dispatches in. An embedding needs neither: it wants the `output_norm`
//! output of **every** row, and it must never disturb the KV cache of a
//! generation the same process is decoding (MET-05).
//!
//! # Request-scoped device KV
//!
//! The pass runs in a fresh sibling session
//! ([`MetalGraph::new_sibling_session`]) of the calling thread's session: same
//! device, same compiled pipelines, same weight cache — so the model's
//! resident weights are bound as they are — but its own command queue, its
//! own device KV cache sized to exactly the input, and its own prefill
//! buffers. The session is dropped when the call returns, which frees them.
//! Nothing of the calling session is read or written.
//!
//! # Micro-batches
//!
//! The rows run in [`HIDDEN_PREFILL_MICRO_BATCH`]-row micro-batches, one
//! command buffer each, positions continuing across micro-batches: the keys
//! and values of earlier micro-batches are in the request's KV cache by the
//! time a later one attends. The GEMM family is chosen once for the whole
//! request, so the rows are bit-identical to a single-batch pass.
//!
//! # Autorelease pools
//!
//! The request-scoped session lives and dies inside one `autoreleasepool`:
//! creating its command queue, allocating its KV cache and workspace,
//! every micro-batch (each drains a pool of its own as well) and releasing
//! all of it on return. An embedding server thread therefore keeps none of
//! a request's autoreleased Metal objects once the request returns.

use metal::objc::rc::autoreleasepool;
use std::sync::Arc;

use super::super::metal_full_layer::functions_3::{
    q1_layer_refs, resolve_q1_layer, resolve_ternary_layer, shared_model_epoch,
    shared_q1_model_epoch, ternary_layer_refs,
};
use super::super::metal_full_layer::{
    CachedModelWeights, FullForwardLayerParams, FullForwardLayerParamsTernary,
};
use super::super::metal_graph::{MetalGraph, MetalGraphError, MetalWeightHandle};
use super::functions::{PrefillFormat, PrefillLayerHandles};
use super::types::LayerConfig;

/// Rows per command buffer of the hidden-state prefill.
pub const HIDDEN_PREFILL_MICRO_BATCH: usize = 128;

/// Model geometry of a hidden-state prefill.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HiddenPrefillShape {
    /// Hidden width.
    pub hidden_size: usize,
    /// FFN intermediate width.
    pub intermediate_size: usize,
    /// Query heads.
    pub nq: usize,
    /// Key/value heads.
    pub nkv: usize,
    /// Per-head width.
    pub head_dim: usize,
    /// Block RMSNorm epsilon.
    pub eps: f32,
    /// Final (`output_norm`) RMSNorm epsilon.
    pub final_norm_eps: f32,
}

/// The rows of one hidden-state prefill request.
#[derive(Clone, Copy, Debug)]
pub struct HiddenPrefillInput<'a> {
    /// `[batch x hidden]` token embeddings, row-major (= column-major per
    /// token, the layout every prefill kernel reads).
    pub hidden_batch: &'a [f32],
    /// `[batch x head_dim/2]` RoPE cosines of positions `0..batch`.
    pub cos_table: &'a [f32],
    /// `[batch x head_dim/2]` RoPE sines of positions `0..batch`.
    pub sin_table: &'a [f32],
    /// Rows.
    pub batch_size: usize,
}

/// Reject a request the request-scoped KV cannot serve.
fn check_request(
    pos_start: usize,
    batch_size: usize,
    max_seq_len: usize,
) -> Result<(), MetalGraphError> {
    if pos_start != 0 {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "a hidden-state prefill starts at position 0 (its KV cache holds only its own \
             rows), got pos_start {pos_start}"
        )));
    }
    if batch_size == 0 || batch_size > max_seq_len {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "a hidden-state prefill needs 1..={max_seq_len} rows, got {batch_size}"
        )));
    }
    Ok(())
}

/// Run the head-free prefill on a fresh sibling session of `base` with a
/// device KV cache of exactly `input.batch_size` positions, micro-batched at
/// `micro_batch` rows.
#[allow(clippy::too_many_arguments)]
fn run_hidden_prefill(
    base: &MetalGraph,
    format: PrefillFormat,
    layers: &[PrefillLayerHandles<'_>],
    final_norm: &MetalWeightHandle,
    shape: &HiddenPrefillShape,
    input: &HiddenPrefillInput<'_>,
    micro_batch: usize,
    hidden_out: &mut Vec<f32>,
) -> Result<(), MetalGraphError> {
    autoreleasepool(|| {
        // The KV cache of this session is the request's own, sized to exactly
        // its rows; dropping the session on return frees it.
        let session = base.new_sibling_session();
        session.encode_prefill_hidden_rows(
            format,
            input.hidden_batch,
            0,
            input.batch_size,
            layers.len(),
            layers,
            input.cos_table,
            input.sin_table,
            LayerConfig {
                hidden_size: shape.hidden_size,
                intermediate_size: shape.intermediate_size,
                n_q_heads: shape.nq,
                n_kv_heads: shape.nkv,
                head_dim: shape.head_dim,
                eps: shape.eps,
                max_seq_len: input.batch_size,
            },
            final_norm,
            shape.final_norm_eps,
            micro_batch,
            hidden_out,
        )
    })
}

/// The final-norm weight of a hidden-state prefill as a transient buffer.
fn transient_final_norm(
    graph: &MetalGraph,
    weights: &[f32],
) -> Result<MetalWeightHandle, MetalGraphError> {
    let bytes: Vec<u8> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
    graph.upload_weight(&bytes)
}

/// Head-free batched prefill of a **Q1_0_g128** model: every row's
/// final-normed hidden state into `hidden_out` (`[batch x hidden]`,
/// row-major), in a request-scoped session (see the module docs).
///
/// Takes the inputs of [`super::try_metal_full_forward_prefill`] minus its
/// LM-head parameters. `pos_start` must be `0` — the request's KV cache holds
/// only its own rows — and `max_seq_len` only bounds the request.
///
/// # Errors
///
/// A non-zero `pos_start`, an empty or over-long batch, a layer slice of the
/// wrong length or mixing model epochs, a weight upload failure, any encode
/// or command-buffer failure, or a timeout (see
/// [`MetalGraphError::is_command_buffer_timeout`]).
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill_hidden(
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
    hidden_out: &mut Vec<f32>,
) -> Result<(), MetalGraphError> {
    check_request(pos_start, batch_size, max_seq_len)?;
    check_layer_count(n_layers, layer_params.len())?;
    let epoch = shared_q1_model_epoch(layer_params)?;
    let graph = MetalGraph::global()?;
    let layers = layer_params
        .iter()
        .map(|lp| resolve_q1_layer(&graph, lp))
        .collect::<Result<Vec<_>, _>>()?;
    let final_norm = resolve_final_norm(&graph, epoch, final_norm_handle, final_norm_bytes)?;
    run_hidden_prefill(
        &graph,
        PrefillFormat::OneBit,
        &q1_layer_refs(&layers),
        &final_norm,
        &HiddenPrefillShape {
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            eps,
            final_norm_eps,
        },
        &HiddenPrefillInput {
            hidden_batch,
            cos_table,
            sin_table,
            batch_size,
        },
        HIDDEN_PREFILL_MICRO_BATCH,
        hidden_out,
    )
}

/// Head-free batched prefill of a **TQ2_0_g128** model — the ternary twin of
/// [`try_metal_full_forward_prefill_hidden`], every lookup keyed under the
/// layers' shared `model_epoch` (`MET-02`).
///
/// # Errors
///
/// As [`try_metal_full_forward_prefill_hidden`].
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill_hidden_ternary(
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
    hidden_out: &mut Vec<f32>,
) -> Result<(), MetalGraphError> {
    check_request(pos_start, batch_size, max_seq_len)?;
    check_layer_count(n_layers, layer_params.len())?;
    let epoch = shared_model_epoch(layer_params)?;
    let graph = MetalGraph::global()?;
    let layers = layer_params
        .iter()
        .map(|lp| resolve_ternary_layer(&graph, lp))
        .collect::<Result<Vec<_>, _>>()?;
    let final_norm = resolve_final_norm(&graph, epoch, final_norm_handle, final_norm_bytes)?;
    run_hidden_prefill(
        &graph,
        PrefillFormat::Ternary,
        &ternary_layer_refs(&layers),
        &final_norm,
        &HiddenPrefillShape {
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            eps,
            final_norm_eps,
        },
        &HiddenPrefillInput {
            hidden_batch,
            cos_table,
            sin_table,
            batch_size,
        },
        HIDDEN_PREFILL_MICRO_BATCH,
        hidden_out,
    )
}

/// Head-free batched prefill binding a model's **cached** fused weight set
/// (Q1 or ternary) — no weight lookup, and for Q1 no Q‖K‖V concatenation per
/// call. `final_norm` is the model's `output_norm` weight.
///
/// # Errors
///
/// An empty batch, a failure to upload the final norm, any encode or
/// command-buffer failure, or a timeout.
pub fn try_metal_full_forward_prefill_hidden_cached(
    input: &HiddenPrefillInput<'_>,
    cached: &CachedModelWeights,
    final_norm: &[f32],
    shape: &HiddenPrefillShape,
    hidden_out: &mut Vec<f32>,
) -> Result<(), MetalGraphError> {
    try_metal_full_forward_prefill_hidden_cached_micro(
        input,
        cached,
        final_norm,
        shape,
        HIDDEN_PREFILL_MICRO_BATCH,
        hidden_out,
    )
}

/// [`try_metal_full_forward_prefill_hidden_cached`] at an explicit
/// micro-batch size (the micro-batch-invariance tests run it single-batch).
pub(crate) fn try_metal_full_forward_prefill_hidden_cached_micro(
    input: &HiddenPrefillInput<'_>,
    cached: &CachedModelWeights,
    final_norm: &[f32],
    shape: &HiddenPrefillShape,
    micro_batch: usize,
    hidden_out: &mut Vec<f32>,
) -> Result<(), MetalGraphError> {
    check_request(0, input.batch_size, input.batch_size)?;
    if final_norm.len() != shape.hidden_size {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "final norm has {} weights for hidden size {}",
            final_norm.len(),
            shape.hidden_size
        )));
    }
    let graph = MetalGraph::global()?;
    let norm = transient_final_norm(&graph, final_norm)?;
    let (format, handles) = match cached {
        CachedModelWeights::Q1(q1) => (PrefillFormat::OneBit, q1_layer_refs(&q1.layers)),
        CachedModelWeights::Ternary(tern) => {
            (PrefillFormat::Ternary, ternary_layer_refs(&tern.layers))
        }
    };
    run_hidden_prefill(
        &graph,
        format,
        &handles,
        &norm,
        shape,
        input,
        micro_batch,
        hidden_out,
    )
}

/// Resolve the final RMSNorm weight under `epoch` (`RawF32`), or upload it
/// transiently when the caller gave bytes but no slot.
fn resolve_final_norm(
    graph: &MetalGraph,
    epoch: u64,
    slot: Option<u64>,
    bytes: Option<&[f32]>,
) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
    match (slot, bytes) {
        (Some(slot), Some(bytes)) => graph.get_or_upload_f32_weight_for_epoch(epoch, slot, bytes),
        (None, Some(bytes)) => transient_final_norm(graph, bytes).map(Arc::new),
        (_, None) => Err(MetalGraphError::InvalidDimensions(
            "a hidden-state prefill needs the final norm weights".into(),
        )),
    }
}

/// Reject a `layer_params` slice whose length disagrees with `n_layers`.
fn check_layer_count(n_layers: usize, given: usize) -> Result<(), MetalGraphError> {
    if given != n_layers {
        return Err(MetalGraphError::EncodingFailed(format!(
            "layer_params length mismatch: need {n_layers}, got {given}"
        )));
    }
    Ok(())
}
