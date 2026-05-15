//! Public entry points for the CUDA prefill path.
//!
//! Exposes:
//!  - [`try_cuda_prefill`] — Q1 (1-bit) batch prefill.
//!  - [`try_cuda_prefill_ternary`] — TQ2 (ternary) batch prefill.
//!
//! Both functions mirror `try_metal_full_forward_prefill` and handle the full
//! per-call lifecycle: weight upload (cached via `get_or_upload_*`), buffer
//! acquisition, layer-by-layer encode dispatch, optional final-norm + LM-head
//! GEMV on the last token, stream sync, and either greedy-token-id readback
//! or full-logits readback.

use cudarc::driver::CudaView;

use crate::gpu_backend::cuda_full_layer::{
    init_attn_modules, profiling, CudaFullForwardLayerParams, CudaFullForwardLayerParamsTernary,
};
use crate::gpu_backend::cuda_graph::{CudaGraph, CudaGraphError};

use super::encode_q1::encode_prefill_layer;
use super::encode_ternary::encode_prefill_layer_ternary;
use super::state::{
    acquire_prefill_buffers, acquire_prefill_kv_cache, acquire_prefill_logits,
    acquire_single_token_buffers, init_prefill_modules, LayerWeightArcs,
};

// =============================================================================
// Public entry point
// =============================================================================

/// Attempt to run batch prefill (ALL transformer layers + LM head) via CUDA.
///
/// Processes `batch_size` tokens simultaneously using GEMM kernels for
/// projections and sequential per-token attention within each layer.
/// Only the last token's logits are returned in `logits_out` / `greedy_token_id_out`.
///
/// Mirrors `try_metal_full_forward_prefill` exactly.
///
/// Returns `Ok(())` on success.  Returns `Err(...)` if CUDA is unavailable or
/// any kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_prefill(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[CudaFullForwardLayerParams<'_>],
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
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
) -> Result<(), CudaGraphError> {
    if layer_params.len() != n_layers {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "layer_params length mismatch: need {n_layers}, got {}",
            layer_params.len()
        )));
    }

    let half_dim = head_dim / 2;

    if hidden_batch.len() < batch_size * hidden_size {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "hidden_batch too short: need {}, got {}",
            batch_size * hidden_size,
            hidden_batch.len()
        )));
    }
    if cos_table.len() < batch_size * half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "cos_table too short: need {}, got {}",
            batch_size * half_dim,
            cos_table.len()
        )));
    }
    if sin_table.len() < batch_size * half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "sin_table too short: need {}, got {}",
            batch_size * half_dim,
            sin_table.len()
        )));
    }

    let graph = CudaGraph::global()?;
    let _t_prefill = profiling().then(std::time::Instant::now);
    let pmods = init_prefill_modules(&graph)?;
    let attn_mods = init_attn_modules(&graph)?;

    // ── Upload / cache all per-layer weights ────────────────────────────
    let mut layer_weight_arcs: Vec<LayerWeightArcs> = Vec::with_capacity(n_layers);

    for lp in layer_params {
        let attn_norm_w =
            graph.get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph.get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph.get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph.get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
        let fused_qkv_w =
            graph.get_or_upload_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w =
            graph.get_or_upload_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;

        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph.get_or_upload_weight_soa_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;

        let down_w = graph.get_or_upload_weight_soa(lp.down_handle, lp.down_bytes)?;

        layer_weight_arcs.push((
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

    // ── Acquire activation buffers ───────────────────────────────────────
    let mut pb_guard = acquire_prefill_buffers(
        &graph,
        batch_size,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        max_seq_len,
    )?;
    let pb = pb_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("prefill_buffers not allocated".into()))?;

    let mut kv_guard = acquire_prefill_kv_cache(&graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("kv_cache not allocated".into()))?;

    let mut st_guard = acquire_single_token_buffers(
        &graph,
        hidden_size,
        nq,
        nkv,
        head_dim,
        max_seq_len,
        intermediate_size,
    )?;
    let st_bufs = st_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("st_buffers not allocated".into()))?;

    // ── Upload hidden batch → GPU ────────────────────────────────────────
    graph
        .stream_arc()
        .memcpy_htod(&hidden_batch[..batch_size * hidden_size], &mut pb.d_input)
        .map_err(|e| CudaGraphError::DriverError(format!("upload hidden_batch: {e}")))?;

    // ── Encode all layers ────────────────────────────────────────────────
    for (layer_idx, lwa) in layer_weight_arcs.iter().enumerate() {
        unsafe {
            encode_prefill_layer(
                &graph,
                &pmods,
                &attn_mods,
                &lwa.0, // attn_norm
                &lwa.1, // fused_qkv
                &lwa.2, // q_norm
                &lwa.3, // k_norm
                &lwa.4, // attn_proj
                &lwa.5, // ffn_norm
                &lwa.6, // gate_up
                &lwa.7, // down
                kv,
                layer_idx,
                pos_start,
                pb,
                st_bufs,
                cos_table,
                sin_table,
                heads_per_group,
                eps,
            )?;
        }
    }

    // ── Final norm + LM head on last token ──────────────────────────────
    if let (Some(fn_handle), Some(fn_bytes)) = (final_norm_handle, final_norm_bytes) {
        let d_final_norm_w = graph.get_or_upload_f32_weight(fn_handle, fn_bytes)?;

        if let (Some(lm_handle), Some(lm_bytes), true) =
            (lm_head_handle, lm_head_bytes, lm_head_out_features > 0)
        {
            let d_lm_head_w = graph.get_or_upload_weight_soa(lm_handle, lm_bytes)?;

            // Extract last token's hidden state (column batch_size-1)
            let last_col_start = (batch_size - 1) * hidden_size;
            let last_col_end = last_col_start + hidden_size;

            // Upload last token's hidden to st_bufs.d_hidden for single-token norm + GEMV
            {
                let src_view: CudaView<f32> = pb.d_input.slice(last_col_start..last_col_end);
                graph
                    .stream_arc()
                    .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                    .map_err(|e| CudaGraphError::DriverError(format!("copy last hidden: {e}")))?;
            }

            // Single-token final RMSNorm
            unsafe {
                graph.launch_rmsnorm_pub(
                    &st_bufs.d_hidden,
                    &d_final_norm_w,
                    &mut st_bufs.d_normed,
                    hidden_size as u32,
                    final_norm_eps,
                )?;
            }

            // Acquire (or reuse) the cached logits buffer.
            let mut logits_guard = acquire_prefill_logits(&graph, lm_head_out_features)?;
            let d_logits = &mut logits_guard
                .as_mut()
                .ok_or_else(|| CudaGraphError::DriverError("logits buf not allocated".into()))?
                .0;

            // LM head GEMV (single token)
            unsafe {
                graph.launch_gemv_pub(
                    &d_lm_head_w,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                )?;
            }

            // Synchronise stream before D2H
            graph
                .stream_arc()
                .synchronize()
                .map_err(|e| CudaGraphError::DriverError(format!("prefill sync: {e}")))?;

            if let Some(out) = greedy_token_id_out {
                let logits_host = graph
                    .stream_arc()
                    .clone_dtoh(d_logits)
                    .map_err(|e| CudaGraphError::DriverError(format!("dtoh logits: {e}")))?;
                *out = logits_host
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
            } else if let Some(out) = logits_out {
                let logits_host = graph
                    .stream_arc()
                    .clone_dtoh(d_logits)
                    .map_err(|e| CudaGraphError::DriverError(format!("dtoh logits: {e}")))?;
                *out = logits_host;
            }

            if profiling() {
                eprintln!(
                    "[cuda-prof] prefill batch={batch_size} pos_start={pos_start}: {:.1}ms (with lm_head)",
                    _t_prefill.expect("profiling").elapsed().as_secs_f64() * 1000.0
                );
            }
            return Ok(());
        }
    }

    // No final norm / LM head requested — just synchronise and return.
    graph
        .stream_arc()
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("prefill sync end: {e}")))?;

    if profiling() {
        eprintln!(
            "[cuda-prof] prefill batch={batch_size} pos_start={pos_start}: {:.1}ms",
            _t_prefill.expect("profiling").elapsed().as_secs_f64() * 1000.0
        );
    }

    Ok(())
}

// =============================================================================
// Ternary public entry point
// =============================================================================

/// Batch prefill (ALL transformer layers + LM head) via CUDA for TQ2 ternary models.
///
/// Mirrors [`try_cuda_prefill`] but uses TQ2 GEMM/GEMV launchers throughout.
/// Caller-supplied handles are used as-is (same namespace as ternary decode 5M–7M).
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_prefill_ternary(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[CudaFullForwardLayerParamsTernary<'_>],
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
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
) -> Result<(), CudaGraphError> {
    if layer_params.len() != n_layers {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "layer_params length mismatch (ternary prefill): need {n_layers}, got {}",
            layer_params.len()
        )));
    }

    let half_dim = head_dim / 2;

    if hidden_batch.len() < batch_size * hidden_size {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "hidden_batch too short (ternary): need {}, got {}",
            batch_size * hidden_size,
            hidden_batch.len()
        )));
    }
    if cos_table.len() < batch_size * half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "cos_table too short (ternary): need {}, got {}",
            batch_size * half_dim,
            cos_table.len()
        )));
    }
    if sin_table.len() < batch_size * half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "sin_table too short (ternary): need {}, got {}",
            batch_size * half_dim,
            sin_table.len()
        )));
    }

    let graph = CudaGraph::global()?;
    let _t_prefill = profiling().then(std::time::Instant::now);
    let pmods = init_prefill_modules(&graph)?;
    let attn_mods = init_attn_modules(&graph)?;

    // ── Upload / cache all per-layer TQ2 weights ────────────────────────
    // Use same type as Q1 path but with TQ2 upload functions.
    let mut layer_weight_arcs: Vec<LayerWeightArcs> = Vec::with_capacity(n_layers);

    for lp in layer_params {
        let attn_norm_w =
            graph.get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph.get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph.get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph.get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;

        let fused_qkv_w =
            graph.get_or_upload_weight_tq2_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w =
            graph.get_or_upload_weight_tq2_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;

        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph.get_or_upload_weight_tq2_soa_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;

        let down_w = graph.get_or_upload_weight_tq2_soa(lp.down_handle, lp.down_bytes)?;

        layer_weight_arcs.push((
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

    // ── Acquire activation buffers ───────────────────────────────────────
    let mut pb_guard = acquire_prefill_buffers(
        &graph,
        batch_size,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        max_seq_len,
    )?;
    let pb = pb_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("prefill_buffers not allocated".into()))?;

    let mut kv_guard = acquire_prefill_kv_cache(&graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("kv_cache not allocated".into()))?;

    let mut st_guard = acquire_single_token_buffers(
        &graph,
        hidden_size,
        nq,
        nkv,
        head_dim,
        max_seq_len,
        intermediate_size,
    )?;
    let st_bufs = st_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("st_buffers not allocated".into()))?;

    // ── Upload hidden batch → GPU ────────────────────────────────────────
    graph
        .stream_arc()
        .memcpy_htod(&hidden_batch[..batch_size * hidden_size], &mut pb.d_input)
        .map_err(|e| CudaGraphError::DriverError(format!("upload hidden_batch tq2: {e}")))?;

    // ── Encode all layers ────────────────────────────────────────────────
    for (layer_idx, lwa) in layer_weight_arcs.iter().enumerate() {
        unsafe {
            encode_prefill_layer_ternary(
                &graph,
                &pmods,
                &attn_mods,
                &lwa.0, // attn_norm
                &lwa.1, // fused_qkv (TQ2)
                &lwa.2, // q_norm
                &lwa.3, // k_norm
                &lwa.4, // attn_proj (TQ2)
                &lwa.5, // ffn_norm
                &lwa.6, // gate_up (TQ2)
                &lwa.7, // down (TQ2)
                kv,
                layer_idx,
                pos_start,
                pb,
                st_bufs,
                cos_table,
                sin_table,
                heads_per_group,
                eps,
            )?;
        }
    }

    // ── Final norm + TQ2 LM head on last token ───────────────────────────
    if let (Some(fn_handle), Some(fn_bytes)) = (final_norm_handle, final_norm_bytes) {
        let d_final_norm_w = graph.get_or_upload_f32_weight(fn_handle, fn_bytes)?;

        if let (Some(lm_handle), Some(lm_bytes), true) =
            (lm_head_handle, lm_head_bytes, lm_head_out_features > 0)
        {
            let d_lm_head_w = graph.get_or_upload_weight_tq2_soa(lm_handle, lm_bytes)?;

            // Extract last token's hidden state (column batch_size-1)
            let last_col_start = (batch_size - 1) * hidden_size;
            let last_col_end = last_col_start + hidden_size;

            {
                let src_view: CudaView<f32> = pb.d_input.slice(last_col_start..last_col_end);
                graph
                    .stream_arc()
                    .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                    .map_err(|e| {
                        CudaGraphError::DriverError(format!("copy last hidden tq2: {e}"))
                    })?;
            }

            // Single-token final RMSNorm
            unsafe {
                graph.launch_rmsnorm_pub(
                    &st_bufs.d_hidden,
                    &d_final_norm_w,
                    &mut st_bufs.d_normed,
                    hidden_size as u32,
                    final_norm_eps,
                )?;
            }

            // Acquire (or reuse) the cached logits buffer.
            let mut logits_guard = acquire_prefill_logits(&graph, lm_head_out_features)?;
            let d_logits = &mut logits_guard
                .as_mut()
                .ok_or_else(|| CudaGraphError::DriverError("logits buf not allocated".into()))?
                .0;

            // TQ2 LM head GEMV (single token)
            unsafe {
                graph.launch_gemv_tq2_v1_pub(
                    &d_lm_head_w,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                )?;
            }

            // Synchronise stream before D2H
            graph
                .stream_arc()
                .synchronize()
                .map_err(|e| CudaGraphError::DriverError(format!("tq2 prefill sync: {e}")))?;

            if let Some(out) = greedy_token_id_out {
                let logits_host = graph
                    .stream_arc()
                    .clone_dtoh(d_logits)
                    .map_err(|e| CudaGraphError::DriverError(format!("dtoh tq2 logits: {e}")))?;
                *out = logits_host
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
            } else if let Some(out) = logits_out {
                let logits_host = graph
                    .stream_arc()
                    .clone_dtoh(d_logits)
                    .map_err(|e| CudaGraphError::DriverError(format!("dtoh tq2 logits: {e}")))?;
                *out = logits_host;
            }

            if profiling() {
                eprintln!(
                    "[cuda-prof] tq2 prefill batch={batch_size} pos_start={pos_start}: {:.1}ms (with lm_head)",
                    _t_prefill.expect("profiling").elapsed().as_secs_f64() * 1000.0
                );
            }
            return Ok(());
        }
    }

    // No final norm / LM head requested — just synchronise and return.
    graph
        .stream_arc()
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("tq2 prefill sync end: {e}")))?;

    if profiling() {
        eprintln!(
            "[cuda-prof] tq2 prefill batch={batch_size} pos_start={pos_start}: {:.1}ms",
            _t_prefill.expect("profiling").elapsed().as_secs_f64() * 1000.0
        );
    }

    Ok(())
}
