//! Public entry point for the K-quant CUDA batch prefill path.
//!
//! Exposes [`try_cuda_prefill_k_quant`] — the K-quant analogue of
//! [`try_cuda_prefill`] / [`try_cuda_prefill_q_std`]. Replaces the sequential
//! single-token GEMV fallback for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and Q8_K
//! models.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaView};

use crate::gpu_backend::cuda_full_layer::{
    acquire_full_layer_buffers, get_or_upload_f32_weight, init_attn_modules,
};
use crate::gpu_backend::cuda_graph::{CudaGraph, CudaGraphError};
use crate::gpu_backend::cuda_prefill::init_prefill_modules;

use super::encode::encode_k_quant_prefill_layer;
use super::launchers::{
    launch_gemm_q2k, launch_gemm_q3k, launch_gemm_q4k, launch_gemm_q5k, launch_gemm_q6k,
    launch_gemm_q8k,
};
use super::state::{
    acquire_k_quant_kv_cache, acquire_k_quant_logits, acquire_k_quant_prefill_buffers,
    init_k_quant_prefill_modules, CudaKQuantPrefillLayerParams, KQuantFormat,
};

// =============================================================================
// Public entry point: try_cuda_prefill_k_quant
// =============================================================================

/// Batch prefill for K-quant quantised models (Q2_K through Q8_K).
///
/// Processes `batch_size` tokens simultaneously using real fused batch GEMM
/// kernels for all linear projections.  Attention is processed per-token
/// sequentially using the shared single-token attention kernels.
///
/// # Constraints
///
/// - `hidden_size` must be a multiple of 256 (QK_K requirement).
/// - All layer weight byte slices must be valid AoS K-quant super-block data.
///
/// # Arguments
///
/// - `hidden_batch` — host-side batched hidden states `[batch_size × hidden_size]`
///   in row-major (token-major) layout. Uploaded to GPU in column-major format.
/// - `logits_out` / `greedy_token_id_out` — if `Some`, runs final norm and LM
///   head for the last token and returns full logits or the argmax token id.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_prefill_k_quant(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[CudaKQuantPrefillLayerParams<'_>],
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
    lm_head_fmt: KQuantFormat,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), CudaGraphError> {
    if batch_size == 0 {
        return Ok(());
    }

    // K-quant requires hidden_size to be a multiple of 256 (= QK_K).
    if hidden_size % 256 != 0 {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "K-quant prefill: hidden_size={hidden_size} must be a multiple of 256"
        )));
    }

    let graph = CudaGraph::global()?;

    let kq_mods = init_k_quant_prefill_modules(&graph)?;
    let pmods = init_prefill_modules(&graph)?;
    let attn_mods = init_attn_modules(&graph)?;

    // Upload / cache all layer weights.
    struct LayerWeightHandles {
        attn_norm: Arc<CudaSlice<f32>>,
        fused_qkv: Arc<CudaSlice<u8>>,
        q_norm: Arc<CudaSlice<f32>>,
        k_norm: Arc<CudaSlice<f32>>,
        attn_proj: Arc<CudaSlice<u8>>,
        ffn_norm: Arc<CudaSlice<f32>>,
        gate_up: Arc<CudaSlice<u8>>,
        down: Arc<CudaSlice<u8>>,
    }

    let mut layer_weights: Vec<LayerWeightHandles> = Vec::with_capacity(n_layers);
    for lp in layer_params.iter().take(n_layers) {
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph.get_or_upload_weight_aos_raw_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;

        layer_weights.push(LayerWeightHandles {
            attn_norm: get_or_upload_f32_weight(&graph, lp.attn_norm_handle, lp.attn_norm_bytes)?,
            fused_qkv: graph
                .get_or_upload_weight_aos_raw(lp.fused_qkv_handle, lp.fused_qkv_bytes)?,
            q_norm: get_or_upload_f32_weight(&graph, lp.q_norm_handle, lp.q_norm_bytes)?,
            k_norm: get_or_upload_f32_weight(&graph, lp.k_norm_handle, lp.k_norm_bytes)?,
            attn_proj: graph
                .get_or_upload_weight_aos_raw(lp.attn_proj_handle, lp.attn_proj_bytes)?,
            ffn_norm: get_or_upload_f32_weight(&graph, lp.ffn_norm_handle, lp.ffn_norm_bytes)?,
            gate_up: gate_up_w,
            down: graph.get_or_upload_weight_aos_raw(lp.down_handle, lp.down_bytes)?,
        });
    }

    // Allocate / acquire the batched prefill activation buffers.
    let mut pb_guard = acquire_k_quant_prefill_buffers(
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
        .ok_or_else(|| CudaGraphError::DriverError("prefill buffer not allocated kquant".into()))?;

    // Allocate / acquire the KV cache.
    let mut kv_guard = acquire_k_quant_kv_cache(&graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("KV cache not allocated kquant".into()))?;

    // Acquire single-token full-layer buffers for sequential attention.
    let mut st_guard = acquire_full_layer_buffers(
        &graph,
        hidden_size,
        nq,
        nkv,
        head_dim,
        max_seq_len,
        intermediate_size,
    )?;
    let st_bufs = st_guard.as_mut().ok_or_else(|| {
        CudaGraphError::DriverError("full-layer buffer not allocated kquant".into())
    })?;

    // Upload the hidden batch to GPU in column-major layout.
    {
        let mut col_major = vec![0.0f32; batch_size * hidden_size];
        for t in 0..batch_size {
            for e in 0..hidden_size {
                col_major[t * hidden_size + e] = hidden_batch[t * hidden_size + e];
            }
        }
        let n = batch_size * hidden_size;
        let mut dst_view = pb.d_input.slice_mut(0..n);
        graph
            .stream_arc()
            .memcpy_htod(&col_major, &mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("upload hidden_batch kquant: {e}")))?;
    }

    // Determine fallback format from first layer (or Q4K if no layers).
    let default_fmt = layer_params
        .first()
        .map_or(KQuantFormat::Q4K, |lp| lp.format);

    // Run each transformer layer.
    for (layer_idx, lw) in layer_weights.iter().enumerate() {
        let fmt = layer_params
            .get(layer_idx)
            .map_or(default_fmt, |lp| lp.format);

        unsafe {
            encode_k_quant_prefill_layer(
                &graph,
                &kq_mods,
                &pmods,
                &attn_mods,
                &lw.attn_norm,
                &lw.fused_qkv,
                &lw.q_norm,
                &lw.k_norm,
                &lw.attn_proj,
                &lw.ffn_norm,
                &lw.gate_up,
                &lw.down,
                kv,
                layer_idx,
                pos_start,
                pb,
                st_bufs,
                cos_table,
                sin_table,
                heads_per_group,
                eps,
                fmt,
            )?;
        }
    }

    // ─── Final norm + LM head (optional) ─────────────────────────────────────
    if logits_out.is_some() || greedy_token_id_out.is_some() {
        let final_norm_h = final_norm_handle.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("final_norm_handle required for logits kquant".into())
        })?;
        let final_norm_b = final_norm_bytes.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("final_norm_bytes required for logits kquant".into())
        })?;
        let lm_head_h = lm_head_handle.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("lm_head_handle required for logits kquant".into())
        })?;
        let lm_head_b = lm_head_bytes.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("lm_head_bytes required for logits kquant".into())
        })?;

        let d_final_norm_w = get_or_upload_f32_weight(&graph, final_norm_h, final_norm_b)?;
        let d_lm_head = graph.get_or_upload_weight_aos_raw(lm_head_h, lm_head_b)?;

        // Extract last token's hidden state into st_bufs.d_hidden.
        let last_t = batch_size - 1;
        {
            let src_view: CudaView<f32> = pb
                .d_input
                .slice(last_t * hidden_size..(last_t + 1) * hidden_size);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                .map_err(|e| {
                    CudaGraphError::DriverError(format!("copy last hidden kquant lm: {e}"))
                })?;
        }

        // Final RMSNorm on the last token's hidden state.
        unsafe {
            graph
                .launch_rmsnorm_pub(
                    &st_bufs.d_hidden,
                    &d_final_norm_w,
                    &mut st_bufs.d_normed,
                    hidden_size as u32,
                    final_norm_eps,
                )
                .map_err(|e| CudaGraphError::DriverError(format!("final norm kquant: {e:?}")))?;
        }

        // LM head: batch GEMM with batch_size=1.
        // d_normed holds the normed last-token hidden state [hidden_size].
        // GEMM output d_logits[row] = sum over k (treating d_normed as 1-column input).
        let mut lm_logits_guard = acquire_k_quant_logits(&graph, lm_head_out_features)?;
        let d_logits = &mut lm_logits_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("logits buf not allocated kquant".into()))?
            .0;

        // Zero logits buffer before GEMM (kernels accumulate with +=).
        {
            let mut d_logits_view = d_logits.slice_mut(0..lm_head_out_features);
            graph
                .stream_arc()
                .memset_zeros(&mut d_logits_view)
                .map_err(|e| CudaGraphError::DriverError(format!("zero logits kquant: {e}")))?;
        }

        // Run batch GEMM with batch_size=1.
        let bs_one = 1u32;
        unsafe {
            match lm_head_fmt {
                KQuantFormat::Q2K => launch_gemm_q2k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
                KQuantFormat::Q3K => launch_gemm_q3k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
                KQuantFormat::Q4K => launch_gemm_q4k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
                KQuantFormat::Q5K => launch_gemm_q5k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
                KQuantFormat::Q6K => launch_gemm_q6k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
                KQuantFormat::Q8K => launch_gemm_q8k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
            }
        }

        // Synchronise stream before D2H copy.
        graph
            .stream_arc()
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("sync kquant lm: {e}")))?;

        let logits_host = graph
            .stream_arc()
            .clone_dtoh(d_logits)
            .map_err(|e| CudaGraphError::DriverError(format!("dtoh logits kquant: {e}")))?;

        drop(lm_logits_guard);

        if let Some(out) = greedy_token_id_out {
            *out = logits_host
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        } else if let Some(out) = logits_out {
            *out = logits_host;
        }

        return Ok(());
    }

    // No LM head requested — just synchronise.
    graph
        .stream_arc()
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("sync kquant end: {e}")))?;

    Ok(())
}
