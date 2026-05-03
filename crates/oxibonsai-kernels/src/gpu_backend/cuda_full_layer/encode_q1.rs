//! Q1 full-layer encode functions for the CUDA backend.
//!
//! Contains `encode_full_layer`, `try_cuda_full_layer`, `encode_full_forward`,
//! `try_cuda_full_forward`, `try_cuda_full_forward_with_gpu_lm_head`, and the
//! private helper `encode_layer_device`.

use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::CudaSlice;
use tracing::warn;

use super::super::cuda_graph::{CudaGraph, CudaGraphError};
use super::{
    acquire_full_layer_buffers, acquire_kv_cache, full_layer_state, get_or_build_model_weights,
    get_or_upload_f32_weight, init_attn_modules, profiling, CuGraphHolder, CudaAttnModules,
    CudaCachedLayerWeights, CudaFullForwardLayerParams, CudaFullLayerBuffers, CudaKvCache,
};

use super::launchers::{
    launch_batched_attn_scores_v2, launch_batched_attn_weighted_sum, launch_batched_softmax,
    launch_fused_kv_store, launch_fused_qk_norm_rope,
};

// =============================================================================
// encode_full_layer
// =============================================================================

/// Encode a complete transformer layer (attention + FFN) on the CUDA stream.
///
/// On entry `hidden[..hidden_size]` contains the current residual stream.
/// On return the same slice has been updated in-place.
#[allow(clippy::too_many_arguments)]
pub fn encode_full_layer(
    graph: &CudaGraph,
    hidden: &mut [f32],
    pos: usize,
    layer_idx: usize,
    d_pre_attn_norm: &CudaSlice<f32>,
    d_fused_qkv_weight: &Arc<CudaSlice<u8>>,
    d_o_weight: &Arc<CudaSlice<u8>>,
    d_q_norm: &CudaSlice<f32>,
    d_k_norm: &CudaSlice<f32>,
    d_post_attn_norm: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    rope_cos: &[f32],
    rope_sin: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    max_seq_len: usize,
    n_layers: usize,
    attn_mods: &CudaAttnModules,
) -> Result<(), CudaGraphError> {
    let h = hidden_size;
    let half_dim = head_dim / 2;

    if hidden.len() < h {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "hidden too short: need {h}, got {}",
            hidden.len()
        )));
    }
    if rope_cos.len() < half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "rope_cos too short: need {half_dim}, got {}",
            rope_cos.len()
        )));
    }
    if rope_sin.len() < half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "rope_sin too short: need {half_dim}, got {}",
            rope_sin.len()
        )));
    }

    let mut fl_guard =
        acquire_full_layer_buffers(graph, h, nq, nkv, head_dim, max_seq_len, intermediate_size)?;
    let bufs = fl_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("full_layer_buffers not allocated".into()))?;

    let mut kv_guard = acquire_kv_cache(graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("kv_cache not allocated".into()))?;

    // Upload host data to GPU.
    graph
        .stream_arc()
        .memcpy_htod(&hidden[..h], &mut bufs.d_hidden)
        .map_err(|e| CudaGraphError::DriverError(format!("upload hidden: {e}")))?;
    graph
        .stream_arc()
        .memcpy_htod(&rope_cos[..half_dim], &mut bufs.d_cos)
        .map_err(|e| CudaGraphError::DriverError(format!("upload cos: {e}")))?;
    graph
        .stream_arc()
        .memcpy_htod(&rope_sin[..half_dim], &mut bufs.d_sin)
        .map_err(|e| CudaGraphError::DriverError(format!("upload sin: {e}")))?;

    // Attention sublayer.
    unsafe {
        super::encode_attn_phase(
            graph,
            attn_mods,
            d_pre_attn_norm,
            d_fused_qkv_weight,
            d_q_norm,
            d_k_norm,
            kv,
            layer_idx,
            pos,
            nq,
            nkv,
            head_dim,
            heads_per_group,
            norm_eps,
            h,
            bufs,
        )?;
    }

    let h_u32 = h as u32;
    let inter_u32 = intermediate_size as u32;

    unsafe {
        // O-projection: d_attn_out -> d_normed (scratch)
        let attn_out_rows = (nq * head_dim) as u32;
        graph.launch_gemv_pub(
            d_o_weight,
            &bufs.d_attn_out,
            &mut bufs.d_normed,
            h_u32,
            attn_out_rows,
        )?;
        // residual_add: hidden += O_proj
        graph.launch_residual_add_pub(&mut bufs.d_hidden, &bufs.d_normed, h_u32)?;
        // FFN RMSNorm: hidden -> normed
        graph.launch_rmsnorm_pub(
            &bufs.d_hidden,
            d_post_attn_norm,
            &mut bufs.d_normed,
            h_u32,
            norm_eps,
        )?;
        // Gate+Up GEMV: normed -> d_gate_up (2*inter rows, k=h)
        graph.launch_gemv_pub(
            d_gate_up_weight,
            &bufs.d_normed,
            &mut bufs.d_gate_up,
            2 * inter_u32,
            h_u32,
        )?;
        // SwiGLU: d_gate_up -> d_swiglu
        graph.launch_swiglu_pub(&bufs.d_gate_up, &mut bufs.d_swiglu, inter_u32)?;
        // Down GEMV: swiglu -> d_normed (h rows, k=inter)
        graph.launch_gemv_pub(
            d_down_weight,
            &bufs.d_swiglu,
            &mut bufs.d_normed,
            h_u32,
            inter_u32,
        )?;
        // residual_add: hidden += down_output
        graph.launch_residual_add_pub(&mut bufs.d_hidden, &bufs.d_normed, h_u32)?;
    }

    // Synchronise and download result.
    graph
        .stream_arc()
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("fl stream sync: {e}")))?;
    graph
        .stream_arc()
        .memcpy_dtoh(&bufs.d_hidden, &mut hidden[..h])
        .map_err(|e| CudaGraphError::DriverError(format!("download hidden fl: {e}")))?;
    graph
        .stream_arc()
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("fl D2H sync: {e}")))?;

    Ok(())
}

// =============================================================================
// Public entry point
// =============================================================================

/// Attempt to run a full transformer layer (attention + FFN) via CUDA.
///
/// Mirrors `try_metal_full_layer` exactly.  Returns `Ok(())` on success,
/// `Err(CudaGraphError)` if CUDA is unavailable or any kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_full_layer(
    hidden: &mut [f32],
    pos: usize,
    layer_idx: usize,
    pre_attn_norm_handle_id: u64,
    pre_attn_norm_bytes: &[f32],
    fused_qkv_handle_id: u64,
    fused_qkv_bytes: &[u8],
    o_handle_id: u64,
    o_bytes: &[u8],
    q_norm_handle_id: u64,
    q_norm_bytes: &[f32],
    k_norm_handle_id: u64,
    k_norm_bytes: &[f32],
    post_attn_norm_handle_id: u64,
    post_attn_norm_bytes: &[f32],
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
    heads_per_group: usize,
    norm_eps: f32,
    max_seq_len: usize,
    n_layers: usize,
) -> Result<(), CudaGraphError> {
    let graph = CudaGraph::global()?;
    let attn_mods = init_attn_modules(&graph)?;

    let d_fused_qkv_weight =
        graph.get_or_upload_weight_soa(fused_qkv_handle_id, fused_qkv_bytes)?;
    let d_o_weight = graph.get_or_upload_weight_soa(o_handle_id, o_bytes)?;
    let d_gate_up_weight = graph.get_or_upload_weight_soa_lazy(gate_up_handle_id, || {
        let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        fused.extend_from_slice(gate_bytes);
        fused.extend_from_slice(up_bytes);
        fused
    })?;
    let d_down_weight = graph.get_or_upload_weight_soa(down_handle_id, down_bytes)?;

    let d_pre_attn_norm =
        get_or_upload_f32_weight(&graph, pre_attn_norm_handle_id, pre_attn_norm_bytes)?;
    let d_post_attn_norm =
        get_or_upload_f32_weight(&graph, post_attn_norm_handle_id, post_attn_norm_bytes)?;
    let d_q_norm = get_or_upload_f32_weight(&graph, q_norm_handle_id, q_norm_bytes)?;
    let d_k_norm = get_or_upload_f32_weight(&graph, k_norm_handle_id, k_norm_bytes)?;

    encode_full_layer(
        &graph,
        hidden,
        pos,
        layer_idx,
        &d_pre_attn_norm,
        &d_fused_qkv_weight,
        &d_o_weight,
        &d_q_norm,
        &d_k_norm,
        &d_post_attn_norm,
        &d_gate_up_weight,
        &d_down_weight,
        rope_cos,
        rope_sin,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        heads_per_group,
        norm_eps,
        max_seq_len,
        n_layers,
        &attn_mods,
    )
}

// =============================================================================
// encode_layer_device  (pure GPU, no upload/download)
// =============================================================================

/// Run one transformer layer entirely on device, reading/writing `bufs.d_hidden`
/// in-place.  Unlike `encode_full_layer` this function performs **no** H2D/D2H
/// transfers — the caller is responsible for uploading before the first layer
/// and downloading after the last layer.
///
/// RoPE buffers (`bufs.d_cos` / `bufs.d_sin`) must already be populated by the
/// caller before the first layer call.
///
/// # Safety
/// All device pointers must be valid and allocated on the same CUDA stream.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_layer_device(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    weights: &CudaCachedLayerWeights,
    kv: &mut CudaKvCache,
    layer_idx: usize,
    _pos: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    hidden_size: usize,
    intermediate_size: usize,
    bufs: &mut CudaFullLayerBuffers,
) -> Result<(), CudaGraphError> {
    let h_u32 = hidden_size as u32;
    let nq_u32 = nq as u32;
    let nkv_u32 = nkv as u32;
    let hd_u32 = head_dim as u32;
    let inter_u32 = intermediate_size as u32;
    let qkv_total_rows = (nq * head_dim + 2 * nkv * head_dim) as u32;
    let heads_per_group_u32 = heads_per_group as u32;
    let max_seq_u32 = bufs.max_seq as u32;
    let inv_sqrt_hd = 1.0f32 / (head_dim as f32).sqrt();
    let layer_offset = kv.layer_offset_elements(layer_idx);

    // ── Attention sublayer ────────────────────────────────────────────────

    // Step 1: RMSNorm(d_hidden → d_normed)
    graph.launch_rmsnorm_pub(
        &bufs.d_hidden,
        &weights.pre_attn_norm,
        &mut bufs.d_normed,
        h_u32,
        norm_eps,
    )?;

    // Step 2: Fused QKV GEMV (d_normed → d_qkv)
    graph.launch_gemv_pub(
        &weights.q_weight,
        &bufs.d_normed,
        &mut bufs.d_qkv,
        qkv_total_rows,
        h_u32,
    )?;

    // Step 3: Fused QK-Norm + RoPE
    let k_offset = nq * head_dim;
    let k_in_view = bufs.d_qkv.slice(k_offset..);
    launch_fused_qk_norm_rope(
        graph,
        mods,
        &bufs.d_qkv,
        &k_in_view,
        &mut bufs.d_q_rope,
        &mut bufs.d_k_rope,
        &weights.q_norm,
        &weights.k_norm,
        &bufs.d_cos,
        &bufs.d_sin,
        nq_u32,
        nkv_u32,
        hd_u32,
        norm_eps,
    )?;

    // Step 4: Fused KV-Store — pos read from d_pos_seqlen[0] by the kernel
    let v_offset = (nq + nkv) * head_dim;
    let v_view = bufs.d_qkv.slice(v_offset..);
    launch_fused_kv_store(
        graph,
        mods,
        &bufs.d_k_rope,
        &v_view,
        &mut kv.k_cache,
        &mut kv.v_cache,
        hd_u32,
        nkv_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        layer_offset,
    )?;

    // Step 5: Batched attention scores V2 — seq_len read from d_pos_seqlen[1]
    launch_batched_attn_scores_v2(
        graph,
        mods,
        &bufs.d_q_rope,
        &kv.k_cache,
        &mut bufs.d_scores,
        hd_u32,
        nq_u32,
        nkv_u32,
        heads_per_group_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        inv_sqrt_hd,
        layer_offset,
    )?;

    // Step 6: Softmax — seq_len read from d_pos_seqlen[1]
    launch_batched_softmax(
        graph,
        mods,
        &mut bufs.d_scores,
        nq_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
    )?;

    // Step 7: Weighted sum — seq_len read from d_pos_seqlen[1]
    launch_batched_attn_weighted_sum(
        graph,
        mods,
        &bufs.d_scores,
        &kv.v_cache,
        &mut bufs.d_attn_out,
        hd_u32,
        nq_u32,
        nkv_u32,
        heads_per_group_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        layer_offset,
    )?;

    // ── FFN sublayer ──────────────────────────────────────────────────────

    // O-projection + residual fused: GEMV(attn_out) + hidden (in-place).
    // V8 kernel handles k=nq*head_dim (4096) with shared-mem.
    let attn_out_rows = (nq * head_dim) as u32;
    graph.launch_gemv_residual_pub(
        &weights.o_weight,
        &bufs.d_attn_out,
        &mut bufs.d_hidden,
        h_u32,
        attn_out_rows,
    )?;
    // FFN RMSNorm: hidden → normed
    graph.launch_rmsnorm_pub(
        &bufs.d_hidden,
        &weights.post_attn_norm,
        &mut bufs.d_normed,
        h_u32,
        norm_eps,
    )?;
    // Fused gate+up GEMV + SwiGLU: normed → d_swiglu (1 kernel vs 2)
    graph.launch_fused_gate_up_swiglu_pub(
        &weights.gate_up_weight,
        &bufs.d_normed,
        &mut bufs.d_swiglu,
        inter_u32,
        h_u32,
    )?;
    // Down projection + residual fused: GEMV(swiglu) + hidden (in-place).
    // V9 kernel handles k=intermediate_size (14336) beyond shared-mem limit.
    graph.launch_gemv_residual_pub(
        &weights.down_weight,
        &bufs.d_swiglu,
        &mut bufs.d_hidden,
        h_u32,
        inter_u32,
    )?;

    Ok(())
}

// =============================================================================
// encode_full_forward  (all layers, no intermediate CPU syncs)
// =============================================================================

/// Run the full forward pass (all layers) on GPU without intermediate syncs.
///
/// # CUDA Graph acceleration
///
/// After the first token, the entire 36-layer kernel sequence is captured as a
/// replayable CUDA driver graph (`CUgraphExec`).  On every subsequent token:
/// 1. Per-token inputs (`hidden_init`, `rope_cos/sin`, `d_pos_seqlen`) are
///    uploaded via regular H2D on the same stream.
/// 2. `cuGraphLaunch` submits the pre-built execution graph to the stream;
///    CUDA guarantees the kernels execute after the preceding H2D ops.
/// 3. A single `cuStreamSynchronize` + D2H transfer yields the result.
///
/// This eliminates ~40ms of per-kernel scheduling overhead (468 launches × ~85 µs
/// at 300 MHz SM clock) for a projected **2× decode speedup**.
///
/// # Graph validity
///
/// The captured graph stores raw device pointers to `d_hidden`, `d_cos`,
/// `d_sin`, `d_pos_seqlen`, `d_scores`, the KV cache, and all weight slices.
/// `acquire_full_layer_buffers` invalidates the graph whenever buffer dimensions
/// change (model switch), so the pointers always remain valid during replay.
///
/// Returns the final hidden state (post-norm if `final_norm_weight` provided).
#[allow(clippy::too_many_arguments)]
pub fn encode_full_forward(
    graph: &Arc<CudaGraph>,
    hidden_init: &[f32],
    all_layer_weights: &[CudaCachedLayerWeights],
    rope_cos: &[f32],
    rope_sin: &[f32],
    pos: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    hidden_size: usize,
    intermediate_size: usize,
    max_seq_len: usize,
    final_norm_weight: Option<&[f32]>,
    final_norm_handle: u64,
) -> Result<Vec<f32>, CudaGraphError> {
    let h = hidden_size;
    let half_dim = head_dim / 2;
    let n_layers = all_layer_weights.len();
    let h_u32 = h as u32;

    if hidden_init.len() < h {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "hidden_init too short: need {h}, got {}",
            hidden_init.len()
        )));
    }
    if rope_cos.len() < half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "rope_cos too short: need {half_dim}, got {}",
            rope_cos.len()
        )));
    }
    if rope_sin.len() < half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "rope_sin too short: need {half_dim}, got {}",
            rope_sin.len()
        )));
    }
    if n_layers == 0 {
        return Err(CudaGraphError::WeightLayoutError(
            "encode_full_forward: no layers provided".into(),
        ));
    }

    let attn_mods = init_attn_modules(graph)?;

    // Allocate / reuse activation buffers.
    let mut fl_guard =
        acquire_full_layer_buffers(graph, h, nq, nkv, head_dim, max_seq_len, intermediate_size)?;
    let bufs = fl_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("full_layer_buffers not allocated".into()))?;

    // Allocate / reuse KV cache.
    let mut kv_guard = acquire_kv_cache(graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("kv_cache not allocated".into()))?;

    let stream = graph.stream_arc();

    // Upload [pos, seq_len] to device buffer (8 bytes).  Written on every token
    // so the captured graph reads the correct values at replay time.
    // Upload [pos, seq_len] to the 2-element device buffer.  Updated every token
    // so the replayed graph reads the correct position at replay time.
    let pos_seqlen_host = [pos as u32, (pos + 1) as u32];
    unsafe {
        graph
            .raw_htod(&pos_seqlen_host, &mut bufs.d_pos_seqlen, 2)
            .map_err(|e| CudaGraphError::DriverError(format!("upload pos_seqlen: {e}")))?;
    }

    // ── Fast path: replay captured CUDA graph (every token after the first) ─
    {
        let graph_guard = full_layer_state()
            .cuda_driver_graph
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(Some(ref holder)) = *graph_guard {
            // Upload per-token inputs on the same stream BEFORE graph launch.
            // CUDA stream ordering ensures these H2D copies complete before the
            // first kernel node in the replayed graph begins executing.
            unsafe {
                graph.raw_htod(&hidden_init[..h], &mut bufs.d_hidden, h)?;
                graph.raw_htod(&rope_cos[..half_dim], &mut bufs.d_cos, half_dim)?;
                graph.raw_htod(&rope_sin[..half_dim], &mut bufs.d_sin, half_dim)?;
            }

            // Submit the entire 36-layer kernel sequence as one driver call.
            unsafe { holder.launch() }
                .map_err(|e| CudaGraphError::DriverError(format!("graph launch: {e}")))?;

            // Enqueue the D2H copy on the same stream — it is automatically
            // ordered after the graph nodes.  One synchronise at the end is
            // sufficient; a separate sync before D2H is not needed.
            let mut result = vec![0.0f32; h];
            unsafe { graph.raw_dtoh(&bufs.d_hidden, &mut result, h)? }
            stream
                .synchronize()
                .map_err(|e| CudaGraphError::DriverError(format!("fast-path sync: {e}")))?;

            return Ok(result);
        }
    } // cuda_driver_graph lock released

    // ── Slow path: first call (or after buffer realloc) ──────────────────────
    // Execute all layers normally, then capture the kernel sequence for replay.

    // Upload initial hidden state and RoPE tables via raw H2D.
    unsafe {
        graph
            .raw_htod(&hidden_init[..h], &mut bufs.d_hidden, h)
            .map_err(|e| CudaGraphError::DriverError(format!("upload hidden_init: {e}")))?;
        graph
            .raw_htod(&rope_cos[..half_dim], &mut bufs.d_cos, half_dim)
            .map_err(|e| CudaGraphError::DriverError(format!("upload cos ff: {e}")))?;
        graph
            .raw_htod(&rope_sin[..half_dim], &mut bufs.d_sin, half_dim)
            .map_err(|e| CudaGraphError::DriverError(format!("upload sin ff: {e}")))?;
    }

    // Loop over all layers — pure device computation.
    for (layer_idx, weights) in all_layer_weights.iter().enumerate() {
        unsafe {
            encode_layer_device(
                graph,
                &attn_mods,
                weights,
                kv,
                layer_idx,
                pos,
                nq,
                nkv,
                head_dim,
                heads_per_group,
                norm_eps,
                h,
                intermediate_size,
                bufs,
            )?;
        }
    }

    // Optional final RMSNorm.
    if let Some(fnorm_data) = final_norm_weight {
        // Ensure weight is cached before capture (so capture sees only kernel launches).
        let d_fnorm = get_or_upload_f32_weight(graph, final_norm_handle, fnorm_data)?;
        unsafe {
            graph.launch_rmsnorm_pub(
                &bufs.d_hidden,
                &d_fnorm,
                &mut bufs.d_normed,
                h_u32,
                norm_eps,
            )?;
        }
        // Copy normed → hidden so d_hidden holds the final output for download.
        stream
            .memcpy_dtod(&bufs.d_normed, &mut bufs.d_hidden)
            .map_err(|e| CudaGraphError::DriverError(format!("dtod normed->hidden: {e}")))?;
    }

    // Enqueue D2H on the same stream — ordered after all preceding kernels.
    // One synchronise after the async copy is sufficient; no pre-D2H sync needed.
    let mut result = vec![0.0f32; h];
    unsafe { graph.raw_dtoh(&bufs.d_hidden, &mut result, h)? }
    stream
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("ff D2H sync: {e}")))?;

    // ── Capture the kernel sequence as a replayable CUDA driver graph ─────────
    // Best-effort: failures here are non-fatal.  We already have a valid
    // `result` from the normal execution above — the graph only accelerates
    // subsequent tokens.
    //
    // Event tracking is permanently disabled (CudaGraph::new), so no
    // cuEventRecord/cuStreamWaitEvent calls are injected.  Capture sees only
    // clean kernel-launch nodes and D2D memcpy nodes.
    {
        if let Ok(ref mut graph_guard) = full_layer_state().cuda_driver_graph.lock() {
            // Only attempt capture if this is the first time (None).
            // Some(None) means a previous attempt failed — don't retry.
            if graph_guard.is_none() {
                let begin_ok = stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
                    .is_ok();
                if !begin_ok {
                    warn!("CUDA graph: begin_capture failed — running without graph replay");
                    // Mark as tried-but-failed so we never retry.
                    **graph_guard = Some(None);
                } else {
                    // Record all layers.  Kernels are NOT executed during capture.
                    let record_ok: bool = (|| -> Result<(), CudaGraphError> {
                        for (layer_idx, weights) in all_layer_weights.iter().enumerate() {
                            unsafe {
                                encode_layer_device(
                                    graph,
                                    &attn_mods,
                                    weights,
                                    kv,
                                    layer_idx,
                                    pos,
                                    nq,
                                    nkv,
                                    head_dim,
                                    heads_per_group,
                                    norm_eps,
                                    h,
                                    intermediate_size,
                                    bufs,
                                )?;
                            }
                        }
                        // Record final norm if present (weight already cached).
                        if let Some(fnorm_data) = final_norm_weight {
                            let d_fnorm =
                                get_or_upload_f32_weight(graph, final_norm_handle, fnorm_data)?;
                            unsafe {
                                graph.launch_rmsnorm_pub(
                                    &bufs.d_hidden,
                                    &d_fnorm,
                                    &mut bufs.d_normed,
                                    h_u32,
                                    norm_eps,
                                )?;
                            }
                            // D2D copy → MemcpyNode in the captured graph.
                            stream
                                .memcpy_dtod(&bufs.d_normed, &mut bufs.d_hidden)
                                .map_err(|e| {
                                    CudaGraphError::DriverError(format!("dtod (capture): {e}"))
                                })?;
                        }
                        Ok(())
                    })()
                    .is_ok();

                    // Must call end_capture whenever begin_capture succeeded.
                    // Leaving the stream in capture mode corrupts future ops.
                    // We call the raw cudarc result functions directly to pass 0 (no flags)
                    // to cuGraphInstantiateWithFlags — the cudarc enum has no 0 variant so
                    // we can't use stream.end_capture() without triggering a debug-mode
                    // enum-validity panic.
                    let end_result =
                        unsafe { cudarc::driver::result::stream::end_capture(stream.cu_stream()) };
                    match end_result {
                        Ok(cu_graph_raw) if !cu_graph_raw.is_null() && record_ok => {
                            let inst_result = unsafe {
                                let mut exec = std::mem::MaybeUninit::uninit();
                                sys::cuGraphInstantiateWithFlags(
                                    exec.as_mut_ptr(),
                                    cu_graph_raw,
                                    0u64,
                                )
                                .result()
                                .map(|_| exec.assume_init())
                            };
                            match inst_result {
                                Ok(cu_graph_exec) => {
                                    let holder = CuGraphHolder {
                                        cu_graph: cu_graph_raw,
                                        cu_graph_exec,
                                        stream: Arc::clone(stream),
                                    };
                                    match unsafe { holder.upload() } {
                                        Ok(()) => {
                                            **graph_guard = Some(Some(holder));
                                            tracing::debug!(
                                                "CUDA graph captured and uploaded successfully"
                                            );
                                        }
                                        Err(e) => {
                                            warn!(
                                                "CUDA graph upload failed: {e} — disabling replay"
                                            );
                                            **graph_guard = Some(None);
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!("CUDA graph instantiate failed: {e} — disabling replay");
                                    unsafe {
                                        let _ =
                                            cudarc::driver::result::graph::destroy(cu_graph_raw);
                                    }
                                    **graph_guard = Some(None);
                                }
                            }
                        }
                        Ok(_) => {
                            warn!("CUDA graph: end_capture returned no graph — disabling replay");
                            **graph_guard = Some(None);
                        }
                        Err(e) => {
                            warn!("CUDA graph: end_capture error: {e} — disabling replay");
                            **graph_guard = Some(None);
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

// =============================================================================
// try_cuda_full_forward  (public entry point)
// =============================================================================

/// Attempt to run the full inference forward pass (all N layers) on CUDA GPU.
///
/// Mirrors `try_metal_full_forward` for the CUDA backend.  All layer weights
/// are uploaded/cached on first call and reused on subsequent tokens.
///
/// Returns `None` on any error (callers fall back to the CPU path).
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_full_forward(
    hidden: &[f32],
    layer_params: &[CudaFullForwardLayerParams<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
    pos: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    hidden_size: usize,
    intermediate_size: usize,
    max_seq_len: usize,
    final_norm_bytes: Option<&[f32]>,
    final_norm_handle: u64,
) -> Option<Vec<f32>> {
    let _t0 = profiling().then(std::time::Instant::now);

    // Retrieve or build the cached model weights (O(1) Arc clones on warm path).
    let (graph, layer_weights) = get_or_build_model_weights(layer_params)?;

    let _t1 = profiling().then(std::time::Instant::now);
    if profiling() {
        eprintln!(
            "[cuda-prof] try_ff pos={pos}: weight_lookup={:.3}ms",
            (_t1.expect("profiling") - _t0.expect("profiling")).as_secs_f64() * 1000.0,
        );
    }

    let r = encode_full_forward(
        &graph,
        hidden,
        &layer_weights,
        rope_cos,
        rope_sin,
        pos,
        nq,
        nkv,
        head_dim,
        heads_per_group,
        norm_eps,
        hidden_size,
        intermediate_size,
        max_seq_len,
        final_norm_bytes,
        final_norm_handle,
    );
    if profiling() {
        let elapsed = _t1.expect("profiling").elapsed().as_secs_f64() * 1000.0;
        let path = if pos == 0 { "slow" } else { "fast" };
        eprintln!("[cuda-prof] encode_ff pos={pos} path={path}: {elapsed:.1}ms");
    }
    if let Err(ref e) = r {
        warn!("CUDA full-forward error at pos={pos}: {e}");
    }
    r.ok()
}

// =============================================================================
// try_cuda_full_forward_with_gpu_lm_head  (GPU LM-head path)
// =============================================================================

/// Run all transformer layers + final RMSNorm + LM-head GEMV entirely on GPU.
///
/// This eliminates the CPU LM-head GEMV which takes ~20ms for large vocabularies.
/// The LM-head weight is uploaded/cached on first call.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_full_forward_with_gpu_lm_head(
    hidden: &[f32],
    layer_params: &[CudaFullForwardLayerParams<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
    pos: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    hidden_size: usize,
    intermediate_size: usize,
    max_seq_len: usize,
    final_norm_bytes: Option<&[f32]>,
    final_norm_handle: u64,
    lm_head_handle: u64,
    lm_head_bytes: &[u8],
    vocab_size: usize,
) -> Option<Vec<f32>> {
    let normed = try_cuda_full_forward(
        hidden,
        layer_params,
        rope_cos,
        rope_sin,
        pos,
        nq,
        nkv,
        head_dim,
        heads_per_group,
        norm_eps,
        hidden_size,
        intermediate_size,
        max_seq_len,
        final_norm_bytes,
        final_norm_handle,
    )?;

    let graph = CudaGraph::global().ok()?;
    let _t_lm = profiling().then(std::time::Instant::now);
    let r = graph.encode_lm_head_gemv(
        &normed,
        lm_head_handle,
        lm_head_bytes,
        vocab_size,
        hidden_size,
    );
    if profiling() {
        eprintln!(
            "[cuda-prof] lm_head pos={pos}: {:.1}ms",
            _t_lm.expect("profiling").elapsed().as_secs_f64() * 1000.0
        );
    }
    r.ok()
}
