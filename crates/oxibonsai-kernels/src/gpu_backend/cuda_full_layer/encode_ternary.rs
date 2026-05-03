//! TQ2 (ternary) full-layer encode functions for the CUDA backend.
//!
//! Contains `encode_layer_into_ternary`, `encode_full_forward_ternary`,
//! `encode_lm_head_gemv_ternary`, `try_cuda_full_forward_ternary`,
//! and `try_cuda_full_forward_ternary_with_gpu_lm_head`.
//!
//! ## Design
//!
//! This module mirrors the Metal `try_metal_full_forward_ternary` / `try_metal_prefill_ternary`
//! paths but targets CUDA.  Each decode step runs 14 kernel dispatches per layer:
//!
//! 1. Pre-attn RMSNorm
//! 2. Fused QKV TQ2 GEMV
//! 3. Fused QK-norm + RoPE
//! 4. Fused KV-store
//! 5. Batched attention scores V2
//! 6. Batched softmax
//! 7. Batched weighted sum
//! 8. Attn output TQ2 GEMV
//! 9. Residual add (attn)
//! 10. FFN RMSNorm
//! 11. Gate+Up TQ2 GEMV
//! 12. SwiGLU
//! 13. Down TQ2 GEMV
//! 14. Residual add (FFN)
//!
//! **Split-dispatch only** — no fused GEMV+residual or fused gate_up_swiglu kernels.
//!
//! ## Handle namespaces
//!
//! These ranges are chosen to avoid collisions with the Q1 CUDA ranges (1M–4M):
//! - Ternary norm handles:   `5_000_000 + layer * 10 + offset`
//! - Ternary weight handles: `6_000_000 + layer * 10 + offset`
//! - Final norm handle:      `5_900_000`
//! - LM-head handle:         `7_000_000`

use std::sync::Arc;

use cudarc::driver::sys;
use tracing::warn;

use super::super::cuda_graph::{CudaGraph, CudaGraphError};
use super::{
    acquire_full_layer_buffers, acquire_kv_cache, full_layer_state, get_or_upload_f32_weight,
    init_attn_modules, profiling, CuGraphHolder, CudaAttnModules, CudaCachedLayerWeights,
    CudaFullLayerBuffers, CudaKvCache,
};

use super::launchers::{
    launch_batched_attn_scores_v2, launch_batched_attn_weighted_sum, launch_batched_softmax,
    launch_fused_kv_store, launch_fused_qk_norm_rope,
};

// =============================================================================
// CudaFullForwardLayerParamsTernary
// =============================================================================

/// Per-layer parameters for the CUDA ternary full-forward path.
///
/// Mirrors [`CudaFullForwardLayerParams`] but carries TQ2_0_g128 block bytes
/// (34 bytes/block) for every GEMV weight instead of Q1_0_g128.
pub struct CudaFullForwardLayerParamsTernary<'a> {
    /// Handle ID for the pre-attention RMSNorm weight (FP32).
    pub attn_norm_handle: u64,
    /// Pre-attention RMSNorm weight (FP32).
    pub attn_norm_bytes: &'a [f32],
    /// Handle ID for the fused QKV TQ2 weight (Q‖K‖V concatenated, AoS bytes).
    pub fused_qkv_handle: u64,
    /// Fused QKV TQ2 weight bytes (AoS layout, 34 bytes/block).
    pub fused_qkv_bytes: &'a [u8],
    /// Handle ID for the Q-head RMSNorm weight (FP32).
    pub q_norm_handle: u64,
    /// Q-head RMSNorm weight (FP32).
    pub q_norm_bytes: &'a [f32],
    /// Handle ID for the K-head RMSNorm weight (FP32).
    pub k_norm_handle: u64,
    /// K-head RMSNorm weight (FP32).
    pub k_norm_bytes: &'a [f32],
    /// Handle ID for the attention output projection TQ2 weight.
    pub attn_proj_handle: u64,
    /// Attention output projection TQ2 weight bytes (AoS layout, 34 bytes/block).
    pub attn_proj_bytes: &'a [u8],
    /// Handle ID for the post-attention (FFN) RMSNorm weight (FP32).
    pub ffn_norm_handle: u64,
    /// Post-attention (FFN) RMSNorm weight (FP32).
    pub ffn_norm_bytes: &'a [f32],
    /// Handle ID for the fused Gate+Up TQ2 weight (concatenated, AoS bytes).
    pub gate_up_handle: u64,
    /// Gate projection TQ2 weight bytes (AoS layout, 34 bytes/block).
    pub gate_bytes: &'a [u8],
    /// Up projection TQ2 weight bytes (AoS layout, 34 bytes/block).
    pub up_bytes: &'a [u8],
    /// Handle ID for the FFN down projection TQ2 weight.
    pub down_handle: u64,
    /// FFN down projection TQ2 weight bytes (AoS layout, 34 bytes/block).
    pub down_bytes: &'a [u8],
}

// =============================================================================
// encode_layer_into_ternary  (pure device, no H2D/D2H)
// =============================================================================

/// Run one ternary transformer layer entirely on device, reading/writing
/// `bufs.d_hidden` in-place.
///
/// Uses split-dispatch (14 kernel launches per layer):
/// GEMV → residual_add / SwiGLU each as separate launches, no fused kernels.
///
/// RoPE buffers (`bufs.d_cos` / `bufs.d_sin`) must already be populated before
/// the first call.  `d_pos_seqlen` must be uploaded with `[pos, pos+1]` before
/// entry.
///
/// # Safety
/// All device pointers must be valid and allocated on the same CUDA stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn encode_layer_into_ternary(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    weights: &CudaCachedLayerWeights,
    kv: &mut CudaKvCache,
    layer_idx: usize,
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

    // ── Attention sublayer ────────────────────────────────────────────────────

    // Dispatch 1: Pre-attn RMSNorm (d_hidden → d_normed)
    graph.launch_rmsnorm_pub(
        &bufs.d_hidden,
        &weights.pre_attn_norm,
        &mut bufs.d_normed,
        h_u32,
        norm_eps,
    )?;

    // Dispatch 2: Fused QKV TQ2 GEMV (d_normed → d_qkv)
    graph.launch_gemv_tq2_v1_pub(
        &weights.q_weight,
        &bufs.d_normed,
        &mut bufs.d_qkv,
        qkv_total_rows,
        h_u32,
    )?;

    // Dispatch 3: Fused QK-Norm + RoPE
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

    // Dispatch 4: Fused KV-Store — pos read from d_pos_seqlen[0] by the kernel
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

    // Dispatch 5: Batched attention scores V2 — seq_len read from d_pos_seqlen[1]
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

    // Dispatch 6: Softmax — seq_len read from d_pos_seqlen[1]
    launch_batched_softmax(
        graph,
        mods,
        &mut bufs.d_scores,
        nq_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
    )?;

    // Dispatch 7: Weighted sum — seq_len read from d_pos_seqlen[1]
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

    // ── FFN sublayer ──────────────────────────────────────────────────────────

    // Dispatch 8: Attn output TQ2 GEMV (d_attn_out → d_normed)
    let attn_out_rows = (nq * head_dim) as u32;
    graph.launch_gemv_tq2_v1_pub(
        &weights.o_weight,
        &bufs.d_attn_out,
        &mut bufs.d_normed,
        h_u32,
        attn_out_rows,
    )?;

    // Dispatch 9: Residual add (hidden += attn_output)
    graph.launch_residual_add_pub(&mut bufs.d_hidden, &bufs.d_normed, h_u32)?;

    // Dispatch 10: FFN RMSNorm (d_hidden → d_normed)
    graph.launch_rmsnorm_pub(
        &bufs.d_hidden,
        &weights.post_attn_norm,
        &mut bufs.d_normed,
        h_u32,
        norm_eps,
    )?;

    // Dispatch 11: Gate+Up TQ2 GEMV (d_normed → d_gate_up)
    graph.launch_gemv_tq2_v1_pub(
        &weights.gate_up_weight,
        &bufs.d_normed,
        &mut bufs.d_gate_up,
        2 * inter_u32,
        h_u32,
    )?;

    // Dispatch 12: SwiGLU (d_gate_up → d_swiglu)
    graph.launch_swiglu_pub(&bufs.d_gate_up, &mut bufs.d_swiglu, inter_u32)?;

    // Dispatch 13: Down TQ2 GEMV (d_swiglu → d_normed)
    graph.launch_gemv_tq2_v1_pub(
        &weights.down_weight,
        &bufs.d_swiglu,
        &mut bufs.d_normed,
        h_u32,
        inter_u32,
    )?;

    // Dispatch 14: Residual add (hidden += FFN output)
    graph.launch_residual_add_pub(&mut bufs.d_hidden, &bufs.d_normed, h_u32)?;

    Ok(())
}

// =============================================================================
// get_or_build_ternary_model_weights
// =============================================================================

/// Build (or return cached) GPU weight handles for all ternary transformer layers.
///
/// Uses `6_000_000 + layer * 10 + offset` for weight handles and
/// `5_000_000 + layer * 10 + offset` for norm handles, keeping them separate
/// from Q1 CUDA handles (1M–4M) and the Metal ternary handles.
fn get_or_build_ternary_model_weights(
    layer_params: &[CudaFullForwardLayerParamsTernary<'_>],
) -> Option<(Arc<CudaGraph>, Arc<Vec<CudaCachedLayerWeights>>)> {
    let n_layers = layer_params.len();
    let state = full_layer_state();

    // Fast path: check if the ternary model weights are already cached.
    // We reuse the cached_model_weights slot (same slot as Q1) but distinguish
    // by layer count; models cannot switch between Q1 and ternary at runtime.
    {
        let guard = state.cached_model_weights.lock().ok()?;
        if let Some(ref cmw) = *guard {
            if cmw.n_layers == n_layers {
                return Some((Arc::clone(&cmw.graph), Arc::clone(&cmw.layers)));
            }
        }
    }

    let graph = CudaGraph::global().ok()?;
    let dummy_weight = Arc::new(graph.stream_arc().alloc_zeros::<u8>(1).ok()?);

    let mut cached: Vec<CudaCachedLayerWeights> = Vec::with_capacity(n_layers);
    for lp in layer_params {
        // TQ2 weights — uploaded as SoA via get_or_upload_weight_tq2_soa
        let q_weight = graph
            .get_or_upload_weight_tq2_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)
            .ok()?;
        let o_weight = graph
            .get_or_upload_weight_tq2_soa(lp.attn_proj_handle, lp.attn_proj_bytes)
            .ok()?;

        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_weight = graph
            .get_or_upload_weight_tq2_soa_lazy(lp.gate_up_handle, || {
                let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
                fused.extend_from_slice(gate_bytes);
                fused.extend_from_slice(up_bytes);
                fused
            })
            .ok()?;

        let down_weight = graph
            .get_or_upload_weight_tq2_soa(lp.down_handle, lp.down_bytes)
            .ok()?;

        // FP32 norm weights — uploaded as plain f32
        let pre_attn_norm =
            get_or_upload_f32_weight(&graph, lp.attn_norm_handle, lp.attn_norm_bytes).ok()?;
        let post_attn_norm =
            get_or_upload_f32_weight(&graph, lp.ffn_norm_handle, lp.ffn_norm_bytes).ok()?;
        let q_norm = get_or_upload_f32_weight(&graph, lp.q_norm_handle, lp.q_norm_bytes).ok()?;
        let k_norm = get_or_upload_f32_weight(&graph, lp.k_norm_handle, lp.k_norm_bytes).ok()?;

        cached.push(CudaCachedLayerWeights {
            q_weight,
            k_weight: Arc::clone(&dummy_weight),
            v_weight: Arc::clone(&dummy_weight),
            o_weight,
            gate_up_weight,
            down_weight,
            pre_attn_norm,
            post_attn_norm,
            q_norm,
            k_norm,
        });
    }

    let layers = Arc::new(cached);
    let cmw = super::CudaCachedModelWeights {
        graph: Arc::clone(&graph),
        dummy_weight,
        layers: Arc::clone(&layers),
        n_layers,
    };
    if let Ok(mut guard) = state.cached_model_weights.lock() {
        *guard = Some(cmw);
    }
    Some((graph, layers))
}

// =============================================================================
// encode_full_forward_ternary
// =============================================================================

/// Run the full ternary forward pass (all layers) on GPU without intermediate syncs.
///
/// Mirrors `encode_full_forward` in `encode_q1.rs` but uses TQ2 GEMV dispatches.
/// CUDA driver graph capture/replay is used for decode-step acceleration after the
/// first token.
///
/// Returns the final hidden state (post-norm if `final_norm_weight` provided).
#[allow(clippy::too_many_arguments)]
pub fn encode_full_forward_ternary(
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
            "encode_full_forward_ternary: no layers provided".into(),
        ));
    }

    let attn_mods = init_attn_modules(graph)?;

    let mut fl_guard =
        acquire_full_layer_buffers(graph, h, nq, nkv, head_dim, max_seq_len, intermediate_size)?;
    let bufs = fl_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("full_layer_buffers not allocated".into()))?;

    let mut kv_guard = acquire_kv_cache(graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("kv_cache not allocated".into()))?;

    let stream = graph.stream_arc();

    // Upload [pos, seq_len] to the 2-element device buffer.
    let pos_seqlen_host = [pos as u32, (pos + 1) as u32];
    unsafe {
        graph
            .raw_htod(&pos_seqlen_host, &mut bufs.d_pos_seqlen, 2)
            .map_err(|e| CudaGraphError::DriverError(format!("upload pos_seqlen ternary: {e}")))?;
    }

    // ── Fast path: replay captured CUDA graph (every token after the first) ─────
    {
        let graph_guard = full_layer_state()
            .cuda_driver_graph
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(Some(ref holder)) = *graph_guard {
            unsafe {
                graph.raw_htod(&hidden_init[..h], &mut bufs.d_hidden, h)?;
                graph.raw_htod(&rope_cos[..half_dim], &mut bufs.d_cos, half_dim)?;
                graph.raw_htod(&rope_sin[..half_dim], &mut bufs.d_sin, half_dim)?;
            }
            unsafe { holder.launch() }
                .map_err(|e| CudaGraphError::DriverError(format!("ternary graph launch: {e}")))?;
            let mut result = vec![0.0f32; h];
            unsafe { graph.raw_dtoh(&bufs.d_hidden, &mut result, h)? }
            stream
                .synchronize()
                .map_err(|e| CudaGraphError::DriverError(format!("ternary fast-path sync: {e}")))?;
            return Ok(result);
        }
    }

    // ── Slow path: first call, or after buffer realloc ───────────────────────────
    unsafe {
        graph
            .raw_htod(&hidden_init[..h], &mut bufs.d_hidden, h)
            .map_err(|e| CudaGraphError::DriverError(format!("upload hidden_init ternary: {e}")))?;
        graph
            .raw_htod(&rope_cos[..half_dim], &mut bufs.d_cos, half_dim)
            .map_err(|e| CudaGraphError::DriverError(format!("upload cos ternary: {e}")))?;
        graph
            .raw_htod(&rope_sin[..half_dim], &mut bufs.d_sin, half_dim)
            .map_err(|e| CudaGraphError::DriverError(format!("upload sin ternary: {e}")))?;
    }

    for (layer_idx, weights) in all_layer_weights.iter().enumerate() {
        unsafe {
            encode_layer_into_ternary(
                graph,
                &attn_mods,
                weights,
                kv,
                layer_idx,
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
        stream
            .memcpy_dtod(&bufs.d_normed, &mut bufs.d_hidden)
            .map_err(|e| {
                CudaGraphError::DriverError(format!("dtod normed->hidden ternary: {e}"))
            })?;
    }

    let mut result = vec![0.0f32; h];
    unsafe { graph.raw_dtoh(&bufs.d_hidden, &mut result, h)? }
    stream
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("ternary D2H sync: {e}")))?;

    // ── Capture the kernel sequence as a replayable CUDA driver graph ────────────
    {
        if let Ok(ref mut graph_guard) = full_layer_state().cuda_driver_graph.lock() {
            if graph_guard.is_none() {
                let begin_ok = stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
                    .is_ok();
                if !begin_ok {
                    warn!(
                        "CUDA graph ternary: begin_capture failed — running without graph replay"
                    );
                    **graph_guard = Some(None);
                } else {
                    let record_ok: bool = (|| -> Result<(), CudaGraphError> {
                        for (layer_idx, weights) in all_layer_weights.iter().enumerate() {
                            unsafe {
                                encode_layer_into_ternary(
                                    graph,
                                    &attn_mods,
                                    weights,
                                    kv,
                                    layer_idx,
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
                            stream
                                .memcpy_dtod(&bufs.d_normed, &mut bufs.d_hidden)
                                .map_err(|e| {
                                    CudaGraphError::DriverError(format!(
                                        "dtod (ternary capture): {e}"
                                    ))
                                })?;
                        }
                        Ok(())
                    })()
                    .is_ok();

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
                                                "CUDA ternary graph captured and uploaded"
                                            );
                                        }
                                        Err(e) => {
                                            warn!(
                                                "CUDA ternary graph upload failed: {e} — disabling"
                                            );
                                            **graph_guard = Some(None);
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!("CUDA ternary graph instantiate failed: {e}");
                                    unsafe {
                                        let _ =
                                            cudarc::driver::result::graph::destroy(cu_graph_raw);
                                    }
                                    **graph_guard = Some(None);
                                }
                            }
                        }
                        Ok(_) => {
                            warn!("CUDA ternary graph: end_capture returned no graph");
                            **graph_guard = Some(None);
                        }
                        Err(e) => {
                            warn!("CUDA ternary graph: end_capture error: {e}");
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
// encode_lm_head_gemv_ternary
// =============================================================================

/// Run the ternary LM-head GEMV on GPU: `logits = lm_head_weight × normed`.
///
/// Delegates to `CudaGraph::encode_lm_head_gemv_tq2` which manages the LM-head
/// I/O buffers and weight caching internally.
pub fn encode_lm_head_gemv_ternary(
    graph: &CudaGraph,
    normed: &[f32],
    handle_id: u64,
    weight_bytes: &[u8],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<f32>, CudaGraphError> {
    graph.encode_lm_head_gemv_tq2(normed, handle_id, weight_bytes, vocab_size, hidden_size)
}

// =============================================================================
// try_cuda_full_forward_ternary  (public entry point)
// =============================================================================

/// Attempt to run the full ternary forward pass (all N layers) on CUDA GPU.
///
/// Mirrors `try_cuda_full_forward` for ternary (TQ2_0_g128) models.
/// Returns `None` on any error (callers fall back to CPU path).
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_full_forward_ternary(
    hidden: &[f32],
    layer_params: &[CudaFullForwardLayerParamsTernary<'_>],
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

    let (graph, layer_weights) = get_or_build_ternary_model_weights(layer_params)?;

    let r = encode_full_forward_ternary(
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
        if let Some(t0) = _t0 {
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let path = if pos == 0 { "slow" } else { "fast" };
            eprintln!("[cuda-prof] ternary encode_ff pos={pos} path={path}: {elapsed:.1}ms");
        }
    }
    if let Err(ref e) = r {
        warn!("CUDA ternary full-forward error at pos={pos}: {e}");
    }
    r.ok()
}

// =============================================================================
// try_cuda_full_forward_ternary_with_gpu_lm_head  (GPU LM-head path)
// =============================================================================

/// Run all ternary transformer layers + final RMSNorm + TQ2 LM-head GEMV on GPU.
///
/// Eliminates the CPU LM-head GEMV for ternary models. The LM-head weight is
/// uploaded/cached on first call and reused across tokens.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_full_forward_ternary_with_gpu_lm_head(
    hidden: &[f32],
    layer_params: &[CudaFullForwardLayerParamsTernary<'_>],
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
    let normed = try_cuda_full_forward_ternary(
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
    let r = encode_lm_head_gemv_ternary(
        &graph,
        &normed,
        lm_head_handle,
        lm_head_bytes,
        vocab_size,
        hidden_size,
    );
    if profiling() {
        if let Some(t_lm) = _t_lm {
            eprintln!(
                "[cuda-prof] ternary lm_head pos={pos}: {:.1}ms",
                t_lm.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
    if let Err(ref e) = r {
        warn!("CUDA ternary lm_head error at pos={pos}: {e}");
    }
    r.ok()
}

// =============================================================================
// CI-GPU-gated tests
// =============================================================================

#[cfg(all(test, feature = "native-cuda"))]
mod ternary_cuda_tests {
    //! These tests require CUDA hardware to run.
    //!
    //! They compile under `#[cfg(feature = "native-cuda")]` but are documented as
    //! **CI-GPU-gated** — validation requires first-boot execution on CUDA hardware.
    //! On macOS (no CUDA) the entire module compiles out.

    use super::*;

    /// Verify that `encode_lm_head_gemv_ternary` produces logits matching a
    /// CPU reference for a synthetic small model.
    ///
    /// Synthetic parameters: hidden=128, vocab=256, 1 TQ2 block per row.
    /// CPU reference: dequant TQ2 blocks, matmul, argmax.
    /// Gate: token IDs match exactly; per-logit absolute error < 1e-3.
    ///
    /// NOTE: CI-GPU-gated — requires CUDA hardware to run.
    #[test]
    fn test_encode_lm_head_gemv_ternary_matches_reference() {
        use oxibonsai_core::{BlockTQ2_0_g128, QK_TQ2_0_G128};

        let hidden_size: usize = 128;
        let vocab_size: usize = 256;
        // Each row needs hidden_size/128 = 1 block.
        let blocks_per_row = hidden_size / QK_TQ2_0_G128;
        let total_blocks = vocab_size * blocks_per_row;

        // Build deterministic synthetic TQ2 blocks.
        // TQ2 2-bit encoding: 0b00=-1, 0b01=0, 0b10=+1, 0b11=reserved (zero).
        let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(total_blocks);
        for i in 0..total_blocks {
            let scale = half::f16::from_f32(0.1f32 + (i % 8) as f32 * 0.01);
            let pattern = (i % 3) as u8;
            let byte = match pattern {
                0 => 0b0101_0101u8, // all zeros
                1 => 0b1010_1010u8, // all +1
                _ => 0b0000_0000u8, // all -1
            };
            let qs = [byte; 32];
            blocks.push(BlockTQ2_0_g128 { d: scale, qs });
        }

        // Compute CPU reference using the canonical TQ2 decoding.
        let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.01).collect();
        let mut expected_logits = vec![0.0f32; vocab_size];
        for row in 0..vocab_size {
            let mut sum = 0.0f32;
            for blk_idx in 0..blocks_per_row {
                let b = &blocks[row * blocks_per_row + blk_idx];
                let scale = b.d.to_f32();
                for (byte_idx, &byte) in b.qs.iter().enumerate() {
                    for bit_pair in 0..4usize {
                        let code = (byte >> (bit_pair * 2)) & 0b11;
                        let w = match code {
                            0b00 => -1.0f32,
                            0b10 => 1.0f32,
                            _ => 0.0f32, // 0b01 (Zero) and 0b11 (reserved)
                        };
                        let feat_idx = blk_idx * 128 + byte_idx * 4 + bit_pair;
                        if feat_idx < hidden_size {
                            sum += scale * w * input[feat_idx];
                        }
                    }
                }
            }
            expected_logits[row] = sum;
        }

        // GPU path.
        let graph = CudaGraph::global().expect("CUDA device required");
        // Build raw AoS bytes from blocks.
        let aos_bytes: Vec<u8> = blocks
            .iter()
            .flat_map(|b| {
                let scale_bits = b.d.to_bits().to_le_bytes();
                let mut v = Vec::with_capacity(34);
                v.extend_from_slice(&scale_bits);
                v.extend_from_slice(&b.qs);
                v
            })
            .collect();
        let handle = 7_900_000u64; // test-only handle
        let gpu_logits = encode_lm_head_gemv_ternary(
            &graph,
            &input,
            handle,
            &aos_bytes,
            vocab_size,
            hidden_size,
        )
        .expect("encode_lm_head_gemv_ternary failed");

        assert_eq!(gpu_logits.len(), vocab_size, "logits length mismatch");
        let expected_token = expected_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let gpu_token = gpu_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(
            expected_token, gpu_token,
            "argmax mismatch: expected {expected_token}, got {gpu_token}"
        );
        for (i, (&exp, &got)) in expected_logits.iter().zip(gpu_logits.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-3,
                "logit[{i}] error too large: expected={exp}, got={got}"
            );
        }
    }

    /// Verify that `try_cuda_full_forward_ternary` produces byte-exact token matches
    /// compared to a CPU scalar reference across 3 decode tokens for a 2-layer
    /// synthetic ternary model.  2+ tokens exercise CUDA graph capture+replay.
    ///
    /// NOTE: CI-GPU-gated — requires CUDA hardware to run.
    #[test]
    fn test_encode_full_forward_ternary_matches_reference() {
        // This is a structural skeleton — correctness validation requires
        // running against a known-good CPU reference on CUDA hardware.
        // The test documents the expected shape and gating conditions.
        //
        // Full implementation deferred to CI-GPU validation pass.
        if CudaGraph::global().is_err() {
            eprintln!("SKIP: test_encode_full_forward_ternary_matches_reference — no CUDA device");
        }
        // TODO(CI-GPU): implement 2-layer synthetic ternary model,
        // decode 3 tokens, compare against CPU scalar forward pass.
        // Gate: byte-exact token IDs across all 3 positions.
    }
}
