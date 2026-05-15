//! K-quant layer encoders for the CUDA prefill path.
//!
//! Provides:
//!  - [`encode_k_quant_ffn_phase`] — batched FFN sublayer (RMSNorm → format-specific
//!    fused gate+up+SwiGLU GEMM → format-specific down GEMM + residual add).
//!  - [`encode_k_quant_prefill_layer`] — full transformer layer with format
//!    dispatch via [`KQuantFormat`] for all linear projections.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};

use crate::gpu_backend::cuda_full_layer::{
    encode_attn_phase_from_qkv, CudaAttnModules, CudaFullLayerBuffers, CudaKvCache,
};
use crate::gpu_backend::cuda_graph::{CudaGraph, CudaGraphError};
use crate::gpu_backend::cuda_prefill::{CudaPrefillBuffers, CudaPrefillModules};

use super::launchers::{
    launch_fused_gate_up_swiglu_q2k, launch_fused_gate_up_swiglu_q3k,
    launch_fused_gate_up_swiglu_q4k, launch_fused_gate_up_swiglu_q5k,
    launch_fused_gate_up_swiglu_q6k, launch_fused_gate_up_swiglu_q8k, launch_gemm_q2k,
    launch_gemm_q3k, launch_gemm_q4k, launch_gemm_q5k, launch_gemm_q6k, launch_gemm_q8k,
};
use super::state::{CudaKQuantPrefillModules, KQuantFormat};

// =============================================================================
// encode_k_quant_ffn_phase
// =============================================================================

/// Batched FFN sublayer for K-quant models.
///
/// Pipeline:
/// 1. Batched RMSNorm: `d_input → d_normed` (all tokens)
/// 2. Fused gate+up+SwiGLU GEMM: `d_normed → d_swiglu` (format-specific)
/// 3. Down GEMM + residual: `d_swiglu → d_input` (format-specific)
///
/// # Safety
/// All device buffers must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn encode_k_quant_ffn_phase(
    graph: &CudaGraph,
    kq_mods: &CudaKQuantPrefillModules,
    pmods: &CudaPrefillModules,
    d_ffn_norm_weight: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    pb: &mut CudaPrefillBuffers,
    eps: f32,
    fmt: KQuantFormat,
) -> Result<(), CudaGraphError> {
    let bs = pb.actual_batch_size as u32;
    let h = pb.hidden_size as u32;
    let inter = pb.intermediate_size as u32;

    // Step 1: Batched RMSNorm → d_normed.
    {
        let cfg = LaunchConfig {
            grid_dim: (bs, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        graph
            .stream_arc()
            .launch_builder(&pmods.batched_rmsnorm)
            .arg(&pb.d_input)
            .arg(d_ffn_norm_weight)
            .arg(&mut pb.d_normed)
            .arg(&h)
            .arg(&bs)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("batched_rmsnorm ffn kquant: {e}")))?;
    }

    // Step 2: Fused gate+up+SwiGLU GEMM (d_normed → d_swiglu).
    // Zero d_gate_up buffer first (fused kernels write directly, not +=).
    {
        let n = 2 * pb.actual_batch_size * pb.intermediate_size;
        let mut dst_view = pb.d_gate_up.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_gate_up kquant: {e}")))?;
    }
    match fmt {
        KQuantFormat::Q2K => launch_fused_gate_up_swiglu_q2k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
        KQuantFormat::Q3K => launch_fused_gate_up_swiglu_q3k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
        KQuantFormat::Q4K => launch_fused_gate_up_swiglu_q4k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
        KQuantFormat::Q5K => launch_fused_gate_up_swiglu_q5k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
        KQuantFormat::Q6K => launch_fused_gate_up_swiglu_q6k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
        KQuantFormat::Q8K => launch_fused_gate_up_swiglu_q8k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
    }

    // Step 3: Down GEMM into d_normed (scratch), then residual add.
    // Zero d_normed first (kernels accumulate with +=).
    {
        let n = pb.actual_batch_size * pb.hidden_size;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_normed down kquant: {e}")))?;
    }
    match fmt {
        KQuantFormat::Q2K => launch_gemm_q2k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
        KQuantFormat::Q3K => launch_gemm_q3k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
        KQuantFormat::Q4K => launch_gemm_q4k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
        KQuantFormat::Q5K => launch_gemm_q5k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
        KQuantFormat::Q6K => launch_gemm_q6k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
        KQuantFormat::Q8K => launch_gemm_q8k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
    }

    let total_bh = (pb.actual_batch_size * pb.hidden_size) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_bh)?;

    Ok(())
}

// =============================================================================
// encode_k_quant_prefill_layer
// =============================================================================

/// Encode one full transformer layer for K-quant batch prefill.
///
/// Same 5-step structure as Phase 24 Q4_0/Q8_0 path, with format dispatch via
/// [`KQuantFormat`] for all linear projections.
///
/// # Safety
/// All device buffers and weight slices must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn encode_k_quant_prefill_layer(
    graph: &CudaGraph,
    kq_mods: &CudaKQuantPrefillModules,
    pmods: &CudaPrefillModules,
    attn_mods: &CudaAttnModules,
    d_attn_norm_weight: &CudaSlice<f32>,
    d_fused_qkv_weight: &Arc<CudaSlice<u8>>,
    d_q_norm_weight: &CudaSlice<f32>,
    d_k_norm_weight: &CudaSlice<f32>,
    d_attn_proj_weight: &Arc<CudaSlice<u8>>,
    d_ffn_norm_weight: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    kv: &mut CudaKvCache,
    layer_idx: usize,
    pos_start: usize,
    pb: &mut CudaPrefillBuffers,
    st_bufs: &mut CudaFullLayerBuffers,
    cos_table: &[f32],
    sin_table: &[f32],
    heads_per_group: usize,
    eps: f32,
    fmt: KQuantFormat,
) -> Result<(), CudaGraphError> {
    let bs = pb.actual_batch_size;
    let h = pb.hidden_size;
    let nq = pb.nq;
    let nkv = pb.nkv;
    let hd = pb.head_dim;
    let half_dim = hd / 2;
    let h_u32 = h as u32;
    let bs_u32 = bs as u32;
    let qkv_total = nq * hd + 2 * nkv * hd;

    // ─── 1. Batched RMSNorm (attn norm): d_input → d_normed ─────────────────
    {
        let cfg = LaunchConfig {
            grid_dim: (bs_u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        graph
            .stream_arc()
            .launch_builder(&pmods.batched_rmsnorm)
            .arg(&pb.d_input)
            .arg(d_attn_norm_weight)
            .arg(&mut pb.d_normed)
            .arg(&h_u32)
            .arg(&bs_u32)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| {
                CudaGraphError::DriverError(format!("batched_rmsnorm attn kquant: {e}"))
            })?;
    }

    // ─── 2. Batched QKV GEMM: d_normed → d_qkv ──────────────────────────────
    // Zero d_qkv first (kernels accumulate with +=).
    {
        let n = bs * qkv_total;
        let mut dst_view = pb.d_qkv.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_qkv kquant: {e}")))?;
    }
    match fmt {
        KQuantFormat::Q2K => launch_gemm_q2k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
        KQuantFormat::Q3K => launch_gemm_q3k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
        KQuantFormat::Q4K => launch_gemm_q4k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
        KQuantFormat::Q5K => launch_gemm_q5k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
        KQuantFormat::Q6K => launch_gemm_q6k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
        KQuantFormat::Q8K => launch_gemm_q8k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
    }

    // ─── 3. Sequential attention for each token ──────────────────────────────
    {
        let n = bs * nq * hd;
        let mut dst_view = pb.d_attn_out.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_attn_out kquant: {e}")))?;
    }

    for t in 0..bs {
        let pos = pos_start + t;

        // Copy this token's hidden state into st_bufs.d_hidden.
        {
            let src_view: CudaView<f32> = pb.d_input.slice(t * h..(t + 1) * h);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                .map_err(|e| {
                    CudaGraphError::DriverError(format!("copy hidden kquant t={t}: {e}"))
                })?;
        }

        // Copy this token's QKV into st_bufs.d_qkv.
        {
            let src_view: CudaView<f32> = pb.d_qkv.slice(t * qkv_total..(t + 1) * qkv_total);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_qkv)
                .map_err(|e| CudaGraphError::DriverError(format!("copy qkv kquant t={t}: {e}")))?;
        }

        // Upload RoPE cos/sin for this token's position.
        let rope_off = t * half_dim;
        graph
            .stream_arc()
            .memcpy_htod(
                &cos_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_cos,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload cos kquant t={t}: {e}")))?;
        graph
            .stream_arc()
            .memcpy_htod(
                &sin_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_sin,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload sin kquant t={t}: {e}")))?;

        // Upload pos and seq_len (pos+1) for this token.
        let pos_seqlen = [pos as u32, (pos + 1) as u32];
        graph
            .stream_arc()
            .memcpy_htod(&pos_seqlen, &mut st_bufs.d_pos_seqlen)
            .map_err(|e| {
                CudaGraphError::DriverError(format!("upload pos_seqlen kquant t={t}: {e}"))
            })?;

        // Run attention steps 3-7.
        encode_attn_phase_from_qkv(
            graph,
            attn_mods,
            d_q_norm_weight,
            d_k_norm_weight,
            kv,
            layer_idx,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            st_bufs,
        )?;

        // Copy attention output back into pb.d_attn_out for this token.
        {
            let attn_col_size = nq * hd;
            let src_view: CudaView<f32> = st_bufs.d_attn_out.slice(0..attn_col_size);
            let mut dst_view = pb
                .d_attn_out
                .slice_mut(t * attn_col_size..(t + 1) * attn_col_size);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut dst_view)
                .map_err(|e| {
                    CudaGraphError::DriverError(format!("copy attn_out kquant t={t}: {e}"))
                })?;
        }
    }

    // ─── 4. Attn output projection + residual ────────────────────────────────
    {
        let n = bs * h;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| {
                CudaGraphError::DriverError(format!("zero d_normed attn_proj kquant: {e}"))
            })?;
    }
    let attn_proj_rows = h_u32;
    let attn_proj_k = (nq * hd) as u32;
    match fmt {
        KQuantFormat::Q2K => launch_gemm_q2k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
        KQuantFormat::Q3K => launch_gemm_q3k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
        KQuantFormat::Q4K => launch_gemm_q4k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
        KQuantFormat::Q5K => launch_gemm_q5k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
        KQuantFormat::Q6K => launch_gemm_q6k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
        KQuantFormat::Q8K => launch_gemm_q8k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
    }

    let total_bh = (bs * h) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_bh)?;

    // ─── 5. Batched FFN sublayer ──────────────────────────────────────────────
    encode_k_quant_ffn_phase(
        graph,
        kq_mods,
        pmods,
        d_ffn_norm_weight,
        d_gate_up_weight,
        d_down_weight,
        pb,
        eps,
        fmt,
    )?;

    Ok(())
}
