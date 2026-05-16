//! TQ2 (ternary) prefill layer encoders.
//!
//! Provides:
//!  - [`encode_prefill_ffn_phase_ternary`] — batched FFN sublayer for TQ2
//!    weights (RMSNorm → TQ2 fused gate+up+SwiGLU GEMM → TQ2 down GEMM +
//!    residual add).
//!  - [`encode_prefill_layer_ternary`] — full transformer layer for the TQ2
//!    prefill path with batched non-attention ops and sequential per-token
//!    `encode_attn_phase_tq2` (TQ2-aware single-token attention).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaView};

use crate::gpu_backend::cuda_full_layer::{
    encode_attn_phase_tq2, CudaAttnModules, CudaFullLayerBuffers, CudaKvCache,
};
use crate::gpu_backend::cuda_graph::{CudaGraph, CudaGraphError};

use super::launchers::{
    launch_batched_rmsnorm, launch_fused_gate_up_swiglu_gemm_tq2, launch_gemm_tq2_v7,
};
use super::state::{CudaPrefillBuffers, CudaPrefillModules};

// =============================================================================
// encode_prefill_ffn_phase_ternary
// =============================================================================

/// Batched FFN sublayer for TQ2 models: RMSNorm → TQ2 fused gate+up+SwiGLU → TQ2 down + residual.
///
/// # Safety
/// All device buffers must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn encode_prefill_ffn_phase_ternary(
    graph: &CudaGraph,
    pmods: &CudaPrefillModules,
    d_ffn_norm_weight: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    pb: &mut CudaPrefillBuffers,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let bs = pb.actual_batch_size as u32;
    let h = pb.hidden_size as u32;
    let inter = pb.intermediate_size as u32;

    // Step 1: Batched RMSNorm (all tokens)
    launch_batched_rmsnorm(
        graph,
        pmods,
        &pb.d_input,
        d_ffn_norm_weight,
        &mut pb.d_normed,
        h,
        bs,
        eps,
    )?;

    // Step 2: Fused TQ2 gate+up+SwiGLU GEMM (all tokens)
    //   d_normed [bs × h, col-major] → d_swiglu [bs × inter, col-major]
    launch_fused_gate_up_swiglu_gemm_tq2(
        graph,
        pmods,
        d_gate_up_weight,
        &pb.d_normed,
        &mut pb.d_swiglu,
        inter,
        h,
        bs,
    )?;

    // Step 3: TQ2 Down GEMM into d_normed (scratch), then in-place residual add.
    {
        let n = pb.actual_batch_size * pb.hidden_size;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_normed tq2 down: {e}")))?;
    }
    launch_gemm_tq2_v7(
        graph,
        pmods,
        d_down_weight,
        &pb.d_swiglu,
        &mut pb.d_normed,
        h,
        inter,
        bs,
    )?;

    let total_bh = (pb.actual_batch_size * pb.hidden_size) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_bh)?;

    Ok(())
}

// =============================================================================
// encode_prefill_layer_ternary
// =============================================================================

/// Encode one full transformer layer for batch prefill using TQ2 (ternary) weights.
///
/// Non-attention batch operations use TQ2 GEMM kernels.  Attention is processed
/// sequentially per token using `encode_attn_phase_tq2` (TQ2-aware single-token
/// attention that runs its own RMSNorm + TQ2 QKV GEMV).
///
/// On entry / exit, `pb.d_input` holds the batched residual stream
/// `[batch_size × hidden_size]` in column-major layout.
///
/// # Safety
/// All device buffers and weight slices must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn encode_prefill_layer_ternary(
    graph: &CudaGraph,
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

    // ════════════════════════════════════════════════════════════════════
    // 1. Batched RMSNorm (attn norm): d_input → d_normed
    // ════════════════════════════════════════════════════════════════════
    launch_batched_rmsnorm(
        graph,
        pmods,
        &pb.d_input,
        d_attn_norm_weight,
        &mut pb.d_normed,
        h_u32,
        bs_u32,
        eps,
    )?;

    // ════════════════════════════════════════════════════════════════════
    // 2. Batched TQ2 QKV GEMM: d_normed → d_qkv (all tokens at once)
    //    Zero-init d_qkv first so accumulate (+=) is correct.
    // ════════════════════════════════════════════════════════════════════
    {
        let n = bs * qkv_total;
        let mut dst_view = pb.d_qkv.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_qkv tq2: {e}")))?;
    }
    launch_gemm_tq2_v7(
        graph,
        pmods,
        d_fused_qkv_weight,
        &pb.d_normed,
        &mut pb.d_qkv,
        qkv_total as u32,
        h_u32,
        bs_u32,
    )?;

    // ════════════════════════════════════════════════════════════════════
    // 3. Sequential attention for each token (TQ2-aware)
    //
    // For each token t at sequence position (pos_start + t):
    //   a) Copy this token's hidden state into st_bufs.d_hidden
    //   b) Upload pos/seqlen to st_bufs.d_pos_seqlen
    //   c) Upload RoPE cos/sin for this token's position
    //   d) Call encode_attn_phase_tq2: runs rmsnorm + TQ2 QKV GEMV +
    //      qk-norm+rope + kv-store + scores + softmax + weighted sum
    //   e) Copy attention output for this token → pb.d_attn_out column t
    // ════════════════════════════════════════════════════════════════════
    {
        let n = bs * nq * hd;
        let mut dst_view = pb.d_attn_out.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_attn_out tq2: {e}")))?;
    }

    for t in 0..bs {
        let pos = pos_start + t;

        // Copy token t's hidden state column into st_bufs.d_hidden
        {
            let src_view: CudaView<f32> = pb.d_input.slice(t * h..(t + 1) * h);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                .map_err(|e| CudaGraphError::DriverError(format!("copy hidden tq2 t={t}: {e}")))?;
        }

        // Upload pos/seqlen [pos, pos+1] into st_bufs.d_pos_seqlen
        let pos_seqlen = [pos as u32, (pos + 1) as u32];
        graph
            .stream_arc()
            .memcpy_htod(&pos_seqlen, &mut st_bufs.d_pos_seqlen)
            .map_err(|e| {
                CudaGraphError::DriverError(format!("upload pos_seqlen tq2 t={t}: {e}"))
            })?;

        // Upload RoPE cos/sin for this token's position.
        let rope_off = t * half_dim;
        graph
            .stream_arc()
            .memcpy_htod(
                &cos_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_cos,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload cos tq2 t={t}: {e}")))?;
        graph
            .stream_arc()
            .memcpy_htod(
                &sin_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_sin,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload sin tq2 t={t}: {e}")))?;

        // Run TQ2-aware single-token attention pipeline (RMSNorm + TQ2 GEMV + attention).
        encode_attn_phase_tq2(
            graph,
            attn_mods,
            d_attn_norm_weight,
            d_fused_qkv_weight,
            d_q_norm_weight,
            d_k_norm_weight,
            kv,
            layer_idx,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            h,
            st_bufs,
        )?;

        // Copy attention output for this token from st_bufs.d_attn_out into
        // the column of pb.d_attn_out [t * nq*hd .. (t+1)*nq*hd]
        {
            let src_view: CudaView<f32> = st_bufs.d_attn_out.slice(..nq * hd);
            let mut dst_view = pb.d_attn_out.slice_mut(t * nq * hd..(t + 1) * nq * hd);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut dst_view)
                .map_err(|e| {
                    CudaGraphError::DriverError(format!("copy attn_out tq2 t={t}: {e}"))
                })?;
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // 4. TQ2 Output projection GEMM + residual (all tokens at once)
    //    attn_out_proj: [h × nq*hd], maps d_attn_out → d_normed (scratch)
    //    then: d_input += d_normed  (residual add)
    // ════════════════════════════════════════════════════════════════════
    {
        let n = bs * h;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_normed tq2 oproj: {e}")))?;
    }
    launch_gemm_tq2_v7(
        graph,
        pmods,
        d_attn_proj_weight,
        &pb.d_attn_out,
        &mut pb.d_normed,
        h_u32,
        (nq * hd) as u32,
        bs_u32,
    )?;
    let total_oproj = (bs * h) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_oproj)?;

    // ════════════════════════════════════════════════════════════════════
    // 5. Batched TQ2 FFN
    // ════════════════════════════════════════════════════════════════════
    encode_prefill_ffn_phase_ternary(
        graph,
        pmods,
        d_ffn_norm_weight,
        d_gate_up_weight,
        d_down_weight,
        pb,
        eps,
    )?;

    Ok(())
}
