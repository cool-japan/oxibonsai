//! Q1 (1-bit) prefill layer encoders.
//!
//! Provides:
//!  - [`encode_prefill_ffn_phase`] — batched FFN sublayer (RMSNorm → fused
//!    gate+up+SwiGLU GEMM → down GEMM + residual add).
//!  - [`encode_prefill_layer`] — full transformer layer for the Q1 prefill
//!    path, with batched non-attention ops and sequential per-token attention.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaView};

use crate::gpu_backend::cuda_full_layer::{
    encode_attn_phase, CudaAttnModules, CudaFullLayerBuffers, CudaKvCache,
};
use crate::gpu_backend::cuda_graph::{CudaGraph, CudaGraphError};

use super::launchers::{launch_batched_rmsnorm, launch_fused_gate_up_swiglu_gemm, launch_gemm_v7};
use super::state::{CudaPrefillBuffers, CudaPrefillModules};

// =============================================================================
// encode_prefill_ffn_phase
// =============================================================================

/// Encode the batched FFN sublayer for all `batch_size` tokens.
///
/// Pipeline:
/// 1. Batched RMSNorm: `d_hidden → d_normed` (all tokens at once)
/// 2. Fused gate+up+SwiGLU GEMM: `d_normed → d_swiglu` (all tokens)
/// 3. Down GEMM + residual: `d_swiglu → d_hidden` (fused residual add)
///
/// On return, `d_hidden` in `pb` contains the updated residual stream.
///
/// # Safety
/// All device buffers must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn encode_prefill_ffn_phase(
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

    // Step 2: Fused gate+up+SwiGLU GEMM (all tokens)
    //   d_normed [bs × h, col-major] → d_swiglu [bs × inter, col-major]
    //   Weight: concatenated gate+up SoA, 2*inter rows, k=h
    launch_fused_gate_up_swiglu_gemm(
        graph,
        pmods,
        d_gate_up_weight,
        &pb.d_normed,
        &mut pb.d_swiglu,
        inter,
        h,
        bs,
    )?;

    // Step 3: Down GEMM into d_normed (scratch), then in-place residual add.
    //
    // gemm_v7 accumulates with +=, so d_normed must be zeroed before the GEMM.
    // d_normed is free here (consumed as GEMM input in step 2 already).
    {
        let n = pb.actual_batch_size * pb.hidden_size;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_normed down: {e}")))?;
    }
    launch_gemm_v7(
        graph,
        pmods,
        d_down_weight,
        &pb.d_swiglu,
        &mut pb.d_normed,
        h,
        inter,
        bs,
    )?;

    // residual add: d_input[i] += d_normed[i]  (total = bs * hidden_size elements)
    let total_bh = (pb.actual_batch_size * pb.hidden_size) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_bh)?;

    Ok(())
}

// =============================================================================
// encode_prefill_layer
// =============================================================================

/// Encode one full transformer layer for batch prefill.
///
/// Non-attention operations use batched GEMM kernels.  Attention is processed
/// sequentially per token (using the existing single-token attention kernels)
/// because each query position needs access to all prior KV entries up to its
/// position — there is no batched attention kernel available.
///
/// On entry / exit, `pb.d_input` holds the batched residual stream
/// `[batch_size × hidden_size]` in column-major layout.
///
/// # Safety
/// All device buffers and weight slices must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn encode_prefill_layer(
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

    // ════════════════════════════════════════════════════════════════════
    // Attention QKV projection is computed PER TOKEN inside `encode_attn_phase`
    // (which runs its own RMSNorm + fused-QKV GEMV on `st_bufs.d_hidden`).  A
    // batched attn-RMSNorm + batched-QKV GEMM was previously run here into
    // `pb.d_normed` / `pb.d_qkv`, but their outputs were never consumed by the
    // per-token attention loop below — they were pure wasted device work.  They
    // are intentionally omitted (see finding: "batched prefill QKV GEMM
    // discarded").  The batched GEMM kernels are still used for the O-projection
    // and the FFN sublayer further down.
    // ════════════════════════════════════════════════════════════════════

    // ════════════════════════════════════════════════════════════════════
    // Sequential attention for each token
    //
    // For each token t at sequence position (pos_start + t), we:
    //   a) Copy this token's hidden state into st_bufs.d_hidden
    //   b) Upload pos/seqlen [pos, pos+1] into st_bufs.d_pos_seqlen so the
    //      fused KV-store writes at the correct position and attention spans
    //      exactly positions 0..=pos
    //   c) Copy this token's RoPE cos/sin into st_bufs.d_cos/d_sin
    //   d) Run the standard single-token attention kernels (rmsnorm, qkv gemv,
    //      qk-norm+rope, kv-store, scores, softmax, weighted sum)
    //   e) Copy attention output back into the column of pb.d_attn_out
    // ════════════════════════════════════════════════════════════════════
    let f_size = std::mem::size_of::<f32>();

    // Zero out d_attn_out before the sequential attention loop.
    {
        let n = bs * nq * hd;
        let mut dst_view = pb.d_attn_out.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_attn_out: {e}")))?;
    }

    for t in 0..bs {
        let pos = pos_start + t;

        // Copy token t's hidden state column into st_bufs.d_hidden
        // Column-major: token t's hidden is at pb.d_input[t * h .. (t+1)*h]
        {
            let src_view: CudaView<f32> = pb.d_input.slice(t * h..(t + 1) * h);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                .map_err(|e| CudaGraphError::DriverError(format!("copy hidden t={t}: {e}")))?;
        }

        // Upload pos/seqlen [pos, pos+1] into st_bufs.d_pos_seqlen.
        //
        // The fused KV-store kernel reads the write position from
        // d_pos_seqlen[0] and the attention span (seq_len) from d_pos_seqlen[1].
        // Without this per-token upload the Q1 batch-prefill path would write
        // every token's K/V at a stale/zero position and attend over the wrong
        // span, corrupting the shared decode KV cache (the ternary loop already
        // does this — mirror it here).
        let pos_seqlen = [pos as u32, (pos + 1) as u32];
        graph
            .stream_arc()
            .memcpy_htod(&pos_seqlen, &mut st_bufs.d_pos_seqlen)
            .map_err(|e| CudaGraphError::DriverError(format!("upload pos_seqlen t={t}: {e}")))?;

        // Upload RoPE cos/sin for this token's position.
        let rope_off = t * half_dim;
        graph
            .stream_arc()
            .memcpy_htod(
                &cos_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_cos,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload cos t={t}: {e}")))?;
        graph
            .stream_arc()
            .memcpy_htod(
                &sin_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_sin,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload sin t={t}: {e}")))?;

        // Run the single-token attention pipeline (rmsnorm → QKV GEMV →
        // qk-norm+rope → kv-store → scores → softmax → weighted sum).
        // encode_attn_phase reads from st_bufs.d_hidden (set above) and computes
        // its own RMSNorm + fused-QKV GEMV — there is no batched QKV to reuse,
        // which is why the previously-dead batched attn RMSNorm/QKV GEMM above
        // was removed.  The KV-store and attention steps read the token position
        // and span from st_bufs.d_pos_seqlen (uploaded above).
        encode_attn_phase(
            graph,
            attn_mods,
            d_attn_norm_weight,
            d_fused_qkv_weight,
            d_q_norm_weight,
            d_k_norm_weight,
            kv,
            layer_idx,
            pos,
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
                .map_err(|e| CudaGraphError::DriverError(format!("copy attn_out t={t}: {e}")))?;
        }

        // Silence f_size unused warning (used contextually in offset calculations)
        let _ = f_size;
    }

    // ════════════════════════════════════════════════════════════════════
    // 4. Output projection GEMM + residual (all tokens at once)
    //    attn_out_proj: [h × nq*hd], maps d_attn_out → d_normed (scratch)
    //    then: d_input += d_normed  (residual add)
    //
    // We use d_normed as a scratch to avoid aliasing d_input as both
    // &mut output and &residual in the fused gemm_v7_residual kernel.
    // ════════════════════════════════════════════════════════════════════
    {
        // Zero d_normed so the accumulating gemm_v7 starts from zero.
        let n = bs * h;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_normed oproj: {e}")))?;
    }
    launch_gemm_v7(
        graph,
        pmods,
        d_attn_proj_weight,
        &pb.d_attn_out,
        &mut pb.d_normed,
        h_u32,
        (nq * hd) as u32,
        bs_u32,
    )?;
    // residual add: d_input[i] += d_normed[i]
    let total_oproj = (bs * h) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_oproj)?;

    // ════════════════════════════════════════════════════════════════════
    // 5. Batched FFN (RMSNorm → fused gate+up+SwiGLU → down + residual)
    // ════════════════════════════════════════════════════════════════════
    encode_prefill_ffn_phase(
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
