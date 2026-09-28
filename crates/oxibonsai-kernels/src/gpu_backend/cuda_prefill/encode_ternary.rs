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

use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};

use crate::gpu_backend::cuda_full_layer::{
    acquire_prefill_rope_chunk, encode_attn_phase_tq2, CudaAttnModules, CudaFullLayerBuffers,
    CudaKvCache,
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

    // ════════════════════════════════════════════════════════════════════
    // Attention QKV projection is computed PER TOKEN inside
    // `encode_attn_phase_tq2` (which runs its own RMSNorm + TQ2 fused-QKV GEMV
    // on this token's column of `pb.d_input`).  A batched attn-RMSNorm +
    // batched-TQ2-QKV GEMM was
    // previously run here into `pb.d_normed` / `pb.d_qkv`, but those outputs were
    // never consumed by the per-token attention loop below — they were pure
    // wasted device work (see finding: "batched prefill QKV GEMM discarded").
    // They are intentionally omitted.  The batched TQ2 GEMM kernels are still
    // used for the O-projection and the FFN sublayer further down.
    // ════════════════════════════════════════════════════════════════════

    // ════════════════════════════════════════════════════════════════════
    // Sequential attention for each token (TQ2-aware)
    //
    // For each token t at sequence position (pos_start + t):
    //   a) Read this token's [pos, pos+1] / RoPE cos / RoPE sin from the
    //      chunk-resident upload below (finding F9)
    //   b) Call encode_attn_phase_tq2: runs rmsnorm + TQ2 QKV GEMV +
    //      qk-norm+rope + kv-store + scores + softmax + weighted sum,
    //      reading this token's hidden state from and writing its attention
    //      output directly into columns of pb.d_input / pb.d_attn_out
    //
    // F9 — copy count. Naively, each token here would need five sub-kilobyte
    // transfers: three host-to-device uploads of `[pos, pos+1]` / cos / sin,
    // plus a device-to-device copy in and out of `encode_attn_phase_tq2`'s
    // scratch slots (`st_bufs.d_hidden` / `st_bufs.d_attn_out`). This
    // function runs once per layer and loops over every token inside, so all
    // five would also repeat `n_layers` times over. The three uploads are
    // one chunk upload each, hoisted out of the loop
    // (`acquire_prefill_rope_chunk`), with the per-token attention reading
    // device views into that buffer — mirroring the Q1 prefill path's
    // identical fix. The remaining two device-to-device copies are
    // eliminated the same way: `encode_attn_phase_tq2`'s `hidden_in` /
    // `attn_out` parameters point steps 1 and 7 straight at this token's
    // column of `pb.d_input` / `pb.d_attn_out`, so `st_bufs.d_hidden` /
    // `st_bufs.d_attn_out` are never touched by batch prefill at all — they
    // stay this buffer set's decode-path scratch slots.
    // ════════════════════════════════════════════════════════════════════
    {
        let n = bs * nq * hd;
        let mut dst_view = pb.d_attn_out.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_attn_out tq2: {e}")))?;
    }

    // F9: one upload of the whole chunk's positions and RoPE tables,
    // replacing `3 x bs` per-token uploads.
    let mut chunk_guard =
        acquire_prefill_rope_chunk(graph, pos_start, bs, half_dim, cos_table, sin_table)?;
    let chunk = chunk_guard.as_mut().ok_or_else(|| {
        CudaGraphError::DriverError("prefill rope chunk missing after upload (tq2)".to_string())
    })?;

    for t in 0..bs {
        // Views into the chunk upload: `[pos, pos+1]` plus this token's RoPE
        // cosines and sines.
        let token_inputs = chunk.token(t)?;

        // This token's hidden-state column and attention-output column,
        // read/written in place (F9) — no per-token device-to-device copy
        // either side.
        let hidden_view: CudaView<f32> = pb.d_input.slice(t * h..(t + 1) * h);
        let mut attn_out_view: CudaViewMut<f32> =
            pb.d_attn_out.slice_mut(t * nq * hd..(t + 1) * nq * hd);

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
            Some(&token_inputs),
            Some(&hidden_view),
            Some(&mut attn_out_view),
        )?;
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
