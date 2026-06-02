//! # CudaGraph — FLUX.2 DiT fused-block encoder (resident forward)
//!
//! [`CudaGraph::encode_dit_single_block`] runs an entire FLUX.2 DiT
//! single-stream transformer block **GPU-resident**: the activation `h` is
//! uploaded once, every op (LayerNorm, modulation, the fused qkv-mlp ternary
//! matmul, the q/k/v reshape, QK-RMSNorm, RoPE, flash-attention, SwiGLU, the
//! `[attn ‖ gated]` concat, the `to_out` ternary matmul, the gated residual add)
//! runs chained on device, and `h` is downloaded once — collapsing the ~11
//! per-op host downloads of the unfused path to a single one.
//!
//! Mirrors `oxibonsai-image::blocks::SingleBlock::forward` bit-for-cos (each
//! kernel is the parity-validated device port of its CPU op). Host-API: host `h`
//! in/out, ternary weights passed as `(handle, aos_bytes)` (uploaded/cached
//! internally, no cudarc type in the signature); the image-crate caller (behind
//! `OXI_DIT_FUSED`) falls back to the CPU block on any `Err`, and `h` is written
//! back only on success — so a failure leaves the input pristine.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

/// Per-block weights for [`CudaGraph::encode_dit_single_blocks`]. Ternary weights
/// are `(stable handle, AoS bytes)` (uploaded/cached internally by handle); the
/// QK-norm vectors are `[head_dim]`. The modulation (`shift`/`scale`/`gate`) and
/// RoPE (`cos`/`sin`) are SHARED across the whole single-block stack, so they are
/// passed once to the encoder rather than per block.
pub struct DitSingleBlockWeights<'a> {
    /// `to_qkv_mlp_proj` ternary weight handle (cache key).
    pub proj_handle: u64,
    /// `to_qkv_mlp_proj` ternary AoS bytes.
    pub proj_bytes: &'a [u8],
    /// `to_out` ternary weight handle (cache key).
    pub out_handle: u64,
    /// `to_out` ternary AoS bytes.
    pub out_bytes: &'a [u8],
    /// Per-head QK-RMSNorm weight for q, `[head_dim]`.
    pub norm_q: &'a [f32],
    /// Per-head QK-RMSNorm weight for k, `[head_dim]`.
    pub norm_k: &'a [f32],
}

impl CudaGraph {
    /// Run one DiT single-stream block on `h` `[seq, hidden]` in place, resident.
    ///
    /// - `proj_*`: the fused `to_qkv_mlp_proj` ternary weight (`(handle,
    ///   aos_bytes)`), `proj_out = 3*hidden + 2*ffn_inner`.
    /// - `out_*`: the `to_out` ternary weight (`in = hidden + ffn_inner`).
    /// - `norm_q`/`norm_k`: per-head QK-RMSNorm weights `[head_dim]`.
    /// - `shift`/`scale`/`gate`: single-block modulation `[hidden]`.
    /// - `cos`/`sin`: RoPE tables `[seq, head_dim/2]`.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on any shape mismatch (incl. `hidden` /
    /// `hidden+ffn_inner` not multiples of 128, the ternary-GEMM constraint), a
    /// weight upload, or a buffer / launch failure. `h` is unmodified on `Err`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_dit_single_block(
        &self,
        h: &mut [f32],
        proj_handle: u64,
        proj_bytes: &[u8],
        proj_out: usize,
        out_handle: u64,
        out_bytes: &[u8],
        norm_q: &[f32],
        norm_k: &[f32],
        shift: &[f32],
        scale: &[f32],
        gate: &[f32],
        cos: &[f32],
        sin: &[f32],
        seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        ffn_inner: usize,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        // ── Validate ─────────────────────────────────────────────────────
        if hidden != num_heads * head_dim {
            return Err(CudaGraphError::DriverError(format!(
                "dit_single_block: hidden {hidden} != num_heads*head_dim {}",
                num_heads * head_dim
            )));
        }
        if head_dim % 2 != 0 {
            return Err(CudaGraphError::DriverError(
                "dit_single_block: head_dim must be even".into(),
            ));
        }
        let half = head_dim / 2;
        let qkv_w = 3 * hidden;
        let mlp_w = 2 * ffn_inner;
        let cat_w = hidden + ffn_inner;
        if proj_out != qkv_w + mlp_w {
            return Err(CudaGraphError::DriverError(format!(
                "dit_single_block: proj_out {proj_out} != 3*hidden+2*ffn_inner {}",
                qkv_w + mlp_w
            )));
        }
        // The ternary GEMM requires the contraction dim be a multiple of 128.
        if hidden % 128 != 0 || cat_w % 128 != 0 {
            return Err(CudaGraphError::DriverError(format!(
                "dit_single_block: gemm_tq2 k must be %128 (hidden {hidden}, hidden+ffn {cat_w})"
            )));
        }
        if h.len() != seq * hidden
            || norm_q.len() != head_dim
            || norm_k.len() != head_dim
            || shift.len() != hidden
            || scale.len() != hidden
            || gate.len() != hidden
            || cos.len() != seq * half
            || sin.len() != seq * half
        {
            return Err(CudaGraphError::DriverError(
                "dit_single_block: a param length is wrong".into(),
            ));
        }
        if seq == 0 {
            return Ok(());
        }

        // ── Upload weights (cached by handle) + params + the resident h. ──
        let d_proj_w =
            self.get_or_upload_weight_tq2_soa_lazy(proj_handle, || proj_bytes.to_vec())?;
        let d_out_w = self.get_or_upload_weight_tq2_soa_lazy(out_handle, || out_bytes.to_vec())?;

        let s = &self.stream;
        macro_rules! up {
            ($host:expr, $what:expr) => {
                s.clone_htod($host).map_err(|e| {
                    CudaGraphError::DriverError(format!("dit_single_block {}: {e}", $what))
                })?
            };
        }
        macro_rules! zeros {
            ($n:expr, $what:expr) => {
                s.alloc_zeros::<f32>($n).map_err(|e| {
                    CudaGraphError::DriverError(format!("dit_single_block {}: {e}", $what))
                })?
            };
        }
        let mut d_h = up!(h, "htod h"); // residual stream
        let mut d_n = up!(h, "htod n"); // n = h.clone() (normed/modulated below)
        let d_shift = up!(shift, "htod shift");
        let d_scale = up!(scale, "htod scale");
        let d_gate = up!(gate, "htod gate");
        let d_nq = up!(norm_q, "htod norm_q");
        let d_nk = up!(norm_k, "htod norm_k");
        let d_cos = up!(cos, "htod cos");
        let d_sin = up!(sin, "htod sin");

        let (sq, hd, nh, hdd) = (seq as u32, hidden as u32, num_heads as u32, head_dim as u32);
        let qkv = num_heads * seq * head_dim; // == seq*hidden
        let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut d_proj = zeros!(seq * proj_out, "alloc proj");
        let mut d_q = zeros!(qkv, "alloc q");
        let mut d_k = zeros!(qkv, "alloc k");
        let mut d_v = zeros!(qkv, "alloc v");
        let mut d_attn = zeros!(seq * hidden, "alloc attn");
        let mut d_mlp = zeros!(seq * mlp_w, "alloc mlp");
        let mut d_gated = zeros!(seq * ffn_inner, "alloc gated");
        let mut d_cat = zeros!(seq * cat_w, "alloc cat");
        let mut d_out = zeros!(seq * hidden, "alloc out");

        // SAFETY: every device buffer above is alloc_zeros'd / uploaded to the
        // exact size each launch reads/writes (validated dims), all on
        // `self.stream`; the launches are ordered on it. The chained kernels are
        // each individually parity-validated against their CPU port.
        unsafe {
            // n = modulate(LN(h)).
            self.launch_dit_layer_norm(&mut d_n, sq, hd, eps)?;
            self.launch_dit_modulate(&mut d_n, &d_shift, &d_scale, sq, hd)?;
            // proj = n · dequant(proj_w)ᵀ  → [seq, proj_out].
            self.launch_gemm_tq2(&d_proj_w, &d_n, &mut d_proj, proj_out as u32, sq, hd)?;
            // Head-major q/k/v from proj columns {0, hidden, 2·hidden}.
            self.launch_dit_tokens_to_heads(&d_proj, &mut d_q, sq, nh, hdd, proj_out as u32, 0)?;
            self.launch_dit_tokens_to_heads(&d_proj, &mut d_k, sq, nh, hdd, proj_out as u32, hd)?;
            self.launch_dit_tokens_to_heads(
                &d_proj,
                &mut d_v,
                sq,
                nh,
                hdd,
                proj_out as u32,
                2 * hd,
            )?;
            // QK-RMSNorm, then RoPE, on q and k.
            let nrows = (num_heads * seq) as u32;
            self.launch_dit_rms_norm_heads(&mut d_q, &d_nq, nrows, hdd, eps)?;
            self.launch_dit_rms_norm_heads(&mut d_k, &d_nk, nrows, hdd, eps)?;
            self.launch_dit_rope(&mut d_q, &d_cos, &d_sin, nh, sq, hdd)?;
            self.launch_dit_rope(&mut d_k, &d_cos, &d_sin, nh, sq, hdd)?;
            // Flash attention → token-major [seq, hidden].
            self.launch_joint_attention_flash_resident(
                &d_q,
                &d_k,
                &d_v,
                &mut d_attn,
                nh,
                sq,
                hdd,
                attn_scale,
            )?;
            // mlp slab = proj[:, 3·hidden..] → swiglu → [seq, ffn_inner].
            self.launch_dit_strided_row_copy(
                &mut d_mlp,
                &d_proj,
                sq,
                mlp_w as u32,
                mlp_w as u32,
                0,
                proj_out as u32,
                qkv_w as u32,
            )?;
            self.launch_dit_swiglu(&d_mlp, &mut d_gated, sq, ffn_inner as u32)?;
            // cat = [attn ‖ gated]  → [seq, hidden+ffn_inner].
            self.launch_dit_strided_row_copy(&mut d_cat, &d_attn, sq, hd, cat_w as u32, 0, hd, 0)?;
            self.launch_dit_strided_row_copy(
                &mut d_cat,
                &d_gated,
                sq,
                ffn_inner as u32,
                cat_w as u32,
                hd,
                ffn_inner as u32,
                0,
            )?;
            // out = cat · dequant(out_w)ᵀ → [seq, hidden]; then h += gate·out.
            self.launch_gemm_tq2(&d_out_w, &d_cat, &mut d_out, hd, sq, cat_w as u32)?;
            self.launch_dit_gated_residual_add(&mut d_h, &d_out, &d_gate, sq, hd)?;
        }

        // ── Single download of the updated residual stream. ──
        s.memcpy_dtoh(&d_h, h)
            .map_err(|e| CudaGraphError::DriverError(format!("dit_single_block dtoh h: {e}")))?;
        s.synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("dit_single_block sync: {e}")))?;
        Ok(())
    }

    /// Run a whole stack of DiT single-stream blocks GPU-RESIDENT.
    ///
    /// Same op-for-op chain as [`Self::encode_dit_single_block`] per block, but
    /// the residual stream `h`, the SHARED modulation (`shift`/`scale`/`gate`)
    /// and RoPE (`cos`/`sin`), and all scratch buffers are uploaded / allocated
    /// **once**; the `N` blocks then run chained on the device-resident `d_h`
    /// (each block's ternary weights are cached by handle, only the tiny
    /// per-block QK-norm vectors are re-staged), and `h` is downloaded **once**
    /// with a single barrier. On a discrete GPU this removes `N-1` PCIe
    /// round-trips of the 18 MiB `h` plus `N-1` device syncs — the dominant DiT
    /// "glue" cost the per-block [`Self::encode_dit_single_block`] path pays.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] as [`Self::encode_dit_single_block`]; `h`
    /// is unmodified on `Err` (the device→host copy is the final step).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_dit_single_blocks(
        &self,
        h: &mut [f32],
        blocks: &[DitSingleBlockWeights],
        proj_out: usize,
        shift: &[f32],
        scale: &[f32],
        gate: &[f32],
        cos: &[f32],
        sin: &[f32],
        seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        ffn_inner: usize,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        if blocks.is_empty() || seq == 0 {
            return Ok(());
        }
        // ── Validate (shared) ────────────────────────────────────────────
        if hidden != num_heads * head_dim || head_dim % 2 != 0 {
            return Err(CudaGraphError::DriverError(
                "dit_single_blocks: hidden != heads*head_dim, or odd head_dim".into(),
            ));
        }
        let half = head_dim / 2;
        let qkv_w = 3 * hidden;
        let mlp_w = 2 * ffn_inner;
        let cat_w = hidden + ffn_inner;
        if proj_out != qkv_w + mlp_w {
            return Err(CudaGraphError::DriverError(
                "dit_single_blocks: proj_out != 3*hidden+2*ffn_inner".into(),
            ));
        }
        if hidden % 128 != 0 || cat_w % 128 != 0 {
            return Err(CudaGraphError::DriverError(
                "dit_single_blocks: gemm_tq2 k must be %128".into(),
            ));
        }
        if h.len() != seq * hidden
            || shift.len() != hidden
            || scale.len() != hidden
            || gate.len() != hidden
            || cos.len() != seq * half
            || sin.len() != seq * half
        {
            return Err(CudaGraphError::DriverError(
                "dit_single_blocks: a shared param length is wrong".into(),
            ));
        }

        let s = &self.stream;
        macro_rules! up {
            ($host:expr, $what:expr) => {
                s.clone_htod($host).map_err(|e| {
                    CudaGraphError::DriverError(format!("dit_single_blocks {}: {e}", $what))
                })?
            };
        }
        macro_rules! zeros {
            ($n:expr, $what:expr) => {
                s.alloc_zeros::<f32>($n).map_err(|e| {
                    CudaGraphError::DriverError(format!("dit_single_blocks {}: {e}", $what))
                })?
            };
        }

        // Resident residual stream + SHARED params + scratch — uploaded/allocated ONCE.
        let mut d_h = up!(h, "htod h");
        let mut d_n = zeros!(seq * hidden, "alloc n");
        let d_shift = up!(shift, "htod shift");
        let d_scale = up!(scale, "htod scale");
        let d_gate = up!(gate, "htod gate");
        let d_cos = up!(cos, "htod cos");
        let d_sin = up!(sin, "htod sin");

        let (sq, hd, nh, hdd) = (seq as u32, hidden as u32, num_heads as u32, head_dim as u32);
        let qkv = num_heads * seq * head_dim;
        let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
        let nrows = (num_heads * seq) as u32;

        let mut d_proj = zeros!(seq * proj_out, "alloc proj");
        let mut d_q = zeros!(qkv, "alloc q");
        let mut d_k = zeros!(qkv, "alloc k");
        let mut d_v = zeros!(qkv, "alloc v");
        let mut d_attn = zeros!(seq * hidden, "alloc attn");
        let mut d_mlp = zeros!(seq * mlp_w, "alloc mlp");
        let mut d_gated = zeros!(seq * ffn_inner, "alloc gated");
        let mut d_cat = zeros!(seq * cat_w, "alloc cat");
        let mut d_out = zeros!(seq * hidden, "alloc out");

        for (bi, b) in blocks.iter().enumerate() {
            if b.norm_q.len() != head_dim || b.norm_k.len() != head_dim {
                return Err(CudaGraphError::DriverError(format!(
                    "dit_single_blocks: block {bi} norm length != head_dim"
                )));
            }
            let d_proj_w =
                self.get_or_upload_weight_tq2_soa_lazy(b.proj_handle, || b.proj_bytes.to_vec())?;
            let d_out_w =
                self.get_or_upload_weight_tq2_soa_lazy(b.out_handle, || b.out_bytes.to_vec())?;
            let d_nq = up!(b.norm_q, "htod norm_q"); // tiny ([head_dim]); per block
            let d_nk = up!(b.norm_k, "htod norm_k");

            // n ← current resident h (device-to-device; no PCIe round-trip).
            s.memcpy_dtod(&d_h, &mut d_n).map_err(|e| {
                CudaGraphError::DriverError(format!("dit_single_blocks dtod h->n: {e}"))
            })?;

            // SAFETY: all device buffers are sized to exactly what each launch
            // reads/writes (validated dims), ordered on `self.stream`. The chain
            // is identical to the parity-validated `encode_dit_single_block`.
            unsafe {
                self.launch_dit_layer_norm(&mut d_n, sq, hd, eps)?;
                self.launch_dit_modulate(&mut d_n, &d_shift, &d_scale, sq, hd)?;
                self.launch_gemm_tq2(&d_proj_w, &d_n, &mut d_proj, proj_out as u32, sq, hd)?;
                self.launch_dit_tokens_to_heads(
                    &d_proj,
                    &mut d_q,
                    sq,
                    nh,
                    hdd,
                    proj_out as u32,
                    0,
                )?;
                self.launch_dit_tokens_to_heads(
                    &d_proj,
                    &mut d_k,
                    sq,
                    nh,
                    hdd,
                    proj_out as u32,
                    hd,
                )?;
                self.launch_dit_tokens_to_heads(
                    &d_proj,
                    &mut d_v,
                    sq,
                    nh,
                    hdd,
                    proj_out as u32,
                    2 * hd,
                )?;
                self.launch_dit_rms_norm_heads(&mut d_q, &d_nq, nrows, hdd, eps)?;
                self.launch_dit_rms_norm_heads(&mut d_k, &d_nk, nrows, hdd, eps)?;
                self.launch_dit_rope(&mut d_q, &d_cos, &d_sin, nh, sq, hdd)?;
                self.launch_dit_rope(&mut d_k, &d_cos, &d_sin, nh, sq, hdd)?;
                self.launch_joint_attention_flash_resident(
                    &d_q,
                    &d_k,
                    &d_v,
                    &mut d_attn,
                    nh,
                    sq,
                    hdd,
                    attn_scale,
                )?;
                self.launch_dit_strided_row_copy(
                    &mut d_mlp,
                    &d_proj,
                    sq,
                    mlp_w as u32,
                    mlp_w as u32,
                    0,
                    proj_out as u32,
                    qkv_w as u32,
                )?;
                self.launch_dit_swiglu(&d_mlp, &mut d_gated, sq, ffn_inner as u32)?;
                self.launch_dit_strided_row_copy(
                    &mut d_cat,
                    &d_attn,
                    sq,
                    hd,
                    cat_w as u32,
                    0,
                    hd,
                    0,
                )?;
                self.launch_dit_strided_row_copy(
                    &mut d_cat,
                    &d_gated,
                    sq,
                    ffn_inner as u32,
                    cat_w as u32,
                    hd,
                    ffn_inner as u32,
                    0,
                )?;
                self.launch_gemm_tq2(&d_out_w, &d_cat, &mut d_out, hd, sq, cat_w as u32)?;
                self.launch_dit_gated_residual_add(&mut d_h, &d_out, &d_gate, sq, hd)?;
            }
        }

        // ── Single download of the residual stream after all blocks. ──
        s.memcpy_dtoh(&d_h, h)
            .map_err(|e| CudaGraphError::DriverError(format!("dit_single_blocks dtoh h: {e}")))?;
        s.synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("dit_single_blocks sync: {e}")))?;
        Ok(())
    }
}
