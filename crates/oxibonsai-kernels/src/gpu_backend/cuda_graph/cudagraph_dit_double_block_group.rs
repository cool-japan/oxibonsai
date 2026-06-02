//! # CudaGraph — FLUX.2 DiT fused **double**-block encoder (resident forward)
//!
//! [`CudaGraph::encode_dit_double_block`] runs an entire FLUX.2 DiT dual-stream
//! transformer block GPU-resident: the image stream `hidden` and the text stream
//! `enc` are each uploaded once, the whole block (per-stream modulated LayerNorm,
//! the six separate q/k/v ternary projections, per-stream QK-RMSNorm, the txt‖img
//! head-major concat, RoPE, the single joint flash-attention over the 1536-token
//! sequence, the two `to_out` projections, the per-stream SwiGLU feed-forwards,
//! and all four gated residual adds) runs chained on device, and the two streams
//! are downloaded once — collapsing the unfused path's ~30 host round-trips per
//! block to two.
//!
//! Mirrors `oxibonsai-image::blocks::DoubleBlock::forward` op-for-op. Every op is
//! a parity-validated device kernel; the txt‖img concat and the post-attention
//! split reuse `strided_row_copy` (no new kernel). Host-API (host streams in/out,
//! ternary weights as `(handle, aos_bytes)`); the image caller (behind
//! `OXI_DIT_FUSED`) falls back to the per-op CPU/GPU block on any `Err`, and the
//! streams are written back only on success.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::CudaSlice;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

/// A ternary weight as `(cache-key handle, out-major AoS block bytes)`.
type TernaryWeight<'a> = (u64, &'a [u8]);
/// A modulation triple `(shift, scale, gate)`, each `[hidden]`.
type ModTriple<'a> = (&'a [f32], &'a [f32], &'a [f32]);

impl CudaGraph {
    /// Project `n` `[seq, hidden]` with a ternary weight `[hidden, hidden]` to
    /// head-major `[num_heads, seq, head_dim]` (`gemm → to_heads`), optionally
    /// applying per-head QK-RMSNorm. Returns the head-major device buffer.
    ///
    /// # Safety
    /// `d_weight` is the SoA upload of a `[hidden, hidden]` ternary weight;
    /// `d_n` holds ≥ `seq*hidden` f32; `norm` (if any) holds ≥ `head_dim`; on
    /// `self.stream`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn dit_project_to_heads(
        &self,
        d_weight: &CudaSlice<u8>,
        d_n: &CudaSlice<f32>,
        seq: u32,
        hidden: u32,
        num_heads: u32,
        head_dim: u32,
        norm: Option<&CudaSlice<f32>>,
        eps: f32,
    ) -> Result<CudaSlice<f32>, CudaGraphError> {
        let mut tok = self.alloc_zeros(seq as usize * hidden as usize)?;
        self.launch_gemm_tq2(d_weight, d_n, &mut tok, hidden, seq, hidden)?;
        let mut heads = self.alloc_zeros(num_heads as usize * seq as usize * head_dim as usize)?;
        self.launch_dit_tokens_to_heads(&tok, &mut heads, seq, num_heads, head_dim, hidden, 0)?;
        if let Some(n) = norm {
            self.launch_dit_rms_norm_heads(&mut heads, n, num_heads * seq, head_dim, eps)?;
        }
        Ok(heads)
    }

    /// Concat two head-major buffers `txt` `[num_heads, seq_txt, head_dim]` and
    /// `img` `[num_heads, seq_img, head_dim]` into `joint` `[num_heads,
    /// seq_txt+seq_img, head_dim]` **txt-first** (the order the CPU
    /// `concat_heads` + RoPE tables expect), via two per-head strided copies.
    ///
    /// # Safety
    /// `d_joint` holds ≥ `num_heads*(seq_txt+seq_img)*head_dim` f32; the sources
    /// hold their `num_heads*seq*head_dim`; on `self.stream`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn dit_concat_heads_into(
        &self,
        d_joint: &mut CudaSlice<f32>,
        d_txt: &CudaSlice<f32>,
        d_img: &CudaSlice<f32>,
        num_heads: u32,
        seq_txt: u32,
        seq_img: u32,
        head_dim: u32,
    ) -> Result<(), CudaGraphError> {
        let seq_joint = seq_txt + seq_img;
        let txt_cols = seq_txt * head_dim;
        let img_cols = seq_img * head_dim;
        let joint_stride = seq_joint * head_dim;
        // txt block → joint[h, 0 .. seq_txt]
        self.launch_dit_strided_row_copy(
            d_joint,
            d_txt,
            num_heads,
            txt_cols,
            joint_stride,
            0,
            txt_cols,
            0,
        )?;
        // img block → joint[h, seq_txt .. seq_joint]
        self.launch_dit_strided_row_copy(
            d_joint,
            d_img,
            num_heads,
            img_cols,
            joint_stride,
            txt_cols,
            img_cols,
            0,
        )?;
        Ok(())
    }

    /// One stream's feed-forward: `linear_in [hidden→2·ffn] → SwiGLU → linear_out
    /// [ffn→hidden]`. Returns `[seq, hidden]`.
    ///
    /// # Safety
    /// `d_lin_in`/`d_lin_out` are the SoA uploads of the `[2·ffn, hidden]` /
    /// `[hidden, ffn]` ternary weights; `d_n` holds ≥ `seq*hidden` f32; on
    /// `self.stream`.
    unsafe fn dit_feed_forward(
        &self,
        d_lin_in: &CudaSlice<u8>,
        d_lin_out: &CudaSlice<u8>,
        d_n: &CudaSlice<f32>,
        seq: u32,
        hidden: u32,
        ffn_inner: u32,
    ) -> Result<CudaSlice<f32>, CudaGraphError> {
        let mut proj = self.alloc_zeros(seq as usize * 2 * ffn_inner as usize)?;
        self.launch_gemm_tq2(d_lin_in, d_n, &mut proj, 2 * ffn_inner, seq, hidden)?;
        let mut gated = self.alloc_zeros(seq as usize * ffn_inner as usize)?;
        self.launch_dit_swiglu(&proj, &mut gated, seq, ffn_inner)?;
        let mut out = self.alloc_zeros(seq as usize * hidden as usize)?;
        self.launch_gemm_tq2(d_lin_out, &gated, &mut out, hidden, seq, ffn_inner)?;
        Ok(out)
    }

    /// On-device clone of `[seq, hidden]` then in-place modulated LayerNorm
    /// (`modulate(LN(x), shift, scale)`) — the per-stream norm input for a
    /// sub-block. Cloned on device because the source stream has been mutated by
    /// a prior residual add (the host copy is stale).
    ///
    /// # Safety
    /// `d_src` holds ≥ `seq*hidden` f32; `d_shift`/`d_scale` hold ≥ `hidden`; on
    /// `self.stream`.
    unsafe fn dit_norm_modulate_clone(
        &self,
        d_src: &CudaSlice<f32>,
        d_shift: &CudaSlice<f32>,
        d_scale: &CudaSlice<f32>,
        seq: u32,
        hidden: u32,
        eps: f32,
    ) -> Result<CudaSlice<f32>, CudaGraphError> {
        let mut n = self.alloc_zeros(seq as usize * hidden as usize)?;
        self.launch_dit_strided_row_copy(&mut n, d_src, seq, hidden, hidden, 0, hidden, 0)?;
        self.launch_dit_layer_norm(&mut n, seq, hidden, eps)?;
        self.launch_dit_modulate(&mut n, d_shift, d_scale, seq, hidden)?;
        Ok(n)
    }

    /// Run one FLUX.2 DiT dual-stream (double) block on the image stream `hidden`
    /// `[seq_img, hidden_size]` and text stream `enc` `[seq_txt, hidden_size]` in
    /// place, resident.
    ///
    /// Weights (each `(handle, aos_bytes)`): `w_to_{q,k,v}` (image q/k/v),
    /// `w_add_{q,k,v}` (text q/k/v), `w_to_out`/`w_to_add_out` (image/text attn
    /// out), `w_ff_{in,out}` (image FFN), `w_ffc_{in,out}` (text FFN). Norms
    /// (`[head_dim]`): `norm_{q,k}_img`, `norm_{q,k}_txt`. Modulation triples
    /// (`(shift, scale, gate)`, `[hidden]`): `img_msa`/`img_mlp`/`txt_msa`/
    /// `txt_mlp`. RoPE `cos`/`sin` are `[seq_txt+seq_img, head_dim/2]`.
    ///
    /// # Errors
    /// [`CudaGraphError::DriverError`] on any shape mismatch (incl. `hidden` /
    /// `ffn_inner` not multiples of 128), a weight upload, or a buffer / launch
    /// failure. `hidden` and `enc` are unmodified on `Err`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_dit_double_block(
        &self,
        hidden: &mut [f32],
        enc: &mut [f32],
        w_to_q: TernaryWeight,
        w_to_k: TernaryWeight,
        w_to_v: TernaryWeight,
        w_add_q: TernaryWeight,
        w_add_k: TernaryWeight,
        w_add_v: TernaryWeight,
        w_to_out: TernaryWeight,
        w_to_add_out: TernaryWeight,
        w_ff_in: TernaryWeight,
        w_ff_out: TernaryWeight,
        w_ffc_in: TernaryWeight,
        w_ffc_out: TernaryWeight,
        norm_q_img: &[f32],
        norm_k_img: &[f32],
        norm_q_txt: &[f32],
        norm_k_txt: &[f32],
        img_msa: ModTriple,
        img_mlp: ModTriple,
        txt_msa: ModTriple,
        txt_mlp: ModTriple,
        cos: &[f32],
        sin: &[f32],
        seq_img: usize,
        seq_txt: usize,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        ffn_inner: usize,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        // ── Validate ─────────────────────────────────────────────────────
        let seq_joint = seq_txt + seq_img;
        let half = head_dim / 2;
        let err = |m: String| {
            Err(CudaGraphError::DriverError(format!(
                "dit_double_block: {m}"
            )))
        };
        if hidden_size != num_heads * head_dim {
            return err(format!(
                "hidden {hidden_size} != num_heads*head_dim {}",
                num_heads * head_dim
            ));
        }
        if head_dim % 2 != 0 {
            return err("head_dim must be even".into());
        }
        // Ternary GEMM contraction dims must be multiples of 128.
        if hidden_size % 128 != 0 || ffn_inner % 128 != 0 {
            return err(format!(
                "gemm_tq2 k must be %128 (hidden {hidden_size}, ffn_inner {ffn_inner})"
            ));
        }
        if hidden.len() != seq_img * hidden_size || enc.len() != seq_txt * hidden_size {
            return err("hidden/enc length mismatch".into());
        }
        for (v, n, want) in [
            (norm_q_img, "norm_q_img", head_dim),
            (norm_k_img, "norm_k_img", head_dim),
            (norm_q_txt, "norm_q_txt", head_dim),
            (norm_k_txt, "norm_k_txt", head_dim),
            (img_msa.0, "img_msa.shift", hidden_size),
            (img_msa.1, "img_msa.scale", hidden_size),
            (img_msa.2, "img_msa.gate", hidden_size),
            (img_mlp.0, "img_mlp.shift", hidden_size),
            (img_mlp.1, "img_mlp.scale", hidden_size),
            (img_mlp.2, "img_mlp.gate", hidden_size),
            (txt_msa.0, "txt_msa.shift", hidden_size),
            (txt_msa.1, "txt_msa.scale", hidden_size),
            (txt_msa.2, "txt_msa.gate", hidden_size),
            (txt_mlp.0, "txt_mlp.shift", hidden_size),
            (txt_mlp.1, "txt_mlp.scale", hidden_size),
            (txt_mlp.2, "txt_mlp.gate", hidden_size),
            (cos, "cos", seq_joint * half),
            (sin, "sin", seq_joint * half),
        ] {
            if v.len() != want {
                return err(format!("{n} len {} != {want}", v.len()));
            }
        }
        if seq_img == 0 || seq_txt == 0 {
            return err("seq_img/seq_txt must be > 0".into());
        }

        // ── Upload the 12 ternary weights (cached by handle). ──
        let up_w = |w: TernaryWeight| self.get_or_upload_weight_tq2_soa_lazy(w.0, || w.1.to_vec());
        let d_to_q = up_w(w_to_q)?;
        let d_to_k = up_w(w_to_k)?;
        let d_to_v = up_w(w_to_v)?;
        let d_add_q = up_w(w_add_q)?;
        let d_add_k = up_w(w_add_k)?;
        let d_add_v = up_w(w_add_v)?;
        let d_to_out = up_w(w_to_out)?;
        let d_to_add_out = up_w(w_to_add_out)?;
        let d_ff_in = up_w(w_ff_in)?;
        let d_ff_out = up_w(w_ff_out)?;
        let d_ffc_in = up_w(w_ffc_in)?;
        let d_ffc_out = up_w(w_ffc_out)?;

        // ── Upload streams, norms, modulation, RoPE. ──
        let mut d_hidden = self.htod(hidden)?;
        let mut d_enc = self.htod(enc)?;
        let d_nq_img = self.htod(norm_q_img)?;
        let d_nk_img = self.htod(norm_k_img)?;
        let d_nq_txt = self.htod(norm_q_txt)?;
        let d_nk_txt = self.htod(norm_k_txt)?;
        let d_im_sh = self.htod(img_msa.0)?;
        let d_im_sc = self.htod(img_msa.1)?;
        let d_im_g = self.htod(img_msa.2)?;
        let d_il_sh = self.htod(img_mlp.0)?;
        let d_il_sc = self.htod(img_mlp.1)?;
        let d_il_g = self.htod(img_mlp.2)?;
        let d_tm_sh = self.htod(txt_msa.0)?;
        let d_tm_sc = self.htod(txt_msa.1)?;
        let d_tm_g = self.htod(txt_msa.2)?;
        let d_tl_sh = self.htod(txt_mlp.0)?;
        let d_tl_sc = self.htod(txt_mlp.1)?;
        let d_tl_g = self.htod(txt_mlp.2)?;
        let d_cos = self.htod(cos)?;
        let d_sin = self.htod(sin)?;

        let (si, st, sj) = (seq_img as u32, seq_txt as u32, seq_joint as u32);
        let (hd, nh, hdd) = (hidden_size as u32, num_heads as u32, head_dim as u32);
        let fi = ffn_inner as u32;
        let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

        // SAFETY: every buffer below is sized to exactly what each launch
        // reads/writes (validated dims) and ordered on `self.stream`; each
        // chained kernel is individually parity-validated against its CPU port.
        unsafe {
            // ── Attention sub-block ──
            // n_h = modulate(LN(hidden), img_msa); n_e = modulate(LN(enc), txt_msa).
            // (Streams unmutated so far → upload-clone is valid.)
            let mut d_n_h = self.htod(hidden)?;
            self.launch_dit_layer_norm(&mut d_n_h, si, hd, eps)?;
            self.launch_dit_modulate(&mut d_n_h, &d_im_sh, &d_im_sc, si, hd)?;
            let mut d_n_e = self.htod(enc)?;
            self.launch_dit_layer_norm(&mut d_n_e, st, hd, eps)?;
            self.launch_dit_modulate(&mut d_n_e, &d_tm_sh, &d_tm_sc, st, hd)?;

            // Per-stream q/k/v → head-major (QK-RMSNorm on q,k; none on v).
            let q_img =
                self.dit_project_to_heads(&d_to_q, &d_n_h, si, hd, nh, hdd, Some(&d_nq_img), eps)?;
            let k_img =
                self.dit_project_to_heads(&d_to_k, &d_n_h, si, hd, nh, hdd, Some(&d_nk_img), eps)?;
            let v_img = self.dit_project_to_heads(&d_to_v, &d_n_h, si, hd, nh, hdd, None, eps)?;
            let q_txt =
                self.dit_project_to_heads(&d_add_q, &d_n_e, st, hd, nh, hdd, Some(&d_nq_txt), eps)?;
            let k_txt =
                self.dit_project_to_heads(&d_add_k, &d_n_e, st, hd, nh, hdd, Some(&d_nk_txt), eps)?;
            let v_txt = self.dit_project_to_heads(&d_add_v, &d_n_e, st, hd, nh, hdd, None, eps)?;

            // Concat txt‖img along seq → joint q/k/v [num_heads, seq_joint, head_dim].
            let mut d_q = self.alloc_zeros(num_heads * seq_joint * head_dim)?;
            let mut d_k = self.alloc_zeros(num_heads * seq_joint * head_dim)?;
            let mut d_v = self.alloc_zeros(num_heads * seq_joint * head_dim)?;
            self.dit_concat_heads_into(&mut d_q, &q_txt, &q_img, nh, st, si, hdd)?;
            self.dit_concat_heads_into(&mut d_k, &k_txt, &k_img, nh, st, si, hdd)?;
            self.dit_concat_heads_into(&mut d_v, &v_txt, &v_img, nh, st, si, hdd)?;

            // RoPE on the joint q,k (shared tables, txt-first order).
            self.launch_dit_rope(&mut d_q, &d_cos, &d_sin, nh, sj, hdd)?;
            self.launch_dit_rope(&mut d_k, &d_cos, &d_sin, nh, sj, hdd)?;

            // One joint flash-attention → token-major [seq_joint, hidden].
            let mut d_attn = self.alloc_zeros(seq_joint * hidden_size)?;
            self.launch_joint_attention_flash_resident(
                &d_q,
                &d_k,
                &d_v,
                &mut d_attn,
                nh,
                sj,
                hdd,
                attn_scale,
            )?;

            // Split: rows [0..seq_txt] → enc attn, [seq_txt..] → img attn.
            let mut d_enc_attn = self.alloc_zeros(seq_txt * hidden_size)?;
            self.launch_dit_strided_row_copy(&mut d_enc_attn, &d_attn, st, hd, hd, 0, hd, 0)?;
            let mut d_img_attn = self.alloc_zeros(seq_img * hidden_size)?;
            self.launch_dit_strided_row_copy(&mut d_img_attn, &d_attn, si, hd, hd, 0, hd, st * hd)?;

            // Per-stream to_out, then gated residual (msa.gate) into the streams.
            let mut d_img_out = self.alloc_zeros(seq_img * hidden_size)?;
            self.launch_gemm_tq2(&d_to_out, &d_img_attn, &mut d_img_out, hd, si, hd)?;
            let mut d_enc_out = self.alloc_zeros(seq_txt * hidden_size)?;
            self.launch_gemm_tq2(&d_to_add_out, &d_enc_attn, &mut d_enc_out, hd, st, hd)?;
            self.launch_dit_gated_residual_add(&mut d_hidden, &d_img_out, &d_im_g, si, hd)?;
            self.launch_dit_gated_residual_add(&mut d_enc, &d_enc_out, &d_tm_g, st, hd)?;

            // ── Feed-forward sub-block (re-norm from the residual-updated streams). ──
            let d_n_h2 =
                self.dit_norm_modulate_clone(&d_hidden, &d_il_sh, &d_il_sc, si, hd, eps)?;
            let d_ff_img = self.dit_feed_forward(&d_ff_in, &d_ff_out, &d_n_h2, si, hd, fi)?;
            self.launch_dit_gated_residual_add(&mut d_hidden, &d_ff_img, &d_il_g, si, hd)?;

            let d_n_e2 = self.dit_norm_modulate_clone(&d_enc, &d_tl_sh, &d_tl_sc, st, hd, eps)?;
            let d_ff_txt = self.dit_feed_forward(&d_ffc_in, &d_ffc_out, &d_n_e2, st, hd, fi)?;
            self.launch_dit_gated_residual_add(&mut d_enc, &d_ff_txt, &d_tl_g, st, hd)?;
        }

        // ── Two downloads: the updated streams. ──
        self.dtoh_sync(&d_hidden, hidden)?;
        self.dtoh_sync(&d_enc, enc)
    }
}
