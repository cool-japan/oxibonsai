//! The two FLUX.2 Klein DiT transformer-block types.
//!
//! - [`DoubleBlock`] — dual-stream (`transformer_blocks.{i}`): separate image
//!   and text streams with joint attention and per-stream modulation + FF.
//! - [`SingleBlock`] — single-stream (`single_transformer_blocks.{j}`): a fused
//!   parallel self-attention + MLP on the concatenated `[txt; img]` stream.
//!
//! Both consume weights resolved from [`crate::DitWeights`] and the shared RoPE
//! tables + modulation parameters produced by [`crate::forward`].

use crate::error::DitResult;
use crate::forward::QkvNorm;
use crate::math::{
    apply_rope_inplace, joint_attention, layer_norm_inplace, modulate_inplace,
    rms_norm_heads_inplace, swiglu, to_heads, RopeTables,
};
use crate::weights::{DitWeights, QuantizedLinear};

/// One `(shift, scale, gate)` modulation triple (each length `hidden`).
#[derive(Debug, Clone)]
pub struct ModTriple {
    /// Additive shift.
    pub shift: Vec<f32>,
    /// Multiplicative `(1 + scale)` factor.
    pub scale: Vec<f32>,
    /// Residual gate.
    pub gate: Vec<f32>,
}

/// The dual-stream modulation for one stream: MSA (attn) + MLP triples.
#[derive(Debug, Clone)]
pub struct DoubleMod {
    /// Attention-path modulation.
    pub msa: ModTriple,
    /// MLP-path modulation.
    pub mlp: ModTriple,
}

/// Add `gate * delta` into `h` (`[rows, dim]`), gate broadcast over rows.
fn gated_residual_add(h: &mut [f32], delta: &[f32], gate: &[f32], rows: usize, dim: usize) {
    for r in 0..rows {
        let hh = &mut h[r * dim..(r + 1) * dim];
        let dd = &delta[r * dim..(r + 1) * dim];
        for i in 0..dim {
            hh[i] += gate[i] * dd[i];
        }
    }
}

/// Resolved per-stream attention projections + QK norms for a double block.
struct DoubleAttnWeights<'a> {
    q: QuantizedLinear<'a>,
    k: QuantizedLinear<'a>,
    v: QuantizedLinear<'a>,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
}

/// A dual-stream (double) transformer block.
pub struct DoubleBlock {
    index: u32,
}

impl DoubleBlock {
    /// Create a handle for double block `index`.
    pub fn new(index: u32) -> Self {
        Self { index }
    }

    /// Run the block, mutating `hidden` (`[seq_img, hidden]`) and `enc`
    /// (`[seq_txt, hidden]`) in place.
    ///
    /// Matches `Flux2TransformerBlock`:
    /// `n_h = mod(LN(h)); n_e = mod(LN(e)); (ai, ae) = attn(n_h, n_e);
    ///  h += gate_msa * ai; e += c_gate_msa * ae;
    ///  h += gate_mlp * ff(mod(LN(h))); e += c_gate_mlp * ff_ctx(mod(LN(e)))`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        weights: &DitWeights,
        hidden: &mut [f32],
        enc: &mut [f32],
        seq_img: usize,
        seq_txt: usize,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        ffn_inner: usize,
        eps: f32,
        rope: &RopeTables,
        mod_img: &DoubleMod,
        mod_txt: &DoubleMod,
    ) -> DitResult<()> {
        // Fused resident double-block GPU forward (default on; `OXI_DIT_FUSED=0`
        // or `OXI_DIT_GPU=0` disables it): one upload / one download per stream.
        // On any GPU error this falls through to the per-op path below with both
        // streams left untouched.
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            if crate::cuda_gpu::dit_gpu_enabled()
                && crate::cuda_gpu::dit_fused_enabled()
                && crate::cuda_gpu::double_block_gpu(
                    weights,
                    self.index,
                    hidden,
                    enc,
                    seq_img,
                    seq_txt,
                    hidden_size,
                    num_heads,
                    head_dim,
                    ffn_inner,
                    eps,
                    rope,
                    mod_img,
                    mod_txt,
                )
                .is_ok()
            {
                return Ok(());
            }
        }

        let p = format!("transformer_blocks.{}", self.index);

        // ── Attention sub-block ──
        // Modulated, LayerNormed copies of each stream.
        let mut n_h = hidden.to_vec();
        layer_norm_inplace(&mut n_h, seq_img, hidden_size, eps);
        modulate_inplace(
            &mut n_h,
            seq_img,
            hidden_size,
            &mod_img.msa.shift,
            &mod_img.msa.scale,
        );
        let mut n_e = enc.to_vec();
        layer_norm_inplace(&mut n_e, seq_txt, hidden_size, eps);
        modulate_inplace(
            &mut n_e,
            seq_txt,
            hidden_size,
            &mod_txt.msa.shift,
            &mod_txt.msa.scale,
        );

        let img_w = DoubleAttnWeights {
            q: weights.quantized_linear(&format!("{p}.attn.to_q"))?,
            k: weights.quantized_linear(&format!("{p}.attn.to_k"))?,
            v: weights.quantized_linear(&format!("{p}.attn.to_v"))?,
            norm_q: weights
                .bf16_tensor(&format!("{p}.attn.norm_q.weight"))?
                .to_f32_vec(),
            norm_k: weights
                .bf16_tensor(&format!("{p}.attn.norm_k.weight"))?
                .to_f32_vec(),
        };
        let txt_w = DoubleAttnWeights {
            q: weights.quantized_linear(&format!("{p}.attn.add_q_proj"))?,
            k: weights.quantized_linear(&format!("{p}.attn.add_k_proj"))?,
            v: weights.quantized_linear(&format!("{p}.attn.add_v_proj"))?,
            norm_q: weights
                .bf16_tensor(&format!("{p}.attn.norm_added_q.weight"))?
                .to_f32_vec(),
            norm_k: weights
                .bf16_tensor(&format!("{p}.attn.norm_added_k.weight"))?
                .to_f32_vec(),
        };

        // Project each stream to head-major q/k/v.
        let (q_img, k_img, v_img) =
            project_qkv(&n_h, &img_w, seq_img, hidden_size, num_heads, head_dim, eps)?;
        let (q_txt, k_txt, v_txt) =
            project_qkv(&n_e, &txt_w, seq_txt, hidden_size, num_heads, head_dim, eps)?;

        // Concat txt-first, then img: [num_heads, seq_txt+seq_img, head_dim].
        let seq_joint = seq_txt + seq_img;
        let mut q = concat_heads(&q_txt, &q_img, num_heads, seq_txt, seq_img, head_dim);
        let mut k = concat_heads(&k_txt, &k_img, num_heads, seq_txt, seq_img, head_dim);
        let v = concat_heads(&v_txt, &v_img, num_heads, seq_txt, seq_img, head_dim);

        // RoPE on q,k (not v).
        apply_rope_inplace(&mut q, num_heads, seq_joint, head_dim, rope)?;
        apply_rope_inplace(&mut k, num_heads, seq_joint, head_dim, rope)?;

        // Out projections (resolved up-front; shared by the fused + unfused paths).
        let to_out = weights.quantized_linear(&format!("{p}.attn.to_out.0"))?;
        let to_add_out = weights.quantized_linear(&format!("{p}.attn.to_add_out"))?;

        let (img_attn, enc_attn) = double_attn_to_out(
            &q,
            &k,
            &v,
            &to_out,
            &to_add_out,
            num_heads,
            seq_txt,
            seq_img,
            head_dim,
            hidden_size,
        )?;

        // Gated residual.
        gated_residual_add(hidden, &img_attn, &mod_img.msa.gate, seq_img, hidden_size);
        gated_residual_add(enc, &enc_attn, &mod_txt.msa.gate, seq_txt, hidden_size);

        // ── Feed-forward sub-block ──
        // image stream
        let mut n_h2 = hidden.to_vec();
        layer_norm_inplace(&mut n_h2, seq_img, hidden_size, eps);
        modulate_inplace(
            &mut n_h2,
            seq_img,
            hidden_size,
            &mod_img.mlp.shift,
            &mod_img.mlp.scale,
        );
        let ff_img = feed_forward(
            weights,
            &format!("{p}.ff"),
            &n_h2,
            seq_img,
            hidden_size,
            ffn_inner,
        )?;
        gated_residual_add(hidden, &ff_img, &mod_img.mlp.gate, seq_img, hidden_size);

        // text stream
        let mut n_e2 = enc.to_vec();
        layer_norm_inplace(&mut n_e2, seq_txt, hidden_size, eps);
        modulate_inplace(
            &mut n_e2,
            seq_txt,
            hidden_size,
            &mod_txt.mlp.shift,
            &mod_txt.mlp.scale,
        );
        let ff_txt = feed_forward(
            weights,
            &format!("{p}.ff_context"),
            &n_e2,
            seq_txt,
            hidden_size,
            ffn_inner,
        )?;
        gated_residual_add(enc, &ff_txt, &mod_txt.mlp.gate, seq_txt, hidden_size);

        Ok(())
    }
}

/// Project the shared stream `x` `[seq, hidden]` through the three Q/K/V
/// projections `(w.q, w.k, w.v)` — which all read the *same* input — returning
/// the token-major `(q_tok, k_tok, v_tok)`, each `[seq, hidden]`.
///
/// Runs the three [`crate::math::ternary_matmul`] calls in sequence; each one
/// independently dispatches GPU-or-CPU via its own gate.
fn project_qkv_tokens(
    x: &[f32],
    w: &DoubleAttnWeights,
    seq: usize,
    hidden: usize,
) -> DitResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let q_tok = crate::math::ternary_matmul(w.q.blocks, x, seq, hidden, hidden)?;
    let k_tok = crate::math::ternary_matmul(w.k.blocks, x, seq, hidden, hidden)?;
    let v_tok = crate::math::ternary_matmul(w.v.blocks, x, seq, hidden, hidden)?;
    Ok((q_tok, k_tok, v_tok))
}

/// Run the double-block joint attention and both `to_out` projections,
/// returning `(img_attn [seq_img, hidden], enc_attn [seq_txt, hidden])`.
///
/// `q`/`k`/`v` are head-major `[num_heads, seq_txt+seq_img, head_dim]` (RoPE
/// applied). Runs the explicit `joint_attention` then the two `ternary_matmul`
/// `to_out` projections in sequence — each of which independently dispatches
/// GPU-or-CPU via its own gate.
#[allow(clippy::too_many_arguments)]
fn double_attn_to_out(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    to_out: &QuantizedLinear<'_>,
    to_add_out: &QuantizedLinear<'_>,
    num_heads: usize,
    seq_txt: usize,
    seq_img: usize,
    head_dim: usize,
    hidden_size: usize,
) -> DitResult<(Vec<f32>, Vec<f32>)> {
    let seq_joint = seq_txt + seq_img;

    // Explicit attention then the two to_out projections (each of which
    // independently dispatches GPU-or-CPU via its own gate).
    let attn = joint_attention(q, k, v, num_heads, seq_joint, head_dim)?;
    let enc_attn_in = &attn[..seq_txt * hidden_size];
    let img_attn_in = &attn[seq_txt * hidden_size..];
    let img_attn = crate::math::ternary_matmul(
        to_out.blocks,
        img_attn_in,
        seq_img,
        to_out.out_features as usize,
        to_out.in_features as usize,
    )?;
    let enc_attn = crate::math::ternary_matmul(
        to_add_out.blocks,
        enc_attn_in,
        seq_txt,
        to_add_out.out_features as usize,
        to_add_out.in_features as usize,
    )?;
    Ok((img_attn, enc_attn))
}

/// Project one stream `[seq, hidden]` → head-major q/k/v with QK-RMSNorm.
fn project_qkv(
    x: &[f32],
    w: &DoubleAttnWeights,
    seq: usize,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> DitResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let (q_tok, k_tok, v_tok) = project_qkv_tokens(x, w, seq, hidden)?;
    // → head-major [num_heads, seq, head_dim]
    let mut q = to_heads(&q_tok, seq, num_heads, head_dim);
    let mut k = to_heads(&k_tok, seq, num_heads, head_dim);
    let v = to_heads(&v_tok, seq, num_heads, head_dim);
    // QK-RMSNorm over head_dim, per (head, token).
    rms_norm_heads_inplace(&mut q, num_heads * seq, head_dim, &w.norm_q, eps);
    rms_norm_heads_inplace(&mut k, num_heads * seq, head_dim, &w.norm_k, eps);
    Ok((q, k, v))
}

/// Concatenate two head-major buffers along the sequence axis (a-first):
/// `[H, sa, d] ++ [H, sb, d]` → `[H, sa + sb, d]`.
fn concat_heads(
    a: &[f32],
    b: &[f32],
    num_heads: usize,
    sa: usize,
    sb: usize,
    head_dim: usize,
) -> Vec<f32> {
    let sj = sa + sb;
    let mut out = vec![0.0f32; num_heads * sj * head_dim];
    for h in 0..num_heads {
        let dst_base = h * sj * head_dim;
        let a_src = &a[h * sa * head_dim..(h + 1) * sa * head_dim];
        out[dst_base..dst_base + sa * head_dim].copy_from_slice(a_src);
        let b_src = &b[h * sb * head_dim..(h + 1) * sb * head_dim];
        out[dst_base + sa * head_dim..dst_base + sj * head_dim].copy_from_slice(b_src);
    }
    out
}

/// Flux2FeedForward: `linear_in (hidden→2*ffn_inner) → SwiGLU → linear_out
/// (ffn_inner→hidden)`. `prefix` is the module path (`...ff` / `...ff_context`).
fn feed_forward(
    weights: &DitWeights,
    prefix: &str,
    x: &[f32],
    seq: usize,
    hidden: usize,
    ffn_inner: usize,
) -> DitResult<Vec<f32>> {
    let lin_in = weights.quantized_linear(&format!("{prefix}.linear_in"))?;
    let proj = crate::math::ternary_matmul(
        lin_in.blocks,
        x,
        seq,
        lin_in.out_features as usize,
        lin_in.in_features as usize,
    )?;
    // proj is [seq, 2*ffn_inner]; SwiGLU → [seq, ffn_inner].
    let gated = swiglu(&proj, seq, ffn_inner);
    let lin_out = weights.quantized_linear(&format!("{prefix}.linear_out"))?;
    let out = crate::math::ternary_matmul(
        lin_out.blocks,
        &gated,
        seq,
        lin_out.out_features as usize,
        lin_out.in_features as usize,
    )?;
    debug_assert_eq!(out.len(), seq * hidden);
    Ok(out)
}

/// A single-stream (single) transformer block.
pub struct SingleBlock {
    index: u32,
}

impl SingleBlock {
    /// Create a handle for single block `index`.
    pub fn new(index: u32) -> Self {
        Self { index }
    }

    /// Run the block in place over the joint stream `h` (`[seq, hidden]`).
    ///
    /// Matches `Flux2SingleTransformerBlock` + `Flux2ParallelSelfAttention`:
    /// `n = mod(LN(h)); proj = to_qkv_mlp_proj(n);
    ///  (qkv, mlp) = split(proj, 3*hidden); q,k,v = split(qkv);
    ///  q,k = rope(qknorm(q,k)); a = attn(q,k,v);
    ///  out = to_out(concat(a, swiglu(mlp))); h += gate * out`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        weights: &DitWeights,
        h: &mut [f32],
        seq: usize,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        ffn_inner: usize,
        eps: f32,
        rope: &RopeTables,
        mod_single: &ModTriple,
        norms: &QkvNorm,
    ) -> DitResult<()> {
        // Fused resident single-block GPU forward (default on; `OXI_DIT_FUSED=0`
        // or `OXI_DIT_GPU=0` disables it): one upload / one download for the
        // whole block. On any GPU error this falls through to the per-op path
        // below with `h` left untouched.
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            if crate::cuda_gpu::dit_gpu_enabled()
                && crate::cuda_gpu::dit_fused_enabled()
                && crate::cuda_gpu::single_block_gpu(
                    weights,
                    self.index,
                    h,
                    seq,
                    hidden_size,
                    num_heads,
                    head_dim,
                    ffn_inner,
                    eps,
                    rope,
                    mod_single,
                    norms,
                )
                .is_ok()
            {
                return Ok(());
            }
        }

        let p = format!("single_transformer_blocks.{}", self.index);

        let mut n = h.to_vec();
        layer_norm_inplace(&mut n, seq, hidden_size, eps);
        modulate_inplace(
            &mut n,
            seq,
            hidden_size,
            &mod_single.shift,
            &mod_single.scale,
        );

        // Fused projection: [seq, hidden] → [seq, 3*hidden + 2*ffn_inner].
        let proj_w = weights.quantized_linear(&format!("{p}.attn.to_qkv_mlp_proj"))?;
        let proj = crate::math::ternary_matmul(
            proj_w.blocks,
            &n,
            seq,
            proj_w.out_features as usize,
            proj_w.in_features as usize,
        )?;
        let proj_out = proj_w.out_features as usize;
        let qkv_width = 3 * hidden_size;
        let mlp_width = proj_out - qkv_width; // = 2 * ffn_inner

        // Split per row into qkv (3*hidden) and mlp (2*ffn_inner), then build
        // head-major q/k/v from the qkv slab.
        let mut q_tok = vec![0.0f32; seq * hidden_size];
        let mut k_tok = vec![0.0f32; seq * hidden_size];
        let mut v_tok = vec![0.0f32; seq * hidden_size];
        let mut mlp = vec![0.0f32; seq * mlp_width];
        for t in 0..seq {
            let row = &proj[t * proj_out..(t + 1) * proj_out];
            q_tok[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(&row[..hidden_size]);
            k_tok[t * hidden_size..(t + 1) * hidden_size]
                .copy_from_slice(&row[hidden_size..2 * hidden_size]);
            v_tok[t * hidden_size..(t + 1) * hidden_size]
                .copy_from_slice(&row[2 * hidden_size..3 * hidden_size]);
            mlp[t * mlp_width..(t + 1) * mlp_width].copy_from_slice(&row[qkv_width..]);
        }

        let mut q = to_heads(&q_tok, seq, num_heads, head_dim);
        let mut k = to_heads(&k_tok, seq, num_heads, head_dim);
        let v = to_heads(&v_tok, seq, num_heads, head_dim);
        rms_norm_heads_inplace(&mut q, num_heads * seq, head_dim, &norms.q, eps);
        rms_norm_heads_inplace(&mut k, num_heads * seq, head_dim, &norms.k, eps);
        apply_rope_inplace(&mut q, num_heads, seq, head_dim, rope)?;
        apply_rope_inplace(&mut k, num_heads, seq, head_dim, rope)?;

        let attn = joint_attention(&q, &k, &v, num_heads, seq, head_dim)?;
        // SwiGLU on the mlp slab: [seq, 2*ffn_inner] → [seq, ffn_inner].
        let gated = swiglu(&mlp, seq, ffn_inner);

        // Concat [attn, gated] per row → [seq, hidden + ffn_inner], then to_out.
        let cat_width = hidden_size + ffn_inner;
        let mut cat = vec![0.0f32; seq * cat_width];
        for t in 0..seq {
            cat[t * cat_width..t * cat_width + hidden_size]
                .copy_from_slice(&attn[t * hidden_size..(t + 1) * hidden_size]);
            cat[t * cat_width + hidden_size..(t + 1) * cat_width]
                .copy_from_slice(&gated[t * ffn_inner..(t + 1) * ffn_inner]);
        }
        let to_out = weights.quantized_linear(&format!("{p}.attn.to_out"))?;
        let out = crate::math::ternary_matmul(
            to_out.blocks,
            &cat,
            seq,
            to_out.out_features as usize,
            to_out.in_features as usize,
        )?;

        gated_residual_add(h, &out, &mod_single.gate, seq, hidden_size);
        Ok(())
    }
}
