//! The forward body of a `qwen35` **linear-attention** (Gated-DeltaNet)
//! layer (M-04 / M-15, design §3.2 / §3.3 / §3.4 / §2.3 / §2.4).
//!
//! ```text
//! h     = residual
//! a     = RMSNorm(h, attn_norm)                    width hidden
//! a'    = hook.rotate(a, hidden)                   ONCE: qkv AND gate
//! qkv   = attn_qkv(a')    [conv_dim]               (q | k | v), v tiled
//! z     = attn_gate(a')   [ssm_inner]              tiled v order
//! alpha = ssm_alpha(a)    beta = ssm_beta(a)       <-- the UN-rotated `a`
//! qkv   = causal depthwise conv1d(k = conv_kernel) over the WHOLE stream
//! qkv   = silu(qkv)                                the WHOLE stream
//! q, k  = l2_norm per head over the JOINT q||k view (2 * n_k_heads heads)
//! v     = qkv[2 * n_k*hk ..]                       re-indexed tiled -> grouped
//! o     = gdn(state[rec_slot], q, k, v, gates)     grouped order
//! o     = silu(z) * RMSNorm(o, ssm_norm)           per v-head, z regrouped
//! o'    = hook.rotate(o, ssm_inner)
//! h    += ssm_out(o')
//! ... FFN identical to a full layer ...
//! ```
//!
//! # The three things that are silently wrong if you get them backwards
//!
//! 1. **`ssm_alpha` / `ssm_beta` are not folded.** They are absent from
//!    `prism.hadamard.weight_names`, so they consume the *un-rotated*
//!    activation. Feeding them the rotated copy still produces finite,
//!    plausible logits.
//! 2. **SiLU covers the whole conv output**, all `conv_dim` channels,
//!    *before* the q/k/v split — not the q‖k part only, and not after.
//! 3. **L2-norm's `eps` is a floor on the norm**, `x / max(sqrt(Σx²), eps)`,
//!    not `x / sqrt(Σx² + eps)` — the two agree everywhere except on a
//!    near-zero head (`fork/models/ops.cpp:4176-4212`; kernel `norms.rs`).
//!
//! # Gates are consumed by name, never by position
//!
//! `gdn_step_f32`/`gdn_prefill_f32` take `(dt_bias, a_neg)` while
//! `gdn_step`/`gdn_chunk` take `(a_neg, dt_bias)`; both are `&[f32]` of the
//! same length, so a transposition compiles. This module never calls
//! either: it builds the gate set through
//! [`crate::hybrid::weights::GdnGateWeights::gates`], whose struct-field
//! construction cannot be transposed, and calls
//! [`oxibonsai_kernels::gated_delta_net_chunk::gdn_prefill_with`] directly.

use oxibonsai_kernels::gated_delta_net::{GdnHeadOrder, GdnPath};
use oxibonsai_kernels::gated_delta_net_chunk::gdn_prefill_with;
use oxibonsai_kernels::norms::{l2_norm_simd, rms_norm_gated_simd};
use oxibonsai_kernels::silu_simd;
use oxibonsai_kernels::ssm_ops::causal_conv1d_k4_prefill;

use crate::error::{ModelError, ModelResult};
use crate::hybrid::block::{HybridBlock, LinearAttnBlock};
use crate::hybrid::forward::{
    folded_input, forward_ffn_chunk, norm_and_rotate, residual_add, scratch_short, ForwardCtx,
};

/// Run one Gated-DeltaNet layer over a chunk of `t_len` tokens.
///
/// The recurrence advances exactly once per token, in order, whatever
/// `t_len` is: `gdn_prefill_with` with `t_len == 1` *is* `gdn_step_with`
/// (design §8.2 G5).
///
/// # Errors
///
/// [`ModelError::ShapeInvariant`] when `index` is not a linear-attention
/// layer or a geometry relationship does not hold, and anything the
/// projections, the conv, the recurrence or the norms return.
pub(crate) fn forward_linear_chunk(
    ctx: &mut ForwardCtx<'_, '_>,
    index: usize,
    t_len: usize,
) -> ModelResult<()> {
    // `&T` is `Copy`: these are independent of the `&mut` borrows taken
    // below (see `block_full`'s note).
    let blocks = ctx.blocks;
    let config = ctx.config;
    let hadamard = ctx.hadamard;
    let vhead_map = ctx.vhead_map;

    let block: &LinearAttnBlock<'_> = blocks
        .get(index)
        .and_then(HybridBlock::as_linear)
        .ok_or_else(|| ModelError::ShapeInvariant {
            tensor: format!("layer {index}"),
            expected: "a Gated-DeltaNet layer".to_string(),
            actual: "not linear attention".to_string(),
        })?;

    let hidden = config.base.hidden_size;
    let intermediate = config.base.intermediate_size;
    let inner = config.ssm_inner_size;
    let conv_dim = config.conv_dim();
    let n_k_heads = config.n_k_heads();
    let n_v_heads = config.n_v_heads();
    let head_k_dim = config.head_k_dim();
    let head_v_dim = config.head_v_dim();
    let qk_width = n_k_heads * head_k_dim;
    let taps = config.ssm_conv_kernel;
    let eps = config.base.rms_norm_eps;
    let rec_slot = block.rec_slot();

    if taps == 0 {
        return Err(ModelError::ShapeInvariant {
            tensor: format!("layer {index}: ssm.conv_kernel"),
            expected: "> 0".to_string(),
            actual: "0".to_string(),
        });
    }
    let history = taps - 1;

    // ── Projections ─────────────────────────────────────────────────────
    {
        let sc = &mut *ctx.scratch;
        norm_and_rotate(
            block.attn_norm(),
            &sc.resid,
            &mut sc.normed,
            &mut sc.rotated,
            hadamard,
            hidden,
            t_len,
        )?;
        let folded = folded_input(&sc.normed, &sc.rotated, hadamard, t_len * hidden)?;
        block.attn_qkv().forward_mat(
            folded,
            sc.qkv
                .get_mut(..t_len * conv_dim)
                .ok_or_else(|| scratch_short("qkv", t_len * conv_dim))?,
            t_len,
        )?;
        block.attn_gate().forward_mat(
            folded,
            sc.z_tiled
                .get_mut(..t_len * inner)
                .ok_or_else(|| scratch_short("z_tiled", t_len * inner))?,
            t_len,
        )?;
    }

    // ── `ssm_alpha` / `ssm_beta` read the UN-rotated activation ─────────
    for t in 0..t_len {
        let sc = &mut *ctx.scratch;
        let lo = t * hidden;
        let a = sc
            .normed
            .get(lo..lo + hidden)
            .ok_or_else(|| scratch_short("normed", lo + hidden))?;
        let head_lo = t * n_v_heads;
        block.ssm_alpha().forward_vec(
            a,
            sc.alpha_tiled
                .get_mut(head_lo..head_lo + n_v_heads)
                .ok_or_else(|| scratch_short("alpha_tiled", head_lo + n_v_heads))?,
        )?;
        block.ssm_beta().forward_vec(
            a,
            sc.beta_tiled
                .get_mut(head_lo..head_lo + n_v_heads)
                .ok_or_else(|| scratch_short("beta_tiled", head_lo + n_v_heads))?,
        )?;
    }

    // ── Depthwise causal conv1d over the whole `qkv` stream ─────────────
    //
    // `causal_conv1d_k4_prefill` wants a channel-major window
    // `[conv_dim][history + t_len]` — the persistent state's `history`
    // taps (oldest first) followed by this chunk's samples — and emits
    // token-major `[t_len][conv_dim]`. The projection above is
    // token-major, so the window is transposed here, and the trailing
    // `history` samples per channel are copied back into the state
    // afterwards (that kernel is a pure function and carries no state).
    {
        let window = history + t_len;
        let conv_state = ctx.recurrent.conv(rec_slot)?;
        if conv_state.len() < conv_dim * history {
            return Err(ModelError::ShapeInvariant {
                tensor: format!("layer {index}: conv state"),
                expected: format!("{} floats", conv_dim * history),
                actual: conv_state.len().to_string(),
            });
        }
        let sc = &mut *ctx.scratch;
        let (qkv, conv_x) = (&sc.qkv, &mut sc.conv_x);
        if conv_x.len() < conv_dim * window {
            return Err(scratch_short("conv_x", conv_dim * window));
        }
        if qkv.len() < t_len * conv_dim {
            return Err(scratch_short("qkv", t_len * conv_dim));
        }
        for (c, row) in conv_x[..conv_dim * window].chunks_mut(window).enumerate() {
            for (h, slot) in row[..history].iter_mut().enumerate() {
                *slot = conv_state.get(c * history + h).copied().unwrap_or(0.0);
            }
            for (t, slot) in row[history..history + t_len].iter_mut().enumerate() {
                *slot = qkv.get(t * conv_dim + c).copied().unwrap_or(0.0);
            }
        }
    }
    {
        let window = history + t_len;
        let sc = &mut *ctx.scratch;
        causal_conv1d_k4_prefill(
            sc.conv_x
                .get(..conv_dim * window)
                .ok_or_else(|| scratch_short("conv_x", conv_dim * window))?,
            block.ssm_conv1d(),
            sc.conv_out
                .get_mut(..t_len * conv_dim)
                .ok_or_else(|| scratch_short("conv_out", t_len * conv_dim))?,
            t_len,
            conv_dim,
        )
        .map_err(ModelError::Kernel)?;
    }
    {
        // Carry the window forward: the last `history` samples fed in.
        // `conv_x` and the recurrent cache are disjoint fields of `ctx`, so
        // this needs no copy of the (up to 21 MB) window.
        let window = history + t_len;
        let conv_x = &ctx.scratch.conv_x;
        let conv_state = ctx.recurrent.conv_mut(rec_slot)?;
        for c in 0..conv_dim {
            let row = c * window + t_len;
            for h in 0..history {
                let carried = conv_x
                    .get(row + h)
                    .copied()
                    .ok_or_else(|| scratch_short("conv_x", row + h + 1))?;
                let slot = conv_state
                    .get_mut(c * history + h)
                    .ok_or_else(|| scratch_short("conv state", c * history + h + 1))?;
                *slot = carried;
            }
        }
    }

    // ── SiLU over the WHOLE conv output, then the q/k/v split ───────────
    {
        let sc = &mut *ctx.scratch;
        let len = t_len * conv_dim;
        // `silu_simd` forbids aliasing, so the activation lands in `qkv`
        // (already consumed by the conv) and is read back from there.
        let (src, dst) = (&sc.conv_out, &mut sc.qkv);
        silu_simd(
            src.get(..len)
                .ok_or_else(|| scratch_short("conv_out", len))?,
            dst.get_mut(..len)
                .ok_or_else(|| scratch_short("qkv", len))?,
        )
        .map_err(ModelError::Kernel)?;
    }

    // ── L2-normalise per head over the JOINT q||k view ──────────────────
    for t in 0..t_len {
        let sc = &mut *ctx.scratch;
        let base = t * conv_dim;
        for head in 0..(2 * n_k_heads) {
            let lo = base + head * head_k_dim;
            let src = sc
                .qkv
                .get(lo..lo + head_k_dim)
                .ok_or_else(|| scratch_short("qkv", lo + head_k_dim))?;
            let out_lo = t * qk_width + (head % n_k_heads) * head_k_dim;
            if head < n_k_heads {
                let dst = sc
                    .gdn_q
                    .get_mut(out_lo..out_lo + head_k_dim)
                    .ok_or_else(|| scratch_short("gdn_q", out_lo + head_k_dim))?;
                l2_norm_simd(src, dst, eps).map_err(ModelError::Kernel)?;
            } else {
                let dst = sc
                    .gdn_k
                    .get_mut(out_lo..out_lo + head_k_dim)
                    .ok_or_else(|| scratch_short("gdn_k", out_lo + head_k_dim))?;
                l2_norm_simd(src, dst, eps).map_err(ModelError::Kernel)?;
            }
        }
    }

    // ── Re-index the v-indexed activations tiled -> grouped (§3.3) ──────
    for t in 0..t_len {
        let sc = &mut *ctx.scratch;
        let v_lo = t * conv_dim + 2 * qk_width;
        let out_lo = t * inner;
        {
            let (qkv, gdn_v) = (&sc.qkv, &mut sc.gdn_v);
            vhead_map.gather_grouped(
                qkv.get(v_lo..v_lo + inner)
                    .ok_or_else(|| scratch_short("qkv", v_lo + inner))?,
                head_v_dim,
                gdn_v
                    .get_mut(out_lo..out_lo + inner)
                    .ok_or_else(|| scratch_short("gdn_v", out_lo + inner))?,
            )?;
        }
        {
            let (z_tiled, z_grouped) = (&sc.z_tiled, &mut sc.z_grouped);
            vhead_map.gather_grouped(
                z_tiled
                    .get(out_lo..out_lo + inner)
                    .ok_or_else(|| scratch_short("z_tiled", out_lo + inner))?,
                head_v_dim,
                z_grouped
                    .get_mut(out_lo..out_lo + inner)
                    .ok_or_else(|| scratch_short("z_grouped", out_lo + inner))?,
            )?;
        }
        let head_lo = t * n_v_heads;
        {
            let (alpha_tiled, alpha) = (&sc.alpha_tiled, &mut sc.alpha);
            vhead_map.gather_grouped_scalar(
                alpha_tiled
                    .get(head_lo..head_lo + n_v_heads)
                    .ok_or_else(|| scratch_short("alpha_tiled", head_lo + n_v_heads))?,
                alpha
                    .get_mut(head_lo..head_lo + n_v_heads)
                    .ok_or_else(|| scratch_short("alpha", head_lo + n_v_heads))?,
            )?;
        }
        {
            let (beta_tiled, beta) = (&sc.beta_tiled, &mut sc.beta);
            vhead_map.gather_grouped_scalar(
                beta_tiled
                    .get(head_lo..head_lo + n_v_heads)
                    .ok_or_else(|| scratch_short("beta_tiled", head_lo + n_v_heads))?,
                beta.get_mut(head_lo..head_lo + n_v_heads)
                    .ok_or_else(|| scratch_short("beta", head_lo + n_v_heads))?,
            )?;
        }
    }

    // ── The gated delta rule ────────────────────────────────────────────
    {
        let dims = vhead_map.gdn_dims(head_k_dim, head_v_dim);
        let sc = &mut *ctx.scratch;
        let state = ctx.recurrent.ssm_mut(rec_slot)?;
        let gates = block.gates().gates(
            sc.alpha
                .get(..t_len * n_v_heads)
                .ok_or_else(|| scratch_short("alpha", t_len * n_v_heads))?,
            sc.beta
                .get(..t_len * n_v_heads)
                .ok_or_else(|| scratch_short("beta", t_len * n_v_heads))?,
        );
        gdn_prefill_with(
            state,
            sc.gdn_q
                .get(..t_len * qk_width)
                .ok_or_else(|| scratch_short("gdn_q", t_len * qk_width))?,
            sc.gdn_k
                .get(..t_len * qk_width)
                .ok_or_else(|| scratch_short("gdn_k", t_len * qk_width))?,
            sc.gdn_v
                .get(..t_len * inner)
                .ok_or_else(|| scratch_short("gdn_v", t_len * inner))?,
            &gates,
            sc.gdn_out
                .get_mut(..t_len * inner)
                .ok_or_else(|| scratch_short("gdn_out", t_len * inner))?,
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .map_err(ModelError::Kernel)?;
    }

    // ── Gated RMSNorm per v-head, then the folded output projection ─────
    for t in 0..t_len {
        let sc = &mut *ctx.scratch;
        for head in 0..n_v_heads {
            let lo = t * inner + head * head_v_dim;
            let (gdn_out, z_grouped, gdn_norm) = (&sc.gdn_out, &sc.z_grouped, &mut sc.gdn_norm);
            rms_norm_gated_simd(
                gdn_out
                    .get(lo..lo + head_v_dim)
                    .ok_or_else(|| scratch_short("gdn_out", lo + head_v_dim))?,
                block.ssm_norm().weight(),
                z_grouped
                    .get(lo..lo + head_v_dim)
                    .ok_or_else(|| scratch_short("z_grouped", lo + head_v_dim))?,
                gdn_norm
                    .get_mut(lo..lo + head_v_dim)
                    .ok_or_else(|| scratch_short("gdn_norm", lo + head_v_dim))?,
                eps,
            )
            .map_err(ModelError::Kernel)?;
        }
    }
    {
        let sc = &mut *ctx.scratch;
        let inner_len = t_len * inner;
        let hidden_len = t_len * hidden;
        if let Some(hook) = hadamard {
            // The state — and therefore `gdn_norm` — is already in grouped
            // order, which is the order `ssm_out`'s columns were folded in
            // (`prism.hadamard.gdn_v_grouped = true`), so
            // `rotate_gdn_out` degenerates to a plain row rotation here.
            hook.rotate_rows_in_place(
                sc.gdn_norm
                    .get_mut(..inner_len)
                    .ok_or_else(|| scratch_short("gdn_norm", inner_len))?,
                inner,
                t_len,
            )?;
        }
        block.ssm_out().forward_mat(
            sc.gdn_norm
                .get(..inner_len)
                .ok_or_else(|| scratch_short("gdn_norm", inner_len))?,
            sc.proj
                .get_mut(..hidden_len)
                .ok_or_else(|| scratch_short("proj", hidden_len))?,
            t_len,
        )?;
        let (resid, proj) = (&mut sc.resid, &sc.proj);
        residual_add(
            resid
                .get_mut(..hidden_len)
                .ok_or_else(|| scratch_short("resid", hidden_len))?,
            proj.get(..hidden_len)
                .ok_or_else(|| scratch_short("proj", hidden_len))?,
        );
    }

    // ── SwiGLU FFN (identical in both layer kinds) ──────────────────────
    forward_ffn_chunk(
        ctx,
        block.post_attn_norm(),
        block.ffn_gate(),
        block.ffn_up(),
        block.ffn_down(),
        hidden,
        intermediate,
        t_len,
    )
}
