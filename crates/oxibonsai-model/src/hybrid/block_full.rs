//! The forward body of a `qwen35` **full-attention** layer (M-16, design
//! §3.2 / §3.4 / §2.6).
//!
//! ```text
//! h   = residual
//! a   = RMSNorm(h, attn_norm)                      width hidden
//! a'  = hook.rotate(a, hidden)                     ONCE, shared by q, k, v
//! qg  = attn_q(a')            [2 * n_head * head_dim], [q|gate] per head
//! q   = RMSNorm(q_h, attn_q_norm); k = RMSNorm(k_h, attn_k_norm)
//! q,k = partial NeoX RoPE (n_rot of head_dim, split-half pairs (j, j+n_rot/2))
//! o   = GQA(q, k, v, kv_cache[kv_slot], 1/sqrt(head_dim))
//! o   = o * sigmoid(gate)                          gate is NOT normed, NOT RoPE'd
//! o'  = hook.rotate(o, n_head * head_dim)
//! h  += attn_output(o')
//! f   = RMSNorm(h, post_attention_norm)
//! f'  = hook.rotate(f, hidden)                     ONCE, shared by gate, up
//! m   = silu(ffn_gate(f')) * ffn_up(f')
//! m'  = hook.rotate(m, intermediate)
//! h  += ffn_down(m')
//! ```
//!
//! # `kv_slot`, never `layer_idx`
//!
//! The hybrid KV cache has one slot per *full* layer (16 for the 27B), not
//! one per stack layer (64). Storing at `layer_idx` would be rejected by
//! the cache's store validation and — through the lossy
//! `store_key_lossy`/`store_value_lossy` forms — silently dropped; reading at `layer_idx` would return
//! another layer's history. Every cache access below goes through
//! `block.kv_slot()`.
//!
//! # Reading the `f16` cache
//!
//! Attention reads the slot through `KvCache::attend_group`, once per
//! (token, kv-head): the `heads_per_group` query heads sharing a KV head walk
//! its history together, each `f16` row widened once on the stack. Nothing
//! is copied out of the cache, and the result is bit-identical to widening
//! the history first and running `fused_attention_head_contiguous` per head
//! (which is what this body used to do, through `keys_for_owned`).

use oxibonsai_kernels::norms::sigmoid_mul_simd;
use oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd;

use crate::error::{ModelError, ModelResult};
use crate::hybrid::block::{FullAttnBlock, HybridBlock};
use crate::hybrid::forward::{
    folded_input, forward_ffn_chunk, norm_and_rotate, residual_add, scratch_short, ForwardCtx,
};

/// Run one full-attention layer over a chunk of `t_len` tokens whose first
/// token sits at absolute position `start_pos`.
///
/// `ctx.scratch.resid` is both the input and the output: the two residual
/// adds happen in place, exactly as they would token by token.
///
/// # Errors
///
/// [`ModelError::ShapeInvariant`] when `index` is not a full-attention
/// layer or a geometry relationship does not hold, and anything the
/// projections, the KV cache, the RoPE table or the attention kernel
/// return.
pub(crate) fn forward_full_chunk(
    ctx: &mut ForwardCtx<'_, '_>,
    index: usize,
    t_len: usize,
    start_pos: usize,
) -> ModelResult<()> {
    // Copy the shared references out of `ctx` first: `&T` is `Copy`, so
    // these are independent of the `&mut` borrows of `ctx.scratch` /
    // `ctx.kv` taken below.
    let blocks = ctx.blocks;
    let config = ctx.config;
    let hadamard = ctx.hadamard;
    let rope = ctx.rope;

    let block: &FullAttnBlock<'_> = blocks
        .get(index)
        .and_then(HybridBlock::as_full)
        .ok_or_else(|| ModelError::ShapeInvariant {
            tensor: format!("layer {index}"),
            expected: "a full-attention layer".to_string(),
            actual: "not full attention".to_string(),
        })?;

    let hidden = config.base.hidden_size;
    let intermediate = config.base.intermediate_size;
    let head_dim = config.base.head_dim;
    let n_heads = config.base.num_attention_heads;
    let n_kv_heads = config.base.num_kv_heads;
    let heads_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let n_rot = config.rope_dimension_count;
    let kv_slot = block.kv_slot();

    if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(ModelError::ShapeInvariant {
            tensor: format!("layer {index}: head_count / head_count_kv"),
            expected: "head_count a whole multiple of head_count_kv".to_string(),
            actual: format!("{n_heads} / {n_kv_heads}"),
        });
    }
    let heads_per_group = n_heads / n_kv_heads;

    // ── Attention block ─────────────────────────────────────────────────
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
        block.attn_q().forward_mat(
            folded,
            sc.q_all
                .get_mut(..t_len * heads_width * 2)
                .ok_or_else(|| scratch_short("q_all", t_len * heads_width * 2))?,
            t_len,
        )?;
        block.attn_k().forward_mat(
            folded,
            sc.k.get_mut(..t_len * kv_width)
                .ok_or_else(|| scratch_short("k", t_len * kv_width))?,
            t_len,
        )?;
        block.attn_v().forward_mat(
            folded,
            sc.v.get_mut(..t_len * kv_width)
                .ok_or_else(|| scratch_short("v", t_len * kv_width))?,
            t_len,
        )?;
    }

    // ── De-interleave q|gate, per-head norms, RoPE, KV store ────────────
    for t in 0..t_len {
        let pos = start_pos + t;
        let (cos, sin) = rope.angles(pos)?;
        let sc = &mut *ctx.scratch;

        for h in 0..n_heads {
            // `attn_q` is `[q(head_dim) | gate(head_dim)]` per head, so the
            // stride is `2 * head_dim` and the gate sits at `+ head_dim`.
            let q_lo = t * heads_width * 2 + h * head_dim * 2;
            let gate_lo = q_lo + head_dim;
            let out_lo = t * heads_width + h * head_dim;

            let q_raw = sc
                .q_all
                .get(q_lo..q_lo + head_dim)
                .ok_or_else(|| scratch_short("q_all", q_lo + head_dim))?;
            let tmp = sc
                .head_tmp
                .get_mut(..head_dim)
                .ok_or_else(|| scratch_short("head_tmp", head_dim))?;
            block.attn_q_norm().forward(q_raw, tmp)?;

            let gate_raw = sc
                .q_all
                .get(gate_lo..gate_lo + head_dim)
                .ok_or_else(|| scratch_short("q_all", gate_lo + head_dim))?;
            sc.gate
                .get_mut(out_lo..out_lo + head_dim)
                .ok_or_else(|| scratch_short("gate", out_lo + head_dim))?
                .copy_from_slice(gate_raw);

            rope_partial_splithalf_simd(
                sc.head_tmp
                    .get(..head_dim)
                    .ok_or_else(|| scratch_short("head_tmp", head_dim))?,
                sc.q.get_mut(out_lo..out_lo + head_dim)
                    .ok_or_else(|| scratch_short("q", out_lo + head_dim))?,
                head_dim,
                n_rot,
                cos,
                sin,
            )
            .map_err(ModelError::Kernel)?;
        }

        for kh in 0..n_kv_heads {
            let lo = t * kv_width + kh * head_dim;
            let k_raw =
                sc.k.get(lo..lo + head_dim)
                    .ok_or_else(|| scratch_short("k", lo + head_dim))?;
            let tmp = sc
                .head_tmp
                .get_mut(..head_dim)
                .ok_or_else(|| scratch_short("head_tmp", head_dim))?;
            block.attn_k_norm().forward(k_raw, tmp)?;
            rope_partial_splithalf_simd(
                sc.head_tmp
                    .get(..head_dim)
                    .ok_or_else(|| scratch_short("head_tmp", head_dim))?,
                sc.k.get_mut(lo..lo + head_dim)
                    .ok_or_else(|| scratch_short("k", lo + head_dim))?,
                head_dim,
                n_rot,
                cos,
                sin,
            )
            .map_err(ModelError::Kernel)?;

            // `try_store_*` (not the lossy forwarders): against a 16-slot
            // sparse cache a rejected store is a whole layer's history lost.
            let key =
                sc.k.get(lo..lo + head_dim)
                    .ok_or_else(|| scratch_short("k", lo + head_dim))?;
            ctx.kv.try_store_key(kv_slot, kh, pos, key)?;
            let value =
                sc.v.get(lo..lo + head_dim)
                    .ok_or_else(|| scratch_short("v", lo + head_dim))?;
            ctx.kv.try_store_value(kv_slot, kh, pos, value)?;
        }
    }

    // ── GQA attention over this layer's slot ────────────────────────────
    //
    // One `attend_group` per (token, kv-head): the `heads_per_group` query
    // heads that share the KV head walk its history together, read in place
    // from the `f16` cache (each row widened once, on the stack) — no
    // per-token history copy, which at chunk 512 / ctx 8192 used to move
    // ~17 GB per full-attention layer per chunk.
    let group_width = heads_per_group * head_dim;
    for t in 0..t_len {
        let seq_len = start_pos + t + 1;
        let sc = &mut *ctx.scratch;
        for kh in 0..n_kv_heads {
            let lo = t * heads_width + kh * group_width;
            let queries =
                sc.q.get(lo..lo + group_width)
                    .ok_or_else(|| scratch_short("q", lo + group_width))?;
            let out = sc
                .attn
                .get_mut(lo..lo + group_width)
                .ok_or_else(|| scratch_short("attn", lo + group_width))?;
            ctx.kv
                .attend_group(kv_slot, kh, seq_len, queries, out)
                .map_err(|e| match e {
                    ModelError::ShapeInvariant { .. } | ModelError::SequenceTooLong { .. } => {
                        ModelError::ShapeInvariant {
                            tensor: format!("layer {index}: kv slot {kv_slot} head {kh}"),
                            expected: format!("{seq_len} positions of stored history"),
                            actual: e.to_string(),
                        }
                    }
                    other => other,
                })?;
        }
    }

    // ── Sigmoid gate, rotate, output projection, residual ───────────────
    {
        let sc = &mut *ctx.scratch;
        let width = t_len * heads_width;
        sigmoid_mul_simd(
            sc.attn
                .get(..width)
                .ok_or_else(|| scratch_short("attn", width))?,
            sc.gate
                .get(..width)
                .ok_or_else(|| scratch_short("gate", width))?,
            sc.attn_gated
                .get_mut(..width)
                .ok_or_else(|| scratch_short("attn_gated", width))?,
        )
        .map_err(ModelError::Kernel)?;
        if let Some(hook) = hadamard {
            hook.rotate_rows_in_place(
                sc.attn_gated
                    .get_mut(..width)
                    .ok_or_else(|| scratch_short("attn_gated", width))?,
                heads_width,
                t_len,
            )?;
        }
        block.attn_output().forward_mat(
            sc.attn_gated
                .get(..width)
                .ok_or_else(|| scratch_short("attn_gated", width))?,
            sc.proj
                .get_mut(..t_len * hidden)
                .ok_or_else(|| scratch_short("proj", t_len * hidden))?,
            t_len,
        )?;
        residual_add(&mut sc.resid[..t_len * hidden], &sc.proj[..t_len * hidden]);
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
