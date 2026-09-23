//! The `qwen35` forward / prefill driver (design §3.10) and the model seam
//! the runtime dispatches through (B2-11).
//!
//! # One body, parameterised by batch
//!
//! [`run_chunk`] is the *only* forward body in this module tree. A decode
//! step is `run_chunk` with `t_len == 1`; a chunked prefill is the same
//! call with `t_len == chunk`. Nothing is duplicated between the two, so
//! design §8.2's G5 ("prefill(T) == T sequential decode steps") is a
//! property of the *shape* of the code rather than of two implementations
//! agreeing by luck:
//!
//! * every projection is one `forward_mat(.., t_len)` call — at `t_len == 1`
//!   the dispatcher's batched entry point degenerates to its own GEMV;
//! * the Gated-DeltaNet recurrence goes through
//!   `oxibonsai_kernels::gdn_prefill_with`, whose `t_len == 1` case *is*
//!   `gdn_step_with` (that function literally forwards to it), so the state
//!   advances exactly once per token, in order, either way;
//! * the depthwise conv goes through
//!   `oxibonsai_kernels::causal_conv1d_k4_prefill`, which that kernel's own
//!   docs pin bitwise to `causal_conv1d_k4_decode` for the same window;
//! * the KV slot is written and read per token, at the absolute position,
//!   so a full-attention layer sees exactly the history it would have seen
//!   one token at a time.
//!
//! # Rotation
//!
//! A folded checkpoint's activations are rotated once per *activation*, not
//! once per matmul (design §3.4). The driver and the two block bodies
//! therefore keep a rotated copy beside the plain one and hand the rotated
//! slice to every folded projection that consumes it, while `ssm_alpha` /
//! `ssm_beta` — the two projections that are **not** in
//! `prism.hadamard.weight_names` — keep reading the *un-rotated* activation.
//!
//! # Layer dumps (design §8.2 G11)
//!
//! [`LayerDump`] records the residual stream after each block, so the CPU
//! path can be used as the reference for the later Metal parity gate
//! without re-running anything.

use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_kernels::rope_mrope::partial_rope_build_table;

use crate::error::{ModelError, ModelResult};
use crate::hybrid::block::HybridBlock;
use crate::hybrid::hadamard::HadamardHook;
use crate::hybrid::recurrent_cache::RecurrentCache;
use crate::hybrid::vhead_map::VHeadMap;
use crate::hybrid::weights::HybridEmbedding;
use crate::kv_cache::KvCache;
use crate::layers::linear::LinearLayer;
use crate::layers::rms_norm::RmsNorm;

/// Tokens processed per prefill chunk unless the caller says otherwise
/// (design §3.10: 512 × 17408 f32 ≈ 35 MB of activation peak).
pub const DEFAULT_PREFILL_CHUNK: usize = 512;

// ─────────────────────────────────────────────────────────────────────────
//  RoPE tables
// ─────────────────────────────────────────────────────────────────────────

/// Precomputed partial-NeoX RoPE angles for every position a sequence can
/// reach (design §2.5).
///
/// Text-only M-RoPE is provably identical to standard NeoX RoPE over the
/// first `n_rot` of `head_dim` when all three position axes carry the token
/// index, so one `(cos, sin)` pair per position is enough; the table is
/// built through [`partial_rope_build_table`], i.e. through the very
/// `mrope_build_tables` the vision path will use, rather than through a
/// second angle formula that could drift from it.
#[derive(Debug, Clone)]
pub struct RopeTables {
    cos: Vec<f32>,
    sin: Vec<f32>,
    n_rot: usize,
    half: usize,
    max_pos: usize,
}

impl RopeTables {
    /// Build the table for positions `0..max_pos`.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] for an odd `n_rot` or a zero
    /// `max_pos`, or a position that does not fit in the `i32` the kernel's
    /// angle builder takes; [`ModelError::Kernel`] from the builder itself.
    pub fn new(n_rot: usize, max_pos: usize, freq_base: f32) -> ModelResult<Self> {
        if !n_rot.is_multiple_of(2) {
            return Err(ModelError::ShapeInvariant {
                tensor: "rope.dimension_count".to_string(),
                expected: "even".to_string(),
                actual: n_rot.to_string(),
            });
        }
        if max_pos == 0 {
            return Err(ModelError::ShapeInvariant {
                tensor: "rope table length".to_string(),
                expected: "> 0".to_string(),
                actual: "0".to_string(),
            });
        }
        let half = n_rot / 2;
        let mut cos = vec![0.0f32; max_pos * half];
        let mut sin = vec![0.0f32; max_pos * half];
        for pos in 0..max_pos {
            let p = i32::try_from(pos).map_err(|_| ModelError::ShapeInvariant {
                tensor: "rope position".to_string(),
                expected: "representable as i32".to_string(),
                actual: pos.to_string(),
            })?;
            let lo = pos * half;
            partial_rope_build_table(
                p,
                n_rot,
                freq_base,
                &mut cos[lo..lo + half],
                &mut sin[lo..lo + half],
            )
            .map_err(ModelError::Kernel)?;
        }
        Ok(Self {
            cos,
            sin,
            n_rot,
            half,
            max_pos,
        })
    }

    /// Rotated dimensions per head (`n_rot`).
    #[inline]
    #[must_use]
    pub fn n_rot(&self) -> usize {
        self.n_rot
    }

    /// Highest position + 1 this table covers.
    #[inline]
    #[must_use]
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// The `(cos, sin)` pair for `pos`, each `n_rot / 2` long.
    ///
    /// # Errors
    ///
    /// [`ModelError::PositionOutOfRange`] when `pos >= max_pos`.
    pub fn angles(&self, pos: usize) -> ModelResult<(&[f32], &[f32])> {
        if pos >= self.max_pos {
            return Err(ModelError::PositionOutOfRange {
                pos,
                max: self.max_pos,
            });
        }
        let lo = pos * self.half;
        let hi = lo + self.half;
        match (self.cos.get(lo..hi), self.sin.get(lo..hi)) {
            (Some(c), Some(s)) => Ok((c, s)),
            // Unreachable: the table is `max_pos * half` long and `pos <
            // max_pos` was just checked. Reported rather than `unwrap`ped.
            _ => Err(ModelError::PositionOutOfRange {
                pos,
                max: self.max_pos,
            }),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Per-layer activation dump (design §8.2 G11)
// ─────────────────────────────────────────────────────────────────────────

/// The residual stream after each block, recorded for the later Metal
/// parity gate (design §8.2 G11: "record a token list and a per-layer dump
/// so B2-15 can diff against it").
///
/// A dump holds one `[t_len][hidden]` row set per layer plus the final
/// post-`output_norm` hidden state, for the tokens of the call it was
/// attached to.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerDump {
    /// The tokens this dump was produced from.
    pub tokens: Vec<u32>,
    /// Absolute position of `tokens[0]`.
    pub start_pos: usize,
    /// Hidden width of every recorded row.
    pub hidden: usize,
    /// The embedding output *after* the inverse rotation, `[t][hidden]`.
    pub embedding: Vec<f32>,
    /// `layers[i]` is the residual stream after block `i`, `[t][hidden]`.
    pub layers: Vec<Vec<f32>>,
    /// The post-`output_norm` hidden state of the last token, `[hidden]`.
    pub final_norm: Vec<f32>,
}

impl LayerDump {
    /// Row `t` of layer `layer`'s record, or `None` when out of range.
    #[must_use]
    pub fn layer_row(&self, layer: usize, t: usize) -> Option<&[f32]> {
        let rows = self.layers.get(layer)?;
        rows.get(t * self.hidden..(t + 1) * self.hidden)
    }

    /// Number of tokens recorded.
    #[must_use]
    pub fn t_len(&self) -> usize {
        self.tokens.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Chunk-wide activation scratch
// ─────────────────────────────────────────────────────────────────────────

/// Every activation buffer one chunk of tokens needs, allocated once and
/// grown only when a larger chunk arrives.
///
/// This replaces per-token allocation *and* the per-layer scratch a decode
/// path would otherwise need: one set of chunk-wide buffers serves all 64
/// layers, so the 27B holds ~35 MB at chunk 512 rather than 64 per-layer
/// copies of the same shapes.
#[derive(Debug, Clone, Default)]
pub struct HybridScratch {
    capacity: usize,
    hidden: usize,
    intermediate: usize,
    heads_width: usize,
    kv_width: usize,
    conv_dim: usize,
    inner: usize,
    qk_width: usize,
    n_v_heads: usize,
    conv_taps: usize,
    /// `[t][hidden]` residual stream.
    pub(crate) resid: Vec<f32>,
    /// `[t][hidden]` normalised activation (un-rotated).
    pub(crate) normed: Vec<f32>,
    /// `[t][hidden]` rotated copy of `normed`.
    pub(crate) rotated: Vec<f32>,
    /// `[t][hidden]` projection output (`attn_output` / `ssm_out` /
    /// `ffn_down`).
    pub(crate) proj: Vec<f32>,
    /// `[t][intermediate]` SwiGLU gate branch.
    pub(crate) ffn_gate: Vec<f32>,
    /// `[t][intermediate]` SwiGLU up branch.
    pub(crate) ffn_up: Vec<f32>,
    /// `[t][intermediate]` `silu(gate) * up`, rotated in place for the
    /// folded `ffn_down` projection.
    pub(crate) ffn_act: Vec<f32>,
    /// `[t][2 * heads_width]` raw `attn_q` output (`q|gate` interleaved).
    pub(crate) q_all: Vec<f32>,
    /// `[t][heads_width]` de-interleaved, normalised, rotated queries.
    pub(crate) q: Vec<f32>,
    /// `[t][heads_width]` de-interleaved output gate.
    pub(crate) gate: Vec<f32>,
    /// `[t][kv_width]` keys.
    pub(crate) k: Vec<f32>,
    /// `[t][kv_width]` values.
    pub(crate) v: Vec<f32>,
    /// `[t][heads_width]` attention output.
    pub(crate) attn: Vec<f32>,
    /// `[t][heads_width]` attention output after the sigmoid gate, rotated
    /// in place for the folded `attn_output` projection.
    pub(crate) attn_gated: Vec<f32>,
    /// `[head_dim]` single-head staging buffer.
    pub(crate) head_tmp: Vec<f32>,
    /// `[t][conv_dim]` `attn_qkv` output.
    pub(crate) qkv: Vec<f32>,
    /// `[conv_dim][conv_taps - 1 + t]` channel-major conv window.
    pub(crate) conv_x: Vec<f32>,
    /// `[t][conv_dim]` conv output (SiLU applied in place).
    pub(crate) conv_out: Vec<f32>,
    /// `[t][qk_width]` L2-normalised queries.
    pub(crate) gdn_q: Vec<f32>,
    /// `[t][qk_width]` L2-normalised keys.
    pub(crate) gdn_k: Vec<f32>,
    /// `[t][inner]` values in grouped v-head order.
    pub(crate) gdn_v: Vec<f32>,
    /// `[t][inner]` `z` gate in tiled order.
    pub(crate) z_tiled: Vec<f32>,
    /// `[t][inner]` `z` gate in grouped order.
    pub(crate) z_grouped: Vec<f32>,
    /// `[t][n_v_heads]` raw `ssm_alpha` in tiled order.
    pub(crate) alpha_tiled: Vec<f32>,
    /// `[t][n_v_heads]` grouped `ssm_alpha`.
    pub(crate) alpha: Vec<f32>,
    /// `[t][n_v_heads]` raw `ssm_beta` in tiled order.
    pub(crate) beta_tiled: Vec<f32>,
    /// `[t][n_v_heads]` grouped `ssm_beta`.
    pub(crate) beta: Vec<f32>,
    /// `[t][inner]` Gated-DeltaNet output (grouped).
    pub(crate) gdn_out: Vec<f32>,
    /// `[t][inner]` gated-RMSNorm output (grouped).
    pub(crate) gdn_norm: Vec<f32>,
    /// `[hidden]` rotated final hidden state for the LM head.
    pub(crate) final_rot: Vec<f32>,
}

impl HybridScratch {
    /// Allocate for `capacity` tokens of `config`'s geometry.
    #[must_use]
    pub fn new(config: &HybridConfig, capacity: usize) -> Self {
        let mut scratch = Self {
            hidden: config.base.hidden_size,
            intermediate: config.base.intermediate_size,
            heads_width: config.base.num_attention_heads * config.base.head_dim,
            kv_width: config.base.num_kv_heads * config.base.head_dim,
            conv_dim: config.conv_dim(),
            inner: config.ssm_inner_size,
            qk_width: config.n_k_heads() * config.head_k_dim(),
            n_v_heads: config.n_v_heads(),
            conv_taps: config.ssm_conv_kernel,
            head_tmp: vec![0.0; config.base.head_dim.max(config.head_k_dim())],
            final_rot: vec![0.0; config.base.hidden_size],
            ..Self::default()
        };
        scratch.grow(capacity.max(1));
        scratch
    }

    /// Tokens this scratch can currently hold.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes currently held by every buffer.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let floats = self.resid.len()
            + self.normed.len()
            + self.rotated.len()
            + self.proj.len()
            + self.ffn_gate.len()
            + self.ffn_up.len()
            + self.ffn_act.len()
            + self.q_all.len()
            + self.q.len()
            + self.gate.len()
            + self.k.len()
            + self.v.len()
            + self.attn.len()
            + self.attn_gated.len()
            + self.head_tmp.len()
            + self.qkv.len()
            + self.conv_x.len()
            + self.conv_out.len()
            + self.gdn_q.len()
            + self.gdn_k.len()
            + self.gdn_v.len()
            + self.z_tiled.len()
            + self.z_grouped.len()
            + self.alpha_tiled.len()
            + self.alpha.len()
            + self.beta_tiled.len()
            + self.beta.len()
            + self.gdn_out.len()
            + self.gdn_norm.len()
            + self.final_rot.len();
        floats * core::mem::size_of::<f32>()
    }

    /// Make sure every buffer holds at least `t_len` tokens.
    pub(crate) fn ensure(&mut self, t_len: usize) {
        if t_len > self.capacity {
            self.grow(t_len);
        }
    }

    fn grow(&mut self, t_len: usize) {
        fn fit(buffer: &mut Vec<f32>, len: usize) {
            buffer.clear();
            buffer.resize(len, 0.0);
        }
        fit(&mut self.resid, t_len * self.hidden);
        fit(&mut self.normed, t_len * self.hidden);
        fit(&mut self.rotated, t_len * self.hidden);
        fit(&mut self.proj, t_len * self.hidden);
        fit(&mut self.ffn_gate, t_len * self.intermediate);
        fit(&mut self.ffn_up, t_len * self.intermediate);
        fit(&mut self.ffn_act, t_len * self.intermediate);
        fit(&mut self.q_all, t_len * self.heads_width * 2);
        fit(&mut self.q, t_len * self.heads_width);
        fit(&mut self.gate, t_len * self.heads_width);
        fit(&mut self.k, t_len * self.kv_width);
        fit(&mut self.v, t_len * self.kv_width);
        fit(&mut self.attn, t_len * self.heads_width);
        fit(&mut self.attn_gated, t_len * self.heads_width);
        fit(&mut self.qkv, t_len * self.conv_dim);
        fit(&mut self.conv_out, t_len * self.conv_dim);
        fit(&mut self.gdn_q, t_len * self.qk_width);
        fit(&mut self.gdn_k, t_len * self.qk_width);
        fit(&mut self.gdn_v, t_len * self.inner);
        fit(&mut self.z_tiled, t_len * self.inner);
        fit(&mut self.z_grouped, t_len * self.inner);
        fit(&mut self.alpha_tiled, t_len * self.n_v_heads);
        fit(&mut self.alpha, t_len * self.n_v_heads);
        fit(&mut self.beta_tiled, t_len * self.n_v_heads);
        fit(&mut self.beta, t_len * self.n_v_heads);
        fit(&mut self.gdn_out, t_len * self.inner);
        fit(&mut self.gdn_norm, t_len * self.inner);
        // Channel-major, so the row *length* grows with the chunk:
        // `[conv_dim][conv_taps - 1 + t_len]`.
        fit(
            &mut self.conv_x,
            self.conv_dim * (self.conv_taps.saturating_sub(1) + t_len),
        );
        self.capacity = t_len;
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Driver
// ─────────────────────────────────────────────────────────────────────────

/// Everything one [`run_chunk`] call reads or writes, borrowed field by
/// field so the caller can hand out disjoint borrows of one model.
pub struct ForwardCtx<'m, 'a> {
    /// The parsed `qwen35` configuration.
    pub config: &'m HybridConfig,
    /// All blocks, in stack order.
    pub blocks: &'m [HybridBlock<'a>],
    /// `token_embd.weight`, still quantized.
    pub embedding: &'m HybridEmbedding<'a>,
    /// Final RMSNorm.
    pub output_norm: &'m RmsNorm,
    /// `output.weight` (folded).
    pub lm_head: &'m LinearLayer<'a>,
    /// The fold, when the checkpoint carries one.
    pub hadamard: Option<&'m HadamardHook>,
    /// Tiled ↔ grouped v-head map.
    pub vhead_map: &'m VHeadMap,
    /// Partial-RoPE angles.
    pub rope: &'m RopeTables,
    /// KV cache, indexed by **`kv_slot`**.
    pub kv: &'m mut KvCache,
    /// Recurrent state, indexed by **`rec_slot`**.
    pub recurrent: &'m mut RecurrentCache,
    /// Chunk-wide activation buffers.
    pub scratch: &'m mut HybridScratch,
}

/// Run one chunk of `tokens` starting at absolute position `start_pos`.
///
/// When `logits` is `Some`, the LM head is evaluated for the **last** token
/// of the chunk only (the decode contract: a prefill does not need a logit
/// row per prompt token, and materialising 512 × 248 320 f32 would be
/// 508 MB). When `dump` is `Some`, every block's output is recorded
/// (design §8.2 G11).
///
/// # Errors
///
/// [`ModelError::PositionOutOfRange`] when the chunk would run past the KV
/// window or the RoPE table, [`ModelError::ShapeMismatch`] for a
/// wrong-sized `logits`, and anything the block bodies, the caches or the
/// kernels return.
pub fn run_chunk(
    ctx: &mut ForwardCtx<'_, '_>,
    tokens: &[u32],
    start_pos: usize,
    logits: Option<&mut [f32]>,
    mut dump: Option<&mut LayerDump>,
) -> ModelResult<()> {
    let t_len = tokens.len();
    if t_len == 0 {
        return Ok(());
    }
    let hidden = ctx.config.base.hidden_size;
    let vocab = ctx.config.base.vocab_size;
    let end_pos = start_pos
        .checked_add(t_len)
        .ok_or_else(|| ModelError::ShapeInvariant {
            tensor: "chunk end position".to_string(),
            expected: "start_pos + t_len representable as usize".to_string(),
            actual: format!("start_pos = {start_pos}, t_len = {t_len}"),
        })?;
    let window = ctx.kv.max_seq_len();
    if end_pos > window {
        return Err(ModelError::PositionOutOfRange {
            pos: end_pos.saturating_sub(1),
            max: window,
        });
    }
    if end_pos > ctx.rope.max_pos() {
        return Err(ModelError::PositionOutOfRange {
            pos: end_pos.saturating_sub(1),
            max: ctx.rope.max_pos(),
        });
    }

    ctx.scratch.ensure(t_len);

    // ── Embedding lookup + inverse rotation (design §3.5) ───────────────
    for (t, &token) in tokens.iter().enumerate() {
        let lo = t * hidden;
        let row = ctx
            .scratch
            .resid
            .get_mut(lo..lo + hidden)
            .ok_or_else(|| scratch_short("resid", (t + 1) * hidden))?;
        ctx.embedding.row(token, hidden, row)?;
        if let Some(hook) = ctx.hadamard {
            hook.inverse_embedding(row)?;
        }
    }
    if let Some(dump) = dump.as_deref_mut() {
        dump.tokens = tokens.to_vec();
        dump.start_pos = start_pos;
        dump.hidden = hidden;
        dump.embedding = ctx.scratch.resid[..t_len * hidden].to_vec();
        dump.layers.clear();
        dump.layers.reserve(ctx.blocks.len());
    }

    // ── The stack ───────────────────────────────────────────────────────
    let n_blocks = ctx.blocks.len();
    for index in 0..n_blocks {
        // `is_full` is read before any `&mut` borrow of `ctx` is taken, so
        // the dispatch does not hold a shared borrow of `*ctx` across the
        // call that needs `&mut ctx.scratch`.
        let is_full = ctx
            .blocks
            .get(index)
            .map(HybridBlock::is_full)
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: format!("block {index}"),
                expected: format!("< {n_blocks} blocks"),
                actual: "out of range".to_string(),
            })?;
        if is_full {
            crate::hybrid::block_full::forward_full_chunk(ctx, index, t_len, start_pos)?;
        } else {
            crate::hybrid::block_linear::forward_linear_chunk(ctx, index, t_len)?;
        }
        if let Some(dump) = dump.as_deref_mut() {
            dump.layers
                .push(ctx.scratch.resid[..t_len * hidden].to_vec());
        }
    }

    ctx.recurrent.advance(t_len);

    // ── Final norm + LM head (last token only) ──────────────────────────
    let last = (t_len - 1) * hidden;
    let tail = ctx
        .scratch
        .resid
        .get(last..last + hidden)
        .ok_or_else(|| scratch_short("resid", t_len * hidden))?;
    let mut normed = vec![0.0f32; hidden];
    ctx.output_norm.forward(tail, &mut normed)?;
    if let Some(dump) = dump {
        dump.final_norm.clear();
        dump.final_norm.extend_from_slice(&normed);
    }

    if let Some(logits) = logits {
        if logits.len() < vocab {
            return Err(ModelError::ShapeMismatch {
                name: "logits".to_string(),
                expected: vec![vocab],
                actual: vec![logits.len()],
            });
        }
        let head_input: &[f32] = match ctx.hadamard {
            Some(hook) => {
                let rot = ctx
                    .scratch
                    .final_rot
                    .get_mut(..hidden)
                    .ok_or_else(|| scratch_short("final_rot", hidden))?;
                rot.copy_from_slice(&normed);
                hook.rotate_rows_in_place(rot, hidden, 1)?;
                rot
            }
            None => &normed,
        };
        ctx.lm_head.forward_vec(head_input, &mut logits[..vocab])?;
    }

    Ok(())
}

/// A scratch buffer shorter than the chunk needs — an internal invariant
/// violation, reported rather than `unwrap`ped.
pub(crate) fn scratch_short(name: &str, needed: usize) -> ModelError {
    ModelError::ShapeInvariant {
        tensor: format!("hybrid scratch '{name}'"),
        expected: format!(">= {needed} floats"),
        actual: "shorter".to_string(),
    }
}

/// Add `src` into `dst`, element by element — the residual add both block
/// bodies end with.
pub(crate) fn residual_add(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

/// Normalise `rows` tokens of the residual stream into `normed`, and —
/// when the checkpoint is folded — produce the rotated copy every folded
/// projection of the layer shares (design §3.4: rotate **once** per
/// activation, not once per matmul).
pub(crate) fn norm_and_rotate(
    norm: &RmsNorm,
    resid: &[f32],
    normed: &mut [f32],
    rotated: &mut [f32],
    hadamard: Option<&HadamardHook>,
    width: usize,
    rows: usize,
) -> ModelResult<()> {
    for t in 0..rows {
        let lo = t * width;
        let hi = lo + width;
        let src = resid
            .get(lo..hi)
            .ok_or_else(|| scratch_short("resid", hi))?;
        let dst = normed
            .get_mut(lo..hi)
            .ok_or_else(|| scratch_short("normed", hi))?;
        norm.forward(src, dst)?;
    }
    if let Some(hook) = hadamard {
        let rows_len = rows * width;
        let dst = rotated
            .get_mut(..rows_len)
            .ok_or_else(|| scratch_short("rotated", rows_len))?;
        let src = normed
            .get(..rows_len)
            .ok_or_else(|| scratch_short("normed", rows_len))?;
        dst.copy_from_slice(src);
        hook.rotate_rows_in_place(dst, width, rows)?;
    }
    Ok(())
}

/// Pick the activation a folded projection consumes: the rotated copy when
/// the checkpoint carries a fold, the plain one otherwise.
#[inline]
pub(crate) fn folded_input<'s>(
    normed: &'s [f32],
    rotated: &'s [f32],
    hadamard: Option<&HadamardHook>,
    len: usize,
) -> ModelResult<&'s [f32]> {
    let src = if hadamard.is_some() { rotated } else { normed };
    src.get(..len)
        .ok_or_else(|| scratch_short("folded activation", len))
}

/// The SwiGLU feed-forward half of a layer, identical in both layer kinds
/// (design §3.4: `post_attention_norm` → rotate once → `ffn_gate`/`ffn_up`
/// → `silu(gate) * up` → rotate → `ffn_down` → residual).
///
/// Shared by [`crate::hybrid::block_full`] and
/// [`crate::hybrid::block_linear`] so the two cannot drift: the FFN is the
/// one part of a `qwen35` layer that is byte-for-byte the same shape in a
/// full-attention and a Gated-DeltaNet block.
///
/// # Errors
///
/// [`ModelError::ShapeInvariant`] for a short scratch buffer, plus anything
/// the norm, the projections or `swiglu_simd` return.
#[allow(
    clippy::too_many_arguments,
    reason = "one layer's FFN: three projections, its norm, two widths and the chunk length"
)]
pub(crate) fn forward_ffn_chunk(
    ctx: &mut ForwardCtx<'_, '_>,
    post_attn_norm: &RmsNorm,
    ffn_gate: &LinearLayer<'_>,
    ffn_up: &LinearLayer<'_>,
    ffn_down: &LinearLayer<'_>,
    hidden: usize,
    intermediate: usize,
    t_len: usize,
) -> ModelResult<()> {
    let hadamard = ctx.hadamard;
    let sc = &mut *ctx.scratch;
    let hidden_len = t_len * hidden;
    let inter_len = t_len * intermediate;

    norm_and_rotate(
        post_attn_norm,
        &sc.resid,
        &mut sc.normed,
        &mut sc.rotated,
        hadamard,
        hidden,
        t_len,
    )?;
    let folded = folded_input(&sc.normed, &sc.rotated, hadamard, hidden_len)?;
    ffn_gate.forward_mat(
        folded,
        sc.ffn_gate
            .get_mut(..inter_len)
            .ok_or_else(|| scratch_short("ffn_gate", inter_len))?,
        t_len,
    )?;
    ffn_up.forward_mat(
        folded,
        sc.ffn_up
            .get_mut(..inter_len)
            .ok_or_else(|| scratch_short("ffn_up", inter_len))?,
        t_len,
    )?;
    oxibonsai_kernels::swiglu_simd(
        sc.ffn_gate
            .get(..inter_len)
            .ok_or_else(|| scratch_short("ffn_gate", inter_len))?,
        sc.ffn_up
            .get(..inter_len)
            .ok_or_else(|| scratch_short("ffn_up", inter_len))?,
        sc.ffn_act
            .get_mut(..inter_len)
            .ok_or_else(|| scratch_short("ffn_act", inter_len))?,
    )
    .map_err(ModelError::Kernel)?;
    if let Some(hook) = hadamard {
        hook.rotate_rows_in_place(
            sc.ffn_act
                .get_mut(..inter_len)
                .ok_or_else(|| scratch_short("ffn_act", inter_len))?,
            intermediate,
            t_len,
        )?;
    }
    ffn_down.forward_mat(
        sc.ffn_act
            .get(..inter_len)
            .ok_or_else(|| scratch_short("ffn_act", inter_len))?,
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
    Ok(())
}

// -------------------------------------------------------------------------
//  The model seam the runtime dispatches through
// -------------------------------------------------------------------------

/// Either kind of model a GGUF in this tree can hold, selected by
/// `general.architecture` (gatekeeper REQUIRED #1(a)).
///
/// A `qwen35` file cannot be loaded as a [`BonsaiModel`] -- it has no
/// `blk.N.attn_q.weight` on 48 of its 64 layers, a `q|gate` interleave on
/// the other 16, a Hadamard fold on every matrix and a recurrent state
/// beside the KV cache -- and a `qwen3` file cannot be loaded as a
/// [`HybridModel`]. This enum is the one place that choice is made, so no
/// caller has to re-derive it from the metadata and no caller can get it
/// wrong in only one of several places.
///
/// # Why an enum and not a trait object
///
/// The two models' `forward` signatures differ in exactly one argument:
/// the dense path takes a `&dyn OneBitKernel` (its block bodies dispatch
/// per call), the hybrid path carries its dispatcher inside each
/// `LinearLayer`. An enum lets the seam take the union once, at one call
/// site, instead of forcing an unused `&dyn` parameter into the hybrid
/// path's hot loop or boxing a second dispatcher per layer.
///
/// Not `Debug`: `BonsaiModel` deliberately is not either (it borrows a
/// whole mmap'd GGUF, and formatting it would walk gigabytes of weights).
pub enum LoadedModel<'a> {
    /// A dense Qwen3 stack (`general.architecture = "qwen3"`).
    Dense(Box<crate::model::BonsaiModel<'a>>),
    /// A `qwen35` hybrid stack (PrismML Bonsai 2).
    Hybrid(Box<crate::hybrid::model::HybridModel<'a>>),
}

impl<'a> LoadedModel<'a> {
    /// `general.architecture` of `gguf`, or `""` when it declares none.
    #[must_use]
    pub fn architecture_of(gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>) -> String {
        gguf.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default()
    }

    /// `true` when `gguf` declares the hybrid architecture.
    #[must_use]
    pub fn is_hybrid_gguf(gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>) -> bool {
        Self::architecture_of(gguf) == oxibonsai_core::config_hybrid::HYBRID_ARCHITECTURE
    }

    /// Load whichever model `gguf` declares.
    ///
    /// # Errors
    ///
    /// Whatever the selected loader returns.
    pub fn from_gguf(
        gguf: &'a oxibonsai_core::gguf::reader::GgufFile<'a>,
        max_seq_len: usize,
    ) -> ModelResult<Self> {
        if Self::is_hybrid_gguf(gguf) {
            Ok(Self::Hybrid(Box::new(
                crate::hybrid::model::HybridModel::from_gguf(gguf, max_seq_len)?,
            )))
        } else {
            Ok(Self::Dense(Box::new(crate::model::BonsaiModel::from_gguf(
                gguf,
                max_seq_len,
            )?)))
        }
    }

    /// `true` for the hybrid arm.
    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        matches!(self, Self::Hybrid(_))
    }

    /// The dense model, if this is one.
    #[must_use]
    pub fn as_dense(&self) -> Option<&crate::model::BonsaiModel<'a>> {
        match self {
            Self::Dense(m) => Some(m),
            Self::Hybrid(_) => None,
        }
    }

    /// The dense model, mutably, if this is one.
    pub fn as_dense_mut(&mut self) -> Option<&mut crate::model::BonsaiModel<'a>> {
        match self {
            Self::Dense(m) => Some(m),
            Self::Hybrid(_) => None,
        }
    }

    /// The hybrid model, if this is one.
    #[must_use]
    pub fn as_hybrid(&self) -> Option<&crate::hybrid::model::HybridModel<'a>> {
        match self {
            Self::Dense(_) => None,
            Self::Hybrid(m) => Some(m),
        }
    }

    /// The hybrid model, mutably, if this is one.
    pub fn as_hybrid_mut(&mut self) -> Option<&mut crate::hybrid::model::HybridModel<'a>> {
        match self {
            Self::Dense(_) => None,
            Self::Hybrid(m) => Some(m),
        }
    }

    /// Vocabulary width of the LM head.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Dense(m) => m.config().vocab_size,
            Self::Hybrid(m) => m.config().base.vocab_size,
        }
    }

    /// Hidden width of the residual stream.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Dense(m) => m.config().hidden_size,
            Self::Hybrid(m) => m.config().base.hidden_size,
        }
    }

    /// Layers in the stack.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        match self {
            Self::Dense(m) => m.config().num_layers,
            Self::Hybrid(m) => m.config().base.num_layers,
        }
    }

    /// The KV window both arms were built with.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        match self {
            Self::Dense(m) => m.kv_cache().max_seq_len(),
            Self::Hybrid(m) => m.max_seq_len(),
        }
    }

    /// One-line summary for `oxibonsai info`.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Dense(m) => {
                let c = m.config();
                format!(
                    "{} | {} layers | hidden {} | {} heads / {} kv heads",
                    c.architecture,
                    c.num_layers,
                    c.hidden_size,
                    c.num_attention_heads,
                    c.num_kv_heads,
                )
            }
            Self::Hybrid(m) => m.describe(),
        }
    }

    /// Single-token decode at `pos`, writing `[vocab_size]` logits.
    ///
    /// `kernel` is only consulted by the dense arm; the hybrid arm carries
    /// its dispatcher inside each `LinearLayer` (see the type docs).
    ///
    /// # Errors
    ///
    /// Whatever the selected model's forward returns.
    pub fn forward(
        &mut self,
        token: u32,
        pos: usize,
        kernel: &dyn oxibonsai_kernels::traits::OneBitKernel,
        logits: &mut [f32],
    ) -> ModelResult<()> {
        match self {
            Self::Dense(m) => m.forward_into(token, pos, kernel, logits),
            Self::Hybrid(m) => m.forward(token, pos, logits),
        }
    }

    /// Prefill `tokens` from `start_pos`, writing the last token's logits.
    ///
    /// # Errors
    ///
    /// Whatever the selected model's prefill returns.
    pub fn forward_prefill(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        kernel: &dyn oxibonsai_kernels::traits::OneBitKernel,
        last_logits: &mut [f32],
    ) -> ModelResult<()> {
        match self {
            Self::Dense(m) => {
                let produced = m.forward_prefill(tokens, start_pos, kernel)?;
                let vocab = m.config().vocab_size;
                let tail =
                    produced
                        .len()
                        .checked_sub(vocab)
                        .ok_or_else(|| ModelError::ShapeMismatch {
                            name: "prefill logits".to_string(),
                            expected: vec![vocab],
                            actual: vec![produced.len()],
                        })?;
                let row = produced
                    .get(tail..)
                    .ok_or_else(|| scratch_short("prefill logits", vocab))?;
                let available = last_logits.len();
                let out =
                    last_logits
                        .get_mut(..vocab)
                        .ok_or_else(|| ModelError::ShapeMismatch {
                            name: "prefill logits".to_string(),
                            expected: vec![vocab],
                            actual: vec![available],
                        })?;
                out.copy_from_slice(row);
                Ok(())
            }
            Self::Hybrid(m) => m.forward_prefill(tokens, start_pos, last_logits),
        }
    }

    /// Clear every per-sequence cache: the KV cursor on both arms, and the
    /// recurrent state on the hybrid one.
    pub fn reset(&mut self) {
        match self {
            Self::Dense(m) => m.reset(),
            Self::Hybrid(m) => m.reset(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::tests_support::bonsai2_config;

    #[test]
    fn rope_table_covers_every_position_and_refuses_past_the_end_bonsai2() {
        let config = bonsai2_config();
        let table = RopeTables::new(config.rope_dimension_count, 16, config.base.rope_freq_base)
            .expect("table builds");
        assert_eq!(table.n_rot(), 64);
        assert_eq!(table.max_pos(), 16);
        for pos in 0..16 {
            let (cos, sin) = table.angles(pos).expect("in range");
            assert_eq!(cos.len(), 32);
            assert_eq!(sin.len(), 32);
            // cos^2 + sin^2 == 1 for every angle.
            for (c, s) in cos.iter().zip(sin) {
                let unit = c.mul_add(*c, s * s);
                assert!((unit - 1.0).abs() < 1e-5, "cos^2+sin^2 = {unit}");
            }
        }
        // Position 0 is the identity rotation.
        let (cos, sin) = table.angles(0).expect("in range");
        assert!(cos.iter().all(|c| (*c - 1.0).abs() < 1e-6));
        assert!(sin.iter().all(|s| s.abs() < 1e-6));
        assert!(matches!(
            table.angles(16),
            Err(ModelError::PositionOutOfRange { pos: 16, max: 16 })
        ));
    }

    #[test]
    fn rope_table_rejects_an_odd_rotation_width_bonsai2() {
        assert!(RopeTables::new(63, 4, 1e7).is_err());
        assert!(RopeTables::new(64, 0, 1e7).is_err());
    }

    #[test]
    fn scratch_grows_to_the_chunk_and_keeps_every_buffer_consistent_bonsai2() {
        let config = bonsai2_config();
        let mut scratch = HybridScratch::new(&config, 1);
        assert_eq!(scratch.capacity(), 1);
        assert_eq!(scratch.resid.len(), config.base.hidden_size);
        assert_eq!(scratch.qkv.len(), config.conv_dim());
        // `conv_x` is channel-major: `[conv_dim][taps - 1 + t]`.
        assert_eq!(
            scratch.conv_x.len(),
            config.conv_dim() * (config.ssm_conv_kernel - 1 + 1)
        );
        let single = scratch.memory_bytes();

        scratch.ensure(8);
        assert_eq!(scratch.capacity(), 8);
        assert_eq!(scratch.resid.len(), 8 * config.base.hidden_size);
        assert_eq!(scratch.ffn_act.len(), 8 * config.base.intermediate_size);
        assert_eq!(scratch.attn_gated.len(), 8 * 6144);
        assert_eq!(
            scratch.conv_x.len(),
            config.conv_dim() * (config.ssm_conv_kernel - 1 + 8)
        );
        assert!(scratch.memory_bytes() > single);

        // Shrinking is a no-op: the buffers stay big enough for the largest
        // chunk seen, so a decode after a prefill re-allocates nothing.
        scratch.ensure(1);
        assert_eq!(scratch.capacity(), 8);
    }

    #[test]
    fn layer_dump_indexes_rows_by_layer_and_token_bonsai2() {
        let dump = LayerDump {
            tokens: vec![7, 8, 9],
            start_pos: 4,
            hidden: 2,
            embedding: vec![0.0; 6],
            layers: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            final_norm: vec![0.5, 0.5],
        };
        assert_eq!(dump.t_len(), 3);
        assert_eq!(dump.layer_row(0, 0), Some(&[1.0f32, 2.0][..]));
        assert_eq!(dump.layer_row(0, 2), Some(&[5.0f32, 6.0][..]));
        assert_eq!(dump.layer_row(0, 3), None);
        assert_eq!(dump.layer_row(1, 0), None);
    }

    #[test]
    fn folded_input_picks_the_rotated_copy_only_when_folded_bonsai2() {
        let normed = vec![1.0f32, 2.0, 3.0, 4.0];
        let rotated = vec![9.0f32, 9.0, 9.0, 9.0];
        let plain = folded_input(&normed, &rotated, None, 4).expect("in range");
        assert_eq!(plain, &normed[..]);
        assert!(folded_input(&normed, &rotated, None, 5).is_err());
    }

    #[test]
    fn residual_add_accumulates_in_place_bonsai2() {
        let mut dst = vec![1.0f32, 2.0, 3.0];
        residual_add(&mut dst, &[0.5, -1.0, 10.0]);
        assert_eq!(dst, vec![1.5, 1.0, 13.0]);
    }
}
