//! The `qwen35` forward / prefill driver (design §3.10) and the model seam
//! the runtime dispatches through.
//!
//! # One body, parameterised by batch
//!
//! [`run_chunk_input`] is the *only* forward body in this module tree
//! ([`run_chunk`] is its token-id form). A decode step is one call with
//! `t_len == 1`; a chunked prefill is the same call with `t_len == chunk`.
//! Nothing is duplicated between the two, so
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
//! # Two kinds of chunk input (design §6.2)
//!
//! A chunk enters block 0 either as **token ids** ([`ChunkInput::Tokens`]:
//! each row is the `token_embd` lookup, inverse-rotated for a folded
//! checkpoint, design §3.5) or as **caller-supplied rows**
//! ([`ChunkInput::Rows`]: e.g. a vision tower's merged image rows, which
//! live in the unrotated embedding basis already and must reach block 0
//! untouched). The two also differ in how rotary positions are named:
//!
//! * the **KV slot** of every row is always its absolute *sequence* index
//!   (`start_pos + t`) — attention is causal in sequence order, exactly as
//!   the reference's 2-D M-RoPE causal mask reduces to for an image laid out
//!   row-major between text tokens;
//! * the **rotary angle** of a token row is its text position (all three
//!   M-RoPE axes equal, which degenerates bitwise to the precomputed
//!   single-axis table, design §2.5), while an image row carries its own
//!   3-axis [`MropePos`] (`t = p0`, `h = p0 + row`, `w = p0 + col`). After an
//!   image, text positions resume at `p0 + max(h, w)`, so they run *behind*
//!   the sequence index — the model keeps that offset
//!   ([`crate::hybrid::HybridModel::rope_delta`]).
//!
//! # Layer dumps (design §8.2 G11)
//!
//! [`LayerDump`] records the residual stream after each block, so the CPU
//! path can be used as the reference for the later Metal parity gate
//! without re-running anything.

use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_kernels::rope_mrope::{mrope_build_tables, partial_rope_build_table};

use crate::error::{ModelError, ModelResult};
use crate::hybrid::block::HybridBlock;
use crate::hybrid::hadamard::HadamardHook;
use crate::hybrid::recurrent_cache::RecurrentCache;
use crate::hybrid::vhead_map::VHeadMap;
use crate::hybrid::weights::HybridEmbedding;
use crate::kv_cache::KvCache;
use crate::layers::linear::LinearLayer;
use crate::layers::rms_norm::RmsNorm;
use crate::layers::rope_mrope::MropePos;

/// Tokens processed per prefill chunk unless the caller says otherwise
/// (design §3.10: 512 × 17408 f32 ≈ 35 MB of activation peak).
pub const DEFAULT_PREFILL_CHUNK: usize = 512;

// ─────────────────────────────────────────────────────────────────────────
//  RoPE tables
// ─────────────────────────────────────────────────────────────────────────

/// Precomputed partial-NeoX RoPE angles for every position a sequence can
/// reach (design §2.5), plus the 3-axis M-RoPE angles of an image row.
///
/// Text-only M-RoPE is provably identical to standard NeoX RoPE over the
/// first `n_rot` of `head_dim` when all three position axes carry the token
/// index, so one `(cos, sin)` pair per position is enough; the table is
/// built through [`partial_rope_build_table`], i.e. through the very
/// [`mrope_build_tables`] the vision rows use, rather than through a second
/// angle formula that could drift from it.
///
/// A table built with [`RopeTables::with_sections`] also knows the model's
/// `rope.dimension_sections`, which [`RopeTables::fill_angles`] needs for a
/// position whose three axes differ (an image row, design §6.2).
#[derive(Debug, Clone)]
pub struct RopeTables {
    cos: Vec<f32>,
    sin: Vec<f32>,
    n_rot: usize,
    half: usize,
    max_pos: usize,
    freq_base: f32,
    /// `rope.dimension_sections`, when the table serves 3-axis positions.
    sections: Option<[u32; 4]>,
}

impl RopeTables {
    /// Build the text table for positions `0..max_pos`.
    ///
    /// A table built this way serves text positions only;
    /// [`RopeTables::fill_angles`] refuses a position whose axes differ.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] for an odd `n_rot` or a zero
    /// `max_pos`, or a position that does not fit in the `i32` the kernel's
    /// angle builder takes; [`ModelError::Kernel`] from the builder itself.
    pub fn new(n_rot: usize, max_pos: usize, freq_base: f32) -> ModelResult<Self> {
        Self::build(n_rot, max_pos, freq_base, None)
    }

    /// [`RopeTables::new`] for a model whose full-attention layers take
    /// 3-axis M-RoPE positions: `sections` is `rope.dimension_sections`
    /// (`[11, 11, 10, 0]` for Bonsai 2).
    ///
    /// # Errors
    ///
    /// As [`RopeTables::new`], plus [`ModelError::Kernel`] when `sections`
    /// sum to zero or leave a rotation pair to the vision `e` axis (which a
    /// `[t, h, w]` position cannot drive) — checked once here, so a vision
    /// row can never discover it mid-forward.
    pub fn with_sections(
        n_rot: usize,
        max_pos: usize,
        freq_base: f32,
        sections: [u32; 4],
    ) -> ModelResult<Self> {
        let table = Self::build(n_rot, max_pos, freq_base, Some(sections))?;
        // Probe the section map with three distinct axes: the kernel
        // refuses sections that need the `e` axis the first time such a
        // sector is reached.
        let mut cos = vec![0.0f32; table.half];
        let mut sin = vec![0.0f32; table.half];
        mrope_build_tables([0, 1, 2], sections, n_rot, freq_base, &mut cos, &mut sin)
            .map_err(ModelError::Kernel)?;
        Ok(table)
    }

    fn build(
        n_rot: usize,
        max_pos: usize,
        freq_base: f32,
        sections: Option<[u32; 4]>,
    ) -> ModelResult<Self> {
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
            freq_base,
            sections,
        })
    }

    /// Rotated dimensions per head (`n_rot`).
    #[inline]
    #[must_use]
    pub fn n_rot(&self) -> usize {
        self.n_rot
    }

    /// The `rope.dimension_sections` this table serves 3-axis positions
    /// with, or `None` for a text-only table.
    #[inline]
    #[must_use]
    pub fn sections(&self) -> Option<[u32; 4]> {
        self.sections
    }

    /// Write the `(cos, sin)` angles of the 3-axis position `pos` into
    /// `cos_out` / `sin_out` (each at least `n_rot / 2` long).
    ///
    /// A text position (`t == h == w`) copies the precomputed row for `t`:
    /// the kernel builder degenerates to exactly that row bitwise (design
    /// §2.5), so a text token spliced between image rows rotates exactly as
    /// it would in a text-only prompt. Any other position is built through
    /// [`mrope_build_tables`] with the model's sections — the reference's
    /// interleaved M-RoPE (`sector % 3` picks `t` / `h` / `w`).
    ///
    /// # Errors
    ///
    /// [`ModelError::PositionOutOfRange`] for an axis at or past
    /// [`RopeTables::max_pos`]; [`ModelError::ShapeInvariant`] for a text-only
    /// table asked for a non-text position or short output buffers;
    /// [`ModelError::Kernel`] from the angle builder.
    pub fn fill_angles(
        &self,
        pos: MropePos,
        cos_out: &mut [f32],
        sin_out: &mut [f32],
    ) -> ModelResult<()> {
        let axis_max = pos.t.max(pos.h).max(pos.w) as usize;
        if axis_max >= self.max_pos {
            return Err(ModelError::PositionOutOfRange {
                pos: axis_max,
                max: self.max_pos,
            });
        }
        let (Some(cos_dst), Some(sin_dst)) =
            (cos_out.get_mut(..self.half), sin_out.get_mut(..self.half))
        else {
            return Err(ModelError::ShapeInvariant {
                tensor: "rope angle buffers".to_string(),
                expected: format!("at least {} entries each", self.half),
                actual: format!("{} / {}", cos_out.len(), sin_out.len()),
            });
        };
        if pos.t == pos.h && pos.t == pos.w {
            let (cos, sin) = self.angles(pos.t as usize)?;
            cos_dst.copy_from_slice(cos);
            sin_dst.copy_from_slice(sin);
            return Ok(());
        }
        let sections = self.sections.ok_or_else(|| ModelError::ShapeInvariant {
            tensor: "rope table".to_string(),
            expected: "rope.dimension_sections for a 3-axis (image) position".to_string(),
            actual: format!(
                "a text-only table asked for (t={}, h={}, w={})",
                pos.t, pos.h, pos.w
            ),
        })?;
        let axis = |v: u32| {
            i32::try_from(v).map_err(|_| ModelError::ShapeInvariant {
                tensor: "rope position".to_string(),
                expected: "representable as i32".to_string(),
                actual: v.to_string(),
            })
        };
        mrope_build_tables(
            [axis(pos.t)?, axis(pos.h)?, axis(pos.w)?],
            sections,
            self.n_rot,
            self.freq_base,
            cos_dst,
            sin_dst,
        )
        .map_err(ModelError::Kernel)
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

/// The residual stream after each block, recorded for the Metal parity
/// gate (design §8.2 G11: a token list plus a per-layer dump to diff
/// against).
///
/// A dump holds one `[t_len][hidden]` row set per layer plus the final
/// post-`output_norm` hidden state, for the rows of the call it was
/// attached to.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerDump {
    /// The tokens this dump was produced from (empty for a chunk of
    /// caller-supplied rows, [`ChunkInput::Rows`]).
    pub tokens: Vec<u32>,
    /// The rotary position of every recorded row (text rows carry equal
    /// axes).
    pub rotary_positions: Vec<MropePos>,
    /// Absolute sequence position of the first row.
    pub start_pos: usize,
    /// Hidden width of every recorded row.
    pub hidden: usize,
    /// What entered block 0, `[t][hidden]`: the embedding *after* the
    /// inverse rotation for token rows, the caller's rows verbatim for
    /// [`ChunkInput::Rows`].
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

    /// Row `t` of what entered block 0, or `None` when out of range.
    #[must_use]
    pub fn embedding_row(&self, t: usize) -> Option<&[f32]> {
        self.embedding.get(t * self.hidden..(t + 1) * self.hidden)
    }

    /// Number of rows recorded.
    #[must_use]
    pub fn t_len(&self) -> usize {
        self.tokens.len().max(self.rotary_positions.len())
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

/// Everything one [`run_chunk_input`] call reads or writes, borrowed field
/// by field so the caller can hand out disjoint borrows of one model.
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

/// What one chunk feeds into block 0 (see the module docs, "Two kinds of
/// chunk input").
#[derive(Debug, Clone, Copy)]
pub enum ChunkInput<'i> {
    /// Token ids. Each row is the `token_embd` lookup, inverse-rotated for
    /// a folded checkpoint (design §3.5); token `t` rotates at the text
    /// position `rope_start + t` on all three M-RoPE axes.
    Tokens {
        /// The chunk's token ids.
        tokens: &'i [u32],
        /// Rotary position of `tokens[0]` — the sequence position minus the
        /// model's M-RoPE offset ([`crate::hybrid::HybridModel::rope_delta`]).
        rope_start: usize,
    },
    /// Caller-supplied rows, `[positions.len()][hidden]`, already in the
    /// basis block 0 expects: written into the residual stream verbatim —
    /// no `token_embd` lookup and **no inverse Hadamard transform** (a
    /// vision tower's rows are in the unrotated basis, design §3.5/§6.2) —
    /// each rotating at its own 3-axis position.
    Rows {
        /// `positions.len() * hidden` floats, row-major.
        rows: &'i [f32],
        /// One rotary position per row.
        positions: &'i [MropePos],
    },
}

impl ChunkInput<'_> {
    /// Rows in this chunk.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Tokens { tokens, .. } => tokens.len(),
            Self::Rows { positions, .. } => positions.len(),
        }
    }

    /// Whether the chunk carries no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How the full-attention layers name this chunk's rotary positions.
    fn rope(&self) -> ChunkRope<'_> {
        match *self {
            Self::Tokens { rope_start, .. } => ChunkRope::Text { start: rope_start },
            Self::Rows { positions, .. } => ChunkRope::Explicit(positions),
        }
    }
}

/// The rotary positions of one chunk's rows, as the full-attention layers
/// consume them.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ChunkRope<'p> {
    /// Row `t` rotates at the text position `start + t` (all three axes).
    Text {
        /// Rotary position of row 0.
        start: usize,
    },
    /// Row `t` rotates at `positions[t]`.
    Explicit(&'p [MropePos]),
}

/// Write the rows `tokens` enter block 0 with into `out`
/// (`[tokens.len()][hidden]`): the `token_embd` row of each id, then — for
/// a folded checkpoint — the inverse Hadamard transform (design §3.5).
///
/// The one implementation both [`run_chunk_input`]'s token path and a
/// caller assembling a mixed prompt
/// ([`crate::hybrid::HybridModel::embed_token_rows`]) use, so a text row of
/// a multimodal prompt is bit-identical to the same token's row in a
/// text-only one.
///
/// # Errors
///
/// [`ModelError::ShapeMismatch`] for an `out` that is not exactly
/// `tokens.len() * hidden` long, plus the embedding lookup's and the
/// transform's own errors (a token id past the vocabulary, a width the fold
/// does not cover).
pub(crate) fn embed_token_rows(
    embedding: &HybridEmbedding<'_>,
    hadamard: Option<&HadamardHook>,
    tokens: &[u32],
    hidden: usize,
    out: &mut [f32],
) -> ModelResult<()> {
    let expected = tokens.len().saturating_mul(hidden);
    if out.len() != expected {
        return Err(ModelError::ShapeMismatch {
            name: "embedded token rows".to_string(),
            expected: vec![expected],
            actual: vec![out.len()],
        });
    }
    if hidden == 0 {
        return Ok(());
    }
    for (&token, row) in tokens.iter().zip(out.chunks_exact_mut(hidden)) {
        embedding.row(token, hidden, row)?;
        if let Some(hook) = hadamard {
            hook.inverse_embedding(row)?;
        }
    }
    Ok(())
}

/// Run one chunk of `tokens` starting at absolute position `start_pos`,
/// rotating token `t` at the text position `start_pos + t` — i.e.
/// [`run_chunk_input`] with [`ChunkInput::Tokens`] and no M-RoPE offset.
///
/// # Errors
///
/// As [`run_chunk_input`].
pub fn run_chunk(
    ctx: &mut ForwardCtx<'_, '_>,
    tokens: &[u32],
    start_pos: usize,
    logits: Option<&mut [f32]>,
    dump: Option<&mut LayerDump>,
) -> ModelResult<()> {
    run_chunk_input(
        ctx,
        ChunkInput::Tokens {
            tokens,
            rope_start: start_pos,
        },
        start_pos,
        logits,
        dump,
    )
}

/// Run one chunk whose first row sits at absolute sequence position
/// `start_pos` (its KV slot and recurrent step; rotary positions come from
/// `input`, see [`ChunkInput`]).
///
/// When `logits` is `Some`, the LM head is evaluated for the **last** row
/// of the chunk only (the decode contract: a prefill does not need a logit
/// row per prompt token, and materialising 512 × 248 320 f32 would be
/// 508 MB). When `dump` is `Some`, every block's output is recorded
/// (design §8.2 G11).
///
/// # Errors
///
/// [`ModelError::PositionOutOfRange`] when the chunk would run past the KV
/// window or the RoPE table, [`ModelError::ShapeMismatch`] for a
/// wrong-sized `logits` or a [`ChunkInput::Rows`] whose `rows` are not
/// `positions.len() * hidden` long, and anything the block bodies, the
/// caches or the kernels return.
pub fn run_chunk_input(
    ctx: &mut ForwardCtx<'_, '_>,
    input: ChunkInput<'_>,
    start_pos: usize,
    logits: Option<&mut [f32]>,
    mut dump: Option<&mut LayerDump>,
) -> ModelResult<()> {
    let t_len = input.len();
    if t_len == 0 {
        return Ok(());
    }
    let hidden = ctx.config.base.hidden_size;
    let vocab = ctx.config.base.vocab_size;
    if let ChunkInput::Rows { rows, .. } = input {
        let expected = t_len.saturating_mul(hidden);
        if rows.len() != expected {
            return Err(ModelError::ShapeMismatch {
                name: "prefill rows".to_string(),
                expected: vec![t_len, hidden],
                actual: vec![rows.len()],
            });
        }
    }
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
    // Lazy KV growth (REQUIRED #4 (4)): make every position of the chunk
    // resident BEFORE any layer runs, so an allocation failure surfaces as
    // a typed error with no layer half-written, instead of mid-stack.
    ctx.kv.try_ensure_capacity(end_pos)?;

    ctx.scratch.ensure(t_len);

    // ── What enters block 0 ──────────────────────────────────────────────
    let rows_len = t_len * hidden;
    let resid = ctx
        .scratch
        .resid
        .get_mut(..rows_len)
        .ok_or_else(|| scratch_short("resid", rows_len))?;
    match input {
        // Embedding lookup + inverse rotation (design §3.5).
        ChunkInput::Tokens { tokens, .. } => {
            embed_token_rows(ctx.embedding, ctx.hadamard, tokens, hidden, resid)?;
        }
        // Verbatim: these rows are already in the basis block 0 reads, and
        // rotating them again would silently corrupt every one of them.
        ChunkInput::Rows { rows, .. } => resid.copy_from_slice(rows),
    }
    if let Some(dump) = dump.as_deref_mut() {
        match input {
            ChunkInput::Tokens { tokens, rope_start } => {
                dump.tokens = tokens.to_vec();
                dump.rotary_positions = (rope_start..rope_start + t_len)
                    .map(|p| MropePos::text(u32::try_from(p).unwrap_or(u32::MAX)))
                    .collect();
            }
            ChunkInput::Rows { positions, .. } => {
                dump.tokens.clear();
                dump.rotary_positions = positions.to_vec();
            }
        }
        dump.start_pos = start_pos;
        dump.hidden = hidden;
        dump.embedding = ctx.scratch.resid[..rows_len].to_vec();
        dump.layers.clear();
        dump.layers.reserve(ctx.blocks.len());
    }
    let rope = input.rope();

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
            crate::hybrid::block_full::forward_full_chunk(ctx, index, t_len, start_pos, rope)?;
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

    /// Whether this model carries recurrent (Gated-DeltaNet) state at all —
    /// `false` for a dense stack, which has no recurrent layers.
    #[must_use]
    pub fn has_recurrent_state(&self) -> bool {
        match self {
            Self::Dense(_) => false,
            Self::Hybrid(m) => m.recurrent().n_layers() > 0,
        }
    }

    /// The recurrent state, when there is one.
    #[must_use]
    pub fn recurrent_state(&self) -> Option<&crate::hybrid::recurrent_cache::RecurrentCache> {
        match self {
            Self::Dense(_) => None,
            Self::Hybrid(m) => Some(m.recurrent()),
        }
    }

    /// Clear the recurrent state only (REQUIRED #6 / RT-28): the hybrid
    /// arm zeroes every Gated-DeltaNet state and conv window; the dense arm
    /// has no recurrent layers, so there is nothing to clear.
    pub fn reset_recurrent(&mut self) {
        match self {
            Self::Dense(m) => m.reset_recurrent(),
            Self::Hybrid(m) => m.reset_recurrent(),
        }
    }

    /// Install a recurrent state after validating its geometry (REQUIRED #6).
    ///
    /// # Errors
    ///
    /// [`ModelError::RecurrentStateMismatch`] when the geometry differs from
    /// the hybrid model's — and always for the dense arm, which has no
    /// recurrent layers to receive a (non-empty) state.
    pub fn set_recurrent_state(
        &mut self,
        state: crate::hybrid::recurrent_cache::RecurrentCache,
    ) -> ModelResult<()> {
        match self {
            Self::Dense(_) => {
                if state.n_layers() == 0 {
                    return Ok(());
                }
                Err(ModelError::RecurrentStateMismatch {
                    expected: "no recurrent layers (a dense qwen3 stack)".to_string(),
                    actual: format!("{} recurrent layers", state.n_layers()),
                })
            }
            Self::Hybrid(m) => m.set_recurrent_state(state),
        }
    }

    /// Take the recurrent state out (leaving a zeroed one of the same
    /// geometry behind), or `None` for the dense arm.
    ///
    /// # Errors
    ///
    /// As [`HybridModel::take_recurrent`](crate::hybrid::model::HybridModel::take_recurrent).
    pub fn take_recurrent_state(
        &mut self,
    ) -> ModelResult<Option<crate::hybrid::recurrent_cache::RecurrentCache>> {
        match self {
            Self::Dense(_) => Ok(None),
            Self::Hybrid(m) => m.take_recurrent().map(Some),
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
            rotary_positions: vec![MropePos::text(4), MropePos::text(5), MropePos::text(6)],
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

    /// REQUIRED #6 at the seam the runtime dispatches through: the hybrid
    /// arm resets / takes / installs its recurrent state (validated), the
    /// dense arm has none and refuses a non-empty one with a typed error.
    #[test]
    fn loaded_model_recurrent_seam_dispatches_by_arm_bonsai2() {
        use crate::hybrid::recurrent_cache::RecurrentCache;
        use crate::hybrid::tests_support::{synthetic_gguf, FixtureOptions, FixtureShape};
        use oxibonsai_core::gguf::reader::GgufFile;

        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut hybrid = LoadedModel::from_gguf(&gguf, 32).expect("hybrid loads");
        assert!(hybrid.is_hybrid());
        assert!(hybrid.has_recurrent_state());
        let config = hybrid
            .as_hybrid()
            .map(|m| m.config().clone())
            .expect("hybrid arm");
        if let Some(model) = hybrid.as_hybrid_mut() {
            model.recurrent_mut().advance(2);
            model.recurrent_mut().ssm_mut(0).expect("slot 0")[1] = 1.0;
        }
        hybrid.reset_recurrent();
        let state = hybrid.recurrent_state().expect("hybrid state");
        assert_eq!(state.token_count(), 0);
        assert!(state.ssm(0).expect("slot 0").iter().all(|v| *v == 0.0));

        let taken = hybrid
            .take_recurrent_state()
            .expect("take succeeds")
            .expect("the hybrid arm has a state");
        hybrid
            .set_recurrent_state(taken)
            .expect("its own state round-trips");

        let mut dense = LoadedModel::Dense(Box::new(crate::model::BonsaiModel::new(
            oxibonsai_core::config::Qwen3Config::tiny_test(),
        )));
        assert!(!dense.has_recurrent_state());
        assert!(dense.recurrent_state().is_none());
        assert!(dense.take_recurrent_state().expect("ok").is_none());
        dense.reset_recurrent();
        let foreign = RecurrentCache::new(&config).expect("state");
        let err = dense
            .set_recurrent_state(foreign)
            .expect_err("a dense stack has no recurrent layers");
        assert_eq!(err.error_code(), "RECURRENT_STATE_MISMATCH");
    }

    #[test]
    fn residual_add_accumulates_in_place_bonsai2() {
        let mut dst = vec![1.0f32, 2.0, 3.0];
        residual_add(&mut dst, &[0.5, -1.0, 10.0]);
        assert_eq!(dst, vec![1.5, 1.0, 13.0]);
    }
}
