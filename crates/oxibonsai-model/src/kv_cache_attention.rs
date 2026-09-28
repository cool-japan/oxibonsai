//! Storage-aware GQA attention read straight out of a [`KvCache`]
//! (B2-11-FIX, gatekeeper REQUIRED #4: "an f16-aware GQA read").
//!
//! # Why this lives beside the cache
//!
//! With `f16` the host default, attention can no longer borrow the history
//! as `&[f32]`, and copying it (`keys_for_owned`) costs `seq_len × head_dim`
//! floats per head per token — 8 MB per head per token at a 8192-token
//! context. [`KvCache::attend_group`] instead reads the storage in place:
//!
//! * **`f32` storage** — unchanged: every query head runs
//!   [`fused_attention_head_contiguous`] over zero-copy slices, so a dense
//!   `f32` cache produces exactly the bits it always did;
//! * **`f16` storage** — [`attend_f16_group`]: all query heads of one GQA
//!   group walk the history together, 32 positions at a time; each key and
//!   value row is widened to `f32` **once** into a stack buffer and shared
//!   by the group's heads. No heap allocation for `head_dim <= 256` and a
//!   group of at most 16 heads (every model this crate ships).
//!
//! # Bit-exactness contract
//!
//! [`attend_f16_group`] replicates [`fused_attention_head_contiguous`]
//! operation for operation — the 32-position blocks, `dot_f32(q, k) *
//! scale`, one rescale per block against the block maximum, then
//! `exp(score - max)` / `sum += e` / `axpy_f32(acc, e, v)` per position in
//! order, and the final `1 / sum` scale — through the very same SIMD
//! helpers. Widening `f16 → f32` is exact, so its output is **bit-identical**
//! to dequantising the whole history and calling the contiguous kernel,
//! which the tests assert with `assert_eq!` on raw `f32`s.

use half::f16;
use half::slice::HalfFloatSliceExt;

use super::{KvCache, KvStorage};
use crate::error::{ModelError, ModelResult};
use crate::layers::attention_fused::{
    axpy_f32, dot_f32, fused_attention_head_contiguous, scale_f32, ATTENTION_BLOCK_SIZE,
};

/// Largest `head_dim` served by the stack buffers (Qwen3: 64/128, Bonsai 2:
/// 256). Larger heads still work, through one heap buffer per call.
const STACK_HEAD_DIM: usize = 256;

/// Largest GQA group (query heads per KV head) served by stack state
/// (Qwen3-8B: 4, Bonsai 2 27B: 6).
const STACK_GROUP: usize = 16;

/// Online-softmax state of one query head (the same three quantities
/// `attention_fused`'s private state carries).
struct HeadState {
    max_val: f32,
    sum_exp: f32,
}

/// Attention of `group = queries.len() / head_dim` query heads over the
/// `f16` key/value rows `keys`/`values` (`[seq_len × head_dim]` each),
/// writing `[group × head_dim]` into `out`.
///
/// Bit-identical, head by head, to [`fused_attention_head_contiguous`] over
/// the widened rows (see the module docs).
///
/// # Errors
///
/// [`ModelError::ShapeMismatch`] when `head_dim` is zero, `queries` is not a
/// whole number of heads, `out` is shorter than `queries`, or `keys` /
/// `values` hold fewer than `seq_len * head_dim` elements.
pub fn attend_f16_group(
    keys: &[f16],
    values: &[f16],
    seq_len: usize,
    head_dim: usize,
    queries: &[f32],
    out: &mut [f32],
) -> ModelResult<()> {
    if head_dim == 0 || !queries.len().is_multiple_of(head_dim) {
        return Err(ModelError::ShapeMismatch {
            name: "attention queries".to_string(),
            expected: vec![head_dim],
            actual: vec![queries.len()],
        });
    }
    let group = queries.len() / head_dim;
    if out.len() < queries.len() {
        return Err(ModelError::ShapeMismatch {
            name: "attention output".to_string(),
            expected: vec![queries.len()],
            actual: vec![out.len()],
        });
    }
    let needed = seq_len.saturating_mul(head_dim);
    if keys.len() < needed || values.len() < needed {
        return Err(ModelError::ShapeMismatch {
            name: "attention history".to_string(),
            expected: vec![needed],
            actual: vec![keys.len().min(values.len())],
        });
    }
    let out = &mut out[..queries.len()];
    // The accumulators ARE the output rows: zero them first (the contiguous
    // kernel's accumulator starts at zero too, and a fully-masked / empty
    // history must yield zeros).
    out.fill(0.0);
    if seq_len == 0 || group == 0 {
        return Ok(());
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut states_stack: [HeadState; STACK_GROUP] = std::array::from_fn(|_| HeadState {
        max_val: f32::NEG_INFINITY,
        sum_exp: 0.0,
    });
    let mut states_heap: Vec<HeadState> = Vec::new();
    let states: &mut [HeadState] = if group <= STACK_GROUP {
        &mut states_stack[..group]
    } else {
        states_heap.extend((0..group).map(|_| HeadState {
            max_val: f32::NEG_INFINITY,
            sum_exp: 0.0,
        }));
        &mut states_heap
    };
    let mut scores_stack = [[0.0f32; ATTENTION_BLOCK_SIZE]; STACK_GROUP];
    let mut scores_heap: Vec<[f32; ATTENTION_BLOCK_SIZE]> = Vec::new();
    let scores: &mut [[f32; ATTENTION_BLOCK_SIZE]] = if group <= STACK_GROUP {
        &mut scores_stack[..group]
    } else {
        scores_heap.resize(group, [0.0f32; ATTENTION_BLOCK_SIZE]);
        &mut scores_heap
    };
    let mut row_stack = [0.0f32; STACK_HEAD_DIM];
    let mut row_heap: Vec<f32> = Vec::new();
    let row: &mut [f32] = if head_dim <= STACK_HEAD_DIM {
        &mut row_stack[..head_dim]
    } else {
        row_heap.resize(head_dim, 0.0);
        &mut row_heap
    };

    let mut pos = 0usize;
    while pos < seq_len {
        let block_end = (pos + ATTENTION_BLOCK_SIZE).min(seq_len);
        let block_len = block_end - pos;

        // Pass 1: scores, one widened key row shared by the whole group.
        for (i, t) in (pos..block_end).enumerate() {
            keys[t * head_dim..(t + 1) * head_dim].convert_to_f32_slice(row);
            for (h, head_scores) in scores.iter_mut().enumerate() {
                let q = &queries[h * head_dim..(h + 1) * head_dim];
                head_scores[i] = dot_f32(q, row) * scale;
            }
        }
        // One rescale per head per block, against the block's own max.
        for (h, state) in states.iter_mut().enumerate() {
            let block_max = scores[h][..block_len]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if block_max > state.max_val {
                if state.max_val != f32::NEG_INFINITY {
                    let rescale = (state.max_val - block_max).exp();
                    state.sum_exp *= rescale;
                    scale_f32(&mut out[h * head_dim..(h + 1) * head_dim], rescale);
                }
                state.max_val = block_max;
            }
        }
        // Pass 2: weighted values, one widened value row shared likewise.
        for (i, t) in (pos..block_end).enumerate() {
            values[t * head_dim..(t + 1) * head_dim].convert_to_f32_slice(row);
            for (h, state) in states.iter_mut().enumerate() {
                let exp_score = (scores[h][i] - state.max_val).exp();
                state.sum_exp += exp_score;
                axpy_f32(&mut out[h * head_dim..(h + 1) * head_dim], exp_score, row);
            }
        }
        pos = block_end;
    }

    for (h, state) in states.iter().enumerate() {
        if state.sum_exp > 0.0 {
            let inv_sum = 1.0 / state.sum_exp;
            scale_f32(&mut out[h * head_dim..(h + 1) * head_dim], inv_sum);
        }
    }
    Ok(())
}

impl KvCache {
    /// GQA attention of one query group against `(layer, kv_head)`'s cached
    /// history `0..seq_len`, read in place from whichever element type the
    /// cache stores (see the module docs).
    ///
    /// `queries` holds `group` query heads row-major (`[group × head_dim]`,
    /// the heads that share `kv_head`); `out` receives `[group × head_dim]`.
    ///
    /// # Errors
    ///
    /// * [`ModelError::ShapeMismatch`] — `layer` / `kv_head` out of range, or
    ///   `queries` / `out` not whole heads;
    /// * [`ModelError::SequenceTooLong`] — `seq_len` past the cache's limit;
    /// * [`ModelError::ShapeInvariant`] — `seq_len` past the **allocated**
    ///   capacity: those positions were never stored, and attending over
    ///   them would silently mix zeros into the softmax.
    pub fn attend_group(
        &self,
        layer: usize,
        kv_head: usize,
        seq_len: usize,
        queries: &[f32],
        out: &mut [f32],
    ) -> ModelResult<()> {
        if layer >= self.num_layers || kv_head >= self.num_kv_heads {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache attend (layer, kv_head)".to_string(),
                expected: vec![self.num_layers, self.num_kv_heads],
                actual: vec![layer, kv_head],
            });
        }
        if seq_len > self.max_seq_len {
            return Err(ModelError::SequenceTooLong {
                seq_len,
                max_ctx: self.max_seq_len,
            });
        }
        if seq_len > self.capacity {
            return Err(ModelError::ShapeInvariant {
                tensor: format!("kv_cache layer {layer} head {kv_head}"),
                expected: format!("seq_len <= allocated capacity {}", self.capacity),
                actual: format!("seq_len = {seq_len} (positions never stored)"),
            });
        }
        let head_dim = self.head_dim;
        if head_dim == 0 || !queries.len().is_multiple_of(head_dim) || out.len() < queries.len() {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache attend queries/output".to_string(),
                expected: vec![head_dim],
                actual: vec![queries.len(), out.len()],
            });
        }
        let start = self.cache_offset(layer, kv_head, 0);
        let end = start + seq_len * head_dim;
        match &self.storage {
            KvStorage::F32 { keys, values } => {
                let (keys, values) = (&keys[start..end], &values[start..end]);
                for (q, o) in queries
                    .chunks_exact(head_dim)
                    .zip(out.chunks_exact_mut(head_dim))
                {
                    fused_attention_head_contiguous(q, keys, values, o, seq_len, head_dim)?;
                }
                Ok(())
            }
            KvStorage::F16 { keys, values } => attend_f16_group(
                &keys[start..end],
                &values[start..end],
                seq_len,
                head_dim,
                queries,
                out,
            ),
        }
    }
}
