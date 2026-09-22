//! Fused attention: memory-efficient single-pass attention computation.
//!
//! Standard attention materializes the full `seq_len x seq_len` attention
//! matrix, requiring O(seq_len^2) memory. For long sequences this dominates
//! memory usage and thrashes caches.
//!
//! This module implements **online softmax** (flash-attention v2 style) to
//! compute attention output in a single pass over KV positions, using only
//! O(block_size) working memory regardless of sequence length.
//!
//! **Algorithm (two passes per block, flash-attention v2 form):**
//! ```text
//! For each block of KV positions:
//!   1. Compute QK^T scores for the block (SIMD dot product)
//!   2. Take the block's max score; rescale the output accumulator ONCE
//!      if it exceeds the running max (SIMD scale)
//!   3. Accumulate all of the block's weighted V contributions against the
//!      now-final running max (SIMD axpy) -- no further rescale this block
//! ```
//!
//! The final output is mathematically identical to standard attention
//! (within floating-point tolerance), but uses constant working memory and,
//! for `head_dim <= MAX_HEAD_DIM`, zero heap allocations.

use crate::error::{ModelError, ModelResult};
use crate::layers::attention::CausalMask;

/// Number of KV positions processed per attention block.
/// 32 positions x head_dim floats fits comfortably in L1 cache.
pub const ATTENTION_BLOCK_SIZE: usize = 32;

/// Upper bound on `head_dim` for the zero-allocation fast path.
///
/// Covers every architecture this crate ships today: Qwen3 (64/128) and
/// Bonsai 2 (256, `qwen35.attention.key_length` / `value_length`). A
/// `head_dim` above this bound is still computed correctly — it just falls
/// back to a one-time heap allocation per call instead of a stack array —
/// so correctness never depends on the bound, only the zero-alloc guarantee
/// does.
const MAX_HEAD_DIM: usize = 256;

// ─── SIMD dot product ─────────────────────────────────────────────────────

/// Compute `dot(a, b)` over the common prefix `a.len().min(b.len())`, with
/// SIMD acceleration where available.
///
/// Dispatches at runtime to:
/// - NEON (aarch64): dual-accumulator 8-wide vfmaq_f32
/// - AVX2+FMA (x86_64): dual-accumulator 16-wide _mm256_fmadd_ps
/// - Scalar fallback for all other targets
///
/// # Length contract (K-M1 item c)
///
/// All three implementations stop at `min(a.len(), b.len())`. They used to
/// disagree: the scalar path already clamped, but the NEON and AVX2 paths
/// drove the loop from `a.len()` alone and dereferenced `b.as_ptr().add(i)`
/// unchecked, so an `a` longer than `b` was an out-of-bounds **read** in
/// release — and the guard here was a `debug_assert_eq!`, compiled out
/// exactly where it mattered. That was reachable: the public entry points of
/// this module deliberately accept `query.len() > head_dim` (they validate
/// with `<`, and [`OnlineSoftmaxState::accumulate`] asserts `>=`), and they
/// passed the whole `query` here against a `head_dim`-length key slice.
/// Those entry points now also narrow `query` to `head_dim` before scoring,
/// so the equal-length case is what the hot path actually takes; this clamp
/// is the backstop that makes the function sound for any caller.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: neon feature confirmed above
            return unsafe { dot_f32_neon(a, b) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx") {
            // SAFETY: avx+fma features confirmed above
            return unsafe { dot_f32_avx2_fma(a, b) };
        }
    }

    dot_f32_scalar(a, b)
}

/// Scalar dot product with 4-way ILP accumulation.
#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += a[base] * b[base];
        acc1 += a[base + 1] * b[base + 1];
        acc2 += a[base + 2] * b[base + 2];
        acc3 += a[base + 3] * b[base + 3];
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    for i in (len - remainder)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// NEON dot product: dual 128-bit accumulators = 8 floats/iter.
///
/// # Safety
///
/// Caller must have confirmed the `neon` target feature. The loop is bounded
/// by `a.len().min(b.len())`, so the unchecked `add(i)` loads stay in bounds
/// for both operands regardless of their relative lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 8 <= n {
        let a0 = vld1q_f32(a.as_ptr().add(i));
        let a1 = vld1q_f32(a.as_ptr().add(i + 4));
        let b0 = vld1q_f32(b.as_ptr().add(i));
        let b1 = vld1q_f32(b.as_ptr().add(i + 4));
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        i += 8;
    }
    let mut tail = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    tail
}

/// AVX2+FMA dot product: dual 256-bit accumulators = 16 floats/iter.
///
/// # Safety
///
/// Caller must have confirmed the `avx` and `fma` target features. The loop
/// is bounded by `a.len().min(b.len())`, so the unchecked `add(i)` loads stay
/// in bounds for both operands regardless of their relative lengths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 16 <= n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }
    // Horizontal sum of acc0 + acc1
    let combined = _mm256_add_ps(acc0, acc1);
    let lo = _mm256_castps256_ps128(combined);
    let hi = _mm256_extractf128_ps(combined, 1);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sum2);
    let sum1 = _mm_add_ss(sum2, shuf2);
    let mut tail = _mm_cvtss_f32(sum1);
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    tail
}

// ─── SIMD axpy / scale (K-M1 item a: V-accumulate and rescale) ───────────

/// Compute `acc[i] += scale * src[i]` for `i` in `0..acc.len().min(src.len())`
/// (BLAS "saxpy"), with SIMD acceleration where available.
///
/// This is the online-softmax V-accumulation step: it has exactly as many
/// FLOPs as [`dot_f32`]'s QK reduction, so it gets the same runtime
/// dispatch instead of a plain scalar loop.
#[inline]
pub fn axpy_f32(acc: &mut [f32], scale: f32, src: &[f32]) {
    let n = acc.len().min(src.len());
    let acc = &mut acc[..n];
    let src = &src[..n];

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: neon feature confirmed above; acc/src truncated to equal length.
            unsafe { axpy_f32_neon(acc, scale, src) };
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx") {
            // SAFETY: avx+fma features confirmed above; acc/src truncated to equal length.
            unsafe { axpy_f32_avx2_fma(acc, scale, src) };
            return;
        }
    }

    axpy_f32_scalar(acc, scale, src);
}

/// Scalar saxpy fallback.
#[inline]
fn axpy_f32_scalar(acc: &mut [f32], scale: f32, src: &[f32]) {
    for (a, &s) in acc.iter_mut().zip(src.iter()) {
        *a += scale * s;
    }
}

/// NEON saxpy: dual 128-bit FMA accumulators = 8 floats/iter.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_f32_neon(acc: &mut [f32], scale: f32, src: &[f32]) {
    use core::arch::aarch64::*;
    let n = acc.len();
    let scale_v = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 8 <= n {
        let a0 = vld1q_f32(acc.as_ptr().add(i));
        let a1 = vld1q_f32(acc.as_ptr().add(i + 4));
        let s0 = vld1q_f32(src.as_ptr().add(i));
        let s1 = vld1q_f32(src.as_ptr().add(i + 4));
        vst1q_f32(acc.as_mut_ptr().add(i), vfmaq_f32(a0, scale_v, s0));
        vst1q_f32(acc.as_mut_ptr().add(i + 4), vfmaq_f32(a1, scale_v, s1));
        i += 8;
    }
    while i < n {
        acc[i] += scale * src[i];
        i += 1;
    }
}

/// AVX2+FMA saxpy: dual 256-bit FMA accumulators = 16 floats/iter.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn axpy_f32_avx2_fma(acc: &mut [f32], scale: f32, src: &[f32]) {
    use core::arch::x86_64::*;
    let n = acc.len();
    let scale_v = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 16 <= n {
        let a0 = _mm256_loadu_ps(acc.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(acc.as_ptr().add(i + 8));
        let s0 = _mm256_loadu_ps(src.as_ptr().add(i));
        let s1 = _mm256_loadu_ps(src.as_ptr().add(i + 8));
        _mm256_storeu_ps(acc.as_mut_ptr().add(i), _mm256_fmadd_ps(scale_v, s0, a0));
        _mm256_storeu_ps(
            acc.as_mut_ptr().add(i + 8),
            _mm256_fmadd_ps(scale_v, s1, a1),
        );
        i += 16;
    }
    while i < n {
        acc[i] += scale * src[i];
        i += 1;
    }
}

/// Compute `acc[i] *= factor` in place (BLAS "sscal"), with SIMD
/// acceleration where available. Shares [`axpy_f32`]'s dispatch tiers.
///
/// This is the online-softmax output-accumulator rescale: with the
/// flash-attention v2 restructuring it runs **at most once per block**
/// (previously: once per new running max within the block, up to
/// [`ATTENTION_BLOCK_SIZE`] times), so unlike the old code it is no longer
/// worth leaving scalar even though it now runs far less often.
#[inline]
pub fn scale_f32(acc: &mut [f32], factor: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: neon feature confirmed above
            unsafe { scale_f32_neon(acc, factor) };
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx") {
            // SAFETY: avx feature confirmed above (fma is not needed for a
            // plain multiply, but gating on the same pair as the other
            // x86_64 helpers in this file keeps dispatch uniform).
            unsafe { scale_f32_avx2(acc, factor) };
            return;
        }
    }

    scale_f32_scalar(acc, factor);
}

/// Scalar sscal fallback.
#[inline]
fn scale_f32_scalar(acc: &mut [f32], factor: f32) {
    for a in acc.iter_mut() {
        *a *= factor;
    }
}

/// NEON sscal: dual 128-bit multiply = 8 floats/iter.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_f32_neon(acc: &mut [f32], factor: f32) {
    use core::arch::aarch64::*;
    let n = acc.len();
    let factor_v = vdupq_n_f32(factor);
    let mut i = 0;
    while i + 8 <= n {
        let a0 = vld1q_f32(acc.as_ptr().add(i));
        let a1 = vld1q_f32(acc.as_ptr().add(i + 4));
        vst1q_f32(acc.as_mut_ptr().add(i), vmulq_f32(a0, factor_v));
        vst1q_f32(acc.as_mut_ptr().add(i + 4), vmulq_f32(a1, factor_v));
        i += 8;
    }
    while i < n {
        acc[i] *= factor;
        i += 1;
    }
}

/// AVX2 sscal: dual 256-bit multiply = 16 floats/iter.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn scale_f32_avx2(acc: &mut [f32], factor: f32) {
    use core::arch::x86_64::*;
    let n = acc.len();
    let factor_v = _mm256_set1_ps(factor);
    let mut i = 0;
    while i + 16 <= n {
        let a0 = _mm256_loadu_ps(acc.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(acc.as_ptr().add(i + 8));
        _mm256_storeu_ps(acc.as_mut_ptr().add(i), _mm256_mul_ps(a0, factor_v));
        _mm256_storeu_ps(acc.as_mut_ptr().add(i + 8), _mm256_mul_ps(a1, factor_v));
        i += 16;
    }
    while i < n {
        acc[i] *= factor;
        i += 1;
    }
}

// ─── Zero-allocation output accumulator (K-M1 item c) ─────────────────────

/// The online-softmax output accumulator: a fixed `[f32; MAX_HEAD_DIM]`
/// stack array for every `head_dim` this crate ships today, falling back to
/// a heap `Vec` only for a hypothetical future `head_dim > MAX_HEAD_DIM`.
///
/// Replaces the old `vec![0.0f32; head_dim]` allocated once per head per
/// token (24 heads x 16 full-attention layers x every token for Bonsai 2).
///
/// `Stack` is deliberately ~1 KiB against `Heap`'s ~24 bytes: the entire
/// point is a large inline buffer for the common case so it never touches
/// the allocator. Boxing `Stack` (clippy's usual suggestion for this lint)
/// would itself be a heap allocation on every call and defeat the point of
/// this type, so the size asymmetry is accepted rather than "fixed".
#[allow(clippy::large_enum_variant)]
enum Accum {
    Stack([f32; MAX_HEAD_DIM]),
    Heap(Vec<f32>),
}

impl Accum {
    #[inline]
    fn new(head_dim: usize) -> Self {
        if head_dim <= MAX_HEAD_DIM {
            Accum::Stack([0.0f32; MAX_HEAD_DIM])
        } else {
            Accum::Heap(vec![0.0f32; head_dim])
        }
    }

    #[inline]
    fn as_slice(&self, head_dim: usize) -> &[f32] {
        match self {
            Accum::Stack(a) => &a[..head_dim],
            Accum::Heap(v) => &v[..head_dim],
        }
    }

    #[inline]
    fn as_mut_slice(&mut self, head_dim: usize) -> &mut [f32] {
        match self {
            Accum::Stack(a) => &mut a[..head_dim],
            Accum::Heap(v) => &mut v[..head_dim],
        }
    }
}

// ─── Online Softmax State ──────────────────────────────────────────────

/// Running state for numerically stable online softmax computation.
///
/// Maintains the running maximum, exponential sum, and weighted output
/// accumulator across blocks of KV positions. The key insight is that
/// when we encounter a new maximum, we can rescale the previous
/// accumulator without needing to revisit past positions.
struct OnlineSoftmaxState {
    /// Running maximum of attention scores seen so far.
    max_val: f32,
    /// Running sum of exp(score - max_val) for all scores seen so far.
    sum_exp: f32,
    /// Dimension of each V vector; needed to slice `output` (which may be
    /// backed by an over-sized fixed array — see [`Accum`]).
    head_dim: usize,
    /// Running weighted sum of V vectors: sum(softmax_weight * V).
    /// Needs rescaling when max_val changes.
    output: Accum,
}

impl OnlineSoftmaxState {
    /// Create a new state for the given head dimension.
    fn new(head_dim: usize) -> Self {
        Self {
            max_val: f32::NEG_INFINITY,
            sum_exp: 0.0,
            head_dim,
            output: Accum::new(head_dim),
        }
    }

    #[inline]
    fn output_slice(&self) -> &[f32] {
        self.output.as_slice(self.head_dim)
    }

    #[inline]
    fn output_mut_slice(&mut self) -> &mut [f32] {
        self.output.as_mut_slice(self.head_dim)
    }

    /// Pass 1 of a block: given the block's own raw max score, rescale the
    /// output accumulator **once** if it exceeds the running max —
    /// flash-attention v2's restructuring of what used to be a rescale on
    /// every new max encountered while scanning the block's scores one at a
    /// time (K-M1 item b).
    ///
    /// The very first block that contributes anything (`self.max_val` still
    /// `NEG_INFINITY`) needs no rescale at all: `Accum::new` already
    /// zero-initialized the accumulator (both the `Stack` and `Heap`
    /// variants), and `sum_exp` starts at `0.0`, so there is nothing yet to
    /// scale. Skipping the redundant `fill(0.0)` here means `Accum::new`'s
    /// construction-time zero is now LOAD-BEARING, not merely a safe
    /// default: for a mask that disallows every position for a given query
    /// (`SlidingWindowConfig { window_size: 0, .. }`), this method is never
    /// called at all, and the output comes ONLY from that construction-time
    /// zero. `fully_masked_query_yields_zero_output` pins exactly that —
    /// if `Accum::new` ever stops zero-filling, that test (not this one)
    /// is what goes red.
    fn rescale_for_block_max(&mut self, block_max: f32) {
        if block_max > self.max_val {
            if self.max_val != f32::NEG_INFINITY {
                let rescale = (self.max_val - block_max).exp();
                self.sum_exp *= rescale;
                scale_f32(self.output_mut_slice(), rescale);
            }
            self.max_val = block_max;
        }
    }

    /// Pass 2 of a block, one score/value pair: accumulate against the
    /// running max, which `rescale_for_block_max` has already finalized for
    /// this block (K-M1 item a: SIMD axpy instead of a scalar loop).
    fn accumulate(&mut self, score: f32, v: &[f32]) {
        // `axpy_f32` truncates to `acc.len().min(src.len())`, so a `v`
        // shorter than `head_dim` would silently produce a partial
        // accumulation instead of a loud failure. The two contiguous entry
        // points always pass an exact `head_dim`-length slice by
        // construction; `fused_attention_head` accepts caller-supplied
        // references and is the one path this can actually catch. `>=`
        // (not `==`): the public API tolerates over-long `query`/`output`
        // slices throughout, so a `v` longer than `head_dim` is legitimate.
        debug_assert!(
            v.len() >= self.head_dim,
            "OnlineSoftmaxState::accumulate: v.len()={} shorter than head_dim={}",
            v.len(),
            self.head_dim
        );
        let exp_score = (score - self.max_val).exp();
        self.sum_exp += exp_score;
        axpy_f32(self.output_mut_slice(), exp_score, v);
    }

    /// Finalize the output by dividing by the total softmax denominator.
    ///
    /// After this call, `self.output` contains the correct attention output.
    fn finalize(&mut self) {
        if self.sum_exp > 0.0 {
            let inv_sum = 1.0 / self.sum_exp;
            scale_f32(self.output_mut_slice(), inv_sum);
        }
    }
}

// ─── Fused attention ───────────────────────────────────────────────────

/// Fused attention for a single query head against KV cache.
///
/// Computes `output = softmax(Q @ K^T / sqrt(d)) @ V` without
/// materializing the full attention matrix.
///
/// This is a drop-in replacement for [`super::attention::attention_head`]
/// with the same semantics but O(ATTENTION_BLOCK_SIZE) working memory
/// instead of O(seq_len).
///
/// # Arguments
/// - `query`: Query vector `[head_dim]`.
/// - `keys`: Slice of key vector references, one per sequence position.
/// - `values`: Slice of value vector references, one per sequence position.
/// - `head_dim`: Dimension of each head.
/// - `output`: Output buffer `[head_dim]`.
///
/// # Errors
/// Returns `ModelError` if dimensions are inconsistent.
pub fn fused_attention_head(
    query: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
    head_dim: usize,
    output: &mut [f32],
) -> ModelResult<()> {
    if query.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "query".to_string(),
            expected: vec![head_dim],
            actual: vec![query.len()],
        });
    }
    if output.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "output".to_string(),
            expected: vec![head_dim],
            actual: vec![output.len()],
        });
    }
    if keys.len() != values.len() {
        return Err(ModelError::ShapeMismatch {
            name: "keys/values length".to_string(),
            expected: vec![keys.len()],
            actual: vec![values.len()],
        });
    }

    let seq_len = keys.len();
    if seq_len == 0 {
        for d in output.iter_mut() {
            *d = 0.0;
        }
        return Ok(());
    }

    // K-M1 item (c): narrow `query` to exactly `head_dim`. The validation
    // above accepts an over-long query (`<`, not `!=`), and every scoring
    // call below pairs it with a `head_dim`-length key, so scoring the whole
    // slice would either read past the key or silently fold extra query
    // dimensions into the score depending on the SIMD tier.
    let query = &query[..head_dim];

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut state = OnlineSoftmaxState::new(head_dim);

    // Process KV positions in blocks of at most ATTENTION_BLOCK_SIZE,
    // scoring and accumulating straight from `keys`/`values` (already O(1)
    // indexable reference slices) with no intermediate Vec.
    let mut pos = 0;
    while pos < seq_len {
        let block_end = (pos + ATTENTION_BLOCK_SIZE).min(seq_len);
        let block_len = block_end - pos;
        let mut block_scores = [0.0f32; ATTENTION_BLOCK_SIZE];

        for (score_slot, t) in block_scores.iter_mut().zip(pos..block_end) {
            *score_slot = dot_f32(query, keys[t]) * scale;
        }

        let block_max = block_scores[..block_len]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        state.rescale_for_block_max(block_max);

        for (&score, t) in block_scores[..block_len].iter().zip(pos..block_end) {
            state.accumulate(score, values[t]);
        }

        pos = block_end;
    }

    state.finalize();

    // Copy result to output
    output[..head_dim].copy_from_slice(state.output_slice());

    Ok(())
}

/// Fused attention with contiguous KV buffers (matching existing API).
///
/// This variant accepts contiguous row-major key/value buffers
/// `[seq_len x head_dim]`, matching the layout used by
/// [`super::attention::attention_head`].
///
/// # Arguments
/// - `query`: Query vector `[head_dim]`.
/// - `keys`: Contiguous key buffer `[seq_len x head_dim]` (row-major).
/// - `values`: Contiguous value buffer `[seq_len x head_dim]` (row-major).
/// - `output`: Output buffer `[head_dim]`.
/// - `seq_len`: Number of KV positions.
/// - `head_dim`: Dimension per head.
pub fn fused_attention_head_contiguous(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) -> ModelResult<()> {
    if query.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "query".to_string(),
            expected: vec![head_dim],
            actual: vec![query.len()],
        });
    }
    if keys.len() < seq_len * head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "keys".to_string(),
            expected: vec![seq_len * head_dim],
            actual: vec![keys.len()],
        });
    }
    if values.len() < seq_len * head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "values".to_string(),
            expected: vec![seq_len * head_dim],
            actual: vec![values.len()],
        });
    }
    if output.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "output".to_string(),
            expected: vec![head_dim],
            actual: vec![output.len()],
        });
    }

    if seq_len == 0 {
        for d in output.iter_mut() {
            *d = 0.0;
        }
        return Ok(());
    }

    // K-M1 item (c): narrow `query` to exactly `head_dim`. The validation
    // above accepts an over-long query (`<`, not `!=`), and every scoring
    // call below pairs it with a `head_dim`-length key, so scoring the whole
    // slice would either read past the key or silently fold extra query
    // dimensions into the score depending on the SIMD tier.
    let query = &query[..head_dim];

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut state = OnlineSoftmaxState::new(head_dim);

    // Process in blocks, indexing the contiguous KV buffers directly by
    // position -- no intermediate Vec<&[f32]> of scattered slice
    // references (that indirection defeats prefetch on what is otherwise a
    // sequential scan).
    let mut pos = 0;
    while pos < seq_len {
        let block_end = (pos + ATTENTION_BLOCK_SIZE).min(seq_len);
        let block_len = block_end - pos;
        let mut block_scores = [0.0f32; ATTENTION_BLOCK_SIZE];

        for (score_slot, t) in block_scores.iter_mut().zip(pos..block_end) {
            let k_slice = &keys[t * head_dim..(t + 1) * head_dim];
            *score_slot = dot_f32(query, k_slice) * scale;
        }

        let block_max = block_scores[..block_len]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        state.rescale_for_block_max(block_max);

        for (&score, t) in block_scores[..block_len].iter().zip(pos..block_end) {
            let v_slice = &values[t * head_dim..(t + 1) * head_dim];
            state.accumulate(score, v_slice);
        }

        pos = block_end;
    }

    state.finalize();
    output[..head_dim].copy_from_slice(state.output_slice());

    Ok(())
}

/// Fused attention with contiguous KV buffers and causal masking.
///
/// Like [`fused_attention_head_contiguous`] but applies a [`CausalMask`]
/// (optionally with sliding window) to skip masked positions in the
/// online softmax accumulation.
///
/// Masked positions (where `!mask.is_allowed(query_pos, t)`) are silently
/// skipped — they contribute neither a score nor a V accumulation — giving
/// numerically identical results to applying `f32::NEG_INFINITY` before
/// softmax, but without allocating a score buffer.
///
/// # Arguments
/// - `query`: Query vector `[head_dim]`.
/// - `keys`: Contiguous key buffer `[seq_len x head_dim]` (row-major).
/// - `values`: Contiguous value buffer `[seq_len x head_dim]` (row-major).
/// - `output`: Output buffer `[head_dim]`.
/// - `seq_len`: Number of KV positions.
/// - `head_dim`: Dimension per head.
/// - `query_pos`: Position of the current query token in the full sequence.
/// - `mask`: Causal mask (with optional sliding window).
///
/// # Errors
/// Returns `ModelError` if dimensions are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn fused_attention_head_contiguous_with_mask(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    query_pos: usize,
    mask: &CausalMask,
) -> ModelResult<()> {
    if query.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "query".to_string(),
            expected: vec![head_dim],
            actual: vec![query.len()],
        });
    }
    if keys.len() < seq_len * head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "keys".to_string(),
            expected: vec![seq_len * head_dim],
            actual: vec![keys.len()],
        });
    }
    if values.len() < seq_len * head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "values".to_string(),
            expected: vec![seq_len * head_dim],
            actual: vec![values.len()],
        });
    }
    if output.len() < head_dim {
        return Err(ModelError::ShapeMismatch {
            name: "output".to_string(),
            expected: vec![head_dim],
            actual: vec![output.len()],
        });
    }

    if seq_len == 0 {
        for d in output.iter_mut() {
            *d = 0.0;
        }
        return Ok(());
    }

    // K-M1 item (c): narrow `query` to exactly `head_dim`. The validation
    // above accepts an over-long query (`<`, not `!=`), and every scoring
    // call below pairs it with a `head_dim`-length key, so scoring the whole
    // slice would either read past the key or silently fold extra query
    // dimensions into the score depending on the SIMD tier.
    let query = &query[..head_dim];

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut state = OnlineSoftmaxState::new(head_dim);

    // Process in blocks; skip masked positions per block. Allowed
    // positions within a block are tracked in a fixed-size stack array
    // (rather than a `Vec<&[f32]>` of scattered references) so accumulate
    // can still index the contiguous KV buffers directly by position.
    let mut pos = 0;
    while pos < seq_len {
        let block_end = (pos + ATTENTION_BLOCK_SIZE).min(seq_len);
        let mut block_scores = [0.0f32; ATTENTION_BLOCK_SIZE];
        let mut block_positions = [0usize; ATTENTION_BLOCK_SIZE];
        let mut count = 0usize;

        for t in pos..block_end {
            if !mask.is_allowed(query_pos, t) {
                continue;
            }
            let k_slice = &keys[t * head_dim..(t + 1) * head_dim];
            block_scores[count] = dot_f32(query, k_slice) * scale;
            block_positions[count] = t;
            count += 1;
        }

        if count > 0 {
            let block_max = block_scores[..count]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            state.rescale_for_block_max(block_max);

            for (&score, &t) in block_scores[..count]
                .iter()
                .zip(block_positions[..count].iter())
            {
                let v_slice = &values[t * head_dim..(t + 1) * head_dim];
                state.accumulate(score, v_slice);
            }
        }

        pos = block_end;
    }

    state.finalize();
    output[..head_dim].copy_from_slice(state.output_slice());

    Ok(())
}

// ─── Vectorized utilities ──────────────────────────────────────────────

/// In-place softmax with numerical stability.
///
/// Uses the max-subtraction trick to prevent overflow:
/// `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`.
///
/// This is a standalone utility that can be used outside fused attention.
pub fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }

    // Find maximum for numerical stability
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Exponentiate and sum
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in logits.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Fused scaled dot product: `dot(q, k) * scale`.
///
/// Computes the dot product of `q` and `k`, then multiplies by `scale`.
/// This is the QK^T / sqrt(d) operation that forms attention scores.
///
/// Uses SIMD via [`dot_f32`] where available.
#[inline]
pub fn scaled_dot_product(q: &[f32], k: &[f32], scale: f32) -> f32 {
    dot_f32(q, k) * scale
}

#[cfg(test)]
#[path = "attention_fused_tests.rs"]
mod tests;
