//! Blockwise, normalized Fast Walsh–Hadamard Transform (FWHT) with a fused
//! sign flip, for PrismML Bonsai 2's Hadamard-folded weights.
//!
//! # Why this exists
//!
//! 401 of the 402 weight matrices in every Bonsai 2 27B GGUF are
//! Hadamard-folded (`prism.hadamard.weight_names`): before a folded matmul
//! consumes an activation, the activation must first be rotated through a
//! **blockwise, normalized Hadamard transform** with a fused `±1` sign flip.
//! This module implements that transform (reference scalar + NEON + AVX2);
//! see `bonsai2-design.md` §2.2 (design source for this package, finding
//! K-07) for the surrounding architecture.
//!
//! # The transform
//!
//! `block_size` (`prism.hadamard.block_size`, always 1024 for Bonsai 2) is a
//! power of two. The folded widths (5120, 6144, 17408) are **not** powers of
//! two, so the transform runs independently over each `block_size`-wide
//! slice of a row — never across the whole row at once.
//!
//! Reference algorithm (`fork/models/ops.cpp:11910-11962`,
//! `ggml_compute_forward_fwht_impl`), transliterated:
//!
//! ```text
//! scale = 1.0 / sqrt(n)                 // applied FIRST, on load
//! for j in 0..n { dst[j] = src[j] * scale }
//! let mut len = 1
//! while len < n {
//!     for i in (0..n).step_by(2 * len) {
//!         for j in 0..len {
//!             let (u, v) = (dst[i + j], dst[i + len + j])
//!             dst[i + j] = u + v
//!             dst[i + len + j] = u - v
//!         }
//!     }
//!     len <<= 1
//! }
//! ```
//!
//! With the `1/sqrt(n)` normalization this transform is an **involution**
//! (`H_norm² = I`, since the unnormalized Sylvester–Hadamard matrix satisfies
//! `H·H = n·I`), so one kernel body serves both directions — only the
//! position of the sign multiply relative to the transform changes:
//!
//! | path | order | source |
//! |---|---|---|
//! | folded matmul (`attn_q/k/v`, `attn_qkv`, `attn_gate`, `ffn_*`, `output`) | `signs → FWHT → matmul` | `fork/src_llama-graph.cpp:1546-1582` (`build_lora_mm`) |
//! | token embedding lookup (inverse) | `lookup → FWHT → signs` | `fork/src_llama-graph.cpp:2420-2436` |
//!
//! [`fwht_forward_signed`] fuses the sign multiply into the same pass that
//! applies the `1/sqrt(block)` scale (the "initial load pass" in
//! `ops.cpp`'s own terms — a pass distinct from, and prior to, the butterfly
//! network itself); [`fwht_inverse_signed`] applies the scale-only load
//! pass, runs the butterfly network, and multiplies by the signs afterward.
//! [`fwht_in_place`] is the unsigned form (scale, then butterflies; no
//! signs), used directly by tests as the ground truth for the involution /
//! orthogonality / naive-matrix acceptance checks, and available to any
//! future caller that has already applied its own sign convention.
//!
//! `signs` is always exactly as wide as the activation it multiplies
//! (`prism.hadamard.sign_widths` gives one `±1` vector per width — 5120,
//! 6144 or 17408 entries — never one vector of `block_size` entries reused
//! across blocks), so every public function here requires
//! `signs.len() == ` the row width, and slices it `block`-wide per
//! iteration, once per `block`-wide chunk of the row.
//!
//! # What is explicitly *not* implemented here (recorded, not forgotten)
//!
//! 1. **`build_lora_mm`'s third argument.** The fork's `build_lora_mm(w, cur,
//!    w_s)` applies `res = res * w_s` **after** the matmul that consumes our
//!    FWHT output (`fork/src_llama-graph.cpp:1578-1582`) — a per-output-row
//!    scale, orthogonal to everything this module does (it never touches the
//!    activation FWHT sees). Bonsai 2's GGUFs carry no `*_s` tensors, so
//!    `w_s` is always `None` today; nothing in *this* module's API needs to
//!    change to support it later; a caller applies it as an independent
//!    elementwise multiply on the matmul's output.
//! 2. **Attention-internal Hadamard rotations.** The fork also rotates K/V
//!    (and, on some architectures, the attention output) through
//!    `llama_mul_mat_hadamard` *inside* attention, for a KV-cache-stored-
//!    in-rotated-basis feature (`fork/src_llama-graph.cpp:2904-2947`).
//!    Bonsai 2's GGUFs carry no metadata for this (the full
//!    `prism.hadamard.*` key set is `{version, block_size, transform, axis,
//!    sign_mode, weight_names, sign_widths, sign_values,
//!    inverse_weight_names, gdn_v_grouped}` — nothing about K/V rotation),
//!    so it is out of scope. Do not re-open this without new GGUF metadata
//!    to justify it.
//! 3. **V-head "tiled → grouped" regrouping and cross-matmul memoization.**
//!    The fork's `build_lora_mm` optionally permutes `ssm_out`'s input
//!    activation from tiled to grouped v-head order before the sign flip,
//!    and memoizes the rotated activation per `(activation, rot)` pair so
//!    q/k/v/gate (which read the *same* activation) transform it only once.
//!    Both concerns are **owned by `oxibonsai-model`'s hybrid block forward
//!    (design §3.3 `VHeadMap`, §3.4 `HadamardHook`/`HadamardScratch`), not by
//!    this crate.** That design deliberately avoids ever materializing a
//!    physical tiled/grouped permutation (`VHeadMap` re-indexes at slice
//!    time instead — "permute nothing"), and its memoization is
//!    caller-discipline-based (call [`fwht_forward_signed`] once per
//!    activation, reuse the output slice for every matmul that reads it),
//!    not a runtime `HashMap` keyed by pointer identity. A kernels-crate
//!    physical-permute or pointer-keyed-cache primitive would duplicate and
//!    contradict that design, so none is exposed here.
//!
//! # Dispatch
//!
//! Every public function picks NEON (`aarch64`, always available — no
//! runtime detection needed) or AVX2+FMA (`x86_64`, runtime-detected via
//! [`is_x86_feature_detected`]) when available, else pure scalar. Per
//! butterfly pass, the smallest lengths fall back to scalar even on a
//! SIMD-capable target — mirroring `ops.cpp`'s own two-phase split
//! (`:11928` scalar passes vs `:11950` SIMD passes) exactly, rather than
//! chasing a shuffle-based vectorization of passes that are already a small,
//! bounded fraction of an operation the design's own budget puts at "~2 % of
//! a 27B forward" (§2.2): NEON vectorizes every pass except `len == 1`
//! (9 of 10 passes for `block == 1024`); AVX2 vectorizes every pass except
//! `len ∈ {1, 2}` (8 of 10).

use crate::error::{KernelError, KernelResult};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Below this many rows, [`fwht_forward_signed_batch`] runs sequentially:
/// rayon's thread-spawn/join overhead outweighs the savings for a handful of
/// rows (e.g. a decode-time batch of 1-4 tokens). Only referenced from the
/// `not(wasm32)` branch (wasm32 has no rayon threads to spawn at all, so
/// there is no threshold to apply there), hence the matching `cfg`: without
/// it this constant is unused — and so `dead_code`-denied — on a wasm32
/// build.
#[cfg(not(target_arch = "wasm32"))]
const BATCH_PAR_MIN_ROWS: usize = 4;

// ═════════════════════════════════════════════════════════════════
//  Public API (bonsai2-design.md §2.2 / §2.7 / §7.2 — normative
//  signatures, reused verbatim by the future `PrismKernel` dispatch trait)
// ═════════════════════════════════════════════════════════════════

/// Unsigned, in-place, normalized blockwise FWHT: `x ← FWHT_block(x) /
/// sqrt(block)`, independently for each `block`-wide slice of `x`.
///
/// This is the involutory building block both signed variants below are
/// built from (`fwht_inverse_signed(fwht_forward_signed(x, s, b), s, b) ==
/// x`), and the direct target of the involution / orthogonality / vs-naive-
/// matrix acceptance tests.
///
/// # Errors
///
/// - [`KernelError::UnsupportedOperation`] if `block` is not a nonzero power
///   of two.
/// - [`KernelError::NotBlockAligned`] if `x.len()` is not a multiple of
///   `block`.
pub fn fwht_in_place(x: &mut [f32], block: usize) -> KernelResult<()> {
    validate_block(x.len(), block)?;
    let scale = inv_sqrt(block);
    for chunk in x.chunks_exact_mut(block) {
        scale_only(chunk, scale);
        butterfly_inplace(chunk);
    }
    Ok(())
}

/// Forward activation transform for a Hadamard-folded weight's input:
/// `x ← FWHT_block(x ⊙ signs) / sqrt(block)`, i.e. **signs, then FWHT**
/// (`fork/src_llama-graph.cpp:1568-1572`), independently per `block`-wide
/// slice of `x`. The sign multiply is fused into the same pass that applies
/// the `1/sqrt(block)` scale, so there is no extra pass over the buffer
/// beyond what the unsigned transform already does.
///
/// `signs` must be exactly as wide as `x` (one `±1` entry per element of the
/// activation, not one per `block`) — see the module docs.
///
/// # Errors
///
/// - [`KernelError::UnsupportedOperation`] if `block` is not a nonzero power
///   of two.
/// - [`KernelError::NotBlockAligned`] if `x.len()` is not a multiple of
///   `block`.
/// - [`KernelError::NamedDimensionMismatch`] if `signs.len() != x.len()`.
pub fn fwht_forward_signed(x: &mut [f32], signs: &[f32], block: usize) -> KernelResult<()> {
    validate_block(x.len(), block)?;
    require_matching_signs(x.len(), signs)?;
    let scale = inv_sqrt(block);
    forward_signed_row(x, signs, block, scale);
    Ok(())
}

/// Inverse transform for a Hadamard-folded token embedding row:
/// `x ← FWHT_block(x) / sqrt(block)`, then `x ← x ⊙ signs`, i.e.
/// **FWHT, then signs** (`fork/src_llama-graph.cpp:2427-2434`),
/// independently per `block`-wide slice of `x`.
///
/// `signs` must be exactly as wide as `x` — see the module docs.
///
/// # Errors
///
/// Same conditions as [`fwht_forward_signed`].
pub fn fwht_inverse_signed(x: &mut [f32], signs: &[f32], block: usize) -> KernelResult<()> {
    validate_block(x.len(), block)?;
    require_matching_signs(x.len(), signs)?;
    let scale = inv_sqrt(block);
    for (chunk, sign_chunk) in x.chunks_exact_mut(block).zip(signs.chunks_exact(block)) {
        scale_only(chunk, scale);
        butterfly_inplace(chunk);
        scale_and_signs(chunk, sign_chunk, 1.0);
    }
    Ok(())
}

/// Batched [`fwht_forward_signed`]: `rows` independent rows, each of width
/// `signs.len()`, packed contiguously in `x` (`x.len() == rows *
/// signs.len()`). The *same* `signs` vector (and the same fused
/// scale-and-sign load pass) is applied to every row — this is the shape
/// prefill needs, where one activation width is shared by many tokens.
///
/// Parallelized across rows with rayon once `rows` is large enough to be
/// worth the overhead (see `BATCH_PAR_MIN_ROWS`); falls back to a
/// sequential loop on `wasm32`, where rayon has no threads.
///
/// # Errors
///
/// - [`KernelError::UnsupportedOperation`] if `block` is not a nonzero power
///   of two, or if `rows * signs.len()` overflows `usize`.
/// - [`KernelError::NotBlockAligned`] if `signs.len()` is not a multiple of
///   `block`.
/// - [`KernelError::NamedDimensionMismatch`] if `x.len() != rows *
///   signs.len()`.
pub fn fwht_forward_signed_batch(
    x: &mut [f32],
    signs: &[f32],
    block: usize,
    rows: usize,
) -> KernelResult<()> {
    let width = signs.len();
    validate_block(width, block)?;
    let expected_len = rows.checked_mul(width).ok_or_else(|| {
        KernelError::UnsupportedOperation(format!(
            "hadamard batch: rows={rows} * width={width} overflows usize"
        ))
    })?;
    if x.len() != expected_len {
        return Err(KernelError::dimension_mismatch("x", expected_len, x.len()));
    }
    // `width == 0` or `rows == 0` ⇒ `expected_len == 0` ⇒ `x` is empty here
    // (the check above already rejected any non-empty `x`); nothing to do,
    // and `chunks_exact_mut(0)` / `par_chunks_mut(0)` would panic, so this
    // guard must come before either is called.
    if width == 0 || rows == 0 {
        return Ok(());
    }
    let scale = inv_sqrt(block);

    #[cfg(target_arch = "wasm32")]
    {
        for row in x.chunks_exact_mut(width) {
            forward_signed_row(row, signs, block, scale);
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if rows < BATCH_PAR_MIN_ROWS {
            for row in x.chunks_exact_mut(width) {
                forward_signed_row(row, signs, block, scale);
            }
        } else {
            x.par_chunks_mut(width).for_each(|row| {
                forward_signed_row(row, signs, block, scale);
            });
        }
    }
    Ok(())
}

// ═════════════════════════════════════════════════════════════════
//  Shared row-level driver
// ═════════════════════════════════════════════════════════════════

/// One row's worth of [`fwht_forward_signed`], factored out so the batch
/// entry point can call it per row without re-validating `signs.len()`
/// against a per-row width every time (the caller already validated it
/// once against the shared `signs` slice).
#[inline]
fn forward_signed_row(row: &mut [f32], signs: &[f32], block: usize, scale: f32) {
    for (chunk, sign_chunk) in row.chunks_exact_mut(block).zip(signs.chunks_exact(block)) {
        scale_and_signs(chunk, sign_chunk, scale);
        butterfly_inplace(chunk);
    }
}

#[inline]
fn inv_sqrt(block: usize) -> f32 {
    1.0_f32 / (block as f32).sqrt()
}

#[inline]
fn require_matching_signs(expected_len: usize, signs: &[f32]) -> KernelResult<()> {
    if signs.len() != expected_len {
        return Err(KernelError::dimension_mismatch(
            "signs",
            expected_len,
            signs.len(),
        ));
    }
    Ok(())
}

/// Validates that `block` is a legal block size (nonzero power of two —
/// `1` is legal: a 1-wide "transform" is the identity, used by the 2-element
/// ordering test's smaller sibling checks) and that `len` (a row or a shared
/// `signs` width) divides evenly into `block`-wide chunks.
fn validate_block(len: usize, block: usize) -> KernelResult<()> {
    if !block.is_power_of_two() {
        return Err(KernelError::UnsupportedOperation(format!(
            "hadamard transform requires a nonzero power-of-two block size, got {block}"
        )));
    }
    if !len.is_multiple_of(block) {
        return Err(KernelError::NotBlockAligned {
            count: len,
            block_size: block,
        });
    }
    Ok(())
}

// ═════════════════════════════════════════════════════════════════
//  Per-block engine: scale(+signs) load pass, then the butterfly network
// ═════════════════════════════════════════════════════════════════

/// Run the complete unnormalized FWHT butterfly network over `buf` in
/// place. `buf.len()` is always a power of two here (callers only ever pass
/// a `block`-wide chunk, and `block` was already validated).
#[inline]
fn butterfly_inplace(buf: &mut [f32]) {
    let n = buf.len();
    let mut len = 1usize;
    while len < n {
        run_one_pass(buf, len);
        len <<= 1;
    }
}

/// Run a single butterfly pass (fixed `len`) over the whole of `buf`,
/// dispatching to the widest available SIMD tier for this `len`, else
/// scalar. `len` is always a power of two and strictly less than
/// `buf.len()` (also a power of two), so `buf.len()` is always an exact
/// multiple of `2 * len` — every tier below relies on this and does no
/// remainder handling.
#[inline]
fn run_one_pass(buf: &mut [f32], len: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if len >= 4 {
            crate::simd_hadamard_neon::butterfly_pass_wide(buf, len);
        } else if len == 2 {
            crate::simd_hadamard_neon::butterfly_pass_len2(buf);
        } else {
            butterfly_pass_scalar(buf, len);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        if has_avx2 && len >= 8 {
            // SAFETY: `has_avx2` confirms AVX2+FMA are available.
            unsafe { butterfly_pass_wide_avx2(buf, len) };
        } else if has_avx2 && len == 4 {
            // SAFETY: `has_avx2` confirms AVX2+FMA are available.
            unsafe { butterfly_pass_len4_avx2(buf) };
        } else {
            butterfly_pass_scalar(buf, len);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        butterfly_pass_scalar(buf, len);
    }
}

/// Multiply every element of `buf` by `scale` (the `1/sqrt(block)` load
/// pass, unsigned form).
#[inline]
fn scale_only(buf: &mut [f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::simd_hadamard_neon::scale_inplace(buf, scale);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2+fma confirmed by the guard above.
            unsafe { scale_avx2(buf, scale) };
        } else {
            scale_scalar(buf, scale);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scale_scalar(buf, scale);
    }
}

/// `buf[i] ← buf[i] * signs[i] * scale` (the fused load pass for
/// [`fwht_forward_signed`], and — with `scale == 1.0` — the trailing
/// sign-only pass for [`fwht_inverse_signed`]).
///
/// `signs.len()` must be `>= buf.len()`; every call site slices `signs` to
/// exactly `buf.len()` via `chunks_exact` before calling this.
#[inline]
fn scale_and_signs(buf: &mut [f32], signs: &[f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::simd_hadamard_neon::scale_and_signs(buf, signs, scale);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2+fma confirmed by the guard above; `signs.len() >=
            // buf.len()` per this function's own precondition.
            unsafe { scale_and_signs_avx2(buf, signs, scale) };
        } else {
            scale_and_signs_scalar(buf, signs, scale);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scale_and_signs_scalar(buf, signs, scale);
    }
}

// ═════════════════════════════════════════════════════════════════
//  Scalar reference (ground truth; also the fallback for x86_64-without-
//  AVX2 and for every other architecture, e.g. wasm32/riscv)
// ═════════════════════════════════════════════════════════════════

#[allow(dead_code)] // unused when built for aarch64 (NEON handles scale/signs there)
#[inline]
fn scale_scalar(buf: &mut [f32], scale: f32) {
    for v in buf.iter_mut() {
        *v *= scale;
    }
}

#[allow(dead_code)] // unused when built for aarch64 (NEON handles scale/signs there)
#[inline]
fn scale_and_signs_scalar(buf: &mut [f32], signs: &[f32], scale: f32) {
    for (v, s) in buf.iter_mut().zip(signs.iter()) {
        *v = *v * *s * scale;
    }
}

/// One butterfly pass at a fixed `len`, matching `ops.cpp:11928-11939`
/// exactly (the reference's own "scalar passes" loop body).
#[inline]
fn butterfly_pass_scalar(buf: &mut [f32], len: usize) {
    let n = buf.len();
    let mut i = 0usize;
    while i < n {
        for j in 0..len {
            let u = buf[i + j];
            let v = buf[i + len + j];
            buf[i + j] = u + v;
            buf[i + len + j] = u - v;
        }
        i += 2 * len;
    }
}

// ═════════════════════════════════════════════════════════════════
//  x86_64 AVX2+FMA
// ═════════════════════════════════════════════════════════════════

/// # Safety
///
/// Caller must have confirmed AVX2 (+FMA) support via
/// `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_avx2(buf: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;

    let n = buf.len();
    let mut i = 0usize;
    if n >= 8 {
        let s = _mm256_set1_ps(scale);
        while i + 8 <= n {
            let v = _mm256_loadu_ps(buf.as_ptr().add(i));
            _mm256_storeu_ps(buf.as_mut_ptr().add(i), _mm256_mul_ps(v, s));
            i += 8;
        }
    }
    for v in &mut buf[i..n] {
        *v *= scale;
    }
}

/// # Safety
///
/// Caller must have confirmed AVX2 (+FMA) support via
/// `is_x86_feature_detected!`, and `signs.len() >= buf.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_and_signs_avx2(buf: &mut [f32], signs: &[f32], scale: f32) {
    use std::arch::x86_64::*;

    let n = buf.len();
    let mut i = 0usize;
    if n >= 8 {
        let s = _mm256_set1_ps(scale);
        while i + 8 <= n {
            let v = _mm256_loadu_ps(buf.as_ptr().add(i));
            let sg = _mm256_loadu_ps(signs.as_ptr().add(i));
            let r = _mm256_mul_ps(_mm256_mul_ps(v, sg), s);
            _mm256_storeu_ps(buf.as_mut_ptr().add(i), r);
            i += 8;
        }
    }
    for j in i..n {
        buf[j] = buf[j] * signs[j] * scale;
    }
}

/// A single butterfly pass, `len >= 8` (a multiple of 8, since `len` is
/// itself a power of two `>= 8`): every group of `2 * len` elements is
/// processed as plain, contiguous 8-wide loads — exactly `ops.cpp`'s own
/// "SIMD passes" section (`:11947-11962`).
///
/// # Safety
///
/// Caller must have confirmed AVX2 (+FMA) support via
/// `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn butterfly_pass_wide_avx2(buf: &mut [f32], len: usize) {
    use std::arch::x86_64::*;

    let n = buf.len();
    let mut i = 0usize;
    while i < n {
        let mut j = 0usize;
        while j < len {
            let u = _mm256_loadu_ps(buf.as_ptr().add(i + j));
            let v = _mm256_loadu_ps(buf.as_ptr().add(i + len + j));
            _mm256_storeu_ps(buf.as_mut_ptr().add(i + j), _mm256_add_ps(u, v));
            _mm256_storeu_ps(buf.as_mut_ptr().add(i + len + j), _mm256_sub_ps(u, v));
            j += 8;
        }
        i += 2 * len;
    }
}

/// The `len == 4` butterfly pass: one group (`2 * len == 8` elements) fills
/// exactly one 256-bit register, so the pass is a 128-bit lane split
/// (`u` = low 128 bits, `v` = high 128 bits) instead of a strided load —
/// the AVX2 analogue of NEON's `len == 2` special case.
///
/// # Safety
///
/// Caller must have confirmed AVX2 (+FMA) support via
/// `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn butterfly_pass_len4_avx2(buf: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = buf.len();
    let mut i = 0usize;
    while i < n {
        // `v8 = [x0..x3 (u), x4..x7 (v)]`, one full group.
        let v8 = _mm256_loadu_ps(buf.as_ptr().add(i));
        let u4 = _mm256_castps256_ps128(v8); // low 128 bits: [x0,x1,x2,x3]
        let v4 = _mm256_extractf128_ps(v8, 1); // high 128 bits: [x4,x5,x6,x7]
        let sum = _mm_add_ps(u4, v4); // [x0+x4, .., x3+x7]
        let diff = _mm_sub_ps(u4, v4); // [x0-x4, .., x3-x7]
                                       // low 128 bits <- sum, high 128 bits <- diff.
        let result = _mm256_insertf128_ps(_mm256_castps128_ps256(sum), diff, 1);
        _mm256_storeu_ps(buf.as_mut_ptr().add(i), result);
        i += 8;
    }
}

// ═════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let scale = x.abs().max(y.abs()).max(1.0);
            assert!(
                diff <= tol * scale,
                "{label} mismatch at [{i}]: {x} vs {y} (diff={diff})"
            );
        }
    }

    fn lcg_random_vec(len: usize, seed: u64) -> Vec<f32> {
        // Deterministic, dependency-free PRNG (xorshift64*) so tests need no
        // external `rand` crate and are fully reproducible.
        let mut state = seed.max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to a signed, non-degenerate f32 range.
                (((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 8.0
            })
            .collect()
    }

    fn random_signs(len: usize, seed: u64) -> Vec<f32> {
        lcg_random_vec(len, seed)
            .into_iter()
            .map(|v| if v >= 0.0 { 1.0 } else { -1.0 })
            .collect()
    }

    // ── validation ──────────────────────────────────────────────

    #[test]
    fn rejects_non_power_of_two_block() {
        let mut x = vec![0.0f32; 12];
        assert!(fwht_in_place(&mut x, 3).is_err());
    }

    #[test]
    fn rejects_zero_block() {
        let mut x = vec![0.0f32; 12];
        assert!(fwht_in_place(&mut x, 0).is_err());
    }

    #[test]
    fn rejects_misaligned_length() {
        let mut x = vec![0.0f32; 10];
        // 10 is not a multiple of 8.
        assert!(fwht_in_place(&mut x, 8).is_err());
    }

    #[test]
    fn rejects_signs_length_mismatch_forward() {
        let mut x = vec![0.0f32; 8];
        let signs = vec![1.0f32; 4];
        assert!(fwht_forward_signed(&mut x, &signs, 4).is_err());
    }

    #[test]
    fn rejects_signs_length_mismatch_inverse() {
        let mut x = vec![0.0f32; 8];
        let signs = vec![1.0f32; 4];
        assert!(fwht_inverse_signed(&mut x, &signs, 4).is_err());
    }

    #[test]
    fn batch_rejects_x_length_mismatch() {
        let signs = vec![1.0f32; 4];
        let mut x = vec![0.0f32; 7]; // not 2*4
        assert!(fwht_forward_signed_batch(&mut x, &signs, 4, 2).is_err());
    }

    #[test]
    fn empty_input_is_a_no_op() {
        let mut x: Vec<f32> = vec![];
        assert!(fwht_in_place(&mut x, 1024).is_ok());
        assert!(x.is_empty());
    }

    #[test]
    fn batch_zero_rows_with_empty_x_is_ok() {
        let signs = vec![1.0f32; 4];
        let mut x: Vec<f32> = vec![];
        assert!(fwht_forward_signed_batch(&mut x, &signs, 4, 0).is_ok());
    }

    #[test]
    fn batch_empty_signs_with_empty_x_is_ok() {
        let signs: Vec<f32> = vec![];
        let mut x: Vec<f32> = vec![];
        assert!(fwht_forward_signed_batch(&mut x, &signs, 4, 5).is_ok());
    }

    // ── involution / orthogonality (design §8.2, §7.2) ─────────

    #[test]
    fn involution_single_block_1024() {
        let original = lcg_random_vec(1024, 42);
        let mut x = original.clone();
        fwht_in_place(&mut x, 1024).expect("valid block");
        fwht_in_place(&mut x, 1024).expect("valid block");
        assert_close(&x, &original, 1e-5, "involution block=1024");
    }

    #[test]
    fn involution_multi_block_widths() {
        // The three real folded widths: 5120 = 5*1024, 6144 = 6*1024,
        // 17408 = 17*1024.
        for &width in &[5120usize, 6144, 17408] {
            let original = lcg_random_vec(width, width as u64);
            let mut x = original.clone();
            fwht_in_place(&mut x, 1024).expect("valid block");
            fwht_in_place(&mut x, 1024).expect("valid block");
            assert_close(&x, &original, 1e-5, &format!("involution width={width}"));
        }
    }

    #[test]
    fn involution_small_blocks() {
        for &block in &[1usize, 2, 4, 8, 16, 32] {
            let original = lcg_random_vec(block * 3, block as u64 + 100);
            let mut x = original.clone();
            fwht_in_place(&mut x, block).expect("valid block");
            fwht_in_place(&mut x, block).expect("valid block");
            assert_close(&x, &original, 1e-5, &format!("involution block={block}"));
        }
    }

    #[test]
    fn orthogonality_preserves_l2_norm() {
        for &block in &[2usize, 4, 8, 64, 1024] {
            let x = lcg_random_vec(block, block as u64 + 7);
            let mut y = x.clone();
            fwht_in_place(&mut y, block).expect("valid block");
            let norm_in: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let norm_out: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm_in - norm_out).abs() <= 1e-5 * norm_in.max(1.0),
                "block={block}: norm_in={norm_in} norm_out={norm_out}"
            );
        }
    }

    // ── vs a naive O(n^2) Sylvester-Hadamard matrix (design §7.2: "vs a
    //    naive O(n^2) H*x/sqrt(n)"; also serves as the from-first-principles
    //    "golden parity vs the fork" check for the transform itself — see
    //    tests/hadamard_golden.rs for the full 64-row version) ──────

    /// Independent O(n²) oracle: evaluates the Sylvester-Hadamard matrix
    /// directly from its closed form (`H[i][j] = (-1)^popcount(i & j)`,
    /// natural/Hadamard order — the same order the in-place butterfly
    /// network produces) rather than reusing any butterfly code, so this is
    /// a true from-first-principles cross-check, not the implementation
    /// tested against itself.
    fn naive_hadamard_transform(x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let scale = 1.0_f32 / (n as f32).sqrt();
        (0..n)
            .map(|i| {
                let mut acc = 0.0f32;
                for (j, &xj) in x.iter().enumerate() {
                    let sign = if ((i & j).count_ones()) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    acc += sign * xj;
                }
                acc * scale
            })
            .collect()
    }

    #[test]
    fn matches_naive_hadamard_matrix_n16() {
        let x = lcg_random_vec(16, 99);
        let mut y = x.clone();
        fwht_in_place(&mut y, 16).expect("valid block");
        let expected = naive_hadamard_transform(&x);
        assert_close(&y, &expected, 1e-5, "vs naive H*x/sqrt(n), n=16");
    }

    // ── forward/inverse round-trip (design §7.2) ────────────────

    #[test]
    fn forward_inverse_round_trip_1024() {
        let original = lcg_random_vec(1024, 7);
        let signs = random_signs(1024, 1234);
        let mut x = original.clone();
        fwht_forward_signed(&mut x, &signs, 1024).expect("valid inputs");
        fwht_inverse_signed(&mut x, &signs, 1024).expect("valid inputs");
        assert_close(&x, &original, 1e-5, "forward/inverse round-trip");
    }

    #[test]
    fn forward_inverse_round_trip_small_blocks() {
        for &block in &[2usize, 4, 8, 32] {
            let original = lcg_random_vec(block * 2, block as u64 + 555);
            let signs = random_signs(block * 2, block as u64 + 999);
            let mut x = original.clone();
            fwht_forward_signed(&mut x, &signs, block).expect("valid inputs");
            fwht_inverse_signed(&mut x, &signs, block).expect("valid inputs");
            assert_close(&x, &original, 1e-5, &format!("round-trip block={block}"));
        }
    }

    // ── ordering BY CONSTRUCTION on a 2-element block (design §8.2 / JSON
    //    spec acceptance): forward is signs-then-FWHT, inverse is
    //    FWHT-then-signs. `s0 != s1` makes the two possible orderings
    //    diverge numerically, so this is a real discriminator, not just a
    //    round-trip check that could pass even with both directions
    //    consistently (and wrongly) swapped. ─────────────────────

    #[test]
    fn forward_ordering_is_signs_then_fwht_on_2_element_block() {
        let a = 3.0f32;
        let b = 1.0f32;
        let s0 = 1.0f32;
        let s1 = -1.0f32;
        let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();

        // Correct (signs first): a'=a*s0, b'=b*s1, then FWHT(a',b').
        let a_signed = a * s0;
        let b_signed = b * s1;
        let correct = [
            (a_signed + b_signed) * inv_sqrt2,
            (a_signed - b_signed) * inv_sqrt2,
        ];

        // Wrong (FWHT first, signs after) — must NOT be what we compute.
        let h0 = (a + b) * inv_sqrt2;
        let h1 = (a - b) * inv_sqrt2;
        let wrong = [h0 * s0, h1 * s1];

        let mut x = [a, b];
        fwht_forward_signed(&mut x, &[s0, s1], 2).expect("2-element block is valid");

        assert_close(&x, &correct, 1e-5, "forward must be signs-then-FWHT");
        let matches_wrong = (x[0] - wrong[0]).abs() < 1e-5 && (x[1] - wrong[1]).abs() < 1e-5;
        assert!(
            !matches_wrong,
            "forward output matches the WRONG (FWHT-then-signs) ordering: {x:?}"
        );
    }

    #[test]
    fn inverse_ordering_is_fwht_then_signs_on_2_element_block() {
        let a = 3.0f32;
        let b = 1.0f32;
        let s0 = 1.0f32;
        let s1 = -1.0f32;
        let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();

        // Correct (FWHT first, signs after).
        let h0 = (a + b) * inv_sqrt2;
        let h1 = (a - b) * inv_sqrt2;
        let correct = [h0 * s0, h1 * s1];

        // Wrong (signs first, then FWHT) — must NOT be what we compute.
        let a_signed = a * s0;
        let b_signed = b * s1;
        let wrong = [
            (a_signed + b_signed) * inv_sqrt2,
            (a_signed - b_signed) * inv_sqrt2,
        ];

        let mut x = [a, b];
        fwht_inverse_signed(&mut x, &[s0, s1], 2).expect("2-element block is valid");

        assert_close(&x, &correct, 1e-5, "inverse must be FWHT-then-signs");
        let matches_wrong = (x[0] - wrong[0]).abs() < 1e-5 && (x[1] - wrong[1]).abs() < 1e-5;
        assert!(
            !matches_wrong,
            "inverse output matches the WRONG (signs-then-FWHT) ordering: {x:?}"
        );
    }

    // ── batch vs per-row (parallel path must match the sequential one) ──

    #[test]
    fn batch_matches_sequential_per_row() {
        let rows = 9usize; // >= BATCH_PAR_MIN_ROWS, exercises the rayon path
        let width = 1024usize;
        let signs = random_signs(width, 4242);

        let mut flat = lcg_random_vec(rows * width, 31337);
        let mut expected = flat.clone();

        fwht_forward_signed_batch(&mut flat, &signs, 1024, rows).expect("valid batch");
        for row in expected.chunks_exact_mut(width) {
            fwht_forward_signed(row, &signs, 1024).expect("valid row");
        }

        assert_close(&flat, &expected, 1e-6, "batch vs sequential per-row");
    }

    #[test]
    fn batch_small_row_count_uses_sequential_path() {
        let rows = 2usize; // below BATCH_PAR_MIN_ROWS
        let width = 8usize;
        let signs = random_signs(width, 17);

        let mut flat = lcg_random_vec(rows * width, 271828);
        let mut expected = flat.clone();

        fwht_forward_signed_batch(&mut flat, &signs, 4, rows).expect("valid batch");
        for row in expected.chunks_exact_mut(width) {
            fwht_forward_signed(row, &signs, 4).expect("valid row");
        }

        assert_close(&flat, &expected, 1e-6, "small batch vs sequential per-row");
    }

    // ── SIMD tier vs scalar reference, every pass-length boundary ───

    #[test]
    fn scale_only_dispatch_matches_scalar_every_length() {
        for len in 0..40usize {
            let input = lcg_random_vec(len, len as u64 + 3);
            let mut dispatched = input.clone();
            let mut scalar = input;
            scale_only(&mut dispatched, 0.03125);
            scale_scalar(&mut scalar, 0.03125);
            assert_close(&dispatched, &scalar, 1e-6, &format!("scale_only len={len}"));
        }
    }

    #[test]
    fn scale_and_signs_dispatch_matches_scalar_every_length() {
        for len in 0..40usize {
            let input = lcg_random_vec(len, len as u64 + 11);
            let signs = random_signs(len, len as u64 + 12);
            let mut dispatched = input.clone();
            let mut scalar = input;
            scale_and_signs(&mut dispatched, &signs, 0.125);
            scale_and_signs_scalar(&mut scalar, &signs, 0.125);
            assert_close(
                &dispatched,
                &scalar,
                1e-6,
                &format!("scale_and_signs len={len}"),
            );
        }
    }

    #[test]
    fn butterfly_dispatch_matches_scalar_every_block_size() {
        for &block in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 1024] {
            let input = lcg_random_vec(block, block as u64 + 500);
            let mut dispatched = input.clone();
            let mut scalar = input;
            butterfly_inplace(&mut dispatched);

            let mut len = 1usize;
            while len < scalar.len() {
                butterfly_pass_scalar(&mut scalar, len);
                len <<= 1;
            }

            assert_close(
                &dispatched,
                &scalar,
                1e-4,
                &format!("butterfly_inplace vs scalar, block={block}"),
            );
        }
    }
}
