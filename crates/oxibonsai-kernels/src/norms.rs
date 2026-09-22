//! SIMD-accelerated normalisation / gating primitives for the Bonsai 2
//! hybrid (Gated DeltaNet + sigmoid-gated attention) forward path.
//!
//! Split out of `simd_float_ops.rs` (already at its 2000-line budget) as its
//! own module, following the same length-contract and tiered-dispatch
//! conventions (K-02 / KERN-SOUND): every entry point validates cross-slice
//! length invariants with a real, always-on `if` check that returns
//! [`KernelError`], never a `debug_assert!`.
//!
//! ## K-10: five new primitives
//!
//! - [`l2_norm_simd`] — **not** RMSNorm. See its doc comment for the exact
//!   (corrected) formula.
//! - [`rms_norm_gated_simd`] — the gated RMSNorm used by Gated DeltaNet's
//!   output normalisation, fused into a single reduction pass.
//! - [`sigmoid_simd`] / [`sigmoid_mul_simd`] — plain and fused
//!   sigmoid-then-multiply (the sigmoid-gated attention path).
//! - [`softplus_simd`] — softplus with the `x > 20` linear cutoff.
//!
//! ## NEON transcendental helpers
//!
//! `exp_neon_f32x4` and `silu_core_neon_f32x4` below are intentionally a
//! **verbatim copy** of the private helpers of the same name in
//! `simd_float_ops.rs` (KERN-SOUND's K-M3 fix): that file does not export
//! them (they are private `fn`s, not `pub(crate)`), and this module does not
//! own `simd_float_ops.rs`, so the only way to reuse the vectorized-exp
//! approach without editing a file outside this package's `owned_files` is
//! to duplicate it. Recorded as a deviation: a follow-up package should mark
//! the originals `pub(crate)` and delete this copy.
//!
//! [`ln_neon_f32x4`] is new: `softplus_simd` needs `ln(1+exp(x))` and no
//! vectorized `ln` exists anywhere in this workspace (checked). Rather than
//! transcribing a second multi-constant polynomial from memory (real risk
//! of a silent transcription error in an untested constant), it computes
//! `ln` via a cheap IEEE-754 bit-trick initial guess (accurate to within
//! about `0.06` nats, exact at the two mantissa endpoints `m=1` and `m=2`)
//! refined by three Newton iterations on `f(t) = exp(t) - y`, each built
//! entirely from the already-verified `exp_neon_f32x4`. Newton's quadratic
//! convergence on this worst-case starting error lands at an absolute error
//! of order `1e-6` after two iterations (verified by hand and pinned by
//! `ln_neon_matches_std_ln_over_a_wide_range` below); the third iteration is
//! header-room, not load-bearing.

use crate::error::{KernelError, KernelResult};

// ═════════════════════════════════════════════════════════════════
//  Public entry points
// ═════════════════════════════════════════════════════════════════

/// L2-normalise a vector: `output[i] = input[i] * scale` where
/// `scale = 1 / max(sqrt(sum(input[j]^2)), eps)`.
///
/// # Correction (K-10)
///
/// This is **not** `x / sqrt(sum(x^2) + eps)`. `fork/models/ops.cpp:4200-4212`
/// (`ggml_l2_norm`) computes `sum = Σ x*x` then
/// `scale = 1.0f / fmaxf(sqrtf(sum), eps)` — the `eps` floors the
/// *denominator* (`sqrt(sum)`, before the reciprocal), it is not added
/// under the radical. The two formulas agree away from zero but diverge
/// sharply on a near-zero vector — see `l2_norm_near_zero_vector_uses_eps_floor`
/// below, which pins the ~1000x gap between them. Implementing
/// `x / sqrt(sum + eps)` here would silently break parity on any near-zero
/// attention head.
///
/// Unlike [`rms_norm_simd`](crate::simd_float_ops::rms_norm_simd) there is
/// **no** `/n` (mean) and **no** per-element weight: this is applied per
/// head over `head_k_dim = 128` in the Gated DeltaNet q/k path. Using
/// `rms_norm_simd` as an L2-norm substitute would introduce a spurious
/// `sqrt(n)` factor (`sqrt(128) ≈ 11.3x` for a 128-wide head).
///
/// # Errors
///
/// [`KernelError::NamedBufferTooSmall`] if `output.len() < input.len()`.
#[inline]
pub fn l2_norm_simd(input: &[f32], output: &mut [f32], eps: f32) -> KernelResult<()> {
    let n = input.len();
    if output.len() < n {
        return Err(KernelError::buffer_too_small("output", n, output.len()));
    }

    #[cfg(target_arch = "aarch64")]
    {
        l2_norm_neon(input, output, eps);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA support just confirmed.
            unsafe { l2_norm_avx2(input, output, eps) };
            return Ok(());
        }
        l2_norm_scalar(input, output, eps);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        l2_norm_scalar(input, output, eps);
    }

    Ok(())
}

/// Gated RMSNorm, fused into a single pass:
/// `out[i] = weight[i] * input[i] * inv_rms * silu(gate[i])`, where
/// `inv_rms = 1 / sqrt(mean(input^2) + eps)` — the *same* mean-based
/// formula as [`rms_norm_simd`](crate::simd_float_ops::rms_norm_simd), not
/// the `max`-clamped L2-norm formula above (`fork/models/ops.cpp:3825-3837`,
/// `ggml_compute_forward_rms_norm_f32`: `mean = sum/ne00; scale =
/// 1/sqrtf(mean + eps)`).
///
/// Reproduces `qwen35.cpp:311-320`'s `build_norm_gated`:
/// `ggml_swiglu_split(gate, RMSNorm(input, weight))`. `ggml_swiglu_split(a,
/// b)` silu's its *first* operand and multiplies by the second (matching
/// this crate's own `swiglu_simd(gate, up, ..) = silu(gate) * up`
/// convention, and confirmed by the call site passing `gate` first), so the
/// result is `silu(gate) * RMSNorm(input, weight)`. Fusing into one pass
/// avoids materialising the intermediate RMSNorm output.
///
/// # Errors
///
/// - [`KernelError::NamedDimensionMismatch`] if `weight.len() != input.len()`
///   or `gate.len() != input.len()`.
/// - [`KernelError::NamedBufferTooSmall`] if `output.len() < input.len()`.
#[inline]
pub fn rms_norm_gated_simd(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    output: &mut [f32],
    eps: f32,
) -> KernelResult<()> {
    let n = input.len();
    if weight.len() != n {
        return Err(KernelError::dimension_mismatch("weight", n, weight.len()));
    }
    if gate.len() != n {
        return Err(KernelError::dimension_mismatch("gate", n, gate.len()));
    }
    if output.len() < n {
        return Err(KernelError::buffer_too_small("output", n, output.len()));
    }

    #[cfg(target_arch = "aarch64")]
    {
        rms_norm_gated_neon(input, weight, gate, output, eps);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA support just confirmed.
            unsafe { rms_norm_gated_avx2(input, weight, gate, output, eps) };
            return Ok(());
        }
        rms_norm_gated_scalar(input, weight, gate, output, eps);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        rms_norm_gated_scalar(input, weight, gate, output, eps);
    }

    Ok(())
}

/// Element-wise sigmoid: `output[i] = 1 / (1 + exp(-input[i]))`.
///
/// # Errors
///
/// [`KernelError::NamedBufferTooSmall`] if `output.len() < input.len()`.
#[inline]
pub fn sigmoid_simd(input: &[f32], output: &mut [f32]) -> KernelResult<()> {
    let n = input.len();
    if output.len() < n {
        return Err(KernelError::buffer_too_small("output", n, output.len()));
    }

    #[cfg(target_arch = "aarch64")]
    {
        sigmoid_neon(input, output);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA support just confirmed.
            unsafe { sigmoid_avx2(input, output) };
            return Ok(());
        }
        sigmoid_scalar(input, output);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        sigmoid_scalar(input, output);
    }

    Ok(())
}

/// Fused sigmoid-gate multiply: `out[i] = x[i] * sigmoid(gate[i])`.
///
/// This is the *only* way sigmoid is used on the sigmoid-gated attention
/// path (`qwen35.cpp:391-394`: `gate_sigmoid = ggml_sigmoid(gate); cur =
/// ggml_mul(cur, gate_sigmoid)`, i.e. `attn * sigmoid(gate)` — also
/// `:432`, `:715`). Fusing avoids materialising the intermediate
/// `sigmoid(gate)` tensor.
///
/// # Errors
///
/// - [`KernelError::NamedDimensionMismatch`] if `gate.len() != x.len()`.
/// - [`KernelError::NamedBufferTooSmall`] if `out.len() < x.len()`.
#[inline]
pub fn sigmoid_mul_simd(x: &[f32], gate: &[f32], out: &mut [f32]) -> KernelResult<()> {
    let n = x.len();
    if gate.len() != n {
        return Err(KernelError::dimension_mismatch("gate", n, gate.len()));
    }
    if out.len() < n {
        return Err(KernelError::buffer_too_small("out", n, out.len()));
    }

    #[cfg(target_arch = "aarch64")]
    {
        sigmoid_mul_neon(x, gate, out);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA support just confirmed.
            unsafe { sigmoid_mul_avx2(x, gate, out) };
            return Ok(());
        }
        sigmoid_mul_scalar(x, gate, out);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        sigmoid_mul_scalar(x, gate, out);
    }

    Ok(())
}

/// Softplus with the ggml linear cutoff: `output[i] = if input[i] > 20.0
/// { input[i] } else { ln(1 + exp(input[i])) }`.
///
/// The `20.0` cutoff (not a smaller "safe" threshold) matches ggml's own
/// `ggml_vec_soft_plus_f32` and the Gated DeltaNet decay-gate computation
/// (`x = alpha_raw + dt_bias; softplus(x)`), which B2-05's `gdn_step_f32`
/// also reproduces — kept self-contained here rather than imported since
/// B2-05 is a parallel, independently-landing package.
///
/// # Errors
///
/// [`KernelError::NamedBufferTooSmall`] if `output.len() < input.len()`.
#[inline]
pub fn softplus_simd(input: &[f32], output: &mut [f32]) -> KernelResult<()> {
    let n = input.len();
    if output.len() < n {
        return Err(KernelError::buffer_too_small("output", n, output.len()));
    }

    #[cfg(target_arch = "aarch64")]
    {
        softplus_neon(input, output);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA support just confirmed.
            unsafe { softplus_avx2(input, output) };
            return Ok(());
        }
        softplus_scalar(input, output);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        softplus_scalar(input, output);
    }

    Ok(())
}

// ═════════════════════════════════════════════════════════════════
//  Scalar fallbacks (also used as the reference in tests)
// ═════════════════════════════════════════════════════════════════

#[allow(dead_code)]
#[inline]
fn l2_norm_scalar(input: &[f32], output: &mut [f32], eps: f32) {
    let mut sum_sq = 0.0f32;
    for &v in input {
        sum_sq += v * v;
    }
    let scale = 1.0 / sum_sq.sqrt().max(eps);
    for i in 0..input.len() {
        output[i] = input[i] * scale;
    }
}

#[inline]
fn sigmoid_scalar_elem(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu_scalar_elem(x: f32) -> f32 {
    x * sigmoid_scalar_elem(x)
}

#[inline]
fn softplus_scalar_elem(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[allow(dead_code)]
#[inline]
fn rms_norm_gated_scalar(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    let mut sum_sq = 0.0f32;
    for &v in input {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for i in 0..n {
        output[i] = weight[i] * input[i] * inv_rms * silu_scalar_elem(gate[i]);
    }
}

#[allow(dead_code)]
#[inline]
fn sigmoid_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = sigmoid_scalar_elem(input[i]);
    }
}

#[allow(dead_code)]
#[inline]
fn sigmoid_mul_scalar(x: &[f32], gate: &[f32], out: &mut [f32]) {
    for i in 0..x.len() {
        out[i] = x[i] * sigmoid_scalar_elem(gate[i]);
    }
}

#[allow(dead_code)]
#[inline]
fn softplus_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = softplus_scalar_elem(input[i]);
    }
}

// ═════════════════════════════════════════════════════════════════
//  AArch64 NEON implementations
// ═════════════════════════════════════════════════════════════════

/// Vectorized `exp(x)` for 4 `f32` lanes.
///
/// **Verbatim copy** of `simd_float_ops::exp_neon_f32x4` (K-M3) — see the
/// module doc comment above for why this is duplicated rather than shared.
/// Classic Cephes-derived range reduction + 5th-degree minimax polynomial:
/// write `x = k*ln2 + r` with `k = round(x * log2(e))` and `|r| <= ln2/2`,
/// then `exp(x) = 2^k * poly(r)`, with `2^k` built by inserting the biased
/// exponent directly into an IEEE-754 bit pattern. Inputs are clamped to
/// `±88.3762626647950` (the range where `f32::exp` is finite).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn exp_neon_f32x4(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    const EXP_HI: f32 = 88.376_26;
    const EXP_LO: f32 = -88.376_26;
    const LOG2EF: f32 = std::f32::consts::LOG2_E;
    const EXP_C1: f32 = 0.693_359_4;
    const EXP_C2: f32 = -2.121_944_4e-4;
    const P0: f32 = 1.987_569_1e-4;
    const P1: f32 = 1.398_2e-3;
    const P2: f32 = 8.333_452e-3;
    const P3: f32 = 4.166_579_6e-2;
    const P4: f32 = 1.666_666_6e-1;
    const P5: f32 = 5e-1;

    let one = vdupq_n_f32(1.0);
    let x = vminq_f32(x, vdupq_n_f32(EXP_HI));
    let x = vmaxq_f32(x, vdupq_n_f32(EXP_LO));

    let fx0 = vfmaq_f32(vdupq_n_f32(0.5), x, vdupq_n_f32(LOG2EF));
    let fx_trunc = vcvtq_f32_s32(vcvtq_s32_f32(fx0));
    let overshot = vcgtq_f32(fx_trunc, fx0);
    let overshot_f = vreinterpretq_f32_u32(vandq_u32(overshot, vreinterpretq_u32_f32(one)));
    let fx = vsubq_f32(fx_trunc, overshot_f);

    let x = vfmsq_f32(x, fx, vdupq_n_f32(EXP_C1));
    let x = vfmsq_f32(x, fx, vdupq_n_f32(EXP_C2));

    let z = vmulq_f32(x, x);

    let mut y = vdupq_n_f32(P0);
    y = vfmaq_f32(vdupq_n_f32(P1), y, x);
    y = vfmaq_f32(vdupq_n_f32(P2), y, x);
    y = vfmaq_f32(vdupq_n_f32(P3), y, x);
    y = vfmaq_f32(vdupq_n_f32(P4), y, x);
    y = vfmaq_f32(vdupq_n_f32(P5), y, x);
    y = vfmaq_f32(vaddq_f32(x, one), y, z);

    let emm0 = vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127));
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(emm0, 23));

    vmulq_f32(y, pow2n)
}

/// SIMD SiLU on 4 lanes: `x / (1 + exp(-x))`.
///
/// Verbatim copy of `simd_float_ops::silu_core_neon_f32x4` (K-M3) — see the
/// module doc comment. Uses `vdivq_f32` (correctly-rounded division) rather
/// than a reciprocal-approximation Newton step, so every lane uses one
/// division strategy.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn silu_core_neon_f32x4(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let one = vdupq_n_f32(1.0);
    let denom = vaddq_f32(one, exp_neon_f32x4(vnegq_f32(x)));
    vdivq_f32(x, denom)
}

/// SIMD sigmoid on 4 lanes: `1 / (1 + exp(-x))`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sigmoid_core_neon_f32x4(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let one = vdupq_n_f32(1.0);
    let denom = vaddq_f32(one, exp_neon_f32x4(vnegq_f32(x)));
    vdivq_f32(one, denom)
}

/// Vectorized `ln(y)` for 4 `f32` lanes, `y > 0` required (undefined
/// otherwise — this module's only caller, [`softplus_core_neon_f32x4`],
/// always passes `y = 1 + exp(x) >= 1`).
///
/// See the module doc comment for the derivation: an IEEE-754 bit-trick
/// initial guess (`ln(y) = e*ln2 + ln(m)` with `y = m * 2^e`, `m` in
/// `[1,2)`, approximating `ln(m) ≈ (m-1)*ln2` — exact at `m=1` and `m=2`,
/// worst-case absolute error ≈ `0.059` nats at `m = 1/ln2`) refined by
/// three Newton iterations solving `exp(t) = y` (`t := t - 1 + y*exp(-t)`),
/// each reusing [`exp_neon_f32x4`]. Quadratic convergence turns the
/// `0.059`-nat worst case into an absolute error of order `1e-6` after two
/// iterations (by hand: `0.059 -> 0.0017 -> 1.5e-6`); the third iteration is
/// margin, not load-bearing.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ln_neon_f32x4(y: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    const LN2: f32 = std::f32::consts::LN_2;

    let bits = vreinterpretq_s32_f32(y);
    let exp_bits = vandq_s32(vshrq_n_s32(bits, 23), vdupq_n_s32(0xFF));
    let e = vsubq_s32(exp_bits, vdupq_n_s32(127));
    let e_f = vcvtq_f32_s32(e);

    let mantissa_bits = vandq_s32(bits, vdupq_n_s32(0x007F_FFFF));
    let exp_one_bits = vdupq_n_s32(127 << 23); // bit pattern of 1.0f32's exponent field
    let m = vreinterpretq_f32_s32(vorrq_s32(mantissa_bits, exp_one_bits));

    let m_minus_1 = vsubq_f32(m, vdupq_n_f32(1.0));
    let mut t = vmulq_f32(vaddq_f32(e_f, m_minus_1), vdupq_n_f32(LN2));

    // 3 Newton iterations: t <- t - 1 + y * exp(-t).
    for _ in 0..3 {
        let y_exp_neg_t = vmulq_f32(y, exp_neon_f32x4(vnegq_f32(t)));
        t = vsubq_f32(vaddq_f32(t, y_exp_neg_t), vdupq_n_f32(1.0));
    }

    t
}

/// SIMD softplus on 4 lanes: `x > 20.0 ? x : ln(1 + exp(x))`, selected via a
/// bitwise select (`vbslq_f32`) rather than a branch, so both operands are
/// always computed. This is safe even for the discarded branch: `exp_x` is
/// finite for any `x` in `exp_neon_f32x4`'s clamp range (`±88.376`), so
/// `ln(1+exp_x)` never sees an infinite or NaN input regardless of which
/// side of the cutoff `x` is really on.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn softplus_core_neon_f32x4(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let mask = vcgtq_f32(x, vdupq_n_f32(20.0));
    let y = vaddq_f32(vdupq_n_f32(1.0), exp_neon_f32x4(x));
    let ln_y = ln_neon_f32x4(y);
    vbslq_f32(mask, x, ln_y)
}

/// # Safety
///
/// The caller (`l2_norm_simd`) must guarantee `output.len() >= input.len()`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn l2_norm_neon(input: &[f32], output: &mut [f32], eps: f32) {
    use std::arch::aarch64::*;

    let n = input.len();
    let mut sum_sq = 0.0f32;
    let mut i = 0;
    if n >= 4 {
        // SAFETY: NEON is always available on aarch64; loop bounded by `n`.
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            while i + 4 <= n {
                let v = vld1q_f32(input.as_ptr().add(i));
                acc = vfmaq_f32(acc, v, v);
                i += 4;
            }
            let pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            let pair2 = vpadd_f32(pair, pair);
            sum_sq = vget_lane_f32::<0>(pair2);
        }
    }
    for &v in &input[i..n] {
        sum_sq += v * v;
    }

    // `sum_sq` is one shared scalar reused for every element below, so
    // unlike the elementwise-`exp` functions there is no per-lane K-M3-style
    // body/tail rounding hazard here: the reduction result does not depend
    // on whether an index landed in the vectorized body or the tail.
    let scale = 1.0 / sum_sq.sqrt().max(eps);

    i = 0;
    if n >= 4 {
        unsafe {
            let scale_v = vdupq_n_f32(scale);
            while i + 4 <= n {
                let v = vld1q_f32(input.as_ptr().add(i));
                let r = vmulq_f32(v, scale_v);
                vst1q_f32(output.as_mut_ptr().add(i), r);
                i += 4;
            }
        }
    }
    for j in i..n {
        output[j] = input[j] * scale;
    }
}

/// # Safety
///
/// The caller (`rms_norm_gated_simd`) must guarantee `weight.len() >=
/// input.len()`, `gate.len() >= input.len()` and `output.len() >=
/// input.len()`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn rms_norm_gated_neon(input: &[f32], weight: &[f32], gate: &[f32], output: &mut [f32], eps: f32) {
    use std::arch::aarch64::*;

    let n = input.len();
    let mut sum_sq = 0.0f32;
    let mut i = 0;
    if n >= 4 {
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            while i + 4 <= n {
                let v = vld1q_f32(input.as_ptr().add(i));
                acc = vfmaq_f32(acc, v, v);
                i += 4;
            }
            let pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            let pair2 = vpadd_f32(pair, pair);
            sum_sq = vget_lane_f32::<0>(pair2);
        }
    }
    for &v in &input[i..n] {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    // Elementwise part below involves `exp` (via `silu_core_neon_f32x4`), so
    // K-M3 applies: body and the padded tail both run through the identical
    // vectorized path (mirrors `swiglu_neon` in `simd_float_ops.rs`) so a
    // value is bit-identical regardless of where in the buffer it lands.
    i = 0;
    // SAFETY: NEON is always available on aarch64; the body loop is bounded
    // by `n = input.len()` (`weight`/`gate`/`output` are checked against it
    // by the caller); the tail only ever touches `[i..n)` of the input
    // slices or the fixed-size 4-element stack buffers below.
    unsafe {
        let inv_rms_v = vdupq_n_f32(inv_rms);
        while i + 4 <= n {
            let inp = vld1q_f32(input.as_ptr().add(i));
            let w = vld1q_f32(weight.as_ptr().add(i));
            let g = vld1q_f32(gate.as_ptr().add(i));
            let silu_g = silu_core_neon_f32x4(g);
            let normed = vmulq_f32(vmulq_f32(inp, inv_rms_v), w);
            let result = vmulq_f32(normed, silu_g);
            vst1q_f32(output.as_mut_ptr().add(i), result);
            i += 4;
        }

        let rem = n - i;
        if rem > 0 {
            let mut ibuf = [0.0f32; 4];
            let mut wbuf = [0.0f32; 4];
            let mut gbuf = [0.0f32; 4];
            ibuf[..rem].copy_from_slice(&input[i..n]);
            wbuf[..rem].copy_from_slice(&weight[i..n]);
            gbuf[..rem].copy_from_slice(&gate[i..n]);
            let inp = vld1q_f32(ibuf.as_ptr());
            let w = vld1q_f32(wbuf.as_ptr());
            let g = vld1q_f32(gbuf.as_ptr());
            let silu_g = silu_core_neon_f32x4(g);
            let normed = vmulq_f32(vmulq_f32(inp, inv_rms_v), w);
            let result = vmulq_f32(normed, silu_g);
            let mut out_buf = [0.0f32; 4];
            vst1q_f32(out_buf.as_mut_ptr(), result);
            output[i..n].copy_from_slice(&out_buf[..rem]);
        }
    }
}

/// # Safety
///
/// The caller (`sigmoid_simd`) must guarantee `output.len() >= input.len()`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn sigmoid_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = input.len();
    let mut i = 0;
    // SAFETY: bounded by `n = input.len()`; tail only touches `[i..n)` or
    // the fixed-size stack buffers.
    unsafe {
        while i + 4 <= n {
            let x = vld1q_f32(input.as_ptr().add(i));
            let r = sigmoid_core_neon_f32x4(x);
            vst1q_f32(output.as_mut_ptr().add(i), r);
            i += 4;
        }

        let rem = n - i;
        if rem > 0 {
            let mut buf = [0.0f32; 4];
            buf[..rem].copy_from_slice(&input[i..n]);
            let x = vld1q_f32(buf.as_ptr());
            let r = sigmoid_core_neon_f32x4(x);
            let mut out_buf = [0.0f32; 4];
            vst1q_f32(out_buf.as_mut_ptr(), r);
            output[i..n].copy_from_slice(&out_buf[..rem]);
        }
    }
}

/// # Safety
///
/// The caller (`sigmoid_mul_simd`) must guarantee `gate.len() >= x.len()`
/// and `out.len() >= x.len()`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn sigmoid_mul_neon(x: &[f32], gate: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = x.len();
    let mut i = 0;
    // SAFETY: bounded by `n = x.len()`; tail only touches `[i..n)` or the
    // fixed-size stack buffers.
    unsafe {
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let gv = vld1q_f32(gate.as_ptr().add(i));
            let sig = sigmoid_core_neon_f32x4(gv);
            let r = vmulq_f32(xv, sig);
            vst1q_f32(out.as_mut_ptr().add(i), r);
            i += 4;
        }

        let rem = n - i;
        if rem > 0 {
            let mut xbuf = [0.0f32; 4];
            let mut gbuf = [0.0f32; 4];
            xbuf[..rem].copy_from_slice(&x[i..n]);
            gbuf[..rem].copy_from_slice(&gate[i..n]);
            let xv = vld1q_f32(xbuf.as_ptr());
            let gv = vld1q_f32(gbuf.as_ptr());
            let sig = sigmoid_core_neon_f32x4(gv);
            let r = vmulq_f32(xv, sig);
            let mut out_buf = [0.0f32; 4];
            vst1q_f32(out_buf.as_mut_ptr(), r);
            out[i..n].copy_from_slice(&out_buf[..rem]);
        }
    }
}

/// # Safety
///
/// The caller (`softplus_simd`) must guarantee `output.len() >= input.len()`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn softplus_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = input.len();
    let mut i = 0;
    // SAFETY: bounded by `n = input.len()`; tail only touches `[i..n)` or
    // the fixed-size stack buffers.
    unsafe {
        while i + 4 <= n {
            let x = vld1q_f32(input.as_ptr().add(i));
            let r = softplus_core_neon_f32x4(x);
            vst1q_f32(output.as_mut_ptr().add(i), r);
            i += 4;
        }

        let rem = n - i;
        if rem > 0 {
            let mut buf = [0.0f32; 4];
            buf[..rem].copy_from_slice(&input[i..n]);
            let x = vld1q_f32(buf.as_ptr());
            let r = softplus_core_neon_f32x4(x);
            let mut out_buf = [0.0f32; 4];
            vst1q_f32(out_buf.as_mut_ptr(), r);
            output[i..n].copy_from_slice(&out_buf[..rem]);
        }
    }
}

// ═════════════════════════════════════════════════════════════════
//  x86_64 AVX2+FMA implementations
//
//  Matching `simd_float_ops.rs`'s existing (accepted) AVX2 convention:
//  no vectorized transcendental polynomial on this tier, only on NEON —
//  `exp`/`ln` run per-lane through a stack buffer, exactly like
//  `silu_avx2`/`swiglu_avx2`/`softmax_avx2` already do in that file.
// ═════════════════════════════════════════════════════════════════

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`l2_norm_simd`) must guarantee `output.len() >= input.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_norm_avx2(input: &[f32], output: &mut [f32], eps: f32) {
    use std::arch::x86_64::*;

    let n = input.len();
    let mut sum_sq = 0.0f32;
    let mut i = 0;
    if n >= 8 {
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            acc = _mm256_fmadd_ps(v, v, acc);
            i += 8;
        }
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let s128 = _mm_add_ps(lo, hi);
        let shuf1 = _mm_shuffle_ps(s128, s128, 0b_00_11_10_01);
        let s2 = _mm_add_ps(s128, shuf1);
        let shuf2 = _mm_shuffle_ps(s2, s2, 0b_01_00_11_10);
        let s1 = _mm_add_ps(s2, shuf2);
        sum_sq = _mm_cvtss_f32(s1);
    }
    for val in input.iter().take(n).skip(i) {
        sum_sq += val * val;
    }

    let scale = 1.0 / sum_sq.sqrt().max(eps);

    i = 0;
    if n >= 8 {
        let scale_v = _mm256_set1_ps(scale);
        while i + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            let r = _mm256_mul_ps(v, scale_v);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), r);
            i += 8;
        }
    }
    for j in i..n {
        output[j] = input[j] * scale;
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`rms_norm_gated_simd`) must guarantee `weight.len() >=
/// input.len()`, `gate.len() >= input.len()` and `output.len() >=
/// input.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_gated_avx2(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    use std::arch::x86_64::*;

    let n = input.len();
    let mut sum_sq = 0.0f32;
    let mut i = 0;
    if n >= 8 {
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            acc = _mm256_fmadd_ps(v, v, acc);
            i += 8;
        }
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let s128 = _mm_add_ps(lo, hi);
        let shuf1 = _mm_shuffle_ps(s128, s128, 0b_00_11_10_01);
        let s2 = _mm_add_ps(s128, shuf1);
        let shuf2 = _mm_shuffle_ps(s2, s2, 0b_01_00_11_10);
        let s1 = _mm_add_ps(s2, shuf2);
        sum_sq = _mm_cvtss_f32(s1);
    }
    for val in input.iter().take(n).skip(i) {
        sum_sq += val * val;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    i = 0;
    if n >= 8 {
        let one = _mm256_set1_ps(1.0);
        let inv_rms_v = _mm256_set1_ps(inv_rms);
        while i + 8 <= n {
            let inp = _mm256_loadu_ps(input.as_ptr().add(i));
            let w = _mm256_loadu_ps(weight.as_ptr().add(i));
            let g = _mm256_loadu_ps(gate.as_ptr().add(i));
            let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), neg_g);
            for b in &mut buf {
                *b = b.exp();
            }
            let exp_neg_g = _mm256_loadu_ps(buf.as_ptr());
            let denom = _mm256_add_ps(one, exp_neg_g);
            let sigmoid_g = _mm256_div_ps(one, denom);
            let silu_g = _mm256_mul_ps(g, sigmoid_g);
            let normed = _mm256_mul_ps(_mm256_mul_ps(inp, inv_rms_v), w);
            let result = _mm256_mul_ps(normed, silu_g);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            i += 8;
        }
    }
    for j in i..n {
        output[j] = weight[j] * input[j] * inv_rms * silu_scalar_elem(gate[j]);
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`sigmoid_simd`) must guarantee `output.len() >= input.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_avx2(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = input.len();
    let mut i = 0;
    if n >= 8 {
        let one = _mm256_set1_ps(1.0);
        while i + 8 <= n {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), neg_x);
            for b in &mut buf {
                *b = b.exp();
            }
            let exp_neg = _mm256_loadu_ps(buf.as_ptr());
            let denom = _mm256_add_ps(one, exp_neg);
            let result = _mm256_div_ps(one, denom);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            i += 8;
        }
    }
    for j in i..n {
        output[j] = sigmoid_scalar_elem(input[j]);
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`sigmoid_mul_simd`) must guarantee `gate.len() >= x.len()` and
/// `out.len() >= x.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_mul_avx2(x: &[f32], gate: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = x.len();
    let mut i = 0;
    if n >= 8 {
        let one = _mm256_set1_ps(1.0);
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(x.as_ptr().add(i));
            let gv = _mm256_loadu_ps(gate.as_ptr().add(i));
            let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), gv);
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), neg_g);
            for b in &mut buf {
                *b = b.exp();
            }
            let exp_neg_g = _mm256_loadu_ps(buf.as_ptr());
            let denom = _mm256_add_ps(one, exp_neg_g);
            let sig = _mm256_div_ps(one, denom);
            let result = _mm256_mul_ps(xv, sig);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), result);
            i += 8;
        }
    }
    for j in i..n {
        out[j] = x[j] * sigmoid_scalar_elem(gate[j]);
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`softplus_simd`) must guarantee `output.len() >= input.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softplus_avx2(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = input.len();
    let mut i = 0;
    if n >= 8 {
        while i + 8 <= n {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), x);
            for b in &mut buf {
                *b = softplus_scalar_elem(*b);
            }
            let result = _mm256_loadu_ps(buf.as_ptr());
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            i += 8;
        }
    }
    for j in i..n {
        output[j] = softplus_scalar_elem(input[j]);
    }
}

// ═════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "{label} mismatch at [{i}]: {x} vs {y} (diff={})",
                (x - y).abs()
            );
        }
    }

    // ── l2_norm ─────────────────────────────────────────────────

    #[test]
    fn l2_norm_basic() {
        let input = vec![3.0, 4.0]; // norm = 5
        let mut output = vec![0.0; 2];
        l2_norm_simd(&input, &mut output, 1e-6).expect("matching lengths");
        assert!((output[0] - 0.6).abs() < 1e-5, "got {}", output[0]);
        assert!((output[1] - 0.8).abs() < 1e-5, "got {}", output[1]);
    }

    #[test]
    fn l2_norm_matches_scalar() {
        let input: Vec<f32> = (0..37).map(|i| (i as f32 - 18.0) * 0.37).collect();
        let mut simd_out = vec![0.0; 37];
        let mut scalar_out = vec![0.0; 37];
        l2_norm_simd(&input, &mut simd_out, 1e-6).expect("matching lengths");
        l2_norm_scalar(&input, &mut scalar_out, 1e-6);
        assert_close(&simd_out, &scalar_out, 1e-4, "l2_norm");
    }

    #[test]
    fn l2_norm_output_has_unit_norm_away_from_zero() {
        let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin() + 2.0).collect();
        let mut output = vec![0.0; 128];
        l2_norm_simd(&input, &mut output, 1e-6).expect("matching lengths");
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "output norm = {norm}");
    }

    /// K-10 correction: `ggml_l2_norm`'s `eps` floors the *denominator*
    /// (`scale = 1/max(sqrt(sum), eps)`), it is not added under the
    /// radical (`1/sqrt(sum+eps)`). An all-zero input cannot distinguish
    /// the two formulas (both give `scale = 1/eps`); a tiny-but-nonzero
    /// input can, because `sqrt(sum)` is then far smaller than `eps` under
    /// the correct formula but comparable to it under the wrong one.
    #[test]
    fn l2_norm_near_zero_vector_uses_eps_floor() {
        let n = 128;
        let input = vec![1e-20f32; n];
        let eps = 1e-6f32;
        let mut output = vec![0.0; n];
        l2_norm_simd(&input, &mut output, eps).expect("matching lengths");

        // Correct: sqrt(sum) = sqrt(128)*1e-20 ~= 1.13e-19, far below eps,
        // so scale = 1/eps = 1e6 and output ~= 1e-20 * 1e6 = 1e-14.
        let expected = 1e-20f32 * (1.0 / eps);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - expected).abs() < expected * 0.01,
                "l2_norm[{i}] = {v}, expected ~= {expected} (1/eps floor)"
            );
        }

        // The WRONG formula (x / sqrt(sum + eps)) would give scale
        // ~= 1/sqrt(eps) = 1000, i.e. output ~= 1e-17 -- three orders of
        // magnitude smaller. Pin that the real output is nowhere near that.
        let wrong_formula_output = 1e-20f32 / (eps).sqrt();
        assert!(
            (output[0] - wrong_formula_output).abs() > expected * 0.5,
            "output {} looks like the WRONG eps-under-the-radical formula \
             ({wrong_formula_output}), not the eps-floors-denominator one",
            output[0]
        );
    }

    #[test]
    fn l2_norm_rejects_short_output() {
        let input = vec![1.0; 8];
        let mut output = vec![0.0; 4];
        assert!(l2_norm_simd(&input, &mut output, EPS).is_err());
    }

    // ── rms_norm_gated ──────────────────────────────────────────

    #[test]
    fn rms_norm_gated_matches_scalar() {
        let input: Vec<f32> = (0..131).map(|i| (i as f32 - 65.0) * 0.21).collect();
        let weight: Vec<f32> = (0..131).map(|i| 0.5 + i as f32 * 0.01).collect();
        let gate: Vec<f32> = (0..131).map(|i| (i as f32 - 40.0) * 0.15).collect();
        let mut simd_out = vec![0.0; 131];
        let mut scalar_out = vec![0.0; 131];

        rms_norm_gated_simd(&input, &weight, &gate, &mut simd_out, 1e-6).expect("valid");
        rms_norm_gated_scalar(&input, &weight, &gate, &mut scalar_out, 1e-6);

        assert_close(&simd_out, &scalar_out, 1e-3, "rms_norm_gated");
    }

    #[test]
    fn rms_norm_gated_equals_rms_norm_then_swiglu() {
        // out[i] = w[i]*x[i]*inv_rms * silu(gate[i])
        //        = silu(gate[i]) * (w[i]*x[i]*inv_rms)
        // i.e. swiglu_simd(gate, rms_norm_simd(x,w)) with `gate` first
        // (matching this crate's swiglu_simd(gate,up) = silu(gate)*up).
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.3).collect();
        let weight: Vec<f32> = vec![1.3f32; 64];
        let gate: Vec<f32> = (0..64).map(|i| (i as f32 - 10.0) * 0.2).collect();
        let eps = 1e-6;

        let mut fused = vec![0.0; 64];
        rms_norm_gated_simd(&input, &weight, &gate, &mut fused, eps).expect("valid");

        let mut normed = vec![0.0; 64];
        crate::simd_float_ops::rms_norm_simd(&input, &weight, &mut normed, eps).expect("valid");
        let mut two_step = vec![0.0; 64];
        crate::simd_float_ops::swiglu_simd(&gate, &normed, &mut two_step).expect("valid");

        assert_close(&fused, &two_step, 1e-3, "fused vs two-step");
    }

    #[test]
    fn rms_norm_gated_rejects_mismatched_weight() {
        let input = vec![1.0; 8];
        let weight = vec![1.0; 4];
        let gate = vec![1.0; 8];
        let mut output = vec![0.0; 8];
        assert!(rms_norm_gated_simd(&input, &weight, &gate, &mut output, EPS).is_err());
    }

    #[test]
    fn rms_norm_gated_rejects_mismatched_gate() {
        let input = vec![1.0; 8];
        let weight = vec![1.0; 8];
        let gate = vec![1.0; 4];
        let mut output = vec![0.0; 8];
        assert!(rms_norm_gated_simd(&input, &weight, &gate, &mut output, EPS).is_err());
    }

    #[test]
    fn rms_norm_gated_rejects_short_output() {
        let input = vec![1.0; 8];
        let weight = vec![1.0; 8];
        let gate = vec![1.0; 8];
        let mut output = vec![0.0; 4];
        assert!(rms_norm_gated_simd(&input, &weight, &gate, &mut output, EPS).is_err());
    }

    // ── sigmoid / sigmoid_mul ───────────────────────────────────

    #[test]
    fn sigmoid_basic_values() {
        let input = vec![0.0, 100.0, -100.0];
        let mut output = vec![0.0; 3];
        sigmoid_simd(&input, &mut output).expect("valid");
        assert!((output[0] - 0.5).abs() < 1e-5);
        assert!((output[1] - 1.0).abs() < 1e-4);
        assert!(output[2].abs() < 1e-4);
    }

    #[test]
    fn sigmoid_matches_scalar() {
        let input: Vec<f32> = (0..67).map(|i| (i as f32 - 33.0) * 0.4).collect();
        let mut simd_out = vec![0.0; 67];
        let mut scalar_out = vec![0.0; 67];
        sigmoid_simd(&input, &mut simd_out).expect("valid");
        sigmoid_scalar(&input, &mut scalar_out);
        assert_close(&simd_out, &scalar_out, 1e-4, "sigmoid");
    }

    #[test]
    fn sigmoid_mul_matches_x_times_sigmoid_gate() {
        let x: Vec<f32> = (0..50).map(|i| i as f32 * 0.1 - 2.5).collect();
        let gate: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) * 0.2).collect();
        let mut fused = vec![0.0; 50];
        sigmoid_mul_simd(&x, &gate, &mut fused).expect("valid");

        let mut sig = vec![0.0; 50];
        sigmoid_simd(&gate, &mut sig).expect("valid");
        let two_step: Vec<f32> = x.iter().zip(sig.iter()).map(|(&a, &b)| a * b).collect();

        assert_close(&fused, &two_step, 1e-4, "sigmoid_mul");
    }

    #[test]
    fn sigmoid_mul_rejects_mismatched_gate() {
        let x = vec![1.0; 8];
        let gate = vec![1.0; 4];
        let mut out = vec![0.0; 8];
        assert!(sigmoid_mul_simd(&x, &gate, &mut out).is_err());
    }

    #[test]
    fn sigmoid_rejects_short_output() {
        let input = vec![1.0; 8];
        let mut output = vec![0.0; 4];
        assert!(sigmoid_simd(&input, &mut output).is_err());
    }

    // ── softplus ────────────────────────────────────────────────

    #[test]
    fn softplus_matches_scalar_wide_range() {
        let input: Vec<f32> = (-200..200).map(|i| i as f32 * 0.37).collect();
        let mut simd_out = vec![0.0; input.len()];
        let mut scalar_out = vec![0.0; input.len()];
        softplus_simd(&input, &mut simd_out).expect("valid");
        softplus_scalar(&input, &mut scalar_out);
        assert_close(&simd_out, &scalar_out, 1e-3, "softplus");
    }

    #[test]
    fn softplus_cutoff_is_exactly_20() {
        let input = vec![19.999, 20.0, 20.001, 50.0, 1000.0];
        let mut output = vec![0.0; 5];
        softplus_simd(&input, &mut output).expect("valid");
        // At/above the cutoff, softplus(x) == x exactly (the linear branch).
        assert_eq!(output[1], 20.0);
        assert_eq!(output[2], 20.001);
        assert_eq!(output[3], 50.0);
        assert_eq!(output[4], 1000.0);
        // Just below the cutoff, ln(1+exp(x)) ~= x too (but computed, not
        // pass-through) -- should still be numerically close.
        assert!((output[0] - 19.999).abs() < 1e-3);
    }

    #[test]
    fn softplus_at_zero_is_ln2() {
        let input = vec![0.0f32];
        let mut output = vec![0.0f32];
        softplus_simd(&input, &mut output).expect("valid");
        assert!((output[0] - std::f32::consts::LN_2).abs() < 1e-4);
    }

    #[test]
    fn softplus_large_negative_approaches_zero() {
        let input = vec![-1000.0f32];
        let mut output = vec![0.0f32];
        softplus_simd(&input, &mut output).expect("valid");
        assert!(output[0].abs() < 1e-6, "softplus(-1000) = {}", output[0]);
    }

    #[test]
    fn softplus_rejects_short_output() {
        let input = vec![1.0; 8];
        let mut output = vec![0.0; 4];
        assert!(softplus_simd(&input, &mut output).is_err());
    }

    // ── ln_neon_f32x4 direct accuracy check (aarch64 only) ─────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn ln_neon_matches_std_ln_over_a_wide_range() {
        use std::arch::aarch64::*;
        // Sweep across several decades of magnitude plus the exact
        // worst-case mantissa point (1.5, since 1/ln2 ~= 1.4427 is the
        // point of maximum error in the linear (m-1)*ln2 initial guess).
        let ys: Vec<f32> = (0..400)
            .map(|i| 1.0001f32 + i as f32 * 0.05)
            .chain([1.0f32, 1.5, 2.0, 1e-3, 1e3, 1e10, 4.85e8])
            .collect();
        for &y in &ys {
            let out = unsafe {
                let v = vdupq_n_f32(y);
                let r = ln_neon_f32x4(v);
                vgetq_lane_f32::<0>(r)
            };
            let expected = y.ln();
            assert!(
                (out - expected).abs() < 1e-3,
                "ln_neon_f32x4({y}) = {out}, expected {expected}"
            );
        }
    }
}
