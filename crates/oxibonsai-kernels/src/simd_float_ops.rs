//! SIMD-accelerated float operations for LLM inference.
//!
//! Provides optimized implementations of common neural network operations
//! used in transformer inference: softmax, RMSNorm, SiLU, SwiGLU, and RoPE.
//!
//! Dispatch strategy:
//! - **aarch64**: ARM NEON intrinsics via `std::arch::aarch64::*`
//! - **x86_64**: AVX2/FMA intrinsics via `std::arch::x86_64::*`
//! - **fallback**: Scalar Rust for all other architectures
//!
//! All functions accept raw `&[f32]` / `&mut [f32]` slices to match the
//! model-layer API. SciRS2-Core's SIMD primitives are used for reductions
//! and element-wise transforms where the `ndarray::ArrayView1` adapter is
//! cheap (zero-copy wrap of a contiguous slice).
//!
//! ## Length contracts (K-02 / M-01)
//!
//! `rms_norm_simd`, `silu_simd`, `swiglu_simd` and `rope_apply_simd` validate
//! every cross-slice length invariant with a real, always-on check that
//! returns [`KernelError::DimensionMismatch`] / [`KernelError::BufferTooSmall`]
//! — never a `debug_assert!` that compiles away in release. The NEON/AVX2
//! helpers they call are therefore only ever driven with slices the public
//! entry point has already proven are long enough; callers must not invoke
//! the private `*_neon` / `*_avx2` functions directly with unchecked slices.
//! `softmax_simd` is intentionally excluded: it has a single in-place buffer
//! with no cross-slice length relationship to validate.

use crate::error::{KernelError, KernelResult};

// ─── Softmax (in-place) ──────────────────────────────────────────

/// Numerically-stable softmax, computed in-place.
///
/// 1. Find `max` via SIMD reduction
/// 2. Subtract max and compute `exp` via SIMD
/// 3. Sum via SIMD reduction
/// 4. Divide every element by the sum
///
/// Falls back to scalar on unsupported platforms.
///
/// ## Deliberately `-> ()`, not `KernelResult<()>` (KERN-SOUND spec-item-1
/// amendment)
///
/// The K-02/M-01 spec text names five entry points that must return
/// `KernelResult<()>`; this is the one that stays infallible, as a decided
/// scope narrowing rather than an oversight:
///
/// - Unlike `rms_norm_simd` / `silu_simd` / `swiglu_simd` / `rope_apply_simd`,
///   `softmax_simd` operates on a single in-place buffer. There is no
///   cross-slice length relationship to validate and no real precondition
///   that can be violated — `values.len()` bounds every SIMD loop by
///   construction, for any length including zero.
/// - Its three external call sites
///   (`oxibonsai-model/src/layers/attention.rs`, `oxibonsai-image/src/math.rs`,
///   `oxibonsai-image/src/te/forward.rs`) are all outside this package's
///   `owned_files`, and none of them use the return value today. Converting
///   this signature would force each of those three call sites to either
///   propagate a `Result` that can never be `Err` through their own
///   (infallible) callers, or discard it with `let _ = softmax_simd(...)`.
///   The latter is the error-swallowing idiom K-02/M-01 exists to eliminate
///   — manufacturing one to satisfy the letter of the spec at a real
///   invariant's expense would make the codebase worse in exactly the
///   dimension this package improves.
#[inline]
pub fn softmax_simd(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        softmax_neon(values);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: we just confirmed AVX2+FMA are available.
            unsafe { softmax_avx2(values) };
            return;
        }
        softmax_scalar(values);
    }

    // Scalar fallback (wasm, riscv, …)
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        softmax_scalar(values);
    }
}

// ─── RMSNorm ─────────────────────────────────────────────────────

/// RMS normalization: `output[i] = weight[i] * input[i] / rms(input)`
/// where `rms(x) = sqrt(mean(x²) + eps)`.
///
/// # Errors
///
/// - [`KernelError::DimensionMismatch`] if `weight.len() != input.len()`.
/// - [`KernelError::BufferTooSmall`] if `output.len() < input.len()`.
#[inline]
pub fn rms_norm_simd(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    eps: f32,
) -> KernelResult<()> {
    let n = input.len();
    if weight.len() != n {
        return Err(KernelError::DimensionMismatch {
            expected: n,
            got: weight.len(),
        });
    }
    if output.len() < n {
        return Err(KernelError::BufferTooSmall {
            needed: n,
            available: output.len(),
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        rms_norm_neon(input, weight, output, eps);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { rms_norm_avx2(input, weight, output, eps) };
            return Ok(());
        }
        rms_norm_scalar(input, weight, output, eps);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        rms_norm_scalar(input, weight, output, eps);
    }

    Ok(())
}

// ─── SiLU element-wise ───────────────────────────────────────────

/// Element-wise SiLU (Swish): `output[i] = input[i] / (1 + exp(-input[i]))`.
///
/// # Errors
///
/// [`KernelError::BufferTooSmall`] if `output.len() < input.len()`.
#[inline]
pub fn silu_simd(input: &[f32], output: &mut [f32]) -> KernelResult<()> {
    let n = input.len();
    if output.len() < n {
        return Err(KernelError::BufferTooSmall {
            needed: n,
            available: output.len(),
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        silu_neon(input, output);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { silu_avx2(input, output) };
            return Ok(());
        }
        silu_scalar(input, output);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        silu_scalar(input, output);
    }

    Ok(())
}

// ─── SwiGLU ──────────────────────────────────────────────────────

/// SwiGLU: `output[i] = silu(gate[i]) * up[i]`.
///
/// # Errors
///
/// - [`KernelError::DimensionMismatch`] if `up.len() != gate.len()`.
/// - [`KernelError::BufferTooSmall`] if `output.len() < gate.len()`.
#[inline]
pub fn swiglu_simd(gate: &[f32], up: &[f32], output: &mut [f32]) -> KernelResult<()> {
    let n = gate.len();
    if up.len() != n {
        return Err(KernelError::DimensionMismatch {
            expected: n,
            got: up.len(),
        });
    }
    if output.len() < n {
        return Err(KernelError::BufferTooSmall {
            needed: n,
            available: output.len(),
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        swiglu_neon(gate, up, output);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { swiglu_avx2(gate, up, output) };
            return Ok(());
        }
        swiglu_scalar(gate, up, output);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        swiglu_scalar(gate, up, output);
    }

    Ok(())
}

// ─── RoPE apply ──────────────────────────────────────────────────

/// Apply rotary position embeddings to a head-dim vector.
///
/// Given `half_dim = input.len() / 2`:
///
/// ```text
/// output[i]            = input[i] * cos[i] - input[half_dim+i] * sin[i]
/// output[half_dim + i] = input[i] * sin[i] + input[half_dim+i] * cos[i]
/// ```
///
/// `cos_table` and `sin_table` must each have length `half_dim`.
///
/// # Errors
///
/// - [`KernelError::DimensionMismatch`] if `cos_table.len() != half_dim` or
///   `sin_table.len() != half_dim` (where `half_dim = input.len() / 2`).
/// - [`KernelError::BufferTooSmall`] if `output.len() < input.len()`.
#[inline]
pub fn rope_apply_simd(
    input: &[f32],
    output: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
) -> KernelResult<()> {
    let head_dim = input.len();
    let half_dim = head_dim / 2;
    if cos_table.len() != half_dim {
        return Err(KernelError::DimensionMismatch {
            expected: half_dim,
            got: cos_table.len(),
        });
    }
    if sin_table.len() != half_dim {
        return Err(KernelError::DimensionMismatch {
            expected: half_dim,
            got: sin_table.len(),
        });
    }
    if output.len() < head_dim {
        return Err(KernelError::BufferTooSmall {
            needed: head_dim,
            available: output.len(),
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        rope_neon(input, output, cos_table, sin_table, half_dim);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { rope_avx2(input, output, cos_table, sin_table, half_dim) };
            return Ok(());
        }
        rope_scalar(input, output, cos_table, sin_table, half_dim);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        rope_scalar(input, output, cos_table, sin_table, half_dim);
    }

    Ok(())
}

// ═════════════════════════════════════════════════════════════════
//  Scalar fallbacks (also used as reference in tests)
// ═════════════════════════════════════════════════════════════════

#[allow(dead_code)]
#[inline]
fn softmax_scalar(values: &mut [f32]) {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in values.iter_mut() {
            *v *= inv_sum;
        }
    }
}

#[allow(dead_code)]
#[inline]
fn rms_norm_scalar(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len();
    let mut sum_sq = 0.0f32;
    for &x in input {
        sum_sq += x * x;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        output[i] = weight[i] * input[i] * inv_rms;
    }
}

#[inline]
fn silu_scalar_elem(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[allow(dead_code)]
#[inline]
fn silu_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = silu_scalar_elem(input[i]);
    }
}

#[allow(dead_code)]
#[inline]
fn swiglu_scalar(gate: &[f32], up: &[f32], output: &mut [f32]) {
    for i in 0..gate.len() {
        output[i] = silu_scalar_elem(gate[i]) * up[i];
    }
}

#[allow(dead_code)]
#[inline]
fn rope_scalar(
    input: &[f32],
    output: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    half_dim: usize,
) {
    for i in 0..half_dim {
        let x0 = input[i];
        let x1 = input[half_dim + i];
        output[i] = x0 * cos_table[i] - x1 * sin_table[i];
        output[half_dim + i] = x0 * sin_table[i] + x1 * cos_table[i];
    }
}

// ═════════════════════════════════════════════════════════════════
//  AArch64 NEON implementations
// ═════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[inline]
fn softmax_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = values.len();

    // ── 1. Find max ─────────────────────────────────────────────
    let mut max_val = f32::NEG_INFINITY;
    let mut i = 0;

    if n >= 4 {
        // SAFETY: NEON is always available on aarch64.
        unsafe {
            let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
            while i + 4 <= n {
                let v = vld1q_f32(values.as_ptr().add(i));
                max_vec = vmaxq_f32(max_vec, v);
                i += 4;
            }
            // horizontal max
            let pair = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
            let pair2 = vpmax_f32(pair, pair);
            max_val = vget_lane_f32::<0>(pair2);
        }
    }
    // tail
    for &v in &values[i..n] {
        if v > max_val {
            max_val = v;
        }
    }

    // ── 2. exp(val - max) ───────────────────────────────────────
    // Vectorized exp (K-M3): the body and the 1-3 element tail both run
    // through `exp_neon_f32x4` (the tail via a padded stack buffer) instead
    // of the tail falling back to `f32::exp()`, so a value is bit-identical
    // regardless of buffer length/offset.
    let mut sum = 0.0f32;
    i = 0;

    if n >= 4 {
        unsafe {
            let max_v = vdupq_n_f32(max_val);
            let mut sum_vec = vdupq_n_f32(0.0);
            while i + 4 <= n {
                let v = vld1q_f32(values.as_ptr().add(i));
                let shifted = vsubq_f32(v, max_v);
                let exp_v = exp_neon_f32x4(shifted);
                vst1q_f32(values.as_mut_ptr().add(i), exp_v);
                sum_vec = vaddq_f32(sum_vec, exp_v);
                i += 4;
            }
            let pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
            let pair2 = vpadd_f32(pair, pair);
            sum = vget_lane_f32::<0>(pair2);
        }
    }
    let rem = n - i;
    if rem > 0 {
        // SAFETY: `buf`/`out_buf` are fixed-size 4-element stack arrays, so
        // the loads/stores below never touch `values` out of the `[i..n)`
        // range that `copy_from_slice` already bounds-checked.
        unsafe {
            let max_v = vdupq_n_f32(max_val);
            // Pad with `max_val` (⇒ shifted = 0 ⇒ exp = 1) so the unused
            // lanes can never contaminate `sum`/`values` even though they
            // are computed; only `out_buf[..rem]` is ever read back.
            let mut buf = [max_val; 4];
            buf[..rem].copy_from_slice(&values[i..n]);
            let v = vld1q_f32(buf.as_ptr());
            let shifted = vsubq_f32(v, max_v);
            let exp_v = exp_neon_f32x4(shifted);
            let mut out_buf = [0.0f32; 4];
            vst1q_f32(out_buf.as_mut_ptr(), exp_v);
            values[i..n].copy_from_slice(&out_buf[..rem]);
            sum += out_buf[..rem].iter().sum::<f32>();
        }
    }

    // ── 3. Divide by sum ────────────────────────────────────────
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        i = 0;
        if n >= 4 {
            unsafe {
                let inv_v = vdupq_n_f32(inv_sum);
                while i + 4 <= n {
                    let v = vld1q_f32(values.as_ptr().add(i));
                    let r = vmulq_f32(v, inv_v);
                    vst1q_f32(values.as_mut_ptr().add(i), r);
                    i += 4;
                }
            }
        }
        for v in &mut values[i..n] {
            *v *= inv_sum;
        }
    }
}

/// Vectorized `exp(x)` for 4 `f32` lanes (K-M3).
///
/// Classic Cephes-derived range reduction + 5th-degree minimax polynomial
/// (the same algorithm as e.g. Julien Pommier's public-domain
/// `sse_mathfun`/`neon_mathfun`): write `x = k*ln2 + r` with
/// `k = round(x * log2(e))` and `|r| <= ln2/2`, then `exp(x) = 2^k * poly(r)`
/// where `poly` is a 5th-degree minimax fit to `e^r` and `2^k` is built by
/// inserting the biased exponent directly into an IEEE-754 bit pattern
/// (`vreinterpretq_f32_s32(vshlq_n_s32(k + 127, 23))`). Accurate to ~1 ulp
/// over the sigmoid's numerically useful range. Inputs are clamped to
/// `±88.3762626647950` (the range where `f32::exp` is finite) so a large
/// `|x|` saturates instead of overflowing the exponent field.
///
/// Note: because of that clamp, `silu`/`swiglu` built on this function are
/// only *absolute*-error-bounded (not asymptotically exact like the scalar
/// path's `x * e^x`) for very negative `x` — e.g. at `x = -100` this returns
/// `x / (1 + e^-88.376…)` (~-3.8e-37) rather than the scalar path's ~-3.7e-42
/// — so do not use this deep tail as a determinism oracle.
///
/// This computes the *same* result regardless of which lane holds "real"
/// data — callers needing body/tail consistency (K-M3) must run a padded
/// tail through this exact function rather than falling back to `f32::exp()`,
/// so a value is bit-identical no matter where in the buffer it lands.
///
/// `pub(crate)` (K-INT8 wave-4b): `norms.rs` imports this instead of holding
/// its own verbatim copy — see its module doc comment and
/// `norms::dedup_parity` for the bit-identity proof.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn exp_neon_f32x4(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
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

    // fx = round(x * log2(e)), via truncate-then-correct-to-floor of (x*log2e + 0.5).
    let fx0 = vfmaq_f32(vdupq_n_f32(0.5), x, vdupq_n_f32(LOG2EF));
    let fx_trunc = vcvtq_f32_s32(vcvtq_s32_f32(fx0));
    let overshot = vcgtq_f32(fx_trunc, fx0);
    let overshot_f = vreinterpretq_f32_u32(vandq_u32(overshot, vreinterpretq_u32_f32(one)));
    let fx = vsubq_f32(fx_trunc, overshot_f);

    // r = x - fx*ln2, two-constant (hi/lo) reduction for precision.
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

    // 2^fx via direct exponent-field insertion.
    let emm0 = vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127));
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(emm0, 23));

    vmulq_f32(y, pow2n)
}

/// SIMD SiLU on 4 lanes: `x / (1 + exp(-x))`.
///
/// Shares the exact same computation for every lane regardless of whether it
/// holds "real" input or tail padding (K-M3), and uses `vdivq_f32`
/// (correctly-rounded `FDIV`, always available on ARMv8 NEON) rather than a
/// `vrecpeq_f32` + Newton-Raphson reciprocal approximation — the previous
/// code used the approximation in the SIMD body but exact division in the
/// scalar tail, so the same call could return two different roundings for
/// different elements. A single division strategy removes that hazard.
///
/// `pub(crate)` — see [`exp_neon_f32x4`]'s doc comment.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn silu_core_neon_f32x4(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let one = vdupq_n_f32(1.0);
    let denom = vaddq_f32(one, exp_neon_f32x4(vnegq_f32(x)));
    vdivq_f32(x, denom)
}

/// # Safety
///
/// The caller (`rms_norm_simd`) must guarantee `weight.len() >= input.len()`
/// and `output.len() >= input.len()`; this function indexes raw pointers
/// derived from `input`/`weight`/`output` bounded only by `input.len()` and
/// does not re-validate lengths itself (K-02).
#[cfg(target_arch = "aarch64")]
#[inline]
fn rms_norm_neon(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    use std::arch::aarch64::*;

    let n = input.len();

    // ── 1. Sum of squares via NEON dot (input · input) ──────────
    let mut sum_sq = 0.0f32;
    let mut i = 0;
    if n >= 4 {
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            while i + 4 <= n {
                let v = vld1q_f32(input.as_ptr().add(i));
                acc = vfmaq_f32(acc, v, v); // acc += v * v  (fused)
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

    // ── 2. inv_rms = 1 / sqrt(mean + eps) ────────────────────
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    // ── 3. output = weight * input * inv_rms ─────────────────
    i = 0;
    if n >= 4 {
        unsafe {
            let scale = vdupq_n_f32(inv_rms);
            while i + 4 <= n {
                let inp = vld1q_f32(input.as_ptr().add(i));
                let w = vld1q_f32(weight.as_ptr().add(i));
                let normalized = vmulq_f32(inp, scale);
                let result = vmulq_f32(w, normalized);
                vst1q_f32(output.as_mut_ptr().add(i), result);
                i += 4;
            }
        }
    }
    for j in i..n {
        output[j] = weight[j] * input[j] * inv_rms;
    }
}

/// # Safety
///
/// The caller (`silu_simd`) must guarantee `output.len() >= input.len()`;
/// this function indexes raw pointers derived from `input`/`output` bounded
/// only by `input.len()` and does not re-validate lengths itself (K-02).
#[cfg(target_arch = "aarch64")]
#[inline]
fn silu_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = input.len();
    let mut i = 0;
    // SAFETY: NEON is always available on aarch64. The body loop is bounded
    // by `n = input.len()` (which `output.len()` is checked against by the
    // caller); the tail below never reads/writes past `[i..n)` of either
    // slice and only ever touches the fixed-size 4-element stack buffers.
    unsafe {
        while i + 4 <= n {
            let x = vld1q_f32(input.as_ptr().add(i));
            let result = silu_core_neon_f32x4(x);
            vst1q_f32(output.as_mut_ptr().add(i), result);
            i += 4;
        }

        // Tail (1-3 remaining elements): pad into a 4-lane stack buffer and
        // run through the SAME vectorized path (K-M3) instead of falling
        // back to `silu_scalar_elem`, which would compute a *different*
        // rounding (libm `exp` + exact division vs. the body's polynomial
        // `exp` + `vdivq_f32`) for the last 1-3 elements of every
        // non-multiple-of-4-length call.
        let rem = n - i;
        if rem > 0 {
            let mut buf = [0.0f32; 4];
            buf[..rem].copy_from_slice(&input[i..n]);
            let x = vld1q_f32(buf.as_ptr());
            let result = silu_core_neon_f32x4(x);
            let mut out_buf = [0.0f32; 4];
            vst1q_f32(out_buf.as_mut_ptr(), result);
            output[i..n].copy_from_slice(&out_buf[..rem]);
        }
    }
}

/// # Safety
///
/// The caller (`swiglu_simd`) must guarantee `up.len() >= gate.len()` and
/// `output.len() >= gate.len()`; this function indexes raw pointers derived
/// from `gate`/`up`/`output` bounded only by `gate.len()` and does not
/// re-validate lengths itself (K-02).
#[cfg(target_arch = "aarch64")]
#[inline]
fn swiglu_neon(gate: &[f32], up: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = gate.len();
    let mut i = 0;
    // SAFETY: see `silu_neon` — bounded by `n = gate.len()` (`up.len()` is
    // checked against it by the caller) or the fixed-size tail buffers.
    unsafe {
        while i + 4 <= n {
            let g = vld1q_f32(gate.as_ptr().add(i));
            let u = vld1q_f32(up.as_ptr().add(i));
            let silu_g = silu_core_neon_f32x4(g);
            let result = vmulq_f32(silu_g, u);
            vst1q_f32(output.as_mut_ptr().add(i), result);
            i += 4;
        }

        // Tail: same rationale as `silu_neon` — pad and reuse the identical
        // vectorized path so body and tail agree bit-for-bit (K-M3).
        let rem = n - i;
        if rem > 0 {
            let mut gbuf = [0.0f32; 4];
            let mut ubuf = [0.0f32; 4];
            gbuf[..rem].copy_from_slice(&gate[i..n]);
            ubuf[..rem].copy_from_slice(&up[i..n]);
            let g = vld1q_f32(gbuf.as_ptr());
            let u = vld1q_f32(ubuf.as_ptr());
            let silu_g = silu_core_neon_f32x4(g);
            let result = vmulq_f32(silu_g, u);
            let mut out_buf = [0.0f32; 4];
            vst1q_f32(out_buf.as_mut_ptr(), result);
            output[i..n].copy_from_slice(&out_buf[..rem]);
        }
    }
}

/// # Safety
///
/// The caller (`rope_apply_simd`) must guarantee `cos_table.len() >= half_dim`,
/// `sin_table.len() >= half_dim`, `input.len() >= 2 * half_dim` and
/// `output.len() >= 2 * half_dim`; this function indexes raw pointers derived
/// from those slices bounded only by `half_dim` and does not re-validate
/// lengths itself (K-02).
#[cfg(target_arch = "aarch64")]
#[inline]
fn rope_neon(
    input: &[f32],
    output: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    half_dim: usize,
) {
    use std::arch::aarch64::*;

    let mut i = 0;
    if half_dim >= 4 {
        unsafe {
            while i + 4 <= half_dim {
                let x0 = vld1q_f32(input.as_ptr().add(i));
                let x1 = vld1q_f32(input.as_ptr().add(half_dim + i));
                let c = vld1q_f32(cos_table.as_ptr().add(i));
                let s = vld1q_f32(sin_table.as_ptr().add(i));

                // output[i]            = x0*cos - x1*sin
                let out_lo = vmlsq_f32(vmulq_f32(x0, c), x1, s);
                // output[half_dim + i] = x0*sin + x1*cos
                let out_hi = vmlaq_f32(vmulq_f32(x1, c), x0, s);

                vst1q_f32(output.as_mut_ptr().add(i), out_lo);
                vst1q_f32(output.as_mut_ptr().add(half_dim + i), out_hi);
                i += 4;
            }
        }
    }
    // tail
    for j in i..half_dim {
        let x0 = input[j];
        let x1 = input[half_dim + j];
        output[j] = x0 * cos_table[j] - x1 * sin_table[j];
        output[half_dim + j] = x0 * sin_table[j] + x1 * cos_table[j];
    }
}

// ═════════════════════════════════════════════════════════════════
//  x86_64 AVX2+FMA implementations
// ═════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = values.len();

    // ── 1. Find max ─────────────────────────────────────────────
    let mut max_val = f32::NEG_INFINITY;
    let mut i = 0;
    if n >= 8 {
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        while i + 8 <= n {
            let v = _mm256_loadu_ps(values.as_ptr().add(i));
            max_vec = _mm256_max_ps(max_vec, v);
            i += 8;
        }
        // horizontal max: 256 → 128 → scalar
        let hi = _mm256_extractf128_ps(max_vec, 1);
        let lo = _mm256_castps256_ps128(max_vec);
        let m128 = _mm_max_ps(lo, hi);
        let shuf1 = _mm_shuffle_ps(m128, m128, 0b_01_00_11_10);
        let m2 = _mm_max_ps(m128, shuf1);
        let shuf2 = _mm_shuffle_ps(m2, m2, 0b_00_00_00_01);
        let m1 = _mm_max_ps(m2, shuf2);
        max_val = _mm_cvtss_f32(m1);
    }
    for val in values.iter().take(n).skip(i) {
        if *val > max_val {
            max_val = *val;
        }
    }

    // ── 2. exp(val - max) ───────────────────────────────────────
    let mut sum = 0.0f32;
    i = 0;
    if n >= 8 {
        let max_v = _mm256_set1_ps(max_val);
        let mut sum_vec = _mm256_setzero_ps();
        while i + 8 <= n {
            let v = _mm256_loadu_ps(values.as_ptr().add(i));
            let shifted = _mm256_sub_ps(v, max_v);
            // extract, exp per lane, reload
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), shifted);
            for b in &mut buf {
                *b = b.exp();
            }
            let exp_v = _mm256_loadu_ps(buf.as_ptr());
            _mm256_storeu_ps(values.as_mut_ptr().add(i), exp_v);
            sum_vec = _mm256_add_ps(sum_vec, exp_v);
            i += 8;
        }
        // horizontal sum
        let hi = _mm256_extractf128_ps(sum_vec, 1);
        let lo = _mm256_castps256_ps128(sum_vec);
        let s128 = _mm_add_ps(lo, hi);
        let shuf1 = _mm_shuffle_ps(s128, s128, 0b_00_11_10_01);
        let s2 = _mm_add_ps(s128, shuf1);
        let shuf2 = _mm_shuffle_ps(s2, s2, 0b_01_00_11_10);
        let s1 = _mm_add_ps(s2, shuf2);
        sum = _mm_cvtss_f32(s1);
    }
    for val in values.iter_mut().take(n).skip(i) {
        let e = (*val - max_val).exp();
        *val = e;
        sum += e;
    }

    // ── 3. Divide by sum ────────────────────────────────────────
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        i = 0;
        if n >= 8 {
            let inv_v = _mm256_set1_ps(inv_sum);
            while i + 8 <= n {
                let v = _mm256_loadu_ps(values.as_ptr().add(i));
                let r = _mm256_mul_ps(v, inv_v);
                _mm256_storeu_ps(values.as_mut_ptr().add(i), r);
                i += 8;
            }
        }
        for val in values.iter_mut().take(n).skip(i) {
            *val *= inv_sum;
        }
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support (`#[target_feature]`
/// UB otherwise). In addition, the caller (`rms_norm_simd`) must guarantee
/// `weight.len() >= input.len()` and `output.len() >= input.len()`; this
/// function indexes raw pointers derived from those slices bounded only by
/// `input.len()` and does not re-validate lengths itself (K-02).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_avx2(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    use std::arch::x86_64::*;

    let n = input.len();

    // ── 1. Sum of squares ───────────────────────────────────────
    let mut sum_sq = 0.0f32;
    let mut i = 0;
    if n >= 8 {
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            acc = _mm256_fmadd_ps(v, v, acc);
            i += 8;
        }
        // horizontal sum
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

    // ── 2. inv_rms ──────────────────────────────────────────────
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    // ── 3. output = weight * input * inv_rms ────────────────────
    i = 0;
    if n >= 8 {
        let scale = _mm256_set1_ps(inv_rms);
        while i + 8 <= n {
            let inp = _mm256_loadu_ps(input.as_ptr().add(i));
            let w = _mm256_loadu_ps(weight.as_ptr().add(i));
            let normed = _mm256_mul_ps(inp, scale);
            let result = _mm256_mul_ps(w, normed);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            i += 8;
        }
    }
    for j in i..n {
        output[j] = weight[j] * input[j] * inv_rms;
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`silu_simd`) must guarantee `output.len() >= input.len()`; this
/// function indexes raw pointers derived from `input`/`output` bounded only
/// by `input.len()` and does not re-validate lengths itself (K-02).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_avx2(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = input.len();
    let mut i = 0;

    if n >= 8 {
        let one = _mm256_set1_ps(1.0);
        while i + 8 <= n {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            // negate
            let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            // exp per lane
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), neg_x);
            for b in &mut buf {
                *b = b.exp();
            }
            let exp_neg = _mm256_loadu_ps(buf.as_ptr());
            // 1 / (1 + exp(-x))
            let denom = _mm256_add_ps(one, exp_neg);
            let recip = _mm256_div_ps(one, denom);
            // x * sigmoid(x)
            let result = _mm256_mul_ps(x, recip);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            i += 8;
        }
    }
    for j in i..n {
        output[j] = silu_scalar_elem(input[j]);
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`swiglu_simd`) must guarantee `up.len() >= gate.len()` and
/// `output.len() >= gate.len()`; this function indexes raw pointers derived
/// from `gate`/`up`/`output` bounded only by `gate.len()` and does not
/// re-validate lengths itself (K-02).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_avx2(gate: &[f32], up: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = gate.len();
    let mut i = 0;

    if n >= 8 {
        let one = _mm256_set1_ps(1.0);
        while i + 8 <= n {
            let g = _mm256_loadu_ps(gate.as_ptr().add(i));
            let u = _mm256_loadu_ps(up.as_ptr().add(i));
            let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), neg_g);
            for b in &mut buf {
                *b = b.exp();
            }
            let exp_neg = _mm256_loadu_ps(buf.as_ptr());
            let denom = _mm256_add_ps(one, exp_neg);
            let recip = _mm256_div_ps(one, denom);
            let silu_g = _mm256_mul_ps(g, recip);
            let result = _mm256_mul_ps(silu_g, u);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            i += 8;
        }
    }
    for j in i..n {
        output[j] = silu_scalar_elem(gate[j]) * up[j];
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`rope_apply_simd`) must guarantee `cos_table.len() >= half_dim`,
/// `sin_table.len() >= half_dim`, `input.len() >= 2 * half_dim` and
/// `output.len() >= 2 * half_dim`; this function indexes raw pointers derived
/// from those slices bounded only by `half_dim` and does not re-validate
/// lengths itself (K-02).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rope_avx2(
    input: &[f32],
    output: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    half_dim: usize,
) {
    use std::arch::x86_64::*;

    let mut i = 0;
    if half_dim >= 8 {
        while i + 8 <= half_dim {
            let x0 = _mm256_loadu_ps(input.as_ptr().add(i));
            let x1 = _mm256_loadu_ps(input.as_ptr().add(half_dim + i));
            let c = _mm256_loadu_ps(cos_table.as_ptr().add(i));
            let s = _mm256_loadu_ps(sin_table.as_ptr().add(i));

            // output[i] = x0*cos - x1*sin  (via FMA: fmsub not available, so mul + fnmadd)
            let x0c = _mm256_mul_ps(x0, c);
            let out_lo = _mm256_fnmadd_ps(x1, s, x0c); // x0c - x1*s

            // output[half+i] = x0*sin + x1*cos  (via FMA)
            let out_hi = _mm256_fmadd_ps(x0, s, _mm256_mul_ps(x1, c));

            _mm256_storeu_ps(output.as_mut_ptr().add(i), out_lo);
            _mm256_storeu_ps(output.as_mut_ptr().add(half_dim + i), out_hi);
            i += 8;
        }
    }
    // tail
    for j in i..half_dim {
        let x0 = input[j];
        let x1 = input[half_dim + j];
        output[j] = x0 * cos_table[j] - x1 * sin_table[j];
        output[half_dim + j] = x0 * sin_table[j] + x1 * cos_table[j];
    }
}

// ═════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── helpers ──────────────────────────────────────────────────

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

    // ── softmax ─────────────────────────────────────────────────

    #[test]
    fn softmax_basic() {
        let mut vals = vec![1.0, 2.0, 3.0, 4.0];
        softmax_simd(&mut vals);

        // Verify probabilities sum to 1
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "softmax sum={sum}");

        // Monotonically increasing
        for w in vals.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn softmax_matches_scalar() {
        let input = vec![0.5, -1.0, 2.3, 0.0, -0.7, 1.1, 3.0, -2.0, 0.3, 1.5];

        let mut simd_out = input.clone();
        softmax_simd(&mut simd_out);

        let mut scalar_out = input;
        softmax_scalar(&mut scalar_out);

        assert_close(&simd_out, &scalar_out, 1e-4, "softmax");
    }

    #[test]
    fn softmax_empty() {
        let mut vals: Vec<f32> = vec![];
        softmax_simd(&mut vals);
        assert!(vals.is_empty());
    }

    #[test]
    fn softmax_single() {
        let mut vals = vec![42.0];
        softmax_simd(&mut vals);
        assert!((vals[0] - 1.0).abs() < EPS);
    }

    #[test]
    fn softmax_large_values() {
        // Numerical stability: large values should not overflow
        let mut vals = vec![1000.0, 1001.0, 1002.0];
        softmax_simd(&mut vals);
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "softmax sum with large vals={sum}");
    }

    // ── rms_norm ────────────────────────────────────────────────

    #[test]
    fn rms_norm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let mut output = vec![0.0; 4];

        rms_norm_simd(&input, &weight, &mut output, 1e-6).expect("matching lengths");

        let rms = (30.0f32 / 4.0).sqrt();
        for i in 0..4 {
            let expected = input[i] / rms;
            assert!(
                (output[i] - expected).abs() < 1e-4,
                "rms_norm [{i}]: {} vs {expected}",
                output[i]
            );
        }
    }

    #[test]
    fn rms_norm_matches_scalar() {
        let input: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 0.3).collect();
        let weight: Vec<f32> = (0..17).map(|i| 0.5 + i as f32 * 0.1).collect();
        let mut simd_out = vec![0.0; 17];
        let mut scalar_out = vec![0.0; 17];

        rms_norm_simd(&input, &weight, &mut simd_out, 1e-5).expect("matching lengths");
        rms_norm_scalar(&input, &weight, &mut scalar_out, 1e-5);

        assert_close(&simd_out, &scalar_out, 1e-4, "rms_norm");
    }

    // ── silu ────────────────────────────────────────────────────

    #[test]
    fn silu_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0];
        let mut output = vec![0.0; 4];
        silu_simd(&input, &mut output).expect("matching lengths");

        // silu(0) = 0, silu(1) ≈ 0.7311
        assert!((output[0]).abs() < EPS);
        assert!((output[1] - 0.7311).abs() < 0.001);
    }

    #[test]
    fn silu_matches_scalar() {
        let input: Vec<f32> = (0..19).map(|i| (i as f32 - 9.0) * 0.5).collect();
        let mut simd_out = vec![0.0; 19];
        let mut scalar_out = vec![0.0; 19];

        silu_simd(&input, &mut simd_out).expect("matching lengths");
        silu_scalar(&input, &mut scalar_out);

        assert_close(&simd_out, &scalar_out, 1e-4, "silu");
    }

    // ── swiglu ──────────────────────────────────────────────────

    #[test]
    fn swiglu_basic() {
        let gate = vec![1.0, 0.0, -1.0];
        let up = vec![2.0, 3.0, 4.0];
        let mut output = vec![0.0; 3];

        swiglu_simd(&gate, &up, &mut output).expect("matching lengths");

        assert!((output[0] - silu_scalar_elem(1.0) * 2.0).abs() < EPS);
        assert!((output[1]).abs() < EPS);
        assert!((output[2] - silu_scalar_elem(-1.0) * 4.0).abs() < EPS);
    }

    #[test]
    fn swiglu_matches_scalar() {
        let gate: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.3).collect();
        let up: Vec<f32> = (0..20).map(|i| 1.0 + i as f32 * 0.1).collect();
        let mut simd_out = vec![0.0; 20];
        let mut scalar_out = vec![0.0; 20];

        swiglu_simd(&gate, &up, &mut simd_out).expect("matching lengths");
        swiglu_scalar(&gate, &up, &mut scalar_out);

        assert_close(&simd_out, &scalar_out, 1e-4, "swiglu");
    }

    // ── rope ────────────────────────────────────────────────────

    #[test]
    fn rope_identity_at_zero_angle() {
        // cos=1, sin=0 → identity transform
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cos_t = vec![1.0, 1.0];
        let sin_t = vec![0.0, 0.0];
        let mut output = vec![0.0; 4];

        rope_apply_simd(&input, &mut output, &cos_t, &sin_t).expect("matching lengths");

        assert_close(&output, &input, EPS, "rope identity");
    }

    #[test]
    fn rope_preserves_norm() {
        let input = vec![1.0, 0.0, 0.5, -0.5, 0.0, 1.0, -0.5, 0.5];
        let half = input.len() / 2;
        let cos_t: Vec<f32> = (0..half).map(|i| (i as f32 * 0.3).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut output = vec![0.0; input.len()];

        rope_apply_simd(&input, &mut output, &cos_t, &sin_t).expect("matching lengths");

        let in_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let out_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (in_norm - out_norm).abs() < 1e-4,
            "rope norm: {in_norm} vs {out_norm}"
        );
    }

    #[test]
    fn rope_matches_scalar() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.2).collect();
        let half = 8;
        let cos_t: Vec<f32> = (0..half).map(|i| (i as f32 * 0.5).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (i as f32 * 0.5).sin()).collect();

        let mut simd_out = vec![0.0; 16];
        let mut scalar_out = vec![0.0; 16];

        rope_apply_simd(&input, &mut simd_out, &cos_t, &sin_t).expect("matching lengths");
        rope_scalar(&input, &mut scalar_out, &cos_t, &sin_t, half);

        assert_close(&simd_out, &scalar_out, 1e-4, "rope");
    }

    // ── edge cases ──────────────────────────────────────────────

    #[test]
    fn softmax_all_same() {
        let mut vals = vec![1.0; 8];
        softmax_simd(&mut vals);
        for &v in &vals {
            assert!((v - 0.125).abs() < EPS);
        }
    }

    #[test]
    fn rms_norm_zero_input() {
        let input = vec![0.0; 4];
        let weight = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        // eps prevents division by zero
        rms_norm_simd(&input, &weight, &mut output, 1e-6).expect("matching lengths");
        // All outputs should be 0 (0 * weight * inv_rms)
        for &v in &output {
            assert!(v.abs() < 1e-2, "rms_norm zero input gave {v}");
        }
    }

    #[test]
    fn silu_negative_large() {
        let input = vec![-100.0; 4];
        let mut output = vec![0.0; 4];
        silu_simd(&input, &mut output).expect("matching lengths");
        for &v in &output {
            // silu(-100) ≈ -100 * sigmoid(-100) ≈ 0
            assert!(v.abs() < 1e-3, "silu(-100) = {v}");
        }
    }

    #[test]
    fn swiglu_odd_length() {
        let gate = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let up = vec![1.0; 5];
        let mut simd_out = vec![0.0; 5];
        let mut scalar_out = vec![0.0; 5];

        swiglu_simd(&gate, &up, &mut simd_out).expect("matching lengths");
        swiglu_scalar(&gate, &up, &mut scalar_out);

        assert_close(&simd_out, &scalar_out, 1e-4, "swiglu odd");
    }

    #[test]
    fn rope_small_dim() {
        // head_dim=2 (half_dim=1), smaller than any SIMD lane width
        let input = vec![1.0, 2.0];
        let cos_t = vec![0.5f32];
        let sin_t = vec![0.866f32]; // ≈ sin(60°)
        let mut output = vec![0.0; 2];

        rope_apply_simd(&input, &mut output, &cos_t, &sin_t).expect("matching lengths");

        let expected_0 = 1.0 * 0.5 - 2.0 * 0.866;
        let expected_1 = 1.0 * 0.866 + 2.0 * 0.5;
        assert!((output[0] - expected_0).abs() < 1e-3);
        assert!((output[1] - expected_1).abs() < 1e-3);
    }

    // ── length-contract errors (K-02 / M-01) ───────────────────────
    //
    // See also `tests/simd_float_ops_contract.rs` (release-profile
    // regression tests for these same contracts).

    #[test]
    fn rms_norm_rejects_mismatched_weight_length() {
        let input = vec![1.0; 8];
        let weight = vec![1.0; 4]; // too short
        let mut output = vec![0.0; 8];
        assert!(rms_norm_simd(&input, &weight, &mut output, EPS).is_err());
    }

    #[test]
    fn rms_norm_rejects_short_output() {
        let input = vec![1.0; 8];
        let weight = vec![1.0; 8];
        let mut output = vec![0.0; 4]; // too short
        assert!(rms_norm_simd(&input, &weight, &mut output, EPS).is_err());
    }

    #[test]
    fn silu_rejects_short_output() {
        let input = vec![1.0; 8];
        let mut output = vec![0.0; 4]; // too short
        assert!(silu_simd(&input, &mut output).is_err());
    }

    #[test]
    fn swiglu_rejects_mismatched_up_length() {
        let gate = vec![1.0; 8];
        let up = vec![1.0; 4]; // too short
        let mut output = vec![0.0; 8];
        assert!(swiglu_simd(&gate, &up, &mut output).is_err());
    }

    #[test]
    fn swiglu_rejects_short_output() {
        let gate = vec![1.0; 8];
        let up = vec![1.0; 8];
        let mut output = vec![0.0; 4]; // too short
        assert!(swiglu_simd(&gate, &up, &mut output).is_err());
    }

    #[test]
    fn rope_apply_rejects_mismatched_cos_table_length() {
        let input = vec![1.0; 8];
        let cos_t = vec![1.0; 2]; // half_dim should be 4
        let sin_t = vec![0.0; 4];
        let mut output = vec![0.0; 8];
        assert!(rope_apply_simd(&input, &mut output, &cos_t, &sin_t).is_err());
    }

    #[test]
    fn rope_apply_rejects_mismatched_sin_table_length() {
        let input = vec![1.0; 8];
        let cos_t = vec![1.0; 4];
        let sin_t = vec![0.0; 2]; // half_dim should be 4
        let mut output = vec![0.0; 8];
        assert!(rope_apply_simd(&input, &mut output, &cos_t, &sin_t).is_err());
    }

    #[test]
    fn rope_apply_rejects_short_output() {
        let input = vec![1.0; 8];
        let cos_t = vec![1.0; 4];
        let sin_t = vec![0.0; 4];
        let mut output = vec![0.0; 4]; // too short (need 8)
        assert!(rope_apply_simd(&input, &mut output, &cos_t, &sin_t).is_err());
    }

    // ── K-M3: body/tail rounding consistency ────────────────────────

    #[test]
    fn silu_bit_identical_regardless_of_offset_and_length() {
        // The same logical value must produce the same output whether it
        // lands in the vectorized "body" (a multiple-of-4 prefix) or the
        // padded "tail" (the last 1-3 elements) of a `silu_simd` call.
        let x = 0.734_f32;
        let mut reference: Option<f32> = None;
        for len in 1..=11usize {
            for pos in 0..len {
                let mut input = vec![0.25f32; len];
                input[pos] = x;
                let mut output = vec![0.0f32; len];
                silu_simd(&input, &mut output).expect("matching lengths");
                let got = output[pos];
                match reference {
                    None => reference = Some(got),
                    Some(r) => assert_eq!(
                        got.to_bits(),
                        r.to_bits(),
                        "silu({x}) at len={len} pos={pos} = {got}, expected bit-identical to {r}"
                    ),
                }
            }
        }
    }

    #[test]
    fn swiglu_bit_identical_regardless_of_offset_and_length() {
        let g = -1.375_f32;
        let u = 2.5_f32;
        let mut reference: Option<f32> = None;
        for len in 1..=11usize {
            for pos in 0..len {
                let mut gate = vec![0.5f32; len];
                let up = vec![u; len];
                gate[pos] = g;
                let mut output = vec![0.0f32; len];
                swiglu_simd(&gate, &up, &mut output).expect("matching lengths");
                let got = output[pos];
                match reference {
                    None => reference = Some(got),
                    Some(r) => assert_eq!(
                        got.to_bits(),
                        r.to_bits(),
                        "swiglu({g},{u}) at len={len} pos={pos} = {got}, expected bit-identical to {r}"
                    ),
                }
            }
        }
    }

    #[test]
    fn softmax_matches_scalar_across_body_and_tail_lengths() {
        // `softmax_neon`'s exp pass (K-M3) now runs the vectorized
        // `exp_neon_f32x4` for both the multiple-of-4 body and the padded
        // 1-3 element tail. Softmax's own normalization couples every
        // element together (the sum spans the whole buffer), so unlike
        // `silu`/`swiglu` there is no fixed-length prefix that stays
        // bit-identical as the buffer grows — instead, sweep every length
        // from 1 to 11 (covering "no tail", "tail only" and "body + tail")
        // and check each against the scalar reference, which is what would
        // catch a padding bug (wrong lane count, NaN/Inf leakage from the
        // discarded pad lanes, an off-by-one in `rem`).
        for len in 1..=11usize {
            let input: Vec<f32> = (0..len)
                .map(|i| (i as f32 - len as f32 / 2.0) * 0.7)
                .collect();

            let mut simd_out = input.clone();
            softmax_simd(&mut simd_out);

            let mut scalar_out = input;
            softmax_scalar(&mut scalar_out);

            assert_close(&simd_out, &scalar_out, 1e-4, &format!("softmax len={len}"));
        }
    }
}
