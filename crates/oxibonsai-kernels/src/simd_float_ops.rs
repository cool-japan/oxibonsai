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
//!
//! ## Body/tail bit-identity (K-M3)
//!
//! Each SIMD kernel splits its buffer into a full-width vector *body* (4
//! lanes on NEON, 8 on AVX2) and a `len % width` *tail*. If the two rounded
//! differently, a value's result would depend on where in the buffer it
//! lands. Covered now, each pinned by a test:
//!
//! - `silu_simd`, `swiglu_simd` and the `exp` pass of `softmax_simd`, on
//!   both tiers: every element — body and tail, the tail padded into a
//!   fixed-size stack buffer — runs through one vector core,
//!   `exp_neon_f32x4` / `silu_core_neon_f32x4` on aarch64 and
//!   `exp_avx2_f32x8` / `silu_core_avx2_f32x8` on x86_64.
//! - `rms_norm_simd`, on AVX2 and NEON: the scalar tail computes
//!   `weight * (input * inv_rms)`, the body's product order.
//! - `rope_apply_simd`, on AVX2: the scalar tail repeats the body's fused
//!   multiply-adds with `f32::mul_add`. On NEON the body's
//!   `vmlsq_f32`/`vmlaq_f32` are *unfused* multiply-then-subtract/add on
//!   AArch64, which the plain scalar tail already matches bit for bit.
//! - [`rms_norm_gated_simd`](crate::norms::rms_norm_gated_simd), on AVX2:
//!   the scalar tail multiplies in the body's order (on NEON its tail runs
//!   through the vector path).
//!
//! What remains by design: `norms.rs`'s AVX2 `sigmoid` / `sigmoid_mul` /
//! `softplus` kernels (and the `silu` factor of its AVX2 `rms_norm_gated`)
//! take a per-lane libm `exp`/`ln`, with one formula in body and tail —
//! self-consistent, but not on the polynomial. The scalar fallbacks are a
//! single loop.
//!
//! `softmax_simd`'s denominator is one sum over the whole row: both tiers
//! reduce the body lane-wise, then add the tail's `exp`s as one grouped
//! sub-sum (`sum += tail.iter().sum()`). On AVX2 that grouping replaced an
//! element-by-element `sum += e`, so for `n >= 8` with `n % 8 >= 2` the
//! denominator's association — and so possibly the last bit of every
//! output — differs from before. Between tiers the sum is reduced in a
//! different lane width and can differ in the last bit.
//!
//! The two tiers' cores are the same algorithm and constants (Cephes range
//! reduction, 5th-degree polynomial, one operation order of correctly
//! rounded IEEE-754 steps), pinned on each architecture by the scalar model
//! test: each core matches, bit for bit, one scalar transcription of that
//! sequence (`simd_float_ops_cephes_model.rs`) over the same sweeps.
//!
//! On x86_64 with AVX2+FMA this changed results: `silu_simd`,
//! `swiglu_simd` and `softmax_simd` now take their `exp` from the same
//! Cephes-style polynomial as the NEON path instead of libm's `f32::exp` —
//! within 1 ulp of the correctly rounded `e^x`, with inputs clamped to
//! `±88.376`, so `silu`/`swiglu` are only absolute-error-bounded below
//! `x = -88.376` and a softmax logit more than about `87.68` below its row
//! maximum comes out as exactly `+0.0` (libm's `exp` still gives a
//! subnormal there, down to about `103.97` below).

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
/// On x86_64 with AVX2+FMA the `exp` is now the same Cephes-style polynomial
/// as on the aarch64 NEON path (within 1 ulp, inputs clamped at `±88.376`),
/// not libm's `f32::exp`: a logit between about `87.68` and `103.97` below
/// the row maximum now yields exactly `+0.0` instead of a subnormal (see the
/// module's "Body/tail bit-identity (K-M3)" section).
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
///   `oxibonsai-image/src/te/forward.rs`) are all outside this crate,
///   and none of them use the return value today. Converting
///   this signature would force each of those three call sites to either
///   propagate a `Result` that can never be `Err` through their own
///   (infallible) callers, or discard it with `let _ = softmax_simd(...)`.
///   The latter is the error-swallowing idiom K-02/M-01 exists to eliminate
///   — manufacturing one to satisfy the letter of the spec at a real
///   invariant's expense would make the codebase worse in exactly the
///   dimension this crate improves.
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
/// On x86_64 with AVX2+FMA the `exp` is now the same Cephes-style polynomial
/// as on the aarch64 NEON path (within 1 ulp, inputs clamped at `±88.376`),
/// not libm's `f32::exp`, so below `input[i] = -88.376` the result is only
/// absolute-error-bounded (`|error| <= |input[i]| * 4.157e-39`).
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
/// On x86_64 with AVX2+FMA the `exp` inside `silu` is now the same
/// Cephes-style polynomial as on the aarch64 NEON path (within 1 ulp, inputs
/// clamped at `±88.376`), not libm's `f32::exp`, so below
/// `gate[i] = -88.376` the `silu` factor is only absolute-error-bounded
/// (`|error| <= |gate[i]| * 4.157e-39`).
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
/// `±EXP_HI` (`88.376_26`: as an `f32` that is `88.37625885…`, bits
/// `0x42b0c0a5`, just below `127.5 * ln 2 = 88.37626552…`, so the biased
/// exponent `fx + 127` always fits its 8-bit field) so a large `|x|`
/// saturates instead of overflowing the exponent field.
///
/// Note: because of that clamp, `silu`/`swiglu` built on this function are
/// only *absolute*-error-bounded (not asymptotically exact like the true
/// `x * e^x` tail) for very negative `x` — e.g. at `x = -100` this returns
/// `x / (1 + e^88.376…)` (~-4.156e-37) rather than the exact ~-3.72e-42. (The
/// `f32` scalar path does not return that exact value either:
/// `silu_scalar_elem(-100.0)` is `-0.0`, because `f32::exp(100.0)` overflows
/// to `+inf` and `x / inf` is `-0.0`.) So do not use this deep tail as a
/// determinism oracle.
///
/// This computes the *same* result regardless of which lane holds "real"
/// data — callers needing body/tail consistency (K-M3) must run a padded
/// tail through this exact function rather than falling back to `f32::exp()`,
/// so a value is bit-identical no matter where in the buffer it lands.
///
/// `neon_core_tests::exp_neon_f32x4_is_lane_wise_the_scalar_cephes_model`
/// pins this function bit-for-bit, lane by lane, to the scalar
/// transcription of its operation sequence (`cephes_model::exp_cephes_model`)
/// over the same sweep its x86_64 twin `exp_avx2_f32x8` is pinned over; the
/// model's own accuracy (within 1 ulp of the correctly rounded `e^x`) and
/// clamp behaviour are tested on every target.
///
/// `pub(crate)`: `norms.rs` imports this instead of holding
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
/// `neon_core_tests::silu_core_neon_f32x4_is_lane_wise_the_scalar_cephes_model`
/// pins it bit-for-bit to `cephes_model::silu_cephes_model`, as
/// `silu_core_avx2_f32x8` is pinned on x86_64.
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

    // ── 3. output = weight * (input * inv_rms) ───────────────
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
    // Tail: the body's product order (K-M3). `(weight * input) * inv_rms`
    // rounds differently, so the last 1-3 elements of a call could differ
    // by an ulp from the same values in the body.
    for j in i..n {
        output[j] = weight[j] * (input[j] * inv_rms);
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
    // Tail: already the body's arithmetic, bit for bit (K-M3). On AArch64
    // `vmlsq_f32(a, b, c)` / `vmlaq_f32(a, b, c)` are *unfused* — `a ∓ b*c`
    // with the product rounded first (`FMUL` then `FSUB`/`FADD`; only
    // `vfmsq_f32`/`vfmaq_f32` fuse) — so the body computes
    // `x0*cos - x1*sin` and `x1*cos + x0*sin` with every product rounded,
    // exactly what this plain scalar loop does (addition commutes exactly).
    // Unlike `rope_avx2`'s FMA body, nothing here needs `mul_add`.
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

/// Vectorized `exp(x)` for 8 `f32` lanes (K-M3) — the x86_64 twin of
/// `exp_neon_f32x4`.
///
/// Same algorithm, constants and operation order as the NEON function, lane
/// for lane: clamp to `±EXP_HI`; `fx = round(x * log2(e))` as the floor of
/// `x * log2(e) + 0.5` by truncate-then-correct; the two-constant (hi/lo)
/// reduction `r = x - fx*C1 - fx*C2`; the 5th-degree minimax polynomial in
/// FMA Horner steps; and `2^fx` by inserting the biased exponent straight
/// into an IEEE-754 bit pattern (`_mm256_slli_epi32::<23>(fx + 127)`). Each
/// NEON step maps onto one AVX2/FMA instruction that performs the same
/// correctly rounded IEEE-754 operation:
///
/// | `exp_neon_f32x4`                    | `exp_avx2_f32x8`                                 |
/// |-------------------------------------|--------------------------------------------------|
/// | `vminq_f32` / `vmaxq_f32`           | `_mm256_min_ps` / `_mm256_max_ps` (NaN note below) |
/// | `vfmaq_f32(a, b, c)` = `a + b*c`    | `_mm256_fmadd_ps(b, c, a)`                       |
/// | `vfmsq_f32(a, b, c)` = `a - b*c`    | `_mm256_fnmadd_ps(b, c, a)`                      |
/// | `vcvtq_s32_f32` (toward zero)       | `_mm256_cvttps_epi32`                            |
/// | `vcvtq_f32_s32`                     | `_mm256_cvtepi32_ps`                             |
/// | `vcgtq_f32` + `vandq_u32`           | `_mm256_cmp_ps::<_CMP_GT_OQ>` + `_mm256_and_ps`  |
/// | `vshlq_n_s32::<23>`                 | `_mm256_slli_epi32::<23>`                        |
///
/// so per lane this is the arithmetic a NEON lane performs on the same
/// non-NaN input (an FMA rounds once on both ISAs, and no step depends on
/// the vector width; a NaN stays NaN on both, payload aside). That is
/// pinned on each architecture rather than asserted across them:
/// `avx2_core_tests::exp_avx2_f32x8_is_lane_wise_the_scalar_cephes_model`
/// holds this function, and
/// `neon_core_tests::exp_neon_f32x4_is_lane_wise_the_scalar_cephes_model`
/// its NEON twin, bit-for-bit to one scalar `f32::mul_add` transcription of
/// the operation sequence (`cephes_model::exp_cephes_model`), over the same
/// sweep.
///
/// Accuracy, measured exhaustively — every one of the 2,237,399,041 `f32`
/// in `[-87, 87]` (reproducible in-repo with the `#[ignore]`d
/// `avx2_core_tests::exp_avx2_f32x8_is_within_1_ulp_of_the_correctly_rounded_exp_for_every_f32_in_minus_87_to_87`,
/// about 7 s), plus every `f32` in `[87, EXP_HI]` (part of the dense sweep
/// every test run): within 1 ulp of the correctly rounded `e^x` (96.54% of
/// `[-87, 87]` exactly correctly rounded) and within 1 ulp of glibc's
/// `expf` (`f32::exp`).
///
/// Inputs are clamped to `±EXP_HI` (`88.376_26`, the `f32`
/// `88.37625885…` just below `127.5 * ln 2`, so the biased exponent
/// `fx + 127` always fits its 8-bit field) and saturate instead of
/// overflowing: every `x >= EXP_HI`, `+inf` included, returns the finite
/// `exp(EXP_HI) = 2.4061436e38`. At the low end `2^fx` has no subnormal
/// encoding, so every `x <= -87.683_13` (where `fx` reaches `-127`) returns
/// exactly `+0.0` where `f32::exp` returns a subnormal below `8.4e-39` (an
/// absolute error under `f32::MIN_POSITIVE`); `-inf` gives `+0.0`, as
/// `f32::exp` does.
///
/// Note: because of that clamp, `silu`/`swiglu` built on this function are
/// only *absolute*-error-bounded (not asymptotically exact like the scalar
/// path's `x * e^x`) for very negative `x` — e.g. at `x = -100` this returns
/// `x / (1 + e^88.376…)` (~-4.156e-37) rather than the exact ~-3.72e-42 —
/// so do not use this deep tail as a determinism oracle. The bound is in
/// [`silu_core_avx2_f32x8`]'s doc comment.
///
/// NaN: x86 `MINPS`/`MAXPS` are not NEON's NaN-propagating `FMIN`/`FMAX` —
/// when either operand is NaN they return the *second* one. The clamp
/// therefore passes `x` second (`min(EXP_HI, x)`, `max(EXP_LO, x)`) so a
/// NaN lane stays NaN, as on NEON; the opposite operand order would quietly
/// turn NaN into `exp(±EXP_HI)` (e.g. for softmax's `inf - inf`).
///
/// This computes the *same* result regardless of which lane holds "real"
/// data — callers needing body/tail consistency (K-M3) must run a padded tail
/// through this exact function rather than falling back to `f32::exp()`, so
/// a value is bit-identical no matter where in the buffer it lands.
///
/// `pub(crate)` for the same reason as its NEON twin: a sibling module's
/// AVX2 path can import it rather than hold its own copy.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA (`#[target_feature]`: calling this on
/// a CPU without them is undefined behaviour). It touches no memory.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) unsafe fn exp_avx2_f32x8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

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

    let one = _mm256_set1_ps(1.0);
    // `x` goes second: MINPS/MAXPS return the second operand for a NaN lane.
    let x = _mm256_min_ps(_mm256_set1_ps(EXP_HI), x);
    let x = _mm256_max_ps(_mm256_set1_ps(EXP_LO), x);

    // fx = round(x * log2(e)), via truncate-then-correct-to-floor of (x*log2e + 0.5).
    let fx0 = _mm256_fmadd_ps(x, _mm256_set1_ps(LOG2EF), _mm256_set1_ps(0.5));
    let fx_trunc = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(fx0));
    let overshot = _mm256_cmp_ps::<_CMP_GT_OQ>(fx_trunc, fx0);
    let overshot_f = _mm256_and_ps(overshot, one);
    let fx = _mm256_sub_ps(fx_trunc, overshot_f);

    // r = x - fx*ln2, two-constant (hi/lo) reduction for precision.
    let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(EXP_C1), x);
    let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(EXP_C2), x);

    let z = _mm256_mul_ps(x, x);

    let mut y = _mm256_set1_ps(P0);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P2));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P3));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P4));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P5));
    y = _mm256_fmadd_ps(y, z, _mm256_add_ps(x, one));

    // 2^fx via direct exponent-field insertion.
    let emm0 = _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127));
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(emm0));

    _mm256_mul_ps(y, pow2n)
}

/// SIMD SiLU on 8 lanes: `x / (1 + exp(-x))` — the x86_64 twin of
/// `silu_core_neon_f32x4`.
///
/// Shares the exact same computation for every lane regardless of whether
/// it holds "real" input or tail padding (K-M3), and divides with
/// `_mm256_div_ps` (correctly rounded `VDIVPS`, the same IEEE-754 operation
/// as NEON's `vdivq_f32`). The previous AVX2 body computed
/// `x * (1 / (1 + e))` instead — a reciprocal *then* a multiply, rounding
/// twice — while its 1-7 element tail called `silu_scalar_elem`
/// (`x / (1 + e)`, rounding once), so the same `x` could come back 1 ulp
/// apart depending only on where in the buffer it landed (`silu(-2.5)` was
/// `0xbe42326a` in the body and `0xbe42326b` in the tail). The negation is
/// a sign-bit flip (`XOR -0.0`, `vnegq_f32`'s exact twin for every input),
/// so each lane runs the NEON arithmetic exactly.
///
/// Accuracy, over 6,000,001 evenly spaced points of `[-30, 30]`: at most
/// 3 ulp from the exact SiLU evaluated in `f64` and rounded once — pinned,
/// platform-independently, by
/// `avx2_core_tests::silu_core_avx2_f32x8_is_within_3_ulp_of_the_f64_silu_over_a_dense_sweep`,
/// which also holds a loose `1e-5` relative sanity bound against the
/// libm-based `silu_scalar_elem` (measured against glibc: at most `3.5e-7`
/// relative, 4 ulp). Bit for bit it is `cephes_model::silu_cephes_model`
/// (`avx2_core_tests::silu_core_avx2_f32x8_is_lane_wise_the_scalar_cephes_model`),
/// as `silu_core_neon_f32x4` is on aarch64.
///
/// For `x < -EXP_HI` the clamp inside [`exp_avx2_f32x8`] makes the result
/// exactly `x / (1 + exp(EXP_HI))`: absolute-error-only, with
/// `|error| <= |x| * 4.157e-39` (`-4.156e-37` at `x = -100`, where the exact
/// value is `-3.72e-42` and `silu_scalar_elem` returns `-0.0` because
/// `f32::exp(100)` overflows) and an unbounded relative error. `-inf` maps
/// to `-inf` (as on NEON; `silu_scalar_elem` gives NaN); NaN propagates.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA. It touches no memory.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) unsafe fn silu_core_avx2_f32x8(
    x: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0));
    let denom = _mm256_add_ps(one, exp_avx2_f32x8(neg_x));
    _mm256_div_ps(x, denom)
}

/// AVX2 softmax. The `exp` pass runs `exp_avx2_f32x8` over the 8-lane body
/// and the padded 1-7 element tail alike (K-M3).
///
/// Denominator association — a numeric change from the earlier libm-based
/// version, kept because it is `softmax_neon`'s grouping too: the body's
/// `exp`s accumulate lane-wise and reduce horizontally as
/// `((l0 + l4) + (l1 + l5)) + ((l2 + l6) + (l3 + l7))`; the tail's `exp`s are
/// then summed among themselves and that sub-sum added once,
/// `sum += out_buf[..rem].iter().sum()`. The earlier version added each
/// tail `exp` to the running sum in turn, `((body + t0) + t1) + …`, so for
/// `n >= 8` with `n % 8 >= 2` the denominator — and so possibly the last bit
/// of every output — can differ from it. (For `n < 8`, or a 1-element tail,
/// the two groupings coincide.)
/// `avx2_core_tests::softmax_body_and_tail_use_the_avx2_exp_core_bitwise`
/// pins this association, and the polynomial in body and tail, bit for bit.
///
/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support (`#[target_feature]`
/// UB otherwise). There is no cross-slice length contract here (see
/// `softmax_simd`): every raw-pointer load/store is bounded by
/// `values.len()`, and the 1-7 element tail of the `exp` pass never touches
/// `values` through a raw pointer — it is copied through fixed-size
/// 8-element stack buffers with bounds-checked `[i..n)` slice copies.
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
    // Vectorized exp (K-M3): the body and the 1-7 element tail both run
    // through `exp_avx2_f32x8` (the tail via a padded stack buffer) instead
    // of the body taking a per-lane libm `exp` and the tail `f32::exp()`, so
    // every element's `exp` comes from the one polynomial — lane for lane the
    // arithmetic `softmax_neon` runs through `exp_neon_f32x4`.
    let max_v = _mm256_set1_ps(max_val);
    let mut sum = 0.0f32;
    i = 0;
    if n >= 8 {
        let mut sum_vec = _mm256_setzero_ps();
        while i + 8 <= n {
            let v = _mm256_loadu_ps(values.as_ptr().add(i));
            let exp_v = exp_avx2_f32x8(_mm256_sub_ps(v, max_v));
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
    let rem = n - i;
    if rem > 0 {
        // Pad with `max_val` (⇒ shifted = 0 ⇒ exp = 1) so the unused lanes
        // can never contaminate `sum`/`values` even though they are
        // computed; only `out_buf[..rem]` is ever read back.
        let mut buf = [max_val; 8];
        buf[..rem].copy_from_slice(&values[i..n]);
        let exp_v = exp_avx2_f32x8(_mm256_sub_ps(_mm256_loadu_ps(buf.as_ptr()), max_v));
        let mut out_buf = [0.0f32; 8];
        _mm256_storeu_ps(out_buf.as_mut_ptr(), exp_v);
        values[i..n].copy_from_slice(&out_buf[..rem]);
        sum += out_buf[..rem].iter().sum::<f32>();
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

    // ── 3. output = weight * (input * inv_rms) ──────────────────
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
    // Tail: the body's product order (K-M3). `(weight * input) * inv_rms`
    // rounds differently, so the last 1-7 elements of a call could differ
    // by an ulp from the same values in the body.
    for j in i..n {
        output[j] = weight[j] * (input[j] * inv_rms);
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`silu_simd`) must guarantee `output.len() >= input.len()`; the
/// 8-lane body indexes raw pointers derived from `input`/`output` bounded
/// only by `input.len()` and does not re-validate lengths itself (K-02).
/// The 1-7 element tail (all of a call shorter than 8) touches neither
/// slice through a raw pointer: it is copied through fixed-size 8-element
/// stack buffers with bounds-checked `[i..n)` slice copies.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_avx2(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = input.len();
    let mut i = 0;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        _mm256_storeu_ps(output.as_mut_ptr().add(i), silu_core_avx2_f32x8(x));
        i += 8;
    }

    // Tail (1-7 remaining elements): pad into an 8-lane stack buffer and run
    // the SAME vector core (K-M3) instead of falling back to
    // `silu_scalar_elem`, which would compute a *different* rounding (libm
    // `exp` + scalar division vs. the body's polynomial `exp` +
    // `_mm256_div_ps`) for the last 1-7 elements of every
    // non-multiple-of-8-length call. The zero pad lanes compute
    // `silu(0) = 0` and are never read back.
    let rem = n - i;
    if rem > 0 {
        let mut buf = [0.0f32; 8];
        buf[..rem].copy_from_slice(&input[i..n]);
        let result = silu_core_avx2_f32x8(_mm256_loadu_ps(buf.as_ptr()));
        let mut out_buf = [0.0f32; 8];
        _mm256_storeu_ps(out_buf.as_mut_ptr(), result);
        output[i..n].copy_from_slice(&out_buf[..rem]);
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`swiglu_simd`) must guarantee `up.len() >= gate.len()` and
/// `output.len() >= gate.len()`; the 8-lane body indexes raw pointers
/// derived from `gate`/`up`/`output` bounded only by `gate.len()` and does
/// not re-validate lengths itself (K-02). The 1-7 element tail touches none
/// of the slices through a raw pointer: it is copied through fixed-size
/// 8-element stack buffers with bounds-checked `[i..n)` slice copies.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_avx2(gate: &[f32], up: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = gate.len();
    let mut i = 0;
    while i + 8 <= n {
        let g = _mm256_loadu_ps(gate.as_ptr().add(i));
        let u = _mm256_loadu_ps(up.as_ptr().add(i));
        let result = _mm256_mul_ps(silu_core_avx2_f32x8(g), u);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    // Tail: same rationale as `silu_avx2` — pad and reuse the identical
    // vector core so body and tail agree bit-for-bit (K-M3).
    let rem = n - i;
    if rem > 0 {
        let mut gbuf = [0.0f32; 8];
        let mut ubuf = [0.0f32; 8];
        gbuf[..rem].copy_from_slice(&gate[i..n]);
        ubuf[..rem].copy_from_slice(&up[i..n]);
        let g = _mm256_loadu_ps(gbuf.as_ptr());
        let u = _mm256_loadu_ps(ubuf.as_ptr());
        let result = _mm256_mul_ps(silu_core_avx2_f32x8(g), u);
        let mut out_buf = [0.0f32; 8];
        _mm256_storeu_ps(out_buf.as_mut_ptr(), result);
        output[i..n].copy_from_slice(&out_buf[..rem]);
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
    // Tail: the body's fused multiply-adds, element by element (K-M3). The
    // previous plain `x0*cos - x1*sin` / `x0*sin + x1*cos` rounded each
    // product on its own, so the last 1-7 pairs of a call could differ by an
    // ulp from the same values in the body. `f32::mul_add` rounds once (a
    // `vfmadd` under this function's `fma` target feature) and negating
    // `x1` is exact, so `(-x1).mul_add(s, x0c)` is the body's
    // `_mm256_fnmadd_ps(x1, s, x0c)` and `x0.mul_add(s, x1 * c)` its
    // `_mm256_fmadd_ps(x0, s, x1 * c)`, bit for bit.
    for j in i..half_dim {
        let x0 = input[j];
        let x1 = input[half_dim + j];
        let c = cos_table[j];
        let s = sin_table[j];
        let x0c = x0 * c;
        output[j] = (-x1).mul_add(s, x0c);
        output[half_dim + j] = x0.mul_add(s, x1 * c);
    }
}

// ═════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════

// Test modules in sibling files, so this module stays inside the 2000-line
// policy ceiling. `#[path]` keeps each a child module of `simd_float_ops`,
// reaching its private items through `use super::*` — the same shape
// `parallel.rs` uses:
//
// - `cephes_model` (every target): the scalar model of the `exp`/`silu`
//   cores both SIMD tiers are pinned to, the shared sweeps, and the model's
//   own accuracy and clamp tests.
// - `avx2_core_tests` (x86_64): `exp_avx2_f32x8` / `silu_core_avx2_f32x8`
//   accuracy (the `#[ignore]`d exhaustive sweep included), clamp edges, the
//   bit-exact pin to the model, and how `silu_avx2` / `swiglu_avx2` /
//   `softmax_avx2` route every element, body and tail, through them.
// - `neon_core_tests` (aarch64): `exp_neon_f32x4` / `silu_core_neon_f32x4`
//   pinned to the same model over the same sweeps, and `softmax_neon`'s
//   body and tail held to its core.
// - `body_tail_tests` (every target): `rms_norm_simd` / `rope_apply_simd`
//   body/tail bit-identity, whose tails are scalar code.
#[cfg(test)]
#[path = "simd_float_ops_cephes_model.rs"]
mod cephes_model;

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "simd_float_ops_avx2_tests.rs"]
mod avx2_core_tests;

#[cfg(all(test, target_arch = "aarch64"))]
#[path = "simd_float_ops_neon_tests.rs"]
mod neon_core_tests;

#[cfg(test)]
#[path = "simd_float_ops_body_tail_tests.rs"]
mod body_tail_tests;

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
    //
    // The vector body is a multiple-of-4 prefix on NEON and a multiple-of-8
    // prefix on AVX2; the rest — all of a call shorter than one vector — is
    // the padded tail. Lengths 1..=24 give the AVX2 kernel zero to three full
    // bodies combined with every tail size, and the NEON kernel every tail
    // size many times over. The per-architecture core tests (accuracy, clamp
    // edges, the bit-exact pin to the scalar Cephes model, softmax's body and
    // tail) live in `simd_float_ops_avx2_tests.rs` / `simd_float_ops_neon_tests.rs`,
    // the model in `simd_float_ops_cephes_model.rs`, and the `rms_norm` /
    // `rope` body/tail tests in `simd_float_ops_body_tail_tests.rs`.

    /// `silu_simd` probes: `tests/simd_float_ops_contract.rs`'s set plus the
    /// exp polynomial's worst-ulp neighbourhood (`-16.799_11`), both sides of
    /// the `exp` clamp (`±88.5`), a signed zero and a tiny magnitude.
    const SILU_PROBES: [f32; 12] = [
        0.734, -2.5, 5.0, -0.001, 12.25, -30.0, 3.0, -16.799_11, -88.5, 88.5, -0.0, 1.0e-6,
    ];

    /// `swiglu_simd` `(gate, up)` probes, the contract test's included.
    const SWIGLU_PROBES: [(f32, f32); 7] = [
        (-1.375, 2.5),
        (3.0, -0.5),
        (0.02, 7.0),
        (-2.5, 1.0),
        (-16.799_11, -3.25),
        (88.5, 0.5),
        (-88.5, 4.0),
    ];

    /// What the non-probe positions hold. NaN and `±inf` neighbours must not
    /// leak into the probe's lane: every core is strictly lane-wise.
    const NEIGHBOURS: [f32; 5] = [0.1, -7.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

    #[test]
    fn silu_bit_identical_regardless_of_offset_and_length() {
        // The same logical value must produce the same output whether it
        // lands in the vectorized "body" or the padded "tail" of a
        // `silu_simd` call, whatever its neighbours hold. The first
        // observation (len=1) is always a pure tail.
        for &x in &SILU_PROBES {
            let mut reference: Option<u32> = None;
            for &neighbour in &NEIGHBOURS {
                for len in 1..=24usize {
                    for pos in 0..len {
                        let mut input = vec![neighbour; len];
                        input[pos] = x;
                        let mut output = vec![0.0f32; len];
                        silu_simd(&input, &mut output).expect("matching lengths");
                        let got = output[pos].to_bits();
                        match reference {
                            None => reference = Some(got),
                            Some(r) => assert_eq!(
                                got, r,
                                "silu({x}) at len={len} pos={pos} (neighbours {neighbour}) = \
                                 {got:#010x}, expected bit-identical to the first observation \
                                 {r:#010x}"
                            ),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn swiglu_bit_identical_regardless_of_offset_and_length() {
        for &(g, u) in &SWIGLU_PROBES {
            let mut reference: Option<u32> = None;
            for &neighbour in &NEIGHBOURS {
                for len in 1..=24usize {
                    for pos in 0..len {
                        let mut gate = vec![neighbour; len];
                        let mut up = vec![neighbour; len];
                        gate[pos] = g;
                        up[pos] = u;
                        let mut output = vec![0.0f32; len];
                        swiglu_simd(&gate, &up, &mut output).expect("matching lengths");
                        let got = output[pos].to_bits();
                        match reference {
                            None => reference = Some(got),
                            Some(r) => assert_eq!(
                                got, r,
                                "swiglu({g},{u}) at len={len} pos={pos} (neighbours {neighbour}) \
                                 = {got:#010x}, expected bit-identical to the first observation \
                                 {r:#010x}"
                            ),
                        }
                    }
                }
            }
        }
    }

    /// Dense counterpart of the two probe sweeps: over 100,003 values
    /// spanning `[-100, 100]` (both sides of the `±88.376` `exp` clamp
    /// included), one long `silu_simd`/`swiglu_simd` call — where nearly
    /// every value sits in the vector body — must reproduce, bit-for-bit,
    /// calling each value alone, which always runs the padded tail.
    #[test]
    fn silu_and_swiglu_long_buffer_matches_length_one_calls_bitwise() {
        const N: usize = 100_003; // not a multiple of 4 or 8: the long call has a tail too
        let gate: Vec<f32> = (0..N)
            .map(|i| -100.0 + 200.0 * i as f32 / (N - 1) as f32)
            .collect();
        let up: Vec<f32> = (0..N).map(|i| ((i % 97) as f32 - 48.0) * 0.37).collect();

        let mut silu_long = vec![0.0f32; N];
        silu_simd(&gate, &mut silu_long).expect("matching lengths");
        let mut swiglu_long = vec![0.0f32; N];
        swiglu_simd(&gate, &up, &mut swiglu_long).expect("matching lengths");

        for (i, (&g, &u)) in gate.iter().zip(&up).enumerate() {
            let mut single = [0.0f32];
            silu_simd(&[g], &mut single).expect("matching lengths");
            assert_eq!(
                silu_long[i].to_bits(),
                single[0].to_bits(),
                "silu({g}) at index {i}: long call {} vs length-1 call {}",
                silu_long[i],
                single[0]
            );
            swiglu_simd(&[g], &[u], &mut single).expect("matching lengths");
            assert_eq!(
                swiglu_long[i].to_bits(),
                single[0].to_bits(),
                "swiglu({g},{u}) at index {i}: long call {} vs length-1 call {}",
                swiglu_long[i],
                single[0]
            );
        }
    }

    #[test]
    fn softmax_matches_scalar_across_body_and_tail_lengths() {
        // `softmax_neon`'s and `softmax_avx2`'s exp pass (K-M3) runs the
        // vectorized `exp_neon_f32x4` / `exp_avx2_f32x8` for both the
        // full-width body and the padded tail (1-3 elements on NEON, 1-7 on
        // AVX2). Softmax's own normalization couples every element together
        // (the sum spans the whole buffer), so unlike `silu`/`swiglu` there
        // is no fixed-length prefix that stays bit-identical as the buffer
        // grows — instead, sweep every length from 1 to 24 (covering "no
        // tail", "tail only" and "body + tail" on both tiers) and check each
        // against the scalar reference, which is what would catch a padding
        // bug (wrong lane count, NaN/Inf leakage from the discarded pad
        // lanes, an off-by-one in `rem`).
        for len in 1..=24usize {
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

    /// Attention masks feed `-inf` logits through `softmax_simd`: on every
    /// tier a masked entry must come out as an exact `+0.0` — in the vector
    /// body and in the padded tail alike — while the unmasked entries still
    /// match the scalar reference and sum to one.
    #[test]
    fn softmax_masked_logits_are_exact_zeros_in_body_and_tail() {
        for len in 2..=24usize {
            for phase in 0..3usize {
                // Every third logit masked, so no row is fully masked.
                let input: Vec<f32> = (0..len)
                    .map(|i| {
                        if i % 3 == phase {
                            f32::NEG_INFINITY
                        } else {
                            (i as f32 * 0.37).sin() * 4.0
                        }
                    })
                    .collect();
                let mut simd_out = input.clone();
                softmax_simd(&mut simd_out);
                let mut scalar_out = input.clone();
                softmax_scalar(&mut scalar_out);

                let mut unmasked_sum = 0.0f32;
                for (i, (&x, (&v, &s))) in input
                    .iter()
                    .zip(simd_out.iter().zip(&scalar_out))
                    .enumerate()
                {
                    if x == f32::NEG_INFINITY {
                        assert_eq!(
                            v.to_bits(),
                            0.0f32.to_bits(),
                            "masked logit at len={len} pos={i} must be +0.0, got {v}"
                        );
                    } else {
                        assert!(
                            (v - s).abs() < 1e-6,
                            "softmax len={len} pos={i}: {v} vs scalar {s}"
                        );
                        unmasked_sum += v;
                    }
                }
                assert!(
                    (unmasked_sum - 1.0).abs() < 1e-5,
                    "len={len} phase={phase}: unmasked probabilities sum to {unmasked_sum}"
                );
            }
        }
    }
}
