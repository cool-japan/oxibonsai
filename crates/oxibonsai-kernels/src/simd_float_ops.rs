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

// ─── Softmax (in-place) ──────────────────────────────────────────

/// Numerically-stable softmax, computed in-place.
///
/// 1. Find `max` via SIMD reduction
/// 2. Subtract max and compute `exp` via SIMD
/// 3. Sum via SIMD reduction
/// 4. Divide every element by the sum
///
/// Falls back to scalar on unsupported platforms.
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
/// # Panics
///
/// Panics if `weight.len() != input.len()` or `output.len() < input.len()`.
#[inline]
pub fn rms_norm_simd(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len();
    assert_eq!(n, weight.len());
    assert!(output.len() >= n);

    #[cfg(target_arch = "aarch64")]
    {
        rms_norm_neon(input, weight, output, eps);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { rms_norm_avx2(input, weight, output, eps) };
            return;
        }
        rms_norm_scalar(input, weight, output, eps);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        rms_norm_scalar(input, weight, output, eps);
    }
}

// ─── SiLU element-wise ───────────────────────────────────────────

/// Element-wise SiLU (Swish): `output[i] = input[i] / (1 + exp(-input[i]))`.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[inline]
pub fn silu_simd(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert!(output.len() >= n);

    #[cfg(target_arch = "aarch64")]
    {
        silu_neon(input, output);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { silu_avx2(input, output) };
            return;
        }
        silu_scalar(input, output);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        silu_scalar(input, output);
    }
}

// ─── SwiGLU ──────────────────────────────────────────────────────

/// SwiGLU: `output[i] = silu(gate[i]) * up[i]`.
///
/// # Panics
///
/// Panics if `up.len() != gate.len()` or `output.len() < gate.len()`.
#[inline]
pub fn swiglu_simd(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    assert_eq!(n, up.len());
    assert!(output.len() >= n);

    #[cfg(target_arch = "aarch64")]
    {
        swiglu_neon(gate, up, output);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { swiglu_avx2(gate, up, output) };
            return;
        }
        swiglu_scalar(gate, up, output);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        swiglu_scalar(gate, up, output);
    }
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
/// # Panics
///
/// Panics if either table has a different length or `output.len() < input.len()`.
#[inline]
pub fn rope_apply_simd(input: &[f32], output: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    let head_dim = input.len();
    let half_dim = head_dim / 2;
    assert_eq!(cos_table.len(), half_dim);
    assert_eq!(sin_table.len(), half_dim);
    assert!(output.len() >= head_dim);

    #[cfg(target_arch = "aarch64")]
    {
        rope_neon(input, output, cos_table, sin_table, half_dim);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { rope_avx2(input, output, cos_table, sin_table, half_dim) };
            return;
        }
        rope_scalar(input, output, cos_table, sin_table, half_dim);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        rope_scalar(input, output, cos_table, sin_table, half_dim);
    }
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
    // We process 4-at-a-time, calling f32::exp() per element.
    // This still wins from reduced branch overhead and cache-friendly access.
    let mut sum = 0.0f32;
    i = 0;

    if n >= 4 {
        unsafe {
            let max_v = vdupq_n_f32(max_val);
            let mut sum_vec = vdupq_n_f32(0.0);
            while i + 4 <= n {
                let v = vld1q_f32(values.as_ptr().add(i));
                let shifted = vsubq_f32(v, max_v);
                // exp per lane (scalar calls, still faster than pure-scalar loop)
                let e0 = vgetq_lane_f32::<0>(shifted).exp();
                let e1 = vgetq_lane_f32::<1>(shifted).exp();
                let e2 = vgetq_lane_f32::<2>(shifted).exp();
                let e3 = vgetq_lane_f32::<3>(shifted).exp();
                let exp_v = float32x4_from_array([e0, e1, e2, e3]);
                vst1q_f32(values.as_mut_ptr().add(i), exp_v);
                sum_vec = vaddq_f32(sum_vec, exp_v);
                i += 4;
            }
            let pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
            let pair2 = vpadd_f32(pair, pair);
            sum = vget_lane_f32::<0>(pair2);
        }
    }
    for v in &mut values[i..n] {
        let e = (*v - max_val).exp();
        *v = e;
        sum += e;
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

/// Helper to build a `float32x4_t` from 4 scalars on NEON.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn float32x4_from_array(arr: [f32; 4]) -> std::arch::aarch64::float32x4_t {
    std::arch::aarch64::vld1q_f32(arr.as_ptr())
}

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

#[cfg(target_arch = "aarch64")]
#[inline]
fn silu_neon(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    // Process 4 elements at a time, scalar exp per lane.
    let mut i = 0;
    if n >= 4 {
        unsafe {
            use std::arch::aarch64::*;
            let one = vdupq_n_f32(1.0);
            while i + 4 <= n {
                let x = vld1q_f32(input.as_ptr().add(i));
                // negate
                let neg_x = vnegq_f32(x);
                // exp per lane
                let e0 = vgetq_lane_f32::<0>(neg_x).exp();
                let e1 = vgetq_lane_f32::<1>(neg_x).exp();
                let e2 = vgetq_lane_f32::<2>(neg_x).exp();
                let e3 = vgetq_lane_f32::<3>(neg_x).exp();
                let exp_neg = float32x4_from_array([e0, e1, e2, e3]);
                // sigmoid = 1 / (1 + exp(-x))
                let denom = vaddq_f32(one, exp_neg);

                // Newton-Raphson reciprocal: 2 iterations
                let recip_est = vrecpeq_f32(denom);
                let recip_1 = vmulq_f32(vrecpsq_f32(denom, recip_est), recip_est);
                let recip_2 = vmulq_f32(vrecpsq_f32(denom, recip_1), recip_1);

                let result = vmulq_f32(x, recip_2);
                vst1q_f32(output.as_mut_ptr().add(i), result);
                i += 4;
            }
        }
    }
    for j in i..n {
        output[j] = silu_scalar_elem(input[j]);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn swiglu_neon(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    let mut i = 0;
    if n >= 4 {
        unsafe {
            use std::arch::aarch64::*;
            let one = vdupq_n_f32(1.0);
            while i + 4 <= n {
                let g = vld1q_f32(gate.as_ptr().add(i));
                let u = vld1q_f32(up.as_ptr().add(i));
                // silu(gate)
                let neg_g = vnegq_f32(g);
                let e0 = vgetq_lane_f32::<0>(neg_g).exp();
                let e1 = vgetq_lane_f32::<1>(neg_g).exp();
                let e2 = vgetq_lane_f32::<2>(neg_g).exp();
                let e3 = vgetq_lane_f32::<3>(neg_g).exp();
                let exp_neg = float32x4_from_array([e0, e1, e2, e3]);
                let denom = vaddq_f32(one, exp_neg);
                let recip_est = vrecpeq_f32(denom);
                let recip_1 = vmulq_f32(vrecpsq_f32(denom, recip_est), recip_est);
                let recip_2 = vmulq_f32(vrecpsq_f32(denom, recip_1), recip_1);
                let silu_g = vmulq_f32(g, recip_2);
                // silu(gate) * up
                let result = vmulq_f32(silu_g, u);
                vst1q_f32(output.as_mut_ptr().add(i), result);
                i += 4;
            }
        }
    }
    for j in i..n {
        output[j] = silu_scalar_elem(gate[j]) * up[j];
    }
}

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

    fn assert_panics_without_writing(len: usize, operation: impl FnOnce(&mut [f32])) {
        let mut output = vec![42.0; len];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            operation(&mut output);
        }));
        assert!(result.is_err());
        assert_eq!(output, vec![42.0; len]);
    }

    #[test]
    fn rms_norm_rejects_invalid_lengths_before_writing() {
        for n in [1, 4, 8, 9] {
            let input = vec![1.0; n];
            for weight_len in [n - 1, n + 1] {
                assert_panics_without_writing(n, |output| {
                    rms_norm_simd(&input, &vec![1.0; weight_len], output, 1e-6);
                });
            }
            assert_panics_without_writing(n - 1, |output| {
                rms_norm_simd(&input, &input, output, 1e-6);
            });
        }
    }

    #[test]
    fn silu_rejects_invalid_lengths_before_writing() {
        for n in [1, 4, 8, 9] {
            assert_panics_without_writing(n - 1, |output| {
                silu_simd(&vec![1.0; n], output);
            });
        }
    }

    #[test]
    fn swiglu_rejects_invalid_lengths_before_writing() {
        for n in [1, 4, 8, 9] {
            let gate = vec![1.0; n];
            for up_len in [n - 1, n + 1] {
                assert_panics_without_writing(n, |output| {
                    swiglu_simd(&gate, &vec![1.0; up_len], output);
                });
            }
            assert_panics_without_writing(n - 1, |output| {
                swiglu_simd(&gate, &gate, output);
            });
        }
    }

    #[test]
    fn rope_rejects_invalid_lengths_before_writing() {
        for half_dim in [1, 4, 8, 9] {
            let input = vec![1.0; 2 * half_dim];
            let table = vec![1.0; half_dim];
            for table_len in [half_dim - 1, half_dim + 1] {
                let invalid_table = vec![1.0; table_len];
                assert_panics_without_writing(input.len(), |output| {
                    rope_apply_simd(&input, output, &invalid_table, &table);
                });
                assert_panics_without_writing(input.len(), |output| {
                    rope_apply_simd(&input, output, &table, &invalid_table);
                });
            }
            assert_panics_without_writing(input.len() - 1, |output| {
                rope_apply_simd(&input, output, &table, &table);
            });
        }
    }

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

        rms_norm_simd(&input, &weight, &mut output, 1e-6);

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

        rms_norm_simd(&input, &weight, &mut simd_out, 1e-5);
        rms_norm_scalar(&input, &weight, &mut scalar_out, 1e-5);

        assert_close(&simd_out, &scalar_out, 1e-4, "rms_norm");
    }

    // ── silu ────────────────────────────────────────────────────

    #[test]
    fn silu_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0];
        let mut output = vec![0.0; 4];
        silu_simd(&input, &mut output);

        // silu(0) = 0, silu(1) ≈ 0.7311
        assert!((output[0]).abs() < EPS);
        assert!((output[1] - 0.7311).abs() < 0.001);
    }

    #[test]
    fn silu_matches_scalar() {
        let input: Vec<f32> = (0..19).map(|i| (i as f32 - 9.0) * 0.5).collect();
        let mut simd_out = vec![0.0; 19];
        let mut scalar_out = vec![0.0; 19];

        silu_simd(&input, &mut simd_out);
        silu_scalar(&input, &mut scalar_out);

        assert_close(&simd_out, &scalar_out, 1e-4, "silu");
    }

    // ── swiglu ──────────────────────────────────────────────────

    #[test]
    fn swiglu_basic() {
        let gate = vec![1.0, 0.0, -1.0];
        let up = vec![2.0, 3.0, 4.0];
        let mut output = vec![0.0; 3];

        swiglu_simd(&gate, &up, &mut output);

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

        swiglu_simd(&gate, &up, &mut simd_out);
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

        rope_apply_simd(&input, &mut output, &cos_t, &sin_t);

        assert_close(&output, &input, EPS, "rope identity");
    }

    #[test]
    fn rope_preserves_norm() {
        let input = vec![1.0, 0.0, 0.5, -0.5, 0.0, 1.0, -0.5, 0.5];
        let half = input.len() / 2;
        let cos_t: Vec<f32> = (0..half).map(|i| (i as f32 * 0.3).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut output = vec![0.0; input.len()];

        rope_apply_simd(&input, &mut output, &cos_t, &sin_t);

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

        rope_apply_simd(&input, &mut simd_out, &cos_t, &sin_t);
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
        rms_norm_simd(&input, &weight, &mut output, 1e-6);
        // All outputs should be 0 (0 * weight * inv_rms)
        for &v in &output {
            assert!(v.abs() < 1e-2, "rms_norm zero input gave {v}");
        }
    }

    #[test]
    fn silu_negative_large() {
        let input = vec![-100.0; 4];
        let mut output = vec![0.0; 4];
        silu_simd(&input, &mut output);
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

        swiglu_simd(&gate, &up, &mut simd_out);
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

        rope_apply_simd(&input, &mut output, &cos_t, &sin_t);

        let expected_0 = 1.0 * 0.5 - 2.0 * 0.866;
        let expected_1 = 1.0 * 0.866 + 2.0 * 0.5;
        assert!((output[0] - expected_0).abs() < 1e-3);
        assert!((output[1] - expected_1).abs() < 1e-3);
    }
}
