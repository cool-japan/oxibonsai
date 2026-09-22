//! ARM NEON kernels for the blockwise Hadamard transform (see
//! [`crate::hadamard`] for the public API, the ordering contract and the
//! `ops.cpp` reference this mirrors).
//!
//! NEON is baseline on every AArch64 target (unlike x86-64's AVX2, it needs
//! no runtime feature detection), so every function here is a plain safe
//! `fn` that wraps its intrinsics in `unsafe { .. }` blocks internally —
//! matching [`crate::simd_float_ops`]'s dispatch style — rather than an
//! `unsafe fn` gated by `#[target_feature]`.
//!
//! Only the lengths that do not divide evenly into a 4-lane `float32x4_t`
//! register need special handling; everything else (`len >= 4`, and the two
//! plain elementwise passes at any length) is a straightforward contiguous
//! vector load/store. `len == 1` is left to
//! [`crate::hadamard`]'s scalar fallback: a 4-lane register at `len == 1`
//! holds *two* independent 2-element groups, which needs an interleave
//! (`vtrn`) to separate the "u" and "v" half of each group — a real but
//! narrow (1 of the 10 passes for `block == 1024`) optimization that this
//! package skips in favor of the lower-risk, already-exhaustively-tested
//! scalar path (see `crate::hadamard`'s module docs for the cost budget that
//! makes this an easy trade).

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// `buf[i] ← buf[i] * scale`, the unsigned `1/sqrt(block)` load pass.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn scale_inplace(buf: &mut [f32], scale: f32) {
    let n = buf.len();
    let mut i = 0usize;
    if n >= 4 {
        // SAFETY: NEON is always available on aarch64; the loop only reads
        // and writes `buf[i..i+4]` for `i + 4 <= n`.
        unsafe {
            let s = vdupq_n_f32(scale);
            while i + 4 <= n {
                let v = vld1q_f32(buf.as_ptr().add(i));
                vst1q_f32(buf.as_mut_ptr().add(i), vmulq_f32(v, s));
                i += 4;
            }
        }
    }
    for v in &mut buf[i..n] {
        *v *= scale;
    }
}

/// `buf[i] ← buf[i] * signs[i] * scale`. `signs.len()` must be `>=
/// buf.len()` (the caller in [`crate::hadamard`] always slices `signs` to
/// exactly `buf.len()` first).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn scale_and_signs(buf: &mut [f32], signs: &[f32], scale: f32) {
    let n = buf.len();
    let mut i = 0usize;
    if n >= 4 {
        // SAFETY: NEON is always available on aarch64; both loads/the store
        // only touch `[i..i+4)` for `i + 4 <= n <= signs.len()`.
        unsafe {
            let s = vdupq_n_f32(scale);
            while i + 4 <= n {
                let v = vld1q_f32(buf.as_ptr().add(i));
                let sg = vld1q_f32(signs.as_ptr().add(i));
                let r = vmulq_f32(vmulq_f32(v, sg), s);
                vst1q_f32(buf.as_mut_ptr().add(i), r);
                i += 4;
            }
        }
    }
    for j in i..n {
        buf[j] = buf[j] * signs[j] * scale;
    }
}

/// A butterfly pass for `len >= 4` (a multiple of 4, since `len` is itself a
/// power of two `>= 4`): every group of `2 * len` elements is processed as
/// plain contiguous 4-wide loads — `ops.cpp`'s own "SIMD passes" section
/// (`:11947-11962`), just NEON instead of `GGML_F32_VEC_*`.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn butterfly_pass_wide(buf: &mut [f32], len: usize) {
    let n = buf.len();
    let mut i = 0usize;
    // SAFETY: NEON is always available on aarch64. `len` is a multiple of 4
    // and `n` is an exact multiple of `2 * len` (both are powers of two with
    // `len < n`), so every `add(i + j)` / `add(i + len + j)` below stays
    // within `buf`.
    unsafe {
        while i < n {
            let mut j = 0usize;
            while j < len {
                let u = vld1q_f32(buf.as_ptr().add(i + j));
                let v = vld1q_f32(buf.as_ptr().add(i + len + j));
                vst1q_f32(buf.as_mut_ptr().add(i + j), vaddq_f32(u, v));
                vst1q_f32(buf.as_mut_ptr().add(i + len + j), vsubq_f32(u, v));
                j += 4;
            }
            i += 2 * len;
        }
    }
}

/// The `len == 2` butterfly pass: one group (`2 * len == 4` elements) fills
/// exactly one `float32x4_t` register, so the pass is a low/high-lane split
/// (`u` = low 2 lanes, `v` = high 2 lanes) rather than a strided load — no
/// interleave/deinterleave needed, unlike `len == 1`.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn butterfly_pass_len2(buf: &mut [f32]) {
    let n = buf.len();
    let mut i = 0usize;
    // SAFETY: NEON is always available on aarch64; `n` is always a multiple
    // of 4 here (this is only ever called for `len == 2`, and `len < n`
    // with both powers of two implies `n >= 2 * len == 4`), so every
    // `add(i)` load/store below stays within `buf`.
    unsafe {
        while i < n {
            // v4 = [x0, x1, x2, x3] = one full group: u = [x0, x1], v = [x2, x3].
            let v4 = vld1q_f32(buf.as_ptr().add(i));
            let u2 = vget_low_f32(v4);
            let v2 = vget_high_f32(v4);
            let sum = vadd_f32(u2, v2); // [x0+x2, x1+x3]
            let diff = vsub_f32(u2, v2); // [x0-x2, x1-x3]
            vst1q_f32(buf.as_mut_ptr().add(i), vcombine_f32(sum, diff));
            i += 4;
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    fn lcg_random_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 8.0
            })
            .collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "{label} mismatch at [{i}]: {x} vs {y}"
            );
        }
    }

    fn scale_scalar_ref(buf: &[f32], scale: f32) -> Vec<f32> {
        buf.iter().map(|v| v * scale).collect()
    }

    fn scale_and_signs_scalar_ref(buf: &[f32], signs: &[f32], scale: f32) -> Vec<f32> {
        buf.iter()
            .zip(signs.iter())
            .map(|(v, s)| v * s * scale)
            .collect()
    }

    fn butterfly_pass_scalar_ref(buf: &[f32], len: usize) -> Vec<f32> {
        let mut out = buf.to_vec();
        let n = out.len();
        let mut i = 0usize;
        while i < n {
            for j in 0..len {
                let u = out[i + j];
                let v = out[i + len + j];
                out[i + j] = u + v;
                out[i + len + j] = u - v;
            }
            i += 2 * len;
        }
        out
    }

    #[test]
    fn scale_inplace_matches_scalar_reference() {
        for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 1024] {
            let input = lcg_random_vec(len, len as u64 + 1);
            let mut got = input.clone();
            scale_inplace(&mut got, 0.0625);
            let expected = scale_scalar_ref(&input, 0.0625);
            assert_close(&got, &expected, 1e-6, &format!("scale_inplace len={len}"));
        }
    }

    #[test]
    fn scale_and_signs_matches_scalar_reference() {
        for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 1024] {
            let input = lcg_random_vec(len, len as u64 + 2);
            let signs: Vec<f32> = (0..len)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let mut got = input.clone();
            scale_and_signs(&mut got, &signs, 0.03125);
            let expected = scale_and_signs_scalar_ref(&input, &signs, 0.03125);
            assert_close(&got, &expected, 1e-6, &format!("scale_and_signs len={len}"));
        }
    }

    #[test]
    fn butterfly_pass_wide_matches_scalar_reference() {
        for &(block, len) in &[(8usize, 4usize), (16, 4), (16, 8), (1024, 512), (1024, 4)] {
            let input = lcg_random_vec(block, (block * 31 + len) as u64 + 3);
            let mut got = input.clone();
            butterfly_pass_wide(&mut got, len);
            let expected = butterfly_pass_scalar_ref(&input, len);
            assert_close(
                &got,
                &expected,
                1e-4,
                &format!("butterfly_pass_wide block={block} len={len}"),
            );
        }
    }

    #[test]
    fn butterfly_pass_len2_matches_scalar_reference() {
        for &block in &[4usize, 8, 32, 1024] {
            let input = lcg_random_vec(block, block as u64 + 4);
            let mut got = input.clone();
            butterfly_pass_len2(&mut got);
            let expected = butterfly_pass_scalar_ref(&input, 2);
            assert_close(
                &got,
                &expected,
                1e-4,
                &format!("butterfly_pass_len2 block={block}"),
            );
        }
    }
}
