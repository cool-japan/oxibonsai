//! Fast f32 GEMM for the DiT forward: `out[m, n] = Σ_k in[m, k] * w[n, k]`.
//!
//! Both operands are row-major with the contraction dimension `k` innermost
//! (`input` is `[m, k]`, `weight` is `[n, k]`), so this is `C = A · Bᵀ`, a pure
//! dot-product GEMM that vectorises cleanly. The kernel:
//! - blocks the `n` (weight-row) loop by 4 so each loaded `input[m]` cache line
//!   is reused across four weight rows (register blocking via 4 FMA accumulators);
//! - uses NEON FMA on aarch64 (scalar fallback elsewhere);
//! - parallelises over `m` rows with scoped threads.
//!
//! Ternary weights are dequantised to f32 once (cached by the caller) and fed
//! through the same kernel, which is dramatically faster than the per-call
//! bit-unpacking ternary GEMV for the repeated-forward parity loop.

/// Compute `out[m, n] = Σ_k input[m, k] * weight[n, k]` into `out` (`[m, n]`).
///
/// `input` is `[m, k]`, `weight` is `[n, k]`, both row-major (so this is
/// `C = A · Bᵀ`). Parallelised over `n` (weight rows): each thread owns a
/// contiguous range of weight rows and streams every input row against them,
/// computing its output columns. Splitting on `n` (rather than `m`) means the
/// large `weight` operand is read only once in total, while the smaller `input`
/// operand is the one re-read per thread — minimising memory traffic for the
/// `n ≫ m` projections that dominate the DiT.
pub fn gemm_abt(input: &[f32], weight: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(weight.len(), n * k);
    debug_assert_eq!(out.len(), m * n);

    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(n.max(1));

    if threads <= 1 || n < 8 || m == 0 {
        gemm_ncols(input, weight, out, m, n, k, 0, n);
        return;
    }

    // Split the n range into `threads` contiguous blocks; each thread computes
    // its columns into the shared output (disjoint column sets, no aliasing).
    let per = n.div_ceil(threads);
    let out_ptr = SyncPtr(out.as_mut_ptr());
    std::thread::scope(|scope| {
        let mut col0 = 0usize;
        while col0 < n {
            let col1 = (col0 + per).min(n);
            let start = col0;
            let end = col1;
            let input = &input;
            let weight = &weight;
            let op = out_ptr;
            scope.spawn(move || {
                // Force capture of the whole `SyncPtr` (Send), not the bare
                // `*mut f32` field (edition-2021 disjoint capture would
                // otherwise capture `op.0`, which is not `Send`).
                let op = op;
                // SAFETY: each thread writes only columns [start, end) of each
                // row; column ranges are disjoint across threads, so the
                // &mut aliasing is non-overlapping.
                let out = unsafe { std::slice::from_raw_parts_mut(op.0, m * n) };
                gemm_ncols(input, weight, out, m, n, k, start, end);
            });
            col0 = col1;
        }
    });
}

/// Raw pointer wrapper to move `*mut f32` into scoped threads. Safe here because
/// the threads write disjoint column ranges of the same `[m, n]` buffer.
#[derive(Clone, Copy)]
struct SyncPtr(*mut f32);
// SAFETY: callers guarantee disjoint, non-overlapping writes per thread.
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

/// Compute output columns `[col_start, col_end)` for every row.
///
/// Uses a 4×4 register micro-kernel (4 input rows × 4 weight rows, 16 FMA
/// accumulators) so each loaded k-element contributes to 8 products — ~4× the
/// arithmetic intensity of a plain dot product, which is what lifts this above
/// the memory-bandwidth wall for the large DiT projections. Edge rows/cols fall
/// back to narrower dot products.
#[allow(clippy::too_many_arguments)]
fn gemm_ncols(
    input: &[f32],
    weight: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    col_start: usize,
    col_end: usize,
) {
    let m4 = m - (m % 4);
    let mut c = col_start;
    let col4 = col_start + ((col_end - col_start) / 4) * 4;
    while c < col4 {
        let w0 = &weight[c * k..c * k + k];
        let w1 = &weight[(c + 1) * k..(c + 1) * k + k];
        let w2 = &weight[(c + 2) * k..(c + 2) * k + k];
        let w3 = &weight[(c + 3) * k..(c + 3) * k + k];
        let mut r = 0;
        while r < m4 {
            let i0 = &input[r * k..r * k + k];
            let i1 = &input[(r + 1) * k..(r + 1) * k + k];
            let i2 = &input[(r + 2) * k..(r + 2) * k + k];
            let i3 = &input[(r + 3) * k..(r + 3) * k + k];
            let block = micro4x4(i0, i1, i2, i3, w0, w1, w2, w3, k);
            for (ri, vals) in block.iter().enumerate() {
                let base = (r + ri) * n + c;
                out[base] = vals[0];
                out[base + 1] = vals[1];
                out[base + 2] = vals[2];
                out[base + 3] = vals[3];
            }
            r += 4;
        }
        while r < m {
            let ir = &input[r * k..r * k + k];
            let (a0, a1, a2, a3) = dot4(ir, w0, w1, w2, w3, k);
            let base = r * n + c;
            out[base] = a0;
            out[base + 1] = a1;
            out[base + 2] = a2;
            out[base + 3] = a3;
            r += 1;
        }
        c += 4;
    }
    // Remaining columns (< 4) as single dot products.
    while c < col_end {
        let w = &weight[c * k..c * k + k];
        for r in 0..m {
            out[r * n + c] = dot1(&input[r * k..r * k + k], w, k);
        }
        c += 1;
    }
}

/// 4×4 micro-kernel: returns `out[ri][ci] = Σ_k i{ri}[k] * w{ci}[k]`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn micro4x4(
    i0: &[f32],
    i1: &[f32],
    i2: &[f32],
    i3: &[f32],
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    k: usize,
) -> [[f32; 4]; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON baseline on aarch64; all slices have >= k elements.
        unsafe { micro4x4_neon(i0, i1, i2, i3, w0, w1, w2, w3, k) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let ins = [i0, i1, i2, i3];
        let ws = [w0, w1, w2, w3];
        let mut o = [[0.0f32; 4]; 4];
        for (ri, &iv) in ins.iter().enumerate() {
            for (ci, &wv) in ws.iter().enumerate() {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += iv[t] * wv[t];
                }
                o[ri][ci] = acc;
            }
        }
        o
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro4x4_neon(
    i0: &[f32],
    i1: &[f32],
    i2: &[f32],
    i3: &[f32],
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    k: usize,
) -> [[f32; 4]; 4] {
    use std::arch::aarch64::*;
    // 16 accumulators: acc[ri][ci].
    let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
    let ip = [i0.as_ptr(), i1.as_ptr(), i2.as_ptr(), i3.as_ptr()];
    let wp = [w0.as_ptr(), w1.as_ptr(), w2.as_ptr(), w3.as_ptr()];
    let k4 = k - (k % 4);
    let mut t = 0;
    while t < k4 {
        let iv = [
            vld1q_f32(ip[0].add(t)),
            vld1q_f32(ip[1].add(t)),
            vld1q_f32(ip[2].add(t)),
            vld1q_f32(ip[3].add(t)),
        ];
        let wv = [
            vld1q_f32(wp[0].add(t)),
            vld1q_f32(wp[1].add(t)),
            vld1q_f32(wp[2].add(t)),
            vld1q_f32(wp[3].add(t)),
        ];
        for ri in 0..4 {
            acc[ri][0] = vfmaq_f32(acc[ri][0], iv[ri], wv[0]);
            acc[ri][1] = vfmaq_f32(acc[ri][1], iv[ri], wv[1]);
            acc[ri][2] = vfmaq_f32(acc[ri][2], iv[ri], wv[2]);
            acc[ri][3] = vfmaq_f32(acc[ri][3], iv[ri], wv[3]);
        }
        t += 4;
    }
    let mut o = [[0.0f32; 4]; 4];
    for ri in 0..4 {
        o[ri][0] = vaddvq_f32(acc[ri][0]);
        o[ri][1] = vaddvq_f32(acc[ri][1]);
        o[ri][2] = vaddvq_f32(acc[ri][2]);
        o[ri][3] = vaddvq_f32(acc[ri][3]);
    }
    // tail
    while t < k {
        for ri in 0..4 {
            let iv = *[i0, i1, i2, i3][ri].get_unchecked(t);
            o[ri][0] += iv * *w0.get_unchecked(t);
            o[ri][1] += iv * *w1.get_unchecked(t);
            o[ri][2] += iv * *w2.get_unchecked(t);
            o[ri][3] += iv * *w3.get_unchecked(t);
        }
        t += 1;
    }
    o
}

/// Four simultaneous dot products `Σ in[i] * wj[i]` over `k` elements.
#[inline]
fn dot4(
    in_row: &[f32],
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    k: usize,
) -> (f32, f32, f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; all slices have >= k elements.
        unsafe { dot4_neon(in_row, w0, w1, w2, w3, k) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot4_scalar(in_row, w0, w1, w2, w3, k)
    }
}

/// Single SIMD dot product `Σ a[i] * b[i]` over `k` elements. Public so the
/// attention inner loops can reuse the NEON path.
#[inline]
pub fn dot(a: &[f32], b: &[f32], k: usize) -> f32 {
    dot1(a, b, k)
}

/// Single dot product over `k` elements.
#[inline]
fn dot1(in_row: &[f32], w: &[f32], k: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; slices have >= k elements.
        unsafe { dot1_neon(in_row, w, k) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: gated on runtime AVX2+FMA detection; slices have >= k elements.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot1_avx2(in_row, w, k) };
        }
        dot1_scalar(in_row, w, k)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot1_scalar(in_row, w, k)
    }
}

/// Scalar reference dot product (non-aarch64 fallback / portable path).
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot1_scalar(in_row: &[f32], w: &[f32], k: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..k {
        acc += in_row[i] * w[i];
    }
    acc
}

/// AVX2+FMA dot product over `k` elements (two 8-wide accumulators + tail).
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available (checked in [`dot1`]); `in_row`
/// and `w` must each have at least `k` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot1_avx2(in_row: &[f32], w: &[f32], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let ip = in_row.as_ptr();
    let wp = w.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let k16 = k - (k % 16);
    let mut i = 0;
    while i < k16 {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ip.add(i)), _mm256_loadu_ps(wp.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ip.add(i + 8)),
            _mm256_loadu_ps(wp.add(i + 8)),
            acc1,
        );
        i += 16;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(acc0, acc1));
    let mut acc = lanes.iter().sum::<f32>();
    while i < k {
        acc += *in_row.get_unchecked(i) * *w.get_unchecked(i);
        i += 1;
    }
    acc
}

/// `o[i] += w * v[i]` for `i in 0..n` (the attention value-accumulation step),
/// SIMD-accelerated on x86_64 with a scalar fallback elsewhere.
#[inline]
pub fn axpy(o: &mut [f32], w: f32, v: &[f32], n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: gated on runtime AVX2+FMA detection; `o`/`v` have >= n elements.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { axpy_avx2(o, w, v, n) };
            return;
        }
    }
    for i in 0..n {
        o[i] += w * v[i];
    }
}

/// AVX2+FMA fused `o[i] += w * v[i]`.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available (checked in [`axpy`]); `o` and
/// `v` must each have at least `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(o: &mut [f32], w: f32, v: &[f32], n: usize) {
    use std::arch::x86_64::*;
    let wv = _mm256_set1_ps(w);
    let op = o.as_mut_ptr();
    let vp = v.as_ptr();
    let n8 = n - (n % 8);
    let mut i = 0;
    while i < n8 {
        let acc = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vp.add(i)), _mm256_loadu_ps(op.add(i)));
        _mm256_storeu_ps(op.add(i), acc);
        i += 8;
    }
    while i < n {
        *o.get_unchecked_mut(i) += w * *v.get_unchecked(i);
        i += 1;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot4_scalar(
    in_row: &[f32],
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    k: usize,
) -> (f32, f32, f32, f32) {
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    for i in 0..k {
        let x = in_row[i];
        a0 += x * w0[i];
        a1 += x * w1[i];
        a2 += x * w2[i];
        a3 += x * w3[i];
    }
    (a0, a1, a2, a3)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot4_neon(
    in_row: &[f32],
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    k: usize,
) -> (f32, f32, f32, f32) {
    use std::arch::aarch64::*;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let ip = in_row.as_ptr();
    let p0 = w0.as_ptr();
    let p1 = w1.as_ptr();
    let p2 = w2.as_ptr();
    let p3 = w3.as_ptr();
    let k16 = k - (k % 16);
    let mut i = 0;
    // 16-wide unroll (4 NEON vectors) per accumulator.
    while i < k16 {
        let x0 = vld1q_f32(ip.add(i));
        let x1 = vld1q_f32(ip.add(i + 4));
        let x2 = vld1q_f32(ip.add(i + 8));
        let x3 = vld1q_f32(ip.add(i + 12));
        acc0 = vfmaq_f32(acc0, x0, vld1q_f32(p0.add(i)));
        acc0 = vfmaq_f32(acc0, x1, vld1q_f32(p0.add(i + 4)));
        acc0 = vfmaq_f32(acc0, x2, vld1q_f32(p0.add(i + 8)));
        acc0 = vfmaq_f32(acc0, x3, vld1q_f32(p0.add(i + 12)));
        acc1 = vfmaq_f32(acc1, x0, vld1q_f32(p1.add(i)));
        acc1 = vfmaq_f32(acc1, x1, vld1q_f32(p1.add(i + 4)));
        acc1 = vfmaq_f32(acc1, x2, vld1q_f32(p1.add(i + 8)));
        acc1 = vfmaq_f32(acc1, x3, vld1q_f32(p1.add(i + 12)));
        acc2 = vfmaq_f32(acc2, x0, vld1q_f32(p2.add(i)));
        acc2 = vfmaq_f32(acc2, x1, vld1q_f32(p2.add(i + 4)));
        acc2 = vfmaq_f32(acc2, x2, vld1q_f32(p2.add(i + 8)));
        acc2 = vfmaq_f32(acc2, x3, vld1q_f32(p2.add(i + 12)));
        acc3 = vfmaq_f32(acc3, x0, vld1q_f32(p3.add(i)));
        acc3 = vfmaq_f32(acc3, x1, vld1q_f32(p3.add(i + 4)));
        acc3 = vfmaq_f32(acc3, x2, vld1q_f32(p3.add(i + 8)));
        acc3 = vfmaq_f32(acc3, x3, vld1q_f32(p3.add(i + 12)));
        i += 16;
    }
    let mut a0 = vaddvq_f32(acc0);
    let mut a1 = vaddvq_f32(acc1);
    let mut a2 = vaddvq_f32(acc2);
    let mut a3 = vaddvq_f32(acc3);
    while i < k {
        let x = *in_row.get_unchecked(i);
        a0 += x * *w0.get_unchecked(i);
        a1 += x * *w1.get_unchecked(i);
        a2 += x * *w2.get_unchecked(i);
        a3 += x * *w3.get_unchecked(i);
        i += 1;
    }
    (a0, a1, a2, a3)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot1_neon(in_row: &[f32], w: &[f32], k: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_f32(0.0);
    let ip = in_row.as_ptr();
    let wp = w.as_ptr();
    let k4 = k - (k % 4);
    let mut i = 0;
    while i < k4 {
        acc = vfmaq_f32(acc, vld1q_f32(ip.add(i)), vld1q_f32(wp.add(i)));
        i += 4;
    }
    let mut a = vaddvq_f32(acc);
    while i < k {
        a += *in_row.get_unchecked(i) * *w.get_unchecked(i);
        i += 1;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_abt_matches_naive() {
        let (m, n, k) = (5usize, 7usize, 13usize);
        let input: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let weight: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).cos()).collect();
        let mut out = vec![0.0f32; m * n];
        gemm_abt(&input, &weight, &mut out, m, n, k);
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += input[r * k + i] * weight[c * k + i];
                }
                assert!(
                    (out[r * n + c] - acc).abs() < 1e-3,
                    "({r},{c}) {} vs {acc}",
                    out[r * n + c]
                );
            }
        }
    }
}
