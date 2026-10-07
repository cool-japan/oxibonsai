//! NEON core tests for `simd_float_ops` (aarch64 only): `exp_neon_f32x4`
//! and `silu_core_neon_f32x4` pinned bit-for-bit, lane by lane, to the
//! scalar model in `simd_float_ops_cephes_model.rs`, over the same sweeps
//! `simd_float_ops_avx2_tests.rs` pins their x86_64 twins over — so "the two
//! tiers compute the same bits" is a tested statement on each architecture,
//! not a cross-ISA claim — and `softmax_neon`'s `exp` pass held to that core
//! in its vector body and padded tail alike.
//!
//! Re-attached with `#[cfg(all(test, target_arch = "aarch64"))] #[path]`, so
//! these are a child module of `simd_float_ops` and reach its private items
//! via `use super::*`. NEON is mandatory on aarch64: no runtime feature check.

use super::cephes_model::{
    exp_cephes_model, exp_model_sweep, silu_cephes_model, silu_model_sweep, softmax_probe_row,
    SOFTMAX_PROBE_MAXIMA,
};
use super::*;
use std::arch::aarch64::{float32x4_t, vld1q_f32, vst1q_f32};

/// Maps `xs` through a 4-lane core one zero-padded vector at a time (the pad
/// lanes of a short last chunk are computed and dropped) — the NEON twin of
/// the AVX2 tests' `map_lanes`.
///
/// # Safety
///
/// `lane_op` must be sound to call on any vector. Every core here is: NEON
/// is mandatory on aarch64 and they touch no memory. The loads and stores go
/// through fixed-size 4-element stack arrays only.
unsafe fn map_lanes(xs: &[f32], lane_op: unsafe fn(float32x4_t) -> float32x4_t) -> Vec<f32> {
    let mut out = Vec::with_capacity(xs.len());
    for chunk in xs.chunks(4) {
        let mut lanes = [0.0f32; 4];
        lanes[..chunk.len()].copy_from_slice(chunk);
        let mut result = [0.0f32; 4];
        vst1q_f32(result.as_mut_ptr(), lane_op(vld1q_f32(lanes.as_ptr())));
        out.extend_from_slice(&result[..chunk.len()]);
    }
    out
}

#[test]
fn exp_neon_f32x4_is_lane_wise_the_scalar_cephes_model() {
    let xs = exp_model_sweep();
    // SAFETY: NEON is mandatory on aarch64; `exp_neon_f32x4` touches no memory.
    let got = unsafe { map_lanes(&xs, exp_neon_f32x4) };
    for (&x, &y) in xs.iter().zip(&got) {
        let want = exp_cephes_model(x);
        assert_eq!(
            y.to_bits(),
            want.to_bits(),
            "exp_neon_f32x4({x:e}) = {y:e} ({:#010x}), but the scalar transcription of its \
             operation sequence gives {want:e} ({:#010x})",
            y.to_bits(),
            want.to_bits()
        );
    }

    // SAFETY: as above.
    let nan = unsafe { map_lanes(&[f32::NAN; 4], exp_neon_f32x4) };
    assert!(
        nan.iter().all(|v| v.is_nan()) && exp_cephes_model(f32::NAN).is_nan(),
        "NaN must stay NaN on both sides, got {nan:?}"
    );
}

#[test]
fn silu_core_neon_f32x4_is_lane_wise_the_scalar_cephes_model() {
    let xs = silu_model_sweep();
    // SAFETY: NEON is mandatory on aarch64; `silu_core_neon_f32x4` touches no
    // memory.
    let got = unsafe { map_lanes(&xs, silu_core_neon_f32x4) };
    for (&x, &y) in xs.iter().zip(&got) {
        let want = silu_cephes_model(x);
        assert_eq!(
            y.to_bits(),
            want.to_bits(),
            "silu_core_neon_f32x4({x:e}) = {y:e} ({:#010x}), but the scalar model gives \
             {want:e} ({:#010x})",
            y.to_bits(),
            want.to_bits()
        );
    }

    // SAFETY: as above.
    let nan = unsafe { map_lanes(&[f32::NAN; 4], silu_core_neon_f32x4) };
    assert!(
        nan.iter().all(|v| v.is_nan()) && silu_cephes_model(f32::NAN).is_nan(),
        "NaN must stay NaN on both sides, got {nan:?}"
    );
}

/// Bit-exact scalar model of `softmax_neon`: the row max; every element's
/// `exp_neon_f32x4(v - max)` (4-lane chunks, the pad lanes of a short last
/// chunk computed and dropped); the denominator associated exactly as the
/// kernel does — each of the 4 body lanes accumulated across the full
/// chunks, reduced by the two `vpadd_f32` as `(l0 + l1) + (l2 + l3)`, then
/// the tail's `exp`s summed among themselves and that sub-sum added once;
/// and every `exp` times `1 / sum`.
fn softmax_neon_model(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = row.iter().map(|&v| v - max).collect();
    // SAFETY: NEON is mandatory on aarch64; `exp_neon_f32x4` touches no memory.
    let exps = unsafe { map_lanes(&shifted, exp_neon_f32x4) };
    // The kernel's split: whole 4-lane vectors, then the 0-3 element tail.
    let (body, tail) = exps.as_chunks::<4>();
    let mut sum = 0.0f32;
    if !body.is_empty() {
        let mut lanes = [0.0f32; 4];
        for chunk in body {
            for (lane, &e) in lanes.iter_mut().zip(chunk) {
                *lane += e;
            }
        }
        sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
    if !tail.is_empty() {
        sum += tail.iter().sum::<f32>();
    }
    let inv_sum = 1.0 / sum;
    exps.iter().map(|&e| e * inv_sum).collect()
}

/// The NEON twin of `avx2_core_tests::softmax_body_and_tail_use_the_avx2_exp_core_bitwise`:
/// a logit `87.75..102.5` below the row max must come out as exactly `+0.0`
/// (the polynomial's flush; libm's `exp` gives a nonzero subnormal there) in
/// the 4-lane body and the padded tail alike, and every row must match
/// [`softmax_neon_model`] bit for bit. Lengths 1..=24: zero to six full
/// bodies with every tail size.
#[test]
fn softmax_body_and_tail_use_the_neon_exp_core_bitwise() {
    for n in 1..=24usize {
        for max in SOFTMAX_PROBE_MAXIMA {
            let (row, flushes) = softmax_probe_row(n, max);
            let mut got = row.clone();
            softmax_simd(&mut got);
            let want = softmax_neon_model(&row);
            for (i, ((&g, &w), &flush)) in got.iter().zip(&want).zip(&flushes).enumerate() {
                if flush {
                    assert_eq!(
                        g.to_bits(),
                        0.0f32.to_bits(),
                        "softmax n={n} max={max} pos={i}: logit {} is {} below the max, \
                         where exp_neon_f32x4 flushes to +0.0 (f32::exp gives {:e}), but the \
                         output is {g:e}",
                        row[i],
                        max - row[i],
                        (row[i] - max).exp()
                    );
                }
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "softmax n={n} max={max} pos={i}: {g:e} ({:#010x}) vs the bit-exact \
                     exp_neon_f32x4 model's {w:e} ({:#010x})",
                    g.to_bits(),
                    w.to_bits()
                );
            }
        }
    }
}
