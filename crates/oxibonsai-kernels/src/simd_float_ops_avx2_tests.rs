//! AVX2 core tests for `simd_float_ops` (the K-M3 port to x86_64):
//! `exp_avx2_f32x8` / `silu_core_avx2_f32x8` accuracy and clamp-edge
//! behaviour, their bit-for-bit agreement with the scalar model of the
//! shared Cephes operation sequence (`simd_float_ops_cephes_model.rs`, which
//! `simd_float_ops_neon_tests.rs` pins the NEON twins to over the same
//! sweeps), and the routing of every element of `silu_avx2` /
//! `swiglu_avx2` / `softmax_avx2` — vector body and padded tail alike —
//! through them.
//!
//! Split out of `simd_float_ops.rs` to keep it inside the 2000-line policy
//! ceiling and re-attached with `#[cfg(all(test, target_arch = "x86_64"))]
//! #[path]`, so these remain a child module of `simd_float_ops` and reach
//! its private items via `use super::*`.
//!
//! Every test returns early (with an `eprintln!`) on a CPU without
//! AVX2+FMA, where `simd_float_ops` never dispatches to these cores.

use super::cephes_model::{
    dense_exp_sweep, dense_silu_sweep, every_f32_in, exp_cephes_model, exp_correctly_rounded,
    exp_model_sweep, from_ordered, ordered, silu_cephes_model, silu_f64, silu_model_sweep,
    softmax_probe_row, ulp_distance, EXP_HI, FLUSH_START, MAX_SILU_ULP_VS_F64,
    MAX_ULP_VS_CORRECTLY_ROUNDED, SOFTMAX_PROBE_MAXIMA,
};
use super::*;
use std::arch::x86_64::{__m256, _mm256_loadu_ps, _mm256_storeu_ps};

/// `exp_avx2_f32x8` vs `f32::exp`. Measured: 1 ulp against glibc's `expf`
/// over every `f32` of `[-87, 87]` (the `#[ignore]`d exhaustive test below),
/// which is the bound pinned on Linux/glibc. Another platform's `expf` may
/// itself sit 1 ulp off the correctly rounded value in the opposite
/// direction, so the bound is 2 there; the platform-independent accuracy
/// claim is [`MAX_ULP_VS_CORRECTLY_ROUNDED`].
const MAX_ULP_VS_F32_EXP: u64 = if cfg!(all(target_os = "linux", target_env = "gnu")) {
    1
} else {
    2
};

/// Below the clamp `silu_core_avx2_f32x8(x)` is `x / (1 + exp(EXP_HI))`,
/// i.e. `x * 4.156_03e-39` rounded once, and the exact value lies between
/// it and zero, so `|error| <= |x| * 4.157e-39`.
const DEEP_TAIL_ABS_ERROR_PER_UNIT_X: f64 = 4.157e-39;

/// Relative-error sanity bound of `silu_core_avx2_f32x8` against the
/// libm-based `silu_scalar_elem` (measured `3.5e-7` on glibc; held loosely
/// because another platform's `expf` moves the reference itself).
const MAX_SILU_REL_ERROR_VS_LIBM: f32 = 1e-5;

/// Lower bound on the fraction of `[-87, 87]` that `exp_avx2_f32x8` rounds
/// exactly correctly (measured `0.965379` by the exhaustive test).
const MIN_CORRECTLY_ROUNDED_FRACTION: f64 = 0.965;

fn avx2_fma_available(test: &str) -> bool {
    let available = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    if !available {
        eprintln!("{test}: skipped, this CPU lacks AVX2+FMA (the AVX2 cores are never dispatched)");
    }
    available
}

/// Maps `xs` through an 8-lane core one zero-padded vector at a time (the
/// pad lanes of a short last chunk are computed and dropped).
///
/// # Safety
///
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
unsafe fn map_lanes(xs: &[f32], lane_op: unsafe fn(__m256) -> __m256) -> Vec<f32> {
    let mut out = Vec::with_capacity(xs.len());
    for chunk in xs.chunks(8) {
        let mut lanes = [0.0f32; 8];
        lanes[..chunk.len()].copy_from_slice(chunk);
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(
            result.as_mut_ptr(),
            lane_op(_mm256_loadu_ps(lanes.as_ptr())),
        );
        out.extend_from_slice(&result[..chunk.len()]);
    }
    out
}

// ── exp_avx2_f32x8 ──────────────────────────────────────────────────

#[test]
fn exp_avx2_f32x8_ulp_error_vs_f32_exp_over_a_dense_sweep() {
    if !avx2_fma_available("exp_avx2_f32x8_ulp_error_vs_f32_exp_over_a_dense_sweep") {
        return;
    }
    let xs = dense_exp_sweep();
    // SAFETY: AVX2+FMA support was just confirmed.
    let got = unsafe { map_lanes(&xs, exp_avx2_f32x8) };

    let (mut worst_libm, mut worst_libm_at) = (0u64, 0.0f32);
    let (mut worst_rounded, mut worst_rounded_at) = (0u64, 0.0f32);
    for (&x, &y) in xs.iter().zip(&got) {
        let vs_libm = ulp_distance(y, x.exp());
        if vs_libm > worst_libm {
            (worst_libm, worst_libm_at) = (vs_libm, x);
        }
        let vs_rounded = ulp_distance(y, exp_correctly_rounded(x));
        if vs_rounded > worst_rounded {
            (worst_rounded, worst_rounded_at) = (vs_rounded, x);
        }
    }
    assert!(
        worst_libm <= MAX_ULP_VS_F32_EXP,
        "exp_avx2_f32x8 is {worst_libm} ulp from f32::exp at x = {worst_libm_at:e} \
         (bound {MAX_ULP_VS_F32_EXP})"
    );
    assert!(
        worst_rounded <= MAX_ULP_VS_CORRECTLY_ROUNDED,
        "exp_avx2_f32x8 is {worst_rounded} ulp from the correctly rounded e^x at \
         x = {worst_rounded_at:e} (bound {MAX_ULP_VS_CORRECTLY_ROUNDED})"
    );
}

#[test]
fn exp_avx2_f32x8_saturates_flushes_and_propagates_nan_at_the_clamp_edges() {
    if !avx2_fma_available("exp_avx2_f32x8_saturates_flushes_and_propagates_nan_at_the_clamp_edges")
    {
        return;
    }
    // SAFETY: AVX2+FMA support was just confirmed (every call below).
    let exp = |xs: &[f32]| unsafe { map_lanes(xs, exp_avx2_f32x8) };

    // Upper edge: exp(EXP_HI) itself is accurate, and every larger input,
    // +inf included, saturates to exactly that finite value.
    let at_hi = exp(&[EXP_HI])[0];
    assert!(
        at_hi.is_finite() && ulp_distance(at_hi, EXP_HI.exp()) <= MAX_ULP_VS_F32_EXP,
        "exp(EXP_HI) = {at_hi:e}, f32::exp gives {:e}",
        EXP_HI.exp()
    );
    let beyond_hi = [
        f32::from_bits(EXP_HI.to_bits() + 1),
        88.5,
        100.0,
        1.0e30,
        f32::MAX,
        f32::INFINITY,
    ];
    for (&x, &y) in beyond_hi.iter().zip(&exp(&beyond_hi)) {
        assert_eq!(
            y.to_bits(),
            at_hi.to_bits(),
            "exp({x:e}) = {y:e}, expected to saturate to exp(EXP_HI) = {at_hi:e}"
        );
    }

    // Lower edge, part 1: on (FLUSH_START, -87] every result is still a
    // properly scaled (sub)normal within the dense sweep's ulp bounds.
    let above_flush: Vec<f32> =
        every_f32_in(f32::from_bits(FLUSH_START.to_bits() - 1), -87.0).collect();
    for (&x, &y) in above_flush.iter().zip(&exp(&above_flush)) {
        assert!(
            y > 0.0
                && ulp_distance(y, x.exp()) <= MAX_ULP_VS_F32_EXP
                && ulp_distance(y, exp_correctly_rounded(x)) <= MAX_ULP_VS_CORRECTLY_ROUNDED,
            "exp({x:e}) = {y:e}, f32::exp gives {:e}",
            x.exp()
        );
    }

    // Lower edge, part 2: from FLUSH_START down through -EXP_HI and beyond
    // (-inf included) the result is exactly +0.0. `f32::exp` is a subnormal
    // (or zero) there, so the absolute error stays below f32::MIN_POSITIVE.
    let mut flushed: Vec<f32> = every_f32_in(-EXP_HI, FLUSH_START).collect();
    flushed.extend([-88.5, -100.0, -1.0e30, f32::MIN, f32::NEG_INFINITY]);
    for (&x, &y) in flushed.iter().zip(&exp(&flushed)) {
        assert_eq!(
            y.to_bits(),
            0.0f32.to_bits(),
            "exp({x:e}) = {y:e}, expected the +0.0 flush"
        );
        assert!(
            x.exp() < f32::MIN_POSITIVE,
            "f32::exp({x:e}) = {:e} is not below f32::MIN_POSITIVE",
            x.exp()
        );
    }

    // NaN propagates (that is what the clamp's operand order is for), and
    // exp(±0) is exactly 1.
    let specials = exp(&[f32::NAN, -f32::NAN, 0.0, -0.0]);
    assert!(
        specials[0].is_nan() && specials[1].is_nan(),
        "NaN must propagate through the clamp, got {specials:?}"
    );
    assert_eq!(specials[2].to_bits(), 1.0f32.to_bits(), "exp(+0.0)");
    assert_eq!(specials[3].to_bits(), 1.0f32.to_bits(), "exp(-0.0)");
}

#[test]
fn exp_avx2_f32x8_is_lane_wise_the_scalar_cephes_model() {
    if !avx2_fma_available("exp_avx2_f32x8_is_lane_wise_the_scalar_cephes_model") {
        return;
    }
    let xs = exp_model_sweep();
    // SAFETY: AVX2+FMA support was just confirmed.
    let got = unsafe { map_lanes(&xs, exp_avx2_f32x8) };
    for (&x, &y) in xs.iter().zip(&got) {
        let want = exp_cephes_model(x);
        assert_eq!(
            y.to_bits(),
            want.to_bits(),
            "exp_avx2_f32x8({x:e}) = {y:e} ({:#010x}), but the scalar transcription of \
             exp_neon_f32x4 gives {want:e} ({:#010x})",
            y.to_bits(),
            want.to_bits()
        );
    }

    // SAFETY: as above.
    let nan = unsafe { map_lanes(&[f32::NAN; 8], exp_avx2_f32x8) };
    assert!(
        nan.iter().all(|v| v.is_nan()) && exp_cephes_model(f32::NAN).is_nan(),
        "NaN must stay NaN on both sides, got {nan:?}"
    );
}

/// Running tally of the exhaustive `exp` sweep below.
#[derive(Clone, Copy)]
struct ExpAccuracyTally {
    values: u64,
    correctly_rounded: u64,
    /// `(ulp distance, x)` of the worst value against the correctly rounded `e^x`.
    worst_vs_rounded: (u64, f32),
    /// `(ulp distance, x)` of the worst value against `f32::exp`.
    worst_vs_libm: (u64, f32),
}

impl ExpAccuracyTally {
    const EMPTY: Self = Self {
        values: 0,
        correctly_rounded: 0,
        worst_vs_rounded: (0, 0.0),
        worst_vs_libm: (0, 0.0),
    };

    fn record(&mut self, x: f32, y: f32) {
        self.values += 1;
        let vs_rounded = ulp_distance(y, exp_correctly_rounded(x));
        if vs_rounded == 0 {
            self.correctly_rounded += 1;
        }
        if vs_rounded > self.worst_vs_rounded.0 {
            self.worst_vs_rounded = (vs_rounded, x);
        }
        let vs_libm = ulp_distance(y, x.exp());
        if vs_libm > self.worst_vs_libm.0 {
            self.worst_vs_libm = (vs_libm, x);
        }
    }

    fn merge(self, other: Self) -> Self {
        let worse = |a: (u64, f32), b: (u64, f32)| if b.0 > a.0 { b } else { a };
        Self {
            values: self.values + other.values,
            correctly_rounded: self.correctly_rounded + other.correctly_rounded,
            worst_vs_rounded: worse(self.worst_vs_rounded, other.worst_vs_rounded),
            worst_vs_libm: worse(self.worst_vs_libm, other.worst_vs_libm),
        }
    }
}

/// Tallies `exp_avx2_f32x8` over every `f32` whose [`ordered`] position lies
/// in `[first, last]` (none if `first > last`), a million values at a time.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA.
unsafe fn tally_exp_accuracy(first: i64, last: i64) -> ExpAccuracyTally {
    const BLOCK: i64 = 1 << 20;
    let mut tally = ExpAccuracyTally::EMPTY;
    let mut start = first;
    while start <= last {
        let end = last.min(start + BLOCK - 1);
        let xs: Vec<f32> = (start..=end).map(from_ordered).collect();
        let got = map_lanes(&xs, exp_avx2_f32x8);
        for (&x, &y) in xs.iter().zip(&got) {
            tally.record(x, y);
        }
        start = end + 1;
    }
    tally
}

/// The exhaustive form of `exp_avx2_f32x8`'s documented accuracy, so the
/// claim is reproducible in-repo: every one of the 2,237,399,041 finite
/// `f32` in `[-87, 87]` is within [`MAX_ULP_VS_CORRECTLY_ROUNDED`] (1) ulp of
/// `e^x` evaluated in `f64` and rounded to `f32`, and within
/// [`MAX_ULP_VS_F32_EXP`] of `f32::exp`; at least 96.5% are exactly
/// correctly rounded (measured: 96.5379%). `[87, EXP_HI]` is swept
/// exhaustively by the dense sweep above, every run. Split across
/// `available_parallelism()` threads: 6.9 s with 5 worker threads in the
/// optimized test profile. Run with:
/// `cargo nextest run -p oxibonsai-kernels --run-ignored only --no-capture
/// -E 'test(every_f32_in_minus_87_to_87)'`.
#[test]
#[ignore = "exhaustive over 2,237,399,041 inputs; run with `--run-ignored only`"]
fn exp_avx2_f32x8_is_within_1_ulp_of_the_correctly_rounded_exp_for_every_f32_in_minus_87_to_87() {
    if !avx2_fma_available(
        "exp_avx2_f32x8_is_within_1_ulp_of_the_correctly_rounded_exp_for_every_f32_in_minus_87_to_87",
    ) {
        return;
    }
    let (lo, hi) = (ordered(-87.0), ordered(87.0));
    let workers = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let span = u64::try_from(hi - lo + 1).expect("-87 <= 87");
    let per_worker = i64::try_from(span.div_ceil(workers as u64)).expect("fits an i64");
    let started = std::time::Instant::now();
    let tally = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers as i64)
            .map(|worker| {
                let first = lo + worker * per_worker;
                let last = hi.min(first + per_worker - 1);
                // SAFETY: AVX2+FMA support was confirmed above, on this CPU.
                scope.spawn(move || unsafe { tally_exp_accuracy(first, last) })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("an exhaustive-sweep worker panicked"))
            .fold(ExpAccuracyTally::EMPTY, ExpAccuracyTally::merge)
    });
    let fraction_correctly_rounded = tally.correctly_rounded as f64 / tally.values as f64;
    eprintln!(
        "exp_avx2_f32x8 over every f32 in [-87, 87]: {} values in {:.1?} on {workers} threads; \
         {:.4}% correctly rounded; worst {} ulp vs the correctly rounded e^x (x = {:e}); \
         worst {} ulp vs f32::exp (x = {:e})",
        tally.values,
        started.elapsed(),
        fraction_correctly_rounded * 100.0,
        tally.worst_vs_rounded.0,
        tally.worst_vs_rounded.1,
        tally.worst_vs_libm.0,
        tally.worst_vs_libm.1,
    );
    assert_eq!(tally.values, 2_237_399_041, "every finite f32 in [-87, 87]");
    assert!(
        tally.worst_vs_rounded.0 <= MAX_ULP_VS_CORRECTLY_ROUNDED,
        "exp_avx2_f32x8 is {} ulp from the correctly rounded e^x at x = {:e} (bound \
         {MAX_ULP_VS_CORRECTLY_ROUNDED})",
        tally.worst_vs_rounded.0,
        tally.worst_vs_rounded.1
    );
    assert!(
        tally.worst_vs_libm.0 <= MAX_ULP_VS_F32_EXP,
        "exp_avx2_f32x8 is {} ulp from f32::exp at x = {:e} (bound {MAX_ULP_VS_F32_EXP})",
        tally.worst_vs_libm.0,
        tally.worst_vs_libm.1
    );
    assert!(
        fraction_correctly_rounded >= MIN_CORRECTLY_ROUNDED_FRACTION,
        "only {:.4}% of [-87, 87] correctly rounded (bound {}%)",
        fraction_correctly_rounded * 100.0,
        MIN_CORRECTLY_ROUNDED_FRACTION * 100.0
    );
}

// ── silu_core_avx2_f32x8 ────────────────────────────────────────────

/// The documented accuracy, platform-independently: over 6,000,001 evenly
/// spaced points of `[-30, 30]`, at most [`MAX_SILU_ULP_VS_F64`] (3) ulp from
/// the exact SiLU evaluated in `f64` and rounded once — plus a loose
/// relative sanity bound against the libm-based `silu_scalar_elem`.
#[test]
fn silu_core_avx2_f32x8_is_within_3_ulp_of_the_f64_silu_over_a_dense_sweep() {
    if !avx2_fma_available(
        "silu_core_avx2_f32x8_is_within_3_ulp_of_the_f64_silu_over_a_dense_sweep",
    ) {
        return;
    }
    let xs = dense_silu_sweep();
    // SAFETY: AVX2+FMA support was just confirmed.
    let got = unsafe { map_lanes(&xs, silu_core_avx2_f32x8) };

    let (mut worst_ulp, mut worst_ulp_at) = (0u64, 0.0f32);
    let (mut worst_relative, mut worst_relative_at) = (0.0f32, 0.0f32);
    for (&x, &y) in xs.iter().zip(&got) {
        let distance = ulp_distance(y, silu_f64(x));
        if distance > worst_ulp {
            (worst_ulp, worst_ulp_at) = (distance, x);
        }
        let want = silu_scalar_elem(x);
        // Relative, with an absolute floor only at the exact zero.
        let relative = (y - want).abs() / want.abs().max(f32::MIN_POSITIVE);
        if relative > worst_relative {
            (worst_relative, worst_relative_at) = (relative, x);
        }
    }
    assert!(
        worst_ulp <= MAX_SILU_ULP_VS_F64,
        "silu_core_avx2_f32x8 is {worst_ulp} ulp from the f64 SiLU at x = {worst_ulp_at:e} \
         (bound {MAX_SILU_ULP_VS_F64})"
    );
    assert!(
        worst_relative <= MAX_SILU_REL_ERROR_VS_LIBM,
        "silu_core_avx2_f32x8 is {worst_relative:e} (relative) from silu_scalar_elem at \
         x = {worst_relative_at} (bound {MAX_SILU_REL_ERROR_VS_LIBM:e})"
    );
}

#[test]
fn silu_core_avx2_f32x8_is_lane_wise_the_scalar_cephes_model() {
    if !avx2_fma_available("silu_core_avx2_f32x8_is_lane_wise_the_scalar_cephes_model") {
        return;
    }
    let xs = silu_model_sweep();
    // SAFETY: AVX2+FMA support was just confirmed.
    let got = unsafe { map_lanes(&xs, silu_core_avx2_f32x8) };
    for (&x, &y) in xs.iter().zip(&got) {
        let want = silu_cephes_model(x);
        assert_eq!(
            y.to_bits(),
            want.to_bits(),
            "silu_core_avx2_f32x8({x:e}) = {y:e} ({:#010x}), but the scalar model gives \
             {want:e} ({:#010x})",
            y.to_bits(),
            want.to_bits()
        );
    }

    // SAFETY: as above.
    let nan = unsafe { map_lanes(&[f32::NAN; 8], silu_core_avx2_f32x8) };
    assert!(
        nan.iter().all(|v| v.is_nan()) && silu_cephes_model(f32::NAN).is_nan(),
        "NaN must stay NaN on both sides, got {nan:?}"
    );
}

#[test]
fn silu_core_avx2_f32x8_is_absolute_error_only_below_the_exp_clamp() {
    if !avx2_fma_available("silu_core_avx2_f32x8_is_absolute_error_only_below_the_exp_clamp") {
        return;
    }
    // SAFETY: AVX2+FMA support was just confirmed (both calls below).
    let exp_at_hi = unsafe { map_lanes(&[EXP_HI], exp_avx2_f32x8) }[0];

    // Every x here is below -EXP_HI: the first f32 past the clamp, a dense
    // band down to about -10,088, then decades out to f32::MIN and -inf.
    let mut xs = vec![f32::from_bits((-EXP_HI).to_bits() + 1)];
    xs.extend((1..=200_000u32).map(|i| -EXP_HI - i as f32 * 0.05));
    xs.extend([
        -1.0e5,
        -1.0e6,
        -1.0e10,
        -1.0e20,
        -1.0e30,
        f32::MIN,
        f32::NEG_INFINITY,
    ]);
    // SAFETY: as above.
    let got = unsafe { map_lanes(&xs, silu_core_avx2_f32x8) };

    for (&x, &y) in xs.iter().zip(&got) {
        // Exactly the saturated formula: the clamp and nothing else. (So the
        // relative error is unbounded: at x = -100 this is -4.156e-37 where
        // the exact value is -3.72e-42; and -inf maps to -inf.)
        let saturated = x / (1.0 + exp_at_hi);
        assert_eq!(
            y.to_bits(),
            saturated.to_bits(),
            "silu_core_avx2_f32x8({x:e}) = {y:e}, expected x / (1 + exp(EXP_HI)) = {saturated:e}"
        );
        if x.is_finite() {
            // ... but the absolute error is bounded by |x| * 4.157e-39,
            // against both the exact value and the scalar path.
            let x64 = f64::from(x);
            let exact = x64 / (1.0 + (-x64).exp());
            let bound = x64.abs() * DEEP_TAIL_ABS_ERROR_PER_UNIT_X;
            let scalar = f64::from(silu_scalar_elem(x));
            assert!(
                (f64::from(y) - exact).abs() <= bound && (f64::from(y) - scalar).abs() <= bound,
                "silu_core_avx2_f32x8({x:e}) = {y:e}: exact {exact:e}, silu_scalar_elem \
                 {scalar:e}, absolute bound {bound:e}"
            );
        }
    }
}

// ── body/tail routing through the cores ─────────────────────────────

/// `silu_simd` / `swiglu_simd` must run *every* element — vector body and
/// padded tail alike — through `silu_core_avx2_f32x8`, for lengths giving
/// zero to three full 8-lane bodies and every tail size.
#[test]
fn silu_and_swiglu_simd_route_every_element_through_the_avx2_core() {
    if !avx2_fma_available("silu_and_swiglu_simd_route_every_element_through_the_avx2_core") {
        return;
    }
    for len in 1..=24usize {
        let gate: Vec<f32> = (0..len)
            .map(|i| (i as f32 - 11.5) * 1.37 + len as f32 * 0.01)
            .collect();
        let up: Vec<f32> = (0..len).map(|i| 0.75 - i as f32 * 0.11).collect();
        // SAFETY: AVX2+FMA support was just confirmed.
        let want = unsafe { map_lanes(&gate, silu_core_avx2_f32x8) };

        let mut silu_out = vec![0.0f32; len];
        silu_simd(&gate, &mut silu_out).expect("matching lengths");
        let mut swiglu_out = vec![0.0f32; len];
        swiglu_simd(&gate, &up, &mut swiglu_out).expect("matching lengths");

        for (i, ((&s, &w), (&sw, &u))) in silu_out
            .iter()
            .zip(&want)
            .zip(swiglu_out.iter().zip(&up))
            .enumerate()
        {
            assert_eq!(
                s.to_bits(),
                w.to_bits(),
                "silu_simd len={len} pos={i}: {s:e} vs the AVX2 core's {w:e}"
            );
            assert_eq!(
                sw.to_bits(),
                (w * u).to_bits(),
                "swiglu_simd len={len} pos={i}: {sw:e} vs the AVX2 core's {:e}",
                w * u
            );
        }
    }
}

/// For `len < 8` all of `softmax_avx2` is the padded tail, so each output
/// must be exactly `exp_avx2_f32x8(v - max) * (1 / sum)`, the sum taken over
/// the real lanes only: a pad lane leaking into the sum (each pad computes
/// `exp(max - max) = 1`) or a tail element taking `f32::exp` would both
/// break bit-equality.
#[test]
fn softmax_tail_only_lengths_use_the_avx2_exp_core_bitwise() {
    if !avx2_fma_available("softmax_tail_only_lengths_use_the_avx2_exp_core_bitwise") {
        return;
    }
    let rows: [&[f32]; 7] = [
        &[0.3],
        &[1.0, -2.0],
        &[0.5, -1.25, 3.0],
        &[-90.0, 0.0, -87.9, 2.0],
        &[f32::NEG_INFINITY, 1.5, -0.5, 0.25, 4.0],
        &[10.0, 9.5, -3.0, 0.0, 7.25, -20.0],
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ];
    for row in rows {
        let mut got = row.to_vec();
        softmax_simd(&mut got);

        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let shifted: Vec<f32> = row.iter().map(|&v| v - max).collect();
        // SAFETY: AVX2+FMA support was just confirmed.
        let exps = unsafe { map_lanes(&shifted, exp_avx2_f32x8) };
        let inv_sum = 1.0 / exps.iter().sum::<f32>();
        for (i, (&g, &e)) in got.iter().zip(&exps).enumerate() {
            assert_eq!(
                g.to_bits(),
                (e * inv_sum).to_bits(),
                "softmax({row:?})[{i}] = {g:e}, expected exp_avx2_f32x8(v - max) / sum = {:e}",
                e * inv_sum
            );
        }
    }
}

/// Bit-exact scalar model of `softmax_avx2`: the row max; every element's
/// `exp_avx2_f32x8(v - max)` (8-lane chunks, the pad lanes of a short last
/// chunk computed and dropped); the denominator associated exactly as the
/// kernel does — each of the 8 body lanes accumulated across the full
/// chunks, those partial sums reduced by the extract-128 / shuffle / add
/// sequence as `((l0 + l4) + (l1 + l5)) + ((l2 + l6) + (l3 + l7))`, then the
/// tail's `exp`s summed among themselves and that sub-sum added once; and
/// every `exp` times `1 / sum`.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA.
unsafe fn softmax_avx2_model(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = row.iter().map(|&v| v - max).collect();
    let exps = map_lanes(&shifted, exp_avx2_f32x8);
    // The kernel's split: whole 8-lane vectors, then the 0-7 element tail.
    let (body, tail) = exps.as_chunks::<8>();
    let mut sum = 0.0f32;
    if !body.is_empty() {
        let mut lanes = [0.0f32; 8];
        for chunk in body {
            for (lane, &e) in lanes.iter_mut().zip(chunk) {
                *lane += e;
            }
        }
        let halves: [f32; 4] = std::array::from_fn(|k| lanes[k] + lanes[k + 4]);
        sum = (halves[0] + halves[1]) + (halves[2] + halves[3]);
    }
    if !tail.is_empty() {
        sum += tail.iter().sum::<f32>();
    }
    let inv_sum = 1.0 / sum;
    exps.iter().map(|&e| e * inv_sum).collect()
}

/// `softmax_avx2`'s 8-lane *body* (the production case, `n >= 8`) must take
/// its `exp` from `exp_avx2_f32x8` exactly as the padded tail does. The probe
/// rows put the unique maximum at index 0 — in the body — and every third
/// logit after it `87.75..102.5` below it: there the polynomial flushes to
/// exactly `+0.0` while a correctly rounded (or libm) `exp` is a nonzero
/// subnormal, so those outputs must be exactly `+0.0` wherever they land. The
/// whole row must also match [`softmax_avx2_model`] bit for bit, which pins
/// the polynomial on the ordinary logits and the denominator's association
/// (lane-wise body, horizontal reduction, grouped tail sub-sum). Lengths
/// 1..=24: zero to three full bodies with every tail size.
#[test]
fn softmax_body_and_tail_use_the_avx2_exp_core_bitwise() {
    if !avx2_fma_available("softmax_body_and_tail_use_the_avx2_exp_core_bitwise") {
        return;
    }
    for n in 1..=24usize {
        for max in SOFTMAX_PROBE_MAXIMA {
            let (row, flushes) = softmax_probe_row(n, max);
            let mut got = row.clone();
            softmax_simd(&mut got);
            // SAFETY: AVX2+FMA support was just confirmed.
            let want = unsafe { softmax_avx2_model(&row) };
            for (i, ((&g, &w), &flush)) in got.iter().zip(&want).zip(&flushes).enumerate() {
                if flush {
                    assert_eq!(
                        g.to_bits(),
                        0.0f32.to_bits(),
                        "softmax n={n} max={max} pos={i}: logit {} is {} below the max, \
                         where exp_avx2_f32x8 flushes to +0.0 (f32::exp gives {:e}), but the \
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
                     exp_avx2_f32x8 model's {w:e} ({:#010x})",
                    g.to_bits(),
                    w.to_bits()
                );
            }
        }
    }
}
