//! Scalar model of the K-M3 vector `exp` / `silu` cores, shared by both SIMD
//! tiers' tests.
//!
//! `exp_neon_f32x4` (aarch64, 4 lanes) and `exp_avx2_f32x8` (x86_64,
//! 8 lanes) are one algorithm with one set of constants and one operation
//! order. No host runs both, so instead of asserting that the two agree,
//! each is pinned bit-for-bit, lane by lane, to [`exp_cephes_model`] — a
//! scalar `f32::mul_add` transcription of that operation sequence — by a
//! test on its own architecture, over the sweeps defined here:
//! `avx2_core_tests::exp_avx2_f32x8_is_lane_wise_the_scalar_cephes_model`
//! and `neon_core_tests::exp_neon_f32x4_is_lane_wise_the_scalar_cephes_model`
//! (and the two `silu_core_*` against [`silu_cephes_model`] likewise). The
//! model's own accuracy and clamp behaviour are pinned by the tests at the
//! bottom of this file, on every architecture, so they carry over to
//! whichever vector core a host actually runs.
//!
//! `#[cfg(test)]` but deliberately not architecture-gated: attached to
//! `simd_float_ops` with `#[path]`, it compiles — and its self-tests run —
//! on every target, the scalar-fallback-only ones included.

/// `EXP_HI` exactly as both vector cores hold it (the `f32`
/// `88.37625885…`, bits `0x42b0_c0a5`, just below `127.5 * ln 2`).
pub(super) const EXP_HI: f32 = 88.376_26;

/// The first `x`, walking down from `-87`, at which the model — and so each
/// vector core — returns `+0.0` (`-87.683_13`): below `-126.5 * ln 2` the
/// rounded exponent `fx` reaches `-127`, whose biased encoding `0` builds
/// `2^fx` as `+0.0`. Located by exhaustive search over `[-EXP_HI, -87]`,
/// which `exp_cephes_model_saturates_flushes_and_keeps_nan` repeats.
pub(super) const FLUSH_START: f32 = f32::from_bits(0xc2af_5dc3);

/// The model (and each vector core) vs the correctly rounded `e^x`: every
/// `f32` of the dense sweep is within 1 ulp — as is every one of the
/// 2,237,399,041 `f32` in `[-87, 87]`
/// (`avx2_core_tests::exp_avx2_f32x8_is_within_1_ulp_of_the_correctly_rounded_exp_for_every_f32_in_minus_87_to_87`,
/// `#[ignore]`d).
pub(super) const MAX_ULP_VS_CORRECTLY_ROUNDED: u64 = 1;

/// The SiLU model (and each vector core) vs the exact `x / (1 + e^-x)`
/// evaluated in `f64` and rounded once, over [`dense_silu_sweep`].
pub(super) const MAX_SILU_ULP_VS_F64: u64 = 3;

/// Maps an `f32` onto an integer line that is monotone in its value, with
/// adjacent floats one apart (`-0.0` and `+0.0` both map to 0).
pub(super) fn ordered(x: f32) -> i64 {
    let bits = i64::from(x.to_bits() as i32);
    if bits < 0 {
        i64::from(i32::MIN) - bits
    } else {
        bits
    }
}

/// Inverse of [`ordered`] (0 maps back to `+0.0`).
pub(super) fn from_ordered(position: i64) -> f32 {
    if position < 0 {
        f32::from_bits((i64::from(i32::MIN) - position) as i32 as u32)
    } else {
        f32::from_bits(position as u32)
    }
}

/// Distance in units in the last place between two non-NaN `f32`s, counted
/// straight across zero.
pub(super) fn ulp_distance(a: f32, b: f32) -> u64 {
    ordered(a).abs_diff(ordered(b))
}

/// Every `f32` in `[lo, hi]`, in increasing order.
pub(super) fn every_f32_in(lo: f32, hi: f32) -> impl Iterator<Item = f32> {
    (ordered(lo)..=ordered(hi)).map(from_ordered)
}

/// `e^x` correctly rounded to `f32` (evaluated in `f64`, whose own error is
/// far below an `f32` half-ulp).
pub(super) fn exp_correctly_rounded(x: f32) -> f32 {
    f64::from(x).exp() as f32
}

/// The exact SiLU `x / (1 + e^-x)`, evaluated in `f64` and rounded once to
/// `f32` — a platform-independent reference (its own error, ~1e-16
/// relative, is far below an `f32` half-ulp), unlike `f32::exp`-based
/// `silu_scalar_elem`, whose accuracy is the platform libm's.
pub(super) fn silu_f64(x: f32) -> f32 {
    let x = f64::from(x);
    (x / (1.0 + (-x).exp())) as f32
}

/// The dense `exp` sweep: 1,740,001 evenly spaced points of `[-87, 87]`
/// (step `1e-4`); every 997th `f32` of `[-87, 87]` in value order (2.24
/// million points spread evenly in ulp space, so tiny `|x|` is covered as
/// densely as large `|x|`); and every one of the 180,390 `f32` in
/// `[87, EXP_HI]`, the upper clamp edge itself included.
pub(super) fn dense_exp_sweep() -> Vec<f32> {
    let mut xs: Vec<f32> = (0..=1_740_000u32)
        .map(|i| (-87.0 + f64::from(i) * 1e-4) as f32)
        .collect();
    xs.extend(
        (ordered(-87.0)..=ordered(87.0))
            .step_by(997)
            .map(from_ordered),
    );
    xs.extend(every_f32_in(87.0, EXP_HI));
    xs
}

/// The bit-exact pin's `exp` sweep: [`dense_exp_sweep`] plus every `f32` of
/// the lower clamp band `[-EXP_HI, -87]` (where results go subnormal, then
/// flush to `+0.0`) and every special the clamp handles. No NaN: a NaN lane
/// is checked separately, since only NaN-ness (not the payload) is pinned.
pub(super) fn exp_model_sweep() -> Vec<f32> {
    let mut xs = dense_exp_sweep();
    xs.extend(every_f32_in(-EXP_HI, -87.0));
    xs.extend([
        88.5,
        -88.5,
        100.0,
        -100.0,
        1.0e30,
        -1.0e30,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
    ]);
    xs
}

/// The SiLU accuracy sweep: 6,000,001 evenly spaced points of `[-30, 30]`.
pub(super) fn dense_silu_sweep() -> Vec<f32> {
    const N: u32 = 6_000_001;
    (0..N)
        .map(|i| -30.0 + 60.0 * i as f32 / (N - 1) as f32)
        .collect()
}

/// The bit-exact pin's SiLU sweep: [`dense_silu_sweep`]; every 997th `f32` of
/// `[-100, 100]` in value order (tiny `|x|` included); every `f32` within
/// `±0.01` of `-EXP_HI` (where `exp(-x)` starts to saturate) and of
/// `-FLUSH_START` (where `exp(-x)` flushes to `+0.0`); and the specials. No
/// NaN, as in [`exp_model_sweep`].
pub(super) fn silu_model_sweep() -> Vec<f32> {
    let mut xs = dense_silu_sweep();
    xs.extend(
        (ordered(-100.0)..=ordered(100.0))
            .step_by(997)
            .map(from_ordered),
    );
    xs.extend(every_f32_in(-EXP_HI - 0.01, -EXP_HI + 0.01));
    xs.extend(every_f32_in(-FLUSH_START - 0.01, -FLUSH_START + 0.01));
    xs.extend([
        1.0e30,
        -1.0e30,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
    ]);
    xs
}

/// Scalar, line-by-line transcription of `exp_neon_f32x4`'s operation
/// sequence — the reference both SIMD tiers are held to. Each `mul_add` is
/// one correctly rounded fused multiply-add, exactly like `vfmaq_f32` /
/// `vfmsq_f32` and `_mm256_fmadd_ps` / `_mm256_fnmadd_ps` (negating `fx` is
/// exact, so `(-fx).mul_add(c, x)` is `x - fx*c` rounded once); `as i32`
/// truncates toward zero like `vcvtq_s32_f32` and `_mm256_cvttps_epi32` on
/// the clamped range; `f32::clamp` keeps NaN, as both clamps do. The
/// constants are transcribed here again rather than shared, so a drifted
/// constant in a kernel fails its bit-exact test instead of silently moving
/// the reference with it.
pub(super) fn exp_cephes_model(x: f32) -> f32 {
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

    let x = x.clamp(EXP_LO, EXP_HI);

    let fx0 = x.mul_add(LOG2EF, 0.5);
    let fx_trunc = (fx0 as i32) as f32;
    let overshot = if fx_trunc > fx0 { 1.0 } else { 0.0 };
    let fx = fx_trunc - overshot;

    let r = (-fx).mul_add(EXP_C1, x);
    let r = (-fx).mul_add(EXP_C2, r);
    let z = r * r;

    let mut y = P0;
    for p in [P1, P2, P3, P4, P5] {
        y = y.mul_add(r, p);
    }
    let y = y.mul_add(z, r + 1.0);

    // `fx` is an integer in [-127, 127] here, so the biased exponent fits.
    let pow2n = f32::from_bits(((fx as i32 + 127) as u32) << 23);
    y * pow2n
}

/// Scalar transcription of `silu_core_neon_f32x4` / `silu_core_avx2_f32x8`:
/// `x / (1 + exp(-x))` with the model `exp`. Both cores negate with a
/// sign-bit flip (`vnegq_f32`, `XOR -0.0`) exactly as `-x` does here, then
/// add and divide once each, correctly rounded.
pub(super) fn silu_cephes_model(x: f32) -> f32 {
    x / (1.0 + exp_cephes_model(-x))
}

/// `exp` arguments — softmax logits relative to the row max — at which the
/// polynomial flushes to exactly `+0.0` while a correctly rounded `e^x` is a
/// nonzero subnormal `f32` (and so is glibc's `expf`): all lie in
/// `(-103, -87.7)`, i.e. at or below [`FLUSH_START`] but above
/// `ln(2^-150) = -103.97`, under which even the exact value rounds to zero.
/// A softmax test that finds `+0.0` at such a logit has proved the kernel
/// took its `exp` from the polynomial, not from libm.
pub(super) const SOFTMAX_FLUSH_OFFSETS: [f32; 5] = [-87.75, -90.0, -95.0, -100.0, -102.5];

/// The row maxima the softmax probe rows are built around: `0.0` (so a
/// logit is its own `exp` argument) and `3.25` (so the kernel's `v - max`
/// subtraction is exercised). Every offset in [`SOFTMAX_FLUSH_OFFSETS`] and
/// every logit [`softmax_probe_row`] derives from these is a small dyadic
/// rational, so `(max + offset) - max == offset` exactly.
pub(super) const SOFTMAX_PROBE_MAXIMA: [f32; 2] = [0.0, 3.25];

/// A softmax probe row of length `n` with its unique maximum `max` at index
/// 0 (inside the vector body whenever `n` reaches one vector width): every
/// third logit after it (`i % 3 == 1`) is `max + offset` for an offset from
/// [`SOFTMAX_FLUSH_OFFSETS`], cycling, and the rest lie between `0.5` and
/// `8.0` below `max`, where `exp` is an ordinary normal `f32`. Returns the
/// row and, per position, whether it holds a flush-range logit.
pub(super) fn softmax_probe_row(n: usize, max: f32) -> (Vec<f32>, Vec<bool>) {
    (0..n)
        .map(|i| {
            if i == 0 {
                (max, false)
            } else if i % 3 == 1 {
                let offset = SOFTMAX_FLUSH_OFFSETS[(i / 3) % SOFTMAX_FLUSH_OFFSETS.len()];
                (max + offset, true)
            } else {
                (max - 0.5 - ((i * 7) % 11) as f32 * 0.75, false)
            }
        })
        .unzip()
}

// ── the model's own accuracy and clamp behaviour, on every target ──────

#[test]
fn exp_cephes_model_is_within_1_ulp_of_the_correctly_rounded_exp() {
    let (mut worst, mut worst_at) = (0u64, 0.0f32);
    for x in dense_exp_sweep() {
        let distance = ulp_distance(exp_cephes_model(x), exp_correctly_rounded(x));
        if distance > worst {
            (worst, worst_at) = (distance, x);
        }
    }
    assert!(
        worst <= MAX_ULP_VS_CORRECTLY_ROUNDED,
        "exp_cephes_model is {worst} ulp from the correctly rounded e^x at x = {worst_at:e} \
         (bound {MAX_ULP_VS_CORRECTLY_ROUNDED})"
    );
}

#[test]
fn exp_cephes_model_saturates_flushes_and_keeps_nan() {
    let at_hi = exp_cephes_model(EXP_HI);
    assert!(
        at_hi.is_finite()
            && ulp_distance(at_hi, exp_correctly_rounded(EXP_HI)) <= MAX_ULP_VS_CORRECTLY_ROUNDED,
        "exp_cephes_model(EXP_HI) = {at_hi:e}"
    );
    for x in exp_model_sweep() {
        let y = exp_cephes_model(x);
        if x >= EXP_HI {
            assert_eq!(
                y.to_bits(),
                at_hi.to_bits(),
                "exp_cephes_model({x:e}) = {y:e}, expected the saturated exp(EXP_HI) = {at_hi:e}"
            );
        } else if x < -EXP_HI {
            assert_eq!(
                y.to_bits(),
                0.0f32.to_bits(),
                "exp_cephes_model({x:e}) = {y:e}, expected the +0.0 flush"
            );
        }
    }
    // FLUSH_START, re-derived: across the whole lower clamp band the result
    // is exactly +0.0 at and below it, and a positive (sub)normal above it.
    for x in every_f32_in(-EXP_HI, -87.0) {
        let y = exp_cephes_model(x);
        if x <= FLUSH_START {
            assert_eq!(
                y.to_bits(),
                0.0f32.to_bits(),
                "exp_cephes_model({x:e}) = {y:e}, expected the +0.0 flush at or below \
                 FLUSH_START = {FLUSH_START:e}"
            );
        } else {
            assert!(
                y > 0.0,
                "exp_cephes_model({x:e}) = {y:e}, expected a positive value above \
                 FLUSH_START = {FLUSH_START:e}"
            );
        }
    }
    assert!(exp_cephes_model(f32::NAN).is_nan(), "NaN must stay NaN");
}

#[test]
fn silu_cephes_model_is_within_3_ulp_of_the_f64_silu() {
    let (mut worst, mut worst_at) = (0u64, 0.0f32);
    for x in dense_silu_sweep() {
        let distance = ulp_distance(silu_cephes_model(x), silu_f64(x));
        if distance > worst {
            (worst, worst_at) = (distance, x);
        }
    }
    assert!(
        worst <= MAX_SILU_ULP_VS_F64,
        "silu_cephes_model is {worst} ulp from the f64 SiLU at x = {worst_at:e} \
         (bound {MAX_SILU_ULP_VS_F64})"
    );
}

#[test]
fn silu_cephes_model_saturates_below_the_exp_clamp_and_keeps_nan() {
    // Below -EXP_HI the clamp makes the result exactly x / (1 + exp(EXP_HI));
    // at and above -FLUSH_START, exp(-x) is +0.0 and the result is x itself.
    let denominator_at_clamp = 1.0 + exp_cephes_model(EXP_HI);
    for x in silu_model_sweep() {
        let y = silu_cephes_model(x);
        if x < -EXP_HI {
            assert_eq!(
                y.to_bits(),
                (x / denominator_at_clamp).to_bits(),
                "silu_cephes_model({x:e}) = {y:e}, expected x / (1 + exp(EXP_HI))"
            );
        } else if x >= -FLUSH_START {
            assert_eq!(
                y.to_bits(),
                x.to_bits(),
                "silu_cephes_model({x:e}) = {y:e}, expected x itself (exp(-x) flushed to +0.0)"
            );
        }
    }
    assert!(silu_cephes_model(f32::NAN).is_nan(), "NaN must stay NaN");
}

/// The softmax probe rows' premise, checked on every target: each flush-range
/// logit sits exactly `offset` below the max, where the model `exp` is `+0.0`
/// but the exact `e^offset` rounds to a nonzero subnormal; every other
/// non-max logit has an ordinary normal `exp`; and the max is unique.
#[test]
fn softmax_probe_rows_separate_the_polynomial_from_a_correctly_rounded_exp() {
    for offset in SOFTMAX_FLUSH_OFFSETS {
        assert_eq!(
            exp_cephes_model(offset).to_bits(),
            0.0f32.to_bits(),
            "the model exp({offset}) must flush to +0.0"
        );
        let rounded = exp_correctly_rounded(offset);
        assert!(
            rounded > 0.0 && rounded < f32::MIN_POSITIVE,
            "the correctly rounded exp({offset}) = {rounded:e} must be a nonzero subnormal"
        );
    }
    for n in 1..=24usize {
        for max in SOFTMAX_PROBE_MAXIMA {
            let (row, flushes) = softmax_probe_row(n, max);
            assert_eq!(row.len(), n);
            assert_eq!(row[0].to_bits(), max.to_bits(), "the max sits at index 0");
            for (i, (&logit, &flush)) in row.iter().zip(&flushes).enumerate().skip(1) {
                let shifted = logit - max;
                if flush {
                    assert!(
                        SOFTMAX_FLUSH_OFFSETS.contains(&shifted),
                        "n={n} max={max} pos={i}: {logit} - {max} = {shifted} is not exact"
                    );
                } else {
                    assert!(
                        shifted < 0.0 && exp_cephes_model(shifted) >= f32::MIN_POSITIVE,
                        "n={n} max={max} pos={i}: logit {logit} must be an ordinary non-max logit"
                    );
                }
            }
        }
    }
}
