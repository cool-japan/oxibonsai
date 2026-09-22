//! Property-based tests for Q1_0_g128 and ternary TQ2_0_g128 kernel
//! correctness (the ternary coverage below is K-04 / T-10: before it
//! existed, this file's only strategy was `arb_block -> BlockQ1_0G128`, and
//! all fuzz/property coverage in the crate was 1-bit-only).

use half::f16;
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
#[cfg(target_arch = "aarch64")]
use oxibonsai_core::{BlockTQ2_0_g128, QK_TQ2_0_G128};
use oxibonsai_kernels::dequant::dequant_1bit_g128;
use oxibonsai_kernels::gemm::gemm_1bit_g128;
use oxibonsai_kernels::gemv::gemv_1bit_g128;
#[cfg(target_arch = "aarch64")]
use oxibonsai_kernels::{dequant_ternary, gemm_ternary, gemv_ternary};
use proptest::prelude::*;

/// Strategy to generate a single BlockQ1_0G128 with a finite, non-zero scale.
fn arb_block() -> impl Strategy<Value = BlockQ1_0G128> {
    (
        prop::num::f32::NORMAL.prop_filter("non-zero finite scale", |v| {
            v.is_finite() && v.abs() > 1e-6 && v.abs() < 100.0
        }),
        prop::array::uniform16(any::<u8>()),
    )
        .prop_map(|(scale, qs)| BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs,
        })
}

/// Strategy for a vector of blocks (1..=4 blocks).
fn arb_blocks(count: usize) -> impl Strategy<Value = Vec<BlockQ1_0G128>> {
    prop::collection::vec(arb_block(), count..=count)
}

/// Strategy to generate a single `BlockTQ2_0_g128` with a finite, non-zero
/// scale and `qs` bytes spanning the *arbitrary* `u8` range — unlike the
/// crate's in-file ternary unit tests, which only ever use `0xAA`/`0x00`/
/// `0x46`, this covers every 2-bit code including the reserved `0b11`
/// (K-01) across the proptest run's cases.
///
/// aarch64-only: every caller below compares against a `simd_neon` kernel,
/// which only exists on this target.
#[cfg(target_arch = "aarch64")]
fn arb_ternary_block() -> impl Strategy<Value = BlockTQ2_0_g128> {
    (
        prop::num::f32::NORMAL.prop_filter("non-zero finite scale", |v| {
            v.is_finite() && v.abs() > 1e-6 && v.abs() < 100.0
        }),
        prop::array::uniform32(any::<u8>()),
    )
        .prop_map(|(scale, qs)| BlockTQ2_0_g128 {
            d: f16::from_f32(scale),
            qs,
        })
}

/// Strategy for a `(n_rows, k, blocks, input)` GEMV case: `n_rows` in
/// `1..=8`, `k` a multiple of 128 in `{128, 256, 384}`, `blocks` exactly
/// `n_rows * (k / 128)` arbitrary-byte ternary blocks, `input` exactly `k`
/// finite f32s. Small ranges deliberately (T-13's nextest slow-timeout
/// margin is thin): this is about hitting every 2-bit code, not about
/// exercising large matrices.
#[cfg(target_arch = "aarch64")]
fn arb_ternary_gemv_case() -> impl Strategy<Value = (usize, usize, Vec<BlockTQ2_0_g128>, Vec<f32>)>
{
    (1usize..=8, 1usize..=3).prop_flat_map(|(n_rows, k_mult)| {
        let k = k_mult * QK_TQ2_0_G128;
        let n_blocks = n_rows * (k / QK_TQ2_0_G128);
        (
            Just(n_rows),
            Just(k),
            prop::collection::vec(arb_ternary_block(), n_blocks..=n_blocks),
            prop::collection::vec(
                prop::num::f32::NORMAL.prop_filter("finite", |v| v.is_finite() && v.abs() < 10.0),
                k..=k,
            ),
        )
    })
}

/// Strategy for a `(m, n_rows, k, blocks, input)` GEMM case, built the same
/// way as [`arb_ternary_gemv_case`] plus a batch dimension `m in 1..=4`.
#[cfg(target_arch = "aarch64")]
fn arb_ternary_gemm_case(
) -> impl Strategy<Value = (usize, usize, usize, Vec<BlockTQ2_0_g128>, Vec<f32>)> {
    (1usize..=4, 1usize..=8, 1usize..=3).prop_flat_map(|(m, n_rows, k_mult)| {
        let k = k_mult * QK_TQ2_0_G128;
        let n_blocks = n_rows * (k / QK_TQ2_0_G128);
        (
            Just(m),
            Just(n_rows),
            Just(k),
            prop::collection::vec(arb_ternary_block(), n_blocks..=n_blocks),
            prop::collection::vec(
                prop::num::f32::NORMAL.prop_filter("finite", |v| v.is_finite() && v.abs() < 10.0),
                (m * k)..=(m * k),
            ),
        )
    })
}

/// House-style relative tolerance (matches `simd_q_std_neon.rs` and
/// `tests/ternary_cross_tier.rs`).
#[cfg(target_arch = "aarch64")]
fn rel_tol(reference: f32) -> f32 {
    1e-4 * reference.abs().max(1.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every dequantized value is either +d or -d (for finite, non-zero scales).
    #[test]
    fn dequant_outputs_are_pm_scale(block in arb_block()) {
        let d = block.d.to_f32();
        if !d.is_finite() || d.abs() < 1e-6 {
            return Ok(());
        }
        let mut output = vec![0.0f32; QK1_0_G128];
        dequant_1bit_g128(&[block], &mut output)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        for (i, &v) in output.iter().enumerate() {
            let abs_diff_pos = (v - d).abs();
            let abs_diff_neg = (v + d).abs();
            let tol = d.abs() * 0.02; // f16 rounding tolerance
            prop_assert!(
                abs_diff_pos < tol || abs_diff_neg < tol,
                "output[{i}]={v} is not close to +d={d} or -d={neg_d}",
                neg_d = -d,
            );
        }
    }

    /// gemv(A, alpha*x) ~= alpha * gemv(A, x) (linearity in input).
    #[test]
    fn gemv_linearity(
        blocks in arb_blocks(1),
        alpha in prop::num::f32::NORMAL.prop_filter("finite nonzero", |v| v.is_finite() && v.abs() > 0.01 && v.abs() < 10.0),
    ) {
        let k = QK1_0_G128;
        let n_rows = 1;
        // Deterministic input
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 0.64).collect();
        let scaled_input: Vec<f32> = input.iter().map(|&v| v * alpha).collect();

        let mut out_base = vec![0.0f32; n_rows];
        let mut out_scaled = vec![0.0f32; n_rows];

        gemv_1bit_g128(&blocks, &input, &mut out_base, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        gemv_1bit_g128(&blocks, &scaled_input, &mut out_scaled, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;

        let expected = out_base[0] * alpha;
        let actual = out_scaled[0];
        let tol = expected.abs() * 0.05 + 1.0; // tolerance for f16 scale rounding
        prop_assert!(
            (expected - actual).abs() < tol,
            "linearity: alpha*gemv(x)={expected} vs gemv(alpha*x)={actual}, alpha={alpha}"
        );
    }

    /// gemm with m=1 matches gemv output.
    #[test]
    fn gemm_m1_equals_gemv(blocks in arb_blocks(2)) {
        let k = QK1_0_G128;
        let n_rows = 2;
        let m = 1;
        // Rebuild blocks for 2 rows, 1 block per row
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 0.64).collect();

        let mut out_gemv = vec![0.0f32; n_rows];
        let mut out_gemm = vec![0.0f32; m * n_rows];

        gemv_1bit_g128(&blocks, &input, &mut out_gemv, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        gemm_1bit_g128(&blocks, &input, &mut out_gemm, m, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;

        for i in 0..n_rows {
            let diff = (out_gemv[i] - out_gemm[i]).abs();
            prop_assert!(
                diff < 0.01,
                "gemm(m=1) vs gemv mismatch at row {i}: gemv={}, gemm={}",
                out_gemv[i],
                out_gemm[i],
            );
        }
    }

    /// NEON dequant output matches reference exactly.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_reference_dequant(block in arb_block()) {
        let mut out_ref = vec![0.0f32; QK1_0_G128];
        let mut out_neon = vec![0.0f32; QK1_0_G128];

        dequant_1bit_g128(&[block], &mut out_ref)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        unsafe {
            oxibonsai_kernels::simd_neon::dequant_1bit_g128_neon(&[block], &mut out_neon)
                .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        }

        for i in 0..QK1_0_G128 {
            let diff = (out_ref[i] - out_neon[i]).abs();
            prop_assert!(
                diff < 0.01,
                "dequant mismatch at {i}: ref={}, neon={}",
                out_ref[i],
                out_neon[i],
            );
        }
    }

    /// NEON gemv matches reference within tolerance.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_reference_gemv(blocks in arb_blocks(1)) {
        let k = QK1_0_G128;
        let n_rows = 1;
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 0.64).collect();

        let mut out_ref = vec![0.0f32; n_rows];
        let mut out_neon = vec![0.0f32; n_rows];

        gemv_1bit_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        unsafe {
            oxibonsai_kernels::simd_neon::gemv_1bit_g128_neon(
                &blocks, &input, &mut out_neon, n_rows, k,
            )
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        }

        for i in 0..n_rows {
            let diff = (out_ref[i] - out_neon[i]).abs();
            prop_assert!(
                diff < 0.5,
                "gemv mismatch at row {i}: ref={}, neon={}",
                out_ref[i],
                out_neon[i],
            );
        }
    }

    // ───────────────── Ternary TQ2_0_g128 (K-04 / T-10) ─────────────────

    /// Ternary dequant: NEON matches reference exactly (bit-exact, same
    /// arithmetic operand order) across arbitrary `qs` bytes -- including
    /// the reserved `0b11` code that K-01 found decoded inconsistently.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn ternary_neon_matches_reference_dequant(block in arb_ternary_block()) {
        let mut out_ref = vec![0.0f32; QK_TQ2_0_G128];
        let mut out_neon = vec![0.0f32; QK_TQ2_0_G128];

        dequant_ternary::dequant_tq2_0_g128(&[block], &mut out_ref)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        unsafe {
            oxibonsai_kernels::simd_neon::dequant_tq2_0_g128_neon(&[block], &mut out_neon)
                .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        }

        for i in 0..QK_TQ2_0_G128 {
            prop_assert_eq!(
                out_ref[i], out_neon[i],
                "ternary dequant mismatch at {}: ref={}, neon={}", i, out_ref[i], out_neon[i],
            );
        }
    }

    /// Ternary GEMV: NEON matches reference within the house-style relative
    /// tolerance, over arbitrary `(n_rows, k, qs bytes)`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn ternary_neon_matches_reference_gemv((n_rows, k, blocks, input) in arb_ternary_gemv_case()) {
        let mut out_ref = vec![0.0f32; n_rows];
        let mut out_neon = vec![0.0f32; n_rows];

        gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        unsafe {
            oxibonsai_kernels::simd_neon::gemv_tq2_0_g128_neon(
                &blocks, &input, &mut out_neon, n_rows, k,
            )
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        }

        for i in 0..n_rows {
            let tol = rel_tol(out_ref[i]);
            prop_assert!(
                (out_ref[i] - out_neon[i]).abs() < tol,
                "ternary gemv mismatch at row {i} (n_rows={n_rows}, k={k}): ref={}, neon={}, tol={tol}",
                out_ref[i], out_neon[i],
            );
        }
    }

    /// Ternary GEMV prefetch variant: same property as
    /// `ternary_neon_matches_reference_gemv`, against the dispatcher's
    /// actual routing target for `KernelTier::Neon` GEMV
    /// (`dispatch.rs`'s `gemv_ternary_g128` always uses the `_prefetch`
    /// variant, not the plain one).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn ternary_neon_matches_reference_gemv_prefetch((n_rows, k, blocks, input) in arb_ternary_gemv_case()) {
        let mut out_ref = vec![0.0f32; n_rows];
        let mut out_neon_pf = vec![0.0f32; n_rows];

        gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        unsafe {
            oxibonsai_kernels::simd_neon::gemv_tq2_0_g128_neon_prefetch(
                &blocks, &input, &mut out_neon_pf, n_rows, k,
            )
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        }

        for i in 0..n_rows {
            let tol = rel_tol(out_ref[i]);
            prop_assert!(
                (out_ref[i] - out_neon_pf[i]).abs() < tol,
                "ternary gemv_prefetch mismatch at row {i} (n_rows={n_rows}, k={k}): ref={}, neon_pf={}, tol={tol}",
                out_ref[i], out_neon_pf[i],
            );
        }
    }

    /// Ternary GEMM: NEON matches the reference `gemm_ternary::gemm_tq2_0_g128`
    /// within the house-style relative tolerance, over arbitrary
    /// `(m, n_rows, k, qs bytes)` -- this is the exact op K-01's canary
    /// (`gemm_tq2_0_g128_neon`) diverged on.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn ternary_neon_matches_reference_gemm((m, n_rows, k, blocks, input) in arb_ternary_gemm_case()) {
        let mut out_ref = vec![0.0f32; m * n_rows];
        let mut out_neon = vec![0.0f32; m * n_rows];

        gemm_ternary::gemm_tq2_0_g128(&blocks, &input, &mut out_ref, m, n_rows, k)
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        unsafe {
            oxibonsai_kernels::simd_neon::gemm_tq2_0_g128_neon(
                &blocks, &input, &mut out_neon, m, n_rows, k,
            )
            .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        }

        for i in 0..(m * n_rows) {
            let tol = rel_tol(out_ref[i]);
            prop_assert!(
                (out_ref[i] - out_neon[i]).abs() < tol,
                "ternary gemm mismatch at idx {i} (m={m}, n_rows={n_rows}, k={k}): ref={}, neon={}, tol={tol}",
                out_ref[i], out_neon[i],
            );
        }
    }
}
