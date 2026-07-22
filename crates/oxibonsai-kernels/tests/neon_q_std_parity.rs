//! Parity tests: scalar Q4_0/Q8_0 reference vs the new NEON GEMV kernels.
//!
//! Mirrors `fp8_simd_parity.rs`: for each format (Q4_0, Q8_0) we generate
//! deterministic pseudo-random blocks, compute an independent oracle by
//! dequantizing with the public `oxibonsai_core` API and doing a plain scalar
//! dot product, then compare against `oxibonsai_kernels::simd_q_std_neon`'s
//! output within a tight tolerance.
//!
//! (The crate's own `gemv_q4_0_scalar` / `gemv_q8_0_scalar` references are
//! `pub(crate)`, so this external test crate cannot call them directly; the
//! dequant+dot oracle below is numerically the same computation and uses only
//! public API, so it is an equally valid parity reference.)
//!
//! These tests are `#[cfg(target_arch = "aarch64")]`-gated and additionally
//! skip at runtime if NEON is somehow unavailable (NEON is mandatory on
//! AArch64, so this is a defensive no-op skip, not an expected path).

#![cfg(target_arch = "aarch64")]

use oxibonsai_core::{BlockQ4_0, BlockQ8_0, QK_Q4_0, QK_Q8_0};
use oxibonsai_kernels::simd_q_std_neon::{gemv_q4_0_neon, gemv_q8_0_neon};

// ─── Deterministic LCG RNG ────────────────────────────────────────────────

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn lcg_rand_f32(state: &mut u64, scale: f32) -> f32 {
    let x = lcg_next(state);
    (((x >> 11) as f32) / (1u64 << 53) as f32) * 2.0 * scale - scale
}

// ─── Fixture builders ──────────────────────────────────────────────────────

fn make_input(len: usize, rng: &mut u64, scale: f32) -> Vec<f32> {
    (0..len).map(|_| lcg_rand_f32(rng, scale)).collect()
}

fn make_q4_0_blocks(n_rows: usize, blocks_per_row: usize, rng: &mut u64) -> Vec<BlockQ4_0> {
    let raw = make_input(n_rows * blocks_per_row * QK_Q4_0, rng, 5.0);
    BlockQ4_0::quantize(&raw).expect("Q4_0 quantize should succeed")
}

fn make_q8_0_blocks(n_rows: usize, blocks_per_row: usize, rng: &mut u64) -> Vec<BlockQ8_0> {
    let raw = make_input(n_rows * blocks_per_row * QK_Q8_0, rng, 5.0);
    BlockQ8_0::quantize(&raw).expect("Q8_0 quantize should succeed")
}

/// Independent oracle: dequantize with the public core API, then a plain
/// scalar dot product per row. Numerically identical to what the crate's
/// internal `gemv_q{4,8}_0_scalar` compute, but reachable without `pub(crate)`.
fn dot_oracle_q4_0(
    blocks: &[BlockQ4_0],
    input: &[f32],
    n_rows: usize,
    blocks_per_row: usize,
) -> Vec<f32> {
    let k = blocks_per_row * QK_Q4_0;
    let mut deq = vec![0.0f32; blocks.len() * QK_Q4_0];
    BlockQ4_0::dequant(blocks, &mut deq).expect("Q4_0 dequant should succeed");

    (0..n_rows)
        .map(|row| {
            let row_deq = &deq[row * k..(row + 1) * k];
            row_deq.iter().zip(input.iter()).map(|(w, x)| w * x).sum()
        })
        .collect()
}

fn dot_oracle_q8_0(
    blocks: &[BlockQ8_0],
    input: &[f32],
    n_rows: usize,
    blocks_per_row: usize,
) -> Vec<f32> {
    let k = blocks_per_row * QK_Q8_0;
    let mut deq = vec![0.0f32; blocks.len() * QK_Q8_0];
    BlockQ8_0::dequant(blocks, &mut deq).expect("Q8_0 dequant should succeed");

    (0..n_rows)
        .map(|row| {
            let row_deq = &deq[row * k..(row + 1) * k];
            row_deq.iter().zip(input.iter()).map(|(w, x)| w * x).sum()
        })
        .collect()
}

// Note: `dot_oracle_*` above sums a per-row dot product using the *global*
// `input` slice (length `k`), matching the GEMV contract where every row is
// multiplied against the same input vector.

fn assert_close(a: &[f32], b: &[f32], label: &str) {
    let rel_tol = 1e-4_f32;
    let abs_tol = 1e-4_f32;
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        let scale = va.abs().max(vb.abs()).max(1.0);
        let tol = abs_tol + rel_tol * scale;
        assert!(
            diff <= tol,
            "{label}[{i}]: |{va} - {vb}| = {diff} > {tol} (abs={abs_tol} + rel={rel_tol}×{scale})"
        );
    }
}

/// (n_rows, in_features) boundary matrix: 1 block, several blocks, odd row
/// counts, larger k — mirrors the shapes exercised in `fp8_simd_parity.rs`.
const SHAPES: [(usize, usize); 6] = [(1, 32), (3, 64), (5, 96), (17, 128), (2, 160), (1, 1024)];

// ─── Q4_0 GEMV parity ──────────────────────────────────────────────────────

#[test]
fn q4_0_neon_gemv_matches_dequant_oracle() {
    let mut rng = 0xFEDC_BA98_7654_3210_u64;
    for (n_rows, in_features) in SHAPES {
        let blocks_per_row = in_features / QK_Q4_0;
        let blocks = make_q4_0_blocks(n_rows, blocks_per_row, &mut rng);
        let input = make_input(in_features, &mut rng, 3.0);

        let oracle = dot_oracle_q4_0(&blocks, &input, n_rows, blocks_per_row);

        let mut neon_out = vec![0.0_f32; n_rows];
        unsafe {
            gemv_q4_0_neon(&blocks, &input, &mut neon_out, n_rows, in_features)
                .expect("neon q4_0 gemv should succeed");
        }

        assert_close(
            &oracle,
            &neon_out,
            &format!("q4_0 neon gemv (n_rows={n_rows}, k={in_features})"),
        );
    }
}

// ─── Q8_0 GEMV parity ──────────────────────────────────────────────────────

#[test]
fn q8_0_neon_gemv_matches_dequant_oracle() {
    let mut rng = 0x0123_4567_89AB_CDEF_u64;
    for (n_rows, in_features) in SHAPES {
        let blocks_per_row = in_features / QK_Q8_0;
        let blocks = make_q8_0_blocks(n_rows, blocks_per_row, &mut rng);
        let input = make_input(in_features, &mut rng, 3.0);

        let oracle = dot_oracle_q8_0(&blocks, &input, n_rows, blocks_per_row);

        let mut neon_out = vec![0.0_f32; n_rows];
        unsafe {
            gemv_q8_0_neon(&blocks, &input, &mut neon_out, n_rows, in_features)
                .expect("neon q8_0 gemv should succeed");
        }

        assert_close(
            &oracle,
            &neon_out,
            &format!("q8_0 neon gemv (n_rows={n_rows}, k={in_features})"),
        );
    }
}

// ─── Error-path parity (dimension validation) ──────────────────────────────

#[test]
fn q4_0_neon_gemv_dimension_errors() {
    let mut rng = 0x1111_2222_3333_4444_u64;
    let blocks = make_q4_0_blocks(1, 1, &mut rng);
    let input = make_input(32, &mut rng, 2.0);

    // Not block-aligned in_features.
    let mut output = vec![0.0f32; 1];
    unsafe {
        assert!(
            gemv_q4_0_neon(&blocks, &input, &mut output, 1, 31).is_err(),
            "expected an error for in_features not a multiple of QK_Q4_0"
        );
    }

    // Output buffer too small.
    let mut tiny_output: Vec<f32> = vec![];
    unsafe {
        assert!(
            gemv_q4_0_neon(&blocks, &input, &mut tiny_output, 1, 32).is_err(),
            "expected an error for an undersized output buffer"
        );
    }

    // Not enough blocks for the requested n_rows.
    let mut output2 = vec![0.0f32; 2];
    unsafe {
        assert!(
            gemv_q4_0_neon(&blocks, &input, &mut output2, 2, 32).is_err(),
            "expected an error for too few weight blocks"
        );
    }
}

#[test]
fn q8_0_neon_gemv_dimension_errors() {
    let mut rng = 0x5555_6666_7777_8888_u64;
    let blocks = make_q8_0_blocks(1, 1, &mut rng);
    let input = make_input(32, &mut rng, 2.0);

    let mut output = vec![0.0f32; 1];
    unsafe {
        assert!(
            gemv_q8_0_neon(&blocks, &input, &mut output, 1, 31).is_err(),
            "expected an error for in_features not a multiple of QK_Q8_0"
        );
    }

    let mut tiny_output: Vec<f32> = vec![];
    unsafe {
        assert!(
            gemv_q8_0_neon(&blocks, &input, &mut tiny_output, 1, 32).is_err(),
            "expected an error for an undersized output buffer"
        );
    }

    let mut output2 = vec![0.0f32; 2];
    unsafe {
        assert!(
            gemv_q8_0_neon(&blocks, &input, &mut output2, 2, 32).is_err(),
            "expected an error for too few weight blocks"
        );
    }
}
