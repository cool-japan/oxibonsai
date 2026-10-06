//! Golden acceptance tests for the blockwise FWHT-1024 kernel (finding
//! K-07, `bonsai2-design.md` §2.2 / §8.2). Black-box: every check goes
//! through `oxibonsai_kernels::hadamard`'s public API only.
//!
//! Covers, per the design's own acceptance list:
//! - involution: `fwht(fwht(x)) == x`;
//! - orthogonality: `‖fwht(x)‖₂ == ‖x‖₂`;
//! - equality with a naive `O(n²)` `H·x/√n` for `n == 1024`, doubling as the
//!   "golden parity vs the fork" check over 64 random rows (the fork's own
//!   reference, `ops.cpp`'s `ggml_compute_forward_fwht_impl`, computes
//!   exactly this linear operator — the natural-order Sylvester-Hadamard
//!   matrix, normalized by `1/√n` — so an independent from-first-principles
//!   evaluation of that matrix *is* a faithful fork cross-check without
//!   needing a compiled fork binary on this machine);
//! - forward/inverse round-trip;
//! - the ordering table (signs-then-FWHT forward, FWHT-then-signs inverse),
//!   proven by construction on a 2-element block where the two possible
//!   orderings provably diverge.

use oxibonsai_kernels::hadamard::{
    fwht_forward_signed, fwht_forward_signed_batch, fwht_in_place, fwht_inverse_signed,
};

/// Deterministic, dependency-free PRNG (xorshift64*) so this file needs no
/// `rand` dependency and every run is reproducible.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 8.0
    }

    fn vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next_f32()).collect()
    }

    fn signs(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| if self.next_f32() >= 0.0 { 1.0 } else { -1.0 })
            .collect()
    }
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let rel_scale = x.abs().max(y.abs()).max(1.0);
        assert!(
            diff <= tol * rel_scale,
            "{label} mismatch at [{i}]: got {x}, expected {y} (diff={diff})"
        );
    }
}

/// Independent O(n²) oracle: the natural-order Sylvester-Hadamard matrix
/// evaluated from its closed form, `H[i][j] = (-1)^popcount(i & j)`,
/// normalized by `1/√n`. This is algorithmically unrelated to the O(n log n)
/// butterfly network under test, so agreement is a real cross-check, not a
/// tautology.
fn naive_normalized_hadamard_transform(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    assert!(n.is_power_of_two(), "naive oracle requires power-of-two n");
    let scale = 1.0_f32 / (n as f32).sqrt();
    (0..n)
        .map(|i| {
            let mut acc = 0.0f32;
            for (j, &xj) in x.iter().enumerate() {
                let parity = (i & j).count_ones() % 2;
                acc += if parity == 0 { xj } else { -xj };
            }
            acc * scale
        })
        .collect()
}

// ── involution ──────────────────────────────────────────────────

#[test]
fn hadamard_involution_holds_for_a_single_1024_block() {
    let mut rng = Lcg::new(1);
    let original = rng.vec(1024);
    let mut x = original.clone();

    fwht_in_place(&mut x, 1024).expect("valid block");
    fwht_in_place(&mut x, 1024).expect("valid block");

    assert_close(&x, &original, 1e-5, "involution (n=1024)");
}

#[test]
fn hadamard_involution_holds_independently_per_block_across_a_real_folded_width() {
    // 17408 = ffn_gate/ffn_up's width: 17 independent 1024-blocks.
    let mut rng = Lcg::new(2);
    let original = rng.vec(17408);
    let mut x = original.clone();

    fwht_in_place(&mut x, 1024).expect("valid width/block");
    fwht_in_place(&mut x, 1024).expect("valid width/block");

    assert_close(&x, &original, 1e-5, "involution (width=17408, block=1024)");
}

// ── orthogonality ───────────────────────────────────────────────

#[test]
fn hadamard_orthogonality_preserves_l2_norm_at_n_1024() {
    let mut rng = Lcg::new(3);
    let x = rng.vec(1024);
    let mut y = x.clone();

    fwht_in_place(&mut y, 1024).expect("valid block");

    let norm_in: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_out: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_in - norm_out).abs() <= 1e-5 * norm_in,
        "orthogonality: ||x||={norm_in}, ||fwht(x)||={norm_out}"
    );
}

// ── vs naive H*x/sqrt(n), n=1024, 64 random rows ("golden parity vs the
//    fork" — see module docs) ──────────────────────────────────

#[test]
fn matches_naive_hadamard_matrix_over_64_random_rows_n_1024() {
    let mut rng = Lcg::new(4);
    for row in 0..64usize {
        let x = rng.vec(1024);
        let mut y = x.clone();
        fwht_in_place(&mut y, 1024).expect("valid block");

        let expected = naive_normalized_hadamard_transform(&x);
        assert_close(
            &y,
            &expected,
            1e-5,
            &format!("golden parity vs naive Hadamard matrix, row {row}"),
        );
    }
}

#[test]
fn matches_naive_hadamard_matrix_over_64_random_rows_smaller_n() {
    // Cheaper sibling of the n=1024 check above, at sizes small enough to
    // also exercise every SIMD pass-length boundary (2, 4, 8, 16, 32, 64).
    for &n in &[2usize, 4, 8, 16, 32, 64] {
        let mut rng = Lcg::new(1000 + n as u64);
        for row in 0..64usize {
            let x = rng.vec(n);
            let mut y = x.clone();
            fwht_in_place(&mut y, n).expect("valid block");

            let expected = naive_normalized_hadamard_transform(&x);
            assert_close(
                &y,
                &expected,
                1e-5,
                &format!("golden parity vs naive Hadamard matrix, n={n} row={row}"),
            );
        }
    }
}

// ── forward/inverse round-trip ──────────────────────────────────

#[test]
fn hadamard_forward_then_inverse_round_trips_at_n_1024() {
    let mut rng = Lcg::new(5);
    let original = rng.vec(1024);
    let signs = rng.signs(1024);
    let mut x = original.clone();

    fwht_forward_signed(&mut x, &signs, 1024).expect("valid inputs");
    fwht_inverse_signed(&mut x, &signs, 1024).expect("valid inputs");

    assert_close(&x, &original, 1e-5, "forward/inverse round-trip (n=1024)");
}

#[test]
fn hadamard_forward_then_inverse_round_trips_across_real_folded_widths() {
    for &width in &[5120usize, 6144, 17408] {
        let mut rng = Lcg::new(width as u64);
        let original = rng.vec(width);
        let signs = rng.signs(width);
        let mut x = original.clone();

        fwht_forward_signed(&mut x, &signs, 1024).expect("valid inputs");
        fwht_inverse_signed(&mut x, &signs, 1024).expect("valid inputs");

        assert_close(
            &x,
            &original,
            1e-5,
            &format!("forward/inverse round-trip (width={width})"),
        );
    }
}

// ── ordering, BY CONSTRUCTION, on a 2-element block ─────────────
//
// `s0 != s1` makes "signs then FWHT" and "FWHT then signs" produce
// numerically different results, so matching the correct formula (and NOT
// matching the other one) is a real, constructive proof of which order this
// crate implements — not just a round-trip that could pass even if both
// directions were consistently (and wrongly) swapped together.

#[test]
fn hadamard_forward_applies_signs_before_fwht() {
    let (a, b) = (3.0f32, 1.0f32);
    let (s0, s1) = (1.0f32, -1.0f32);
    let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();

    let correct = [
        (a * s0 + b * s1) * inv_sqrt2, // signs first, then FWHT
        (a * s0 - b * s1) * inv_sqrt2,
    ];
    let wrong = [
        (a + b) * inv_sqrt2 * s0, // FWHT first, then signs (the other order)
        (a - b) * inv_sqrt2 * s1,
    ];

    let mut x = [a, b];
    fwht_forward_signed(&mut x, &[s0, s1], 2).expect("2-element block is valid");

    assert_close(&x, &correct, 1e-5, "forward: signs-then-FWHT");
    assert!(
        (x[0] - wrong[0]).abs() > 1e-3 || (x[1] - wrong[1]).abs() > 1e-3,
        "forward must not match the FWHT-then-signs ordering, got {x:?}"
    );
}

#[test]
fn hadamard_inverse_applies_fwht_before_signs() {
    let (a, b) = (3.0f32, 1.0f32);
    let (s0, s1) = (1.0f32, -1.0f32);
    let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();

    let correct = [
        (a + b) * inv_sqrt2 * s0, // FWHT first, then signs
        (a - b) * inv_sqrt2 * s1,
    ];
    let wrong = [
        (a * s0 + b * s1) * inv_sqrt2, // signs first, then FWHT (the other order)
        (a * s0 - b * s1) * inv_sqrt2,
    ];

    let mut x = [a, b];
    fwht_inverse_signed(&mut x, &[s0, s1], 2).expect("2-element block is valid");

    assert_close(&x, &correct, 1e-5, "inverse: FWHT-then-signs");
    assert!(
        (x[0] - wrong[0]).abs() > 1e-3 || (x[1] - wrong[1]).abs() > 1e-3,
        "inverse must not match the signs-then-FWHT ordering, got {x:?}"
    );
}

// ── batch entry point, black-box ────────────────────────────────

#[test]
fn hadamard_batch_forward_matches_per_row_forward() {
    let width = 1024usize;
    let rows = 12usize;
    let mut rng = Lcg::new(6);
    let signs = rng.signs(width);
    let base = rng.vec(rows * width);

    let mut via_batch = base.clone();
    fwht_forward_signed_batch(&mut via_batch, &signs, 1024, rows).expect("valid batch");

    let mut via_rows = base;
    for row in via_rows.chunks_exact_mut(width) {
        fwht_forward_signed(row, &signs, 1024).expect("valid row");
    }

    assert_close(&via_batch, &via_rows, 1e-6, "batch vs per-row forward");
}

// ── error paths (black-box) ──────────────────────────────────────

#[test]
fn hadamard_rejects_non_power_of_two_block_size() {
    let mut x = vec![0.0f32; 30];
    assert!(fwht_in_place(&mut x, 30).is_err());
}

#[test]
fn hadamard_rejects_misaligned_row_length() {
    let mut x = vec![0.0f32; 100]; // not a multiple of 1024
    assert!(fwht_in_place(&mut x, 1024).is_err());
}

#[test]
fn hadamard_rejects_wrong_length_signs() {
    let mut x = vec![0.0f32; 1024];
    let signs = vec![1.0f32; 512]; // half the required width
    assert!(fwht_forward_signed(&mut x, &signs, 1024).is_err());
    assert!(fwht_inverse_signed(&mut x, &signs, 1024).is_err());
}

#[test]
fn hadamard_batch_rejects_inconsistent_row_count() {
    let signs = vec![1.0f32; 1024];
    let mut x = vec![0.0f32; 1024 * 3]; // claims to be 3 rows...
    assert!(fwht_forward_signed_batch(&mut x, &signs, 1024, 4).is_err()); // ...but 4 is asserted
}
