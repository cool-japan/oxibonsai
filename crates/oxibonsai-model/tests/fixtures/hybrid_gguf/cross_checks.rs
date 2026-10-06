//! Self-checks of the f64 primitives, exposed as `pub fn`s so
//! `hybrid_fixture_tests.rs` can assert them without duplicating the math.
//! They are independent of `build()` and of the reference forward driver, so
//! a wiring bug in the driver cannot mask a bug in a primitive.

use super::math::{
    apply_partial_rope_f64, fwht_forward_signed_f64, fwht_inverse_signed_f64,
    gdn_quadratic_reconstruction_f64, gdn_step_fused_f64, gdn_step_naive_3pass_f64,
};
use super::Xorshift64Star;

// ═════════════════════════════════════════════════════════════════════════
// 12. Internal self-checks — exposed as `pub fn`s so
//     `hybrid_fixture_tests.rs` can assert them without duplicating the
//     math, but independent of `build()`/`run_reference_forward` so a
//     wiring bug in the forward driver cannot mask a bug in a primitive.
// ═════════════════════════════════════════════════════════════════════════

/// `inverse(forward(x)) == x` (design §7.2's FWHT round-trip row), for a
/// deterministic pseudo-random `x` and sign vector of the given width.
pub fn fwht_round_trip_check(seed: u64, width: usize, block: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Xorshift64Star::new(seed);
    let x: Vec<f64> = (0..width).map(|_| rng.next_range_f64(-3.0, 3.0)).collect();
    let signs: Vec<f64> = (0..width)
        .map(|_| {
            if rng.next_u64().is_multiple_of(2) {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    let folded = fwht_forward_signed_f64(&x, &signs, block);
    let recovered = fwht_inverse_signed_f64(&folded, &signs, block);
    (x, recovered)
}

/// `(fused_two_steps, naive_two_steps, quadratic_two_steps)`, each
/// `[2][head_v_dim]` — see [`gdn_cross_check`].
pub type GdnCrossCheckResult = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// One GDN step computed three independent ways: fused single-pass, naive
/// 3-pass, and (via a 2-step sequential run feeding the quadratic
/// reconstruction) the O(T²) closed form. Returns
/// [`GdnCrossCheckResult`] for the caller to compare pairwise.
pub fn gdn_cross_check(seed: u64, head_k_dim: usize, head_v_dim: usize) -> GdnCrossCheckResult {
    let mut rng = Xorshift64Star::new(seed);
    let mut gen_vec =
        |n: usize| -> Vec<f64> { (0..n).map(|_| rng.next_range_f64(-1.0, 1.0)).collect() };

    let q_steps: Vec<Vec<f64>> = (0..2).map(|_| gen_vec(head_k_dim)).collect();
    let k_steps: Vec<Vec<f64>> = (0..2).map(|_| gen_vec(head_k_dim)).collect();
    let v_steps: Vec<Vec<f64>> = (0..2).map(|_| gen_vec(head_v_dim)).collect();
    let alpha_steps = gen_vec(2);
    let beta_steps = gen_vec(2);
    let dt_bias = rng.next_range_f64(-0.5, 0.5);
    let a_neg = -rng.next_range_f64(0.1, 1.0);

    let mut state_fused = vec![0.0f64; head_v_dim * head_k_dim];
    let mut state_naive = vec![0.0f64; head_v_dim * head_k_dim];
    let mut fused_out = Vec::with_capacity(2);
    let mut naive_out = Vec::with_capacity(2);
    let mut decay_hist = Vec::with_capacity(2);
    let mut delta_hist = Vec::with_capacity(2);
    for t in 0..2 {
        let (out, decay, delta) = gdn_step_fused_f64(
            &mut state_fused,
            &q_steps[t],
            &k_steps[t],
            &v_steps[t],
            alpha_steps[t],
            beta_steps[t],
            dt_bias,
            a_neg,
            head_k_dim,
            head_v_dim,
        );
        fused_out.push(out);
        decay_hist.push(decay);
        delta_hist.push(delta);

        let out2 = gdn_step_naive_3pass_f64(
            &mut state_naive,
            &q_steps[t],
            &k_steps[t],
            &v_steps[t],
            alpha_steps[t],
            beta_steps[t],
            dt_bias,
            a_neg,
            head_k_dim,
            head_v_dim,
        );
        naive_out.push(out2);
    }

    let quadratic_out =
        gdn_quadratic_reconstruction_f64(&decay_hist, &delta_hist, &k_steps, &q_steps, head_v_dim);

    (fused_out, naive_out, quadratic_out)
}

/// Applies partial RoPE and returns `(before, after)` so a caller can
/// assert the untouched tail (`n_rot..head_dim`) is bit-identical while
/// the rotated head (`0..n_rot`) changed.
pub fn partial_rope_check(
    seed: u64,
    pos: usize,
    n_rot: usize,
    head_dim: usize,
    freq_base: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Xorshift64Star::new(seed);
    let before: Vec<f64> = (0..head_dim)
        .map(|_| rng.next_range_f64(-3.0, 3.0))
        .collect();
    let mut after = before.clone();
    apply_partial_rope_f64(&mut after, pos, n_rot, freq_base);
    (before, after)
}
