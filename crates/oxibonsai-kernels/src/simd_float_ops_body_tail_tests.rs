//! Body/tail bit-identity (K-M3) for the `simd_float_ops` kernels whose tail
//! is plain scalar code rather than a padded run of a vector core:
//! `rms_norm_simd` and `rope_apply_simd`. Such a tail must evaluate the
//! body's exact operation sequence — the same products in the same order,
//! fused where the body fuses — or the same element rounds differently
//! depending on whether it lands in the vector body or the tail.
//!
//! Architecture-neutral on purpose: the property holds on every tier (8-lane
//! AVX2 bodies, 4-lane NEON bodies, and the scalar fallbacks' single loop),
//! so these run everywhere, the aarch64 `rms_norm_neon` tail included.
//! Re-attached with `#[cfg(test)] #[path]`, a child module of
//! `simd_float_ops` like its sibling test files.

use super::*;

/// `(input value, weight value, eps)` probes for `rms_norm_simd`. Uniform
/// input makes every element share one `inv_rms`, so equal weights must give
/// equal outputs wherever they sit.
const RMS_NORM_PROBES: [(f32, f32, f32); 8] = [
    (0.37, 0.913, 1e-5),
    (-1.3, 1.7, 1e-6),
    (2.75, -0.45, 1e-5),
    (1.0e-3, 3.1, 1e-6),
    (17.0, 0.0125, 1e-5),
    (0.1, -2.6, 0.0),
    (-0.731, 1.234_5, 1e-6),
    (5.5, 0.071, 1e-5),
];

/// Weights the non-probe positions hold. Weight never enters the shared
/// `inv_rms` (only the input does), so they cannot disturb the probe.
const RMS_NORM_NEIGHBOUR_WEIGHTS: [f32; 3] = [0.5, -3.75, 11.0];

/// `rms_norm_simd(input, weight)[pos]`, with every input element `x` and the
/// weight `w` at `pos` (the other weights `neighbour`).
fn rms_norm_at(len: usize, pos: usize, (x, w, eps): (f32, f32, f32), neighbour: f32) -> f32 {
    let input = vec![x; len];
    let mut weight = vec![neighbour; len];
    weight[pos] = w;
    let mut output = vec![f32::NAN; len];
    rms_norm_simd(&input, &weight, &mut output, eps).expect("matching lengths");
    output[pos]
}

/// For each length 1..=24 (zero to three AVX2 bodies, zero to six NEON
/// bodies, every tail size) the probe weight must produce one output bit
/// pattern at every position — body, tail, or a length that is all tail —
/// with uniform weights (`output[0]` vs `output[len - 1]`) and with the
/// probe alone among other weights. Before the fix both the AVX2 and NEON
/// tails computed `(weight * input) * inv_rms` against the body's
/// `weight * (input * inv_rms)`.
#[test]
fn rms_norm_body_and_tail_agree_bitwise_for_uniform_input() {
    for probe in RMS_NORM_PROBES {
        let (x, w, eps) = probe;
        for len in 1..=24usize {
            let input = vec![x; len];
            let weight = vec![w; len];
            let mut output = vec![f32::NAN; len];
            rms_norm_simd(&input, &weight, &mut output, eps).expect("matching lengths");
            assert_eq!(
                output[0].to_bits(),
                output[len - 1].to_bits(),
                "rms_norm len={len} x={x} w={w} eps={eps}: output[0] = {:e} (body) but \
                 output[{}] = {:e} (tail)",
                output[0],
                len - 1,
                output[len - 1]
            );

            let reference = rms_norm_at(len, 0, probe, RMS_NORM_NEIGHBOUR_WEIGHTS[0]);
            for neighbour in RMS_NORM_NEIGHBOUR_WEIGHTS {
                for pos in 0..len {
                    let got = rms_norm_at(len, pos, probe, neighbour);
                    assert_eq!(
                        got.to_bits(),
                        reference.to_bits(),
                        "rms_norm len={len} pos={pos} x={x} w={w} eps={eps} (neighbour weights \
                         {neighbour}) = {got:e}, expected {reference:e} as at pos=0"
                    );
                }
            }
        }
    }
}

/// `(x0, x1, angle)` probes for `rope_apply_simd`: one rotation pair, its
/// `cos`/`sin` taken from the angle.
const ROPE_PROBES: [(f32, f32, f32); 6] = [
    (0.37, -0.81, 0.3),
    (-1.3, 1.9, 1.1),
    (2.75, 0.05, 2.7),
    (0.913, 0.913, -0.9),
    (-17.5, 4.25, 0.785),
    (1.0e-3, -6.0, 3.0),
];

/// What the non-probe pairs hold. NaN and `±inf` must not leak into the
/// probe's pair: the rotation is strictly element-wise.
const ROPE_NEIGHBOURS: [f32; 4] = [0.6, -2.5, f32::NAN, f32::INFINITY];

/// The same rotation pair `(x0, x1)` with the same `cos`/`sin` must produce
/// the same two output bit patterns at every position of every `half_dim`
/// in 1..=24 — the AVX2 body is a multiple-of-8 prefix, NEON's a
/// multiple-of-4 prefix — whatever its neighbours hold. Before the fix the
/// AVX2 tail rounded `x0*cos - x1*sin` and `x0*sin + x1*cos` with each
/// product separately, against the body's fused multiply-adds.
#[test]
fn rope_body_and_tail_agree_bitwise_regardless_of_offset_and_length() {
    for (x0, x1, angle) in ROPE_PROBES {
        let (c, s) = (angle.cos(), angle.sin());
        let mut reference: Option<(u32, u32)> = None;
        for neighbour in ROPE_NEIGHBOURS {
            for half_dim in 1..=24usize {
                for pos in 0..half_dim {
                    let mut input = vec![neighbour; 2 * half_dim];
                    let mut cos_table = vec![neighbour; half_dim];
                    let mut sin_table = vec![neighbour; half_dim];
                    input[pos] = x0;
                    input[half_dim + pos] = x1;
                    cos_table[pos] = c;
                    sin_table[pos] = s;
                    let mut output = vec![f32::NAN; 2 * half_dim];
                    rope_apply_simd(&input, &mut output, &cos_table, &sin_table)
                        .expect("matching lengths");
                    let got = (output[pos].to_bits(), output[half_dim + pos].to_bits());
                    match reference {
                        None => reference = Some(got),
                        Some(r) => assert_eq!(
                            got, r,
                            "rope(x0={x0}, x1={x1}, angle={angle}) at half_dim={half_dim} \
                             pos={pos} (neighbours {neighbour}) = ({:#010x}, {:#010x}), \
                             expected bit-identical to the first observation \
                             ({:#010x}, {:#010x})",
                            got.0, got.1, r.0, r.1
                        ),
                    }
                }
            }
        }
    }
}
