//! Dense FP32 GEMV — the LM-head projection kernel (K-12 / M-23).
//!
//! `out[row] = dot(weights[row], input)` over a dense row-major
//! `out_features × in_features` FP32 matrix. The one production caller is the
//! FP32 output projection in `oxibonsai-model`
//! (`model/types/lm_head.rs` → `BonsaiModel::apply_lm_head`), which is the
//! single largest GEMV in the model: `151 669 × 4096` for the shipped 8B and
//! **`248 320 × 5120` for PrismML Bonsai 2 27B**, where it dominates decode.
//!
//! # Why this module exists
//!
//! The body below was previously private to `oxibonsai-model`'s `lm_head.rs`,
//! so it was the only GEMV in the tree that could not be reached through
//! [`KernelDispatcher`](crate::dispatch::KernelDispatcher): the Metal and CUDA
//! tiers had no way to claim the LM head even in principle. Hoisting it here
//! puts it behind the same dispatcher every other format's GEMV goes through
//! (see [`KernelDispatcher::gemv_f32`](crate::dispatch::KernelDispatcher::gemv_f32)).
//!
//! # Bit-exactness contract
//!
//! This is a **verbatim hoist**. The lane count, the accumulator layout, the
//! order of the fold, the remainder handling and the Rayon row threshold are
//! all exactly what `lm_head.rs` did before the move, because the wave's
//! parity gate measures these logits: the hoisted kernel must produce
//! byte-identical logits to the pre-hoist body on a real model. Any
//! reassociation — even one that is "obviously" equivalent in exact
//! arithmetic — changes f32 results and breaks that gate.
//!
//! # Why this is a real SIMD tier without `unsafe`
//!
//! [`dot_f32`] accumulates into `LANES` **independent** accumulators and only
//! folds them at the end. Each `acc[l] += x[l] * y[l]` step is lane-wise, so
//! LLVM lowers the inner loop to NEON `fmla` (`aarch64`, 2 × `float32x4_t`) or
//! AVX2/FMA `vfmadd` (`x86_64`, 1 × `__m256`) without any reassociation
//! permission: the grouping is explicit in the source, not inferred. Targets
//! with neither keep the same code as a plain unrolled scalar loop.
//!
//! The lane grouping means the summation order differs from a naive scalar
//! loop, so results can differ in the last ULP — the same trade every SIMD
//! kernel in this crate already makes, and it is deterministic: the order
//! depends only on `in_features`, never on thread scheduling (Rayon
//! parallelizes *across* rows, and each row is summed by exactly one thread).

// On WASM (`wasm32`), rayon has no threads to hand work off to — see
// `crate::parallel`'s module doc for the same rule applied to the quantized
// GEMV/GEMM entry points. The `out_features >= PAR_MIN_ROWS` branch below
// falls back to the identical sequential loop on that target instead.
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::error::{KernelError, KernelResult};

/// Number of independent accumulator lanes in [`dot_f32`].
///
/// 8 = one AVX2 `__m256` or two NEON `float32x4_t` registers, which keeps both
/// targets' FMA pipelines busy without spilling.
pub const LANES: usize = 8;

/// Minimum output rows before the projection is spread across Rayon workers.
///
/// Mirrors [`crate::parallel`]'s `par_gemv_min_rows` policy: below this,
/// thread hand-off costs more than the projection itself. Every real LM head
/// (vocab ≥ 32 000) is far above it; small test fixtures are below.
///
/// Kept as a hard-coded constant rather than read from
/// [`PlatformProfile`](crate::tuning::PlatformProfile) precisely because the
/// pre-hoist `lm_head.rs` hard-coded it: making the threshold tunable here
/// would change which rows take the Rayon branch and therefore, on a machine
/// whose profile differs, the summation order — the one thing this hoist must
/// not change.
pub const PAR_MIN_ROWS: usize = 256;

/// Dot product of `a[..n]` and `b[..n]` (`n = min(a.len(), b.len())`) with
/// [`LANES`] independent accumulator lanes.
///
/// Truncating to the shorter operand is part of the contract, not an
/// accident: callers pass a full weight row against an input that may be a
/// prefix of a larger scratch buffer.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((slot, &xv), &yv) in acc.iter_mut().zip(x.iter()).zip(y.iter()) {
            *slot += xv * yv;
        }
    }
    let mut sum = 0.0f32;
    for slot in acc {
        sum += slot;
    }
    for (&xv, &yv) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
        sum += xv * yv;
    }
    sum
}

/// `out[..out_features] = weights[out_features × in_features] · input[..in_features]`.
///
/// An **empty** `weights` slice denotes an all-zero projection (the weightless
/// config-only model constructors, M-33): `out` is filled with zeros without
/// ever materializing the `out_features × in_features` zero matrix (4.9 GiB
/// together with the embedding for the 27B config). This is not a special
/// numeric case — it is exactly what multiplying by a zero matrix produces.
///
/// # Errors
///
/// - [`KernelError::NamedBufferTooSmall`] naming `"output"` when
///   `out.len() < out_features`.
/// - [`KernelError::NamedDimensionMismatch`] naming `"input"` when
///   `input.len() < in_features`, or `"weights"` when `weights` is shorter
///   than `out_features * in_features`.
/// - [`KernelError::UnsupportedOperation`] when `out_features * in_features`
///   overflows `usize`.
pub fn gemv_f32(
    weights: &[f32],
    input: &[f32],
    out: &mut [f32],
    out_features: usize,
    in_features: usize,
) -> KernelResult<()> {
    if out.len() < out_features {
        return Err(KernelError::NamedBufferTooSmall {
            name: "output",
            needed: out_features,
            available: out.len(),
        });
    }
    let out = &mut out[..out_features];
    if weights.is_empty() {
        out.fill(0.0);
        return Ok(());
    }
    // Checked before the operand-length guards so a nonsensical shape is
    // reported as such, rather than as "input too short" for an
    // `in_features` no allocation could ever satisfy.
    let needed = out_features.checked_mul(in_features).ok_or_else(|| {
        KernelError::UnsupportedOperation(format!(
            "gemv_f32: {out_features} x {in_features} overflows usize"
        ))
    })?;
    if input.len() < in_features {
        return Err(KernelError::NamedDimensionMismatch {
            name: "input",
            expected: in_features,
            got: input.len(),
        });
    }
    if weights.len() < needed {
        return Err(KernelError::NamedDimensionMismatch {
            name: "weights",
            expected: needed,
            got: weights.len(),
        });
    }
    let input = &input[..in_features];
    // On WASM: no rayon threads available — always take the sequential loop,
    // which is bit-for-bit the same per-row body as the Rayon branch below.
    #[cfg(target_arch = "wasm32")]
    {
        for (row, slot) in out.iter_mut().enumerate() {
            let start = row * in_features;
            *slot = dot_f32(&weights[start..start + in_features], input);
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if out_features >= PAR_MIN_ROWS {
            out.par_iter_mut().enumerate().for_each(|(row, slot)| {
                let start = row * in_features;
                *slot = dot_f32(&weights[start..start + in_features], input);
            });
        } else {
            for (row, slot) in out.iter_mut().enumerate() {
                let start = row * in_features;
                *slot = dot_f32(&weights[start..start + in_features], input);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact pre-hoist `lm_head.rs` body, reproduced here so the hoist's
    /// bit-exactness is asserted against the original code rather than
    /// against a re-derivation of it. Kept byte-for-byte identical to what
    /// `crates/oxibonsai-model/src/model/types/lm_head.rs` carried before
    /// this module existed.
    fn pre_hoist_dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mut acc = [0.0f32; 8];
        let mut a_chunks = a.chunks_exact(8);
        let mut b_chunks = b.chunks_exact(8);
        for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            for ((slot, &xv), &yv) in acc.iter_mut().zip(x.iter()).zip(y.iter()) {
                *slot += xv * yv;
            }
        }
        let mut sum = 0.0f32;
        for slot in acc {
            sum += slot;
        }
        for (&xv, &yv) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
            sum += xv * yv;
        }
        sum
    }

    fn pattern(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 40) as u32 as f32) / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn dot_f32_is_bit_identical_to_the_pre_hoist_body() {
        for len in [0usize, 1, 7, 8, 9, 63, 64, 65, 1024, 5120] {
            let a = pattern(len, 0x1234);
            let b = pattern(len, 0x9876);
            assert_eq!(
                dot_f32(&a, &b).to_bits(),
                pre_hoist_dot(&a, &b).to_bits(),
                "len={len}"
            );
        }
    }

    #[test]
    fn dot_f32_truncates_to_the_shorter_operand() {
        let a = pattern(16, 0xAAAA);
        let b = pattern(16, 0xBBBB);
        assert_eq!(
            dot_f32(&a[..5], &b).to_bits(),
            dot_f32(&a[..5], &b[..5]).to_bits()
        );
    }

    /// Both branches of the row threshold must agree bit-for-bit with the
    /// pre-hoist per-row body, including the Rayon one.
    #[test]
    fn gemv_f32_matches_the_pre_hoist_body_on_both_branches() {
        for (out_features, in_features) in [(4usize, 3usize), (255, 64), (256, 64), (300, 128)] {
            let weights = pattern(out_features * in_features, 0x5EED);
            let input = pattern(in_features, 0xC0FFEE);
            let mut got = vec![0.0f32; out_features];
            gemv_f32(&weights, &input, &mut got, out_features, in_features)
                .expect("well-formed shapes");
            for (row, value) in got.iter().enumerate() {
                let start = row * in_features;
                let want = pre_hoist_dot(&weights[start..start + in_features], &input);
                assert_eq!(
                    value.to_bits(),
                    want.to_bits(),
                    "row {row} of {out_features}x{in_features}"
                );
            }
        }
    }

    #[test]
    fn empty_weights_produce_zero_logits_without_allocating_the_matrix() {
        let mut out = vec![7.0f32; 12];
        gemv_f32(&[], &[1.0; 4], &mut out, 12, 4).expect("zero head");
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn shape_errors_are_named() {
        let weights = pattern(12, 1);
        let mut out = vec![0.0f32; 4];
        // output too small
        assert!(matches!(
            gemv_f32(&weights, &[1.0; 3], &mut out[..3], 4, 3),
            Err(KernelError::NamedBufferTooSmall { name: "output", .. })
        ));
        // input too short
        assert!(matches!(
            gemv_f32(&weights, &[1.0; 2], &mut out, 4, 3),
            Err(KernelError::NamedDimensionMismatch { name: "input", .. })
        ));
        // weights too short
        assert!(matches!(
            gemv_f32(&weights[..5], &[1.0; 3], &mut out, 4, 3),
            Err(KernelError::NamedDimensionMismatch {
                name: "weights",
                ..
            })
        ));
    }

    #[test]
    fn overflowing_shape_is_rejected_not_wrapped() {
        let mut out = vec![0.0f32; 4];
        let err = gemv_f32(&[1.0, 2.0], &[1.0; 2], &mut out, 4, usize::MAX)
            .expect_err("out_features * in_features must not wrap");
        assert!(matches!(err, KernelError::UnsupportedOperation(_)), "{err}");
    }
}
