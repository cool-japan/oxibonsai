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
//! all exactly what `lm_head.rs` did before the move, because the parity gate
//! measures these logits: the hoisted kernel must produce byte-identical
//! logits to the pre-hoist body on a real model. Any reassociation — even one
//! that is "obviously" equivalent in exact arithmetic — changes f32 results
//! and breaks that gate.
//!
//! # One accumulation routine for GEMV and GEMM
//!
//! `dot_tile` computes an `MR × NR` block of dot products in a single pass
//! over the operands, and [`dot_f32`] is its `1 × 1` instance.
//! [`gemm_f32`](crate::gemm_f32::gemm_f32) runs the same routine over register
//! tiles, so every element it produces goes through the identical schedule —
//! [`LANES`] independent lane accumulators over the full-width chunks in
//! order, a lane fold `0..LANES`, then the remainder elements one at a time —
//! and a GEMM row is bit-identical to a GEMV of that row. A tile only changes
//! which *independent* outputs share a loop nest (several outputs interleaved
//! give the add latency of one output something to hide behind, and one
//! operand load feeds several outputs); it never shares, splits or reorders
//! the additions *within* one output.
//!
//! # SIMD: explicit on `aarch64` and `x86_64`, portable elsewhere
//!
//! The lane accumulation is `acc[l] += x[l] * y[l]` over [`LANES`]
//! **independent** accumulators, folded only at the end, so it maps one to one
//! onto vector registers: two 128-bit vectors per chunk (`float32x4_t` on
//! `aarch64`, `__m128` on `x86_64`; both are baseline features of those
//! targets, so no runtime detection is involved). It is written with those
//! intrinsics rather than left to the auto-vectorizer because the
//! auto-vectorized loop proved unreliable: the same source measured about 2.6x
//! slower compiled into this crate as a library under the workspace's fat-LTO
//! release profile than compiled into a binary crate (its hot loops carry
//! extra lane moves), while the intrinsic form runs at the same speed in both.
//! Other targets run a portable array loop with the identical arithmetic, and
//! the tests hold every vector form to it bit for bit.
//!
//! Each step is one multiply and one add, each rounded on its own (`fmul` and
//! `fadd`, `mulps` and `addps`) — deliberately not a fused multiply-add, which
//! would change the last bit and make the result depend on the target.
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
/// 8 = two 128-bit vector registers (NEON `float32x4_t`, SSE `__m128`): two
/// independent accumulator chains per output, few enough that a register tile
/// of several outputs keeps every accumulator and operand in registers.
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
///
/// This is `dot_tile` at `1 × 1`, so it is the reference schedule every
/// tiled caller reproduces bit for bit.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let [[sum]] = dot_tile::<1, 1>([a], [b]);
    sum
}

/// The dot products of every `a_rows[r]` with every `w_rows[s]`, computed in a
/// single pass: `result[r][s]` is [`dot_f32`]`(a_rows[r], w_rows[s])` bit for
/// bit, for every `MR × NR`.
///
/// All `MR + NR` operands are truncated to the shortest one's length (the rule
/// [`dot_f32`] applies to its two operands), so no operand data can make this
/// panic; `MR == 0` or `NR == 0` does not compile.
///
/// The `MR * NR` outputs keep `MR * NR * LANES` accumulator lanes live at
/// once. One output alone has only two independent vector accumulator chains,
/// so its adds wait on each other; several outputs interleaved in one loop
/// keep the floating-point pipes full, and each operand chunk loaded serves
/// `NR` (resp. `MR`) outputs instead of one. Each output's own arithmetic is
/// untouched: a plain product added to its lane (never a fused multiply-add),
/// the lanes folded in order `0..LANES` starting from `0.0`, then the
/// remainder elements added one at a time.
#[inline(always)]
pub(crate) fn dot_tile<const MR: usize, const NR: usize>(
    a_rows: [&[f32]; MR],
    w_rows: [&[f32]; NR],
) -> [[f32; NR]; MR] {
    const {
        assert!(
            MR > 0 && NR > 0,
            "dot_tile needs at least one row on each side"
        )
    };
    let len = a_rows
        .iter()
        .chain(w_rows.iter())
        .map(|row| row.len())
        .min()
        .unwrap_or(0);
    let chunk_count = len / LANES;
    let a_split: [(&[[f32; LANES]], &[f32]); MR] =
        std::array::from_fn(|r| a_rows[r][..len].as_chunks::<LANES>());
    let w_split: [(&[[f32; LANES]], &[f32]); NR] =
        std::array::from_fn(|s| w_rows[s][..len].as_chunks::<LANES>());
    // Every operand has exactly `chunk_count` full chunks; re-slicing to that
    // common length lets the loop below run without per-operand bounds checks.
    let a_chunks: [&[[f32; LANES]]; MR] = std::array::from_fn(|r| &a_split[r].0[..chunk_count]);
    let w_chunks: [&[[f32; LANES]]; NR] = std::array::from_fn(|s| &w_split[s].0[..chunk_count]);

    let acc = accumulate_lanes::<MR, NR>(&a_chunks, &w_chunks, chunk_count);

    let mut out = [[0.0f32; NR]; MR];
    for (r, out_row) in out.iter_mut().enumerate() {
        for (s, slot_out) in out_row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for slot in acc[r][s] {
                sum += slot;
            }
            for (&xv, &yv) in a_split[r].1.iter().zip(w_split[s].1.iter()) {
                sum += xv * yv;
            }
            *slot_out = sum;
        }
    }
    out
}

/// The lane-accumulation core of [`dot_tile`], portable form: for every output
/// `(r, s)` and lane `l`, `acc[r][s][l]` is `0.0` with `a[r][c][l] * w[s][c][l]`
/// added for `c = 0, 1, …` in order — the product and the add each rounded on
/// their own, never fused.
///
/// This is the reference the vector forms below reproduce bit for bit. It is
/// what runs on targets without a vector form here, and it is kept compiled in
/// test builds everywhere so the vector forms are held to it.
///
/// The operand chunks are copied out by value (`ys`, `x`) rather than
/// borrowed: that keeps the loads independent of the accumulator stores, which
/// is what lets LLVM hold the accumulators in registers.
#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
#[inline(always)]
fn accumulate_lanes_portable<const MR: usize, const NR: usize>(
    a_chunks: &[&[[f32; LANES]]; MR],
    w_chunks: &[&[[f32; LANES]]; NR],
    chunk_count: usize,
) -> [[[f32; LANES]; NR]; MR] {
    let mut acc = [[[0.0f32; LANES]; NR]; MR];
    for c in 0..chunk_count {
        let ys: [[f32; LANES]; NR] = std::array::from_fn(|s| w_chunks[s][c]);
        for (a_row, acc_row) in a_chunks.iter().zip(acc.iter_mut()) {
            let x = a_row[c];
            for (y, lanes) in ys.iter().zip(acc_row.iter_mut()) {
                for ((slot, &xv), &yv) in lanes.iter_mut().zip(x.iter()).zip(y.iter()) {
                    *slot += xv * yv;
                }
            }
        }
    }
    acc
}

/// [`accumulate_lanes_portable`] on a target with no vector form here.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn accumulate_lanes<const MR: usize, const NR: usize>(
    a_chunks: &[&[[f32; LANES]]; MR],
    w_chunks: &[&[[f32; LANES]]; NR],
    chunk_count: usize,
) -> [[[f32; LANES]; NR]; MR] {
    accumulate_lanes_portable::<MR, NR>(a_chunks, w_chunks, chunk_count)
}

/// [`accumulate_lanes_portable`] written with NEON intrinsics: each eight-lane
/// chunk is two `float32x4_t` vectors, and every step is one `fmul` and one
/// `fadd` (`vmulq_f32`, `vaddq_f32`) — the same two roundings as the scalar
/// form, in the same order, so the result is identical.
///
/// Written out rather than left to the auto-vectorizer because the portable
/// loop proved unreliable: the same source ran about 2.6x slower compiled into
/// this crate as a library under the workspace's fat-LTO release profile (how
/// it is built into the binaries) than compiled into a binary crate, and its
/// hot loops carried extra per-lane moves. This form runs at the same speed in
/// both.
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)] // the value-only intrinsics are safe on newer toolchains
#[inline(always)]
fn accumulate_lanes<const MR: usize, const NR: usize>(
    a_chunks: &[&[[f32; LANES]]; MR],
    w_chunks: &[&[[f32; LANES]]; NR],
    chunk_count: usize,
) -> [[[f32; LANES]; NR]; MR] {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32,
    };

    /// Lanes `0..4` and `4..8` of one chunk.
    #[inline(always)]
    fn load(chunk: &[f32; LANES]) -> [float32x4_t; 2] {
        let base = chunk.as_ptr();
        // SAFETY: `chunk` is eight contiguous `f32`, so the two four-lane loads
        // (at element offsets 0 and 4) stay inside it.
        unsafe { [vld1q_f32(base), vld1q_f32(base.add(4))] }
    }

    // SAFETY: NEON is part of the aarch64 baseline, so every intrinsic used
    // here is available; the only pointer accesses are the in-bounds loads
    // above and the in-bounds stores below.
    unsafe {
        let mut acc = [[[vdupq_n_f32(0.0); 2]; NR]; MR];
        for c in 0..chunk_count {
            let ys: [[float32x4_t; 2]; NR] = std::array::from_fn(|s| load(&w_chunks[s][c]));
            for (a_row, acc_row) in a_chunks.iter().zip(acc.iter_mut()) {
                let x = load(&a_row[c]);
                for (y, lanes) in ys.iter().zip(acc_row.iter_mut()) {
                    lanes[0] = vaddq_f32(lanes[0], vmulq_f32(x[0], y[0]));
                    lanes[1] = vaddq_f32(lanes[1], vmulq_f32(x[1], y[1]));
                }
            }
        }
        let mut out = [[[0.0f32; LANES]; NR]; MR];
        for (out_row, acc_row) in out.iter_mut().zip(acc.iter()) {
            for (lanes_out, lanes) in out_row.iter_mut().zip(acc_row.iter()) {
                let base = lanes_out.as_mut_ptr();
                vst1q_f32(base, lanes[0]);
                vst1q_f32(base.add(4), lanes[1]);
            }
        }
        out
    }
}

/// [`accumulate_lanes_portable`] written with SSE intrinsics (part of the
/// `x86_64` baseline): each eight-lane chunk is two `__m128` vectors, and every
/// step is one `mulps` and one `addps` — never a fused multiply-add — so the
/// result is identical to the scalar form. See the `aarch64` form for why it
/// is written out.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)] // the value-only intrinsics are safe on newer toolchains
#[inline(always)]
fn accumulate_lanes<const MR: usize, const NR: usize>(
    a_chunks: &[&[[f32; LANES]]; MR],
    w_chunks: &[&[[f32; LANES]]; NR],
    chunk_count: usize,
) -> [[[f32; LANES]; NR]; MR] {
    use std::arch::x86_64::{
        __m128, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps, _mm_storeu_ps,
    };

    /// Lanes `0..4` and `4..8` of one chunk.
    #[inline(always)]
    fn load(chunk: &[f32; LANES]) -> [__m128; 2] {
        let base = chunk.as_ptr();
        // SAFETY: `chunk` is eight contiguous `f32`, so the two four-lane
        // unaligned loads (at element offsets 0 and 4) stay inside it.
        unsafe { [_mm_loadu_ps(base), _mm_loadu_ps(base.add(4))] }
    }

    // SAFETY: SSE is part of the x86_64 baseline, so every intrinsic used here
    // is available; the only pointer accesses are the in-bounds loads above and
    // the in-bounds stores below.
    unsafe {
        let mut acc = [[[_mm_setzero_ps(); 2]; NR]; MR];
        for c in 0..chunk_count {
            let ys: [[__m128; 2]; NR] = std::array::from_fn(|s| load(&w_chunks[s][c]));
            for (a_row, acc_row) in a_chunks.iter().zip(acc.iter_mut()) {
                let x = load(&a_row[c]);
                for (y, lanes) in ys.iter().zip(acc_row.iter_mut()) {
                    lanes[0] = _mm_add_ps(lanes[0], _mm_mul_ps(x[0], y[0]));
                    lanes[1] = _mm_add_ps(lanes[1], _mm_mul_ps(x[1], y[1]));
                }
            }
        }
        let mut out = [[[0.0f32; LANES]; NR]; MR];
        for (out_row, acc_row) in out.iter_mut().zip(acc.iter()) {
            for (lanes_out, lanes) in out_row.iter_mut().zip(acc_row.iter()) {
                let base = lanes_out.as_mut_ptr();
                _mm_storeu_ps(base, lanes[0]);
                _mm_storeu_ps(base.add(4), lanes[1]);
            }
        }
        out
    }
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

    /// The pre-hoist `lm_head.rs` body, reproduced here so the hoist's
    /// bit-exactness is asserted against the original code rather than
    /// against a re-derivation of it. The arithmetic is kept identical to
    /// what `crates/oxibonsai-model/src/model/types/lm_head.rs` carried before
    /// this module existed: eight independent accumulators fed one 8-wide
    /// chunk at a time, the accumulators summed in index order, then the
    /// sub-chunk tail added element by element. Only the chunking is spelled
    /// differently (`as_chunks::<8>()` instead of `chunks_exact(8)`, per
    /// `clippy::chunks_exact_to_as_chunks`): it yields the same chunks in the
    /// same order and the same remainder slice.
    fn pre_hoist_dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mut acc = [0.0f32; 8];
        let (a_chunks, a_remainder) = a.as_chunks::<8>();
        let (b_chunks, b_remainder) = b.as_chunks::<8>();
        for (x, y) in a_chunks.iter().zip(b_chunks.iter()) {
            for ((slot, &xv), &yv) in acc.iter_mut().zip(x.iter()).zip(y.iter()) {
                *slot += xv * yv;
            }
        }
        let mut sum = 0.0f32;
        for slot in acc {
            sum += slot;
        }
        for (&xv, &yv) in a_remainder.iter().zip(b_remainder.iter()) {
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

    /// Every tile shape the GEMM instantiates, plus a few it does not, must
    /// reproduce the pre-hoist body bit for bit on every element, at lengths
    /// on both sides of every lane / chunk boundary (a length that is not a
    /// multiple of `LANES` exercises the remainder loop).
    #[test]
    fn dot_tile_is_bit_identical_to_the_pre_hoist_body_for_every_shape() {
        fn check<const MR: usize, const NR: usize>(len: usize) {
            let a: Vec<Vec<f32>> = (0..MR).map(|r| pattern(len, 0x100 + r as u64)).collect();
            let w: Vec<Vec<f32>> = (0..NR).map(|s| pattern(len, 0x900 + s as u64)).collect();
            let a_rows: [&[f32]; MR] = std::array::from_fn(|r| a[r].as_slice());
            let w_rows: [&[f32]; NR] = std::array::from_fn(|s| w[s].as_slice());
            let got = dot_tile::<MR, NR>(a_rows, w_rows);
            for (r, got_row) in got.iter().enumerate() {
                for (s, value) in got_row.iter().enumerate() {
                    assert_eq!(
                        value.to_bits(),
                        pre_hoist_dot(&a[r], &w[s]).to_bits(),
                        "{MR}x{NR} tile, len={len}, element ({r}, {s})"
                    );
                    assert_eq!(
                        value.to_bits(),
                        dot_f32(&a[r], &w[s]).to_bits(),
                        "{MR}x{NR} tile vs dot_f32, len={len}, element ({r}, {s})"
                    );
                }
            }
        }
        for len in [
            0usize, 1, 7, 8, 9, 13, 15, 16, 17, 63, 64, 65, 129, 1024, 1155, 5120,
        ] {
            check::<1, 1>(len);
            check::<1, 2>(len);
            check::<2, 1>(len);
            check::<1, 8>(len);
            check::<2, 2>(len);
            check::<2, 4>(len);
            check::<3, 2>(len);
            check::<3, 3>(len);
            check::<4, 2>(len);
            check::<4, 4>(len);
        }
    }

    /// The shortest operand sets the length for the whole tile, exactly as it
    /// does for `dot_f32`'s two operands.
    #[test]
    fn dot_tile_truncates_every_operand_to_the_shortest() {
        let a0 = pattern(40, 1);
        let a1 = pattern(23, 2);
        let w0 = pattern(31, 3);
        let w1 = pattern(57, 4);
        let got = dot_tile::<2, 2>([&a0, &a1], [&w0, &w1]);
        for (r, a_row) in [&a0, &a1].into_iter().enumerate() {
            for (s, w_row) in [&w0, &w1].into_iter().enumerate() {
                assert_eq!(
                    got[r][s].to_bits(),
                    pre_hoist_dot(&a_row[..23], &w_row[..23]).to_bits(),
                    "element ({r}, {s})"
                );
            }
        }
        // A tile with an empty operand is the all-zero dot (`0.0` folded from
        // untouched lanes), never a panic.
        let empty = dot_tile::<1, 2>([&a0], [&w0[..0], &w1]);
        assert_eq!(empty, [[0.0, 0.0]]);
    }

    /// NaN and infinity travel through the tile exactly as through the
    /// scalar body: a NaN operand poisons every output that reads it and no
    /// other, and an infinity that meets no opposite infinity stays infinite.
    #[test]
    fn dot_tile_propagates_non_finite_values_like_the_scalar_body() {
        let mut a0 = pattern(20, 5);
        let a1 = pattern(20, 6);
        let mut w0 = pattern(20, 7);
        let w1 = pattern(20, 8);
        a0[3] = f32::NAN;
        w0[17] = f32::INFINITY;
        let got = dot_tile::<2, 2>([&a0, &a1], [&w0, &w1]);
        assert!(
            got[0][0].is_nan() && got[0][1].is_nan(),
            "row 0 reads a NaN"
        );
        assert!(got[1][0].is_infinite() || got[1][0].is_nan());
        assert_eq!(
            got[1][0].to_bits(),
            pre_hoist_dot(&a1, &w0).to_bits(),
            "the infinity flows through identically"
        );
        assert_eq!(got[1][1].to_bits(), pre_hoist_dot(&a1, &w1).to_bits());
    }

    /// The vector form of the lane accumulation (NEON on `aarch64`, SSE on
    /// `x86_64`; elsewhere the portable loop itself) is bit-identical to the
    /// portable loop for every tile shape and for chunk counts on both sides
    /// of every unroll boundary — on ordinary data and on the values where a
    /// vector unit could differ from scalar code: subnormals (flush-to-zero),
    /// signed zeros, overflow to infinity, and NaN.
    #[test]
    fn the_vector_lane_accumulation_matches_the_portable_loop() {
        fn chunks(count: usize, seed: u64, special: bool) -> Vec<[f32; LANES]> {
            let mut flat = pattern(count * LANES, seed);
            if special {
                let specials = [
                    f32::MIN_POSITIVE / 4.0,
                    -0.0,
                    0.0,
                    3.0e38,
                    -3.0e38,
                    1.0e-30,
                    f32::MIN_POSITIVE,
                ];
                for (idx, slot) in flat.iter_mut().enumerate().step_by(5) {
                    *slot = specials[(idx / 5) % specials.len()];
                }
                if let Some(slot) = flat.get_mut(3) {
                    *slot = f32::NAN;
                }
            }
            flat.as_chunks::<LANES>().0.to_vec()
        }
        fn check<const MR: usize, const NR: usize>(count: usize, special: bool) {
            let a: Vec<Vec<[f32; LANES]>> = (0..MR)
                .map(|r| chunks(count, 0x40 + r as u64, special))
                .collect();
            let w: Vec<Vec<[f32; LANES]>> = (0..NR)
                .map(|s| chunks(count, 0x80 + s as u64, special))
                .collect();
            let a_refs: [&[[f32; LANES]]; MR] = std::array::from_fn(|r| a[r].as_slice());
            let w_refs: [&[[f32; LANES]]; NR] = std::array::from_fn(|s| w[s].as_slice());
            let vector = accumulate_lanes::<MR, NR>(&a_refs, &w_refs, count);
            let portable = accumulate_lanes_portable::<MR, NR>(&a_refs, &w_refs, count);
            for r in 0..MR {
                for s in 0..NR {
                    for l in 0..LANES {
                        let (v, p) = (vector[r][s][l], portable[r][s][l]);
                        assert!(
                            v.to_bits() == p.to_bits() || (v.is_nan() && p.is_nan()),
                            "{MR}x{NR}, {count} chunks, special={special}, ({r}, {s}) lane {l}: \
                             vector {v} ({:#010x}), portable {p} ({:#010x})",
                            v.to_bits(),
                            p.to_bits()
                        );
                    }
                }
            }
        }
        for special in [false, true] {
            for count in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 33, 128] {
                check::<1, 1>(count, special);
                check::<1, 2>(count, special);
                check::<2, 1>(count, special);
                check::<4, 2>(count, special);
                check::<2, 4>(count, special);
                check::<1, 8>(count, special);
                check::<3, 3>(count, special);
                check::<4, 4>(count, special);
            }
        }
    }

    /// Both branches of the row threshold must agree bit-for-bit with the
    /// pre-hoist per-row body, including the Rayon one.
    #[test]
    fn gemv_f32_matches_the_pre_hoist_body_on_both_branches() {
        // The last two are LM-head widths (8B and 27B hidden sizes), with
        // enough rows to take the Rayon branch.
        for (out_features, in_features) in [
            (4usize, 3usize),
            (255, 64),
            (256, 64),
            (300, 128),
            (256, 4096),
            (300, 5120),
        ] {
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
