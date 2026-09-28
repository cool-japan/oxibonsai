//! Per-block int8 quantization of the **activation** side of a ternary /
//! 1-bit / 2-bit GEMV or GEMM (K-14, step 1).
//!
//! The INT8 dot-product tier ([`crate::simd_dot_int8`],
//! [`crate::dispatch_int8`]) needs both operands as `i8`. The weight side is
//! already integral — a `TQ2_0_g128` code is one of `{-1, 0, +1}`, a `PQ2_0`
//! code one of `{-1, 0, +1, +2}`, a `Q1_0_g128` bit one of `{-1, +1}` — so
//! only the activation has to be quantized, and it is quantized **once per
//! call**, amortised across all `n_rows` weight rows. That amortisation is
//! the whole reason the tier can pay for a lossy activation at all: one
//! pass over `k` floats buys `n_rows * k` int8 MACs.
//!
//! ## Contract
//!
//! For each `qk`-wide block `b` of one activation row:
//!
//! ```text
//! s_b     = max_j |x_j| / 127                    (0 if the block is all-zero)
//! code_j  = clamp(round(x_j / s_b), -127, +127)  (0 if s_b == 0)
//! sum_b   = Σ_j code_j                           (exact, in i32)
//! ```
//!
//! and the GEMV reconstructs `d_b * s_b * (Σ_j w_j * code_j)`, the integer
//! sum being exact: `|w| <= 2`, `|code| <= 127`, `qk <= 128`, so
//! `|Σ| <= 2 * 127 * 128 = 32 512`, three orders of magnitude inside `i32`.
//!
//! `127` (not `128`) is the positive limit so the representable range is
//! symmetric and `-x` quantizes to `-code(x)` exactly — a zero-mean
//! activation must not acquire a systematic sign bias.
//!
//! ## `sum_b`, and why it is stored
//!
//! AVX-512 VNNI's `_mm512_dpbusd_epi32` is **u8 × i8**, not i8 × i8: one
//! operand must be unsigned. The weight side is the one that can be made
//! unsigned, by *biasing* it — `biased = value + 1`, which lands every
//! format's codes in `0..=3` (see [`crate::simd_dot_int8`]'s two tables) —
//! and the bias is then removed with one subtraction per block:
//!
//! ```text
//! Σ_j value_j * code_j = Σ_j (value_j + 1) * code_j − Σ_j code_j
//!                      = dpbusd(biased, code)      − sum_b
//! ```
//!
//! `sum_b` depends only on the activation, so computing it here makes the
//! correction free in the inner loop. Every tier — scalar, NEON and VNNI —
//! uses this same biased identity, so the one algebraic step the x86 path
//! depends on is exercised by the scalar tests that run on any host.
//!
//! ## Layouts
//!
//! A SIMD 2-bit decode does not produce weights in `j` order: masking a
//! 16-byte `qs` chunk at shift `2s` yields the 16 weights `{4i + s}`, i.e.
//! **stride 4**. Rather than permute 16 decoded weight lanes per chunk
//! inside the hot loop (`n_rows` times), the activation is permuted once,
//! here, into the same order ([`Int8Layout::Stride4`]). The 1-bit format's
//! bit-expansion is naturally sequential, so it uses
//! [`Int8Layout::Sequential`] and no permutation at all. The two layouts are
//! a fixed bijection of each other, so a dot product taken consistently on
//! either side gives the identical integer.

use crate::error::{KernelError, KernelResult};

/// Largest activation magnitude an `i8` code may take.
///
/// `127`, not `128` — see the module doc.
pub const INT8_ACTIVATION_MAX: f32 = 127.0;

/// How a quantized activation row is ordered.
///
/// Chosen by the GEMV/GEMM entry point that owns the quantization, never by
/// the caller; the kernels re-check it, so a mismatch is an error and never
/// a silently wrong dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8Layout {
    /// Weight order `j` (the 1-bit family's bit-expansion order).
    Sequential,
    /// Stride-4 within each 64-weight group: position `g*64 + s*16 + i`
    /// holds activation element `g*64 + 4*i + s` (the 2-bit family's SIMD
    /// decode order).
    Stride4,
}

impl Int8Layout {
    /// Permuted position of activation element `j` inside a row of `k`
    /// elements.
    ///
    /// The single definition of [`Int8Layout::Stride4`]; every kernel and
    /// every test derives its indexing from this one function.
    #[must_use]
    #[inline]
    pub const fn position_of(self, j: usize) -> usize {
        match self {
            Self::Sequential => j,
            Self::Stride4 => {
                let group = j / 64;
                let within = j % 64;
                // within = 4*i + s  ->  s*16 + i
                let i = within / 4;
                let s = within % 4;
                group * 64 + s * 16 + i
            }
        }
    }
}

/// One activation matrix (`rows x k`), quantized to int8 with a per-block
/// scale.
///
/// Owns its buffers so a GEMV can quantize once and hand the same value to
/// every weight row; `rows > 1` is the prefill/GEMM case, where the whole
/// `m x k` activation is quantized once for the entire weight matrix.
#[derive(Debug, Clone)]
pub struct Int8Activation {
    /// `rows * k` int8 codes, each row in `layout` order.
    codes: Vec<i8>,
    /// `rows * blocks_per_row` block scales.
    scales: Vec<f32>,
    /// `rows * blocks_per_row` exact block code sums (the VNNI bias
    /// correction; see the module doc).
    sums: Vec<i32>,
    rows: usize,
    k: usize,
    qk: usize,
    layout: Int8Layout,
}

impl Int8Activation {
    /// Quantize `input` (`rows x k`, row-major FP32) into int8 blocks of
    /// `qk` elements.
    ///
    /// # Errors
    ///
    /// - [`KernelError::NotBlockAligned`] if `k` is not a multiple of `qk`,
    ///   or `qk` is zero or not a multiple of 64 (both layouts are defined
    ///   in 64-element groups).
    /// - [`KernelError::NamedDimensionMismatch`] if `input` is shorter than
    ///   `rows * k`.
    pub fn quantize(
        input: &[f32],
        rows: usize,
        k: usize,
        qk: usize,
        layout: Int8Layout,
    ) -> KernelResult<Self> {
        if qk == 0 || !qk.is_multiple_of(64) || !k.is_multiple_of(qk) {
            return Err(KernelError::NotBlockAligned {
                count: k,
                block_size: qk,
            });
        }
        if input.len() < rows * k {
            return Err(KernelError::dimension_mismatch(
                "input",
                rows * k,
                input.len(),
            ));
        }

        let blocks_per_row = k / qk;
        let mut codes = vec![0i8; rows * k];
        let mut scales = vec![0.0f32; rows * blocks_per_row];
        let mut sums = vec![0i32; rows * blocks_per_row];

        for r in 0..rows {
            let row = &input[r * k..(r + 1) * k];
            for b in 0..blocks_per_row {
                let block = &row[b * qk..(b + 1) * qk];
                // `f32::max` follows IEEE 754's `maxNum` and silently
                // ignores a NaN operand, so folding `amax` with it would make
                // a NaN activation element quantize to code `0` whenever
                // another element in the same block was large enough to give
                // a finite, nonzero `scale` — the block would reconstruct as
                // if that NaN had never been there, the opposite of the f32
                // reference path's NaN contagion (any NaN weight-activation
                // product poisons the whole sum). A block that holds *any*
                // NaN gets a NaN `scale` instead, so it propagates through
                // the row kernels' `scale * acc as f32` exactly like the f32
                // path would — silent divergence here would otherwise hide a
                // real upstream numerical fault whenever the INT8 tier is on.
                let mut amax = 0.0f32;
                let mut has_nan = false;
                for &v in block {
                    if v.is_nan() {
                        has_nan = true;
                    } else {
                        amax = amax.max(v.abs());
                    }
                }
                let scale = if has_nan {
                    f32::NAN
                } else {
                    amax / INT8_ACTIVATION_MAX
                };
                scales[r * blocks_per_row + b] = scale;
                if scale <= 0.0 || !scale.is_finite() {
                    // All-zero block, a NaN-holding block (scale forced to
                    // NaN above), or one whose `amax` is otherwise not
                    // finite: every code stays 0, and so does `sum`, which
                    // the reconstruction multiplies by a non-finite scale
                    // anyway (propagating NaN, not silently dropping it).
                    continue;
                }
                let inv = 1.0f32 / scale;
                let mut sum = 0i32;
                for (j, &x) in block.iter().enumerate() {
                    let q = (x * inv)
                        .round()
                        .clamp(-INT8_ACTIVATION_MAX, INT8_ACTIVATION_MAX)
                        as i8;
                    sum += q as i32;
                    let pos = b * qk + layout.position_of(j);
                    codes[r * k + pos] = q;
                }
                sums[r * blocks_per_row + b] = sum;
            }
        }

        Ok(Self {
            codes,
            scales,
            sums,
            rows,
            k,
            qk,
            layout,
        })
    }

    /// Number of activation rows.
    #[must_use]
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Inner dimension.
    #[must_use]
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Elements per quantization block.
    #[must_use]
    #[inline]
    pub fn qk(&self) -> usize {
        self.qk
    }

    /// Quantization blocks per row.
    #[must_use]
    #[inline]
    pub fn blocks_per_row(&self) -> usize {
        self.k / self.qk
    }

    /// The order [`Self::codes_row`] is stored in.
    #[must_use]
    #[inline]
    pub fn layout(&self) -> Int8Layout {
        self.layout
    }

    /// Row `r`'s `k` int8 codes, in [`Self::layout`] order.
    ///
    /// # Panics
    ///
    /// Never for `r < rows()`; the slice bounds are established by
    /// [`Self::quantize`].
    #[must_use]
    #[inline]
    pub fn codes_row(&self, r: usize) -> &[i8] {
        &self.codes[r * self.k..(r + 1) * self.k]
    }

    /// Row `r`'s per-block scales.
    #[must_use]
    #[inline]
    pub fn scales_row(&self, r: usize) -> &[f32] {
        let bpr = self.blocks_per_row();
        &self.scales[r * bpr..(r + 1) * bpr]
    }

    /// Row `r`'s per-block exact code sums (the VNNI bias correction).
    #[must_use]
    #[inline]
    pub fn sums_row(&self, r: usize) -> &[i32] {
        let bpr = self.blocks_per_row();
        &self.sums[r * bpr..(r + 1) * bpr]
    }

    /// Reject an activation whose shape or layout does not match what a
    /// kernel expects, with a real (always-on) check rather than a
    /// `debug_assert!` (K-02 / KERN-SOUND).
    ///
    /// # Errors
    ///
    /// - [`KernelError::NotBlockAligned`] on a `qk` or layout mismatch.
    /// - [`KernelError::NamedDimensionMismatch`] on a `k` mismatch.
    /// - [`KernelError::NamedBufferTooSmall`] if it has too few rows.
    pub fn expect_shape(
        &self,
        rows: usize,
        k: usize,
        qk: usize,
        layout: Int8Layout,
    ) -> KernelResult<()> {
        if self.qk != qk || self.layout != layout {
            return Err(KernelError::NotBlockAligned {
                count: self.qk,
                block_size: qk,
            });
        }
        if self.k != k {
            return Err(KernelError::dimension_mismatch("activation_k", k, self.k));
        }
        if self.rows < rows {
            return Err(KernelError::buffer_too_small(
                "activation_rows",
                rows,
                self.rows,
            ));
        }
        Ok(())
    }

    /// Relative quantization error of this activation against the `f32`
    /// original, as `||x - s*q|| / ||x||` — a diagnostic for the accuracy
    /// gate, not used by any kernel.
    ///
    /// Returns `0.0` for an all-zero input.
    ///
    /// # Errors
    ///
    /// [`KernelError::NamedDimensionMismatch`] if `input` is shorter than
    /// `rows() * k()`.
    pub fn relative_error(&self, input: &[f32]) -> KernelResult<f32> {
        if input.len() < self.rows * self.k {
            return Err(KernelError::dimension_mismatch(
                "input",
                self.rows * self.k,
                input.len(),
            ));
        }
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for r in 0..self.rows {
            let codes = self.codes_row(r);
            let scales = self.scales_row(r);
            for (b, &scale) in scales.iter().enumerate() {
                for j in 0..self.qk {
                    let orig = input[r * self.k + b * self.qk + j];
                    let pos = b * self.qk + self.layout.position_of(j);
                    let rebuilt = scale * codes[pos] as f32;
                    let diff = (orig - rebuilt) as f64;
                    num += diff * diff;
                    den += (orig as f64) * (orig as f64);
                }
            }
        }
        if den == 0.0 {
            return Ok(0.0);
        }
        Ok((num.sqrt() / den.sqrt()) as f32)
    }
}

#[cfg(test)]
mod int8_activation_tests {
    use super::*;

    #[test]
    fn stride4_positions_are_a_permutation_of_each_64_group() {
        let mut seen = [false; 128];
        for j in 0..128usize {
            let p = Int8Layout::Stride4.position_of(j);
            assert!(p < 128, "position {p} out of range for j={j}");
            assert!(!seen[p], "position {p} produced twice");
            seen[p] = true;
            assert_eq!(p / 64, j / 64, "j={j} must stay inside its 64-group");
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn stride4_matches_the_simd_decode_order() {
        // A 16-byte `qs` chunk masked at shift 2*s yields lanes {4i + s};
        // position s*16 + i must therefore hold element 4i + s.
        for s in 0..4usize {
            for i in 0..16usize {
                assert_eq!(Int8Layout::Stride4.position_of(4 * i + s), s * 16 + i);
            }
        }
    }

    #[test]
    fn sequential_layout_is_the_identity() {
        for j in 0..256usize {
            assert_eq!(Int8Layout::Sequential.position_of(j), j);
        }
    }

    #[test]
    fn quantize_uses_a_symmetric_range_and_exact_sums() {
        let mut input = vec![0.0f32; 128];
        for (j, v) in input.iter_mut().enumerate() {
            *v = (j as f32 - 64.0) / 8.0;
        }
        let act = Int8Activation::quantize(&input, 1, 128, 128, Int8Layout::Sequential)
            .expect("quantize");
        let scales = act.scales_row(0);
        assert_eq!(scales.len(), 1);
        let amax = input.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!((scales[0] - amax / 127.0).abs() < 1e-9);

        let codes = act.codes_row(0);
        assert!(codes.iter().all(|&c| (-127..=127).contains(&(c as i32))));
        let expect_sum: i32 = codes.iter().map(|&c| c as i32).sum();
        assert_eq!(act.sums_row(0), &[expect_sum]);
    }

    #[test]
    fn quantize_negates_exactly() {
        let mut input = vec![0.0f32; 128];
        let mut state = 0x1234_5678u32;
        for v in input.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((state >> 8) as i32 % 1000 - 500) as f32 / 37.0;
        }
        let neg: Vec<f32> = input.iter().map(|v| -v).collect();
        let a =
            Int8Activation::quantize(&input, 1, 128, 128, Int8Layout::Stride4).expect("quantize");
        let b = Int8Activation::quantize(&neg, 1, 128, 128, Int8Layout::Stride4).expect("quantize");
        for (x, y) in a.codes_row(0).iter().zip(b.codes_row(0).iter()) {
            assert_eq!(*x, -*y, "quantization must be sign-symmetric");
        }
    }

    #[test]
    fn all_zero_block_quantizes_to_zero_scale_and_zero_codes() {
        let input = vec![0.0f32; 128];
        let act =
            Int8Activation::quantize(&input, 1, 128, 128, Int8Layout::Stride4).expect("quantize");
        assert_eq!(act.scales_row(0), &[0.0]);
        assert_eq!(act.sums_row(0), &[0]);
        assert!(act.codes_row(0).iter().all(|&c| c == 0));
        assert_eq!(act.relative_error(&input).expect("relative_error"), 0.0);
    }

    /// A NaN activation element must poison its
    /// whole block's `scale` (and therefore, downstream, that block's
    /// contribution to the dot product) rather than silently quantizing to
    /// code `0` and vanishing — `f32::max` alone would do exactly that,
    /// since IEEE 754 `maxNum` ignores a NaN operand.
    #[test]
    fn a_nan_element_poisons_its_blocks_scale_instead_of_vanishing() {
        let mut input = vec![1.0f32; 128];
        input[5] = f32::NAN;
        let act =
            Int8Activation::quantize(&input, 1, 128, 128, Int8Layout::Stride4).expect("quantize");
        assert!(
            act.scales_row(0)[0].is_nan(),
            "a block containing a NaN element must get a NaN scale, not a \
             finite one derived from the other elements"
        );
        // Without the fix, the other (finite, all equal to 1.0) elements
        // would still set a well-defined, nonzero scale via `f32::max`, so
        // this negative check pins that the fix is really in effect, not
        // just that *some* non-finite path was hit.
        assert_ne!(
            act.scales_row(0)[0].to_bits(),
            (1.0f32 / INT8_ACTIVATION_MAX).to_bits(),
            "the NaN must not have been silently dropped from the amax fold"
        );
    }

    /// Companion to the single-block test above: a block with **no** NaN in
    /// the same row as one that does must be entirely unaffected — NaN
    /// contagion is per-block, not per-row.
    #[test]
    fn a_nan_in_one_block_does_not_poison_a_sibling_block_in_the_same_row() {
        let mut input = vec![1.0f32; 256]; // two 128-wide blocks
        input[10] = f32::NAN; // only block 0
        let act =
            Int8Activation::quantize(&input, 1, 256, 128, Int8Layout::Stride4).expect("quantize");
        assert!(act.scales_row(0)[0].is_nan(), "block 0 must be poisoned");
        assert!(
            (act.scales_row(0)[1] - 1.0 / INT8_ACTIVATION_MAX).abs() < 1e-9,
            "block 1 (no NaN) must quantize normally: got {}",
            act.scales_row(0)[1]
        );
        assert!(act.codes_row(0)[128..].iter().all(|&c| c == 127));
    }

    #[test]
    fn stride4_round_trips_every_element_to_its_own_position() {
        let mut input = vec![0.0f32; 256];
        for (j, v) in input.iter_mut().enumerate() {
            *v = j as f32 + 1.0;
        }
        // 127 distinct magnitudes cannot survive 256 distinct inputs, so
        // check placement rather than value: the largest element must land
        // at its permuted position with code 127.
        let act =
            Int8Activation::quantize(&input, 1, 256, 128, Int8Layout::Stride4).expect("quantize");
        let codes = act.codes_row(0);
        let last_in_block1 = 127usize; // element index inside block 1
        let pos = 128 + Int8Layout::Stride4.position_of(last_in_block1);
        assert_eq!(codes[pos], 127);
    }

    #[test]
    fn relative_error_stays_small_for_a_smooth_activation() {
        let mut input = vec![0.0f32; 512];
        for (j, v) in input.iter_mut().enumerate() {
            *v = ((j as f32) * 0.01).sin() * 3.0;
        }
        let act =
            Int8Activation::quantize(&input, 1, 512, 128, Int8Layout::Stride4).expect("quantize");
        let err = act.relative_error(&input).expect("relative_error");
        assert!(err < 0.01, "int8 activation relative error {err} too large");
    }

    #[test]
    fn quantize_rejects_a_misaligned_k() {
        let input = vec![0.0f32; 100];
        assert!(
            Int8Activation::quantize(&input, 1, 100, 128, Int8Layout::Stride4).is_err(),
            "k must be a multiple of qk"
        );
    }

    #[test]
    fn quantize_rejects_a_short_input() {
        let input = vec![0.0f32; 64];
        let err = Int8Activation::quantize(&input, 1, 128, 128, Int8Layout::Stride4)
            .expect_err("must reject");
        assert_eq!(err.buffer_name(), Some("input"));
    }

    #[test]
    fn expect_shape_rejects_a_layout_mismatch() {
        let input = vec![1.0f32; 128];
        let act =
            Int8Activation::quantize(&input, 1, 128, 128, Int8Layout::Stride4).expect("quantize");
        assert!(act.expect_shape(1, 128, 128, Int8Layout::Stride4).is_ok());
        assert!(act
            .expect_shape(1, 128, 128, Int8Layout::Sequential)
            .is_err());
        assert!(act.expect_shape(1, 128, 64, Int8Layout::Stride4).is_err());
        assert!(act.expect_shape(2, 128, 128, Int8Layout::Stride4).is_err());
        assert!(act.expect_shape(1, 256, 128, Int8Layout::Stride4).is_err());
    }

    #[test]
    fn multi_row_rows_are_quantized_independently() {
        let mut input = vec![0.0f32; 2 * 128];
        for j in 0..128 {
            input[j] = 1.0; // row 0: amax 1
            input[128 + j] = 100.0; // row 1: amax 100
        }
        let act = Int8Activation::quantize(&input, 2, 128, 128, Int8Layout::Sequential)
            .expect("quantize");
        assert!((act.scales_row(0)[0] - 1.0 / 127.0).abs() < 1e-9);
        assert!((act.scales_row(1)[0] - 100.0 / 127.0).abs() < 1e-6);
        assert!(act.codes_row(0).iter().all(|&c| c == 127));
        assert!(act.codes_row(1).iter().all(|&c| c == 127));
    }
}
