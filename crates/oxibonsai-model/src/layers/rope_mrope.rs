//! `PartialRopeTable` / `MropeTable`: model-crate wrappers around
//! `oxibonsai_kernels::rope_mrope`'s `rope_partial_splithalf_simd` /
//! `mrope_build_tables` (K-11 / M-06).
//!
//! `PartialRopeTable` is the table Bonsai 2's text-only full-attention
//! layers actually use (`n_rot = 64`, `head_dim = 256`, `freq_base = 1e7`):
//! precomputed per position, like [`crate::layers::rope::RopeTable`].
//! `MropeTable` is the general 3-axis (vision-ready) primitive; it computes
//! its table per call rather than precomputing one indexed by all `(t, h,
//! w)` combinations, which would be combinatorially infeasible once the
//! axes actually diverge (image patches).
//!
//! # Why `PartialRopeTable` calls `partial_rope_build_table` instead of
//! re-deriving the same formula
//!
//! `f32::powf` is a transcendental function: two independently type-written
//! expressions computing "the same" `theta_scale = freq_base^(-2/n_rot)`
//! are not guaranteed to be bit-identical `f32` values (verified empirically
//! while writing `oxibonsai-kernels`'s `rope_mrope` tests — a hand-written
//! reference loop with compile-time-constant inputs, which the compiler is
//! free to constant-fold via its own evaluator, differed from the same
//! formula's runtime `powf` call by 1 ULP at one position). The package's
//! acceptance gate is "`PartialRopeTable` vs `MropeTable` degeneracy bitwise
//! over 10 000 positions" for `t == h == w`, which is only provable by
//! construction, not by "these two loops look like the same math": both
//! structs must run the identical compiled `theta_scale`/iteration code.
//! `partial_rope_build_table` (in the kernels crate) *is* that identical
//! code path — it degenerates [`oxibonsai_kernels::rope_mrope::mrope_build_tables`]
//! to a single position — so `PartialRopeTable::new` and `MropeTable::apply_head`
//! (with `t == h == w`) both bottom out in the same function body.

use crate::error::{ModelError, ModelResult};

/// Precomputed table for partial NeoX RoPE over the first `n_rot` of a
/// `head_dim`-wide head, one text position axis (`p_t = p_h = p_w`).
#[derive(Debug, Clone)]
pub struct PartialRopeTable {
    /// `[max_pos][n_rot/2]`, row-major by position.
    cos: Vec<f32>,
    sin: Vec<f32>,
    n_rot: usize,
    head_dim: usize,
    max_pos: usize,
}

impl PartialRopeTable {
    /// Precompute the table for every position `0..max_pos`.
    ///
    /// # Errors
    ///
    /// [`ModelError::Kernel`] if `n_rot` is odd (propagated from
    /// [`oxibonsai_kernels::rope_mrope::partial_rope_build_table`] /
    /// [`oxibonsai_kernels::error::KernelError::NotBlockAligned`]) — a real
    /// error rather than a panic, since `new` is not `#[cfg(test)]`-only
    /// code and must not `unwrap`/`expect` a caller-supplied `n_rot`.
    pub fn new(head_dim: usize, n_rot: usize, max_pos: usize, freq_base: f32) -> ModelResult<Self> {
        let half = n_rot / 2;
        let mut cos = vec![0.0f32; max_pos * half];
        let mut sin = vec![0.0f32; max_pos * half];

        for pos in 0..max_pos {
            let cos_row = &mut cos[pos * half..(pos + 1) * half];
            let sin_row = &mut sin[pos * half..(pos + 1) * half];
            oxibonsai_kernels::rope_mrope::partial_rope_build_table(
                pos as i32, n_rot, freq_base, cos_row, sin_row,
            )?;
        }

        Ok(Self {
            cos,
            sin,
            n_rot,
            head_dim,
            max_pos,
        })
    }

    /// Rotate one `head_dim`-wide head in place-equivalent (`input` and
    /// `output` may be the same length but must be distinct slices, per
    /// [`oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd`]):
    /// dims `[0, n_rot)` are rotated, `[n_rot, head_dim)` pass through
    /// bitwise-unchanged.
    ///
    /// # Errors
    ///
    /// - [`ModelError::PositionOutOfRange`] if `pos >= self.max_pos()`.
    /// - [`ModelError::Kernel`] if `input`/`output` are shorter than
    ///   `head_dim` (propagated).
    pub fn apply_head(&self, input: &[f32], output: &mut [f32], pos: usize) -> ModelResult<()> {
        if pos >= self.max_pos {
            return Err(ModelError::PositionOutOfRange {
                pos,
                max: self.max_pos,
            });
        }
        let half = self.n_rot / 2;
        let cos_row = &self.cos[pos * half..(pos + 1) * half];
        let sin_row = &self.sin[pos * half..(pos + 1) * half];

        oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd(
            input,
            output,
            self.head_dim,
            self.n_rot,
            cos_row,
            sin_row,
        )?;

        Ok(())
    }

    /// Rotate `n_heads` consecutive `head_dim`-wide heads (each head is one
    /// `[h*head_dim, (h+1)*head_dim)` slice of `input`/`output`) at the same
    /// position.
    ///
    /// # Errors
    ///
    /// - [`ModelError::PositionOutOfRange`] if `pos >= self.max_pos()`.
    /// - [`ModelError::ShapeMismatch`] if `input.len() < n_heads *
    ///   head_dim` or `output.len() < n_heads * head_dim`.
    pub fn apply_heads(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_heads: usize,
        pos: usize,
    ) -> ModelResult<()> {
        if pos >= self.max_pos {
            return Err(ModelError::PositionOutOfRange {
                pos,
                max: self.max_pos,
            });
        }
        let needed = n_heads * self.head_dim;
        if input.len() < needed {
            return Err(ModelError::ShapeMismatch {
                name: "PartialRopeTable::apply_heads input".to_string(),
                expected: vec![needed],
                actual: vec![input.len()],
            });
        }
        if output.len() < needed {
            return Err(ModelError::ShapeMismatch {
                name: "PartialRopeTable::apply_heads output".to_string(),
                expected: vec![needed],
                actual: vec![output.len()],
            });
        }

        let half = self.n_rot / 2;
        let cos_row = &self.cos[pos * half..(pos + 1) * half];
        let sin_row = &self.sin[pos * half..(pos + 1) * half];

        for h in 0..n_heads {
            let in_head = &input[h * self.head_dim..(h + 1) * self.head_dim];
            let out_head = &mut output[h * self.head_dim..(h + 1) * self.head_dim];
            oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd(
                in_head,
                out_head,
                self.head_dim,
                self.n_rot,
                cos_row,
                sin_row,
            )?;
        }

        Ok(())
    }

    /// Number of precomputed positions (`0..max_pos()` are valid for
    /// [`Self::apply_head`] / [`Self::apply_heads`]).
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }
}

/// 3-axis rotary position, one per token (text: `t == h == w == token
/// index`; vision: the three axes diverge per image patch — wired by a
/// later package).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MropePos {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

impl MropePos {
    /// A text position: all three axes equal the token index.
    pub fn text(pos: u32) -> Self {
        Self {
            t: pos,
            h: pos,
            w: pos,
        }
    }
}

/// General interleaved-M-RoPE (`IMROPE`) table, indexed by the three
/// position axes and `ggml`'s `rope.dimension_sections`.
///
/// Unlike [`PartialRopeTable`], this does **not** precompute a table over
/// all position combinations (infeasible once the axes diverge, e.g. an
/// image's `h`/`w` patch grid) — [`Self::apply_head`] /
/// [`Self::apply_heads`] build the small (`n_rot/2`-wide) `cos`/`sin` table
/// fresh on each call via
/// [`oxibonsai_kernels::rope_mrope::mrope_build_tables`].
#[derive(Debug, Clone)]
pub struct MropeTable {
    sections: [u32; 4],
    n_rot: usize,
    head_dim: usize,
    freq_base: f32,
}

impl MropeTable {
    pub fn new(head_dim: usize, n_rot: usize, sections: [u32; 4], freq_base: f32) -> Self {
        Self {
            sections,
            n_rot,
            head_dim,
            freq_base,
        }
    }

    /// Rotate one `head_dim`-wide head at the given 3-axis position.
    ///
    /// # Errors
    ///
    /// [`ModelError::Kernel`] if `n_rot` is odd, `sections` sum to zero,
    /// the sector selection needs the unrepresentable vision `e` axis, or
    /// `input`/`output` are shorter than `head_dim` (all propagated from
    /// the underlying kernel calls).
    pub fn apply_head(&self, input: &[f32], output: &mut [f32], pos: MropePos) -> ModelResult<()> {
        let half = self.n_rot / 2;
        let mut cos = vec![0.0f32; half];
        let mut sin = vec![0.0f32; half];
        oxibonsai_kernels::rope_mrope::mrope_build_tables(
            [pos.t as i32, pos.h as i32, pos.w as i32],
            self.sections,
            self.n_rot,
            self.freq_base,
            &mut cos,
            &mut sin,
        )?;

        oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd(
            input,
            output,
            self.head_dim,
            self.n_rot,
            &cos,
            &sin,
        )?;

        Ok(())
    }

    /// Rotate `n_heads` consecutive `head_dim`-wide heads at the same
    /// 3-axis position.
    ///
    /// # Errors
    ///
    /// Same as [`Self::apply_head`], plus [`ModelError::ShapeMismatch`] if
    /// `input.len() < n_heads * head_dim` or `output.len() < n_heads *
    /// head_dim`.
    pub fn apply_heads(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_heads: usize,
        pos: MropePos,
    ) -> ModelResult<()> {
        let needed = n_heads * self.head_dim;
        if input.len() < needed {
            return Err(ModelError::ShapeMismatch {
                name: "MropeTable::apply_heads input".to_string(),
                expected: vec![needed],
                actual: vec![input.len()],
            });
        }
        if output.len() < needed {
            return Err(ModelError::ShapeMismatch {
                name: "MropeTable::apply_heads output".to_string(),
                expected: vec![needed],
                actual: vec![output.len()],
            });
        }

        let half = self.n_rot / 2;
        let mut cos = vec![0.0f32; half];
        let mut sin = vec![0.0f32; half];
        oxibonsai_kernels::rope_mrope::mrope_build_tables(
            [pos.t as i32, pos.h as i32, pos.w as i32],
            self.sections,
            self.n_rot,
            self.freq_base,
            &mut cos,
            &mut sin,
        )?;

        for h in 0..n_heads {
            let in_head = &input[h * self.head_dim..(h + 1) * self.head_dim];
            let out_head = &mut output[h * self.head_dim..(h + 1) * self.head_dim];
            oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd(
                in_head,
                out_head,
                self.head_dim,
                self.n_rot,
                &cos,
                &sin,
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BONSAI2_HEAD_DIM: usize = 256;
    const BONSAI2_N_ROT: usize = 64;
    const BONSAI2_SECTIONS: [u32; 4] = [11, 11, 10, 0];
    const BONSAI2_FREQ_BASE: f32 = 1.0e7;

    // ── PartialRopeTable ────────────────────────────────────────

    #[test]
    fn partial_rope_table_identity_at_position_zero() {
        let table = PartialRopeTable::new(8, 4, 16, 10_000.0).expect("valid n_rot");
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];
        table.apply_head(&input, &mut output, 0).expect("valid");
        // pos=0 -> theta=0 for every rotated dim -> identity, and the tail
        // is a verbatim copy regardless of position.
        assert_eq!(output, input);
    }

    #[test]
    fn partial_rope_table_leaves_tail_untouched() {
        let table = PartialRopeTable::new(BONSAI2_HEAD_DIM, BONSAI2_N_ROT, 100, BONSAI2_FREQ_BASE)
            .expect("valid n_rot");
        let input: Vec<f32> = (0..BONSAI2_HEAD_DIM).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![f32::NAN; BONSAI2_HEAD_DIM];
        table.apply_head(&input, &mut output, 42).expect("valid");
        for i in BONSAI2_N_ROT..BONSAI2_HEAD_DIM {
            assert_eq!(output[i].to_bits(), input[i].to_bits(), "tail dim {i}");
        }
    }

    #[test]
    fn partial_rope_table_rejects_out_of_range_position() {
        let table = PartialRopeTable::new(8, 4, 8, 10_000.0).expect("valid n_rot");
        let input = vec![0.0f32; 8];
        let mut output = vec![0.0f32; 8];
        let err = table
            .apply_head(&input, &mut output, 8)
            .expect_err("pos == max_pos must be rejected");
        assert!(matches!(
            err,
            ModelError::PositionOutOfRange { pos: 8, max: 8 }
        ));
    }

    #[test]
    fn partial_rope_table_apply_heads_matches_per_head_apply_head() {
        let table = PartialRopeTable::new(BONSAI2_HEAD_DIM, BONSAI2_N_ROT, 10, BONSAI2_FREQ_BASE)
            .expect("valid n_rot");
        let n_heads = 3;
        let input: Vec<f32> = (0..n_heads * BONSAI2_HEAD_DIM)
            .map(|i| (i as f32 - 200.0) * 0.03)
            .collect();
        let mut batched = vec![0.0f32; n_heads * BONSAI2_HEAD_DIM];
        table
            .apply_heads(&input, &mut batched, n_heads, 7)
            .expect("valid");

        for h in 0..n_heads {
            let in_head = &input[h * BONSAI2_HEAD_DIM..(h + 1) * BONSAI2_HEAD_DIM];
            let mut single = vec![0.0f32; BONSAI2_HEAD_DIM];
            table.apply_head(in_head, &mut single, 7).expect("valid");
            let batched_head = &batched[h * BONSAI2_HEAD_DIM..(h + 1) * BONSAI2_HEAD_DIM];
            assert_eq!(batched_head, single.as_slice(), "head {h}");
        }
    }

    #[test]
    fn partial_rope_table_apply_heads_rejects_short_input() {
        let table = PartialRopeTable::new(8, 4, 8, 10_000.0).expect("valid n_rot");
        let input = vec![0.0f32; 8]; // only 1 head's worth
        let mut output = vec![0.0f32; 16];
        assert!(table.apply_heads(&input, &mut output, 2, 0).is_err());
    }

    // ── MropeTable ──────────────────────────────────────────────

    #[test]
    fn mrope_table_matches_partial_rope_table_bitwise_for_text_positions() {
        // The package acceptance gate: for t==h==w (text), PartialRopeTable
        // and MropeTable must agree bitwise. Both bottom out in
        // `oxibonsai_kernels::rope_mrope::mrope_build_tables`'s exact
        // compiled `theta_scale`/iteration code (see the module doc
        // comment), so this is provable, not coincidental.
        let max_pos = 10_000;
        let partial =
            PartialRopeTable::new(BONSAI2_HEAD_DIM, BONSAI2_N_ROT, max_pos, BONSAI2_FREQ_BASE)
                .expect("valid n_rot");
        let mrope = MropeTable::new(
            BONSAI2_HEAD_DIM,
            BONSAI2_N_ROT,
            BONSAI2_SECTIONS,
            BONSAI2_FREQ_BASE,
        );

        let input: Vec<f32> = (0..BONSAI2_HEAD_DIM)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();

        for pos in 0..max_pos {
            let mut out_partial = vec![0.0f32; BONSAI2_HEAD_DIM];
            partial
                .apply_head(&input, &mut out_partial, pos)
                .expect("valid");

            let mut out_mrope = vec![0.0f32; BONSAI2_HEAD_DIM];
            mrope
                .apply_head(&input, &mut out_mrope, MropePos::text(pos as u32))
                .expect("valid");

            for i in 0..BONSAI2_HEAD_DIM {
                assert_eq!(
                    out_partial[i].to_bits(),
                    out_mrope[i].to_bits(),
                    "pos={pos} dim={i}: PartialRopeTable vs MropeTable diverged"
                );
            }
        }
    }

    #[test]
    fn mrope_table_leaves_tail_untouched() {
        let mrope = MropeTable::new(
            BONSAI2_HEAD_DIM,
            BONSAI2_N_ROT,
            BONSAI2_SECTIONS,
            BONSAI2_FREQ_BASE,
        );
        let input: Vec<f32> = (0..BONSAI2_HEAD_DIM).map(|i| (i as f32) * 0.02).collect();
        let mut output = vec![f32::NAN; BONSAI2_HEAD_DIM];
        mrope
            .apply_head(&input, &mut output, MropePos::text(123))
            .expect("valid");
        for i in BONSAI2_N_ROT..BONSAI2_HEAD_DIM {
            assert_eq!(output[i].to_bits(), input[i].to_bits(), "tail dim {i}");
        }
    }

    #[test]
    fn mrope_table_apply_heads_matches_per_head_apply_head() {
        let mrope = MropeTable::new(
            BONSAI2_HEAD_DIM,
            BONSAI2_N_ROT,
            BONSAI2_SECTIONS,
            BONSAI2_FREQ_BASE,
        );
        let n_heads = 4;
        let pos = MropePos::text(99);
        let input: Vec<f32> = (0..n_heads * BONSAI2_HEAD_DIM)
            .map(|i| (i as f32 - 100.0) * 0.02)
            .collect();
        let mut batched = vec![0.0f32; n_heads * BONSAI2_HEAD_DIM];
        mrope
            .apply_heads(&input, &mut batched, n_heads, pos)
            .expect("valid");

        for h in 0..n_heads {
            let in_head = &input[h * BONSAI2_HEAD_DIM..(h + 1) * BONSAI2_HEAD_DIM];
            let mut single = vec![0.0f32; BONSAI2_HEAD_DIM];
            mrope.apply_head(in_head, &mut single, pos).expect("valid");
            let batched_head = &batched[h * BONSAI2_HEAD_DIM..(h + 1) * BONSAI2_HEAD_DIM];
            assert_eq!(batched_head, single.as_slice(), "head {h}");
        }
    }

    #[test]
    fn mrope_table_apply_heads_rejects_short_output() {
        let mrope = MropeTable::new(8, 4, [1, 1, 1, 0], 10_000.0);
        let input = vec![0.0f32; 16];
        let mut output = vec![0.0f32; 8]; // only 1 head's worth, need 2
        assert!(mrope
            .apply_heads(&input, &mut output, 2, MropePos::default())
            .is_err());
    }

    #[test]
    fn mrope_table_propagates_kernel_error_for_unsupported_sections() {
        // sections[0] == 0 forces sector 0 to need the vision `e` axis
        // (see `oxibonsai_kernels::rope_mrope`'s test of the same shape).
        let mrope = MropeTable::new(8, 8, [0, 1, 1, 0], 10_000.0);
        let input = vec![0.0f32; 8];
        let mut output = vec![0.0f32; 8];
        let err = mrope
            .apply_head(&input, &mut output, MropePos::text(5))
            .expect_err("must propagate the e-axis error");
        assert!(matches!(err, ModelError::Kernel(_)));
    }

    #[test]
    fn mrope_pos_text_sets_all_three_axes_equal() {
        let p = MropePos::text(42);
        assert_eq!(p.t, 42);
        assert_eq!(p.h, 42);
        assert_eq!(p.w, 42);
    }
}
