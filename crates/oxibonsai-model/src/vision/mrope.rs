//! 3-axis M-RoPE positions for a prompt with images spliced into it
//! (bonsai2-design.md §6.2, §2.5).
//!
//! # The layout
//!
//! Every row of a multimodal prompt carries a rotary position with three
//! axes `(t, h, w)` ([`MropePos`]). The reference (the PrismML llama.cpp
//! fork's `libmtmd`, `mtmd_image_tokens_get_decoder_pos` with
//! `MTMD_POS_TYPE_MROPE`) lays a prompt out as:
//!
//! * a **text** token at rotary position `p` has `t = h = w = p`, and the
//!   next token is at `p + 1`;
//! * an **image** whose merged grid is `h x w` (rows of 2 x 2 patch windows,
//!   row-major) and whose first row starts at rotary position `p0` puts
//!   merged token `(row, col)` at `t = p0`, `h = p0 + row`, `w = p0 + col`,
//!   and the token after the image continues at `p0 + max(h, w)`
//!   (`mtmd_image_tokens_get_n_pos`).
//!
//! So an image of `h * w` rows advances the rotary cursor by only
//! `max(h, w)`: after it, text rotary positions run behind the sequence
//! index by `h * w - max(h, w)` (the model's `rope_delta`).
//!
//! The reference hands the axes to the attention as `pos[0] = t`,
//! `pos[1] = y` (row), `pos[2] = x` (column) (`set_position_mrope_2d`), and
//! `ggml_mrope_cache_init`'s interleaved rule (`sector % 3`) reads `pos[1]`
//! as `theta_h` and `pos[2]` as `theta_w` — which is exactly
//! `mrope_build_tables([t, h, w], ..)`'s axis order.
//!
//! # Angles
//!
//! [`mrope_angles`] builds one row's `(cos, sin)` table through the kernels
//! crate's [`mrope_build_tables`] with the model's
//! `rope.dimension_sections` (`[11, 11, 10, 0]` for Bonsai 2). For a text
//! position the result is bitwise the single-axis table
//! (`layers::rope_mrope`'s degeneracy tests); for an image row the three
//! axes diverge and the interleaved sectors pick them apart.

use oxibonsai_kernels::rope_mrope::mrope_build_tables;

use super::GridSize;
use crate::error::{ModelError, ModelResult};
pub use crate::layers::rope_mrope::MropePos;

/// A rotary position as the `u32` an [`MropePos`] axis stores.
fn axis(p: usize) -> ModelResult<u32> {
    u32::try_from(p).map_err(|_| ModelError::ShapeInvariant {
        tensor: "M-RoPE position".to_string(),
        expected: "a rotary position representable as u32".to_string(),
        actual: p.to_string(),
    })
}

/// How far an image of merged grid `grid` advances the rotary cursor:
/// `max(h, w)` (`mtmd_image_tokens_get_n_pos` for M-RoPE models).
#[must_use]
pub const fn image_rope_span(grid: GridSize) -> usize {
    if grid.h > grid.w {
        grid.h
    } else {
        grid.w
    }
}

/// The rotary positions of an image of merged grid `grid` whose first row
/// starts at rotary position `p0`: one per merged token, row-major,
/// `(t, h, w) = (p0, p0 + row, p0 + col)`.
///
/// # Errors
///
/// [`ModelError::ShapeInvariant`] for an empty grid or a position past
/// `u32::MAX`.
pub fn image_positions(p0: usize, grid: GridSize) -> ModelResult<Vec<MropePos>> {
    if grid.h == 0 || grid.w == 0 {
        return Err(ModelError::ShapeInvariant {
            tensor: "image grid".to_string(),
            expected: "a non-empty merged grid".to_string(),
            actual: format!("{} x {}", grid.h, grid.w),
        });
    }
    let t = axis(p0)?;
    // The largest axis value any row reaches, checked once.
    axis(p0.saturating_add(image_rope_span(grid)))?;
    let mut out = Vec::with_capacity(grid.n_tokens());
    for row in 0..grid.h {
        for col in 0..grid.w {
            out.push(MropePos {
                t,
                h: axis(p0 + row)?,
                w: axis(p0 + col)?,
            });
        }
    }
    Ok(out)
}

/// The rotary positions of `n` consecutive text tokens starting at `p0`
/// (all three axes equal).
///
/// # Errors
///
/// [`ModelError::ShapeInvariant`] for a position past `u32::MAX`.
pub fn text_positions(p0: usize, n: usize) -> ModelResult<Vec<MropePos>> {
    axis(p0.saturating_add(n))?;
    (p0..p0 + n).map(|p| axis(p).map(MropePos::text)).collect()
}

/// The rotary position the token after `positions` takes: one past the
/// largest axis value any of them uses — `p + 1` after a text token,
/// `p0 + max(h, w)` after an image. `None` for an empty slice.
#[must_use]
pub fn next_rope_position(positions: &[MropePos]) -> Option<usize> {
    positions
        .iter()
        .map(|p| p.t.max(p.h).max(p.w) as usize)
        .max()
        .map(|m| m + 1)
}

/// Walks a prompt piece by piece, handing out rotary positions in the
/// design §6.2 layout (text: one position per token; image: its grid,
/// then `max(h, w)` positions of advance).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropeCursor {
    next: usize,
    positions: Vec<MropePos>,
}

impl MropeCursor {
    /// A cursor whose first row rotates at `start`.
    #[must_use]
    pub fn new(start: usize) -> Self {
        Self {
            next: start,
            positions: Vec::new(),
        }
    }

    /// Append `n` text rows.
    ///
    /// # Errors
    ///
    /// As [`text_positions`].
    pub fn push_text(&mut self, n: usize) -> ModelResult<()> {
        self.positions.extend(text_positions(self.next, n)?);
        self.next += n;
        Ok(())
    }

    /// Append one image of merged grid `grid`.
    ///
    /// # Errors
    ///
    /// As [`image_positions`].
    pub fn push_image(&mut self, grid: GridSize) -> ModelResult<()> {
        self.positions.extend(image_positions(self.next, grid)?);
        self.next += image_rope_span(grid);
        Ok(())
    }

    /// The rotary position the next row would take.
    #[must_use]
    pub fn next_position(&self) -> usize {
        self.next
    }

    /// Every position handed out so far, in row order.
    #[must_use]
    pub fn positions(&self) -> &[MropePos] {
        &self.positions
    }

    /// Consume the cursor into its positions.
    #[must_use]
    pub fn into_positions(self) -> Vec<MropePos> {
        self.positions
    }
}

/// Build the `(cos, sin)` table of one row at the 3-axis position `pos`:
/// `n_rot / 2` entries each, through the kernels crate's
/// [`mrope_build_tables`] with `sections` = `rope.dimension_sections` (the
/// reference's interleaved M-RoPE, `ggml_mrope_cache_init` with
/// `is_imrope`).
///
/// # Errors
///
/// [`ModelError::Kernel`] for an odd `n_rot`, sections that sum to zero or
/// need the vision `e` axis, or short output buffers;
/// [`ModelError::ShapeInvariant`] for an axis past `i32::MAX`.
pub fn mrope_angles(
    pos: MropePos,
    sections: [u32; 4],
    n_rot: usize,
    freq_base: f32,
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) -> ModelResult<()> {
    let to_i32 = |v: u32| {
        i32::try_from(v).map_err(|_| ModelError::ShapeInvariant {
            tensor: "M-RoPE position".to_string(),
            expected: "representable as i32".to_string(),
            actual: v.to_string(),
        })
    };
    mrope_build_tables(
        [to_i32(pos.t)?, to_i32(pos.h)?, to_i32(pos.w)?],
        sections,
        n_rot,
        freq_base,
        cos_out,
        sin_out,
    )
    .map_err(ModelError::Kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::rope_mrope::{MropeTable, PartialRopeTable};
    use oxibonsai_kernels::rope_mrope::partial_rope_build_table;

    /// Bonsai 2's `rope.dimension_sections`, rotation width and base.
    const SECTIONS: [u32; 4] = [11, 11, 10, 0];
    const N_ROT: usize = 64;
    const HEAD_DIM: usize = 256;
    const FREQ_BASE: f32 = 1e7;

    fn grid(h: usize, w: usize) -> GridSize {
        GridSize { h, w }
    }

    #[test]
    fn image_positions_are_row_major_over_the_merged_grid() {
        let positions = image_positions(4, grid(2, 3)).expect("positions");
        let expected: Vec<(u32, u32, u32)> = vec![
            (4, 4, 4),
            (4, 4, 5),
            (4, 4, 6),
            (4, 5, 4),
            (4, 5, 5),
            (4, 5, 6),
        ];
        let got: Vec<(u32, u32, u32)> = positions.iter().map(|p| (p.t, p.h, p.w)).collect();
        assert_eq!(got, expected);
        assert_eq!(image_rope_span(grid(2, 3)), 3);
        assert_eq!(image_rope_span(grid(6, 8)), 8);
        assert_eq!(image_rope_span(grid(9, 4)), 9);
        assert!(image_positions(0, grid(0, 3)).is_err());
        assert!(image_positions(0, grid(3, 0)).is_err());
    }

    /// The reference's layout of the golden vision prompt: `<|im_start|>`,
    /// `user`, `\n`, `<|vision_start|>` (4 text rows), the 256 x 192
    /// fixture's 6 x 8 merged grid (48 image rows at `p0 = 4`), then 15 text
    /// rows resuming at `4 + max(6, 8) = 12` — 67 sequence rows over 27
    /// rotary positions, leaving an offset of 40.
    #[test]
    fn the_golden_vision_prompt_layout_resumes_text_at_p0_plus_max_h_w() {
        let mut cursor = MropeCursor::new(0);
        cursor.push_text(4).expect("text");
        cursor.push_image(grid(6, 8)).expect("image");
        cursor.push_text(15).expect("text");
        let positions = cursor.positions();
        assert_eq!(positions.len(), 4 + 48 + 15);
        assert_eq!(positions[3], MropePos::text(3));
        // First and last image rows.
        assert_eq!(positions[4], MropePos { t: 4, h: 4, w: 4 });
        assert_eq!(positions[4 + 47], MropePos { t: 4, h: 9, w: 11 });
        // The text after the image (the reference's server logged its
        // batch ending at position 22 with 11 tokens: 12..=22).
        assert_eq!(positions[52], MropePos::text(12));
        assert_eq!(positions[62], MropePos::text(22));
        assert_eq!(cursor.next_position(), 27);
        assert_eq!(next_rope_position(positions), Some(27));
        assert_eq!(positions.len() - cursor.next_position(), 40);
    }

    #[test]
    fn text_positions_are_consecutive_and_equal_on_every_axis() {
        let positions = text_positions(7, 3).expect("positions");
        assert_eq!(
            positions,
            vec![MropePos::text(7), MropePos::text(8), MropePos::text(9)]
        );
        assert_eq!(next_rope_position(&positions), Some(10));
        assert_eq!(next_rope_position(&[]), None);
        assert!(text_positions(u32::MAX as usize, 2).is_err());
    }

    /// An independent `f64` evaluation of the reference's interleaved
    /// M-RoPE angle for rotation pair `k` (`ggml_mrope_cache_init` with
    /// `is_imrope`, `indep_sects = false`): sector `k % 3` picks the axis
    /// (`1 -> h` below `3 * s[1]`, `2 -> w` below `3 * s[2]`, `0 -> t`
    /// below `3 * s[0]`) and the angle is `axis * base^(-2k / n_rot)`.
    fn reference_angle(pos: MropePos, k: usize) -> f64 {
        let sect_dims: usize = SECTIONS.iter().map(|&s| s as usize).sum();
        let sector = k % sect_dims;
        let axis = if sector % 3 == 1 && sector < 3 * SECTIONS[1] as usize {
            pos.h
        } else if sector % 3 == 2 && sector < 3 * SECTIONS[2] as usize {
            pos.w
        } else {
            assert!(sector.is_multiple_of(3) && sector < 3 * SECTIONS[0] as usize);
            pos.t
        };
        f64::from(axis) * f64::from(FREQ_BASE).powf(-2.0 * k as f64 / N_ROT as f64)
    }

    /// The non-degenerate (vision) case: three distinct axes, checked
    /// against the independent `f64` angle, and visibly different from the
    /// text table at any single one of them.
    #[test]
    fn vision_angles_follow_the_interleaved_sector_rule() {
        let pos = MropePos { t: 4, h: 9, w: 11 };
        let half = N_ROT / 2;
        let mut cos = vec![0.0f32; half];
        let mut sin = vec![0.0f32; half];
        mrope_angles(pos, SECTIONS, N_ROT, FREQ_BASE, &mut cos, &mut sin).expect("angles");
        for k in 0..half {
            let theta = reference_angle(pos, k);
            assert!(
                (f64::from(cos[k]) - theta.cos()).abs() < 1e-5,
                "cos[{k}]: {} vs {}",
                cos[k],
                theta.cos()
            );
            assert!(
                (f64::from(sin[k]) - theta.sin()).abs() < 1e-5,
                "sin[{k}]: {} vs {}",
                sin[k],
                theta.sin()
            );
        }
        // Pair 0 is a `t` sector, pair 1 an `h` sector, pair 2 a `w` one.
        let mut text_cos = vec![0.0f32; half];
        let mut text_sin = vec![0.0f32; half];
        for axis_value in [4u32, 9, 11] {
            partial_rope_build_table(
                axis_value as i32,
                N_ROT,
                FREQ_BASE,
                &mut text_cos,
                &mut text_sin,
            )
            .expect("text table");
            assert_ne!(
                cos, text_cos,
                "a vision row must not rotate like text at {axis_value}"
            );
        }
    }

    /// The degenerate (text) case through this module's entry point: equal
    /// axes give the single-axis table bit for bit.
    #[test]
    fn text_positions_degenerate_to_the_single_axis_table_bitwise() {
        let half = N_ROT / 2;
        for p in [0u32, 1, 12, 27, 4095, 262_143] {
            let mut cos = vec![0.0f32; half];
            let mut sin = vec![0.0f32; half];
            mrope_angles(
                MropePos::text(p),
                SECTIONS,
                N_ROT,
                FREQ_BASE,
                &mut cos,
                &mut sin,
            )
            .expect("angles");
            let mut want_cos = vec![0.0f32; half];
            let mut want_sin = vec![0.0f32; half];
            partial_rope_build_table(p as i32, N_ROT, FREQ_BASE, &mut want_cos, &mut want_sin)
                .expect("text table");
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&cos), bits(&want_cos), "cos at {p}");
            assert_eq!(bits(&sin), bits(&want_sin), "sin at {p}");
        }
    }

    /// Rotating a head with [`mrope_angles`] and the kernels' split-half
    /// rotation is exactly `MropeTable::apply_head` (the model crate's 3-axis
    /// table), and at a text position exactly `PartialRopeTable`.
    #[test]
    fn a_rotated_head_matches_the_mrope_table_bitwise() {
        let input: Vec<f32> = (0..HEAD_DIM)
            .map(|i| ((i as f32) * 0.37).sin() * 1.5)
            .collect();
        let table = MropeTable::new(HEAD_DIM, N_ROT, SECTIONS, FREQ_BASE);
        let half = N_ROT / 2;
        for pos in [
            MropePos { t: 4, h: 9, w: 11 },
            MropePos {
                t: 100,
                h: 104,
                w: 100,
            },
            MropePos::text(21),
        ] {
            let mut cos = vec![0.0f32; half];
            let mut sin = vec![0.0f32; half];
            mrope_angles(pos, SECTIONS, N_ROT, FREQ_BASE, &mut cos, &mut sin).expect("angles");
            let mut ours = vec![0.0f32; HEAD_DIM];
            oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd(
                &input, &mut ours, HEAD_DIM, N_ROT, &cos, &sin,
            )
            .expect("rotate");
            let mut theirs = vec![0.0f32; HEAD_DIM];
            table
                .apply_head(&input, &mut theirs, pos)
                .expect("table rotate");
            assert_eq!(ours, theirs, "{pos:?}");
            // The pass-through tail is untouched.
            assert_eq!(&ours[N_ROT..], &input[N_ROT..]);
        }
        let partial = PartialRopeTable::new(HEAD_DIM, N_ROT, 64, FREQ_BASE).expect("table");
        let mut text_out = vec![0.0f32; HEAD_DIM];
        partial
            .apply_head(&input, &mut text_out, 21)
            .expect("partial rotate");
        let mut via_mrope = vec![0.0f32; HEAD_DIM];
        table
            .apply_head(&input, &mut via_mrope, MropePos::text(21))
            .expect("mrope rotate");
        assert_eq!(text_out, via_mrope);
    }
}
