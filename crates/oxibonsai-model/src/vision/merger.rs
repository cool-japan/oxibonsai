//! The splice layer: where an encoded image's rows enter the text stream
//! (bonsai2-design.md §6.2).
//!
//! The merger MLP itself (`v.post_ln` -> 2 x 2 merge -> `mm.0` -> GELU ->
//! `mm.2`) runs inside [`super::VisionTower::encode`]; its output is one
//! `[projection_dim]` row per merged token, in the language model's
//! **unrotated** embedding basis. This module decides where those rows go.
//!
//! # The contract
//!
//! The chat template renders each image as
//! `<|vision_start|><|image_pad|><|vision_end|>` (the real Bonsai 2
//! template's `render_content` macro; `Picture N: ` precedes it only under
//! `add_vision_id`). Tokenized, that is exactly one `<|image_pad|>` id per
//! image, directly between the two markers. [`plan_splice`] checks that
//! shape — one pad per image, every pad bracketed, nothing else — and
//! replaces each pad with the image's `h * w` merged rows, so a prompt of
//! `n` token ids and images of merged grids `g_i` becomes
//! `n - images + sum(g_i.h * g_i.w)` rows. This is the reference's own
//! expansion (`mtmd_tokenize`: `img_beg` + the image's `n_tokens` embedding
//! rows + `img_end` for the Qwen-VL projectors).
//!
//! # The trap this layer exists to avoid
//!
//! A text row is `token_embd[id]` **followed by the inverse Hadamard
//! transform** (the embedding table is stored in the rotated basis, design
//! §3.5). An image row must **not** go through that transform: the vision
//! tower is an ordinary, unfolded network whose output is already in the
//! basis block 0 reads. Rotating it anyway produces rows with the same norm
//! and plausible statistics — and a model that describes a different
//! picture. The model-side prefill (`HybridModel::forward_prefill_pieces`)
//! embeds text pieces through the lookup + inverse transform and copies
//! image rows verbatim; its tests assert both halves.

use std::ops::Range;

use super::GridSize;

/// The three special tokens an image occupies in a tokenized prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VisionTokenIds {
    /// `<|vision_start|>`.
    pub vision_start: u32,
    /// `<|vision_end|>`.
    pub vision_end: u32,
    /// `<|image_pad|>` — one per image in the rendered prompt, replaced by
    /// the image's rows.
    pub image_pad: u32,
}

impl VisionTokenIds {
    /// The Bonsai 2 vocabulary's ids (design Appendix A.1, read from the
    /// real `PQ2_0` GGUF): `<|vision_start|>` 248053, `<|vision_end|>`
    /// 248054, `<|image_pad|>` 248056.
    pub const BONSAI2: Self = Self {
        vision_start: 248_053,
        vision_end: 248_054,
        image_pad: 248_056,
    };
}

/// Why a tokenized prompt and its images do not splice.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SpliceError {
    /// The prompt holds a different number of `<|image_pad|>` ids than
    /// there are images.
    #[error(
        "the rendered prompt holds {pads} <|image_pad|> placeholder(s) for {images} image(s); \
         every image needs exactly one placeholder"
    )]
    PadCountMismatch {
        /// `<|image_pad|>` ids found in the prompt.
        pads: usize,
        /// Images supplied.
        images: usize,
    },
    /// A placeholder is not directly between `<|vision_start|>` and
    /// `<|vision_end|>`.
    #[error(
        "the <|image_pad|> placeholder at token {index} is not bracketed by <|vision_start|> \
         and <|vision_end|>"
    )]
    MisplacedPad {
        /// Token index of the placeholder.
        index: usize,
    },
    /// An image with an empty merged grid.
    #[error("image {index} has an empty merged grid ({h} x {w})")]
    EmptyImage {
        /// Image index.
        index: usize,
        /// Merged rows.
        h: usize,
        /// Merged columns.
        w: usize,
    },
    /// The expanded prompt's row count does not fit in `usize`.
    #[error("the expanded prompt is too long to address")]
    Overflow,
}

impl SpliceError {
    /// A short, stable code for monitoring and API error bodies.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::PadCountMismatch { .. } => "image_placeholder_count_mismatch",
            Self::MisplacedPad { .. } => "image_placeholder_misplaced",
            Self::EmptyImage { .. } => "image_grid_empty",
            Self::Overflow => "image_prompt_overflow",
        }
    }
}

/// One stretch of a spliced prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpliceSegment {
    /// Token ids `range` of the rendered prompt (never containing a pad).
    Text(Range<usize>),
    /// Image `index`'s merged rows, in place of its `<|image_pad|>`.
    Image(usize),
}

/// Where every row of a spliced prompt comes from: text ranges of the
/// tokenized prompt interleaved with images, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplicePlan {
    segments: Vec<SpliceSegment>,
    grids: Vec<GridSize>,
    total_rows: usize,
    text_rows: usize,
}

impl SplicePlan {
    /// The segments, in prompt order.
    #[must_use]
    pub fn segments(&self) -> &[SpliceSegment] {
        &self.segments
    }

    /// The merged grid of every image, in placeholder order.
    #[must_use]
    pub fn grids(&self) -> &[GridSize] {
        &self.grids
    }

    /// Rows the model sees: every non-placeholder token plus every image's
    /// `h * w` rows.
    #[must_use]
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Rows that are text tokens.
    #[must_use]
    pub fn text_rows(&self) -> usize {
        self.text_rows
    }

    /// Rows that are image rows.
    #[must_use]
    pub fn image_rows(&self) -> usize {
        self.total_rows - self.text_rows
    }
}

/// Rows a prompt of `n_tokens` ids with one placeholder per image in
/// `grids` expands to: `n_tokens - grids.len() + sum(h * w)`.
///
/// # Errors
///
/// [`SpliceError::PadCountMismatch`] when `n_tokens` cannot even hold one
/// placeholder per image, [`SpliceError::Overflow`] when the sum does not
/// fit in `usize`.
pub fn expanded_row_count(n_tokens: usize, grids: &[GridSize]) -> Result<usize, SpliceError> {
    let text = n_tokens
        .checked_sub(grids.len())
        .ok_or(SpliceError::PadCountMismatch {
            pads: n_tokens,
            images: grids.len(),
        })?;
    grids.iter().try_fold(text, |acc, g| {
        g.h.checked_mul(g.w)
            .and_then(|n| acc.checked_add(n))
            .ok_or(SpliceError::Overflow)
    })
}

/// Check that `tokens` holds exactly one bracketed `<|image_pad|>` per
/// image in `grids` (in order) and plan the splice.
///
/// # Errors
///
/// [`SpliceError::PadCountMismatch`], [`SpliceError::MisplacedPad`],
/// [`SpliceError::EmptyImage`] or [`SpliceError::Overflow`] — every one a
/// property of the request, never a partial plan.
pub fn plan_splice(
    tokens: &[u32],
    grids: &[GridSize],
    ids: VisionTokenIds,
) -> Result<SplicePlan, SpliceError> {
    let pads: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| (t == ids.image_pad).then_some(i))
        .collect();
    if pads.len() != grids.len() {
        return Err(SpliceError::PadCountMismatch {
            pads: pads.len(),
            images: grids.len(),
        });
    }
    for (index, grid) in grids.iter().enumerate() {
        if grid.h == 0 || grid.w == 0 {
            return Err(SpliceError::EmptyImage {
                index,
                h: grid.h,
                w: grid.w,
            });
        }
    }
    let mut segments = Vec::with_capacity(2 * pads.len() + 1);
    let mut cursor = 0usize;
    for (image, &pad) in pads.iter().enumerate() {
        let before = pad.checked_sub(1).and_then(|i| tokens.get(i));
        let after = tokens.get(pad + 1);
        if before != Some(&ids.vision_start) || after != Some(&ids.vision_end) {
            return Err(SpliceError::MisplacedPad { index: pad });
        }
        if pad > cursor {
            segments.push(SpliceSegment::Text(cursor..pad));
        }
        segments.push(SpliceSegment::Image(image));
        cursor = pad + 1;
    }
    if cursor < tokens.len() {
        segments.push(SpliceSegment::Text(cursor..tokens.len()));
    }
    let total_rows = expanded_row_count(tokens.len(), grids)?;
    Ok(SplicePlan {
        segments,
        grids: grids.to_vec(),
        total_rows,
        text_rows: tokens.len() - pads.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDS: VisionTokenIds = VisionTokenIds::BONSAI2;
    const IM_START: u32 = 248_045;
    const NL: u32 = 198;

    fn grid(h: usize, w: usize) -> GridSize {
        GridSize { h, w }
    }

    /// The golden vision prompt's shape (stand-in ids for the text; the
    /// real-model gate checks the real ones): 4 text ids, the bracketed
    /// placeholder, 14 more text ids after `<|vision_end|>` — 20 ids that
    /// expand to (192 / 32) * (256 / 32) = 48 image rows plus 19 text rows
    /// = 67, the reference server's own `usage.prompt_tokens`.
    #[test]
    fn the_golden_fixture_expands_to_h_times_w_rows() {
        let mut tokens = vec![IM_START, 872, NL, IDS.vision_start, IDS.image_pad];
        tokens.push(IDS.vision_end);
        tokens.extend([
            74785, 419, 2168, 26753, 13, 248_046, NL, IM_START, 77091, NL,
        ]);
        tokens.extend([248_068, 271, 248_069, 271]);
        assert_eq!(tokens.len(), 20);
        let fixture = grid(192 / 32, 256 / 32);
        assert_eq!(fixture.n_tokens(), 48);
        let plan = plan_splice(&tokens, &[fixture], IDS).expect("plan");
        assert_eq!(plan.total_rows(), 67);
        assert_eq!(plan.image_rows(), 48);
        assert_eq!(plan.text_rows(), 19);
        assert_eq!(
            plan.segments(),
            &[
                SpliceSegment::Text(0..4),
                SpliceSegment::Image(0),
                SpliceSegment::Text(5..20),
            ]
        );
        assert_eq!(expanded_row_count(20, &[fixture]), Ok(67));
    }

    #[test]
    fn several_images_expand_in_placeholder_order() {
        let tokens = [
            IDS.vision_start,
            IDS.image_pad,
            IDS.vision_end,
            7,
            IDS.vision_start,
            IDS.image_pad,
            IDS.vision_end,
        ];
        let plan = plan_splice(&tokens, &[grid(2, 3), grid(1, 1)], IDS).expect("plan");
        assert_eq!(
            plan.segments(),
            &[
                SpliceSegment::Text(0..1),
                SpliceSegment::Image(0),
                SpliceSegment::Text(2..5),
                SpliceSegment::Image(1),
                SpliceSegment::Text(6..7),
            ]
        );
        assert_eq!(plan.total_rows(), 7 - 2 + 6 + 1);
        assert_eq!(plan.grids(), &[grid(2, 3), grid(1, 1)]);
    }

    #[test]
    fn a_placeholder_count_that_does_not_match_the_images_is_refused() {
        let tokens = [IDS.vision_start, IDS.image_pad, IDS.vision_end];
        let err = plan_splice(&tokens, &[], IDS).expect_err("no image");
        assert_eq!(err, SpliceError::PadCountMismatch { pads: 1, images: 0 });
        assert_eq!(err.code(), "image_placeholder_count_mismatch");
        let err = plan_splice(&tokens, &[grid(1, 1), grid(1, 1)], IDS).expect_err("two images");
        assert_eq!(err, SpliceError::PadCountMismatch { pads: 1, images: 2 });
        let err = plan_splice(&[1, 2, 3], &[grid(1, 1)], IDS).expect_err("no pad");
        assert_eq!(err, SpliceError::PadCountMismatch { pads: 0, images: 1 });
    }

    #[test]
    fn an_unbracketed_placeholder_is_refused() {
        for tokens in [
            vec![IDS.image_pad, IDS.vision_end],
            vec![IDS.vision_start, IDS.image_pad],
            vec![7, IDS.image_pad, IDS.vision_end],
            vec![IDS.vision_start, IDS.image_pad, 7],
        ] {
            let err = plan_splice(&tokens, &[grid(1, 1)], IDS).expect_err("misplaced");
            assert!(
                matches!(err, SpliceError::MisplacedPad { .. }),
                "{tokens:?}"
            );
            assert_eq!(err.code(), "image_placeholder_misplaced");
        }
    }

    #[test]
    fn an_empty_grid_is_refused() {
        let tokens = [IDS.vision_start, IDS.image_pad, IDS.vision_end];
        let err = plan_splice(&tokens, &[grid(0, 4)], IDS).expect_err("empty");
        assert_eq!(
            err,
            SpliceError::EmptyImage {
                index: 0,
                h: 0,
                w: 4
            }
        );
    }

    #[test]
    fn a_text_only_prompt_is_one_text_segment() {
        let plan = plan_splice(&[1, 2, 3], &[], IDS).expect("plan");
        assert_eq!(plan.segments(), &[SpliceSegment::Text(0..3)]);
        assert_eq!(plan.total_rows(), 3);
        assert_eq!(plan.image_rows(), 0);
    }

    #[test]
    fn expansion_overflow_is_an_error_not_a_wrap() {
        let huge = grid(usize::MAX, 2);
        assert_eq!(expanded_row_count(1, &[huge]), Err(SpliceError::Overflow));
    }
}
