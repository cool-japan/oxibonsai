//! Image preprocessing for the Qwen3-VL vision tower: the PrismML fork's
//! `mtmd_image_preprocessor_dyn_size` (`tools/mtmd/mtmd-image.cpp`),
//! reproduced step for step (bonsai2-design.md §6.2).
//!
//! # The reference pipeline (Qwen2-VL / 2.5-VL / 3-VL projectors)
//!
//! 1. **Smart resize** (`img_tool::calc_size_preserved_ratio`, "smart_resize"
//!    in transformers): round each side to the nearest multiple of
//!    `patch_size * spatial_merge` (32), at least 32; if the area exceeds
//!    `max_pixels`, shrink both sides by `sqrt(h * w / max_pixels)` and
//!    floor to 32; else if it is below `min_pixels`, grow both by
//!    `sqrt(min_pixels / (h * w))` and ceil to 32. All of it in `f32`,
//!    in the reference's operation order ([`smart_resize`]).
//! 2. **Resize into that box** (`img_tool::resize`, `PAD_CEIL` — the Qwen-VL
//!    default): scale by `min(tw / sw, th / sh)`, resize to `ceil(side *
//!    scale)` (capped at the box) with Pillow's separable **bicubic** filter
//!    (`a = -0.5`, 22-bit fixed-point weights, [`resample`]), and centre the
//!    result on a black canvas of the box size. An image already the box
//!    size is copied, untouched.
//! 3. Normalise with `image_mean` / `image_std` (0.5 / 0.5) — done by
//!    [`super::VisionTower::encode`], which owns the metadata.
//!
//! The bounds come from token counts: `max_pixels = max_tokens * 32^2`
//! and `min_pixels = min_tokens * 32^2`. The reference defaults for the
//! Qwen-VL family are 8 and 4096 tokens; this crate's per-image budget
//! defaults to [`DEFAULT_IMAGE_MAX_TOKENS`] (1024, the Bonsai demo's own
//! default, `--image-max-tokens`), and the minimum is
//! [`QWEN_VL_MIN_IMAGE_TOKENS`] (8) capped at the budget. The reference
//! warns that grounding tasks want at least 1024 image tokens; like its
//! server default, nothing here upscales past the minimum to get there.
//!
//! # The budget is a hard limit
//!
//! The smart resize keeps the aspect ratio and never lets a side fall
//! below 32, so a very elongated image can still need more merged tokens
//! than the budget (e.g. 32 x 100 000): [`prepare_image`] refuses that with
//! [`ImageInputError::TooManyTokens`] instead of distorting the image or
//! overrunning the budget.

use rayon::prelude::*;

use super::image_decode::ImageInputError;
use super::{GridSize, ImageRgb8, VisionTower};

/// The per-image merged-token budget unless the caller says otherwise
/// (`--image-max-tokens`; the Bonsai demo's default).
pub const DEFAULT_IMAGE_MAX_TOKENS: usize = 1024;

/// The reference's minimum merged-token count for the Qwen-VL projectors
/// (`set_limit_image_tokens(8, ..)`).
pub const QWEN_VL_MIN_IMAGE_TOKENS: usize = 8;

/// The largest budget accepted (`--image-max-tokens`): 16 384 merged tokens
/// is a 4096 x 4096 image, 4x the reference's own default maximum.
pub const MAX_IMAGE_MAX_TOKENS: usize = 16_384;

/// Pillow-compatible resampling filters (`resize_algo`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeFilter {
    /// Triangle filter, support 1.
    Bilinear,
    /// Keys cubic with `a = -0.5` (Pillow's), support 2 — the Qwen-VL
    /// default.
    #[default]
    Bicubic,
    /// Lanczos-3, support 3.
    Lanczos,
}

impl ResizeFilter {
    fn support(self) -> f64 {
        match self {
            Self::Bilinear => 1.0,
            Self::Bicubic => 2.0,
            Self::Lanczos => 3.0,
        }
    }

    /// The filter weight at distance `x` (`resample_filter`).
    fn weight(self, x: f64) -> f64 {
        match self {
            Self::Lanczos => {
                if (-3.0..3.0).contains(&x) {
                    let sinc = |v: f64| {
                        if v == 0.0 {
                            1.0
                        } else {
                            let pi_v = v * std::f64::consts::PI;
                            pi_v.sin() / pi_v
                        }
                    };
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
            Self::Bilinear => {
                let x = x.abs();
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Self::Bicubic => {
                let x = x.abs();
                const A: f64 = -0.5;
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
        }
    }
}

/// How an image is brought to the tower's geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreprocessConfig {
    /// ViT patch side (16 for Bonsai 2).
    pub patch_size: usize,
    /// Spatial merge (2): sides must be multiples of `patch_size * spatial_merge`.
    pub spatial_merge: usize,
    /// Merged tokens an image is grown to at least.
    pub min_tokens: usize,
    /// Merged tokens an image may use at most (`--image-max-tokens`).
    pub max_tokens: usize,
    /// The resampling filter.
    pub filter: ResizeFilter,
    /// The canvas colour around an aspect-preserved resize.
    pub pad_color: [u8; 3],
}

impl PreprocessConfig {
    /// The Qwen-VL preprocessing of the reference for a tower with the
    /// given patch and merge sizes and a `max_tokens` budget: bicubic, black
    /// padding, minimum `min(8, max_tokens)` tokens.
    ///
    /// # Errors
    ///
    /// [`ImageInputError::InvalidConfig`] for a zero patch or merge size, or
    /// a budget outside `1..=`[`MAX_IMAGE_MAX_TOKENS`].
    pub fn qwen_vl(
        patch_size: usize,
        spatial_merge: usize,
        max_tokens: usize,
    ) -> Result<Self, ImageInputError> {
        if patch_size == 0 || spatial_merge == 0 {
            return Err(ImageInputError::InvalidConfig {
                reason: format!("patch size {patch_size} / spatial merge {spatial_merge}"),
            });
        }
        if !(1..=MAX_IMAGE_MAX_TOKENS).contains(&max_tokens) {
            return Err(ImageInputError::InvalidConfig {
                reason: format!(
                    "--image-max-tokens {max_tokens} is outside 1..={MAX_IMAGE_MAX_TOKENS}"
                ),
            });
        }
        Ok(Self {
            patch_size,
            spatial_merge,
            min_tokens: QWEN_VL_MIN_IMAGE_TOKENS.min(max_tokens),
            max_tokens,
            filter: ResizeFilter::Bicubic,
            pad_color: [0, 0, 0],
        })
    }

    /// [`PreprocessConfig::qwen_vl`] for `tower`'s own geometry.
    ///
    /// # Errors
    ///
    /// As [`PreprocessConfig::qwen_vl`].
    pub fn for_tower(tower: &VisionTower, max_tokens: usize) -> Result<Self, ImageInputError> {
        let cfg = tower.config();
        Self::qwen_vl(cfg.patch_size, cfg.spatial_merge, max_tokens)
    }

    /// `patch_size * spatial_merge`: the side unit of the merged grid (32).
    #[must_use]
    pub fn merge_unit(&self) -> usize {
        self.patch_size * self.spatial_merge
    }

    /// `min_tokens * merge_unit^2`.
    #[must_use]
    pub fn min_pixels(&self) -> usize {
        self.min_tokens.saturating_mul(self.merge_unit().pow(2))
    }

    /// `max_tokens * merge_unit^2`.
    #[must_use]
    pub fn max_pixels(&self) -> usize {
        self.max_tokens.saturating_mul(self.merge_unit().pow(2))
    }
}

/// The reference's `calc_size_preserved_ratio` without a longest-edge
/// bound: the `(width, height)` an image of `width x height` is resized to.
/// Every step is the reference's own `f32` operation, in its order.
///
/// A zero side yields `(0, 0)`, as in the reference.
#[must_use]
pub fn smart_resize(
    width: usize,
    height: usize,
    align: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    if width == 0 || height == 0 || align == 0 {
        return (0, 0);
    }
    let f = align as f32;
    // `static_cast<int>(std::round(x / f)) * f` and friends.
    let round_by = |x: f32| ((x / f).round() as usize).saturating_mul(align);
    let ceil_by = |x: f32| ((x / f).ceil() as usize).saturating_mul(align);
    let floor_by = |x: f32| ((x / f).floor() as usize).saturating_mul(align);
    let (w, h) = (width as f32, height as f32);
    let mut w_bar = align.max(round_by(w));
    let mut h_bar = align.max(round_by(h));
    let area = h_bar.saturating_mul(w_bar);
    if max_pixels > 0 && area > max_pixels {
        let beta = (h * w / max_pixels as f32).sqrt();
        h_bar = align.max(floor_by(h / beta));
        w_bar = align.max(floor_by(w / beta));
    } else if min_pixels > 0 && area < min_pixels {
        let beta = (min_pixels as f32 / (h * w)).sqrt();
        h_bar = ceil_by(h * beta);
        w_bar = ceil_by(w * beta);
    }
    (w_bar, h_bar)
}

/// One axis of Pillow's separable resampling (`precompute_weights`): for
/// every output index, the first input index, the tap count and the
/// 22-bit fixed-point weights (`ksize` per output, zero-padded).
struct AxisWeights {
    bounds: Vec<(usize, usize)>,
    weights: Vec<i32>,
    ksize: usize,
}

/// `PRECISION_BITS`: 32 - 8 (u8 samples) - 2 (accumulation headroom).
const PRECISION_BITS: u32 = 22;

fn precompute_weights(in_size: usize, out_size: usize, filter: ResizeFilter) -> AxisWeights {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let ksize = (support.ceil() as usize) * 2 + 1;
    let mut pre = vec![0.0f64; out_size * ksize];
    let mut bounds = Vec::with_capacity(out_size);
    let ss = 1.0 / filterscale;
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        // `static_cast<int>` truncates toward zero, then the clamp.
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).clamp(0, in_size as i64) as usize;
        let count = xmax.saturating_sub(xmin);
        let row = &mut pre[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0f64;
        for (x, w) in row.iter_mut().enumerate().take(count) {
            *w = filter.weight(((x + xmin) as f64 - center + 0.5) * ss);
            ww += *w;
        }
        if ww != 0.0 {
            for w in row.iter_mut().take(count) {
                *w /= ww;
            }
        }
        bounds.push((xmin, count));
    }
    let fxp = f64::from(1u32 << PRECISION_BITS);
    let weights = pre
        .iter()
        .map(|&w| {
            // Pillow adds +/- 0.5 and truncates toward zero.
            (w * fxp + if w < 0.0 { -0.5 } else { 0.5 }) as i32
        })
        .collect();
    AxisWeights {
        bounds,
        weights,
        ksize,
    }
}

/// `clip8(acc >> PRECISION_BITS)`.
#[inline]
fn clip8(acc: i64) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

/// The rounding bias every accumulator starts from (0.5 in fixed point).
const ROUND_BIAS: i64 = 1 << (PRECISION_BITS - 1);

/// Horizontal pass: `src` is `height` rows of `in_w` RGB pixels; the result
/// is `height` rows of `out_w`.
fn resample_horizontal(
    src: &[u8],
    in_w: usize,
    height: usize,
    out_w: usize,
    ax: &AxisWeights,
) -> Vec<u8> {
    let mut out = vec![0u8; out_w * height * 3];
    out.par_chunks_mut(out_w * 3)
        .zip(src.par_chunks(in_w * 3))
        .for_each(|(dst_row, src_row)| {
            for xx in 0..out_w {
                let (xmin, count) = ax.bounds[xx];
                let k = &ax.weights[xx * ax.ksize..xx * ax.ksize + count];
                let mut ss = [ROUND_BIAS; 3];
                for (x, &w) in k.iter().enumerate() {
                    let p = &src_row[(xmin + x) * 3..(xmin + x) * 3 + 3];
                    for c in 0..3 {
                        ss[c] += i64::from(p[c]) * i64::from(w);
                    }
                }
                for c in 0..3 {
                    dst_row[xx * 3 + c] = clip8(ss[c]);
                }
            }
        });
    out
}

/// Vertical pass: `src` is rows of `width` RGB pixels; the result has
/// `out_h` rows.
fn resample_vertical(src: &[u8], width: usize, out_h: usize, ax: &AxisWeights) -> Vec<u8> {
    let row_elems = width * 3;
    let mut out = vec![0u8; row_elems * out_h];
    out.par_chunks_mut(row_elems)
        .enumerate()
        .for_each(|(yy, dst_row)| {
            let (ymin, count) = ax.bounds[yy];
            let k = &ax.weights[yy * ax.ksize..yy * ax.ksize + count];
            for (i, dst) in dst_row.iter_mut().enumerate() {
                let mut acc = ROUND_BIAS;
                for (y, &w) in k.iter().enumerate() {
                    acc += i64::from(src[(ymin + y) * row_elems + i]) * i64::from(w);
                }
                *dst = clip8(acc);
            }
        });
    out
}

/// Pillow's separable resampling of `img` to `width x height` (the
/// reference's `resize_pillow`): horizontal pass first, then vertical; an
/// axis whose size does not change is skipped, and an unchanged image is
/// copied.
///
/// # Errors
///
/// [`ImageInputError::Empty`] for a zero source or target side.
pub fn resample(
    img: &ImageRgb8,
    width: usize,
    height: usize,
    filter: ResizeFilter,
) -> Result<ImageRgb8, ImageInputError> {
    if img.width == 0 || img.height == 0 || width == 0 || height == 0 {
        return Err(ImageInputError::Empty { width, height });
    }
    let need_h = width != img.width;
    let need_v = height != img.height;
    let data = match (need_h, need_v) {
        (false, false) => img.data.clone(),
        (true, false) => {
            let ax = precompute_weights(img.width, width, filter);
            resample_horizontal(&img.data, img.width, img.height, width, &ax)
        }
        (false, true) => {
            let ax = precompute_weights(img.height, height, filter);
            resample_vertical(&img.data, img.width, height, &ax)
        }
        (true, true) => {
            let ax_h = precompute_weights(img.width, width, filter);
            let ax_v = precompute_weights(img.height, height, filter);
            let tmp = resample_horizontal(&img.data, img.width, img.height, width, &ax_h);
            resample_vertical(&tmp, width, height, &ax_v)
        }
    };
    ImageRgb8::new(width, height, data).map_err(|_| ImageInputError::Empty { width, height })
}

/// The reference's `img_tool::resize` with `PAD_CEIL`: an image already
/// `width x height` is copied; otherwise it is scaled by `min(width / w,
/// height / h)` (in `f32`), resized to `ceil(side * scale)` (capped at the
/// box) with `filter`, and centred on a `pad`-coloured canvas at
/// `((width - new_w) / 2, (height - new_h) / 2)`.
///
/// # Errors
///
/// [`ImageInputError::Empty`] for a zero side.
pub fn resize_into_box(
    img: &ImageRgb8,
    width: usize,
    height: usize,
    filter: ResizeFilter,
    pad: [u8; 3],
) -> Result<ImageRgb8, ImageInputError> {
    if img.width == width && img.height == height {
        return Ok(img.clone());
    }
    if img.width == 0 || img.height == 0 || width == 0 || height == 0 {
        return Err(ImageInputError::Empty { width, height });
    }
    let scale_w = width as f32 / img.width as f32;
    let scale_h = height as f32 / img.height as f32;
    let scale = scale_w.min(scale_h);
    let new_w = ((img.width as f32 * scale).ceil() as usize).clamp(1, width);
    let new_h = ((img.height as f32 * scale).ceil() as usize).clamp(1, height);
    let resized = resample(img, new_w, new_h, filter)?;
    let mut canvas = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        canvas.extend_from_slice(&pad);
    }
    let off_x = (width - new_w) / 2;
    let off_y = (height - new_h) / 2;
    for y in 0..new_h {
        let src = &resized.data[y * new_w * 3..(y + 1) * new_w * 3];
        let dst = ((off_y + y) * width + off_x) * 3;
        canvas[dst..dst + new_w * 3].copy_from_slice(src);
    }
    ImageRgb8::new(width, height, canvas).map_err(|_| ImageInputError::Empty { width, height })
}

/// An image brought to the tower's geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedImage {
    /// The resized image: both sides multiples of the merge unit.
    pub image: ImageRgb8,
    /// Its merged grid (`height / 32 x width / 32`).
    pub grid: GridSize,
    /// The source size, `(width, height)`.
    pub source: (usize, usize),
}

/// Bring `img` to the tower's geometry exactly as the reference does (see
/// the module docs): smart-resize the box, resize into it, and check the
/// merged grid against the budget.
///
/// # Errors
///
/// [`ImageInputError::Empty`] for a zero side;
/// [`ImageInputError::TooManyTokens`] when even the smallest grid the
/// aspect-preserving resize allows exceeds `cfg.max_tokens`.
pub fn prepare_image(
    img: &ImageRgb8,
    cfg: &PreprocessConfig,
) -> Result<PreparedImage, ImageInputError> {
    if img.width == 0 || img.height == 0 {
        return Err(ImageInputError::Empty {
            width: img.width,
            height: img.height,
        });
    }
    let unit = cfg.merge_unit();
    let (w, h) = smart_resize(
        img.width,
        img.height,
        unit,
        cfg.min_pixels(),
        cfg.max_pixels(),
    );
    let grid = GridSize {
        h: h / unit,
        w: w / unit,
    };
    let tokens = grid.h.saturating_mul(grid.w);
    if tokens > cfg.max_tokens || tokens == 0 {
        return Err(ImageInputError::TooManyTokens {
            width: img.width,
            height: img.height,
            grid_h: grid.h,
            grid_w: grid.w,
            tokens,
            max_tokens: cfg.max_tokens,
        });
    }
    let image = resize_into_box(img, w, h, cfg.filter, cfg.pad_color)?;
    Ok(PreparedImage {
        image,
        grid,
        source: (img.width, img.height),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("test vector hex"))
            .collect()
    }

    fn cfg(max_tokens: usize) -> PreprocessConfig {
        PreprocessConfig::qwen_vl(16, 2, max_tokens).expect("config")
    }

    /// The reference's box for the golden fixture and its neighbours.
    #[test]
    fn smart_resize_matches_the_reference_arithmetic() {
        let c = cfg(DEFAULT_IMAGE_MAX_TOKENS);
        let unit = c.merge_unit();
        let (min_p, max_p) = (c.min_pixels(), c.max_pixels());
        // The 256 x 192 fixture is already on the grid: 8 x 6 merged.
        assert_eq!(smart_resize(256, 192, unit, min_p, max_p), (256, 192));
        // Rounding to the nearest multiple of 32, half away from zero
        // (208 / 32 = 6.5 -> 7).
        assert_eq!(smart_resize(250, 190, unit, min_p, max_p), (256, 192));
        assert_eq!(smart_resize(271, 208, unit, min_p, max_p), (256, 224));
        assert_eq!(smart_resize(272, 208, unit, min_p, max_p), (288, 224));
        // Too small: grown to >= 8 tokens, `ceil` to 32 on each side
        // (sqrt(8192 / 400) = 4.5254 -> 90.5 -> 96).
        assert_eq!(smart_resize(20, 20, unit, min_p, max_p), (96, 96));
        // Too large: shrunk to <= 1024 tokens (4000 x 3000 ->
        // beta = sqrt(12e6 / 1048576) = 3.3830 -> 1182.4 x 886.8 -> 1152 x 864).
        assert_eq!(smart_resize(4000, 3000, unit, min_p, max_p), (1152, 864));
        // A side never falls below 32 (beta = 1.7469: 100 000 -> 57 216).
        assert_eq!(smart_resize(32, 100_000, unit, min_p, max_p), (32, 57_216));
        assert_eq!(smart_resize(0, 5, unit, min_p, max_p), (0, 0));
    }

    /// A tiny case checkable by hand: 4 -> 2 bicubic along one axis.
    /// `scale = 2`, `support = 4`, `ksize = 9`; output 0 is centred at 1.0
    /// and reads inputs 0..=3 at filter distances -0.25, 0.25, 0.75, 1.25.
    /// Keys with `a = -0.5`: `(1.5|x| - 2.5)x^2 + 1` below 1 gives
    /// 0.8671875 (twice) and 0.2265625; `(((x - 5)x + 8)x - 4)a` at 1.25
    /// gives -0.0703125. Normalised by their sum 1.890625 and scaled by
    /// 2^22 with Pillow's round-half-away: 1 923 834, 1 923 834, 502 623,
    /// -155 987.
    #[test]
    fn a_bicubic_weight_row_matches_the_hand_computed_one() {
        let ax = precompute_weights(4, 2, ResizeFilter::Bicubic);
        assert_eq!(ax.ksize, 9);
        assert_eq!(ax.bounds[0], (0, 4));
        assert_eq!(&ax.weights[..4], &[1_923_834, 1_923_834, 502_623, -155_987]);
        assert!(ax.weights[4..9].iter().all(|&w| w == 0));
        // A flat row stays flat (the weights are normalised).
        let img = ImageRgb8::new(4, 1, vec![100; 12]).expect("image");
        let out = resample(&img, 2, 1, ResizeFilter::Bicubic).expect("resample");
        assert_eq!(out.data, vec![100; 6]);
    }

    /// Pillow's own `Image.resize(.., BICUBIC / BILINEAR)` output, byte for
    /// byte: the reference's `resize_pillow` is a port of the same
    /// `Resample.c`.
    #[test]
    fn resample_is_byte_identical_to_pillow() {
        for (name, src, dst, (sw, sh, tw, th), filter) in [
            (
                "down",
                RESIZE_BICUBIC_DOWN_SRC,
                RESIZE_BICUBIC_DOWN_DST,
                RESIZE_BICUBIC_DOWN_DIMS,
                ResizeFilter::Bicubic,
            ),
            (
                "up",
                RESIZE_BICUBIC_UP_SRC,
                RESIZE_BICUBIC_UP_DST,
                RESIZE_BICUBIC_UP_DIMS,
                ResizeFilter::Bicubic,
            ),
            (
                "w only",
                RESIZE_BICUBIC_W_ONLY_SRC,
                RESIZE_BICUBIC_W_ONLY_DST,
                RESIZE_BICUBIC_W_ONLY_DIMS,
                ResizeFilter::Bicubic,
            ),
            (
                "h only",
                RESIZE_BICUBIC_H_ONLY_SRC,
                RESIZE_BICUBIC_H_ONLY_DST,
                RESIZE_BICUBIC_H_ONLY_DIMS,
                ResizeFilter::Bicubic,
            ),
            (
                "big down",
                RESIZE_BICUBIC_BIG_DOWN_SRC,
                RESIZE_BICUBIC_BIG_DOWN_DST,
                RESIZE_BICUBIC_BIG_DOWN_DIMS,
                ResizeFilter::Bicubic,
            ),
            (
                "bilinear",
                RESIZE_BILINEAR_DOWN_SRC,
                RESIZE_BILINEAR_DOWN_DST,
                RESIZE_BILINEAR_DOWN_DIMS,
                ResizeFilter::Bilinear,
            ),
        ] {
            let img = ImageRgb8::new(sw, sh, hex(src)).expect("source");
            let out = resample(&img, tw, th, filter).expect("resample");
            assert_eq!(out.data, hex(dst), "{name}");
        }
    }

    /// `PAD_CEIL`: a 250 x 190 image goes into the 256 x 192 box at scale
    /// min(256/250, 192/190) = 1.0105, i.e. resized to 253 x 192 and centred
    /// 1 pixel from the left on black.
    #[test]
    fn resize_into_box_preserves_the_aspect_ratio_and_pads_with_black() {
        let img = ImageRgb8::new(250, 190, vec![200; 250 * 190 * 3]).expect("image");
        let out = resize_into_box(&img, 256, 192, ResizeFilter::Bicubic, [0, 0, 0]).expect("box");
        assert_eq!((out.width, out.height), (256, 192));
        let px = |x: usize, y: usize| out.pixel(x, y).expect("in range");
        assert_eq!(px(0, 0), [0, 0, 0]);
        assert_eq!(px(1, 0), [200, 200, 200]);
        assert_eq!(px(253, 191), [200, 200, 200]);
        assert_eq!(px(254, 100), [0, 0, 0]);
        assert_eq!(px(255, 100), [0, 0, 0]);
        // Already the box size: copied untouched.
        let exact = ImageRgb8::new(32, 32, (0..32 * 32 * 3).map(|i| (i % 251) as u8).collect())
            .expect("image");
        let same = resize_into_box(&exact, 32, 32, ResizeFilter::Bicubic, [0, 0, 0]).expect("copy");
        assert_eq!(same, exact);
    }

    #[test]
    fn prepare_image_keeps_the_fixture_geometry_and_enforces_the_budget() {
        let fixture = ImageRgb8::new(256, 192, vec![7; 256 * 192 * 3]).expect("image");
        let prepared = prepare_image(&fixture, &cfg(1024)).expect("prepare");
        assert_eq!(prepared.grid, GridSize { h: 6, w: 8 });
        assert_eq!(prepared.grid.n_tokens(), 48);
        assert_eq!(prepared.image, fixture, "an on-grid image is not resampled");

        // A budget below the natural grid downscales (never upscales)...
        let small = prepare_image(&fixture, &cfg(12)).expect("prepare");
        assert!(small.grid.n_tokens() <= 12, "{:?}", small.grid);
        assert!(small.image.width < 256 && small.image.height < 192);

        // ...and refuses only when even the minimum grid does not fit.
        let strip = ImageRgb8::new(32, 4000, vec![1; 32 * 4000 * 3]).expect("image");
        let err = prepare_image(&strip, &cfg(8)).expect_err("too elongated");
        assert_eq!(err.code(), "image_too_many_tokens");
        assert!(err.to_string().contains("--image-max-tokens"), "{err}");

        // The budget bounds themselves.
        assert!(PreprocessConfig::qwen_vl(16, 2, 0).is_err());
        assert!(PreprocessConfig::qwen_vl(16, 2, MAX_IMAGE_MAX_TOKENS + 1).is_err());
        assert_eq!(cfg(4).min_tokens, 4);
        assert_eq!(cfg(1024).min_tokens, QWEN_VL_MIN_IMAGE_TOKENS);
    }

    // ── Pillow reference vectors ─────────────────────────────────────────
    const RESIZE_BICUBIC_DOWN_SRC: &str = "82223065c281ff1c6bfa2ef15324d1e446a9eb68bee0322f4b61966dd7a47781d446c7a6141a59e4000abb7dcfc0daa7\
         47ab7ab683bd7a08167105fb7dd700f93bd665d06a8169f3a5ec704a678b3b396baefb73f5f87ff6533e5ba2bdcd737b\
         636fc8c21bebdf002c";
    const RESIZE_BICUBIC_DOWN_DST: &str =
        "845b52b16fa57857c6bc3578b288a675b4a46d527e726a5ada9a8290949e7c6ea2aa4d7e";
    const RESIZE_BICUBIC_DOWN_DIMS: (usize, usize, usize, usize) = (7, 5, 4, 3);
    const RESIZE_BICUBIC_UP_SRC: &str = "c97d8735922b7277ad349e23c1dba2ff05bb";
    const RESIZE_BICUBIC_UP_DST: &str = "e37898968163298c204885766b7fb6b1817688915d52a144767788935db84c93316db252a4cc89d05dabe318bc1a9c0f\
         5fc24ccde1adfe4fbdff00be";
    const RESIZE_BICUBIC_UP_DIMS: (usize, usize, usize, usize) = (3, 2, 5, 4);
    const RESIZE_BICUBIC_W_ONLY_SRC: &str = "568be7f81f2f67068b6f9f8898ea54ca3c461144d917e2b58bcd95f944aff83119a57ae0f65b8ee15dfb8267e2cc59de\
         a44a672c9962886de8632dd531f25aa4e07038c8bd3444e760e249b0b7e26757a4";
    const RESIZE_BICUBIC_W_ONLY_DST: &str =
        "9f3e88847c707d727a3ebdb3e94574d864bab55ce47a6c766966c763d7814a99b0889aac";
    const RESIZE_BICUBIC_W_ONLY_DIMS: (usize, usize, usize, usize) = (9, 3, 4, 3);
    const RESIZE_BICUBIC_H_ONLY_SRC: &str = "3787bb336f87c77f7e2c62f7fb06cd128f8877deece8eb3b0ad9bc6bf03dca90fbc3a26198abc20d77ff98e4da25767d\
         acf879a091311dfd003060571a64a421526b9b3d6195b803d58fe973ea8c59bede";
    const RESIZE_BICUBIC_H_ONLY_DST: &str = "3376de92439f6b89876ad4b6e9a59441be9975af905ca2e7afc29417b33568885e587e5e8079a88093766fb97d";
    const RESIZE_BICUBIC_H_ONLY_DIMS: (usize, usize, usize, usize) = (3, 9, 3, 5);
    const RESIZE_BICUBIC_BIG_DOWN_SRC: &str = "c3a9d3e16f70cbe3203ee6296f04ceb386683a94dfbae6cb7b041d40d501cbaf4b097671caeb8175353c47bc0432c719\
         71c78f9b4ffb64e169ad3b1dbf1ff5884bbfa8a35d4bf59662d5164a4dd273a8a9227db8ebf6371fa2ddd6834485dc20\
         52cae03837e9da5dd0534b38de50b5674d48768b0d73b23d165f62cd7b92734e4a5df79d81a267f376b51edbd0c9ee3c\
         e1394088c7dfa6e5364774feac0754ca57aba275e2bb6a9f593bd91570d267a1efc2be89bb72da276c47bef8f2ddbfc5\
         e2d1dd8d4d1cfe4a5b1f4a2e655ccc0b524b8a38de163deb3db2741c93323cda6a4de9b2d91a814b601630133839007b\
         1497f1def8f6af923b883b91b63dffb533663323c2d687a6d06a7aac6807352a16d635f699087ce20441193ce6620f7f\
         a84091c57507f3c0f43cb6dc84b477f3021ca2f02ed47ca4640fe62867cf5259d784a01f805eeded9754554657745be7\
         edfb4f9a47e30ff241b2f8c36c0ebb610971b61c18e8584a5b0d546574d37ae77a21219aa2050a1bf6462dd3c577de8a\
         ea113cb0f4a7457750d4b463dba2cbcfdbd05c7c3c78c7fcaff54cd09bf7c7180a8381ef667cf181d497110334c8f5cd\
         c50ad0c4968c3421c4e7827b32a0d124fc3746b5d14538e69a5652644555c2a1edbce81c700263a307ab4dd18181cee9\
         a2fc24183d9926b028334a26ff24a07c2b1fbe223956a13adbd1fa9f039b05034df532b1bafe2515d915d5e56e0aaf14\
         4832a70893be139af8ff3590d46db07680917638f079151644d1a74a56f1a69534b46286ecdfa3f48d7c7b746b5e6547\
         fcd43d16e818d71c30e20106e4db835053219946a9a32db890f99786a0d6af7160cbba18b9de648d44b998070431339c\
         3550dddfa9d654c5c3299849259f1064783099e52ac9b198d9472db339de4a090fe7e8ceb0b3e4c083a716c39b9d6da7\
         7b17b919c60ac2d4d1070395dbaff3938ef004cc85c4c7208f841bcd9028f4843006cc427096c9bda5a524f72890ba3d\
         c5fc2f2d011f2e63407f88ec43db4a578c7e12063bedf0d102f251993c757a3e9ec9c83d5520dda12f51cd42760a2fd3\
         dd493785ff7534d125af11584c86c53b0964cefcb7c6727409077cc027b8a0597333144d265b8a4564bf8b6c6d948626\
         beb958fcfe9a32b554abe090372b025b17e0a01fa6307fde1a7f74718cced770241ff5894140462a1e15d7d46f2a0f1e\
         95fa17277f650d30e3f0a2866cb65d701811acca5fc709e4b530789be9180a6c974d65f634a44c2406cd6f14e1360a2e\
         b3f60754d2481c0566ac448a6faf68b62d67e6b9b46b401361c0a1514c2a2465c605a23e96d1affaf563d1dc9ce4cfb5\
         511aa25ffcb3acc4be608a329a08ba7cf4d61c47e2bacd955c16e9390f86577f4a0059284badc7da7258fe1ffacb1f16\
         68e644903a3f72b3e14dbbe59ebbd679c4f3937f7bf0fd0bff111e30af3e3dcf689b72e4a099dccf9fd82cdb78726395\
         d80e8425ed3248efeb50badb6579037d2c9a3749b5336300c527f29855f6b4dfedbbaeb8dffb5d4a982168166ecbe0bf\
         c40934382eb0cabdd8e2ae8d44a492d28b09acda0e916c3b3f83713350c57935b949f3e1181a9af53c8c6e0bc0e45d66\
         4a8cb205c44e67d5ff48dc83a4f510df71f66b1da66894628d390468c15f4b64841afa858eb9dd6903698276804719bb\
         42bd7c95f80c2242e12a3b484cb27805a3e394f2985d64a6ca37de87aa2fd1c1992e886aa2922ac67325c03953949902\
         c1008eccef9bfed8bc72a5b3c256acb8ceb1d7cae22b322f7a88ad6b43b0ab84943284b24b70f2b8208bd130af2c53af\
         106ddbcffe0464d1380110fe650ef2a2ada6faaad105a39851ceb41e4e0aacd1c042534b11f14ce9f1244ecb4699c011\
         7ff6fb782a6e578f59c2813b4f2085bff8afdebda8fb038b384093004e1a2485364597e2d3ab57af77c2d79f478ca513\
         a792f9ee6c5591b1de30c6bef81e1812274fce192de9704991236ffbe998e1c3ca80fa8bdb963356d330b7a2f784fff8\
         539431d5f804e8560e974e35e74cd9d0509ea9ad771615872a5be83f7a1c8ae8ec395cd0f36bd839c89cf58f219e88f4\
         b4af66b4430a8025d739614b660447736acedfacc8bda455b232a0f1b01826d1bf73b8792a79450cf68036aff0f7d7b0\
         a68dcfc32bcddc3959d49cff6723f97593b902873830d05e204c164439749e8d3d2b64ed612b9904c116c640f1e1c720\
         cdf6b5cdc359e748573d77019e3d0e74041bfba04e529cadec8984057b9043174d1e58c48d98fb83c46600901c468da0\
         b9b2652e2a27fa74a5be1282f0c46f9dd23a6533f8c3b2e4ca7082f13a9775beabbde492eaac5e61091cdce9e6833f85\
         1e60d88f653208b60bd7f9cfd8bd492f66ceab3eb701149928ba6d06fc370ce9227f902f1d7d032cd85fb8a80799a841\
         249ba31c04133c38e01f2c980a043d90a5c973109f4d754dff218abb1a06a04e76a9002eb3b6086343a88aed6beb0493\
         9b277381cbf0f1b929adfebcc4d261973a80c490d99fc7b362ae6b1bc8ba0e1a7e1a084c3dec439186428bebb77ee09f\
         af2790f6f9c86bbd011042d25cfe743d3499ad61d3a560c6424bb4f0418ffc9c021f43bd010f2d775718c16f9628a724\
         52ac7290586916e33d6d87df907d8fcd62ccc95555ec3831e10e2f4a3d2c27408067d4ec37492e94e530b7e7fb3eff74\
         1c3dfb0b1bf2d2fa16e8b3a3dc66909c909e22c292fdc65515ec8f5682ec9c5a50bf5ef9abfc9e36c5bbbb1cbe1f81a0\
         04e14115d8403e44785dcff611c810bab400d3316da35e3dfb1f2cbfc7835143ada6291b23471f458295d15cac36e563\
         4c3eb0f463f60f6ffc55b12ecc0083b9a39f0f1107fa854d9baec566176fc16defaba8322c8f31d05519d2a403087aee\
         514dd9de9a07218d40f5d18c915f73b033034362af398498a97abb63ea36156fd8cceef805d1446002f09103d03dd51c\
         bfc937a1dba6ddd73c34501b62e9ed68ba60860f823e7573381d2f96a66bcd61644a20d291e5bae1a513090f49064fa3\
         44ea9f236d075ebc502ba428eeef1abac08a5d57a7973ee1d863a35aca02c45cb01315fcb80b7995f39a4b10126ddecc\
         8af4af22619df65ced9ab91429cb010eb54f11716817f044e6c086747f4ebff57f552043dfd30bfefc469d16a090bcf0\
         4f1ff0d57ae6c9ebb8d0326e8176b41b841edc566cfe1b57efe5c319169523ae478bf103873a47e262fa656ee1f68366\
         909b695cae1df892af06f37af7a7cc202dc311fd9856e678861bbe77d98ed297b216a516dc72a1f269c7d8d41290f9c0\
         cc8362def34bc697e2614d7bc010139588a04f55227441baeb598d6ce3557eea80501f8e627e6c0a3ca624895a420b07\
         18a701ed6031ae6424ed307c1b1b11157fdbc18a8f895919bc33ea054dd7a97ed745c5ad0702733be5f3328fcacd1c44\
         bccbb2d49008da84f1416c01641d3c44ffcdc2622f6bb2f46e29243de273322a18549a766cf8d7f74728c66975889c0e\
         5cb8dfe4791ccc069682de6b8e596741875e45cb234c8db2c482519d8d071e637c846006675847aa18d426f915b337ad\
         7c6697555ba790a3f22601122307fe033cb078115aef24ef8cf5f1171cb3ab68af0a685d63de8a93dc61b99f3022c6dd\
         d6c755dae1760faa9c66d9cf0028236f71766111978283019bc2f98e04df2793a7078215b883fe92e8e109e4e9ca14be\
         ab0bd47e648a6b32e8b97ea221cc1ca0ed9cd454307bedf16d53da16629b716713ada2dcfaf7b75b64a4c400619ec67a\
         410d6a1818ded04de95f6b2a1082400b45c8d0cd07e091d023e7cd38a9e872d84934f1707bc6a1958527afb2a5e72108\
         1f51116ec0b7ebc639d391594e182954d91f15c0d88ee15e73fa1a9aa5a701001337c32f518dee9c5956af324594293e\
         0a4809707dc13ed1ff63a50b389826b841f49716419c5d1364193be58b189009c8753c337a44a67b3f0835f2ff9542a6\
         1ea71ccd2927ba7a4cd756691a231e672fa2ec7598e1f68708ed962ae943b72220de80112c6f2413a5908a4bbd0e6802\
         769cd2d416e733d52d1f63e392a0694cb87132146afd7a999beafbd59f82b226d61f5182df0ae18bfaecbd180f097175\
         03a8f4be6cd6ed8d6a4b217d134ffa038919ca7a5ca37c174f97a75491674b0a88a7dcb623adb69cc88423c8ded3aa68\
         6fc3a4962972230c1d28fc58b1476af047862aff364f3500c78ca338f56d170a265629981f91e9545ed4d23d02324e42\
         3dd5241ca4fc1dc2efd422ccde012005c7fded7215f99f482b388948afb5e4accd5365cc89b240960c66c58fdc66a1c2\
         84e01bcd29a88b1eecad0750f89d22ea127f8b84bdf3e3cb38fba4044396cbfe96dc94ced134f27f6ce18274a08c3907\
         285f1ff3c7ccc56c28321bd8543a93eae1e9bb526b6db154f6acadefc2a40987a1b0aa2b583ef6e008a387f6d08506d3\
         5fcdbae46ebe14359d99eff692613a8afcde12a874ea6956344227779009c4ab6fe85ab4b96c05c7c60ec669340e90fb\
         18537420d89e73d4e6051bf3626b095707a18589646e8edb600ffa6f501a96074473216a7bbdbf607cc14ce46b1e2204\
         276b278898c6a31a15c1d4e0559ace5bb28e31d1d7a65534c89b51aa767e8dce3705fd8afd90a1a5094b18ec3c845401\
         9e7218818fb240dc8344571c71088f9be6a8e7cd04658ddcc20258cc009d6a2fa2372e1b5bf7b8a961e7c1976d7185ba\
         6ad23e241c0b5f3a8ca7ad3694714c92407db714013747dbcd88807a660bd2edecc60a33139a392e1dd651b48decbbfc\
         d6c1ebfe3ac0b12eb41b51e4fe5bed42994070c7d841e1b4c45bd7c3be6ad85dd5f19d906136116d405c6942be4bef43\
         804836587cdfb31801b2a5c3a1ff19826b87889bae0840e34411953c0fd2e9e36cef86599a9632d5a460e3965dd0a60f\
         818e3ec065cde985b1368bf69c77d62367e363b40da5d89fffcfd2ff47edd26bd29fc6b84c6e1974e2c4fd3bd4eae3a5\
         e329bb5b7c79db23d88d149820f7c0b49ef93d542358e8b42932ac6d169d0b2650444af9838a3e35d25d71650cf54e20";
    const RESIZE_BICUBIC_BIG_DOWN_DST: &str = "81847f8d7f81967973807c8f9684a08b7f99747f8d798d83835b68738b75727175787c7b8f7a6b797ba1766c80827f8f\
         82995a7d858376917c798e858882779e74687c7c9e8f8882817c8f8d898b798e9d67807281978475976c737d7e556779\
         9074608f77759f72928f93827a835b8f7d81747a996e859b7694758aaa6e857f6a8c768c6975726b737977777f74759e\
         875c88706e827891778884787e90746c82757961748b7c6d848096958ca68d8b9b92807b947a78696f8e7c7786";
    const RESIZE_BICUBIC_BIG_DOWN_DIMS: (usize, usize, usize, usize) = (40, 30, 9, 7);
    const RESIZE_BILINEAR_DOWN_SRC: &str = "8284b9d2e1811f66c32f7c4070c30dd93292a76652d0ab36c8017ff19a9a4f655c41edf781aaf860c017052e865de4f9\
         49b15ae48323bed4dca2b83ca3b1535dbcabc23a02c96aa9ce0f332183b0411fb952c588bcdc85c5798e0fd80251398a\
         72a0be77aaf579537d";
    const RESIZE_BILINEAR_DOWN_DST: &str =
        "b48b83797b8b65a376997f74677a8e99896f92a19983aa6da69c757e7553626ea8637cab";
    const RESIZE_BILINEAR_DOWN_DIMS: (usize, usize, usize, usize) = (7, 5, 4, 3);
}
