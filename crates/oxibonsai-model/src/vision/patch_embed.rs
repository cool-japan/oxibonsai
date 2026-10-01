//! Patch embedding of the Qwen3-VL vision tower: normalisation,
//! patchification with both temporal kernel slices, 2 x 2 merge-window
//! ordering and the resized learned position embedding (bonsai2-design.md
//! §6.2; the fork's `clip_graph_qwen2vl::build_inp_with_temporal_merge` and
//! the head of `clip_graph_qwen3vl::build`).
//!
//! # Both temporal slices
//!
//! Qwen-VL patchifies video as `2 x 16 x 16` space-time patches, and a
//! still image is treated as a two-frame clip of the same picture: the fork
//! convolves the image with `v.patch_embd.weight` and with
//! `v.patch_embd.weight.1` and adds the two results. Convolution is linear
//! in the kernel, so this module sums the two slices once at load and runs
//! a single `[n_patches x 3p^2] . [hidden x 3p^2]^T` GEMM per image.
//!
//! # Merge-window order
//!
//! Before the first block the patches are permuted so that the four
//! patches of every 2 x 2 merge window are consecutive: windows in
//! row-major order over the merged grid, and inside a window
//! `(dy, dx) = (0,0), (0,1), (1,0), (1,1)`. This is what the reference's
//! permute/reshape sequence does, and it is why the merger can later read
//! four consecutive rows as one 4 * hidden token. The attention is global
//! and the positions travel with the tokens (the 2-D RoPE table is built in
//! the same order), so the order changes nothing inside the blocks.
//!
//! # Position embedding
//!
//! `v.position_embd.weight` holds one row per cell of a square grid (48 x 48
//! for Bonsai 2, raster order). For an image whose patch grid differs it is
//! resized with ggml's bilinear, align-corners `upscale` kernel —
//! [`resize_position_grid`] reproduces that kernel's `f32` arithmetic —
//! then permuted into window order and added.

use oxibonsai_kernels::KernelDispatcher;
use rayon::prelude::*;

use super::clip_loader::{VisionConfig, SPATIAL_MERGE};
use super::ImageRgb8;
use crate::error::{ModelError, ModelResult};

/// Patch coordinates `(row, column)` of every token, in 2 x 2 merge-window
/// order: windows row-major over the merged grid, and inside each window
/// `(0,0), (0,1), (1,0), (1,1)`.
///
/// `grid_h` and `grid_w` are the patch-grid sides and must both be even
/// (the caller's geometry check guarantees it); an odd trailing row or
/// column would simply not be visited.
#[must_use]
pub fn window_order(grid_h: usize, grid_w: usize) -> Vec<(usize, usize)> {
    let mut order = Vec::with_capacity(grid_h.saturating_mul(grid_w));
    for wy in (0..grid_h / SPATIAL_MERGE).map(|i| i * SPATIAL_MERGE) {
        for wx in (0..grid_w / SPATIAL_MERGE).map(|i| i * SPATIAL_MERGE) {
            for dy in 0..SPATIAL_MERGE {
                for dx in 0..SPATIAL_MERGE {
                    order.push((wy + dy, wx + dx));
                }
            }
        }
    }
    order
}

/// Resize a square `side x side` grid of `hidden`-wide rows (raster order,
/// row `y * side + x`) to `grid_h x grid_w` with ggml's bilinear,
/// align-corners interpolation, returning raster order again.
///
/// This reproduces `ggml_compute_forward_upscale_f32`'s
/// `GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS` branch in the
/// same `f32` arithmetic: scale factor `(out - 1) / (in - 1)` (or `out / in`
/// when either side is 1), source coordinate `i / scale`, the two
/// neighbours clamped to the grid, the fraction measured from the clamped
/// lower neighbour and clamped to `[0, 1]`, and the four-term weighted sum
/// in the kernel's order. An identity resize returns the grid unchanged.
///
/// # Errors
///
/// [`ModelError::ShapeMismatch`] when `pos.len() != side * side * hidden`,
/// [`ModelError::ShapeInvariant`] for an empty grid.
pub fn resize_position_grid(
    pos: &[f32],
    side: usize,
    hidden: usize,
    grid_h: usize,
    grid_w: usize,
) -> ModelResult<Vec<f32>> {
    let expected = side.saturating_mul(side).saturating_mul(hidden);
    if pos.len() != expected {
        return Err(ModelError::ShapeMismatch {
            name: "position embedding grid".to_string(),
            expected: vec![side * side, hidden],
            actual: vec![pos.len()],
        });
    }
    if side == 0 || grid_h == 0 || grid_w == 0 || hidden == 0 {
        return Err(ModelError::ShapeInvariant {
            tensor: "position embedding grid".to_string(),
            expected: "non-empty source and target grids".to_string(),
            actual: format!("{side}x{side} -> {grid_h}x{grid_w}, width {hidden}"),
        });
    }
    if grid_h == side && grid_w == side {
        return Ok(pos.to_vec());
    }
    let axis = |out: usize| {
        let scale = if out > 1 && side > 1 {
            (out - 1) as f32 / (side - 1) as f32
        } else {
            out as f32 / side as f32
        };
        let last = side as i64 - 1;
        (0..out)
            .map(|i| {
                let s = i as f32 / scale;
                let lo = s.floor() as i64;
                let lo_c = lo.clamp(0, last);
                let hi_c = (lo + 1).clamp(0, last);
                let frac = (s - lo_c as f32).clamp(0.0, 1.0);
                (lo_c as usize, hi_c as usize, frac)
            })
            .collect::<Vec<_>>()
    };
    let xs = axis(grid_w);
    let ys = axis(grid_h);
    let mut out = vec![0.0f32; grid_h * grid_w * hidden];
    out.par_chunks_mut(hidden)
        .enumerate()
        .for_each(|(cell, dst)| {
            let (y0, y1, dy) = ys[cell / grid_w];
            let (x0, x1, dx) = xs[cell % grid_w];
            let a = &pos[(y0 * side + x0) * hidden..][..hidden];
            let b = &pos[(y0 * side + x1) * hidden..][..hidden];
            let c = &pos[(y1 * side + x0) * hidden..][..hidden];
            let d = &pos[(y1 * side + x1) * hidden..][..hidden];
            for (i, value) in dst.iter_mut().enumerate() {
                *value = a[i] * (1.0 - dx) * (1.0 - dy)
                    + b[i] * dx * (1.0 - dy)
                    + c[i] * (1.0 - dx) * dy
                    + d[i] * dx * dy;
            }
        });
    Ok(out)
}

/// Normalise an RGB8 image to planar `f32` (`[3][height][width]`) with the
/// fork's two steps: `x / 255`, then `(x - mean[c]) / std[c]`.
#[must_use]
pub fn normalize_planar(img: &ImageRgb8, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let plane = img.width * img.height;
    let mut out = vec![0.0f32; 3 * plane];
    let (r, rest) = out.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    let planes = [r, g, b];
    for (c, dst) in planes.into_iter().enumerate() {
        dst.par_iter_mut()
            .zip(img.data.par_chunks_exact(3))
            .for_each(|(value, px)| {
                *value = (f32::from(px[c]) / 255.0 - mean[c]) / std[c];
            });
    }
    out
}

/// The patch-embedding stage of the tower: kernel, bias and position grid.
#[derive(Debug, Clone)]
pub(crate) struct PatchEmbed {
    /// `[hidden x 3 * p * p]`: the sum of the two temporal kernel slices,
    /// each row ordered channel, kernel row, kernel column.
    kernel: Vec<f32>,
    /// `[hidden]`.
    bias: Vec<f32>,
    /// `[pos_grid^2 x hidden]`, raster order.
    pos_embd: Vec<f32>,
    pos_grid: usize,
    patch: usize,
    hidden: usize,
}

impl PatchEmbed {
    /// Build the stage from the two kernel slices (each `[hidden x 3p^2]`),
    /// the bias and the stored position grid.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when a buffer's length disagrees with
    /// `cfg`.
    pub(crate) fn new(
        cfg: &VisionConfig,
        slice0: Vec<f32>,
        slice1: &[f32],
        bias: Vec<f32>,
        pos_embd: Vec<f32>,
    ) -> ModelResult<Self> {
        let kernel_len = cfg.hidden * cfg.patch_len();
        for (name, len, want) in [
            ("v.patch_embd.weight", slice0.len(), kernel_len),
            ("v.patch_embd.weight.1", slice1.len(), kernel_len),
            ("v.patch_embd.bias", bias.len(), cfg.hidden),
            (
                "v.position_embd.weight",
                pos_embd.len(),
                cfg.pos_grid * cfg.pos_grid * cfg.hidden,
            ),
        ] {
            if len != want {
                return Err(ModelError::ShapeMismatch {
                    name: name.to_string(),
                    expected: vec![want],
                    actual: vec![len],
                });
            }
        }
        let mut kernel = slice0;
        kernel
            .par_iter_mut()
            .zip(slice1.par_iter())
            .for_each(|(k0, k1)| *k0 += *k1);
        Ok(Self {
            kernel,
            bias,
            pos_embd,
            pos_grid: cfg.pos_grid,
            patch: cfg.patch_size,
            hidden: cfg.hidden,
        })
    }

    /// Bytes held by the stage's buffers.
    pub(crate) fn resident_bytes(&self) -> usize {
        std::mem::size_of_val(self.kernel.as_slice())
            + std::mem::size_of_val(self.bias.as_slice())
            + std::mem::size_of_val(self.pos_embd.as_slice())
    }

    /// Embed a normalised planar image (`[3][height][width]`) into
    /// `[n_patches x hidden]` rows in merge-window order, patch bias and
    /// position embedding included.
    ///
    /// `width` and `height` must be multiples of `2 * patch` (the caller
    /// validates the geometry first).
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when `planar` does not hold
    /// `3 * width * height` values, [`ModelError::ShapeInvariant`] for a
    /// size that is not a whole number of merge windows, and the GEMM's
    /// own shape errors.
    pub(crate) fn embed_planar(
        &self,
        planar: &[f32],
        width: usize,
        height: usize,
        dispatcher: &KernelDispatcher,
    ) -> ModelResult<Vec<f32>> {
        let p = self.patch;
        let unit = p * SPATIAL_MERGE;
        if width == 0 || height == 0 || !width.is_multiple_of(unit) || !height.is_multiple_of(unit)
        {
            return Err(ModelError::ShapeInvariant {
                tensor: "image".to_string(),
                expected: format!("non-empty sides that are multiples of {unit}"),
                actual: format!("{width} x {height}"),
            });
        }
        // `planar` exists in memory, so a size whose value count overflows
        // cannot match it: report the mismatch instead of overflowing.
        let plane = width.saturating_mul(height);
        if Some(planar.len()) != plane.checked_mul(3) {
            return Err(ModelError::ShapeMismatch {
                name: "normalised image (planar RGB)".to_string(),
                expected: vec![3, height, width],
                actual: vec![planar.len()],
            });
        }
        let (grid_h, grid_w) = (height / p, width / p);
        let order = window_order(grid_h, grid_w);
        let n = order.len();
        let patch_len = 3 * p * p;

        let mut patches = vec![0.0f32; n * patch_len];
        patches
            .par_chunks_mut(patch_len)
            .zip(order.par_iter())
            .for_each(|(row, &(py, px))| {
                for c in 0..3 {
                    for ky in 0..p {
                        let src = c * plane + (py * p + ky) * width + px * p;
                        let dst = c * p * p + ky * p;
                        row[dst..dst + p].copy_from_slice(&planar[src..src + p]);
                    }
                }
            });

        let mut out = vec![0.0f32; n * self.hidden];
        dispatcher.gemm_f32(
            &patches,
            &self.kernel,
            Some(&self.bias),
            n,
            patch_len,
            self.hidden,
            &mut out,
        )?;

        let resized;
        let grid: &[f32] = if grid_h == self.pos_grid && grid_w == self.pos_grid {
            &self.pos_embd
        } else {
            resized =
                resize_position_grid(&self.pos_embd, self.pos_grid, self.hidden, grid_h, grid_w)?;
            &resized
        };
        let hidden = self.hidden;
        out.par_chunks_mut(hidden)
            .zip(order.par_iter())
            .for_each(|(row, &(py, px))| {
                let src = &grid[(py * grid_w + px) * hidden..][..hidden];
                for (value, add) in row.iter_mut().zip(src) {
                    *value += *add;
                }
            });
        Ok(out)
    }
}
