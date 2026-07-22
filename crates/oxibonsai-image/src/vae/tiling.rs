//! Tiled VAE decode helpers — spatial tiling with halo overlap so that peak
//! activation memory is O(tile_size) rather than O(output_resolution²).
//!
//! # Architecture
//!
//! The VAE decoder has two natural tiling boundaries:
//!
//! * [`TileBoundary::AfterConvNormOut`] (default): run the **entire** decode
//!   through `conv_norm_out` (GroupNorm) in one shot, which guarantees correct
//!   global per-group statistics without any two-pass bookkeeping. Then tile
//!   only the final `silu → conv_out` (k=3, pad=1) with a 1-pixel halo. This
//!   is **seam-free by construction** and the only boundary needed for the
//!   current 512 px → next-power-of-two use-case where the bottleneck is the
//!   `conv_out` im2col, not the GroupNorm planes.
//!
//! * [`TileBoundary::AfterMid`]: additionally bound the **up-block** convolution
//!   im2col — the real multi-GB peak of a high-resolution decode. Rather than
//!   spatially tiling the activation planes (which would force a two-pass
//!   GroupNorm to keep per-group statistics global), the up-blocks keep their
//!   activation planes whole — they are small relative to the im2col — and run
//!   each convolution through [`crate::vae::conv::Conv2d::forward_tiled`], which
//!   builds the im2col one row-block at a time. GroupNorm therefore still sees
//!   the whole plane (exact global stats) and the output is **bit-identical to
//!   the untiled CPU decode** (row-blocking never reassociates a conv), while the
//!   up-block conv scratch drops from the untiled ~3.6 GB peak at 512² to
//!   `O(tile_px² · k·k·in_ch)` (≈1 GB at `tile_px = 256`). Against a GPU untiled
//!   decode it matches to `cos ≈ 1`, like the `AfterConvNormOut` tail tiling.
//!
//!   The two-pass GroupNorm helpers ([`crate::vae::norm::GroupNorm::compute_stats`]
//!   / [`crate::vae::norm::GroupNorm::apply_precomputed`]) remain available as a
//!   general utility, but the `AfterMid` path does not need them because it never
//!   splits a GroupNorm plane. The up-block convs run on the CPU under this
//!   boundary (the host im2col is precisely what is being bounded).

use crate::math::silu;
use crate::vae::conv::Conv2d;
use crate::vae::decoder::Map;
use crate::vae::error::VaeResult;

/// Which stage is the entry point for tiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileBoundary {
    /// Tile only `silu → conv_out` (default). Everything through
    /// `conv_norm_out` runs whole (correct global GroupNorm stats).
    /// Halo = 1 px (k=3 conv). Zero seam risk.
    AfterConvNormOut,
    /// Like [`Self::AfterConvNormOut`], but additionally bounds the up-block
    /// convolution im2col (the real multi-GB peak) by running each up-block conv
    /// through [`crate::vae::conv::Conv2d::forward_tiled`] with a `tile_px²`-pixel
    /// budget. The activation planes stay whole (GroupNorm stats remain global),
    /// so the result is bit-identical to the untiled CPU decode (`cos ≈ 1` vs a
    /// GPU decode); only the up-block scratch memory shrinks (from ~3.6 GB at 512²
    /// to ≈1 GB at `tile_px = 256`). The up-block convs run on the CPU under this
    /// boundary. See the module docs.
    AfterMid,
}

/// Configuration for tiled VAE decode.
#[derive(Clone, Copy, Debug)]
pub struct TileConfig {
    /// Core tile side at the **final output resolution** (pixels). Default: 256.
    /// The actual window extracted is `tile_px + 2*halo` per side; the halo
    /// is discarded after the convolution.
    pub tile_px: usize,
    /// Which stage to begin tiling at.
    pub boundary: TileBoundary,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_px: 256,
            boundary: TileBoundary::AfterConvNormOut,
        }
    }
}

/// Parameters for [`extract_window`], bundled to avoid the 8-argument limit.
pub struct WindowParams {
    /// Number of channels.
    pub c: usize,
    /// Source height.
    pub h: usize,
    /// Source width.
    pub w: usize,
    /// Starting row (may be negative for top halo).
    pub r_start: isize,
    /// Starting column (may be negative for left halo).
    pub c_start: isize,
    /// Window height (core + 2 * halo).
    pub win_h: usize,
    /// Window width (core + 2 * halo).
    pub win_w: usize,
}

/// Extract a spatial window from `src` (NCHW layout, `[c, h, w]`) into a new
/// `Vec<f32>`.
///
/// The window covers rows `[r_start, r_start+win_h)` and columns
/// `[c_start, c_start+win_w)` in the source space (these already include the
/// halo: the caller subtracts `halo` before calling). Out-of-bounds positions
/// (global image edges) are zero-filled, matching `build_im2col`'s zero-pad.
pub fn extract_window(src: &[f32], p: &WindowParams) -> Vec<f32> {
    let (c, h, w, r_start, c_start, win_h, win_w) =
        (p.c, p.h, p.w, p.r_start, p.c_start, p.win_h, p.win_w);
    let mut out = vec![0.0f32; c * win_h * win_w];
    for ch in 0..c {
        for dr in 0..win_h {
            let src_r = r_start + dr as isize;
            if src_r < 0 || src_r >= h as isize {
                continue;
            }
            let src_r = src_r as usize;
            for dc in 0..win_w {
                let src_c = c_start + dc as isize;
                if src_c < 0 || src_c >= w as isize {
                    continue;
                }
                let src_c = src_c as usize;
                let src_idx = ch * h * w + src_r * w + src_c;
                let dst_idx = ch * win_h * win_w + dr * win_w + dc;
                out[dst_idx] = src[src_idx];
            }
        }
    }
    out
}

/// Write the core region of `tile_data` (shape `[c, tile_h, tile_w]`) into
/// `canvas` (shape `[c, canvas_h, canvas_w]`) at position `(out_r, out_c)`.
///
/// Only the `core_h × core_w` interior starting at `(halo, halo)` within the
/// tile is written; the halo border is discarded.
#[allow(clippy::too_many_arguments)]
pub fn write_core_to_canvas(
    canvas: &mut [f32],
    canvas_c: usize,
    canvas_h: usize,
    canvas_w: usize,
    tile_data: &[f32],
    tile_h: usize,
    tile_w: usize,
    halo: usize,
    out_r: usize,
    out_c: usize,
    core_h: usize,
    core_w: usize,
) {
    for ch in 0..canvas_c {
        for dr in 0..core_h {
            for dc in 0..core_w {
                let tile_r = halo + dr;
                let tile_c = halo + dc;
                let tile_idx = ch * tile_h * tile_w + tile_r * tile_w + tile_c;
                let canvas_idx = ch * canvas_h * canvas_w + (out_r + dr) * canvas_w + (out_c + dc);
                canvas[canvas_idx] = tile_data[tile_idx];
            }
        }
    }
}

/// Iterate over non-overlapping core tiles covering `[0, total)`.
///
/// Each item is `(core_start, core_size)`. The last tile may be smaller than
/// `tile_size` to cover the remainder.
pub fn tile_grid(total: usize, tile_size: usize) -> impl Iterator<Item = (usize, usize)> {
    let n = total.div_ceil(tile_size);
    (0..n).map(move |i| {
        let start = i * tile_size;
        let size = tile_size.min(total - start);
        (start, size)
    })
}

/// Tile `silu → conv_out` on an already-normalised plane `pre_silu` of shape
/// `[c, h, w]`.
///
/// Halo = 1 px (k=3 conv, pad=1). The halo strips drawn from the full-plane
/// neighbour data are real pixels (not zero), so interior seams are exact.
/// Global edge positions are zero-filled, matching `build_im2col`'s zero-pad.
///
/// Returns a `Map` of shape `[conv_out.out_ch, h, w]`.
///
/// # Errors
/// [`crate::vae::error::VaeError::Shape`] from `Conv2d::forward` on a
/// malformed window.
pub fn tile_silu_conv_out(
    pre_silu: &[f32],
    c: usize,
    h: usize,
    w: usize,
    conv_out: &Conv2d,
    tile_px: usize,
) -> VaeResult<Map> {
    let halo: usize = 1; // k=3 conv needs 1-pixel overlap
    let out_c = conv_out.out_ch;
    let mut canvas = vec![0.0f32; out_c * h * w];

    for (row_start, core_h) in tile_grid(h, tile_px) {
        for (col_start, core_w) in tile_grid(w, tile_px) {
            let win_h = core_h + 2 * halo;
            let win_w = core_w + 2 * halo;

            // Extract window (includes halo, zero-filled at global edges).
            let mut window = extract_window(
                pre_silu,
                &WindowParams {
                    c,
                    h,
                    w,
                    r_start: row_start as isize - halo as isize,
                    c_start: col_start as isize - halo as isize,
                    win_h,
                    win_w,
                },
            );

            // SiLU in-place on the window.
            for v in window.iter_mut() {
                *v = silu(*v);
            }

            // conv_out on the window.
            let conv_result = conv_out.forward(&window, win_h, win_w)?;

            // Write the core region (halo strips discarded).
            write_core_to_canvas(
                &mut canvas,
                out_c,
                h,
                w,
                &conv_result.data,
                conv_result.h,
                conv_result.w,
                halo,
                row_start,
                col_start,
                core_h,
                core_w,
            );
        }
    }

    Ok(Map::new_pub(canvas, out_c, h, w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vae::conv::Conv2d;
    use crate::vae::ops::silu_inplace;

    /// Pin this test process to the CPU conv path. The bit-identity contract
    /// (module docs) is against the **untiled CPU decode**; the GPU implicit-GEMM
    /// conv reassociates its f32 reduction shape-dependently, so GPU-tiled vs
    /// GPU-untiled only matches to `cos ≈ 1` (asserted separately below). Safe
    /// under nextest's process-per-test model: the `OXI_VAE_GPU` gate is a
    /// `OnceLock` read on first conv use, and nothing in this process has run a
    /// conv before the test body.
    fn force_cpu_conv() {
        std::env::set_var("OXI_VAE_GPU", "0");
    }

    #[test]
    fn tile_grid_covers_canvas() {
        for total in [1usize, 64, 100, 128, 256, 512] {
            for ts in [32usize, 64, 128] {
                let tiles: Vec<(usize, usize)> = tile_grid(total, ts).collect();
                let covered: usize = tiles.iter().map(|(_, s)| *s).sum();
                assert_eq!(
                    covered, total,
                    "tile_grid: covered {covered} != total {total} (ts={ts})"
                );
                // Non-overlapping and contiguous.
                let mut expected_start = 0usize;
                for &(start, size) in &tiles {
                    assert_eq!(
                        start, expected_start,
                        "gap at {start} (expected {expected_start})"
                    );
                    assert!(size > 0, "zero-size tile");
                    expected_start += size;
                }
                assert_eq!(expected_start, total);
            }
        }
    }

    #[test]
    fn tiled_conv_out_equals_untiled() {
        force_cpu_conv();
        // Tiny deterministic test: 3 in-channels, 2 out-channels, 8x8 spatial.
        // k=3, pad=1. Tile with tile_px=4 forces a 2x2=4-tile grid.
        let in_c = 3usize;
        let out_c = 2usize;
        let h = 8usize;
        let w = 8usize;
        let k = 3usize;
        // Weight [out, kH, kW, in] = [2, 3, 3, 3] = 54 elements.
        let weight: Vec<f32> = (0..out_c * k * k * in_c)
            .map(|i| i as f32 * 0.03 - 0.8)
            .collect();
        let bias = vec![0.1f32, -0.2f32];
        let conv =
            Conv2d::from_weights(&weight, &[out_c, k, k, in_c], &bias, 1).expect("build conv");

        // Deterministic input after GroupNorm (pre-silu).
        let pre_silu: Vec<f32> = (0..in_c * h * w)
            .map(|i| (i as f32 * 0.07) - (in_c * h * w) as f32 * 0.035)
            .collect();

        // Untiled reference: silu then conv.
        let mut ref_silu = pre_silu.clone();
        silu_inplace(&mut ref_silu);
        let ref_out = conv.forward(&ref_silu, h, w).expect("ref conv");

        // Tiled path (4 tiles).
        let tiled =
            tile_silu_conv_out(&pre_silu, in_c, h, w, &conv, 4).expect("tile_silu_conv_out");

        assert_eq!(tiled.c, out_c);
        assert_eq!(tiled.h, h);
        assert_eq!(tiled.w, w);
        assert_eq!(tiled.data.len(), ref_out.data.len());

        // Bit-exact: the halo-window extraction reproduces the same pixel
        // neighbourhood the untiled conv sees, and the GEMM kernel routing
        // (see `crate::gemm::dot4_neon`, kept in lockstep with
        // `micro4x4_neon`) no longer reassociates the dot-product reduction
        // differently across row-chunk sizes. This is the default tiling
        // boundary (`TileConfig::default().boundary ==
        // TileBoundary::AfterConvNormOut`), so its bit-identity claim is
        // load-bearing — assert exact equality rather than a cosine
        // threshold, matching the sibling `AfterMid` tests in `conv.rs`,
        // `resnet.rs`, and `decoder.rs`.
        assert_eq!(
            tiled.data, ref_out.data,
            "tile_silu_conv_out must be bit-identical to the untiled conv"
        );
    }

    /// Regression for the dot4_neon/micro4x4_neon kernel-routing hazard: at
    /// a spatial size that forces a non-multiple-of-4 per-tile row count,
    /// tiling must still be bit-identical to the untiled path.
    #[test]
    fn tiled_conv_out_equals_untiled_odd_tile_rows() {
        force_cpu_conv();
        let in_c = 4usize;
        let out_c = 3usize;
        let h = 11usize;
        let w = 9usize;
        let k = 3usize;
        let weight: Vec<f32> = (0..out_c * k * k * in_c)
            .map(|i| (i as f32 * 0.017 - 0.6).sin())
            .collect();
        let bias = vec![0.05f32, -0.1f32, 0.2f32];
        let conv =
            Conv2d::from_weights(&weight, &[out_c, k, k, in_c], &bias, 1).expect("build conv");

        let pre_silu: Vec<f32> = (0..in_c * h * w)
            .map(|i| ((i as f32 * 0.031).cos()) - 0.2)
            .collect();

        let mut ref_silu = pre_silu.clone();
        silu_inplace(&mut ref_silu);
        let ref_out = conv.forward(&ref_silu, h, w).expect("ref conv");

        // tile_px = 3 forces core row/col counts (3, 3, 3, 2) — several
        // tile row-block sizes, exercising both the batched-4 and tail
        // kernel paths in gemm_ncols.
        let tiled =
            tile_silu_conv_out(&pre_silu, in_c, h, w, &conv, 3).expect("tile_silu_conv_out");

        assert_eq!(tiled.c, out_c);
        assert_eq!(tiled.h, h);
        assert_eq!(tiled.w, w);
        assert_eq!(
            tiled.data, ref_out.data,
            "tile_silu_conv_out must be bit-identical to the untiled conv (odd tile rows)"
        );
    }

    /// The tiled-vs-untiled contract under the *default* conv dispatch
    /// (GPU implicit-GEMM when the `metal` build is active, CPU otherwise):
    /// the GPU conv reassociates its f32 reduction shape-dependently, so
    /// tiled and untiled outputs are only guaranteed to agree at ulp level
    /// (`cos ≈ 1`, per the module docs) — not bitwise. On the CPU path this
    /// tolerance check is subsumed by the exact-equality tests above.
    #[test]
    fn tiled_conv_out_matches_untiled_under_default_dispatch() {
        let in_c = 3usize;
        let out_c = 2usize;
        let h = 8usize;
        let w = 8usize;
        let k = 3usize;
        let weight: Vec<f32> = (0..out_c * k * k * in_c)
            .map(|i| i as f32 * 0.03 - 0.8)
            .collect();
        let bias = vec![0.1f32, -0.2f32];
        let conv =
            Conv2d::from_weights(&weight, &[out_c, k, k, in_c], &bias, 1).expect("build conv");
        let pre_silu: Vec<f32> = (0..in_c * h * w)
            .map(|i| (i as f32 * 0.07) - (in_c * h * w) as f32 * 0.035)
            .collect();

        let mut ref_silu = pre_silu.clone();
        silu_inplace(&mut ref_silu);
        let ref_out = conv.forward(&ref_silu, h, w).expect("ref conv");
        let tiled =
            tile_silu_conv_out(&pre_silu, in_c, h, w, &conv, 4).expect("tile_silu_conv_out");

        assert_eq!(tiled.data.len(), ref_out.data.len());
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in tiled.data.iter().zip(ref_out.data.iter()) {
            let rel = (a - b).abs() / b.abs().max(1.0);
            assert!(
                rel <= 1e-4,
                "tiled vs untiled diverged beyond ulp tolerance: {a} vs {b} (rel {rel})"
            );
            dot += f64::from(a) * f64::from(b);
            na += f64::from(a) * f64::from(a);
            nb += f64::from(b) * f64::from(b);
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE);
        assert!(
            cos >= 0.999_999,
            "tiled vs untiled cosine {cos} below the module-doc contract"
        );
    }
}
