//! 2-D convolution for the VAE decoder (im2col + dense f32 GEMM).
//!
//! All convs here are stride-1, "same" padded (`k=3 → pad=1`, `k=1 → pad=0`),
//! with a bias. The weight uses the **MLX layout `[out, kH, kW, in]`** (as
//! exported), so each output channel's flattened weight row is ordered
//! `(kh, kw, ci)`. The im2col patch matrix is built with the *same* element
//! order, then [`crate::gemm::gemm_abt`] computes
//! `out[hw, oc] = Σ_k patch[hw, k] * weight[oc, k]` and the result is written
//! back to NCHW with the per-channel bias added.
//!
//! Everything operates on flat `Vec<f32>` NCHW buffers (`[C, H, W]`, batch 1).

use crate::gemm::gemm_abt;
use crate::vae::error::{VaeError, VaeResult};

/// A loaded convolution layer: MLX-layout weight `[out, kH, kW, in]` + bias.
pub struct Conv2d {
    /// Weight, row-major `[out, kH, kW, in]` (flattened per output channel as
    /// `(kh, kw, ci)`).
    pub weight: Vec<f32>,
    /// Per-output-channel bias `[out]`.
    pub bias: Vec<f32>,
    /// Output channels.
    pub out_ch: usize,
    /// Input channels.
    pub in_ch: usize,
    /// Kernel height (= width; square kernels only).
    pub k: usize,
    /// Padding (same on all sides).
    pub pad: usize,
}

impl Conv2d {
    /// Build a `Conv2d` from an exported MLX weight tensor `[out, kH, kW, in]`
    /// and a bias `[out]`. `pad` is the spatial padding (1 for k=3, 0 for k=1).
    ///
    /// # Errors
    /// [`VaeError::Shape`] if the weight is not 4-D, non-square, or the bias
    /// length disagrees with the weight's output dimension.
    pub fn from_weights(
        weight: &[f32],
        weight_shape: &[usize],
        bias: &[f32],
        pad: usize,
    ) -> VaeResult<Self> {
        if weight_shape.len() != 4 {
            return Err(VaeError::Shape(format!(
                "conv weight must be 4-D [out,kH,kW,in], got {weight_shape:?}"
            )));
        }
        let (out_ch, kh, kw, in_ch) = (
            weight_shape[0],
            weight_shape[1],
            weight_shape[2],
            weight_shape[3],
        );
        if kh != kw {
            return Err(VaeError::Shape(format!(
                "conv kernel must be square, got {kh}x{kw}"
            )));
        }
        if weight.len() != out_ch * kh * kw * in_ch {
            return Err(VaeError::Shape(format!(
                "conv weight len {} != product {:?}",
                weight.len(),
                weight_shape
            )));
        }
        if bias.len() != out_ch {
            return Err(VaeError::Shape(format!(
                "conv bias len {} != out_ch {out_ch}",
                bias.len()
            )));
        }
        Ok(Self {
            weight: weight.to_vec(),
            bias: bias.to_vec(),
            out_ch,
            in_ch,
            k: kh,
            pad,
        })
    }

    /// Run the convolution on an NCHW input `[in_ch, h, w]` (batch 1), returning
    /// the NCHW output `[out_ch, h, w]` (stride 1, "same" geometry: output
    /// `h_out = h + 2*pad - k + 1`).
    ///
    /// # Errors
    /// [`VaeError::Shape`] if the input length disagrees with `in_ch * h * w`.
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> VaeResult<ConvOut> {
        if input.len() != self.in_ch * h * w {
            return Err(VaeError::Shape(format!(
                "conv input len {} != in_ch*h*w {}",
                input.len(),
                self.in_ch * h * w
            )));
        }
        // GPU-first: route through the Metal f32 conv (im2col+GEMM) when the
        // `metal` feature is built, on macOS, and `OXI_VAE_GPU=1`. On ANY error
        // (graph unavailable, encode failure, …) silently fall through to the
        // CPU path below — a GPU failure can never break the decode.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if crate::vae::gpu::vae_gpu_enabled() {
                if let Ok(out) = crate::vae::gpu::conv2d_gpu(
                    &self.weight,
                    &self.bias,
                    input,
                    self.in_ch,
                    self.out_ch,
                    h,
                    w,
                    self.k,
                    self.pad,
                ) {
                    return Ok(ConvOut {
                        data: out.data,
                        h: out.h,
                        w: out.w,
                    });
                }
            }
        }
        // CUDA sibling of the Metal block above (target_os-disjoint: Linux/
        // Windows). Same `OXI_VAE_GPU` toggle; on ANY error silently fall through
        // to the CPU path below.
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            if crate::vae::cuda_gpu::vae_gpu_enabled() {
                if let Ok(out) = crate::vae::cuda_gpu::conv2d_gpu(
                    &self.weight,
                    &self.bias,
                    input,
                    self.in_ch,
                    self.out_ch,
                    h,
                    w,
                    self.k,
                    self.pad,
                ) {
                    return Ok(ConvOut {
                        data: out.data,
                        h: out.h,
                        w: out.w,
                    });
                }
            }
        }
        let k = self.k;
        let pad = self.pad;
        // Same-stride-1 output geometry.
        let h_out = h + 2 * pad + 1 - k;
        let w_out = w + 2 * pad + 1 - k;
        let patch_dim = k * k * self.in_ch;
        let spatial = h_out * w_out;

        // ── im2col: patches[hw, (kh*k + kw)*in_ch + ci] ──
        // For k=1, pad=0 this degenerates to a pure channel-mix (patch == input
        // column), but the general path handles it correctly without a special
        // case, so we keep one code path.
        let mut patches = vec![0.0f32; spatial * patch_dim];
        build_im2col(input, &mut patches, self.in_ch, h, w, k, pad, h_out, w_out);

        // ── GEMM: out_spatial[hw, oc] = Σ_k patches[hw,k] * weight[oc,k] ──
        let mut out_spatial = vec![0.0f32; spatial * self.out_ch];
        gemm_abt(
            &patches,
            &self.weight,
            &mut out_spatial,
            spatial,
            self.out_ch,
            patch_dim,
        );

        // ── transpose [hw, oc] → NCHW [oc, hw] and add bias ──
        let mut out = vec![0.0f32; self.out_ch * spatial];
        for oc in 0..self.out_ch {
            let b = self.bias[oc];
            let dst = &mut out[oc * spatial..(oc + 1) * spatial];
            for (hw, slot) in dst.iter_mut().enumerate() {
                *slot = out_spatial[hw * self.out_ch + oc] + b;
            }
        }
        Ok(ConvOut {
            data: out,
            h: h_out,
            w: w_out,
        })
    }
}

impl Conv2d {
    /// Memory-bounded convolution: the same per-output-pixel arithmetic as the
    /// CPU path of [`Self::forward`], but the im2col patch matrix is built for at
    /// most `max_tile_pixels` output pixels at a time instead of the whole plane.
    /// This caps the peak scratch allocation at `O(max_tile_pixels · k·k·in_ch)`
    /// rather than `O(h_out·w_out · k·k·in_ch)` — the multi-GB im2col peak of the
    /// high-resolution up-block convs.
    ///
    /// Tiling is by whole output rows: `row_block = max(1, max_tile_pixels /
    /// w_out)` output rows per chunk. The full input plane stays whole in memory
    /// (it is small relative to the im2col), so every output pixel sees exactly
    /// the same neighbourhood as the whole-plane computation and the result is
    /// **bit-identical to the untiled CPU convolution** (no seams, no blending;
    /// row-blocking never reassociates a per-pixel dot product). This is the
    /// memory lever behind [`crate::vae::tiling::TileBoundary::AfterMid`].
    ///
    /// CPU-only by design: the GPU conv path never materialises a host im2col, so
    /// it has no host-memory peak to bound; `AfterMid` is a host-RAM lever. (Vs
    /// the GPU [`Self::forward`] the values therefore match only to `cos ≈ 1`, as
    /// GPU sum reassociation differs from the CPU order — the same relationship
    /// the `AfterConvNormOut` tail tiling already has.)
    ///
    /// # Errors
    /// [`VaeError::Shape`] if the input length disagrees with `in_ch * h * w`.
    pub fn forward_tiled(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
        max_tile_pixels: usize,
    ) -> VaeResult<ConvOut> {
        if input.len() != self.in_ch * h * w {
            return Err(VaeError::Shape(format!(
                "conv input len {} != in_ch*h*w {}",
                input.len(),
                self.in_ch * h * w
            )));
        }
        let k = self.k;
        let pad = self.pad;
        let h_out = h + 2 * pad + 1 - k;
        let w_out = w + 2 * pad + 1 - k;
        let patch_dim = k * k * self.in_ch;
        let out_ch = self.out_ch;
        let spatial = h_out * w_out;
        let mut out = vec![0.0f32; out_ch * spatial];
        // At least one output row per chunk; bound the im2col by the pixel budget.
        let row_block = (max_tile_pixels / w_out.max(1)).max(1);
        let mut oh0 = 0usize;
        while oh0 < h_out {
            let oh1 = (oh0 + row_block).min(h_out);
            let block_spatial = (oh1 - oh0) * w_out;

            // im2col for output rows [oh0, oh1) into a bounded buffer.
            let mut patches = vec![0.0f32; block_spatial * patch_dim];
            build_im2col_row_range(
                input,
                &mut patches,
                self.in_ch,
                h,
                w,
                k,
                pad,
                w_out,
                oh0,
                oh1,
            );

            // GEMM: out_spatial[local_hw, oc] = Σ_k patches[local_hw,k]·weight[oc,k].
            let mut out_spatial = vec![0.0f32; block_spatial * out_ch];
            gemm_abt(
                &patches,
                &self.weight,
                &mut out_spatial,
                block_spatial,
                out_ch,
                patch_dim,
            );

            // Transpose [local_hw, oc] → NCHW rows [oh0, oh1) and add bias.
            let global_base = oh0 * w_out;
            for oc in 0..out_ch {
                let b = self.bias[oc];
                let dst = &mut out[oc * spatial..(oc + 1) * spatial];
                for local in 0..block_spatial {
                    dst[global_base + local] = out_spatial[local * out_ch + oc] + b;
                }
            }
            oh0 = oh1;
        }
        Ok(ConvOut {
            data: out,
            h: h_out,
            w: w_out,
        })
    }
}

/// Output of a convolution: NCHW data `[out_ch, h, w]` plus the new spatial dims.
pub struct ConvOut {
    /// NCHW output buffer `[out_ch, h, w]`.
    pub data: Vec<f32>,
    /// Output height.
    pub h: usize,
    /// Output width.
    pub w: usize,
}

/// Build the im2col patch matrix `patches[hw, (kh*k + kw)*in_ch + ci]` for an
/// NCHW input, with zero padding. The element order `(kh, kw, ci)` matches the
/// MLX weight layout `[out, kH, kW, in]`.
#[allow(clippy::too_many_arguments)]
fn build_im2col(
    input: &[f32],
    patches: &mut [f32],
    in_ch: usize,
    h: usize,
    w: usize,
    k: usize,
    pad: usize,
    h_out: usize,
    w_out: usize,
) {
    // Whole plane == the full output-row range [0, h_out). One code path.
    build_im2col_row_range(input, patches, in_ch, h, w, k, pad, w_out, 0, h_out);
}

/// im2col for a contiguous output-row range `[oh0, oh1)`, written into a buffer
/// laid out `[(oh1 - oh0) * w_out, patch_dim]` (local row 0 == output row `oh0`).
///
/// Identical element order (`(kh, kw, ci)` matching the MLX weight layout) and
/// zero-padding at global edges as the full-plane build; [`Conv2d::forward_tiled`]
/// uses it to bound the im2col to one row block, and [`build_im2col`] delegates
/// the whole plane to it.
#[allow(clippy::too_many_arguments)]
fn build_im2col_row_range(
    input: &[f32],
    patches: &mut [f32],
    in_ch: usize,
    h: usize,
    w: usize,
    k: usize,
    pad: usize,
    w_out: usize,
    oh0: usize,
    oh1: usize,
) {
    let patch_dim = k * k * in_ch;
    let hw_plane = h * w;
    for oh in oh0..oh1 {
        let local_row = oh - oh0;
        for ow in 0..w_out {
            let base = (local_row * w_out + ow) * patch_dim;
            let row = &mut patches[base..base + patch_dim];
            for kh in 0..k {
                // Source row in the (padded) input.
                let ih = oh + kh;
                if ih < pad || ih >= h + pad {
                    continue; // padded → zeros (row already zeroed)
                }
                let ih = ih - pad;
                for kw in 0..k {
                    let iw = ow + kw;
                    if iw < pad || iw >= w + pad {
                        continue;
                    }
                    let iw = iw - pad;
                    let dst_base = (kh * k + kw) * in_ch;
                    let src_base = ih * w + iw;
                    for ci in 0..in_ch {
                        row[dst_base + ci] = input[ci * hw_plane + src_base];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1x1_is_channelwise_matmul() {
        // 2 in-ch, 3 out-ch, k=1 p=0 on a 1x2 image. Weight [out,1,1,in].
        // out[o, x] = Σ_i w[o,i] * in[i, x] + bias[o].
        let weight = vec![
            1.0, 0.0, // o0: in0
            0.0, 2.0, // o1: 2*in1
            1.0, 1.0, // o2: in0+in1
        ];
        let shape = [3usize, 1, 1, 2];
        let bias = vec![0.0, 0.0, 10.0];
        let conv = Conv2d::from_weights(&weight, &shape, &bias, 0).expect("conv");
        // input NCHW [2,1,2]: in0=[3,4], in1=[5,6]
        let input = vec![3.0, 4.0, 5.0, 6.0];
        let out = conv.forward(&input, 1, 2).expect("fwd");
        assert_eq!((out.h, out.w), (1, 2));
        // o0 = in0 = [3,4]; o1 = 2*in1 = [10,12]; o2 = in0+in1+10 = [18,20]
        assert_eq!(out.data, vec![3.0, 4.0, 10.0, 12.0, 18.0, 20.0]);
    }

    /// Row-blocking must not change the result: `forward_tiled` at any pixel
    /// budget must be **bit-identical** to the whole-plane `forward_tiled`, across
    /// k=1 and k=3 convs. The whole-plane reference (`build_im2col_row_range(0,
    /// h_out)`) is exactly the im2col + GEMM the untiled CPU [`Conv2d::forward`]
    /// runs (which `build_im2col` delegates to), so this pins the tiled path to
    /// the canonical convolution — without depending on `forward`'s GPU dispatch
    /// (`OXI_VAE_GPU`), which would otherwise reassociate the f32 sums.
    #[test]
    fn forward_tiled_is_bit_identical_across_budgets() {
        for &(in_c, out_c, k, pad) in &[(3usize, 4usize, 3usize, 1usize), (5, 2, 1, 0)] {
            let h = 9usize;
            let w = 7usize;
            let weight: Vec<f32> = (0..out_c * k * k * in_c)
                .map(|i| (i as f32 * 0.017).sin() * 0.4)
                .collect();
            let bias: Vec<f32> = (0..out_c).map(|i| 0.05 * i as f32 - 0.1).collect();
            let conv =
                Conv2d::from_weights(&weight, &[out_c, k, k, in_c], &bias, pad).expect("conv");
            let input: Vec<f32> = (0..in_c * h * w)
                .map(|i| (i as f32 * 0.031).cos() * 1.3 - 0.2)
                .collect();

            // Whole-plane reference (one chunk == the untiled CPU im2col path).
            let reference = conv.forward_tiled(&input, h, w, h * w).expect("reference");
            // Sweep budgets from a single row up to the whole plane.
            for &budget in &[1usize, w, 2 * w, 3 * w + 1, h * w] {
                let tiled = conv
                    .forward_tiled(&input, h, w, budget)
                    .expect("forward_tiled");
                assert_eq!(tiled.h, reference.h);
                assert_eq!(tiled.w, reference.w);
                assert_eq!(
                    tiled.data, reference.data,
                    "forward_tiled(budget={budget}) diverged from whole-plane \
                     (in_c={in_c}, out_c={out_c}, k={k})"
                );
            }
        }
    }

    #[test]
    fn conv3x3_same_padding_box_filter() {
        // 1 in-ch, 1 out-ch, k=3 p=1, all-ones weight (box filter), no bias.
        let weight = vec![1.0f32; 9];
        let shape = [1usize, 3, 3, 1];
        let bias = vec![0.0];
        let conv = Conv2d::from_weights(&weight, &shape, &bias, 1).expect("conv");
        let input: Vec<f32> = (1..=9).map(|v| v as f32).collect(); // 1..9
        let out = conv.forward(&input, 3, 3).expect("fwd");
        assert_eq!((out.h, out.w), (3, 3));
        // center (1,1) sees all of 1..9 => 45.
        assert_eq!(out.data[4], 45.0);
        // top-left corner (0,0) sees [1,2,4,5] (rest padded) => 12.
        assert_eq!(out.data[0], 12.0);
    }
}
