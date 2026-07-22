//! ResNet block for the VAE decoder (`Flux2ResnetBlock2D`).
//!
//! `h = conv2(silu(norm2(conv1(silu(norm1(x))))))`; the residual is `x` itself
//! (or `conv_shortcut(x)` when `in_channels != out_channels`), added at the end:
//! `out = h + residual`. `norm1` is sized to the input channels, `norm2` to the
//! output channels; both convs are k=3, pad=1, stride 1. All NCHW `[C, H, W]`.

use crate::vae::conv::{Conv2d, ConvOut};
use crate::vae::error::{VaeError, VaeResult};
use crate::vae::norm::GroupNorm;
use crate::vae::ops::silu_inplace;

/// How the block's internal convolutions are executed: whole-plane im2col
/// ([`Conv2d::forward`]) or a memory-bounded row-blocked im2col
/// ([`Conv2d::forward_tiled`], the `AfterMid` lever). The two produce
/// bit-identical output; only the peak scratch memory differs.
#[derive(Clone, Copy)]
enum ConvExec {
    /// Whole-plane convolution (default).
    Whole,
    /// Row-blocked convolution capped at this many output pixels per im2col.
    Tiled(usize),
}

impl ConvExec {
    /// Run `conv` on `x` `[in, h, w]` under this execution mode.
    fn run(self, conv: &Conv2d, x: &[f32], h: usize, w: usize) -> VaeResult<ConvOut> {
        match self {
            ConvExec::Whole => conv.forward(x, h, w),
            ConvExec::Tiled(max_tile_pixels) => conv.forward_tiled(x, h, w, max_tile_pixels),
        }
    }
}

/// A residual block: two (GroupNorm → SiLU → Conv) stages plus a shortcut.
pub struct ResnetBlock2D {
    /// First GroupNorm (input channels).
    pub norm1: GroupNorm,
    /// First conv (in → out, k=3 p=1).
    pub conv1: Conv2d,
    /// Second GroupNorm (output channels).
    pub norm2: GroupNorm,
    /// Second conv (out → out, k=3 p=1).
    pub conv2: Conv2d,
    /// Optional 1×1 shortcut conv (present iff in != out channels).
    pub conv_shortcut: Option<Conv2d>,
    /// Input channels.
    pub in_ch: usize,
    /// Output channels.
    pub out_ch: usize,
}

impl ResnetBlock2D {
    /// Run the block on an NCHW input `[in_ch, h, w]`, returning the NCHW output
    /// `[out_ch, h, w]` (spatial unchanged).
    ///
    /// # Errors
    /// [`VaeError::Shape`] on a length mismatch or a propagated conv/norm error.
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> VaeResult<Vec<f32>> {
        self.run(input, h, w, ConvExec::Whole)
    }

    /// Memory-bounded variant of [`Self::forward`]: the internal convolutions run
    /// through [`Conv2d::forward_tiled`] so their im2col scratch is bounded to
    /// `max_tile_pixels` output pixels, while GroupNorm / SiLU / the residual add
    /// operate on the whole plane exactly as in [`Self::forward`]. The result is
    /// **bit-identical to the untiled CPU block** (GroupNorm statistics stay
    /// global, row-blocking never reassociates a conv); only the peak conv scratch
    /// memory shrinks. Used by the `AfterMid` up-block tiling path. Convolutions
    /// run on the CPU (the bounded im2col is a host-RAM lever).
    ///
    /// # Errors
    /// [`VaeError::Shape`] on a length mismatch or a propagated conv/norm error.
    pub fn forward_tiled(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
        max_tile_pixels: usize,
    ) -> VaeResult<Vec<f32>> {
        self.run(input, h, w, ConvExec::Tiled(max_tile_pixels))
    }

    /// Shared block body; `exec` selects whole-plane or row-blocked convolutions.
    fn run(&self, input: &[f32], h: usize, w: usize, exec: ConvExec) -> VaeResult<Vec<f32>> {
        let hw = h * w;
        if input.len() != self.in_ch * hw {
            return Err(VaeError::Shape(format!(
                "resnet input len {} != in_ch*H*W {}",
                input.len(),
                self.in_ch * hw
            )));
        }
        // h = silu(norm1(x)) → conv1
        let mut hs = input.to_vec();
        self.norm1.forward_inplace(&mut hs, h, w)?;
        silu_inplace(&mut hs);
        let conv1 = exec.run(&self.conv1, &hs, h, w)?;
        // → silu(norm2(.)) → conv2
        let mut hs = conv1.data;
        self.norm2.forward_inplace(&mut hs, conv1.h, conv1.w)?;
        silu_inplace(&mut hs);
        let conv2 = exec.run(&self.conv2, &hs, conv1.h, conv1.w)?;
        let mut out = conv2.data;
        // residual = x (or conv_shortcut(x))
        if let Some(shortcut) = self.conv_shortcut.as_ref() {
            let res = exec.run(shortcut, input, h, w)?;
            if res.data.len() != out.len() {
                return Err(VaeError::Shape(format!(
                    "resnet shortcut len {} != main len {}",
                    res.data.len(),
                    out.len()
                )));
            }
            for (o, r) in out.iter_mut().zip(res.data.iter()) {
                *o += *r;
            }
        } else {
            if input.len() != out.len() {
                return Err(VaeError::Shape(format!(
                    "resnet identity residual len {} != main len {} (in_ch {} out_ch {})",
                    input.len(),
                    out.len(),
                    self.in_ch,
                    self.out_ch
                )));
            }
            for (o, r) in out.iter_mut().zip(input.iter()) {
                *o += *r;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic ResnetBlock2D with `groups`-group GroupNorms and
    /// `in_ch`/`out_ch` channels (a 1×1 shortcut is added iff they differ).
    fn build_block(in_ch: usize, out_ch: usize, groups: usize) -> ResnetBlock2D {
        let gn = |c: usize, seed: f32| {
            let weight: Vec<f32> = (0..c).map(|i| 1.0 + 0.03 * (i as f32 + seed)).collect();
            let bias: Vec<f32> = (0..c).map(|i| -0.02 * (i as f32 + seed)).collect();
            GroupNorm::new(&weight, &bias, groups, 1e-6).expect("groupnorm")
        };
        let conv = |o: usize, i: usize, seed: f32| {
            let weight: Vec<f32> = (0..o * 3 * 3 * i)
                .map(|n| ((n as f32 + seed) * 0.013).sin() * 0.3)
                .collect();
            let bias: Vec<f32> = (0..o).map(|n| 0.01 * n as f32 - 0.05).collect();
            Conv2d::from_weights(&weight, &[o, 3, 3, i], &bias, 1).expect("conv")
        };
        let conv_shortcut = if in_ch != out_ch {
            let weight: Vec<f32> = (0..out_ch * in_ch)
                .map(|n| (n as f32 * 0.02).cos() * 0.2)
                .collect();
            let bias = vec![0.0f32; out_ch];
            Some(Conv2d::from_weights(&weight, &[out_ch, 1, 1, in_ch], &bias, 0).expect("shortcut"))
        } else {
            None
        };
        ResnetBlock2D {
            norm1: gn(in_ch, 0.0),
            conv1: conv(out_ch, in_ch, 1.0),
            norm2: gn(out_ch, 2.0),
            conv2: conv(out_ch, out_ch, 3.0),
            conv_shortcut,
            in_ch,
            out_ch,
        }
    }

    /// Row-blocking must not change the block's result: `forward_tiled` at any
    /// pixel budget must be bit-identical to the whole-plane `forward_tiled`, both
    /// with and without a channel-changing 1×1 shortcut. The whole-plane reference
    /// runs the same CPU im2col + GEMM as the untiled `forward` (which routes to
    /// the GPU under `native-cuda`, so it is not used as the reference here — the
    /// GPU would reassociate the f32 sums). This is the contract `AfterMid` relies
    /// on.
    #[test]
    fn forward_tiled_is_bit_identical_across_budgets() {
        let h = 10usize;
        let w = 6usize;
        for &(in_ch, out_ch) in &[(4usize, 4usize), (8usize, 4usize)] {
            let block = build_block(in_ch, out_ch, 4);
            let input: Vec<f32> = (0..in_ch * h * w)
                .map(|i| (i as f32 * 0.037).sin() * 1.1 + 0.1)
                .collect();
            let reference = block
                .forward_tiled(&input, h, w, h * w)
                .expect("whole-plane reference");
            for &budget in &[1usize, w, 3 * w, h * w] {
                let tiled = block
                    .forward_tiled(&input, h, w, budget)
                    .expect("forward_tiled");
                assert_eq!(
                    tiled, reference,
                    "resnet forward_tiled(budget={budget}) diverged from whole-plane \
                     (in_ch={in_ch}, out_ch={out_ch})"
                );
            }
        }
    }
}
