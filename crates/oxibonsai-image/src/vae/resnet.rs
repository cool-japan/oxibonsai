//! ResNet block for the VAE decoder (`Flux2ResnetBlock2D`).
//!
//! `h = conv2(silu(norm2(conv1(silu(norm1(x))))))`; the residual is `x` itself
//! (or `conv_shortcut(x)` when `in_channels != out_channels`), added at the end:
//! `out = h + residual`. `norm1` is sized to the input channels, `norm2` to the
//! output channels; both convs are k=3, pad=1, stride 1. All NCHW `[C, H, W]`.

use crate::vae::conv::Conv2d;
use crate::vae::error::{VaeError, VaeResult};
use crate::vae::norm::GroupNorm;
use crate::vae::ops::silu_inplace;

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
        let conv1 = self.conv1.forward(&hs, h, w)?;
        // → silu(norm2(.)) → conv2
        let mut hs = conv1.data;
        self.norm2.forward_inplace(&mut hs, conv1.h, conv1.w)?;
        silu_inplace(&mut hs);
        let conv2 = self.conv2.forward(&hs, conv1.h, conv1.w)?;
        let mut out = conv2.data;
        // residual = x (or conv_shortcut(x))
        if let Some(shortcut) = self.conv_shortcut.as_ref() {
            let res = shortcut.forward(input, h, w)?;
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
