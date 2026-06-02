//! Single-head spatial self-attention for the VAE mid-block.
//!
//! Matches `Flux2AttentionBlock` (1 head, head_dim = channels): GroupNorm →
//! `to_q`/`to_k`/`to_v` (Linear, bias) → scaled-dot-product attention with
//! `scale = 1/sqrt(C)` over the flattened `H*W` sequence → `to_out` (Linear,
//! bias) → residual add to the (un-normed) input.
//!
//! Internally the VAE stack is NCHW `[C, H, W]`; attention needs a token-major
//! `[H*W, C]` view, so we transpose in and out. The SDPA itself reuses
//! [`crate::math::joint_attention`] (num_heads = 1).

use crate::math::{dense_matmul, joint_attention};
use crate::vae::error::{VaeError, VaeResult};
use crate::vae::norm::GroupNorm;

/// A bias-augmented linear layer `[out, in]` (PyTorch / MLX `nn.Linear` layout).
pub struct Linear {
    /// Row-major weight `[out, in]`.
    pub weight: Vec<f32>,
    /// Bias `[out]`.
    pub bias: Vec<f32>,
    /// Output features.
    pub out_dim: usize,
    /// Input features.
    pub in_dim: usize,
}

impl Linear {
    /// Build a `Linear` from a `[out, in]` weight and `[out]` bias.
    ///
    /// # Errors
    /// [`VaeError::Shape`] if the weight is not 2-D or the bias length disagrees
    /// with the output dimension.
    pub fn from_weights(weight: &[f32], weight_shape: &[usize], bias: &[f32]) -> VaeResult<Self> {
        if weight_shape.len() != 2 {
            return Err(VaeError::Shape(format!(
                "linear weight must be 2-D [out,in], got {weight_shape:?}"
            )));
        }
        let (out_dim, in_dim) = (weight_shape[0], weight_shape[1]);
        if weight.len() != out_dim * in_dim {
            return Err(VaeError::Shape(format!(
                "linear weight len {} != out*in {}",
                weight.len(),
                out_dim * in_dim
            )));
        }
        if bias.len() != out_dim {
            return Err(VaeError::Shape(format!(
                "linear bias len {} != out_dim {out_dim}",
                bias.len()
            )));
        }
        Ok(Self {
            weight: weight.to_vec(),
            bias: bias.to_vec(),
            out_dim,
            in_dim,
        })
    }

    /// Apply the linear to a `[rows, in]` token-major buffer, returning
    /// `[rows, out]` (bias added).
    ///
    /// # Errors
    /// Propagates [`crate::error::DitError`] from the GEMM as
    /// [`VaeError::Shape`].
    pub fn forward(&self, x: &[f32], rows: usize) -> VaeResult<Vec<f32>> {
        let mut out = dense_matmul(x, &self.weight, rows, self.out_dim, self.in_dim)
            .map_err(|e| VaeError::Shape(e.to_string()))?;
        for r in 0..rows {
            let row = &mut out[r * self.out_dim..(r + 1) * self.out_dim];
            for (i, v) in row.iter_mut().enumerate() {
                *v += self.bias[i];
            }
        }
        Ok(out)
    }
}

/// A spatial self-attention block (GroupNorm + 4 Linears, single head).
pub struct AttentionBlock {
    /// Pre-attention GroupNorm.
    pub group_norm: GroupNorm,
    /// Query projection.
    pub to_q: Linear,
    /// Key projection.
    pub to_k: Linear,
    /// Value projection.
    pub to_v: Linear,
    /// Output projection.
    pub to_out: Linear,
    /// Channels (= head_dim, single head).
    pub channels: usize,
}

impl AttentionBlock {
    /// Run the attention block on an NCHW buffer `[C, H, W]` (batch 1),
    /// returning the residual-added NCHW output `[C, H, W]`.
    ///
    /// # Errors
    /// [`VaeError::Shape`] on a length mismatch, or a propagated GEMM/GroupNorm
    /// error.
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> VaeResult<Vec<f32>> {
        let c = self.channels;
        let hw = h * w;
        if input.len() != c * hw {
            return Err(VaeError::Shape(format!(
                "attention input len {} != C*H*W {}",
                input.len(),
                c * hw
            )));
        }
        // GroupNorm on a copy (input is the residual).
        let mut normed_nchw = input.to_vec();
        self.group_norm.forward_inplace(&mut normed_nchw, h, w)?;
        // NCHW [C, hw] → token-major [hw, C].
        let normed = nchw_to_tokens(&normed_nchw, c, hw);
        // q/k/v: [hw, C] each.
        let q = self.to_q.forward(&normed, hw)?;
        let k = self.to_k.forward(&normed, hw)?;
        let v = self.to_v.forward(&normed, hw)?;
        // Single-head SDPA over the hw sequence: returns [hw, C].
        let attended =
            joint_attention(&q, &k, &v, 1, hw, c).map_err(|e| VaeError::Shape(e.to_string()))?;
        // Output projection: [hw, C].
        let projected = self.to_out.forward(&attended, hw)?;
        // Residual add (token-major) then back to NCHW.
        let mut out = vec![0.0f32; c * hw];
        for ci in 0..c {
            for s in 0..hw {
                out[ci * hw + s] = input[ci * hw + s] + projected[s * c + ci];
            }
        }
        Ok(out)
    }
}

/// Transpose an NCHW plane `[C, hw]` to token-major `[hw, C]`.
fn nchw_to_tokens(x: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hw * c];
    for ci in 0..c {
        for s in 0..hw {
            out[s * c + ci] = x[ci * hw + s];
        }
    }
    out
}
