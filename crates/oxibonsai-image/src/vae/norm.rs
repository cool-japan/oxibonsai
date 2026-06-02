//! GroupNorm for the VAE decoder (PyTorch-compatible, computed in f32).
//!
//! Matches MLX's `nn.GroupNorm(num_groups=32, pytorch_compatible=True,
//! eps=1e-6)`: the channels are split into `num_groups` *contiguous* groups of
//! `group_size = C / num_groups` channels each, and each group is normalised
//! over all its channels across every spatial position
//! (`(x - mean) / sqrt(var + eps)`, population variance), then a per-channel
//! affine `weight[c] * y + bias[c]` is applied.
//!
//! The MLX wrapper feeds NHWC, but since the grouping is over `group_size`
//! contiguous channels the result is identical to applying standard PyTorch
//! GroupNorm to an NCHW tensor — which is what we do here (the rest of the VAE
//! stack is NCHW).

use crate::vae::error::{VaeError, VaeResult};

/// A GroupNorm layer: `num_groups`, per-channel affine `weight`/`bias` `[C]`.
pub struct GroupNorm {
    /// Number of groups (32 throughout the VAE).
    pub num_groups: usize,
    /// Per-channel scale `[C]`.
    pub weight: Vec<f32>,
    /// Per-channel shift `[C]`.
    pub bias: Vec<f32>,
    /// Channels.
    pub channels: usize,
    /// Epsilon (inside the sqrt).
    pub eps: f32,
}

impl GroupNorm {
    /// Build a GroupNorm from affine `weight`/`bias` `[C]`.
    ///
    /// # Errors
    /// [`VaeError::Shape`] if `weight.len() != bias.len()`, or `channels` is not
    /// divisible by `num_groups`.
    pub fn new(weight: &[f32], bias: &[f32], num_groups: usize, eps: f32) -> VaeResult<Self> {
        if weight.len() != bias.len() {
            return Err(VaeError::Shape(format!(
                "groupnorm weight len {} != bias len {}",
                weight.len(),
                bias.len()
            )));
        }
        let channels = weight.len();
        if num_groups == 0 || channels % num_groups != 0 {
            return Err(VaeError::Shape(format!(
                "groupnorm channels {channels} not divisible by num_groups {num_groups}"
            )));
        }
        Ok(Self {
            num_groups,
            weight: weight.to_vec(),
            bias: bias.to_vec(),
            channels,
            eps,
        })
    }

    /// Normalise an NCHW buffer `[C, H, W]` (batch 1) in place.
    ///
    /// # Errors
    /// [`VaeError::Shape`] if `x.len() != channels * h * w`.
    pub fn forward_inplace(&self, x: &mut [f32], h: usize, w: usize) -> VaeResult<()> {
        let hw = h * w;
        if x.len() != self.channels * hw {
            return Err(VaeError::Shape(format!(
                "groupnorm input len {} != C*H*W {}",
                x.len(),
                self.channels * hw
            )));
        }
        // GPU-first: route through the Metal f32 GroupNorm when the `metal`
        // feature is built, on macOS, and `OXI_VAE_GPU=1`. On ANY error silently
        // fall through to the CPU path below.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if crate::vae::gpu::vae_gpu_enabled()
                && crate::vae::gpu::groupnorm_gpu(
                    x,
                    &self.weight,
                    &self.bias,
                    self.channels,
                    hw,
                    self.num_groups,
                    self.eps,
                )
                .is_ok()
            {
                return Ok(());
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
            if crate::vae::cuda_gpu::vae_gpu_enabled()
                && crate::vae::cuda_gpu::groupnorm_gpu(
                    x,
                    &self.weight,
                    &self.bias,
                    self.channels,
                    hw,
                    self.num_groups,
                    self.eps,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
        let gs = self.channels / self.num_groups; // channels per group
        let group_elems = gs * hw;
        let inv_n = 1.0f64 / group_elems as f64;
        for g in 0..self.num_groups {
            let c0 = g * gs;
            let base = c0 * hw;
            let group = &mut x[base..base + group_elems];
            // Mean / population variance in f64 for stability.
            let mut mean = 0.0f64;
            for &v in group.iter() {
                mean += v as f64;
            }
            mean *= inv_n;
            let mut var = 0.0f64;
            for &v in group.iter() {
                let d = v as f64 - mean;
                var += d * d;
            }
            var *= inv_n;
            let inv_std = (1.0 / (var + self.eps as f64).sqrt()) as f32;
            let mean_f = mean as f32;
            // Normalise + per-channel affine.
            for ci in 0..gs {
                let c = c0 + ci;
                let wgt = self.weight[c];
                let bia = self.bias[c];
                let chan = &mut group[ci * hw..(ci + 1) * hw];
                for v in chan.iter_mut() {
                    *v = (*v - mean_f) * inv_std * wgt + bia;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_groups_normalize_independently() {
        // 2 channels, 1x2 spatial, num_groups=2 => each channel its own group.
        // Affine identity (weight=1, bias=0). Each group is zero-mean/unit-var.
        let weight = vec![1.0, 1.0];
        let bias = vec![0.0, 0.0];
        let gn = GroupNorm::new(&weight, &bias, 2, 0.0).expect("gn");
        let mut x = vec![1.0, 3.0, 10.0, 14.0]; // ch0=[1,3], ch1=[10,14]
        gn.forward_inplace(&mut x, 1, 2).expect("fwd");
        // ch0: mean 2, std 1 => [-1, 1]; ch1: mean 12, std 2 => [-1, 1].
        for v in &x {
            assert!((v.abs() - 1.0).abs() < 1e-5, "{v}");
        }
        assert!(x[0] < 0.0 && x[1] > 0.0 && x[2] < 0.0 && x[3] > 0.0);
    }

    #[test]
    fn affine_is_applied_per_channel() {
        // 2 channels in ONE group: normalize over both, then per-channel affine.
        let weight = vec![2.0, 0.5];
        let bias = vec![1.0, -1.0];
        let gn = GroupNorm::new(&weight, &bias, 1, 0.0).expect("gn");
        let mut x = vec![0.0, 2.0, 0.0, 2.0]; // both channels [0,2]
        gn.forward_inplace(&mut x, 1, 2).expect("fwd");
        // group mean=1, var=1 => normalized [-1,1,-1,1].
        // ch0 affine: 2*n + 1 => [-1, 3]; ch1 affine: 0.5*n - 1 => [-1.5, -0.5].
        assert!((x[0] - (-1.0)).abs() < 1e-5);
        assert!((x[1] - 3.0).abs() < 1e-5);
        assert!((x[2] - (-1.5)).abs() < 1e-5);
        assert!((x[3] - (-0.5)).abs() < 1e-5);
    }
}
