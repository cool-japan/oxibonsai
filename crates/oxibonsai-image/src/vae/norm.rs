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

/// Per-group mean and inv_std computed over the full spatial plane.
/// Used by the two-pass GroupNorm tiling path.
#[derive(Clone)]
pub struct GroupStats {
    /// Per-group mean, length = `num_groups`.
    pub mean: Vec<f32>,
    /// Per-group inverse standard deviation (`1 / sqrt(var + eps)`),
    /// length = `num_groups`.
    pub inv_std: Vec<f32>,
}

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

    /// Pass 1 of the two-pass tiling GroupNorm: compute per-group `(mean,
    /// inv_std)` over the **full** plane `x` of shape `[self.channels, h, w]`.
    /// Does NOT modify `x`.  The returned [`GroupStats`] can then be fed to
    /// [`Self::apply_precomputed`] on each tile window.
    ///
    /// The accumulation mirrors `forward_inplace` exactly (f64 accumulators,
    /// population variance, `eps`).
    ///
    /// # Errors
    /// [`VaeError::Shape`] if `x.len() != self.channels * h * w`.
    pub fn compute_stats(&self, x: &[f32], h: usize, w: usize) -> VaeResult<GroupStats> {
        let hw = h * w;
        if x.len() != self.channels * hw {
            return Err(VaeError::Shape(format!(
                "groupnorm compute_stats input len {} != C*H*W {}",
                x.len(),
                self.channels * hw
            )));
        }
        let gs = self.channels / self.num_groups;
        let group_elems = gs * hw;
        let inv_n = 1.0f64 / group_elems as f64;
        let mut mean_out = Vec::with_capacity(self.num_groups);
        let mut inv_std_out = Vec::with_capacity(self.num_groups);
        for g in 0..self.num_groups {
            let c0 = g * gs;
            let base = c0 * hw;
            let group = &x[base..base + group_elems];
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
            mean_out.push(mean as f32);
            inv_std_out.push(inv_std);
        }
        Ok(GroupStats {
            mean: mean_out,
            inv_std: inv_std_out,
        })
    }

    /// Pass 2 of the two-pass tiling GroupNorm: apply precomputed `stats`
    /// (from [`Self::compute_stats`] over the full plane) to a tile window
    /// `x` of shape `[self.channels, h, w]`.  Applies the per-channel affine
    /// (`weight`, `bias`), matching the apply step in `forward_inplace`.
    ///
    /// # Errors
    /// [`VaeError::Shape`] if `x.len() != self.channels * h * w` or
    /// `stats.mean.len() != self.num_groups`.
    pub fn apply_precomputed(
        &self,
        x: &mut [f32],
        h: usize,
        w: usize,
        stats: &GroupStats,
    ) -> VaeResult<()> {
        let hw = h * w;
        if x.len() != self.channels * hw {
            return Err(VaeError::Shape(format!(
                "groupnorm apply_precomputed input len {} != C*H*W {}",
                x.len(),
                self.channels * hw
            )));
        }
        if stats.mean.len() != self.num_groups || stats.inv_std.len() != self.num_groups {
            return Err(VaeError::Shape(format!(
                "groupnorm apply_precomputed stats len {}/{} != num_groups {}",
                stats.mean.len(),
                stats.inv_std.len(),
                self.num_groups
            )));
        }
        let gs = self.channels / self.num_groups;
        for g in 0..self.num_groups {
            let mean_f = stats.mean[g];
            let inv_std = stats.inv_std[g];
            let c0 = g * gs;
            for ci in 0..gs {
                let c = c0 + ci;
                let wgt = self.weight[c];
                let bia = self.bias[c];
                let base = c * hw;
                for v in x[base..base + hw].iter_mut() {
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
    fn groupnorm_two_pass_equals_one_pass() {
        // 4 channels, 2 groups (gs=2), 3x3 spatial.
        // Deterministic values: channel c, pixel p -> (c*9 + p) as f32 + 0.5.
        let c = 4usize;
        let h = 3usize;
        let w = 3usize;
        let weight: Vec<f32> = (0..c).map(|i| 1.0 + 0.1 * i as f32).collect();
        let bias: Vec<f32> = (0..c).map(|i| -0.05 * i as f32).collect();
        let gn = GroupNorm::new(&weight, &bias, 2, 1e-6).expect("gn");
        let x_orig: Vec<f32> = (0..c * h * w).map(|i| 0.5 + i as f32 * 0.37).collect();

        // One-pass reference.
        let mut x_one = x_orig.clone();
        gn.forward_inplace(&mut x_one, h, w).expect("one-pass");

        // Two-pass: compute stats over full plane, then apply to same plane.
        let stats = gn.compute_stats(&x_orig, h, w).expect("stats");
        let mut x_two = x_orig.clone();
        gn.apply_precomputed(&mut x_two, h, w, &stats)
            .expect("apply");

        for (i, (&a, &b)) in x_one.iter().zip(x_two.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "two-pass != one-pass at index {i}: one={a} two={b}"
            );
        }
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
