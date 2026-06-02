//! Element-wise / reshape ops for the VAE decode path: BatchNorm-stats denorm,
//! 2×2 unpatchify, nearest ×2 upsample, and in-place SiLU.
//!
//! All buffers are flat NCHW (`[C, H, W]`, batch 1).

use crate::math::silu;
use crate::vae::error::{VaeError, VaeResult};

/// Denormalise packed latents with stored BatchNorm stats:
/// `out = x * sqrt(var + eps) + mean`, with `mean`/`var` broadcast per channel.
///
/// `x` is NCHW `[C, H, W]`; `mean`/`var` are `[C]`.
///
/// # Errors
/// [`VaeError::Shape`] if `mean`/`var` lengths disagree with `C`, or `x` length
/// disagrees with `C * h * w`.
pub fn bn_denorm(
    x: &[f32],
    mean: &[f32],
    var: &[f32],
    c: usize,
    h: usize,
    w: usize,
    eps: f32,
) -> VaeResult<Vec<f32>> {
    let hw = h * w;
    if x.len() != c * hw {
        return Err(VaeError::Shape(format!(
            "bn_denorm input len {} != C*H*W {}",
            x.len(),
            c * hw
        )));
    }
    if mean.len() != c || var.len() != c {
        return Err(VaeError::Shape(format!(
            "bn_denorm mean/var len ({}/{}) != C {c}",
            mean.len(),
            var.len()
        )));
    }
    let mut out = vec![0.0f32; c * hw];
    for ci in 0..c {
        let std = (var[ci] + eps).sqrt();
        let m = mean[ci];
        let src = &x[ci * hw..(ci + 1) * hw];
        let dst = &mut out[ci * hw..(ci + 1) * hw];
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s * std + m;
        }
    }
    Ok(out)
}

/// Unpatchify packed latents `[C, H, W]` → `[C/4, H*2, W*2]`.
///
/// Mirrors the MLX op on NCHW `[1, C, H, W]`:
/// `reshape(1, C/4, 2, 2, H, W) → transpose(0,1,4,2,5,3) → reshape(1, C/4, H*2,
/// W*2)`. Concretely, output `(a, h_out, w_out)` reads input channel
/// `a*4 + (h_out % 2) * 2 + (w_out % 2)` at spatial `(h_out / 2, w_out / 2)`.
///
/// # Errors
/// [`VaeError::Shape`] if `C` is not a multiple of 4 or the length is wrong.
pub fn unpatchify(x: &[f32], c: usize, h: usize, w: usize) -> VaeResult<UnpatchOut> {
    if c % 4 != 0 {
        return Err(VaeError::Shape(format!(
            "unpatchify channels {c} not a multiple of 4"
        )));
    }
    let hw = h * w;
    if x.len() != c * hw {
        return Err(VaeError::Shape(format!(
            "unpatchify input len {} != C*H*W {}",
            x.len(),
            c * hw
        )));
    }
    let c_out = c / 4;
    let h_out = h * 2;
    let w_out = w * 2;
    let mut out = vec![0.0f32; c_out * h_out * w_out];
    for a in 0..c_out {
        for ho in 0..h_out {
            let hh = ho / 2;
            let b = ho % 2;
            for wo in 0..w_out {
                let ww = wo / 2;
                let d = wo % 2;
                let c_in = a * 4 + b * 2 + d;
                out[(a * h_out + ho) * w_out + wo] = x[(c_in * h + hh) * w + ww];
            }
        }
    }
    Ok(UnpatchOut {
        data: out,
        c: c_out,
        h: h_out,
        w: w_out,
    })
}

/// Result of [`unpatchify`]: NCHW data plus new dims.
pub struct UnpatchOut {
    /// NCHW data `[c, h, w]`.
    pub data: Vec<f32>,
    /// New channel count.
    pub c: usize,
    /// New height.
    pub h: usize,
    /// New width.
    pub w: usize,
}

/// Nearest-neighbour ×2 upsample of an NCHW buffer `[C, H, W]` → `[C, 2H, 2W]`
/// (each pixel repeated 2× along H and W), matching MLX
/// `repeat(x, 2, axis=H); repeat(., 2, axis=W)`.
///
/// # Errors
/// [`VaeError::Shape`] if the input length disagrees with `C * h * w`.
pub fn upsample_nearest2x(x: &[f32], c: usize, h: usize, w: usize) -> VaeResult<UnpatchOut> {
    let hw = h * w;
    if x.len() != c * hw {
        return Err(VaeError::Shape(format!(
            "upsample input len {} != C*H*W {}",
            x.len(),
            c * hw
        )));
    }
    // GPU-first: route through the Metal nearest ×2 upsample when the `metal`
    // feature is built, on macOS, and `OXI_VAE_GPU=1`. On ANY error silently
    // fall through to the CPU path below.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if crate::vae::gpu::vae_gpu_enabled() {
            if let Ok(up) = crate::vae::gpu::upsample_gpu(x, c, h, w) {
                return Ok(UnpatchOut {
                    data: up.data,
                    c,
                    h: up.h,
                    w: up.w,
                });
            }
        }
    }
    // CUDA sibling of the Metal block above (target_os-disjoint: Linux/Windows).
    // Same `OXI_VAE_GPU` toggle; on ANY error silently fall through to the CPU
    // path below.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        if crate::vae::cuda_gpu::vae_gpu_enabled() {
            if let Ok(up) = crate::vae::cuda_gpu::upsample_gpu(x, c, h, w) {
                return Ok(UnpatchOut {
                    data: up.data,
                    c,
                    h: up.h,
                    w: up.w,
                });
            }
        }
    }
    let h_out = h * 2;
    let w_out = w * 2;
    let mut out = vec![0.0f32; c * h_out * w_out];
    for ci in 0..c {
        let src = &x[ci * hw..(ci + 1) * hw];
        let dst = &mut out[ci * h_out * w_out..(ci + 1) * h_out * w_out];
        for ho in 0..h_out {
            let hh = ho / 2;
            for wo in 0..w_out {
                let ww = wo / 2;
                dst[ho * w_out + wo] = src[hh * w + ww];
            }
        }
    }
    Ok(UnpatchOut {
        data: out,
        c,
        h: h_out,
        w: w_out,
    })
}

/// Apply SiLU element-wise, in place.
pub fn silu_inplace(x: &mut [f32]) {
    // GPU-first: route through the Metal SiLU when the `metal` feature is built,
    // on macOS, and `OXI_VAE_GPU=1`. On ANY error silently fall through to the
    // CPU path below.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if crate::vae::gpu::vae_gpu_enabled() && crate::vae::gpu::silu_gpu(x).is_ok() {
            return;
        }
    }
    // CUDA sibling of the Metal block above (target_os-disjoint: Linux/Windows).
    // Same `OXI_VAE_GPU` toggle; on ANY error silently fall through to the CPU
    // path below.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        if crate::vae::cuda_gpu::vae_gpu_enabled() && crate::vae::cuda_gpu::silu_gpu(x).is_ok() {
            return;
        }
    }
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bn_denorm_matches_formula() {
        // x*sqrt(var+eps)+mean, per channel.
        let x = vec![1.0, -1.0, 2.0, 0.0]; // ch0=[1,-1], ch1=[2,0]
        let mean = vec![10.0, -5.0];
        let var = vec![3.0, 0.0];
        let out = bn_denorm(&x, &mean, &var, 2, 1, 2, 1.0).expect("bn");
        // std0 = sqrt(3+1)=2 => [12, 8]; std1 = sqrt(0+1)=1 => [-3, -5].
        assert_eq!(out, vec![12.0, 8.0, -3.0, -5.0]);
    }

    #[test]
    fn upsample_repeats_2x() {
        // 1 channel, 1x2 => 2x4, each pixel duplicated in H and W.
        let x = vec![1.0, 2.0];
        let up = upsample_nearest2x(&x, 1, 1, 2).expect("up");
        assert_eq!((up.c, up.h, up.w), (1, 2, 4));
        // rows identical (H repeat), cols pairwise equal (W repeat).
        assert_eq!(up.data, vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn unpatchify_inverts_patch_layout() {
        // C=4, H=W=1 => C_out=1, 2x2 output. The 4 channels become the 2x2
        // block in (b,d) order: c_in = (h_out%2)*2 + (w_out%2).
        // channel values 10,11,12,13 at the single pixel.
        let x = vec![10.0, 11.0, 12.0, 13.0];
        let out = unpatchify(&x, 4, 1, 1).expect("unpatch");
        assert_eq!((out.c, out.h, out.w), (1, 2, 2));
        // (0,0)->c0=10, (0,1)->c1=11, (1,0)->c2=12, (1,1)->c3=13.
        assert_eq!(out.data, vec![10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn unpatchify_then_dims_double() {
        // C=128,H=W=32 => C_out=32, 64x64 (the real decode shapes).
        let x = vec![0.0f32; 128 * 32 * 32];
        let out = unpatchify(&x, 128, 32, 32).expect("unpatch");
        assert_eq!((out.c, out.h, out.w), (32, 64, 64));
    }
}
