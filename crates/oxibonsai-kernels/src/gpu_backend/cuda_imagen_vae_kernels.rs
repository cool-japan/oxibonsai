//! CUDA C kernel sources for the **FLUX.2 VAE decoder** per-op f32 GPU
//! primitives (CUDA mirror of the Metal `kernel_sources/vae.rs` kernels).
//!
//! These four element/structural kernels accelerate the pure-CPU VAE decode of
//! the FLUX.2 image model (the convs are ~60 % of the VAE FLOPs at 512×512, and
//! the CPU path is im2col-memory-bound). They are purely additive — no existing
//! kernel is touched — and each is a faithful port of the proven Metal kernel,
//! to be parity-validated against the CPU reference in `oxibonsai-image::vae`
//! (`conv.rs`, `norm.rs`, `ops.rs`) once compiled on Linux/Windows.
//!
//! All buffers are flat NCHW `[C, H, W]` f32 (batch 1), matching the CPU stack.
//!
//! | Kernel                  | Op | Metal mirror / CPU reference |
//! |-------------------------|----|------------------------------|
//! | `im2col_f32`            | im2col patch extraction `[rows, kH·kW·C_in]` in `(kH,kW,C_in)` order | `MSL_IM2COL_F32` / `conv.rs::build_im2col` |
//! | `groupnorm_f32`         | GroupNorm (32 groups, eps 1e-6, per-channel affine) | `MSL_GROUPNORM_F32` / `norm.rs::forward_inplace` |
//! | `silu_f32`              | element-wise `x / (1 + exp(-x))` | `MSL_SILU_F32` / `math::silu` |
//! | `upsample_nearest_f32`  | nearest ×2 `[C,H,W] → [C,2H,2W]` | `MSL_UPSAMPLE_NEAREST_F32` / `ops::upsample_nearest2x` |
//!
//! The `im2col_f32` output feeds the existing parity-clean f32 GEMM kernel
//! (`gemm_f32` in `cuda_imagen_gemm_kernels.rs`). The `(kH,kW,C_in)` element
//! order is exactly the MLX conv-weight `[C_out, kH, kW, C_in]` flattening, so
//! the GEMM dot is correct with the weight passed directly as `W = [N=C_out,
//! K=kH·kW·C_in]`.
//!
//! # Numerics
//!
//! Unlike Metal (no `double`), CUDA has `double`, so `groupnorm_f32` accumulates
//! the per-group mean / variance in **f64** — exactly mirroring the f64 CPU
//! reference in `norm.rs` (strictly tighter than Metal's Kahan-compensated f32).
//!
//! # Build
//!
//! Compiled once at process startup via `cudarc::nvrtc::compile_ptx`; no CUDA
//! SDK headers required (all intrinsics — `expf`, `rsqrt`, `__shared__`,
//! `__syncthreads` — are NVRTC built-ins).

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

/// Combined CUDA C source for the four FLUX.2 VAE-decoder f32 primitives.
///
/// Entry points (`extern "C" __global__`): `im2col_f32`, `groupnorm_f32`,
/// `silu_f32`, `upsample_nearest_f32`.
pub const CUDA_IMAGEN_VAE_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA kernels — FLUX.2 VAE decoder per-op f32 primitives.
   No CUDA SDK headers required; all intrinsics are NVRTC built-ins.
   CUDA mirror of crates/oxibonsai-kernels/src/gpu_backend/kernel_sources/vae.rs
   ========================================================================= */

/* =========================================================================
   Kernel 1 — im2col_f32
   im2col patch extraction for a TILE of output rows, in (kH, kW, C_in) order
   (the MLX conv-weight [C_out, kH, kW, C_in] flattening — LOAD-BEARING).

   Produces patches[local_row * patch_dim + j] for output spatial rows
   [row_start, row_start + tile_rows), with patch_dim = k*k*c_in and
   j = (kh*k + kw)*c_in + ci. Zero padding: out-of-range source → 0.

   input   : NCHW [C_in, H*W]  (input[ci*H*W + ih*W + iw])
   patches : [tile_rows, patch_dim]  (written)

   Grid:  (ceil(n_elems / 256), 1, 1)   Block: (256, 1, 1)
   (one thread per patch element; n_elems = tile_rows * patch_dim)
   ========================================================================= */
extern "C" __global__ void im2col_f32(
    const float* __restrict__ input,
    float*       __restrict__ patches,
    unsigned int c_in,
    unsigned int h,
    unsigned int w,
    unsigned int k,
    unsigned int pad,
    unsigned int w_out,
    unsigned int row_start,
    unsigned int n_elems
) {
    const unsigned int gid = blockIdx.x * 256u + threadIdx.x;
    if (gid >= n_elems) return;

    const unsigned int patch_dim = k * k * c_in;
    const unsigned int local_row = gid / patch_dim;   /* 0..tile_rows-1 */
    const unsigned int j         = gid % patch_dim;   /* 0..patch_dim-1 */

    /* Output spatial position (global, across the full H_out*W_out plane). */
    const unsigned int out_idx = row_start + local_row;
    const unsigned int oh = out_idx / w_out;
    const unsigned int ow = out_idx % w_out;

    /* Decompose j = (kh*k + kw)*c_in + ci  ->  (kh, kw, ci). */
    const unsigned int kc  = k * c_in;
    const unsigned int kh  = j / kc;
    const unsigned int rem = j % kc;
    const unsigned int kw  = rem / c_in;
    const unsigned int ci  = rem % c_in;

    /* Padded source coordinates (same geometry as the CPU build_im2col). */
    const unsigned int ih_p = oh + kh;
    const unsigned int iw_p = ow + kw;

    float v = 0.0f;
    if (ih_p >= pad && ih_p < h + pad && iw_p >= pad && iw_p < w + pad) {
        const unsigned int ih = ih_p - pad;
        const unsigned int iw = iw_p - pad;
        v = input[(unsigned long long)ci * (unsigned long long)h * (unsigned long long)w
                + (unsigned long long)ih * (unsigned long long)w
                + (unsigned long long)iw];
    }
    patches[gid] = v;
}

/* =========================================================================
   Kernel 2 — groupnorm_f32
   PyTorch-compatible GroupNorm, in place on NCHW [C, H*W].

   One block per group (num_groups, 32 in the VAE). Each group spans
   gs = C / num_groups contiguous channels x hw spatial positions; the
   per-group population mean / variance are reduced over all gs*hw elements
   (accumulated in DOUBLE to match the f64 CPU reference in norm.rs), then
   (x - mean) / sqrt(var + eps) is scaled by the per-channel affine
   weight[c] / bias[c].

   x      : NCHW [C, hw]  (read-write / in-place)
   weight : [C] per-channel scale
   bias   : [C] per-channel shift

   Grid:  (num_groups, 1, 1)   Block: (256, 1, 1)
   ========================================================================= */
extern "C" __global__ void groupnorm_f32(
    float*       __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    unsigned int channels,
    unsigned int hw,
    unsigned int num_groups,
    float eps
) {
    const unsigned int GN_THREADS = 256u;
    __shared__ double red[256];
    __shared__ double mean_sh;
    __shared__ double inv_std_sh;

    const unsigned int g = blockIdx.x;          /* group index (one block/group) */
    if (g >= num_groups) return;
    const unsigned int gs = channels / num_groups;   /* channels per group */
    const unsigned long long group_elems = (unsigned long long)gs * (unsigned long long)hw;
    const unsigned long long base = (unsigned long long)(g * gs) * (unsigned long long)hw;

    const unsigned int lid = threadIdx.x;

    /* -- Pass 1: f64 partial sum over this thread's stride. -- */
    double sum = 0.0;
    for (unsigned long long i = lid; i < group_elems; i += GN_THREADS) {
        sum += (double)x[base + i];
    }
    red[lid] = sum;
    __syncthreads();
    /* Tree reduction of the 256 partials (f64). */
    for (unsigned int stride = GN_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) red[lid] += red[lid + stride];
        __syncthreads();
    }
    if (lid == 0u) {
        mean_sh = red[0] / (double)group_elems;
    }
    __syncthreads();
    const double mean = mean_sh;

    /* -- Pass 2: f64 partial sum of (x - mean)^2. -- */
    double vsum = 0.0;
    for (unsigned long long i = lid; i < group_elems; i += GN_THREADS) {
        const double d = (double)x[base + i] - mean;
        vsum += d * d;
    }
    red[lid] = vsum;
    __syncthreads();
    for (unsigned int stride = GN_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) red[lid] += red[lid + stride];
        __syncthreads();
    }
    if (lid == 0u) {
        const double var = red[0] / (double)group_elems;
        inv_std_sh = rsqrt(var + (double)eps);
    }
    __syncthreads();
    const double inv_std = inv_std_sh;

    /* -- Pass 3: normalize + per-channel affine, in place. -- */
    /* Element i (within the group) belongs to channel c0 + i/hw. */
    const unsigned int c0 = g * gs;
    for (unsigned long long i = lid; i < group_elems; i += GN_THREADS) {
        const unsigned int ci = (unsigned int)(i / (unsigned long long)hw); /* local channel 0..gs-1 */
        const unsigned int c  = c0 + ci;
        const double wgt = (double)weight[c];
        const double bia = (double)bias[c];
        x[base + i] = (float)(((double)x[base + i] - mean) * inv_std * wgt + bia);
    }
}

/* =========================================================================
   Kernel 3 — silu_f32
   Element-wise SiLU (x * sigmoid(x) = x / (1 + exp(-x))), in place.
   Mirrors oxibonsai-image::math::silu exactly.

   Grid:  (ceil(n / 256), 1, 1)   Block: (256, 1, 1)
   ========================================================================= */
extern "C" __global__ void silu_f32(
    float*       __restrict__ x,
    unsigned int n
) {
    const unsigned int gid = blockIdx.x * 256u + threadIdx.x;
    if (gid >= n) return;
    const float v = x[gid];
    x[gid] = v / (1.0f + expf(-v));
}

/* =========================================================================
   Kernel 4 — upsample_nearest_f32
   Nearest-neighbour x2 upsample of an NCHW buffer [C, H, W] -> [C, 2H, 2W]
   (each pixel repeated 2x along H and W). Mirrors
   oxibonsai-image::vae::ops::upsample_nearest2x.

   input  : [C, H*W]
   output : [C, 2H*2W]  (written, n_out = C*2H*2W)

   Grid:  (ceil(n_out / 256), 1, 1)   Block: (256, 1, 1)
   (one thread per output element)
   ========================================================================= */
extern "C" __global__ void upsample_nearest_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    unsigned int c,
    unsigned int h,
    unsigned int w
) {
    const unsigned int n_out = c * (2u * h) * (2u * w);
    const unsigned int gid = blockIdx.x * 256u + threadIdx.x;
    if (gid >= n_out) return;
    const unsigned int w_out  = w * 2u;
    const unsigned int hw_out = h * 2u * w_out;   /* output elements per channel */
    const unsigned int ch = gid / hw_out;
    const unsigned int r  = gid % hw_out;
    const unsigned int ho = r / w_out;
    const unsigned int wo = r % w_out;
    const unsigned int hh = ho / 2u;
    const unsigned int ww = wo / 2u;
    output[gid] = input[(unsigned long long)ch * (unsigned long long)h * (unsigned long long)w
                      + (unsigned long long)hh * (unsigned long long)w
                      + (unsigned long long)ww];
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The four VAE-decoder entry-point names must be present in the source.
    #[test]
    fn src_has_entry_points() {
        for needle in [
            "im2col_f32",
            "groupnorm_f32",
            "silu_f32",
            "upsample_nearest_f32",
        ] {
            assert!(
                CUDA_IMAGEN_VAE_SRC.contains(needle),
                "CUDA_IMAGEN_VAE_SRC missing entry point `{needle}`"
            );
        }
    }
}
