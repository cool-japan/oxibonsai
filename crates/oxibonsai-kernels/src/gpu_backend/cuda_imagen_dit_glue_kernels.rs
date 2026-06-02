//! CUDA-C kernel source for the FLUX.2 DiT "glue" ops — the per-block element-
//! wise / reduction primitives that sit between the ternary matmuls and the
//! joint flash-attention (LayerNorm, modulation, QK-RMSNorm, SwiGLU, interleaved
//! RoPE, gated residual add).
//!
//! These exist so the DiT forward can stay **GPU-resident**: today every op
//! round-trips the activation host↔device (profiling: ~96% of CUDA-API CPU time
//! is the pageable device→host downloads). With a device-side kernel for each
//! glue op, the matmul outputs feed the norms/attention without touching the
//! host. Each kernel is a **parity-first plain FP32** port of the matching CPU
//! reference in `oxibonsai-image::math` (`layer_norm_inplace`, `modulate_inplace`,
//! `rms_norm_heads_inplace`, `swiglu`, `apply_rope_inplace`) and
//! `oxibonsai-image::blocks::gated_residual_add` — validated at cos ≥ 0.999.
//!
//! All buffers are flat f32. No CUDA SDK headers required; `expf`/`sqrt` are
//! NVRTC built-ins. Reductions accumulate in `double` (CUDA has `double`), at
//! least as tight as the CPU f32 reference.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

/// Combined CUDA-C source for the six DiT glue kernels. Compiled once at startup
/// via `cudarc::nvrtc::compile_ptx`.
///
/// Entry points: `modulate_f32`, `gated_residual_add_f32`, `layer_norm_f32`,
/// `rms_norm_heads_f32`, `swiglu_f32`, `rope_interleaved_f32`.
pub const CUDA_IMAGEN_DIT_GLUE_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA imagen kernels — FLUX.2 DiT glue ops (parity prototype).
   Plain FP32, double-accumulate reductions. Mirror oxibonsai-image::math /
   ::blocks. No CUDA SDK headers; expf/sqrt are NVRTC built-ins.
   ========================================================================= */

/* SiLU: x * sigmoid(x) = x / (1 + exp(-x)). Matches math::silu. */
static __device__ __forceinline__ float silu_f32_dev(float x) {
    return x / (1.0f + expf(-x));
}

/* ── modulate_f32 ──────────────────────────────────────────────────────────
   In-place y = (1 + scale[i]) * x + shift[i] over x[rows, dim]; scale/shift are
   length-`dim`, broadcast over rows. Mirrors math::modulate_inplace. */
extern "C" __global__ void modulate_f32(
    float* __restrict__ x,
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    unsigned int rows,
    unsigned int dim
) {
    const unsigned long long total = (unsigned long long)rows * (unsigned long long)dim;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    for (unsigned long long idx =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const unsigned int i = (unsigned int)(idx % (unsigned long long)dim);
        x[idx] = (1.0f + scale[i]) * x[idx] + shift[i];
    }
}

/* ── gated_residual_add_f32 ─────────────────────────────────────────────────
   In-place h[r,i] += gate[i] * delta[r,i] over [rows, dim]; gate length `dim`,
   broadcast over rows. Mirrors blocks::gated_residual_add. */
extern "C" __global__ void gated_residual_add_f32(
    float* __restrict__ h,
    const float* __restrict__ delta,
    const float* __restrict__ gate,
    unsigned int rows,
    unsigned int dim
) {
    const unsigned long long total = (unsigned long long)rows * (unsigned long long)dim;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    for (unsigned long long idx =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const unsigned int i = (unsigned int)(idx % (unsigned long long)dim);
        h[idx] += gate[i] * delta[idx];
    }
}

/* ── layer_norm_f32 (affine = false) ────────────────────────────────────────
   In-place, one block per row: y = (x - mean) / sqrt(var + eps), population
   variance, TWO-PASS (mean then var) to match math::layer_norm_inplace. Block =
   (LN_THREADS,1,1); reduces over `dim` in double. */
#define LN_THREADS 256u
extern "C" __global__ void layer_norm_f32(
    float* __restrict__ x,
    unsigned int rows,
    unsigned int dim,
    float eps
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    float* __restrict__ xr = x + (unsigned long long)row * (unsigned long long)dim;
    const unsigned int tid = threadIdx.x;

    __shared__ double red[LN_THREADS];

    /* Pass 1: sum → mean. */
    double s = 0.0;
    for (unsigned int i = tid; i < dim; i += LN_THREADS) {
        s += (double)xr[i];
    }
    red[tid] = s;
    __syncthreads();
    for (unsigned int off = LN_THREADS >> 1; off > 0u; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const double mean = red[0] / (double)dim;
    __syncthreads();

    /* Pass 2: sum (x-mean)^2 → var. */
    double v = 0.0;
    for (unsigned int i = tid; i < dim; i += LN_THREADS) {
        const double d = (double)xr[i] - mean;
        v += d * d;
    }
    red[tid] = v;
    __syncthreads();
    for (unsigned int off = LN_THREADS >> 1; off > 0u; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const double var = red[0] / (double)dim;
    const double inv_std = 1.0 / sqrt(var + (double)eps);

    for (unsigned int i = tid; i < dim; i += LN_THREADS) {
        xr[i] = (float)(((double)xr[i] - mean) * inv_std);
    }
}

/* ── rms_norm_heads_f32 (QK-RMSNorm) ────────────────────────────────────────
   In-place, one block per `head_dim` chunk (rows = num_heads*seq chunks):
   y = weight[i] * x / sqrt(mean(x^2) + eps). Single-pass sumsq in double.
   Mirrors math::rms_norm_heads_inplace. */
#define RMS_THREADS 128u
extern "C" __global__ void rms_norm_heads_f32(
    float* __restrict__ x,
    const float* __restrict__ weight,
    unsigned int rows,
    unsigned int head_dim,
    float eps
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    float* __restrict__ xr = x + (unsigned long long)row * (unsigned long long)head_dim;
    const unsigned int tid = threadIdx.x;

    __shared__ double red[RMS_THREADS];
    double ss = 0.0;
    for (unsigned int i = tid; i < head_dim; i += RMS_THREADS) {
        const double v = (double)xr[i];
        ss += v * v;
    }
    red[tid] = ss;
    __syncthreads();
    for (unsigned int off = RMS_THREADS >> 1; off > 0u; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const double ms = red[0] / (double)head_dim;
    const double inv_rms = 1.0 / sqrt(ms + (double)eps);
    for (unsigned int i = tid; i < head_dim; i += RMS_THREADS) {
        xr[i] = (float)((double)weight[i] * (double)xr[i] * inv_rms);
    }
}

/* ── swiglu_f32 ─────────────────────────────────────────────────────────────
   out[r, i] = silu(x[r, i]) * x[r, half + i] over a [rows, 2*half] input,
   producing [rows, half]. Mirrors math::swiglu. */
extern "C" __global__ void swiglu_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int rows,
    unsigned int half
) {
    const unsigned long long total = (unsigned long long)rows * (unsigned long long)half;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    const unsigned int full = half * 2u;
    for (unsigned long long idx =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const unsigned int r = (unsigned int)(idx / (unsigned long long)half);
        const unsigned int i = (unsigned int)(idx % (unsigned long long)half);
        const unsigned long long base = (unsigned long long)r * (unsigned long long)full;
        const float gate = x[base + i];
        const float up = x[base + (unsigned long long)half + i];
        out[idx] = silu_f32_dev(gate) * up;
    }
}

/* ── rope_interleaved_f32 ───────────────────────────────────────────────────
   In-place interleaved (adjacent-pair) RoPE on head-major x[num_heads, seq,
   head_dim]. One thread per (head, token, pair i<half): rotate (x[2i], x[2i+1])
   by (cos[t,i], sin[t,i]). `cost`/`sint` are [seq, half]. Mirrors
   math::apply_rope_inplace. */
extern "C" __global__ void rope_interleaved_f32(
    float* __restrict__ x,
    const float* __restrict__ cost,
    const float* __restrict__ sint,
    unsigned int num_heads,
    unsigned int seq,
    unsigned int head_dim
) {
    const unsigned int half = head_dim >> 1;
    const unsigned long long total =
        (unsigned long long)num_heads * (unsigned long long)seq * (unsigned long long)half;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    for (unsigned long long idx =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const unsigned int i = (unsigned int)(idx % (unsigned long long)half);
        const unsigned long long tmp = idx / (unsigned long long)half;
        const unsigned int t = (unsigned int)(tmp % (unsigned long long)seq);
        const unsigned int h = (unsigned int)(tmp / (unsigned long long)seq);
        const unsigned long long base =
            ((unsigned long long)h * (unsigned long long)seq + (unsigned long long)t)
            * (unsigned long long)head_dim;
        const float real = x[base + (unsigned long long)(2u * i)];
        const float imag = x[base + (unsigned long long)(2u * i + 1u)];
        const float c = cost[(unsigned long long)t * (unsigned long long)half + i];
        const float s = sint[(unsigned long long)t * (unsigned long long)half + i];
        x[base + (unsigned long long)(2u * i)] = real * c - imag * s;
        x[base + (unsigned long long)(2u * i + 1u)] = imag * c + real * s;
    }
}

/* ── tokens_to_heads_f32 (reshape / gather) ─────────────────────────────────
   Gather a token-major slice `src[t, src_off + h*head_dim + d]` (row stride
   `src_stride`, hidden = num_heads*head_dim) into a head-major contiguous
   `dst[h, t, d]` = `dst[(h*seq + t)*head_dim + d]`. Mirrors
   `oxibonsai-image::math::to_heads` applied to a column-slice of a fused proj. */
extern "C" __global__ void tokens_to_heads_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int seq,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int src_stride,
    unsigned int src_off
) {
    const unsigned long long total =
        (unsigned long long)num_heads * (unsigned long long)seq * (unsigned long long)head_dim;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    for (unsigned long long idx =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const unsigned int d = (unsigned int)(idx % (unsigned long long)head_dim);
        const unsigned long long rem = idx / (unsigned long long)head_dim;
        const unsigned int t = (unsigned int)(rem % (unsigned long long)seq);
        const unsigned int h = (unsigned int)(rem / (unsigned long long)seq);
        dst[idx] = src[(unsigned long long)t * (unsigned long long)src_stride
                       + (unsigned long long)src_off
                       + (unsigned long long)h * (unsigned long long)head_dim
                       + (unsigned long long)d];
    }
}

/* ── strided_row_copy_f32 (reshape) ─────────────────────────────────────────
   Per-row slice copy: dst[t*dst_stride + dst_off + j] = src[t*src_stride +
   src_off + j] for t<rows, j<cols. Used to extract the mlp slab from the fused
   proj and to build the [attn ‖ gated] concat without per-row host copies. */
extern "C" __global__ void strided_row_copy_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    unsigned int rows,
    unsigned int cols,
    unsigned int dst_stride,
    unsigned int dst_off,
    unsigned int src_stride,
    unsigned int src_off
) {
    const unsigned long long total = (unsigned long long)rows * (unsigned long long)cols;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    for (unsigned long long idx =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const unsigned int t = (unsigned int)(idx / (unsigned long long)cols);
        const unsigned int j = (unsigned int)(idx % (unsigned long long)cols);
        dst[(unsigned long long)t * (unsigned long long)dst_stride
            + (unsigned long long)dst_off + (unsigned long long)j] =
            src[(unsigned long long)t * (unsigned long long)src_stride
                + (unsigned long long)src_off + (unsigned long long)j];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::CUDA_IMAGEN_DIT_GLUE_SRC;

    #[test]
    fn src_has_all_entry_points() {
        for ep in [
            "modulate_f32",
            "gated_residual_add_f32",
            "layer_norm_f32",
            "rms_norm_heads_f32",
            "swiglu_f32",
            "rope_interleaved_f32",
            "tokens_to_heads_f32",
            "strided_row_copy_f32",
        ] {
            assert!(
                CUDA_IMAGEN_DIT_GLUE_SRC.contains(ep),
                "missing DiT glue entry point: {ep}"
            );
        }
    }

    #[test]
    fn src_is_parity_first_fp32() {
        // No tensor cores / f16 staging in the parity prototype.
        assert!(!CUDA_IMAGEN_DIT_GLUE_SRC.contains("wmma"));
        assert!(!CUDA_IMAGEN_DIT_GLUE_SRC.contains("mma.sync"));
        assert!(!CUDA_IMAGEN_DIT_GLUE_SRC.contains("__half"));
        // double-accumulated reductions (≥ the CPU f32 reference).
        assert!(CUDA_IMAGEN_DIT_GLUE_SRC.contains("__shared__ double"));
    }
}
