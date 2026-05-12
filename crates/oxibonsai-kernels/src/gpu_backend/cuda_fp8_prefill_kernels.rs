//! CUDA C kernel source strings for OxiBonsai FP8 E4M3/E5M2 batch GEMM (prefill) operations.
//!
//! # Prefill kernel catalogue
//!
//! | Kernel                                    | Description                                                   |
//! |-------------------------------------------|---------------------------------------------------------------|
//! | `gemm_fp8_e4m3`                           | Batch GEMM: FP8 E4M3 AoS, col-major I/O, col_sums[8]        |
//! | `gemm_fp8_e4m3_residual`                  | FP8 E4M3 GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_fp8_e4m3`      | Fused gate+up FP8 E4M3 GEMM with SwiGLU epilogue            |
//! | `gemv_fp8_e4m3_pf`                        | Single-token FP8 E4M3 GEMV (for sequential attention pass)  |
//! | `gemm_fp8_e5m2`                           | Batch GEMM: FP8 E5M2 AoS, col-major I/O, col_sums[8]        |
//! | `gemm_fp8_e5m2_residual`                  | FP8 E5M2 GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_fp8_e5m2`      | Fused gate+up FP8 E5M2 GEMM with SwiGLU epilogue            |
//! | `gemv_fp8_e5m2_pf`                        | Single-token FP8 E5M2 GEMV (for sequential attention pass)  |
//!
//! # Block layout (AoS, 34 bytes/block — matches `BlockFP8E4M3` / `BlockFP8E5M2`)
//!
//! ```text
//! bytes  0-31: 32 FP8 quantized weights   (E4M3 or E5M2)
//! bytes 32-33: FP16 LE block scale
//! ```
//!
//! This differs from Q8_0 (scale at bytes 0-1, weights at 2-33).
//! Scale access: `bptr[32] | ((unsigned short)bptr[33] << 8u)`.
//!
//! # Batch tensor layout
//!
//! All batch inputs/outputs use **column-major** layout: `buf[col * dim + element]`
//! where `col` is the batch/token index.
//!
//! # Grid / block config
//!
//! All kernels:
//! - Grid:  `(ceil(n_rows / 8), 1, 1)` — 8 warps per CTA
//! - Block: `(256, 1, 1)` — 8 warps × 32 lanes

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

/// CUDA C source for FP8 E4M3/E5M2 batch GEMM (prefill) kernels.
///
/// All kernels use AoS weight layout (blocks stored contiguously as-is from GGUF).
/// Batch tensors use column-major layout: `buf[col * dim + element]`.
pub const CUDA_FP8_PREFILL_KERNELS_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA FP8 E4M3 / E5M2 prefill (batch GEMM) kernels.

   FP8 AoS block (34 bytes): [q0..q31, scale_lo, scale_hi]
     bytes  0-31: 32 FP8 quantized weights (E4M3 or E5M2)
     bytes 32-33: FP16 LE block scale
   Scale access: bptr[32] | ((unsigned short)bptr[33] << 8u)
   Weight at index w: fp8_pf_e4m3_to_float(bptr[w])  (w in [0, 32))

   Batch tensors: column-major  buf[col * dim + element]
   Grid:  (ceil(n_rows/8), 1, 1)  — 8 warps per CTA
   Block: (256, 1, 1)             — 8 warps × 32 lanes
   ========================================================================= */

/* ── Hardware FP16 → FP32 via PTX (SM 6.0+, 1 instruction) ─────────────── */
static __device__ __forceinline__ float fp8_pf_fast_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* ── SiLU activation: x · σ(x) ─────────────────────────────────────────── */
static __device__ __forceinline__ float fp8_pf_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ── FP8 E4M3FN decode (OFP8, bias=7, 4-bit exp, 3-bit mantissa) ─────────
   Format: s[7] exp[6:3] man[2:0], bias=7
   Normal:  (-1)^s * 2^(exp-7) * (1 + man/8)
   Denorm:  (-1)^s * 2^(-6) * (man/8)
   NaN:     exp=0b1111 AND man=0b111 (patterns 0x7f, 0xff) → 0 for inference
   ─────────────────────────────────────────────────────────────────────────── */
static __device__ __forceinline__ float fp8_pf_e4m3_to_float(unsigned char b) {
    /* NaN patterns: 0x7f and 0xff → treat as 0 for inference */
    if (b == 0x7Fu || b == 0xFFu) return 0.0f;
    const unsigned int sign = (b >> 7u) & 1u;
    const unsigned int exp  = (b >> 3u) & 15u;  /* 4-bit exponent */
    const unsigned int mant = b & 7u;            /* 3-bit mantissa */
    float val;
    if (exp == 0u) {
        /* Denormal: (-1)^s * 2^(-6) * (mant/8) */
        val = (float)mant * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else {
        /* Normal: 2^(exp-7) * (1 + mant/8)
           Assemble as IEEE-754 f32: ((exp - 7 + 127) << 23) | (mant << 20) */
        val = __int_as_float(((exp - 7u + 127u) << 23u) | (mant << 20u));
    }
    return sign ? -val : val;
}

/* ── FP8 E5M2 decode (standard, bias=15, 5-bit exp, 2-bit mantissa) ──────
   Format: s[7] exp[6:2] man[1:0], bias=15
   Normal:  (-1)^s * 2^(exp-15) * (1 + man/4)
   Denorm:  (-1)^s * 2^(-14) * (man/4)
   Inf/NaN: exp=31 → 0 for inference
   ─────────────────────────────────────────────────────────────────────────── */
static __device__ __forceinline__ float fp8_pf_e5m2_to_float(unsigned char b) {
    const unsigned int exp  = (b >> 2u) & 31u;  /* 5-bit exponent */
    const unsigned int mant = b & 3u;            /* 2-bit mantissa */
    if (exp == 31u) return 0.0f;                 /* Inf / NaN → 0 */
    const unsigned int sign = (b >> 7u) & 1u;
    float val;
    if (exp == 0u) {
        /* Denormal: (-1)^s * 2^(-14) * (mant/4) */
        val = (float)mant * (1.0f / 4.0f) * (1.0f / 16384.0f);
    } else {
        /* Normal: 2^(exp-15) * (1 + mant/4)
           Assemble as IEEE-754 f32: ((exp - 15 + 127) << 23) | (mant << 21) */
        val = __int_as_float(((exp - 15u + 127u) << 23u) | (mant << 21u));
    }
    return sign ? -val : val;
}

/* =========================================================================
   Kernel 1 — gemm_fp8_e4m3
   Batch FP8 E4M3 GEMM. Accumulates into outputs with +=.
   blocks: AoS, 34 bytes/block (32 FP8 weights + 2 scale bytes)
   Scale at bytes 32-33, weights at bytes 0-31.
   inputs:  col-major [batch_size * k]
   outputs: col-major [batch_size * n_rows], accumulated with +=
   k must be a positive multiple of 32.
   ========================================================================= */
extern "C" __global__ void gemm_fp8_e4m3(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ inputs,
    float*               __restrict__ outputs,
    unsigned int n_rows,
    unsigned int k,
    unsigned int batch_size
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 5u;  /* k / 32 */

    /* Process batch columns in 8-column outer chunks (cap-of-8 fix). */
    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            /* Load AoS block for this row — 34 bytes/block, weights-first layout */
            const unsigned char* bptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 34u;
            /* Scale is at bytes 32-33 (after the 32 FP8 weight bytes) */
            const unsigned short d_raw = (unsigned short)bptr[32u] | ((unsigned short)bptr[33u] << 8u);
            const float scale = fp8_pf_fast_fp16_to_float(d_raw);
            const unsigned int base = b << 5u;  /* b * 32 */

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float bsum = 0.0f;
                #pragma unroll 8
                for (unsigned int w = 0u; w < 32u; ++w) {
                    bsum += fp8_pf_e4m3_to_float(bptr[w]) * xbase[w];
                }
                col_sums[col] += scale * bsum;
            }
        }

        /* Warp-shuffle reduction and write outputs (column-major, accumulate +=) */
        for (unsigned int col = 0u; col < cols; ++col) {
            float s = col_sums[col];
            s += __shfl_down_sync(0xffffffffu, s, 16u);
            s += __shfl_down_sync(0xffffffffu, s,  8u);
            s += __shfl_down_sync(0xffffffffu, s,  4u);
            s += __shfl_down_sync(0xffffffffu, s,  2u);
            s += __shfl_down_sync(0xffffffffu, s,  1u);
            if (lane == 0u)
                outputs[(unsigned long long)(col_base + col) * n_rows + row] += s;
        }
    }
}

/* =========================================================================
   Kernel 2 — gemm_fp8_e4m3_residual
   Batch FP8 E4M3 GEMM + fused in-place residual add.
   For each (row, col): outputs[col*n_rows+row] = residual[col*n_rows+row] + sum
   ========================================================================= */
extern "C" __global__ void gemm_fp8_e4m3_residual(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ inputs,
    float*               __restrict__ outputs,
    unsigned int n_rows,
    unsigned int k,
    unsigned int batch_size,
    const float* __restrict__ residual
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 5u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 34u;
            const unsigned short d_raw = (unsigned short)bptr[32u] | ((unsigned short)bptr[33u] << 8u);
            const float scale = fp8_pf_fast_fp16_to_float(d_raw);
            const unsigned int base = b << 5u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float bsum = 0.0f;
                #pragma unroll 8
                for (unsigned int w = 0u; w < 32u; ++w) {
                    bsum += fp8_pf_e4m3_to_float(bptr[w]) * xbase[w];
                }
                col_sums[col] += scale * bsum;
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            float s = col_sums[col];
            s += __shfl_down_sync(0xffffffffu, s, 16u);
            s += __shfl_down_sync(0xffffffffu, s,  8u);
            s += __shfl_down_sync(0xffffffffu, s,  4u);
            s += __shfl_down_sync(0xffffffffu, s,  2u);
            s += __shfl_down_sync(0xffffffffu, s,  1u);
            if (lane == 0u) {
                const unsigned long long idx = (unsigned long long)(col_base + col) * n_rows + row;
                outputs[idx] = residual[idx] + s;
            }
        }
    }
}

/* =========================================================================
   Kernel 3 — fused_gate_up_swiglu_gemm_fp8_e4m3
   Batch fused gate+up FP8 E4M3 GEMM with SwiGLU epilogue.

   The concatenated gate+up weight matrix has 2*n_ffn_rows rows total:
     gate rows:  0   .. n_ffn_rows-1
     up   rows:  n_ffn_rows .. 2*n_ffn_rows-1
   blocks pointer covers all 2*n_ffn_rows rows in AoS layout.

   For each (row r, col c):
     outputs[c * n_ffn_rows + r] = SiLU(gate_sum(r,c)) * up_sum(r,c)

   Output buffer must be zeroed before calling (kernel writes, not +=).
   ========================================================================= */
extern "C" __global__ void fused_gate_up_swiglu_gemm_fp8_e4m3(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ inputs,
    float*               __restrict__ outputs,
    unsigned int n_ffn_rows,
    unsigned int k,
    unsigned int batch_size
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_ffn_rows) return;

    const unsigned int blocks_per_row = k >> 5u;
    const unsigned int up_row_offset  = n_ffn_rows * blocks_per_row;  /* block index offset for up row r */

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8];
        float up_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) { gate_sums[c] = 0.0f; up_sums[c] = 0.0f; }

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            /* Gate block (row r) — weights-first layout */
            const unsigned char* gbptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 34u;
            const unsigned short gd_raw = (unsigned short)gbptr[32u] | ((unsigned short)gbptr[33u] << 8u);
            const float gscale = fp8_pf_fast_fp16_to_float(gd_raw);

            /* Up block (row r + n_ffn_rows) — weights-first layout */
            const unsigned char* ubptr = blocks + (unsigned long long)(up_row_offset + row * blocks_per_row + b) * 34u;
            const unsigned short ud_raw = (unsigned short)ubptr[32u] | ((unsigned short)ubptr[33u] << 8u);
            const float uscale = fp8_pf_fast_fp16_to_float(ud_raw);

            const unsigned int base = b << 5u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float gsum = 0.0f;
                float usum = 0.0f;
                #pragma unroll 8
                for (unsigned int w = 0u; w < 32u; ++w) {
                    const float x = xbase[w];
                    gsum += fp8_pf_e4m3_to_float(gbptr[w]) * x;
                    usum += fp8_pf_e4m3_to_float(ubptr[w]) * x;
                }
                gate_sums[col] += gscale * gsum;
                up_sums[col]   += uscale * usum;
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            float gs = gate_sums[col];
            float us = up_sums[col];
            gs += __shfl_down_sync(0xffffffffu, gs, 16u);
            gs += __shfl_down_sync(0xffffffffu, gs,  8u);
            gs += __shfl_down_sync(0xffffffffu, gs,  4u);
            gs += __shfl_down_sync(0xffffffffu, gs,  2u);
            gs += __shfl_down_sync(0xffffffffu, gs,  1u);
            us += __shfl_down_sync(0xffffffffu, us, 16u);
            us += __shfl_down_sync(0xffffffffu, us,  8u);
            us += __shfl_down_sync(0xffffffffu, us,  4u);
            us += __shfl_down_sync(0xffffffffu, us,  2u);
            us += __shfl_down_sync(0xffffffffu, us,  1u);
            if (lane == 0u) {
                outputs[(unsigned long long)(col_base + col) * n_ffn_rows + row] = fp8_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Kernel 4 — gemv_fp8_e4m3_pf
   Single-token FP8 E4M3 GEMV (for attention inner loop / sequential pass).
   output[row] = sum over k of weight_row * input
   ========================================================================= */
extern "C" __global__ void gemv_fp8_e4m3_pf(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 5u;

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 34u;
        /* Scale at bytes 32-33 */
        const unsigned short d_raw = (unsigned short)bptr[32u] | ((unsigned short)bptr[33u] << 8u);
        const float scale = fp8_pf_fast_fp16_to_float(d_raw);
        const float* xbase = input + (b << 5u);
        float bsum = 0.0f;
        #pragma unroll 8
        for (unsigned int w = 0u; w < 32u; ++w) {
            bsum += fp8_pf_e4m3_to_float(bptr[w]) * xbase[w];
        }
        acc += scale * bsum;
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}

/* =========================================================================
   Kernel 5 — gemm_fp8_e5m2
   Batch FP8 E5M2 GEMM. Accumulates into outputs with +=.
   blocks: AoS, 34 bytes/block (32 FP8 E5M2 weights + 2 scale bytes)
   ========================================================================= */
extern "C" __global__ void gemm_fp8_e5m2(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ inputs,
    float*               __restrict__ outputs,
    unsigned int n_rows,
    unsigned int k,
    unsigned int batch_size
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 5u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 34u;
            const unsigned short d_raw = (unsigned short)bptr[32u] | ((unsigned short)bptr[33u] << 8u);
            const float scale = fp8_pf_fast_fp16_to_float(d_raw);
            const unsigned int base = b << 5u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float bsum = 0.0f;
                #pragma unroll 8
                for (unsigned int w = 0u; w < 32u; ++w) {
                    bsum += fp8_pf_e5m2_to_float(bptr[w]) * xbase[w];
                }
                col_sums[col] += scale * bsum;
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            float s = col_sums[col];
            s += __shfl_down_sync(0xffffffffu, s, 16u);
            s += __shfl_down_sync(0xffffffffu, s,  8u);
            s += __shfl_down_sync(0xffffffffu, s,  4u);
            s += __shfl_down_sync(0xffffffffu, s,  2u);
            s += __shfl_down_sync(0xffffffffu, s,  1u);
            if (lane == 0u)
                outputs[(unsigned long long)(col_base + col) * n_rows + row] += s;
        }
    }
}

/* =========================================================================
   Kernel 6 — gemm_fp8_e5m2_residual
   Batch FP8 E5M2 GEMM + fused in-place residual add.
   ========================================================================= */
extern "C" __global__ void gemm_fp8_e5m2_residual(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ inputs,
    float*               __restrict__ outputs,
    unsigned int n_rows,
    unsigned int k,
    unsigned int batch_size,
    const float* __restrict__ residual
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 5u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 34u;
            const unsigned short d_raw = (unsigned short)bptr[32u] | ((unsigned short)bptr[33u] << 8u);
            const float scale = fp8_pf_fast_fp16_to_float(d_raw);
            const unsigned int base = b << 5u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float bsum = 0.0f;
                #pragma unroll 8
                for (unsigned int w = 0u; w < 32u; ++w) {
                    bsum += fp8_pf_e5m2_to_float(bptr[w]) * xbase[w];
                }
                col_sums[col] += scale * bsum;
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            float s = col_sums[col];
            s += __shfl_down_sync(0xffffffffu, s, 16u);
            s += __shfl_down_sync(0xffffffffu, s,  8u);
            s += __shfl_down_sync(0xffffffffu, s,  4u);
            s += __shfl_down_sync(0xffffffffu, s,  2u);
            s += __shfl_down_sync(0xffffffffu, s,  1u);
            if (lane == 0u) {
                const unsigned long long idx = (unsigned long long)(col_base + col) * n_rows + row;
                outputs[idx] = residual[idx] + s;
            }
        }
    }
}

/* =========================================================================
   Kernel 7 — fused_gate_up_swiglu_gemm_fp8_e5m2
   Batch fused gate+up FP8 E5M2 GEMM with SwiGLU epilogue.

   Concatenated gate+up weight matrix: 2*n_ffn_rows rows total.
     gate rows 0..n_ffn_rows-1, up rows n_ffn_rows..2*n_ffn_rows-1.
   blocks pointer covers all 2*n_ffn_rows rows in FP8 E5M2 AoS layout.

   For each (row r, col c):
     outputs[c * n_ffn_rows + r] = SiLU(gate_sum(r,c)) * up_sum(r,c)
   ========================================================================= */
extern "C" __global__ void fused_gate_up_swiglu_gemm_fp8_e5m2(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ inputs,
    float*               __restrict__ outputs,
    unsigned int n_ffn_rows,
    unsigned int k,
    unsigned int batch_size
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_ffn_rows) return;

    const unsigned int blocks_per_row = k >> 5u;
    const unsigned int up_row_offset  = n_ffn_rows * blocks_per_row;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8];
        float up_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) { gate_sums[c] = 0.0f; up_sums[c] = 0.0f; }

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            /* Gate block — weights-first layout */
            const unsigned char* gbptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 34u;
            const unsigned short gd_raw = (unsigned short)gbptr[32u] | ((unsigned short)gbptr[33u] << 8u);
            const float gscale = fp8_pf_fast_fp16_to_float(gd_raw);

            /* Up block — weights-first layout */
            const unsigned char* ubptr = blocks + (unsigned long long)(up_row_offset + row * blocks_per_row + b) * 34u;
            const unsigned short ud_raw = (unsigned short)ubptr[32u] | ((unsigned short)ubptr[33u] << 8u);
            const float uscale = fp8_pf_fast_fp16_to_float(ud_raw);

            const unsigned int base = b << 5u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float gsum = 0.0f;
                float usum = 0.0f;
                #pragma unroll 8
                for (unsigned int w = 0u; w < 32u; ++w) {
                    const float x = xbase[w];
                    gsum += fp8_pf_e5m2_to_float(gbptr[w]) * x;
                    usum += fp8_pf_e5m2_to_float(ubptr[w]) * x;
                }
                gate_sums[col] += gscale * gsum;
                up_sums[col]   += uscale * usum;
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            float gs = gate_sums[col];
            float us = up_sums[col];
            gs += __shfl_down_sync(0xffffffffu, gs, 16u);
            gs += __shfl_down_sync(0xffffffffu, gs,  8u);
            gs += __shfl_down_sync(0xffffffffu, gs,  4u);
            gs += __shfl_down_sync(0xffffffffu, gs,  2u);
            gs += __shfl_down_sync(0xffffffffu, gs,  1u);
            us += __shfl_down_sync(0xffffffffu, us, 16u);
            us += __shfl_down_sync(0xffffffffu, us,  8u);
            us += __shfl_down_sync(0xffffffffu, us,  4u);
            us += __shfl_down_sync(0xffffffffu, us,  2u);
            us += __shfl_down_sync(0xffffffffu, us,  1u);
            if (lane == 0u) {
                outputs[(unsigned long long)(col_base + col) * n_ffn_rows + row] = fp8_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Kernel 8 — gemv_fp8_e5m2_pf
   Single-token FP8 E5M2 GEMV (for attention inner loop / sequential pass).
   ========================================================================= */
extern "C" __global__ void gemv_fp8_e5m2_pf(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 5u;

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 34u;
        /* Scale at bytes 32-33 */
        const unsigned short d_raw = (unsigned short)bptr[32u] | ((unsigned short)bptr[33u] << 8u);
        const float scale = fp8_pf_fast_fp16_to_float(d_raw);
        const float* xbase = input + (b << 5u);
        float bsum = 0.0f;
        #pragma unroll 8
        for (unsigned int w = 0u; w < 32u; ++w) {
            bsum += fp8_pf_e5m2_to_float(bptr[w]) * xbase[w];
        }
        acc += scale * bsum;
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}
"#;
