//! CUDA C kernel source strings for OxiBonsai Q4_0 and Q8_0 batch GEMM (prefill) operations.
//!
//! # Prefill kernel catalogue
//!
//! | Kernel                            | Description                                              |
//! |-----------------------------------|----------------------------------------------------------|
//! | `gemm_q4_0`                       | Batch GEMM: Q4_0 AoS, col-major I/O, col_sums[8]       |
//! | `gemm_q4_0_residual`              | GEMM + fused residual add                               |
//! | `fused_gate_up_swiglu_gemm_q4_0`  | Fused gate+up Q4_0 GEMM with SwiGLU epilogue            |
//! | `gemv_q4_0_pf`                    | Single-token Q4_0 GEMV (for sequential attention pass)  |
//! | `gemm_q8_0`                       | Batch GEMM: Q8_0 AoS, col-major I/O, col_sums[8]       |
//! | `gemm_q8_0_residual`              | Q8_0 GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q8_0`  | Fused gate+up Q8_0 GEMM with SwiGLU epilogue            |
//! | `gemv_q8_0_pf`                    | Single-token Q8_0 GEMV (for sequential attention pass)  |
//!
//! # Block layout
//!
//! **Q4_0** (18 bytes/block, 32 weights):
//! ```text
//! bytes 0-1:   FP16 LE scale (d)
//! bytes 2-17:  16 nibble bytes → 32 int4 weights
//! Dequant: w[j] = d * (nibble[j] - 8)
//!   even j → qs[j/2] & 0x0F, odd j → (qs[j/2] >> 4) & 0x0F
//! ```
//!
//! **Q8_0** (34 bytes/block, 32 weights):
//! ```text
//! bytes 0-1:   FP16 LE scale (d)
//! bytes 2-33:  32 signed int8 weights
//! Dequant: w[j] = d * qs[j]
//! ```
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

/// CUDA C source for Q4_0 and Q8_0 batch GEMM (prefill) kernels.
///
/// All kernels use AoS weight layout (blocks stored contiguously as-is from GGUF).
/// Batch tensors use column-major layout: `buf[col * dim + element]`.
pub const CUDA_Q_STD_PREFILL_KERNELS_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA Q4_0 / Q8_0 prefill (batch GEMM) kernels.

   Q4_0 AoS block (18 bytes): [d_lo, d_hi, qs[0]..qs[15]]
     scale = FP16 LE, 16 nibble bytes → 32 int4 weights
     w[j] = scale * (nibble[j] - 8)

   Q8_0 AoS block (34 bytes): [d_lo, d_hi, qs[0]..qs[31]]
     scale = FP16 LE, 32 signed int8 weights
     w[j] = scale * qs[j]

   Batch tensors: column-major  buf[col * dim + element]
   Grid:  (ceil(n_rows/8), 1, 1)  — 8 warps per CTA
   Block: (256, 1, 1)             — 8 warps × 32 lanes
   ========================================================================= */

/* ── Hardware FP16 → FP32 via PTX (SM 6.0+, 1 instruction) ─────────────── */
static __device__ __forceinline__ float q_pf_fast_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* ── SiLU activation: x · σ(x) ─────────────────────────────────────────── */
static __device__ __forceinline__ float q_pf_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* =========================================================================
   Kernel 1 — gemm_q4_0
   Batch Q4_0 GEMM. Accumulates into outputs with +=.
   blocks: AoS, 18 bytes/block (2 scale + 16 nibble bytes = 32 weights/block)
   inputs:  col-major [batch_size * k]
   outputs: col-major [batch_size * n_rows], accumulated with +=
   k must be a positive multiple of 32.
   ========================================================================= */
extern "C" __global__ void gemm_q4_0(
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
            /* Load AoS block for this row */
            const unsigned char* bptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 18u;
            const unsigned short d_raw = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
            const float scale = q_pf_fast_fp16_to_float(d_raw);
            const unsigned int base = b << 5u;  /* b * 32 */

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float bsum = 0.0f;
                #pragma unroll 16
                for (unsigned int nb = 0u; nb < 16u; ++nb) {
                    const unsigned int byte = bptr[2u + nb];
                    const float w0 = (float)((int)(byte & 0x0Fu) - 8);
                    const float w1 = (float)((int)((byte >> 4u) & 0x0Fu) - 8);
                    bsum += w0 * xbase[nb * 2u] + w1 * xbase[nb * 2u + 1u];
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
   Kernel 2 — gemm_q4_0_residual
   Batch Q4_0 GEMM + fused in-place residual add.
   For each (row, col): outputs[col*n_rows+row] = residual[col*n_rows+row] + sum
   ========================================================================= */
extern "C" __global__ void gemm_q4_0_residual(
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
            const unsigned char* bptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 18u;
            const unsigned short d_raw = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
            const float scale = q_pf_fast_fp16_to_float(d_raw);
            const unsigned int base = b << 5u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float bsum = 0.0f;
                #pragma unroll 16
                for (unsigned int nb = 0u; nb < 16u; ++nb) {
                    const unsigned int byte = bptr[2u + nb];
                    const float w0 = (float)((int)(byte & 0x0Fu) - 8);
                    const float w1 = (float)((int)((byte >> 4u) & 0x0Fu) - 8);
                    bsum += w0 * xbase[nb * 2u] + w1 * xbase[nb * 2u + 1u];
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
   Kernel 3 — fused_gate_up_swiglu_gemm_q4_0
   Batch fused gate+up Q4_0 GEMM with SwiGLU epilogue.

   The concatenated gate+up weight matrix has 2*n_ffn_rows rows total:
     gate rows:  0   .. n_ffn_rows-1
     up   rows:  n_ffn_rows .. 2*n_ffn_rows-1
   blocks pointer covers all 2*n_ffn_rows rows in AoS layout.

   For each (row r, col c):
     outputs[c * n_ffn_rows + r] = SiLU(gate_sum(r,c)) * up_sum(r,c)

   Output buffer must be zeroed before calling (kernel writes, not +=).
   ========================================================================= */
extern "C" __global__ void fused_gate_up_swiglu_gemm_q4_0(
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
            /* Gate block (row r) */
            const unsigned char* gbptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 18u;
            const unsigned short gd_raw = (unsigned short)gbptr[0] | ((unsigned short)gbptr[1] << 8u);
            const float gscale = q_pf_fast_fp16_to_float(gd_raw);

            /* Up block (row r + n_ffn_rows) */
            const unsigned char* ubptr = blocks + (unsigned long long)((up_row_offset + row * blocks_per_row + b)) * 18u;
            const unsigned short ud_raw = (unsigned short)ubptr[0] | ((unsigned short)ubptr[1] << 8u);
            const float uscale = q_pf_fast_fp16_to_float(ud_raw);

            const unsigned int base = b << 5u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float gsum = 0.0f;
                float usum = 0.0f;
                #pragma unroll 16
                for (unsigned int nb = 0u; nb < 16u; ++nb) {
                    const float x0 = xbase[nb * 2u];
                    const float x1 = xbase[nb * 2u + 1u];
                    const float gw0 = (float)((int)(gbptr[2u + nb] & 0x0Fu) - 8);
                    const float gw1 = (float)((int)((gbptr[2u + nb] >> 4u) & 0x0Fu) - 8);
                    const float uw0 = (float)((int)(ubptr[2u + nb] & 0x0Fu) - 8);
                    const float uw1 = (float)((int)((ubptr[2u + nb] >> 4u) & 0x0Fu) - 8);
                    gsum += gw0 * x0 + gw1 * x1;
                    usum += uw0 * x0 + uw1 * x1;
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
                outputs[(unsigned long long)(col_base + col) * n_ffn_rows + row] = q_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Kernel 4 — gemv_q4_0_pf
   Single-token Q4_0 GEMV (same as gemv_q4_0, here for intra-PTX reuse).
   output[row] = sum over k of weight_row * input
   ========================================================================= */
extern "C" __global__ void gemv_q4_0_pf(
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
        const unsigned char* bptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 18u;
        const unsigned short d_raw = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
        const float scale = q_pf_fast_fp16_to_float(d_raw);
        const float* xbase = input + (b << 5u);
        float bsum = 0.0f;
        #pragma unroll 16
        for (unsigned int nb = 0u; nb < 16u; ++nb) {
            const unsigned int byte = bptr[2u + nb];
            const float w0 = (float)((int)(byte & 0x0Fu) - 8);
            const float w1 = (float)((int)((byte >> 4u) & 0x0Fu) - 8);
            bsum += w0 * xbase[nb * 2u] + w1 * xbase[nb * 2u + 1u];
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
   Kernel 5 — gemm_q8_0
   Batch Q8_0 GEMM. Accumulates into outputs with +=.
   blocks: AoS, 34 bytes/block (2 scale + 32 signed int8 weights)
   ========================================================================= */
extern "C" __global__ void gemm_q8_0(
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
            const unsigned short d_raw = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
            const float scale = q_pf_fast_fp16_to_float(d_raw);
            const unsigned int base = b << 5u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float bsum = 0.0f;
                #pragma unroll 32
                for (unsigned int j = 0u; j < 32u; ++j) {
                    const int q = (int)(signed char)bptr[2u + j];
                    bsum += (float)q * xbase[j];
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
   Kernel 6 — gemm_q8_0_residual
   Batch Q8_0 GEMM + fused in-place residual add.
   ========================================================================= */
extern "C" __global__ void gemm_q8_0_residual(
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
            const unsigned short d_raw = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
            const float scale = q_pf_fast_fp16_to_float(d_raw);
            const unsigned int base = b << 5u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float bsum = 0.0f;
                #pragma unroll 32
                for (unsigned int j = 0u; j < 32u; ++j) {
                    const int q = (int)(signed char)bptr[2u + j];
                    bsum += (float)q * xbase[j];
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
   Kernel 7 — fused_gate_up_swiglu_gemm_q8_0
   Batch fused gate+up Q8_0 GEMM with SwiGLU epilogue.

   Concatenated gate+up weight matrix: 2*n_ffn_rows rows total.
     gate rows 0..n_ffn_rows-1, up rows n_ffn_rows..2*n_ffn_rows-1.
   blocks pointer covers all 2*n_ffn_rows rows in Q8_0 AoS layout.

   For each (row r, col c):
     outputs[c * n_ffn_rows + r] = SiLU(gate_sum(r,c)) * up_sum(r,c)
   ========================================================================= */
extern "C" __global__ void fused_gate_up_swiglu_gemm_q8_0(
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
            /* Gate block */
            const unsigned char* gbptr = blocks + (unsigned long long)(row * blocks_per_row + b) * 34u;
            const unsigned short gd_raw = (unsigned short)gbptr[0] | ((unsigned short)gbptr[1] << 8u);
            const float gscale = q_pf_fast_fp16_to_float(gd_raw);

            /* Up block */
            const unsigned char* ubptr = blocks + (unsigned long long)((up_row_offset + row * blocks_per_row + b)) * 34u;
            const unsigned short ud_raw = (unsigned short)ubptr[0] | ((unsigned short)ubptr[1] << 8u);
            const float uscale = q_pf_fast_fp16_to_float(ud_raw);

            const unsigned int base = b << 5u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                const float* xbase = inp + base;
                float gsum = 0.0f;
                float usum = 0.0f;
                #pragma unroll 32
                for (unsigned int j = 0u; j < 32u; ++j) {
                    const float x = xbase[j];
                    const int gq = (int)(signed char)gbptr[2u + j];
                    const int uq = (int)(signed char)ubptr[2u + j];
                    gsum += (float)gq * x;
                    usum += (float)uq * x;
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
                outputs[(unsigned long long)(col_base + col) * n_ffn_rows + row] = q_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Kernel 8 — gemv_q8_0_pf
   Single-token Q8_0 GEMV (same as gemv_q8_0, here for intra-PTX reuse).
   ========================================================================= */
extern "C" __global__ void gemv_q8_0_pf(
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
        const unsigned short d_raw = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
        const float scale = q_pf_fast_fp16_to_float(d_raw);
        const float* xbase = input + (b << 5u);
        float bsum = 0.0f;
        #pragma unroll 32
        for (unsigned int j = 0u; j < 32u; ++j) {
            const int q = (int)(signed char)bptr[2u + j];
            bsum += (float)q * xbase[j];
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
