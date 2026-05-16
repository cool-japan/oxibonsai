//! CUDA C kernel source strings for OxiBonsai batch/prefill operations.
//!
//! # Prefill kernel catalogue
//!
//! | Kernel                           | Description |
//! |----------------------------------|-------------|
//! | `gemm_q1_g128_v7`                | Batch GEMM: 1 warp per weight row, all batch cols |
//! | `gemm_q1_g128_v7_residual`       | GEMM + fused in-place residual add |
//! | `fused_gate_up_swiglu_gemm_q1`   | Fused gate+up Q1 GEMM with SwiGLU epilogue |
//! | `batched_swiglu`                 | Element-wise SiLU(gate)*up for batch_size vectors |
//! | `batched_rmsnorm_v2`             | Per-token RMSNorm for batch_size tokens |
//!
//! # Layout convention
//!
//! All batch tensors use **column-major** layout: `buf[col * dim + element]`
//! where `col` is the batch/token index and `dim` is the vector dimension.
//! This matches the Metal MSL prefill kernels in `kernel_sources/prefill.rs`.
//!
//! # Weight layout (SoA)
//!
//! Q1_0_G128 weights are stored in Structure-of-Arrays layout:
//! `[scales: n_rows×n_blocks × 2 bytes FP16][data: n_rows×n_blocks × 16 bytes]`

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

/// CUDA C source for batch GEMM and prefill-specific kernels.
///
/// Compiled once at process startup via `cudarc::nvrtc::compile_ptx`.
pub const CUDA_PREFILL_KERNELS_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA prefill kernels — batch GEMM and batched activations.
   Ported from MSL (kernel_sources/prefill.rs) to CUDA C.

   Weight layout: SoA — [scales: n_rows*n_blocks × 2B FP16]
                         [data:   n_rows*n_blocks × 16B  ]

   Batch tensor layout: column-major — buf[col * dim + element]
   ========================================================================= */

/* ── Hardware FP16 → FP32 via PTX (1 instruction on SM6.0+) ────────────── */
static __device__ __forceinline__ float fast_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* ── SiLU activation: x · σ(x) ─────────────────────────────────────────── */
static __device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* =========================================================================
   Kernel 1 — gemm_q1_g128_v7
   Batch Q1_0_G128 GEMM.

   Each warp processes 1 weight row × all batch_size batch columns
   (in 8-column outer chunks to handle any batch_size).
   Weights are loaded once per block iteration (L1 cache retains them
   across columns), following the Metal MSL gemm_q1_g128_v7 pattern.

   Input/output use column-major layout:
     inputs[col * k + element]
     outputs[col * n_rows + row]     <- accumulated with +=

   Weight layout (SoA):
     [scales: n_rows*n_blocks × 2B FP16][data: n_rows*n_blocks × 16B]

   Grid:  (ceil(n_rows/8), 1, 1)   — 8 warps per CTA
   Block: (256, 1, 1)              — 8 warps × 32 lanes
   ========================================================================= */
extern "C" __global__ void gemm_q1_g128_v7(
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

    const unsigned int n_blocks = k >> 7u;   /* k / 128 */

    /* SoA weight pointers */
    const unsigned short* __restrict__ scales =
        (const unsigned short* __restrict__)blocks;
    const unsigned int* __restrict__ data =
        (const unsigned int* __restrict__)(blocks + (unsigned long long)n_rows * n_blocks * 2u);

    /* Process batch columns in 8-column outer chunks so any batch_size is handled correctly.
       This mirrors the MSL gemm_q1_g128_v7 fix (prefill.rs lines 59-86). */
    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < n_blocks; b += 32u) {
            const unsigned int  g     = row * n_blocks + b;
            const float         scale = fast_fp16_to_float(scales[g]);
            const unsigned int* bp    = data + (unsigned long long)g * 4u;
            const unsigned int  w0 = bp[0u], w1 = bp[1u], w2 = bp[2u], w3 = bp[3u];
            const unsigned int  base  = b * 128u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                float block_sum = 0.0f;

                #pragma unroll
                for (unsigned int bit = 0u; bit < 32u; ++bit) {
                    block_sum +=
                        (((w0 >> bit) & 1u) ? 1.0f : -1.0f) * inp[base        + bit]
                      + (((w1 >> bit) & 1u) ? 1.0f : -1.0f) * inp[base + 32u  + bit]
                      + (((w2 >> bit) & 1u) ? 1.0f : -1.0f) * inp[base + 64u  + bit]
                      + (((w3 >> bit) & 1u) ? 1.0f : -1.0f) * inp[base + 96u  + bit];
                }
                col_sums[col] += scale * block_sum;
            }
        }

        /* Warp-shuffle reduction and write outputs (column-major) */
        for (unsigned int col = 0u; col < cols; ++col) {
            float s = col_sums[col];
            s += __shfl_down_sync(0xffffffffu, s, 16u);
            s += __shfl_down_sync(0xffffffffu, s,  8u);
            s += __shfl_down_sync(0xffffffffu, s,  4u);
            s += __shfl_down_sync(0xffffffffu, s,  2u);
            s += __shfl_down_sync(0xffffffffu, s,  1u);

            if (lane == 0u) outputs[(unsigned long long)(col_base + col) * n_rows + row] += s;
        }
    }
}

/* =========================================================================
   Kernel 2 — gemm_q1_g128_v7_residual
   Batch GEMM + fused in-place residual add.

   For each (row, col): outputs[col*n_rows + row] = residual[col*n_rows + row] + sum

   Same grid/block config as gemm_q1_g128_v7.
   ========================================================================= */
extern "C" __global__ void gemm_q1_g128_v7_residual(
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

    const unsigned int n_blocks = k >> 7u;

    const unsigned short* __restrict__ scales =
        (const unsigned short* __restrict__)blocks;
    const unsigned int* __restrict__ data =
        (const unsigned int* __restrict__)(blocks + (unsigned long long)n_rows * n_blocks * 2u);

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < n_blocks; b += 32u) {
            const unsigned int  g     = row * n_blocks + b;
            const float         scale = fast_fp16_to_float(scales[g]);
            const unsigned int* bp    = data + (unsigned long long)g * 4u;
            const unsigned int  w0 = bp[0u], w1 = bp[1u], w2 = bp[2u], w3 = bp[3u];
            const unsigned int  base  = b * 128u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                float block_sum = 0.0f;

                #pragma unroll
                for (unsigned int bit = 0u; bit < 32u; ++bit) {
                    block_sum +=
                        (((w0 >> bit) & 1u) ? 1.0f : -1.0f) * inp[base        + bit]
                      + (((w1 >> bit) & 1u) ? 1.0f : -1.0f) * inp[base + 32u  + bit]
                      + (((w2 >> bit) & 1u) ? 1.0f : -1.0f) * inp[base + 64u  + bit]
                      + (((w3 >> bit) & 1u) ? 1.0f : -1.0f) * inp[base + 96u  + bit];
                }
                col_sums[col] += scale * block_sum;
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
   Kernel 3 — fused_gate_up_swiglu_gemm_q1
   Batch fused gate+up Q1 GEMM with SwiGLU epilogue for prefill.

   Reads gate row `r` and up row `r + n_rows` from the concatenated SoA
   weight matrix simultaneously, for all batch columns, and writes:
     outputs[col * n_rows + row] = SiLU(gate_sum) * up_sum

   Weight layout (concatenated gate+up, SoA):
     [scales: 2*n_rows*n_blocks × 2B][data: 2*n_rows*n_blocks × 16B]
     gate rows:  0   .. n_rows-1
     up rows:    n_rows .. 2*n_rows-1

   Grid:  (ceil(n_rows/8), 1, 1)
   Block: (256, 1, 1)
   ========================================================================= */
extern "C" __global__ void fused_gate_up_swiglu_gemm_q1(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ inputs,
    float*               __restrict__ outputs,
    unsigned int n_rows,
    unsigned int k,
    unsigned int batch_size
) {
    const unsigned int warp_id   = threadIdx.x >> 5u;
    const unsigned int lane      = threadIdx.x & 31u;
    const unsigned int row       = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int total_rows = 2u * n_rows;
    const unsigned int n_blocks   = k >> 7u;

    const unsigned short* __restrict__ scales =
        (const unsigned short* __restrict__)blocks;
    const unsigned int* __restrict__ data =
        (const unsigned int* __restrict__)(blocks + (unsigned long long)total_rows * n_blocks * 2u);

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8];
        float up_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) { gate_sums[c] = 0.0f; up_sums[c] = 0.0f; }

        for (unsigned int b = lane; b < n_blocks; b += 32u) {
            /* ── Gate block (row r) ─────────────────────────────────────────── */
            const unsigned int  g_gate  = row * n_blocks + b;
            const float         sg      = fast_fp16_to_float(scales[g_gate]);
            const unsigned int* bp_g    = data + (unsigned long long)g_gate * 4u;
            const unsigned int  wg0 = bp_g[0u], wg1 = bp_g[1u], wg2 = bp_g[2u], wg3 = bp_g[3u];

            /* ── Up block (row r + n_rows) ──────────────────────────────────── */
            const unsigned int  g_up    = (row + n_rows) * n_blocks + b;
            const float         su      = fast_fp16_to_float(scales[g_up]);
            const unsigned int* bp_u    = data + (unsigned long long)g_up * 4u;
            const unsigned int  wu0 = bp_u[0u], wu1 = bp_u[1u], wu2 = bp_u[2u], wu3 = bp_u[3u];

            const unsigned int base = b * 128u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                float gate_sum = 0.0f;
                float up_sum   = 0.0f;

                #pragma unroll
                for (unsigned int bit = 0u; bit < 32u; ++bit) {
                    const float i0 = inp[base        + bit];
                    const float i1 = inp[base + 32u  + bit];
                    const float i2 = inp[base + 64u  + bit];
                    const float i3 = inp[base + 96u  + bit];
                    gate_sum += (((wg0 >> bit) & 1u) ? 1.0f : -1.0f) * i0
                              + (((wg1 >> bit) & 1u) ? 1.0f : -1.0f) * i1
                              + (((wg2 >> bit) & 1u) ? 1.0f : -1.0f) * i2
                              + (((wg3 >> bit) & 1u) ? 1.0f : -1.0f) * i3;
                    up_sum   += (((wu0 >> bit) & 1u) ? 1.0f : -1.0f) * i0
                              + (((wu1 >> bit) & 1u) ? 1.0f : -1.0f) * i1
                              + (((wu2 >> bit) & 1u) ? 1.0f : -1.0f) * i2
                              + (((wu3 >> bit) & 1u) ? 1.0f : -1.0f) * i3;
                }
                gate_sums[col] += sg * gate_sum;
                up_sums[col]   += su * up_sum;
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
                outputs[(unsigned long long)(col_base + col) * n_rows + row] = silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Kernel 4 — batched_swiglu
   Element-wise SiLU(gate)*up for batch_size vectors at once.

   Input layout:  gate_up[gid]              = gate element
                  gate_up[gid + n*bs_f]     = up element
   where n*bs_f = n * batch_size, and gid = blockIdx.x*256 + threadIdx.x
   covers [0 .. n*batch_size).

   Output: output[gid] = SiLU(gate_up[gid]) * gate_up[gid + n*batch_size]

   Grid:  (ceil(n*batch_size/256), 1, 1)
   Block: (256, 1, 1)
   ========================================================================= */
extern "C" __global__ void batched_swiglu(
    const float* __restrict__ gate_up,
    float*       __restrict__ output,
    unsigned int n,
    unsigned int batch_size
) {
    const unsigned long long total   = (unsigned long long)n * batch_size;
    const unsigned long long gid     = (unsigned long long)blockIdx.x * 256u + threadIdx.x;
    if (gid >= total) return;

    const float gate_elem = gate_up[gid];
    const float up_elem   = gate_up[gid + total];
    output[gid] = silu(gate_elem) * up_elem;
}

/* =========================================================================
   Kernel 5 — batched_rmsnorm_v2
   Per-token RMSNorm for batch_size tokens simultaneously.

   Each block is responsible for one token (batch item).  All 256 threads
   in the block cooperate to:
     1. Accumulate partial sum-of-squares for this token's vector.
     2. Perform warp-shuffle reduction to get the full SS.
     3. Compute rms_inv = 1 / sqrt(SS / n + eps).
     4. Write output[b*n + i] = input[b*n + i] * rms_inv * weight[i].

   Input/output use row-major layout here: input[b * n + element].
   (This is the standard AoS layout for activations uploaded from host.)

   Grid:  (batch_size, 1, 1)   — one block per token
   Block: (256, 1, 1)
   ========================================================================= */
extern "C" __global__ void batched_rmsnorm_v2(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    unsigned int n,
    unsigned int batch_size,
    float eps
) {
    __shared__ float warp_ss[8];

    const unsigned int b    = blockIdx.x;   /* token index */
    if (b >= batch_size) return;

    const unsigned int tid   = threadIdx.x;
    const unsigned int lane  = tid & 31u;
    const unsigned int warp  = tid >> 5u;

    const float* in_row  = input  + (unsigned long long)b * n;
    float*       out_row = output + (unsigned long long)b * n;

    /* Phase 1: partial sum-of-squares */
    float pss = 0.0f;
    for (unsigned int i = tid; i < n; i += 256u) {
        float x = in_row[i];
        pss += x * x;
    }

    /* Phase 2: warp-level reduction */
    pss += __shfl_down_sync(0xffffffffu, pss, 16u);
    pss += __shfl_down_sync(0xffffffffu, pss,  8u);
    pss += __shfl_down_sync(0xffffffffu, pss,  4u);
    pss += __shfl_down_sync(0xffffffffu, pss,  2u);
    pss += __shfl_down_sync(0xffffffffu, pss,  1u);

    if (lane == 0u) warp_ss[warp] = pss;
    __syncthreads();

    /* Phase 3: thread 0 sums warp partials, broadcasts rms_inv */
    if (tid == 0u) {
        float total = 0.0f;
        for (unsigned int w = 0u; w < 8u; ++w) total += warp_ss[w];
        warp_ss[0] = 1.0f / sqrtf(total / (float)n + eps);
    }
    __syncthreads();

    const float rms_inv = warp_ss[0];

    /* Phase 4: normalise and weight */
    for (unsigned int i = tid; i < n; i += 256u) {
        out_row[i] = in_row[i] * rms_inv * weight[i];
    }
}

/* =========================================================================
   TQ2_0_G128 weight decode helpers
   ========================================================================= */

/* ── TQ2 weight decode helper ─────────────────────────────────────────── */
static __device__ __forceinline__ float pf_decode_tq2(unsigned int code) {
    return (code == 2u) ? 1.0f : ((code == 0u) ? -1.0f : 0.0f);
}

/* Dot product of 4 TQ2 weights (one byte) against 4 input values. */
static __device__ __forceinline__ float pf_byte_dot_tq2(unsigned int bval, const float* x) {
    return pf_decode_tq2((bval      ) & 3u) * x[0]
         + pf_decode_tq2((bval >> 2u) & 3u) * x[1]
         + pf_decode_tq2((bval >> 4u) & 3u) * x[2]
         + pf_decode_tq2((bval >> 6u) & 3u) * x[3];
}

/* =========================================================================
   Kernel 6 — gemm_tq2_g128_v7
   Batch TQ2_0_G128 GEMM (accumulates into output with +=).

   Weight layout (SoA):
     [scales: n_rows*blocks_per_row × 2B FP16 at offset 0]
     [qs:     n_rows*blocks_per_row × 32B at offset = total_blocks * 2]
   blocks_per_row = k / 128
   Each qs block: 8 × u32 (32 bytes = 128 ternary 2-bit weights)
   TQ2 decode: 0b00 → -1.0f, 0b01 → 0.0f, 0b10 → +1.0f, 0b11 → 0.0f

   Grid:  (ceil(n_rows/8), 1, 1)   — 8 warps per CTA
   Block: (256, 1, 1)              — 8 warps × 32 lanes
   ========================================================================= */
extern "C" __global__ void gemm_tq2_g128_v7(
    const unsigned char* __restrict__ soa_raw,
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

    const unsigned int blocks_per_row = k >> 7u;
    const unsigned long long total_blocks = (unsigned long long)n_rows * blocks_per_row;
    const unsigned long long qs_offset   = total_blocks * 2u;
    const unsigned short* __restrict__ scales = (const unsigned short* __restrict__)soa_raw;

    /* Process batch columns in 8-column outer chunks (cap-of-8 fix). */
    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned long long block_idx = (unsigned long long)row * blocks_per_row + b;
            const float scale = fast_fp16_to_float(scales[block_idx]);
            /* Load 32-byte qs block as 8 × u32 via non-caching vector loads */
            const unsigned int* qs_ptr =
                (const unsigned int*)(soa_raw + qs_offset + block_idx * 32u);
            unsigned int w0, w1, w2, w3, w4, w5, w6, w7;
            asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
                : "l"(qs_ptr));
            asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(w4), "=r"(w5), "=r"(w6), "=r"(w7)
                : "l"(qs_ptr + 4u));

            const unsigned int base = b * 128u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                float bsum = 0.0f;
                bsum += pf_byte_dot_tq2((w0      ) & 0xFFu, inp + base +   0u);
                bsum += pf_byte_dot_tq2((w0 >>  8) & 0xFFu, inp + base +   4u);
                bsum += pf_byte_dot_tq2((w0 >> 16) & 0xFFu, inp + base +   8u);
                bsum += pf_byte_dot_tq2((w0 >> 24) & 0xFFu, inp + base +  12u);
                bsum += pf_byte_dot_tq2((w1      ) & 0xFFu, inp + base +  16u);
                bsum += pf_byte_dot_tq2((w1 >>  8) & 0xFFu, inp + base +  20u);
                bsum += pf_byte_dot_tq2((w1 >> 16) & 0xFFu, inp + base +  24u);
                bsum += pf_byte_dot_tq2((w1 >> 24) & 0xFFu, inp + base +  28u);
                bsum += pf_byte_dot_tq2((w2      ) & 0xFFu, inp + base +  32u);
                bsum += pf_byte_dot_tq2((w2 >>  8) & 0xFFu, inp + base +  36u);
                bsum += pf_byte_dot_tq2((w2 >> 16) & 0xFFu, inp + base +  40u);
                bsum += pf_byte_dot_tq2((w2 >> 24) & 0xFFu, inp + base +  44u);
                bsum += pf_byte_dot_tq2((w3      ) & 0xFFu, inp + base +  48u);
                bsum += pf_byte_dot_tq2((w3 >>  8) & 0xFFu, inp + base +  52u);
                bsum += pf_byte_dot_tq2((w3 >> 16) & 0xFFu, inp + base +  56u);
                bsum += pf_byte_dot_tq2((w3 >> 24) & 0xFFu, inp + base +  60u);
                bsum += pf_byte_dot_tq2((w4      ) & 0xFFu, inp + base +  64u);
                bsum += pf_byte_dot_tq2((w4 >>  8) & 0xFFu, inp + base +  68u);
                bsum += pf_byte_dot_tq2((w4 >> 16) & 0xFFu, inp + base +  72u);
                bsum += pf_byte_dot_tq2((w4 >> 24) & 0xFFu, inp + base +  76u);
                bsum += pf_byte_dot_tq2((w5      ) & 0xFFu, inp + base +  80u);
                bsum += pf_byte_dot_tq2((w5 >>  8) & 0xFFu, inp + base +  84u);
                bsum += pf_byte_dot_tq2((w5 >> 16) & 0xFFu, inp + base +  88u);
                bsum += pf_byte_dot_tq2((w5 >> 24) & 0xFFu, inp + base +  92u);
                bsum += pf_byte_dot_tq2((w6      ) & 0xFFu, inp + base +  96u);
                bsum += pf_byte_dot_tq2((w6 >>  8) & 0xFFu, inp + base + 100u);
                bsum += pf_byte_dot_tq2((w6 >> 16) & 0xFFu, inp + base + 104u);
                bsum += pf_byte_dot_tq2((w6 >> 24) & 0xFFu, inp + base + 108u);
                bsum += pf_byte_dot_tq2((w7      ) & 0xFFu, inp + base + 112u);
                bsum += pf_byte_dot_tq2((w7 >>  8) & 0xFFu, inp + base + 116u);
                bsum += pf_byte_dot_tq2((w7 >> 16) & 0xFFu, inp + base + 120u);
                bsum += pf_byte_dot_tq2((w7 >> 24) & 0xFFu, inp + base + 124u);
                col_sums[col] += scale * bsum;
            }
        }

        /* Warp-shuffle reduction and write outputs (column-major, accumulate) */
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
   Kernel 7 — gemm_tq2_g128_v7_residual
   Batch TQ2 GEMM + fused in-place residual add.

   For each (row, col): outputs[col*n_rows + row] = residual[col*n_rows + row] + sum

   Same grid/block config as gemm_tq2_g128_v7.
   ========================================================================= */
extern "C" __global__ void gemm_tq2_g128_v7_residual(
    const unsigned char* __restrict__ soa_raw,
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

    const unsigned int blocks_per_row = k >> 7u;
    const unsigned long long total_blocks = (unsigned long long)n_rows * blocks_per_row;
    const unsigned long long qs_offset   = total_blocks * 2u;
    const unsigned short* __restrict__ scales = (const unsigned short* __restrict__)soa_raw;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned long long block_idx = (unsigned long long)row * blocks_per_row + b;
            const float scale = fast_fp16_to_float(scales[block_idx]);
            const unsigned int* qs_ptr =
                (const unsigned int*)(soa_raw + qs_offset + block_idx * 32u);
            unsigned int w0, w1, w2, w3, w4, w5, w6, w7;
            asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
                : "l"(qs_ptr));
            asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(w4), "=r"(w5), "=r"(w6), "=r"(w7)
                : "l"(qs_ptr + 4u));

            const unsigned int base = b * 128u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                float bsum = 0.0f;
                bsum += pf_byte_dot_tq2((w0      ) & 0xFFu, inp + base +   0u);
                bsum += pf_byte_dot_tq2((w0 >>  8) & 0xFFu, inp + base +   4u);
                bsum += pf_byte_dot_tq2((w0 >> 16) & 0xFFu, inp + base +   8u);
                bsum += pf_byte_dot_tq2((w0 >> 24) & 0xFFu, inp + base +  12u);
                bsum += pf_byte_dot_tq2((w1      ) & 0xFFu, inp + base +  16u);
                bsum += pf_byte_dot_tq2((w1 >>  8) & 0xFFu, inp + base +  20u);
                bsum += pf_byte_dot_tq2((w1 >> 16) & 0xFFu, inp + base +  24u);
                bsum += pf_byte_dot_tq2((w1 >> 24) & 0xFFu, inp + base +  28u);
                bsum += pf_byte_dot_tq2((w2      ) & 0xFFu, inp + base +  32u);
                bsum += pf_byte_dot_tq2((w2 >>  8) & 0xFFu, inp + base +  36u);
                bsum += pf_byte_dot_tq2((w2 >> 16) & 0xFFu, inp + base +  40u);
                bsum += pf_byte_dot_tq2((w2 >> 24) & 0xFFu, inp + base +  44u);
                bsum += pf_byte_dot_tq2((w3      ) & 0xFFu, inp + base +  48u);
                bsum += pf_byte_dot_tq2((w3 >>  8) & 0xFFu, inp + base +  52u);
                bsum += pf_byte_dot_tq2((w3 >> 16) & 0xFFu, inp + base +  56u);
                bsum += pf_byte_dot_tq2((w3 >> 24) & 0xFFu, inp + base +  60u);
                bsum += pf_byte_dot_tq2((w4      ) & 0xFFu, inp + base +  64u);
                bsum += pf_byte_dot_tq2((w4 >>  8) & 0xFFu, inp + base +  68u);
                bsum += pf_byte_dot_tq2((w4 >> 16) & 0xFFu, inp + base +  72u);
                bsum += pf_byte_dot_tq2((w4 >> 24) & 0xFFu, inp + base +  76u);
                bsum += pf_byte_dot_tq2((w5      ) & 0xFFu, inp + base +  80u);
                bsum += pf_byte_dot_tq2((w5 >>  8) & 0xFFu, inp + base +  84u);
                bsum += pf_byte_dot_tq2((w5 >> 16) & 0xFFu, inp + base +  88u);
                bsum += pf_byte_dot_tq2((w5 >> 24) & 0xFFu, inp + base +  92u);
                bsum += pf_byte_dot_tq2((w6      ) & 0xFFu, inp + base +  96u);
                bsum += pf_byte_dot_tq2((w6 >>  8) & 0xFFu, inp + base + 100u);
                bsum += pf_byte_dot_tq2((w6 >> 16) & 0xFFu, inp + base + 104u);
                bsum += pf_byte_dot_tq2((w6 >> 24) & 0xFFu, inp + base + 108u);
                bsum += pf_byte_dot_tq2((w7      ) & 0xFFu, inp + base + 112u);
                bsum += pf_byte_dot_tq2((w7 >>  8) & 0xFFu, inp + base + 116u);
                bsum += pf_byte_dot_tq2((w7 >> 16) & 0xFFu, inp + base + 120u);
                bsum += pf_byte_dot_tq2((w7 >> 24) & 0xFFu, inp + base + 124u);
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
   Kernel 8 — fused_gate_up_swiglu_gemm_tq2
   Batch fused gate+up TQ2 GEMM with SwiGLU epilogue for prefill.

   Weight layout (concatenated gate+up SoA, total_rows = 2*n_rows):
     [scales: 2*n_rows*blocks_per_row × 2B FP16]
     [qs:     2*n_rows*blocks_per_row × 32B, at offset = total_blocks_fused * 2]
   Gate row r: block_idx = r * blocks_per_row + b
   Up   row r: block_idx = (r + n_rows) * blocks_per_row + b

   For each (row, col): outputs[col * n_rows + row] = SiLU(gate_s) * up_s

   Grid:  (ceil(n_rows/8), 1, 1)
   Block: (256, 1, 1)
   ========================================================================= */
extern "C" __global__ void fused_gate_up_swiglu_gemm_tq2(
    const unsigned char* __restrict__ soa_raw,
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

    const unsigned int total_rows_fused = 2u * n_rows;
    const unsigned int blocks_per_row   = k >> 7u;
    const unsigned long long total_blocks_fused =
        (unsigned long long)total_rows_fused * blocks_per_row;
    const unsigned long long qs_offset = total_blocks_fused * 2u;
    const unsigned short* __restrict__ scales = (const unsigned short* __restrict__)soa_raw;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8];
        float up_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) { gate_sums[c] = 0.0f; up_sums[c] = 0.0f; }

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            /* ── Gate block (row r) ─────────────────────────────────────────── */
            const unsigned long long gate_idx =
                (unsigned long long)row * blocks_per_row + b;
            const float sg = fast_fp16_to_float(scales[gate_idx]);
            const unsigned int* gqs =
                (const unsigned int*)(soa_raw + qs_offset + gate_idx * 32u);
            unsigned int gw0, gw1, gw2, gw3, gw4, gw5, gw6, gw7;
            asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(gw0), "=r"(gw1), "=r"(gw2), "=r"(gw3)
                : "l"(gqs));
            asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(gw4), "=r"(gw5), "=r"(gw6), "=r"(gw7)
                : "l"(gqs + 4u));

            /* ── Up block (row r + n_rows) ──────────────────────────────────── */
            const unsigned long long up_idx =
                (unsigned long long)(row + n_rows) * blocks_per_row + b;
            const float su = fast_fp16_to_float(scales[up_idx]);
            const unsigned int* uqs =
                (const unsigned int*)(soa_raw + qs_offset + up_idx * 32u);
            unsigned int uw0, uw1, uw2, uw3, uw4, uw5, uw6, uw7;
            asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(uw0), "=r"(uw1), "=r"(uw2), "=r"(uw3)
                : "l"(uqs));
            asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(uw4), "=r"(uw5), "=r"(uw6), "=r"(uw7)
                : "l"(uqs + 4u));

            const unsigned int base = b * 128u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* inp = inputs + (unsigned long long)(col_base + col) * k;
                float gsum = 0.0f;
                float usum = 0.0f;
                gsum += pf_byte_dot_tq2((gw0      ) & 0xFFu, inp + base +   0u);
                gsum += pf_byte_dot_tq2((gw0 >>  8) & 0xFFu, inp + base +   4u);
                gsum += pf_byte_dot_tq2((gw0 >> 16) & 0xFFu, inp + base +   8u);
                gsum += pf_byte_dot_tq2((gw0 >> 24) & 0xFFu, inp + base +  12u);
                gsum += pf_byte_dot_tq2((gw1      ) & 0xFFu, inp + base +  16u);
                gsum += pf_byte_dot_tq2((gw1 >>  8) & 0xFFu, inp + base +  20u);
                gsum += pf_byte_dot_tq2((gw1 >> 16) & 0xFFu, inp + base +  24u);
                gsum += pf_byte_dot_tq2((gw1 >> 24) & 0xFFu, inp + base +  28u);
                gsum += pf_byte_dot_tq2((gw2      ) & 0xFFu, inp + base +  32u);
                gsum += pf_byte_dot_tq2((gw2 >>  8) & 0xFFu, inp + base +  36u);
                gsum += pf_byte_dot_tq2((gw2 >> 16) & 0xFFu, inp + base +  40u);
                gsum += pf_byte_dot_tq2((gw2 >> 24) & 0xFFu, inp + base +  44u);
                gsum += pf_byte_dot_tq2((gw3      ) & 0xFFu, inp + base +  48u);
                gsum += pf_byte_dot_tq2((gw3 >>  8) & 0xFFu, inp + base +  52u);
                gsum += pf_byte_dot_tq2((gw3 >> 16) & 0xFFu, inp + base +  56u);
                gsum += pf_byte_dot_tq2((gw3 >> 24) & 0xFFu, inp + base +  60u);
                gsum += pf_byte_dot_tq2((gw4      ) & 0xFFu, inp + base +  64u);
                gsum += pf_byte_dot_tq2((gw4 >>  8) & 0xFFu, inp + base +  68u);
                gsum += pf_byte_dot_tq2((gw4 >> 16) & 0xFFu, inp + base +  72u);
                gsum += pf_byte_dot_tq2((gw4 >> 24) & 0xFFu, inp + base +  76u);
                gsum += pf_byte_dot_tq2((gw5      ) & 0xFFu, inp + base +  80u);
                gsum += pf_byte_dot_tq2((gw5 >>  8) & 0xFFu, inp + base +  84u);
                gsum += pf_byte_dot_tq2((gw5 >> 16) & 0xFFu, inp + base +  88u);
                gsum += pf_byte_dot_tq2((gw5 >> 24) & 0xFFu, inp + base +  92u);
                gsum += pf_byte_dot_tq2((gw6      ) & 0xFFu, inp + base +  96u);
                gsum += pf_byte_dot_tq2((gw6 >>  8) & 0xFFu, inp + base + 100u);
                gsum += pf_byte_dot_tq2((gw6 >> 16) & 0xFFu, inp + base + 104u);
                gsum += pf_byte_dot_tq2((gw6 >> 24) & 0xFFu, inp + base + 108u);
                gsum += pf_byte_dot_tq2((gw7      ) & 0xFFu, inp + base + 112u);
                gsum += pf_byte_dot_tq2((gw7 >>  8) & 0xFFu, inp + base + 116u);
                gsum += pf_byte_dot_tq2((gw7 >> 16) & 0xFFu, inp + base + 120u);
                gsum += pf_byte_dot_tq2((gw7 >> 24) & 0xFFu, inp + base + 124u);
                usum += pf_byte_dot_tq2((uw0      ) & 0xFFu, inp + base +   0u);
                usum += pf_byte_dot_tq2((uw0 >>  8) & 0xFFu, inp + base +   4u);
                usum += pf_byte_dot_tq2((uw0 >> 16) & 0xFFu, inp + base +   8u);
                usum += pf_byte_dot_tq2((uw0 >> 24) & 0xFFu, inp + base +  12u);
                usum += pf_byte_dot_tq2((uw1      ) & 0xFFu, inp + base +  16u);
                usum += pf_byte_dot_tq2((uw1 >>  8) & 0xFFu, inp + base +  20u);
                usum += pf_byte_dot_tq2((uw1 >> 16) & 0xFFu, inp + base +  24u);
                usum += pf_byte_dot_tq2((uw1 >> 24) & 0xFFu, inp + base +  28u);
                usum += pf_byte_dot_tq2((uw2      ) & 0xFFu, inp + base +  32u);
                usum += pf_byte_dot_tq2((uw2 >>  8) & 0xFFu, inp + base +  36u);
                usum += pf_byte_dot_tq2((uw2 >> 16) & 0xFFu, inp + base +  40u);
                usum += pf_byte_dot_tq2((uw2 >> 24) & 0xFFu, inp + base +  44u);
                usum += pf_byte_dot_tq2((uw3      ) & 0xFFu, inp + base +  48u);
                usum += pf_byte_dot_tq2((uw3 >>  8) & 0xFFu, inp + base +  52u);
                usum += pf_byte_dot_tq2((uw3 >> 16) & 0xFFu, inp + base +  56u);
                usum += pf_byte_dot_tq2((uw3 >> 24) & 0xFFu, inp + base +  60u);
                usum += pf_byte_dot_tq2((uw4      ) & 0xFFu, inp + base +  64u);
                usum += pf_byte_dot_tq2((uw4 >>  8) & 0xFFu, inp + base +  68u);
                usum += pf_byte_dot_tq2((uw4 >> 16) & 0xFFu, inp + base +  72u);
                usum += pf_byte_dot_tq2((uw4 >> 24) & 0xFFu, inp + base +  76u);
                usum += pf_byte_dot_tq2((uw5      ) & 0xFFu, inp + base +  80u);
                usum += pf_byte_dot_tq2((uw5 >>  8) & 0xFFu, inp + base +  84u);
                usum += pf_byte_dot_tq2((uw5 >> 16) & 0xFFu, inp + base +  88u);
                usum += pf_byte_dot_tq2((uw5 >> 24) & 0xFFu, inp + base +  92u);
                usum += pf_byte_dot_tq2((uw6      ) & 0xFFu, inp + base +  96u);
                usum += pf_byte_dot_tq2((uw6 >>  8) & 0xFFu, inp + base + 100u);
                usum += pf_byte_dot_tq2((uw6 >> 16) & 0xFFu, inp + base + 104u);
                usum += pf_byte_dot_tq2((uw6 >> 24) & 0xFFu, inp + base + 108u);
                usum += pf_byte_dot_tq2((uw7      ) & 0xFFu, inp + base + 112u);
                usum += pf_byte_dot_tq2((uw7 >>  8) & 0xFFu, inp + base + 116u);
                usum += pf_byte_dot_tq2((uw7 >> 16) & 0xFFu, inp + base + 120u);
                usum += pf_byte_dot_tq2((uw7 >> 24) & 0xFFu, inp + base + 124u);
                gate_sums[col] += sg * gsum;
                up_sums[col]   += su * usum;
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
                outputs[(unsigned long long)(col_base + col) * n_rows + row] = silu(gs) * us;
            }
        }
    }
}
"#;
