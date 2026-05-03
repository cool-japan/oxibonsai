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
"#;
