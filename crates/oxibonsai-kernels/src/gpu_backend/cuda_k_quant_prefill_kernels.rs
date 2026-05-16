//! CUDA C kernel source strings for OxiBonsai K-quant batch GEMM (prefill) operations.
//!
//! # K-quant prefill kernel catalogue
//!
//! | Kernel                               | Description                                              |
//! |--------------------------------------|----------------------------------------------------------|
//! | `gemm_q2k`                           | Batch GEMM: Q2_K AoS, col-major I/O, col_sums[8]       |
//! | `gemm_q2k_residual`                  | Q2_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q2k`      | Fused gate+up Q2_K GEMM with SwiGLU epilogue            |
//! | `gemm_q3k`                           | Batch GEMM: Q3_K AoS, col-major I/O, col_sums[8]       |
//! | `gemm_q3k_residual`                  | Q3_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q3k`      | Fused gate+up Q3_K GEMM with SwiGLU epilogue            |
//! | `gemm_q4k`                           | Batch GEMM: Q4_K AoS, col-major I/O, col_sums[8]       |
//! | `gemm_q4k_residual`                  | Q4_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q4k`      | Fused gate+up Q4_K GEMM with SwiGLU epilogue            |
//! | `gemm_q5k`                           | Batch GEMM: Q5_K AoS, col-major I/O, col_sums[8]       |
//! | `gemm_q5k_residual`                  | Q5_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q5k`      | Fused gate+up Q5_K GEMM with SwiGLU epilogue            |
//! | `gemm_q6k`                           | Batch GEMM: Q6_K AoS, col-major I/O, col_sums[8]       |
//! | `gemm_q6k_residual`                  | Q6_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q6k`      | Fused gate+up Q6_K GEMM with SwiGLU epilogue            |
//! | `gemm_q8k`                           | Batch GEMM: Q8_K AoS, col-major I/O, col_sums[8]       |
//! | `gemm_q8k_residual`                  | Q8_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q8k`      | Fused gate+up Q8_K GEMM with SwiGLU epilogue            |
//!
//! # Block layouts (QK_K = 256 weights per super-block)
//!
//! **Q2_K** (84 bytes/block, 256 weights):
//! ```text
//! bytes  0-15: scales[16]  — low nibble = sub_sc, high nibble = sub_mn
//! bytes 16-79: qs[64]      — 2 bits/weight, 4/byte (LSB-first)
//! bytes 80-81: d (FP16 LE)
//! bytes 82-83: dmin (FP16 LE)
//! 16 sub-blocks × 16 weights. dequant: d*sc*q - dmin*mn (q ∈ [0,3])
//! ```
//!
//! **Q3_K** (110 bytes/block, 256 weights):
//! ```text
//! bytes  0-31:  hmask[32]  — high bit/weight, 8/byte
//! bytes 32-95:  qs[64]     — low 2 bits/weight, 4/byte (LSB-first)
//! bytes 96-107: scales[12] — 16×4-bit signed nibbles, 2/byte
//! bytes 108-109: d (FP16 LE)
//! q3 = lo2|(hi<<2), q3_signed = q3-4. signed_sc = nibble-8.
//! dequant: d*signed_sc*q3_signed
//! ```
//!
//! **Q4_K** (144 bytes/block, 256 weights):
//! ```text
//! bytes  0- 1: d (FP16 LE)
//! bytes  2- 3: dmin (FP16 LE)
//! bytes  4-15: scales[12]  — 6-bit sc[8] + 6-bit mn[8] (decoded by helper)
//! bytes 16-143: qs[128]    — 4 bits/weight, 2/byte (nibbles)
//! 8 sub-blocks × 32 weights. dequant: d*sc[sub]*q - dmin*mn[sub]
//! ```
//!
//! **Q5_K** (176 bytes/block, 256 weights):
//! ```text
//! bytes  0- 1: d (FP16 LE)
//! bytes  2- 3: dmin (FP16 LE)
//! bytes  4-15: scales[12]  — same 6-bit packing as Q4_K
//! bytes 16-47: qh[32]      — high bit/weight, 8/byte
//! bytes 48-175: qs[128]    — low 4 bits, 2/byte (nibbles)
//! 8 sub-blocks × 32 weights. q5 = nibble|(hi<<4). dequant: d*sc[sub]*q5 - dmin*mn[sub]
//! ```
//!
//! **Q6_K** (210 bytes/block, 256 weights):
//! ```text
//! bytes  0-127:  ql[128]    — low 4 bits/weight, 2/byte (nibbles)
//! bytes 128-191: qh[64]     — high 2 bits/weight, 4/byte (2 bits each)
//! bytes 192-207: scales[16] — signed int8, 1/sub-block
//! bytes 208-209: d (FP16 LE)
//! 16 sub-blocks × 16 weights. q6 = nibble|(hi2<<4), q6_signed = q6-32.
//! dequant: d*scales_i8[sub]*q6_signed
//! ```
//!
//! **Q8_K** (292 bytes/block, 256 weights):
//! ```text
//! bytes  0-3:   d (FP32 LE)    — NOTE: float, not FP16!
//! bytes  4-259: qs[256] (i8)   — 256 signed int8 weights
//! bytes 260-291: bsums (i16)   — not used in GEMM
//! dequant: d_f32 * qs[i]
//! ```
//!
//! # Grid / block config
//! - Grid:  `(ceil(n_rows / 8), 1, 1)` — 8 warps per CTA
//! - Block: `(256, 1, 1)` — 8 warps × 32 lanes
//! - `k` must be a positive multiple of 256 for all K-quant formats

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

/// CUDA C source for all K-quant batch GEMM (prefill) kernels.
///
/// All kernels use AoS weight layout (super-blocks stored contiguously as-is from GGUF).
/// Batch tensors use column-major layout: `buf[col * dim + element]`.
/// The cap-of-8 outer loop prevents silent bugs when batch_size > 8.
pub const CUDA_K_QUANT_PREFILL_KERNELS_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA K-quant prefill (batch GEMM) kernels.
   Formats: Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K
   QK_K = 256 weights per super-block for all formats.

   Batch tensors: column-major  buf[col * dim + element]
   Grid:  (ceil(n_rows/8), 1, 1)  — 8 warps per CTA, 1 warp/row
   Block: (256, 1, 1)             — 8 warps × 32 lanes
   k must be a positive multiple of 256.
   ========================================================================= */

/* ── Hardware FP16 → FP32 via PTX (SM 6.0+, 1 instruction) ─────────────── */
static __device__ __forceinline__ float kq_pf_fast_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* ── SiLU activation: x · σ(x) ─────────────────────────────────────────── */
static __device__ __forceinline__ float kq_pf_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ── Q4_K / Q5_K: decode 12-byte scales array into 8 × 6-bit sc and mn ─── */
static __device__ void kq_pf_decode_6bit_scales(
    const unsigned char* s,
    unsigned char sc_out[8],
    unsigned char mn_out[8]
) {
    /* Low 4 bits of scales from bytes 0..3 (2 per byte) */
    sc_out[0] = s[0] & 0x0Fu;  sc_out[1] = (s[0] >> 4u) & 0x0Fu;
    sc_out[2] = s[1] & 0x0Fu;  sc_out[3] = (s[1] >> 4u) & 0x0Fu;
    sc_out[4] = s[2] & 0x0Fu;  sc_out[5] = (s[2] >> 4u) & 0x0Fu;
    sc_out[6] = s[3] & 0x0Fu;  sc_out[7] = (s[3] >> 4u) & 0x0Fu;
    /* Low 4 bits of mins from bytes 4..7 */
    mn_out[0] = s[4] & 0x0Fu;  mn_out[1] = (s[4] >> 4u) & 0x0Fu;
    mn_out[2] = s[5] & 0x0Fu;  mn_out[3] = (s[5] >> 4u) & 0x0Fu;
    mn_out[4] = s[6] & 0x0Fu;  mn_out[5] = (s[6] >> 4u) & 0x0Fu;
    mn_out[6] = s[7] & 0x0Fu;  mn_out[7] = (s[7] >> 4u) & 0x0Fu;
    /* Upper 2 bits of scales from bytes 8..9 */
    sc_out[0] |= ((s[8] >> 0u) & 0x03u) << 4u;
    sc_out[1] |= ((s[8] >> 2u) & 0x03u) << 4u;
    sc_out[2] |= ((s[8] >> 4u) & 0x03u) << 4u;
    sc_out[3] |= ((s[8] >> 6u) & 0x03u) << 4u;
    sc_out[4] |= ((s[9] >> 0u) & 0x03u) << 4u;
    sc_out[5] |= ((s[9] >> 2u) & 0x03u) << 4u;
    sc_out[6] |= ((s[9] >> 4u) & 0x03u) << 4u;
    sc_out[7] |= ((s[9] >> 6u) & 0x03u) << 4u;
    /* Upper 2 bits of mins from bytes 10..11 */
    mn_out[0] |= ((s[10] >> 0u) & 0x03u) << 4u;
    mn_out[1] |= ((s[10] >> 2u) & 0x03u) << 4u;
    mn_out[2] |= ((s[10] >> 4u) & 0x03u) << 4u;
    mn_out[3] |= ((s[10] >> 6u) & 0x03u) << 4u;
    mn_out[4] |= ((s[11] >> 0u) & 0x03u) << 4u;
    mn_out[5] |= ((s[11] >> 2u) & 0x03u) << 4u;
    mn_out[6] |= ((s[11] >> 4u) & 0x03u) << 4u;
    mn_out[7] |= ((s[11] >> 6u) & 0x03u) << 4u;
}

/* =========================================================================
   Q2_K kernels (84 bytes/block, 256 weights)
   Block: [scales:16][qs:64][d_f16:2 @80][dmin_f16:2 @82]
   16 sub-blocks × 16 weights. 2-bit quant, per-sub scale/min.
   dequant: d*sc*q - dmin*mn  (q ∈ [0,3], sc=low nibble, mn=high nibble)
   ========================================================================= */

/* ── Kernel 1: gemm_q2k ─────────────────────────────────────────────────── */
extern "C" __global__ void gemm_q2k(
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

    const unsigned int blocks_per_row = k >> 8u;  /* k / 256 */

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 84u;

            const unsigned short d_raw    = (unsigned short)bptr[80]
                                          | ((unsigned short)bptr[81] << 8u);
            const unsigned short dmin_raw = (unsigned short)bptr[82]
                                          | ((unsigned short)bptr[83] << 8u);
            const float d    = kq_pf_fast_fp16_to_float(d_raw);
            const float dmin = kq_pf_fast_fp16_to_float(dmin_raw);

            const unsigned int x_base = b << 8u;  /* b * 256 */

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 16
                for (unsigned int sub = 0u; sub < 16u; ++sub) {
                    const unsigned char sc_byte = bptr[sub];
                    const float sub_sc = (float)(sc_byte & 0x0Fu);
                    const float sub_mn = (float)((sc_byte >> 4u) & 0x0Fu);
                    const unsigned int w_base = sub * 16u;
                    const unsigned int q_base = sub * 4u;

                    float sub_acc  = 0.0f;
                    float sub_xsum = 0.0f;
                    #pragma unroll 4
                    for (unsigned int qb = 0u; qb < 4u; ++qb) {
                        const unsigned char byte_val = bptr[16u + q_base + qb];
                        #pragma unroll 4
                        for (unsigned int bit = 0u; bit < 4u; ++bit) {
                            const float q = (float)((byte_val >> (bit * 2u)) & 0x3u);
                            const float x = xbase[w_base + qb * 4u + bit];
                            sub_acc  += q * x;
                            sub_xsum += x;
                        }
                    }
                    bsum += d * sub_sc * sub_acc - dmin * sub_mn * sub_xsum;
                }
                col_sums[col] += bsum;
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

/* ── Kernel 2: gemm_q2k_residual ────────────────────────────────────────── */
extern "C" __global__ void gemm_q2k_residual(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 84u;

            const unsigned short d_raw    = (unsigned short)bptr[80]
                                          | ((unsigned short)bptr[81] << 8u);
            const unsigned short dmin_raw = (unsigned short)bptr[82]
                                          | ((unsigned short)bptr[83] << 8u);
            const float d    = kq_pf_fast_fp16_to_float(d_raw);
            const float dmin = kq_pf_fast_fp16_to_float(dmin_raw);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 16
                for (unsigned int sub = 0u; sub < 16u; ++sub) {
                    const unsigned char sc_byte = bptr[sub];
                    const float sub_sc = (float)(sc_byte & 0x0Fu);
                    const float sub_mn = (float)((sc_byte >> 4u) & 0x0Fu);
                    const unsigned int w_base = sub * 16u;
                    const unsigned int q_base = sub * 4u;

                    float sub_acc  = 0.0f;
                    float sub_xsum = 0.0f;
                    #pragma unroll 4
                    for (unsigned int qb = 0u; qb < 4u; ++qb) {
                        const unsigned char byte_val = bptr[16u + q_base + qb];
                        #pragma unroll 4
                        for (unsigned int bit = 0u; bit < 4u; ++bit) {
                            const float q = (float)((byte_val >> (bit * 2u)) & 0x3u);
                            const float x = xbase[w_base + qb * 4u + bit];
                            sub_acc  += q * x;
                            sub_xsum += x;
                        }
                    }
                    bsum += d * sub_sc * sub_acc - dmin * sub_mn * sub_xsum;
                }
                col_sums[col] += bsum;
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

/* ── Kernel 3: fused_gate_up_swiglu_gemm_q2k ───────────────────────────── */
extern "C" __global__ void fused_gate_up_swiglu_gemm_q2k(
    const unsigned char* __restrict__ gate_up_blocks,
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned long long up_block_offset =
        (unsigned long long)n_rows * blocks_per_row;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8];
        float up_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) { gate_sums[c] = 0.0f; up_sums[c] = 0.0f; }

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned long long g_idx = (unsigned long long)(row * blocks_per_row + b);
            const unsigned char* gbptr = gate_up_blocks + g_idx * 84u;
            const unsigned char* ubptr = gate_up_blocks + (up_block_offset + g_idx) * 84u;

            const unsigned short gd_raw    = (unsigned short)gbptr[80]
                                           | ((unsigned short)gbptr[81] << 8u);
            const unsigned short gdmin_raw = (unsigned short)gbptr[82]
                                           | ((unsigned short)gbptr[83] << 8u);
            const float gd    = kq_pf_fast_fp16_to_float(gd_raw);
            const float gdmin = kq_pf_fast_fp16_to_float(gdmin_raw);

            const unsigned short ud_raw    = (unsigned short)ubptr[80]
                                           | ((unsigned short)ubptr[81] << 8u);
            const unsigned short udmin_raw = (unsigned short)ubptr[82]
                                           | ((unsigned short)ubptr[83] << 8u);
            const float ud    = kq_pf_fast_fp16_to_float(ud_raw);
            const float udmin = kq_pf_fast_fp16_to_float(udmin_raw);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float gbsum = 0.0f;
                float ubsum = 0.0f;

                #pragma unroll 16
                for (unsigned int sub = 0u; sub < 16u; ++sub) {
                    const unsigned char gsc_byte = gbptr[sub];
                    const float gsub_sc = (float)(gsc_byte & 0x0Fu);
                    const float gsub_mn = (float)((gsc_byte >> 4u) & 0x0Fu);
                    const unsigned char usc_byte = ubptr[sub];
                    const float usub_sc = (float)(usc_byte & 0x0Fu);
                    const float usub_mn = (float)((usc_byte >> 4u) & 0x0Fu);

                    const unsigned int w_base = sub * 16u;
                    const unsigned int q_base = sub * 4u;

                    float gsub_acc = 0.0f; float gsub_xsum = 0.0f;
                    float usub_acc = 0.0f; float usub_xsum = 0.0f;
                    #pragma unroll 4
                    for (unsigned int qb = 0u; qb < 4u; ++qb) {
                        const unsigned char gbyte = gbptr[16u + q_base + qb];
                        const unsigned char ubyte = ubptr[16u + q_base + qb];
                        #pragma unroll 4
                        for (unsigned int bit = 0u; bit < 4u; ++bit) {
                            const float x  = xbase[w_base + qb * 4u + bit];
                            const float gq = (float)((gbyte >> (bit * 2u)) & 0x3u);
                            const float uq = (float)((ubyte >> (bit * 2u)) & 0x3u);
                            gsub_acc  += gq * x;
                            usub_acc  += uq * x;
                            gsub_xsum += x;
                            usub_xsum += x;
                        }
                    }
                    gbsum += gd * gsub_sc * gsub_acc - gdmin * gsub_mn * gsub_xsum;
                    ubsum += ud * usub_sc * usub_acc - udmin * usub_mn * usub_xsum;
                }
                gate_sums[col] += gbsum;
                up_sums[col]   += ubsum;
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
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Q3_K kernels (110 bytes/block, 256 weights)
   Block: [hmask:32 @0][qs:64 @32][scales:12 @96][d_f16:2 @108]
   16 sub-blocks × 16 weights. q3=lo2|(hi<<2), q3_signed=q3-4.
   signed_sc = nibble-8.  dequant: d*signed_sc*q3_signed
   ========================================================================= */

/* ── Kernel 4: gemm_q3k ─────────────────────────────────────────────────── */
extern "C" __global__ void gemm_q3k(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 110u;

            const unsigned short d_raw = (unsigned short)bptr[108]
                                       | ((unsigned short)bptr[109] << 8u);
            const float d = kq_pf_fast_fp16_to_float(d_raw);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 16
                for (unsigned int sub = 0u; sub < 16u; ++sub) {
                    const unsigned char sc_byte = bptr[96u + sub / 2u];
                    const unsigned int nibble = (sub & 1u) == 0u
                        ? (sc_byte & 0x0Fu)
                        : ((sc_byte >> 4u) & 0x0Fu);
                    const float signed_sc = (float)(int)nibble - 8.0f;
                    const unsigned int w_base = sub * 16u;

                    float sub_acc = 0.0f;
                    #pragma unroll 16
                    for (unsigned int j = 0u; j < 16u; ++j) {
                        const unsigned int wi = w_base + j;
                        const unsigned int hi  = (bptr[wi >> 3u] >> (wi & 7u)) & 0x1u;
                        const unsigned int lo2 = (bptr[32u + (wi >> 2u)] >> ((wi & 3u) * 2u)) & 0x3u;
                        const int q3_code   = (int)(lo2 | (hi << 2u));
                        const int q3_signed = q3_code - 4;
                        sub_acc += (float)q3_signed * xbase[wi];
                    }
                    bsum += d * signed_sc * sub_acc;
                }
                col_sums[col] += bsum;
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

/* ── Kernel 5: gemm_q3k_residual ────────────────────────────────────────── */
extern "C" __global__ void gemm_q3k_residual(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 110u;

            const unsigned short d_raw = (unsigned short)bptr[108]
                                       | ((unsigned short)bptr[109] << 8u);
            const float d = kq_pf_fast_fp16_to_float(d_raw);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 16
                for (unsigned int sub = 0u; sub < 16u; ++sub) {
                    const unsigned char sc_byte = bptr[96u + sub / 2u];
                    const unsigned int nibble = (sub & 1u) == 0u
                        ? (sc_byte & 0x0Fu)
                        : ((sc_byte >> 4u) & 0x0Fu);
                    const float signed_sc = (float)(int)nibble - 8.0f;
                    const unsigned int w_base = sub * 16u;

                    float sub_acc = 0.0f;
                    #pragma unroll 16
                    for (unsigned int j = 0u; j < 16u; ++j) {
                        const unsigned int wi = w_base + j;
                        const unsigned int hi  = (bptr[wi >> 3u] >> (wi & 7u)) & 0x1u;
                        const unsigned int lo2 = (bptr[32u + (wi >> 2u)] >> ((wi & 3u) * 2u)) & 0x3u;
                        const int q3_code   = (int)(lo2 | (hi << 2u));
                        const int q3_signed = q3_code - 4;
                        sub_acc += (float)q3_signed * xbase[wi];
                    }
                    bsum += d * signed_sc * sub_acc;
                }
                col_sums[col] += bsum;
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

/* ── Kernel 6: fused_gate_up_swiglu_gemm_q3k ───────────────────────────── */
extern "C" __global__ void fused_gate_up_swiglu_gemm_q3k(
    const unsigned char* __restrict__ gate_up_blocks,
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned long long up_block_offset =
        (unsigned long long)n_rows * blocks_per_row;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8];
        float up_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) { gate_sums[c] = 0.0f; up_sums[c] = 0.0f; }

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned long long g_idx = (unsigned long long)(row * blocks_per_row + b);
            const unsigned char* gbptr = gate_up_blocks + g_idx * 110u;
            const unsigned char* ubptr = gate_up_blocks + (up_block_offset + g_idx) * 110u;

            const unsigned short gd_raw = (unsigned short)gbptr[108]
                                        | ((unsigned short)gbptr[109] << 8u);
            const float gd = kq_pf_fast_fp16_to_float(gd_raw);

            const unsigned short ud_raw = (unsigned short)ubptr[108]
                                        | ((unsigned short)ubptr[109] << 8u);
            const float ud = kq_pf_fast_fp16_to_float(ud_raw);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float gbsum = 0.0f;
                float ubsum = 0.0f;

                #pragma unroll 16
                for (unsigned int sub = 0u; sub < 16u; ++sub) {
                    const unsigned char gsc_byte = gbptr[96u + sub / 2u];
                    const unsigned int gnibble = (sub & 1u) == 0u
                        ? (gsc_byte & 0x0Fu) : ((gsc_byte >> 4u) & 0x0Fu);
                    const float g_signed_sc = (float)(int)gnibble - 8.0f;

                    const unsigned char usc_byte = ubptr[96u + sub / 2u];
                    const unsigned int unibble = (sub & 1u) == 0u
                        ? (usc_byte & 0x0Fu) : ((usc_byte >> 4u) & 0x0Fu);
                    const float u_signed_sc = (float)(int)unibble - 8.0f;

                    const unsigned int w_base = sub * 16u;
                    float gsub_acc = 0.0f;
                    float usub_acc = 0.0f;

                    #pragma unroll 16
                    for (unsigned int j = 0u; j < 16u; ++j) {
                        const unsigned int wi = w_base + j;
                        const unsigned int ghi  = (gbptr[wi >> 3u] >> (wi & 7u)) & 0x1u;
                        const unsigned int glo2 = (gbptr[32u + (wi >> 2u)] >> ((wi & 3u) * 2u)) & 0x3u;
                        const int gq3  = (int)(glo2 | (ghi << 2u)) - 4;

                        const unsigned int uhi  = (ubptr[wi >> 3u] >> (wi & 7u)) & 0x1u;
                        const unsigned int ulo2 = (ubptr[32u + (wi >> 2u)] >> ((wi & 3u) * 2u)) & 0x3u;
                        const int uq3  = (int)(ulo2 | (uhi << 2u)) - 4;

                        const float x = xbase[wi];
                        gsub_acc += (float)gq3 * x;
                        usub_acc += (float)uq3 * x;
                    }
                    gbsum += gd * g_signed_sc * gsub_acc;
                    ubsum += ud * u_signed_sc * usub_acc;
                }
                gate_sums[col] += gbsum;
                up_sums[col]   += ubsum;
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
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Q4_K kernels (144 bytes/block, 256 weights)
   Block: [d_f16:2 @0][dmin_f16:2 @2][scales:12 @4][qs:128 @16]
   8 sub-blocks × 32 weights. 6-bit scale decode.
   dequant: d*sc[sub]*q - dmin*mn[sub]
   ========================================================================= */

/* ── Kernel 7: gemm_q4k ─────────────────────────────────────────────────── */
extern "C" __global__ void gemm_q4k(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 144u;

            const unsigned short d_raw    = (unsigned short)bptr[0]
                                          | ((unsigned short)bptr[1] << 8u);
            const unsigned short dmin_raw = (unsigned short)bptr[2]
                                          | ((unsigned short)bptr[3] << 8u);
            const float d    = kq_pf_fast_fp16_to_float(d_raw);
            const float dmin = kq_pf_fast_fp16_to_float(dmin_raw);

            unsigned char sc[8], mn[8];
            kq_pf_decode_6bit_scales(bptr + 4u, sc, mn);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 8
                for (unsigned int sub = 0u; sub < 8u; ++sub) {
                    const float sc_f = (float)sc[sub];
                    const float mn_f = (float)mn[sub];
                    const unsigned char* qs_sub = bptr + 16u + sub * 16u;
                    const float* x_sub = xbase + sub * 32u;

                    float sub_acc  = 0.0f;
                    float sub_xsum = 0.0f;
                    #pragma unroll 16
                    for (unsigned int nb = 0u; nb < 16u; ++nb) {
                        const unsigned int bv = qs_sub[nb];
                        const float q0 = (float)(bv & 0x0Fu);
                        const float q1 = (float)((bv >> 4u) & 0x0Fu);
                        const float x0 = x_sub[nb * 2u];
                        const float x1 = x_sub[nb * 2u + 1u];
                        sub_acc  += q0 * x0 + q1 * x1;
                        sub_xsum += x0 + x1;
                    }
                    bsum += d * sc_f * sub_acc - dmin * mn_f * sub_xsum;
                }
                col_sums[col] += bsum;
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

/* ── Kernel 8: gemm_q4k_residual ────────────────────────────────────────── */
extern "C" __global__ void gemm_q4k_residual(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 144u;

            const unsigned short d_raw    = (unsigned short)bptr[0]
                                          | ((unsigned short)bptr[1] << 8u);
            const unsigned short dmin_raw = (unsigned short)bptr[2]
                                          | ((unsigned short)bptr[3] << 8u);
            const float d    = kq_pf_fast_fp16_to_float(d_raw);
            const float dmin = kq_pf_fast_fp16_to_float(dmin_raw);

            unsigned char sc[8], mn[8];
            kq_pf_decode_6bit_scales(bptr + 4u, sc, mn);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 8
                for (unsigned int sub = 0u; sub < 8u; ++sub) {
                    const float sc_f = (float)sc[sub];
                    const float mn_f = (float)mn[sub];
                    const unsigned char* qs_sub = bptr + 16u + sub * 16u;
                    const float* x_sub = xbase + sub * 32u;

                    float sub_acc  = 0.0f;
                    float sub_xsum = 0.0f;
                    #pragma unroll 16
                    for (unsigned int nb = 0u; nb < 16u; ++nb) {
                        const unsigned int bv = qs_sub[nb];
                        const float q0 = (float)(bv & 0x0Fu);
                        const float q1 = (float)((bv >> 4u) & 0x0Fu);
                        const float x0 = x_sub[nb * 2u];
                        const float x1 = x_sub[nb * 2u + 1u];
                        sub_acc  += q0 * x0 + q1 * x1;
                        sub_xsum += x0 + x1;
                    }
                    bsum += d * sc_f * sub_acc - dmin * mn_f * sub_xsum;
                }
                col_sums[col] += bsum;
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

/* ── Kernel 9: fused_gate_up_swiglu_gemm_q4k ───────────────────────────── */
extern "C" __global__ void fused_gate_up_swiglu_gemm_q4k(
    const unsigned char* __restrict__ gate_up_blocks,
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned long long up_block_offset =
        (unsigned long long)n_rows * blocks_per_row;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8];
        float up_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) { gate_sums[c] = 0.0f; up_sums[c] = 0.0f; }

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned long long g_idx = (unsigned long long)(row * blocks_per_row + b);
            const unsigned char* gbptr = gate_up_blocks + g_idx * 144u;
            const unsigned char* ubptr = gate_up_blocks + (up_block_offset + g_idx) * 144u;

            const unsigned short gd_raw    = (unsigned short)gbptr[0]
                                           | ((unsigned short)gbptr[1] << 8u);
            const unsigned short gdmin_raw = (unsigned short)gbptr[2]
                                           | ((unsigned short)gbptr[3] << 8u);
            const float gd    = kq_pf_fast_fp16_to_float(gd_raw);
            const float gdmin = kq_pf_fast_fp16_to_float(gdmin_raw);

            const unsigned short ud_raw    = (unsigned short)ubptr[0]
                                           | ((unsigned short)ubptr[1] << 8u);
            const unsigned short udmin_raw = (unsigned short)ubptr[2]
                                           | ((unsigned short)ubptr[3] << 8u);
            const float ud    = kq_pf_fast_fp16_to_float(ud_raw);
            const float udmin = kq_pf_fast_fp16_to_float(udmin_raw);

            unsigned char gsc[8], gmn[8], usc[8], umn[8];
            kq_pf_decode_6bit_scales(gbptr + 4u, gsc, gmn);
            kq_pf_decode_6bit_scales(ubptr + 4u, usc, umn);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float gbsum = 0.0f;
                float ubsum = 0.0f;

                #pragma unroll 8
                for (unsigned int sub = 0u; sub < 8u; ++sub) {
                    const float gsc_f = (float)gsc[sub];
                    const float gmn_f = (float)gmn[sub];
                    const float usc_f = (float)usc[sub];
                    const float umn_f = (float)umn[sub];
                    const unsigned char* gqs_sub = gbptr + 16u + sub * 16u;
                    const unsigned char* uqs_sub = ubptr + 16u + sub * 16u;
                    const float* x_sub = xbase + sub * 32u;

                    float gsub_acc = 0.0f; float gsub_xsum = 0.0f;
                    float usub_acc = 0.0f; float usub_xsum = 0.0f;
                    #pragma unroll 16
                    for (unsigned int nb = 0u; nb < 16u; ++nb) {
                        const float x0 = x_sub[nb * 2u];
                        const float x1 = x_sub[nb * 2u + 1u];
                        const float gq0 = (float)(gqs_sub[nb] & 0x0Fu);
                        const float gq1 = (float)((gqs_sub[nb] >> 4u) & 0x0Fu);
                        const float uq0 = (float)(uqs_sub[nb] & 0x0Fu);
                        const float uq1 = (float)((uqs_sub[nb] >> 4u) & 0x0Fu);
                        gsub_acc  += gq0 * x0 + gq1 * x1;
                        gsub_xsum += x0 + x1;
                        usub_acc  += uq0 * x0 + uq1 * x1;
                        usub_xsum += x0 + x1;
                    }
                    gbsum += gd * gsc_f * gsub_acc - gdmin * gmn_f * gsub_xsum;
                    ubsum += ud * usc_f * usub_acc - udmin * umn_f * usub_xsum;
                }
                gate_sums[col] += gbsum;
                up_sums[col]   += ubsum;
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
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Q5_K kernels (176 bytes/block, 256 weights)
   Block: [d_f16:2 @0][dmin_f16:2 @2][scales:12 @4][qh:32 @16][qs:128 @48]
   8 sub-blocks × 32 weights. q5 = nibble|(hi<<4).
   dequant: d*sc[sub]*q5 - dmin*mn[sub]
   ========================================================================= */

/* ── Kernel 10: gemm_q5k ────────────────────────────────────────────────── */
extern "C" __global__ void gemm_q5k(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 176u;

            const unsigned short d_raw    = (unsigned short)bptr[0]
                                          | ((unsigned short)bptr[1] << 8u);
            const unsigned short dmin_raw = (unsigned short)bptr[2]
                                          | ((unsigned short)bptr[3] << 8u);
            const float d    = kq_pf_fast_fp16_to_float(d_raw);
            const float dmin = kq_pf_fast_fp16_to_float(dmin_raw);

            unsigned char sc[8], mn[8];
            kq_pf_decode_6bit_scales(bptr + 4u, sc, mn);

            const unsigned char* qh = bptr + 16u;
            const unsigned char* qs = bptr + 48u;
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 8
                for (unsigned int sub = 0u; sub < 8u; ++sub) {
                    const float sc_f = (float)sc[sub];
                    const float mn_f = (float)mn[sub];
                    const unsigned char* qs_sub = qs + sub * 16u;
                    const float* x_sub = xbase + sub * 32u;

                    float sub_acc  = 0.0f;
                    float sub_xsum = 0.0f;
                    #pragma unroll 16
                    for (unsigned int nb = 0u; nb < 16u; ++nb) {
                        const unsigned int wi0 = sub * 32u + nb * 2u;
                        const unsigned int wi1 = wi0 + 1u;
                        const unsigned int hi0 = (qh[wi0 >> 3u] >> (wi0 & 7u)) & 0x1u;
                        const unsigned int hi1 = (qh[wi1 >> 3u] >> (wi1 & 7u)) & 0x1u;
                        const unsigned int bv  = qs_sub[nb];
                        const float q0 = (float)((bv & 0x0Fu) | (hi0 << 4u));
                        const float q1 = (float)(((bv >> 4u) & 0x0Fu) | (hi1 << 4u));
                        const float x0 = x_sub[nb * 2u];
                        const float x1 = x_sub[nb * 2u + 1u];
                        sub_acc  += q0 * x0 + q1 * x1;
                        sub_xsum += x0 + x1;
                    }
                    bsum += d * sc_f * sub_acc - dmin * mn_f * sub_xsum;
                }
                col_sums[col] += bsum;
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

/* ── Kernel 11: gemm_q5k_residual ───────────────────────────────────────── */
extern "C" __global__ void gemm_q5k_residual(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 176u;

            const unsigned short d_raw    = (unsigned short)bptr[0]
                                          | ((unsigned short)bptr[1] << 8u);
            const unsigned short dmin_raw = (unsigned short)bptr[2]
                                          | ((unsigned short)bptr[3] << 8u);
            const float d    = kq_pf_fast_fp16_to_float(d_raw);
            const float dmin = kq_pf_fast_fp16_to_float(dmin_raw);

            unsigned char sc[8], mn[8];
            kq_pf_decode_6bit_scales(bptr + 4u, sc, mn);

            const unsigned char* qh = bptr + 16u;
            const unsigned char* qs = bptr + 48u;
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 8
                for (unsigned int sub = 0u; sub < 8u; ++sub) {
                    const float sc_f = (float)sc[sub];
                    const float mn_f = (float)mn[sub];
                    const unsigned char* qs_sub = qs + sub * 16u;
                    const float* x_sub = xbase + sub * 32u;

                    float sub_acc  = 0.0f;
                    float sub_xsum = 0.0f;
                    #pragma unroll 16
                    for (unsigned int nb = 0u; nb < 16u; ++nb) {
                        const unsigned int wi0 = sub * 32u + nb * 2u;
                        const unsigned int wi1 = wi0 + 1u;
                        const unsigned int hi0 = (qh[wi0 >> 3u] >> (wi0 & 7u)) & 0x1u;
                        const unsigned int hi1 = (qh[wi1 >> 3u] >> (wi1 & 7u)) & 0x1u;
                        const unsigned int bv  = qs_sub[nb];
                        const float q0 = (float)((bv & 0x0Fu) | (hi0 << 4u));
                        const float q1 = (float)(((bv >> 4u) & 0x0Fu) | (hi1 << 4u));
                        const float x0 = x_sub[nb * 2u];
                        const float x1 = x_sub[nb * 2u + 1u];
                        sub_acc  += q0 * x0 + q1 * x1;
                        sub_xsum += x0 + x1;
                    }
                    bsum += d * sc_f * sub_acc - dmin * mn_f * sub_xsum;
                }
                col_sums[col] += bsum;
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

/* ── Kernel 12: fused_gate_up_swiglu_gemm_q5k ──────────────────────────── */
extern "C" __global__ void fused_gate_up_swiglu_gemm_q5k(
    const unsigned char* __restrict__ gate_up_blocks,
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned long long up_block_offset =
        (unsigned long long)n_rows * blocks_per_row;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8];
        float up_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) { gate_sums[c] = 0.0f; up_sums[c] = 0.0f; }

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned long long g_idx = (unsigned long long)(row * blocks_per_row + b);
            const unsigned char* gbptr = gate_up_blocks + g_idx * 176u;
            const unsigned char* ubptr = gate_up_blocks + (up_block_offset + g_idx) * 176u;

            const unsigned short gd_raw    = (unsigned short)gbptr[0]
                                           | ((unsigned short)gbptr[1] << 8u);
            const unsigned short gdmin_raw = (unsigned short)gbptr[2]
                                           | ((unsigned short)gbptr[3] << 8u);
            const float gd    = kq_pf_fast_fp16_to_float(gd_raw);
            const float gdmin = kq_pf_fast_fp16_to_float(gdmin_raw);

            const unsigned short ud_raw    = (unsigned short)ubptr[0]
                                           | ((unsigned short)ubptr[1] << 8u);
            const unsigned short udmin_raw = (unsigned short)ubptr[2]
                                           | ((unsigned short)ubptr[3] << 8u);
            const float ud    = kq_pf_fast_fp16_to_float(ud_raw);
            const float udmin = kq_pf_fast_fp16_to_float(udmin_raw);

            unsigned char gsc[8], gmn[8], usc[8], umn[8];
            kq_pf_decode_6bit_scales(gbptr + 4u, gsc, gmn);
            kq_pf_decode_6bit_scales(ubptr + 4u, usc, umn);

            const unsigned char* gqh = gbptr + 16u;
            const unsigned char* gqs = gbptr + 48u;
            const unsigned char* uqh = ubptr + 16u;
            const unsigned char* uqs = ubptr + 48u;
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float gbsum = 0.0f;
                float ubsum = 0.0f;

                #pragma unroll 8
                for (unsigned int sub = 0u; sub < 8u; ++sub) {
                    const float gsc_f = (float)gsc[sub];
                    const float gmn_f = (float)gmn[sub];
                    const float usc_f = (float)usc[sub];
                    const float umn_f = (float)umn[sub];
                    const unsigned char* gqs_sub = gqs + sub * 16u;
                    const unsigned char* uqs_sub = uqs + sub * 16u;
                    const float* x_sub = xbase + sub * 32u;

                    float gsub_acc = 0.0f; float gsub_xsum = 0.0f;
                    float usub_acc = 0.0f; float usub_xsum = 0.0f;
                    #pragma unroll 16
                    for (unsigned int nb = 0u; nb < 16u; ++nb) {
                        const unsigned int wi0 = sub * 32u + nb * 2u;
                        const unsigned int wi1 = wi0 + 1u;
                        const unsigned int ghi0 = (gqh[wi0 >> 3u] >> (wi0 & 7u)) & 0x1u;
                        const unsigned int ghi1 = (gqh[wi1 >> 3u] >> (wi1 & 7u)) & 0x1u;
                        const unsigned int uhi0 = (uqh[wi0 >> 3u] >> (wi0 & 7u)) & 0x1u;
                        const unsigned int uhi1 = (uqh[wi1 >> 3u] >> (wi1 & 7u)) & 0x1u;
                        const unsigned int gbv  = gqs_sub[nb];
                        const unsigned int ubv  = uqs_sub[nb];
                        const float gq0 = (float)((gbv & 0x0Fu) | (ghi0 << 4u));
                        const float gq1 = (float)(((gbv >> 4u) & 0x0Fu) | (ghi1 << 4u));
                        const float uq0 = (float)((ubv & 0x0Fu) | (uhi0 << 4u));
                        const float uq1 = (float)(((ubv >> 4u) & 0x0Fu) | (uhi1 << 4u));
                        const float x0 = x_sub[nb * 2u];
                        const float x1 = x_sub[nb * 2u + 1u];
                        gsub_acc  += gq0 * x0 + gq1 * x1;
                        gsub_xsum += x0 + x1;
                        usub_acc  += uq0 * x0 + uq1 * x1;
                        usub_xsum += x0 + x1;
                    }
                    gbsum += gd * gsc_f * gsub_acc - gdmin * gmn_f * gsub_xsum;
                    ubsum += ud * usc_f * usub_acc - udmin * umn_f * usub_xsum;
                }
                gate_sums[col] += gbsum;
                up_sums[col]   += ubsum;
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
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Q6_K kernels (210 bytes/block, 256 weights)
   Block: [ql:128 @0][qh:64 @128][scales_i8:16 @192][d_f16:2 @208]
   16 sub-blocks × 16 weights. q6=nibble|(hi2<<4), q6_signed=q6-32.
   dequant: d*scales_i8[sub]*q6_signed
   ========================================================================= */

/* ── Kernel 13: gemm_q6k ────────────────────────────────────────────────── */
extern "C" __global__ void gemm_q6k(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 210u;

            const unsigned short d_raw = (unsigned short)bptr[208]
                                       | ((unsigned short)bptr[209] << 8u);
            const float d = kq_pf_fast_fp16_to_float(d_raw);

            const unsigned char*  ql        = bptr;
            const unsigned char*  qh        = bptr + 128u;
            const signed char*    scales_i8 = (const signed char*)(bptr + 192u);
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 16
                for (unsigned int sub = 0u; sub < 16u; ++sub) {
                    const float sc = (float)(int)scales_i8[sub];
                    const unsigned int w_base = sub * 16u;

                    float sub_acc = 0.0f;
                    #pragma unroll 16
                    for (unsigned int j = 0u; j < 16u; ++j) {
                        const unsigned int wi = w_base + j;
                        const unsigned int nibble = (ql[wi >> 1u] >> ((wi & 1u) * 4u)) & 0x0Fu;
                        const unsigned int hi2    = (qh[wi >> 2u] >> ((wi & 3u) * 2u)) & 0x03u;
                        const int q6        = (int)(nibble | (hi2 << 4u));
                        const int q6_signed = q6 - 32;
                        sub_acc += (float)q6_signed * xbase[wi];
                    }
                    bsum += d * sc * sub_acc;
                }
                col_sums[col] += bsum;
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

/* ── Kernel 14: gemm_q6k_residual ───────────────────────────────────────── */
extern "C" __global__ void gemm_q6k_residual(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 210u;

            const unsigned short d_raw = (unsigned short)bptr[208]
                                       | ((unsigned short)bptr[209] << 8u);
            const float d = kq_pf_fast_fp16_to_float(d_raw);

            const unsigned char*  ql        = bptr;
            const unsigned char*  qh        = bptr + 128u;
            const signed char*    scales_i8 = (const signed char*)(bptr + 192u);
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 16
                for (unsigned int sub = 0u; sub < 16u; ++sub) {
                    const float sc = (float)(int)scales_i8[sub];
                    const unsigned int w_base = sub * 16u;

                    float sub_acc = 0.0f;
                    #pragma unroll 16
                    for (unsigned int j = 0u; j < 16u; ++j) {
                        const unsigned int wi = w_base + j;
                        const unsigned int nibble = (ql[wi >> 1u] >> ((wi & 1u) * 4u)) & 0x0Fu;
                        const unsigned int hi2    = (qh[wi >> 2u] >> ((wi & 3u) * 2u)) & 0x03u;
                        const int q6        = (int)(nibble | (hi2 << 4u));
                        const int q6_signed = q6 - 32;
                        sub_acc += (float)q6_signed * xbase[wi];
                    }
                    bsum += d * sc * sub_acc;
                }
                col_sums[col] += bsum;
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

/* ── Kernel 15: fused_gate_up_swiglu_gemm_q6k ──────────────────────────── */
extern "C" __global__ void fused_gate_up_swiglu_gemm_q6k(
    const unsigned char* __restrict__ gate_up_blocks,
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned long long up_block_offset =
        (unsigned long long)n_rows * blocks_per_row;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8];
        float up_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) { gate_sums[c] = 0.0f; up_sums[c] = 0.0f; }

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned long long g_idx = (unsigned long long)(row * blocks_per_row + b);
            const unsigned char* gbptr = gate_up_blocks + g_idx * 210u;
            const unsigned char* ubptr = gate_up_blocks + (up_block_offset + g_idx) * 210u;

            const unsigned short gd_raw = (unsigned short)gbptr[208]
                                        | ((unsigned short)gbptr[209] << 8u);
            const float gd = kq_pf_fast_fp16_to_float(gd_raw);

            const unsigned short ud_raw = (unsigned short)ubptr[208]
                                        | ((unsigned short)ubptr[209] << 8u);
            const float ud = kq_pf_fast_fp16_to_float(ud_raw);

            const unsigned char*  gql       = gbptr;
            const unsigned char*  gqh       = gbptr + 128u;
            const signed char*    gsc_i8    = (const signed char*)(gbptr + 192u);
            const unsigned char*  uql       = ubptr;
            const unsigned char*  uqh       = ubptr + 128u;
            const signed char*    usc_i8    = (const signed char*)(ubptr + 192u);
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float gbsum = 0.0f;
                float ubsum = 0.0f;

                #pragma unroll 16
                for (unsigned int sub = 0u; sub < 16u; ++sub) {
                    const float gsc = (float)(int)gsc_i8[sub];
                    const float usc = (float)(int)usc_i8[sub];
                    const unsigned int w_base = sub * 16u;

                    float gsub_acc = 0.0f;
                    float usub_acc = 0.0f;
                    #pragma unroll 16
                    for (unsigned int j = 0u; j < 16u; ++j) {
                        const unsigned int wi = w_base + j;
                        const unsigned int gnibble = (gql[wi >> 1u] >> ((wi & 1u) * 4u)) & 0x0Fu;
                        const unsigned int ghi2    = (gqh[wi >> 2u] >> ((wi & 3u) * 2u)) & 0x03u;
                        const int gq6        = (int)(gnibble | (ghi2 << 4u));
                        const int gq6_signed = gq6 - 32;

                        const unsigned int unibble = (uql[wi >> 1u] >> ((wi & 1u) * 4u)) & 0x0Fu;
                        const unsigned int uhi2    = (uqh[wi >> 2u] >> ((wi & 3u) * 2u)) & 0x03u;
                        const int uq6        = (int)(unibble | (uhi2 << 4u));
                        const int uq6_signed = uq6 - 32;

                        const float x = xbase[wi];
                        gsub_acc += (float)gq6_signed * x;
                        usub_acc += (float)uq6_signed * x;
                    }
                    gbsum += gd * gsc * gsub_acc;
                    ubsum += ud * usc * usub_acc;
                }
                gate_sums[col] += gbsum;
                up_sums[col]   += ubsum;
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
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Q8_K kernels (292 bytes/block, 256 weights)
   Block: [d_f32:4 @0][qs:256 i8 @4][bsums:32 @260]
   d is f32 (not f16!). dequant: d_f32 * qs[i]
   ========================================================================= */

/* ── Kernel 16: gemm_q8k ────────────────────────────────────────────────── */
extern "C" __global__ void gemm_q8k(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 292u;

            /* d is stored as FP32 LE at bytes 0-3 */
            const float d = *(const float*)(bptr + 0u);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 32
                for (unsigned int j = 0u; j < 256u; ++j) {
                    const int q = (int)(signed char)bptr[4u + j];
                    bsum += (float)q * xbase[j];
                }
                col_sums[col] += d * bsum;
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

/* ── Kernel 17: gemm_q8k_residual ───────────────────────────────────────── */
extern "C" __global__ void gemm_q8k_residual(
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

    const unsigned int blocks_per_row = k >> 8u;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) col_sums[c] = 0.0f;

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned char* bptr = blocks
                + (unsigned long long)(row * blocks_per_row + b) * 292u;

            const float d = *(const float*)(bptr + 0u);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float bsum = 0.0f;

                #pragma unroll 32
                for (unsigned int j = 0u; j < 256u; ++j) {
                    const int q = (int)(signed char)bptr[4u + j];
                    bsum += (float)q * xbase[j];
                }
                col_sums[col] += d * bsum;
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

/* ── Kernel 18: fused_gate_up_swiglu_gemm_q8k ──────────────────────────── */
extern "C" __global__ void fused_gate_up_swiglu_gemm_q8k(
    const unsigned char* __restrict__ gate_up_blocks,
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned long long up_block_offset =
        (unsigned long long)n_rows * blocks_per_row;

    for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) {
        const unsigned int cols_remaining = batch_size - col_base;
        const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8];
        float up_sums[8];
        #pragma unroll
        for (unsigned int c = 0u; c < 8u; ++c) { gate_sums[c] = 0.0f; up_sums[c] = 0.0f; }

        for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
            const unsigned long long g_idx = (unsigned long long)(row * blocks_per_row + b);
            const unsigned char* gbptr = gate_up_blocks + g_idx * 292u;
            const unsigned char* ubptr = gate_up_blocks + (up_block_offset + g_idx) * 292u;

            const float gd = *(const float*)(gbptr + 0u);
            const float ud = *(const float*)(ubptr + 0u);

            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                float gbsum = 0.0f;
                float ubsum = 0.0f;

                #pragma unroll 32
                for (unsigned int j = 0u; j < 256u; ++j) {
                    const float x  = xbase[j];
                    const int gq = (int)(signed char)gbptr[4u + j];
                    const int uq = (int)(signed char)ubptr[4u + j];
                    gbsum += (float)gq * x;
                    ubsum += (float)uq * x;
                }
                gate_sums[col] += gd * gbsum;
                up_sums[col]   += ud * ubsum;
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
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}
"#;
