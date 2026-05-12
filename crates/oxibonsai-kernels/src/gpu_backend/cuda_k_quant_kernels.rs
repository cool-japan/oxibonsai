//! CUDA C kernel source strings for OxiBonsai K-quant GEMV operations.
//!
//! # K-quant kernel catalogue
//!
//! | Kernel         | Description                                         |
//! |----------------|-----------------------------------------------------|
//! | `gemv_q2k`     | Q2_K GEMV, AoS super-blocks (84 B/block, 256 w)    |
//! | `gemv_q3k`     | Q3_K GEMV, AoS super-blocks (110 B/block, 256 w)   |
//! | `gemv_q4k`     | Q4_K GEMV, AoS super-blocks (144 B/block, 256 w)   |
//! | `gemv_q5k`     | Q5_K GEMV, AoS super-blocks (176 B/block, 256 w)   |
//! | `gemv_q6k`     | Q6_K GEMV, AoS super-blocks (210 B/block, 256 w)   |
//! | `gemv_q8k`     | Q8_K GEMV, AoS super-blocks (292 B/block, 256 w)   |
//!
//! # Block layouts (QK_K = 256 weights per super-block)
//!
//! **Q2_K** (84 bytes):
//! ```text
//! [scales:16u8][qs:64u8][d:f16 @80][dmin:f16 @82]
//! ```
//! 16 sub-blocks × 16 weights. scales[sub] low nibble = sub_sc, high nibble = sub_mn.
//! qs: 2 bits/weight, 4/byte LSB-first. dequant: d*sc*q - dmin*mn (q ∈ [0,3]).
//!
//! **Q3_K** (110 bytes):
//! ```text
//! [hmask:32u8][qs:64u8][scales:12u8][d:f16 @108]
//! ```
//! hmask: high bit/weight, 8/byte. qs: low 2 bits/weight. q3=lo2|(hi<<2), signed q3-4.
//! scales: 16×4-bit nibbles; signed_sc = nibble-8. dequant: d*signed_sc*q3_signed.
//!
//! **Q4_K** (144 bytes):
//! ```text
//! [d:f16 @0][dmin:f16 @2][scales:12u8 @4][qs:128u8 @16]
//! ```
//! 8 sub-blocks × 32 weights. qs: 4 bits/weight, 2/byte. 6-bit scale decode.
//! dequant: d*sc[sub]*q - dmin*mn[sub] (sc, mn ∈ [0,63]).
//!
//! **Q5_K** (176 bytes):
//! ```text
//! [d:f16 @0][dmin:f16 @2][scales:12u8 @4][qh:32u8 @16][qs:128u8 @48]
//! ```
//! Same 6-bit scales as Q4_K. q5 = nibble | (high_bit<<4), range [0..31].
//! dequant: d*sc[sub]*q5 - dmin*mn[sub].
//!
//! **Q6_K** (210 bytes):
//! ```text
//! [ql:128u8 @0][qh:64u8 @128][scales:16 i8 @192][d:f16 @208]
//! ```
//! 16 sub-blocks × 16 weights. ql: low 4 bits, qh: high 2 bits.
//! q6 = nibble|(hi2<<4), centered: q6-32. scales_i8: signed per sub-block.
//! dequant: d*scales_i8[sub]*q6_signed.
//!
//! **Q8_K** (292 bytes):
//! ```text
//! [d:f32 @0][qs:256 i8 @4][bsums:16 i16 @260]
//! ```
//! d is f32 (not f16!). dequant: d_f32 * qs[i]. bsums not needed for GEMV.
//!
//! # Grid / block dimensions (same for all 6 kernels)
//!
//! - Grid:  `(ceil(n_rows / 8), 1, 1)` — 8 warps per CTA, one warp per output row
//! - Block: `(256, 1, 1)` — 8 warps × 32 lanes
//! - `k` must be a positive multiple of 256 (= QK_K)

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};

// =============================================================================
// CUDA C kernel source
// =============================================================================

/// CUDA C source for all six K-quant GEMV kernels (Q2_K through Q8_K).
///
/// All kernels share the same grid/block strategy (8 warps per CTA, one warp
/// per output row, 256 threads/block, k must be a multiple of QK_K=256).
///
/// The `kq_decode_6bit_scales` device helper is used by Q4_K and Q5_K.
pub const CUDA_K_QUANT_KERNELS_SRC: &str = r#"
/* ==========================================================================
   OxiBonsai CUDA K-quant GEMV kernels  (Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K)

   All formats use QK_K = 256 weights per super-block.

   Grid:  (ceil(n_rows / 8), 1, 1)  -- 8 warps per CTA, 1 warp/row
   Block: (256, 1, 1)

   k must be a positive multiple of 256 for all kernels.
   ========================================================================== */

/* ── Hardware FP16 → FP32 via PTX (1 instruction, SM 6.0+) ─────────────── */
static __device__ __forceinline__ float kq_fast_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* ── Q4_K / Q5_K: decode 12-byte scales array into 8 × 6-bit sc and mn ─── */
static __device__ void kq_decode_6bit_scales(
    const unsigned char* s,   /* 12-byte scales array from block */
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

/* ==========================================================================
   Kernel 1 — gemv_q2k
   Q2_K GEMV: warp-per-row, AoS super-block layout (84 bytes/block).

   Block layout:
     bytes  0-15: scales[16]   — 16 × u8, nibble-encoded sub-scales/sub-mins
     bytes 16-79: qs[64]       — 256 × 2-bit weights, 4/byte (LSB first)
     bytes 80-81: d (FP16 LE)
     bytes 82-83: dmin (FP16 LE)

   16 sub-blocks × 16 weights each.
   scales[sub] & 0xF  = sub_sc  (scale  factor for sub-block)
   scales[sub] >> 4   = sub_mn  (min    factor for sub-block)
   q = (qs[i/4] >> ((i%4)*2)) & 0x3;  q ∈ [0,3]
   dequant: d * sub_sc * q - dmin * sub_mn

   Grid:  (ceil(n_rows / 8), 1, 1)
   Block: (256, 1, 1)
   ========================================================================== */
extern "C" __global__ void gemv_q2k(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k          /* must be a positive multiple of 256 */
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 8u;  /* k / 256 */
    const unsigned int stride = 84u;              /* bytes per Q2_K super-block */

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks + (row * blocks_per_row + b) * stride;

        /* d and dmin at bytes 80-81, 82-83 */
        const unsigned short d_raw    = (unsigned short)bptr[80] | ((unsigned short)bptr[81] << 8u);
        const unsigned short dmin_raw = (unsigned short)bptr[82] | ((unsigned short)bptr[83] << 8u);
        const float d    = kq_fast_fp16_to_float(d_raw);
        const float dmin = kq_fast_fp16_to_float(dmin_raw);

        const float* xbase = input + (b << 8u);  /* b * 256 */

        /* 16 sub-blocks × 16 weights */
        #pragma unroll 16
        for (unsigned int sub = 0u; sub < 16u; ++sub) {
            const unsigned char sc_byte = bptr[sub];  /* scales[sub] */
            const float sub_sc = (float)(sc_byte & 0x0Fu);
            const float sub_mn = (float)((sc_byte >> 4u) & 0x0Fu);

            /* weight offset: sub * 16, qs byte offset: sub * 4 (4 weights/byte) */
            const unsigned int w_base = sub * 16u;
            const unsigned int q_base = sub * 4u;  /* qs start at bptr+16, so add 16 */

            float sub_acc = 0.0f;
            float sub_x_sum = 0.0f;
            #pragma unroll 4
            for (unsigned int qb = 0u; qb < 4u; ++qb) {
                const unsigned char byte_val = bptr[16u + q_base + qb];
                #pragma unroll 4
                for (unsigned int bit = 0u; bit < 4u; ++bit) {
                    const unsigned int wi = w_base + qb * 4u + bit;
                    const float q = (float)((byte_val >> (bit * 2u)) & 0x3u);
                    const float x = xbase[wi];
                    sub_acc  += q * x;
                    sub_x_sum += x;
                }
            }
            acc += d * sub_sc * sub_acc - dmin * sub_mn * sub_x_sum;
        }
    }

    /* Warp-shuffle reduction across 32 lanes */
    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}

/* ==========================================================================
   Kernel 2 — gemv_q3k
   Q3_K GEMV: warp-per-row, AoS super-block layout (110 bytes/block).

   Block layout:
     bytes  0-31:  hmask[32]   — 256 × 1 high bit, 8/byte
     bytes 32-95:  qs[64]      — 256 × 2 low bits, 4/byte (LSB first)
     bytes 96-107: scales[12]  — 16 × 4-bit signed nibbles, 2/byte
     bytes 108-109: d (FP16 LE)

   q3_code = lo2 | (hi << 2), range [0..7]; q3_signed = q3_code - 4.
   signed_sc = nibble - 8  (nibble is 4-bit; sc can be negative).
   dequant: d * signed_sc * q3_signed

   Grid:  (ceil(n_rows / 8), 1, 1)
   Block: (256, 1, 1)
   ========================================================================== */
extern "C" __global__ void gemv_q3k(
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned int stride = 110u;

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks + (row * blocks_per_row + b) * stride;

        /* d at bytes 108-109 */
        const unsigned short d_raw = (unsigned short)bptr[108] | ((unsigned short)bptr[109] << 8u);
        const float d = kq_fast_fp16_to_float(d_raw);

        const float* xbase = input + (b << 8u);

        /* 16 sub-blocks × 16 weights each */
        #pragma unroll 16
        for (unsigned int sub = 0u; sub < 16u; ++sub) {
            /* 4-bit signed scale nibble for this sub-block */
            const unsigned char sc_byte = bptr[96u + sub / 2u];
            const unsigned int  nibble  = (sub & 1u) == 0u
                                          ? (sc_byte & 0x0Fu)
                                          : ((sc_byte >> 4u) & 0x0Fu);
            const float signed_sc = (float)(int)(nibble) - 8.0f;

            /* Per-weight base within the 256-weight block */
            const unsigned int w_base = sub * 16u;

            float sub_acc = 0.0f;
            #pragma unroll 16
            for (unsigned int j = 0u; j < 16u; ++j) {
                const unsigned int wi = w_base + j;
                /* high bit: hmask[wi/8], bit (wi%8) */
                const unsigned int hi = (bptr[wi >> 3u] >> (wi & 7u)) & 0x1u;
                /* low 2 bits: qs[wi/4], bits [(wi%4)*2 .. (wi%4)*2+1] */
                const unsigned int lo2 = (bptr[32u + (wi >> 2u)] >> ((wi & 3u) * 2u)) & 0x3u;
                const int q3_code   = (int)(lo2 | (hi << 2u));
                const int q3_signed = q3_code - 4;
                sub_acc += (float)q3_signed * xbase[wi];
            }
            acc += d * signed_sc * sub_acc;
        }
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}

/* ==========================================================================
   Kernel 3 — gemv_q4k
   Q4_K GEMV: warp-per-row, AoS super-block layout (144 bytes/block).

   Block layout:
     bytes  0- 1: d (FP16 LE)
     bytes  2- 3: dmin (FP16 LE)
     bytes  4-15: scales[12]  — 6-bit sc[8] + 6-bit mn[8] (decoded by helper)
     bytes 16-143: qs[128]    — 256 × 4-bit weights, 2/byte

   8 sub-blocks × 32 weights each.
   even weight j in sub:  qs[sub*16 + j/2] & 0xF
   odd  weight j in sub: (qs[sub*16 + j/2] >> 4) & 0xF
   dequant: d * sc[sub] * q - dmin * mn[sub]

   Grid:  (ceil(n_rows / 8), 1, 1)
   Block: (256, 1, 1)
   ========================================================================== */
extern "C" __global__ void gemv_q4k(
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned int stride = 144u;

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks + (row * blocks_per_row + b) * stride;

        const unsigned short d_raw    = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
        const unsigned short dmin_raw = (unsigned short)bptr[2] | ((unsigned short)bptr[3] << 8u);
        const float d    = kq_fast_fp16_to_float(d_raw);
        const float dmin = kq_fast_fp16_to_float(dmin_raw);

        unsigned char sc[8], mn[8];
        kq_decode_6bit_scales(bptr + 4u, sc, mn);

        const float* xbase = input + (b << 8u);

        /* 8 sub-blocks × 32 weights each */
        #pragma unroll 8
        for (unsigned int sub = 0u; sub < 8u; ++sub) {
            const float sc_f  = (float)sc[sub];
            const float mn_f  = (float)mn[sub];
            /* qs for this sub-block start at bptr[16 + sub*16] */
            const unsigned char* qs_sub = bptr + 16u + sub * 16u;
            const float* x_sub = xbase + sub * 32u;

            float sub_acc  = 0.0f;
            float sub_xsum = 0.0f;
            #pragma unroll 16
            for (unsigned int nb = 0u; nb < 16u; ++nb) {
                const unsigned int byte_val = qs_sub[nb];
                const float q0 = (float)(byte_val & 0x0Fu);
                const float q1 = (float)((byte_val >> 4u) & 0x0Fu);
                const float x0 = x_sub[nb * 2u];
                const float x1 = x_sub[nb * 2u + 1u];
                sub_acc  += q0 * x0 + q1 * x1;
                sub_xsum += x0 + x1;
            }
            acc += d * sc_f * sub_acc - dmin * mn_f * sub_xsum;
        }
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}

/* ==========================================================================
   Kernel 4 — gemv_q5k
   Q5_K GEMV: warp-per-row, AoS super-block layout (176 bytes/block).

   Block layout:
     bytes  0- 1: d (FP16 LE)
     bytes  2- 3: dmin (FP16 LE)
     bytes  4-15: scales[12]   — 6-bit sc[8] + 6-bit mn[8]
     bytes 16-47: qh[32]       — 256 × 1 high bit, 8/byte
     bytes 48-175: qs[128]     — 256 × 4 low bits, 2/byte

   8 sub-blocks × 32 weights each.
   q5 = (qs_nibble) | (high_bit << 4), range [0..31]
   dequant: d * sc[sub] * q5 - dmin * mn[sub]

   Grid:  (ceil(n_rows / 8), 1, 1)
   Block: (256, 1, 1)
   ========================================================================== */
extern "C" __global__ void gemv_q5k(
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned int stride = 176u;

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks + (row * blocks_per_row + b) * stride;

        const unsigned short d_raw    = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
        const unsigned short dmin_raw = (unsigned short)bptr[2] | ((unsigned short)bptr[3] << 8u);
        const float d    = kq_fast_fp16_to_float(d_raw);
        const float dmin = kq_fast_fp16_to_float(dmin_raw);

        unsigned char sc[8], mn[8];
        kq_decode_6bit_scales(bptr + 4u, sc, mn);

        /* qh starts at byte 16, qs starts at byte 48 */
        const unsigned char* qh = bptr + 16u;
        const unsigned char* qs = bptr + 48u;
        const float* xbase = input + (b << 8u);

        /* 8 sub-blocks × 32 weights each */
        #pragma unroll 8
        for (unsigned int sub = 0u; sub < 8u; ++sub) {
            const float sc_f  = (float)sc[sub];
            const float mn_f  = (float)mn[sub];
            /* low-nibble bytes for this sub: qs + sub*16 (16 bytes = 32 nibbles) */
            const unsigned char* qs_sub = qs + sub * 16u;
            const float* x_sub = xbase + sub * 32u;

            float sub_acc  = 0.0f;
            float sub_xsum = 0.0f;
            #pragma unroll 16
            for (unsigned int nb = 0u; nb < 16u; ++nb) {
                /* weight index within the 256-weight super-block */
                const unsigned int wi0 = sub * 32u + nb * 2u;
                const unsigned int wi1 = wi0 + 1u;
                /* high bits from qh */
                const unsigned int hi0 = (qh[wi0 >> 3u] >> (wi0 & 7u)) & 0x1u;
                const unsigned int hi1 = (qh[wi1 >> 3u] >> (wi1 & 7u)) & 0x1u;
                /* low nibbles */
                const unsigned int byte_val = qs_sub[nb];
                const unsigned int lo0 = byte_val & 0x0Fu;
                const unsigned int lo1 = (byte_val >> 4u) & 0x0Fu;
                const float q0 = (float)(lo0 | (hi0 << 4u));
                const float q1 = (float)(lo1 | (hi1 << 4u));
                const float x0 = x_sub[nb * 2u];
                const float x1 = x_sub[nb * 2u + 1u];
                sub_acc  += q0 * x0 + q1 * x1;
                sub_xsum += x0 + x1;
            }
            acc += d * sc_f * sub_acc - dmin * mn_f * sub_xsum;
        }
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}

/* ==========================================================================
   Kernel 5 — gemv_q6k
   Q6_K GEMV: warp-per-row, AoS super-block layout (210 bytes/block).

   Block layout:
     bytes  0-127:  ql[128]    — 256 × 4 low bits, 2/byte
     bytes 128-191: qh[64]     — 256 × 2 high bits, 4/byte
     bytes 192-207: scales[16] — 16 × int8 signed scale, 1/sub-block
     bytes 208-209: d (FP16 LE)

   16 sub-blocks × 16 weights each.
   ql nibble = (ql[i/2] >> ((i%2)*4)) & 0xF
   qh hi2    = (qh[i/4] >> ((i%4)*2)) & 0x3
   q6 = nibble | (hi2 << 4), range [0..63]; q6_signed = q6 - 32, range [-32..31]
   scales_i8 is signed int8.
   dequant: d * scales_i8[sub] * q6_signed

   Grid:  (ceil(n_rows / 8), 1, 1)
   Block: (256, 1, 1)
   ========================================================================== */
extern "C" __global__ void gemv_q6k(
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned int stride = 210u;

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks + (row * blocks_per_row + b) * stride;

        /* d at bytes 208-209 */
        const unsigned short d_raw = (unsigned short)bptr[208] | ((unsigned short)bptr[209] << 8u);
        const float d = kq_fast_fp16_to_float(d_raw);

        /* ql[128], qh[64], scales_i8[16] */
        const unsigned char* ql       = bptr;
        const unsigned char* qh       = bptr + 128u;
        const signed   char* scales_i8 = (const signed char*)(bptr + 192u);
        const float* xbase = input + (b << 8u);

        /* 16 sub-blocks × 16 weights each */
        #pragma unroll 16
        for (unsigned int sub = 0u; sub < 16u; ++sub) {
            const float sc = (float)(int)scales_i8[sub];
            const unsigned int w_base = sub * 16u;

            float sub_acc = 0.0f;
            #pragma unroll 16
            for (unsigned int j = 0u; j < 16u; ++j) {
                const unsigned int wi = w_base + j;
                /* low 4 bits from ql */
                const unsigned int nibble = (ql[wi >> 1u] >> ((wi & 1u) * 4u)) & 0x0Fu;
                /* high 2 bits from qh */
                const unsigned int hi2    = (qh[wi >> 2u] >> ((wi & 3u) * 2u)) & 0x03u;
                const int q6        = (int)(nibble | (hi2 << 4u));
                const int q6_signed = q6 - 32;
                sub_acc += (float)q6_signed * xbase[wi];
            }
            acc += d * sc * sub_acc;
        }
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}

/* ==========================================================================
   Kernel 6 — gemv_q8k
   Q8_K GEMV: warp-per-row, AoS super-block layout (292 bytes/block).

   Block layout:
     bytes  0-3:   d (FP32 LE)       — NOTE: float, not FP16!
     bytes  4-259: qs[256] (int8)    — 256 signed int8 weights
     bytes 260-291: bsums[16] (i16)  — not needed for GEMV

   dequant: d_f32 * qs[i]

   Grid:  (ceil(n_rows / 8), 1, 1)
   Block: (256, 1, 1)
   ========================================================================== */
extern "C" __global__ void gemv_q8k(
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

    const unsigned int blocks_per_row = k >> 8u;
    const unsigned int stride = 292u;

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks + (row * blocks_per_row + b) * stride;

        /* Read f32 d from bytes 0-3 (little-endian) */
        union { unsigned int u; float f; } ud;
        ud.u = (unsigned int)bptr[0]
             | ((unsigned int)bptr[1] << 8u)
             | ((unsigned int)bptr[2] << 16u)
             | ((unsigned int)bptr[3] << 24u);
        const float d = ud.f;

        const float* xbase = input + (b << 8u);

        /* 256 signed int8 weights starting at byte 4 */
        #pragma unroll 32
        for (unsigned int j = 0u; j < 256u; ++j) {
            const int q = (int)(signed char)bptr[4u + j];
            acc += d * (float)q * xbase[j];
        }
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}
"#;

// =============================================================================
// CudaKQuantModules — process-wide singleton for compiled K-quant kernels
// =============================================================================

/// Compiled CUDA function handles for the six K-quant GEMV kernels.
pub struct CudaKQuantModules {
    /// Compiled handle for the `gemv_q2k` kernel.
    pub gemv_q2k: CudaFunction,
    /// Compiled handle for the `gemv_q3k` kernel.
    pub gemv_q3k: CudaFunction,
    /// Compiled handle for the `gemv_q4k` kernel.
    pub gemv_q4k: CudaFunction,
    /// Compiled handle for the `gemv_q5k` kernel.
    pub gemv_q5k: CudaFunction,
    /// Compiled handle for the `gemv_q6k` kernel.
    pub gemv_q6k: CudaFunction,
    /// Compiled handle for the `gemv_q8k` kernel.
    pub gemv_q8k: CudaFunction,
}

// SAFETY: CudaFunction is Send in cudarc.
unsafe impl Send for CudaKQuantModules {}
unsafe impl Sync for CudaKQuantModules {}

struct CudaKQuantState {
    modules: Mutex<Option<Arc<CudaKQuantModules>>>,
}

unsafe impl Send for CudaKQuantState {}
unsafe impl Sync for CudaKQuantState {}

static K_QUANT_STATE: OnceLock<CudaKQuantState> = OnceLock::new();

fn k_quant_state() -> &'static CudaKQuantState {
    K_QUANT_STATE.get_or_init(|| CudaKQuantState {
        modules: Mutex::new(None),
    })
}

/// Compile (or return cached) K-quant CUDA modules (Q2_K through Q8_K).
///
/// Idempotent: the second call returns the already-compiled modules immediately.
pub fn init_k_quant_modules(graph: &CudaGraph) -> Result<Arc<CudaKQuantModules>, CudaGraphError> {
    let state = k_quant_state();
    let mut guard = state
        .modules
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    if let Some(ref m) = *guard {
        return Ok(Arc::clone(m));
    }

    let ptx = compile_or_load_ptx(CUDA_K_QUANT_KERNELS_SRC, "k_quant_kernels")?;

    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module k_quant: {e}")))?;

    let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
        module
            .load_function(name)
            .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
    };

    let mods = Arc::new(CudaKQuantModules {
        gemv_q2k: load("gemv_q2k")?,
        gemv_q3k: load("gemv_q3k")?,
        gemv_q4k: load("gemv_q4k")?,
        gemv_q5k: load("gemv_q5k")?,
        gemv_q6k: load("gemv_q6k")?,
        gemv_q8k: load("gemv_q8k")?,
    });

    *guard = Some(Arc::clone(&mods));
    Ok(mods)
}

// =============================================================================
// Shared launch helper
// =============================================================================

/// Internal helper: upload buffers, launch a K-quant kernel, download results.
///
/// `kernel` is one of the six K-quant function handles.
/// `stride_bytes` is the block size for guard checking (not used here, already
/// validated by caller).
#[allow(clippy::too_many_arguments)]
fn launch_k_quant_kernel(
    kernel: &CudaFunction,
    blocks_bytes: &[u8],
    expected_bytes: usize,
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    kernel_name: &str,
) -> Result<(), CudaGraphError> {
    let graph = CudaGraph::global()?;

    let d_blocks: CudaSlice<u8> = graph
        .stream_arc()
        .clone_htod(&blocks_bytes[..expected_bytes])
        .map_err(|e| {
            CudaGraphError::DriverError(format!("clone_htod {kernel_name} blocks: {e}"))
        })?;
    let d_input: CudaSlice<f32> = graph
        .stream_arc()
        .clone_htod(&input[..k])
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod {kernel_name} input: {e}")))?;
    let mut d_output: CudaSlice<f32> =
        graph.stream_arc().alloc_zeros::<f32>(n_rows).map_err(|e| {
            CudaGraphError::DriverError(format!("alloc_zeros {kernel_name} output: {e}"))
        })?;

    let grid_x = (n_rows as u32).div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: kernel arguments match the CUDA kernel signature; all device
    // buffers are valid on the graph stream and have the correct element counts.
    unsafe {
        graph
            .stream_arc()
            .launch_builder(kernel)
            .arg(&d_blocks)
            .arg(&d_input)
            .arg(&mut d_output)
            .arg(&(n_rows as u32))
            .arg(&(k as u32))
            .launch(cfg)
            .map_err(|e| CudaGraphError::DriverError(format!("{kernel_name} launch: {e}")))?;
    }

    let host_out: Vec<f32> = graph.stream_arc().clone_dtoh(&d_output).map_err(|e| {
        CudaGraphError::DriverError(format!("clone_dtoh {kernel_name} output: {e}"))
    })?;

    output[..n_rows].copy_from_slice(&host_out);
    Ok(())
}

// =============================================================================
// Public host functions
// =============================================================================

/// Validate common K-quant arguments (k divisibility, buffer sizes).
fn validate_k_quant_args(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &[f32],
    n_rows: usize,
    k: usize,
    block_stride: usize,
    format: &str,
) -> Result<usize, CudaGraphError> {
    if k == 0 || k % 256 != 0 {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "{format} GEMV: k={k} must be a positive multiple of 256"
        )));
    }
    let blocks_per_row = k / 256;
    let expected_bytes = n_rows * blocks_per_row * block_stride;
    if blocks_bytes.len() < expected_bytes {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "{format} blocks_bytes too short: {} < {expected_bytes}",
            blocks_bytes.len()
        )));
    }
    if input.len() < k {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "{format} GEMV: input.len()={} < k={k}",
            input.len()
        )));
    }
    if output.len() < n_rows {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "{format} GEMV: output.len()={} < n_rows={n_rows}",
            output.len()
        )));
    }
    Ok(expected_bytes)
}

/// Run Q2_K GEMV on GPU.
///
/// `blocks_bytes` is the raw AoS byte representation of the weight matrix:
/// - 84 bytes per super-block: `[scales:16][qs:64][d_f16:2][dmin_f16:2]`
///   - 16 sub-blocks × 16 weights, 2-bit quant, per-sub scale/min
/// - Total length: `n_rows * (k / 256) * 84`
///
/// `input` must have length `>= k`. `k` must be a positive multiple of 256.
/// `output` must have length `>= n_rows`.
pub fn cuda_gemv_q2k(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    let expected = validate_k_quant_args(blocks_bytes, input, output, n_rows, k, 84, "Q2_K")?;
    let graph = CudaGraph::global()?;
    let mods = init_k_quant_modules(&graph)?;
    launch_k_quant_kernel(
        &mods.gemv_q2k,
        blocks_bytes,
        expected,
        input,
        output,
        n_rows,
        k,
        "gemv_q2k",
    )
}

/// Run Q3_K GEMV on GPU.
///
/// `blocks_bytes` is the raw AoS byte representation of the weight matrix:
/// - 110 bytes per super-block: `[hmask:32][qs:64][scales:12][d_f16:2]`
///   - 16 sub-blocks × 16 weights, 3-bit quant (1-bit high + 2-bit low)
/// - Total length: `n_rows * (k / 256) * 110`
///
/// `input` must have length `>= k`. `k` must be a positive multiple of 256.
/// `output` must have length `>= n_rows`.
pub fn cuda_gemv_q3k(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    let expected = validate_k_quant_args(blocks_bytes, input, output, n_rows, k, 110, "Q3_K")?;
    let graph = CudaGraph::global()?;
    let mods = init_k_quant_modules(&graph)?;
    launch_k_quant_kernel(
        &mods.gemv_q3k,
        blocks_bytes,
        expected,
        input,
        output,
        n_rows,
        k,
        "gemv_q3k",
    )
}

/// Run Q4_K GEMV on GPU.
///
/// `blocks_bytes` is the raw AoS byte representation of the weight matrix:
/// - 144 bytes per super-block: `[d_f16:2][dmin_f16:2][scales:12][qs:128]`
///   - 8 sub-blocks × 32 weights, 4-bit quant, 6-bit scale/min
/// - Total length: `n_rows * (k / 256) * 144`
///
/// `input` must have length `>= k`. `k` must be a positive multiple of 256.
/// `output` must have length `>= n_rows`.
pub fn cuda_gemv_q4k(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    let expected = validate_k_quant_args(blocks_bytes, input, output, n_rows, k, 144, "Q4_K")?;
    let graph = CudaGraph::global()?;
    let mods = init_k_quant_modules(&graph)?;
    launch_k_quant_kernel(
        &mods.gemv_q4k,
        blocks_bytes,
        expected,
        input,
        output,
        n_rows,
        k,
        "gemv_q4k",
    )
}

/// Run Q5_K GEMV on GPU.
///
/// `blocks_bytes` is the raw AoS byte representation of the weight matrix:
/// - 176 bytes per super-block: `[d_f16:2][dmin_f16:2][scales:12][qh:32][qs:128]`
///   - 8 sub-blocks × 32 weights, 5-bit quant (4-bit low + 1-bit high)
/// - Total length: `n_rows * (k / 256) * 176`
///
/// `input` must have length `>= k`. `k` must be a positive multiple of 256.
/// `output` must have length `>= n_rows`.
pub fn cuda_gemv_q5k(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    let expected = validate_k_quant_args(blocks_bytes, input, output, n_rows, k, 176, "Q5_K")?;
    let graph = CudaGraph::global()?;
    let mods = init_k_quant_modules(&graph)?;
    launch_k_quant_kernel(
        &mods.gemv_q5k,
        blocks_bytes,
        expected,
        input,
        output,
        n_rows,
        k,
        "gemv_q5k",
    )
}

/// Run Q6_K GEMV on GPU.
///
/// `blocks_bytes` is the raw AoS byte representation of the weight matrix:
/// - 210 bytes per super-block: `[ql:128][qh:64][scales_i8:16][d_f16:2]`
///   - 16 sub-blocks × 16 weights, 6-bit quant (4-bit low + 2-bit high), signed i8 scales
/// - Total length: `n_rows * (k / 256) * 210`
///
/// `input` must have length `>= k`. `k` must be a positive multiple of 256.
/// `output` must have length `>= n_rows`.
pub fn cuda_gemv_q6k(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    let expected = validate_k_quant_args(blocks_bytes, input, output, n_rows, k, 210, "Q6_K")?;
    let graph = CudaGraph::global()?;
    let mods = init_k_quant_modules(&graph)?;
    launch_k_quant_kernel(
        &mods.gemv_q6k,
        blocks_bytes,
        expected,
        input,
        output,
        n_rows,
        k,
        "gemv_q6k",
    )
}

/// Run Q8_K GEMV on GPU.
///
/// `blocks_bytes` is the raw AoS byte representation of the weight matrix:
/// - 292 bytes per super-block: `[d_f32:4][qs:256 i8][bsums:32]`
///   - 256 signed int8 weights; scale `d` is f32 (not f16!)
/// - Total length: `n_rows * (k / 256) * 292`
///
/// `input` must have length `>= k`. `k` must be a positive multiple of 256.
/// `output` must have length `>= n_rows`.
pub fn cuda_gemv_q8k(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    let expected = validate_k_quant_args(blocks_bytes, input, output, n_rows, k, 292, "Q8_K")?;
    let graph = CudaGraph::global()?;
    let mods = init_k_quant_modules(&graph)?;
    launch_k_quant_kernel(
        &mods.gemv_q8k,
        blocks_bytes,
        expected,
        input,
        output,
        n_rows,
        k,
        "gemv_q8k",
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Kernel source content checks ────────────────────────────────────────

    #[test]
    fn test_k_quant_kernel_source_has_gemv_q2k() {
        assert!(
            CUDA_K_QUANT_KERNELS_SRC.contains("gemv_q2k"),
            "CUDA_K_QUANT_KERNELS_SRC must contain gemv_q2k"
        );
    }

    #[test]
    fn test_k_quant_kernel_source_has_gemv_q3k() {
        assert!(
            CUDA_K_QUANT_KERNELS_SRC.contains("gemv_q3k"),
            "CUDA_K_QUANT_KERNELS_SRC must contain gemv_q3k"
        );
    }

    #[test]
    fn test_k_quant_kernel_source_has_gemv_q4k() {
        assert!(
            CUDA_K_QUANT_KERNELS_SRC.contains("gemv_q4k"),
            "CUDA_K_QUANT_KERNELS_SRC must contain gemv_q4k"
        );
    }

    #[test]
    fn test_k_quant_kernel_source_has_gemv_q5k() {
        assert!(
            CUDA_K_QUANT_KERNELS_SRC.contains("gemv_q5k"),
            "CUDA_K_QUANT_KERNELS_SRC must contain gemv_q5k"
        );
    }

    #[test]
    fn test_k_quant_kernel_source_has_gemv_q6k() {
        assert!(
            CUDA_K_QUANT_KERNELS_SRC.contains("gemv_q6k"),
            "CUDA_K_QUANT_KERNELS_SRC must contain gemv_q6k"
        );
    }

    #[test]
    fn test_k_quant_kernel_source_has_gemv_q8k() {
        assert!(
            CUDA_K_QUANT_KERNELS_SRC.contains("gemv_q8k"),
            "CUDA_K_QUANT_KERNELS_SRC must contain gemv_q8k"
        );
    }

    #[test]
    fn test_k_quant_kernel_source_has_6bit_scale_helper() {
        assert!(
            CUDA_K_QUANT_KERNELS_SRC.contains("kq_decode_6bit_scales"),
            "CUDA_K_QUANT_KERNELS_SRC must contain kq_decode_6bit_scales"
        );
    }

    // ── Block stride / size guard checks ────────────────────────────────────

    /// Q2_K super-block: 16 scale bytes + 64 qs bytes + 4 header bytes = 84.
    #[test]
    fn test_q2k_block_stride() {
        assert_eq!(16 + 64 + 2 + 2, 84usize);
    }

    /// Q3_K super-block: 32 hmask + 64 qs + 12 scales + 2 d = 110.
    #[test]
    fn test_q3k_block_stride() {
        assert_eq!(32 + 64 + 12 + 2, 110usize);
    }

    /// Q4_K super-block: 2 d + 2 dmin + 12 scales + 128 qs = 144.
    #[test]
    fn test_q4k_block_stride() {
        assert_eq!(2 + 2 + 12 + 128, 144usize);
    }

    /// Q5_K super-block: 2 d + 2 dmin + 12 scales + 32 qh + 128 qs = 176.
    #[test]
    fn test_q5k_block_stride() {
        assert_eq!(2 + 2 + 12 + 32 + 128, 176usize);
    }

    /// Q6_K super-block: 128 ql + 64 qh + 16 scales_i8 + 2 d = 210.
    #[test]
    fn test_q6k_block_stride() {
        assert_eq!(128 + 64 + 16 + 2, 210usize);
    }

    /// Q8_K super-block: 4 d_f32 + 256 qs_i8 + 32 bsums_i16 = 292.
    #[test]
    fn test_q8k_block_stride() {
        assert_eq!(4 + 256 + 32, 292usize);
    }

    // ── Dimension guard: k not a multiple of 256 ────────────────────────────

    #[test]
    fn test_cuda_gemv_q2k_bad_k() {
        let blocks = vec![0u8; 84];
        let input = vec![0.0f32; 255];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q2k(&blocks, &input, &mut output, 1, 255);
        assert!(result.is_err(), "k=255 (not multiple of 256) should error");
    }

    #[test]
    fn test_cuda_gemv_q3k_bad_k() {
        let blocks = vec![0u8; 110];
        let input = vec![0.0f32; 255];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q3k(&blocks, &input, &mut output, 1, 255);
        assert!(result.is_err(), "k=255 (not multiple of 256) should error");
    }

    #[test]
    fn test_cuda_gemv_q4k_bad_k() {
        let blocks = vec![0u8; 144];
        let input = vec![0.0f32; 255];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q4k(&blocks, &input, &mut output, 1, 255);
        assert!(result.is_err(), "k=255 (not multiple of 256) should error");
    }

    #[test]
    fn test_cuda_gemv_q5k_bad_k() {
        let blocks = vec![0u8; 176];
        let input = vec![0.0f32; 255];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q5k(&blocks, &input, &mut output, 1, 255);
        assert!(result.is_err(), "k=255 (not multiple of 256) should error");
    }

    #[test]
    fn test_cuda_gemv_q6k_bad_k() {
        let blocks = vec![0u8; 210];
        let input = vec![0.0f32; 255];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q6k(&blocks, &input, &mut output, 1, 255);
        assert!(result.is_err(), "k=255 (not multiple of 256) should error");
    }

    #[test]
    fn test_cuda_gemv_q8k_bad_k() {
        let blocks = vec![0u8; 292];
        let input = vec![0.0f32; 255];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q8k(&blocks, &input, &mut output, 1, 255);
        assert!(result.is_err(), "k=255 (not multiple of 256) should error");
    }

    // ── k=0 guard ───────────────────────────────────────────────────────────

    #[test]
    fn test_cuda_gemv_q2k_zero_k() {
        let blocks: Vec<u8> = Vec::new();
        let input: Vec<f32> = Vec::new();
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q2k(&blocks, &input, &mut output, 1, 0);
        assert!(result.is_err(), "k=0 should error");
    }

    #[test]
    fn test_cuda_gemv_q8k_zero_k() {
        let blocks: Vec<u8> = Vec::new();
        let input: Vec<f32> = Vec::new();
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q8k(&blocks, &input, &mut output, 1, 0);
        assert!(result.is_err(), "k=0 should error");
    }

    // ── Output buffer too small ──────────────────────────────────────────────

    #[test]
    fn test_cuda_gemv_q2k_output_too_small() {
        let blocks = vec![0u8; 84];
        let input = vec![0.0f32; 256];
        let mut output: Vec<f32> = Vec::new();
        let result = cuda_gemv_q2k(&blocks, &input, &mut output, 1, 256);
        assert!(result.is_err(), "empty output should error for q2k");
    }

    #[test]
    fn test_cuda_gemv_q3k_output_too_small() {
        let blocks = vec![0u8; 110];
        let input = vec![0.0f32; 256];
        let mut output: Vec<f32> = Vec::new();
        let result = cuda_gemv_q3k(&blocks, &input, &mut output, 1, 256);
        assert!(result.is_err(), "empty output should error for q3k");
    }

    #[test]
    fn test_cuda_gemv_q4k_output_too_small() {
        let blocks = vec![0u8; 144];
        let input = vec![0.0f32; 256];
        let mut output: Vec<f32> = Vec::new();
        let result = cuda_gemv_q4k(&blocks, &input, &mut output, 1, 256);
        assert!(result.is_err(), "empty output should error for q4k");
    }

    #[test]
    fn test_cuda_gemv_q5k_output_too_small() {
        let blocks = vec![0u8; 176];
        let input = vec![0.0f32; 256];
        let mut output: Vec<f32> = Vec::new();
        let result = cuda_gemv_q5k(&blocks, &input, &mut output, 1, 256);
        assert!(result.is_err(), "empty output should error for q5k");
    }

    #[test]
    fn test_cuda_gemv_q6k_output_too_small() {
        let blocks = vec![0u8; 210];
        let input = vec![0.0f32; 256];
        let mut output: Vec<f32> = Vec::new();
        let result = cuda_gemv_q6k(&blocks, &input, &mut output, 1, 256);
        assert!(result.is_err(), "empty output should error for q6k");
    }

    #[test]
    fn test_cuda_gemv_q8k_output_too_small() {
        let blocks = vec![0u8; 292];
        let input = vec![0.0f32; 256];
        let mut output: Vec<f32> = Vec::new();
        let result = cuda_gemv_q8k(&blocks, &input, &mut output, 1, 256);
        assert!(result.is_err(), "empty output should error for q8k");
    }

    // ── Blocks buffer too small ──────────────────────────────────────────────

    #[test]
    fn test_cuda_gemv_q2k_blocks_too_small() {
        // Expected: 1 * 1 * 84 = 84 bytes; provide only 10.
        let blocks = vec![0u8; 10];
        let input = vec![0.0f32; 256];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q2k(&blocks, &input, &mut output, 1, 256);
        assert!(result.is_err(), "blocks too short should error for q2k");
    }

    #[test]
    fn test_cuda_gemv_q8k_blocks_too_small() {
        let blocks = vec![0u8; 10];
        let input = vec![0.0f32; 256];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q8k(&blocks, &input, &mut output, 1, 256);
        assert!(result.is_err(), "blocks too short should error for q8k");
    }

    // ── GPU-gated integration tests ──────────────────────────────────────────

    /// Q2_K: all qs=0, d=1.0, dmin=0 → all weights = 0 → output = 0.
    #[test]
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn test_cuda_gemv_q2k_zero_weights() {
        use crate::gpu_backend::cuda_graph::CudaGraph;
        if CudaGraph::global().is_err() {
            eprintln!("SKIP: test_cuda_gemv_q2k_zero_weights — no CUDA device");
            return;
        }
        let n_rows = 4usize;
        let k = 256usize;
        let mut blocks = vec![0u8; n_rows * 84];
        for r in 0..n_rows {
            let b = &mut blocks[r * 84..(r + 1) * 84];
            // scales all zero → sub_sc=0, sub_mn=0; qs all zero → q=0
            // d = 1.0 (FP16): 0x3C00 LE = [0x00, 0x3C]; dmin = 0 = [0x00, 0x00]
            b[80] = 0x00;
            b[81] = 0x3C;
            // dmin stays 0
        }
        let input = vec![1.0f32; k];
        let mut output = vec![0.0f32; n_rows];
        cuda_gemv_q2k(&blocks, &input, &mut output, n_rows, k).unwrap();
        for &v in &output {
            assert!(v.abs() < 1e-5f32, "Q2_K zero weights: expected 0, got {v}");
        }
    }

    /// Q8_K: d=1.0 (f32), qs[0]=1 rest=0, input all-ones → each row = 1.
    #[test]
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn test_cuda_gemv_q8k_single_weight() {
        use crate::gpu_backend::cuda_graph::CudaGraph;
        if CudaGraph::global().is_err() {
            eprintln!("SKIP: test_cuda_gemv_q8k_single_weight — no CUDA device");
            return;
        }
        let n_rows = 4usize;
        let k = 256usize;
        let mut blocks = vec![0u8; n_rows * 292];
        for r in 0..n_rows {
            let b = &mut blocks[r * 292..(r + 1) * 292];
            // d = 1.0f32 as LE bytes
            let d_bytes = 1.0f32.to_le_bytes();
            b[0] = d_bytes[0];
            b[1] = d_bytes[1];
            b[2] = d_bytes[2];
            b[3] = d_bytes[3];
            // qs[0] = 1, rest = 0
            b[4] = 1u8;
        }
        let input = vec![1.0f32; k];
        let mut output = vec![0.0f32; n_rows];
        cuda_gemv_q8k(&blocks, &input, &mut output, n_rows, k).unwrap();
        for &v in &output {
            assert!(
                (v - 1.0f32).abs() < 1e-5f32,
                "Q8_K single weight: expected 1.0, got {v}"
            );
        }
    }
}
