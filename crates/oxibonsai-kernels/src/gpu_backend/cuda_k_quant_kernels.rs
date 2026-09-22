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
//! # The ggml walk is load-bearing
//!
//! The per-format `kq_dot_q*k` device helpers are fused transliterations of
//! ggml's `dequantize_row_q*_K` (`ggml/src/ggml-quants.c`). ggml's output
//! cursor is decoupled from its byte cursors, so the *n*-th value it emits —
//! the one that multiplies `x[n]` — is generally **not** read from byte lane
//! *n*. The byte-exact CPU reference (`oxibonsai_core::BlockQ*K::dequant`) is
//! the normative source for these loops.
//!
//! # Block layouts (QK_K = 256 weights per super-block)
//!
//! **Q2_K** (84 bytes):
//! ```text
//! [scales:16u8][qs:64u8][d:f16 @80][dmin:f16 @82]
//! ```
//! Per 128-element half, `shift` steps 0, 2, 4, 6 and `is` advances twice per
//! step: output `n*128 + j*32 + l` reads `qs[n*32 + l] >> 2j` under
//! `scales[is]`, output `n*128 + j*32 + 16 + l` reads `qs[n*32 + 16 + l] >> 2j`
//! under `scales[is + 1]`. sc = scales\[is\] & 0xF, mn = scales\[is\] >> 4.
//! dequant: d*sc*q - dmin*mn (q ∈ \[0,3\]).
//!
//! **Q3_K** (110 bytes):
//! ```text
//! [hmask:32u8][qs:64u8][scales:12u8][d:f16 @108]
//! ```
//! Same 128-half / `shift` / `is` walk as Q2_K. The 16 six-bit scales come from
//! the `kmask1`/`kmask2` `aux[4]` shuffle and carry a **-32** bias (they are not
//! 4-bit nibbles). hmask is **inverted**: `q - (hm[l] & m ? 0 : 4)`, with the
//! same 32 hmask bytes reused across all eight `m = 1 << (4n + j)` steps.
//! dequant: d*(sc-32)*(q - hi).
//!
//! **Q4_K** (144 bytes):
//! ```text
//! [d:f16 @0][dmin:f16 @2][scales:12u8 @4][qs:128u8 @16]
//! ```
//! Per 64-element group: 32 **low** nibbles under `get_scale_min_k4(is)` then 32
//! **high** nibbles under `get_scale_min_k4(is + 1)`, `is += 2` per group. For
//! `j < 4` the 6-bit scale is a full byte (`q[j] & 63`), not two nibbles.
//! dequant: d*sc*q - dmin*mn (sc, mn ∈ \[0,63\]).
//!
//! **Q5_K** (176 bytes):
//! ```text
//! [d:f16 @0][dmin:f16 @2][scales:12u8 @4][qh:32u8 @16][qs:128u8 @48]
//! ```
//! Q4_K's walk plus the 5th bit: `qh[l]` is indexed `l in 0..32` and does **not**
//! advance per group; the masks `u1`/`u2` shift left by 2 per group.
//! dequant: d*sc*(q + 16*qh_bit) - dmin*mn.
//!
//! **Q6_K** (210 bytes):
//! ```text
//! [ql:128u8 @0][qh:64u8 @128][scales:16 i8 @192][d:f16 @208]
//! ```
//! Per 128-element half and `l in 0..32`, four interleaved lanes `y[l]`,
//! `y[l+32]`, `y[l+64]`, `y[l+96]` with sub-scales `sc[is+0,+2,+4,+6]`,
//! `is = l/16`. q6 = nibble|(hi2<<4), centered: q6-32.
//! dequant: d*scales_i8*(q6-32).
//!
//! **Q8_K** (292 bytes):
//! ```text
//! [d:f32 @0][qs:256 i8 @4][bsums:16 i16 @260]
//! ```
//! d is f32 (not f16!). Element-sequential in ggml too. dequant: d_f32 * qs\[i\].
//! bsums not needed for GEMV.
//!
//! # Grid / block dimensions (same for all 6 kernels)
//!
//! - Grid:  `(ceil(n_rows / 8), 1, 1)` — 8 warps per CTA, one warp per output row
//! - Block: `(256, 1, 1)` — 8 warps × 32 lanes
//! - `k` must be a positive multiple of 256 (= QK_K)
//! - One lane decodes one whole super-block: ggml's `is` / `m` / `u1` / `u2`
//!   cursors are sequential state and must not be split across lanes.

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
/// # The ggml walk is load-bearing
///
/// Each format has a `kq_dot_q*k` device helper that is a **fused
/// transliteration of ggml's `dequantize_row_q*_K`** (`ggml/src/ggml-quants.c`)
/// against a 256-element input slice. ggml's output cursor is decoupled from
/// its byte cursors, so the *n*-th value it emits — the value that multiplies
/// `x[n]` — is generally **not** read from byte lane *n*. The byte-exact CPU
/// reference (`oxibonsai_core::BlockQ*K::dequant`) is the normative source for
/// these loops; the helpers reproduce them cursor for cursor. One warp lane
/// decodes one whole super-block: the `is` / `m` / `u1` / `u2` cursors are
/// sequential state and must not be split across lanes.
///
/// The `kq_scale_min_k4` device helper (ggml's `get_scale_min_k4`) is used by
/// Q4_K and Q5_K.
pub const CUDA_K_QUANT_KERNELS_SRC: &str = r#"
/* ==========================================================================
   OxiBonsai CUDA K-quant GEMV kernels  (Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K)

   All formats use QK_K = 256 weights per super-block.

   Grid:  (ceil(n_rows / 8), 1, 1)  -- 8 warps per CTA, 1 warp/row
   Block: (256, 1, 1)

   k must be a positive multiple of 256 for all kernels.

   Every kq_dot_q*k helper below mirrors ggml's dequantize_row_q*_K walk
   (ggml/src/ggml-quants.c) element for element, fused with the dot product:
   the output cursor y indexes the input, the byte cursors index the block.
   ========================================================================== */

/* ── Hardware FP16 → FP32 via PTX (1 instruction, SM 6.0+) ─────────────── */
static __device__ __forceinline__ float kq_fast_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* ── ggml get_scale_min_k4 (ggml-quants.c:935) ─────────────────────────────
   Sub-block j's 6-bit scale and 6-bit min out of the 12-byte packed `scales`
   array shared by Q4_K and Q5_K. For j < 4 the scale is a FULL 6-bit value
   read from byte j -- not a 4-bit nibble.                                   */
static __device__ __forceinline__ void kq_scale_min_k4(
    const unsigned char* __restrict__ q,
    unsigned int j,
    unsigned char* sc_out,
    unsigned char* m_out
) {
    if (j < 4u) {
        *sc_out = (unsigned char)(q[j] & 63u);
        *m_out  = (unsigned char)(q[j + 4u] & 63u);
    } else {
        *sc_out = (unsigned char)((q[j + 4u] & 0x0Fu) | (((unsigned int)q[j - 4u] >> 6u) << 4u));
        *m_out  = (unsigned char)(((unsigned int)q[j + 4u] >> 4u)
                                  | (((unsigned int)q[j] >> 6u) << 4u));
    }
}

/* ── Q3_K: unpack the 12 packed scale bytes into 16 biased 6-bit scales ───
   ggml-quants.c:1374-1381 (kmask1 / kmask2 aux[4] shuffle). The +32 bias is
   removed by the caller.                                                    */
static __device__ void kq_unpack_q3k_scales(
    const unsigned char* __restrict__ s,
    unsigned char* out
) {
    const unsigned int kmask1 = 0x03030303u;
    const unsigned int kmask2 = 0x0f0f0f0fu;
    const unsigned int a0_in = (unsigned int)s[0] | ((unsigned int)s[1] << 8u)
                             | ((unsigned int)s[2] << 16u) | ((unsigned int)s[3] << 24u);
    const unsigned int a1_in = (unsigned int)s[4] | ((unsigned int)s[5] << 8u)
                             | ((unsigned int)s[6] << 16u) | ((unsigned int)s[7] << 24u);
    const unsigned int tmp   = (unsigned int)s[8] | ((unsigned int)s[9] << 8u)
                             | ((unsigned int)s[10] << 16u) | ((unsigned int)s[11] << 24u);

    unsigned int aux[4];
    aux[2] = ((a0_in >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
    aux[3] = ((a1_in >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
    aux[0] = (a0_in & kmask2) | ((tmp & kmask1) << 4u);
    aux[1] = (a1_in & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

    #pragma unroll
    for (unsigned int w = 0u; w < 4u; ++w) {
        out[4u * w + 0u] = (unsigned char)(aux[w] & 0xFFu);
        out[4u * w + 1u] = (unsigned char)((aux[w] >> 8u) & 0xFFu);
        out[4u * w + 2u] = (unsigned char)((aux[w] >> 16u) & 0xFFu);
        out[4u * w + 3u] = (unsigned char)((aux[w] >> 24u) & 0xFFu);
    }
}

/* ── Q2_K super-block · dot product ────────────────────────────────────────
   Block: [scales:16 @0][qs:64 @16][d:f16 @80][dmin:f16 @82]  (84 bytes)
   dequantize_row_q2_K (ggml-quants.c:1016): per 128-element half, `shift`
   steps 0,2,4,6 and `is` advances twice per step -- output n*128 + j*32 + l
   reads qs[n*32 + l] >> 2j under scales[is], output n*128 + j*32 + 16 + l
   reads qs[n*32 + 16 + l] >> 2j under scales[is + 1].
   dequant: d*(sc & 0xF)*q - dmin*(sc >> 4)                                  */
static __device__ float kq_dot_q2k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    const unsigned short d_raw    = (unsigned short)bptr[80] | ((unsigned short)bptr[81] << 8u);
    const unsigned short dmin_raw = (unsigned short)bptr[82] | ((unsigned short)bptr[83] << 8u);
    const float d    = kq_fast_fp16_to_float(d_raw);
    const float dmin = kq_fast_fp16_to_float(dmin_raw);
    const unsigned char* qs = bptr + 16u;

    float acc = 0.0f;
    unsigned int y = 0u;      /* output cursor */
    unsigned int is = 0u;     /* scales cursor */
    unsigned int q_off = 0u;  /* qs byte cursor */

    for (unsigned int n = 0u; n < 2u; ++n) {        /* QK_K / 128 */
        unsigned int shift = 0u;
        for (unsigned int j = 0u; j < 4u; ++j) {
            for (unsigned int part = 0u; part < 2u; ++part) {
                const unsigned int sc = bptr[is];
                ++is;
                const float dl = d * (float)(sc & 0x0Fu);
                const float ml = dmin * (float)(sc >> 4u);
                const unsigned char* qp = qs + q_off + part * 16u;
                float qsum = 0.0f;
                float xsum = 0.0f;
                #pragma unroll 16
                for (unsigned int l = 0u; l < 16u; ++l) {
                    const float xv = x[y + l];
                    qsum += (float)((qp[l] >> shift) & 3u) * xv;
                    xsum += xv;
                }
                acc += dl * qsum - ml * xsum;
                y += 16u;
            }
            shift += 2u;
        }
        q_off += 32u;
    }
    return acc;
}

/* ── Q3_K super-block · dot product ────────────────────────────────────────
   Block: [hmask:32 @0][qs:64 @32][scales:12 @96][d:f16 @108]  (110 bytes)
   dequantize_row_q3_K (ggml-quants.c:1360): the Q2_K walk, 6-bit scales from
   the kmask shuffle biased by -32, and the INVERTED hmask convention
   (q - (hm[l] & m ? 0 : 4)) with m = 1 << (4n + j) reusing the same 32 hmask
   bytes across all eight steps.
   dequant: d*(sc - 32)*(q - hi)                                             */
static __device__ float kq_dot_q3k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    const unsigned short d_raw = (unsigned short)bptr[108] | ((unsigned short)bptr[109] << 8u);
    const float d_all = kq_fast_fp16_to_float(d_raw);
    const unsigned char* hm = bptr;        /* hmask[32] */
    const unsigned char* qs = bptr + 32u;  /* qs[64]    */

    unsigned char sc[16];
    kq_unpack_q3k_scales(bptr + 96u, sc);

    float acc = 0.0f;
    unsigned int y = 0u;
    unsigned int is = 0u;
    unsigned int q_off = 0u;
    unsigned int m = 1u;

    for (unsigned int n = 0u; n < 2u; ++n) {
        unsigned int shift = 0u;
        for (unsigned int j = 0u; j < 4u; ++j) {
            for (unsigned int part = 0u; part < 2u; ++part) {
                const float dl = d_all * (float)((int)sc[is] - 32);
                ++is;
                const unsigned char* qp = qs + q_off + part * 16u;
                const unsigned char* hp = hm + part * 16u;
                float qsum = 0.0f;
                #pragma unroll 16
                for (unsigned int l = 0u; l < 16u; ++l) {
                    const int hi = ((unsigned int)hp[l] & m) != 0u ? 0 : 4;
                    const int q  = (int)((qp[l] >> shift) & 3u);
                    qsum += (float)(q - hi) * x[y + l];
                }
                acc += dl * qsum;
                y += 16u;
            }
            shift += 2u;
            m <<= 1u;
        }
        q_off += 32u;
    }
    return acc;
}

/* ── Q4_K super-block · dot product ────────────────────────────────────────
   Block: [d:f16 @0][dmin:f16 @2][scales:12 @4][qs:128 @16]  (144 bytes)
   dequantize_row_q4_K (ggml-quants.c:1584): per 64-element group, 32 LOW
   nibbles under get_scale_min_k4(is) then 32 HIGH nibbles under
   get_scale_min_k4(is + 1); is += 2 per group.
   dequant: d*sc*q - dmin*mn                                                 */
static __device__ float kq_dot_q4k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    const unsigned short d_raw    = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
    const unsigned short dmin_raw = (unsigned short)bptr[2] | ((unsigned short)bptr[3] << 8u);
    const float d    = kq_fast_fp16_to_float(d_raw);
    const float dmin = kq_fast_fp16_to_float(dmin_raw);
    const unsigned char* scales = bptr + 4u;
    const unsigned char* qs     = bptr + 16u;

    float acc = 0.0f;
    unsigned int y = 0u;
    unsigned int q_off = 0u;
    unsigned int is = 0u;

    for (unsigned int g = 0u; g < 4u; ++g) {   /* (0..256).step_by(64) */
        unsigned char sc1, mn1, sc2, mn2;
        kq_scale_min_k4(scales, is, &sc1, &mn1);
        kq_scale_min_k4(scales, is + 1u, &sc2, &mn2);
        const float d1 = d * (float)sc1;
        const float m1 = dmin * (float)mn1;
        const float d2 = d * (float)sc2;
        const float m2 = dmin * (float)mn2;

        float qsum = 0.0f;
        float xsum = 0.0f;
        #pragma unroll 32
        for (unsigned int l = 0u; l < 32u; ++l) {          /* 32 LOW nibbles */
            const float xv = x[y + l];
            qsum += (float)(qs[q_off + l] & 0x0Fu) * xv;
            xsum += xv;
        }
        acc += d1 * qsum - m1 * xsum;
        y += 32u;

        qsum = 0.0f;
        xsum = 0.0f;
        #pragma unroll 32
        for (unsigned int l = 0u; l < 32u; ++l) {          /* then 32 HIGH   */
            const float xv = x[y + l];
            qsum += (float)((unsigned int)qs[q_off + l] >> 4u) * xv;
            xsum += xv;
        }
        acc += d2 * qsum - m2 * xsum;
        y += 32u;

        q_off += 32u;
        is += 2u;
    }
    return acc;
}

/* ── Q5_K super-block · dot product ────────────────────────────────────────
   Block: [d:f16 @0][dmin:f16 @2][scales:12 @4][qh:32 @16][qs:128 @48] (176 B)
   dequantize_row_q5_K (ggml-quants.c:1786): Q4_K's walk plus the 5th bit --
   qh[l] is indexed l in 0..32 and does NOT advance per group; the masks
   u1 / u2 shift left by 2 per 64-element group.
   dequant: d*sc*(q + 16*qh_bit) - dmin*mn                                   */
static __device__ float kq_dot_q5k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    const unsigned short d_raw    = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
    const unsigned short dmin_raw = (unsigned short)bptr[2] | ((unsigned short)bptr[3] << 8u);
    const float d    = kq_fast_fp16_to_float(d_raw);
    const float dmin = kq_fast_fp16_to_float(dmin_raw);
    const unsigned char* scales = bptr + 4u;
    const unsigned char* qh     = bptr + 16u;   /* 32 bytes, not advanced */
    const unsigned char* ql     = bptr + 48u;

    float acc = 0.0f;
    unsigned int y = 0u;
    unsigned int ql_off = 0u;
    unsigned int is = 0u;
    unsigned int u1 = 1u;
    unsigned int u2 = 2u;

    for (unsigned int g = 0u; g < 4u; ++g) {   /* (0..256).step_by(64) */
        unsigned char sc1, mn1, sc2, mn2;
        kq_scale_min_k4(scales, is, &sc1, &mn1);
        kq_scale_min_k4(scales, is + 1u, &sc2, &mn2);
        const float d1 = d * (float)sc1;
        const float m1 = dmin * (float)mn1;
        const float d2 = d * (float)sc2;
        const float m2 = dmin * (float)mn2;

        float qsum = 0.0f;
        float xsum = 0.0f;
        #pragma unroll 32
        for (unsigned int l = 0u; l < 32u; ++l) {
            const float xv = x[y + l];
            const unsigned int hi = ((unsigned int)qh[l] & u1) != 0u ? 16u : 0u;
            qsum += (float)((ql[ql_off + l] & 0x0Fu) + hi) * xv;
            xsum += xv;
        }
        acc += d1 * qsum - m1 * xsum;
        y += 32u;

        qsum = 0.0f;
        xsum = 0.0f;
        #pragma unroll 32
        for (unsigned int l = 0u; l < 32u; ++l) {
            const float xv = x[y + l];
            const unsigned int hi = ((unsigned int)qh[l] & u2) != 0u ? 16u : 0u;
            qsum += (float)(((unsigned int)ql[ql_off + l] >> 4u) + hi) * xv;
            xsum += xv;
        }
        acc += d2 * qsum - m2 * xsum;
        y += 32u;

        ql_off += 32u;
        is += 2u;
        u1 <<= 2u;
        u2 <<= 2u;
    }
    return acc;
}

/* ── Q6_K super-block · dot product ────────────────────────────────────────
   Block: [ql:128 @0][qh:64 @128][scales:16 i8 @192][d:f16 @208]  (210 bytes)
   dequantize_row_q6_K (ggml-quants.c:1994): per 128-element half and
   l in 0..32, four interleaved lanes y[l], y[l+32], y[l+64], y[l+96] with
   sub-scales sc[is+0], sc[is+2], sc[is+4], sc[is+6] where is = l / 16.
   dequant: d*sc*(q6 - 32)                                                   */
static __device__ float kq_dot_q6k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    const unsigned short d_raw = (unsigned short)bptr[208] | ((unsigned short)bptr[209] << 8u);
    const float d = kq_fast_fp16_to_float(d_raw);
    const unsigned char* ql = bptr;
    const unsigned char* qh = bptr + 128u;
    const signed char* scales_i8 = (const signed char*)(bptr + 192u);

    float acc = 0.0f;
    unsigned int y = 0u;
    unsigned int ql_off = 0u;
    unsigned int qh_off = 0u;
    unsigned int sc_off = 0u;

    for (unsigned int n = 0u; n < 2u; ++n) {   /* (0..256).step_by(128) */
        float dsc[8];
        #pragma unroll
        for (unsigned int t = 0u; t < 8u; ++t) {
            dsc[t] = d * (float)(int)scales_i8[sc_off + t];
        }

        #pragma unroll 32
        for (unsigned int l = 0u; l < 32u; ++l) {
            const unsigned int is = l >> 4u;   /* l / 16 */
            const unsigned int hq = qh[qh_off + l];
            const int q1 = (int)((ql[ql_off + l] & 0x0Fu) | ((hq & 3u) << 4u)) - 32;
            const int q2 = (int)((ql[ql_off + l + 32u] & 0x0Fu) | (((hq >> 2u) & 3u) << 4u)) - 32;
            const int q3 = (int)(((unsigned int)ql[ql_off + l] >> 4u)
                                 | (((hq >> 4u) & 3u) << 4u)) - 32;
            const int q4 = (int)(((unsigned int)ql[ql_off + l + 32u] >> 4u)
                                 | (((hq >> 6u) & 3u) << 4u)) - 32;

            acc += dsc[is]      * (float)q1 * x[y + l];
            acc += dsc[is + 2u] * (float)q2 * x[y + l + 32u];
            acc += dsc[is + 4u] * (float)q3 * x[y + l + 64u];
            acc += dsc[is + 6u] * (float)q4 * x[y + l + 96u];
        }

        y += 128u;
        ql_off += 64u;
        qh_off += 32u;
        sc_off += 8u;
    }
    return acc;
}

/* ── Q8_K super-block · dot product ────────────────────────────────────────
   Block: [d:f32 @0][qs:256 i8 @4][bsums:32 @260]  (292 bytes)
   `d` is f32, NOT f16. ggml's dequantize_row_q8_K is element-sequential, so
   no cursor walk is needed. bsums is unused by GEMV.                        */
static __device__ float kq_dot_q8k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    union { unsigned int u; float f; } ud;
    ud.u = (unsigned int)bptr[0]
         | ((unsigned int)bptr[1] << 8u)
         | ((unsigned int)bptr[2] << 16u)
         | ((unsigned int)bptr[3] << 24u);
    const float d = ud.f;
    const signed char* qs = (const signed char*)(bptr + 4u);

    float acc = 0.0f;
    #pragma unroll 32
    for (unsigned int j = 0u; j < 256u; ++j) {
        acc += d * (float)(int)qs[j] * x[j];
    }
    return acc;
}

/* ── Warp-shuffle reduction across 32 lanes ────────────────────────────── */
static __device__ __forceinline__ float kq_warp_reduce(float acc) {
    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    return acc;
}

/* ==========================================================================
   The six GEMV kernels. Each warp owns one output row and strides the row's
   super-blocks by lane; one lane decodes one whole block.

   Grid:  (ceil(n_rows / 8), 1, 1)
   Block: (256, 1, 1)
   ========================================================================== */

/* Kernel 1 — gemv_q2k (84 bytes/super-block) */
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
    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks
            + (unsigned long long)(row * blocks_per_row + b) * 84u;
        acc += kq_dot_q2k(bptr, input + (b << 8u));
    }
    acc = kq_warp_reduce(acc);
    if (lane == 0u) output[row] = acc;
}

/* Kernel 2 — gemv_q3k (110 bytes/super-block) */
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
    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks
            + (unsigned long long)(row * blocks_per_row + b) * 110u;
        acc += kq_dot_q3k(bptr, input + (b << 8u));
    }
    acc = kq_warp_reduce(acc);
    if (lane == 0u) output[row] = acc;
}

/* Kernel 3 — gemv_q4k (144 bytes/super-block) */
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
    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks
            + (unsigned long long)(row * blocks_per_row + b) * 144u;
        acc += kq_dot_q4k(bptr, input + (b << 8u));
    }
    acc = kq_warp_reduce(acc);
    if (lane == 0u) output[row] = acc;
}

/* Kernel 4 — gemv_q5k (176 bytes/super-block) */
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
    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks
            + (unsigned long long)(row * blocks_per_row + b) * 176u;
        acc += kq_dot_q5k(bptr, input + (b << 8u));
    }
    acc = kq_warp_reduce(acc);
    if (lane == 0u) output[row] = acc;
}

/* Kernel 5 — gemv_q6k (210 bytes/super-block) */
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
    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks
            + (unsigned long long)(row * blocks_per_row + b) * 210u;
        acc += kq_dot_q6k(bptr, input + (b << 8u));
    }
    acc = kq_warp_reduce(acc);
    if (lane == 0u) output[row] = acc;
}

/* Kernel 6 — gemv_q8k (292 bytes/super-block) */
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
    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks
            + (unsigned long long)(row * blocks_per_row + b) * 292u;
        acc += kq_dot_q8k(bptr, input + (b << 8u));
    }
    acc = kq_warp_reduce(acc);
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
    if k == 0 || !k.is_multiple_of(256) {
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
///   - decoded by ggml's `dequantize_row_q2_K` walk (`kq_dot_q2k`)
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
///   - 6-bit scales from the `kmask1`/`kmask2` shuffle, biased by -32;
///     inverted hmask — see `kq_dot_q3k`
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
///   - 8 sub-blocks × 32 weights, 4-bit quant, 6-bit `get_scale_min_k4`
///     scale/min; 32 low then 32 high nibbles per 64-element group
///     (`kq_dot_q4k`)
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
///   - Q4_K's walk plus the `u1`/`u2` `qh` masks (`kq_dot_q5k`)
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
///   - four-lane interleave with `sc[is+0,+2,+4,+6]` (`kq_dot_q6k`)
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
            CUDA_K_QUANT_KERNELS_SRC.contains("kq_scale_min_k4"),
            "CUDA_K_QUANT_KERNELS_SRC must contain kq_scale_min_k4 (ggml get_scale_min_k4)"
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
