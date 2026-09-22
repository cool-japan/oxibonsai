//! CUDA C kernel source strings for OxiBonsai K-quant batch GEMM (prefill) operations.
//!
//! # K-quant prefill kernel catalogue
//!
//! | Kernel                               | Description                                              |
//! |--------------------------------------|----------------------------------------------------------|
//! | `gemm_q2k`                           | Batch GEMM: Q2_K AoS, col-major I/O, col_sums\[8\]       |
//! | `gemm_q2k_residual`                  | Q2_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q2k`      | Fused gate+up Q2_K GEMM with SwiGLU epilogue            |
//! | `gemm_q3k`                           | Batch GEMM: Q3_K AoS, col-major I/O, col_sums\[8\]       |
//! | `gemm_q3k_residual`                  | Q3_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q3k`      | Fused gate+up Q3_K GEMM with SwiGLU epilogue            |
//! | `gemm_q4k`                           | Batch GEMM: Q4_K AoS, col-major I/O, col_sums\[8\]       |
//! | `gemm_q4k_residual`                  | Q4_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q4k`      | Fused gate+up Q4_K GEMM with SwiGLU epilogue            |
//! | `gemm_q5k`                           | Batch GEMM: Q5_K AoS, col-major I/O, col_sums\[8\]       |
//! | `gemm_q5k_residual`                  | Q5_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q5k`      | Fused gate+up Q5_K GEMM with SwiGLU epilogue            |
//! | `gemm_q6k`                           | Batch GEMM: Q6_K AoS, col-major I/O, col_sums\[8\]       |
//! | `gemm_q6k_residual`                  | Q6_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q6k`      | Fused gate+up Q6_K GEMM with SwiGLU epilogue            |
//! | `gemm_q8k`                           | Batch GEMM: Q8_K AoS, col-major I/O, col_sums\[8\]       |
//! | `gemm_q8k_residual`                  | Q8_K GEMM + fused residual add                          |
//! | `fused_gate_up_swiglu_gemm_q8k`      | Fused gate+up Q8_K GEMM with SwiGLU epilogue            |
//!
//! # The ggml walk is load-bearing
//!
//! Every kernel in this file reduces one super-block through the matching
//! `kq_pf_dot_q*k` device helper, and those helpers are **fused
//! transliterations of ggml's `dequantize_row_q*_K`**
//! (`ggml/src/ggml-quants.c`). ggml's output cursor is decoupled from its byte
//! cursors, so the *n*-th value it emits — the value that multiplies `x[n]` —
//! is generally **not** read from byte lane *n*. The byte-exact CPU reference
//! (`oxibonsai_core::BlockQ*K::dequant`) is the normative source for these
//! loops. Keeping the block walk in exactly six helpers is deliberate: the
//! pre-`core-gguf-K0` layout bug was copied into all 18 kernels.
//!
//! # Block layouts (QK_K = 256 weights per super-block)
//!
//! **Q2_K** (84 bytes/block, 256 weights):
//! ```text
//! bytes  0-15: scales[16]  — low nibble = sub_sc, high nibble = sub_mn
//! bytes 16-79: qs[64]      — 2 bits/weight, 4/byte (LSB-first)
//! bytes 80-81: d (FP16 LE)
//! bytes 82-83: dmin (FP16 LE)
//! Per 128-element half `shift` steps 0,2,4,6 and `is` advances twice per step:
//! output n*128 + j*32 + l     <- qs[n*32 + l]      >> 2j  under scales[is]
//! output n*128 + j*32 + 16+l  <- qs[n*32 + 16 + l] >> 2j  under scales[is+1]
//! dequant: d*sc*q - dmin*mn (q in [0,3])
//! ```
//!
//! **Q3_K** (110 bytes/block, 256 weights):
//! ```text
//! bytes  0-31:  hmask[32]  — high bit/weight, 8/byte, INVERTED sense
//! bytes 32-95:  qs[64]     — low 2 bits/weight, 4/byte (LSB-first)
//! bytes 96-107: scales[12] — 16 x 6-bit via the kmask1/kmask2 aux[4] shuffle
//! bytes 108-109: d (FP16 LE)
//! Q2_K's walk; sc carries a -32 bias; hi = (hm[l] & m) ? 0 : 4 with
//! m = 1 << (4n + j) reusing the same 32 hmask bytes for all eight steps.
//! dequant: d*(sc-32)*(q - hi)
//! ```
//!
//! **Q4_K** (144 bytes/block, 256 weights):
//! ```text
//! bytes  0- 1: d (FP16 LE)
//! bytes  2- 3: dmin (FP16 LE)
//! bytes  4-15: scales[12]  — 6-bit sc[8] + 6-bit mn[8] via get_scale_min_k4
//! bytes 16-143: qs[128]    — 4 bits/weight, 2/byte (nibbles)
//! Per 64-element group: 32 LOW nibbles under get_scale_min_k4(is), then 32
//! HIGH nibbles under get_scale_min_k4(is + 1); is += 2 per group.
//! dequant: d*sc*q - dmin*mn
//! ```
//!
//! **Q5_K** (176 bytes/block, 256 weights):
//! ```text
//! bytes  0- 1: d (FP16 LE)
//! bytes  2- 3: dmin (FP16 LE)
//! bytes  4-15: scales[12]  — same 6-bit packing as Q4_K
//! bytes 16-47: qh[32]      — 5th bit, indexed l in 0..32, NOT advanced
//! bytes 48-175: qs[128]    — low 4 bits, 2/byte (nibbles)
//! Q4_K's walk; the u1/u2 masks shift left by 2 per 64-element group.
//! dequant: d*sc*(q + 16*qh_bit) - dmin*mn
//! ```
//!
//! **Q6_K** (210 bytes/block, 256 weights):
//! ```text
//! bytes  0-127:  ql[128]    — low 4 bits/weight, 2/byte (nibbles)
//! bytes 128-191: qh[64]     — high 2 bits/weight, 4/byte (2 bits each)
//! bytes 192-207: scales[16] — signed int8, 1/sub-block
//! bytes 208-209: d (FP16 LE)
//! Per 128-element half and l in 0..32, four interleaved lanes y[l], y[l+32],
//! y[l+64], y[l+96] with sub-scales sc[is+0,+2,+4,+6], is = l/16.
//! dequant: d*scales_i8*(q6-32)
//! ```
//!
//! **Q8_K** (292 bytes/block, 256 weights):
//! ```text
//! bytes  0-3:   d (FP32 LE)    — NOTE: float, not FP16!
//! bytes  4-259: qs[256] (i8)   — 256 signed int8 weights
//! bytes 260-291: bsums (i16)   — not used in GEMM
//! Element-sequential in ggml too. dequant: d_f32 * qs[i]
//! ```
//!
//! # Grid / block config
//! - Grid:  `(ceil(n_rows / 8), 1, 1)` — 8 warps per CTA
//! - Block: `(256, 1, 1)` — 8 warps × 32 lanes
//! - `k` must be a positive multiple of 256 for all K-quant formats
//! - One lane decodes one whole super-block: ggml's `is` / `m` / `u1` / `u2`
//!   cursors are sequential state and must not be split across lanes.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

/// CUDA C source for all K-quant batch GEMM (prefill) kernels.
///
/// All kernels use AoS weight layout (super-blocks stored contiguously as-is from GGUF).
/// Batch tensors use column-major layout: `buf[col * dim + element]`.
/// The cap-of-8 outer loop prevents silent bugs when batch_size > 8.
///
/// Per-format block decoding lives in the six `kq_pf_dot_q*k` helpers, each a
/// fused transliteration of ggml's `dequantize_row_q*_K`; the kernels below only
/// stride blocks, reduce warps and apply their epilogue.
pub const CUDA_K_QUANT_PREFILL_KERNELS_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA K-quant prefill (batch GEMM) kernels.
   Formats: Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K
   QK_K = 256 weights per super-block for all formats.

   Batch tensors: column-major  buf[col * dim + element]
   Grid:  (ceil(n_rows/8), 1, 1)  — 8 warps per CTA, 1 warp/row
   Block: (256, 1, 1)             — 8 warps × 32 lanes
   k must be a positive multiple of 256.

   Every kq_pf_dot_q*k helper mirrors ggml's dequantize_row_q*_K walk
   (ggml/src/ggml-quants.c) element for element, fused with the dot product:
   the output cursor y indexes the input, the byte cursors index the block.
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

/* ── Warp-shuffle reduction across 32 lanes ────────────────────────────── */
static __device__ __forceinline__ float kq_pf_warp_reduce(float acc) {
    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    return acc;
}

/* ── ggml get_scale_min_k4 (ggml-quants.c:935) ─────────────────────────────
   Sub-block j's 6-bit scale and 6-bit min out of the 12-byte packed `scales`
   array shared by Q4_K and Q5_K. For j < 4 the scale is a FULL 6-bit value
   read from byte j -- not a 4-bit nibble.                                   */
static __device__ __forceinline__ void kq_pf_scale_min_k4(
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
static __device__ void kq_pf_unpack_q3k_scales(
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

/* ── Q2_K super-block · dot product (dequantize_row_q2_K) ──────────────── */
static __device__ float kq_pf_dot_q2k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    const unsigned short d_raw    = (unsigned short)bptr[80] | ((unsigned short)bptr[81] << 8u);
    const unsigned short dmin_raw = (unsigned short)bptr[82] | ((unsigned short)bptr[83] << 8u);
    const float d    = kq_pf_fast_fp16_to_float(d_raw);
    const float dmin = kq_pf_fast_fp16_to_float(dmin_raw);
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

/* ── Q3_K super-block · dot product (dequantize_row_q3_K) ──────────────── */
static __device__ float kq_pf_dot_q3k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    const unsigned short d_raw = (unsigned short)bptr[108] | ((unsigned short)bptr[109] << 8u);
    const float d_all = kq_pf_fast_fp16_to_float(d_raw);
    const unsigned char* hm = bptr;        /* hmask[32] */
    const unsigned char* qs = bptr + 32u;  /* qs[64]    */

    unsigned char sc[16];
    kq_pf_unpack_q3k_scales(bptr + 96u, sc);

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

/* ── Q4_K super-block · dot product (dequantize_row_q4_K) ──────────────── */
static __device__ float kq_pf_dot_q4k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    const unsigned short d_raw    = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
    const unsigned short dmin_raw = (unsigned short)bptr[2] | ((unsigned short)bptr[3] << 8u);
    const float d    = kq_pf_fast_fp16_to_float(d_raw);
    const float dmin = kq_pf_fast_fp16_to_float(dmin_raw);
    const unsigned char* scales = bptr + 4u;
    const unsigned char* qs     = bptr + 16u;

    float acc = 0.0f;
    unsigned int y = 0u;
    unsigned int q_off = 0u;
    unsigned int is = 0u;

    for (unsigned int g = 0u; g < 4u; ++g) {   /* (0..256).step_by(64) */
        unsigned char sc1, mn1, sc2, mn2;
        kq_pf_scale_min_k4(scales, is, &sc1, &mn1);
        kq_pf_scale_min_k4(scales, is + 1u, &sc2, &mn2);
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

/* ── Q5_K super-block · dot product (dequantize_row_q5_K) ──────────────── */
static __device__ float kq_pf_dot_q5k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    const unsigned short d_raw    = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
    const unsigned short dmin_raw = (unsigned short)bptr[2] | ((unsigned short)bptr[3] << 8u);
    const float d    = kq_pf_fast_fp16_to_float(d_raw);
    const float dmin = kq_pf_fast_fp16_to_float(dmin_raw);
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
        kq_pf_scale_min_k4(scales, is, &sc1, &mn1);
        kq_pf_scale_min_k4(scales, is + 1u, &sc2, &mn2);
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

/* ── Q6_K super-block · dot product (dequantize_row_q6_K) ──────────────── */
static __device__ float kq_pf_dot_q6k(
    const unsigned char* __restrict__ bptr,
    const float* __restrict__ x
) {
    const unsigned short d_raw = (unsigned short)bptr[208] | ((unsigned short)bptr[209] << 8u);
    const float d = kq_pf_fast_fp16_to_float(d_raw);
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

/* ── Q8_K super-block · dot product (element-sequential) ───────────────── */
static __device__ float kq_pf_dot_q8k(
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

/* =========================================================================
   Q2_K kernels (84 bytes/block, 256 weights) — all three variants
   reduce one super-block through kq_pf_dot_q2k().
   ========================================================================= */

/* ── Kernel 1: gemm_q2k ──────────────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q2k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u)
                outputs[(unsigned long long)(col_base + col) * n_rows + row] += s;
        }
    }
}

/* ── Kernel 2: gemm_q2k_residual ─────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q2k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u) {
                const unsigned long long idx = (unsigned long long)(col_base + col) * n_rows + row;
                outputs[idx] = residual[idx] + s;
            }
        }
    }
}

/* ── Kernel 3: fused_gate_up_swiglu_gemm_q2k ─────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                gate_sums[col] += kq_pf_dot_q2k(gbptr, xbase);
                up_sums[col]   += kq_pf_dot_q2k(ubptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float gs = kq_pf_warp_reduce(gate_sums[col]);
            const float us = kq_pf_warp_reduce(up_sums[col]);
            if (lane == 0u) {
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Q3_K kernels (110 bytes/block, 256 weights) — all three variants
   reduce one super-block through kq_pf_dot_q3k().
   ========================================================================= */

/* ── Kernel 4: gemm_q3k ──────────────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q3k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u)
                outputs[(unsigned long long)(col_base + col) * n_rows + row] += s;
        }
    }
}

/* ── Kernel 5: gemm_q3k_residual ─────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q3k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u) {
                const unsigned long long idx = (unsigned long long)(col_base + col) * n_rows + row;
                outputs[idx] = residual[idx] + s;
            }
        }
    }
}

/* ── Kernel 6: fused_gate_up_swiglu_gemm_q3k ─────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                gate_sums[col] += kq_pf_dot_q3k(gbptr, xbase);
                up_sums[col]   += kq_pf_dot_q3k(ubptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float gs = kq_pf_warp_reduce(gate_sums[col]);
            const float us = kq_pf_warp_reduce(up_sums[col]);
            if (lane == 0u) {
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Q4_K kernels (144 bytes/block, 256 weights) — all three variants
   reduce one super-block through kq_pf_dot_q4k().
   ========================================================================= */

/* ── Kernel 7: gemm_q4k ──────────────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q4k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u)
                outputs[(unsigned long long)(col_base + col) * n_rows + row] += s;
        }
    }
}

/* ── Kernel 8: gemm_q4k_residual ─────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q4k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u) {
                const unsigned long long idx = (unsigned long long)(col_base + col) * n_rows + row;
                outputs[idx] = residual[idx] + s;
            }
        }
    }
}

/* ── Kernel 9: fused_gate_up_swiglu_gemm_q4k ─────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                gate_sums[col] += kq_pf_dot_q4k(gbptr, xbase);
                up_sums[col]   += kq_pf_dot_q4k(ubptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float gs = kq_pf_warp_reduce(gate_sums[col]);
            const float us = kq_pf_warp_reduce(up_sums[col]);
            if (lane == 0u) {
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Q5_K kernels (176 bytes/block, 256 weights) — all three variants
   reduce one super-block through kq_pf_dot_q5k().
   ========================================================================= */

/* ── Kernel 10: gemm_q5k ─────────────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q5k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u)
                outputs[(unsigned long long)(col_base + col) * n_rows + row] += s;
        }
    }
}

/* ── Kernel 11: gemm_q5k_residual ────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q5k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u) {
                const unsigned long long idx = (unsigned long long)(col_base + col) * n_rows + row;
                outputs[idx] = residual[idx] + s;
            }
        }
    }
}

/* ── Kernel 12: fused_gate_up_swiglu_gemm_q5k ────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                gate_sums[col] += kq_pf_dot_q5k(gbptr, xbase);
                up_sums[col]   += kq_pf_dot_q5k(ubptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float gs = kq_pf_warp_reduce(gate_sums[col]);
            const float us = kq_pf_warp_reduce(up_sums[col]);
            if (lane == 0u) {
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Q6_K kernels (210 bytes/block, 256 weights) — all three variants
   reduce one super-block through kq_pf_dot_q6k().
   ========================================================================= */

/* ── Kernel 13: gemm_q6k ─────────────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q6k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u)
                outputs[(unsigned long long)(col_base + col) * n_rows + row] += s;
        }
    }
}

/* ── Kernel 14: gemm_q6k_residual ────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q6k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u) {
                const unsigned long long idx = (unsigned long long)(col_base + col) * n_rows + row;
                outputs[idx] = residual[idx] + s;
            }
        }
    }
}

/* ── Kernel 15: fused_gate_up_swiglu_gemm_q6k ────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                gate_sums[col] += kq_pf_dot_q6k(gbptr, xbase);
                up_sums[col]   += kq_pf_dot_q6k(ubptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float gs = kq_pf_warp_reduce(gate_sums[col]);
            const float us = kq_pf_warp_reduce(up_sums[col]);
            if (lane == 0u) {
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}

/* =========================================================================
   Q8_K kernels (292 bytes/block, 256 weights) — all three variants
   reduce one super-block through kq_pf_dot_q8k().
   ========================================================================= */

/* ── Kernel 16: gemm_q8k ─────────────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q8k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u)
                outputs[(unsigned long long)(col_base + col) * n_rows + row] += s;
        }
    }
}

/* ── Kernel 17: gemm_q8k_residual ────────────────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                col_sums[col] += kq_pf_dot_q8k(bptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float s = kq_pf_warp_reduce(col_sums[col]);
            if (lane == 0u) {
                const unsigned long long idx = (unsigned long long)(col_base + col) * n_rows + row;
                outputs[idx] = residual[idx] + s;
            }
        }
    }
}

/* ── Kernel 18: fused_gate_up_swiglu_gemm_q8k ────────────────────────── */
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
            const unsigned int x_base = b << 8u;

            for (unsigned int col = 0u; col < cols; ++col) {
                const float* xbase = inputs + (unsigned long long)(col_base + col) * k + x_base;
                gate_sums[col] += kq_pf_dot_q8k(gbptr, xbase);
                up_sums[col]   += kq_pf_dot_q8k(ubptr, xbase);
            }
        }

        for (unsigned int col = 0u; col < cols; ++col) {
            const float gs = kq_pf_warp_reduce(gate_sums[col]);
            const float us = kq_pf_warp_reduce(up_sums[col]);
            if (lane == 0u) {
                outputs[(unsigned long long)(col_base + col) * n_rows + row] =
                    kq_pf_silu(gs) * us;
            }
        }
    }
}
"#;
