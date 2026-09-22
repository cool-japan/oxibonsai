//! Metal MSL kernel sources for K-quant GEMV operations
//! (`Q2_K` / `Q3_K` / `Q4_K` / `Q5_K` / `Q6_K` / `Q8_K`), single-token decode path.
//!
//! These mirror the CUDA kernels in
//! `crates/oxibonsai-kernels/src/gpu_backend/cuda_k_quant_kernels.rs` and are the
//! Metal counterparts of the scalar reference kernels
//! (`crates/oxibonsai-kernels/src/gemv_q{2,3,4,5,6,8}k.rs`), which are the parity
//! oracle. All formats use `QK_K = 256` weights per super-block.
//!
//! # The ggml walk is load-bearing
//!
//! Every kernel below is a **fused transliteration of ggml's
//! `dequantize_row_q*_K`** (`ggml/src/ggml-quants.c`), not an "equivalent"
//! element-sequential decode: ggml's output cursor `y` is decoupled from the
//! byte cursors, so the *n*-th value ggml emits — the value that multiplies
//! `input[n]` — is generally **not** read from byte lane *n*. The byte-exact
//! CPU reference (`oxibonsai_core::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K,
//! BlockQ6K}::dequant`) is the normative source for these loops; the kernels
//! reproduce them cursor for cursor.
//!
//! # Block layouts (AoS, matching the `#[repr(C)]` core structs; offsets in bytes)
//!
//! - **Q2_K** (84): `[scales:16 @0][qs:64 @16][d:f16 @80][dmin:f16 @82]`
//!   Per 128-element half, `shift` steps 0, 2, 4, 6 and `is` advances twice per
//!   step: output `n*128 + j*32 + l` reads `qs[n*32 + l] >> 2j` under
//!   `scales[is]`, output `n*128 + j*32 + 16 + l` reads `qs[n*32 + 16 + l] >> 2j`
//!   under `scales[is + 1]`. `sc = scales[is] & 0xF`, `mn = scales[is] >> 4`.
//!   Dequant: `d*sc*q - dmin*mn`.
//! - **Q3_K** (110): `[hmask:32 @0][qs:64 @32][scales:12 @96][d:f16 @108]`
//!   Same 128-half / `shift` / `is` walk as Q2_K. The 16 six-bit scales come
//!   from the `kmask1`/`kmask2` `aux[4]` shuffle and carry a `-32` bias; the
//!   `hmask` bit is **inverted** (`q - (hm[l] & m ? 0 : 4)`) and the 32 hmask
//!   bytes are reused across all eight `m = 1 << (4n + j)` steps.
//!   Dequant: `d*(sc-32)*(q - hi)`.
//! - **Q4_K** (144): `[d:f16 @0][dmin:f16 @2][scales:12 @4][qs:128 @16]`
//!   Per 64-element group: 32 **low** nibbles under `get_scale_min_k4(is)` then
//!   32 **high** nibbles under `get_scale_min_k4(is + 1)`, `is += 2` per group.
//!   Dequant: `d*sc*q - dmin*mn`.
//! - **Q5_K** (176): `[d:f16 @0][dmin:f16 @2][scales:12 @4][qh:32 @16][qs:128 @48]`
//!   Q4_K's walk plus the 5th bit: `qh[l]` is indexed `l in 0..32` and does
//!   **not** advance per group; the masks `u1`/`u2` shift left by 2 per group.
//!   Dequant: `d*sc*(q + 16*qh_bit) - dmin*mn`.
//! - **Q6_K** (210): `[ql:128 @0][qh:64 @128][scales:16 i8 @192][d:f16 @208]`
//!   Per 128-element half and `l in 0..32`, four interleaved lanes
//!   `y[l]`, `y[l+32]`, `y[l+64]`, `y[l+96]` with scales `sc[is+0,+2,+4,+6]`,
//!   `is = l/16`. 6-bit values biased by `-32`.
//!   Dequant: `d*sc*(q6-32)`.
//! - **Q8_K** (292): `[d:f32 @0][qs:256 i8 @4][bsums:32 @260]`
//!   `d` is **f32** (not f16!). Dequant: `d*qs[i]`. `bsums` unused by GEMV.
//!   Q8_K is element-sequential in ggml too, so this kernel is unchanged.
//!
//! # Dispatch (identical for all six kernels)
//!
//! Grid:  `[ceil(n_rows / 8), 1, 1]` — 8 simdgroups per threadgroup, one row per simdgroup.
//! Block: `[256, 1, 1]` — 8 simdgroups × 32 lanes.
//! `k` must be a positive multiple of 256.
//!
//! One lane decodes one whole super-block: ggml's `is` / `m` / `u1` / `u2`
//! cursors are sequential state and must not be split across lanes.
//!
//! Buffer indices: blocks (0), input (1), output (2), n_rows (3), k (4).

/// Metal MSL kernel: `Q2_K` GEMV — one simdgroup per output row.
///
/// Transliterates `dequantize_row_q2_K` (`ggml-quants.c:1016`): `is++` under a
/// `shift` stepping 0, 2, 4, 6 per 128-element half.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q2K_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q2k(
    device const uchar* blocks [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& n_rows      [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k >> 8u;  // k / 256
    const uint stride = 84u;
    float acc = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        device const uchar* bptr = blocks + (row * blocks_per_row + b) * stride;
        const ushort d_raw    = ushort(bptr[80]) | (ushort(bptr[81]) << 8u);
        const ushort dmin_raw = ushort(bptr[82]) | (ushort(bptr[83]) << 8u);
        const float d    = float(as_type<half>(d_raw));
        const float dmin = float(as_type<half>(dmin_raw));
        device const uchar* qs = bptr + 16u;
        device const float* xbase = input + (b << 8u);  // b * 256

        // ggml cursors: y = output position, is = scales index, q_off = qs byte.
        uint y = 0u;
        uint is = 0u;
        uint q_off = 0u;

        for (uint n = 0u; n < 2u; ++n) {          // QK_K / 128
            uint shift = 0u;
            for (uint j = 0u; j < 4u; ++j) {
                uint sc = uint(bptr[is]);
                ++is;
                float dl = d * float(sc & 0x0Fu);
                float ml = dmin * float(sc >> 4u);
                float qsum = 0.0f;
                float xsum = 0.0f;
                for (uint l = 0u; l < 16u; ++l) {
                    const float x = xbase[y + l];
                    qsum += float((qs[q_off + l] >> shift) & 3u) * x;
                    xsum += x;
                }
                acc += dl * qsum - ml * xsum;
                y += 16u;

                sc = uint(bptr[is]);
                ++is;
                dl = d * float(sc & 0x0Fu);
                ml = dmin * float(sc >> 4u);
                qsum = 0.0f;
                xsum = 0.0f;
                for (uint l = 0u; l < 16u; ++l) {
                    const float x = xbase[y + l];
                    qsum += float((qs[q_off + l + 16u] >> shift) & 3u) * x;
                    xsum += x;
                }
                acc += dl * qsum - ml * xsum;
                y += 16u;

                shift += 2u;
            }
            q_off += 32u;
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

/// Metal MSL kernel: `Q3_K` GEMV — one simdgroup per output row.
///
/// Transliterates `dequantize_row_q3_K` (`ggml-quants.c:1360`): the
/// `kmask1`/`kmask2` `aux[4]` scale shuffle, the `-32` bias and the inverted
/// `hmask` convention.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q3K_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Unpack the 12 packed Q3_K scale bytes into 16 biased 6-bit scales
// (ggml-quants.c:1374-1381). Callers subtract the +32 bias.
static void kq_unpack_q3k_scales(device const uchar* s, thread uchar* out) {
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    const uint a0_in = uint(s[0]) | (uint(s[1]) << 8u) | (uint(s[2]) << 16u) | (uint(s[3]) << 24u);
    const uint a1_in = uint(s[4]) | (uint(s[5]) << 8u) | (uint(s[6]) << 16u) | (uint(s[7]) << 24u);
    const uint tmp   = uint(s[8]) | (uint(s[9]) << 8u) | (uint(s[10]) << 16u) | (uint(s[11]) << 24u);

    const uint aux2 = ((a0_in >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
    const uint aux3 = ((a1_in >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
    const uint aux0 = (a0_in & kmask2) | ((tmp & kmask1) << 4u);
    const uint aux1 = (a1_in & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

    const uint aux[4] = { aux0, aux1, aux2, aux3 };
    for (uint w = 0u; w < 4u; ++w) {
        out[4u * w + 0u] = uchar(aux[w] & 0xFFu);
        out[4u * w + 1u] = uchar((aux[w] >> 8u) & 0xFFu);
        out[4u * w + 2u] = uchar((aux[w] >> 16u) & 0xFFu);
        out[4u * w + 3u] = uchar((aux[w] >> 24u) & 0xFFu);
    }
}

kernel void gemv_q3k(
    device const uchar* blocks [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& n_rows      [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k >> 8u;
    const uint stride = 110u;
    float acc = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        device const uchar* bptr = blocks + (row * blocks_per_row + b) * stride;
        const ushort d_raw = ushort(bptr[108]) | (ushort(bptr[109]) << 8u);
        const float d_all = float(as_type<half>(d_raw));

        device const uchar* hm = bptr;          // hmask[32]
        device const uchar* qs = bptr + 32u;    // qs[64]
        device const float* xbase = input + (b << 8u);

        thread uchar sc[16];
        kq_unpack_q3k_scales(bptr + 96u, sc);

        uint y = 0u;
        uint is = 0u;
        uint q_off = 0u;
        uint m = 1u;  // hmask bit for this (n, j) step: 1 << (4n + j)

        for (uint n = 0u; n < 2u; ++n) {
            uint shift = 0u;
            for (uint j = 0u; j < 4u; ++j) {
                float dl = d_all * float(int(sc[is]) - 32);
                ++is;
                float qsum = 0.0f;
                for (uint l = 0u; l < 16u; ++l) {
                    const int hi = (uint(hm[l]) & m) != 0u ? 0 : 4;
                    const int q  = int((qs[q_off + l] >> shift) & 3u);
                    qsum += float(q - hi) * xbase[y + l];
                }
                acc += dl * qsum;
                y += 16u;

                dl = d_all * float(int(sc[is]) - 32);
                ++is;
                qsum = 0.0f;
                for (uint l = 0u; l < 16u; ++l) {
                    const int hi = (uint(hm[l + 16u]) & m) != 0u ? 0 : 4;
                    const int q  = int((qs[q_off + l + 16u] >> shift) & 3u);
                    qsum += float(q - hi) * xbase[y + l];
                }
                acc += dl * qsum;
                y += 16u;

                shift += 2u;
                m <<= 1u;
            }
            q_off += 32u;
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

/// Metal MSL kernel: `Q4_K` GEMV — one simdgroup per output row.
///
/// Transliterates `dequantize_row_q4_K` (`ggml-quants.c:1584`), using ggml's
/// `get_scale_min_k4` (`kq_scale_min_k4_q4k` below) and the 32-low / 32-high nibble
/// emission order.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q4K_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ggml `get_scale_min_k4` (ggml-quants.c:935): sub-block `j`'s 6-bit scale and
// 6-bit min out of the 12-byte packed `scales` array (Q4_K / Q5_K layout).
// For j < 4 the scale is a FULL 6-bit value read from byte j — not a nibble.
static void kq_scale_min_k4_q4k(device const uchar* q, uint j,
                            thread uchar* sc_out, thread uchar* m_out) {
    if (j < 4u) {
        *sc_out = uchar(q[j] & 63u);
        *m_out  = uchar(q[j + 4u] & 63u);
    } else {
        *sc_out = uchar((q[j + 4u] & 0x0Fu) | ((uint(q[j - 4u]) >> 6u) << 4u));
        *m_out  = uchar((uint(q[j + 4u]) >> 4u) | ((uint(q[j]) >> 6u) << 4u));
    }
}

kernel void gemv_q4k(
    device const uchar* blocks [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& n_rows      [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k >> 8u;
    const uint stride = 144u;
    float acc = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        device const uchar* bptr = blocks + (row * blocks_per_row + b) * stride;
        const ushort d_raw    = ushort(bptr[0]) | (ushort(bptr[1]) << 8u);
        const ushort dmin_raw = ushort(bptr[2]) | (ushort(bptr[3]) << 8u);
        const float d    = float(as_type<half>(d_raw));
        const float dmin = float(as_type<half>(dmin_raw));

        device const uchar* scales = bptr + 4u;
        device const uchar* qs     = bptr + 16u;
        device const float* xbase  = input + (b << 8u);

        uint y = 0u;
        uint q_off = 0u;
        uint is = 0u;

        for (uint g = 0u; g < 4u; ++g) {  // (0..256).step_by(64)
            thread uchar sc1;
            thread uchar mn1;
            thread uchar sc2;
            thread uchar mn2;
            kq_scale_min_k4_q4k(scales, is, &sc1, &mn1);
            kq_scale_min_k4_q4k(scales, is + 1u, &sc2, &mn2);
            const float d1 = d * float(sc1);
            const float m1 = dmin * float(mn1);
            const float d2 = d * float(sc2);
            const float m2 = dmin * float(mn2);

            float qsum = 0.0f;
            float xsum = 0.0f;
            for (uint l = 0u; l < 32u; ++l) {   // 32 LOW nibbles
                const float x = xbase[y + l];
                qsum += float(qs[q_off + l] & 0x0Fu) * x;
                xsum += x;
            }
            acc += d1 * qsum - m1 * xsum;
            y += 32u;

            qsum = 0.0f;
            xsum = 0.0f;
            for (uint l = 0u; l < 32u; ++l) {   // then 32 HIGH nibbles
                const float x = xbase[y + l];
                qsum += float(uint(qs[q_off + l]) >> 4u) * x;
                xsum += x;
            }
            acc += d2 * qsum - m2 * xsum;
            y += 32u;

            q_off += 32u;
            is += 2u;
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

/// Metal MSL kernel: `Q5_K` GEMV — one simdgroup per output row.
///
/// Transliterates `dequantize_row_q5_K` (`ggml-quants.c:1786`): Q4_K's walk plus
/// the `u1`/`u2` high-bit masks that shift left by 2 per 64-element group over a
/// `qh` array indexed `l in 0..32`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q5K_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ggml `get_scale_min_k4` (ggml-quants.c:935) — see the Q4_K kernel.
static void kq_scale_min_k4_q5k(device const uchar* q, uint j,
                            thread uchar* sc_out, thread uchar* m_out) {
    if (j < 4u) {
        *sc_out = uchar(q[j] & 63u);
        *m_out  = uchar(q[j + 4u] & 63u);
    } else {
        *sc_out = uchar((q[j + 4u] & 0x0Fu) | ((uint(q[j - 4u]) >> 6u) << 4u));
        *m_out  = uchar((uint(q[j + 4u]) >> 4u) | ((uint(q[j]) >> 6u) << 4u));
    }
}

kernel void gemv_q5k(
    device const uchar* blocks [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& n_rows      [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k >> 8u;
    const uint stride = 176u;
    float acc = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        device const uchar* bptr = blocks + (row * blocks_per_row + b) * stride;
        const ushort d_raw    = ushort(bptr[0]) | (ushort(bptr[1]) << 8u);
        const ushort dmin_raw = ushort(bptr[2]) | (ushort(bptr[3]) << 8u);
        const float d    = float(as_type<half>(d_raw));
        const float dmin = float(as_type<half>(dmin_raw));

        device const uchar* scales = bptr + 4u;
        device const uchar* qh     = bptr + 16u;   // 32 bytes, NOT advanced per group
        device const uchar* ql     = bptr + 48u;
        device const float* xbase  = input + (b << 8u);

        uint y = 0u;
        uint ql_off = 0u;
        uint is = 0u;
        uint u1 = 1u;
        uint u2 = 2u;

        for (uint g = 0u; g < 4u; ++g) {  // (0..256).step_by(64)
            thread uchar sc1;
            thread uchar mn1;
            thread uchar sc2;
            thread uchar mn2;
            kq_scale_min_k4_q5k(scales, is, &sc1, &mn1);
            kq_scale_min_k4_q5k(scales, is + 1u, &sc2, &mn2);
            const float d1 = d * float(sc1);
            const float m1 = dmin * float(mn1);
            const float d2 = d * float(sc2);
            const float m2 = dmin * float(mn2);

            float qsum = 0.0f;
            float xsum = 0.0f;
            for (uint l = 0u; l < 32u; ++l) {
                const float x = xbase[y + l];
                const uint hi = (uint(qh[l]) & u1) != 0u ? 16u : 0u;
                qsum += float((ql[ql_off + l] & 0x0Fu) + hi) * x;
                xsum += x;
            }
            acc += d1 * qsum - m1 * xsum;
            y += 32u;

            qsum = 0.0f;
            xsum = 0.0f;
            for (uint l = 0u; l < 32u; ++l) {
                const float x = xbase[y + l];
                const uint hi = (uint(qh[l]) & u2) != 0u ? 16u : 0u;
                qsum += float((uint(ql[ql_off + l]) >> 4u) + hi) * x;
                xsum += x;
            }
            acc += d2 * qsum - m2 * xsum;
            y += 32u;

            ql_off += 32u;
            is += 2u;
            u1 <<= 2u;
            u2 <<= 2u;
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

/// Metal MSL kernel: `Q6_K` GEMV — one simdgroup per output row.
///
/// Transliterates `dequantize_row_q6_K` (`ggml-quants.c:1994`): the
/// n-step-128 / `l in 0..32` four-lane interleave with `sc[is+0,+2,+4,+6]`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q6K_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q6k(
    device const uchar* blocks [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& n_rows      [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k >> 8u;
    const uint stride = 210u;
    float acc = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        device const uchar* bptr = blocks + (row * blocks_per_row + b) * stride;
        const ushort d_raw = ushort(bptr[208]) | (ushort(bptr[209]) << 8u);
        const float d = float(as_type<half>(d_raw));

        device const uchar* ql = bptr;
        device const uchar* qh = bptr + 128u;
        device const uchar* scales_i8 = bptr + 192u;
        device const float* xbase = input + (b << 8u);

        uint y = 0u;
        uint ql_off = 0u;
        uint qh_off = 0u;
        uint sc_off = 0u;

        for (uint n = 0u; n < 2u; ++n) {   // (0..256).step_by(128)
            // d * signed int8 sub-scale, for the eight sub-blocks of this half.
            thread float dsc[8];
            for (uint t = 0u; t < 8u; ++t) {
                const int raw = int(scales_i8[sc_off + t]);
                dsc[t] = d * float(raw < 128 ? raw : raw - 256);
            }

            for (uint l = 0u; l < 32u; ++l) {
                const uint is = l >> 4u;   // l / 16
                const uint hq = uint(qh[qh_off + l]);
                const int q1 = int((ql[ql_off + l] & 0x0Fu) | ((hq & 3u) << 4u)) - 32;
                const int q2 = int((ql[ql_off + l + 32u] & 0x0Fu) | (((hq >> 2u) & 3u) << 4u)) - 32;
                const int q3 = int((uint(ql[ql_off + l]) >> 4u) | (((hq >> 4u) & 3u) << 4u)) - 32;
                const int q4 =
                    int((uint(ql[ql_off + l + 32u]) >> 4u) | (((hq >> 6u) & 3u) << 4u)) - 32;

                acc += dsc[is]      * float(q1) * xbase[y + l];
                acc += dsc[is + 2u] * float(q2) * xbase[y + l + 32u];
                acc += dsc[is + 4u] * float(q3) * xbase[y + l + 64u];
                acc += dsc[is + 6u] * float(q4) * xbase[y + l + 96u];
            }

            y += 128u;
            ql_off += 64u;
            qh_off += 32u;
            sc_off += 8u;
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

/// Metal MSL kernel: `Q8_K` GEMV — one simdgroup per output row.
///
/// Note: the super-block scale is **f32** (bytes 0-3), unlike the other K-quant
/// formats which use f16. ggml's `dequantize_row_q8_K` is element-sequential, so
/// this kernel needs no cursor walk.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q8K_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q8k(
    device const uchar* blocks [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& n_rows      [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k >> 8u;
    const uint stride = 292u;
    float acc = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        device const uchar* bptr = blocks + (row * blocks_per_row + b) * stride;
        // f32 scale at bytes 0-3 (little-endian).
        const uint d_bits = uint(bptr[0])
                          | (uint(bptr[1]) << 8u)
                          | (uint(bptr[2]) << 16u)
                          | (uint(bptr[3]) << 24u);
        const float d = as_type<float>(d_bits);
        device const float* xbase = input + (b << 8u);

        for (uint j = 0u; j < 256u; ++j) {
            const int raw = int(bptr[4u + j]);
            const int q = raw < 128 ? raw : raw - 256;  // signed int8
            acc += d * float(q) * xbase[j];
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Tests — host-only kernel source string assertions
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn k_quant_kernels_contain_entry_points() {
        use super::*;
        assert!(MSL_GEMV_Q2K_V1.contains("kernel void gemv_q2k"));
        assert!(MSL_GEMV_Q3K_V1.contains("kernel void gemv_q3k"));
        assert!(MSL_GEMV_Q4K_V1.contains("kernel void gemv_q4k"));
        assert!(MSL_GEMV_Q5K_V1.contains("kernel void gemv_q5k"));
        assert!(MSL_GEMV_Q6K_V1.contains("kernel void gemv_q6k"));
        assert!(MSL_GEMV_Q8K_V1.contains("kernel void gemv_q8k"));
        // ggml's `get_scale_min_k4` is present in the Q4_K / Q5_K sources.
        assert!(MSL_GEMV_Q4K_V1.contains("kq_scale_min_k4"));
        assert!(MSL_GEMV_Q5K_V1.contains("kq_scale_min_k4"));
        // Per-format block strides.
        assert!(MSL_GEMV_Q2K_V1.contains("stride = 84u"));
        assert!(MSL_GEMV_Q3K_V1.contains("stride = 110u"));
        assert!(MSL_GEMV_Q4K_V1.contains("stride = 144u"));
        assert!(MSL_GEMV_Q5K_V1.contains("stride = 176u"));
        assert!(MSL_GEMV_Q6K_V1.contains("stride = 210u"));
        assert!(MSL_GEMV_Q8K_V1.contains("stride = 292u"));
        // simdgroup reduction everywhere.
        assert!(MSL_GEMV_Q2K_V1.contains("simd_sum"));
        assert!(MSL_GEMV_Q8K_V1.contains("simd_sum"));
    }

    /// Structural guard against a regression to the pre-`core-gguf-K0`
    /// element-sequential walk: each kernel must carry the ggml cursor that the
    /// old, byte-incompatible layout lacked.
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn k_quant_kernels_use_the_ggml_walk() {
        use super::*;
        // Q2_K / Q3_K: `is++` under a shift stepping 0, 2, 4, 6 per 128-half.
        assert!(MSL_GEMV_Q2K_V1.contains("shift += 2u"));
        assert!(MSL_GEMV_Q3K_V1.contains("shift += 2u"));
        // Q3_K: kmask1/kmask2 aux[4] scale unpack, -32 bias, inverted hmask.
        assert!(MSL_GEMV_Q3K_V1.contains("kq_unpack_q3k_scales"));
        assert!(MSL_GEMV_Q3K_V1.contains("0x03030303u"));
        assert!(MSL_GEMV_Q3K_V1.contains("0x0f0f0f0fu"));
        assert!(MSL_GEMV_Q3K_V1.contains("m <<= 1u"));
        // Q4_K / Q5_K: full 6-bit scale for j < 4, and `is += 2` per 64-group.
        assert!(MSL_GEMV_Q4K_V1.contains("q[j] & 63u"));
        assert!(MSL_GEMV_Q5K_V1.contains("q[j] & 63u"));
        assert!(MSL_GEMV_Q4K_V1.contains("is += 2u"));
        assert!(MSL_GEMV_Q5K_V1.contains("is += 2u"));
        // Q5_K: the u1/u2 masks shift left by 2 per group.
        assert!(MSL_GEMV_Q5K_V1.contains("u1 <<= 2u"));
        assert!(MSL_GEMV_Q5K_V1.contains("u2 <<= 2u"));
        // Q6_K: four-lane interleave with sc[is+0,+2,+4,+6].
        assert!(MSL_GEMV_Q6K_V1.contains("dsc[is + 6u]"));
        // The old nibble-pair scale unpack must be gone from every source.
        for src in [
            MSL_GEMV_Q2K_V1,
            MSL_GEMV_Q3K_V1,
            MSL_GEMV_Q4K_V1,
            MSL_GEMV_Q5K_V1,
            MSL_GEMV_Q6K_V1,
        ] {
            assert!(!src.contains("kq_decode_6bit_scales"));
        }
    }
}
