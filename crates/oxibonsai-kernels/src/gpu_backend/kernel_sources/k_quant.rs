//! Metal MSL kernel sources for K-quant GEMV operations
//! (`Q2_K` / `Q3_K` / `Q4_K` / `Q5_K` / `Q6_K` / `Q8_K`), single-token decode path.
//!
//! These mirror the CUDA kernels in
//! `crates/oxibonsai-kernels/src/gpu_backend/cuda_k_quant_kernels.rs` and are the
//! Metal counterparts of the scalar reference kernels
//! (`crates/oxibonsai-kernels/src/gemv_q{2,3,4,5,6,8}k.rs`), which are the parity
//! oracle. All formats use `QK_K = 256` weights per super-block.
//!
//! The dequant arithmetic matches the scalar path; only the reduction order
//! differs (simdgroup tree-sum vs. sequential scan), so results agree to within
//! f32 rounding.
//!
//! # Block layouts (AoS, matching the `#[repr(C)]` core structs; offsets in bytes)
//!
//! - **Q2_K** (84): `[scales:16 @0][qs:64 @16][d:f16 @80][dmin:f16 @82]`
//!   16 sub-blocks × 16 weights. `sc = scales[sub]&0xF`, `mn = scales[sub]>>4`.
//!   `q ∈ [0,3]` (4/byte, LSB first). Dequant: `d*sc*q - dmin*mn`.
//! - **Q3_K** (110): `[hmask:32 @0][qs:64 @32][scales:12 @96][d:f16 @108]`
//!   `q3 = lo2 | (hi<<2)`, signed `q3-4`. `signed_sc = nibble-8`.
//!   Dequant: `d*signed_sc*(q3-4)`.
//! - **Q4_K** (144): `[d:f16 @0][dmin:f16 @2][scales:12 @4][qs:128 @16]`
//!   8 sub-blocks × 32 weights, 4-bit. 6-bit scale/min decode.
//!   Dequant: `d*sc[sub]*q - dmin*mn[sub]`.
//! - **Q5_K** (176): `[d:f16 @0][dmin:f16 @2][scales:12 @4][qh:32 @16][qs:128 @48]`
//!   `q5 = nibble | (high_bit<<4)`, range `[0,31]`.
//!   Dequant: `d*sc[sub]*q5 - dmin*mn[sub]`.
//! - **Q6_K** (210): `[ql:128 @0][qh:64 @128][scales:16 i8 @192][d:f16 @208]`
//!   16 sub-blocks × 16 weights. `q6 = nibble | (hi2<<4)`, centered `q6-32`.
//!   Dequant: `d*scales_i8[sub]*(q6-32)`.
//! - **Q8_K** (292): `[d:f32 @0][qs:256 i8 @4][bsums:32 @260]`
//!   `d` is **f32** (not f16!). Dequant: `d*qs[i]`. `bsums` unused by GEMV.
//!
//! # Dispatch (identical for all six kernels)
//!
//! Grid:  `[ceil(n_rows / 8), 1, 1]` — 8 simdgroups per threadgroup, one row per simdgroup.
//! Block: `[256, 1, 1]` — 8 simdgroups × 32 lanes.
//! `k` must be a positive multiple of 256.
//!
//! Buffer indices: blocks (0), input (1), output (2), n_rows (3), k (4).

/// Metal MSL kernel: `Q2_K` GEMV — one simdgroup per output row.
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
        device const float* xbase = input + (b << 8u);  // b * 256

        for (uint sub = 0u; sub < 16u; ++sub) {
            const uint sc_byte = uint(bptr[sub]);
            const float sub_sc = float(sc_byte & 0x0Fu);
            const float sub_mn = float((sc_byte >> 4u) & 0x0Fu);
            const uint w_base = sub * 16u;
            const uint q_base = sub * 4u;

            float sub_acc = 0.0f;
            float sub_xsum = 0.0f;
            for (uint qb = 0u; qb < 4u; ++qb) {
                const uint byte_val = uint(bptr[16u + q_base + qb]);
                for (uint bit = 0u; bit < 4u; ++bit) {
                    const uint wi = w_base + qb * 4u + bit;
                    const float q = float((byte_val >> (bit * 2u)) & 0x3u);
                    const float x = xbase[wi];
                    sub_acc  += q * x;
                    sub_xsum += x;
                }
            }
            acc += d * sub_sc * sub_acc - dmin * sub_mn * sub_xsum;
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

/// Metal MSL kernel: `Q3_K` GEMV — one simdgroup per output row.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q3K_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

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
        const float d = float(as_type<half>(d_raw));
        device const float* xbase = input + (b << 8u);

        for (uint sub = 0u; sub < 16u; ++sub) {
            const uint sc_byte = uint(bptr[96u + sub / 2u]);
            const uint nibble  = (sub & 1u) == 0u
                                 ? (sc_byte & 0x0Fu)
                                 : ((sc_byte >> 4u) & 0x0Fu);
            const float signed_sc = float(int(nibble)) - 8.0f;
            const uint w_base = sub * 16u;

            float sub_acc = 0.0f;
            for (uint j = 0u; j < 16u; ++j) {
                const uint wi  = w_base + j;
                const uint hi  = (uint(bptr[wi >> 3u]) >> (wi & 7u)) & 0x1u;
                const uint lo2 = (uint(bptr[32u + (wi >> 2u)]) >> ((wi & 3u) * 2u)) & 0x3u;
                const int q3_code   = int(lo2 | (hi << 2u));
                const int q3_signed = q3_code - 4;
                sub_acc += float(q3_signed) * xbase[wi];
            }
            acc += d * signed_sc * sub_acc;
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

/// Metal MSL kernel: `Q4_K` GEMV — one simdgroup per output row.
///
/// Uses the shared `kq_decode_6bit_scales` helper (6-bit sub-block scale/min).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q4K_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Decode the 12-byte scales array into 8 x 6-bit sc and mn (Q4_K / Q5_K layout).
static void kq_decode_6bit_scales(device const uchar* s,
                                  thread uchar* sc_out,
                                  thread uchar* mn_out) {
    sc_out[0] = s[0] & 0x0Fu;  sc_out[1] = (s[0] >> 4u) & 0x0Fu;
    sc_out[2] = s[1] & 0x0Fu;  sc_out[3] = (s[1] >> 4u) & 0x0Fu;
    sc_out[4] = s[2] & 0x0Fu;  sc_out[5] = (s[2] >> 4u) & 0x0Fu;
    sc_out[6] = s[3] & 0x0Fu;  sc_out[7] = (s[3] >> 4u) & 0x0Fu;
    mn_out[0] = s[4] & 0x0Fu;  mn_out[1] = (s[4] >> 4u) & 0x0Fu;
    mn_out[2] = s[5] & 0x0Fu;  mn_out[3] = (s[5] >> 4u) & 0x0Fu;
    mn_out[4] = s[6] & 0x0Fu;  mn_out[5] = (s[6] >> 4u) & 0x0Fu;
    mn_out[6] = s[7] & 0x0Fu;  mn_out[7] = (s[7] >> 4u) & 0x0Fu;
    sc_out[0] |= ((s[8] >> 0u) & 0x03u) << 4u;
    sc_out[1] |= ((s[8] >> 2u) & 0x03u) << 4u;
    sc_out[2] |= ((s[8] >> 4u) & 0x03u) << 4u;
    sc_out[3] |= ((s[8] >> 6u) & 0x03u) << 4u;
    sc_out[4] |= ((s[9] >> 0u) & 0x03u) << 4u;
    sc_out[5] |= ((s[9] >> 2u) & 0x03u) << 4u;
    sc_out[6] |= ((s[9] >> 4u) & 0x03u) << 4u;
    sc_out[7] |= ((s[9] >> 6u) & 0x03u) << 4u;
    mn_out[0] |= ((s[10] >> 0u) & 0x03u) << 4u;
    mn_out[1] |= ((s[10] >> 2u) & 0x03u) << 4u;
    mn_out[2] |= ((s[10] >> 4u) & 0x03u) << 4u;
    mn_out[3] |= ((s[10] >> 6u) & 0x03u) << 4u;
    mn_out[4] |= ((s[11] >> 0u) & 0x03u) << 4u;
    mn_out[5] |= ((s[11] >> 2u) & 0x03u) << 4u;
    mn_out[6] |= ((s[11] >> 4u) & 0x03u) << 4u;
    mn_out[7] |= ((s[11] >> 6u) & 0x03u) << 4u;
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

        thread uchar sc[8];
        thread uchar mn[8];
        kq_decode_6bit_scales(bptr + 4u, sc, mn);

        device const float* xbase = input + (b << 8u);
        for (uint sub = 0u; sub < 8u; ++sub) {
            const float sc_f = float(sc[sub]);
            const float mn_f = float(mn[sub]);
            device const uchar* qs_sub = bptr + 16u + sub * 16u;
            device const float* x_sub = xbase + sub * 32u;

            float sub_acc  = 0.0f;
            float sub_xsum = 0.0f;
            for (uint nb = 0u; nb < 16u; ++nb) {
                const uint byte_val = uint(qs_sub[nb]);
                const float q0 = float(byte_val & 0x0Fu);
                const float q1 = float((byte_val >> 4u) & 0x0Fu);
                const float x0 = x_sub[nb * 2u];
                const float x1 = x_sub[nb * 2u + 1u];
                sub_acc  += q0 * x0 + q1 * x1;
                sub_xsum += x0 + x1;
            }
            acc += d * sc_f * sub_acc - dmin * mn_f * sub_xsum;
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

/// Metal MSL kernel: `Q5_K` GEMV — one simdgroup per output row.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q5K_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

static void kq_decode_6bit_scales(device const uchar* s,
                                  thread uchar* sc_out,
                                  thread uchar* mn_out) {
    sc_out[0] = s[0] & 0x0Fu;  sc_out[1] = (s[0] >> 4u) & 0x0Fu;
    sc_out[2] = s[1] & 0x0Fu;  sc_out[3] = (s[1] >> 4u) & 0x0Fu;
    sc_out[4] = s[2] & 0x0Fu;  sc_out[5] = (s[2] >> 4u) & 0x0Fu;
    sc_out[6] = s[3] & 0x0Fu;  sc_out[7] = (s[3] >> 4u) & 0x0Fu;
    mn_out[0] = s[4] & 0x0Fu;  mn_out[1] = (s[4] >> 4u) & 0x0Fu;
    mn_out[2] = s[5] & 0x0Fu;  mn_out[3] = (s[5] >> 4u) & 0x0Fu;
    mn_out[4] = s[6] & 0x0Fu;  mn_out[5] = (s[6] >> 4u) & 0x0Fu;
    mn_out[6] = s[7] & 0x0Fu;  mn_out[7] = (s[7] >> 4u) & 0x0Fu;
    sc_out[0] |= ((s[8] >> 0u) & 0x03u) << 4u;
    sc_out[1] |= ((s[8] >> 2u) & 0x03u) << 4u;
    sc_out[2] |= ((s[8] >> 4u) & 0x03u) << 4u;
    sc_out[3] |= ((s[8] >> 6u) & 0x03u) << 4u;
    sc_out[4] |= ((s[9] >> 0u) & 0x03u) << 4u;
    sc_out[5] |= ((s[9] >> 2u) & 0x03u) << 4u;
    sc_out[6] |= ((s[9] >> 4u) & 0x03u) << 4u;
    sc_out[7] |= ((s[9] >> 6u) & 0x03u) << 4u;
    mn_out[0] |= ((s[10] >> 0u) & 0x03u) << 4u;
    mn_out[1] |= ((s[10] >> 2u) & 0x03u) << 4u;
    mn_out[2] |= ((s[10] >> 4u) & 0x03u) << 4u;
    mn_out[3] |= ((s[10] >> 6u) & 0x03u) << 4u;
    mn_out[4] |= ((s[11] >> 0u) & 0x03u) << 4u;
    mn_out[5] |= ((s[11] >> 2u) & 0x03u) << 4u;
    mn_out[6] |= ((s[11] >> 4u) & 0x03u) << 4u;
    mn_out[7] |= ((s[11] >> 6u) & 0x03u) << 4u;
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

        thread uchar sc[8];
        thread uchar mn[8];
        kq_decode_6bit_scales(bptr + 4u, sc, mn);

        device const uchar* qh = bptr + 16u;
        device const uchar* qs = bptr + 48u;
        device const float* xbase = input + (b << 8u);

        for (uint sub = 0u; sub < 8u; ++sub) {
            const float sc_f = float(sc[sub]);
            const float mn_f = float(mn[sub]);
            device const uchar* qs_sub = qs + sub * 16u;
            device const float* x_sub = xbase + sub * 32u;

            float sub_acc  = 0.0f;
            float sub_xsum = 0.0f;
            for (uint nb = 0u; nb < 16u; ++nb) {
                const uint wi0 = sub * 32u + nb * 2u;
                const uint wi1 = wi0 + 1u;
                const uint hi0 = (uint(qh[wi0 >> 3u]) >> (wi0 & 7u)) & 0x1u;
                const uint hi1 = (uint(qh[wi1 >> 3u]) >> (wi1 & 7u)) & 0x1u;
                const uint byte_val = uint(qs_sub[nb]);
                const uint lo0 = byte_val & 0x0Fu;
                const uint lo1 = (byte_val >> 4u) & 0x0Fu;
                const float q0 = float(lo0 | (hi0 << 4u));
                const float q1 = float(lo1 | (hi1 << 4u));
                const float x0 = x_sub[nb * 2u];
                const float x1 = x_sub[nb * 2u + 1u];
                sub_acc  += q0 * x0 + q1 * x1;
                sub_xsum += x0 + x1;
            }
            acc += d * sc_f * sub_acc - dmin * mn_f * sub_xsum;
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

/// Metal MSL kernel: `Q6_K` GEMV — one simdgroup per output row.
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

        for (uint sub = 0u; sub < 16u; ++sub) {
            const int sc_raw = int(scales_i8[sub]);
            const float sc = float(sc_raw < 128 ? sc_raw : sc_raw - 256);  // signed int8
            const uint w_base = sub * 16u;

            float sub_acc = 0.0f;
            for (uint j = 0u; j < 16u; ++j) {
                const uint wi = w_base + j;
                const uint nibble = (uint(ql[wi >> 1u]) >> ((wi & 1u) * 4u)) & 0x0Fu;
                const uint hi2    = (uint(qh[wi >> 2u]) >> ((wi & 3u) * 2u)) & 0x03u;
                const int q6        = int(nibble | (hi2 << 4u));
                const int q6_signed = q6 - 32;
                sub_acc += float(q6_signed) * xbase[wi];
            }
            acc += d * sc * sub_acc;
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) output[row] = row_sum;
}
"#;

/// Metal MSL kernel: `Q8_K` GEMV — one simdgroup per output row.
///
/// Note: the super-block scale is **f32** (bytes 0-3), unlike the other K-quant
/// formats which use f16.
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
        // The 6-bit scale helper is present in the Q4_K / Q5_K sources.
        assert!(MSL_GEMV_Q4K_V1.contains("kq_decode_6bit_scales"));
        assert!(MSL_GEMV_Q5K_V1.contains("kq_decode_6bit_scales"));
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
}
