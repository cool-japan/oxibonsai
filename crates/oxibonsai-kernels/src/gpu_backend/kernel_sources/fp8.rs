//! Metal MSL kernel sources for FP8 (E4M3FN and E5M2) GEMV operations.
//!
//! Mirrors the CUDA FP8 kernels in `crates/oxibonsai-kernels/src/gpu_backend/cuda_fp8_kernels.rs`.
//!
//! # Block layout (AoS, 34 bytes/block — matches `BlockFP8E4M3` / `BlockFP8E5M2`)
//!
//! ```text
//! Block[i] = [q0, q1, ..., q31, scale_lo, scale_hi]
//!             ^^^^^^^^^^^^^^^^^ 32 FP8 bytes ^^^^   ^^ FP16 LE scale ^^
//! ```
//!
//! This matches the `#[repr(C)]` layout of `BlockFP8E4M3 { qs: [u8; 32], d: f16 }`:
//! `qs` occupies bytes 0-31, `d` (FP16) occupies bytes 32-33.
//!
//! # Dispatch
//!
//! Grid:  `[ceil(n_rows / 8), 1, 1]` — 8 simdgroups per threadgroup, one row per simdgroup
//! Block: `[256, 1, 1]` — 8 simdgroups × 32 lanes
//!
//! Buffer indices (scirs2-core naming convention):
//! - `"x"` → blocks (0, u8*)
//! - `"y"` → input  (1, float*)
//! - `"result"` → output (2, float*)
//! - `"n"` → n_rows (3, scalar uint)
//! - `"k"` → k      (4, scalar uint)

/// Metal MSL kernel: FP8 E4M3FN GEMV.
///
/// Format: s[7] exp[6:3] man[2:0], bias=7
/// - Normal: `(-1)^s * 2^(exp - 7) * (1 + man/8)`
/// - Denorm: `(-1)^s * 2^(-6) * (man/8)`
/// - NaN patterns (0x7F, 0xFF) → 0 for inference
///
/// One simdgroup per row. Each lane strides 32-block chunks.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_FP8_E4M3_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

// FP8 E4M3FN decode helper (bias=7, no infinity).
static inline float fp8_e4m3_to_float(uchar b) {
    // NaN patterns: 0x7F and 0xFF → return 0 for inference.
    if (b == 0x7Fu || b == 0xFFu) return 0.0f;
    const uint sign = (uint(b) >> 7u) & 1u;
    const uint exp  = (uint(b) >> 3u) & 15u;  // 4-bit exponent
    const uint mant = uint(b) & 7u;            // 3-bit mantissa
    float val;
    if (exp == 0u) {
        // Denormal: 2^(-6) * (mant / 8)
        val = float(mant) * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else {
        // Normal: 2^(exp - 7) * (1 + mant/8)
        // Assemble as IEEE-754 f32: ((exp - 7 + 127) << 23) | (mant << 20)
        const uint bits = ((exp - 7u + 127u) << 23u) | (mant << 20u);
        val = as_type<float>(bits);
    }
    return sign ? -val : val;
}

kernel void gemv_fp8_e4m3(
    device const uchar* blocks_raw [[buffer(0)]],
    device const float* input      [[buffer(1)]],
    device float* output           [[buffer(2)]],
    constant uint& n_rows          [[buffer(3)]],
    constant uint& k               [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k >> 5u;  // k / 32
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const uint base_byte = block_idx * 34u;

        // Scale at bytes 32-33 (after the 32 FP8 weight bytes).
        const ushort scale_bits =
              ushort(blocks_raw[base_byte + 32u])
            | (ushort(blocks_raw[base_byte + 33u]) << 8u);
        const float scale = float(as_type<half>(scale_bits));

        // Dot product: 32 FP8 weights at bytes 0..31 with 32 float inputs.
        const uint inp_base = b * 32u;
        float block_sum = 0.0f;
        for (uint w = 0u; w < 32u; ++w) {
            block_sum += fp8_e4m3_to_float(blocks_raw[base_byte + w]) * input[inp_base + w];
        }
        local_sum += scale * block_sum;
    }

    // Sum across all 32 lanes within the simdgroup.
    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

/// Metal MSL kernel: FP8 E5M2 GEMV.
///
/// Format: s[7] exp[6:2] man[1:0], bias=15
/// - Normal: `(-1)^s * 2^(exp - 15) * (1 + man/4)`
/// - Denorm: `(-1)^s * 2^(-14) * (man/4)`
/// - Inf/NaN (exp=31) → 0 for inference
///
/// Identical structure to `gemv_fp8_e4m3`, decode path differs only.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_FP8_E5M2_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

// FP8 E5M2 decode helper (bias=15, with infinity).
static inline float fp8_e5m2_to_float(uchar b) {
    const uint exp  = (uint(b) >> 2u) & 31u;  // 5-bit exponent
    const uint mant = uint(b) & 3u;            // 2-bit mantissa
    // Inf / NaN: exp = 31 → 0 for inference.
    if (exp == 31u) return 0.0f;
    const uint sign = (uint(b) >> 7u) & 1u;
    float val;
    if (exp == 0u) {
        // Denormal: 2^(-14) * (mant / 4)
        val = float(mant) * (1.0f / 4.0f) * (1.0f / 16384.0f);
    } else {
        // Normal: 2^(exp - 15) * (1 + mant/4)
        // Assemble as IEEE-754 f32: ((exp - 15 + 127) << 23) | (mant << 21)
        const uint bits = ((exp - 15u + 127u) << 23u) | (mant << 21u);
        val = as_type<float>(bits);
    }
    return sign ? -val : val;
}

kernel void gemv_fp8_e5m2(
    device const uchar* blocks_raw [[buffer(0)]],
    device const float* input      [[buffer(1)]],
    device float* output           [[buffer(2)]],
    constant uint& n_rows          [[buffer(3)]],
    constant uint& k               [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k >> 5u;  // k / 32
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const uint base_byte = block_idx * 34u;

        const ushort scale_bits =
              ushort(blocks_raw[base_byte + 32u])
            | (ushort(blocks_raw[base_byte + 33u]) << 8u);
        const float scale = float(as_type<half>(scale_bits));

        const uint inp_base = b * 32u;
        float block_sum = 0.0f;
        for (uint w = 0u; w < 32u; ++w) {
            block_sum += fp8_e5m2_to_float(blocks_raw[base_byte + w]) * input[inp_base + w];
        }
        local_sum += scale * block_sum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Tests — host-only kernel source string assertions
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn fp8_kernels_contain_entry_points() {
        use super::*;
        assert!(MSL_GEMV_FP8_E4M3_V1.contains("kernel void gemv_fp8_e4m3"));
        assert!(MSL_GEMV_FP8_E5M2_V1.contains("kernel void gemv_fp8_e5m2"));
        // Decode helpers
        assert!(MSL_GEMV_FP8_E4M3_V1.contains("fp8_e4m3_to_float"));
        assert!(MSL_GEMV_FP8_E5M2_V1.contains("fp8_e5m2_to_float"));
        // Standard buffer-index annotations
        assert!(MSL_GEMV_FP8_E4M3_V1.contains("[[buffer(0)]]"));
        assert!(MSL_GEMV_FP8_E4M3_V1.contains("[[buffer(4)]]"));
        // Block size constant
        assert!(MSL_GEMV_FP8_E4M3_V1.contains("* 34u"));
        assert!(MSL_GEMV_FP8_E5M2_V1.contains("* 34u"));
    }
}
