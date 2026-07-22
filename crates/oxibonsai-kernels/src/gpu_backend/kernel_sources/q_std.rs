//! Metal MSL kernel sources for the standard GGUF `Q4_0` and `Q8_0` GEMV
//! operations (single-token decode path).
//!
//! These mirror the CUDA kernels in
//! `crates/oxibonsai-kernels/src/gpu_backend/cuda_q_std_kernels.rs` and are the
//! Metal counterparts of the scalar reference kernels in
//! `crates/oxibonsai-kernels/src/gemv_q4_0.rs` / `gemv_q8_0.rs`, which are the
//! parity oracle. The dequant arithmetic is bit-exact against the scalar path;
//! only the reduction order differs (simdgroup tree-sum vs. sequential scan),
//! so results agree to within f32 rounding.
//!
//! # Block layout (AoS, matches the `#[repr(C)]` core structs)
//!
//! **Q4_0** (18 bytes/block, 32 weights — `BlockQ4_0 { d: f16, qs: [u8; 16] }`):
//! ```text
//! [d_lo, d_hi, qs[0], ..., qs[15]]
//!  ^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
//!  FP16 LE     16 nibble bytes → 32 int4 weights (llama.cpp lo-hi split)
//! ```
//! Dequant: `w[j] = d * (nibble[j] - 8)`; elements 0..15 use the lower nibble of
//! bytes 0..15, elements 16..31 use the upper nibble.
//!
//! **Q8_0** (34 bytes/block, 32 weights — `BlockQ8_0 { d: f16, qs: [i8; 32] }`):
//! ```text
//! [d_lo, d_hi, qs[0], ..., qs[31]]
//!  ^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
//!  FP16 LE     32 signed int8 weights
//! ```
//! Dequant: `w[j] = d * qs[j]`.
//!
//! # Dispatch
//!
//! Grid:  `[ceil(n_rows / 8), 1, 1]` — 8 simdgroups per threadgroup, one row per simdgroup.
//! Block: `[256, 1, 1]` — 8 simdgroups × 32 lanes.
//!
//! Buffer indices:
//! - blocks (0, `uchar*`), input (1, `float*`), output (2, `float*`),
//!   n_rows (3, `uint`), k (4, `uint`).

/// Metal MSL kernel: `Q4_0` GEMV — one simdgroup per output row.
///
/// Each lane strides the 18-byte blocks of its row, decodes the FP16 scale and
/// the 16 nibble bytes (llama.cpp lo-hi split), accumulates a partial dot
/// product, and the simdgroup tree-reduces via `simd_sum`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q4_0_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q4_0(
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

    const uint blocks_per_row = k >> 5u;  // k / 32
    const uint stride = 18u;               // bytes per Q4_0 block
    float acc = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        device const uchar* bptr = blocks + (row * blocks_per_row + b) * stride;
        // FP16 scale at bytes 0-1 (little-endian).
        const ushort d_raw = ushort(bptr[0]) | (ushort(bptr[1]) << 8u);
        const float scale = float(as_type<half>(d_raw));

        device const float* xbase = input + (b << 5u);  // b * 32
        for (uint nb = 0u; nb < 16u; ++nb) {
            const uint byte = uint(bptr[2u + nb]);
            const float w0 = scale * (float(int(byte & 0x0Fu) - 8));       // element nb
            const float w1 = scale * (float(int((byte >> 4u) & 0x0Fu) - 8)); // element nb+16
            acc += w0 * xbase[nb] + w1 * xbase[nb + 16u];
        }
    }

    const float row_sum = simd_sum(acc);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

/// Metal MSL kernel: `Q8_0` GEMV — one simdgroup per output row.
///
/// Identical structure to `gemv_q4_0`; the block is 34 bytes and each of the 32
/// weights is a signed int8 scaled by the FP16 block scale.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q8_0_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q8_0(
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

    const uint blocks_per_row = k >> 5u;  // k / 32
    const uint stride = 34u;               // bytes per Q8_0 block
    float acc = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        device const uchar* bptr = blocks + (row * blocks_per_row + b) * stride;
        const ushort d_raw = ushort(bptr[0]) | (ushort(bptr[1]) << 8u);
        const float scale = float(as_type<half>(d_raw));

        device const float* xbase = input + (b << 5u);  // b * 32
        for (uint j = 0u; j < 32u; ++j) {
            const int raw = int(bptr[2u + j]);
            const int q = raw < 128 ? raw : raw - 256;  // int8 two's-complement
            acc += scale * float(q) * xbase[j];
        }
    }

    const float row_sum = simd_sum(acc);
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
    fn q_std_kernels_contain_entry_points() {
        use super::*;
        assert!(MSL_GEMV_Q4_0_V1.contains("kernel void gemv_q4_0"));
        assert!(MSL_GEMV_Q8_0_V1.contains("kernel void gemv_q8_0"));
        // Standard buffer-index annotations.
        assert!(MSL_GEMV_Q4_0_V1.contains("[[buffer(0)]]"));
        assert!(MSL_GEMV_Q4_0_V1.contains("[[buffer(4)]]"));
        // Block strides.
        assert!(MSL_GEMV_Q4_0_V1.contains("stride = 18u"));
        assert!(MSL_GEMV_Q8_0_V1.contains("stride = 34u"));
        // simdgroup reduction.
        assert!(MSL_GEMV_Q4_0_V1.contains("simd_sum"));
        assert!(MSL_GEMV_Q8_0_V1.contains("simd_sum"));
    }
}
