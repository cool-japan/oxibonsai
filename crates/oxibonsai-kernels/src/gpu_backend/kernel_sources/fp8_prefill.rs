//! Metal MSL kernel sources for FP8 (E4M3FN and E5M2) **batch prefill** GEMM operations.
//!
//! Mirrors the CUDA FP8 prefill kernels in
//! `crates/oxibonsai-kernels/src/gpu_backend/cuda_fp8_prefill_kernels.rs`.
//!
//! # Block layout (AoS, 34 bytes/block — matches `BlockFP8E4M3` / `BlockFP8E5M2`)
//!
//! ```text
//! Block[i] = [q0, q1, ..., q31, scale_lo, scale_hi]
//!             ^^^^^^^^^^^^^^^^^ 32 FP8 bytes ^^^^   ^^ FP16 LE scale ^^
//! ```
//!
//! Weights at bytes 0-31, FP16 scale at bytes 32-33. Scale access:
//! `as_type<half>(blocks_raw[base + 32u] | (blocks_raw[base + 33u] << 8u))`.
//!
//! # Batch tensor layout
//!
//! All batch inputs/outputs use **column-major** layout: `buf[col * dim + element]`
//! where `col` is the batch/token index. This matches the Q1 / TQ2 V7 batch
//! GEMM convention in `prefill.rs`.
//!
//! # Dispatch
//!
//! Grid:  `[ceil(n_rows / 8), 1, 1]` — 8 simdgroups per threadgroup, one row per simdgroup
//! Block: `[256, 1, 1]` — 8 simdgroups × 32 lanes
//!
//! # Buffer indices (match Q1 V7 / TQ2 V7 convention)
//!
//! Batch GEMM:
//! - `buffer(0)` = blocks_raw  (u8, FP8 weight data, AoS layout)
//! - `buffer(1)` = inputs      (f32, batch × k, column-major)
//! - `buffer(2)` = outputs     (f32, batch × n_rows, column-major)
//! - `buffer(3)` = n_rows      (u32)
//! - `buffer(4)` = batch_size  (u32)
//! - `buffer(5)` = k           (u32)
//! - `buffer(6)` = residual    (f32, batch × n_rows, column-major; only for `_residual` variant)
//!
//! Single-token GEMV (`gemv_*_pf` variants — for use inside the sequential
//! attention inner loop of batch prefill):
//! - `buffer(0)` = blocks_raw  (u8)
//! - `buffer(1)` = input       (f32, k)
//! - `buffer(2)` = output      (f32, n_rows)
//! - `buffer(3)` = n_rows      (u32)
//! - `buffer(4)` = k           (u32)
//!
//! Cap-of-8 outer loop: every batch GEMM iterates `for col_base in 0..batch_size step 8u`
//! so arbitrary `batch_size` values are handled correctly (matches the issue
//! tracked in `kernel_pattern_capof8` memory and resolved across all V7 kernels).

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 1 — gemm_fp8_e4m3
// Batch FP8 E4M3 GEMM. Accumulates into outputs with +=.
// ═══════════════════════════════════════════════════════════════════════════

/// FP8 E4M3 batch GEMM (cap-of-8, column-major I/O, AoS 34-byte blocks).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_FP8_E4M3_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

// FP8 E4M3FN decode (bias=7, no infinity; NaN patterns 0x7F/0xFF → 0).
static inline float pf_fp8_e4m3_to_float(uchar b) {
    if (b == 0x7Fu || b == 0xFFu) return 0.0f;
    const uint sign = (uint(b) >> 7u) & 1u;
    const uint exp  = (uint(b) >> 3u) & 15u;
    const uint mant = uint(b) & 7u;
    float val;
    if (exp == 0u) {
        val = float(mant) * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else {
        const uint bits = ((exp - 7u + 127u) << 23u) | (mant << 20u);
        val = as_type<float>(bits);
    }
    return sign ? -val : val;
}

kernel void gemm_fp8_e4m3(
    device const uchar* blocks_raw [[buffer(0)]],
    device const float* inputs     [[buffer(1)]],
    device float* outputs          [[buffer(2)]],
    constant uint& n_rows          [[buffer(3)]],
    constant uint& batch_size      [[buffer(4)]],
    constant uint& k               [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;
    const uint blocks_per_row = k >> 5u;

    for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {
        const uint cols_remaining = batch_size - col_base;
        const uint cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8] = {0,0,0,0,0,0,0,0};

        for (uint b = lane; b < blocks_per_row; b += 32u) {
            const uint block_idx = row * blocks_per_row + b;
            const uint base_byte = block_idx * 34u;
            const ushort scale_bits =
                  ushort(blocks_raw[base_byte + 32u])
                | (ushort(blocks_raw[base_byte + 33u]) << 8u);
            const float scale = float(as_type<half>(scale_bits));
            const uint inp_base = b * 32u;

            for (uint cc = 0u; cc < cols; cc++) {
                device const float* xbase = inputs + (col_base + cc) * k + inp_base;
                float bsum = 0.0f;
                for (uint w = 0u; w < 32u; ++w) {
                    bsum += pf_fp8_e4m3_to_float(blocks_raw[base_byte + w]) * xbase[w];
                }
                col_sums[cc] += scale * bsum;
            }
        }

        for (uint cc = 0u; cc < cols; cc++) {
            float row_sum = simd_sum(col_sums[cc]);
            if (lane == 0u) outputs[(col_base + cc) * n_rows + row] += row_sum;
        }
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 2 — gemm_fp8_e4m3_residual
// Batch FP8 E4M3 GEMM + fused in-place residual add.
// outputs[col*n_rows+row] = residual[col*n_rows+row] + sum
// ═══════════════════════════════════════════════════════════════════════════

/// FP8 E4M3 batch GEMM with fused residual add.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_FP8_E4M3_RESIDUAL_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float pf_fp8_e4m3_to_float(uchar b) {
    if (b == 0x7Fu || b == 0xFFu) return 0.0f;
    const uint sign = (uint(b) >> 7u) & 1u;
    const uint exp  = (uint(b) >> 3u) & 15u;
    const uint mant = uint(b) & 7u;
    float val;
    if (exp == 0u) {
        val = float(mant) * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else {
        const uint bits = ((exp - 7u + 127u) << 23u) | (mant << 20u);
        val = as_type<float>(bits);
    }
    return sign ? -val : val;
}

kernel void gemm_fp8_e4m3_residual(
    device const uchar* blocks_raw [[buffer(0)]],
    device const float* inputs     [[buffer(1)]],
    device float* outputs          [[buffer(2)]],
    constant uint& n_rows          [[buffer(3)]],
    constant uint& batch_size      [[buffer(4)]],
    constant uint& k               [[buffer(5)]],
    device const float* residual   [[buffer(6)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;
    const uint blocks_per_row = k >> 5u;

    for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {
        const uint cols_remaining = batch_size - col_base;
        const uint cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8] = {0,0,0,0,0,0,0,0};

        for (uint b = lane; b < blocks_per_row; b += 32u) {
            const uint block_idx = row * blocks_per_row + b;
            const uint base_byte = block_idx * 34u;
            const ushort scale_bits =
                  ushort(blocks_raw[base_byte + 32u])
                | (ushort(blocks_raw[base_byte + 33u]) << 8u);
            const float scale = float(as_type<half>(scale_bits));
            const uint inp_base = b * 32u;

            for (uint cc = 0u; cc < cols; cc++) {
                device const float* xbase = inputs + (col_base + cc) * k + inp_base;
                float bsum = 0.0f;
                for (uint w = 0u; w < 32u; ++w) {
                    bsum += pf_fp8_e4m3_to_float(blocks_raw[base_byte + w]) * xbase[w];
                }
                col_sums[cc] += scale * bsum;
            }
        }

        for (uint cc = 0u; cc < cols; cc++) {
            float row_sum = simd_sum(col_sums[cc]);
            if (lane == 0u) {
                const uint idx = (col_base + cc) * n_rows + row;
                outputs[idx] = residual[idx] + row_sum;
            }
        }
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 3 — fused_gate_up_swiglu_gemm_fp8_e4m3
// Batch fused gate+up FP8 E4M3 GEMM with SwiGLU epilogue.
//
// blocks pointer covers 2*n_ffn_rows rows in AoS layout:
//   gate rows 0..n_ffn_rows-1, up rows n_ffn_rows..2*n_ffn_rows-1.
// For each (row r, col c):
//   outputs[c * n_ffn_rows + r] = SiLU(gate_sum(r,c)) * up_sum(r,c)
// Output buffer must be zeroed before calling (kernel writes, not +=).
// ═══════════════════════════════════════════════════════════════════════════

/// FP8 E4M3 fused gate+up GEMM with SwiGLU epilogue.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float pf_fp8_e4m3_to_float(uchar b) {
    if (b == 0x7Fu || b == 0xFFu) return 0.0f;
    const uint sign = (uint(b) >> 7u) & 1u;
    const uint exp  = (uint(b) >> 3u) & 15u;
    const uint mant = uint(b) & 7u;
    float val;
    if (exp == 0u) {
        val = float(mant) * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else {
        const uint bits = ((exp - 7u + 127u) << 23u) | (mant << 20u);
        val = as_type<float>(bits);
    }
    return sign ? -val : val;
}

// SiLU activation: x * sigmoid(x)
static inline float pf_silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void fused_gate_up_swiglu_gemm_fp8_e4m3(
    device const uchar* blocks_raw [[buffer(0)]],
    device const float* inputs     [[buffer(1)]],
    device float* outputs          [[buffer(2)]],
    constant uint& n_ffn_rows      [[buffer(3)]],
    constant uint& batch_size      [[buffer(4)]],
    constant uint& k               [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_ffn_rows) return;
    const uint blocks_per_row = k >> 5u;
    const uint up_row_offset  = n_ffn_rows * blocks_per_row;

    for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {
        const uint cols_remaining = batch_size - col_base;
        const uint cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8] = {0,0,0,0,0,0,0,0};
        float up_sums[8]   = {0,0,0,0,0,0,0,0};

        for (uint b = lane; b < blocks_per_row; b += 32u) {
            const uint gate_block_idx = row * blocks_per_row + b;
            const uint up_block_idx   = up_row_offset + row * blocks_per_row + b;
            const uint gbase = gate_block_idx * 34u;
            const uint ubase = up_block_idx * 34u;

            const ushort gd_raw =
                  ushort(blocks_raw[gbase + 32u])
                | (ushort(blocks_raw[gbase + 33u]) << 8u);
            const float gscale = float(as_type<half>(gd_raw));

            const ushort ud_raw =
                  ushort(blocks_raw[ubase + 32u])
                | (ushort(blocks_raw[ubase + 33u]) << 8u);
            const float uscale = float(as_type<half>(ud_raw));

            const uint inp_base = b * 32u;
            for (uint cc = 0u; cc < cols; cc++) {
                device const float* xbase = inputs + (col_base + cc) * k + inp_base;
                float gsum = 0.0f;
                float usum = 0.0f;
                for (uint w = 0u; w < 32u; ++w) {
                    const float x = xbase[w];
                    gsum += pf_fp8_e4m3_to_float(blocks_raw[gbase + w]) * x;
                    usum += pf_fp8_e4m3_to_float(blocks_raw[ubase + w]) * x;
                }
                gate_sums[cc] += gscale * gsum;
                up_sums[cc]   += uscale * usum;
            }
        }

        for (uint cc = 0u; cc < cols; cc++) {
            float gs = simd_sum(gate_sums[cc]);
            float us = simd_sum(up_sums[cc]);
            if (lane == 0u) {
                outputs[(col_base + cc) * n_ffn_rows + row] = pf_silu(gs) * us;
            }
        }
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 4 — gemv_fp8_e4m3_pf
// Single-token FP8 E4M3 GEMV — for the sequential attention inner loop
// inside batch prefill (output projection per-row pass, etc.).
//
// Identical math to MSL_GEMV_FP8_E4M3_V1 (Phase 27), provided here under
// the unified `_pf` naming alongside the batch kernels so they all sit in
// one compiled library.
// ═══════════════════════════════════════════════════════════════════════════

/// FP8 E4M3 single-token GEMV (prefill family).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_FP8_E4M3_PF_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float pf_fp8_e4m3_to_float(uchar b) {
    if (b == 0x7Fu || b == 0xFFu) return 0.0f;
    const uint sign = (uint(b) >> 7u) & 1u;
    const uint exp  = (uint(b) >> 3u) & 15u;
    const uint mant = uint(b) & 7u;
    float val;
    if (exp == 0u) {
        val = float(mant) * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else {
        const uint bits = ((exp - 7u + 127u) << 23u) | (mant << 20u);
        val = as_type<float>(bits);
    }
    return sign ? -val : val;
}

kernel void gemv_fp8_e4m3_pf(
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
    const uint blocks_per_row = k >> 5u;

    float local_sum = 0.0f;
    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint base_byte = (row * blocks_per_row + b) * 34u;
        const ushort scale_bits =
              ushort(blocks_raw[base_byte + 32u])
            | (ushort(blocks_raw[base_byte + 33u]) << 8u);
        const float scale = float(as_type<half>(scale_bits));
        const uint inp_base = b * 32u;
        float bsum = 0.0f;
        for (uint w = 0u; w < 32u; ++w) {
            bsum += pf_fp8_e4m3_to_float(blocks_raw[base_byte + w]) * input[inp_base + w];
        }
        local_sum += scale * bsum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) output[row] = row_sum;
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 5 — gemm_fp8_e5m2
// Batch FP8 E5M2 GEMM. Accumulates with +=.
// ═══════════════════════════════════════════════════════════════════════════

/// FP8 E5M2 batch GEMM.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_FP8_E5M2_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

// FP8 E5M2 decode (bias=15, exp=31 → 0).
static inline float pf_fp8_e5m2_to_float(uchar b) {
    const uint exp  = (uint(b) >> 2u) & 31u;
    const uint mant = uint(b) & 3u;
    if (exp == 31u) return 0.0f;
    const uint sign = (uint(b) >> 7u) & 1u;
    float val;
    if (exp == 0u) {
        val = float(mant) * (1.0f / 4.0f) * (1.0f / 16384.0f);
    } else {
        const uint bits = ((exp - 15u + 127u) << 23u) | (mant << 21u);
        val = as_type<float>(bits);
    }
    return sign ? -val : val;
}

kernel void gemm_fp8_e5m2(
    device const uchar* blocks_raw [[buffer(0)]],
    device const float* inputs     [[buffer(1)]],
    device float* outputs          [[buffer(2)]],
    constant uint& n_rows          [[buffer(3)]],
    constant uint& batch_size      [[buffer(4)]],
    constant uint& k               [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;
    const uint blocks_per_row = k >> 5u;

    for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {
        const uint cols_remaining = batch_size - col_base;
        const uint cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8] = {0,0,0,0,0,0,0,0};

        for (uint b = lane; b < blocks_per_row; b += 32u) {
            const uint block_idx = row * blocks_per_row + b;
            const uint base_byte = block_idx * 34u;
            const ushort scale_bits =
                  ushort(blocks_raw[base_byte + 32u])
                | (ushort(blocks_raw[base_byte + 33u]) << 8u);
            const float scale = float(as_type<half>(scale_bits));
            const uint inp_base = b * 32u;

            for (uint cc = 0u; cc < cols; cc++) {
                device const float* xbase = inputs + (col_base + cc) * k + inp_base;
                float bsum = 0.0f;
                for (uint w = 0u; w < 32u; ++w) {
                    bsum += pf_fp8_e5m2_to_float(blocks_raw[base_byte + w]) * xbase[w];
                }
                col_sums[cc] += scale * bsum;
            }
        }

        for (uint cc = 0u; cc < cols; cc++) {
            float row_sum = simd_sum(col_sums[cc]);
            if (lane == 0u) outputs[(col_base + cc) * n_rows + row] += row_sum;
        }
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 6 — gemm_fp8_e5m2_residual
// ═══════════════════════════════════════════════════════════════════════════

/// FP8 E5M2 batch GEMM with fused residual add.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_FP8_E5M2_RESIDUAL_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float pf_fp8_e5m2_to_float(uchar b) {
    const uint exp  = (uint(b) >> 2u) & 31u;
    const uint mant = uint(b) & 3u;
    if (exp == 31u) return 0.0f;
    const uint sign = (uint(b) >> 7u) & 1u;
    float val;
    if (exp == 0u) {
        val = float(mant) * (1.0f / 4.0f) * (1.0f / 16384.0f);
    } else {
        const uint bits = ((exp - 15u + 127u) << 23u) | (mant << 21u);
        val = as_type<float>(bits);
    }
    return sign ? -val : val;
}

kernel void gemm_fp8_e5m2_residual(
    device const uchar* blocks_raw [[buffer(0)]],
    device const float* inputs     [[buffer(1)]],
    device float* outputs          [[buffer(2)]],
    constant uint& n_rows          [[buffer(3)]],
    constant uint& batch_size      [[buffer(4)]],
    constant uint& k               [[buffer(5)]],
    device const float* residual   [[buffer(6)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;
    const uint blocks_per_row = k >> 5u;

    for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {
        const uint cols_remaining = batch_size - col_base;
        const uint cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8] = {0,0,0,0,0,0,0,0};

        for (uint b = lane; b < blocks_per_row; b += 32u) {
            const uint block_idx = row * blocks_per_row + b;
            const uint base_byte = block_idx * 34u;
            const ushort scale_bits =
                  ushort(blocks_raw[base_byte + 32u])
                | (ushort(blocks_raw[base_byte + 33u]) << 8u);
            const float scale = float(as_type<half>(scale_bits));
            const uint inp_base = b * 32u;

            for (uint cc = 0u; cc < cols; cc++) {
                device const float* xbase = inputs + (col_base + cc) * k + inp_base;
                float bsum = 0.0f;
                for (uint w = 0u; w < 32u; ++w) {
                    bsum += pf_fp8_e5m2_to_float(blocks_raw[base_byte + w]) * xbase[w];
                }
                col_sums[cc] += scale * bsum;
            }
        }

        for (uint cc = 0u; cc < cols; cc++) {
            float row_sum = simd_sum(col_sums[cc]);
            if (lane == 0u) {
                const uint idx = (col_base + cc) * n_rows + row;
                outputs[idx] = residual[idx] + row_sum;
            }
        }
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 7 — fused_gate_up_swiglu_gemm_fp8_e5m2
// ═══════════════════════════════════════════════════════════════════════════

/// FP8 E5M2 fused gate+up GEMM with SwiGLU epilogue.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float pf_fp8_e5m2_to_float(uchar b) {
    const uint exp  = (uint(b) >> 2u) & 31u;
    const uint mant = uint(b) & 3u;
    if (exp == 31u) return 0.0f;
    const uint sign = (uint(b) >> 7u) & 1u;
    float val;
    if (exp == 0u) {
        val = float(mant) * (1.0f / 4.0f) * (1.0f / 16384.0f);
    } else {
        const uint bits = ((exp - 15u + 127u) << 23u) | (mant << 21u);
        val = as_type<float>(bits);
    }
    return sign ? -val : val;
}

static inline float pf_silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void fused_gate_up_swiglu_gemm_fp8_e5m2(
    device const uchar* blocks_raw [[buffer(0)]],
    device const float* inputs     [[buffer(1)]],
    device float* outputs          [[buffer(2)]],
    constant uint& n_ffn_rows      [[buffer(3)]],
    constant uint& batch_size      [[buffer(4)]],
    constant uint& k               [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_ffn_rows) return;
    const uint blocks_per_row = k >> 5u;
    const uint up_row_offset  = n_ffn_rows * blocks_per_row;

    for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {
        const uint cols_remaining = batch_size - col_base;
        const uint cols = cols_remaining < 8u ? cols_remaining : 8u;

        float gate_sums[8] = {0,0,0,0,0,0,0,0};
        float up_sums[8]   = {0,0,0,0,0,0,0,0};

        for (uint b = lane; b < blocks_per_row; b += 32u) {
            const uint gbase = (row * blocks_per_row + b) * 34u;
            const uint ubase = (up_row_offset + row * blocks_per_row + b) * 34u;

            const ushort gd_raw =
                  ushort(blocks_raw[gbase + 32u])
                | (ushort(blocks_raw[gbase + 33u]) << 8u);
            const float gscale = float(as_type<half>(gd_raw));

            const ushort ud_raw =
                  ushort(blocks_raw[ubase + 32u])
                | (ushort(blocks_raw[ubase + 33u]) << 8u);
            const float uscale = float(as_type<half>(ud_raw));

            const uint inp_base = b * 32u;
            for (uint cc = 0u; cc < cols; cc++) {
                device const float* xbase = inputs + (col_base + cc) * k + inp_base;
                float gsum = 0.0f;
                float usum = 0.0f;
                for (uint w = 0u; w < 32u; ++w) {
                    const float x = xbase[w];
                    gsum += pf_fp8_e5m2_to_float(blocks_raw[gbase + w]) * x;
                    usum += pf_fp8_e5m2_to_float(blocks_raw[ubase + w]) * x;
                }
                gate_sums[cc] += gscale * gsum;
                up_sums[cc]   += uscale * usum;
            }
        }

        for (uint cc = 0u; cc < cols; cc++) {
            float gs = simd_sum(gate_sums[cc]);
            float us = simd_sum(up_sums[cc]);
            if (lane == 0u) {
                outputs[(col_base + cc) * n_ffn_rows + row] = pf_silu(gs) * us;
            }
        }
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 8 — gemv_fp8_e5m2_pf
// ═══════════════════════════════════════════════════════════════════════════

/// FP8 E5M2 single-token GEMV (prefill family).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_FP8_E5M2_PF_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float pf_fp8_e5m2_to_float(uchar b) {
    const uint exp  = (uint(b) >> 2u) & 31u;
    const uint mant = uint(b) & 3u;
    if (exp == 31u) return 0.0f;
    const uint sign = (uint(b) >> 7u) & 1u;
    float val;
    if (exp == 0u) {
        val = float(mant) * (1.0f / 4.0f) * (1.0f / 16384.0f);
    } else {
        const uint bits = ((exp - 15u + 127u) << 23u) | (mant << 21u);
        val = as_type<float>(bits);
    }
    return sign ? -val : val;
}

kernel void gemv_fp8_e5m2_pf(
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
    const uint blocks_per_row = k >> 5u;

    float local_sum = 0.0f;
    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint base_byte = (row * blocks_per_row + b) * 34u;
        const ushort scale_bits =
              ushort(blocks_raw[base_byte + 32u])
            | (ushort(blocks_raw[base_byte + 33u]) << 8u);
        const float scale = float(as_type<half>(scale_bits));
        const uint inp_base = b * 32u;
        float bsum = 0.0f;
        for (uint w = 0u; w < 32u; ++w) {
            bsum += pf_fp8_e5m2_to_float(blocks_raw[base_byte + w]) * input[inp_base + w];
        }
        local_sum += scale * bsum;
    }

    float row_sum = simd_sum(local_sum);
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
    fn fp8_prefill_kernels_contain_entry_points() {
        use super::*;
        // E4M3 batch + fused + pf
        assert!(MSL_GEMM_FP8_E4M3_V1.contains("kernel void gemm_fp8_e4m3"));
        assert!(MSL_GEMM_FP8_E4M3_RESIDUAL_V1.contains("kernel void gemm_fp8_e4m3_residual"));
        assert!(MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1
            .contains("kernel void fused_gate_up_swiglu_gemm_fp8_e4m3"));
        assert!(MSL_GEMV_FP8_E4M3_PF_V1.contains("kernel void gemv_fp8_e4m3_pf"));
        // E5M2 batch + fused + pf
        assert!(MSL_GEMM_FP8_E5M2_V1.contains("kernel void gemm_fp8_e5m2"));
        assert!(MSL_GEMM_FP8_E5M2_RESIDUAL_V1.contains("kernel void gemm_fp8_e5m2_residual"));
        assert!(MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1
            .contains("kernel void fused_gate_up_swiglu_gemm_fp8_e5m2"));
        assert!(MSL_GEMV_FP8_E5M2_PF_V1.contains("kernel void gemv_fp8_e5m2_pf"));
        // All kernels touch the 34B AoS layout (scale at bytes 32-33).
        for src in [
            MSL_GEMM_FP8_E4M3_V1,
            MSL_GEMM_FP8_E4M3_RESIDUAL_V1,
            MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1,
            MSL_GEMV_FP8_E4M3_PF_V1,
            MSL_GEMM_FP8_E5M2_V1,
            MSL_GEMM_FP8_E5M2_RESIDUAL_V1,
            MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1,
            MSL_GEMV_FP8_E5M2_PF_V1,
        ] {
            assert!(src.contains("* 34u"));
            assert!(src.contains("[[buffer(0)]]"));
        }
        // Batch kernels carry the cap-of-8 outer loop pattern.
        for src in [
            MSL_GEMM_FP8_E4M3_V1,
            MSL_GEMM_FP8_E4M3_RESIDUAL_V1,
            MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1,
            MSL_GEMM_FP8_E5M2_V1,
            MSL_GEMM_FP8_E5M2_RESIDUAL_V1,
            MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1,
        ] {
            assert!(src.contains("col_base += 8u"));
        }
        // Residual kernels expose buffer(6).
        assert!(MSL_GEMM_FP8_E4M3_RESIDUAL_V1.contains("[[buffer(6)]]"));
        assert!(MSL_GEMM_FP8_E5M2_RESIDUAL_V1.contains("[[buffer(6)]]"));
        // Fused gate-up kernels carry the SiLU helper.
        assert!(MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1.contains("pf_silu"));
        assert!(MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1.contains("pf_silu"));
    }
}
