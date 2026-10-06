//! Active decode-path Metal kernels (single-token generation).
//!
//! Contains V7 GEMV, V7 residual GEMV,
//! and fused gate+up+SwiGLU kernels.

/// Q1_0_g128 GEMV V7 — fully unrolled inner loop with aggressive register use.
///
/// Weight buffer uses SoA (Structure-of-Arrays) layout:
///   `[all scales: total_blocks × 2 bytes][all data: total_blocks × 16 bytes]`
/// where `total_blocks = n_rows * blocks_per_row`.
/// Scale reads use sequential 2-byte stride (perfect coalescing).
/// Data reads use 16-byte aligned uint4 loads (no shift-OR needed).
///
/// Key changes from V1:
/// 1. Pre-load all 4 uint32 words via uint4 load from SoA data region
/// 2. Fully unrolled inner loops — zero loop counter overhead, zero branching
/// 3. Each word processed in its own scope with a local accumulator (sum0..3)
///    to maximise instruction-level parallelism
/// 4. Final reduction `sum0+sum1+sum2+sum3` enables compiler reordering
///
/// Same dispatch parameters as V1: 256 threads, `[ceil(n_rows/8), 1, 1]`.
///
/// Buffers: `"x"` → blocks (0), `"y"` → input (1, float4*),
///          `"result"` → output (2)
/// Scalars: `"n"` → n_rows (3), `"k"` → k (4)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V7: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v7(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float4* input4        [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& k                   [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks = n_rows * blocks_per_row;
    const uint data_offset = total_blocks * 2u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const float scale = float(*(device const half*)(blocks_raw + block_idx * 2u));

        // Load 16 data bytes as uint4 from SoA data region (aligned, no shift-OR)
        uint4 packed = *(device const uint4*)(blocks_raw + data_offset + block_idx * 16u);
        uint w0 = packed.x, w1 = packed.y, w2 = packed.z, w3 = packed.w;

        const uint inp_base = b * 32u;

        // Word 0: 8 float4 values (bits 0-31)
        float sum0 = 0.0f;
        {
            uint bits = w0;
            float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s0, input4[inp_base + 0u]); bits >>= 4u;
            float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s1, input4[inp_base + 1u]); bits >>= 4u;
            float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s2, input4[inp_base + 2u]); bits >>= 4u;
            float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s3, input4[inp_base + 3u]); bits >>= 4u;
            float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s4, input4[inp_base + 4u]); bits >>= 4u;
            float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s5, input4[inp_base + 5u]); bits >>= 4u;
            float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s6, input4[inp_base + 6u]); bits >>= 4u;
            float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s7, input4[inp_base + 7u]);
        }

        // Word 1: next 8 float4 values (bits 32-63)
        float sum1 = 0.0f;
        {
            uint bits = w1;
            float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s0, input4[inp_base + 8u]); bits >>= 4u;
            float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s1, input4[inp_base + 9u]); bits >>= 4u;
            float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s2, input4[inp_base + 10u]); bits >>= 4u;
            float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s3, input4[inp_base + 11u]); bits >>= 4u;
            float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s4, input4[inp_base + 12u]); bits >>= 4u;
            float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s5, input4[inp_base + 13u]); bits >>= 4u;
            float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s6, input4[inp_base + 14u]); bits >>= 4u;
            float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s7, input4[inp_base + 15u]);
        }

        // Word 2: next 8 float4 values (bits 64-95)
        float sum2 = 0.0f;
        {
            uint bits = w2;
            float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s0, input4[inp_base + 16u]); bits >>= 4u;
            float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s1, input4[inp_base + 17u]); bits >>= 4u;
            float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s2, input4[inp_base + 18u]); bits >>= 4u;
            float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s3, input4[inp_base + 19u]); bits >>= 4u;
            float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s4, input4[inp_base + 20u]); bits >>= 4u;
            float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s5, input4[inp_base + 21u]); bits >>= 4u;
            float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s6, input4[inp_base + 22u]); bits >>= 4u;
            float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s7, input4[inp_base + 23u]);
        }

        // Word 3: last 8 float4 values (bits 96-127)
        float sum3 = 0.0f;
        {
            uint bits = w3;
            float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s0, input4[inp_base + 24u]); bits >>= 4u;
            float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s1, input4[inp_base + 25u]); bits >>= 4u;
            float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s2, input4[inp_base + 26u]); bits >>= 4u;
            float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s3, input4[inp_base + 27u]); bits >>= 4u;
            float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s4, input4[inp_base + 28u]); bits >>= 4u;
            float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s5, input4[inp_base + 29u]); bits >>= 4u;
            float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s6, input4[inp_base + 30u]); bits >>= 4u;
            float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s7, input4[inp_base + 31u]);
        }

        local_sum += scale * (sum0 + sum1 + sum2 + sum3);
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

/// Fused GEMV + residual V7 — fully unrolled inner loop.
///
/// Same unrolled structure as V7 with SoA (Structure-of-Arrays) weight layout:
///   `[all scales: total_blocks × 2 bytes][all data: total_blocks × 16 bytes]`
/// Final write adds residual: `output[row] = residual[row] + row_sum`
///
/// Buffers: `"x"` → blocks (0), `"y"` → input (1, float4*),
///          `"result"` → output (2), `"n"` → n_rows (3), `"k"` → k (4),
///          residual (5)
///
/// Dispatch: `[ceil(n_rows / 8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V7_RESIDUAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v7_residual(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float4* input4        [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& k                   [[buffer(4)]],
    device const float* residual       [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks = n_rows * blocks_per_row;
    const uint data_offset = total_blocks * 2u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const float scale = float(*(device const half*)(blocks_raw + block_idx * 2u));

        uint4 packed = *(device const uint4*)(blocks_raw + data_offset + block_idx * 16u);
        uint w0 = packed.x, w1 = packed.y, w2 = packed.z, w3 = packed.w;

        const uint inp_base = b * 32u;

        float sum0 = 0.0f;
        {
            uint bits = w0;
            float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s0, input4[inp_base + 0u]); bits >>= 4u;
            float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s1, input4[inp_base + 1u]); bits >>= 4u;
            float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s2, input4[inp_base + 2u]); bits >>= 4u;
            float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s3, input4[inp_base + 3u]); bits >>= 4u;
            float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s4, input4[inp_base + 4u]); bits >>= 4u;
            float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s5, input4[inp_base + 5u]); bits >>= 4u;
            float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s6, input4[inp_base + 6u]); bits >>= 4u;
            float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum0 += dot(s7, input4[inp_base + 7u]);
        }

        float sum1 = 0.0f;
        {
            uint bits = w1;
            float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s0, input4[inp_base + 8u]); bits >>= 4u;
            float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s1, input4[inp_base + 9u]); bits >>= 4u;
            float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s2, input4[inp_base + 10u]); bits >>= 4u;
            float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s3, input4[inp_base + 11u]); bits >>= 4u;
            float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s4, input4[inp_base + 12u]); bits >>= 4u;
            float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s5, input4[inp_base + 13u]); bits >>= 4u;
            float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s6, input4[inp_base + 14u]); bits >>= 4u;
            float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum1 += dot(s7, input4[inp_base + 15u]);
        }

        float sum2 = 0.0f;
        {
            uint bits = w2;
            float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s0, input4[inp_base + 16u]); bits >>= 4u;
            float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s1, input4[inp_base + 17u]); bits >>= 4u;
            float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s2, input4[inp_base + 18u]); bits >>= 4u;
            float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s3, input4[inp_base + 19u]); bits >>= 4u;
            float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s4, input4[inp_base + 20u]); bits >>= 4u;
            float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s5, input4[inp_base + 21u]); bits >>= 4u;
            float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s6, input4[inp_base + 22u]); bits >>= 4u;
            float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum2 += dot(s7, input4[inp_base + 23u]);
        }

        float sum3 = 0.0f;
        {
            uint bits = w3;
            float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s0, input4[inp_base + 24u]); bits >>= 4u;
            float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s1, input4[inp_base + 25u]); bits >>= 4u;
            float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s2, input4[inp_base + 26u]); bits >>= 4u;
            float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s3, input4[inp_base + 27u]); bits >>= 4u;
            float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s4, input4[inp_base + 28u]); bits >>= 4u;
            float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s5, input4[inp_base + 29u]); bits >>= 4u;
            float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s6, input4[inp_base + 30u]); bits >>= 4u;
            float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
            sum3 += dot(s7, input4[inp_base + 31u]);
        }

        local_sum += scale * (sum0 + sum1 + sum2 + sum3);
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = residual[row] + row_sum;
    }
}
"#;

/// Fused Gate+Up+SwiGLU kernel for Q1_0_g128 weights.
///
/// Combines the separate gate_up GEMV and SwiGLU dispatches into a single
/// kernel.  Each simdgroup computes one output position: it reads both
/// the gate row and the up row from the concatenated weight matrix, then
/// applies `silu(gate) * up` in the epilogue.
///
/// Weight buffer uses SoA (Structure-of-Arrays) layout:
///   `[all scales: total_blocks × 2 bytes][all data: total_blocks × 16 bytes]`
/// where `total_blocks = 2 * inter_size * blocks_per_row` (gate + up combined).
/// Rows `[0..inter_size)` = gate, rows `[inter_size..2*inter_size)` = up.
///
/// Buffers:
/// - buffer(0) = blocks_raw (u8, gate+up weights in SoA layout)
/// - buffer(1) = input4     (f32, read as float4*, normed hidden state)
/// - buffer(2) = output     (f32, swiglu output `[inter_size]`)
/// - buffer(3) = inter_size (u32, scalar — 12288 for Bonsai-8B)
/// - buffer(4) = k          (u32, scalar — hidden_size = 4096)
///
/// Dispatch: `[ceil(inter_size/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_GATE_UP_SWIGLU_Q1: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_gate_up_swiglu_q1(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float4* input4        [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& inter_size          [[buffer(3)]],
    constant uint& k                   [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    // Each simdgroup processes one output position
    const uint pos = tgid * 8u + sgid;
    if (pos >= inter_size) return;

    const uint blocks_per_row = k / 128u;
    // SoA layout: total_blocks covers both gate and up halves
    const uint total_blocks = 2u * inter_size * blocks_per_row;
    const uint data_offset = total_blocks * 2u;
    // Gate row: position pos in the first half of the weight matrix
    const uint gate_block_base = pos * blocks_per_row;
    // Up row: position (pos + inter_size) in the second half
    const uint up_block_base = (pos + inter_size) * blocks_per_row;

    float gate_local = 0.0f;
    float up_local = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint inp_base = b * 32u;

        // ── Gate block ──────────────────────────────────────────────
        {
            const uint block_idx = gate_block_base + b;
            const float scale = float(*(device const half*)(blocks_raw + block_idx * 2u));
            uint4 packed = *(device const uint4*)(blocks_raw + data_offset + block_idx * 16u);
            uint w0 = packed.x, w1 = packed.y, w2 = packed.z, w3 = packed.w;

            // Word 0: 8 float4 values
            float sum0 = 0.0f;
            {
                uint bits = w0;
                float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s0, input4[inp_base + 0u]); bits >>= 4u;
                float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s1, input4[inp_base + 1u]); bits >>= 4u;
                float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s2, input4[inp_base + 2u]); bits >>= 4u;
                float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s3, input4[inp_base + 3u]); bits >>= 4u;
                float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s4, input4[inp_base + 4u]); bits >>= 4u;
                float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s5, input4[inp_base + 5u]); bits >>= 4u;
                float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s6, input4[inp_base + 6u]); bits >>= 4u;
                float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s7, input4[inp_base + 7u]);
            }

            // Word 1: next 8 float4 values
            float sum1 = 0.0f;
            {
                uint bits = w1;
                float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s0, input4[inp_base + 8u]); bits >>= 4u;
                float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s1, input4[inp_base + 9u]); bits >>= 4u;
                float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s2, input4[inp_base + 10u]); bits >>= 4u;
                float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s3, input4[inp_base + 11u]); bits >>= 4u;
                float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s4, input4[inp_base + 12u]); bits >>= 4u;
                float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s5, input4[inp_base + 13u]); bits >>= 4u;
                float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s6, input4[inp_base + 14u]); bits >>= 4u;
                float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s7, input4[inp_base + 15u]);
            }

            // Word 2: next 8 float4 values
            float sum2 = 0.0f;
            {
                uint bits = w2;
                float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s0, input4[inp_base + 16u]); bits >>= 4u;
                float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s1, input4[inp_base + 17u]); bits >>= 4u;
                float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s2, input4[inp_base + 18u]); bits >>= 4u;
                float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s3, input4[inp_base + 19u]); bits >>= 4u;
                float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s4, input4[inp_base + 20u]); bits >>= 4u;
                float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s5, input4[inp_base + 21u]); bits >>= 4u;
                float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s6, input4[inp_base + 22u]); bits >>= 4u;
                float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s7, input4[inp_base + 23u]);
            }

            // Word 3: last 8 float4 values
            float sum3 = 0.0f;
            {
                uint bits = w3;
                float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s0, input4[inp_base + 24u]); bits >>= 4u;
                float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s1, input4[inp_base + 25u]); bits >>= 4u;
                float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s2, input4[inp_base + 26u]); bits >>= 4u;
                float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s3, input4[inp_base + 27u]); bits >>= 4u;
                float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s4, input4[inp_base + 28u]); bits >>= 4u;
                float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s5, input4[inp_base + 29u]); bits >>= 4u;
                float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s6, input4[inp_base + 30u]); bits >>= 4u;
                float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s7, input4[inp_base + 31u]);
            }

            gate_local += scale * (sum0 + sum1 + sum2 + sum3);
        }

        // ── Up block ────────────────────────────────────────────────
        {
            const uint block_idx = up_block_base + b;
            const float scale = float(*(device const half*)(blocks_raw + block_idx * 2u));
            uint4 packed = *(device const uint4*)(blocks_raw + data_offset + block_idx * 16u);
            uint w0 = packed.x, w1 = packed.y, w2 = packed.z, w3 = packed.w;

            // Word 0
            float sum0 = 0.0f;
            {
                uint bits = w0;
                float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s0, input4[inp_base + 0u]); bits >>= 4u;
                float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s1, input4[inp_base + 1u]); bits >>= 4u;
                float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s2, input4[inp_base + 2u]); bits >>= 4u;
                float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s3, input4[inp_base + 3u]); bits >>= 4u;
                float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s4, input4[inp_base + 4u]); bits >>= 4u;
                float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s5, input4[inp_base + 5u]); bits >>= 4u;
                float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s6, input4[inp_base + 6u]); bits >>= 4u;
                float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum0 += dot(s7, input4[inp_base + 7u]);
            }

            // Word 1
            float sum1 = 0.0f;
            {
                uint bits = w1;
                float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s0, input4[inp_base + 8u]); bits >>= 4u;
                float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s1, input4[inp_base + 9u]); bits >>= 4u;
                float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s2, input4[inp_base + 10u]); bits >>= 4u;
                float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s3, input4[inp_base + 11u]); bits >>= 4u;
                float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s4, input4[inp_base + 12u]); bits >>= 4u;
                float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s5, input4[inp_base + 13u]); bits >>= 4u;
                float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s6, input4[inp_base + 14u]); bits >>= 4u;
                float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum1 += dot(s7, input4[inp_base + 15u]);
            }

            // Word 2
            float sum2 = 0.0f;
            {
                uint bits = w2;
                float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s0, input4[inp_base + 16u]); bits >>= 4u;
                float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s1, input4[inp_base + 17u]); bits >>= 4u;
                float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s2, input4[inp_base + 18u]); bits >>= 4u;
                float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s3, input4[inp_base + 19u]); bits >>= 4u;
                float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s4, input4[inp_base + 20u]); bits >>= 4u;
                float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s5, input4[inp_base + 21u]); bits >>= 4u;
                float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s6, input4[inp_base + 22u]); bits >>= 4u;
                float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum2 += dot(s7, input4[inp_base + 23u]);
            }

            // Word 3
            float sum3 = 0.0f;
            {
                uint bits = w3;
                float4 s0 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s0, input4[inp_base + 24u]); bits >>= 4u;
                float4 s1 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s1, input4[inp_base + 25u]); bits >>= 4u;
                float4 s2 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s2, input4[inp_base + 26u]); bits >>= 4u;
                float4 s3 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s3, input4[inp_base + 27u]); bits >>= 4u;
                float4 s4 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s4, input4[inp_base + 28u]); bits >>= 4u;
                float4 s5 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s5, input4[inp_base + 29u]); bits >>= 4u;
                float4 s6 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s6, input4[inp_base + 30u]); bits >>= 4u;
                float4 s7 = float4(select(-1.0f,1.0f,bool(bits&1u)), select(-1.0f,1.0f,bool(bits&2u)), select(-1.0f,1.0f,bool(bits&4u)), select(-1.0f,1.0f,bool(bits&8u)));
                sum3 += dot(s7, input4[inp_base + 31u]);
            }

            up_local += scale * (sum0 + sum1 + sum2 + sum3);
        }
    }

    // Parallel reduction across 32 lanes
    float gate_result = simd_sum(gate_local);
    float up_result = simd_sum(up_local);

    if (lane == 0u) {
        // SwiGLU epilogue: silu(gate) * up
        float silu_g = gate_result / (1.0f + exp(-gate_result));
        output[pos] = silu_g * up_result;
    }
}
"#;
