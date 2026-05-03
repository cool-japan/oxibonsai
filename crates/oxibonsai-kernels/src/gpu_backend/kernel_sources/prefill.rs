//! Active prefill-path Metal kernels (batch/prompt processing).
//!
//! Contains V7 GEMM, V7 residual GEMM, fused gate+up+SwiGLU GEMM,
//! batched SwiGLU, and batched RMSNorm kernels.
//!
//! Weight buffers use **SoA (Structure-of-Arrays)** layout:
//! `[all scales: total_blocks × 2 bytes][all data: total_blocks × 16 bytes]`

/// V7-based GEMM: 1D grid with weight-tiled batch processing.
///
/// Each simdgroup processes 1 weight row × ALL batch columns, loading weights
/// once per block iteration (L1 cache retains weights across columns).
/// V7 inner loop (fully unrolled, simd_sum reduction).  Input/output are
/// column-major: `inputs[col * k + elem]`, `outputs[col * n_rows + row]`.
///
/// Weight buffer uses SoA layout:
/// `[scales: n_rows*blocks_per_row × 2B][data: n_rows*blocks_per_row × 16B]`
///
/// Buffers:
/// - buffer(0) = blocks_raw  (u8, Q1_0_g128 weight data, SoA layout)
/// - buffer(1) = inputs      (f32, batch × k, column-major)
/// - buffer(2) = outputs     (f32, batch × n_rows, column-major)
/// - buffer(3) = n_rows      (u32)
/// - buffer(4) = batch_size  (u32)
/// - buffer(5) = k           (u32)
///
/// Dispatch: `[ceil(n_rows/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_Q1_G128_V7: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_q1_g128_v7(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float* inputs         [[buffer(1)]],
    device float* outputs              [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& batch_size          [[buffer(4)]],
    constant uint& k                   [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks = n_rows * blocks_per_row;
    const uint data_offset = total_blocks * 2u;
    const uint cols = min(batch_size, 8u);

    float col_sums[8] = {0,0,0,0,0,0,0,0};

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const float scale = float(*(device const half*)(blocks_raw + block_idx * 2u));
        uint4 packed = *(device const uint4*)(blocks_raw + data_offset + block_idx * 16u);
        const uint inp_base = b * 32u;

        { // Chunk 0: packed.x
            uint bits = packed.x;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
                                     +dot(s4,in4[inp_base+4u])+dot(s5,in4[inp_base+5u])+dot(s6,in4[inp_base+6u])+dot(s7,in4[inp_base+7u]));
            }
        }
        { // Chunk 1: packed.y
            uint bits = packed.y;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
                                     +dot(s4,in4[inp_base+12u])+dot(s5,in4[inp_base+13u])+dot(s6,in4[inp_base+14u])+dot(s7,in4[inp_base+15u]));
            }
        }
        { // Chunk 2: packed.z
            uint bits = packed.z;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
                                     +dot(s4,in4[inp_base+20u])+dot(s5,in4[inp_base+21u])+dot(s6,in4[inp_base+22u])+dot(s7,in4[inp_base+23u]));
            }
        }
        { // Chunk 3: packed.w
            uint bits = packed.w;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                     +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
            }
        }
    }

    for (uint col = 0u; col < cols; col++) {
        float row_sum = simd_sum(col_sums[col]);
        if (lane == 0u) outputs[col * n_rows + row] = row_sum;
    }
}
"#;

/// V7-based GEMM with residual addition.
///
/// Same as `gemm_q1_g128_v7` but adds a residual value: `out = residual + gemv_result`.
///
/// Weight buffer uses SoA layout:
/// `[scales: n_rows*blocks_per_row × 2B][data: n_rows*blocks_per_row × 16B]`
///
/// Buffers:
/// - buffer(0) = blocks_raw  (u8, Q1_0_g128 weight data, SoA layout)
/// - buffer(1) = inputs      (f32, batch × k, column-major)
/// - buffer(2) = outputs     (f32, batch × n_rows, column-major)
/// - buffer(3) = n_rows      (u32)
/// - buffer(4) = batch_size  (u32)
/// - buffer(5) = k           (u32)
/// - buffer(6) = residual    (f32, batch × n_rows, column-major)
///
/// Dispatch: `[ceil(n_rows/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_Q1_G128_V7_RESIDUAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_q1_g128_v7_residual(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float* inputs         [[buffer(1)]],
    device float* outputs              [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& batch_size          [[buffer(4)]],
    constant uint& k                   [[buffer(5)]],
    device const float* residual       [[buffer(6)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks = n_rows * blocks_per_row;
    const uint data_offset = total_blocks * 2u;
    const uint cols = min(batch_size, 8u);

    float col_sums[8] = {0,0,0,0,0,0,0,0};

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const float scale = float(*(device const half*)(blocks_raw + block_idx * 2u));
        uint4 packed = *(device const uint4*)(blocks_raw + data_offset + block_idx * 16u);
        const uint inp_base = b * 32u;

        { // Chunk 0: packed.x
            uint bits = packed.x;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
                                     +dot(s4,in4[inp_base+4u])+dot(s5,in4[inp_base+5u])+dot(s6,in4[inp_base+6u])+dot(s7,in4[inp_base+7u]));
            }
        }
        { // Chunk 1: packed.y
            uint bits = packed.y;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
                                     +dot(s4,in4[inp_base+12u])+dot(s5,in4[inp_base+13u])+dot(s6,in4[inp_base+14u])+dot(s7,in4[inp_base+15u]));
            }
        }
        { // Chunk 2: packed.z
            uint bits = packed.z;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
                                     +dot(s4,in4[inp_base+20u])+dot(s5,in4[inp_base+21u])+dot(s6,in4[inp_base+22u])+dot(s7,in4[inp_base+23u]));
            }
        }
        { // Chunk 3: packed.w
            uint bits = packed.w;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                col_sums[col]+=scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                     +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
            }
        }
    }

    for (uint col = 0u; col < cols; col++) {
        float row_sum = simd_sum(col_sums[col]);
        if (lane == 0u) outputs[col * n_rows + row] = residual[col * n_rows + row] + row_sum;
    }
}
"#;

/// Fused gate+up+SwiGLU GEMM for batch prefill (V7-based).
///
/// 1D grid with weight-tiled batch processing: each simdgroup computes one
/// FFN output position for ALL batch columns, loading gate+up weights once.
/// Applies `silu(gate) * up` in the epilogue.
///
/// Weight buffer uses SoA layout over concatenated gate+up rows
/// (total_blocks = 2*inter_size*blocks_per_row):
/// `[scales: total_blocks × 2B][data: total_blocks × 16B]`
///
/// Buffers:
/// - buffer(0) = blocks_raw  (u8, gate+up weights concatenated, SoA layout)
/// - buffer(1) = inputs      (f32, batch × k, column-major)
/// - buffer(2) = outputs     (f32, batch × inter_size, column-major)
/// - buffer(3) = inter_size  (u32)
/// - buffer(4) = batch_size  (u32)
/// - buffer(5) = k           (u32)
///
/// Dispatch: `[ceil(inter_size/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_GATE_UP_SWIGLU_GEMM_Q1: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_gate_up_swiglu_gemm_q1(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float* inputs         [[buffer(1)]],
    device float* outputs              [[buffer(2)]],
    constant uint& inter_size          [[buffer(3)]],
    constant uint& batch_size          [[buffer(4)]],
    constant uint& k                   [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint pos = tgid * 8u + sgid;
    if (pos >= inter_size) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks = 2u * inter_size * blocks_per_row;
    const uint data_offset = total_blocks * 2u;
    const uint gate_block_base = pos * blocks_per_row;
    const uint up_block_base = (inter_size + pos) * blocks_per_row;
    const uint cols = min(batch_size, 8u);

    float gate_sums[8] = {0,0,0,0,0,0,0,0};
    float up_sums[8] = {0,0,0,0,0,0,0,0};

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint inp_base = b * 32u;

        // ── Gate: load and process 4 chunks ──
        {
            const uint gate_block_idx = gate_block_base + b;
            const float gate_scale = float(*(device const half*)(blocks_raw + gate_block_idx * 2u));
            uint4 gate_packed = *(device const uint4*)(blocks_raw + data_offset + gate_block_idx * 16u);
        { // gate chunk 0: gate_packed.x
            uint bits = gate_packed.x;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                gate_sums[col]+=gate_scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
                                     +dot(s4,in4[inp_base+4u])+dot(s5,in4[inp_base+5u])+dot(s6,in4[inp_base+6u])+dot(s7,in4[inp_base+7u]));
            }
        }
        { // gate chunk 1: gate_packed.y
            uint bits = gate_packed.y;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                gate_sums[col]+=gate_scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
                                     +dot(s4,in4[inp_base+12u])+dot(s5,in4[inp_base+13u])+dot(s6,in4[inp_base+14u])+dot(s7,in4[inp_base+15u]));
            }
        }
        { // gate chunk 2: gate_packed.z
            uint bits = gate_packed.z;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                gate_sums[col]+=gate_scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
                                     +dot(s4,in4[inp_base+20u])+dot(s5,in4[inp_base+21u])+dot(s6,in4[inp_base+22u])+dot(s7,in4[inp_base+23u]));
            }
        }
        { // gate chunk 3: gate_packed.w
            uint bits = gate_packed.w;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                gate_sums[col]+=gate_scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                     +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
            }
        }
        }

        // ── Up: load and process 4 chunks ──
        {
            const uint up_block_idx = up_block_base + b;
            const float up_scale = float(*(device const half*)(blocks_raw + up_block_idx * 2u));
            uint4 up_packed = *(device const uint4*)(blocks_raw + data_offset + up_block_idx * 16u);
        { // up chunk 0: up_packed.x
            uint bits = up_packed.x;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                up_sums[col]+=up_scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
                                     +dot(s4,in4[inp_base+4u])+dot(s5,in4[inp_base+5u])+dot(s6,in4[inp_base+6u])+dot(s7,in4[inp_base+7u]));
            }
        }
        { // up chunk 1: up_packed.y
            uint bits = up_packed.y;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                up_sums[col]+=up_scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
                                     +dot(s4,in4[inp_base+12u])+dot(s5,in4[inp_base+13u])+dot(s6,in4[inp_base+14u])+dot(s7,in4[inp_base+15u]));
            }
        }
        { // up chunk 2: up_packed.z
            uint bits = up_packed.z;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                up_sums[col]+=up_scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
                                     +dot(s4,in4[inp_base+20u])+dot(s5,in4[inp_base+21u])+dot(s6,in4[inp_base+22u])+dot(s7,in4[inp_base+23u]));
            }
        }
        { // up chunk 3: up_packed.w
            uint bits = up_packed.w;
            float4 s0=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s1=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s2=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s3=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s4=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s5=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s6=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u))); bits>>=4u;
            float4 s7=float4(select(-1.0f,1.0f,bool(bits&1u)),select(-1.0f,1.0f,bool(bits&2u)),select(-1.0f,1.0f,bool(bits&4u)),select(-1.0f,1.0f,bool(bits&8u)));
            for (uint col=0u; col<cols; col++) {
                device const float4* in4=(device const float4*)(inputs+col*k);
                up_sums[col]+=up_scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                     +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
            }
        }
        }
    }

    for (uint col = 0u; col < cols; col++) {
        float gate_val = simd_sum(gate_sums[col]);
        float up_val = simd_sum(up_sums[col]);
        if (lane == 0u) {
            float silu_g = gate_val / (1.0f + exp(-gate_val));
            outputs[col * inter_size + pos] = silu_g * up_val;
        }
    }
}
"#;

/// Batched SwiGLU: processes B vectors from concatenated [gate | up] layout.
///
/// Input layout: for batch `b`, gate = gate_up[b * inter * 2 .. b * inter * 2 + inter],
///               up = gate_up[b * inter * 2 + inter .. b * inter * 2 + inter * 2].
/// Output layout: output[b * inter + elem].
///
/// Buffers:
///   - buffer(0) = gate_up `[batch_size × inter × 2]` (f32)
///   - buffer(1) = output  `[batch_size × inter]` (f32)
///   - buffer(2) = inter (u32)
///   - buffer(3) = batch_size (u32)
///
/// Dispatch: `[ceil(inter/256), batch_size, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_SWIGLU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_swiglu(
    device const float* gate_up  [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& inter         [[buffer(2)]],
    constant uint& batch_size    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint elem  = gid.x;
    uint batch = gid.y;
    if (elem >= inter || batch >= batch_size) return;

    uint offset = batch * inter * 2u;
    float g = gate_up[offset + elem];
    float u = gate_up[offset + inter + elem];
    float silu_g = g / (1.0f + exp(-g));
    output[batch * inter + elem] = silu_g * u;
}
"#;

/// Batched RMSNorm V2: one threadgroup per head, 256 threads.
///
/// Each threadgroup processes `dim` elements for a single head using
/// shared-memory parallel reduction for the sum-of-squares.
///
/// Buffers:
///   - `input`  `[num_heads × dim]` (f32)
///   - `weight` `[dim]` (f32, shared across all heads)
///   - `output` `[num_heads × dim]` (f32)
///   - `eps`    (f32 scalar)
///   - `dim`    (u32 scalar)
///
/// Dispatch: `[num_heads, 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_RMSNORM_V2: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_rmsnorm_v2(
    device const float* input,
    device const float* weight,
    device float* output,
    constant float& eps,
    constant uint& dim,
    uint tgpig  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint head = tgpig;
    uint offset = head * dim;

    threadgroup float shared_sum[256];
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = input[offset + i];
        local_sq += v * v;
    }
    shared_sum[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_inv = rsqrt(shared_sum[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += tg_size) {
        output[offset + i] = input[offset + i] * rms_inv * weight[i];
    }
}
"#;

/// V7-style GEMM for **TQ2_0_g128 (ternary)** weights, batched prefill.
///
/// Each simdgroup processes one weight row × ALL batch columns (in 8-column
/// outer chunks so arbitrary batch sizes are supported, unlike the Q1 V7
/// kernel which silently caps `cols` at 8). Loads each TQ2 block once per
/// outer chunk so the L1 cache retains weights across columns. Decode lives
/// in `decode_tq2` / `decode_byte_tq2` — copied **byte-for-byte** from
/// `MSL_GEMV_TQ2_G128_V1` so the batched kernel produces bit-identical
/// results to the per-position GEMV path.
///
/// Weight buffer uses SoA layout (same as the Q1 prefill kernels):
/// `[scales: total_blocks × 2 bytes][data: total_blocks × 32 bytes]`
///
/// Buffers:
/// - buffer(0) = soa_raw    (u8, TQ2_0_g128 weight data, SoA layout)
/// - buffer(1) = inputs     (f32, batch × k, column-major)
/// - buffer(2) = outputs    (f32, batch × n_rows, column-major)
/// - buffer(3) = n_rows     (u32)
/// - buffer(4) = batch_size (u32)
/// - buffer(5) = k          (u32)
///
/// Dispatch: `[ceil(n_rows/8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_TQ2_G128_V7: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float pf_decode_tq2(uint code) {
    return select(select(0.0f, -1.0f, code == 0u), 1.0f, code == 2u);
}

inline float4 pf_decode_byte_tq2(uint b) {
    return float4(
        pf_decode_tq2((b     ) & 3u),
        pf_decode_tq2((b >> 2) & 3u),
        pf_decode_tq2((b >> 4) & 3u),
        pf_decode_tq2((b >> 6) & 3u)
    );
}

kernel void gemm_tq2_g128_v7(
    device const uchar*  soa_raw    [[buffer(0)]],
    device const float*  inputs     [[buffer(1)]],
    device       float*  outputs    [[buffer(2)]],
    constant uint&       n_rows     [[buffer(3)]],
    constant uint&       batch_size [[buffer(4)]],
    constant uint&       k          [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks   = n_rows * blocks_per_row;
    const uint qs_offset      = total_blocks * 2u;

    // Iterate over batch columns in groups of up to 8 so the kernel handles
    // arbitrary batch sizes correctly (unlike the Q1 V7 kernel which is
    // capped at 8 columns).  Each outer iteration reloads weights once and
    // accumulates dot products against up to 8 input columns.
    for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {
        const uint cols_remaining = batch_size - col_base;
        const uint cols = cols_remaining < 8u ? cols_remaining : 8u;

        float col_sums[8] = {0,0,0,0,0,0,0,0};

        for (uint b = lane; b < blocks_per_row; b += 32u) {
            const uint block_idx = row * blocks_per_row + b;
            const float scale = float(*(device const half*)(soa_raw + block_idx * 2u));
            const uint qs_base = qs_offset + block_idx * 32u;

            // Pack 32 quantised bytes into eight 32-bit words (LSB-first
            // byte order, identical to the GEMV reference).
            uint w0 = (uint)(soa_raw[qs_base +  0]) | ((uint)(soa_raw[qs_base +  1]) << 8) | ((uint)(soa_raw[qs_base +  2]) << 16) | ((uint)(soa_raw[qs_base +  3]) << 24);
            uint w1 = (uint)(soa_raw[qs_base +  4]) | ((uint)(soa_raw[qs_base +  5]) << 8) | ((uint)(soa_raw[qs_base +  6]) << 16) | ((uint)(soa_raw[qs_base +  7]) << 24);
            uint w2 = (uint)(soa_raw[qs_base +  8]) | ((uint)(soa_raw[qs_base +  9]) << 8) | ((uint)(soa_raw[qs_base + 10]) << 16) | ((uint)(soa_raw[qs_base + 11]) << 24);
            uint w3 = (uint)(soa_raw[qs_base + 12]) | ((uint)(soa_raw[qs_base + 13]) << 8) | ((uint)(soa_raw[qs_base + 14]) << 16) | ((uint)(soa_raw[qs_base + 15]) << 24);
            uint w4 = (uint)(soa_raw[qs_base + 16]) | ((uint)(soa_raw[qs_base + 17]) << 8) | ((uint)(soa_raw[qs_base + 18]) << 16) | ((uint)(soa_raw[qs_base + 19]) << 24);
            uint w5 = (uint)(soa_raw[qs_base + 20]) | ((uint)(soa_raw[qs_base + 21]) << 8) | ((uint)(soa_raw[qs_base + 22]) << 16) | ((uint)(soa_raw[qs_base + 23]) << 24);
            uint w6 = (uint)(soa_raw[qs_base + 24]) | ((uint)(soa_raw[qs_base + 25]) << 8) | ((uint)(soa_raw[qs_base + 26]) << 16) | ((uint)(soa_raw[qs_base + 27]) << 24);
            uint w7 = (uint)(soa_raw[qs_base + 28]) | ((uint)(soa_raw[qs_base + 29]) << 8) | ((uint)(soa_raw[qs_base + 30]) << 16) | ((uint)(soa_raw[qs_base + 31]) << 24);

            // Decode the 32 bytes (= 128 weights) into 32 float4 lanes.
            float4 d00 = pf_decode_byte_tq2((w0      ) & 0xFFu);
            float4 d01 = pf_decode_byte_tq2((w0 >>  8) & 0xFFu);
            float4 d02 = pf_decode_byte_tq2((w0 >> 16) & 0xFFu);
            float4 d03 = pf_decode_byte_tq2((w0 >> 24) & 0xFFu);
            float4 d04 = pf_decode_byte_tq2((w1      ) & 0xFFu);
            float4 d05 = pf_decode_byte_tq2((w1 >>  8) & 0xFFu);
            float4 d06 = pf_decode_byte_tq2((w1 >> 16) & 0xFFu);
            float4 d07 = pf_decode_byte_tq2((w1 >> 24) & 0xFFu);
            float4 d08 = pf_decode_byte_tq2((w2      ) & 0xFFu);
            float4 d09 = pf_decode_byte_tq2((w2 >>  8) & 0xFFu);
            float4 d10 = pf_decode_byte_tq2((w2 >> 16) & 0xFFu);
            float4 d11 = pf_decode_byte_tq2((w2 >> 24) & 0xFFu);
            float4 d12 = pf_decode_byte_tq2((w3      ) & 0xFFu);
            float4 d13 = pf_decode_byte_tq2((w3 >>  8) & 0xFFu);
            float4 d14 = pf_decode_byte_tq2((w3 >> 16) & 0xFFu);
            float4 d15 = pf_decode_byte_tq2((w3 >> 24) & 0xFFu);
            float4 d16 = pf_decode_byte_tq2((w4      ) & 0xFFu);
            float4 d17 = pf_decode_byte_tq2((w4 >>  8) & 0xFFu);
            float4 d18 = pf_decode_byte_tq2((w4 >> 16) & 0xFFu);
            float4 d19 = pf_decode_byte_tq2((w4 >> 24) & 0xFFu);
            float4 d20 = pf_decode_byte_tq2((w5      ) & 0xFFu);
            float4 d21 = pf_decode_byte_tq2((w5 >>  8) & 0xFFu);
            float4 d22 = pf_decode_byte_tq2((w5 >> 16) & 0xFFu);
            float4 d23 = pf_decode_byte_tq2((w5 >> 24) & 0xFFu);
            float4 d24 = pf_decode_byte_tq2((w6      ) & 0xFFu);
            float4 d25 = pf_decode_byte_tq2((w6 >>  8) & 0xFFu);
            float4 d26 = pf_decode_byte_tq2((w6 >> 16) & 0xFFu);
            float4 d27 = pf_decode_byte_tq2((w6 >> 24) & 0xFFu);
            float4 d28 = pf_decode_byte_tq2((w7      ) & 0xFFu);
            float4 d29 = pf_decode_byte_tq2((w7 >>  8) & 0xFFu);
            float4 d30 = pf_decode_byte_tq2((w7 >> 16) & 0xFFu);
            float4 d31 = pf_decode_byte_tq2((w7 >> 24) & 0xFFu);

            const uint inp_base = b * 32u;
            for (uint cc = 0u; cc < cols; cc++) {
                const uint col = col_base + cc;
                device const float4* in4 = (device const float4*)(inputs + col * k);
                float block_sum = 0.0f;
                block_sum += dot(d00, in4[inp_base +  0u]);
                block_sum += dot(d01, in4[inp_base +  1u]);
                block_sum += dot(d02, in4[inp_base +  2u]);
                block_sum += dot(d03, in4[inp_base +  3u]);
                block_sum += dot(d04, in4[inp_base +  4u]);
                block_sum += dot(d05, in4[inp_base +  5u]);
                block_sum += dot(d06, in4[inp_base +  6u]);
                block_sum += dot(d07, in4[inp_base +  7u]);
                block_sum += dot(d08, in4[inp_base +  8u]);
                block_sum += dot(d09, in4[inp_base +  9u]);
                block_sum += dot(d10, in4[inp_base + 10u]);
                block_sum += dot(d11, in4[inp_base + 11u]);
                block_sum += dot(d12, in4[inp_base + 12u]);
                block_sum += dot(d13, in4[inp_base + 13u]);
                block_sum += dot(d14, in4[inp_base + 14u]);
                block_sum += dot(d15, in4[inp_base + 15u]);
                block_sum += dot(d16, in4[inp_base + 16u]);
                block_sum += dot(d17, in4[inp_base + 17u]);
                block_sum += dot(d18, in4[inp_base + 18u]);
                block_sum += dot(d19, in4[inp_base + 19u]);
                block_sum += dot(d20, in4[inp_base + 20u]);
                block_sum += dot(d21, in4[inp_base + 21u]);
                block_sum += dot(d22, in4[inp_base + 22u]);
                block_sum += dot(d23, in4[inp_base + 23u]);
                block_sum += dot(d24, in4[inp_base + 24u]);
                block_sum += dot(d25, in4[inp_base + 25u]);
                block_sum += dot(d26, in4[inp_base + 26u]);
                block_sum += dot(d27, in4[inp_base + 27u]);
                block_sum += dot(d28, in4[inp_base + 28u]);
                block_sum += dot(d29, in4[inp_base + 29u]);
                block_sum += dot(d30, in4[inp_base + 30u]);
                block_sum += dot(d31, in4[inp_base + 31u]);
                col_sums[cc] += scale * block_sum;
            }
        }

        for (uint cc = 0u; cc < cols; cc++) {
            float row_sum = simd_sum(col_sums[cc]);
            if (lane == 0u) outputs[(col_base + cc) * n_rows + row] = row_sum;
        }
    }
}
"#;
