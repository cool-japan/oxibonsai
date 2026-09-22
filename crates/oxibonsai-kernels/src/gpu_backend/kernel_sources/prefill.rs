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
/// Batch columns are processed in 8-column outer chunks
/// (`for col_base in 0..batch_size step 8u`) so arbitrary `batch_size`
/// values are handled correctly. (An earlier version silently capped
/// `cols` at 8 and zeroed columns 8..N — see issue tracking ultra-#1.)
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

    // Iterate batch columns in groups of up to 8 so any batch_size is handled
    // correctly. Each outer iteration reloads the weight row once and
    // accumulates dot products against up to 8 input columns.
    for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {
        const uint cols_remaining = batch_size - col_base;
        const uint cols = cols_remaining < 8u ? cols_remaining : 8u;

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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    col_sums[cc]+=scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    col_sums[cc]+=scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    col_sums[cc]+=scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    col_sums[cc]+=scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                         +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
                }
            }
        }

        for (uint cc = 0u; cc < cols; cc++) {
            float row_sum = simd_sum(col_sums[cc]);
            if (lane == 0u) outputs[(col_base + cc) * n_rows + row] = row_sum;
        }
    }
}
"#;

/// V7-based GEMM with residual addition.
///
/// Same as `gemm_q1_g128_v7` but adds a residual value: `out = residual + gemv_result`.
/// Batch columns are processed in 8-column outer chunks
/// (`for col_base in 0..batch_size step 8u`) so arbitrary `batch_size`
/// values are handled correctly. (An earlier version silently capped
/// `cols` at 8 and zeroed columns 8..N — see issue tracking ultra-#1.)
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

    // Iterate batch columns in groups of up to 8 so any batch_size is handled
    // correctly. Each outer iteration reloads the weight row once and
    // accumulates dot products against up to 8 input columns.
    for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {
        const uint cols_remaining = batch_size - col_base;
        const uint cols = cols_remaining < 8u ? cols_remaining : 8u;

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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    col_sums[cc]+=scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    col_sums[cc]+=scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    col_sums[cc]+=scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    col_sums[cc]+=scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                         +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
                }
            }
        }

        for (uint cc = 0u; cc < cols; cc++) {
            float row_sum = simd_sum(col_sums[cc]);
            if (lane == 0u) {
                const uint col = col_base + cc;
                outputs[col * n_rows + row] = residual[col * n_rows + row] + row_sum;
            }
        }
    }
}
"#;

/// Fused gate+up+SwiGLU GEMM for batch prefill (V7-based).
///
/// 1D grid with weight-tiled batch processing: each simdgroup computes one
/// FFN output position for ALL batch columns, loading gate+up weights once
/// per outer 8-column chunk. Applies `silu(gate) * up` in the epilogue.
///
/// Batch columns are processed in 8-column outer chunks
/// (`for col_base in 0..batch_size step 8u`) so arbitrary `batch_size`
/// values are handled correctly. (An earlier version silently capped
/// `cols` at 8 and zeroed columns 8..N — see issue tracking ultra-#1.)
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

    // Iterate batch columns in groups of up to 8 so any batch_size is handled
    // correctly. Each outer iteration reloads the gate+up weight rows once
    // and accumulates dot products against up to 8 input columns.
    for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {
        const uint cols_remaining = batch_size - col_base;
        const uint cols = cols_remaining < 8u ? cols_remaining : 8u;

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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    gate_sums[cc]+=gate_scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    gate_sums[cc]+=gate_scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    gate_sums[cc]+=gate_scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    gate_sums[cc]+=gate_scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    up_sums[cc]+=up_scale*(dot(s0,in4[inp_base+0u])+dot(s1,in4[inp_base+1u])+dot(s2,in4[inp_base+2u])+dot(s3,in4[inp_base+3u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    up_sums[cc]+=up_scale*(dot(s0,in4[inp_base+8u])+dot(s1,in4[inp_base+9u])+dot(s2,in4[inp_base+10u])+dot(s3,in4[inp_base+11u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    up_sums[cc]+=up_scale*(dot(s0,in4[inp_base+16u])+dot(s1,in4[inp_base+17u])+dot(s2,in4[inp_base+18u])+dot(s3,in4[inp_base+19u])
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
                for (uint cc=0u; cc<cols; cc++) {
                    const uint col = col_base + cc;
                    device const float4* in4=(device const float4*)(inputs+col*k);
                    up_sums[cc]+=up_scale*(dot(s0,in4[inp_base+24u])+dot(s1,in4[inp_base+25u])+dot(s2,in4[inp_base+26u])+dot(s3,in4[inp_base+27u])
                                         +dot(s4,in4[inp_base+28u])+dot(s5,in4[inp_base+29u])+dot(s6,in4[inp_base+30u])+dot(s7,in4[inp_base+31u]));
                }
            }
            }
        }

        for (uint cc = 0u; cc < cols; cc++) {
            float gate_val = simd_sum(gate_sums[cc]);
            float up_val = simd_sum(up_sums[cc]);
            if (lane == 0u) {
                float silu_g = gate_val / (1.0f + exp(-gate_val));
                outputs[(col_base + cc) * inter_size + pos] = silu_g * up_val;
            }
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

// ═══════════════════════════════════════════════════════════════════════════
// Batched prefill attention (perf-01)
//
// The two kernels below replace the per-prompt-token attention loop that used
// to sit inside `encode_layer_prefill*` — six tiny dispatches per token per
// layer (`fused_qk_norm`, `fused_qk_rope`, `fused_kv_store`,
// `batched_attention_scores_v2`, `batched_softmax`,
// `batched_attention_weighted_sum`). For a 1501-token prompt on a 28-layer
// model that is 252 168 dispatches, which made a *batched* prefill 1.49×
// SLOWER than running the same 1500 positions through the sequential decode
// path. They are replaced by exactly **two** batch-wide dispatches per layer.
//
// Both kernels live in their own Metal library, compiled lazily by
// `metal_prefill::attention` — `metal_graph/pipelines.rs`'s combined library
// and `build.rs`'s `ACTIVE_KERNELS` whitelist are outside this package's
// ownership, and the separate-library form is the same one the optional bf16
// TE GEMM already uses (`try_compile_bf16_pipeline`).
// ═══════════════════════════════════════════════════════════════════════════

/// Query-tile height of [`MSL_PREFILL_FLASH_ATTENTION`] (prompt tokens whose
/// attention one threadgroup computes). Must match `PFA_BQ` in the MSL.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const PREFILL_FLASH_BQ: usize = 64;

/// Key-tile width of [`MSL_PREFILL_FLASH_ATTENTION`] (keys consumed per
/// online-softmax step). Must match `PFA_BK` in the MSL.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const PREFILL_FLASH_BK: usize = 32;

/// Threads per threadgroup of [`MSL_PREFILL_FLASH_ATTENTION`]
/// (`PFA_SIMDGROUPS * 32`). Must match `PFA_THREADS` in the MSL.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const PREFILL_FLASH_THREADS: usize = 256;

/// Largest `head_dim` [`MSL_PREFILL_FLASH_ATTENTION`] can serve.
///
/// The bound comes from the threadgroup-memory budget: `KVsh` is sized
/// `PFA_BK × PFA_DMAX` floats (32 × 128 × 4 B = 16 KiB) and shares a 32 KiB
/// threadgroup allocation with `Ssh` (8 KiB) and the softmax state. A model
/// with a larger `head_dim` (e.g. Bonsai 2's 256) falls back to the
/// per-token attention loop, which has no such cap.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const PREFILL_FLASH_MAX_HEAD_DIM: usize = 128;

/// Threads per threadgroup of [`MSL_PREFILL_QKV_PREPARE`].
///
/// A power of two ≤ 256 (the `shared_sum` bound), so the RMSNorm tree
/// reduction `for (stride = tpg/2; stride > 0; stride >>= 1)` is exact.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const PREFILL_QKV_PREPARE_THREADS: usize = 128;

/// Batch-wide fused QK-RMSNorm + RoPE + KV-cache store for the prefill path.
///
/// One dispatch replaces `batch × 3` single-token dispatches (`fused_qk_norm`,
/// `fused_qk_rope`, `fused_kv_store`). Each threadgroup owns one
/// `(prompt token, head slot)` pair, where the head slot enumerates
/// `nq` Q heads, then `nkv` K heads, then `nkv` V heads:
///
/// * **Q head** — RMSNorm (`q_weight`) + RoPE at the token's position,
///   written **in place** over the Q section of `qkv`. In-place is safe: a
///   thread reads and writes only its own `(d, d + half_dim)` pair, and the
///   RMSNorm reduction is separated from the first write by a threadgroup
///   barrier. This is what lets the batched path run with no new buffers —
///   `qkv` is already `batch × qkv_dim`.
/// * **K head** — RMSNorm (`k_weight`) + RoPE, converted to `half` and stored
///   straight into `k_cache` (no intermediate f32 buffer).
/// * **V head** — copied verbatim from the QKV projection into `v_cache` as
///   `half` (V is neither normed nor rotated).
///
/// The arithmetic is element-for-element the same as `fused_qk_norm_rope`
/// followed by `fused_kv_store`: `rsqrt(Σx²/head_dim + eps)`, then
/// `fma(x0, c, -(x1*s))` / `fma(x0, s, x1*c)`, then a `half` narrowing cast.
///
/// Buffers:
///   - `qkv`       `[batch × qkv_dim]` f32 column-major, `qkv_dim = (nq+2·nkv)·head_dim` (in/out)
///   - `q_weight`  `[head_dim]` f32
///   - `k_weight`  `[head_dim]` f32
///   - `cos_buf`   `[batch × half_dim]` f32
///   - `sin_buf`   `[batch × half_dim]` f32
///   - `k_cache`   `[n_layers × nkv × max_seq × head_dim]` f16
///   - `v_cache`   `[n_layers × nkv × max_seq × head_dim]` f16
///   - scalars: `nq`, `nkv`, `head_dim`, `eps`, `max_seq`, `pos_start`,
///     `batch_size`, `layer_offset`
///
/// Dispatch: `[batch, nq + 2·nkv, 1]` threadgroups,
/// `[PREFILL_QKV_PREPARE_THREADS, 1, 1]` threads.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_PREFILL_QKV_PREPARE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Launch width — must match PREFILL_QKV_PREPARE_THREADS on the Rust side.
// Taken as a constant rather than [[threads_per_threadgroup]] because MSL
// requires every grid-geometry attribute of a kernel to have the same width,
// and [[threadgroup_position_in_grid]] here is a uint2.
constant constexpr uint PQP_THREADS = 128u;

kernel void prefill_qkv_prepare(
    device float* qkv            [[buffer(0)]],
    device const float* q_weight [[buffer(1)]],
    device const float* k_weight [[buffer(2)]],
    device const float* cos_buf  [[buffer(3)]],
    device const float* sin_buf  [[buffer(4)]],
    device half* k_cache         [[buffer(5)]],
    device half* v_cache         [[buffer(6)]],
    constant uint& nq            [[buffer(7)]],
    constant uint& nkv           [[buffer(8)]],
    constant uint& head_dim      [[buffer(9)]],
    constant float& eps          [[buffer(10)]],
    constant uint& max_seq       [[buffer(11)]],
    constant uint& pos_start     [[buffer(12)]],
    constant uint& batch_size    [[buffer(13)]],
    constant uint& layer_offset  [[buffer(14)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint t    = tgid.x;              // prompt token within the batch
    const uint slot = tgid.y;              // head slot: Q | K | V
    // Both guards are threadgroup-uniform, so the barriers below stay uniform.
    if (t >= batch_size || slot >= nq + 2u * nkv) {
        return;
    }

    const uint qkv_dim  = (nq + 2u * nkv) * head_dim;
    const uint half_dim = head_dim / 2u;
    const uint pos      = pos_start + t;

    // ── V head: verbatim copy into the KV cache (no norm, no RoPE). ────────
    if (slot >= nq + nkv) {
        const uint hv = slot - nq - nkv;
        device const float* src =
            qkv + t * qkv_dim + (nq + nkv) * head_dim + hv * head_dim;
        device half* dst = v_cache + layer_offset + (hv * max_seq + pos) * head_dim;
        for (uint i = tid; i < head_dim; i += PQP_THREADS) {
            dst[i] = half(src[i]);
        }
        return;
    }

    const bool is_q = (slot < nq);
    const uint head = is_q ? slot : (slot - nq);
    device float* base =
        qkv + t * qkv_dim + (is_q ? 0u : nq * head_dim) + head * head_dim;
    device const float* w = is_q ? q_weight : k_weight;

    // ── RMSNorm reduction over this head's head_dim elements. ──────────────
    threadgroup float shared_sum[PQP_THREADS];
    float local_sq = 0.0f;
    for (uint i = tid; i < head_dim; i += PQP_THREADS) {
        const float v = base[i];
        local_sq += v * v;
    }
    shared_sum[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = PQP_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rms_inv = rsqrt(shared_sum[0] / float(head_dim) + eps);

    device const float* cos_row = cos_buf + t * half_dim;
    device const float* sin_row = sin_buf + t * half_dim;

    // ── Normalise + rotate. Q writes back in place; K goes to the cache. ───
    if (is_q) {
        for (uint d = tid; d < half_dim; d += PQP_THREADS) {
            const float x0 = base[d] * rms_inv * w[d];
            const float x1 = base[d + half_dim] * rms_inv * w[d + half_dim];
            const float c = cos_row[d];
            const float s = sin_row[d];
            base[d]            = fma(x0, c, -(x1 * s));
            base[d + half_dim] = fma(x0, s,   x1 * c);
        }
    } else {
        device half* dst = k_cache + layer_offset + (head * max_seq + pos) * head_dim;
        for (uint d = tid; d < half_dim; d += PQP_THREADS) {
            const float x0 = base[d] * rms_inv * w[d];
            const float x1 = base[d + half_dim] * rms_inv * w[d + half_dim];
            const float c = cos_row[d];
            const float s = sin_row[d];
            dst[d]            = half(fma(x0, c, -(x1 * s)));
            dst[d + half_dim] = half(fma(x0, s,   x1 * c));
        }
    }
}
"#;

/// Batch-wide **causal GQA flash attention** over the prefill batch.
///
/// One dispatch replaces `batch × 3` single-token dispatches
/// (`batched_attention_scores_v2`, `batched_softmax`,
/// `batched_attention_weighted_sum`) and, critically, never materialises the
/// `batch × n_ctx` score matrix: it is the same flash-attention v2
/// (online-softmax) structure as the shipping DiT kernel
/// `joint_attention_flash_f32`, driving `simdgroup_float8x8` hardware matrix
/// units for both `S = scale·Q·Kᵀ` and `O += P·V`.
///
/// Differences from the DiT kernel, all four load-bearing here:
///
/// 1. **Causal mask.** Query row `r` of the tile sits at absolute position
///    `pos_start + q0 + r` and may only attend to keys `0 ..= that position`.
///    Masking is applied inside the online softmax (masked keys never enter
///    the running max and get `P = 0`), and whole key-tiles beyond the tile's
///    last query position are never iterated.
/// 2. **GQA.** `kv_head = q_head / heads_per_group`.
/// 3. **`half` K/V read from the persistent KV cache**, whose layout is
///    `[layer][kv_head][max_seq][head_dim]` — so the stage loop adds
///    `layer_offset + kv_head*max_seq*head_dim` and converts to f32 exactly
///    like `batched_attention_scores_v2`'s `float(key[i])`.
/// 4. **Strided Q / out.** Q is read straight out of the (already RoPE'd,
///    in-place) Q section of the batched `qkv` buffer, so the 8×8 A-fragment
///    load uses `ld = q_row_stride = qkv_dim`; the output is the batched
///    `attn_out` buffer with `ld = out_row_stride = nq·head_dim`.
///
/// The online softmax is mathematically exact versus the full-row softmax the
/// per-token path computed (f32 reassociation only).
///
/// Buffers:
///   - `q`        `[batch × q_row_stride]` f32 (post-norm, post-RoPE Q section)
///   - `k_cache`  `[n_layers × nkv × max_seq × head_dim]` f16
///   - `v_cache`  `[n_layers × nkv × max_seq × head_dim]` f16
///   - `out`      `[batch × out_row_stride]` f32
///   - scalars: `nq`, `nkv`, `heads_per_group`, `head_dim`, `q_row_stride`,
///     `out_row_stride`, `max_seq`, `pos_start`, `batch_size`, `scale`,
///     `layer_offset`
///
/// Dispatch: `[ceil(batch/PFA_BQ), nq, 1]` threadgroups,
/// `[PREFILL_FLASH_THREADS, 1, 1]` threads. Requires
/// `head_dim % 8 == 0` and `head_dim ≤ PREFILL_FLASH_MAX_HEAD_DIM`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_PREFILL_FLASH_ATTENTION: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Tile / simdgroup geometry — must match the PREFILL_FLASH_* Rust constants
// and dispatch_prefill_flash_attention.
constant constexpr uint PFA_BQ = 64u;                       // query rows / threadgroup
constant constexpr uint PFA_BK = 32u;                       // keys / online-softmax step
constant constexpr uint PFA_SIMDGROUPS = 8u;
constant constexpr uint PFA_THREADS = PFA_SIMDGROUPS * 32u; // 256
constant constexpr uint PFA_FRAG = 8u;                      // hardware 8x8 edge
constant constexpr uint PFA_SG_M = PFA_BQ / PFA_SIMDGROUPS; // 8 query rows / simdgroup
constant constexpr uint PFA_MFRAGS = PFA_SG_M / PFA_FRAG;   // 1 M-fragment / simdgroup
constant constexpr uint PFA_NFRAGS = PFA_BK / PFA_FRAG;     // 4 S-column fragments
constant constexpr uint PFA_DMAX = 128u;                    // head_dim cap
constant constexpr uint PFA_DFRAGS_MAX = PFA_DMAX / PFA_FRAG;

kernel void prefill_flash_attention(
    device const float* q            [[buffer(0)]],
    device const half* k_cache       [[buffer(1)]],
    device const half* v_cache       [[buffer(2)]],
    device float* out                [[buffer(3)]],
    constant uint& nq                [[buffer(4)]],
    constant uint& nkv               [[buffer(5)]],
    constant uint& heads_per_group   [[buffer(6)]],
    constant uint& head_dim          [[buffer(7)]],
    constant uint& q_row_stride      [[buffer(8)]],
    constant uint& out_row_stride    [[buffer(9)]],
    constant uint& max_seq           [[buffer(10)]],
    constant uint& pos_start         [[buffer(11)]],
    constant uint& batch_size        [[buffer(12)]],
    constant float& scale            [[buffer(13)]],
    constant uint& layer_offset      [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const uint h = tgid.y;                      // query head
    if (h >= nq || nkv == 0u) {
        return;
    }
    const uint q0 = tgid.x * PFA_BQ;            // first prompt token of this tile
    if (q0 >= batch_size) {
        return;
    }
    const uint q_valid = min(PFA_BQ, batch_size - q0);
    const uint kv_head = h / heads_per_group;
    const uint head_off = layer_offset + kv_head * max_seq * head_dim;
    const uint d_frags = head_dim / PFA_FRAG;
    const uint sg_m0 = sgid * PFA_SG_M;         // tile-local first query row
    const uint q_row0 = q0 + sg_m0;             // batch-global first query row

    threadgroup float KVsh[PFA_BK * PFA_DMAX];  // KT[head_dim][BK] then V[BK][head_dim]
    threadgroup float Ssh[PFA_BQ * PFA_BK];     // S, then P
    threadgroup float mrow[PFA_BQ];             // running max
    threadgroup float lrow[PFA_BQ];             // running normaliser
    threadgroup float corr[PFA_BQ];             // per-row rescale this tile
    threadgroup float diagsh[PFA_SIMDGROUPS * PFA_MFRAGS * PFA_FRAG * PFA_FRAG];

    simdgroup_float8x8 oacc[PFA_MFRAGS][PFA_DFRAGS_MAX];
    for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
        for (uint di = 0u; di < d_frags; di++) {
            oacc[mi][di] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    if (lid < PFA_BQ) {
        mrow[lid] = -INFINITY;
        lrow[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Keys any row of this tile can see: 0 ..= (pos_start + q0 + q_valid - 1).
    const uint kv_len = pos_start + q0 + q_valid;
    const uint k_tiles = (kv_len + PFA_BK - 1u) / PFA_BK;

    for (uint kt = 0u; kt < k_tiles; kt++) {
        const uint k0 = kt * PFA_BK;
        const uint k_valid = min(PFA_BK, kv_len - k0);

        // Stage K-transposed: KVsh[d*PFA_BK + j] = k_cache[kv_head, k0+j, d].
        for (uint i = lid; i < head_dim * PFA_BK; i += PFA_THREADS) {
            const uint d = i / PFA_BK;
            const uint j = i % PFA_BK;
            float val = 0.0f;
            if (j < k_valid) {
                val = float(k_cache[head_off + (k0 + j) * head_dim + d]);
            }
            KVsh[d * PFA_BK + j] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S[PFA_BQ, PFA_BK] = Q * K^T via 8x8 MACs (scale applied later).
        simdgroup_float8x8 sacc[PFA_MFRAGS][PFA_NFRAGS];
        for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
            for (uint ni = 0u; ni < PFA_NFRAGS; ni++) {
                sacc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        for (uint kf = 0u; kf < d_frags; kf++) {
            simdgroup_float8x8 qfrag[PFA_MFRAGS];
            simdgroup_float8x8 kfrag[PFA_NFRAGS];
            for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
                const uint qr = q_row0 + mi * PFA_FRAG;
                // Out-of-range rows are clamped so the load stays in bounds;
                // their S is never read (q_valid guards softmax + writeback).
                const uint qr_c = (qr < batch_size) ? qr : (batch_size - 1u);
                const device float* qsrc =
                    q + qr_c * q_row_stride + h * head_dim + kf * PFA_FRAG;
                simdgroup_load(qfrag[mi], qsrc, q_row_stride);
            }
            for (uint ni = 0u; ni < PFA_NFRAGS; ni++) {
                const uint koff = (kf * PFA_FRAG) * PFA_BK + ni * PFA_FRAG;
                simdgroup_load(kfrag[ni], KVsh + koff, PFA_BK);
            }
            for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
                for (uint ni = 0u; ni < PFA_NFRAGS; ni++) {
                    simdgroup_multiply_accumulate(
                        sacc[mi][ni], qfrag[mi], kfrag[ni], sacc[mi][ni]);
                }
            }
        }
        for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
            for (uint ni = 0u; ni < PFA_NFRAGS; ni++) {
                const uint soff = (sg_m0 + mi * PFA_FRAG) * PFA_BK + ni * PFA_FRAG;
                simdgroup_store(sacc[mi][ni], Ssh + soff, PFA_BK);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Causal online softmax (one query row per strided thread).
        for (uint r = lid; r < PFA_BQ; r += PFA_THREADS) {
            float c = 1.0f;
            // Keys of THIS tile visible to row r: j < row_valid.
            uint row_valid = 0u;
            if (r < q_valid) {
                const uint qpos = pos_start + q0 + r;
                if (k0 <= qpos) {
                    row_valid = min(k_valid, qpos - k0 + 1u);
                }
            }
            if (row_valid == 0u) {
                // Padded query row, or a key-tile entirely in this row's
                // future: contribute nothing and leave (m, l, O) untouched.
                for (uint j = 0u; j < PFA_BK; j++) {
                    Ssh[r * PFA_BK + j] = 0.0f;
                }
            } else {
                float tile_max = -INFINITY;
                for (uint j = 0u; j < row_valid; j++) {
                    tile_max = max(tile_max, Ssh[r * PFA_BK + j] * scale);
                }
                const float m_old = mrow[r];
                const float m_new = max(m_old, tile_max);
                float tile_sum = 0.0f;
                for (uint j = 0u; j < PFA_BK; j++) {
                    float p = 0.0f;
                    if (j < row_valid) {
                        p = exp(Ssh[r * PFA_BK + j] * scale - m_new);
                    }
                    Ssh[r * PFA_BK + j] = p;
                    tile_sum += p;
                }
                c = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
                lrow[r] = lrow[r] * c + tile_sum;
                mrow[r] = m_new;
            }
            corr[r] = c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale the O accumulator: O = diag(corr) * O.
        {
            const uint dbase = (sgid * PFA_MFRAGS) * (PFA_FRAG * PFA_FRAG);
            const uint lane = lid % 32u;
            for (uint idx = lane; idx < PFA_MFRAGS * PFA_FRAG * PFA_FRAG; idx += 32u) {
                const uint mi = idx / (PFA_FRAG * PFA_FRAG);
                const uint e  = idx % (PFA_FRAG * PFA_FRAG);
                const uint rr = e / PFA_FRAG;
                const uint cc = e % PFA_FRAG;
                float val = 0.0f;
                if (rr == cc) {
                    val = corr[sg_m0 + mi * PFA_FRAG + rr];
                }
                diagsh[dbase + idx] = val;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint dbase = (sgid * PFA_MFRAGS) * (PFA_FRAG * PFA_FRAG);
            for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
                simdgroup_float8x8 dfrag;
                simdgroup_load(dfrag, diagsh + dbase + mi * (PFA_FRAG * PFA_FRAG), PFA_FRAG);
                for (uint di = 0u; di < d_frags; di++) {
                    simdgroup_float8x8 tmp;
                    simdgroup_multiply(tmp, dfrag, oacc[mi][di]);
                    oacc[mi][di] = tmp;
                }
            }
        }

        // Stage V over the (now dead) K tile, then O += P * V.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < PFA_BK * head_dim; i += PFA_THREADS) {
            const uint j = i / head_dim;
            const uint d = i % head_dim;
            float val = 0.0f;
            if (j < k_valid) {
                val = float(v_cache[head_off + (k0 + j) * head_dim + d]);
            }
            KVsh[j * head_dim + d] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kf = 0u; kf < PFA_NFRAGS; kf++) {
            simdgroup_float8x8 pfrag[PFA_MFRAGS];
            for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
                const uint poff = (sg_m0 + mi * PFA_FRAG) * PFA_BK + kf * PFA_FRAG;
                simdgroup_load(pfrag[mi], Ssh + poff, PFA_BK);
            }
            for (uint di = 0u; di < d_frags; di++) {
                simdgroup_float8x8 vfrag;
                const uint voff = (kf * PFA_FRAG) * head_dim + di * PFA_FRAG;
                simdgroup_load(vfrag, KVsh + voff, head_dim);
                for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
                    simdgroup_multiply_accumulate(oacc[mi][di], pfrag[mi], vfrag, oacc[mi][di]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Writeback: out[t, h*head_dim + d] = O[t,d] / lrow[t].
    threadgroup float* Csh = KVsh;
    for (uint sg = 0u; sg < PFA_SIMDGROUPS; sg++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
                for (uint di = 0u; di < d_frags; di++) {
                    const uint coff = (mi * PFA_FRAG) * head_dim + di * PFA_FRAG;
                    simdgroup_store(oacc[mi][di], Csh + coff, head_dim);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint idx = lid % 32u; idx < PFA_SG_M * head_dim; idx += 32u) {
                const uint mm = idx / head_dim;
                const uint d  = idx % head_dim;
                const uint tile_row = sg_m0 + mm;
                if (tile_row < q_valid) {
                    const float l = lrow[tile_row];
                    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    out[(q0 + tile_row) * out_row_stride + h * head_dim + d] =
                        Csh[mm * head_dim + d] * inv;
                }
            }
        }
    }
}
"#;

#[cfg(test)]
mod prefill_attention_tests {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use super::*;

    /// The MSL tile literals and the exported Rust constants are two copies of
    /// the same geometry; the dispatch layer computes its grid from the Rust
    /// side, so a silent divergence would launch the wrong grid.
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn prefill_flash_tile_constants_match_the_msl() {
        assert_eq!(PREFILL_FLASH_BQ, 64);
        assert_eq!(PREFILL_FLASH_BK, 32);
        assert_eq!(PREFILL_FLASH_THREADS, 256);
        assert_eq!(PREFILL_FLASH_MAX_HEAD_DIM, 128);
        assert!(MSL_PREFILL_FLASH_ATTENTION.contains("PFA_BQ = 64u"));
        assert!(MSL_PREFILL_FLASH_ATTENTION.contains("PFA_BK = 32u"));
        assert!(MSL_PREFILL_FLASH_ATTENTION.contains("PFA_SIMDGROUPS = 8u"));
        assert!(MSL_PREFILL_FLASH_ATTENTION.contains("PFA_DMAX = 128u"));
        assert!(PREFILL_QKV_PREPARE_THREADS.is_power_of_two());
        // The MSL `shared_sum` array is 256 floats and the tree reduction
        // halves `threads_per_threadgroup`, so the launch width must be a
        // power of two no larger than that.
        const { assert!(PREFILL_QKV_PREPARE_THREADS <= 256) };
    }

    /// Both kernels must declare the entry points the dispatch layer looks up
    /// by name, and the flash kernel must really use the matrix units.
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn prefill_attention_sources_declare_their_entry_points() {
        assert!(MSL_PREFILL_QKV_PREPARE.contains("kernel void prefill_qkv_prepare"));
        assert!(MSL_PREFILL_QKV_PREPARE
            .contains(&format!("PQP_THREADS = {PREFILL_QKV_PREPARE_THREADS}u")));
        assert!(MSL_PREFILL_FLASH_ATTENTION.contains("kernel void prefill_flash_attention"));
        assert!(MSL_PREFILL_FLASH_ATTENTION.contains("#include <metal_simdgroup_matrix>"));
        assert!(MSL_PREFILL_FLASH_ATTENTION.contains("simdgroup_multiply_accumulate"));
    }
}
