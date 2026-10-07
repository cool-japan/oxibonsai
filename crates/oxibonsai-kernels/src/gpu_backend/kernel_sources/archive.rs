//! Archived/historical Metal and CUDA kernel sources.
//!
//! These kernels are kept for reference and testing.
//! Active decode kernels are in `decode.rs` (V7).
//! Active prefill kernels are in `prefill.rs` (V7 GEMM).

#![allow(dead_code)]

/// Q1_0_g128 GEMV — simdgroup parallel reduction.
///
/// Each simdgroup (32 threads) cooperatively computes one output row.
/// With 256 threads/threadgroup, 8 rows are computed per threadgroup.
///
/// Buffers: `"x"` → blocks (0), `"y"` → input (1), `"result"` → output (2)
/// Scalars: `"n"` → n_rows (3), `"k"` → k (4)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float4* input4        [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& k                   [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    // 8 simdgroups per threadgroup, each handles one output row
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint row_byte_offset = row * blocks_per_row * 18u;
    float local_sum = 0.0f;

    // Each lane in the simdgroup processes blocks_per_row/32 blocks
    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint blk = row_byte_offset + b * 18u;
        const float scale = float(*(device const half*)(blocks_raw + blk));
        device const uchar* qs = blocks_raw + blk + 2u;

        const uint inp_base = b * 32u;
        float block_sum = 0.0f;

        for (uint w = 0u; w < 4u; w++) {
            uint bits = uint(qs[w * 4u])
                      | (uint(qs[w * 4u + 1u]) << 8u)
                      | (uint(qs[w * 4u + 2u]) << 16u)
                      | (uint(qs[w * 4u + 3u]) << 24u);

            const uint f4_base = inp_base + w * 8u;

            for (uint i = 0u; i < 8u; i++) {
                float4 inp = input4[f4_base + i];
                float4 signs = float4(
                    select(-1.0f, 1.0f, bool(bits & 1u)),
                    select(-1.0f, 1.0f, bool(bits & 2u)),
                    select(-1.0f, 1.0f, bool(bits & 4u)),
                    select(-1.0f, 1.0f, bool(bits & 8u))
                );
                block_sum += dot(signs, inp);
                bits >>= 4u;
            }
        }
        local_sum += scale * block_sum;
    }

    // Hardware-accelerated parallel reduction across 32 lanes
    float row_sum = simd_sum(local_sum);

    // Lane 0 writes the final result
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

/// Optimized Q1_0_g128 GEMV — 4 rows per simdgroup, register-cached input.
///
/// Each simdgroup (32 threads) computes 4 output rows simultaneously,
/// amortizing input loads across rows. 2 simdgroups per threadgroup
/// (64 threads) for simpler scheduling.
///
/// Key improvements over V1:
/// - 4× row throughput per simdgroup (N_R=4)
/// - 16-element register cache (`yl[16]`) reused across all rows
/// - Stride-4 block processing for better memory coalescing
/// - `fma()` for scale accumulation
///
/// Buffers: `"x"` → blocks (0), `"y"` → input (1, float* not float4*),
///          `"result"` → output (2)
/// Scalars: `"n"` → n_rows (3), `"k"` → k (4)
///
/// Dispatch: `[ceil(n_rows / 8), 1, 1]` threadgroups, `[64, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V2: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v2(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float* input_f32      [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& k                   [[buffer(4)]],
    uint tgpig  [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint NR = 4u;    // rows per simdgroup
    const uint NSG = 2u;   // simdgroups per threadgroup

    const uint first_row = (tgpig * NSG + uint(sgitg)) * NR;
    if (first_row >= n_rows) return;

    const uint nb = k / 128u;  // blocks per row

    // Thread partitioning: 32 threads handle 128 elements
    // Each thread processes 16 contiguous elements from a block
    const uint ix = uint(tiisg) / 8u;          // which block offset (0-3)
    const uint il = (uint(tiisg) % 8u) * 16u;  // which 16-element chunk within block

    // Accumulators for NR rows
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float yl[16];

    // Input pointer: start at ix-th block, il-th element
    device const float* yb = input_f32 + ix * 128u + il;

    // Process blocks with stride of 4 (32 threads / 8 threads per block)
    for (uint ib = ix; ib < nb; ib += 4u) {
        // Load 16 input elements into registers
        for (uint i = 0u; i < 16u; i++) {
            yl[i] = yb[i];
        }

        // Process all NR rows with the same cached input
        for (uint row_offset = 0u; row_offset < NR; row_offset++) {
            uint row = first_row + row_offset;
            if (row >= n_rows) break;

            uint blk_byte = row * nb * 18u + ib * 18u;
            float scale = float(*(device const half*)(blocks_raw + blk_byte));
            device const uchar* qs = blocks_raw + blk_byte + 2u;

            // Compute byte offset for our 16-element chunk
            uint bit_offset = il;  // which bit position in the 128-bit field
            uint byte_start = bit_offset / 8u;

            float chunk_sum = 0.0f;

            // Process 16 elements: 2 bytes of bit data
            uint bits_lo = uint(qs[byte_start]);
            uint bits_hi = uint(qs[byte_start + 1u]);
            uint bits16 = bits_lo | (bits_hi << 8u);

            for (uint i = 0u; i < 16u; i++) {
                float sign = ((bits16 >> i) & 1u) != 0u ? 1.0f : -1.0f;
                chunk_sum = fma(sign, yl[i], chunk_sum);
            }

            sumf[row_offset] = fma(scale, chunk_sum, sumf[row_offset]);
        }

        yb += 128u * 4u;  // advance by 4 blocks
    }

    // Simdgroup reduction + write
    for (uint row_offset = 0u; row_offset < NR; row_offset++) {
        uint row = first_row + row_offset;
        if (row >= n_rows) continue;
        float tot = simd_sum(sumf[row_offset]);
        if (tiisg == 0u) {
            output[row] = tot;
        }
    }
}
"#;

/// V3 Q1_0_g128 GEMV — shared memory input caching with bank-conflict-free reads.
///
/// All 32 lanes in a simdgroup cooperate on each Q1_0_g128 block.
/// Each lane handles 4 elements (one float4) from the block, reading from
/// threadgroup shared memory instead of device memory.
///
/// Key improvements over V1:
/// - Only one coalesced load of input per threadgroup (vs 8 redundant loads in V1)
/// - Bank-conflict-free shared memory reads (lane L reads bank L)
/// - Tiled processing supports k > 4096 (e.g. down_proj k=12288)
/// - 16KB shared memory tile (fits in 32KB threadgroup memory limit)
///
/// Buffers: `"x"` → blocks (0), `"y"` → input (1, float4*),
///          `"result"` → output (2)
/// Scalars: `"n"` → n_rows (3), `"k"` → k (4)
///
/// Dispatch: `[ceil(n_rows / 8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V3: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v3(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float4* input4        [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& k                   [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    // Tile size: 4096 floats = 1024 float4 = 32 Q1 blocks = 16KB
    // Fits in 32KB threadgroup memory limit
    const uint TILE_F4 = 1024u;
    const uint TILE_BLOCKS = 32u;

    threadgroup float4 shared_input4[1024]; // 16KB

    const uint rows_per_tg = 8u;
    const uint row = tgid * rows_per_tg + sgid;
    const uint tid = sgid * 32u + lane; // 0..255

    const uint blocks_per_row = k / 128u;
    float local_sum = 0.0f;

    // Process input in tiles of TILE_BLOCKS (handles k > 4096, e.g. down_proj k=12288)
    for (uint tile = 0u; tile * TILE_BLOCKS < blocks_per_row; tile++) {
        uint block_start = tile * TILE_BLOCKS;
        uint f4_start = tile * TILE_F4;
        uint tile_blocks = min(TILE_BLOCKS, blocks_per_row - block_start);
        uint tile_f4 = tile_blocks * 32u;

        // Phase 1: Cooperative coalesced load of input tile into shared memory
        // 256 threads load up to 1024 float4 = 4 loads per thread (for full tile)
        for (uint i = tid; i < tile_f4; i += 256u) {
            shared_input4[i] = input4[f4_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: Each simdgroup processes its row for this tile
        if (row < n_rows) {
            const uint row_bytes = row * blocks_per_row * 18u;

            // All 32 lanes cooperate on each block
            for (uint b = 0u; b < tile_blocks; b++) {
                const uint global_b = block_start + b;
                const uint blk_off = row_bytes + global_b * 18u;
                const float scale = float(*(device const half*)(blocks_raw + blk_off));
                device const uchar* qs = blocks_raw + blk_off + 2u;

                // Each lane reads 1 float4 (4 elements) from shared memory
                // Lane L reads shared_input4[b*32 + L] -> bank L -> NO bank conflict
                float4 inp = shared_input4[b * 32u + lane];

                // Extract the 4 bits for this lane's elements
                // Lane L handles elements [L*4, L*4+3] = byte L/2, bits [(L%2)*4, (L%2)*4+3]
                uchar byte_val = qs[lane >> 1u];
                uint bits = uint(byte_val) >> ((lane & 1u) << 2u);

                float4 signs;
                signs.x = select(-1.0f, 1.0f, bool(bits & 1u));
                signs.y = select(-1.0f, 1.0f, bool(bits & 2u));
                signs.z = select(-1.0f, 1.0f, bool(bits & 4u));
                signs.w = select(-1.0f, 1.0f, bool(bits & 8u));

                local_sum += scale * dot(signs, inp);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < n_rows) {
        float row_sum = simd_sum(local_sum);
        if (lane == 0u) {
            output[row] = row_sum;
        }
    }
}
"#;

/// V4 Q1_0_g128 GEMV — half-precision dot products with float32 accumulation.
///
/// Same structure as V1 but converts float4 input to half4 in-register
/// and uses `dot(half4, half4)` which runs at 2× throughput on Apple GPU.
/// Sign extraction also produces half4 values.
///
/// Buffers: `"x"` → blocks (0), `"y"` → input (1, float4*),
///          `"result"` → output (2)
/// Scalars: `"n"` → n_rows (3), `"k"` → k (4)
///
/// Dispatch: `[ceil(n_rows / 8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V4: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v4(
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
    const uint row_byte_offset = row * blocks_per_row * 18u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint blk = row_byte_offset + b * 18u;
        const float scale = float(*(device const half*)(blocks_raw + blk));
        device const uchar* qs = blocks_raw + blk + 2u;

        const uint inp_base = b * 32u;
        float block_sum = 0.0f;

        for (uint w = 0u; w < 4u; w++) {
            uint bits = uint(qs[w * 4u])
                      | (uint(qs[w * 4u + 1u]) << 8u)
                      | (uint(qs[w * 4u + 2u]) << 16u)
                      | (uint(qs[w * 4u + 3u]) << 24u);

            const uint f4_base = inp_base + w * 8u;

            for (uint i = 0u; i < 8u; i++) {
                half4 inp = half4(input4[f4_base + i]);
                half4 signs = half4(
                    select(half(-1.0h), half(1.0h), bool(bits & 1u)),
                    select(half(-1.0h), half(1.0h), bool(bits & 2u)),
                    select(half(-1.0h), half(1.0h), bool(bits & 4u)),
                    select(half(-1.0h), half(1.0h), bool(bits & 8u))
                );
                block_sum += float(dot(signs, inp));
                bits >>= 4u;
            }
        }
        local_sum += scale * block_sum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

/// V5 Q1_0_g128 GEMV — 32 rows per threadgroup (1024 threads = max Apple GPU TG size).
///
/// Same algorithm as V1 but with 32 simdgroups per threadgroup instead of 8.
/// This maximizes GPU occupancy to better hide memory latency on Apple Silicon.
///
/// Buffers: `"x"` → blocks (0), `"y"` → input (1, float4*),
///          `"result"` → output (2)
/// Scalars: `"n"` → n_rows (3), `"k"` → k (4)
///
/// Dispatch: `[ceil(n_rows / 32), 1, 1]` threadgroups, `[1024, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V5: &str = r#"
#include <metal_stdlib>
using namespace metal;

// V5: 32 rows per threadgroup (1024 threads = 32 simdgroups × 32 lanes)
// This maximizes GPU occupancy to better hide memory latency.
kernel void gemv_q1_g128_v5(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float4* input4        [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& k                   [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    // 32 simdgroups per threadgroup, each handles one output row
    const uint row = tgid * 32u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint row_byte_offset = row * blocks_per_row * 18u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint blk = row_byte_offset + b * 18u;
        const float scale = float(*(device const half*)(blocks_raw + blk));
        device const uchar* qs = blocks_raw + blk + 2u;

        const uint inp_base = b * 32u;
        float block_sum = 0.0f;

        for (uint w = 0u; w < 4u; w++) {
            uint bits = uint(qs[w * 4u])
                      | (uint(qs[w * 4u + 1u]) << 8u)
                      | (uint(qs[w * 4u + 2u]) << 16u)
                      | (uint(qs[w * 4u + 3u]) << 24u);

            const uint f4_base = inp_base + w * 8u;

            for (uint i = 0u; i < 8u; i++) {
                float4 inp = input4[f4_base + i];
                float4 signs = float4(
                    select(-1.0f, 1.0f, bool(bits & 1u)),
                    select(-1.0f, 1.0f, bool(bits & 2u)),
                    select(-1.0f, 1.0f, bool(bits & 4u)),
                    select(-1.0f, 1.0f, bool(bits & 8u))
                );
                block_sum += dot(signs, inp);
                bits >>= 4u;
            }
        }
        local_sum += scale * block_sum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

/// Q1_0_g128 GEMV V6 — 4 rows per threadgroup (128 threads = 4 simdgroups × 32 lanes).
///
/// Same algorithm as V1 but with smaller threadgroups.
/// Hypothesis: smaller TGs allow more concurrent TGs per GPU core,
/// improving memory latency hiding.
///
/// Dispatch: `[ceil(n_rows/4), 1, 1]` threadgroups, `[128, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V6: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v6(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float4* input4        [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& k                   [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    // 4 simdgroups per threadgroup, each handles one output row
    const uint row = tgid * 4u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint row_byte_offset = row * blocks_per_row * 18u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint blk = row_byte_offset + b * 18u;
        const float scale = float(*(device const half*)(blocks_raw + blk));
        device const uchar* qs = blocks_raw + blk + 2u;

        const uint inp_base = b * 32u;
        float block_sum = 0.0f;

        for (uint w = 0u; w < 4u; w++) {
            uint bits = uint(qs[w * 4u])
                      | (uint(qs[w * 4u + 1u]) << 8u)
                      | (uint(qs[w * 4u + 2u]) << 16u)
                      | (uint(qs[w * 4u + 3u]) << 24u);

            const uint f4_base = inp_base + w * 8u;

            for (uint i = 0u; i < 8u; i++) {
                float4 inp = input4[f4_base + i];
                float4 signs = float4(
                    select(-1.0f, 1.0f, bool(bits & 1u)),
                    select(-1.0f, 1.0f, bool(bits & 2u)),
                    select(-1.0f, 1.0f, bool(bits & 4u)),
                    select(-1.0f, 1.0f, bool(bits & 8u))
                );
                block_sum += dot(signs, inp);
                bits >>= 4u;
            }
        }
        local_sum += scale * block_sum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

/// Q1_0_g128 GEMM — 2-D grid: one thread per (weight_row, batch_col).
///
/// Buffers: `"x"` → blocks (0), `"y"` → input (1), `"result"` → output (2)
/// Scalars: `"n"` → n_rows (3), `"m"` → m (4), `"k"` → k (5)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_Q1_G128: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_q1_g128(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& m                   [[buffer(4)]],
    constant uint& k                   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.x;
    uint col = gid.y;

    if (row >= n_rows || col >= m) return;

    const uint blocks_per_row = k / 128u;
    const uint row_byte_offset = row * blocks_per_row * 18u;
    float acc = 0.0f;

    // Cast column's input slice to float4 for vectorised reads
    device const float4* col_input4 = (device const float4*)(input + col * k);

    for (uint b = 0u; b < blocks_per_row; b++) {
        const uint blk = row_byte_offset + b * 18u;
        const float scale = float(*(device const half*)(blocks_raw + blk));
        device const uchar* qs = blocks_raw + blk + 2u;

        const uint inp_base = b * 32u;
        float block_sum = 0.0f;

        for (uint w = 0u; w < 4u; w++) {
            uint bits = uint(qs[w * 4u])
                      | (uint(qs[w * 4u + 1u]) << 8u)
                      | (uint(qs[w * 4u + 2u]) << 16u)
                      | (uint(qs[w * 4u + 3u]) << 24u);

            const uint f4_base = inp_base + w * 8u;

            for (uint i = 0u; i < 8u; i++) {
                float4 inp = col_input4[f4_base + i];
                float4 signs = float4(
                    select(-1.0f, 1.0f, bool(bits & 1u)),
                    select(-1.0f, 1.0f, bool(bits & 2u)),
                    select(-1.0f, 1.0f, bool(bits & 4u)),
                    select(-1.0f, 1.0f, bool(bits & 8u))
                );
                block_sum += dot(signs, inp);
                bits >>= 4u;
            }
        }

        acc += scale * block_sum;
    }

    output[col * n_rows + row] = acc;
}
"#;

/// Fused GEMV + residual add: `output[row] = residual[row] + gemv_result[row]`.
///
/// Identical to V1 (`gemv_q1_g128`) except the final write adds the GEMV
/// result to a residual vector instead of storing it directly.  This fuses
/// two dispatches (GEMV + residual_add) into one, eliminating a separate
/// kernel launch and an extra read/write of the output buffer.
///
/// The `output` and `residual` buffers are allowed to alias (same buffer).
/// Since only lane 0 per row writes and each row is independent, there is
/// no data race.
///
/// Buffers: `"x"` → blocks (0), `"y"` → input (1, float4*),
///          `"result"` → output (2), `"n"` → n_rows (3), `"k"` → k (4),
///          residual (5)
///
/// Dispatch: `[ceil(n_rows / 8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_RESIDUAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Fused GEMV + residual add: output[row] = residual[row] + scale * dot(signs, input)
kernel void gemv_q1_g128_residual(
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
    const uint row_byte_offset = row * blocks_per_row * 18u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint blk = row_byte_offset + b * 18u;
        const float scale = float(*(device const half*)(blocks_raw + blk));
        device const uchar* qs = blocks_raw + blk + 2u;

        const uint inp_base = b * 32u;
        float block_sum = 0.0f;

        for (uint w = 0u; w < 4u; w++) {
            uint bits = uint(qs[w * 4u])
                      | (uint(qs[w * 4u + 1u]) << 8u)
                      | (uint(qs[w * 4u + 2u]) << 16u)
                      | (uint(qs[w * 4u + 3u]) << 24u);

            const uint f4_base = inp_base + w * 8u;

            for (uint i = 0u; i < 8u; i++) {
                float4 inp = input4[f4_base + i];
                float4 signs = float4(
                    select(-1.0f, 1.0f, bool(bits & 1u)),
                    select(-1.0f, 1.0f, bool(bits & 2u)),
                    select(-1.0f, 1.0f, bool(bits & 4u)),
                    select(-1.0f, 1.0f, bool(bits & 8u))
                );
                block_sum += dot(signs, inp);
                bits >>= 4u;
            }
        }
        local_sum += scale * block_sum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = residual[row] + row_sum;
    }
}
"#;

/// Q1_0_g128 GEMV V8 — branchless IEEE 754 sign-bit XOR.
///
/// Eliminates `select()` and `dot()` from the inner loop by directly
/// manipulating the sign bit of each input float via XOR.  If the
/// quantisation bit is 0 (weight = −1), we flip bit 31 of the input
/// float; if the bit is 1 (weight = +1), the input is unchanged.
///
/// Per 4-element group: 4 uint XORs + 4 float adds (no multiply, no select,
/// no dot).
///
/// Dispatch: `[ceil(n_rows / 8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V8: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v8(
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
    const uint row_byte_offset = row * blocks_per_row * 18u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint blk = row_byte_offset + b * 18u;
        const float scale = float(*(device const half*)(blocks_raw + blk));

        device const uchar* qs = blocks_raw + blk + 2u;
        uint w0 = uint(qs[0])  | (uint(qs[1])  << 8u) | (uint(qs[2])  << 16u) | (uint(qs[3])  << 24u);
        uint w1 = uint(qs[4])  | (uint(qs[5])  << 8u) | (uint(qs[6])  << 16u) | (uint(qs[7])  << 24u);
        uint w2 = uint(qs[8])  | (uint(qs[9])  << 8u) | (uint(qs[10]) << 16u) | (uint(qs[11]) << 24u);
        uint w3 = uint(qs[12]) | (uint(qs[13]) << 8u) | (uint(qs[14]) << 16u) | (uint(qs[15]) << 24u);

        const uint inp_base = b * 32u;

        // Macro: reinterpret float4 as uint4, XOR sign bits based on quant
        // bits, reinterpret back, and accumulate.
        // bit=1 → weight=+1 → keep sign   → mask=0
        // bit=0 → weight=-1 → flip sign   → mask=0x80000000
        #define APPLY_SIGNS(bits, idx) { \
            uint4 ib = as_type<uint4>(input4[inp_base + (idx)]); \
            uint4 mask = uint4( \
                (((bits) & 1u) ^ 1u) << 31u, \
                ((((bits) >> 1u) & 1u) ^ 1u) << 31u, \
                ((((bits) >> 2u) & 1u) ^ 1u) << 31u, \
                ((((bits) >> 3u) & 1u) ^ 1u) << 31u \
            ); \
            float4 sv = as_type<float4>(ib ^ mask); \
            block_sum += sv.x + sv.y + sv.z + sv.w; \
        }

        float block_sum = 0.0f;

        // Word 0: 8 float4 groups
        APPLY_SIGNS(w0, 0u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 1u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 2u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 3u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 4u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 5u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 6u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 7u);

        // Word 1
        APPLY_SIGNS(w1, 8u);  w1 >>= 4u;
        APPLY_SIGNS(w1, 9u);  w1 >>= 4u;
        APPLY_SIGNS(w1, 10u); w1 >>= 4u;
        APPLY_SIGNS(w1, 11u); w1 >>= 4u;
        APPLY_SIGNS(w1, 12u); w1 >>= 4u;
        APPLY_SIGNS(w1, 13u); w1 >>= 4u;
        APPLY_SIGNS(w1, 14u); w1 >>= 4u;
        APPLY_SIGNS(w1, 15u);

        // Word 2
        APPLY_SIGNS(w2, 16u); w2 >>= 4u;
        APPLY_SIGNS(w2, 17u); w2 >>= 4u;
        APPLY_SIGNS(w2, 18u); w2 >>= 4u;
        APPLY_SIGNS(w2, 19u); w2 >>= 4u;
        APPLY_SIGNS(w2, 20u); w2 >>= 4u;
        APPLY_SIGNS(w2, 21u); w2 >>= 4u;
        APPLY_SIGNS(w2, 22u); w2 >>= 4u;
        APPLY_SIGNS(w2, 23u);

        // Word 3
        APPLY_SIGNS(w3, 24u); w3 >>= 4u;
        APPLY_SIGNS(w3, 25u); w3 >>= 4u;
        APPLY_SIGNS(w3, 26u); w3 >>= 4u;
        APPLY_SIGNS(w3, 27u); w3 >>= 4u;
        APPLY_SIGNS(w3, 28u); w3 >>= 4u;
        APPLY_SIGNS(w3, 29u); w3 >>= 4u;
        APPLY_SIGNS(w3, 30u); w3 >>= 4u;
        APPLY_SIGNS(w3, 31u);

        #undef APPLY_SIGNS

        local_sum += scale * block_sum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

/// Fused GEMV + residual V8 — branchless IEEE 754 sign-bit XOR.
///
/// Same as V8 but final write adds residual:
/// `output[row] = residual[row] + row_sum`
///
/// Dispatch: `[ceil(n_rows / 8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V8_RESIDUAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v8_residual(
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
    const uint row_byte_offset = row * blocks_per_row * 18u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint blk = row_byte_offset + b * 18u;
        const float scale = float(*(device const half*)(blocks_raw + blk));

        device const uchar* qs = blocks_raw + blk + 2u;
        uint w0 = uint(qs[0])  | (uint(qs[1])  << 8u) | (uint(qs[2])  << 16u) | (uint(qs[3])  << 24u);
        uint w1 = uint(qs[4])  | (uint(qs[5])  << 8u) | (uint(qs[6])  << 16u) | (uint(qs[7])  << 24u);
        uint w2 = uint(qs[8])  | (uint(qs[9])  << 8u) | (uint(qs[10]) << 16u) | (uint(qs[11]) << 24u);
        uint w3 = uint(qs[12]) | (uint(qs[13]) << 8u) | (uint(qs[14]) << 16u) | (uint(qs[15]) << 24u);

        const uint inp_base = b * 32u;

        #define APPLY_SIGNS(bits, idx) { \
            uint4 ib = as_type<uint4>(input4[inp_base + (idx)]); \
            uint4 mask = uint4( \
                (((bits) & 1u) ^ 1u) << 31u, \
                ((((bits) >> 1u) & 1u) ^ 1u) << 31u, \
                ((((bits) >> 2u) & 1u) ^ 1u) << 31u, \
                ((((bits) >> 3u) & 1u) ^ 1u) << 31u \
            ); \
            float4 sv = as_type<float4>(ib ^ mask); \
            block_sum += sv.x + sv.y + sv.z + sv.w; \
        }

        float block_sum = 0.0f;

        APPLY_SIGNS(w0, 0u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 1u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 2u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 3u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 4u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 5u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 6u);  w0 >>= 4u;
        APPLY_SIGNS(w0, 7u);

        APPLY_SIGNS(w1, 8u);  w1 >>= 4u;
        APPLY_SIGNS(w1, 9u);  w1 >>= 4u;
        APPLY_SIGNS(w1, 10u); w1 >>= 4u;
        APPLY_SIGNS(w1, 11u); w1 >>= 4u;
        APPLY_SIGNS(w1, 12u); w1 >>= 4u;
        APPLY_SIGNS(w1, 13u); w1 >>= 4u;
        APPLY_SIGNS(w1, 14u); w1 >>= 4u;
        APPLY_SIGNS(w1, 15u);

        APPLY_SIGNS(w2, 16u); w2 >>= 4u;
        APPLY_SIGNS(w2, 17u); w2 >>= 4u;
        APPLY_SIGNS(w2, 18u); w2 >>= 4u;
        APPLY_SIGNS(w2, 19u); w2 >>= 4u;
        APPLY_SIGNS(w2, 20u); w2 >>= 4u;
        APPLY_SIGNS(w2, 21u); w2 >>= 4u;
        APPLY_SIGNS(w2, 22u); w2 >>= 4u;
        APPLY_SIGNS(w2, 23u);

        APPLY_SIGNS(w3, 24u); w3 >>= 4u;
        APPLY_SIGNS(w3, 25u); w3 >>= 4u;
        APPLY_SIGNS(w3, 26u); w3 >>= 4u;
        APPLY_SIGNS(w3, 27u); w3 >>= 4u;
        APPLY_SIGNS(w3, 28u); w3 >>= 4u;
        APPLY_SIGNS(w3, 29u); w3 >>= 4u;
        APPLY_SIGNS(w3, 30u); w3 >>= 4u;
        APPLY_SIGNS(w3, 31u);

        #undef APPLY_SIGNS

        local_sum += scale * block_sum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = residual[row] + row_sum;
    }
}
"#;

/// Q1_0_g128 GEMV V9 — 4 rows per simdgroup (8 lanes per row).
///
/// Instead of 1 row/SG (32 lanes/row), V9 packs 4 rows into each simdgroup
/// with 8 lanes cooperating per row.  This gives 4× more loop iterations per
/// lane, enabling better GPU memory latency hiding:
/// - k=4096:  32 blocks / 8 = 4 iterations (vs. 1 in V7)
/// - k=14336: 112 blocks / 8 = 14 iterations (excellent pipelining)
///
/// 32 rows per TG (8 simdgroups × 4 rows/sg).
/// Reduction uses simd_shuffle_xor across 8-lane groups.
///
/// Dispatch: `[ceil(n_rows / 32), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V9: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v9(
    device const uchar* blocks_raw     [[buffer(0)]],
    device const float4* input4        [[buffer(1)]],
    device float* output               [[buffer(2)]],
    constant uint& n_rows              [[buffer(3)]],
    constant uint& k                   [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    // 4 rows per simdgroup, 8 lanes per row
    const uint row_in_sg = lane >> 3u;   // 0..3
    const uint sublane = lane & 7u;      // 0..7

    // 32 rows per TG (8 simdgroups x 4 rows/sg)
    const uint row = tgid * 32u + sgid * 4u + row_in_sg;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint row_byte_offset = row * blocks_per_row * 18u;
    float local_sum = 0.0f;

    for (uint b = sublane; b < blocks_per_row; b += 8u) {
        const uint blk = row_byte_offset + b * 18u;
        const float scale = float(*(device const half*)(blocks_raw + blk));

        device const uchar* qs = blocks_raw + blk + 2u;
        uint w0 = uint(qs[0])  | (uint(qs[1])  << 8u) | (uint(qs[2])  << 16u) | (uint(qs[3])  << 24u);
        uint w1 = uint(qs[4])  | (uint(qs[5])  << 8u) | (uint(qs[6])  << 16u) | (uint(qs[7])  << 24u);
        uint w2 = uint(qs[8])  | (uint(qs[9])  << 8u) | (uint(qs[10]) << 16u) | (uint(qs[11]) << 24u);
        uint w3 = uint(qs[12]) | (uint(qs[13]) << 8u) | (uint(qs[14]) << 16u) | (uint(qs[15]) << 24u);

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

    // Reduce across 8 sublanes using XOR shuffle
    // Lane groups: 0-7 (row0), 8-15 (row1), 16-23 (row2), 24-31 (row3)
    // XOR with 1,2,4 stays within each 8-lane group
    float row_sum = local_sum;
    row_sum += simd_shuffle_xor(row_sum, 1u);
    row_sum += simd_shuffle_xor(row_sum, 2u);
    row_sum += simd_shuffle_xor(row_sum, 4u);

    if (sublane == 0u) {
        output[row] = row_sum;
    }
}
"#;

/// Fused GEMV + residual V9 — 4 rows per simdgroup (8 lanes per row).
///
/// Same as V9 but final write adds residual:
/// `output[row] = residual[row] + row_sum`
///
/// Dispatch: `[ceil(n_rows / 32), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V9_RESIDUAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v9_residual(
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
    // 4 rows per simdgroup, 8 lanes per row
    const uint row_in_sg = lane >> 3u;   // 0..3
    const uint sublane = lane & 7u;      // 0..7

    // 32 rows per TG (8 simdgroups x 4 rows/sg)
    const uint row = tgid * 32u + sgid * 4u + row_in_sg;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint row_byte_offset = row * blocks_per_row * 18u;
    float local_sum = 0.0f;

    for (uint b = sublane; b < blocks_per_row; b += 8u) {
        const uint blk = row_byte_offset + b * 18u;
        const float scale = float(*(device const half*)(blocks_raw + blk));

        device const uchar* qs = blocks_raw + blk + 2u;
        uint w0 = uint(qs[0])  | (uint(qs[1])  << 8u) | (uint(qs[2])  << 16u) | (uint(qs[3])  << 24u);
        uint w1 = uint(qs[4])  | (uint(qs[5])  << 8u) | (uint(qs[6])  << 16u) | (uint(qs[7])  << 24u);
        uint w2 = uint(qs[8])  | (uint(qs[9])  << 8u) | (uint(qs[10]) << 16u) | (uint(qs[11]) << 24u);
        uint w3 = uint(qs[12]) | (uint(qs[13]) << 8u) | (uint(qs[14]) << 16u) | (uint(qs[15]) << 24u);

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

    // Reduce across 8 sublanes using XOR shuffle
    // Lane groups: 0-7 (row0), 8-15 (row1), 16-23 (row2), 24-31 (row3)
    // XOR with 1,2,4 stays within each 8-lane group
    float row_sum = local_sum;
    row_sum += simd_shuffle_xor(row_sum, 1u);
    row_sum += simd_shuffle_xor(row_sum, 2u);
    row_sum += simd_shuffle_xor(row_sum, 4u);

    if (sublane == 0u) {
        output[row] = residual[row] + row_sum;
    }
}
"#;

/// Q1_0_g128 GEMV V10 — FP16 inner-loop accumulation.
///
/// Combines V7's fully-unrolled structure with FP16 (half precision)
/// arithmetic for all inner-loop computations.  Apple GPU's FP16 ALU
/// throughput is 2–4× FP32, yielding significant speedup on
/// compute-bound GEMV.
///
/// The kernel reads float4 input (same signature as V7) and converts
/// to half4 inline per element.  All sign vectors, dot products, and
/// block accumulators use `half`.  Only the final `simd_sum` result is
/// promoted to `float` for the output write.
///
/// Dispatch: `[ceil(n_rows / 8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V10: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v10(
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
    const uint row_byte_offset = row * blocks_per_row * 18u;
    half local_sum = 0.0h;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint blk = row_byte_offset + b * 18u;
        const half scale = *(device const half*)(blocks_raw + blk);

        device const uchar* qs = blocks_raw + blk + 2u;
        uint w0 = uint(qs[0])  | (uint(qs[1])  << 8u) | (uint(qs[2])  << 16u) | (uint(qs[3])  << 24u);
        uint w1 = uint(qs[4])  | (uint(qs[5])  << 8u) | (uint(qs[6])  << 16u) | (uint(qs[7])  << 24u);
        uint w2 = uint(qs[8])  | (uint(qs[9])  << 8u) | (uint(qs[10]) << 16u) | (uint(qs[11]) << 24u);
        uint w3 = uint(qs[12]) | (uint(qs[13]) << 8u) | (uint(qs[14]) << 16u) | (uint(qs[15]) << 24u);

        const uint inp_base = b * 32u;

        // Word 0: 8 half4 values (bits 0-31)
        half sum0 = 0.0h;
        {
            uint bits = w0;
            half4 s0 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s0, half4(input4[inp_base + 0u])); bits >>= 4u;
            half4 s1 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s1, half4(input4[inp_base + 1u])); bits >>= 4u;
            half4 s2 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s2, half4(input4[inp_base + 2u])); bits >>= 4u;
            half4 s3 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s3, half4(input4[inp_base + 3u])); bits >>= 4u;
            half4 s4 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s4, half4(input4[inp_base + 4u])); bits >>= 4u;
            half4 s5 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s5, half4(input4[inp_base + 5u])); bits >>= 4u;
            half4 s6 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s6, half4(input4[inp_base + 6u])); bits >>= 4u;
            half4 s7 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s7, half4(input4[inp_base + 7u]));
        }

        // Word 1: next 8 half4 values (bits 32-63)
        half sum1 = 0.0h;
        {
            uint bits = w1;
            half4 s0 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s0, half4(input4[inp_base + 8u])); bits >>= 4u;
            half4 s1 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s1, half4(input4[inp_base + 9u])); bits >>= 4u;
            half4 s2 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s2, half4(input4[inp_base + 10u])); bits >>= 4u;
            half4 s3 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s3, half4(input4[inp_base + 11u])); bits >>= 4u;
            half4 s4 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s4, half4(input4[inp_base + 12u])); bits >>= 4u;
            half4 s5 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s5, half4(input4[inp_base + 13u])); bits >>= 4u;
            half4 s6 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s6, half4(input4[inp_base + 14u])); bits >>= 4u;
            half4 s7 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s7, half4(input4[inp_base + 15u]));
        }

        // Word 2: next 8 half4 values (bits 64-95)
        half sum2 = 0.0h;
        {
            uint bits = w2;
            half4 s0 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s0, half4(input4[inp_base + 16u])); bits >>= 4u;
            half4 s1 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s1, half4(input4[inp_base + 17u])); bits >>= 4u;
            half4 s2 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s2, half4(input4[inp_base + 18u])); bits >>= 4u;
            half4 s3 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s3, half4(input4[inp_base + 19u])); bits >>= 4u;
            half4 s4 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s4, half4(input4[inp_base + 20u])); bits >>= 4u;
            half4 s5 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s5, half4(input4[inp_base + 21u])); bits >>= 4u;
            half4 s6 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s6, half4(input4[inp_base + 22u])); bits >>= 4u;
            half4 s7 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s7, half4(input4[inp_base + 23u]));
        }

        // Word 3: last 8 half4 values (bits 96-127)
        half sum3 = 0.0h;
        {
            uint bits = w3;
            half4 s0 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s0, half4(input4[inp_base + 24u])); bits >>= 4u;
            half4 s1 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s1, half4(input4[inp_base + 25u])); bits >>= 4u;
            half4 s2 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s2, half4(input4[inp_base + 26u])); bits >>= 4u;
            half4 s3 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s3, half4(input4[inp_base + 27u])); bits >>= 4u;
            half4 s4 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s4, half4(input4[inp_base + 28u])); bits >>= 4u;
            half4 s5 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s5, half4(input4[inp_base + 29u])); bits >>= 4u;
            half4 s6 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s6, half4(input4[inp_base + 30u])); bits >>= 4u;
            half4 s7 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s7, half4(input4[inp_base + 31u]));
        }

        local_sum += scale * (sum0 + sum1 + sum2 + sum3);
    }

    half row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = float(row_sum);
    }
}
"#;

/// Q1_0_g128 GEMV V10 + residual add — FP16 inner-loop accumulation.
///
/// Fuses the residual addition into the V10 FP16 kernel's final write:
/// `output[row] = residual[row] + float(gemv_result)`.
///
/// Dispatch: `[ceil(n_rows / 8), 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_Q1_G128_V10_RESIDUAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_q1_g128_v10_residual(
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
    const uint row_byte_offset = row * blocks_per_row * 18u;
    half local_sum = 0.0h;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint blk = row_byte_offset + b * 18u;
        const half scale = *(device const half*)(blocks_raw + blk);

        device const uchar* qs = blocks_raw + blk + 2u;
        uint w0 = uint(qs[0])  | (uint(qs[1])  << 8u) | (uint(qs[2])  << 16u) | (uint(qs[3])  << 24u);
        uint w1 = uint(qs[4])  | (uint(qs[5])  << 8u) | (uint(qs[6])  << 16u) | (uint(qs[7])  << 24u);
        uint w2 = uint(qs[8])  | (uint(qs[9])  << 8u) | (uint(qs[10]) << 16u) | (uint(qs[11]) << 24u);
        uint w3 = uint(qs[12]) | (uint(qs[13]) << 8u) | (uint(qs[14]) << 16u) | (uint(qs[15]) << 24u);

        const uint inp_base = b * 32u;

        half sum0 = 0.0h;
        {
            uint bits = w0;
            half4 s0 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s0, half4(input4[inp_base + 0u])); bits >>= 4u;
            half4 s1 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s1, half4(input4[inp_base + 1u])); bits >>= 4u;
            half4 s2 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s2, half4(input4[inp_base + 2u])); bits >>= 4u;
            half4 s3 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s3, half4(input4[inp_base + 3u])); bits >>= 4u;
            half4 s4 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s4, half4(input4[inp_base + 4u])); bits >>= 4u;
            half4 s5 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s5, half4(input4[inp_base + 5u])); bits >>= 4u;
            half4 s6 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s6, half4(input4[inp_base + 6u])); bits >>= 4u;
            half4 s7 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum0 += dot(s7, half4(input4[inp_base + 7u]));
        }

        half sum1 = 0.0h;
        {
            uint bits = w1;
            half4 s0 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s0, half4(input4[inp_base + 8u])); bits >>= 4u;
            half4 s1 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s1, half4(input4[inp_base + 9u])); bits >>= 4u;
            half4 s2 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s2, half4(input4[inp_base + 10u])); bits >>= 4u;
            half4 s3 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s3, half4(input4[inp_base + 11u])); bits >>= 4u;
            half4 s4 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s4, half4(input4[inp_base + 12u])); bits >>= 4u;
            half4 s5 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s5, half4(input4[inp_base + 13u])); bits >>= 4u;
            half4 s6 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s6, half4(input4[inp_base + 14u])); bits >>= 4u;
            half4 s7 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum1 += dot(s7, half4(input4[inp_base + 15u]));
        }

        half sum2 = 0.0h;
        {
            uint bits = w2;
            half4 s0 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s0, half4(input4[inp_base + 16u])); bits >>= 4u;
            half4 s1 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s1, half4(input4[inp_base + 17u])); bits >>= 4u;
            half4 s2 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s2, half4(input4[inp_base + 18u])); bits >>= 4u;
            half4 s3 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s3, half4(input4[inp_base + 19u])); bits >>= 4u;
            half4 s4 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s4, half4(input4[inp_base + 20u])); bits >>= 4u;
            half4 s5 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s5, half4(input4[inp_base + 21u])); bits >>= 4u;
            half4 s6 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s6, half4(input4[inp_base + 22u])); bits >>= 4u;
            half4 s7 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum2 += dot(s7, half4(input4[inp_base + 23u]));
        }

        half sum3 = 0.0h;
        {
            uint bits = w3;
            half4 s0 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s0, half4(input4[inp_base + 24u])); bits >>= 4u;
            half4 s1 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s1, half4(input4[inp_base + 25u])); bits >>= 4u;
            half4 s2 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s2, half4(input4[inp_base + 26u])); bits >>= 4u;
            half4 s3 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s3, half4(input4[inp_base + 27u])); bits >>= 4u;
            half4 s4 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s4, half4(input4[inp_base + 28u])); bits >>= 4u;
            half4 s5 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s5, half4(input4[inp_base + 29u])); bits >>= 4u;
            half4 s6 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s6, half4(input4[inp_base + 30u])); bits >>= 4u;
            half4 s7 = half4(select(-1.0h,1.0h,bool(bits&1u)), select(-1.0h,1.0h,bool(bits&2u)), select(-1.0h,1.0h,bool(bits&4u)), select(-1.0h,1.0h,bool(bits&8u)));
            sum3 += dot(s7, half4(input4[inp_base + 31u]));
        }

        local_sum += scale * (sum0 + sum1 + sum2 + sum3);
    }

    half row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = residual[row] + float(row_sum);
    }
}
"#;

/// Fused SwiGLU for CUDA (placeholder).
#[cfg(feature = "cuda")]
pub const CUDA_SWIGLU_FUSED: &str = r#"
extern "C" __global__ void swiglu_fused(
    const float* __restrict__ gate_up,
    float* __restrict__ output,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float g = gate_up[gid];
    float u = gate_up[n + gid];
    float silu_g = g / (1.0f + expf(-g));
    output[gid] = silu_g * u;
}
"#;

/// Q1_0_g128 GEMV — one thread per output row.
#[cfg(feature = "cuda")]
pub const CUDA_GEMV_Q1_G128: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void gemv_q1_g128(
    const unsigned char* __restrict__ blocks_raw,
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n_rows,
    unsigned int k)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_rows) return;

    const unsigned int blocks_per_row = k / 128u;
    const unsigned int row_byte_offset = gid * blocks_per_row * 18u;
    const float4* input4 = (const float4*)input;
    float acc = 0.0f;

    for (unsigned int b = 0u; b < blocks_per_row; b++) {
        const unsigned int blk = row_byte_offset + b * 18u;
        unsigned short d_bits = *((const unsigned short*)(blocks_raw + blk));
        float scale = __half2float(*((const __half*)&d_bits));
        const unsigned char* qs = blocks_raw + blk + 2u;

        const unsigned int inp_base = b * 32u;
        float block_sum = 0.0f;

        for (unsigned int w = 0u; w < 4u; w++) {
            unsigned int bits = (unsigned int)qs[w * 4u]
                              | ((unsigned int)qs[w * 4u + 1u] << 8u)
                              | ((unsigned int)qs[w * 4u + 2u] << 16u)
                              | ((unsigned int)qs[w * 4u + 3u] << 24u);

            const unsigned int f4_base = inp_base + w * 8u;

            for (unsigned int i = 0u; i < 8u; i++) {
                float4 inp = input4[f4_base + i];
                float s0 = (bits & 1u) ? 1.0f : -1.0f;
                float s1 = (bits & 2u) ? 1.0f : -1.0f;
                float s2 = (bits & 4u) ? 1.0f : -1.0f;
                float s3 = (bits & 8u) ? 1.0f : -1.0f;
                block_sum += s0 * inp.x + s1 * inp.y + s2 * inp.z + s3 * inp.w;
                bits >>= 4u;
            }
        }

        acc = __fmaf_rn(scale, block_sum, acc);
    }

    output[gid] = acc;
}
"#;

/// Q1_0_g128 GEMM — 2-D grid: (weight_row, batch_col).
#[cfg(feature = "cuda")]
pub const CUDA_GEMM_Q1_G128: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void gemm_q1_g128(
    const unsigned char* __restrict__ blocks_raw,
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n_rows,
    unsigned int k,
    unsigned int m)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n_rows || col >= m) return;

    const unsigned int blocks_per_row = k / 128u;
    const unsigned int row_byte_offset = row * blocks_per_row * 18u;
    const float4* col_input4 = (const float4*)(input + col * k);
    float acc = 0.0f;

    for (unsigned int b = 0u; b < blocks_per_row; b++) {
        const unsigned int blk = row_byte_offset + b * 18u;
        unsigned short d_bits = *((const unsigned short*)(blocks_raw + blk));
        float scale = __half2float(*((const __half*)&d_bits));
        const unsigned char* qs = blocks_raw + blk + 2u;

        const unsigned int inp_base = b * 32u;
        float block_sum = 0.0f;

        for (unsigned int w = 0u; w < 4u; w++) {
            unsigned int bits = (unsigned int)qs[w * 4u]
                              | ((unsigned int)qs[w * 4u + 1u] << 8u)
                              | ((unsigned int)qs[w * 4u + 2u] << 16u)
                              | ((unsigned int)qs[w * 4u + 3u] << 24u);

            const unsigned int f4_base = inp_base + w * 8u;

            for (unsigned int i = 0u; i < 8u; i++) {
                float4 inp = col_input4[f4_base + i];
                float s0 = (bits & 1u) ? 1.0f : -1.0f;
                float s1 = (bits & 2u) ? 1.0f : -1.0f;
                float s2 = (bits & 4u) ? 1.0f : -1.0f;
                float s3 = (bits & 8u) ? 1.0f : -1.0f;
                block_sum += s0 * inp.x + s1 * inp.y + s2 * inp.z + s3 * inp.w;
                bits >>= 4u;
            }
        }

        acc = __fmaf_rn(scale, block_sum, acc);
    }

    output[col * n_rows + row] = acc;
}
"#;

/// Numerically-stable softmax.
#[cfg(feature = "cuda")]
pub const CUDA_SOFTMAX: &str = r#"
extern "C" __global__ void softmax(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int size)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;

    float max_val = input[0];
    for (unsigned int i = 1u; i < size; i++) {
        max_val = fmaxf(max_val, input[i]);
    }

    float my_exp = expf(input[gid] - max_val);

    float sum_exp = 0.0f;
    for (unsigned int i = 0u; i < size; i++) {
        sum_exp += expf(input[i] - max_val);
    }

    output[gid] = (sum_exp > 0.0f) ? (my_exp / sum_exp) : (1.0f / (float)size);
}
"#;

/// Element-wise ReLU.
#[cfg(feature = "cuda")]
pub const CUDA_RELU: &str = r#"
extern "C" __global__ void relu(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    output[gid] = fmaxf(0.0f, input[gid]);
}
"#;

/// RMSNorm.
#[cfg(feature = "cuda")]
pub const CUDA_RMSNORM: &str = r#"
extern "C" __global__ void rmsnorm(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float eps,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    float sum_sq = 0.0f;
    for (unsigned int i = 0u; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = rsqrtf(sum_sq / (float)n + eps);

    output[gid] = input[gid] * rms * weight[gid];
}
"#;

/// SiLU activation.
#[cfg(feature = "cuda")]
pub const CUDA_SILU: &str = r#"
extern "C" __global__ void silu(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float x = input[gid];
    output[gid] = x / (1.0f + expf(-x));
}
"#;

/// FP32 matrix-vector multiply.
#[cfg(feature = "cuda")]
pub const CUDA_MATVEC_F32: &str = r#"
extern "C" __global__ void matvec_f32(
    const float* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ output,
    unsigned int m,
    unsigned int k)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= m) return;

    float sum = 0.0f;
    unsigned int row_offset = gid * k;
    for (unsigned int j = 0u; j < k; j++) {
        sum += a[row_offset + j] * x[j];
    }
    output[gid] = sum;
}
"#;

/// Residual add in-place.
#[cfg(feature = "cuda")]
pub const CUDA_RESIDUAL_ADD: &str = r#"
extern "C" __global__ void residual_add(
    float* __restrict__ a,
    const float* __restrict__ b,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    a[gid] += b[gid];
}
"#;

/// RMSNorm with weight (weighted variant).
#[cfg(feature = "cuda")]
pub const CUDA_RMSNORM_WEIGHTED: &str = r#"
extern "C" __global__ void rmsnorm_weighted(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    unsigned int n,
    float eps)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    float sum_sq = 0.0f;
    for (unsigned int i = 0u; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = rsqrtf(sum_sq / (float)n + eps);

    output[gid] = input[gid] * rms * weight[gid];
}
"#;
