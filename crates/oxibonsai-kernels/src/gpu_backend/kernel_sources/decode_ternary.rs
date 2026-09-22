//! TQ2_0_g128 / PQ2_0 Metal GEMV kernels (decode path).
//!
//! Both formats pack 128 weights into a 34-byte block as 32 B of LSB-first
//! 2-bit codes plus one `f16` scale, and both are uploaded in the same SoA
//! layout `[all d: N×2 B][all qs: N×32 B]` (see `metal_graph::reformat`). They
//! differ in exactly two places:
//!
//! | | `TQ2_0_g128` (ggml 42) | `PQ2_0` (ggml 142) |
//! |---|---|---|
//! | AoS block order | `qs` first, `d` last | `d` first, `qs` last |
//! | code `0b11` | `0.0` (reserved, never emitted) | `+2.0` (legal) |
//!
//! `decode_tq2` therefore keeps `0b11 → 0.0f` — that mapping is load-bearing
//! (VERIFIED.md K-01: six GPU decoders and two `oxibonsai-core` decoders pin it,
//! and the CPU-vs-Metal byte-parity guard depends on it) — while `decode_pq2` is
//! a *separate* table implementing the fork's `y = (q - 1) * d`
//! (`ggml-quants.c:501-511`, `// 00=-1, 01=0, 10=+1, 11=+2`).

/// TQ2_0_g128 GEMV V1 — SIMD-group-per-row, SoA weight layout.
///
/// SoA layout: `[all d: N×2 bytes FP16 LE][all qs: N×32 bytes]`
/// Encoding: 0b00→-1, 0b01→0, 0b10→+1, 0b11→0 (4 weights/byte, LSB-first)
/// Dispatch: 256 threads, `[ceil(n_rows/8), 1, 1]` threadgroups.
/// Buffers: "x"→SoA weights(0), "y"→input float4*(1), "result"→output(2)
/// Scalars: "n"→n_rows(3), "k"→k(4)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_TQ2_G128_V1: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float decode_tq2(uint code) {
    return select(select(0.0f, -1.0f, code == 0u), 1.0f, code == 2u);
}

inline float4 decode_byte_tq2(uint b) {
    return float4(
        decode_tq2((b     ) & 3u),
        decode_tq2((b >> 2) & 3u),
        decode_tq2((b >> 4) & 3u),
        decode_tq2((b >> 6) & 3u)
    );
}

kernel void gemv_tq2_g128_v1(
    device const uchar*  soa_raw   [[buffer(0)]],
    device const float4* input4    [[buffer(1)]],
    device       float*  output    [[buffer(2)]],
    constant uint&       n_rows    [[buffer(3)]],
    constant uint&       k         [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks   = n_rows * blocks_per_row;
    const uint qs_offset      = total_blocks * 2u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const float scale = float(*(device const half*)(soa_raw + block_idx * 2u));
        const uint qs_base = qs_offset + block_idx * 32u;
        uint w0 = (uint)(soa_raw[qs_base +  0]) | ((uint)(soa_raw[qs_base +  1]) << 8) | ((uint)(soa_raw[qs_base +  2]) << 16) | ((uint)(soa_raw[qs_base +  3]) << 24);
        uint w1 = (uint)(soa_raw[qs_base +  4]) | ((uint)(soa_raw[qs_base +  5]) << 8) | ((uint)(soa_raw[qs_base +  6]) << 16) | ((uint)(soa_raw[qs_base +  7]) << 24);
        uint w2 = (uint)(soa_raw[qs_base +  8]) | ((uint)(soa_raw[qs_base +  9]) << 8) | ((uint)(soa_raw[qs_base + 10]) << 16) | ((uint)(soa_raw[qs_base + 11]) << 24);
        uint w3 = (uint)(soa_raw[qs_base + 12]) | ((uint)(soa_raw[qs_base + 13]) << 8) | ((uint)(soa_raw[qs_base + 14]) << 16) | ((uint)(soa_raw[qs_base + 15]) << 24);
        uint w4 = (uint)(soa_raw[qs_base + 16]) | ((uint)(soa_raw[qs_base + 17]) << 8) | ((uint)(soa_raw[qs_base + 18]) << 16) | ((uint)(soa_raw[qs_base + 19]) << 24);
        uint w5 = (uint)(soa_raw[qs_base + 20]) | ((uint)(soa_raw[qs_base + 21]) << 8) | ((uint)(soa_raw[qs_base + 22]) << 16) | ((uint)(soa_raw[qs_base + 23]) << 24);
        uint w6 = (uint)(soa_raw[qs_base + 24]) | ((uint)(soa_raw[qs_base + 25]) << 8) | ((uint)(soa_raw[qs_base + 26]) << 16) | ((uint)(soa_raw[qs_base + 27]) << 24);
        uint w7 = (uint)(soa_raw[qs_base + 28]) | ((uint)(soa_raw[qs_base + 29]) << 8) | ((uint)(soa_raw[qs_base + 30]) << 16) | ((uint)(soa_raw[qs_base + 31]) << 24);
        const uint inp_base = b * 32u;
        float block_sum = 0.0f;
        { uint w = w0;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 0u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 1u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 2u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 3u]); }
        { uint w = w1;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 4u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 5u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 6u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 7u]); }
        { uint w = w2;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 8u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 9u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 10u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 11u]); }
        { uint w = w3;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 12u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 13u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 14u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 15u]); }
        { uint w = w4;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 16u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 17u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 18u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 19u]); }
        { uint w = w5;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 20u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 21u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 22u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 23u]); }
        { uint w = w6;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 24u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 25u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 26u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 27u]); }
        { uint w = w7;
          block_sum += dot(decode_byte_tq2((w     )&0xFFu), input4[inp_base + 28u]);
          block_sum += dot(decode_byte_tq2((w >> 8)&0xFFu), input4[inp_base + 29u]);
          block_sum += dot(decode_byte_tq2((w >>16)&0xFFu), input4[inp_base + 30u]);
          block_sum += dot(decode_byte_tq2((w >>24)&0xFFu), input4[inp_base + 31u]); }
        local_sum += scale * block_sum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// PQ2_0 (ggml type 142) decode table + GEMV
// ═══════════════════════════════════════════════════════════════════════════

/// `PQ2_0` 2-bit decode table as an MSL fragment: `decode_pq2` / `decode_byte_pq2`.
///
/// `decode_pq2(code) = float(code) - 1.0f`, i.e. `00→-1, 01→0, 10→+1, 11→+2`,
/// which is exactly `y = ((int)q - 1) * d` from the PrismML fork
/// (`ggml-quants.c:501-511`). This is a **different table** from `decode_tq2`,
/// which keeps the reserved code `0b11` at `0.0f` (VERIFIED.md K-01); the two
/// must never be substituted for one another.
///
/// Prepend this fragment to any kernel that needs the PQ2 table — e.g.
/// [`MSL_GEMV_PQ2_G128_V1`], which references it and does **not** define it, so
/// concatenating the two in this order compiles exactly once.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_DECODE_PQ2_FN: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float decode_pq2(uint code) {
    // 00 -> -1, 01 -> 0, 10 -> +1, 11 -> +2   (fork: y = ((int)q - 1) * d)
    return float(code) - 1.0f;
}

inline float4 decode_byte_pq2(uint b) {
    return float4(
        decode_pq2((b     ) & 3u),
        decode_pq2((b >> 2) & 3u),
        decode_pq2((b >> 4) & 3u),
        decode_pq2((b >> 6) & 3u)
    );
}
"#;

/// `PQ2_0` GEMV V1 — SIMD-group-per-row, SoA weight layout.
///
/// Structurally identical to [`MSL_GEMV_TQ2_G128_V1`] (8 rows per threadgroup,
/// 256 threads, one simdgroup per row, `[ceil(n_rows/8), 1, 1]` threadgroups)
/// and consumes the **same** SoA buffer `[all d: N×2 B FP16 LE][all qs: N×32 B]`
/// produced by `metal_graph::reformat::reformat_pq2_aos_to_soa`. The only
/// difference is the decode table: `decode_byte_pq2` (`0b11 → +2`) instead of
/// `decode_byte_tq2` (`0b11 → 0`).
///
/// Buffers: SoA weights(0), input `float4*`(1), output(2); scalars n_rows(3), k(4).
///
/// **Requires [`MSL_DECODE_PQ2_FN`] to be concatenated before it** (it supplies
/// `decode_pq2` / `decode_byte_pq2`). The per-byte inner loop has a constant trip
/// count and is fully unrolled by the MSL compiler; it is deliberately kept in
/// loop form (rather than the hand-unrolled `uint` packing of the TQ2 V1 kernel)
/// until B2-15 profiles it on the 27B.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMV_PQ2_G128_V1: &str = r#"
kernel void gemv_pq2_g128_v1(
    device const uchar*  soa_raw   [[buffer(0)]],
    device const float4* input4    [[buffer(1)]],
    device       float*  output    [[buffer(2)]],
    constant uint&       n_rows    [[buffer(3)]],
    constant uint&       k         [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8u + sgid;
    if (row >= n_rows) return;

    const uint blocks_per_row = k / 128u;
    const uint total_blocks   = n_rows * blocks_per_row;
    const uint qs_offset      = total_blocks * 2u;
    float local_sum = 0.0f;

    for (uint b = lane; b < blocks_per_row; b += 32u) {
        const uint block_idx = row * blocks_per_row + b;
        const float scale = float(*(device const half*)(soa_raw + block_idx * 2u));
        const uint qs_base  = qs_offset + block_idx * 32u;
        const uint inp_base = b * 32u;
        float block_sum = 0.0f;
        for (uint j = 0u; j < 32u; ++j) {
            block_sum += dot(decode_byte_pq2((uint)soa_raw[qs_base + j]), input4[inp_base + j]);
        }
        local_sum += scale * block_sum;
    }

    float row_sum = simd_sum(local_sum);
    if (lane == 0u) {
        output[row] = row_sum;
    }
}
"#;

/// CPU reference for the `PQ2_0` 2-bit decode table (`00→-1, 01→0, 10→+1, 11→+2`).
///
/// The scalar twin of `decode_pq2` in [`MSL_DECODE_PQ2_FN`]; the on-device test
/// in this module asserts the two agree for all four codes. Only the low two
/// bits of `code` are used.
///
/// `TQ2_0_g128` deliberately has **no** sibling here: its table (`0b11 → 0`)
/// lives in `oxibonsai-core` so the CPU and GPU decoders cannot drift (K-01).
#[cfg(all(feature = "metal", target_os = "macos"))]
#[inline]
#[must_use]
pub const fn decode_pq2_code(code: u8) -> f32 {
    (code & 0b11) as f32 - 1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests (on-device: the MSL text above is compiled and run)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

    /// Compile `source`, run `entry_point` over one threadgroup of
    /// `threads` threads with a single shared f32 output buffer of `n_out`
    /// elements, and return the buffer contents.
    fn run_probe(source: &str, entry_point: &str, threads: u64, n_out: usize) -> Vec<f32> {
        let device = Device::system_default().expect("probe requires a Metal device");
        let library = device
            .new_library_with_source(source, &CompileOptions::new())
            .expect("MSL fragment must compile");
        let function = library
            .get_function(entry_point, None)
            .expect("entry point must exist");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline creation must succeed");

        let out = device.new_buffer(
            (n_out * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let queue = device.new_command_queue();
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&out), 0);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads, 1, 1));
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out.contents() as *const f32;
        (0..n_out)
            .map(|i| unsafe { std::ptr::read(ptr.add(i)) })
            .collect()
    }

    /// `decode_pq2` on the GPU must equal the fork's `(q - 1)` table, and
    /// `decode_byte_pq2` must unpack LSB-first.
    #[test]
    fn decode_pq2_matches_the_fork_table_on_device() {
        if Device::system_default().is_none() {
            return;
        }
        const PROBE: &str = r#"
kernel void probe_decode_pq2(device float* out [[buffer(0)]],
                             uint tid [[thread_position_in_grid]])
{
    if (tid == 0u) {
        for (uint c = 0u; c < 4u; ++c) { out[c] = decode_pq2(c); }
        // 0xE4 = 0b11_10_01_00 -> lanes (LSB-first) 00, 01, 10, 11.
        float4 unpacked = decode_byte_pq2(0xE4u);
        out[4] = unpacked.x;
        out[5] = unpacked.y;
        out[6] = unpacked.z;
        out[7] = unpacked.w;
    }
}
"#;
        let source = format!("{MSL_DECODE_PQ2_FN}{PROBE}");
        let got = run_probe(&source, "probe_decode_pq2", 1, 8);
        assert_eq!(&got[0..4], &[-1.0, 0.0, 1.0, 2.0], "decode_pq2 table");
        assert_eq!(&got[4..8], &[-1.0, 0.0, 1.0, 2.0], "decode_byte_pq2 lanes");
        for code in 0u8..4 {
            assert_eq!(
                decode_pq2_code(code),
                got[code as usize],
                "CPU reference must match the GPU table for code {code}"
            );
        }
    }

    /// K-01 regression: `decode_tq2` keeps the reserved code `0b11` at `0.0`.
    ///
    /// Six GPU decoders and the two `oxibonsai-core` CPU decoders pin this
    /// mapping; changing it would silently alter every legacy TQ2_0_g128 file
    /// and break the CPU-vs-Metal byte-parity guard.
    #[test]
    fn decode_tq2_keeps_the_reserved_code_at_zero_on_device() {
        if Device::system_default().is_none() {
            return;
        }
        const PROBE: &str = r#"
kernel void probe_decode_tq2(device float* out [[buffer(0)]],
                             uint tid [[thread_position_in_grid]])
{
    if (tid == 0u) {
        for (uint c = 0u; c < 4u; ++c) { out[c] = decode_tq2(c); }
    }
}
"#;
        let source = format!("{MSL_GEMV_TQ2_G128_V1}{PROBE}");
        let got = run_probe(&source, "probe_decode_tq2", 1, 4);
        assert_eq!(&got[..], &[-1.0, 0.0, 1.0, 0.0], "decode_tq2 table (K-01)");
    }

    /// End-to-end: `gemv_pq2_g128_v1` over a hand-built SoA buffer must match a
    /// scalar CPU reference, including blocks that use the `0b11 → +2` code.
    #[test]
    fn gemv_pq2_matches_the_scalar_reference_on_device() {
        if Device::system_default().is_none() {
            return;
        }
        use half::f16;

        let n_rows = 5usize;
        let k = 256usize; // 2 blocks per row
        let blocks_per_row = k / 128;
        let n_blocks = n_rows * blocks_per_row;

        // Deterministic codes covering all four values, including 0b11.
        let mut qs = vec![0u8; n_blocks * 32];
        for (i, byte) in qs.iter_mut().enumerate() {
            let c = |s: usize| ((i * 7 + s * 13) % 4) as u8;
            *byte = c(0) | (c(1) << 2) | (c(2) << 4) | (c(3) << 6);
        }
        let scales: Vec<f16> = (0..n_blocks)
            .map(|b| f16::from_f32(0.0625 + 0.015_625 * b as f32))
            .collect();
        let input: Vec<f32> = (0..k).map(|i| i as f32 * 0.013 - 0.7).collect();

        // SoA: [all scales][all qs] — exactly what reformat_pq2_aos_to_soa emits.
        let mut soa = Vec::with_capacity(n_blocks * 34);
        for s in &scales {
            soa.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        soa.extend_from_slice(&qs);

        // Scalar reference.
        let mut expected = vec![0f32; n_rows];
        for (row, out) in expected.iter_mut().enumerate() {
            let mut acc = 0f32;
            for b in 0..blocks_per_row {
                let block_idx = row * blocks_per_row + b;
                let mut block_sum = 0f32;
                for j in 0..32 {
                    let byte = qs[block_idx * 32 + j];
                    for lane in 0..4 {
                        let code = (byte >> (lane * 2)) & 0b11;
                        block_sum += decode_pq2_code(code) * input[b * 128 + j * 4 + lane];
                    }
                }
                acc += scales[block_idx].to_f32() * block_sum;
            }
            *out = acc;
        }

        // GPU run.
        let device = Device::system_default().expect("device checked above");
        let library = device
            .new_library_with_source(
                &format!("{MSL_DECODE_PQ2_FN}{MSL_GEMV_PQ2_G128_V1}"),
                &CompileOptions::new(),
            )
            .expect("PQ2 GEMV must compile");
        let function = library
            .get_function("gemv_pq2_g128_v1", None)
            .expect("gemv_pq2_g128_v1 entry point");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline creation");

        let shared = MTLResourceOptions::StorageModeShared;
        let w_buf = device.new_buffer_with_data(
            soa.as_ptr() as *const std::ffi::c_void,
            soa.len() as u64,
            shared,
        );
        let in_buf = device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * 4) as u64,
            shared,
        );
        let out_buf = device.new_buffer((n_rows * 4) as u64, shared);

        let queue = device.new_command_queue();
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&w_buf), 0);
        encoder.set_buffer(1, Some(&in_buf), 0);
        encoder.set_buffer(2, Some(&out_buf), 0);
        let n_rows_u32 = n_rows as u32;
        let k_u32 = k as u32;
        encoder.set_bytes(3, 4, &n_rows_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        let tg = n_rows.div_ceil(8) as u64;
        encoder.dispatch_thread_groups(MTLSize::new(tg, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        for (row, want) in expected.iter().enumerate() {
            let got = unsafe { std::ptr::read(ptr.add(row)) };
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "row {row}: GPU {got} vs CPU {want}"
            );
        }
    }
}
