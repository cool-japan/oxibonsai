//! TQ2 GEMV parity tests for the Metal graph dispatch engine.
//!
//! Validates `encode_gemv_tq2` and the fused tail (`encode_tail_and_commit_
//! ternary`) against the CPU scalar reference. Split out of `tests.rs`.

use metal::Device;

use super::buffers::upload_f32;
use super::graph::MetalGraph;

/// Correctness: Metal TQ2 GEMV must match the scalar reference within tolerance.
///
/// Uses small shapes so the test runs fast but exercises:
/// - SoA AoS→SoA reformat
/// - MSL `gemv_tq2_g128_v1` (SIMD-group-per-row, 8 rows/threadgroup)
/// - The 2-bit ternary encoding `0b00→-1, 0b01→0, 0b10→+1, 0b11→0`
#[test]
fn test_encode_gemv_tq2_matches_reference() {
    if Device::system_default().is_none() {
        return;
    }
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;

    let graph = MetalGraph::new().expect("failed to create MetalGraph");

    let n_rows = 16usize;
    let k = 256usize; // 2 blocks per row
    let blocks_per_row = k / 128;

    // Build a deterministic set of ternary blocks covering every 2-bit code.
    let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(n_rows * blocks_per_row);
    for row in 0..n_rows {
        for bk in 0..blocks_per_row {
            let mut qs = [0u8; 32];
            for (byte_idx, b) in qs.iter_mut().enumerate() {
                let seed = row * 31 + bk * 17 + byte_idx;
                let c0 = (seed % 3) as u8;
                let c1 = ((seed / 3) % 3) as u8;
                let c2 = ((seed / 9) % 3) as u8;
                let c3 = ((seed / 27) % 3) as u8;
                *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
            }
            blocks.push(BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(0.125 + 0.03125 * row as f32),
            });
        }
    }

    let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.5).collect();

    // Reference via scalar kernel.
    let mut expected = vec![0f32; n_rows];
    crate::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut expected, n_rows, k)
        .expect("scalar reference GEMV failed");

    // Upload via Metal SoA path.
    let aos_bytes = {
        let ptr = blocks.as_ptr() as *const u8;
        let len = std::mem::size_of_val(blocks.as_slice());
        unsafe { std::slice::from_raw_parts(ptr, len) }
    };
    let handle = graph
        .upload_tq2_weight_soa(aos_bytes)
        .expect("upload_tq2_weight_soa failed");

    let mut got = vec![0f32; n_rows];
    graph
        .encode_gemv_tq2(&handle, &input, &mut got, n_rows, k)
        .expect("encode_gemv_tq2 failed");

    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "row {i}: expected {a}, got {b} (|Δ|={})",
            (a - b).abs()
        );
    }
}

/// Correctness test: `encode_tail_and_commit_ternary` must produce the same
/// greedy token ID as the CPU scalar reference (RMSNorm → TQ2 GEMV → argmax).
///
/// Synthetic geometry: hidden_size = 128, vocab_size = 256.
/// The LM head has 256 rows × 128 columns (2 TQ2_0_g128 blocks per row).
#[test]
fn test_encode_tail_ternary_matches_reference() {
    if Device::system_default().is_none() {
        return;
    }
    use crate::gpu_backend::metal_full_layer::FullLayerBuffers;
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;

    let hidden_size = 128usize;
    let vocab_size = 256usize;
    let blocks_per_row = hidden_size / 128; // = 1

    let graph = MetalGraph::new().expect("MetalGraph::new failed");

    // ── Build deterministic hidden vector ──────────────────────────────
    let hidden_vec: Vec<f32> = (0..hidden_size)
        .map(|i| (i as f32) * 0.005 - 0.32)
        .collect();

    // ── Build deterministic RMSNorm weight (all ones for simplicity) ──
    let norm_weight: Vec<f32> = vec![1.0f32; hidden_size];
    let norm_eps = 1e-5f32;

    // ── Build deterministic TQ2 LM-head weight blocks ─────────────────
    let total_blocks = vocab_size * blocks_per_row;
    let mut lm_blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(total_blocks);
    for row in 0..vocab_size {
        for bk in 0..blocks_per_row {
            let mut qs = [0u8; 32];
            for (byte_idx, byte) in qs.iter_mut().enumerate() {
                let seed = row * 37 + bk * 13 + byte_idx;
                let c0 = (seed % 3) as u8;
                let c1 = ((seed / 3) % 3) as u8;
                let c2 = ((seed / 9) % 3) as u8;
                let c3 = ((seed / 27) % 3) as u8;
                *byte = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
            }
            lm_blocks.push(BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(0.0625 + 0.015625 * (row as f32 * 0.5 + bk as f32)),
            });
        }
    }

    // ── CPU reference: RMSNorm → scalar TQ2 GEMV → argmax ─────────────
    // RMSNorm: out_i = (x_i / sqrt(mean(x^2) + eps)) * weight_i
    let sq_mean: f32 = hidden_vec.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;
    let rms_scale = 1.0 / (sq_mean + norm_eps).sqrt();
    let normed_ref: Vec<f32> = hidden_vec
        .iter()
        .zip(norm_weight.iter())
        .map(|(&x, &w)| x * rms_scale * w)
        .collect();

    let mut logits_ref = vec![0.0f32; vocab_size];
    crate::gemv_ternary::gemv_tq2_0_g128(
        &lm_blocks,
        &normed_ref,
        &mut logits_ref,
        vocab_size,
        hidden_size,
    )
    .expect("scalar gemv_tq2_0_g128 failed");

    let expected_token: u32 = logits_ref
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .expect("logits_ref is empty");

    // ── Upload norm weight and LM head to GPU ─────────────────────────
    let norm_handle = graph
        .get_or_upload_f32_weight(9_000_001u64, &norm_weight)
        .expect("upload norm_weight failed");

    let lm_aos_bytes: &[u8] = {
        let ptr = lm_blocks.as_ptr() as *const u8;
        let len = std::mem::size_of_val(lm_blocks.as_slice());
        unsafe { std::slice::from_raw_parts(ptr, len) }
    };
    let lm_handle = graph
        .get_or_upload_tq2_weight_soa(9_000_002u64, lm_aos_bytes)
        .expect("upload_tq2_weight_soa failed");

    // ── Allocate FullLayerBuffers with minimal dimensions ──────────────
    // We only need hidden_buf and normed_buf for the tail; set the other
    // dimensions to their minimum viable values.
    let bufs = FullLayerBuffers::allocate(
        &graph.device,
        hidden_size,
        hidden_size, // intermediate_size — any positive value
        1,           // nq
        1,           // nkv
        64,          // head_dim — must be even
        1,           // max_seq
    )
    .expect("FullLayerBuffers::allocate failed");

    // Upload hidden vector into hidden_buf.
    unsafe { upload_f32(&bufs.hidden_buf, &hidden_vec) };

    // ── Run the GPU ternary tail ───────────────────────────────────────
    let mut got_token: u32 = u32::MAX;
    let mut hidden_mut = hidden_vec.clone();

    let cmd_buf = graph.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();

    graph
        .encode_tail_and_commit_ternary(
            encoder,
            cmd_buf,
            &bufs,
            &mut hidden_mut,
            hidden_size,
            Some(&norm_handle),
            norm_eps,
            Some(&lm_handle),
            vocab_size,
            None,
            Some(&mut got_token),
            false,
            None,
        )
        .expect("encode_tail_and_commit_ternary failed");

    // ── Verify token IDs match bit-exactly ────────────────────────────
    assert_eq!(
        got_token, expected_token,
        "GPU greedy token {got_token} != CPU reference token {expected_token}"
    );
}
