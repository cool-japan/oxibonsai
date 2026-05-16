//! Tests for the Metal graph dispatch engine.

use metal::{Device, MTLResourceOptions};
use std::sync::Arc;

use super::buffers::{alloc_buf, download_f32, upload_f32};
use super::graph::MetalGraph;

/// Smoke test: verify MetalGraph can be created on a Metal-capable system.
#[test]
fn test_metal_graph_creation() {
    // Skip if no Metal device (e.g. CI linux).
    if Device::system_default().is_none() {
        return;
    }
    let graph = MetalGraph::new();
    assert!(graph.is_ok(), "MetalGraph::new() failed: {:?}", graph.err());
}

/// Test weight upload round-trip.
#[test]
fn test_weight_upload() {
    if Device::system_default().is_none() {
        return;
    }
    let graph = MetalGraph::new().expect("failed to create MetalGraph");
    let data = vec![0u8; 1024];
    let handle = graph.upload_weight(&data);
    assert!(handle.is_ok());
    let handle = handle.expect("upload_weight failed");
    assert_eq!(handle.byte_len(), 1024);
}

/// Test the singleton accessor.
#[test]
fn test_global_singleton() {
    if Device::system_default().is_none() {
        return;
    }
    let g1 = MetalGraph::global();
    assert!(g1.is_ok());
    let g2 = MetalGraph::global();
    assert!(g2.is_ok());
    // Both should point to the same allocation.
    let g1 = g1.expect("global failed");
    let g2 = g2.expect("global failed");
    assert!(Arc::ptr_eq(&g1, &g2));
}

/// Test residual_add via a minimal single-op dispatch.
#[test]
fn test_residual_add_single() {
    if Device::system_default().is_none() {
        return;
    }
    let graph = MetalGraph::new().expect("failed to create MetalGraph");
    let n = 256usize;
    let opts = MTLResourceOptions::StorageModeShared;

    let a_buf = alloc_buf(&graph.device, (n * 4) as u64, opts).expect("alloc a_buf");
    let b_buf = alloc_buf(&graph.device, (n * 4) as u64, opts).expect("alloc b_buf");

    // a = [1.0; 256], b = [2.0; 256]
    let a_data: Vec<f32> = vec![1.0; n];
    let b_data: Vec<f32> = vec![2.0; n];
    unsafe {
        upload_f32(&a_buf, &a_data);
        upload_f32(&b_buf, &b_data);
    }

    let cmd_buf = graph.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    graph.dispatch_residual_add(encoder, &a_buf, &b_buf, n as u32);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let mut result = vec![0.0f32; n];
    unsafe { download_f32(&a_buf, &mut result) };

    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 3.0).abs() < 1e-6,
            "residual_add mismatch at index {i}: expected 3.0, got {v}"
        );
    }
}

/// Test GEMM Q1 dispatch: batch_size=4, verifies output matches expected values.
///
/// Uses a trivial weight matrix (all-ones signs, scale=1.0h) so that
/// each output element equals the sum of the corresponding input column.
#[test]
fn test_gemm_q1_batch4() {
    if Device::system_default().is_none() {
        return;
    }
    let graph = MetalGraph::new().expect("failed to create MetalGraph");
    let opts = MTLResourceOptions::StorageModeShared;

    let n_rows: u32 = 8;
    let k: u32 = 128;
    let batch_size: u32 = 4;
    let blocks_per_row = (k / 128) as usize;

    // Build Q1_g128 weight buffer in SoA layout:
    // [scales: total_blocks × 2B][data: total_blocks × 16B]
    // All signs = 1 (bit set), scale = 1.0h = 0x3C00 LE → [0x00, 0x3C]
    let total_blocks = n_rows as usize * blocks_per_row;
    let total_weight_bytes = total_blocks * 2 + total_blocks * 16;
    let data_section = total_blocks * 2;
    let mut weight_data = vec![0u8; total_weight_bytes];
    for row in 0..n_rows as usize {
        for b in 0..blocks_per_row {
            let block_idx = row * blocks_per_row + b;
            // f16 1.0 in little-endian at scale position
            weight_data[block_idx * 2] = 0x00;
            weight_data[block_idx * 2 + 1] = 0x3C;
            // All 128 sign bits = 1 (all +1) at data position
            let d = data_section + block_idx * 16;
            for j in 0..16 {
                weight_data[d + j] = 0xFF;
            }
        }
    }

    let weight_buf =
        alloc_buf(&graph.device, total_weight_bytes as u64, opts).expect("alloc weight_buf");
    unsafe {
        std::ptr::copy_nonoverlapping(
            weight_data.as_ptr(),
            weight_buf.contents() as *mut u8,
            total_weight_bytes,
        );
    }

    // Input: 4 columns of k=128 floats (column-major: input[col * k + i])
    // col 0 = all 1.0, col 1 = all 2.0, col 2 = all 0.5, col 3 = all -1.0
    let col_values = [1.0f32, 2.0, 0.5, -1.0];
    let input_floats = batch_size as usize * k as usize;
    let mut input_data = vec![0.0f32; input_floats];
    for col in 0..batch_size as usize {
        for i in 0..k as usize {
            input_data[col * k as usize + i] = col_values[col];
        }
    }

    let input_buf =
        alloc_buf(&graph.device, (input_floats * 4) as u64, opts).expect("alloc input_buf");
    unsafe {
        upload_f32(&input_buf, &input_data);
    }

    // Output: batch_size columns of n_rows floats (column-major)
    let output_floats = batch_size as usize * n_rows as usize;
    let output_buf =
        alloc_buf(&graph.device, (output_floats * 4) as u64, opts).expect("alloc output_buf");

    // Dispatch GEMM
    let cmd_buf = graph.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    graph.dispatch_gemm_q1_v7(
        encoder,
        &weight_buf,
        &input_buf,
        &output_buf,
        n_rows,
        k,
        batch_size,
    );
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Read back and verify
    let mut result = vec![0.0f32; output_floats];
    unsafe {
        download_f32(&output_buf, &mut result);
    }

    // Expected: each row for col c = scale * (sum of 128 signs * col_value)
    // With all signs = +1 and scale = 1.0:
    //   col 0: 1.0 * 128 * 1.0 = 128.0
    //   col 1: 1.0 * 128 * 2.0 = 256.0
    //   col 2: 1.0 * 128 * 0.5 = 64.0
    //   col 3: 1.0 * 128 * -1.0 = -128.0
    let expected_col_sums = [128.0f32, 256.0, 64.0, -128.0];
    for (col, expected) in expected_col_sums.iter().enumerate() {
        for row in 0..n_rows as usize {
            let idx = col * n_rows as usize + row;
            assert!(
                (result[idx] - expected).abs() < 0.1,
                "GEMM mismatch at col={col} row={row}: expected {expected}, got {}",
                result[idx]
            );
        }
    }
}

/// Test GEMM Q1 vs independent GEMVs: batch GEMM should match individual GEMV results.
#[test]
fn test_gemm_matches_gemv() {
    if Device::system_default().is_none() {
        return;
    }
    let graph = MetalGraph::new().expect("failed to create MetalGraph");
    let opts = MTLResourceOptions::StorageModeShared;

    let n_rows: u32 = 16;
    let k: u32 = 256; // 2 blocks per row
    let batch_size: u32 = 4;
    let blocks_per_row = (k / 128) as usize;
    let total_blocks = n_rows as usize * blocks_per_row;
    let total_weight_bytes = total_blocks * 2 + total_blocks * 16;

    // Build weight in SoA layout with alternating signs: block 0 all +1, block 1 all -1
    let data_section = total_blocks * 2;
    let mut weight_data = vec![0u8; total_weight_bytes];
    for row in 0..n_rows as usize {
        for b in 0..blocks_per_row {
            let block_idx = row * blocks_per_row + b;
            weight_data[block_idx * 2] = 0x00; // f16 1.0 LE
            weight_data[block_idx * 2 + 1] = 0x3C;
            let fill = if b % 2 == 0 { 0xFF } else { 0x00 }; // +1 or -1
            let d = data_section + block_idx * 16;
            for j in 0..16 {
                weight_data[d + j] = fill;
            }
        }
    }

    let weight_buf =
        alloc_buf(&graph.device, total_weight_bytes as u64, opts).expect("alloc weight_buf");
    unsafe {
        std::ptr::copy_nonoverlapping(
            weight_data.as_ptr(),
            weight_buf.contents() as *mut u8,
            total_weight_bytes,
        );
    }

    // Input: varied per column
    let input_floats = batch_size as usize * k as usize;
    let mut input_data = vec![0.0f32; input_floats];
    for col in 0..batch_size as usize {
        for i in 0..k as usize {
            input_data[col * k as usize + i] = (col as f32 + 1.0) * 0.1;
        }
    }

    let input_buf =
        alloc_buf(&graph.device, (input_floats * 4) as u64, opts).expect("alloc input_buf");
    unsafe {
        upload_f32(&input_buf, &input_data);
    }

    // GEMM output
    let output_floats = batch_size as usize * n_rows as usize;
    let gemm_out_buf =
        alloc_buf(&graph.device, (output_floats * 4) as u64, opts).expect("alloc gemm_out_buf");

    {
        let cmd = graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        graph.dispatch_gemm_q1_v7(
            enc,
            &weight_buf,
            &input_buf,
            &gemm_out_buf,
            n_rows,
            k,
            batch_size,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let mut gemm_result = vec![0.0f32; output_floats];
    unsafe {
        download_f32(&gemm_out_buf, &mut gemm_result);
    }

    // Now run individual GEMVs and compare
    for col in 0..batch_size as usize {
        // Upload this column's input into a separate buffer
        let col_input = &input_data[col * k as usize..(col + 1) * k as usize];
        let col_in_buf =
            alloc_buf(&graph.device, (k as usize * 4) as u64, opts).expect("alloc col_in_buf");
        unsafe {
            upload_f32(&col_in_buf, col_input);
        }

        let col_out_buf = alloc_buf(&graph.device, (n_rows as usize * 4) as u64, opts)
            .expect("alloc col_out_buf");

        let cmd = graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        graph.dispatch_gemv_q1(enc, &weight_buf, &col_in_buf, &col_out_buf, n_rows, k);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut gemv_result = vec![0.0f32; n_rows as usize];
        unsafe {
            download_f32(&col_out_buf, &mut gemv_result);
        }

        for row in 0..n_rows as usize {
            let gemm_val = gemm_result[col * n_rows as usize + row];
            let gemv_val = gemv_result[row];
            assert!(
                (gemm_val - gemv_val).abs() < 1e-3,
                "GEMM/GEMV mismatch col={col} row={row}: gemm={gemm_val}, gemv={gemv_val}"
            );
        }
    }
}

/// Test batched SwiGLU dispatch.
#[test]
fn test_batched_swiglu() {
    if Device::system_default().is_none() {
        return;
    }
    let graph = MetalGraph::new().expect("failed to create MetalGraph");
    let opts = MTLResourceOptions::StorageModeShared;

    let inter: u32 = 64;
    let batch_size: u32 = 3;

    // gate_up: [batch_size × inter × 2] floats
    // For each batch b and element e:
    //   gate = gate_up[b * inter * 2 + e]
    //   up   = gate_up[b * inter * 2 + inter + e]
    let gate_up_len = batch_size as usize * inter as usize * 2;
    let mut gate_up_data = vec![0.0f32; gate_up_len];
    for b in 0..batch_size as usize {
        for e in 0..inter as usize {
            let base = b * inter as usize * 2;
            gate_up_data[base + e] = (b as f32 + 1.0) * 0.5; // gate
            gate_up_data[base + inter as usize + e] = (e as f32) * 0.1; // up
        }
    }

    let gate_up_buf =
        alloc_buf(&graph.device, (gate_up_len * 4) as u64, opts).expect("alloc gate_up_buf");
    unsafe {
        upload_f32(&gate_up_buf, &gate_up_data);
    }

    let output_len = batch_size as usize * inter as usize;
    let output_buf =
        alloc_buf(&graph.device, (output_len * 4) as u64, opts).expect("alloc output_buf");

    let cmd = graph.command_queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    graph.dispatch_batched_swiglu(enc, &gate_up_buf, &output_buf, inter, batch_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let mut result = vec![0.0f32; output_len];
    unsafe {
        download_f32(&output_buf, &mut result);
    }

    // Verify: output[b * inter + e] = silu(gate) * up
    for b in 0..batch_size as usize {
        for e in 0..inter as usize {
            let g = (b as f32 + 1.0) * 0.5;
            let u = (e as f32) * 0.1;
            let silu_g = g / (1.0 + (-g).exp());
            let expected = silu_g * u;
            let actual = result[b * inter as usize + e];
            assert!(
                (actual - expected).abs() < 1e-4,
                "batched_swiglu mismatch b={b} e={e}: expected {expected}, got {actual}"
            );
        }
    }
}

/// Test batched RMSNorm dispatch.
#[test]
fn test_batched_rmsnorm() {
    if Device::system_default().is_none() {
        return;
    }
    let graph = MetalGraph::new().expect("failed to create MetalGraph");
    let opts = MTLResourceOptions::StorageModeShared;

    let dim: u32 = 64;
    let batch_size: u32 = 3;
    let eps: f32 = 1e-5;

    // Input: batch_size vectors of dim floats
    let input_len = batch_size as usize * dim as usize;
    let mut input_data = vec![0.0f32; input_len];
    for b in 0..batch_size as usize {
        for i in 0..dim as usize {
            input_data[b * dim as usize + i] = (b as f32 + 1.0) * (i as f32 + 1.0) * 0.01;
        }
    }

    // Weight: all 1.0 (identity scaling)
    let weight_data = vec![1.0f32; dim as usize];

    let input_buf =
        alloc_buf(&graph.device, (input_len * 4) as u64, opts).expect("alloc input_buf");
    let weight_buf =
        alloc_buf(&graph.device, (dim as usize * 4) as u64, opts).expect("alloc weight_buf");
    let output_buf =
        alloc_buf(&graph.device, (input_len * 4) as u64, opts).expect("alloc output_buf");

    unsafe {
        upload_f32(&input_buf, &input_data);
        upload_f32(&weight_buf, &weight_data);
    }

    let cmd = graph.command_queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    graph.dispatch_batched_rmsnorm(
        enc,
        &input_buf,
        &weight_buf,
        &output_buf,
        eps,
        dim,
        batch_size,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let mut result = vec![0.0f32; input_len];
    unsafe {
        download_f32(&output_buf, &mut result);
    }

    // Verify: for each batch, check RMS normalization
    for b in 0..batch_size as usize {
        let offset = b * dim as usize;
        let slice = &input_data[offset..offset + dim as usize];

        // Compute expected RMS
        let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
        let rms_inv = 1.0 / (sq_sum / dim as f32 + eps).sqrt();

        for i in 0..dim as usize {
            let expected = slice[i] * rms_inv; // weight = 1.0
            let actual = result[offset + i];
            assert!(
                (actual - expected).abs() < 1e-3,
                "batched_rmsnorm mismatch b={b} i={i}: expected {expected}, got {actual}"
            );
        }
    }
}

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
