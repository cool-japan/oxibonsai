//! Cross-format quantization comparison tests.
//!
//! Compares Q1\_0\_g128 (1-bit) and INT8 quantization across several dimensions:
//! accuracy (MSE, SNR), compression ratio, and round-trip export fidelity.

use oxibonsai_model::export::{
    export_stats, export_to_gguf, ExportConfig, ExportError, ExportFormat, WeightTensor,
};
use oxibonsai_model::quantize::{analyze_quantization_error, quantize_q1_0_g128, GROUP_SIZE};
use oxibonsai_model::quantize_int8::{
    analyze_int8_error, compare_quantization_methods, quantize_per_channel, quantize_per_tensor,
    Int8Mode,
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Build a smooth ramp tensor with `n` elements centered at zero.
fn ramp_weights(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 - (n / 2) as f32) * (1.0 / n as f32))
        .collect()
}

/// Build a tensor padded to a multiple of `GROUP_SIZE`.
fn ramp_aligned(n: usize) -> Vec<f32> {
    let padded = n.div_ceil(GROUP_SIZE) * GROUP_SIZE;
    ramp_weights(padded)
}

// ─── Q1_0 vs INT8 error comparison ───────────────────────────────────────────

#[test]
fn test_q1_0_vs_int8_error_comparison() {
    let weights = ramp_aligned(512);
    let num_channels = 4;

    let cmp = compare_quantization_methods(&weights, Some(num_channels))
        .expect("compare_quantization_methods");

    // INT8 per-tensor should be considerably more accurate than 1-bit Q1_0.
    assert!(
        cmp.int8_per_tensor.mse < cmp.q1_0.mse,
        "INT8 per-tensor MSE ({}) should be < Q1_0 MSE ({})",
        cmp.int8_per_tensor.mse,
        cmp.q1_0.mse
    );

    // INT8 per-channel should be at least as good as per-tensor.
    let int8_pc = cmp.int8_per_channel.expect("per-channel result present");
    assert!(
        int8_pc.mse <= cmp.int8_per_tensor.mse + 1e-9,
        "INT8 per-channel MSE ({}) should be ≤ per-tensor MSE ({})",
        int8_pc.mse,
        cmp.int8_per_tensor.mse
    );

    // Q1_0 bits-per-weight should be ≈ 1.125.
    assert!(
        (cmp.q1_0.bits_per_weight - 1.125).abs() < 1e-5,
        "Q1_0 bits/weight should be 1.125, got {}",
        cmp.q1_0.bits_per_weight
    );

    // INT8 bits-per-weight should be 8.0.
    assert_eq!(cmp.int8_per_tensor.bits_per_weight, 8.0);
}

// ─── Q1_0 SNR threshold ───────────────────────────────────────────────────────

#[test]
fn test_quantization_snr_q1_0_above_threshold() {
    // Smooth ramp → Q1_0 degrades significantly, but SNR should still be > 0 dB.
    let weights = ramp_aligned(1024);
    let quantized = quantize_q1_0_g128(&weights).expect("quantize_q1_0_g128");
    let err = analyze_quantization_error(&weights, &quantized).expect("analyze");

    // For a smooth ramp the SNR will be low (1-bit), but must be finite and > 0.
    assert!(
        err.snr_db.is_finite(),
        "Q1_0 SNR should be finite, got {}",
        err.snr_db
    );
    assert!(
        err.snr_db > 0.0,
        "Q1_0 SNR should be positive, got {} dB",
        err.snr_db
    );
}

// ─── INT8 SNR threshold ───────────────────────────────────────────────────────

#[test]
fn test_quantization_snr_int8_above_threshold() {
    // INT8 on a smooth ramp should achieve SNR well above 40 dB.
    let weights = ramp_aligned(1024);
    let q = quantize_per_tensor(&weights);
    let err = analyze_int8_error(&weights, &q);

    assert!(
        err.snr_db > 40.0,
        "INT8 per-tensor SNR should exceed 40 dB for a smooth ramp, got {} dB",
        err.snr_db
    );
}

// ─── Q1_0 compression ratio ───────────────────────────────────────────────────

#[test]
fn test_compression_ratio_q1_0() {
    // 1024 weights → 8 blocks × 18 bytes = 144 bytes; f32 = 4096 bytes.
    // ratio ≈ 28.4
    let tensors = vec![WeightTensor::new("w", vec![1.0; 1024], vec![1024])];
    let config = ExportConfig::new(ExportFormat::Q1_0G128, "m");
    let stats = export_stats(&tensors, &config);
    let expected_ratio = (1024.0_f32 * 4.0) / (8.0 * 18.0);
    assert!(
        (stats.compression_ratio - expected_ratio).abs() < 0.5,
        "Q1_0 compression ratio should be ~{expected_ratio:.1}, got {:.2}",
        stats.compression_ratio
    );
}

// ─── INT8 compression ratio ───────────────────────────────────────────────────

#[test]
fn test_compression_ratio_int8() {
    // 1024 weights, 8 channels:
    //   quantized bytes = 1024 (i8) + 8*4 (scales) = 1056
    //   original bytes  = 1024*4 = 4096
    //   ratio ≈ 3.88
    let tensors = vec![WeightTensor::new("w", vec![1.0; 1024], vec![8, 128])];
    let config = ExportConfig::new(ExportFormat::Int8PerChannel, "m");
    let stats = export_stats(&tensors, &config);
    // estimate_export_size for INT8: data.len() + num_channels * 4
    let expected_exported = 1024 + 8 * 4; // 1056
    assert_eq!(stats.exported_bytes, expected_exported);
    let expected_ratio = 4096.0_f32 / 1056.0_f32;
    assert!(
        (stats.compression_ratio - expected_ratio).abs() < 0.05,
        "INT8 compression ratio should be ~{expected_ratio:.3}, got {:.3}",
        stats.compression_ratio
    );
}

// ─── Export and reimport tensor ───────────────────────────────────────────────

#[test]
fn test_export_and_reimport_tensor() {
    // Export a simple FP32 tensor to GGUF, then verify the magic and tensor
    // count in the resulting bytes (full parse is handled by oxibonsai-core).
    let data: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        data.clone(),
        vec![128],
    )];
    let config = ExportConfig::new(ExportFormat::Float32, "reimport-test")
        .with_description("roundtrip test");

    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export_to_gguf");

    // Verify GGUF header magic: ASCII "GGUF" = bytes [0x47,0x47,0x55,0x46] → LE u32 0x46554747.
    let magic = u32::from_le_bytes(bytes[0..4].try_into().expect("slice"));
    assert_eq!(
        magic, 0x4655_4747,
        "exported bytes should start with GGUF magic"
    );

    // Version = 3 (LE u32 at offset 4).
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("slice"));
    assert_eq!(version, 3, "GGUF version should be 3");

    // tensor_count = 1 (LE u64 at offset 8).
    let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().expect("slice"));
    assert_eq!(tensor_count, 1, "should have exactly 1 tensor");

    // The raw f32 value 0.0 should appear somewhere in the data section.
    let needle = 0.0_f32.to_le_bytes();
    let found = bytes.windows(4).any(|w| w == needle.as_slice());
    assert!(found, "f32 value 0.0 should be present in the GGUF payload");

    // INT8 per-channel has no GGUF tensor-type id and no loader anywhere in
    // this workspace, so `export_to_gguf` refuses it (see
    // `ExportFormat::Int8PerChannel` docs) rather than silently writing
    // packed bytes mislabeled as `TensorType::F32`.
    let data_q: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let tensors_q = vec![WeightTensor::new(
        "blk.0.attn_k.weight",
        data_q,
        vec![4, 64],
    )];
    let config_q = ExportConfig::new(ExportFormat::Int8PerChannel, "reimport-int8");
    let result_q = export_to_gguf(&tensors_q, &config_q, &[]);
    assert!(
        matches!(result_q, Err(ExportError::NoLoaderForFormat { .. })),
        "Int8PerChannel export should be refused (no matching GGUF loader), got {result_q:?}"
    );

    // Verify Q1_0 path.
    let data_q1: Vec<f32> = vec![1.0_f32; 128];
    let tensors_q1 = vec![WeightTensor::new("w", data_q1, vec![128])];
    let config_q1 = ExportConfig::new(ExportFormat::Q1_0G128, "reimport-q1");
    let bytes_q1 = export_to_gguf(&tensors_q1, &config_q1, &[]).expect("export q1_0");
    assert_eq!(
        u32::from_le_bytes(bytes_q1[0..4].try_into().expect("s")),
        0x4655_4747
    );
}

// ─── Per-channel mode field ───────────────────────────────────────────────────

#[test]
fn test_int8_per_channel_mode_field() {
    let weights: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let q = quantize_per_channel(&weights, 4).expect("per-channel");
    assert_eq!(q.mode, Int8Mode::PerChannel { num_channels: 4 });
    assert_eq!(q.scales.len(), 4);
}
