//! Unit tests for `export.rs` (split out per the 2000-line file-length
//! policy — gatekeeper OPTIONAL #O1, waves 3+3.5 review: this file was at
//! 1983/1990 lines before the wave-3.5 `Q2K`/`Q3K`/`Q8K` addition, with no
//! headroom left). `#[path = "export_tests.rs"] mod tests;` in `export.rs`
//! keeps this as that module's content verbatim (`use super::*;` reaches
//! everything `export.rs` defines, exactly as it did as an inline module).

use super::*;

// ── export_config_default_fp32_exceptions ─────────────────────────────

#[test]
fn test_export_config_default_fp32_exceptions() {
    let exceptions = ExportConfig::default_fp32_exceptions();
    assert!(exceptions.contains(&"token_embd.weight".to_string()));
    assert!(exceptions.contains(&"output_norm.weight".to_string()));
    assert!(exceptions.contains(&"output.weight".to_string()));
    assert_eq!(exceptions.len(), 3);
}

// ── weight_tensor_num_elements ────────────────────────────────────────

#[test]
fn test_weight_tensor_num_elements() {
    let t = WeightTensor::new("test", vec![0.0; 256], vec![16, 16]);
    assert_eq!(t.num_elements(), 256);
    assert_eq!(t.memory_bytes_f32(), 1024);
}

// ── estimate_export_size_fp32 ─────────────────────────────────────────

#[test]
fn test_estimate_export_size_fp32() {
    let tensors = vec![WeightTensor::new("w", vec![1.0; 256], vec![256, 1])];
    let config = ExportConfig::new(ExportFormat::Float32, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(size, 256 * 4);
}

// ── estimate_export_size_q1_0 ─────────────────────────────────────────

#[test]
fn test_estimate_export_size_q1_0() {
    // 256 weights → 2 groups → 2 * 18 = 36 bytes
    let tensors = vec![WeightTensor::new("w", vec![1.0; 256], vec![256, 1])];
    let config = ExportConfig::new(ExportFormat::Q1_0G128, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(
        size,
        2 * 18,
        "Q1_0 size for 256 weights should be {}",
        2 * 18
    );
}

// ── export_stats_compression_ratio ────────────────────────────────────

#[test]
fn test_export_stats_compression_ratio() {
    // 512 weights in Q1_0: 4 blocks × 18 = 72 bytes; original: 512*4 = 2048.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512, 1])];
    let config = ExportConfig::new(ExportFormat::Q1_0G128, "m");
    let stats = export_stats(&tensors, &config);
    assert!(
        stats.compression_ratio > 1.0,
        "Q1_0 should compress better than FP32"
    );
    assert_eq!(stats.quantized_tensors, 1);
    assert_eq!(stats.fp32_tensors, 0);
}

// ── export_to_gguf_basic ──────────────────────────────────────────────

#[test]
fn test_export_to_gguf_basic() {
    // 128 weights → 1 Q1_0 block (18 bytes)
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        vec![1.0; 128],
        vec![128, 1],
    )];
    let config =
        ExportConfig::new(ExportFormat::Q1_0G128, "test-model").with_description("unit test");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
    // Must start with GGUF magic: ASCII "GGUF" = bytes [0x47,0x47,0x55,0x46] → LE u32 0x46554747
    let magic = u32::from_le_bytes(bytes[0..4].try_into().expect("slice"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");
}

// ── export_fp32_tensor_unchanged ──────────────────────────────────────

#[test]
fn test_export_fp32_tensor_unchanged() {
    let data: Vec<f32> = (0..4).map(|i| i as f32).collect();
    let tensors = vec![WeightTensor::new("w", data.clone(), vec![4, 1])];
    let config = ExportConfig::new(ExportFormat::Float32, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
    // The GGUF file should contain the f32 data somewhere in its body.
    // Find the 4-byte LE encoding of 3.0f32 = 0x40400000.
    let needle = 3.0_f32.to_le_bytes();
    let found = bytes.windows(4).any(|w| w == needle.as_slice());
    assert!(found, "float 3.0 should be present in the exported bytes");
}

// ── export_skips_empty_tensors ────────────────────────────────────────

#[test]
fn test_export_skips_empty_tensors() {
    let tensors = vec![
        WeightTensor::new("good", vec![1.0; 128], vec![128, 1]),
        WeightTensor::new("empty", vec![], vec![0, 1]),
    ];
    let config = ExportConfig::new(ExportFormat::Float32, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
    // Tensor count in GGUF header (bytes 8..16 as u64) should be 1.
    let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().expect("slice"));
    assert_eq!(tensor_count, 1, "empty tensor should be skipped");
}

// ── TernaryG128 export ────────────────────────────────────────────────

#[test]
fn test_estimate_export_size_ternary_g128() {
    // 128 weights → 1 TQ2_0_g128 block → 34 bytes
    let tensors = vec![WeightTensor::new("w", vec![1.0; 128], vec![128, 1])];
    let config = ExportConfig::new(ExportFormat::TernaryG128, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(
        size, 34,
        "128-weight tensor in TernaryG128 should be 34 bytes"
    );
}

#[test]
fn test_estimate_export_size_ternary_g128_two_blocks() {
    // 256 weights → 2 TQ2_0_g128 blocks → 68 bytes
    let tensors = vec![WeightTensor::new("w", vec![1.0; 256], vec![256, 1])];
    let config = ExportConfig::new(ExportFormat::TernaryG128, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(
        size, 68,
        "256-weight tensor in TernaryG128 should be 68 bytes"
    );
}

#[test]
fn test_export_stats_ternary_g128_compression() {
    // 512 weights in TernaryG128: 4 blocks × 34 = 136 bytes; original: 512*4 = 2048.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512, 1])];
    let config = ExportConfig::new(ExportFormat::TernaryG128, "m");
    let stats = export_stats(&tensors, &config);
    assert!(
        stats.compression_ratio > 1.0,
        "TernaryG128 should compress better than FP32"
    );
    assert_eq!(stats.quantized_tensors, 1);
    assert_eq!(stats.fp32_tensors, 0);
}

#[test]
fn test_export_to_gguf_ternary_g128_basic() {
    // 128 weights → 1 TQ2_0_g128 block → valid GGUF with magic header.
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        vec![1.0; 128],
        vec![128, 1],
    )];
    let config = ExportConfig::new(ExportFormat::TernaryG128, "ternary-model");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
    let magic = u32::from_le_bytes(bytes[0..4].try_into().expect("slice"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");
}

#[test]
fn test_ternary_g128_fp32_exception_tensors_stay_fp32() {
    // output_norm.weight should stay F32 even under TernaryG128.
    let tensors = vec![
        WeightTensor::new("blk.0.attn_q.weight", vec![1.0; 128], vec![128, 1]),
        WeightTensor::new("output_norm.weight", vec![1.0; 128], vec![128, 1]),
    ];
    let config = ExportConfig::new(ExportFormat::TernaryG128, "m")
        .with_fp32_layers(vec!["output_norm.weight".to_string()]);
    let stats = export_stats(&tensors, &config);
    assert_eq!(stats.fp32_tensors, 1, "output_norm.weight should stay FP32");
    assert_eq!(
        stats.quantized_tensors, 1,
        "attn_q.weight should be ternary-quantized"
    );
}

// ── FP8E4M3 export ────────────────────────────────────────────────────────

#[test]
fn test_export_fp8_e4m3_roundtrip() {
    // 128 weights (4 FP8 blocks × 32 weights each) → 4 × 34 = 136 bytes of FP8 data.
    // The GGUF tensor data section must contain exactly that many bytes.
    let n_weights = 128usize;
    let n_blocks = n_weights / oxibonsai_core::quant_fp8::QK_FP8;
    let expected_bytes = n_blocks * oxibonsai_core::quant_fp8::BLOCK_FP8_BYTES;
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        vec![1.0; n_weights],
        vec![n_weights],
    )];
    let config = ExportConfig::new(ExportFormat::FP8E4M3, "fp8-e4m3-model");
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("FP8E4M3 export");
    // Verify GGUF magic.
    let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic slice"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");
    // The raw tensor bytes must appear somewhere in the output; their length
    // is 4 × 34 = 136 bytes. We verify the total file size is at least that.
    assert!(
        gguf_bytes.len() >= expected_bytes,
        "GGUF file too small: {} < {}",
        gguf_bytes.len(),
        expected_bytes,
    );
}

#[test]
fn test_export_fp8_e5m2_roundtrip() {
    // 64 weights (2 FP8 blocks × 32 weights each) → 2 × 34 = 68 bytes of FP8 data.
    let n_weights = 64usize;
    let n_blocks = n_weights / oxibonsai_core::quant_fp8::QK_FP8;
    let expected_bytes = n_blocks * oxibonsai_core::quant_fp8::BLOCK_FP8_BYTES;
    let tensors = vec![WeightTensor::new(
        "blk.0.ffn_gate.weight",
        vec![2.0; n_weights],
        vec![n_weights],
    )];
    let config = ExportConfig::new(ExportFormat::FP8E5M2, "fp8-e5m2-model");
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("FP8E5M2 export");
    let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic slice"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");
    assert!(
        gguf_bytes.len() >= expected_bytes,
        "GGUF file too small: {} < {}",
        gguf_bytes.len(),
        expected_bytes,
    );
}

#[test]
fn test_export_fp8_size_estimate() {
    // 32 weights → 1 FP8 block → 34 bytes.
    let tensors_32 = vec![WeightTensor::new("w", vec![1.0; 32], vec![32, 1])];
    let config_e4m3 = ExportConfig::new(ExportFormat::FP8E4M3, "m");
    let config_e5m2 = ExportConfig::new(ExportFormat::FP8E5M2, "m");
    assert_eq!(
        estimate_export_size(&tensors_32, &config_e4m3),
        34,
        "32 weights in FP8E4M3 → 1 block → 34 bytes"
    );
    assert_eq!(
        estimate_export_size(&tensors_32, &config_e5m2),
        34,
        "32 weights in FP8E5M2 → 1 block → 34 bytes"
    );

    // 256 weights → 8 blocks → 272 bytes.
    let tensors_256 = vec![WeightTensor::new("w", vec![1.0; 256], vec![256, 1])];
    assert_eq!(
        estimate_export_size(&tensors_256, &config_e4m3),
        8 * 34,
        "256 weights → 8 blocks → 272 bytes"
    );

    // Verify compression ratio > 1 (FP8 is 34/32 bytes/weight ≈ 1.0625 vs 4.0 for FP32).
    let stats = export_stats(&tensors_256, &config_e4m3);
    assert!(
        stats.compression_ratio > 1.0,
        "FP8E4M3 should compress better than FP32"
    );
    // Expected ratio: 256*4 / (8*34) = 1024 / 272 ≈ 3.76
    assert!(
        stats.compression_ratio > 3.0,
        "FP8E4M3 compression ratio should be > 3.0, got {}",
        stats.compression_ratio
    );
    assert_eq!(stats.quantized_tensors, 1);
    assert_eq!(stats.fp32_tensors, 0);
}

#[test]
fn test_fp8_fp32_exception_tensors_stay_fp32() {
    // output_norm.weight should stay F32 even under FP8E4M3 and FP8E5M2.
    let tensors = vec![
        WeightTensor::new("blk.0.attn_q.weight", vec![1.0; 64], vec![64, 1]),
        WeightTensor::new("output_norm.weight", vec![1.0; 64], vec![64, 1]),
    ];
    let config = ExportConfig::new(ExportFormat::FP8E4M3, "m")
        .with_fp32_layers(vec!["output_norm.weight".to_string()]);
    let stats = export_stats(&tensors, &config);
    assert_eq!(stats.fp32_tensors, 1, "output_norm.weight should stay FP32");
    assert_eq!(
        stats.quantized_tensors, 1,
        "attn_q.weight should be FP8-quantized"
    );
}

// ── Q4_0 export tests ─────────────────────────────────────────────────────

#[test]
fn test_export_q4_0_roundtrip() {
    // 64 floats → 2 Q4_0 blocks × 18 bytes = 36 bytes of quantized data.
    use oxibonsai_core::quant_std::{BlockQ4_0, BLOCK_Q4_0_BYTES, QK_Q4_0};
    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 8.0).collect();
    let config = ExportConfig::new(ExportFormat::Q4_0, "q4-0-model");
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        input.clone(),
        vec![n],
    )];
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q4_0 export");

    // Validate GGUF magic.
    let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

    // Validate exported byte count covers at least 2 blocks × 18 bytes.
    let expected_raw = (n / QK_Q4_0) * BLOCK_Q4_0_BYTES;
    assert!(
        gguf_bytes.len() >= expected_raw,
        "GGUF file ({} bytes) must cover at least {} raw data bytes",
        gguf_bytes.len(),
        expected_raw,
    );

    // Verify roundtrip error is acceptable (Q4_0 is 4-bit, error < 10% of max range).
    let blocks = BlockQ4_0::quantize(&input).expect("Q4_0 quantize");
    assert_eq!(blocks.len(), n / QK_Q4_0, "block count matches");
    let mut output = vec![0.0f32; n];
    BlockQ4_0::dequant(&blocks, &mut output).expect("Q4_0 dequant");
    let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let threshold = max_range * 0.15;
    assert!(
        max_err <= threshold,
        "Q4_0 roundtrip max error {max_err} > threshold {threshold} (max_range={max_range})"
    );
}

#[test]
fn test_export_q8_0_roundtrip() {
    // 64 floats → 2 Q8_0 blocks × 34 bytes = 68 bytes of quantized data.
    use oxibonsai_core::quant_std::{BlockQ8_0, BLOCK_Q8_0_BYTES, QK_Q8_0};
    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 16.0).collect();
    let config = ExportConfig::new(ExportFormat::Q8_0, "q8-0-model");
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        input.clone(),
        vec![n],
    )];
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q8_0 export");

    let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

    let expected_raw = (n / QK_Q8_0) * BLOCK_Q8_0_BYTES;
    assert!(
        gguf_bytes.len() >= expected_raw,
        "GGUF file ({} bytes) must cover at least {} raw Q8_0 data bytes",
        gguf_bytes.len(),
        expected_raw,
    );

    // Verify roundtrip error is < 1% of max range (Q8_0 is high fidelity).
    let blocks = BlockQ8_0::quantize(&input).expect("Q8_0 quantize");
    assert_eq!(blocks.len(), n / QK_Q8_0, "block count matches");
    let mut output = vec![0.0f32; n];
    BlockQ8_0::dequant(&blocks, &mut output).expect("Q8_0 dequant");
    let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let threshold = max_range * 0.01;
    assert!(
        max_err <= threshold,
        "Q8_0 roundtrip max error {max_err} > threshold {threshold} (max_range={max_range})"
    );
}

// ── Q4K export tests ──────────────────────────────────────────────────────

#[test]
fn test_export_q4k_roundtrip() {
    // 512 floats → 2 Q4_K super-blocks × 144 bytes = 288 bytes of quantized data.
    use oxibonsai_core::quant_k::{BlockQ4K, BLOCK_Q4_K_BYTES, QK_K};
    let n = 512usize;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1 - 25.6).sin()).collect();
    let config = ExportConfig::new(ExportFormat::Q4K, "q4k-model");
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        input.clone(),
        vec![n],
    )];
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q4K export");

    let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

    let expected_raw = (n / QK_K) * BLOCK_Q4_K_BYTES;
    assert!(
        gguf_bytes.len() >= expected_raw,
        "GGUF file ({} bytes) must cover at least {} raw Q4K data bytes",
        gguf_bytes.len(),
        expected_raw,
    );

    // Verify roundtrip error < 5% of max range.
    let blocks = BlockQ4K::quantize(&input).expect("Q4K quantize");
    assert_eq!(blocks.len(), n / QK_K, "Q4K block count matches");
    let mut output = vec![0.0f32; n];
    BlockQ4K::dequant(&blocks, &mut output).expect("Q4K dequant");
    let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let threshold = (max_range * 0.08).max(0.1);
    assert!(
        max_err <= threshold,
        "Q4K roundtrip max error {max_err} > threshold {threshold}"
    );
}

// ── Q5K export tests ──────────────────────────────────────────────────────

#[test]
fn test_export_q5k_roundtrip() {
    // 512 floats → 2 Q5_K super-blocks × 176 bytes = 352 bytes of quantized data.
    use oxibonsai_core::quant_k::QK_K;
    use oxibonsai_core::quant_k_ext::{BlockQ5K, BLOCK_Q5K_BYTES};
    let n = 512usize;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07 - 17.9).cos()).collect();
    let config = ExportConfig::new(ExportFormat::Q5K, "q5k-model");
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        input.clone(),
        vec![n],
    )];
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q5K export");

    let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

    let expected_raw = (n / QK_K) * BLOCK_Q5K_BYTES;
    assert!(
        gguf_bytes.len() >= expected_raw,
        "GGUF file ({} bytes) must cover at least {} raw Q5K data bytes",
        gguf_bytes.len(),
        expected_raw,
    );

    // Verify roundtrip error < 5% of max range.
    let blocks = BlockQ5K::quantize(&input).expect("Q5K quantize");
    assert_eq!(blocks.len(), n / QK_K, "Q5K block count matches");
    let mut output = vec![0.0f32; n];
    BlockQ5K::dequant(&blocks, &mut output).expect("Q5K dequant");
    let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let threshold = (max_range * 0.08).max(0.1);
    assert!(
        max_err <= threshold,
        "Q5K roundtrip max error {max_err} > threshold {threshold}"
    );
}

// ── Q6K export tests ──────────────────────────────────────────────────────

#[test]
fn test_export_q6k_roundtrip() {
    // 512 floats → 2 Q6_K super-blocks × 210 bytes = 420 bytes of quantized data.
    use oxibonsai_core::quant_k::QK_K;
    use oxibonsai_core::quant_k_ext::{BlockQ6K, BLOCK_Q6K_BYTES};
    let n = 512usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 12.8).collect();
    let config = ExportConfig::new(ExportFormat::Q6K, "q6k-model");
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        input.clone(),
        vec![n],
    )];
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q6K export");

    let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

    let expected_raw = (n / QK_K) * BLOCK_Q6K_BYTES;
    assert!(
        gguf_bytes.len() >= expected_raw,
        "GGUF file ({} bytes) must cover at least {} raw Q6K data bytes",
        gguf_bytes.len(),
        expected_raw,
    );

    // Verify roundtrip error < 3% of max range (Q6K is high fidelity).
    let blocks = BlockQ6K::quantize(&input).expect("Q6K quantize");
    assert_eq!(blocks.len(), n / QK_K, "Q6K block count matches");
    let mut output = vec![0.0f32; n];
    BlockQ6K::dequant(&blocks, &mut output).expect("Q6K dequant");
    let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let threshold = (max_range * 0.05).max(0.1);
    assert!(
        max_err <= threshold,
        "Q6K roundtrip max error {max_err} > threshold {threshold}"
    );
}

// ── Size estimation tests ─────────────────────────────────────────────────

#[test]
fn test_estimate_export_size_q4_0() {
    // 64 elements → 2 blocks × 18 bytes = 36 bytes.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 64], vec![64, 1])];
    let config = ExportConfig::new(ExportFormat::Q4_0, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(size, 2 * 18, "Q4_0: 64 weights → 2 blocks → 36 bytes");
}

#[test]
fn test_estimate_export_size_q8_0() {
    // 64 elements → 2 blocks × 34 bytes = 68 bytes.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 64], vec![64, 1])];
    let config = ExportConfig::new(ExportFormat::Q8_0, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(size, 2 * 34, "Q8_0: 64 weights → 2 blocks → 68 bytes");
}

#[test]
fn test_estimate_export_size_q4k() {
    // 512 elements → 2 super-blocks × 144 bytes = 288 bytes.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512, 1])];
    let config = ExportConfig::new(ExportFormat::Q4K, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(
        size,
        2 * 144,
        "Q4K: 512 weights → 2 super-blocks → 288 bytes"
    );
}

#[test]
fn test_estimate_export_size_q5k() {
    // 512 elements → 2 super-blocks × 176 bytes = 352 bytes.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512, 1])];
    let config = ExportConfig::new(ExportFormat::Q5K, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(
        size,
        2 * 176,
        "Q5K: 512 weights → 2 super-blocks → 352 bytes"
    );
}

#[test]
fn test_estimate_export_size_q6k() {
    // 512 elements → 2 super-blocks × 210 bytes = 420 bytes.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512, 1])];
    let config = ExportConfig::new(ExportFormat::Q6K, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(
        size,
        2 * 210,
        "Q6K: 512 weights → 2 super-blocks → 420 bytes"
    );
}

// ── GGUF type name tests ──────────────────────────────────────────────────

#[test]
fn test_export_format_type_name_q4_0() {
    // Verify that the quant_str for Q4_0 matches the expected GGUF string.
    // We check by inspecting the metadata written into the GGUF file.
    let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 64], vec![64, 1])];
    let config = ExportConfig::new(ExportFormat::Q4_0, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q4_0 export");
    // "Q4_0" string should appear somewhere in the metadata section.
    let needle = b"Q4_0";
    let found = bytes.windows(needle.len()).any(|w| w == needle);
    assert!(
        found,
        "GGUF metadata should contain \"Q4_0\" quantization string"
    );
}

#[test]
fn test_export_format_type_name_q4k() {
    // Verify that the quant_str for Q4K emits "Q4_K" in the GGUF metadata.
    let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 256], vec![256, 1])];
    let config = ExportConfig::new(ExportFormat::Q4K, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q4K export");
    let needle = b"Q4_K";
    let found = bytes.windows(needle.len()).any(|w| w == needle);
    assert!(
        found,
        "GGUF metadata should contain \"Q4_K\" quantization string"
    );
}

#[test]
fn test_export_format_type_name_q5k() {
    // Verify that Q5K emits "Q5_K" in the GGUF metadata.
    let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 256], vec![256, 1])];
    let config = ExportConfig::new(ExportFormat::Q5K, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q5K export");
    let needle = b"Q5_K";
    let found = bytes.windows(needle.len()).any(|w| w == needle);
    assert!(
        found,
        "GGUF metadata should contain \"Q5_K\" quantization string"
    );
}

#[test]
fn test_export_format_type_name_q6k() {
    // Verify that Q6K emits "Q6_K" in the GGUF metadata.
    let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 256], vec![256, 1])];
    let config = ExportConfig::new(ExportFormat::Q6K, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q6K export");
    let needle = b"Q6_K";
    let found = bytes.windows(needle.len()).any(|w| w == needle);
    assert!(
        found,
        "GGUF metadata should contain \"Q6_K\" quantization string"
    );
}

#[test]
fn test_export_format_type_name_q8_0() {
    // Verify that Q8_0 emits "Q8_0" in the GGUF metadata.
    let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 64], vec![64, 1])];
    let config = ExportConfig::new(ExportFormat::Q8_0, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q8_0 export");
    let needle = b"Q8_0";
    let found = bytes.windows(needle.len()).any(|w| w == needle);
    assert!(
        found,
        "GGUF metadata should contain \"Q8_0\" quantization string"
    );
}

// ── Compression sanity tests ──────────────────────────────────────────────

#[test]
fn test_q4_0_produces_smaller_output_than_float32() {
    // 32 elements: Q4_0 = 1 block × 18 bytes; Float32 = 32 × 4 = 128 bytes.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 32], vec![32, 1])];
    let config_q4 = ExportConfig::new(ExportFormat::Q4_0, "m");
    let config_f32 = ExportConfig::new(ExportFormat::Float32, "m");
    let q4_size = estimate_export_size(&tensors, &config_q4);
    let f32_size = estimate_export_size(&tensors, &config_f32);
    assert_eq!(q4_size, 18, "Q4_0 32 weights = 18 bytes");
    assert_eq!(f32_size, 128, "Float32 32 weights = 128 bytes");
    assert!(
        q4_size < f32_size,
        "Q4_0 ({q4_size} bytes) must be smaller than Float32 ({f32_size} bytes)"
    );
}

#[test]
fn test_q8_0_compression_vs_float32() {
    // Q8_0: 32 weights → 34 bytes (8.5 bits/weight vs 32 bits/weight).
    let tensors = vec![WeightTensor::new("w", vec![0.5f32; 32], vec![32, 1])];
    let config_q8 = ExportConfig::new(ExportFormat::Q8_0, "m");
    let config_f32 = ExportConfig::new(ExportFormat::Float32, "m");
    let q8_size = estimate_export_size(&tensors, &config_q8);
    let f32_size = estimate_export_size(&tensors, &config_f32);
    assert!(
        q8_size < f32_size,
        "Q8_0 ({q8_size} bytes) must be smaller than Float32 ({f32_size} bytes)"
    );
}

// ── Q1_0_g128 sign-convention round-trip ────────────────────────────────
//
// Regression test for a bug where the exporter used the inverse sign-bit
// convention of every reader/kernel in the workspace, silently negating
// every 1-bit weight after export→load. Uses non-uniform, mixed-sign
// data (a uniform test vector cannot expose a global sign inversion) and
// decodes the exported bytes through the SAME public types the real GGUF
// loader (`oxibonsai-model::model::weight_loaders::load_f32_tensor`) and
// the CPU/CUDA/Metal kernels use: `GgufFile::parse` +
// `oxibonsai_core::tensor::BlockQ1_0G128::weight`.
#[test]
fn test_export_q1_0_g128_sign_convention_roundtrip_via_real_loader() {
    use crate::quantize::GROUP_SIZE;
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_core::tensor::BlockQ1_0G128;

    // 128 mixed-sign, varying-magnitude weights (one full Q1_0_g128 group).
    // Deliberately avoid exact zero (sign of zero is convention-arbitrary).
    let original: Vec<f32> = (0..GROUP_SIZE).map(|i| ((i as f32) - 63.5) * 0.1).collect();
    assert_eq!(original.len(), GROUP_SIZE);

    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        original.clone(),
        vec![GROUP_SIZE, 1],
    )];
    let config = ExportConfig::new(ExportFormat::Q1_0G128, "sign-roundtrip-model");
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("export Q1_0_g128");

    // Parse the exported GGUF bytes with the SAME reader the model loader uses.
    let gguf = GgufFile::parse(&gguf_bytes).expect("parse exported GGUF");
    let data = gguf
        .tensor_data("blk.0.attn_q.weight")
        .expect("tensor_data");
    let blocks = BlockQ1_0G128::slice_from_bytes(data).expect("slice_from_bytes");
    assert_eq!(blocks.len(), 1, "128 weights should be exactly one block");

    for (i, &orig) in original.iter().enumerate() {
        let decoded = blocks[0].weight(i);
        assert_eq!(
            decoded.is_sign_positive(),
            orig.is_sign_positive(),
            "weight[{i}]: original={orig}, decoded={decoded} — sign mismatch \
                 (export used the wrong Q1_0_g128 sign-bit convention)"
        );
    }
}

#[test]
fn test_new_formats_fp32_exception_respected() {
    // output_norm.weight must stay FP32 regardless of Q4_0 / Q8_0 / K-quant format.
    // 256 elements so that the K-quant super-block (256) also divides
    // `ne0`; a shorter row would be kept in F32 by the block-alignment
    // rule and mask what this test is checking.
    let tensors = vec![
        WeightTensor::new("blk.0.attn_q.weight", vec![1.0; 256], vec![256, 1]),
        WeightTensor::new("output_norm.weight", vec![1.0; 256], vec![256, 1]),
    ];
    let fp32_exceptions = vec!["output_norm.weight".to_string()];
    for fmt in &[
        ExportFormat::Q4_0,
        ExportFormat::Q8_0,
        ExportFormat::Q4K,
        ExportFormat::Q5K,
        ExportFormat::Q6K,
        ExportFormat::Q2K,
        ExportFormat::Q3K,
        ExportFormat::Q8K,
    ] {
        let config = ExportConfig::new(*fmt, "m").with_fp32_layers(fp32_exceptions.clone());
        let stats = export_stats(&tensors, &config);
        assert_eq!(
            stats.fp32_tensors, 1,
            "output_norm.weight must stay FP32 for format {fmt:?}"
        );
        assert_eq!(
            stats.quantized_tensors, 1,
            "attn_q.weight must be quantized for format {fmt:?}"
        );
    }
}

// ── Q2K export tests (wave-3.5 deviation routing: ExportFormat wiring) ─────

#[test]
fn test_export_q2k_roundtrip() {
    // 512 floats -> 2 Q2_K super-blocks x 84 bytes = 168 bytes of quantized data.
    use oxibonsai_core::quant_k::{BlockQ2K, BLOCK_Q2_K_BYTES, QK_K};
    let n = 512usize;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.09 - 23.0).sin()).collect();
    let config = ExportConfig::new(ExportFormat::Q2K, "q2k-model");
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        input.clone(),
        vec![n],
    )];
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q2K export");

    let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

    let expected_raw = (n / QK_K) * BLOCK_Q2_K_BYTES;
    assert!(
        gguf_bytes.len() >= expected_raw,
        "GGUF file ({} bytes) must cover at least {} raw Q2K data bytes",
        gguf_bytes.len(),
        expected_raw,
    );

    // 2-bit quantization is coarse; verify the roundtrip stays within a
    // generous (but non-trivial) error bound rather than an exact match.
    let blocks = BlockQ2K::quantize(&input).expect("Q2K quantize");
    assert_eq!(blocks.len(), n / QK_K, "Q2K block count matches");
    let mut output = vec![0.0f32; n];
    BlockQ2K::dequant(&blocks, &mut output).expect("Q2K dequant");
    let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let threshold = (max_range * 0.35).max(0.1);
    assert!(
        max_err <= threshold,
        "Q2K roundtrip max error {max_err} > threshold {threshold}"
    );
}

// ── Q3K export tests ────────────────────────────────────────────────────────

#[test]
fn test_export_q3k_roundtrip() {
    // 512 floats -> 2 Q3_K super-blocks x 110 bytes = 220 bytes of quantized data.
    use oxibonsai_core::quant_k::{BlockQ3K, BLOCK_Q3K_BYTES, QK_K};
    let n = 512usize;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11 - 28.2).cos()).collect();
    let config = ExportConfig::new(ExportFormat::Q3K, "q3k-model");
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        input.clone(),
        vec![n],
    )];
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q3K export");

    let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

    let expected_raw = (n / QK_K) * BLOCK_Q3K_BYTES;
    assert!(
        gguf_bytes.len() >= expected_raw,
        "GGUF file ({} bytes) must cover at least {} raw Q3K data bytes",
        gguf_bytes.len(),
        expected_raw,
    );

    let blocks = BlockQ3K::quantize(&input).expect("Q3K quantize");
    assert_eq!(blocks.len(), n / QK_K, "Q3K block count matches");
    let mut output = vec![0.0f32; n];
    BlockQ3K::dequant(&blocks, &mut output).expect("Q3K dequant");
    let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let threshold = (max_range * 0.2).max(0.1);
    assert!(
        max_err <= threshold,
        "Q3K roundtrip max error {max_err} > threshold {threshold}"
    );
}

// ── Q8K export tests ────────────────────────────────────────────────────────

#[test]
fn test_export_q8k_roundtrip() {
    // 512 floats -> 2 Q8_K super-blocks x 292 bytes = 584 bytes of quantized data.
    use oxibonsai_core::quant_k::{BlockQ8K, BLOCK_Q8K_BYTES, QK_K};
    let n = 512usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.06 - 15.36).collect();
    let config = ExportConfig::new(ExportFormat::Q8K, "q8k-model");
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        input.clone(),
        vec![n],
    )];
    let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q8K export");

    let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
    assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

    let expected_raw = (n / QK_K) * BLOCK_Q8K_BYTES;
    assert!(
        gguf_bytes.len() >= expected_raw,
        "GGUF file ({} bytes) must cover at least {} raw Q8K data bytes",
        gguf_bytes.len(),
        expected_raw,
    );

    // 8-bit K-quant is high fidelity: verify roundtrip error < 1% of range.
    let blocks = BlockQ8K::quantize(&input).expect("Q8K quantize");
    assert_eq!(blocks.len(), n / QK_K, "Q8K block count matches");
    let mut output = vec![0.0f32; n];
    BlockQ8K::dequant(&blocks, &mut output).expect("Q8K dequant");
    let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let threshold = (max_range * 0.01).max(0.05);
    assert!(
        max_err <= threshold,
        "Q8K roundtrip max error {max_err} > threshold {threshold}"
    );
}

// ── Size estimation + GGUF type name tests for Q2K/Q3K/Q8K ─────────────────

#[test]
fn test_estimate_export_size_q2k() {
    // 512 elements -> 2 super-blocks x 84 bytes = 168 bytes.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512, 1])];
    let config = ExportConfig::new(ExportFormat::Q2K, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(
        size,
        2 * 84,
        "Q2K: 512 weights -> 2 super-blocks -> 168 bytes"
    );
}

#[test]
fn test_estimate_export_size_q3k() {
    // 512 elements -> 2 super-blocks x 110 bytes = 220 bytes.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512, 1])];
    let config = ExportConfig::new(ExportFormat::Q3K, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(
        size,
        2 * 110,
        "Q3K: 512 weights -> 2 super-blocks -> 220 bytes"
    );
}

#[test]
fn test_estimate_export_size_q8k() {
    // 512 elements -> 2 super-blocks x 292 bytes = 584 bytes.
    let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512, 1])];
    let config = ExportConfig::new(ExportFormat::Q8K, "m");
    let size = estimate_export_size(&tensors, &config);
    assert_eq!(
        size,
        2 * 292,
        "Q8K: 512 weights -> 2 super-blocks -> 584 bytes"
    );
}

#[test]
fn test_export_format_type_name_q2k() {
    let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 256], vec![256, 1])];
    let config = ExportConfig::new(ExportFormat::Q2K, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q2K export");
    let needle = b"Q2_K";
    let found = bytes.windows(needle.len()).any(|w| w == needle);
    assert!(
        found,
        "GGUF metadata should contain \"Q2_K\" quantization string"
    );
}

#[test]
fn test_export_format_type_name_q3k() {
    let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 256], vec![256, 1])];
    let config = ExportConfig::new(ExportFormat::Q3K, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q3K export");
    let needle = b"Q3_K";
    let found = bytes.windows(needle.len()).any(|w| w == needle);
    assert!(
        found,
        "GGUF metadata should contain \"Q3_K\" quantization string"
    );
}

#[test]
fn test_export_format_type_name_q8k() {
    let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 256], vec![256, 1])];
    let config = ExportConfig::new(ExportFormat::Q8K, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q8K export");
    let needle = b"Q8_K";
    let found = bytes.windows(needle.len()).any(|w| w == needle);
    assert!(
        found,
        "GGUF metadata should contain \"Q8_K\" quantization string"
    );
}
