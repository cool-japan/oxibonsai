#![cfg(feature = "training")]
//! Integration tests for the model compression pipeline.

use oxibonsai_model::compression::{
    compress_model, estimate_compressed_size, CompressionConfig, CompressionError, CompressionStage,
};
use oxibonsai_model::model_merge::WeightTensor;

// ──────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────

fn make_tensor(name: &str, data: Vec<f32>, shape: Vec<usize>) -> WeightTensor {
    WeightTensor::new(name, data, shape)
}

fn linear_data(n: usize) -> Vec<f32> {
    (1..=n).map(|i| i as f32).collect()
}

fn simple_tensors() -> Vec<WeightTensor> {
    vec![
        make_tensor("layer.weight", linear_data(16), vec![4, 4]),
        make_tensor("layer.bias", linear_data(4), vec![4]),
    ]
}

// ──────────────────────────────────────────────────────────────────
// 1. compression_config_new_empty
// ──────────────────────────────────────────────────────────────────

#[test]
fn compression_config_new_empty() {
    let cfg = CompressionConfig::new();
    assert_eq!(cfg.stages.len(), 0, "new config must have 0 stages");
    assert!(!cfg.skip_embedding_layers);
}

// ──────────────────────────────────────────────────────────────────
// 2. compression_config_add_stage
// ──────────────────────────────────────────────────────────────────

#[test]
fn compression_config_add_stage() {
    let cfg = CompressionConfig::new().add_stage(CompressionStage::QuantizeInt8);
    assert_eq!(
        cfg.stages.len(),
        1,
        "config must have 1 stage after add_stage"
    );
    assert_eq!(cfg.stages[0].name(), "quantize_int8");
}

// ──────────────────────────────────────────────────────────────────
// 3. compression_config_prune_then_quantize
// ──────────────────────────────────────────────────────────────────

#[test]
fn compression_config_prune_then_quantize() {
    let cfg = CompressionConfig::prune_then_quantize(0.5);
    assert_eq!(
        cfg.stages.len(),
        2,
        "prune_then_quantize must produce 2 stages"
    );
    assert_eq!(cfg.stages[0].name(), "prune");
    assert_eq!(cfg.stages[1].name(), "quantize_int8");
}

// ──────────────────────────────────────────────────────────────────
// 4. compress_model_empty_tensors_error
// ──────────────────────────────────────────────────────────────────

#[test]
fn compress_model_empty_tensors_error() {
    let config = CompressionConfig::quantize_only();
    let result = compress_model(&[], &config);
    assert!(
        matches!(result, Err(CompressionError::EmptyModel)),
        "expected EmptyModel error, got: {result:?}",
    );
}

// ──────────────────────────────────────────────────────────────────
// 5. compress_model_empty_pipeline_error
// ──────────────────────────────────────────────────────────────────

#[test]
fn compress_model_empty_pipeline_error() {
    let tensors = simple_tensors();
    let config = CompressionConfig::new(); // no stages
    let result = compress_model(&tensors, &config);
    assert!(
        matches!(result, Err(CompressionError::EmptyPipeline)),
        "expected EmptyPipeline error, got: {result:?}",
    );
}

// ──────────────────────────────────────────────────────────────────
// 6. compress_model_prune_only_reduces_nonzero
// ──────────────────────────────────────────────────────────────────

#[test]
fn compress_model_prune_only_reduces_nonzero() {
    let tensors = vec![make_tensor("layer.weight", linear_data(20), vec![4, 5])];
    let config = CompressionConfig::prune_only(0.5);
    let result = compress_model(&tensors, &config).expect("prune_only should succeed");

    let original_nonzero = 20; // all values 1..=20 are nonzero
    let compressed_nonzero = result.total_nonzero();
    assert!(
        compressed_nonzero < original_nonzero,
        "nonzero after pruning ({compressed_nonzero}) must be fewer than original ({original_nonzero})",
    );
}

// ──────────────────────────────────────────────────────────────────
// 7. compress_model_quantize_only_runs
// ──────────────────────────────────────────────────────────────────

#[test]
fn compress_model_quantize_only_runs() {
    let tensors = simple_tensors();
    let config = CompressionConfig::quantize_only();
    let result = compress_model(&tensors, &config).expect("quantize_only should succeed");

    assert_eq!(
        result.compressed_tensors.len(),
        tensors.len(),
        "same number of tensors after quantization",
    );
    assert_eq!(result.stage_stats.len(), 1);
    assert_eq!(result.stage_stats[0].stage_name, "quantize_int8");
}

// ──────────────────────────────────────────────────────────────────
// 8. compress_model_prune_then_quantize
// ──────────────────────────────────────────────────────────────────

#[test]
fn compress_model_prune_then_quantize() {
    let tensors = simple_tensors();
    let config = CompressionConfig::prune_then_quantize(0.4);
    let result = compress_model(&tensors, &config).expect("prune_then_quantize should succeed");

    assert_eq!(
        result.compressed_tensors.len(),
        tensors.len(),
        "tensor count unchanged",
    );
    assert_eq!(
        result.stage_stats.len(),
        2,
        "must have 2 stage stats entries"
    );
    assert_eq!(result.stage_stats[0].stage_name, "prune");
    assert_eq!(result.stage_stats[1].stage_name, "quantize_int8");
}

// ──────────────────────────────────────────────────────────────────
// 9. stage_stats_compression_ratio_prune
// ──────────────────────────────────────────────────────────────────

#[test]
fn stage_stats_compression_ratio_prune() {
    // Prune keeps f32 storage → ratio = 1.0 (memory unchanged)
    // But quantize INT8 gives ratio > 1.0
    let tensors = vec![make_tensor("layer.weight", linear_data(16), vec![4, 4])];
    let config = CompressionConfig::quantize_only();
    let result = compress_model(&tensors, &config).expect("quantize ok");

    let ratio = result.stage_stats[0].compression_ratio();
    assert!(
        ratio > 1.0,
        "INT8 quantization stage compression ratio must be > 1.0, got {ratio}",
    );
}

// ──────────────────────────────────────────────────────────────────
// 10. compression_result_summary_nonempty
// ──────────────────────────────────────────────────────────────────

#[test]
fn compression_result_summary_nonempty() {
    let tensors = simple_tensors();
    let config = CompressionConfig::quantize_only();
    let result = compress_model(&tensors, &config).expect("compress ok");
    let summary = result.summary();
    assert!(!summary.is_empty(), "summary must not be empty",);
    assert!(
        summary.contains("Compression Summary"),
        "summary must contain 'Compression Summary', got: {summary}",
    );
}

// ──────────────────────────────────────────────────────────────────
// 11. compression_result_overall_sparsity
// ──────────────────────────────────────────────────────────────────

#[test]
fn compression_result_overall_sparsity() {
    // Create a tensor where exactly half the elements will be pruned
    let data: Vec<f32> = (1..=10).map(|i| i as f32).collect();
    let tensors = vec![make_tensor("w", data, vec![10])];
    let config = CompressionConfig::prune_only(0.5);
    let result = compress_model(&tensors, &config).expect("prune ok");

    let sparsity = result.overall_sparsity();
    assert!(
        sparsity > 0.0 && sparsity < 1.0,
        "sparsity must be in (0, 1), got {sparsity}",
    );
    // With sparsity=0.5 on 10 elements, expect ~50% zeros
    assert!(
        (sparsity - 0.5).abs() < 0.15,
        "expected sparsity near 0.5, got {sparsity}",
    );
}

// ──────────────────────────────────────────────────────────────────
// 12. estimate_size_prune_smaller
// ──────────────────────────────────────────────────────────────────

#[test]
fn estimate_size_prune_smaller() {
    let tensors = vec![make_tensor("layer.weight", linear_data(256), vec![16, 16])];
    let original_bytes: usize = tensors.iter().map(|t| t.data.len() * 4).sum();

    // QuantizeInt8 should give an estimate smaller than the original
    let config_q = CompressionConfig::quantize_only();
    let estimate_q = estimate_compressed_size(&tensors, &config_q);
    assert!(
        estimate_q < original_bytes,
        "INT8 estimate ({estimate_q}) must be smaller than original ({original_bytes})",
    );

    // Prune-only: storage is unchanged in dense format → estimate == original
    let config_p = CompressionConfig::prune_only(0.5);
    let estimate_p = estimate_compressed_size(&tensors, &config_p);
    assert_eq!(
        estimate_p, original_bytes,
        "prune-only estimate must equal original for dense f32 storage",
    );

    // Prune + quantize: should be smaller than original
    let config_pq = CompressionConfig::prune_then_quantize(0.5);
    let estimate_pq = estimate_compressed_size(&tensors, &config_pq);
    assert!(
        estimate_pq < original_bytes,
        "prune+quantize estimate ({estimate_pq}) must be smaller than original ({original_bytes})",
    );
}

// ──────────────────────────────────────────────────────────────────
// Bonus: skip_embedding_layers
// ──────────────────────────────────────────────────────────────────

#[test]
fn skip_embedding_layers_leaves_embed_unchanged() {
    let embed_data = linear_data(16);
    let weight_data = linear_data(16);
    let tensors = vec![
        make_tensor("embed.weight", embed_data.clone(), vec![4, 4]),
        make_tensor("layer.weight", weight_data.clone(), vec![4, 4]),
    ];
    let mut config = CompressionConfig::prune_only(0.5);
    config.skip_embedding_layers = true;

    let result = compress_model(&tensors, &config).expect("compress ok");

    // embed.weight should be unchanged
    let embed_out = &result.compressed_tensors[0];
    assert_eq!(
        embed_out.data, embed_data,
        "embed layer must not be changed"
    );

    // layer.weight should have been pruned (fewer nonzero)
    let weight_out = &result.compressed_tensors[1];
    let nonzero = weight_out.data.iter().filter(|&&x| x != 0.0).count();
    assert!(
        nonzero < 16,
        "layer.weight must be pruned, nonzero={nonzero}",
    );

    // Stage stats should reflect 1 processed + 1 skipped
    assert_eq!(result.stage_stats[0].tensors_processed, 1);
    assert_eq!(result.stage_stats[0].tensors_skipped, 1);
}

// ──────────────────────────────────────────────────────────────────
// Bonus: Clip stage
// ──────────────────────────────────────────────────────────────────

#[test]
fn clip_stage_reduces_nonzero() {
    let tensors = vec![make_tensor("layer.weight", linear_data(20), vec![4, 5])];
    let config = CompressionConfig::new().add_stage(CompressionStage::Clip { percentile: 0.3 });
    let result = compress_model(&tensors, &config).expect("clip ok");

    let compressed_nonzero = result.total_nonzero();
    assert!(
        compressed_nonzero < 20,
        "clip must reduce nonzero count, got {compressed_nonzero}",
    );
}

#[test]
fn clip_stage_invalid_percentile_returns_error() {
    let tensors = simple_tensors();
    let config = CompressionConfig::new().add_stage(CompressionStage::Clip { percentile: 0.0 });
    let result = compress_model(&tensors, &config);
    assert!(
        matches!(result, Err(CompressionError::InvalidPercentile(_))),
        "expected InvalidPercentile error, got: {result:?}",
    );
}

// ──────────────────────────────────────────────────────────────────
// Bonus: estimate_compressed_size edge cases
// ──────────────────────────────────────────────────────────────────

#[test]
fn estimate_compressed_size_empty_tensors_returns_zero() {
    let config = CompressionConfig::quantize_only();
    assert_eq!(estimate_compressed_size(&[], &config), 0);
}

#[test]
fn estimate_compressed_size_empty_pipeline_returns_zero() {
    let tensors = simple_tensors();
    let config = CompressionConfig::new();
    assert_eq!(estimate_compressed_size(&tensors, &config), 0);
}
