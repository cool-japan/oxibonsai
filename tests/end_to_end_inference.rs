//! End-to-end inference pipeline tests.
//!
//! Tests the full pipeline: engine creation, generation, sampling,
//! stats tracking, batch generation, convenience functions, and builder patterns.

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_kernels::OneBitKernel;
use oxibonsai_model::ModelVariant;
use oxibonsai_runtime::builders::{ConfigBuilder, EngineBuilder, SamplerBuilder};
use oxibonsai_runtime::convenience::{
    estimate_memory_requirements, format_bytes, format_token_count,
};
use oxibonsai_runtime::engine::{EngineStats, InferenceEngine};
use oxibonsai_runtime::presets::SamplingPreset;
use oxibonsai_runtime::sampling::SamplingParams;

// ── Helper ───────────────────────────────────────────────────────────────

fn make_tiny_engine() -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42)
}

// HOTFIX-TESTMEM: kept only for `model_info_from_engine` below, which is
// `#[ignore]`d specifically because it needs this full-size engine (see that
// test for why). `model_variant_detection_8b/4b/1_7b` used to have their own
// `make_8b_engine()`/`make_engine_with_config()`-built engines too, but no
// longer construct a model at all — see the comment on
// `model_variant_detection_8b`.
fn make_8b_engine() -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::bonsai_8b(), SamplingParams::default(), 42)
}

// ── 1. Create engine with tiny_test config → generate 1 token ────────────

#[test]
fn tiny_engine_generate_single_token() {
    let mut engine = make_tiny_engine();
    // With tiny config (no real weights), forward pass uses zero embeddings
    // which should produce logits and sample a token
    let result = engine.generate(&[1], 1);
    assert!(
        result.is_ok(),
        "generation should succeed: {:?}",
        result.err()
    );
    let tokens = result.expect("generation should succeed");
    // Zero embeddings may hit EOS or produce exactly 1 token
    assert!(tokens.len() <= 1, "should produce at most 1 token");
}

// ── 2. Generate multiple tokens → verify increasing position ─────────────

#[test]
fn tiny_engine_generate_multiple_tokens() {
    let mut engine = make_tiny_engine();
    let result = engine.generate(&[1, 2, 3], 5);
    assert!(result.is_ok(), "multi-token generation should succeed");
    let tokens = result.expect("generation should succeed");
    // With zero weights, may produce 0 or more tokens (EOS may occur)
    assert!(tokens.len() <= 5, "should produce at most 5 tokens");
}

// ── 3. Sampler with greedy (temp=0) → deterministic output ───────────────

#[test]
fn greedy_sampling_deterministic() {
    let greedy_params = SamplingPreset::Greedy.params();
    let mut engine1 = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params.clone(), 42);
    let mut engine2 = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params, 42);

    let prompt = vec![1u32, 2, 3];
    let result1 = engine1
        .generate(&prompt, 3)
        .expect("engine1 should generate");
    let result2 = engine2
        .generate(&prompt, 3)
        .expect("engine2 should generate");

    assert_eq!(
        result1, result2,
        "greedy sampling with same seed should be deterministic"
    );
}

// ── 4. Sampler with different seeds → potentially different outputs ──────

#[test]
fn different_seeds_may_differ() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut engine_a = InferenceEngine::new(Qwen3Config::tiny_test(), params.clone(), 42);
    let mut engine_b = InferenceEngine::new(Qwen3Config::tiny_test(), params, 12345);

    let prompt = vec![1u32, 2, 3, 4, 5];
    let result_a = engine_a
        .generate(&prompt, 10)
        .expect("engine_a should generate");
    let result_b = engine_b
        .generate(&prompt, 10)
        .expect("engine_b should generate");

    // We can't guarantee they differ (zero weights may always pick same token),
    // but both should succeed without error
    assert!(result_a.len() <= 10);
    assert!(result_b.len() <= 10);
}

// ── 5. Engine stats tracking ─────────────────────────────────────────────

#[test]
fn engine_stats_accumulate() {
    let stats = EngineStats::new();
    assert_eq!(stats.tokens_generated(), 0);
    assert_eq!(stats.requests_completed(), 0);
    assert!((stats.avg_tokens_per_request() - 0.0).abs() < f64::EPSILON);

    stats.record_request(10);
    assert_eq!(stats.tokens_generated(), 10);
    assert_eq!(stats.requests_completed(), 1);
    assert!((stats.avg_tokens_per_request() - 10.0).abs() < f64::EPSILON);

    stats.record_request(20);
    assert_eq!(stats.tokens_generated(), 30);
    assert_eq!(stats.requests_completed(), 2);
    assert!((stats.avg_tokens_per_request() - 15.0).abs() < f64::EPSILON);

    stats.record_request(0);
    assert_eq!(stats.tokens_generated(), 30);
    assert_eq!(stats.requests_completed(), 3);
    assert!((stats.avg_tokens_per_request() - 10.0).abs() < f64::EPSILON);
}

#[test]
fn engine_stats_uptime_increases() {
    let stats = EngineStats::new();
    let u1 = stats.uptime_seconds();
    // Spin briefly
    for _ in 0..1000 {
        std::hint::black_box(0);
    }
    let u2 = stats.uptime_seconds();
    assert!(u2 >= u1, "uptime should not decrease");
}

#[test]
fn engine_stats_after_generation() {
    let mut engine = make_tiny_engine();
    let _result = engine.generate(&[1], 1);
    // After one generation, stats should reflect the request
    let stats = engine.stats();
    assert!(stats.requests_completed() >= 1);
}

// ── 6. Reset cache and regenerate → same output ──────────────────────────

#[test]
fn reset_and_regenerate_same_output() {
    let greedy_params = SamplingPreset::Greedy.params();
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params, 42);

    let prompt = vec![1u32, 2, 3];
    let first_run = engine
        .generate(&prompt, 3)
        .expect("first generation should succeed");

    engine.reset();

    // Re-create with same seed by creating a new engine (rng state reset)
    let greedy_params2 = SamplingPreset::Greedy.params();
    let mut engine2 = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params2, 42);
    let second_run = engine2
        .generate(&prompt, 3)
        .expect("second generation should succeed");

    assert_eq!(
        first_run, second_run,
        "deterministic generation after reset should match fresh engine"
    );
}

// ── 7. Batch generation with multiple prompts ────────────────────────────

#[test]
fn batch_generate_returns_correct_count() {
    let mut engine = make_tiny_engine();
    let prompts = vec![vec![1u32], vec![2, 3], vec![4, 5, 6], vec![7]];
    let results = engine.batch_generate(&prompts, 3);
    assert_eq!(
        results.len(),
        4,
        "batch_generate should return one result per prompt"
    );
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.is_ok(),
            "prompt {i} should succeed: {:?}",
            result.as_ref().err()
        );
    }
}

#[test]
fn batch_generate_empty_batch() {
    let mut engine = make_tiny_engine();
    let results = engine.batch_generate(&[], 5);
    assert!(
        results.is_empty(),
        "empty batch should return empty results"
    );
}

// ── 8. Generate with empty prompt → empty output ─────────────────────────

#[test]
fn generate_empty_prompt_returns_empty() {
    let mut engine = make_tiny_engine();
    let result = engine
        .generate(&[], 10)
        .expect("empty prompt should succeed");
    assert!(
        result.is_empty(),
        "empty prompt should produce no output tokens"
    );
}

// ── 9. Generate with max_tokens=0 → empty output ────────────────────────

#[test]
fn generate_max_tokens_zero_returns_empty_or_minimal() {
    let mut engine = make_tiny_engine();
    let result = engine
        .generate(&[1, 2, 3], 0)
        .expect("max_tokens=0 should succeed");
    assert!(
        result.is_empty(),
        "max_tokens=0 should produce no output tokens"
    );
}

// ── 10. Model variant detection ──────────────────────────────────────────
//
// HOTFIX-TESTMEM: these three used to build a real engine via
// `make_8b_engine()`/`make_engine_with_config(bonsai_4b()/bonsai_1_7b())`,
// i.e. `BonsaiModel::new(config)`, which allocates ~5 GB / ~2.7 GB / ~1.3 GB
// of token_embd + output_weight tables (plus a KV cache) just to read back
// `model.variant().name()`. `BonsaiModel::variant()` (model/types/mod.rs) is
// `ModelVariant::from_config_and_sample_tensor_type(&self.config,
// self.dominant_quant_type)`, and `ModelVariant::from_config` is a pure
// function of `(num_layers, hidden_size)` alone (model_registry.rs) — it
// never reads the model's weights. Calling it directly on the config proves
// the identical detection result with no model construction at all.
// `model_variant_detection_tiny_config_is_custom` below separately keeps the
// real `engine.dense_model()?.variant()` method chain covered end to end.

#[test]
fn model_variant_detection_8b() {
    let config = Qwen3Config::bonsai_8b();
    assert_eq!(config.num_layers, 36);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(ModelVariant::from_config(&config).name(), "Bonsai-8B");
}

#[test]
fn model_variant_detection_4b() {
    let config = Qwen3Config::bonsai_4b();
    assert_eq!(config.num_layers, 24);
    assert_eq!(ModelVariant::from_config(&config).name(), "Bonsai-4B");
}

#[test]
fn model_variant_detection_1_7b() {
    let config = Qwen3Config::bonsai_1_7b();
    // Real GGUF header value (models/Ternary-Bonsai-1.7B.gguf): 28 layers.
    assert_eq!(config.num_layers, 28);
    assert_eq!(ModelVariant::from_config(&config).name(), "Bonsai-1.7B");
}

/// Coverage twin for the three tests above: exercises the real
/// `engine.dense_model()?.variant()` method (config -> `BonsaiModel` ->
/// registry), which the direct `ModelVariant::from_config` calls above bypass
/// on purpose to avoid constructing a multi-GB model. `tiny_test()`'s (2, 64)
/// doesn't match any known architecture, so detection correctly falls back
/// to `Custom`.
#[test]
fn model_variant_detection_tiny_config_is_custom() {
    let engine = make_tiny_engine();
    let model = engine
        .dense_model()
        .expect("a config-built engine holds a dense model");
    assert_eq!(model.variant(), ModelVariant::Custom);
    assert_eq!(model.variant().name(), "Custom");
}

// ── 11. Convenience functions ────────────────────────────────────────────

#[test]
fn format_bytes_edge_cases() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(1), "1 B");
    assert_eq!(format_bytes(1023), "1023 B");
    assert_eq!(format_bytes(1024), "1.00 KB");
    assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
    assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    assert_eq!(format_bytes(1024u64 * 1024 * 1024 * 1024), "1.00 TB");
}

#[test]
fn format_token_count_ranges() {
    assert_eq!(format_token_count(0), "0 tokens");
    assert_eq!(format_token_count(1), "1 tokens");
    assert_eq!(format_token_count(999), "999 tokens");
    assert_eq!(format_token_count(1000), "1.0K tokens");
    assert_eq!(format_token_count(1_000_000), "1.0M tokens");
    assert_eq!(format_token_count(1_000_000_000), "1.0B tokens");
}

#[test]
fn estimate_memory_for_bonsai_8b() {
    let est = estimate_memory_requirements(
        1_100_000_000, // ~1.1GB model
        4096,
        8,
        128,
        36,
    );
    assert_eq!(est.model_weights_bytes, 1_100_000_000);
    assert!(est.kv_cache_bytes > 0);
    assert!(est.total_bytes > est.model_weights_bytes);
    assert!(est.fits_in_memory);
}

#[test]
fn estimate_memory_huge_model_no_fit() {
    let est = estimate_memory_requirements(
        200_000_000_000, // 200GB
        65536,
        64,
        128,
        80,
    );
    assert!(!est.fits_in_memory);
}

// ── 12. Builder patterns end-to-end ──────────────────────────────────────

#[test]
fn sampler_builder_to_engine() {
    let sampler = SamplerBuilder::new()
        .temperature(0.5)
        .top_k(20)
        .top_p(0.85)
        .repetition_penalty(1.05)
        .seed(99)
        .build()
        .expect("sampler build should succeed");
    assert!((sampler.params().temperature - 0.5).abs() < f32::EPSILON);
    assert_eq!(sampler.params().top_k, 20);
}

#[test]
fn config_builder_to_engine() {
    let model_path = std::env::temp_dir().join("test_model.gguf");
    let config = ConfigBuilder::new()
        .model_path(model_path.to_string_lossy().as_ref())
        .max_seq_len(2048)
        .port(9090)
        .temperature(0.3)
        .build()
        .expect("config build should succeed");
    assert_eq!(config.server.port, 9090);
    assert_eq!(config.model.max_seq_len, 2048);
}

#[test]
fn engine_builder_full_chain() {
    let (config, sampler) = EngineBuilder::new()
        .config(
            ConfigBuilder::new()
                .port(7777)
                .build()
                .expect("inner config should build"),
        )
        .sampler(SamplerBuilder::new().temperature(0.2).seed(123))
        .kernel_tier("reference")
        .build()
        .expect("engine builder should succeed");

    assert_eq!(config.server.port, 7777);
    assert!((sampler.params().temperature - 0.2).abs() < f32::EPSILON);
}

#[test]
fn engine_builder_defaults_produce_valid_config() {
    let (config, _sampler) = EngineBuilder::default()
        .build()
        .expect("default engine builder should succeed");
    assert!(config.validate().is_ok());
}

// ── 13. Preset integration with engine ───────────────────────────────────

#[test]
fn all_presets_create_valid_engines() {
    // T-09: used to construct an engine per preset and assert nothing
    // beyond "did not panic during construction". Actually drive a short
    // generation through each preset's engine and check the real contract
    // `generate` documents: `Ok` and at most `max_tokens` token ids.
    for preset in SamplingPreset::all() {
        let params = preset.params();
        let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), params, 42);
        let tokens = engine
            .generate(&[1, 2, 3], 4)
            .unwrap_or_else(|e| panic!("preset {preset:?}: generate failed: {e}"));
        assert!(
            tokens.len() <= 4,
            "preset {preset:?}: generate returned {} tokens, expected at most 4",
            tokens.len()
        );
    }
}

#[test]
fn greedy_preset_zero_temperature() {
    let params = SamplingPreset::Greedy.params();
    assert!(
        params.temperature < f32::EPSILON,
        "greedy preset should have zero temperature"
    );
}

#[test]
fn creative_preset_high_temperature() {
    let params = SamplingPreset::Creative.params();
    assert!(
        params.temperature >= 0.9,
        "creative preset should have high temperature"
    );
}

// ── 14. Engine kernel info ───────────────────────────────────────────────

#[test]
fn engine_kernel_available() {
    let engine = make_tiny_engine();
    let kernel = engine.kernel();
    let name = kernel.name();
    assert!(!name.is_empty(), "kernel should have a name");
}

// ── 15. Engine session tracking ──────────────────────────────────────────

#[test]
fn engine_session_initial_state() {
    let engine = make_tiny_engine();
    assert_eq!(engine.active_sessions(), 0);
    assert_eq!(engine.session_count(), 0);
}

// ── 16. Model info accessors ─────────────────────────────────────────────

#[test]
#[ignore = "constructs a full Bonsai-8B engine (~5 GB token_embd + \
            output_weight tables plus a ~1.2 GB KV cache) solely to read \
            back accessor values; run explicitly with \
            `cargo test -- --ignored` to validate real-size model info. See \
            `model_info_from_engine_tiny_config` for the default-run twin."]
fn model_info_from_engine() {
    let engine = make_8b_engine();
    let model = engine
        .dense_model()
        .expect("a config-built engine holds a dense model");
    assert_eq!(model.num_layers(), 36);
    assert_eq!(model.hidden_size(), 4096);
    assert_eq!(model.context_length(), 65536);
    assert!(model.num_parameters() > 0);
    assert!(model.model_size_bytes() > 0);
    assert!(model.kv_cache_memory_bytes() > 0);
}

/// Tiny-config twin of `model_info_from_engine` (kept `#[ignore]`d above
/// because it needs a full production-size engine to reach a non-`Custom`
/// variant). `tiny_test()`'s (2, 64) maps to `ModelVariant::Custom`, whose
/// `param_count`/`expected_model_size_bytes` are defined as exactly 0
/// (model_registry.rs), so this twin asserts the `Custom` contract — and
/// still exercises `kv_cache_memory_bytes()`, which is independent of
/// variant — on a ~78 MB fixture instead of a ~6 GB one.
#[test]
fn model_info_from_engine_tiny_config() {
    let engine = make_tiny_engine();
    let model = engine
        .dense_model()
        .expect("a config-built engine holds a dense model");
    assert_eq!(model.num_layers(), 2);
    assert_eq!(model.hidden_size(), 64);
    assert_eq!(model.context_length(), 512);
    assert_eq!(
        model.num_parameters(),
        0,
        "Custom variant reports 0 parameters"
    );
    assert_eq!(
        model.model_size_bytes(),
        0,
        "Custom variant reports 0 model size bytes"
    );
    assert!(model.kv_cache_memory_bytes() > 0);
}

#[test]
fn tiny_model_info() {
    let engine = make_tiny_engine();
    let model = engine
        .dense_model()
        .expect("a config-built engine holds a dense model");
    assert_eq!(model.num_layers(), 2);
    assert_eq!(model.hidden_size(), 64);
    assert_eq!(model.context_length(), 512);
}

// ── 17. Batch generation stats ───────────────────────────────────────────

#[test]
fn batch_generation_updates_stats() {
    let mut engine = make_tiny_engine();
    let prompts = vec![vec![1u32], vec![2u32]];
    let _results = engine.batch_generate(&prompts, 2);

    let stats = engine.stats();
    // At least 2 requests should be recorded (one per prompt)
    assert!(stats.requests_completed() >= 2);
}

// ── 18. Consecutive generations ──────────────────────────────────────────

#[test]
fn consecutive_generations_accumulate_stats() {
    let mut engine = make_tiny_engine();
    let _r1 = engine.generate(&[1], 1);
    let _r2 = engine.generate(&[2], 1);
    let _r3 = engine.generate(&[3], 1);

    let stats = engine.stats();
    assert!(
        stats.requests_completed() >= 3,
        "should have at least 3 requests"
    );
}

// ── 19. Config validation edge cases ─────────────────────────────────────

#[test]
fn config_builder_invalid_temperature() {
    let result = ConfigBuilder::new().temperature(-1.0).build();
    assert!(
        result.is_err(),
        "negative temperature should fail validation"
    );
}

#[test]
fn config_builder_invalid_top_p() {
    let result = ConfigBuilder::new().top_p(2.0).build();
    assert!(result.is_err(), "top_p > 1.0 should fail validation");
}

#[test]
fn config_builder_max_seq_len_zero() {
    let result = ConfigBuilder::new().max_seq_len(0).build();
    assert!(result.is_err(), "max_seq_len=0 should fail validation");
}

#[test]
fn sampler_builder_invalid_negative_temperature() {
    let result = SamplerBuilder::new().temperature(-0.5).build();
    assert!(result.is_err(), "negative temperature should fail");
}

#[test]
fn sampler_builder_invalid_top_p_above_one() {
    let result = SamplerBuilder::new().top_p(1.5).build();
    assert!(result.is_err(), "top_p > 1.0 should fail");
}

#[test]
fn sampler_builder_invalid_repetition_penalty_below_one() {
    let result = SamplerBuilder::new().repetition_penalty(0.5).build();
    assert!(result.is_err(), "rep_pen < 1.0 should fail");
}
