//! Builder and preset tests for the runtime crate.
//!
//! Tests SamplerBuilder, ConfigBuilder, EngineBuilder, and SamplingPresets
//! for correct validation, defaults, and integration.

use oxibonsai_runtime::builders::{ConfigBuilder, EngineBuilder, SamplerBuilder};
use oxibonsai_runtime::presets::SamplingPreset;
use oxibonsai_runtime::sampling::SamplingParams;

// ── SamplerBuilder ───────────────────────────────────────────────────────

#[test]
fn sampler_builder_all_defaults_valid() {
    let sampler = SamplerBuilder::new()
        .build()
        .expect("default sampler should be valid");
    let p = sampler.params();
    assert!((p.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(p.top_k, 40);
    assert!((p.top_p - 0.9).abs() < f32::EPSILON);
    // Gatekeeper REQUIRED #18 (waves 3+3.5 review, B2-14): corrected from a
    // stale `1.1` after `SamplerBuilder::new()`'s default was fixed to
    // `1.0`, matching `sampling::SamplingParams::default()` (RT-24 /
    // gatekeeper REQUIRED #1(a)).
    assert!((p.repetition_penalty - 1.0).abs() < f32::EPSILON);
}

#[test]
fn sampler_builder_negative_temperature_error() {
    let result = SamplerBuilder::new().temperature(-1.0).build();
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(
        err.to_string().contains("temperature"),
        "error should mention temperature"
    );
}

#[test]
fn sampler_builder_invalid_top_p_above_one() {
    let result = SamplerBuilder::new().top_p(2.0).build();
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(
        err.to_string().contains("top_p"),
        "error should mention top_p"
    );
}

#[test]
fn sampler_builder_invalid_top_p_below_zero() {
    let result = SamplerBuilder::new().top_p(-0.5).build();
    assert!(result.is_err());
}

#[test]
fn sampler_builder_invalid_repetition_penalty() {
    let result = SamplerBuilder::new().repetition_penalty(0.5).build();
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(
        err.to_string().contains("repetition_penalty"),
        "error should mention repetition_penalty"
    );
}

#[test]
fn sampler_builder_zero_temperature_valid() {
    let sampler = SamplerBuilder::new()
        .temperature(0.0)
        .build()
        .expect("zero temperature (greedy) should be valid");
    assert!(sampler.params().temperature < f32::EPSILON);
}

#[test]
fn sampler_builder_high_temperature_valid() {
    let sampler = SamplerBuilder::new()
        .temperature(10.0)
        .build()
        .expect("high temperature should be valid");
    assert!((sampler.params().temperature - 10.0).abs() < f32::EPSILON);
}

#[test]
fn sampler_builder_boundary_top_p_zero() {
    let sampler = SamplerBuilder::new()
        .top_p(0.0)
        .build()
        .expect("top_p=0.0 should be valid");
    assert!(sampler.params().top_p < f32::EPSILON);
}

#[test]
fn sampler_builder_boundary_top_p_one() {
    let sampler = SamplerBuilder::new()
        .top_p(1.0)
        .build()
        .expect("top_p=1.0 should be valid");
    assert!((sampler.params().top_p - 1.0).abs() < f32::EPSILON);
}

#[test]
fn sampler_builder_exact_repetition_penalty_one() {
    let sampler = SamplerBuilder::new()
        .repetition_penalty(1.0)
        .build()
        .expect("rep_pen=1.0 should be valid");
    assert!((sampler.params().repetition_penalty - 1.0).abs() < f32::EPSILON);
}

#[test]
fn sampler_builder_full_chain() {
    let sampler = SamplerBuilder::new()
        .temperature(0.3)
        .top_k(20)
        .top_p(0.85)
        .repetition_penalty(1.15)
        .seed(999)
        .build()
        .expect("full chain should succeed");
    let p = sampler.params();
    assert!((p.temperature - 0.3).abs() < f32::EPSILON);
    assert_eq!(p.top_k, 20);
    assert!((p.top_p - 0.85).abs() < f32::EPSILON);
    assert!((p.repetition_penalty - 1.15).abs() < f32::EPSILON);
}

#[test]
fn sampler_builder_default_trait() {
    let builder = SamplerBuilder::default();
    assert!(builder.build().is_ok());
}

// ── ConfigBuilder ────────────────────────────────────────────────────────

#[test]
fn config_builder_defaults_sensible() {
    let config = ConfigBuilder::new()
        .build()
        .expect("default config should be valid");
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 8080);
    assert_eq!(config.model.max_seq_len, 4096);
    assert!((config.sampling.temperature - 0.7).abs() < f32::EPSILON);
}

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
fn config_builder_invalid_max_seq_len() {
    let result = ConfigBuilder::new().max_seq_len(0).build();
    assert!(result.is_err(), "max_seq_len=0 should fail validation");
}

#[test]
fn config_builder_full_chain() {
    let model_path = std::env::temp_dir().join("model.gguf");
    let tokenizer_path = std::env::temp_dir().join("tokenizer.json");
    let config = ConfigBuilder::new()
        .model_path(model_path.display().to_string())
        .tokenizer_path(tokenizer_path.display().to_string())
        .max_seq_len(8192)
        .host("127.0.0.1")
        .port(3000)
        .log_level("debug")
        .json_logs(true)
        .temperature(0.5)
        .top_k(50)
        .top_p(0.95)
        .repetition_penalty(1.2)
        .max_tokens(1024)
        .build()
        .expect("full chain should succeed");

    assert_eq!(
        config.model.model_path.as_deref(),
        Some(model_path.to_str().expect("path is valid UTF-8"))
    );
    assert_eq!(config.model.max_seq_len, 8192);
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 3000);
    assert!(config.observability.json_logs);
    assert_eq!(config.sampling.max_tokens, 1024);
}

#[test]
fn config_builder_default_trait() {
    let builder = ConfigBuilder::default();
    assert!(builder.build().is_ok());
}

// ── EngineBuilder ────────────────────────────────────────────────────────

#[test]
fn engine_builder_defaults() {
    let (config, _sampler) = EngineBuilder::new()
        .build()
        .expect("default engine builder should succeed");
    assert_eq!(config.server.port, 8080);
}

#[test]
fn engine_builder_with_custom_config() {
    let config = ConfigBuilder::new()
        .port(9090)
        .build()
        .expect("config should build");
    let (result_config, _) = EngineBuilder::new()
        .config(config)
        .build()
        .expect("engine builder with config should succeed");
    assert_eq!(result_config.server.port, 9090);
}

#[test]
fn engine_builder_with_custom_sampler() {
    let sampler_builder = SamplerBuilder::new().temperature(0.3).seed(99);
    let (_, sampler) = EngineBuilder::new()
        .sampler(sampler_builder)
        .build()
        .expect("engine builder with sampler should succeed");
    assert!((sampler.params().temperature - 0.3).abs() < f32::EPSILON);
}

#[test]
fn engine_builder_with_kernel_tier() {
    let builder = EngineBuilder::new().kernel_tier("reference");
    assert_eq!(builder.configured_kernel_tier(), Some("reference"));
    assert!(builder.build().is_ok());
}

#[test]
fn engine_builder_invalid_sampler_propagates_error() {
    let sampler_builder = SamplerBuilder::new().temperature(-1.0);
    let result = EngineBuilder::new().sampler(sampler_builder).build();
    assert!(result.is_err(), "invalid sampler should propagate error");
}

#[test]
fn engine_builder_default_trait() {
    let builder = EngineBuilder::default();
    assert!(builder.build().is_ok());
}

// ── SamplingPresets ──────────────────────────────────────────────────────

#[test]
fn all_presets_produce_valid_params() {
    for preset in SamplingPreset::all() {
        let params = preset.params();
        assert!(
            params.temperature >= 0.0,
            "preset {} has negative temperature",
            preset.name()
        );
        assert!(
            params.top_p >= 0.0 && params.top_p <= 1.0,
            "preset {} has invalid top_p: {}",
            preset.name(),
            params.top_p
        );
        assert!(
            params.repetition_penalty >= 1.0,
            "preset {} has rep_pen < 1.0: {}",
            preset.name(),
            params.repetition_penalty
        );
    }
}

#[test]
fn preset_temperatures_reasonable() {
    for preset in SamplingPreset::all() {
        let params = preset.params();
        assert!(
            params.temperature <= 2.0,
            "preset {} has unreasonably high temperature: {}",
            preset.name(),
            params.temperature
        );
    }
}

#[test]
fn preset_names_and_descriptions_non_empty() {
    for preset in SamplingPreset::all() {
        assert!(!preset.name().is_empty(), "preset name should not be empty");
        assert!(
            !preset.description().is_empty(),
            "preset description should not be empty"
        );
    }
}

#[test]
fn preset_into_sampling_params() {
    let params: SamplingParams = SamplingPreset::Balanced.into();
    assert!((params.temperature - 0.7).abs() < f32::EPSILON);

    let params: SamplingParams = SamplingPreset::Greedy.into();
    assert!(params.temperature < f32::EPSILON);
}

#[test]
fn preset_display() {
    assert_eq!(format!("{}", SamplingPreset::Balanced), "Balanced");
    assert_eq!(format!("{}", SamplingPreset::Creative), "Creative");
    assert_eq!(format!("{}", SamplingPreset::Precise), "Precise");
    assert_eq!(format!("{}", SamplingPreset::Greedy), "Greedy");
    assert_eq!(
        format!("{}", SamplingPreset::Conversational),
        "Conversational"
    );
}

#[test]
fn preset_count() {
    assert_eq!(SamplingPreset::all().len(), 5);
}

// ── Builder + Preset integration ─────────────────────────────────────────

#[test]
fn builder_from_preset_balanced() {
    let params = SamplingPreset::Balanced.params();
    let sampler = SamplerBuilder::new()
        .temperature(params.temperature)
        .top_k(params.top_k)
        .top_p(params.top_p)
        .repetition_penalty(params.repetition_penalty)
        .build()
        .expect("balanced preset params should build");
    assert!((sampler.params().temperature - 0.7).abs() < f32::EPSILON);
}

#[test]
fn builder_from_preset_greedy() {
    let params = SamplingPreset::Greedy.params();
    let sampler = SamplerBuilder::new()
        .temperature(params.temperature)
        .top_k(params.top_k)
        .top_p(params.top_p)
        .repetition_penalty(params.repetition_penalty)
        .build()
        .expect("greedy preset params should build");
    assert!(sampler.params().temperature < f32::EPSILON);
}

#[test]
fn engine_builder_with_preset_sampler() {
    let preset_params = SamplingPreset::Precise.params();
    let (_, sampler) = EngineBuilder::new()
        .sampler(
            SamplerBuilder::new()
                .temperature(preset_params.temperature)
                .top_k(preset_params.top_k)
                .top_p(preset_params.top_p)
                .repetition_penalty(preset_params.repetition_penalty),
        )
        .build()
        .expect("engine builder with preset sampler should succeed");
    assert!((sampler.params().temperature - 0.1).abs() < f32::EPSILON);
}
