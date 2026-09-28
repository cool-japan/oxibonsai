//! Tests for the layered configuration system.

use oxibonsai_runtime::config::{
    ImagenConfig, ModelConfig, ObservabilityConfig, OxiBonsaiConfig, RopeScalingMode,
    SamplingConfig, ServerConfig,
};

// ═══════════════���════════════════��═════════════════════════════
// Default values
// ════��══════════════════════════════════════════════════��══════

#[test]
fn default_config_has_expected_values() {
    let cfg = OxiBonsaiConfig::default();
    assert_eq!(cfg.server.host, "0.0.0.0");
    assert_eq!(cfg.server.port, 8080);
    assert!((cfg.sampling.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.sampling.top_k, 40);
    assert!((cfg.sampling.top_p - 0.9).abs() < f32::EPSILON);
    // Gatekeeper REQUIRED #18 (waves 3+3.5 review, B2-14): corrected from a
    // stale `1.1` after `sampling::SamplingParams::default()` was fixed to
    // `1.0` (RT-24 / gatekeeper REQUIRED #1(a), the P0 CPU-vs-Metal greedy
    // parity fix) — every default-constructed sampling config must agree.
    assert!((cfg.sampling.repetition_penalty - 1.0).abs() < f32::EPSILON);
    assert_eq!(cfg.sampling.max_tokens, 512);
    assert!(cfg.model.model_path.is_none());
    assert!(cfg.model.tokenizer_path.is_none());
    assert_eq!(cfg.model.max_seq_len, 4096);
    assert_eq!(cfg.observability.log_level, "info");
    assert!(!cfg.observability.json_logs);
}

#[test]
fn server_config_defaults() {
    let cfg = ServerConfig::default();
    assert_eq!(cfg.host, "0.0.0.0");
    assert_eq!(cfg.port, 8080);
}

#[test]
fn sampling_config_defaults_match_params() {
    let cfg = SamplingConfig::default();
    assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 40);
    assert!((cfg.top_p - 0.9).abs() < f32::EPSILON);
    // See the identical note in `default_config_has_expected_values` above.
    assert!((cfg.repetition_penalty - 1.0).abs() < f32::EPSILON);
    assert_eq!(cfg.max_tokens, 512);
}

#[test]
fn model_config_defaults() {
    let cfg = ModelConfig::default();
    assert!(cfg.model_path.is_none());
    assert!(cfg.tokenizer_path.is_none());
    assert_eq!(cfg.max_seq_len, 4096);
}

#[test]
fn observability_config_default_log_level_is_info() {
    let cfg = ObservabilityConfig::default();
    assert_eq!(cfg.log_level, "info");
    assert!(!cfg.json_logs);
}

// ══════════════════════════════════════════════════════════════
// TOML parsing
// ═══════════��══════════════════════════════════════════════════

#[test]
fn toml_with_all_fields_parses_correctly() {
    let model_path = std::env::temp_dir().join("model.gguf");
    let tokenizer_path = std::env::temp_dir().join("tokenizer.json");
    let toml_str = format!(
        r#"
[server]
host = "127.0.0.1"
port = 3000

[sampling]
temperature = 0.5
top_k = 50
top_p = 0.95
repetition_penalty = 1.2
max_tokens = 1024

[model]
model_path = "{}"
tokenizer_path = "{}"
max_seq_len = 8192

[observability]
log_level = "debug"
json_logs = true
"#,
        model_path.display(),
        tokenizer_path.display()
    );
    let cfg: OxiBonsaiConfig = toml::from_str(&toml_str).expect("should parse complete TOML");
    assert_eq!(cfg.server.host, "127.0.0.1");
    assert_eq!(cfg.server.port, 3000);
    assert!((cfg.sampling.temperature - 0.5).abs() < f32::EPSILON);
    assert_eq!(cfg.sampling.top_k, 50);
    assert!((cfg.sampling.top_p - 0.95).abs() < f32::EPSILON);
    assert!((cfg.sampling.repetition_penalty - 1.2).abs() < f32::EPSILON);
    assert_eq!(cfg.sampling.max_tokens, 1024);
    assert_eq!(
        cfg.model.model_path.as_deref(),
        Some(model_path.to_str().expect("path is valid UTF-8"))
    );
    assert_eq!(
        cfg.model.tokenizer_path.as_deref(),
        Some(tokenizer_path.to_str().expect("path is valid UTF-8"))
    );
    assert_eq!(cfg.model.max_seq_len, 8192);
    assert_eq!(cfg.observability.log_level, "debug");
    assert!(cfg.observability.json_logs);
}

#[test]
fn toml_with_partial_fields_uses_defaults() {
    let toml_str = r#"
[server]
port = 9090
"#;
    let cfg: OxiBonsaiConfig = toml::from_str(toml_str).expect("should parse partial TOML");
    assert_eq!(cfg.server.port, 9090);
    assert_eq!(cfg.server.host, "0.0.0.0"); // default
    assert!((cfg.sampling.temperature - 0.7).abs() < f32::EPSILON); // default
    assert_eq!(cfg.sampling.top_k, 40); // default
    assert_eq!(cfg.model.max_seq_len, 4096); // default
    assert_eq!(cfg.observability.log_level, "info"); // default
}

#[test]
fn toml_with_only_sampling_section() {
    let toml_str = r#"
[sampling]
temperature = 0.0
top_k = 1
"#;
    let cfg: OxiBonsaiConfig = toml::from_str(toml_str).expect("should parse sampling-only TOML");
    assert!(cfg.sampling.temperature.abs() < f32::EPSILON);
    assert_eq!(cfg.sampling.top_k, 1);
    // Others remain default
    assert_eq!(cfg.server.port, 8080);
}

#[test]
fn invalid_toml_returns_error() {
    let bad_toml = "this is not valid [[[toml";
    let result: Result<OxiBonsaiConfig, _> = toml::from_str(bad_toml);
    assert!(result.is_err(), "invalid TOML should fail to parse");
}

#[test]
fn empty_toml_string_parses_to_defaults() {
    let cfg: OxiBonsaiConfig = toml::from_str("").expect("empty string should parse to defaults");
    assert_eq!(cfg.server.host, "0.0.0.0");
    assert_eq!(cfg.server.port, 8080);
    assert!((cfg.sampling.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.observability.log_level, "info");
}

// ═��════════════════���═══════════════════════════════════════════
// File loading
// ════════════════════════��═════════════════════════════════════

#[test]
fn missing_file_path_returns_default_via_load_or_default() {
    let cfg = OxiBonsaiConfig::load_or_default(None);
    assert_eq!(cfg.server.host, "0.0.0.0");
    assert_eq!(cfg.server.port, 8080);
}

#[test]
fn nonexistent_file_returns_default_via_load_or_default() {
    let path = std::env::temp_dir().join("nonexistent_oxibonsai_test_99999.toml");
    let cfg = OxiBonsaiConfig::load_or_default(Some(&path));
    assert_eq!(cfg.server.port, 8080);
}

#[test]
fn load_from_temp_file() {
    let dir = std::env::temp_dir();
    let path = dir.join("oxibonsai_config_test_load.toml");
    std::fs::write(
        &path,
        r#"
[server]
host = "10.0.0.1"
port = 5555

[sampling]
temperature = 0.3
"#,
    )
    .expect("write temp config");

    let cfg = OxiBonsaiConfig::load(&path).expect("should load temp config");
    assert_eq!(cfg.server.host, "10.0.0.1");
    assert_eq!(cfg.server.port, 5555);
    assert!((cfg.sampling.temperature - 0.3).abs() < f32::EPSILON);
    // Defaults for unspecified
    assert_eq!(cfg.sampling.top_k, 40);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_invalid_file_returns_error() {
    let dir = std::env::temp_dir();
    let path = dir.join("oxibonsai_config_test_invalid.toml");
    std::fs::write(&path, "not valid toml {{{").expect("write invalid config");

    let result = OxiBonsaiConfig::load(&path);
    assert!(result.is_err(), "invalid file should return error");

    let _ = std::fs::remove_file(&path);
}

// ═════��════════════════════════════════════════════════════════
// Round-trip: serialize then deserialize
// ════════════════════��════════════════════════════════════��════

#[test]
fn config_roundtrip_serialize_deserialize() {
    let original = OxiBonsaiConfig {
        server: ServerConfig {
            host: "192.168.1.1".to_string(),
            port: 7777,
        },
        sampling: SamplingConfig {
            temperature: 0.42,
            top_k: 30,
            top_p: 0.85,
            repetition_penalty: 1.3,
            max_tokens: 2048,
        },
        model: ModelConfig {
            model_path: Some("/path/to/model.gguf".to_string()),
            tokenizer_path: Some("/path/to/tokenizer.json".to_string()),
            max_seq_len: 16384,
            // RT-ADMIN-CFG (spec item 9) added these two fields in
            // config.rs to prepare B2-12's (wave 3) context-guard formula;
            // this exhaustive same-crate struct literal needs them listed
            // too. `#[non_exhaustive]` would not help here since this file
            // is in the same crate as the struct.
            max_context: None,
            ctx_budget_bytes: None,
            // B2-14 (wave 4b) added `rope_scaling` to `ModelConfig` (the
            // `--rope-scaling auto|on|off` CLI/TOML control) — same
            // situation as `max_context`/`ctx_budget_bytes` above.
            rope_scaling: RopeScalingMode::Off,
        },
        observability: ObservabilityConfig {
            log_level: "trace".to_string(),
            json_logs: true,
        },
        // B2-12 (wave 3) added the `imagen` field to `OxiBonsaiConfig` (the
        // enabling half of an `[imagen]` config-file section; CLI wiring is
        // B2-14's) — same situation as `max_context`/`ctx_budget_bytes`
        // above: this exhaustive same-crate struct literal needs it listed.
        imagen: ImagenConfig {
            model_path: Some("/models/bonsai-image".to_string()),
            width: 768,
            height: 768,
            steps: 12,
            guidance_scale: 5.0,
            seed: Some(7),
            output_dir: Some("/tmp/imagen-out".to_string()),
        },
    };

    let toml_str = toml::to_string(&original).expect("serialize");
    let parsed: OxiBonsaiConfig = toml::from_str(&toml_str).expect("deserialize");

    assert_eq!(parsed.server.host, original.server.host);
    assert_eq!(parsed.server.port, original.server.port);
    assert!((parsed.sampling.temperature - original.sampling.temperature).abs() < f32::EPSILON);
    assert_eq!(parsed.sampling.top_k, original.sampling.top_k);
    assert!((parsed.sampling.top_p - original.sampling.top_p).abs() < f32::EPSILON);
    assert!(
        (parsed.sampling.repetition_penalty - original.sampling.repetition_penalty).abs()
            < f32::EPSILON
    );
    assert_eq!(parsed.sampling.max_tokens, original.sampling.max_tokens);
    assert_eq!(parsed.model.model_path, original.model.model_path);
    assert_eq!(parsed.model.tokenizer_path, original.model.tokenizer_path);
    assert_eq!(parsed.model.max_seq_len, original.model.max_seq_len);
    assert_eq!(parsed.model.rope_scaling, original.model.rope_scaling);
    assert_eq!(
        parsed.observability.log_level,
        original.observability.log_level
    );
    assert_eq!(
        parsed.observability.json_logs,
        original.observability.json_logs
    );
    assert_eq!(parsed.imagen.model_path, original.imagen.model_path);
    assert_eq!(parsed.imagen.width, original.imagen.width);
    assert_eq!(parsed.imagen.height, original.imagen.height);
    assert_eq!(parsed.imagen.steps, original.imagen.steps);
    assert!((parsed.imagen.guidance_scale - original.imagen.guidance_scale).abs() < f32::EPSILON);
    assert_eq!(parsed.imagen.seed, original.imagen.seed);
    assert_eq!(parsed.imagen.output_dir, original.imagen.output_dir);
}
