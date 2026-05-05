//! Integration tests for layered configuration loading.
//!
//! These exercise the four layers of [`oxibonsai_serve::config::ServerConfig`]:
//!
//! 1. Defaults
//! 2. TOML file
//! 3. Environment variables
//! 4. CLI arguments
//!
//! plus merging precedence and TOML round-tripping.

use std::path::PathBuf;

use oxibonsai_serve::config::{ConfigError, PartialServerConfig, ServerConfig};

fn temp_toml(contents: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut p = std::env::temp_dir();
    let name = format!(
        "oxibonsai-serve-config-{}-{}.toml",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    p.push(name);
    std::fs::write(&p, contents).expect("write temp toml");
    p
}

// ─── Default + roundtrip ──────────────────────────────────────────────────

#[test]
fn defaults_validate_cleanly() {
    let cfg = ServerConfig::default();
    cfg.validate().expect("defaults must validate");
}

#[test]
fn toml_roundtrip_defaults() {
    let cfg = ServerConfig::default();
    let serialized = cfg.to_toml_string().expect("serialize defaults");
    let reparsed = ServerConfig::from_toml(&serialized).expect("parse back");
    assert_eq!(cfg, reparsed);
}

#[test]
fn empty_toml_yields_defaults() {
    let cfg = ServerConfig::from_toml("").expect("empty toml should parse");
    assert_eq!(cfg, ServerConfig::default());
}

// ─── TOML parsing ─────────────────────────────────────────────────────────

#[test]
fn toml_parses_bind_section() {
    let body = r#"
[bind]
host = "192.168.1.10"
port = 9999
"#;
    let cfg = ServerConfig::from_toml(body).expect("parse");
    assert_eq!(cfg.bind.host, "192.168.1.10");
    assert_eq!(cfg.bind.port, 9999);
}

#[test]
fn toml_parses_sampling_section() {
    let body = r#"
[sampling]
default_max_tokens = 512
default_temperature = 0.3
default_top_p = 0.95
"#;
    let cfg = ServerConfig::from_toml(body).expect("parse");
    assert_eq!(cfg.sampling.default_max_tokens, 512);
    assert!((cfg.sampling.default_temperature - 0.3).abs() < f32::EPSILON);
    assert!((cfg.sampling.default_top_p - 0.95).abs() < f32::EPSILON);
}

#[test]
fn toml_parses_observability_section() {
    let body = r#"
[observability]
log_level = "debug"
metrics_enabled = false
metrics_path = "/prom"
"#;
    let cfg = ServerConfig::from_toml(body).expect("parse");
    assert_eq!(cfg.observability.log_level, "debug");
    assert!(!cfg.observability.metrics_enabled);
    assert_eq!(cfg.observability.metrics_path, "/prom");
}

#[test]
fn toml_parses_auth_bearer() {
    let body = r#"
[auth]
bearer_token = "abcdefghijklmnop"
"#;
    let cfg = ServerConfig::from_toml(body).expect("parse");
    assert_eq!(cfg.auth.bearer_token.as_deref(), Some("abcdefghijklmnop"));
}

#[test]
fn toml_parses_limits_section() {
    let body = r#"
[limits]
max_input_tokens = 2048
max_concurrent_requests = 128
per_request_timeout_ms = 5000
"#;
    let cfg = ServerConfig::from_toml(body).expect("parse");
    assert_eq!(cfg.limits.max_input_tokens, 2048);
    assert_eq!(cfg.limits.max_concurrent_requests, 128);
    assert_eq!(cfg.limits.per_request_timeout_ms, 5000);
}

#[test]
fn toml_parses_top_level_seed() {
    let body = r#"
seed = 1337
"#;
    let cfg = ServerConfig::from_toml(body).expect("parse");
    assert_eq!(cfg.seed, 1337);
}

#[test]
fn toml_partial_preserves_options() {
    // Only override port; leave everything else as default.
    let body = r#"
[bind]
port = 7777
"#;
    let partial = PartialServerConfig::from_toml_str(body).expect("parse");
    assert!(partial.host.is_none());
    assert_eq!(partial.port, Some(7777));
    assert!(partial.default_max_tokens.is_none());
}

#[test]
fn toml_bad_syntax_errors() {
    let err = ServerConfig::from_toml("this = is not [").expect_err("should fail");
    assert!(matches!(err, ConfigError::TomlParse(_)));
}

// ─── File-based loading ───────────────────────────────────────────────────

#[test]
fn from_toml_file_reads_from_disk() {
    let path = temp_toml(
        r#"
[bind]
host = "10.0.0.1"
port = 1234
"#,
    );
    let cfg = ServerConfig::from_toml_file(&path).expect("read");
    assert_eq!(cfg.bind.host, "10.0.0.1");
    assert_eq!(cfg.bind.port, 1234);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn from_toml_file_missing_path_is_io_error() {
    let missing = std::env::temp_dir().join("oxibonsai-serve-does-not-exist-12345.toml");
    let err = ServerConfig::from_toml_file(&missing).expect_err("should fail");
    assert!(matches!(err, ConfigError::Io { .. }));
}

// ─── Layered loading ──────────────────────────────────────────────────────

#[test]
fn load_with_only_defaults() {
    let cfg = ServerConfig::load(None, None, None).expect("load");
    assert_eq!(cfg, ServerConfig::default());
}

#[test]
fn load_toml_overrides_defaults() {
    let path = temp_toml(
        r#"
[bind]
port = 3333
"#,
    );
    let cfg = ServerConfig::load(Some(&path), None, None).expect("load");
    assert_eq!(cfg.bind.port, 3333);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_env_overrides_toml() {
    let path = temp_toml(
        r#"
[bind]
port = 3333
"#,
    );
    let env = PartialServerConfig {
        port: Some(4444),
        ..Default::default()
    };
    let cfg = ServerConfig::load(Some(&path), Some(env), None).expect("load");
    assert_eq!(cfg.bind.port, 4444);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_cli_overrides_env() {
    let env = PartialServerConfig {
        port: Some(5555),
        ..Default::default()
    };
    let cli = PartialServerConfig {
        port: Some(6666),
        ..Default::default()
    };
    let cfg = ServerConfig::load(None, Some(env), Some(cli)).expect("load");
    assert_eq!(cfg.bind.port, 6666);
}

#[test]
fn load_cli_beats_all_lower_layers() {
    let path = temp_toml(
        r#"
[bind]
port = 3333
[sampling]
default_temperature = 0.2
"#,
    );
    let env = PartialServerConfig {
        port: Some(4444),
        ..Default::default()
    };
    let cli = PartialServerConfig {
        port: Some(9000),
        default_temperature: Some(0.9),
        ..Default::default()
    };
    let cfg = ServerConfig::load(Some(&path), Some(env), Some(cli)).expect("load");
    assert_eq!(cfg.bind.port, 9000);
    assert!((cfg.sampling.default_temperature - 0.9).abs() < f32::EPSILON);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_validates_final_config() {
    let cli = PartialServerConfig {
        port: Some(0), // invalid
        ..Default::default()
    };
    let err = ServerConfig::load(None, None, Some(cli)).expect_err("should fail");
    assert!(matches!(err, ConfigError::Validation(_)));
}

#[test]
fn partial_merge_is_left_identity() {
    let p = PartialServerConfig {
        port: Some(4242),
        ..Default::default()
    };
    let merged = PartialServerConfig::default().merge(p.clone());
    assert_eq!(merged, p);
}

#[test]
fn canonical_example_parses_and_validates() {
    // The canonical `examples/server_config.toml` must be kept in sync with
    // the `ServerConfig` schema so newcomers copy-pasting it get a usable
    // starting point.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("server_config.toml");
    let body = std::fs::read_to_string(&path).expect("read canonical example");
    let cfg = ServerConfig::from_toml(&body).expect("parse canonical example");
    cfg.validate().expect("canonical example must validate");
}

#[test]
fn partial_merge_is_right_wins() {
    let a = PartialServerConfig {
        port: Some(1),
        host: Some("a".to_string()),
        ..Default::default()
    };
    let b = PartialServerConfig {
        port: Some(2),
        ..Default::default()
    };
    let merged = a.merge(b);
    assert_eq!(merged.port, Some(2));
    assert_eq!(merged.host.as_deref(), Some("a"));
}
