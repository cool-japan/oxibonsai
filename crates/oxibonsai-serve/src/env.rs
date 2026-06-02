//! Environment-variable → configuration mapping.
//!
//! Recognised variables (all with `OXIBONSAI_` prefix):
//!
//! | Variable                       | Target field                           | Type   |
//! |--------------------------------|----------------------------------------|--------|
//! | `OXIBONSAI_HOST`               | `bind.host`                            | string |
//! | `OXIBONSAI_PORT`               | `bind.port`                            | u16    |
//! | `OXIBONSAI_MODEL_PATH`         | `model.path`                           | path   |
//! | `OXIBONSAI_TOKENIZER_PATH`     | `tokenizer.path`                       | path   |
//! | `OXIBONSAI_TOKENIZER_KIND`     | `tokenizer.kind`                       | string |
//! | `OXIBONSAI_MAX_TOKENS`         | `sampling.default_max_tokens`          | usize  |
//! | `OXIBONSAI_TEMPERATURE`        | `sampling.default_temperature`         | f32    |
//! | `OXIBONSAI_TOP_P`              | `sampling.default_top_p`               | f32    |
//! | `OXIBONSAI_MAX_INPUT_TOKENS`   | `limits.max_input_tokens`              | usize  |
//! | `OXIBONSAI_MAX_CONCURRENT`     | `limits.max_concurrent_requests`       | usize  |
//! | `OXIBONSAI_ENGINE_POOL_SIZE`   | `limits.engine_pool_size`              | usize  |
//! | `OXIBONSAI_REQUEST_TIMEOUT_MS` | `limits.per_request_timeout_ms`        | u64    |
//! | `OXIBONSAI_BEARER_TOKEN`       | `auth.bearer_token`                    | string |
//! | `OXIBONSAI_LOG_LEVEL`          | `observability.log_level`              | string |
//! | `OXIBONSAI_METRICS_ENABLED`    | `observability.metrics_enabled`        | bool   |
//! | `OXIBONSAI_METRICS_PATH`       | `observability.metrics_path`           | string |
//! | `OXIBONSAI_SEED`               | `seed`                                 | u64    |

use std::collections::HashMap;
use std::path::PathBuf;

use crate::config::{ConfigError, PartialServerConfig};

/// Parse an environment-variable map into a [`PartialServerConfig`].
///
/// Unrecognised keys are ignored (so arbitrary `OXIBONSAI_*` env vars do not
/// cause a hard failure).  Recognised keys with malformed values produce a
/// [`ConfigError::EnvParse`].
pub fn parse_env_map<I>(vars: I) -> Result<PartialServerConfig, ConfigError>
where
    I: IntoIterator<Item = (String, String)>,
{
    let map: HashMap<String, String> = vars.into_iter().collect();

    let mut out = PartialServerConfig::default();

    // ─── String-valued fields ────────────────────────────────────────────
    if let Some(v) = map.get("OXIBONSAI_HOST") {
        out.host = Some(v.to_string());
    }
    if let Some(v) = map.get("OXIBONSAI_MODEL_PATH") {
        out.model_path = Some(PathBuf::from(v));
    }
    if let Some(v) = map.get("OXIBONSAI_TOKENIZER_PATH") {
        out.tokenizer_path = Some(PathBuf::from(v));
    }
    if let Some(v) = map.get("OXIBONSAI_TOKENIZER_KIND") {
        out.tokenizer_kind = Some(v.to_string());
    }
    if let Some(v) = map.get("OXIBONSAI_BEARER_TOKEN") {
        out.bearer_token = Some(v.to_string());
    }
    if let Some(v) = map.get("OXIBONSAI_LOG_LEVEL") {
        out.log_level = Some(v.to_string());
    }
    if let Some(v) = map.get("OXIBONSAI_METRICS_PATH") {
        out.metrics_path = Some(v.to_string());
    }
    if let Some(v) = map.get("OXIBONSAI_QUANTIZATION_HINT") {
        out.quantization_hint = Some(v.to_string());
    }

    // ─── Numeric fields ──────────────────────────────────────────────────
    if let Some(v) = map.get("OXIBONSAI_PORT") {
        out.port = Some(parse_u16("OXIBONSAI_PORT", v)?);
    }
    if let Some(v) = map.get("OXIBONSAI_MAX_TOKENS") {
        out.default_max_tokens = Some(parse_usize("OXIBONSAI_MAX_TOKENS", v)?);
    }
    if let Some(v) = map.get("OXIBONSAI_TEMPERATURE") {
        out.default_temperature = Some(parse_f32("OXIBONSAI_TEMPERATURE", v)?);
    }
    if let Some(v) = map.get("OXIBONSAI_TOP_P") {
        out.default_top_p = Some(parse_f32("OXIBONSAI_TOP_P", v)?);
    }
    if let Some(v) = map.get("OXIBONSAI_MAX_INPUT_TOKENS") {
        out.max_input_tokens = Some(parse_usize("OXIBONSAI_MAX_INPUT_TOKENS", v)?);
    }
    if let Some(v) = map.get("OXIBONSAI_MAX_CONCURRENT") {
        out.max_concurrent_requests = Some(parse_usize("OXIBONSAI_MAX_CONCURRENT", v)?);
    }
    if let Some(v) = map.get("OXIBONSAI_ENGINE_POOL_SIZE") {
        out.engine_pool_size = Some(parse_usize("OXIBONSAI_ENGINE_POOL_SIZE", v)?);
    }
    if let Some(v) = map.get("OXIBONSAI_REQUEST_TIMEOUT_MS") {
        out.per_request_timeout_ms = Some(parse_u64("OXIBONSAI_REQUEST_TIMEOUT_MS", v)?);
    }
    if let Some(v) = map.get("OXIBONSAI_SEED") {
        out.seed = Some(parse_u64("OXIBONSAI_SEED", v)?);
    }

    // ─── Bool fields ─────────────────────────────────────────────────────
    if let Some(v) = map.get("OXIBONSAI_METRICS_ENABLED") {
        out.metrics_enabled = Some(parse_bool("OXIBONSAI_METRICS_ENABLED", v)?);
    }

    Ok(out)
}

/// Read from the live process environment (`std::env::vars`) into a partial
/// config.
///
/// This is a thin wrapper over [`parse_env_map`] provided for convenience in
/// `main`.
pub fn parse_process_env() -> Result<PartialServerConfig, ConfigError> {
    parse_env_map(std::env::vars())
}

// ─── Helper parsers ──────────────────────────────────────────────────────

fn parse_u16(name: &str, value: &str) -> Result<u16, ConfigError> {
    value.parse::<u16>().map_err(|e| ConfigError::EnvParse {
        name: name.to_string(),
        reason: format!("expected u16 ({e})"),
    })
}

fn parse_u64(name: &str, value: &str) -> Result<u64, ConfigError> {
    value.parse::<u64>().map_err(|e| ConfigError::EnvParse {
        name: name.to_string(),
        reason: format!("expected u64 ({e})"),
    })
}

fn parse_usize(name: &str, value: &str) -> Result<usize, ConfigError> {
    value.parse::<usize>().map_err(|e| ConfigError::EnvParse {
        name: name.to_string(),
        reason: format!("expected usize ({e})"),
    })
}

fn parse_f32(name: &str, value: &str) -> Result<f32, ConfigError> {
    value.parse::<f32>().map_err(|e| ConfigError::EnvParse {
        name: name.to_string(),
        reason: format!("expected f32 ({e})"),
    })
}

/// Parse a boolean env var accepting common spellings.
///
/// Truthy: `true`, `yes`, `on`, `1`.
/// Falsy: `false`, `no`, `off`, `0`.
/// Case-insensitive.
fn parse_bool(name: &str, value: &str) -> Result<bool, ConfigError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(ConfigError::EnvParse {
            name: name.to_string(),
            reason: format!("expected bool, got {other:?}"),
        }),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_env_yields_empty_partial() {
        let p = parse_env_map(std::iter::empty()).expect("should parse");
        assert_eq!(p, PartialServerConfig::default());
    }

    #[test]
    fn host_port_are_parsed() {
        let p = parse_env_map([
            ("OXIBONSAI_HOST".to_string(), "1.2.3.4".to_string()),
            ("OXIBONSAI_PORT".to_string(), "9090".to_string()),
        ])
        .expect("parse");
        assert_eq!(p.host.as_deref(), Some("1.2.3.4"));
        assert_eq!(p.port, Some(9090));
    }

    #[test]
    fn bad_port_errors() {
        let err = parse_env_map([("OXIBONSAI_PORT".to_string(), "abc".to_string())])
            .expect_err("should fail");
        assert!(matches!(err, ConfigError::EnvParse { .. }));
    }

    #[test]
    fn bool_yes_is_true() {
        assert!(parse_bool("X", "yes").expect("parse"));
        assert!(parse_bool("X", "on").expect("parse"));
        assert!(parse_bool("X", "1").expect("parse"));
        assert!(!parse_bool("X", "no").expect("parse"));
        assert!(!parse_bool("X", "off").expect("parse"));
        assert!(!parse_bool("X", "0").expect("parse"));
    }

    #[test]
    fn bool_bad_errors() {
        assert!(parse_bool("X", "maybe").is_err());
    }
}
