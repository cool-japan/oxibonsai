//! Layered server configuration.
//!
//! `ServerConfig` is the production-ready configuration object for
//! `oxibonsai-serve`.  It is built from up to four layers, each overriding the
//! previous one on a per-field basis:
//!
//! 1. **Defaults** — [`ServerConfig::default`] (baked in)
//! 2. **TOML file** — optional `--config <PATH>`
//! 3. **Environment variables** — `OXIBONSAI_*` prefix (see [`crate::env`])
//! 4. **CLI arguments** — [`crate::args::ServerArgs`] (highest precedence)
//!
//! Each layer is represented as a [`PartialServerConfig`] where every field is
//! an `Option`.  Merging is then a trivial `if Some(x) { self.x = Some(x) }`
//! pattern.  The final conversion to a concrete `ServerConfig` happens once at
//! the top level via [`ServerConfig::from_partial`].
//!
//! This scheme makes it easy to test each layer in isolation and to prove the
//! layering identity (see `tests/config_tests.rs` and
//! `tests/property_tests.rs`).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

// ─── Errors ──────────────────────────────────────────────────────────────

/// Errors arising while loading, parsing or validating a configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The supplied TOML string could not be parsed.
    #[error("failed to parse TOML config: {0}")]
    TomlParse(String),

    /// An environment variable could not be interpreted.
    #[error("failed to parse environment variable {name}: {reason}")]
    EnvParse {
        /// The name of the offending variable.
        name: String,
        /// Human-readable explanation of the parse failure.
        reason: String,
    },

    /// A validation rule was violated.
    #[error("configuration validation failed: {0}")]
    Validation(String),

    /// The config file could not be read from disk.
    #[error("failed to read config file {path}: {source}")]
    Io {
        /// Path of the file that could not be read.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

// ─── Sub-sections ────────────────────────────────────────────────────────

/// Bind-address section of the config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BindConfig {
    /// Host or interface to bind to.
    pub host: String,
    /// TCP port to listen on.
    pub port: u16,
}

impl Default for BindConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
        }
    }
}

/// Model-file section of the config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelConfig {
    /// Optional path to a GGUF file.
    pub path: Option<PathBuf>,
    /// Optional quantization hint (e.g. "TQ2" or "Q8_0").
    pub quantization_hint: Option<String>,
}

/// Tokenizer section of the config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TokenizerConfigSection {
    /// Optional path to a tokenizer.json file.
    pub path: Option<PathBuf>,
    /// Tokenizer kind — e.g. "huggingface" or "oxitok".
    pub kind: Option<String>,
}

/// Sampling-defaults section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SamplingConfig {
    /// Default maximum tokens to generate.
    pub default_max_tokens: usize,
    /// Default temperature.
    pub default_temperature: f32,
    /// Default top-p.
    pub default_top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            default_max_tokens: 256,
            default_temperature: 0.7,
            default_top_p: 1.0,
        }
    }
}

/// Resource-limit section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LimitsConfig {
    /// Maximum prompt length (in tokens).
    pub max_input_tokens: usize,
    /// Maximum number of concurrent requests.
    pub max_concurrent_requests: usize,
    /// Per-request timeout, in milliseconds.
    pub per_request_timeout_ms: u64,
    /// Number of inference-engine replicas for concurrent CPU serving.
    ///
    /// `None` (the default) resolves to `min(4, CPU cores)` on CPU tiers, so a
    /// few requests can generate in parallel out of the box. Replicas share one
    /// `Arc<[f32]>` token-embedding table, so each extra replica only costs a KV
    /// cache. An explicit value overrides this; the value is auto-clamped to `1`
    /// on the GPU/Metal tier (a process-global singleton). Distinct from
    /// [`Self::max_concurrent_requests`], which bounds HTTP-level admission
    /// rather than the number of generation engines.
    #[serde(default)]
    pub engine_pool_size: Option<usize>,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_input_tokens: 8192,
            max_concurrent_requests: 32,
            per_request_timeout_ms: 60_000,
            engine_pool_size: None,
        }
    }
}

/// Authentication section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AuthConfig {
    /// Optional bearer token required on auth-protected endpoints.
    pub bearer_token: Option<String>,
}

/// Observability section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ObservabilityConfig {
    /// Log level (one of: error/warn/info/debug/trace/off).
    pub log_level: String,
    /// Whether Prometheus metrics are enabled.
    pub metrics_enabled: bool,
    /// Path to serve Prometheus metrics at.
    pub metrics_path: String,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            metrics_enabled: true,
            metrics_path: "/metrics".to_string(),
        }
    }
}

// ─── Top-level config ────────────────────────────────────────────────────

/// Production-ready server configuration.
///
/// Obtain an instance with [`ServerConfig::load`] or via the explicit
/// layering helpers ([`ServerConfig::from_partial`],
/// [`PartialServerConfig::merge`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ServerConfig {
    /// Bind-address section.
    #[serde(default)]
    pub bind: BindConfig,
    /// Model section.
    #[serde(default)]
    pub model: ModelConfig,
    /// Tokenizer section.
    #[serde(default)]
    pub tokenizer: TokenizerConfigSection,
    /// Sampling defaults.
    #[serde(default)]
    pub sampling: SamplingConfig,
    /// Resource limits.
    #[serde(default)]
    pub limits: LimitsConfig,
    /// Authentication.
    #[serde(default)]
    pub auth: AuthConfig,
    /// Observability.
    #[serde(default)]
    pub observability: ObservabilityConfig,
    /// RNG seed (for deterministic sampling).
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_seed() -> u64 {
    42
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: BindConfig::default(),
            model: ModelConfig::default(),
            tokenizer: TokenizerConfigSection::default(),
            sampling: SamplingConfig::default(),
            limits: LimitsConfig::default(),
            auth: AuthConfig::default(),
            observability: ObservabilityConfig::default(),
            seed: default_seed(),
        }
    }
}

// ─── Partial config (for layering) ───────────────────────────────────────

/// Partial counterpart to [`ServerConfig`] where every field is optional.
///
/// This is the shape used by the TOML/env/CLI loading layers and by
/// [`PartialServerConfig::merge`].  Unset fields leave the previously layered
/// value untouched.
///
/// Deliberately *not* `#[non_exhaustive]` so downstream crates and integration
/// tests can construct partials with struct-expression syntax.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PartialServerConfig {
    /// Bind host.
    pub host: Option<String>,
    /// Bind port.
    pub port: Option<u16>,
    /// Model path.
    pub model_path: Option<PathBuf>,
    /// Quantization hint.
    pub quantization_hint: Option<String>,
    /// Tokenizer path.
    pub tokenizer_path: Option<PathBuf>,
    /// Tokenizer kind.
    pub tokenizer_kind: Option<String>,
    /// Default max tokens.
    pub default_max_tokens: Option<usize>,
    /// Default temperature.
    pub default_temperature: Option<f32>,
    /// Default top-p.
    pub default_top_p: Option<f32>,
    /// Maximum input tokens.
    pub max_input_tokens: Option<usize>,
    /// Maximum concurrent requests.
    pub max_concurrent_requests: Option<usize>,
    /// Number of inference-engine replicas for concurrent CPU serving.
    pub engine_pool_size: Option<usize>,
    /// Per-request timeout, milliseconds.
    pub per_request_timeout_ms: Option<u64>,
    /// Bearer token.
    pub bearer_token: Option<String>,
    /// Log level.
    pub log_level: Option<String>,
    /// Metrics enabled.
    pub metrics_enabled: Option<bool>,
    /// Metrics path.
    pub metrics_path: Option<String>,
    /// RNG seed.
    pub seed: Option<u64>,
}

impl PartialServerConfig {
    /// Merge `other` into `self`: every field set in `other` overrides the
    /// corresponding field in `self`.  Returns the merged result.
    pub fn merge(mut self, other: PartialServerConfig) -> Self {
        macro_rules! merge_field {
            ($name:ident) => {
                if other.$name.is_some() {
                    self.$name = other.$name;
                }
            };
        }
        merge_field!(host);
        merge_field!(port);
        merge_field!(model_path);
        merge_field!(quantization_hint);
        merge_field!(tokenizer_path);
        merge_field!(tokenizer_kind);
        merge_field!(default_max_tokens);
        merge_field!(default_temperature);
        merge_field!(default_top_p);
        merge_field!(max_input_tokens);
        merge_field!(max_concurrent_requests);
        merge_field!(engine_pool_size);
        merge_field!(per_request_timeout_ms);
        merge_field!(bearer_token);
        merge_field!(log_level);
        merge_field!(metrics_enabled);
        merge_field!(metrics_path);
        merge_field!(seed);
        self
    }

    /// Parse a TOML string into a partial config.
    ///
    /// Unlike [`ServerConfig::from_toml`], the result is a partial config so
    /// missing fields remain `None` and do not override downstream layers.
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        // We round-trip through a fully-populated helper struct so that any
        // extra fields are rejected and section-based layout is preserved.
        let helper: TomlHelper =
            toml::from_str(s).map_err(|e| ConfigError::TomlParse(e.to_string()))?;
        Ok(helper.into_partial())
    }
}

// ─── TOML helper shape ────────────────────────────────────────────────────

/// Mirror of the TOML schema, used only during parsing.
#[derive(Debug, Default, Deserialize)]
struct TomlHelper {
    #[serde(default)]
    bind: Option<BindPartial>,
    #[serde(default)]
    model: Option<ModelPartial>,
    #[serde(default)]
    tokenizer: Option<TokenizerPartial>,
    #[serde(default)]
    sampling: Option<SamplingPartial>,
    #[serde(default)]
    limits: Option<LimitsPartial>,
    #[serde(default)]
    auth: Option<AuthPartial>,
    #[serde(default)]
    observability: Option<ObservabilityPartial>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct BindPartial {
    host: Option<String>,
    port: Option<u16>,
}
#[derive(Debug, Default, Deserialize)]
struct ModelPartial {
    path: Option<PathBuf>,
    quantization_hint: Option<String>,
}
#[derive(Debug, Default, Deserialize)]
struct TokenizerPartial {
    path: Option<PathBuf>,
    kind: Option<String>,
}
#[derive(Debug, Default, Deserialize)]
struct SamplingPartial {
    default_max_tokens: Option<usize>,
    default_temperature: Option<f32>,
    default_top_p: Option<f32>,
}
#[derive(Debug, Default, Deserialize)]
struct LimitsPartial {
    max_input_tokens: Option<usize>,
    max_concurrent_requests: Option<usize>,
    engine_pool_size: Option<usize>,
    per_request_timeout_ms: Option<u64>,
}
#[derive(Debug, Default, Deserialize)]
struct AuthPartial {
    bearer_token: Option<String>,
}
#[derive(Debug, Default, Deserialize)]
struct ObservabilityPartial {
    log_level: Option<String>,
    metrics_enabled: Option<bool>,
    metrics_path: Option<String>,
}

impl TomlHelper {
    fn into_partial(self) -> PartialServerConfig {
        let bind = self.bind.unwrap_or_default();
        let model = self.model.unwrap_or_default();
        let tok = self.tokenizer.unwrap_or_default();
        let samp = self.sampling.unwrap_or_default();
        let lim = self.limits.unwrap_or_default();
        let auth = self.auth.unwrap_or_default();
        let obs = self.observability.unwrap_or_default();
        PartialServerConfig {
            host: bind.host,
            port: bind.port,
            model_path: model.path,
            quantization_hint: model.quantization_hint,
            tokenizer_path: tok.path,
            tokenizer_kind: tok.kind,
            default_max_tokens: samp.default_max_tokens,
            default_temperature: samp.default_temperature,
            default_top_p: samp.default_top_p,
            max_input_tokens: lim.max_input_tokens,
            max_concurrent_requests: lim.max_concurrent_requests,
            engine_pool_size: lim.engine_pool_size,
            per_request_timeout_ms: lim.per_request_timeout_ms,
            bearer_token: auth.bearer_token,
            log_level: obs.log_level,
            metrics_enabled: obs.metrics_enabled,
            metrics_path: obs.metrics_path,
            seed: self.seed,
        }
    }
}

// ─── ServerConfig construction helpers ────────────────────────────────────

impl ServerConfig {
    /// Parse a TOML string directly into a full `ServerConfig`, using defaults
    /// for any missing fields.
    pub fn from_toml(s: &str) -> Result<Self, ConfigError> {
        let partial = PartialServerConfig::from_toml_str(s)?;
        Ok(Self::from_partial(partial))
    }

    /// Build a full `ServerConfig` from a partial, filling unset fields with
    /// the [`Default`] values.
    pub fn from_partial(p: PartialServerConfig) -> Self {
        let mut out = Self::default();
        if let Some(v) = p.host {
            out.bind.host = v;
        }
        if let Some(v) = p.port {
            out.bind.port = v;
        }
        if let Some(v) = p.model_path {
            out.model.path = Some(v);
        }
        if let Some(v) = p.quantization_hint {
            out.model.quantization_hint = Some(v);
        }
        if let Some(v) = p.tokenizer_path {
            out.tokenizer.path = Some(v);
        }
        if let Some(v) = p.tokenizer_kind {
            out.tokenizer.kind = Some(v);
        }
        if let Some(v) = p.default_max_tokens {
            out.sampling.default_max_tokens = v;
        }
        if let Some(v) = p.default_temperature {
            out.sampling.default_temperature = v;
        }
        if let Some(v) = p.default_top_p {
            out.sampling.default_top_p = v;
        }
        if let Some(v) = p.max_input_tokens {
            out.limits.max_input_tokens = v;
        }
        if let Some(v) = p.max_concurrent_requests {
            out.limits.max_concurrent_requests = v;
        }
        if let Some(v) = p.engine_pool_size {
            out.limits.engine_pool_size = Some(v);
        }
        if let Some(v) = p.per_request_timeout_ms {
            out.limits.per_request_timeout_ms = v;
        }
        if let Some(v) = p.bearer_token {
            out.auth.bearer_token = Some(v);
        }
        if let Some(v) = p.log_level {
            out.observability.log_level = v;
        }
        if let Some(v) = p.metrics_enabled {
            out.observability.metrics_enabled = v;
        }
        if let Some(v) = p.metrics_path {
            out.observability.metrics_path = v;
        }
        if let Some(v) = p.seed {
            out.seed = v;
        }
        out
    }

    /// Serialize the current config to a TOML string.
    pub fn to_toml_string(&self) -> Result<String, ConfigError> {
        toml::to_string_pretty(self).map_err(|e| ConfigError::TomlParse(e.to_string()))
    }

    /// Load a configuration from a file on disk.
    pub fn from_toml_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, ConfigError> {
        let p = path.as_ref();
        let body = std::fs::read_to_string(p).map_err(|e| ConfigError::Io {
            path: p.display().to_string(),
            source: e,
        })?;
        Self::from_toml(&body)
    }

    /// Load a partial configuration from a file on disk.
    pub fn partial_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<PartialServerConfig, ConfigError> {
        let p = path.as_ref();
        let body = std::fs::read_to_string(p).map_err(|e| ConfigError::Io {
            path: p.display().to_string(),
            source: e,
        })?;
        PartialServerConfig::from_toml_str(&body)
    }

    /// Layered loader.
    ///
    /// 1. Start from [`ServerConfig::default`].
    /// 2. If `toml_path` is `Some`, merge the TOML file on top.
    /// 3. If `env` is `Some`, merge the env-derived partial on top.
    /// 4. If `cli` is `Some`, merge the CLI-derived partial on top (highest
    ///    precedence).
    ///
    /// The final result is validated via [`ServerConfig::validate`] before
    /// being returned.
    pub fn load(
        toml_path: Option<&std::path::Path>,
        env_partial: Option<PartialServerConfig>,
        cli_partial: Option<PartialServerConfig>,
    ) -> Result<Self, ConfigError> {
        let mut merged = PartialServerConfig::default();

        if let Some(p) = toml_path {
            let from_file = Self::partial_from_file(p)?;
            merged = merged.merge(from_file);
        }
        if let Some(env) = env_partial {
            merged = merged.merge(env);
        }
        if let Some(cli) = cli_partial {
            merged = merged.merge(cli);
        }

        let cfg = Self::from_partial(merged);
        cfg.validate()?;
        Ok(cfg)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_roundtrip() {
        let cfg = ServerConfig::default();
        let toml = cfg.to_toml_string().expect("to_toml");
        let parsed = ServerConfig::from_toml(&toml).expect("from_toml");
        assert_eq!(cfg, parsed);
    }

    #[test]
    fn partial_default_is_empty() {
        let p = PartialServerConfig::default();
        assert!(p.host.is_none());
        assert!(p.port.is_none());
    }

    #[test]
    fn partial_merge_overrides() {
        let a = PartialServerConfig {
            port: Some(1),
            log_level: Some("info".to_string()),
            ..Default::default()
        };
        let b = PartialServerConfig {
            port: Some(2),
            ..Default::default()
        };
        let merged = a.merge(b);
        assert_eq!(merged.port, Some(2));
        assert_eq!(merged.log_level.as_deref(), Some("info"));
    }

    #[test]
    fn from_partial_applies_fields() {
        let p = PartialServerConfig {
            host: Some("1.2.3.4".to_string()),
            port: Some(9999),
            default_top_p: Some(0.9),
            ..Default::default()
        };
        let cfg = ServerConfig::from_partial(p);
        assert_eq!(cfg.bind.host, "1.2.3.4");
        assert_eq!(cfg.bind.port, 9999);
        assert!((cfg.sampling.default_top_p - 0.9).abs() < f32::EPSILON);
    }
}
