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
    /// `127.0.0.1` (loopback-only) is the safe default (findings `sec-15` /
    /// `SV-07` / `sec-M2`): a fresh install that sets no `--host` / `host=`
    /// TOML key / `OXIBONSAI_HOST` env var binds to loopback only, so it is
    /// never reachable from the network before an operator makes an
    /// explicit decision. Binding a non-loopback address (e.g. `0.0.0.0`)
    /// without either a bearer token or `auth.insecure_no_auth = true` is
    /// refused at startup — see `main.rs`'s bind-safety guard.
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
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
    /// Maximum accepted HTTP request body size, in bytes (findings `SV-27` /
    /// `sec-16`). Enforced via `axum::extract::DefaultBodyLimit::max`; a
    /// request whose body exceeds this is rejected with `413 Payload Too
    /// Large` before the JSON body is fully buffered.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Hard ceiling on a request's effective `max_tokens` (finding `SV-28`):
    /// `--max-output-tokens`, `[limits] max_output_tokens` or
    /// `OXIBONSAI_MAX_OUTPUT_TOKENS`. A request above it is refused with
    /// `400`. `None` keeps `oxibonsai_runtime::server::MAX_OUTPUT_TOKENS`'s
    /// compiled-in default; `0` is rejected at parse time (it would refuse
    /// every request instead of capping it).
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
}

/// Default request body ceiling: 4 MiB. Comfortably above axum's implicit
/// 2 MiB default (which a full-context multi-turn conversation on a
/// 262 K-token model can exceed — see `CONTEXT.md`'s Bonsai 2 facts) while
/// still bounding memory per in-flight request.
pub fn default_max_body_bytes() -> usize {
    4 * 1024 * 1024
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_input_tokens: 8192,
            max_concurrent_requests: 32,
            per_request_timeout_ms: 60_000,
            engine_pool_size: None,
            max_body_bytes: default_max_body_bytes(),
            max_output_tokens: None,
        }
    }
}

/// Cross-Origin Resource Sharing section (findings `sec-06` / `SV-06` /
/// `cli-10`).
///
/// Empty `allowed_origins` (the default) means **no** CORS headers are
/// emitted on any route — CORS is strictly opt-in, matching
/// `oxibonsai_runtime::middleware::MiddlewareConfig::default()`. Configure
/// via repeatable `--cors-origin <ORIGIN>`, the `[cors] allowed_origins`
/// TOML array, or a comma-separated `OXIBONSAI_CORS_ORIGIN` env var.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CorsSection {
    /// Allowed origins. An entry of the literal `"*"` allows all origins
    /// (only ever emitted as a literal `*` response header when
    /// `allow_credentials` is `false` — see
    /// `oxibonsai_runtime::middleware::CorsConfig::is_unrestricted_wildcard`).
    pub allowed_origins: Vec<String>,
    /// Whether to send `Access-Control-Allow-Credentials: true`. Rejected at
    /// validation time when combined with a literal `"*"` in
    /// `allowed_origins` (finding `sec-09` / `SV-20`): the two together let
    /// any web page read authenticated responses from any origin, and the
    /// Fetch spec forbids the browser from honoring that combination anyway.
    pub allow_credentials: bool,
}

/// Per-client token-bucket rate-limit section (findings `sec-07` / `RT-34` /
/// `SV-10` / `cli-10`).
///
/// `rpm` (requests per minute) is `None` by default, meaning rate limiting
/// is disabled — an operator opts in explicitly via `--rate-limit-rpm`, the
/// `[rate_limit] rpm` TOML key, or `OXIBONSAI_RATE_LIMIT_RPM`. Internally
/// this is converted to the `rps` (requests per second) unit
/// `oxibonsai_runtime::rate_limiter::RateLimitConfig` uses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RateLimitSection {
    /// Steady-state requests per minute, per client. `None` disables rate
    /// limiting.
    pub rpm: Option<f64>,
    /// Burst capacity: maximum tokens (requests) a client can accumulate
    /// before being throttled. Only meaningful when `rpm` is set.
    pub burst: f64,
}

/// Default burst capacity when rate limiting is enabled but no explicit
/// `--rate-limit-burst` is given: two minutes' worth of the configured rate
/// cushions short bursts without materially weakening the steady-state cap.
pub fn default_rate_limit_burst() -> f64 {
    20.0
}

impl Default for RateLimitSection {
    fn default() -> Self {
        Self {
            rpm: None,
            burst: default_rate_limit_burst(),
        }
    }
}

/// Authentication section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AuthConfig {
    /// Optional bearer token required on auth-protected endpoints.
    ///
    /// Finding `SV-33`: a value set here via `--bearer-token` is visible to
    /// any local user via `ps`. Prefer [`AuthConfig::bearer_token_file`] or
    /// the `OXIBONSAI_BEARER_TOKEN` environment variable; `main.rs` emits a
    /// startup `warn!` when the CLI flag specifically was used.
    pub bearer_token: Option<String>,
    /// Path to a file containing the bearer token (its trimmed contents are
    /// used verbatim), as a `ps`-safe alternative to `--bearer-token`
    /// (finding `SV-33`). Takes precedence over `bearer_token` when both are
    /// set — see `main.rs`'s `resolve_bearer_token`.
    pub bearer_token_file: Option<PathBuf>,
    /// Optional admin credential for `/admin/*`, independent of
    /// `bearer_token` (finding `sec-15`). When unset, the admin-token
    /// environment variable (`OXI_ADMIN_TOKEN`, read by
    /// `oxibonsai_runtime::server::AuthConfig::from_env`) is still honored;
    /// when both are unset `/admin/*` is refused with `403` unconditionally.
    pub admin_token: Option<String>,
    /// Explicit operator opt-out of the "refuse a non-loopback bind with no
    /// auth configured" startup guard (findings `sec-15` / `SV-07`). Only
    /// meaningful when `bind.host` resolves to a non-loopback address and
    /// neither `bearer_token` nor `admin_token` is set; has no effect
    /// otherwise. Prefer configuring a token instead of setting this.
    pub insecure_no_auth: bool,
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

/// Browser-UI section (finding `SV-26`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct UiSection {
    /// Mount the bundled chat UI at `GET /ui`: `--enable-ui`, `[ui] enabled`
    /// or `OXIBONSAI_ENABLE_UI`. Off by default — the UI is a debugging
    /// convenience, unauthenticated whenever the whole server is.
    pub enabled: bool,
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
    /// CORS policy.
    #[serde(default)]
    pub cors: CorsSection,
    /// Per-client rate limiting.
    #[serde(default)]
    pub rate_limit: RateLimitSection,
    /// Browser UI.
    #[serde(default)]
    pub ui: UiSection,
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
            cors: CorsSection::default(),
            rate_limit: RateLimitSection::default(),
            ui: UiSection::default(),
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
    /// Path to a file containing the bearer token (finding `SV-33`).
    pub bearer_token_file: Option<PathBuf>,
    /// Log level.
    pub log_level: Option<String>,
    /// Metrics enabled.
    pub metrics_enabled: Option<bool>,
    /// Metrics path.
    pub metrics_path: Option<String>,
    /// Admin token (`--admin-token` / `admin_token` / `OXI_ADMIN_TOKEN`; see
    /// [`AuthConfig::admin_token`]).
    pub admin_token: Option<String>,
    /// Explicit opt-out of the non-loopback-bind-without-auth guard.
    pub insecure_no_auth: Option<bool>,
    /// Maximum accepted request body size, in bytes.
    pub max_body_bytes: Option<usize>,
    /// CORS allowed origins (repeatable `--cors-origin`, `[cors]
    /// allowed_origins`, or comma-separated `OXIBONSAI_CORS_ORIGIN`).
    pub cors_allowed_origins: Option<Vec<String>>,
    /// Whether to send `Access-Control-Allow-Credentials: true`.
    pub cors_allow_credentials: Option<bool>,
    /// Rate limit: requests per minute per client.
    pub rate_limit_rpm: Option<f64>,
    /// Rate limit: burst capacity.
    pub rate_limit_burst: Option<f64>,
    /// Mount the bundled chat UI at `GET /ui` (see [`UiSection::enabled`]).
    pub enable_ui: Option<bool>,
    /// Per-request `max_tokens` ceiling (see [`LimitsConfig::max_output_tokens`]).
    pub max_output_tokens: Option<usize>,
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
        merge_field!(bearer_token_file);
        merge_field!(log_level);
        merge_field!(metrics_enabled);
        merge_field!(metrics_path);
        merge_field!(admin_token);
        merge_field!(insecure_no_auth);
        merge_field!(max_body_bytes);
        merge_field!(cors_allowed_origins);
        merge_field!(cors_allow_credentials);
        merge_field!(rate_limit_rpm);
        merge_field!(rate_limit_burst);
        merge_field!(enable_ui);
        merge_field!(max_output_tokens);
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
        let partial = helper.into_partial();
        reject_zero_output_ceiling(partial.max_output_tokens, "[limits] max_output_tokens")?;
        Ok(partial)
    }
}

/// `0` as the `max_tokens` ceiling would refuse every request outright
/// instead of capping it — the same rule `--max-output-tokens` applies.
fn reject_zero_output_ceiling(value: Option<usize>, source: &str) -> Result<(), ConfigError> {
    if value == Some(0) {
        return Err(ConfigError::Validation(format!(
            "{source} must be at least 1 (a zero ceiling would reject every request outright \
             instead of capping it)"
        )));
    }
    Ok(())
}

/// The environment layer of the two SRV-OPENAI knobs: `OXIBONSAI_ENABLE_UI`
/// (`true`/`false`/`1`/`0`) and `OXIBONSAI_MAX_OUTPUT_TOKENS` (a positive
/// integer). Unrelated variables are ignored. Kept next to the fields it
/// fills; `crate::env::parse_env_map` merges it into the full env layer.
///
/// # Errors
///
/// [`ConfigError::EnvParse`] for a malformed value (including a `0`
/// ceiling).
pub fn parse_ui_env_map<I>(vars: I) -> Result<PartialServerConfig, ConfigError>
where
    I: IntoIterator<Item = (String, String)>,
{
    let mut out = PartialServerConfig::default();
    for (name, value) in vars {
        match name.as_str() {
            "OXIBONSAI_ENABLE_UI" => {
                let enabled = match value.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "yes" | "on" => true,
                    "0" | "false" | "no" | "off" => false,
                    _ => {
                        return Err(ConfigError::EnvParse {
                            name,
                            reason: format!("expected a boolean, got '{value}'"),
                        })
                    }
                };
                out.enable_ui = Some(enabled);
            }
            "OXIBONSAI_MAX_OUTPUT_TOKENS" => {
                let ceiling = value
                    .trim()
                    .parse::<usize>()
                    .map_err(|e| ConfigError::EnvParse {
                        name: name.clone(),
                        reason: format!("expected a positive integer ({e})"),
                    })?;
                if ceiling == 0 {
                    return Err(ConfigError::EnvParse {
                        name,
                        reason: "must be at least 1 (a zero ceiling would reject every \
                                 request outright instead of capping it)"
                            .to_string(),
                    });
                }
                out.max_output_tokens = Some(ceiling);
            }
            _ => {}
        }
    }
    Ok(out)
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
    cors: Option<CorsPartial>,
    #[serde(default)]
    rate_limit: Option<RateLimitPartial>,
    #[serde(default)]
    ui: Option<UiPartial>,
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
    max_body_bytes: Option<usize>,
    max_output_tokens: Option<usize>,
}
#[derive(Debug, Default, Deserialize)]
struct UiPartial {
    enabled: Option<bool>,
}
#[derive(Debug, Default, Deserialize)]
struct AuthPartial {
    bearer_token: Option<String>,
    bearer_token_file: Option<PathBuf>,
    admin_token: Option<String>,
    insecure_no_auth: Option<bool>,
}
#[derive(Debug, Default, Deserialize)]
struct ObservabilityPartial {
    log_level: Option<String>,
    metrics_enabled: Option<bool>,
    metrics_path: Option<String>,
}
#[derive(Debug, Default, Deserialize)]
struct CorsPartial {
    allowed_origins: Option<Vec<String>>,
    allow_credentials: Option<bool>,
}
#[derive(Debug, Default, Deserialize)]
struct RateLimitPartial {
    rpm: Option<f64>,
    burst: Option<f64>,
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
        let cors = self.cors.unwrap_or_default();
        let rl = self.rate_limit.unwrap_or_default();
        let ui = self.ui.unwrap_or_default();
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
            bearer_token_file: auth.bearer_token_file,
            log_level: obs.log_level,
            metrics_enabled: obs.metrics_enabled,
            metrics_path: obs.metrics_path,
            admin_token: auth.admin_token,
            insecure_no_auth: auth.insecure_no_auth,
            max_body_bytes: lim.max_body_bytes,
            cors_allowed_origins: cors.allowed_origins,
            cors_allow_credentials: cors.allow_credentials,
            rate_limit_rpm: rl.rpm,
            rate_limit_burst: rl.burst,
            enable_ui: ui.enabled,
            max_output_tokens: lim.max_output_tokens,
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
        if let Some(v) = p.bearer_token_file {
            out.auth.bearer_token_file = Some(v);
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
        if let Some(v) = p.admin_token {
            out.auth.admin_token = Some(v);
        }
        if let Some(v) = p.insecure_no_auth {
            out.auth.insecure_no_auth = v;
        }
        if let Some(v) = p.max_body_bytes {
            out.limits.max_body_bytes = v;
        }
        if let Some(v) = p.cors_allowed_origins {
            out.cors.allowed_origins = v;
        }
        if let Some(v) = p.cors_allow_credentials {
            out.cors.allow_credentials = v;
        }
        if let Some(v) = p.rate_limit_rpm {
            out.rate_limit.rpm = Some(v);
        }
        if let Some(v) = p.rate_limit_burst {
            out.rate_limit.burst = v;
        }
        if let Some(v) = p.enable_ui {
            out.ui.enabled = v;
        }
        if let Some(v) = p.max_output_tokens {
            out.limits.max_output_tokens = Some(v);
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

    // ─── sec-15 / SV-07 / sec-M2: safe bind default ────────────────────────

    #[test]
    fn default_bind_host_is_loopback() {
        // A fresh install (no --host / TOML / env) must never be reachable
        // from the network before the operator makes an explicit decision.
        let cfg = ServerConfig::default();
        assert_eq!(cfg.bind.host, "127.0.0.1");
    }

    // ─── cors / rate_limit sections (SV-06/sec-06/sec-07/SV-10/cli-10) ─────

    #[test]
    fn cors_and_rate_limit_default_to_disabled() {
        let cfg = ServerConfig::default();
        assert!(cfg.cors.allowed_origins.is_empty());
        assert!(!cfg.cors.allow_credentials);
        assert!(cfg.rate_limit.rpm.is_none());
    }

    #[test]
    fn toml_cors_and_rate_limit_sections_parse() {
        let toml = r#"
[cors]
allowed_origins = ["https://app.example.com", "https://b.example.com"]
allow_credentials = false

[rate_limit]
rpm = 600.0
burst = 50.0
"#;
        let cfg = ServerConfig::from_toml(toml).expect("parse");
        assert_eq!(
            cfg.cors.allowed_origins,
            vec![
                "https://app.example.com".to_string(),
                "https://b.example.com".to_string()
            ]
        );
        assert!(!cfg.cors.allow_credentials);
        assert_eq!(cfg.rate_limit.rpm, Some(600.0));
        assert!((cfg.rate_limit.burst - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn admin_token_and_insecure_no_auth_roundtrip() {
        let toml = r#"
[auth]
admin_token = "super-secret-admin-token"
insecure_no_auth = true
"#;
        let cfg = ServerConfig::from_toml(toml).expect("parse");
        assert_eq!(
            cfg.auth.admin_token.as_deref(),
            Some("super-secret-admin-token")
        );
        assert!(cfg.auth.insecure_no_auth);

        let rendered = cfg.to_toml_string().expect("to_toml");
        let reparsed = ServerConfig::from_toml(&rendered).expect("from_toml");
        assert_eq!(reparsed, cfg);
    }

    #[test]
    fn bearer_token_file_roundtrip() {
        let toml = r#"
[auth]
bearer_token_file = "/etc/oxibonsai/bearer-token"
"#;
        let cfg = ServerConfig::from_toml(toml).expect("parse");
        assert_eq!(
            cfg.auth.bearer_token_file,
            Some(PathBuf::from("/etc/oxibonsai/bearer-token"))
        );
        assert!(cfg.auth.bearer_token.is_none());
    }

    #[test]
    fn bearer_token_file_via_partial_merge() {
        let a = PartialServerConfig {
            bearer_token: Some("cli-flag-value".to_string()),
            ..Default::default()
        };
        let b = PartialServerConfig {
            bearer_token_file: Some(PathBuf::from("/run/secrets/token")),
            ..Default::default()
        };
        let merged = a.merge(b);
        assert_eq!(merged.bearer_token.as_deref(), Some("cli-flag-value"));
        assert_eq!(
            merged.bearer_token_file,
            Some(PathBuf::from("/run/secrets/token"))
        );
    }

    #[test]
    fn max_body_bytes_has_a_sane_default_and_is_configurable() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.limits.max_body_bytes, default_max_body_bytes());
        assert!(
            cfg.limits.max_body_bytes > 2 * 1024 * 1024,
            "must exceed axum's implicit 2 MiB default"
        );

        let toml = "[limits]\nmax_body_bytes = 1048576\n";
        let cfg = ServerConfig::from_toml(toml).expect("parse");
        assert_eq!(cfg.limits.max_body_bytes, 1_048_576);
    }

    #[test]
    fn partial_new_fields_default_to_none() {
        let p = PartialServerConfig::default();
        assert!(p.admin_token.is_none());
        assert!(p.insecure_no_auth.is_none());
        assert!(p.max_body_bytes.is_none());
        assert!(p.cors_allowed_origins.is_none());
        assert!(p.cors_allow_credentials.is_none());
        assert!(p.rate_limit_rpm.is_none());
        assert!(p.rate_limit_burst.is_none());
    }

    #[test]
    fn partial_merge_overrides_new_fields() {
        let a = PartialServerConfig {
            rate_limit_rpm: Some(60.0),
            cors_allowed_origins: Some(vec!["https://a.example".to_string()]),
            ..Default::default()
        };
        let b = PartialServerConfig {
            rate_limit_rpm: Some(120.0),
            ..Default::default()
        };
        let merged = a.merge(b);
        assert_eq!(merged.rate_limit_rpm, Some(120.0));
        assert_eq!(
            merged.cors_allowed_origins,
            Some(vec!["https://a.example".to_string()])
        );
    }

    // ─── SRV-OPENAI: `--enable-ui` / `--max-output-tokens` gain TOML + env ──

    #[test]
    fn ui_and_output_ceiling_default_to_off_and_unset() {
        let cfg = ServerConfig::default();
        assert!(!cfg.ui.enabled);
        assert_eq!(cfg.limits.max_output_tokens, None);
        let p = PartialServerConfig::default();
        assert!(p.enable_ui.is_none());
        assert!(p.max_output_tokens.is_none());
    }

    #[test]
    fn ui_and_output_ceiling_parse_from_toml() {
        let cfg =
            ServerConfig::from_toml("[ui]\nenabled = true\n[limits]\nmax_output_tokens = 2048\n")
                .expect("parse");
        assert!(cfg.ui.enabled);
        assert_eq!(cfg.limits.max_output_tokens, Some(2048));
        // ...and round-trip through the serialized form.
        let text = cfg.to_toml_string().expect("serialize");
        assert_eq!(ServerConfig::from_toml(&text).expect("reparse"), cfg);
    }

    #[test]
    fn a_zero_output_ceiling_is_refused_in_toml_and_env() {
        let err = PartialServerConfig::from_toml_str("[limits]\nmax_output_tokens = 0\n")
            .expect_err("a zero ceiling");
        assert!(err.to_string().contains("max_output_tokens"), "{err}");
        let err = parse_ui_env_map([("OXIBONSAI_MAX_OUTPUT_TOKENS".to_string(), "0".to_string())])
            .expect_err("a zero ceiling");
        assert!(
            err.to_string().contains("OXIBONSAI_MAX_OUTPUT_TOKENS"),
            "{err}"
        );
    }

    #[test]
    fn ui_env_layer_parses_both_variables_and_ignores_others() {
        let partial = parse_ui_env_map([
            ("OXIBONSAI_ENABLE_UI".to_string(), "true".to_string()),
            ("OXIBONSAI_MAX_OUTPUT_TOKENS".to_string(), "512".to_string()),
            ("OXIBONSAI_PORT".to_string(), "not-mine".to_string()),
        ])
        .expect("parse");
        assert_eq!(partial.enable_ui, Some(true));
        assert_eq!(partial.max_output_tokens, Some(512));
        assert_eq!(partial.port, None, "other variables belong to crate::env");
        let off = parse_ui_env_map([("OXIBONSAI_ENABLE_UI".to_string(), "0".to_string())])
            .expect("parse");
        assert_eq!(off.enable_ui, Some(false));
        let err = parse_ui_env_map([("OXIBONSAI_ENABLE_UI".to_string(), "maybe".to_string())])
            .expect_err("not a boolean");
        assert!(err.to_string().contains("OXIBONSAI_ENABLE_UI"), "{err}");
    }

    #[test]
    fn ui_and_output_ceiling_layer_cli_over_env_over_toml() {
        let toml_layer = PartialServerConfig::from_toml_str(
            "[ui]\nenabled = false\n[limits]\nmax_output_tokens = 100\n",
        )
        .expect("toml");
        let env_layer =
            parse_ui_env_map([("OXIBONSAI_MAX_OUTPUT_TOKENS".to_string(), "200".to_string())])
                .expect("env");
        let cli_layer = PartialServerConfig {
            enable_ui: Some(true),
            ..Default::default()
        };
        let cfg = ServerConfig::from_partial(toml_layer.merge(env_layer).merge(cli_layer));
        assert!(cfg.ui.enabled, "the CLI layer wins");
        assert_eq!(cfg.limits.max_output_tokens, Some(200), "env beats TOML");
    }

    // ─── SV-15/SV-16/SV-23: every config field must have a real consumer ───

    /// Exhaustive struct destructure of [`ServerConfig`] and every
    /// sub-section. `#[non_exhaustive]` only blocks *external* exhaustive
    /// matches, so this compiles inside the defining crate; adding a new
    /// top-level or sub-section field without updating this test is a
    /// compile error, which is what makes this stronger than a plain
    /// "does it exist" check. Each binding is annotated with exactly where
    /// it is read outside this crate (finding `SV-15`: "six validated
    /// configuration fields are never read").
    #[test]
    fn every_config_field_has_a_documented_consumer() {
        let cfg = ServerConfig::default();
        let ServerConfig {
            bind,
            model,
            tokenizer,
            sampling,
            limits,
            auth,
            observability,
            cors,
            rate_limit,
            ui,
            seed,
        } = cfg;

        let BindConfig { host: _, port: _ } = bind; // main.rs: resolved into the listener SocketAddr + the sec-15 non-loopback guard.
        let ModelConfig {
            path: _,
            quantization_hint: _,
        } = model; // main.rs: build_pool_from_gguf(path, ...) / quantization_hint diagnostic logging.
        let TokenizerConfigSection { path: _, kind: _ } = tokenizer; // main.rs: TokenizerBridge::from_file / auto-detection + validation.rs kind whitelist.
        let SamplingConfig {
            default_max_tokens: _, // KNOWN GAP: no seam in oxibonsai_runtime::server to thread a per-router default (see deviations); main.rs logs a startup advisory when this diverges from the hardcoded server-side literal.
            default_temperature: _,
            default_top_p: _,
        } = sampling; // main.rs: SamplingParams { temperature, top_p, .. } fed into the engine pool.
        let LimitsConfig {
            max_input_tokens: _,
            max_concurrent_requests: _,
            per_request_timeout_ms: _,
            engine_pool_size: _,
            max_body_bytes: _,
            max_output_tokens: _, // KNOWN GAP (B2-14 deviation): main.rs still passes `cli_args.max_output_tokens` to `RouterBuildOptions::new`; it must pass `config.limits.max_output_tokens` (which already layers the CLI flag over TOML/env once `args.rs::to_partial` sets it).
        } = limits; // main.rs: RequestLimits::with_max_input_tokens/with_timeout_ms, GlobalConcurrencyLimitLayer, DefaultBodyLimit::max.
        let AuthConfig {
            bearer_token: _,
            bearer_token_file: _,
            admin_token: _,
            insecure_no_auth: _,
        } = auth; // main.rs: resolve_bearer_token (SV-33) -> bearer_auth layer state, AuthConfig::with_admin_token (admin router), the bind-safety guard.
        let ObservabilityConfig {
            log_level: _,
            metrics_enabled: _,
            metrics_path: _,
        } = observability; // main.rs: the (now single, post-config) tracing subscriber + the metrics_path_mw gate/rewrite.
        let CorsSection {
            allowed_origins: _,
            allow_credentials: _,
        } = cors; // main.rs: CorsConfig::from_origins(..) -> apply_middleware.
        let RateLimitSection { rpm: _, burst: _ } = rate_limit; // main.rs: rpm/60.0 -> RateLimitConfig.rps, fed to rate_limiter::rate_limit_layer.
        let UiSection { enabled: _ } = ui; // KNOWN GAP (B2-14 deviation): main.rs must pass `config.ui.enabled` instead of `cli_args.enable_ui` to `RouterBuildOptions::new`.
        let _seed: u64 = seed; // main.rs: build_pool_from_gguf(.., seed, ..).
    }
}
