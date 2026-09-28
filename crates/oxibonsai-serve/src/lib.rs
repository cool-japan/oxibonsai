//! Library interface for oxibonsai-serve.
//!
//! Exposes the argument-parsing, banner, configuration, environment,
//! validation, and metrics modules so they can be exercised from integration
//! tests without going through `main`.
//!
//! The four "uplift" modules added for the Alpha→Stable milestone are:
//!
//! - [`config`]     — layered configuration (`defaults < TOML < env < CLI`)
//! - [`mod@env`]    — `OXIBONSAI_*` environment-variable parsing
//! - [`validation`] — invariants over a fully-merged [`config::ServerConfig`]
//! - [`metrics`]    — hand-rolled Prometheus text-exposition registry
//!
//! [`embedder`] builds the model-backed `/v1/embeddings` embedder the binary
//! hands its router, exactly as `oxibonsai serve` does.

pub mod args;
pub mod banner;
pub mod config;
pub mod embedder;
pub mod env;
pub mod metrics;
pub mod validation;

pub use args::{ParseError, ServerArgs};
pub use config::{
    AuthConfig, BindConfig, ConfigError, LimitsConfig, ModelConfig, ObservabilityConfig,
    PartialServerConfig, SamplingConfig, ServerConfig, TokenizerConfigSection,
};
pub use env::{parse_env_map, parse_process_env};
pub use metrics::{MetricsRegistry, DEFAULT_HISTOGRAM_BUCKETS};
pub use validation::{MAX_DEFAULT_MAX_TOKENS, MIN_BEARER_TOKEN_LEN, VALID_LOG_LEVELS};
