//! Configuration validation.
//!
//! A [`crate::config::ServerConfig`] is considered well-formed when it passes
//! [`ServerConfig::validate`].  The rules are:
//!
//! | Field                                      | Rule                                   |
//! |--------------------------------------------|----------------------------------------|
//! | `bind.port`                                | `1..=65535`                            |
//! | `sampling.default_max_tokens`              | `1..=8192`                             |
//! | `sampling.default_temperature`             | `0.0..=2.0` (and finite)               |
//! | `sampling.default_top_p`                   | `0.0..=1.0` (and finite)               |
//! | `observability.log_level`                  | ∈ { error, warn, info, debug, trace, off } |
//! | `model.path`                               | Must exist on disk if set              |
//! | `tokenizer.path`                           | Must exist on disk if set              |
//! | `tokenizer.kind`                           | ∈ [`VALID_TOKENIZER_KINDS`] if set     |
//! | `auth.bearer_token`                        | ≥ 16 characters if set                 |
//! | `observability.metrics_path`               | Non-empty; starts with `/`; not a built-in route or `/admin/*`; no `{`/`}` |
//! | `limits.max_concurrent_requests`           | ≥ 1                                    |
//! | `limits.per_request_timeout_ms`            | ≥ 1                                    |
//! | `limits.max_input_tokens`                  | ≥ 1                                    |
//! | `limits.max_body_bytes`                    | ≥ 1                                    |
//! | `auth.admin_token`                         | ≥ 16 characters if set                 |
//! | `cors.allowed_origins` + `allow_credentials` | never `["*"]` with credentials on    |
//! | `rate_limit.rpm`                           | finite and > 0 if set                  |
//! | `rate_limit.burst`                         | finite and > 0                         |

use crate::config::{ConfigError, ServerConfig};

/// Whitelist of accepted `log_level` values.
pub const VALID_LOG_LEVELS: &[&str] = &["error", "warn", "info", "debug", "trace", "off"];

/// Minimum bearer-token (and admin-token) length, in UTF-8 bytes.
pub const MIN_BEARER_TOKEN_LEN: usize = 16;

/// Upper bound on `default_max_tokens`.
pub const MAX_DEFAULT_MAX_TOKENS: usize = 8192;

/// Every path `hardening::build_router` / `create_router_full` mount
/// unconditionally, sourced from `oxibonsai_runtime::server::create_router_full`
/// (chat/completions/models/health/metrics), `embeddings::create_embeddings_router`,
/// `web_ui::create_ui_router`, `admin::create_admin_router`, and this
/// crate's own `/metrics/serve` (finding `SV-24`). A non-default
/// `observability.metrics_path` that collides with one of these would make
/// `axum::Router::route` panic at startup (`Router::route` refuses a
/// duplicate path+method registration) — see [`ServerConfig::validate`]'s
/// metrics-path check. `/admin/*` is additionally checked by prefix below
/// since that surface can grow new endpoints independently of this list.
pub const RESERVED_ROUTE_PATHS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/chat/completions/extended",
    "/v1/completions",
    "/v1/models",
    "/v1/embeddings",
    "/health",
    "/readyz",
    "/metrics",
    "/metrics/serve",
    "/ui",
    "/ui/health",
];

// Deliberately NOT listed here: `/v1/models/{model}` (`SV-21`). Unlike every
// other entry above, that route's tail segment is client-supplied (a model
// id), so labelling raw request paths by exact string match would let an
// attacker mint unbounded metric-series cardinality just by requesting
// `/v1/models/<anything>` repeatedly -- exactly the unbounded-growth vector
// `record_http_metrics`'s exact-match design exists to close. It falls
// through to the `"other"` bucket like any other unmatched path, which is
// the safe behavior here, not an oversight.

/// Validate the three "serve hardening" parameters that both `oxibonsai-serve`
/// (via [`ServerConfig::validate`]) and the `oxibonsai serve` CLI subcommand
/// must reject identically (findings `SV-30` / `sec-M3`): an unset-or-short
/// bearer token is fine (auth is optional), but a *configured* token must
/// meet the minimum length, and the two admission knobs must be positive —
/// `max_concurrent_requests == 0` builds a zero-slot budget that sheds
/// every request forever, and `request_timeout_ms == 0` fires the timeout
/// before any handler can complete.
///
/// The single canonical implementation now lives in
/// `oxibonsai_runtime::serve_shared` (both crates already depend on
/// `oxibonsai-runtime` for other things, e.g. `hardening.rs`'s router and
/// `src/cli/admission.rs`'s middleware, so this needs no new dependency);
/// `src/cli/admission.rs::validate_serve_args` re-exports the very same
/// function under its historical name. This re-export keeps
/// `ServerConfig::validate` below, and this crate's own tests, unchanged.
pub use oxibonsai_runtime::serve_shared::validate_serve_params;

/// Whitelist of `tokenizer.kind` values this build can actually honor.
///
/// `TokenizerBridge::from_file` (in `oxibonsai-runtime`) always loads the file
/// through the HuggingFace `tokenizers` crate — there is no alternate backend
/// selectable by any code path. Rather than silently accepting (and ignoring)
/// a value such as `"oxitok"` that the doc comment on
/// [`crate::config::TokenizerConfigSection::kind`] gestures at but which is
/// not actually implemented, `validate` rejects anything outside this list so
/// the mismatch surfaces at startup instead of as a confusing runtime
/// behavior gap.
pub const VALID_TOKENIZER_KINDS: &[&str] = &["huggingface", "hf"];

impl ServerConfig {
    /// Validate the configuration, returning a [`ConfigError::Validation`] on
    /// the first rule that fails.
    ///
    /// Rules are deliberately conservative — unusual values (e.g. `port=0`)
    /// trip validation and force the operator to think.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // ─── Bind ────────────────────────────────────────────────────────
        if self.bind.port == 0 {
            return Err(ConfigError::Validation(
                "bind.port must be in [1, 65535]".to_string(),
            ));
        }

        // ─── Sampling ────────────────────────────────────────────────────
        if self.sampling.default_max_tokens == 0
            || self.sampling.default_max_tokens > MAX_DEFAULT_MAX_TOKENS
        {
            return Err(ConfigError::Validation(format!(
                "sampling.default_max_tokens must be in [1, {MAX_DEFAULT_MAX_TOKENS}], got {}",
                self.sampling.default_max_tokens
            )));
        }
        if !self.sampling.default_temperature.is_finite()
            || self.sampling.default_temperature < 0.0
            || self.sampling.default_temperature > 2.0
        {
            return Err(ConfigError::Validation(format!(
                "sampling.default_temperature must be in [0, 2], got {}",
                self.sampling.default_temperature
            )));
        }
        if !self.sampling.default_top_p.is_finite()
            || self.sampling.default_top_p < 0.0
            || self.sampling.default_top_p > 1.0
        {
            return Err(ConfigError::Validation(format!(
                "sampling.default_top_p must be in [0, 1], got {}",
                self.sampling.default_top_p
            )));
        }

        // ─── Observability ───────────────────────────────────────────────
        if !VALID_LOG_LEVELS
            .iter()
            .any(|l| l.eq_ignore_ascii_case(&self.observability.log_level))
        {
            return Err(ConfigError::Validation(format!(
                "observability.log_level must be one of {VALID_LOG_LEVELS:?}, got {:?}",
                self.observability.log_level
            )));
        }
        if self.observability.metrics_path.is_empty()
            || !self.observability.metrics_path.starts_with('/')
        {
            return Err(ConfigError::Validation(format!(
                "observability.metrics_path must be an absolute HTTP path, got {:?}",
                self.observability.metrics_path
            )));
        }
        // `main.rs`'s `hardening::build_router` mounts a *second* route at
        // `observability.metrics_path` (when it differs from the default
        // `/metrics`) alongside every other route this binary already
        // serves. `axum::Router::route` panics at startup if the same path
        // is registered twice, so a colliding value here must be rejected
        // here -- with a clear config error -- rather than reaching that
        // panic. `/metrics` itself is exempt: it is the default value, and
        // `build_router` skips mounting an alias for it.
        if self.observability.metrics_path != "/metrics" {
            let path = self.observability.metrics_path.as_str();
            let is_admin_path = path == "/admin" || path.starts_with("/admin/");
            if is_admin_path || RESERVED_ROUTE_PATHS.contains(&path) {
                return Err(ConfigError::Validation(format!(
                    "observability.metrics_path {path:?} collides with a built-in route -- \
                     choose a different path"
                )));
            }
            if path.contains(['{', '}']) {
                return Err(ConfigError::Validation(format!(
                    "observability.metrics_path {path:?} must not contain '{{' or '}}' (axum \
                     path-parameter syntax); use a plain static path"
                )));
            }
        }

        // ─── Limits + auth (shared canonical checks, SV-30/sec-M3) ────────
        //
        // Delegates the three checks `src/cli/admission.rs::validate_serve_args`
        // parallels to the single canonical `validate_serve_params` above, so
        // this crate has exactly one implementation of "what a valid bearer
        // token / concurrency limit / timeout looks like".
        validate_serve_params(
            self.auth.bearer_token.as_deref(),
            self.limits.max_concurrent_requests,
            self.limits.per_request_timeout_ms,
        )
        .map_err(|msg| ConfigError::Validation(format!("auth/limits: {msg}")))?;

        if self.limits.max_input_tokens == 0 {
            return Err(ConfigError::Validation(
                "limits.max_input_tokens must be ≥ 1".to_string(),
            ));
        }
        if self.limits.max_body_bytes == 0 {
            return Err(ConfigError::Validation(
                "limits.max_body_bytes must be ≥ 1".to_string(),
            ));
        }

        // ─── Auth (admin token) ──────────────────────────────────────────
        if let Some(ref tok) = self.auth.admin_token {
            if tok.len() < MIN_BEARER_TOKEN_LEN {
                return Err(ConfigError::Validation(format!(
                    "auth.admin_token must be at least {MIN_BEARER_TOKEN_LEN} chars"
                )));
            }
        }

        // ─── CORS (finding sec-09 / SV-20) ───────
        //
        // `oxibonsai_runtime::middleware::CorsConfig::is_unrestricted_wildcard`
        // already makes this combination impossible to *emit* at the HTTP
        // layer (it echoes the exact origin instead of `*` once credentials
        // are on), but a config that asks for it is still an operator
        // mistake worth catching at startup rather than silently
        // reinterpreting.
        if self.cors.allow_credentials && self.cors.allowed_origins.iter().any(|o| o == "*") {
            return Err(ConfigError::Validation(
                "cors.allow_credentials cannot be combined with a literal \"*\" in \
                 cors.allowed_origins: list the exact origin(s) that need credentialed \
                 access instead"
                    .to_string(),
            ));
        }

        // ─── Rate limit ──────────────────────────────────────────────────
        if let Some(rpm) = self.rate_limit.rpm {
            if !rpm.is_finite() || rpm <= 0.0 {
                return Err(ConfigError::Validation(format!(
                    "rate_limit.rpm must be a finite number > 0 when set, got {rpm}"
                )));
            }
        }
        if !self.rate_limit.burst.is_finite() || self.rate_limit.burst <= 0.0 {
            return Err(ConfigError::Validation(format!(
                "rate_limit.burst must be a finite number > 0, got {}",
                self.rate_limit.burst
            )));
        }

        // ─── Paths (existence) ───────────────────────────────────────────
        if let Some(ref path) = self.model.path {
            if !path.exists() {
                return Err(ConfigError::Validation(format!(
                    "model.path does not exist: {}",
                    path.display()
                )));
            }
        }
        if let Some(ref path) = self.tokenizer.path {
            if !path.exists() {
                return Err(ConfigError::Validation(format!(
                    "tokenizer.path does not exist: {}",
                    path.display()
                )));
            }
        }
        // SV-33: `auth.bearer_token_file` names a file `main.rs` reads at
        // startup; fail fast at config-validation time rather than with a
        // confusing I/O error deep inside `resolve_bearer_token`.
        if let Some(ref path) = self.auth.bearer_token_file {
            if !path.exists() {
                return Err(ConfigError::Validation(format!(
                    "auth.bearer_token_file does not exist: {}",
                    path.display()
                )));
            }
        }

        // ─── Tokenizer kind ──────────────────────────────────────────────
        if let Some(ref kind) = self.tokenizer.kind {
            if !VALID_TOKENIZER_KINDS
                .iter()
                .any(|k| k.eq_ignore_ascii_case(kind))
            {
                return Err(ConfigError::Validation(format!(
                    "tokenizer.kind {kind:?} is not supported by this build (no such backend \
                     is implemented); expected one of {VALID_TOKENIZER_KINDS:?}"
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let cfg = ServerConfig::default();
        cfg.validate().expect("defaults should validate");
    }

    #[test]
    fn port_zero_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.bind.port = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn bad_log_level_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.observability.log_level = "loud".to_string();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn bad_top_p_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.sampling.default_top_p = 1.5;
        assert!(cfg.validate().is_err());
    }

    // ─── metrics_path route-collision guard (startup-panic prevention) ────

    #[test]
    fn metrics_path_colliding_with_a_reserved_route_is_rejected() {
        // `/metrics` itself is deliberately exempt (it is the default and
        // never gets an alias route) -- see
        // `metrics_path_of_literal_metrics_is_accepted_despite_being_reserved`.
        for reserved in RESERVED_ROUTE_PATHS.iter().filter(|&&p| p != "/metrics") {
            let mut cfg = ServerConfig::default();
            cfg.observability.metrics_path = (*reserved).to_string();
            let err = cfg
                .validate()
                .expect_err(&format!("{reserved} must be rejected as metrics_path"));
            match err {
                ConfigError::Validation(msg) => assert!(
                    msg.contains("metrics_path"),
                    "error should mention metrics_path: {msg}"
                ),
                other => panic!("expected Validation, got {other:?}"),
            }
        }
    }

    #[test]
    fn metrics_path_of_literal_metrics_is_accepted_despite_being_reserved() {
        // The default value itself must remain valid -- `build_router` skips
        // mounting an alias route when it equals "/metrics" exactly.
        let cfg = ServerConfig::default();
        assert_eq!(cfg.observability.metrics_path, "/metrics");
        cfg.validate()
            .expect("the default metrics_path must validate");
    }

    #[test]
    fn metrics_path_under_admin_prefix_is_rejected() {
        for candidate in ["/admin", "/admin/status", "/admin/anything-future"] {
            let mut cfg = ServerConfig::default();
            cfg.observability.metrics_path = candidate.to_string();
            assert!(
                cfg.validate().is_err(),
                "{candidate} must be rejected (collides with /admin/*)"
            );
        }
    }

    #[test]
    fn metrics_path_with_axum_path_param_syntax_is_rejected() {
        for candidate in ["/metrics/{id}", "/{wildcard}"] {
            let mut cfg = ServerConfig::default();
            cfg.observability.metrics_path = candidate.to_string();
            let err = cfg
                .validate()
                .expect_err(&format!("{candidate} must be rejected"));
            match err {
                ConfigError::Validation(msg) => assert!(msg.contains('{') || msg.contains("path-")),
                other => panic!("expected Validation, got {other:?}"),
            }
        }
    }

    #[test]
    fn non_colliding_custom_metrics_path_is_accepted() {
        let mut cfg = ServerConfig::default();
        cfg.observability.metrics_path = "/custom-metrics".to_string();
        cfg.validate()
            .expect("a non-colliding custom path should validate");
    }

    #[test]
    fn short_bearer_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.auth.bearer_token = Some("short".to_string());
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn unsupported_tokenizer_kind_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.tokenizer.kind = Some("oxitok".to_string());
        let err = cfg
            .validate()
            .expect_err("unsupported kind must be rejected");
        match err {
            ConfigError::Validation(msg) => assert!(
                msg.contains("tokenizer.kind"),
                "error should mention tokenizer.kind, got: {msg}"
            ),
            other => panic!("expected ConfigError::Validation, got {other:?}"),
        }
    }

    #[test]
    fn supported_tokenizer_kind_accepted() {
        let mut cfg = ServerConfig::default();
        cfg.tokenizer.kind = Some("huggingface".to_string());
        cfg.validate().expect("huggingface kind should validate");

        // Case-insensitive and the short alias both accepted.
        cfg.tokenizer.kind = Some("HuggingFace".to_string());
        cfg.validate()
            .expect("case-insensitive kind should validate");

        cfg.tokenizer.kind = Some("hf".to_string());
        cfg.validate().expect("hf alias should validate");
    }

    // ─── validate_serve_params (SV-30 / sec-M3 canonical checks) ───────────

    #[test]
    fn validate_serve_params_accepts_sane_defaults() {
        assert!(validate_serve_params(None, 32, 60_000).is_ok());
        assert!(validate_serve_params(Some(&"x".repeat(MIN_BEARER_TOKEN_LEN)), 1, 1).is_ok());
    }

    #[test]
    fn validate_serve_params_rejects_short_token() {
        let err = validate_serve_params(Some("short"), 32, 60_000).expect_err("must reject");
        assert!(err.contains("bearer token"));
    }

    #[test]
    fn validate_serve_params_rejects_zero_concurrency() {
        let err = validate_serve_params(None, 0, 60_000).expect_err("must reject");
        assert!(err.contains("max_concurrent_requests"));
    }

    #[test]
    fn validate_serve_params_rejects_zero_timeout() {
        let err = validate_serve_params(None, 32, 0).expect_err("must reject");
        assert!(err.contains("request_timeout_ms"));
    }

    // ─── New ServerConfig fields ────────────────────────────────────────────

    #[test]
    fn zero_max_body_bytes_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.limits.max_body_bytes = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn short_admin_token_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.auth.admin_token = Some("short".to_string());
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn long_admin_token_accepted() {
        let mut cfg = ServerConfig::default();
        cfg.auth.admin_token = Some("x".repeat(MIN_BEARER_TOKEN_LEN));
        cfg.validate().expect("long admin token should validate");
    }

    #[test]
    fn wildcard_cors_with_credentials_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.cors.allowed_origins = vec!["*".to_string()];
        cfg.cors.allow_credentials = true;
        let err = cfg
            .validate()
            .expect_err("must reject wildcard + credentials");
        match err {
            ConfigError::Validation(msg) => assert!(msg.contains("allow_credentials")),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn wildcard_cors_without_credentials_is_accepted() {
        let mut cfg = ServerConfig::default();
        cfg.cors.allowed_origins = vec!["*".to_string()];
        cfg.cors.allow_credentials = false;
        cfg.validate()
            .expect("wildcard without credentials is fine");
    }

    #[test]
    fn specific_origin_with_credentials_is_accepted() {
        let mut cfg = ServerConfig::default();
        cfg.cors.allowed_origins = vec!["https://app.example.com".to_string()];
        cfg.cors.allow_credentials = true;
        cfg.validate()
            .expect("a specific origin may be combined with credentials");
    }

    #[test]
    fn non_finite_or_non_positive_rpm_rejected() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut cfg = ServerConfig::default();
            cfg.rate_limit.rpm = Some(bad);
            assert!(cfg.validate().is_err(), "rpm={bad} should be rejected");
        }
    }

    #[test]
    fn positive_rpm_accepted() {
        let mut cfg = ServerConfig::default();
        cfg.rate_limit.rpm = Some(600.0);
        cfg.validate().expect("positive finite rpm should validate");
    }

    #[test]
    fn non_positive_burst_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.rate_limit.burst = 0.0;
        assert!(cfg.validate().is_err());
    }

    // ─── SV-33: bearer_token_file ───────────────────────────────────────────

    #[test]
    fn missing_bearer_token_file_rejected() {
        let mut cfg = ServerConfig::default();
        cfg.auth.bearer_token_file =
            Some(std::env::temp_dir().join("oxibonsai_serve_definitely_missing_token_file"));
        let err = cfg.validate().expect_err("must reject a missing file");
        match err {
            ConfigError::Validation(msg) => assert!(msg.contains("bearer_token_file")),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn existing_bearer_token_file_accepted() {
        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_serve_validation_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("bearer-token");
        std::fs::write(&path, "a-real-secret-token-value\n").expect("write token file");

        let mut cfg = ServerConfig::default();
        cfg.auth.bearer_token_file = Some(path);
        cfg.validate().expect("an existing file should validate");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unset_tokenizer_kind_accepted() {
        let cfg = ServerConfig::default();
        assert!(cfg.tokenizer.kind.is_none());
        cfg.validate().expect("unset kind should validate");
    }
}
