//! OxiBonsai server binary.
//!
//! Layered configuration loader (`defaults < TOML < env < CLI`) followed by
//! model / tokenizer wiring and an Axum-based HTTP server with OpenAI-style
//! chat-completion endpoints.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use axum::error_handling::HandleErrorLayer;
use axum::http::StatusCode;
use axum::{BoxError, Json};
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::GgufTensorType;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_pool::{build_pool_from_gguf, EnginePool};
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{create_router_with_pool, serve_with_shutdown, shutdown_signal};
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;
use oxibonsai_serve::{
    args::parse_args_from,
    banner,
    config::{PartialServerConfig, ServerConfig},
    env::parse_process_env,
};
use tower::ServiceBuilder;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            error!(%err, "oxibonsai-serve startup failed");
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // ── 1. Parse command-line arguments ──────────────────────────────────
    let argv: Vec<String> = std::env::args().collect();
    let cli_args = match parse_args_from(&argv)? {
        Some(a) => a,
        // --help or --version was printed; exit cleanly.
        None => return Ok(()),
    };

    // ── 2. Install a *temporary* tracing subscriber so config / env / CLI
    //       parsing errors show up cleanly.  It will be replaced once the
    //       final `log_level` is known.
    let bootstrap_filter = tracing_subscriber::EnvFilter::try_new(&cli_args.log_level)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(bootstrap_filter)
        .with_target(false)
        .compact()
        .try_init();

    // ── 3. Load layered configuration ────────────────────────────────────
    let toml_path: Option<PathBuf> = cli_args.config_path.as_ref().map(PathBuf::from);
    let env_partial = parse_process_env()?;
    let cli_partial: PartialServerConfig = cli_args.to_partial();

    let config = ServerConfig::load(toml_path.as_deref(), Some(env_partial), Some(cli_partial))?;

    // ── 4. Print banner ───────────────────────────────────────────────────
    banner::print_banner();
    info!(
        "{}",
        banner::startup_message(&config.bind.host, config.bind.port)
    );

    // ── 5. Build inference engine ─────────────────────────────────────────
    //
    // If a GGUF model path is configured we load it eagerly via
    // `InferenceEngine::from_gguf_path`.  Any failure is *fatal* — the
    // operator asked for a specific model, so silently falling back to a
    // tiny test config would be misleading.
    let sampling = SamplingParams {
        temperature: config.sampling.default_temperature,
        top_p: config.sampling.default_top_p,
        ..SamplingParams::default()
    };

    // Engine-pool size precedence: config value (set via TOML / env /
    // OXIBONSAI_ENGINE_POOL_SIZE, all already merged into `config.limits` by the
    // layered loader) if present, else unset. We pass `None` through when unset
    // so the pool builder applies the CPU default of `min(4, cores)`; replicas
    // share one `Arc<[f32]>` token-embedding table, so each extra replica only
    // adds a KV cache. The GPU/Metal tier is clamped back to 1 by the builder.
    let requested_pool_size: Option<usize> = config.limits.engine_pool_size;

    let pool: Arc<EnginePool> = match config.model.path.as_ref() {
        Some(path) => {
            info!(path = %path.display(), "loading GGUF model");
            match build_pool_from_gguf(
                path,
                sampling.clone(),
                config.seed,
                config.limits.max_input_tokens,
                requested_pool_size,
            ) {
                Ok((pool, _tier, size)) => {
                    info!(pool_size = size, "GGUF model loaded");
                    pool
                }
                Err(err) => {
                    error!(
                        path = %path.display(),
                        %err,
                        "failed to load GGUF model"
                    );
                    return Err(format!("failed to load GGUF model: {err}").into());
                }
            }
        }
        None => {
            warn!("no --model path supplied; falling back to tiny_test engine");
            if config.model.quantization_hint.is_some() {
                warn!(
                    "model.quantization_hint is set but no --model path was supplied; \
                     ignoring (there is no GGUF file to associate it with)"
                );
            }
            let tiny = Qwen3Config::tiny_test();
            let engine = InferenceEngine::new(tiny, sampling, config.seed);
            // No GGUF to share across replicas; wrap the single in-memory engine
            // in a 1-element pool (byte-identical to the prior single-engine
            // fallback).
            EnginePool::new(vec![engine])
        }
    };

    // `model.quantization_hint` is documented (`examples/server_config.toml`)
    // as an *informational* label — "the loader picks the real quantization
    // from the file" — so it deliberately does not gate model loading here.
    // What it *must not* do is disappear silently: surface it in the log at
    // `info` level when it looks like a real OxiBonsai/GGUF quant-type name,
    // and `warn` when it looks like a typo, so operators get feedback either
    // way instead of a config value that is parsed, validated and then never
    // read again.
    if let (Some(hint), Some(path)) = (
        config.model.quantization_hint.as_deref(),
        config.model.path.as_ref(),
    ) {
        if quantization_hint_recognized(hint) {
            info!(
                quantization_hint = hint,
                path = %path.display(),
                "quantization hint recorded (informational; actual quantization is \
                 auto-detected from the GGUF file's own tensor metadata)"
            );
        } else {
            warn!(
                quantization_hint = hint,
                known = ?known_quant_type_names(),
                "model.quantization_hint does not resemble any known OxiBonsai/GGUF \
                 quantization type name — likely a typo. This field is informational only \
                 and does not affect model loading."
            );
        }
    }

    // ── 6. Load tokenizer (optional) ──────────────────────────────────────
    //
    // Resolution order:
    //   (a) explicit `config.tokenizer.path` (CLI / TOML / env)
    //   (b) auto-detect alongside the configured model
    //   (c) give up but tell the user *exactly* where we looked and how to fix it
    let tokenizer = match config.tokenizer.path.as_ref() {
        Some(path) => match TokenizerBridge::from_file(&path.display().to_string()) {
            Ok(t) => {
                info!(path = %path.display(), "tokenizer loaded");
                Some(t)
            }
            Err(err) => {
                error!(path = %path.display(), %err, "failed to load tokenizer");
                return Err(format!("failed to load tokenizer: {err}").into());
            }
        },
        None => {
            // Try auto-detection only if we know the model path.  Otherwise
            // there is nothing to derive candidate paths from.
            let lookup = match config.model.path.as_ref() {
                Some(model_path) => tokenizer_lookup::resolve_tokenizer_for_model(model_path),
                None => tokenizer_lookup::TokenizerLookup::default(),
            };
            match lookup.found {
                Some(found) => match TokenizerBridge::from_file(&found) {
                    Ok(t) => {
                        info!(path = %found, "auto-detected tokenizer alongside model");
                        Some(t)
                    }
                    Err(err) => {
                        error!(path = %found, %err, "failed to load auto-detected tokenizer");
                        return Err(format!("failed to load tokenizer: {err}").into());
                    }
                },
                None => {
                    warn!(
                        "{}",
                        tokenizer_lookup::missing_tokenizer_warning(&lookup.searched)
                    );
                    None
                }
            }
        }
    };

    // ── 7. Build router (with optional bearer auth) ───────────────────────
    //
    // A fresh `InferenceMetrics` is created here, matching the previous
    // `create_router(engine, tokenizer)` path (which built one internally). The
    // pool serves the configured number of replicas (default min(4, cores) on
    // CPU, clamped to 1 on GPU/Metal); a 1-element pool is byte-identical to the
    // prior single-engine router.
    let metrics = Arc::new(InferenceMetrics::new());
    let base_router = create_router_with_pool(pool, tokenizer, metrics);
    let router = if let Some(ref token) = config.auth.bearer_token {
        let state = middleware::BearerAuthState {
            token: token.clone(),
        };
        info!("bearer-token authentication enabled");
        base_router.layer(axum::middleware::from_fn_with_state(
            state,
            middleware::bearer_auth,
        ))
    } else {
        base_router
    };

    // ── 7b. Admission control: enforce `limits.max_concurrent_requests` and
    //        `limits.per_request_timeout_ms` ───────────────────────────────
    //
    // Both fields are validated to be ≥ 1 by `ServerConfig::validate` (called
    // inside `ServerConfig::load` above), so unwrapping them into `Duration`
    // / `usize` here is always well-formed.
    //
    // Layer order (outermost first, matching axum's documented
    // `HandleErrorLayer` + `ServiceBuilder` pattern): `HandleErrorLayer` wraps
    // everything below so both kinds of admission failure — an overloaded
    // `load_shed` and an elapsed `timeout` — become proper HTTP responses
    // instead of tearing down the connection (axum requires an `Infallible`
    // error type on the outermost service). `load_shed` turns "concurrency
    // limit reached" from a queue (which a bare concurrency-limit layer would
    // do on its own) into an immediate rejection, matching "bounds HTTP-level
    // admission" from the `limits.max_concurrent_requests` doc comment.
    //
    // `GlobalConcurrencyLimitLayer` (not `ServiceBuilder::concurrency_limit`,
    // i.e. `tower::limit::ConcurrencyLimitLayer`) is required here:
    // `axum::Router::layer` applies the given `Layer` independently to *every
    // registered route* (`PathRouter::layer` calls `layer.clone().layer(..)`
    // once per route, not once for the router as a whole), so a bare
    // `ConcurrencyLimitLayer` — which allocates a fresh `Arc<Semaphore>`
    // inside its own `Layer::layer()` — would silently create one
    // independent semaphore *per route* instead of one shared budget across
    // the whole HTTP surface. `GlobalConcurrencyLimitLayer` pre-builds the
    // `Arc<Semaphore>` once, before any `.layer()` call, and every per-route
    // clone shares that same semaphore, giving the process-wide admission
    // bound `limits.max_concurrent_requests` documents.
    let concurrency_semaphore =
        tower::limit::GlobalConcurrencyLimitLayer::new(config.limits.max_concurrent_requests);
    let admission = ServiceBuilder::new()
        .layer(HandleErrorLayer::new(handle_admission_error))
        .load_shed()
        .layer(concurrency_semaphore)
        .timeout(Duration::from_millis(config.limits.per_request_timeout_ms));
    let router = router.layer(admission);

    // ── 8. Resolve bind address ───────────────────────────────────────────
    let addr_str = format!("{}:{}", config.bind.host, config.bind.port);
    let addr: SocketAddr = addr_str
        .parse()
        .map_err(|e| format!("invalid bind address '{addr_str}': {e}"))?;

    info!(%addr, "starting listener");

    // ── 9. Serve with graceful shutdown ───────────────────────────────────
    serve_with_shutdown(router, addr, shutdown_signal()).await?;

    info!("oxibonsai-serve exited cleanly");
    Ok(())
}

/// Every quantization-type name the current build of OxiBonsai actually
/// knows how to decode, sourced from [`GgufTensorType`] (type IDs 0..=44 is a
/// generous upper bound — `from_id` rejects unregistered IDs) rather than a
/// hand-maintained string list that could drift out of sync with the real
/// enum.
fn known_quant_type_names() -> Vec<&'static str> {
    (0u32..=44)
        .filter_map(|id| GgufTensorType::from_id(id).ok())
        .map(|t| t.name())
        .collect()
}

/// Best-effort sanity check for an operator-supplied `model.quantization_hint`.
///
/// Accepts an exact (case-insensitive) match against a known type name, and
/// also a prefix match in either direction so that a deliberately-abbreviated
/// hint (the shipped example config uses `quantization_hint = "TQ2"` for the
/// `TQ2_0` / `TQ2_0_g128` types) is not flagged as a typo.
fn quantization_hint_recognized(hint: &str) -> bool {
    let hint_upper = hint.to_ascii_uppercase();
    if hint_upper.is_empty() {
        return false;
    }
    known_quant_type_names().into_iter().any(|name| {
        let name_upper = name.to_ascii_uppercase();
        name_upper == hint_upper
            || name_upper.starts_with(&hint_upper)
            || hint_upper.starts_with(&name_upper)
    })
}

/// Convert an admission-layer error (an overloaded `load_shed` or an elapsed
/// `timeout`) into an OpenAI-style JSON error response.
///
/// Required because axum's `Router` demands an `Infallible` error type on the
/// outermost service; `HandleErrorLayer` is the documented bridge from the
/// `tower::BoxError` the admission stack produces back into a `Response`. See
/// <https://docs.rs/axum/latest/axum/error_handling/index.html>.
async fn handle_admission_error(err: BoxError) -> (StatusCode, Json<serde_json::Value>) {
    let (status, kind, message) = if err.is::<tower::load_shed::error::Overloaded>() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded_error",
            "server is at its configured limits.max_concurrent_requests capacity; \
             retry after a short backoff"
                .to_string(),
        )
    } else if err.is::<tower::timeout::error::Elapsed>() {
        (
            StatusCode::REQUEST_TIMEOUT,
            "timeout_error",
            "request exceeded the configured limits.per_request_timeout_ms budget".to_string(),
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            format!("unhandled admission-layer error: {err}"),
        )
    };
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": kind,
                "param": null,
                "code": null,
            }
        })),
    )
}

/// Tokenizer auto-discovery used when the operator does not pass an explicit
/// `--tokenizer` / `tokenizer.path`.  Mirrors the helper in the `oxibonsai`
/// CLI binary so both surfaces present the same searched-paths list and the
/// same "to fix" instructions.
mod tokenizer_lookup {
    use std::path::{Path, PathBuf};

    /// Result of attempting to locate a `tokenizer.json` for a given model.
    #[derive(Debug, Default)]
    pub struct TokenizerLookup {
        /// Resolved tokenizer path (UTF-8 `String` so the existing
        /// `TokenizerBridge::from_file(&str)` API can consume it directly).
        pub found: Option<String>,
        /// Every candidate path that was inspected during auto-detection,
        /// in the order they were probed.
        pub searched: Vec<PathBuf>,
    }

    /// Strip a trailing GGUF quantization suffix (e.g. `-Q2_0`, `-Q4_K_M`,
    /// `-F16`, `-BF16`, `-F32`) from a model basename.
    fn strip_quant_suffix(basename: &str) -> &str {
        let Some(dash_pos) = basename.rfind('-') else {
            return basename;
        };
        let suffix = &basename[dash_pos + 1..];
        if suffix.is_empty() {
            return basename;
        }
        let is_float = matches!(suffix, "F16" | "BF16" | "F32");
        let is_quant = {
            let mut chars = suffix.chars();
            match chars.next() {
                Some('Q') => {
                    let rest: String = chars.collect();
                    if rest.is_empty() {
                        false
                    } else {
                        let mut parts = rest.split('_');
                        let first = parts.next().unwrap_or("");
                        if first.is_empty() || !first.chars().all(|c| c.is_ascii_digit()) {
                            false
                        } else {
                            parts.all(|p| {
                                !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric())
                            })
                        }
                    }
                }
                _ => false,
            }
        };
        if is_float || is_quant {
            &basename[..dash_pos]
        } else {
            basename
        }
    }

    /// Build the ordered list of candidate `tokenizer.json` paths to probe
    /// for a given model file.  Duplicates are removed so the warning stays
    /// compact.
    fn tokenizer_candidates(model_path: &Path) -> Vec<PathBuf> {
        let parent = model_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let mut out: Vec<PathBuf> = Vec::new();
        let push_unique = |p: PathBuf, out: &mut Vec<PathBuf>| {
            if !out.iter().any(|existing| existing == &p) {
                out.push(p);
            }
        };

        push_unique(parent.join("tokenizer.json"), &mut out);
        push_unique(parent.join("..").join("tokenizer.json"), &mut out);

        if let Some(stem) = model_path.file_stem().and_then(|s| s.to_str()) {
            let base = strip_quant_suffix(stem);
            for variant in [
                base.to_string(),
                format!("{base}-unpacked"),
                format!("{base}-ONNX"),
            ] {
                push_unique(parent.join(&variant).join("tokenizer.json"), &mut out);
            }
        }

        for ancestor in model_path.ancestors().skip(1) {
            if ancestor.file_name().and_then(|n| n.to_str()) == Some("models") {
                push_unique(ancestor.join("tokenizer.json"), &mut out);
                break;
            }
        }

        out
    }

    /// Resolve a tokenizer next to a configured model path.
    pub fn resolve_tokenizer_for_model(model_path: &Path) -> TokenizerLookup {
        let candidates = tokenizer_candidates(model_path);
        for candidate in &candidates {
            if candidate.exists() {
                return TokenizerLookup {
                    found: Some(candidate.to_string_lossy().into_owned()),
                    searched: candidates,
                };
            }
        }
        TokenizerLookup {
            found: None,
            searched: candidates,
        }
    }

    /// Build the multi-line "no tokenizer found" warning message.
    pub fn missing_tokenizer_warning(searched: &[PathBuf]) -> String {
        let mut msg = String::from("no tokenizer found. Searched:\n");
        if searched.is_empty() {
            msg.push_str("  (no candidate paths — model path was not provided)\n");
        } else {
            for path in searched {
                msg.push_str(&format!("  - {}\n", path.display()));
            }
        }
        msg.push_str("To fix:\n");
        msg.push_str("  - Pass --tokenizer <path/to/tokenizer.json>, OR\n");
        msg.push_str(
            "  - Run ./scripts/download_tokenizer.sh to fetch the Qwen3 tokenizer to models/tokenizer.json\n",
        );
        msg.push_str("Continuing with raw token IDs in output.");
        msg
    }
}

/// Bearer-auth middleware.
///
/// Kept inline here (rather than in `oxibonsai-runtime`) because auth is a
/// deployment concern of the server binary, not the inference core.
mod middleware {
    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{header, Request, StatusCode};
    use axum::middleware::Next;
    use axum::response::{IntoResponse, Response};
    use axum::Json;

    /// State shared by the bearer-auth middleware.
    #[derive(Debug, Clone)]
    pub struct BearerAuthState {
        /// The expected token.  Any request that does not present exactly this
        /// token in `Authorization: Bearer <token>` is rejected with 401.
        pub token: String,
    }

    /// `axum::middleware::from_fn_with_state` handler.
    pub async fn bearer_auth(
        State(state): State<BearerAuthState>,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        // Allow `/health` and `/metrics` through unauthenticated — they are
        // needed for load balancers and Prometheus scrapers.
        let path = req.uri().path();
        if path == "/health" || path == "/metrics" {
            return next.run(req).await;
        }

        let header_value = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());

        let presented = match header_value.and_then(|h| h.strip_prefix("Bearer ")) {
            Some(tok) => tok.trim(),
            None => {
                return unauthorized("missing or malformed Authorization header").into_response();
            }
        };

        if !constant_time_eq(presented.as_bytes(), state.token.as_bytes()) {
            return unauthorized("invalid bearer token").into_response();
        }

        next.run(req).await
    }

    /// Constant-time byte-string comparison.
    ///
    /// Plain `PartialEq`/`!=` on `&str`/`String` short-circuits on the first
    /// mismatching byte, which leaks a timing signal proportional to the
    /// length of the correct-token prefix a caller has guessed so far — an
    /// attacker who can measure request latency precisely enough could in
    /// principle recover the bearer token byte-by-byte. This walks every byte
    /// of both inputs unconditionally and folds the differences with `|=` so
    /// the number of set bits (and therefore, modulo compiler
    /// vectorization/optimization, the *shape* of the work performed) does
    /// not depend on where the first mismatch occurs.
    ///
    /// Deliberately implemented locally (no `subtle` dependency) per the
    /// no-new-dependency policy — this is the same fixed-time XOR-fold
    /// technique `subtle::ConstantTimeEq` uses internally for byte slices.
    ///
    /// Note: an early return on length mismatch does leak *length* via
    /// timing, not content. This mirrors `subtle`'s own behavior for
    /// differently-sized slices and is standard practice for secret
    /// comparisons — token length is not itself treated as a secret here.
    fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut diff: u8 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            diff |= x ^ y;
        }
        diff == 0
    }

    fn unauthorized(msg: &str) -> (StatusCode, Json<serde_json::Value>) {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": {
                    "message": msg,
                    "type": "auth_error",
                    "param": null,
                    "code": null,
                }
            })),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::constant_time_eq;

        #[test]
        fn equal_bytes_are_equal() {
            assert!(constant_time_eq(b"my-secret-token", b"my-secret-token"));
        }

        #[test]
        fn different_bytes_are_unequal() {
            assert!(!constant_time_eq(b"my-secret-token", b"not-the-token!!"));
        }

        #[test]
        fn different_lengths_are_unequal() {
            assert!(!constant_time_eq(b"short", b"a-much-longer-token"));
            assert!(!constant_time_eq(b"a-much-longer-token", b"short"));
        }

        #[test]
        fn empty_slices_are_equal() {
            assert!(constant_time_eq(b"", b""));
        }

        #[test]
        fn single_byte_difference_at_any_position_is_detected() {
            let base = b"0123456789abcdef";
            for i in 0..base.len() {
                let mut mutated = *base;
                mutated[i] ^= 0xFF;
                assert!(
                    !constant_time_eq(base, &mutated),
                    "failed to detect mismatch at byte {i}"
                );
            }
        }
    }
}

#[cfg(test)]
mod quant_hint_tests {
    use super::{known_quant_type_names, quantization_hint_recognized};

    #[test]
    fn known_quant_type_names_is_non_empty_and_stable_examples_present() {
        let names = known_quant_type_names();
        assert!(names.contains(&"TQ2_0"));
        assert!(names.contains(&"TQ2_0_g128"));
        assert!(names.contains(&"Q1_0_g128"));
        assert!(names.contains(&"F8_E4M3"));
        assert!(names.contains(&"Q4_K"));
    }

    #[test]
    fn exact_case_insensitive_match_is_recognized() {
        assert!(quantization_hint_recognized("Q8_0"));
        assert!(quantization_hint_recognized("q8_0"));
        assert!(quantization_hint_recognized("TQ2_0_g128"));
    }

    #[test]
    fn abbreviated_hint_from_example_config_is_recognized() {
        // `examples/server_config.toml` ships `quantization_hint = "TQ2"` as
        // the canonical example — it must not be flagged as an unrecognized
        // typo.
        assert!(quantization_hint_recognized("TQ2"));
    }

    #[test]
    fn garbage_hint_is_not_recognized() {
        assert!(!quantization_hint_recognized("NOT_A_REAL_QUANT_TYPE"));
        assert!(!quantization_hint_recognized(""));
    }
}
