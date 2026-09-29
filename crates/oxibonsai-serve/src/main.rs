//! OxiBonsai server binary.
//!
//! Layered configuration loader (`defaults < TOML < env < CLI`) followed by
//! model / tokenizer wiring and an Axum-based HTTP server with OpenAI-style
//! chat-completion endpoints.
//!
//! # Hardening (findings `sec-15`/`SV-07`/`sec-M2`, `sec-07`/`RT-34`/`SV-10`,
//! `cli-18`, `SV-06`, `SV-15`/`SV-16`/`SV-23`, `sec-20`/`perf-M1`,
//! `SV-27`/`sec-16`, `SV-24`, `SV-30`/`sec-M3`, `SV-33`, `sec-12`)
//!
//! The bind-safety guard, bearer/admin token resolution, checksum
//! verification, the extra `OXIBONSAI_*` env overrides, and the fully
//! composed router (see [`hardening::build_router`]) live in the sibling
//! [`hardening`] module — split out to keep this file under the workspace's
//! 2000-line-per-file policy.
//!
//! The bind address defaults to loopback ([`oxibonsai_serve::config::BindConfig`]);
//! binding a non-loopback host without a bearer token requires an explicit
//! `--insecure-no-auth` acknowledgement (see [`hardening::bind_safety_check`]).

mod hardening;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::GgufTensorType;
use oxibonsai_runtime::embed_engine::ModelEmbedder;
use oxibonsai_runtime::engine::{Backend, InferenceEngine};
use oxibonsai_runtime::engine_pool::{build_pool_from_gguf_parts, EnginePool, PoolBuild};
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{install_shutdown_signals, serve_with_shutdown};
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;
use oxibonsai_serve::{
    args::parse_args_from,
    banner,
    config::{PartialServerConfig, ServerConfig},
    embedder,
    env::parse_process_env,
    metrics::MetricsRegistry,
    tokenizer_ladder,
};
use tracing::{error, info, warn};

/// The boxed error `run` propagates to `main`.
type StartupError = Box<dyn std::error::Error + Send + Sync>;

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

async fn run() -> Result<(), StartupError> {
    // ── 1. Parse command-line arguments ──────────────────────────────────
    let argv: Vec<String> = std::env::args().collect();
    let cli_args = match parse_args_from(&argv)? {
        Some(a) => a,
        // --help or --version was printed; exit cleanly.
        None => return Ok(()),
    };

    // ── 2. Load layered configuration ────────────────────────────────────
    //
    // No tracing subscriber exists yet at this point (see step 3 below for
    // why that is now deliberate, not merely "not yet installed" -- finding
    // `SV-15`(a)). A failure here propagates to `main()`, which prints it
    // with `eprintln!` regardless of tracing state.
    let toml_path: Option<PathBuf> = cli_args.config_path.as_ref().map(PathBuf::from);
    let mut env_partial = parse_process_env()?;
    hardening::apply_extra_env_overrides(&mut env_partial, std::env::vars());
    let cli_partial: PartialServerConfig = cli_args.to_partial();

    let mut config =
        ServerConfig::load(toml_path.as_deref(), Some(env_partial), Some(cli_partial))?;

    // ── 3. Install the (single, final) tracing subscriber ────────────────
    //
    // Finding `SV-15`(a): the previous code installed a *bootstrap*
    // subscriber here using only `--log-level`/its built-in default, before
    // TOML/env were merged, with a comment promising it would "be replaced
    // once the final log_level is known" -- but `tracing`'s global
    // subscriber can only ever be set once per process, so that promise was
    // never kept and `observability.log_level` set via TOML or
    // `OXIBONSAI_LOG_LEVEL` had zero effect. Nothing above this line emits a
    // `tracing` event (config-loading errors propagate via `?` and are
    // printed by `main()`'s `eprintln!` regardless of subscriber state), so
    // installing the *real*, fully-merged `config.observability.log_level`
    // as the one and only subscriber is a strict improvement with no loss
    // of early diagnostics.
    let filter = tracing_subscriber::EnvFilter::try_new(&config.observability.log_level)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .try_init();

    // ── 4. Print banner ───────────────────────────────────────────────────
    banner::print_banner();
    info!(
        "{}",
        banner::startup_message(&config.bind.host, config.bind.port)
    );

    // sec-06/SV-15(b): `observability.metrics_enabled`/`metrics_path` are
    // read here (see `hardening::metrics_gate_mw` / the metrics-path alias
    // route, mounted in `hardening::build_router`) instead of being
    // validated and discarded.
    if !config.observability.metrics_enabled {
        info!("Prometheus metrics are disabled (observability.metrics_enabled = false)");
    } else if config.observability.metrics_path != "/metrics" {
        info!(
            metrics_path = %config.observability.metrics_path,
            "Prometheus metrics are also served at the configured observability.metrics_path"
        );
    }

    // SV-15(c): `sampling.default_max_tokens` is threaded through
    // `hardening::build_router` -> `RouterOptions::with_default_max_tokens`
    // below, so the per-request fallback a client omitting `max_tokens`
    // gets is this configured value, not a hardcoded literal. This call is
    // now a defense-in-depth trip-wire rather than a "this is broken"
    // report -- see its own doc comment.
    hardening::warn_if_default_max_tokens_diverges(&config.sampling);

    // ── 5. sec-15 / SV-07 / sec-M2: refuse an unsafe bind ────────────────
    //
    // SV-33: resolve the effective bearer token -- `auth.bearer_token_file`
    // (ps-safe) wins over the raw `auth.bearer_token` string when both are
    // set -- and warn loudly when the token reached us via the literal
    // `--bearer-token` command-line flag, which is visible to any local
    // user via `ps` (and lands in shell history).
    if cli_args.bearer_token.is_some() {
        warn!(
            "--bearer-token was passed on the command line, which is visible to other local \
             users via `ps` (and lands in shell history); prefer --bearer-token-file or the \
             OXIBONSAI_BEARER_TOKEN environment variable instead"
        );
    }
    config.auth.bearer_token = hardening::resolve_bearer_token(&config).map_err(|e| {
        error!(%e, "failed to resolve auth.bearer_token_file");
        e
    })?;

    let admin_auth = hardening::resolve_admin_auth(&config);
    hardening::bind_safety_check(&config, admin_auth.admin_enabled())?;
    if config.auth.bearer_token.is_none() {
        warn!(
            host = %config.bind.host,
            "no auth.bearer_token / --bearer-token / OXIBONSAI_BEARER_TOKEN configured: the \
             inference endpoints are unauthenticated on this listener; /admin/* is separately \
             gated (403 unless auth.admin_token / OXI_ADMIN_TOKEN is set)"
        );
    }

    // ── 6. sec-12: verify the model's checksum, if one is known ──────────
    let checksums_path: PathBuf = std::env::var("OXIBONSAI_CHECKSUMS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("scripts/checksums.sha256"));
    if let Some(ref path) = config.model.path {
        if let Err(msg) = hardening::verify_model_checksum(path, &checksums_path) {
            error!(path = %path.display(), %msg, "model checksum verification failed");
            return Err(msg.into());
        }
    }

    // ── 7. Build inference engine ─────────────────────────────────────────
    //
    // If a GGUF model path is configured we load it eagerly via
    // `InferenceEngine::from_gguf_path`. Any failure is *fatal* — the
    // operator asked for a specific model, so silently falling back to a
    // tiny test config would be misleading.
    // `repetition_penalty` is named explicitly
    // here rather than left to `..SamplingParams::default()`'s spread. The
    // `Default` impl is already `1.0` (the same fix), so this is behaviorally
    // a no-op today -- it exists so a *future* change to that default cannot
    // silently reintroduce a hidden penalty at the one place that seeds
    // every engine replica's startup sampler.
    let sampling = SamplingParams {
        temperature: config.sampling.default_temperature,
        top_p: config.sampling.default_top_p,
        repetition_penalty: 1.0,
        ..SamplingParams::default()
    };

    let (pool, built) = load_pool(&config, &sampling)?;

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

    // ── 8. Load tokenizer (optional; TOK-08 vocab-aware) ──────────────────
    let (tokenizer, tokenizer_source) = resolve_serving_tokenizer(built.as_ref(), &config)?;

    // ── 8b. Model-backed `/v1/embeddings` ─────────────────────────────────
    let (embedder, embedder_unavailable) = serve_embedder(
        built.as_ref(),
        tokenizer_source.as_ref(),
        &sampling,
        &config,
    )?;

    // The served engine's resolved variant + effective kernel tier, for
    // `/admin/status` and `/admin/config` — captured from a lease (the
    // engine itself is never exposed by `EnginePool`) rather than a
    // process-wide registration.
    let engine_report = {
        let lease = pool
            .acquire()
            .await
            .map_err(|e| format!("engine pool: {e}"))?;
        oxibonsai_runtime::admin::EngineReport::from_engine(&lease)
    };

    // ── 9. Build the fully-hardened router ────────────────────────────────
    //
    // A fresh `InferenceMetrics` (the runtime's own registry, actually
    // rendered at `/metrics`) is created here, matching the previous
    // `create_router(engine, tokenizer)` path. `serve_metrics` is this
    // crate's own `MetricsRegistry` (finding `SV-24`): previously built and
    // never mounted, now serving real per-route HTTP counters at
    // `/metrics/serve`.
    let metrics = Arc::new(InferenceMetrics::new());
    let serve_metrics = Arc::new(MetricsRegistry::new());
    // sec-20/perf-M1: capture the REAL pool size before `pool` is consumed
    // by `hardening::build_router` -> `create_router_full`.
    let pool_size = pool.size();
    let mut router_build_options = hardening::RouterBuildOptions::new(
        admin_auth,
        pool_size,
        // SV-26 / SV-28: the merged config's own `ui.enabled` /
        // `limits.max_output_tokens` (CLI flag > env > TOML > default,
        // already layered by `ServerConfig::load`) — not the raw CLI-only
        // fields, which would silently ignore a TOML/env-set value.
        config.ui.enabled,
        config.limits.max_output_tokens,
    )
    .with_embedder(embedder)
    .with_engine_report(engine_report);
    if let Some((code, message)) = embedder_unavailable {
        router_build_options = router_build_options.with_embedder_unavailable(code, message);
    }
    let router = hardening::build_router(
        Arc::clone(&pool),
        tokenizer,
        Arc::clone(&metrics),
        Arc::clone(&serve_metrics),
        &config,
        router_build_options,
    );

    // ── 10. Resolve bind address ───────────────────────────────────────────
    let addr_str = format!("{}:{}", config.bind.host, config.bind.port);
    let addr: SocketAddr = addr_str
        .parse()
        .map_err(|e| format!("invalid bind address '{addr_str}': {e}"))?;

    info!(%addr, "starting listener");

    // ── 11. Serve with graceful shutdown ───────────────────────────────────
    //
    // Finding `SV-29`: a failed signal-handler registration now refuses to
    // start instead of degrading (`install_shutdown_signals` propagates the
    // `io::Error` via `?`), rather than falling back to a silently-degraded
    // shutdown path.
    let signals = install_shutdown_signals()?;
    serve_with_shutdown(router, addr, signals).await?;

    info!("oxibonsai-serve exited cleanly");
    Ok(())
}

/// Step 7 of `run`: the engine pool, and — for a GGUF model — the parts of its
/// build the model-backed embedder needs.
///
/// Engine-pool size precedence: config value (set via TOML / env /
/// OXIBONSAI_ENGINE_POOL_SIZE, all already merged into `config.limits` by the
/// layered loader) if present, else unset. `None` is passed through when unset
/// so the pool builder applies the CPU default of `min(4, cores)`; replicas
/// share one `Arc<[f32]>` token-embedding table, so each extra replica only
/// adds a KV cache. The GPU/Metal tier is clamped back to 1 by the builder.
///
/// `build_pool_from_gguf_parts` (not the tuple-returning
/// `build_pool_from_gguf`) keeps the leaked GGUF and the shared
/// token-embedding table ([`PoolBuild`]) the embedder is built from — the same
/// construction `oxibonsai serve` uses, on the auto-selected backend.
fn load_pool(
    config: &ServerConfig,
    sampling: &SamplingParams,
) -> Result<(Arc<EnginePool>, Option<PoolBuild>), StartupError> {
    let requested_pool_size: Option<usize> = config.limits.engine_pool_size;
    match config.model.path.as_ref() {
        Some(path) => {
            info!(path = %path.display(), "loading GGUF model");
            match build_pool_from_gguf_parts(
                path,
                sampling.clone(),
                config.seed,
                config.limits.max_input_tokens,
                requested_pool_size,
                Backend::Auto,
            ) {
                Ok(built) => {
                    info!(pool_size = built.size, "GGUF model loaded");
                    Ok((Arc::clone(&built.pool), Some(built)))
                }
                Err(err) => {
                    error!(
                        path = %path.display(),
                        %err,
                        "failed to load GGUF model"
                    );
                    Err(format!("failed to load GGUF model: {err}").into())
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
            let engine = InferenceEngine::new(tiny, sampling.clone(), config.seed);
            // No GGUF to share across replicas; wrap the single in-memory engine
            // in a 1-element pool (byte-identical to the prior single-engine
            // fallback).
            Ok((EnginePool::new(vec![engine]), None))
        }
    }
}

/// Step 8 of `run`: resolve the serving tokenizer (TOK-08).
///
/// With a real GGUF model, [`tokenizer_ladder::resolve`] is the SAME
/// contract `run`/`chat`/`oxibonsai serve` apply: an explicit
/// `tokenizer.path` (hard error if it does not fit), else an auto-detected
/// candidate that fits, else the GGUF's own embedded vocabulary, else none —
/// resolved ONCE (it logs its own outcome). Without one (the `tiny_test` dev
/// fallback used only when no `--model` was given), there is no vocabulary
/// to check an explicit path against, so it loads as-is.
fn resolve_serving_tokenizer(
    built: Option<&PoolBuild>,
    config: &ServerConfig,
) -> Result<
    (
        Option<TokenizerBridge>,
        Option<tokenizer_ladder::TokenizerSource>,
    ),
    StartupError,
> {
    let Some(built) = built else {
        return Ok(match config.tokenizer.path.as_ref() {
            Some(path) => {
                let tok = load_explicit_tokenizer(path)?;
                info!(path = %path.display(), "tokenizer loaded");
                (Some(tok), None)
            }
            None => (None, None),
        });
    };
    let model_path = config
        .model
        .path
        .as_deref()
        .unwrap_or_else(|| Path::new(""));
    match tokenizer_ladder::resolve(config.tokenizer.path.as_deref(), model_path, built.gguf) {
        Ok(Some((tok, source))) => Ok((Some(tok), Some(source))),
        Ok(None) => {
            let searched = tokenizer_ladder::searched_candidates(model_path);
            let mut msg = String::from("no tokenizer found. Searched:\n");
            for path in &searched {
                msg.push_str(&format!("  - {}\n", path.display()));
            }
            msg.push_str(
                "To fix: pass tokenizer.path (or --tokenizer-path); some endpoints will \
                 answer with raw token ids or refuse text prompts in the meantime.",
            );
            warn!("{msg}");
            Ok((None, None))
        }
        Err(e) => {
            error!(%e, "failed to resolve the serving tokenizer");
            Err(e.into())
        }
    }
}

/// Load a single, unchecked tokenizer instance from an explicit path (the
/// no-GGUF-model fallback only — there is no model vocabulary to check it
/// against). A failure is fatal: the operator configured a file this
/// process cannot read.
///
/// Uses the same Pure-Rust native backend as [`tokenizer_ladder::resolve`]
/// (`TokenizerBridge::native_from_file`, not [`TokenizerBridge::from_file`]):
/// a mixed-backend build (the `hf-tokenizer` feature compiled in) would
/// otherwise route this one load through the HuggingFace `tokenizers`
/// crate, which rejects a `tokenizer.json` whose `pre_tokenizer`/`decoder`
/// omit fields (e.g. `add_prefix_space`) the native backend never required.
fn load_explicit_tokenizer(path: &std::path::Path) -> Result<TokenizerBridge, StartupError> {
    TokenizerBridge::native_from_file(&path.display().to_string()).map_err(|err| {
        error!(path = %path.display(), %err, "failed to load tokenizer");
        format!("failed to load tokenizer: {err}").into()
    })
}

/// Step 8b of `run`: the model-backed embedder `/v1/embeddings` answers
/// from, alongside why there is none.
///
/// A dedicated embedding engine off the pool's own leaked GGUF and shared
/// token-embedding table, exactly as `oxibonsai serve` builds it
/// ([`embedder::build_embedder`]); `None` — the route's honest `501` — without
/// a GGUF model, without a tokenizer, or when construction genuinely fails.
/// `tokenizer_source` is [`resolve_serving_tokenizer`]'s own resolved
/// source, rebuilt quietly into a second instance
/// ([`tokenizer_ladder::rebuild`]) — a `TokenizerBridge` is not `Clone`.
fn serve_embedder(
    built: Option<&PoolBuild>,
    tokenizer_source: Option<&tokenizer_ladder::TokenizerSource>,
    sampling: &SamplingParams,
    config: &ServerConfig,
) -> Result<
    (
        Option<Arc<ModelEmbedder>>,
        Option<embedder::EmbedderUnavailable>,
    ),
    StartupError,
> {
    let Some(built) = built else {
        let reason = "no GGUF model: an embedder needs a real model".to_string();
        info!("{reason}; /v1/embeddings answers 501");
        return Ok((None, Some((None, reason))));
    };
    let embedder_tokenizer = match tokenizer_source {
        Some(source) => Some(tokenizer_ladder::rebuild(source, built.gguf).map_err(|e| {
            error!(error = %e, "failed to rebuild the embedder's own tokenizer instance");
            StartupError::from(e)
        })?),
        None => None,
    };
    Ok(embedder::build_embedder(
        built,
        embedder_tokenizer,
        sampling.clone(),
        config.seed,
        config.limits.max_input_tokens,
    ))
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

/// Bearer-auth middleware.
///
/// Kept inline here (rather than in `oxibonsai-runtime`) because auth is a
/// deployment concern of the server binary, not the inference core. Used by
/// [`hardening::build_router`] via `crate::middleware`.
mod middleware {
    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{header, Method, Request, StatusCode};
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
        // SV-06 (correction): an `OPTIONS` preflight carries no credentials
        // by specification, so it must never be rejected on that basis —
        // checked *before* the `Authorization` lookup. When CORS is also
        // configured this is already unreachable (the outer `cors_mw`
        // short-circuits preflight before this layer ever runs — see
        // `hardening::build_router`), but this keeps the same guarantee
        // even when CORS is disabled entirely.
        if req.method() == Method::OPTIONS {
            return next.run(req).await;
        }

        // `/health`, `/metrics` and `/metrics/serve` are needed for load
        // balancers and Prometheus scrapers; `/ui/health` is the chat web
        // UI's own liveness probe (finding SV-06's correction (b)).
        let path = req.uri().path();
        if matches!(
            path,
            "/health" | "/metrics" | "/metrics/serve" | "/ui/health"
        ) {
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

/// Exercises `/v1/embeddings` through the binary's own composition: a
/// [`ServerConfig`] naming a (synthetic, weighted, dense) GGUF and a
/// tokenizer, then exactly `run`'s steps — [`load_pool`],
/// [`resolve_serving_tokenizer`], [`serve_embedder`] and
/// `hardening::build_router` with the embedder in its
/// `RouterBuildOptions` — so the forced wiring through the binary-private
/// router composition is what answers `/v1/embeddings`.
///
/// The fixture itself is the shared
/// `oxibonsai_testkit::dense_fixture` — this module used to hand-roll a
/// byte-for-byte near-duplicate of the copy in
/// `tests/server_integration_tests.rs`.
#[cfg(test)]
mod embeddings_route_tests {
    use super::*;
    use oxibonsai_testkit::dense_fixture::{
        byte_tokenizer_json, weighted_dense_gguf, write_atomic, HIDDEN,
    };
    use tower::ServiceExt;

    /// A private directory under the system temp dir holding the model (and,
    /// when asked, the tokenizer), plus the config naming them — kept alive
    /// by the returned guard. The model sits one level down, so every
    /// candidate the tokenizer auto-detection probes next to it (its own
    /// directory and that directory's parent) is inside this private tree.
    ///
    /// Both files are written atomically (`write_atomic`, a temp name then
    /// `rename`): `std::fs::write` alone is not atomic, and a reader racing
    /// a torn write would see a partial file. A `missing field
    /// add_prefix_space` deserialize failure observed under a loaded,
    /// workspace-wide `--all-features` run had a different cause — feature
    /// unification turns on the HuggingFace `tokenizers` backend, which
    /// [`tokenizer_ladder::resolve`] now avoids by always using
    /// `TokenizerBridge::native_from_file` — but this atomic write stays:
    /// it is still the correct way to publish a fixture file to a path a
    /// concurrent test process may probe.
    fn served_config(with_tokenizer: bool) -> (ServerConfig, tempfile::TempDir) {
        let dir = tempfile::Builder::new()
            .prefix("oxibonsai-serve-bin-embeddings-")
            .tempdir_in(std::env::temp_dir())
            .expect("create the temp dir");
        let model_dir = dir.path().join("weights");
        std::fs::create_dir(&model_dir).expect("create the model dir");
        let model = model_dir.join("model.gguf");
        write_atomic(&model, &weighted_dense_gguf()).expect("write the GGUF");
        let mut config = ServerConfig::default();
        config.model.path = Some(model);
        config.limits.max_input_tokens = 256;
        config.limits.engine_pool_size = Some(1);
        if with_tokenizer {
            let tokenizer = dir.path().join("byte-tokenizer.json");
            write_atomic(&tokenizer, byte_tokenizer_json().as_bytes())
                .expect("write the tokenizer");
            config.tokenizer.path = Some(tokenizer);
        }
        (config, dir)
    }

    /// `run`'s steps 7-9 over `config`.
    fn served_router(config: &ServerConfig) -> axum::Router {
        let sampling = SamplingParams::default();
        let (pool, built) = load_pool(config, &sampling).expect("the pool builds");
        let (tokenizer, tokenizer_source) =
            resolve_serving_tokenizer(built.as_ref(), config).expect("the tokenizer step");
        let (embedder, _reason) =
            serve_embedder(built.as_ref(), tokenizer_source.as_ref(), &sampling, config)
                .expect("the embedder step");
        let pool_size = pool.size();
        hardening::build_router(
            pool,
            tokenizer,
            Arc::new(InferenceMetrics::new()),
            Arc::new(MetricsRegistry::new()),
            config,
            hardening::RouterBuildOptions::new(
                hardening::resolve_admin_auth(config),
                pool_size,
                false,
                None,
            )
            .with_embedder(embedder),
        )
    }

    async fn post_embeddings(app: axum::Router, input: &str) -> (u16, serde_json::Value) {
        let req = axum::http::Request::post("/v1/embeddings")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::json!({ "input": input }).to_string(),
            ))
            .expect("build the request");
        let resp = app.oneshot(req).await.expect("response");
        let status = resp.status().as_u16();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body");
        (status, serde_json::from_slice(&bytes).unwrap_or_default())
    }

    #[tokio::test]
    async fn the_binary_serves_v1_embeddings_from_its_gguf_model() {
        let (config, _dir) = served_config(true);
        let (status, json) = post_embeddings(served_router(&config), "king").await;
        assert_eq!(status, 200, "{json}");
        assert_eq!(json["model"], "bonsai-embeddings-model", "{json}");
        assert_eq!(json["dimension"].as_u64(), Some(HIDDEN as u64), "{json}");
        assert_eq!(json["usage"]["prompt_tokens"].as_u64(), Some(4), "{json}");
        let norm = json["data"][0]["embedding"]
            .as_array()
            .map(|v| {
                v.iter()
                    .filter_map(serde_json::Value::as_f64)
                    .map(|x| x * x)
                    .sum::<f64>()
            })
            .unwrap_or_default()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "unit length, got {norm}");
    }

    /// Without a tokenizer (none configured, none next to the model) and
    /// without a GGUF model at all, the binary keeps the honest `501`.
    #[tokio::test]
    async fn the_binary_without_an_embedder_keeps_the_honest_501() {
        let (config, _dir) = served_config(false);
        let (status, json) = post_embeddings(served_router(&config), "king").await;
        assert_eq!(status, 501, "no tokenizer: {json}");

        let model_less = ServerConfig::default();
        let (status, json) = post_embeddings(served_router(&model_less), "king").await;
        assert_eq!(status, 501, "no GGUF model: {json}");
    }
}
