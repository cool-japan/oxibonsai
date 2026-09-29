//! `oxibonsai serve` — OpenAI-compatible HTTP API server.
//!
//! Mirrors the hardening the standalone `oxibonsai-serve` binary applies
//! (`crates/oxibonsai-serve/src/hardening.rs`) with the same layer order —
//! routes → admission (concurrency + timeout) → `DefaultBodyLimit` →
//! `Content-Length` precheck → rate limit → bearer auth → CORS — and exposes
//! every hardening knob as a flag (each wins over its `OXIBONSAI_*`
//! environment variable; see `oxibonsai serve --help`).
//!
//! Model-dependent behaviour is resolved once the GGUF is parsed:
//!
//! * the context window defaults to 8192 for a Bonsai 2 `qwen35` model
//!   (4096 otherwise) and is refused above the model's own limit or the
//!   RAM-derived one (design §5.6 / Appendix A.3);
//! * the pool's baseline sampling parameters are the model's own
//!   `general.sampling.*` defaults (RT-17; Bonsai 2: 1.0 / 0.95 / 20), each
//!   per-request field overriding them;
//! * the tokenizer goes through the same vocab-aware ladder and hard
//!   compatibility check as `run` (TOK-08), with the GGUF's own chat
//!   template attached — resolved ONCE for the
//!   whole startup, however many consumers (router, embedder, RAG) need
//!   their own instance ([`resolve_all_serving_tokenizers`]);
//! * the model's declared `general.sampling.min_p` becomes every replica's
//!   sampling baseline; a request's own `min_p` overrides it for that
//!   request only;
//! * `--think`/`--no-think`, `--reasoning-effort` and `--tools` become the
//!   server-wide defaults a chat request inherits when it carries none of
//!   its own ([`inject_chat_defaults`]).

use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Router;
use oxibonsai_runtime::middleware::{apply_middleware, CorsConfig, MiddlewareConfig};
use oxibonsai_runtime::rate_limiter::{rate_limit_layer, RateLimitConfig};
use oxibonsai_runtime::serve_shared::verify_model_checksum;
use oxibonsai_runtime::server::{
    create_router_full, install_shutdown_signals, serve_with_shutdown,
    AuthConfig as AdminAuthConfig, RequestLimits, RouterOptions,
};

use super::admission;
use super::args::EmbeddingBackendChoice;
use super::bonsai2;
use super::generate::ChatContract;
use super::model_desc;
use super::model_source::ModelSource;
use super::util::{build_sampling_params, missing_tokenizer_warning};

// ─── RT-17: the pool's baseline sampling parameters ────────────────────────
//
// The pre-RT-17 literals (`run`/`chat`'s own final fallbacks) apply only
// when the model declares no `general.sampling.*` value; `repetition_penalty`
// is always the no-op 1.0 (never the hidden non-1.0 value
// an old `SamplingParams::default()` carried).

/// The baseline `SamplingParams` the engine pool is built with, before any
/// per-request JSON-body override: the model's own `general.sampling.*`
/// defaults (RT-17), else the shared `util::DEFAULT_*` literals — through the
/// SAME shared `build_sampling_params` constructor `run`/`chat` use.
fn baseline_sampling_params(
    declared: &oxibonsai_runtime::sampling::GgufSamplingDefaults,
) -> oxibonsai_runtime::sampling::SamplingParams {
    use super::util::{
        DEFAULT_REPETITION_PENALTY, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P,
    };
    use oxibonsai_runtime::sampling::{
        resolve_sampling_default_f32, resolve_sampling_default_usize,
    };
    build_sampling_params(
        resolve_sampling_default_f32(None, declared.temperature, DEFAULT_TEMPERATURE),
        resolve_sampling_default_usize(None, declared.top_k, DEFAULT_TOP_K),
        resolve_sampling_default_f32(None, declared.top_p, DEFAULT_TOP_P),
        DEFAULT_REPETITION_PENALTY,
    )
}

/// The baseline for a model that declares no sampling defaults at all
/// (exactly what `run`/`chat` resolve to with no flag, no config value and
/// no `general.sampling.*` metadata).
#[cfg(test)]
fn default_sampling_params() -> oxibonsai_runtime::sampling::SamplingParams {
    baseline_sampling_params(&oxibonsai_runtime::sampling::GgufSamplingDefaults::default())
}

/// Resolved arguments for `oxibonsai serve`, merged from CLI flags and
/// `--config` in `mod.rs` (cli-04).
pub(crate) struct ServeArgs {
    pub(crate) model: Option<String>,
    pub(crate) host: String,
    pub(crate) port: u16,
    /// `None` = the per-architecture default (8192 for `qwen35`, else 4096).
    pub(crate) max_seq_len: Option<usize>,
    /// `[sampling].seed` from `--config`, already validated (`mod.rs`'s
    /// `resolve_seed_override`; serve has no `--seed` flag of its own).
    /// `None` when unset, in which case [`resolve_seed`] falls back to
    /// `OXIBONSAI_SEED` or a pseudo-random value.
    pub(crate) toml_seed: Option<u64>,
    pub(crate) tokenizer: Option<String>,
    pub(crate) pool_size: Option<usize>,
    pub(crate) bearer_token: Option<String>,
    pub(crate) max_concurrent_requests: usize,
    pub(crate) request_timeout_ms: u64,
    #[cfg(feature = "rag")]
    pub(crate) rag: bool,
    pub(crate) backend: oxibonsai_runtime::engine_seam::Backend,
    pub(crate) rope_scaling: oxibonsai_runtime::config::RopeScalingMode,
    /// cli-11 server-wide defaults (a request's own fields always win).
    pub(crate) enable_thinking: Option<bool>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) tools: Option<String>,
    /// F-M4: already applied to `OXIBONSAI_CUDA_DEVICE` before the async
    /// runtime started (`mod.rs::apply_pre_runtime_env`); kept for the log.
    pub(crate) cuda_device: Option<u32>,
    pub(crate) bearer_token_file: Option<String>,
    pub(crate) rate_limit_rpm: Option<u32>,
    pub(crate) rate_limit_burst: Option<u32>,
    pub(crate) cors_origin: Option<String>,
    pub(crate) cors_allow_credentials: bool,
    pub(crate) max_body_bytes: Option<u64>,
    /// SV-26: mount the bundled UI at `GET /ui`.
    pub(crate) enable_ui: bool,
    /// SV-28: hard ceiling on a request's effective `max_tokens`.
    pub(crate) max_output_tokens: Option<usize>,
    pub(crate) ptq1_transcode: bool,
    pub(crate) prefill_chunk: Option<usize>,
    /// §5.7 vision flags: validated, then a typed `NOT_YET_SUPPORTED`.
    pub(crate) vision: bonsai2::VisionRequest,
    pub(crate) embedding_backend: EmbeddingBackendChoice,
    /// `--embedding-corpus`: the documents `--embedding-backend tfidf`
    /// fits its vocabulary on.
    pub(crate) embedding_corpus: Option<String>,
}

pub(crate) async fn run(args: ServeArgs) -> anyhow::Result<()> {
    let ServeArgs {
        model,
        host,
        port,
        max_seq_len,
        toml_seed,
        tokenizer,
        pool_size,
        bearer_token,
        max_concurrent_requests,
        request_timeout_ms,
        #[cfg(feature = "rag")]
        rag,
        backend,
        rope_scaling,
        enable_thinking,
        reasoning_effort,
        tools,
        cuda_device,
        bearer_token_file,
        rate_limit_rpm,
        rate_limit_burst,
        cors_origin,
        cors_allow_credentials,
        max_body_bytes,
        enable_ui,
        max_output_tokens,
        ptq1_transcode,
        prefill_chunk,
        vision,
        embedding_backend,
        embedding_corpus,
    } = args;

    vision.reject_until_supported()?;
    // `--embedding-backend tfidf`: the corpus is read and validated before
    // any model is resolved or loaded.
    let tfidf_corpus = resolve_tfidf_corpus(embedding_backend, embedding_corpus.as_deref())?;
    if let Some(device) = cuda_device {
        tracing::info!(
            device,
            "CUDA device ordinal applied via OXIBONSAI_CUDA_DEVICE before the async runtime \
             started (takes effect on a native-CUDA build)"
        );
    }
    let contract = ChatContract::from_flags(enable_thinking, reasoning_effort, tools.as_deref())?;

    let model = model
        .or_else(|| std::env::var("OXI_MODEL").ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            anyhow::anyhow!("no model: pass --model <gguf> or set OXI_MODEL (e.g. in .env)")
        })?;
    let tokenizer = tokenizer.or_else(|| {
        std::env::var("OXI_TOKENIZER")
            .ok()
            .filter(|s| !s.is_empty())
    });

    // SV-33: `--bearer-token` is visible to other local users via `ps` (and
    // lands in shell history) — warn when it was the source, and let a
    // token FILE (`--bearer-token-file`, else `OXIBONSAI_BEARER_TOKEN_FILE`;
    // ps-safe) override it. The flag path is passed explicitly: the process
    // environment is never mutated on this (multi-threaded) runtime.
    let bearer_token_flag_used = bearer_token.is_some();
    let bearer_token = bearer_token.or_else(|| {
        std::env::var("OXIBONSAI_BEARER_TOKEN")
            .ok()
            .filter(|s| !s.is_empty())
    });
    let bearer_token =
        resolve_bearer_token_file_override(bearer_token_file.as_deref(), bearer_token)?;
    if bearer_token_flag_used {
        tracing::warn!(
            "--bearer-token was passed on the command line, which is visible to other local \
             users via `ps` (and lands in shell history); prefer --bearer-token-file / \
             OXIBONSAI_BEARER_TOKEN_FILE or the OXIBONSAI_BEARER_TOKEN environment variable"
        );
    }

    // sec-M3 / SV-30: validate the hardening knobs (a zero concurrency or
    // timeout admits a permanently broken server; a one-character token is
    // trivially guessable).
    admission::validate_serve_args(
        bearer_token.as_deref(),
        max_concurrent_requests,
        request_timeout_ms,
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    // sec-15 / SV-07 / sec-M2: refuse a non-loopback bind with no auth.
    let insecure_no_auth = env_flag("OXIBONSAI_INSECURE_NO_AUTH");
    bind_safety_check(&host, bearer_token.as_deref(), insecure_no_auth)?;

    // sec-12: verify the model's checksum, if one is known.
    let checksums_path: PathBuf = std::env::var("OXIBONSAI_CHECKSUMS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("scripts/checksums.sha256"));
    verify_model_checksum(Path::new(&model), &checksums_path).map_err(|e| anyhow::anyhow!(e))?;

    // Engine-pool size precedence: --pool-size > OXIBONSAI_ENGINE_POOL_SIZE
    // > the resolver's own default (hybrid 1; CPU min(4, cores); GPU 1).
    let requested_pool_size: Option<usize> = pool_size.or_else(|| {
        std::env::var("OXIBONSAI_ENGINE_POOL_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
    });

    tracing::info!(model = %model, host = %host, port, "starting server");

    // One mapping (or `--ptq1-transcode` image), leaked for the process
    // lifetime: the guards below and every pool replica read the same bytes.
    let source = ModelSource::open(&model, ptq1_transcode)?;
    let weight_bytes = source.weight_bytes();
    let transcoded = source.transcoded_tensors();
    let bytes = source.into_static();
    let gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static> = Box::leak(Box::new(
        oxibonsai_core::gguf::reader::GgufFile::parse(bytes)?,
    ));
    let arch = gguf
        .metadata
        .get_string(oxibonsai_core::gguf::tensor_info::keys::GENERAL_ARCHITECTURE)
        .unwrap_or("")
        .to_string();

    // The context guard (design §5.6 / Appendix A.3): qwen35 → 8192
    // default + the RAM/model-limit guard; the `--rope-scaling` pre-flight
    // (shared with run/chat).
    let max_seq_len = super::cmd_run::apply_bonsai2_load_time_guards(
        gguf,
        &arch,
        weight_bytes,
        max_seq_len,
        rope_scaling,
    )?;

    // RT-17: the model's own sampling defaults are the pool's baseline.
    let declared = oxibonsai_runtime::sampling::GgufSamplingDefaults::from_metadata(&gguf.metadata);
    let params = baseline_sampling_params(&declared);
    tracing::info!(
        temperature = params.temperature,
        top_k = params.top_k,
        top_p = params.top_p,
        from_gguf =
            declared.temperature.is_some() || declared.top_p.is_some() || declared.top_k.is_some(),
        "baseline sampling parameters (per-request fields override)"
    );
    // GGUF-declared `general.sampling.min_p` becomes every replica's
    // baseline (`InferenceEngine::set_min_p`, applied pool-wide below once
    // the pool exists): a request's own `min_p` overrides it for that
    // request only and is restored afterwards (`server/sampling_scope.rs`).
    let declared_min_p = declared.min_p.unwrap_or(0.0);
    if declared_min_p > 0.0 {
        tracing::info!(
            min_p = declared_min_p,
            "applying the model's declared general.sampling.min_p as every replica's baseline \
             (a request's own min_p overrides it for that request only)"
        );
    }
    let metrics = Arc::new(oxibonsai_runtime::InferenceMetrics::new());

    // cli-M5: a pseudo-random seed unless OXIBONSAI_SEED or `[sampling].seed`
    // (--config) pins one.
    let seed = resolve_seed(toml_seed);
    tracing::info!(seed, "resolved RNG seed");

    let built = oxibonsai_runtime::engine_pool::build_pool_from_static_gguf_with_rope(
        gguf,
        params.clone(),
        seed,
        max_seq_len,
        requested_pool_size,
        backend,
        rope_scaling.into(),
    )?;
    let pool = Arc::clone(&built.pool);
    pool.set_metrics_all(&metrics)?;
    // The GGUF-declared min_p (or 0.0, disabling the filter) is every
    // replica's baseline BEFORE the first lease is ever handed out.
    pool.set_min_p_all(declared_min_p)?;
    let pool_size_actual = pool.size();
    // `--prefill-chunk` on every replica, and the engine-accessor summary
    // line (cli-16) — held all at once, before serving. The admin engine
    // report is captured here too (replica #1 is only ever reachable
    // through a lease) and threaded through `RouterOptions` below, rather
    // than a process-wide registration.
    let mut engine_report = None;
    {
        let mut leases = Vec::with_capacity(pool_size_actual);
        for _ in 0..pool_size_actual {
            leases.push(
                pool.acquire()
                    .await
                    .map_err(|e| anyhow::anyhow!("engine pool: {e}"))?,
            );
        }
        for lease in &mut leases {
            bonsai2::apply_prefill_chunk(lease, prefill_chunk)?;
        }
        if let Some(first) = leases.first() {
            let mut summary = model_desc::engine_summary(first);
            if transcoded > 0 {
                summary.push_str(&format!(
                    " | --ptq1-transcode: {transcoded} PTQ1_0 tensors re-encoded to PQ2_0"
                ));
            }
            tracing::info!(replicas = pool_size_actual, "{summary}");
            // `/admin/status` and `/admin/config`
            // report the resolved variant and the effective kernel tier.
            engine_report = Some(oxibonsai_runtime::admin::EngineReport::from_engine(first));
        }
    }

    // TOK-08: every tokenizer instance the server
    // needs (router, embedder, RAG) comes from ONE authoritative
    // resolution, with the GGUF's own template attached; further instances
    // are derived quietly (`TokenizerBridge` is not `Clone`), instead of
    // repeating the resolution's own logging and TOK-08 compatibility check
    // once per consumer.
    #[cfg(feature = "rag")]
    let need_rag = rag;
    #[cfg(not(feature = "rag"))]
    let need_rag = false;
    let tokenizers = resolve_all_serving_tokenizers(
        tokenizer.as_deref(),
        &model,
        gguf,
        matches!(embedding_backend, EmbeddingBackendChoice::Model),
        need_rag,
    )?;
    if tokenizers.router.is_none() {
        tracing::warn!("{}", missing_tokenizer_warning(&tokenizers.lookup.searched));
    }

    // SV-25 / RT-08: `/v1/embeddings` from a dedicated embedding engine
    // (dense OR hybrid models alike); TF-IDF from a fitted
    // registry through the SAME router seam as the model-backed embedder
    // (no CLI-side interceptor); `None` from either arm carries its
    // reason into the `501` body.
    let mut embeddings_registry = None;
    let mut embedder_unavailable = None;
    let embedder = match embedding_backend {
        EmbeddingBackendChoice::Model => {
            let (built_embedder, reason) = build_embedder(
                &built,
                tokenizers.embedder,
                EmbeddingEngineLoad {
                    params,
                    seed,
                    max_seq_len,
                    backend,
                    rope_scaling,
                },
            );
            embedder_unavailable = reason;
            built_embedder
        }
        EmbeddingBackendChoice::Tfidf => {
            let corpus = tfidf_corpus.unwrap_or_default();
            let registry = oxibonsai_runtime::embeddings::EmbedderRegistry::new(TFIDF_MAX_FEATURES);
            registry.fit_tfidf(&corpus);
            tracing::info!(
                documents = corpus.len(),
                max_features = TFIDF_MAX_FEATURES,
                "serving /v1/embeddings from a TF-IDF vocabulary fitted on --embedding-corpus \
                 (lexical, non-semantic vectors)"
            );
            embeddings_registry = Some(registry);
            None
        }
        EmbeddingBackendChoice::None => {
            tracing::info!("--embedding-backend none: /v1/embeddings answers 501");
            embedder_unavailable = Some((None, "--embedding-backend none".to_string()));
            None
        }
    };

    let admin_auth = resolve_admin_auth();

    let mut router_options = RouterOptions::default()
        .with_limits(
            RequestLimits::default()
                .with_max_input_tokens(Some(max_seq_len))
                .with_timeout_ms(request_timeout_ms),
        )
        .with_auth(admin_auth)
        .with_embedder(embedder)
        .with_enable_ui(enable_ui);
    if let Some(ceiling) = max_output_tokens {
        router_options = router_options.with_max_output_tokens_ceiling(ceiling);
    }
    if let Some(registry) = embeddings_registry {
        router_options = router_options.with_embeddings_registry(registry);
    }
    if let Some((code, message)) = embedder_unavailable {
        router_options = router_options.with_embedder_unavailable(code, message);
    }
    if let Some(report) = engine_report {
        router_options = router_options.with_engine_report(report);
    }
    if enable_ui {
        tracing::info!("serving the bundled chat UI at GET /ui (--enable-ui)");
    }

    #[cfg_attr(not(feature = "rag"), allow(unused_mut))]
    let mut router = create_router_full(
        Arc::clone(&pool),
        tokenizers.router,
        Arc::clone(&metrics),
        router_options,
    );

    #[cfg(feature = "rag")]
    if rag {
        tracing::info!("mounting RAG HTTP API (/rag/index, /rag/query, /rag/stats)");
        router = router.merge(oxibonsai_runtime::rag_server::create_rag_router_with_pool(
            Arc::clone(&pool),
            tokenizers.rag,
        ));
    }

    // ── Hardening: a flag wins over its `OXIBONSAI_*` env var ──────────────
    let max_body_bytes = max_body_bytes
        .map(|b| usize::try_from(b).unwrap_or(usize::MAX))
        .or_else(|| env_usize("OXIBONSAI_MAX_BODY_BYTES"))
        .unwrap_or(4 * 1024 * 1024);
    let opts = HardeningOptions {
        bearer_token,
        max_concurrent_requests,
        request_timeout_ms,
        max_body_bytes,
        cors_origins: cors_origin
            .map(|o| vec![o])
            .unwrap_or_else(env_cors_origins),
        cors_allow_credentials: cors_allow_credentials
            || env_flag("OXIBONSAI_CORS_ALLOW_CREDENTIALS"),
        rate_limit_rpm: rate_limit_rpm
            .map(f64::from)
            .or_else(|| env_f64("OXIBONSAI_RATE_LIMIT_RPM")),
        rate_limit_burst: rate_limit_burst
            .map(f64::from)
            .or_else(|| env_f64("OXIBONSAI_RATE_LIMIT_BURST"))
            .unwrap_or(20.0),
        chat_defaults: ChatDefaults::from_contract(&contract),
    };
    if !opts.chat_defaults.is_empty() {
        tracing::info!(
            enable_thinking = ?opts.chat_defaults.enable_thinking,
            reasoning_effort = ?opts.chat_defaults.reasoning_effort,
            tools = opts.chat_defaults.tools_json.is_some(),
            "server-wide chat defaults (applied to a request that carries none of its own)"
        );
    }
    let router = harden_router(router, pool_size_actual, &opts, &host);

    let addr_str = format!("{host}:{port}");
    let addr: std::net::SocketAddr = addr_str
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid bind address '{addr_str}': {e}"))?;

    // Graceful shutdown (SIGTERM + Ctrl+C) and connect-info wiring, exactly
    // as the standalone `oxibonsai-serve` binary; a failed signal-handler
    // registration refuses to start (SV-29).
    let signals = install_shutdown_signals()?;
    serve_with_shutdown(router, addr, signals)
        .await
        .map_err(anyhow::Error::from_boxed)?;

    Ok(())
}

// ─── `--embedding-backend tfidf` (RT-EMBEDDINGS residue) ────────────────────

/// Vocabulary cap (`max_features`) of the TF-IDF embedder: the same
/// dimension the runtime router's own embeddings registry is built with.
const TFIDF_MAX_FEATURES: usize = 512;

/// Validate the `--embedding-backend` / `--embedding-corpus` pair and read
/// the corpus (one document per non-blank line). `Ok(None)` for every
/// backend but `tfidf`.
///
/// # Errors
///
/// `tfidf` without a corpus, a corpus with any other backend, an unreadable
/// corpus file, or one with no documents.
fn resolve_tfidf_corpus(
    backend: EmbeddingBackendChoice,
    corpus_path: Option<&str>,
) -> anyhow::Result<Option<Vec<String>>> {
    match (backend, corpus_path) {
        (EmbeddingBackendChoice::Tfidf, None) => anyhow::bail!(
            "--embedding-backend tfidf needs --embedding-corpus <file> (one document per line): \
             the TF-IDF vocabulary is fitted once, at startup, never on client requests"
        ),
        (EmbeddingBackendChoice::Tfidf, Some(path)) => load_embedding_corpus(path).map(Some),
        (_, Some(_)) => {
            anyhow::bail!("--embedding-corpus only applies with --embedding-backend tfidf")
        }
        (_, None) => Ok(None),
    }
}

/// Read a TF-IDF corpus: one document per line, surrounding whitespace
/// trimmed, blank lines skipped.
///
/// # Errors
///
/// An unreadable file, or one that holds no document at all.
fn load_embedding_corpus(path: &str) -> anyhow::Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read --embedding-corpus '{path}': {e}"))?;
    let documents: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();
    if documents.is_empty() {
        anyhow::bail!("--embedding-corpus '{path}' holds no documents (one per line)");
    }
    Ok(documents)
}

/// One serving tokenizer instance (TOK-08): the
/// `run`/`chat` ladder — an explicit `--tokenizer`, else a vocab-matching
/// `tokenizer.json` next to the model, else the vocabulary embedded in the
/// GGUF — with the hard compatibility check and the GGUF's own
/// `tokenizer.chat_template` attached (the built-in ChatML fallback only
/// for a file that ships none). Logs "resolved chat template" and the
/// tokenizer-source line exactly once, whatever the outcome. Test-only:
/// `run` itself resolves through [`resolve_all_serving_tokenizers`], which
/// covers the multi-consumer case this single-tokenizer helper does not
/// need to.
#[cfg(test)]
fn load_serving_tokenizer(
    explicit: Option<&str>,
    model: &str,
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
) -> anyhow::Result<(
    Option<oxibonsai_runtime::TokenizerBridge>,
    super::util::TokenizerLookup,
)> {
    let (tok, lookup, _source) = super::util::resolve_serving_tokenizer(explicit, model, gguf)?;
    Ok((tok, lookup))
}

/// Every `TokenizerBridge` instance `serve` needs, all derived from ONE
/// authoritative resolution: `TokenizerBridge` is not `Clone`, so the
/// router, the model-backed embedder and the RAG router each need their own
/// instance. Resolving once — the vocab-aware ladder plus the TOK-08
/// compatibility check, with their "resolved chat template" and
/// tokenizer-source logging — and deriving any further instances quietly
/// ([`super::util::rebuild_serving_tokenizer`]) logs each line exactly once
/// per startup, however many consumers need their own instance.
struct ServingTokenizers {
    /// The router's own tokenizer (`None` = no usable tokenizer at all).
    router: Option<oxibonsai_runtime::TokenizerBridge>,
    /// `router`'s lookup, for [`missing_tokenizer_warning`].
    lookup: super::util::TokenizerLookup,
    /// A further instance for the model-backed embedder, when requested.
    embedder: Option<oxibonsai_runtime::TokenizerBridge>,
    /// A further instance for the RAG router, when requested (only the
    /// `rag` feature has a consumer for it).
    #[cfg(feature = "rag")]
    rag: Option<oxibonsai_runtime::TokenizerBridge>,
}

/// Resolve [`ServingTokenizers`]. Sync and free of any I/O beyond parsing
/// `gguf`'s already-loaded metadata, so it is directly unit-testable (unlike
/// `run`, which is async and loads a real model) under
/// `tracing::subscriber::with_default`.
fn resolve_all_serving_tokenizers(
    explicit: Option<&str>,
    model: &str,
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    need_embedder: bool,
    need_rag: bool,
) -> anyhow::Result<ServingTokenizers> {
    let (router, lookup, source) = super::util::resolve_serving_tokenizer(explicit, model, gguf)?;
    let rebuild = |need: bool| -> anyhow::Result<Option<oxibonsai_runtime::TokenizerBridge>> {
        match (need, &source) {
            (true, Some(source)) => super::util::rebuild_serving_tokenizer(source, gguf).map(Some),
            _ => Ok(None),
        }
    };
    let embedder = rebuild(need_embedder)?;
    #[cfg(feature = "rag")]
    let rag = rebuild(need_rag)?;
    // Without the `rag` feature nothing consumes a RAG tokenizer, so the
    // request is honoured by not building one.
    #[cfg(not(feature = "rag"))]
    let _ = need_rag;
    Ok(ServingTokenizers {
        router,
        lookup,
        embedder,
        #[cfg(feature = "rag")]
        rag,
    })
}

/// How the dedicated embedding engine is built: the pool's own sampling
/// baseline and seed, and the SAME `--backend` / `--rope-scaling` the pool
/// honours (a `--backend cpu` server must never upload weights to the GPU
/// for its embedder).
struct EmbeddingEngineLoad {
    params: oxibonsai_runtime::sampling::SamplingParams,
    seed: u64,
    max_seq_len: usize,
    backend: oxibonsai_runtime::engine_seam::Backend,
    rope_scaling: oxibonsai_runtime::config::RopeScalingMode,
}

/// The dense embedding engine: built off the pool's own leaked GGUF and
/// shared token-embedding table, with `--backend` and `--rope-scaling`
/// applied exactly as for the pool's replicas.
fn build_embedding_engine(
    built: &oxibonsai_runtime::engine_pool::PoolBuild,
    load: &EmbeddingEngineLoad,
    window: usize,
) -> oxibonsai_runtime::RuntimeResult<oxibonsai_runtime::InferenceEngine<'static>> {
    let rope: oxibonsai_core::config::RopeScalingOverride = load.rope_scaling.into();
    oxibonsai_runtime::engine_seam::resolve_rope_scaling_at_load(built.gguf, rope)?;
    let _rope_scope = oxibonsai_core::config::RopeScalingOverrideScope::enter(rope);
    oxibonsai_runtime::InferenceEngine::from_gguf_static_with_embd_and_backend(
        built.gguf,
        load.params.clone(),
        load.seed,
        window,
        Arc::clone(&built.shared_token_embd),
        load.backend,
    )
}

/// Why there is no model-backed embedder: an `error.code` (an engine
/// refusal's own code, e.g. `NOT_A_DENSE_MODEL`), or `None` for a generic
/// reason, and a human-readable message — for
/// `RouterOptions::with_embedder_unavailable`.
type EmbedderUnavailable = (Option<&'static str>, String);

/// The model-backed embedder `/v1/embeddings` is served from (SV-25 /
/// RT-08), or `None` — the route's honest `501` — when there can be none,
/// alongside why: `error.code` (an engine refusal's own code, e.g.
/// `NOT_A_DENSE_MODEL`) and a human-readable message, carried into the
/// `501` body through `RouterOptions::with_embedder_unavailable` instead of
/// staying log-only.
///
/// A dense OR hybrid (`qwen35`) model (hybrids embed too) gets a
/// dedicated embedding engine ([`build_embedding_engine`]) with its KV
/// window bounded by the embedder's input ceiling. The `NOT_A_DENSE_MODEL`
/// arm stays as a defensive branch for a future dense-only refinement of the
/// engine seam; it is not reachable for the models this build actually
/// supports.
fn build_embedder(
    built: &oxibonsai_runtime::engine_pool::PoolBuild,
    tokenizer: Option<oxibonsai_runtime::TokenizerBridge>,
    load: EmbeddingEngineLoad,
) -> (
    Option<Arc<oxibonsai_runtime::embed_engine::ModelEmbedder>>,
    Option<EmbedderUnavailable>,
) {
    let Some(tokenizer) = tokenizer else {
        let reason = "no tokenizer: an embedder needs one to encode text".to_string();
        tracing::info!("{reason}; /v1/embeddings answers 501");
        return (None, Some((None, reason)));
    };
    let window = load.max_seq_len.clamp(
        1,
        oxibonsai_runtime::embed_engine::DEFAULT_MAX_EMBEDDING_TOKENS,
    );
    let tokenizer = Arc::new(tokenizer);
    let result = if oxibonsai_model::hybrid::LoadedModel::is_hybrid_gguf(built.gguf) {
        oxibonsai_runtime::embed_engine::ModelEmbedder::from_static_gguf(
            built.gguf,
            Arc::clone(&built.shared_token_embd),
            tokenizer,
            load.params,
            load.seed,
            window,
        )
    } else {
        build_embedding_engine(built, &load, window).and_then(|engine| {
            oxibonsai_runtime::embed_engine::ModelEmbedder::from_engine(engine, tokenizer)
        })
    };
    match result {
        Ok(embedder) => {
            tracing::info!(
                window,
                backend = %load.backend,
                "serving /v1/embeddings from a dedicated embedding engine"
            );
            (Some(embedder), None)
        }
        Err(e) if oxibonsai_runtime::engine::engine_error_code(&e) == Some("NOT_A_DENSE_MODEL") => {
            tracing::info!(
                error = %e,
                "embeddings are unavailable for this model; /v1/embeddings answers 501"
            );
            (
                None,
                Some((
                    oxibonsai_runtime::engine::engine_error_code(&e),
                    format!("embeddings are unavailable for this model: {e}"),
                )),
            )
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "failed to build the embedding engine; /v1/embeddings answers 501"
            );
            (
                None,
                Some((
                    oxibonsai_runtime::engine::engine_error_code(&e),
                    format!("failed to build the embedding engine: {e}"),
                )),
            )
        }
    }
}

// ─── cli-11: server-wide chat defaults ─────────────────────────────────────

/// The chat paths the server-wide defaults apply to.
const CHAT_PATHS: [&str; 2] = ["/v1/chat/completions", "/v1/chat/completions/extended"];

/// `--think`/`--no-think`, `--reasoning-effort` and `--tools` as
/// server-wide defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ChatDefaults {
    enable_thinking: Option<bool>,
    reasoning_effort: Option<String>,
    /// Raw JSON text (key order preserved end to end).
    tools_json: Option<String>,
}

impl ChatDefaults {
    fn from_contract(contract: &ChatContract) -> Self {
        Self {
            enable_thinking: contract.enable_thinking,
            reasoning_effort: contract.reasoning_effort.clone(),
            tools_json: contract.tools_json.clone(),
        }
    }

    fn is_empty(&self) -> bool {
        self.enable_thinking.is_none()
            && self.reasoning_effort.is_none()
            && self.tools_json.is_none()
    }
}

/// Apply the server-wide defaults to one chat request body: each default is
/// inserted as a TOP-LEVEL field only when the request carries neither that
/// top-level field nor its `chat_template_kwargs.*` twin (the server reads
/// both; the nested one wins). Returns `None` — forward the ORIGINAL bytes
/// untouched — when nothing needs adding or the body is not a JSON object.
///
/// Pure text splicing, never a `serde_json::Value` round trip: the body is
/// parsed only to learn which keys exist, and the new fields are inserted
/// right after the opening `{`, so every byte of the client's own JSON
/// (including a tools schema's key order) reaches the handler unchanged.
pub(crate) fn inject_chat_defaults(body: &[u8], defaults: &ChatDefaults) -> Option<Vec<u8>> {
    if defaults.is_empty() {
        return None;
    }
    let parsed: serde_json::Value = serde_json::from_slice(body).ok()?;
    let object = parsed.as_object()?;
    let kwargs = object
        .get("chat_template_kwargs")
        .and_then(serde_json::Value::as_object);
    let has = |key: &str| object.contains_key(key) || kwargs.is_some_and(|k| k.contains_key(key));

    let mut fields: Vec<String> = Vec::new();
    if let Some(enable) = defaults.enable_thinking {
        if !has("enable_thinking") {
            fields.push(format!("\"enable_thinking\":{enable}"));
        }
    }
    if let Some(effort) = &defaults.reasoning_effort {
        if !has("reasoning_effort") {
            let quoted = serde_json::to_string(effort).ok()?;
            fields.push(format!("\"reasoning_effort\":{quoted}"));
        }
    }
    if let Some(tools) = &defaults.tools_json {
        if !object.contains_key("tools") {
            fields.push(format!("\"tools\":{tools}"));
        }
    }
    if fields.is_empty() {
        return None;
    }
    let open = body.iter().position(|b| !b.is_ascii_whitespace())?;
    if body[open] != b'{' {
        return None;
    }
    let mut insert = fields.join(",");
    if !object.is_empty() {
        insert.push(',');
    }
    let mut out = Vec::with_capacity(body.len() + insert.len());
    out.extend_from_slice(&body[..=open]);
    out.extend_from_slice(insert.as_bytes());
    out.extend_from_slice(&body[open + 1..]);
    Some(out)
}

/// State of [`chat_defaults_mw`].
struct ChatDefaultsState {
    defaults: ChatDefaults,
    max_body_bytes: usize,
}

/// Rewrite a chat request's body with the server-wide defaults (see
/// [`inject_chat_defaults`]); every other request passes through untouched.
/// The body is read bounded by `max_body_bytes` (the `Content-Length` guard
/// outside this layer already refused a declared oversize).
async fn chat_defaults_mw(
    State(state): State<Arc<ChatDefaultsState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if req.method() != axum::http::Method::POST || !CHAT_PATHS.contains(&req.uri().path()) {
        return next.run(req).await;
    }
    let (mut parts, body) = req.into_parts();
    let bytes = match axum::body::to_bytes(body, state.max_body_bytes).await {
        Ok(bytes) => bytes,
        Err(e) => {
            let body = serde_json::json!({
                "error": {
                    "message": format!("failed to read the request body: {e}"),
                    "type": "invalid_request_error",
                    "param": serde_json::Value::Null,
                    "code": "content_too_large",
                }
            });
            return (StatusCode::PAYLOAD_TOO_LARGE, axum::Json(body)).into_response();
        }
    };
    let body = match inject_chat_defaults(&bytes, &state.defaults) {
        Some(rewritten) => {
            parts.headers.insert(
                axum::http::header::CONTENT_LENGTH,
                axum::http::HeaderValue::from(rewritten.len()),
            );
            Body::from(rewritten)
        }
        None => Body::from(bytes),
    };
    next.run(Request::from_parts(parts, body)).await
}

/// Active `Content-Length` precheck — the identical guard
/// `crates/oxibonsai-serve/src/hardening.rs::content_length_guard_mw`
/// mounts (SV-30/sec-M3 "one shared stack"): a request
/// whose *declared* size exceeds the limit gets a synchronous `413` before
/// it can take (or queue for) an admission permit. Only the header is
/// inspected; a request without `Content-Length` (chunked) passes, and is
/// still bounded by `DefaultBodyLimit` inside.
async fn content_length_guard_mw(
    State(max_body_bytes): State<Arc<usize>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let declared_len = req
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok());
    if let Some(len) = declared_len {
        if len > *max_body_bytes {
            let body = serde_json::json!({
                "error": {
                    "message": format!(
                        "request body is {len} bytes (Content-Length), which exceeds the \
                         server's limit of {max_body_bytes} bytes"
                    ),
                    "type": "invalid_request_error",
                    "param": serde_json::Value::Null,
                    "code": "content_too_large",
                }
            });
            return (StatusCode::PAYLOAD_TOO_LARGE, axum::Json(body)).into_response();
        }
    }
    next.run(req).await
}

// ─── Router hardening (SV-06 / sec-06 / sec-07 / SV-10 / cli-18) ───────────

/// Everything [`harden_router`] needs beyond the base router and the real
/// pool size, gathered into one struct so the function stays within
/// clippy's argument-count budget and is trivial to construct directly in
/// tests (see `tests::router_tests` below).
struct HardeningOptions {
    bearer_token: Option<String>,
    max_concurrent_requests: usize,
    request_timeout_ms: u64,
    max_body_bytes: usize,
    cors_origins: Vec<String>,
    cors_allow_credentials: bool,
    rate_limit_rpm: Option<f64>,
    rate_limit_burst: f64,
    /// cli-11 server-wide chat defaults (empty = no rewriting layer).
    chat_defaults: ChatDefaults,
}

/// Apply the full hardening stack to an already-assembled base router
/// (routes + any RAG merge already done — see [`run`]).
///
/// Layer order, innermost first (see the module docs of
/// `oxibonsai_runtime::middleware` and `oxibonsai_runtime::rate_limiter` for
/// why this exact order is required — findings `SV-06` / `sec-06` /
/// `sec-07` / `SV-10` / `cli-18`):
///
/// ```text
/// routes -> [chat-defaults rewrite] -> admission (concurrency+timeout)
///        -> DefaultBodyLimit -> Content-Length precheck -> rate-limit
///        -> bearer-auth -> CORS (outermost)
/// ```
///
/// `--embedding-backend tfidf` is no longer a separate layer here: `run`
/// passes its fitted `EmbedderRegistry` into `RouterOptions::with_embeddings_registry`
/// before the base router is even built, so `POST /v1/embeddings` is one
/// of `router`'s own routes and every layer below applies to it
/// exactly as to any other route.
///
/// — the same stack `crates/oxibonsai-serve/src/hardening.rs::build_router`
/// builds, plus the optional chat-defaults rewrite
/// innermost (only mounted when `--think`/`--no-think`/`--reasoning-effort`/
/// `--tools` set a server-wide default).
///
/// Split out from [`run`] (which cannot be unit tested directly — it loads
/// a real GGUF model) so this composition is directly testable against a
/// tiny in-memory pool, mirroring
/// `crates/oxibonsai-serve/src/main.rs::build_router`.
fn harden_router(
    mut router: Router,
    pool_size: usize,
    opts: &HardeningOptions,
    host_for_log: &str,
) -> Router {
    // sec-20/perf-M1: derive the admission ceiling from the real engine
    // pool size rather than admitting `max_concurrent_requests` (default
    // 32) against a single-replica GPU-tier pool.
    //
    // Applied here, BEFORE `DefaultBodyLimit` below, so
    // `DefaultBodyLimit` ends up mounted OUTSIDE (more outer than) this
    // admission stack.
    // cli-11: the server-wide chat defaults, innermost so the body read is
    // covered by admission's timeout exactly like the handler's own read.
    if !opts.chat_defaults.is_empty() {
        router = router.layer(axum::middleware::from_fn_with_state(
            Arc::new(ChatDefaultsState {
                defaults: opts.chat_defaults.clone(),
                max_body_bytes: opts.max_body_bytes,
            }),
            chat_defaults_mw,
        ));
    }

    let effective_max_concurrent_requests =
        admission::resolve_admission_limit(opts.max_concurrent_requests, pool_size);
    router = admission::apply_admission(
        router,
        effective_max_concurrent_requests,
        opts.request_timeout_ms,
    );

    // SV-27/sec-16: explicit, configurable request body ceiling instead of
    // axum's implicit 2 MiB default.
    //
    // Moved from *inside* `admission` (above) to
    // *outside* it (still inside rate-limit, below). This reorder is
    // correctness-neutral for the permit-holding concern the finding named
    // -- `axum`'s `DefaultBodyLimit` only inserts a request extension
    // consulted later by the `Bytes`/`Json` extractors deep inside the
    // eventual handler, unconditionally calling its inner service with no
    // check of its own at either position -- but it is the right relative
    // position for when a real synchronous body-size guard is added outside
    // `admission`. The fuller
    // version of this note lives on `crates/oxibonsai-serve/src/hardening.rs::build_router`.
    router = router.layer(DefaultBodyLimit::max(opts.max_body_bytes));

    // The ACTIVE synchronous `Content-Length`
    // precheck `oxibonsai-serve`'s `build_router` mounts at this same
    // position — a request whose declared size exceeds the limit is refused
    // with a real 413 before it can hold (or wait for) an admission permit.
    router = router.layer(axum::middleware::from_fn_with_state(
        Arc::new(opts.max_body_bytes),
        content_length_guard_mw,
    ));

    // sec-07/RT-34/SV-10/cli-10: rate limiting, mounted *outside* admission
    // so a rate-limited request never consumes a concurrency permit
    // (cli-18). `--rate-limit-rpm`/`--rate-limit-burst` (else
    // `OXIBONSAI_RATE_LIMIT_RPM`/`_BURST`).
    if let Some(rpm) = opts.rate_limit_rpm {
        let rate_cfg = RateLimitConfig::from_rpm(rpm, opts.rate_limit_burst);
        router = router.layer(rate_limit_layer(rate_cfg));
    }

    // sec-15/SV-07/sec-07/cli-18: bearer auth, mounted *outside* rate
    // limiting so an unauthenticated request never consumes a rate-limit
    // token either. `/admin/*` is gated unconditionally and independently
    // by `create_router_full` (see `run`) regardless of whether this is
    // set (finding `sec-15`) -- this specifically protects the inference
    // routes.
    let router = match &opts.bearer_token {
        Some(token) => {
            tracing::info!("bearer-token authentication enabled");
            let state = admission::BearerAuthState {
                token: token.clone(),
            };
            router.layer(axum::middleware::from_fn_with_state(
                state,
                admission::bearer_auth,
            ))
        }
        None => {
            // sec-M2/sec-15: `/admin/*` is always gated by
            // `create_router_full` (403 without a configured admin token)
            // regardless of this branch; the warning below is about the
            // inference endpoints only.
            tracing::warn!(
                host = %host_for_log,
                "no --bearer-token / OXIBONSAI_BEARER_TOKEN configured: the inference \
                 endpoints are unauthenticated on this listener; /admin/* is separately \
                 gated (403 unless OXI_ADMIN_TOKEN is set) -- set --bearer-token or keep \
                 --host at 127.0.0.1"
            );
            router
        }
    };

    // SV-06/sec-06/cli-10: CORS is the outermost layer, so its `OPTIONS`
    // preflight short-circuit runs *before* the nested bearer-auth layer
    // ever sees the request (`--cors-origin`, else `OXIBONSAI_CORS_ORIGIN`).
    if opts.cors_origins.is_empty() {
        router
    } else {
        let cors = CorsConfig {
            allow_credentials: opts.cors_allow_credentials,
            ..CorsConfig::from_origins(opts.cors_origins.clone())
        };
        apply_middleware(router, MiddlewareConfig::none().with_cors(cors))
    }
}

// ─── sec-15 / SV-07 / sec-M2: bind-safety guard ─────────────────────────────

/// Returns `true` when `host` resolves to a loopback address (`127.0.0.0/8`,
/// `::1`, or the literal `"localhost"`).
fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// Refuse to bind a non-loopback host with no auth configured, unless the
/// operator explicitly acknowledges the risk (findings `sec-15` / `SV-07` /
/// `sec-M2`). Kept identical in spirit to
/// `crates/oxibonsai-serve/src/main.rs::bind_safety_check` (that crate
/// cannot be depended on from here).
fn bind_safety_check(
    host: &str,
    bearer_token: Option<&str>,
    insecure_no_auth: bool,
) -> anyhow::Result<()> {
    if is_loopback_host(host) {
        return Ok(());
    }
    if bearer_token.is_some() || insecure_no_auth {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "refusing to bind to non-loopback host '{host}' with no bearer token configured: \
         set --bearer-token / OXIBONSAI_BEARER_TOKEN, bind to a loopback address instead \
         (127.0.0.1, ::1, or localhost), or explicitly acknowledge the risk with \
         OXIBONSAI_INSECURE_NO_AUTH=1"
    ))
}

// ─── Admin-token env asymmetry between the two binaries ────────────────────

/// Resolve the `/admin/*` authentication policy the same way
/// `crates/oxibonsai-serve/src/hardening.rs::resolve_admin_auth` does: an
/// explicit `OXIBONSAI_ADMIN_TOKEN` takes precedence; otherwise fall back to
/// the `OXI_ADMIN_TOKEN` environment variable (`AuthConfig::from_env`, the
/// pre-existing convention every router constructor already honors).
///
/// Before this fix this binary (`oxibonsai serve`) called
/// `AdminAuthConfig::from_env()` directly, i.e. it only ever honoured
/// `OXI_ADMIN_TOKEN` -- unlike the standalone `oxibonsai-serve` binary,
/// which already resolved `OXIBONSAI_ADMIN_TOKEN` first via its
/// `apply_extra_env_overrides` + `resolve_admin_auth`. An operator who set
/// `OXIBONSAI_ADMIN_TOKEN` therefore got a token-gated `/admin` on one
/// binary and a `403`-locked `/admin` on the other, for what looked like
/// the same configuration.
///
/// No separate `.trim()` guard is needed here: `AuthConfig::with_admin_token`
/// already treats a whitespace-only token as "not configured" and falls
/// back to [`AdminAuthConfig::locked`] internally.
///
/// This is the only place in the module that reads either admin-token
/// environment variable; the actual decision lives in the pure
/// [`resolve_admin_auth_from`] below -- see its doc comment for why the
/// split exists.
fn resolve_admin_auth() -> AdminAuthConfig {
    resolve_admin_auth_from(
        std::env::var("OXIBONSAI_ADMIN_TOKEN").ok().as_deref(),
        AdminAuthConfig::from_env(),
    )
}

/// Pure decision logic behind [`resolve_admin_auth`]: `new_token` wins
/// outright when it is a non-empty string (a whitespace-only value still
/// delegates to [`AdminAuthConfig::with_admin_token`]'s own "not
/// configured" handling rather than falling back to `legacy` -- see the
/// note above); otherwise the already-resolved `legacy` policy is
/// returned unchanged. Neither branch touches the process environment.
///
/// This split exists because `resolve_admin_auth`'s decision used to be
/// inline here, reading `OXIBONSAI_ADMIN_TOKEN` directly, and the
/// `resolve_admin_auth_from_*` tests below drove it by mutating that
/// variable and `OXI_ADMIN_TOKEN` -- the two SAME process-global env vars
/// -- via `unsafe { std::env::set_var / remove_var }`, with no
/// serialization between them. `cargo test` runs unit tests on parallel
/// threads by default, so those tests raced each other (measured
/// ~25-30% failure rate over repeated runs). Worse, they also raced
/// `router_tests::router_for` (below), whose `RouterOptions::default()`
/// reads `OXI_ADMIN_TOKEN` via `AuthConfig::from_env()`
/// (`crates/oxibonsai-runtime/src/server.rs:573`) on every call: a
/// `set_var` concurrent with any `getenv` in the same process is unsound
/// (hence `set_var`/`remove_var` being `unsafe` as of edition 2024), and
/// it demonstrably happened here. A test-module `Mutex<()>` shared by
/// those tests would have fixed the flakiness but not the soundness
/// issue, since `router_tests` has no reason to know about (or take)
/// that lock.
///
/// Taking both the candidate *and* the fallback as plain parameters --
/// rather than only the candidate, with the fallback still resolved by
/// an inline `AdminAuthConfig::from_env()` call inside this function --
/// mirrors the in-tree precedent for exactly this shape:
/// `crates/oxibonsai-serve/src/hardening.rs::apply_extra_env_overrides`
/// takes its `vars` explicitly instead of reading the process env, which
/// is why the serve crate's own env-derived config tests do not race. It
/// also keeps every test below fully discriminating: a test that instead
/// asserted equality against a freshly-called `AdminAuthConfig::from_env()`
/// would pass even if the fallback arm were mistakenly hardcoded to
/// `AdminAuthConfig::locked()`, since `OXI_ADMIN_TOKEN` is unset in the
/// ambient test environment either way -- a vacuous test would have
/// replaced a flaky one.
///
/// Scope note: this removes this finding's `setenv`/`getenv` race on the
/// two admin-token variables specifically; it does not change how the
/// other, unrelated env-derived tests in this module (`OXIBONSAI_SEED`,
/// `OXIBONSAI_BEARER_TOKEN_FILE`, `OXIBONSAI_CORS_ORIGIN`, ...) manage
/// process environment mutation, which is outside this finding's scope.
fn resolve_admin_auth_from(new_token: Option<&str>, legacy: AdminAuthConfig) -> AdminAuthConfig {
    match new_token {
        Some(token) if !token.is_empty() => AdminAuthConfig::with_admin_token(token),
        _ => legacy,
    }
}

// ─── SV-33: ps-safe bearer token override ──────────────────────────────────

/// Override `flag_or_env` with the contents of a bearer-token FILE — the
/// `ps`-safe option finding `SV-33` recommends: `flag_path`
/// (`--bearer-token-file`) when given, else `OXIBONSAI_BEARER_TOKEN_FILE`.
/// The flag path is an explicit parameter, never written into the process
/// environment (that would be a `set_var` on the multi-threaded runtime).
/// Returns `flag_or_env` unchanged when neither names a file.
fn resolve_bearer_token_file_override(
    flag_path: Option<&str>,
    flag_or_env: Option<String>,
) -> anyhow::Result<Option<String>> {
    let (path, source) = match flag_path {
        Some(path) => (path.to_string(), "--bearer-token-file"),
        None => match std::env::var("OXIBONSAI_BEARER_TOKEN_FILE") {
            Ok(path) => (path, "OXIBONSAI_BEARER_TOKEN_FILE"),
            Err(_) => return Ok(flag_or_env),
        },
    };
    read_bearer_token_file(&path, source).map(Some)
}

/// Read and trim one bearer-token file (`source` names where the path came
/// from, for the error).
fn read_bearer_token_file(path: &str, source: &str) -> anyhow::Result<String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read {source} {path}: {e}"))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(anyhow::anyhow!("{source} {path} is empty"));
    }
    Ok(trimmed.to_string())
}

// ─── cli-M5: pseudo-random default seed ─────────────────────────────────────

/// Resolve the RNG seed: an `OXIBONSAI_SEED` env override, else `toml_seed`
/// (`[sampling].seed` from `--config`, already validated by
/// `resolve_seed_override` in `mod.rs`), else a process/time-derived
/// pseudo-random value (finding `cli-M5`: a hardcoded `42` never varies).
/// Per-request reproducibility comes from a request's own `seed` field;
/// this is only the replicas' starting state. No `rand` crate is reachable
/// from this crate without a new Cargo dependency, and none is needed: this
/// is not cryptographically random, it only needs to differ from one server
/// start to the next.
fn resolve_seed(toml_seed: Option<u64>) -> u64 {
    if let Ok(v) = std::env::var("OXIBONSAI_SEED") {
        if let Ok(seed) = v.parse::<u64>() {
            return seed;
        }
    }
    if let Some(seed) = toml_seed {
        return seed;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    // A stack-address-derived salt (affected by ASLR) adds entropy beyond
    // wall-clock time without any additional dependency.
    let probe = 0u8;
    let stack_salt = std::ptr::addr_of!(probe) as u64;
    nanos ^ pid.rotate_left(32) ^ stack_salt.rotate_left(16)
}

// ─── sec-12 / SV-30 / sec-M3: model checksum verification ──────────────────
//
// `lookup_expected_checksum` / `verify_model_checksum` (which computes a
// real streaming SHA-256, no `Ok(None)` stub) now live once, canonically,
// in `oxibonsai_runtime::serve_shared` (also used by
// `crates/oxibonsai-serve/src/hardening.rs`), closing the "duplicated
// verbatim across two binaries" finding at the same time as the checksum
// stub. `verify_model_checksum` is imported at the top of this file (used
// by `run` above); `lookup_expected_checksum` is imported directly inside
// this module's own tests, which are the only place in this crate that
// still reference it by name.

// ─── OXIBONSAI_* env var helpers (the fallbacks behind each serve flag) ────

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok())
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
}

fn env_cors_origins() -> Vec<String> {
    std::env::var("OXIBONSAI_CORS_ORIGIN")
        .ok()
        .map(|v| {
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
#[path = "cmd_serve_tests.rs"]
mod tests;
