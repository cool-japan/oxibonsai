//! `oxibonsai serve` — OpenAI-compatible HTTP API server.

use std::sync::Arc;

use super::admission;
use super::util::{missing_tokenizer_warning, resolve_tokenizer};

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run(
    model: Option<String>,
    host: String,
    port: u16,
    max_seq_len: usize,
    tokenizer: Option<String>,
    pool_size: Option<usize>,
    bearer_token: Option<String>,
    max_concurrent_requests: usize,
    request_timeout_ms: u64,
    #[cfg(feature = "rag")] rag: bool,
) -> anyhow::Result<()> {
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

    // Engine-pool size precedence: --pool-size flag > env
    // OXIBONSAI_ENGINE_POOL_SIZE > unset. When unset we pass `None`
    // so `build_pool_from_gguf`/`resolve_pool_size` apply the CPU
    // default of `min(4, cores)` — replicas now share one `Arc<[f32]>`
    // token-embedding table, so the extra per-replica cost is just a
    // KV cache. GPU/Metal is always clamped back to 1 by the resolver.
    let requested_pool_size: Option<usize> = pool_size.or_else(|| {
        std::env::var("OXIBONSAI_ENGINE_POOL_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
    });

    tracing::info!(model = %model, host = %host, port, "starting server");

    let params = oxibonsai_runtime::sampling::SamplingParams::default();
    let metrics = Arc::new(oxibonsai_runtime::InferenceMetrics::new());

    // Build a pool of engine replicas sharing one leaked `'static`
    // GGUF (replica #1 mmaps + leaks; the rest reuse it zero-copy).
    let (pool, _tier, _size) = oxibonsai_runtime::engine_pool::build_pool_from_gguf(
        &model,
        params,
        42,
        max_seq_len,
        requested_pool_size,
    )?;
    // Wire the shared metrics onto every replica, preserving the
    // per-engine telemetry the single-engine path recorded.
    pool.set_metrics_all(&metrics)?;

    // Resolve the tokenizer path once; a fresh `TokenizerBridge` is
    // built per router below since `TokenizerBridge` does not
    // implement `Clone` and the RAG router (when mounted) needs its
    // own independent handle.
    let lookup = resolve_tokenizer(tokenizer.as_deref(), &model);
    if lookup.found.is_none() {
        tracing::warn!("{}", missing_tokenizer_warning(&lookup.searched));
    }
    let load_tok = || -> anyhow::Result<Option<oxibonsai_runtime::TokenizerBridge>> {
        match &lookup.found {
            Some(p) => Ok(Some(oxibonsai_runtime::TokenizerBridge::from_file(p)?)),
            None => Ok(None),
        }
    };

    let tok = load_tok()?;

    #[cfg_attr(not(feature = "rag"), allow(unused_mut))]
    let mut router =
        oxibonsai_runtime::server::create_router_with_pool(Arc::clone(&pool), tok, metrics);

    #[cfg(feature = "rag")]
    if rag {
        let rag_tok = load_tok()?;
        tracing::info!("mounting RAG HTTP API (/rag/index, /rag/query, /rag/stats)");
        router = router.merge(oxibonsai_runtime::rag_server::create_rag_router_with_pool(
            Arc::clone(&pool),
            rag_tok,
        ));
    }

    // ── Hardening: bearer auth + admission control ──────────
    //
    // Mirrors the protection `oxibonsai-serve` (the standalone
    // hardened binary) applies, so this documented `oxibonsai
    // serve` entry point offers equivalent protection instead of
    // handing the runtime's bare router straight to
    // `axum::serve`. See `cli::admission`.
    let bearer_token = bearer_token.or_else(|| {
        std::env::var("OXIBONSAI_BEARER_TOKEN")
            .ok()
            .filter(|s| !s.is_empty())
    });
    let router = match bearer_token {
        Some(token) => {
            tracing::info!("bearer-token authentication enabled");
            let state = admission::BearerAuthState { token };
            router.layer(axum::middleware::from_fn_with_state(
                state,
                admission::bearer_auth,
            ))
        }
        None => {
            tracing::warn!(
                "no --bearer-token / OXIBONSAI_BEARER_TOKEN configured: all \
                 endpoints (including /admin/*) are unauthenticated on this \
                 listener; set --bearer-token or keep --host at 127.0.0.1"
            );
            router
        }
    };
    let router = admission::apply_admission(router, max_concurrent_requests, request_timeout_ms);

    let addr_str = format!("{host}:{port}");
    let addr: std::net::SocketAddr = addr_str
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid bind address '{addr_str}': {e}"))?;

    // Serve with graceful shutdown (SIGTERM + Ctrl+C) and
    // connect-info wiring (`into_make_service_with_connect_info::
    // <SocketAddr>`), mirroring the standalone `oxibonsai-serve`
    // binary's `serve_with_shutdown` helper
    // (crates/oxibonsai-serve/src/main.rs) instead of handing the
    // router straight to a bare `axum::serve(listener,
    // router).await?`. This both (a) lets in-flight requests
    // finish before the process exits and (b) lets the wave-2
    // trusted-proxy rate limiter's `MaybePeerAddr` extractor
    // (oxibonsai-runtime::middleware) see the real TCP peer
    // address instead of falling back to a single shared
    // "unknown" bucket for every direct client.
    oxibonsai_runtime::server::serve_with_shutdown(
        router,
        addr,
        oxibonsai_runtime::server::shutdown_signal(),
    )
    .await
    .map_err(anyhow::Error::from_boxed)?;

    Ok(())
}
