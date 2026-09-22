//! `oxibonsai serve` — OpenAI-compatible HTTP API server.
//!
//! Mirrors the hardening the standalone `oxibonsai-serve` binary applies
//! (`crates/oxibonsai-serve/src/main.rs`), adapted to the flags this
//! subcommand actually exposes (`src/cli/args.rs`'s `Serve` variant, owned
//! by CLI-CORE this wave — not touched here). Knobs with no CLI flag yet
//! (CORS, rate limiting, the admin token, `--insecure-no-auth`, the request
//! body ceiling) are read from `OXIBONSAI_*` environment variables instead;
//! see this package's recorded deviations for the exact flags CLI-CORE
//! should add to `args.rs` next.

use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::Router;
use oxibonsai_runtime::middleware::{apply_middleware, CorsConfig, MiddlewareConfig};
use oxibonsai_runtime::rate_limiter::{rate_limit_layer, RateLimitConfig};
use oxibonsai_runtime::serve_shared::verify_model_checksum;
use oxibonsai_runtime::server::{
    create_router_full, install_shutdown_signals, serve_with_shutdown,
    AuthConfig as AdminAuthConfig, RequestLimits, RouterOptions,
};

use super::admission;
use super::util::{build_sampling_params, missing_tokenizer_warning, resolve_tokenizer};

// ─── FIX2-SERVE item 6: the same resolved defaults `src/cli/mod.rs` uses ───
//
// `run`/`chat`/`benchmark`/`eval` resolve their `--temperature`/`--top-k`/
// `--top-p`/`--repetition-penalty` flags (all `Option<T>` in `args.rs`) to
// these exact literals when the operator passes none of them (see
// `src/cli/mod.rs`'s `util::resolve_f32(.., "sampling", "temperature", 0.7)`
// and siblings). This subcommand has no equivalent flags yet (B2-14, wave
// 4), so these constants stand in for "what the flag would have defaulted
// to" and are passed to the SAME shared `build_sampling_params` constructor,
// so a test can assert the two paths agree byte-for-byte.
const DEFAULT_SAMPLING_TEMPERATURE: f32 = 0.7;
const DEFAULT_SAMPLING_TOP_K: usize = 40;
const DEFAULT_SAMPLING_TOP_P: f32 = 0.9;
const DEFAULT_SAMPLING_REPETITION_PENALTY: f32 = 1.0;

/// The baseline `SamplingParams` this server builds its engine pool with,
/// before any per-request JSON-body override (FIX2-SERVE item 6). Routed
/// through the same shared `build_sampling_params` constructor CLI-CORE
/// added for `run`/`chat`/`benchmark`/`eval`, with the same resolved
/// defaults, instead of `SamplingParams::default()` (whose own
/// `repetition_penalty` is `1.1`). Factored out of [`run`] so it is
/// directly unit-testable against the run path's own resolved defaults
/// without needing a real GGUF model.
fn default_sampling_params() -> oxibonsai_runtime::sampling::SamplingParams {
    build_sampling_params(
        DEFAULT_SAMPLING_TEMPERATURE,
        DEFAULT_SAMPLING_TOP_K,
        DEFAULT_SAMPLING_TOP_P,
        DEFAULT_SAMPLING_REPETITION_PENALTY,
    )
}

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

    // SV-33: `--bearer-token` is visible to other local users via `ps` (and
    // lands in shell history) -- warn when it was the source, and let
    // `OXIBONSAI_BEARER_TOKEN_FILE` (ps-safe) override it. No
    // `--bearer-token-file` flag exists on this subcommand yet
    // (`src/cli/args.rs` is owned by CLI-CORE this wave -- see this
    // package's recorded deviations for the exact flag to add).
    let bearer_token_flag_used = bearer_token.is_some();
    let bearer_token = bearer_token.or_else(|| {
        std::env::var("OXIBONSAI_BEARER_TOKEN")
            .ok()
            .filter(|s| !s.is_empty())
    });
    let bearer_token = resolve_bearer_token_file_override(bearer_token)?;
    if bearer_token_flag_used {
        tracing::warn!(
            "--bearer-token was passed on the command line, which is visible to other local \
             users via `ps` (and lands in shell history); prefer OXIBONSAI_BEARER_TOKEN_FILE \
             or the OXIBONSAI_BEARER_TOKEN environment variable instead"
        );
    }

    // sec-M3 / SV-30: validate the three hardening flags this binary
    // accepts but historically checked none of (`oxibonsai-serve`'s
    // `validation.rs` already validated all three) -- a zero here used to
    // admit a permanently broken server (`--max-concurrent-requests 0`
    // rejects every request forever; `--request-timeout-ms 0` times every
    // request out immediately; a one-character `--bearer-token` is a
    // trivially guessable credential).
    admission::validate_serve_args(
        bearer_token.as_deref(),
        max_concurrent_requests,
        request_timeout_ms,
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    // sec-15 / SV-07 / sec-M2: refuse an unsafe bind. `--host` already
    // defaults to `127.0.0.1` (`src/cli/args.rs`, owned by CLI-CORE this
    // wave), so this specifically catches an *explicit* non-loopback host
    // with no auth configured.
    let insecure_no_auth = env_flag("OXIBONSAI_INSECURE_NO_AUTH");
    bind_safety_check(&host, bearer_token.as_deref(), insecure_no_auth)?;

    // sec-12: verify the model's checksum, if one is known.
    let checksums_path: PathBuf = std::env::var("OXIBONSAI_CHECKSUMS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("scripts/checksums.sha256"));
    verify_model_checksum(Path::new(&model), &checksums_path).map_err(|e| anyhow::anyhow!(e))?;

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

    // FIX2-SERVE item 6, the serve half of the orchestrator's P0
    // CPU-vs-Metal divergence fix: `oxibonsai_runtime::sampling::SamplingParams::default()`
    // carries `repetition_penalty: 1.1` -- a hidden, undocumented penalty
    // baked into this server's own greedy baseline, with no flag surface on
    // this subcommand to override it (no `--temperature`/`--top-k`/`--top-p`/
    // `--repetition-penalty` flags exist here yet; that is B2-14's, wave 4).
    // `default_sampling_params()` routes through the SAME shared
    // `build_sampling_params` constructor CLI-CORE already applied to
    // `run`/`chat`/`benchmark`/`eval`, with `repetition_penalty` resolved to
    // `1.0`, not the struct's own `1.1` -- so `oxibonsai serve`'s baseline
    // sampling config is byte-for-byte what `oxibonsai run`/`chat` resolve
    // to when no CLI flag overrides them, and "greedy means greedy" (a
    // per-request `temperature: 0` in the JSON body) now holds on
    // `oxibonsai serve` too.
    let params = default_sampling_params();
    let metrics = Arc::new(oxibonsai_runtime::InferenceMetrics::new());

    // cli-M5: default to a pseudo-random seed instead of the hardcoded 42
    // (no `--seed` flag exists on this subcommand yet -- see the module
    // docs -- so `OXIBONSAI_SEED` is the interim override).
    let seed = resolve_seed();
    tracing::info!(seed, "resolved RNG seed");

    // Build a pool of engine replicas sharing one leaked `'static`
    // GGUF (replica #1 mmaps + leaks; the rest reuse it zero-copy).
    let (pool, _tier, _size) = oxibonsai_runtime::engine_pool::build_pool_from_gguf(
        &model,
        params,
        seed,
        max_seq_len,
        requested_pool_size,
    )?;
    // Wire the shared metrics onto every replica, preserving the
    // per-engine telemetry the single-engine path recorded.
    pool.set_metrics_all(&metrics)?;
    // sec-20/perf-M1: capture the REAL pool size before `pool` is consumed
    // by `create_router_full` below.
    let pool_size_actual = pool.size();

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

    // sec-15: admin auth is independent of the inference bearer token.
    // FIX2-SERVE item 4 (admin-token env asymmetry): honour
    // `OXIBONSAI_ADMIN_TOKEN` first -- matching
    // `crates/oxibonsai-serve/src/hardening.rs::resolve_admin_auth` (via
    // that binary's `apply_extra_env_overrides`) -- then fall back to
    // `OXI_ADMIN_TOKEN` (`AuthConfig::from_env`, the pre-existing
    // convention every router constructor already honors). Before this
    // fix, an operator who set `OXIBONSAI_ADMIN_TOKEN` got a token-gated
    // `/admin` on the standalone `oxibonsai-serve` binary and a
    // 403-locked `/admin` on this one (`oxibonsai serve`).
    let admin_auth = resolve_admin_auth();

    // SV-15/SV-16/SV-23: thread `max_seq_len` through as the admission-time
    // `max_input_tokens` ceiling too (mirroring `oxibonsai-serve`'s single
    // `limits.max_input_tokens` field, which serves both the KV-cache size
    // and the per-request prompt budget), and `request_timeout_ms` as the
    // per-request deadline the chat handlers already consult once it is
    // non-`None`.
    let router_options = RouterOptions::default()
        .with_limits(
            RequestLimits::default()
                .with_max_input_tokens(Some(max_seq_len))
                .with_timeout_ms(request_timeout_ms),
        )
        .with_auth(admin_auth);

    #[cfg_attr(not(feature = "rag"), allow(unused_mut))]
    let mut router =
        create_router_full(Arc::clone(&pool), tok, Arc::clone(&metrics), router_options);

    #[cfg(feature = "rag")]
    if rag {
        let rag_tok = load_tok()?;
        tracing::info!("mounting RAG HTTP API (/rag/index, /rag/query, /rag/stats)");
        router = router.merge(oxibonsai_runtime::rag_server::create_rag_router_with_pool(
            Arc::clone(&pool),
            rag_tok,
        ));
    }

    // ── Hardening: body limit + admission + rate limit + bearer auth + CORS ──
    let opts = HardeningOptions {
        bearer_token,
        max_concurrent_requests,
        request_timeout_ms,
        max_body_bytes: env_usize("OXIBONSAI_MAX_BODY_BYTES").unwrap_or(4 * 1024 * 1024),
        cors_origins: env_cors_origins(),
        cors_allow_credentials: env_flag("OXIBONSAI_CORS_ALLOW_CREDENTIALS"),
        rate_limit_rpm: env_f64("OXIBONSAI_RATE_LIMIT_RPM"),
        rate_limit_burst: env_f64("OXIBONSAI_RATE_LIMIT_BURST").unwrap_or(20.0),
    };
    let router = harden_router(router, pool_size_actual, &opts, &host);

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
    //
    // SV-29: a failed signal-handler registration now refuses to start
    // instead of degrading (`install_shutdown_signals` propagates the
    // `io::Error`), rather than the old `shutdown_signal()` which only
    // ever logged and fell back silently.
    let signals = install_shutdown_signals()?;
    serve_with_shutdown(router, addr, signals)
        .await
        .map_err(anyhow::Error::from_boxed)?;

    Ok(())
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
/// routes -> admission (concurrency+timeout) -> DefaultBodyLimit
///        -> rate-limit -> bearer-auth -> CORS (outermost)
/// ```
///
/// (FIX2-SERVE item 3: `DefaultBodyLimit` moved from inside `admission` to
/// outside it, still inside rate-limit, mirroring
/// `crates/oxibonsai-serve/src/hardening.rs::build_router`'s matching fix —
/// see the note at its call site below for what this reorder does and does
/// not achieve.)
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
    // FIX2-SERVE item 3: applied here, BEFORE `DefaultBodyLimit` below, so
    // `DefaultBodyLimit` ends up mounted OUTSIDE (more outer than) this
    // admission stack.
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
    // FIX2-SERVE item 3: moved from *inside* `admission` (above) to
    // *outside* it (still inside rate-limit, below). This reorder is
    // correctness-neutral for the permit-holding concern the finding named
    // -- `axum`'s `DefaultBodyLimit` only inserts a request extension
    // consulted later by the `Bytes`/`Json` extractors deep inside the
    // eventual handler, unconditionally calling its inner service with no
    // check of its own at either position -- but it is the right relative
    // position for when a real synchronous body-size guard is added outside
    // `admission`; see this package's recorded deviations, and the fuller
    // version of this note on `crates/oxibonsai-serve/src/hardening.rs::build_router`.
    router = router.layer(DefaultBodyLimit::max(opts.max_body_bytes));

    // sec-07/RT-34/SV-10/cli-10: rate limiting, mounted *outside* admission
    // so a rate-limited request never consumes a concurrency permit
    // (cli-18). No `--rate-limit-rpm`/`--rate-limit-burst` flag exists on
    // this subcommand yet (see the module docs), so this is
    // `OXIBONSAI_RATE_LIMIT_RPM`/`_BURST`-only for now.
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
            // sec-M2's correction, `REQUIRED #6`: this used to claim
            // `/admin/*` was unauthenticated too, which stopped being true
            // once `create_router_full` started gating it unconditionally
            // (403 without a configured admin token) -- state the real
            // sec-15 behavior instead.
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
    // ever sees the request. No `--cors-origin` flag exists on this
    // subcommand yet (see the module docs), so this is
    // `OXIBONSAI_CORS_ORIGIN`-only for now.
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
/// cannot be depended on from here -- see this package's recorded
/// deviations).
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

// ─── FIX2-SERVE item 4: admin-token env asymmetry between the two binaries ──

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
/// [`resolve_admin_auth_from`] below (verifier follow-up on this same
/// finding -- see its doc comment for why the split exists).
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

/// Override `flag_or_env` with the contents of `OXIBONSAI_BEARER_TOKEN_FILE`
/// (read and trimmed) when that variable is set — the `ps`-safe option
/// finding `SV-33` recommends. Returns `flag_or_env` unchanged when the
/// variable is unset.
fn resolve_bearer_token_file_override(
    flag_or_env: Option<String>,
) -> anyhow::Result<Option<String>> {
    let Ok(path) = std::env::var("OXIBONSAI_BEARER_TOKEN_FILE") else {
        return Ok(flag_or_env);
    };
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("failed to read OXIBONSAI_BEARER_TOKEN_FILE {path}: {e}"))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(anyhow::anyhow!(
            "OXIBONSAI_BEARER_TOKEN_FILE {path} is empty"
        ));
    }
    Ok(Some(trimmed.to_string()))
}

// ─── cli-M5: pseudo-random default seed ─────────────────────────────────────

/// Resolve the RNG seed: an `OXIBONSAI_SEED` env override, else a
/// process/time-derived pseudo-random value (finding `cli-M5`: the previous
/// literal `42` never varied). No `--seed` flag exists on this subcommand
/// yet (`src/cli/args.rs`'s `Serve` variant is owned by CLI-CORE this wave
/// -- see this package's recorded deviations for the exact flag to add),
/// and no `rand` crate is reachable from this crate without a new Cargo
/// dependency this package cannot add (`Cargo.toml` is not in its
/// `owned_files`). Not cryptographically random; it only needs to differ
/// from one server start to the next.
fn resolve_seed() -> u64 {
    if let Ok(v) = std::env::var("OXIBONSAI_SEED") {
        if let Ok(seed) = v.parse::<u64>() {
            return seed;
        }
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

// ─── OXIBONSAI_* env var helpers (no --flag surface yet; see module docs) ──

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
mod tests {
    use super::*;

    // ── is_loopback_host / bind_safety_check ────────────────────────────────

    #[test]
    fn loopback_hosts_are_recognized() {
        for host in ["127.0.0.1", "127.5.5.5", "::1", "localhost", "LOCALHOST"] {
            assert!(is_loopback_host(host), "{host} should be loopback");
        }
    }

    #[test]
    fn non_loopback_hosts_are_not_loopback() {
        for host in ["0.0.0.0", "192.168.1.1", "example.com", ""] {
            assert!(!is_loopback_host(host), "{host} should not be loopback");
        }
    }

    #[test]
    fn loopback_bind_never_needs_auth() {
        bind_safety_check("127.0.0.1", None, false).expect("loopback bind is always safe");
    }

    #[test]
    fn non_loopback_bind_without_auth_is_refused() {
        let err = bind_safety_check("0.0.0.0", None, false).expect_err("must refuse");
        assert!(err.to_string().contains("0.0.0.0"));
    }

    #[test]
    fn non_loopback_bind_with_bearer_token_is_allowed() {
        bind_safety_check("0.0.0.0", Some(&"x".repeat(20)), false)
            .expect("a configured bearer token makes this safe");
    }

    #[test]
    fn non_loopback_bind_with_insecure_no_auth_is_allowed() {
        bind_safety_check("0.0.0.0", None, true).expect("explicit opt-out is honored");
    }

    // ── resolve_admin_auth_from (FIX2-SERVE item 4: admin-token env asymmetry) ──
    //
    // Rewritten as a verifier follow-up on this same finding: these used to
    // exercise `resolve_admin_auth` directly by mutating the two SAME
    // process-global env vars (`OXIBONSAI_ADMIN_TOKEN`, `OXI_ADMIN_TOKEN`)
    // via `unsafe { std::env::set_var / remove_var }` with no
    // serialization -- flaky against each other under `cargo test`'s
    // default parallel test threads, and unsound against `router_tests`'
    // concurrent `getenv` on `OXI_ADMIN_TOKEN` (see `resolve_admin_auth_from`'s
    // doc comment above for the full explanation). They now call the pure
    // `resolve_admin_auth_from` with both the candidate token and the
    // pre-resolved legacy policy as plain arguments, so none of them touch
    // either environment variable at all -- and each asserts a distinct,
    // concrete expected outcome rather than delegating to a fresh
    // `AdminAuthConfig::from_env()` call, so a broken fallback arm cannot
    // hide behind `OXI_ADMIN_TOKEN` happening to be unset in the test
    // environment.

    #[test]
    fn resolve_admin_auth_from_prefers_new_token_over_legacy() {
        let resolved = resolve_admin_auth_from(
            Some("new-convention-token-value"),
            AdminAuthConfig::with_admin_token("legacy-convention-token-value"),
        );
        assert_eq!(
            resolved,
            AdminAuthConfig::with_admin_token("new-convention-token-value")
        );
    }

    #[test]
    fn resolve_admin_auth_from_falls_back_to_legacy_when_new_token_is_none() {
        let resolved = resolve_admin_auth_from(
            None,
            AdminAuthConfig::with_admin_token("legacy-convention-token-value"),
        );
        assert_eq!(
            resolved,
            AdminAuthConfig::with_admin_token("legacy-convention-token-value")
        );
    }

    #[test]
    fn resolve_admin_auth_from_falls_back_to_legacy_when_new_token_is_empty() {
        let resolved = resolve_admin_auth_from(
            Some(""),
            AdminAuthConfig::with_admin_token("legacy-convention-token-value"),
        );
        assert_eq!(
            resolved,
            AdminAuthConfig::with_admin_token("legacy-convention-token-value")
        );
    }

    #[test]
    fn resolve_admin_auth_from_locked_when_neither_is_configured() {
        let resolved = resolve_admin_auth_from(None, AdminAuthConfig::locked());
        assert_eq!(resolved, AdminAuthConfig::locked());
    }

    #[test]
    fn resolve_admin_auth_from_ignores_legacy_when_new_token_is_whitespace_only() {
        // A previously-untested edge case documented on
        // `resolve_admin_auth_from`'s doc comment: a whitespace-only new
        // token is non-empty (so it does NOT hit the fallback arm), but
        // `AdminAuthConfig::with_admin_token` itself treats whitespace-only
        // as "not configured", so this locks even though a legacy token is
        // available -- unlike an *empty* new token
        // (`resolve_admin_auth_from_falls_back_to_legacy_when_new_token_is_empty`
        // above), which does fall back to it.
        let resolved = resolve_admin_auth_from(
            Some("   "),
            AdminAuthConfig::with_admin_token("legacy-convention-token-value"),
        );
        assert_eq!(resolved, AdminAuthConfig::locked());
    }

    // ── default_sampling_params (FIX2-SERVE item 6: greedy penalties) ───────

    /// THE regression test for the serve half of the P0 CPU-vs-Metal
    /// divergence: `oxibonsai serve`'s baseline `SamplingParams` must match
    /// what `oxibonsai run`/`chat`/`benchmark`/`eval` resolve to for the
    /// same (default, no-flag) inputs -- both now go through the identical
    /// shared `build_sampling_params` constructor. `SamplingParams` carries
    /// no `PartialEq` (it is owned by `oxibonsai-runtime`, not this
    /// package), so fields are compared individually.
    ///
    /// KNOWN LIMITATION (verifier-noted, acknowledged rather than papered
    /// over): `default_sampling_params()` above is *itself* defined as
    /// `build_sampling_params(DEFAULT_SAMPLING_TEMPERATURE, ...)`, so
    /// comparing it here against `build_sampling_params(0.7, 40, 0.9, 1.0)`
    /// feeds the identical pure constructor the identical literals on both
    /// sides and cannot catch `src/cli/mod.rs`'s own independent
    /// `--temperature`/`--top-k`/`--top-p`/`--repetition-penalty` flag
    /// resolution drifting away from these values -- `mod.rs` is not in
    /// this package's `owned_files`, and it inlines its fallback literals
    /// directly into `util::resolve_f32(..)` calls with no exported
    /// constant this test could import instead. Asserting the run-path
    /// literals explicitly (e.g. `run_path_params.temperature == 0.7`)
    /// would not close this gap either: `build_sampling_params` is a pure
    /// field-literal passthrough (`src/cli/util.rs`), so such an assertion
    /// would be tautological by construction, not a real check. What this
    /// test DOES verify, non-tautologically: `default_sampling_params()`'s
    /// own `DEFAULT_SAMPLING_*` constants (this module) have not drifted
    /// from what they are supposed to mirror, and -- the concrete
    /// regression at the bottom of this test -- that the server's baseline
    /// never again silently reverts to `SamplingParams::default()`'s
    /// hidden `repetition_penalty` of `1.1`. A future owner of `mod.rs`
    /// closing the remaining gap would export its literals as named
    /// constants for this test to import.
    #[test]
    fn serve_default_sampling_params_match_the_run_path_for_the_same_inputs() {
        let serve_params = default_sampling_params();
        // "The run path, same inputs": the literal defaults `src/cli/mod.rs`
        // resolves `run`'s `--temperature`/`--top-k`/`--top-p`/
        // `--repetition-penalty` flags to when none of them are passed.
        let run_path_params = build_sampling_params(0.7, 40, 0.9, 1.0);

        assert!(
            (serve_params.temperature - run_path_params.temperature).abs() < f32::EPSILON,
            "temperature mismatch: serve={} run={}",
            serve_params.temperature,
            run_path_params.temperature
        );
        assert_eq!(serve_params.top_k, run_path_params.top_k);
        assert!(
            (serve_params.top_p - run_path_params.top_p).abs() < f32::EPSILON,
            "top_p mismatch: serve={} run={}",
            serve_params.top_p,
            run_path_params.top_p
        );
        assert!(
            (serve_params.repetition_penalty - run_path_params.repetition_penalty).abs()
                < f32::EPSILON,
            "repetition_penalty mismatch: serve={} run={}",
            serve_params.repetition_penalty,
            run_path_params.repetition_penalty
        );
        assert_eq!(serve_params.max_tokens, run_path_params.max_tokens);

        // The concrete regression this guards against: the server's
        // baseline must never again silently carry
        // `SamplingParams::default()`'s hidden `repetition_penalty` of
        // `1.1` -- "greedy means greedy" (a per-request `temperature: 0`)
        // must hold on `oxibonsai serve` exactly as it does on `oxibonsai
        // run`.
        assert!((serve_params.repetition_penalty - 1.0).abs() < f32::EPSILON);
    }

    // ── resolve_seed ─────────────────────────────────────────────────────────

    #[test]
    fn resolve_seed_env_override_wins() {
        // SAFETY (test-only): `std::env::set_var`/`remove_var` are process
        // wide; this test is careful to always restore the prior value so
        // it cannot leak into a sibling test running in the same process.
        let prior = std::env::var("OXIBONSAI_SEED").ok();
        unsafe {
            std::env::set_var("OXIBONSAI_SEED", "123456789");
        }
        let seed = resolve_seed();
        match prior {
            Some(v) => unsafe { std::env::set_var("OXIBONSAI_SEED", v) },
            None => unsafe { std::env::remove_var("OXIBONSAI_SEED") },
        }
        assert_eq!(seed, 123_456_789);
    }

    #[test]
    fn resolve_seed_ignores_malformed_override() {
        let prior = std::env::var("OXIBONSAI_SEED").ok();
        unsafe {
            std::env::set_var("OXIBONSAI_SEED", "not-a-number");
        }
        // Falls back to the pseudo-random (time-derived) path; two calls a
        // moment apart must not be a hardcoded constant. This is a real
        // assertion (not a `!= 42` probabilistic guess): the nanosecond
        // component alone makes two calls separated by any measurable delay
        // collide with astronomically low probability, and a constant
        // fallback -- the actual regression this guards against -- would
        // fail it deterministically.
        let first = resolve_seed();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let second = resolve_seed();
        match prior {
            Some(v) => unsafe { std::env::set_var("OXIBONSAI_SEED", v) },
            None => unsafe { std::env::remove_var("OXIBONSAI_SEED") },
        }
        assert_ne!(
            first, second,
            "the pseudo-random fallback must not be a constant"
        );
    }

    // ── resolve_bearer_token_file_override (SV-33) ──────────────────────────

    #[test]
    fn bearer_token_file_override_passes_through_when_unset() {
        let prior = std::env::var("OXIBONSAI_BEARER_TOKEN_FILE").ok();
        unsafe {
            std::env::remove_var("OXIBONSAI_BEARER_TOKEN_FILE");
        }
        let result = resolve_bearer_token_file_override(Some("flag-value".to_string()))
            .expect("unset var must not error");
        if let Some(v) = prior {
            unsafe { std::env::set_var("OXIBONSAI_BEARER_TOKEN_FILE", v) };
        }
        assert_eq!(result, Some("flag-value".to_string()));
    }

    #[test]
    fn bearer_token_file_override_wins_and_is_trimmed() {
        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_cli_bearer_token_file_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("token");
        std::fs::write(&path, "  from-file  \n").expect("write token file");

        let prior = std::env::var("OXIBONSAI_BEARER_TOKEN_FILE").ok();
        unsafe {
            std::env::set_var(
                "OXIBONSAI_BEARER_TOKEN_FILE",
                path.to_string_lossy().as_ref(),
            );
        }
        let result = resolve_bearer_token_file_override(Some("flag-value".to_string()));
        match prior {
            Some(v) => unsafe { std::env::set_var("OXIBONSAI_BEARER_TOKEN_FILE", v) },
            None => unsafe { std::env::remove_var("OXIBONSAI_BEARER_TOKEN_FILE") },
        }
        assert_eq!(result.expect("resolve"), Some("from-file".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bearer_token_file_override_errors_on_missing_file() {
        // Never a hardcoded absolute path (project policy): a process-unique
        // path under the real temp dir, mirroring the sibling test in
        // `crates/oxibonsai-serve/src/hardening.rs`
        // (`resolve_bearer_token_errors_when_file_is_unreadable`).
        let missing = std::env::temp_dir().join(format!(
            "oxibonsai_cli_definitely_missing_bearer_token_file_{}",
            std::process::id()
        ));
        let prior = std::env::var("OXIBONSAI_BEARER_TOKEN_FILE").ok();
        unsafe {
            std::env::set_var("OXIBONSAI_BEARER_TOKEN_FILE", &missing);
        }
        let result = resolve_bearer_token_file_override(None);
        match prior {
            Some(v) => unsafe { std::env::set_var("OXIBONSAI_BEARER_TOKEN_FILE", v) },
            None => unsafe { std::env::remove_var("OXIBONSAI_BEARER_TOKEN_FILE") },
        }
        assert!(result.is_err());
    }

    // ── checksum helpers ─────────────────────────────────────────────────────
    //
    // `lookup_expected_checksum` has no non-test call site left in this
    // *binary* target now that `verify_model_checksum`'s implementation
    // lives in `oxibonsai_runtime::serve_shared` -- imported here, inside
    // `#[cfg(test)]`, rather than at module level, so a non-test build of
    // this bin target does not see (and cannot warn about) an otherwise
    // -unused import.
    use oxibonsai_runtime::serve_shared::lookup_expected_checksum;

    #[test]
    fn lookup_matches_by_bare_filename() {
        let text =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  models/Foo.gguf\n";
        let got = lookup_expected_checksum(text, Path::new("/elsewhere/Foo.gguf"));
        assert_eq!(
            got,
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string())
        );
    }

    #[test]
    fn lookup_is_none_for_unknown_file() {
        let text =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  models/Foo.gguf\n";
        assert!(lookup_expected_checksum(text, Path::new("models/Unknown.gguf")).is_none());
    }

    #[test]
    fn verify_is_a_noop_when_checksums_file_is_absent() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "oxibonsai_cli_checksum_test_{}",
            std::process::id()
        ));
        let missing_checksums = tmp_dir.join("does-not-exist.sha256");
        let model = tmp_dir.join("Model.gguf");
        verify_model_checksum(&model, &missing_checksums)
            .expect("missing checksums file must not block loading");
    }

    // NOTE (sec-12 verifier finding): the previous version of this test
    // ("verify_warns_but_does_not_fail_when_hasher_is_unavailable") never
    // actually wrote a model file, only the checksums entry -- so it
    // exercised `verify_model_checksum`'s `Err(_)` (unreadable file)
    // branch, not an "unavailable hasher" branch, regardless of what
    // `compute_sha256_hex` returned. It is replaced below by tests that
    // exercise the real equality/mismatch branches end to end through this
    // binary's own imported `verify_model_checksum`, plus one correctly
    // named/scoped test for the unreadable-file case it was accidentally
    // covering.

    #[test]
    fn verify_succeeds_when_the_checksum_really_matches() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "oxibonsai_cli_checksum_test_matches_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let model = tmp_dir.join("Model.gguf");
        std::fs::write(&model, b"the real bytes of a model file").expect("write model");
        let hex = oxibonsai_runtime::serve_shared::compute_sha256_hex(&model)
            .expect("compute hex")
            .expect("digest is always Some for a readable file");
        let checksums_path = tmp_dir.join("checksums.sha256");
        std::fs::write(
            &checksums_path,
            format!(
                "{hex}  {}\n",
                model.file_name().expect("filename").to_string_lossy()
            ),
        )
        .expect("write checksums file");

        verify_model_checksum(&model, &checksums_path)
            .expect("a genuinely matching checksum must not block loading");
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    /// THE regression test finding `sec-12` was missing: a known-good
    /// checksum entry that no longer matches the file on disk must be
    /// **fatal** -- this is `verify_model_checksum`'s one fatal branch, and
    /// it was unreachable dead code while `compute_sha256_hex` always
    /// returned `Ok(None)`.
    #[test]
    fn verify_fails_when_a_known_good_entry_no_longer_matches_a_corrupted_file() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "oxibonsai_cli_checksum_test_corrupted_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let model = tmp_dir.join("Model.gguf");

        std::fs::write(&model, b"the original, known-good model bytes").expect("write model");
        let good_hex = oxibonsai_runtime::serve_shared::compute_sha256_hex(&model)
            .expect("compute hex")
            .expect("digest is always Some for a readable file");
        let checksums_path = tmp_dir.join("checksums.sha256");
        std::fs::write(
            &checksums_path,
            format!(
                "{good_hex}  {}\n",
                model.file_name().expect("filename").to_string_lossy()
            ),
        )
        .expect("write checksums file");

        // Corrupt the file in place; the checksums file still lists the
        // ORIGINAL (now stale) digest as "known-good".
        std::fs::write(
            &model,
            b"corrupted! these are not the original bytes at all",
        )
        .expect("corrupt model");

        let err = verify_model_checksum(&model, &checksums_path)
            .expect_err("a stale known-good entry against corrupted bytes must be fatal");
        assert!(
            err.contains("checksum mismatch"),
            "error should explain the mismatch: {err}"
        );
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn verify_degrades_gracefully_when_the_model_file_cannot_be_opened() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "oxibonsai_cli_checksum_test_missing_model_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let model = tmp_dir.join("Model.gguf"); // Deliberately never created.
        let checksums_path = tmp_dir.join("checksums.sha256");
        std::fs::write(&checksums_path, format!("{}  Model.gguf\n", "c".repeat(64)))
            .expect("write checksums file");
        verify_model_checksum(&model, &checksums_path)
            .expect("an unreadable model file must not block loading (degrades to a warning)");
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    // ── env helpers ──────────────────────────────────────────────────────────

    #[test]
    fn env_cors_origins_splits_and_trims_when_set() {
        let prior = std::env::var("OXIBONSAI_CORS_ORIGIN").ok();
        unsafe {
            std::env::set_var(
                "OXIBONSAI_CORS_ORIGIN",
                "https://a.example.com, https://b.example.com ,",
            );
        }
        let origins = env_cors_origins();
        match prior {
            Some(v) => unsafe { std::env::set_var("OXIBONSAI_CORS_ORIGIN", v) },
            None => unsafe { std::env::remove_var("OXIBONSAI_CORS_ORIGIN") },
        }
        assert_eq!(
            origins,
            vec![
                "https://a.example.com".to_string(),
                "https://b.example.com".to_string()
            ]
        );
    }

    #[test]
    fn env_flag_accepts_common_spellings() {
        for (raw, expected) in [
            ("1", true),
            ("true", true),
            ("YES", true),
            ("on", true),
            ("0", false),
            ("false", false),
            ("no", false),
            ("garbage", false),
        ] {
            let prior = std::env::var("OXIBONSAI_TEST_FLAG_CMD_SERVE").ok();
            unsafe {
                std::env::set_var("OXIBONSAI_TEST_FLAG_CMD_SERVE", raw);
            }
            let got = env_flag("OXIBONSAI_TEST_FLAG_CMD_SERVE");
            match &prior {
                Some(v) => unsafe { std::env::set_var("OXIBONSAI_TEST_FLAG_CMD_SERVE", v) },
                None => unsafe { std::env::remove_var("OXIBONSAI_TEST_FLAG_CMD_SERVE") },
            }
            assert_eq!(got, expected, "raw={raw:?}");
        }
    }

    // ── harden_router: full composition against a tiny in-memory pool ──────

    mod router_tests {
        use super::*;
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use oxibonsai_core::config::Qwen3Config;
        use oxibonsai_runtime::engine::InferenceEngine;
        use oxibonsai_runtime::engine_pool::EnginePool;
        use oxibonsai_runtime::metrics::InferenceMetrics;
        use oxibonsai_runtime::sampling::SamplingParams;
        use std::time::Duration;
        use tower::ServiceExt;

        fn default_opts() -> HardeningOptions {
            HardeningOptions {
                bearer_token: None,
                max_concurrent_requests: 32,
                request_timeout_ms: 60_000,
                max_body_bytes: 4 * 1024 * 1024,
                cors_origins: Vec::new(),
                cors_allow_credentials: false,
                rate_limit_rpm: None,
                rate_limit_burst: 20.0,
            }
        }

        fn router_for(opts: HardeningOptions) -> Router {
            let engine =
                InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
            let pool = EnginePool::new(vec![engine]);
            let pool_size = pool.size();
            let metrics = Arc::new(InferenceMetrics::new());
            let router_options = RouterOptions::default().with_auth(AdminAuthConfig::locked());
            let base = create_router_full(pool, None, metrics, router_options);
            harden_router(base, pool_size, &opts, "127.0.0.1")
        }

        #[tokio::test]
        async fn health_is_reachable_with_no_config() {
            let router = router_for(default_opts());
            let resp = router
                .oneshot(
                    Request::get("/health")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(resp.status(), StatusCode::OK);
        }

        #[tokio::test]
        async fn admin_is_refused_by_default() {
            let router = router_for(default_opts());
            let resp = router
                .oneshot(
                    Request::get("/admin/status")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        }

        #[tokio::test]
        async fn bearer_auth_protects_inference_routes_when_configured() {
            let mut opts = default_opts();
            opts.bearer_token = Some("x".repeat(20));
            let router = router_for(opts);

            let unauth = router
                .clone()
                .oneshot(
                    Request::get("/v1/models")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

            let health = router
                .oneshot(
                    Request::get("/health")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(health.status(), StatusCode::OK, "/health must stay exempt");
        }

        /// cli-18: two consecutive unauthenticated requests must be
        /// rejected identically. **Not discriminating on its own**: both
        /// requests run sequentially through `oneshot`, so the sole permit
        /// (if bearer auth were ever mounted inside the admission stack)
        /// would already be released before the second request starts, and
        /// this assertion would hold under BOTH layer orderings. Kept as a
        /// cheap idempotency check; `unauthenticated_request_is_401_while_the_sole_permit_is_held`
        /// below is the test that actually proves the layer ordering.
        #[tokio::test]
        async fn unauthenticated_request_never_reaches_admission() {
            let mut opts = default_opts();
            opts.bearer_token = Some("x".repeat(20));
            opts.max_concurrent_requests = 1;
            let router = router_for(opts);

            for _ in 0..2 {
                let resp = router
                    .clone()
                    .oneshot(
                        Request::get("/v1/models")
                            .body(Body::empty())
                            .expect("request"),
                    )
                    .await
                    .expect("response");
                assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
            }
        }

        /// VERIFIER-ADDED (cli-18), ported from the discriminating version
        /// added to `crates/oxibonsai-serve/src/hardening.rs`'s
        /// `build_router_tests`: the sibling
        /// `unauthenticated_request_never_reaches_admission` above drives
        /// its two requests sequentially through `oneshot`, so the permit
        /// is released before the second request starts and the assertion
        /// holds under BOTH layer orderings -- it does not actually
        /// discriminate. Here a slow *authenticated* chat completion is
        /// kept in flight on a separate task, holding the sole concurrency
        /// permit (effective ceiling = `resolve_admission_limit(1, 1)` =
        /// `1`); an unauthenticated request issued while it runs must still
        /// be `401`. If bearer auth were mounted INSIDE the admission
        /// stack, `load_shed` would answer `503` instead.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn unauthenticated_request_is_401_while_the_sole_permit_is_held() {
            let mut opts = default_opts();
            opts.bearer_token = Some("x".repeat(20));
            opts.max_concurrent_requests = 1;
            let router = router_for(opts);

            let slow_router = router.clone();
            let slow = tokio::spawn(async move {
                let payload = serde_json::json!({
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 500,
                });
                let req = Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {}", "x".repeat(20)))
                    .body(Body::from(payload.to_string()))
                    .expect("request");
                slow_router.oneshot(req).await.map(|r| r.status())
            });

            tokio::time::sleep(Duration::from_millis(30)).await;
            let resp = router
                .oneshot(
                    Request::get("/v1/models")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            let slow_status = slow.await.expect("join").expect("slow response");
            assert_eq!(
                slow_status,
                StatusCode::OK,
                "the authenticated request must have been the one holding the permit"
            );
            assert_eq!(
                resp.status(),
                StatusCode::UNAUTHORIZED,
                "an unauthenticated request must be 401 even while the sole admission permit \
                 is held (a 503 means bearer auth sits INSIDE the admission stack)"
            );
        }

        /// sec-20/perf-M1: the `503` an overloaded admission layer returns
        /// must carry `Retry-After` -- previously an omission (only the
        /// `429` rate-limit path below set it). Same permit-holding trick
        /// as `unauthenticated_request_is_401_while_the_sole_permit_is_held`,
        /// but with no bearer token configured, so the second request
        /// actually reaches (and is shed by) the admission layer instead of
        /// being rejected by auth first.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn overloaded_request_gets_503_with_retry_after() {
            let mut opts = default_opts();
            opts.max_concurrent_requests = 1;
            let router = router_for(opts);

            let slow_router = router.clone();
            let slow = tokio::spawn(async move {
                let payload = serde_json::json!({
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 500,
                });
                let req = Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .expect("request");
                slow_router.oneshot(req).await.map(|r| r.status())
            });

            tokio::time::sleep(Duration::from_millis(30)).await;
            let resp = router
                .oneshot(
                    Request::get("/v1/models")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            let slow_status = slow.await.expect("join").expect("slow response");
            assert_eq!(
                slow_status,
                StatusCode::OK,
                "the in-flight request must have been the one holding the sole permit"
            );
            assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert!(
                resp.headers().get("retry-after").is_some(),
                "a 503 from the admission layer must carry Retry-After"
            );
        }

        #[tokio::test]
        async fn rate_limit_returns_429_with_retry_after() {
            let mut opts = default_opts();
            opts.rate_limit_rpm = Some(60.0);
            opts.rate_limit_burst = 1.0;
            let router = router_for(opts);

            let first = router
                .clone()
                .oneshot(
                    Request::get("/v1/models")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(first.status(), StatusCode::OK);

            let second = router
                .oneshot(
                    Request::get("/v1/models")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
            assert!(second.headers().get("retry-after").is_some());
        }

        #[tokio::test]
        async fn cors_preflight_bypasses_bearer_auth() {
            let mut opts = default_opts();
            opts.bearer_token = Some("x".repeat(20));
            opts.cors_origins = vec!["https://app.example.com".to_string()];
            let router = router_for(opts);

            let preflight = Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("origin", "https://app.example.com")
                .body(Body::empty())
                .expect("preflight request");
            let resp = router.oneshot(preflight).await.expect("response");
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "an OPTIONS preflight must bypass bearer auth entirely (SV-06)"
            );
        }

        #[tokio::test]
        async fn no_cors_configured_emits_no_cors_headers() {
            let router = router_for(default_opts());
            let resp = router
                .oneshot(
                    Request::get("/health")
                        .header("origin", "https://anything.example.com")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert!(resp.headers().get("access-control-allow-origin").is_none());
        }

        #[tokio::test]
        async fn oversized_body_is_rejected_with_413() {
            let mut opts = default_opts();
            opts.max_body_bytes = 16;
            let router = router_for(opts);

            let body = Body::from(vec![b'a'; 4096]);
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(body)
                .expect("request");
            let resp = router.oneshot(req).await.expect("response");
            assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        }

        /// FIX2-SERVE item 3, ported from the sibling test added to
        /// `crates/oxibonsai-serve/src/hardening.rs`'s `build_router_tests`:
        /// `DefaultBodyLimit` was reordered from inside the admission stack
        /// to outside it (still inside rate-limiting). **Not discriminating
        /// on its own** (same caveat as
        /// `unauthenticated_request_never_reaches_admission` above for a
        /// different layer pair): `axum::extract::DefaultBodyLimit` only
        /// inserts a request extension consulted later by the handler's own
        /// extractor and never rejects anything itself at either position,
        /// so this only proves the ceiling still applies once
        /// `max_concurrent_requests` is tightened to its minimum (1), and
        /// that the response is genuinely `413`, not a `408`/`504` timeout
        /// from admission's own `.timeout(..)`.
        #[tokio::test]
        async fn oversized_body_is_rejected_with_413_even_with_a_single_permit() {
            let mut opts = default_opts();
            opts.max_body_bytes = 16;
            opts.max_concurrent_requests = 1;
            let router = router_for(opts);

            let body = Body::from(vec![b'a'; 4096]);
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(body)
                .expect("request");
            let resp = router.oneshot(req).await.expect("response");
            assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        }
    }
}
