//! Server hardening regressions (round 4).
//!
//! One test per verified finding, exercised through the real router built by
//! `create_router*` wherever the defect is observable over HTTP:
//!
//! | finding | what is asserted |
//! |---|---|
//! | `sec-01` / `TOK-M2` | the prompt sanitizer is linear, and no `<|` survives it |
//! | `sec-03` | a second request is served while a long generation is in flight |
//! | `sec-05` | an over-budget prompt is a `400` naming the real numbers, not a `500` |
//! | `sec-08` | SSE carries the usage chunk and always terminates with `[DONE]`; the router's deadline fires for a held engine, plain (`504`) and streamed (`event: error`), on all three generation routes, cancelling the generation; a stream queued for a replica times out before it opens; the admission layer's backstop is never reached on those routes |
//! | `sec-15` | `/admin/*` is refused without the admin credential |
//! | `SV-04` | every error path carries the JSON envelope and `X-Request-ID` |
//! | `SV-05` | a malformed body is a JSON envelope, not axum's plain text |
//! | `SV-29` | signal installation is fallible and reported, never `expect()`ed |
//! | `RT-03` | two identical greedy requests produce identical output |

#![cfg(feature = "server")]

use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_control::RecurrentState;
use oxibonsai_runtime::engine_pool::EnginePool;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{
    create_router_full, create_router_with_auth, install_shutdown_signals,
    neutralize_special_markers, validate_request_budget, AuthConfig, RequestLimits, RouterOptions,
};

const ADMIN_TOKEN: &str = "test-admin-token";

fn engine() -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42)
}

/// Qwen3's `<|im_start|>` id: the tokenizer-less routers of this suite serve
/// a `Qwen3Config::tiny_test()` engine (the Qwen3 vocabulary size) and run a
/// text prompt as this single token.
const QWEN3_IM_START: u32 = 151_644;

/// A tokenizer-less router over `engine`. Without a tokenizer a server needs
/// a configured prompt start token to accept a text prompt at all (it
/// answers `400 tokenizer_required` otherwise), and the answer's text is
/// empty — nothing to render it with — while `usage` still counts every
/// generated token.
fn tokenizerless_router(engine: InferenceEngine<'static>) -> axum::Router {
    oxibonsai_runtime::server::create_router_full(
        oxibonsai_runtime::engine_pool::EnginePool::new(vec![engine]),
        None,
        std::sync::Arc::new(oxibonsai_runtime::metrics::InferenceMetrics::new()),
        oxibonsai_runtime::server::RouterOptions::default().with_prompt_start_token(QWEN3_IM_START),
    )
}

fn router() -> axum::Router {
    tokenizerless_router(engine())
}

fn router_with_limits(limits: RequestLimits) -> axum::Router {
    router_over(engine(), limits)
}

/// [`router_with_limits`] over a caller-built single-replica `engine`.
fn router_over(engine: InferenceEngine<'static>, limits: RequestLimits) -> axum::Router {
    create_router_full(
        EnginePool::new(vec![engine]),
        None,
        Arc::new(InferenceMetrics::new()),
        RouterOptions::default()
            .with_limits(limits)
            .with_auth(AuthConfig::with_admin_token(ADMIN_TOKEN))
            .with_prompt_start_token(QWEN3_IM_START),
    )
}

fn chat_request(body: serde_json::Value) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("request")
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    serde_json::from_slice(&bytes).expect("response body must be JSON")
}

async fn body_text(resp: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    String::from_utf8_lossy(&bytes).into_owned()
}

// ── sec-01: the prompt sanitizer must be linear ───────────────────────────────

/// The P0: `neutralize_special_markers` was quadratic, so a 400 KB prompt
/// burned 149 s of a worker thread (measured, release) before generation
/// started — an unauthenticated remote CPU DoS.
///
/// The thresholds were originally ~3 orders of magnitude above the measured
/// linear cost, which is loose enough that a 100x performance regression back
/// toward quadratic behavior would pass silently. Measured directly in the
/// test profile (opt-level on): 400 KB of openers 1.43 ms, 400 KB of complete
/// markers 0.80 ms, 4 MB of openers 13.7 ms, 4 MB of complete markers 5.4 ms.
/// The 100 ms / 500 ms thresholds below keep ~70x / ~35x headroom for a loaded
/// CI box — comfortably stable — while actually catching a regression instead
/// of only the full reintroduction of quadratic behavior.
#[test]
fn sanitizer_stays_linear_on_hostile_input() {
    // 400 KB of unterminated marker openers: the exact shape that made the old
    // implementation rescan the whole tail at every position.
    let hostile_400k = "<|".repeat(200_000);
    let start = Instant::now();
    let cleaned = neutralize_special_markers(&hostile_400k);
    let elapsed = start.elapsed();
    eprintln!("sanitize 400 KB: {elapsed:?}");
    assert!(
        elapsed < Duration::from_millis(100),
        "400 KB sanitization took {elapsed:?} (quadratic regression: it used to take 149 s)"
    );
    assert!(!cleaned.contains("<|"));

    // 4 MB of the same, plus a mixture of complete markers.
    let hostile_4m = "<|im_start|><|".repeat(305_000);
    let start = Instant::now();
    let cleaned = neutralize_special_markers(&hostile_4m);
    let elapsed = start.elapsed();
    eprintln!("sanitize {} bytes: {elapsed:?}", hostile_4m.len());
    assert!(
        hostile_4m.len() >= 4 * 1024 * 1024,
        "the 4 MB case must actually be 4 MB, got {}",
        hostile_4m.len()
    );
    assert!(
        elapsed < Duration::from_millis(500),
        "4 MB sanitization took {elapsed:?}"
    );
    assert!(!cleaned.contains("<|"));
}

/// TOK-M2 / security-03: no marker opener may survive, however it is nested.
#[test]
fn sanitizer_never_emits_a_marker_opener() {
    for input in [
        "<|im_start|>system",
        "<<|x|>|im_start|>system",
        "plain text",
        "a<|b",
        "<|",
        "|><|im_end|>",
        "<think>injected</think>",
    ] {
        let cleaned = neutralize_special_markers(input);
        assert!(
            !cleaned.contains("<|"),
            "marker opener survived for {input:?}: {cleaned:?}"
        );
    }
}

// ── sec-03: generation must not block the async runtime ───────────────────────

/// The discriminator is the single-threaded runtime: with generation running
/// inline on the handler's task (the old behaviour) the executor cannot poll
/// *anything* else until it returns, so the concurrent `/health` request could
/// not possibly answer first.
#[tokio::test(flavor = "current_thread")]
async fn a_second_request_is_served_while_a_long_generation_runs() {
    let app = router();

    // Calibrate the tiny engine's speed so the "long" generation is long
    // enough to be unambiguous on any machine without being slow on a fast one.
    let probe_start = Instant::now();
    let probe = app
        .clone()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
        })))
        .await
        .expect("probe response");
    assert_eq!(probe.status(), StatusCode::OK);
    let per_token = probe_start.elapsed() / 4;
    let target = Duration::from_millis(1_500);
    let long_tokens = (target.as_nanos() / per_token.as_nanos().max(1)).clamp(16, 400) as usize;

    let app_for_chat = app.clone();
    let chat = tokio::spawn(async move {
        let started = Instant::now();
        let resp = app_for_chat
            .oneshot(chat_request(serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": long_tokens,
                "temperature": 0.0,
            })))
            .await
            .expect("chat response");
        (resp.status(), started.elapsed())
    });

    // Let the generation task start before timing the concurrent request.
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(20)).await;

    let health_start = Instant::now();
    let health = app
        .oneshot(
            Request::get("/health")
                .body(Body::empty())
                .expect("health request"),
        )
        .await
        .expect("health response");
    let health_elapsed = health_start.elapsed();
    assert_eq!(health.status(), StatusCode::OK);

    let (chat_status, chat_elapsed) = chat.await.expect("generation task");
    assert_eq!(chat_status, StatusCode::OK);
    assert!(
        health_elapsed < chat_elapsed,
        "the concurrent request ({health_elapsed:?}) must finish before the long \
         generation ({chat_elapsed:?}); generation is blocking the runtime"
    );
}

// ── sec-05 / RT-13 / SV-16: prompt and context budget ─────────────────────────

#[tokio::test]
async fn over_context_request_is_rejected_with_400_and_the_real_numbers() {
    // tiny_test has a 512-token context; a 512-token completion cannot fit
    // alongside the prompt. This used to reach the engine and come back as an
    // opaque 500.
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let json = body_json(resp).await;
    assert_eq!(json["error"]["code"], "context_length_exceeded");
    assert_eq!(json["error"]["context_length"], 512);
    assert_eq!(json["error"]["max_tokens"], 512);
    assert!(json["error"]["n_prompt_tokens"].is_number());
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn max_input_tokens_is_enforced_per_request() {
    // The shared validator is what `oxibonsai-serve` will call for its
    // `limits.max_input_tokens` field (SV-16), so assert it directly too.
    let err = validate_request_budget(4096, 16, 262_144, Some(2048)).expect_err("must reject");
    assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    assert_eq!(err.to_json()["error"]["code"], "max_input_tokens_exceeded");
    assert!(validate_request_budget(1024, 16, 262_144, Some(2048)).is_ok());
}

#[tokio::test]
async fn oversized_prompt_bytes_are_rejected_before_tokenization() {
    let app = router_with_limits(RequestLimits::default().with_max_prompt_bytes(1024));
    let resp = app
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "x".repeat(4096)}],
            "max_tokens": 4,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let json = body_json(resp).await;
    assert_eq!(json["error"]["code"], "prompt_too_large");
    assert_eq!(json["error"]["max_prompt_bytes"], 1024);
    assert_eq!(json["error"]["prompt_bytes"], 4096);
}

#[tokio::test]
async fn a_request_inside_the_budget_still_succeeds() {
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp).await;
    assert_eq!(json["object"], "chat.completion");
    assert!(json["usage"]["completion_tokens"].is_number());
}

// ── sec-08: SSE keep-alive, usage chunk, terminal [DONE] ──────────────────────

#[tokio::test]
async fn streaming_emits_a_usage_chunk_before_done_when_requested() {
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "stream": true,
            "stream_options": {"include_usage": true},
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);

    let body = body_text(resp).await;
    assert!(body.contains("chat.completion.chunk"), "body: {body}");
    assert!(
        body.trim_end().ends_with("data: [DONE]"),
        "an SSE stream must end with [DONE]: {body}"
    );

    let chunks: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .filter_map(|payload| serde_json::from_str::<serde_json::Value>(payload).ok())
        .collect();

    // OpenAI's contract in `include_usage` mode: every chunk but the last
    // carries an explicit `"usage": null`, starting with the role chunk.
    let role_chunk = chunks.first().expect("a role chunk must open the stream");
    assert_eq!(role_chunk["choices"][0]["delta"]["role"], "assistant");
    assert!(
        role_chunk.get("usage").is_some() && role_chunk["usage"].is_null(),
        "non-final chunks must carry an explicit null usage: {role_chunk}"
    );

    let usage_line = chunks
        .iter()
        .find(|chunk| chunk["usage"].is_object())
        .expect("a usage-carrying chunk must precede [DONE]");
    assert!(usage_line["usage"]["prompt_tokens"].is_number());
    assert!(usage_line["usage"]["completion_tokens"].is_number());
    assert_eq!(
        usage_line["choices"].as_array().map(Vec::len),
        Some(0),
        "the usage chunk carries no choices"
    );
}

#[tokio::test]
async fn streaming_without_stream_options_carries_no_usage() {
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "stream": true,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_text(resp).await;
    assert!(!body.contains("\"usage\""), "body: {body}");
    assert!(body.trim_end().ends_with("data: [DONE]"), "body: {body}");
}

// ── sec-08 (end-to-end): the router-configured deadline actually fires ───────
//
// `server/sse.rs` already unit-tests that `sse_response` itself honors a
// deadline it is handed directly; these two prove the deadline configured on
// the *router* (`RequestLimits::per_request_timeout`) really reaches it, for
// both response shapes, when driven through the real HTTP surface.
//
// Neither test estimates how long a generation takes. Whether a timed
// generation outlasts a deadline depends on the host's speed *at that moment*,
// and that is not stable inside one test binary: the same request takes
// several times longer while the other tests of the binary (or an instrumented
// `cargo llvm-cov` build) compete for the CPU than after they finish, so any
// deadline derived from a timing probe can come out longer than the whole
// generation. Instead the engine is *held*: an [`EngineGate`] parks the engine
// thread inside the per-request reset, so no token can be produced until the
// test lets go, however fast or slow the host is. The deadline fires because
// the test holds the engine, never because a timing guess came out short.

/// The whole-request deadline of the two tests below, in milliseconds.
///
/// Determinism comes from the gate, not from this number: it only has to
/// outlast the request's own setup (rendering, validating, taking the
/// replica — milliseconds), so that the stage the timeout reports is the
/// engine's (`prefill`) rather than an earlier one, and the streamed head is
/// out before the handler-level deadline.
const HELD_ENGINE_DEADLINE_MS: u64 = 1_000;

/// How long a parked engine thread waits for the gate to open before it gives
/// up on its own. A test that fails before it opens the gate releases it when
/// its guard drops; this bound is the second line of defence, because dropping
/// a tokio runtime waits for every `spawn_blocking` task, so a thread parked
/// forever would turn a failed assertion into a hung test binary.
const GATE_PARK_LIMIT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct GateState {
    /// The next per-request reset parks (set once the warm-up is done).
    armed: bool,
    /// An engine thread is parked in the gate.
    parked: bool,
    /// Released: the parked thread continues, and no later reset parks.
    open: bool,
}

/// Holds the engine's next step until the test releases it.
///
/// Both generation paths — `run_blocking_generation` for a plain request and
/// the blocking task of the streaming handler — call `InferenceEngine::reset`
/// as the first thing they do on the blocking thread, after the request has
/// taken its replica (stage `prefill`) and before any token exists. `reset`
/// clears an attached [`RecurrentState`], which is a public extension point, so
/// a test can park exactly there without any hook in the product code.
#[derive(Default)]
struct EngineGate {
    state: Mutex<GateState>,
    changed: Condvar,
}

impl EngineGate {
    fn lock(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// From now on the next per-request reset parks until [`Self::open`].
    fn arm(&self) {
        self.lock().armed = true;
    }

    /// Release the held engine thread; also what dropping an [`OpenOnDrop`]
    /// does. Idempotent.
    fn open(&self) {
        self.lock().open = true;
        self.changed.notify_all();
    }

    /// Wait (without blocking the async runtime) until an engine thread is
    /// parked in the gate; `false` if none arrived within `limit`.
    async fn wait_until_parked(&self, limit: Duration) -> bool {
        let deadline = Instant::now() + limit;
        while Instant::now() < deadline {
            if self.lock().parked {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        self.lock().parked
    }
}

/// The attached [`RecurrentState`]: parks the engine thread in its reset.
struct ParkInReset(Arc<EngineGate>);

impl RecurrentState for ParkInReset {
    fn reset_recurrent(&mut self) {
        let gate = &self.0;
        let mut state = gate.lock();
        if !state.armed || state.open {
            return;
        }
        state.parked = true;
        gate.changed.notify_all();
        // Parked until the test opens the gate (bounded, see GATE_PARK_LIMIT).
        let _released = gate
            .changed
            .wait_timeout_while(state, GATE_PARK_LIMIT, |held| !held.open)
            .unwrap_or_else(PoisonError::into_inner);
    }
}

/// Opens its gate when dropped, so a panicking test cannot leave the engine
/// thread parked.
struct OpenOnDrop(Arc<EngineGate>);

impl Drop for OpenOnDrop {
    fn drop(&mut self) {
        self.0.open();
    }
}

/// A single-replica router whose engine is held by the returned gate once
/// that is armed, with the request deadline set to [`HELD_ENGINE_DEADLINE_MS`].
fn held_engine_router() -> (axum::Router, Arc<EngineGate>, OpenOnDrop) {
    let gate = Arc::new(EngineGate::default());
    let mut held = engine();
    held.set_recurrent_state(Box::new(ParkInReset(Arc::clone(&gate))));
    let app = router_over(
        held,
        RequestLimits::default().with_timeout_ms(HELD_ENGINE_DEADLINE_MS),
    );
    let release = OpenOnDrop(Arc::clone(&gate));
    (app, gate, release)
}

/// Resolve the served model id once, with the gate still unarmed (the first
/// resolution takes a replica of its own, which is no part of what these tests
/// hold), then arm the gate.
async fn warm_up_then_arm(app: &axum::Router, gate: &EngineGate) {
    let resp = app
        .clone()
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("warm-up response");
    assert_eq!(resp.status(), StatusCode::OK);
    gate.arm();
}

/// The JSON payload of the `event: error` frame of an SSE `body`, if it has one.
fn sse_error_event(body: &str) -> Option<serde_json::Value> {
    body.split("\n\n").find_map(|frame| {
        let is_error = frame.lines().any(|line| line == "event: error");
        let data = frame.lines().find_map(|line| line.strip_prefix("data: "))?;
        if is_error {
            serde_json::from_str(data).ok()
        } else {
            None
        }
    })
}

#[tokio::test]
async fn non_streaming_request_times_out_with_504() {
    let (app, gate, _release) = held_engine_router();
    warm_up_then_arm(&app, &gate).await;

    let resp = app
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT);
    let json = body_json(resp).await;
    assert_eq!(json["error"]["code"], "request_timeout");
    assert_eq!(json["error"]["type"], "server_error");
    // The deadline names the stage it caught the request in: it holds a
    // replica and has produced no token, because its engine thread is parked.
    assert_eq!(json["error"]["phase"], "prefill", "{json}");
    let message = json["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains(&format!("{HELD_ENGINE_DEADLINE_MS} ms")),
        "the message names the configured limit: {message}"
    );
    assert!(
        gate.wait_until_parked(GATE_PARK_LIMIT).await,
        "the generation never reached the held engine step"
    );
    gate.open();
}

#[tokio::test]
async fn streaming_request_times_out_with_an_sse_error_event() {
    let (app, gate, _release) = held_engine_router();
    warm_up_then_arm(&app, &gate).await;

    let resp = app
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": true,
        })))
        .await
        .expect("response");
    // The deadline fires INSIDE the SSE body, not on the response head: the
    // head (and the body's role chunk) is sent as soon as the stream is set
    // up — the engine thread is parked by then, so no token can follow — and
    // the body's own deadline, which starts once the head is out, is what
    // ends the stream.
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_text(resp).await;
    assert!(
        body.contains("\"role\":\"assistant\""),
        "the role chunk precedes the timeout: {body}"
    );
    let error = sse_error_event(&body)
        .unwrap_or_else(|| panic!("the stream must carry an `event: error` frame: {body}"));
    assert_eq!(error["error"]["code"], "request_timeout", "{body}");
    assert_eq!(error["error"]["type"], "server_error", "{body}");
    assert_eq!(error["error"]["phase"], "prefill", "{body}");
    assert!(
        body.trim_end().ends_with("data: [DONE]"),
        "an SSE stream must still end with [DONE] after a timeout: {body}"
    );
    assert!(
        gate.wait_until_parked(GATE_PARK_LIMIT).await,
        "the generation never reached the held engine step"
    );
    gate.open();
}

// ── sec-15 / SV-07: the admin surface is authenticated ────────────────────────

#[tokio::test]
async fn admin_is_refused_when_no_token_is_configured() {
    // Built with an explicit `AuthConfig::locked()` rather than through the
    // environment-dependent `router()` helper: the convenience constructors
    // resolve the credential from `OXI_ADMIN_TOKEN`, so asserting this via
    // `router()` would silently flip 403 -> 401 on any machine that happens to
    // export that variable. `create_router_with_options`'s own
    // `AuthConfig::from_env()` convenience path is exercised separately in
    // `oxibonsai_runtime`'s own test suite; this assertion must hold
    // regardless of the ambient environment.
    let resp = create_router_with_auth(engine(), None, AuthConfig::locked())
        .oneshot(
            Request::get("/admin/config")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let json = body_json(resp).await;
    assert_eq!(json["error"]["code"], "admin_auth_not_configured");
    assert_eq!(json["error"]["type"], "authentication_error");
}

// ── sec-15 regression: the admin-auth layer must not swallow the app's 404 ───

/// The security regression found after the original `sec-15` fix
/// landed: `Router::layer` (unlike `Router::route_layer`) also wraps the
/// router's own fallback, and `Router::merge` then propagates that wrapped
/// fallback to the WHOLE merged app. That turned every unmatched path on the
/// entire server — not just under `/admin` — into a `403` carrying the
/// admin-auth configuration hints instead of an ordinary `404`: a broken 404
/// contract *and* an information-disclosure regression, introduced by the fix
/// meant to close `sec-15`. Assert the fallback stays a plain `404` with no
/// admin-auth leakage regardless of the admin-auth state (a locked-down admin
/// surface is the harness that reproduced the bug, since `403` is exactly
/// what `admin_auth_not_configured` returns; a configured token proves the
/// fallback isn't gated when a credential DOES exist either), and that a
/// route which genuinely exists — even requested with the "wrong" HTTP
/// method — is still gated rather than falling through as an unmatched path,
/// so a future `.layer(...)` regression here fails loudly on both ends.
#[tokio::test]
async fn unknown_path_is_404_not_403_regardless_of_admin_auth_state() {
    for (app, admin_configured) in [
        (
            create_router_with_auth(engine(), None, AuthConfig::locked()),
            false,
        ),
        (
            create_router_with_auth(engine(), None, AuthConfig::with_admin_token(ADMIN_TOKEN)),
            true,
        ),
    ] {
        for path in ["/no-such-path", "/v1/typo", "/admin/nope"] {
            let resp = app
                .clone()
                .oneshot(Request::get(path).body(Body::empty()).expect("request"))
                .await
                .expect("response");
            assert_eq!(
                resp.status(),
                StatusCode::NOT_FOUND,
                "{path} must be a plain 404 (admin_configured={admin_configured})"
            );
            let body = body_text(resp).await;
            assert!(
                !body.contains("admin_auth_not_configured") && !body.contains("OXI_ADMIN_TOKEN"),
                "a 404 on {path} must never leak admin-auth configuration hints: {body}"
            );
        }

        // A route that genuinely exists, requested with the "wrong" method,
        // must still be gated by the admin-auth middleware rather than
        // falling through as if it were an unmatched path.
        let resp = app
            .oneshot(
                Request::get("/admin/reset-metrics")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_ne!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "a registered admin route must never appear as 404, even under the wrong method \
             (admin_configured={admin_configured})"
        );
        let expected = if admin_configured {
            StatusCode::UNAUTHORIZED
        } else {
            StatusCode::FORBIDDEN
        };
        assert_eq!(resp.status(), expected);
    }
}

#[tokio::test]
async fn admin_requires_the_configured_credential() {
    let app = create_router_with_auth(engine(), None, AuthConfig::with_admin_token(ADMIN_TOKEN));

    let unauthenticated = app
        .clone()
        .oneshot(
            Request::get("/admin/config")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);
    assert!(
        unauthenticated.headers().get("www-authenticate").is_some(),
        "a 401 must advertise the scheme"
    );
    let json = body_json(unauthenticated).await;
    assert_eq!(json["error"]["code"], "missing_admin_token");

    let wrong = app
        .clone()
        .oneshot(
            Request::get("/admin/config")
                .header("authorization", "Bearer wrong")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);

    let authorized = app
        .oneshot(
            Request::get("/admin/config")
                .header("authorization", format!("Bearer {ADMIN_TOKEN}"))
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(authorized.status(), StatusCode::OK);
    let json = body_json(authorized).await;
    assert_eq!(json["model"]["id"], "Bonsai-Tiny-Test");
}

#[tokio::test]
async fn mutating_admin_route_is_gated_too() {
    let app = create_router_with_auth(engine(), None, AuthConfig::with_admin_token(ADMIN_TOKEN));
    let resp = app
        .clone()
        .oneshot(
            Request::post("/admin/reset-metrics")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let resp = app
        .oneshot(
            Request::post("/admin/reset-metrics")
                .header("x-admin-token", ADMIN_TOKEN)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn inference_routes_are_not_gated_by_admin_auth() {
    // The admin layer must not leak onto the OpenAI surface.
    let app = router();
    for path in ["/health", "/v1/models", "/metrics"] {
        let resp = app
            .clone()
            .oneshot(Request::get(path).body(Body::empty()).expect("request"))
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK, "{path} must stay public");
    }
}

// ── SV-04 / SV-05: one error envelope everywhere ──────────────────────────────

#[tokio::test]
async fn malformed_json_returns_the_error_envelope() {
    let resp = router()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{\"messages\": "))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let json = body_json(resp).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert!(json["error"]["message"].is_string());
}

#[tokio::test]
async fn wrong_field_type_is_400_with_an_envelope_not_422_plain_text() {
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": "not a number",
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let json = body_json(resp).await;
    assert_eq!(json["error"]["code"], "invalid_request_body");
}

#[tokio::test]
async fn validation_errors_carry_the_request_id_header() {
    let resp = router()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-request-id", "11111111-1111-1111-1111-111111111111")
                .body(Body::from(
                    serde_json::json!({
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 0,
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        resp.headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok()),
        Some("11111111-1111-1111-1111-111111111111"),
        "error responses must carry the correlation id too (SV-04)"
    );
    let json = body_json(resp).await;
    assert_eq!(json["error"]["param"], "max_tokens");
}

// ── SV-29 / deps-11: shutdown signal installation ─────────────────────────────

#[tokio::test]
async fn shutdown_signal_installation_is_reported_not_expected() {
    // The defect was `expect("failed to install SIGTERM handler")` in
    // production code; the seam must be a Result the caller can act on.
    assert!(install_shutdown_signals().is_ok());
}

// ── RT-03: no engine state crosses requests ───────────────────────────────────

#[tokio::test]
async fn identical_greedy_requests_produce_identical_output() {
    let app = router();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "deterministic"}],
        "max_tokens": 6,
        "temperature": 0.0,
    });

    let first = app
        .clone()
        .oneshot(chat_request(body.clone()))
        .await
        .expect("response");
    assert_eq!(first.status(), StatusCode::OK);
    let first_json = body_json(first).await;

    let second = app.oneshot(chat_request(body)).await.expect("response");
    assert_eq!(second.status(), StatusCode::OK);
    let second_json = body_json(second).await;

    assert_eq!(
        first_json["choices"][0]["message"]["content"],
        second_json["choices"][0]["message"]["content"],
        "greedy decoding must not depend on the previous request's engine state"
    );
    assert_eq!(
        first_json["usage"]["completion_tokens"],
        second_json["usage"]["completion_tokens"]
    );
}

// ── The deadline of every generation endpoint ─────────────────────────────────
//
// `/v1/chat/completions/extended` and `/v1/completions` carry the base
// endpoint's own deadline: the same `504 request_timeout` naming the stage in
// `error.phase` while no response has started, the same SSE `error` event
// then `[DONE]` once a stream is open, and the same cancellation. The engine
// is held exactly as above ([`EngineGate`]), or its only replica is taken by
// the test (the request queues: `waiting_for_engine`). Every case also checks
// that the generation was cancelled: once the engine is released the replica
// comes back without having produced a single token (the engine's own
// `tokens_generated_total` stays at zero), which an uncancelled four-token
// generation could not do.

/// The three generation routes.
const GENERATION_ROUTES: [&str; 3] = [
    "/v1/chat/completions",
    "/v1/chat/completions/extended",
    "/v1/completions",
];

/// A request every generation route accepts on a tokenizer-less router.
fn generation_body(path: &str, stream: bool) -> serde_json::Value {
    if path == "/v1/completions" {
        serde_json::json!({
            "prompt": "hi",
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": stream,
        })
    } else {
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": stream,
        })
    }
}

fn post_to(path: &str, body: &serde_json::Value) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(body).expect("serialize")))
        .expect("request")
}

/// A held-engine router whose pool and engine metrics the test also holds.
struct HeldServer {
    app: axum::Router,
    gate: Arc<EngineGate>,
    _release: OpenOnDrop,
    pool: Arc<EnginePool>,
    /// The engine's own counters: what it generated, whoever asked.
    engine_metrics: Arc<InferenceMetrics>,
}

impl HeldServer {
    /// A single-replica router with the deadline [`HELD_ENGINE_DEADLINE_MS`],
    /// its served model already resolved (the first resolution takes a
    /// replica of its own).
    async fn start() -> Self {
        let gate = Arc::new(EngineGate::default());
        let mut held = engine();
        held.set_recurrent_state(Box::new(ParkInReset(Arc::clone(&gate))));
        let engine_metrics = Arc::new(InferenceMetrics::new());
        held.set_metrics(Arc::clone(&engine_metrics));
        let pool = EnginePool::new(vec![held]);
        let app = create_router_full(
            Arc::clone(&pool),
            None,
            Arc::new(InferenceMetrics::new()),
            RouterOptions::default()
                .with_limits(RequestLimits::default().with_timeout_ms(HELD_ENGINE_DEADLINE_MS))
                .with_auth(AuthConfig::with_admin_token(ADMIN_TOKEN))
                .with_prompt_start_token(QWEN3_IM_START),
        );
        let server = Self {
            app,
            _release: OpenOnDrop(Arc::clone(&gate)),
            gate,
            pool,
            engine_metrics,
        };
        let resp = server
            .app
            .clone()
            .oneshot(
                Request::get("/v1/models")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("warm-up response");
        assert_eq!(resp.status(), StatusCode::OK);
        server
    }

    /// Release the held engine thread and wait (bounded) for the replica:
    /// the request's generation was cancelled before it produced a token.
    async fn release_and_expect_no_token(&self, what: &str) {
        assert!(
            self.gate.wait_until_parked(GATE_PARK_LIMIT).await,
            "{what}: the generation never reached the held engine step"
        );
        self.gate.open();
        let back = tokio::time::timeout(GATE_PARK_LIMIT, self.pool.acquire()).await;
        assert!(
            matches!(back, Ok(Ok(_))),
            "{what}: the replica never came back"
        );
        assert_eq!(
            self.engine_metrics.tokens_generated_total.get(),
            0,
            "{what}: the deadline must cancel the generation, which then produces nothing"
        );
    }
}

#[tokio::test]
async fn every_generation_route_times_out_with_a_504_naming_the_stage() {
    for path in GENERATION_ROUTES {
        let server = HeldServer::start().await;
        server.gate.arm();
        let resp = server
            .app
            .clone()
            .oneshot(post_to(path, &generation_body(path, false)))
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT, "{path}");
        let json = body_json(resp).await;
        assert_eq!(json["error"]["code"], "request_timeout", "{path}: {json}");
        assert_eq!(json["error"]["type"], "server_error", "{path}: {json}");
        assert_eq!(json["error"]["phase"], "prefill", "{path}: {json}");
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains(&format!("{HELD_ENGINE_DEADLINE_MS} ms")),
            "{path}: the message names the configured limit: {message}"
        );
        server.release_and_expect_no_token(path).await;
    }
}

#[tokio::test]
async fn every_generation_route_ends_an_open_stream_with_an_sse_error_event() {
    for path in GENERATION_ROUTES {
        let server = HeldServer::start().await;
        server.gate.arm();
        let resp = server
            .app
            .clone()
            .oneshot(post_to(path, &generation_body(path, true)))
            .await
            .expect("response");
        // The stream is open (the engine thread is parked by then, so no
        // token can follow); the body's own deadline ends it.
        assert_eq!(resp.status(), StatusCode::OK, "{path}");
        let body = body_text(resp).await;
        let error = sse_error_event(&body)
            .unwrap_or_else(|| panic!("{path}: no `event: error` frame: {body}"));
        assert_eq!(error["error"]["code"], "request_timeout", "{path}: {body}");
        assert_eq!(error["error"]["type"], "server_error", "{path}: {body}");
        assert_eq!(error["error"]["phase"], "prefill", "{path}: {body}");
        assert!(
            body.trim_end().ends_with("data: [DONE]"),
            "{path}: an SSE stream must still end with [DONE] after a timeout: {body}"
        );
        server.release_and_expect_no_token(path).await;
    }
}

#[tokio::test]
async fn a_stream_queued_for_a_replica_times_out_before_it_opens() {
    for path in GENERATION_ROUTES {
        let server = HeldServer::start().await;
        // The test takes the only replica: the request queues for it.
        let taken = server.pool.acquire().await.expect("take the replica");
        let resp = server
            .app
            .clone()
            .oneshot(post_to(path, &generation_body(path, true)))
            .await
            .expect("response");
        // No stream was opened: a plain JSON `504`, not an SSE body.
        assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT, "{path}");
        let json = body_json(resp).await;
        assert_eq!(json["error"]["code"], "request_timeout", "{path}: {json}");
        assert_eq!(
            json["error"]["phase"], "waiting_for_engine",
            "{path}: {json}"
        );
        drop(taken);
        let back = tokio::time::timeout(GATE_PARK_LIMIT, server.pool.acquire()).await;
        assert!(matches!(back, Ok(Ok(_))), "{path}: the replica is free");
        assert_eq!(server.engine_metrics.tokens_generated_total.get(), 0);
    }
}

/// `oxibonsai serve` and `oxibonsai-serve` put an admission timeout of the
/// same `--request-timeout-ms` plus a two-second grace in front of this
/// router; it answers a bare `408` when the router has not produced a
/// response head by then. On every generation route the handler's own
/// deadline answers first — streamed or not, caught in prefill or queued for
/// a replica — so that backstop is never what a client of these routes gets:
/// the response head always arrives well inside the grace, typed.
#[tokio::test]
async fn the_admission_backstop_is_unreachable_on_every_generation_route() {
    const ADMISSION_TIMEOUT_GRACE_MS: u64 = 2_000;
    let backstop = Duration::from_millis(HELD_ENGINE_DEADLINE_MS + ADMISSION_TIMEOUT_GRACE_MS);
    for path in GENERATION_ROUTES {
        for stream in [false, true] {
            for queued in [false, true] {
                let server = HeldServer::start().await;
                let taken = if queued {
                    Some(server.pool.acquire().await.expect("take the replica"))
                } else {
                    server.gate.arm();
                    None
                };
                let started = Instant::now();
                let head = tokio::time::timeout(
                    backstop,
                    server
                        .app
                        .clone()
                        .oneshot(post_to(path, &generation_body(path, stream))),
                )
                .await;
                let what = format!("{path}, stream {stream}, queued {queued}");
                let resp = match head {
                    Ok(Ok(resp)) => resp,
                    Ok(Err(e)) => panic!("{what}: {e}"),
                    Err(_) => panic!("{what}: no response head within the admission backstop"),
                };
                assert!(started.elapsed() < backstop, "{what}");
                let open_stream = stream && !queued;
                if open_stream {
                    assert_eq!(resp.status(), StatusCode::OK, "{what}");
                    let body = body_text(resp).await;
                    let error = sse_error_event(&body)
                        .unwrap_or_else(|| panic!("{what}: no `event: error` frame: {body}"));
                    assert_eq!(error["error"]["code"], "request_timeout", "{what}");
                } else {
                    assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT, "{what}");
                    let json = body_json(resp).await;
                    assert_eq!(json["error"]["code"], "request_timeout", "{what}: {json}");
                    assert!(json["error"]["phase"].is_string(), "{what}: {json}");
                }
                drop(taken);
                if !queued {
                    server.gate.open();
                }
            }
        }
    }
}
