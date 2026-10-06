//! Abandoned and late requests on the three generation endpoints, through
//! the real router, with the engine parked by the test.
//!
//! The engine is weightless and scripted (`InferenceEngine::script_generation_held`):
//! its first generation draws one token and then waits — up to the script's
//! own hold limit, five seconds — until its cancellation token is cancelled.
//! A test therefore knows exactly where the generation is (decoding, one
//! token in) when it abandons the request or lets the deadline fire, and a
//! replica that comes back well inside that limit can only have come back
//! because something cancelled the generation; without the cancellation
//! the replica stays busy for the whole hold and then finishes the script.
//!
//! * **Abandonment** (a non-streamed request whose handler future is
//!   dropped mid-generation): the replica is handed out again within
//!   [`REPLICA_BACK_WITHIN`] and the next request on it answers normally;
//!   for a multi-generation request (`/extended` with `n = 3`, a
//!   `/v1/completions` batch of three prompts) the remaining choices never
//!   run. Removing the guard (`CancelOnAbandon`) leaves the replica held
//!   for the whole hold; removing the loop's cancellation check runs the
//!   remaining choices (each one's per-run reset is counted).
//! * **Not truncated**: a request that completes normally answers the whole
//!   script, so the guard is disarmed only once the answer is in hand.
//! * **Deadline in decode**: a non-streamed request answers `504
//!   request_timeout` with `error.phase = decode` and `generated_tokens`; an
//!   open stream ends with the same error in an SSE `error` event, then
//!   `[DONE]`; either way the replica comes back promptly (the deadline
//!   cancelled the generation).
//! * **A generous deadline changes no answer**, streamed or not.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use serde_json::{json, Value};

use crate::engine_control::RecurrentState;
use crate::engine_pool::EnginePool;
use crate::metrics::InferenceMetrics;
use crate::sampling::SamplingParams;
use crate::server::{create_router_full, RequestLimits, RouterOptions};
use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

/// Every generation's script: 26 visible bytes, then the engine's EOS.
const SCRIPT: &str = "abcdefghijklmnopqrstuvwxyz";

/// How long a replica may take to come back once its request was abandoned
/// or timed out: far below the script's five-second hold, so only a
/// cancelled generation can make it.
const REPLICA_BACK_WITHIN: Duration = Duration::from_millis(2_000);

/// The deadline of the decode-phase tests. Determinism comes from the
/// parked engine, not from this number: it only has to outlast the
/// request's setup and its first token (milliseconds), and stay below the
/// script's hold.
const DEADLINE_MS: u64 = 1_000;

/// Upper bound on waiting for a generation to reach the engine.
const START_LIMIT: Duration = Duration::from_secs(10);

/// The three generation endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endpoint {
    Chat,
    Extended,
    Completions,
}

impl Endpoint {
    const ALL: [Self; 3] = [Self::Chat, Self::Extended, Self::Completions];

    fn path(self) -> &'static str {
        match self {
            Self::Chat => "/v1/chat/completions",
            Self::Extended => "/v1/chat/completions/extended",
            Self::Completions => "/v1/completions",
        }
    }

    /// A request for `max_tokens` tokens; `choices` is `n` on `/extended`
    /// and the batch size on `/v1/completions` (the base endpoint serves one).
    fn body(self, choices: usize, stream: bool) -> Value {
        match self {
            Self::Chat => json!({
                "messages": [{ "role": "user", "content": "hi" }],
                "max_tokens": 64,
                "stream": stream,
            }),
            Self::Extended => json!({
                "messages": [{ "role": "user", "content": "hi" }],
                "max_tokens": 64,
                "n": choices,
                "stream": stream,
            }),
            Self::Completions => {
                let prompts: Vec<String> = (0..choices).map(|i| format!("hi {i}")).collect();
                let prompt = if choices == 1 {
                    json!("hi")
                } else {
                    json!(prompts)
                };
                json!({ "prompt": prompt, "max_tokens": 64, "stream": stream })
            }
        }
    }

    /// The text of every choice of a non-streamed answer.
    fn texts(self, answer: &Value) -> Vec<String> {
        answer["choices"]
            .as_array()
            .map(|choices| {
                choices
                    .iter()
                    .map(|choice| match self {
                        Self::Chat | Self::Extended => choice["message"]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        Self::Completions => {
                            choice["text"].as_str().unwrap_or_default().to_string()
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The streamed text of an SSE body (every chunk's delta, in order).
    fn streamed_text(self, body: &str) -> String {
        fx::sse_payloads(body)
            .iter()
            .filter_map(|chunk| match self {
                Self::Chat | Self::Extended => chunk["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_string),
                Self::Completions => chunk["choices"][0]["text"].as_str().map(str::to_string),
            })
            .collect()
    }
}

/// Counts the engine's per-request resets: every generation of a request
/// starts with one, so the count says how many generations ran.
struct CountResets(Arc<AtomicUsize>);

impl RecurrentState for CountResets {
    fn reset_recurrent(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// A single-replica router over a scripted engine, and what the test reads.
struct Fixture {
    app: axum::Router,
    pool: Arc<EnginePool>,
    /// The engine's own metrics: a finished prefill is the sign that the
    /// generation reached the engine.
    engine_metrics: Arc<InferenceMetrics>,
    resets: Arc<AtomicUsize>,
}

impl Fixture {
    /// `held`: the first generation waits after its first token until it is
    /// cancelled (see the module docs).
    fn new(held: bool, limits: RequestLimits) -> Self {
        let mut engine = fx::weightless_engine(fx::BYTE_VOCAB, SamplingParams::default(), 42);
        if held {
            engine.script_generation_held(fx::byte_ids(SCRIPT), 1);
        } else {
            engine.script_generation(fx::byte_ids(SCRIPT));
        }
        let engine_metrics = Arc::new(InferenceMetrics::new());
        engine.set_metrics(Arc::clone(&engine_metrics));
        let resets = Arc::new(AtomicUsize::new(0));
        engine.set_recurrent_state(Box::new(CountResets(Arc::clone(&resets))));
        let pool = EnginePool::new(vec![engine]);
        let app = create_router_full(
            Arc::clone(&pool),
            Some(fx::byte_tokenizer()),
            Arc::new(InferenceMetrics::new()),
            RouterOptions::default().with_limits(limits),
        );
        Self {
            app,
            pool,
            engine_metrics,
            resets,
        }
    }

    /// Resolve the served model once (the first resolution takes a replica
    /// of its own, which is no part of what a test holds).
    async fn warm_up(&self) {
        let response = tower::ServiceExt::oneshot(
            self.app.clone(),
            axum::http::Request::get("/v1/models")
                .body(axum::body::Body::empty())
                .expect("build the request"),
        )
        .await
        .expect("the router answers");
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// Whether the pool hands its replica out again within
    /// [`REPLICA_BACK_WITHIN`].
    async fn replica_back(&self) -> bool {
        matches!(
            tokio::time::timeout(REPLICA_BACK_WITHIN, self.pool.acquire()).await,
            Ok(Ok(_))
        )
    }

    /// Start `body` on `endpoint`, wait until its generation has finished a
    /// prefill on the engine (the script then parks it after one token), and
    /// drop the handler future — a client that went away.
    async fn abandon_mid_generation(&self, endpoint: Endpoint, body: &Value) {
        let prefills = self.engine_metrics.prefill_duration_seconds.count();
        let request = tokio::spawn(fx::post(self.app.clone(), endpoint.path(), body.clone()));
        let started = wait_until(START_LIMIT, || {
            self.engine_metrics.prefill_duration_seconds.count() > prefills
        })
        .await;
        assert!(
            started,
            "{endpoint:?}: the generation never reached the engine"
        );
        request.abort();
        assert!(
            request.await.is_err(),
            "{endpoint:?}: the request finished on its own; it was meant to be abandoned"
        );
    }
}

/// Poll `ready` (every few milliseconds, at most `limit`); whether it held.
async fn wait_until(limit: Duration, ready: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < limit {
        if ready() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    ready()
}

/// The JSON payload of the `event: error` frame of an SSE `body`, if any.
fn sse_error_event(body: &str) -> Option<Value> {
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

/// The answer with its per-response identity (`id`, `created`) removed.
fn without_identity(mut answer: Value) -> Value {
    if let Some(object) = answer.as_object_mut() {
        object.remove("id");
        object.remove("created");
        object.remove("system_fingerprint");
    }
    answer
}

// ── Abandonment ──────────────────────────────────────────────────────────────

/// One endpoint's abandoned non-streamed request (see the module docs).
async fn an_abandoned_request_frees_its_replica(endpoint: Endpoint) {
    let fixture = Fixture::new(true, RequestLimits::default());
    fixture.warm_up().await;
    fixture
        .abandon_mid_generation(endpoint, &endpoint.body(1, false))
        .await;
    assert!(
        fixture.replica_back().await,
        "{endpoint:?}: the replica of an abandoned request must come back promptly, not after \
         the generation ran on for nobody"
    );

    // The next request on the same (only) replica answers in full.
    let (status, _, text) = fx::post(
        fixture.app.clone(),
        endpoint.path(),
        endpoint.body(1, false),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{endpoint:?}: {text}");
    let answer: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(
        endpoint.texts(&answer),
        vec![SCRIPT.to_string()],
        "{answer}"
    );
}

#[tokio::test]
async fn an_abandoned_chat_request_frees_its_replica() {
    an_abandoned_request_frees_its_replica(Endpoint::Chat).await;
}

#[tokio::test]
async fn an_abandoned_extended_request_frees_its_replica() {
    an_abandoned_request_frees_its_replica(Endpoint::Extended).await;
}

#[tokio::test]
async fn an_abandoned_completion_frees_its_replica() {
    an_abandoned_request_frees_its_replica(Endpoint::Completions).await;
}

/// A client that closes its TCP connection while its non-streamed request
/// is generating, over a real socket: the HTTP server drops the handler, the
/// guard cancels the generation, and the replica comes back promptly — on
/// every endpoint. (Without the guard the replica stays held for the
/// script's whole hold.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_that_closes_its_connection_frees_the_replica() {
    use tokio::io::AsyncWriteExt;

    for endpoint in Endpoint::ALL {
        let fixture = Fixture::new(true, RequestLimits::default());
        fixture.warm_up().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a loopback listener");
        let address = listener.local_addr().expect("the listener's address");
        let app = fixture.app.clone();
        let server = tokio::spawn(async move { axum::serve(listener, app).await });

        let body = endpoint.body(1, false).to_string();
        let request = format!(
            "POST {} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{body}",
            endpoint.path(),
            body.len()
        );
        let prefills = fixture.engine_metrics.prefill_duration_seconds.count();
        let mut client = tokio::net::TcpStream::connect(address)
            .await
            .expect("connect to the server");
        client
            .write_all(request.as_bytes())
            .await
            .expect("send the request");
        let started = wait_until(START_LIMIT, || {
            fixture.engine_metrics.prefill_duration_seconds.count() > prefills
        })
        .await;
        assert!(
            started,
            "{endpoint:?}: the generation never reached the engine"
        );

        // The client goes away: the connection is closed, not half-closed.
        drop(client);
        assert!(
            fixture.replica_back().await,
            "{endpoint:?}: a client that closed its connection must free the replica at once"
        );
        server.abort();
    }
}

/// A multi-generation request abandoned during its first choice: the
/// replica comes back promptly and the remaining choices never run — the
/// request's resets are the blocking task's own and the first choice's.
async fn an_abandoned_multi_generation_request_runs_no_further_choice(endpoint: Endpoint) {
    let fixture = Fixture::new(true, RequestLimits::default());
    fixture.warm_up().await;
    let resets_before = fixture.resets.load(Ordering::SeqCst);
    fixture
        .abandon_mid_generation(endpoint, &endpoint.body(3, false))
        .await;
    assert!(
        fixture.replica_back().await,
        "{endpoint:?}: the replica must come back promptly"
    );
    assert_eq!(
        fixture.resets.load(Ordering::SeqCst) - resets_before,
        2,
        "{endpoint:?}: the blocking task's reset and the first choice's, and no other choice"
    );
    assert_eq!(
        fixture.engine_metrics.prefill_duration_seconds.count(),
        1,
        "{endpoint:?}: only the first choice was prefilled"
    );

    // The replica answers a full multi-choice request afterwards.
    let (status, _, text) = fx::post(
        fixture.app.clone(),
        endpoint.path(),
        endpoint.body(3, false),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{endpoint:?}: {text}");
    let answer: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(
        endpoint.texts(&answer),
        vec![SCRIPT.to_string(); 3],
        "{answer}"
    );
}

#[tokio::test]
async fn an_abandoned_n_choice_extended_request_runs_no_further_choice() {
    an_abandoned_multi_generation_request_runs_no_further_choice(Endpoint::Extended).await;
}

#[tokio::test]
async fn an_abandoned_completion_batch_runs_no_further_prompt() {
    an_abandoned_multi_generation_request_runs_no_further_choice(Endpoint::Completions).await;
}

// ── A normally completed request is whole ────────────────────────────────────

async fn a_completed_request_is_not_truncated(endpoint: Endpoint) {
    let fixture = Fixture::new(false, RequestLimits::default());
    let choices = if endpoint == Endpoint::Chat { 1 } else { 2 };
    let (status, _, text) = fx::post(
        fixture.app.clone(),
        endpoint.path(),
        endpoint.body(choices, false),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{endpoint:?}: {text}");
    let answer: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(
        endpoint.texts(&answer),
        vec![SCRIPT.to_string(); choices],
        "{endpoint:?}: every choice carries the whole script: {answer}"
    );
    assert_eq!(
        answer["usage"]["completion_tokens"].as_u64(),
        Some((SCRIPT.len() * choices) as u64),
        "{answer}"
    );
    for choice in answer["choices"].as_array().into_iter().flatten() {
        assert_eq!(choice["finish_reason"], "stop", "{answer}");
    }
    assert!(fixture.replica_back().await);
}

#[tokio::test]
async fn a_completed_chat_request_is_not_truncated() {
    a_completed_request_is_not_truncated(Endpoint::Chat).await;
}

#[tokio::test]
async fn a_completed_extended_request_is_not_truncated() {
    a_completed_request_is_not_truncated(Endpoint::Extended).await;
}

#[tokio::test]
async fn a_completed_completion_is_not_truncated() {
    a_completed_request_is_not_truncated(Endpoint::Completions).await;
}

// ── The deadline catches a request in decode ─────────────────────────────────

async fn a_non_streamed_deadline_in_decode_cancels_and_names_the_stage(endpoint: Endpoint) {
    let fixture = Fixture::new(true, RequestLimits::default().with_timeout_ms(DEADLINE_MS));
    fixture.warm_up().await;
    let (status, _, text) = fx::post(
        fixture.app.clone(),
        endpoint.path(),
        endpoint.body(1, false),
    )
    .await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{endpoint:?}: {text}");
    let error: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(error["error"]["code"], "request_timeout", "{error}");
    assert_eq!(error["error"]["type"], "server_error", "{error}");
    assert_eq!(error["error"]["phase"], "decode", "{error}");
    assert_eq!(error["error"]["generated_tokens"], 1, "{error}");
    let message = error["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains(&format!("per-request timeout of {DEADLINE_MS} ms")),
        "{message}"
    );
    assert!(
        fixture.replica_back().await,
        "{endpoint:?}: the deadline must cancel the generation"
    );
}

#[tokio::test]
async fn a_chat_deadline_in_decode_cancels_and_names_the_stage() {
    a_non_streamed_deadline_in_decode_cancels_and_names_the_stage(Endpoint::Chat).await;
}

#[tokio::test]
async fn an_extended_deadline_in_decode_cancels_and_names_the_stage() {
    a_non_streamed_deadline_in_decode_cancels_and_names_the_stage(Endpoint::Extended).await;
}

#[tokio::test]
async fn a_completion_deadline_in_decode_cancels_and_names_the_stage() {
    a_non_streamed_deadline_in_decode_cancels_and_names_the_stage(Endpoint::Completions).await;
}

async fn an_open_stream_deadline_in_decode_ends_the_stream_and_cancels(endpoint: Endpoint) {
    let fixture = Fixture::new(true, RequestLimits::default().with_timeout_ms(DEADLINE_MS));
    fixture.warm_up().await;
    let (status, _, body) =
        fx::post(fixture.app.clone(), endpoint.path(), endpoint.body(1, true)).await;
    // The stream was open before the deadline expired: the error travels in
    // the body.
    assert_eq!(status, StatusCode::OK, "{endpoint:?}: {body}");
    assert_eq!(
        endpoint.streamed_text(&body),
        "a",
        "{endpoint:?}: the one token generated before the hold reached the client: {body}"
    );
    let error = sse_error_event(&body)
        .unwrap_or_else(|| panic!("{endpoint:?}: no `event: error` frame: {body}"));
    assert_eq!(error["error"]["code"], "request_timeout", "{body}");
    assert_eq!(error["error"]["phase"], "decode", "{body}");
    assert_eq!(error["error"]["generated_tokens"], 1, "{body}");
    assert!(
        body.trim_end().ends_with("data: [DONE]"),
        "{endpoint:?}: [DONE] follows the error: {body}"
    );
    assert!(
        fixture.replica_back().await,
        "{endpoint:?}: the stream's deadline must cancel the generation"
    );
}

#[tokio::test]
async fn an_open_chat_stream_deadline_in_decode_ends_the_stream_and_cancels() {
    an_open_stream_deadline_in_decode_ends_the_stream_and_cancels(Endpoint::Chat).await;
}

#[tokio::test]
async fn an_open_extended_stream_deadline_in_decode_ends_the_stream_and_cancels() {
    an_open_stream_deadline_in_decode_ends_the_stream_and_cancels(Endpoint::Extended).await;
}

#[tokio::test]
async fn an_open_completion_stream_deadline_in_decode_ends_the_stream_and_cancels() {
    an_open_stream_deadline_in_decode_ends_the_stream_and_cancels(Endpoint::Completions).await;
}

// ── A generous deadline changes nothing ──────────────────────────────────────

#[tokio::test]
async fn a_generous_deadline_changes_no_answer_on_any_endpoint() {
    for endpoint in Endpoint::ALL {
        for stream in [false, true] {
            let mut answers = Vec::new();
            for limits in [
                RequestLimits::default(),
                RequestLimits::default().with_timeout_ms(600_000),
            ] {
                let fixture = Fixture::new(false, limits);
                let (status, _, text) = fx::post(
                    fixture.app.clone(),
                    endpoint.path(),
                    endpoint.body(1, stream),
                )
                .await;
                assert_eq!(status, StatusCode::OK, "{endpoint:?} {stream}: {text}");
                answers.push(if stream {
                    Value::String(endpoint.streamed_text(&text))
                } else {
                    without_identity(serde_json::from_str(&text).unwrap_or(Value::Null))
                });
            }
            assert_eq!(answers[0], answers[1], "{endpoint:?}, stream {stream}");
            if stream {
                assert_eq!(answers[0], Value::String(SCRIPT.to_string()));
            }
        }
    }
}
