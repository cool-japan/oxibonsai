//! A non-streamed request whose client goes away, through the real router:
//! the generation it started is cancelled, and a request that runs several
//! generations (`n` choices on `/v1/chat/completions/extended`, a prompt
//! batch on `/v1/completions`) runs none after the one in flight.
//!
//! Deterministic, with no timing involved: the engine carries an attached
//! [`RecurrentState`] (a public extension point the engine resets at the
//! start of every generation) that counts every reset and parks the engine
//! thread in the request's first one until the test lets go. The test drops
//! the handler future while the engine is parked — a client that went away —
//! and only then releases it. From that point:
//!
//! * with the abandonment guard, the request's token is already cancelled,
//!   so no generation produces a token (the engine's own
//!   `tokens_generated_total` stays at zero) and no further choice starts
//!   (exactly one reset: the request's own, before its first choice);
//! * without the guard, every choice runs to its end and generates tokens;
//! * without the loop's cancellation check, every remaining choice still
//!   resets the engine (each one's reset is counted).
//!
//! The in-crate tests (`server::blocking::abandon_tests`) cover the same on
//! a generation abandoned mid-decode.

#![cfg(feature = "server")]

use std::sync::atomic::{AtomicUsize, Ordering};
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
use oxibonsai_runtime::server::{create_router_full, RouterOptions};

/// Qwen3's `<|im_start|>`: the tokenizer-less router runs a text prompt as
/// this single token.
const QWEN3_IM_START: u32 = 151_644;

/// Upper bound on any wait in these tests; a parked engine thread gives up
/// on its own after it too, so a failing test cannot hang the binary.
const LIMIT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct GateState {
    /// The next reset parks.
    armed: bool,
    /// An engine thread is parked.
    parked: bool,
    /// Released: no later reset parks.
    open: bool,
}

/// Parks the engine in the request's first reset and counts every reset.
#[derive(Default)]
struct Gate {
    state: Mutex<GateState>,
    changed: Condvar,
    resets: AtomicUsize,
}

impl Gate {
    fn lock(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn arm(&self) {
        self.resets.store(0, Ordering::SeqCst);
        self.lock().armed = true;
    }

    fn open(&self) {
        self.lock().open = true;
        self.changed.notify_all();
    }

    fn resets(&self) -> usize {
        self.resets.load(Ordering::SeqCst)
    }

    async fn wait_until_parked(&self) -> bool {
        let start = Instant::now();
        while start.elapsed() < LIMIT {
            if self.lock().parked {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        self.lock().parked
    }
}

/// The engine's attached recurrent state.
struct ParkAndCount(Arc<Gate>);

impl RecurrentState for ParkAndCount {
    fn reset_recurrent(&mut self) {
        let gate = &self.0;
        gate.resets.fetch_add(1, Ordering::SeqCst);
        let mut state = gate.lock();
        if !state.armed || state.open {
            return;
        }
        state.parked = true;
        gate.changed.notify_all();
        let _released = gate
            .changed
            .wait_timeout_while(state, LIMIT, |held| !held.open)
            .unwrap_or_else(PoisonError::into_inner);
    }
}

/// Opens the gate when dropped, so a failing test cannot leave the engine
/// thread parked.
struct OpenOnDrop(Arc<Gate>);

impl Drop for OpenOnDrop {
    fn drop(&mut self) {
        self.0.open();
    }
}

struct Server {
    app: axum::Router,
    pool: Arc<EnginePool>,
    gate: Arc<Gate>,
    engine_metrics: Arc<InferenceMetrics>,
    _release: OpenOnDrop,
}

impl Server {
    /// A tokenizer-less single-replica router over a greedy tiny engine, its
    /// served model already resolved (the first resolution takes a replica
    /// of its own and resets nothing).
    async fn start() -> Self {
        let gate = Arc::new(Gate::default());
        let mut engine = InferenceEngine::new(
            Qwen3Config::tiny_test(),
            SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            42,
        );
        engine.set_recurrent_state(Box::new(ParkAndCount(Arc::clone(&gate))));
        let engine_metrics = Arc::new(InferenceMetrics::new());
        engine.set_metrics(Arc::clone(&engine_metrics));
        let pool = EnginePool::new(vec![engine]);
        let app = create_router_full(
            Arc::clone(&pool),
            None,
            Arc::new(InferenceMetrics::new()),
            RouterOptions::default().with_prompt_start_token(QWEN3_IM_START),
        );
        let server = Self {
            app,
            pool,
            _release: OpenOnDrop(Arc::clone(&gate)),
            gate,
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

    /// Answer `body` on `path` normally (the gate unarmed).
    async fn answer(&self, path: &str, body: &serde_json::Value) -> serde_json::Value {
        let resp = self
            .app
            .clone()
            .oneshot(post_to(path, body))
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK, "{path}");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read the body");
        serde_json::from_slice(&bytes).expect("a JSON answer")
    }

    /// Start `body` on `path`, wait until the engine is parked in the
    /// request's first reset, drop the handler future (the client goes
    /// away), then release the engine and wait for the replica.
    async fn abandon(&self, path: &str, body: &serde_json::Value) {
        self.gate.arm();
        let tokens_before = self.engine_metrics.tokens_generated_total.get();
        let request = tokio::spawn(self.app.clone().oneshot(post_to(path, body)));
        assert!(
            self.gate.wait_until_parked().await,
            "{path}: the request never reached the engine"
        );
        request.abort();
        assert!(
            request.await.is_err(),
            "{path}: the request was not abandoned"
        );
        self.gate.open();
        let back = tokio::time::timeout(LIMIT, self.pool.acquire()).await;
        assert!(
            matches!(back, Ok(Ok(_))),
            "{path}: the replica never came back"
        );
        assert_eq!(
            self.engine_metrics.tokens_generated_total.get(),
            tokens_before,
            "{path}: an abandoned request's generation must be cancelled before it produces a token"
        );
    }
}

fn post_to(path: &str, body: &serde_json::Value) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(body).expect("serialize")))
        .expect("request")
}

fn chat_body() -> serde_json::Value {
    serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    })
}

fn extended_body(n: usize) -> serde_json::Value {
    serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "n": n,
    })
}

fn completion_body(prompts: usize) -> serde_json::Value {
    let prompts: Vec<String> = (0..prompts).map(|i| format!("hi {i}")).collect();
    serde_json::json!({ "prompt": prompts, "max_tokens": 8 })
}

#[tokio::test]
async fn an_abandoned_chat_request_generates_nothing() {
    let server = Server::start().await;
    server.abandon("/v1/chat/completions", &chat_body()).await;
    assert_eq!(server.gate.resets(), 1, "the request's one reset");
    // The replica answers the next request in full.
    let answer = server.answer("/v1/chat/completions", &chat_body()).await;
    assert_eq!(answer["usage"]["completion_tokens"], 8, "{answer}");
}

#[tokio::test]
async fn an_abandoned_extended_request_runs_none_of_its_choices() {
    let server = Server::start().await;
    server
        .abandon("/v1/chat/completions/extended", &extended_body(3))
        .await;
    assert_eq!(
        server.gate.resets(),
        1,
        "the request's own reset, and no choice's: none of the three choices ran"
    );
    let answer = server
        .answer("/v1/chat/completions/extended", &extended_body(3))
        .await;
    assert_eq!(
        answer["choices"].as_array().map(Vec::len),
        Some(3),
        "{answer}"
    );
    assert_eq!(answer["usage"]["completion_tokens"], 24, "{answer}");
}

#[tokio::test]
async fn an_abandoned_completion_batch_runs_none_of_its_prompts() {
    let server = Server::start().await;
    server.abandon("/v1/completions", &completion_body(3)).await;
    assert_eq!(
        server.gate.resets(),
        1,
        "the request's own reset, and no prompt's: none of the three prompts ran"
    );
    let answer = server.answer("/v1/completions", &completion_body(3)).await;
    assert_eq!(
        answer["choices"].as_array().map(Vec::len),
        Some(3),
        "{answer}"
    );
    assert_eq!(answer["usage"]["completion_tokens"], 24, "{answer}");
}
