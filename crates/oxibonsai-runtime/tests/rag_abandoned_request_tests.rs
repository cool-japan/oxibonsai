//! A `/rag/query` request whose client goes away, or whose deadline expires,
//! through the real RAG router: the generation it started is cancelled before
//! it produces a token, and the replica serves the next request in full.
//!
//! Deterministic, with no timing involved in what is asserted: the engine
//! carries an attached [`RecurrentState`] (a public extension point the engine
//! resets at the start of every generation) that counts every reset and parks
//! the engine thread in the request's first one until the test lets go. The
//! test drops the handler future — or lets the request's deadline expire —
//! while the engine is parked, and only then releases it. From that point:
//!
//! * with the abandonment guard (and the deadline's own cancel), the request's
//!   token is already cancelled, so the generation produces no token (the
//!   engine's own `tokens_generated_total` stays at zero) and the replica
//!   returns to the pool;
//! * without them the generation runs to its end and generates tokens.
//!
//! A handler that generates inline never resets the engine through the
//! blocking-pool seam, so it never parks: against it these tests fail on "the
//! request never reached the engine". The in-crate tests
//! (`rag_server::tests`) cover the same on a generation abandoned mid-decode.

#![cfg(feature = "rag")]

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
use oxibonsai_runtime::rag_server::{create_rag_router_with_options, RagRouterOptions};
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::RequestLimits;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;

/// Upper bound on any wait in these tests; a parked engine thread gives up
/// on its own after it too, so a failing test cannot hang the binary.
const LIMIT: Duration = Duration::from_secs(30);

/// The vocabulary of the byte-level tokenizer below (256 byte tokens and
/// `<|im_end|>`) and of the engine that pairs with it.
const VOCAB: usize = 257;

/// How many tokens a query asks for; a greedy engine over equal logits never
/// reaches its EOS, so a completed query reports exactly this many.
const MAX_TOKENS: usize = 8;

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

/// A byte-level BPE tokenizer: id = byte value, plus `<|im_end|>` as id 256,
/// so every id the engine below can sample decodes.
fn byte_tokenizer() -> TokenizerBridge {
    let mut vocab = serde_json::Map::new();
    for byte in 0..=255u8 {
        vocab.insert(
            oxibonsai_tokenizer::byte_to_unicode(byte).to_string(),
            serde_json::Value::from(u32::from(byte)),
        );
    }
    let json = serde_json::json!({
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "added_tokens": [{ "id": 256, "content": "<|im_end|>", "special": true }],
        "pre_tokenizer": { "type": "ByteLevel" },
        "decoder": { "type": "ByteLevel" },
    })
    .to_string();
    TokenizerBridge::native_from_json_str(&json).expect("the byte-level tokenizer loads")
}

struct Server {
    app: axum::Router,
    pool: Arc<EnginePool>,
    gate: Arc<Gate>,
    engine_metrics: Arc<InferenceMetrics>,
    server_metrics: Arc<InferenceMetrics>,
    _release: OpenOnDrop,
}

impl Server {
    /// A single-replica RAG router over a tiny engine (equal logits, so a
    /// greedy request always draws id 0), enforcing `timeout_ms` as the
    /// request deadline when given.
    fn start(timeout_ms: Option<u64>) -> Self {
        let gate = Arc::new(Gate::default());
        let config = Qwen3Config {
            vocab_size: VOCAB,
            max_context_length: 4096,
            ..Qwen3Config::tiny_test()
        };
        let mut engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        engine.set_eos_token_ids([256]);
        engine.set_recurrent_state(Box::new(ParkAndCount(Arc::clone(&gate))));
        let engine_metrics = Arc::new(InferenceMetrics::new());
        engine.set_metrics(Arc::clone(&engine_metrics));
        let pool = EnginePool::new(vec![engine]);
        let server_metrics = Arc::new(InferenceMetrics::new());
        let mut limits = RequestLimits::default();
        if let Some(timeout_ms) = timeout_ms {
            limits = limits.with_timeout_ms(timeout_ms);
        }
        let app = create_rag_router_with_options(
            Arc::clone(&pool),
            Some(byte_tokenizer()),
            RagRouterOptions::default()
                .with_limits(limits)
                .with_metrics(Arc::clone(&server_metrics)),
        );
        Self {
            app,
            pool,
            _release: OpenOnDrop(Arc::clone(&gate)),
            gate,
            engine_metrics,
            server_metrics,
        }
    }

    /// Send the query and read the whole answer.
    async fn ask(&self) -> (StatusCode, serde_json::Value) {
        let response = self
            .app
            .clone()
            .oneshot(query_request())
            .await
            .expect("response");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read the body");
        (
            status,
            serde_json::from_slice(&bytes).expect("a JSON answer"),
        )
    }

    /// Whether the pool hands its replica out again within [`LIMIT`].
    async fn replica_back(&self) -> bool {
        matches!(
            tokio::time::timeout(LIMIT, self.pool.acquire()).await,
            Ok(Ok(_))
        )
    }
}

/// A greedy query (equal logits make every draw id 0).
fn query_request() -> Request<Body> {
    Request::post("/rag/query")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "query": "hi",
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            })
            .to_string(),
        ))
        .expect("request")
}

/// A client that goes away while its generation is parked at the start: the
/// generation, released afterwards, produces no token at all, and the replica
/// answers the next request in full.
#[tokio::test]
async fn an_abandoned_rag_query_generates_nothing() {
    let server = Server::start(None);
    server.gate.arm();
    let tokens_before = server.engine_metrics.tokens_generated_total.get();
    let request = tokio::spawn(server.app.clone().oneshot(query_request()));
    assert!(
        server.gate.wait_until_parked().await,
        "the request never reached the engine"
    );
    request.abort();
    assert!(
        request.await.is_err(),
        "the request was not abandoned: it answered on its own"
    );
    server.gate.open();
    assert!(server.replica_back().await, "the replica never came back");
    assert_eq!(
        server.engine_metrics.tokens_generated_total.get(),
        tokens_before,
        "an abandoned request's generation must be cancelled before it produces a token"
    );
    assert_eq!(server.gate.resets(), 1, "the request's one reset");

    // The replica answers the next request in full.
    let (status, answer) = server.ask().await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["usage"]["completion_tokens"], MAX_TOKENS, "{answer}");
}

/// A request whose deadline expires while its generation is parked at the
/// start answers `504 request_timeout`, names the stage it was caught in
/// (`prefill`: it holds a replica and has produced no token), and cancels the
/// generation, which — released afterwards — produces no token.
#[tokio::test]
async fn a_rag_query_past_its_deadline_answers_504_and_cancels_the_generation() {
    // The deadline only has to outlast the request's retrieval and the engine
    // reaching its parked reset (milliseconds); what is asserted does not
    // depend on how much longer it is.
    let server = Server::start(Some(1_000));
    server.gate.arm();
    let tokens_before = server.engine_metrics.tokens_generated_total.get();
    let (status, answer) = server.ask().await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{answer}");
    assert_eq!(answer["error"]["code"], "request_timeout", "{answer}");
    assert_eq!(answer["error"]["type"], "server_error", "{answer}");
    assert_eq!(answer["error"]["phase"], "prefill", "{answer}");
    assert!(
        answer["error"].get("generated_tokens").is_none(),
        "no token was generated: {answer}"
    );
    assert_eq!(server.server_metrics.errors_total.get(), 1);
    assert!(
        server.gate.wait_until_parked().await,
        "the generation was parked when the deadline fired"
    );

    server.gate.open();
    assert!(server.replica_back().await, "the replica never came back");
    assert_eq!(
        server.engine_metrics.tokens_generated_total.get(),
        tokens_before,
        "the deadline must cancel the generation before it produces a token"
    );
}

/// A generous deadline changes nothing: the query answers in full, and
/// nothing is counted as an error.
#[tokio::test]
async fn a_rag_query_inside_its_deadline_answers_in_full() {
    let server = Server::start(Some(600_000));
    let (status, answer) = server.ask().await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["usage"]["completion_tokens"], MAX_TOKENS, "{answer}");
    assert_eq!(server.server_metrics.errors_total.get(), 0);
    assert!(server.replica_back().await);
}
