//! In-crate tests of the RAG endpoints (`rag_server.rs`).
//!
//! The engines here are weightless and scripted
//! (`InferenceEngine::script_generation_held`): a held engine's first
//! generation draws one token and then waits — up to the script's own hold
//! limit, five seconds — until its cancellation token is cancelled. A test
//! therefore knows exactly where the generation is (decoding, one token in)
//! when it abandons the request or lets a deadline fire, and a replica that
//! comes back well inside that limit can only have come back because
//! something cancelled the generation.
//!
//! * **Abandonment**: a RAG query whose handler future is dropped
//!   mid-generation frees its replica promptly, and the next request on it
//!   answers in full.
//! * **Deadline**: a router built with a request timeout answers `504
//!   request_timeout` with the stage the deadline caught the request in, and
//!   the replica is free afterwards.
//! * **Not blocking**: on a single-threaded runtime another route answers
//!   while a RAG generation is in flight; every long stage (retrieval,
//!   generation, indexing) is seen to run off the runtime's own thread.
//! * **Unchanged answers**: a query that completes answers the exact bytes
//!   the inline handler answered (for a prompt that fits one prefill chunk; a
//!   longer prompt is prefilled in chunks, as the chat routes' prompts are).

use super::*;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Method, Request};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::metrics::InferenceMetrics;
use crate::sampling::SamplingParams;
use crate::server::{create_router_full, RouterOptions};
use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

/// Every scripted generation's text: 26 visible bytes, then the engine's EOS.
const SCRIPT: &str = "abcdefghijklmnopqrstuvwxyz";

/// How long a replica may take to come back once its request was abandoned
/// or timed out: far below the script's five-second hold, so only a
/// cancelled generation can make it.
const REPLICA_BACK_WITHIN: Duration = Duration::from_millis(2_000);

/// The deadline of the tests that need one to expire while the request is
/// parked. Determinism comes from the parked engine (or the held replica),
/// not from this number: it only has to outlast the request's retrieval and
/// its first token (milliseconds) and stay below the script's hold.
const DEADLINE_MS: u64 = 1_000;

/// Upper bound on waiting for a generation to reach the engine.
const START_LIMIT: Duration = Duration::from_secs(10);

/// A single-replica server over a scripted engine — the chat router with the
/// RAG router merged into it, as `oxibonsai serve --rag` assembles them —
/// and what a test reads.
struct Fixture {
    app: Router,
    pool: Arc<EnginePool>,
    /// The engine's own metrics: a finished prefill is the sign that the
    /// generation reached the engine.
    engine_metrics: Arc<InferenceMetrics>,
    /// The server's metrics, which the RAG router was given.
    server_metrics: Arc<InferenceMetrics>,
}

impl Fixture {
    /// A router without a deadline. `held`: the first generation waits after
    /// its first token until it is cancelled (see the module docs).
    fn new(held: bool) -> Self {
        Self::with_limits(held, RequestLimits::default())
    }

    /// A router whose RAG routes enforce `timeout_ms` as the request deadline.
    fn with_deadline(held: bool, timeout_ms: u64) -> Self {
        Self::with_limits(held, RequestLimits::default().with_timeout_ms(timeout_ms))
    }

    fn with_limits(held: bool, limits: RequestLimits) -> Self {
        let mut engine = fx::weightless_engine(fx::BYTE_VOCAB, SamplingParams::default(), 42);
        if held {
            engine.script_generation_held(fx::byte_ids(SCRIPT), 1);
        } else {
            engine.script_generation(fx::byte_ids(SCRIPT));
        }
        let engine_metrics = Arc::new(InferenceMetrics::new());
        engine.set_metrics(Arc::clone(&engine_metrics));
        let pool = EnginePool::new(vec![engine]);
        let server_metrics = Arc::new(InferenceMetrics::new());
        let chat = create_router_full(
            Arc::clone(&pool),
            None,
            Arc::clone(&server_metrics),
            RouterOptions::default(),
        );
        let rag = create_rag_router_with_options(
            Arc::clone(&pool),
            Some(fx::byte_tokenizer()),
            RagRouterOptions::default()
                .with_limits(limits)
                .with_metrics(Arc::clone(&server_metrics)),
        );
        Self {
            app: chat.merge(rag),
            pool,
            engine_metrics,
            server_metrics,
        }
    }

    /// Whether the pool hands its replica out again within
    /// [`REPLICA_BACK_WITHIN`].
    async fn replica_back(&self) -> bool {
        matches!(
            tokio::time::timeout(REPLICA_BACK_WITHIN, self.pool.acquire()).await,
            Ok(Ok(_))
        )
    }

    /// How many prefills the engine has finished.
    fn prefills(&self) -> u64 {
        self.engine_metrics.prefill_duration_seconds.count()
    }

    /// Wait until a generation has finished a prefill on the engine (the
    /// held script then parks it after one token).
    async fn wait_for_generation(&self, prefills_before: u64) -> bool {
        wait_until(START_LIMIT, || self.prefills() > prefills_before).await
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

fn query_body() -> Value {
    json!({ "query": "hi", "max_tokens": 64 })
}

async fn get(app: Router, path: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(Request::get(path).body(Body::empty()).expect("request"))
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// A server's `504` for `body` (a `/rag/query` request), parsed.
async fn timed_out(fixture: &Fixture, body: Value) -> Value {
    let (status, _, text) = fx::post(fixture.app.clone(), "/rag/query", body).await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{text}");
    serde_json::from_str(&text).unwrap_or(Value::Null)
}

// ── Abandonment ──────────────────────────────────────────────────────────────

/// A client that goes away mid-generation frees the replica at once, and the
/// next request on it answers in full.
///
/// Fails on the handler that generates inline: its future cannot be dropped
/// while the engine runs (the request has already answered by the time the
/// test could abandon it), and nothing cancels the generation, so the
/// replica stays held for the script's whole hold.
#[tokio::test]
async fn an_abandoned_rag_query_frees_its_replica() {
    let fixture = Fixture::new(true);
    let prefills = fixture.prefills();
    let request = tokio::spawn(fx::post(fixture.app.clone(), "/rag/query", query_body()));
    assert!(
        fixture.wait_for_generation(prefills).await,
        "the generation never reached the engine"
    );
    request.abort();
    assert!(
        request.await.is_err(),
        "the request finished on its own; it was meant to be abandoned"
    );
    assert!(
        fixture.replica_back().await,
        "the replica of an abandoned request must come back promptly, not after the \
         generation ran on for nobody"
    );

    // The next request on the same (only) replica answers in full.
    let (status, _, text) = fx::post(fixture.app.clone(), "/rag/query", query_body()).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let answer: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(answer["answer"], SCRIPT, "{answer}");
    assert_eq!(
        answer["usage"]["completion_tokens"],
        SCRIPT.len(),
        "{answer}"
    );
}

/// An abandoned `/rag/index` leaves the index it found: the swap only
/// happens once the client is still there to read the answer.
///
/// Fails on the handler that indexes inline: it indexes and swaps within its
/// first poll, so the request has already completed when the test abandons it.
#[tokio::test(flavor = "current_thread")]
async fn an_abandoned_rag_index_leaves_the_index_untouched() {
    let fixture = Fixture::new(false);
    let documents: Vec<String> = (0..200)
        .map(|i| format!("document {i} covers subject{i} and topic{i} in some detail"))
        .collect();
    let request = tokio::spawn(fx::post(
        fixture.app.clone(),
        "/rag/index",
        json!({ "documents": documents }),
    ));
    // One yield: the handler runs up to the point it waits for its indexing
    // stage (the stage itself runs on the blocking pool, concurrently with
    // this test).
    tokio::task::yield_now().await;
    request.abort();
    assert!(
        request.await.is_err(),
        "the request completed before it could be abandoned; it indexed inline"
    );

    let (status, text) = get(fixture.app.clone(), "/rag/stats").await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let stats: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(
        stats["documents_indexed"], 0,
        "an abandoned index request must not replace the index: {stats}"
    );

    // The same request, not abandoned, indexes.
    let (status, _, text) = fx::post(
        fixture.app.clone(),
        "/rag/index",
        json!({ "documents": ["a second request that is read to the end"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
}

// ── The deadline ─────────────────────────────────────────────────────────────

/// A router built with a request timeout answers `504 request_timeout` once
/// the deadline expires mid-decode, with the chat routes' error object: the
/// stage (`decode`), the tokens generated so far, and a message that names the
/// timeout. The deadline cancels the generation, so the replica is free again
/// well inside the script's hold, and the timeout is counted in the server's
/// metrics.
///
/// This test cannot compile against the handler that predates the deadline:
/// `RagRouterOptions` does not exist there. Run against an inline handler
/// that ignores the options, it fails on the status: the generation holds
/// for the script's five seconds and the request answers `200`.
#[tokio::test]
async fn a_deadline_in_decode_answers_504_and_frees_the_replica() {
    let fixture = Fixture::with_deadline(true, DEADLINE_MS);
    let error = timed_out(&fixture, query_body()).await;
    assert_eq!(error["error"]["code"], "request_timeout", "{error}");
    assert_eq!(error["error"]["type"], "server_error", "{error}");
    assert_eq!(error["error"]["phase"], "decode", "{error}");
    assert_eq!(error["error"]["generated_tokens"], 1, "{error}");
    let message = error["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains(&format!("per-request timeout of {DEADLINE_MS} ms")),
        "{message}"
    );
    assert_eq!(fixture.server_metrics.errors_total.get(), 1);
    assert!(
        fixture.replica_back().await,
        "the deadline must cancel the generation, freeing its replica"
    );

    // The next request on the replica answers in full: the deadline's cancel
    // died with the request that armed it.
    let (status, _, text) = fx::post(fixture.app.clone(), "/rag/query", query_body()).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let answer: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(answer["answer"], SCRIPT, "{answer}");
}

/// A request that cannot get a replica is caught by the deadline in the
/// queue, and says so: the retrieval finished, the engine did not start.
#[tokio::test]
async fn a_deadline_while_queued_for_a_replica_names_the_queue() {
    let fixture = Fixture::with_deadline(false, DEADLINE_MS);
    let held = fixture.pool.acquire().await.expect("the only replica");
    let error = timed_out(&fixture, query_body()).await;
    assert_eq!(error["error"]["code"], "request_timeout", "{error}");
    assert_eq!(error["error"]["phase"], "waiting_for_engine", "{error}");
    assert!(
        error["error"].get("generated_tokens").is_none(),
        "no token was generated: {error}"
    );
    drop(held);
    assert!(fixture.replica_back().await);
}

/// Without a configured timeout nothing answers `504`, and a timeout of `0`
/// is no timeout (`RequestLimits::with_timeout_ms`); a generous one changes
/// no answer (see the golden test below).
#[test]
fn a_zero_or_missing_timeout_is_no_deadline() {
    let pool = EnginePool::new(vec![InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        SamplingParams::default(),
        42,
    )]);
    let state = RagState::new(Arc::clone(&pool), None);
    assert_eq!(state.request_timeout, None);
    let state = RagState::new(Arc::clone(&pool), None).with_options(
        RagRouterOptions::default().with_limits(RequestLimits::default().with_timeout_ms(0)),
    );
    assert_eq!(state.request_timeout, None);
    let state = RagState::new(pool, None).with_options(
        RagRouterOptions::default().with_limits(RequestLimits::default().with_timeout_ms(1_234)),
    );
    assert_eq!(state.request_timeout, Some(Duration::from_millis(1_234)));
}

// ── The async worker is not blocked ──────────────────────────────────────────

/// While a RAG generation is in flight on a single-threaded runtime, other
/// routes — a probe of the chat router and a RAG route — answer: the
/// generation does not occupy the only worker.
///
/// Fails on the handler that generates inline: the runtime's one thread is
/// inside the engine for the script's whole hold, so by the time the test
/// could send the second request the RAG request has already finished.
#[tokio::test(flavor = "current_thread")]
async fn other_routes_answer_while_a_rag_generation_is_in_flight() {
    let fixture = Fixture::new(true);
    let prefills = fixture.prefills();
    let rag = tokio::spawn(fx::post(fixture.app.clone(), "/rag/query", query_body()));
    assert!(
        fixture.wait_for_generation(prefills).await,
        "the generation never reached the engine"
    );
    assert!(
        !rag.is_finished(),
        "the RAG request had already finished when the generation was seen to start: it ran \
         inline on the async worker"
    );

    for path in ["/health", "/rag/stats"] {
        let (status, text) = get(fixture.app.clone(), path).await;
        assert_eq!(status, StatusCode::OK, "{path}: {text}");
        assert!(
            !rag.is_finished(),
            "{path}: the generation was still in flight while the route answered"
        );
    }

    rag.abort();
    assert!(
        rag.await.is_err(),
        "the held request is abandoned, not answered"
    );
    assert!(fixture.replica_back().await);
}

/// Every long stage of a request — indexing, retrieval and generation — runs
/// on a thread of the blocking pool, never on the thread that drives the
/// handlers (here the test's own: the runtime is single-threaded).
///
/// The handler that did this work inline has no stage record to ask: this
/// test cannot compile against it.
#[tokio::test(flavor = "current_thread")]
async fn every_long_stage_runs_off_the_runtime_thread() {
    let pool = EnginePool::new(vec![fx::scripted_byte_engine(SCRIPT)]);
    let state = Arc::new(RagState::new(pool, Some(fx::byte_tokenizer())));
    let app = Router::new()
        .route("/rag/index", axum::routing::post(index_documents))
        .route("/rag/query", axum::routing::post(rag_query))
        .with_state(Arc::clone(&state));

    let (status, _, text) = fx::post(
        app.clone(),
        "/rag/index",
        json!({ "documents": ["Rust is a systems programming language."] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let (status, _, text) = fx::post(
        app,
        "/rag/query",
        json!({ "query": "What is Rust?", "max_tokens": 64 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let answer: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(answer["answer"], SCRIPT, "{answer}");

    let runtime_thread = std::thread::current().id();
    let log = state
        .stage_log
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    let stages: Vec<&str> = log.iter().map(|(stage, _)| *stage).collect();
    assert_eq!(stages, ["indexing", "retrieval", "generation"]);
    for (stage, thread) in log {
        assert_ne!(
            thread, runtime_thread,
            "the {stage} stage ran on the thread that drives the async handlers"
        );
    }
}

/// A stage whose handler future is dropped sees its token cancelled, and one
/// that finishes in time does not: the guard `run_stage` holds cancels
/// exactly the abandoned.
#[tokio::test]
async fn an_abandoned_stage_sees_its_token_cancelled_and_a_finished_one_does_not() {
    let pool = EnginePool::new(vec![fx::scripted_byte_engine(SCRIPT)]);
    let state = Arc::new(RagState::new(pool, None));

    // Finished in time: the token stays live and the result comes back.
    let live = CancellationToken::new();
    let value = state
        .run_stage("probe", &live, |token| (7, token.is_cancelled()))
        .await
        .expect("the stage ran");
    assert_eq!(value, (7, false));
    assert!(!live.is_cancelled());

    // Abandoned while running: the stage is parked until the test lets go.
    let abandoned = CancellationToken::new();
    let running = Arc::new(AtomicBool::new(false));
    let (release, parked) = std::sync::mpsc::channel::<()>();
    let (observed_tx, observed) = std::sync::mpsc::channel::<bool>();
    let task = {
        let state = Arc::clone(&state);
        let abandoned = abandoned.clone();
        let running = Arc::clone(&running);
        tokio::spawn(async move {
            state
                .run_stage("probe", &abandoned, move |token| {
                    running.store(true, Ordering::SeqCst);
                    let _ = parked.recv_timeout(START_LIMIT);
                    let _ = observed_tx.send(token.is_cancelled());
                })
                .await
        })
    };
    assert!(
        wait_until(START_LIMIT, || running.load(Ordering::SeqCst)).await,
        "the stage never started"
    );
    task.abort();
    assert!(task.await.is_err());
    assert!(
        abandoned.is_cancelled(),
        "dropping the handler cancels the stage"
    );
    release.send(()).expect("the stage is still parked");
    assert_eq!(
        observed.recv_timeout(START_LIMIT),
        Ok(true),
        "the running stage observes the cancellation"
    );
}

/// A stage that panics is a `500`, not a hung or poisoned server.
#[tokio::test]
async fn a_panicking_stage_is_a_500() {
    let pool = EnginePool::new(vec![fx::scripted_byte_engine(SCRIPT)]);
    let state = Arc::new(RagState::new(pool, None));
    let outcome = state
        .run_stage("probe", &CancellationToken::new(), |_token| -> u8 {
            panic!("a stage that fails")
        })
        .await;
    let Err(error) = outcome else {
        panic!("a panicking stage must not report success");
    };
    assert_eq!(error.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(error.message(), "RAG probe task failed");
}

/// The two blocking stages' early exits: a request already over starts no
/// work.
#[test]
fn a_stage_whose_request_is_over_does_no_work() {
    let cancelled = CancellationToken::new();
    cancelled.cancel();

    let documents = vec!["one document".to_string(), "another document".to_string()];
    assert!(matches!(
        build_index(&documents, ChunkConfig::default(), &cancelled),
        Err(IndexError::Cancelled)
    ));
    let live = CancellationToken::new();
    let Ok(built) = build_index(&documents, ChunkConfig::default(), &live) else {
        panic!("a live request indexes");
    };
    assert_eq!(built.document_ids, [0, 1]);
    assert!(built.total_chunks >= 2);

    let pool = EnginePool::new(vec![fx::scripted_byte_engine(SCRIPT)]);
    let state = RagState::new(pool, None);
    let Err(error) = state.prepare_query("hi", 3, &cancelled) else {
        panic!("an abandoned request retrieves nothing");
    };
    assert_eq!(error.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(state.prepare_query("hi", 3, &live).is_ok());
}

// ── A normally completed query is unchanged ──────────────────────────────────

/// One query's response body, exactly as the client reads it.
async fn query_text(app: &Router, body: Value) -> String {
    let (status, _, text) = fx::post(app.clone(), "/rag/query", body).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    text
}

/// Index one document, then ask two questions of one replica whose every
/// draw is up to the sampler's own PRNG (`uniform_letters_engine`).
async fn indexed_questions(app: &Router) -> (String, String) {
    let (status, _, text) = fx::post(
        app.clone(),
        "/rag/index",
        json!({ "documents": ["Photosynthesis converts sunlight into chemical energy."] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let first = query_text(
        app,
        json!({ "query": "How do plants make energy?", "max_tokens": 12, "include_context": true }),
    )
    .await;
    let second = query_text(
        app,
        json!({ "query": "What does sunlight become?", "max_tokens": 9, "temperature": 1.5 }),
    )
    .await;
    (first, second)
}

/// The answers, the prompt, the retrieved context and the usage figures of
/// two consecutive queries on one replica are the exact bytes the handler
/// produced when it still generated inline on the async worker: any drift in
/// the sampler's draws (a changed parameter swap, a changed reset, a changed
/// order of calls) shows up as a different letter.
///
/// The goldens were recorded from that inline handler with this engine, seed
/// and these requests, so the test passes on both sides of the move onto the
/// blocking pool by construction. Both prompts are 101 tokens, below the
/// 512-token prefill chunk: it pins the draws and the response bytes, not the
/// numerics of a chunked prefill (`a_long_prompt_is_prefilled_in_chunks_and_answers_in_full`
/// covers the chunked path's behaviour).
#[tokio::test]
async fn a_completed_rag_query_answers_the_recorded_golden_bytes() {
    let pool = EnginePool::new(vec![fx::uniform_letters_engine(7)]);
    let app = create_rag_router_with_pool(pool, Some(fx::byte_tokenizer()));
    let (first, second) = indexed_questions(&app).await;
    assert_eq!(first, GOLDEN_FIRST);
    assert_eq!(second, GOLDEN_SECOND);
}

/// The same flow under a generous deadline: nothing is cancelled, nothing is
/// counted, and the answers are the same bytes.
#[tokio::test]
async fn a_generous_deadline_changes_no_rag_answer() {
    let metrics = Arc::new(InferenceMetrics::new());
    let pool = EnginePool::new(vec![fx::uniform_letters_engine(7)]);
    let app = create_rag_router_with_options(
        pool,
        Some(fx::byte_tokenizer()),
        RagRouterOptions::default()
            .with_limits(RequestLimits::default().with_timeout_ms(600_000))
            .with_metrics(Arc::clone(&metrics)),
    );
    let (first, second) = indexed_questions(&app).await;
    assert_eq!(first, GOLDEN_FIRST);
    assert_eq!(second, GOLDEN_SECOND);
    assert_eq!(metrics.errors_total.get(), 0);
}

/// A prompt longer than a prefill chunk is ingested in chunks — the request
/// arms its replica exactly as the chat routes do, so a deadline or a
/// departed client is observed between chunks of a long retrieved context —
/// and still answers the whole script.
#[tokio::test]
async fn a_long_prompt_is_prefilled_in_chunks_and_answers_in_full() {
    use crate::server::deadline::CANCELLATION_PREFILL_CHUNK_TOKENS;

    let fixture = Fixture::new(false);
    let document = "chunked prefill keeps long prompts cancellable. ".repeat(24);
    let (status, _, text) = fx::post(
        fixture.app.clone(),
        "/rag/index",
        json!({ "documents": [document] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");

    let (status, _, text) = fx::post(
        fixture.app.clone(),
        "/rag/query",
        json!({ "query": "chunked prefill", "max_tokens": 64 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let answer: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    let prompt_tokens = answer["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    // A tail of at most `PREFILL_PER_TOKEN_MAX_TOKENS` is folded into the
    // previous window (F-3), so one chunk plus that many tokens is one call.
    let one_call_max = CANCELLATION_PREFILL_CHUNK_TOKENS
        + oxibonsai_model::chunked_prefill::PREFILL_PER_TOKEN_MAX_TOKENS;
    assert!(
        prompt_tokens > one_call_max as u64,
        "the prompt must be longer than one chunk (plus a foldable tail) to be chunked: \
         {prompt_tokens}"
    );
    assert_eq!(answer["answer"], SCRIPT, "{answer}");

    // The replica was armed like a chat request's: prefill in chunks.
    let lease = fixture.pool.acquire().await.expect("the only replica");
    assert_eq!(
        lease.prefill_chunk_tokens(),
        Some(CANCELLATION_PREFILL_CHUNK_TOKENS)
    );
}

const GOLDEN_FIRST: &str = "{\"answer\":\"cqmqfwegqarj\",\"retrieved_chunks\":[\"Photosynthesis converts sunlight into chemical energy.\"],\"prompt_used\":\"Photosynthesis converts sunlight into chemical energy.\\n\\nQuestion: How do plants make energy?\\n\\nAnswer:\",\"usage\":{\"documents_searched\":1,\"chunks_retrieved\":1,\"prompt_tokens\":101,\"completion_tokens\":12}}";
const GOLDEN_SECOND: &str = "{\"answer\":\"ordgnbmoa\",\"retrieved_chunks\":null,\"prompt_used\":\"Photosynthesis converts sunlight into chemical energy.\\n\\nQuestion: What does sunlight become?\\n\\nAnswer:\",\"usage\":{\"documents_searched\":1,\"chunks_retrieved\":1,\"prompt_tokens\":101,\"completion_tokens\":9}}";

/// A rejected query keeps the shared error envelope byte for byte (the
/// handler now builds it with the chat routes' `ApiError`, which renders the
/// same body): the zero-norm `400` below was recorded from the handler that
/// built it with the shared `error_response` helper directly.
#[tokio::test]
async fn a_rejected_query_keeps_its_error_body_byte_for_byte() {
    let pool = EnginePool::new(vec![fx::scripted_byte_engine(SCRIPT)]);
    let app = create_rag_router_with_pool(pool, Some(fx::byte_tokenizer()));
    let (status, _, text) = fx::post(
        app.clone(),
        "/rag/index",
        json!({ "documents": ["Rust is a systems programming language."] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");

    let (status, _, text) = fx::post(
        app.clone(),
        "/rag/query",
        json!({ "query": "zzqvx wwpqr fjklm bbxyzq" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        text,
        "{\"error\":{\"code\":null,\"message\":\"query embedding has zero norm; refusing to \
         return an arbitrary ranking\",\"param\":null,\"type\":\"invalid_request_error\"}}"
    );

    for (body, message) in [
        (json!({ "query": "  " }), "query must not be empty"),
        (
            json!({ "query": "Rust", "top_k": 51 }),
            "top_k (51) must be between 1 and 50 inclusive",
        ),
    ] {
        let (status, _, text) = fx::post(app.clone(), "/rag/query", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
        assert_eq!(
            text,
            format!(
                "{{\"error\":{{\"code\":null,\"message\":\"{message}\",\"param\":null,\
                 \"type\":\"invalid_request_error\"}}}}"
            )
        );
    }
}

/// The error type the handlers now return renders the body of the helper they
/// returned before, for every status they use.
#[tokio::test]
async fn api_errors_render_the_shared_envelope() {
    for status in [
        StatusCode::BAD_REQUEST,
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::SERVICE_UNAVAILABLE,
    ] {
        let new = api_error(status, "a message").into_response();
        let old = error_response(status, "a message");
        assert_eq!(new.status(), old.status());
        assert_eq!(new.headers(), old.headers());
        let new_body = axum::body::to_bytes(new.into_body(), usize::MAX).await;
        let old_body = axum::body::to_bytes(old.into_body(), usize::MAX).await;
        assert_eq!(new_body.expect("body"), old_body.expect("body"));
    }
}

/// A blank document is refused where it was refused before: `400`, naming the
/// document, with the index left as it was.
#[tokio::test]
async fn a_document_that_cannot_be_indexed_is_a_400_naming_it() {
    let pool = EnginePool::new(vec![fx::scripted_byte_engine(SCRIPT)]);
    let app = create_rag_router_with_pool(pool, None);
    let (status, _, text) = fx::post(
        app.clone(),
        "/rag/index",
        json!({ "documents": ["a document with words", "   "] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    let error: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    let message = error["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.starts_with("failed to index document 1:"),
        "{message}"
    );
    let (_, text) = get(app, "/rag/stats").await;
    let stats: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(stats["documents_indexed"], 0, "{stats}");
}

/// A lock poisoned by a panicking holder no longer takes the endpoints down:
/// the lock guards a pointer to an immutable pipeline, which is valid whatever
/// panicked while it was held.
#[tokio::test]
async fn a_poisoned_pipeline_lock_is_recovered() {
    let pool = EnginePool::new(vec![fx::scripted_byte_engine(SCRIPT)]);
    let state = Arc::new(RagState::new(pool, None));
    let caught: Result<(), _> = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = state.pipeline.lock();
        panic!("a holder panics with the pipeline lock held");
    }));
    assert!(caught.is_err());
    assert!(state.pipeline.is_poisoned());

    let app = Router::new()
        .route("/rag/stats", axum::routing::get(rag_stats))
        .route("/rag/index", axum::routing::post(index_documents))
        .with_state(Arc::clone(&state));
    let (status, text) = get(app.clone(), "/rag/stats").await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let (status, _, text) = fx::post(
        app.clone(),
        "/rag/index",
        json!({ "documents": ["indexed through a recovered lock"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let (_, text) = get(app, "/rag/stats").await;
    let stats: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(stats["documents_indexed"], 1, "{stats}");
}

// ── Moved from the inline test module ────────────────────────────────────────

#[test]
fn human_bytes_formatting() {
    assert_eq!(human_bytes(0), "0 B");
    assert_eq!(human_bytes(512), "512 B");
    assert_eq!(human_bytes(1024), "1.00 KiB");
    assert_eq!(human_bytes(1024 * 1024), "1.00 MiB");
    assert_eq!(human_bytes(1024 * 1024 * 1024), "1.00 GiB");
}

#[test]
fn rough_token_count_basic() {
    assert_eq!(rough_token_count(""), 0);
    assert_eq!(rough_token_count("one two three"), 3);
    assert_eq!(rough_token_count("  spaces  everywhere  "), 2);
}

#[test]
fn rag_state_creates_without_panic() {
    use oxibonsai_core::config::Qwen3Config;

    let config = Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
    let pool = EnginePool::new(vec![engine]);
    let _state = RagState::new(pool, None);
}

// ── RAG, unowned sibling: /rag/query must route through the
// retriever's own guarded retrieval API ────────────────────────────────────

#[tokio::test]
async fn rag_query_rejects_a_fully_out_of_vocabulary_query_against_a_nonempty_index() {
    use oxibonsai_core::config::Qwen3Config;

    let config = Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
    let app = create_rag_router(engine);

    // `/rag/index` refits the TF-IDF vocabulary from exactly the
    // submitted documents (BOOTSTRAP_CORPUS is not merged in), so the
    // vocabulary here is precisely the union of these three documents'
    // tokens -- several unrelated documents, so a positive-control
    // query sharing vocabulary with only one of them still has to
    // survive ranking against the others.
    let index_req = Request::builder()
        .method(Method::POST)
        .uri("/rag/index")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "documents": [
                    "Rust is a systems programming language with memory safety.",
                    "Python is popular for data science and machine learning.",
                    "Tokyo is the capital of Japan and a major travel destination."
                ]
            })
            .to_string(),
        ))
        .expect("build index request");
    let index_resp = app
        .clone()
        .oneshot(index_req)
        .await
        .expect("index response");
    assert_eq!(
        index_resp.status(),
        StatusCode::OK,
        "sanity: indexing three documents must succeed"
    );

    // Positive control: an in-vocabulary query must still succeed with
    // 200, so a 400 on the OOV query below is known to come from the
    // zero-norm guard specifically, not from every query being
    // rejected (e.g. a broken embedder or an over-eager guard).
    let control_req = Request::builder()
        .method(Method::POST)
        .uri("/rag/query")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({ "query": "Rust memory safety", "max_tokens": 1 }).to_string(),
        ))
        .expect("build control request");
    let control_resp = app
        .clone()
        .oneshot(control_req)
        .await
        .expect("control response");
    assert_eq!(
        control_resp.status(),
        StatusCode::OK,
        "sanity: an in-vocabulary query must still return 200"
    );

    // Every token here is out-of-vocabulary against the just-indexed
    // documents' TF-IDF vocabulary, so the query embeds to an all-zero
    // vector -- degenerate for the store's default Cosine metric
    // (RAG-21 / RAG-EVAL-IMG-21). Before this fix, `/rag/query`
    // bypassed `Retriever::retrieve`'s guard entirely (calling
    // `retriever.store().search_with_threshold` directly) and silently
    // returned 200 OK with an arbitrary insertion-order ranking
    // instead.
    let query_req = Request::builder()
        .method(Method::POST)
        .uri("/rag/query")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({ "query": "zzqvx wwpqr fjklm bbxyzq" }).to_string(),
        ))
        .expect("build query request");
    let query_resp = app.oneshot(query_req).await.expect("query response");
    assert_eq!(
        query_resp.status(),
        StatusCode::BAD_REQUEST,
        "a fully out-of-vocabulary query against a non-empty Cosine-metric index must be \
         rejected (RAG-21), not silently ranked"
    );
}
