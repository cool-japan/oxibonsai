//! OpenAI-compatible RAG endpoints for OxiBonsai.
//!
//! Feature-gated with `#[cfg(feature = "rag")]`. The feature brings the
//! `server` feature with it: the RAG routes run their generation through the
//! chat routes' blocking-pool seam and answer to the server's per-request
//! deadline.
//!
//! # Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | POST | `/rag/index` | Index documents into the RAG store |
//! | POST | `/rag/query` | RAG-augmented generation |
//! | GET | `/rag/stats` | Pipeline statistics as JSON |
//! | DELETE | `/rag/index` | Clear the vector index |
//!
//! # Usage
//!
//! ```rust,no_run
//! use oxibonsai_runtime::rag_server::create_rag_router;
//! use oxibonsai_runtime::engine::InferenceEngine;
//! use oxibonsai_core::config::Qwen3Config;
//! use oxibonsai_runtime::sampling::SamplingParams;
//!
//! let config = Qwen3Config::tiny_test();
//! let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
//! let router = create_rag_router(engine);
//! ```
//!
//! # Where the work runs
//!
//! No handler runs a long synchronous computation on an async worker thread,
//! where it would starve every other request served by that worker (the
//! probe routes included):
//!
//! | Route | Work | Where it runs |
//! |-------|------|---------------|
//! | `POST /rag/query` | embedding the query, searching the vector store, assembling the prompt, tokenising it | the blocking pool, as one stage |
//! | `POST /rag/query` | the generation | the blocking pool, through the chat routes' `run_blocking_generation` |
//! | `POST /rag/index` | fitting the TF-IDF vocabulary, chunking and embedding every document | the blocking pool |
//! | `GET /rag/stats` | a pointer read and one pass over the stored chunks' sizes (a few nanoseconds per chunk) | the handler, off a snapshot: no lock is held across the pass |
//! | `DELETE /rag/index` | a fit over the fixed bootstrap corpus | the handler (microseconds) |
//!
//! The pipeline is shared as an immutable snapshot: a query clones the
//! pointer under the state's lock and searches without holding it, and an
//! index or clear swaps in a whole new pipeline. No lock is ever held across
//! work, so a long search never delays `/rag/stats` or an index swap, and an
//! index that is replaced mid-query leaves that query on a consistent view.
//!
//! # Deadline and cancellation
//!
//! `/rag/query` behaves like the chat routes' non-streamed path:
//!
//! * a handler future that is dropped before the answer is in hand — the
//!   client disconnected, or a layer outside the handler (the admission
//!   layer's timeout) gave up on it — cancels the generation at its next
//!   step, so the replica serves the next request instead of decoding to
//!   `max_tokens` for nobody;
//! * a router built with a request timeout ([`RagRouterOptions::with_limits`])
//!   enforces it inside the handler: an expired deadline cancels the
//!   generation and answers `504` with `error.code: request_timeout`,
//!   `error.phase` naming the stage the request was in (`preparing` while it
//!   retrieves and tokenises, `waiting_for_engine`, `prefill`, `decode`) and,
//!   in decode, `error.generated_tokens` — the error object of the chat
//!   routes, not a new one.
//!
//! `/rag/index` has no deadline of its own (the admission layer's timeout is
//! its backstop), but a request that is dropped stops its indexing loop
//! between documents and never replaces the index: an abandoned `/rag/index`
//! leaves the index it found. A single document is indexed in one library
//! call, so the loop cannot stop inside it.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::Router;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use oxibonsai_rag::chunker::ChunkConfig;
use oxibonsai_rag::embedding::TfIdfEmbedder;
use oxibonsai_rag::pipeline::{RagConfig, RagPipeline};

use crate::engine::InferenceEngine;
use crate::engine_control::CancellationToken;
use crate::engine_pool::EnginePool;
use crate::metrics::InferenceMetrics;
use crate::server::api_error::ApiError;
use crate::server::blocking::CancelOnAbandon;
use crate::server::deadline::{enforce_deadline, CancelSlot};
use crate::server::RequestLimits;
use crate::tokenizer_bridge::TokenizerBridge;

mod index;
mod query;

use index::{build_index, BuiltIndex, IndexError};

// ─────────────────────────────────────────────────────────────────────────────
// Default corpus used to bootstrap the TF-IDF vocabulary.
//
// TfIdfEmbedder needs a corpus to `fit()` its vocabulary.  We pre-seed it
// with a small general-purpose corpus so the embedder is immediately usable
// before any documents are indexed.  Once `index_documents` is called the
// pipeline is rebuilt with the new corpus vocabulary.
// ─────────────────────────────────────────────────────────────────────────────

const BOOTSTRAP_CORPUS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence and machine learning are transforming software.",
    "Rust is a systems programming language focused on safety performance and concurrency.",
    "Retrieval-augmented generation combines search with language model generation.",
    "Vector embeddings represent semantic meaning in high-dimensional space.",
];

/// Default vocabulary size cap for `TfIdfEmbedder`.
const DEFAULT_MAX_FEATURES: usize = 512;

// ─────────────────────────────────────────────────────────────────────────────
// Request / Response types
// ─────────────────────────────────────────────────────────────────────────────

/// Request body for `POST /rag/index`.
#[derive(Debug, Deserialize)]
pub struct IndexDocumentRequest {
    /// Raw text documents to index.
    pub documents: Vec<String>,
    /// Character window size for chunking (default: `ChunkConfig` default).
    pub chunk_size: Option<usize>,
    /// Character overlap between adjacent chunks (default: `ChunkConfig` default).
    pub chunk_overlap: Option<usize>,
}

/// Response body for `POST /rag/index`.
#[derive(Debug, Serialize)]
pub struct IndexDocumentResponse {
    /// Number of documents successfully indexed.
    pub indexed: usize,
    /// Total number of chunks stored in the vector index.
    pub chunks: usize,
    /// Assigned document identifiers (one per document, 0-based sequential).
    pub document_ids: Vec<usize>,
}

/// Request body for `POST /rag/query`.
#[derive(Debug, Deserialize)]
pub struct RagQueryRequest {
    /// The question or query string.
    pub query: String,
    /// Maximum number of tokens to generate (default: 256).
    pub max_tokens: Option<usize>,
    /// Number of context chunks to retrieve (default: 3).
    pub top_k: Option<usize>,
    /// Sampling temperature forwarded to the inference engine when a tokenizer
    /// is configured. Applied only when finite and within `[0.0, 2.0]`.
    pub temperature: Option<f32>,
    /// When `true`, the retrieved chunks are included in the response.
    pub include_context: Option<bool>,
}

/// Response body for `POST /rag/query`.
#[derive(Debug, Serialize)]
pub struct RagQueryResponse {
    /// Generated answer from the language model.
    pub answer: String,
    /// The context chunks that were retrieved (present when
    /// `include_context: true` was requested).
    pub retrieved_chunks: Option<Vec<String>>,
    /// The full prompt that was built from the retrieved context and the query.
    /// When a tokenizer is configured this prompt is encoded and sent to the
    /// model; without a tokenizer it is returned for inspection only.
    pub prompt_used: String,
    /// Token / retrieval usage statistics.
    pub usage: RagUsage,
}

/// Token and retrieval usage information.
#[derive(Debug, Serialize)]
pub struct RagUsage {
    /// Number of documents in the index at query time.
    pub documents_searched: usize,
    /// Number of chunks returned by the retriever.
    pub chunks_retrieved: usize,
    /// Approximate prompt token count (one token ≈ one whitespace-separated word).
    pub prompt_tokens: usize,
    /// Number of tokens generated by the model.
    pub completion_tokens: usize,
}

/// Response body for `GET /rag/stats`.
#[derive(Debug, Serialize)]
pub struct RagStatsResponse {
    /// Number of documents currently indexed.
    pub documents_indexed: usize,
    /// Number of chunks currently in the vector store.
    pub chunks_indexed: usize,
    /// Embedding vector dimensionality.
    pub embedding_dim: usize,
    /// Approximate heap bytes used by the vector store.
    pub store_memory_bytes: usize,
    /// Human-readable representation of `store_memory_bytes`.
    pub store_memory_human: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared error response helper
// ─────────────────────────────────────────────────────────────────────────────

/// Build a JSON error response.
///
/// Delegates to the shared [`crate::http_error`] envelope so `/rag/*` errors use
/// the same `{"error": {message, type, param, code}}` shape as the
/// OpenAI-compatible chat/embeddings routes mounted on the same router
/// (finding `serve-api-10`), instead of the previous flat
/// `{"error": "<string>"}`.
fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    crate::http_error::error_response(status, message, None)
}

/// The same envelope as [`error_response`], as the error type the deadline
/// and the blocking-pool seams share with the chat routes (it renders the
/// identical body).
fn api_error(status: StatusCode, message: impl Into<String>) -> ApiError {
    ApiError::new(status, message)
}

// ─────────────────────────────────────────────────────────────────────────────
// Human-readable byte formatting
// ─────────────────────────────────────────────────────────────────────────────

fn human_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * KB;
    const GB: usize = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GiB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MiB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KiB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple token count heuristic (whitespace-split)
// ─────────────────────────────────────────────────────────────────────────────

fn rough_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

// ─────────────────────────────────────────────────────────────────────────────
// Router options
// ─────────────────────────────────────────────────────────────────────────────

/// Everything [`create_rag_router_with_options`] takes besides the engine
/// pool and the tokenizer.
///
/// The default builds the router the other constructors build: no deadline,
/// and a router-local metrics instance nobody scrapes.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct RagRouterOptions {
    /// The server's per-request limits. Only
    /// [`RequestLimits::per_request_timeout`] applies to the RAG routes: it
    /// is the deadline `/rag/query` enforces inside its handler, the same
    /// value the chat routes enforce (`--request-timeout-ms`).
    pub limits: RequestLimits,
    /// The metrics a request that outlasts its deadline is counted in
    /// (`errors_total`). Pass the instance the chat router records into so
    /// `/metrics` shows RAG timeouts alongside theirs; `None` keeps a
    /// router-local instance.
    pub metrics: Option<Arc<InferenceMetrics>>,
}

impl RagRouterOptions {
    /// Set the per-request limits (see [`Self::limits`]).
    #[must_use]
    pub fn with_limits(mut self, limits: RequestLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Record timed-out requests into `metrics` (see [`Self::metrics`]).
    #[must_use]
    pub fn with_metrics(mut self, metrics: Arc<InferenceMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RagState
// ─────────────────────────────────────────────────────────────────────────────

/// Shared state for the RAG server.
///
/// Holds the RAG pipeline — an immutable snapshot behind a lock that is held
/// only to read or replace the pointer — and an [`EnginePool`] of
/// inference-engine replicas — the same pooling pattern the base
/// `/v1/chat/completions` server uses, so RAG requests can generate
/// concurrently up to `pool.size()` instead of serializing on a single mutex.
/// An optional [`TokenizerBridge`] enables real text generation from the
/// retrieved-context prompt.
pub struct RagState {
    /// The RAG pipeline, as an immutable snapshot. The `std::sync::Mutex`
    /// guards the pointer only: a reader clones it and works on its own
    /// snapshot, so the lock is never held across the (long) retrieval or
    /// indexing work and a poisoned lock cannot leave a half-written index.
    pipeline: Mutex<Arc<RagPipeline<TfIdfEmbedder>>>,
    /// Pool of inference-engine replicas shared across all RAG requests.
    engines: Arc<EnginePool>,
    /// Optional tokenizer. When present, the RAG prompt is encoded, generated,
    /// and decoded to real text; when absent, generation is skipped honestly
    /// (see [`rag_query`]).
    tokenizer: Option<TokenizerBridge>,
    /// The deadline `/rag/query` enforces inside its handler; `None` is no
    /// deadline of its own (only the layers outside the handler time it).
    request_timeout: Option<Duration>,
    /// Where a request that outlasts `request_timeout` is counted.
    metrics: Arc<InferenceMetrics>,
    /// Test seam: which thread each long-running stage ran on.
    #[cfg(test)]
    stage_log: Mutex<Vec<(&'static str, std::thread::ThreadId)>>,
}

impl RagState {
    /// Create a new [`RagState`] backed by the provided engine pool and an
    /// optional tokenizer.
    ///
    /// The RAG pipeline is initialised with a bootstrap corpus so the
    /// `TfIdfEmbedder` vocabulary is non-empty from the start. The state has
    /// no deadline of its own until [`Self::with_options`] gives it one.
    pub fn new(engines: Arc<EnginePool>, tokenizer: Option<TokenizerBridge>) -> Self {
        Self {
            pipeline: Mutex::new(Arc::new(bootstrap_pipeline())),
            engines,
            tokenizer,
            request_timeout: None,
            metrics: Arc::new(InferenceMetrics::new()),
            #[cfg(test)]
            stage_log: Mutex::new(Vec::new()),
        }
    }

    /// Apply `options`: the deadline `/rag/query` enforces, and the metrics a
    /// request that outlasts it is counted in.
    #[must_use]
    pub fn with_options(mut self, options: RagRouterOptions) -> Self {
        self.request_timeout = options.limits.per_request_timeout;
        if let Some(metrics) = options.metrics {
            self.metrics = metrics;
        }
        self
    }

    /// The current pipeline. The lock is held for the pointer clone only.
    fn snapshot(&self) -> Arc<RagPipeline<TfIdfEmbedder>> {
        // The guarded value is a plain `Arc` that is valid whatever panicked
        // while the lock was held, so a poisoned lock is recovered.
        let guard = self.pipeline.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::clone(&guard)
    }

    /// Replace the pipeline with `pipeline`. The previous one is released
    /// after the lock is, and goes away with its last in-flight reader.
    fn replace_pipeline(&self, pipeline: RagPipeline<TfIdfEmbedder>) {
        let previous = {
            let mut guard = self.pipeline.lock().unwrap_or_else(PoisonError::into_inner);
            std::mem::replace(&mut *guard, Arc::new(pipeline))
        };
        drop(previous);
    }

    /// Run `work` — one long synchronous stage of a request — on the blocking
    /// pool, so it never occupies an async worker thread.
    ///
    /// `cancel` is cancelled if the handler future is dropped before the
    /// result is in hand (the client went away, or a layer outside the
    /// handler gave up on it), so a stage that checks it can stop; `work`
    /// receives the same token. The blocking task itself cannot be
    /// interrupted from outside: it runs until it returns.
    ///
    /// # Errors
    ///
    /// `500` when the blocking task itself fails (panicked or was cancelled);
    /// `work`'s own result is passed through untouched as `T`.
    async fn run_stage<T, F>(
        self: &Arc<Self>,
        stage: &'static str,
        cancel: &CancellationToken,
        work: F,
    ) -> Result<T, ApiError>
    where
        F: FnOnce(&CancellationToken) -> T + Send + 'static,
        T: Send + 'static,
    {
        #[cfg(test)]
        let probe = Arc::clone(self);
        let token = cancel.clone();
        let abandon = CancelOnAbandon::new(cancel.clone());
        let joined = tokio::task::spawn_blocking(move || {
            #[cfg(test)]
            probe.note_stage(stage);
            work(&token)
        })
        .await;
        // The stage's result (or its failure) is in hand: nothing is abandoned.
        abandon.disarm();
        joined.map_err(|error| {
            tracing::error!(%error, stage, "RAG stage task failed");
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("RAG {stage} task failed"),
            )
        })
    }

    /// Record that the calling thread is running `stage` (tests read it back
    /// to show which stages ran off the async worker).
    #[cfg(test)]
    fn note_stage(&self, stage: &'static str) {
        self.stage_log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((stage, std::thread::current().id()));
    }
}

/// A fresh pipeline over the bootstrap corpus, so the TF-IDF vocabulary is
/// non-empty before any document is indexed.
fn bootstrap_pipeline() -> RagPipeline<TfIdfEmbedder> {
    let embedder = TfIdfEmbedder::fit(BOOTSTRAP_CORPUS, DEFAULT_MAX_FEATURES);
    RagPipeline::new(embedder, RagConfig::default())
}

/// The `error` of a stage that was told its request is over. Nobody reads
/// it: the handler future that would is already gone.
fn abandoned(stage: &str) -> ApiError {
    api_error(
        StatusCode::SERVICE_UNAVAILABLE,
        format!("the request was abandoned before {stage} began"),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler: POST /rag/index
// ─────────────────────────────────────────────────────────────────────────────

/// Index one or more documents into the RAG vector store.
///
/// If `chunk_size` / `chunk_overlap` are provided, the default `ChunkConfig`
/// is overridden.  After indexing the TF-IDF vocabulary is re-fitted against
/// the newly provided corpus so that future queries benefit from in-domain
/// term frequencies.
///
/// The fit and the per-document indexing loop run on the blocking pool; the
/// new index replaces the old one only once every document is indexed, and
/// never for a request whose client has gone away.
pub async fn index_documents(
    State(state): State<Arc<RagState>>,
    Json(req): Json<IndexDocumentRequest>,
) -> impl IntoResponse {
    if req.documents.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "documents list must not be empty");
    }

    // Build chunk config, honouring optional overrides.
    let mut chunk_config = ChunkConfig::default();
    if let Some(size) = req.chunk_size {
        chunk_config.chunk_size = size;
    }
    if let Some(overlap) = req.chunk_overlap {
        chunk_config.overlap = overlap;
    }

    // Reject chunk_size/chunk_overlap combinations that would blow up the
    // indexed corpus (finding `security-04`). `chunk_document`'s sliding
    // window steps by `chunk_size - overlap` characters; a near-equal
    // chunk_size/overlap pair (e.g. size=512, overlap=511) drives the step
    // toward 1, so an N-character document yields up to ~chunk_size × N
    // stored characters — durably retained in the in-process vector store.
    // Capping overlap at half of chunk_size bounds worst-case amplification
    // to ~2×, in addition to `ChunkConfig::validate()`'s existing (weaker)
    // `overlap < chunk_size` check performed later during indexing.
    if chunk_config.chunk_size == 0 {
        return error_response(StatusCode::BAD_REQUEST, "chunk_size must be > 0");
    }
    if chunk_config.overlap >= chunk_config.chunk_size {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "chunk_overlap ({}) must be < chunk_size ({})",
                chunk_config.overlap, chunk_config.chunk_size
            ),
        );
    }
    if chunk_config.overlap > chunk_config.chunk_size / 2 {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "chunk_overlap ({}) must be <= half of chunk_size ({}) to bound \
                 per-document storage amplification",
                chunk_config.overlap, chunk_config.chunk_size
            ),
        );
    }

    // The heavy half runs off the async worker. The guard inside `run_stage`
    // cancels the token if this future is dropped while it runs, and the
    // swap below only happens once the result is in hand.
    let documents = req.documents;
    let cancel = CancellationToken::new();
    let built = match state
        .run_stage("indexing", &cancel, move |cancel| {
            build_index(&documents, chunk_config, cancel)
        })
        .await
    {
        Ok(Ok(built)) => built,
        Ok(Err(IndexError::Document { index, source })) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("failed to index document {index}: {source}"),
            );
        }
        Ok(Err(IndexError::Cancelled)) => return abandoned("indexing").into_response(),
        Err(error) => return error.into_response(),
    };

    let BuiltIndex {
        pipeline,
        document_ids,
        total_chunks,
    } = built;
    let indexed = document_ids.len();

    // Swap the pipeline in.
    state.replace_pipeline(pipeline);

    let resp = IndexDocumentResponse {
        indexed,
        chunks: total_chunks,
        document_ids,
    };
    (StatusCode::OK, Json(resp)).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler: POST /rag/query
// ─────────────────────────────────────────────────────────────────────────────

/// RAG-augmented generation.
///
/// 1. Retrieves the top-k most relevant context chunks for `query`.
/// 2. Builds a prompt from the context and query.
/// 3. Runs inference on a replica of the engine pool.
/// 4. Returns the answer along with optional context and usage metadata.
///
/// Retrieval, prompt assembly and tokenisation run on the blocking pool, and
/// so does the generation (through the same seam as the chat routes'
/// non-streamed path), so no request ever occupies an async worker for its
/// duration. The request runs under the router's per-request deadline
/// ([`RagRouterOptions::with_limits`]) and its generation is cancelled when
/// the client goes away; see the module docs.
pub async fn rag_query(
    State(state): State<Arc<RagState>>,
    Json(req): Json<RagQueryRequest>,
) -> impl IntoResponse {
    // The request's handle on its generation and its stage record: the
    // deadline cancels whatever the handler starts and names the stage it
    // caught the request in.
    let slot = CancelSlot::default();
    let outcome = enforce_deadline(
        state.request_timeout,
        &state.metrics,
        &slot,
        query::rag_query_inner(Arc::clone(&state), req, slot.clone()),
    )
    .await;
    outcome.unwrap_or_else(IntoResponse::into_response)
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler: GET /rag/stats
// ─────────────────────────────────────────────────────────────────────────────

/// Return pipeline statistics as JSON.
///
/// Reads the current pipeline's snapshot, so it never waits for a query or
/// an index in flight.
pub async fn rag_stats(State(state): State<Arc<RagState>>) -> impl IntoResponse {
    let stats = state.snapshot().stats();

    let resp = RagStatsResponse {
        documents_indexed: stats.documents_indexed,
        chunks_indexed: stats.chunks_indexed,
        embedding_dim: stats.embedding_dim,
        store_memory_bytes: stats.store_memory_bytes,
        store_memory_human: human_bytes(stats.store_memory_bytes),
    };

    (StatusCode::OK, Json(resp)).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler: DELETE /rag/index
// ─────────────────────────────────────────────────────────────────────────────

/// Clear the vector index, resetting the pipeline to an empty state.
///
/// The TF-IDF embedder is re-fitted on the bootstrap corpus so the pipeline
/// remains usable after the clear.
pub async fn clear_index(State(state): State<Arc<RagState>>) -> impl IntoResponse {
    state.replace_pipeline(bootstrap_pipeline());

    let body = serde_json::json!({ "status": "cleared" });
    (StatusCode::OK, Json(body)).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Router factory
// ─────────────────────────────────────────────────────────────────────────────

/// Build and return the Axum router for all RAG endpoints.
///
/// The provided `engine` is wrapped in a single-replica [`EnginePool`] with no
/// tokenizer attached. Use [`create_rag_router_with_pool`] to serve from a
/// multi-replica pool and to attach a tokenizer for real text generation.
pub fn create_rag_router(engine: InferenceEngine<'static>) -> Router {
    create_rag_router_with_pool(EnginePool::new(vec![engine]), None)
}

/// Build the RAG router from a pre-built [`EnginePool`] and optional tokenizer.
///
/// This mirrors the base server's `create_router_with_pool`: requests generate
/// concurrently up to `pool.size()`. When a tokenizer is supplied, `/rag/query`
/// encodes the retrieved-context prompt, runs the engine, and decodes the
/// output to real text.
///
/// The router has no deadline of its own: only the layers outside it time a
/// request. Use [`create_rag_router_with_options`] to give `/rag/query` the
/// server's per-request deadline.
pub fn create_rag_router_with_pool(
    engines: Arc<EnginePool>,
    tokenizer: Option<TokenizerBridge>,
) -> Router {
    create_rag_router_with_options(engines, tokenizer, RagRouterOptions::default())
}

/// [`create_rag_router_with_pool`] with the server's per-request limits: when
/// `options` carries a request timeout ([`RagRouterOptions::with_limits`]),
/// `/rag/query` enforces it inside its handler exactly as the chat routes
/// do — an expired deadline cancels the generation and answers `504
/// request_timeout` with the stage it caught the request in.
pub fn create_rag_router_with_options(
    engines: Arc<EnginePool>,
    tokenizer: Option<TokenizerBridge>,
    options: RagRouterOptions,
) -> Router {
    let state = Arc::new(RagState::new(engines, tokenizer).with_options(options));

    Router::new()
        .route("/rag/index", axum::routing::post(index_documents))
        .route("/rag/index", axum::routing::delete(clear_index))
        .route("/rag/query", axum::routing::post(rag_query))
        .route("/rag/stats", axum::routing::get(rag_stats))
        .with_state(state)
}

#[cfg(test)]
#[path = "rag_server_tests.rs"]
mod tests;
