//! OpenAI-compatible RAG endpoints for OxiBonsai.
//!
//! Feature-gated with `#[cfg(feature = "rag")]`.
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

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::Router;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use oxibonsai_rag::embedding::{Embedder, TfIdfEmbedder};
use oxibonsai_rag::pipeline::{RagConfig, RagPipeline};

use crate::engine::InferenceEngine;
use crate::engine_pool::EnginePool;
use crate::sampling::SamplingParams;
use crate::tokenizer_bridge::TokenizerBridge;

/// Hard upper bound on the number of tokens a single `/rag/query` request may
/// ask the engine to generate, mirroring the chat server's output ceiling so an
/// oversized `max_tokens` cannot drive an unbounded allocation.
const MAX_RAG_OUTPUT_TOKENS: usize = 8192;

/// Inclusive bounds on the client-supplied `top_k` for `/rag/query`. Values
/// outside this range are rejected with `400 Bad Request` rather than
/// silently clamped, so callers get honest feedback instead of a
/// mismatch between the requested and actual retrieval depth (finding
/// `serve-api-06`/`rag-eval-01`).
const MIN_RAG_TOP_K: usize = 1;
const MAX_RAG_TOP_K: usize = 50;

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
// RagState
// ─────────────────────────────────────────────────────────────────────────────

/// Shared state for the RAG server.
///
/// Holds the RAG pipeline (protected by a `std::sync::Mutex` for blocking
/// operations) and an [`EnginePool`] of inference-engine replicas — the same
/// pooling pattern the base `/v1/chat/completions` server uses, so RAG requests
/// can generate concurrently up to `pool.size()` instead of serializing on a
/// single mutex. An optional [`TokenizerBridge`] enables real text generation
/// from the retrieved-context prompt.
pub struct RagState {
    /// The RAG pipeline.  Uses a `std::sync::Mutex` because all RAG operations
    /// are synchronous (no `.await` points inside the lock).
    pipeline: Mutex<RagPipeline<TfIdfEmbedder>>,
    /// Pool of inference-engine replicas shared across all RAG requests.
    engines: Arc<EnginePool>,
    /// Optional tokenizer. When present, the RAG prompt is encoded, generated,
    /// and decoded to real text; when absent, generation is skipped honestly
    /// (see [`rag_query`]).
    tokenizer: Option<TokenizerBridge>,
}

impl RagState {
    /// Create a new [`RagState`] backed by the provided engine pool and an
    /// optional tokenizer.
    ///
    /// The RAG pipeline is initialised with a bootstrap corpus so the
    /// `TfIdfEmbedder` vocabulary is non-empty from the start.
    pub fn new(engines: Arc<EnginePool>, tokenizer: Option<TokenizerBridge>) -> Self {
        let embedder = TfIdfEmbedder::fit(BOOTSTRAP_CORPUS, DEFAULT_MAX_FEATURES);
        let pipeline = RagPipeline::new(embedder, RagConfig::default());
        Self {
            pipeline: Mutex::new(pipeline),
            engines,
            tokenizer,
        }
    }
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
pub async fn index_documents(
    State(state): State<Arc<RagState>>,
    Json(req): Json<IndexDocumentRequest>,
) -> impl IntoResponse {
    if req.documents.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "documents list must not be empty");
    }

    // Build chunk config, honouring optional overrides.
    let mut chunk_config = oxibonsai_rag::chunker::ChunkConfig::default();
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

    // Build a fresh TF-IDF embedder fitted on the new corpus so that
    // vocabulary is always in-domain.
    let doc_refs: Vec<&str> = req.documents.iter().map(String::as_str).collect();
    let embedder = TfIdfEmbedder::fit(&doc_refs, DEFAULT_MAX_FEATURES);

    let rag_config = RagConfig::default().with_chunk_config(chunk_config);

    // Replace the pipeline with a freshly fitted one.
    let mut new_pipeline = RagPipeline::new(embedder, rag_config);

    let mut document_ids: Vec<usize> = Vec::with_capacity(req.documents.len());
    let mut total_chunks = 0usize;

    for (doc_idx, doc) in req.documents.iter().enumerate() {
        match new_pipeline.index_document(doc) {
            Ok(chunk_count) => {
                document_ids.push(doc_idx);
                total_chunks += chunk_count;
            }
            Err(e) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    format!("failed to index document {doc_idx}: {e}"),
                );
            }
        }
    }

    let indexed = document_ids.len();

    // Swap the pipeline in under the mutex.
    match state.pipeline.lock() {
        Ok(mut guard) => {
            *guard = new_pipeline;
        }
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("pipeline lock poisoned: {e}"),
            );
        }
    }

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
/// 3. Runs inference via the shared `InferenceEngine`.
/// 4. Returns the answer along with optional context and usage metadata.
pub async fn rag_query(
    State(state): State<Arc<RagState>>,
    Json(req): Json<RagQueryRequest>,
) -> impl IntoResponse {
    if req.query.trim().is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "query must not be empty");
    }

    let max_tokens = req
        .max_tokens
        .unwrap_or(256)
        .clamp(1, MAX_RAG_OUTPUT_TOKENS);
    let top_k = match req.top_k {
        None => 3,
        Some(k) if (MIN_RAG_TOP_K..=MAX_RAG_TOP_K).contains(&k) => k,
        Some(k) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "top_k ({k}) must be between {MIN_RAG_TOP_K} and {MAX_RAG_TOP_K} inclusive"
                ),
            );
        }
    };
    let include_context = req.include_context.unwrap_or(false);

    // ── 1. Build prompt via RAG pipeline ────────────────────────────────────
    //
    // The pipeline's own `Retriever` is always constructed with a *fixed*
    // `RetrieverConfig` (top_k baked in at pipeline-construction time), so
    // `RagPipeline::build_prompt` cannot be asked to use a different top_k
    // per request. To make the client-supplied `top_k` genuinely drive both
    // the reported chunk count *and* the generation prompt (finding
    // `serve-api-06`/`rag-eval-01`), we perform retrieval ourselves against
    // the pipeline's embedder + vector store directly, then assemble the
    // context/prompt using the exact same defaults `RagState` constructs its
    // pipelines with (`RagConfig::default()`), and finally feed that same
    // context into the generation step below.
    let (prompt, retrieved_chunks, docs_searched, chunks_retrieved) = {
        let pipeline_guard = match state.pipeline.lock() {
            Ok(g) => g,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("pipeline lock poisoned: {e}"),
                );
            }
        };

        let stats = pipeline_guard.stats();
        let docs_searched = stats.documents_indexed;

        let retriever = pipeline_guard.retriever();
        let default_retriever_config = oxibonsai_rag::retriever::RetrieverConfig::default();

        let query_vec = match retriever.embedder().embed(&req.query) {
            Ok(v) => v,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("query embedding failed: {e}"),
                );
            }
        };

        // `search_with_threshold` returns an empty Vec (never errors) when
        // the store is empty, so no separate "no documents yet" branch is
        // needed here.
        let results = retriever.store().search_with_threshold(
            &query_vec,
            top_k,
            default_retriever_config.min_score,
        );

        let retrieved_texts: Vec<String> = results.iter().map(|r| r.chunk.text.clone()).collect();
        let chunks_retrieved = retrieved_texts.len();

        // Assemble the context block the same way
        // `RagPipeline::retrieve_context` does: concatenate chunk texts with
        // the configured separator, dropping chunks that would exceed
        // `max_context_chars`.
        let rag_defaults = RagConfig::default();
        let sep = &rag_defaults.context_separator;
        let mut parts: Vec<&str> = Vec::with_capacity(results.len());
        let mut total_chars = 0usize;
        for result in &results {
            let text_len = result.chunk.text.len();
            let sep_len = if parts.is_empty() { 0 } else { sep.len() };
            if total_chars + sep_len + text_len > rag_defaults.max_context_chars
                && !parts.is_empty()
            {
                break;
            }
            total_chars += sep_len + text_len;
            parts.push(&result.chunk.text);
        }
        let context = parts.join(sep.as_str());

        let prompt = rag_defaults
            .prompt_template
            .replace("{context}", &context)
            .replace("{query}", &req.query);

        (prompt, retrieved_texts, docs_searched, chunks_retrieved)
    };

    // ── 2. Generate an answer from the real RAG prompt ───────────────────────
    // When a tokenizer is configured we encode the *actual* context+query
    // prompt, run the engine on those tokens, and decode the output back to
    // text — so the answer genuinely depends on the retrieved context and the
    // query. Without a tokenizer we cannot map the prompt text onto the model's
    // vocabulary, so we skip generation and say so honestly rather than
    // fabricating an answer from a fixed start token.
    let (answer, completion_tokens, prompt_tokens_count) = match &state.tokenizer {
        Some(tokenizer) => {
            let input_tokens = match tokenizer.encode(&prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "prompt encoded to an empty token sequence",
                    );
                }
                Err(e) => {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("prompt tokenisation failed: {e}"),
                    );
                }
            };
            let prompt_tokens_count = input_tokens.len();

            // Honor the request temperature when it is a valid value.
            let mut params = SamplingParams::default();
            if let Some(temperature) = req.temperature {
                if temperature.is_finite() && (0.0..=2.0).contains(&temperature) {
                    params.temperature = temperature;
                }
            }

            let output_tokens = {
                let mut lease = match state.engines.acquire().await {
                    Ok(lease) => lease,
                    Err(e) => {
                        return error_response(
                            StatusCode::SERVICE_UNAVAILABLE,
                            format!("engine pool acquire failed: {e}"),
                        );
                    }
                };
                match lease.generate_with_params(&input_tokens, max_tokens, &params) {
                    Ok(tokens) => tokens,
                    Err(e) => {
                        return error_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("generation failed: {e}"),
                        );
                    }
                }
            };

            let completion_tokens = output_tokens.len();
            let answer = match tokenizer.decode(&output_tokens) {
                Ok(text) => text,
                Err(e) => {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("decoding generated tokens failed: {e}"),
                    );
                }
            };
            (answer, completion_tokens, prompt_tokens_count)
        }
        None => {
            // No tokenizer: retrieval succeeded and the prompt was built, but we
            // cannot run the model on raw text. Be transparent instead of
            // returning numeric token IDs dressed up as an answer.
            let answer = "[no tokenizer configured: retrieval succeeded and the \
                          prompt was built, but text generation is unavailable on \
                          this server]"
                .to_string();
            (answer, 0usize, rough_token_count(&prompt))
        }
    };

    // ── 4. Build response ────────────────────────────────────────────────────
    let resp = RagQueryResponse {
        answer,
        retrieved_chunks: if include_context {
            Some(retrieved_chunks)
        } else {
            None
        },
        prompt_used: prompt,
        usage: RagUsage {
            documents_searched: docs_searched,
            chunks_retrieved,
            prompt_tokens: prompt_tokens_count,
            completion_tokens,
        },
    };

    (StatusCode::OK, Json(resp)).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler: GET /rag/stats
// ─────────────────────────────────────────────────────────────────────────────

/// Return pipeline statistics as JSON.
pub async fn rag_stats(State(state): State<Arc<RagState>>) -> impl IntoResponse {
    let stats = match state.pipeline.lock() {
        Ok(guard) => guard.stats(),
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("pipeline lock poisoned: {e}"),
            )
            .into_response();
        }
    };

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
    let embedder = TfIdfEmbedder::fit(BOOTSTRAP_CORPUS, DEFAULT_MAX_FEATURES);
    let fresh_pipeline = RagPipeline::new(embedder, RagConfig::default());

    match state.pipeline.lock() {
        Ok(mut guard) => {
            *guard = fresh_pipeline;
        }
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("pipeline lock poisoned: {e}"),
            );
        }
    }

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
pub fn create_rag_router_with_pool(
    engines: Arc<EnginePool>,
    tokenizer: Option<TokenizerBridge>,
) -> Router {
    let state = Arc::new(RagState::new(engines, tokenizer));

    Router::new()
        .route("/rag/index", axum::routing::post(index_documents))
        .route("/rag/index", axum::routing::delete(clear_index))
        .route("/rag/query", axum::routing::post(rag_query))
        .route("/rag/stats", axum::routing::get(rag_stats))
        .with_state(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
}
