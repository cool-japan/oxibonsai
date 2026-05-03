//! OpenAI-compatible chat completions server.
//!
//! Provides an Axum-based HTTP server with the following endpoints:
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | POST | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
//! | GET | `/v1/models` | List available models |
//! | GET | `/health` | Liveness probe |
//! | GET | `/metrics` | Prometheus text exposition |
//!
//! Use [`create_router`] or [`create_router_with_metrics`] to build
//! the Axum router, then serve it with `axum::serve`.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{
    sse::{Event, Sse},
    IntoResponse, Json, Response,
};
use axum::Router;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

use crate::engine::InferenceEngine;
use crate::metrics::InferenceMetrics;
use crate::tokenizer_bridge::TokenizerBridge;

/// Server state.
pub struct AppState {
    engine: Mutex<InferenceEngine<'static>>,
    tokenizer: Option<TokenizerBridge>,
    metrics: Arc<InferenceMetrics>,
}

impl AppState {
    /// Acquire a mutable guard over the inference engine.
    pub async fn engine_lock(&self) -> tokio::sync::MutexGuard<'_, InferenceEngine<'static>> {
        self.engine.lock().await
    }

    /// Access the optional tokenizer.
    pub fn tokenizer(&self) -> Option<&TokenizerBridge> {
        self.tokenizer.as_ref()
    }

    /// Access the shared metrics instance.
    pub fn metrics(&self) -> &Arc<InferenceMetrics> {
        &self.metrics
    }
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat completion request.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    0.7
}

/// Chat completion response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// Token usage info.
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// A choice in the completion response.
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// SSE streaming chunk (OpenAI-compatible).
#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

/// A choice in the SSE streaming chunk.
#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: ChunkDelta,
    finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Serialize)]
struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

/// Create the Axum router.
pub fn create_router(
    engine: InferenceEngine<'static>,
    tokenizer: Option<TokenizerBridge>,
) -> Router {
    create_router_with_metrics(engine, tokenizer, Arc::new(InferenceMetrics::new()))
}

/// Create the Axum router with a shared metrics instance.
pub fn create_router_with_metrics(
    engine: InferenceEngine<'static>,
    tokenizer: Option<TokenizerBridge>,
    metrics: Arc<InferenceMetrics>,
) -> Router {
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer,
        metrics,
    });

    // The embeddings router carries its own Arc<EmbeddingAppState>; merge it
    // before attaching the main AppState so the states don't conflict.
    let embeddings_router = crate::embeddings::create_embeddings_router(512);

    Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat_completions),
        )
        .route(
            "/v1/chat/completions/extended",
            axum::routing::post(crate::api_extensions::extended_chat_completions),
        )
        .route(
            "/v1/completions",
            axum::routing::post(crate::completions::create_completion),
        )
        .route("/v1/models", axum::routing::get(list_models))
        .route("/health", axum::routing::get(health))
        .route("/metrics", axum::routing::get(prometheus_metrics))
        .with_state(state)
        .merge(embeddings_router)
}

async fn health() -> &'static str {
    "ok"
}

/// Prometheus metrics endpoint.
async fn prometheus_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.metrics.render_prometheus();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

async fn list_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": "bonsai-8b",
            "object": "model",
            "owned_by": "oxibonsai"
        }]
    }))
}

#[tracing::instrument(skip(state))]
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    let request_start = std::time::Instant::now();
    state.metrics.requests_total.inc();
    state.metrics.active_requests.inc();

    // Build prompt from messages
    let prompt_text = build_prompt(&body.messages);

    // Tokenize
    let prompt_tokens = if let Some(tok) = &state.tokenizer {
        tok.encode(&prompt_text).map_err(|_| {
            state.metrics.errors_total.inc();
            state.metrics.active_requests.dec();
            StatusCode::INTERNAL_SERVER_ERROR
        })?
    } else {
        // Fallback: single start token
        vec![151644]
    };

    state
        .metrics
        .prompt_tokens_total
        .inc_by(prompt_tokens.len() as u64);

    let result = if body.stream {
        // ── SSE streaming mode ──
        chat_completions_stream(Arc::clone(&state), prompt_tokens, body.max_tokens).await
    } else {
        // ── Non-streaming mode ──
        chat_completions_non_stream(Arc::clone(&state), prompt_tokens, body.max_tokens).await
    };

    let elapsed = request_start.elapsed().as_secs_f64();
    state.metrics.request_duration_seconds.observe(elapsed);
    state.metrics.active_requests.dec();

    if result.is_err() {
        state.metrics.errors_total.inc();
    }

    result
}

/// Non-streaming chat completion handler.
async fn chat_completions_non_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
) -> Result<Response, StatusCode> {
    let prompt_len = prompt_tokens.len();

    let mut engine = state.engine.lock().await;
    let output_tokens = engine.generate(&prompt_tokens, max_tokens).map_err(|e| {
        tracing::error!(error = %e, "generation failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let completion_len = output_tokens.len();

    // Record token metrics
    state
        .metrics
        .tokens_generated_total
        .inc_by(completion_len as u64);

    // Decode
    let content = if let Some(tok) = &state.tokenizer {
        tok.decode(&output_tokens)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    } else {
        format!("{output_tokens:?}")
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", rand_id()),
        object: "chat.completion".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
            total_tokens: prompt_len + completion_len,
        },
    };

    Ok(Json(response).into_response())
}

/// SSE streaming chat completion handler.
async fn chat_completions_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
) -> Result<Response, StatusCode> {
    let completion_id = format!("chatcmpl-{}", rand_id());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();

    // Spawn generation task that locks the engine and streams tokens
    let gen_state = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let mut engine = rt.block_on(gen_state.engine.lock());
        let _result = engine.generate_streaming(&prompt_tokens, max_tokens, &token_tx);
        // token_tx is dropped here, closing the channel
    });

    // Build SSE stream from the token receiver
    let id_for_stream = completion_id;
    let state_for_stream = Arc::clone(&state);

    // First, send a role delta
    let role_chunk = ChatCompletionChunk {
        id: id_for_stream.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: "bonsai-8b".to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };

    let role_event = match serde_json::to_string(&role_chunk) {
        Ok(json) => json,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    let id_clone = id_for_stream.clone();

    // Convert token receiver into a stream of SSE events
    let token_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(token_rx);

    // Per-request streaming-decode state.  BPE tokens may straddle UTF-8
    // codepoint boundaries (CJK, emoji), so we buffer through HF's
    // step_decode_stream and only emit a chunk when a complete UTF-8 piece is
    // ready.  Mid-codepoint tokens yield `Ok(None)` and are filtered out.
    let mut stream_state = state_for_stream
        .tokenizer
        .as_ref()
        .map(|t| t.new_decode_stream(true));

    let content_stream = token_stream.filter_map(move |token_id| {
        let text = match (&state_for_stream.tokenizer, stream_state.as_mut()) {
            (Some(tok), Some(state)) => match tok.step_decode(state, token_id) {
                Ok(Some(txt)) => txt,
                Ok(None) => return None,
                Err(_) => format!("[{token_id}]"),
            },
            _ => format!("[{token_id}]"),
        };

        let chunk = ChatCompletionChunk {
            id: id_clone.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: "bonsai-8b".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(text),
                },
                finish_reason: None,
            }],
        };

        Some(serde_json::to_string(&chunk).unwrap_or_default())
    });

    // Build finish chunk
    let finish_chunk = ChatCompletionChunk {
        id: id_for_stream,
        object: "chat.completion.chunk".to_string(),
        created,
        model: "bonsai-8b".to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
    };
    let finish_json = serde_json::to_string(&finish_chunk).unwrap_or_default();

    // Prepend role event, append finish event and [DONE]
    let role_stream = tokio_stream::once(role_event);

    let full_stream = role_stream
        .chain(content_stream)
        .chain(tokio_stream::once(finish_json))
        .map(|json_str| -> Result<Event, Infallible> { Ok(Event::default().data(json_str)) })
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));

    Ok(Sse::new(full_stream).into_response())
}

/// Build a simple prompt from chat messages.
fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|im_end|>\n");
            }
            "user" => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|im_end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|im_end|>\n");
            }
            _ => {
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
        }
    }
    // Signal model to respond as assistant
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Generate a short random-ish ID for completion responses.
fn rand_id() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{ts:x}")
}

// ─── Graceful shutdown ─────────────────────────────────────────────────

/// Start server with graceful shutdown support.
///
/// Binds to `addr`, serves `router`, and shuts down cleanly when
/// `shutdown_signal` completes. In-flight requests are given time
/// to finish before the server exits.
pub async fn serve_with_shutdown(
    router: Router,
    addr: std::net::SocketAddr,
    shutdown_signal: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "server listening");

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    tracing::info!("server shut down gracefully");
    Ok(())
}

/// Create a shutdown signal that responds to SIGTERM and SIGINT (Ctrl+C).
///
/// Completes when either signal is received, allowing the server to
/// begin its graceful shutdown procedure.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            tracing::info!("received Ctrl+C, initiating shutdown");
        }
        () = terminate => {
            tracing::info!("received SIGTERM, initiating shutdown");
        }
    }
}

/// Create the full server setup: router + graceful shutdown future.
///
/// Returns a future that runs the server until a shutdown signal is received.
pub async fn create_server(
    engine: InferenceEngine<'static>,
    tokenizer: Option<TokenizerBridge>,
    addr: std::net::SocketAddr,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let metrics = Arc::new(InferenceMetrics::new());
    let router = create_router_with_metrics(engine, tokenizer, metrics);
    serve_with_shutdown(router, addr, shutdown_signal()).await
}

// ─── Request queue depth tracking ──────────────────────────────────────

/// Server configuration with request management.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Maximum number of queued requests before rejecting new ones.
    pub max_queue_depth: usize,
    /// Request timeout in seconds.
    pub request_timeout_seconds: u64,
    /// Address to bind to.
    pub bind_addr: std::net::SocketAddr,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_queue_depth: 128,
            request_timeout_seconds: 60,
            bind_addr: std::net::SocketAddr::from(([127, 0, 0, 1], 8080)),
        }
    }
}

/// Request queue depth tracker.
///
/// Thread-safe counter for tracking how many requests are currently
/// queued or in-flight. Used to implement backpressure.
pub struct QueueDepthTracker {
    current: std::sync::atomic::AtomicUsize,
    max_depth: usize,
}

impl QueueDepthTracker {
    /// Create a new tracker with the given maximum depth.
    pub fn new(max_depth: usize) -> Self {
        Self {
            current: std::sync::atomic::AtomicUsize::new(0),
            max_depth: max_depth.max(1),
        }
    }

    /// Try to acquire a slot. Returns `true` if successful, `false` if queue is full.
    pub fn try_acquire(&self) -> bool {
        let current = self.current.load(std::sync::atomic::Ordering::Relaxed);
        if current >= self.max_depth {
            return false;
        }
        // CAS loop for correctness under contention
        self.current
            .compare_exchange(
                current,
                current + 1,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Release a slot.
    pub fn release(&self) {
        self.current
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
    }

    /// Current queue depth.
    pub fn depth(&self) -> usize {
        self.current.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Maximum allowed depth.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Whether the queue has capacity for more requests.
    pub fn has_capacity(&self) -> bool {
        self.depth() < self.max_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_prompt_simple() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let p = build_prompt(&msgs);
        assert!(p.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn build_prompt_system_and_user() {
        let msgs = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            },
        ];
        let p = build_prompt(&msgs);
        assert!(p.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
        assert!(p.contains("<|im_start|>user\nHi<|im_end|>"));
    }

    #[test]
    fn build_prompt_multi_turn() {
        let msgs = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "4".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "And 3+3?".to_string(),
            },
        ];
        let p = build_prompt(&msgs);
        assert!(p.contains("<|im_start|>assistant\n4<|im_end|>"));
        assert!(p.contains("And 3+3?"));
    }

    #[test]
    fn rand_id_is_nonempty() {
        let id = rand_id();
        assert!(!id.is_empty());
    }

    #[test]
    fn default_max_tokens_value() {
        assert_eq!(default_max_tokens(), 256);
    }

    #[test]
    fn default_temperature_value() {
        assert!((default_temperature() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn create_router_builds_without_tokenizer() {
        let config = oxibonsai_core::config::Qwen3Config::bonsai_8b();
        let params = crate::sampling::SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let _router = create_router(engine, None);
    }

    #[test]
    fn create_router_with_shared_metrics() {
        let config = oxibonsai_core::config::Qwen3Config::bonsai_8b();
        let params = crate::sampling::SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let metrics = Arc::new(InferenceMetrics::new());
        let _router = create_router_with_metrics(engine, None, Arc::clone(&metrics));
        // Metrics should be accessible from outside
        assert_eq!(metrics.requests_total.get(), 0);
    }

    // ── ServerConfig tests ──

    #[test]
    fn server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.max_queue_depth, 128);
        assert_eq!(config.request_timeout_seconds, 60);
        assert_eq!(
            config.bind_addr,
            std::net::SocketAddr::from(([127, 0, 0, 1], 8080))
        );
    }

    // ── QueueDepthTracker tests ──

    #[test]
    fn queue_depth_tracker_basic() {
        let tracker = QueueDepthTracker::new(3);
        assert_eq!(tracker.depth(), 0);
        assert_eq!(tracker.max_depth(), 3);
        assert!(tracker.has_capacity());

        assert!(tracker.try_acquire());
        assert_eq!(tracker.depth(), 1);
        assert!(tracker.try_acquire());
        assert_eq!(tracker.depth(), 2);
        assert!(tracker.try_acquire());
        assert_eq!(tracker.depth(), 3);
        assert!(!tracker.has_capacity());

        // Should fail when full
        assert!(!tracker.try_acquire());

        tracker.release();
        assert_eq!(tracker.depth(), 2);
        assert!(tracker.has_capacity());
        assert!(tracker.try_acquire());
    }

    #[test]
    fn queue_depth_tracker_min_capacity() {
        let tracker = QueueDepthTracker::new(0);
        assert_eq!(tracker.max_depth(), 1);
        assert!(tracker.try_acquire());
        assert!(!tracker.try_acquire());
    }
}
