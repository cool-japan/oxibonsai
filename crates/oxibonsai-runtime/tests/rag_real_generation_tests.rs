//! Regression tests for the RAG `/rag/query` generation path (finding 7).
//!
//! The previous implementation ignored the retrieved context and query
//! entirely, generating from a hard-coded single start token and returning raw
//! numeric token IDs as the "answer". These tests pin the corrected behavior:
//!
//!   * With a tokenizer attached, the *real* context+query prompt is encoded,
//!     generated, and decoded to text (completion actually runs on the prompt).
//!   * Without a tokenizer, the endpoint is honest — it returns a transparent
//!     message and `completion_tokens = 0`, never numeric token IDs dressed up
//!     as a model answer.

#![cfg(feature = "rag")]

use std::path::Path;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_pool::EnginePool;
use oxibonsai_runtime::rag_server::{create_rag_router, create_rag_router_with_pool};
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;

/// Bundled Qwen3 tokenizer fixture (relative to the crate root). Tests needing
/// a real tokenizer skip themselves when it is absent.
const FIXTURE_TOKENIZER: &str = "../../models/tokenizer.json";

fn engine() -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42)
}

fn json_request(method: Method, path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("request")
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    serde_json::from_slice(&bytes).expect("parse json")
}

fn maybe_tokenizer() -> Option<TokenizerBridge> {
    if !Path::new(FIXTURE_TOKENIZER).exists() {
        eprintln!("skipped: tokenizer fixture missing at {FIXTURE_TOKENIZER}");
        return None;
    }
    TokenizerBridge::from_file(FIXTURE_TOKENIZER).ok()
}

// ── With a tokenizer: real generation on the real prompt ──────────────────────

#[tokio::test]
async fn rag_query_with_tokenizer_generates_from_context_and_query() {
    let Some(tokenizer) = maybe_tokenizer() else {
        return;
    };

    let router = create_rag_router_with_pool(EnginePool::new(vec![engine()]), Some(tokenizer));

    // Index a distinctive document.
    let index = json_request(
        Method::POST,
        "/rag/index",
        serde_json::json!({
            "documents": ["Photosynthesis converts sunlight into chemical energy in plants."]
        }),
    );
    let resp = router.clone().oneshot(index).await.expect("index response");
    assert_eq!(resp.status(), StatusCode::OK);

    // Query it, asking for the retrieved context back.
    let query = json_request(
        Method::POST,
        "/rag/query",
        serde_json::json!({
            "query": "How do plants make energy?",
            "max_tokens": 4,
            "include_context": true,
        }),
    );
    let resp = router.clone().oneshot(query).await.expect("query response");
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp).await;

    // The prompt actually built from the retrieved context + the query.
    let prompt = json["prompt_used"].as_str().expect("prompt_used string");
    assert!(
        prompt.contains("How do plants make energy?"),
        "prompt must be built from the real query; got {prompt:?}"
    );
    assert!(
        prompt.contains("Photosynthesis") || prompt.contains("sunlight"),
        "prompt must include the retrieved context; got {prompt:?}"
    );

    // Real generation ran on the encoded prompt: some tokens were produced and
    // decoded to a text answer (not numeric token IDs).
    let completion_tokens = json["usage"]["completion_tokens"]
        .as_u64()
        .expect("completion_tokens number");
    assert!(
        completion_tokens > 0,
        "generation should have produced at least one token"
    );
    assert!(json["answer"].is_string(), "answer must be decoded text");

    // The retrieved context is surfaced back to the caller.
    let chunks = json["retrieved_chunks"].as_array().expect("chunks array");
    assert!(
        chunks
            .iter()
            .any(|c| c.as_str().unwrap_or_default().contains("Photosynthesis")),
        "retrieved_chunks should contain the indexed document; got {json}"
    );
}

// ── Without a tokenizer: honest, non-fabricated response ───────────────────────

#[tokio::test]
async fn rag_query_without_tokenizer_is_honest_not_fabricated() {
    let router = create_rag_router(engine());

    let query = json_request(
        Method::POST,
        "/rag/query",
        serde_json::json!({ "query": "What is Rust?", "max_tokens": 5 }),
    );
    let resp = router.oneshot(query).await.expect("query response");
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp).await;

    // No numeric-token-ID fabrication: no generation happened, and the answer
    // says so transparently.
    assert_eq!(
        json["usage"]["completion_tokens"], 0,
        "no tokenizer → no generated tokens; got {json}"
    );
    let answer = json["answer"].as_str().expect("answer string");
    assert!(
        answer.contains("tokenizer"),
        "answer should transparently state the tokenizer limitation; got {answer:?}"
    );
    // The prompt was still genuinely built from the query.
    assert!(
        json["prompt_used"]
            .as_str()
            .unwrap_or_default()
            .contains("What is Rust?"),
        "prompt_used should reflect the real query; got {json}"
    );
}
