//! Integration tests for the `/v1/embeddings` endpoint.
//!
//! Exercises both the HTTP API layer (via `axum::Router::oneshot`) and the
//! underlying `EmbedderRegistry` / `EmbeddingInput` types directly.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use oxibonsai_runtime::embeddings::{create_embeddings_router, EmbedderRegistry, EmbeddingInput};

// ── Helper ────────────────────────────────────────────────────────────────────

/// POST `body` to `/v1/embeddings` and return (status, parsed JSON).
async fn post_embeddings(
    app: axum::Router,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let req = Request::post("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&body).expect("body serialisation"),
        ))
        .expect("request build");

    let resp = app.oneshot(req).await.expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

// ── HTTP endpoint tests ────────────────────────────────────────────────────────

/// A valid single-text request must return HTTP 200.
#[tokio::test]
async fn test_embeddings_returns_200() {
    let app = create_embeddings_router(32);
    let body = serde_json::json!({ "input": "hello world" });
    let (status, _json) = post_embeddings(app, body).await;
    assert_eq!(status, StatusCode::OK, "expected 200 OK");
}

/// The response must conform to the OpenAI embeddings schema:
/// `object == "list"`, `data` is an array, `model` is a string, `usage` is present.
#[tokio::test]
async fn test_embeddings_response_shape() {
    let app = create_embeddings_router(32);
    let body = serde_json::json!({ "input": "test input" });
    let (status, json) = post_embeddings(app, body).await;
    assert_eq!(status, StatusCode::OK);

    assert_eq!(
        json["object"].as_str().expect("object field"),
        "list",
        "object must be 'list'"
    );
    assert!(json["data"].is_array(), "data must be an array");
    assert!(json["model"].as_str().is_some(), "model must be a string");
    assert!(json["usage"].is_object(), "usage must be an object");
}

/// When `dimensions` is specified, each embedding vector must be truncated to
/// at most that many dimensions.  A single-text input uses the IdentityEmbedder
/// (TF-IDF is only fitted for batches of ≥ 2), whose dimension equals the
/// configured `default_dim`.  Requesting truncation below `default_dim` must
/// shorten the vector accordingly.
#[tokio::test]
async fn test_embeddings_dimension_matches() {
    let requested_dim: usize = 8;
    // default_dim = 64; single-text → IdentityEmbedder (dim 64) → truncate to 8.
    let app = create_embeddings_router(64);
    let body = serde_json::json!({
        "input": "dimension truncation test",
        "dimensions": requested_dim
    });
    let (status, json) = post_embeddings(app, body).await;
    assert_eq!(status, StatusCode::OK);

    let data = json["data"].as_array().expect("data array");
    assert!(!data.is_empty(), "data must not be empty");

    let embedding = data[0]["embedding"]
        .as_array()
        .expect("embedding must be a float array");
    assert_eq!(
        embedding.len(),
        requested_dim,
        "embedding length must equal requested dimensions"
    );
}

/// A batch of two texts must produce two `data` entries, each with `object == "embedding"`.
#[tokio::test]
async fn test_embeddings_batch_input_produces_multiple_data() {
    let app = create_embeddings_router(32);
    let body = serde_json::json!({ "input": ["first text", "second text"] });
    let (status, json) = post_embeddings(app, body).await;
    assert_eq!(status, StatusCode::OK);

    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2, "expected two embeddings for batch of two");

    for (i, item) in data.iter().enumerate() {
        assert_eq!(
            item["object"].as_str().expect("object field"),
            "embedding",
            "data[{i}].object must be 'embedding'"
        );
        assert_eq!(
            item["index"].as_u64().expect("index field") as usize,
            i,
            "data[{i}].index must be {i}"
        );
    }
}

/// The `usage.prompt_tokens` and `usage.total_tokens` fields must be positive.
#[tokio::test]
async fn test_embeddings_usage_tokens_positive() {
    let app = create_embeddings_router(32);
    let body = serde_json::json!({ "input": "usage token count test" });
    let (status, json) = post_embeddings(app, body).await;
    assert_eq!(status, StatusCode::OK);

    let usage = &json["usage"];
    let prompt = usage["prompt_tokens"].as_u64().expect("prompt_tokens");
    let total = usage["total_tokens"].as_u64().expect("total_tokens");
    assert!(prompt > 0, "prompt_tokens must be > 0");
    assert_eq!(
        prompt, total,
        "total_tokens must equal prompt_tokens for embeddings"
    );
}

// ── EmbeddingInput unit tests ─────────────────────────────────────────────────

/// `EmbeddingInput::Single` produces one string.
#[test]
fn test_embedding_input_single() {
    let input = EmbeddingInput::Single("hello".to_string());
    assert_eq!(input.len(), 1);
    assert!(!input.is_empty());
    let strings = input.as_strings();
    assert_eq!(strings, vec!["hello"]);
}

/// `EmbeddingInput::Batch` preserves order and count.
#[test]
fn test_embedding_input_batch() {
    let input = EmbeddingInput::Batch(vec![
        "alpha".to_string(),
        "beta".to_string(),
        "gamma".to_string(),
    ]);
    assert_eq!(input.len(), 3);
    let strings = input.as_strings();
    assert_eq!(strings[0], "alpha");
    assert_eq!(strings[1], "beta");
    assert_eq!(strings[2], "gamma");
}

/// `EmbeddingInput::TokenIds` converts to a space-separated decimal string.
#[test]
fn test_embedding_input_token_ids() {
    let input = EmbeddingInput::TokenIds(vec![10u32, 20, 30]);
    assert_eq!(input.len(), 1);
    let strings = input.as_strings();
    assert_eq!(strings[0], "10 20 30");
}

/// `EmbeddingInput::BatchTokenIds` converts each sub-vec independently.
#[test]
fn test_embedding_input_batch_token_ids() {
    let input = EmbeddingInput::BatchTokenIds(vec![vec![1u32, 2], vec![99u32]]);
    assert_eq!(input.len(), 2);
    let strings = input.as_strings();
    assert_eq!(strings[0], "1 2");
    assert_eq!(strings[1], "99");
}

// ── EmbedderRegistry unit tests ───────────────────────────────────────────────

/// The registry must embed texts without panicking and return non-empty vectors.
#[test]
fn test_embedder_registry_basic() {
    let registry = EmbedderRegistry::new(16);
    let texts = vec!["the quick brown fox".to_string()];
    let embeddings = registry.embed_texts(&texts);
    assert_eq!(embeddings.len(), 1);
    assert_eq!(
        embeddings[0].len(),
        16,
        "embedding dimension must equal configured dim"
    );
    assert!(
        embeddings[0].iter().any(|&v| v != 0.0),
        "embedding must not be all zeros"
    );
}

/// `encode_base64` must return a non-empty RFC 4648 base64 string for
/// non-empty input (finding `serve-api-08`: this endpoint previously emitted
/// lowercase hex, which silently corrupts data for real OpenAI-SDK clients
/// that `base64.b64decode(...)` the result per the API contract).
#[test]
fn test_encode_base64_non_empty() {
    let vec = vec![0.0f32, 1.0f32, -1.0f32, 0.5f32];
    let encoded = EmbedderRegistry::encode_base64(&vec);
    // 4 values × 4 bytes = 16 bytes → ceil(16/3)*4 = 24 base64 chars.
    assert_eq!(
        encoded.len(),
        24,
        "expected 24 base64 chars for 4 f32 values (16 bytes)"
    );
    assert!(!encoded.is_empty());
    // Must be valid RFC 4648 base64 alphabet characters (+ padding).
    assert!(
        encoded
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'='),
        "encoding must only contain RFC 4648 base64 characters, got: {encoded}"
    );
}

/// `encode_base64` known-vector test: the RFC 4648 base64 encoding of
/// `[0.0f32, 1.0f32, -1.0f32, 0.5f32]` must match a real base64 decoder's
/// expectation. Expected string independently computed via Python's
/// standard `base64` module: `base64.b64encode(struct.pack("<4f", 0.0, 1.0,
/// -1.0, 0.5))` == `b"AAAAAAAAgD8AAIC/AAAAPw=="`.
#[test]
fn test_encode_base64_known_value() {
    let encoded = EmbedderRegistry::encode_base64(&[0.0f32, 1.0f32, -1.0f32, 0.5f32]);
    assert_eq!(
        encoded, "AAAAAAAAgD8AAIC/AAAAPw==",
        "RFC 4648 base64 encoding of [0.0, 1.0, -1.0, 0.5] as little-endian f32 bytes"
    );
}

/// Fitting TF-IDF then embedding must return vectors of the vocabulary dimension.
#[test]
fn test_embedder_registry_fit_and_embed() {
    let registry = EmbedderRegistry::new(50);
    let corpus: Vec<String> = (0..10)
        .map(|i| format!("sentence {i} about topic{} and related{}", i % 3, i % 5))
        .collect();
    registry.fit_tfidf(&corpus);

    let dim_after_fit = registry.embedding_dim();
    assert!(
        dim_after_fit > 0,
        "embedding dim must be positive after fit"
    );

    let embeddings = registry.embed_texts(&corpus[0..2]);
    assert_eq!(embeddings.len(), 2);
    for emb in &embeddings {
        assert_eq!(
            emb.len(),
            dim_after_fit,
            "embedding dim must match registry dim after fit"
        );
    }
}
