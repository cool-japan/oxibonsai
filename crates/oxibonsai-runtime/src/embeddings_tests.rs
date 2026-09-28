//! Unit tests for [`crate::embeddings`].
//!
//! Attached to `embeddings.rs` as its `#[cfg(test)] mod tests` via
//! `#[path]`, so the tests keep full access to the module's private items
//! (`EmbedderRegistry`, `create_embeddings`, `base64_encode_bytes`, the
//! `ActiveRequestGuard`, ...) while `embeddings.rs` itself stays under the
//! workspace 2000-line ceiling — the same split `api_extensions.rs` /
//! `api_extensions_tests.rs` and `engine.rs` / `engine_tests.rs` already use.
//!
//! Every test here is byte-for-byte the one that used to live inline in
//! `embeddings.rs`; the move added no test and removed none.
//!
//! Distinct from the crate's external integration files
//! `tests/embeddings_tests.rs` and `tests/embeddings_model_backed.rs`, which
//! exercise the public router surface from outside the crate.

use super::*;

// ── EmbeddingInput ────────────────────────────────────────────────────────

#[test]
fn embedding_input_single_as_strings() {
    let input = EmbeddingInput::Single("hello world".to_string());
    assert_eq!(input.as_strings(), vec!["hello world"]);
    assert_eq!(input.len(), 1);
    assert!(!input.is_empty());
}

#[test]
fn embedding_input_batch_as_strings() {
    let input = EmbeddingInput::Batch(vec!["foo".to_string(), "bar".to_string()]);
    let strings = input.as_strings();
    assert_eq!(strings.len(), 2);
    assert_eq!(strings[0], "foo");
    assert_eq!(strings[1], "bar");
    assert_eq!(input.len(), 2);
}

#[test]
fn embedding_input_token_ids_as_strings() {
    let input = EmbeddingInput::TokenIds(vec![1u32, 2, 3]);
    let strings = input.as_strings();
    assert_eq!(strings.len(), 1);
    assert_eq!(strings[0], "1 2 3");
}

#[test]
fn embedding_input_batch_token_ids_as_strings() {
    let input = EmbeddingInput::BatchTokenIds(vec![vec![10u32, 20], vec![30u32]]);
    let strings = input.as_strings();
    assert_eq!(strings.len(), 2);
    assert_eq!(strings[0], "10 20");
    assert_eq!(strings[1], "30");
}

#[test]
fn embedding_input_empty_batch_is_empty() {
    let input = EmbeddingInput::Batch(vec![]);
    assert!(input.is_empty());
    assert_eq!(input.len(), 0);
}

// ── EmbedderRegistry ─────────────────────────────────────────────────────

#[test]
fn embedder_registry_basic_embed() {
    let registry = EmbedderRegistry::new(32);
    let texts = vec!["hello world".to_string(), "foo bar baz".to_string()];
    let embeddings = registry.embed_texts(&texts);
    assert_eq!(embeddings.len(), 2);
    // Each embedding must have exactly `default_dim` elements.
    for emb in &embeddings {
        assert_eq!(emb.len(), 32, "expected 32 dimensions, got {}", emb.len());
    }
}

#[test]
fn embedder_registry_tfidf_fit_changes_dim() {
    let registry = EmbedderRegistry::new(64);
    let corpus: Vec<String> = (0..20)
        .map(|i| format!("document number {i} with some unique words term{i}"))
        .collect();
    registry.fit_tfidf(&corpus);
    // After fitting the dimension comes from the TF-IDF vocabulary.
    let dim = registry.embedding_dim();
    assert!(dim > 0, "expected positive dimension after fit");
}

#[test]
fn embedder_registry_fit_empty_corpus_is_noop() {
    let registry = EmbedderRegistry::new(16);
    registry.fit_tfidf(&[]);
    // Should still use IdentityEmbedder (dim == default_dim).
    assert_eq!(registry.embedding_dim(), 16);
}

#[test]
fn embedder_registry_embed_after_fit() {
    let registry = EmbedderRegistry::new(32);
    let corpus: Vec<String> = vec![
        "the quick brown fox".to_string(),
        "jumped over the lazy dog".to_string(),
        "the fox and the dog".to_string(),
    ];
    registry.fit_tfidf(&corpus);
    let embeddings = registry.embed_texts(&corpus);
    for emb in &embeddings {
        assert!(!emb.is_empty(), "embedding must not be empty after fit");
    }
}

// ── Poisoned-lock recovery (finding #70) ─────────────────────────────────

/// Regression test: a panic on another thread while holding
/// `EmbedderRegistry`'s internal `tfidf` mutex must not turn every
/// subsequent `POST /v1/embeddings` request into a permanent panic.
/// Before the fix, `embed_texts`/`fit_tfidf`/`embedding_dim` all used
/// `.lock().expect("... poisoned")`, so a single unrelated panic while
/// holding the lock would wedge this (server-reachable) route for the
/// rest of the process lifetime.
#[test]
fn embedder_registry_recovers_from_poisoned_tfidf_lock() {
    let registry = Arc::new(EmbedderRegistry::new(16));

    // Poison the `tfidf` mutex from a background thread that panics
    // while holding the lock.
    {
        let registry = Arc::clone(&registry);
        let handle = std::thread::spawn(move || {
            let _guard = registry.tfidf.lock().expect("lock for poisoning");
            panic!("intentional panic to poison the tfidf mutex");
        });
        let result = handle.join();
        assert!(result.is_err(), "background thread should have panicked");
    }

    // The mutex is now poisoned. Operations that touch it must recover
    // instead of panicking.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let corpus: Vec<String> = vec![
            "the quick brown fox".to_string(),
            "jumped over the lazy dog".to_string(),
        ];
        registry.fit_tfidf(&corpus);
        let embeddings = registry.embed_texts(&corpus);
        let dim = registry.embedding_dim();
        (embeddings, dim)
    }));

    assert!(
        outcome.is_ok(),
        "operations on an EmbedderRegistry with a poisoned `tfidf` mutex must not panic"
    );
    let (embeddings, dim) = outcome.expect("checked is_ok above");
    assert_eq!(embeddings.len(), 2);
    assert!(
        dim > 0,
        "embedding_dim should still be usable after poison recovery"
    );
}

// ── encode_base64 (finding serve-api-08: must be real RFC 4648 base64,
//    not hex) ──────────────────────────────────────────────────────────

#[test]
fn encode_base64_non_empty() {
    let vec = vec![1.0f32, 0.5f32, -1.0f32];
    let encoded = EmbedderRegistry::encode_base64(&vec);
    // 3 f32 values → 12 bytes → 12/3*4 = 16 base64 chars, no padding.
    assert_eq!(
        encoded.len(),
        16,
        "expected 16 base64 chars for 3 f32 values (12 bytes), got {}",
        encoded.len()
    );
    assert!(!encoded.is_empty());
    // Every character must be a valid RFC 4648 base64 alphabet character.
    assert!(encoded
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'='));
}

#[test]
fn encode_base64_empty_input() {
    let encoded = EmbedderRegistry::encode_base64(&[]);
    assert!(encoded.is_empty());
}

#[test]
fn encode_base64_deterministic() {
    let vec = vec![std::f32::consts::PI, 2.71f32];
    let a = EmbedderRegistry::encode_base64(&vec);
    let b = EmbedderRegistry::encode_base64(&vec);
    assert_eq!(a, b, "encoding must be deterministic");
}

/// Regression test for finding `serve-api-08`: the previous
/// implementation emitted lowercase hex ("0000803f") under the
/// `encoding_format: "base64"` contract. The expected string below was
/// computed independently with Python's standard `base64` module
/// (`base64.b64encode(struct.pack("<f", 1.0))` == `b"AACAPw=="`), so this
/// test verifies interoperability with a real RFC 4648 base64 decoder,
/// not just internal self-consistency.
#[test]
fn encode_base64_known_value_matches_real_base64_decoder() {
    // f32::to_le_bytes(1.0) == [0x00, 0x00, 0x80, 0x3f]
    let vec = vec![1.0f32];
    let encoded = EmbedderRegistry::encode_base64(&vec);
    assert_eq!(encoded, "AACAPw==");
}

/// Second independently-computed known vector: `base64.b64encode(
/// struct.pack("<2f", 1.0, 0.5))` == `b"AACAPwAAAD8="`.
#[test]
fn encode_base64_known_value_two_floats() {
    let vec = vec![1.0f32, 0.5f32];
    let encoded = EmbedderRegistry::encode_base64(&vec);
    assert_eq!(encoded, "AACAPwAAAD8=");
}

/// Full round-trip: encode with the production encoder, decode with an
/// independent, standard-conformant base64 decoder (implemented here
/// for the test only), and confirm the reconstructed `f32` bytes match
/// the originals exactly. This is the "real decoder" check the finding
/// asked for: any RFC 4648-conformant decoder (including a real
/// `base64.b64decode`) must be able to reverse our output.
#[test]
fn encode_base64_round_trips_through_independent_decoder() {
    let original = vec![1.0f32, -2.5f32, 0.0f32, std::f32::consts::PI, -999.125f32];
    let encoded = EmbedderRegistry::encode_base64(&original);
    let decoded_bytes = test_base64_decode(&encoded);

    let mut expected_bytes = Vec::with_capacity(original.len() * 4);
    for v in &original {
        expected_bytes.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(
        decoded_bytes, expected_bytes,
        "round-trip through an independent base64 decoder must reproduce \
         the exact little-endian f32 byte sequence"
    );

    // Reinterpret the decoded bytes as f32 values and confirm they match.
    let decoded_floats: Vec<f32> = decoded_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(decoded_floats, original);
}

/// Minimal standard-conformant RFC 4648 base64 decoder, used only to
/// independently verify [`EmbedderRegistry::encode_base64`]'s output in
/// tests (kept separate from the production encoder so the test does
/// not just check the encoder against itself).
fn test_base64_decode(s: &str) -> Vec<u8> {
    fn value_of(c: u8) -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => 0, // padding '=' contributes no bits
        }
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let pad = chunk.iter().filter(|&&b| b == b'=').count();
        let c0 = value_of(chunk[0]);
        let c1 = value_of(*chunk.get(1).unwrap_or(&b'A'));
        let c2 = value_of(*chunk.get(2).unwrap_or(&b'A'));
        let c3 = value_of(*chunk.get(3).unwrap_or(&b'A'));
        let packed = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;
        let b0 = ((packed >> 16) & 0xff) as u8;
        let b1 = ((packed >> 8) & 0xff) as u8;
        let b2 = (packed & 0xff) as u8;
        match pad {
            0 => out.extend_from_slice(&[b0, b1, b2]),
            1 => out.extend_from_slice(&[b0, b1]),
            2 => out.push(b0),
            _ => {}
        }
    }
    out
}

// ── EmbeddingResponse serialisation ──────────────────────────────────────

#[test]
fn embedding_response_serialises_correctly() {
    let resp = EmbeddingResponse {
        object: "list".to_owned(),
        data: vec![EmbeddingObject {
            object: "embedding".to_owned(),
            embedding: EmbeddingData::Float(vec![0.1, 0.2]),
            index: 0,
        }],
        model: "bonsai-embeddings".to_owned(),
        usage: EmbeddingUsage {
            prompt_tokens: 3,
            total_tokens: 3,
        },
        dimension: 2,
        normalized: true,
    };
    let json = serde_json::to_string(&resp).expect("serialisation must succeed");
    assert!(json.contains("\"object\":\"list\""));
    assert!(json.contains("\"object\":\"embedding\""));
    assert!(json.contains("\"index\":0"));
    assert!(json.contains("\"dimension\":2"));
    assert!(json.contains("\"normalized\":true"));
}

// ── RT-08 / SV-02 / sec-19: statelessness, batch cap, backend naming ────

use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

/// POST `body` to `/v1/embeddings` on `app` and return (status, JSON).
async fn post(app: Router, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
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

/// The core acceptance property: embedding the same text twice across two
/// *separate* requests on the *same* router returns a bit-identical
/// vector. Before the fix, a request with >= 2 texts silently fit TF-IDF
/// from its own input, so a second request's embedding of the same text
/// (now scored against a vocabulary the first request installed) could
/// differ in both value and dimension from the first.
#[tokio::test]
async fn embedding_same_text_twice_across_two_requests_is_bit_identical() {
    let app = create_embeddings_router(32);

    // First request: a multi-text batch that, under the old auto-fit
    // behaviour, would have installed a TF-IDF vocabulary derived from
    // *these specific texts* — poisoning every later request.
    let (status1, _) = post(
        app.clone(),
        serde_json::json!({ "input": ["alpha document one", "beta document two"] }),
    )
    .await;
    assert_eq!(status1, StatusCode::OK);

    // Second, unrelated request embeds a fixed probe text.
    let (status_a, json_a) = post(app.clone(), serde_json::json!({ "input": "probe text" })).await;
    assert_eq!(status_a, StatusCode::OK);

    // A third request, with yet another multi-text batch that would
    // previously have re-fit TF-IDF to a *different* vocabulary.
    let (status2, _) = post(
        app.clone(),
        serde_json::json!({ "input": ["gamma document three", "delta document four", "epsilon"] }),
    )
    .await;
    assert_eq!(status2, StatusCode::OK);

    // Re-embedding the exact same probe text must be bit-identical.
    let (status_b, json_b) = post(app, serde_json::json!({ "input": "probe text" })).await;
    assert_eq!(status_b, StatusCode::OK);

    assert_eq!(
        json_a["data"][0]["embedding"], json_b["data"][0]["embedding"],
        "the same text embedded a request apart must return a bit-identical vector"
    );
    assert_eq!(
        json_a["data"][0]["embedding"]
            .as_array()
            .expect("array")
            .len(),
        json_b["data"][0]["embedding"]
            .as_array()
            .expect("array")
            .len(),
        "the embedding dimension must not drift across requests either"
    );
}

/// Same property at the library level (no HTTP layer): two independently
/// constructed registries (simulating two separate process lifetimes)
/// must embed the same text identically, since neither one ever mutates
/// from request content.
#[test]
fn two_independent_registries_embed_the_same_text_identically() {
    let a = EmbedderRegistry::new(24);
    let b = EmbedderRegistry::new(24);
    let out_a = a.embed_texts(&["consistent text".to_string()]);
    let out_b = b.embed_texts(&["consistent text".to_string()]);
    assert_eq!(out_a, out_b);
}

/// `fit_tfidf` remains available as an explicit, administrative
/// operation (existing callers, including the sibling
/// `tests/embeddings_tests.rs` integration suite, rely on this) — the fix
/// removes the *automatic* per-request call from the handler, not the
/// method itself.
#[test]
fn fit_tfidf_remains_an_explicit_public_operation() {
    let registry = EmbedderRegistry::new(50);
    assert_eq!(registry.backend_name(), "identity");
    let corpus: Vec<String> = (0..5).map(|i| format!("doc {i} content")).collect();
    registry.fit_tfidf(&corpus);
    assert_eq!(registry.backend_name(), "tfidf");
}

/// A batch over the (default) cap is rejected with `400`, not silently
/// truncated or accepted at unbounded cost.
#[tokio::test]
async fn batch_over_the_default_cap_is_rejected_with_400() {
    let app = create_embeddings_router(16);
    let inputs: Vec<String> = (0..(DEFAULT_MAX_EMBEDDING_BATCH_SIZE + 1))
        .map(|i| format!("text {i}"))
        .collect();
    let (status, _) = post(app, serde_json::json!({ "input": inputs })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// A batch at or under the cap is accepted.
#[tokio::test]
async fn batch_at_the_cap_is_accepted() {
    let app = create_embeddings_router(16);
    let inputs: Vec<String> = (0..4).map(|i| format!("text {i}")).collect();
    let (status, json) = post(app, serde_json::json!({ "input": inputs })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["data"].as_array().expect("data array").len(), 4);
}

/// `with_max_batch_size` is honoured by the handler (via
/// [`create_embeddings_router_with_model`], the only router constructor
/// that exposes registry construction to this test without a model
/// dependency — installing a trivial passthrough embedder here only to
/// reach the registry builder, and asserting the cap check fires before
/// any embedding happens).
#[tokio::test]
async fn configured_max_batch_size_is_enforced() {
    struct TinyEmbedder;
    impl Embedder for TinyEmbedder {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, oxibonsai_rag::error::RagError> {
            Ok(vec![1.0])
        }
        fn embedding_dim(&self) -> usize {
            1
        }
    }
    let state = Arc::new(EmbeddingAppState::from_registry(
        EmbedderRegistry::new(4)
            .with_max_batch_size(2)
            .with_model_embedder(Arc::new(TinyEmbedder)),
    ));
    let app = Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state);

    let (status_ok, _) = post(app.clone(), serde_json::json!({ "input": ["a", "b"] })).await;
    assert_eq!(status_ok, StatusCode::OK);

    let (status_over, _) = post(app, serde_json::json!({ "input": ["a", "b", "c"] })).await;
    assert_eq!(status_over, StatusCode::BAD_REQUEST);
}

/// `with_max_batch_size(0)` clamps to `1` rather than making the endpoint
/// refuse every request, including a single-input one.
#[test]
fn max_batch_size_zero_is_clamped_to_one() {
    let registry = EmbedderRegistry::new(8).with_max_batch_size(0);
    assert_eq!(registry.max_batch_size(), 1);
}

/// The response `model` field reports which backend answered, not the
/// client's arbitrary `model` request field.
#[tokio::test]
async fn response_model_field_reports_the_real_backend_not_the_client_value() {
    let app = create_embeddings_router(16);
    let (status, json) = post(
        app,
        serde_json::json!({ "input": "hello", "model": "text-embedding-3-large" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json["model"].as_str().expect("model field"),
        "bonsai-embeddings-identity",
        "the client's claimed model name must not be echoed back verbatim"
    );
}

// ── Model-backed embedder seam ───────────────────────────────────────────

/// A trivial deterministic test double standing in for a future
/// `BonsaiModel`-backed embedder.
struct FixedVectorEmbedder {
    dim: usize,
}

impl Embedder for FixedVectorEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, oxibonsai_rag::error::RagError> {
        // Deterministic function of the text length, just distinctive
        // enough to tell apart from Identity/TF-IDF output in a test.
        let mut v = vec![0.0f32; self.dim];
        v[0] = text.len() as f32;
        l2_normalize(&mut v);
        Ok(v)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

#[test]
fn model_embedder_takes_priority_over_identity_and_tfidf() {
    let registry =
        EmbedderRegistry::new(8).with_model_embedder(Arc::new(FixedVectorEmbedder { dim: 4 }));
    // Even after fitting TF-IDF, the model backend must still win.
    registry.fit_tfidf(&["a document".to_string(), "another document".to_string()]);
    assert_eq!(registry.backend_name(), "model");
    assert_eq!(registry.embedding_dim(), 4);
    let out = registry.embed_texts(&["four".to_string()]);
    assert_eq!(out[0].len(), 4);
    assert!(
        out[0][0] > 0.0,
        "expected the model embedder's distinctive first component"
    );
}

#[tokio::test]
async fn router_with_model_embedder_serves_the_model_backend() {
    let app = create_embeddings_router_with_model(8, Arc::new(FixedVectorEmbedder { dim: 4 }));
    let (status, json) = post(app, serde_json::json!({ "input": "hi" })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json["model"].as_str().expect("model field"),
        "bonsai-embeddings-model"
    );
    assert_eq!(
        json["data"][0]["embedding"]
            .as_array()
            .expect("embedding array")
            .len(),
        4
    );
}

// ── D-1 / SV-02: create_embeddings_router_requiring_model ────────────────

#[tokio::test]
async fn router_requiring_model_refuses_with_501_when_none_is_installed() {
    let app = create_embeddings_router_requiring_model(8);
    let (status, json) = post(app, serde_json::json!({ "input": "hi" })).await;
    assert_eq!(
        status,
        StatusCode::NOT_IMPLEMENTED,
        "no model-backed embedder is installed, so this must refuse honestly rather than \
         answer with a byte-hash vector"
    );
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("model-backed"),
        "the error must explain why, not just carry a bare status: {json}"
    );
}

#[tokio::test]
async fn router_requiring_model_is_independent_of_the_default_router() {
    // The default `create_embeddings_router` (used elsewhere in this
    // test suite, and by library embedders) must keep answering 200
    // with the stateless fallback -- this constructor is additive, not
    // a change to that one's documented behavior.
    let default_app = create_embeddings_router(8);
    let (status, _) = post(default_app, serde_json::json!({ "input": "hi" })).await;
    assert_eq!(status, StatusCode::OK);
}

// ── Dimension truncation renormalises (SV-02 correction (a)) ─────────────

#[tokio::test]
async fn dimensions_truncation_returns_a_unit_vector() {
    let app = create_embeddings_router(32);
    let (status, json) = post(
        app,
        serde_json::json!({ "input": "renormalisation test", "dimensions": 5 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let vec: Vec<f32> = json["data"][0]["embedding"]
        .as_array()
        .expect("embedding array")
        .iter()
        .map(|v| v.as_f64().expect("f64") as f32)
        .collect();
    assert_eq!(vec.len(), 5);
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "truncated embedding must be re-normalised to unit length, got norm {norm}"
    );
}

/// When `dimensions` is >= the natural size, the vector is returned
/// unmodified (still unit-length, since every backend already normalises).
#[tokio::test]
async fn dimensions_at_or_above_natural_size_is_a_no_op() {
    let app = create_embeddings_router(8);
    let (status, json) = post(
        app,
        serde_json::json!({ "input": "no truncation needed", "dimensions": 999 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json["data"][0]["embedding"]
            .as_array()
            .expect("embedding array")
            .len(),
        8
    );
}

/// `dimensions: 0` is rejected rather than silently truncating every
/// embedding to an empty vector and returning `200`.
#[tokio::test]
async fn dimensions_zero_is_rejected_with_400() {
    let app = create_embeddings_router(16);
    let (status, json) = post(
        app,
        serde_json::json!({ "input": "hello", "dimensions": 0 }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "dimensions");
}

// ── dimension / normalized response fields (RT-08 fix item 1) ───────────

#[tokio::test]
async fn response_reports_the_natural_dimension() {
    let app = create_embeddings_router(16);
    let (status, json) = post(app, serde_json::json!({ "input": "hello world" })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["dimension"].as_u64(), Some(16));
}

#[tokio::test]
async fn response_dimension_field_reflects_truncation() {
    let app = create_embeddings_router(32);
    let (status, json) = post(
        app,
        serde_json::json!({ "input": "hello world", "dimensions": 5 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["dimension"].as_u64(), Some(5));
}

#[tokio::test]
async fn response_normalized_field_is_true_for_a_genuine_embedding() {
    let app = create_embeddings_router(16);
    let (status, json) = post(app, serde_json::json!({ "input": "hello world" })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["normalized"].as_bool(), Some(true));
}

/// A fully out-of-vocabulary TF-IDF query (default, non-strict mode)
/// falls back to an all-zero vector — the response must report
/// `normalized: false` for it rather than unconditionally claiming
/// `true` regardless of what actually happened (the exact defect class
/// this fix's `dimension`/`normalized` fields exist to avoid
/// reintroducing).
#[tokio::test]
async fn response_normalized_field_is_false_when_an_item_falls_back_to_zero_vector() {
    let registry = EmbedderRegistry::new(16);
    registry.fit_tfidf(&["alpha document".to_string(), "beta document".to_string()]);
    let state = Arc::new(EmbeddingAppState::from_registry(registry));
    let app = Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state);

    let (status, json) = post(app, serde_json::json!({ "input": "zzz" })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json["normalized"].as_bool(),
        Some(false),
        "an all-zero fallback vector must be reported as not normalised, got {json}"
    );
}

/// RT-EMBEDDINGS BLOCKING 2 / SV-02 follow-up: `normalized` must reflect
/// the vector actually shipped, not the pre-truncation one. Reproduced
/// with the real TF-IDF backend: fit on 12 documents that each pair a
/// unique `termN` with the word every document shares (`shared`), so
/// `shared` has the highest document frequency and sorts into vocabulary
/// column 0, while every `termN` ties at document frequency 1 and is
/// broken alphabetically -- putting `term5` at column 8, past a
/// `dimensions: 2` truncation.
///
/// Self-validating: the *untruncated* embedding is asserted non-zero
/// first, proving `term5` really does embed to a genuine, non-degenerate
/// vector, so the truncated all-zero result below is known to come from
/// truncation discarding the signal rather than a broken fixture. Before
/// the fix, `any_zero_vector` was computed on this same non-zero
/// pre-truncation vector, so the truncated response below reported
/// `normalized: true` for a shipped `[0.0, 0.0]`.
#[tokio::test]
async fn response_normalized_field_is_false_when_truncation_zeroes_every_component() {
    let corpus: Vec<String> = (0..12).map(|i| format!("term{i} shared")).collect();
    let registry = EmbedderRegistry::new(64);
    registry.fit_tfidf(&corpus);
    let state = Arc::new(EmbeddingAppState::from_registry(registry));
    let app = Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state);

    // No truncation: `term5` must embed to a genuine (non-degenerate)
    // unit vector -- the fixture's precondition.
    let (status, json) = post(app.clone(), serde_json::json!({ "input": "term5" })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["normalized"].as_bool(), Some(true));
    let full = json["data"][0]["embedding"]
        .as_array()
        .expect("float embedding array")
        .clone();
    assert!(
        full.iter().any(|v| v.as_f64().unwrap_or(0.0) != 0.0),
        "fixture precondition: term5's untruncated embedding must have a \
         non-zero component, got {full:?}"
    );

    // Truncated to the first 2 columns: term5's only non-zero column
    // (8, "shared" occupies 0) is discarded, so the shipped vector is
    // exactly [0.0, 0.0].
    let (status, json) = post(
        app,
        serde_json::json!({ "input": "term5", "dimensions": 2 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["dimension"].as_u64(), Some(2));
    let truncated = json["data"][0]["embedding"]
        .as_array()
        .expect("float embedding array");
    assert!(
        truncated.iter().all(|v| v.as_f64() == Some(0.0)),
        "fixture precondition: truncating to 2 columns must zero every \
         component, got {truncated:?}"
    );
    assert_eq!(
        json["normalized"].as_bool(),
        Some(false),
        "a vector that truncated to all-zero must not be reported as \
         normalized, got {json}"
    );
}

/// Sibling of the above: a truncation that *keeps* the surviving signal
/// must still report `normalized: true` -- the fix moves the check to
/// run after truncation, it does not make the field unconditionally
/// `false` whenever `dimensions` is set. `shared` is the
/// highest-document-frequency term in the same fixture and therefore
/// sorts into vocabulary column 0, so a `dimensions: 2` truncation keeps
/// it.
#[tokio::test]
async fn response_normalized_field_is_true_when_truncation_keeps_a_unit_vector() {
    let corpus: Vec<String> = (0..12).map(|i| format!("term{i} shared")).collect();
    let registry = EmbedderRegistry::new(64);
    registry.fit_tfidf(&corpus);
    let state = Arc::new(EmbeddingAppState::from_registry(registry));
    let app = Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state);

    let (status, json) = post(
        app,
        serde_json::json!({ "input": "shared", "dimensions": 2 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["dimension"].as_u64(), Some(2));
    let embedding = json["data"][0]["embedding"]
        .as_array()
        .expect("float embedding array");
    let norm_sq: f64 = embedding
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0).powi(2))
        .sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-4,
        "fixture precondition: the truncated vector must still be a \
         (re-normalised) unit vector, got {json}"
    );
    assert_eq!(
        json["normalized"].as_bool(),
        Some(true),
        "a truncation that keeps a genuine unit vector must still report \
         normalized: true, got {json}"
    );
}

// ── encoding_format validation (SV-02) ───────────────────────────────────

#[tokio::test]
async fn unknown_encoding_format_is_rejected_with_400() {
    let app = create_embeddings_router(16);
    let (status, json) = post(
        app,
        serde_json::json!({ "input": "hello", "encoding_format": "binary" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "encoding_format");
}

#[tokio::test]
async fn encoding_format_float_is_accepted_explicitly() {
    let app = create_embeddings_router(16);
    let (status, json) = post(
        app,
        serde_json::json!({ "input": "hello", "encoding_format": "float" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["data"][0]["embedding"].is_array());
}

// ── Per-item input length cap (sec-19) ───────────────────────────────────

#[tokio::test]
async fn input_over_the_max_input_len_is_rejected_with_400() {
    let app = create_embeddings_router(16);
    let long_text = "a".repeat(DEFAULT_MAX_EMBEDDING_INPUT_CHARS + 1);
    let (status, json) = post(app, serde_json::json!({ "input": long_text })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "input");
}

#[tokio::test]
async fn input_at_the_max_input_len_is_accepted() {
    let app = create_embeddings_router(16);
    let text = "a".repeat(DEFAULT_MAX_EMBEDDING_INPUT_CHARS);
    let (status, _json) = post(app, serde_json::json!({ "input": text })).await;
    assert_eq!(status, StatusCode::OK);
}

/// Only the offending item's index is named for a multi-input batch, so
/// a caller can locate which entry needs shortening.
#[tokio::test]
async fn per_item_cap_names_the_offending_batch_index() {
    let app = create_embeddings_router(16);
    let long_text = "a".repeat(DEFAULT_MAX_EMBEDDING_INPUT_CHARS + 1);
    let (status, json) = post(app, serde_json::json!({ "input": ["short", long_text] })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = json["error"]["message"].as_str().expect("message");
    assert!(
        message.contains("input[1]"),
        "expected the message to name the offending index (1), got {message:?}"
    );
}

#[tokio::test]
async fn configured_max_input_len_is_enforced() {
    let state = Arc::new(EmbeddingAppState::from_registry(
        EmbedderRegistry::new(4).with_max_input_len(5),
    ));
    let app = Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state);

    let (status_ok, _) = post(app.clone(), serde_json::json!({ "input": "hello" })).await;
    assert_eq!(status_ok, StatusCode::OK);

    let (status_over, json) = post(app, serde_json::json!({ "input": "hello world" })).await;
    assert_eq!(status_over, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "input");
}

#[test]
fn max_input_len_zero_is_clamped_to_one() {
    let registry = EmbedderRegistry::new(8).with_max_input_len(0);
    assert_eq!(registry.max_input_len(), 1);
}

// ── require_model_backend gate (RT-08 / SV-02 correction (b)) ────────────

#[tokio::test]
async fn require_model_backend_returns_501_when_no_model_installed() {
    let state = Arc::new(EmbeddingAppState::from_registry(
        EmbedderRegistry::new(16).with_require_model_backend(true),
    ));
    let app = Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state);

    let (status, _json) = post(app, serde_json::json!({ "input": "hello" })).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn require_model_backend_still_serves_200_when_a_model_is_installed() {
    let state = Arc::new(EmbeddingAppState::from_registry(
        EmbedderRegistry::new(16)
            .with_require_model_backend(true)
            .with_model_embedder(Arc::new(FixedVectorEmbedder { dim: 4 })),
    ));
    let app = Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state);

    let (status, json) = post(app, serde_json::json!({ "input": "hello" })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["model"].as_str(), Some("bonsai-embeddings-model"));
}

#[test]
fn model_backend_required_but_missing_reflects_the_gate() {
    let no_gate = EmbedderRegistry::new(8);
    assert!(!no_gate.model_backend_required_but_missing());

    let gated_no_model = EmbedderRegistry::new(8).with_require_model_backend(true);
    assert!(gated_no_model.model_backend_required_but_missing());

    let gated_with_model = EmbedderRegistry::new(8)
        .with_require_model_backend(true)
        .with_model_embedder(Arc::new(FixedVectorEmbedder { dim: 4 }));
    assert!(!gated_with_model.model_backend_required_but_missing());
}

/// The default router (the one `server.rs` actually calls, and the one
/// the sibling `tests/embeddings_tests.rs` integration suite exercises
/// directly) is unaffected by the gate: `with_require_model_backend`
/// defaults to `false`, so it keeps serving the stateless fallback at
/// `200` — see the module docs' "Backends and determinism" section for
/// why the live `server.rs` call site does not opt into the gate today.
#[tokio::test]
async fn default_router_is_unaffected_by_the_require_model_backend_gate() {
    let app = create_embeddings_router(16);
    let (status, _json) = post(app, serde_json::json!({ "input": "hello" })).await;
    assert_eq!(status, StatusCode::OK);
}

// ── EMBED-WIRE item 3: text over the token ceiling refuses ────────────────

/// A deterministic double implementing both [`Embedder`] and
/// [`EmbeddingTokenCounter`], so the `context_length_exceeded` guard can be
/// exercised without a real model or tokenizer: `count_tokens` is simply the
/// character count, which makes the assertions below exact and easy to
/// reason about.
struct CountingEmbedder {
    dim: usize,
    max_tokens: usize,
}

impl Embedder for CountingEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, oxibonsai_rag::error::RagError> {
        let mut v = vec![0.0f32; self.dim];
        v[0] = text.len() as f32 + 1.0;
        l2_normalize(&mut v);
        Ok(v)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

impl crate::embed_engine::EmbeddingTokenCounter for CountingEmbedder {
    fn count_tokens(&self, text: &str) -> Option<usize> {
        Some(text.chars().count())
    }

    fn max_input_tokens(&self) -> Option<usize> {
        Some(self.max_tokens)
    }
}

/// A registry with `CountingEmbedder` wired as both the model backend
/// ([`EmbedderRegistry::with_model_embedder`]) and the token counter
/// ([`EmbedderRegistry::with_token_counter`]) — pairing them is what keeps
/// [`EmbeddingAppState::from_registry`]'s debug_assert from firing (item 4).
fn counting_registry(max_tokens: usize) -> EmbedderRegistry {
    let double = Arc::new(CountingEmbedder { dim: 4, max_tokens });
    EmbedderRegistry::new(4)
        .with_model_embedder(Arc::clone(&double) as Arc<dyn Embedder>)
        .with_token_counter(
            Arc::clone(&double) as Arc<dyn crate::embed_engine::EmbeddingTokenCounter>
        )
}

#[tokio::test]
async fn text_input_over_the_token_ceiling_is_refused_with_context_length_exceeded() {
    let app =
        create_embeddings_router_from_state(EmbeddingAppState::from_registry(counting_registry(5)));
    let (status, json) = post(app, serde_json::json!({ "input": "123456" })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(
        json["error"]["code"].as_str(),
        Some("context_length_exceeded"),
        "{json}"
    );
    assert_eq!(json["error"]["param"].as_str(), Some("input"), "{json}");
    assert_eq!(json["error"]["n_tokens"].as_u64(), Some(6), "{json}");
    assert_eq!(json["error"]["max_tokens"].as_u64(), Some(5), "{json}");
}

#[tokio::test]
async fn text_input_at_the_token_ceiling_is_accepted() {
    let app =
        create_embeddings_router_from_state(EmbeddingAppState::from_registry(counting_registry(5)));
    let (status, json) = post(app, serde_json::json!({ "input": "12345" })).await;
    assert_eq!(status, StatusCode::OK, "{json}");
}

#[tokio::test]
async fn only_the_offending_batch_item_is_named_in_a_multi_input_request() {
    let app =
        create_embeddings_router_from_state(EmbeddingAppState::from_registry(counting_registry(5)));
    let (status, json) = post(
        app,
        serde_json::json!({ "input": ["short", "way too long an input"] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("input[1]"),
        "the second (over-length) item must be named, not the first: {json}"
    );
}

// ── EMBED-WIRE item 4: builder misuse guard ────────────────────────────────

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "with_token_counter/with_token_embedder was installed without")]
fn with_token_counter_without_a_model_embedder_trips_the_debug_assert() {
    let orphaned = EmbedderRegistry::new(8).with_token_counter(Arc::new(CountingEmbedder {
        dim: 8,
        max_tokens: 100,
    }));
    let _ = EmbeddingAppState::from_registry(orphaned);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "with_token_counter/with_token_embedder was installed without")]
fn with_token_embedder_without_a_model_embedder_trips_the_debug_assert() {
    struct NoOpTokenEmbedder;
    impl crate::embed_engine::TokenSequenceEmbedder for NoOpTokenEmbedder {
        fn embed_token_ids(
            &self,
            _tokens: &[u32],
        ) -> Result<Vec<f32>, oxibonsai_rag::error::RagError> {
            Ok(vec![0.0; 4])
        }
    }
    let orphaned = EmbedderRegistry::new(8).with_token_embedder(Arc::new(NoOpTokenEmbedder));
    let _ = EmbeddingAppState::from_registry(orphaned);
}

#[test]
fn with_token_counter_paired_with_a_model_embedder_does_not_panic() {
    // The non-misuse case: pairing `with_token_counter` with
    // `with_model_embedder` (what `counting_registry` above does, and what
    // `EmbedderRegistry::with_model` does automatically for a `ModelEmbedder`)
    // must NOT trip the guard, regardless of build profile.
    let registry = counting_registry(100);
    let _state = EmbeddingAppState::from_registry(registry);
}

#[test]
fn embedding_dim_agrees_with_the_vectors_the_with_model_embedder_path_returns() {
    // Spec item 4's other half: once a model backend IS installed (even the
    // generic `with_model_embedder` path, not just `with_model`),
    // `embedding_dim()` must equal the length of the vectors that path
    // actually returns.
    let registry =
        EmbedderRegistry::new(999).with_model_embedder(Arc::new(FixedVectorEmbedder { dim: 4 }));
    let vectors = registry.embed_texts(&["hello".to_string(), "world".to_string()]);
    for (i, v) in vectors.iter().enumerate() {
        assert_eq!(
            v.len(),
            registry.embedding_dim(),
            "item {i}: embedding_dim() must agree with the vectors actually returned"
        );
    }
}
