//! Server integration tests covering all endpoints.
//!
//! Uses `axum::body::Body` + `tower::ServiceExt::oneshot` to make in-process
//! HTTP requests without binding to a real socket.  All tests are gated on the
//! `server` feature (the default).

#[cfg(feature = "server")]
mod server_tests {
    use std::sync::Arc;

    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
    };
    use bytes::Bytes;
    use http_body_util::BodyExt;
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_runtime::{
        admin::AdminState,
        engine::InferenceEngine,
        engine_pool::EnginePool,
        metrics::InferenceMetrics,
        sampling::SamplingParams,
        server::{create_router_full, RouterOptions},
    };
    use serde_json::Value;
    use tower::ServiceExt;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Qwen3's `<|im_start|>` id: [`make_server`] serves a
    /// `Qwen3Config::tiny_test()` engine (the Qwen3 vocabulary size) without
    /// a tokenizer and runs a text prompt as this single token.
    const QWEN3_IM_START: u32 = 151_644;

    /// A tokenizer-less server over the tiny test model. Without a tokenizer
    /// a server needs a configured prompt start token to accept a text
    /// prompt at all (it answers `400 tokenizer_required` otherwise), and the
    /// answer's text is empty while `usage` still counts every token.
    fn make_server() -> axum::Router {
        let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
        let metrics = Arc::new(InferenceMetrics::new());
        create_router_full(
            EnginePool::new(vec![engine]),
            None,
            metrics,
            RouterOptions::default().with_prompt_start_token(QWEN3_IM_START),
        )
    }

    async fn collect_body(body: Body) -> Bytes {
        body.collect()
            .await
            .expect("body collection should succeed")
            .to_bytes()
    }

    async fn body_json(body: Body) -> Value {
        let bytes = collect_body(body).await;
        serde_json::from_slice(&bytes).expect("response should be valid JSON")
    }

    // ── Health endpoint ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_health_endpoint_ok() {
        let app = make_server();
        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(resp.status(), StatusCode::OK, "/health must return 200 OK");
    }

    // ── Models endpoint ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_models_endpoint_reports_loaded_model() {
        // `/v1/models` reports the *actually-loaded* model, reading its `id`
        // from the engine's real configuration — never a hard-coded literal.
        // `make_server` loads `Qwen3Config::tiny_test()`, whose `model_name` is
        // `"Bonsai-Tiny-Test"`, so that is the id the endpoint must return.
        let app = make_server();
        let req = Request::builder()
            .method(Method::GET)
            .uri("/v1/models")
            .body(Body::empty())
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(resp.status(), StatusCode::OK, "/v1/models must return 200");

        let json = body_json(resp.into_body()).await;
        assert_eq!(
            json["object"].as_str(),
            Some("list"),
            "object must be 'list'"
        );
        let models = json["data"].as_array().expect("data must be an array");
        let ids: Vec<&str> = models.iter().filter_map(|m| m["id"].as_str()).collect();

        let expected = Qwen3Config::tiny_test().model_name;
        assert!(
            ids.contains(&expected.as_str()),
            "models list must report the loaded model {expected:?}; got: {ids:?}"
        );
    }

    // ── Chat completions ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_chat_completions_basic() {
        let app = make_server();

        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": false
        });

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/chat/completions basic request must return 200"
        );

        let json = body_json(resp.into_body()).await;
        assert!(
            json["choices"].as_array().is_some(),
            "response must contain 'choices'"
        );
    }

    // ── Completions endpoint ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_completions_endpoint_basic() {
        let app = make_server();

        let body = serde_json::json!({
            "prompt": "hello",
            "max_tokens": 2,
            "temperature": 0.0
        });

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/completions basic request must return 200"
        );

        let json = body_json(resp.into_body()).await;
        assert_eq!(
            json["object"].as_str(),
            Some("text_completion"),
            "completions response must have object=text_completion"
        );
    }

    // ── Embeddings endpoint ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_embeddings_endpoint_refuses_without_a_model_backend() {
        // `RT-08`: `make_server()` builds its router the same way the real
        // `oxibonsai serve` binary does (`server.rs::create_router_full`),
        // whose `/v1/embeddings` registry requires a model backend and so
        // disables the stateless TF-IDF/`IdentityEmbedder` fallback. With no
        // model-backed `Embedder` installed, the endpoint must refuse
        // honestly with `501 Not Implemented` naming the missing backend
        // rather than silently answer `200` with a non-semantic byte-hash
        // vector. The bare `create_embeddings_router` (still `200` by its own
        // documented contract) is covered elsewhere, e.g.
        // `tests/cli_surface_tests.rs`'s `embeddings_base64` module and
        // `crates/oxibonsai-runtime/tests/embeddings_tests.rs`.
        let app = make_server();

        let body = serde_json::json!({
            "model": "bonsai-8b",
            "input": "hello world"
        });

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/embeddings")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::NOT_IMPLEMENTED,
            "/v1/embeddings must refuse with 501 when no model-backed embedder is installed \
             (the stateless TF-IDF/identity fallback is disabled on the running server), \
             not silently answer 200 with a byte-hash vector"
        );

        let json = body_json(resp.into_body()).await;
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("model-backed"),
            "the 501 body must name the missing model-backed embedder, not just carry a bare \
             status code; got: {json}"
        );
    }

    // ── Prometheus metrics endpoint ───────────────────────────────────────────

    #[tokio::test]
    async fn test_metrics_endpoint_prometheus_format() {
        let app = make_server();
        let req = Request::builder()
            .method(Method::GET)
            .uri("/metrics")
            .body(Body::empty())
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(resp.status(), StatusCode::OK, "/metrics must return 200");

        let bytes = collect_body(resp.into_body()).await;
        let text = std::str::from_utf8(&bytes).expect("metrics should be valid UTF-8");
        // Prometheus text format uses `#` for comments / TYPE declarations,
        // or may be empty when no metrics have been recorded yet.
        assert!(
            text.is_empty() || text.contains('#') || text.contains('_'),
            "Prometheus output should be valid text format; got: {text}"
        );
    }

    // ── Extended chat completions ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_extended_chat_completions_basic() {
        let app = make_server();

        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "n": 1
        });

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions/extended")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/chat/completions/extended must return 200"
        );

        let json = body_json(resp.into_body()).await;
        assert!(
            json["choices"].as_array().is_some(),
            "extended chat response must contain 'choices'"
        );
    }

    // ── Admin status endpoint ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_status_endpoint() {
        // Build a standalone admin router (without wrapping in the main server).
        let metrics = Arc::new(InferenceMetrics::new());
        let state = Arc::new(AdminState::new(Arc::clone(&metrics)));

        // create_admin_router returns Router<Arc<AdminState>> (already state-seeded).
        // We need to convert it to Router<()> via into_make_service or finalise state.
        let admin_router = axum::Router::new()
            .route(
                "/admin/status",
                axum::routing::get(oxibonsai_runtime::admin::get_status),
            )
            .with_state(state);

        let req = Request::builder()
            .method(Method::GET)
            .uri("/admin/status")
            .body(Body::empty())
            .expect("request should build");

        let resp = admin_router
            .oneshot(req)
            .await
            .expect("oneshot should succeed");

        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/admin/status must return 200"
        );

        let json = body_json(resp.into_body()).await;
        assert!(
            json["version"].as_str().is_some(),
            "admin status must include version field; got: {json}"
        );
        assert!(
            json["uptime_secs"].as_u64().is_some(),
            "admin status must include uptime_secs; got: {json}"
        );
    }
}

/// The model-backed `/v1/embeddings` path, proven from
/// OUTSIDE the `oxibonsai-runtime` crate through the exact production seam
/// (`server::create_router_full` + `RouterOptions::with_embedder`) a real
/// `oxibonsai serve` binary uses — a distinct, additional gate from
/// `crates/oxibonsai-runtime/tests/embeddings_model_backed.rs`'s own
/// outside-the-crate coverage and from `server.rs`'s own in-crate
/// `embedder_wiring` unit tests, the same way the sibling
/// `test_embeddings_endpoint_refuses_without_a_model_backend` (below, in
/// `server_tests`) is a distinct gate from the crate's own `501` unit tests:
/// it catches an accidental visibility regression (a type or function this
/// needs turning non-`pub`) that no in-crate test could.
///
/// Runs under the default `server` feature only: the synthetic GGUF is built
/// through `oxibonsai_testkit::gguf_fixture::GgufFixtureBuilder` (a real
/// `[dev-dependencies]` of the root package)
/// and the fixture tokenizer is the native backend's own
/// `TokenizerBridge::native_from_json_str`, so this module no longer needs
/// `native-tokenizer`'s `oxibonsai-tokenizer` dependency at all.
#[cfg(feature = "server")]
mod model_backed_embeddings_tests {
    use std::sync::{Arc, Mutex};

    use axum::{
        body::Body,
        http::{header, Request, StatusCode},
    };
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_kernels::dispatch::KernelTier;
    use oxibonsai_model::model::BonsaiModel;
    use oxibonsai_runtime::{
        embed_engine::ModelEmbedder,
        engine::InferenceEngine,
        engine_pool::EnginePool,
        metrics::InferenceMetrics,
        sampling::SamplingParams,
        server::{create_router_full, RouterOptions},
        tokenizer_bridge::TokenizerBridge,
    };
    use oxibonsai_testkit::gguf_fixture::{FixtureQuant, GgufFixtureBuilder};
    use oxibonsai_testkit::workspace;
    use tower::ServiceExt;

    // Same shape recipe as
    // `crates/oxibonsai-runtime/tests/embeddings_model_backed.rs`'s own
    // synthetic fixture (proven to load and forward correctly through
    // `BonsaiModel::from_gguf` for the `qwen3` dense architecture) — now
    // built through the shared testkit builder instead of a hand-rolled
    // byte-level copy (T-07).
    const HIDDEN: usize = 128;
    const INTER: usize = 256;
    const LAYERS: usize = 2;
    const N_Q: usize = 4;
    const N_KV: usize = 2;
    const HEAD_DIM: usize = 32;
    const VOCAB: usize = 256;
    const MAX_SEQ: usize = 256;

    fn greedy_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 8,
        }
    }

    /// A small, fully-ternary (`TQ2_0_g128`) synthetic GGUF with real,
    /// deterministic weights, built through
    /// [`oxibonsai_testkit::gguf_fixture::GgufFixtureBuilder`] — the same
    /// shape (tensor names, shapes, `qwen3.*` metadata) the hand-rolled
    /// `FixtureRng`/`tq2_0_g128_pattern`/`build_fixture_gguf` trio used to
    /// produce, minus the local duplicate builder.
    fn build_fixture_gguf() -> Vec<u8> {
        let mut builder = GgufFixtureBuilder::new();
        builder
            .metadata_str("general.architecture", "qwen3")
            .metadata_str("general.name", "ServerIntegrationEmbedFixture")
            .metadata_u32("qwen3.embedding_length", HIDDEN as u32)
            .metadata_u32("qwen3.block_count", LAYERS as u32)
            .metadata_u32("qwen3.attention.head_count", N_Q as u32)
            .metadata_u32("qwen3.attention.head_count_kv", N_KV as u32)
            .metadata_u32("qwen3.feed_forward_length", INTER as u32)
            .metadata_u32("qwen3.vocab_size", VOCAB as u32)
            .metadata_u32("qwen3.context_length", 512)
            .metadata_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
            .metadata_f32("qwen3.rope.freq_base", 10_000.0);

        let mut seed = 1u64;
        let mut next_seed = || {
            seed = seed.wrapping_add(7919);
            seed
        };

        builder
            .tensor(
                "token_embd.weight",
                &[HIDDEN as u64, VOCAB as u64],
                FixtureQuant::F32,
                next_seed(),
            )
            .expect("token_embd.weight");
        builder
            .tensor(
                "output_norm.weight",
                &[HIDDEN as u64],
                FixtureQuant::F32,
                next_seed(),
            )
            .expect("output_norm.weight");
        builder
            .tensor(
                "output.weight",
                &[HIDDEN as u64, VOCAB as u64],
                FixtureQuant::TQ2_0_g128,
                next_seed(),
            )
            .expect("output.weight: HIDDEN is a multiple of QK_TQ2_0_G128");

        for layer in 0..LAYERS {
            let pfx = format!("blk.{layer}");
            for name in ["attn_norm.weight", "ffn_norm.weight"] {
                builder
                    .tensor(
                        &format!("{pfx}.{name}"),
                        &[HIDDEN as u64],
                        FixtureQuant::F32,
                        next_seed(),
                    )
                    .expect(name);
            }
            for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                builder
                    .tensor(
                        &format!("{pfx}.{name}"),
                        &[HEAD_DIM as u64],
                        FixtureQuant::F32,
                        next_seed(),
                    )
                    .expect(name);
            }

            for (name, in_dim, out_dim) in [
                ("attn_q.weight", HIDDEN, N_Q * HEAD_DIM),
                ("attn_k.weight", HIDDEN, N_KV * HEAD_DIM),
                ("attn_v.weight", HIDDEN, N_KV * HEAD_DIM),
                ("attn_output.weight", N_Q * HEAD_DIM, HIDDEN),
                ("ffn_gate.weight", HIDDEN, INTER),
                ("ffn_up.weight", HIDDEN, INTER),
                ("ffn_down.weight", INTER, HIDDEN),
            ] {
                builder
                    .tensor(
                        &format!("{pfx}.{name}"),
                        &[in_dim as u64, out_dim as u64],
                        FixtureQuant::TQ2_0_g128,
                        next_seed(),
                    )
                    .expect(name);
            }
        }

        builder.build().expect("GgufFixtureBuilder::build")
    }

    /// A minimal char-level HF `tokenizer.json`: one single-character `BPE`
    /// vocab entry per lowercase ASCII letter, no merges — so the
    /// synthetic-fixture test's own request (plain lowercase `"abcd"`,
    /// [`synthetic_embedder`]'s only caller) always tokenizes to exactly one
    /// token per character, the same "each char its own token" contract
    /// `OxiTokenizer::char_level_stub` (the `native-tokenizer`-gated fixture
    /// this replaces) used to provide — proven by
    /// [`char_level_tokenizer_json_tokenizes_abcd_to_exactly_four_tokens`]
    /// below rather than assumed.
    const CHAR_LEVEL_TOKENIZER_JSON: &str = r##"{
        "model": {
            "type": "BPE",
            "vocab": {
                "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7,
                "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14,
                "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21,
                "w": 22, "x": 23, "y": 24, "z": 25
            },
            "merges": []
        },
        "added_tokens": [],
        "pre_tokenizer": { "type": "ByteLevel" },
        "decoder": { "type": "ByteLevel" }
    }"##;

    fn synthetic_embedder() -> Arc<ModelEmbedder> {
        let bytes: &'static [u8] = Box::leak(build_fixture_gguf().into_boxed_slice());
        let gguf: &'static GgufFile<'static> = Box::leak(Box::new(
            GgufFile::parse(bytes).expect("parse fixture gguf"),
        ));
        let model = BonsaiModel::from_gguf(gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
        let engine = InferenceEngine::from_model_with_tier(
            model,
            KernelTier::Reference,
            greedy_params(),
            42,
        );
        let tokenizer = Arc::new(
            TokenizerBridge::native_from_json_str(CHAR_LEVEL_TOKENIZER_JSON)
                .expect("char-level fixture tokenizer.json must parse"),
        );
        Arc::new(ModelEmbedder::new(Arc::new(Mutex::new(engine)), tokenizer))
    }

    /// The full server router (`create_router_full`, exactly what
    /// `oxibonsai serve` builds) with `embedder` installed as the
    /// model-backed `/v1/embeddings` backend. The chat-facing engine is an
    /// unrelated weight-less `tiny_test()` one — this test exercises the
    /// embeddings route, not chat.
    fn router_with_embedder(embedder: Arc<ModelEmbedder>) -> (axum::Router, Arc<InferenceMetrics>) {
        let chat_engine =
            InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
        let metrics = Arc::new(InferenceMetrics::new());
        let app = create_router_full(
            EnginePool::new(vec![chat_engine]),
            None,
            Arc::clone(&metrics),
            RouterOptions::default().with_embedder(Some(embedder)),
        );
        (app, metrics)
    }

    async fn post_embeddings(
        app: axum::Router,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let req = Request::post("/v1/embeddings")
            .header(header::CONTENT_TYPE, "application/json")
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

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Pins [`CHAR_LEVEL_TOKENIZER_JSON`]'s own contract directly (no HTTP
    /// round trip, no model): every letter of `"abcd"` is its own token, in
    /// order, none of them the `ByteLevel` pre-tokenizer's ordinary-ASCII
    /// identity mapping producing anything unexpected. This is what makes
    /// `expected_tokens == 4` in the test below a checked fact rather than
    /// an assumption baked into the vocabulary.
    #[test]
    fn char_level_tokenizer_json_tokenizes_abcd_to_exactly_four_tokens() {
        let tokenizer = TokenizerBridge::native_from_json_str(CHAR_LEVEL_TOKENIZER_JSON)
            .expect("char-level fixture tokenizer.json must parse");
        let ids = tokenizer.encode("abcd").expect("encode");
        assert_eq!(
            ids,
            vec![0, 1, 2, 3],
            "\"abcd\" must be exactly a,b,c,d in order"
        );
    }

    /// The synthetic-fixture half of the spec's "weighted synthetic ternary
    /// GGUF from the testkit, or the real 1.7B when OXI_MODEL is set" — this
    /// one always runs, on every machine, with no model files required.
    #[tokio::test]
    async fn model_backed_embeddings_endpoint_serves_200_with_unit_norm_and_real_usage() {
        let embedder = synthetic_embedder();
        // Derived from the fixture tokenizer itself (never hardcoded): the
        // exact count `CHAR_LEVEL_TOKENIZER_JSON` produces for this
        // request's own input, before `embedder` is moved into the router.
        let expected_tokens = embedder.tokenize("abcd").expect("tokenize").len() as u64;
        let (app, metrics) = router_with_embedder(embedder);

        let (status, json) = post_embeddings(app, serde_json::json!({ "input": "abcd" })).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "a real ModelEmbedder installed through RouterOptions::with_embedder on the FULL \
             server router must serve 200: {json}"
        );
        assert_eq!(json["object"].as_str(), Some("list"));

        let dim = json["dimension"].as_u64().expect("dimension field") as usize;
        assert_eq!(
            dim, HIDDEN,
            "the reported dimension must be the model's hidden size"
        );

        let data = json["data"].as_array().expect("data array");
        assert_eq!(data.len(), 1);
        let vector: Vec<f32> = data[0]["embedding"]
            .as_array()
            .expect("embedding array under encoding_format=float")
            .iter()
            .map(|v| v.as_f64().expect("embedding component must be a number") as f32)
            .collect();
        assert_eq!(vector.len(), dim);
        let norm = l2_norm(&vector);
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "a model-backed embedding must be L2-normalised; got norm {norm}"
        );

        // char-level tokenizer: "abcd" is unambiguously 4 tokens — asserted
        // directly (a fixture sanity check), not
        // just implied by `expected_tokens`.
        assert_eq!(
            expected_tokens, 4,
            "the char-level fixture must tokenize \"abcd\" to 4 tokens"
        );
        assert_eq!(
            json["usage"]["prompt_tokens"].as_u64(),
            Some(expected_tokens),
            "usage.prompt_tokens must come from the model's own tokenizer: {json}"
        );

        assert_eq!(metrics.requests_total.get(), 1);
        assert_eq!(metrics.errors_total.get(), 0);
        assert_eq!(
            metrics.prompt_tokens_total.get(),
            expected_tokens,
            "prompt_tokens_total must equal the fixture tokenizer's own count for \"abcd\""
        );
        assert!(
            (metrics.active_requests.get() - 0.0).abs() < f64::EPSILON,
            "the in-flight gauge must return to zero once the request is done"
        );
    }

    /// The real-1.7B half of the spec's "or the real 1.7B when OXI_MODEL is
    /// set" — self-skips (prints and returns, does not fail) when `OXI_MODEL`
    /// is unset, following the same established convention
    /// `crates/oxibonsai-runtime/tests/metal_concurrency_tests.rs` uses for
    /// this exact environment variable. The tokenizer is resolved
    /// independently via `oxibonsai_testkit::workspace::models_dir`, since
    /// `tokenizer.json` is a separate file from whichever GGUF `OXI_MODEL`
    /// names. This crate is a real `[dev-dependencies]` of the root package,
    /// so this test calls the testkit's own
    /// `$OXIBONSAI_MODELS_DIR`-or-`<repo-root>/models` resolver directly
    /// instead of keeping a duplicate local copy; the one behavioural
    /// difference is that the testkit version does not `trim()` the env var
    /// before checking it is non-empty, which is immaterial here (nothing in
    /// this workspace sets `OXIBONSAI_MODELS_DIR` to a whitespace-only
    /// value).
    #[tokio::test]
    async fn model_backed_embeddings_endpoint_serves_200_on_the_real_1_7b_when_available() {
        let Some(model_path) = std::env::var_os("OXI_MODEL") else {
            eprintln!(
                "model_backed_embeddings_endpoint_serves_200_on_the_real_1_7b_when_available: \
                 OXI_MODEL not set -- skipping (set OXI_MODEL=<path to a dense GGUF, e.g. \
                 Ternary-Bonsai-1.7B.gguf> to run this against a real model)"
            );
            return;
        };
        let tokenizer_path = workspace::models_dir().join("tokenizer.json");
        if !tokenizer_path.exists() {
            eprintln!(
                "model_backed_embeddings_endpoint_serves_200_on_the_real_1_7b_when_available: \
                 {} not found -- skipping (set OXIBONSAI_MODELS_DIR)",
                tokenizer_path.display()
            );
            return;
        }
        let tokenizer_path_str = tokenizer_path
            .to_str()
            .expect("models/ path is valid UTF-8");
        let tokenizer = Arc::new(
            TokenizerBridge::from_file(tokenizer_path_str).expect("load the real tokenizer.json"),
        );

        let embedder = match ModelEmbedder::from_gguf_path(
            &model_path,
            tokenizer,
            greedy_params(),
            42,
            MAX_SEQ,
        ) {
            Ok(embedder) => embedder,
            // `OXI_MODEL` is this repo's general "which GGUF to test with"
            // variable and is NOT guaranteed to name a dense model (a
            // hybrid `qwen35` file is a legitimate value elsewhere, e.g.
            // `embed_engine.rs`'s own hybrid-refusal test avoids reusing it
            // for exactly this reason). Skip gracefully rather than panic
            // when it points at one here, instead of asserting this test's
            // own precondition on the caller's environment.
            Err(err)
                if oxibonsai_runtime::engine_seam::engine_error_code(&err)
                    == Some("NOT_A_DENSE_MODEL") =>
            {
                eprintln!(
                    "model_backed_embeddings_endpoint_serves_200_on_the_real_1_7b_when_available: \
                     OXI_MODEL ({}) is a hybrid (qwen35) model, not a dense one -- skipping",
                    model_path.to_string_lossy()
                );
                return;
            }
            Err(err) => panic!("the real GGUF named by OXI_MODEL must load: {err}"),
        };
        let expected_dim = embedder.dimension();
        let expected_tokens = embedder.tokenize("hello world").expect("tokenize").len() as u64;

        let (app, metrics) = router_with_embedder(embedder);
        let (status, json) =
            post_embeddings(app, serde_json::json!({ "input": "hello world" })).await;
        assert_eq!(status, StatusCode::OK, "{json}");
        assert_eq!(json["dimension"].as_u64(), Some(expected_dim as u64));

        let vector: Vec<f32> = json["data"][0]["embedding"]
            .as_array()
            .expect("embedding array")
            .iter()
            .map(|v| v.as_f64().expect("number") as f32)
            .collect();
        assert_eq!(vector.len(), expected_dim);
        let norm = l2_norm(&vector);
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "a real-model embedding must be unit-length; got {norm}"
        );
        assert_eq!(
            json["usage"]["prompt_tokens"].as_u64(),
            Some(expected_tokens),
            "usage.prompt_tokens must equal the real tokenizer's own count for the same text: \
             {json}"
        );
        assert_eq!(metrics.requests_total.get(), 1);
        assert_eq!(metrics.errors_total.get(), 0);
        assert_eq!(
            metrics.prompt_tokens_total.get(),
            expected_tokens,
            "prompt_tokens_total must equal the real tokenizer's own count for the same text"
        );
    }
}

// When the server feature is not enabled, provide a placeholder so the file
// compiles cleanly.
#[cfg(not(feature = "server"))]
#[test]
fn test_server_skipped_no_feature() {
    eprintln!("server feature not enabled; skipping server integration tests");
}
