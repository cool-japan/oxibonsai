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
        admin::AdminState, engine::InferenceEngine, metrics::InferenceMetrics,
        sampling::SamplingParams, server::create_router_with_metrics,
    };
    use serde_json::Value;
    use tower::ServiceExt;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_server() -> axum::Router {
        let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
        let metrics = Arc::new(InferenceMetrics::new());
        create_router_with_metrics(engine, None, metrics)
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
        // D-1 (wave 2.5, `RT-EMBEDDINGS` blocking 1; re-confirmed FIX3-BUILD,
        // wave 3.5): `make_server()` builds its router the same way the real
        // `oxibonsai serve` binary does (`server.rs::create_router_full`),
        // which mounts `/v1/embeddings` via
        // `crate::embeddings::create_embeddings_router_requiring_model`. That
        // constructor disables the stateless TF-IDF/`IdentityEmbedder`
        // fallback, so with no model-backed `Embedder` installed (there is no
        // seam to build one from yet — see `oxibonsai_runtime::embeddings`'s
        // module docs), the endpoint must refuse honestly with `501 Not
        // Implemented` naming the missing backend rather than silently
        // answer `200` with a non-semantic byte-hash vector. This replaces
        // the former `test_embeddings_endpoint_basic`, which asserted `200`
        // against this same router and went red the moment D-1 landed; the
        // bare, unaffected `create_embeddings_router` (still `200` by its
        // own documented, unchanged contract) is covered elsewhere, e.g.
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
             (D-1: the stateless TF-IDF/identity fallback is disabled on the running server), \
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

/// EMBED-WIRE item 6: the model-backed `/v1/embeddings` path, proven from
/// OUTSIDE the `oxibonsai-runtime` crate through the exact production seam
/// (`server::create_router_full` + `RouterOptions::with_embedder`) a real
/// `oxibonsai serve` binary uses — a distinct, additional gate from
/// `crates/oxibonsai-runtime/tests/embeddings_model_backed.rs`'s own
/// outside-the-crate coverage and from `server.rs`'s own in-crate
/// `embedder_wiring` unit tests, the same way this package's sibling
/// `test_embeddings_endpoint_refuses_without_a_model_backend` (below, in
/// `server_tests`) is a distinct gate from the crate's own `501` unit tests:
/// it catches an accidental visibility regression (a type or function this
/// needs turning non-`pub`) that no in-crate test could.
///
/// Needs `native-tokenizer` (the `OxiTokenizer::char_level_stub` fixture
/// tokenizer) in addition to `server`.
#[cfg(all(feature = "server", feature = "native-tokenizer"))]
mod model_backed_embeddings_tests {
    use std::sync::{Arc, Mutex};

    use axum::{
        body::Body,
        http::{header, Request, StatusCode},
    };
    use half::f16;
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
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
    use oxibonsai_tokenizer::OxiTokenizer;
    use tower::ServiceExt;

    // Same shape recipe as
    // `crates/oxibonsai-runtime/tests/embeddings_model_backed.rs`'s own
    // synthetic fixture (proven to load and forward correctly through
    // `BonsaiModel::from_gguf` for the `qwen3` dense architecture). Built
    // directly on `oxibonsai_core::gguf::writer` here rather than through
    // `oxibonsai_testkit::gguf_fixture::GgufFixtureBuilder`, because
    // `oxibonsai-testkit` is not a dependency of THIS crate
    // (`oxibonsai-cli`) — see this package's `deviations`.
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

    /// A minimal deterministic linear-congruential generator, matching
    /// `oxibonsai_testkit::gguf_fixture::Lcg`'s algorithm exactly (same
    /// multiplier/increment) so this fixture's byte layout is the same shape
    /// of "known-good" pattern that crate's own fixtures produce. Local
    /// rather than imported (see the module comment above).
    struct FixtureRng(u64);

    impl FixtureRng {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            })
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            self.0
        }

        /// One ternary 2-bit lane in `{0, 1, 2}` — never the reserved `3`
        /// (`0b11`), which `TQ2_0`-family kernels treat as invalid.
        fn next_ternary_lane(&mut self) -> u8 {
            ((self.next_u64() >> 33) % 3) as u8
        }

        /// One byte packing four valid ternary lanes.
        fn next_valid_tq2_byte(&mut self) -> u8 {
            let mut byte = 0u8;
            for lane in 0..4u8 {
                byte |= self.next_ternary_lane() << (2 * lane);
            }
            byte
        }
    }

    /// `TQ2_0_g128` blob: 32 bytes of 2-bit codes (`00->-1, 01->0, 10->+1`;
    /// `11` is reserved and never emitted) followed by a two-byte FP16
    /// scale, per 128 weights.
    fn tq2_0_g128_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
        assert_eq!(
            num_weights % 128,
            0,
            "num_weights must be a multiple of 128"
        );
        let num_blocks = num_weights / 128;
        let mut data = Vec::with_capacity(num_blocks * 34);
        let mut rng = FixtureRng::new(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
        for _ in 0..num_blocks {
            for _ in 0..32 {
                data.push(rng.next_valid_tq2_byte());
            }
            let scale =
                0.25_f32 + ((rng.next_u64() >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
            data.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        }
        data
    }

    /// An FP32 tensor whose values vary with the index, so no two embedding
    /// rows are identical.
    fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 4);
        for i in 0..n {
            let phase = (i as f32) * 0.013_f32;
            let value = scale * (1.0_f32 + 0.25_f32 * phase.sin());
            v.extend_from_slice(&value.to_le_bytes());
        }
        v
    }

    /// A small, fully-ternary (`TQ2_0_g128`) synthetic GGUF with real,
    /// deterministic weights.
    fn build_fixture_gguf() -> Vec<u8> {
        let mut writer = GgufWriter::new();

        writer.add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen3".to_string()),
        );
        writer.add_metadata(
            "general.name",
            MetadataWriteValue::Str("ServerIntegrationEmbedFixture".to_string()),
        );
        writer.add_metadata(
            "qwen3.embedding_length",
            MetadataWriteValue::U32(HIDDEN as u32),
        );
        writer.add_metadata("qwen3.block_count", MetadataWriteValue::U32(LAYERS as u32));
        writer.add_metadata(
            "qwen3.attention.head_count",
            MetadataWriteValue::U32(N_Q as u32),
        );
        writer.add_metadata(
            "qwen3.attention.head_count_kv",
            MetadataWriteValue::U32(N_KV as u32),
        );
        writer.add_metadata(
            "qwen3.feed_forward_length",
            MetadataWriteValue::U32(INTER as u32),
        );
        writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(VOCAB as u32));
        writer.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
        writer.add_metadata(
            "qwen3.attention.layer_norm_rms_epsilon",
            MetadataWriteValue::F32(1e-6),
        );
        writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

        writer.add_tensor(TensorEntry {
            name: "token_embd.weight".to_string(),
            shape: vec![HIDDEN as u64, VOCAB as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(VOCAB * HIDDEN, 0.5),
        });
        writer.add_tensor(TensorEntry {
            name: "output_norm.weight".to_string(),
            shape: vec![HIDDEN as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(HIDDEN, 1.0),
        });
        writer.add_tensor(TensorEntry {
            name: "output.weight".to_string(),
            shape: vec![HIDDEN as u64, VOCAB as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(VOCAB * HIDDEN, 0xCAFE_BABE),
        });

        for layer in 0..LAYERS {
            let pfx = format!("blk.{layer}");
            for (name, dim) in [
                ("attn_norm.weight", HIDDEN),
                ("ffn_norm.weight", HIDDEN),
                ("attn_q_norm.weight", HEAD_DIM),
                ("attn_k_norm.weight", HEAD_DIM),
            ] {
                writer.add_tensor(TensorEntry {
                    name: format!("{pfx}.{name}"),
                    shape: vec![dim as u64],
                    tensor_type: TensorType::F32,
                    data: f32_pattern(dim, 1.0),
                });
            }

            let seed = 0x2000_0000_u64.wrapping_add((layer as u64) << 16);
            for (name, in_dim, out_dim, bump) in [
                ("attn_q.weight", HIDDEN, N_Q * HEAD_DIM, 0u64),
                ("attn_k.weight", HIDDEN, N_KV * HEAD_DIM, 1),
                ("attn_v.weight", HIDDEN, N_KV * HEAD_DIM, 2),
                ("attn_output.weight", N_Q * HEAD_DIM, HIDDEN, 3),
                ("ffn_gate.weight", HIDDEN, INTER, 4),
                ("ffn_up.weight", HIDDEN, INTER, 5),
                ("ffn_down.weight", INTER, HIDDEN, 6),
            ] {
                writer.add_tensor(TensorEntry {
                    name: format!("{pfx}.{name}"),
                    shape: vec![in_dim as u64, out_dim as u64],
                    tensor_type: TensorType::TQ2_0_g128,
                    data: tq2_0_g128_pattern(in_dim * out_dim, seed.wrapping_add(bump)),
                });
            }
        }

        writer.to_bytes().expect("GgufWriter::to_bytes")
    }

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
        let tokenizer = Arc::new(TokenizerBridge::from_native_tokenizer(
            OxiTokenizer::char_level_stub(VOCAB),
        ));
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

    /// The synthetic-fixture half of the spec's "weighted synthetic ternary
    /// GGUF from the testkit, or the real 1.7B when OXI_MODEL is set" — this
    /// one always runs, on every machine, with no model files required.
    #[tokio::test]
    async fn model_backed_embeddings_endpoint_serves_200_with_unit_norm_and_real_usage() {
        let embedder = synthetic_embedder();
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

        // char-level tokenizer: "abcd" is unambiguously 4 tokens.
        assert_eq!(
            json["usage"]["prompt_tokens"].as_u64(),
            Some(4),
            "usage.prompt_tokens must come from the model's own tokenizer: {json}"
        );

        assert_eq!(metrics.requests_total.get(), 1);
        assert_eq!(metrics.errors_total.get(), 0);
        assert!(metrics.prompt_tokens_total.get() > 0);
        assert!(
            (metrics.active_requests.get() - 0.0).abs() < f64::EPSILON,
            "the in-flight gauge must return to zero once the request is done"
        );
    }

    /// `tokenizer.json`'s directory: `OXIBONSAI_MODELS_DIR` if set (the same
    /// override `oxibonsai_testkit::workspace::models_dir` honours — not
    /// depended on directly here, see this package's `deviations`), else
    /// `<repo-root>/models` resolved from this crate's own
    /// `CARGO_MANIFEST_DIR` (this crate — `oxibonsai-cli` — IS the workspace
    /// root, so no `../..` climb is needed, unlike `oxibonsai-testkit`'s own
    /// two-levels-down crate). Never a hardcoded absolute path.
    fn models_dir() -> std::path::PathBuf {
        if let Ok(dir) = std::env::var("OXIBONSAI_MODELS_DIR") {
            if !dir.trim().is_empty() {
                return std::path::PathBuf::from(dir);
            }
        }
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models")
    }

    /// The real-1.7B half of the spec's "or the real 1.7B when OXI_MODEL is
    /// set" — self-skips (prints and returns, does not fail) when `OXI_MODEL`
    /// is unset, following the same established convention
    /// `crates/oxibonsai-runtime/tests/metal_concurrency_tests.rs` uses for
    /// this exact environment variable. The tokenizer is resolved
    /// independently via [`models_dir`], since `tokenizer.json` is a
    /// separate file from whichever GGUF `OXI_MODEL` names.
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
        let tokenizer_path = models_dir().join("tokenizer.json");
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
        assert!(metrics.prompt_tokens_total.get() > 0);
    }
}

// When the server feature is not enabled, provide a placeholder so the file
// compiles cleanly.
#[cfg(not(feature = "server"))]
#[test]
fn test_server_skipped_no_feature() {
    eprintln!("server feature not enabled; skipping server integration tests");
}
