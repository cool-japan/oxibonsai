use super::*;

// NOTE: the prompt-assembly and marker-neutralization tests moved with
// their code into `server::sanitize`; the shutdown / queue-tracker tests
// moved into `server::lifecycle`; the tool-call-parsing, streaming
// terminal-event and usage-chunk tests moved into `server::chat` (the
// 2000-line-file split that also moved `chat_completions` itself).

#[test]
fn resolve_prompt_sanitization_default_on() {
    // The env var is process-global; only assert the unset-default here to
    // avoid racing other tests. When unset, sanitization is on.
    if std::env::var("OXI_DISABLE_PROMPT_SANITIZATION").is_err() {
        assert!(resolve_prompt_sanitization());
    }
}

// ── SV-03: ChatCompletionResponse matches the OpenAI response shape ──

/// Schema test against the real OpenAI `chat.completion` object
/// (`SV-03`): `created`/`model` were found silently missing from the
/// canonical response (see [`ChatCompletionResponse`]'s field docs for
/// why both are non-optional now) and no test asserted the full member
/// set. Serializes a response and checks every documented member is
/// present with the right shape, rather than re-checking `created`/
/// `model` in isolation.
#[test]
fn chat_completion_response_matches_openai_schema() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-test123".to_string(),
        object: "chat.completion".to_string(),
        created: 1_700_000_000,
        model: "Bonsai-Tiny-Test".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage::text("assistant", "hello"),
            finish_reason: "stop".to_string(),
            logprobs: None,
        }],
        usage: Usage {
            prompt_tokens: 3,
            completion_tokens: 1,
            total_tokens: 4,
        },
    };

    let json = serde_json::to_value(&response).expect("response must serialize");

    // Top-level `chat.completion` object members.
    assert_eq!(json["object"], "chat.completion");
    assert!(json["id"].is_string(), "id must be a string; got {json}");
    assert!(
        json["created"].is_u64(),
        "created must be a Unix timestamp; got {json}"
    );
    assert!(
        json["model"].is_string(),
        "model must be a string; got {json}"
    );
    assert!(
        json["choices"].is_array(),
        "choices must be an array; got {json}"
    );
    assert!(
        json["usage"].is_object(),
        "usage must be an object; got {json}"
    );

    // Per-choice members.
    let choice = &json["choices"][0];
    assert!(choice["index"].is_u64());
    assert!(choice["message"].is_object());
    assert_eq!(choice["message"]["role"], "assistant");
    assert!(choice["finish_reason"].is_string());
    // `logprobs: None` must be omitted entirely (OpenAI only includes it
    // when requested), never serialized as an explicit null member.
    assert!(
        choice.get("logprobs").is_none(),
        "logprobs must be omitted, not null, when not requested; got {json}"
    );

    // Usage members.
    let usage = &json["usage"];
    assert!(usage["prompt_tokens"].is_u64());
    assert!(usage["completion_tokens"].is_u64());
    assert!(usage["total_tokens"].is_u64());
}

// ── SV-12: validate_chat_request's honour-or-reject paths ────────────

/// A `ChatCompletionRequest` with only the required field set and every
/// optional field at its post-deserialization default.
fn minimal_request() -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .expect("minimal request must deserialize")
}

#[test]
fn minimal_request_passes_validation() {
    let req = minimal_request();
    validate_chat_request(&req, 256, MAX_OUTPUT_TOKENS).expect("a bare request must be valid");
}

/// Table-driven coverage of SV-12's "honour or reject, naming the field"
/// 400 paths of `validate_chat_request`. Each case starts from
/// [`minimal_request`] (already known valid), applies one mutation that
/// must fail validation, and checks the rejection names the right field.
#[test]
fn validate_chat_request_rejects_every_unsupported_or_out_of_range_field() {
    /// One table-driven case: a label, the request mutation to apply, and
    /// the field name `validate_chat_request` must name in its
    /// rejection. A named type alias instead of the bare tuple type
    /// (clippy `type_complexity`).
    type ValidationCase = (&'static str, fn(&mut ChatCompletionRequest), &'static str);

    let cases: &[ValidationCase] = &[
        ("min_p out of [0,1] range", |r| r.min_p = Some(1.5), "min_p"),
        ("negative min_p", |r| r.min_p = Some(-0.1), "min_p"),
        ("non-finite min_p", |r| r.min_p = Some(f32::NAN), "min_p"),
        (
            "non-empty logit_bias is not yet supported",
            |r| r.logit_bias = Some(std::collections::HashMap::from([("123".to_string(), 1.0)])),
            "logit_bias",
        ),
        (
            "response_format other than text is not yet supported",
            |r| {
                r.response_format = Some(crate::api_types::ResponseFormat {
                    format_type: "json_object".to_string(),
                    json_schema: None,
                })
            },
            "response_format",
        ),
        (
            "repetition_penalty below 1.0 is invalid",
            |r| r.repetition_penalty = Some(0.5),
            "repetition_penalty",
        ),
        (
            "top_k above the sanity ceiling is invalid",
            |r| r.top_k = Some(2_000_000),
            "top_k",
        ),
    ];

    for (label, mutate, expected_param) in cases {
        let mut req = minimal_request();
        mutate(&mut req);
        let err = validate_chat_request(&req, 256, MAX_OUTPUT_TOKENS)
            .expect_err(&format!("case {label:?} must be rejected"));
        assert_eq!(
            err.1, *expected_param,
            "case {label:?} must name the field {expected_param:?}, got {:?}",
            err.1
        );
    }
}

/// `min_p` anywhere in `[0.0, 1.0]` is honoured per request (it used to be
/// refused for any value above `0.0`), so validation accepts it.
#[test]
fn per_request_min_p_validation_accepts_the_whole_range() {
    for min_p in [0.0f32, 0.05, 0.5, 1.0] {
        let mut req = minimal_request();
        req.min_p = Some(min_p);
        validate_chat_request(&req, 256, MAX_OUTPUT_TOKENS)
            .unwrap_or_else(|e| panic!("min_p {min_p} must validate: {e:?}"));
    }
}

/// `seed` together with `logprobs: true` is honoured (a freshly seeded
/// sampler runs the logits-capturing generation), so validation accepts
/// the combination rather than refusing `seed` by name.
#[test]
fn seed_with_logprobs_validates() {
    let mut req = minimal_request();
    req.seed = Some(42);
    req.logprobs = Some(true);
    req.top_logprobs = Some(2);
    validate_chat_request(&req, 256, MAX_OUTPUT_TOKENS)
        .expect("seed + logprobs is supported and must validate");
}

/// RT-12: a seeded streaming request is honoured (a fresh sampler seeded
/// from it drives the stream — see
/// `chat_stream_with_a_seed_is_reproducible_and_seed_dependent` below), so
/// validation must accept it rather than reject `seed` by name.
#[test]
fn validate_chat_request_accepts_seed_with_stream() {
    let mut req = minimal_request();
    req.seed = Some(42);
    req.stream = true;
    validate_chat_request(&req, 256, MAX_OUTPUT_TOKENS)
        .expect("seed + stream is supported and must validate");
}

#[test]
fn validate_chat_request_accepts_zero_min_p_and_a_supported_response_format() {
    // `min_p: 0.0` is the documented "disabled" sentinel, and
    // `{"type": "text"}` is OpenAI's own default -- neither must be
    // treated as the unsupported cases above.
    let mut req = minimal_request();
    req.min_p = Some(0.0);
    req.response_format = Some(crate::api_types::ResponseFormat {
        format_type: "text".to_string(),
        json_schema: None,
    });
    req.repetition_penalty = Some(1.0);
    req.top_k = Some(40);
    validate_chat_request(&req, 256, MAX_OUTPUT_TOKENS)
        .expect("explicit-but-benign values must not be rejected");
}

// ── RT-07 correction: unrecognized roles must be a 400, not a silent
//    splice into the prompt ────────────────────────────────────────────

#[test]
fn validate_chat_request_accepts_every_known_role() {
    for role in ["system", "user", "assistant", "tool"] {
        let mut req = minimal_request();
        req.messages[0].role = role.to_string();
        validate_chat_request(&req, 256, MAX_OUTPUT_TOKENS)
            .unwrap_or_else(|_| panic!("role {role:?} must be accepted"));
    }
}

#[test]
fn validate_chat_request_rejects_an_unrecognized_role() {
    let mut req = minimal_request();
    req.messages[0].role = "developer".to_string();
    let err = validate_chat_request(&req, 256, MAX_OUTPUT_TOKENS)
        .expect_err("an unrecognized role must be rejected");
    assert_eq!(err.1, "messages");
    assert!(
        err.0.contains("developer"),
        "the rejection must name the offending role, got: {}",
        err.0
    );
}

#[test]
fn validate_chat_request_rejects_an_unrecognized_role_on_a_later_message() {
    let mut req = minimal_request();
    req.messages.push(crate::server::ChatMessage {
        role: "narrator".to_string(),
        content: Some("once upon a time".to_string()),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    });
    let err = validate_chat_request(&req, 256, MAX_OUTPUT_TOKENS)
        .expect_err("a bad role anywhere in the list must be rejected");
    assert_eq!(err.1, "messages");
    assert!(err.0.contains("messages[1]"), "got: {}", err.0);
}

#[test]
fn validate_chat_request_enforces_the_configurable_ceiling_not_just_the_hardcoded_one() {
    // SV-28: the ceiling passed in, not `MAX_OUTPUT_TOKENS`, is what
    // must be enforced -- this is what makes it *configurable*.
    let req = minimal_request();
    assert!(validate_chat_request(&req, 100, 100).is_ok());
    let err = validate_chat_request(&req, 101, 100).expect_err("must reject above the ceiling");
    assert_eq!(err.1, "max_tokens");
}

// ── SV-15(c) / SV-12 (max_completion_tokens): resolve_effective_max_tokens

#[test]
fn resolve_effective_max_tokens_precedence() {
    let mut req = minimal_request();
    // Neither set: falls back to the server-configured default.
    assert_eq!(resolve_effective_max_tokens(&req, 256), 256);

    // Only the deprecated field set: it wins over the server default.
    req.max_tokens = Some(64);
    assert_eq!(resolve_effective_max_tokens(&req, 256), 64);

    // Both set: max_completion_tokens (the modern field) wins.
    req.max_completion_tokens = Some(128);
    assert_eq!(resolve_effective_max_tokens(&req, 256), 128);

    // Only the modern field set.
    req.max_tokens = None;
    assert_eq!(resolve_effective_max_tokens(&req, 256), 128);
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
    // This test only needs *a* config to build a router
    // with, not a production-sized one. `Qwen3Config::bonsai_8b()` made
    // `InferenceEngine::new` -> `BonsaiModel::new` allocate ~5 GB of
    // token_embd + output_weight tables (plus a ~1.2 GB KV cache) just to
    // construct a router in this unit test, ballooning this single test
    // to > 4 GB RSS. `tiny_test()` exercises the identical construction
    // path with negligible memory.
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let params = crate::sampling::SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    let _router = create_router(engine, None);
}

#[test]
fn create_router_with_shared_metrics() {
    // See `create_router_builds_without_tokenizer` above: a tiny config.
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let params = crate::sampling::SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    let metrics = Arc::new(InferenceMetrics::new());
    let _router = create_router_with_metrics(engine, None, Arc::clone(&metrics));
    // Metrics should be accessible from outside
    assert_eq!(metrics.requests_total.get(), 0);
}

// ── SV-14 / SV-21: /readyz and /v1/models/{model} actually exist ─────

mod endpoint_existence {
    use super::*;
    use axum::body::Body;
    use tower::ServiceExt;

    fn tiny_router() -> Router {
        let config = oxibonsai_core::config::Qwen3Config::tiny_test();
        let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        create_router(engine, None)
    }

    async fn get_json(app: Router, path: &str) -> (StatusCode, serde_json::Value) {
        let req = axum::http::Request::get(path)
            .body(Body::empty())
            .expect("build request");
        let resp = app.oneshot(req).await.expect("response");
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    #[tokio::test]
    async fn readyz_reports_ready_when_a_model_is_loaded_and_a_slot_is_free() {
        // README.md previously advertised readiness checks that did not
        // exist at all (SV-14): a bare `#[test]` that `create_router`
        // compiles proves nothing about this. This drives a real
        // request through the real route.
        let (status, json) = get_json(tiny_router(), "/readyz").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["status"], "ready");
        assert_eq!(json["model_loaded"], true);
        assert_eq!(json["engine_slot_available"], true);
    }

    /// A busy server is ready: with the only replica leased (as a running
    /// generation holds it) `/readyz` is still `200 ready`, and its body says
    /// that no replica is idle. `503 not_ready` is for a server that cannot
    /// serve at all, not for one that is saturated.
    #[tokio::test]
    async fn readyz_stays_ready_while_every_replica_is_leased() {
        let engine = InferenceEngine::new(
            oxibonsai_core::config::Qwen3Config::tiny_test(),
            SamplingParams::default(),
            42,
        );
        let pool = EnginePool::new(vec![engine]);
        let app = create_router_full(
            Arc::clone(&pool),
            None,
            Arc::new(InferenceMetrics::new()),
            RouterOptions::default(),
        );

        // The first `/readyz` resolves the model descriptor by leasing a
        // replica, so ask once while the replica is idle: the answer below is
        // then served from the cache and never waits for a replica.
        let (status, json) = get_json(app.clone(), "/readyz").await;
        assert_eq!(status, StatusCode::OK, "{json}");
        assert_eq!(json["engine_slot_available"], true);

        let lease = pool.acquire().await.expect("the replica is idle");
        assert_eq!(pool.idle_count(), 0, "the only replica is leased");
        let (status, json) = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            get_json(app.clone(), "/readyz"),
        )
        .await
        .expect("a saturated server answers its readiness probe without waiting");
        assert_eq!(status, StatusCode::OK, "{json}");
        assert_eq!(json["status"], "ready");
        assert_eq!(json["model_loaded"], true);
        assert_eq!(json["engine_slot_available"], false);

        drop(lease);
        let (status, json) = get_json(app, "/readyz").await;
        assert_eq!(status, StatusCode::OK, "{json}");
        assert_eq!(json["engine_slot_available"], true);
    }

    #[tokio::test]
    async fn get_model_by_id_returns_the_real_loaded_model() {
        let app = tiny_router();
        let descriptor_id = {
            // Resolve the same id the route itself will report, without
            // hardcoding `tiny_test()`'s literal model name here.
            let config = oxibonsai_core::config::Qwen3Config::tiny_test();
            config.model_name.clone()
        };
        let (status, json) = get_json(app, &format!("/v1/models/{descriptor_id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["id"], descriptor_id);
        assert_eq!(json["object"], "model");
    }

    #[tokio::test]
    async fn get_model_by_unknown_id_is_a_real_404_not_axums_bare_default() {
        let (status, json) = get_json(tiny_router(), "/v1/models/no-such-model").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(
            json["error"]["code"], "model_not_found",
            "must be the canonical OpenAI error envelope, not an empty body: {json}"
        );
    }
}

// ── SV-25 / RT-08: the full router's embedder ─────────────────────────

mod embedder_wiring {
    use super::*;
    use axum::body::Body;
    use tower::ServiceExt;

    /// `create_router_full` over a one-replica pool, exactly as a server
    /// binary builds it.
    fn full_router(options: RouterOptions, metrics: &Arc<InferenceMetrics>) -> Router {
        let engine = InferenceEngine::new(
            oxibonsai_core::config::Qwen3Config::tiny_test(),
            SamplingParams::default(),
            42,
        );
        create_router_full(
            EnginePool::new(vec![engine]),
            None,
            Arc::clone(metrics),
            options,
        )
    }

    /// A dedicated, weighted dense embedding engine (real Transformer
    /// blocks) behind a char-level tokenizer whose ids fit its vocabulary.
    fn dense_embedder() -> Arc<crate::embed_engine::ModelEmbedder> {
        let config = oxibonsai_core::config::Qwen3Config {
            hidden_size: 128,
            intermediate_size: 256,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            vocab_size: 64,
            max_context_length: 64,
            ..oxibonsai_core::config::Qwen3Config::tiny_test()
        };
        let engine = InferenceEngine::from_model_with_tier(
            oxibonsai_model::model::BonsaiModel::new_for_testing_with_blocks(config),
            oxibonsai_kernels::KernelTier::Reference,
            SamplingParams::default(),
            42,
        );
        let tokenizer = Arc::new(TokenizerBridge::from_native_tokenizer(
            oxibonsai_tokenizer::OxiTokenizer::char_level_stub(64),
        ));
        crate::embed_engine::ModelEmbedder::from_engine(engine, tokenizer)
            .expect("a dense engine embeds")
    }

    async fn post_embeddings(
        app: Router,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let req = axum::http::Request::post("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&body).expect("body serialisation"),
            ))
            .expect("build request");
        let resp = app.oneshot(req).await.expect("response");
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    /// No embedder configured: the honest `501`, still counted on the
    /// router's shared metrics (`SV-25`).
    #[tokio::test]
    async fn without_an_embedder_the_full_router_refuses_and_counts_it() {
        let metrics = Arc::new(InferenceMetrics::new());
        let app = full_router(RouterOptions::default(), &metrics);
        let (status, json) = post_embeddings(app, serde_json::json!({ "input": "hi" })).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{json}");
        assert_eq!(metrics.requests_total.get(), 1);
        assert_eq!(metrics.errors_total.get(), 1);
    }

    /// `RouterOptions::with_embedder` reaches the route through the
    /// destructured `embedder` binding: real vectors of the model's
    /// hidden width, recorded on the same metrics every route uses.
    #[tokio::test]
    async fn a_configured_embedder_serves_real_vectors_on_the_shared_metrics() {
        let metrics = Arc::new(InferenceMetrics::new());
        let app = full_router(
            RouterOptions::default().with_embedder(Some(dense_embedder())),
            &metrics,
        );
        let (status, json) = post_embeddings(app, serde_json::json!({ "input": "hello" })).await;
        assert_eq!(status, StatusCode::OK, "{json}");
        let vector = json["data"][0]["embedding"]
            .as_array()
            .unwrap_or_else(|| panic!("an embedding vector: {json}"));
        assert_eq!(vector.len(), 128, "the model's hidden width");
        assert!(
            vector
                .iter()
                .all(|v| v.as_f64().is_some_and(f64::is_finite)),
            "{json}"
        );
        assert_eq!(metrics.requests_total.get(), 1);
        assert_eq!(metrics.errors_total.get(), 0);
        assert!(metrics.prompt_tokens_total.get() > 0);
    }
}

// ── Real-model gates on the legacy ternary 1.7B (`OXI_MODEL` + `OXI_TOKENIZER`) ──
//
// Each test resolves the model/tokenizer from `OXI_MODEL`/`OXI_TOKENIZER`
// when set, else through the testkit `models/`/`$OXIBONSAI_MODELS_DIR`
// resolver (`real_model_paths`), and self-skips when neither locates a
// file, recording `executed: false` under `Capability::LegacyModels` with
// the reason. Each records `executed: true` only after every assertion has
// passed. `gpu_argmax_routing` holds the greedy-route gate, `real_model` the
// fallback-template gate. Run them with:
//
//   OXI_MODEL=<path>/Ternary-Bonsai-1.7B.gguf OXI_TOKENIZER=<path>/tokenizer.json \
//     cargo test -p oxibonsai-runtime --release --all-features --lib -- \
//     --test-threads=1 temperature_zero_completion_takes_the_metal_greedy_gpu_path \
//     real_model_fallback_template_tool_call_and_no_think

/// `OXI_MODEL` and `OXI_TOKENIZER` when both are set explicitly, else the
/// legacy 1.7B GGUF and its tokenizer resolved through the testkit
/// `models/`/`$OXIBONSAI_MODELS_DIR` fallback (the same resolver
/// `int8_native_forward_tests.rs` and its siblings use), else the recorded
/// self-skip. An explicit `OXI_MODEL`/`OXI_TOKENIZER` always wins over the
/// fallback for either half independently, so a developer pointing at a
/// non-default file still gets it honoured.
fn real_model_paths(test: &str) -> Option<(String, String)> {
    let var = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
    let model = var("OXI_MODEL").or_else(|| {
        oxibonsai_testkit::workspace::find_model("Ternary-Bonsai-1.7B.gguf")
            .map(|p| p.to_string_lossy().into_owned())
    });
    let tokenizer = var("OXI_TOKENIZER").or_else(|| {
        oxibonsai_testkit::workspace::find_model("tokenizer.json")
            .map(|p| p.to_string_lossy().into_owned())
    });
    match (model, tokenizer) {
        (Some(model), Some(tokenizer)) => Some((model, tokenizer)),
        _ => {
            eprintln!(
                "capability report: {test} SKIPPED — set OXI_MODEL/OXI_TOKENIZER, or \
                 OXIBONSAI_MODELS_DIR to a directory holding Ternary-Bonsai-1.7B.gguf and \
                 tokenizer.json (checked: {:?})",
                oxibonsai_testkit::workspace::models_dir()
            );
            oxibonsai_testkit::capability::record_skipped(
                oxibonsai_testkit::capability::Capability::LegacyModels,
                test,
            );
            None
        }
    }
}

/// POST `body` to `path` and return the status and the body text.
async fn post_text(app: Router, path: &str, body: &serde_json::Value) -> (StatusCode, String) {
    let req = axum::http::Request::post(path)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(body).expect("serialize the request"),
        ))
        .expect("build the request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("response body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// `temperature: 0` on the real 1.7B: the fused Metal GPU-argmax route.
mod gpu_argmax_routing {
    use super::*;
    use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

    /// Legacy prompt 3.
    const P3_PROMPT: &str = "Once upon a time, in a small village by the sea,";
    /// The unpenalised greedy continuation of [`P3_PROMPT`] at `max_tokens:
    /// 32` on `Ternary-Bonsai-1.7B.gguf` through the fused Metal GPU-argmax
    /// route (captured with `oxibonsai run --temperature 0`).
    const P3_METAL_GREEDY_GOLDEN: &str = " there lived a young girl named Lila. She was known for her kindness and her love for the sea. One day, she discovered a mysterious shell that gl";
    /// What a hidden `repetition_penalty: 1.1` used to produce for the same
    /// request — asserted absent, not just "golden present", so a partial
    /// regression (some other penalty creeping back in) still fails loudly
    /// even if a future model or tokenizer change also moves the golden. The
    /// two texts agree through "her love for the sea" and diverge only after
    /// it, which is what makes the pair a real discriminator.
    const P3_OLD_PENALISED_TEXT: &str = " there lived a young girl named Lila. She was known for her kindness and her love for the sea, which she would often explore with her father, who";

    /// `temperature: 0` through `/v1/completions` on the real 1.7B applies
    /// no repetition penalty and takes the fused Metal GPU-argmax path
    /// (`InferenceEngine::greedy_gpu_eligible`), reproducing the Metal greedy
    /// golden. A build without the Metal GPU path (no `metal` feature, or
    /// not macOS) records the skip with that reason: there is no GPU-argmax
    /// route there to discriminate, and the CPU tier's text differs.
    #[tokio::test]
    async fn temperature_zero_completion_takes_the_metal_greedy_gpu_path() {
        const TEST: &str =
            "oxibonsai-runtime::lib::temperature_zero_completion_takes_the_metal_greedy_gpu_path";
        let Some((model_path, tokenizer_path)) = real_model_paths(TEST) else {
            return;
        };
        if !cfg!(all(feature = "metal", target_os = "macos")) {
            eprintln!(
                "capability report: {TEST} SKIPPED — this build has no Metal GPU-argmax path \
                 (needs the `metal` feature on macOS)"
            );
            record_skipped(Capability::LegacyModels, TEST);
            return;
        }
        let start = std::time::Instant::now();
        let tokenizer = TokenizerBridge::from_file(&tokenizer_path).expect("OXI_TOKENIZER loads");

        // `SamplingParams::default()` carries `repetition_penalty: 1.0`,
        // which is what makes a plain `temperature: 0` request genuinely
        // unpenalised with no per-request override needed to prove it.
        let params = crate::sampling::SamplingParams::default();
        assert!(
            (params.repetition_penalty - 1.0).abs() < f32::EPSILON,
            "SamplingParams::default() must be repetition_penalty 1.0 for this test to \
             discriminate anything"
        );
        let engine = InferenceEngine::from_gguf_path(&model_path, params, 42, 4096)
            .expect("OXI_MODEL loads");
        assert!(
            engine.uses_fused_gpu_decode(),
            "the model must decode through the fused Metal graph for greedy_gpu_eligible to \
             be reachable at all — if this fails, the environment (not the code) is the problem"
        );
        let app = create_router(engine, Some(tokenizer));

        let body = serde_json::json!({
            "prompt": P3_PROMPT,
            "max_tokens": 32,
            "temperature": 0.0
        });
        let (status, text) = post_text(app, "/v1/completions", &body).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let json: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let completion = json["choices"][0]["text"]
            .as_str()
            .expect("choices[0].text is a string");
        assert_ne!(
            completion, P3_OLD_PENALISED_TEXT,
            "temperature:0 reproduced the repetition-penalised text — a hidden \
             repetition_penalty is back"
        );
        assert_eq!(completion, P3_METAL_GREEDY_GOLDEN);
        record_executed_timed(Capability::LegacyModels, TEST, start.elapsed());
    }
}

/// The built-in Qwen3/ChatML fallback template on the real 1.7B: `tools`
/// and `enable_thinking: false`.
mod real_model {
    use super::*;
    use oxibonsai_testkit::capability::{record_executed_timed, Capability};

    /// The weather tool every tool-calling request below advertises.
    fn weather_tools() -> serde_json::Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"}
                    },
                    "required": ["city"]
                }
            }
        }])
    }

    /// The legacy 1.7B ships no `tokenizer.chat_template`, so the server
    /// renders the built-in Qwen3/ChatML fallback. That fallback must (a)
    /// put the `tools` schema in front of the model — a request naming a
    /// city comes back as a parsed `get_weather` call with `finish_reason:
    /// "tool_calls"`, streamed or not — and (b) honour `enable_thinking:
    /// false` by closing an empty think block itself, so the answer carries
    /// no `<think>` literal and no `reasoning_content`.
    ///
    /// The tool turn runs with thinking off: under the fallback's official
    /// Qwen3 JSON instruction the real ternary 1.7B then answers with
    /// exactly `<tool_call>\n{"name": "get_weather", "arguments": {"city":
    /// "Tokyo"}}\n</tool_call>`, while with thinking on it tends to drop the
    /// `<tool_call>` opener after its reasoning.
    #[tokio::test]
    async fn real_model_fallback_template_tool_call_and_no_think() {
        const TEST: &str =
            "oxibonsai-runtime::lib::real_model_fallback_template_tool_call_and_no_think";
        let Some((model_path, tokenizer_path)) = real_model_paths(TEST) else {
            return;
        };
        let start = std::time::Instant::now();
        let tokenizer = TokenizerBridge::from_file(&tokenizer_path).expect("OXI_TOKENIZER loads");
        let engine = InferenceEngine::from_gguf_path(
            &model_path,
            crate::sampling::SamplingParams::default(),
            42,
            4096,
        )
        .expect("OXI_MODEL loads");
        let app = create_router(engine, Some(tokenizer));

        // (a) the tools schema reaches the model and its call is parsed.
        let tool_request = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": "What is the weather like in Tokyo right now? Use the get_weather tool."
            }],
            "tools": weather_tools(),
            "chat_template_kwargs": {"enable_thinking": false},
            "temperature": 0.0,
            "max_tokens": 256
        });
        let (status, text) = post_text(app.clone(), "/v1/chat/completions", &tool_request).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let json: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        eprintln!("real 1.7B fallback-template tool turn: {json}");
        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "{json}");
        let calls = choice["message"]["tool_calls"]
            .as_array()
            .unwrap_or_else(|| panic!("tool_calls array: {json}"));
        assert!(!calls.is_empty(), "{json}");
        assert_eq!(calls[0]["function"]["name"], "get_weather", "{json}");
        let arguments: serde_json::Value = serde_json::from_str(
            calls[0]["function"]["arguments"]
                .as_str()
                .unwrap_or_default(),
        )
        .unwrap_or_else(|e| panic!("arguments are a JSON string ({e}): {json}"));
        assert!(
            arguments["city"]
                .as_str()
                .is_some_and(|city| city.to_ascii_lowercase().contains("tokyo")),
            "{json}"
        );
        assert!(
            choice["message"].get("reasoning_content").is_none(),
            "{json}"
        );

        // The same request streamed: the call arrives as one `tool_calls`
        // delta and the final chunk reports `finish_reason: "tool_calls"`.
        let mut stream_request = tool_request.clone();
        stream_request["stream"] = serde_json::json!(true);
        let (status, sse) = post_text(app.clone(), "/v1/chat/completions", &stream_request).await;
        assert_eq!(status, StatusCode::OK, "{sse}");
        eprintln!("real 1.7B fallback-template streamed tool turn: {sse}");
        let chunks: Vec<serde_json::Value> = sse
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .filter_map(|data| serde_json::from_str(data).ok())
            .collect();
        let streamed_calls: Vec<&serde_json::Value> = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["tool_calls"].as_array())
            .flatten()
            .collect();
        assert_eq!(streamed_calls.len(), calls.len(), "{sse}");
        assert_eq!(streamed_calls[0]["index"], 0, "{sse}");
        assert_eq!(
            streamed_calls[0]["function"]["name"], "get_weather",
            "{sse}"
        );
        assert_eq!(
            streamed_calls[0]["function"]["arguments"], calls[0]["function"]["arguments"],
            "the streamed call carries the same arguments as the non-streamed one: {sse}"
        );
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk["choices"][0]["finish_reason"] == "tool_calls"),
            "{sse}"
        );

        // (b) `enable_thinking: false` closes the think block in the prompt.
        let no_think_request = serde_json::json!({
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "chat_template_kwargs": {"enable_thinking": false},
            "temperature": 0.0,
            "max_tokens": 64
        });
        let (status, text) = post_text(app, "/v1/chat/completions", &no_think_request).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let json: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        eprintln!("real 1.7B fallback-template no-think turn: {json}");
        let message = &json["choices"][0]["message"];
        let content = message["content"].as_str().unwrap_or_default();
        assert!(!content.trim().is_empty(), "{json}");
        assert!(
            !content.contains("<think>") && !content.contains("</think>"),
            "no think literal may reach content: {json}"
        );
        assert!(
            message.get("reasoning_content").is_none(),
            "reasoning_content must be absent: {json}"
        );
        record_executed_timed(Capability::LegacyModels, TEST, start.elapsed());
    }
}

// ── RT-26 restore invariant ────────────────────────────────────────

/// Companion to `server::chat::tests`'s
/// `logprobs_with_mismatched_temperature_...` tests: those prove a
/// per-request `logprobs: true` temperature/top_p override is
/// *applied*; this proves it is *restored* afterward — the drop-safety
/// half of the `RT-26` fix (the request-scoped sampler installed around
/// `generate_with_logprobs`): a request's override must never leak
/// onto the next request served by the same pool replica.
#[tokio::test]
async fn logprobs_temperature_override_does_not_leak_onto_the_next_request() {
    let ambient = SamplingParams {
        temperature: 0.9,
        ..SamplingParams::default()
    };
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config, ambient, 42);
    let pool = EnginePool::new(vec![engine]);
    let app = crate::tokenizer_bridge::chat_render::test_fixtures::tokenizerless_router_with_pool(
        Arc::clone(&pool),
        Arc::new(InferenceMetrics::new()),
    );

    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "logprobs": true,
        "temperature": 0.0,
    });
    let req = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).expect("serialize request"),
        ))
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "sanity: the override must be accepted, not rejected"
    );

    // The pool has exactly one replica; acquiring it again after the
    // request completed must observe the *ambient* temperature, not
    // the request's `0.0` override -- proving the swap-then-restore
    // dance actually restores rather than leaking the override onto
    // whichever request this replica serves next.
    let lease = pool.acquire().await.expect("acquire the sole replica back");
    assert_eq!(
        lease.sampling_params().temperature,
        0.9,
        "the per-request temperature override must be restored after the logprobs call, \
         not leaked onto the next request served by this pool replica"
    );
}

// ── RT-12: a seeded chat stream is honoured ────────────────────────────

mod seeded_chat_stream {
    use super::*;
    use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

    /// A router over one engine whose every step is an equal choice among
    /// 26 letters, so each sampled token is decided by the sampler's PRNG
    /// alone.
    fn uniform_router(ambient_seed: u64) -> axum::Router {
        create_router(
            fx::uniform_letters_engine(ambient_seed),
            Some(fx::byte_tokenizer()),
        )
    }

    /// Stream one chat completion on `app` (seeded when `seed` is `Some`)
    /// and return its concatenated `content`.
    async fn streamed_content(app: axum::Router, seed: Option<u64>) -> String {
        let mut body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "temperature": 0.9,
            "stream": true,
        });
        if let Some(seed) = seed {
            body["seed"] = serde_json::json!(seed);
        }
        let (status, _, sse) = fx::post(app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK, "{sse}");
        assert!(sse.trim_end().ends_with("data: [DONE]"), "{sse}");
        fx::delta_texts(&sse, "content").concat()
    }

    /// The same request seed streams the same text on engines with
    /// DIFFERENT ambient seeds (the seed really drives the stream), and a
    /// different request seed streams different text (it is not ignored in
    /// favour of some fixed stream).
    #[tokio::test]
    async fn chat_stream_with_a_seed_is_reproducible_and_seed_dependent() {
        let first = streamed_content(uniform_router(1), Some(7)).await;
        let again = streamed_content(uniform_router(999), Some(7)).await;
        let other = streamed_content(uniform_router(1), Some(8)).await;
        assert!(!first.is_empty(), "the seeded stream must carry content");
        assert_eq!(first, again, "same request seed, different ambient seeds");
        assert_ne!(
            first, other,
            "a different request seed must change the stream"
        );
    }

    /// A seeded stream runs on a fresh sampler and hands the replica's own
    /// sampler back untouched: the unseeded request that follows it streams
    /// exactly what it would have streamed had the seeded one never run.
    #[tokio::test]
    async fn a_seeded_chat_stream_leaves_the_replicas_own_sampler_untouched() {
        let with_a_seeded_request_first = uniform_router(5);
        let _ = streamed_content(with_a_seeded_request_first.clone(), Some(7)).await;
        let after_seeded = streamed_content(with_a_seeded_request_first, None).await;
        let untouched = streamed_content(uniform_router(5), None).await;
        assert!(!untouched.is_empty());
        assert_eq!(
            after_seeded, untouched,
            "the seeded request must not advance or replace the replica's ambient sampler"
        );
    }
}

// ── Every engine refusal code over HTTP ──────────────────────────────────

mod engine_refusals_over_http {
    use super::*;
    use crate::engine_seam::EngineError;
    use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

    /// A router whose sole replica refuses every generation with `refusal`
    /// (the scripted-logits seam fails the first logit row it reads).
    fn refusing_router(refusal: EngineError) -> axum::Router {
        let mut engine = fx::weightless_engine(fx::BYTE_VOCAB, SamplingParams::default(), 42);
        engine.script_failure(refusal);
        create_router(engine, Some(fx::byte_tokenizer()))
    }

    /// The audit of the HTTP mapping after the refusals moved from
    /// `RuntimeError::Config("[CODE] …")` to `RuntimeError::Engine`: no route
    /// ever keyed on the `Config` variant, so every code keeps the generic
    /// generation-failure mapping — `500` on the non-streaming chat,
    /// extended-chat and legacy-completions routes, and the stream's
    /// terminal error object (never a clean `finish_reason`) on the
    /// streaming ones — and every one of them names the stable `[CODE]`.
    #[tokio::test]
    async fn every_engine_refusal_code_is_a_server_error_that_names_its_code() {
        let chat = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        });
        let completion = serde_json::json!({ "prompt": "hi", "max_tokens": 4 });
        let streaming = |body: &serde_json::Value| {
            let mut body = body.clone();
            body["stream"] = serde_json::json!(true);
            body
        };
        let refusals = EngineError::one_of_each_kind();
        assert_eq!(refusals.len(), EngineError::ALL_CODES.len());
        for (refusal, code) in refusals.into_iter().zip(EngineError::ALL_CODES) {
            let tag = format!("[{code}]");
            let non_streaming = [
                ("/v1/chat/completions", chat.clone()),
                ("/v1/chat/completions/extended", chat.clone()),
                ("/v1/completions", completion.clone()),
            ];
            for (path, body) in non_streaming {
                let (status, _, text) =
                    fx::post(refusing_router(refusal.clone()), path, body).await;
                assert_eq!(
                    status,
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "{code} {path}: {text}"
                );
                let json: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
                let message = json["error"]["message"].as_str().unwrap_or_default();
                assert!(message.contains(&tag), "{code} {path}: {text}");
            }
            let streams = [
                ("/v1/chat/completions", streaming(&chat)),
                ("/v1/chat/completions/extended", streaming(&chat)),
                ("/v1/completions", streaming(&completion)),
            ];
            for (path, body) in streams {
                let (status, _, sse) = fx::post(refusing_router(refusal.clone()), path, body).await;
                assert_eq!(status, StatusCode::OK, "{code} {path} (stream): {sse}");
                let error = fx::sse_payloads(&sse)
                    .into_iter()
                    .find(|payload| payload["error"].is_object());
                let message = error
                    .as_ref()
                    .and_then(|e| e["error"]["message"].as_str())
                    .unwrap_or_default();
                assert!(
                    message.contains(&tag),
                    "{code} {path}: the stream must end with an error object naming the code: {sse}"
                );
                assert!(
                    !sse.contains("\"finish_reason\":\"stop\""),
                    "{code} {path}: a refused generation must not look like a clean stop: {sse}"
                );
            }
        }
    }
}
