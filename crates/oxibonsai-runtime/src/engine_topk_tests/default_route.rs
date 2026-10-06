//! The sampled top-k route is **off by default**: an engine nobody
//! configured, and a server request it serves, decode the full logit row and
//! never move the route's counters, while the same engine opted in with
//! `set_sampled_topk(SampledTopKConfig::gpu_candidates())` serves its decode
//! steps from GPU candidates — with byte-identical output either way.
//!
//! The fixture is the fused ternary model of the sibling `metal` module with
//! a 128-token vocabulary. The shipped sampling default (`top_k` 40) has to lie
//! below the vocabulary for the route to be eligible at all: on the 32-token
//! fixture both arms would decode the full row, and "stays 0 by default, moves
//! once opted in" could never hold for the opted-in arm.

use std::sync::Arc;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::MetalGraph;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

use super::metal::{fused_ternary_gguf_with_vocab, MAX_SEQ};
use super::*;
use crate::metrics::InferenceMetrics;

/// Vocabulary of the fixture: above the shipped default `top_k` (40), which
/// itself is within the default candidate count (64).
const VOCAB: usize = 128;
/// Tokens decoded per request.
const TOKENS: usize = 12;
/// Prompt of the engine-level test (all ids inside [`VOCAB`]).
const PROMPT: [u32; 4] = [1, 5, 9, 2];

/// `SamplingParams::default()`, verified to be a request the route could
/// serve if it were on (so the default arm proves something).
fn shipped_default_params() -> SamplingParams {
    let params = SamplingParams::default();
    assert!(params.temperature > 0.0, "the shipped default samples");
    assert!(
        params.top_k > 0 && params.top_k < VOCAB && params.top_k <= DEFAULT_SAMPLED_TOPK_CANDIDATES,
        "the shipped default top_k ({}) must be eligible for the route on the {VOCAB}-token \
         fixture",
        params.top_k
    );
    assert!(
        params.repetition_penalty == 1.0,
        "the shipped default carries no penalty"
    );
    params
}

/// A default-constructed fused engine decodes sampled requests through the
/// full-row path — its sampled-route counters stay 0 and the request is
/// counted as a full-row request — and `SampledTopKConfig::gpu_candidates()`
/// turns the route on for the same engine without changing a token.
#[test]
fn a_default_fused_engine_decodes_the_full_row_and_the_opt_in_serves_candidates() {
    const TEST: &str = "oxibonsai-runtime::lib::\
         a_default_fused_engine_decodes_the_full_row_and_the_opt_in_serves_candidates";
    let Ok(_session) = MetalGraph::bind_new_session() else {
        eprintln!("capability report: {TEST} SKIPPED -- no Metal device");
        record_skipped(Capability::Metal, TEST);
        return;
    };
    let bytes = fused_ternary_gguf_with_vocab(VOCAB);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let params = shipped_default_params();
    let metrics = Arc::new(InferenceMetrics::new());

    // Nothing below configures the route until the explicit opt-in.
    let mut engine =
        InferenceEngine::from_gguf(&gguf, params.clone(), 11, MAX_SEQ).expect("engine");
    if !engine.uses_fused_gpu_decode() {
        eprintln!("capability report: {TEST} SKIPPED -- the fixture is not on the fused route");
        record_skipped(Capability::Metal, TEST);
        return;
    }
    let started = std::time::Instant::now();
    engine.set_metrics(Arc::clone(&metrics));
    assert_eq!(engine.sampled_topk(), SampledTopKConfig::default());
    assert_eq!(engine.sampled_topk().mode, SampledTopKMode::Off);
    assert!(
        !engine.sampled_topk_eligible(false),
        "a default engine is not eligible for the route"
    );

    // The default request: the classic full-row sampler.
    let via_default = engine.generate(&PROMPT, TOKENS).expect("default decode");
    assert_eq!(via_default.len(), TOKENS, "no EOS inside the fixture");
    assert_eq!(engine.stats().sampled_topk_steps(), 0);
    assert_eq!(engine.stats().sampled_topk_full_row_steps(), 0);
    assert_eq!(
        engine.stats().sampled_full_row_requests(),
        1,
        "the request took the fused sampled path and decoded the full row"
    );
    assert_eq!(metrics.sampled_topk_steps_total.get(), 0);
    assert_eq!(metrics.sampled_topk_full_row_steps_total.get(), 0);
    assert_eq!(metrics.sampled_full_row_requests_total.get(), 1);

    let mut reference =
        InferenceEngine::from_gguf(&gguf, params.clone(), 11, MAX_SEQ).expect("engine");
    let classic = classic_sampler_loop(&mut reference, &params, 11, &PROMPT, TOKENS);
    assert_eq!(via_default, classic, "the default is the classic sampler");

    // The same seeded request with the route off, then on: the same tokens.
    engine.reset();
    let seeded_off = engine
        .generate_with_seed(&PROMPT, TOKENS, 23, &params)
        .expect("seeded, default");
    assert_eq!(engine.stats().sampled_topk_steps(), 0);
    assert_eq!(engine.stats().sampled_full_row_requests(), 2);

    engine.set_sampled_topk(SampledTopKConfig::gpu_candidates());
    assert!(
        engine.sampled_topk_eligible(false),
        "once opted in, the shipped default request is eligible"
    );
    engine.reset();
    let seeded_on = engine
        .generate_with_seed(&PROMPT, TOKENS, 23, &params)
        .expect("seeded, opted in");
    assert_eq!(
        seeded_on, seeded_off,
        "the route never changes the output of a seeded request"
    );
    let served = (TOKENS - 1) as u64;
    assert_eq!(engine.stats().sampled_topk_steps(), served);
    assert_eq!(engine.stats().sampled_topk_full_row_steps(), 0);
    assert_eq!(
        engine.stats().sampled_full_row_requests(),
        2,
        "an opted-in request is not a full-row request"
    );
    assert_eq!(metrics.sampled_topk_steps_total.get(), served);
    assert_eq!(metrics.sampled_full_row_requests_total.get(), 2);
    record_executed_timed(Capability::Metal, TEST, started.elapsed());
}

#[cfg(feature = "server")]
mod server {
    use super::*;
    use crate::engine_pool::EnginePool;
    use crate::server::{create_router_full, RouterOptions};

    /// Id a text prompt runs as on the tokenizer-less test server (inside
    /// [`VOCAB`]).
    const START_TOKEN: u32 = 5;

    /// Send `request` through `app` and return the status and the body text.
    async fn call(
        app: axum::Router,
        request: axum::http::Request<axum::body::Body>,
    ) -> (axum::http::StatusCode, String) {
        let response = tower::ServiceExt::oneshot(app, request)
            .await
            .expect("response");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The value of the unlabelled Prometheus series `name`.
    fn series(exposition: &str, name: &str) -> u64 {
        exposition
            .lines()
            .find_map(|line| line.strip_prefix(name)?.strip_prefix(' '))
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or_else(|| panic!("series `{name}` is missing from:\n{exposition}"))
    }

    /// One server over a fused engine (opted in or not), one default chat
    /// request with no sampling field at all, and the `/metrics` scrape that
    /// follows it: `(sampled_topk_steps, sampled_full_row_requests)`.
    async fn serve_one_default_request(
        gguf: &'static GgufFile<'static>,
        route: Option<SampledTopKConfig>,
        test: &str,
    ) -> Option<(u64, u64)> {
        let metrics = Arc::new(InferenceMetrics::new());
        let mut engine = InferenceEngine::from_gguf(gguf, shipped_default_params(), 11, MAX_SEQ)
            .expect("engine");
        if !engine.uses_fused_gpu_decode() {
            eprintln!("capability report: {test} SKIPPED -- the fixture is not on the fused route");
            return None;
        }
        engine.set_metrics(Arc::clone(&metrics));
        // The default arm never touches the route's configuration.
        if let Some(route) = route {
            engine.set_sampled_topk(route);
        }
        let app = create_router_full(
            EnginePool::new(vec![engine]),
            None,
            Arc::clone(&metrics),
            RouterOptions::default().with_prompt_start_token(START_TOKEN),
        );

        // No `temperature`, `top_k` or `top_p`: the server's own defaults over
        // the engine's startup sampling parameters (`SamplingParams::default()`).
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": TOKENS,
        });
        let request = axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::to_vec(&body).expect("serialize the request"),
            ))
            .expect("build the request");
        let (status, text) = call(app.clone(), request).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{text}");
        let json: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(
            json["usage"]["completion_tokens"].as_u64(),
            Some(TOKENS as u64),
            "{json}"
        );

        let scrape = axum::http::Request::get("/metrics")
            .body(axum::body::Body::empty())
            .expect("build the scrape");
        let (status, exposition) = call(app, scrape).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{exposition}");
        let steps = series(&exposition, "oxibonsai_sampled_topk_steps_total");
        let full_row_requests = series(&exposition, "oxibonsai_sampled_full_row_requests_total");
        assert_eq!(
            steps,
            metrics.sampled_topk_steps_total.get(),
            "the scrape reports the engine's counter"
        );
        Some((steps, full_row_requests))
    }

    /// A default server request — no route configuration anywhere, no
    /// sampling field in the body — is decoded on the full row: the scraped
    /// `oxibonsai_sampled_topk_steps_total` stays 0 and the request is
    /// counted as a full-row request. The same request against a server
    /// whose engine opted in is served from GPU candidates (`TOKENS - 1`
    /// steps) and is not a full-row request.
    #[tokio::test]
    async fn a_default_server_request_takes_the_full_row_path_and_an_opted_in_server_serves_candidates(
    ) {
        const TEST: &str = "oxibonsai-runtime::lib::\
             a_default_server_request_takes_the_full_row_path_and_an_opted_in_server_serves_candidates";
        let Ok(_session) = MetalGraph::bind_new_session() else {
            eprintln!("capability report: {TEST} SKIPPED -- no Metal device");
            record_skipped(Capability::Metal, TEST);
            return;
        };
        // The router needs `'static` weights: the fixture lives for the process.
        let bytes: &'static [u8] =
            Box::leak(fused_ternary_gguf_with_vocab(VOCAB).into_boxed_slice());
        let gguf: &'static GgufFile<'static> =
            Box::leak(Box::new(GgufFile::parse(bytes).expect("parse")));
        let started = std::time::Instant::now();

        let Some((default_steps, default_full_row)) =
            serve_one_default_request(gguf, None, TEST).await
        else {
            record_skipped(Capability::Metal, TEST);
            return;
        };
        assert_eq!(
            default_steps, 0,
            "a default server request must not use the GPU top-k route"
        );
        assert_eq!(
            default_full_row, 1,
            "a default server request is decoded on the full row"
        );

        let Some((opted_in_steps, opted_in_full_row)) =
            serve_one_default_request(gguf, Some(SampledTopKConfig::gpu_candidates()), TEST).await
        else {
            record_skipped(Capability::Metal, TEST);
            return;
        };
        assert_eq!(
            opted_in_steps,
            (TOKENS - 1) as u64,
            "an opted-in server serves every decode step from candidates"
        );
        assert_eq!(opted_in_full_row, 0);
        record_executed_timed(Capability::Metal, TEST, started.elapsed());
    }
}
