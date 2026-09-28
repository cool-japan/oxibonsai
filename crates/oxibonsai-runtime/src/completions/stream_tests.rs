//! Unit tests for [`crate::completions`]'s SSE streaming submodule.
//!
//! Attached to `completions/stream.rs` as its `#[cfg(test)] mod tests` via
//! `#[path]`, so they keep full access to that module's private items; kept
//! in their own file to hold `stream.rs` under the workspace's 2000-line
//! ceiling.

use super::*;
use crate::tokenizer_bridge::chat_render::test_fixtures as fx;
use axum::body::to_bytes;

/// What every generation on [`test_router`]'s engine emits, token by
/// token (one byte-level token per character), before it stops on EOS:
/// longer than any `max_tokens` below, and starting with `[` so a stop
/// sequence of `"["` matches on the very first token.
const SCRIPT: &str = "[0][1][2][3][4][5][6][7][8][9][10][11][12]";

/// A weightless engine scripted to emit [`SCRIPT`], with the byte-level
/// tokenizer that decodes it: deterministic text on
/// every streaming path, whatever the sampling parameters. (Before
/// `PieceDecoder`, these tests ran without a tokenizer and leaned on the
/// stream rendering every id as `"[<id>]"`; a stream without a
/// tokenizer now emits no text at all — see
/// `a_stream_without_a_tokenizer_emits_no_id_syntax_and_still_finishes`.)
fn test_router() -> axum::Router {
    crate::server::create_router(fx::scripted_byte_engine(SCRIPT), Some(fx::byte_tokenizer()))
}

/// Every `choices[0].text` of an SSE body, concatenated.
fn streamed_text(body: &str) -> String {
    fx::sse_payloads(body)
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["text"].as_str().map(str::to_string))
        .collect()
}

/// [`test_router`] whose first generation waits, after its first token,
/// to be cancelled (`fx::scripted_byte_engine_held`). The plain streaming
/// path matches a stop sequence on the receiving side, so without the
/// hold the weightless model could finish every token before the
/// receiver has read the first one — and the test would be measuring
/// that race, not whether the match cancels generation.
fn held_stop_router() -> axum::Router {
    crate::server::create_router(
        fx::scripted_byte_engine_held(SCRIPT),
        Some(fx::byte_tokenizer()),
    )
}

async fn post_sse(app: axum::Router, body: serde_json::Value) -> (axum::http::StatusCode, String) {
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let req = Request::post("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&body).expect("body serialisation"),
        ))
        .expect("request build");
    let resp = app.oneshot(req).await.expect("response");
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read sse body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// POST `body` to `/v1/completions` on `app` and return `(status,
/// parsed JSON)`, for a plain (non-`stream`) request — the counterpart
/// to [`post_sse`] above, needed by the cancellation-poisoning tests
/// further down which must send a *non-streaming* request as the
/// second half of the pair. Mirrors
/// `completions_tests.rs::post_completion` exactly; duplicated locally
/// rather than shared since the two `mod tests` are unrelated siblings.
async fn post_completion(
    app: axum::Router,
    body: serde_json::Value,
) -> (axum::http::StatusCode, serde_json::Value) {
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let req = Request::post("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&body).expect("body serialisation"),
        ))
        .expect("request build");
    let resp = app.oneshot(req).await.expect("response");
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

#[tokio::test]
async fn stream_true_returns_ok_not_400() {
    let app = test_router();
    let (status, _) = post_sse(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
}

#[tokio::test]
async fn stream_emits_text_completion_chunks_and_done() {
    let app = test_router();
    let (status, body) = post_sse(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(
        body.contains("\"object\":\"text_completion\""),
        "body: {body}"
    );
    assert!(body.contains("\"finish_reason\":"), "body: {body}");
    assert!(body.trim_end().ends_with("data: [DONE]"), "body: {body}");
}

#[tokio::test]
async fn stream_with_include_usage_emits_a_usage_chunk() {
    let app = test_router();
    let (status, body) = post_sse(
        app,
        serde_json::json!({
            "prompt": "hello", "max_tokens": 3, "stream": true,
            "stream_options": {"include_usage": true},
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(body.contains("\"prompt_tokens\":"), "body: {body}");
    assert!(body.contains("\"total_tokens\":"), "body: {body}");
}

#[tokio::test]
async fn stream_without_include_usage_never_emits_a_real_usage_object() {
    let app = test_router();
    let (status, body) = post_sse(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(!body.contains("\"prompt_tokens\":"), "body: {body}");
}

// ── Batched prompts stream: one SSE stream, chunks indexed by prompt ──

/// Every `choices[0].text` of an SSE body for prompt `index`, concatenated.
fn streamed_text_of(body: &str, index: u64) -> String {
    fx::sse_payloads(body)
        .iter()
        .filter(|chunk| chunk["choices"][0]["index"].as_u64() == Some(index))
        .filter_map(|chunk| chunk["choices"][0]["text"].as_str().map(str::to_string))
        .collect()
}

/// Every `finish_reason` of an SSE body, with the index it was sent for.
fn finish_reasons(body: &str) -> Vec<(u64, String)> {
    fx::sse_payloads(body)
        .iter()
        .filter_map(|chunk| {
            let choice = &chunk["choices"][0];
            let reason = choice["finish_reason"].as_str()?;
            Some((choice["index"].as_u64()?, reason.to_string()))
        })
        .collect()
}

/// A router over the uniform-letters engine: each streamed token is visible
/// text drawn by the (seeded) sampler, so two prompts of a batch stream
/// different, reproducible text.
fn uniform_router() -> axum::Router {
    crate::server::create_router(fx::uniform_letters_engine(3), Some(fx::byte_tokenizer()))
}

/// A batched prompt streams: per index, the concatenated text equals the
/// non-streamed batch response's text for that index, each index gets
/// exactly one `finish_reason`, and every chunk of prompt `i` precedes
/// every chunk of prompt `i + 1`.
#[tokio::test]
async fn batched_prompt_stream_matches_the_non_stream_batch_per_index() {
    for logprobs in [false, true] {
        let mut request = serde_json::json!({
            "prompt": ["first", "second", "third"],
            "max_tokens": 6,
            "temperature": 0.9,
            "seed": 11,
        });
        if logprobs {
            request["logprobs"] = serde_json::json!(1);
        }
        let (status, json) = post_completion(uniform_router(), request.clone()).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{json}");
        let mut stream_request = request.clone();
        stream_request["stream"] = serde_json::json!(true);
        let (status, sse) = post_sse(uniform_router(), stream_request).await;
        assert_eq!(status, axum::http::StatusCode::OK, "body: {sse}");
        assert!(sse.trim_end().ends_with("data: [DONE]"), "body: {sse}");

        for index in 0..3u64 {
            let non_stream = json["choices"][index as usize]["text"]
                .as_str()
                .expect("choice text");
            assert!(!non_stream.is_empty(), "{json}");
            assert_eq!(
                streamed_text_of(&sse, index),
                non_stream,
                "prompt {index} (logprobs={logprobs}): {sse}"
            );
        }
        let reasons = finish_reasons(&sse);
        assert_eq!(
            reasons,
            vec![
                (0, "length".to_string()),
                (1, "length".to_string()),
                (2, "length".to_string()),
            ],
            "one finish_reason per index, in prompt order: {sse}"
        );
        let indices: Vec<u64> = fx::sse_payloads(&sse)
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["index"].as_u64())
            .collect();
        assert!(
            indices.windows(2).all(|pair| pair[0] <= pair[1]),
            "prompt i's chunks precede prompt i + 1's: {indices:?}"
        );
    }
}

/// `stream_options.include_usage` on a streamed batch: one usage chunk,
/// after the last finish chunk, aggregated over every prompt.
#[tokio::test]
async fn batched_prompt_stream_aggregates_include_usage() {
    let (status, sse) = post_sse(
        uniform_router(),
        serde_json::json!({
            "prompt": ["a", "bb"], "max_tokens": 4, "stream": true, "seed": 5,
            "stream_options": {"include_usage": true},
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "body: {sse}");
    let payloads = fx::sse_payloads(&sse);
    let usage_chunks: Vec<&serde_json::Value> = payloads
        .iter()
        .filter(|chunk| chunk["usage"].is_object())
        .collect();
    assert_eq!(usage_chunks.len(), 1, "{sse}");
    assert!(
        payloads
            .last()
            .is_some_and(|last| last["usage"].is_object()),
        "the usage chunk comes last: {sse}"
    );
    assert_eq!(usage_chunks[0]["usage"]["completion_tokens"], 8, "{sse}");
    assert_eq!(finish_reasons(&sse).len(), 2, "{sse}");
}

/// A stop sequence stops only its own prompt of a streamed batch: that
/// index reports `"stop"`, the next prompt still streams to its limit.
#[tokio::test]
async fn batched_prompt_stream_stops_each_prompt_on_its_own() {
    let (status, sse) = post_sse(
        test_router(),
        serde_json::json!({
            "prompt": ["a", "b"], "max_tokens": 12, "stream": true, "stop": "[2]"
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "body: {sse}");
    assert_eq!(streamed_text_of(&sse, 0), "[0][1]", "{sse}");
    assert_eq!(streamed_text_of(&sse, 1), "[0][1]", "{sse}");
    assert_eq!(
        finish_reasons(&sse),
        vec![(0, "stop".to_string()), (1, "stop".to_string())],
        "{sse}"
    );
}

// ── `logprobs` + `stream` is a real, honoured combination ────────────

/// `stream: true` + `logprobs` must be accepted (`200`) and every
/// `text_completion` chunk carrying non-empty text must carry its own
/// `logprobs` object with the legacy-completions parallel-array shape
/// (`tokens`/`token_logprobs`/`top_logprobs`/`text_offset`, all the
/// same length) — B8's "stream per-token logprobs in the
/// text_completion chunks" — ending in a real `finish_reason` and
/// `[DONE]`.
#[tokio::test]
async fn stream_with_logprobs_emits_a_logprobs_object_per_chunk() {
    let app = test_router();
    let (status, body) = post_sse(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 4, "stream": true, "logprobs": 2 }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "body: {body}");
    assert!(
        body.trim_end().ends_with("data: [DONE]"),
        "must still terminate with [DONE]: {body}"
    );
    assert!(
        body.contains("\"finish_reason\":\"stop\"")
            || body.contains("\"finish_reason\":\"length\""),
        "must carry a real finish_reason: {body}"
    );

    let mut saw_a_logprobs_chunk = false;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(data).expect("chunk must be JSON");
        let Some(choice) = v["choices"].get(0) else {
            continue;
        };
        let Some(text) = choice["text"].as_str() else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        let logprobs = &choice["logprobs"];
        assert!(
            !logprobs.is_null(),
            "a non-empty text_completion delta must carry logprobs: {data}"
        );
        let tokens = logprobs["tokens"].as_array().expect("tokens array");
        let token_logprobs = logprobs["token_logprobs"]
            .as_array()
            .expect("token_logprobs");
        let top_logprobs = logprobs["top_logprobs"].as_array().expect("top_logprobs");
        let text_offset = logprobs["text_offset"].as_array().expect("text_offset");
        assert_eq!(tokens.len(), token_logprobs.len());
        assert_eq!(tokens.len(), top_logprobs.len());
        assert_eq!(tokens.len(), text_offset.len());
        assert!(!tokens.is_empty(), "tokens must not be empty: {data}");
        saw_a_logprobs_chunk = true;
    }
    assert!(
        saw_a_logprobs_chunk,
        "no chunk in the stream carried a logprobs object: {body}"
    );
}

/// `text_offset` must be a genuine running cumulative offset across
/// chunks, not reset to `0` on every one — the same convention
/// `completions.rs::build_completion_logprobs` uses non-streaming.
#[tokio::test]
async fn stream_with_logprobs_text_offset_accumulates_across_chunks() {
    let app = test_router();
    let (status, body) = post_sse(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 6, "stream": true, "logprobs": 1 }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);

    let mut offsets: Vec<u64> = Vec::new();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(data).expect("chunk json");
        if let Some(offs) = v["choices"][0]["logprobs"]["text_offset"].as_array() {
            for o in offs {
                offsets.push(o.as_u64().expect("offset is a number"));
            }
        }
    }
    assert!(
        offsets.len() >= 2,
        "need at least 2 offsets to prove accumulation, got {offsets:?}"
    );
    for pair in offsets.windows(2) {
        assert!(
            pair[1] > pair[0],
            "text_offset must strictly increase across tokens, got {offsets:?}"
        );
    }
}

/// `logprobs` + `stream` + `stop` together: the stream must still stop
/// emitting once the stop sequence is found — proving B8's dedicated
/// pipeline honours `stop` too, not just the plain content path.
#[tokio::test]
async fn stream_with_logprobs_and_stop_sequence_truncates() {
    let app = test_router();
    let (status, body) = post_sse(
        app,
        serde_json::json!({
            "prompt": "hello", "max_tokens": 32, "stream": true, "logprobs": 1, "stop": "["
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(
        body.contains("\"finish_reason\":\"stop\""),
        "a stop-sequence match must report \"stop\": {body}"
    );
    let mut total_text = String::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(t) = v["choices"][0]["text"].as_str() {
                    total_text.push_str(t);
                }
            }
        }
    }
    assert_eq!(
        total_text, "",
        "text must be truncated at the stop sequence found at position 0, got {total_text:?}"
    );
}

/// `stream` + `seed` is honoured, on both
/// streaming paths. Every step of the engine is an equal choice among
/// 26 letters, so each sampled token is decided by the sampler's PRNG
/// alone: two engines with DIFFERENT ambient seeds stream the same text
/// for the same request seed (the seed really drives the stream), and a
/// different request seed streams different text (it is not ignored in
/// favour of some fixed stream).
#[tokio::test]
async fn stream_with_a_seed_is_reproducible_and_seed_dependent() {
    async fn run(ambient_seed: u64, request_seed: u64, logprobs: bool) -> String {
        let app = crate::server::create_router(
            fx::uniform_letters_engine(ambient_seed),
            Some(fx::byte_tokenizer()),
        );
        let mut body = serde_json::json!({
            "prompt": "hello", "max_tokens": 16, "stream": true,
            "temperature": 0.9, "seed": request_seed,
        });
        if logprobs {
            body["logprobs"] = serde_json::json!(1);
        }
        let (status, sse) = post_sse(app, body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "body: {sse}");
        assert!(sse.trim_end().ends_with("data: [DONE]"), "body: {sse}");
        streamed_text(&sse)
    }

    for logprobs in [false, true] {
        let first = run(1, 7, logprobs).await;
        let again = run(999, 7, logprobs).await;
        let other = run(1, 8, logprobs).await;
        assert!(
            !first.is_empty(),
            "the seeded stream must carry text (logprobs={logprobs})"
        );
        assert_eq!(
            first, again,
            "the same request seed must stream identical text on engines with different \
             ambient seeds (logprobs={logprobs})"
        );
        assert_ne!(
            first, other,
            "a different request seed must stream different text (logprobs={logprobs})"
        );
    }
}

/// Without a tokenizer there is no text to show, and the stream must not
/// invent any (it used to render every id as `"[<id>]"`) — yet generation
/// still runs, and the stream still ends with a real `finish_reason` and
/// `[DONE]`. (A tokenizer-less server needs a configured prompt start token
/// to accept a text prompt at all.)
#[tokio::test]
async fn a_stream_without_a_tokenizer_emits_no_id_syntax_and_still_finishes() {
    for logprobs in [false, true] {
        let mut engine = crate::engine::InferenceEngine::new(
            oxibonsai_core::config::Qwen3Config::tiny_test(),
            SamplingParams::default(),
            42,
        );
        engine.script_generation(vec![10, 20, 30]);
        let app = fx::tokenizerless_router(engine);
        let mut body = serde_json::json!({
            "prompt": "hello", "max_tokens": 8, "stream": true,
            "stream_options": {"include_usage": true},
        });
        if logprobs {
            body["logprobs"] = serde_json::json!(1);
        }
        let (status, sse) = post_sse(app, body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "body: {sse}");
        assert!(sse.trim_end().ends_with("data: [DONE]"), "body: {sse}");
        assert_eq!(
            streamed_text(&sse),
            "",
            "no text without a tokenizer (logprobs={logprobs}): {sse}"
        );
        for id in ["[10]", "[20]", "[30]"] {
            assert!(!sse.contains(id), "id syntax {id} leaked: {sse}");
        }
        assert!(
            sse.contains("\"finish_reason\":\"stop\""),
            "the scripted generation ends on EOS (logprobs={logprobs}): {sse}"
        );
        let usage = fx::sse_payloads(&sse)
            .into_iter()
            .find(|chunk| chunk["usage"].is_object())
            .unwrap_or_default();
        assert_eq!(
            usage["usage"]["completion_tokens"].as_u64(),
            Some(3),
            "all three tokens were still generated (logprobs={logprobs}): {sse}"
        );
    }
}

/// a streaming `/v1/completions` response carries
/// `x-request-id` like the non-streaming one — the client's own id when
/// it sent a well-formed one, a fresh one otherwise.
#[tokio::test]
async fn streaming_completions_answer_with_an_x_request_id() {
    use axum::http::Request;
    use tower::ServiceExt;

    const CLIENT_ID: &str = "0123456789abcdef0123456789abcdef";
    for logprobs in [false, true] {
        let mut body = serde_json::json!({ "prompt": "hello", "max_tokens": 4, "stream": true });
        if logprobs {
            body["logprobs"] = serde_json::json!(1);
        }
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .header(crate::server::REQUEST_ID_HEADER, CLIENT_ID)
            .body(axum::body::Body::from(
                serde_json::to_vec(&body).expect("body serialisation"),
            ))
            .expect("request build");
        let resp = test_router().oneshot(req).await.expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        let echoed = resp
            .headers()
            .get(crate::server::REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.replace('-', ""));
        assert_eq!(
            echoed.as_deref(),
            Some(CLIENT_ID),
            "the client's request id must be echoed (logprobs={logprobs})"
        );

        let (status, headers, _) = fx::post(test_router(), "/v1/completions", body.clone()).await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(
            headers
                .get(crate::server::REQUEST_ID_HEADER)
                .is_some_and(|v| v.len() == 36),
            "a fresh request id must be generated when none was sent \
             (logprobs={logprobs}): {headers:?}"
        );
    }
}

/// Wait (bounded) for `histogram` to reach `expected` observations: the
/// SSE pump (`sse_response`) owns the stream on its own task, so the
/// stream's drop — and its duration observation — lands just after the
/// client stops reading, not synchronously with it.
async fn wait_for_observations(histogram: &crate::metrics::Histogram, expected: u64) -> u64 {
    for _ in 0..500 {
        if histogram.count() >= expected {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    histogram.count()
}

/// a stream observes `request_duration_seconds`
/// once, when it ends — both when it is read to the end and when the
/// client drops it without reading a byte.
#[tokio::test]
async fn a_stream_observes_its_request_duration_when_it_ends() {
    use axum::http::Request;
    use tower::ServiceExt;

    for logprobs in [false, true] {
        let metrics = Arc::new(crate::metrics::InferenceMetrics::new());
        let app = crate::server::create_router_with_metrics(
            fx::scripted_byte_engine(SCRIPT),
            Some(fx::byte_tokenizer()),
            Arc::clone(&metrics),
        );
        let mut body = serde_json::json!({ "prompt": "hello", "max_tokens": 4, "stream": true });
        if logprobs {
            body["logprobs"] = serde_json::json!(1);
        }
        assert_eq!(metrics.request_duration_seconds.count(), 0);

        let (status, sse) = post_sse(app.clone(), body.clone()).await;
        assert_eq!(status, axum::http::StatusCode::OK, "body: {sse}");
        assert_eq!(
            wait_for_observations(&metrics.request_duration_seconds, 1).await,
            1,
            "a stream read to the end observes its duration once (logprobs={logprobs})"
        );

        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::to_vec(&body).expect("body serialisation"),
            ))
            .expect("request build");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        drop(resp);
        assert_eq!(
            wait_for_observations(&metrics.request_duration_seconds, 2).await,
            2,
            "a stream the client dropped unread still observes its duration \
             (logprobs={logprobs})"
        );
    }
}

#[tokio::test]
async fn stream_with_echo_includes_the_prompt_text() {
    let app = test_router();
    let (status, body) = post_sse(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true, "echo": true }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(body.contains("hello"), "body: {body}");
}

// ── B7: a stop-sequence match actually cancels generation ────────────

/// [`test_router`]'s engine is scripted to emit [`SCRIPT`], whose first
/// token is `[`, so a stop sequence of `"["` matches on the very first
/// token — deterministic and model-behavior-independent, the same
/// "stop at position 0" shape
/// `completions_tests.rs::handler_stop_sequence_truncates_the_completion_end_to_end`
/// uses for the non-streaming path. Before the B7 fix this
/// reported `finish_reason: "length"` (`generate_streaming_with_params`
/// was never told to stop, so it ran all the way to `max_tokens`); now
/// it must report `"stop"` AND generation must actually have been cut
/// short (fewer than `max_tokens` tokens produced).
#[tokio::test]
async fn stream_stop_sequence_match_reports_stop_not_length_and_actually_cancels() {
    let app = held_stop_router();
    let (status, body) = post_sse(
        app,
        serde_json::json!({
            "prompt": "hello", "max_tokens": 32, "stream": true, "stop": "["
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(
        body.contains("\"finish_reason\":\"stop\""),
        "a stop-sequence match must report finish_reason \"stop\", not \"length\": {body}"
    );
    assert!(
        !body.contains("\"finish_reason\":\"length\""),
        "must not ALSO carry a length finish event: {body}"
    );

    // The content itself must be empty (truncated at position 0, the
    // very first token) -- proving the match was caught immediately,
    // not merely mislabeled after running to completion.
    let mut total_text = String::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(t) = v["choices"][0]["text"].as_str() {
                    total_text.push_str(t);
                }
            }
        }
    }
    assert_eq!(
        total_text, "",
        "text must be truncated at the stop sequence found at position 0, got {total_text:?}"
    );
}

/// Companion to the above: WITHOUT a stop sequence, the same request
/// still runs to `max_tokens` and reports `"length"` — proving the fix
/// didn't just always report `"stop"`.
#[tokio::test]
async fn stream_without_stop_sequence_still_reports_length_at_the_token_limit() {
    let app = test_router();
    let (status, body) = post_sse(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(
        body.contains("\"finish_reason\":\"length\""),
        "no stop sequence configured: must exhaust max_tokens and report \"length\": {body}"
    );
}

// ── cross-request cancellation isolation, exercised over real HTTP ────
//
// `EngineLease::Drop` (`engine_pool.rs`, `SV-09`) already guarantees
// `engine_pool::tests::a_cancelled_request_does_not_poison_the_next_lease`:
// a request's cancellation token is cleared (`clear_cancellation_token`)
// the moment its lease returns to the pool, whether or not that token
// was ever actually cancelled. That test proves the guarantee at the
// pool API; `test_router()` here builds a pool with exactly one
// replica, so these two confirm the same guarantee still holds when the
// token really was cancelled mid-stream (a live stop-sequence match,
// `B7`) and the request is driven through the real `/v1/completions`
// handlers (both the plain-content and the logprobs streaming closures)
// rather than the pool directly.
#[tokio::test]
async fn a_stream_stopped_by_a_real_cancel_does_not_affect_the_next_request() {
    let app = held_stop_router();
    let (stream_status, body) = post_sse(
        app.clone(),
        serde_json::json!({
            "prompt": "hello", "max_tokens": 32, "stream": true, "stop": "["
        }),
    )
    .await;
    assert_eq!(stream_status, axum::http::StatusCode::OK);
    assert!(
        body.contains("\"finish_reason\":\"stop\""),
        "sanity: the stop sequence must actually have matched: {body}"
    );

    let (status, json) = post_completion(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3 }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(
        json["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "a request after a real cancellation on this replica must still \
         generate tokens, got {json}"
    );
}

// ── Stop-sequence cancellation on the real model ────────────────────
//
// Every test above uses `Qwen3Config::tiny_test()` — a real
// `InferenceEngine`, but not real generation, so a byte-accurate
// stop-sequence match landing inside real, unpredictable model output is
// untested there. The real GGUF and its tokenizer come from `OXI_MODEL` /
// `OXI_TOKENIZER` only; without them the test records a skip, so a fresh
// clone still passes `cargo test`.

/// The capability record name of the real-model test below.
const REAL_STREAM_STOP_TEST: &str = "oxibonsai-runtime::lib::\
     real_model_stream_stop_sequence_matches_the_non_stream_text_and_reports_stop";

/// The real-GGUF `InferenceEngine` + tokenizer as a router, or `None` —
/// with the skip recorded — when `OXI_MODEL` / `OXI_TOKENIZER` are not set.
fn real_model_router() -> Option<axum::Router> {
    use oxibonsai_testkit::capability::{record_skipped, Capability};
    let var = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
    let (Some(model_path), Some(tokenizer_path)) = (var("OXI_MODEL"), var("OXI_TOKENIZER")) else {
        eprintln!(
            "capability report: {REAL_STREAM_STOP_TEST} SKIPPED — set OXI_MODEL \
             (Ternary-Bonsai-1.7B.gguf) and OXI_TOKENIZER (its tokenizer.json)"
        );
        record_skipped(Capability::LegacyModels, REAL_STREAM_STOP_TEST);
        return None;
    };
    let tokenizer = crate::tokenizer_bridge::TokenizerBridge::from_file(&tokenizer_path)
        .expect("OXI_TOKENIZER loads");
    let params = SamplingParams::default();
    let engine = crate::engine::InferenceEngine::from_gguf_path(&model_path, params, 42, 4096)
        .expect("OXI_MODEL loads");
    Some(crate::server::create_router(engine, Some(tokenizer)))
}

/// Legacy prompt 3 at temperature 0 (deterministic): with `stop: ["sea"]`,
/// generation must actually be cut short at that match — not merely have
/// its SSE emission suppressed while decoding ran on to `max_tokens` in
/// the background — and the streamed text must equal the non-streaming
/// path's text byte for byte, since both go through the same
/// `StopSequenceMatcher` over the same deterministic generation.
#[tokio::test]
async fn real_model_stream_stop_sequence_matches_the_non_stream_text_and_reports_stop() {
    let Some(app) = real_model_router() else {
        return;
    };
    let stream_body = serde_json::json!({
        "prompt": "Once upon a time, in a small village by the sea,",
        "max_tokens": 32,
        "temperature": 0.0,
        "stop": ["sea"],
        "stream": true
    });

    let (stream_status, sse_body) = post_sse(app.clone(), stream_body).await;
    assert_eq!(stream_status, axum::http::StatusCode::OK);
    assert!(
        sse_body.contains("\"finish_reason\":\"stop\""),
        "a real stop-sequence match must report finish_reason \"stop\": {sse_body}"
    );
    assert!(
        !sse_body.contains("\"finish_reason\":\"length\""),
        "must not ALSO carry a length finish event: {sse_body}"
    );
    let mut stream_text = String::new();
    for line in sse_body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(t) = v["choices"][0]["text"].as_str() {
                    stream_text.push_str(t);
                }
            }
        }
    }
    assert!(
        !stream_text.contains("sea"),
        "the stop sequence itself must never reach the client: {stream_text:?}"
    );

    let non_stream_body = serde_json::json!({
        "prompt": "Once upon a time, in a small village by the sea,",
        "max_tokens": 32,
        "temperature": 0.0,
        "stop": ["sea"]
    });
    let (status, json) = post_completion(app, non_stream_body).await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    let non_stream_text = json["choices"][0]["text"].as_str().unwrap_or_default();

    assert_eq!(
        stream_text, non_stream_text,
        "streamed text must equal the non-streaming path's text byte for byte"
    );
    oxibonsai_testkit::capability::record_executed(
        oxibonsai_testkit::capability::Capability::LegacyModels,
        REAL_STREAM_STOP_TEST,
    );
}

/// Same confirmation through the streaming loop's logprobs branch
/// (`generate_streaming_logprobs_chunks`), which arms its own cancellation
/// for every prompt.
#[tokio::test]
async fn a_logprobs_stream_stopped_by_a_real_cancel_does_not_affect_the_next_request() {
    let app = test_router();
    let (stream_status, body) = post_sse(
        app.clone(),
        serde_json::json!({
            "prompt": "hello", "max_tokens": 32, "stream": true, "logprobs": 2, "stop": "["
        }),
    )
    .await;
    assert_eq!(stream_status, axum::http::StatusCode::OK);
    assert!(
        body.contains("\"finish_reason\":\"stop\""),
        "sanity: the stop sequence must actually have matched: {body}"
    );

    let (status, json) = post_completion(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3 }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(
        json["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "a request after a real cancellation on this replica must still \
         generate tokens, got {json}"
    );
}
