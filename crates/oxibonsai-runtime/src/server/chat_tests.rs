//! Unit tests for [`crate::server::chat`].
//!
//! Attached to `chat.rs` as its `#[cfg(test)] mod tests` via `#[path]`, so
//! the tests keep full access to the module's private items
//! (`SamplingOverrides`, `parse_base_tool_calls`, `correct_logprob_bytes`,
//! `StopTracker`, `stream_terminal_payloads`, ...) while `chat.rs` itself
//! stays under the workspace's 2000-line ceiling — mirrors
//! `api_extensions.rs` / `api_extensions_tests.rs` (itself modelled on
//! `engine.rs` / `engine_tests.rs`). Split out by B2-13 when adding the
//! XML-tool-call and TOK-M1-closing tests pushed `chat.rs` from 1997 to
//! 2042 lines; no test content changed in the move.

use super::*;

// ── gatekeeper REQUIRED#1(b): SamplingOverrides seeds from the engine's
//    ambient params, never from SamplingParams::default() ─────────────

/// A discriminating ambient config: every field differs from
/// `SamplingParams::default()`, so a bug that seeds from the bare
/// default instead of this value is caught on every field, not just
/// the one(s) a request happens to override.
fn distinctive_ambient() -> SamplingParams {
    SamplingParams {
        temperature: 0.9,
        top_k: 7,
        top_p: 0.55,
        repetition_penalty: 1.0,
        max_tokens: 33,
    }
}

#[test]
fn sampling_overrides_seed_from_ambient_not_default_when_request_overrides_nothing() {
    // A request that overrides *only* temperature (the one field
    // `ChatCompletionRequest` cannot leave `None`) must still carry
    // every other ambient value through untouched.
    let overrides = SamplingOverrides {
        temperature: 0.0,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let ambient = distinctive_ambient();
    let params = overrides.apply(&ambient);

    assert_eq!(params.temperature, 0.0, "the request's own override wins");
    assert_eq!(
        params.top_k, 7,
        "must come from the engine's ambient config, not SamplingParams::default()'s 40"
    );
    assert!(
        (params.top_p - 0.55).abs() < f32::EPSILON,
        "must come from ambient, not SamplingParams::default()'s 0.9: got {}",
        params.top_p
    );
    assert_eq!(
        params.repetition_penalty, 1.0,
        "must come from ambient (which is already 1.0 here, matching the fixed default)"
    );
}

#[test]
fn sampling_overrides_apply_lets_every_explicit_field_win_over_ambient() {
    let overrides = SamplingOverrides {
        temperature: 0.3,
        top_p: Some(0.42),
        top_k: Some(99),
        repetition_penalty: Some(1.25),
    };
    let params = overrides.apply(&distinctive_ambient());
    assert_eq!(params.temperature, 0.3);
    assert!((params.top_p - 0.42).abs() < f32::EPSILON);
    assert_eq!(params.top_k, 99);
    assert_eq!(params.repetition_penalty, 1.25);
    // max_tokens has no request-level override seam on this struct
    // (the effective completion length is threaded separately via
    // `resolve_effective_max_tokens`) -- ambient passes through as-is.
    assert_eq!(params.max_tokens, 33);
}

#[test]
fn sampling_overrides_from_request_reads_every_field() {
    let mut body = minimal_chat_request();
    body.temperature = 0.0;
    body.top_p = Some(0.8);
    body.top_k = Some(5);
    body.repetition_penalty = Some(1.2);
    let overrides = SamplingOverrides::from_request(&body);
    assert_eq!(overrides.temperature, 0.0);
    assert_eq!(overrides.top_p, Some(0.8));
    assert_eq!(overrides.top_k, Some(5));
    assert_eq!(overrides.repetition_penalty, Some(1.2));
}

/// A `ChatCompletionRequest` with only the required field set and every
/// optional field at its post-deserialization default -- the shared
/// fixture the tests in this module build on top of.
fn minimal_chat_request() -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .expect("minimal request must deserialize")
}

// ── RT-26 fix regression: an omitted temperature must never 400 ──────

#[test]
fn logprobs_with_omitted_temperature_is_not_rejected_even_against_a_custom_ambient() {
    // The bug this guards against: comparing the wire value of
    // `temperature` (which defaults to 0.7 whether or not the client
    // sent it) straight against the engine's ambient temperature would
    // 400 *every* logprobs request that said nothing about temperature,
    // on any server whose operator configured a non-0.7 default.
    let body = minimal_chat_request(); // temperature omitted -> wire default 0.7
    assert!(
        (body.temperature - default_temperature()).abs() < f32::EPSILON,
        "sanity: an omitted field must deserialize to the wire default"
    );

    let ambient = SamplingParams {
        temperature: 0.9, // operator-configured; deliberately != 0.7
        ..distinctive_ambient()
    };
    let overrides = SamplingOverrides::from_request(&body);
    let params = overrides.apply(&ambient);

    // Reproduce exactly the guard's own condition.
    let temperature_explicit = (overrides.temperature - default_temperature()).abs() > f32::EPSILON;
    assert!(
        !temperature_explicit,
        "an omitted temperature must never be treated as an explicit override"
    );
    let _ = params; // silence unused-var lints if the condition above is ever inlined away
}

#[test]
fn logprobs_with_an_explicit_mismatched_temperature_is_applied_not_rejected() {
    // The counterpart, post-fix: a client that explicitly asks for a
    // different temperature/top_p together with `logprobs` must have it
    // APPLIED, not rejected. `chat_completions_non_stream` swaps this
    // exact `SamplingOverrides::apply` overlay onto the engine's sampler
    // for the duration of the `generate_with_logprobs` call (see the
    // handler); this asserts the overlay itself carries the client's
    // explicit values through rather than silently keeping the ambient
    // ones, which is the value the handler's swap applies to the
    // sampler.
    let mut body = minimal_chat_request();
    body.temperature = 0.0; // explicit greedy request
    body.top_p = Some(0.33);
    let ambient = distinctive_ambient(); // temperature: 0.9, top_p: 0.55

    assert!(
        (body.temperature - ambient.temperature).abs() > f32::EPSILON,
        "the explicit value must actually differ from ambient for this test to mean anything"
    );

    let overrides = SamplingOverrides::from_request(&body);
    let params = overrides.apply(&ambient);
    assert_eq!(
        params.temperature, 0.0,
        "the explicit temperature must win over the ambient value, not be silently dropped"
    );
    assert!(
        (params.top_p - 0.33).abs() < f32::EPSILON,
        "the explicit top_p must win over the ambient value, not be silently dropped"
    );
}

/// End-to-end companion to the two unit tests above: drives a real
/// request through the real `/v1/chat/completions` route (not just the
/// `SamplingOverrides` overlay in isolation) to prove the handler itself
/// no longer 400s an explicit `temperature`/`top_p` override together
/// with `logprobs: true`, and that logprobs are genuinely computed
/// under that override.
#[tokio::test]
async fn logprobs_with_mismatched_temperature_and_top_p_succeeds_end_to_end() {
    let ambient = SamplingParams {
        temperature: 0.9,
        top_p: 0.55,
        ..distinctive_ambient()
    };
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config, ambient, 42);
    let app = create_router(engine, None);

    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "logprobs": true,
        "temperature": 0.0,
        "top_p": 0.33,
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
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    assert_eq!(
        status,
        StatusCode::OK,
        "an explicit temperature/top_p override together with logprobs:true must be \
         applied, not rejected: {json}"
    );
    assert!(
        json["choices"][0]["logprobs"]["content"].is_array(),
        "logprobs must actually be computed and returned under the override: {json}"
    );
}

// ── serve-api-03: base-endpoint tool-call parsing ────────────────────

#[test]
fn base_tool_calls_parsed_from_generated_text() {
    let text = r#"sure<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#;
    let calls = parse_base_tool_calls(text, true).expect("tool call should be parsed");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(calls[0].r#type, "function");
    assert!(calls[0].id.starts_with("call_"));
}

#[test]
fn base_tool_calls_none_when_inactive() {
    // Same text, but tool calling disabled (no tools / tool_choice: none)
    // must not fabricate a tool call.
    let text = r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#;
    assert!(parse_base_tool_calls(text, false).is_none());
}

#[test]
fn base_tool_calls_none_for_plain_text() {
    assert!(parse_base_tool_calls("just a normal answer", true).is_none());
}

/// B2-13/RT-11: the base endpoint must parse Bonsai 2's real XML tool-call
/// shape, not just the legacy JSON payload.
#[test]
fn base_tool_calls_parsed_from_xml_shape() {
    let text = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>\nParis\n</parameter>\n",
        "</function>\n</tool_call>",
    );
    let calls = parse_base_tool_calls(text, true).expect("XML tool call should be parsed");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("valid JSON arguments");
    assert_eq!(args["city"], "Paris");
}

/// A truncated `<tool_call>` (opened but never closed) must not panic
/// and must not fabricate a call — the raw text is reported as ordinary
/// content instead (RT-11).
#[test]
fn base_tool_calls_truncated_xml_yields_none_not_panic() {
    let text = "<tool_call>\n<function=get_weather>\nstill going";
    assert!(parse_base_tool_calls(text, true).is_none());
}

// ── serve-api-01 / serve-api-02: streaming terminal event ────────────

#[test]
fn stream_terminal_reports_length_when_truncated() {
    let json = stream_terminal_json(Ok(8), 8, "id", 1, "m", false);
    assert!(json.contains("\"finish_reason\":\"length\""), "got: {json}");
    assert!(!json.contains("\"error\""));
    assert!(
        !json.contains("usage"),
        "usage must be omitted unless the client asked for it: {json}"
    );
}

#[test]
fn stream_terminal_reports_stop_when_natural() {
    let json = stream_terminal_json(Ok(3), 8, "id", 1, "m", false);
    assert!(json.contains("\"finish_reason\":\"stop\""), "got: {json}");
}

#[test]
fn stream_terminal_surfaces_error() {
    // A mid-stream failure must produce an error object, never a bogus
    // clean finish chunk (finding serve-api-02).
    let json = stream_terminal_json(
        Err("forward pass failed".to_string()),
        8,
        "id",
        1,
        "m",
        false,
    );
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(v["error"]["message"], "forward pass failed");
    assert_eq!(v["error"]["type"], "server_error");
    assert!(
        !json.contains("finish_reason"),
        "error event must not masquerade as a normal finish: {json}"
    );
}

// ── sec-08: stream_options.include_usage ─────────────────────────────

#[test]
fn include_usage_adds_a_final_usage_chunk() {
    let payloads = stream_terminal_payloads(Ok(3), 8, "id", 1, "m", true, 11);
    assert_eq!(
        payloads.len(),
        2,
        "finish chunk + usage chunk: {payloads:?}"
    );

    let finish: serde_json::Value =
        serde_json::from_str(&payloads[0]).expect("finish chunk is JSON");
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert!(
        finish["usage"].is_null(),
        "non-final chunks carry an explicit null usage: {}",
        payloads[0]
    );

    let usage: serde_json::Value = serde_json::from_str(&payloads[1]).expect("usage chunk is JSON");
    assert_eq!(usage["object"], "chat.completion.chunk");
    assert_eq!(usage["choices"].as_array().map(Vec::len), Some(0));
    assert_eq!(usage["usage"]["prompt_tokens"], 11);
    assert_eq!(usage["usage"]["completion_tokens"], 3);
    assert_eq!(usage["usage"]["total_tokens"], 14);
}

#[test]
fn usage_chunk_is_omitted_without_stream_options() {
    let payloads = stream_terminal_payloads(Ok(3), 8, "id", 1, "m", false, 11);
    assert_eq!(payloads.len(), 1);
    assert!(!payloads[0].contains("usage"));
}

#[test]
fn failed_generation_emits_no_usage_chunk() {
    // A usage chunk after an error event would imply a clean completion.
    let payloads = stream_terminal_payloads(Err("boom".to_string()), 8, "id", 1, "m", true, 11);
    assert_eq!(payloads.len(), 1);
    let v: serde_json::Value = serde_json::from_str(&payloads[0]).expect("valid JSON");
    assert_eq!(v["error"]["message"], "boom");
}

#[test]
fn usage_placeholder_shape() {
    assert_eq!(usage_placeholder(true), Some(serde_json::Value::Null));
    assert_eq!(usage_placeholder(false), None);
}

#[test]
fn rand_id_is_nonempty() {
    let id = rand_id();
    assert!(!id.is_empty());
}

// ── RT-23 / min_p / stop-sequence unit coverage ──────────────────────

#[test]
fn find_first_stop_match_picks_earliest_and_skips_empty() {
    let stops = vec!["".to_string(), "STOP".to_string(), "END".to_string()];
    assert_eq!(
        find_first_stop_match("hello END world STOP", &stops),
        Some((6, "END"))
    );
    assert_eq!(find_first_stop_match("nothing here", &stops), None);
    assert_eq!(find_first_stop_match("", &stops), None);
}

#[test]
fn stop_tracker_releases_up_to_the_match_and_then_nothing_more() {
    let mut tracker = StopTracker::default();
    let stops = vec!["STOP".to_string()];
    // "hello ST" then "OP!" — the match ("STOP") spans the chunk
    // boundary: chunk 1 contributes "ST", chunk 2 completes it with
    // "OP". `max_stop_len = 4` ("STOP"), so chunk 1 must hold back its
    // last 3 bytes (" ST") — releasing only "hello" — precisely so that
    // completing match is still caught here rather than leaking "ST"
    // out through chunk 1 the way `RT-06`'s bug class would.
    let r1 = tracker.push_and_release("hello ST", &stops, 4);
    assert!(!tracker.stopped, "no match yet after chunk 1 alone");
    assert_eq!(
        r1, "hello",
        "chunk 1 must hold back the last 3 (max_stop_len-1) bytes"
    );
    assert_eq!(
        &tracker.accumulated[tracker.emitted_len..],
        " ST",
        "the held-back tail must still contain the in-progress partial match"
    );

    let r2 = tracker.push_and_release("OP!", &stops, 4);
    assert!(tracker.stopped, "STOP must be detected once OP arrives");
    assert_eq!(
        format!("{r1}{r2}"),
        "hello ",
        "must never emit the stop sequence itself, only what preceded it"
    );

    // Nothing more is ever released after a match, even if more text
    // (or a natural end-of-stream flush) arrives.
    assert_eq!(tracker.push_and_release(" more", &stops, 4), "");
    assert_eq!(tracker.take_remaining(), "");
}

#[test]
fn stop_tracker_take_remaining_flushes_the_holdback_on_natural_end() {
    let mut tracker = StopTracker::default();
    let stops = vec!["STOP".to_string()];
    let released = tracker.push_and_release("hello world", &stops, 4);
    // "hello world" is 11 bytes; holdback is 3, so "hello wo" (8 bytes)
    // should already be released and "rld" held back.
    assert_eq!(released, "hello wo");
    assert!(!tracker.stopped);
    assert_eq!(
        tracker.take_remaining(),
        "rld",
        "natural end must flush whatever the holdback window is still sitting on"
    );
    // Idempotent: nothing left to flush a second time.
    assert_eq!(tracker.take_remaining(), "");
}

#[test]
fn stop_tracker_empty_stop_list_never_holds_anything_back() {
    let mut tracker = StopTracker::default();
    let released = tracker.push_and_release("hello", &[], 0);
    assert_eq!(released, "hello");
    assert_eq!(tracker.take_remaining(), "");
}

// ── TOK-M1: logprob `bytes` must be the raw vocabulary bytes ─────────

/// A byte-level vocab whose ids 9/10 are the GPT-2 byte-level spellings
/// of `0xC3`/`0xA9` -- the two raw bytes of `é` -- exactly like
/// `tokenizer_bridge`'s own `TINY_TOKENIZER_JSON` fixture. Individually,
/// neither id's `piece()` is valid UTF-8 on its own, so
/// `String::from_utf8_lossy` renders each as `U+FFFD`: the shipping
/// TOK-M1 defect, reproduced in miniature.
const BYTE_FRAGMENT_TOKENIZER_JSON: &str = r##"{
    "model": {
        "type": "BPE",
        "vocab": {
            "!": 0, "\"": 1, "#": 2, "$": 3,
            "H": 4, "i": 5, "Ġ": 6, "a": 7, "b": 8,
            "Ã": 9, "©": 10, "Hi": 11
        },
        "merges": ["H i"]
    },
    "added_tokens": [
        { "id": 12, "content": "<|im_start|>", "special": true },
        { "id": 13, "content": "<tool_call>", "special": false }
    ],
    "pre_tokenizer": { "type": "ByteLevel" },
    "decoder": { "type": "ByteLevel" }
}"##;

#[test]
fn correct_logprob_bytes_recovers_the_real_byte_fragment_not_replacement_char_bytes() {
    let tok = TokenizerBridge::native_from_json_str(BYTE_FRAGMENT_TOKENIZER_JSON)
        .expect("byte-fragment tokenizer fixture should load");

    // What `api_types::compute_logprobs`/`token_bytes` produces today
    // for a byte-fragment token: the display string is already
    // `U+FFFD` (lossy), so `bytes` is `U+FFFD`'s own 3-byte UTF-8
    // encoding rather than the token's real single raw byte. TOK-M1
    // closed in full: an ALTERNATIVE (`top_logprobs`) needs exactly
    // the same fix-up as the chosen entry, not just the chosen one.
    let corrupted = vec![
        crate::api_types::LogprobsContent {
            id: 9,
            token: "\u{FFFD}".to_string(),
            logprob: -0.1,
            bytes: Some("\u{FFFD}".as_bytes().to_vec()),
            top_logprobs: vec![crate::api_types::TopLogprob {
                id: 10,
                token: "\u{FFFD}".to_string(),
                logprob: -1.5,
                bytes: Some("\u{FFFD}".as_bytes().to_vec()),
            }],
        },
        crate::api_types::LogprobsContent {
            id: 10,
            token: "\u{FFFD}".to_string(),
            logprob: -0.2,
            bytes: Some("\u{FFFD}".as_bytes().to_vec()),
            top_logprobs: vec![],
        },
    ];

    let fixed = correct_logprob_bytes(corrupted, Some(&tok));

    assert_eq!(
        fixed[0].bytes,
        Some(vec![0xC3]),
        "id 9's real vocabulary byte (0xC3, half of 'é') must replace U+FFFD's encoding"
    );
    assert_eq!(
        fixed[0].top_logprobs[0].bytes,
        Some(vec![0xA9]),
        "an alternative's id 10 must be fixed up too, not just the chosen entry"
    );
    assert_eq!(
        fixed[1].bytes,
        Some(vec![0xA9]),
        "id 10's real vocabulary byte (0xA9, half of 'é') must replace U+FFFD's encoding"
    );
    // The (already-correct, per the module docs' documented residual)
    // display string is untouched -- only `bytes` is corrected.
    assert_eq!(fixed[0].token, "\u{FFFD}");
    assert_eq!(fixed[1].token, "\u{FFFD}");
}

#[test]
fn correct_logprob_bytes_recovers_a_whole_valid_token_unchanged() {
    // A token whose piece is already complete, valid UTF-8 (the
    // overwhelming majority) must come back byte-identical, not just
    // "close": `id 4` = "H", `id 5` = "i".
    let tok = TokenizerBridge::native_from_json_str(BYTE_FRAGMENT_TOKENIZER_JSON)
        .expect("byte-fragment tokenizer fixture should load");
    let content = vec![crate::api_types::LogprobsContent {
        id: 4,
        token: "H".to_string(),
        logprob: -0.01,
        bytes: Some(b"H".to_vec()),
        top_logprobs: vec![],
    }];
    let fixed = correct_logprob_bytes(content, Some(&tok));
    assert_eq!(fixed[0].bytes, Some(b"H".to_vec()));
}

#[test]
fn correct_logprob_bytes_is_a_no_op_without_a_tokenizer() {
    let content = vec![crate::api_types::LogprobsContent {
        id: 9,
        token: "<9>".to_string(),
        logprob: -0.1,
        bytes: None,
        top_logprobs: vec![],
    }];
    let fixed = correct_logprob_bytes(content.clone(), None);
    assert_eq!(fixed[0].bytes, content[0].bytes);
    assert_eq!(fixed[0].token, content[0].token);
}

// ── SV-08: StreamLifecycleGuard's RAII metrics on drop/disconnect ────

/// Simulates a client that disconnects before ever polling the SSE
/// body: the guard must still decrement the gauge, record a
/// latency-histogram sample at the *true* elapsed time (not the
/// near-zero an SSE-head-time observation would give), and record a
/// workload-rate sample -- purely from being dropped.
#[test]
fn stream_lifecycle_guard_records_metrics_on_drop_even_without_being_polled() {
    let metrics = Arc::new(InferenceMetrics::new());
    metrics.active_requests.inc();
    let active_guard = ActiveRequestGuard(Arc::clone(&metrics));

    let mut tracker = crate::request_metrics::RequestRateTracker::new();
    tracker.record_first_token();
    let tracker = Arc::new(std::sync::Mutex::new(tracker));
    let rate_aggregator = Arc::new(crate::request_metrics::RequestRateAggregator::new());

    let request_start = std::time::Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(5));

    let guard = StreamLifecycleGuard {
        inner: (),
        rate_aggregator: Arc::clone(&rate_aggregator),
        tracker,
        request_start,
        metrics: Arc::clone(&metrics),
        _active_guard: active_guard,
    };
    // Simulate an early client disconnect: drop without ever polling.
    drop(guard);

    assert_eq!(
        metrics.active_requests.get(),
        0.0,
        "the in-flight gauge must be decremented on drop, not left dangling"
    );
    assert_eq!(
        metrics.request_duration_seconds.count(),
        1,
        "the latency histogram must get exactly one sample from Drop"
    );
    assert!(
        metrics.request_duration_seconds.sum() >= 0.005,
        "the recorded latency must reflect the real elapsed time, not near-zero \
         (observing at SSE-head construction under-reports this to ~0)"
    );
    assert_eq!(
        rate_aggregator.snapshot().completed_requests,
        1,
        "a rate sample must be recorded because at least one token was emitted"
    );
}

#[test]
fn stream_lifecycle_guard_skips_the_rate_sample_when_no_token_was_emitted() {
    // A prefill error (or a disconnect before the first token) must not
    // feed a spurious zero-elapsed sample into `/admin/workload-stats`,
    // even though the histogram and gauge are still recorded.
    let metrics = Arc::new(InferenceMetrics::new());
    metrics.active_requests.inc();
    let active_guard = ActiveRequestGuard(Arc::clone(&metrics));
    let tracker = Arc::new(std::sync::Mutex::new(
        crate::request_metrics::RequestRateTracker::new(),
    ));
    let rate_aggregator = Arc::new(crate::request_metrics::RequestRateAggregator::new());

    let guard = StreamLifecycleGuard {
        inner: (),
        rate_aggregator: Arc::clone(&rate_aggregator),
        tracker,
        request_start: std::time::Instant::now(),
        metrics: Arc::clone(&metrics),
        _active_guard: active_guard,
    };
    drop(guard);

    assert_eq!(metrics.active_requests.get(), 0.0);
    assert_eq!(
        metrics.request_duration_seconds.count(),
        1,
        "the latency histogram is recorded regardless of whether any token was emitted"
    );
    assert_eq!(
        rate_aggregator.snapshot().completed_requests,
        0,
        "no rate sample should be recorded when zero tokens were emitted"
    );
}
