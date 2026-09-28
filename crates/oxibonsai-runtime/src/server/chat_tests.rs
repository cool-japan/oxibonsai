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
    let (leading, calls) = match parse_base_tool_calls(text, true) {
        BaseToolCallOutcome::Found {
            leading_text,
            calls,
        } => (leading_text, calls),
        other => panic!("expected Found, got {other:?}"),
    };
    assert_eq!(
        leading, "sure",
        "B6: the natural-language preamble must be kept"
    );
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
    assert_eq!(
        parse_base_tool_calls(text, false),
        BaseToolCallOutcome::None
    );
}

#[test]
fn base_tool_calls_none_for_plain_text() {
    assert_eq!(
        parse_base_tool_calls("just a normal answer", true),
        BaseToolCallOutcome::None
    );
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
    let calls = match parse_base_tool_calls(text, true) {
        BaseToolCallOutcome::Found { calls, .. } => calls,
        other => panic!("expected Found, got {other:?}"),
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("valid JSON arguments");
    assert_eq!(args["city"], "Paris");
}

/// A truncated `<tool_call>` (opened but never closed) must not panic
/// and must not fabricate a call — the minor finding's fix: the partial
/// XML itself is never reported as content, only whatever real leading
/// text preceded it (here, none).
#[test]
fn base_tool_calls_truncated_xml_yields_truncated_not_panic() {
    let text = "<tool_call>\n<function=get_weather>\nstill going";
    assert_eq!(
        parse_base_tool_calls(text, true),
        BaseToolCallOutcome::Truncated {
            leading_text: String::new()
        }
    );
}

/// The minor finding's other half: a truncated call WITH a real preamble
/// keeps that preamble.
#[test]
fn base_tool_calls_truncated_xml_keeps_the_leading_preamble() {
    let text = "Let me check.\n<tool_call>\n<function=get_weather>\nstill going";
    assert_eq!(
        parse_base_tool_calls(text, true),
        BaseToolCallOutcome::Truncated {
            leading_text: "Let me check.\n".to_string()
        }
    );
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

// ── B2 (RT-07): required structural per-role test on the serving path ──
//
// Exercises `to_render_messages` (this file's own conversion from the real
// `server::ChatMessage` wire type) + `chat_render::render_chat_prompt`
// together — the actual base-endpoint code path, not just the
// `RenderMessage`-level tests `chat_templates.rs` already has (which cover
// only the fallback template and never exercise `tool_call_id`).

fn tokenizer_with_tool_and_think_tokens() -> TokenizerBridge {
    const TOKENIZER_JSON: &str = r##"{
        "model": {
            "type": "BPE",
            "vocab": {
                "!": 0, "\"": 1, "#": 2, "$": 3,
                "H": 4, "i": 5, "Ġ": 6, "a": 7, "b": 8, "y": 9, "o": 10, "u": 11,
                "s": 12, "e": 13, "m": 14, "t": 15, "1": 16, "c": 17, "2": 18
            },
            "merges": []
        },
        "added_tokens": [
            { "id": 100, "content": "<think>", "special": false },
            { "id": 101, "content": "</think>", "special": false },
            { "id": 102, "content": "<tool_call>", "special": false },
            { "id": 103, "content": "</tool_call>", "special": false }
        ],
        "pre_tokenizer": { "type": "ByteLevel" },
        "decoder": { "type": "ByteLevel" }
    }"##;
    TokenizerBridge::native_from_json_str(TOKENIZER_JSON).expect("fixture must load")
}

#[test]
fn structural_per_role_rendering_on_the_serving_path() {
    let tok = tokenizer_with_tool_and_think_tokens();
    // The real server message shape: system, user, an assistant turn
    // carrying `tool_calls`, a `tool` response WITH `tool_call_id` (the
    // gap finding B2 names explicitly), then a follow-up user turn.
    let messages = vec![
        ChatMessage::text("system", "sys"),
        ChatMessage::text("user", "hi"),
        ChatMessage {
            role: "assistant".to_string(),
            content: Some(String::new()),
            tool_calls: Some(vec![crate::api_types::ToolCallResult::new_function(
                "call_1".to_string(),
                "get_weather".to_string(),
                r#"{"city":"Tokyo"}"#.to_string(),
            )]),
            tool_call_id: None,
        },
        ChatMessage {
            role: "tool".to_string(),
            content: Some("sunny".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
        },
        ChatMessage::text("user", "thanks"),
    ];

    let render_messages = to_render_messages(&messages, &[]);
    assert_eq!(render_messages.len(), 5);
    // `tool_call_id` really does flow all the way from the wire type
    // through to the render layer (B2's own point: no existing test
    // covered this field at all).
    assert_eq!(
        render_messages[3].tool_call_id.as_deref(),
        Some("call_1"),
        "tool_call_id must survive the server::ChatMessage -> RenderMessage conversion"
    );

    let opts = oxibonsai_tokenizer::chat_templates::RenderOptions {
        add_generation_prompt: true,
        ..Default::default()
    };
    let (text, ids) = chat_render::render_chat_prompt(
        &tok,
        &SpecialTokenGuard::from_tokenizer(&tok),
        &render_messages,
        &opts,
        true,
    )
    .expect("must render");
    assert!(!ids.is_empty());

    // system
    assert!(
        text.contains("<|im_start|>system\nsys<|im_end|>\n"),
        "system role must render structurally: {text:?}"
    );
    // user
    assert!(
        text.contains("<|im_start|>user\nhi<|im_end|>\n"),
        "user role must render structurally: {text:?}"
    );
    // assistant tool_calls -> the real XML shape, not the legacy JSON
    // payload, and not silently dropped (RT-07's own complaint).
    assert!(
        text.contains("<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"),
        "assistant tool_calls must render as the XML shape: {text:?}"
    );
    // tool (RT-07's headline bug): must be wrapped in a real
    // `<|im_start|>user` / `<tool_response>` turn, never bare content +
    // `\n` (the OLD `_` arm's mis-rendering).
    assert!(
        text.contains("<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n"),
        "tool role (with tool_call_id) must render as a wrapped user turn, not bare \
         content: {text:?}"
    );
    assert!(
        !text.contains("sunny\n<|im_start|>user\nthanks"),
        "must never be the OLD bare-content-plus-newline mis-rendering: {text:?}"
    );
    // the trailing follow-up user turn
    assert!(
        text.contains("<|im_start|>user\nthanks<|im_end|>\n"),
        "the follow-up user turn must still render: {text:?}"
    );
    // ends with the generation prompt
    assert!(
        text.ends_with("<|im_start|>assistant\n"),
        "must end with the generation prompt: {text:?}"
    );
}

#[test]
fn structural_rendering_errors_honestly_never_renders_garbage() {
    // spec item 1: "an unsupported construct must ERROR" — an unrecognized
    // role reaching the render layer (validate_chat_request already
    // rejects this before generation in the real handler; this pins the
    // render layer's OWN behavior independently) must error, not silently
    // splice unrecognized content into the prompt.
    let tok = tokenizer_with_tool_and_think_tokens();
    let messages = vec![ChatMessage::text("narrator", "once upon a time")];
    let render_messages = to_render_messages(&messages, &[]);
    let err = chat_render::render_chat_prompt(
        &tok,
        &SpecialTokenGuard::from_tokenizer(&tok),
        &render_messages,
        &oxibonsai_tokenizer::chat_templates::RenderOptions {
            add_generation_prompt: true,
            ..Default::default()
        },
        true,
    )
    .expect_err("an unrecognized role must error, never render garbage");
    assert_eq!(err.status(), StatusCode::BAD_REQUEST);
}

// ── B4: tools raw JSON text preserves the client's own key order ───────

#[test]
fn chat_request_extras_tools_raw_json_preserves_key_order() {
    // Deliberately non-alphabetical key order in both the outer object and
    // the nested JSON-schema `properties` map.
    let body = r#"{
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"]}}}]
    }"#;
    let extras: ChatRequestExtras = serde_json::from_str(body).expect("must deserialize");
    let raw = extras.tools_raw_json().expect("tools must be present");

    // The FALLBACK template (`ResolvedChatTemplate::default_fallback`)
    // never references `tools` at all — attach a minimal real-Jinja
    // template that does, mirroring the real template's own `tool |
    // tojson` mechanism, so this test actually exercises the tojson/
    // `Value::from_json_str` seam B4's fix lives in.
    const TOOLS_TEMPLATE: &str = "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}\
         {% if tools %}{% for t in tools %}{{ t | tojson }}{% endfor %}{% endif %}";
    let tok = tokenizer_with_tool_and_think_tokens().with_chat_template(
        oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate::Jinja(std::sync::Arc::new(
            oxibonsai_tokenizer::jinja::JinjaTemplate::compile(TOOLS_TEMPLATE)
                .expect("must compile"),
        )),
    );
    let messages = vec![ChatMessage::text("user", "weather?")];
    let (text, _) = chat_render::render_chat_prompt(
        &tok,
        &SpecialTokenGuard::from_tokenizer(&tok),
        &to_render_messages(&messages, &[]),
        &oxibonsai_tokenizer::chat_templates::RenderOptions {
            add_generation_prompt: true,
            tools: Some(raw),
            ..Default::default()
        },
        true,
    )
    .expect("must render");

    // `city` must render before `days` (the client's own order) inside the
    // rendered `<tools>` JSON block — not re-sorted alphabetically (which
    // would put "days" first).
    let city_pos = text.find("\"city\"").expect("city key present");
    let days_pos = text.find("\"days\"").expect("days key present");
    assert!(
        city_pos < days_pos,
        "tools JSON key order must be preserved (city before days): {text:?}"
    );
    // And `type`/`properties`/`required` themselves must precede in the
    // client's own order too.
    let type_pos = text.find("\"type\": \"object\"").expect("type key present");
    let properties_pos = text.find("\"properties\"").expect("properties key present");
    assert!(type_pos < properties_pos, "got: {text:?}");
}

#[test]
fn chat_request_extras_reads_chat_template_kwargs() {
    let body = r#"{
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}
    }"#;
    let extras: ChatRequestExtras = serde_json::from_str(body).expect("must deserialize");
    assert_eq!(extras.effective_enable_thinking(), Some(false));
    assert_eq!(extras.effective_reasoning_effort().as_deref(), Some("low"));
}

#[test]
fn chat_request_extras_accepts_bare_top_level_fields_too() {
    let body = r#"{
        "messages": [{"role": "user", "content": "hi"}],
        "enable_thinking": false,
        "reasoning_effort": "medium"
    }"#;
    let extras: ChatRequestExtras = serde_json::from_str(body).expect("must deserialize");
    assert_eq!(extras.effective_enable_thinking(), Some(false));
    assert_eq!(
        extras.effective_reasoning_effort().as_deref(),
        Some("medium")
    );
}

#[test]
fn chat_request_extras_defaults_when_absent() {
    let body = r#"{"messages": [{"role": "user", "content": "hi"}]}"#;
    let extras: ChatRequestExtras = serde_json::from_str(body).expect("must deserialize");
    assert_eq!(extras.effective_enable_thinking(), None);
    assert_eq!(extras.effective_reasoning_effort(), None);
    assert!(extras.tools_raw_json().is_none());
}

// ── B1 / B3 end-to-end smoke: real chat-template rendering + reasoning
//    split through the actual HTTP route, with a tokenizer that has
//    <think> ids ────────────────────────────────────────────────────────

fn router_with_think_capable_tokenizer() -> axum::Router {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    create_router(engine, Some(tokenizer_with_tool_and_think_tokens()))
}

/// The base endpoint must render through the real Jinja engine (not the
/// old hardcoded ChatML segment builder) and stay a valid `200` end to
/// end, for a tokenizer whose vocabulary DOES define `<think>`/`</think>`
/// — i.e. `started_in_think` resolves `true` and a
/// `crate::reasoning::ReasoningSplitter` genuinely drives the response,
/// not just a permanent pass-through.
#[tokio::test]
async fn base_endpoint_renders_through_the_real_template_with_a_think_capable_tokenizer() {
    let app = router_with_think_capable_tokenizer();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4
    });
    let req = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).expect("serialize"),
        ))
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("valid JSON");
    assert_eq!(status, StatusCode::OK, "{json}");
    // `reasoning_content`, when present, must be a string (never null-ish
    // garbage) -- it may legitimately be absent if the tiny random model
    // never actually reaches `</think>` within max_tokens.
    if let Some(rc) = json["choices"][0]["message"].get("reasoning_content") {
        assert!(rc.is_string(), "reasoning_content must be a string: {json}");
    }
    assert!(
        json["choices"][0]["message"]["content"].is_string()
            || json["choices"][0]["message"]["content"].is_null(),
        "content must be a string or null (tool_calls case): {json}"
    );
}

/// `chat_template_kwargs.enable_thinking: false` must be accepted (not
/// silently ignored — B3a) and still produce a normal `200` response.
#[tokio::test]
async fn base_endpoint_accepts_chat_template_kwargs_enable_thinking_false() {
    let app = router_with_think_capable_tokenizer();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "chat_template_kwargs": {"enable_thinking": false}
    });
    let req = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).expect("serialize"),
        ))
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
}

/// An unrecognized `chat_template_kwargs`/extras shape must not crash the
/// handler — `ChatRequestExtras`'s best-effort `unwrap_or_default()` is
/// exactly for this: a field with the wrong TYPE (not just an unknown
/// field) still leaves every other request-shape validation
/// (`ChatCompletionRequest` itself) to do its job.
#[tokio::test]
async fn base_endpoint_tolerates_a_malformed_chat_template_kwargs_value() {
    let app = router_with_think_capable_tokenizer();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "chat_template_kwargs": "not an object"
    });
    let req = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).expect("serialize"),
        ))
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    // Must not panic/500 -- a malformed extras shape simply resolves to
    // `ChatRequestExtras::default()` and the request proceeds normally.
    assert_eq!(resp.status(), StatusCode::OK);
}

/// `messages: []` must now be an honest `400` (spec item 1: "an
/// unsupported construct must ERROR") instead of the OLD hardcoded
/// builder's silent "just the generation prompt" behavior — the real
/// template's own `raise_exception('No messages provided.')` surfaces
/// through `chat_render::api_error_from_jinja`.
///
/// (`validate_chat_request` does not itself reject an empty `messages`
/// array today, so this pins the render layer as the actual backstop.)
#[tokio::test]
async fn base_endpoint_empty_messages_is_400_not_a_silent_bare_prompt() {
    let app = router_with_think_capable_tokenizer();
    let body = serde_json::json!({ "messages": [], "max_tokens": 4 });
    let req = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).expect("serialize"),
        ))
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── B11/SV-11 wiring, exercised over the real HTTP route ──────────────
//
// `chat_render.rs`'s own tests pin that `preprocess_message_content_and_reasoning`
// + `to_render_messages` + `render_chat_prompt` together deliver a raw
// body's `reasoning_content` into the actual rendered text; these confirm
// `chat_completions` really does call that pre-pass and thread its result
// through, end to end, over the real route (the response never exposes
// the rendered prompt text itself — a tiny test model's generated output
// does not meaningfully reflect it — so this is a "does not error, wires
// through cleanly" check, not a byte-content one).

#[tokio::test]
async fn base_endpoint_accepts_reasoning_content_on_a_replayed_assistant_turn() {
    let app = router_with_think_capable_tokenizer();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "user greets"},
            {"role": "user", "content": "thanks"}
        ],
        "max_tokens": 4
    });
    let req = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).expect("serialize"),
        ))
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn base_endpoint_flattens_a_text_only_vision_content_array() {
    let app = router_with_think_capable_tokenizer();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": " world"}
            ]}
        ],
        "max_tokens": 4
    });
    let req = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).expect("serialize"),
        ))
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "a text-only content array must be flattened and accepted, not rejected at the schema level"
    );
}

#[tokio::test]
async fn base_endpoint_rejects_an_image_url_content_part_honestly() {
    let app = router_with_think_capable_tokenizer();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}}
            ]}
        ],
        "max_tokens": 4
    });
    let req = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).expect("serialize"),
        ))
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    let message = json["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.to_lowercase().contains("image"),
        "the rejection must name image_url specifically, not an opaque schema error: {json}"
    );
}

// ── cross-endpoint cancellation isolation, exercised over real HTTP ───
//
// `EngineLease::Drop` (`engine_pool.rs`, `SV-09`) already guarantees
// `engine_pool::tests::a_cancelled_request_does_not_poison_the_next_lease`:
// a request's cancellation token is cleared the moment its lease returns to
// the pool, regardless of whether that token was ever actually cancelled.
// `create_router` here is built around exactly one `InferenceEngine`
// replica shared by every route, so two sequential requests against the
// same `app` — even against two *different* endpoints — necessarily reuse
// it. This confirms the pool-level guarantee still holds when the token
// really was cancelled mid-stream (a live stop-sequence match,
// `chat_completions_stream`'s `SV-09` wiring) and the *next* request comes
// through a sibling endpoint that never calls `arm_cancellation()` itself —
// `/v1/completions` (`completions.rs::create_completion`'s non-streaming
// path) — rather than back through chat, whose own handler always re-arms
// a fresh token before it checks cancellation and so would not exercise
// this at all.

async fn post_chat(app: axum::Router, body: serde_json::Value) -> (StatusCode, String) {
    let req = axum::http::Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).expect("serialize"),
        ))
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn post_legacy_completion(
    app: axum::Router,
    body: serde_json::Value,
) -> (StatusCode, String) {
    let req = axum::http::Request::post("/v1/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).expect("serialize"),
        ))
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn a_stream_stopped_by_a_real_cancel_does_not_affect_the_next_request_on_another_endpoint() {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
    let app = create_router(engine, None);

    let (stream_status, body) = post_chat(
        app.clone(),
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "stream": true,
            "stop": "["
        }),
    )
    .await;
    assert_eq!(stream_status, StatusCode::OK);
    assert!(
        body.contains("\"finish_reason\":\"stop\""),
        "sanity: the stop sequence must actually have matched: {body}"
    );

    // The victim: `/v1/completions`, non-streaming, on the SAME replica.
    // `create_completion` never calls `arm_cancellation()` itself, so it
    // has no way to overwrite a poisoned `cancel` left behind by the
    // request above.
    let (status, body) =
        post_legacy_completion(app, serde_json::json!({ "prompt": "hi", "max_tokens": 3 })).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let json: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
    assert!(
        json["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "a /v1/completions request right after a REAL cancellation on this \
         replica (via a *different* endpoint) must still generate tokens, got {json}"
    );
}
