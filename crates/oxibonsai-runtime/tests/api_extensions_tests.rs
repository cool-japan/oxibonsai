//! Integration tests for the extended OpenAI-compatible API.
//!
//! Covers api_types helpers, api_extensions utilities, and the HTTP endpoint.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::collections::HashMap;
use tower::ServiceExt;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::api_extensions::{apply_frequency_penalty, JsonModeEnforcer, StopChecker};
use oxibonsai_runtime::api_types::{
    compute_logprobs, generate_tool_call_id, is_valid_json, parse_tool_call, StopSequences, Tool,
    ToolChoice,
};
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::create_router;

// ── Helper ────────────────────────────────────────────────────────────────────

fn test_router() -> axum::Router {
    let config = Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    create_router(engine, None)
}

// ── api_types deserialization ─────────────────────────────────────────────────

#[test]
fn test_tool_definition_deserialize() {
    let json = serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    });

    let tool: Tool = serde_json::from_value(json).expect("should deserialize Tool");
    assert_eq!(tool.tool_type, "function");
    assert_eq!(tool.function.name, "get_weather");
    assert!(tool.function.description.is_some());
    assert!(tool.function.parameters.is_some());
}

#[test]
fn test_tool_definition_minimal_deserialize() {
    let json = serde_json::json!({
        "type": "function",
        "function": {
            "name": "my_fn"
        }
    });
    let tool: Tool = serde_json::from_value(json).expect("should deserialize minimal Tool");
    assert_eq!(tool.function.name, "my_fn");
    assert!(tool.function.description.is_none());
    assert!(tool.function.parameters.is_none());
}

#[test]
fn test_tool_choice_string_deserialize() {
    // "auto"
    let json = serde_json::json!("auto");
    let tc: ToolChoice = serde_json::from_value(json).expect("should deserialize string choice");
    match tc {
        ToolChoice::String(s) => assert_eq!(s, "auto"),
        ToolChoice::Named(_) => panic!("expected String variant"),
    }

    // "none"
    let json2 = serde_json::json!("none");
    let tc2: ToolChoice = serde_json::from_value(json2).expect("should deserialize none");
    match tc2 {
        ToolChoice::String(s) => assert_eq!(s, "none"),
        ToolChoice::Named(_) => panic!("expected String variant"),
    }
}

#[test]
fn test_tool_choice_named_deserialize() {
    let json = serde_json::json!({
        "type": "function",
        "function": {"name": "specific_fn"}
    });
    let tc: ToolChoice = serde_json::from_value(json).expect("should deserialize named choice");
    match tc {
        ToolChoice::Named(n) => {
            assert_eq!(n.tool_type, "function");
            assert_eq!(n.function.name, "specific_fn");
        }
        ToolChoice::String(_) => panic!("expected Named variant"),
    }
}

// ── StopSequences ─────────────────────────────────────────────────────────────

#[test]
fn test_stop_sequences_single() {
    let json = serde_json::json!("STOP");
    let ss: StopSequences = serde_json::from_value(json).expect("single stop");
    assert_eq!(ss.as_slice(), &["STOP"]);
    assert_eq!(ss.into_vec(), vec!["STOP"]);
}

#[test]
fn test_stop_sequences_multiple() {
    let json = serde_json::json!(["END", "STOP", "DONE"]);
    let ss: StopSequences = serde_json::from_value(json).expect("multiple stops");
    assert_eq!(ss.as_slice(), &["END", "STOP", "DONE"]);
    assert_eq!(ss.clone().into_vec(), vec!["END", "STOP", "DONE"]);
    // as_slice returns the same elements as into_vec
    let v = ss.into_vec();
    assert_eq!(v.len(), 3);
}

// ── StopChecker ───────────────────────────────────────────────────────────────

#[test]
fn test_stop_checker_finds_sequence() {
    let checker = StopChecker::new(vec!["STOP".to_string(), "END".to_string()]);
    assert_eq!(checker.check("Hello STOP world"), Some("STOP"));
    assert_eq!(checker.check("Goodbye END now"), Some("END"));
    assert!(checker.check("No match").is_none());
}

#[test]
fn test_stop_checker_truncates_correctly() {
    let checker = StopChecker::new(vec!["<stop>".to_string()]);
    let (truncated, hit) = checker.truncate_at_stop("Hello world<stop>extra text here");
    assert_eq!(truncated, "Hello world");
    assert!(hit, "should have hit the stop sequence");
}

#[test]
fn test_stop_checker_no_match() {
    let checker = StopChecker::new(vec!["NEVER_APPEARS".to_string()]);
    let (truncated, hit) = checker.truncate_at_stop("Hello world, nothing special here");
    assert_eq!(truncated, "Hello world, nothing special here");
    assert!(!hit, "should not hit any stop sequence");
}

#[test]
fn test_stop_checker_empty() {
    let checker = StopChecker::new(vec![]);
    assert!(checker.is_empty());
    let (text, hit) = checker.truncate_at_stop("some text");
    assert_eq!(text, "some text");
    assert!(!hit);
}

#[test]
fn test_stop_checker_picks_earliest() {
    let checker = StopChecker::new(vec!["world".to_string(), "Hello".to_string()]);
    let (truncated, hit) = checker.truncate_at_stop("Hello world");
    // "Hello" appears first
    assert_eq!(truncated, "");
    assert!(hit);
}

// ── JsonModeEnforcer ──────────────────────────────────────────────────────────

#[test]
fn test_json_mode_enforcer_valid_json_passthrough() {
    let enforcer = JsonModeEnforcer::new();
    let json = r#"{"answer": 42, "unit": "meters"}"#;
    assert_eq!(enforcer.enforce(json), json);
}

#[test]
fn test_json_mode_enforcer_valid_array_passthrough() {
    let enforcer = JsonModeEnforcer::new();
    let arr = r#"[1, 2, 3]"#;
    assert_eq!(enforcer.enforce(arr), arr);
}

#[test]
fn test_json_mode_enforcer_extracts_json_substring() {
    let enforcer = JsonModeEnforcer::new();
    let text = r#"Here is the result: {"name": "Alice", "age": 30} and that's it."#;
    let result = enforcer.enforce(text);
    assert!(
        is_valid_json(&result),
        "result should be valid JSON, got: {result}"
    );
    let v: serde_json::Value = serde_json::from_str(&result).expect("parse");
    assert_eq!(v["name"], "Alice");
}

#[test]
fn test_json_mode_enforcer_wraps_invalid() {
    let enforcer = JsonModeEnforcer::new();
    let text = "This is not JSON at all!";
    let result = enforcer.enforce(text);
    assert!(
        is_valid_json(&result),
        "wrapped result should be valid JSON, got: {result}"
    );
    let v: serde_json::Value = serde_json::from_str(&result).expect("parse");
    assert!(
        v.get("response").is_some(),
        "should have 'response' key in fallback object"
    );
}

// ── is_valid_json ─────────────────────────────────────────────────────────────

#[test]
fn test_is_valid_json_object() {
    assert!(is_valid_json(r#"{"key": "value", "n": 42}"#));
    assert!(is_valid_json(r#"{}"#));
}

#[test]
fn test_is_valid_json_array() {
    assert!(is_valid_json(r#"[1, 2, 3]"#));
    assert!(is_valid_json(r#"[]"#));
    assert!(is_valid_json(r#"["a", "b"]"#));
}

#[test]
fn test_is_valid_json_invalid() {
    assert!(!is_valid_json("not json"));
    assert!(!is_valid_json(""));
    assert!(!is_valid_json("{unclosed"));
    assert!(!is_valid_json("plain text"));
}

// ── compute_logprobs ──────────────────────────────────────────────────────────

#[test]
fn test_compute_logprobs_top_tokens() {
    // logits: token 1 has the highest value
    let logits = vec![1.0f32, 5.0, 2.0, 0.5, 1.5];
    let lp = compute_logprobs(&logits, 1, 3, &|id| format!("tok{id}"));

    assert_eq!(lp.token, "tok1", "chosen token should be tok1");
    assert!(
        lp.logprob <= 0.0,
        "log probability must be non-positive, got {}",
        lp.logprob
    );
    assert_eq!(lp.top_logprobs.len(), 3, "should have 3 top logprobs");
    // The highest logit (token 1) should be first in top_logprobs
    assert_eq!(lp.top_logprobs[0].token, "tok1");
    // All top logprobs should be <= 0
    for tlp in &lp.top_logprobs {
        assert!(
            tlp.logprob <= 0.0,
            "top logprob {} should be <= 0",
            tlp.logprob
        );
    }
}

#[test]
fn test_compute_logprobs_empty_logits() {
    let lp = compute_logprobs(&[], 0, 5, &|id| format!("t{id}"));
    assert_eq!(lp.token, "t0");
    assert_eq!(lp.top_logprobs.len(), 0);
}

#[test]
fn test_compute_logprobs_bytes_present() {
    let logits = vec![1.0f32, 2.0];
    let lp = compute_logprobs(&logits, 0, 1, &|id| format!("w{id}"));
    assert!(lp.bytes.is_some(), "non-empty token should have bytes");
    let bytes = lp.bytes.as_ref().expect("bytes");
    assert!(!bytes.is_empty());
}

// ── parse_tool_call ───────────────────────────────────────────────────────────

#[test]
fn test_parse_tool_call_valid() {
    let text = r#"<tool_call>{"name":"get_weather","arguments":{"city":"London"}}</tool_call>"#;
    let tc = parse_tool_call(text, "call_test001").expect("should parse tool call");
    assert_eq!(tc.id, "call_test001");
    assert_eq!(tc.tool_type, "function");
    assert_eq!(tc.function.name, "get_weather");
    // arguments should be valid JSON
    assert!(
        is_valid_json(&tc.function.arguments),
        "arguments should be valid JSON: {}",
        tc.function.arguments
    );
}

#[test]
fn test_parse_tool_call_no_arguments() {
    let text = r#"<tool_call>{"name":"ping"}</tool_call>"#;
    let tc = parse_tool_call(text, "call_ping").expect("should parse");
    assert_eq!(tc.function.name, "ping");
    assert_eq!(tc.function.arguments, "{}");
}

#[test]
fn test_parse_tool_call_invalid() {
    assert!(parse_tool_call("No tool call here", "call_x").is_none());
    assert!(parse_tool_call("<tool_call>bad json</tool_call>", "call_y").is_none());
    assert!(parse_tool_call("", "call_z").is_none());
}

#[test]
fn test_parse_tool_call_with_surrounding_text() {
    let text = r#"I need to call a function. <tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> Done."#;
    let tc = parse_tool_call(text, "call_add").expect("should parse with surrounding text");
    assert_eq!(tc.function.name, "add");
}

// ── generate_tool_call_id ─────────────────────────────────────────────────────

#[test]
fn test_generate_tool_call_id_prefix() {
    let id = generate_tool_call_id();
    assert!(
        id.starts_with("call_"),
        "tool call ID must start with 'call_', got: {id}"
    );
    assert!(
        id.len() > 5,
        "tool call ID should have content after prefix, got: {id}"
    );
}

#[test]
fn test_generate_tool_call_id_uniqueness() {
    // Generate a handful of IDs and verify they're non-empty strings
    let ids: Vec<String> = (0..5).map(|_| generate_tool_call_id()).collect();
    for id in &ids {
        assert!(id.starts_with("call_"), "all IDs should start with call_");
    }
}

// ── apply_frequency_penalty ───────────────────────────────────────────────────

#[test]
fn test_apply_frequency_penalty_reduces_seen() {
    let mut logits = vec![2.0f32, 3.0, 4.0];
    let mut counts = HashMap::new();
    counts.insert(2u32, 3usize); // token 2 seen 3 times
    apply_frequency_penalty(&mut logits, &counts, 0.5, 0.0);
    // token 2: 4.0 - (0.5 * 3) = 4.0 - 1.5 = 2.5
    assert!(
        (logits[2] - 2.5).abs() < 1e-5,
        "expected 2.5, got {}",
        logits[2]
    );
    // others unchanged
    assert!((logits[0] - 2.0).abs() < 1e-5);
    assert!((logits[1] - 3.0).abs() < 1e-5);
}

#[test]
fn test_apply_presence_penalty_reduces_any_seen() {
    let mut logits = vec![1.0f32, 2.0, 3.0];
    let mut counts = HashMap::new();
    counts.insert(0u32, 5usize); // seen 5 times — presence penalty ignores count
    apply_frequency_penalty(&mut logits, &counts, 0.0, 0.5);
    // logit[0] = 1.0 - 0.5 = 0.5
    assert!(
        (logits[0] - 0.5).abs() < 1e-5,
        "expected 0.5, got {}",
        logits[0]
    );
    assert!((logits[1] - 2.0).abs() < 1e-5);
}

#[test]
fn test_apply_frequency_penalty_no_change_when_zero() {
    let mut logits = vec![1.0f32, 2.0, 3.0];
    let mut counts = HashMap::new();
    counts.insert(0u32, 10usize);
    apply_frequency_penalty(&mut logits, &counts, 0.0, 0.0);
    assert!((logits[0] - 1.0).abs() < 1e-5);
}

// ── Extended endpoint HTTP tests ──────────────────────────────────────────────

#[tokio::test]
async fn test_extended_chat_completions_endpoint() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Hello, extended API!"}
        ],
        "max_tokens": 5,
        "temperature": 0.0
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "extended endpoint should return 200"
    );

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse JSON");

    assert!(json["id"].is_string(), "response should have 'id'");
    assert_eq!(json["object"], "chat.completion");
    assert!(json["choices"].is_array(), "should have 'choices'");
    assert!(json["usage"].is_object(), "should have 'usage'");

    let choices = json["choices"].as_array().expect("choices array");
    assert!(!choices.is_empty(), "should have at least one choice");
    assert_eq!(choices[0]["message"]["role"], "assistant");
}

#[tokio::test]
async fn test_extended_chat_response_has_system_fingerprint() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Test fingerprint"}
        ],
        "max_tokens": 3
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse JSON");

    let fp = json["system_fingerprint"].as_str();
    assert!(
        fp.is_some(),
        "system_fingerprint should be present, got: {json}"
    );
    let fp_str = fp.expect("fingerprint string");
    assert!(
        fp_str.starts_with("fp_"),
        "fingerprint should start with 'fp_', got: {fp_str}"
    );
}

#[tokio::test]
async fn test_extended_endpoint_with_n_completions() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Count to three"}
        ],
        "max_tokens": 4,
        "n": 2
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse JSON");

    let choices = json["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 2, "n=2 should produce 2 choices");
    assert_eq!(choices[0]["index"], 0);
    assert_eq!(choices[1]["index"], 1);
}

/// Regression test for finding 30: real per-token logprobs are not available
/// at this layer (`generate_with_seed` does not expose per-step logits), so
/// the handler must honestly report `logprobs: null` — mirroring
/// `completions.rs` — instead of fabricating a non-null-but-empty
/// `{"content": []}` that looks like a successful-but-token-less response.
#[tokio::test]
async fn test_extended_endpoint_with_logprobs() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 2,
        "logprobs": true,
        "top_logprobs": 3
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse JSON");

    let choices = json["choices"].as_array().expect("choices");
    assert!(!choices.is_empty());
    // Wave-2: real per-token logprobs are now captured via the engine's
    // logits-capturing variant and returned in the OpenAI shape
    // (`logprobs.content` is an array of {token, logprob, top_logprobs}).
    let content = choices[0]["logprobs"]["content"]
        .as_array()
        .expect("logprobs.content must be an array");
    for entry in content {
        assert!(entry["token"].is_string(), "each entry has a token string");
        assert!(
            entry["logprob"].is_number(),
            "each entry has a numeric logprob"
        );
        assert!(
            entry["top_logprobs"].is_array(),
            "each entry has a top_logprobs array"
        );
    }
}

#[tokio::test]
async fn test_extended_endpoint_with_stop_sequences() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Generate text"}
        ],
        "max_tokens": 10,
        "stop": ["DONE"]
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(resp.status(), StatusCode::OK);
}

// ── Regression: real loaded-model id (finding serve-api-04) ──────────────────

/// The response `model` field and the `system_fingerprint` must reflect the
/// actually-loaded model (`Qwen3Config::tiny_test()` names it
/// `"Bonsai-Tiny-Test"`), not the hardcoded `"bonsai-8b"` literal.
#[tokio::test]
async fn test_extended_endpoint_reports_real_model_id() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "What model are you?"}
        ],
        "max_tokens": 3
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse JSON");

    assert_eq!(
        json["model"], "Bonsai-Tiny-Test",
        "model field must reflect the real loaded model, not a hardcoded literal, got: {json}"
    );
    assert_ne!(
        json["model"], "bonsai-8b",
        "model field must not be the old hardcoded literal"
    );
}

// ── Regression: stream: true (finding serve-api-05) ───────────────────────────

/// `stream: true` with no tools/n/json-mode must produce a real SSE response
/// (not a silently-ignored buffered JSON one): `text/event-stream`
/// content-type, `data:` chunks, and a terminal `[DONE]` marker.
#[tokio::test]
async fn test_extended_endpoint_stream_returns_sse() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Stream this"}
        ],
        "max_tokens": 5,
        "stream": true
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(resp.status(), StatusCode::OK);

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("text/event-stream"),
        "stream:true must respond with SSE, got content-type: {content_type}"
    );

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let body_str = String::from_utf8(bytes.to_vec()).expect("SSE body must be UTF-8");

    assert!(
        body_str.contains("chat.completion.chunk"),
        "SSE body should contain chat.completion.chunk events, got: {body_str}"
    );
    assert!(
        body_str.contains("\"role\":\"assistant\""),
        "first SSE event should carry the assistant role delta, got: {body_str}"
    );
    assert!(
        body_str.trim_end().ends_with("data: [DONE]"),
        "SSE stream must terminate with a [DONE] marker, got: {body_str}"
    );
    assert!(
        body_str.contains("\"finish_reason\":\"stop\"")
            || body_str.contains("\"finish_reason\":\"length\""),
        "SSE stream must carry a real finish_reason, got: {body_str}"
    );
}

/// `stream: true` combined with `tools` must be honestly rejected with `400`
/// (tool-call parsing needs the complete text) rather than silently ignoring
/// either field.
#[tokio::test]
async fn test_extended_endpoint_stream_with_tools_rejected() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Call a tool"}
        ],
        "max_tokens": 5,
        "stream": true,
        "tools": [
            {"type": "function", "function": {"name": "get_weather"}}
        ]
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "stream:true + tools must be rejected honestly, not silently ignored"
    );
}

/// `stream: true` combined with `n > 1` must be honestly rejected with `400`.
#[tokio::test]
async fn test_extended_endpoint_stream_with_n_rejected() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Multiple please"}
        ],
        "max_tokens": 5,
        "stream": true,
        "n": 2
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// `stream: true` combined with a JSON-mode `response_format` must be
/// honestly rejected with `400` (JSON extraction/wrapping needs the complete
/// text).
#[tokio::test]
async fn test_extended_endpoint_stream_with_json_mode_rejected() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Return JSON"}
        ],
        "max_tokens": 5,
        "stream": true,
        "response_format": {"type": "json_object"}
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_extended_endpoint_json_mode() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Return JSON"}
        ],
        "max_tokens": 10,
        "response_format": {"type": "json_object"}
    });

    let req = Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("build request");

    let resp = app.oneshot(req).await.expect("send request");
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse JSON response");

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content should be string");
    assert!(
        is_valid_json(content),
        "with json_object mode, content should be valid JSON: {content}"
    );
}
