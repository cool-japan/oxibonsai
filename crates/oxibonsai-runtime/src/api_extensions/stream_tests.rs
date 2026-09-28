//! `stream: true` on `POST /v1/chat/completions/extended`: tool calls over
//! SSE and `stream_options.include_usage`, over the real HTTP route.
//!
//! Attached to `api_extensions/stream.rs` as its `#[cfg(test)] mod tests`
//! via `#[path]`. A weightless engine scripted to emit an exact id sequence
//! over the byte-level vocabulary whose `<tool_call>`/`</tool_call>` are
//! single added tokens drives each test; every scripted response is also
//! requested without streaming, and the two must agree on content, tool
//! calls and `finish_reason`.

use super::*;
use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

/// A router whose engine emits exactly `script` (then EOS) on every
/// generation, over the marker-carrying vocabulary.
fn scripted_router(script: &[u32]) -> axum::Router {
    let mut engine = fx::weightless_engine(fx::MARKER_VOCAB, SamplingParams::default(), 42);
    engine.script_generation(script.to_vec());
    crate::server::create_router(engine, Some(fx::byte_tokenizer_with_markers()))
}

/// An extended request advertising the `get_weather` tool.
fn tools_request(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "messages": [{"role": "user", "content": "weather in Tokyo?"}],
        "max_tokens": 256,
        "stream": stream,
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }}],
    })
}

/// The ids of an XML `get_weather(city)` call opened and closed by the
/// vocabulary's own marker tokens.
fn xml_call(city: &str) -> Vec<u32> {
    let mut ids = vec![fx::TOOL_CALL_OPEN];
    ids.extend(fx::byte_ids(&format!(
        "\n<function=get_weather>\n<parameter=city>\n{city}\n</parameter>\n</function>\n"
    )));
    ids.push(fx::TOOL_CALL_CLOSE);
    ids
}

/// Every `delta.tool_calls` entry of an SSE body, in order.
fn streamed_calls(body: &str) -> Vec<serde_json::Value> {
    fx::sse_payloads(body)
        .iter()
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["tool_calls"]
                .as_array()
                .cloned()
        })
        .flatten()
        .collect()
}

/// Every non-null `finish_reason` of an SSE body.
fn finish_reasons(body: &str) -> Vec<String> {
    fx::sse_payloads(body)
        .iter()
        .filter_map(|chunk| {
            chunk["choices"][0]["finish_reason"]
                .as_str()
                .map(str::to_string)
        })
        .collect()
}

/// Stream `script` and request it without streaming; assert they agree and
/// return the SSE body and the JSON response.
async fn stream_and_json(script: &[u32]) -> (String, serde_json::Value) {
    let path = "/v1/chat/completions/extended";
    let (status, _, body) = fx::post(scripted_router(script), path, tools_request(true)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
    let (status, _, text) = fx::post(scripted_router(script), path, tools_request(false)).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let json: serde_json::Value = serde_json::from_str(&text).expect("a chat.completion");
    let choice = &json["choices"][0];

    assert_eq!(
        finish_reasons(&body),
        vec![choice["finish_reason"]
            .as_str()
            .unwrap_or_default()
            .to_string()],
        "{body} vs {json}"
    );
    assert_eq!(
        fx::delta_texts(&body, "content").concat(),
        choice["message"]["content"].as_str().unwrap_or_default(),
        "{body} vs {json}"
    );
    let json_calls = choice["tool_calls"].as_array().cloned().unwrap_or_default();
    let calls = streamed_calls(&body);
    assert_eq!(calls.len(), json_calls.len(), "{body} vs {json}");
    for (index, (streamed, collected)) in calls.iter().zip(&json_calls).enumerate() {
        assert_eq!(streamed["index"], index, "{body}");
        assert_eq!(streamed["type"], "function", "{body}");
        assert_eq!(
            streamed["function"], collected["function"],
            "{body} vs {json}"
        );
    }
    (body, json)
}

#[tokio::test]
async fn stream_tools_on_extended_one_xml_call_is_one_tool_calls_delta() {
    let (body, json) = stream_and_json(&xml_call("Tokyo")).await;
    let calls = streamed_calls(&body);
    assert_eq!(calls.len(), 1, "{body}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Tokyo"}"#);
    assert!(
        calls[0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("call_")),
        "{body}"
    );
    assert_eq!(finish_reasons(&body), ["tool_calls"]);
    assert!(fx::delta_texts(&body, "content").is_empty(), "{body}");
    assert!(json["choices"][0]["message"]["content"].is_null(), "{json}");
}

#[tokio::test]
async fn stream_tools_on_extended_prose_then_two_calls() {
    let mut script = fx::byte_ids("Checking both. ");
    script.extend(xml_call("Tokyo"));
    script.extend(fx::byte_ids("\n"));
    script.extend(xml_call("Paris"));
    let (body, json) = stream_and_json(&script).await;
    assert_eq!(fx::delta_texts(&body, "content").concat(), "Checking both.");
    let calls = streamed_calls(&body);
    assert_eq!(calls.len(), 2, "{body}");
    assert_eq!(calls[1]["index"], 1);
    assert_eq!(calls[1]["function"]["arguments"], r#"{"city":"Paris"}"#);
    assert_eq!(finish_reasons(&body), ["tool_calls"]);
    assert_eq!(json["choices"][0]["message"]["content"], "Checking both.");
}

#[tokio::test]
async fn stream_tools_on_extended_json_form_call_is_parsed() {
    let mut script = vec![fx::TOOL_CALL_OPEN];
    script.extend(fx::byte_ids(
        "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n",
    ));
    script.push(fx::TOOL_CALL_CLOSE);
    let (body, _) = stream_and_json(&script).await;
    let calls = streamed_calls(&body);
    assert_eq!(calls.len(), 1, "{body}");
    assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Tokyo"}"#);
}

#[tokio::test]
async fn stream_tools_on_extended_unclosed_call_is_flushed_as_content() {
    let mut script = fx::byte_ids("Checking. ");
    script.push(fx::TOOL_CALL_OPEN);
    script.extend(fx::byte_ids(
        "\n<function=get_weather>\n<parameter=city>\nTok",
    ));
    let (body, _) = stream_and_json(&script).await;
    assert!(streamed_calls(&body).is_empty(), "{body}");
    assert_eq!(
        fx::delta_texts(&body, "content").concat(),
        "Checking. <tool_call>\n<function=get_weather>\n<parameter=city>\nTok"
    );
    assert_eq!(finish_reasons(&body), ["stop"]);
}

#[tokio::test]
async fn stream_tools_on_extended_a_tag_spelled_from_ordinary_tokens_does_not_trigger() {
    let text = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>";
    let (body, _) = stream_and_json(&fx::byte_ids(text)).await;
    assert!(streamed_calls(&body).is_empty(), "{body}");
    assert_eq!(fx::delta_texts(&body, "content").concat(), text);
    assert_eq!(finish_reasons(&body), ["stop"]);
}

#[tokio::test]
async fn stream_tools_on_extended_with_n_above_one_is_refused_naming_n() {
    let mut request = tools_request(true);
    request["n"] = serde_json::json!(2);
    let (status, _, body) = fx::post(
        scripted_router(&xml_call("Tokyo")),
        "/v1/chat/completions/extended",
        request,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("a JSON error");
    assert_eq!(json["error"]["param"], "n", "{json}");
}

// ── stream_options.include_usage ──────────────────────────────────────

/// With `stream_options.include_usage`, every chunk carries `"usage": null`
/// and one final usage chunk — empty `choices`, the real token counts —
/// follows the finish chunk, exactly as on the base endpoint.
#[tokio::test]
async fn extended_stream_include_usage_yields_a_final_usage_chunk() {
    let script = fx::byte_ids("Hello there.");
    let (status, _, body) = fx::post(
        scripted_router(&script),
        "/v1/chat/completions/extended",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "stream": true,
            "stream_options": {"include_usage": true},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let payloads = fx::sse_payloads(&body);
    let (last, rest) = payloads.split_last().expect("chunks");
    assert!(
        rest.iter().all(|chunk| chunk["usage"].is_null()
            && chunk.as_object().is_some_and(|o| o.contains_key("usage"))),
        "every chunk before the usage chunk carries \"usage\": null: {body}"
    );
    assert_eq!(last["choices"].as_array().map(Vec::len), Some(0), "{body}");
    assert_eq!(last["usage"]["completion_tokens"], script.len(), "{body}");
    let prompt_tokens = last["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    assert!(prompt_tokens > 0, "{body}");
    assert_eq!(
        last["usage"]["total_tokens"].as_u64(),
        Some(prompt_tokens + script.len() as u64),
        "{body}"
    );
    assert_eq!(fx::delta_texts(&body, "content").concat(), "Hello there.");
}

/// Without `stream_options.include_usage` no chunk mentions `usage`.
#[tokio::test]
async fn extended_stream_without_include_usage_sends_no_usage() {
    let (status, _, body) = fx::post(
        scripted_router(&fx::byte_ids("Hi.")),
        "/v1/chat/completions/extended",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(!body.contains("usage"), "{body}");
}
