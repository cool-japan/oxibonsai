//! `stream: true` + `tools` on `/v1/chat/completions`, over the real HTTP
//! route.
//!
//! Attached to `chat_stream.rs` as its `#[cfg(test)] mod tests` via
//! `#[path]`. A weightless engine scripted to emit an exact id sequence over
//! the byte-level vocabulary whose `<think>`/`</think>`/`<tool_call>`/
//! `</tool_call>` are single added tokens (the shape of the shipped Qwen3 /
//! Bonsai 2 vocabularies) drives every test, so each one controls the
//! model's output token by token. Every scripted response is also requested
//! without streaming, and the two must agree on content, tool calls and
//! `finish_reason`.

use super::*;
use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

/// A router whose engine emits exactly `script` (then EOS) on every
/// generation, over the marker-carrying vocabulary, optionally under a
/// custom chat template.
fn scripted_router(script: &[u32], template: Option<&str>) -> axum::Router {
    let mut engine = fx::weightless_engine(fx::MARKER_VOCAB, SamplingParams::default(), 42);
    engine.script_generation(script.to_vec());
    let mut tokenizer = fx::byte_tokenizer_with_markers();
    if let Some(source) = template {
        tokenizer = tokenizer.with_chat_template(
            oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate::Jinja(Arc::new(
                oxibonsai_tokenizer::jinja::JinjaTemplate::compile(source)
                    .expect("the test template compiles"),
            )),
        );
    }
    create_router(engine, Some(tokenizer))
}

/// A chat request advertising the `get_weather` tool.
fn tools_request(stream: bool, max_tokens: usize) -> serde_json::Value {
    serde_json::json!({
        "messages": [{"role": "user", "content": "weather in Tokyo?"}],
        "max_tokens": max_tokens,
        "stream": stream,
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }}],
    })
}

/// The ids of an XML `get_weather` call whose parameters are `params`, in
/// order, opened and closed by the vocabulary's own marker tokens.
fn xml_call(params: &[(&str, &str)]) -> Vec<u32> {
    let mut body = String::from("\n<function=get_weather>\n");
    for (key, value) in params {
        body.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
    }
    body.push_str("</function>\n");
    let mut ids = vec![fx::TOOL_CALL_OPEN];
    ids.extend(fx::byte_ids(&body));
    ids.push(fx::TOOL_CALL_CLOSE);
    ids
}

/// One streamed chat response, decoded.
#[derive(Debug, Default)]
struct Streamed {
    /// `(channel, payload)` of every delta after the role delta, in order:
    /// `("reasoning", text)`, `("content", text)` or `("tool_call", json)`.
    events: Vec<(&'static str, serde_json::Value)>,
    /// Every non-null `finish_reason`.
    finish_reasons: Vec<String>,
    /// The usage chunk's `usage`, if one was sent.
    usage: Option<serde_json::Value>,
}

impl Streamed {
    fn parse(body: &str) -> Self {
        let mut out = Streamed::default();
        for chunk in fx::sse_payloads(body) {
            if chunk["usage"].is_object() {
                out.usage = Some(chunk["usage"].clone());
            }
            let Some(choice) = chunk["choices"].get(0) else {
                continue;
            };
            let delta = &choice["delta"];
            if let Some(text) = delta["reasoning_content"].as_str() {
                out.events.push(("reasoning", serde_json::json!(text)));
            }
            if let Some(text) = delta["content"].as_str() {
                out.events.push(("content", serde_json::json!(text)));
            }
            if let Some(calls) = delta["tool_calls"].as_array() {
                assert_eq!(calls.len(), 1, "one call per tool_calls delta: {chunk}");
                out.events.push(("tool_call", calls[0].clone()));
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                out.finish_reasons.push(reason.to_string());
            }
        }
        out
    }

    fn channel(&self, name: &str) -> String {
        self.events
            .iter()
            .filter(|(channel, _)| *channel == name)
            .filter_map(|(_, value)| value.as_str())
            .collect()
    }

    fn tool_calls(&self) -> Vec<&serde_json::Value> {
        self.events
            .iter()
            .filter(|(channel, _)| *channel == "tool_call")
            .map(|(_, value)| value)
            .collect()
    }
}

/// Stream `script` and request it without streaming; assert the two agree
/// (content, tool calls with their arguments, `finish_reason`) and return
/// both.
async fn stream_and_json(
    script: &[u32],
    template: Option<&str>,
    max_tokens: usize,
) -> (Streamed, serde_json::Value) {
    let (status, _, body) = fx::post(
        scripted_router(script, template),
        "/v1/chat/completions",
        tools_request(true, max_tokens),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
    let streamed = Streamed::parse(&body);

    let (status, _, text) = fx::post(
        scripted_router(script, template),
        "/v1/chat/completions",
        tools_request(false, max_tokens),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let json: serde_json::Value = serde_json::from_str(&text).expect("a chat.completion");
    let choice = &json["choices"][0];

    assert_eq!(
        streamed.finish_reasons,
        vec![choice["finish_reason"]
            .as_str()
            .unwrap_or_default()
            .to_string()],
        "one final chunk, with the non-streamed finish_reason: {body} vs {json}"
    );
    assert_eq!(
        streamed.channel("content"),
        choice["message"]["content"].as_str().unwrap_or_default(),
        "streamed content equals the non-streamed content: {body} vs {json}"
    );
    assert_eq!(
        streamed.channel("reasoning"),
        choice["message"]["reasoning_content"]
            .as_str()
            .unwrap_or_default(),
        "{body} vs {json}"
    );
    let json_calls = choice["message"]["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let streamed_calls = streamed.tool_calls();
    assert_eq!(streamed_calls.len(), json_calls.len(), "{body} vs {json}");
    for (index, (streamed_call, json_call)) in streamed_calls.iter().zip(&json_calls).enumerate() {
        assert_eq!(streamed_call["index"], index, "{body}");
        assert_eq!(streamed_call["type"], "function", "{body}");
        assert!(
            streamed_call["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("call_")),
            "{body}"
        );
        assert_eq!(
            streamed_call["function"], json_call["function"],
            "{body} vs {json}"
        );
    }
    (streamed, json)
}

#[tokio::test]
async fn stream_tools_one_xml_call_is_one_tool_calls_delta_and_finishes_with_tool_calls() {
    let (streamed, json) = stream_and_json(&xml_call(&[("city", "Tokyo")]), None, 256).await;
    assert_eq!(streamed.events.len(), 1, "{streamed:?}");
    let call = streamed.tool_calls()[0];
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(call["function"]["arguments"], r#"{"city":"Tokyo"}"#);
    assert_eq!(streamed.finish_reasons, ["tool_calls"]);
    assert!(json["choices"][0]["message"]["content"].is_null(), "{json}");
}

#[tokio::test]
async fn stream_tools_two_calls_are_two_indexed_deltas() {
    let mut script = xml_call(&[("city", "Tokyo")]);
    script.extend(fx::byte_ids("\n"));
    script.extend(xml_call(&[("city", "Paris")]));
    let (streamed, _) = stream_and_json(&script, None, 512).await;
    let calls = streamed.tool_calls();
    assert_eq!(calls.len(), 2, "{streamed:?}");
    assert_eq!(calls[0]["index"], 0);
    assert_eq!(calls[1]["index"], 1);
    assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Tokyo"}"#);
    assert_eq!(calls[1]["function"]["arguments"], r#"{"city":"Paris"}"#);
    assert_ne!(calls[0]["id"], calls[1]["id"]);
    assert_eq!(streamed.finish_reasons, ["tool_calls"]);
}

#[tokio::test]
async fn stream_tools_json_form_call_is_parsed() {
    let mut script = vec![fx::TOOL_CALL_OPEN];
    script.extend(fx::byte_ids(
        "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n",
    ));
    script.push(fx::TOOL_CALL_CLOSE);
    let (streamed, _) = stream_and_json(&script, None, 256).await;
    let calls = streamed.tool_calls();
    assert_eq!(calls.len(), 1, "{streamed:?}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Tokyo"}"#);
    assert_eq!(streamed.finish_reasons, ["tool_calls"]);
}

#[tokio::test]
async fn stream_tools_prose_then_call_streams_the_prose_first() {
    let mut script = fx::byte_ids("Let me check. ");
    script.extend(xml_call(&[("city", "Tokyo")]));
    let (streamed, json) = stream_and_json(&script, None, 256).await;
    let first_call = streamed
        .events
        .iter()
        .position(|(channel, _)| *channel == "tool_call")
        .expect("a tool_calls delta");
    assert!(
        streamed.events[..first_call]
            .iter()
            .all(|(channel, _)| *channel == "content"),
        "the prose streams before the call: {streamed:?}"
    );
    assert_eq!(streamed.channel("content"), "Let me check.");
    assert_eq!(
        json["choices"][0]["message"]["content"], "Let me check.",
        "{json}"
    );
    assert_eq!(streamed.finish_reasons, ["tool_calls"]);
}

/// A block still open when the generation ends (here: the model reached
/// EOS mid-call) is not a call: its text is flushed as content and the
/// finish reason is the natural one.
#[tokio::test]
async fn stream_tools_unclosed_call_is_flushed_as_content_with_the_natural_finish_reason() {
    let mut script = fx::byte_ids("Checking. ");
    script.push(fx::TOOL_CALL_OPEN);
    script.extend(fx::byte_ids(
        "\n<function=get_weather>\n<parameter=city>\nTok",
    ));
    let (streamed, _) = stream_and_json(&script, None, 256).await;
    assert!(streamed.tool_calls().is_empty(), "{streamed:?}");
    assert_eq!(
        streamed.channel("content"),
        "Checking. <tool_call>\n<function=get_weather>\n<parameter=city>\nTok"
    );
    assert_eq!(streamed.finish_reasons, ["stop"]);

    // Cut off by `max_tokens` instead: the natural finish is "length".
    let (streamed, _) = stream_and_json(&script, None, script.len() - 2).await;
    assert!(streamed.tool_calls().is_empty(), "{streamed:?}");
    assert!(streamed
        .channel("content")
        .starts_with("Checking. <tool_call>"));
    assert_eq!(streamed.finish_reasons, ["length"]);
}

/// The vocabulary has a `<tool_call>` token, so the text `<tool_call>`
/// spelled out of ordinary byte tokens (a model writing ABOUT the tag) is
/// content and never opens a call.
#[tokio::test]
async fn stream_tools_a_tag_spelled_from_ordinary_tokens_does_not_trigger() {
    let text = "Use <tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call> like so.";
    let (streamed, _) = stream_and_json(&fx::byte_ids(text), None, 256).await;
    assert!(streamed.tool_calls().is_empty(), "{streamed:?}");
    assert_eq!(streamed.channel("content"), text);
    assert_eq!(streamed.finish_reasons, ["stop"]);
}

#[tokio::test]
async fn stream_tools_reasoning_deltas_come_first_and_the_parser_only_sees_content() {
    let mut script = vec![fx::THINK_OPEN];
    script.extend(fx::byte_ids("\nI will call <tool_call> now.\n"));
    script.push(fx::THINK_CLOSE);
    script.extend(fx::byte_ids("\n\n"));
    script.extend(xml_call(&[("city", "Tokyo")]));
    let (streamed, json) = stream_and_json(&script, None, 512).await;
    let last_reasoning = streamed
        .events
        .iter()
        .rposition(|(channel, _)| *channel == "reasoning")
        .expect("reasoning deltas");
    let first_call = streamed
        .events
        .iter()
        .position(|(channel, _)| *channel == "tool_call")
        .expect("a tool_calls delta");
    assert!(last_reasoning < first_call, "{streamed:?}");
    assert_eq!(
        streamed.channel("reasoning"),
        "I will call <tool_call> now.\n"
    );
    assert_eq!(streamed.tool_calls().len(), 1);
    assert_eq!(
        json["choices"][0]["message"]["reasoning_content"],
        "I will call <tool_call> now.\n"
    );
}

/// `stream_options.include_usage` still yields the usage chunk after the
/// `tool_calls` finish chunk.
#[tokio::test]
async fn stream_tools_include_usage_still_yields_the_usage_chunk() {
    let script = xml_call(&[("city", "Tokyo")]);
    let mut request = tools_request(true, 256);
    request["stream_options"] = serde_json::json!({"include_usage": true});
    let (status, _, body) = fx::post(
        scripted_router(&script, None),
        "/v1/chat/completions",
        request,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let streamed = Streamed::parse(&body);
    assert_eq!(streamed.finish_reasons, ["tool_calls"]);
    let usage = streamed.usage.expect("a usage chunk");
    assert_eq!(usage["completion_tokens"], script.len(), "{body}");
}

/// Several streamed choices are never interleaved: `stream` + `tools` +
/// `n > 1` is refused with a `400` naming `n`.
#[tokio::test]
async fn stream_tools_with_n_above_one_is_refused_naming_n() {
    let mut request = tools_request(true, 16);
    request["n"] = serde_json::json!(2);
    let (status, _, body) = fx::post(
        scripted_router(&xml_call(&[("city", "Tokyo")]), None),
        "/v1/chat/completions",
        request,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("a JSON error");
    assert_eq!(json["error"]["param"], "n", "{json}");
}

/// The OpenAI `arguments` string keeps the model's own parameter order —
/// `zeta` before `alpha` stays `zeta` before `alpha` — through the
/// non-streamed response and the stream alike.
#[tokio::test]
async fn xml_tool_call_arguments_keep_emission_order() {
    let script = xml_call(&[("zeta", "last letter"), ("middle", "7"), ("alpha", "first")]);
    let (streamed, json) = stream_and_json(&script, None, 512).await;
    let expected = r#"{"zeta":"last letter","middle":7,"alpha":"first"}"#;
    assert_eq!(streamed.tool_calls()[0]["function"]["arguments"], expected);
    assert_eq!(
        json["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        expected
    );
}

/// A template that teaches the XML tool-call form (all three markers) is
/// parsed with the Qwen3-Coder rules: its generation prompt opens the think
/// span, the reasoning's leading whitespace is dropped, and a `<tool_call>`
/// ends the reasoning without a `</think>`.
#[tokio::test]
async fn stream_tools_under_an_xml_form_template_end_the_reasoning_at_the_call() {
    const XML_FORM: &str = "{% if tools %}<|im_start|>system\nCall a tool as <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call><|im_end|>\n{% endif %}{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}";
    let mut script = fx::byte_ids("\n  plan the call\n");
    script.extend(xml_call(&[("city", "Tokyo")]));
    let (streamed, json) = stream_and_json(&script, Some(XML_FORM), 512).await;
    assert_eq!(streamed.channel("reasoning"), "plan the call\n");
    assert_eq!(streamed.tool_calls().len(), 1, "{streamed:?}");
    assert_eq!(streamed.finish_reasons, ["tool_calls"]);
    assert!(json["choices"][0]["message"]["content"].is_null(), "{json}");
}
