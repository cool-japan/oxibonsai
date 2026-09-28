//! Unit tests for [`crate::server::response_pipeline`].
//!
//! Attached to `response_pipeline.rs` as its `#[cfg(test)] mod tests` via
//! `#[path]`. The endpoint-level behaviour (SSE event sequences, streamed
//! vs non-streamed parity) is covered by the `stream_tools_*` tests of both
//! chat endpoints; these pin the pipeline's own ordering rules.

use super::*;
use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

/// A stop stage that never stops (no stop sequences).
#[derive(Default)]
struct NoStop;

impl ContentStop for NoStop {
    fn push_text(&mut self, text: &str) -> String {
        text.to_string()
    }

    fn release_held(&mut self) -> String {
        String::new()
    }

    fn mark_stopped(&mut self) {}

    fn stopped(&self) -> bool {
        false
    }
}

/// A stop stage over one stop string, holding back a possible prefix of it
/// (the contract the endpoints' stages implement), with an optional stop id.
struct OneStop {
    stop: String,
    stop_id: Option<u32>,
    pending: String,
    stopped: bool,
}

impl OneStop {
    fn new(stop: &str, stop_id: Option<u32>) -> Self {
        Self {
            stop: stop.to_string(),
            stop_id,
            pending: String::new(),
            stopped: false,
        }
    }
}

impl ContentStop for OneStop {
    fn push_text(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        self.pending.push_str(text);
        if let Some(pos) = self.pending.find(&self.stop) {
            self.stopped = true;
            let out = self.pending[..pos].to_string();
            self.pending.clear();
            return out;
        }
        let keep = (1..self.stop.len())
            .rev()
            .find(|&k| self.pending.ends_with(&self.stop[..k]))
            .unwrap_or(0);
        let cut = self.pending.len() - keep;
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        out
    }

    fn is_stop_id(&self, id: u32) -> bool {
        !self.stopped && self.stop_id == Some(id)
    }

    fn release_held(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }

    fn mark_stopped(&mut self) {
        self.stopped = true;
    }

    fn stopped(&self) -> bool {
        self.stopped
    }
}

/// The byte ids of `text` followed by nothing (see `fx::byte_ids`).
fn ids(text: &str) -> Vec<u32> {
    fx::byte_ids(text)
}

/// A generation: ordinary text, then one XML `get_weather(city=Tokyo)` call.
fn text_then_xml_call(prefix: &str) -> Vec<u32> {
    let mut out = ids(prefix);
    out.push(fx::TOOL_CALL_OPEN);
    out.extend(ids(
        "\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n",
    ));
    out.push(fx::TOOL_CALL_CLOSE);
    out
}

fn tools_shape(tok: &TokenizerBridge) -> ResponseShape {
    ResponseShape::resolve(Some(tok), &[], None, true)
}

fn collect<S: ContentStop>(
    tok: &TokenizerBridge,
    shape: &ResponseShape,
    stop: S,
    generated: &[u32],
) -> CollectedResponse {
    ResponsePipeline::new(shape, Some(tok), stop).collect(Some(tok), generated)
}

#[test]
fn content_before_a_call_is_released_then_the_call_and_the_finish_is_tool_calls() {
    let tok = fx::byte_tokenizer_with_markers();
    let response = collect(
        &tok,
        &tools_shape(&tok),
        NoStop,
        &text_then_xml_call("Checking. "),
    );
    assert_eq!(response.content, "Checking.");
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].function.name, "get_weather");
    assert_eq!(
        response.tool_calls[0].function.arguments,
        r#"{"city":"Tokyo"}"#
    );
    assert_eq!(response.end.calls, 1);
    assert_eq!(response.end.finish_reason(5, 5), "tool_calls");
    assert_eq!(response.reasoning_content, None);
}

#[test]
fn the_streamed_events_concatenate_to_the_collected_response() {
    let tok = fx::byte_tokenizer_with_markers();
    let shape = tools_shape(&tok);
    let generated = text_then_xml_call("Let me look that up.");
    let mut pipeline = ResponsePipeline::new(&shape, Some(&tok), NoStop);
    let mut events = Vec::new();
    for &id in &generated {
        events.extend(pipeline.push(Some(&tok), id));
    }
    let (tail, end) = pipeline.finish();
    events.extend(tail);
    let collected = collect(&tok, &shape, NoStop, &generated);
    let streamed_content: String = events
        .iter()
        .filter_map(|event| match event {
            ResponseEvent::Content(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(streamed_content, collected.content);
    assert_eq!(end, collected.end);
    let streamed_calls: Vec<&ToolCall> = events
        .iter()
        .filter_map(|event| match event {
            ResponseEvent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(streamed_calls.len(), collected.tool_calls.len());
    assert_eq!(
        streamed_calls[0].function.arguments,
        collected.tool_calls[0].function.arguments
    );
    // The call is released at its closing token, after all the content.
    assert!(matches!(events.last(), Some(ResponseEvent::ToolCall(_))));
}

#[test]
fn reasoning_comes_first_and_the_parser_only_sees_the_content_phase() {
    let tok = fx::byte_tokenizer_with_markers();
    // A model-emitted think span whose reasoning spells a call out: it is
    // reasoning, never a call.
    let mut generated = vec![fx::THINK_OPEN];
    generated.extend(ids("\nI could write <tool_call> here.\n"));
    generated.push(fx::THINK_CLOSE);
    generated.extend(ids("\n\n"));
    generated.extend(text_then_xml_call(""));
    let response = collect(&tok, &tools_shape(&tok), NoStop, &generated);
    assert_eq!(
        response.reasoning_content.as_deref(),
        Some("I could write <tool_call> here.\n")
    );
    assert_eq!(response.content, "");
    assert_eq!(response.tool_calls.len(), 1);
}

#[test]
fn a_stop_sequence_before_a_call_drops_the_call_and_reports_stop() {
    let tok = fx::byte_tokenizer_with_markers();
    let response = collect(
        &tok,
        &tools_shape(&tok),
        OneStop::new("STOP", None),
        &text_then_xml_call("Answer STOP more"),
    );
    assert_eq!(response.content, "Answer ");
    assert!(response.tool_calls.is_empty());
    assert!(response.end.stopped);
    assert_eq!(response.end.finish_reason(99, 100), "stop");
}

#[test]
fn a_stop_string_inside_a_call_does_not_stop_anything() {
    let tok = fx::byte_tokenizer_with_markers();
    let response = collect(
        &tok,
        &tools_shape(&tok),
        OneStop::new("Tokyo", None),
        &text_then_xml_call(""),
    );
    assert_eq!(response.tool_calls.len(), 1);
    assert!(!response.end.stopped);
}

#[test]
fn held_back_stop_prefix_is_released_before_a_call() {
    let tok = fx::byte_tokenizer_with_markers();
    // "ST" could still grow into "STOP" until the call opens.
    let response = collect(
        &tok,
        &tools_shape(&tok),
        OneStop::new("STOP", None),
        &text_then_xml_call("Checking ST"),
    );
    assert_eq!(response.content, "Checking ST");
    assert_eq!(response.tool_calls.len(), 1);
}

#[test]
fn an_unclosed_block_is_released_as_content_with_the_natural_finish() {
    let tok = fx::byte_tokenizer_with_markers();
    let mut generated = ids("Checking. ");
    generated.push(fx::TOOL_CALL_OPEN);
    generated.extend(ids("\n<function=get_weather>\n<parameter=city>\nTok"));
    let response = collect(&tok, &tools_shape(&tok), NoStop, &generated);
    assert_eq!(
        response.content,
        "Checking. <tool_call>\n<function=get_weather>\n<parameter=city>\nTok"
    );
    assert!(response.tool_calls.is_empty());
    assert!(response.end.unclosed);
    let generated_len = generated.len();
    assert_eq!(
        response.end.finish_reason(generated_len, generated_len),
        "length"
    );
    assert_eq!(response.end.finish_reason(generated_len, 1000), "stop");
}

#[test]
fn tool_call_text_spelled_from_ordinary_tokens_is_content_when_the_id_resolved() {
    let tok = fx::byte_tokenizer_with_markers();
    let text = "Use <tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call> like this.";
    let response = collect(&tok, &tools_shape(&tok), NoStop, &ids(text));
    assert_eq!(response.content, text);
    assert!(response.tool_calls.is_empty());
    assert_eq!(response.end.calls, 0);
}

#[test]
fn a_json_form_call_is_parsed_with_the_models_key_order() {
    let tok = fx::byte_tokenizer_with_markers();
    let mut generated = vec![fx::TOOL_CALL_OPEN];
    generated.extend(ids(
        "\n{\"name\": \"get_weather\", \"arguments\": {\"zeta\": 1, \"alpha\": \"x\"}}\n",
    ));
    generated.push(fx::TOOL_CALL_CLOSE);
    let response = collect(&tok, &tools_shape(&tok), NoStop, &generated);
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(
        response.tool_calls[0].function.arguments,
        r#"{"zeta":1,"alpha":"x"}"#
    );
    assert_eq!(response.content, "");
}

#[test]
fn without_tools_a_call_block_is_ordinary_content() {
    let tok = fx::byte_tokenizer_with_markers();
    let shape = ResponseShape::resolve(Some(&tok), &[], None, false);
    let response = collect(&tok, &shape, NoStop, &text_then_xml_call("Hi "));
    assert!(response.tool_calls.is_empty());
    assert!(response
        .content
        .starts_with("Hi <tool_call>\n<function=get_weather>"));
}

#[test]
fn a_stop_token_matched_by_id_releases_what_came_before_it() {
    let tok = fx::byte_tokenizer_with_markers();
    let mut generated = ids("Answer ST");
    generated.push(fx::THINK_CLOSE);
    generated.extend(ids("never shown"));
    let shape = ResponseShape::resolve(Some(&tok), &[], None, false);
    let response = collect(
        &tok,
        &shape,
        OneStop::new("STOP", Some(fx::THINK_CLOSE)),
        &generated,
    );
    assert_eq!(response.content, "Answer ST");
    assert!(response.end.stopped);
}

#[test]
fn no_tokenizer_means_empty_content_and_still_a_natural_finish() {
    let shape = ResponseShape::resolve(None, &[1, 2, 3], None, true);
    assert_eq!(
        shape,
        ResponseShape {
            tools_active: true,
            ..ResponseShape::default()
        }
    );
    let response = ResponsePipeline::new(&shape, None, NoStop).collect(None, &[5, 6, 7]);
    assert_eq!(response.content, "");
    assert_eq!(response.reasoning_content, None);
    assert!(response.tool_calls.is_empty());
    assert_eq!(response.end.finish_reason(3, 3), "length");
}

#[test]
fn a_closed_think_tail_stops_watching_for_a_model_opener() {
    let tok = fx::byte_tokenizer_with_markers();
    let open_tail = ResponseShape::resolve(Some(&tok), &[], Some("<|im_start|>assistant\n"), false);
    assert_eq!(open_tail.watched_think_open_id, Some(fx::THINK_OPEN));
    let closed_tail = ResponseShape::resolve(
        Some(&tok),
        &[],
        Some("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        false,
    );
    assert_eq!(closed_tail.watched_think_open_id, None);
    // With the tail closed, a `<think>` the model emits anyway is content.
    let mut generated = vec![fx::THINK_OPEN];
    generated.extend(ids("x"));
    generated.push(fx::THINK_CLOSE);
    generated.extend(ids("4"));
    let response = collect(&tok, &closed_tail, NoStop, &generated);
    assert_eq!(response.reasoning_content, None);
    assert_eq!(response.content, "<think>x</think>4");
    // …while an open tail routes it to reasoning.
    let response = collect(&tok, &open_tail, NoStop, &generated);
    assert_eq!(response.reasoning_content.as_deref(), Some("x"));
    assert_eq!(response.content, "4");
}

#[test]
fn prompt_tail_closes_think_needs_the_close_marker_at_the_very_end() {
    assert!(prompt_tail_closes_think("a<think>\n\n</think>\n\n"));
    assert!(prompt_tail_closes_think("</think>"));
    assert!(!prompt_tail_closes_think("<think>\n"));
    assert!(!prompt_tail_closes_think(
        "<think>x</think>answer<|im_end|>\n<|im_start|>assistant\n"
    ));
}

#[test]
fn the_qwen3_coder_format_drops_the_answers_leading_whitespace_and_ends_reasoning_at_a_call() {
    let tok = fx::byte_tokenizer_with_markers();
    let shape = ResponseShape {
        started_in_think: true,
        watched_think_open_id: None,
        think_close_id: Some(fx::THINK_CLOSE),
        reasoning_format: ReasoningFormat::Qwen3Coder,
        tool_call_open_id: Some(fx::TOOL_CALL_OPEN),
        tool_call_close_id: Some(fx::TOOL_CALL_CLOSE),
        tools_active: true,
    };
    // Reasoning that runs straight into a call (no `</think>`).
    let mut generated = ids("\n  plan it\n");
    generated.extend(text_then_xml_call(""));
    let response = collect(&tok, &shape, NoStop, &generated);
    assert_eq!(response.reasoning_content.as_deref(), Some("plan it\n"));
    assert_eq!(response.tool_calls.len(), 1);
    // And an answer after `</think>` loses every leading whitespace char.
    let mut generated = ids("why");
    generated.push(fx::THINK_CLOSE);
    generated.extend(ids("\n\n\n  4"));
    let response = collect(&tok, &shape, NoStop, &generated);
    assert_eq!(response.reasoning_content.as_deref(), Some("why"));
    assert_eq!(response.content, "4");
}

#[test]
fn finish_reason_precedence() {
    let end = |calls: usize, stopped: bool| ResponseEnd {
        calls,
        stopped,
        unclosed: false,
    };
    assert_eq!(end(1, true).finish_reason(10, 10), "tool_calls");
    assert_eq!(end(0, true).finish_reason(10, 10), "stop");
    assert_eq!(end(0, false).finish_reason(10, 10), "length");
    assert_eq!(end(0, false).finish_reason(3, 10), "stop");
}
