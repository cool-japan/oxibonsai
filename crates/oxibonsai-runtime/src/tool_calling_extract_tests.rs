//! Unit tests for [`super::ToolCallStreamExtractor`].

use super::*;

/// Token ids of the fixture vocabulary's markers.
const OPEN: u32 = 900;
const CLOSE: u32 = 901;
/// An ordinary token id (its text is whatever the test passes).
const TEXT: u32 = 1;

const XML_CALL_BODY: &str =
    "\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n";

/// Feed `pieces` (`(id, text)`) and finish; returns every event and the end
/// summary.
fn run(
    mut extractor: ToolCallStreamExtractor,
    pieces: &[(u32, &str)],
) -> (Vec<ToolStreamEvent>, ToolStreamEnd) {
    let mut events = Vec::new();
    for &(id, text) in pieces {
        events.extend(extractor.push(id, text));
    }
    let (tail, end) = extractor.finish();
    events.extend(tail);
    (events, end)
}

fn gated() -> ToolCallStreamExtractor {
    ToolCallStreamExtractor::new(Some(OPEN), Some(CLOSE))
}

fn text_matched() -> ToolCallStreamExtractor {
    ToolCallStreamExtractor::new(None, None)
}

fn content(events: &[ToolStreamEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            ToolStreamEvent::Content(text) => Some(text.as_str()),
            ToolStreamEvent::Call(_) => None,
        })
        .collect()
}

fn calls(events: &[ToolStreamEvent]) -> Vec<(String, String)> {
    events
        .iter()
        .filter_map(|e| match e {
            ToolStreamEvent::Call(call) => {
                Some((call.function.name.clone(), call.function.arguments.clone()))
            }
            ToolStreamEvent::Content(_) => None,
        })
        .collect()
}

#[test]
fn a_gated_xml_call_is_released_when_its_block_closes() {
    let mut extractor = gated();
    assert!(extractor.push(OPEN, "<tool_call>").is_empty());
    assert!(extractor.in_block());
    assert!(extractor.push(TEXT, XML_CALL_BODY).is_empty());
    let events = extractor.push(CLOSE, "</tool_call>");
    assert_eq!(
        calls(&events),
        vec![("get_weather".to_string(), r#"{"city":"Tokyo"}"#.to_string())]
    );
    assert!(!extractor.in_block());
    let (tail, end) = extractor.finish();
    assert!(tail.is_empty());
    assert_eq!(end.calls, 1);
    assert!(!end.unclosed && !end.malformed);
}

#[test]
fn a_spelled_out_opener_is_content_when_the_vocabulary_has_the_token() {
    let spelled = format!("<tool_call>{XML_CALL_BODY}</tool_call>");
    let (events, end) = run(gated(), &[(TEXT, "About the tag: "), (TEXT, &spelled)]);
    assert!(calls(&events).is_empty(), "{events:?}");
    assert_eq!(content(&events), format!("About the tag: {spelled}"));
    assert_eq!(end.calls, 0);
}

#[test]
fn prose_then_a_call_keeps_the_prose_and_drops_the_separating_whitespace() {
    let (events, end) = run(
        gated(),
        &[
            (TEXT, "Let me check"),
            (TEXT, ".\n"),
            (OPEN, "<tool_call>"),
            (TEXT, XML_CALL_BODY),
            (CLOSE, "</tool_call>"),
        ],
    );
    assert_eq!(content(&events), "Let me check.");
    assert_eq!(calls(&events).len(), 1);
    assert!(matches!(events.last(), Some(ToolStreamEvent::Call(_))));
    assert_eq!(end.calls, 1);
}

#[test]
fn two_calls_separated_by_whitespace_are_both_released() {
    let second = "\n<function=get_time>\n<parameter=zone>\nJST\n</parameter>\n</function>\n";
    let (events, end) = run(
        gated(),
        &[
            (OPEN, "<tool_call>"),
            (TEXT, XML_CALL_BODY),
            (CLOSE, "</tool_call>"),
            (TEXT, "\n"),
            (OPEN, "<tool_call>"),
            (TEXT, second),
            (CLOSE, "</tool_call>"),
        ],
    );
    let names: Vec<String> = calls(&events).into_iter().map(|(name, _)| name).collect();
    assert_eq!(names, ["get_weather", "get_time"]);
    assert_eq!(content(&events), "");
    assert_eq!(end.calls, 2);
}

#[test]
fn a_json_form_call_is_released_with_its_argument_order() {
    let (events, _) = run(
        gated(),
        &[
            (OPEN, "<tool_call>"),
            (
                TEXT,
                "\n{\"name\": \"get_weather\", \"arguments\": {\"zeta\": 1, \"alpha\": \"x\"}}\n",
            ),
            (CLOSE, "</tool_call>"),
        ],
    );
    assert_eq!(
        calls(&events),
        vec![(
            "get_weather".to_string(),
            r#"{"zeta":1,"alpha":"x"}"#.to_string()
        )]
    );
}

#[test]
fn an_unclosed_block_is_released_as_content_at_the_end() {
    let (events, end) = run(
        gated(),
        &[
            (TEXT, "Checking.\n"),
            (OPEN, "<tool_call>"),
            (TEXT, "\n<function=get_weather>\n<parameter=ci"),
        ],
    );
    assert!(calls(&events).is_empty());
    assert_eq!(
        content(&events),
        "Checking.\n<tool_call>\n<function=get_weather>\n<parameter=ci",
        "the held separator and the whole partial block come back verbatim"
    );
    assert!(end.unclosed);
    assert_eq!(end.calls, 0);
}

#[test]
fn a_malformed_block_and_everything_after_it_are_content() {
    let (events, end) = run(
        gated(),
        &[
            (OPEN, "<tool_call>"),
            (TEXT, "not a call"),
            (CLOSE, "</tool_call>"),
            (TEXT, " then text"),
        ],
    );
    assert!(calls(&events).is_empty());
    assert_eq!(
        content(&events),
        "<tool_call>not a call</tool_call> then text"
    );
    assert!(end.malformed);
}

#[test]
fn text_after_a_call_is_dropped_with_everything_after_it() {
    let (events, end) = run(
        gated(),
        &[
            (OPEN, "<tool_call>"),
            (TEXT, XML_CALL_BODY),
            (CLOSE, "</tool_call>"),
            (TEXT, "\nDone."),
            (OPEN, "<tool_call>"),
            (TEXT, XML_CALL_BODY),
            (CLOSE, "</tool_call>"),
        ],
    );
    assert_eq!(calls(&events).len(), 1, "{events:?}");
    assert_eq!(content(&events), "");
    assert_eq!(end.calls, 1);
}

#[test]
fn trailing_whitespace_after_a_call_is_silently_dropped() {
    let (events, end) = run(
        gated(),
        &[
            (OPEN, "<tool_call>"),
            (TEXT, XML_CALL_BODY),
            (CLOSE, "</tool_call>"),
            (TEXT, "\n\n"),
        ],
    );
    assert_eq!(content(&events), "");
    assert_eq!(end.calls, 1);
}

#[test]
fn an_empty_close_token_still_closes_the_block() {
    // A special-flagged `</tool_call>` decodes to nothing.
    let (events, end) = run(gated(), &[(OPEN, ""), (TEXT, XML_CALL_BODY), (CLOSE, "")]);
    assert_eq!(calls(&events).len(), 1);
    assert_eq!(end.calls, 1);
}

#[test]
fn text_matched_openers_hold_back_a_partial_tag_until_it_resolves() {
    let mut extractor = text_matched();
    let events = extractor.push(TEXT, "Hi <tool");
    assert_eq!(content(&events), "Hi");
    let events = extractor.push(TEXT, "box> ok");
    assert_eq!(content(&events), " <toolbox> ok");
    let (tail, end) = extractor.finish();
    assert!(tail.is_empty());
    assert_eq!(end.calls, 0);
}

#[test]
fn text_matched_openers_split_across_pieces_open_a_block() {
    let (events, end) = run(
        text_matched(),
        &[
            (TEXT, "Sure.\n<tool_"),
            (TEXT, "call>"),
            (TEXT, XML_CALL_BODY),
            (TEXT, "</tool_"),
            (TEXT, "call>"),
        ],
    );
    assert_eq!(content(&events), "Sure.");
    assert_eq!(calls(&events).len(), 1);
    assert_eq!(end.calls, 1);
}

#[test]
fn text_matched_a_whole_response_in_one_piece_parses_every_call() {
    let text = format!(
        "Plan.\n<tool_call>{XML_CALL_BODY}</tool_call>\n<tool_call>\n{{\"name\": \"b\"}}\n</tool_call>"
    );
    let (events, end) = run(text_matched(), &[(TEXT, &text)]);
    assert_eq!(content(&events), "Plan.");
    let names: Vec<String> = calls(&events).into_iter().map(|(name, _)| name).collect();
    assert_eq!(names, ["get_weather", "b"]);
    assert_eq!(end.calls, 2);
}

#[test]
fn a_response_without_calls_keeps_every_byte_including_trailing_whitespace() {
    let (events, end) = run(gated(), &[(TEXT, "  Hello"), (TEXT, " world \n")]);
    assert_eq!(content(&events), "  Hello world \n");
    assert_eq!(end, ToolStreamEnd::default());
}

#[test]
fn a_whitespace_only_prefix_before_a_call_leaves_no_content() {
    let (events, _) = run(
        gated(),
        &[
            (TEXT, " \n"),
            (OPEN, "<tool_call>"),
            (TEXT, XML_CALL_BODY),
            (CLOSE, "</tool_call>"),
        ],
    );
    assert_eq!(content(&events), "");
    assert_eq!(calls(&events).len(), 1);
}

mod fuzz {
    use super::*;
    use proptest::prelude::*;

    fn piece() -> impl Strategy<Value = (u32, String)> {
        prop_oneof![
            Just((OPEN, "<tool_call>".to_string())),
            Just((CLOSE, "</tool_call>".to_string())),
            Just((TEXT, "<tool_call>".to_string())),
            Just((TEXT, "</tool_call>".to_string())),
            Just((TEXT, "<function=f>".to_string())),
            Just((TEXT, "</function>".to_string())),
            Just((TEXT, "<parameter=k>\nv\n</parameter>".to_string())),
            Just((TEXT, "{\"name\": \"f\"}".to_string())),
            Just((TEXT, "\n".to_string())),
            "[a-z <>/_=\n]{0,8}".prop_map(|s| (TEXT, s)),
        ]
    }

    proptest! {
        /// Never panics, and never loses or invents bytes in a response
        /// that contains no opener at all.
        #[test]
        fn extraction_never_panics(pieces in prop::collection::vec(piece(), 0..24), gate in any::<bool>()) {
            let extractor = if gate { gated() } else { text_matched() };
            let refs: Vec<(u32, &str)> = pieces.iter().map(|(id, s)| (*id, s.as_str())).collect();
            let _ = run(extractor, &refs);
        }

        #[test]
        fn content_without_openers_passes_through_unchanged(text in "[a-z ,.!?\n]{0,64}") {
            let (events, end) = run(gated(), &[(TEXT, text.as_str())]);
            prop_assert_eq!(content(&events), text);
            prop_assert_eq!(end, ToolStreamEnd::default());
        }
    }
}
