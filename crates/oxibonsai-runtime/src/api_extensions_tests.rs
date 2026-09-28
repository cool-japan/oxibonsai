//! Unit tests for [`crate::api_extensions`].
//!
//! Attached to `api_extensions.rs` as its `#[cfg(test)] mod tests` via
//! `#[path]`, so the tests keep full access to the module's private items
//! (`StreamDecodeState`, `resolve_sampling_params`, `idempotency_cache_key`,
//! ...) while `api_extensions.rs` itself stays under the workspace
//! 2000-line ceiling (mirrors `engine.rs` / `engine_tests.rs`).
//!
//! Distinct from the crate's external integration file,
//! `tests/api_extensions_tests.rs` (which cannot reach these
//! `pub(crate)`/private items at all): this file covers the findings RT-05,
//! RT-06, RT-12, SV-25, SV-32 and `TOK-M2` on this endpoint.

use super::*;
use crate::engine::InferenceEngine;

#[test]
fn json_mode_enforcer_valid_passthrough() {
    let enforcer = JsonModeEnforcer::new();
    let json = r#"{"key": "value"}"#;
    assert_eq!(enforcer.enforce(json), json);
}

#[test]
fn json_mode_enforcer_extracts_substring() {
    let enforcer = JsonModeEnforcer::new();
    let text = r#"Here is some text {"key": "value"} and more"#;
    let result = enforcer.enforce(text);
    assert!(
        crate::api_types::is_valid_json(&result),
        "result should be valid JSON, got: {result}"
    );
}

#[test]
fn json_mode_enforcer_wraps_invalid() {
    let enforcer = JsonModeEnforcer::new();
    let text = "not json at all";
    let result = enforcer.enforce(text);
    assert!(
        crate::api_types::is_valid_json(&result),
        "result should be valid JSON, got: {result}"
    );
    let v: serde_json::Value = serde_json::from_str(&result).expect("should parse as json");
    assert!(v.get("response").is_some(), "should have 'response' key");
}

#[test]
fn stop_checker_finds_sequence() {
    let checker = StopChecker::new(vec!["STOP".to_string(), "END".to_string()]);
    assert_eq!(checker.check("Hello STOP world"), Some("STOP"));
    assert_eq!(checker.check("No match here"), None);
}

#[test]
fn stop_checker_truncates_correctly() {
    let checker = StopChecker::new(vec!["<end>".to_string()]);
    let (truncated, hit) = checker.truncate_at_stop("Hello world<end>more text");
    assert_eq!(truncated, "Hello world");
    assert!(hit);
}

#[test]
fn stop_checker_no_match() {
    let checker = StopChecker::new(vec!["nope".to_string()]);
    let (truncated, hit) = checker.truncate_at_stop("Hello world");
    assert_eq!(truncated, "Hello world");
    assert!(!hit);
}

#[test]
fn stop_checker_is_empty() {
    let empty = StopChecker::new(vec![]);
    assert!(empty.is_empty());
    let non_empty = StopChecker::new(vec!["x".to_string()]);
    assert!(!non_empty.is_empty());
}

#[test]
fn apply_frequency_penalty_reduces_seen() {
    let mut logits = vec![1.0f32, 2.0, 3.0];
    let mut counts = HashMap::new();
    counts.insert(1u32, 2usize); // token 1 seen twice
    apply_frequency_penalty(&mut logits, &counts, 0.5, 0.0);
    // token 1 logit should be reduced by 0.5 * 2 = 1.0
    assert!(
        (logits[1] - 1.0).abs() < 1e-5,
        "expected 1.0, got {}",
        logits[1]
    );
    // others unchanged
    assert!((logits[0] - 1.0).abs() < 1e-5);
    assert!((logits[2] - 3.0).abs() < 1e-5);
}

#[test]
fn apply_presence_penalty_reduces_seen() {
    let mut logits = vec![1.0f32, 2.0, 3.0];
    let mut counts = HashMap::new();
    counts.insert(0u32, 1usize);
    apply_frequency_penalty(&mut logits, &counts, 0.0, 1.0);
    assert!(
        (logits[0] - 0.0).abs() < 1e-5,
        "expected 0.0, got {}",
        logits[0]
    );
    assert!((logits[1] - 2.0).abs() < 1e-5);
}

#[test]
fn extract_balanced_object() {
    let text = r#"prefix {"a":1} suffix"#;
    let result = extract_balanced(text, '{', '}');
    assert_eq!(result.as_deref(), Some(r#"{"a":1}"#));
}

#[test]
fn extract_balanced_array() {
    let text = r#"pre [1,2,3] post"#;
    let result = extract_balanced(text, '[', ']');
    assert_eq!(result.as_deref(), Some("[1,2,3]"));
}

// ── determine_extended_finish_reason (finding 28 regression) ─────────────

#[test]
fn finish_reason_tool_calls_takes_priority() {
    // Even if the run also happened to exhaust max_tokens, a parsed tool
    // call must win.
    assert_eq!(
        determine_extended_finish_reason(true, false, 10, 10),
        "tool_calls"
    );
    assert_eq!(
        determine_extended_finish_reason(true, true, 10, 10),
        "tool_calls"
    );
}

#[test]
fn finish_reason_stop_sequence_wins_over_length() {
    // A stop sequence match must report "stop" even when output_len
    // happens to equal max_tokens.
    assert_eq!(determine_extended_finish_reason(false, true, 8, 8), "stop");
}

#[test]
fn finish_reason_length_when_truncated() {
    // Regression for finding 28: previously this always returned "stop"
    // regardless of truncation.
    assert_eq!(
        determine_extended_finish_reason(false, false, 8, 8),
        "length"
    );
    assert_eq!(
        determine_extended_finish_reason(false, false, 10, 8),
        "length"
    );
}

#[test]
fn finish_reason_stop_on_natural_eos() {
    assert_eq!(determine_extended_finish_reason(false, false, 3, 8), "stop");
}

// ── StreamDecodeState (RT-06 — reproduce first) ───────────────────────────

#[test]
fn stream_decode_state_holds_back_split_stop_sequence() {
    // Reproduces RT-06: a stop sequence ("STOP") arrives split across two
    // separate decode steps ("Hello ST" then "OP world"), simulating a
    // stop string split across two SSE chunks. The OLD per-chunk-only
    // check (`accumulated.find` combined with a `.max(chunk_start)`
    // clamp that only ever suppressed the *current* chunk) would have
    // already sent "Hello ST" -- including the leaked "ST" prefix -- to
    // the client on the first chunk, before the second chunk ever
    // revealed the match.
    let mut state = StreamDecodeState::new(&["STOP".to_string()], HashSet::new());
    let first = state.feed("Hello ST");
    assert_eq!(
        first,
        Some("Hello ".to_string()),
        "text that is provably safe (outside the hold-back window) is \
         emitted immediately -- but the trailing 'ST', an unfinished \
         prefix of 'STOP', must NOT be part of it"
    );
    let second = state.feed("OP world");
    assert_eq!(
        second, None,
        "the stop sequence completes exactly where the previous chunk \
         already stopped emitting, so there is nothing new to reveal \
         before it -- and, crucially, 'ST' was never sent in `first`"
    );
    assert!(state.is_stopped());
    assert_eq!(
        state.finish(),
        None,
        "after a stop match, finish() must not flush anything else"
    );
}

#[test]
fn stream_decode_state_never_leaks_any_fragment_of_the_stop_sequence() {
    // The actual regression property, generalized: the concatenation of
    // every `Some` returned by `feed` must never contain any part of the
    // configured stop sequence, no matter how the text is chunked.
    let mut state = StreamDecodeState::new(&["<|end|>".to_string()], HashSet::new());
    let mut emitted = String::new();
    for chunk in ["prefix <", "|", "end", "|>", " trailing"] {
        if let Some(v) = state.feed(chunk) {
            emitted.push_str(&v);
        }
    }
    assert_eq!(emitted, "prefix ");
    assert!(
        !emitted.contains('<'),
        "no fragment of the marker must leak, got: {emitted:?}"
    );
}

#[test]
fn stream_decode_state_holds_back_multibyte_char_boundary_safely() {
    // RT-06's corrected fix: the hold-back cut must land on a UTF-8 char
    // boundary. Feeds a multi-byte string ("日本語", 3 bytes per
    // codepoint) immediately followed by an unfinished stop-sequence
    // prefix, so the hold-back window does not align with a codepoint
    // boundary by coincidence -- if `hold_back_len` ever regressed to a
    // raw byte-count cut, this panics instead of merely mis-behaving.
    let mut state = StreamDecodeState::new(&["STOP".to_string()], HashSet::new());
    let mut emitted = String::new();
    for chunk in ["日本語ST", "OP done"] {
        if let Some(v) = state.feed(chunk) {
            emitted.push_str(&v);
        }
    }
    assert_eq!(emitted, "日本語");
}

#[test]
fn stream_decode_state_token_id_fast_path_never_reveals_the_marker() {
    // The preferential id match: a stop sequence that is itself a single
    // special token (e.g. a model's own `<|im_end|>`) must stop
    // generation the instant that token id arrives, before any of its
    // text is even considered. This is also the *only* way to exercise
    // this path at all in this crate: an HTTP-level test always goes
    // through `create_router(engine, None)` (no tokenizer attached), so
    // `stop_token_ids` is always computed empty there.
    let mut stop_ids = HashSet::new();
    stop_ids.insert(999u32);
    let mut state = StreamDecodeState::new(&["<|im_end|>".to_string()], stop_ids);
    assert!(
        !state.hit_stop_by_id(1),
        "an unrelated token id must not trip the fast path"
    );
    assert!(state.feed("hello ").is_some());
    assert!(
        state.hit_stop_by_id(999),
        "the configured marker's own token id must trip the fast path"
    );
    assert!(state.is_stopped());
    assert_eq!(
        state.feed("<|im_end|>"),
        None,
        "once stopped, feed must stay inert even if called again"
    );
    // Nothing was held back here ("hello " fully cleared the hold-back
    // window on its own), so the flush companion must
    // be a no-op too -- it must never manufacture output that was never
    // there. The case where something *is* held back at the moment the id
    // fast path fires is covered by
    // `stream_decode_state_flush_before_stop_matches_the_reported_regression`
    // and `stream_decode_state_flushes_realistic_held_back_prefix_before_id_stop`
    // below.
    assert_eq!(
        state.flush_before_stop(),
        None,
        "flush_before_stop must not reveal anything when the hold-back \
         window was already empty"
    );
}

// ── RT-06 regression: the id fast path must flush the
//    hold-back window, not silently drop it (`finish()`'s `hit_stop` guard
//    is correct for the *text*-match path, which already flushed its own
//    safe prefix inline, but was wrong for the id path, which never
//    flushes anything on its own) ───────────────────────────────────────

#[test]
fn stream_decode_state_flush_before_stop_matches_the_reported_regression() {
    // The exact reported repro: "AB" is held back as an
    // unfinished prefix of "ABC" when the *unrelated* single-token marker
    // "<|im_end|>" (id 999) arrives and stops generation. "AB" is real
    // model output that was never part of any matched stop sequence and
    // must survive.
    let mut stop_ids = HashSet::new();
    stop_ids.insert(999u32);
    let mut state =
        StreamDecodeState::new(&["ABC".to_string(), "<|im_end|>".to_string()], stop_ids);
    assert_eq!(
        state.feed("xyzAB"),
        Some("xyz".to_string()),
        "'AB' is an unfinished prefix of 'ABC' and must be held back"
    );
    assert!(
        state.hit_stop_by_id(999),
        "the configured marker's own token id must trip the fast path"
    );
    assert_eq!(
        state.finish(),
        None,
        "finish() must stay a no-op once hit_stop is set -- it is not the \
         right place to fix this, since the text-match path already \
         flushed its own safe prefix inline via feed()"
    );
    assert_eq!(
        state.flush_before_stop(),
        Some("AB".to_string()),
        "the id fast path must not silently drop text that was already \
         decoded and safely held back before it fired"
    );
    assert_eq!(
        state.flush_before_stop(),
        None,
        "flush_before_stop must be idempotent -- nothing left to flush twice"
    );
}

#[test]
fn stream_decode_state_flushes_realistic_held_back_prefix_before_id_stop() {
    // The realistic production trigger named in the finding: `stop:
    // ["</think>", "\nUser:"]` with a tokenizer attached. The newline that
    // normally precedes `</think>` is held back as an unfinished prefix of
    // the *other* configured stop sequence, "\nUser:"; the model then
    // emits `</think>` as its own single token, tripping the id fast path.
    // The held-back newline belongs to neither matched sequence and must
    // reach the client, not be eaten alongside the marker it happened to
    // precede.
    let mut stop_ids = HashSet::new();
    stop_ids.insert(555u32); // the `</think>` token's id
    let mut state =
        StreamDecodeState::new(&["</think>".to_string(), "\nUser:".to_string()], stop_ids);
    assert_eq!(
        state.feed("Here is the answer.\n"),
        Some("Here is the answer.".to_string()),
        "the trailing newline is an unfinished prefix of '\\nUser:' and \
         must be held back"
    );
    assert!(
        state.hit_stop_by_id(555),
        "the </think> token id must trip the fast path"
    );
    assert_eq!(
        state.flush_before_stop(),
        Some("\n".to_string()),
        "the held-back newline was never part of the </think> marker that \
         actually stopped generation -- it is real output and must be \
         flushed, not dropped"
    );
    assert_eq!(
        state.finish(),
        None,
        "nothing should be left to flush a second time"
    );
}

/// A narrower residual of the `step_decode`-`Ok(None)` fix: a
/// stop-configured id landing in the post-`</think>` newline window
/// (`Phase::JustClosed`), before any real content token, must still stop
/// generation. Confirmed live on the real 27B model: `<|im_end|>` really is
/// vocabulary-flagged `special` there, so it really does decode to `""`
/// via `step_decode`, and a model that stops immediately after its
/// reasoning (no separate content at all) would hit exactly this.
///
/// Both halves are pinned: the id fast path trips for the configured id
/// whatever the reasoning phase, and the splitter hands that empty-text id
/// on as (empty) content instead of swallowing it as a `Boundary` — which
/// is what lets the stop stage behind the split (where the response
/// pipeline checks stop ids) see it at all.
#[test]
fn a_stop_id_landing_in_the_post_think_newline_window_still_stops_generation() {
    const CLOSE_THINK_ID: u32 = 248069;
    const IM_END_ID: u32 = 248046; // confirmed special on the real 27B model
    let mut stop_ids = HashSet::new();
    stop_ids.insert(IM_END_ID);
    let mut decode_loop = StreamDecodeState::new(&["<|im_end|>".to_string()], stop_ids);
    let mut splitter = crate::reasoning::ReasoningSplitter::new(true, Some(CLOSE_THINK_ID));

    // A token or two of reasoning, matching the loop's own per-token shape.
    assert!(!matches!(
        splitter.push(1, "thinking"),
        crate::reasoning::ReasoningChunk::Boundary
    ));

    // `</think>` — enters `Phase::JustClosed`.
    assert_eq!(
        splitter.push(CLOSE_THINK_ID, "</think>"),
        crate::reasoning::ReasoningChunk::Boundary
    );

    // `<|im_end|>` arrives immediately after, decoding to `""` (special,
    // `step_decode` returns `None`) -- and we are (correctly) not
    // `in_reasoning()` here (`JustClosed`, not `Reasoning`).
    assert!(
        !splitter.in_reasoning(),
        "must be in JustClosed, not Reasoning, for this to be the case under test"
    );
    assert!(
        decode_loop.hit_stop_by_id(IM_END_ID),
        "the id fast path must trip for the configured stop id regardless of reasoning phase"
    );
    // Nothing of the claimed id is ever emitted: the stream stops here.

    // And the splitter itself no longer hides such an id: an empty-text id
    // in the post-`</think>` window is handed on as (empty) content rather
    // than swallowed as a `Boundary`, so a stage after the split — the stop
    // check, the tool-call extractor — still sees the id.
    let mut splitter_after_close =
        crate::reasoning::ReasoningSplitter::new(true, Some(CLOSE_THINK_ID));
    let _ = splitter_after_close.push(1, "thinking");
    let _ = splitter_after_close.push(CLOSE_THINK_ID, "</think>");
    assert_eq!(
        splitter_after_close.push(IM_END_ID, ""),
        crate::reasoning::ReasoningChunk::Content(String::new()),
        "an empty-text id after `</think>` must reach the next stage"
    );
}

#[test]
fn stream_decode_state_flushes_held_back_tail_on_finish() {
    // Generation ends (EOS / max_tokens) before an unfinished prefix
    // ever grows into the full stop sequence: it must be flushed as
    // real output on `finish()`, not silently dropped.
    let mut state = StreamDecodeState::new(&["NEVER_APPEARS".to_string()], HashSet::new());
    assert_eq!(
        state.feed("hello NEVER"),
        Some("hello ".to_string()),
        "'NEVER' is an unfinished prefix of 'NEVER_APPEARS' and must be \
         held back; 'hello ' is provably safe and emitted immediately"
    );
    assert_eq!(
        state.finish(),
        Some("NEVER".to_string()),
        "generation ended before 'NEVER' grew into the full stop \
         sequence, so it must be flushed, not dropped"
    );
    assert_eq!(
        state.finish(),
        None,
        "finish() must be idempotent -- nothing left to flush twice"
    );
}

#[test]
fn stream_decode_state_no_stop_sequences_emits_immediately() {
    // The common case (no `stop` configured): every token's text is
    // emitted immediately, with no artificial hold-back.
    let mut state = StreamDecodeState::new(&[], HashSet::new());
    assert_eq!(state.feed("hello"), Some("hello".to_string()));
    assert_eq!(state.feed(" world"), Some(" world".to_string()));
    assert_eq!(state.finish(), None);
}

// ── resolve_sampling_params: request fields over the engine defaults ─────

#[test]
fn resolve_sampling_params_seeds_from_engine_defaults_when_request_omits_fields() {
    let engine_defaults = SamplingParams {
        temperature: 0.55,
        top_k: 77,
        top_p: 0.8,
        repetition_penalty: 1.0,
        max_tokens: 999,
    };
    let resolved = resolve_sampling_params(&engine_defaults, None, None, None);
    assert_eq!(resolved.temperature, 0.55);
    assert_eq!(resolved.top_p, 0.8);
    assert_eq!(
        resolved.repetition_penalty, 1.0,
        "must come from the engine, never the hardcoded 1.1 literal nor \
         SamplingParams::default()"
    );
    assert_eq!(
        resolved.top_k, 77,
        "fields the request has no knob for must also come from the \
         engine, not SamplingParams::default()"
    );
}

#[test]
fn resolve_sampling_params_honours_client_overrides() {
    let engine_defaults = SamplingParams {
        temperature: 0.55,
        top_k: 77,
        top_p: 0.8,
        repetition_penalty: 1.0,
        max_tokens: 999,
    };
    let resolved = resolve_sampling_params(&engine_defaults, Some(0.0), Some(0.5), Some(1.3));
    assert_eq!(resolved.temperature, 0.0);
    assert_eq!(resolved.top_p, 0.5);
    assert_eq!(resolved.repetition_penalty, 1.3);
}

// ── HTTP-level regression tests ────────────────────────────────────────
//
// A fresh `Qwen3Config::tiny_test()` engine per test behind a
// tokenizer-less router (text prompts run as the configured prompt start
// token; the answer's text is empty while `usage` counts every token).

fn test_router() -> axum::Router {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    crate::tokenizer_bridge::chat_render::test_fixtures::tokenizerless_router(engine)
}

/// A short, unique-per-call string so tests sharing the process-global
/// [`idempotency_cache`] static cannot see each other's entries.
fn unique_test_key(label: &str) -> String {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{label}-{n}-{ts}")
}

async fn post_extended(
    app: axum::Router,
    body: serde_json::Value,
    idempotency_key: Option<&str>,
) -> axum::response::Response {
    let mut builder = axum::http::Request::post("/v1/chat/completions/extended")
        .header("content-type", "application/json");
    if let Some(key) = idempotency_key {
        builder = builder.header("idempotency-key", key);
    }
    let req = builder
        .body(axum::body::Body::from(
            serde_json::to_string(&body).expect("serialize request body"),
        ))
        .expect("build request");
    tower::ServiceExt::oneshot(app, req)
        .await
        .expect("send request")
}

// ── RT-05: real completion_tokens, not a word-count estimate ──────────────

#[tokio::test]
async fn extended_endpoint_reports_real_completion_tokens_not_a_word_count_estimate() {
    let app = test_router();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Count to ten please"}],
        "max_tokens": 2,
        "temperature": 0.0,
    });
    let resp = post_extended(app, body, None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse JSON");

    let finish_reason = json["choices"][0]["finish_reason"]
        .as_str()
        .expect("finish_reason");
    let completion_tokens = json["usage"]["completion_tokens"]
        .as_u64()
        .expect("completion_tokens");
    // RT-05's discriminator: with a tiny greedy model and max_tokens=2,
    // if the run exhausts the length limit (finish_reason == "length")
    // then *exactly* 2 tokens were generated -- a fact the old
    // whitespace-split estimate of the decoded text had no reason to
    // reproduce exactly.
    if finish_reason == "length" {
        assert_eq!(
            completion_tokens, 2,
            "finish_reason == \"length\" means exactly max_tokens tokens \
             were generated; a whitespace-split estimate of the decoded \
             text is very unlikely to equal that exactly"
        );
    } else {
        assert!(completion_tokens <= 2);
    }
}

/// A `stop` sequence that truncates the *visible* text partway through
/// separates the real token count from any estimate derived from the text:
/// the engine emits exactly the 15 byte tokens of `"1 2 3 4 5 6 7 8"`
/// (one token per character), `stop: [" 4"]` truncates the content to
/// `"1 2 3"` (3 words), and `completion_tokens` must still be 15 — a
/// whitespace-split estimate of either text (8 or 3 words) never is.
#[tokio::test]
async fn extended_endpoint_completion_tokens_survive_stop_sequence_truncation() {
    use crate::tokenizer_bridge::chat_render::test_fixtures as fx;
    let app = crate::server::create_router(
        fx::scripted_byte_engine("1 2 3 4 5 6 7 8"),
        Some(fx::byte_tokenizer()),
    );
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Count to ten please"}],
        "max_tokens": 15,
        "temperature": 0.0,
        "stop": [" 4"],
    });
    let resp = post_extended(app, body, None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse JSON");

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    let completion_tokens = json["usage"]["completion_tokens"]
        .as_u64()
        .expect("completion_tokens");
    assert_eq!(content, "1 2 3", "{json}");
    let word_count = content.split_whitespace().count() as u64;
    assert!(
        word_count < completion_tokens,
        "the 'stop' sequence must have truncated the visible content well \
         below the real generated-token count -- content={content:?} \
         word_count={word_count} completion_tokens={completion_tokens}; if \
         this ever holds with equality, RT-05 regressed back to a \
         whitespace-split estimate of the (truncated) text"
    );
    assert_eq!(
        completion_tokens, 15,
        "completion_tokens must count every token the engine actually \
         emitted, not a word-count of the stop-truncated final text"
    );
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

// ── RT-12: seed reaches the streaming path too ────────────────────────────

/// Extract just the generated-content-bearing part of an SSE body
/// (`choices`, which carries every delta and the final `finish_reason`),
/// dropping the per-request `id`/`created`/`model` fields that
/// legitimately differ between any two separate requests regardless of
/// seeding.
fn normalize_sse_deltas(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .map(|mut v| v["choices"].take())
        .collect()
}

/// Whether any delta in `normalize_sse_deltas`' output carries text: a
/// stream compared for equality must actually contain some, or two empty
/// streams would compare equal whatever the seed did.
fn deltas_carry_text(deltas: &[serde_json::Value]) -> bool {
    deltas.iter().any(|choices| {
        choices[0]["delta"]["content"]
            .as_str()
            .is_some_and(|text| !text.is_empty())
    })
}

#[tokio::test]
async fn extended_endpoint_stream_same_seed_produces_byte_identical_output() {
    // Constructing the two engines with *different* ambient seeds (1 and
    // 999) makes this test actually discriminate: if the fix regressed
    // to "the request `seed` is silently ignored", the two streams would
    // very likely diverge, each falling back to its own engine's ambient
    // seed, instead of matching. Every step is an equal choice among 26
    // letters of a byte-level vocabulary, so each sampled token is visible
    // text picked by the PRNG alone (a stream without a tokenizer carries
    // no text at all, which would make this comparison vacuous).
    async fn run(ambient_seed: u64, request_seed: u64) -> String {
        use crate::tokenizer_bridge::chat_render::test_fixtures as fx;
        let app = crate::server::create_router(
            fx::uniform_letters_engine(ambient_seed),
            Some(fx::byte_tokenizer()),
        );
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "max_tokens": 8,
            "temperature": 0.9,
            "stream": true,
            "seed": request_seed,
        });
        let resp = post_extended(app, body, None).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        String::from_utf8(bytes.to_vec()).expect("utf8 SSE body")
    }

    let a = run(1, 777).await;
    let b = run(999, 777).await;
    assert!(deltas_carry_text(&normalize_sse_deltas(&a)), "{a}");
    assert_ne!(
        normalize_sse_deltas(&a),
        normalize_sse_deltas(&run(1, 778).await),
        "a different request seed must stream different content"
    );
    assert_eq!(
        normalize_sse_deltas(&a),
        normalize_sse_deltas(&b),
        "two streams with the same request seed, on engines with \
         different ambient seeds, must produce identical generated \
         content -- proving the request seed actually drives the \
         streaming sampler (ids/timestamps legitimately differ per \
         request and are excluded from this comparison)"
    );
}

#[tokio::test]
async fn extended_endpoint_stream_without_seed_stays_deterministic_via_ambient_seed() {
    // RT-12's correction: omitting `seed` must leave the previous
    // (ambient-PRNG-driven) unseeded behavior completely unchanged --
    // still exactly as deterministic as before (driven by the engine's
    // own construction seed), never forcibly reset to a hardcoded
    // default seed nor made non-deterministic.
    async fn run() -> String {
        use crate::tokenizer_bridge::chat_render::test_fixtures as fx;
        let app = crate::server::create_router(
            fx::uniform_letters_engine(4242),
            Some(fx::byte_tokenizer()),
        );
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "temperature": 0.9,
            "stream": true,
        });
        let resp = post_extended(app, body, None).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        String::from_utf8(bytes.to_vec()).expect("utf8 SSE body")
    }

    let a = run().await;
    let b = run().await;
    assert!(deltas_carry_text(&normalize_sse_deltas(&a)), "{a}");
    assert_eq!(
        normalize_sse_deltas(&a),
        normalize_sse_deltas(&b),
        "an unseeded streaming request must stay deterministic via the \
         engine's own construction seed, unaffected by RT-12's fix \
         (ids/timestamps legitimately differ per request and are \
         excluded from this comparison)"
    );
}

// ── SV-25: metrics ─────────────────────────────────────────────────────

#[tokio::test]
async fn extended_endpoint_records_metrics() {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    let metrics = Arc::new(InferenceMetrics::new());
    let app =
        crate::tokenizer_bridge::chat_render::test_fixtures::tokenizerless_router_with_metrics(
            engine,
            Arc::clone(&metrics),
        );

    assert_eq!(metrics.requests_total.get(), 0);
    assert_eq!(metrics.active_requests.get(), 0.0);

    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
    });
    let resp = post_extended(app, body, None).await;
    assert_eq!(resp.status(), StatusCode::OK);

    assert_eq!(
        metrics.requests_total.get(),
        1,
        "SV-25: the extended endpoint must increment requests_total"
    );
    assert_eq!(metrics.errors_total.get(), 0);
    assert!(
        metrics.prompt_tokens_total.get() >= 1,
        "SV-25: prompt_tokens_total must be non-zero"
    );
    assert!(
        metrics.tokens_generated_total.get() >= 1,
        "SV-25: tokens_generated_total must be non-zero for a \
         non-streaming completion"
    );
    assert_eq!(
        metrics.active_requests.get(),
        0.0,
        "the guard must have decremented active_requests back to 0 \
         after the request completed"
    );
}

#[tokio::test]
async fn extended_endpoint_stream_records_metrics() {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    let metrics = Arc::new(InferenceMetrics::new());
    let app =
        crate::tokenizer_bridge::chat_render::test_fixtures::tokenizerless_router_with_metrics(
            engine,
            Arc::clone(&metrics),
        );

    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
        "stream": true,
    });
    let resp = post_extended(app, body, None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert!(!bytes.is_empty());

    assert_eq!(
        metrics.requests_total.get(),
        1,
        "SV-25: the streaming path must also increment requests_total"
    );
    // The `ActiveRequestGuard` moved into the stream-driver task is dropped
    // at the tail of that task's async block, in the same drop sequence as
    // `payload_tx` -- nothing orders that drop *before* the SSE body future
    // above finishes being read, so there is a theoretical window where the
    // client-visible body is fully read while the server-side task hasn't
    // unwound yet. Poll with a short bounded retry rather than a bare
    // equality assert so a benign scheduling delay can't flake this test
    // (the common case still resolves on the very first check, with no
    // added latency).
    let mut active = metrics.active_requests.get();
    for _ in 0..50 {
        if active == 0.0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        active = metrics.active_requests.get();
    }
    assert_eq!(
        active, 0.0,
        "the guard moved into the stream-driver task must decrement \
         active_requests once the stream finishes"
    );
}

// ── repetition_penalty validation ──────────────────────────────────────

#[tokio::test]
async fn extended_endpoint_rejects_non_positive_repetition_penalty() {
    for bad in [0.0f32, -1.0] {
        let app = test_router();
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "repetition_penalty": bad,
        });
        let resp = post_extended(app, body, None).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "repetition_penalty={bad} must be rejected, not silently \
             coerced"
        );
    }
}

// ── logprobs + sampling params: honoured on the logits-capturing path ───
//
// The request's sampling configuration is installed on the replica for the
// `generate_with_logprobs` call too (`RequestSampling`), so `logprobs`
// combines with `temperature` / `top_p` / `repetition_penalty` / `seed`.

/// `logprobs` with `temperature: 0` really samples greedily: every step of
/// the uniform-letters engine is an equal choice among `a`..=`z`, which only
/// a greedy draw (the lowest id wins a tie) turns into `"aaaa"`, while the
/// engine's ambient temperature (`0.9`) would draw random letters.
#[tokio::test]
async fn extended_endpoint_honours_logprobs_with_temperature_top_p_and_repetition_penalty() {
    use crate::tokenizer_bridge::chat_render::test_fixtures as fx;
    let app =
        crate::server::create_router(fx::uniform_letters_engine(5), Some(fx::byte_tokenizer()));
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "logprobs": true,
        "top_logprobs": 2,
        "temperature": 0.0,
        "top_p": 0.5,
        "repetition_penalty": 1.3,
    });
    let resp = post_extended(app, body, None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse JSON");
    assert_eq!(json["choices"][0]["message"]["content"], "aaaa", "{json}");
    let entries = json["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("logprobs.content array");
    assert_eq!(entries.len(), 4, "{json}");
}

#[tokio::test]
async fn extended_endpoint_allows_logprobs_with_frequency_or_presence_penalty() {
    // frequency_penalty and presence_penalty are honored on the logprobs
    // path, so this combination must succeed.
    let app = test_router();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "logprobs": true,
        "frequency_penalty": 0.5,
    });
    let resp = post_extended(app, body, None).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

// ── SV-32: idempotency ─────────────────────────────────────────────────

#[tokio::test]
async fn extended_endpoint_idempotency_key_replays_cached_response() {
    let app = test_router();
    let key = unique_test_key("replay");
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
    });

    let resp1 = post_extended(app.clone(), body.clone(), Some(&key)).await;
    assert_eq!(resp1.status(), StatusCode::OK);
    let bytes1 = axum::body::to_bytes(resp1.into_body(), usize::MAX)
        .await
        .expect("body1");

    let resp2 = post_extended(app, body, Some(&key)).await;
    assert_eq!(resp2.status(), StatusCode::OK);
    let bytes2 = axum::body::to_bytes(resp2.into_body(), usize::MAX)
        .await
        .expect("body2");

    assert_eq!(
        bytes1, bytes2,
        "SV-32: replaying the same Idempotency-Key + body must return \
         the exact cached response instead of re-running generation \
         (which, on this stochastic-by-default engine, would very \
         likely produce different output the second time)"
    );
}

#[tokio::test]
async fn extended_endpoint_idempotency_key_reused_with_different_body_is_a_cache_miss() {
    let app = test_router();
    let key = unique_test_key("diff-body");
    let body_a = serde_json::json!({
        "messages": [{"role": "user", "content": "prompt A"}],
        "max_tokens": 2,
    });
    let body_b = serde_json::json!({
        "messages": [{"role": "user", "content": "a very different prompt B"}],
        "max_tokens": 2,
    });

    let resp_a = post_extended(app.clone(), body_a, Some(&key)).await;
    assert_eq!(resp_a.status(), StatusCode::OK);
    let bytes_a = axum::body::to_bytes(resp_a.into_body(), usize::MAX)
        .await
        .expect("body a");
    let json_a: serde_json::Value = serde_json::from_slice(&bytes_a).expect("json a");

    let resp_b = post_extended(app, body_b, Some(&key)).await;
    assert_eq!(
        resp_b.status(),
        StatusCode::OK,
        "a different body under a reused key must still be served \
         (a cache miss), not blocked"
    );
    let bytes_b = axum::body::to_bytes(resp_b.into_body(), usize::MAX)
        .await
        .expect("body b");
    let json_b: serde_json::Value = serde_json::from_slice(&bytes_b).expect("json b");

    // `id` is freshly generated per real response (`rand_ext_id`), so
    // two independently-run generations always differ there -- this is
    // what actually proves generation ran twice rather than the cache
    // replaying client A's response for client B's different prompt
    // (the cross-client leak this test guards against).
    assert_ne!(
        json_a["id"], json_b["id"],
        "SV-32: a different request body under a reused Idempotency-Key \
         must never replay the other body's cached response"
    );
}

#[test]
fn idempotency_cache_key_differs_for_different_bodies_under_the_same_header() {
    let req_a: ExtendedChatRequest = serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
    }))
    .expect("parse a");
    let req_b: ExtendedChatRequest = serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "a completely different message"}],
    }))
    .expect("parse b");

    let key_a = idempotency_cache_key("same-header-value", &req_a);
    let key_b = idempotency_cache_key("same-header-value", &req_b);
    assert_ne!(
        key_a, key_b,
        "SV-32: the same Idempotency-Key header with two different \
         request bodies must produce two different cache keys"
    );
}

/// Regression test: `idempotency_cache_key` used to
/// fold in only `tools.is_some()` (never the tool definitions themselves)
/// and dropped `logprobs`/`top_logprobs`/`tool_choice`/`user` entirely, so
/// two requests differing *only* in one of those fields collided on the
/// same cache key -- ships-with-false-confidence territory identical to
/// the RT-05 finding this same review flagged, if left untested.
#[test]
fn idempotency_cache_key_differs_when_logprobs_or_tools_are_added() {
    let base = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
    });
    let req_plain: ExtendedChatRequest = serde_json::from_value(base.clone()).expect("parse plain");
    let key_plain = idempotency_cache_key("same-header-value", &req_plain);

    let mut with_logprobs = base.clone();
    with_logprobs["logprobs"] = serde_json::json!(true);
    let req_logprobs: ExtendedChatRequest =
        serde_json::from_value(with_logprobs).expect("parse logprobs");
    assert_ne!(
        key_plain,
        idempotency_cache_key("same-header-value", &req_logprobs),
        "logprobs must be folded into the cache key fingerprint"
    );

    let mut with_top_logprobs = base.clone();
    with_top_logprobs["top_logprobs"] = serde_json::json!(5);
    let req_top_logprobs: ExtendedChatRequest =
        serde_json::from_value(with_top_logprobs).expect("parse top_logprobs");
    assert_ne!(
        key_plain,
        idempotency_cache_key("same-header-value", &req_top_logprobs),
        "top_logprobs must be folded into the cache key fingerprint"
    );

    let mut with_user = base.clone();
    with_user["user"] = serde_json::json!("end-user-123");
    let req_user: ExtendedChatRequest = serde_json::from_value(with_user).expect("parse user");
    assert_ne!(
        key_plain,
        idempotency_cache_key("same-header-value", &req_user),
        "user must be folded into the cache key fingerprint"
    );

    let mut with_tools_a = base.clone();
    with_tools_a["tools"] = serde_json::json!([{
        "type": "function",
        "function": {"name": "get_weather", "parameters": {}}
    }]);
    let req_tools_a: ExtendedChatRequest =
        serde_json::from_value(with_tools_a).expect("parse tools a");
    let key_tools_a = idempotency_cache_key("same-header-value", &req_tools_a);
    assert_ne!(
        key_plain, key_tools_a,
        "adding any tool definition must change the cache key"
    );

    // The actual regression: two DIFFERENT tool sets, both non-empty, must
    // not collide -- the old code only ever hashed `tools.is_some()`, so
    // any two non-empty tool lists were indistinguishable to it.
    let mut with_tools_b = base;
    with_tools_b["tools"] = serde_json::json!([{
        "type": "function",
        "function": {"name": "send_email", "parameters": {}}
    }]);
    let req_tools_b: ExtendedChatRequest =
        serde_json::from_value(with_tools_b).expect("parse tools b");
    let key_tools_b = idempotency_cache_key("same-header-value", &req_tools_b);
    assert_ne!(
        key_tools_a, key_tools_b,
        "two different (both non-empty) tool definitions must produce \
         different cache keys, not just \"tools present vs absent\""
    );
}

/// End-to-end companion to the unit test above: a client that reuses an
/// `Idempotency-Key` and adds `logprobs: true` on the second call must get
/// a freshly generated, correctly-shaped response, never the first call's
/// cached (non-logprobs) body served back with a lying `200`.
#[tokio::test]
async fn extended_endpoint_idempotency_key_reused_with_logprobs_added_is_a_cache_miss() {
    let app = test_router();
    let key = unique_test_key("logprobs-fingerprint");
    let body_no_logprobs = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
    });
    let body_with_logprobs = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
        "logprobs": true,
    });

    let resp1 = post_extended(app.clone(), body_no_logprobs, Some(&key)).await;
    assert_eq!(resp1.status(), StatusCode::OK);
    let bytes1 = axum::body::to_bytes(resp1.into_body(), usize::MAX)
        .await
        .expect("body1");
    let json1: serde_json::Value = serde_json::from_slice(&bytes1).expect("json1");
    assert!(
        json1["choices"][0]["logprobs"].is_null(),
        "sanity check: the first request did not ask for logprobs, got {json1}"
    );

    let resp2 = post_extended(app, body_with_logprobs, Some(&key)).await;
    assert_eq!(
        resp2.status(),
        StatusCode::OK,
        "a request that adds logprobs: true under a reused key must still \
         be served fresh (a cache miss), not blocked"
    );
    let bytes2 = axum::body::to_bytes(resp2.into_body(), usize::MAX)
        .await
        .expect("body2");
    let json2: serde_json::Value = serde_json::from_slice(&bytes2).expect("json2");
    assert!(
        !json2["choices"][0]["logprobs"].is_null(),
        "a reused Idempotency-Key with logprobs: true added on replay must \
         never silently return the first request's cached \
         (non-logprobs-shaped) response -- got {json2}"
    );
}

// ── TOK-M2: control-token injection via the extended endpoint ───────────
//
// The raw-text `<|...|>`
// guard alone (`neutralize_special_markers`) never matches `<think>`,
// `</think>`, `<tool_call>`, or `</tool_call>` -- none of them contain
// `<|` -- even though each is a real, atomic control token in the shipped
// vocabularies (Bonsai 2's `token_type = 4` added tokens). Before this
// fix, a user message consisting of exactly one of those strings was
// tokenized straight through (`build_extended_prompt` + `tok.encode`) as
// the model's real control-token id, while the base
// `/v1/chat/completions` endpoint's vocabulary-driven `SpecialTokenGuard`
// silently dropped that same id: the two mounted endpoints disagreed
// about prompt-injection safety.

/// Build a tiny native tokenizer whose added vocabulary includes `<think>`
/// as a protected, control-shaped token -- the same class
/// `SpecialTokenGuard::from_tokenizer` guards by shape
/// (`is_control_token_content`: wrapped in `<...>`, no whitespace) even
/// when it is not flagged `special`, exactly like the real Bonsai 2
/// vocabulary's `token_type = 4` entries.
fn tokenizer_with_control_token() -> crate::tokenizer_bridge::TokenizerBridge {
    use oxibonsai_tokenizer::{BpeMerges, OxiTokenizer, TokenizerConfig, Vocabulary};

    let mut vocab = Vocabulary::new();
    vocab.add_special("<unk>", 0);
    vocab.add_special("<bos>", 1);
    vocab.add_special("<eos>", 2);
    vocab.add_special("<pad>", 3);
    // ChatML template pieces, atomic like the real vocab.
    vocab.add_protected("<|im_start|>", 4);
    vocab.add_protected("<|im_end|>", 5);
    // The guarded control token under test: shaped like a real added token,
    // never containing `<|`, so the raw-text guard alone cannot catch it --
    // only the vocabulary-driven carve-out can.
    vocab.add_protected("<think>", 6);

    // Plain printable ASCII so ordinary message text (roles, "hi", "user",
    // newlines) encodes without ever falling back to an unmapped char.
    let mut next_id = 10u32;
    for byte in 0x20u8..=0x7Eu8 {
        vocab.insert(&char::from(byte).to_string(), next_id);
        next_id += 1;
    }
    vocab.insert("\n", next_id);

    let tokenizer = OxiTokenizer::new(vocab, BpeMerges::new(), TokenizerConfig::default());
    crate::tokenizer_bridge::TokenizerBridge::from_native_tokenizer(tokenizer)
}

#[tokio::test]
async fn extended_endpoint_drops_guarded_control_token_in_message_content() {
    // Precondition: prove this is a real gap, not a vacuous test -- the
    // raw-text guard really does let `<think>` through untouched (it
    // contains no `<|`).
    assert_eq!(
        crate::server::neutralize_special_markers("<think>"),
        "<think>",
        "precondition: the raw-text `<|...|>` guard must NOT catch \
         `<think>` on its own -- that is exactly what makes TOK-M2 a real \
         gap rather than something the pre-existing guard already covered"
    );

    async fn prompt_tokens_for(content: &str) -> u64 {
        let config = oxibonsai_core::config::Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let metrics = Arc::new(InferenceMetrics::new());
        let app = crate::server::create_router_with_metrics(
            engine,
            Some(tokenizer_with_control_token()),
            Arc::clone(&metrics),
        );
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2,
            "temperature": 0.0,
        });
        let resp = post_extended(app, body, None).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "request with content {content:?} must succeed"
        );
        metrics.prompt_tokens_total.get()
    }

    // A message whose entire content is the guarded control token must
    // contribute exactly as many prompt tokens as an empty message: the
    // vocabulary-driven `SpecialTokenGuard` carve-out must drop the id
    // entirely, matching the base `/v1/chat/completions` endpoint.
    let baseline = prompt_tokens_for("").await;
    let with_control_token = prompt_tokens_for("<think>").await;
    assert_eq!(
        with_control_token, baseline,
        "TOK-M2: a message containing only a guarded control token \
         (`<think>`) must contribute zero prompt tokens beyond an empty \
         message -- the extended endpoint must drop it exactly like the \
         base /v1/chat/completions endpoint does, not encode it as the \
         model's real control-token id"
    );
}

// ── B1/B2/B3: real chat-template rendering + reasoning split wired into
//    the extended endpoint too (non-streaming AND streaming) ────────────

/// A vocabulary that DOES define `<think>`/`</think>` (as the shipped Qwen3
/// 1.7B/8B and Bonsai 2 vocabularies do; `tokenizer_with_control_token`
/// above does not, which is why it cannot double as this fixture) plus full
/// printable-ASCII coverage so ordinary message text round-trips.
fn think_capable_tokenizer() -> crate::tokenizer_bridge::TokenizerBridge {
    use oxibonsai_tokenizer::{BpeMerges, OxiTokenizer, TokenizerConfig, Vocabulary};

    let mut vocab = Vocabulary::new();
    vocab.add_special("<unk>", 0);
    vocab.add_special("<bos>", 1);
    vocab.add_special("<eos>", 2);
    vocab.add_special("<pad>", 3);
    vocab.add_protected("<|im_start|>", 4);
    vocab.add_protected("<|im_end|>", 5);
    vocab.add_protected("<think>", 6);
    vocab.add_protected("</think>", 7);

    let mut next_id = 10u32;
    for byte in 0x20u8..=0x7Eu8 {
        vocab.insert(&char::from(byte).to_string(), next_id);
        next_id += 1;
    }
    vocab.insert("\n", next_id);

    let tokenizer = OxiTokenizer::new(vocab, BpeMerges::new(), TokenizerConfig::default());
    crate::tokenizer_bridge::TokenizerBridge::from_native_tokenizer(tokenizer)
}

fn think_capable_router() -> axum::Router {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    crate::server::create_router(engine, Some(think_capable_tokenizer()))
}

/// The extended endpoint must render through the real Jinja engine (B1)
/// and stay a valid `200` end to end for a tokenizer whose vocabulary DOES
/// define `<think>`/`</think>` — `reasoning_content`, if present, must be
/// a well-formed string (the native `ChatMessage::reasoning_content` field;
/// see `extended_endpoint_returns_a_model_emitted_think_span_as_native_reasoning_content`
/// for its exact value on a scripted generation).
#[tokio::test]
async fn extended_endpoint_renders_through_the_real_template_with_a_think_capable_tokenizer() {
    let app = think_capable_router();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4
    });
    let resp = post_extended(app, body, None).await;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("valid JSON");
    assert_eq!(status, StatusCode::OK, "{json}");
    if let Some(rc) = json["choices"][0]["message"].get("reasoning_content") {
        assert!(rc.is_string(), "reasoning_content must be a string: {json}");
    }
}

/// Same tokenizer, streaming: the reasoning-split wiring of the extended
/// endpoint's stream must not crash and
/// must still produce a well-formed SSE stream ending in `[DONE]`.
#[tokio::test]
async fn extended_endpoint_stream_with_a_think_capable_tokenizer_is_well_formed_sse() {
    let app = think_capable_router();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "stream": true
    });
    let resp = post_extended(app, body, None).await;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body");
    let text = String::from_utf8_lossy(&bytes);
    assert_eq!(status, StatusCode::OK, "{text}");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "must terminate with [DONE]: {text}"
    );
    // Every SSE data line must be valid JSON (whether an ordinary content
    // delta or a `reasoning_content` one).
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            let _: serde_json::Value = serde_json::from_str(data)
                .unwrap_or_else(|e| panic!("SSE data line must be valid JSON ({e}): {data}"));
        }
    }
}

/// TOK-07/RT-09 (an unsupported construct must ERROR): `messages: []`
/// reaching the render layer must be an honest `400`/`500`-free error
/// response, not a silent bare prompt (the old `build_extended_prompt`
/// behavior) or a panic.
#[tokio::test]
async fn extended_endpoint_empty_messages_errors_honestly() {
    let app = think_capable_router();
    let body = serde_json::json!({ "messages": [], "max_tokens": 4 });
    let resp = post_extended(app, body, None).await;
    assert!(
        resp.status().is_client_error(),
        "empty messages must be a client error, got {}",
        resp.status()
    );
}

// ── B11/SV-11 wiring, exercised over the real HTTP route ──────────────
//
// Mirrors `server/chat.rs`'s identical tests: `chat_render.rs`'s own tests
// pin that `preprocess_message_content_and_reasoning` + `to_render_messages`
// + `render_chat_prompt` together deliver a raw body's `reasoning_content`
// into the actual rendered text; these confirm `extended_chat_completions`
// really does call that pre-pass and thread its result through, end to
// end, over the real route.

#[tokio::test]
async fn extended_endpoint_accepts_reasoning_content_on_a_replayed_assistant_turn() {
    let app = think_capable_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "user greets"},
            {"role": "user", "content": "thanks"}
        ],
        "max_tokens": 4
    });
    let resp = post_extended(app, body, None).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn extended_endpoint_flattens_a_text_only_vision_content_array() {
    let app = think_capable_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": " world"}
            ]}
        ],
        "max_tokens": 4
    });
    let resp = post_extended(app, body, None).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "a text-only content array must be flattened and accepted, not rejected at the schema level"
    );
}

#[tokio::test]
async fn extended_endpoint_rejects_an_image_url_content_part_honestly() {
    let app = think_capable_router();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}}
            ]}
        ],
        "max_tokens": 4
    });
    let resp = post_extended(app, body, None).await;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
}

// ── On the extended endpoint: native
//    `reasoning_content`, and tool calls gated on the vocabulary's own
//    `<tool_call>` id, over a scripted generation ─────────────────────────

mod scripted_extended {
    use super::*;
    use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

    const EXTENDED: &str = "/v1/chat/completions/extended";

    /// A router whose engine emits exactly `script` (then EOS) on every
    /// generation, over the marker-carrying byte-level vocabulary.
    fn scripted_router(script: Vec<u32>) -> axum::Router {
        let mut engine = fx::weightless_engine(fx::MARKER_VOCAB, SamplingParams::default(), 42);
        engine.script_generation(script);
        crate::server::create_router(engine, Some(fx::byte_tokenizer_with_markers()))
    }

    /// The model's own `<think>` + `reasoning` + `</think>` + `answer`.
    fn think_span(reasoning: &str, answer: &str) -> Vec<u32> {
        let mut ids = vec![fx::THINK_OPEN];
        ids.extend(fx::byte_ids(reasoning));
        ids.push(fx::THINK_CLOSE);
        ids.extend(fx::byte_ids(answer));
        ids
    }

    fn chat_body(stream: bool) -> serde_json::Value {
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 256,
            "stream": stream,
        })
    }

    async fn post_json(app: axum::Router, body: serde_json::Value) -> serde_json::Value {
        let (status, _, text) = fx::post(app, EXTENDED, body).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        serde_json::from_str(&text).expect("a JSON chat.completion")
    }

    #[tokio::test]
    async fn extended_endpoint_streams_a_model_emitted_think_span_as_reasoning_before_content() {
        let app = scripted_router(think_span("\nplan the answer\n", "\n\nThe answer."));
        let (status, _, body) = fx::post(app, EXTENDED, chat_body(true)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let mut order = Vec::new();
        for chunk in fx::sse_payloads(&body) {
            let delta = &chunk["choices"][0]["delta"];
            assert!(
                !(delta["reasoning_content"].is_string() && delta["content"].is_string()),
                "a delta carries one channel, never both: {chunk}"
            );
            if let Some(text) = delta["reasoning_content"].as_str() {
                order.push(("reasoning", text.to_string()));
            }
            if let Some(text) = delta["content"].as_str() {
                order.push(("content", text.to_string()));
            }
        }
        let last_reasoning = order
            .iter()
            .rposition(|(channel, _)| *channel == "reasoning")
            .expect("reasoning deltas");
        let first_content = order
            .iter()
            .position(|(channel, _)| *channel == "content")
            .expect("content deltas");
        assert!(last_reasoning < first_content, "{order:?}");
        let reasoning: String = order
            .iter()
            .filter(|(channel, _)| *channel == "reasoning")
            .map(|(_, text)| text.as_str())
            .collect();
        let content: String = order
            .iter()
            .filter(|(channel, _)| *channel == "content")
            .map(|(_, text)| text.as_str())
            .collect();
        assert_eq!(reasoning, "plan the answer\n");
        assert_eq!(content, "The answer.");
        assert!(
            !body.contains("<think>") && !body.contains("</think>"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn extended_endpoint_returns_a_model_emitted_think_span_as_native_reasoning_content() {
        let json = post_json(
            scripted_router(think_span("\nplan the answer\n", "\n\nThe answer.")),
            chat_body(false),
        )
        .await;
        let message = &json["choices"][0]["message"];
        assert_eq!(message["reasoning_content"], "plan the answer\n", "{json}");
        assert_eq!(message["content"], "The answer.", "{json}");
    }

    #[tokio::test]
    async fn extended_endpoint_omits_whitespace_only_reasoning_everywhere() {
        let script = think_span("\n \n\t\n", "\n\nHello.");
        let (status, _, body) =
            fx::post(scripted_router(script.clone()), EXTENDED, chat_body(true)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(
            fx::delta_texts(&body, "reasoning_content").is_empty(),
            "{body}"
        );
        assert_eq!(fx::delta_texts(&body, "content").concat(), "Hello.");

        let json = post_json(scripted_router(script), chat_body(false)).await;
        let message = &json["choices"][0]["message"];
        assert!(message.get("reasoning_content").is_none(), "{json}");
        assert_eq!(message["content"], "Hello.", "{json}");
    }

    fn tools_body() -> serde_json::Value {
        serde_json::json!({
            "messages": [{"role": "user", "content": "weather in Tokyo?"}],
            "max_tokens": 256,
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }}],
        })
    }

    const CALL_JSON: &str = "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n";

    /// Spelled out of ordinary tokens, `<tool_call>` is content;
    /// opened by the vocabulary's own token, the same call is parsed.
    #[tokio::test]
    async fn extended_endpoint_parses_a_tool_call_only_when_the_vocabulary_token_opens_it() {
        let mut spelled = fx::byte_ids("<tool_call>");
        spelled.extend(fx::byte_ids(CALL_JSON));
        spelled.extend(fx::byte_ids("</tool_call>"));
        let json = post_json(scripted_router(spelled), tools_body()).await;
        let choice = &json["choices"][0];
        assert!(
            choice
                .get("tool_calls")
                .is_none_or(serde_json::Value::is_null),
            "{json}"
        );
        assert_eq!(choice["finish_reason"], "stop", "{json}");
        let content = choice["message"]["content"].as_str().unwrap_or_default();
        assert!(content.contains("<tool_call>"), "{json}");

        let mut opened = vec![fx::TOOL_CALL_OPEN];
        opened.extend(fx::byte_ids(CALL_JSON));
        opened.push(fx::TOOL_CALL_CLOSE);
        let json = post_json(scripted_router(opened), tools_body()).await;
        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "{json}");
        let calls = choice["tool_calls"].as_array().expect("tool_calls");
        assert_eq!(calls.len(), 1, "{json}");
        assert_eq!(calls[0]["function"]["name"], "get_weather", "{json}");
    }
}
