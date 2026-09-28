//! Unit tests for [`crate::completions`].
//!
//! Attached to `completions.rs` as its `#[cfg(test)] mod tests` via
//! `#[path]`, so the tests keep full access to the module's private items
//! (`validate_completion_request`, `ValidatedRequest`, `build_completion_choice`,
//! `build_completion_logprobs`, ...) while `completions.rs` itself stays under
//! the workspace's 2000-line ceiling — mirrors `server/chat.rs` /
//! `server/chat_tests.rs` and `api_extensions.rs` / `api_extensions_tests.rs`.

use super::*;
use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

/// `Result::expect_err` requires `T: Debug` (to render the `Ok` value in
/// its own panic message); `ValidatedRequest` intentionally does not
/// derive `Debug` (it embeds `StopChecker`, which does not either since
/// it lives in `api_extensions.rs`, outside this file's ownership), so
/// this local helper extracts the error without that bound.
fn expect_err(result: Result<ValidatedRequest, ApiError>) -> ApiError {
    match result {
        Ok(_) => panic!("expected validation to reject the request, but it succeeded"),
        Err(e) => e,
    }
}

fn base_request(prompt: PromptInput) -> CompletionRequest {
    CompletionRequest {
        model: None,
        prompt,
        max_tokens: 16,
        temperature: None,
        top_p: None,
        n: None,
        stream: None,
        stream_options: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        repetition_penalty: None,
        logprobs: None,
        echo: None,
        seed: None,
        suffix: None,
        user: None,
        best_of: None,
        logit_bias: None,
        top_k: None,
        min_p: None,
    }
}

#[test]
fn prompt_input_single_as_strings() {
    let p = PromptInput::Single("hello world".to_string());
    assert_eq!(p.as_strings(), vec!["hello world"]);
}

#[test]
fn prompt_input_batch_as_strings() {
    let p = PromptInput::Batch(vec!["foo".to_string(), "bar".to_string()]);
    assert_eq!(p.as_strings(), vec!["foo", "bar"]);
}

#[test]
fn prompt_input_single_first() {
    let p = PromptInput::Single("hello".to_string());
    assert_eq!(p.first(), "hello");
}

#[test]
fn prompt_input_batch_first() {
    let p = PromptInput::Batch(vec!["alpha".to_string(), "beta".to_string()]);
    assert_eq!(p.first(), "alpha");
}

#[test]
fn prompt_input_empty_batch_first() {
    let p = PromptInput::Batch(vec![]);
    assert_eq!(p.first(), "");
}

#[test]
fn build_completion_response_no_echo() {
    let choice = build_completion_choice(ChoiceInputs {
        index: 0,
        prompt: "Say hello",
        completion: " world",
        echo: false,
        completion_tokens: 2,
        max_tokens: 16,
        hit_stop: false,
        logprobs: None,
    });
    let resp = build_completion_response("cmpl-abc", "bonsai-8b", 1_000_000, vec![choice], 4, 2);
    assert_eq!(resp.object, "text_completion");
    assert_eq!(resp.choices[0].text, " world");
    assert_eq!(resp.usage.prompt_tokens, 4);
    assert_eq!(resp.usage.completion_tokens, 2);
    assert_eq!(resp.usage.total_tokens, 6);
}

#[test]
fn build_completion_response_with_echo() {
    let choice = build_completion_choice(ChoiceInputs {
        index: 0,
        prompt: "Say hello",
        completion: " world",
        echo: true,
        completion_tokens: 2,
        max_tokens: 16,
        hit_stop: false,
        logprobs: None,
    });
    let resp = build_completion_response("cmpl-abc", "bonsai-8b", 1_000_000, vec![choice], 4, 2);
    assert_eq!(resp.choices[0].text, "Say hello world");
}

#[test]
fn build_completion_response_id_preserved() {
    let choice = build_completion_choice(ChoiceInputs {
        index: 0,
        prompt: "prompt",
        completion: "completion",
        echo: false,
        completion_tokens: 1,
        max_tokens: 16,
        hit_stop: false,
        logprobs: None,
    });
    let resp = build_completion_response("cmpl-xyz", "bonsai-8b", 42, vec![choice], 1, 1);
    assert_eq!(resp.id, "cmpl-xyz");
    assert_eq!(resp.created, 42);
}

/// Regression test for finding 31: `build_completion_choice` must derive
/// `finish_reason` from the *caller-supplied* `max_tokens`, not the
/// hardcoded literal `16` the field defaults to. A request that
/// overrides `max_tokens` away from `16` and is truncated exactly at
/// that limit must report `"length"`, not `"stop"`.
#[test]
fn build_completion_response_uses_real_max_tokens_for_finish_reason() {
    // max_tokens = 5, completion_tokens = 5 (exhausted the limit) ->
    // "length". Under the old hardcoded-16 bug this would incorrectly
    // report "stop" because 5 < 16.
    let truncated = build_completion_choice(ChoiceInputs {
        index: 0,
        prompt: "prompt",
        completion: "completion",
        echo: false,
        completion_tokens: 5,
        max_tokens: 5,
        hit_stop: false,
        logprobs: None,
    });
    assert_eq!(truncated.finish_reason, "length");

    // max_tokens = 100, completion_tokens = 30 (stopped early on EOS,
    // well under the limit) -> "stop". Under the old hardcoded-16 bug
    // this would incorrectly report "length" because 30 >= 16.
    let natural_stop = build_completion_choice(ChoiceInputs {
        index: 0,
        prompt: "prompt",
        completion: "completion",
        echo: false,
        completion_tokens: 30,
        max_tokens: 100,
        hit_stop: false,
        logprobs: None,
    });
    assert_eq!(natural_stop.finish_reason, "stop");
}

/// A stop-sequence hit always reports `"stop"`, even if (by construction
/// of the caller) `completion_tokens >= max_tokens` — the point of
/// `hit_stop` is that it wins over the length-based determination.
#[test]
fn hit_stop_forces_stop_finish_reason_even_at_the_token_limit() {
    let choice = build_completion_choice(ChoiceInputs {
        index: 0,
        prompt: "prompt",
        completion: "trunc",
        echo: false,
        completion_tokens: 16,
        max_tokens: 16,
        hit_stop: true,
        logprobs: None,
    });
    assert_eq!(choice.finish_reason, "stop");
}

/// Regression test for finding serve-api-09: every prompt in a batch
/// must produce its own [`CompletionChoice`] with a matching `index`,
/// not just the first one.
#[test]
fn build_completion_response_batch_has_one_choice_per_prompt() {
    let choices = vec![
        build_completion_choice(ChoiceInputs {
            index: 0,
            prompt: "first",
            completion: "alpha",
            echo: false,
            completion_tokens: 1,
            max_tokens: 16,
            hit_stop: false,
            logprobs: None,
        }),
        build_completion_choice(ChoiceInputs {
            index: 1,
            prompt: "second",
            completion: "beta",
            echo: false,
            completion_tokens: 1,
            max_tokens: 16,
            hit_stop: false,
            logprobs: None,
        }),
        build_completion_choice(ChoiceInputs {
            index: 2,
            prompt: "third",
            completion: "gamma",
            echo: false,
            completion_tokens: 1,
            max_tokens: 16,
            hit_stop: false,
            logprobs: None,
        }),
    ];
    let resp = build_completion_response("cmpl-batch", "bonsai-8b", 1, choices, 3, 3);
    assert_eq!(resp.choices.len(), 3, "one choice per batch prompt");
    assert_eq!(resp.choices[0].index, 0);
    assert_eq!(resp.choices[0].text, "alpha");
    assert_eq!(resp.choices[1].index, 1);
    assert_eq!(resp.choices[1].text, "beta");
    assert_eq!(resp.choices[2].index, 2);
    assert_eq!(resp.choices[2].text, "gamma");
}

#[test]
fn determine_finish_reason_stop() {
    assert_eq!(determine_finish_reason(8, 16, false), "stop");
}

#[test]
fn determine_finish_reason_length() {
    assert_eq!(determine_finish_reason(16, 16, false), "length");
}

#[test]
fn determine_finish_reason_hit_stop_overrides_length() {
    assert_eq!(determine_finish_reason(16, 16, true), "stop");
}

#[test]
fn completion_id_from_nanos_nonempty() {
    let id = completion_id_from_nanos();
    assert!(!id.is_empty());
}

#[test]
fn unix_timestamp_secs_nonzero() {
    let ts = unix_timestamp_secs();
    // Any reasonable Unix timestamp will be well above 0
    assert!(ts > 1_000_000_000);
}

#[test]
fn serialise_completion_response() {
    let choice = build_completion_choice(ChoiceInputs {
        index: 0,
        prompt: "prompt",
        completion: "result",
        echo: false,
        completion_tokens: 5,
        max_tokens: 16,
        hit_stop: false,
        logprobs: None,
    });
    let resp = build_completion_response("cmpl-test", "bonsai-8b", 99, vec![choice], 3, 5);
    let json = serde_json::to_string(&resp).expect("serialisation must succeed");
    assert!(json.contains("\"object\":\"text_completion\""));
    assert!(json.contains("\"finish_reason\""));
}

// ── build_completion_logprobs ────────────────────────────────────────────

fn logprobs_content(token: &str, logprob: f32) -> LogprobsContent {
    LogprobsContent {
        id: 0,
        token: token.to_string(),
        logprob,
        bytes: None,
        top_logprobs: vec![],
    }
}

#[test]
fn build_completion_logprobs_untruncated_keeps_every_token() {
    let content = vec![logprobs_content("ab", -0.1), logprobs_content("cd", -0.2)];
    // "abcd" is 4 chars, nothing truncated.
    let logprobs = build_completion_logprobs(&content, 4, 0);
    assert_eq!(logprobs.tokens, vec!["ab", "cd"]);
    assert_eq!(logprobs.token_logprobs, vec![-0.1, -0.2]);
    assert_eq!(logprobs.text_offset, vec![0, 2]);
}

#[test]
fn build_completion_logprobs_applies_base_offset_for_echo() {
    let content = vec![logprobs_content("hi", -0.1)];
    let logprobs = build_completion_logprobs(&content, 2, 10);
    assert_eq!(logprobs.text_offset, vec![10]);
}

#[test]
fn build_completion_logprobs_drops_tokens_past_the_stop_truncation() {
    let content = vec![
        logprobs_content("ab", -0.1),
        logprobs_content("cd", -0.2),
        logprobs_content("ef", -0.3),
    ];
    // Only the first 3 characters ("abc") survived stop-truncation, so
    // the second token (starting at char 2, "cd") is still partially
    // visible and kept, but the third ("ef", starting at char 4) must be
    // dropped entirely.
    let logprobs = build_completion_logprobs(&content, 3, 0);
    assert_eq!(logprobs.tokens, vec!["ab", "cd"]);
    assert_eq!(logprobs.token_logprobs, vec![-0.1, -0.2]);
}

#[test]
fn build_completion_logprobs_zero_length_truncation_drops_everything() {
    let content = vec![logprobs_content("ab", -0.1)];
    let logprobs = build_completion_logprobs(&content, 0, 0);
    assert!(logprobs.tokens.is_empty());
    assert!(logprobs.token_logprobs.is_empty());
    assert!(logprobs.text_offset.is_empty());
}

#[test]
fn build_completion_logprobs_top_logprobs_are_json_objects() {
    let mut content = logprobs_content("a", -0.05);
    content.top_logprobs = vec![
        crate::api_types::TopLogprob {
            id: 0,
            token: "a".to_string(),
            logprob: -0.05,
            bytes: None,
        },
        crate::api_types::TopLogprob {
            id: 1,
            token: "b".to_string(),
            logprob: -1.2,
            bytes: None,
        },
    ];
    let logprobs = build_completion_logprobs(&[content], 1, 0);
    let obj = logprobs.top_logprobs[0]
        .as_object()
        .expect("top_logprobs entry must be a JSON object");
    // `top.logprob` is `f32`; round-tripping it through
    // `serde_json::json!` widens it to `f64`, so compare after narrowing
    // back to `f32` rather than against an `f64` literal (which is not
    // bit-identical to the widened `f32` value).
    assert_eq!(
        obj.get("a")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32),
        Some(-0.05_f32)
    );
    assert_eq!(
        obj.get("b")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32),
        Some(-1.2_f32)
    );
}

// ── validate_completion_request ──────────────────────────────────────────

#[test]
fn validate_accepts_a_plain_request() {
    let req = base_request(PromptInput::Single("hello".to_string()));
    let validated = validate_completion_request(req).expect("must validate");
    assert_eq!(validated.prompts, vec!["hello".to_string()]);
    assert!(!validated.custom_sampling);
    assert!(validated.seed.is_none());
    assert!(validated.logprobs_top_k.is_none());
}

#[test]
fn validate_rejects_max_tokens_zero() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.max_tokens = 0;
    let err = expect_err(validate_completion_request(req));
    assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[test]
fn validate_rejects_max_tokens_over_the_ceiling() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.max_tokens = MAX_OUTPUT_TOKENS + 1;
    assert!(validate_completion_request(req).is_err());
}

#[test]
fn validate_rejects_n_other_than_one() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.n = Some(2);
    assert!(validate_completion_request(req).is_err());
}

#[test]
fn validate_accepts_n_equal_to_one() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.n = Some(1);
    assert!(validate_completion_request(req).is_ok());
}

#[test]
fn validate_rejects_empty_batch() {
    let req = base_request(PromptInput::Batch(vec![]));
    assert!(validate_completion_request(req).is_err());
}

#[test]
fn validate_rejects_batch_over_the_cap() {
    let prompts: Vec<String> = (0..(MAX_COMPLETION_BATCH_SIZE + 1))
        .map(|i| format!("p{i}"))
        .collect();
    let req = base_request(PromptInput::Batch(prompts));
    assert!(validate_completion_request(req).is_err());
}

/// A bare `stream: true` with a single prompt validates to the streaming
/// branch.
#[test]
fn validate_accepts_stream_true_alone() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.stream = Some(true);
    let validated = validate_completion_request(req).expect("stream: true alone must validate");
    assert!(validated.stream);
}

/// A batched prompt streams (one SSE stream, chunks indexed by prompt), so
/// `stream: true` with a batch validates to the streaming branch with every
/// prompt kept — see `completions::stream::tests`'s `batched_prompt_stream_*`
/// for the end-to-end proof.
#[test]
fn batched_prompt_stream_validates_to_the_streaming_branch() {
    let mut req = base_request(PromptInput::Batch(vec!["a".to_string(), "b".to_string()]));
    req.stream = Some(true);
    let validated = validate_completion_request(req).expect("a streamed batch must validate");
    assert!(validated.stream);
    assert_eq!(validated.prompts, vec!["a".to_string(), "b".to_string()]);
}

/// `stream` and `logprobs` together are ACCEPTED by validation (honoured
/// for real by the streaming `logprobs` loop — see `completions::stream`'s
/// own tests for the end-to-end chunk-shape proof).
#[test]
fn validate_accepts_stream_with_logprobs() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.stream = Some(true);
    req.logprobs = Some(2);
    let validated = validate_completion_request(req).expect("must validate");
    assert!(validated.stream);
    assert_eq!(validated.logprobs_top_k, Some(2));
}

/// `stream` + `seed` is honoured (a freshly seeded
/// sampler drives the stream — see `completions::stream`'s
/// `stream_with_a_seed_is_reproducible_and_seed_dependent`), so validation
/// must accept it and carry the seed through.
#[test]
fn validate_accepts_stream_with_seed() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.stream = Some(true);
    req.seed = Some(7);
    let validated = match validate_completion_request(req) {
        Ok(validated) => validated,
        Err(e) => panic!("stream + seed must validate, got {}", e.status()),
    };
    assert!(validated.stream);
    assert_eq!(validated.seed, Some(7), "the seed must reach the stream");
}

/// `stream: false` is the common, explicit-default case and must not be
/// rejected.
#[test]
fn validate_accepts_stream_false() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.stream = Some(false);
    assert!(validate_completion_request(req).is_ok());
}

/// `stream` omitted entirely must not be rejected.
#[test]
fn validate_accepts_stream_omitted() {
    let req = base_request(PromptInput::Single("hi".to_string()));
    assert!(validate_completion_request(req).is_ok());
}

/// A non-empty `suffix` is rejected naming the field (RT-32).
#[test]
fn validate_rejects_nonempty_suffix() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.suffix = Some("the end".to_string());
    assert!(validate_completion_request(req).is_err());
}

/// An empty-string `suffix` is a no-op, not a rejection — a client that
/// always sends `suffix: ""` must not be broken by this fix.
#[test]
fn validate_accepts_empty_suffix() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.suffix = Some(String::new());
    assert!(validate_completion_request(req).is_ok());
}

#[test]
fn validate_rejects_frequency_penalty_out_of_range() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.frequency_penalty = Some(3.0);
    assert!(validate_completion_request(req).is_err());
}

#[test]
fn validate_rejects_presence_penalty_out_of_range() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.presence_penalty = Some(-3.0);
    assert!(validate_completion_request(req).is_err());
}

#[test]
fn validate_rejects_non_finite_temperature() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.temperature = Some(f32::NAN);
    assert!(validate_completion_request(req).is_err());
}

#[test]
fn validate_rejects_top_p_out_of_range() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.top_p = Some(0.0);
    assert!(validate_completion_request(req).is_err());
}

#[test]
fn validate_marks_custom_sampling_when_temperature_set() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.temperature = Some(0.5);
    let validated = validate_completion_request(req).expect("must validate");
    assert!(validated.custom_sampling);
}

/// B9 correction: `seed` and `logprobs` together are now ACCEPTED by
/// validation (honoured for real via the sampler swap in
/// `create_completion`'s logprobs branch — see the handler-level
/// determinism test further down for the end-to-end proof).
#[test]
fn validate_accepts_seed_and_logprobs_together() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.seed = Some(42);
    req.logprobs = Some(3);
    let validated = validate_completion_request(req).expect("must validate");
    assert_eq!(validated.seed, Some(42));
    assert_eq!(validated.logprobs_top_k, Some(3));
}

/// B9 correction: `logprobs` and `temperature` together are now ACCEPTED —
/// `generate_with_logprobs`'s sampler is swapped for the resolved params
/// around the call (see the handler-level test further down).
#[test]
fn validate_accepts_logprobs_and_temperature_together() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.logprobs = Some(3);
    req.temperature = Some(0.1);
    assert!(validate_completion_request(req).is_ok());
}

/// Same correction, for `top_p` instead of `temperature`.
#[test]
fn validate_accepts_logprobs_and_top_p_together() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.logprobs = Some(3);
    req.top_p = Some(0.5);
    assert!(validate_completion_request(req).is_ok());
}

/// `logprobs` combined with only frequency/presence penalties (no
/// `temperature`/`top_p`) must still be ACCEPTED: penalties are honoured
/// on the logprobs path via `set_penalties` before the logits-capturing
/// decode loop runs, so there is no dropped field to guard against here
/// — the new guard must not over-reject this working combination.
#[test]
fn validate_accepts_logprobs_and_penalties_together() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.logprobs = Some(3);
    req.frequency_penalty = Some(0.5);
    let validated = validate_completion_request(req).expect("must validate");
    assert_eq!(validated.logprobs_top_k, Some(3));
    assert!(validated.penalties.is_active());
}

#[test]
fn validate_accepts_seed_alone() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.seed = Some(42);
    let validated = validate_completion_request(req).expect("must validate");
    assert_eq!(validated.seed, Some(42));
}

#[test]
fn validate_accepts_logprobs_alone() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.logprobs = Some(3);
    let validated = validate_completion_request(req).expect("must validate");
    assert_eq!(validated.logprobs_top_k, Some(3));
}

#[test]
fn validate_stop_sequences_reach_the_stop_checker() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.stop = Some(StopSequences::Single("STOP".to_string()));
    let validated = validate_completion_request(req).expect("must validate");
    assert!(!validated.stop_checker.is_empty());
    let (truncated, hit) = validated.stop_checker.truncate_at_stop("hello STOP world");
    assert!(hit);
    assert_eq!(truncated, "hello ");
}

#[test]
fn validate_no_stop_sequences_yields_an_empty_stop_checker() {
    let req = base_request(PromptInput::Single("hi".to_string()));
    let validated = validate_completion_request(req).expect("must validate");
    assert!(validated.stop_checker.is_empty());
}

/// Regression: `StopChecker::truncate_at_stop` does `text.find("")`,
/// which matches at position 0 of *any* string — an unfiltered empty
/// stop sequence would truncate every completion to `""`. Before this
/// fix `stop` was never read at all, so `stop: [""]` was harmless by
/// omission; now that `stop` is honoured, an empty entry must be
/// filtered out rather than newly breaking every response that sets it
/// (mirroring `pipeline.rs`'s own `StopMatcher`, which drops empty
/// strings for the identical reason).
#[test]
fn validate_filters_out_an_empty_stop_sequence() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.stop = Some(StopSequences::Single(String::new()));
    let validated = validate_completion_request(req).expect("must validate");
    assert!(
        validated.stop_checker.is_empty(),
        "an empty stop sequence must be dropped, not installed"
    );
    let (truncated, hit) = validated.stop_checker.truncate_at_stop("hello world");
    assert!(!hit, "an empty stop sequence must never match");
    assert_eq!(truncated, "hello world");
}

/// Same regression, in a batch that mixes an empty entry with a real one
/// — only the empty entry is dropped, the real one still works.
#[test]
fn validate_filters_empty_stop_sequence_but_keeps_real_ones() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.stop = Some(StopSequences::Multiple(vec![
        String::new(),
        "STOP".to_string(),
    ]));
    let validated = validate_completion_request(req).expect("must validate");
    assert!(!validated.stop_checker.is_empty());
    let (truncated, hit) = validated.stop_checker.truncate_at_stop("hello STOP world");
    assert!(hit);
    assert_eq!(truncated, "hello ");
}

// ── Handler-level (end-to-end) tests ─────────────────────────────────────
//
// Everything above exercises `validate_completion_request` and the pure
// helper functions directly. The three headline `RT-32` / `SV-22`
// behaviors — logprobs actually populated (not always `null`), `stream:
// true` actually rejected, and a stop sequence actually truncating the
// *decoded* text — only really live in the handler itself (the
// `run_blocking_generation` closure, the post-generation decode +
// `StopChecker::truncate_at_stop` pass), which the pure-function tests
// cannot reach. These build a real router with the workspace's tiny test
// model, the same pattern the sibling (unowned)
// `tests/completions_tests.rs` integration suite and
// `server/blocking.rs`'s own tests both already use.

/// The tiny test model behind a tokenizer-less router: a text prompt runs
/// as the configured prompt start token, and the completion text is empty
/// (no tokenizer to render it) while `usage` and `logprobs` still count
/// every generated token.
fn test_router() -> axum::Router {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let engine = crate::engine::InferenceEngine::new(config, params, 42);
    fx::tokenizerless_router(engine)
}

/// The generated ids of choice 0 as a tokenizer-less server reports them in
/// `logprobs.tokens` (`"<id>"` strings).
fn logprob_tokens(json: &serde_json::Value) -> Vec<String> {
    json["choices"][0]["logprobs"]["tokens"]
        .as_array()
        .map(|tokens| {
            tokens
                .iter()
                .filter_map(|token| token.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// POST `body` to `/v1/completions` on `app` and return (status, JSON).
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
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

/// `logprobs` must actually be populated, not the `null` SV-22 named.
/// With no tokenizer configured (`test_router`'s `None`), per-token
/// decoding falls back to `<id>` strings, so the exact token count
/// varies with the tiny test model's own (deterministic, but
/// implementation-detail) generation length — assert the field is
/// *present* and internally consistent (equal-length parallel arrays),
/// not an exact count.
#[tokio::test]
async fn handler_logprobs_field_is_populated_not_null() {
    let app = test_router();
    let (status, json) = post_completion(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3, "logprobs": 2 }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);

    let logprobs = &json["choices"][0]["logprobs"];
    assert!(
        !logprobs.is_null(),
        "logprobs must be populated when requested, not always null (SV-22); got {json}"
    );
    let tokens = logprobs["tokens"].as_array().expect("tokens array");
    let token_logprobs = logprobs["token_logprobs"]
        .as_array()
        .expect("token_logprobs array");
    let top_logprobs = logprobs["top_logprobs"]
        .as_array()
        .expect("top_logprobs array");
    let text_offset = logprobs["text_offset"]
        .as_array()
        .expect("text_offset array");
    assert_eq!(tokens.len(), token_logprobs.len());
    assert_eq!(tokens.len(), top_logprobs.len());
    assert_eq!(tokens.len(), text_offset.len());
}

/// A request without `logprobs` must still report `null` — the fix adds
/// the field, it does not turn it on unconditionally.
#[tokio::test]
async fn handler_logprobs_is_still_null_when_not_requested() {
    let app = test_router();
    let (status, json) = post_completion(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3 }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(json["choices"][0]["logprobs"].is_null());
}

/// `stream: true` alone resolves to real SSE: status is `200 OK`, and the body is SSE text (not the non-streaming JSON
/// object shape), so `post_completion`'s `serde_json::from_slice` on it
/// falls back to `Value::Null`. The detailed chunk-shape assertions
/// (`text_completion` chunks, `[DONE]`, `stream_options.include_usage`)
/// live in `stream::tests`, which reads the raw body as text rather
/// than attempting a JSON parse.
#[tokio::test]
async fn handler_stream_true_alone_is_ok_not_400() {
    let app = test_router();
    let (status, json) = post_completion(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(
        json,
        serde_json::Value::Null,
        "an SSE body must not parse as a single JSON document"
    );
}

/// `stream: true` with `logprobs` is real SSE end to end — see
/// `completions::stream::tests` for the detailed, byte-level SSE chunk-shape
/// proof.
#[tokio::test]
async fn handler_stream_with_logprobs_is_ok_not_400() {
    let app = test_router();
    let (status, json) = post_completion(
        app,
        serde_json::json!({
            "prompt": "hello", "max_tokens": 3, "stream": true, "logprobs": 2
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    assert_eq!(
        json,
        serde_json::Value::Null,
        "an SSE body must not parse as a single JSON document"
    );
}

/// `logprobs` combined with `temperature` is ACCEPTED end-to-end and the
/// override is GENUINELY applied to the logprobs decode loop, not silently
/// dropped while returning `200` — proven two-sided: the router's generated
/// ids (read from `logprobs.tokens`) must match a direct
/// `generate_with_logprobs` call with the override applied (same seed, same
/// config) and must NOT match one that kept the engine's distinct ambient
/// temperature instead.
#[tokio::test]
async fn handler_logprobs_and_temperature_together_is_honoured_not_rejected() {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let ambient = SamplingParams {
        temperature: 0.9, // distinctive; NOT the request's override below
        ..SamplingParams::default()
    };
    let fallback_prompt = vec![fx::QWEN3_IM_START];
    let id_to_token = |id: u32| format!("<{id}>");

    let mut with_override =
        crate::engine::InferenceEngine::new(config.clone(), ambient.clone(), 42);
    with_override.sampler.set_params(SamplingParams {
        temperature: 0.0,
        ..ambient.clone()
    });
    let (override_tokens, _) = with_override
        .generate_with_logprobs(&fallback_prompt, 3, 2, &id_to_token)
        .expect("reference generation (override applied)");
    let override_ids = fx::id_token_strings(&override_tokens);

    let mut without_override =
        crate::engine::InferenceEngine::new(config.clone(), ambient.clone(), 42);
    let (ambient_tokens, _) = without_override
        .generate_with_logprobs(&fallback_prompt, 3, 2, &id_to_token)
        .expect("reference generation (ambient, unmodified)");
    assert_ne!(
        override_ids,
        fx::id_token_strings(&ambient_tokens),
        "sanity: the override must actually change the deterministic output, \
         or this test cannot discriminate anything"
    );

    let router_engine = crate::engine::InferenceEngine::new(config, ambient, 42);
    let app = fx::tokenizerless_router(router_engine);
    let (status, json) = post_completion(
        app,
        serde_json::json!({
            "prompt": "hello", "max_tokens": 3, "logprobs": 2, "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(
        status,
        axum::http::StatusCode::OK,
        "must be accepted, not rejected: {json}"
    );
    assert!(
        !json["choices"][0]["logprobs"].is_null(),
        "logprobs must still be populated: {json}"
    );
    assert_eq!(
        logprob_tokens(&json),
        override_ids,
        "the temperature override must reach the logprobs decode loop, not the ambient sampler"
    );
}

/// Same correction, for `top_p` instead of `temperature`.
#[tokio::test]
async fn handler_logprobs_and_top_p_together_is_honoured_not_rejected() {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let ambient = SamplingParams {
        temperature: 0.8,
        top_p: 1.0, // distinctive; NOT the request's override below
        ..SamplingParams::default()
    };
    let fallback_prompt = vec![fx::QWEN3_IM_START];
    let id_to_token = |id: u32| format!("<{id}>");

    let mut with_override =
        crate::engine::InferenceEngine::new(config.clone(), ambient.clone(), 42);
    with_override.sampler.set_params(SamplingParams {
        top_p: 0.05,
        ..ambient.clone()
    });
    let (override_tokens, _) = with_override
        .generate_with_logprobs(&fallback_prompt, 3, 2, &id_to_token)
        .expect("reference generation (override applied)");
    let override_ids = fx::id_token_strings(&override_tokens);

    let mut without_override =
        crate::engine::InferenceEngine::new(config.clone(), ambient.clone(), 42);
    let (ambient_tokens, _) = without_override
        .generate_with_logprobs(&fallback_prompt, 3, 2, &id_to_token)
        .expect("reference generation (ambient, unmodified)");
    assert_ne!(
        override_ids,
        fx::id_token_strings(&ambient_tokens),
        "sanity: the override must actually change the deterministic output, \
         or this test cannot discriminate anything"
    );

    let router_engine = crate::engine::InferenceEngine::new(config, ambient, 42);
    let app = fx::tokenizerless_router(router_engine);
    let (status, json) = post_completion(
        app,
        serde_json::json!({
            "prompt": "hello", "max_tokens": 3, "logprobs": 2, "top_p": 0.05
        }),
    )
    .await;
    assert_eq!(
        status,
        axum::http::StatusCode::OK,
        "must be accepted, not rejected: {json}"
    );
    assert_eq!(
        logprob_tokens(&json),
        override_ids,
        "the top_p override must reach the logprobs decode loop, not the ambient sampler"
    );
}

/// B9: `seed` combined with `logprobs` is now honoured end-to-end — two
/// requests with the SAME explicit `seed` must produce byte-identical
/// output, proving a fresh seeded sampler (not the engine's own, freely
/// advancing PRNG) actually drove the logprobs decode loop.
#[tokio::test]
async fn handler_logprobs_and_seed_together_is_deterministic() {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let params = SamplingParams {
        temperature: 0.85,
        ..SamplingParams::default()
    };
    let body = serde_json::json!({
        "prompt": "hello", "max_tokens": 4, "logprobs": 2, "seed": 777
    });

    let engine1 = crate::engine::InferenceEngine::new(config.clone(), params.clone(), 1);
    let (status1, json1) = post_completion(fx::tokenizerless_router(engine1), body.clone()).await;
    let engine2 = crate::engine::InferenceEngine::new(config, params, 99); // different startup seed
    let (status2, json2) = post_completion(fx::tokenizerless_router(engine2), body).await;

    assert_eq!(status1, axum::http::StatusCode::OK, "{json1}");
    assert_eq!(status2, axum::http::StatusCode::OK, "{json2}");
    assert!(!logprob_tokens(&json1).is_empty(), "{json1}");
    assert_eq!(
        logprob_tokens(&json1),
        logprob_tokens(&json2),
        "the same explicit seed must reproduce the same logprobs-path output \
         even when the two engines' own startup PRNG seeds differ"
    );
    assert_eq!(
        json1["choices"][0]["logprobs"]["token_logprobs"],
        json2["choices"][0]["logprobs"]["token_logprobs"]
    );
}

/// `logprobs` combined with only a frequency/presence penalty must still
/// succeed end-to-end — the new guard must not over-reject a combination
/// that is actually honoured correctly (penalties are applied on the
/// logprobs path via `set_penalties`).
#[tokio::test]
async fn handler_logprobs_and_frequency_penalty_together_still_succeeds() {
    let app = test_router();
    let (status, json) = post_completion(
        app,
        serde_json::json!({
            "prompt": "hello", "max_tokens": 3, "logprobs": 2, "frequency_penalty": 0.5
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(!json["choices"][0]["logprobs"].is_null());
}

/// A router whose engine emits exactly `[0][1][2]…` (one byte token per
/// character), decoded by the byte-level fixture tokenizer: deterministic
/// text starting with `[`, whatever the sampling parameters.
fn scripted_text_router() -> axum::Router {
    crate::server::create_router(
        fx::scripted_byte_engine("[0][1][2][3][4][5][6][7][8][9]"),
        Some(fx::byte_tokenizer()),
    )
}

/// A stop sequence actually truncates the *decoded* completion text
/// end-to-end through the real handler, not just through `StopChecker` in
/// isolation: the scripted text starts with `[`, so `stop: "["` matches at
/// position 0 whatever the generation length.
#[tokio::test]
async fn handler_stop_sequence_truncates_the_completion_end_to_end() {
    let app = scripted_text_router();
    let (status, json) = post_completion(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 4, "stop": "[" }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    let text = json["choices"][0]["text"].as_str().expect("text field");
    assert_eq!(
        text, "",
        "text must be truncated at the stop sequence found at position 0, got {text:?}"
    );
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

/// Regression guard for the empty-stop-sequence bug, end-to-end: an
/// empty `stop` entry must not truncate every completion to `""`.
#[tokio::test]
async fn handler_empty_stop_sequence_does_not_truncate_everything() {
    let app = scripted_text_router();
    let (status, json) = post_completion(
        app,
        serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stop": "" }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    let text = json["choices"][0]["text"].as_str().expect("text field");
    assert!(
        text.starts_with('['),
        "an empty stop sequence must not truncate the completion; got {text:?}"
    );
}

/// `sec-03` concurrency guard, pinned on this route specifically: a long
/// `/v1/completions` generation must not block a concurrent `/health`
/// request on the same runtime. The underlying seam
/// (`run_blocking_generation`) already has its own unit tests in
/// `server/blocking.rs`, but nothing previously exercised the property
/// through the real `/v1/completions` handler; this mirrors
/// `server_hardening_round4.rs`'s
/// `a_second_request_is_served_while_a_long_generation_runs`, which
/// pins the identical property for the base `/v1/chat/completions`
/// route.
#[tokio::test]
async fn handler_long_completion_does_not_block_a_concurrent_health_check() {
    use std::time::{Duration, Instant};

    let app = test_router();

    // Calibrate the tiny engine's speed so the "long" completion is long
    // enough to be unambiguous on any machine without being needlessly
    // slow on a fast one.
    let probe_start = Instant::now();
    let (probe_status, _) = post_completion(
        app.clone(),
        serde_json::json!({ "prompt": "hello", "max_tokens": 4 }),
    )
    .await;
    assert_eq!(probe_status, axum::http::StatusCode::OK);
    let per_token = probe_start.elapsed() / 4;
    let target = Duration::from_millis(1_500);
    let long_tokens = (target.as_nanos() / per_token.as_nanos().max(1)).clamp(16, 400) as usize;

    let app_for_completion = app.clone();
    let completion = tokio::spawn(async move {
        let started = Instant::now();
        let (status, _) = post_completion(
            app_for_completion,
            serde_json::json!({ "prompt": "hello", "max_tokens": long_tokens }),
        )
        .await;
        (status, started.elapsed())
    });

    // Let the generation task actually start before timing the
    // concurrent request.
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(20)).await;

    let health_start = Instant::now();
    let health = {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;
        app.oneshot(
            Request::get("/health")
                .body(Body::empty())
                .expect("health request"),
        )
        .await
        .expect("health response")
    };
    let health_elapsed = health_start.elapsed();
    assert_eq!(health.status(), axum::http::StatusCode::OK);

    let (completion_status, completion_elapsed) = completion.await.expect("completion task");
    assert_eq!(completion_status, axum::http::StatusCode::OK);
    assert!(
        health_elapsed < completion_elapsed,
        "the concurrent health check ({health_elapsed:?}) must finish before the long \
         completion ({completion_elapsed:?}); generation is blocking the runtime"
    );
}

// ── repetition_penalty validated identically to chat, and the
//    temperature:0 router test ─────────────────────────────────────────

/// `repetition_penalty` below `1.0` must be rejected, matching
/// `ChatCompletionRequest`'s `>= 1.0` rule (the earlier `> 0.0` text this
/// endpoint used let a value like `0.5` — which REWARDS rather than merely
/// fails to penalise repeated tokens — through with `200`, while the base
/// chat endpoint already rejected it with `400`).
#[test]
fn validate_rejects_repetition_penalty_below_one() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.repetition_penalty = Some(0.5);
    let err = expect_err(validate_completion_request(req));
    assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[test]
fn validate_accepts_repetition_penalty_of_exactly_one() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.repetition_penalty = Some(1.0);
    assert!(validate_completion_request(req).is_ok());
}

#[test]
fn validate_accepts_repetition_penalty_above_one() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.repetition_penalty = Some(1.3);
    assert!(validate_completion_request(req).is_ok());
}

/// The synthetic-engine router test for greedy parameter resolution: a
/// `temperature: 0` `/v1/completions` request, against an engine started
/// with a DISTINCTIVE ambient `repetition_penalty` (so a bug that silently
/// keeps the ambient value — or any other hidden penalty — instead of the
/// greedy `1.0` shape is caught), must resolve to EXACTLY the params
/// `InferenceEngine::greedy_gpu_eligible` requires: `temperature < eps`,
/// `repetition_penalty == 1.0`, no active frequency/presence penalties.
///
/// `greedy_gpu_eligible` itself is unreachable on a synthetic `tiny_test()`
/// engine (`uses_fused_gpu_decode()` is always `false` off a real GGUF —
/// the `server::tests::gpu_argmax_routing` module covers the real-model,
/// real-GPU case, self-skipping unless `OXI_MODEL`/`OXI_TOKENIZER` name
/// both files), so
/// this proves the *params* resolution is correct by output-token equality
/// against a second, identical engine driven directly with that exact
/// greedy shape: a tiny_test() model's generation is a deterministic
/// function of (config, seed, params) alone, so any divergence can only
/// come from the router resolving the wrong params.
#[tokio::test]
async fn temperature_zero_resolves_to_the_greedy_gpu_eligible_param_shape() {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    // Distinctive: NOT 1.0, so a bug that keeps the ambient value (or
    // `SamplingParams::default()`'s) instead of the client's implicit
    // `repetition_penalty: 1.0` (greedy has no reason to specify one, but
    // the resolved value must still come out at 1.0) is caught.
    let ambient = SamplingParams {
        repetition_penalty: 1.35,
        ..SamplingParams::default()
    };
    // No tokenizer is attached to the router, so every prompt string runs
    // as its configured prompt start token — the reference call below uses
    // that exact prompt for a fair comparison.
    let fallback_prompt = vec![fx::QWEN3_IM_START];

    // Reference: ask the engine directly for the exact greedy_gpu_eligible
    // shape (`InferenceEngine::greedy_gpu_eligible`'s own three conditions).
    let mut reference_engine =
        crate::engine::InferenceEngine::new(config.clone(), ambient.clone(), 42);
    let greedy_params = SamplingParams {
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..ambient.clone()
    };
    let reference_tokens = reference_engine
        .generate_with_params_and_penalties(
            &fallback_prompt,
            8,
            &greedy_params,
            &PenaltyParams::default(),
        )
        .expect("reference generation must succeed");
    let reference_ids = fx::id_token_strings(&reference_tokens);

    // Router: the real HTTP route, same ambient config, `temperature: 0`
    // and no other sampling overrides -- the request shape that once took a
    // hidden penalty on the real model. A tokenizer-less completion's text
    // is empty, so the generated ids are read from `logprobs.tokens`.
    let router_engine = crate::engine::InferenceEngine::new(config, ambient, 42);
    let app = fx::tokenizerless_router(router_engine);
    let (status, json) = post_completion(
        app,
        serde_json::json!({
            "prompt": "hello", "max_tokens": 8, "temperature": 0.0, "logprobs": 1
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    let router_ids = logprob_tokens(&json);

    assert_eq!(
        router_ids, reference_ids,
        "temperature:0 through /v1/completions must resolve to EXACTLY the \
         greedy_gpu_eligible params shape (repetition_penalty 1.0), not the \
         ambient 1.35 or any other hidden penalty"
    );
}

// ── The declared OpenAI field set: honour or refuse by name, deny the rest ──

#[test]
fn validate_rejects_best_of_other_than_one() {
    for best_of in [0usize, 2, 5] {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.best_of = Some(best_of);
        let err = expect_err(validate_completion_request(req));
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(
            err.to_json()["error"]["param"],
            "best_of",
            "best_of={best_of}"
        );
    }
}

#[test]
fn validate_rejects_a_non_empty_logit_bias() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.logit_bias = Some(std::collections::HashMap::from([("42".to_string(), 5.0)]));
    let err = expect_err(validate_completion_request(req));
    assert_eq!(err.to_json()["error"]["param"], "logit_bias");
}

#[test]
fn validate_accepts_the_declared_no_op_values() {
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.best_of = Some(1);
    req.logit_bias = Some(std::collections::HashMap::new());
    req.echo = Some(true);
    req.user = Some("u".to_string());
    req.model = Some("any".to_string());
    req.suffix = Some(String::new());
    req.top_k = Some(40);
    req.min_p = Some(0.05);
    let validated = validate_completion_request(req).expect("must validate");
    assert_eq!(validated.req_top_k, Some(40));
    assert_eq!(validated.min_p, Some(0.05));
    assert!(
        validated.custom_sampling,
        "a top_k override customizes sampling"
    );
}

#[test]
fn validate_rejects_out_of_range_min_p_and_top_k() {
    for min_p in [-0.1f32, 1.5, f32::NAN] {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.min_p = Some(min_p);
        let err = expect_err(validate_completion_request(req));
        assert_eq!(err.to_json()["error"]["param"], "min_p", "min_p={min_p}");
    }
    let mut req = base_request(PromptInput::Single("hi".to_string()));
    req.top_k = Some(2_000_000);
    let err = expect_err(validate_completion_request(req));
    assert_eq!(err.to_json()["error"]["param"], "top_k");
}

/// A field outside the declared set is refused with a `400` naming it —
/// a misspelt parameter is an error, never silently ignored.
#[tokio::test]
async fn handler_refuses_an_unknown_field_naming_it() {
    let (status, json) = post_completion(
        test_router(),
        serde_json::json!({ "prompt": "hello", "max_tokns": 3 }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(json["error"]["param"], "max_tokns", "{json}");
    assert_eq!(json["error"]["code"], "unknown_parameter", "{json}");
    assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
}

/// Every documented OpenAI completions field (and each sampling extension)
/// is declared, so a request carrying all of them with supported values is
/// accepted.
#[tokio::test]
async fn handler_accepts_every_declared_field() {
    let (status, json) = post_completion(
        test_router(),
        serde_json::json!({
            "model": "m", "prompt": "hello", "suffix": "", "max_tokens": 2,
            "temperature": 0.5, "top_p": 0.9, "n": 1, "stream": false,
            "stream_options": {"include_usage": false}, "logprobs": 1, "echo": false,
            "stop": ["zzz"], "presence_penalty": 0.1, "frequency_penalty": 0.1,
            "best_of": 1, "logit_bias": {}, "user": "u", "seed": 3,
            "top_k": 5, "min_p": 0.01, "repetition_penalty": 1.1
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
}

/// `top_k` is honoured, not just accepted: on an engine whose every step is
/// an equal choice among 26 letters, `top_k: 1` keeps one candidate, so the
/// whole completion repeats one letter (without it the letters vary).
#[tokio::test]
async fn handler_honours_top_k() {
    let router =
        || crate::server::create_router(fx::uniform_letters_engine(8), Some(fx::byte_tokenizer()));
    let (status, json) = post_completion(
        router(),
        serde_json::json!({ "prompt": "go", "max_tokens": 12, "temperature": 1.0, "top_k": 1 }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    let text = json["choices"][0]["text"].as_str().unwrap_or_default();
    assert_eq!(text.len(), 12, "{json}");
    assert!(text.bytes().all(|b| b == text.as_bytes()[0]), "{text}");

    let (_, json) = post_completion(
        router(),
        serde_json::json!({ "prompt": "go", "max_tokens": 12, "temperature": 1.0 }),
    )
    .await;
    let text = json["choices"][0]["text"].as_str().unwrap_or_default();
    assert!(!text.bytes().all(|b| b == text.as_bytes()[0]), "{text}");
}

// ── A tokenizer-less completion: empty text, streamed or not ───────────

/// Without a tokenizer there is no text to render: the non-streamed
/// completion's text is empty — the same as the stream, which sends no text
/// — while `usage` counts every generated token.
#[tokio::test]
async fn handler_without_a_tokenizer_answers_empty_text_like_the_stream() {
    let (status, json) = post_completion(
        test_router(),
        serde_json::json!({ "prompt": "hello", "max_tokens": 3 }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    assert_eq!(json["choices"][0]["text"], "", "{json}");
    assert!(
        json["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "{json}"
    );
}
