//! Unit tests for `generate.rs` (sibling file, declared there via `#[path]`,
//! so `super` still names that module).

use super::*;
use crate::cli::test_fixtures::{
    byte_level_tokenizer_json, scratch_dir, THINKING_TEMPLATE, THINK_CLOSE_ID,
};
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

fn tokenizer() -> TokenizerBridge {
    TokenizerBridge::native_from_json_str(&byte_level_tokenizer_json())
        .expect("the byte-level fixture tokenizer loads")
}

fn thinking_template() -> ResolvedChatTemplate {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen35".to_string()),
    );
    w.add_metadata(
        "tokenizer.chat_template",
        MetadataWriteValue::Str(THINKING_TEMPLATE.to_string()),
    );
    let bytes = w.to_bytes().expect("serialize");
    let gguf = GgufFile::parse(&bytes).expect("parse");
    ResolvedChatTemplate::from_gguf(&gguf.metadata).expect("the test template compiles")
}

// ── ChatContract / --tools ──────────────────────────────────────────────────

#[test]
fn an_empty_contract_renders_the_templates_undefined_branch() {
    let contract = ChatContract::from_flags(None, None, None).expect("no tools");
    assert!(contract.is_empty());
    let options = contract.render_options(true);
    assert!(options.add_generation_prompt);
    assert_eq!(
        options.enable_thinking, None,
        "never defaulted: the template decides"
    );
    assert_eq!(options.reasoning_effort, None);
    assert_eq!(options.tools, None);
}

#[test]
fn tools_file_text_is_kept_byte_for_byte_with_its_key_order() {
    let dir = scratch_dir("tools_order");
    let path = dir.join("tools.json");
    // Keys deliberately NOT in sorted order, plus a trailing newline.
    let text = r#"[{"type": "function", "function": {"name": "zeta", "parameters": {"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "number"}}}, "description": "last"}}]"#;
    std::fs::write(&path, format!("{text}\n")).expect("write tools");
    let contract = ChatContract::from_flags(
        Some(false),
        Some("low".to_string()),
        Some(path.to_string_lossy().as_ref()),
    )
    .expect("a JSON array loads");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!contract.is_empty());
    assert_eq!(
        contract.tools_json.as_deref(),
        Some(text),
        "only the trailing newline is trimmed"
    );
    let options = contract.render_options(true);
    assert_eq!(options.tools.as_deref(), Some(text));
    assert_eq!(options.enable_thinking, Some(false));
    assert_eq!(options.reasoning_effort.as_deref(), Some("low"));
}

#[test]
fn a_tools_file_that_is_not_a_json_array_is_refused() {
    let dir = scratch_dir("tools_bad");
    let object = dir.join("object.json");
    std::fs::write(&object, r#"{"type": "function"}"#).expect("write");
    let err = load_tools_file(object.to_string_lossy().as_ref()).expect_err("an object");
    assert!(err.to_string().contains("JSON array"), "{err}");

    let invalid = dir.join("invalid.json");
    std::fs::write(&invalid, "[{").expect("write");
    let err = load_tools_file(invalid.to_string_lossy().as_ref()).expect_err("invalid JSON");
    assert!(err.to_string().contains("not valid JSON"), "{err}");

    let missing = dir.join("missing.json");
    let err = load_tools_file(missing.to_string_lossy().as_ref()).expect_err("missing");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        err.to_string().contains("failed to read --tools file"),
        "{err}"
    );
}

#[test]
fn render_prompt_follows_the_contract_through_the_template() {
    let template = thinking_template();
    let messages = [RenderMessage::new("user", "hi")];
    let undefined = render_prompt(&template, &messages, &ChatContract::default()).expect("render");
    assert!(
        undefined.ends_with("<|im_start|>assistant\n<think>\n"),
        "{undefined:?}"
    );
    let off = render_prompt(
        &template,
        &messages,
        &ChatContract {
            enable_thinking: Some(false),
            ..ChatContract::default()
        },
    )
    .expect("render");
    assert!(off.ends_with("<think>\n\n</think>\n\n"), "{off:?}");
    assert!(
        off.starts_with("<|im_start|>user\nhi<|im_end|>\n"),
        "{off:?}"
    );
}

#[test]
fn reasoning_display_resolves_the_two_flags() {
    assert_eq!(
        ReasoningDisplay::from_flags(false, false),
        (ReasoningDisplay::Show, false)
    );
    assert_eq!(
        ReasoningDisplay::from_flags(true, false),
        (ReasoningDisplay::Show, true)
    );
    assert_eq!(
        ReasoningDisplay::from_flags(false, true),
        (ReasoningDisplay::Hide, true)
    );
}

// ── TokenPrinter: the reasoning split on the `</think>` id ──────────────────

fn feed(printer: &mut TokenPrinter<'_>, ids: &[u32]) {
    for &id in ids {
        printer.push(id).expect("decode");
    }
}

#[test]
fn a_prompt_left_inside_think_splits_reasoning_from_content_on_the_close_id() {
    let tok = tokenizer();
    assert_eq!(tok.think_close_id(), Some(THINK_CLOSE_ID));
    for display in [ReasoningDisplay::Show, ReasoningDisplay::Hide] {
        let mut printer = TokenPrinter::new(Some(&tok), true, display, false);
        feed(&mut printer, &tok.encode("2+2 is 4.").expect("encode"));
        feed(&mut printer, &[THINK_CLOSE_ID]);
        feed(&mut printer, &tok.encode("\n\nFour").expect("encode"));
        assert_eq!(printer.full_text(), "2+2 is 4.Four");
        let (reasoning, content) = printer.finish(None);
        assert_eq!(reasoning.as_deref(), Some("2+2 is 4."), "{display:?}");
        assert_eq!(
            content, "Four",
            "the post-boundary \\n\\n reaches neither channel"
        );
    }
}

#[test]
fn a_prompt_outside_think_is_all_content() {
    let tok = tokenizer();
    let mut printer = TokenPrinter::new(Some(&tok), false, ReasoningDisplay::Show, false);
    feed(&mut printer, &tok.encode("plain").expect("encode"));
    let (reasoning, content) = printer.finish(None);
    assert_eq!(reasoning, None);
    assert_eq!(content, "plain");
}

#[test]
fn whitespace_only_reasoning_is_reported_as_none_like_the_server() {
    let tok = tokenizer();
    let mut printer = TokenPrinter::new(Some(&tok), true, ReasoningDisplay::Show, false);
    feed(&mut printer, &tok.encode("\n").expect("encode"));
    feed(&mut printer, &[THINK_CLOSE_ID]);
    feed(&mut printer, &tok.encode("\n\n4").expect("encode"));
    let (reasoning, content) = printer.finish(None);
    assert_eq!(reasoning, None);
    assert_eq!(content, "4");
}

#[test]
fn a_buffered_printer_truncates_the_content_at_the_stop_sequence() {
    let tok = tokenizer();
    let mut printer = TokenPrinter::new(Some(&tok), false, ReasoningDisplay::Show, false);
    feed(
        &mut printer,
        &tok.encode("answer STOP trailing").expect("encode"),
    );
    let truncate = |text: &str| text.split("STOP").next().unwrap_or(text).to_string();
    let (_, content) = printer.finish(Some(&truncate));
    assert_eq!(content, "answer ");
}

#[test]
fn without_a_tokenizer_the_printer_counts_raw_ids() {
    let mut printer = TokenPrinter::new(None, true, ReasoningDisplay::Show, true);
    feed(&mut printer, &[5, 6, 7]);
    assert!(matches!(printer, TokenPrinter::Raw { count: 3 }));
    assert_eq!(printer.full_text(), "");
    assert_eq!(printer.finish(None), (None, String::new()));
}

// ── The CLI's own sampler (grammar / `--stop` path only) ────────────────────
//
// `only_a_sampled_request_with_min_p_needs_the_cli_sampler` tested
// `needs_cli_sampler`, retired along with the CLI-owned min-p decode loop it
// gated: the engine's own decode path applies `min_p` directly now,
// proven token-for-token identical by
// `cmd_run_tests::min_p_engine_path_matches_the_cli_loop`). `cli_sampler`
// itself stays (the grammar/`--stop` path still builds its plain sampler
// from it), so its own tests below are unchanged.

fn unfiltered_params(temperature: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..SamplingParams::default()
    }
}

#[test]
fn cli_sampler_applies_min_p() {
    // p ∝ e^2, e^1.9, e^0, e^-1: tokens 2 and 3 fall below 0.5 × p_max.
    let logits = [2.0f32, 1.9, 0.0, -1.0];
    let mut filtered = cli_sampler(unfiltered_params(1.0), 7, PenaltyParams::default(), 0.5);
    assert_eq!(filtered.min_p(), 0.5);
    for _ in 0..500 {
        let id = filtered.sample(&logits).expect("sample");
        assert!(id <= 1, "min-p 0.5 must never pick token {id}");
    }
    // Without min-p the tail is reachable (P(no tail draw in 500) ≈ 1e-20).
    let mut unfiltered = cli_sampler(unfiltered_params(1.0), 7, PenaltyParams::default(), 0.0);
    let tail = (0..500)
        .filter_map(|_| unfiltered.sample(&logits).ok())
        .filter(|&id| id >= 2)
        .count();
    assert!(
        tail > 0,
        "the unfiltered sampler must reach the tail tokens"
    );
}

#[test]
fn cli_sampler_is_deterministic_per_seed() {
    let logits = [1.0f32, 0.9, 0.8, 0.7, 0.6, 0.5];
    let draw = |seed: u64| -> Vec<u32> {
        let mut sampler = cli_sampler(unfiltered_params(0.8), seed, PenaltyParams::default(), 0.05);
        (0..64)
            .filter_map(|_| sampler.sample(&logits).ok())
            .collect()
    };
    assert_eq!(
        draw(42),
        draw(42),
        "the same seed reproduces the same draws"
    );
    assert_ne!(draw(42), draw(43), "a different seed draws differently");
}
