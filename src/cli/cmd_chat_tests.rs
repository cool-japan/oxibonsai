//! Unit tests for `cmd_chat.rs` (sibling file, declared there via `#[path]`,
//! so `super` still names that module).

use super::*;
use crate::cli::test_fixtures::{self, byte_level_tokenizer_json, THINKING_TEMPLATE};
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

fn tokenizer_with_thinking_template() -> oxibonsai_runtime::TokenizerBridge {
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
    let template = ResolvedChatTemplate::from_gguf(&gguf.metadata).expect("compiles");
    oxibonsai_runtime::TokenizerBridge::native_from_json_str(&byte_level_tokenizer_json())
        .expect("fixture tokenizer")
        .with_chat_template(template)
}

fn conversation() -> Vec<RenderMessage> {
    vec![
        RenderMessage::new("system", "Be brief."),
        RenderMessage::new("user", "Hi"),
        RenderMessage::new("assistant", "Hello!").with_reasoning_content("user greets"),
        RenderMessage::new("user", "Bye"),
    ]
}

#[test]
fn every_turn_renders_the_whole_history_through_the_gguf_template() {
    let tok = tokenizer_with_thinking_template();
    let template = tok.resolved_chat_template();
    let mut history = conversation();
    let turn = render_turn(
        &tok,
        &template,
        &mut history,
        &ChatContract::default(),
        16,
        4096,
    )
    .expect("fits");
    let expected = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
                    <|im_start|>assistant\n<think>\nuser greets\n</think>\n\nHello!<|im_end|>\n\
                    <|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n";
    assert_eq!(
        generate::render_prompt(&template, &history, &ChatContract::default()).expect("render"),
        expected,
        "the previous assistant turn is re-rendered with its reasoning_content"
    );
    assert_eq!(turn.tokens, tok.encode(expected).expect("encode"));
    assert!(
        turn.started_in_think,
        "the template's default leaves <think> open"
    );
    assert_eq!(turn.dropped_messages, 0);
    assert_eq!(history.len(), 4);
}

#[test]
fn no_think_closes_the_generation_prompts_think_block() {
    let tok = tokenizer_with_thinking_template();
    let template = tok.resolved_chat_template();
    let mut history = vec![RenderMessage::new("user", "Hi")];
    let contract = ChatContract {
        enable_thinking: Some(false),
        ..ChatContract::default()
    };
    let turn = render_turn(&tok, &template, &mut history, &contract, 16, 4096).expect("fits");
    assert!(!turn.started_in_think);
    let rendered = generate::render_prompt(&template, &history, &contract).expect("render");
    assert!(
        rendered.ends_with("<think>\n\n</think>\n\n"),
        "{rendered:?}"
    );
}

#[test]
fn the_oldest_non_system_messages_are_dropped_to_fit_the_context() {
    let tok = tokenizer_with_thinking_template();
    let template = tok.resolved_chat_template();
    let contract = ChatContract::default();
    let full_len = {
        let rendered =
            generate::render_prompt(&template, &conversation(), &contract).expect("render");
        tok.encode(&rendered).expect("encode").len()
    };
    let max_tokens = 8;
    let mut history = conversation();
    let turn = render_turn(
        &tok,
        &template,
        &mut history,
        &contract,
        max_tokens,
        full_len + max_tokens - 1,
    )
    .expect("fits after dropping");
    assert!(turn.dropped_messages >= 1);
    assert!(
        turn.tokens.len() < full_len,
        "the dropped turn(s) shortened the prompt"
    );
    assert_eq!(
        history.first().map(|m| m.role.as_str()),
        Some("system"),
        "system is kept"
    );
    assert_eq!(
        history.last().map(|m| m.content.as_str()),
        Some("Bye"),
        "the latest user turn is never dropped"
    );
    assert_eq!(history.len(), 4 - turn.dropped_messages);
}

#[test]
fn a_latest_message_that_cannot_fit_alone_is_refused() {
    let tok = tokenizer_with_thinking_template();
    let template = tok.resolved_chat_template();
    let mut history = vec![RenderMessage::new(
        "user",
        "a fairly long message that will not fit",
    )];
    let err = match render_turn(
        &tok,
        &template,
        &mut history,
        &ChatContract::default(),
        8,
        16,
    ) {
        Ok(_) => panic!("cannot fit in 16 tokens"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("too long for the context window"), "{err}");
    assert!(err.contains("--ctx is 16"), "{err}");
}

// ── golden2/apply_template.json: all 5 cases byte-identical (B2-13 G7) ──────

/// The raw text of the JSON value at `key` inside `object_text` (the first
/// occurrence), found by bracket matching — the tool definitions must reach
/// the template as the file's own bytes, never through a
/// `serde_json::Value` round trip that would reorder their keys.
fn raw_json_value<'t>(object_text: &'t str, key: &str) -> Option<&'t str> {
    let key_pos = object_text.find(&format!("\"{key}\""))?;
    let after_key = &object_text[key_pos + key.len() + 2..];
    let colon = after_key.find(':')?;
    let value = after_key[colon + 1..].trim_start();
    let open = value.chars().next()?;
    let close = match open {
        '[' => ']',
        '{' => '}',
        _ => return None,
    };
    let (mut depth, mut in_string, mut escaped) = (0usize, false, false);
    for (i, c) in value.char_indices() {
        if in_string {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            c if c == open => depth += 1,
            c if c == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(&value[..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// The text of each top-level element of the JSON array `text`.
fn split_top_level_array(text: &str) -> Vec<&str> {
    let body = text.trim();
    let inner = &body[1..body.len() - 1];
    let (mut out, mut depth, mut start, mut in_string, mut escaped) =
        (Vec::new(), 0usize, None, false, false);
    for (i, c) in inner.char_indices() {
        if in_string {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' | '[' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' | ']' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start.take() {
                        out.push(&inner[s..=i]);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// B2-13's G7 contract, through `oxibonsai chat`'s own path: the REAL
/// 27B's embedded tokenizer + `tokenizer.chat_template` (resolved exactly as
/// `chat` resolves it), the contract built from flags (`--no-think`,
/// `--reasoning-effort`, `--tools <file>` written from the golden's raw
/// text), rendered by [`generate::render_prompt`] and encoded by
/// [`render_turn`] — byte-identical to the reference renderer on all 5
/// recorded cases.
#[test]
fn chat_renders_all_five_apply_template_cases_byte_identically() {
    let Some(model) = test_fixtures::env_path(
        "OXI_BONSAI2_PQ2_GGUF",
        "the real Ternary-Bonsai-2-27B-PQ2_0.gguf (tokenizer + chat template)",
    ) else {
        return;
    };
    let Some(golden_dir) = test_fixtures::env_path(
        "OXI_BONSAI2_GOLDEN_DIR",
        "the golden2 directory holding apply_template.json",
    ) else {
        return;
    };
    let _real = test_fixtures::real_model_lock();
    let golden_text =
        std::fs::read_to_string(golden_dir.join("apply_template.json")).expect("read golden");
    let cases: Vec<serde_json::Value> = serde_json::from_str(&golden_text).expect("golden JSON");
    let raw_cases = split_top_level_array(&golden_text);
    assert_eq!(cases.len(), 5, "the recorded golden has 5 cases");
    assert_eq!(raw_cases.len(), 5);

    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(&model).expect("mmap");
    let gguf = GgufFile::parse(&mmap).expect("parse");
    let model_str = model.to_string_lossy().into_owned();
    let expected_vocab = crate::cli::util::model_vocab_size(&gguf).ok();
    let lookup = resolve_tokenizer_vocab_aware(None, &model_str, expected_vocab);
    let tok = cmd_run::resolve_model_tokenizer(
        None,
        &lookup,
        &gguf,
        expected_vocab,
        TokenizerBackendChoice::Auto,
        false,
    )
    .expect("resolve")
    .expect("the 27B carries its own tokenizer");
    let template = tok.resolved_chat_template();
    assert!(
        matches!(template, ResolvedChatTemplate::Jinja(_)),
        "the GGUF's own template, never the fallback"
    );

    let scratch = test_fixtures::scratch_dir("apply_template");
    for (index, (case, raw)) in cases.iter().zip(&raw_cases).enumerate() {
        let input = &case["input"];
        let kwargs = &input["chat_template_kwargs"];
        let tools_path = raw_json_value(raw, "tools").map(|tools_text| {
            let path = scratch.join(format!("tools_{index}.json"));
            std::fs::write(&path, tools_text).expect("write tools");
            path.to_string_lossy().into_owned()
        });
        let contract = ChatContract::from_flags(
            kwargs["enable_thinking"].as_bool(),
            kwargs["reasoning_effort"].as_str().map(str::to_string),
            tools_path.as_deref(),
        )
        .expect("contract");
        let mut history: Vec<RenderMessage> = input["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(|m| {
                let message = RenderMessage::new(
                    m["role"].as_str().unwrap_or_default(),
                    m["content"].as_str().unwrap_or_default(),
                );
                match m["reasoning_content"].as_str() {
                    Some(reasoning) => message.with_reasoning_content(reasoning),
                    None => message,
                }
            })
            .collect();
        let expected = case["output"]["prompt"].as_str().expect("golden prompt");
        let rendered = generate::render_prompt(&template, &history, &contract).expect("render");
        assert_eq!(rendered, expected, "apply_template case {index} differs");
        let turn = render_turn(&tok, &template, &mut history, &contract, 16, 262_144)
            .expect("render_turn");
        assert_eq!(
            turn.tokens,
            tok.encode(expected).expect("encode"),
            "case {index}: chat encodes exactly the golden prompt"
        );
        assert_eq!(
            turn.started_in_think,
            !expected.ends_with("</think>\n\n"),
            "case {index}"
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
}
