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
        &text_rows,
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
    let turn = render_turn(
        &tok,
        &template,
        &mut history,
        &contract,
        16,
        4096,
        &text_rows,
    )
    .expect("fits");
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
        &text_rows,
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
        history.last().and_then(|m| m.content.as_text()),
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
        &text_rows,
    ) {
        Ok(_) => panic!("cannot fit in 16 tokens"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("exceeds max context 16"), "{err}");
    assert!(err.contains("this message alone does not fit"), "{err}");
}

/// When the prompt itself fits but
/// nothing is left to drop and the requested `--max-tokens` would still
/// carry it past `--ctx`, the turn is NOT refused — the budget is clamped
/// to exactly what remains and generation proceeds (a graceful "context
/// window full" stop, not a crash).
#[test]
fn a_fitting_prompt_with_no_room_to_drop_clamps_the_budget_instead_of_erroring() {
    let tok = tokenizer_with_thinking_template();
    let template = tok.resolved_chat_template();
    let contract = ChatContract::default();
    let mut history = vec![RenderMessage::new("user", "Hi")];
    let prompt_len = {
        let rendered = generate::render_prompt(&template, &history, &contract).expect("render");
        tok.encode(&rendered).expect("encode").len()
    };
    let max_context = prompt_len + 3;
    let turn = render_turn(
        &tok,
        &template,
        &mut history,
        &contract,
        50,
        max_context,
        &text_rows,
    )
    .expect("the prompt itself fits; this must not be an error");
    assert_eq!(turn.dropped_messages, 0, "nothing WAS droppable");
    assert_eq!(
        turn.effective_max_tokens, 3,
        "clamped to exactly what remains of the context"
    );
    assert_eq!(history.len(), 1, "the sole message is kept, never dropped");
}

// ── vendored apply_template.json: all 5 cases byte-identical (G7) ──────

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

/// The G7 contract, through `oxibonsai chat`'s own path: the REAL
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
        "the golden directory holding apply_template.json",
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
        let turn = render_turn(
            &tok,
            &template,
            &mut history,
            &contract,
            16,
            262_144,
            &text_rows,
        )
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

// ── --image sessions ────────────────────────────────────────────────────────

/// One already-encoded 8 x 6 image (48 rows of width 4) under the Bonsai 2
/// marker ids.
fn one_image_session() -> SessionImages {
    let grid = GridSize { h: 6, w: 8 };
    SessionImages::new(
        vec![EncodedImage {
            rows: vec![0.25; 48 * 4],
            grid,
            source: (256, 192),
        }],
        VisionTokenIds::BONSAI2,
    )
}

#[test]
fn a_session_prompt_is_multimodal_while_it_holds_the_image_message() {
    let session = one_image_session();
    let ids = VisionTokenIds::BONSAI2;
    assert_eq!(session.len(), 1);
    let with_image = vec![1u32, ids.vision_start, ids.image_pad, ids.vision_end, 2, 3];
    assert!(session.placed_in(&with_image));
    assert_eq!(session.rows(&with_image).expect("rows"), 6 - 1 + 48);
    let prompt = session.prompt(with_image.clone()).expect("spliced");
    assert_eq!(prompt.image_count(), 1);
    assert_eq!(prompt.len(), 53);
    assert_eq!(prompt.tokens(), with_image.as_slice());

    // Once the image message is dropped the conversation is plain text.
    let text_only = vec![1u32, 2, 3];
    assert!(!session.placed_in(&text_only));
    assert_eq!(session.rows(&text_only).expect("rows"), 3);
    assert_eq!(
        session.prompt(text_only.clone()).expect("text"),
        ChatPrompt::Text(text_only)
    );

    // A second placeholder (a pasted `<|image_pad|>`) cannot splice.
    let two_pads = vec![
        ids.vision_start,
        ids.image_pad,
        ids.vision_end,
        ids.vision_start,
        ids.image_pad,
        ids.vision_end,
    ];
    let msg = session
        .prompt(two_pads.clone())
        .expect_err("two placeholders, one image")
        .to_string();
    assert!(
        msg.starts_with("[image_placeholder_count_mismatch]"),
        "{msg}"
    );
    assert!(session.rows(&two_pads).is_err());
}

/// The context check counts what the model will see: with the image rows
/// the same conversation no longer fits, so the oldest turns are dropped;
/// an error from the row count refuses the turn.
#[test]
fn render_turn_budgets_with_the_expanded_row_count() {
    let tok = tokenizer_with_thinking_template();
    let template = tok.resolved_chat_template();
    let contract = ChatContract::default();
    let full_len = {
        let rendered =
            generate::render_prompt(&template, &conversation(), &contract).expect("render");
        tok.encode(&rendered).expect("encode").len()
    };
    let max_tokens = 8;
    let max_context = full_len + max_tokens;

    let mut history = conversation();
    let turn = render_turn(
        &tok,
        &template,
        &mut history,
        &contract,
        max_tokens,
        max_context,
        &text_rows,
    )
    .expect("fits as text");
    assert_eq!(turn.dropped_messages, 0);

    let mut history = conversation();
    let with_image_rows = |tokens: &[u32]| Ok(tokens.len() + 47);
    let turn = render_turn(
        &tok,
        &template,
        &mut history,
        &contract,
        max_tokens,
        max_context,
        &with_image_rows,
    )
    .expect("fits after dropping");
    assert!(
        turn.dropped_messages >= 1,
        "the image rows count against the context"
    );

    let mut history = conversation();
    let refused = |_: &[u32]| -> anyhow::Result<usize> { anyhow::bail!("[code] no splice") };
    let err = match render_turn(
        &tok,
        &template,
        &mut history,
        &contract,
        max_tokens,
        max_context,
        &refused,
    ) {
        Ok(_) => panic!("the row count refused the prompt"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("no splice"), "{err}");
}

/// The images of a `chat --image` session: prepared, context-checked and
/// encoded once, before the first turn.
#[test]
fn session_images_are_encoded_once_and_checked_against_the_context() {
    let dir = crate::cli::test_fixtures::scratch_dir("chat_session_images");
    let vision = crate::cli::bonsai2::VisionRequest {
        mmproj: Some(crate::cli::bonsai2::tests::synthetic_projector(&dir)),
        images: vec![crate::cli::bonsai2::tests::pattern_png_file(&dir)],
        image_max_tokens: None,
    };
    let service = vision
        .load_service(
            "qwen35",
            &crate::cli::bonsai2::tests::bonsai2_vocabulary(),
            crate::cli::bonsai2::cli_image_policy(),
        )
        .expect("load")
        .expect("requested");
    assert!(
        encode_session_images(&crate::cli::bonsai2::VisionRequest::default(), None, 64)
            .expect("no images")
            .is_none()
    );
    let too_small = match encode_session_images(&vision, Some(&service), 40) {
        Ok(_) => panic!("48 image rows cannot fit 40 positions"),
        Err(e) => e.to_string(),
    };
    let session = encode_session_images(&vision, Some(&service), 4096)
        .expect("encode")
        .expect("one image");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(too_small.contains("occupy 48 positions"), "{too_small}");
    assert!(too_small.contains("max context 40"), "{too_small}");
    assert_eq!(session.len(), 1);
    let ids = VisionTokenIds::BONSAI2;
    let tokens = vec![ids.vision_start, ids.image_pad, ids.vision_end];
    assert_eq!(session.rows(&tokens).expect("rows"), 2 + 48);
}
