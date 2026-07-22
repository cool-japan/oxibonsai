//! Regression tests for tokenizer fidelity gaps fixed in this session:
//!
//! * `tokenizer-01`: non-special `added_tokens` (`"special": false`) must
//!   still be atomically protected from pre-tokenization during `encode`,
//!   matching HuggingFace's `AddedVocabulary` semantics. Real Qwen
//!   `tokenizer.json` files rely on this for `<tool_call>`,
//!   `<|fim_prefix|>`, `<think>`, etc.
//! * `tokenizer-02`: `detect_byte_level` must recurse into `Sequence`-wrapped
//!   `pre_tokenizer`/`decoder` objects (the real HF JSON shape), not assume a
//!   bare top-level array.
//! * `tokenizer-04`: a malformed BPE `model.vocab` with two distinct token
//!   strings sharing one numeric ID must be rejected at parse time instead of
//!   producing non-deterministic `decode()` output across process runs.

use std::path::PathBuf;

use oxibonsai_tokenizer::{HfTokenizerJson, TokenizerError};

/// Path to the real bundled Qwen-family `tokenizer.json` used to build
/// Ternary-Bonsai-1.7B/8B, if present in this checkout. Tests that need it
/// skip gracefully (rather than fail) when it is absent, since the file is
/// a large real-model asset not guaranteed to be checked out everywhere.
fn real_qwen_tokenizer_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tokenizer.json")
}

// ── tokenizer-01: non-special added tokens are atomically protected ────────

/// A minimal Qwen-shaped fixture: ByteLevel pre/post, a couple of ordinary
/// BPE merges, plus one `special: true` and one `special: false` added
/// token — mirroring the real file's `<|im_start|>` (true) vs. `<tool_call>`
/// (false) split. Neither added token has any BPE merge path that could
/// reconstitute it from pretokenized pieces, so if protection regresses the
/// encode falls through to shredded/UNK output instead of the atomic ID.
fn qwen_shaped_fixture_json() -> &'static str {
    r#"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [
            {"id": 1000, "content": "<|im_start|>", "special": true},
            {"id": 1001, "content": "<tool_call>", "special": false},
            {"id": 1002, "content": "<|fim_prefix|>", "special": false}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "<unk>": 0,
                "a": 1, "b": 2, "c": 3, "d": 4,
                "ab": 5, "cd": 6, "abcd": 7,
                "Ġa": 8, "Ġb": 9,
                "<|im_start|>": 1000,
                "<tool_call>": 1001,
                "<|fim_prefix|>": 1002
            },
            "merges": ["a b", "c d", "ab cd"]
        }
    }"#
}

#[test]
fn non_special_added_token_encoded_atomically_not_shredded() {
    let tok = HfTokenizerJson::parse(qwen_shaped_fixture_json())
        .expect("parse")
        .into_tokenizer()
        .expect("into_tokenizer");

    let ids = tok.encode("<tool_call>").expect("encode");
    assert_eq!(
        ids,
        vec![1001],
        "non-special added token <tool_call> must encode to its single \
         trained ID, not be shredded by pre-tokenization: {ids:?}"
    );

    let ids = tok.encode("<|fim_prefix|>").expect("encode");
    assert_eq!(
        ids,
        vec![1002],
        "non-special added token <|fim_prefix|> must encode atomically: {ids:?}"
    );
}

#[test]
fn special_and_non_special_added_tokens_both_protected_in_mixed_text() {
    let tok = HfTokenizerJson::parse(qwen_shaped_fixture_json())
        .expect("parse")
        .into_tokenizer()
        .expect("into_tokenizer");

    let ids = tok
        .encode("<|im_start|><tool_call>ab<|fim_prefix|>")
        .expect("encode");
    assert_eq!(
        ids,
        vec![1000, 1001, 5, 1002],
        "special and non-special added tokens must both carve out atomically \
         around ordinary BPE content: {ids:?}"
    );
}

#[test]
fn non_special_added_token_not_reachable_via_special_tokens_field() {
    // The narrower `special_tokens` set (used for decode-skip / BOS-EOS-style
    // semantics) must stay accurate to the JSON `special` flag even though
    // protection is now broader.
    let parsed = HfTokenizerJson::parse(qwen_shaped_fixture_json()).expect("parse");
    assert!(parsed.special_tokens.contains_key("<|im_start|>"));
    assert!(!parsed.special_tokens.contains_key("<tool_call>"));
    assert!(!parsed.special_tokens.contains_key("<|fim_prefix|>"));

    // But the protected superset must contain all three.
    assert!(parsed.protected_tokens.contains_key("<|im_start|>"));
    assert!(parsed.protected_tokens.contains_key("<tool_call>"));
    assert!(parsed.protected_tokens.contains_key("<|fim_prefix|>"));
}

// ── tokenizer-02: Sequence-wrapped pre_tokenizer/decoder detection ─────────

#[test]
fn sequence_wrapped_pre_tokenizer_detects_byte_level() {
    // Real HF shape: pre_tokenizer is an OBJECT {"type":"Sequence",
    // "pretokenizers":[...]}, never a bare top-level array. The ByteLevel
    // entry is nested inside `pretokenizers`.
    let json = r#"{
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\s+"}},
                {"type": "ByteLevel", "add_prefix_space": false}
            ]
        },
        "model": {
            "type": "BPE",
            "vocab": {"a": 0},
            "merges": []
        }
    }"#;
    let parsed = HfTokenizerJson::parse(json).expect("parse");
    assert!(
        parsed.byte_level,
        "Sequence-wrapped pre_tokenizer containing ByteLevel must be detected"
    );
}

#[test]
fn sequence_wrapped_decoder_detects_byte_level() {
    // Same shape but on the `decoder` side, using the `decoders` field name.
    let json = r#"{
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "x"}, "content": "y"},
                {"type": "ByteLevel"}
            ]
        },
        "model": {
            "type": "BPE",
            "vocab": {"a": 0},
            "merges": []
        }
    }"#;
    let parsed = HfTokenizerJson::parse(json).expect("parse");
    assert!(
        parsed.byte_level,
        "Sequence-wrapped decoder containing ByteLevel must be detected"
    );
}

#[test]
fn sequence_wrapped_without_byte_level_stays_false() {
    let json = r#"{
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Whitespace"},
                {"type": "Punctuation"}
            ]
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "WordPiece"}
            ]
        },
        "model": {
            "type": "BPE",
            "vocab": {"a": 0},
            "merges": []
        }
    }"#;
    let parsed = HfTokenizerJson::parse(json).expect("parse");
    assert!(
        !parsed.byte_level,
        "no ByteLevel entry anywhere must not be detected as byte-level"
    );
}

#[test]
fn real_qwen_tokenizer_json_has_byte_level_and_protects_all_added_tokens() {
    let path = real_qwen_tokenizer_path();
    if !path.exists() {
        eprintln!(
            "skipping: real tokenizer.json fixture not present at {}",
            path.display()
        );
        return;
    }
    let json = std::fs::read_to_string(&path).expect("read real tokenizer.json");
    let parsed = HfTokenizerJson::parse(&json).expect("parse real tokenizer.json");
    assert!(parsed.byte_level, "Qwen tokenizer.json is byte-level");

    // Non-special added tokens from the real file (special == false in the
    // JSON) must still land in the protected superset.
    for (content, id) in [
        ("<tool_call>", 151657u32),
        ("</tool_call>", 151658),
        ("<|fim_prefix|>", 151659),
        ("<|fim_middle|>", 151660),
        ("<|fim_suffix|>", 151661),
        ("<|fim_pad|>", 151662),
        ("<|repo_name|>", 151663),
        ("<|file_sep|>", 151664),
        ("<tool_response>", 151665),
        ("</tool_response>", 151666),
    ] {
        assert_eq!(
            parsed.protected_tokens.get(content),
            Some(&id),
            "{content} must be tracked as protected"
        );
        assert!(
            !parsed.special_tokens.contains_key(content),
            "{content} is special:false in the source file and must stay out \
             of the narrower special_tokens set"
        );
    }

    let tok = parsed.into_tokenizer().expect("into_tokenizer");
    for (content, id) in [
        ("<tool_call>", 151657u32),
        ("</tool_call>", 151658),
        ("<|fim_prefix|>", 151659),
        ("<|fim_middle|>", 151660),
        ("<|fim_suffix|>", 151661),
    ] {
        let ids = tok.encode(content).expect("encode");
        assert_eq!(
            ids,
            vec![id],
            "{content} must encode atomically to its trained ID, not be \
             shredded by pre-tokenization + BPE: {ids:?}"
        );
    }

    // A realistic FIM prompt must carve out all three markers atomically.
    let fim_prompt = "<|fim_prefix|>def foo(<|fim_suffix|>):\n    pass<|fim_middle|>";
    let ids = tok.encode(fim_prompt).expect("encode fim prompt");
    assert!(ids.contains(&151659), "fim_prefix must appear: {ids:?}");
    assert!(ids.contains(&151661), "fim_suffix must appear: {ids:?}");
    assert!(ids.contains(&151660), "fim_middle must appear: {ids:?}");
}

// ── tokenizer-04: duplicate BPE vocab IDs are rejected ─────────────────────

#[test]
fn duplicate_bpe_vocab_id_is_rejected() {
    let json = r#"{
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 0},
            "merges": []
        }
    }"#;
    let err = HfTokenizerJson::parse(json).expect_err("duplicate id must be rejected");
    match err {
        TokenizerError::HfFormat(msg) => {
            assert!(
                msg.contains('0'),
                "error message should mention the colliding id: {msg}"
            );
        }
        other => panic!("expected HfFormat error, got {other:?}"),
    }
}

#[test]
fn unique_bpe_vocab_ids_parse_normally() {
    let json = r#"{
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2},
            "merges": []
        }
    }"#;
    assert!(HfTokenizerJson::parse(json).is_ok());
}
