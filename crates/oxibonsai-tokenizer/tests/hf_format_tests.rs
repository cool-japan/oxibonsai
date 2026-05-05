//! Integration tests for HuggingFace `tokenizer.json` parsing
//! (Qwen3 / Llama-3 / Mistral / Gemma fixture snippets + error cases).
//!
//! The fixtures in this file are synthetic — they are small hand-crafted
//! subsets of the real vendor tokenizers, chosen so that test behaviour stays
//! deterministic and dependency-free.

use oxibonsai_tokenizer::{
    byte_to_unicode, bytes_to_unicode_map, unicode_to_byte, HfModelType, HfTokenizerJson,
    OxiTokenizer, TokenizerError,
};

// ── WordPiece model type ─────────────────────────────────────────────────────

/// Helper: minimal BERT-style WordPiece tokenizer.json fixture.
fn bert_wordpiece_fixture() -> &'static str {
    // Use a normal string literal (with escaped backslashes) so that the
    // `"##ing"` JSON key does not accidentally terminate a raw-string literal.
    concat!(
        r#"{"model":{"type":"WordPiece","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"#,
        r#""hello":4,"world":5,"play":6,"#,
        "\"##ing\":7",
        r#"},"unk_token":"[UNK]","max_input_chars_per_word":100},"#,
        r#""added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null}"#,
    )
}

// ── Small per-vendor fixtures ────────────────────────────────────────────────

/// Qwen3-style fixture — ByteLevel pre-tokenizer + decoder, a few merges, and
/// two special tokens lifted from the real file.
fn qwen3_fixture() -> &'static str {
    r#"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [
            {"id": 151643, "content": "<|endoftext|>", "special": true},
            {"id": 151644, "content": "<|im_start|>", "special": true}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "<unk>": 0,
                "a": 1, "b": 2, "c": 3, "d": 4,
                "ab": 5, "cd": 6, "abcd": 7,
                "\u0120a": 8, "\u0120b": 9,
                "<|endoftext|>": 151643,
                "<|im_start|>": 151644
            },
            "merges": ["a b", "c d", "ab cd"]
        }
    }"#
}

/// Llama-3 style fixture — notable for having `<|begin_of_text|>` and
/// array-form merges.
fn llama3_fixture() -> &'static str {
    r#"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "added_tokens": [
            {"id": 128000, "content": "<|begin_of_text|>", "special": true},
            {"id": 128001, "content": "<|end_of_text|>", "special": true},
            {"id": 128009, "content": "<|eot_id|>", "special": true}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "h": 0, "i": 1, "hi": 2,
                "<|begin_of_text|>": 128000,
                "<|end_of_text|>": 128001,
                "<|eot_id|>": 128009
            },
            "merges": [["h", "i"]]
        }
    }"#
}

/// Mistral / SentencePiece-inspired fixture — uses a `<s>` / `</s>` pair and
/// byte-fallback tokens.
fn mistral_fixture() -> &'static str {
    r#"{
        "added_tokens": [
            {"id": 1, "content": "<s>", "special": true},
            {"id": 2, "content": "</s>", "special": true}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "<unk>": 0,
                "<s>": 1,
                "</s>": 2,
                "a": 3, "b": 4, "ab": 5,
                "<0x41>": 65, "<0x42>": 66
            },
            "merges": ["a b"]
        }
    }"#
}

/// Gemma fixture — special `<start_of_turn>` / `<end_of_turn>` tokens and
/// ByteLevel pre-tokenizer.
fn gemma_fixture() -> &'static str {
    r#"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "added_tokens": [
            {"id": 106, "content": "<start_of_turn>", "special": true},
            {"id": 107, "content": "<end_of_turn>", "special": true}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "<unk>": 0,
                "<start_of_turn>": 106,
                "<end_of_turn>": 107,
                "a": 1, "b": 2
            },
            "merges": []
        }
    }"#
}

// ── Qwen3 ────────────────────────────────────────────────────────────────────

#[test]
fn qwen3_parses() {
    let parsed = HfTokenizerJson::parse(qwen3_fixture()).expect("qwen3 parse ok");
    assert!(parsed.vocab.contains_key("<|im_start|>"));
    assert!(parsed.byte_level);
}

#[test]
fn qwen3_special_tokens_present() {
    let parsed = HfTokenizerJson::parse(qwen3_fixture()).expect("qwen3 parse ok");
    assert!(parsed.special_tokens.contains_key("<|endoftext|>"));
    assert!(parsed.special_tokens.contains_key("<|im_start|>"));
    assert_eq!(
        parsed.special_tokens.get("<|endoftext|>").copied(),
        Some(151643)
    );
}

#[test]
fn qwen3_merge_order_preserved() {
    let parsed = HfTokenizerJson::parse(qwen3_fixture()).expect("qwen3 parse ok");
    assert_eq!(parsed.merges.len(), 3);
    assert_eq!(parsed.merges[0], ("a".to_owned(), "b".to_owned()));
    assert_eq!(parsed.merges[2], ("ab".to_owned(), "cd".to_owned()));
}

#[test]
fn qwen3_into_tokenizer_ok() {
    let parsed = HfTokenizerJson::parse(qwen3_fixture()).expect("qwen3 parse ok");
    let tok = parsed.into_tokenizer().expect("to tokenizer");
    assert!(tok.vocab_size() >= 8);
    assert!(tok.config().byte_level_decode);
}

#[test]
fn qwen3_encode_smoke() {
    let parsed = HfTokenizerJson::parse(qwen3_fixture()).expect("qwen3 parse ok");
    let tok = parsed.into_tokenizer().expect("to tokenizer");
    let ids = tok.encode("a b").expect("encode");
    assert!(!ids.is_empty());
}

// ── Llama-3 ──────────────────────────────────────────────────────────────────

#[test]
fn llama3_parses_array_merges() {
    let parsed = HfTokenizerJson::parse(llama3_fixture()).expect("llama3 parse ok");
    assert_eq!(parsed.merges.len(), 1);
    assert_eq!(parsed.merges[0], ("h".to_owned(), "i".to_owned()));
}

#[test]
fn llama3_specials_in_vocab() {
    let parsed = HfTokenizerJson::parse(llama3_fixture()).expect("llama3 parse ok");
    assert_eq!(parsed.vocab.get("<|begin_of_text|>"), Some(&128000));
    assert_eq!(parsed.vocab.get("<|eot_id|>"), Some(&128009));
}

#[test]
fn llama3_into_tokenizer() {
    let parsed = HfTokenizerJson::parse(llama3_fixture()).expect("llama3 parse ok");
    let tok = parsed.into_tokenizer().expect("to tokenizer");
    // "hi" should merge into a single token (id 2).
    let ids = tok.encode("hi").expect("encode");
    assert!(ids.contains(&2));
}

// ── Mistral ──────────────────────────────────────────────────────────────────

#[test]
fn mistral_parses() {
    let parsed = HfTokenizerJson::parse(mistral_fixture()).expect("mistral parse ok");
    assert!(parsed.special_tokens.contains_key("<s>"));
    assert!(parsed.special_tokens.contains_key("</s>"));
    // No ByteLevel detection expected.
    assert!(!parsed.byte_level);
}

#[test]
fn mistral_byte_fallback_tokens() {
    let parsed = HfTokenizerJson::parse(mistral_fixture()).expect("mistral parse ok");
    assert_eq!(parsed.vocab.get("<0x41>"), Some(&65));
    assert_eq!(parsed.vocab.get("<0x42>"), Some(&66));
}

#[test]
fn mistral_into_tokenizer() {
    let parsed = HfTokenizerJson::parse(mistral_fixture()).expect("mistral parse ok");
    let tok = parsed.into_tokenizer().expect("to tokenizer");
    assert!(tok.vocab_size() >= 7);
}

// ── Gemma ────────────────────────────────────────────────────────────────────

#[test]
fn gemma_parses() {
    let parsed = HfTokenizerJson::parse(gemma_fixture()).expect("gemma parse ok");
    assert!(parsed.special_tokens.contains_key("<start_of_turn>"));
    assert!(parsed.special_tokens.contains_key("<end_of_turn>"));
    assert!(parsed.byte_level);
}

#[test]
fn gemma_into_tokenizer() {
    let parsed = HfTokenizerJson::parse(gemma_fixture()).expect("gemma parse ok");
    let tok = parsed.into_tokenizer().expect("to tokenizer");
    assert!(tok.vocab_size() >= 5);
}

// ── Error paths ──────────────────────────────────────────────────────────────

#[test]
fn malformed_json_rejected() {
    let err = HfTokenizerJson::parse("{not valid json").expect_err("must fail");
    assert!(matches!(err, TokenizerError::HfFormat(_)));
}

#[test]
fn missing_model_rejected() {
    let err = HfTokenizerJson::parse(r#"{"other":1}"#).expect_err("must fail");
    match err {
        TokenizerError::HfFormat(msg) => assert!(msg.contains("model")),
        other => panic!("expected HfFormat, got {other:?}"),
    }
}

#[test]
fn missing_vocab_rejected() {
    let json = r#"{"model":{"type":"BPE","merges":[]}}"#;
    let err = HfTokenizerJson::parse(json).expect_err("must fail");
    match err {
        TokenizerError::HfFormat(msg) => assert!(msg.contains("vocab")),
        other => panic!("expected HfFormat, got {other:?}"),
    }
}

#[test]
fn missing_merges_rejected() {
    let json = r#"{"model":{"type":"BPE","vocab":{"a":0}}}"#;
    let err = HfTokenizerJson::parse(json).expect_err("must fail");
    match err {
        TokenizerError::HfFormat(msg) => assert!(msg.contains("merges")),
        other => panic!("expected HfFormat, got {other:?}"),
    }
}

#[test]
fn non_object_vocab_rejected() {
    let json = r#"{"model":{"vocab":[1,2,3],"merges":[]}}"#;
    let err = HfTokenizerJson::parse(json).expect_err("must fail");
    assert!(matches!(err, TokenizerError::HfFormat(_)));
}

#[test]
fn non_array_merges_rejected() {
    let json = r#"{"model":{"vocab":{"a":0},"merges":"oops"}}"#;
    let err = HfTokenizerJson::parse(json).expect_err("must fail");
    assert!(matches!(err, TokenizerError::HfFormat(_)));
}

#[test]
fn malformed_merge_entry_rejected() {
    let json = r#"{"model":{"vocab":{"a":0},"merges":[42]}}"#;
    let err = HfTokenizerJson::parse(json).expect_err("must fail");
    assert!(matches!(err, TokenizerError::HfFormat(_)));
}

#[test]
fn vocab_non_integer_id_rejected() {
    let json = r#"{"model":{"vocab":{"a":"oops"},"merges":[]}}"#;
    let err = HfTokenizerJson::parse(json).expect_err("must fail");
    assert!(matches!(err, TokenizerError::HfFormat(_)));
}

// ── Merge form variants ──────────────────────────────────────────────────────

#[test]
fn merges_string_form() {
    let json = r#"{
        "model": {"vocab": {"a":0,"b":1,"ab":2}, "merges": ["a b"]}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse");
    assert_eq!(p.merges[0], ("a".to_owned(), "b".to_owned()));
}

#[test]
fn merges_array_form() {
    let json = r#"{
        "model": {"vocab": {"a":0,"b":1,"ab":2}, "merges": [["a","b"]]}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse");
    assert_eq!(p.merges[0], ("a".to_owned(), "b".to_owned()));
}

#[test]
fn merges_mixed_form_error() {
    // Mixing forms is allowed — each entry is parsed individually.
    let json = r#"{
        "model": {"vocab": {"a":0,"b":1,"c":2,"d":3,"ab":4,"cd":5},
                  "merges": ["a b", ["c","d"]]}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("mixed merges parse");
    assert_eq!(p.merges.len(), 2);
    assert_eq!(p.merges[0], ("a".to_owned(), "b".to_owned()));
    assert_eq!(p.merges[1], ("c".to_owned(), "d".to_owned()));
}

#[test]
fn merges_triple_array_rejected() {
    let json = r#"{
        "model": {"vocab": {"a":0}, "merges": [["a","b","c"]]}
    }"#;
    let err = HfTokenizerJson::parse(json).expect_err("must fail");
    assert!(matches!(err, TokenizerError::HfFormat(_)));
}

// ── ByteLevel detection ──────────────────────────────────────────────────────

#[test]
fn byte_level_via_decoder_field() {
    let json = r#"{
        "decoder": {"type": "ByteLevel"},
        "model": {"vocab": {"a":0}, "merges": []}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse");
    assert!(p.byte_level);
}

#[test]
fn byte_level_via_pretokenizer_array() {
    let json = r#"{
        "pre_tokenizer": [{"type":"ByteLevel"},{"type":"Whitespace"}],
        "model": {"vocab": {"a":0}, "merges": []}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse");
    assert!(p.byte_level);
}

#[test]
fn no_byte_level_when_absent() {
    let json = r#"{
        "pre_tokenizer": {"type":"Whitespace"},
        "model": {"vocab": {"a":0}, "merges": []}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse");
    assert!(!p.byte_level);
}

// ── ByteLevel roundtrip across all 256 bytes ────────────────────────────────

#[test]
fn byte_level_map_256_entries_distinct() {
    let table = bytes_to_unicode_map();
    let mut seen = std::collections::HashSet::new();
    for &ch in table.iter() {
        assert!(seen.insert(ch));
    }
    assert_eq!(seen.len(), 256);
}

#[test]
fn byte_level_roundtrip_all_256() {
    for b in 0u16..=255u16 {
        let b = b as u8;
        let ch = byte_to_unicode(b);
        assert_eq!(unicode_to_byte(ch), Some(b), "roundtrip failed on {b:#x}");
    }
}

#[test]
fn byte_level_non_mapped_char_returns_none() {
    // 'ぷ' (U+3077) is well outside the 256-entry table.
    assert_eq!(unicode_to_byte('ぷ'), None);
}

// ── Added tokens semantics ───────────────────────────────────────────────────

#[test]
fn non_special_added_token_goes_into_vocab_not_specials() {
    let json = r#"{
        "added_tokens": [
            {"id": 42, "content": "foo", "special": false}
        ],
        "model": {"vocab": {"a":0}, "merges": []}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse");
    assert_eq!(p.vocab.get("foo"), Some(&42));
    assert!(!p.special_tokens.contains_key("foo"));
}

#[test]
fn special_added_token_appears_in_both() {
    let json = r#"{
        "added_tokens": [
            {"id": 99, "content": "<|special|>", "special": true}
        ],
        "model": {"vocab": {"a":0}, "merges": []}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse");
    assert_eq!(p.vocab.get("<|special|>"), Some(&99));
    assert_eq!(p.special_tokens.get("<|special|>"), Some(&99));
}

// ── Merged-token missing from vocab is silently skipped ──────────────────────

#[test]
fn merge_with_missing_merged_token_is_skipped() {
    // "a b" merge but no "ab" in the vocab — should not error.
    let json = r#"{
        "model": {"vocab": {"a":0,"b":1}, "merges": ["a b"]}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse");
    let tok = p.into_tokenizer().expect("to tokenizer");
    // Tokenizer should still encode "a".
    let ids = tok.encode("a").expect("encode");
    assert!(!ids.is_empty());
}

// ── JSON-file IO ─────────────────────────────────────────────────────────────

#[test]
fn from_json_file_roundtrip() {
    let tmp = std::env::temp_dir().join("oxibonsai_hf_test.json");
    std::fs::write(&tmp, qwen3_fixture()).expect("write tmp");
    let tok = OxiTokenizer::from_json_file(&tmp).expect("load");
    assert!(tok.vocab_size() >= 8);
    // Cleanup best-effort.
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn from_json_file_missing_path() {
    let missing = std::env::temp_dir().join("oxibonsai_nonexistent_xyz.json");
    let _ = std::fs::remove_file(&missing);
    match OxiTokenizer::from_json_file(&missing) {
        Err(TokenizerError::Io(_)) => {}
        Err(other) => panic!("expected Io error, got {other:?}"),
        Ok(_) => panic!("expected error loading a non-existent file"),
    }
}

// ── BPE model type tolerance ─────────────────────────────────────────────────

#[test]
fn non_bpe_model_still_parses_when_vocab_present() {
    // The parser is permissive about `type` field — as long as `vocab` and
    // `merges` exist, it will accept the document.  A true "non-BPE model"
    // rejection would require a `type` check; we document current behaviour.
    let json = r#"{
        "model": {"type":"WordPiece","vocab":{"a":0},"merges":[]}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse permissive");
    assert_eq!(p.vocab.len(), 1);
}

// ── bos/eos/unk/pad hints ────────────────────────────────────────────────────

#[test]
fn top_level_string_special_tokens_picked_up() {
    let json = r#"{
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model": {"vocab":{"<s>":1,"</s>":2,"a":0},"merges":[]}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse");
    assert_eq!(p.bos_token.as_deref(), Some("<s>"));
    assert_eq!(p.eos_token.as_deref(), Some("</s>"));
    let tok = p.into_tokenizer().expect("to tokenizer");
    assert_eq!(tok.bos_id(), 1);
    assert_eq!(tok.eos_id(), 2);
}

#[test]
fn object_form_special_tokens_picked_up() {
    let json = r#"{
        "bos_token": {"content":"<s>"},
        "model": {"vocab":{"<s>":7,"a":0},"merges":[]}
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse");
    assert_eq!(p.bos_token.as_deref(), Some("<s>"));
    let tok = p.into_tokenizer().expect("to tokenizer");
    assert_eq!(tok.bos_id(), 7);
}

// ── Unigram model type ───────────────────────────────────────────────────────

/// A `"type":"Unigram"` document is parsed successfully and the model type is
/// correctly identified.
#[test]
fn hf_format_parses_unigram_model_type() {
    let json = r#"{
        "model": {
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["hello", -2.3], ["world", -3.1]],
            "unk_id": 0
        }
    }"#;
    let p = HfTokenizerJson::parse(json).expect("unigram parse should succeed");
    assert_eq!(p.model_type, HfModelType::Unigram);
    // Conversion to tokenizer should also succeed.
    let tok = p.into_tokenizer().expect("into_tokenizer for Unigram");
    assert!(tok.is_unigram());
}

/// The `unigram_vocab` and `unigram_unk_id` fields are populated correctly
/// after parsing a Unigram document.
#[test]
fn hf_format_unigram_vocab_and_unk_id() {
    let json = r#"{
        "model": {
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["hello", -2.3], ["world", -3.1]],
            "unk_id": 0
        }
    }"#;
    let p = HfTokenizerJson::parse(json).expect("unigram parse");

    // unigram_unk_id should be 0.
    assert_eq!(p.unigram_unk_id, Some(0));

    // unigram_vocab should contain 3 entries.
    let uv = p.unigram_vocab.as_ref().expect("unigram_vocab must be Some");
    assert_eq!(uv.len(), 3);
    assert_eq!(uv[0].0, "<unk>");
    assert!((uv[0].1 - 0.0_f64).abs() < 1e-9);
    assert_eq!(uv[1].0, "hello");
    assert!((uv[1].1 - (-2.3_f64)).abs() < 1e-6);
    assert_eq!(uv[2].0, "world");

    // vocab_map (the token→id map derived for decode) must have 3 entries.
    assert_eq!(p.vocab.len(), 3);
    assert_eq!(p.vocab.get("<unk>"), Some(&0));
    assert_eq!(p.vocab.get("hello"), Some(&1));
    assert_eq!(p.vocab.get("world"), Some(&2));
}

/// A Unigram JSON with `model.type == "Unigram"` but no `vocab` array returns
/// a clear `HfFormat` error.
#[test]
fn hf_format_unigram_missing_vocab_errors() {
    let json = r#"{
        "model": {
            "type": "Unigram",
            "unk_id": 0
        }
    }"#;
    let err = HfTokenizerJson::parse(json).expect_err("missing unigram vocab should error");
    match err {
        TokenizerError::HfFormat(msg) => {
            assert!(
                msg.contains("vocab"),
                "error message should mention 'vocab', got: {msg}"
            );
        }
        other => panic!("expected HfFormat, got {other:?}"),
    }
}

/// Existing BPE tokenizer.json files continue to parse and produce working
/// tokenizers after the Unigram code was added.
#[test]
fn hf_format_bpe_unaffected_by_unigram_code() {
    // This is the Qwen3 fixture used elsewhere — it must still work unchanged.
    let json = r#"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [
            {"id": 151643, "content": "<|endoftext|>", "special": true}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "<unk>": 0,
                "a": 1, "b": 2, "ab": 3,
                "<|endoftext|>": 151643
            },
            "merges": ["a b"]
        }
    }"#;
    let p = HfTokenizerJson::parse(json).expect("BPE parse must succeed");
    assert_eq!(p.model_type, HfModelType::Bpe);
    assert!(p.unigram_vocab.is_none());
    assert!(p.unigram_unk_id.is_none());

    let tok = p.into_tokenizer().expect("into_tokenizer for BPE");
    assert!(!tok.is_unigram());
    assert!(tok.config().byte_level_decode);
    let ids = tok.encode("ab").expect("encode");
    // "ab" should merge into id=3.
    assert!(ids.contains(&3), "expected merged token 3 in {ids:?}");
}

// ── WordPiece model type tests ───────────────────────────────────────────────

/// A `"type":"WordPiece"` document is parsed as `HfModelType::WordPiece`.
#[test]
fn hf_wordpiece_model_type_detected() {
    let p = HfTokenizerJson::parse(bert_wordpiece_fixture()).expect("WordPiece parse ok");
    assert_eq!(
        p.model_type,
        HfModelType::WordPiece,
        "model_type must be WordPiece"
    );
}

/// The vocab map from `model.vocab` is correctly populated for WordPiece models.
#[test]
fn hf_wordpiece_vocab_mapping() {
    let p = HfTokenizerJson::parse(bert_wordpiece_fixture()).expect("WordPiece parse ok");
    // The vocab must contain all 8 tokens from the fixture.
    assert_eq!(p.vocab.len(), 8, "vocab should have 8 entries");
    assert_eq!(p.vocab.get("[PAD]"), Some(&0u32));
    assert_eq!(p.vocab.get("[UNK]"), Some(&1u32));
    assert_eq!(p.vocab.get("hello"), Some(&4u32));
    assert_eq!(p.vocab.get("##ing"), Some(&7u32));
    // WordPiece models have no merges.
    assert!(p.merges.is_empty(), "WordPiece models must not have merges");
    // Unigram-specific fields must be absent.
    assert!(p.unigram_vocab.is_none());
    assert!(p.unigram_unk_id.is_none());
}

/// The `unk_token` string from `model.unk_token` is resolved to the correct ID
/// when `into_tokenizer()` constructs the [`WordPieceVocab`].
#[test]
fn hf_wordpiece_unk_token_resolution() {
    let p = HfTokenizerJson::parse(bert_wordpiece_fixture()).expect("WordPiece parse ok");
    // unk_token must be picked up from model.unk_token.
    assert_eq!(p.unk_token.as_deref(), Some("[UNK]"));
    let tok = p.into_tokenizer().expect("into_tokenizer for WordPiece");
    assert!(tok.is_wordpiece(), "tokenizer must be WordPiece");
    // Encoding a word not in the vocabulary should produce a single UNK (id=1).
    let ids = tok.encode("unknownword").expect("encode");
    assert_eq!(ids, vec![1u32], "unknown word must map to UNK id=1");
}

/// Existing BPE tokenizer.json files are completely unaffected by the
/// WordPiece parsing branch.
#[test]
fn hf_bpe_unaffected_by_wordpiece_branch() {
    let json = r#"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "vocab": {"<unk>": 0, "a": 1, "b": 2, "ab": 3},
            "merges": ["a b"]
        }
    }"#;
    let p = HfTokenizerJson::parse(json).expect("BPE parse must still work");
    assert_eq!(p.model_type, HfModelType::Bpe);
    assert!(p.wordpiece_max_chars.is_none(), "BPE must not set wordpiece_max_chars");
    let tok = p.into_tokenizer().expect("into_tokenizer for BPE");
    assert!(!tok.is_wordpiece(), "BPE tokenizer must not be WordPiece");
    assert!(!tok.is_unigram(), "BPE tokenizer must not be Unigram");
    let ids = tok.encode("ab").expect("encode");
    assert!(ids.contains(&3), "BPE merge must still produce token 3");
}

/// A WordPiece tokenizer loaded via `from_hf_tokenizer_json` encodes words
/// correctly including continuation-piece segmentation.
#[test]
fn hf_wordpiece_into_tokenizer_and_encode() {
    let p = HfTokenizerJson::parse(bert_wordpiece_fixture()).expect("WordPiece parse ok");
    let tok = p.into_tokenizer().expect("into_tokenizer");
    // "playing" → play(6) + ##ing(7)
    let ids = tok.encode("playing").expect("encode");
    assert_eq!(ids, vec![6u32, 7], "playing must segment to play + ##ing");
    // "hello world" → hello(4) + world(5)
    let ids2 = tok.encode("hello world").expect("encode");
    assert_eq!(ids2, vec![4u32, 5]);
}

/// `max_input_chars_per_word` is parsed and respected.
#[test]
fn hf_wordpiece_max_input_chars_respected() {
    let json = r#"{
        "model": {
            "type": "WordPiece",
            "vocab": {"[UNK]": 0, "ab": 1},
            "unk_token": "[UNK]",
            "max_input_chars_per_word": 2
        }
    }"#;
    let p = HfTokenizerJson::parse(json).expect("parse ok");
    assert_eq!(p.wordpiece_max_chars, Some(2usize));
    let tok = p.into_tokenizer().expect("into_tokenizer");
    // "abc" = 3 chars > max 2 → entire word becomes UNK (id=0)
    let ids = tok.encode("abc").expect("encode");
    assert_eq!(ids, vec![0u32], "word exceeding max_input_chars_per_word must be UNK");
    // "ab" = 2 chars == max 2 → normal encoding, id=1
    let ids2 = tok.encode("ab").expect("encode");
    assert_eq!(ids2, vec![1u32]);
}

/// A WordPiece JSON missing `model.vocab` returns a clear `HfFormat` error.
#[test]
fn hf_wordpiece_missing_vocab_errors() {
    let json = r#"{
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]"
        }
    }"#;
    let err = HfTokenizerJson::parse(json).expect_err("missing vocab must error");
    match err {
        TokenizerError::HfFormat(msg) => {
            assert!(
                msg.to_lowercase().contains("vocab"),
                "error message must mention vocab, got: {msg}"
            );
        }
        other => panic!("expected HfFormat, got {other:?}"),
    }
}
