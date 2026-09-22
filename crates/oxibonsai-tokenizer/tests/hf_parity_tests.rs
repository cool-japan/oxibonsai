//! HF-parity tests for `oxibonsai-tokenizer` (B2-08 acceptance criteria).
//!
//! Covers the parts of the design-doc §8.2 B2-08 acceptance criterion that
//! are reachable **without** the real Bonsai 2 27B GGUF vocabulary (248 320
//! tokens / 247 587 merges), which is not available in this sandboxed
//! worktree:
//!
//! - **Decode-side exactness against the real golden**
//!   (`scratchpad/golden2/tokenize.json`, transcribed inline below — never
//!   read from a scratchpad path at runtime, per the "no absolute paths in
//!   tests" rule): for all 5 golden texts, build a byte-level vocabulary
//!   from the golden's own `(id, piece)` pairs and assert `decode(ids) ==
//!   text` exactly. This exercises the exact real `(id, piece, byte)`
//!   triples PrismML's own tokenizer produced, including the emoji
//!   byte-fallback split (three tokens reassembling one 🍣 codepoint) and
//!   `<think>` as the single special token id 248068 — the two cases the
//!   acceptance criterion names explicitly. It also, incidentally,
//!   reproduces the exact TOK-01 collision (id 0 = `!`, matching this
//!   crate's default `unk_token_id`) inside real, unmodified golden data.
//!   *Encode-side* byte-exactness needs the full 247 587-entry merge table
//!   and is out of reach here — recorded as a deviation.
//! - **The qwen35 vs qwen2 pre-tokenizer delta** (TOK-04): the verified
//!   `\p{M}` (combining mark) admission in two places, using synthetic
//!   vocabularies (since the real qwen35 vocab is unavailable).
//! - **The NFC divergence** (TOK-09): the proven HF behaviour (composing
//!   `"e" + U+0301` into `"é"` before encoding) reproduced with a small
//!   synthetic vocabulary containing both the composed token and the
//!   decomposed base-character tokens.
//! - **`decode(encode(s)) == s`** as a proptest over arbitrary UTF-8 (the
//!   acceptance criterion's literal wording), and its NFC-aware variant
//!   `decode(encode(s)) == NFC(s)` once normalization is enabled (TOK-12
//!   verdict correction).
//! - **`add_bos = false` respected** via [`OxiTokenizer::from_gguf_metadata`]
//!   (TOK-05), using a synthetic `MetadataStore` shaped like the real
//!   Bonsai 2 GGUF header (`tokenizer.ggml.add_bos_token = false`).

use oxibonsai_tokenizer::bpe::{pretokenize_by_kind, PreTokenizerKind};
use oxibonsai_tokenizer::hf_format::bytes_to_unicode_map;
use oxibonsai_tokenizer::{BpeMerges, OxiTokenizer, TokenizerConfig, Vocabulary};

// ── Golden decode reconstruction (design §7.0 / §8.2 acceptance) ────────────

/// Map `piece` (its literal UTF-8 bytes) through the GPT-2 byte-level
/// alphabet, producing the vocabulary *key* a real byte-level
/// `tokenizer.json` / GGUF vocab would store for it. Used uniformly for
/// every kind of piece transcribed below (plain ASCII, multi-byte CJK,
/// control characters, and the literal `<|...|>`/`<think>` special-token
/// spellings — all of which happen to be self-inverse under this table
/// since they are pure printable ASCII).
fn byte_level_key(piece_bytes: &[u8]) -> String {
    let table = bytes_to_unicode_map();
    piece_bytes.iter().map(|&b| table[b as usize]).collect()
}

/// Build a byte-level [`OxiTokenizer`] whose vocabulary is *exactly* the
/// given `(id, piece_bytes)` pairs (no merges — decode only needs the
/// vocabulary), then assert `decode(ids) == expected_text` for the ids in
/// the given order (repeats allowed, matching `<|im_start|>` appearing
/// twice in the golden's 5th text).
fn assert_golden_decode(pieces: &[(u32, &[u8])], expected_text: &str) {
    let mut vocab = Vocabulary::new();
    for &(id, bytes) in pieces {
        vocab.insert(&byte_level_key(bytes), id);
    }
    let mut config = TokenizerConfig::default();
    config.byte_level_decode = true;
    let tok = OxiTokenizer::new(vocab, BpeMerges::new(), config);
    let ids: Vec<u32> = pieces.iter().map(|&(id, _)| id).collect();
    let decoded = tok.decode(&ids).expect("decode should succeed");
    assert_eq!(
        decoded, expected_text,
        "golden decode mismatch (ids = {ids:?})"
    );
}

#[test]
fn golden_text_1_the_capital_of_japan_is() {
    assert_golden_decode(
        &[
            (760, b"The"),
            (6511, b" capital"),
            (314, b" of"),
            (6124, b" Japan"),
            (369, b" is"),
        ],
        "The capital of Japan is",
    );
}

#[test]
fn golden_text_2_def_fibonacci() {
    assert_golden_decode(
        &[
            (727, b"def"),
            (73111, b" fibonacci"),
            (1393, b"(n"),
            (1590, b"):"),
        ],
        "def fibonacci(n):",
    );
}

#[test]
fn golden_text_3_once_upon_a_time() {
    assert_golden_decode(
        &[
            (12162, b"Once"),
            (5028, b" upon"),
            (264, b" a"),
            (854, b" time"),
            (11, b","),
            (303, b" in"),
            (264, b" a"),
            (2526, b" small"),
            (13721, b" village"),
            (539, b" by"),
            (279, b" the"),
            (9117, b" sea"),
            (11, b","),
        ],
        "Once upon a time, in a small village by the sea,",
    );
}

#[test]
fn golden_text_4_emoji_byte_fallback_and_punctuation_collision() {
    // The acceptance criterion's named case: the emoji byte-fallback split
    // (🍣 = U+1F363, UTF-8 F0 9F 8D A3, reassembled here from a leading
    // space + 2 bytes, then 1 byte, then 1 byte across 3 tokens) AND id 0 =
    // '!' — the exact TOK-01 collision (this crate's default
    // `unk_token_id`), occurring here inside real, unmodified golden data.
    assert_golden_decode(
        &[
            (9419, b"Hello"),
            (11, b","),
            (1814, b" world"),
            (0, b"!"),
            (220, b" "),
            (247359, "日本語".as_bytes()),
            (15303, "の".as_bytes()),
            (210342, "テキスト".as_bytes()),
            (36298, "です".as_bytes()),
            (1710, "。".as_bytes()),
            (10838, &[32, 240, 159]),
            (235, &[141]),
            (96, &[163]),
            (1228, b" test"),
            (16, b"1"),
            (17, b"2"),
            (18, b"3"),
            (628, b" can"),
            (914, b"'t"),
            (2677, b" won"),
            (914, b"'t"),
            (220, b" "),
            (1923, b" double"),
            (220, b" "),
            (3433, b" space"),
            (271, b"\n\n"),
            (902, b"new"),
            (7718, b"lines"),
            (197, b"\t"),
            (8320, b"Tab"),
        ],
        "Hello, world! 日本語のテキストです。 🍣 test123 can't won't  double  space\n\nnewlines\tTab",
    );
}

#[test]
fn golden_text_5_think_is_a_single_special_token() {
    // The acceptance criterion's other named case: `<think>` = 248068 is a
    // SINGLE token (not shredded into `<`,`t`,`h`,... by pre-tokenization).
    assert_golden_decode(
        &[
            (248045, b"<|im_start|>"),
            (846, b"user"),
            (198, b"\n"),
            (12675, b"Hi"),
            (248046, b"<|im_end|>"),
            (198, b"\n"),
            (248045, b"<|im_start|>"),
            (74455, b"assistant"),
            (198, b"\n"),
            (248068, b"<think>"),
            (198, b"\n"),
        ],
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n",
    );
}

// ── TOK-04: qwen35 vs qwen2 -- the \p{M} (combining mark) delta ─────────────

#[test]
fn qwen35_admits_combining_marks_into_a_letter_run_qwen2_does_not() {
    // Verified delta (design Appendix A.2 / llama-vocab.cpp:373-388): qwen35
    // adds `\p{M}` to the letter-run alternative. A base letter immediately
    // followed by a combining mark must stay in ONE piece under qwen35, but
    // split into two pieces under qwen2/gpt2 (the mark falls through to the
    // "symbol run" alternative there, since it is not `\p{L}`).
    let text = "e\u{0301}gal"; // "e" + COMBINING ACUTE ACCENT + "gal"

    let qwen35_pieces = pretokenize_by_kind(text, PreTokenizerKind::Qwen35);
    assert_eq!(
        qwen35_pieces,
        vec!["e\u{0301}gal".to_string()],
        "qwen35 must keep the base letter + combining mark + trailing letters together: {qwen35_pieces:?}"
    );

    let qwen2_pieces = pretokenize_by_kind(text, PreTokenizerKind::Qwen2);
    assert_ne!(
        qwen2_pieces, qwen35_pieces,
        "qwen2 must NOT admit \\p{{M}} into the letter run (that is the whole delta)"
    );
    // Losslessness holds for both regardless of the split boundary.
    assert_eq!(qwen35_pieces.concat(), text);
    assert_eq!(qwen2_pieces.concat(), text);
}

#[test]
fn qwen35_admits_combining_marks_into_the_symbol_run_negation_too() {
    // The second verified \p{M} insertion point: `[^\s\p{L}\p{M}\p{N}]+`
    // (the symbol-run alternative) must NOT swallow a combining mark that
    // immediately follows a run of symbols under qwen35 (it is excluded
    // from that character class precisely so it stays available for the
    // letter-run alternative instead).
    let text = "!!!\u{0301}"; // symbol run followed by a bare combining mark
    let pieces = pretokenize_by_kind(text, PreTokenizerKind::Qwen35);
    assert_eq!(pieces.concat(), text, "must stay lossless: {pieces:?}");
    // The combining mark must not have been silently fused into the
    // symbol-run piece as if it were an ordinary symbol character sharing
    // its run: it is not \p{L} either, so — with no preceding letter to
    // attach to — it becomes its own byte-level piece via the fallback
    // branches, distinctly from "!!!\u{0301}" as one glued token.
    assert!(
        pieces.len() >= 2,
        "combining mark after a symbol run must not silently fuse into it: {pieces:?}"
    );
}

// ── TOK-09: NFC divergence proof ────────────────────────────────────────────

#[test]
fn nfc_composes_decomposed_input_before_encoding() {
    // The exact proven divergence: HF composes "e" + U+0301 into a single
    // "é" before BPE, so it tokenizes identically to the precomposed form.
    // Reproduced with a small vocabulary where "é" (precomposed) has its
    // own dedicated token, while "e" and the bare combining accent do not
    // merge into it without NFC.
    let mut vocab = Vocabulary::new();
    vocab.insert("\u{00e9}", 100); // precomposed "é"
    vocab.insert("g", 1);
    vocab.insert("a", 2);
    vocab.insert("l", 3);
    vocab.insert("e", 4);
    vocab.insert("\u{0301}", 5); // bare combining acute accent (fallback char)

    let base_config = TokenizerConfig::default();

    let without_nfc = OxiTokenizer::new(vocab.clone(), BpeMerges::new(), base_config.clone());
    let mut nfc_config = base_config;
    nfc_config.normalize_nfc = true;
    let with_nfc = OxiTokenizer::new(vocab, BpeMerges::new(), nfc_config);

    let decomposed = "e\u{0301}gal";
    let precomposed = "\u{00e9}gal";

    // Without NFC: the decomposed form's "e" and the combining accent
    // remain separate characters/tokens -- pretokenize_gpt2 groups the
    // whole letter-run together as ONE piece ("e\u{0301}gal", since `e`,
    // the combining mark under `char::is_alphabetic` in the legacy
    // non-byte-level path... — regardless of pretokenization grouping, the
    // BPE character-level fallback means the two forms produce DIFFERENT
    // id sequences (the precomposed "é" resolves to id 100 directly; the
    // decomposed form cannot, since no merge joins "e"+"\u{0301}" into
    // "é").
    let ids_decomposed_no_nfc = without_nfc.encode(decomposed).expect("encode");
    let ids_precomposed_no_nfc = without_nfc.encode(precomposed).expect("encode");
    assert_ne!(
        ids_decomposed_no_nfc, ids_precomposed_no_nfc,
        "without NFC, decomposed and precomposed forms must diverge (the proven bug)"
    );

    // With NFC: both forms compose to the identical token sequence.
    let ids_decomposed_with_nfc = with_nfc.encode(decomposed).expect("encode");
    let ids_precomposed_with_nfc = with_nfc.encode(precomposed).expect("encode");
    assert_eq!(
        ids_decomposed_with_nfc, ids_precomposed_with_nfc,
        "with NFC, decomposed and precomposed forms must tokenize identically"
    );
    assert!(
        ids_decomposed_with_nfc.contains(&100),
        "NFC must compose the accent onto 'e', producing the dedicated 'é' token: {ids_decomposed_with_nfc:?}"
    );
}

#[test]
fn unsupported_normalizer_errors_rather_than_silently_skipping() {
    // Spec requirement: "ERROR on an unsupported normalizer rather than
    // silently skipping it."
    let json = r#"{
        "normalizer": {"type": "NFKC"},
        "model": {"vocab": {"a": 0}, "merges": []}
    }"#;
    let err = match oxibonsai_tokenizer::HfTokenizerJson::parse(json) {
        Err(e) => e,
        Ok(_) => panic!("an unsupported normalizer must be a hard parse error"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("NFKC"),
        "error should name the unsupported normalizer type: {msg}"
    );
}

#[test]
fn nfc_wrapped_in_a_sequence_is_recognised() {
    let json = r#"{
        "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFC"}]},
        "model": {"vocab": {"a": 0}, "merges": []}
    }"#;
    let parsed = oxibonsai_tokenizer::HfTokenizerJson::parse(json).expect("should parse");
    assert!(parsed.normalize_nfc);
}

// ── TOK-09 gap closure: post_processor is no longer silently ignored ───────

#[test]
fn post_processor_absent_parses_fine() {
    let json = r#"{"model": {"vocab": {"a": 0}, "merges": []}}"#;
    assert!(oxibonsai_tokenizer::HfTokenizerJson::parse(json).is_ok());
}

#[test]
fn post_processor_null_parses_fine() {
    let json = r#"{
        "post_processor": null,
        "model": {"vocab": {"a": 0}, "merges": []}
    }"#;
    assert!(oxibonsai_tokenizer::HfTokenizerJson::parse(json).is_ok());
}

#[test]
fn post_processor_byte_level_is_accepted_as_a_documented_noop() {
    // Real Qwen2/Qwen3-family files may pair a `ByteLevel` post-processor
    // with the `ByteLevel` pre_tokenizer/decoder -- it only patches
    // offset bookkeeping, never the token ids, so this must not error.
    let json = r#"{
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false},
        "model": {"vocab": {"a": 0}, "merges": []}
    }"#;
    assert!(oxibonsai_tokenizer::HfTokenizerJson::parse(json).is_ok());
}

#[test]
fn post_processor_byte_level_wrapped_in_sequence_is_accepted() {
    let json = r#"{
        "post_processor": {"type": "Sequence", "processors": [{"type": "ByteLevel"}]},
        "model": {"vocab": {"a": 0}, "merges": []}
    }"#;
    assert!(oxibonsai_tokenizer::HfTokenizerJson::parse(json).is_ok());
}

#[test]
fn unsupported_post_processor_errors_rather_than_silently_dropping() {
    // The proven TOK-09-shaped gap this closes: a `TemplateProcessing`
    // post-processor declaring BOS/EOS insertion must not be silently
    // dropped -- it must be a hard parse error naming the type.
    let json = r#"{
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]}}
        },
        "model": {"vocab": {"a": 0, "<s>": 1}, "merges": []}
    }"#;
    let err = match oxibonsai_tokenizer::HfTokenizerJson::parse(json) {
        Err(e) => e,
        Ok(_) => panic!("a TemplateProcessing post_processor must be a hard parse error"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("TemplateProcessing"),
        "error should name the unsupported post_processor type: {msg}"
    );
}

#[test]
fn post_processor_missing_type_field_errors() {
    let json = r#"{
        "post_processor": {"single": []},
        "model": {"vocab": {"a": 0}, "merges": []}
    }"#;
    assert!(oxibonsai_tokenizer::HfTokenizerJson::parse(json).is_err());
}

// ── Minor finding: Split `behavior`/`invert` must not be silently ignored ──

#[test]
fn split_without_behavior_or_invert_fields_still_parses() {
    // The shape every existing fixture in this crate already uses
    // (`added_token_protection_tests.rs`): no explicit `behavior`/`invert`
    // must keep working exactly as before (defaults to Isolated/false).
    let json = r#"{
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\s+"}},
                {"type": "ByteLevel"}
            ]
        },
        "model": {"vocab": {"a": 0}, "merges": []}
    }"#;
    let parsed = oxibonsai_tokenizer::HfTokenizerJson::parse(json).expect("should parse");
    assert_eq!(parsed.pretokenize_pattern.as_deref(), Some("\\s+"));
}

#[test]
fn split_with_explicit_isolated_behavior_still_parses() {
    let json = r#"{
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "a+"}, "behavior": "Isolated"},
        "model": {"vocab": {"a": 0}, "merges": []}
    }"#;
    let parsed = oxibonsai_tokenizer::HfTokenizerJson::parse(json).expect("should parse");
    assert_eq!(parsed.pretokenize_pattern.as_deref(), Some("a+"));
}

#[test]
fn split_with_non_isolated_behavior_errors_rather_than_silently_mis_splitting() {
    let json = r#"{
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "MergedWithPrevious"},
        "model": {"vocab": {"a": 0}, "merges": []}
    }"#;
    let err = match oxibonsai_tokenizer::HfTokenizerJson::parse(json) {
        Err(e) => e,
        Ok(_) => panic!("a non-Isolated Split behavior must be a hard parse error"),
    };
    assert!(err.to_string().contains("MergedWithPrevious"));
}

#[test]
fn split_with_invert_true_errors_rather_than_silently_mis_splitting() {
    let json = r#"{
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\s+"}, "invert": true},
        "model": {"vocab": {"a": 0}, "merges": []}
    }"#;
    let err = match oxibonsai_tokenizer::HfTokenizerJson::parse(json) {
        Err(e) => e,
        Ok(_) => panic!("Split invert=true must be a hard parse error"),
    };
    assert!(err.to_string().contains("invert"));
}

// ── Round-trip proptests (acceptance: decode(encode(s)) == s) ───────────────

mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn full_byte_level_tokenizer(normalize_nfc: bool) -> OxiTokenizer {
        let table = bytes_to_unicode_map();
        let mut vocab = Vocabulary::new();
        for b in 0u16..=255 {
            vocab.insert(&table[b as usize].to_string(), u32::from(b));
        }
        let mut config = TokenizerConfig::default();
        config.byte_level_decode = true;
        config.normalize_nfc = normalize_nfc;
        OxiTokenizer::new(vocab, BpeMerges::new(), config)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Acceptance criterion, verbatim: `decode(encode(s)) == s` for
        /// arbitrary UTF-8, with no merges (every character round-trips as
        /// its own byte-level token(s)).
        #[test]
        fn decode_encode_roundtrip_arbitrary_utf8(s in "\\PC{0,128}") {
            let tok = full_byte_level_tokenizer(false);
            let ids = tok.encode(&s).expect("encode should succeed");
            let decoded = tok.decode(&ids).expect("decode should succeed");
            prop_assert_eq!(decoded, s);
        }

        /// TOK-12 verdict correction: once NFC is enabled, the round trip
        /// target is `NFC(s)`, not `s` -- decomposed input is expected to
        /// come back precomposed.
        #[test]
        fn decode_encode_roundtrip_matches_nfc_when_enabled(s in "\\PC{0,128}") {
            use unicode_normalization::UnicodeNormalization;
            let tok = full_byte_level_tokenizer(true);
            let ids = tok.encode(&s).expect("encode should succeed");
            let decoded = tok.decode(&ids).expect("decode should succeed");
            let expected: String = s.nfc().collect();
            prop_assert_eq!(decoded, expected);
        }
    }
}

// ── TOK-05: add_bos = false respected via the GGUF loader ───────────────────

#[test]
fn gguf_loader_respects_add_bos_false() {
    // Synthetic MetadataStore shaped like the real Bonsai 2 27B header:
    // `tokenizer.ggml.add_bos_token = False`.
    fn str_bytes(s: &str) -> Vec<u8> {
        let mut bytes = (s.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(s.as_bytes());
        bytes
    }
    fn kv_string(key: &str, value: &str, data: &mut Vec<u8>, count: &mut u64) {
        data.extend_from_slice(&str_bytes(key));
        data.extend_from_slice(
            &(oxibonsai_core::gguf::types::GgufValueType::String as u32).to_le_bytes(),
        );
        data.extend_from_slice(&str_bytes(value));
        *count += 1;
    }
    fn kv_bool(key: &str, value: bool, data: &mut Vec<u8>, count: &mut u64) {
        data.extend_from_slice(&str_bytes(key));
        data.extend_from_slice(
            &(oxibonsai_core::gguf::types::GgufValueType::Bool as u32).to_le_bytes(),
        );
        data.push(u8::from(value));
        *count += 1;
    }
    fn kv_string_array(key: &str, values: &[&str], data: &mut Vec<u8>, count: &mut u64) {
        data.extend_from_slice(&str_bytes(key));
        data.extend_from_slice(
            &(oxibonsai_core::gguf::types::GgufValueType::Array as u32).to_le_bytes(),
        );
        data.extend_from_slice(
            &(oxibonsai_core::gguf::types::GgufValueType::String as u32).to_le_bytes(),
        );
        data.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            data.extend_from_slice(&str_bytes(v));
        }
        *count += 1;
    }
    fn kv_u32(key: &str, value: u32, data: &mut Vec<u8>, count: &mut u64) {
        data.extend_from_slice(&str_bytes(key));
        data.extend_from_slice(
            &(oxibonsai_core::gguf::types::GgufValueType::Uint32 as u32).to_le_bytes(),
        );
        data.extend_from_slice(&value.to_le_bytes());
        *count += 1;
    }

    let mut data = Vec::new();
    let mut count = 0u64;
    kv_string("tokenizer.ggml.model", "gpt2", &mut data, &mut count);
    kv_string("tokenizer.ggml.pre", "qwen35", &mut data, &mut count);
    kv_string_array(
        "tokenizer.ggml.tokens",
        &["<|endoftext|>", "<|im_end|>", "a", "b"],
        &mut data,
        &mut count,
    );
    kv_u32("tokenizer.ggml.bos_token_id", 0, &mut data, &mut count);
    kv_u32("tokenizer.ggml.eos_token_id", 1, &mut data, &mut count);
    kv_bool("tokenizer.ggml.add_bos_token", false, &mut data, &mut count);

    let (store, _) = oxibonsai_core::MetadataStore::parse(&data, 0, count)
        .expect("well-formed synthetic GGUF metadata block");

    let tok = OxiTokenizer::from_gguf_metadata(&store).expect("gguf tokenizer should build");
    assert!(
        !tok.config().add_bos,
        "add_bos_token = false in the GGUF must be respected"
    );
    // And it must actually take effect: no BOS id prepended.
    let ids = tok.encode("a").expect("encode");
    assert_ne!(ids.first().copied(), Some(0));
}
