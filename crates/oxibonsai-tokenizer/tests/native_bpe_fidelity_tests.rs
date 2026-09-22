//! Regression tests for the native ByteLevel-BPE encode/decode path,
//! rebuilt from scratch for TOK-12.
//!
//! ## Why this file was rebuilt
//!
//! The previous version of this file built its central fixture
//! (`byte_level_roundtrip_tokenizer`) by deliberately moving the config's
//! `unk`/`bos`/`eos`/`pad` ids *out* of the `0..=255` byte range —
//! documented in its own comment as working around the exact defect TOK-01
//! later named and fixed at the root (`build_special_ids` blindly trusting
//! the config quartet instead of asking the vocabulary). That made this
//! file evidence that the workaround worked, not evidence that decode is
//! correct, and — per the TOK-12 verdict correction — it also missed the
//! *complementary* half of the same bug: with the old `build_special_ids`,
//! a real special token whose id did **not** happen to be one of the
//! config's four (the common case — e.g. `<|im_start|>` at some id in the
//! high hundreds) was never recognised as special at all, so it leaked
//! verbatim into decoded text. Two of this file's own tests
//! (`multiline_chat_prompt_roundtrips`,
//! `chat_template_markers_map_to_trained_ids`) asserted exactly that leak
//! as if it were correct behaviour (`decode(encode(chat_prompt)) ==
//! chat_prompt`, literal `<|im_start|>`/`<|im_end|>` markers included).
//!
//! This rebuild:
//! - drops the id-relocation workaround (specials now sit at ordinary,
//!   arbitrarily-chosen ids — the fix must handle the natural collision
//!   with `TokenizerConfig::default()`'s `unk=0,bos=1,eos=2,pad=3`
//!   directly, not avoid it);
//! - corrects the two chat-prompt round-trip tests: `decode` now correctly
//!   *skips* real specials (matching `tokenizers`' `skip_special_tokens =
//!   true` default), so the assertion is that the *content* survives, not
//!   the wrapper markers;
//! - adds the two cases the TOK-12 verdict named as missing: (1) the bare
//!   single-byte printable tokens at ids `0..=3` (`!`, `"`, `#`, `$`) as
//!   **standalone** inputs — the exact shape that triggers the collision,
//!   which a "words" corpus does not; (2) a decode-side case asserting a
//!   real special is skipped *and* a protected-but-non-special token
//!   (`<think>`-style) is kept, pinning both halves of the TOK-01 fix and
//!   the TOK-05 GGUF `token_type` carve-out rule at once;
//! - keeps the byte-level round-trip corpus (CJK, emoji, combining marks,
//!   source code, Markdown, whitespace-run preservation) and the TOK-06
//!   performance guard, both unaffected by the id-relocation bug.

use std::time::{Duration, Instant};

use oxibonsai_tokenizer::{
    bytes_to_unicode_map, BpeMerges, ChatMessage, ChatTemplateKind, HfTokenizerJson, OxiTokenizer,
    TokenizerConfig, Vocabulary,
};

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// A fully-populated ByteLevel BPE tokenizer whose vocabulary carries all
/// 256 byte-level unicode code points (ids `0..=255`) plus three real
/// ChatML specials and one protected-but-non-special reasoning marker.
///
/// Unlike the pre-TOK-12 version of this fixture, the special/protected ids
/// are **not** chosen to avoid any particular range — `TokenizerConfig`
/// stays at its plain default (`unk=0, bos=1, eos=2, pad=3`), which
/// deliberately collides with the real byte tokens for `!`/`"`/`#`/`$`.
/// That collision is exactly what `build_special_ids` must resolve
/// correctly (TOK-01): those four ids must remain ordinary, decodable
/// bytes.
fn byte_level_roundtrip_tokenizer() -> OxiTokenizer {
    let table = bytes_to_unicode_map();
    let mut vocab = Vocabulary::new();
    for b in 0u16..=255 {
        vocab.insert(&table[b as usize].to_string(), u32::from(b));
    }
    vocab.add_special("<|im_start|>", 1000);
    vocab.add_special("<|im_end|>", 1001);
    vocab.add_special("<|endoftext|>", 1002);
    // A protected-but-non-special reasoning marker (mirrors Bonsai 2's
    // `<think>`/`<tool_call>`, USER_DEFINED token_type): carved out
    // atomically on encode, but must still render on decode.
    vocab.add_protected("<think>", 1003);

    let mut config = TokenizerConfig::default();
    config.byte_level_decode = true;

    OxiTokenizer::new(vocab, BpeMerges::new(), config)
}

/// The crate's Qwen3-style fixture: ByteLevel pre-tokenizer + decoder, a
/// couple of merges, and the two real special tokens `<|endoftext|>`
/// (151643) and `<|im_start|>` (151644) — loaded through the JSON path,
/// which declares no top-level `unk_token`/`bos_token`/`eos_token`/
/// `pad_token`, exactly like the real Qwen3 `tokenizer.json` (the precise
/// precondition for the TOK-01 collision via
/// `HfTokenizerJson::into_tokenizer`'s `TokenizerConfig::default()`
/// fallback).
fn qwen3_fixture_json() -> &'static str {
    // Vocab covers exactly the characters this file's `qwen3_fixture_json`
    // tests need, with `!`/`"`/`#`/`$` deliberately pinned to ids 0..=3 --
    // matching the REAL Qwen3/GPT-2 vocab.json ordering (the byte-level
    // alphabet's "printable, self-mapping" bytes are assigned ids in
    // ascending byte order first: `!`=0x21 is the lowest, so it gets id 0)
    // and reproducing the exact TOK-01 finder evidence
    // (`decode("Hello, world!") -> "Hello, world"`, i.e. id 0 = '!').
    // `\u0120` (U+0120) is the ByteLevel remap of a literal space.
    // NOTE: a two-hash raw-string delimiter (`r##"..."##`) is required --
    // the JSON vocab's `"#": 2` entry contains the literal byte sequence
    // `"#`, which would otherwise prematurely close a single-hash `r#"..."#`
    // raw string right there.
    r##"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [
            {"id": 151643, "content": "<|endoftext|>", "special": true},
            {"id": 151644, "content": "<|im_start|>", "special": true}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "!": 0, "\"": 1, "#": 2, "$": 3,
                "a": 4, "b": 5, "c": 6, "d": 7,
                "e": 8, "H": 9, "h": 10, "i": 11,
                "s": 12, "1": 13, "5": 14, "\u0120": 15,
                "ab": 16, "cd": 17, "abcd": 18,
                "<|endoftext|>": 151643,
                "<|im_start|>": 151644
            },
            "merges": ["a b", "c d", "ab cd"]
        }
    }"##
}

fn assert_roundtrip(tok: &OxiTokenizer, text: &str) {
    let ids = tok.encode(text).expect("encode should succeed");
    let decoded = tok.decode(&ids).expect("decode should succeed");
    assert_eq!(decoded, text, "round trip mismatch (ids = {ids:?})");
}

// ── TOK-01: the real Qwen3-shaped fixture's ordinary punctuation survives ────

#[test]
fn standalone_punctuation_at_ids_0_to_3_survives_decode() {
    // The exact shape the TOK-01 finder's corpus missed: '!', '"', '#', '$'
    // as their OWN input (each is a single byte-level token occupying ids
    // 0..=3 in this fixture, colliding with TokenizerConfig::default()'s
    // unk/bos/eos/pad) rather than embedded inside a longer word.
    let tok = OxiTokenizer::from_hf_tokenizer_json(qwen3_fixture_json()).expect("load qwen3");
    for (text, expected_id) in [("!", 0u32), ("\"", 1), ("#", 2), ("$", 3)] {
        let ids = tok.encode(text).expect("encode");
        assert_eq!(ids, vec![expected_id], "encode({text:?})");
        let decoded = tok.decode(&ids).expect("decode");
        assert_eq!(
            decoded, text,
            "byte token at id {expected_id} (colliding with a \
             TokenizerConfig::default() special id) must survive decode standalone"
        );
    }
    // And together, exactly as the original finder's reproduction did.
    let ids = tok.encode("He said \"hi\" #1 $5!").expect("encode");
    let decoded = tok.decode(&ids).expect("decode");
    assert_eq!(decoded, "He said \"hi\" #1 $5!");
}

#[test]
fn real_specials_are_skipped_but_leak_verbatim_before_the_fix_would_have_shown() {
    // The symmetric half of TOK-01 (verdict correction): a real special
    // whose id does NOT coincide with any config field (151643/151644, far
    // outside the 0..=3 quartet) must still be recognised as special and
    // skipped on decode -- the old `build_special_ids(config)` (no
    // vocabulary consultation) never looked at these ids at all.
    let tok = OxiTokenizer::from_hf_tokenizer_json(qwen3_fixture_json()).expect("load qwen3");
    let ids = tok.encode("<|im_start|>a<|endoftext|>").expect("encode");
    assert!(tok.is_special(151644));
    assert!(tok.is_special(151643));
    let decoded = tok.decode(&ids).expect("decode");
    assert_eq!(
        decoded, "a",
        "real specials must be skipped on decode, not leaked as literal text"
    );
}

// ── Special/added-token protection (encode) ──────────────────────────────────

#[test]
fn special_token_emitted_atomically_not_shredded() {
    // Before the fix, `<|im_start|>` was pre-tokenized into ["<","|","im",...]
    // and every piece fell through to unk_token_id. Now it must resolve to
    // its trained ID verbatim, and the trailing `<|endoftext|>` to 151643.
    let tok = OxiTokenizer::from_hf_tokenizer_json(qwen3_fixture_json()).expect("load qwen3");
    let ids = tok.encode("<|im_start|>a<|endoftext|>").expect("encode");
    assert_eq!(ids, vec![151644, 4, 151643]);
    assert!(ids.contains(&151644), "im_start must be atomic: {ids:?}");
    assert!(ids.contains(&151643), "endoftext must be atomic: {ids:?}");
}

#[test]
fn chat_template_markers_map_to_trained_ids() {
    // The advertised flagship workflow: render a ChatML/Qwen prompt and
    // encode it. The turn-boundary markers must map to their atomic IDs and
    // decode back to the CONTENT only -- real specials are skipped on
    // decode (matching HF's `skip_special_tokens=True` default), which is
    // the corrected expectation this test previously got backwards.
    let tok = byte_level_roundtrip_tokenizer();
    let rendered = ChatTemplateKind::Qwen.render(&[ChatMessage::user("hi there")]);
    assert!(rendered.contains("<|im_start|>"));
    assert!(rendered.contains("<|im_end|>"));

    let ids = ChatTemplateKind::Qwen
        .encode(&tok, &[ChatMessage::user("hi there")])
        .expect("encode chat template");

    assert!(
        ids.contains(&1000),
        "<|im_start|> (1000) must appear atomically: {ids:?}"
    );
    assert!(
        ids.contains(&1001),
        "<|im_end|> (1001) must appear atomically: {ids:?}"
    );
    let decoded = tok.decode(&ids).expect("decode");
    assert_eq!(
        decoded, "user\nhi there\n",
        "specials must be skipped on decode; only the rendered CONTENT survives"
    );
}

#[test]
fn leftmost_longest_special_wins() {
    // Two consecutive specials with no text between them.
    let tok = byte_level_roundtrip_tokenizer();
    let ids = tok.encode("<|im_start|><|im_end|>").expect("encode");
    assert_eq!(ids, vec![1000, 1001]);
}

#[test]
fn non_special_text_unaffected_when_no_special_registered() {
    // A tokenizer with no registered special tokens must behave exactly like a
    // plain byte-level encoder (single fast-path segment).
    let table = bytes_to_unicode_map();
    let mut vocab = Vocabulary::new();
    for b in 0u16..=255 {
        vocab.insert(&table[b as usize].to_string(), u32::from(b));
    }
    let mut config = TokenizerConfig::default();
    config.byte_level_decode = true;
    let tok = OxiTokenizer::new(vocab, BpeMerges::new(), config);
    assert_roundtrip(&tok, "plain text, no specials!");
}

// ── TOK-01 / TOK-05: CONTROL is skipped, USER_DEFINED-style is rendered ──────

#[test]
fn control_special_skipped_user_defined_style_protected_token_rendered() {
    // Mirrors the GGUF token_type carve-out rule (TOK-05): a genuinely
    // special (CONTROL-equivalent) token vanishes on decode; a
    // protected-but-not-special (USER_DEFINED-equivalent) token like
    // `<think>` is carved out atomically on ENCODE but still renders on
    // DECODE, because it is real model output a caller must see.
    let tok = byte_level_roundtrip_tokenizer();
    let ids = tok
        .encode("<|im_start|><think>hi<|im_end|>")
        .expect("encode");
    assert!(ids.contains(&1003), "<think> must be atomic: {ids:?}");
    let decoded = tok.decode(&ids).expect("decode");
    assert_eq!(
        decoded, "<think>hi",
        "CONTROL-style specials skip; USER_DEFINED-style protected tokens render"
    );
}

// ── Bytes→unicode forward mapping in encode (CJK / emoji / accents) ──────────

#[test]
fn cjk_roundtrips_through_byte_level_encode() {
    let tok = byte_level_roundtrip_tokenizer();
    assert_roundtrip(&tok, "日本語のトークナイザ");
    assert_roundtrip(&tok, "中文分词器");
    assert_roundtrip(&tok, "한국어 토크나이저");
}

#[test]
fn emoji_roundtrips_through_byte_level_encode() {
    let tok = byte_level_roundtrip_tokenizer();
    assert_roundtrip(&tok, "hello 😀 world 🚀🔥");
    // ZWJ family sequence + variation selector.
    assert_roundtrip(&tok, "👨‍👩‍👧‍👦 ✅");
    // A long homogeneous emoji run with no whitespace -- the exact shape
    // that stresses `bpe_merge_symbols` with a large single pre-token
    // (TOK-06's adversarial case).
    assert_roundtrip(&tok, &"🍣".repeat(200));
}

#[test]
fn accented_latin_roundtrips_through_byte_level_encode() {
    let tok = byte_level_roundtrip_tokenizer();
    assert_roundtrip(&tok, "café naïve résumé Straße");
}

#[test]
fn combining_marks_roundtrip_through_byte_level_encode() {
    // Decomposed forms (base char + standalone combining mark), NOT run
    // through NFC here (this fixture has `normalize_nfc = false` by
    // default) -- the byte-level path must still round-trip them
    // byte-for-byte even though no composition happens.
    let tok = byte_level_roundtrip_tokenizer();
    assert_roundtrip(&tok, "e\u{0301}gal"); // e + COMBINING ACUTE ACCENT
    assert_roundtrip(&tok, "a\u{0300}\u{0301}\u{0302}"); // stacked marks
    assert_roundtrip(&tok, "न\u{094d}"); // Devanagari + virama
}

#[test]
fn source_code_roundtrips_through_byte_level_encode() {
    let tok = byte_level_roundtrip_tokenizer();
    assert_roundtrip(
        &tok,
        "def fibonacci(n: int) -> int:\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
    );
    assert_roundtrip(
        &tok,
        "fn main() {\n\tlet x: Vec<u32> = vec![1, 2, 3];\n\tprintln!(\"{:?}\", x);\n}\n",
    );
}

#[test]
fn markdown_roundtrips_through_byte_level_encode() {
    let tok = byte_level_roundtrip_tokenizer();
    assert_roundtrip(
        &tok,
        "# Title\n\n- item one\n- item **two**\n\n```rust\nlet x = 1;\n```\n\n> quote\n",
    );
}

#[test]
fn non_ascii_is_not_silently_dropped() {
    let tok = byte_level_roundtrip_tokenizer();
    let text = "aé中";
    let ids = tok.encode(text).expect("encode");
    assert_eq!(
        ids.len(),
        text.len(),
        "each UTF-8 byte must become a byte-level token: {ids:?}"
    );
}

// ── Whitespace-run preservation ───────────────────────────────────────────────

#[test]
fn tabs_newlines_and_carriage_returns_roundtrip() {
    let tok = byte_level_roundtrip_tokenizer();
    assert_roundtrip(&tok, "hello\tworld");
    assert_roundtrip(&tok, "hello\nworld");
    assert_roundtrip(&tok, "hello\r\nworld");
    assert_roundtrip(&tok, "line1\n\nline2\n");
}

#[test]
fn repeated_spaces_roundtrip() {
    let tok = byte_level_roundtrip_tokenizer();
    assert_roundtrip(&tok, "a  b");
    assert_roundtrip(&tok, "a    b");
    assert_roundtrip(&tok, "indented:    value");
    assert_roundtrip(&tok, "trailing spaces   ");
    assert_roundtrip(&tok, "   leading spaces");
}

#[test]
fn blank_line_with_trailing_whitespace_roundtrips() {
    // The exact TOK-03 divergence shape: a blank line carrying trailing
    // spaces between two other lines.
    let tok = byte_level_roundtrip_tokenizer();
    assert_roundtrip(&tok, "a\n    \n\nb");
    assert_roundtrip(&tok, "def f():\n    return 1\n    \n\nclass A:\n    pass\n");
}

#[test]
fn mixed_whitespace_structure_is_distinct_from_single_space() {
    // "hello\nworld" and "hello world" must NOT encode to the same ids — the
    // bug collapsed every whitespace run to a single space marker.
    let tok = byte_level_roundtrip_tokenizer();
    let nl = tok.encode("hello\nworld").expect("encode");
    let sp = tok.encode("hello world").expect("encode");
    assert_ne!(nl, sp, "newline must not collapse to a single space");
    assert_roundtrip(&tok, "hello\nworld");
    assert_roundtrip(&tok, "hello world");
}

#[test]
fn multiline_chat_prompt_roundtrips() {
    // Corrected expectation (TOK-12): specials are skipped on decode, so
    // the round trip reproduces the message CONTENT, not the literal
    // `<|im_start|>`/`<|im_end|>` wrapper text.
    let tok = byte_level_roundtrip_tokenizer();
    let rendered = ChatTemplateKind::Qwen.render(&[
        ChatMessage::system("You are helpful."),
        ChatMessage::user("What is 2+2?\nExplain."),
        ChatMessage::assistant("It is 4."),
    ]);
    let ids = tok.encode(&rendered).expect("encode");
    let decoded = tok.decode(&ids).expect("decode");
    assert_eq!(
        decoded,
        "system\nYou are helpful.\nuser\nWhat is 2+2?\nExplain.\nassistant\nIt is 4.\n"
    );
}

// ── TOK-06: O(1) merge-priority / O(n log n) merge-loop performance guard ────

#[test]
fn large_merge_table_encode_is_fast() {
    // A synthetic 150K-merge table (Qwen3-scale). The bulk are misses on
    // the test text; a handful of real, high-rank merges force the merge
    // loop to resolve priorities. Generous bound (see `bpe.rs`'s own
    // complexity tests for the tight `t(2n) < 3*t(n)` guard) — this is a
    // regression trip-wire against the pre-fix multi-second behaviour, not
    // a tight perf benchmark, since this machine runs other agents'
    // builds concurrently.
    let table = bytes_to_unicode_map();
    let mut vocab = Vocabulary::new();
    for b in 0u16..=255 {
        vocab.insert(&table[b as usize].to_string(), u32::from(b));
    }

    let mut merges = BpeMerges::new();
    for i in 0..150_000u32 {
        merges.add_merge(&format!("m{i}a"), &format!("m{i}b"), 0);
    }
    merges.add_merge("x", "x", 0);
    merges.add_merge("xx", "xx", 0);
    merges.add_merge("y", "y", 0);
    merges.add_merge("yy", "yy", 0);
    assert!(merges.len() >= 150_000);

    let mut config = TokenizerConfig::default();
    config.byte_level_decode = true;
    let tok = OxiTokenizer::new(vocab, merges, config);

    let text = "xxxx yyyy ".repeat(1000);
    assert!(text.len() >= 9_000);

    let start = Instant::now();
    let ids = tok.encode(&text).expect("encode");
    let elapsed = start.elapsed();

    assert!(!ids.is_empty());
    // Gatekeeper (waves 2+2.5 review) REQUIRED #2: the bound is lifted (not
    // dropped) in a debug build, which is measurably slower than release —
    // see `crates/oxibonsai-tokenizer/src/bpe.rs`'s
    // `bpe_merge_symbols_is_not_quadratic_wall_clock` doc comment for the
    // same reasoning applied there.
    let bound = if cfg!(debug_assertions) {
        Duration::from_secs(20)
    } else {
        Duration::from_secs(5)
    };
    assert!(
        elapsed < bound,
        "encoding a 10KB text with a 150K-merge vocab took {elapsed:?} (bound {bound:?}) \
         — the merge loop may have regressed to quadratic"
    );
}

// ── Sanity: the parser-loaded qwen3 path still merges correctly ──────────────

#[test]
fn qwen3_byte_level_merges_apply() {
    let parsed = HfTokenizerJson::parse(qwen3_fixture_json()).expect("parse");
    let tok = parsed.into_tokenizer().expect("into tokenizer");
    // "abcd" should merge a+b -> ab, c+d -> cd, ab+cd -> abcd (id 18).
    let ids = tok.encode("abcd").expect("encode");
    assert_eq!(
        ids,
        vec![18],
        "byte-level BPE merges must chain to abcd (18)"
    );
}
