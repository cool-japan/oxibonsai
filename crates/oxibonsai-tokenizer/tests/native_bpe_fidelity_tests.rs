//! Regression tests for the native ByteLevel-BPE encode path fidelity fixes:
//!
//! * special/added-token protection (chat-template markers keep their trained
//!   IDs instead of being shredded into UNK by pre-tokenization),
//! * GPT-2 bytes→unicode forward mapping wired into `encode` (CJK, emoji and
//!   accented Latin round-trip against a byte-level vocab),
//! * whitespace-run preservation (tabs, newlines, carriage returns and runs of
//!   repeated spaces survive an encode→decode round trip), and
//! * the `O(1)` merge-priority lookup (encoding a ~10KB text against a
//!   150K-merge synthetic vocab completes well within a second).
//!
//! These paths are exercised through the public `OxiTokenizer` API only.

use std::time::{Duration, Instant};

use oxibonsai_tokenizer::{
    bytes_to_unicode_map, BpeMerges, ChatMessage, ChatTemplateKind, HfTokenizerJson, OxiTokenizer,
    TokenizerConfig, Vocabulary,
};

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// Build a fully-populated ByteLevel BPE tokenizer whose vocabulary carries all
/// 256 byte-level unicode code points (ids `0..=255`) plus three ChatML special
/// tokens.  With every single byte present and no merges, any text encodes to a
/// sequence of byte tokens and decodes back byte-for-byte — the ideal round-trip
/// fixture for the bytes→unicode and whitespace-preservation fixes.
fn byte_level_roundtrip_tokenizer() -> OxiTokenizer {
    let table = bytes_to_unicode_map();
    let mut vocab = Vocabulary::new();
    for b in 0u16..=255 {
        vocab.insert(&table[b as usize].to_string(), u32::from(b));
    }
    // ChatML markers, given IDs well outside the byte range so they never
    // collide with a byte token during decode.
    vocab.add_special("<|im_start|>", 1000);
    vocab.add_special("<|im_end|>", 1001);
    vocab.add_special("<|endoftext|>", 1002);

    let mut config = TokenizerConfig::default();
    config.byte_level_decode = true;
    // Move the config special IDs out of the 0..=255 byte range so that
    // decoding a byte token (e.g. NUL → id 0) is never mistaken for a special
    // token and silently dropped.
    config.unk_token_id = 1003;
    config.bos_token_id = 1004;
    config.eos_token_id = 1005;
    config.pad_token_id = 1006;

    OxiTokenizer::new(vocab, BpeMerges::new(), config)
}

/// The crate's Qwen3-style fixture: ByteLevel pre-tokenizer + decoder, a couple
/// of merges, and the two real special tokens `<|endoftext|>` (151643) and
/// `<|im_start|>` (151644).
fn qwen3_fixture_json() -> &'static str {
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
                "Ġa": 8, "Ġb": 9,
                "<|endoftext|>": 151643,
                "<|im_start|>": 151644
            },
            "merges": ["a b", "c d", "ab cd"]
        }
    }"#
}

fn assert_roundtrip(tok: &OxiTokenizer, text: &str) {
    let ids = tok.encode(text).expect("encode should succeed");
    let decoded = tok.decode(&ids).expect("decode should succeed");
    assert_eq!(decoded, text, "round trip mismatch (ids = {ids:?})");
}

// ── Finding 8: special/added-token protection ────────────────────────────────

#[test]
fn special_token_emitted_atomically_not_shredded() {
    // Before the fix, `<|im_start|>` was pre-tokenized into ["<","|","im",...]
    // and every piece fell through to unk_token_id (0). Now it must resolve to
    // its trained ID (151644) verbatim, and the trailing `<|endoftext|>` to
    // 151643.
    let tok = OxiTokenizer::from_hf_tokenizer_json(qwen3_fixture_json()).expect("load qwen3");
    let ids = tok.encode("<|im_start|>a<|endoftext|>").expect("encode");
    assert_eq!(ids, vec![151644, 1, 151643]);
    // Explicitly assert the special IDs did not degrade into a run of UNK.
    assert!(ids.contains(&151644), "im_start must be atomic: {ids:?}");
    assert!(ids.contains(&151643), "endoftext must be atomic: {ids:?}");
    assert!(
        !ids.contains(&0),
        "no UNK (0) should be produced for the special markers: {ids:?}"
    );
}

#[test]
fn chat_template_markers_map_to_trained_ids() {
    // The advertised flagship workflow: render a ChatML/Qwen prompt and encode
    // it. The turn-boundary markers must map to their atomic IDs and the whole
    // thing must round-trip.
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
    // Full round trip of the rendered chat text.
    let decoded = tok.decode(&ids).expect("decode");
    assert_eq!(decoded, rendered);
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
    config.unk_token_id = 1000;
    config.bos_token_id = 1001;
    config.eos_token_id = 1002;
    config.pad_token_id = 1003;
    let tok = OxiTokenizer::new(vocab, BpeMerges::new(), config);
    assert_roundtrip(&tok, "plain text, no specials!");
}

// ── Finding 9: bytes→unicode forward mapping in encode ───────────────────────

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
}

#[test]
fn accented_latin_roundtrips_through_byte_level_encode() {
    let tok = byte_level_roundtrip_tokenizer();
    assert_roundtrip(&tok, "café naïve résumé Straße");
}

#[test]
fn non_ascii_is_not_silently_dropped() {
    // Before the fix, non-ASCII characters inside a pre-token were silently
    // dropped (no byte-fallback token existed). Now every UTF-8 byte becomes a
    // byte-level token, so the id count matches the raw UTF-8 byte length.
    let tok = byte_level_roundtrip_tokenizer();
    let text = "aé中";
    let ids = tok.encode(text).expect("encode");
    assert_eq!(
        ids.len(),
        text.len(),
        "each UTF-8 byte must become a byte-level token: {ids:?}"
    );
}

// ── Finding 14: whitespace-run preservation ──────────────────────────────────

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
    let tok = byte_level_roundtrip_tokenizer();
    let rendered = ChatTemplateKind::Qwen.render(&[
        ChatMessage::system("You are helpful."),
        ChatMessage::user("What is 2+2?\nExplain."),
        ChatMessage::assistant("It is 4."),
    ]);
    assert_roundtrip(&tok, &rendered);
}

// ── Finding 34: O(1) merge-priority lookup benchmark guard ───────────────────

#[test]
fn large_merge_table_encode_is_fast() {
    // Build a synthetic 150K-merge table (Qwen3-scale). The bulk are misses on
    // the test text; a handful of real, high-rank merges force the merge loop to
    // resolve priorities. With the O(total-merges) linear scan this would take
    // many seconds; the O(1) HashMap lookup keeps it in the millisecond range.
    let table = bytes_to_unicode_map();
    let mut vocab = Vocabulary::new();
    for b in 0u16..=255 {
        vocab.insert(&table[b as usize].to_string(), u32::from(b));
    }

    let mut merges = BpeMerges::new();
    for i in 0..150_000u32 {
        // Distinct pairs that never occur in the test text (pure misses).
        merges.add_merge(&format!("m{i}a"), &format!("m{i}b"), 0);
    }
    // Real merges, inserted last so they carry the highest priority ranks — the
    // worst case for a linear position() scan.
    merges.add_merge("x", "x", 0);
    merges.add_merge("xx", "xx", 0);
    merges.add_merge("y", "y", 0);
    merges.add_merge("yy", "yy", 0);
    assert!(merges.len() >= 150_000);

    let mut config = TokenizerConfig::default();
    config.byte_level_decode = true;
    config.unk_token_id = 1000;
    config.bos_token_id = 1001;
    config.eos_token_id = 1002;
    config.pad_token_id = 1003;
    let tok = OxiTokenizer::new(vocab, merges, config);

    // ~10KB of short words that repeatedly hit the high-rank merges.
    let text = "xxxx yyyy ".repeat(1000);
    assert!(text.len() >= 9_000);

    let start = Instant::now();
    let ids = tok.encode(&text).expect("encode");
    let elapsed = start.elapsed();

    assert!(!ids.is_empty());
    assert!(
        elapsed < Duration::from_secs(5),
        "encoding a 10KB text with a 150K-merge vocab took {elapsed:?} — the \
         O(1) merge-priority lookup regressed to a linear scan"
    );
}

// ── Sanity: the parser-loaded qwen3 path still merges correctly ──────────────

#[test]
fn qwen3_byte_level_merges_apply() {
    let parsed = HfTokenizerJson::parse(qwen3_fixture_json()).expect("parse");
    let tok = parsed.into_tokenizer().expect("into tokenizer");
    // "abcd" should merge a+b -> ab, c+d -> cd, ab+cd -> abcd (id 7).
    let ids = tok.encode("abcd").expect("encode");
    assert_eq!(ids, vec![7], "byte-level BPE merges must chain to abcd (7)");
}
