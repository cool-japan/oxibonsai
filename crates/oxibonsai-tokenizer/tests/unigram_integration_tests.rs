//! Integration tests for the Unigram tokenizer via [`OxiTokenizer`].
//!
//! These tests exercise the full stack: building a `UnigramVocab`, attaching it
//! to an `OxiTokenizer` via `with_unigram`, and verifying that `encode`,
//! `decode`, and auxiliary methods behave correctly.

use oxibonsai_tokenizer::{OxiTokenizer, TokenizerConfig, UnigramVocab, Vocabulary};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a small `UnigramVocab` from a slice of `(token, log_prob)` pairs.
fn make_unigram(entries: &[(&str, f64)], unk_id: u32) -> UnigramVocab {
    UnigramVocab::new(
        entries.iter().map(|(s, p)| (s.to_string(), *p)).collect(),
        unk_id,
    )
    .expect("UnigramVocab::new should succeed in test helper")
}

/// Build an `OxiTokenizer` backed by a `UnigramVocab`.
///
/// The `Vocabulary` is populated from the same entries so that `decode` works.
fn make_tokenizer(entries: &[(&str, f64)], unk_id: u32) -> OxiTokenizer {
    let unigram_vocab = make_unigram(entries, unk_id);

    let mut vocabulary = Vocabulary::new();
    for (idx, (token, _)) in entries.iter().enumerate() {
        vocabulary.insert(token, idx as u32);
    }

    OxiTokenizer::with_unigram(vocabulary, unigram_vocab, TokenizerConfig::default())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// `with_unigram()` + `encode()` produces correct token IDs.
#[test]
fn oxi_tokenizer_with_unigram_encodes() {
    // id=0 "<unk>", id=1 "hello", id=2 " ", id=3 "world"
    let tok = make_tokenizer(
        &[
            ("<unk>", 0.0),
            ("hello", -1.0),
            (" ", -0.5),
            ("world", -1.0),
        ],
        0,
    );

    let ids = tok.encode("hello world").expect("encode should succeed");
    // "hello world" pre-tokenizes into ["hello", "world"] — space handling
    // is done by pretokenize.  At minimum the result must not be empty.
    assert!(!ids.is_empty());
    // "hello" should appear as token 1.
    assert!(ids.contains(&1), "expected 'hello' (id=1) in {ids:?}");
    // "world" should appear as token 3.
    assert!(ids.contains(&3), "expected 'world' (id=3) in {ids:?}");
}

/// `is_unigram()` returns `true` for Unigram tokenizers and `false` for BPE.
#[test]
fn oxi_tokenizer_is_unigram_flag() {
    let unigram_tok = make_tokenizer(&[("<unk>", 0.0), ("a", -1.0)], 0);
    assert!(unigram_tok.is_unigram(), "should report is_unigram == true");

    let bpe_tok = OxiTokenizer::char_level_stub(64);
    assert!(
        !bpe_tok.is_unigram(),
        "BPE stub should report is_unigram == false"
    );
}

/// Encode then decode gives back the original text (ASCII only).
///
/// The default `TokenizerConfig` reserves IDs 0-3 for UNK/BOS/EOS/PAD and
/// silently drops them in `decode`.  We put non-character tokens at 0-3 and
/// start real tokens at ID 4 so that decode does not suppress them.
#[test]
fn unigram_decode_roundtrip() {
    // id=0 reserved for <unk> (matches config default unk_token_id=0)
    // ids 1,2,3 are BOS/EOS/PAD — put placeholder entries there so the
    // unigram vocab indices align with the vocabulary.
    //
    // We use a vocabulary that starts real chars at id=4 by putting three
    // dummy entries after <unk>.
    let entries: &[(&str, f64)] = &[
        ("<unk>", 0.0), // id=0 (unk)
        ("<bos>", 0.0), // id=1 (bos, skipped in decode)
        ("<eos>", 0.0), // id=2 (eos, skipped in decode)
        ("<pad>", 0.0), // id=3 (pad, skipped in decode)
        ("h", -1.0),    // id=4
        ("e", -1.0),    // id=5
        ("l", -1.0),    // id=6
        ("o", -1.0),    // id=7
    ];

    let unigram_vocab = UnigramVocab::new(
        entries.iter().map(|(s, p)| (s.to_string(), *p)).collect(),
        0, // unk_id
    )
    .expect("UnigramVocab::new");

    let mut vocabulary = Vocabulary::new();
    vocabulary.add_special("<unk>", 0);
    vocabulary.add_special("<bos>", 1);
    vocabulary.add_special("<eos>", 2);
    vocabulary.add_special("<pad>", 3);
    vocabulary.insert("h", 4);
    vocabulary.insert("e", 5);
    vocabulary.insert("l", 6);
    vocabulary.insert("o", 7);

    let tok = OxiTokenizer::with_unigram(vocabulary, unigram_vocab, TokenizerConfig::default());

    // Pretokenize splits on whitespace; test with a single word.
    let text = "hello";
    let ids = tok.encode(text).expect("encode");
    // ids must be [4, 5, 6, 6, 7]
    assert_eq!(ids, vec![4, 5, 6, 6, 7], "unexpected encoding: {ids:?}");
    let decoded = tok.decode(&ids).expect("decode");
    assert_eq!(decoded, text, "roundtrip failed: ids={ids:?}");
}

/// Encoding a string with multibyte UTF-8 codepoints does not panic.
#[test]
fn unigram_handles_multibyte_utf8() {
    // CJK characters: 日(3 bytes) 本(3 bytes) 語(3 bytes)
    let entries: &[(&str, f64)] = &[
        ("<unk>", 0.0),
        ("日", -1.0),
        ("本", -1.0),
        ("語", -1.0),
        ("日本語", -0.5),
    ];
    let tok = make_tokenizer(entries, 0);

    // Must not panic.
    let ids = tok.encode("日本語").expect("encode should not fail");
    // Should produce at least one token.
    assert!(!ids.is_empty());
}

/// Encoding an empty string produces an empty token list.
#[test]
fn unigram_encode_empty_string() {
    let tok = make_tokenizer(&[("<unk>", 0.0), ("a", -1.0)], 0);
    let ids = tok.encode("").expect("encode empty string should succeed");
    assert!(ids.is_empty(), "empty input should yield empty token list");
}

/// Characters not in the vocabulary fall back to the UNK token.
#[test]
fn unigram_encode_unknown_chars() {
    // Vocabulary only covers "<unk>" and "a"; 'z' is unknown.
    let tok = make_tokenizer(&[("<unk>", 0.0), ("a", -1.0)], 0);

    let ids = tok
        .encode("z")
        .expect("encode should succeed with UNK fallback");
    // UNK token is id=0.
    assert_eq!(ids, vec![0], "unknown char should produce unk_id=0");
}

/// BOS/EOS injection still works for Unigram tokenizers.
#[test]
fn unigram_bos_eos_injection() {
    let unigram_vocab = make_unigram(&[("<unk>", 0.0), ("a", -1.0), ("b", -1.5)], 0);

    let mut vocabulary = Vocabulary::new();
    vocabulary.add_special("<unk>", 0);
    vocabulary.add_special("<bos>", 10);
    vocabulary.add_special("<eos>", 11);
    vocabulary.insert("a", 1);
    vocabulary.insert("b", 2);

    let mut config = TokenizerConfig::default();
    config.add_bos = true;
    config.add_eos = true;
    config.bos_token_id = 10;
    config.eos_token_id = 11;
    config.unk_token_id = 0;

    let tok = OxiTokenizer::with_unigram(vocabulary, unigram_vocab, config);
    let ids = tok.encode("a").expect("encode");
    assert_eq!(ids[0], 10, "first token should be BOS");
    assert_eq!(
        *ids.last().expect("must have last"),
        11,
        "last token should be EOS"
    );
}

/// max_length truncation applies to Unigram output as it does for BPE.
#[test]
fn unigram_max_length_truncates() {
    let entries: &[(&str, f64)] = &[
        ("<unk>", 0.0),
        ("a", -1.0),
        ("b", -1.0),
        ("c", -1.0),
        ("d", -1.0),
        ("e", -1.0),
    ];
    let unigram_vocab = make_unigram(entries, 0);

    let mut vocabulary = Vocabulary::new();
    for (idx, (token, _)) in entries.iter().enumerate() {
        vocabulary.insert(token, idx as u32);
    }

    let mut config = TokenizerConfig::default();
    config.max_length = Some(2);

    let tok = OxiTokenizer::with_unigram(vocabulary, unigram_vocab, config);
    let ids = tok.encode("abcde").expect("encode");
    assert!(
        ids.len() <= 2,
        "expected truncation to 2 tokens, got {ids:?}"
    );
}
