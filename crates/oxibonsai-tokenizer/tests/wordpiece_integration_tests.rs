//! Integration tests for the WordPiece tokenizer algorithm.
//!
//! These tests verify that [`WordPieceVocab`] and [`OxiTokenizer::with_wordpiece`]
//! behave correctly both as standalone units and when wired together through the
//! high-level `OxiTokenizer` API.

use oxibonsai_tokenizer::{
    tokenizer::TokenizerConfig, vocab::Vocabulary, OxiTokenizer, WordPieceVocab,
    WORDPIECE_CONTINUATION_PREFIX,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a minimal BERT-style vocabulary for testing.
///
/// ID assignments:
///  0 → [PAD]
///  1 → [UNK]
///  2 → [CLS]
///  3 → [SEP]
///  4 → hello
///  5 → world
///  6 → play
///  7 → ##ing
///  8 → ##s
///  9 → foo
/// 10 → ##bar
fn make_wordpiece_vocab() -> WordPieceVocab {
    let tokens: Vec<String> = vec![
        "[PAD]".into(),
        "[UNK]".into(),
        "[CLS]".into(),
        "[SEP]".into(),
        "hello".into(),
        "world".into(),
        "play".into(),
        "##ing".into(),
        "##s".into(),
        "foo".into(),
        "##bar".into(),
    ];
    WordPieceVocab::new(tokens, 1).expect("make_wordpiece_vocab must succeed")
}

/// Build a [`Vocabulary`] that mirrors the WordPiece vocab, so that the
/// high-level `OxiTokenizer` can decode IDs back to strings.
fn make_oxi_vocabulary() -> Vocabulary {
    let mut v = Vocabulary::new();
    v.add_special("[PAD]", 0);
    v.add_special("[UNK]", 1);
    v.add_special("[CLS]", 2);
    v.add_special("[SEP]", 3);
    v.insert("hello", 4);
    v.insert("world", 5);
    v.insert("play", 6);
    v.insert("##ing", 7);
    v.insert("##s", 8);
    v.insert("foo", 9);
    v.insert("##bar", 10);
    v
}

/// Build an [`OxiTokenizer`] using the WordPiece engine.
fn make_oxi_tokenizer() -> OxiTokenizer {
    let wp_vocab = make_wordpiece_vocab();
    let vocabulary = make_oxi_vocabulary();
    // TokenizerConfig is #[non_exhaustive] — build via Default then mutate.
    let mut config = TokenizerConfig::default();
    config.unk_token_id = 1;
    config.pad_token_id = 0;
    OxiTokenizer::with_wordpiece(vocabulary, wp_vocab, config)
}

// ── is_wordpiece / is_unigram flags ──────────────────────────────────────────

#[test]
fn oxi_tokenizer_with_wordpiece_reports_is_wordpiece() {
    let tok = make_oxi_tokenizer();
    assert!(tok.is_wordpiece(), "tokenizer should report is_wordpiece = true");
}

#[test]
fn oxi_tokenizer_with_wordpiece_is_not_unigram() {
    let tok = make_oxi_tokenizer();
    assert!(!tok.is_unigram(), "wordpiece tokenizer should not be unigram");
}

#[test]
fn bpe_tokenizer_is_not_wordpiece() {
    // char_level_stub builds a BPE tokenizer.
    let tok = OxiTokenizer::char_level_stub(200);
    assert!(!tok.is_wordpiece(), "BPE tokenizer must not be wordpiece");
}

// ── Encoding ──────────────────────────────────────────────────────────────────

#[test]
fn oxi_tokenizer_wordpiece_encode_single_word() {
    let tok = make_oxi_tokenizer();
    let ids = tok.encode("hello").expect("encode must succeed");
    assert_eq!(ids, vec![4u32]);
}

#[test]
fn oxi_tokenizer_wordpiece_encode_multiword() {
    let tok = make_oxi_tokenizer();
    let ids = tok.encode("hello world").expect("encode must succeed");
    assert_eq!(ids, vec![4u32, 5]);
}

#[test]
fn oxi_tokenizer_wordpiece_encode_continuation_tokens() {
    let tok = make_oxi_tokenizer();
    // "playing" → play(6) + ##ing(7)
    let ids = tok.encode("playing").expect("encode must succeed");
    assert_eq!(ids, vec![6u32, 7]);
}

#[test]
fn oxi_tokenizer_wordpiece_encode_unknown_token() {
    let tok = make_oxi_tokenizer();
    // "xyz" has no match in the vocabulary → single UNK (id=1)
    let ids = tok.encode("xyz").expect("encode must succeed");
    assert_eq!(ids, vec![1u32]);
}

#[test]
fn oxi_tokenizer_wordpiece_encode_empty_string() {
    let tok = make_oxi_tokenizer();
    let ids = tok.encode("").expect("encode must succeed");
    assert_eq!(ids, Vec::<u32>::new());
}

#[test]
fn oxi_tokenizer_wordpiece_foobar_segmentation() {
    let tok = make_oxi_tokenizer();
    // "foobar" = foo(9) + ##bar(10)
    let ids = tok.encode("foobar").expect("encode must succeed");
    assert_eq!(ids, vec![9u32, 10]);
}

// ── Decoding ─────────────────────────────────────────────────────────────────

#[test]
fn oxi_tokenizer_wordpiece_decode_single_word() {
    let tok = make_oxi_tokenizer();
    // Encode and decode "hello"; the OxiTokenizer decode path reads from
    // Vocabulary (not from WordPieceVocab) but the IDs match by construction.
    let ids = tok.encode("hello").expect("encode");
    // id 4 → "hello" in the Vocabulary; no byte-level or Ġ-stripping applies.
    let text = tok.decode(&ids).expect("decode");
    assert_eq!(text, "hello");
}

// ── WordPieceVocab standalone roundtrip ──────────────────────────────────────

#[test]
fn wordpiece_vocab_roundtrip_encode_decode() {
    let vocab = make_wordpiece_vocab();
    let text = "hello world";
    let ids = vocab.encode(text);
    let decoded = vocab.decode(&ids);
    assert_eq!(decoded, text, "encode→decode must be identity for known words");
}

#[test]
fn wordpiece_vocab_roundtrip_with_continuation() {
    let vocab = make_wordpiece_vocab();
    let ids = vocab.encode("playing");
    let decoded = vocab.decode(&ids);
    assert_eq!(decoded, "playing");
}

// ── Continuation prefix constant ─────────────────────────────────────────────

#[test]
fn continuation_prefix_constant_value() {
    assert_eq!(WORDPIECE_CONTINUATION_PREFIX, "##");
}

#[test]
fn continuation_prefix_in_vocab_decode_token() {
    let vocab = make_wordpiece_vocab();
    // Token id=7 is "##ing" — the vocabulary stores the prefix verbatim.
    assert_eq!(vocab.decode_token(7), Some("##ing"));
}

// ── Max-char limit plumbing through OxiTokenizer ─────────────────────────────

#[test]
fn wordpiece_max_chars_limit_triggers_unk() {
    // Build a very restrictive vocab with max 2 chars.
    let tokens: Vec<String> = vec!["[UNK]".into(), "ab".into()];
    let wp_vocab = WordPieceVocab::new(tokens, 0)
        .expect("vocab ok")
        .with_max_input_chars(2);
    let mut vocabulary = Vocabulary::new();
    vocabulary.add_special("[UNK]", 0);
    vocabulary.insert("ab", 1);
    // TokenizerConfig is #[non_exhaustive] — build via Default then mutate.
    let mut config = TokenizerConfig::default();
    config.unk_token_id = 0;
    let tok = OxiTokenizer::with_wordpiece(vocabulary, wp_vocab, config);
    // "abc" = 3 chars > 2 → UNK
    let ids = tok.encode("abc").expect("encode");
    assert_eq!(ids, vec![0u32], "word exceeding max chars must be UNK");
}
