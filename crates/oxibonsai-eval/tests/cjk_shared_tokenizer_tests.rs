//! RAG-EVAL-IMG-06: `rouge::tokenize` collapsed any unspaced CJK sentence
//! into a single token (whitespace-only splitting), so BLEU/ROUGE/METEOR of
//! an *identical* Japanese sentence scored 0.0 instead of 1.0. The verifier
//! also found the identical defect in `qa.rs`'s SQuAD F1 (never opened by
//! the original finder) — both are fixed from the one shared tokenizer in
//! `rouge::cjk_aware_tokens`, exercised here from all three call sites.

use oxibonsai_eval::bleu::{sentence_bleu, BleuConfig};
use oxibonsai_eval::qa::f1_score;
use oxibonsai_eval::rouge::{tokenize, CorpusRouge, RougeLScore, RougeNScore};

/// The exact reproduction string from the verified finding.
const JA_SENTENCE: &str = "吾輩は猫である。名前はまだ無い。";

#[test]
fn tokenize_splits_cjk_into_one_token_per_character() {
    // Before the fix this was a single element (the whole sentence minus
    // punctuation), which is why every n-gram order above 1 was empty.
    let tokens = tokenize(JA_SENTENCE);
    // 14 CJK letters: 吾輩は猫であ る (7) + 名前はまだ無い (7); the two
    // "。" punctuation marks are dropped as separators, not counted.
    assert_eq!(
        tokens.len(),
        14,
        "expected one token per CJK character, got {tokens:?}"
    );
    assert_eq!(tokens[0], "吾");
    assert_eq!(tokens[1], "輩");
}

#[test]
fn tokenize_mixed_latin_and_cjk_keeps_latin_words_whole() {
    // Latin runs stay as single lowercase word tokens; CJK runs still split
    // per character even when adjacent to Latin text.
    let tokens = tokenize("Hello 世界");
    assert_eq!(tokens, vec!["hello", "世", "界"]);
}

#[test]
fn tokenize_punctuation_only_input_is_empty_not_a_panic() {
    assert!(tokenize("... !? — 。、").is_empty());
}

#[test]
fn tokenize_emoji_only_input_is_empty_not_a_panic() {
    // Emoji are Unicode category "Symbol", not "Letter"/"Number", so
    // `char::is_alphanumeric()` is false for them — they are pure
    // separators here, same as punctuation, and must not panic.
    assert!(tokenize("😀😃😄🎉").is_empty());
}

#[test]
fn tokenize_does_not_panic_on_a_lone_combining_mark() {
    // A combining diacritic with no base character is a degenerate but
    // legal `&str`; iterating by `char` (never by byte offset) must handle
    // it without panicking.
    let _ = tokenize("\u{0301}");
}

#[test]
fn bleu_identical_japanese_sentence_is_one() {
    // ACCEPTANCE: bleu(ja, ja) == 1.0.
    let cfg = BleuConfig::default();
    let s = sentence_bleu(JA_SENTENCE, &[JA_SENTENCE], &cfg);
    assert!(
        (s.bleu - 1.0).abs() < 1e-6,
        "expected BLEU(ja, ja) = 1.0, got {}",
        s.bleu
    );
    assert!((s.brevity_penalty - 1.0).abs() < 1e-6);
}

#[test]
fn bleu_english_identical_still_one_no_regression() {
    // The English path must be completely unaffected by the CJK fallback.
    let cfg = BleuConfig::default();
    let s = sentence_bleu("the cat sat on the mat", &["the cat sat on the mat"], &cfg);
    assert!((s.bleu - 1.0).abs() < 1e-6, "got {}", s.bleu);
}

#[test]
fn rouge_identical_japanese_sentence_scores_one() {
    let r1 = RougeNScore::compute(JA_SENTENCE, JA_SENTENCE, 1);
    assert!((r1.f1 - 1.0).abs() < 1e-6, "ROUGE-1 f1={}", r1.f1);
    let rl = RougeLScore::compute(JA_SENTENCE, JA_SENTENCE);
    assert!((rl.f1 - 1.0).abs() < 1e-6, "ROUGE-L f1={}", rl.f1);
}

#[test]
fn corpus_rouge_identical_japanese_pair_scores_one() {
    // `CorpusRouge::compute` is what `oxibonsai eval` actually calls, so
    // this is the shipping-binary path the finder flagged as affected.
    let pairs = [(JA_SENTENCE, JA_SENTENCE)];
    let rouge = CorpusRouge::compute(&pairs);
    let r1 = rouge
        .rouge_1
        .expect("rouge_1 present for a non-empty corpus");
    assert!((r1.f1 - 1.0).abs() < 1e-6, "got {}", r1.f1);
}

#[test]
fn squad_f1_identical_japanese_is_one() {
    // ACCEPTANCE: squad_f1(ja, ja) == 1.0.
    let f1 = f1_score(JA_SENTENCE, JA_SENTENCE);
    assert!((f1 - 1.0).abs() < 1e-6, "got {f1}");
}

#[test]
fn squad_f1_partial_overlap_japanese_reflects_real_overlap() {
    // Verified-finding reproduction: prediction has 10 CJK characters,
    // reference has the first 3 ("東京都"), all 3 of which recur in the
    // prediction. Hand-derived: precision = 3/10, recall = 3/3 = 1.0,
    // F1 = 2·p·r/(p+r) = 2·0.3·1 / (0.3+1) = 0.6/1.3 = 6/13 ≈ 0.4615.
    // Before the fix this was 0.0 (the whole prediction collapsed to one
    // token that could never equal the reference's one token).
    let f1 = f1_score("東京都に住んでいます", "東京都");
    let expected = 6.0 / 13.0;
    assert!(
        (f1 - expected).abs() < 1e-4,
        "expected {expected}, got {f1}"
    );
    assert!(f1 > 0.0 && f1 < 1.0, "expected partial credit, got {f1}");
}

#[test]
fn squad_exact_match_still_requires_true_equality() {
    use oxibonsai_eval::qa::exact_match;
    // The CJK tokenizer fix must not make distinct Japanese strings
    // spuriously equal under exact match.
    assert_eq!(exact_match("東京都に住んでいます", "東京都"), 0.0);
    assert_eq!(exact_match("東京都", "東京都"), 1.0);
}
