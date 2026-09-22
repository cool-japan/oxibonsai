//! RAG-EVAL-IMG-33: BLEU smoothing's relationship to "Chen & Cherry 2014
//! m2/m3" was re-verified against the source in this package (see the
//! doc comments on `bleu::SmoothingMethod`), not against a live
//! NLTK/sacreBLEU install (unavailable offline in this environment). These
//! tests lock in the exact — hand-derived — arithmetic each variant
//! actually computes, as a regression guard for the refactor that removed
//! the duplicated `ngram_counts` (now shared with `crate::rouge`) and
//! collapsed `AddOne`'s redundant zero/non-zero branches into one formula.

use oxibonsai_eval::bleu::{sentence_bleu, BleuConfig, SmoothingMethod};

#[test]
fn add_one_smoothing_matches_hand_derived_uniform_formula() {
    // candidate="a b c", reference="a b d", max_n=1.
    // Unigram matches: {a, b} clipped-match, {c} does not ⇒ matches=2, total=3.
    // AddOne: p_1 = (2+1)/(3+1) = 0.75. Equal lengths ⇒ brevity penalty = 1.0.
    // BLEU = 1.0 * 0.75 = 0.75.
    let cfg = BleuConfig::new(1, SmoothingMethod::AddOne);
    let s = sentence_bleu("a b c", &["a b d"], &cfg);
    assert!((s.bleu - 0.75).abs() < 1e-5, "got {}", s.bleu);
    assert!(
        (s.precisions[0] - 0.75).abs() < 1e-5,
        "got {:?}",
        s.precisions
    );
}

#[test]
fn exp_decay_smoothing_matches_hand_derived_formula() {
    // candidate="a b", reference="a c", max_n=2.
    // n=1: matches={a} ⇒ matches=1, total=2 ⇒ p_1 = 1/2 = 0.5 (non-zero,
    //   no smoothing needed, `zero_streak` stays/resets to 0).
    // n=2: candidate bigram [a,b] vs reference bigram [a,c] ⇒ no match ⇒
    //   matches=0, total=1 (one candidate bigram). ExpDecay's zero branch:
    //   `zero_streak` becomes 1 (k=1), denom = 2^1 * c_len(=2) = 4,
    //   p_2 = 1/4 = 0.25 — note the denominator is the *candidate token
    //   length* (2), not the order's own n-gram count (1), which is this
    //   variant's documented (not paper-exact) behaviour.
    // Geometric mean over 2 orders: sqrt(0.5 * 0.25) = sqrt(0.125).
    // Equal candidate/reference length ⇒ brevity penalty = 1.0.
    let cfg = BleuConfig::new(2, SmoothingMethod::ExpDecay);
    let s = sentence_bleu("a b", &["a c"], &cfg);
    let expected = 0.125f32.sqrt();
    assert!(
        (s.bleu - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        s.bleu
    );
    assert!((s.precisions[0] - 0.5).abs() < 1e-5);
    assert!((s.precisions[1] - 0.25).abs() < 1e-5);
}

#[test]
fn none_smoothing_still_collapses_to_zero_on_a_missing_order() {
    // Regression guard for `SmoothingMethod::None`, unaffected by the
    // AddOne/ExpDecay changes: a missing higher-order n-gram still
    // collapses the whole score to 0, not silently smoothed.
    let cfg = BleuConfig::new(2, SmoothingMethod::None);
    let s = sentence_bleu("a b", &["a c"], &cfg);
    assert_eq!(s.bleu, 0.0);
}
