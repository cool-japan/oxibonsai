//! RAG-EVAL-IMG-05: effective-order chrF++.
//!
//! Before the fix, an n-gram order for which the reference had no n-grams
//! (i.e. the reference was shorter than that order) was scored as a hard
//! `0.0` and folded into the average anyway, so `chrf("a", "a")` was
//! `0.1667` and `chrf("abc", "abc")` was `0.5` instead of `1.0`.
//!
//! A second, narrower regression was caught in review: the gate that fixed
//! the above checked only the *reference* side (`reference.len() < n`), not
//! the candidate side, in **both** `order_f_beta_chars` and
//! `order_f_beta_words`. sacreBLEU's effective order counts an order only
//! when `n_hyp > 0 AND n_ref > 0` — both sides must have at least one
//! n-gram of that length. With only the reference gated, a *short candidate
//! against a long reference* still had every order the reference reached
//! counted, including orders the candidate could not possibly score on,
//! which were forced to a hard `F=0.0` and dragged the average down
//! (`chrf("a", "abcdef")` was `0.033333`, not `0.200000`). The fix gates on
//! `reference.len() < n || cand.len() < n` in both functions — see
//! `crates/oxibonsai-eval/src/chrf.rs`.
//!
//! ## Reference values — provenance
//!
//! This environment has no network access, so no live `sacrebleu` install
//! was run here; nothing in this file is the literal output of a
//! `sacrebleu` invocation. The six pairs in
//! [`chrf_matches_sacrebleu_reference_values_on_whitespace_free_pairs`]
//! were instead cross-checked two independent ways — re-derived directly
//! from the F-beta definition this module implements
//! (`F = (1+β²)·p·r / (β²·p+r)`, β=2 ⇒ β²=4), and against a reviewer's
//! independent re-implementation of sacreBLEU's `_compute_f_score` — and
//! the two agree to 6 decimal places on all six. They are labelled as
//! sacreBLEU's *computed* values for this subset, not as the output of an
//! actual sacreBLEU run.
//!
//! What makes agreement here structural rather than incidental is that all
//! six pairs fall where this crate's two other, still-open divergences
//! from sacreBLEU (documented in `chrf.rs`'s module doc) cannot move the
//! result:
//!
//! - **No whitespace** in any of the six strings, so the open
//!   whitespace-stripping divergence (sacreBLEU strips whitespace before
//!   extracting character n-grams; this crate does not) never triggers.
//! - **Equal-length inputs, or only one effective order.** `("a","a")`,
//!   `("abc","abc")`, `("hello","hello")` and `("ab","ac")` all compare
//!   equal-length strings, which makes n-gram totals equal on both sides
//!   at every order, hence `p_n == r_n` and `F_n == p_n` exactly — so the
//!   open "average `F_n`" vs. "average `p_n`/`r_n` separately, combine
//!   once" divergence cannot move the result either. `("a","abcdef")` and
//!   `("abcdef","a")` have only a single effective order (`n=1`) once the
//!   both-sides gate is applied, so there is nothing left to average
//!   between the two methods.
//!
//! This closes package spec item 4's acceptance clause ("gate against
//! sacreBLEU reference values embedded in the test") for the subset where
//! this implementation and sacreBLEU are provably identical, and replaces
//! the earlier hand-derived-only vectors for these same pairs.

use oxibonsai_eval::chrf::{chrf, chrf_plus_plus};

#[test]
fn chrf_self_comparison_is_one_for_short_strings_one_to_eight_chars() {
    // For any non-empty `s`, comparing it to itself makes candidate and
    // reference n-gram multisets identical at every order both reach, so
    // precision = recall = 1.0 and F = 1.0 for every order the effective-
    // order rule actually includes (at minimum n=1, always includable for
    // a non-empty string) — hence the average is exactly 1.0 regardless of
    // how short `s` is. This is the direct acceptance criterion from the
    // package spec ("chrf(s,s) == 1.0 for 1..8-char strings").
    for len in 1..=8usize {
        let s: String = "abcdefgh".chars().take(len).collect();
        let score = chrf(&s, &s).score;
        assert!(
            (score - 1.0).abs() < 1e-6,
            "chrf({s:?}, {s:?}) = {score}, expected 1.0 (len={len})"
        );
    }
}

#[test]
fn chrf_self_comparison_is_one_even_past_the_default_order() {
    // Default order is 6; strings longer than that must still self-score
    // 1.0 once every order 1..=6 is populated by a long-enough reference.
    let s = "the quick brown fox jumps over";
    let score = chrf(s, s).score;
    assert!((score - 1.0).abs() < 1e-6, "got {score}");
}

#[test]
fn chrf_matches_sacrebleu_reference_values_on_whitespace_free_pairs() {
    // See the module doc above for the provenance of these six values and
    // why they are provably (not just empirically) comparable between this
    // implementation and sacreBLEU.
    let cases: &[(&str, &str, f32)] = &[
        ("a", "a", 1.000_000),
        ("abc", "abc", 1.000_000),
        ("hello", "hello", 1.000_000),
        ("ab", "ac", 0.250_000),
        ("a", "abcdef", 0.200_000),
        ("abcdef", "a", 0.500_000),
    ];
    for &(cand, reference, expected) in cases {
        let score = chrf(cand, reference).score;
        assert!(
            (score - expected).abs() < 1e-5,
            "chrf({cand:?}, {reference:?}) = {score}, expected {expected}"
        );
    }
}

#[test]
fn chrf_plus_plus_word_order_gate_also_checks_the_candidate_side() {
    // Same regression as the char-order case above, but for
    // `order_f_beta_words` (chrf.rs:165) — the *other* location the
    // finding named. This is a hand-derived-and-machine-verified
    // structural guard, not a sacreBLEU claim: both strings contain
    // whitespace, so this crate's still-open whitespace-stripping
    // divergence from sacreBLEU applies and a real sacreBLEU run would not
    // necessarily match this exact number.
    //
    // candidate = "ab" (1 word), reference = "ab cd" (2 words), chrF++
    // (char order=6, word order=2, β=2).
    //
    // Effective orders under the *fixed* (both-sides) gate: char n=1, char
    // n=2 (candidate reaches both; higher char orders don't), word n=1
    // (both sides reach it) — word n=2 is correctly excluded because the
    // 1-word candidate has zero word-bigrams.
    //   char n=1: p=2/2=1.0, r=2/5=0.4   -> F=0.454545...
    //   char n=2: p=1/1=1.0, r=1/4=0.25  -> F=0.294118...
    //   word n=1: p=1/1=1.0, r=1/2=0.5   -> F=0.555556...
    //   average of 3 = 0.434740 (verified by running this crate).
    //
    // Under the *regressed* (reference-only) gate, word n=2 would still
    // count (the 2-word reference reaches it) and get forced to a hard
    // `F=0.0` (the 1-word candidate has no word-bigrams at all), pulling
    // the 4-way average down to 0.326055 instead — a ~1.09x error this
    // test would have caught.
    let score = chrf_plus_plus("ab", "ab cd").score;
    assert!(
        (score - 0.434_740).abs() < 1e-5,
        "chrf_plus_plus(\"ab\", \"ab cd\") = {score}, expected ~0.434740 \
         (0.326055 would mean the word-order gate regressed to reference-only)"
    );
}

#[test]
fn chrf_both_empty_is_one_and_one_sided_empty_is_zero() {
    assert!((chrf("", "").score - 1.0).abs() < 1e-6);
    assert_eq!(chrf("", "x").score, 0.0);
    assert_eq!(chrf("x", "").score, 0.0);
}

#[test]
fn chrf_does_not_panic_on_multibyte_input_shorter_than_default_order() {
    // Regression guard: iterating by `char` (not byte) and the
    // effective-order skip must both hold for multi-byte UTF-8 content
    // shorter than the default order of 6.
    let score = chrf("日本語", "日本語").score;
    assert!((score - 1.0).abs() < 1e-6, "got {score}");
}
