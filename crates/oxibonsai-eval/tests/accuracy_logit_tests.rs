//! Regression tests for `McLogitEvaluator::score` — degenerate (empty / all-`NaN`)
//! per-choice logit slates must never be silently credited as "correct" via a
//! default-index-0 argmax fallback. This evaluator backs ARC / HellaSwag /
//! Winogrande / MMLU logit-based scoring.

use oxibonsai_eval::McLogitEvaluator;

/// An empty slate with `correct_answer == 0` must not be credited as correct.
#[test]
fn score_empty_slate_not_credited_as_correct() {
    let eval = McLogitEvaluator::new();
    let result = eval.score(&[], 0);
    assert_eq!(result.picked, 0);
    assert!(!result.correct, "empty slate must never score as correct");
}

/// A slate that is entirely `NaN`, with `correct_answer == 0`, must not be
/// credited as correct — even though the internal fallback index happens to
/// be 0, `correct` must be forced `false`.
#[test]
fn score_all_nan_slate_not_credited_as_correct() {
    let eval = McLogitEvaluator::new();
    let result = eval.score(&[f32::NAN, f32::NAN, f32::NAN, f32::NAN], 0);
    assert_eq!(result.picked, 0);
    assert!(!result.correct, "all-NaN slate must never score as correct");
}

/// A slate mixing `NaN` and real values still picks the best real value.
#[test]
fn score_partial_nan_slate_picks_best_real_value() {
    let eval = McLogitEvaluator::new();
    let result = eval.score(&[f32::NAN, 1.0, f32::NAN, 9.0], 3);
    assert_eq!(result.picked, 3);
    assert!(result.correct);
}

/// Sanity check: a normal, fully-populated slate still behaves as before.
#[test]
fn score_normal_slate_picks_argmax() {
    let eval = McLogitEvaluator::new();
    let result = eval.score(&[0.1, 0.9, 0.2, 0.05], 1);
    assert_eq!(result.picked, 1);
    assert!(result.correct);
}
