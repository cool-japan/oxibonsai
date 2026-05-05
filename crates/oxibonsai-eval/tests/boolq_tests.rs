//! Integration tests for the BoolQ yes/no question answering evaluator.

use oxibonsai_eval::boolq::{BoolQDataset, BoolQEvaluator, BoolQItem};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

fn make_item(passage: &str, question: &str, answer: bool) -> BoolQItem {
    BoolQItem {
        passage: passage.to_string(),
        question: question.to_string(),
        answer,
    }
}

fn sky_yes() -> BoolQItem {
    make_item("The sky is blue during the day.", "Is the sky blue?", true)
}

fn sky_no() -> BoolQItem {
    make_item("The sky turns red at sunset.", "Is the sky always blue?", false)
}

// ──────────────────────────────────────────────────────────────────────────────
// 1. extract_answer — yes variants
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_extract_yes_variants() {
    assert_eq!(BoolQEvaluator::extract_answer("Yes"), Some(true));
    assert_eq!(BoolQEvaluator::extract_answer("YES"), Some(true));
    assert_eq!(BoolQEvaluator::extract_answer("yes"), Some(true));
    assert_eq!(BoolQEvaluator::extract_answer("Yes, that is correct."), Some(true));
    assert_eq!(BoolQEvaluator::extract_answer("yes."), Some(true));
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. extract_answer — no variants
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_extract_no_variants() {
    assert_eq!(BoolQEvaluator::extract_answer("No"), Some(false));
    assert_eq!(BoolQEvaluator::extract_answer("NO"), Some(false));
    assert_eq!(BoolQEvaluator::extract_answer("no"), Some(false));
    assert_eq!(BoolQEvaluator::extract_answer("No, it is not."), Some(false));
    assert_eq!(BoolQEvaluator::extract_answer("no."), Some(false));
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. extract_answer — None for ambiguous or unknown inputs
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_extract_none_for_ambiguous() {
    assert_eq!(BoolQEvaluator::extract_answer("maybe"), None);
    assert_eq!(BoolQEvaluator::extract_answer(""), None);
    assert_eq!(BoolQEvaluator::extract_answer("I don't know"), None);
    assert_eq!(BoolQEvaluator::extract_answer("Perhaps"), None);
    assert_eq!(BoolQEvaluator::extract_answer("uncertain"), None);
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. score — correct yes
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_score_correct_yes() {
    let eval = BoolQEvaluator::new();
    assert!(eval.score("Yes", true));
    assert!(!eval.score("No", true));
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. score — correct no
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_score_correct_no() {
    let eval = BoolQEvaluator::new();
    assert!(eval.score("No", false));
    assert!(!eval.score("Yes", false));
}

// ──────────────────────────────────────────────────────────────────────────────
// 6. build_prompt format
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_build_prompt_format() {
    let eval = BoolQEvaluator::new();
    let item = make_item("The sky is blue.", "Is the sky blue?", true);
    let prompt = eval.build_prompt(&item);
    assert!(prompt.contains("Passage: The sky is blue."));
    assert!(prompt.contains("Question: Is the sky blue?"));
    assert!(prompt.ends_with("Answer:"));
}

// ──────────────────────────────────────────────────────────────────────────────
// 7. evaluate_completions — perfect score
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_evaluate_completions_perfect() {
    let ds = BoolQDataset::from_items(vec![sky_yes(), sky_no()]);
    let eval = BoolQEvaluator::new();
    let completions = vec!["Yes".to_string(), "No".to_string()];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.correct, 2);
    assert_eq!(result.total, 2);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 8. evaluate_completions — counts yes/no predictions separately
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_evaluate_completions_counts_yes_no() {
    let ds = BoolQDataset::from_items(vec![
        make_item("P1", "Q1", true),
        make_item("P2", "Q2", false),
    ]);
    let eval = BoolQEvaluator::new();
    let completions = vec!["Yes".to_string(), "No".to_string()];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.yes_predicted, 1);
    assert_eq!(result.no_predicted, 1);
    assert_eq!(result.correct, 2);
}

// ──────────────────────────────────────────────────────────────────────────────
// 9. evaluate_completions — all wrong
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_evaluate_completions_all_wrong() {
    let ds = BoolQDataset::from_items(vec![sky_yes(), sky_no()]);
    let eval = BoolQEvaluator::new();
    // Swap the correct answers
    let completions = vec!["No".to_string(), "Yes".to_string()];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.correct, 0);
    assert_eq!(result.total, 2);
    assert!(result.accuracy.abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 10. evaluate_logits — logit-based yes prediction
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_evaluate_logits_yes() {
    let ds = BoolQDataset::from_items(vec![make_item("P", "Q", true)]);
    let eval = BoolQEvaluator::new();
    // logit_yes=3.0 > logit_no=1.0 → predict yes → correct (gold=true)
    let result = eval.evaluate_logits(&ds, &[[3.0_f32, 1.0_f32]]);
    assert_eq!(result.correct, 1);
    assert_eq!(result.yes_predicted, 1);
    assert_eq!(result.no_predicted, 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// 11. evaluate_logits — logit-based no prediction
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_evaluate_logits_no() {
    let ds = BoolQDataset::from_items(vec![make_item("P", "Q", false)]);
    let eval = BoolQEvaluator::new();
    // logit_yes=0.5 < logit_no=2.0 → predict no → correct (gold=false)
    let result = eval.evaluate_logits(&ds, &[[0.5_f32, 2.0_f32]]);
    assert_eq!(result.correct, 1);
    assert_eq!(result.yes_predicted, 0);
    assert_eq!(result.no_predicted, 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// 12. Empty dataset
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_empty_dataset() {
    let ds = BoolQDataset::from_items(vec![]);
    let result = BoolQEvaluator::new().evaluate_completions(&ds, &[]);
    assert_eq!(result.total, 0);
    assert_eq!(result.correct, 0);
    assert!((result.accuracy - 0.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 13. Default impl
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_default_impl() {
    let _ = BoolQEvaluator::default();
}

// ──────────────────────────────────────────────────────────────────────────────
// 14. Short string no-panic
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_short_string_no_panic() {
    assert_eq!(BoolQEvaluator::extract_answer("y"), None);
    assert_eq!(BoolQEvaluator::extract_answer("n"), None);
    assert_eq!(BoolQEvaluator::extract_answer("a"), None);
}

// ──────────────────────────────────────────────────────────────────────────────
// 15. Leading whitespace extraction
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_leading_whitespace() {
    assert_eq!(BoolQEvaluator::extract_answer("  Yes"), Some(true));
    assert_eq!(BoolQEvaluator::extract_answer("\t\nNo"), Some(false));
    assert_eq!(BoolQEvaluator::extract_answer("   YES, definitely"), Some(true));
}

// ──────────────────────────────────────────────────────────────────────────────
// 16. accuracy_pct matches accuracy * 100
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn boolq_accuracy_pct_matches_accuracy() {
    let ds = BoolQDataset::from_items(vec![sky_yes()]);
    let eval = BoolQEvaluator::new();
    let result = eval.evaluate_completions(&ds, &["Yes".to_string()]);
    assert!((result.accuracy_pct - result.accuracy * 100.0).abs() < 1e-4);
}
