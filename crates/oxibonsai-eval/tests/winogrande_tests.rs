//! Integration tests for the WinoGrande commonsense reasoning evaluator.

use oxibonsai_eval::winogrande::{WinoGrandeDataset, WinoGrandeEvaluator, WinoGrandeItem};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Canonical "trophy" item used across many tests. `answer` is 1-based.
fn trophy_item(answer: u8) -> WinoGrandeItem {
    WinoGrandeItem {
        sentence: "The trophy doesn't fit in the suitcase because the ___ is too large."
            .to_string(),
        option1: "trophy".to_string(),
        option2: "suitcase".to_string(),
        answer,
    }
}

/// Build a dataset from a list of `(answer,)` tuples using trophy items.
fn trophy_dataset(answers: &[u8]) -> WinoGrandeDataset {
    WinoGrandeDataset::from_items(answers.iter().map(|&a| trophy_item(a)).collect())
}

// ──────────────────────────────────────────────────────────────────────────────
// 1. Construction and len
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_construction_and_len() {
    let ds = WinoGrandeDataset::from_items(vec![trophy_item(1), trophy_item(2)]);
    assert_eq!(ds.len(), 2);
    assert!(!ds.is_empty());
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Empty dataset returns zero accuracy
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_empty_dataset_returns_zero_accuracy() {
    let ds = WinoGrandeDataset::from_items(vec![]);
    let eval = WinoGrandeEvaluator::new();
    let result = eval.evaluate_completions(&ds, &[]);
    assert_eq!(result.total, 0);
    assert_eq!(result.correct, 0);
    assert!((result.accuracy - 0.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. evaluate_completions — all correct → 100% accuracy
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_evaluate_completions_perfect() {
    // answer==1 → option1 → index 0 → letter A
    // answer==2 → option2 → index 1 → letter B
    let ds = WinoGrandeDataset::from_items(vec![
        trophy_item(1), // correct = A
        WinoGrandeItem {
            sentence: "S".into(),
            option1: "X".into(),
            option2: "Y".into(),
            answer: 2, // correct = B
        },
    ]);
    let eval = WinoGrandeEvaluator::new();
    let completions = vec!["A".to_string(), "B".to_string()];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.correct, 2);
    assert_eq!(result.total, 2);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. evaluate_completions — all wrong → 0% accuracy
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_evaluate_completions_all_wrong() {
    // answer==1 → correct = A; we supply B → wrong
    // answer==2 → correct = B; we supply A → wrong
    let ds = WinoGrandeDataset::from_items(vec![trophy_item(1), trophy_item(2)]);
    let eval = WinoGrandeEvaluator::new();
    let completions = vec!["B".to_string(), "A".to_string()];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.correct, 0);
    assert_eq!(result.total, 2);
    assert!(result.accuracy.abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. evaluate_completions — partial score (1 of 2 correct)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_evaluate_completions_partial() {
    let ds = trophy_dataset(&[1, 2]);
    let eval = WinoGrandeEvaluator::new();
    // First correct (A matches answer==1), second wrong (A given but B needed)
    let completions = vec!["A".to_string(), "A".to_string()];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.correct, 1);
    assert_eq!(result.total, 2);
    assert!((result.accuracy - 0.5).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 6. evaluate_logits — argmax picks the correct choice
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_evaluate_logits_argmax() {
    // answer==1 → option1 → index 0 → logit_A > logit_B → correct
    let ds = WinoGrandeDataset::from_items(vec![trophy_item(1)]);
    let eval = WinoGrandeEvaluator::new();
    let logits = vec![vec![2.0_f32, 1.0_f32]]; // logit_A > logit_B → predict A
    let result = eval.evaluate_logits(&ds, &logits);
    assert_eq!(result.correct, 1);
    assert_eq!(result.total, 1);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 7. evaluate_logits — argmax picks wrong choice
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_evaluate_logits_argmax_wrong() {
    // answer==1 → correct index 0; logit_B > logit_A → picks index 1 → wrong
    let ds = WinoGrandeDataset::from_items(vec![trophy_item(1)]);
    let eval = WinoGrandeEvaluator::new();
    let logits = vec![vec![0.5_f32, 3.0_f32]]; // logit_B > logit_A → wrong
    let result = eval.evaluate_logits(&ds, &logits);
    assert_eq!(result.correct, 0);
    assert_eq!(result.total, 1);
    assert!(result.accuracy.abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 8. accuracy_pct matches accuracy * 100
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_accuracy_pct_matches_accuracy() {
    let ds = WinoGrandeDataset::from_items(vec![trophy_item(1)]);
    let eval = WinoGrandeEvaluator::new();
    let result = eval.evaluate_completions(&ds, &["A".to_string()]);
    assert!((result.accuracy_pct - result.accuracy * 100.0).abs() < 1e-4);
}

// ──────────────────────────────────────────────────────────────────────────────
// 9. Default impl is equivalent to new()
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_default_impl() {
    let _ = WinoGrandeEvaluator::default();
}

// ──────────────────────────────────────────────────────────────────────────────
// 10. as_mc_dataset produces two choices per item
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_as_mc_dataset_has_two_choices() {
    let ds = WinoGrandeDataset::from_items(vec![trophy_item(1)]);
    let mc = ds.as_mc_dataset();
    assert_eq!(mc.questions.len(), 1);
    assert_eq!(mc.questions[0].choices.len(), 2);
    // answer==1 → option1 → index 0
    assert_eq!(mc.questions[0].correct_answer, 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// 11. as_mc_dataset answer mapping for answer==2
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_as_mc_dataset_answer2_maps_to_index1() {
    let ds = WinoGrandeDataset::from_items(vec![trophy_item(2)]);
    let mc = ds.as_mc_dataset();
    // answer==2 → option2 → index 1
    assert_eq!(mc.questions[0].correct_answer, 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// 12. Larger batch evaluation with mixed answers
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn winogrande_batch_mixed_answers() {
    // 4 items: answers 1, 2, 1, 2 → correct letters A, B, A, B
    let ds = trophy_dataset(&[1, 2, 1, 2]);
    let eval = WinoGrandeEvaluator::new();
    // 3 correct, 1 wrong (last item: supply A but correct is B)
    let completions = vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "A".to_string(),
    ];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.correct, 3);
    assert_eq!(result.total, 4);
    assert!((result.accuracy - 0.75).abs() < 1e-6);
}
