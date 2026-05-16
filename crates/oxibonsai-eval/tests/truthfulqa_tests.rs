//! Integration tests for the TruthfulQA evaluator (MC1 and MC2 modes).

use oxibonsai_eval::{TruthfulQaDataset, TruthfulQaEvaluator, TruthfulQaItem, TruthfulQaMode};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Build a minimal TruthfulQaItem with the given MC1 correct index.
///
/// `mc1_choices` is `["correct", "wrong_a", "wrong_b"]`;
/// `mc1_correct_idx` = `mc1_idx` (0-based).
///
/// For MC2 we reuse the same choice list and set `mc2_correct_indices = [mc1_idx]`.
fn item_mc1(mc1_idx: usize) -> TruthfulQaItem {
    TruthfulQaItem {
        question: "Test question?".to_string(),
        mc1_correct_idx: mc1_idx,
        mc1_choices: vec![
            "answer_0".to_string(),
            "answer_1".to_string(),
            "answer_2".to_string(),
        ],
        mc2_correct_indices: vec![mc1_idx],
        mc2_choices: vec![
            "answer_0".to_string(),
            "answer_1".to_string(),
            "answer_2".to_string(),
        ],
    }
}

/// Build an item where `mc2_correct_indices` is explicitly specified.
fn item_mc2(mc2_correct: Vec<usize>, n_choices: usize) -> TruthfulQaItem {
    let choices: Vec<String> = (0..n_choices).map(|i| format!("choice_{i}")).collect();
    TruthfulQaItem {
        question: "MC2 question?".to_string(),
        mc1_correct_idx: 0,
        mc1_choices: choices.clone(),
        mc2_correct_indices: mc2_correct,
        mc2_choices: choices,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MC1 tests
// ──────────────────────────────────────────────────────────────────────────────

/// 1. All argmax selections correct → MC1 accuracy = 1.0
#[test]
fn test_mc1_perfect_score() {
    let dataset = TruthfulQaDataset::from_items(vec![
        item_mc1(0), // correct is index 0
        item_mc1(1), // correct is index 1
        item_mc1(2), // correct is index 2
    ]);
    let eval = TruthfulQaEvaluator::mc1();
    // Give each item its own correct index the highest logit.
    let logits = vec![
        vec![5.0_f32, 1.0_f32, 1.0_f32],  // argmax = 0 → correct
        vec![0.0_f32, 9.0_f32, 0.0_f32],  // argmax = 1 → correct
        vec![-1.0_f32, 0.0_f32, 7.0_f32], // argmax = 2 → correct
    ];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.mode, TruthfulQaMode::Mc1);
    assert_eq!(result.correct, 3);
    assert_eq!(result.total, 3);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

/// 2. All argmax selections wrong → MC1 accuracy = 0.0
#[test]
fn test_mc1_all_wrong() {
    let dataset = TruthfulQaDataset::from_items(vec![item_mc1(0), item_mc1(0)]);
    let eval = TruthfulQaEvaluator::mc1();
    // Give index 1 the highest logit; correct is 0 → wrong every time.
    let logits = vec![
        vec![0.0_f32, 5.0_f32, 0.0_f32], // argmax = 1 ≠ 0
        vec![0.0_f32, 5.0_f32, 0.0_f32], // argmax = 1 ≠ 0
    ];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.correct, 0);
    assert_eq!(result.total, 2);
    assert!(result.accuracy.abs() < 1e-6);
}

/// 3. Half correct → MC1 accuracy = 0.5
#[test]
fn test_mc1_partial() {
    let dataset = TruthfulQaDataset::from_items(vec![
        item_mc1(0), // correct index = 0
        item_mc1(0), // correct index = 0
    ]);
    let eval = TruthfulQaEvaluator::mc1();
    let logits = vec![
        vec![5.0_f32, 1.0_f32, 1.0_f32], // argmax = 0 → correct
        vec![0.0_f32, 8.0_f32, 0.0_f32], // argmax = 1 → wrong
    ];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.correct, 1);
    assert_eq!(result.total, 2);
    assert!((result.accuracy - 0.5).abs() < 1e-6);
}

/// 4. result.mode is Mc1 for mc1 evaluator
#[test]
fn test_mc1_mode_is_mc1() {
    let dataset = TruthfulQaDataset::from_items(vec![item_mc1(0)]);
    let eval = TruthfulQaEvaluator::mc1();
    let result = eval.evaluate_logits(&dataset, &[vec![1.0_f32, 0.0_f32, 0.0_f32]]);
    assert_eq!(result.mode, TruthfulQaMode::Mc1);
}

// ──────────────────────────────────────────────────────────────────────────────
// MC2 tests
// ──────────────────────────────────────────────────────────────────────────────

/// 5. All logit mass on correct answers → MC2 score ≈ 1.0
#[test]
fn test_mc2_all_correct_mass_on_correct() {
    // correct indices: [0, 1]; incorrect: [2]
    let item = item_mc2(vec![0, 1], 3);
    let dataset = TruthfulQaDataset::from_items(vec![item]);
    let eval = TruthfulQaEvaluator::mc2();
    // Very large logits for 0 and 1, very small for 2 → essentially all mass on correct.
    let logits = vec![vec![100.0_f32, 100.0_f32, -100.0_f32]];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.mode, TruthfulQaMode::Mc2);
    assert!(result.accuracy > 0.99, "accuracy={}", result.accuracy);
}

/// 6. All logit mass on wrong answers → MC2 score ≈ 0.0
#[test]
fn test_mc2_all_mass_on_wrong() {
    // correct index: [0]; incorrect: [1, 2]
    let item = item_mc2(vec![0], 3);
    let dataset = TruthfulQaDataset::from_items(vec![item]);
    let eval = TruthfulQaEvaluator::mc2();
    // Very large logits for 1 and 2 → essentially no mass on index 0 (correct).
    let logits = vec![vec![-100.0_f32, 100.0_f32, 100.0_f32]];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert!(result.accuracy < 0.01, "accuracy={}", result.accuracy);
}

/// 7. Mixed correct/incorrect mass → MC2 score in (0, 1)
#[test]
fn test_mc2_mixed_correct_incorrect() {
    // correct index: [0]; incorrect: [1]
    let item = item_mc2(vec![0], 2);
    let dataset = TruthfulQaDataset::from_items(vec![item]);
    let eval = TruthfulQaEvaluator::mc2();
    // Equal logits → softmax = [0.5, 0.5] → score = 0.5
    let logits = vec![vec![0.0_f32, 0.0_f32]];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert!(result.accuracy > 0.0);
    assert!(result.accuracy < 1.0);
    assert!(
        (result.accuracy - 0.5).abs() < 1e-5,
        "expected ≈0.5, got {}",
        result.accuracy
    );
}

/// 8. MC2 accuracy is a continuous value, not forced to 0 or 1
#[test]
fn test_mc2_continuous_score_not_just_01() {
    // Two items: first gets ≈0.7 score, second gets ≈0.3 score → mean ≈ 0.5
    let item1 = item_mc2(vec![0], 2); // 1 correct, 1 incorrect
    let item2 = item_mc2(vec![0], 2);
    let dataset = TruthfulQaDataset::from_items(vec![item1, item2]);
    let eval = TruthfulQaEvaluator::mc2();
    let logits = vec![
        vec![1.0_f32, -1.0_f32], // softmax ≈ [0.88, 0.12] → score ≈ 0.88
        vec![-1.0_f32, 1.0_f32], // softmax ≈ [0.12, 0.88] → score ≈ 0.12
    ];
    let result = eval.evaluate_logits(&dataset, &logits);
    // Mean ≈ (0.88 + 0.12) / 2 = 0.5
    assert!(
        (result.accuracy - 0.5).abs() < 0.01,
        "got {}",
        result.accuracy
    );
    // The accuracy itself should not be exactly 0 or 1.
    assert!(result.accuracy > 0.0);
    assert!(result.accuracy < 1.0);
}

/// 9. result.mode is Mc2 for mc2 evaluator
#[test]
fn test_mc2_mode_is_mc2() {
    let dataset = TruthfulQaDataset::from_items(vec![item_mc2(vec![0], 2)]);
    let eval = TruthfulQaEvaluator::mc2();
    let result = eval.evaluate_logits(&dataset, &[vec![1.0_f32, 0.0_f32]]);
    assert_eq!(result.mode, TruthfulQaMode::Mc2);
}

// ──────────────────────────────────────────────────────────────────────────────
// Empty-dataset guard tests
// ──────────────────────────────────────────────────────────────────────────────

/// 10. Empty dataset with MC1 mode → 0/0
#[test]
fn test_truthfulqa_empty_mc1() {
    let dataset = TruthfulQaDataset::from_items(vec![]);
    let eval = TruthfulQaEvaluator::mc1();
    let result = eval.evaluate_logits(&dataset, &[]);
    assert_eq!(result.total, 0);
    assert_eq!(result.correct, 0);
    assert!(result.accuracy.abs() < 1e-6);
}

/// 11. Empty dataset with MC2 mode → 0/0
#[test]
fn test_truthfulqa_empty_mc2() {
    let dataset = TruthfulQaDataset::from_items(vec![]);
    let eval = TruthfulQaEvaluator::mc2();
    let result = eval.evaluate_logits(&dataset, &[]);
    assert_eq!(result.total, 0);
    assert_eq!(result.correct, 0);
    assert!(result.accuracy.abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// Construction tests
// ──────────────────────────────────────────────────────────────────────────────

/// 12. from_items stores items in insertion order
#[test]
fn test_truthfulqa_dataset_from_items() {
    let items = vec![item_mc1(0), item_mc1(1), item_mc1(2)];
    let ds = TruthfulQaDataset::from_items(items);
    assert_eq!(ds.len(), 3);
    assert_eq!(ds.items[0].mc1_correct_idx, 0);
    assert_eq!(ds.items[1].mc1_correct_idx, 1);
    assert_eq!(ds.items[2].mc1_correct_idx, 2);
}

// ──────────────────────────────────────────────────────────────────────────────
// Softmax normalization test
// ──────────────────────────────────────────────────────────────────────────────

/// 13. MC2 softmax over all choices sums to 1.0 within tolerance
///
/// We verify indirectly: for an item where all choices are correct, the
/// MC2 score must equal 1.0 (all probability mass is on correct answers).
#[test]
fn test_mc2_softmax_normalization() {
    // Mark all 4 choices as correct → score = sum_correct / sum_all = 1.0
    let item = item_mc2(vec![0, 1, 2, 3], 4);
    let dataset = TruthfulQaDataset::from_items(vec![item]);
    let eval = TruthfulQaEvaluator::mc2();
    let logits = vec![vec![1.0_f32, -1.0_f32, 2.0_f32, 0.0_f32]];
    let result = eval.evaluate_logits(&dataset, &logits);
    // All choices correct → score must be 1.0.
    assert!(
        (result.accuracy - 1.0).abs() < 1e-5,
        "expected 1.0, got {}",
        result.accuracy
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Edge-case: single-item datasets
// ──────────────────────────────────────────────────────────────────────────────

/// 14. MC1 single item, 2 choices
#[test]
fn test_mc1_single_item() {
    let item = TruthfulQaItem {
        question: "Is the sky blue?".to_string(),
        mc1_correct_idx: 0,
        mc1_choices: vec!["Yes".to_string(), "No".to_string()],
        mc2_correct_indices: vec![0],
        mc2_choices: vec!["Yes".to_string(), "No".to_string()],
    };
    let dataset = TruthfulQaDataset::from_items(vec![item]);
    let eval = TruthfulQaEvaluator::mc1();
    // Logit for "Yes" higher → picks index 0 → correct.
    let result = eval.evaluate_logits(&dataset, &[vec![3.0_f32, -1.0_f32]]);
    assert_eq!(result.correct, 1);
    assert_eq!(result.total, 1);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

/// 15. MC2 single correct answer only → score = prob of that answer
#[test]
fn test_mc2_single_correct_only() {
    // Only index 1 is correct among 3 choices.
    let item = item_mc2(vec![1], 3);
    let dataset = TruthfulQaDataset::from_items(vec![item]);
    let eval = TruthfulQaEvaluator::mc2();
    // Make index 1 dominate → score ≈ 1.0.
    let logits = vec![vec![-10.0_f32, 10.0_f32, -10.0_f32]];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert!(result.accuracy > 0.99, "got {}", result.accuracy);
    assert_eq!(result.mode, TruthfulQaMode::Mc2);
}
