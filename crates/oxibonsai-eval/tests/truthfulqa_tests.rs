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

// ──────────────────────────────────────────────────────────────────────────────
// Regression: empty / NaN logits must never be silently credited via a
// default-index-0 argmax fallback (see the `argmax` internal helper and
// `evaluate_mc1`/`evaluate_mc2`).
// ──────────────────────────────────────────────────────────────────────────────

/// 16. MC1 item with an empty logits vector whose gold answer is index 0 must NOT be credited as correct — it must be counted as skipped instead.
#[test]
fn test_mc1_empty_logits_not_credited_as_correct() {
    // Gold answer is index 0: if `argmax` on an empty slice defaulted to 0,
    // this item would be wrongly scored "correct" with zero evidence.
    let dataset = TruthfulQaDataset::from_items(vec![item_mc1(0)]);
    let eval = TruthfulQaEvaluator::mc1();
    let logits: Vec<Vec<f32>> = vec![Vec::new()];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.total, 1);
    assert_eq!(
        result.correct, 0,
        "empty logits must never count as correct"
    );
    assert_eq!(result.skipped, 1);
    assert!(result.accuracy.abs() < 1e-6);
}

/// 17. MC1 item whose logits are entirely `NaN` and whose gold answer is index 0 must NOT be credited as correct.
#[test]
fn test_mc1_all_nan_logits_not_credited_as_correct() {
    let dataset = TruthfulQaDataset::from_items(vec![item_mc1(0)]);
    let eval = TruthfulQaEvaluator::mc1();
    let logits = vec![vec![f32::NAN, f32::NAN, f32::NAN]];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.total, 1);
    assert_eq!(
        result.correct, 0,
        "all-NaN logits must never count as correct"
    );
    assert_eq!(result.skipped, 1);
    assert!(result.accuracy.abs() < 1e-6);
}

/// 18. MC1 item with a mix of `NaN` and real values still picks the best real (non-NaN) value rather than being skipped or defaulting.
#[test]
fn test_mc1_partial_nan_logits_picks_best_real_value() {
    let dataset = TruthfulQaDataset::from_items(vec![item_mc1(2)]);
    let eval = TruthfulQaEvaluator::mc1();
    // Index 2 has the best real value; index 0/1 are NaN.
    let logits = vec![vec![f32::NAN, f32::NAN, 3.0_f32]];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.total, 1);
    assert_eq!(result.correct, 1);
    assert_eq!(result.skipped, 0);
}

/// 19. MC2 item with entirely `NaN` logits must score 0.0 and be marked skipped, never poisoning the aggregate accuracy with `NaN`.
#[test]
fn test_mc2_all_nan_logits_scores_zero_not_nan() {
    let dataset = TruthfulQaDataset::from_items(vec![item_mc2(vec![0], 3), item_mc1(0)]);
    let eval = TruthfulQaEvaluator::mc2();
    let logits = vec![
        vec![f32::NAN, f32::NAN, f32::NAN],
        vec![5.0_f32, -1.0_f32, -1.0_f32],
    ];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.total, 2);
    assert_eq!(result.skipped, 1);
    assert!(
        !result.accuracy.is_nan(),
        "NaN-poisoned item must not corrupt the aggregate accuracy"
    );
    // Second item has almost all mass on the correct answer → contributes
    // ~1.0; first item contributes 0.0 (skipped) → mean ≈ 0.5.
    assert!(
        (result.accuracy - 0.5).abs() < 0.01,
        "got {}",
        result.accuracy
    );
}

/// 20. MC1 empty per-item logits vector even when the gold answer is a non-zero index must also be counted as skipped, not merely "wrong".
#[test]
fn test_mc1_empty_logits_skipped_regardless_of_gold_index() {
    let dataset = TruthfulQaDataset::from_items(vec![item_mc1(1)]);
    let eval = TruthfulQaEvaluator::mc1();
    let logits: Vec<Vec<f32>> = vec![Vec::new()];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.correct, 0);
    assert_eq!(result.skipped, 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// Regression: a length mismatch that leaves the gold index out of range must
// be routed through `skipped`, matching the documented contract on
// `evaluate_logits` -- not silently scored via a partial/truncated
// evaluation (finding `rag-eval-03`).
// ──────────────────────────────────────────────────────────────────────────────

/// Test 21 (MC1): `mc1_correct_idx` beyond the provided logits' length must
/// be counted as skipped, not silently evaluated against a choice set that
/// cannot possibly contain the correct answer.
#[test]
fn test_mc1_out_of_range_correct_idx_counts_as_skipped() {
    // Gold answer is index 5, but only 3 logits are provided.
    let item = TruthfulQaItem {
        question: "Out of range question?".to_string(),
        mc1_correct_idx: 5,
        mc1_choices: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        mc2_correct_indices: vec![5],
        mc2_choices: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    };
    let dataset = TruthfulQaDataset::from_items(vec![item]);
    let eval = TruthfulQaEvaluator::mc1();
    let logits = vec![vec![5.0_f32, 1.0_f32, 1.0_f32]];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.total, 1);
    assert_eq!(
        result.correct, 0,
        "an out-of-range gold index must never be credited as correct"
    );
    assert_eq!(
        result.skipped, 1,
        "an out-of-range gold index must be counted as skipped, per the \
         documented evaluate_logits contract"
    );
    assert!(result.accuracy.abs() < 1e-6);
}

/// Test 22 (MC2): when *any* correct index is out of range of the provided
/// logits, the whole item must be skipped rather than silently scored from
/// only the in-range subset of correct indices (the original defect: the
/// item would previously score `probs[0]` alone and count as a normal,
/// non-skipped result).
#[test]
fn test_mc2_partial_out_of_range_correct_indices_counts_as_skipped() {
    // Correct indices are [0, 5], but only 3 logits are provided -- index 5
    // is out of range while index 0 is in range.
    let item = TruthfulQaItem {
        question: "Partial out-of-range question?".to_string(),
        mc1_correct_idx: 0,
        mc1_choices: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        mc2_correct_indices: vec![0, 5],
        mc2_choices: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    };
    let dataset = TruthfulQaDataset::from_items(vec![item]);
    let eval = TruthfulQaEvaluator::mc2();
    // If the bug were still present, softmax([100, -100, -100]) would put
    // ~all mass on index 0, and the old code would silently score this item
    // ~1.0 (not skipped) using only the in-range correct index.
    let logits = vec![vec![100.0_f32, -100.0_f32, -100.0_f32]];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.total, 1);
    assert_eq!(
        result.skipped, 1,
        "an item with any out-of-range correct index must be skipped, not \
         partially scored from the in-range subset"
    );
    assert_eq!(result.correct, 0);
    assert!(
        result.accuracy.abs() < 1e-6,
        "a fully-skipped item must contribute 0.0 to the aggregate accuracy, got {}",
        result.accuracy
    );
}

/// Test 23 (MC2): when correct indices are entirely within range, the fix
/// must not change behavior -- normal scoring still applies.
#[test]
fn test_mc2_all_in_range_correct_indices_scored_normally() {
    let item = item_mc2(vec![0], 3);
    let dataset = TruthfulQaDataset::from_items(vec![item]);
    let eval = TruthfulQaEvaluator::mc2();
    let logits = vec![vec![100.0_f32, -100.0_f32, -100.0_f32]];
    let result = eval.evaluate_logits(&dataset, &logits);
    assert_eq!(result.skipped, 0, "in-range items must not be skipped");
    assert!(result.accuracy > 0.99, "got {}", result.accuracy);
}
