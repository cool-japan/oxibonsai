//! Integration tests for the GSM8K evaluator.

use oxibonsai_eval::dataset::EvalDataset;
use oxibonsai_eval::gsm8k::{gsm8k_example, Gsm8kEvaluator};

const TOL: f64 = 1e-9;

// ──────────────────────────────────────────────────────────────────────────────
// 1. extract_final_answer — simple integer
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_final_answer_simple_integer() {
    let result = Gsm8kEvaluator::extract_final_answer("The answer is #### 42");
    assert!(result.is_some(), "expected Some, got None");
    let val = result.unwrap();
    assert!((val - 42.0).abs() < TOL, "got {}", val);
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. extract_final_answer — negative integer
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_final_answer_negative() {
    let result = Gsm8kEvaluator::extract_final_answer("Result: #### -5");
    assert!(result.is_some(), "expected Some, got None");
    let val = result.unwrap();
    assert!((val - (-5.0)).abs() < TOL, "got {}", val);
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. extract_final_answer — decimal value
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_final_answer_decimal() {
    let result = Gsm8kEvaluator::extract_final_answer("So the final cost is #### 2.50");
    assert!(result.is_some(), "expected Some, got None");
    let val = result.unwrap();
    assert!((val - 2.50_f64).abs() < 1e-10, "got {val}");
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. extract_final_answer — returns None when marker is absent
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_final_answer_none_when_missing() {
    assert!(
        Gsm8kEvaluator::extract_final_answer("no marker here").is_none(),
        "expected None for text without ####"
    );
    assert!(
        Gsm8kEvaluator::extract_final_answer("").is_none(),
        "expected None for empty string"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. extract_final_answer — picks the LAST marker
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_final_answer_picks_last_marker() {
    // Two "####" lines; the last one (42) should win.
    let text = "Intermediate step: #### 10\nFinal check: #### 42";
    let result = Gsm8kEvaluator::extract_final_answer(text);
    assert!(result.is_some(), "expected Some, got None");
    let val = result.unwrap();
    assert!((val - 42.0).abs() < TOL, "expected 42.0, got {}", val);
}

// ──────────────────────────────────────────────────────────────────────────────
// 6. extract_final_answer — comma-separated thousands
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_final_answer_with_commas() {
    let result = Gsm8kEvaluator::extract_final_answer("Total cost: #### 1,234");
    assert!(result.is_some(), "expected Some, got None");
    let val = result.unwrap();
    assert!((val - 1234.0).abs() < TOL, "got {}", val);
}

// ──────────────────────────────────────────────────────────────────────────────
// 7. score — integer completion against integer gold
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn score_integer_vs_integer() {
    let ev = Gsm8kEvaluator::new();
    assert!(
        ev.score("some reasoning #### 42", "#### 42"),
        "identical integer answers should score true"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// 8. score — float completion matches integer gold (within tolerance)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn score_float_vs_integer() {
    let ev = Gsm8kEvaluator::new();
    // 42.0 and 42 should compare as equal within tolerance.
    assert!(
        ev.score("#### 42.0", "#### 42"),
        "42.0 vs 42 should be within tolerance"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// 9. score — wrong answer returns false
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn score_wrong_answer() {
    let ev = Gsm8kEvaluator::new();
    assert!(!ev.score("#### 41", "#### 42"), "41 vs 42 should not match");
}

// ──────────────────────────────────────────────────────────────────────────────
// 10. score — completion with no marker returns false
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn score_no_marker_in_completion() {
    let ev = Gsm8kEvaluator::new();
    assert!(
        !ev.score("I think the answer is forty-two", "#### 42"),
        "completion without #### should score false"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// 11. evaluate_dataset — all completions correct
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn evaluate_dataset_all_correct() {
    let mut dataset = EvalDataset::new("gsm8k-test");
    dataset.add(gsm8k_example("g0", "Alice has 3 apples...", "#### 3"));
    dataset.add(gsm8k_example("g1", "Bob earns $5...", "#### 5"));
    dataset.add(gsm8k_example("g2", "Total widgets...", "#### 100"));

    let completions = vec![
        "Let me think... #### 3".to_string(),
        "Step 1... #### 5".to_string(),
        "The total is #### 100".to_string(),
    ];

    let ev = Gsm8kEvaluator::new();
    let result = ev.evaluate_dataset(&dataset, &completions);

    assert_eq!(result.correct, 3);
    assert_eq!(result.total, 3);
    assert_eq!(result.no_answer_extracted, 0);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 12. evaluate_dataset — partial correctness, some no-answer-extracted
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn evaluate_dataset_partial() {
    let mut dataset = EvalDataset::new("gsm8k-partial");
    dataset.add(gsm8k_example("g0", "Q1", "#### 10")); // completion correct
    dataset.add(gsm8k_example("g1", "Q2", "#### 20")); // completion wrong
    dataset.add(gsm8k_example("g2", "Q3", "#### 30")); // completion has no marker
    dataset.add(gsm8k_example("g3", "Q4", "#### 40")); // completion correct

    let completions = vec![
        "#### 10".to_string(),                 // correct
        "#### 99".to_string(),                 // wrong
        "I don't know the answer".to_string(), // no marker
        "Step by step #### 40".to_string(),    // correct
    ];

    let ev = Gsm8kEvaluator::new();
    let result = ev.evaluate_dataset(&dataset, &completions);

    assert_eq!(result.total, 4);
    assert_eq!(result.correct, 2);
    assert_eq!(result.no_answer_extracted, 1);
    assert!((result.accuracy - 0.5).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 13. evaluate_dataset — empty dataset returns zero result
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn evaluate_dataset_empty() {
    let dataset = EvalDataset::new("empty");
    let completions: Vec<String> = vec![];

    let ev = Gsm8kEvaluator::new();
    let result = ev.evaluate_dataset(&dataset, &completions);

    assert_eq!(result.correct, 0);
    assert_eq!(result.total, 0);
    assert_eq!(result.no_answer_extracted, 0);
    assert!(result.accuracy.abs() < 1e-6);
    assert!(result.no_answer_rate().abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 14. evaluate_dataset — skips examples without expected_output
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn evaluate_dataset_skips_none_expected_output() {
    use oxibonsai_eval::dataset::EvalExample;

    let mut dataset = EvalDataset::new("mixed");
    // One example with gold answer.
    dataset.add(gsm8k_example("g0", "Q1", "#### 7"));
    // One example without gold answer (e.g. test split).
    dataset.add(EvalExample {
        id: "g1".to_string(),
        input: "Q2".to_string(),
        expected_output: None,
        metadata: std::collections::HashMap::new(),
    });

    let completions = vec!["#### 7".to_string(), "#### 42".to_string()];

    let ev = Gsm8kEvaluator::new();
    let result = ev.evaluate_dataset(&dataset, &completions);

    // Only g0 is evaluated; g1 (no expected_output) is skipped.
    assert_eq!(result.total, 1);
    assert_eq!(result.correct, 1);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 15. accuracy_pct and no_answer_rate helpers
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn gsm8k_result_helper_methods() {
    let mut dataset = EvalDataset::new("helpers");
    dataset.add(gsm8k_example("g0", "Q1", "#### 1"));
    dataset.add(gsm8k_example("g1", "Q2", "#### 2"));
    dataset.add(gsm8k_example("g2", "Q3", "#### 3"));
    dataset.add(gsm8k_example("g3", "Q4", "#### 4"));

    let completions = vec![
        "#### 1".to_string(),         // correct
        "no answer here".to_string(), // no marker
        "no answer here".to_string(), // no marker
        "#### 4".to_string(),         // correct
    ];

    let ev = Gsm8kEvaluator::new();
    let result = ev.evaluate_dataset(&dataset, &completions);

    assert_eq!(result.total, 4);
    assert_eq!(result.correct, 2);
    assert_eq!(result.no_answer_extracted, 2);

    let pct = result.accuracy_pct();
    assert!((pct - 50.0).abs() < 1e-4, "accuracy_pct = {}", pct);

    let nar = result.no_answer_rate();
    assert!((nar - 0.5).abs() < 1e-6, "no_answer_rate = {}", nar);
}
