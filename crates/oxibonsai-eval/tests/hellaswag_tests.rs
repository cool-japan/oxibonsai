//! Integration tests for the HellaSwag sentence-completion evaluator.

use std::io::Write;

use oxibonsai_eval::{HellaSwagDataset, HellaSwagEvaluator, HellaSwagItem};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

fn make_item(id: &str, ctx: &str, endings: Vec<&str>, label: usize) -> HellaSwagItem {
    HellaSwagItem {
        id: id.to_string(),
        activity_label: "test".to_string(),
        ctx: ctx.to_string(),
        endings: endings.iter().map(|s| s.to_string()).collect(),
        label,
    }
}

/// Build a small canonical dataset with four items.
///
/// Labels cycle through 0, 1, 2, 3 so we can construct "all-correct" vectors
/// by mapping label index → answer letter.
fn four_item_dataset() -> HellaSwagDataset {
    HellaSwagDataset::from_items(vec![
        make_item(
            "0",
            "The cat sat on the mat.",
            vec!["A1", "B1", "C1", "D1"],
            0,
        ),
        make_item("1", "She opened the door.", vec!["A2", "B2", "C2", "D2"], 1),
        make_item(
            "2",
            "He picked up the phone.",
            vec!["A3", "B3", "C3", "D3"],
            2,
        ),
        make_item(
            "3",
            "They walked to school.",
            vec!["A4", "B4", "C4", "D4"],
            3,
        ),
    ])
}

/// Map a 0-based label to the expected answer-letter string.
fn letter(idx: usize) -> String {
    match idx {
        0 => "A".to_string(),
        1 => "B".to_string(),
        2 => "C".to_string(),
        3 => "D".to_string(),
        _ => panic!("unsupported label index {idx}"),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// 1. Empty dataset returns zero accuracy
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_empty_dataset() {
    let ds = HellaSwagDataset::from_items(vec![]);
    let eval = HellaSwagEvaluator::new();
    let result = eval.evaluate_completions(&ds, &[]);
    assert_eq!(result.total, 0);
    assert_eq!(result.correct, 0);
    assert!((result.accuracy - 0.0).abs() < 1e-6);
    assert!((result.accuracy_pct - 0.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Perfect completions → accuracy 1.0
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_perfect_completions() {
    let ds = four_item_dataset();
    let eval = HellaSwagEvaluator::new();
    // Labels: 0→A, 1→B, 2→C, 3→D
    let completions: Vec<String> = ds.items.iter().map(|it| letter(it.label)).collect();
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.correct, 4);
    assert_eq!(result.total, 4);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. All wrong completions → accuracy 0.0
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_all_wrong_completions() {
    let ds = four_item_dataset();
    let eval = HellaSwagEvaluator::new();
    // Supply the opposite of each correct label.
    let completions: Vec<String> = ds
        .items
        .iter()
        .map(|it| letter((it.label + 1) % 4))
        .collect();
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.correct, 0);
    assert_eq!(result.total, 4);
    assert!(result.accuracy.abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. Partial score: 2 of 4 correct → accuracy 0.5
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_partial_score() {
    let ds = four_item_dataset();
    let eval = HellaSwagEvaluator::new();
    // Items 0 and 1 have labels 0 and 1 (A and B).
    // Supply A, B (correct), then wrong for 2 and 3.
    let completions = vec![
        "A".to_string(), // item 0 label=0 → correct
        "B".to_string(), // item 1 label=1 → correct
        "A".to_string(), // item 2 label=2 → wrong
        "A".to_string(), // item 3 label=3 → wrong
    ];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.correct, 2);
    assert_eq!(result.total, 4);
    assert!((result.accuracy - 0.5).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. Logit argmax picks the correct choice
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_logit_argmax_correct() {
    let ds = HellaSwagDataset::from_items(vec![make_item(
        "0",
        "Context sentence.",
        vec!["e0", "e1", "e2", "e3"],
        2, // correct is index 2
    )]);
    let eval = HellaSwagEvaluator::new();
    // Make index 2 have the highest logit.
    let logits = vec![vec![-1.0_f32, 0.0_f32, 3.0_f32, 1.0_f32]];
    let result = eval.evaluate_logits(&ds, &logits);
    assert_eq!(result.correct, 1);
    assert_eq!(result.total, 1);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 6. Logit argmax picks the wrong choice
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_logit_argmax_wrong() {
    let ds = HellaSwagDataset::from_items(vec![make_item(
        "0",
        "Context sentence.",
        vec!["e0", "e1", "e2", "e3"],
        1, // correct is index 1
    )]);
    let eval = HellaSwagEvaluator::new();
    // Highest logit is at index 3, not 1.
    let logits = vec![vec![0.0_f32, 0.5_f32, 0.1_f32, 5.0_f32]];
    let result = eval.evaluate_logits(&ds, &logits);
    assert_eq!(result.correct, 0);
    assert_eq!(result.total, 1);
    assert!(result.accuracy.abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 7. evaluate_completions with exactly 4 choices works correctly
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_4_choices_enforced() {
    let item = make_item(
        "0",
        "A chef is preparing dinner.",
        vec![
            "He chops vegetables.",
            "He eats breakfast.",
            "He goes to bed.",
            "He reads a book.",
        ],
        0,
    );
    assert_eq!(item.endings.len(), 4);
    let ds = HellaSwagDataset::from_items(vec![item]);
    let eval = HellaSwagEvaluator::new();
    let result = eval.evaluate_completions(&ds, &["A".to_string()]);
    assert_eq!(result.correct, 1);
    assert_eq!(result.total, 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// 8. as_mc_dataset() conversion is correct
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_as_mc_dataset_conversion() {
    let item = make_item("42", "She lifted the box.", vec!["A", "B", "C", "D"], 3);
    let ds = HellaSwagDataset::from_items(vec![item.clone()]);
    let mc = ds.as_mc_dataset();

    assert_eq!(mc.questions.len(), 1);
    let q = &mc.questions[0];
    assert_eq!(q.id, "42");
    assert_eq!(q.question, "She lifted the box.");
    assert_eq!(q.choices, vec!["A", "B", "C", "D"]);
    assert_eq!(q.correct_answer, 3);
    // activity_label is stored as subject
    assert_eq!(q.subject.as_deref(), Some("test"));
}

// ──────────────────────────────────────────────────────────────────────────────
// 9. accuracy_pct == accuracy * 100
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_accuracy_pct() {
    let ds = four_item_dataset();
    let eval = HellaSwagEvaluator::new();
    let completions: Vec<String> = ds.items.iter().map(|it| letter(it.label)).collect();
    let result = eval.evaluate_completions(&ds, &completions);
    assert!((result.accuracy_pct - result.accuracy * 100.0).abs() < 1e-4);
    assert!((result.accuracy_pct - 100.0).abs() < 1e-4);
}

// ──────────────────────────────────────────────────────────────────────────────
// 10. HellaSwagEvaluator::default() is equivalent to new()
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_default_impl() {
    let _ = HellaSwagEvaluator::default();
    // Smoke-test: evaluating with default is valid.
    let ds = HellaSwagDataset::from_items(vec![]);
    let result = HellaSwagEvaluator::default().evaluate_completions(&ds, &[]);
    assert_eq!(result.total, 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// 11. from_items constructor stores items in insertion order
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_dataset_from_items() {
    let item_a = make_item("a", "ctx_a", vec!["e0", "e1", "e2", "e3"], 0);
    let item_b = make_item("b", "ctx_b", vec!["f0", "f1", "f2", "f3"], 2);
    let ds = HellaSwagDataset::from_items(vec![item_a.clone(), item_b.clone()]);
    assert_eq!(ds.items.len(), 2);
    assert_eq!(ds.items[0].id, "a");
    assert_eq!(ds.items[1].id, "b");
    assert_eq!(ds.items[0].label, 0);
    assert_eq!(ds.items[1].label, 2);
}

// ──────────────────────────────────────────────────────────────────────────────
// 12. len() returns the correct count
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_dataset_len() {
    let ds = four_item_dataset();
    assert_eq!(ds.len(), 4);
    assert!(!ds.is_empty());

    let empty = HellaSwagDataset::from_items(vec![]);
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}

// ──────────────────────────────────────────────────────────────────────────────
// 13. from_jsonl round-trip
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hellaswag_from_jsonl() {
    let jsonl = r#"{"ind": "0", "activity_label": "Cooking", "ctx": "She boiled water.", "endings": ["a", "b", "c", "d"], "label": 2}
{"ind": 1, "activity_label": "Cleaning", "ctx": "He swept the floor.", "endings": ["w", "x", "y", "z"], "label": 0}
"#;
    let mut tmp = std::env::temp_dir();
    tmp.push("hellaswag_test_jsonl.jsonl");
    {
        let mut f = std::fs::File::create(&tmp).expect("create tmp");
        f.write_all(jsonl.as_bytes()).expect("write jsonl");
    }

    let ds = HellaSwagDataset::from_jsonl(&tmp).expect("parse");
    assert_eq!(ds.len(), 2);
    assert_eq!(ds.items[0].id, "0");
    assert_eq!(ds.items[0].activity_label, "Cooking");
    assert_eq!(ds.items[0].label, 2);
    assert_eq!(ds.items[1].id, "1");
    assert_eq!(ds.items[1].label, 0);

    std::fs::remove_file(&tmp).ok();
}
