use oxibonsai_eval::dataset::{McDataset, MultipleChoiceQuestion};
use oxibonsai_eval::mmlu::{MmluEvaluator, MmluResult};

// ── Helpers ────────────────────────────────────────────────────────────────────

fn make_q(id: &str, subject: Option<&str>, correct: usize) -> MultipleChoiceQuestion {
    MultipleChoiceQuestion {
        id: id.to_string(),
        question: format!("Question {id}"),
        choices: vec![
            "A) choice 0".to_string(),
            "B) choice 1".to_string(),
            "C) choice 2".to_string(),
            "D) choice 3".to_string(),
        ],
        correct_answer: correct,
        subject: subject.map(|s| s.to_string()),
        difficulty: None,
    }
}

fn dataset_with_subjects() -> McDataset {
    let mut ds = McDataset::new("mmlu_test");
    // subject: "math" (3 questions)
    ds.add(make_q("math/0", Some("math"), 0)); // A correct
    ds.add(make_q("math/1", Some("math"), 1)); // B correct
    ds.add(make_q("math/2", Some("math"), 2)); // C correct
                                               // subject: "history" (2 questions)
    ds.add(make_q("history/0", Some("history"), 0)); // A correct
    ds.add(make_q("history/1", Some("history"), 3)); // D correct
    ds
}

// ── Basic construction and empty dataset ──────────────────────────────────────

#[test]
fn evaluator_new_default_are_equivalent() {
    let a = MmluEvaluator::new();
    let b = MmluEvaluator::default();
    let ds = McDataset::new("mmlu");
    let ra = a.evaluate_logits(&ds, &[]);
    let rb = b.evaluate_logits(&ds, &[]);
    assert_eq!(ra.total, 0);
    assert_eq!(rb.total, 0);
}

#[test]
fn empty_dataset_completions_zero_accuracy() {
    let eval = MmluEvaluator::new();
    let ds = McDataset::new("mmlu");
    let result: MmluResult = eval.evaluate_completions(&ds, &[]);
    assert_eq!(result.total, 0);
    assert_eq!(result.correct, 0);
    assert!((result.accuracy - 0.0).abs() < 1e-6);
    assert!(result.by_subject.is_empty());
}

#[test]
fn empty_dataset_logits_zero_accuracy() {
    let eval = MmluEvaluator::new();
    let ds = McDataset::new("mmlu");
    let result = eval.evaluate_logits(&ds, &[]);
    assert_eq!(result.total, 0);
    assert_eq!(result.correct, 0);
    assert!((result.accuracy - 0.0).abs() < 1e-6);
}

// ── Completion-based evaluation ────────────────────────────────────────────────

#[test]
fn completions_all_correct_100pct() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    // correct_answer indices: 0, 1, 2, 0, 3 → letters: A, B, C, A, D
    let completions = vec![
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "A".to_string(),
        "D".to_string(),
    ];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.total, 5);
    assert_eq!(result.correct, 5);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
    assert!((result.accuracy_pct - 100.0).abs() < 1e-4);
}

#[test]
fn completions_all_wrong_zero_accuracy() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    let completions = vec![
        "D".to_string(), // wrong (correct = A)
        "A".to_string(), // wrong (correct = B)
        "D".to_string(), // wrong (correct = C)
        "D".to_string(), // wrong (correct = A)
        "A".to_string(), // wrong (correct = D)
    ];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.total, 5);
    assert_eq!(result.correct, 0);
    assert!((result.accuracy - 0.0).abs() < 1e-6);
}

#[test]
fn completions_partial_accuracy() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    // correct 3 out of 5 (math all correct, history all wrong)
    let completions = vec![
        "A".to_string(), // math/0 ✓
        "B".to_string(), // math/1 ✓
        "C".to_string(), // math/2 ✓
        "B".to_string(), // history/0 ✗
        "A".to_string(), // history/1 ✗
    ];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.total, 5);
    assert_eq!(result.correct, 3);
    let expected = 3.0 / 5.0;
    assert!((result.accuracy - expected).abs() < 1e-6);
}

// ── Logit-based evaluation ─────────────────────────────────────────────────────

#[test]
fn logits_all_correct_100pct() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    // correct: 0, 1, 2, 0, 3
    let per_choice_logits = vec![
        vec![1.0f32, 0.0, 0.0, 0.0], // argmax=0 ✓
        vec![0.0, 1.0, 0.0, 0.0],    // argmax=1 ✓
        vec![0.0, 0.0, 1.0, 0.0],    // argmax=2 ✓
        vec![1.0, 0.0, 0.0, 0.0],    // argmax=0 ✓
        vec![0.0, 0.0, 0.0, 1.0],    // argmax=3 ✓
    ];
    let result = eval.evaluate_logits(&ds, &per_choice_logits);
    assert_eq!(result.total, 5);
    assert_eq!(result.correct, 5);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

#[test]
fn logits_all_wrong_zero_accuracy() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    let per_choice_logits = vec![
        vec![0.0f32, 0.0, 0.0, 1.0], // argmax=3 ✗ (correct=0)
        vec![1.0, 0.0, 0.0, 0.0],    // argmax=0 ✗ (correct=1)
        vec![1.0, 0.0, 0.0, 0.0],    // argmax=0 ✗ (correct=2)
        vec![0.0, 1.0, 0.0, 0.0],    // argmax=1 ✗ (correct=0)
        vec![1.0, 0.0, 0.0, 0.0],    // argmax=0 ✗ (correct=3)
    ];
    let result = eval.evaluate_logits(&ds, &per_choice_logits);
    assert_eq!(result.total, 5);
    assert_eq!(result.correct, 0);
    assert!((result.accuracy - 0.0).abs() < 1e-6);
}

#[test]
fn logits_partial_accuracy() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    // 2 out of 5 correct
    let per_choice_logits = vec![
        vec![1.0f32, 0.0, 0.0, 0.0], // ✓ (correct=0)
        vec![0.0, 1.0, 0.0, 0.0],    // ✓ (correct=1)
        vec![1.0, 0.0, 0.0, 0.0],    // ✗ (correct=2)
        vec![0.0, 1.0, 0.0, 0.0],    // ✗ (correct=0)
        vec![1.0, 0.0, 0.0, 0.0],    // ✗ (correct=3)
    ];
    let result = eval.evaluate_logits(&ds, &per_choice_logits);
    assert_eq!(result.total, 5);
    assert_eq!(result.correct, 2);
    let expected = 2.0 / 5.0;
    assert!((result.accuracy - expected).abs() < 1e-6);
}

#[test]
fn accuracy_pct_matches_accuracy_times_100() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    let per_choice_logits = vec![
        vec![1.0f32, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let result = eval.evaluate_logits(&ds, &per_choice_logits);
    assert!((result.accuracy_pct - result.accuracy * 100.0).abs() < 1e-4);
}

// ── Per-subject breakdown ──────────────────────────────────────────────────────

#[test]
fn by_subject_completions_correct_subjects() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    let completions = vec![
        "A".to_string(), // math/0 ✓
        "B".to_string(), // math/1 ✓
        "C".to_string(), // math/2 ✓
        "A".to_string(), // history/0 ✓
        "D".to_string(), // history/1 ✓
    ];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.by_subject.len(), 2);

    let math = result.by_subject.get("math").expect("math subject missing");
    assert_eq!(math.correct, 3);
    assert_eq!(math.total, 3);
    assert!((math.accuracy - 1.0).abs() < 1e-6);

    let history = result
        .by_subject
        .get("history")
        .expect("history subject missing");
    assert_eq!(history.correct, 2);
    assert_eq!(history.total, 2);
    assert!((history.accuracy - 1.0).abs() < 1e-6);
}

#[test]
fn by_subject_logits_correct_subjects() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    let per_choice_logits = vec![
        vec![1.0f32, 0.0, 0.0, 0.0], // math/0 ✓
        vec![0.0, 1.0, 0.0, 0.0],    // math/1 ✓
        vec![1.0, 0.0, 0.0, 0.0],    // math/2 ✗
        vec![1.0, 0.0, 0.0, 0.0],    // history/0 ✓
        vec![0.0, 0.0, 0.0, 1.0],    // history/1 ✓
    ];
    let result = eval.evaluate_logits(&ds, &per_choice_logits);
    let math = result.by_subject.get("math").expect("math subject");
    assert_eq!(math.correct, 2);
    assert_eq!(math.total, 3);
    let history = result.by_subject.get("history").expect("history subject");
    assert_eq!(history.correct, 2);
    assert_eq!(history.total, 2);
}

#[test]
fn subject_extracted_from_id_when_field_absent() {
    let mut ds = McDataset::new("mmlu");
    // subject field is None — should extract from id "physics/0" → "physics"
    ds.add(MultipleChoiceQuestion {
        id: "physics/0".to_string(),
        question: "Which is correct?".to_string(),
        choices: vec![
            "A) a".to_string(),
            "B) b".to_string(),
            "C) c".to_string(),
            "D) d".to_string(),
        ],
        correct_answer: 0,
        subject: None,
        difficulty: None,
    });
    ds.add(MultipleChoiceQuestion {
        id: "physics/1".to_string(),
        question: "Another question?".to_string(),
        choices: vec![
            "A) a".to_string(),
            "B) b".to_string(),
            "C) c".to_string(),
            "D) d".to_string(),
        ],
        correct_answer: 1,
        subject: None,
        difficulty: None,
    });
    let eval = MmluEvaluator::new();
    let completions = vec!["A".to_string(), "B".to_string()];
    let result = eval.evaluate_completions(&ds, &completions);
    assert!(
        result.by_subject.contains_key("physics"),
        "expected 'physics' subject"
    );
    let phys = result.by_subject.get("physics").unwrap();
    assert_eq!(phys.total, 2);
    assert_eq!(phys.correct, 2);
}

#[test]
fn subject_extracted_from_id_no_slash() {
    let mut ds = McDataset::new("mmlu");
    // id has no slash — use entire id as subject
    ds.add(MultipleChoiceQuestion {
        id: "chemistry".to_string(),
        question: "Atoms?".to_string(),
        choices: vec![
            "A) a".to_string(),
            "B) b".to_string(),
            "C) c".to_string(),
            "D) d".to_string(),
        ],
        correct_answer: 2,
        subject: None,
        difficulty: None,
    });
    let eval = MmluEvaluator::new();
    let result = eval.evaluate_completions(&ds, &["C".to_string()]);
    assert!(result.by_subject.contains_key("chemistry"));
    let chem = result.by_subject.get("chemistry").unwrap();
    assert_eq!(chem.correct, 1);
}

#[test]
fn subject_field_takes_priority_over_id() {
    let mut ds = McDataset::new("mmlu");
    ds.add(MultipleChoiceQuestion {
        id: "biology/0".to_string(),
        question: "DNA?".to_string(),
        choices: vec![
            "A) a".to_string(),
            "B) b".to_string(),
            "C) c".to_string(),
            "D) d".to_string(),
        ],
        correct_answer: 0,
        subject: Some("molecular_biology".to_string()),
        difficulty: None,
    });
    let eval = MmluEvaluator::new();
    let result = eval.evaluate_completions(&ds, &["A".to_string()]);
    assert!(
        !result.by_subject.contains_key("biology"),
        "id prefix should be ignored when subject field is set"
    );
    assert!(result.by_subject.contains_key("molecular_biology"));
}

#[test]
fn multiple_subjects_independent_breakdown() {
    let mut ds = McDataset::new("mmlu");
    let subjects = ["algebra", "geometry", "calculus"];
    for (i, subj) in subjects.iter().enumerate() {
        for j in 0..4 {
            ds.add(MultipleChoiceQuestion {
                id: format!("{subj}/{j}"),
                question: format!("Q {subj} {j}"),
                choices: vec![
                    "A) a".to_string(),
                    "B) b".to_string(),
                    "C) c".to_string(),
                    "D) d".to_string(),
                ],
                correct_answer: i % 4,
                subject: Some(subj.to_string()),
                difficulty: None,
            });
        }
    }
    let eval = MmluEvaluator::new();
    // All correct for algebra (answer=0), all wrong for others
    let mut completions: Vec<String> = (0..4).map(|_| "A".to_string()).collect();
    completions.extend((0..4).map(|_| "D".to_string())); // geometry (correct=1)
    completions.extend((0..4).map(|_| "A".to_string())); // calculus (correct=2)
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.by_subject.len(), 3);
    let alg = result.by_subject.get("algebra").unwrap();
    assert_eq!(alg.correct, 4);
    let geom = result.by_subject.get("geometry").unwrap();
    assert_eq!(geom.correct, 0);
}

// ── Truncation when completions shorter than dataset ──────────────────────────

#[test]
fn fewer_completions_than_questions_only_annotated_scored() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    // Only 3 completions provided for 5 questions
    let completions = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let result = eval.evaluate_completions(&ds, &completions);
    assert_eq!(result.total, 3);
}

#[test]
fn fewer_logits_than_questions_only_annotated_scored() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    let per_choice_logits = vec![vec![1.0f32, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
    let result = eval.evaluate_logits(&ds, &per_choice_logits);
    assert_eq!(result.total, 2);
}

// ── Evaluate-by-subject helpers ───────────────────────────────────────────────

#[test]
fn evaluate_by_subject_completions_returns_subject_map() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    let completions = vec![
        "A".to_string(),
        "A".to_string(), // wrong for math/1 (correct=B)
        "C".to_string(),
        "A".to_string(),
        "D".to_string(),
    ];
    let by_subj = eval.evaluate_by_subject_completions(&ds, &completions);
    let math = by_subj.get("math").expect("math");
    assert_eq!(math.total, 3);
    assert_eq!(math.correct, 2); // math/0 ✓, math/1 ✗, math/2 ✓
    let history = by_subj.get("history").expect("history");
    assert_eq!(history.total, 2);
    assert_eq!(history.correct, 2);
}

#[test]
fn evaluate_by_subject_logits_returns_subject_map() {
    let eval = MmluEvaluator::new();
    let ds = dataset_with_subjects();
    let per_choice_logits = vec![
        vec![0.0f32, 0.0, 0.0, 1.0], // math/0 ✗
        vec![0.0, 1.0, 0.0, 0.0],    // math/1 ✓
        vec![0.0, 0.0, 1.0, 0.0],    // math/2 ✓
        vec![0.0, 1.0, 0.0, 0.0],    // history/0 ✗
        vec![0.0, 0.0, 0.0, 1.0],    // history/1 ✓
    ];
    let by_subj = eval.evaluate_by_subject_logits(&ds, &per_choice_logits);
    let math = by_subj.get("math").unwrap();
    assert_eq!(math.correct, 2);
    assert_eq!(math.total, 3);
    let history = by_subj.get("history").unwrap();
    assert_eq!(history.correct, 1);
    assert_eq!(history.total, 2);
}
