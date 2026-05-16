//! Integration tests for the ARC (AI2 Reasoning Challenge) evaluator.

use oxibonsai_eval::arc::{ArcEvaluator, ArcResult, ArcSplit};
use oxibonsai_eval::dataset::{McDataset, MultipleChoiceQuestion};
use oxibonsai_eval::AccuracyResult;
use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Helper: build a synthetic McDataset
// ──────────────────────────────────────────────────────────────────────────────

/// Build a 4-choice question where choice index `correct` is the answer.
fn four_choice_q(id: &str, correct: usize) -> MultipleChoiceQuestion {
    MultipleChoiceQuestion {
        id: id.to_string(),
        question: format!("Question {}", id),
        choices: vec![
            "Alpha".to_string(),
            "Beta".to_string(),
            "Gamma".to_string(),
            "Delta".to_string(),
        ],
        correct_answer: correct,
        subject: None,
        difficulty: None,
    }
}

/// Build a 5-choice question where choice index `correct` is the answer.
fn five_choice_q(id: &str, correct: usize) -> MultipleChoiceQuestion {
    MultipleChoiceQuestion {
        id: id.to_string(),
        question: format!("Question {}", id),
        choices: vec![
            "Alpha".to_string(),
            "Beta".to_string(),
            "Gamma".to_string(),
            "Delta".to_string(),
            "Epsilon".to_string(),
        ],
        correct_answer: correct,
        subject: None,
        difficulty: None,
    }
}

fn dataset_from_questions(questions: Vec<MultipleChoiceQuestion>) -> McDataset {
    let mut ds = McDataset::new("test-arc");
    for q in questions {
        ds.add(q);
    }
    ds
}

/// Map a 0-based index to the letter that McEvaluator expects as first char.
fn answer_letter(idx: usize) -> String {
    let letter = match idx {
        0 => 'A',
        1 => 'B',
        2 => 'C',
        3 => 'D',
        4 => 'E',
        _ => panic!("unsupported index"),
    };
    letter.to_string()
}

// ──────────────────────────────────────────────────────────────────────────────
// 1. ArcEvaluator::easy() creates Easy split
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn arc_evaluator_easy_creation() {
    let ev = ArcEvaluator::easy();
    assert_eq!(ev.split(), ArcSplit::Easy);
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. ArcEvaluator::challenge() creates Challenge split
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn arc_evaluator_challenge_creation() {
    let ev = ArcEvaluator::challenge();
    assert_eq!(ev.split(), ArcSplit::Challenge);
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. ArcSplit::name() returns the canonical benchmark strings
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn arc_split_name() {
    assert_eq!(ArcSplit::Easy.name(), "ARC-Easy");
    assert_eq!(ArcSplit::Challenge.name(), "ARC-Challenge");
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. evaluate_completions — all correct → 100% accuracy
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn arc_evaluate_completions_all_correct() {
    // Four questions with different correct answers.
    let questions = vec![
        four_choice_q("q0", 0), // answer A
        four_choice_q("q1", 1), // answer B
        four_choice_q("q2", 2), // answer C
        four_choice_q("q3", 3), // answer D
    ];
    let ds = dataset_from_questions(questions);

    let completions: Vec<String> = vec![
        answer_letter(0),
        answer_letter(1),
        answer_letter(2),
        answer_letter(3),
    ];

    let ev = ArcEvaluator::easy();
    let result = ev.evaluate_completions(&ds, &completions);

    assert_eq!(result.correct, 4);
    assert_eq!(result.total, 4);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. evaluate_completions — all wrong → 0% accuracy
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn arc_evaluate_completions_all_wrong() {
    let questions = vec![
        four_choice_q("q0", 0), // correct = A, we give B
        four_choice_q("q1", 1), // correct = B, we give C
        four_choice_q("q2", 2), // correct = C, we give D
        four_choice_q("q3", 3), // correct = D, we give A
    ];
    let ds = dataset_from_questions(questions);

    let completions: Vec<String> = vec![
        answer_letter(1),
        answer_letter(2),
        answer_letter(3),
        answer_letter(0),
    ];

    let ev = ArcEvaluator::challenge();
    let result = ev.evaluate_completions(&ds, &completions);

    assert_eq!(result.correct, 0);
    assert_eq!(result.total, 4);
    assert!(result.accuracy.abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 6. evaluate_completions — partial score (2 of 4 correct)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn arc_evaluate_completions_partial() {
    let questions = vec![
        four_choice_q("q0", 0), // correct = A
        four_choice_q("q1", 1), // correct = B
        four_choice_q("q2", 2), // correct = C
        four_choice_q("q3", 3), // correct = D
    ];
    let ds = dataset_from_questions(questions);

    // First two correct, last two wrong.
    let completions: Vec<String> = vec![
        answer_letter(0), // correct
        answer_letter(1), // correct
        answer_letter(0), // wrong (correct is C)
        answer_letter(0), // wrong (correct is D)
    ];

    let ev = ArcEvaluator::easy();
    let result = ev.evaluate_completions(&ds, &completions);

    assert_eq!(result.correct, 2);
    assert_eq!(result.total, 4);
    assert!((result.accuracy - 0.5).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 7. evaluate_logits — argmax picks the correct choice
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn arc_evaluate_logits_picks_max() {
    // Three questions; correct answer is the index with the highest logit.
    let questions = vec![
        four_choice_q("q0", 2), // correct = C (index 2)
        four_choice_q("q1", 0), // correct = A (index 0)
        four_choice_q("q2", 3), // correct = D (index 3)
    ];
    let ds = dataset_from_questions(questions);

    // Provide per-choice logits where the highest value == correct_answer index.
    let logits: Vec<Vec<f32>> = vec![
        vec![-1.0, -2.0, 0.5, -0.5], // argmax = 2 → correct (C)
        vec![1.0, 0.0, -1.0, -2.0],  // argmax = 0 → correct (A)
        vec![-3.0, -2.0, -1.0, 2.0], // argmax = 3 → correct (D)
    ];

    let ev = ArcEvaluator::challenge();
    let result = ev.evaluate_logits(&ds, &logits);

    assert_eq!(result.correct, 3);
    assert_eq!(result.total, 3);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 8. ArcResult::from_accuracy_result propagates fields correctly
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn arc_result_from_accuracy_result() {
    let accuracy_result = AccuracyResult {
        correct: 7,
        total: 10,
        accuracy: 0.7,
        by_subject: HashMap::new(),
    };

    let arc_result = ArcResult::from_accuracy_result(ArcSplit::Challenge, accuracy_result);

    assert_eq!(arc_result.split, ArcSplit::Challenge);
    assert_eq!(arc_result.correct, 7);
    assert_eq!(arc_result.total, 10);
    assert!((arc_result.accuracy - 0.7).abs() < 1e-6);
    assert!((arc_result.accuracy_pct() - 70.0).abs() < 1e-4);
    assert_eq!(arc_result.split_name(), "ARC-Challenge");
}

// ──────────────────────────────────────────────────────────────────────────────
// 9. Five-choice question (ARC can have 5 options, keyed A-E)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn arc_five_choice_question() {
    // ARC sometimes presents E as the fifth option (index 4).
    // The McLogitEvaluator handles variable-length logit slates; we verify that
    // a five-element logit vector is scored correctly.
    let questions = vec![
        five_choice_q("q0", 4), // correct = E (index 4)
        five_choice_q("q1", 0), // correct = A (index 0)
    ];
    let ds = dataset_from_questions(questions);

    let logits: Vec<Vec<f32>> = vec![
        vec![-1.0, -1.0, -1.0, -1.0, 2.0], // argmax = 4 → correct (E)
        vec![3.0, 1.0, 0.0, -1.0, -2.0],   // argmax = 0 → correct (A)
    ];

    let ev = ArcEvaluator::easy();
    let result = ev.evaluate_logits(&ds, &logits);

    assert_eq!(result.correct, 2);
    assert_eq!(result.total, 2);
    assert!((result.accuracy - 1.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 10. Empty dataset → zero result (guard against division by zero)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn arc_empty_dataset_returns_zero_accuracy() {
    let ds = McDataset::new("empty");
    let completions: Vec<String> = vec![];

    let ev = ArcEvaluator::easy();
    let result = ev.evaluate_completions(&ds, &completions);

    assert_eq!(result.correct, 0);
    assert_eq!(result.total, 0);
    assert!(result.accuracy.abs() < 1e-6);
}
