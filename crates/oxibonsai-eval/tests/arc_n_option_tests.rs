//! RAG-EVAL-IMG-04: the ARC evaluator must be able to *score* (not just
//! render logits for) a 5-option question through the string-completion
//! path, and the false "extract_answer also handles 'E'" comment must be
//! gone. `arc_tests.rs` already covers the 5-option *logit* path
//! (`arc_five_choice_question`); this file adds the completion path, which
//! was the actual gap.

use oxibonsai_eval::arc::ArcEvaluator;
use oxibonsai_eval::dataset::{McDataset, MultipleChoiceQuestion};

fn five_choice_q(id: &str, correct: usize) -> MultipleChoiceQuestion {
    MultipleChoiceQuestion {
        id: id.to_string(),
        question: format!("Question {id}"),
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

#[test]
fn arc_evaluate_completions_scores_five_option_item() {
    let mut ds = McDataset::new("arc-5opt");
    ds.add(five_choice_q("q0", 4)); // correct = E
    ds.add(five_choice_q("q1", 0)); // correct = A

    let completions = vec!["E".to_string(), "A".to_string()];
    let ev = ArcEvaluator::challenge();
    let result = ev.evaluate_completions(&ds, &completions);

    assert_eq!(result.correct, 2, "both 5-option completions should score");
    assert_eq!(result.total, 2);
}

#[test]
fn arc_evaluate_completions_five_option_wrong_answer_not_credited() {
    let mut ds = McDataset::new("arc-5opt");
    ds.add(five_choice_q("q0", 4)); // correct = E

    // "A" should not be silently credited as correct just because E is a
    // valid letter for this dataset.
    let completions = vec!["A".to_string()];
    let ev = ArcEvaluator::easy();
    let result = ev.evaluate_completions(&ds, &completions);
    assert_eq!(result.correct, 0);
    assert_eq!(result.total, 1);
}

#[test]
fn arc_evaluate_completions_four_option_item_still_rejects_e() {
    // A 4-choice ARC-Easy item must not accept "E" as a valid letter — the
    // bound comes from the *question's own* choice count, not a global max.
    let q = MultipleChoiceQuestion {
        id: "q0".to_string(),
        question: "Q".to_string(),
        choices: vec![
            "Alpha".to_string(),
            "Beta".to_string(),
            "Gamma".to_string(),
            "Delta".to_string(),
        ],
        correct_answer: 3,
        subject: None,
        difficulty: None,
    };
    let mut ds = McDataset::new("arc-4opt");
    ds.add(q);
    let ev = ArcEvaluator::easy();
    let result = ev.evaluate_completions(&ds, &["E".to_string()]);
    assert_eq!(result.correct, 0, "E is out of range for a 4-choice item");
}

#[test]
fn arc_format_question_renders_fifth_choice() {
    let ev = ArcEvaluator::challenge();
    let q = five_choice_q("q0", 4);
    let formatted = ev.format_question(&q);
    assert!(
        formatted.contains("E) Epsilon"),
        "ArcEvaluator::format_question must surface a 5th option: {formatted}"
    );
}
