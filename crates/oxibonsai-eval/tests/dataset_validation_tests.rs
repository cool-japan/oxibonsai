//! RAG-M1: `McDataset::from_jsonl` accepted `correct_answer >= choices.len()`
//! and empty `choices`, silently producing a question that can never be
//! scored correct. Validate on load with a named error, and offer
//! `try_add` for the same guarantee on in-process construction.

use oxibonsai_eval::dataset::{McDataset, MultipleChoiceQuestion};
use oxibonsai_eval::EvalError;

/// `McDataset` does not implement `Debug` (it holds arbitrary
/// [`MultipleChoiceQuestion`] data, not itself a test-only concern), so
/// `Result::unwrap_err` is unavailable on `Result<McDataset, EvalError>`;
/// extract the error message via `match` instead.
fn expect_err_message(result: Result<McDataset, EvalError>) -> String {
    match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected Err, got Ok"),
    }
}

#[test]
fn from_jsonl_rejects_out_of_range_correct_answer() {
    // Exact reproduction from the missed finding.
    let jsonl = r#"{"id":"1","question":"Q","choices":["A","B"],"correct_answer":7}"#;
    let result = McDataset::from_jsonl("bad", jsonl);
    let msg = expect_err_message(result);
    assert!(
        msg.contains('7') && msg.contains('2'),
        "error message should mention the bad index and the choice count: {msg}"
    );
}

#[test]
fn from_jsonl_rejects_empty_choices() {
    let jsonl = r#"{"id":"1","question":"Q","choices":[],"correct_answer":0}"#;
    let result = McDataset::from_jsonl("bad", jsonl);
    assert!(
        result.is_err(),
        "empty choices must be rejected at load time"
    );
}

#[test]
fn from_jsonl_reports_the_correct_line_number() {
    let jsonl =
        "{\"id\":\"1\",\"question\":\"Q1\",\"choices\":[\"A\",\"B\"],\"correct_answer\":0}\n\
                 {\"id\":\"2\",\"question\":\"Q2\",\"choices\":[\"A\",\"B\"],\"correct_answer\":9}";
    let result = McDataset::from_jsonl("bad", jsonl);
    let msg = expect_err_message(result);
    assert!(
        msg.contains("line 2"),
        "the error should point at the second (bad) line: {msg}"
    );
}

#[test]
fn from_jsonl_still_accepts_a_well_formed_dataset() {
    let jsonl = r#"{"id":"1","question":"Q","choices":["A","B","C","D"],"correct_answer":2}"#;
    let ds = McDataset::from_jsonl("good", jsonl).expect("valid data must still load");
    assert_eq!(ds.len(), 1);
    assert_eq!(ds.questions[0].correct_answer, 2);
}

#[test]
fn from_jsonl_boundary_correct_answer_equal_to_len_is_rejected() {
    // correct_answer == choices.len() is the off-by-one edge (a 0-based
    // index equal to the length is out of range, not "the last choice").
    let jsonl = r#"{"id":"1","question":"Q","choices":["A","B"],"correct_answer":2}"#;
    assert!(McDataset::from_jsonl("bad", jsonl).is_err());
}

// ──────────────────────────────────────────────────────────────────────────────
// try_add
// ──────────────────────────────────────────────────────────────────────────────

fn q(choices: Vec<&str>, correct_answer: usize) -> MultipleChoiceQuestion {
    MultipleChoiceQuestion {
        id: "1".to_string(),
        question: "Q".to_string(),
        choices: choices.into_iter().map(str::to_string).collect(),
        correct_answer,
        subject: None,
        difficulty: None,
    }
}

#[test]
fn try_add_rejects_out_of_range_correct_answer() {
    let mut ds = McDataset::new("test");
    let result = ds.try_add(q(vec!["A", "B"], 7));
    assert!(matches!(result, Err(EvalError::InvalidFormat(_))));
    assert_eq!(ds.len(), 0, "a rejected question must not be inserted");
}

#[test]
fn try_add_rejects_empty_choices() {
    let mut ds = McDataset::new("test");
    let result = ds.try_add(q(vec![], 0));
    assert!(result.is_err());
    assert_eq!(ds.len(), 0);
}

#[test]
fn try_add_accepts_well_formed_question() {
    let mut ds = McDataset::new("test");
    ds.try_add(q(vec!["A", "B", "C"], 1))
        .expect("well-formed question must be accepted");
    assert_eq!(ds.len(), 1);
}

#[test]
fn add_keeps_its_original_infallible_signature() {
    // `add` must remain `fn(&mut self, MultipleChoiceQuestion)` with no
    // `Result` — changing that would break every other call site in this
    // crate (tests, hellaswag.rs,
    // winogrande.rs, ...), since an ignored `Result` trips
    // `unused_must_use` under `-D warnings`. This test is a compile-time
    // guarantee: it only builds if `add` still returns `()`.
    let mut ds = McDataset::new("test");
    let () = ds.add(q(vec!["A", "B"], 9)); // permissive: not validated
    assert_eq!(ds.len(), 1);
    assert_eq!(ds.questions[0].correct_answer, 9);
}
