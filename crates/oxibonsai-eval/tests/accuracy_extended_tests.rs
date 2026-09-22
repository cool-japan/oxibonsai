//! Regression tests for the accuracy.rs fixes in package EVAL-METRICS:
//!
//! - RAG-EVAL-IMG-01: `McEvaluator::format_question` panicked on any
//!   non-ASCII answer choice (byte-index slicing at a fixed offset).
//! - RAG-EVAL-IMG-03: `extract_answer` mis-scored reasoning-model output,
//!   silently crediting `"Answer: B"` as choice A, and defaulted to index 0
//!   instead of returning `None` when nothing could be extracted.
//! - RAG-M3: `format_question`'s chained `.replace()` calls let question or
//!   choice text containing literal `{a}`..`{d}` expand into a later
//!   choice's placeholder.
//! - RAG-EVAL-IMG-04: 5+ option support (see also `arc_n_option_tests.rs`).

use oxibonsai_eval::accuracy::McEvaluator;
use oxibonsai_eval::dataset::MultipleChoiceQuestion;

fn q_with_choices(choices: Vec<&str>, correct_answer: usize) -> MultipleChoiceQuestion {
    MultipleChoiceQuestion {
        id: "q1".to_string(),
        question: "What is the capital?".to_string(),
        choices: choices.into_iter().map(str::to_string).collect(),
        correct_answer,
        subject: None,
        difficulty: None,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RAG-EVAL-IMG-01 — format_question must not panic on non-ASCII choices
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn format_question_cjk_labelled_choice_does_not_panic_and_strips_label() {
    // "あ: 東京" — a one-character CJK label followed by ": " and CJK
    // answer text. The original implementation sliced at the fixed byte
    // offset 2, which lands mid-character inside "あ" (3 bytes in UTF-8)
    // and panics with "byte index 2 is not a char boundary".
    let eval = McEvaluator::new();
    let q = q_with_choices(vec!["あ: 東京", "B: 大阪", "C: 京都", "D: 名古屋"], 0);
    let formatted = eval.format_question(&q); // must not panic
    assert!(
        formatted.contains("東京"),
        "label should be stripped, leaving the answer text: {formatted}"
    );
    assert!(
        !formatted.contains("あ:"),
        "the CJK label prefix should have been stripped: {formatted}"
    );
}

#[test]
fn format_question_emoji_labelled_choice_does_not_panic_and_strips_label() {
    // An emoji is 4 bytes in UTF-8 — an even more extreme case than a
    // 3-byte CJK character for the same fixed-offset-slicing bug.
    let eval = McEvaluator::new();
    let q = q_with_choices(vec!["😀: happy", "B: sad", "C: angry", "D: calm"], 0);
    let formatted = eval.format_question(&q); // must not panic
    assert!(
        formatted.contains("happy"),
        "label should be stripped: {formatted}"
    );
    assert!(!formatted.contains("😀:"), "formatted: {formatted}");
}

#[test]
fn format_question_bare_label_without_content_is_left_unchanged() {
    // A label with nothing after the colon ("A:") must NOT be stripped
    // down to an empty string — only strip when non-whitespace content
    // remains after the colon, matching the original `s.len() >= 3` guard's
    // intent (verified pre-existing behaviour: this renders as "A) A:").
    let eval = McEvaluator::new();
    let q = q_with_choices(vec!["A:", "B: y", "C: z", "D: w"], 0);
    let formatted = eval.format_question(&q);
    assert!(
        formatted.contains("A) A:"),
        "bare label with no content should be left as-is: {formatted}"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// RAG-M3 — chained substitution injection
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn format_question_question_text_placeholder_is_not_reinterpreted() {
    // A question stem containing literal "{b}" must not be re-scanned once
    // it has been substituted in for `{question}` — the real `{b}`
    // placeholder (from the template) must still resolve to choice index 1.
    let eval = McEvaluator::new();
    let q = MultipleChoiceQuestion {
        id: "q1".to_string(),
        question: "Pick {b} now".to_string(),
        choices: vec![
            "one".to_string(),
            "TWO".to_string(),
            "three".to_string(),
            "four".to_string(),
        ],
        correct_answer: 1,
        subject: None,
        difficulty: None,
    };
    let formatted = eval.format_question(&q);
    assert!(
        formatted.contains("Pick {b} now"),
        "the literal placeholder-looking text in the question must survive verbatim: {formatted}"
    );
    assert!(
        formatted.contains("B) TWO"),
        "the template's real {{b}} placeholder must still resolve to choice 1: {formatted}"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// RAG-EVAL-IMG-04 — N-option rendering (5th choice not silently dropped)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn format_question_five_choices_appends_fifth_option() {
    let eval = McEvaluator::new(); // standard 4-slot {a}..{d} template
    let q = q_with_choices(vec!["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], 4);
    let formatted = eval.format_question(&q);
    assert!(
        formatted.contains("E) Epsilon"),
        "a 5th choice must be rendered even though the template only wires {{a}}..{{d}}: {formatted}"
    );
    assert!(
        formatted.contains("D) Delta\nE) Epsilon\nAnswer:"),
        "the 5th choice must appear alongside the others, before the trailing Answer: marker: {formatted}"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// RAG-EVAL-IMG-03 — extract_answer correctness
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_answer_headline_bug_is_fixed() {
    let eval = McEvaluator::new();
    // Previously: the completion's first character is the 'A' of the word
    // "Answer" itself, so this was silently credited as choice A (index 0)
    // even though the model clearly chose B.
    assert_eq!(eval.extract_answer("Answer: B"), Some(1));
}

#[test]
fn extract_answer_strips_think_block_before_reading_the_answer() {
    let eval = McEvaluator::new();
    assert_eq!(
        eval.extract_answer("<think>Let me reason about this for a while.</think>B"),
        Some(1)
    );
    // A completion made *only* of reasoning, with the real answer stated
    // via the marker inside the visible remainder.
    assert_eq!(
        eval.extract_answer("<think>hmm, tricky</think>Answer: C"),
        Some(2)
    );
}

#[test]
fn extract_answer_uses_the_last_marker_occurrence() {
    let eval = McEvaluator::new();
    assert_eq!(
        eval.extract_answer("Answer: A, on reflection, Answer: B"),
        Some(1)
    );
}

#[test]
fn extract_answer_does_not_credit_prose_after_the_marker() {
    let eval = McEvaluator::new();
    // The character right after "Answer:" here starts a longer word
    // ("Before"/"after"), so it must not be treated as a standalone answer
    // letter — and, critically, must NOT fall back to crediting the 'A' of
    // "Answer" itself (which is exactly the RAG-EVAL-IMG-03 bug reappearing
    // through the fallback path).
    assert_eq!(
        eval.extract_answer("Answer: Before we begin, the choice is A"),
        None
    );
    assert_eq!(eval.extract_answer("The answer: after all, B"), None);
}

#[test]
fn extract_answer_accepts_standalone_letter_with_trailing_punctuation() {
    let eval = McEvaluator::new();
    assert_eq!(eval.extract_answer("Answer: B."), Some(1));
    assert_eq!(eval.extract_answer("Answer: B)"), Some(1));
    assert_eq!(eval.extract_answer("Answer:\nB\n"), Some(1));
}

#[test]
fn extract_answer_legacy_first_char_fallback_still_works_without_a_marker() {
    let eval = McEvaluator::new();
    // No "answer:" marker anywhere in these — preserves the crate's
    // original first-character heuristic exactly.
    assert_eq!(eval.extract_answer("B is correct"), Some(1));
    assert_eq!(eval.extract_answer("  D"), Some(3));
}

#[test]
fn extract_answer_never_defaults_to_zero_on_unparseable_input() {
    let eval = McEvaluator::new();
    // None of these should ever resolve to `Some(0)` by default — every
    // public entry point in this crate must return `None`, not a fabricated
    // index, when it cannot determine an answer.
    for input in [
        "",
        "   ",
        "I'm not sure",
        "<think>still thinking with no conclusion",
        "42",
        "Answer:",
        "Answer:   ",
    ] {
        assert_eq!(
            eval.extract_answer(input),
            None,
            "expected None for {input:?}"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// N-option bounding (score_completion_bounded / extract_answer_bounded)
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_answer_bounded_accepts_e_for_five_choices_but_not_for_four() {
    let eval = McEvaluator::new();
    assert_eq!(eval.extract_answer_bounded("E", 5), Some(4));
    assert_eq!(eval.extract_answer_bounded("E", 4), None);
    // The crate's historical default (`extract_answer`) is bounded to 4.
    assert_eq!(eval.extract_answer("E"), None);
}

#[test]
fn score_completion_bounded_scores_five_option_question() {
    let eval = McEvaluator::new();
    let q = q_with_choices(vec!["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], 4);
    assert!(eval.score_completion_bounded("E", q.correct_answer, q.choices.len()));
    assert!(!eval.score_completion_bounded("A", q.correct_answer, q.choices.len()));
}

#[test]
fn evaluate_dataset_scores_five_option_questions_via_per_question_bound() {
    use oxibonsai_eval::dataset::McDataset;

    let eval = McEvaluator::new();
    let mut ds = McDataset::new("five-choice");
    ds.add(q_with_choices(
        vec!["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        4,
    ));
    ds.add(q_with_choices(vec!["one", "two", "three", "four"], 1));

    let completions = vec!["E".to_string(), "B".to_string()];
    let result = eval.evaluate_dataset(&ds, &completions);
    assert_eq!(result.correct, 2);
    assert_eq!(result.total, 2);
}
