//! GSM8K (Grade School Math 8K) evaluation harness.
//!
//! GSM8K answers are extracted from chain-of-thought solutions by locating the
//! **last** occurrence of the `"#### N"` delimiter that the GSM8K dataset
//! appends to every gold solution.  Model completions are expected to follow the
//! same convention.
//!
//! Numeric comparison uses an absolute tolerance of `1e-6` (or a relative
//! tolerance of `1e-6 × max(|a|, 1)` for large values) so that answers like
//! `42` and `42.0` compare equal, and `1,234` with embedded commas is parsed
//! correctly.
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_eval::gsm8k::Gsm8kEvaluator;
//!
//! let ev = Gsm8kEvaluator::new();
//! assert_eq!(Gsm8kEvaluator::extract_final_answer("Step 1... #### 42"), Some(42.0));
//! assert!(ev.score("Chain of thought #### 42", "#### 42"));
//! ```

use crate::dataset::{EvalDataset, EvalExample};

// ──────────────────────────────────────────────────────────────────────────────
// Gsm8kEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator for the GSM8K grade-school math benchmark.
///
/// The evaluator is stateless; all methods are either inherent functions or
/// `&self` methods that carry no mutable state. It implements [`Default`] so
/// it can be constructed by frameworks that require that bound.
pub struct Gsm8kEvaluator;

impl Gsm8kEvaluator {
    /// Create a new evaluator (equivalent to [`Default::default`]).
    pub fn new() -> Self {
        Self
    }

    /// Extract the numeric final answer from a GSM8K-formatted string.
    ///
    /// The extraction algorithm:
    /// 1. Find the **last** occurrence of the literal `"####"` in `text`.
    /// 2. Slice everything after the `"####"` marker.
    /// 3. Trim ASCII whitespace.
    /// 4. Strip embedded commas (e.g. `"1,234"` → `"1234"`).
    /// 5. Parse the resulting string as `f64`.
    /// 6. Return `None` if the marker is absent or parsing fails.
    ///
    /// Supported formats: integers (`42`), negatives (`-5`), decimals (`3.14`),
    /// comma-separated integers (`1,234`).
    pub fn extract_final_answer(text: &str) -> Option<f64> {
        // Step 1: find the last "####" marker.
        let marker = "####";
        let marker_pos = text.rfind(marker)?;

        // Step 2: everything after the marker.
        let after = &text[marker_pos + marker.len()..];

        // Step 3: trim ASCII whitespace from both ends.
        let trimmed = after.trim();

        // Step 4: strip commas (allow "1,234" → "1234").
        // We also need to handle a possible trailing newline after the number
        // that rfind/trim might not cover (covered by trim already).
        // Extract the numeric prefix: optional '-', then digits and '.'.
        // This avoids accepting things like "42abc" as 42 — only pure numeric
        // strings (after comma removal) are accepted.
        let no_commas: String = trimmed.chars().filter(|&c| c != ',').collect();

        // Step 5: parse as f64 — std parse handles leading '-' and decimals.
        no_commas.parse::<f64>().ok()
    }

    /// Score a single model completion against a gold answer string.
    ///
    /// Both strings are passed through [`extract_final_answer`]; the function
    /// returns `true` iff both succeed and the values agree within tolerance:
    ///
    /// ```text
    /// |a - b| < 1e-6 × max(|a|, 1.0)
    /// ```
    ///
    /// This relative formulation avoids false negatives for large integers while
    /// remaining tight enough to reject off-by-one errors.
    ///
    /// Returns `false` if either extraction fails.
    pub fn score(&self, completion: &str, gold: &str) -> bool {
        let Some(pred_val) = Self::extract_final_answer(completion) else {
            return false;
        };
        let Some(gold_val) = Self::extract_final_answer(gold) else {
            return false;
        };
        let tol = 1e-6_f64 * pred_val.abs().max(1.0);
        (pred_val - gold_val).abs() < tol
    }

    /// Evaluate model completions against a [`EvalDataset`].
    ///
    /// Each [`EvalExample::expected_output`] is used as the gold answer string;
    /// examples with `None` expected output are **skipped** (not counted as
    /// wrong).  The returned [`Gsm8kResult`] records accuracy, the count of
    /// correct answers, the total number of evaluated examples, and how many
    /// completions had no extractable `"####"` marker.
    pub fn evaluate_dataset(&self, dataset: &EvalDataset, completions: &[String]) -> Gsm8kResult {
        let mut correct: usize = 0;
        let mut total: usize = 0;
        let mut no_answer_extracted: usize = 0;

        for (example, completion) in dataset.examples.iter().zip(completions.iter()) {
            let Some(ref gold) = example.expected_output else {
                continue;
            };

            total += 1;

            // Track whether the completion had a parseable answer.
            if Self::extract_final_answer(completion).is_none() {
                no_answer_extracted += 1;
                // score() would return false, but we track the reason explicitly.
                continue;
            }

            if self.score(completion, gold) {
                correct += 1;
            }
        }

        let accuracy = if total == 0 {
            0.0_f32
        } else {
            correct as f32 / total as f32
        };

        Gsm8kResult {
            correct,
            total,
            accuracy,
            no_answer_extracted,
        }
    }
}

impl Default for Gsm8kEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Gsm8kResult
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregated results from a GSM8K evaluation run.
#[derive(Debug, Clone)]
pub struct Gsm8kResult {
    /// Number of questions answered correctly.
    pub correct: usize,
    /// Total number of questions evaluated (excludes examples with no gold answer).
    pub total: usize,
    /// Accuracy as a fraction in \[0, 1\].
    pub accuracy: f32,
    /// Number of completions from which no `"#### N"` answer could be extracted.
    ///
    /// These are counted in `total` but not in `correct`. A high value here
    /// suggests the model is not following the expected output format.
    pub no_answer_extracted: usize,
}

impl Gsm8kResult {
    /// Return accuracy as a percentage in \[0, 100\].
    pub fn accuracy_pct(&self) -> f32 {
        self.accuracy * 100.0
    }

    /// The fraction of completions that contained no parseable `"#### N"` marker.
    ///
    /// Returns 0 if `total == 0`.
    pub fn no_answer_rate(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.no_answer_extracted as f32 / self.total as f32
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Dataset builder helpers (public so tests / CLI tools can construct datasets)
// ──────────────────────────────────────────────────────────────────────────────

/// Build a single [`EvalExample`] for a GSM8K problem.
///
/// `input` is the question text shown to the model; `gold_answer` should be a
/// string that includes `"#### N"` (the whole solution can be stored here, or
/// just the answer line).
pub fn gsm8k_example(id: &str, input: &str, gold_answer: &str) -> EvalExample {
    EvalExample {
        id: id.to_string(),
        input: input.to_string(),
        expected_output: Some(gold_answer.to_string()),
        metadata: std::collections::HashMap::new(),
    }
}
