//! BoolQ yes/no question answering evaluation harness.
//!
//! BoolQ (Clark et al., 2019) tests reading comprehension via binary yes/no
//! questions. Each item has a passage, a question, and a boolean answer.
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_eval::boolq::{BoolQDataset, BoolQEvaluator, BoolQItem};
//!
//! let evaluator = BoolQEvaluator::new();
//! let answer = BoolQEvaluator::extract_answer("Yes, that is correct.");
//! assert_eq!(answer, Some(true));
//! ```

// ──────────────────────────────────────────────────────────────────────────────
// BoolQItem
// ──────────────────────────────────────────────────────────────────────────────

/// A single BoolQ dataset item.
#[derive(Debug, Clone, PartialEq)]
pub struct BoolQItem {
    /// The context passage from which the answer can be inferred.
    pub passage: String,
    /// The yes/no question about the passage.
    pub question: String,
    /// Gold answer: `true` = yes, `false` = no.
    pub answer: bool,
}

// ──────────────────────────────────────────────────────────────────────────────
// BoolQDataset
// ──────────────────────────────────────────────────────────────────────────────

/// A collection of [`BoolQItem`] instances.
pub struct BoolQDataset {
    /// All items in insertion order.
    pub items: Vec<BoolQItem>,
}

impl BoolQDataset {
    /// Create a dataset from a vector of items.
    pub fn from_items(items: Vec<BoolQItem>) -> Self {
        Self { items }
    }

    /// Return the number of items in the dataset.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Return `true` if the dataset contains no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BoolQResult
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregated result from a BoolQ evaluation pass.
#[derive(Debug, Clone)]
pub struct BoolQResult {
    /// Fraction of items answered correctly (0.0–1.0).
    pub accuracy: f32,
    /// Accuracy as a percentage (0.0–100.0).
    pub accuracy_pct: f32,
    /// Number of correctly answered items.
    pub correct: usize,
    /// Total items evaluated (excludes items beyond the shorter of dataset/completions).
    pub total: usize,
    /// Number of items where the model predicted "yes" (true).
    pub yes_predicted: usize,
    /// Number of items where the model predicted "no" (false).
    pub no_predicted: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// BoolQEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator for BoolQ yes/no question answering.
///
/// The evaluator is stateless; all scoring logic is encoded in its methods.
/// It implements [`Default`] for use with frameworks that require that bound.
pub struct BoolQEvaluator;

impl BoolQEvaluator {
    /// Create a new evaluator (equivalent to [`Default::default`]).
    pub fn new() -> Self {
        Self
    }

    /// Build a prompt string for a BoolQ item.
    ///
    /// Format: `"Passage: {passage}\nQuestion: {question}\nAnswer:"`
    pub fn build_prompt(&self, item: &BoolQItem) -> String {
        format!(
            "Passage: {}\nQuestion: {}\nAnswer:",
            item.passage, item.question
        )
    }

    /// Extract a boolean answer from a model completion string.
    ///
    /// Algorithm:
    /// 1. Strip leading ASCII whitespace.
    /// 2. Take the first 3 characters and lowercase them.
    /// 3. If the prefix is `"yes"`, return `Some(true)`.
    /// 4. If the prefix starts with `"no"`, return `Some(false)`.
    /// 5. Otherwise return `None`.
    ///
    /// A string shorter than 2 characters always returns `None`.
    pub fn extract_answer(completion: &str) -> Option<bool> {
        let trimmed = completion.trim_start();
        // Need at least "no" (2 chars) to match anything.
        if trimmed.len() < 2 {
            return None;
        }
        // Collect up to 3 chars for case-insensitive prefix matching.
        let prefix: String = trimmed.chars().take(3).collect::<String>().to_lowercase();
        if prefix.starts_with("yes") {
            Some(true)
        } else if prefix.starts_with("no") {
            Some(false)
        } else {
            None
        }
    }

    /// Score a single completion against the gold boolean answer.
    ///
    /// Returns `true` iff [`extract_answer`](Self::extract_answer) produces
    /// `Some(gold)`, `false` otherwise (including when extraction fails).
    pub fn score(&self, completion: &str, gold: bool) -> bool {
        Self::extract_answer(completion) == Some(gold)
    }

    /// Evaluate a list of string completions against the dataset.
    ///
    /// Only the first `min(dataset.len(), completions.len())` items are scored.
    /// Completions where [`extract_answer`](Self::extract_answer) returns `None`
    /// are counted as wrong but not added to `yes_predicted` or `no_predicted`.
    pub fn evaluate_completions(
        &self,
        dataset: &BoolQDataset,
        completions: &[String],
    ) -> BoolQResult {
        let n = dataset.items.len().min(completions.len());
        let mut correct = 0usize;
        let mut yes_predicted = 0usize;
        let mut no_predicted = 0usize;

        for (i, completion) in completions.iter().enumerate().take(n) {
            let prediction = Self::extract_answer(completion);
            match prediction {
                Some(true) => yes_predicted += 1,
                Some(false) => no_predicted += 1,
                None => {}
            }
            if prediction == Some(dataset.items[i].answer) {
                correct += 1;
            }
        }

        let total = n;
        let accuracy = if total == 0 {
            0.0_f32
        } else {
            correct as f32 / total as f32
        };
        BoolQResult {
            accuracy,
            accuracy_pct: accuracy * 100.0,
            correct,
            total,
            yes_predicted,
            no_predicted,
        }
    }

    /// Evaluate using logit pairs `[logit_yes, logit_no]` per item.
    ///
    /// Prediction: if `logit_pairs[i][0] > logit_pairs[i][1]` then `true` (yes),
    /// otherwise `false` (no). The prediction is compared to `dataset.items[i].answer`.
    ///
    /// Only the first `min(dataset.len(), logit_pairs.len())` items are scored.
    pub fn evaluate_logits(
        &self,
        dataset: &BoolQDataset,
        logit_pairs: &[[f32; 2]],
    ) -> BoolQResult {
        let n = dataset.items.len().min(logit_pairs.len());
        let mut correct = 0usize;
        let mut yes_predicted = 0usize;
        let mut no_predicted = 0usize;

        for (i, pair) in logit_pairs.iter().enumerate().take(n) {
            let pred_yes = pair[0] > pair[1];
            if pred_yes {
                yes_predicted += 1;
            } else {
                no_predicted += 1;
            }
            if pred_yes == dataset.items[i].answer {
                correct += 1;
            }
        }

        let total = n;
        let accuracy = if total == 0 {
            0.0_f32
        } else {
            correct as f32 / total as f32
        };
        BoolQResult {
            accuracy,
            accuracy_pct: accuracy * 100.0,
            correct,
            total,
            yes_predicted,
            no_predicted,
        }
    }
}

impl Default for BoolQEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_answer_yes() {
        assert_eq!(
            BoolQEvaluator::extract_answer("Yes, that is correct."),
            Some(true)
        );
    }

    #[test]
    fn extract_answer_no() {
        assert_eq!(
            BoolQEvaluator::extract_answer("No, it is not."),
            Some(false)
        );
    }

    #[test]
    fn extract_answer_case_insensitive() {
        assert_eq!(BoolQEvaluator::extract_answer("YES"), Some(true));
        assert_eq!(BoolQEvaluator::extract_answer("NO"), Some(false));
        assert_eq!(BoolQEvaluator::extract_answer("yes."), Some(true));
    }

    #[test]
    fn extract_answer_leading_whitespace() {
        assert_eq!(BoolQEvaluator::extract_answer("  yes"), Some(true));
        assert_eq!(BoolQEvaluator::extract_answer("\t\nno"), Some(false));
    }

    #[test]
    fn extract_answer_none_for_garbage() {
        assert_eq!(BoolQEvaluator::extract_answer("maybe"), None);
        assert_eq!(BoolQEvaluator::extract_answer(""), None);
        assert_eq!(BoolQEvaluator::extract_answer("I don't know"), None);
    }

    #[test]
    fn extract_answer_short_string_no_panic() {
        assert_eq!(BoolQEvaluator::extract_answer("y"), None);
        assert_eq!(BoolQEvaluator::extract_answer("n"), None);
    }
}
