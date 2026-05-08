//! MMLU (Massive Multitask Language Understanding) evaluation harness.
//!
//! MMLU (Hendrycks et al., 2020) is a 57-subject four-choice multiple-choice
//! benchmark spanning STEM, humanities, social sciences, and professional
//! domains. Each question in the [`McDataset`] is expected to carry its subject
//! name in the `id` field as `"<subject>/<index>"` or in the `subject` field
//! directly.
//!
//! [`MmluEvaluator`] provides both overall accuracy and a per-subject breakdown
//! via [`MmluResult::by_subject`]. Subject extraction uses the following
//! priority:
//!
//! 1. If `question.subject` is `Some(s)`, use `s` directly.
//! 2. Otherwise extract from `question.id`: take the portion before the first
//!    `'/'`, or the whole `id` if `'/'` is absent.
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_eval::mmlu::MmluEvaluator;
//! use oxibonsai_eval::dataset::{McDataset, MultipleChoiceQuestion};
//!
//! let evaluator = MmluEvaluator::new();
//! let ds = McDataset::new("mmlu");
//! let result = evaluator.evaluate_logits(&ds, &[]);
//! assert_eq!(result.total, 0);
//! ```

use std::collections::HashMap;

use crate::accuracy::{AccuracyResult, McEvaluator, McLogitEvaluator};
use crate::dataset::{McDataset, MultipleChoiceQuestion};

// ──────────────────────────────────────────────────────────────────────────────
// Subject extraction
// ──────────────────────────────────────────────────────────────────────────────

/// Extract the subject name from a [`MultipleChoiceQuestion`].
///
/// Priority:
/// 1. `question.subject` — if present, returned as-is.
/// 2. `question.id`      — the portion before the first `'/'`, or the whole id.
fn extract_subject(question: &MultipleChoiceQuestion) -> String {
    if let Some(ref subj) = question.subject {
        return subj.clone();
    }
    // Fall back to parsing from id: "abstract_algebra/42" → "abstract_algebra"
    match question.id.find('/') {
        Some(pos) => question.id[..pos].to_string(),
        None => question.id.clone(),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MmluResult
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregated result from an MMLU evaluation run.
#[derive(Debug)]
pub struct MmluResult {
    /// Overall accuracy as a fraction in [0, 1].
    pub accuracy: f32,
    /// Overall accuracy as a percentage in [0, 100].
    pub accuracy_pct: f32,
    /// Number of correctly answered questions.
    pub correct: usize,
    /// Total number of questions evaluated.
    pub total: usize,
    /// Per-subject accuracy breakdown. Keys are subject names.
    pub by_subject: HashMap<String, AccuracyResult>,
}

// ──────────────────────────────────────────────────────────────────────────────
// MmluEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator for the MMLU benchmark.
///
/// Delegates to [`McEvaluator`] (completion-based path) and
/// [`McLogitEvaluator`] (logit-based path) for both overall scoring and
/// per-subject breakdown. MMLU is structurally identical to standard 4-choice
/// multiple-choice evaluation.
pub struct MmluEvaluator {
    /// String-completion-based evaluator.
    mc: McEvaluator,
    /// Logit-based evaluator.
    mc_logit: McLogitEvaluator,
}

impl MmluEvaluator {
    /// Create a new evaluator with the standard 4-choice MMLU prompt template.
    pub fn new() -> Self {
        let template =
            "{question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:".to_string();
        Self {
            mc: McEvaluator {
                prompt_template: template.clone(),
            },
            mc_logit: McLogitEvaluator {
                prompt_template: template,
            },
        }
    }

    // ── String-completion evaluation ──────────────────────────────────────────

    /// Evaluate using model string completions for overall accuracy.
    ///
    /// Each `completions[i]` must begin with the letter of the chosen answer
    /// (A–D). The by-subject breakdown is computed in the same pass.
    pub fn evaluate_completions(
        &self,
        dataset: &McDataset,
        completions: &[String],
    ) -> MmluResult {
        let overall = self.mc.evaluate_dataset(dataset, completions);
        let by_subject = self.evaluate_by_subject_completions(dataset, completions);
        MmluResult {
            accuracy: overall.accuracy,
            accuracy_pct: overall.accuracy * 100.0,
            correct: overall.correct,
            total: overall.total,
            by_subject,
        }
    }

    /// Evaluate using per-choice logit scores for overall accuracy.
    ///
    /// `per_choice_logits[i]` is a `Vec<f32>` of length 4, one log-probability
    /// per answer option. The argmax is selected as the prediction.
    pub fn evaluate_logits(
        &self,
        dataset: &McDataset,
        per_choice_logits: &[Vec<f32>],
    ) -> MmluResult {
        let overall = self.mc_logit.evaluate_dataset(dataset, per_choice_logits);
        let by_subject = self.evaluate_by_subject_logits(dataset, per_choice_logits);
        MmluResult {
            accuracy: overall.accuracy,
            accuracy_pct: overall.accuracy * 100.0,
            correct: overall.correct,
            total: overall.total,
            by_subject,
        }
    }

    // ── Per-subject evaluation ────────────────────────────────────────────────

    /// Compute per-subject accuracy from string completions.
    ///
    /// Returns a map from subject name → [`AccuracyResult`]. Questions without
    /// an extractable subject are grouped under their full `id`.
    pub fn evaluate_by_subject_completions(
        &self,
        dataset: &McDataset,
        completions: &[String],
    ) -> HashMap<String, AccuracyResult> {
        self.compute_by_subject(dataset, completions.len(), |i| {
            let q = &dataset.questions[i];
            let completion = &completions[i];
            self.mc.score_completion(completion, q.correct_answer)
        })
    }

    /// Compute per-subject accuracy from per-choice logit scores.
    ///
    /// Returns a map from subject name → [`AccuracyResult`]. Questions without
    /// an extractable subject are grouped under their full `id`.
    pub fn evaluate_by_subject_logits(
        &self,
        dataset: &McDataset,
        per_choice_logits: &[Vec<f32>],
    ) -> HashMap<String, AccuracyResult> {
        self.compute_by_subject(dataset, per_choice_logits.len(), |i| {
            let q = &dataset.questions[i];
            let slate = &per_choice_logits[i];
            let result = self.mc_logit.score(slate, q.correct_answer);
            result.correct
        })
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    /// Generic per-subject aggregator.
    ///
    /// Iterates over `min(dataset.len(), annotation_count)` questions and calls
    /// `is_correct(i)` to determine whether question `i` was answered correctly.
    /// Groups results by subject extracted via [`extract_subject`].
    fn compute_by_subject(
        &self,
        dataset: &McDataset,
        annotation_count: usize,
        mut is_correct: impl FnMut(usize) -> bool,
    ) -> HashMap<String, AccuracyResult> {
        // subject → (correct, total)
        let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
        let n = dataset.questions.len().min(annotation_count);

        for i in 0..n {
            let q = &dataset.questions[i];
            let subject = extract_subject(q);
            let entry = counts.entry(subject).or_insert((0, 0));
            entry.1 += 1;
            if is_correct(i) {
                entry.0 += 1;
            }
        }

        counts
            .into_iter()
            .map(|(subject, (correct, total))| {
                let accuracy = if total == 0 {
                    0.0_f32
                } else {
                    correct as f32 / total as f32
                };
                (
                    subject,
                    AccuracyResult {
                        correct,
                        total,
                        accuracy,
                        by_subject: std::collections::HashMap::new(),
                    },
                )
            })
            .collect()
    }
}

impl Default for MmluEvaluator {
    fn default() -> Self {
        Self::new()
    }
}
