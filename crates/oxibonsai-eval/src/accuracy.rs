//! Accuracy evaluation: multiple-choice (MMLU-style) and exact-match scoring.

use std::collections::HashMap;

use serde::Serialize;

use crate::dataset::{EvalDataset, McDataset, MultipleChoiceQuestion};

// ──────────────────────────────────────────────────────────────────────────────
// AccuracyResult
// ──────────────────────────────────────────────────────────────────────────────

/// Accuracy statistics from a completed evaluation run.
#[derive(Debug, Serialize)]
pub struct AccuracyResult {
    /// Number of correctly answered examples.
    pub correct: usize,
    /// Total number of examples attempted.
    pub total: usize,
    /// Accuracy as a fraction in [0, 1].
    pub accuracy: f32,
    /// Per-subject accuracy (fraction). Key is subject name.
    pub by_subject: HashMap<String, f32>,
}

impl AccuracyResult {
    /// Return accuracy as a percentage in [0, 100].
    pub fn accuracy_pct(&self) -> f32 {
        self.accuracy * 100.0
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// McEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator for MMLU-style multiple-choice questions.
///
/// The prompt template supports the following placeholders:
/// - `{question}` — the question stem
/// - `{a}`, `{b}`, `{c}`, `{d}` — choices 0–3 (text after the label prefix)
pub struct McEvaluator {
    /// Template used to format questions into prompts.
    pub prompt_template: String,
}

impl Default for McEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl McEvaluator {
    /// Create an evaluator with the standard four-choice template.
    pub fn new() -> Self {
        Self {
            prompt_template: "{question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:".to_string(),
        }
    }

    /// Create an evaluator with a custom prompt template.
    pub fn with_template(template: &str) -> Self {
        Self {
            prompt_template: template.to_string(),
        }
    }

    /// Format a question into a prompt string using the stored template.
    ///
    /// Choice text is taken from each entry of `q.choices`. If there are fewer
    /// than 4 choices the missing slots are replaced with an empty string.
    pub fn format_question(&self, q: &MultipleChoiceQuestion) -> String {
        let get = |i: usize| -> &str { q.choices.get(i).map(String::as_str).unwrap_or("") };

        // Strip leading label prefix like "A: " if present
        let strip_label = |s: &str| -> String {
            if s.len() >= 3 && s.chars().nth(1) == Some(':') {
                s[2..].trim().to_string()
            } else {
                s.to_string()
            }
        };

        self.prompt_template
            .replace("{question}", &q.question)
            .replace("{a}", &strip_label(get(0)))
            .replace("{b}", &strip_label(get(1)))
            .replace("{c}", &strip_label(get(2)))
            .replace("{d}", &strip_label(get(3)))
    }

    /// Return `true` if the completion starts with the correct answer letter.
    ///
    /// `correct_answer` is a 0-based index; 0 → 'A', 1 → 'B', 2 → 'C', 3 → 'D'.
    pub fn score_completion(&self, completion: &str, correct_answer: usize) -> bool {
        match self.extract_answer(completion) {
            Some(idx) => idx == correct_answer,
            None => false,
        }
    }

    /// Parse the first letter of the completion into a 0-based answer index.
    ///
    /// Accepts 'A'/'a' → 0, 'B'/'b' → 1, 'C'/'c' → 2, 'D'/'d' → 3.
    /// Returns `None` for any other leading character.
    pub fn extract_answer(&self, completion: &str) -> Option<usize> {
        let first = completion.trim().chars().next()?;
        match first.to_ascii_uppercase() {
            'A' => Some(0),
            'B' => Some(1),
            'C' => Some(2),
            'D' => Some(3),
            _ => None,
        }
    }

    /// Evaluate a multiple-choice dataset given one completion per question.
    ///
    /// `completions` must have the same length as `dataset.questions`.
    /// Mismatched lengths are handled gracefully: only the shorter slice is used.
    pub fn evaluate_dataset(&self, dataset: &McDataset, completions: &[String]) -> AccuracyResult {
        let mut correct = 0usize;
        let mut total = 0usize;

        // subject → (correct, total)
        let mut by_subject_counts: HashMap<String, (usize, usize)> = HashMap::new();

        for (q, completion) in dataset.questions.iter().zip(completions.iter()) {
            total += 1;
            let is_correct = self.score_completion(completion, q.correct_answer);
            if is_correct {
                correct += 1;
            }

            if let Some(ref subj) = q.subject {
                let entry = by_subject_counts.entry(subj.clone()).or_insert((0, 0));
                entry.1 += 1;
                if is_correct {
                    entry.0 += 1;
                }
            }
        }

        let accuracy = if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        };

        let by_subject = by_subject_counts
            .into_iter()
            .map(|(subj, (c, t))| (subj, if t == 0 { 0.0 } else { c as f32 / t as f32 }))
            .collect();

        AccuracyResult {
            correct,
            total,
            accuracy,
            by_subject,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// McLogitEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// One scoring outcome from a logit-based multiple-choice evaluation.
#[derive(Debug, Clone)]
pub struct LogitMcResult {
    /// Index of the option the evaluator picked.
    pub picked: usize,
    /// Whether [`LogitMcResult::picked`] equals the ground-truth index.
    pub correct: bool,
    /// Per-choice log-probabilities supplied by the caller.
    pub per_choice: Vec<f32>,
}

/// Logit-based multiple-choice evaluator.
///
/// Unlike [`McEvaluator`] which parses a completion string, this evaluator
/// takes caller-supplied per-choice log-probabilities (typically the sum of
/// token log-probs for each candidate continuation) and picks the argmax.
///
/// The `prompt_template` is purely descriptive — it is stored so higher-level
/// harnesses can record which prompt produced the scores but is not used for
/// scoring itself.
#[derive(Debug, Clone)]
pub struct McLogitEvaluator {
    /// Prompt template that was used to generate the per-choice log-probs.
    pub prompt_template: String,
}

impl Default for McLogitEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl McLogitEvaluator {
    /// Construct a new evaluator with the standard four-choice template.
    pub fn new() -> Self {
        Self {
            prompt_template: "{question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:".to_string(),
        }
    }

    /// Construct with a custom template.
    pub fn with_template(template: &str) -> Self {
        Self {
            prompt_template: template.to_string(),
        }
    }

    /// Score one question given per-choice log-probabilities.
    ///
    /// `per_choice[i]` is the total log-probability for choice `i`. The picked
    /// choice is the argmax (ties → lowest index, `NaN` entries are never
    /// picked). An empty slice, or a slice whose entries are all `NaN` (no
    /// comparable value to base a pick on), returns a result with
    /// `picked = 0` and `correct = false` unconditionally — never routed
    /// through the argmax fallback, which would otherwise silently credit
    /// index 0 as "correct" whenever `correct_answer == 0`. Callers should
    /// still pre-validate that the slice is non-empty for meaningful scoring.
    pub fn score(&self, per_choice: &[f32], correct_answer: usize) -> LogitMcResult {
        let mut best: Option<(usize, f32)> = None;
        for (i, &v) in per_choice.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            match best {
                Some((_, best_val)) if v <= best_val => {}
                _ => best = Some((i, v)),
            }
        }
        match best {
            Some((best_idx, _)) => LogitMcResult {
                picked: best_idx,
                correct: best_idx == correct_answer,
                per_choice: per_choice.to_vec(),
            },
            None => LogitMcResult {
                picked: 0,
                correct: false,
                per_choice: per_choice.to_vec(),
            },
        }
    }

    /// Evaluate an entire [`McDataset`] given per-question log-probability slates.
    ///
    /// `per_question[i]` must have exactly one log-prob per choice of
    /// `dataset.questions[i]`. Mismatched slate shapes cause that question to
    /// be scored as incorrect.
    pub fn evaluate_dataset(
        &self,
        dataset: &McDataset,
        per_question: &[Vec<f32>],
    ) -> AccuracyResult {
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut by_subject_counts: HashMap<String, (usize, usize)> = HashMap::new();

        for (q, slate) in dataset.questions.iter().zip(per_question.iter()) {
            total += 1;
            let out = self.score(slate, q.correct_answer);
            if out.correct {
                correct += 1;
            }
            if let Some(ref subj) = q.subject {
                let entry = by_subject_counts.entry(subj.clone()).or_insert((0, 0));
                entry.1 += 1;
                if out.correct {
                    entry.0 += 1;
                }
            }
        }

        let accuracy = if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        };
        let by_subject = by_subject_counts
            .into_iter()
            .map(|(s, (c, t))| (s, if t == 0 { 0.0 } else { c as f32 / t as f32 }))
            .collect();

        AccuracyResult {
            correct,
            total,
            accuracy,
            by_subject,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ExactMatchEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator that scores completions by exact string match.
pub struct ExactMatchEvaluator {
    /// When `true`, both strings are lowercased and stripped before comparison.
    pub normalize: bool,
    /// When `true`, the expected output must be a substring of the completion.
    pub partial_match: bool,
}

impl Default for ExactMatchEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactMatchEvaluator {
    /// Create an evaluator that does case-sensitive exact matching without normalisation.
    pub fn new() -> Self {
        Self {
            normalize: false,
            partial_match: false,
        }
    }

    /// Score a single (completion, expected) pair.
    pub fn score(&self, completion: &str, expected: &str) -> bool {
        let (c, e) = if self.normalize {
            (
                completion.trim().to_lowercase(),
                expected.trim().to_lowercase(),
            )
        } else {
            (completion.to_string(), expected.to_string())
        };

        if self.partial_match {
            c.contains(e.as_str())
        } else {
            c == e
        }
    }

    /// Evaluate over a full dataset.
    ///
    /// `completions` must parallel `dataset.examples`. Only examples with a
    /// non-`None` `expected_output` are scored; the rest are skipped.
    pub fn evaluate_dataset(
        &self,
        dataset: &EvalDataset,
        completions: &[String],
    ) -> AccuracyResult {
        let mut correct = 0usize;
        let mut total = 0usize;

        for (ex, completion) in dataset.examples.iter().zip(completions.iter()) {
            if let Some(ref expected) = ex.expected_output {
                total += 1;
                if self.score(completion, expected) {
                    correct += 1;
                }
            }
        }

        let accuracy = if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        };

        AccuracyResult {
            correct,
            total,
            accuracy,
            by_subject: HashMap::new(),
        }
    }
}
