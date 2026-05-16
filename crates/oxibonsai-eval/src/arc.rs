//! ARC (AI2 Reasoning Challenge) evaluation harness.
//!
//! ARC-Easy and ARC-Challenge are four-to-five option multiple-choice benchmarks
//! from Clark et al. (2018). Both splits use the same [`McDataset`] structure
//! already in this crate; this module provides thin delegation wrappers that
//! carry the split identity and produce a strongly-typed [`ArcResult`].
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_eval::arc::{ArcEvaluator, ArcSplit};
//! use oxibonsai_eval::dataset::McDataset;
//!
//! let evaluator = ArcEvaluator::easy();
//! assert_eq!(evaluator.split(), ArcSplit::Easy);
//! ```

use crate::accuracy::{AccuracyResult, McEvaluator, McLogitEvaluator};
use crate::dataset::McDataset;

// ──────────────────────────────────────────────────────────────────────────────
// ArcSplit
// ──────────────────────────────────────────────────────────────────────────────

/// Which partition of the ARC benchmark this evaluator targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArcSplit {
    /// ARC-Easy: questions that the majority of retrieval and word-co-occurrence
    /// algorithms could answer correctly.
    Easy,
    /// ARC-Challenge: questions answered incorrectly by both a retrieval-based
    /// algorithm and a word co-occurrence algorithm.
    Challenge,
}

impl ArcSplit {
    /// The canonical benchmark name string for this split.
    pub fn name(&self) -> &'static str {
        match self {
            ArcSplit::Easy => "ARC-Easy",
            ArcSplit::Challenge => "ARC-Challenge",
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ArcEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator for ARC-Easy or ARC-Challenge benchmarks.
///
/// Delegates to [`McEvaluator`] (completion-based) and [`McLogitEvaluator`]
/// (logit-based) — ARC is structurally identical to MMLU multiple-choice.
///
/// ARC questions may have 4 **or** 5 options (keyed A-E). The underlying
/// evaluators already handle variable-length choice vectors gracefully.
pub struct ArcEvaluator {
    /// Which split this evaluator represents.
    split: ArcSplit,
    /// String-completion-based evaluator (delegates to [`McEvaluator`]).
    mc: McEvaluator,
    /// Logit-based evaluator (delegates to [`McLogitEvaluator`]).
    mc_logit: McLogitEvaluator,
}

impl ArcEvaluator {
    /// Construct an evaluator for the ARC-Easy split.
    pub fn easy() -> Self {
        Self::new(ArcSplit::Easy)
    }

    /// Construct an evaluator for the ARC-Challenge split.
    pub fn challenge() -> Self {
        Self::new(ArcSplit::Challenge)
    }

    /// Internal constructor shared by [`easy`](Self::easy) and
    /// [`challenge`](Self::challenge).
    fn new(split: ArcSplit) -> Self {
        // ARC questions typically present 4 or 5 options labelled A-E.
        // The template uses {a}..{d}; a fifth choice would be at index 4 but
        // McEvaluator::extract_answer also handles 'E' if we extend the template.
        // For now, use the same standard template as MMLU — callers that need
        // five-option support can inject a custom template via McEvaluator::with_template.
        let template = "{question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:".to_string();

        Self {
            split,
            mc: McEvaluator {
                prompt_template: template.clone(),
            },
            mc_logit: McLogitEvaluator {
                prompt_template: template,
            },
        }
    }

    /// Which split this evaluator represents.
    pub fn split(&self) -> ArcSplit {
        self.split
    }

    /// Evaluate by comparing model completions to answer letter choices.
    ///
    /// `completions` must be the same length as `dataset.questions`; surplus or
    /// missing entries are handled by zipping (shortest wins). Each completion is
    /// expected to begin with the letter of the chosen answer (A, B, C, D, or E).
    pub fn evaluate_completions(
        &self,
        dataset: &McDataset,
        completions: &[String],
    ) -> AccuracyResult {
        self.mc.evaluate_dataset(dataset, completions)
    }

    /// Evaluate by selecting the choice with the highest per-choice log-probability.
    ///
    /// `per_choice_logits[i]` is a vector of log-probabilities — one per answer
    /// option of `dataset.questions[i]`. The evaluator picks the argmax.
    pub fn evaluate_logits(
        &self,
        dataset: &McDataset,
        per_choice_logits: &[Vec<f32>],
    ) -> AccuracyResult {
        self.mc_logit.evaluate_dataset(dataset, per_choice_logits)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ArcResult
// ──────────────────────────────────────────────────────────────────────────────

/// Strongly-typed result from an ARC evaluation run.
///
/// Mirrors the fields of [`AccuracyResult`] while attaching the ARC split
/// identity so that results from different splits remain unambiguous.
#[derive(Debug, Clone)]
pub struct ArcResult {
    /// Which split produced this result.
    pub split: ArcSplit,
    /// Accuracy as a fraction in \[0, 1\].
    pub accuracy: f32,
    /// Number of correctly answered questions.
    pub correct: usize,
    /// Total number of questions evaluated.
    pub total: usize,
}

impl ArcResult {
    /// Build an [`ArcResult`] from a generic [`AccuracyResult`] and the split identity.
    pub fn from_accuracy_result(split: ArcSplit, result: AccuracyResult) -> Self {
        Self {
            split,
            accuracy: result.accuracy,
            correct: result.correct,
            total: result.total,
        }
    }

    /// Return accuracy as a percentage in \[0, 100\].
    pub fn accuracy_pct(&self) -> f32 {
        self.accuracy * 100.0
    }

    /// The canonical benchmark name of the split ("ARC-Easy" or "ARC-Challenge").
    pub fn split_name(&self) -> &'static str {
        self.split.name()
    }
}
