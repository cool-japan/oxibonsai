//! WinoGrande commonsense reasoning evaluation harness.
//!
//! WinoGrande (Sakaguchi et al., 2019) tests commonsense reasoning via binary
//! fill-in-the-blank questions. Each question presents a sentence with a blank,
//! and two plausible options. The task is to select the contextually correct option.
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_eval::winogrande::{WinoGrandeDataset, WinoGrandeEvaluator, WinoGrandeItem};
//!
//! let items = vec![WinoGrandeItem {
//!     sentence: "The trophy doesn't fit in the suitcase because the ___ is too big.".to_string(),
//!     option1: "trophy".to_string(),
//!     option2: "suitcase".to_string(),
//!     answer: 1,
//! }];
//! let dataset = WinoGrandeDataset::from_items(items);
//! let evaluator = WinoGrandeEvaluator::new();
//! assert!(!dataset.is_empty());
//! ```

use crate::accuracy::{AccuracyResult, McEvaluator, McLogitEvaluator};
use crate::dataset::{McDataset, MultipleChoiceQuestion};

// ──────────────────────────────────────────────────────────────────────────────
// WinoGrandeItem
// ──────────────────────────────────────────────────────────────────────────────

/// A single WinoGrande dataset item.
///
/// Each item contains a sentence with a blank slot, two candidate option strings,
/// and the gold answer (1 for option1, 2 for option2).
#[derive(Debug, Clone, PartialEq)]
pub struct WinoGrandeItem {
    /// The sentence with a blank (use "_" or any marker; the blank is filled by one of the options).
    pub sentence: String,
    /// First candidate option (fills the blank when answer == 1).
    pub option1: String,
    /// Second candidate option (fills the blank when answer == 2).
    pub option2: String,
    /// Correct answer: 1 (option1) or 2 (option2).
    pub answer: u8,
}

// ──────────────────────────────────────────────────────────────────────────────
// WinoGrandeDataset
// ──────────────────────────────────────────────────────────────────────────────

/// A collection of [`WinoGrandeItem`] instances.
pub struct WinoGrandeDataset {
    /// All items in insertion order.
    pub items: Vec<WinoGrandeItem>,
}

impl WinoGrandeDataset {
    /// Create a dataset from a vector of items.
    pub fn from_items(items: Vec<WinoGrandeItem>) -> Self {
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

    /// Convert to [`McDataset`] for use with [`McEvaluator`] and [`McLogitEvaluator`].
    ///
    /// Each WinoGrande item becomes a two-choice multiple-choice question:
    /// - Choice index 0 → option1 (letter A).
    /// - Choice index 1 → option2 (letter B).
    /// - `correct_answer` index: 0 if `item.answer == 1`, 1 if `item.answer == 2`.
    pub fn as_mc_dataset(&self) -> McDataset {
        let mut mc = McDataset::new("winogrande");
        for (i, item) in self.items.iter().enumerate() {
            // answer field uses 1-based indexing; map to 0-based for McDataset.
            let correct_answer: usize = if item.answer == 1 { 0 } else { 1 };
            mc.add(MultipleChoiceQuestion {
                id: format!("winogrande-{}", i),
                question: item.sentence.clone(),
                choices: vec![item.option1.clone(), item.option2.clone()],
                correct_answer,
                subject: None,
                difficulty: None,
            });
        }
        mc
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// WinoGrandeResult
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregated result from a WinoGrande evaluation pass.
#[derive(Debug, Clone)]
pub struct WinoGrandeResult {
    /// Fraction of items answered correctly (0.0–1.0).
    pub accuracy: f32,
    /// Accuracy as a percentage (0.0–100.0).
    pub accuracy_pct: f32,
    /// Number of correctly answered items.
    pub correct: usize,
    /// Total number of items evaluated.
    pub total: usize,
}

impl WinoGrandeResult {
    /// Build a [`WinoGrandeResult`] from a generic [`AccuracyResult`].
    fn from_accuracy(acc: AccuracyResult) -> Self {
        let accuracy = if acc.total == 0 {
            0.0
        } else {
            acc.correct as f32 / acc.total as f32
        };
        Self {
            accuracy,
            accuracy_pct: accuracy * 100.0,
            correct: acc.correct,
            total: acc.total,
        }
    }

    /// Return accuracy as a percentage in \[0, 100\].
    pub fn accuracy_pct(&self) -> f32 {
        self.accuracy_pct
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// WinoGrandeEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator for WinoGrande commonsense reasoning.
///
/// Delegates to [`McEvaluator`] (completion-based) and [`McLogitEvaluator`]
/// (logit-based). WinoGrande is structurally a 2-choice multiple-choice task
/// where "A" corresponds to option1 and "B" corresponds to option2.
pub struct WinoGrandeEvaluator {
    /// String-completion-based evaluator.
    mc: McEvaluator,
    /// Logit-based evaluator.
    mc_logit: McLogitEvaluator,
}

impl WinoGrandeEvaluator {
    /// Create a new evaluator with the standard WinoGrande two-choice prompt template.
    pub fn new() -> Self {
        let template = "{question}\nA) {a}\nB) {b}\nAnswer:".to_string();
        Self {
            mc: McEvaluator {
                prompt_template: template.clone(),
            },
            mc_logit: McLogitEvaluator {
                prompt_template: template,
            },
        }
    }

    /// Evaluate string completions against the dataset.
    ///
    /// `completions[i]` is the model's generated answer for `dataset.items[i]`.
    /// Each completion must begin with "A" (→ option1) or "B" (→ option2).
    pub fn evaluate_completions(
        &self,
        dataset: &WinoGrandeDataset,
        completions: &[String],
    ) -> WinoGrandeResult {
        let mc_dataset = dataset.as_mc_dataset();
        let acc = self.mc.evaluate_dataset(&mc_dataset, completions);
        WinoGrandeResult::from_accuracy(acc)
    }

    /// Evaluate using per-choice logit scores.
    ///
    /// `per_choice_logits[i]` is a `Vec<f32>` of length 2 containing
    /// `[logit_A, logit_B]`. Prediction is the argmax of the two logits.
    pub fn evaluate_logits(
        &self,
        dataset: &WinoGrandeDataset,
        per_choice_logits: &[Vec<f32>],
    ) -> WinoGrandeResult {
        let mc_dataset = dataset.as_mc_dataset();
        let acc = self
            .mc_logit
            .evaluate_dataset(&mc_dataset, per_choice_logits);
        WinoGrandeResult::from_accuracy(acc)
    }

    /// Format a prompt string for a single WinoGrande item using the stored template.
    ///
    /// This is useful for constructing prompts to feed into a language model.
    pub fn build_prompt(&self, item: &WinoGrandeItem) -> String {
        self.mc.format_question(&MultipleChoiceQuestion {
            id: String::new(),
            question: item.sentence.clone(),
            choices: vec![item.option1.clone(), item.option2.clone()],
            correct_answer: 0,
            subject: None,
            difficulty: None,
        })
    }
}

impl Default for WinoGrandeEvaluator {
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

    fn make_dataset() -> WinoGrandeDataset {
        WinoGrandeDataset::from_items(vec![
            WinoGrandeItem {
                sentence: "The trophy doesn't fit in the suitcase because the ___ is too large."
                    .to_string(),
                option1: "trophy".to_string(),
                option2: "suitcase".to_string(),
                answer: 1,
            },
            WinoGrandeItem {
                sentence: "The cat sat on the mat because ___ was comfortable.".to_string(),
                option1: "mat".to_string(),
                option2: "cat".to_string(),
                answer: 1,
            },
        ])
    }

    #[test]
    fn winogrande_dataset_len() {
        let ds = make_dataset();
        assert_eq!(ds.len(), 2);
        assert!(!ds.is_empty());
    }

    #[test]
    fn winogrande_empty_dataset() {
        let ds = WinoGrandeDataset::from_items(vec![]);
        assert!(ds.is_empty());
    }

    #[test]
    fn winogrande_as_mc_dataset_correct_answer_index_zero_for_answer1() {
        let item = WinoGrandeItem {
            sentence: "S".into(),
            option1: "X".into(),
            option2: "Y".into(),
            answer: 1,
        };
        let ds = WinoGrandeDataset::from_items(vec![item]);
        let mc = ds.as_mc_dataset();
        // answer==1 → option1 → index 0 → letter A
        assert_eq!(mc.questions[0].correct_answer, 0);
        assert_eq!(mc.questions[0].choices.len(), 2);
    }

    #[test]
    fn winogrande_as_mc_dataset_correct_answer_index_one_for_answer2() {
        let item = WinoGrandeItem {
            sentence: "S".into(),
            option1: "X".into(),
            option2: "Y".into(),
            answer: 2,
        };
        let ds = WinoGrandeDataset::from_items(vec![item]);
        let mc = ds.as_mc_dataset();
        // answer==2 → option2 → index 1 → letter B
        assert_eq!(mc.questions[0].correct_answer, 1);
    }
}
