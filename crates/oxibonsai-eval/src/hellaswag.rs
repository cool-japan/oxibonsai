//! HellaSwag sentence-completion evaluation harness.
//!
//! HellaSwag (Zellers et al., 2019) tests grounded commonsense inference via
//! 4-way sentence completion. Each item presents an activity label, a context
//! sentence, and exactly four candidate endings. The task is to select the most
//! plausible continuation.
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_eval::hellaswag::{HellaSwagDataset, HellaSwagEvaluator, HellaSwagItem};
//!
//! let items = vec![HellaSwagItem {
//!     id: "0".to_string(),
//!     activity_label: "Cooking".to_string(),
//!     ctx: "She put the pasta into boiling water.".to_string(),
//!     endings: vec![
//!         "She stirred occasionally.".to_string(),
//!         "She went to sleep.".to_string(),
//!         "She threw the pot away.".to_string(),
//!         "She ate the raw pasta.".to_string(),
//!     ],
//!     label: 0,
//! }];
//! let dataset = HellaSwagDataset::from_items(items);
//! let evaluator = HellaSwagEvaluator::new();
//! assert!(!dataset.is_empty());
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::accuracy::{AccuracyResult, McEvaluator, McLogitEvaluator};
use crate::dataset::{McDataset, MultipleChoiceQuestion};
use crate::error::EvalError;

// ──────────────────────────────────────────────────────────────────────────────
// Raw JSON shape for JSONL parsing
// ──────────────────────────────────────────────────────────────────────────────

/// Mirrors the on-disk JSONL structure for HellaSwag.
#[derive(Debug, Deserialize)]
struct HellaSwagRecord {
    ind: Value,
    activity_label: String,
    ctx: String,
    endings: Vec<String>,
    label: Value,
}

// ──────────────────────────────────────────────────────────────────────────────
// HellaSwagItem
// ──────────────────────────────────────────────────────────────────────────────

/// A single HellaSwag dataset item.
///
/// Each item contains an activity label, a context sentence (the stem), and
/// exactly four candidate endings. The `label` field is the 0-based index of the
/// correct ending.
#[derive(Debug, Clone, PartialEq)]
pub struct HellaSwagItem {
    /// Unique identifier for this item (typically a numeric string from the dataset).
    pub id: String,
    /// High-level activity category (e.g. "Grooming and self care").
    pub activity_label: String,
    /// The context / premise sentence shown to the model.
    pub ctx: String,
    /// Exactly four candidate sentence endings.
    pub endings: Vec<String>,
    /// 0-based index of the correct ending.
    pub label: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// HellaSwagDataset
// ──────────────────────────────────────────────────────────────────────────────

/// A collection of [`HellaSwagItem`] instances.
pub struct HellaSwagDataset {
    /// All items in insertion order.
    pub items: Vec<HellaSwagItem>,
}

impl HellaSwagDataset {
    /// Create a dataset from a vector of [`HellaSwagItem`] instances.
    pub fn from_items(items: Vec<HellaSwagItem>) -> Self {
        Self { items }
    }

    /// Parse a HellaSwag JSONL file from disk.
    ///
    /// Each line must be a JSON object with the fields:
    /// `"ind"`, `"activity_label"`, `"ctx"`, `"endings"` (array of 4 strings),
    /// and `"label"` (integer 0–3).
    ///
    /// Returns [`EvalError::Io`] on I/O failures and [`EvalError::ParseError`] on
    /// malformed lines.
    pub fn from_jsonl(path: &Path) -> Result<Self, EvalError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut items = Vec::new();

        for (line_no, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let record: HellaSwagRecord = serde_json::from_str(trimmed).map_err(|e| {
                EvalError::ParseError(format!("hellaswag: line {}: {}", line_no + 1, e))
            })?;

            // `ind` may be a string or an integer in the wild.
            let id = match &record.ind {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                other => {
                    return Err(EvalError::ParseError(format!(
                        "hellaswag: line {}: unexpected type for \"ind\": {}",
                        line_no + 1,
                        other
                    )))
                }
            };

            // `label` may likewise be a string or an integer.
            let label: usize = match &record.label {
                Value::Number(n) => n.as_u64().ok_or_else(|| {
                    EvalError::ParseError(format!(
                        "hellaswag: line {}: \"label\" is not a non-negative integer",
                        line_no + 1
                    ))
                })? as usize,
                Value::String(s) => s.trim().parse::<usize>().map_err(|e| {
                    EvalError::ParseError(format!(
                        "hellaswag: line {}: cannot parse string \"label\": {}",
                        line_no + 1,
                        e
                    ))
                })?,
                other => {
                    return Err(EvalError::ParseError(format!(
                        "hellaswag: line {}: unexpected type for \"label\": {}",
                        line_no + 1,
                        other
                    )))
                }
            };

            items.push(HellaSwagItem {
                id,
                activity_label: record.activity_label,
                ctx: record.ctx,
                endings: record.endings,
                label,
            });
        }

        Ok(Self { items })
    }

    /// Return the number of items in this dataset.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Return `true` if this dataset contains no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Convert to a [`McDataset`] for delegation to [`McEvaluator`] or
    /// [`McLogitEvaluator`].
    ///
    /// Each HellaSwag item becomes a four-choice `MultipleChoiceQuestion` where:
    /// - `question` = `item.ctx`
    /// - `choices`  = `item.endings` (4 elements)
    /// - `correct_answer` = `item.label`
    /// - `subject`  = `item.activity_label`
    ///
    /// A malformed item (empty `endings`, or `label` out of range for them
    /// -- `from_jsonl` performs no such validation, RAG-M1) is skipped
    /// rather than silently turned into a question that can never be
    /// scored correct. This function has no `Result` in its signature and
    /// neither do its callers ([`HellaSwagEvaluator::evaluate_completions`]
    /// / `evaluate_logits`, both returning a plain [`HellaSwagResult`]), so
    /// propagating would ripple a breaking signature change across the
    /// public evaluator API for a condition a real HellaSwag file should
    /// never hit; each skip is logged to stderr (with a final count) so a
    /// malformed dataset is visible instead of silently mis-scored.
    pub fn as_mc_dataset(&self) -> McDataset {
        let mut mc = McDataset::new("hellaswag");
        let mut skipped = 0usize;
        for item in &self.items {
            let question = MultipleChoiceQuestion {
                id: item.id.clone(),
                question: item.ctx.clone(),
                choices: item.endings.clone(),
                correct_answer: item.label,
                subject: Some(item.activity_label.clone()),
                difficulty: None,
            };
            if let Err(e) = mc.try_add(question) {
                skipped += 1;
                tracing::warn!(id = ?item.id, error = %e, "hellaswag: skipping malformed item");
            }
        }
        if skipped > 0 {
            tracing::warn!(
                skipped,
                total = self.items.len(),
                "hellaswag: skipped malformed item(s) during McDataset conversion (RAG-M1)"
            );
        }
        mc
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// HellaSwagResult
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregated result from a HellaSwag evaluation pass.
#[derive(Debug, Clone)]
pub struct HellaSwagResult {
    /// Accuracy as a fraction in [0, 1].
    pub accuracy: f32,
    /// Accuracy as a percentage in [0, 100].
    pub accuracy_pct: f32,
    /// Number of correctly answered items.
    pub correct: usize,
    /// Total number of items evaluated.
    pub total: usize,
}

impl HellaSwagResult {
    /// Build a [`HellaSwagResult`] from a generic [`AccuracyResult`].
    fn from_accuracy(acc: AccuracyResult) -> Self {
        let accuracy = if acc.total == 0 {
            0.0_f32
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
}

// ──────────────────────────────────────────────────────────────────────────────
// HellaSwagEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator for the HellaSwag sentence-completion benchmark.
///
/// Delegates internally to [`McEvaluator`] (string-completion path) and
/// [`McLogitEvaluator`] (logit-based path). HellaSwag is structurally a
/// 4-choice multiple-choice task, so no custom scoring logic is required beyond
/// converting the dataset and forwarding the call.
pub struct HellaSwagEvaluator {
    /// String-completion-based multiple-choice evaluator.
    mc: McEvaluator,
    /// Logit-based multiple-choice evaluator.
    mc_logit: McLogitEvaluator,
}

impl HellaSwagEvaluator {
    /// Create a new evaluator with the standard 4-choice prompt template.
    pub fn new() -> Self {
        let template = "{question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:".to_string();
        Self {
            mc: McEvaluator {
                prompt_template: template.clone(),
            },
            mc_logit: McLogitEvaluator {
                prompt_template: template,
            },
        }
    }

    /// Evaluate using model string completions.
    ///
    /// `completions[i]` must begin with "A", "B", "C", or "D" (case-insensitive)
    /// to be counted as a valid prediction. Mismatches are scored as incorrect.
    ///
    /// `completions` must have the same length as `dataset.items`; the shorter
    /// of the two is used.
    pub fn evaluate_completions(
        &self,
        dataset: &HellaSwagDataset,
        completions: &[String],
    ) -> HellaSwagResult {
        let mc_dataset = dataset.as_mc_dataset();
        let acc = self.mc.evaluate_dataset(&mc_dataset, completions);
        HellaSwagResult::from_accuracy(acc)
    }

    /// Evaluate using per-choice logit scores.
    ///
    /// `per_choice_logits[i]` is a `Vec<f32>` of length 4 where each element is
    /// the (log-)probability assigned to the corresponding ending. The choice with
    /// the highest value is selected (argmax). Ties are broken by lowest index.
    pub fn evaluate_logits(
        &self,
        dataset: &HellaSwagDataset,
        per_choice_logits: &[Vec<f32>],
    ) -> HellaSwagResult {
        let mc_dataset = dataset.as_mc_dataset();
        let acc = self
            .mc_logit
            .evaluate_dataset(&mc_dataset, per_choice_logits);
        HellaSwagResult::from_accuracy(acc)
    }
}

impl Default for HellaSwagEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inline tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RAG-M1: as_mc_dataset must not silently accept a malformed item ────

    #[test]
    fn as_mc_dataset_rejects_an_item_with_no_endings() {
        // `from_jsonl` performs no length/range validation on `endings` or
        // `label` (RAG-M1's own evidence), so a malformed JSONL line (e.g.
        // `{"endings": [], "label": 0}`) reaches here with `endings: vec![]`
        // and `label: 0` -- empty choices AND an out-of-range correct
        // answer. Before this fix `McDataset::add` accepted it
        // unconditionally, producing a question that can never be scored
        // correct; reproduced directly via `from_items` here (bypassing
        // JSONL parsing) so the conversion itself is what's under test.
        let bad = HellaSwagItem {
            id: "bad".to_string(),
            activity_label: "x".to_string(),
            ctx: "y".to_string(),
            endings: vec![],
            label: 0,
        };
        let dataset = HellaSwagDataset::from_items(vec![bad]);
        let mc = dataset.as_mc_dataset();
        assert!(
            mc.questions.is_empty(),
            "a malformed (0 endings) HellaSwag item must be skipped, not silently \
             added as an unscoreable question; got {:?}",
            mc.questions
        );
    }

    #[test]
    fn as_mc_dataset_rejects_an_out_of_range_label_with_nonempty_endings() {
        let bad = HellaSwagItem {
            id: "bad2".to_string(),
            activity_label: "x".to_string(),
            ctx: "y".to_string(),
            endings: vec!["a".to_string(), "b".to_string()],
            label: 5, // out of range for 2 endings
        };
        let dataset = HellaSwagDataset::from_items(vec![bad]);
        let mc = dataset.as_mc_dataset();
        assert!(
            mc.questions.is_empty(),
            "an out-of-range label must be skipped, not silently added; got {:?}",
            mc.questions
        );
    }

    #[test]
    fn as_mc_dataset_still_converts_well_formed_items() {
        // Regression guard: the switch to `try_add` must not reject valid
        // HellaSwag items.
        let item = HellaSwagItem {
            id: "42".to_string(),
            activity_label: "Cooking".to_string(),
            ctx: "She stirred the pot.".to_string(),
            endings: vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
            label: 2,
        };
        let dataset = HellaSwagDataset::from_items(vec![item]);
        let mc = dataset.as_mc_dataset();
        assert_eq!(mc.questions.len(), 1);
        assert_eq!(mc.questions[0].correct_answer, 2);
        assert_eq!(mc.questions[0].choices.len(), 4);
    }
}
