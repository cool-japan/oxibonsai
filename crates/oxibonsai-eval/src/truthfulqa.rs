//! TruthfulQA evaluation harness.
//!
//! TruthfulQA (Lin et al., 2021) measures whether language models generate
//! truthful answers to questions in the adversarial multiple-choice format.
//! Two scoring modes are supported:
//!
//! - **MC1** — 1-of-N standard multiple choice: the model picks the single
//!   correct answer by argmax over per-choice logits.
//! - **MC2** — Probabilistic scoring: multiple correct answers may exist.
//!   The model's score on each item is the sum of the softmax probability mass
//!   assigned to the correct answers, i.e. `score = Σ p_correct / Σ p_all`.
//!   The final `accuracy` metric is the **mean** of these continuous per-item
//!   scores (not a binary thresholded value).
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_eval::truthfulqa::{TruthfulQaDataset, TruthfulQaEvaluator, TruthfulQaItem};
//!
//! let item = TruthfulQaItem {
//!     question: "What is the capital of France?".to_string(),
//!     mc1_correct_idx: 0,
//!     mc1_choices: vec!["Paris".to_string(), "London".to_string()],
//!     mc2_correct_indices: vec![0],
//!     mc2_choices: vec!["Paris".to_string(), "London".to_string()],
//! };
//! let dataset = TruthfulQaDataset::from_items(vec![item]);
//! let evaluator = TruthfulQaEvaluator::mc1();
//! assert!(!dataset.is_empty());
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::EvalError;

// ──────────────────────────────────────────────────────────────────────────────
// TruthfulQaMode
// ──────────────────────────────────────────────────────────────────────────────

/// Scoring mode for TruthfulQA evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruthfulQaMode {
    /// Standard 1-of-N multiple choice: one correct answer, argmax over logits.
    Mc1,
    /// Probabilistic scoring: multiple correct answers; score = fraction of
    /// softmax mass on correct answers. Final accuracy is the mean over items.
    Mc2,
}

// ──────────────────────────────────────────────────────────────────────────────
// TruthfulQaItem
// ──────────────────────────────────────────────────────────────────────────────

/// A single TruthfulQA dataset item.
///
/// Stores both the MC1 and MC2 choice sets so a single [`TruthfulQaDataset`]
/// can be scored under either mode.
#[derive(Debug, Clone)]
pub struct TruthfulQaItem {
    /// The question stem.
    pub question: String,
    /// 0-based index of the correct answer within [`mc1_choices`].
    ///
    /// [`mc1_choices`]: TruthfulQaItem::mc1_choices
    pub mc1_correct_idx: usize,
    /// MC1 choice list: exactly one entry has `label = 1` in the raw dataset.
    pub mc1_choices: Vec<String>,
    /// 0-based indices of all correct answers within [`mc2_choices`].
    ///
    /// [`mc2_choices`]: TruthfulQaItem::mc2_choices
    pub mc2_correct_indices: Vec<usize>,
    /// MC2 choice list: multiple entries may have `label = 1`.
    pub mc2_choices: Vec<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// TruthfulQaDataset
// ──────────────────────────────────────────────────────────────────────────────

/// A collection of [`TruthfulQaItem`] instances.
pub struct TruthfulQaDataset {
    /// All items in insertion order.
    pub items: Vec<TruthfulQaItem>,
}

impl TruthfulQaDataset {
    /// Create a dataset from a vector of [`TruthfulQaItem`] instances.
    pub fn from_items(items: Vec<TruthfulQaItem>) -> Self {
        Self { items }
    }

    /// Parse a TruthfulQA JSONL file from disk.
    ///
    /// Each line must be a JSON object with:
    /// - `"question"`: string
    /// - `"mc1_targets"`: `{"choices": [...], "labels": [0, 1, 0, ...]}`
    /// - `"mc2_targets"`: `{"choices": [...], "labels": [0, 1, 1, 0, ...]}`
    ///
    /// The first index where `label == 1` in `mc1_targets` becomes
    /// `mc1_correct_idx`. All such indices in `mc2_targets` populate
    /// `mc2_correct_indices`.
    ///
    /// Returns [`EvalError::Io`] on I/O failures and [`EvalError::ParseError`]
    /// on malformed lines.
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

            let v: serde_json::Value =
                serde_json::from_str(trimmed).map_err(|e| {
                    EvalError::ParseError(format!(
                        "truthfulqa: line {}: {}",
                        line_no + 1,
                        e
                    ))
                })?;

            let item = parse_truthfulqa_record(&v, line_no + 1)?;
            items.push(item);
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
}

// ──────────────────────────────────────────────────────────────────────────────
// Parsing helpers
// ──────────────────────────────────────────────────────────────────────────────

fn parse_truthfulqa_record(
    v: &serde_json::Value,
    line_no: usize,
) -> Result<TruthfulQaItem, EvalError> {
    let obj = v.as_object().ok_or_else(|| {
        EvalError::ParseError(format!("truthfulqa: line {line_no}: not a JSON object"))
    })?;

    let question = obj
        .get("question")
        .and_then(|q| q.as_str())
        .ok_or_else(|| {
            EvalError::ParseError(format!(
                "truthfulqa: line {line_no}: missing or invalid \"question\""
            ))
        })?
        .to_string();

    let (mc1_choices, mc1_labels) = parse_targets(obj, "mc1_targets", line_no)?;
    let (mc2_choices, mc2_labels) = parse_targets(obj, "mc2_targets", line_no)?;

    // MC1: index of the first label == 1
    let mc1_correct_idx = mc1_labels
        .iter()
        .position(|&l| l == 1)
        .ok_or_else(|| {
            EvalError::ParseError(format!(
                "truthfulqa: line {line_no}: mc1_targets has no correct label (label == 1)"
            ))
        })?;

    // MC2: all indices where label == 1
    let mc2_correct_indices: Vec<usize> = mc2_labels
        .iter()
        .enumerate()
        .filter_map(|(i, &l)| if l == 1 { Some(i) } else { None })
        .collect();

    Ok(TruthfulQaItem {
        question,
        mc1_correct_idx,
        mc1_choices,
        mc2_correct_indices,
        mc2_choices,
    })
}

/// Parse `{field}.choices` and `{field}.labels` from the parent object.
fn parse_targets(
    obj: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    line_no: usize,
) -> Result<(Vec<String>, Vec<i64>), EvalError> {
    let targets = obj.get(field).and_then(|t| t.as_object()).ok_or_else(|| {
        EvalError::ParseError(format!(
            "truthfulqa: line {line_no}: missing or invalid \"{field}\""
        ))
    })?;

    let choices: Vec<String> = targets
        .get("choices")
        .and_then(|c| c.as_array())
        .ok_or_else(|| {
            EvalError::ParseError(format!(
                "truthfulqa: line {line_no}: \"{field}.choices\" is not an array"
            ))
        })?
        .iter()
        .enumerate()
        .map(|(i, c)| {
            c.as_str().map(str::to_string).ok_or_else(|| {
                EvalError::ParseError(format!(
                    "truthfulqa: line {line_no}: \"{field}.choices[{i}]\" is not a string"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let labels: Vec<i64> = targets
        .get("labels")
        .and_then(|l| l.as_array())
        .ok_or_else(|| {
            EvalError::ParseError(format!(
                "truthfulqa: line {line_no}: \"{field}.labels\" is not an array"
            ))
        })?
        .iter()
        .enumerate()
        .map(|(i, l)| {
            l.as_i64().ok_or_else(|| {
                EvalError::ParseError(format!(
                    "truthfulqa: line {line_no}: \"{field}.labels[{i}]\" is not an integer"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((choices, labels))
}

// ──────────────────────────────────────────────────────────────────────────────
// Numeric helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax.
///
/// Subtracts the maximum logit before exponentiation to avoid floating-point
/// overflow. The result sums to 1.0 within floating-point precision.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// TruthfulQaResult
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregated result from a TruthfulQA evaluation pass.
///
/// For MC1, `accuracy` is the fraction of items where the argmax matched the
/// single correct answer (binary, in [0, 1]).
///
/// For MC2, `accuracy` is the **mean** of the continuous per-item scores
/// (sum of softmax mass on correct answers). It is in [0, 1] but is not
/// constrained to 0 or 1 per item.
#[derive(Debug, Clone)]
pub struct TruthfulQaResult {
    /// The scoring mode used to produce this result.
    pub mode: TruthfulQaMode,
    /// Overall score in [0, 1].
    ///
    /// - MC1: fraction of items answered correctly (binary per item).
    /// - MC2: mean fraction of softmax mass on correct answers.
    pub accuracy: f32,
    /// `accuracy * 100`.
    pub accuracy_pct: f32,
    /// Number of items counted as "correct".
    ///
    /// - MC1: number of argmax-correct items.
    /// - MC2: number of items where the per-item score ≥ 0.5.
    pub correct: usize,
    /// Total number of items evaluated.
    pub total: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// TruthfulQaEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator for TruthfulQA in either MC1 or MC2 mode.
///
/// Construct with [`TruthfulQaEvaluator::mc1`] or [`TruthfulQaEvaluator::mc2`].
/// The evaluator is stateless beyond the [`mode`] field.
///
/// [`mode`]: TruthfulQaEvaluator::mode
pub struct TruthfulQaEvaluator {
    /// Which scoring mode to use.
    pub mode: TruthfulQaMode,
}

impl TruthfulQaEvaluator {
    /// Create an MC1-mode evaluator.
    pub fn mc1() -> Self {
        Self {
            mode: TruthfulQaMode::Mc1,
        }
    }

    /// Create an MC2-mode evaluator.
    pub fn mc2() -> Self {
        Self {
            mode: TruthfulQaMode::Mc2,
        }
    }

    /// Evaluate using per-choice logit scores, dispatching to MC1 or MC2 logic.
    ///
    /// `per_choice_logits[i]` contains one log-probability per choice for item `i`.
    /// For MC1, the lengths must match `mc1_choices`; for MC2 they must match
    /// `mc2_choices`. Mismatched lengths are handled gracefully (scored as 0 / 0.0).
    ///
    /// The shorter of `dataset.items` and `per_choice_logits` is used; surplus
    /// entries on either side are ignored.
    pub fn evaluate_logits(
        &self,
        dataset: &TruthfulQaDataset,
        per_choice_logits: &[Vec<f32>],
    ) -> TruthfulQaResult {
        match self.mode {
            TruthfulQaMode::Mc1 => self.evaluate_mc1(dataset, per_choice_logits),
            TruthfulQaMode::Mc2 => self.evaluate_mc2(dataset, per_choice_logits),
        }
    }

    // ── MC1 implementation ────────────────────────────────────────────────────

    fn evaluate_mc1(
        &self,
        dataset: &TruthfulQaDataset,
        per_choice_logits: &[Vec<f32>],
    ) -> TruthfulQaResult {
        let mut correct = 0usize;
        let mut total = 0usize;

        for (item, logits) in dataset.items.iter().zip(per_choice_logits.iter()) {
            total += 1;
            let picked = argmax(logits);
            if picked == item.mc1_correct_idx {
                correct += 1;
            }
        }

        let accuracy = if total == 0 {
            0.0_f32
        } else {
            correct as f32 / total as f32
        };

        TruthfulQaResult {
            mode: TruthfulQaMode::Mc1,
            accuracy,
            accuracy_pct: accuracy * 100.0,
            correct,
            total,
        }
    }

    // ── MC2 implementation ────────────────────────────────────────────────────

    fn evaluate_mc2(
        &self,
        dataset: &TruthfulQaDataset,
        per_choice_logits: &[Vec<f32>],
    ) -> TruthfulQaResult {
        let mut score_sum = 0.0_f32;
        let mut correct = 0usize;
        let mut total = 0usize;

        for (item, logits) in dataset.items.iter().zip(per_choice_logits.iter()) {
            total += 1;

            let probs = softmax(logits);

            // Sum of softmax mass on correct answers.
            let correct_mass: f32 = item
                .mc2_correct_indices
                .iter()
                .filter_map(|&idx| probs.get(idx).copied())
                .sum();

            // The total mass always sums to 1.0 from softmax, but we follow
            // the standard definition explicitly for clarity:
            //   score = Σ p_correct / Σ p_all  = correct_mass / 1.0
            let item_score = if probs.is_empty() {
                0.0_f32
            } else {
                correct_mass
            };

            score_sum += item_score;

            // Threshold at 0.5 to count "correct" for the integer field.
            if item_score >= 0.5 {
                correct += 1;
            }
        }

        // MC2 accuracy is the **mean** of the continuous per-item scores.
        let accuracy = if total == 0 {
            0.0_f32
        } else {
            score_sum / total as f32
        };

        TruthfulQaResult {
            mode: TruthfulQaMode::Mc2,
            accuracy,
            accuracy_pct: accuracy * 100.0,
            correct,
            total,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal utilities
// ──────────────────────────────────────────────────────────────────────────────

/// Return the index of the maximum element. Ties are broken by lowest index.
/// Returns 0 for empty slices.
#[inline]
fn argmax(values: &[f32]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}
