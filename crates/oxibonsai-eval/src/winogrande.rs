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

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde_json::Value;

use crate::accuracy::{AccuracyResult, McEvaluator, McLogitEvaluator};
use crate::dataset::{McDataset, MultipleChoiceQuestion};
use crate::error::EvalError;

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

    /// Parse a WinoGrande JSONL file from disk.
    ///
    /// Each line must be a JSON object with the fields `"sentence"`,
    /// `"option1"`, `"option2"` (all strings), and `"answer"` (the gold
    /// answer, `1` or `2`, either as a JSON integer or as a string — the
    /// upstream HuggingFace `winogrande` dataset ships it as a string).
    ///
    /// Mirrors [`crate::hellaswag::HellaSwagDataset::from_jsonl`]'s
    /// tolerant-type-parsing style so the CLI (RAG-EVAL-IMG-16) can wire
    /// WinoGrande to a real model exactly like the other file-backed
    /// datasets in this crate.
    ///
    /// Returns [`EvalError::Io`] on I/O failures and [`EvalError::ParseError`]
    /// on malformed lines (missing/mistyped fields, or an `answer` outside
    /// `{1, 2}`).
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

            let v: Value = serde_json::from_str(trimmed).map_err(|e| {
                EvalError::ParseError(format!("winogrande: line {}: {}", line_no + 1, e))
            })?;
            let obj = v.as_object().ok_or_else(|| {
                EvalError::ParseError(format!(
                    "winogrande: line {}: not a JSON object",
                    line_no + 1
                ))
            })?;

            let field_str = |name: &str| -> Result<String, EvalError> {
                obj.get(name)
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| {
                        EvalError::ParseError(format!(
                            "winogrande: line {}: missing or invalid \"{name}\" field",
                            line_no + 1
                        ))
                    })
            };

            let sentence = field_str("sentence")?;
            let option1 = field_str("option1")?;
            let option2 = field_str("option2")?;

            // `answer` may be a JSON integer or a numeric string ("1"/"2"),
            // matching the upstream HuggingFace `winogrande` JSONL export.
            let answer: u8 = match obj.get("answer") {
                Some(Value::Number(n)) => {
                    let raw = n.as_u64().ok_or_else(|| {
                        EvalError::ParseError(format!(
                            "winogrande: line {}: \"answer\" is not a non-negative integer",
                            line_no + 1
                        ))
                    })?;
                    // `as u8` would silently wrap an out-of-range value
                    // (e.g. 258 -> 2), which could then pass the `== 1 ||
                    // == 2` range check below with a WRONG gold label
                    // instead of being rejected. `u8::try_from` errors
                    // instead of wrapping.
                    u8::try_from(raw).map_err(|_| {
                        EvalError::ParseError(format!(
                            "winogrande: line {}: \"answer\" {raw} is out of range for a 1/2 \
                             label",
                            line_no + 1
                        ))
                    })?
                }
                Some(Value::String(s)) => s.trim().parse::<u8>().map_err(|e| {
                    EvalError::ParseError(format!(
                        "winogrande: line {}: cannot parse string \"answer\": {e}",
                        line_no + 1
                    ))
                })?,
                _ => {
                    return Err(EvalError::ParseError(format!(
                        "winogrande: line {}: missing or invalid \"answer\" field",
                        line_no + 1
                    )))
                }
            };
            if answer != 1 && answer != 2 {
                return Err(EvalError::ParseError(format!(
                    "winogrande: line {}: \"answer\" must be 1 or 2, got {answer}",
                    line_no + 1
                )));
            }

            items.push(WinoGrandeItem {
                sentence,
                option1,
                option2,
                answer,
            });
        }

        Ok(Self { items })
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
    ///
    /// Routed through [`McDataset::try_add`] rather than the permissive
    /// `add` (RAG-M1): `choices` is always this fixed two-element vector
    /// and `correct_answer` is always 0 or 1, so no `WinoGrandeItem` can
    /// currently produce a rejected question -- this is defensive
    /// hardening (matching the equivalent HellaSwag conversion) against a
    /// future change to this mapping, logged to stderr like that sibling
    /// rather than silently dropped, should it ever trigger.
    pub fn as_mc_dataset(&self) -> McDataset {
        let mut mc = McDataset::new("winogrande");
        let mut skipped = 0usize;
        for (i, item) in self.items.iter().enumerate() {
            // answer field uses 1-based indexing; map to 0-based for McDataset.
            let correct_answer: usize = if item.answer == 1 { 0 } else { 1 };
            let id = format!("winogrande-{i}");
            let question = MultipleChoiceQuestion {
                id: id.clone(),
                question: item.sentence.clone(),
                choices: vec![item.option1.clone(), item.option2.clone()],
                correct_answer,
                subject: None,
                difficulty: None,
            };
            if let Err(e) = mc.try_add(question) {
                skipped += 1;
                tracing::warn!(id = %id, error = %e, "winogrande: skipping malformed item");
            }
        }
        if skipped > 0 {
            tracing::warn!(
                skipped,
                total = self.items.len(),
                "winogrande: skipped malformed item(s) during McDataset conversion (RAG-M1)"
            );
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

    // ── RAG-M1: as_mc_dataset must go through try_add, not the permissive
    // add ─────────────────────────────────────────────────────────────────

    #[test]
    fn malformed_mc_question_is_rejected_by_try_add_not_silently_added() {
        // `WinoGrandeDataset::as_mc_dataset` cannot itself construct a
        // malformed `MultipleChoiceQuestion` today: `choices` is always the
        // fixed two-element `[option1, option2]` and `correct_answer` is
        // always 0 or 1 via the `answer == 1` branch, so it can never
        // violate `try_add`'s `choices.is_empty()` / out-of-range checks
        // (RAG-M1). This is therefore a *mechanism* test, not a
        // reproduction of a currently-reachable defect: it pins the
        // rejection path the conversion now depends on, so that if a
        // future change (e.g. a variable-length options list) ever lets a
        // malformed item reach it, `try_add` -- not the permissive
        // `McDataset::add` -- is what stands between it and a
        // silently-always-wrong question.
        let mut mc = McDataset::new("winogrande");
        let malformed = MultipleChoiceQuestion {
            id: "winogrande-bad".to_string(),
            question: "S".to_string(),
            choices: vec![],
            correct_answer: 0,
            subject: None,
            difficulty: None,
        };
        assert!(
            mc.try_add(malformed).is_err(),
            "an empty-choices question must be rejected"
        );
        assert!(mc.questions.is_empty());
    }

    #[test]
    fn as_mc_dataset_still_converts_well_formed_items_after_try_add_switch() {
        // Regression guard: the switch to `try_add` must not reject valid
        // WinoGrande items (both existing tests above already cover this
        // implicitly by indexing `mc.questions[0]`; this makes the "still
        // converts everything" property explicit for a multi-item batch).
        let ds = WinoGrandeDataset::from_items(vec![
            WinoGrandeItem {
                sentence: "A".into(),
                option1: "X".into(),
                option2: "Y".into(),
                answer: 1,
            },
            WinoGrandeItem {
                sentence: "B".into(),
                option1: "X".into(),
                option2: "Y".into(),
                answer: 2,
            },
        ]);
        let mc = ds.as_mc_dataset();
        assert_eq!(mc.questions.len(), 2);
    }

    // ── from_jsonl (RAG-EVAL-IMG-16: file-backed loader so the CLI can
    // wire WinoGrande to a real model like every other dataset here) ──────

    fn write_temp_jsonl(tag: &str, contents: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "oxibonsai_winogrande_test_{tag}_{}_{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, contents).expect("write temp jsonl");
        path
    }

    #[test]
    fn from_jsonl_parses_string_and_integer_answers() {
        let jsonl = concat!(
            "{\"sentence\": \"A because B.\", \"option1\": \"A\", \"option2\": \"B\", \"answer\": \"1\"}\n",
            "{\"sentence\": \"C because D.\", \"option1\": \"C\", \"option2\": \"D\", \"answer\": 2}\n",
        );
        let path = write_temp_jsonl("mixed_answer_types", jsonl);
        let dataset = WinoGrandeDataset::from_jsonl(&path).expect("valid jsonl must parse");
        let _ = std::fs::remove_file(&path);

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.items[0].answer, 1);
        assert_eq!(dataset.items[1].answer, 2);
        assert_eq!(dataset.items[0].option1, "A");
        assert_eq!(dataset.items[1].option2, "D");
    }

    #[test]
    fn from_jsonl_skips_blank_lines() {
        let jsonl =
            "\n{\"sentence\": \"S\", \"option1\": \"X\", \"option2\": \"Y\", \"answer\": 1}\n\n";
        let path = write_temp_jsonl("blank_lines", jsonl);
        let dataset = WinoGrandeDataset::from_jsonl(&path).expect("valid jsonl must parse");
        let _ = std::fs::remove_file(&path);
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn from_jsonl_rejects_out_of_range_answer() {
        let jsonl =
            "{\"sentence\": \"S\", \"option1\": \"X\", \"option2\": \"Y\", \"answer\": 3}\n";
        let path = write_temp_jsonl("bad_answer", jsonl);
        let result = WinoGrandeDataset::from_jsonl(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "answer must be 1 or 2");
    }

    #[test]
    fn from_jsonl_rejects_answer_that_would_wrap_to_a_valid_label_as_u8() {
        // `258 as u8` wraps to `2`, which would then silently PASS the
        // `answer == 1 || answer == 2` range check with a wrong gold
        // label instead of being rejected. `u8::try_from` must catch this
        // before that check ever runs.
        let jsonl =
            "{\"sentence\": \"S\", \"option1\": \"X\", \"option2\": \"Y\", \"answer\": 258}\n";
        let path = write_temp_jsonl("wrapping_answer", jsonl);
        let result = WinoGrandeDataset::from_jsonl(&path);
        let _ = std::fs::remove_file(&path);
        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("258"),
                    "error should name the real out-of-range value; got: {msg}"
                );
            }
            Ok(_) => panic!("258 must not silently wrap to a valid 1/2 label"),
        }
    }

    #[test]
    fn from_jsonl_rejects_missing_field() {
        let jsonl = "{\"sentence\": \"S\", \"option1\": \"X\", \"answer\": 1}\n";
        let path = write_temp_jsonl("missing_field", jsonl);
        let result = WinoGrandeDataset::from_jsonl(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "missing option2 must be rejected");
    }

    #[test]
    fn from_jsonl_reports_io_error_for_missing_file() {
        let path = std::env::temp_dir().join("oxibonsai_winogrande_does_not_exist.jsonl");
        let result = WinoGrandeDataset::from_jsonl(&path);
        assert!(matches!(result, Err(EvalError::Io(_))));
    }
}
