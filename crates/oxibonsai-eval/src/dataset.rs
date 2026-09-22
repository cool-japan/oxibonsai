//! Dataset types and loaders for the evaluation harness.
//!
//! Provides [`EvalDataset`] for free-form text evaluation and [`McDataset`]
//! for MMLU-style multiple-choice evaluation. Both support JSONL loading
//! and deterministic sampling via a simple LCG (no external rand crate).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::EvalError;

// ──────────────────────────────────────────────────────────────────────────────
// LCG helper (Knuth's multiplicative congruential generator)
// ──────────────────────────────────────────────────────────────────────────────

/// Advance one LCG step and return the new state.
///
/// Parameters from Numerical Recipes:
/// - multiplier = 1664525
/// - increment  = 1013904223
/// - modulus    = 2^32 (implicit via u32 overflow)
#[inline]
fn lcg_step(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ──────────────────────────────────────────────────────────────────────────────
// EvalExample
// ──────────────────────────────────────────────────────────────────────────────

/// A single evaluation example for free-form text tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalExample {
    /// Unique identifier for this example.
    pub id: String,
    /// The input prompt / context fed to the model.
    pub input: String,
    /// Expected output, if known (used for scoring).
    pub expected_output: Option<String>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, Value>,
}

// ──────────────────────────────────────────────────────────────────────────────
// MultipleChoiceQuestion
// ──────────────────────────────────────────────────────────────────────────────

/// A multiple-choice question in MMLU format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleChoiceQuestion {
    /// Unique identifier for this question.
    pub id: String,
    /// The question stem.
    pub question: String,
    /// Answer choices, e.g. `["A: option1", "B: option2", ...]`.
    pub choices: Vec<String>,
    /// Index of the correct choice (0-based).
    pub correct_answer: usize,
    /// Subject area (e.g. "high_school_biology").
    pub subject: Option<String>,
    /// Difficulty label (e.g. "easy", "medium", "hard").
    pub difficulty: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// EvalDataset
// ──────────────────────────────────────────────────────────────────────────────

/// A named collection of [`EvalExample`] instances.
pub struct EvalDataset {
    /// Human-readable dataset name.
    pub name: String,
    /// All examples in insertion order.
    pub examples: Vec<EvalExample>,
}

impl EvalDataset {
    /// Create an empty dataset with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            examples: Vec::new(),
        }
    }

    /// Append an example to the dataset.
    pub fn add(&mut self, example: EvalExample) {
        self.examples.push(example);
    }

    /// Return the number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Return `true` if the dataset contains no examples.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Parse a JSONL string into a dataset.
    ///
    /// Each line must be a JSON object with at least an `"input"` field.
    /// `"id"`, `"expected_output"`, and `"metadata"` are optional.
    pub fn from_jsonl(name: &str, jsonl: &str) -> Result<Self, EvalError> {
        let mut dataset = EvalDataset::new(name);
        for (line_no, line) in jsonl.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(trimmed)
                .map_err(|e| EvalError::ParseError(format!("line {}: {}", line_no + 1, e)))?;
            let obj = v.as_object().ok_or_else(|| {
                EvalError::InvalidFormat(format!("line {} is not a JSON object", line_no + 1))
            })?;

            let input = obj
                .get("input")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    EvalError::InvalidFormat(format!(
                        "line {}: missing \"input\" field",
                        line_no + 1
                    ))
                })?
                .to_string();

            let id = obj
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("{}", line_no));

            let expected_output = obj
                .get("expected_output")
                .and_then(Value::as_str)
                .map(str::to_string);

            let metadata: HashMap<String, Value> = obj
                .get("metadata")
                .and_then(Value::as_object)
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();

            dataset.add(EvalExample {
                id,
                input,
                expected_output,
                metadata,
            });
        }
        Ok(dataset)
    }

    /// Serialise the dataset back to JSONL format (one JSON object per line).
    pub fn to_jsonl(&self) -> String {
        self.examples
            .iter()
            .filter_map(|ex| serde_json::to_string(ex).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Return a deterministic random sample of `n` examples.
    ///
    /// Uses a 64-bit LCG seeded with `seed`. If `n >= self.len()`, returns a
    /// clone of the entire dataset in a shuffled order.
    pub fn sample(&self, n: usize, seed: u64) -> EvalDataset {
        let count = n.min(self.len());
        let mut indices: Vec<usize> = (0..self.len()).collect();

        // Fisher-Yates shuffle driven by LCG
        let mut state = seed;
        for i in (1..indices.len()).rev() {
            state = lcg_step(state);
            let j = (state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let mut sampled = EvalDataset::new(&self.name);
        for &idx in indices.iter().take(count) {
            sampled.add(self.examples[idx].clone());
        }
        sampled
    }

    /// Explicit alias for [`EvalDataset::sample`], surfacing the seeded
    /// nature of the sampler in the name.
    ///
    /// Given identical `(n, seed)` inputs, the returned dataset is bit-identical
    /// across runs and across platforms (LCG constants are fixed).
    pub fn sample_with_seed(&self, n: usize, seed: u64) -> EvalDataset {
        self.sample(n, seed)
    }

    /// Split the dataset into train and test subsets.
    ///
    /// The first `floor(len * train_ratio)` examples become the training set;
    /// the remainder form the test set. Order is preserved.
    pub fn split(&self, train_ratio: f32) -> (EvalDataset, EvalDataset) {
        let split_at = ((self.len() as f32) * train_ratio.clamp(0.0, 1.0)) as usize;
        let mut train = EvalDataset::new(&format!("{}-train", self.name));
        let mut test = EvalDataset::new(&format!("{}-test", self.name));
        for (i, ex) in self.examples.iter().enumerate() {
            if i < split_at {
                train.add(ex.clone());
            } else {
                test.add(ex.clone());
            }
        }
        (train, test)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// McDataset
// ──────────────────────────────────────────────────────────────────────────────

/// A named collection of [`MultipleChoiceQuestion`] instances.
pub struct McDataset {
    /// Human-readable dataset name.
    pub name: String,
    /// All questions in insertion order.
    pub questions: Vec<MultipleChoiceQuestion>,
}

impl McDataset {
    /// Create an empty multiple-choice dataset.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            questions: Vec::new(),
        }
    }

    /// Append a question, trusting the caller that it is well-formed.
    ///
    /// This never rejects a question — including one with empty `choices`
    /// or a `correct_answer` outside `0..choices.len()` — so it silently
    /// produces a question that can never be scored correct (RAG-M1).
    /// Prefer [`Self::try_add`] for data coming from outside the current
    /// process (dataset conversions, deserialisation); this method is kept
    /// for in-process construction (including this crate's own tests) that
    /// already knows its data is valid and must not start returning
    /// `Result` (an `add(...)` call whose result is silently ignored would
    /// trip the `unused_must_use` lint under `-D warnings`).
    pub fn add(&mut self, q: MultipleChoiceQuestion) {
        self.questions.push(q);
    }

    /// Append a question, rejecting one with empty `choices` or a
    /// `correct_answer` outside `0..choices.len()` (RAG-M1) instead of
    /// silently accepting a question that can never be scored correct.
    pub fn try_add(&mut self, q: MultipleChoiceQuestion) -> Result<(), EvalError> {
        if let Some(problem) = mc_question_problem(&q) {
            return Err(EvalError::InvalidFormat(format!(
                "question {:?}: {problem}",
                q.id
            )));
        }
        self.questions.push(q);
        Ok(())
    }

    /// Return the number of questions.
    pub fn len(&self) -> usize {
        self.questions.len()
    }

    /// Return `true` if the dataset contains no questions.
    pub fn is_empty(&self) -> bool {
        self.questions.is_empty()
    }

    /// Parse a JSONL string into a multiple-choice dataset.
    ///
    /// Each line must have `"id"`, `"question"`, `"choices"` (array), and
    /// `"correct_answer"` (integer). `"subject"` and `"difficulty"` are optional.
    pub fn from_jsonl(name: &str, jsonl: &str) -> Result<Self, EvalError> {
        let mut dataset = McDataset::new(name);
        for (line_no, line) in jsonl.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(trimmed)
                .map_err(|e| EvalError::ParseError(format!("line {}: {}", line_no + 1, e)))?;
            let obj = v.as_object().ok_or_else(|| {
                EvalError::InvalidFormat(format!("line {} is not a JSON object", line_no + 1))
            })?;

            let id = obj
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("{}", line_no));

            let question = obj
                .get("question")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    EvalError::InvalidFormat(format!(
                        "line {}: missing \"question\" field",
                        line_no + 1
                    ))
                })?
                .to_string();

            let choices: Vec<String> = obj
                .get("choices")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    EvalError::InvalidFormat(format!(
                        "line {}: missing or invalid \"choices\" field",
                        line_no + 1
                    ))
                })?
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    c.as_str().map(str::to_string).ok_or_else(|| {
                        EvalError::InvalidFormat(format!(
                            "line {}: choice {} is not a string",
                            line_no + 1,
                            i
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;

            let correct_answer = obj
                .get("correct_answer")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    EvalError::InvalidFormat(format!(
                        "line {}: missing or invalid \"correct_answer\" field",
                        line_no + 1
                    ))
                })? as usize;

            let subject = obj
                .get("subject")
                .and_then(Value::as_str)
                .map(str::to_string);
            let difficulty = obj
                .get("difficulty")
                .and_then(Value::as_str)
                .map(str::to_string);

            let q = MultipleChoiceQuestion {
                id,
                question,
                choices,
                correct_answer,
                subject,
                difficulty,
            };
            // Validate the answer key *before* accepting the line: an
            // out-of-range `correct_answer` or empty `choices` would
            // otherwise load successfully and then silently score 0%
            // forever, indistinguishable from a model that answered
            // everything wrong (RAG-M1).
            if let Some(problem) = mc_question_problem(&q) {
                return Err(EvalError::InvalidFormat(format!(
                    "line {}: {problem}",
                    line_no + 1
                )));
            }
            dataset.questions.push(q);
        }
        Ok(dataset)
    }

    /// Return a new dataset containing only questions with the given subject.
    pub fn filter_by_subject(&self, subject: &str) -> McDataset {
        let mut out = McDataset::new(&format!("{}-{}", self.name, subject));
        for q in &self.questions {
            if q.subject.as_deref() == Some(subject) {
                out.add(q.clone());
            }
        }
        out
    }

    /// Return a deterministic random sample of `n` questions.
    ///
    /// Uses the same 64-bit LCG scheme as [`EvalDataset::sample_with_seed`].
    pub fn sample_with_seed(&self, n: usize, seed: u64) -> McDataset {
        let count = n.min(self.len());
        let mut indices: Vec<usize> = (0..self.len()).collect();

        let mut state = seed;
        for i in (1..indices.len()).rev() {
            state = lcg_step(state);
            let j = (state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let mut sampled = McDataset::new(&self.name);
        for &idx in indices.iter().take(count) {
            sampled.add(self.questions[idx].clone());
        }
        sampled
    }

    /// Return a sorted, deduplicated list of all subject names in this dataset.
    pub fn subjects(&self) -> Vec<String> {
        let mut seen: Vec<String> = self
            .questions
            .iter()
            .filter_map(|q| q.subject.clone())
            .collect();
        seen.sort();
        seen.dedup();
        seen
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared validation
// ──────────────────────────────────────────────────────────────────────────────

/// Check a [`MultipleChoiceQuestion`]'s answer-key invariants, returning a
/// human-readable problem description (not yet wrapped in an
/// [`EvalError`], so each call site can add its own context — a source
/// line number in [`McDataset::from_jsonl`], the question id in
/// [`McDataset::try_add`]) or `None` if the question is well-formed.
///
/// Used by both [`McDataset::from_jsonl`] (untrusted JSONL input) and
/// [`McDataset::try_add`] (in-process construction) so a dataset can never
/// silently carry an unscoreable question: a converted 5-option ARC item
/// saved with a 1-based `correct_answer`, or an empty `choices` array,
/// would otherwise load successfully and then score 0% forever with no
/// diagnostic — indistinguishable from a model that answered everything
/// wrong (RAG-M1).
fn mc_question_problem(q: &MultipleChoiceQuestion) -> Option<String> {
    if q.choices.is_empty() {
        return Some("choices must not be empty".to_string());
    }
    if q.correct_answer >= q.choices.len() {
        return Some(format!(
            "correct_answer {} out of range for {} choices",
            q.correct_answer,
            q.choices.len()
        ));
    }
    None
}
