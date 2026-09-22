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
/// - `{a}`, `{b}`, `{c}`, … up to `{z}` — choices 0, 1, 2, … (text after the
///   label prefix). The standard template only wires up `{a}..{d}`, but a
///   question with more choices than the template references (e.g. a
///   5-option ARC-Challenge item through the 4-slot default template) still
///   gets every choice rendered — see [`McEvaluator::format_question`].
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
    /// The template is scanned left to right exactly once, substituting
    /// `{question}` with `q.question` and `{a}`, `{b}`, `{c}`, … (any single
    /// lowercase letter up to `{z}`) with the correspondingly-indexed,
    /// label-stripped entry of `q.choices` (missing choices become an empty
    /// string, as before). Because the scan never revisits text it has
    /// already emitted, a question or choice string that itself contains
    /// literal `{a}`..`{d}`-style text can never be re-interpreted as a
    /// placeholder (RAG-M3) — the old implementation used four chained
    /// [`str::replace`] calls, so substituting `{question}` first and then
    /// `{a}` second let question text containing `"{a}"` expand into the
    /// next choice's placeholder.
    ///
    /// If `q.choices` has more entries than the template has lettered
    /// placeholders for (e.g. a 5-option ARC-Challenge item rendered
    /// through the standard 4-slot `{a}..{d}` template), the extra choices
    /// are appended, each as its own `"<LETTER>) <text>"` line, right after
    /// the last placeholder the template *did* reference — so a fixed
    /// 4-choice template still surfaces every option instead of silently
    /// dropping the fifth (RAG-EVAL-IMG-04).
    pub fn format_question(&self, q: &MultipleChoiceQuestion) -> String {
        let template = self.prompt_template.as_str();
        let mut out = String::with_capacity(template.len() + q.question.len());
        // Highest choice index substituted via an explicit `{letter}`
        // placeholder so far, and the byte offset in `out` right after it —
        // that offset is where any *un-templated* trailing choices get
        // appended.
        let mut max_placeholder_idx: Option<usize> = None;
        let mut insert_at: usize = 0;

        // Repeatedly slice off "everything up to and including the next
        // recognised placeholder" using `str::find`, which always returns
        // char-boundary-safe byte offsets. This never re-scans text already
        // pushed into `out` (fixing RAG-M3) and never indexes a string by a
        // hand-computed offset (fixing RAG-EVAL-IMG-01's panic on
        // non-ASCII choice text, via `strip_choice_label` below).
        let mut rest = template;
        while let Some(brace_rel) = rest.find('{') {
            out.push_str(&rest[..brace_rel]);
            let after_brace = &rest[brace_rel + 1..];
            match after_brace.find('}') {
                Some(name_len) => {
                    let name = &after_brace[..name_len];
                    let first_char_lowercase_ascii =
                        name.len() == 1 && name.as_bytes()[0].is_ascii_lowercase();
                    if name == "question" {
                        out.push_str(&q.question);
                        rest = &after_brace[name_len + 1..];
                    } else if first_char_lowercase_ascii {
                        let idx = (name.as_bytes()[0] - b'a') as usize;
                        out.push_str(&strip_choice_label(get_choice(q, idx)));
                        max_placeholder_idx = Some(max_placeholder_idx.map_or(idx, |m| m.max(idx)));
                        insert_at = out.len();
                        rest = &after_brace[name_len + 1..];
                    } else {
                        // Unrecognised placeholder (e.g. `{foo}`, `{A}`,
                        // `{}`): leave it as literal text. Push the '{' now
                        // and let the rest — including the eventual '}' —
                        // fall through the loop as ordinary text.
                        out.push('{');
                        rest = after_brace;
                    }
                }
                None => {
                    // Unterminated '{' with no matching '}' anywhere after
                    // it: keep it literal and stop looking for more
                    // placeholders in what remains.
                    out.push('{');
                    rest = after_brace;
                }
            }
        }
        out.push_str(rest);

        // If the template never referenced a single lettered placeholder
        // (e.g. a custom template with only `{question}`), there is no
        // sensible mid-template anchor — append every choice at the very
        // end of the rendered text instead of at the recorded (unset,
        // zero) `insert_at`.
        let insert_at = if max_placeholder_idx.is_some() {
            insert_at
        } else {
            out.len()
        };
        let next_idx = max_placeholder_idx.map_or(0, |m| m + 1);
        if next_idx < q.choices.len() {
            let mut extra = String::new();
            for idx in next_idx..q.choices.len().min(26) {
                let letter = (b'A' + idx as u8) as char;
                extra.push('\n');
                extra.push(letter);
                extra.push_str(") ");
                extra.push_str(&strip_choice_label(get_choice(q, idx)));
            }
            out.insert_str(insert_at, &extra);
        }

        out
    }

    /// Return `true` if the completion's extracted answer (bounded to the
    /// crate's historical 4-choice default; see
    /// [`Self::score_completion_bounded`] for N-option datasets) equals
    /// `correct_answer`.
    ///
    /// `correct_answer` is a 0-based index; 0 → 'A', 1 → 'B', 2 → 'C', 3 → 'D'.
    pub fn score_completion(&self, completion: &str, correct_answer: usize) -> bool {
        self.score_completion_bounded(completion, correct_answer, 4)
    }

    /// Like [`Self::score_completion`], but the extracted letter must index
    /// within `0..num_choices` rather than the fixed default of 4 — the
    /// form [`AccuracyResult`]-producing evaluators should use once they
    /// know a specific question's choice count (see
    /// [`McEvaluator::evaluate_dataset`], and [`crate::arc::ArcEvaluator`]
    /// for 5-option ARC items).
    pub fn score_completion_bounded(
        &self,
        completion: &str,
        correct_answer: usize,
        num_choices: usize,
    ) -> bool {
        self.extract_answer_bounded(completion, num_choices) == Some(correct_answer)
    }

    /// [`Self::extract_answer_bounded`] with the crate's historical default
    /// of 4 choices (A–D).
    pub fn extract_answer(&self, completion: &str) -> Option<usize> {
        self.extract_answer_bounded(completion, 4)
    }

    /// Extract a 0-based answer index from a model completion, accepting up
    /// to `num_choices` options (A, B, C, … up to the `num_choices`-th
    /// letter).
    ///
    /// Algorithm:
    /// 1. Strip any `<think>...</think>` block(s) — reasoning models often
    ///    restate the final answer *after* their chain-of-thought, and the
    ///    thinking text itself is not a reliable place to look for one.
    /// 2. Scan (from the end) for the last `Answer:` marker (case
    ///    insensitive) that is immediately followed — after optional
    ///    whitespace — by a **standalone** letter: one not itself followed
    ///    by another alphanumeric character. This rejects prose like
    ///    `"Answer: Before we begin, A"` (the letter-looking character
    ///    starts a longer word) instead of either mis-crediting it or
    ///    silently falling through to some other heuristic.
    ///    - If such a marker+letter pair is found, that letter is the
    ///      answer.
    ///    - If the text contains an `Answer:` marker at all but *none* of
    ///      its occurrences are followed by a standalone letter, extraction
    ///      fails (`None`) — it does **not** fall back to step 3, because
    ///      the completion's first character is frequently the `'A'` of the
    ///      word "Answer" itself, which is exactly the RAG-EVAL-IMG-03 bug
    ///      (`"Answer: B"` being silently credited as choice A).
    /// 3. If there is no `Answer:` marker anywhere, fall back to the
    ///    crate's original heuristic: the first character of the
    ///    (trimmed) text, if it is a letter.
    ///
    /// The resulting letter is mapped to a 0-based index (`'A'` → 0, `'B'`
    /// → 1, …); the index must be `< num_choices` or extraction fails.
    /// Returns `None` — never a default index — whenever no answer can be
    /// determined, so a caller can distinguish "the model answered
    /// something else" from "we couldn't tell".
    pub fn extract_answer_bounded(&self, completion: &str, num_choices: usize) -> Option<usize> {
        let cleaned = strip_think_blocks(completion);
        let letter = match scan_answer_marker(&cleaned) {
            AnswerMarker::Found(c) => c,
            AnswerMarker::FoundButUnusable => return None,
            AnswerMarker::Absent => {
                let c = cleaned.trim().chars().next()?;
                if !c.is_ascii_alphabetic() {
                    return None;
                }
                c.to_ascii_uppercase()
            }
        };
        let idx = (letter as u8 - b'A') as usize;
        (idx < num_choices).then_some(idx)
    }

    /// Evaluate a multiple-choice dataset given one completion per question.
    ///
    /// `completions` must have the same length as `dataset.questions`.
    /// Mismatched lengths are handled gracefully: only the shorter slice is used.
    ///
    /// Each question is scored with its *own* choice count (via
    /// [`Self::score_completion_bounded`]), not the crate's 4-choice
    /// default — so a 5-option ARC-Challenge item's `"E"` completion is
    /// scoreable (RAG-EVAL-IMG-04) while a 2-option WinoGrande item can
    /// never be credited for a spurious `"C"`.
    pub fn evaluate_dataset(&self, dataset: &McDataset, completions: &[String]) -> AccuracyResult {
        let mut correct = 0usize;
        let mut total = 0usize;

        // subject → (correct, total)
        let mut by_subject_counts: HashMap<String, (usize, usize)> = HashMap::new();

        for (q, completion) in dataset.questions.iter().zip(completions.iter()) {
            total += 1;
            let is_correct =
                self.score_completion_bounded(completion, q.correct_answer, q.choices.len());
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
// Internal helpers — `format_question` / `extract_answer_bounded` support
// ──────────────────────────────────────────────────────────────────────────────

/// `q.choices[i]`, or `""` if there is no such choice (mirrors the original
/// "missing slots become empty" contract of [`McEvaluator::format_question`]).
fn get_choice(q: &MultipleChoiceQuestion, i: usize) -> &str {
    q.choices.get(i).map(String::as_str).unwrap_or("")
}

/// Strip a leading single-character label prefix like `"A: "` from a choice
/// string, e.g. turning `"A: Paris"` into `"Paris"`.
///
/// Char-boundary safe (RAG-EVAL-IMG-01): the original implementation
/// sliced at the fixed byte offset `2`, which panics whenever the label
/// character is multi-byte in UTF-8 (any CJK character, most emoji, …),
/// e.g. a choice literally labelled `"あ: 東京"`. This version walks
/// [`str::char_indices`] instead, so the byte offset used to slice off the
/// prefix is always the true width of whatever the first character is.
///
/// Only strips the prefix when there is non-whitespace content left after
/// it — a bare label like `"A:"` (no answer text at all) is returned
/// unchanged, exactly as the original `s.len() >= 3` guard intended, so a
/// dataset that uses bare-label placeholders does not collapse to an empty
/// choice string.
fn strip_choice_label(s: &str) -> String {
    let mut chars = s.char_indices();
    if let (Some(_), Some((second_byte, second_char))) = (chars.next(), chars.next()) {
        if second_char == ':' {
            let after_colon = second_byte + second_char.len_utf8();
            let rest = s[after_colon..].trim();
            if !rest.is_empty() {
                return rest.to_string();
            }
        }
    }
    s.to_string()
}

/// Remove every `<think>...</think>` span from `s`. An unterminated
/// `<think>` (no matching `</think>`) is treated as "still thinking" and
/// everything from that point to the end of the string is dropped, since
/// there is no reliable answer inside an unfinished reasoning block.
pub(crate) fn strip_think_blocks(s: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find(OPEN) {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + OPEN.len()..];
        match after_open.find(CLOSE) {
            Some(end) => rest = &after_open[end + CLOSE.len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Outcome of [`scan_answer_marker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnswerMarker {
    /// No `answer:` marker (case-insensitive) anywhere in the text.
    Absent,
    /// The last `answer:` occurrence followed by a standalone letter.
    Found(char),
    /// At least one `answer:` occurrence exists, but none of them are
    /// followed (after optional whitespace) by a standalone letter.
    FoundButUnusable,
}

/// Scan `s` for the *last* case-insensitive `answer:` marker that is
/// immediately followed — after any amount of whitespace — by a standalone
/// ASCII letter (not itself followed by another alphanumeric character).
///
/// Scanning proceeds from the end of the string backwards one `answer:`
/// occurrence at a time so that, e.g., `"Answer: A, on reflection, Answer:
/// B"` yields `B`, not `A`.
fn scan_answer_marker(s: &str) -> AnswerMarker {
    const MARKER: &str = "answer:";
    let lower = s.to_lowercase();
    if !lower.contains(MARKER) {
        return AnswerMarker::Absent;
    }

    let mut search_end = lower.len();
    loop {
        let byte_idx = match lower[..search_end].rfind(MARKER) {
            Some(b) => b,
            None => return AnswerMarker::FoundButUnusable,
        };
        let after = &lower[byte_idx + MARKER.len()..];
        let mut chars = after.chars().skip_while(|c| c.is_whitespace());
        if let Some(c) = chars.next() {
            if c.is_ascii_alphabetic() {
                let followed_by_word_char = chars.next().is_some_and(|c2| c2.is_alphanumeric());
                if !followed_by_word_char {
                    return AnswerMarker::Found(c.to_ascii_uppercase());
                }
            }
        }
        if byte_idx == 0 {
            return AnswerMarker::FoundButUnusable;
        }
        search_end = byte_idx;
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
