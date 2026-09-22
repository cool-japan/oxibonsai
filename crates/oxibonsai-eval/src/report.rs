//! Evaluation report builder.
//!
//! [`EvalReport`] collects typed result entries from perplexity, accuracy, and
//! throughput evaluations, then formats them as JSON or GitHub-flavoured Markdown.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::accuracy::AccuracyResult;
use crate::error::EvalError;
use crate::perplexity::PerplexityResult;
use crate::throughput::ThroughputResult;

// ──────────────────────────────────────────────────────────────────────────────
// EvalResultEntry
// ──────────────────────────────────────────────────────────────────────────────

/// A single named metric entry in an evaluation report.
#[derive(Debug, Serialize)]
pub struct EvalResultEntry {
    /// Task or dataset name (e.g. "wikitext-2", "mmlu-biology").
    pub task: String,
    /// Metric name (e.g. "perplexity", "accuracy", "throughput").
    pub metric: String,
    /// Numeric value of the metric.
    pub value: f32,
    /// Unit string (e.g. "PPL", "%", "t/s").
    pub unit: String,
    /// Optional freeform notes (e.g. "stride=512").
    pub notes: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// EvalReport
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluation report collecting metrics from one or more evaluation tasks.
#[derive(Debug, Serialize)]
pub struct EvalReport {
    /// Name of the evaluated model.
    pub model_name: String,
    /// ISO 8601 timestamp when the report was created.
    pub timestamp: String,
    /// All metric entries added to this report.
    pub results: Vec<EvalResultEntry>,
}

impl EvalReport {
    /// Create a new empty report for `model_name`, stamped with the current UTC time.
    pub fn new(model_name: &str) -> Self {
        let timestamp = iso8601_now();
        Self {
            model_name: model_name.to_string(),
            timestamp,
            results: Vec::new(),
        }
    }

    /// Append a raw result entry.
    pub fn add(&mut self, entry: EvalResultEntry) {
        self.results.push(entry);
    }

    /// Append perplexity statistics for the named task.
    pub fn add_perplexity(&mut self, task: &str, result: &PerplexityResult) {
        self.add(EvalResultEntry {
            task: task.to_string(),
            metric: "perplexity".to_string(),
            value: result.mean_ppl,
            unit: "PPL".to_string(),
            notes: Some(format!("n={}, std={:.2}", result.n_samples, result.std_ppl)),
        });
    }

    /// Append accuracy statistics for the named task.
    pub fn add_accuracy(&mut self, task: &str, result: &AccuracyResult) {
        self.add(EvalResultEntry {
            task: task.to_string(),
            metric: "accuracy".to_string(),
            value: result.accuracy_pct(),
            unit: "%".to_string(),
            notes: Some(format!("{}/{}", result.correct, result.total)),
        });
    }

    /// Append throughput statistics.
    pub fn add_throughput(&mut self, result: &ThroughputResult) {
        self.add(EvalResultEntry {
            task: "throughput".to_string(),
            metric: "tokens_per_second".to_string(),
            value: result.tokens_per_second,
            unit: "t/s".to_string(),
            notes: Some(result.latency_breakdown()),
        });
    }

    /// Serialise the report to a pretty-printed JSON string, propagating
    /// any serialisation failure instead of discarding it.
    ///
    /// Realistically infallible for this struct today (every field is a
    /// `String`, `f32`, `Option<String>` or `Vec` of those), but returning
    /// `Result` means a caller that *does* hit a future failure (e.g. after
    /// a field addition that is not always representable) gets a real
    /// error instead of [`Self::to_json`]'s legacy placeholder (RAG-M6: the
    /// original `to_json` swallowed the `serde_json::Error` into a bare
    /// `"{}"`, which the CLI then happily wrote to disk as if it were a
    /// real report).
    ///
    /// Maps a `serde_json` failure to [`EvalError::Serialization`] (via the
    /// `#[from]` conversion) rather than [`EvalError::InvalidFormat`], so a
    /// caller that needs to distinguish "the input was malformed" from "the
    /// serialisation layer itself failed" can match on the real variant
    /// instead of parsing a formatted string (gatekeeper REQUIRED #12).
    pub fn try_to_json(&self) -> Result<String, EvalError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Serialise the report to a pretty-printed JSON string, silently
    /// returning the placeholder `"{}"` on failure instead of propagating
    /// the error.
    ///
    /// This is the crate's original, non-propagating signature, kept only
    /// so existing callers keep compiling unchanged; prefer
    /// [`Self::try_to_json`] in new code, which reports a real error
    /// instead of writing `"{}"` to disk as if it were the report (RAG-M6).
    pub fn to_json(&self) -> String {
        self.try_to_json().unwrap_or_else(|_| "{}".to_string())
    }

    /// Render the report as a GitHub-flavoured Markdown table.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!("# Evaluation Report — {}\n\n", self.model_name));
        md.push_str(&format!("**Timestamp:** {}\n\n", self.timestamp));
        md.push_str("| Task | Metric | Value | Unit | Notes |\n");
        md.push_str("|------|--------|------:|------|-------|\n");
        for entry in &self.results {
            let notes = entry.notes.as_deref().unwrap_or("-");
            md.push_str(&format!(
                "| {} | {} | {:.4} | {} | {} |\n",
                entry.task, entry.metric, entry.value, entry.unit, notes
            ));
        }
        md
    }

    /// One-line summary of the most important metrics.
    pub fn summary(&self) -> String {
        let ppl = self
            .results
            .iter()
            .find(|e| e.metric == "perplexity")
            .map(|e| format!("{:.2}", e.value))
            .unwrap_or_else(|| "N/A".to_string());

        let acc = self
            .results
            .iter()
            .find(|e| e.metric == "accuracy")
            .map(|e| format!("{:.1}%", e.value))
            .unwrap_or_else(|| "N/A".to_string());

        let tps = self
            .results
            .iter()
            .find(|e| e.metric == "tokens_per_second")
            .map(|e| format!("{:.1}", e.value))
            .unwrap_or_else(|| "N/A".to_string());

        format!(
            "Model: {} | PPL: {} | Acc: {} | TPS: {}",
            self.model_name, ppl, acc, tps
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Return the current UTC time as an ISO 8601 string (seconds precision).
///
/// Falls back to a static placeholder if the system clock is unavailable.
fn iso8601_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Manual conversion from Unix timestamp to ISO 8601 without external crates.
    // We use a simple decomposition valid for dates after 1970-01-01.
    let (year, month, day, hour, minute, second) = unix_secs_to_datetime(secs);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    )
}

/// Decompose a Unix timestamp (seconds since epoch) into (Y, M, D, h, m, s).
fn unix_secs_to_datetime(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let second = secs % 60;
    let minutes = secs / 60;
    let minute = minutes % 60;
    let hours = minutes / 60;
    let hour = hours % 24;
    let days = hours / 24;

    // Gregorian calendar decomposition
    // Days since 1970-01-01 → (year, day-of-year)
    let mut year = 1970u64;
    let mut remaining = days;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    let leap = is_leap(year);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u64;
    for &md in &month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        month += 1;
    }
    let day = remaining + 1;

    (year, month, day, hour, minute, second)
}

#[inline]
fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}
