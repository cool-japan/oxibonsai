//! # oxibonsai-eval
//!
//! Model evaluation harness for OxiBonsai.
//!
//! Provides utilities for:
//!
//! - **Perplexity** — measures how well a model predicts held-out text.
//! - **MMLU-style multiple choice** — accuracy on four-option questions
//!   (both string-parsing [`McEvaluator`] and logit-based
//!   [`accuracy::McLogitEvaluator`]).
//! - **Exact match** — token-level accuracy for text-generation tasks.
//! - **BLEU** — corpus / sentence BLEU with 1..N orders and smoothing.
//! - **chrF / chrF++** — character n-gram F-score (Popović 2015).
//! - **METEOR (lexical)** — exact-match-only METEOR.
//! - **SQuAD F1 + EM** — standard SQuAD 1.1 normalisation.
//! - **Calibration** — ECE, Brier score, NLL (numerically stable).
//! - **Bootstrap CIs** — seed-deterministic percentile intervals.
//! - **Streaming / online** — running perplexity and accuracy counters.
//! - **Throughput benchmarking** — tokens-per-second and latency statistics.
//! - **Dataset loading** — JSONL-based [`EvalDataset`] and [`McDataset`].
//! - **Report generation** — JSON and Markdown evaluation reports.
//!
//! ## Quick start
//!
//! ```rust
//! use oxibonsai_eval::perplexity::PerplexityEvaluator;
//!
//! let eval = PerplexityEvaluator::new();
//! // Perfect predictions → PPL ≈ 1.0
//! let ppl = eval.compute(&[0.0f32; 10]);
//! assert!((ppl - 1.0).abs() < 1e-5);
//! ```

pub mod accuracy;
pub mod arc;
pub mod bleu;
pub mod boolq;
pub mod bootstrap;
pub mod calibration;
pub mod chrf;
pub mod dataset;
pub mod error;
pub mod gsm8k;
pub mod meteor;
pub mod perplexity;
pub mod qa;
pub mod report;
pub mod rouge;
pub mod streaming;
pub mod throughput;
pub mod winogrande;

#[cfg(test)]
mod tests;

// ──────────────────────────────────────────────────────────────────────────────
// Public re-exports
// ──────────────────────────────────────────────────────────────────────────────

pub use accuracy::{
    AccuracyResult, ExactMatchEvaluator, LogitMcResult, McEvaluator, McLogitEvaluator,
};
pub use arc::{ArcEvaluator, ArcResult, ArcSplit};
pub use boolq::{BoolQDataset, BoolQEvaluator, BoolQItem, BoolQResult};
pub use gsm8k::{Gsm8kEvaluator, Gsm8kResult};
pub use winogrande::{WinoGrandeDataset, WinoGrandeEvaluator, WinoGrandeItem, WinoGrandeResult};
pub use bleu::{corpus_bleu, sentence_bleu, BleuConfig, BleuScore, SmoothingMethod};
pub use bootstrap::{bootstrap_ci, ConfidenceInterval};
pub use calibration::{
    brier_score, calibration_all, expected_calibration_error, nll_from_logits, BinStat,
    CalibrationResult,
};
pub use chrf::{chrf, chrf_plus_plus, chrf_with, ChrfScore};
pub use dataset::{EvalDataset, EvalExample, McDataset, MultipleChoiceQuestion};
pub use error::EvalError;
pub use meteor::{align_tokens, meteor, meteor_multi, MeteorConfig, MeteorScore};
pub use perplexity::{PerplexityEvaluator, PerplexityResult};
pub use qa::{
    corpus_em_f1, exact_match as qa_exact_match, f1_score as qa_f1_score, normalize_answer,
    normalize_tokens, score_multi as qa_score_multi, QaScore,
};
pub use report::{EvalReport, EvalResultEntry};
pub use rouge::{
    ngram_counts, tokenize, CorpusRouge, RougeLScore, RougeNScore, RougeSScore, TokenSeq,
};
pub use streaming::{OnlineAccuracy, OnlinePerplexity};
pub use throughput::{percentile, time_fn, ThroughputBenchmark, ThroughputResult};
