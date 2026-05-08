# oxibonsai-eval TODO

> Model evaluation harness: perplexity, accuracy, ROUGE, throughput metrics
> 19 source files, 11 test files, 219 integration + proptest tests passing
> Version: 0.1.4 — Status: Stable
> Last updated: 2026-05-05

## Status: Stable — Phase 18: WinoGrande + BoolQ evaluators added

Full evaluation framework with perplexity, multiple-choice accuracy (string +
logit), ROUGE scoring, BLEU, chrF/chrF++, METEOR (lexical), SQuAD EM+F1,
calibration (ECE/Brier/NLL), bootstrap CIs, streaming/online counters,
throughput benchmarking, and dataset loading.

## Done

- [x] `PerplexityEvaluator` — from logits, bits-per-byte (BPB) metric
- [x] Pure Rust log-softmax computation (no external math dependency)
- [x] `McEvaluator` — MMLU-style multiple-choice accuracy with per-subject breakdown
- [x] `McLogitEvaluator` — logit-based MC scoring (argmax over per-choice log-probs)
- [x] `ExactMatchEvaluator` — text matching evaluation
- [x] `EvalDataset` — JSONL loading, sampling (`sample_with_seed`), train/test splits
- [x] `McDataset` / `MultipleChoiceQuestion` — structured MC dataset support (`sample_with_seed`)
- [x] Deterministic sampling with LCG RNG (no external rand dependency)
- [x] ROUGE-N/L/S scoring metrics
- [x] BLEU — sentence + corpus, brevity penalty, three smoothing modes
- [x] chrF / chrF++ — character n-gram F-score with word-order mixing
- [x] METEOR (lexical) — alignment + fragmentation penalty
- [x] SQuAD-style QA — normalisation, EM, token F1, corpus aggregation
- [x] Calibration — ECE (equal-width bins), multi-class Brier, stable NLL
- [x] Bootstrap confidence intervals — xorshift64\*, seed-deterministic
- [x] Streaming — `OnlinePerplexity` + `OnlineAccuracy`
- [x] Throughput evaluator — tokens-per-second, latency benchmarking
- [x] JSON/Markdown report generation
- [x] Error types (`EvalError`) — `#[non_exhaustive]`, `#[from] std::io::Error`
- [x] Criterion benchmark harness (`benches/eval_bench.rs`)
- [x] Tests for ROUGE metrics, perplexity, accuracy scoring
- [x] Alpha → Stable uplift for `oxibonsai-eval` (2026-04-19)

## Phase 17 — ARC + GSM8K Evaluators

- [x] **`ArcEvaluator`** — ARC-Easy/Challenge 4/5-way MC via `McEvaluator`/`McLogitEvaluator`; `ArcSplit::{Easy, Challenge}`; both completion and logit scoring paths; 10 integration tests
- [x] **`Gsm8kEvaluator`** — `extract_final_answer` scans from end for `#### N`, supports negative + decimal; numeric exact-match with 1e-6 tolerance; `GsmkResult` with `no_answer_extracted` counter; 15 integration tests

## Phase 19 — MMLU + HellaSwag + TruthfulQA Evaluators

- [x] **`MmluEvaluator`** — 57-subject 4-choice MMLU benchmark; delegates to `McEvaluator`/`McLogitEvaluator`; `MmluResult` with `by_subject: HashMap<String, AccuracyResult>`; subject extracted from `question.subject` or `question.id` (before first `/`); `evaluate_completions`, `evaluate_logits`, `evaluate_by_subject_completions`, `evaluate_by_subject_logits`; 20 tests in `tests/mmlu_tests.rs`
- [x] **`HellaSwagEvaluator`** — 4-choice sentence completion; reuses `McEvaluator`/`McLogitEvaluator` via `as_mc_dataset()`; `HellaSwagDataset`/`HellaSwagItem`/`HellaSwagResult`; 12+ integration tests
- [x] **`TruthfulQaEvaluator`** — MC1 (single-correct argmax) + MC2 (fraction of probability mass on correct answers); `TruthfulQaMode::{Mc1, Mc2}`; `TruthfulQaResult`; 12+ integration tests

## Phase 18 — WinoGrande + BoolQ Evaluators

- [x] **`WinoGrandeEvaluator`** — 2-choice commonsense fill-in-the-blank; delegates to `McEvaluator`/`McLogitEvaluator` via `as_mc_dataset()` (1-based answer → 0-based index mapping); `WinoGrandeResult` with accuracy_pct; 12 integration tests
- [x] **`BoolQEvaluator`** — yes/no passage QA; `extract_answer()` with case-insensitive prefix match; completion + logit scoring paths; yes/no prediction counters; `BoolQResult`; 16 integration tests
