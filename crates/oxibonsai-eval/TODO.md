# oxibonsai-eval TODO

> Model evaluation harness: perplexity, accuracy, ROUGE, throughput metrics
> 17 source files, 9 test files, 151 integration + proptest tests passing
> Version: 0.1.3 — Status: Stable
> Last updated: 2026-05-03

## Status: Stable — BLEU / chrF / METEOR / SQuAD / calibration / bootstrap / streaming added

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
