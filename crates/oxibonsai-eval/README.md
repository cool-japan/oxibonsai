# oxibonsai-eval

[![Version](https://img.shields.io/badge/version-0.2.3-blue.svg)](https://crates.io/crates/oxibonsai-eval)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-274%20passing-brightgreen.svg)]()

Model evaluation harness for OxiBonsai — ROUGE, perplexity, accuracy, throughput.

Provides perplexity measurement, MMLU-style multiple-choice accuracy,
ROUGE-N/L/S scoring, exact-match scoring, throughput benchmarking, JSONL
dataset loading, and JSON/Markdown report generation.

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Status

**Stable** (v0.2.3) — 274 tests passing.

## Features

- `PerplexityEvaluator` — from log-probs or logits; bits-per-byte metric
- `McEvaluator` — MMLU-style multiple-choice with per-subject breakdown
- `ExactMatchEvaluator` — text-match evaluation; `exact_match` / `f1_score` QA scoring
- ROUGE scoring: `RougeNScore` (ROUGE-1/2), `RougeLScore`, `RougeSScore`, `CorpusRouge`
- BLEU scoring: `BleuScore`, `sentence_bleu`, `corpus_bleu`
- ChrF (character n-gram F-score) metric
- METEOR metric
- Bootstrap confidence intervals for all metrics
- `ThroughputBenchmark` — tokens/s, prefill/decode latency, p95/p99
- `EvalDataset` — JSONL loading, train/test splits, deterministic sampling
- `EvalReport` — JSON and Markdown report generation
- Zero external API dependencies — pure Rust

## Usage

```toml
[dependencies]
oxibonsai-eval = "0.2.3"
```

```rust
use oxibonsai_eval::{corpus_bleu, BleuConfig, PerplexityEvaluator};

// Perplexity from token log-probabilities
let log_probs = vec![-1.2, -0.8, -2.1, -1.5];
let ppl = PerplexityEvaluator::new().compute(&log_probs);
println!("Perplexity: {:.2}", ppl);

// Corpus BLEU
let hypotheses = vec!["the cat sat on the mat"];
let references = vec![vec!["the cat is on the mat"]];
let bleu = corpus_bleu(&hypotheses, &references, &BleuConfig::default());
println!("BLEU: {:.4}", bleu.bleu);
```

## License

Apache-2.0 — COOLJAPAN OU
