//! TOK-06 regression guard at the public `OxiTokenizer::encode` level.
//!
//! T-15 (folded from deps-10, RAG-EVAL-IMG-24, TOK-16) asks for the
//! `tokenizer_bench.rs` throughput benchmark to double as TOK-06's
//! regression guard, "not only in the bench" — a criterion benchmark
//! (`harness = false`, its own `fn main`) never runs under `cargo test`, so
//! the actual pass/fail gate has to be a real `#[test]`, in this file.
//!
//! `crates/oxibonsai-tokenizer/src/bpe.rs` already carries two TOK-06
//! guards (`bpe_merge_symbols_is_not_quadratic`,
//! `fifteen_kb_mixed_text_encodes_quickly`) against the low-level
//! `bpe_encode` primitive with a hand-built vocabulary. This file is a
//! complementary guard through the *public* `OxiTokenizer::encode` API
//! (pre-tokenization regex, NFC normalization and special-token carve-out
//! included) against a *trained* tokenizer on a realistic mixed corpus
//! (ASCII prose, Japanese/CJK, emoji, source code) — the same corpus shape
//! used by `benches/tokenizer_bench.rs`'s `tok06_encode_size_sweep`, kept in
//! sync by duplicating the same small helpers (bench and test targets are
//! separate compilation units; there is nowhere shared to put them without
//! touching a source file outside this package's remit).
//!
//! The CJK/emoji runs are not decorative: they contain no ASCII whitespace,
//! which is exactly the no-natural-break shape that made the pre-fix
//! `bpe_merge_symbols` quadratic (measured 800 B / 15 ms -> 6400 B / 961 ms
//! on a whitespace-free emoji run).

use std::time::{Duration, Instant};

use oxibonsai_tokenizer::OxiTokenizer;
use oxibonsai_tokenizer::{BpeTrainer, TrainerConfig};

/// One paragraph mixing ASCII prose, Japanese (CJK), emoji, and a small
/// source-code snippet — kept in sync with
/// `benches/tokenizer_bench.rs::mixed_paragraph`.
fn mixed_paragraph() -> String {
    let mut s = String::new();
    s.push_str(
        "The quick brown fox jumps over the lazy dog while the compiler optimizes \
         the inner loop for speed and correctness. ",
    );
    s.push_str(
        "実際のプロダクション環境では日本語のテキストも大量に処理されるため、\
         トークナイザのスループットは性能上とても重要な指標になります。",
    );
    s.push_str(" 🦀🔥🚀✨🎉🐍🧠📈💡🌏🛠🧵🧩🔬🎯 ");
    s.push_str(
        "fn gemv_dispatch(blocks: &[Block], input: &[f32], output: &mut [f32]) -> Result<()> {\n",
    );
    s.push_str("    for row in 0..n_rows {\n        let mut sum = 0.0f32;\n");
    s.push_str("        for bi in 0..blocks_per_row { sum += dot(blocks[bi], input); }\n");
    s.push_str("        output[row] = sum;\n    }\n    Ok(())\n}\n");
    s
}

/// Repeat [`mixed_paragraph`] until at least `target_bytes` long.
fn corpus_of_size(target_bytes: usize) -> String {
    let unit = mixed_paragraph();
    let mut out = String::with_capacity(target_bytes + unit.len());
    while out.len() < target_bytes {
        out.push_str(&unit);
    }
    out
}

/// Train a small-but-real tokenizer so the merge loop has real work to do
/// (an untrained vocabulary would only ever exercise byte-fallback).
fn build_tokenizer() -> OxiTokenizer {
    let docs: Vec<String> = (0..64).map(|_| mixed_paragraph()).collect();
    let corpus: Vec<&str> = docs.iter().map(String::as_str).collect();
    let mut trainer = BpeTrainer::new(TrainerConfig::new(2048));
    trainer
        .train(&corpus)
        .expect("training a small, well-formed corpus should never fail")
        .to_oxi_tokenizer()
}

/// Time the minimum of `RUNS` `encode` calls at `n` bytes (min-of-N, not a
/// single sample: this test runs in the release gate on a shared,
/// multi-agent build machine per session CONTEXT.md, where a single sample
/// can be skewed by a co-scheduled build).
fn min_encode_time(tokenizer: &OxiTokenizer, n: usize) -> Duration {
    const RUNS: u32 = 5;
    let text = corpus_of_size(n);
    let mut best = Duration::MAX;
    for _ in 0..RUNS {
        let start = Instant::now();
        let ids = tokenizer
            .encode(&text)
            .expect("encode should never fail on well-formed UTF-8");
        let elapsed = start.elapsed();
        assert!(
            !ids.is_empty(),
            "encode of {n} non-empty bytes produced zero ids"
        );
        best = best.min(elapsed);
    }
    best
}

/// The literal TOK-06 gate, generalized: a real mixed corpus at the
/// finding's own largest repro size (6400 B) must encode comfortably fast
/// through the public API, not just the internal `bpe_encode` primitive.
///
/// Bound kept generous (a regression guard needs to catch a 10x-or-worse
/// blowup, not flake on a 20% one): the pre-fix behavior was 961 ms at this
/// size; a healthy implementation finishes in low single-digit
/// milliseconds even including tokenizer training overhead amortized away
/// (training happens once, outside the timed loop).
#[test]
fn six_kb_mixed_corpus_encodes_quickly_through_public_api() {
    let tokenizer = build_tokenizer();
    let elapsed = min_encode_time(&tokenizer, 6400);
    assert!(
        elapsed < Duration::from_millis(300),
        "encoding a 6.4 KB mixed (ASCII/CJK/emoji/code) corpus through \
         OxiTokenizer::encode took {elapsed:?} (pre-TOK-06-fix behavior was \
         ~961 ms at this size; a healthy implementation is comfortably under \
         300 ms even on a loaded, shared build machine) — the merge loop may \
         have regressed to quadratic"
    );
}

/// The complexity-ratio half of the TOK-06 gate: doubling the input must
/// not roughly quadruple the time. O(n log n) predicts the ratio tends to
/// 2 as n grows; O(n^2) predicts 4. Assert comfortably below the quadratic
/// bound rather than pinning the sub-quadratic prediction exactly, so
/// ordinary timer/scheduler noise on a shared machine cannot flake this
/// test — it only needs to catch an order-of-magnitude regression.
#[test]
fn doubling_the_corpus_does_not_roughly_quadruple_encode_time() {
    let tokenizer = build_tokenizer();

    let t_n = min_encode_time(&tokenizer, 3200);
    let t_2n = min_encode_time(&tokenizer, 6400);

    assert!(
        t_2n < t_n * 3,
        "t(2n)={t_2n:?} should be well under 3x t(n)={t_n:?} (O(n^2) would \
         give ~4x) — OxiTokenizer::encode's merge loop may have regressed \
         to quadratic (TOK-06)"
    );
}
