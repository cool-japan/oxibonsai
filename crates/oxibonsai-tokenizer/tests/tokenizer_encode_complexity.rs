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
//! (pre-tokenization and special-token carve-out included) against a
//! *trained* tokenizer on a realistic mixed corpus (ASCII prose,
//! Japanese/CJK, emoji, source code) — the same corpus shape used by
//! `benches/tokenizer_bench.rs`'s `tok06_encode_size_sweep`. The small corpus
//! helpers are duplicated rather than shared: bench and test targets are
//! separate compilation units, and neither is a library.
//!
//! The CJK/emoji runs mirror the bench corpus: they contain no ASCII
//! whitespace, the no-natural-break shape that made the pre-fix
//! `bpe_merge_symbols` quadratic (measured 800 B / 15 ms -> 6400 B / 961 ms
//! on a whitespace-free emoji run). This test's trained tokenizer stores
//! non-ASCII bytes as `<0xHH>` symbols, so those runs reach it through byte
//! fallback with no merge work; the single-piece gate below therefore grows
//! an ASCII-letter run ([`ASCII_WORD_RUN`]) that the merge table does cover.
//!
//! ## The two complexity gates are deterministic work counts, not timings
//!
//! Both read `oxibonsai_tokenizer::bpe::merge_work_counter`, which counts
//! one unit per heap pop inside `bpe_merge_symbols`'s merge loop — the
//! loop's real work, not a proxy for the input size. It is `thread_local`
//! and only counts on a thread that called `reset_merge_work_counter`, so
//! concurrently running tests cannot perturb each other and production
//! callers never count at all. A wall-clock ratio of the same shape flaked
//! at a measured 3.24x under parallel-test CPU contention while passing in
//! isolation in 0.04 s; the timing forms are kept below, `#[ignore]`d, for a
//! human on quiet hardware.
//!
//! The two gates catch different regressions:
//! - [`doubling_the_corpus_does_not_roughly_quadruple_encode_work`] doubles a
//!   corpus of REPEATED paragraphs. Every piece keeps the same length, so the
//!   merge loop's own per-piece complexity cannot show here — what this
//!   catches is `encode` itself doing super-linear merge work in the NUMBER
//!   of pieces (for example re-encoding already-encoded text).
//! - [`quadrupling_one_whitespace_free_piece_does_not_square_merge_work`]
//!   grows ONE piece. `O(n log n)` (and the loop's `O(n)` pop count) predicts
//!   `work(4n)/work(n) ≈ 4`; `O(n^2)` predicts `16` — this is the gate on the
//!   merge loop itself, through the public API.

use std::time::{Duration, Instant};

use oxibonsai_tokenizer::bpe::{merge_work_counter, reset_merge_work_counter};
use oxibonsai_tokenizer::OxiTokenizer;
use oxibonsai_tokenizer::{pretokenize, BpeTrainer, TrainerConfig};

/// The prose words of [`mixed_paragraph`] run together: ASCII letters only,
/// so the legacy pre-tokenizer this test's trained tokenizer uses
/// ([`pretokenize`], which splits on whitespace and ASCII punctuation)
/// keeps any number of back-to-back copies in ONE piece, and the merge
/// table — learned over the paragraph's bytes, where printable ASCII bytes
/// are their own characters — has real character-level rules for it. (The
/// paragraph's CJK/emoji runs are no use here: that trainer stores every
/// non-ASCII byte as a `<0xHH>` symbol, which never matches the character
/// symbols `bpe_merge_symbols` starts from, so they cost zero merge work.)
const ASCII_WORD_RUN: &str = "thequickbrownfoxjumpsoverthelazydogwhilethecompiler\
                              optimizestheinnerloopforspeedandcorrectness";

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
/// single sample: a single sample on a shared, multi-agent build machine
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

/// Merge-loop work (heap pops inside `bpe_merge_symbols`) for ONE `encode`
/// call on `text`, on this test's own thread: the counter is reset (which
/// also turns counting on for this thread) immediately before the call and
/// read immediately after it. Deterministic, so a single call is enough.
fn merge_work_for(tokenizer: &OxiTokenizer, text: &str) -> u64 {
    reset_merge_work_counter();
    let ids = tokenizer
        .encode(text)
        .expect("encode should never fail on well-formed UTF-8");
    let work = merge_work_counter();
    assert!(
        !ids.is_empty(),
        "encode of {} non-empty bytes produced zero ids",
        text.len()
    );
    work
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

/// Supplementary wall-clock check (not a complexity gate — the two
/// work-counter tests below are), same shape as
/// [`six_kb_mixed_corpus_encodes_quickly_through_public_api`] but at 16x the
/// corpus size (100 KB), calibrated from the same pre-fix reference point
/// (800 B / 15 ms, 6400 B / 961 ms, i.e. ~64x growth per 8x size under
/// quadratic behavior). `#[ignore]`d because it is a wall-clock measurement,
/// and a wall-clock check never gates CI on its own here. Run explicitly
/// with `cargo test --release -p oxibonsai-tokenizer --test
/// tokenizer_encode_complexity -- --ignored --nocapture`.
#[test]
#[ignore = "wall-clock absolute bound: informational only, never gates CI; see \
            doubling_the_corpus_does_not_roughly_quadruple_encode_work for the CI gate"]
fn hundred_kb_mixed_corpus_encodes_without_quadratic_blowup() {
    let tokenizer = build_tokenizer();
    let elapsed = min_encode_time(&tokenizer, 102_400);
    assert!(
        elapsed < Duration::from_secs(8),
        "encoding a 100 KB mixed (ASCII/CJK/emoji/code) corpus through \
         OxiTokenizer::encode took {elapsed:?} — a healthy O(n log n) merge \
         loop finishes this comfortably under 8s (extrapolating the 6.4 KB \
         sibling test's own healthy timings), while a regression to \
         quadratic (the pre-fix behavior was 800 B/15 ms -> 6400 B/961 ms, \
         ~64x growth per 8x size) would take on the order of minutes at this \
         size — OxiTokenizer::encode's merge loop may have regressed to \
         quadratic (TOK-06)"
    );
}

/// Corpus-doubling complexity gate on the merge-loop work counter (see the
/// module doc for what this does and does not catch): doubling a corpus of
/// repeated paragraphs must roughly double the merge-loop work, never
/// roughly quadruple it. Healthy behavior is exactly linear here (every
/// repetition contributes the same pieces), so `< 3x` leaves a wide margin
/// on the healthy side while still failing a `~4x` regression.
#[test]
fn doubling_the_corpus_does_not_roughly_quadruple_encode_work() {
    let tokenizer = build_tokenizer();

    let work_n = merge_work_for(&tokenizer, &corpus_of_size(3200));
    let work_2n = merge_work_for(&tokenizer, &corpus_of_size(6400));
    eprintln!(
        "corpus doubling: merge work(n)={work_n} work(2n)={work_2n} ratio={:.3}",
        work_2n as f64 / work_n.max(1) as f64
    );

    assert!(
        work_n > 0 && work_2n > 0,
        "the merge-loop work counter must observe real merge work for a \
         trained tokenizer on a non-empty corpus (work_n={work_n}, \
         work_2n={work_2n}) — otherwise the ratio below is vacuous"
    );
    assert!(
        work_2n < work_n * 3,
        "merge work(2n)={work_2n} should be well under 3x work(n)={work_n} \
         (super-linear work in the number of pieces would give ~4x) — \
         OxiTokenizer::encode may be re-processing already-encoded text \
         (TOK-06)"
    );
}

/// Single-piece complexity gate on the merge-loop work counter (see the
/// module doc): one whitespace-free piece ([`ASCII_WORD_RUN`] repeated)
/// grown 4x must cost roughly 4x the merge-loop work (`O(n log n)`; the
/// loop's own heap-pop count is `O(n)`), never roughly 16x (`O(n^2)`, the
/// pre-fix behavior). The threshold — 8x plus a +64 slack for constant
/// overhead at small `n` — is the same one `bpe.rs`'s own
/// `bpe_merge_symbols_is_not_quadratic` uses: halfway (in ratio) between the
/// two complexity classes.
#[test]
fn quadrupling_one_whitespace_free_piece_does_not_square_merge_work() {
    const COPIES_N: usize = 16;
    let tokenizer = build_tokenizer();

    let run_n = ASCII_WORD_RUN.repeat(COPIES_N);
    let run_4n = ASCII_WORD_RUN.repeat(COPIES_N * 4);
    // Without this, a pre-tokenizer change that split the run into many
    // short pieces would turn this into a second corpus-scaling test that
    // passes whatever the merge loop's complexity is.
    for (label, run) in [("n", &run_n), ("4n", &run_4n)] {
        let pieces = pretokenize(run);
        assert_eq!(
            pieces.len(),
            1,
            "the {label} run must pre-tokenize to exactly one piece for this \
             test to measure the merge loop itself, got {} pieces",
            pieces.len()
        );
    }

    let work_n = merge_work_for(&tokenizer, &run_n);
    let work_4n = merge_work_for(&tokenizer, &run_4n);
    eprintln!(
        "single piece x4: merge work(n)={work_n} work(4n)={work_4n} ratio={:.3}",
        work_4n as f64 / work_n.max(1) as f64
    );

    assert!(
        work_n > 0,
        "the trained tokenizer must have merge rules for the prose letters \
         it was trained on, or this test measures nothing (work_n={work_n})"
    );
    assert!(
        work_4n <= 8 * work_n + 64,
        "merge work(4n)={work_4n} should stay well under 8x work(n)={work_n} \
         (+64 slack) for one {}-char whitespace-free piece — O(n^2) would \
         give ~16x — OxiTokenizer::encode's merge loop may have regressed to \
         quadratic (TOK-06)",
        run_4n.chars().count()
    );
}

/// Wall-clock sibling of the corpus-doubling gate above, preserved for a
/// human who wants to see actual timings (same reasoning as `bpe.rs`'s own
/// `bpe_merge_symbols_is_not_quadratic_wall_clock`): `#[ignore]`d because a
/// duration-based ratio assertion flakes under co-scheduled CPU contention
/// on a shared, multi-agent build machine and must never gate CI. Run
/// explicitly with `cargo test --release -p oxibonsai-tokenizer --test
/// tokenizer_encode_complexity -- --ignored --nocapture`.
#[test]
#[ignore = "wall-clock ratio: flakes under parallel-test CPU contention; see \
            doubling_the_corpus_does_not_roughly_quadruple_encode_work for the CI gate"]
fn doubling_the_corpus_does_not_roughly_quadruple_encode_time() {
    let tokenizer = build_tokenizer();

    let t_n = min_encode_time(&tokenizer, 3200);
    let t_2n = min_encode_time(&tokenizer, 6400);

    eprintln!(
        "t(n)={t_n:?} t(2n)={t_2n:?} ratio={:.3}",
        t_2n.as_secs_f64() / t_n.as_secs_f64().max(1e-12)
    );
    assert!(
        t_2n < t_n * 3,
        "t(2n)={t_2n:?} should be well under 3x t(n)={t_n:?} (O(n^2) would \
         give ~4x) — OxiTokenizer::encode's merge loop may have regressed \
         to quadratic (TOK-06)"
    );
}
