//! Encode/decode throughput benchmarks for [`OxiTokenizer`].
//!
//! T-15 (folded from deps-10, RAG-EVAL-IMG-24, TOK-16): the `[[bench]]`
//! entry in `Cargo.toml` used to resolve to a no-op placeholder. This file
//! replaces it with:
//!
//! 1. `tokenizer_encode_throughput` / `tokenizer_decode_throughput` — real
//!    encode/decode throughput on a mixed corpus (ASCII prose, Japanese
//!    (CJK), emoji, and a source-code snippet) at several sizes.
//! 2. `tok06_encode_size_sweep` — the criterion-visible, tracked-over-time
//!    counterpart to TOK-06's regression guard. The actual pass/fail gate
//!    is the `#[test]` in
//!    `crates/oxibonsai-tokenizer/tests/tokenizer_encode_complexity.rs`
//!    (criterion benchmarks report numbers, they do not assert), which
//!    exercises the same [`build_tokenizer`] / [`corpus_of_size`] shapes
//!    through the public `OxiTokenizer::encode` API rather than the
//!    lower-level `bpe_encode` primitive that
//!    `crates/oxibonsai-tokenizer/src/bpe.rs`'s own
//!    `bpe_merge_symbols_is_not_quadratic` / `fifteen_kb_mixed_text_encodes_quickly`
//!    tests already cover.
//! 3. `special_id_lookup` — `Vocabulary::is_special_id` (a linear scan over
//!    the special-token set), benchmarked as the RT-ADMIN-CFG blocking-1 /
//!    decision D-5 precondition for flipping the workspace's `hf-tokenizer`
//!    default off (see the root/runtime/facade/serve `Cargo.toml` comments
//!    next to that flip for the measured numbers and the decision they
//!    support).
//!
//! CJK and emoji runs are deliberately included in the mixed corpus even
//! though they carry no ASCII whitespace: that is exactly the
//! no-natural-break shape that made the pre-fix `bpe_merge_symbols`
//! quadratic (TOK-06 measured 800 B / 15 ms -> 6400 B / 961 ms on a
//! whitespace-free emoji run).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxibonsai_tokenizer::{BpeTrainer, OxiTokenizer, TrainerConfig, Vocabulary};

/// One paragraph mixing all four corpus classes T-15 asks for: ASCII prose,
/// Japanese (CJK), emoji, and a small source-code snippet.
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

/// Repeat [`mixed_paragraph`] until the result is at least `target_bytes`
/// long (the final chunk is not truncated, so the real length can exceed
/// `target_bytes` slightly; benchmarks report throughput against the real
/// length, not the nominal target).
fn corpus_of_size(target_bytes: usize) -> String {
    let unit = mixed_paragraph();
    let mut out = String::with_capacity(target_bytes + unit.len());
    while out.len() < target_bytes {
        out.push_str(&unit);
    }
    out
}

/// Train a small-but-real BPE tokenizer on a multi-document corpus built
/// from repeated [`mixed_paragraph`]s.
///
/// Training (not just hand-inserting a couple of vocab entries) matters
/// here: TOK-06 is a defect in the merge-application loop, and an
/// untrained/degenerate vocabulary with no real merges would make every
/// benchmark measure only the byte-fallback path, never the merge loop the
/// regression guard exists to protect.
fn build_tokenizer() -> OxiTokenizer {
    let docs: Vec<String> = (0..64).map(|_| mixed_paragraph()).collect();
    let corpus: Vec<&str> = docs.iter().map(String::as_str).collect();
    let mut trainer = BpeTrainer::new(TrainerConfig::new(2048));
    trainer
        .train(&corpus)
        .expect("training a small, well-formed corpus should never fail")
        .to_oxi_tokenizer()
}

const SIZES: &[usize] = &[1024, 4096, 16384, 65536];

fn bench_encode_throughput(c: &mut Criterion) {
    let tokenizer = build_tokenizer();
    let mut group = c.benchmark_group("tokenizer_encode_throughput");
    for &size in SIZES {
        let text = corpus_of_size(size);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &text, |b, text| {
            b.iter(|| {
                let ids = tokenizer
                    .encode(black_box(text))
                    .expect("encode of well-formed UTF-8 should never fail");
                black_box(ids)
            });
        });
    }
    group.finish();
}

fn bench_decode_throughput(c: &mut Criterion) {
    let tokenizer = build_tokenizer();
    let mut group = c.benchmark_group("tokenizer_decode_throughput");
    for &size in SIZES {
        let text = corpus_of_size(size);
        let ids = tokenizer
            .encode(&text)
            .expect("encode of the decode fixture should never fail");
        group.throughput(Throughput::Elements(ids.len() as u64));
        group.bench_with_input(BenchmarkId::new("decode", size), &ids, |b, ids| {
            b.iter(|| {
                let text = tokenizer
                    .decode(black_box(ids))
                    .expect("decode of ids this tokenizer itself produced should never fail");
                black_box(text)
            });
        });
    }
    group.finish();
}

/// TOK-06 size sweep, mirroring the finding's own repro sizes (800 B..6400
/// B doubling). Criterion does not assert; see this file's module doc for
/// where the actual regression gate lives.
fn bench_tok06_size_sweep(c: &mut Criterion) {
    let tokenizer = build_tokenizer();
    let mut group = c.benchmark_group("tok06_encode_size_sweep");
    for &size in &[800usize, 1600, 3200, 6400] {
        let text = corpus_of_size(size);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &text, |b, text| {
            b.iter(|| {
                let ids = tokenizer
                    .encode(black_box(text))
                    .expect("encode of well-formed UTF-8 should never fail");
                black_box(ids)
            });
        });
    }
    group.finish();
}

/// `Vocabulary::is_special_id` (`crates/oxibonsai-tokenizer/src/vocab.rs`)
/// is a linear scan over the special-token registry, called per decoded id
/// by `oxibonsai_runtime::tokenizer_bridge::native_decode_id_into` /
/// `native_step_decode`. Benchmarked at a realistic special-token count (33,
/// the real Bonsai 2 vocabulary's count) plus smaller/larger counts to show
/// the scaling, for both a miss (id is not special — the common case in a
/// decode stream) and a hit.
///
/// `miss` is the number the RT-ADMIN-CFG blocking-1 / decision D-5 manifest
/// comments (root/runtime/facade/serve `Cargo.toml`) actually rely on: it is
/// `Vocabulary::is_special_id`'s true worst case (`.values().any(..)` must
/// exhaust every entry to conclude "not found"), and its cost is therefore
/// deterministic in `n_special` regardless of hashing. `hit`, by contrast,
/// is included to show a found-case sample, not a worst case: `is_special_id`
/// short-circuits on the first match, but `special_tokens` is a
/// `std::collections::HashMap`, whose default `RandomState` hasher is
/// re-seeded from OS randomness once per process — so which bucket (and
/// therefore which scan position) `hit_id` lands in is not controlled by
/// insertion order and is not reproducible run to run of this binary. Do
/// not read `hit` as a worst-case bound; only `miss` is.
fn bench_special_id_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_id_lookup");
    for &n_special in &[8usize, 33, 128] {
        let mut vocab = Vocabulary::new();
        for i in 0..n_special as u32 {
            vocab.add_special(&format!("<special{i}>"), 100_000 + i);
        }
        let miss_id = 42u32;
        // Not the "first inserted" id read as a deliberate best case: any
        // registered id is an equally arbitrary probe point once hashed
        // (see the module doc above), so this is simply *a* representative
        // hit, not a worst-case one.
        let hit_id = 100_000u32;

        group.bench_with_input(BenchmarkId::new("miss", n_special), &miss_id, |b, &id| {
            b.iter(|| black_box(vocab.is_special_id(black_box(id))));
        });
        group.bench_with_input(BenchmarkId::new("hit", n_special), &hit_id, |b, &id| {
            b.iter(|| black_box(vocab.is_special_id(black_box(id))));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode_throughput,
    bench_decode_throughput,
    bench_tok06_size_sweep,
    bench_special_id_lookup,
);
criterion_main!(benches);
