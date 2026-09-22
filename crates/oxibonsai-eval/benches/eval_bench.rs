//! BLEU / ROUGE / chrF throughput benchmarks.
//!
//! T-15 (folded from deps-10, RAG-EVAL-IMG-24, TOK-16): the `[[bench]]`
//! entry in `Cargo.toml` used to resolve to a no-op placeholder. This file
//! replaces it with real throughput measurements on a realistic reference
//! set that includes a CJK (Japanese) corpus alongside English — the
//! EVAL-METRICS regression surface these three metrics live on (BLEU's
//! n-gram counting, ROUGE's LCS/n-gram overlap, and chrF's
//! effective-order character n-grams are all sensitive to input length and
//! to non-ASCII, multi-byte-per-`char` text).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxibonsai_eval::{
    chrf, chrf_plus_plus, corpus_bleu, sentence_bleu, BleuConfig, CorpusRouge, RougeLScore,
    RougeNScore, RougeSScore,
};

/// English candidate/reference sentence pairs (machine-translation-style:
/// close but not identical, so n-gram overlap is partial rather than
/// trivially 100%).
const EN_PAIRS: &[(&str, &str)] = &[
    (
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown fox jumped over a lazy dog.",
    ),
    (
        "OxiBonsai is a pure Rust inference engine for ternary language models.",
        "OxiBonsai is a Pure Rust inference engine for 1-bit and ternary language models.",
    ),
    (
        "The kernel dispatcher selects AVX2, AVX-512, or NEON at runtime.",
        "The kernel dispatcher picks AVX2, AVX-512 or NEON depending on the CPU.",
    ),
    (
        "Gated DeltaNet layers recur over a fixed-size state instead of a growing cache.",
        "Gated DeltaNet layers recurse over a small state rather than a growing KV cache.",
    ),
    (
        "The GGUF format stores tensors alongside a key-value metadata block.",
        "GGUF files store tensor data next to a metadata key-value section.",
    ),
];

/// Japanese candidate/reference sentence pairs — the CJK half of the
/// reference set. Character-level metrics (chrF) and word-boundary-free
/// scripts stress code paths that pure-ASCII English pairs cannot.
const JA_PAIRS: &[(&str, &str)] = &[
    (
        "オクシボンサイは純粋なRustで書かれた1ビット推論エンジンです。",
        "オクシボンサイは純粋なRust製の1ビット推論エンジンです。",
    ),
    (
        "量子化されたモデルはメモリ使用量を大幅に削減します。",
        "量子化モデルはメモリ使用量を大きく削減できます。",
    ),
    (
        "カーネルディスパッチャは実行時にAVX2かNEONかを選択します。",
        "カーネルディスパッチャは実行時にAVX2またはNEONを選びます。",
    ),
    (
        "ゲート付きデルタネット層は固定サイズの状態を再帰的に更新します。",
        "ゲーテッドデルタネット層は固定サイズの状態を再帰更新します。",
    ),
    (
        "GGUF形式はテンソルとメタデータを一つのファイルにまとめます。",
        "GGUFフォーマットはテンソルとメタデータを単一ファイルに格納します。",
    ),
];

/// Build a corpus of `n` (candidate, reference) pairs by cycling through
/// the mixed English + Japanese pair pool.
fn corpus_of_len(n: usize) -> Vec<(String, String)> {
    let pool: Vec<(&str, &str)> = EN_PAIRS.iter().chain(JA_PAIRS.iter()).copied().collect();
    (0..n)
        .map(|i| {
            let (c, r) = pool[i % pool.len()];
            (c.to_owned(), r.to_owned())
        })
        .collect()
}

fn bench_sentence_bleu(c: &mut Criterion) {
    let cfg = BleuConfig::default();
    let mut group = c.benchmark_group("sentence_bleu");

    group.bench_function("english", |b| {
        let (candidate, reference) = EN_PAIRS[0];
        b.iter(|| {
            black_box(sentence_bleu(
                black_box(candidate),
                black_box(&[reference]),
                &cfg,
            ))
        });
    });
    group.bench_function("japanese", |b| {
        let (candidate, reference) = JA_PAIRS[0];
        b.iter(|| {
            black_box(sentence_bleu(
                black_box(candidate),
                black_box(&[reference]),
                &cfg,
            ))
        });
    });

    group.finish();
}

fn bench_corpus_bleu(c: &mut Criterion) {
    let cfg = BleuConfig::default();
    let mut group = c.benchmark_group("corpus_bleu");

    for &n in &[10usize, 100, 500] {
        let pairs = corpus_of_len(n);
        let candidates: Vec<&str> = pairs.iter().map(|(c, _)| c.as_str()).collect();
        let references: Vec<Vec<&str>> = pairs.iter().map(|(_, r)| vec![r.as_str()]).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("mixed_en_ja", n), &n, |b, _| {
            b.iter(|| {
                black_box(corpus_bleu(
                    black_box(&candidates),
                    black_box(&references),
                    &cfg,
                ))
            });
        });
    }

    group.finish();
}

fn bench_rouge(c: &mut Criterion) {
    let mut group = c.benchmark_group("rouge_sentence");

    group.bench_function("rouge1_english", |b| {
        let (candidate, reference) = EN_PAIRS[0];
        b.iter(|| {
            black_box(RougeNScore::compute(
                black_box(candidate),
                black_box(reference),
                1,
            ))
        });
    });
    group.bench_function("rouge2_english", |b| {
        let (candidate, reference) = EN_PAIRS[0];
        b.iter(|| {
            black_box(RougeNScore::compute(
                black_box(candidate),
                black_box(reference),
                2,
            ))
        });
    });
    group.bench_function("rouge1_japanese", |b| {
        let (candidate, reference) = JA_PAIRS[0];
        b.iter(|| {
            black_box(RougeNScore::compute(
                black_box(candidate),
                black_box(reference),
                1,
            ))
        });
    });
    group.bench_function("rougeL_english", |b| {
        let (candidate, reference) = EN_PAIRS[0];
        b.iter(|| {
            black_box(RougeLScore::compute(
                black_box(candidate),
                black_box(reference),
            ))
        });
    });
    group.bench_function("rougeL_japanese", |b| {
        let (candidate, reference) = JA_PAIRS[0];
        b.iter(|| {
            black_box(RougeLScore::compute(
                black_box(candidate),
                black_box(reference),
            ))
        });
    });
    group.bench_function("rougeS_english", |b| {
        let (candidate, reference) = EN_PAIRS[0];
        b.iter(|| {
            black_box(RougeSScore::compute(
                black_box(candidate),
                black_box(reference),
            ))
        });
    });

    group.finish();
}

fn bench_corpus_rouge(c: &mut Criterion) {
    let mut group = c.benchmark_group("corpus_rouge");

    for &n in &[10usize, 100, 500] {
        let pairs = corpus_of_len(n);
        let pair_refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(c, r)| (c.as_str(), r.as_str()))
            .collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("mixed_en_ja", n), &n, |b, _| {
            b.iter(|| black_box(CorpusRouge::compute(black_box(&pair_refs))));
        });
    }

    group.finish();
}

fn bench_chrf(c: &mut Criterion) {
    let mut group = c.benchmark_group("chrf_sentence");

    group.bench_function("chrf_english", |b| {
        let (candidate, reference) = EN_PAIRS[0];
        b.iter(|| black_box(chrf(black_box(candidate), black_box(reference))));
    });
    group.bench_function("chrf_japanese", |b| {
        let (candidate, reference) = JA_PAIRS[0];
        b.iter(|| black_box(chrf(black_box(candidate), black_box(reference))));
    });
    group.bench_function("chrf_plus_plus_english", |b| {
        let (candidate, reference) = EN_PAIRS[0];
        b.iter(|| black_box(chrf_plus_plus(black_box(candidate), black_box(reference))));
    });
    group.bench_function("chrf_plus_plus_japanese", |b| {
        let (candidate, reference) = JA_PAIRS[0];
        b.iter(|| black_box(chrf_plus_plus(black_box(candidate), black_box(reference))));
    });

    group.finish();
}

/// chrF at a longer document scale (concatenated pairs), where the
/// `Vec<char>` collection + n-gram hashing cost actually shows up per
/// EVAL-METRICS.
fn bench_chrf_document_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("chrf_document");

    for &n in &[10usize, 100, 500] {
        let pairs = corpus_of_len(n);
        let candidate: String = pairs
            .iter()
            .map(|(c, _)| c.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let reference: String = pairs
            .iter()
            .map(|(_, r)| r.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        group.throughput(Throughput::Bytes(candidate.len() as u64));
        group.bench_with_input(BenchmarkId::new("mixed_en_ja", n), &n, |b, _| {
            b.iter(|| black_box(chrf(black_box(&candidate), black_box(&reference))));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sentence_bleu,
    bench_corpus_bleu,
    bench_rouge,
    bench_corpus_rouge,
    bench_chrf,
    bench_chrf_document_scale,
);
criterion_main!(benches);
