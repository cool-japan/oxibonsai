//! Micro-benchmarks for the OxiBonsai RAG pipeline.
//!
//! The suite exercises four hot paths:
//!
//! - **Indexing throughput** — embedding + vector-store insertion rate.
//! - **Query latency** — end-to-end cost of embedding a query and returning
//!   the top-k chunks.
//! - **`chunk_document` at multi-MiB scale** — the RAG-EVAL-IMG-14 (called
//!   "RAG-14" in the package's addenda) O(n²) regression guard: a prior
//!   version re-scanned each chunk's byte offset from byte 0
//!   (`byte_offset_of_char`), making a 4 MiB document take ~21.7 s to chunk.
//!   `chunk_document` is now O(n) (precomputed `char_indices()` offsets); this
//!   benchmark exists so a future regression to the old shape is visible in
//!   `cargo bench` output long before it becomes a 4 MiB / 21 s surprise.
//! - **Retrieval latency at scale** — `VectorStore::search` at 10k/100k
//!   entries, built directly (bypassing chunking/embedding) so the
//!   benchmark measures the brute-force cosine-similarity scan itself.
//!
//! Indexing/query benchmarks use [`IdentityEmbedder`] so that timings
//! reflect the OxiBonsai code paths rather than third-party embedding
//! backends.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxibonsai_rag::chunker::{chunk_document, ChunkConfig};
use oxibonsai_rag::embedding::IdentityEmbedder;
use oxibonsai_rag::retriever::{Retriever, RetrieverConfig};
use oxibonsai_rag::{Chunk, VectorStore};

const CORPUS: &[&str] = &[
    "Rust is a systems programming language focused on safety and speed.",
    "Python is a high-level interpreted language with dynamic typing.",
    "Go emphasises simplicity, concurrency primitives, and fast compilation.",
    "Haskell is a purely functional language with lazy evaluation semantics.",
    "C++ combines low-level hardware control with high-level abstractions.",
    "Zig aims to be a modern successor to C with explicit control over allocations.",
    "Elixir runs on the BEAM virtual machine and targets fault-tolerant systems.",
    "Kotlin compiles to JVM bytecode and interoperates smoothly with Java.",
];

fn bench_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexing");
    group.throughput(Throughput::Elements(CORPUS.len() as u64));

    group.bench_function("identity_64_dim", |b| {
        b.iter(|| {
            let embedder = IdentityEmbedder::new(64).expect("valid dim");
            let mut retriever = Retriever::new(embedder, RetrieverConfig::default());
            let chunk_cfg = ChunkConfig::default();
            for doc in CORPUS {
                let _ = retriever.add_document(black_box(doc), &chunk_cfg);
            }
            black_box(retriever.chunk_count())
        });
    });

    group.finish();
}

fn bench_query_latency(c: &mut Criterion) {
    let embedder = IdentityEmbedder::new(64).expect("valid dim");
    let mut retriever = Retriever::new(
        embedder,
        RetrieverConfig::default()
            .with_top_k(3)
            .with_min_score(-1.0),
    );
    let chunk_cfg = ChunkConfig::default();
    for doc in CORPUS {
        retriever
            .add_document(doc, &chunk_cfg)
            .expect("seed document");
    }

    let mut group = c.benchmark_group("query_latency");
    group.bench_function("top3", |b| {
        b.iter(|| {
            let results = retriever
                .retrieve(black_box("systems programming language"))
                .expect("retrieve");
            black_box(results.len())
        });
    });
    group.finish();
}

// ─── chunk_document at multi-MiB scale (RAG-EVAL-IMG-14 regression guard) ──

/// Build a `len`-byte-ish synthetic document: repeated sentences (ASCII, so
/// `char_indices()` offsets equal byte offsets — the benchmark is about the
/// chunker's own complexity, not UTF-8 decoding cost) with no long
/// whitespace-free run, matching real prose shape rather than a worst-case
/// adversarial input.
fn synthetic_document(len: usize) -> String {
    const SENTENCE: &str =
        "The quantized transformer streams ternary weights through a SIMD gather-scale kernel. ";
    let mut s = String::with_capacity(len + SENTENCE.len());
    while s.len() < len {
        s.push_str(SENTENCE);
    }
    s
}

fn bench_chunk_document_multi_mib(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_document_multi_mib");
    // Criterion's default sample size (100) would mean a few hundred full
    // passes over the 4 MiB document; O(n) chunking is fast enough for that
    // to be fine, but the whole point of this guard is to still notice
    // quickly if it ever regresses to O(n^2) again, so keep the sample
    // count and measurement time bounded rather than letting criterion's
    // adaptive loop discover a multi-minute-per-sample cost on its own.
    group.sample_size(10);

    for &(label, len) in &[
        ("256KiB", 256 * 1024usize),
        ("1MiB", 1024 * 1024),
        ("4MiB", 4 * 1024 * 1024),
    ] {
        let text = synthetic_document(len);
        let config = ChunkConfig::default();
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new("chunk", label), &text, |b, text| {
            b.iter(|| {
                let chunks = chunk_document(black_box(text), 0, black_box(&config));
                black_box(chunks.len())
            });
        });
    }

    group.finish();
}

// ─── Retrieval latency at scale ─────────────────────────────────────────────

/// Small, dependency-free xorshift PRNG so the benchmark does not need to
/// add a `rand` dependency to this crate's `Cargo.toml` just to fill
/// vectors with non-degenerate values.
fn xorshift_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn random_unit_vector(state: &mut u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| {
            let bits = xorshift_next(state);
            // Map to roughly [-1.0, 1.0]; VectorStore::insert L2-normalizes
            // internally, so the exact scale does not matter. `bits >> 40`
            // is a 24-bit value, so dividing by `1 << 24` alone gives
            // [0, 1) (never negative) -- scale by 2 and re-center so the
            // range is actually signed, as documented.
            ((bits >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Build a [`VectorStore`] with `n` random `dim`-dimensional entries.
fn seeded_vector_store(n: usize, dim: usize) -> VectorStore {
    let mut store = VectorStore::new(dim);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for i in 0..n {
        let vector = random_unit_vector(&mut state, dim);
        let chunk = Chunk::new(format!("chunk {i}"), 0, i, 0);
        store
            .insert(vector, chunk)
            .expect("insert should succeed for a matching-dim vector");
    }
    store
}

fn bench_retrieval_at_scale(c: &mut Criterion) {
    const DIM: usize = 128;
    let mut group = c.benchmark_group("retrieval_at_scale");
    group.sample_size(10);

    for &n in &[10_000usize, 100_000] {
        let store = seeded_vector_store(n, DIM);
        let mut state = 0xC2B2_AE3D_27D4_EB4Fu64;
        let query = random_unit_vector(&mut state, DIM);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("search_top10", n), &query, |b, query| {
            b.iter(|| black_box(store.search(black_box(query), 10)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_indexing,
    bench_query_latency,
    bench_chunk_document_multi_mib,
    bench_retrieval_at_scale,
);
criterion_main!(benches);
