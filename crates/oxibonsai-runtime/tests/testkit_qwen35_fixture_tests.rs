//! Loads `oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf` through
//! [`InferenceEngine`] from OUTSIDE `oxibonsai-runtime` (T-07).
//!
//! The testkit fixture is the workspace's one synthetic hybrid (`qwen35`)
//! file: the crate's own seam tests (`engine_seam_tests.rs`) load it too.
//! This file proves it is usable through the public API alone: it loads as
//! a CPU hybrid engine through the exact production constructor
//! (`InferenceEngine::from_gguf_with_backend`) the CLI and server use, runs a
//! real greedy decode step, and embeds — a hybrid model serves embeddings
//! (the model's final-normed hidden state, mean-pooled and L2-normalised) —
//! all reachable without any crate-private item.

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_seam::Backend;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_testkit::qwen35_fixture::{synthetic_qwen35_gguf, HIDDEN};

/// KV window for every engine built here: comfortably above the fixture's
/// prompts, small enough to run in a debug build.
const MAX_SEQ_LEN: usize = 64;

/// A prompt entirely within the fixture's `VOCAB = 512` — arbitrary,
/// deterministic ids, no relation to any real tokenizer.
const PROMPT: [u32; 4] = [7, 11, 13, 17];

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 4,
    }
}

/// A fresh CPU hybrid engine over the testkit fixture.
fn cpu_hybrid_engine<'a>(gguf: &'a GgufFile<'a>) -> InferenceEngine<'a> {
    InferenceEngine::from_gguf_with_backend(gguf, greedy_params(), 42, MAX_SEQ_LEN, Backend::Cpu)
        .expect("the testkit qwen35 fixture must load as a hybrid engine")
}

/// The testkit fixture parses, loads as a hybrid engine pinned to the CPU
/// backend, and both a real greedy step and an embedding work through the
/// ordinary public API — proven from outside `oxibonsai-runtime`.
#[test]
fn testkit_qwen35_fixture_loads_as_a_cpu_hybrid_engine_and_runs_a_greedy_step() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("the real reader must parse the testkit fixture");

    let mut engine = cpu_hybrid_engine(&gguf);

    assert!(
        engine.is_hybrid(),
        "a qwen35 file must load as a hybrid engine"
    );
    assert_eq!(engine.architecture(), "qwen35");
    assert_eq!(engine.backend(), Backend::Cpu);
    assert!(
        !engine.uses_fused_gpu_decode(),
        "`--backend cpu` pins a hybrid to its CPU model: no fused GPU route"
    );

    // A real greedy step runs on the CPU hybrid path.
    let generated = engine
        .generate(&PROMPT, 3)
        .expect("greedy generate must succeed");
    assert!(
        !generated.is_empty(),
        "a greedy step over a real hybrid forward must produce at least one token"
    );

    // A hybrid model embeds, through the same public seam a dense one uses:
    // a finite, unit-length vector of the model's hidden width.
    assert_eq!(engine.embedding_dim(), HIDDEN);
    let vector = engine
        .embed(&PROMPT)
        .expect("a hybrid engine serves embeddings");
    assert_eq!(vector.len(), HIDDEN);
    assert!(vector.iter().all(|v| v.is_finite()));
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "unit norm, got {norm}");
}

/// Determinism guard alongside the greedy step above: `!generated.is_empty()`
/// only proves a step ran, not that the hybrid CPU forward is reproducible.
/// Two independently constructed engines, built from the same fixture bytes
/// and the same seed, must reach byte-identical greedy output — a cheap way
/// to catch a hidden source of nondeterminism (e.g. an uninitialised buffer,
/// or a hash-map iteration order leaking into the forward pass) that a
/// single run cannot.
#[test]
fn testkit_qwen35_fixture_greedy_decode_is_deterministic_across_fresh_engines() {
    let bytes = synthetic_qwen35_gguf();

    let run_once = || -> Vec<u32> {
        let gguf = GgufFile::parse(&bytes).expect("the real reader must parse the testkit fixture");
        let mut engine = cpu_hybrid_engine(&gguf);
        engine
            .generate(&PROMPT, 3)
            .expect("greedy generate must succeed")
    };

    let first = run_once();
    let second = run_once();
    assert_eq!(
        first, second,
        "two fresh engines built from the same testkit fixture bytes and the \
         same seed must produce byte-identical greedy token sequences"
    );
}

/// The embedding twin of the determinism guard: two fresh engines embed the
/// same ids to bit-identical vectors, a generation in between does not
/// change what an engine embeds, and different ids embed differently.
#[test]
fn testkit_qwen35_fixture_embedding_is_deterministic_across_fresh_engines() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("the real reader must parse the testkit fixture");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

    let mut first = cpu_hybrid_engine(&gguf);
    let mut second = cpu_hybrid_engine(&gguf);
    let a = first.embed(&PROMPT).expect("embed");
    let b = second.embed(&PROMPT).expect("embed");
    assert_eq!(bits(&a), bits(&b), "fresh engines must embed identically");

    // A generation leaves recurrent state and KV positions behind; the next
    // embedding must not see them.
    first.generate(&PROMPT, 3).expect("generate");
    let c = first.embed(&PROMPT).expect("embed after a generation");
    assert_eq!(
        bits(&a),
        bits(&c),
        "an embedding must not depend on what the engine generated before"
    );

    let other = first.embed(&PROMPT[..2]).expect("embed a shorter input");
    assert_ne!(bits(&a), bits(&other));
}
