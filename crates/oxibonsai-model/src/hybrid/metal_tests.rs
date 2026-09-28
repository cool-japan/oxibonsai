//! The Metal runner against the CPU [`HybridModel`] on the synthetic
//! `qwen35` fixture: every weight format the kernels decode, the
//! `head_dim = 256` attention path, chunked prefill, the per-layer dump,
//! zero-copy residency, reset and the typed refusals.
//!
//! The CPU reference is built on an explicit CPU kernel tier, so no
//! projection of the reference can itself be routed to the GPU.

use std::sync::Arc;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::gguf::writer::TensorType;
use oxibonsai_kernels::gpu_backend::metal_graph::MetalGraph;
use oxibonsai_kernels::{cpu_kernel_tier, KernelDispatcher};

use super::*;
use crate::hybrid::tests_support::{synthetic_gguf, FixtureOptions, FixtureShape};

/// Prompt the lockstep checks prefill before decoding.
const PROMPT: [u32; 5] = [1, 7, 42, 300, 5];
/// KV window of the fixture models.
const MAX_SEQ: usize = 64;

/// `false` only on a host without a Metal device: on a host with one, a
/// library that fails to build fails the test instead of skipping it.
fn metal_available() -> bool {
    match MetalGraph::global() {
        Ok(_) => true,
        Err(MetalGraphError::DeviceNotFound) => false,
        Err(e) => panic!("the combined Metal library must build on this device: {e}"),
    }
}

fn fixture(shape: FixtureShape, quant: TensorType) -> Vec<u8> {
    synthetic_gguf(
        shape,
        FixtureOptions {
            quant,
            ..FixtureOptions::default()
        },
    )
}

fn cpu_model<'a>(gguf: &'a GgufFile<'a>) -> HybridModel<'a> {
    let config = HybridModel::config_from_gguf(gguf).expect("fixture config");
    let kernel = Arc::new(KernelDispatcher::with_tier(cpu_kernel_tier()));
    HybridModel::from_gguf_with(gguf, config, MAX_SEQ, &kernel).expect("fixture loads")
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

/// `max |a − b| / max |b|`.
fn worst_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{what}[{i}]: {a} vs {b}");
    }
}

/// One logit row of the runner against the CPU's: nearly identical, and
/// the same greedy token unless the CPU's own top-2 is a near-tie.
/// Returns `(cosine, worst relative error)`.
fn check_step(label: &str, step: usize, gpu: &[f32], cpu: &[f32]) -> (f64, f32) {
    let cos = cosine(gpu, cpu);
    let rel = worst_rel(gpu, cpu);
    // Measured on the M3: worst relative error 1.8e-6..3.8e-6 over every
    // format and the head_dim-256 fixture; the bounds leave ~25x headroom.
    assert!(cos >= 0.999_999, "{label} step {step}: logit cosine {cos}");
    assert!(
        rel <= 1e-4,
        "{label} step {step}: worst relative error {rel:e}"
    );
    let (g, c) = (argmax(gpu), argmax(cpu));
    if g != c {
        let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let gap = cpu[c] - cpu[g];
        assert!(
            gap <= 1e-4 * scale,
            "{label} step {step}: greedy token {g} (GPU) vs {c} (CPU), CPU gap {gap:e}"
        );
    }
    (cos, rel)
}

/// Prefill [`PROMPT`] and decode `steps` greedy tokens in lockstep (both
/// fed the CPU's choice), checking every logit row.
fn lockstep(label: &str, cpu: &mut HybridModel<'_>, gpu: &mut HybridMetalRunner<'_>, steps: usize) {
    let vocab = cpu.config().base.vocab_size;
    let (mut cl, mut gl) = (vec![0.0f32; vocab], vec![0.0f32; vocab]);
    cpu.reset();
    gpu.reset();
    cpu.forward_prefill(&PROMPT, 0, &mut cl)
        .expect("cpu prefill");
    gpu.forward_prefill(&PROMPT, 0, &mut gl)
        .expect("metal prefill");
    let (mut worst_cos, mut worst_rel) = check_step(label, 0, &gl, &cl);
    for step in 0..steps {
        let token = u32::try_from(argmax(&cl)).expect("token id");
        let pos = PROMPT.len() + step;
        cpu.forward(token, pos, &mut cl).expect("cpu decode");
        gpu.forward_into(token, pos, &mut gl).expect("metal decode");
        let (cos, rel) = check_step(label, step + 1, &gl, &cl);
        worst_cos = worst_cos.min(cos);
        worst_rel = worst_rel.max(rel);
    }
    eprintln!("{label}: {steps} steps, worst logit cosine {worst_cos:.9}, worst relative error {worst_rel:.3e}");
}

/// Every weight format the Metal GEMVs decode, end to end against the CPU
/// forward: prefill plus eight greedy steps.
#[test]
fn metal_runner_tracks_the_cpu_model_for_every_weight_format_bonsai2() {
    if !metal_available() {
        return;
    }
    for quant in [
        TensorType::PQ2_0,
        TensorType::PTQ1_0,
        TensorType::Q2_0G64,
        TensorType::TQ2_0_g128,
        TensorType::Q1_0G128,
        TensorType::F32,
    ] {
        let bytes = fixture(FixtureShape::default(), quant);
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut cpu = cpu_model(&gguf);
        HybridMetalRunner::check_supported(&cpu).expect("every fixture format is supported");
        let mut gpu = HybridMetalRunner::new(&cpu).expect("runner builds");
        assert!(!gpu.is_mapped());
        assert_eq!(gpu.vocab_size(), cpu.config().base.vocab_size);
        lockstep(&format!("{quant:?}"), &mut cpu, &mut gpu, 8);
    }
}

/// The widened attention path (`head_dim = 256`, the 27B's): partial RoPE
/// over the first `n_rot` dims, the 256-wide score staging and the gate.
#[test]
fn metal_runner_serves_head_dim_256_like_the_cpu_bonsai2() {
    if !metal_available() {
        return;
    }
    let shape = FixtureShape {
        head_dim: 256,
        ..FixtureShape::default()
    };
    let bytes = fixture(shape, TensorType::PQ2_0);
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut cpu = cpu_model(&gguf);
    assert_eq!(cpu.config().base.head_dim, 256);
    let mut gpu = HybridMetalRunner::new(&cpu).expect("runner builds");
    lockstep("head_dim 256", &mut cpu, &mut gpu, 12);
}

/// The residual stream after every layer tracks the CPU's per-layer dump,
/// and the embedding rows (lookup + inverse rotation) are bit-identical.
#[test]
fn metal_runner_layer_dump_tracks_the_cpu_dump_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture(FixtureShape::default(), TensorType::PTQ1_0);
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut cpu = cpu_model(&gguf);
    let mut gpu = HybridMetalRunner::new(&cpu).expect("runner builds");
    let vocab = cpu.config().base.vocab_size;
    let mut cl = vec![0.0f32; vocab];
    let dump = cpu
        .forward_with_dump(&PROMPT, 0, Some(&mut cl))
        .expect("cpu dump");
    assert_bits(
        &gpu.embed(&PROMPT).expect("embed"),
        &dump.embedding,
        "embedding rows",
    );
    let (layers, gl) = gpu.forward_with_dump(&PROMPT, 0).expect("metal dump");
    assert_eq!(layers.len(), dump.layers.len());
    for (i, (g, c)) in layers.iter().zip(&dump.layers).enumerate() {
        let cos = cosine(g, c);
        assert!(cos >= 0.999_999, "layer {i}: residual cosine {cos}");
    }
    check_step("dump", 0, &gl, &cl);
}

/// A prefill split into several GPU calls is bit-identical to feeding the
/// tokens one at a time, and a reset replays the same logits.
#[test]
fn metal_runner_chunked_prefill_is_bitwise_sequential_decode_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture(FixtureShape::default(), TensorType::PQ2_0);
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut cpu = cpu_model(&gguf);
    cpu.set_prefill_chunk(3).expect("chunk");
    let mut gpu = HybridMetalRunner::new(&cpu).expect("runner builds");
    assert_eq!(gpu.max_batch(), 3);
    let tokens: Vec<u32> = (0..8u32).map(|i| (i * 37 + 11) % 512).collect();
    let vocab = gpu.vocab_size();

    let mut chunked = vec![0.0f32; vocab];
    gpu.forward_prefill(&tokens, 0, &mut chunked)
        .expect("chunked prefill");
    gpu.reset();
    let mut sequential = vec![0.0f32; vocab];
    for (pos, &token) in tokens.iter().enumerate() {
        gpu.forward_into(token, pos, &mut sequential)
            .expect("decode");
    }
    assert_bits(
        &chunked,
        &sequential,
        "chunked prefill vs sequential decode",
    );

    gpu.reset();
    let mut replay = vec![0.0f32; vocab];
    gpu.forward_prefill(&tokens, 0, &mut replay)
        .expect("replay");
    assert_bits(&replay, &chunked, "replay after reset");
    assert!(gpu.last_gpu_seconds() > 0.0);
}

/// Built over the file mapping, the runner binds every matrix in place and
/// computes exactly what the copying runner does.
#[test]
fn metal_runner_mapped_weights_are_zero_copy_and_bitwise_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture(FixtureShape::default(), TensorType::PTQ1_0);
    let path = std::env::temp_dir().join(format!(
        "oxibonsai_metal_runner_{}_{:?}.gguf",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::write(&path, &bytes).expect("write fixture");
    {
        let mmap = mmap_gguf_file(&path).expect("mmap fixture");
        let gguf = GgufFile::parse(&mmap).expect("fixture parses");
        let cpu = cpu_model(&gguf);
        let mut mapped = HybridMetalRunner::new_mapped(&cpu, &mmap).expect("mapped runner");
        let mut copied = HybridMetalRunner::new(&cpu).expect("copied runner");
        assert!(mapped.is_mapped());
        assert!(!copied.is_mapped());
        assert_eq!(mapped.weight_bytes(), copied.weight_bytes());
        let vocab = mapped.vocab_size();
        let (mut a, mut b) = (vec![0.0f32; vocab], vec![0.0f32; vocab]);
        mapped.forward_prefill(&PROMPT, 0, &mut a).expect("mapped");
        copied.forward_prefill(&PROMPT, 0, &mut b).expect("copied");
        assert_bits(&a, &b, "mapped vs copied prefill");
        for step in 0..4usize {
            let token = u32::try_from(argmax(&a)).expect("token id");
            let pos = PROMPT.len() + step;
            mapped.forward_into(token, pos, &mut a).expect("mapped");
            copied.forward_into(token, pos, &mut b).expect("copied");
            assert_bits(&a, &b, "mapped vs copied decode");
        }
    }
    std::fs::remove_file(&path).expect("remove fixture");
}

/// Geometry the kernels cannot serve is refused with a typed error before
/// any device work, and calls outside the window or with a short logit
/// buffer are refused like the CPU model refuses them.
#[test]
fn metal_runner_refuses_what_it_cannot_serve_bonsai2() {
    // A 256-wide Gated-DeltaNet key head: the CPU runs it, the GPU
    // recurrence holds at most 128 channels per lane group.
    let wide = FixtureShape {
        state_size: 256,
        ..FixtureShape::default()
    };
    let bytes = fixture(wide, TensorType::PQ2_0);
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cpu = cpu_model(&gguf);
    let err = HybridMetalRunner::check_supported(&cpu).expect_err("head_k_dim 256 is refused");
    assert!(
        matches!(
            err,
            ModelError::Kernel(KernelError::UnsupportedOperation(_))
        ),
        "{err:?}"
    );
    assert!(err.to_string().contains("head_k_dim"), "{err}");
    let built = HybridMetalRunner::new(&cpu);
    assert!(built.is_err(), "construction refuses it too");

    if !metal_available() {
        return;
    }
    let bytes = fixture(FixtureShape::default(), TensorType::PQ2_0);
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cpu = cpu_model(&gguf);
    let mut gpu = HybridMetalRunner::new(&cpu).expect("runner builds");
    let window = gpu.max_seq_len();
    assert!(matches!(
        gpu.forward(1, window),
        Err(ModelError::PositionOutOfRange { .. })
    ));
    let mut short = vec![0.0f32; 3];
    assert!(matches!(
        gpu.forward_into(1, 0, &mut short),
        Err(ModelError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        gpu.forward(1_000_000, 0),
        Err(ModelError::PositionOutOfRange { .. })
    ));
    let mut logits = vec![0.0f32; gpu.vocab_size()];
    assert!(matches!(
        gpu.forward_prefill(&[1, 2, 3], window - 2, &mut logits),
        Err(ModelError::PositionOutOfRange { .. })
    ));
    // An empty prefill is a no-op, as on the CPU.
    gpu.forward_prefill(&[], 0, &mut logits)
        .expect("empty prefill");
}
