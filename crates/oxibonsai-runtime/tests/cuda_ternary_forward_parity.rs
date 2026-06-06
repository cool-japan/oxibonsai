//! CUDA ternary (TQ2_0_g128) CPU↔GPU parity gate.
//!
//! The cross-backend determinism guard (`cross_backend_determinism_tests.rs`)
//! only covers Metal; the CUDA ternary path was historically "deferred, needs
//! hw". This test fills that gap on real CUDA hardware: it drives the real
//! ternary GGUF on the scalar CPU reference (`KernelTier::Reference`) and the
//! CUDA GPU path (`KernelTier::Gpu`) and asserts identical greedy output.
//!
//! It is what surfaced the prefill→decode KV-cache handoff bug: with the CUDA
//! TQ2 batch prefill enabled, prompts longer than ~16 tokens diverged from CPU
//! by a large margin (decode logit Δ ≈ 7.3) because batch prefill writes the
//! prompt KV into a prefill-private cache that the per-token decode path never
//! reads. With that path disabled (the fix), the sequential per-token prefill
//! shares the decode KV cache and CPU↔CUDA agree (decode Δ ≈ 0.002).
//!
//! Ignored by default (needs the multi-GB model + a CUDA GPU). Run with:
//!   OXI_MODEL=/abs/path/Ternary-Bonsai-8B.gguf \
//!     cargo test -p oxibonsai-runtime --features native-cuda \
//!     --test cuda_ternary_forward_parity -- --ignored --nocapture

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;

const MAX_SEQ: usize = 512;

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    }
}

/// Greedily generate `n` tokens from `prompt` on the given kernel tier.
fn run(gguf_bytes: &[u8], tier: KernelTier, prompt: &[u32], n: usize) -> Vec<u32> {
    let gguf = GgufFile::parse(gguf_bytes).expect("parse gguf");
    let model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("from_gguf");
    let mut engine = InferenceEngine::from_model_with_tier(model, tier, greedy_params(), 42);
    engine.generate(prompt, n).expect("generate")
}

fn read_model() -> Option<Vec<u8>> {
    let path = std::env::var("OXI_MODEL")
        .unwrap_or_else(|_| "models/Ternary-Bonsai-8B.gguf".to_string());
    match std::fs::read(&path) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skip: cannot read OXI_MODEL ({path}): {e}");
            None
        }
    }
}

/// Greedy CPU-reference vs CUDA-Gpu output must match on the real ternary model.
/// The prompt length is the regression trigger (the bug appeared above ~16
/// tokens); override the count via `OXI_PROMPT_LEN` (default 20, i.e. >16).
#[test]
#[ignore = "requires real ternary GGUF + CUDA GPU; run with --ignored"]
fn real_ternary_cpu_cuda_parity() {
    let Some(gguf) = read_model() else { return };
    let plen: usize = std::env::var("OXI_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let prompt: Vec<u32> = (0..plen as u32).map(|i| 1000 + i * 37).collect();
    let n = 4;
    let cpu = run(&gguf, KernelTier::Reference, &prompt, n);
    let cuda = run(&gguf, KernelTier::Gpu, &prompt, n);
    eprintln!("── real model (plen={plen}) ──\n  cpu  = {cpu:?}\n  cuda = {cuda:?}");
    let first = cpu
        .iter()
        .zip(cuda.iter())
        .position(|(a, b)| a != b)
        .map(|i| i as i32)
        .unwrap_or(-1);
    assert_eq!(cpu, cuda, "real ternary CPU vs CUDA diverge (first diff at index {first})");
}

/// Diagnostic: magnitude of the CPU-vs-CUDA logit divergence at the first decode
/// step. Benign FP non-associativity is ~1e-2; the KV-handoff bug produced ~7.
/// Useful to distinguish a near-tie argmax flip from genuine corruption.
#[test]
#[ignore = "requires real ternary GGUF + CUDA GPU; run with --ignored"]
fn real_ternary_decode_logit_delta() {
    let Some(gguf) = read_model() else { return };
    let plen: usize = std::env::var("OXI_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let prompt: Vec<u32> = (0..plen as u32).map(|i| 1000 + i * 37).collect();

    let drive = |tier: KernelTier| -> (Vec<f32>, Vec<f32>) {
        let parsed = GgufFile::parse(&gguf).expect("parse");
        let mut model = BonsaiModel::from_gguf(&parsed, MAX_SEQ).expect("from_gguf");
        let kernel = KernelDispatcher::with_tier(tier);
        let p = model.forward_prefill(&prompt, 0, &kernel).expect("prefill");
        let tok0 = p
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap();
        let d = model.forward(tok0, plen, &kernel).expect("decode");
        (p, d)
    };
    let (cpu_p, cpu_d) = drive(KernelTier::Reference);
    let (cuda_p, cuda_d) = drive(KernelTier::Gpu);
    let maxd = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    eprintln!("── real logit delta (plen={plen}) ──");
    eprintln!("  PREFILL max|Δ|={:.4}", maxd(&cpu_p, &cuda_p));
    eprintln!("  DECODE  max|Δ|={:.4}", maxd(&cpu_d, &cuda_d));
    // Decode must stay within FP-noise of the CPU reference.
    assert!(
        maxd(&cpu_d, &cuda_d) < 0.5,
        "decode logit divergence too large: {:.4}",
        maxd(&cpu_d, &cuda_d)
    );
}
