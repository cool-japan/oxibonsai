//! Synthetic-model CPU↔CUDA batch-prefill→decode parity gate (no external GGUF).
//!
//! This is the regression gate for the CUDA prefill→decode KV-cache handoff bug.
//! Historically, CUDA batch prefill wrote the prompt K/V into a *prefill-private*
//! GPU KV cache while per-token decode read a *different* (`FULL_LAYER_STATE`)
//! cache, so any prompt longer than 16 tokens made decode attend over stale
//! (all-zero) KV and silently corrupt generation (decode logit Δ vs CPU ≈ 7.3).
//! The fix unifies the two caches (`cuda_prefill` delegates to
//! `cuda_full_layer::acquire_kv_cache`) and, for Q1, uploads the per-token
//! `d_pos_seqlen` the fused KV-store needs.
//!
//! Unlike `cuda_ternary_forward_parity.rs` (which needs a multi-GB real GGUF),
//! this test assembles tiny fully-Q1 and fully-ternary GGUFs in memory, so it
//! runs unattended on any CUDA box.  It drives a >16-token prompt through
//! `forward_prefill` (which routes multi-token batches to the fused CUDA prefill
//! path) then several decode steps, and asserts the CUDA (`KernelTier::Gpu`)
//! logits and greedy tokens match the scalar CPU reference (`KernelTier::Reference`).
//!
//! Gracefully skips (passes) when no CUDA device is present.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;

// ── Synthetic model dimensions (CUDA-friendly: k=hidden multiple of 128,
//    head_dim=64, all weight tensors a multiple of 128 weights). ───────────────
const MAX_SEQ: usize = 256;
const H: usize = 128; // hidden_size (= nq * head_dim, multiple of 128 for g128)
const INTER: usize = 256; // intermediate_size (multiple of 128)
const NUM_LAYERS: usize = 2;
const NQ: usize = 2;
const NKV: usize = 1;
const HD: usize = 64; // head_dim
const VOCAB: usize = 128;

/// Returns `true` when a CUDA device is accessible.
fn cuda_available() -> bool {
    oxibonsai_kernels::CudaGraph::global().is_ok()
}

/// Deterministic FP32 tensor whose values vary with the index.
fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    for i in 0..n {
        let phase = (i as f32) * 0.013_f32;
        let val = scale * (1.0_f32 + 0.25_f32 * phase.sin());
        v.extend_from_slice(&val.to_le_bytes());
    }
    v
}

/// Build a `TQ2_0_g128` weight blob (34 bytes/block: 32B of 2-bit codes + FP16 scale).
fn tq2_0_g128_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(
        num_weights % 128,
        0,
        "num_weights must be a multiple of 128"
    );
    let num_blocks = num_weights / 128;
    let mut data = Vec::with_capacity(num_blocks * 34);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..num_blocks {
        for _ in 0..32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            data.push((state >> 33) as u8);
        }
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale_f32 = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        data.extend_from_slice(&f16::from_f32(scale_f32).to_le_bytes());
    }
    data
}

/// Build a `Q1_0G128` weight blob (18 bytes/block: FP16 scale + 16B of 128 sign bits).
fn q1_0_g128_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(
        num_weights % 128,
        0,
        "num_weights must be a multiple of 128"
    );
    let num_blocks = num_weights / 128;
    let mut data = Vec::with_capacity(num_blocks * 18);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..num_blocks {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale_f32 = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        data.extend_from_slice(&f16::from_f32(scale_f32).to_le_bytes());
        for _ in 0..16 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            data.push((state >> 33) as u8);
        }
    }
    data
}

/// Emit one quantised projection tensor in the requested format.
fn quant_tensor(
    name: String,
    shape: Vec<u64>,
    num_weights: usize,
    seed: u64,
    ternary: bool,
) -> TensorEntry {
    if ternary {
        TensorEntry {
            name,
            shape,
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(num_weights, seed),
        }
    } else {
        TensorEntry {
            name,
            shape,
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_pattern(num_weights, seed),
        }
    }
}

/// Build a fully-quantised synthetic GGUF (`ternary=true` → TQ2, else Q1).
fn build_synthetic_gguf(ternary: bool) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("CudaPrefillParityTest".to_string()),
    );
    writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(H as u32));
    writer.add_metadata(
        "qwen3.block_count",
        MetadataWriteValue::U32(NUM_LAYERS as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count",
        MetadataWriteValue::U32(NQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count_kv",
        MetadataWriteValue::U32(NKV as u32),
    );
    writer.add_metadata(
        "qwen3.feed_forward_length",
        MetadataWriteValue::U32(INTER as u32),
    );
    writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(VOCAB as u32));
    writer.add_metadata(
        "qwen3.context_length",
        MetadataWriteValue::U32(MAX_SEQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![H as u64, VOCAB as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(VOCAB * H, 0.5),
    });
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![H as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(H, 1.0),
    });
    writer.add_tensor(quant_tensor(
        "output.weight".to_string(),
        vec![H as u64, VOCAB as u64],
        VOCAB * H,
        0xCAFE_BABE,
        ternary,
    ));

    for layer in 0..NUM_LAYERS {
        let pfx = format!("blk.{layer}");
        for (suffix, dim) in [
            ("attn_norm", H),
            ("ffn_norm", H),
            ("attn_q_norm", HD),
            ("attn_k_norm", HD),
        ] {
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.{suffix}.weight"),
                shape: vec![dim as u64],
                tensor_type: TensorType::F32,
                data: f32_pattern(dim, 1.0),
            });
        }
        let s = 0x1000_0000_u64.wrapping_add((layer as u64) << 16);
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_q.weight"),
            vec![H as u64, (NQ * HD) as u64],
            NQ * HD * H,
            s,
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_k.weight"),
            vec![H as u64, (NKV * HD) as u64],
            NKV * HD * H,
            s.wrapping_add(1),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_v.weight"),
            vec![H as u64, (NKV * HD) as u64],
            NKV * HD * H,
            s.wrapping_add(2),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_output.weight"),
            vec![(NQ * HD) as u64, H as u64],
            H * NQ * HD,
            s.wrapping_add(3),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.ffn_gate.weight"),
            vec![H as u64, INTER as u64],
            INTER * H,
            s.wrapping_add(4),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.ffn_up.weight"),
            vec![H as u64, INTER as u64],
            INTER * H,
            s.wrapping_add(5),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.ffn_down.weight"),
            vec![INTER as u64, H as u64],
            H * INTER,
            s.wrapping_add(6),
            ternary,
        ));
    }

    writer.to_bytes().expect("GgufWriter::to_bytes")
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Prefill `prompt`, then teacher-force `decode_inputs` at increasing positions.
/// Returns `(prefill_last_logits, per-decode-step logits)`.
fn drive_teacher_forced(
    bytes: &[u8],
    tier: KernelTier,
    prompt: &[u32],
    decode_inputs: &[u32],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let gguf = GgufFile::parse(bytes).expect("parse gguf");
    let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("from_gguf");
    let kernel = KernelDispatcher::with_tier(tier);
    let prefill = model
        .forward_prefill(prompt, 0, &kernel)
        .expect("forward_prefill");
    let mut decode_logits = Vec::with_capacity(decode_inputs.len());
    for (i, &tok) in decode_inputs.iter().enumerate() {
        let lg = model
            .forward(tok, prompt.len() + i, &kernel)
            .expect("decode forward");
        decode_logits.push(lg);
    }
    (prefill, decode_logits)
}

/// Prefill `prompt`, then greedily decode `n_steps` tokens (argmax feedback).
fn drive_greedy(bytes: &[u8], tier: KernelTier, prompt: &[u32], n_steps: usize) -> Vec<u32> {
    let gguf = GgufFile::parse(bytes).expect("parse gguf");
    let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("from_gguf");
    let kernel = KernelDispatcher::with_tier(tier);
    let mut logits = model
        .forward_prefill(prompt, 0, &kernel)
        .expect("forward_prefill");
    let mut out = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let next = argmax(&logits);
        out.push(next);
        logits = model
            .forward(next, prompt.len() + step, &kernel)
            .expect("greedy decode forward");
    }
    out
}

/// Assert two logit vectors agree within a cross-backend tolerance (CPU scalar
/// vs CUDA fp32 compute with an FP16 KV cache).  The broken KV handoff produced
/// Δ ≈ 7.3; this threshold is far below that yet tolerant of FP16-KV noise.
fn assert_logits_close(cpu: &[f32], gpu: &[f32], label: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: logit length mismatch");
    let mut max_abs = 0.0_f32;
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}: non-finite logit at {i}"
        );
        let abs = (a - b).abs();
        let rel = abs / a.abs().max(b.abs()).max(1e-3);
        if abs > max_abs {
            max_abs = abs;
        }
        assert!(
            abs < 0.25 || rel < 0.05,
            "{label}: logit[{i}] CPU={a:.5} CUDA={b:.5} abs={abs:.4} rel={rel:.4}"
        );
    }
    eprintln!("{label}: max_abs logit delta = {max_abs:.5}");
}

fn run_parity(ternary: bool) {
    let name = if ternary { "ternary(TQ2)" } else { "Q1" };
    if !cuda_available() {
        eprintln!("skip {name}: no CUDA device available");
        return;
    }
    let bytes = build_synthetic_gguf(ternary);

    // 20-token prompt (> 16) forces the fused CUDA batch-prefill path rather than
    // the <=16 sequential fast path.
    let prompt: Vec<u32> = (0..20u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let decode_inputs: Vec<u32> = (0..6u32).map(|i| (i * 5 + 11) % VOCAB as u32).collect();

    let (cpu_prefill, cpu_decode) =
        drive_teacher_forced(&bytes, KernelTier::Reference, &prompt, &decode_inputs);
    let (gpu_prefill, gpu_decode) =
        drive_teacher_forced(&bytes, KernelTier::Gpu, &prompt, &decode_inputs);

    // Prefill's last-token logits must agree (validates the batched prefill math).
    assert_logits_close(&cpu_prefill, &gpu_prefill, &format!("{name} prefill-last"));

    // Each decode step attends over the prompt KV — the decisive check for the
    // handoff bug (a stale/zero KV cache would make these diverge wildly).
    for (i, (c, g)) in cpu_decode.iter().zip(gpu_decode.iter()).enumerate() {
        assert_logits_close(c, g, &format!("{name} decode[{i}]"));
    }

    // Greedy sequences (tolerance-free) must be identical.
    let cpu_greedy = drive_greedy(&bytes, KernelTier::Reference, &prompt, 8);
    let gpu_greedy = drive_greedy(&bytes, KernelTier::Gpu, &prompt, 8);
    assert_eq!(
        cpu_greedy, gpu_greedy,
        "{name}: greedy decode sequence CPU vs CUDA mismatch"
    );
    eprintln!("{name}: greedy sequence match = {cpu_greedy:?}");
}

#[test]
fn cuda_synthetic_q1_prefill_decode_parity() {
    run_parity(false);
}

#[test]
fn cuda_synthetic_ternary_prefill_decode_parity() {
    run_parity(true);
}

// ── FP8 (E4M3 / E5M2) prefill→decode fallback parity ─────────────────────────
//
// The FP8 CUDA batch-prefill path (`try_cuda_prefill_with_lm_head_fp8`) writes
// the prompt K/V into a *GPU-private* KV cache (`FP8_PREFILL_STATE` in
// `cuda_fp8_prefill.rs`) that per-token FP8 decode — which attends over
// `self.kv_cache` on the CPU — never reads.  Like the Q4_0/Q8_0/K-quant paths,
// the FP8 batch prefill is therefore disabled by default (guarded by
// `OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL`) so `forward_prefill` falls back to the
// bit-correct sequential per-token path, which populates `self.kv_cache`.
//
// This test proves the fallback two ways:
//
//  1. A deterministic, weight-independent regression gate: after a >16-token
//     GPU-tier prefill, every prompt position must have a non-zero key in the CPU
//     `self.kv_cache` that decode reads.  The guarded fallback (sequential
//     per-token) populates it; the FP8 batch-prefill path (env override) writes
//     only the GPU-private cache and leaves every CPU KV row at zero — verified
//     directly, so the check fails the instant the guard/fallback stops engaging,
//     regardless of how benign the synthetic weights are.
//
//  2. A >16-token prompt followed by several decode steps must produce
//     logits/greedy tokens matching the scalar CPU reference — end-to-end output
//     correctness through the fallback.

/// Build a `F8_E4M3` / `F8_E5M2` weight blob (34 bytes/block: 32 E4M3/E5M2 codes
/// followed by a FP16 scale).  Values are quantised from a deterministic float
/// pattern so the bytes are always valid FP8 codes (never NaN/Inf encodings).
fn fp8_pattern(num_weights: usize, seed: u64, e4m3: bool) -> Vec<u8> {
    assert_eq!(
        num_weights % 32,
        0,
        "num_weights must be a multiple of 32 (QK_FP8)"
    );
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut vals = Vec::with_capacity(num_weights);
    for _ in 0..num_weights {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = ((state >> 33) as u32 as f32) / (u32::MAX as f32); // 0..1
        vals.push((u - 0.5) * 0.5); // small centred weights
    }
    if e4m3 {
        let blocks = oxibonsai_core::BlockFP8E4M3::quantize(&vals).expect("fp8 e4m3 quantize");
        let mut data = Vec::with_capacity(blocks.len() * 34);
        for b in &blocks {
            data.extend_from_slice(&b.qs);
            data.extend_from_slice(&b.d.to_le_bytes());
        }
        data
    } else {
        let blocks = oxibonsai_core::BlockFP8E5M2::quantize(&vals).expect("fp8 e5m2 quantize");
        let mut data = Vec::with_capacity(blocks.len() * 34);
        for b in &blocks {
            data.extend_from_slice(&b.qs);
            data.extend_from_slice(&b.d.to_le_bytes());
        }
        data
    }
}

/// Emit one FP8 projection tensor (E4M3 when `e4m3`, else E5M2).
fn fp8_tensor(
    name: String,
    shape: Vec<u64>,
    num_weights: usize,
    seed: u64,
    e4m3: bool,
) -> TensorEntry {
    TensorEntry {
        name,
        shape,
        tensor_type: if e4m3 {
            TensorType::F8_E4M3
        } else {
            TensorType::F8_E5M2
        },
        data: fp8_pattern(num_weights, seed, e4m3),
    }
}

/// Build a fully-FP8 synthetic GGUF (`e4m3=true` → E4M3, else E5M2).
fn build_synthetic_fp8_gguf(e4m3: bool) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("CudaFp8PrefillParityTest".to_string()),
    );
    writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(H as u32));
    writer.add_metadata(
        "qwen3.block_count",
        MetadataWriteValue::U32(NUM_LAYERS as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count",
        MetadataWriteValue::U32(NQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count_kv",
        MetadataWriteValue::U32(NKV as u32),
    );
    writer.add_metadata(
        "qwen3.feed_forward_length",
        MetadataWriteValue::U32(INTER as u32),
    );
    writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(VOCAB as u32));
    writer.add_metadata(
        "qwen3.context_length",
        MetadataWriteValue::U32(MAX_SEQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![H as u64, VOCAB as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(VOCAB * H, 0.5),
    });
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![H as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(H, 1.0),
    });
    writer.add_tensor(fp8_tensor(
        "output.weight".to_string(),
        vec![H as u64, VOCAB as u64],
        VOCAB * H,
        0xCAFE_BABE,
        e4m3,
    ));

    for layer in 0..NUM_LAYERS {
        let pfx = format!("blk.{layer}");
        for (suffix, dim) in [
            ("attn_norm", H),
            ("ffn_norm", H),
            ("attn_q_norm", HD),
            ("attn_k_norm", HD),
        ] {
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.{suffix}.weight"),
                shape: vec![dim as u64],
                tensor_type: TensorType::F32,
                data: f32_pattern(dim, 1.0),
            });
        }
        let s = 0x1000_0000_u64.wrapping_add((layer as u64) << 16);
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.attn_q.weight"),
            vec![H as u64, (NQ * HD) as u64],
            NQ * HD * H,
            s,
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.attn_k.weight"),
            vec![H as u64, (NKV * HD) as u64],
            NKV * HD * H,
            s.wrapping_add(1),
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.attn_v.weight"),
            vec![H as u64, (NKV * HD) as u64],
            NKV * HD * H,
            s.wrapping_add(2),
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.attn_output.weight"),
            vec![(NQ * HD) as u64, H as u64],
            H * NQ * HD,
            s.wrapping_add(3),
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.ffn_gate.weight"),
            vec![H as u64, INTER as u64],
            INTER * H,
            s.wrapping_add(4),
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.ffn_up.weight"),
            vec![H as u64, INTER as u64],
            INTER * H,
            s.wrapping_add(5),
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.ffn_down.weight"),
            vec![INTER as u64, H as u64],
            H * INTER,
            s.wrapping_add(6),
            e4m3,
        ));
    }

    writer.to_bytes().expect("GgufWriter::to_bytes")
}

/// Prefill `prompt` on the GPU tier and return the layer-0 head-0 key-vector norm
/// at each prompt position, read back from the CPU `self.kv_cache`.
///
/// This is the weight-independent regression gate for the FP8 split-cache bug: the
/// guarded fallback runs the sequential per-token path, which populates
/// `self.kv_cache`, so every norm is non-zero.  If the FP8 batch-prefill path is
/// re-enabled (`OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL=1`), it writes only the
/// GPU-private cache and leaves every CPU KV row at zero — the exact state that
/// makes decode attend over stale/zero prompt KV.
fn gpu_prefill_prompt_kv_norms(bytes: &[u8], prompt: &[u32]) -> Vec<f32> {
    let gguf = GgufFile::parse(bytes).expect("parse gguf");
    let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("from_gguf");
    let kernel = KernelDispatcher::with_tier(KernelTier::Gpu);
    model
        .forward_prefill(prompt, 0, &kernel)
        .expect("forward_prefill");
    let keys = model.kv_cache().keys_for(0, 0, prompt.len());
    (0..prompt.len())
        .map(|p| {
            let row = &keys[p * HD..(p + 1) * HD];
            row.iter().map(|x| x * x).sum::<f32>().sqrt()
        })
        .collect()
}

fn run_fp8_parity(e4m3: bool) {
    let name = if e4m3 { "FP8(E4M3)" } else { "FP8(E5M2)" };
    if !cuda_available() {
        eprintln!("skip {name}: no CUDA device available");
        return;
    }
    let bytes = build_synthetic_fp8_gguf(e4m3);

    // 20-token prompt (> 16) forces the FP8 batch-prefill dispatch; the guard
    // then falls back to the sequential per-token path that populates the CPU KV
    // cache decode reads.
    let prompt: Vec<u32> = (0..20u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let decode_inputs: Vec<u32> = (0..6u32).map(|i| (i * 5 + 11) % VOCAB as u32).collect();

    // Regression gate (weight-independent): after the >16-token GPU-tier prefill,
    // every prompt position must have a non-zero key in the CPU self.kv_cache the
    // per-token FP8 decode reads.  The guarded fallback populates it; the disabled
    // FP8 batch-prefill path would write only its GPU-private cache and leave every
    // row at zero.  This assertion fails the instant the guard stops engaging.
    let kv_norms = gpu_prefill_prompt_kv_norms(&bytes, &prompt);
    for (p, &n) in kv_norms.iter().enumerate() {
        assert!(
            n > 1e-6,
            "{name}: CPU KV cache key at prompt position {p} is zero (norm={n}) — the \
             FP8 batch prefill wrote a GPU-private cache decode never reads; the \
             split-cache guard/fallback is not engaged"
        );
    }
    eprintln!(
        "{name}: all {} prompt KV positions populated in CPU cache",
        kv_norms.len()
    );

    let (cpu_prefill, cpu_decode) =
        drive_teacher_forced(&bytes, KernelTier::Reference, &prompt, &decode_inputs);
    let (gpu_prefill, gpu_decode) =
        drive_teacher_forced(&bytes, KernelTier::Gpu, &prompt, &decode_inputs);

    // Prefill's last-token logits must agree (validates the fallback prefill math).
    assert_logits_close(&cpu_prefill, &gpu_prefill, &format!("{name} prefill-last"));

    // Each decode step attends over the prompt KV — the decisive anti-corruption
    // check.  With the split-cache bug decode would attend over all-zero KV and
    // these would diverge wildly; the fallback keeps them in parity.
    for (i, (c, g)) in cpu_decode.iter().zip(gpu_decode.iter()).enumerate() {
        assert_logits_close(c, g, &format!("{name} decode[{i}]"));
    }

    // Greedy sequences (tolerance-free) must be identical.
    let cpu_greedy = drive_greedy(&bytes, KernelTier::Reference, &prompt, 8);
    let gpu_greedy = drive_greedy(&bytes, KernelTier::Gpu, &prompt, 8);
    assert_eq!(
        cpu_greedy, gpu_greedy,
        "{name}: greedy decode sequence CPU vs CUDA mismatch"
    );
    eprintln!("{name}: greedy sequence match = {cpu_greedy:?}");
}

#[test]
fn cuda_synthetic_fp8_e4m3_prefill_decode_parity() {
    run_fp8_parity(true);
}

#[test]
fn cuda_synthetic_fp8_e5m2_prefill_decode_parity() {
    run_fp8_parity(false);
}
