//! MET-02 (Q1 half) regression tests: the fused 1-bit Metal paths key their
//! norm / LM-head buffers on per-load epoch-namespaced slots
//! ([`super::forward_metal::Q1MetalSlots`]), so two different Q1 models in
//! one process can never be served each other's weights, and a model's
//! buffers leave the cache with the model.

use super::forward_metal::Q1MetalSlots;
use super::BonsaiModel;
use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::gpu_backend::metal_full_layer::types::WeightKind;
use oxibonsai_kernels::{KernelDispatcher, KernelTier, MetalGraph};
use std::collections::HashSet;
use std::sync::Arc;

const LAYERS: usize = 2;
const H: usize = 128;
const INTER: usize = 256;
const NQ: usize = 4;
const NKV: usize = 2;
const HD: usize = 32;
const VOCAB: usize = 32;
const CONTEXT: usize = 64;

/// Norm / LM-head slots one `LAYERS`-layer model populates: four norms per
/// layer, the final norm and the LM head.
const SLOTS_PER_MODEL: usize = LAYERS * 4 + 2;

#[test]
fn q1_slots_of_two_loads_are_disjoint_and_tagged() {
    let a = Q1MetalSlots::fresh();
    let b = Q1MetalSlots::fresh();
    assert_ne!(a.epoch(), b.epoch(), "every load mints a fresh epoch");
    let keys_a: HashSet<u64> = a.cache_keys(64).into_iter().map(|(s, _)| s).collect();
    let keys_b: HashSet<u64> = b.cache_keys(64).into_iter().map(|(s, _)| s).collect();
    assert_eq!(
        keys_a.len(),
        64 * 4 + 2,
        "every slot of one model is distinct"
    );
    assert!(
        keys_a.is_disjoint(&keys_b),
        "two models' Q1 slots must never coincide"
    );
    for slot in keys_a.iter().chain(&keys_b) {
        assert!(
            slot >> 63 == 1,
            "tagged: never an address (bit 63 clear) nor a small handle counter"
        );
    }
    // The pre-MET-02 literals are not in the namespace at all.
    for literal in [1_000_000u64, 1_000_013, 2_000_000, 3_000_000] {
        assert!(!keys_a.contains(&literal));
    }
    // The local layout within one model is the old one, shifted.
    assert_eq!(a.norm_base(1) - a.norm_base(0), 10);
    assert_eq!(a.final_norm() - a.norm_base(0), 1_000_000);
    assert_eq!(a.lm_head() - a.final_norm(), 1_000_000);
    let kinds: Vec<WeightKind> = a.cache_keys(1).into_iter().map(|(_, k)| k).collect();
    assert_eq!(
        kinds,
        vec![
            WeightKind::RawF32,
            WeightKind::RawF32,
            WeightKind::RawF32,
            WeightKind::RawF32,
            WeightKind::RawF32,
            WeightKind::Q1Soa
        ]
    );
    // Slots built from one epoch address the same buffers.
    let again = Q1MetalSlots::with_epoch(a.epoch());
    assert_eq!(again.lm_head(), a.lm_head());
    assert_eq!(again.norm_base(7), a.norm_base(7));
    // Never handed to the kernels: releasing is a no-op that never reaches
    // the Metal graph.
    assert!(!a.is_in_use());
    assert_eq!(a.release().expect("an unused release cannot fail"), 0);
}

/// `Q1_0_g128` bytes: an f16 scale then 16 sign bytes per 128 weights.
fn q1_blocks(num_weights: usize, seed: u64) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_weights / 128 * 18);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..num_weights / 128 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        data.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        for _ in 0..16 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            data.push((state >> 33) as u8);
        }
    }
    data
}

fn f32_bytes(n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (scale * (1.0 + 0.25 * ((i as f32) * 0.013).sin())).to_le_bytes())
        .collect()
}

/// A `LAYERS`-layer all-Q1 `qwen3` GGUF whose norms are scaled by
/// `norm_scale` and whose LM head is seeded by `head_seed` — two calls with
/// different values give two models that differ exactly in the buffers
/// MET-02 is about. The geometry is the one the Q1 Metal prefill parity
/// suite runs on.
fn q1_gguf(norm_scale: f32, head_seed: u64) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    w.add_metadata(
        "general.name",
        MetadataWriteValue::Str("Met02Q1".to_string()),
    );
    for (k, v) in [
        ("qwen3.embedding_length", H),
        ("qwen3.block_count", LAYERS),
        ("qwen3.attention.head_count", NQ),
        ("qwen3.attention.head_count_kv", NKV),
        ("qwen3.feed_forward_length", INTER),
        ("qwen3.vocab_size", VOCAB),
        ("qwen3.context_length", CONTEXT),
    ] {
        w.add_metadata(k, MetadataWriteValue::U32(v as u32));
    }
    w.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    w.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));
    let mut add = |name: String, shape: Vec<u64>, tensor_type: TensorType, data: Vec<u8>| {
        w.add_tensor(TensorEntry {
            name,
            shape,
            tensor_type,
            data,
        });
    };
    add(
        "token_embd.weight".to_string(),
        vec![H as u64, VOCAB as u64],
        TensorType::F32,
        f32_bytes(H * VOCAB, 0.5),
    );
    add(
        "output_norm.weight".to_string(),
        vec![H as u64],
        TensorType::F32,
        f32_bytes(H, norm_scale),
    );
    add(
        "output.weight".to_string(),
        vec![H as u64, VOCAB as u64],
        TensorType::Q1_0G128,
        q1_blocks(H * VOCAB, head_seed),
    );
    for layer in 0..LAYERS {
        for (name, len) in [
            ("attn_norm", H),
            ("ffn_norm", H),
            ("attn_q_norm", HD),
            ("attn_k_norm", HD),
        ] {
            add(
                format!("blk.{layer}.{name}.weight"),
                vec![len as u64],
                TensorType::F32,
                f32_bytes(len, norm_scale),
            );
        }
        for (name, rows, cols, bump) in [
            ("attn_q", H, NQ * HD, 0u64),
            ("attn_k", H, NKV * HD, 1),
            ("attn_v", H, NKV * HD, 2),
            ("attn_output", NQ * HD, H, 3),
            ("ffn_gate", H, INTER, 4),
            ("ffn_up", H, INTER, 5),
            ("ffn_down", INTER, H, 6),
        ] {
            add(
                format!("blk.{layer}.{name}.weight"),
                vec![rows as u64, cols as u64],
                TensorType::Q1_0G128,
                q1_blocks(rows * cols, 0x2000 + (layer as u64) * 16 + bump),
            );
        }
    }
    w.to_bytes().expect("GgufWriter::to_bytes")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "logit vectors of different lengths");
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

/// Upload `model` and run its fused single-command-buffer Q1 forward at
/// position 0 (one self-contained command buffer: the token's own K/V is the
/// only attention state it reads).
fn fused_logits(model: &mut BonsaiModel<'_>, gpu: &KernelDispatcher, token: u32) -> Vec<f32> {
    model.upload_weights_to_gpu(gpu);
    let mut hidden = vec![0.0f32; model.config().hidden_size];
    model
        .token_embd
        .copy_row(token, &mut hidden)
        .expect("embedding row");
    let mut logits = Vec::new();
    model
        .try_metal_full_forward_with_lm_head(&mut hidden, 0, &mut logits)
        .unwrap_or_else(|e| panic!("the fused Q1 Metal forward must run on this GPU: {e}"));
    logits
}

/// The same token through the model's CPU forward — an independent
/// reference that never touches the fused path's slots.
fn cpu_logits(gguf: &GgufFile<'_>, token: u32) -> Vec<f32> {
    let cpu = KernelDispatcher::with_tier(KernelTier::Reference);
    let mut model = BonsaiModel::from_gguf(gguf, CONTEXT).expect("CPU reference loads");
    model.forward(token, 0, &cpu).expect("CPU forward")
}

/// The MET-02 collision, end to end on the GPU, with its sensitivity
/// control and the slot lifetime.
///
/// Runs in a **private** Metal graph bound to this thread — its own weight
/// cache, device KV cache and queue ([`MetalGraph::new`]) — so nothing it
/// uploads perturbs the process-default session other lib tests share
/// (`gpu_cache`'s tests assert on that session's global cached-weight
/// count), and the cache-count deltas asserted here are exact.
#[test]
fn two_different_q1_models_never_share_fused_metal_weights() {
    let Ok(gpu) = KernelDispatcher::try_with_tier(KernelTier::Gpu) else {
        eprintln!("skip: no accelerated GPU backend on this host");
        return;
    };
    let graph = match MetalGraph::new() {
        Ok(graph) => Arc::new(graph),
        Err(e) => {
            eprintln!("skip: no Metal device: {e}");
            return;
        }
    };
    let _private = MetalGraph::bind_scope(Arc::clone(&graph));
    let count = || graph.cached_weight_count().expect("cached weight count");

    let bytes_a = q1_gguf(1.0, 0xA11CE);
    let bytes_b = q1_gguf(2.5, 0xB0B);
    let gguf_a = GgufFile::parse(&bytes_a).expect("model A parses");
    let gguf_b = GgufFile::parse(&bytes_b).expect("model B parses");
    let token = 3u32;
    let cpu_a = cpu_logits(&gguf_a, token);
    let cpu_b = cpu_logits(&gguf_b, token);
    assert!(
        cosine(&cpu_a, &cpu_b) < 0.9,
        "the two fixtures must be genuinely different models"
    );

    // (1) Two different Q1 models in one process: each is served its own
    // norms and LM head. With the pre-MET-02 literal slots model B hit
    // model A's buffers here.
    let mut model_a = BonsaiModel::from_gguf(&gguf_a, CONTEXT).expect("model A loads");
    let mut model_b = BonsaiModel::from_gguf(&gguf_b, CONTEXT).expect("model B loads");
    let gpu_a = fused_logits(&mut model_a, &gpu, token);
    let gpu_b = fused_logits(&mut model_b, &gpu, token);
    for (name, gpu_logits, cpu) in [("A", &gpu_a, &cpu_a), ("B", &gpu_b, &cpu_b)] {
        let cos = cosine(gpu_logits, cpu);
        assert!(
            cos > 0.999,
            "model {name}: fused Metal logits disagree with its own CPU forward (cos {cos:.6})"
        );
    }

    // (2) Sensitivity control: the same B weights forced onto A's epoch
    // reproduce the old collision — the check above can fail.
    let mut model_c = BonsaiModel::from_gguf(&gguf_b, CONTEXT).expect("model C loads");
    model_c.metal_q1_slots = Q1MetalSlots::with_epoch(model_a.q1_metal_slots().epoch());
    let before_c = count();
    let gpu_c = fused_logits(&mut model_c, &gpu, token);
    let collided = cosine(&gpu_c, &cpu_b);
    assert!(
        collided < 0.9,
        "control: sharing A's slots must serve A's norms and LM head (cos {collided:.6})"
    );
    let added_by_c = count() - before_c;
    assert_eq!(
        added_by_c,
        LAYERS * 4,
        "control: C uploads only its own block weights; every norm / LM-head slot is a hit"
    );

    // (3) Release: exactly the model's own norm / LM-head entries leave the
    // cache, once.
    let before = count();
    assert_eq!(
        model_a.release_q1_metal_slots().expect("release A"),
        SLOTS_PER_MODEL
    );
    assert_eq!(before - count(), SLOTS_PER_MODEL, "A's entries evicted");
    let before = count();
    assert_eq!(
        model_c.release_q1_metal_slots().expect("release C"),
        SLOTS_PER_MODEL,
        "C swept its (A's) slots"
    );
    assert_eq!(count(), before, "already evicted with A: nothing else goes");
    let before = count();
    assert_eq!(
        model_b.release_q1_metal_slots().expect("release B"),
        SLOTS_PER_MODEL
    );
    assert_eq!(before - count(), SLOTS_PER_MODEL, "B's entries evicted");
    assert_eq!(
        model_b.release_q1_metal_slots().expect("second release"),
        0,
        "a second release is a no-op"
    );
    assert!(!model_b.q1_metal_slots().is_in_use());

    // (4) Dropping a model that used the fused path releases its entries —
    // per-load slots must not leak one copy per load/unload cycle.
    let mut model_d = BonsaiModel::from_gguf(&gguf_a, CONTEXT).expect("model D loads");
    let gpu_d = fused_logits(&mut model_d, &gpu, token);
    assert!(cosine(&gpu_d, &cpu_a) > 0.999, "a reload of A computes A");
    assert!(model_d.q1_metal_slots().is_in_use());
    let before = count();
    drop(model_d);
    assert_eq!(
        before - count(),
        SLOTS_PER_MODEL,
        "dropping a Q1 model evicts its norm / LM-head entries"
    );
}
