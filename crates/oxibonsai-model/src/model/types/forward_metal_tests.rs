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
    //
    // C is parsed from B's own GGUF, so it is a replica of B's mapping: the
    // scirs2 backend deduplicates C's block uploads onto B's resident handle
    // ids (C's block weights *are* B's buffers), and every norm / LM-head
    // slot, forced onto A's epoch, is a hit on A's buffers — C uploads
    // nothing at all. (Before the scirs2 content dedupe merged, C's block
    // uploads minted fresh handle ids and this delta was `LAYERS * 4`.)
    let mut model_c = BonsaiModel::from_gguf(&gguf_b, CONTEXT).expect("model C loads");
    model_c.metal_q1_slots = Q1MetalSlots::with_epoch(model_a.q1_metal_slots().epoch());
    let before_c = count();
    let gpu_c = fused_logits(&mut model_c, &gpu, token);
    let collided = cosine(&gpu_c, &cpu_b);
    assert!(
        collided < 0.9,
        "control: sharing A's slots must serve A's norms and LM head (cos {collided:.6})"
    );
    for (b, c) in model_b.blocks.iter().zip(&model_c.blocks) {
        assert_eq!(
            c.fused_qkv_gpu_handle().map(|h| h.id()),
            b.fused_qkv_gpu_handle().map(|h| h.id()),
            "dedupe: C's block weights are B's resident buffers"
        );
    }
    let added_by_c = count() - before_c;
    assert_eq!(
        added_by_c, 0,
        "control: C's block weights dedupe onto B's and every norm / LM-head slot is a hit on \
         A's — C uploads nothing"
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
        0,
        "C's (A's) slots already left with A's release: nothing is left under the epoch"
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

    // (4) Dropping the last model of a mapping that used the fused path
    // releases its entries — per-load slots must not leak one copy per
    // load/unload cycle. A reload of A's *bytes* into a fresh buffer is a new
    // mapping (its own epoch), and it is its mapping's only model.
    let bytes_d = bytes_a.clone();
    let gguf_d = GgufFile::parse(&bytes_d).expect("model D parses");
    let mut model_d = BonsaiModel::from_gguf(&gguf_d, CONTEXT).expect("model D loads");
    assert_ne!(
        model_d.q1_metal_slots().epoch(),
        model_a.q1_metal_slots().epoch(),
        "another mapping of the same bytes is another namespace"
    );
    let gpu_d = fused_logits(&mut model_d, &gpu, token);
    assert!(cosine(&gpu_d, &cpu_a) > 0.999, "a reload of A computes A");
    assert!(model_d.q1_metal_slots().is_in_use());
    let before = count();
    drop(model_d);
    assert_eq!(
        before - count(),
        SLOTS_PER_MODEL,
        "dropping a mapping's last Q1 model evicts its norm / LM-head entries"
    );

    // (5) …while dropping a replica whose sibling is still alive evicts
    // nothing: E joins A's mapping (same parsed GGUF), shares its epoch and
    // its buffers, and only the last of the two releases them.
    let mut model_e = BonsaiModel::from_gguf(&gguf_a, CONTEXT).expect("model E loads");
    assert_eq!(
        model_e.q1_metal_slots().epoch(),
        model_a.q1_metal_slots().epoch(),
        "a replica of A's mapping shares A's epoch"
    );
    assert_eq!(model_a.q1_metal_slots().replicas(), 2);
    let gpu_e = fused_logits(&mut model_e, &gpu, token);
    assert!(cosine(&gpu_e, &cpu_a) > 0.999, "a replica of A computes A");
    let before = count();
    drop(model_e);
    assert_eq!(count(), before, "A is still alive: E's drop evicts nothing");
    assert_eq!(model_a.q1_metal_slots().replicas(), 1);
    assert!(model_a.q1_metal_slots().is_in_use());
    drop(model_a);
    assert_eq!(
        before - count(),
        SLOTS_PER_MODEL,
        "the mapping's last replica takes its entries with it"
    );
}

/// Device weight-upload accounting, on the in-crate Q1 replica fixture (a Q1
/// LM head, GPU-tier layer dispatchers, replicas over one leaked weight set):
///
/// 1. a batched prefill (`try_metal_prefill_with_lm_head`) followed by greedy
///    decode (`forward_greedy_gpu`, which builds the cached Q1 weight set)
///    uploads the norms and the LM head **once** — `bytes_uploaded` after the
///    decode equals `bytes_uploaded` after the prefill;
/// 2. a second replica of the same weights joins the first's namespace (same
///    epoch, same slots) and prefills + decodes without uploading a byte,
///    producing the same token;
/// 3. dropping one replica keeps the buffers for the other, and dropping the
///    last frees exactly the namespace's norm / LM-head entries.
///
/// The regression this guards, measured on the real Bonsai-8B before the
/// shared-slot fix: the prefill keyed the LM head on the epoch slots and the
/// cached greedy builder on the old literals — +84.5 MiB within one replica —
/// and every replica minted its own epoch (+88.6 MB per extra replica). Runs
/// in a private Metal graph so every count here is exact.
#[test]
fn q1_prefill_then_greedy_decode_uploads_the_lm_head_once_and_replicas_share_it() {
    use super::testing_fixture::Q1ReplicaFixture;

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

    let config = oxibonsai_core::config::Qwen3Config {
        hidden_size: H,
        intermediate_size: INTER,
        num_layers: LAYERS,
        num_attention_heads: NQ,
        num_kv_heads: NKV,
        head_dim: HD,
        value_length: HD,
        vocab_size: VOCAB,
        max_context_length: CONTEXT,
        ..oxibonsai_core::config::Qwen3Config::tiny_test()
    };
    let fixture = Q1ReplicaFixture::new(config, Arc::new(gpu), 0x5EED);
    let prompt: Vec<u32> = vec![3, 17, 5, 29, 11];
    let decode_pos = prompt.len();

    // ── Replica 1: prefill, then greedy decode ──
    let mut first = fixture.replica().expect("replica 1");
    first.upload_weights_to_gpu(&KernelDispatcher::auto_detect());
    assert!(first.q1_metal_slots().is_shared_mapping());
    let bytes_start = graph.bytes_uploaded();
    let logits = first
        .try_metal_prefill_with_lm_head(&prompt, 0)
        .unwrap_or_else(|e| panic!("the fused Q1 prefill must run on this GPU: {e}"));
    assert_eq!(logits.len(), fixture.config().vocab_size);
    let after_prefill = graph.bytes_uploaded();
    assert!(
        after_prefill > bytes_start,
        "the prefill uploaded the model"
    );
    let lm_head_slot = first.q1_metal_slots().lm_head();
    let epoch = first.q1_metal_slots().epoch();
    let lm_head_key = oxibonsai_kernels::gpu_backend::metal_full_layer::types::WeightKey::new(
        epoch,
        WeightKind::Q1Soa,
        lm_head_slot,
    );
    let lm_head = graph
        .get_or_upload_keyed(lm_head_key, || {
            Err(oxibonsai_kernels::MetalGraphError::ExecutionFailed(
                "probe".into(),
            ))
        })
        .expect("the LM head is resident under the mapping epoch after the prefill");
    let token_1 = first
        .forward_greedy_gpu(prompt[decode_pos - 1], decode_pos)
        .unwrap_or_else(|e| panic!("the cached Q1 greedy decode must run on this GPU: {e}"));
    assert_eq!(
        graph.bytes_uploaded(),
        after_prefill,
        "the greedy decode's cached weight set must bind the prefill's buffers, not upload the \
         LM head and norms a second time"
    );
    {
        let guard = first.gpu_weight_cache.lock().expect("cache lock");
        match guard.as_ref().expect("the greedy decode built the cache") {
            oxibonsai_kernels::CachedModelWeights::Q1(q1) => assert!(
                Arc::ptr_eq(&q1.lm_head, &lm_head),
                "the cached LM head is the prefill's buffer"
            ),
            oxibonsai_kernels::CachedModelWeights::Ternary(_) => {
                panic!("a Q1 model built a ternary cache")
            }
        }
    }

    // ── Replica 2: same weights, same namespace, no upload ──
    let mut second = fixture.replica().expect("replica 2");
    second.upload_weights_to_gpu(&KernelDispatcher::auto_detect());
    assert_eq!(
        second.q1_metal_slots().epoch(),
        epoch,
        "replicas share the epoch"
    );
    assert_eq!(second.q1_metal_slots().lm_head(), lm_head_slot);
    assert_eq!(
        second.q1_metal_slots().norm_base(1),
        first.q1_metal_slots().norm_base(1)
    );
    assert_eq!(first.q1_metal_slots().replicas(), 2);
    let before_second = count();
    let second_logits = second
        .try_metal_prefill_with_lm_head(&prompt, 0)
        .expect("replica 2 prefill");
    let token_2 = second
        .forward_greedy_gpu(prompt[decode_pos - 1], decode_pos)
        .expect("replica 2 greedy decode");
    assert_eq!(
        graph.bytes_uploaded(),
        after_prefill,
        "a second replica of the same weights must not upload a byte"
    );
    assert_eq!(count(), before_second);
    assert_eq!(token_2, token_1, "replicas decode the same token");
    assert_eq!(
        second_logits
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "replicas prefill to the same logits"
    );

    // ── Release: with the last replica, and exactly the namespace ──
    drop(first);
    assert_eq!(
        graph.bytes_uploaded(),
        after_prefill,
        "a surviving replica keeps the shared buffers"
    );
    let before = count();
    drop(second);
    assert_eq!(
        before - count(),
        SLOTS_PER_MODEL,
        "the last replica frees exactly the namespace's norm / LM-head entries"
    );
    drop(lm_head);
}

// ═══════════════════════════════════════════════════════════════════════════
// M-18: the Metal prefill router (sequential route, timeout fallback) and
// the head-free Metal hidden prefill of `forward_hidden`
// ═══════════════════════════════════════════════════════════════════════════

use oxibonsai_kernels::gpu_backend::PrefillRoute;

/// `TQ2_0_g128` bytes (qs first, then the f16 scale), every 2-bit code valid.
fn tq2_blocks(num_weights: usize, seed: u64) -> Vec<u8> {
    let mut lcg = oxibonsai_testkit::gguf_fixture::Lcg::new(seed.wrapping_add(0x5157_4A12));
    let mut data = Vec::with_capacity(num_weights / 128 * 34);
    for _ in 0..num_weights / 128 {
        for _ in 0..32 {
            data.push(lcg.next_valid_tq2_byte());
        }
        let scale = 0.25_f32 + ((lcg.next_u64() >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5;
        data.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
    }
    data
}

/// The [`q1_gguf`] geometry with every projection and the LM head
/// `TQ2_0_g128` — the ternary Metal route.
fn tq2_gguf(head_seed: u64) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    w.add_metadata(
        "general.name",
        MetadataWriteValue::Str("M18Tq2".to_string()),
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
        f32_bytes(H, 1.0),
    );
    add(
        "output.weight".to_string(),
        vec![H as u64, VOCAB as u64],
        TensorType::TQ2_0_g128,
        tq2_blocks(H * VOCAB, head_seed),
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
                f32_bytes(len, 1.0),
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
                TensorType::TQ2_0_g128,
                tq2_blocks(rows * cols, 0x3000 + (layer as u64) * 16 + bump),
            );
        }
    }
    w.to_bytes().expect("GgufWriter::to_bytes")
}

/// A GPU-tier dispatcher wired to a live backend, or `None` (skip).
fn gpu_dispatcher() -> Option<KernelDispatcher> {
    let gpu = KernelDispatcher::auto_detect();
    (gpu.tier() == KernelTier::Gpu).then_some(gpu)
}

/// A private Metal graph bound to this thread, so nothing these tests leave
/// in a session (device KV, parked command buffers) reaches another test.
fn private_metal_graph() -> Option<(Arc<MetalGraph>, oxibonsai_kernels::SessionScope)> {
    let graph = Arc::new(MetalGraph::new().ok()?);
    let scope = MetalGraph::bind_scope(Arc::clone(&graph));
    Some((graph, scope))
}

/// A model made GPU-resident the way a production Metal engine is: the
/// weight upload and the eager fused-weight cache.
fn gpu_ready<'a>(gguf: &'a GgufFile<'a>, gpu: &KernelDispatcher) -> BonsaiModel<'a> {
    let mut model = BonsaiModel::from_gguf(gguf, CONTEXT).expect("fixture loads");
    model.upload_weights_to_gpu(gpu);
    model
        .get_or_create_gpu_cache()
        .unwrap_or_else(|e| panic!("fused weight cache: {e}"));
    model
}

fn logit_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Prefill `prompt` through `forward_prefill`, then decode three greedy
/// steps; every step's logits.
fn prefill_then_decode(
    model: &mut BonsaiModel<'_>,
    gpu: &KernelDispatcher,
    prompt: &[u32],
) -> Vec<Vec<f32>> {
    let mut steps = vec![model.forward_prefill(prompt, 0, gpu).expect("prefill")];
    for step in 0..3 {
        let token = argmax(steps.last().map_or(&[][..], Vec::as_slice));
        steps.push(
            model
                .forward(token, prompt.len() + step, gpu)
                .expect("decode step"),
        );
    }
    steps
}

/// The same, with the prompt fed one `forward` per token — the sequential
/// prefill the M-18 router's sequential route must reproduce exactly.
fn forward_each_then_decode(
    model: &mut BonsaiModel<'_>,
    gpu: &KernelDispatcher,
    prompt: &[u32],
) -> Vec<Vec<f32>> {
    let mut last = Vec::new();
    for (pos, &token) in prompt.iter().enumerate() {
        last = model.forward(token, pos, gpu).expect("forward");
    }
    let mut steps = vec![last];
    for step in 0..3 {
        let token = argmax(steps.last().map_or(&[][..], Vec::as_slice));
        steps.push(
            model
                .forward(token, prompt.len() + step, gpu)
                .expect("decode step"),
        );
    }
    steps
}

const M18_PROMPT: [u32; 12] = [3, 17, 5, 29, 11, 2, 7, 19, 23, 1, 4, 9];

/// M-18 (c): with the route pinned to sequential (the "fused path predicted
/// slower" decision), `forward_prefill` runs no fused batch call and produces
/// exactly what `forward` token by token produces — the prefill logits and
/// the three decode steps after it, bit for bit, on both weight formats.
#[test]
fn forward_prefill_on_the_sequential_route_is_bit_identical_to_forward() {
    let Some(gpu) = gpu_dispatcher() else {
        eprintln!("skip: no accelerated GPU backend on this host");
        return;
    };
    let Some((graph, _scope)) = private_metal_graph() else {
        eprintln!("skip: no Metal device");
        return;
    };
    for (label, bytes) in [("Q1", q1_gguf(1.0, 0xA11CE)), ("TQ2", tq2_gguf(0x7E57))] {
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = gpu_ready(&gguf, &gpu);
        model.force_metal_prefill_route(Some(PrefillRoute::Sequential));
        let fused_before = graph.prefill_run_count();
        let routed = prefill_then_decode(&mut model, &gpu, &M18_PROMPT);
        assert_eq!(
            graph.prefill_run_count(),
            fused_before,
            "{label}: the sequential route must not run the fused batch prefill"
        );
        assert!(
            model.gpu_path_active(),
            "{label}: the device KV latch is set"
        );
        model.force_metal_prefill_route(None);
        model.reset();
        let reference = forward_each_then_decode(&mut model, &gpu, &M18_PROMPT);
        for (step, (a, b)) in routed.iter().zip(&reference).enumerate() {
            assert_eq!(
                logit_bits(a),
                logit_bits(b),
                "{label}: step {step} differs between the sequential route and forward"
            );
        }
    }
}

/// Counts WARN-or-worse events while installed.
struct WarnCounter(Arc<std::sync::atomic::AtomicUsize>);

impl tracing::Subscriber for WarnCounter {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.level() <= &tracing::Level::WARN
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() <= tracing::Level::WARN {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

/// M-18 (a) + (c): a fused prefill that misses its deadline (injected) falls
/// back to the sequential route with exactly one warning, and the logits and
/// the next decode steps are those of a sequential prefill, bit for bit.
#[test]
fn forward_prefill_timeout_falls_back_to_sequential_with_one_warning() {
    let Some(gpu) = gpu_dispatcher() else {
        eprintln!("skip: no accelerated GPU backend on this host");
        return;
    };
    let Some((graph, _scope)) = private_metal_graph() else {
        eprintln!("skip: no Metal device");
        return;
    };
    for (label, bytes) in [("Q1", q1_gguf(1.0, 0xB0B0)), ("TQ2", tq2_gguf(0x0D0D))] {
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = gpu_ready(&gguf, &gpu);
        let reference = forward_each_then_decode(&mut model, &gpu, &M18_PROMPT);
        model.reset();

        let warnings = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fused_before = graph.prefill_run_count();
        MetalGraph::force_prefill_timeouts(1);
        let prefill = tracing::subscriber::with_default(WarnCounter(Arc::clone(&warnings)), || {
            model.forward_prefill(&M18_PROMPT, 0, &gpu)
        });
        MetalGraph::force_prefill_timeouts(0);
        let prefill = prefill.expect("the timeout falls back instead of failing");
        assert_eq!(
            warnings.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "{label}: a missed prefill deadline logs exactly one warning"
        );
        assert_eq!(
            graph.prefill_run_count(),
            fused_before,
            "{label}: the timed-out fused call did not complete"
        );
        assert_eq!(
            logit_bits(&prefill),
            logit_bits(&reference[0]),
            "{label}: the fallback prefill is the sequential prefill"
        );
        let mut last = prefill;
        for (step, want) in reference.iter().enumerate().skip(1) {
            let token = argmax(&last);
            last = model
                .forward(token, M18_PROMPT.len() + step - 1, &gpu)
                .expect("decode after the fallback");
            assert_eq!(
                logit_bits(&last),
                logit_bits(want),
                "{label}: decode step {step} after the fallback"
            );
        }
        let cost = model.metal_prefill_cost();
        assert!(
            !cost.fused_buckets.is_empty(),
            "{label}: the timeout is recorded in the cost model"
        );
    }
}

/// The head-free Metal hidden prefill: `forward_hidden` on a GPU dispatcher
/// takes it (its rows are the direct Metal pass's, the MET-05 latch stays
/// clear) and its rows match the per-token CPU reference — every row to a
/// cosine of at least 0.999, the pooled vector to at least 0.9999 — on both
/// weight formats.
#[test]
fn forward_hidden_on_the_metal_route_matches_the_sequential_reference() {
    let Some(gpu) = gpu_dispatcher() else {
        eprintln!("skip: no accelerated GPU backend on this host");
        return;
    };
    let Some((_graph, _scope)) = private_metal_graph() else {
        eprintln!("skip: no Metal device");
        return;
    };
    let reference_kernel = KernelDispatcher::with_tier(KernelTier::Reference);
    let tokens: Vec<u32> = (0..40u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    for (label, bytes) in [("Q1", q1_gguf(1.0, 0xE3BE)), ("TQ2", tq2_gguf(0xE3BF))] {
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = gpu_ready(&gguf, &gpu);
        let direct = model
            .try_metal_forward_hidden(&tokens, &gpu)
            .expect("the Metal hidden prefill runs")
            .expect("a GPU dispatcher and a resident cache take the Metal route");
        let rows = model.forward_hidden(&tokens, &gpu).expect("forward_hidden");
        // Bit-equal to the direct Metal pass: the batched CPU pass computes
        // the same rows in a different arithmetic order and would not be.
        assert_eq!(
            logit_bits(&rows),
            logit_bits(&direct),
            "{label}: forward_hidden took the Metal route"
        );
        assert!(!model.gpu_path_active(), "{label}: no device-KV latch");
        let reference = model
            .forward_hidden_sequential(&tokens, &reference_kernel)
            .expect("the per-token reference");
        assert_eq!(rows.len(), reference.len());
        for (row, (a, b)) in rows
            .chunks_exact(H)
            .zip(reference.chunks_exact(H))
            .enumerate()
        {
            let cos = cosine(a, b);
            assert!(cos >= 0.999, "{label}: row {row} cos {cos}");
        }
        let pool = |r: &[f32]| {
            BonsaiModel::mean_pool_normalized(r, H, tokens.len(), "pool").expect("pool")
        };
        let pooled = cosine(&pool(&rows), &pool(&reference));
        assert!(pooled >= 0.9999, "{label}: pooled cos {pooled}");
    }
}

/// The Metal hidden prefill declines — `Ok(None)`, nothing run — what it
/// does not serve: a CPU dispatcher, a sliding-window model, a model whose
/// fused weight cache is not resident.
#[test]
fn forward_hidden_metal_route_declines_what_it_does_not_serve() {
    let Some(gpu) = gpu_dispatcher() else {
        eprintln!("skip: no accelerated GPU backend on this host");
        return;
    };
    let Some((_graph, _scope)) = private_metal_graph() else {
        eprintln!("skip: no Metal device");
        return;
    };
    let tokens = [5u32, 6, 7, 8];
    let bytes = tq2_gguf(0xDEC1);
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cold = BonsaiModel::from_gguf(&gguf, CONTEXT).expect("loads");
    assert!(
        cold.try_metal_forward_hidden(&tokens, &gpu)
            .expect("declining is not an error")
            .is_none(),
        "no resident fused weight cache"
    );
    let mut model = gpu_ready(&gguf, &gpu);
    let cpu = KernelDispatcher::with_tier(KernelTier::Reference);
    assert!(model
        .try_metal_forward_hidden(&tokens, &cpu)
        .expect("declining is not an error")
        .is_none());
    model.config.sliding_window = Some(4);
    assert!(model
        .try_metal_forward_hidden(&tokens, &gpu)
        .expect("declining is not an error")
        .is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// Autorelease pools: the dense Metal path does not grow the process
// ═══════════════════════════════════════════════════════════════════════════
//
// `-[MTLCommandQueue commandBuffer]` and
// `-[MTLCommandBuffer computeCommandEncoder]` return autoreleased objects. A
// thread with no autorelease pool keeps every one of them until it exits, so
// a long-lived decode thread that does not drain a pool per forward grows by
// the size of a command buffer and its encoder on every token. The test
// below measures the process footprint across many forwards of every dense
// Metal route (decode, greedy decode, fused batch prefill, speculative
// verify, head-free hidden prefill) and of the per-block host-KV decode a
// model takes when no fused route applies (its projections and LM head go
// through the scirs2-core GPU backend), on both weight formats, in a child
// process of its own so that no other test of this binary allocates while
// it measures.

/// Set in the child process the footprint test re-runs itself in; the child
/// does the measuring.
const FOOTPRINT_CHILD_ENV: &str = "OXIBONSAI_METAL_FOOTPRINT_PROBE";

/// Prefix of the one line the child prints per measured phase.
const FOOTPRINT_REPORT: &str = "metal-footprint:";

/// Growth one measured phase may add. An undrained command buffer and its
/// encoder cost well over 1 KiB per forward, so a phase of
/// [`FOOTPRINT_FORWARDS`] undrained forwards grows by several MiB.
const FOOTPRINT_GROWTH_CEILING: i64 = 1 << 20;

/// Forwards per measured phase (decode: 50 rounds of 31 steps).
const FOOTPRINT_FORWARDS: usize = 1550;

/// Decode steps per decode round of the footprint test.
const FOOTPRINT_DECODE_STEPS: usize = 31;

/// Measured phases: six routes on two weight formats.
const FOOTPRINT_PHASES: usize = 12;

/// `TASK_VM_INFO`, the `task_info` flavor carrying `phys_footprint`.
const TASK_VM_INFO: u32 = 22;

/// `TASK_VM_INFO_REV1_COUNT`: `task_vm_info` up to and including
/// `phys_footprint`, in 32-bit words.
const TASK_VM_INFO_REV1_WORDS: usize = 38;

/// Word offset of `phys_footprint` (a 64-bit field) in `task_vm_info`.
const PHYS_FOOTPRINT_WORD: usize = 36;

// SAFETY (declarations): the Mach `task_info` call and the `mach_task_self_`
// port as `<mach/task.h>` / `<mach/mach_init.h>` declare them; both live in
// libSystem, which every macOS process links.
extern "C" {
    #[link_name = "mach_task_self_"]
    static MACH_TASK_SELF: u32;
    fn task_info(
        target_task: u32,
        flavor: u32,
        task_info_out: *mut i32,
        task_info_out_cnt: *mut u32,
    ) -> i32;
}

/// This process's `phys_footprint` (`task_info(TASK_VM_INFO)`): dirty
/// anonymous memory, compressed pages and device allocations — what the
/// kernel's memory ledger charges the process, clean file pages excluded.
fn phys_footprint_bytes() -> i64 {
    let mut words = [0i32; TASK_VM_INFO_REV1_WORDS];
    let mut count = TASK_VM_INFO_REV1_WORDS as u32;
    // SAFETY: `words` is a caller-owned buffer of `count` 32-bit words, so
    // the kernel writes only inside it; `MACH_TASK_SELF` is initialised by
    // libSystem before `main`.
    let rc = unsafe { task_info(MACH_TASK_SELF, TASK_VM_INFO, words.as_mut_ptr(), &mut count) };
    assert_eq!(rc, 0, "task_info(TASK_VM_INFO) failed: kern_return_t {rc}");
    assert!(
        count as usize >= TASK_VM_INFO_REV1_WORDS,
        "task_info(TASK_VM_INFO) returned {count} words, fewer than rev1's {TASK_VM_INFO_REV1_WORDS}"
    );
    let low = u64::from(words[PHYS_FOOTPRINT_WORD] as u32);
    let high = u64::from(words[PHYS_FOOTPRINT_WORD + 1] as u32);
    let bytes = i64::try_from(low | (high << 32)).expect("footprint fits i64");
    assert!(
        bytes > 0,
        "task_info(TASK_VM_INFO) reported a zero footprint"
    );
    bytes
}

/// `bytes` as signed MiB.
fn mib(bytes: i64) -> String {
    format!("{:+.3} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// Run `warm` untimed units of `unit`, then `units` measured ones; print the
/// phase's report line and return the footprint growth over the measured
/// units.
fn measure_footprint(
    label: &str,
    forwards: usize,
    warm: usize,
    units: usize,
    mut unit: impl FnMut(),
) -> i64 {
    for _ in 0..warm {
        unit();
    }
    let before = phys_footprint_bytes();
    for _ in 0..units {
        unit();
    }
    let after = phys_footprint_bytes();
    let growth = after - before;
    println!(
        "{FOOTPRINT_REPORT} {label}: {forwards} forwards grew the process footprint by {} \
         ({before} -> {after} bytes)",
        mib(growth)
    );
    growth
}

/// Record a phase whose growth passed [`FOOTPRINT_GROWTH_CEILING`].
fn note_footprint(failures: &mut Vec<String>, label: &str, growth: i64) {
    if growth > FOOTPRINT_GROWTH_CEILING {
        failures.push(format!("{label}: {}", mib(growth)));
    }
}

/// The measuring half of
/// [`metal_dense_forwards_do_not_grow_the_process_footprint`], run in the
/// child process: every dense Metal route on both weight formats, each
/// phase warmed up first (pipelines, scratch, allocator high-water marks),
/// then measured.
fn footprint_probe() {
    let gpu = gpu_dispatcher().expect("the parent process saw an accelerated GPU backend");
    let (graph, _scope) = private_metal_graph().expect("the parent process saw a Metal device");
    let hidden_tokens: Vec<u32> = (0..40u32).map(|i| (i * 5 + 1) % VOCAB as u32).collect();
    let rounds = FOOTPRINT_FORWARDS / FOOTPRINT_DECODE_STEPS;
    let mut failures = Vec::new();
    for (format, bytes) in [("Q1", q1_gguf(1.0, 0xF007)), ("TQ2", tq2_gguf(0xF008))] {
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = gpu_ready(&gguf, &gpu);

        // Single-token fused decode + LM head (`forward`), reset every round.
        let label = format!("{format} decode");
        let growth = measure_footprint(&label, FOOTPRINT_FORWARDS, 10, rounds, || {
            model.reset();
            let mut token = 3u32;
            for pos in 0..FOOTPRINT_DECODE_STEPS {
                let logits = model.forward(token, pos, &gpu).expect("fused decode");
                token = argmax(&logits);
            }
            assert!(model.gpu_path_active(), "{format}: decode ran on Metal");
        });
        note_footprint(&mut failures, &label, growth);

        // Greedy fused decode (`forward_greedy_gpu`: argmax on the GPU).
        let label = format!("{format} greedy decode");
        let growth = measure_footprint(&label, FOOTPRINT_FORWARDS, 10, rounds, || {
            model.reset();
            let mut token = 3u32;
            for pos in 0..FOOTPRINT_DECODE_STEPS {
                token = model
                    .forward_greedy_gpu(token, pos)
                    .expect("fused greedy decode");
            }
            assert!(
                model.gpu_path_active(),
                "{format}: greedy decode ran on Metal"
            );
        });
        note_footprint(&mut failures, &label, growth);

        // Fused batch prefill (`forward_prefill` pinned to the fused route).
        model.force_metal_prefill_route(Some(PrefillRoute::Fused));
        let runs_before = graph.prefill_run_count();
        let label = format!("{format} fused prefill");
        let growth = measure_footprint(&label, FOOTPRINT_FORWARDS, 20, FOOTPRINT_FORWARDS, || {
            model.reset();
            model
                .forward_prefill(&M18_PROMPT, 0, &gpu)
                .expect("fused prefill");
        });
        model.force_metal_prefill_route(None);
        assert_eq!(
            graph.prefill_run_count() - runs_before,
            (20 + FOOTPRINT_FORWARDS) as u64,
            "{format}: every prefill ran the fused batch path"
        );
        note_footprint(&mut failures, &label, growth);

        // Batched speculative verify (every row's argmax).
        let runs_before = graph.prefill_run_count();
        let label = format!("{format} verify prefill");
        let growth = measure_footprint(&label, FOOTPRINT_FORWARDS, 20, FOOTPRINT_FORWARDS, || {
            model.reset();
            let ids = model
                .try_metal_prefill_verify(&M18_PROMPT, 0)
                .expect("verify prefill");
            assert_eq!(ids.len(), M18_PROMPT.len());
        });
        assert_eq!(
            graph.prefill_run_count() - runs_before,
            (20 + FOOTPRINT_FORWARDS) as u64,
            "{format}: every verify ran the batched prefill"
        );
        note_footprint(&mut failures, &label, growth);

        // Head-free hidden prefill (the Metal half of `forward_hidden`).
        let label = format!("{format} hidden prefill");
        let growth = measure_footprint(&label, FOOTPRINT_FORWARDS, 20, FOOTPRINT_FORWARDS, || {
            let rows = model
                .try_metal_forward_hidden(&hidden_tokens, &gpu)
                .expect("hidden prefill")
                .expect("a resident fused cache takes the Metal route");
            assert_eq!(rows.len(), hidden_tokens.len() * H);
        });
        note_footprint(&mut failures, &label, growth);

        // Per-block host-KV decode on the GPU tier (`forward`'s fallback when
        // no fused Metal route applies). A declared sliding window turns the
        // fused routes off (they attend over the full cache), so every
        // block's projections and the LM head dispatch through the scirs2-core
        // GPU backend, whose command buffers only `forward` pools. The window
        // is wider than the 31 positions decoded, so it changes no result.
        model.config.sliding_window = Some(CONTEXT);
        let label = format!("{format} block-dispatch decode");
        let growth = measure_footprint(&label, FOOTPRINT_FORWARDS, 10, rounds, || {
            model.reset();
            let mut token = 3u32;
            for pos in 0..FOOTPRINT_DECODE_STEPS {
                let logits = model
                    .forward(token, pos, &gpu)
                    .expect("per-block GPU-tier decode");
                token = argmax(&logits);
            }
            assert!(
                !model.gpu_path_active(),
                "{format}: the block-dispatch decode ran on the host KV cache, not a fused route"
            );
        });
        model.config.sliding_window = None;
        note_footprint(&mut failures, &label, growth);
    }
    assert!(
        failures.is_empty(),
        "dense Metal forwards grew the process footprint past {} per phase — something \
         each forward creates outlives it (an undrained autorelease pool?): {failures:?}",
        mib(FOOTPRINT_GROWTH_CEILING)
    );
}

/// The dense Metal path does not grow the process: 1550 forwards of each
/// route — fused single-token decode, fused greedy decode, fused batch
/// prefill, batched speculative verify, head-free hidden prefill, and the
/// per-block host-KV decode of the scirs2-core fallback — on the Q1 and the
/// TQ2 fixture, weights uploaded, grow the process footprint by at most
/// 1 MiB per route. Every forward's command buffers and encoders are
/// autoreleased objects the Metal path drains per call; left to the thread
/// they cost well over 1 KiB per forward until it exits.
///
/// The measurement runs in a child process running only this test (the
/// test binary re-executed with `--exact`), since other tests of this
/// binary allocate on their own threads meanwhile.
#[test]
fn metal_dense_forwards_do_not_grow_the_process_footprint() {
    if std::env::var_os(FOOTPRINT_CHILD_ENV).is_some() {
        footprint_probe();
        return;
    }
    if gpu_dispatcher().is_none() {
        eprintln!("skip: no accelerated GPU backend on this host");
        return;
    }
    if MetalGraph::new().is_err() {
        eprintln!("skip: no Metal device");
        return;
    }
    let path = module_path!();
    let module = path.split_once("::").map_or(path, |(_, rest)| rest);
    let name = format!("{module}::metal_dense_forwards_do_not_grow_the_process_footprint");
    let exe = std::env::current_exe().expect("the test binary's path");
    let output = std::process::Command::new(exe)
        .args([name.as_str(), "--exact", "--nocapture", "--test-threads=1"])
        .env(FOOTPRINT_CHILD_ENV, "1")
        .output()
        .expect("the footprint probe process starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // `--nocapture` lets the first report share a line with libtest's own
    // `test <name> ... ` prefix, so a report is found anywhere in a line.
    let reports: Vec<&str> = stdout
        .lines()
        .chain(stderr.lines())
        .filter_map(|line| line.find(FOOTPRINT_REPORT).map(|at| &line[at..]))
        .collect();
    for line in &reports {
        eprintln!("{line}");
    }
    assert!(
        output.status.success(),
        "the footprint probe failed ({}):\n{stdout}\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("1 passed"),
        "the probe process ran no test named {name}:\n{stdout}"
    );
    assert_eq!(
        reports.len(),
        FOOTPRINT_PHASES,
        "every measured phase reports once:\n{stdout}\n{stderr}"
    );
}
