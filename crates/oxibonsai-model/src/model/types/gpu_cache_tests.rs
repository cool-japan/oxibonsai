//! Tests for `gpu_cache.rs` — slot identity (MET-02), host-copy freedom and
//! the cached shape (MET-03), no per-call model copy (perf-03), the
//! per-mapping epoch (C1), release on unload and the last-replica release.
//!
//! Split out of `gpu_cache.rs` (compiled as its `#[cfg(test)] #[path]`
//! child module, so `super::*` is unchanged) to keep that file under the
//! 2000-line policy ceiling.

use super::*;
use crate::test_alloc::count_allocations;
use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::gpu_backend::metal_full_layer::types::WeightKey;
use oxibonsai_kernels::{CachedModelWeights, MetalGraphError, MetalWeightHandle};
use std::sync::{Arc, Mutex, MutexGuard};

// ── Synthetic fully-ternary fixture ──────────────────────────────────

/// Hidden size of the fixture (≥ 128: one `TQ2_0_g128` block).
const FIXTURE_HIDDEN: usize = 128;
/// FFN intermediate size of the fixture (multiple of 128).
const FIXTURE_INTER: usize = 256;
/// Transformer layers in the fixture.
const FIXTURE_LAYERS: usize = 2;
/// Attention heads in the fixture.
const FIXTURE_NQ: usize = 4;
/// KV heads in the fixture.
const FIXTURE_NKV: usize = 2;
/// Head dimension of the fixture (`FIXTURE_HIDDEN / FIXTURE_NQ`).
const FIXTURE_HD: usize = 32;
/// Vocabulary size of the fixture.
const FIXTURE_VOCAB: usize = 32;
/// Context length the fixture's models are built with.
const FIXTURE_MAX_SEQ: usize = 64;

/// Build a `TQ2_0_g128` blob that emits only the three ternary codes.
///
/// Blocks are 34 bytes: 32 bytes of 2-bit codes (four per byte, LSB-first)
/// then an f16 scale. The reserved code `0b11` is the `PQ2_0` `+2` encoding
/// and `upload_tq2_weight_soa` rejects any block containing it, so the
/// generator folds each lane into `{0, 1, 2}`.
fn tq2_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(
        num_weights % 128,
        0,
        "num_weights must be a multiple of 128"
    );
    let mut data = Vec::with_capacity(num_weights / 128 * 34);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..num_weights / 128 {
        for _ in 0..32 {
            let mut byte = 0u8;
            for lane in 0..4 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                byte |= (((state >> 33) % 3) as u8) << (2 * lane);
            }
            data.push(byte);
        }
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        data.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
    }
    data
}

/// Build an index-varying FP32 tensor, so nothing degenerates to a constant.
fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    for i in 0..n {
        let val = scale * (1.0_f32 + 0.25_f32 * ((i as f32) * 0.013_f32).sin());
        v.extend_from_slice(&val.to_le_bytes());
    }
    v
}

/// Assemble a synthetic fully-ternary GGUF in memory.
///
/// `salt` perturbs the weight seeds so two fixtures are distinguishable;
/// each `Vec` is its own allocation, so two fixtures also mean two distinct
/// sets of tensor addresses — which is what the slot table keys on.
fn synthetic_ternary_gguf(salt: u64) -> Vec<u8> {
    let (h, inter, nq, nkv, hd, vocab) = (
        FIXTURE_HIDDEN,
        FIXTURE_INTER,
        FIXTURE_NQ,
        FIXTURE_NKV,
        FIXTURE_HD,
        FIXTURE_VOCAB,
    );
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("GpuCacheSlotTest".to_string()),
    );
    writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(h as u32));
    writer.add_metadata(
        "qwen3.block_count",
        MetadataWriteValue::U32(FIXTURE_LAYERS as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count",
        MetadataWriteValue::U32(nq as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count_kv",
        MetadataWriteValue::U32(nkv as u32),
    );
    writer.add_metadata(
        "qwen3.feed_forward_length",
        MetadataWriteValue::U32(inter as u32),
    );
    writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(vocab as u32));
    writer.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
    writer.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(vocab * h, 0.5),
    });
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![h as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(h, 1.0),
    });
    writer.add_tensor(TensorEntry {
        name: "output.weight".to_string(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::TQ2_0_g128,
        data: tq2_pattern(vocab * h, 0xCAFE_BABE ^ salt),
    });

    for layer in 0..FIXTURE_LAYERS {
        let pfx = format!("blk.{layer}");
        for (name, len) in [
            (format!("{pfx}.attn_norm.weight"), h),
            (format!("{pfx}.ffn_norm.weight"), h),
            (format!("{pfx}.attn_q_norm.weight"), hd),
            (format!("{pfx}.attn_k_norm.weight"), hd),
        ] {
            writer.add_tensor(TensorEntry {
                name,
                shape: vec![len as u64],
                tensor_type: TensorType::F32,
                data: f32_pattern(len, 1.0),
            });
        }
        let seed = 0x1000_0000_u64.wrapping_add((layer as u64) << 16) ^ salt;
        for (name, rows, cols, bump) in [
            (format!("{pfx}.attn_q.weight"), h, nq * hd, 0),
            (format!("{pfx}.attn_k.weight"), h, nkv * hd, 1),
            (format!("{pfx}.attn_v.weight"), h, nkv * hd, 2),
            (format!("{pfx}.attn_output.weight"), nq * hd, h, 3),
            (format!("{pfx}.ffn_gate.weight"), h, inter, 4),
            (format!("{pfx}.ffn_up.weight"), h, inter, 5),
            (format!("{pfx}.ffn_down.weight"), inter, h, 6),
        ] {
            writer.add_tensor(TensorEntry {
                name,
                shape: vec![rows as u64, cols as u64],
                tensor_type: TensorType::TQ2_0_g128,
                data: tq2_pattern(rows * cols, seed.wrapping_add(bump)),
            });
        }
    }
    writer.to_bytes().expect("GgufWriter::to_bytes")
}

/// Serialise every GPU-touching test in this binary.
///
/// Since `MET-08` each session has its own KV cache, but the **weight
/// cache** these tests probe and count is still one process-wide store on the
/// shared `MetalDevice` (sharing it is the point: N replicas, 1x weights),
/// and the tail-failure seam they flip is process-global too — so two of them
/// running concurrently would observe each other's uploads and
/// `cached_weight_count()` deltas. A poisoned lock is recovered rather than
/// propagated: one panicking test must not turn every sibling into a second
/// failure.
fn gpu_serial() -> MutexGuard<'static, ()> {
    static GPU_LOCK: Mutex<()> = Mutex::new(());
    GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// Look a slot up **without** being willing to upload it.
///
/// `get_or_upload_keyed` only runs the closure on a miss, so a closure that
/// always fails turns the call into a pure residency probe: `Some` means the
/// buffer is cached under exactly this `(epoch, kind, slot)`, `None` means it
/// is not (or is cached under a different kind).
fn probe(
    graph: &MetalGraph,
    epoch: u64,
    slot: u64,
    kind: WeightKind,
) -> Option<Arc<MetalWeightHandle>> {
    graph
        .get_or_upload_keyed(WeightKey::new(epoch, kind, slot), || {
            Err(MetalGraphError::ExecutionFailed(
                "probe: not resident".into(),
            ))
        })
        .ok()
}

/// Every `(slot, kind)` pair a ternary model owns, layers then tail.
fn all_cache_keys(
    layers: &[TernaryLayerSlots],
    tail: Option<TernaryTailSlots>,
) -> Vec<(u64, WeightKind)> {
    let mut keys: Vec<(u64, WeightKind)> = layers.iter().flat_map(|s| s.cache_keys()).collect();
    if let Some(tail) = tail {
        keys.extend(tail.cache_keys());
    }
    keys
}

// ── Pure tests: the slot table ───────────────────────────────────────

/// The mapped-tensor namespace cannot reach the Q1 path's slots.
///
/// The Q1 fused paths key their norms, final norm and LM head on **tagged**
/// slots — `1 << 63 | epoch << 24 | local`, composed over the model's
/// mapping epoch (`super::super::q1_slots`) — so bit 63 is set on every one
/// of them, while a mapped tensor's slot is a user-space address: at least
/// [`MIN_TENSOR_SLOT`] (macOS `__PAGEZERO`) and never with bit 63 set. The
/// two sets are disjoint by construction, whatever the epoch. (Before MET-02
/// the Q1 slots were the literals `1_000_000 + layer * 10 + off`, `2_000_000`
/// and `3_000_000`, and the ternary fallback path used `2_000_000 +
/// layer * 10` for **its** norms, so a mixed process handed the ternary
/// layer-0 attention norm the Q1 final-norm buffer — same
/// `WeightKind::RawF32`, a silent stale hit rather than an error.)
#[test]
fn mapped_tensor_slots_cannot_reach_the_q1_literal_handles() {
    use super::super::q1_slots::{SlotNamespace, MAX_SLOT_LAYERS, SLOT_TAG};

    // Every Q1 slot of the largest model, under the smallest and the
    // largest epoch, carries the tag.
    for epoch in [0u64, 1, u64::MAX] {
        let ns = SlotNamespace::new(epoch);
        for slot in ns.cuda_keys(2).into_iter().chain([
            ns.norm_base(MAX_SLOT_LAYERS - 1) + 3,
            ns.weight_fallback_base(MAX_SLOT_LAYERS - 1) + 3,
        ]) {
            assert_eq!(slot & SLOT_TAG, SLOT_TAG, "Q1 slot {slot:#x} is untagged");
        }
    }
    // Every mapped-tensor slot of a real model sits in the address range.
    let bytes = synthetic_ternary_gguf(0x5107);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
    let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");
    let (layers, tail) = model.ternary_gpu_slots().expect("derive ternary slots");
    for (slot, _) in all_cache_keys(&layers, tail) {
        assert!(
            (MIN_TENSOR_SLOT..SLOT_TAG).contains(&slot),
            "mapped-tensor slot {slot:#x} left the user-space address range"
        );
    }
    // And the historical literals can never be a tagged slot.
    for literal in [1_000_000u64, 2_000_000, 3_000_000] {
        assert_eq!(literal & SLOT_TAG, 0);
    }
}

/// A slot below the floor is refused rather than keyed on.
#[test]
fn tensor_slot_rejects_an_address_below_the_floor() {
    // SAFETY: `from_raw_parts` with `len == 0` requires only a non-null,
    // aligned pointer — `0x1000` is both for `u8` — and the slice is never
    // dereferenced: `tensor_slot` reads `as_ptr()` and nothing else.
    let below: &[u8] =
        unsafe { std::slice::from_raw_parts(std::ptr::without_provenance(0x1000), 0) };
    let err = tensor_slot(below, "fake_tensor").expect_err("below-floor slot must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("fake_tensor"),
        "error must name the tensor: {msg}"
    );
}

/// Every slot of a real ternary model is distinct, above the floor and
/// stable across repeated derivation — which is what makes the fused path
/// and the fallback path agree (MET-02).
#[test]
fn ternary_slots_are_distinct_stable_and_above_the_floor() {
    let bytes = synthetic_ternary_gguf(0);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
    let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");

    let (layers, tail) = model.ternary_gpu_slots().expect("derive ternary slots");
    assert_eq!(layers.len(), FIXTURE_LAYERS);
    let tail = tail.expect("synthetic fixture has a ternary LM head");

    let keys = all_cache_keys(&layers, Some(tail));
    assert_eq!(keys.len(), FIXTURE_LAYERS * 8 + 2);
    for (slot, _) in &keys {
        assert!(
            *slot >= MIN_TENSOR_SLOT,
            "slot {slot:#x} is below the mapped-tensor floor"
        );
    }
    let mut sorted: Vec<u64> = keys.iter().map(|(slot, _)| *slot).collect();
    sorted.sort_unstable();
    let before = sorted.len();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        before,
        "two ternary weights resolved to the same slot"
    );

    let (layers_again, tail_again) = model.ternary_gpu_slots().expect("re-derive ternary slots");
    assert_eq!(layers, layers_again, "slot derivation must be stable");
    assert_eq!(
        Some(tail),
        tail_again,
        "tail slot derivation must be stable"
    );
}

/// Two models loaded from two GGUFs never share a slot.
///
/// The old per-layer literals (`5_000_000 + layer * 10`, …) were identical
/// for every ternary model in the process, so a draft model and a target
/// model decoded through each other's weights.
#[test]
fn two_models_do_not_share_slots() {
    let bytes_a = synthetic_ternary_gguf(0);
    let bytes_b = synthetic_ternary_gguf(0xABCD);
    let gguf_a = GgufFile::parse(&bytes_a).expect("parse model A");
    let gguf_b = GgufFile::parse(&bytes_b).expect("parse model B");
    let model_a = BonsaiModel::from_gguf(&gguf_a, FIXTURE_MAX_SEQ).expect("load model A");
    let model_b = BonsaiModel::from_gguf(&gguf_b, FIXTURE_MAX_SEQ).expect("load model B");

    let (layers_a, tail_a) = model_a.ternary_gpu_slots().expect("slots A");
    let (layers_b, tail_b) = model_b.ternary_gpu_slots().expect("slots B");
    let keys_a = all_cache_keys(&layers_a, tail_a);
    let keys_b = all_cache_keys(&layers_b, tail_b);
    for key in &keys_a {
        assert!(
            !keys_b.contains(key),
            "slot {:#x} is shared between two independently loaded models",
            key.0
        );
    }
}

// ── GPU tests ────────────────────────────────────────────────────────

/// The ternary GPU cache keeps no host copy of the weights (MET-03).
///
/// `CachedTernaryWeights` used to hold `Vec<Vec<u8>>` copies of every
/// projection for the life of the model — measured at 9.51 GB peak
/// footprint for the 2.18 GB 8B. The weights now live only on the GPU (and
/// in the mapping they were read from, which is clean, file-backed and
/// reclaimable), and the cached value carries nothing but the LM-head row
/// count.
#[test]
fn ternary_gpu_cache_retains_no_host_bytes() {
    let _gpu = gpu_serial();
    let Ok(graph) = MetalGraph::global() else {
        return; // no Metal device in this environment
    };
    let bytes = synthetic_ternary_gguf(0x11);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
    let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");
    model
        .get_or_create_gpu_cache()
        .expect("build the ternary GPU weight cache");

    // MET-03 shape: the cache is eight GPU handles per layer plus the tail —
    // and structurally nothing else. The pre-MET-03 struct carried six
    // `Vec<u8>`/`Vec<Vec<u8>>` host-copy fields (`qkv_concats`,
    // `attn_proj_bytes`, `gate_bytes`, `up_bytes`, `down_bytes`,
    // `lm_head_bytes`) that this test used to prove empty; they no longer
    // exist, so a host copy is now impossible by type, and the size check
    // below pins that: a per-layer entry is exactly eight `Arc`s.
    assert_eq!(
        std::mem::size_of::<oxibonsai_kernels::CachedTernaryLayerWeights>(),
        8 * std::mem::size_of::<Arc<MetalWeightHandle>>(),
        "a cached ternary layer must hold exactly the eight GPU handles"
    );
    let (layers, tail) = model.ternary_gpu_slots().expect("derive ternary slots");
    let tail = tail.expect("synthetic fixture has a ternary LM head");
    // C1: the cache is keyed under the model's own mapping epoch — a real,
    // never-reused epoch, not the legacy one every model used to share.
    let epoch = model.gpu_mapping_epoch();
    assert_ne!(epoch, oxibonsai_kernels::LEGACY_MODEL_EPOCH);
    {
        let guard = model
            .gpu_weight_cache
            .lock()
            .expect("gpu_weight_cache lock");
        match guard.as_ref().expect("cache populated") {
            CachedModelWeights::Ternary(tern) => {
                assert_eq!(tern.model_epoch, epoch);
                assert_eq!(tern.layers.len(), FIXTURE_LAYERS);
                assert_eq!(tern.lm_head_out_features, FIXTURE_VOCAB);
                // Every cached handle IS the resident buffer under its slot:
                // the cache adds no copy of its own.
                for (lw, slots) in tern.layers.iter().zip(&layers) {
                    for (handle, slot, kind) in [
                        (&lw.attn_norm, slots.attn_norm, WeightKind::RawF32),
                        (&lw.q_norm, slots.q_norm, WeightKind::RawF32),
                        (&lw.k_norm, slots.k_norm, WeightKind::RawF32),
                        (&lw.ffn_norm, slots.ffn_norm, WeightKind::RawF32),
                        (&lw.fused_qkv, slots.fused_qkv, WeightKind::Tq2Soa),
                        (&lw.attn_proj, slots.attn_proj, WeightKind::Tq2Soa),
                        (&lw.gate_up, slots.gate_up, WeightKind::Tq2Soa),
                        (&lw.down, slots.down, WeightKind::Tq2Soa),
                    ] {
                        let resident = probe(&graph, epoch, slot, kind)
                            .unwrap_or_else(|| panic!("slot {slot:#x} ({kind}) not resident"));
                        assert!(
                            Arc::ptr_eq(handle, &resident),
                            "the cached handle for slot {slot:#x} is not the resident buffer"
                        );
                        assert_eq!(
                            handle.kind(),
                            kind,
                            "slot {slot:#x} cached under the wrong kind"
                        );
                    }
                }
                let final_norm = tern.final_norm.as_ref().expect("tail: final norm");
                let lm_head = tern.lm_head.as_ref().expect("tail: lm head");
                assert!(Arc::ptr_eq(
                    final_norm,
                    &probe(&graph, epoch, tail.final_norm, WeightKind::RawF32).expect("final norm")
                ));
                assert!(Arc::ptr_eq(
                    lm_head,
                    &probe(&graph, epoch, tail.lm_head, WeightKind::Tq2Soa).expect("lm head")
                ));
            }
            CachedModelWeights::Q1(_) => panic!("ternary model produced a Q1 cache"),
        }
    }

    // The bytes really did go to the GPU: every slot is resident — under the
    // mapping epoch and nowhere else — and the fused QKV buffer is exactly
    // Q‖K‖V long.
    let tail = Some(tail);
    for (slot, kind) in all_cache_keys(&layers, tail) {
        let handle = probe(&graph, epoch, slot, kind)
            .unwrap_or_else(|| panic!("slot {slot:#x} ({kind}) is not GPU-resident"));
        assert!(
            handle.byte_len() > 0,
            "slot {slot:#x} uploaded an empty buffer"
        );
        assert_eq!(
            handle.kind(),
            kind,
            "slot {slot:#x} cached under the wrong kind"
        );
        assert!(
            probe(&graph, oxibonsai_kernels::LEGACY_MODEL_EPOCH, slot, kind).is_none(),
            "slot {slot:#x} ({kind}) is also resident under the legacy epoch"
        );
    }
    let block = &model.blocks[0];
    let expected_qkv = [
        block.attn_q_blocks_ternary().expect("attn_q").len(),
        block.attn_k_blocks_ternary().expect("attn_k").len(),
        block.attn_v_blocks_ternary().expect("attn_v").len(),
    ]
    .iter()
    .sum::<usize>()
        * 34;
    let qkv = probe(&graph, epoch, layers[0].fused_qkv, WeightKind::Tq2Soa).expect("fused qkv");
    assert_eq!(
        qkv.byte_len(),
        expected_qkv,
        "the fused QKV buffer must hold all three projections"
    );

    let _ = model.release_metal_weights();
}

/// The empty [`FUSED_QKV_ALREADY_RESIDENT`] slice fails closed.
///
/// Every ternary path binds `fused_qkv_bytes` to an empty slice because the
/// buffer is known resident by then (the Q‖K‖V concatenation cannot be
/// borrowed from the mapping, and keeping a host copy of it is MET-03).
/// This pins the property that makes that safe: were the residency
/// invariant ever broken, the kernel-side upload **errors** rather than
/// binding a zero-length buffer to the GEMV, so the caller falls back to
/// the CPU instead of reading garbage.
#[test]
fn an_empty_fused_qkv_upload_fails_closed() {
    let _gpu = gpu_serial();
    let Ok(graph) = MetalGraph::global() else {
        return;
    };
    let bytes = synthetic_ternary_gguf(0x55);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
    let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");
    let (layers, _tail) = model.ternary_gpu_slots().expect("derive ternary slots");
    let slot = layers[0].fused_qkv;
    let epoch = model.gpu_mapping_epoch();
    // Start from a genuinely free slot, so the upload cannot be short-cut
    // by a cache hit.
    let _ = model.release_metal_weights();
    assert!(
        probe(&graph, epoch, slot, WeightKind::Tq2Soa).is_none(),
        "slot {slot:#x} must be free for this test to mean anything"
    );

    // The exact lookup the kernels make when a binding's residency
    // prologue was skipped: the model's own key, the empty stand-in bytes.
    graph
        .get_or_upload_tq2_weight_soa_for_epoch(epoch, slot, FUSED_QKV_ALREADY_RESIDENT)
        .expect_err("an empty TQ2 upload must fail, not allocate a zero-length buffer");
    assert!(
        probe(&graph, epoch, slot, WeightKind::Tq2Soa).is_none(),
        "a failed upload must leave nothing behind in the cache"
    );
}

/// Binding the ternary weights allocates a bounded, weight-size-independent
/// amount of host memory (perf-03).
///
/// Every batched prefill call used to rebuild five `Vec<Vec<u8>>` holding
/// the entire quantized model — `5 × n_layers + 1` allocations totalling the
/// whole model, per call, with no memoization. The binding now borrows from
/// the mapping, so the only allocations left are the parameter vector and
/// the slot vector.
#[test]
fn ternary_binding_does_not_copy_the_model_per_call() {
    let _gpu = gpu_serial();
    let Ok(_graph) = MetalGraph::global() else {
        return;
    };
    let bytes = synthetic_ternary_gguf(0x22);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
    let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");
    model.get_or_create_gpu_cache().expect("warm the GPU cache");

    // Warm any lazily-initialised global the first binding would touch, so
    // the measured call sees only its own allocations.
    let warm = model.ternary_gpu_binding().expect("warm-up binding");
    drop(warm);

    let (result, allocations) = count_allocations(|| model.ternary_gpu_binding());
    let binding = result.expect("binding on a warm cache");
    assert_eq!(binding.layer_params.len(), FIXTURE_LAYERS);
    assert!(
        allocations <= 8,
        "binding the ternary weights made {allocations} allocations; it must be a small \
         constant (the parameter vector and the slot vector), not a copy of the model"
    );

    let _ = model.release_metal_weights();
}

/// A tail failure followed by the fallback path must not upload a second
/// copy of the model (MET-02).
///
/// `try_metal_full_forward_with_lm_head_ternary` uploads every layer's
/// weights and only then runs the final-norm → LM-head tail, so a
/// deterministic tail failure leaves the model fully resident and sends
/// `BonsaiModel::forward()` into `try_metal_full_forward_ternary_inner`.
/// That path used to own a different handle namespace, so the fallback
/// doubled GPU residency on the very first token.
///
/// **Assertion (b), the `cached_weight_count()` delta, is the load-bearing
/// one.** A second namespace uploads its duplicate at a *different* slot
/// and leaves the canonical entries untouched, so (a)'s per-slot
/// `Arc::ptr_eq` check passes straight through it — verified by
/// reintroducing a shifted `attn_proj`/`down` namespace, which (a) missed
/// and (b) caught as `left: 22, right: 18`. The price is that (b) is a
/// process-global count: it would also trip if some future test elsewhere
/// in this lib-test binary uploaded a Metal weight concurrently (none does
/// today — `gpu_serial()` covers every GPU test here). Assertions (a) and
/// (c) are immune to that, so a red (b) with green (a)/(c) means look for a
/// concurrent GPU test before suspecting the slot table.
#[test]
fn forced_tail_failure_does_not_duplicate_gpu_weights() {
    let _gpu = gpu_serial();
    let Ok(graph) = MetalGraph::global() else {
        return;
    };
    let bytes = synthetic_ternary_gguf(0x33);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
    let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");

    let (layers, tail) = model.ternary_gpu_slots().expect("derive ternary slots");
    let keys = all_cache_keys(&layers, tail);
    let epoch = model.gpu_mapping_epoch();

    // ── Fused path, with the tail forced to fail after the uploads ──
    let mut hidden = vec![0.05_f32; FIXTURE_HIDDEN];
    let mut logits = Vec::new();
    crate::model::types::forward_metal::set_force_ternary_tail_failure(true);
    let fused = model.try_metal_full_forward_with_lm_head_ternary(&mut hidden, 0, &mut logits);
    crate::model::types::forward_metal::set_force_ternary_tail_failure(false);
    let err = fused.expect_err("the tail failure seam must make the fused path fail");
    assert!(
        err.to_string().contains("OXIBONSAI_FORCE_METAL_TAIL_FAIL"),
        "the failure must come from the seam, not from something else: {err}"
    );

    let before: Vec<Arc<MetalWeightHandle>> = keys
        .iter()
        .map(|(slot, kind)| {
            probe(&graph, epoch, *slot, *kind)
                .unwrap_or_else(|| panic!("slot {slot:#x} not resident after the fused path"))
        })
        .collect();
    let cached_before = graph.cached_weight_count().expect("cached weight count");

    // ── The fallback `forward()` takes on that failure ──────────────
    model
        .try_metal_full_forward_ternary_inner(&mut hidden, 0)
        .expect("the ternary fallback path must run on the already-resident weights");

    // (a) The canonical slots still hold the very same buffers.
    for (i, (slot, kind)) in keys.iter().enumerate() {
        let after = probe(&graph, epoch, *slot, *kind)
            .unwrap_or_else(|| panic!("slot {slot:#x} disappeared during the fallback"));
        assert!(
            Arc::ptr_eq(&before[i], &after),
            "slot {slot:#x} ({kind}) was re-uploaded by the fallback path: the two ternary \
             handle namespaces are back"
        );
    }

    // (b) …and no buffer appeared *anywhere else* either, which is what the
    // old 2M/3M namespace did: it left the 5M/6M entries untouched and
    // uploaded a whole second copy beside them.
    let cached_after = graph.cached_weight_count().expect("cached weight count");
    assert_eq!(
        cached_after,
        cached_before,
        "the ternary fallback path added {} GPU weight buffers; every weight it binds must \
         already be resident under this model's slots",
        cached_after.saturating_sub(cached_before)
    );

    // (c) Structurally: every handle the fallback binds is one of this
    // model's canonical slots, so there is no second namespace to drift
    // into in the first place.
    let canonical: Vec<u64> = keys.iter().map(|(slot, _)| *slot).collect();
    let binding = model.ternary_gpu_binding().expect("rebuild the binding");
    for (i, lp) in binding.layer_params.iter().enumerate() {
        for (name, handle) in [
            ("attn_norm", lp.attn_norm_handle),
            ("q_norm", lp.q_norm_handle),
            ("k_norm", lp.k_norm_handle),
            ("ffn_norm", lp.ffn_norm_handle),
            ("fused_qkv", lp.fused_qkv_handle),
            ("attn_proj", lp.attn_proj_handle),
            ("gate_up", lp.gate_up_handle),
            ("down", lp.down_handle),
        ] {
            assert!(
                canonical.contains(&handle),
                "layer {i} binds {name} to {handle:#x}, which is not one of this model's slots"
            );
        }
    }

    drop(before);
    let _ = model.release_metal_weights();
}

/// Unloading a model releases its GPU buffers.
///
/// Nothing evicts `MetalGraph`'s weight cache on its own, so without this a
/// model that is unloaded keeps its whole quantized self on the GPU — 7.2 GB
/// per load for the 27B `PQ2_0`.
#[test]
fn release_metal_weights_drops_this_models_buffers() {
    let _gpu = gpu_serial();
    let Ok(graph) = MetalGraph::global() else {
        return;
    };
    let bytes = synthetic_ternary_gguf(0x44);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
    let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");
    model
        .get_or_create_gpu_cache()
        .expect("build the GPU cache");

    let (layers, tail) = model.ternary_gpu_slots().expect("derive ternary slots");
    let keys = all_cache_keys(&layers, tail);
    let epoch = model.gpu_mapping_epoch();
    for (slot, kind) in &keys {
        assert!(
            probe(&graph, epoch, *slot, *kind).is_some(),
            "slot {slot:#x} should be resident before release"
        );
    }

    let resident_before = graph.resident_weight_bytes().expect("resident bytes");
    let released = model.release_metal_weights().expect("release this model");
    assert_eq!(released, keys.len(), "every owned slot must be released");

    for (slot, kind) in &keys {
        assert!(
            probe(&graph, epoch, *slot, *kind).is_none(),
            "slot {slot:#x} is still cached after release"
        );
    }
    let resident_after = graph.resident_weight_bytes().expect("resident bytes");
    assert!(
        resident_after < resident_before,
        "releasing {} buffers must lower the resident-byte gauge ({resident_before} → \
         {resident_after})",
        keys.len()
    );

    // The model rebuilds its cache on demand afterwards.
    model
        .get_or_create_gpu_cache()
        .expect("the cache must rebuild after a release");
    for (slot, kind) in &keys {
        assert!(
            probe(&graph, epoch, *slot, *kind).is_some(),
            "slot {slot:#x} should be resident again after a rebuild"
        );
    }
    let _ = model.release_metal_weights();
}

/// Dropping the *last* (here, only) replica sharing a model's ternary GPU
/// buffers releases them — spec 1(c) / ACCEPTANCE "`release_model` drops
/// the handles on Drop" — with no explicit `release_metal_weights()` call
/// anywhere in this test.
#[test]
fn drop_of_the_last_ternary_replica_releases_the_shared_gpu_buffers() {
    let _gpu = gpu_serial();
    let Ok(graph) = MetalGraph::global() else {
        return;
    };
    let bytes = synthetic_ternary_gguf(0x66);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
    let (keys, epoch) = {
        let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load model");
        model.get_or_create_gpu_cache().expect("warm the GPU cache");
        let (layers, tail) = model.ternary_gpu_slots().expect("derive slots");
        let keys = all_cache_keys(&layers, tail);
        let epoch = model.gpu_mapping_epoch();
        assert_eq!(
            model.q1_metal_slots().replicas(),
            1,
            "this model is its mapping's only replica"
        );
        for (slot, kind) in &keys {
            assert!(
                probe(&graph, epoch, *slot, *kind).is_some(),
                "slot {slot:#x} should be resident before drop"
            );
        }
        (keys, epoch)
        // `model` is dropped here, at the end of this block — nothing
        // else calls `release_metal_weights()` in this test.
    };
    for (slot, kind) in &keys {
        assert!(
            probe(&graph, epoch, *slot, *kind).is_none(),
            "slot {slot:#x} is still cached after its only model dropped"
        );
    }
}

/// Dropping one of several replicas that share a model's ternary GPU
/// buffers must NOT evict them out from under the replicas still alive —
/// the failure mode a bare, unconditional `Drop -> release_metal_weights`
/// would reintroduce (a 27B pool re-uploading 7.2 GB mid-serve). Once the
/// last replica drops, the buffers must still be released.
#[test]
fn drop_of_a_non_last_ternary_replica_does_not_evict_the_shared_buffers() {
    let _gpu = gpu_serial();
    let Ok(graph) = MetalGraph::global() else {
        return;
    };
    let bytes = synthetic_ternary_gguf(0x88);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");

    let replica_a = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load replica A");
    let replica_b = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load replica B");
    replica_a.get_or_create_gpu_cache().expect("warm A");
    replica_b.get_or_create_gpu_cache().expect("warm B");

    // Non-vacuous: both replicas must resolve to the very same slots
    // (MET-02's "engine-pool replicas share one upload"), or dropping one
    // trivially wouldn't touch the other's and the rest of this test
    // would pass for the wrong reason.
    let (layers_a, tail_a) = replica_a.ternary_gpu_slots().expect("slots A");
    let (layers_b, tail_b) = replica_b.ternary_gpu_slots().expect("slots B");
    assert_eq!(
        layers_a, layers_b,
        "two replicas of one GgufFile must resolve to the same layer slots"
    );
    assert_eq!(
        tail_a, tail_b,
        "two replicas of one GgufFile must resolve to the same tail slots"
    );
    // …and to the same mapping epoch (C1: one registry entry, two replicas).
    let epoch = replica_a.gpu_mapping_epoch();
    assert_eq!(replica_b.gpu_mapping_epoch(), epoch);
    assert_eq!(replica_a.q1_metal_slots().replicas(), 2);
    let keys = all_cache_keys(&layers_a, tail_a);

    for (slot, kind) in &keys {
        assert!(
            probe(&graph, epoch, *slot, *kind).is_some(),
            "slot {slot:#x} should be resident once both replicas are warm"
        );
    }

    drop(replica_a);

    // Replica B is still alive and shares these exact slots: none of them
    // may have been evicted by A's drop.
    for (slot, kind) in &keys {
        assert!(
            probe(&graph, epoch, *slot, *kind).is_some(),
            "slot {slot:#x} was evicted while a sibling replica was still alive"
        );
    }

    drop(replica_b);

    // Now the last replica is gone: the shared buffers must be released.
    for (slot, kind) in &keys {
        assert!(
            probe(&graph, epoch, *slot, *kind).is_none(),
            "slot {slot:#x} is still cached after the last replica dropped"
        );
    }
}

/// MET-03: the cached decode entry points are bit-identical to the uncached
/// ones — single-token logits, greedy tokens, batched prefill and verify — and
/// no path uploads a buffer beyond the model's own slot table.
///
/// The "uncached" arms are the genuinely uncached references
/// (`*_ternary_uncached`: the per-call binding plus the kernels' lookup-per-
/// weight entry points, keyed under the mapping epoch — the batched ones are
/// exactly what B3 re-keyed), and the model's routing entries
/// (`try_metal_prefill_with_lm_head_ternary` / `…_verify_ternary_path`, now
/// cached, B4) must agree with both.
///
/// Each run gets its **own** Metal session (`MET-08`), so two device KV caches
/// never see each other's writes: position `p` of the cached run attends over
/// what the cached run wrote, exactly as the uncached run did. All sessions sit
/// on one **isolated** device, so the residency count is this test's alone.
#[test]
fn cached_ternary_decode_is_bit_identical_to_the_uncached_path() {
    // Still serialised: the isolated device only privatises the weight cache;
    // the allocation counter a sibling test reads is process-global.
    let _gpu = gpu_serial();
    let Ok(device) = oxibonsai_kernels::MetalDevice::isolated() else {
        return;
    };
    let in_session = |f: &mut dyn FnMut()| {
        let session = MetalGraph::new_session_on(&device);
        MetalGraph::with_session(&session, f);
    };
    let graph = MetalGraph::new_session_on(&device);
    let bytes = synthetic_ternary_gguf(0x99);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
    let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");
    in_session(&mut || model.get_or_create_gpu_cache().expect("warm the GPU cache"));
    let (layers, tail) = model.ternary_gpu_slots().expect("derive ternary slots");
    let owned = all_cache_keys(&layers, tail).len();
    let resident_before = graph.cached_weight_count().expect("count");
    assert_eq!(
        resident_before, owned,
        "the isolated device holds exactly this model"
    );

    let tokens: [u32; 5] = [3, 17, 5, 29, 11];
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();

    let mut uncached: Vec<Vec<f32>> = Vec::new();
    in_session(&mut || {
        for (pos, &t) in tokens.iter().enumerate() {
            uncached.push(
                model
                    .forward_logits_gpu_ternary_uncached(t, pos)
                    .expect("uncached logits"),
            );
        }
    });
    let mut cached: Vec<Vec<f32>> = Vec::new();
    in_session(&mut || {
        for (pos, &t) in tokens.iter().enumerate() {
            cached.push(
                model
                    .forward_logits_gpu_ternary_cached(t, pos)
                    .expect("cached logits"),
            );
        }
    });
    let mut greedy: Vec<u32> = Vec::new();
    in_session(&mut || {
        for (pos, &t) in tokens.iter().enumerate() {
            greedy.push(
                model
                    .forward_greedy_gpu_ternary_cached(t, pos)
                    .expect("cached greedy"),
            );
        }
    });
    for (pos, (a, b)) in uncached.iter().zip(&cached).enumerate() {
        assert_eq!(a.len(), FIXTURE_VOCAB);
        assert_eq!(
            bits(a),
            bits(b),
            "pos {pos}: cached logits diverged from uncached"
        );
        let mut best = 0usize;
        for (i, v) in a.iter().enumerate() {
            if *v > a[best] {
                best = i;
            }
        }
        assert_eq!(
            greedy[pos] as usize, best,
            "pos {pos}: cached greedy != argmax"
        );
    }

    // Batched prefill + verify: cached vs the strict uncached references, and
    // the model's (cached) routing entries vs both.
    let prompt: Vec<u32> = vec![7, 1, 30, 4, 12, 9];
    let mut uncached_prefill = Vec::new();
    in_session(&mut || {
        uncached_prefill = model
            .prefill_logits_gpu_ternary_uncached(&prompt, 0)
            .expect("uncached batched prefill");
    });
    let mut cached_prefill = Vec::new();
    in_session(&mut || {
        cached_prefill = model
            .prefill_logits_gpu_ternary_cached(&prompt, 0)
            .expect("cached batched prefill");
    });
    let mut routed_prefill = Vec::new();
    in_session(&mut || {
        routed_prefill = model
            .try_metal_prefill_with_lm_head_ternary(&prompt, 0)
            .expect("the model's batched prefill route");
    });
    assert_eq!(cached_prefill.len(), FIXTURE_VOCAB);
    assert_eq!(bits(&uncached_prefill), bits(&cached_prefill));
    assert_eq!(bits(&routed_prefill), bits(&cached_prefill));
    let mut uncached_verify = Vec::new();
    in_session(&mut || {
        uncached_verify = model
            .prefill_verify_gpu_ternary_uncached(&prompt, 0)
            .expect("uncached verify");
    });
    let mut cached_verify = Vec::new();
    in_session(&mut || {
        cached_verify = model
            .prefill_verify_gpu_ternary_cached(&prompt, 0)
            .expect("cached verify");
    });
    let mut routed_verify = Vec::new();
    in_session(&mut || {
        routed_verify = model
            .try_metal_prefill_verify_ternary_path(&prompt, 0)
            .expect("the model's verify route");
    });
    assert_eq!(cached_verify.len(), prompt.len());
    assert_eq!(uncached_verify, cached_verify);
    assert_eq!(routed_verify, cached_verify);

    assert_eq!(
        graph.cached_weight_count().expect("count"),
        resident_before,
        "no path may upload a buffer beyond the model's {owned} slots"
    );
    in_session(&mut || {
        let _ = model.release_metal_weights();
    });
}

// ── C1: one epoch per GGUF mapping ───────────────────────────────────

/// Every replica of one GGUF mapping joins one namespace — one epoch, handed
/// to every block too — while another mapping gets its own, even over
/// byte-identical content (a namespace is about the mapping, not the bytes).
/// Pure construction: nothing here touches the GPU.
#[test]
fn replicas_of_one_mapping_share_one_epoch_and_other_mappings_do_not() {
    let bytes_a = synthetic_ternary_gguf(0x1A);
    let bytes_b = synthetic_ternary_gguf(0x1A); // same content, another allocation
    let gguf_a = GgufFile::parse(&bytes_a).expect("parse A");
    let gguf_b = GgufFile::parse(&bytes_b).expect("parse B");
    let a1 = BonsaiModel::from_gguf(&gguf_a, FIXTURE_MAX_SEQ).expect("load A1");
    let a2 = BonsaiModel::from_gguf(&gguf_a, FIXTURE_MAX_SEQ).expect("load A2");
    let b = BonsaiModel::from_gguf(&gguf_b, FIXTURE_MAX_SEQ).expect("load B");

    let epoch_a = a1.gpu_mapping_epoch();
    assert_eq!(a2.gpu_mapping_epoch(), epoch_a, "replicas share the epoch");
    assert_ne!(
        b.gpu_mapping_epoch(),
        epoch_a,
        "another mapping — even of identical bytes — is another namespace"
    );
    assert!(a1.q1_metal_slots().is_shared_mapping());
    assert_eq!(a1.q1_metal_slots().replicas(), 2);
    assert_eq!(b.q1_metal_slots().replicas(), 1);
    assert_eq!(a1.q1_metal_slots().lm_head(), a2.q1_metal_slots().lm_head());
    for block in a1.blocks.iter().chain(&a2.blocks) {
        assert_eq!(
            block.gpu_slot_epoch(),
            epoch_a,
            "every block keys on its model's mapping namespace"
        );
    }
    drop(a2);
    assert_eq!(
        a1.q1_metal_slots().replicas(),
        1,
        "a dropped replica leaves"
    );
    assert_eq!(
        a1.gpu_mapping_epoch(),
        epoch_a,
        "the survivor keeps the epoch"
    );

    // A weight-less model has nothing mapped: a private namespace.
    let weightless = BonsaiModel::new(oxibonsai_core::config::Qwen3Config::tiny_test());
    assert!(!weightless.q1_metal_slots().is_shared_mapping());
    assert_ne!(weightless.gpu_mapping_epoch(), epoch_a);
}

/// C1 acceptance: a model dropped and re-loaded at a **reused address**
/// never binds a stale buffer.
///
/// One allocation is loaded as model A, warmed and dropped; then A's old
/// weights are planted back at A's slot addresses — under A's old epoch and
/// under the legacy epoch, i.e. exactly what a missed release (or the pre-C1
/// address-only keying) leaves resident — and the same allocation is
/// overwritten with **different** weights of identical layout and re-loaded
/// as model B. Same allocation and layout mean the same tensor addresses, so
/// B's slots coincide with A's; B must still mint a new epoch, bind only
/// buffers of its own, and compute its own CPU forward's logits. The control
/// forces a replica of B onto A's epoch and must see the stale weights —
/// proving the check can fail.
#[test]
fn a_model_reloaded_at_a_reused_address_never_binds_a_stale_buffer() {
    use super::super::forward_metal::Q1MetalSlots;
    use oxibonsai_kernels::{KernelDispatcher, KernelTier};

    let _gpu = gpu_serial();
    let Ok(device) = oxibonsai_kernels::MetalDevice::isolated() else {
        return;
    };
    let session = MetalGraph::new_session_on(&device);
    MetalGraph::with_session(&session, || {
        let token = 3u32;
        let mut buffer = synthetic_ternary_gguf(0xA0A0);

        // ── Model A: load, warm, remember its epoch and anchor slots ──
        let (epoch_a, slots_a, stale_attn_proj) = {
            let gguf = GgufFile::parse(&buffer).expect("parse A");
            let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load A");
            model.get_or_create_gpu_cache().expect("warm A");
            let (layers, _tail) = model.ternary_gpu_slots().expect("slots A");
            let attn_proj = ternary_bytes(model.blocks[0].attn_output_blocks_ternary(), "o")
                .expect("A's attention output")
                .to_vec();
            (model.gpu_mapping_epoch(), layers, attn_proj)
            // A and its parse drop here: its only replica is gone.
        };
        let slot = slots_a[0].attn_proj;
        assert!(
            probe(&session, epoch_a, slot, WeightKind::Tq2Soa).is_none(),
            "A's last replica released its buffers"
        );
        assert_eq!(
            super::super::q1_slots::global_epoch_of(slots_a[0].fused_qkv),
            None,
            "the registry forgot A's mapping"
        );

        // Plant the stale state a missed release would leave behind.
        let stale = session
            .get_or_upload_tq2_weight_soa_for_epoch(epoch_a, slot, &stale_attn_proj)
            .expect("plant A's weights under A's epoch");
        session
            .get_or_upload_tq2_weight_soa_for_epoch(
                oxibonsai_kernels::LEGACY_MODEL_EPOCH,
                slot,
                &stale_attn_proj,
            )
            .expect("plant A's weights under the legacy epoch");

        // ── Model B: different weights, identical layout, same allocation ──
        let other = synthetic_ternary_gguf(0xB0B0);
        assert_eq!(other.len(), buffer.len(), "identical layout");
        buffer.copy_from_slice(&other);
        let gguf = GgufFile::parse(&buffer).expect("parse B");
        let model_b = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load B");
        let (layers_b, _) = model_b.ternary_gpu_slots().expect("slots B");
        assert_eq!(
            layers_b, slots_a,
            "same allocation and layout: B's tensors sit at A's addresses"
        );
        let epoch_b = model_b.gpu_mapping_epoch();
        assert_ne!(
            epoch_b, epoch_a,
            "a new mapping at a reused address must mint a new epoch"
        );

        let gpu_b = model_b
            .forward_logits_gpu_ternary_cached(token, 0)
            .expect("B's fused logits");
        let cpu_b = {
            let cpu = KernelDispatcher::with_tier(KernelTier::Reference);
            let mut reference = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("CPU B");
            reference.forward(token, 0, &cpu).expect("B's CPU forward")
        };
        let close = |a: &[f32], b: &[f32]| {
            a.len() == b.len()
                && a.iter().zip(b).all(|(x, y)| {
                    let abs = (x - y).abs();
                    abs < 1e-3 || abs / x.abs().max(y.abs()).max(1e-3) < 1e-3
                })
        };
        assert!(
            close(&gpu_b, &cpu_b),
            "B's GPU logits must be B's own, not a stale mix with A's"
        );
        let own = probe(&session, epoch_b, slot, WeightKind::Tq2Soa).expect("B's own buffer");
        assert!(
            !Arc::ptr_eq(&own, &stale),
            "B bound the stale buffer A left at the reused address"
        );

        // ── Control: the same weights forced onto A's epoch go stale ──
        let mut forced = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load C");
        forced.metal_q1_slots = Q1MetalSlots::with_epoch(epoch_a);
        let gpu_forced = forced
            .forward_logits_gpu_ternary_cached(token, 0)
            .expect("forced logits");
        assert!(
            !close(&gpu_forced, &cpu_b),
            "control: a model keyed under A's epoch must be served A's stale attention output"
        );
        drop(forced);
        drop(model_b);
        let _ = session.release_model(oxibonsai_kernels::LEGACY_MODEL_EPOCH);
    });
}

// ── C2: the block's fused arms and the model cache share one buffer ──

/// C2 (the `M-21` residue): on Metal a GPU-uploaded ternary block keeps no
/// Q‖K‖V / gate‖up concatenation in the kernel's own weight cache, and its
/// fused arms key the concatenations on the model's mapping namespace — the
/// very `(epoch, slot)` keys the model's full-forward ternary cache uses — so
/// running the block path and then building the model cache holds **one**
/// copy of each: the concatenations are byte-identical, the cache binds the
/// buffer the block uploaded (`Arc::ptr_eq`), and `bytes_uploaded` grows by
/// exactly the model's own weight set, not a byte more. All three block
/// forward kinds then run without growing it.
#[test]
fn block_fused_arms_and_the_model_cache_bind_one_buffer() {
    use crate::kv_cache::KvCache;
    use crate::layers::rope::RopeTable;
    use oxibonsai_kernels::{KernelDispatcher, KernelTier};

    let _gpu = gpu_serial();
    let Ok(device) = oxibonsai_kernels::MetalDevice::isolated() else {
        return;
    };
    let Ok(gpu_kernel) = KernelDispatcher::try_with_tier(KernelTier::Gpu) else {
        return;
    };
    let session = MetalGraph::new_session_on(&device);
    MetalGraph::with_session(&session, || {
        let bytes = synthetic_ternary_gguf(0xC2C2);
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let mut model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load");
        model.upload_weights_to_gpu(&gpu_kernel);
        let epoch = model.gpu_mapping_epoch();
        let (layers, tail) = model.ternary_gpu_slots().expect("slots");
        for block in &model.blocks {
            assert!(
                block.fused_qkv_gpu_handle_ternary().is_none()
                    && block.fused_gate_up_gpu_handle_ternary().is_none(),
                "Metal uploads no ternary concatenation to the kernel's cache"
            );
            assert_eq!(block.gpu_slot_epoch(), epoch);
        }
        let block = &model.blocks[0];
        assert_eq!(block.ternary_fused_qkv_slot(), Some(layers[0].fused_qkv));
        assert_eq!(block.ternary_fused_gate_up_slot(), Some(layers[0].gate_up));

        // Byte identity: the parts the block arms concatenate are, in order,
        // the parts the model cache concatenates for this layer.
        let part = |b: Option<&[oxibonsai_core::BlockTQ2_0_g128]>| {
            ternary_bytes(b, "part").expect("ternary part").to_vec()
        };
        let block_qkv = [
            part(block.attn_q_blocks_ternary()),
            part(block.attn_k_blocks_ternary()),
            part(block.attn_v_blocks_ternary()),
        ]
        .concat();
        let block_gate_up = [
            part(block.ffn_gate_blocks_ternary()),
            part(block.ffn_up_blocks_ternary()),
        ]
        .concat();
        let mut scratch = Vec::new();
        fill_scratch(
            &mut scratch,
            &[
                ternary_bytes(block.attn_q_blocks_ternary(), "q").expect("q"),
                ternary_bytes(block.attn_k_blocks_ternary(), "k").expect("k"),
                ternary_bytes(block.attn_v_blocks_ternary(), "v").expect("v"),
            ],
        );
        assert_eq!(scratch, block_qkv, "the Q‖K‖V concatenations are identical");
        fill_scratch(
            &mut scratch,
            &[
                ternary_bytes(block.ffn_gate_blocks_ternary(), "gate").expect("gate"),
                ternary_bytes(block.ffn_up_blocks_ternary(), "up").expect("up"),
            ],
        );
        assert_eq!(
            scratch, block_gate_up,
            "the gate‖up concatenations are identical"
        );

        // The block path first: exactly its two concatenations go up.
        let (hd, nkv, h) = (FIXTURE_HD, FIXTURE_NKV, FIXTURE_HIDDEN);
        let rope = RopeTable::new(hd, 16, 10_000.0);
        let mut kv = KvCache::new(FIXTURE_LAYERS, nkv, hd, 16);
        let input: Vec<f32> = (0..h).map(|i| ((i * 7) % 23) as f32 * 0.01 - 0.1).collect();
        let before = session.bytes_uploaded();
        let mut hidden = input.clone();
        block
            .forward(&mut hidden, 0, &mut kv, &rope, &gpu_kernel)
            .expect("block forward");
        let after_block = session.bytes_uploaded();
        assert_eq!(
            after_block - before,
            (block_qkv.len() + block_gate_up.len()) as u64,
            "the block path must upload exactly its Q‖K‖V and gate‖up concatenations"
        );
        let qkv_buf = probe(&session, epoch, layers[0].fused_qkv, WeightKind::Tq2Soa)
            .expect("the block's Q‖K‖V is resident under the mapping key");
        let gate_up_buf = probe(&session, epoch, layers[0].gate_up, WeightKind::Tq2Soa)
            .expect("the block's gate‖up is resident under the mapping key");

        // Then the model cache: it binds those very buffers, and the device
        // holds exactly the model's own weight set.
        model
            .get_or_create_gpu_cache()
            .expect("build the model cache");
        let keys = all_cache_keys(&layers, tail);
        let expected: u64 = keys
            .iter()
            .map(|(slot, kind)| {
                probe(&session, epoch, *slot, *kind)
                    .map(|h| h.byte_len() as u64)
                    .unwrap_or_else(|| panic!("slot {slot:#x} missing"))
            })
            .sum();
        assert_eq!(session.cached_weight_count().expect("count"), keys.len());
        assert_eq!(
            session.bytes_uploaded() - before,
            expected,
            "running both paths must hold one copy of the model"
        );
        {
            let guard = model.gpu_weight_cache.lock().expect("cache lock");
            match guard.as_ref().expect("populated") {
                CachedModelWeights::Ternary(tern) => {
                    assert!(Arc::ptr_eq(&tern.layers[0].fused_qkv, &qkv_buf));
                    assert!(Arc::ptr_eq(&tern.layers[0].gate_up, &gate_up_buf));
                }
                CachedModelWeights::Q1(_) => panic!("ternary model produced a Q1 cache"),
            }
        }

        // Every block forward kind again: nothing new goes up.
        let resident = session.bytes_uploaded();
        let mut kv_sw = KvCache::new(FIXTURE_LAYERS, nkv, hd, 16);
        let mut hidden_sw = input.clone();
        block
            .forward_with_sliding_window(&mut hidden_sw, 0, &mut kv_sw, &rope, &gpu_kernel, None)
            .expect("sliding-window forward");
        let mut kv_stats = KvCache::new(FIXTURE_LAYERS, nkv, hd, 16);
        let mut hidden_stats = input;
        block
            .forward_with_stats(&mut hidden_stats, 0, &mut kv_stats, &rope, &gpu_kernel)
            .expect("stats forward");
        assert_eq!(session.bytes_uploaded(), resident);
        for (name, other) in [("sliding-window", &hidden_sw), ("stats", &hidden_stats)] {
            let worst = hidden
                .iter()
                .zip(other.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 1e-4,
                "{name} forward diverged from forward over one shared buffer: {worst}"
            );
        }
        let _ = model.release_metal_weights();
    });
}
