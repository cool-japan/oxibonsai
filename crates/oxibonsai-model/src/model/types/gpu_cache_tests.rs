//! Tests for `gpu_cache.rs` — slot identity (MET-02), host-copy freedom and
//! the cached shape (MET-03), no per-call model copy (perf-03), release on
//! unload and the refcounted `Drop`.
//!
//! Split out of `gpu_cache.rs` (compiled as its `#[cfg(test)] #[path]`
//! child module, so `super::*` is unchanged) to keep that file under the
//! 2000-line policy ceiling.

use super::*;
use crate::test_alloc::count_allocations;
use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
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
/// buffer is cached under exactly this `(kind, slot)`, `None` means it is
/// not (or is cached under a different kind).
fn probe(graph: &MetalGraph, slot: u64, kind: WeightKind) -> Option<Arc<MetalWeightHandle>> {
    graph
        .get_or_upload_keyed(WeightKey::new(TERNARY_GPU_EPOCH, kind, slot), || {
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

/// The mapped-tensor namespace cannot reach the Q1 path's literal handles.
///
/// The Q1 decode path keys its norms on `1_000_000 + layer * 10 + off`, its
/// final norm on `2_000_000` and its LM head on `3_000_000`. Before MET-02
/// the ternary fallback path used `2_000_000 + layer * 10` for **its**
/// norms, so a mixed process handed the ternary layer-0 attention norm the
/// Q1 final-norm buffer — same `WeightKind::RawF32`, so a silent stale hit
/// rather than an error.
#[test]
fn mapped_tensor_slots_cannot_reach_the_q1_literal_handles() {
    let highest_q1_literal = 3_000_000u64 + 100_000 * 10 + 3;
    assert!(
        MIN_TENSOR_SLOT > highest_q1_literal,
        "the mapped-tensor floor {MIN_TENSOR_SLOT} must sit above every Q1 literal handle \
         ({highest_q1_literal})"
    );
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
    {
        let guard = model
            .gpu_weight_cache
            .lock()
            .expect("gpu_weight_cache lock");
        match guard.as_ref().expect("cache populated") {
            CachedModelWeights::Ternary(tern) => {
                assert_eq!(tern.model_epoch, TERNARY_GPU_EPOCH);
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
                        let resident = probe(&graph, slot, kind)
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
                    &probe(&graph, tail.final_norm, WeightKind::RawF32).expect("final norm")
                ));
                assert!(Arc::ptr_eq(
                    lm_head,
                    &probe(&graph, tail.lm_head, WeightKind::Tq2Soa).expect("lm head")
                ));
            }
            CachedModelWeights::Q1(_) => panic!("ternary model produced a Q1 cache"),
        }
    }

    // The bytes really did go to the GPU: every slot is resident and the
    // fused QKV buffer is exactly Q‖K‖V long.
    let tail = Some(tail);
    for (slot, kind) in all_cache_keys(&layers, tail) {
        let handle = probe(&graph, slot, kind)
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
    let qkv = probe(&graph, layers[0].fused_qkv, WeightKind::Tq2Soa).expect("fused qkv");
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
    // Start from a genuinely free slot, so the upload cannot be short-cut
    // by a cache hit.
    let _ = model.release_metal_weights();
    assert!(
        probe(&graph, slot, WeightKind::Tq2Soa).is_none(),
        "slot {slot:#x} must be free for this test to mean anything"
    );

    graph
        .get_or_upload_tq2_weight_soa(slot, FUSED_QKV_ALREADY_RESIDENT)
        .expect_err("an empty TQ2 upload must fail, not allocate a zero-length buffer");
    assert!(
        probe(&graph, slot, WeightKind::Tq2Soa).is_none(),
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
            probe(&graph, *slot, *kind)
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
        let after = probe(&graph, *slot, *kind)
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
    for (slot, kind) in &keys {
        assert!(
            probe(&graph, *slot, *kind).is_some(),
            "slot {slot:#x} should be resident before release"
        );
    }

    let resident_before = graph.resident_weight_bytes().expect("resident bytes");
    let released = model.release_metal_weights().expect("release this model");
    assert_eq!(released, keys.len(), "every owned slot must be released");

    for (slot, kind) in &keys {
        assert!(
            probe(&graph, *slot, *kind).is_none(),
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
            probe(&graph, *slot, *kind).is_some(),
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
    let keys = {
        let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load model");
        model.get_or_create_gpu_cache().expect("warm the GPU cache");
        let (layers, tail) = model.ternary_gpu_slots().expect("derive slots");
        let keys = all_cache_keys(&layers, tail);
        for (slot, kind) in &keys {
            assert!(
                probe(&graph, *slot, *kind).is_some(),
                "slot {slot:#x} should be resident before drop"
            );
        }
        keys
        // `model` is dropped here, at the end of this block — nothing
        // else calls `release_metal_weights()` in this test.
    };
    for (slot, kind) in &keys {
        assert!(
            probe(&graph, *slot, *kind).is_none(),
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
    let keys = all_cache_keys(&layers_a, tail_a);

    for (slot, kind) in &keys {
        assert!(
            probe(&graph, *slot, *kind).is_some(),
            "slot {slot:#x} should be resident once both replicas are warm"
        );
    }

    drop(replica_a);

    // Replica B is still alive and shares these exact slots: none of them
    // may have been evicted by A's drop.
    for (slot, kind) in &keys {
        assert!(
            probe(&graph, *slot, *kind).is_some(),
            "slot {slot:#x} was evicted while a sibling replica was still alive"
        );
    }

    drop(replica_b);

    // Now the last replica is gone: the shared buffers must be released.
    for (slot, kind) in &keys {
        assert!(
            probe(&graph, *slot, *kind).is_none(),
            "slot {slot:#x} is still cached after the last replica dropped"
        );
    }
}

/// MET-03: the cached decode entry points are bit-identical to the uncached
/// ones — single-token logits, greedy tokens, batched prefill and verify — and
/// no path uploads a buffer beyond the model's own slot table.
///
/// Each run gets its **own** Metal session (`MET-08`), so two device KV caches
/// never see each other's writes: position `p` of the cached run attends over
/// what the cached run wrote, exactly as the uncached run did. All sessions sit
/// on one **isolated** device, so the residency count is this test's alone.
#[test]
fn cached_ternary_decode_is_bit_identical_to_the_uncached_path() {
    // Still serialised: the isolated device only privatises the weight cache.
    // The forced-tail-failure seam `forward_logits_gpu_ternary_uncached` goes
    // through, and the allocation counter a sibling test reads, are
    // process-global.
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

    // Batched prefill + verify: cached vs the strict uncached entry points.
    let prompt: Vec<u32> = vec![7, 1, 30, 4, 12, 9];
    let mut uncached_prefill = Vec::new();
    in_session(&mut || {
        uncached_prefill = model
            .try_metal_prefill_with_lm_head_ternary(&prompt, 0)
            .expect("uncached batched prefill");
    });
    let mut cached_prefill = Vec::new();
    in_session(&mut || {
        cached_prefill = model
            .prefill_logits_gpu_ternary_cached(&prompt, 0)
            .expect("cached batched prefill");
    });
    assert_eq!(cached_prefill.len(), FIXTURE_VOCAB);
    assert_eq!(bits(&uncached_prefill), bits(&cached_prefill));
    let mut uncached_verify = Vec::new();
    in_session(&mut || {
        uncached_verify = model
            .try_metal_prefill_verify_ternary_path(&prompt, 0)
            .expect("uncached verify");
    });
    let mut cached_verify = Vec::new();
    in_session(&mut || {
        cached_verify = model
            .prefill_verify_gpu_ternary_cached(&prompt, 0)
            .expect("cached verify");
    });
    assert_eq!(cached_verify.len(), prompt.len());
    assert_eq!(uncached_verify, cached_verify);

    assert_eq!(
        graph.cached_weight_count().expect("count"),
        resident_before,
        "no path may upload a buffer beyond the model's {owned} slots"
    );
    in_session(&mut || {
        let _ = model.release_metal_weights();
    });
}
