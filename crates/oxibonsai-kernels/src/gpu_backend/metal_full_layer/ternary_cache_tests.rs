//! Tests for the ternary epoch keying (`MET-02`), the cached ternary shape
//! (`MET-03`) and the N-part fused TQ2 GEMV (`M-21`) in `functions_3.rs`.
//!
//! Every GPU test runs inside its own **isolated** device (`MetalGraph::new()`
//! bound as the thread's session), so its weight cache, counters and KV cache
//! are private: the residency assertions below cannot be perturbed by another
//! test running in parallel in this binary. Each GPU test no-ops on a host
//! without a Metal device.

use super::*;
use crate::gpu_backend::metal_full_layer::types::{WeightKey, WeightKind};

/// Hidden size (one `TQ2_0_g128` block per row of the square projections).
const H: usize = 128;
/// FFN intermediate size.
const INTER: usize = 256;
/// Query heads.
const NQ: usize = 4;
/// KV heads.
const NKV: usize = 2;
/// Head dimension (`H / NQ`).
const HD: usize = 32;
/// LM-head rows.
const VOCAB: usize = 32;
/// Transformer layers.
const LAYERS: usize = 2;
/// KV-cache capacity.
const MAX_SEQ: usize = 64;
/// RMSNorm epsilon.
const EPS: f32 = 1e-6;
/// Slot of the final RMSNorm weight (layers use `layer * 16 + 0..8`).
const FINAL_NORM_SLOT: u64 = 0x7000;
/// Slot of the LM head.
const LM_HEAD_SLOT: u64 = 0x7001;

/// One layer's weights, owned so the parameter structs can borrow them.
struct LayerData {
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    qkv: Vec<u8>,
    attn_proj: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

/// A whole synthetic fully-ternary model.
struct Fixture {
    layers: Vec<LayerData>,
    final_norm: Vec<f32>,
    lm_head: Vec<u8>,
}

/// Deterministic `TQ2_0_g128` blocks for a `rows x cols` matrix, emitting only
/// the three ternary codes (the upload path rejects `0b11`).
fn tq2(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n_blocks = rows * cols / 128;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    let mut out = Vec::with_capacity(n_blocks * 34);
    for _ in 0..n_blocks {
        for _ in 0..32 {
            let mut byte = 0u8;
            for lane in 0..4 {
                byte |= ((next() % 3) as u8) << (2 * lane);
            }
            out.push(byte);
        }
        let scale = 0.02 + (next() % 1000) as f32 * 0.00004;
        out.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
    }
    out
}

/// A positive, index-varying RMSNorm weight.
fn norm(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| 0.8 + 0.4 * (((i as u64 * 7 + seed * 13) % 17) as f32 / 17.0))
        .collect()
}

impl Fixture {
    fn new(seed: u64) -> Self {
        let layers = (0..LAYERS as u64)
            .map(|l| {
                let s = seed * 100 + l * 10;
                let mut qkv = tq2(NQ * HD, H, s + 1);
                qkv.extend_from_slice(&tq2(NKV * HD, H, s + 2));
                qkv.extend_from_slice(&tq2(NKV * HD, H, s + 3));
                LayerData {
                    attn_norm: norm(H, s + 4),
                    q_norm: norm(HD, s + 5),
                    k_norm: norm(HD, s + 6),
                    ffn_norm: norm(H, s + 7),
                    qkv,
                    attn_proj: tq2(H, NQ * HD, s + 8),
                    gate: tq2(INTER, H, s + 9),
                    up: tq2(INTER, H, s + 10),
                    down: tq2(H, INTER, s + 11),
                }
            })
            .collect();
        Self {
            layers,
            final_norm: norm(H, seed + 99),
            lm_head: tq2(VOCAB, H, seed + 98),
        }
    }

    /// Per-layer parameters keyed under `epoch`, slots `layer * 16 + 0..8`.
    fn params(&self, epoch: u64) -> Vec<FullForwardLayerParamsTernary<'_>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(l, d)| {
                let base = (l as u64) * 16;
                FullForwardLayerParamsTernary {
                    model_epoch: epoch,
                    attn_norm_handle: base,
                    attn_norm_bytes: &d.attn_norm,
                    fused_qkv_handle: base + 1,
                    fused_qkv_bytes: &d.qkv,
                    q_norm_handle: base + 2,
                    q_norm_bytes: &d.q_norm,
                    k_norm_handle: base + 3,
                    k_norm_bytes: &d.k_norm,
                    attn_proj_handle: base + 4,
                    attn_proj_bytes: &d.attn_proj,
                    ffn_norm_handle: base + 5,
                    ffn_norm_bytes: &d.ffn_norm,
                    gate_up_handle: base + 6,
                    gate_bytes: &d.gate,
                    up_bytes: &d.up,
                    down_handle: base + 7,
                    down_bytes: &d.down,
                }
            })
            .collect()
    }

    /// Every `(slot, kind)` the model owns: 8 per layer, then the tail.
    fn keys(&self) -> Vec<(u64, WeightKind)> {
        let mut keys = Vec::new();
        for l in 0..self.layers.len() as u64 {
            let base = l * 16;
            for (off, kind) in [
                (0, WeightKind::RawF32),
                (1, WeightKind::Tq2Soa),
                (2, WeightKind::RawF32),
                (3, WeightKind::RawF32),
                (4, WeightKind::Tq2Soa),
                (5, WeightKind::RawF32),
                (6, WeightKind::Tq2Soa),
                (7, WeightKind::Tq2Soa),
            ] {
                keys.push((base + off, kind));
            }
        }
        keys.push((FINAL_NORM_SLOT, WeightKind::RawF32));
        keys.push((LM_HEAD_SLOT, WeightKind::Tq2Soa));
        keys
    }
}

/// RoPE cos/sin rows for `pos` (`theta = 10000`, split-half pairing).
fn rope(pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = HD / 2;
    let angles: Vec<f32> = (0..half)
        .map(|i| pos as f32 * 10_000f32.powf(-2.0 * i as f32 / HD as f32))
        .collect();
    (
        angles.iter().map(|a| a.cos()).collect(),
        angles.iter().map(|a| a.sin()).collect(),
    )
}

/// A deterministic input hidden state for `(pos, salt)`.
fn hidden_for(pos: usize, salt: usize) -> Vec<f32> {
    (0..H)
        .map(|i| ((i * 31 + pos * 7 + salt * 3) % 23) as f32 * 0.05 - 0.55)
        .collect()
}

/// Run `f` inside a fresh isolated device + session, or `None` without Metal.
fn isolated<R>(f: impl FnOnce(&Arc<MetalGraph>) -> R) -> Option<R> {
    let graph = Arc::new(MetalGraph::new().ok()?);
    Some(MetalGraph::with_session(&graph, || f(&graph)))
}

/// Residency probe: `Some` iff `key` is cached with exactly that kind.
fn probe(graph: &MetalGraph, key: WeightKey) -> Option<Arc<MetalWeightHandle>> {
    graph
        .get_or_upload_keyed(key, || {
            Err(MetalGraphError::ExecutionFailed(
                "probe: not resident".into(),
            ))
        })
        .ok()
}

/// One uncached single-token forward returning the logits.
fn uncached_logits(fx: &Fixture, epoch: u64, pos: usize, salt: usize) -> Vec<f32> {
    let params = fx.params(epoch);
    let (cos, sin) = rope(pos);
    let mut hidden = hidden_for(pos, salt);
    let mut logits = Vec::new();
    try_metal_prefill_ternary(
        &mut hidden,
        pos,
        LAYERS,
        &params,
        &cos,
        &sin,
        H,
        INTER,
        NQ,
        NKV,
        HD,
        EPS,
        MAX_SEQ,
        Some(FINAL_NORM_SLOT),
        Some(&fx.final_norm),
        EPS,
        Some(LM_HEAD_SLOT),
        Some(&fx.lm_head),
        VOCAB,
        &mut logits,
    )
    .expect("uncached ternary forward");
    logits
}

/// One cached single-token forward returning the logits.
fn cached_logits(cache: &CachedModelWeights, pos: usize, salt: usize) -> Vec<f32> {
    let (cos, sin) = rope(pos);
    let mut hidden = hidden_for(pos, salt);
    let mut logits = Vec::new();
    try_metal_prefill_ternary_cached(
        &mut hidden,
        pos,
        cache,
        &cos,
        &sin,
        H,
        INTER,
        NQ,
        NKV,
        HD,
        EPS,
        MAX_SEQ,
        EPS,
        &mut logits,
    )
    .expect("cached ternary forward");
    logits
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// MET-02: the epoch is threaded through every lookup
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn a_forward_mixing_two_model_epochs_is_rejected_before_any_gpu_work() {
    let fx = Fixture::new(1);
    let mut params = fx.params(7);
    params[1].model_epoch = 8;
    let err = shared_model_epoch(&params).expect_err("mixed epochs must be rejected");
    assert!(err.to_string().contains("epoch"), "{err}");
    // The builder and the uncached entry reject it too, before touching the
    // device (so this holds on a host with no GPU as well).
    assert!(build_cached_weights_ternary_only(&params, None, None, 0).is_err());
    let (cos, sin) = rope(0);
    let mut hidden = hidden_for(0, 0);
    let res = try_metal_full_forward_ternary(
        &mut hidden,
        0,
        LAYERS,
        &params,
        &cos,
        &sin,
        H,
        INTER,
        NQ,
        NKV,
        HD,
        EPS,
        MAX_SEQ,
        None,
        None,
        EPS,
        None,
        None,
        0,
        None,
        None,
    );
    assert!(res.is_err(), "a mixed-epoch forward must not run");
}

#[test]
fn every_ternary_lookup_is_keyed_on_the_layer_epoch_and_released_with_it() {
    let Some(()) = isolated(|graph| {
        let fx = Fixture::new(2);
        let epoch = MetalGraph::next_model_epoch();
        let logits = uncached_logits(&fx, epoch, 0, 0);
        assert_eq!(logits.len(), VOCAB);
        assert!(logits.iter().all(|x| x.is_finite()));

        let keys = fx.keys();
        for (slot, kind) in &keys {
            assert!(
                probe(graph, WeightKey::new(epoch, *kind, *slot)).is_some(),
                "slot {slot:#x} ({kind}) is not resident under the layer epoch {epoch}"
            );
            assert!(
                probe(graph, WeightKey::legacy(*kind, *slot)).is_none(),
                "slot {slot:#x} ({kind}) leaked into the legacy epoch — the kernel ignored \
                 `model_epoch`"
            );
        }
        assert_eq!(
            graph.cached_weight_count().expect("count"),
            keys.len(),
            "exactly the model's buffers, once each"
        );

        // `release_model` frees exactly this model's buffers — the property the
        // epoch exists for.
        let released = graph.release_model(epoch).expect("release");
        assert_eq!(released, keys.len());
        for (slot, kind) in &keys {
            assert!(probe(graph, WeightKey::new(epoch, *kind, *slot)).is_none());
        }
        assert_eq!(graph.resident_weight_bytes().expect("bytes"), 0);
    }) else {
        return;
    };
}

// ─────────────────────────────────────────────────────────────────────────
// MET-03: the cached shape binds the same buffers and is bit-identical
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn the_cached_ternary_path_is_bit_identical_and_adds_no_buffers() {
    let Some(()) = isolated(|graph| {
        let fx = Fixture::new(3);
        let epoch = MetalGraph::next_model_epoch();

        // Uncached decode of a 6-token sequence (the KV cache carries state).
        let steps = 6usize;
        let uncached: Vec<Vec<f32>> = (0..steps)
            .map(|p| uncached_logits(&fx, epoch, p, 1))
            .collect();
        let resident = graph.cached_weight_count().expect("count");

        // The cache is built over the SAME epoch and slots: pure cache hits.
        let params = fx.params(epoch);
        let cache = build_cached_weights_ternary_only(
            &params,
            Some((FINAL_NORM_SLOT, &fx.final_norm)),
            Some((LM_HEAD_SLOT, &fx.lm_head)),
            VOCAB,
        )
        .expect("build the ternary cache");
        assert_eq!(
            graph.cached_weight_count().expect("count"),
            resident,
            "building the cache over resident slots must not upload a second copy"
        );
        let CachedModelWeights::Ternary(tern) = &cache else {
            panic!("a ternary build must produce the ternary variant");
        };
        assert_eq!(tern.model_epoch, epoch);
        assert_eq!(tern.layers.len(), LAYERS);
        assert_eq!(tern.lm_head_out_features, VOCAB);
        let first = &tern.layers[0];
        let resident_qkv =
            probe(graph, WeightKey::new(epoch, WeightKind::Tq2Soa, 1)).expect("fused qkv resident");
        assert!(
            Arc::ptr_eq(&first.fused_qkv, &resident_qkv),
            "the cache must hold the resident buffer"
        );
        assert_eq!(first.fused_qkv.kind(), WeightKind::Tq2Soa);
        assert_eq!(first.attn_norm.kind(), WeightKind::RawF32);

        // Re-run the same sequence through the cached entry: every position
        // rewrites the KV slots it then reads, so the inputs are identical.
        for (p, expected) in uncached.iter().enumerate() {
            let got = cached_logits(&cache, p, 1);
            assert_eq!(
                bits(&got),
                bits(expected),
                "step {p}: the cached ternary path diverged from the uncached one"
            );
        }

        // The greedy cached entry returns the argmax of those same logits
        // (first index on ties, like the GPU kernel).
        let last = steps - 1;
        let (cos, sin) = rope(last);
        let mut hidden = hidden_for(last, 1);
        let mut token = u32::MAX;
        try_metal_forward_greedy_ternary_cached(
            &mut hidden,
            last,
            &cache,
            &cos,
            &sin,
            H,
            INTER,
            NQ,
            NKV,
            HD,
            EPS,
            MAX_SEQ,
            EPS,
            &mut token,
        )
        .expect("cached greedy");
        let row = &uncached[last];
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        assert_eq!(
            token as usize, best,
            "cached greedy token != argmax of the logits"
        );
    }) else {
        return;
    };
}

#[test]
fn a_cache_without_a_tail_serves_hidden_state_forwards_only() {
    let Some(()) = isolated(|_graph| {
        let fx = Fixture::new(4);
        let epoch = MetalGraph::next_model_epoch();
        let params = fx.params(epoch);
        let cache = build_cached_weights_ternary_only(&params, None, None, 0)
            .expect("a tail-less cache is valid");

        // Uncached hidden-state-only forward …
        let (cos, sin) = rope(0);
        let mut uncached = hidden_for(0, 2);
        try_metal_full_forward_ternary(
            &mut uncached,
            0,
            LAYERS,
            &params,
            &cos,
            &sin,
            H,
            INTER,
            NQ,
            NKV,
            HD,
            EPS,
            MAX_SEQ,
            None,
            None,
            EPS,
            None,
            None,
            0,
            None,
            None,
        )
        .expect("uncached hidden-only forward");
        // … equals the cached one bit for bit.
        let mut cached = hidden_for(0, 2);
        try_metal_full_forward_ternary_cached(
            &mut cached,
            0,
            &cache,
            &cos,
            &sin,
            H,
            INTER,
            NQ,
            NKV,
            HD,
            EPS,
            MAX_SEQ,
            EPS,
            None,
            None,
        )
        .expect("cached hidden-only forward");
        assert_eq!(bits(&cached), bits(&uncached));
        assert_ne!(
            bits(&cached),
            bits(&hidden_for(0, 2)),
            "the forward must change hidden"
        );

        // Asking that cache for logits is an error, not a silent download.
        let mut logits = Vec::new();
        let err = try_metal_prefill_ternary_cached(
            &mut cached,
            1,
            &cache,
            &cos,
            &sin,
            H,
            INTER,
            NQ,
            NKV,
            HD,
            EPS,
            MAX_SEQ,
            EPS,
            &mut logits,
        )
        .expect_err("no tail, no logits");
        assert!(err.to_string().contains("tail"), "{err}");
    }) else {
        return;
    };
}

#[test]
fn the_ternary_cache_builder_rejects_a_half_or_inconsistent_tail() {
    let fx = Fixture::new(5);
    let params = fx.params(LEGACY_MODEL_EPOCH);
    assert!(
        build_cached_weights_ternary_only(
            &params,
            Some((FINAL_NORM_SLOT, &fx.final_norm)),
            None,
            VOCAB
        )
        .is_err(),
        "a final norm without an LM head is half a tail"
    );
    assert!(
        build_cached_weights_ternary_only(
            &params,
            Some((FINAL_NORM_SLOT, &fx.final_norm)),
            Some((LM_HEAD_SLOT, &fx.lm_head)),
            0
        )
        .is_err(),
        "a tail with zero LM-head rows is inconsistent"
    );
    assert!(build_cached_weights_ternary_only(&params, None, None, VOCAB).is_err());
}

#[test]
fn the_cached_batched_prefill_matches_the_uncached_one_bit_for_bit() {
    let Some(()) = isolated(|_graph| {
        let fx = Fixture::new(6);
        // The uncached batched prefill (`metal_prefill`) still keys legacy, so
        // compare it against a cache built over legacy-epoch parameters.
        let params = fx.params(LEGACY_MODEL_EPOCH);
        let batch = 5usize;
        let hidden_batch: Vec<f32> = (0..batch).flat_map(|t| hidden_for(t, 3)).collect();
        let half = HD / 2;
        let mut cos_table = vec![0f32; batch * half];
        let mut sin_table = vec![0f32; batch * half];
        for t in 0..batch {
            let (c, s) = rope(t);
            cos_table[t * half..(t + 1) * half].copy_from_slice(&c);
            sin_table[t * half..(t + 1) * half].copy_from_slice(&s);
        }

        let mut uncached = Vec::new();
        crate::gpu_backend::try_metal_full_forward_prefill_ternary(
            &hidden_batch,
            batch,
            0,
            LAYERS,
            &params,
            &cos_table,
            &sin_table,
            H,
            INTER,
            NQ,
            NKV,
            HD,
            EPS,
            MAX_SEQ,
            Some(FINAL_NORM_SLOT),
            Some(&fx.final_norm),
            EPS,
            Some(LM_HEAD_SLOT),
            Some(&fx.lm_head),
            VOCAB,
            Some(&mut uncached),
            None,
        )
        .expect("uncached batched prefill");

        let cache = build_cached_weights_ternary_only(
            &params,
            Some((FINAL_NORM_SLOT, &fx.final_norm)),
            Some((LM_HEAD_SLOT, &fx.lm_head)),
            VOCAB,
        )
        .expect("build");
        let mut cached = Vec::new();
        try_metal_full_forward_prefill_ternary_cached(
            &hidden_batch,
            batch,
            0,
            &cache,
            &cos_table,
            &sin_table,
            H,
            INTER,
            NQ,
            NKV,
            HD,
            EPS,
            MAX_SEQ,
            EPS,
            Some(&mut cached),
            None,
        )
        .expect("cached batched prefill");
        assert_eq!(cached.len(), VOCAB);
        assert_eq!(bits(&cached), bits(&uncached));

        // Verify twin: per-position greedy ids agree too.
        let mut uncached_ids = Vec::new();
        crate::gpu_backend::try_metal_full_forward_prefill_verify_ternary(
            &hidden_batch,
            batch,
            0,
            LAYERS,
            &params,
            &cos_table,
            &sin_table,
            H,
            INTER,
            NQ,
            NKV,
            HD,
            EPS,
            MAX_SEQ,
            Some(FINAL_NORM_SLOT),
            Some(&fx.final_norm),
            EPS,
            Some(LM_HEAD_SLOT),
            Some(&fx.lm_head),
            VOCAB,
            &mut uncached_ids,
        )
        .expect("uncached verify");
        let mut cached_ids = Vec::new();
        try_metal_full_forward_prefill_verify_ternary_cached(
            &hidden_batch,
            batch,
            0,
            &cache,
            &cos_table,
            &sin_table,
            H,
            INTER,
            NQ,
            NKV,
            HD,
            EPS,
            MAX_SEQ,
            EPS,
            &mut cached_ids,
        )
        .expect("cached verify");
        assert_eq!(cached_ids.len(), batch);
        assert_eq!(cached_ids, uncached_ids);
    }) else {
        return;
    };
}

// ─────────────────────────────────────────────────────────────────────────
// M-21: the N-part fused TQ2 GEMV
// ─────────────────────────────────────────────────────────────────────────

/// Scalar reference: `out[r] = Σ_b d_b · Σ_j code(b, j) · x[b·128 + j]`.
fn tq2_gemv_reference(aos: &[u8], x: &[f32], n_rows: usize, k: usize) -> Vec<f32> {
    let bpr = k / 128;
    (0..n_rows)
        .map(|r| {
            let mut acc = 0f32;
            for b in 0..bpr {
                let blk = &aos[(r * bpr + b) * 34..(r * bpr + b + 1) * 34];
                let d = half::f16::from_le_bytes([blk[32], blk[33]]).to_f32();
                let mut s = 0f32;
                for (byte_idx, byte) in blk[..32].iter().enumerate() {
                    for lane in 0..4 {
                        let code = (byte >> (2 * lane)) & 0b11;
                        let w = f32::from(code) - 1.0;
                        s += w * x[b * 128 + byte_idx * 4 + lane];
                    }
                }
                acc += d * s;
            }
            acc
        })
        .collect()
}

#[test]
fn the_fused_tq2_gemv_matches_the_reference_for_two_and_three_parts() {
    let Some(()) = isolated(|_graph| {
        let epoch = MetalGraph::next_model_epoch();
        let k = 256usize;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.37).sin()).collect();
        for (slot, rows) in [(0x10u64, vec![48usize, 16, 16]), (0x20, vec![64, 64])] {
            let parts: Vec<Vec<u8>> = rows
                .iter()
                .enumerate()
                .map(|(i, r)| tq2(*r, k, slot + i as u64))
                .collect();
            let part_refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
            let n_rows: usize = rows.iter().sum();
            let mut out = vec![0f32; n_rows];
            try_metal_gemv_tq2_fused(&x, &mut out, epoch, slot, &part_refs, n_rows, k)
                .expect("fused tq2 gemv");
            let concat: Vec<u8> = parts.concat();
            let reference = tq2_gemv_reference(&concat, &x, n_rows, k);
            for (i, (g, r)) in out.iter().zip(&reference).enumerate() {
                assert!(
                    (g - r).abs() <= 1e-4 * r.abs().max(1.0),
                    "{}-part row {i}: gpu {g} vs reference {r}",
                    rows.len()
                );
            }
            // Second call: a cache hit, same answer.
            let mut again = vec![0f32; n_rows];
            try_metal_gemv_tq2_fused(&x, &mut again, epoch, slot, &part_refs, n_rows, k)
                .expect("fused tq2 gemv (hit)");
            assert_eq!(bits(&again), bits(&out));
        }
    }) else {
        return;
    };
}

#[test]
fn the_fused_tq2_gemv_refuses_a_foreign_buffer_and_mismatched_parts() {
    // Parts that do not add up to `n_rows` rows: rejected before any GPU work.
    let k = 128usize;
    let q = tq2(8, k, 1);
    let x = vec![0.5f32; k];
    let mut out = vec![0f32; 24];
    let err = try_metal_gemv_tq2_fused(&x, &mut out, LEGACY_MODEL_EPOCH, 1, &[&q], 24, k)
        .expect_err("8 rows of parts cannot serve 24 output rows");
    assert!(
        matches!(err, MetalGraphError::InvalidDimensions(_)),
        "{err:?}"
    );

    let Some(()) = isolated(|graph| {
        let epoch = MetalGraph::next_model_epoch();
        // Another producer put a Q-only buffer in the slot the fused call is
        // about to use — the CUDA M-21 defect, reproduced on Metal.
        graph
            .get_or_upload_tq2_weight_soa_for_epoch(epoch, 0x55, &q)
            .expect("seed a q-only buffer");
        let (k_part, v_part) = (tq2(8, k, 2), tq2(8, k, 3));
        let mut out = vec![0f32; 24];
        let err =
            try_metal_gemv_tq2_fused(&x, &mut out, epoch, 0x55, &[&q, &k_part, &v_part], 24, k)
                .expect_err("a resident buffer of the wrong size must be refused");
        assert!(err.to_string().contains("refusing"), "{err}");
    }) else {
        return;
    };
}
