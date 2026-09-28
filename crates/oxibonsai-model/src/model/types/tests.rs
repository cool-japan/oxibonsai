//! Regression tests for `BonsaiModel`'s forward path.
//!
//! Covers MET-05 (no silent backend switch), M-02/M-32 (row-wise embedding),
//! M-22 (zero per-token allocations), M-23 (SIMD LM head), M-25 (deterministic
//! dominant quant type), M-29 (no panics on the forward path), M-33 + the
//! O(1)-memory config-only constructor, sec-11 (context length is the
//! configured one, clamped and grown) and M-Missed-3 (`reset`).

use super::*;
use crate::test_alloc::count_allocations;
use oxibonsai_core::config::RopeScaling;
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_kernels::{KernelDispatcher, KernelTier};

/// Reference (scalar CPU) dispatcher: `is_gpu_accelerated() == false`, so the
/// forward pass takes the host-KV block path on every machine.
fn cpu_kernel() -> KernelDispatcher {
    KernelDispatcher::with_tier(KernelTier::Reference)
}

/// Small config whose dimensions satisfy the Q1_0_g128 fixture constraints
/// (`hidden % 128 == 0`, `intermediate % 128 == 0`).
fn small_config(num_layers: usize, vocab: usize, max_ctx: usize) -> Qwen3Config {
    Qwen3Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_layers,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        value_length: 64,
        vocab_size: vocab,
        max_context_length: max_ctx,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10_000.0,
        rope_scaling: RopeScaling::None,
        // FIX3-MODEL added this field (M-17). Every test in this file wants
        // the full-causal path, which is what `None` selects.
        sliding_window: None,
        architecture: "test".to_string(),
        model_name: "test".to_string(),
    }
}

/// Deterministic dense FP32 LM head of shape `[out_features × in_features]`.
fn dense_lm_head(out_features: usize, in_features: usize) -> OutputWeight<'static> {
    let weights = (0..out_features * in_features)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    OutputWeight::Fp32 {
        weights,
        out_features,
        in_features,
    }
}

// ── M-23: SIMD LM head ────────────────────────────────────────────────────

/// Naive reference: exactly the scalar triple loop `forward` used to run.
fn lm_head_reference(
    weights: &[f32],
    input: &[f32],
    out: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    for (i, logit) in out.iter_mut().enumerate().take(out_features) {
        let row_start = i * in_features;
        let mut sum = 0.0f32;
        for j in 0..in_features {
            sum += weights[row_start + j] * input[j];
        }
        *logit = sum;
    }
}

#[test]
fn lm_head_simd_matches_the_scalar_reference() {
    // Below and above the Rayon row threshold, and with an `in_features` that
    // is not a multiple of the lane count so the remainder loop is exercised.
    for &(out_features, in_features) in &[(8usize, 8usize), (16, 130), (512, 128), (300, 65)] {
        let weights: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.031)
            .collect();
        let input: Vec<f32> = (0..in_features)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.17)
            .collect();
        let mut expected = vec![0.0f32; out_features];
        lm_head_reference(&weights, &input, &mut expected, out_features, in_features);
        let mut actual = vec![0.0f32; out_features];
        lm_head::forward_f32(&weights, &input, &mut actual, out_features, in_features)
            .expect("SIMD LM head");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1e-4 * e.abs().max(1.0),
                "row {i} of [{out_features}x{in_features}]: simd={a} scalar={e}"
            );
        }
    }
}

#[test]
fn lm_head_empty_weights_is_the_zero_matrix() {
    // M-33: the weight-less constructors store the all-zero LM head compactly.
    let mut logits = vec![7.0f32; 12];
    lm_head::forward_f32(&[], &[1.0; 4], &mut logits, 12, 4).expect("zero LM head");
    assert!(logits.iter().all(|&v| v == 0.0), "{logits:?}");
}

#[test]
fn lm_head_rejects_short_buffers() {
    let weights = vec![0.5f32; 4 * 3];
    let mut logits = vec![0.0f32; 2];
    assert!(lm_head::forward_f32(&weights, &[1.0; 3], &mut logits, 4, 3).is_err());
    let mut logits = vec![0.0f32; 4];
    assert!(lm_head::forward_f32(&weights, &[1.0; 2], &mut logits, 4, 3).is_err());
    assert!(lm_head::forward_f32(&weights[..5], &[1.0; 3], &mut logits, 4, 3).is_err());
}

#[test]
fn lm_head_dot_handles_ragged_lengths() {
    let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..19).map(|i| (19 - i) as f32).collect();
    let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    assert!((lm_head::dot_f32(&a, &b) - expected).abs() < 1e-3);
    // Shorter operand wins.
    assert!((lm_head::dot_f32(&a[..5], &b) - lm_head::dot_f32(&a[..5], &b[..5])).abs() < 1e-6);
}

// ── M-02 / M-32 / M-33: embedding table ───────────────────────────────────

#[test]
fn constant_embedding_costs_nothing_and_yields_the_expected_row() {
    let table = EmbeddingTable::constant(0.25, 151_936, 4096);
    assert_eq!(table.len(), 151_936 * 4096);
    assert_eq!(
        table.resident_bytes(),
        0,
        "a synthetic table must not materialize {} elements",
        table.len()
    );
    let mut row = vec![0.0f32; 4096];
    table.copy_row(1234, &mut row).expect("row");
    assert!(row.iter().all(|&v| v == 0.25));
}

#[test]
fn embedding_row_lookup_is_bounds_checked() {
    let table = EmbeddingTable::constant(1.0, 4, 8);
    let mut row = vec![0.0f32; 8];
    assert!(table.copy_row(3, &mut row).is_ok());
    assert!(
        table.copy_row(4, &mut row).is_err(),
        "token id == vocab must be rejected"
    );
    let mut short = vec![0.0f32; 7];
    assert!(table.copy_row(0, &mut short).is_err());
}

#[test]
fn dense_embedding_shares_one_allocation_and_reports_it() {
    let hidden = 4;
    let vocab = 3;
    let data: Vec<f32> = (0..vocab * hidden).map(|i| i as f32).collect();
    let arc: std::sync::Arc<[f32]> = data.into();
    let table = EmbeddingTable::dense(std::sync::Arc::clone(&arc), vocab, hidden);
    let mut row = vec![0.0f32; hidden];
    table.copy_row(2, &mut row).expect("row");
    assert_eq!(row, vec![8.0, 9.0, 10.0, 11.0]);
    // The dense view is the very same allocation, not a copy.
    let handle = table.dense_handle().expect("dense handle");
    assert!(std::sync::Arc::ptr_eq(&handle, &arc));
    assert_eq!(table.resident_bytes(), vocab * hidden * 4);
    // The batched gather the GPU prefill paths use sees the same numbers as
    // `copy_row` (M-02, wave 2.5 — this replaces the `&table[8..12]` dense
    // `Index` assertion, which no longer exists).
    let mut batch = vec![0.0f32; 2 * hidden];
    table.copy_rows(&[1, 2], &mut batch).expect("batch");
    assert_eq!(batch, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
}

/// A non-dense table never materializes anything — not on a row lookup and
/// not on the batched gather the GPU prefill paths use (M-02, wave 2.5).
///
/// This is the strengthened successor of
/// `quantized_embedding_materializes_only_on_dense_index`, which asserted the
/// *old* behaviour: that the dense `Index<Range<usize>>` escape hatch
/// materialized `vocab × hidden` FP32 elements on first use. That hatch is
/// deleted, so the assertion flips from "materializes once" to "never
/// materializes" — the memory win M-02 exists to deliver.
#[test]
fn a_quantized_embedding_never_materializes_a_dense_table() {
    // A constant table stands in for a quantized one here: both synthesize
    // rows with no dense allocation behind them.
    let table = EmbeddingTable::constant(0.5, 6, 4);
    assert_eq!(table.resident_bytes(), 0);
    let mut row = vec![0.0f32; 4];
    table.copy_row(5, &mut row).expect("row");
    assert_eq!(row, vec![0.5; 4]);
    assert_eq!(table.resident_bytes(), 0, "copy_row must stay lazy");
    let mut batch = vec![0.0f32; 3 * 4];
    table.copy_rows(&[0, 3, 5], &mut batch).expect("batch");
    assert_eq!(batch, vec![0.5; 12]);
    assert_eq!(
        table.resident_bytes(),
        0,
        "the batched gather must stay lazy too — there is no escape hatch left"
    );
    assert!(
        table.dense_handle().is_none(),
        "a synthetic table has no dense allocation to hand out"
    );
}

// ── M-25: deterministic dominant quantization type ────────────────────────

#[test]
fn dominant_quant_type_ignores_float_tensors() {
    // 113 F32 norms against 98 ternary matrices: the quantized type still wins,
    // which is the whole point of the tensor filter.
    let counts = vec![
        (GgufTensorType::F32, 113usize),
        (GgufTensorType::TQ2_0_g128, 98),
        (GgufTensorType::F16, 4),
    ];
    assert_eq!(
        dominant_from_counts(counts.into_iter()),
        GgufTensorType::TQ2_0_g128
    );
}

#[test]
fn dominant_quant_type_breaks_ties_by_lowest_type_id() {
    let a = vec![
        (GgufTensorType::TQ2_0_g128, 7usize),
        (GgufTensorType::Q1_0_g128, 7),
    ];
    let b = vec![
        (GgufTensorType::Q1_0_g128, 7usize),
        (GgufTensorType::TQ2_0_g128, 7),
    ];
    let first = dominant_from_counts(a.into_iter());
    let second = dominant_from_counts(b.into_iter());
    assert_eq!(first, second, "iteration order must not change the answer");
    assert!(
        (GgufTensorType::Q1_0_g128 as u32) < (GgufTensorType::TQ2_0_g128 as u32),
        "test assumes Q1_0_g128 has the lower type id"
    );
    assert_eq!(first, GgufTensorType::Q1_0_g128);
}

#[test]
fn dominant_quant_type_falls_back_when_no_weights_are_quantized() {
    let counts = vec![(GgufTensorType::F32, 9usize)];
    assert_eq!(
        dominant_from_counts(counts.into_iter()),
        GgufTensorType::Q1_0_g128
    );
}

// ── M-29: no panics on the forward path ───────────────────────────────────

#[test]
fn output_weight_kind_names_every_variant() {
    assert_eq!(OutputWeight::zero_fp32(4, 4).kind(), "F32");
    // The remaining variants need real weight blocks to construct; their names
    // are compile-time constants, so the invariant worth testing here is that
    // the diagnostic never returns an empty string for the one variant a
    // weight-less model can hold.
    assert!(!OutputWeight::zero_fp32(4, 4).kind().is_empty());
}

#[test]
fn forward_rejects_a_short_logits_buffer_instead_of_panicking() {
    let mut model = BonsaiModel::new(small_config(0, 32, 64));
    let kernel = cpu_kernel();
    let mut logits = vec![0.0f32; 8];
    let err = model
        .forward_into(0, 0, &kernel, &mut logits)
        .expect_err("short buffer must be rejected");
    assert!(matches!(err, ModelError::ShapeMismatch { .. }), "{err}");
}

#[test]
fn forward_rejects_an_out_of_range_token_instead_of_panicking() {
    let mut model = BonsaiModel::new(small_config(0, 32, 64));
    let kernel = cpu_kernel();
    assert!(model.forward(31, 0, &kernel).is_ok());
    assert!(model.forward(32, 0, &kernel).is_err());
}

// ── MET-05: no silent GPU→CPU backend switch ──────────────────────────────

#[test]
fn gpu_fallback_error_round_trips_through_its_accessor() {
    let err = gpu_fallback_requires_cache_rebuild(4096);
    assert_eq!(gpu_fallback_cache_rebuild_pos(&err), Some(4096));
    // MET-05 (wave 2.5): the marker is now the typed variant's `error_code()`,
    // not a prefix inside the message, so assert on the code itself — a
    // stricter check than the substring test it replaces (the code is the
    // whole string, not a fragment of a longer sentence).
    assert_eq!(err.error_code(), GPU_FALLBACK_CACHE_REBUILD_CODE);
    assert!(
        matches!(err, ModelError::GpuFallbackRequiresCacheRebuild { pos } if pos == 4096),
        "expected the typed variant, got: {err}"
    );
    assert!(
        err.to_string().contains("4096"),
        "the message must still name the position: {err}"
    );
    assert_eq!(
        gpu_fallback_cache_rebuild_pos(&ModelError::Internal("something else".into())),
        None
    );
    assert_eq!(
        gpu_fallback_cache_rebuild_pos(&ModelError::SequenceTooLong {
            seq_len: 1,
            max_ctx: 1
        }),
        None
    );
}

#[test]
fn sampled_forward_refuses_to_attend_over_a_stale_host_kv_cache() {
    let mut model = BonsaiModel::new_for_testing_with_blocks(small_config(1, 64, 512));
    let kernel = cpu_kernel();
    // Positions 0 and 1 run on the CPU: the host cache holds them.
    model.forward(1, 0, &kernel).expect("cpu pos 0");
    model.forward(2, 1, &kernel).expect("cpu pos 1");
    // A fused GPU path then takes over: it maintains its OWN device KV cache
    // and writes nothing to the host cache.
    model.note_device_kv_used();
    assert!(model.gpu_path_active());
    // Continuing on the CPU at the next contiguous position is still fine —
    // the host cache really does hold 0..=1.
    model.forward(3, 2, &kernel).expect("contiguous cpu step");
    model.note_device_kv_used();
    // ... but jumping to a position whose history only the GPU has must NOT
    // silently return logits computed over zeros.
    let err = model
        .forward(4, 9, &kernel)
        .expect_err("silent backend switch must be refused");
    assert_eq!(
        gpu_fallback_cache_rebuild_pos(&err),
        Some(9),
        "expected the distinguished rebuild error, got: {err}"
    );
}

#[test]
fn host_kv_rebuild_notification_reopens_the_cpu_path() {
    let mut model = BonsaiModel::new_for_testing_with_blocks(small_config(1, 64, 512));
    let kernel = cpu_kernel();
    model.note_device_kv_used();
    assert!(model.forward(1, 6, &kernel).is_err());
    // The runtime replays the committed prefix on the CPU and says so.
    model.mark_host_kv_rebuilt(6);
    assert!(!model.gpu_path_active());
    model
        .forward(1, 6, &kernel)
        .expect("after the rebuild the CPU path is coherent again");
}

#[test]
fn reset_clears_the_backend_latch_and_the_host_watermark() {
    let mut model = BonsaiModel::new_for_testing_with_blocks(small_config(1, 64, 512));
    let kernel = cpu_kernel();
    model.forward(1, 0, &kernel).expect("pos 0");
    model.note_device_kv_used();
    assert!(model.gpu_path_active());
    // M-Missed-3: a new conversation starts from a clean slate.
    model.reset();
    assert!(!model.gpu_path_active());
    assert_eq!(model.kv_cache().seq_len(), 0);
    model
        .forward(1, 0, &kernel)
        .expect("a fresh sequence needs no rebuild");
    model.reset_cache();
    assert!(!model.gpu_path_active());
}

#[test]
fn prefill_fallback_refuses_a_stale_host_cache_before_looping() {
    let mut model = BonsaiModel::new_for_testing_with_blocks(small_config(1, 64, 512));
    let kernel = cpu_kernel();
    model.note_device_kv_used();
    let err = model
        .forward_prefill(&[1, 2, 3], 12, &kernel)
        .expect_err("sequential prefill fallback must not attend over zeros");
    assert_eq!(gpu_fallback_cache_rebuild_pos(&err), Some(12));
    let err = model
        .forward_prefill_verify(&[1, 2, 3], 12, &kernel)
        .expect_err("sequential verify fallback must not attend over zeros");
    assert_eq!(gpu_fallback_cache_rebuild_pos(&err), Some(12));
}

#[test]
fn force_cpu_decode_seam_parses_and_applies() {
    // The seam is parsed from `OXIBONSAI_FORCE_CPU_DECODE_AFTER` (the variable
    // `InferenceEngine::generate` already reads for the greedy loop) once per
    // model; the parser is tested directly so no test mutates process-wide
    // environment state that its neighbours share.
    assert_eq!(parse_force_cpu_after(None), None);
    assert_eq!(
        parse_force_cpu_after(Some("not a number".to_string())),
        None
    );
    assert_eq!(parse_force_cpu_after(Some("3".to_string())), Some(3));

    let mut model = BonsaiModel::new(small_config(0, 32, 64));
    assert!(!model.force_cpu_at(0), "unset seam never forces the CPU");
    model.force_cpu_decode_after = Some(2);
    assert!(!model.force_cpu_at(1));
    assert!(
        model.force_cpu_at(2),
        "positions at/after the seam force the CPU"
    );
    assert!(model.force_cpu_at(9));
}

// ── sec-11: context length comes from the configuration ───────────────────

#[test]
fn context_length_is_the_configured_one_not_a_hard_coded_4096() {
    let model = BonsaiModel::new(small_config(0, 32, 300));
    assert_eq!(model.max_context(), 300);
    let capped = BonsaiModel::new_with_context_cap(small_config(0, 32, 300), Some(64));
    assert_eq!(
        capped.max_context(),
        64,
        "a RAM-derived cap must clamp the configured context"
    );
}

#[test]
fn positions_beyond_the_configured_context_are_rejected() {
    let mut model = BonsaiModel::new(small_config(0, 32, 40));
    let kernel = cpu_kernel();
    model.forward(0, 39, &kernel).expect("last valid position");
    let err = model
        .forward(0, 40, &kernel)
        .expect_err("a 40-context config must reject position 40");
    match err {
        ModelError::SequenceTooLong { seq_len, max_ctx } => {
            assert_eq!((seq_len, max_ctx), (41, 40));
        }
        other => panic!("expected SequenceTooLong, got {other}"),
    }
}

#[test]
fn kv_cache_grows_on_demand_and_keeps_its_contents() {
    let mut model = BonsaiModel::new(small_config(2, 32, 512));
    let start = model.kv_cache().allocated_seq_len();
    assert!(
        start <= WEIGHTLESS_PREALLOC_CONTEXT,
        "the config-only constructor must start small, got {start}"
    );
    assert_eq!(
        model.kv_cache().max_seq_len(),
        512,
        "the cache's logical limit is the effective context from the start"
    );
    assert!(
        model.kv_cache().is_f16() && model.kv_cache().is_lazy(),
        "REQUIRED #4 (3)+(4): the host default is a lazy f16 cache"
    );
    let head_dim = model.config().head_dim;
    // Exactly representable in f16, so the round trip is exact.
    let key: Vec<f32> = (0..head_dim).map(|i| i as f32 + 0.5).collect();
    model
        .kv_cache_mut()
        .try_store_key(0, 0, 3, &key)
        .expect("seed a cached key");
    model.host_kv_written = 4;

    let kernel = cpu_kernel();
    model.forward(0, start + 5, &kernel).expect("grown forward");
    assert!(
        model.kv_cache().allocated_seq_len() > start + 5,
        "cache must have grown past the requested position"
    );
    assert_eq!(model.kv_cache().max_seq_len(), 512, "the limit never moves");
    assert_eq!(
        model.rope.max_seq_len(),
        model.kv_cache().allocated_seq_len()
    );
    let kept = model.kv_cache().keys_for(0, 0, 4);
    assert_eq!(
        &kept[3 * head_dim..4 * head_dim],
        &key[..],
        "growth must preserve every cached position"
    );
}

/// Replaces "a loaded model's caches do not grow behind the GPU's back"
/// (REQUIRED #4 (4) makes every model's host KV grow lazily): the invariant
/// that rule protected — a device-side KV cache must never be re-geometried
/// mid-sequence — now holds by construction, because every device cache is
/// sized from the host cache's fixed LOGICAL limit (`max_seq_len()`), which
/// growth never changes. So growth is allowed even while the MET-05 GPU
/// latch is set, and the device geometry stays put across it.
#[test]
fn host_kv_growth_never_changes_the_device_kv_geometry() {
    let mut model = BonsaiModel::new(small_config(0, 32, 4096));
    let device_geometry = model.kv_cache().max_seq_len();
    assert_eq!(device_geometry, 4096);
    let before = model.kv_cache().allocated_seq_len();
    model.note_device_kv_used();
    assert!(model.gpu_path_active());
    model
        .ensure_context_capacity(1000)
        .expect("growth within the effective context, GPU latch or not");
    assert!(model.kv_cache().allocated_seq_len() > 1000);
    assert!(model.kv_cache().allocated_seq_len() > before);
    assert_eq!(
        model.kv_cache().max_seq_len(),
        device_geometry,
        "host growth must never move the geometry a device KV cache is sized from"
    );
    // The effective context is still the hard ceiling.
    let err = model
        .ensure_context_capacity(4096)
        .expect_err("position 4096 is past a 4096-token context");
    match err {
        ModelError::SequenceTooLong { seq_len, max_ctx } => {
            assert_eq!((seq_len, max_ctx), (4097, 4096));
        }
        other => panic!("expected SequenceTooLong, got {other}"),
    }
}

/// Lazy growth across a chunk boundary on the real decode loop, with a
/// populated model: every position written before the growth is still
/// there after it (REQUIRED #4 (4)'s "growth across a chunk boundary").
#[test]
fn decode_loop_grows_the_host_kv_across_a_chunk_boundary() {
    let config = small_config(1, 32, 1024);
    let mut model = BonsaiModel::new_for_testing_with_blocks(config.clone());
    let kernel = cpu_kernel();
    let first = model.kv_cache().allocated_seq_len();
    assert_eq!(first, crate::kv_cache::GROWTH_CHUNK_POSITIONS);
    for pos in 0..first + 3 {
        model
            .forward(u32::try_from(pos % 32).expect("fits"), pos, &kernel)
            .expect("decode step");
    }
    assert!(model.kv_cache().allocated_seq_len() > first);
    let seq = first + 3;
    let keys = model.kv_cache().keys_for(0, 0, seq);
    // Positions before the boundary were written and survive the growth
    // (a fresh position's key is never all-zero on this fixture).
    for pos in [0usize, first / 2, first - 1, first, first + 2] {
        let row = &keys[pos * config.head_dim..(pos + 1) * config.head_dim];
        assert!(
            row.iter().any(|&x| x != 0.0),
            "position {pos}: key lost across the chunk-boundary growth"
        );
    }
}

#[test]
fn a_loaded_model_pre_allocates_the_whole_context_it_was_given() {
    // Regression: a loaded model cannot grow (its device KV cache is allocated
    // from `kv_cache.max_seq_len()`), so pre-allocating less than the effective
    // context would deny the caller the `--max-seq-len` they asked for while
    // `max_context()` still advertised it. `effective_context` is what bounds
    // the request; the pre-allocation must not shrink it further.
    for &requested in &[512usize, 4096, 8192, 32_768] {
        assert_eq!(
            prealloc_context(requested, None),
            requested,
            "a non-growing model must pre-allocate its whole effective context"
        );
    }
    assert_eq!(prealloc_context(0, None), 1, "never a degenerate cache");
    // A growing model starts small and is capped by its window.
    assert_eq!(prealloc_context(32_768, Some(128)), 128);
    assert_eq!(prealloc_context(64, Some(128)), 64);
    assert_eq!(prealloc_context(0, Some(128)), 1);
}

#[test]
fn a_growing_model_reaches_the_context_its_config_declares() {
    // End-to-end counterpart of the unit check above: a model that starts with
    // a small window must still serve every position up to `max_context`.
    let mut model = BonsaiModel::new(small_config(0, 32, 600));
    let kernel = cpu_kernel();
    model.forward(0, 599, &kernel).expect("last valid position");
    assert!(model.kv_cache().max_seq_len() >= 600);
    assert!(model.kv_cache().allocated_seq_len() >= 600);
    assert!(model.forward(0, 600, &kernel).is_err());
}

#[test]
fn effective_context_takes_the_tightest_constraint() {
    let config = small_config(0, 32, 1000);
    assert_eq!(effective_context(&config, None, None), 1000);
    assert_eq!(effective_context(&config, Some(4096), None), 1000);
    assert_eq!(effective_context(&config, Some(256), None), 256);
    assert_eq!(effective_context(&config, Some(256), Some(128)), 128);
    // A zero means "no constraint from that source", never a zero-sized cache.
    let mut unbounded = small_config(0, 32, 0);
    unbounded.max_context_length = 0;
    assert_eq!(
        effective_context(&unbounded, None, None),
        MAX_PREALLOC_CONTEXT
    );
    assert_eq!(effective_context(&unbounded, Some(77), None), 77);
}

// ── M-33 + addendum: the config-only constructor is O(1) memory ───────────

#[test]
fn weightless_constructor_is_o1_memory_for_the_8b_config() {
    let config = Qwen3Config::bonsai_8b();
    let vocab_x_hidden = config.vocab_size * config.hidden_size;
    let (model, allocs) = count_allocations(|| BonsaiModel::new(Qwen3Config::bonsai_8b()));
    let footprint = model.footprint_bytes();
    assert!(
        footprint < 64 * 1024 * 1024,
        "BonsaiModel::new(bonsai_8b()) must stay under 64 MiB, got {footprint} B \
         ({} MiB); a materialized token_embd alone would be {} MiB",
        footprint / (1024 * 1024),
        vocab_x_hidden * 4 / (1024 * 1024),
    );
    assert!(
        allocs < 64,
        "the constructor must make O(1) allocations, got {allocs}"
    );
    // The model is still fully usable, and still reports the real vocabulary.
    assert_eq!(model.config().vocab_size, config.vocab_size);
    assert!(model.kv_cache_memory_bytes() > 0);
}

#[test]
fn weightless_forward_still_produces_zero_logits() {
    // Behaviour parity with the materialized zero tables it replaces.
    let mut model = BonsaiModel::new(small_config(0, 48, 64));
    let kernel = cpu_kernel();
    let logits = model.forward(7, 0, &kernel).expect("forward");
    assert_eq!(logits.len(), 48);
    assert!(logits.iter().all(|&v| v == 0.0), "{logits:?}");
}

// ── M-22: zero heap allocations per token ─────────────────────────────────

/// Run `f` inside a **private** Rayon thread pool instead of the process-wide
/// default one.
///
/// `lm_head::forward_f32`'s parallel branch (and `oxibonsai-kernels`' own
/// dispatch inside `TransformerBlock::forward`) call into `rayon::prelude`.
/// From a thread that is not itself a Rayon worker — exactly this test's
/// thread — every such top-level call goes through
/// `rayon_core::registry::Registry::in_worker_cold`, which injects the job
/// into the pool's shared `crossbeam_deque::Injector` and blocks until it
/// completes. That injector grows in fixed-size blocks and allocates a new
/// one whenever the current block fills, which is a genuine allocation and
/// is attributed (correctly) to whichever thread's push crosses the
/// boundary.
///
/// Against the *default, process-wide* pool, that boundary is shared with
/// every other test in this binary that also uses Rayon concurrently
/// (`cargo test` runs the whole crate's tests in one process by default), so
/// which test's thread pays for the next block is a race — confirmed by
/// temporarily capturing a backtrace on the phantom allocation this test used
/// to see intermittently: `crossbeam_deque::deque::Injector::push` ->
/// `Registry::inject` -> `Registry::in_worker_cold`, reached from this same
/// `forward_into` call, racing an unrelated concurrently-running test's own
/// `.par_iter()` call for the next slot in the *shared* injector — not a leak
/// in this crate's forward path. A pool built fresh for one test starts with
/// an empty injector that only this test ever pushes into, and the handful
/// of calls a test makes below never fill even one block, so the allocation
/// this function exists to dodge has no shared boundary left to race on.
///
/// A brand-new pool has its own second one-time cost, confirmed the same way:
/// `rayon_core::sleep::Sleep` (the idle-worker coordination a thread falls
/// into via `join_context` -> `wait_until_cold` while it waits on a
/// still-running job) lazily boxes a pthread `Mutex` *and* `Condvar`
/// (`std::sys::sync::once_box::OnceBox::initialize`) the first time each
/// participant actually blocks on them. A join whose both halves finish
/// near-instantly can occasionally resolve without either side ever really
/// blocking, and which of the pool's threads ends up parking on a given
/// round is itself scheduler-dependent, so a single round — or even a
/// handful, measured empirically — is a probability of full coverage, not a
/// guarantee. Below, `num_threads(1)` first cuts this down to one worker
/// (plus the calling thread) needing to be warmed at all, and 16 rounds of a
/// join whose second half is forced slower than the first — so *something*
/// always blocks for real — brought repeated full-suite stress runs (`cargo
/// test -p oxibonsai-model --lib`, tens of consecutive invocations) to zero
/// observed failures where 1 round left an intermittent one in roughly 1 in
/// 5.
fn with_isolated_rayon_pool<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    // A single worker thread, not two: every worker has its own `Sleep`
    // state to warm (see the doc comment above), and cutting the pool to the
    // minimum that can still exercise the inject-and-wait path halves how
    // much of that state a fixed warm-up budget has to cover.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("private rayon pool for allocation-sensitive test");
    pool.install(|| {
        for _ in 0..16 {
            rayon::join(
                || {},
                || std::thread::sleep(std::time::Duration::from_millis(2)),
            );
        }
    });
    pool.install(f)
}

#[test]
fn forward_into_is_allocation_free_per_token() {
    with_isolated_rayon_pool(|| {
        let config = small_config(1, 512, 512);
        let mut model = BonsaiModel::new_for_testing_with_blocks(config.clone());
        // Exercise the real SIMD LM head (512 rows is above the Rayon
        // threshold), not the compact zero one.
        model.output_weight = dense_lm_head(config.vocab_size, config.hidden_size);
        let kernel = cpu_kernel();
        let mut logits = vec![0.0f32; config.vocab_size];

        // Warm-up: sizes the scratch buffers and touches every
        // lazily-initialized global exactly once, on this test's own private
        // pool.
        for pos in 0..4 {
            model
                .forward_into(pos as u32, pos, &kernel, &mut logits)
                .expect("warm-up token");
        }

        for pos in 4..8 {
            let (result, allocs) =
                count_allocations(|| model.forward_into(pos as u32, pos, &kernel, &mut logits));
            result.expect("measured token");
            assert_eq!(
                allocs, 0,
                "pos={pos}: forward_into must not allocate per token, got {allocs}"
            );
        }
    });
}

#[test]
fn forward_into_is_allocation_free_per_token_serial_lm_head() {
    // `count_allocations` (crate::test_alloc) only sees allocations made on
    // the CALLING thread. `forward_into_is_allocation_free_per_token` uses
    // vocab=512, which is >= `lm_head`'s Rayon threshold (256 rows) and so
    // runs the LM head across worker threads the counter cannot see — an
    // allocation there would pass that test silently. vocab=128 keeps
    // `out_features` below the threshold, forcing `lm_head::forward_f32`'s
    // serial `for` loop on the calling thread itself, where the counter
    // actually covers it.
    with_isolated_rayon_pool(|| {
        let config = small_config(1, 128, 512);
        let mut model = BonsaiModel::new_for_testing_with_blocks(config.clone());
        model.output_weight = dense_lm_head(config.vocab_size, config.hidden_size);
        let kernel = cpu_kernel();
        let mut logits = vec![0.0f32; config.vocab_size];

        for pos in 0..4 {
            model
                .forward_into(pos as u32, pos, &kernel, &mut logits)
                .expect("warm-up token");
        }

        for pos in 4..8 {
            let (result, allocs) =
                count_allocations(|| model.forward_into(pos as u32, pos, &kernel, &mut logits));
            result.expect("measured token");
            assert_eq!(
                allocs, 0,
                "pos={pos}: serial-branch forward_into must not allocate per token, got {allocs}"
            );
        }
    });
}

#[test]
fn forward_into_is_allocation_free_without_blocks() {
    // Narrower guard on the code this package owns end to end: embedding row
    // lookup -> output norm -> LM head, with no `TransformerBlock` involved.
    with_isolated_rayon_pool(|| {
        let config = small_config(0, 512, 64);
        let mut model = BonsaiModel::new(config.clone());
        model.output_weight = dense_lm_head(config.vocab_size, config.hidden_size);
        let kernel = cpu_kernel();
        let mut logits = vec![0.0f32; config.vocab_size];
        model
            .forward_into(1, 0, &kernel, &mut logits)
            .expect("warm-up");
        let (result, allocs) = count_allocations(|| model.forward_into(2, 1, &kernel, &mut logits));
        result.expect("measured token");
        assert_eq!(allocs, 0, "embedding + norm + LM head must not allocate");
    });
}

#[test]
fn forward_and_forward_into_agree() {
    let config = small_config(1, 96, 128);
    let mut model = BonsaiModel::new_for_testing_with_blocks(config.clone());
    model.output_weight = dense_lm_head(config.vocab_size, config.hidden_size);
    let kernel = cpu_kernel();
    let owned = model.forward(5, 0, &kernel).expect("forward");
    model.reset();
    let mut into = vec![0.0f32; config.vocab_size];
    model
        .forward_into(5, 0, &kernel, &mut into)
        .expect("forward_into");
    assert_eq!(owned, into);
    assert_eq!(owned.len(), config.vocab_size);
}

// ── engine-pool seam (the shared token-embedding handle) ──────────────────

#[test]
fn shared_token_embd_handle_is_stable_for_a_weightless_model() {
    let model = BonsaiModel::new(small_config(0, 32, 64));
    let a = model.shared_token_embd();
    let b = model.shared_token_embd();
    assert!(
        std::sync::Arc::ptr_eq(&a, &b),
        "repeated handle pulls must alias one allocation"
    );
    assert!(a.is_empty(), "a synthetic table has nothing dense to share");
}

// ═══════════════════════════════════════════════════════════════════════════
// M-08 — RoPE scaling is APPLIED, not just parsed
// ═══════════════════════════════════════════════════════════════════════════

use crate::layers::rope_scaling::RopeScalingStrategy;

/// The YaRN declaration `models/Bonsai-8B.gguf` really carries, scaled down to
/// a small `original_context_length` so a table that covers it is cheap.
fn yarn_scaling(original_context_length: u32) -> RopeScaling {
    RopeScaling::Yarn {
        factor: 4.0,
        original_context_length,
        attn_factor: None,
        beta_fast: None,
        beta_slow: None,
    }
}

/// The strategy `yarn_scaling` must convert to — the upstream `beta_fast`
/// (32.0) and `beta_slow` (1.0) defaults spelled out.
fn yarn_strategy(original_max_position: usize) -> RopeScalingStrategy {
    RopeScalingStrategy::Yarn {
        original_max_position,
        factor: 4.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attn_factor: None,
    }
}

/// Largest absolute cos/sin difference between two tables over `rows` rows.
fn max_table_delta(a: &RopeTable, b: &RopeTable, rows: usize) -> f32 {
    let mut worst = 0.0f32;
    for pos in 0..rows {
        for (x, y) in a.cos_at(pos).iter().zip(b.cos_at(pos)) {
            worst = worst.max((x - y).abs());
        }
        for (x, y) in a.sin_at(pos).iter().zip(b.sin_at(pos)) {
            worst = worst.max((x - y).abs());
        }
    }
    worst
}

/// A config-only model must get the very table `new_with_scaling` builds —
/// not the unscaled one `RopeTable::new` used to produce (M-08 BLOCKING 1).
#[test]
fn a_config_only_model_gets_the_scaled_rope_table() {
    let mut config = small_config(0, 32, 4096);
    config.rope_scaling = yarn_scaling(1024);
    let model = BonsaiModel::new(config.clone());
    let rows = model.rope.max_seq_len();
    assert!(rows > 0);

    let expected = RopeTable::new_with_scaling(
        config.head_dim,
        rows,
        config.rope_freq_base,
        Some(&yarn_strategy(1024)),
    )
    .expect("the fixture's YaRN parameters are valid");
    assert_eq!(
        max_table_delta(&model.rope, &expected, rows),
        0.0,
        "the model's table must be exactly `new_with_scaling`'s"
    );

    let unscaled = RopeTable::new(config.head_dim, rows, config.rope_freq_base);
    assert!(
        max_table_delta(&model.rope, &unscaled, rows) > 1e-3,
        "a YaRN table must differ from an unscaled one — otherwise the wiring \
         is indistinguishable from the bug"
    );
    assert_ne!(
        model.rope.attention_scale(),
        1.0,
        "YaRN folds an mscale into cos/sin; an unscaled table has none"
    );
}

/// Declaring no scaling must still give exactly the unscaled table, so the
/// wiring is a no-op for every model that does not ask for it.
#[test]
fn a_config_without_scaling_is_identical_to_the_unscaled_table() {
    let config = small_config(0, 32, 4096);
    assert_eq!(config.rope_scaling, RopeScaling::None);
    let model = BonsaiModel::new(config.clone());
    let rows = model.rope.max_seq_len();
    let unscaled = RopeTable::new(config.head_dim, rows, config.rope_freq_base);
    assert_eq!(max_table_delta(&model.rope, &unscaled, rows), 0.0);
    assert_eq!(model.rope.attention_scale(), 1.0);
}

/// Growing the context must rebuild through the *scaled* path. Before M-08's
/// wiring, `grow_context` called `RopeTable::new`, so a sequence that outgrew
/// its first allocation silently lost YaRN — in exactly the long-context
/// regime the scaling exists for.
#[test]
fn growing_the_context_keeps_the_scaling() {
    let mut config = small_config(0, 32, 4096);
    config.rope_scaling = yarn_scaling(1024);
    let mut model = BonsaiModel::new(config.clone());
    let before = model.rope.max_seq_len();
    model
        .ensure_context_capacity(before * 3)
        .expect("growth within the configured context");
    let rows = model.rope.max_seq_len();
    assert!(rows > before, "the table must actually have grown");

    let expected = RopeTable::new_with_scaling(
        config.head_dim,
        rows,
        config.rope_freq_base,
        Some(&yarn_strategy(1024)),
    )
    .expect("valid");
    assert_eq!(max_table_delta(&model.rope, &expected, rows), 0.0);
}

/// The magnitude half of M-08's gate: at the far end of an extended context a
/// YaRN table and an unscaled one are nowhere near each other. Uses the real
/// `Bonsai-8B` geometry (`head_dim 128`, `freq_base 1e6`, `factor 4.0`,
/// `original_context_length 16384`) and its last valid position, 65535.
#[test]
fn yarn_diverges_from_unscaled_rope_at_position_65535() {
    const HEAD_DIM: usize = 128;
    const ROWS: usize = 65_536;
    const BASE: f32 = 1_000_000.0;
    let strategy = RopeScalingStrategy::Yarn {
        original_max_position: 16_384,
        factor: 4.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attn_factor: None,
    };
    let scaled = RopeTable::new_with_scaling(HEAD_DIM, ROWS, BASE, Some(&strategy))
        .expect("Bonsai-8B's declared YaRN parameters are valid");
    let unscaled = RopeTable::new(HEAD_DIM, ROWS, BASE);

    let last = ROWS - 1;
    let delta = scaled
        .cos_at(last)
        .iter()
        .zip(unscaled.cos_at(last))
        .chain(scaled.sin_at(last).iter().zip(unscaled.sin_at(last)))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        delta > 1e-3,
        "YaRN must measurably change the rotation at position {last}, got {delta}"
    );
    // ... and at position 0 too: YaRN's mscale is unconditional, which is why
    // ignoring the declaration was wrong at *every* position, not only past
    // the original context.
    assert!(
        (scaled.cos_at(0)[0] - unscaled.cos_at(0)[0]).abs() > 1e-3,
        "the mscale must be visible at position 0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Synthetic fully-ternary fixture (M-02 / M-18)
// ═══════════════════════════════════════════════════════════════════════════

/// Fixture geometry: one `TQ2_0_g128` block per row, four heads.
const FIX_HIDDEN: usize = 128;
const FIX_INTER: usize = 256;
const FIX_NQ: usize = 4;
const FIX_NKV: usize = 2;
const FIX_HD: usize = 32;
const FIX_VOCAB: usize = 32;
const FIX_LAYERS: usize = 1;
const FIX_CTX: usize = 64;

/// `TQ2_0_g128` bytes emitting only the three legal ternary codes.
///
/// Blocks are 34 bytes: 32 bytes of 2-bit codes (four per byte, LSB-first)
/// then an f16 scale. The reserved `0b11` is `PQ2_0`'s `+2`, which the CQ-14
/// load screen rejects, so each lane is folded into `{0, 1, 2}`.
fn fixture_tq2(num_weights: usize, seed: u64) -> Vec<u8> {
    assert!(num_weights.is_multiple_of(128));
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
        data.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
    }
    data
}

/// Index-varying FP32 bytes, so nothing degenerates into a constant.
fn fixture_f32(n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (scale * (1.0_f32 + 0.25_f32 * ((i as f32) * 0.013_f32).sin())).to_le_bytes())
        .collect()
}

/// A synthetic GGUF whose `token_embd.weight` is **quantized** — the layout
/// every shipped Bonsai model uses, and the one M-02's resident-set claim is
/// about. `scaling` optionally plants a `qwen3.rope.scaling.*` declaration.
fn fixture_ternary_gguf(scaling: Option<(f32, u32)>) -> Vec<u8> {
    use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};
    use oxibonsai_core::MetadataWriteValue;

    let mut w = GgufWriter::new();
    for (k, v) in [
        ("general.architecture", "qwen3"),
        ("general.name", "Fix2ModelWiring"),
    ] {
        w.add_metadata(k, MetadataWriteValue::Str(v.to_string()));
    }
    for (k, v) in [
        ("qwen3.embedding_length", FIX_HIDDEN as u32),
        ("qwen3.block_count", FIX_LAYERS as u32),
        ("qwen3.attention.head_count", FIX_NQ as u32),
        ("qwen3.attention.head_count_kv", FIX_NKV as u32),
        ("qwen3.feed_forward_length", FIX_INTER as u32),
        ("qwen3.vocab_size", FIX_VOCAB as u32),
        ("qwen3.context_length", FIX_CTX as u32),
    ] {
        w.add_metadata(k, MetadataWriteValue::U32(v));
    }
    w.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    w.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));
    if let Some((factor, original)) = scaling {
        w.add_metadata(
            "qwen3.rope.scaling.type",
            MetadataWriteValue::Str("yarn".to_string()),
        );
        w.add_metadata("qwen3.rope.scaling.factor", MetadataWriteValue::F32(factor));
        w.add_metadata(
            "qwen3.rope.scaling.original_context_length",
            MetadataWriteValue::U32(original),
        );
    }

    // The whole point: the embedding is TQ2_0_g128, not FP32.
    w.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![FIX_HIDDEN as u64, FIX_VOCAB as u64],
        tensor_type: TensorType::TQ2_0_g128,
        data: fixture_tq2(FIX_VOCAB * FIX_HIDDEN, 0x5EED),
    });
    w.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![FIX_HIDDEN as u64],
        tensor_type: TensorType::F32,
        data: fixture_f32(FIX_HIDDEN, 1.0),
    });
    w.add_tensor(TensorEntry {
        name: "output.weight".to_string(),
        shape: vec![FIX_HIDDEN as u64, FIX_VOCAB as u64],
        tensor_type: TensorType::TQ2_0_g128,
        data: fixture_tq2(FIX_VOCAB * FIX_HIDDEN, 0xCAFE),
    });
    for layer in 0..FIX_LAYERS {
        let pfx = format!("blk.{layer}");
        for (name, len) in [
            (format!("{pfx}.attn_norm.weight"), FIX_HIDDEN),
            (format!("{pfx}.ffn_norm.weight"), FIX_HIDDEN),
            (format!("{pfx}.attn_q_norm.weight"), FIX_HD),
            (format!("{pfx}.attn_k_norm.weight"), FIX_HD),
        ] {
            w.add_tensor(TensorEntry {
                name,
                shape: vec![len as u64],
                tensor_type: TensorType::F32,
                data: fixture_f32(len, 1.0),
            });
        }
        for (name, rows, cols, bump) in [
            (
                format!("{pfx}.attn_q.weight"),
                FIX_HIDDEN,
                FIX_NQ * FIX_HD,
                0,
            ),
            (
                format!("{pfx}.attn_k.weight"),
                FIX_HIDDEN,
                FIX_NKV * FIX_HD,
                1,
            ),
            (
                format!("{pfx}.attn_v.weight"),
                FIX_HIDDEN,
                FIX_NKV * FIX_HD,
                2,
            ),
            (
                format!("{pfx}.attn_output.weight"),
                FIX_NQ * FIX_HD,
                FIX_HIDDEN,
                3,
            ),
            (format!("{pfx}.ffn_gate.weight"), FIX_HIDDEN, FIX_INTER, 4),
            (format!("{pfx}.ffn_up.weight"), FIX_HIDDEN, FIX_INTER, 5),
            (format!("{pfx}.ffn_down.weight"), FIX_INTER, FIX_HIDDEN, 6),
        ] {
            w.add_tensor(TensorEntry {
                name,
                shape: vec![rows as u64, cols as u64],
                tensor_type: TensorType::TQ2_0_g128,
                data: fixture_tq2(rows * cols, 0x1000 + (layer as u64) * 64 + bump),
            });
        }
    }
    w.to_bytes().expect("GgufWriter::to_bytes")
}

// ═══════════════════════════════════════════════════════════════════════════
// M-02 — the quantized embedding never materializes
// ═══════════════════════════════════════════════════════════════════════════

/// The acceptance measurement: a real load + prefill + decode cycle on a model
/// whose `token_embd.weight` is quantized must leave `resident_bytes()` at
/// **0** throughout. Before the wave-2.5 fix, the first multi-token prompt
/// tripped the dense `Index` hatch and materialized the whole FP32 table.
#[test]
fn a_quantized_embedding_costs_no_resident_bytes_across_prefill_and_decode() {
    use oxibonsai_core::gguf::reader::GgufFile;

    let bytes = fixture_ternary_gguf(None);
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let mut model = BonsaiModel::from_gguf(&gguf, FIX_CTX).expect("load fixture");
    assert_eq!(
        model.token_embd.resident_bytes(),
        0,
        "loading must keep the table quantized"
    );

    let kernel = cpu_kernel();
    let prompt: Vec<u32> = (0..6u32).map(|i| (i * 5 + 1) % FIX_VOCAB as u32).collect();
    model
        .forward_prefill(&prompt, 0, &kernel)
        .expect("batched prefill");
    assert_eq!(
        model.token_embd.resident_bytes(),
        0,
        "a multi-token prefill must not materialize the table (M-02)"
    );

    for step in 0..4usize {
        let pos = prompt.len() + step;
        model
            .forward((pos % FIX_VOCAB) as u32, pos, &kernel)
            .expect("decode step");
    }
    assert_eq!(
        model.token_embd.resident_bytes(),
        0,
        "decode must not materialize the table either"
    );
    // The measured delta M-02 exists to deliver: the same GGUF loaded with a
    // dense FP32 embedding handed in (the engine-pool seam) reports exactly
    // one dense copy of the table, and the quantized load reports none — with
    // the whole difference showing up in `footprint_bytes`.
    let dense: std::sync::Arc<[f32]> = load_f32_tensor(&gguf, "token_embd.weight")
        .expect("dequantize")
        .into();
    let dense_bytes = FIX_VOCAB * FIX_HIDDEN * std::mem::size_of::<f32>();
    assert_eq!(dense.len(), FIX_VOCAB * FIX_HIDDEN);
    let dense_model =
        BonsaiModel::from_gguf_with_embd(&gguf, FIX_CTX, dense).expect("load with a dense table");
    assert_eq!(dense_model.token_embd.resident_bytes(), dense_bytes);

    let quantized_fresh = BonsaiModel::from_gguf(&gguf, FIX_CTX).expect("reload");
    assert_eq!(quantized_fresh.token_embd.resident_bytes(), 0);
    assert_eq!(
        dense_model.footprint_bytes() - quantized_fresh.footprint_bytes(),
        dense_bytes,
        "the quantized load must be lighter by exactly one dense embedding table"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// M-08 — a GGUF-loaded model routes through the same config field
// ═══════════════════════════════════════════════════════════════════════════

/// A GGUF that declares YaRN must produce the scaled table, and the same one a
/// config-only model with the identical declaration gets.
#[test]
fn a_gguf_declared_yarn_reaches_the_rope_table() {
    use oxibonsai_core::gguf::reader::GgufFile;

    let scaled_bytes = fixture_ternary_gguf(Some((4.0, 16)));
    let scaled_gguf = GgufFile::parse(&scaled_bytes).expect("parse");
    let scaled = BonsaiModel::from_gguf(&scaled_gguf, FIX_CTX).expect("load scaled");
    assert_eq!(
        scaled.config().rope_scaling,
        yarn_scaling(16),
        "the loader must carry the declaration into the config"
    );

    let plain_bytes = fixture_ternary_gguf(None);
    let plain_gguf = GgufFile::parse(&plain_bytes).expect("parse");
    let plain = BonsaiModel::from_gguf(&plain_gguf, FIX_CTX).expect("load plain");

    let rows = scaled.rope.max_seq_len();
    assert_eq!(rows, plain.rope.max_seq_len());
    assert!(
        max_table_delta(&scaled.rope, &plain.rope, rows) > 1e-3,
        "the declared YaRN must actually change the table"
    );

    let expected = RopeTable::new_with_scaling(FIX_HD, rows, 10_000.0, Some(&yarn_strategy(16)))
        .expect("valid");
    assert_eq!(max_table_delta(&scaled.rope, &expected, rows), 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// M-18 — the chunked-prefill executor is actually driven
// ═══════════════════════════════════════════════════════════════════════════

/// Chunked and single-shot prefill must agree **numerically**. They are in
/// fact bit-identical whenever the chunk split leaves no one-token tail —
/// [`chunked_prefill_matches_the_single_shot_prefill_bit_for_bit`] asserts
/// exactly that — so this test covers the one case where they legitimately
/// differ: 13 tokens at chunk size 4, i.e. 4, 4, 4, **1**.
///
/// **Why a tolerance here (FIX3-PERF item 1, orchestrator decision D-4).**
/// The paragraph this one replaces blamed `for_each_register_block!`'s
/// tile-width selection — the claim being that one `m = 13` call and four
/// calls of `m = 4, 4, 4, 1` can land the same row in different-width
/// register tiles whose FMA reduction then reassociates. That explanation is
/// **wrong**, and it contradicted the kernel's own module doc
/// (`oxibonsai-kernels/src/gemm_ternary.rs`: "Batch rows never interact, so
/// blocking over `m` cannot perturb a row's reduction"). Re-measured:
/// `gemm_tq2_0_g128_blocked` at `m = 13` is bit-identical to the same call
/// split 4, 4, 4, 1, and both blocked CPU kernels are plain row-independent
/// loops, so `MR` cannot reassociate anything.
///
/// **The real cause is a code-path split.**
/// [`crate::model::types::prefill_cpu::CPU_PREFILL_MIN_TOKENS`] is 2, so
/// `forward_prefill_cpu` *declines* a one-token chunk and
/// `forward_prefill_unchunked` falls through to `forward_sequential` — the
/// per-token GEMV sweep — for that last chunk, while the single-shot run
/// produces the same position from the batched, register-blocked GEMM. Two
/// legitimately different (and individually correct) CPU code paths, hence a
/// few float32 ULPs of difference. Measured on this fixture, prompt @ chunk
/// size: **12 @ 4 bit-exact, 12 @ 3 bit-exact, 16 @ 8 bit-exact, 13 @ 4
/// differs by 6.68e-6 absolute, 13 @ 6 differs by 6.68e-6** — the divergence
/// appears if and only if the split leaves a one-token tail, which is
/// precisely the signature of the `CPU_PREFILL_MIN_TOKENS` decline and not
/// of anything inside the kernel.
#[test]
fn chunked_prefill_matches_the_single_shot_prefill_within_tolerance() {
    use oxibonsai_core::gguf::reader::GgufFile;

    let bytes = fixture_ternary_gguf(None);
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let kernel = cpu_kernel();
    let prompt: Vec<u32> = (0..13u32).map(|i| (i * 3 + 2) % FIX_VOCAB as u32).collect();

    let mut one_shot = BonsaiModel::from_gguf(&gguf, FIX_CTX).expect("load");
    one_shot.set_prefill_chunk_tokens(0);
    assert!(
        crate::chunked_prefill::prefill_chunk_plan(prompt.len(), 0).is_none(),
        "chunk size 0 must disable chunking"
    );
    let single = one_shot
        .forward_prefill(&prompt, 0, &kernel)
        .expect("single-shot prefill");

    let mut chunked = BonsaiModel::from_gguf(&gguf, FIX_CTX).expect("load");
    chunked.set_prefill_chunk_tokens(4);
    let plan = crate::chunked_prefill::prefill_chunk_plan(prompt.len(), 4)
        .expect("a 13-token prompt at chunk 4 must chunk");
    assert_eq!(
        crate::chunked_prefill::create_prefill_chunks(&prompt, &plan).len(),
        4,
        "13 tokens at chunk size 4 is 4 chunks"
    );
    let multi = chunked
        .forward_prefill(&prompt, 0, &kernel)
        .expect("chunked prefill");

    assert_eq!(
        single.len(),
        multi.len(),
        "chunking must not change the logit count"
    );
    let mut max_rel = 0.0f32;
    let mut max_rel_idx = 0usize;
    let scale = single
        .iter()
        .chain(multi.iter())
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(f32::MIN_POSITIVE);
    for (i, (a, b)) in single.iter().zip(multi.iter()).enumerate() {
        let rel = (a - b).abs() / scale;
        if rel > max_rel {
            max_rel = rel;
            max_rel_idx = i;
        }
    }
    assert!(
        max_rel <= 1e-4,
        "chunking changed logit {max_rel_idx} by {max_rel} of the vectors' own scale \
         (single={}, multi={}) — more than K-18's register-block reassociation should \
         ever cause",
        single[max_rel_idx],
        multi[max_rel_idx],
    );
    assert_eq!(
        one_shot.host_kv_valid_until(),
        chunked.host_kv_valid_until(),
        "both must leave the host KV cache covering the same prefix"
    );
}

/// The default threshold leaves every prompt the test suite uses on the
/// single-shot path, so the Metal fused batch path's dispatch granularity is
/// unchanged (M-18's carried caveat).
#[test]
fn the_default_chunk_threshold_is_a_no_op_for_this_test_suite() {
    let model = BonsaiModel::new(small_config(0, 32, 64));
    assert_eq!(
        model.prefill_chunk_tokens(),
        crate::chunked_prefill::DEFAULT_PREFILL_CHUNK_TOKENS
    );
    // 20 tokens is the longest prompt any test in the workspace hands to
    // `forward_prefill` (`cuda_synthetic_prefill_parity`).
    assert!(crate::chunked_prefill::prefill_chunk_plan(20, model.prefill_chunk_tokens()).is_none());
}

/// The no-tail half of the pair above: when the chunk split leaves no
/// one-token chunk, chunked and single-shot prefill are identical **bit for
/// bit**, not merely within a tolerance.
///
/// 12 tokens at chunk size 4 is three chunks of 4, every one of them at or
/// above [`crate::model::types::prefill_cpu::CPU_PREFILL_MIN_TOKENS`], so
/// every chunk runs the same batched register-blocked GEMM the single-shot
/// call runs and the split is invisible. This is the stronger guarantee that
/// the old (mis-explained) `..._bit_for_bit` test claimed to hold in general
/// and that D-4 rules must not be lost when the tail case is honestly
/// relaxed: 12 @ 3 and 16 @ 8 were measured bit-exact too.
#[test]
fn chunked_prefill_matches_the_single_shot_prefill_bit_for_bit() {
    use oxibonsai_core::gguf::reader::GgufFile;

    let bytes = fixture_ternary_gguf(None);
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let kernel = cpu_kernel();
    let prompt: Vec<u32> = (0..12u32).map(|i| (i * 3 + 2) % FIX_VOCAB as u32).collect();

    let mut one_shot = BonsaiModel::from_gguf(&gguf, FIX_CTX).expect("load");
    one_shot.set_prefill_chunk_tokens(0);
    let single = one_shot
        .forward_prefill(&prompt, 0, &kernel)
        .expect("single-shot prefill");

    let mut chunked = BonsaiModel::from_gguf(&gguf, FIX_CTX).expect("load");
    chunked.set_prefill_chunk_tokens(4);
    let plan = crate::chunked_prefill::prefill_chunk_plan(prompt.len(), 4)
        .expect("a 12-token prompt at chunk 4 must chunk");
    let chunks = crate::chunked_prefill::create_prefill_chunks(&prompt, &plan);
    assert_eq!(chunks.len(), 3, "12 tokens at chunk size 4 is 3 chunks");
    assert!(
        chunks
            .iter()
            .all(|c| c.tokens.len() >= crate::model::types::prefill_cpu::CPU_PREFILL_MIN_TOKENS),
        "this test's whole point is that no chunk is short enough to be declined"
    );
    let multi = chunked
        .forward_prefill(&prompt, 0, &kernel)
        .expect("chunked prefill");

    assert_eq!(
        single, multi,
        "a chunk split with no one-token tail must be bit-for-bit invisible"
    );
    assert_eq!(
        one_shot.host_kv_valid_until(),
        chunked.host_kv_valid_until(),
        "both must leave the host KV cache covering the same prefix"
    );
}

/// Largest absolute element-wise difference between two equal-length vectors.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Cosine similarity in f64, so the statistic itself adds no float noise.
fn cosine_f64(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Flatten every prefilled key and value of a model's host KV cache.
fn kv_snapshot(model: &BonsaiModel<'_>, seq_len: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for layer in 0..FIX_LAYERS {
        for head in 0..FIX_NKV {
            out.extend_from_slice(model.kv_cache().keys_for(layer, head, seq_len));
            out.extend_from_slice(model.kv_cache().values_for(layer, head, seq_len));
        }
    }
    out
}

/// Run `forward_prefill` on a freshly loaded fixture model with `tier`'s
/// dispatcher, returning the last position's logits and the whole KV cache.
fn prefill_on_tier(gguf_bytes: &[u8], tier: KernelTier, prompt: &[u32]) -> (Vec<f32>, Vec<f32>) {
    use oxibonsai_core::gguf::reader::GgufFile;
    let gguf = GgufFile::parse(gguf_bytes).expect("parse fixture GGUF");
    let kernel = KernelDispatcher::with_tier(tier);
    let mut model = BonsaiModel::from_gguf(&gguf, FIX_CTX).expect("load fixture");
    model.set_prefill_chunk_tokens(0);
    let logits = model
        .forward_prefill(prompt, 0, &kernel)
        .expect("prefill should succeed");
    let kv = kv_snapshot(&model, prompt.len());
    (logits, kv)
}

/// **K-18 / PARITY-RED-1: the `Reference` tier and the host's native CPU
/// tier must produce bit-identical `forward_prefill` results.**
///
/// Wave 3.5 measured this end to end on the real `Ternary-Bonsai-1.7B.gguf`
/// — Reference vs the auto-detected CPU tier (NEON on this M3), 64
/// self-generated greedy steps, all three prompt slots, batched prefill:
/// **exactly 0.0 at every step, identical token chains** (and the same for a
/// per-token prefill) — and asserted it nowhere. That measurement is what
/// cleared the batched CPU prefill of the legacy parity RED, which is a
/// Reference-vs-Metal numeric-bound shape problem and not a CPU one, so the invariant is worth a
/// standing test rather than a paragraph in a report.
///
/// # Exactly what this pins, and what it does not
///
/// It pins **routing**: the `kernel` argument callers pass to
/// `forward_prefill` must never re-route a ternary model's arithmetic. It
/// holds today for two structural reasons, both worth stating because they
/// are what a future change would break:
///
/// * `LinearTernary` (and `Linear1Bit`) store their own
///   `Arc<KernelDispatcher>`, chosen once at load time, so the per-call
///   `kernel` argument gates only the GPU fast paths
///   (`kernel.is_gpu_accelerated()`), never the CPU ternary GEMV/GEMM; and
/// * [`crate::model::types::prefill_cpu`]'s `prefill_dispatcher()` pins
///   `cpu_kernel_tier()` for the batched path, deliberately, so a GPU-tier
///   caller cannot drag device work behind a host-KV contract (MET-05).
///
/// It does **not** claim that the two tiers' kernels are bit-identical to
/// each other — they are not, and never were: measured, `Reference` against
/// `cpu_kernel_tier()` on the same ternary GEMM differs by up to 3.6e-7 at
/// `k = 128` and 1.05e-5 at `k = 1024`, for the register-blocked kernels and
/// the plain tier kernels alike and by the same amount, because a 4-lane
/// accumulator plus a horizontal sum is a different (equally valid)
/// summation order from a scalar sweep. That per-tier self-consistency is
/// pinned in `oxibonsai-kernels`
/// (`gemm_ternary::blocked_tests::every_tier_blocked_gemm_is_bit_identical_to_that_tiers_own_gemm`).
#[test]
fn forward_prefill_is_bit_exact_across_the_reference_and_native_cpu_tiers() {
    let cpu_tier = oxibonsai_kernels::cpu_kernel_tier();
    let bytes = fixture_ternary_gguf(None);
    // 12 tokens: one batched call, no one-token tail anywhere.
    let prompt: Vec<u32> = (0..12u32).map(|i| (i * 3 + 2) % FIX_VOCAB as u32).collect();

    let (ref_logits, ref_kv) = prefill_on_tier(&bytes, KernelTier::Reference, &prompt);
    let (cpu_logits, cpu_kv) = prefill_on_tier(&bytes, cpu_tier, &prompt);

    assert_eq!(
        ref_logits.len(),
        cpu_logits.len(),
        "both tiers must produce a full logit vector"
    );
    assert_eq!(
        ref_logits,
        cpu_logits,
        "forward_prefill diverged between KernelTier::Reference and {cpu_tier:?} \
         (max|delta| = {})",
        max_abs_diff(&ref_logits, &cpu_logits)
    );
    assert_eq!(
        ref_kv,
        cpu_kv,
        "the prefilled KV cache diverged between KernelTier::Reference and {cpu_tier:?} \
         (max|delta| = {})",
        max_abs_diff(&ref_kv, &cpu_kv)
    );

    // Sensitivity control: the comparison above must be capable of failing.
    // A one-token change in the prompt has to move both vectors, otherwise
    // this test would pass on any two runs whatsoever.
    let mut other = prompt.clone();
    other[0] = (other[0] + 1) % FIX_VOCAB as u32;
    let (other_logits, other_kv) = prefill_on_tier(&bytes, KernelTier::Reference, &other);
    assert_ne!(
        ref_logits, other_logits,
        "sensitivity control: changing a prompt token must change the logits"
    );
    assert_ne!(
        ref_kv, other_kv,
        "sensitivity control: changing a prompt token must change the KV cache"
    );
}

/// **The measured cost of batching, pinned.** Within one tier, the batched
/// register-blocked prefill and the per-token GEMV sweep agree to a few
/// float32 ULPs — they are *not* bit-identical, and this is the assertion
/// that says by how much.
///
/// Wave 3.5's measurement on the real `Ternary-Bonsai-1.7B.gguf`, 64
/// self-generated greedy steps, both CPU tiers, all three prompt slots:
/// worst |delta| **9.06e-6** (slot 0, step 45) and **8.58e-6** (slot 1, step
/// 0), with **identical token chains** throughout and the same series to
/// every printed digit on both tiers. On this synthetic fixture the same
/// comparison measures **2.68e-6**. The ceilings below — cos >= 0.9999 and
/// max |delta| <= 1e-4 — sit an order of magnitude above the worst figure
/// ever observed, so they catch a real regression without flagging the noise
/// the two code paths are entitled to.
#[test]
fn batched_prefill_agrees_with_the_per_token_prefill_within_a_tier() {
    use oxibonsai_core::gguf::reader::GgufFile;

    let bytes = fixture_ternary_gguf(None);
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let prompt: Vec<u32> = (0..13u32).map(|i| (i * 3 + 2) % FIX_VOCAB as u32).collect();

    for tier in [KernelTier::Reference, oxibonsai_kernels::cpu_kernel_tier()] {
        let kernel = KernelDispatcher::with_tier(tier);

        let mut batched = BonsaiModel::from_gguf(&gguf, FIX_CTX).expect("load");
        batched.set_prefill_chunk_tokens(0);
        let batched_logits = batched
            .forward_prefill(&prompt, 0, &kernel)
            .expect("batched prefill");

        let mut per_token = BonsaiModel::from_gguf(&gguf, FIX_CTX).expect("load");
        let mut seq_logits = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            seq_logits = per_token
                .forward(tok, pos, &kernel)
                .expect("per-token forward");
        }

        assert_eq!(batched_logits.len(), seq_logits.len());
        let cos = cosine_f64(&batched_logits, &seq_logits);
        assert!(
            cos >= 0.9999,
            "{tier:?}: batched prefill diverged from the per-token sweep: cos={cos}"
        );
        let delta = max_abs_diff(&batched_logits, &seq_logits);
        assert!(
            delta <= 1e-4,
            "{tier:?}: batched prefill differs from the per-token sweep by {delta:e}, \
             more than the 1e-4 ceiling (worst ever measured: 9.06e-6 on the real 1.7B model)"
        );
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &x)| {
                    if x > acc.1 {
                        (i, x)
                    } else {
                        acc
                    }
                })
                .0
        };
        assert_eq!(
            argmax(&batched_logits),
            argmax(&seq_logits),
            "{tier:?}: the greedy token must not change between the batched and \
             per-token prefills"
        );
    }
}
