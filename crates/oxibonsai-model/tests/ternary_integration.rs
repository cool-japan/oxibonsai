//! End-to-end integration tests for the ternary (TQ2_0_g128) code paths.
//!
//! # What is tested
//!
//! 1. `ternary_gguf_loads_and_runs_forward` — Builds a synthetic GGUF with
//!    Q1_0_g128 attention/FFN blocks and a TQ2_0_g128 output projection,
//!    loads it through `BonsaiModel::from_gguf`, and runs one forward pass,
//!    asserting all output logits are finite `f32` values.
//!
//! 2. `ternary_export_round_trip` — Quantizes a synthetic f32 tensor via
//!    `ExportFormat::TernaryG128`, writes it to `temp_dir()`, reloads it
//!    with `GgufFile::parse`, dequantizes, and asserts MSE < 0.15 vs the
//!    original.
//!
//! 3. `ternary_variant_detection_from_gguf` — Constructs a minimal GGUF
//!    with a single TQ2_0_g128 tensor, parses it, calls
//!    `TensorStore::count_by_type()` to find the dominant type, and asserts
//!    that `ModelVariant::from_config_and_sample_tensor_type` returns the
//!    expected `TernaryBonsai*` variants for all three model sizes.
//!
//! 4. `linear_ternary_gemm_batch_forward` — Tests the batched GEMM path
//!    (`LinearTernary::forward_batch`) with three rows carrying distinct
//!    ternary patterns (+1 only, -1 only, zero only) and a two-token batch.

use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_core::{BlockTQ2_0_g128, GgufTensorType, Qwen3Config, QK_TQ2_0_G128};
use oxibonsai_kernels::KernelDispatcher;
use oxibonsai_model::export::{export_to_gguf, ExportConfig, ExportFormat, WeightTensor};
use oxibonsai_model::layers::linear::LinearTernary;
use oxibonsai_model::ModelVariant;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a byte buffer for a Q1_0_g128 weight matrix.
///
/// The block layout is 18 bytes each: 2-byte f16 scale + 16 bytes sign bits.
/// We fill every block with a uniform pattern: all sign bits set (→ +scale).
fn q1_0_g128_data(num_weights: usize) -> Vec<u8> {
    assert_eq!(num_weights % 128, 0, "num_weights must be multiple of 128");
    let num_blocks = num_weights / 128;
    let mut data = Vec::with_capacity(num_blocks * 18);
    // f16::ONE = 0x3C00 in little-endian
    let scale_bytes = f16::ONE.to_le_bytes();
    for _ in 0..num_blocks {
        data.extend_from_slice(&scale_bytes); // scale
        data.extend_from_slice(&[0xFFu8; 16]); // 128 sign bits — all +1
    }
    data
}

/// Build a byte buffer for a TQ2_0_g128 weight matrix.
///
/// Each block is 34 bytes: 32-byte qs + 2-byte f16 scale (PrismML layout).
/// `0xAA` = `0b10101010` → every 2-bit lane is `0b10` → code +1.
fn tq2_0_g128_data(num_weights: usize) -> Vec<u8> {
    assert_eq!(num_weights % 128, 0, "num_weights must be multiple of 128");
    let num_blocks = num_weights / 128;
    let mut data = Vec::with_capacity(num_blocks * 34);
    let scale_bytes = f16::ONE.to_le_bytes();
    for _ in 0..num_blocks {
        data.extend_from_slice(&[0xAAu8; 32]); // qs: all +1 codes
        data.extend_from_slice(&scale_bytes); // scale
    }
    data
}

/// Build a synthetic GGUF for `BonsaiModel::from_gguf`.
///
/// Config dimensions chosen so all layer shapes are valid multiples of 128:
/// - hidden_size = 128  (must be ≥ 128 for TQ2_0_g128 output projection)
/// - intermediate = 256
/// - num_layers = 2
/// - nq = 4, nkv = 2, head_dim = 32 (128/4)
/// - vocab = 32  (deliberately tiny to keep test data small)
///
/// Tensor types:
/// - token_embd.weight    → F32
/// - output_norm.weight   → F32
/// - output.weight        → TQ2_0_g128 (exercises the ternary output path)
/// - blk.N.attn_norm / ffn_norm / attn_q_norm / attn_k_norm → F32
/// - blk.N.attn_q / k / v / output / ffn_gate / ffn_up / ffn_down → Q1_0_g128
fn build_tiny_ternary_gguf() -> Vec<u8> {
    let h: usize = 128; // hidden_size — must be multiple of 128 for TQ2_0_g128
    let inter: usize = 256; // intermediate_size
    let num_layers: usize = 2;
    let nq: usize = 4; // num_attention_heads
    let nkv: usize = 2; // num_kv_heads
    let hd: usize = 32; // head_dim = h / nq = 128 / 4
    let vocab: usize = 32; // deliberately tiny

    let mut writer = GgufWriter::new();

    // ── Metadata keys read by Qwen3Config::from_metadata ──────────────────
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("TinyTernaryTest".to_string()),
    );
    writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(h as u32));
    writer.add_metadata(
        "qwen3.block_count",
        MetadataWriteValue::U32(num_layers as u32),
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

    // ── Helper: write an F32 tensor of `n` elements (all 1.0) ─────────────
    let f32_ones = |n: usize| -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.extend_from_slice(&1.0_f32.to_le_bytes());
        }
        v
    };

    // ── Token embedding: vocab × hidden (F32) ─────────────────────────────
    // Shape in GGUF convention: [hidden, vocab] (innermost dim first)
    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::F32,
        data: f32_ones(vocab * h),
    });

    // ── Output norm: hidden (F32) ──────────────────────────────────────────
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![h as u64],
        tensor_type: TensorType::F32,
        data: f32_ones(h),
    });

    // ── Output projection: hidden × vocab (TQ2_0_g128) ────────────────────
    // Shape [hidden, vocab] in GGUF: [in_features, out_features]
    // The weight_loader reads shape[0] as in_features, shape[1] as out_features.
    writer.add_tensor(TensorEntry {
        name: "output.weight".to_string(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::TQ2_0_g128,
        data: tq2_0_g128_data(vocab * h),
    });

    // ── Per-layer tensors ──────────────────────────────────────────────────
    for layer in 0..num_layers {
        let pfx = format!("blk.{layer}");

        // RMSNorm weights (F32)
        for suffix in &["attn_norm.weight", "ffn_norm.weight"] {
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.{suffix}"),
                shape: vec![h as u64],
                tensor_type: TensorType::F32,
                data: f32_ones(h),
            });
        }
        for suffix in &["attn_q_norm.weight", "attn_k_norm.weight"] {
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.{suffix}"),
                shape: vec![hd as u64],
                tensor_type: TensorType::F32,
                data: f32_ones(hd),
            });
        }

        // Attention projections (Q1_0_g128)
        // attn_q: out = nq*hd, in = h
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_q.weight"),
            shape: vec![h as u64, (nq * hd) as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_data(nq * hd * h),
        });
        // attn_k: out = nkv*hd, in = h
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_k.weight"),
            shape: vec![h as u64, (nkv * hd) as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_data(nkv * hd * h),
        });
        // attn_v: same as k
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_v.weight"),
            shape: vec![h as u64, (nkv * hd) as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_data(nkv * hd * h),
        });
        // attn_output: out = h, in = nq*hd
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_output.weight"),
            shape: vec![(nq * hd) as u64, h as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_data(h * nq * hd),
        });
        // ffn_gate: out = inter, in = h
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_gate.weight"),
            shape: vec![h as u64, inter as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_data(inter * h),
        });
        // ffn_up: out = inter, in = h
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_up.weight"),
            shape: vec![h as u64, inter as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_data(inter * h),
        });
        // ffn_down: out = h, in = inter
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_down.weight"),
            shape: vec![inter as u64, h as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_data(h * inter),
        });
    }

    writer.to_bytes().expect("GgufWriter::to_bytes")
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: load synthetic ternary GGUF and run a forward pass
// ─────────────────────────────────────────────────────────────────────────────

/// Build a synthetic GGUF with a ternary output projection, load via
/// `BonsaiModel::from_gguf`, and verify that one forward pass produces
/// finite logits.
///
/// The output.weight tensor uses `TQ2_0_g128`, exercising the ternary output
/// projection code path in `weight_loaders::load_output_weight`.
#[test]
fn ternary_gguf_loads_and_runs_forward() {
    use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
    use oxibonsai_model::model::BonsaiModel;

    let bytes = build_tiny_ternary_gguf();
    let gguf = GgufFile::parse(&bytes).expect("GgufFile::parse on synthetic GGUF");

    // Use a small max_seq_len matching the synthetic config's context_length.
    let max_seq_len = 512;
    let mut model = BonsaiModel::from_gguf(&gguf, max_seq_len).expect("BonsaiModel::from_gguf");

    // Sanity-check: config should have been parsed from our metadata.
    assert_eq!(model.config().hidden_size, 128);
    assert_eq!(model.config().num_layers, 2);
    assert_eq!(model.config().num_attention_heads, 4);

    // Run one forward pass with token_id=0 at position 0.
    let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
    let logits = model.forward(0, 0, &kernel).expect("forward pass");

    // Vocab size was overridden from tensor shape: output.weight has out_features=32.
    assert_eq!(logits.len(), 32, "logit count should match tensor vocab=32");

    // Every logit must be a finite f32 value.
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1b: shared token-embedding Arc across replicas
// ─────────────────────────────────────────────────────────────────────────────

/// Prove that the engine-pool sharing seam works at the model level: building
/// several [`BonsaiModel`]s via [`BonsaiModel::from_gguf_with_embd`] from one
/// `Arc<[f32]>` token-embedding table yields models that all point at the
/// *same* allocation (ptr-equal), each with its own independent KV cache, and
/// that the shared `Arc`'s strong count equals the number of live replicas.
///
/// This is the whole point of Part A/B: collapse N duplicate ~1.16 GiB
/// embedding tables (for the 1.7B) into one shared allocation.
#[test]
fn shared_token_embd_arc_is_one_allocation_across_replicas() {
    use oxibonsai_model::model::BonsaiModel;
    use std::sync::Arc;

    let bytes = build_tiny_ternary_gguf();
    let gguf = GgufFile::parse(&bytes).expect("GgufFile::parse on synthetic GGUF");
    let max_seq_len = 512;

    // Load the token-embedding table ONCE via the canonical `from_gguf` path,
    // then extract its shared handle. This mirrors what the pool builder does
    // with replica #1.
    let replica1 = BonsaiModel::from_gguf(&gguf, max_seq_len).expect("replica #1 from_gguf");
    let shared = replica1.shared_token_embd();

    // Build two more replicas that REUSE the same Arc, exactly as the pool
    // builder does for replicas 2..N via `from_gguf_static_with_embd`.
    let replica2 = BonsaiModel::from_gguf_with_embd(&gguf, max_seq_len, Arc::clone(&shared))
        .expect("replica #2 from_gguf_with_embd");
    let replica3 = BonsaiModel::from_gguf_with_embd(&gguf, max_seq_len, Arc::clone(&shared))
        .expect("replica #3 from_gguf_with_embd");

    let e1 = replica1.shared_token_embd();
    let e2 = replica2.shared_token_embd();
    let e3 = replica3.shared_token_embd();

    // The whole point: all replicas share ONE token_embd allocation.
    assert!(
        Arc::ptr_eq(&e1, &e2),
        "replica #1 and #2 token_embd must be the SAME allocation"
    );
    assert!(
        Arc::ptr_eq(&e2, &e3),
        "replica #2 and #3 token_embd must be the SAME allocation"
    );

    // Same values, obviously (it is literally the same memory).
    assert_eq!(e1.len(), e2.len());
    assert_eq!(&e1[..], &e2[..]);

    // Each replica must own a DISTINCT, independent KV cache (the per-request
    // mutable state that must NOT be shared).
    let kv1 = replica1.kv_cache() as *const _;
    let kv2 = replica2.kv_cache() as *const _;
    let kv3 = replica3.kv_cache() as *const _;
    assert_ne!(kv1, kv2, "replicas #1 and #2 must have distinct KV caches");
    assert_ne!(kv2, kv3, "replicas #2 and #3 must have distinct KV caches");
    assert_ne!(kv1, kv3, "replicas #1 and #3 must have distinct KV caches");

    // Strong count: 3 replica-held clones + `shared` + the 3 freshly-pulled
    // handles (e1/e2/e3) all alias one allocation.
    drop(e1);
    drop(e2);
    drop(e3);
    // After dropping the temporary handles, exactly the 3 replicas + `shared`
    // hold the Arc.
    assert_eq!(
        Arc::strong_count(&shared),
        4,
        "expected 3 replicas + the shared handle to alias one allocation"
    );

    // Dropping a replica decrements the count, proving they really share it.
    drop(replica3);
    assert_eq!(
        Arc::strong_count(&shared),
        3,
        "dropping a replica must release one reference to the shared embd"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: ternary export round-trip through temp_dir()
// ─────────────────────────────────────────────────────────────────────────────

/// Export a 256-element f32 tensor with `ExportFormat::TernaryG128`, write
/// the GGUF bytes to a file in `temp_dir()`, reload via `GgufFile::parse`,
/// dequantize the tensor, and assert MSE < 0.15 vs the original weights.
///
/// This test exercises the full `export_to_gguf` → file I/O → parse → dequant
/// pipeline for the ternary format, going beyond the in-memory unit tests in
/// `quantize_ternary.rs`.
#[test]
fn ternary_export_round_trip() {
    // 256 weights (2 blocks): alternating pattern {+1, -1, 0, …}.
    let weights: Vec<f32> = (0..256_usize)
        .map(|i| match i % 3 {
            0 => 1.0_f32,
            1 => -1.0_f32,
            _ => 0.0_f32,
        })
        .collect();

    // Export to GGUF bytes via the production pipeline.
    let tensor = WeightTensor::new("test.weight", weights.clone(), vec![256]);
    let config = ExportConfig::new(ExportFormat::TernaryG128, "round-trip-test");
    let gguf_bytes = export_to_gguf(&[tensor], &config, &[]).expect("export_to_gguf");
    assert!(!gguf_bytes.is_empty(), "exported GGUF should not be empty");

    // Write to a temp file.
    let temp_path = {
        let mut p = std::env::temp_dir();
        p.push("oxibonsai_ternary_round_trip_test.gguf");
        p
    };
    std::fs::write(&temp_path, &gguf_bytes).expect("write temp GGUF file");

    // Reload from disk.
    let reloaded = std::fs::read(&temp_path).expect("read temp GGUF file");
    let gguf = GgufFile::parse(&reloaded).expect("GgufFile::parse reloaded");

    // Clean up the temp file regardless of test outcome.
    let _ = std::fs::remove_file(&temp_path);

    // Locate the tensor in the parsed file.
    let info = gguf
        .tensors
        .get("test.weight")
        .expect("test.weight tensor should be present");
    assert_eq!(
        info.tensor_type,
        GgufTensorType::TQ2_0_g128,
        "exported tensor should have TQ2_0_g128 type"
    );

    // Dequantize the raw bytes back to f32.
    let raw = gguf
        .tensor_data("test.weight")
        .expect("tensor data should be accessible");
    let blocks = BlockTQ2_0_g128::slice_from_bytes(raw).expect("slice_from_bytes on tensor data");
    let mut decoded = vec![0.0_f32; blocks.len() * QK_TQ2_0_G128];
    BlockTQ2_0_g128::dequant(blocks, &mut decoded).expect("dequant");

    // MSE over the original 256 elements (decoded may be longer due to padding).
    let mse: f32 = weights
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / weights.len() as f32;
    assert!(
        mse < 0.15,
        "ternary round-trip MSE={mse:.4} exceeds 0.15 threshold"
    );

    // Sign preservation: large-magnitude weights (|w| > 0.5) must have the
    // same sign after dequantization.
    for (orig, dec) in weights.iter().zip(decoded.iter()) {
        if orig.abs() > 0.5 {
            assert_eq!(
                orig.signum() as i32,
                dec.signum() as i32,
                "sign flip: orig={orig:.2}, decoded={dec:.2}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: variant detection from a parsed GGUF's dominant tensor type
// ─────────────────────────────────────────────────────────────────────────────

/// Build a minimal GGUF with one TQ2_0_g128 tensor, parse it, call
/// `TensorStore::count_by_type()` to determine the dominant type, and
/// verify that `ModelVariant::from_config_and_sample_tensor_type` returns
/// the correct `TernaryBonsai*` variant for all three model sizes.
///
/// Also checks that Q1_0_g128 (1-bit) tensors do NOT trigger ternary detection,
/// and that a Custom config remains Custom even with ternary tensors.
#[test]
fn ternary_variant_detection_from_gguf() {
    // ── Step 1: build and parse a minimal GGUF with one TQ2_0_g128 tensor ──
    let mut writer = GgufWriter::new();
    writer.add_tensor(TensorEntry {
        name: "blk.0.attn_q.weight".to_string(),
        shape: vec![128_u64, 128_u64],
        tensor_type: TensorType::TQ2_0_g128,
        data: tq2_0_g128_data(128 * 128),
    });
    let gguf_bytes = writer.to_bytes().expect("GgufWriter::to_bytes");
    let gguf = GgufFile::parse(&gguf_bytes).expect("GgufFile::parse");

    // ── Step 2: count by type and find the dominant type ───────────────────
    let counts = gguf.tensors.count_by_type();
    assert_eq!(counts.len(), 1, "only one tensor type should be present");
    let dominant_type = counts
        .into_iter()
        .max_by_key(|(_ty, count)| *count)
        .map(|(ty, _)| ty)
        .expect("at least one tensor type");
    assert_eq!(
        dominant_type,
        GgufTensorType::TQ2_0_g128,
        "dominant type from parsed GGUF should be TQ2_0_g128"
    );
    assert!(
        dominant_type.is_ternary(),
        "TQ2_0_g128 should report is_ternary() = true"
    );

    // ── Step 3: variant detection for all three model sizes ────────────────

    // 8B: 36 layers, hidden=4096
    let variant_8b =
        ModelVariant::from_config_and_sample_tensor_type(&Qwen3Config::bonsai_8b(), dominant_type);
    assert_eq!(
        variant_8b,
        ModelVariant::TernaryBonsai8B,
        "8B architecture + TQ2_0_g128 → TernaryBonsai8B"
    );

    // 4B: 24 layers, hidden=2560
    let variant_4b =
        ModelVariant::from_config_and_sample_tensor_type(&Qwen3Config::bonsai_4b(), dominant_type);
    assert_eq!(
        variant_4b,
        ModelVariant::TernaryBonsai4B,
        "4B architecture + TQ2_0_g128 → TernaryBonsai4B"
    );

    // 1.7B: 16 layers, hidden=1536
    let variant_1_7b = ModelVariant::from_config_and_sample_tensor_type(
        &Qwen3Config::bonsai_1_7b(),
        dominant_type,
    );
    assert_eq!(
        variant_1_7b,
        ModelVariant::TernaryBonsai1_7B,
        "1.7B architecture + TQ2_0_g128 → TernaryBonsai1_7B"
    );

    // ── Step 4: 1-bit type must NOT trigger ternary upgrade ───────────────
    let onebit_type = GgufTensorType::Q1_0_g128;
    assert!(!onebit_type.is_ternary(), "Q1_0_g128 should not be ternary");
    assert_eq!(
        ModelVariant::from_config_and_sample_tensor_type(&Qwen3Config::bonsai_8b(), onebit_type),
        ModelVariant::Bonsai8B,
        "8B architecture + Q1_0_g128 → Bonsai8B (not ternary)"
    );

    // ── Step 5: Custom config stays Custom even with ternary type ─────────
    let mut custom_cfg = Qwen3Config::bonsai_8b();
    custom_cfg.num_layers = 99;
    custom_cfg.hidden_size = 9999;
    let custom_variant =
        ModelVariant::from_config_and_sample_tensor_type(&custom_cfg, dominant_type);
    assert_eq!(
        custom_variant,
        ModelVariant::Custom,
        "unrecognized architecture + TQ2_0_g128 → Custom"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 (bonus): LinearTernary batched GEMM path with mixed sign rows
// ─────────────────────────────────────────────────────────────────────────────

/// Test `LinearTernary::forward_batch` (GEMM path) with three weight rows
/// that have distinct ternary patterns: all +1, all -1, and all zero.
///
/// A batch of two identical input vectors is processed; both outputs must
/// match the expected dot products, demonstrating that the GEMM path
/// produces correct results across the full batch dimension.
///
/// This is distinct from the existing `linear_ternary_forward_all_pos` unit
/// test (which uses GEMV / single-row / single-vector) by exercising:
/// - Multiple output rows with different signs
/// - Batch dimension > 1
#[test]
fn linear_ternary_gemm_batch_forward() {
    // Weight row patterns:
    //   row 0: all +1 → qs = 0xAA (every 2-bit lane = 0b10 = +1)
    //   row 1: all -1 → qs = 0x00 (every 2-bit lane = 0b00 = -1)
    //   row 2: all  0 → qs = 0x55 (every 2-bit lane = 0b01 =  0)
    //
    // Each row needs exactly 1 TQ2_0_g128 block (128 weights).
    let block_pos = BlockTQ2_0_g128 {
        qs: [0xAAu8; 32], // all +1
        d: f16::ONE,
    };
    let block_neg = BlockTQ2_0_g128 {
        qs: [0x00u8; 32], // all -1
        d: f16::ONE,
    };
    let block_zero = BlockTQ2_0_g128 {
        qs: [0x55u8; 32], // all  0
        d: f16::ONE,
    };

    // 3 output rows × 128 input features → 3 blocks (one per row)
    let blocks = [block_pos, block_neg, block_zero];
    let out_features: usize = 3;
    let in_features: usize = 128; // one block per row

    let kernel = Arc::new(KernelDispatcher::auto_detect());
    let layer =
        LinearTernary::new(&blocks, out_features, in_features, kernel).expect("LinearTernary::new");

    // Input: batch of 2 identical vectors, all 1.0
    let batch: usize = 2;
    let input = vec![1.0_f32; batch * in_features];
    let mut output = vec![0.0_f32; batch * out_features];
    layer
        .forward_batch(&input, &mut output, batch)
        .expect("forward_batch");

    // Expected per row per batch item:
    //   row 0: 128 × (+1) × 1.0 × scale(1.0) = 128.0
    //   row 1: 128 × (-1) × 1.0 × scale(1.0) = -128.0
    //   row 2: 128 × (0)  × 1.0 × scale(1.0) = 0.0
    let tolerance = 1.0_f32;
    for b in 0..batch {
        let base = b * out_features;
        let row0 = output[base];
        let row1 = output[base + 1];
        let row2 = output[base + 2];

        assert!(
            (row0 - 128.0).abs() < tolerance,
            "batch={b} row=0: expected ~128.0, got {row0}"
        );
        assert!(
            (row1 + 128.0).abs() < tolerance,
            "batch={b} row=1: expected ~-128.0, got {row1}"
        );
        assert!(
            row2.abs() < tolerance,
            "batch={b} row=2: expected ~0.0, got {row2}"
        );
    }
}
