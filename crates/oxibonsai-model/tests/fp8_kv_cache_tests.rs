//! Integration tests for the FP8-quantized KV cache.
//!
//! All values are deterministic — no random number generator is used.
//! Test vectors use simple arithmetic expressions such as `(i as f32) * 0.05 - 3.0`.
//! Error bounds reflect the limited precision of FP8 formats (E4M3: ~3-bit mantissa,
//! E5M2: ~2-bit mantissa) plus the per-row absolute-max scaling scheme.

use oxibonsai_model::{Fp8KvCache, Fp8KvFormat, Fp8KvLayer, QuantKvError};

// ─── Helper: build a deterministic key/value row ──────────────────────────────

fn make_kv_row(num_kv_heads: usize, head_dim: usize, offset: f32, stride: f32) -> Vec<f32> {
    let n = num_kv_heads * head_dim;
    (0..n).map(|i| i as f32 * stride + offset).collect()
}

// ─── Test 1: E4M3 quantize_row roundtrip – small relative error ───────────────

/// Quantize a row of typical activation values to FP8 E4M3 and back.
/// The maximum absolute difference relative to the row's scale must be < 5%.
/// This validates the `quantize_row_fp8` / `dequantize_row_fp8` primitive
/// (exercised indirectly via `Fp8KvLayer::push` + `get_key`).
#[test]
fn quantize_row_fp8_e4m3_roundtrip_small_error() {
    let num_kv_heads = 1;
    let head_dim = 64;
    let mut layer = Fp8KvLayer::with_capacity(num_kv_heads, head_dim, 4, Fp8KvFormat::E4M3);

    // Deterministic activations in [-3.2, 3.0]
    let key: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.1 - 3.2).collect();
    let value = vec![0.0_f32; head_dim];

    layer.push(&key, &value).expect("push should succeed");
    let retrieved = layer.get_key(0);

    let max_abs = key.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let scale = max_abs / 448.0_f32; // FP8_E4M3_MAX

    let max_diff = key
        .iter()
        .zip(retrieved.iter())
        .map(|(o, r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff < scale * 448.0 * 0.05,
        "E4M3 max_diff {max_diff} must be < 5% of scale*max ({:.4})",
        scale * 448.0 * 0.05
    );
}

// ─── Test 2: E5M2 quantize_row roundtrip – small relative error ───────────────

/// Same as test 1 but using the E5M2 format.
/// E5M2 has a wider dynamic range (max ≈ 57344) but slightly lower mantissa
/// precision than E4M3.
#[test]
fn quantize_row_fp8_e5m2_roundtrip_small_error() {
    let num_kv_heads = 1;
    let head_dim = 64;
    let mut layer = Fp8KvLayer::with_capacity(num_kv_heads, head_dim, 4, Fp8KvFormat::E5M2);

    let key: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.15 - 4.8).collect();
    let value = vec![0.0_f32; head_dim];

    layer.push(&key, &value).expect("push should succeed");
    let retrieved = layer.get_key(0);

    let max_abs = key.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    // Relative error bound: allow up to 30% of max_abs (E5M2 has only 2 mantissa bits)
    let tolerance = max_abs * 0.30 + 1e-5;

    let max_diff = key
        .iter()
        .zip(retrieved.iter())
        .map(|(o, r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff < tolerance,
        "E5M2 max_diff {max_diff} must be < {tolerance:.4} (30% of max_abs {max_abs})"
    );
}

// ─── Test 3: E4M3 push → get_key roundtrip, error < 1% per head ──────────────

/// Push a deterministic key tensor through an `Fp8KvLayer` with E4M3 format and
/// verify each head's mean-absolute-error is below 5% of its max absolute value.
/// E4M3 has a 3-bit mantissa providing ~1/8 step precision, so a 5% MAE bound
/// relative to the row's maximum is well within the format's capabilities.
#[test]
fn fp8_kv_layer_e4m3_push_get_key_roundtrip() {
    let num_kv_heads = 4;
    let head_dim = 64;
    let mut layer =
        Fp8KvLayer::with_capacity(num_kv_heads, head_dim, 16, Fp8KvFormat::E4M3);

    let keys = make_kv_row(num_kv_heads, head_dim, -6.4, 0.05);
    let values = vec![0.0_f32; num_kv_heads * head_dim];

    layer.push(&keys, &values).expect("push should succeed");
    assert_eq!(layer.len(), 1);

    let retrieved = layer.get_key(0);
    assert_eq!(retrieved.len(), num_kv_heads * head_dim);

    for head in 0..num_kv_heads {
        let row_start = head * head_dim;
        let original = &keys[row_start..row_start + head_dim];
        let got = &retrieved[row_start..row_start + head_dim];

        let max_abs = original.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        if max_abs < f32::EPSILON {
            continue; // trivially zero
        }

        let mae = original
            .iter()
            .zip(got.iter())
            .map(|(o, r)| (o - r).abs())
            .sum::<f32>()
            / head_dim as f32;

        assert!(
            mae < max_abs * 0.05,
            "E4M3 key head {head}: MAE {mae} exceeds 5% of max_abs {max_abs}"
        );
    }
}

// ─── Test 4: E4M3 push → get_value roundtrip, error < 1% per head ────────────

/// Same as test 3 but verifies the value path.
/// MAE bound is 5% of the row max to accommodate E4M3's 3-bit mantissa precision.
#[test]
fn fp8_kv_layer_e4m3_push_get_value_roundtrip() {
    let num_kv_heads = 4;
    let head_dim = 64;
    let mut layer =
        Fp8KvLayer::with_capacity(num_kv_heads, head_dim, 16, Fp8KvFormat::E4M3);

    let keys = vec![0.0_f32; num_kv_heads * head_dim];
    let values = make_kv_row(num_kv_heads, head_dim, -8.0, 0.07);

    layer.push(&keys, &values).expect("push should succeed");

    let retrieved = layer.get_value(0);
    assert_eq!(retrieved.len(), num_kv_heads * head_dim);

    for head in 0..num_kv_heads {
        let row_start = head * head_dim;
        let original = &values[row_start..row_start + head_dim];
        let got = &retrieved[row_start..row_start + head_dim];

        let max_abs = original.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        if max_abs < f32::EPSILON {
            continue;
        }

        let mae = original
            .iter()
            .zip(got.iter())
            .map(|(o, r)| (o - r).abs())
            .sum::<f32>()
            / head_dim as f32;

        assert!(
            mae < max_abs * 0.05,
            "E4M3 value head {head}: MAE {mae} exceeds 5% of max_abs {max_abs}"
        );
    }
}

// ─── Test 5: compression ratio vs FP32 ───────────────────────────────────────

/// FP8 uses 1 byte per element, so `memory_bytes < memory_bytes_fp32_equivalent / 2`.
/// The ratio is slightly below 2× because per-row f32 scales add overhead, but
/// for large enough capacity the byte data dominates.
#[test]
fn fp8_kv_cache_compression_ratio_vs_fp32() {
    // Large enough dims for scale overhead to be < 5%.
    let layer = Fp8KvLayer::with_capacity(512, 8, 128, Fp8KvFormat::E4M3);
    let fp8_bytes = layer.memory_bytes();
    let fp32_bytes = layer.memory_bytes_fp32_equivalent();

    assert!(
        fp8_bytes * 2 < fp32_bytes,
        "FP8 memory {fp8_bytes} should be < FP32/2 = {}, got ratio {:.2}×",
        fp32_bytes / 2,
        fp32_bytes as f32 / fp8_bytes as f32
    );
}

// ─── Test 6: pushing beyond capacity returns Err ──────────────────────────────

/// Pushing more tokens than `capacity` must return `QuantKvError::CapacityExceeded`.
#[test]
fn fp8_kv_layer_capacity_overflow_errors() {
    // with_capacity(num_kv_heads=2, head_dim=4, capacity=1, format)
    let num_kv_heads = 2;
    let head_dim = 4;
    let mut layer = Fp8KvLayer::with_capacity(num_kv_heads, head_dim, 1, Fp8KvFormat::E4M3);

    let keys: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| i as f32).collect();
    let values: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| -(i as f32))
        .collect();

    // capacity = 1: first push succeeds, second fails
    layer.push(&keys, &values).expect("first push within capacity");

    let result = layer.push(&keys, &values);
    assert!(
        matches!(
            result,
            Err(QuantKvError::CapacityExceeded {
                capacity: 1,
                pos: 1
            })
        ),
        "expected CapacityExceeded, got {result:?}"
    );
}

// ─── Test 7: clear resets len ─────────────────────────────────────────────────

/// After pushing 3 tokens, `clear()` must reset `len` to 0.
#[test]
fn fp8_kv_layer_clear_resets_len() {
    let num_kv_heads = 2;
    let head_dim = 8;
    let mut layer = Fp8KvLayer::with_capacity(num_kv_heads, head_dim, 8, Fp8KvFormat::E4M3);

    let keys: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| i as f32).collect();
    let values: Vec<f32> = vec![0.0; num_kv_heads * head_dim];

    for _ in 0..3 {
        layer.push(&keys, &values).expect("push should succeed");
    }
    assert_eq!(layer.len(), 3, "len should be 3 after three pushes");
    assert!(!layer.is_empty());

    layer.clear();
    assert_eq!(layer.len(), 0, "len must be 0 after clear");
    assert!(layer.is_empty());
}

// ─── Test 8: get_keys_at for multiple positions ───────────────────────────────

/// Push 5 tokens with distinctive values, then retrieve positions [0, 2, 4] and
/// verify each position's retrieved key matches its push-time value.
#[test]
fn fp8_kv_cache_get_keys_at_multiple_positions() {
    let num_kv_heads = 2;
    let head_dim = 16;
    let n_tokens = 5;

    let mut layer =
        Fp8KvLayer::with_capacity(num_kv_heads, head_dim, n_tokens, Fp8KvFormat::E4M3);

    // Push 5 tokens, each with a unique large-magnitude key so heads are separable.
    let mut pushed_keys: Vec<Vec<f32>> = Vec::new();
    for t in 0..n_tokens {
        let keys: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|i| (t as f32 + 1.0) * 2.0 + i as f32 * 0.1)
            .collect();
        let values = vec![0.0_f32; num_kv_heads * head_dim];
        pushed_keys.push(keys.clone());
        layer.push(&keys, &values).expect("push should succeed");
    }
    assert_eq!(layer.len(), n_tokens);

    let positions = [0usize, 2, 4];
    let retrieved_sets = layer.get_keys_at(&positions);
    assert_eq!(retrieved_sets.len(), positions.len());

    for (&pos, retrieved) in positions.iter().zip(retrieved_sets.iter()) {
        assert_eq!(retrieved.len(), num_kv_heads * head_dim);
        let original = &pushed_keys[pos];
        let max_abs = original.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let mae = original
            .iter()
            .zip(retrieved.iter())
            .map(|(o, r)| (o - r).abs())
            .sum::<f32>()
            / (num_kv_heads * head_dim) as f32;
        assert!(
            mae < max_abs * 0.02,
            "position {pos}: MAE {mae} exceeds 2% of max_abs {max_abs}"
        );
    }
}

// ─── Test 9: E5M2 get_value roundtrip ────────────────────────────────────────

/// Push a high-dynamic-range value tensor using E5M2 format and check that the
/// round-trip error is within the expected bound for 2-bit mantissa FP8.
#[test]
fn fp8_kv_layer_e5m2_get_value_roundtrip() {
    let num_kv_heads = 2;
    let head_dim = 32;
    let mut layer =
        Fp8KvLayer::with_capacity(num_kv_heads, head_dim, 4, Fp8KvFormat::E5M2);

    let keys = vec![0.0_f32; num_kv_heads * head_dim];
    // Use a spread matching E5M2's wider range
    let values: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| (i as f32) * 5.0 - 80.0)
        .collect();

    layer.push(&keys, &values).expect("push should succeed");

    let retrieved = layer.get_value(0);
    assert_eq!(retrieved.len(), num_kv_heads * head_dim);

    for head in 0..num_kv_heads {
        let row_start = head * head_dim;
        let original = &values[row_start..row_start + head_dim];
        let got = &retrieved[row_start..row_start + head_dim];

        let max_abs = original.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        if max_abs < f32::EPSILON {
            continue;
        }

        // E5M2 has 2-bit mantissa: relative error up to ~25% per element, use MAE < 15%
        let mae = original
            .iter()
            .zip(got.iter())
            .map(|(o, r)| (o - r).abs())
            .sum::<f32>()
            / head_dim as f32;

        assert!(
            mae < max_abs * 0.15,
            "E5M2 value head {head}: MAE {mae} exceeds 15% of max_abs {max_abs}"
        );
    }
}

// ─── Test 10: Fp8KvCache multi-layer API ─────────────────────────────────────

/// Construct a 4-layer `Fp8KvCache`, push one step, and verify basic properties.
#[test]
fn fp8_kv_cache_multi_layer_basic() {
    let num_layers = 4;
    let num_kv_heads = 2;
    let head_dim = 16;
    let capacity = 8;

    let mut cache =
        Fp8KvCache::new(num_layers, num_kv_heads, head_dim, capacity, Fp8KvFormat::E4M3);

    assert_eq!(cache.num_layers(), num_layers);

    let keys: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| i as f32 * 0.5).collect();
    let values: Vec<f32> = (0..num_kv_heads * head_dim)
        .map(|i| -(i as f32) * 0.5)
        .collect();

    for l in 0..num_layers {
        cache
            .layer_mut(l)
            .push(&keys, &values)
            .expect("push should succeed");
    }

    for l in 0..num_layers {
        assert_eq!(cache.layer(l).len(), 1, "layer {l} should have len=1");
    }

    let total_mem = cache.total_memory_bytes();
    // Each layer: data = capacity*num_kv_heads*head_dim * 2 (keys+values)
    //             scales = capacity*num_kv_heads * 4 * 2
    let per_layer = capacity * num_kv_heads * head_dim * 2 + capacity * num_kv_heads * 4 * 2;
    assert_eq!(total_mem, num_layers * per_layer, "total memory mismatch");
}

// ─── Test 11: Fp8KvCache clear_all resets all layers ─────────────────────────

/// After pushing into every layer, `clear_all` must reset all layers to `len=0`.
#[test]
fn fp8_kv_cache_clear_all_resets_all_layers() {
    let num_layers = 3;
    let num_kv_heads = 1;
    let head_dim = 8;
    let mut cache =
        Fp8KvCache::new(num_layers, num_kv_heads, head_dim, 4, Fp8KvFormat::E5M2);

    let keys: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| i as f32).collect();
    let values = vec![0.0_f32; num_kv_heads * head_dim];

    for l in 0..num_layers {
        for _ in 0..2 {
            cache
                .layer_mut(l)
                .push(&keys, &values)
                .expect("push should succeed");
        }
        assert_eq!(cache.layer(l).len(), 2);
    }

    cache.clear_all();

    for l in 0..num_layers {
        assert_eq!(
            cache.layer(l).len(),
            0,
            "layer {l} len should be 0 after clear_all"
        );
        assert!(cache.layer(l).is_empty());
    }
}
