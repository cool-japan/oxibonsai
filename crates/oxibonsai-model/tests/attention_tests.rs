//! Tests for attention mechanism, softmax, dot product, and KV cache.

use oxibonsai_model::kv_cache::KvCache;
use oxibonsai_model::layers::attention::{attention_head, dot, softmax};

// ══════════════════════════════════════════════════════════════
// Softmax tests
// ══════════════════════════════════════════════════════════════

#[test]
fn softmax_output_sums_to_one() {
    let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    softmax(&mut values);
    let sum: f32 = values.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax should sum to 1.0, got {sum}"
    );
}

#[test]
fn softmax_larger_input_gives_larger_probability() {
    let mut values = vec![1.0, 5.0, 3.0];
    softmax(&mut values);
    assert!(
        values[1] > values[0],
        "5.0 should have higher prob than 1.0"
    );
    assert!(
        values[1] > values[2],
        "5.0 should have higher prob than 3.0"
    );
    assert!(
        values[2] > values[0],
        "3.0 should have higher prob than 1.0"
    );
}

#[test]
fn softmax_uniform_input_gives_uniform_output() {
    let mut values = vec![3.0; 5];
    softmax(&mut values);
    for &v in &values {
        assert!(
            (v - 0.2).abs() < 1e-5,
            "uniform input should give 1/N = 0.2, got {v}"
        );
    }
}

#[test]
fn softmax_handles_negative_values() {
    let mut values = vec![-10.0, -5.0, -1.0];
    softmax(&mut values);
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "should sum to 1.0 with negatives");
    // -1.0 should have highest probability
    assert!(values[2] > values[1]);
    assert!(values[1] > values[0]);
}

#[test]
fn softmax_single_element() {
    let mut values = vec![42.0];
    softmax(&mut values);
    assert!((values[0] - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_empty_is_noop() {
    let mut values: Vec<f32> = vec![];
    softmax(&mut values);
    assert!(values.is_empty());
}

#[test]
fn softmax_large_values_no_overflow() {
    let mut values = vec![1000.0, 1001.0, 999.0];
    softmax(&mut values);
    let sum: f32 = values.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "should handle large values: sum={sum}"
    );
    for &v in &values {
        assert!(v.is_finite(), "all values should be finite");
    }
}

#[test]
fn softmax_very_negative_values() {
    let mut values = vec![-1000.0, -999.0, -1001.0];
    softmax(&mut values);
    let sum: f32 = values.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "should handle very negative: sum={sum}"
    );
}

// ══════════════════════════════════════════════════════════════
// Dot product tests
// ══════════════════════════════════════════════════════════════

#[test]
fn dot_product_known_vectors() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = dot(&a, &b);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!((result - 32.0).abs() < 1e-5);
}

#[test]
fn dot_product_orthogonal_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let result = dot(&a, &b);
    assert!(result.abs() < 1e-5, "orthogonal dot should be 0");
}

#[test]
fn dot_product_parallel_vectors() {
    let a = vec![1.0, 0.0];
    let b = vec![5.0, 0.0];
    let result = dot(&a, &b);
    assert!((result - 5.0).abs() < 1e-5);
}

#[test]
fn dot_product_with_zero_vector() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![0.0, 0.0, 0.0];
    let result = dot(&a, &b);
    assert!(result.abs() < 1e-5);
}

#[test]
fn dot_product_negative_values() {
    let a = vec![1.0, -1.0];
    let b = vec![1.0, 1.0];
    let result = dot(&a, &b);
    // 1*1 + (-1)*1 = 0
    assert!(result.abs() < 1e-5);
}

// ══════════════════════════════════════════════════════════════
// Attention head tests
// ══════════════════════════════════════════════════════════════

#[test]
fn attention_single_token_returns_value() {
    let head_dim = 4;
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let keys = vec![1.0, 0.0, 0.0, 0.0];
    let values = vec![10.0, 20.0, 30.0, 40.0];
    let mut output = vec![0.0; 4];

    attention_head(&query, &keys, &values, &mut output, 1, head_dim)
        .expect("single token attention");

    // Single token: softmax([score]) = [1.0], output = values
    for i in 0..4 {
        assert!(
            (output[i] - values[i]).abs() < 1e-4,
            "single token should output value at dim {i}"
        );
    }
}

#[test]
fn attention_two_tokens_weighted_sum() {
    let head_dim = 4;
    let query = vec![1.0, 0.0, 0.0, 0.0];
    // Two keys: first aligned with query, second orthogonal
    let keys = vec![
        1.0, 0.0, 0.0, 0.0, // token 0: aligned
        0.0, 1.0, 0.0, 0.0, // token 1: orthogonal
    ];
    let values = vec![
        10.0, 0.0, 0.0, 0.0, // token 0 values
        0.0, 10.0, 0.0, 0.0, // token 1 values
    ];
    let mut output = vec![0.0; 4];

    attention_head(&query, &keys, &values, &mut output, 2, head_dim).expect("two token attention");

    // Token 0 has higher score (aligned), so output[0] > output[1]
    assert!(
        output[0] > output[1],
        "aligned token should dominate: output[0]={}, output[1]={}",
        output[0],
        output[1]
    );
}

#[test]
fn attention_two_tokens_equal_keys() {
    let head_dim = 4;
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let keys = vec![
        1.0, 0.0, 0.0, 0.0, // same key
        1.0, 0.0, 0.0, 0.0, // same key
    ];
    let values = vec![
        2.0, 4.0, 6.0, 8.0, // token 0 values
        10.0, 20.0, 30.0, 40.0, // token 1 values
    ];
    let mut output = vec![0.0; 4];

    attention_head(&query, &keys, &values, &mut output, 2, head_dim).expect("equal keys attention");

    // Equal scores -> equal weights (0.5 each)
    for d in 0..4 {
        let expected = (values[d] + values[4 + d]) / 2.0;
        assert!(
            (output[d] - expected).abs() < 1e-3,
            "dim {d}: expected {expected}, got {}",
            output[d]
        );
    }
}

#[test]
fn attention_output_is_finite() {
    let head_dim = 8;
    let query: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
    let keys: Vec<f32> = (0..24).map(|i| (i as f32) * 0.05).collect();
    let values: Vec<f32> = (0..24).map(|i| (i as f32) * 0.01).collect();
    let mut output = vec![0.0; 8];

    attention_head(&query, &keys, &values, &mut output, 3, head_dim)
        .expect("multi-token attention");

    for (d, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{d}] should be finite, got {v}");
    }
}

// ══════════════════════════════════════════════════════════════
// KV Cache tests
// ══════════════════════════════════════════════════════════════

#[test]
fn kv_cache_store_and_retrieve_position_0() {
    let mut cache = KvCache::new(1, 1, 4, 16);

    let key = vec![1.0, 2.0, 3.0, 4.0];
    let value = vec![5.0, 6.0, 7.0, 8.0];
    cache
        .try_store_key(0, 0, 0, &key)
        .expect("in-range key store");
    cache
        .try_store_value(0, 0, 0, &value)
        .expect("in-range value store");
    cache.advance();

    let keys = cache.keys_for(0, 0, 1);
    let values = cache.values_for(0, 0, 1);

    assert_eq!(keys, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(values, &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn kv_cache_multiple_positions_stored_correctly() {
    let mut cache = KvCache::new(1, 1, 4, 16);

    cache
        .try_store_key(0, 0, 0, &[1.0, 0.0, 0.0, 0.0])
        .expect("in-range key store");
    cache.advance();
    cache
        .try_store_key(0, 0, 1, &[0.0, 1.0, 0.0, 0.0])
        .expect("in-range key store");
    cache.advance();
    cache
        .try_store_key(0, 0, 2, &[0.0, 0.0, 1.0, 0.0])
        .expect("in-range key store");
    cache.advance();

    let keys = cache.keys_for(0, 0, 3);
    // Position 0: [1,0,0,0]
    assert!((keys[0] - 1.0).abs() < 1e-5);
    // Position 1: [0,1,0,0]
    assert!((keys[4 + 1] - 1.0).abs() < 1e-5);
    // Position 2: [0,0,1,0]
    assert!((keys[8 + 2] - 1.0).abs() < 1e-5);
}

#[test]
fn kv_cache_keys_and_values_are_independent() {
    let mut cache = KvCache::new(1, 1, 4, 16);

    cache
        .try_store_key(0, 0, 0, &[1.0, 1.0, 1.0, 1.0])
        .expect("in-range key store");
    cache
        .try_store_value(0, 0, 0, &[2.0, 2.0, 2.0, 2.0])
        .expect("in-range value store");

    let keys = cache.keys_for(0, 0, 1);
    let values = cache.values_for(0, 0, 1);

    // Keys and values should be different
    assert!((keys[0] - 1.0).abs() < 1e-5);
    assert!((values[0] - 2.0).abs() < 1e-5);
}

#[test]
fn kv_cache_capacity_matches_config() {
    let cache = KvCache::new(2, 4, 128, 512);
    assert_eq!(cache.max_seq_len(), 512);
    // Memory: 2 layers * 4 heads * 512 positions * 128 dims * 4 bytes * 2 (K+V)
    let expected = 2 * 4 * 512 * 128 * 4 * 2;
    assert_eq!(cache.memory_bytes(), expected);
}

#[test]
fn kv_cache_clear_resets_seq_len() {
    let mut cache = KvCache::new(1, 1, 4, 16);
    cache
        .try_store_key(0, 0, 0, &[1.0; 4])
        .expect("in-range key store");
    cache.advance();
    cache.advance();
    assert_eq!(cache.seq_len(), 2);

    cache.clear();
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn kv_cache_multi_layer_multi_head() {
    let mut cache = KvCache::new(2, 2, 4, 8);

    // Layer 0, Head 0
    cache
        .try_store_key(0, 0, 0, &[1.0, 0.0, 0.0, 0.0])
        .expect("in-range key store");
    // Layer 0, Head 1
    cache
        .try_store_key(0, 1, 0, &[0.0, 1.0, 0.0, 0.0])
        .expect("in-range key store");
    // Layer 1, Head 0
    cache
        .try_store_key(1, 0, 0, &[0.0, 0.0, 1.0, 0.0])
        .expect("in-range key store");
    // Layer 1, Head 1
    cache
        .try_store_key(1, 1, 0, &[0.0, 0.0, 0.0, 1.0])
        .expect("in-range key store");

    let k00 = cache.keys_for(0, 0, 1);
    let k01 = cache.keys_for(0, 1, 1);
    let k10 = cache.keys_for(1, 0, 1);
    let k11 = cache.keys_for(1, 1, 1);

    assert!((k00[0] - 1.0).abs() < 1e-5);
    assert!((k01[1] - 1.0).abs() < 1e-5);
    assert!((k10[2] - 1.0).abs() < 1e-5);
    assert!((k11[3] - 1.0).abs() < 1e-5);
}

#[test]
fn kv_cache_seq_len_advances() {
    let mut cache = KvCache::new(1, 1, 4, 16);
    assert_eq!(cache.seq_len(), 0);
    cache.advance();
    assert_eq!(cache.seq_len(), 1);
    cache.advance();
    assert_eq!(cache.seq_len(), 2);
    cache.advance();
    assert_eq!(cache.seq_len(), 3);
}
