//! Unit tests for [`super`] — the fused single-head attention kernels.
//!
//! Split out of `attention_fused.rs` (K-M1): the test module alone was ~1110
//! lines and the combined file stood at 1945 of the 2000-line COOLJAPAN
//! limit, which had already forced a working regression test to be reverted
//! for want of ~20 lines. Declared from the parent with
//! `#[cfg(test)] #[path = "attention_fused_tests.rs"] mod tests;`, so this is
//! still the module `crate::layers::attention_fused::tests` — `super::*`,
//! `super::super::attention::…` and every test name are unchanged.

use super::*;

/// Reference standard attention for comparison.
fn reference_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    use super::super::attention::attention_head;
    attention_head(query, keys, values, output, seq_len, head_dim)
        .expect("reference attention should succeed");
}

#[test]
fn fused_matches_standard_single_token() {
    let head_dim = 4;
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let keys = vec![1.0, 0.0, 0.0, 0.0];
    let values = vec![0.0, 1.0, 2.0, 3.0];

    let mut out_std = vec![0.0f32; head_dim];
    let mut out_fused = vec![0.0f32; head_dim];

    reference_attention(&query, &keys, &values, &mut out_std, 1, head_dim);
    fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, 1, head_dim)
        .expect("fused attention should succeed");

    for i in 0..head_dim {
        assert!(
            (out_std[i] - out_fused[i]).abs() < 1e-5,
            "dim {i}: std={}, fused={}",
            out_std[i],
            out_fused[i]
        );
    }
}

#[test]
fn fused_matches_standard_multiple_tokens() {
    let head_dim = 8;
    let seq_len = 10;

    // Generate deterministic test data
    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();

    let mut out_std = vec![0.0f32; head_dim];
    let mut out_fused = vec![0.0f32; head_dim];

    reference_attention(&query, &keys, &values, &mut out_std, seq_len, head_dim);
    fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, seq_len, head_dim)
        .expect("fused attention should succeed");

    for i in 0..head_dim {
        assert!(
            (out_std[i] - out_fused[i]).abs() < 1e-4,
            "dim {i}: std={}, fused={}",
            out_std[i],
            out_fused[i]
        );
    }
}

#[test]
fn fused_matches_standard_large_seq() {
    // Sequence longer than ATTENTION_BLOCK_SIZE to test multi-block
    let head_dim = 16;
    let seq_len = 100; // > ATTENTION_BLOCK_SIZE (32)

    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.2) - 1.0).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 7 + 3) % 23) as f32 * 0.04 - 0.5)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 11 + 5) % 19) as f32 * 0.06 - 0.6)
        .collect();

    let mut out_std = vec![0.0f32; head_dim];
    let mut out_fused = vec![0.0f32; head_dim];

    reference_attention(&query, &keys, &values, &mut out_std, seq_len, head_dim);
    fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, seq_len, head_dim)
        .expect("fused attention should succeed");

    let max_diff = out_std
        .iter()
        .zip(out_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-3,
        "max difference between standard and fused: {max_diff}"
    );
}

#[test]
fn fused_with_slice_api() {
    let head_dim = 4;
    let seq_len = 3;

    let query = vec![1.0, 0.5, -0.5, 0.0];
    let k0 = vec![0.5, 0.5, 0.0, 0.0];
    let k1 = vec![0.0, 1.0, 0.0, 0.0];
    let k2 = vec![-0.5, 0.0, 1.0, 0.0];
    let v0 = vec![1.0, 0.0, 0.0, 0.0];
    let v1 = vec![0.0, 1.0, 0.0, 0.0];
    let v2 = vec![0.0, 0.0, 1.0, 0.0];

    let keys_refs: Vec<&[f32]> = vec![&k0, &k1, &k2];
    let values_refs: Vec<&[f32]> = vec![&v0, &v1, &v2];

    let mut output = vec![0.0f32; head_dim];
    fused_attention_head(&query, &keys_refs, &values_refs, head_dim, &mut output)
        .expect("fused attention should succeed");

    // Build contiguous buffers for standard attention comparison
    let mut keys_flat = vec![0.0f32; seq_len * head_dim];
    let mut values_flat = vec![0.0f32; seq_len * head_dim];
    for (t, (k, v)) in keys_refs.iter().zip(values_refs.iter()).enumerate() {
        keys_flat[t * head_dim..(t + 1) * head_dim].copy_from_slice(k);
        values_flat[t * head_dim..(t + 1) * head_dim].copy_from_slice(v);
    }

    let mut out_std = vec![0.0f32; head_dim];
    reference_attention(
        &query,
        &keys_flat,
        &values_flat,
        &mut out_std,
        seq_len,
        head_dim,
    );

    for i in 0..head_dim {
        assert!(
            (out_std[i] - output[i]).abs() < 1e-4,
            "dim {i}: std={}, fused={}",
            out_std[i],
            output[i]
        );
    }
}

#[test]
fn fused_empty_sequence() {
    let head_dim = 4;
    let keys: Vec<&[f32]> = vec![];
    let values: Vec<&[f32]> = vec![];
    let query = vec![1.0; head_dim];
    let mut output = vec![99.0f32; head_dim];

    fused_attention_head(&query, &keys, &values, head_dim, &mut output)
        .expect("fused attention should handle empty seq");

    for &v in &output {
        assert!((v - 0.0).abs() < f32::EPSILON);
    }
}

#[test]
fn softmax_inplace_basic() {
    let mut vals = vec![1.0, 2.0, 3.0];
    softmax_inplace(&mut vals);
    let sum: f32 = vals.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(vals[0] < vals[1]);
    assert!(vals[1] < vals[2]);
}

#[test]
fn softmax_inplace_single() {
    let mut vals = vec![5.0];
    softmax_inplace(&mut vals);
    assert!((vals[0] - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_inplace_empty() {
    let mut vals: Vec<f32> = vec![];
    softmax_inplace(&mut vals); // Should not panic
}

#[test]
fn scaled_dot_product_basic() {
    let q = vec![1.0, 2.0, 3.0, 4.0];
    let k = vec![4.0, 3.0, 2.0, 1.0];
    let scale = 0.5;
    let result = scaled_dot_product(&q, &k, scale);
    // dot = 4+6+6+4 = 20, scaled = 10.0
    assert!((result - 10.0).abs() < 1e-5);
}

#[test]
fn scaled_dot_product_non_multiple_of_4() {
    let q = vec![1.0, 2.0, 3.0];
    let k = vec![4.0, 5.0, 6.0];
    let scale = 1.0;
    let result = scaled_dot_product(&q, &k, scale);
    // dot = 4+10+18 = 32
    assert!((result - 32.0).abs() < 1e-5);
}

#[test]
fn fused_validation_errors() {
    let head_dim = 4;
    let query = vec![1.0; 2]; // Too short
    let keys: Vec<&[f32]> = vec![];
    let values: Vec<&[f32]> = vec![];
    let mut output = vec![0.0f32; head_dim];

    let result = fused_attention_head(&query, &keys, &values, head_dim, &mut output);
    assert!(result.is_err());
}

#[test]
fn fused_contiguous_validation_errors() {
    let head_dim = 4;
    let query = vec![1.0; head_dim];
    let keys = vec![1.0; 4]; // seq_len=1 matches
    let values = vec![1.0; 2]; // Too short for seq_len=1
    let mut output = vec![0.0f32; head_dim];

    let result = fused_attention_head_contiguous(&query, &keys, &values, &mut output, 1, head_dim);
    assert!(result.is_err());
}

#[test]
fn fused_head_dim_128() {
    // Realistic head_dim matching Qwen3-8B
    let head_dim = 128;
    let seq_len = 50;

    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.03) - 2.0).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 7 + 3) % 31) as f32 * 0.02 - 0.3)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 13 + 7) % 23) as f32 * 0.04 - 0.5)
        .collect();

    let mut out_std = vec![0.0f32; head_dim];
    let mut out_fused = vec![0.0f32; head_dim];

    reference_attention(&query, &keys, &values, &mut out_std, seq_len, head_dim);
    fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, seq_len, head_dim)
        .expect("fused attention should succeed");

    let max_diff = out_std
        .iter()
        .zip(out_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-3,
        "max difference between standard and fused: {max_diff}"
    );
}

// ── P11.1 Step 6: New tests ────────────────────────────────────────────

/// Test masked fused attention matches naive masked attention within 1e-3.
#[test]
fn fused_masked_matches_naive_masked() {
    use super::super::attention::attention_head_with_mask;
    let head_dim = 8;
    let seq_len = 12;
    let query_pos = 6; // causal: can attend to 0..=6

    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.15) - 0.5).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 5 + 2) % 17) as f32 * 0.05 - 0.4)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 7 + 3) % 13) as f32 * 0.08 - 0.5)
        .collect();

    let mask = CausalMask::new(32);

    let mut out_naive = vec![0.0f32; head_dim];
    let mut out_fused = vec![0.0f32; head_dim];

    attention_head_with_mask(
        &query,
        &keys,
        &values,
        &mut out_naive,
        seq_len,
        head_dim,
        query_pos,
        &mask,
    )
    .expect("naive masked attention should succeed");

    fused_attention_head_contiguous_with_mask(
        &query,
        &keys,
        &values,
        &mut out_fused,
        seq_len,
        head_dim,
        query_pos,
        &mask,
    )
    .expect("fused masked attention should succeed");

    let max_diff = out_naive
        .iter()
        .zip(out_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "masked: max diff = {max_diff}; naive={out_naive:?}, fused={out_fused:?}"
    );
}

/// Masked fused attention at `head_dim = 256` (Bonsai 2), spanning
/// several block boundaries and a mid-sequence query position so both
/// fully-attended and windowed-away blocks are exercised.
#[test]
fn fused_masked_head_dim_256() {
    use super::super::attention::attention_head_with_mask;
    let head_dim = 256;
    let seq_len = 150;
    let query_pos = 90; // several full blocks allowed, several masked away

    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.01) - 1.2).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 5 + 2) % 37) as f32 * 0.015 - 0.28)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 9 + 4) % 29) as f32 * 0.02 - 0.29)
        .collect();

    let mask = CausalMask::new(seq_len);

    let mut out_naive = vec![0.0f32; head_dim];
    let mut out_fused = vec![0.0f32; head_dim];

    attention_head_with_mask(
        &query,
        &keys,
        &values,
        &mut out_naive,
        seq_len,
        head_dim,
        query_pos,
        &mask,
    )
    .expect("naive masked attention should succeed");

    fused_attention_head_contiguous_with_mask(
        &query,
        &keys,
        &values,
        &mut out_fused,
        seq_len,
        head_dim,
        query_pos,
        &mask,
    )
    .expect("fused masked attention should succeed");

    let max_diff = out_naive
        .iter()
        .zip(out_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "masked head_dim=256: max diff = {max_diff}"
    );
}

/// Pins `Accum::new`'s construction-time zero fill as LOAD-BEARING
/// (see the comment on `rescale_for_block_max`), not merely a defensive
/// default that would be safe to remove. This test does NOT verify the
/// `fill(0.0)` removal in `rescale_for_block_max` itself -- that is
/// covered by the 1e-5 frozen-old parity tests, which exercise the
/// normal (some-block-contributes) path. What this test covers is the
/// one path those parity tests do not: a mask that disallows every
/// position for every query, so every block's `count` stays `0` in
/// `fused_attention_head_contiguous_with_mask`, `rescale_for_block_max`
/// is never called at all, and the output is well-defined ONLY because
/// `Accum::new` already zeroed it at construction. If `Accum::new` ever
/// stops zero-filling (e.g. a future switch to `MaybeUninit`), THIS is
/// the test that must catch it. `SlidingWindowConfig::new(0, 0)`
/// reaches `is_in_window`'s `window_size == 0` early return, which is
/// `false` even for a query attending to itself.
#[test]
fn fully_masked_query_yields_zero_output() {
    use crate::layers::sliding_window::SlidingWindowConfig;

    let head_dim = 16;
    let seq_len = 70; // spans more than one ATTENTION_BLOCK_SIZE block
    let query_pos = 40;

    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.1) - 0.5).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 5 + 1) % 23) as f32 * 0.03 - 0.4)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 7 + 2) % 19) as f32 * 0.04 - 0.3)
        .collect();

    let mask = CausalMask::with_sliding_window(seq_len, SlidingWindowConfig::new(0, 0));
    // Confirm the premise: even self-attention is disallowed.
    assert!(!mask.is_allowed(query_pos, query_pos));

    let mut output = vec![f32::NAN; head_dim]; // pre-poisoned: must be overwritten
    fused_attention_head_contiguous_with_mask(
        &query,
        &keys,
        &values,
        &mut output,
        seq_len,
        head_dim,
        query_pos,
        &mask,
    )
    .expect("fully-masked fused attention should still succeed, not error");

    assert!(
        output.iter().all(|&x| x == 0.0),
        "fully-masked query must yield an all-zero output, got {output:?}"
    );
}

/// Long-context test at S=4096 exercising many block boundaries.
#[test]
fn fused_long_context_4096() {
    let head_dim = 16;
    let seq_len = 4096;

    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.1) - 0.8).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 3 + 1) % 29) as f32 * 0.02 - 0.3)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 7 + 5) % 17) as f32 * 0.03 - 0.25)
        .collect();

    let mut out_std = vec![0.0f32; head_dim];
    let mut out_fused = vec![0.0f32; head_dim];

    reference_attention(&query, &keys, &values, &mut out_std, seq_len, head_dim);
    fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, seq_len, head_dim)
        .expect("fused long-context attention should succeed");

    let max_diff = out_std
        .iter()
        .zip(out_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "long-context S=4096: max diff = {max_diff}"
    );
}

/// `head_dim = 256` (Bonsai 2) spanning several block boundaries —
/// exercises the `Accum::Stack` fast path at the exact bound.
#[test]
fn fused_head_dim_256_multi_block() {
    let head_dim = 256;
    let seq_len = 200; // ~7 blocks of ATTENTION_BLOCK_SIZE=32

    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.008) - 1.0).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 7 + 3) % 41) as f32 * 0.012 - 0.24)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 11 + 5) % 33) as f32 * 0.018 - 0.29)
        .collect();

    let mut out_std = vec![0.0f32; head_dim];
    let mut out_fused = vec![0.0f32; head_dim];

    reference_attention(&query, &keys, &values, &mut out_std, seq_len, head_dim);
    fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, seq_len, head_dim)
        .expect("fused attention should succeed at head_dim=256");

    let max_diff = out_std
        .iter()
        .zip(out_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "head_dim=256: max diff = {max_diff}");
}

/// `head_dim = 320 > MAX_HEAD_DIM` exercises the `Accum::Heap` fallback
/// branch for correctness (it is intentionally excluded from the
/// zero-allocation test below, since it is expected to allocate).
#[test]
fn fused_head_dim_320_heap_fallback() {
    let head_dim = 320;
    let seq_len = 70;

    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.006) - 0.9).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 7 + 3) % 43) as f32 * 0.01 - 0.21)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 11 + 5) % 31) as f32 * 0.015 - 0.23)
        .collect();

    let mut out_std = vec![0.0f32; head_dim];
    let mut out_fused = vec![0.0f32; head_dim];

    reference_attention(&query, &keys, &values, &mut out_std, seq_len, head_dim);
    fused_attention_head_contiguous(&query, &keys, &values, &mut out_fused, seq_len, head_dim)
        .expect("fused attention should succeed above MAX_HEAD_DIM");

    let max_diff = out_std
        .iter()
        .zip(out_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "head_dim=320 (heap fallback): max diff = {max_diff}"
    );
}

/// Multi-head test: 4 Q heads, 2 KV heads (GQA 2:1), head_dim=32.
#[test]
fn fused_multi_head_gqa() {
    let head_dim = 32;
    let seq_len = 40;
    let num_q_heads = 4;
    let num_kv_heads = 2;
    let q_heads_per_kv = num_q_heads / num_kv_heads;

    let query_all: Vec<f32> = (0..num_q_heads * head_dim)
        .map(|i| (i as f32 * 0.05) - 1.0)
        .collect();
    let keys: Vec<Vec<f32>> = (0..num_kv_heads)
        .map(|kv| {
            (0..seq_len * head_dim)
                .map(|i| ((i * (kv + 3) + 1) % 23) as f32 * 0.04 - 0.45)
                .collect()
        })
        .collect();
    let values: Vec<Vec<f32>> = (0..num_kv_heads)
        .map(|kv| {
            (0..seq_len * head_dim)
                .map(|i| ((i * (kv + 5) + 2) % 19) as f32 * 0.06 - 0.55)
                .collect()
        })
        .collect();

    let mut out_fused = vec![0.0f32; num_q_heads * head_dim];
    let mut out_ref = vec![0.0f32; num_q_heads * head_dim];

    // Reference: naive per-head
    for q_head in 0..num_q_heads {
        let kv_head = q_head / q_heads_per_kv;
        let q_start = q_head * head_dim;
        let mut head_out = vec![0.0f32; head_dim];
        reference_attention(
            &query_all[q_start..q_start + head_dim],
            &keys[kv_head],
            &values[kv_head],
            &mut head_out,
            seq_len,
            head_dim,
        );
        out_ref[q_start..q_start + head_dim].copy_from_slice(&head_out);
    }

    // Fused: per-head
    for q_head in 0..num_q_heads {
        let kv_head = q_head / q_heads_per_kv;
        let q_start = q_head * head_dim;
        fused_attention_head_contiguous(
            &query_all[q_start..q_start + head_dim],
            &keys[kv_head],
            &values[kv_head],
            &mut out_fused[q_start..q_start + head_dim],
            seq_len,
            head_dim,
        )
        .expect("fused multi-head attention should succeed");
    }

    let max_diff = out_ref
        .iter()
        .zip(out_fused.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "multi-head GQA: max diff = {max_diff}");
}

// ── P11.2: SIMD dot correctness ────────────────────────────────────────

/// Verify SIMD dot_f32 matches scalar for various lengths.
#[test]
fn dot_f32_matches_scalar() {
    let test_lengths = [128usize, 127, 513, 1024];

    for &n in &test_lengths {
        // Deterministic pseudo-random vectors
        let a: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + 3) % 31) as f32 * 0.1 - 1.5)
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| ((i * 11 + 5) % 23) as f32 * 0.1 - 1.2)
            .collect();

        let scalar_val = dot_f32_scalar(&a, &b);
        let simd_val = dot_f32(&a, &b);

        let denom = scalar_val.abs().max(1.0);
        let rel_err = (simd_val - scalar_val).abs() / denom;
        assert!(
            rel_err < 1e-5,
            "n={n}: scalar={scalar_val}, simd={simd_val}, rel_err={rel_err}"
        );
    }
}

// ── K-M1 item (a): axpy_f32 / scale_f32 SIMD correctness ───────────────

#[test]
fn axpy_f32_matches_scalar() {
    let test_lengths = [128usize, 127, 513, 1024, 256];

    for &n in &test_lengths {
        let src: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + 3) % 31) as f32 * 0.1 - 1.5)
            .collect();
        let scale = 0.37f32;

        let mut acc_scalar: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 0.4).collect();
        let mut acc_simd = acc_scalar.clone();

        axpy_f32_scalar(&mut acc_scalar, scale, &src);
        axpy_f32(&mut acc_simd, scale, &src);

        for i in 0..n {
            let diff = (acc_scalar[i] - acc_simd[i]).abs();
            assert!(
                diff < 1e-4,
                "n={n} i={i}: scalar={}, simd={}",
                acc_scalar[i],
                acc_simd[i]
            );
        }
    }
}

#[test]
fn scale_f32_matches_scalar() {
    let test_lengths = [128usize, 127, 513, 1024, 256];

    for &n in &test_lengths {
        let factor = 0.83f32;
        let mut acc_scalar: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02) - 0.7).collect();
        let mut acc_simd = acc_scalar.clone();

        scale_f32_scalar(&mut acc_scalar, factor);
        scale_f32(&mut acc_simd, factor);

        for i in 0..n {
            let diff = (acc_scalar[i] - acc_simd[i]).abs();
            assert!(
                diff < 1e-5,
                "n={n} i={i}: scalar={}, simd={}",
                acc_scalar[i],
                acc_simd[i]
            );
        }
    }
}

// ── K-M1 item (c): over-long operands are ignored, never read past ─────

/// `dot_f32` must stop at the shorter operand.
///
/// Before K-M1 item (c) only `dot_f32_scalar` clamped; `dot_f32_neon` and
/// `dot_f32_avx2_fma` drove the loop from `a.len()` alone and dereferenced
/// `b.as_ptr().add(i)` unchecked, so `a` longer than `b` read past the end of
/// `b`'s allocation in release — the `debug_assert_eq!` that was supposed to
/// catch it is compiled out exactly there. Both length orders are checked:
/// the result must equal the dot product over the common prefix, computed
/// independently here rather than by calling the function under test.
#[test]
fn dot_f32_stops_at_the_shorter_operand() {
    let long: Vec<f32> = (0..40).map(|i| (i as f32) * 0.13 - 1.7).collect();
    let short: Vec<f32> = (0..24).map(|i| (i as f32) * -0.07 + 0.9).collect();
    let common: f32 = long
        .iter()
        .zip(short.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>();

    for (a, b) in [(&long, &short), (&short, &long)] {
        let got = dot_f32(a, b);
        assert!(
            (got - common).abs() < 1e-3,
            "dot_f32(len={}, len={}) = {got}, expected the common-prefix dot \
             {common} — a SIMD path that drives its loop from one operand's \
             length alone reads past the other",
            a.len(),
            b.len()
        );
    }

    // Degenerate ends of the contract.
    assert_eq!(dot_f32(&long, &[]), 0.0);
    assert_eq!(dot_f32(&[], &long), 0.0);
}

/// The public attention entry points accept `query.len() > head_dim` (they
/// validate with `<`, not `!=`). An over-long query must produce **bit-exact**
/// the same output as the exactly-sized one: the excess dimensions are not
/// folded into any score, and — before item (c) — they were also what made the
/// NEON `dot_f32` run off the end of a `head_dim`-length key slice.
///
/// `head_dim = 40` engages the NEON 8-wide body plus a remainder, and
/// `seq_len = 70` spans three 32-position blocks.
#[test]
fn over_long_query_matches_exact_length_query_at_every_entry_point() {
    use crate::layers::sliding_window::SlidingWindowConfig;

    let head_dim = 40usize;
    let seq_len = 70usize;
    let query_pos = seq_len - 1;

    let query_exact: Vec<f32> = (0..head_dim)
        .map(|i| ((i * 5 + 3) % 13) as f32 * 0.11 - 0.6)
        .collect();
    // Trailing junk large enough that folding any of it into a score would
    // move the output well outside any float tolerance.
    let mut query_long = query_exact.clone();
    query_long.extend_from_slice(&[
        900.0, -750.0, 610.0, -480.0, 330.0, -220.0, 170.0, -95.0, 40.0,
    ]);

    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 7 + 2) % 31) as f32 * 0.03 - 0.45)
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 13 + 5) % 19) as f32 * 0.05 - 0.5)
        .collect();
    let key_refs: Vec<&[f32]> = (0..seq_len)
        .map(|t| &keys[t * head_dim..(t + 1) * head_dim])
        .collect();
    let value_refs: Vec<&[f32]> = (0..seq_len)
        .map(|t| &values[t * head_dim..(t + 1) * head_dim])
        .collect();
    let mask = CausalMask::with_sliding_window(seq_len, SlidingWindowConfig::new(24, 3));

    let mut exact = vec![0.0f32; head_dim];
    let mut over = vec![0.0f32; head_dim];

    fused_attention_head_contiguous(&query_exact, &keys, &values, &mut exact, seq_len, head_dim)
        .expect("contiguous, exact-length query");
    fused_attention_head_contiguous(&query_long, &keys, &values, &mut over, seq_len, head_dim)
        .expect("contiguous, over-long query");
    assert_eq!(
        exact, over,
        "fused_attention_head_contiguous: an over-long query must be ignored \
         past head_dim, bit for bit"
    );

    exact.fill(0.0);
    over.fill(0.0);
    fused_attention_head_contiguous_with_mask(
        &query_exact,
        &keys,
        &values,
        &mut exact,
        seq_len,
        head_dim,
        query_pos,
        &mask,
    )
    .expect("masked, exact-length query");
    fused_attention_head_contiguous_with_mask(
        &query_long,
        &keys,
        &values,
        &mut over,
        seq_len,
        head_dim,
        query_pos,
        &mask,
    )
    .expect("masked, over-long query");
    assert_eq!(
        exact, over,
        "fused_attention_head_contiguous_with_mask: an over-long query must be \
         ignored past head_dim, bit for bit"
    );

    exact.fill(0.0);
    over.fill(0.0);
    fused_attention_head(&query_exact, &key_refs, &value_refs, head_dim, &mut exact)
        .expect("slice API, exact-length query");
    fused_attention_head(&query_long, &key_refs, &value_refs, head_dim, &mut over)
        .expect("slice API, over-long query");
    assert_eq!(
        exact, over,
        "fused_attention_head: an over-long query must be ignored past \
         head_dim, bit for bit"
    );

    // The fixture must be non-degenerate, or equality proves nothing.
    assert!(
        exact.iter().any(|v| v.abs() > 1e-6),
        "degenerate fixture: the attention output is all zeros"
    );
}

// ── K-M1 regression harness ─────────────────────────────────────────
//
// A frozen copy of the PRE-CHANGE algorithm (scalar V-accumulate,
// rescale on every new max within a block, `Vec`-per-block). Used only
// to prove the acceptance criterion ("parity to 1e-5 vs the current
// implementation") and to provide a same-binary "before" baseline for
// the ignored micro-benchmark below. Deliberately NOT deduplicated
// against the real implementation: it must keep computing the OLD
// algorithm even as the real one evolves, or it stops being a useful
// regression oracle.
mod old_algorithm {
    use super::*;

    struct OldOnlineSoftmaxState {
        max_val: f32,
        sum_exp: f32,
        output: Vec<f32>,
    }

    impl OldOnlineSoftmaxState {
        fn new(head_dim: usize) -> Self {
            Self {
                max_val: f32::NEG_INFINITY,
                sum_exp: 0.0,
                output: vec![0.0f32; head_dim],
            }
        }

        fn update(&mut self, scores: &[f32], values: &[&[f32]], head_dim: usize) {
            for (idx, &score) in scores.iter().enumerate() {
                let v = values[idx];
                if score > self.max_val {
                    let rescale = if self.max_val == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (self.max_val - score).exp()
                    };
                    self.sum_exp *= rescale;
                    for d in 0..head_dim {
                        self.output[d] *= rescale;
                    }
                    self.max_val = score;
                }
                let exp_score = (score - self.max_val).exp();
                self.sum_exp += exp_score;
                for (out_d, &v_d) in self.output[..head_dim].iter_mut().zip(v.iter()) {
                    *out_d += exp_score * v_d;
                }
            }
        }

        fn finalize(&mut self) {
            if self.sum_exp > 0.0 {
                let inv_sum = 1.0 / self.sum_exp;
                for d in self.output.iter_mut() {
                    *d *= inv_sum;
                }
            }
        }
    }

    pub(super) fn old_fused_attention_head_contiguous(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        output: &mut [f32],
        seq_len: usize,
        head_dim: usize,
    ) -> ModelResult<()> {
        if seq_len == 0 {
            for d in output.iter_mut() {
                *d = 0.0;
            }
            return Ok(());
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut state = OldOnlineSoftmaxState::new(head_dim);
        let mut pos = 0;
        while pos < seq_len {
            let block_end = (pos + ATTENTION_BLOCK_SIZE).min(seq_len);
            let block_len = block_end - pos;
            let mut block_scores = Vec::with_capacity(block_len);
            let mut block_values: Vec<&[f32]> = Vec::with_capacity(block_len);
            for t in pos..block_end {
                let k_slice = &keys[t * head_dim..(t + 1) * head_dim];
                block_scores.push(dot_f32(query, k_slice) * scale);
                block_values.push(&values[t * head_dim..(t + 1) * head_dim]);
            }
            state.update(&block_scores, &block_values, head_dim);
            pos = block_end;
        }
        state.finalize();
        output[..head_dim].copy_from_slice(&state.output[..head_dim]);
        Ok(())
    }

    /// Masked sibling of [`old_fused_attention_head_contiguous`]: same
    /// frozen pre-change internals (`OldOnlineSoftmaxState::update`'s
    /// per-score rescale, `Vec`-per-block), with the identical mask
    /// skip `fused_attention_head_contiguous_with_mask` applies. Exists
    /// because that masked entry point got the same flash-v2
    /// restructuring as the unmasked one but, before this test, had no
    /// 1e-5 frozen-old comparison of its own (only 1e-3/1e-4 against
    /// the independent naive reference).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn old_fused_attention_head_contiguous_with_mask(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        output: &mut [f32],
        seq_len: usize,
        head_dim: usize,
        query_pos: usize,
        mask: &CausalMask,
    ) -> ModelResult<()> {
        if seq_len == 0 {
            for d in output.iter_mut() {
                *d = 0.0;
            }
            return Ok(());
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut state = OldOnlineSoftmaxState::new(head_dim);
        let mut pos = 0;
        while pos < seq_len {
            let block_end = (pos + ATTENTION_BLOCK_SIZE).min(seq_len);
            let mut block_scores = Vec::with_capacity(block_end - pos);
            let mut block_values: Vec<&[f32]> = Vec::with_capacity(block_end - pos);
            for t in pos..block_end {
                if !mask.is_allowed(query_pos, t) {
                    continue;
                }
                let k_slice = &keys[t * head_dim..(t + 1) * head_dim];
                block_scores.push(dot_f32(query, k_slice) * scale);
                block_values.push(&values[t * head_dim..(t + 1) * head_dim]);
            }
            if !block_scores.is_empty() {
                state.update(&block_scores, &block_values, head_dim);
            }
            pos = block_end;
        }
        state.finalize();
        output[..head_dim].copy_from_slice(&state.output[..head_dim]);
        Ok(())
    }
}

/// ACCEPTANCE: parity to 1e-5 vs the pre-change algorithm on random
/// (seq_len, head_dim) pairs including head_dim=256 and seq_len
/// crossing several block boundaries.
#[test]
fn new_matches_old_algorithm_within_1e5() {
    use old_algorithm::old_fused_attention_head_contiguous;

    let cases: &[(usize, usize)] = &[
        (1, 8),
        (10, 8),
        (33, 16),
        (65, 32),
        (127, 64),
        (200, 64),
        (257, 128),
        (129, 256),
        (300, 256),
    ];

    for &(seq_len, head_dim) in cases {
        let query: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 3 + 1) % 11) as f32 * 0.07 - 0.4)
            .collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 2) % 31) as f32 * 0.03 - 0.45)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 13 + 5) % 19) as f32 * 0.05 - 0.5)
            .collect();

        let mut out_old = vec![0.0f32; head_dim];
        let mut out_new = vec![0.0f32; head_dim];
        old_fused_attention_head_contiguous(
            &query,
            &keys,
            &values,
            &mut out_old,
            seq_len,
            head_dim,
        )
        .expect("old algorithm should succeed");
        fused_attention_head_contiguous(&query, &keys, &values, &mut out_new, seq_len, head_dim)
            .expect("new algorithm should succeed");

        for i in 0..head_dim {
            let diff = (out_old[i] - out_new[i]).abs();
            assert!(
                diff < 1e-5,
                "seq_len={seq_len} head_dim={head_dim} dim={i}: old={}, new={}, diff={diff}",
                out_old[i],
                out_new[i]
            );
        }
    }
}

/// ACCEPTANCE (masked variant): the same 1e-5 frozen-old parity check
/// as [`new_matches_old_algorithm_within_1e5`], but for
/// `fused_attention_head_contiguous_with_mask` — closes the gap where
/// only the unmasked entry point had a frozen-old comparison.
#[test]
fn new_matches_old_algorithm_with_mask_within_1e5() {
    use old_algorithm::old_fused_attention_head_contiguous_with_mask;

    // (seq_len, head_dim, query_pos): query_pos < seq_len - 1 in most
    // cases so both "allowed" and "masked away" positions are exercised
    // within the same call, across several block boundaries.
    let cases: &[(usize, usize, usize)] = &[
        (10, 8, 5),
        (65, 32, 40),
        (127, 64, 126),
        (257, 128, 100),
        (300, 256, 299),
    ];

    for &(seq_len, head_dim, query_pos) in cases {
        let query: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 3 + 1) % 11) as f32 * 0.07 - 0.4)
            .collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 2) % 31) as f32 * 0.03 - 0.45)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 13 + 5) % 19) as f32 * 0.05 - 0.5)
            .collect();
        let mask = CausalMask::new(seq_len);

        let mut out_old = vec![0.0f32; head_dim];
        let mut out_new = vec![0.0f32; head_dim];
        old_fused_attention_head_contiguous_with_mask(
            &query,
            &keys,
            &values,
            &mut out_old,
            seq_len,
            head_dim,
            query_pos,
            &mask,
        )
        .expect("old masked algorithm should succeed");
        fused_attention_head_contiguous_with_mask(
            &query,
            &keys,
            &values,
            &mut out_new,
            seq_len,
            head_dim,
            query_pos,
            &mask,
        )
        .expect("new masked algorithm should succeed");

        for i in 0..head_dim {
            let diff = (out_old[i] - out_new[i]).abs();
            assert!(
                diff < 1e-5,
                "masked seq_len={seq_len} head_dim={head_dim} query_pos={query_pos} \
                 dim={i}: old={}, new={}, diff={diff}",
                out_old[i],
                out_new[i]
            );
        }
    }
}

/// `fused_attention_head` (the reference-slice-per-position API) shares
/// `OnlineSoftmaxState` with `fused_attention_head_contiguous` and got
/// the identical flash-v2 restructuring, so it is checked against the
/// SAME frozen contiguous oracle above (via `&[&[f32]]` views over the
/// same buffers) rather than freezing a third old implementation.
#[test]
fn fused_attention_head_matches_old_contiguous_oracle_within_1e5() {
    use old_algorithm::old_fused_attention_head_contiguous;

    let cases: &[(usize, usize)] = &[(1, 8), (33, 16), (127, 64), (300, 256)];

    for &(seq_len, head_dim) in cases {
        let query: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 3 + 1) % 11) as f32 * 0.07 - 0.4)
            .collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 2) % 31) as f32 * 0.03 - 0.45)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 13 + 5) % 19) as f32 * 0.05 - 0.5)
            .collect();
        let key_refs: Vec<&[f32]> = (0..seq_len)
            .map(|t| &keys[t * head_dim..(t + 1) * head_dim])
            .collect();
        let value_refs: Vec<&[f32]> = (0..seq_len)
            .map(|t| &values[t * head_dim..(t + 1) * head_dim])
            .collect();

        let mut out_old = vec![0.0f32; head_dim];
        let mut out_new = vec![0.0f32; head_dim];
        old_fused_attention_head_contiguous(
            &query,
            &keys,
            &values,
            &mut out_old,
            seq_len,
            head_dim,
        )
        .expect("old algorithm should succeed");
        fused_attention_head(&query, &key_refs, &value_refs, head_dim, &mut out_new)
            .expect("new reference-slice algorithm should succeed");

        for i in 0..head_dim {
            let diff = (out_old[i] - out_new[i]).abs();
            assert!(
                diff < 1e-5,
                "seq_len={seq_len} head_dim={head_dim} dim={i}: old={}, new={}, diff={diff}",
                out_old[i],
                out_new[i]
            );
        }
    }
}

/// ACCEPTANCE (K-M1, non-contiguous allowed set): sliding-window +
/// attention-sink mask parity against the frozen pre-change masked oracle
/// at 1e-5.
///
/// `sink_tokens > 0` with a small window makes the allowed set genuinely
/// NON-CONTIGUOUS (positions `0..sink` plus a recent window), so individual
/// 32-position blocks hold a mix of allowed and masked positions — the case
/// the `block_scores[count]` / `block_positions[count]` compaction in
/// `fused_attention_head_contiguous_with_mask` exists for, and the one the
/// other masked oracle test cannot reach: `CausalMask::new`'s allowed set is
/// always a contiguous prefix. The in-test premise assertion fails loudly if
/// a future mask change makes a case contiguous again, so this can never
/// silently stop covering compaction.
#[test]
fn sliding_window_sink_mask_matches_old_within_1e5() {
    use crate::layers::sliding_window::SlidingWindowConfig;
    use old_algorithm::old_fused_attention_head_contiguous_with_mask;

    // (seq_len, head_dim, query_pos, window_size, sink_tokens)
    let cases: &[(usize, usize, usize, usize, usize)] = &[
        (300, 256, 299, 40, 4),
        (300, 256, 150, 33, 7),
        (257, 128, 200, 64, 3),
        (129, 64, 128, 35, 5),
        (70, 16, 69, 10, 2),
    ];

    for &(seq_len, head_dim, query_pos, window_size, sink_tokens) in cases {
        let query: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 3 + 1) % 11) as f32 * 0.07 - 0.4)
            .collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 2) % 31) as f32 * 0.03 - 0.45)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 13 + 5) % 19) as f32 * 0.05 - 0.5)
            .collect();
        let mask = CausalMask::with_sliding_window(
            seq_len,
            SlidingWindowConfig::new(window_size, sink_tokens),
        );

        // Premise check: the allowed set really is non-contiguous, i.e. at
        // least one block holds both an allowed and a masked position.
        let allowed: Vec<bool> = (0..seq_len)
            .map(|t| mask.is_allowed(query_pos, t))
            .collect();
        let mixed_block = (0..seq_len.div_ceil(ATTENTION_BLOCK_SIZE)).any(|b| {
            let lo = b * ATTENTION_BLOCK_SIZE;
            let hi = ((b + 1) * ATTENTION_BLOCK_SIZE).min(seq_len);
            allowed[lo..hi].iter().any(|&a| a) && allowed[lo..hi].iter().any(|&a| !a)
        });
        assert!(
            mixed_block,
            "seq_len={seq_len} window={window_size} sink={sink_tokens}: no block mixes \
             allowed and masked positions -- this case would not exercise compaction"
        );

        let mut out_old = vec![0.0f32; head_dim];
        let mut out_new = vec![0.0f32; head_dim];
        old_fused_attention_head_contiguous_with_mask(
            &query,
            &keys,
            &values,
            &mut out_old,
            seq_len,
            head_dim,
            query_pos,
            &mask,
        )
        .expect("old masked algorithm should succeed");
        fused_attention_head_contiguous_with_mask(
            &query,
            &keys,
            &values,
            &mut out_new,
            seq_len,
            head_dim,
            query_pos,
            &mask,
        )
        .expect("new masked algorithm should succeed");

        for i in 0..head_dim {
            let diff = (out_old[i] - out_new[i]).abs();
            assert!(
                diff < 1e-5,
                "sw seq_len={seq_len} head_dim={head_dim} query_pos={query_pos} \
                 window={window_size} sink={sink_tokens} dim={i}: old={}, new={}, diff={diff}",
                out_old[i],
                out_new[i]
            );
        }
    }
}

// ── K-M1 item (c): zero-allocation regression test ─────────────────────
//
// The counting `#[global_allocator]` this needs lives in
// `crate::test_alloc` rather than here: Rust permits exactly one
// `#[global_allocator]` per test binary, and `model::types` needs the
// identical mechanism for its own zero-allocation regression test
// (M-22) in this same crate's `--lib` unit-test binary. Sharing one
// declaration avoids an `error: cannot define multiple global
// allocators` collision between the two.
use crate::test_alloc::count_allocations;

#[test]
fn fused_attention_head_contiguous_zero_allocs() {
    for &head_dim in &[64usize, 128, 256] {
        let seq_len = 200; // several ATTENTION_BLOCK_SIZE=32 boundaries
        let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.01) - 1.0).collect();
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 1) % 29) as f32 * 0.02 - 0.3)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 11 + 3) % 23) as f32 * 0.03 - 0.4)
            .collect();
        let mut output = vec![0.0f32; head_dim];

        // Warm-up call outside the measured region (e.g. first-touch of
        // any lazily-initialized state): there is none on this path
        // today, but this keeps the test robust if that ever changes.
        fused_attention_head_contiguous(&query, &keys, &values, &mut output, seq_len, head_dim)
            .expect("warm-up call should succeed");

        let (result, allocs) = count_allocations(|| {
            fused_attention_head_contiguous(&query, &keys, &values, &mut output, seq_len, head_dim)
        });
        result.expect("measured call should succeed");
        assert_eq!(
            allocs, 0,
            "head_dim={head_dim}: expected zero heap allocations, got {allocs}"
        );
    }
}

// ── K-M1 acceptance: decode-attention speedup vs the pre-change code ───
//
// Release-only (debug-mode SIMD timing is meaningless) and `#[ignore]`d
// so it never runs inside the normal `cargo test` gate. Run explicitly:
//   cargo test --release -p oxibonsai-model --lib \
//     layers::attention_fused::tests::decode_attention_speedup_vs_old_algorithm \
//     -- --ignored --nocapture
//
// CORRECTED (post-review): an earlier version of this comment claimed
// "a real, measured outcome (up to ~1.85x observed)" for the spec's
// "expect >= 1.5x" and hard-asserted only `speedup > 1.0` as the one
// property that "held across every run at every configuration tried".
// Neither claim survived independent re-measurement and both are wrong;
// this is the honest replacement.
//
// This is a hand-rolled `std::time::Instant` wall-clock comparison, not
// a statistically-sound benchmark (no warm-up isolation, no outlier
// rejection, one sample per invocation) -- that is itself a gap: the
// acceptance criterion calls for "a criterion micro-benchmark", and this
// is not one. Repeated runs of this exact test (head_dim=256,
// seq_len=8192), run serially with no other load on the machine, gave
// 1.38x / 1.00x / 0.86x / 1.09x / 1.14x -- i.e. sometimes BELOW 1.0x. A
// sweep across seq_len in {256, 1024, 4096, 8192} x head_dim in
// {128, 256} gave 0.96x-1.36x, never >= 1.5x. Four further independent
// idle-machine release runs gave 0.87x / 0.94x / 1.05x / 1.23x -- two of
// them BELOW 1.0x. So the honest summary of the restructuring (SIMD
// V-accumulate + O(1)-per-block rescale instead of a scalar loop
// rescaled on every new max) is: **a wash to modestly faster at this
// configuration**, not "roughly 1.0x to 1.4x", which an earlier revision
// of this comment claimed and which the sub-1.0x runs contradict.
//
// Note also that seq_len=8192 is close to the worst case for showing the
// win this change actually targets: the removed cost is per-call
// allocation churn plus a per-new-max rescale, both roughly O(head_dim)
// per block, while the retained cost is the O(seq_len x head_dim) score
// and accumulate work. A long context therefore dilutes the saving into
// the noise; the change pays off most at short-to-medium contexts and in
// allocation count (which `fused_attention_head_contiguous_zero_allocs`
// pins exactly, and which is load-independent).
//
// The withdrawn `>= 1.5x` acceptance criterion is NOT reproducible and
// must not be reported as delivered. Nothing here should be treated as a
// stable floor in either direction, so no ratio is asserted at all: this
// test can fail to compile or panic on a real error, but a slow run is
// reported, not failed.
#[test]
#[ignore = "release-only wall-clock benchmark; see comment above for the invocation"]
fn decode_attention_speedup_vs_old_algorithm() {
    use old_algorithm::old_fused_attention_head_contiguous;
    use std::time::Instant;

    // head_dim=256 matches Bonsai 2 (this finding's own motivating
    // model: `qwen35.attention.key_length` / `value_length`), not
    // Qwen3's 128 -- the ratio is markedly overhead-bound (~1.2x) at
    // head_dim=128 and only opens up once each block's compute
    // amortizes the fixed per-call cost. seq_len=8192 keeps the ~8 MB
    // keys+values working set cache-resident on this M3 so the
    // measurement reflects the compute-side fix (SIMD + O(1) rescale)
    // rather than DRAM bandwidth (see the comment above).
    let head_dim = 256usize;
    let seq_len = 8192usize;
    let iters = 50usize;

    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.01) - 0.6).collect();
    // Keys grow increasingly aligned with `query` as position `t`
    // approaches the end of the sequence: this is the finding's own
    // worst case for the OLD algorithm ("scores usually increase toward
    // the recent end of a causal KV cache, so the bad case is the
    // common case"), modeling the recency bias a real causal decode
    // exhibits (unlike a weak trend riding on top of dominant noise,
    // which would understate the win this restructuring targets).
    let keys: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| {
            let t = i / head_dim;
            let d = i % head_dim;
            let noise = ((i * 7 + 2) % 31) as f32 * 0.004 - 0.06;
            let alignment = (t as f32 / seq_len as f32) * query[d] * 1.5;
            noise + alignment
        })
        .collect();
    let values: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i * 13 + 5) % 19) as f32 * 0.03 - 0.4)
        .collect();
    let mut out = vec![0.0f32; head_dim];

    let start_old = Instant::now();
    for _ in 0..iters {
        old_fused_attention_head_contiguous(&query, &keys, &values, &mut out, seq_len, head_dim)
            .expect("old algorithm should succeed");
    }
    let old_elapsed = start_old.elapsed();

    let start_new = Instant::now();
    for _ in 0..iters {
        fused_attention_head_contiguous(&query, &keys, &values, &mut out, seq_len, head_dim)
            .expect("new algorithm should succeed");
    }
    let new_elapsed = start_new.elapsed();

    let speedup = old_elapsed.as_secs_f64() / new_elapsed.as_secs_f64();
    // No pass/fail assertion on `speedup`: this is a wall-clock
    // measurement on a shared, non-isolated machine, not a correctness
    // check, and this exact configuration has been observed both above
    // and below 1.0x across repeated idle-machine runs (see the doc
    // comment above) -- asserting any floor here is either flaky
    // (`> 1.0`) or already known-false (`>= 1.5`). The measured range
    // to cite is the one recorded in the package notes and in this
    // test's doc comment above, not a single run of this test.
    println!(
        "decode attention: old={old_elapsed:?} new={new_elapsed:?} speedup={speedup:.2}x \
         (seq_len={seq_len}, head_dim={head_dim}, iters={iters}) -- see this test's doc \
         comment: measured outcomes span 0.86x-1.38x, i.e. a wash to modestly faster at \
         this configuration, NOT the withdrawn >= 1.5x; no ratio is asserted here \
         (wall-clock measurement, not a correctness check)"
    );
}
