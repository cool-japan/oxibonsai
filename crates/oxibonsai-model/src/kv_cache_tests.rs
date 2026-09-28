//! Unit tests of [`crate::kv_cache`], split out of `kv_cache.rs`
//! (B2-11-FIX). With the legacy `KvCache::store_key` / `store_value`
//! forwarders gone (M-26), the two tests that pin the error-swallowing
//! contract call `store_key_lossy` / `store_value_lossy` by name and every
//! other store goes through `try_store_*` — an in-range store that fails is a
//! test failure, not a silent no-op. The `PagedKvCache` tests keep that type's
//! own `store_key` / `store_value`.

use super::*;

#[test]
fn store_key_out_of_range_pos_does_not_panic_and_leaves_cache_untouched() {
    // Regression test for a release-build panic: previously the only
    // bounds check was `debug_assert!`, which is compiled out in
    // release, so an out-of-range `pos` computed an offset past the
    // end of the pre-allocated `keys` Vec and panicked on the slice
    // index. The lossy stores must now be no-ops instead.
    let mut cache = KvCache::new(1, 1, 4, 8);
    cache.store_key_lossy(0, 0, 100, &[1.0, 2.0, 3.0, 4.0]);
    cache.store_value_lossy(0, 0, 100, &[5.0, 6.0, 7.0, 8.0]);

    // The cache must remain entirely zeroed: nothing was written.
    cache.set_seq_len(8);
    let keys = cache.keys_for(0, 0, 8);
    let values = cache.values_for(0, 0, 8);
    assert!(keys.iter().all(|&x| x == 0.0));
    assert!(values.iter().all(|&x| x == 0.0));
}

#[test]
fn store_key_out_of_range_layer_head_and_bad_len_does_not_panic() {
    let mut cache = KvCache::new(2, 2, 4, 8);
    // Out-of-range layer, out-of-range head, and mismatched slice length
    // must all be silently rejected rather than panicking.
    cache.store_key_lossy(99, 0, 0, &[1.0, 2.0, 3.0, 4.0]);
    cache.store_value_lossy(0, 99, 0, &[1.0, 2.0, 3.0, 4.0]);
    cache.store_key_lossy(0, 0, 0, &[1.0, 2.0]); // wrong length (expects 4)

    let keys = cache.keys_for(0, 0, 1);
    assert!(keys.iter().all(|&x| x == 0.0));
}

#[test]
fn try_store_key_reports_sequence_too_long() {
    let mut cache = KvCache::new(1, 1, 4, 8);
    let err = cache
        .try_store_key(0, 0, 8, &[1.0, 2.0, 3.0, 4.0])
        .expect_err("pos == max_seq_len must be rejected");
    match err {
        ModelError::SequenceTooLong { seq_len, max_ctx } => {
            assert_eq!(seq_len, 9);
            assert_eq!(max_ctx, 8);
        }
        other => panic!("expected SequenceTooLong, got {other:?}"),
    }
}

#[test]
fn try_store_value_reports_shape_mismatch_for_bad_layer_head_and_len() {
    let mut cache = KvCache::new(2, 2, 4, 8);

    let layer_err = cache
        .try_store_value(5, 0, 0, &[0.0; 4])
        .expect_err("out-of-range layer must be rejected");
    assert!(matches!(layer_err, ModelError::ShapeMismatch { .. }));

    let head_err = cache
        .try_store_value(0, 5, 0, &[0.0; 4])
        .expect_err("out-of-range head must be rejected");
    assert!(matches!(head_err, ModelError::ShapeMismatch { .. }));

    let len_err = cache
        .try_store_value(0, 0, 0, &[0.0; 3])
        .expect_err("wrong-length value must be rejected");
    assert!(matches!(len_err, ModelError::ShapeMismatch { .. }));
}

#[test]
fn try_store_key_succeeds_and_is_readable() {
    let mut cache = KvCache::new(1, 1, 4, 8);
    cache
        .try_store_key(0, 0, 2, &[1.0, 2.0, 3.0, 4.0])
        .expect("in-range store must succeed");
    cache.set_seq_len(3);
    let keys = cache.keys_for(0, 0, 3);
    assert_eq!(&keys[8..12], &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn truncate_clamps_down_never_grows() {
    let mut c = KvCache::new(1, 1, 4, 16);
    c.set_seq_len(10);
    c.truncate(6);
    assert_eq!(c.seq_len(), 6);
    // truncate past current len is a no-op
    c.truncate(20);
    assert_eq!(c.seq_len(), 6);
}

#[test]
fn kv_cache_store_and_retrieve() {
    let mut cache = KvCache::new(2, 8, 128, 16);

    let key = vec![1.0f32; 128];
    let value = vec![2.0f32; 128];

    cache
        .try_store_key(0, 0, 0, &key)
        .expect("in-range key store");
    cache
        .try_store_value(0, 0, 0, &value)
        .expect("in-range value store");
    cache.advance();

    let keys = cache.keys_for(0, 0, 1);
    let values = cache.values_for(0, 0, 1);

    assert_eq!(keys.len(), 128);
    assert_eq!(values.len(), 128);
    assert!((keys[0] - 1.0).abs() < 1e-5);
    assert!((values[0] - 2.0).abs() < 1e-5);
}

#[test]
fn kv_cache_multiple_positions() {
    let mut cache = KvCache::new(1, 1, 4, 8);

    cache
        .try_store_key(0, 0, 0, &[1.0, 2.0, 3.0, 4.0])
        .expect("in-range key store");
    cache.advance();
    cache
        .try_store_key(0, 0, 1, &[5.0, 6.0, 7.0, 8.0])
        .expect("in-range key store");
    cache.advance();

    let keys = cache.keys_for(0, 0, 2);
    assert_eq!(keys.len(), 8);
    assert!((keys[0] - 1.0).abs() < 1e-5);
    assert!((keys[4] - 5.0).abs() < 1e-5);
}

#[test]
fn kv_cache_memory_size() {
    let cache = KvCache::new(36, 8, 128, 4096);
    // 36 layers * 8 heads * 4096 seq * 128 dim * 4 bytes * 2 (K+V)
    let expected = 36 * 8 * 4096 * 128 * 4 * 2;
    assert_eq!(cache.memory_bytes(), expected);
}

// `kv_cache_utilization` removed with `KvCache::utilization_ratio`
// itself (M-19) — see the removal note left at its former call site.

#[test]
fn kv_cache_policy_default() {
    let policy = KvCachePolicy::default();
    assert_eq!(policy, KvCachePolicy::Standard);
}

#[test]
fn kv_cache_set_seq_len_clamps_to_max() {
    let mut cache = KvCache::new(1, 1, 4, 8);
    cache.set_seq_len(4);
    assert_eq!(cache.seq_len(), 4);
    cache.set_seq_len(100);
    assert_eq!(cache.seq_len(), 8); // clamped
}

#[test]
fn kv_cache_extract_inject_roundtrip() {
    // Two layers, two KV heads, head_dim=4, block_size=4 → per_layer = 32 floats.
    let num_layers = 2;
    let num_kv_heads = 2;
    let head_dim = 4;
    let block_size = 4;
    let max_seq = 16;
    let mut cache = KvCache::new(num_layers, num_kv_heads, head_dim, max_seq);

    // Populate layer 1 at positions 0..4 with deterministic key/value patterns.
    for head in 0..num_kv_heads {
        for pos in 0..block_size {
            let key: Vec<f32> = (0..head_dim)
                .map(|d| (head as f32 + 1.0) * 100.0 + pos as f32 * 10.0 + d as f32)
                .collect();
            let value: Vec<f32> = (0..head_dim)
                .map(|d| (head as f32 + 1.0) * 1000.0 + pos as f32 * 10.0 + d as f32)
                .collect();
            cache
                .try_store_key(1, head, pos, &key)
                .expect("in-range key store");
            cache
                .try_store_value(1, head, pos, &value)
                .expect("in-range value store");
        }
    }

    // Extract, then inject into a fresh cache and re-extract.
    let (k_block, v_block) = cache.extract_block(1, 0, block_size);
    let per_layer = num_kv_heads * block_size * head_dim;
    assert_eq!(k_block.len(), per_layer);
    assert_eq!(v_block.len(), per_layer);

    let mut fresh = KvCache::new(num_layers, num_kv_heads, head_dim, max_seq);
    fresh.inject_block(1, 0, block_size, &k_block, &v_block);
    fresh.set_seq_len(block_size);

    let (k_block_2, v_block_2) = fresh.extract_block(1, 0, block_size);
    assert_eq!(k_block_2, k_block);
    assert_eq!(v_block_2, v_block);

    // Re-read via keys_for / values_for to verify the position-major layout.
    for head in 0..num_kv_heads {
        let original_keys = cache.keys_for(1, head, block_size);
        let restored_keys = fresh.keys_for(1, head, block_size);
        assert_eq!(
            original_keys, restored_keys,
            "head {head} keys must round-trip"
        );
        let original_values = cache.values_for(1, head, block_size);
        let restored_values = fresh.values_for(1, head, block_size);
        assert_eq!(
            original_values, restored_values,
            "head {head} values must round-trip"
        );
    }
}

#[test]
fn kv_cache_extract_inject_at_offset() {
    // Verify extract/inject behave correctly for non-zero start_pos.
    let mut cache = KvCache::new(1, 1, 2, 16);
    // Write a recognisable pattern at positions 4..8.
    for pos in 0..4 {
        let key = vec![pos as f32, pos as f32 + 0.5];
        let value = vec![-(pos as f32), -(pos as f32) - 0.5];
        cache
            .try_store_key(0, 0, 4 + pos, &key)
            .expect("in-range key store");
        cache
            .try_store_value(0, 0, 4 + pos, &value)
            .expect("in-range value store");
    }
    let (k, v) = cache.extract_block(0, 4, 4);
    let mut other = KvCache::new(1, 1, 2, 16);
    other.inject_block(0, 4, 4, &k, &v);
    for pos in 0..4 {
        let original_k = cache.keys_for(0, 0, 8);
        let restored_k = other.keys_for(0, 0, 8);
        // positions 0..4 are zeros in both; positions 4..8 must match.
        let off = (4 + pos) * 2;
        assert!((restored_k[off] - original_k[off]).abs() < 1e-6);
        assert!((restored_k[off + 1] - original_k[off + 1]).abs() < 1e-6);
    }
}

// ── Paged KV Cache tests ──

#[test]
fn paged_kv_cache_store_and_retrieve() {
    let mut cache = PagedKvCache::with_page_size(2, 1, 4, 16, 4);

    let key = vec![1.0, 2.0, 3.0, 4.0];
    let value = vec![5.0, 6.0, 7.0, 8.0];

    cache.store_key(0, 0, 0, &key);
    cache.store_value(0, 0, 0, &value);
    cache.advance();

    let keys = cache.keys_for(0, 0, 1);
    let values = cache.values_for(0, 0, 1);

    assert_eq!(keys.len(), 4);
    assert_eq!(values.len(), 4);
    assert!((keys[0] - 1.0).abs() < 1e-5);
    assert!((values[0] - 5.0).abs() < 1e-5);
}

#[test]
fn paged_kv_cache_cross_page_boundary() {
    let mut cache = PagedKvCache::with_page_size(1, 1, 4, 16, 2);

    // Store in page 0 (positions 0, 1)
    cache.store_key(0, 0, 0, &[1.0, 2.0, 3.0, 4.0]);
    cache.store_key(0, 0, 1, &[5.0, 6.0, 7.0, 8.0]);
    // Store in page 1 (positions 2, 3)
    cache.store_key(0, 0, 2, &[9.0, 10.0, 11.0, 12.0]);

    let keys = cache.keys_for(0, 0, 3);
    assert_eq!(keys.len(), 12);
    assert!((keys[0] - 1.0).abs() < 1e-5);
    assert!((keys[4] - 5.0).abs() < 1e-5);
    assert!((keys[8] - 9.0).abs() < 1e-5);
}

#[test]
fn paged_kv_cache_lazy_allocation() {
    let cache = PagedKvCache::with_page_size(1, 1, 4, 1024, 256);
    assert_eq!(cache.total_pages(), 0);
    assert_eq!(cache.memory_usage_bytes(), 0);
}

#[test]
fn paged_kv_cache_memory_grows() {
    let mut cache = PagedKvCache::with_page_size(1, 1, 4, 1024, 4);

    assert_eq!(cache.memory_usage_bytes(), 0);

    cache.store_key(0, 0, 0, &[1.0; 4]);
    // 1 page allocated: 4 positions * 4 dims * 4 bytes * 2 (K+V)
    let one_page_bytes = 4 * 4 * 4 * 2;
    assert_eq!(cache.memory_usage_bytes(), one_page_bytes);

    // Trigger second page allocation
    cache.store_key(0, 0, 4, &[1.0; 4]);
    assert_eq!(cache.memory_usage_bytes(), one_page_bytes * 2);
}

#[test]
fn paged_kv_cache_clear() {
    let mut cache = PagedKvCache::with_page_size(1, 1, 4, 16, 4);
    cache.store_key(0, 0, 0, &[1.0; 4]);
    cache.advance();

    assert!(cache.total_pages() > 0);
    cache.clear();
    assert_eq!(cache.total_pages(), 0);
    assert_eq!(cache.seq_len(), 0);
}

// `paged_kv_cache_utilization` removed with `PagedKvCache::utilization_ratio`
// itself (M-19) — see the removal note left at its former call site.

// ── M-07: sparse (f16) KV cache ──────────────────────────────────────

/// THE gate: `new_sparse(16, ...)` for the Bonsai 2 27B geometry
/// (16 full-attention slots x 4 KV heads x 256 head_dim) allocates
/// exactly 64 KiB/token = 65_536 bytes when `max_seq == 1`.
#[test]
fn new_sparse_bonsai2_27b_geometry_is_exactly_64_kib_per_token() {
    let cache = KvCache::new_sparse(16, 4, 256, 1);
    assert_eq!(
        cache.memory_bytes(),
        65_536,
        "16 slots x 4 kv heads x 256 head_dim x 2 (K+V) x 2 bytes (f16) must be 65536"
    );
    // And the per-token figure scales linearly with context length.
    let cache_2k = KvCache::new_sparse(16, 4, 256, 2048);
    assert_eq!(cache_2k.memory_bytes(), 65_536 * 2048);
}

#[test]
fn new_sparse_reports_slot_count_as_num_layers_and_is_sparse() {
    let cache = KvCache::new_sparse(16, 4, 256, 4096);
    assert_eq!(cache.num_layers(), 16);
    assert_eq!(cache.num_kv_heads(), 4);
    assert_eq!(cache.head_dim(), 256);
    assert_eq!(cache.max_seq_len(), 4096);
    assert!(cache.is_sparse());

    let dense = KvCache::new(64, 4, 256, 4096);
    assert!(!dense.is_sparse());
}

/// A dense cache with the *same* nominal geometry is exactly 2x the
/// bytes of a sparse one with the same layer/head/dim/seq numbers —
/// isolating the f16-vs-f32 half of the M-07 win from the
/// layer-count-reduction half (already covered by comparing 16 vs 64
/// layers in the acceptance test above).
#[test]
fn sparse_is_exactly_half_the_bytes_of_dense_at_equal_geometry() {
    let dense = KvCache::new(16, 4, 256, 100);
    let sparse = KvCache::new_sparse(16, 4, 256, 100);
    assert_eq!(sparse.memory_bytes() * 2, dense.memory_bytes());
}

#[test]
fn try_new_sparse_succeeds_for_reasonable_geometry() {
    let cache =
        KvCache::try_new_sparse(16, 4, 256, 8192).expect("reasonable sparse alloc must succeed");
    assert_eq!(cache.memory_bytes(), 65_536 * 8192);
}

#[test]
fn try_new_and_try_new_sparse_reject_overflowing_geometry_instead_of_panicking() {
    let err = KvCache::try_new(usize::MAX, usize::MAX, usize::MAX, usize::MAX)
        .expect_err("an overflowing element count must be rejected, not panic");
    assert!(matches!(err, ModelError::KvAllocation { .. }), "{err}");

    let err = KvCache::try_new_sparse(usize::MAX, usize::MAX, usize::MAX, usize::MAX)
        .expect_err("an overflowing sparse element count must be rejected, not panic");
    assert!(matches!(err, ModelError::KvAllocation { .. }), "{err}");

    // A count that does not overflow `usize` but whose *byte* total does
    // (elements * 2 (K+V) * size_of::<f32>()) must also be rejected
    // rather than reaching the allocation itself.
    let huge_but_not_overflowing = usize::MAX / 4;
    let err = KvCache::try_new(huge_but_not_overflowing, 1, 1, 1)
        .expect_err("a byte count past isize::MAX must be rejected");
    assert!(matches!(err, ModelError::KvAllocation { .. }), "{err}");
}

#[test]
fn try_new_accepts_the_same_geometry_new_would_build() {
    let cache = KvCache::try_new(2, 4, 64, 128).expect("small geometry must succeed");
    assert_eq!(cache.num_layers(), 2);
    assert_eq!(
        cache.memory_bytes(),
        KvCache::new(2, 4, 64, 128).memory_bytes()
    );
}

/// Supersedes the pre-B2-11-FIX contract ("`keys_for` on an `f16` cache
/// returns an EMPTY slice, not wrong data"): that empty slice was the
/// release-build-silent footgun the wave-3.5 triage flagged, and with `f16`
/// now the host default it would have handed every legacy reader an empty
/// history. `keys_for`/`values_for` now return the real rows (from the
/// `f32` read-back mirror — never garbage, never empty) for every storage
/// mode, and the mirror follows every write.
#[test]
fn sparse_keys_for_and_values_for_return_the_real_dequantized_rows() {
    let mut cache = KvCache::new_sparse(1, 1, 4, 8);
    cache
        .try_store_key(0, 0, 0, &[1.0, 2.0, 3.0, 4.0])
        .expect("store must succeed on a sparse cache");
    cache
        .try_store_value(0, 0, 0, &[-1.0, 0.5, 0.25, 8.0])
        .expect("store must succeed on a sparse cache");
    cache.set_seq_len(1);
    assert_eq!(
        cache.keys_for(0, 0, 1),
        &[1.0f32, 2.0, 3.0, 4.0],
        "keys_for must serve f16 storage with the real (exactly representable) rows"
    );
    assert_eq!(cache.values_for(0, 0, 1), &[-1.0f32, 0.5, 0.25, 8.0]);
    // A later write is visible to the next read (the mirror is rebuilt).
    cache
        .try_store_key(0, 0, 1, &[5.0, 6.0, 7.0, 8.0])
        .expect("second store");
    assert_eq!(
        cache.keys_for(0, 0, 2),
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    // A request past the allocation is clamped to it, never fabricated.
    assert_eq!(cache.keys_for(0, 0, 9).len(), 8 * 4);
}

#[test]
fn sparse_keys_for_owned_and_values_for_owned_roundtrip() {
    let mut cache = KvCache::new_sparse(2, 2, 4, 8);
    let key = vec![1.0f32, -2.5, 3.25, 0.125];
    let value = vec![-0.5f32, 6.0, -7.0, 8.5];
    cache
        .try_store_key(1, 1, 3, &key)
        .expect("sparse store_key must succeed");
    cache
        .try_store_value(1, 1, 3, &value)
        .expect("sparse store_value must succeed");
    cache.set_seq_len(4);

    let read_key = cache.keys_for_owned(1, 1, 4);
    let read_value = cache.values_for_owned(1, 1, 4);
    assert_eq!(read_key.len(), 16);
    assert_eq!(read_value.len(), 16);
    // f16 round-trip: exact for these values (all representable in f16).
    for (orig, got) in key.iter().zip(&read_key[12..16]) {
        assert!((orig - got).abs() < 1e-3, "orig={orig} got={got}");
    }
    for (orig, got) in value.iter().zip(&read_value[12..16]) {
        assert!((orig - got).abs() < 1e-3, "orig={orig} got={got}");
    }
    // Untouched positions/heads must read back as zero.
    assert!(read_key[0..12].iter().all(|&x| x == 0.0));
}

#[test]
fn dense_keys_for_owned_matches_keys_for() {
    let mut cache = KvCache::new(1, 1, 4, 8);
    cache
        .try_store_key(0, 0, 2, &[1.0, 2.0, 3.0, 4.0])
        .expect("in-range key store");
    cache.set_seq_len(3);
    assert_eq!(
        cache.keys_for(0, 0, 3),
        cache.keys_for_owned(0, 0, 3).as_slice()
    );
}

#[test]
fn sparse_extract_inject_roundtrip() {
    let num_layers = 2;
    let num_kv_heads = 2;
    let head_dim = 4;
    let block_size = 4;
    let max_seq = 16;
    let mut cache = KvCache::new_sparse(num_layers, num_kv_heads, head_dim, max_seq);
    assert!(cache.is_sparse());

    for head in 0..num_kv_heads {
        for pos in 0..block_size {
            let key: Vec<f32> = (0..head_dim)
                .map(|d| (head as f32 + 1.0) * 10.0 + pos as f32 + d as f32 * 0.25)
                .collect();
            let value: Vec<f32> = (0..head_dim)
                .map(|d| -((head as f32 + 1.0) * 10.0 + pos as f32 + d as f32 * 0.25))
                .collect();
            cache
                .try_store_key(1, head, pos, &key)
                .expect("sparse try_store_key");
            cache
                .try_store_value(1, head, pos, &value)
                .expect("sparse try_store_value");
        }
    }

    let (k_block, v_block) = cache.extract_block(1, 0, block_size);
    let mut fresh = KvCache::new_sparse(num_layers, num_kv_heads, head_dim, max_seq);
    fresh.inject_block(1, 0, block_size, &k_block, &v_block);
    fresh.set_seq_len(block_size);

    for head in 0..num_kv_heads {
        let original = cache.keys_for_owned(1, head, block_size);
        let restored = fresh.keys_for_owned(1, head, block_size);
        for (o, r) in original.iter().zip(&restored) {
            assert!((o - r).abs() < 1e-2, "head {head}: orig={o} restored={r}");
        }
    }
}

#[test]
fn kv_cache_backing_bytes_per_element_and_is_sparse() {
    assert_eq!(KvCacheBacking::DenseF32.bytes_per_element(), 4);
    assert_eq!(KvCacheBacking::DenseF16.bytes_per_element(), 2);
    assert_eq!(KvCacheBacking::DenseQ8.bytes_per_element(), 1);
    assert_eq!(KvCacheBacking::DenseFp8.bytes_per_element(), 1);
    assert_eq!(KvCacheBacking::DenseQ4.bytes_per_element(), 1);
    assert_eq!(KvCacheBacking::SparseF16.bytes_per_element(), 2);

    assert!(!KvCacheBacking::DenseF32.is_sparse());
    assert!(!KvCacheBacking::DenseF16.is_sparse());
    assert!(!KvCacheBacking::DenseQ8.is_sparse());
    assert!(!KvCacheBacking::DenseFp8.is_sparse());
    assert!(!KvCacheBacking::DenseQ4.is_sparse());
    assert!(KvCacheBacking::SparseF16.is_sparse());
}

#[test]
fn kv_cache_backing_variants_are_distinct() {
    // `#[non_exhaustive]` + `PartialEq, Eq, Hash` must actually
    // distinguish every variant (regression guard against a copy-paste
    // derive that quietly makes two variants compare equal).
    use std::collections::HashSet;
    let all = [
        KvCacheBacking::DenseF32,
        KvCacheBacking::DenseF16,
        KvCacheBacking::DenseQ8,
        KvCacheBacking::DenseFp8,
        KvCacheBacking::DenseQ4,
        KvCacheBacking::SparseF16,
    ];
    let unique: HashSet<_> = all.iter().copied().collect();
    assert_eq!(unique.len(), all.len());
}

// ── M-07: lazy growth in chunks ─────────────────────────────────────────

#[test]
fn ensure_capacity_rounds_up_to_the_next_growth_chunk() {
    let mut cache = KvCache::new(1, 1, 4, 10);
    assert_eq!(cache.max_seq_len(), 10);
    cache.ensure_capacity(10); // already satisfied: no-op
    assert_eq!(cache.max_seq_len(), 10);

    cache.ensure_capacity(11); // one past capacity: grow by a full chunk
    assert_eq!(cache.max_seq_len(), GROWTH_CHUNK_POSITIONS);

    cache.ensure_capacity(GROWTH_CHUNK_POSITIONS + 1);
    assert_eq!(cache.max_seq_len(), GROWTH_CHUNK_POSITIONS * 2);
}

#[test]
fn ensure_capacity_never_shrinks() {
    let mut cache = KvCache::new(1, 1, 4, GROWTH_CHUNK_POSITIONS * 4);
    let before = cache.max_seq_len();
    cache.ensure_capacity(1);
    assert_eq!(cache.max_seq_len(), before, "must never shrink");
}

#[test]
fn ensure_capacity_doubles_instead_of_creeping_by_one_chunk_once_the_cache_is_large() {
    // Perf minor (wave-3 review): once a cache is already much larger
    // than one growth chunk, growing by exactly one
    // `GROWTH_CHUNK_POSITIONS` chunk at a time makes every
    // `try_grow_to` re-copy roughly the cache's whole existing content
    // again, for O(n^2) total bytes moved across a long decode loop
    // (the review measured ~696 reallocs averaging ~89K positions at a
    // 178K context). Past that point growth must double the existing
    // capacity instead of creeping forward by one chunk.
    let mut cache = KvCache::new(1, 1, 1, 100_000);
    cache.ensure_capacity(100_001);
    assert_eq!(
        cache.max_seq_len(),
        200_000,
        "once max_seq_len dominates one growth chunk, growth must double it \
         rather than creep forward by GROWTH_CHUNK_POSITIONS"
    );
}

#[test]
fn grow_to_preserves_dense_data_and_seq_len() {
    // Distinct values for every (layer, head, pos, dim) combination, not
    // just one (layer, head) pair, so a base-offset mistake in the
    // direct-copy growth path (wrong layer/head stride, or mixing up the
    // old vs. new `max_seq_len` in the offset formula) shows up as a
    // mismatch instead of coincidentally reading back correct data.
    let (num_layers, num_kv_heads, head_dim) = (2, 2, 4);
    let value_at = |layer: usize, head: usize, pos: usize, d: usize| -> f32 {
        (layer * 1000 + head * 100 + pos * 10 + d) as f32
    };
    let mut cache = KvCache::new(num_layers, num_kv_heads, head_dim, 8);
    for layer in 0..num_layers {
        for head in 0..num_kv_heads {
            for pos in 0..6 {
                let key: Vec<f32> = (0..head_dim)
                    .map(|d| value_at(layer, head, pos, d))
                    .collect();
                let value: Vec<f32> = (0..head_dim)
                    .map(|d| -value_at(layer, head, pos, d))
                    .collect();
                cache
                    .try_store_key(layer, head, pos, &key)
                    .expect("in-range key store");
                cache
                    .try_store_value(layer, head, pos, &value)
                    .expect("in-range value store");
            }
        }
    }
    cache.set_seq_len(6);

    cache.grow_to(1024);
    assert_eq!(cache.max_seq_len(), 1024);
    assert_eq!(cache.seq_len(), 6, "grow_to must not change seq_len");
    assert!(!cache.is_sparse());

    for layer in 0..num_layers {
        for head in 0..num_kv_heads {
            let keys = cache.keys_for(layer, head, 6);
            let values = cache.values_for(layer, head, 6);
            for pos in 0..6 {
                let expected: Vec<f32> = (0..head_dim)
                    .map(|d| value_at(layer, head, pos, d))
                    .collect();
                assert_eq!(
                    &keys[pos * head_dim..pos * head_dim + head_dim],
                    expected.as_slice(),
                    "layer {layer} head {head} pos {pos} keys must survive growth"
                );
                let expected_v: Vec<f32> = (0..head_dim)
                    .map(|d| -value_at(layer, head, pos, d))
                    .collect();
                assert_eq!(
                    &values[pos * head_dim..pos * head_dim + head_dim],
                    expected_v.as_slice(),
                    "layer {layer} head {head} pos {pos} values must survive growth"
                );
            }
        }
    }
    // A position beyond what was written must still read zero.
    assert!(cache.extract_block(0, 6, 1).0.iter().all(|&x| x == 0.0));
}

#[test]
fn grow_to_preserves_sparse_data_and_stays_sparse() {
    // Same multi-(layer, head) regression guard as the dense version
    // above, for the f16-backed copy path.
    let (num_layers, num_kv_heads, head_dim) = (2, 2, 4);
    let value_at = |layer: usize, head: usize, pos: usize, d: usize| -> f32 {
        (layer * 100 + head * 10 + pos) as f32 + d as f32 * 0.25
    };
    let mut cache = KvCache::new_sparse(num_layers, num_kv_heads, head_dim, 8);
    for layer in 0..num_layers {
        for head in 0..num_kv_heads {
            for pos in 0..4 {
                let key: Vec<f32> = (0..head_dim)
                    .map(|d| value_at(layer, head, pos, d))
                    .collect();
                let value: Vec<f32> = (0..head_dim)
                    .map(|d| -value_at(layer, head, pos, d))
                    .collect();
                cache
                    .try_store_key(layer, head, pos, &key)
                    .expect("sparse store must succeed");
                cache
                    .try_store_value(layer, head, pos, &value)
                    .expect("sparse store must succeed");
            }
        }
    }
    cache.set_seq_len(4);

    cache.grow_to(512);
    assert!(cache.is_sparse(), "grow_to must preserve the storage mode");
    assert_eq!(cache.max_seq_len(), 512);
    assert_eq!(
        cache.memory_bytes(),
        num_layers * num_kv_heads * head_dim * 512 * 2 * 2,
        "grown sparse cache must still cost 2 bytes/element, not silently become dense"
    );

    for layer in 0..num_layers {
        for head in 0..num_kv_heads {
            let read_key = cache.keys_for_owned(layer, head, 4);
            let read_value = cache.values_for_owned(layer, head, 4);
            for pos in 0..4 {
                for d in 0..head_dim {
                    let expected = value_at(layer, head, pos, d);
                    let got_k = read_key[pos * head_dim + d];
                    assert!(
                        (expected - got_k).abs() < 1e-2,
                        "layer {layer} head {head} pos {pos} d {d}: expected={expected} got={got_k}"
                    );
                    let got_v = read_value[pos * head_dim + d];
                    assert!(
                        (-expected - got_v).abs() < 1e-2,
                        "layer {layer} head {head} pos {pos} d {d} (value): expected={} got={got_v}",
                        -expected
                    );
                }
            }
        }
    }
}

#[test]
fn grow_to_is_a_no_op_when_already_large_enough() {
    let mut cache = KvCache::new(1, 1, 4, 100);
    cache
        .try_store_key(0, 0, 0, &[1.0, 2.0, 3.0, 4.0])
        .expect("in-range key store");
    cache.grow_to(50); // smaller than current — must not shrink or touch data
    assert_eq!(cache.max_seq_len(), 100);
    cache.set_seq_len(1);
    assert_eq!(cache.keys_for(0, 0, 1), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn try_ensure_capacity_and_try_grow_to_reject_overflow_leaving_cache_unchanged() {
    let mut cache = KvCache::new(1, 1, 4, 8);
    cache
        .try_store_key(0, 0, 0, &[1.0, 2.0, 3.0, 4.0])
        .expect("in-range key store");

    let err = cache
        .try_grow_to(usize::MAX)
        .expect_err("an overflowing grow target must be rejected");
    assert!(matches!(err, ModelError::KvAllocation { .. }), "{err}");
    // The cache must be completely unchanged after a rejected grow.
    assert_eq!(cache.max_seq_len(), 8);
    cache.set_seq_len(1);
    assert_eq!(cache.keys_for(0, 0, 1), &[1.0, 2.0, 3.0, 4.0]);

    let err = cache
        .try_ensure_capacity(usize::MAX)
        .expect_err("an overflowing ensure_capacity target must be rejected");
    assert!(matches!(err, ModelError::KvAllocation { .. }), "{err}");
    assert_eq!(cache.max_seq_len(), 8);
}
