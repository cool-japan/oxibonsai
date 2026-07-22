//! Integration tests for the PagedAttention KV cache implementation.

use oxibonsai_model::paged_kv_cache::{
    BlockPool, BlockTable, PagedKvCache, PagedKvError, DEFAULT_BLOCK_SIZE,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Build a simple cache: 4 pages, block_size=4, 2 layers, 2 heads, head_dim=4.
fn small_cache() -> PagedKvCache {
    PagedKvCache::new_with_block_size(
        /*capacity=*/ 4, /*block_size=*/ 4, /*num_layers=*/ 2,
        /*num_kv_heads=*/ 2, /*head_dim=*/ 4,
    )
}

/// Return a key/value vec with all elements set to `v` (slot_len = 2*4 = 8).
fn kv_vec(v: f32) -> Vec<f32> {
    vec![v; 8] // num_kv_heads * head_dim
}

// ---------------------------------------------------------------------------
// 1. block_pool_allocate_and_free
// ---------------------------------------------------------------------------

#[test]
fn block_pool_allocate_and_free() {
    let mut pool = BlockPool::new(3, DEFAULT_BLOCK_SIZE, 1, 2, 4);
    let a = pool.allocate().expect("first alloc");
    let b = pool.allocate().expect("second alloc");
    let _c = pool.allocate().expect("third alloc");

    // Pool is now empty.
    assert!(pool.allocate().is_none());

    // All three indices must be distinct.
    assert_ne!(a, b);

    // Free one page and reallocate — should get the same physical index back.
    pool.free(b);
    let d = pool.allocate().expect("realloc after free");
    assert_eq!(d, b); // free list is LIFO
    assert_eq!(pool.free_count(), 0);
}

// ---------------------------------------------------------------------------
// 2. block_pool_utilization
// ---------------------------------------------------------------------------

#[test]
fn block_pool_utilization() {
    let mut pool = BlockPool::new(4, DEFAULT_BLOCK_SIZE, 1, 2, 4);
    assert_eq!(pool.utilization(), 0.0);

    pool.allocate();
    assert!((pool.utilization() - 0.25).abs() < 1e-6);

    pool.allocate();
    assert!((pool.utilization() - 0.50).abs() < 1e-6);

    pool.allocate();
    pool.allocate();
    assert!((pool.utilization() - 1.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// 3. block_pool_oom
// ---------------------------------------------------------------------------

#[test]
fn block_pool_oom() {
    let mut pool = BlockPool::new(2, DEFAULT_BLOCK_SIZE, 1, 2, 4);
    pool.allocate().expect("first");
    pool.allocate().expect("second");
    assert!(pool.allocate().is_none(), "should be OOM");
}

// ---------------------------------------------------------------------------
// 4. block_table_append_and_get
// ---------------------------------------------------------------------------

#[test]
fn block_table_append_and_get() {
    let mut table = BlockTable::new(2, DEFAULT_BLOCK_SIZE);
    table.append_block(0, 42);
    table.append_block(0, 99);
    table.append_block(1, 7);

    assert_eq!(table.get_block(0, 0), Some(42));
    assert_eq!(table.get_block(0, 1), Some(99));
    assert_eq!(table.get_block(1, 0), Some(7));
    assert_eq!(table.get_block(0, 2), None); // out of range
    assert_eq!(table.get_block(3, 0), None); // layer out of range
}

// ---------------------------------------------------------------------------
// 5. paged_cache_create_sequence
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_create_sequence() {
    let mut cache = small_cache();
    let s0 = cache.create_sequence();
    let s1 = cache.create_sequence();
    assert_ne!(s0, s1);
}

// ---------------------------------------------------------------------------
// 6. paged_cache_write_read_kv
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_write_read_kv() {
    let mut cache = small_cache();
    let seq = cache.create_sequence();

    let key = kv_vec(1.0);
    let val = kv_vec(2.0);
    cache.write_kv(seq, 0, 0, &key, &val).expect("write");

    let (k, v) = cache.read_kv(seq, 0, 0).expect("read");
    assert_eq!(k, key.as_slice());
    assert_eq!(v, val.as_slice());
}

// ---------------------------------------------------------------------------
// 7. paged_cache_multi_token_write_read
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_multi_token_write_read() {
    // block_size=4, 2 layers, 2 heads, head_dim=4 → 20 tokens span 5 blocks per layer.
    // Use a larger pool so we don't run out.
    let mut cache = PagedKvCache::new_with_block_size(
        /*capacity=*/ 20, /*block_size=*/ 4, /*num_layers=*/ 2,
        /*num_kv_heads=*/ 2, /*head_dim=*/ 4,
    );
    let seq = cache.create_sequence();

    // Write 20 tokens into layer 0.
    for pos in 0..20_usize {
        let key = vec![pos as f32; 8];
        let val = vec![-(pos as f32); 8];
        cache.write_kv(seq, 0, pos, &key, &val).expect("write");
    }

    // Read all back and verify.
    for pos in 0..20_usize {
        let (k, v) = cache.read_kv(seq, 0, pos).expect("read");
        assert!(
            k.iter().all(|&x| (x - pos as f32).abs() < 1e-6),
            "key mismatch at {pos}"
        );
        assert!(
            v.iter().all(|&x| (x + pos as f32).abs() < 1e-6),
            "val mismatch at {pos}"
        );
    }
}

// ---------------------------------------------------------------------------
// 8. paged_cache_ensure_capacity
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_ensure_capacity() {
    let mut cache = small_cache();
    let seq = cache.create_sequence();

    // Initially no blocks allocated.
    assert_eq!(cache.pool_utilization(), 0.0);

    // Requesting capacity for 5 tokens with block_size=4 needs 2 blocks × 2 layers = 4.
    cache.ensure_capacity(seq, 5).expect("ensure");

    // All 4 pool pages used (4 capacity, 2 layers × 2 blocks).
    assert!((cache.pool_utilization() - 1.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// 9. paged_cache_drop_sequence_frees_blocks
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_drop_sequence_frees_blocks() {
    let mut cache = small_cache();
    let seq = cache.create_sequence();

    cache.ensure_capacity(seq, 4).expect("ensure"); // 1 block per layer = 2 pages
    let util_before = cache.pool_utilization();
    assert!(util_before > 0.0);

    cache.drop_sequence(seq).expect("drop");
    assert_eq!(cache.pool_utilization(), 0.0);
}

// ---------------------------------------------------------------------------
// 10. paged_cache_multi_sequence_isolation
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_multi_sequence_isolation() {
    // 8 pages: enough for two sequences each needing 1 block per layer (2 layers).
    let mut cache = PagedKvCache::new_with_block_size(8, 4, 2, 2, 4);
    let s0 = cache.create_sequence();
    let s1 = cache.create_sequence();

    let key_a = kv_vec(11.0);
    let val_a = kv_vec(22.0);
    let key_b = kv_vec(33.0);
    let val_b = kv_vec(44.0);

    cache.write_kv(s0, 0, 0, &key_a, &val_a).expect("write s0");
    cache.write_kv(s1, 0, 0, &key_b, &val_b).expect("write s1");

    let (k0, v0) = cache.read_kv(s0, 0, 0).expect("read s0");
    let (k1, v1) = cache.read_kv(s1, 0, 0).expect("read s1");

    assert_eq!(k0, key_a.as_slice());
    assert_eq!(v0, val_a.as_slice());
    assert_eq!(k1, key_b.as_slice());
    assert_eq!(v1, val_b.as_slice());
}

// ---------------------------------------------------------------------------
// 11. paged_cache_oom_error
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_oom_error() {
    // 2 pages total, 2 layers → can only hold 1 token (1 block per layer).
    let mut cache = PagedKvCache::new_with_block_size(2, 4, 2, 2, 4);
    let seq = cache.create_sequence();

    // Write token 0 — allocates 1 block per layer = 2 pages.
    cache
        .write_kv(seq, 0, 0, &kv_vec(1.0), &kv_vec(1.0))
        .expect("first write");

    // Now write token 4 — needs a second block per layer, but pool is full.
    let err = cache.write_kv(seq, 0, 4, &kv_vec(2.0), &kv_vec(2.0));
    assert!(matches!(err, Err(PagedKvError::OutOfMemory)));
}

// ---------------------------------------------------------------------------
// 12. paged_cache_sequence_not_found
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_sequence_not_found() {
    let cache = small_cache();
    let err = cache.read_kv(9999, 0, 0);
    assert!(matches!(err, Err(PagedKvError::SequenceNotFound(9999))));
}

// ---------------------------------------------------------------------------
// 13. paged_cache_pool_utilization_tracking
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_pool_utilization_tracking() {
    // 8 pages, block_size=4, 2 layers, 2 heads, head_dim=4.
    let mut cache = PagedKvCache::new_with_block_size(8, 4, 2, 2, 4);
    assert_eq!(cache.pool_utilization(), 0.0);

    let seq = cache.create_sequence();

    // Write token 0 → 2 pages consumed (one per layer).
    cache
        .write_kv(seq, 0, 0, &kv_vec(1.0), &kv_vec(1.0))
        .expect("write t0 l0");
    // Layer 0 block already allocated; layer 1 write also needs allocation.
    cache
        .write_kv(seq, 1, 0, &kv_vec(1.0), &kv_vec(1.0))
        .expect("write t0 l1");

    let util = cache.pool_utilization();
    assert!(util > 0.0, "utilization should be positive: {util}");
}

// ---------------------------------------------------------------------------
// 14. paged_cache_multi_layer
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_multi_layer() {
    let mut cache = PagedKvCache::new_with_block_size(8, 4, 2, 2, 4);
    let seq = cache.create_sequence();

    let k0 = kv_vec(10.0);
    let v0 = kv_vec(20.0);
    let k1 = kv_vec(30.0);
    let v1 = kv_vec(40.0);

    cache.write_kv(seq, 0, 0, &k0, &v0).expect("layer 0 write");
    cache.write_kv(seq, 1, 0, &k1, &v1).expect("layer 1 write");

    let (rk0, rv0) = cache.read_kv(seq, 0, 0).expect("layer 0 read");
    let (rk1, rv1) = cache.read_kv(seq, 1, 0).expect("layer 1 read");

    assert_eq!(rk0, k0.as_slice());
    assert_eq!(rv0, v0.as_slice());
    assert_eq!(rk1, k1.as_slice());
    assert_eq!(rv1, v1.as_slice());
}

// ---------------------------------------------------------------------------
// 15. paged_cache_sequence_length
// ---------------------------------------------------------------------------

#[test]
fn paged_cache_sequence_length() {
    // sequence_length() must report the number of positions actually
    // *written*, not the block-rounded allocated capacity (regression test:
    // it used to return `token_capacity(0)`, e.g. 4 after writing a single
    // token with block_size=4).
    let mut cache = PagedKvCache::new_with_block_size(16, 4, 2, 2, 4);
    let seq = cache.create_sequence();

    assert_eq!(cache.sequence_length(seq), 0);

    // Write token 0 → allocates 1 block (capacity 4) but only 1 position written.
    cache
        .write_kv(seq, 0, 0, &kv_vec(1.0), &kv_vec(1.0))
        .expect("write 0");
    assert_eq!(cache.sequence_length(seq), 1);

    // Write token 4 → crosses into second block (capacity 8) but only 5 positions written.
    cache
        .write_kv(seq, 0, 4, &kv_vec(2.0), &kv_vec(2.0))
        .expect("write 4");
    assert_eq!(cache.sequence_length(seq), 5);
}

#[test]
fn paged_cache_sequence_length_tracks_max_across_out_of_order_writes_and_layers() {
    // sequence_length() must track the highest `token_pos + 1` written, even
    // when writes arrive out of order or on a layer other than layer 0.
    let mut cache = PagedKvCache::new_with_block_size(16, 4, 2, 2, 4);
    let seq = cache.create_sequence();

    // Write position 6 on layer 1 first, then position 2 on layer 0.
    cache
        .write_kv(seq, 1, 6, &kv_vec(3.0), &kv_vec(3.0))
        .expect("write layer 1 pos 6");
    assert_eq!(cache.sequence_length(seq), 7);

    cache
        .write_kv(seq, 0, 2, &kv_vec(4.0), &kv_vec(4.0))
        .expect("write layer 0 pos 2");
    // The later, lower-position write must not shrink the reported length.
    assert_eq!(cache.sequence_length(seq), 7);
}

#[test]
fn paged_cache_sequence_length_unknown_sequence_is_zero() {
    let cache = small_cache();
    assert_eq!(cache.sequence_length(999), 0);
}
