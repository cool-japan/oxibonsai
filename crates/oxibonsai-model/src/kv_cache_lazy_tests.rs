//! Tests of the B2-11-FIX additions to [`crate::kv_cache`]: bounded lazy
//! allocation, the element-type / layout split, the read-back accessors and the
//! storage-aware attention read.

use super::*;
use crate::layers::attention_fused::fused_attention_head_contiguous;

/// Deterministic xorshift64* stream in `[-1, 1)`.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
        (bits as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

/// A row whose every element encodes its `(layer, head, pos, d)` (exactly
/// representable in both `f32` and `f16` for the small ranges used here).
fn tagged_row(layer: usize, head: usize, pos: usize, head_dim: usize) -> Vec<f32> {
    (0..head_dim)
        .map(|d| (layer * 7 + head * 3) as f32 + (pos % 512) as f32 * 0.5 + d as f32 * 0.25)
        .collect()
}

// ── capacity model ──────────────────────────────────────────────────────────

#[test]
fn lazy_cache_reports_a_fixed_limit_and_a_small_allocation() {
    let cache =
        KvCache::try_new_lazy(KvCacheBacking::DenseF16, 2, 2, 8, 1000, 16).expect("lazy cache");
    assert_eq!(cache.max_seq_len(), 1000, "the logical limit");
    assert_eq!(cache.allocated_seq_len(), 16, "only the first allocation");
    assert!(cache.is_lazy());
    assert!(cache.is_f16());
    assert!(!cache.is_sparse(), "DenseF16 is dense, not layer-sparse");
    assert_eq!(cache.backing(), KvCacheBacking::DenseF16);
    assert_eq!(cache.memory_bytes(), 2 * 2 * 16 * 8 * 2 * 2);
    assert_eq!(cache.max_memory_bytes(), 2 * 2 * 1000 * 8 * 2 * 2);
}

#[test]
fn backing_reports_element_type_and_layout_independently() {
    assert_eq!(KvCache::new(2, 1, 4, 8).backing(), KvCacheBacking::DenseF32);
    let sparse = KvCache::new_sparse(2, 1, 4, 8);
    assert_eq!(sparse.backing(), KvCacheBacking::SparseF16);
    assert!(sparse.is_sparse() && sparse.is_f16() && !sparse.is_lazy());
    let lazy_sparse =
        KvCache::try_new_lazy(KvCacheBacking::SparseF16, 16, 4, 256, 8192, 256).expect("lazy");
    assert!(lazy_sparse.is_sparse() && lazy_sparse.is_f16());
    // The Bonsai 2 27B figure, per allocated token.
    assert_eq!(lazy_sparse.memory_bytes(), 65_536 * 256);
    let lazy_f32 =
        KvCache::try_new_lazy(KvCacheBacking::DenseF32, 2, 1, 4, 64, 8).expect("lazy f32");
    assert!(!lazy_f32.is_f16() && !lazy_f32.is_sparse());
}

#[test]
fn try_new_lazy_rejects_the_quantized_backings() {
    for backing in [
        KvCacheBacking::DenseQ8,
        KvCacheBacking::DenseFp8,
        KvCacheBacking::DenseQ4,
    ] {
        let err = KvCache::try_new_lazy(backing, 1, 1, 4, 16, 4)
            .expect_err("quantized tiers live in kv_cache_quant");
        assert!(matches!(err, ModelError::ShapeInvariant { .. }), "{err}");
    }
    let err = KvCache::try_new_lazy(KvCacheBacking::DenseF16, usize::MAX, 4, 4, 4, 1)
        .expect_err("an overflowing limit must be refused up front");
    assert!(matches!(err, ModelError::KvAllocation { .. }), "{err}");
}

#[test]
fn lazy_growth_across_a_chunk_boundary_preserves_every_position() {
    for backing in [KvCacheBacking::DenseF32, KvCacheBacking::DenseF16] {
        let (layers, heads, hd) = (2usize, 2usize, 8usize);
        let mut cache =
            KvCache::try_new_lazy(backing, layers, heads, hd, 4096, 256).expect("lazy cache");
        let last = GROWTH_CHUNK_POSITIONS + 44; // crosses the first chunk boundary
        for pos in 0..=last {
            for layer in 0..layers {
                for head in 0..heads {
                    let row = tagged_row(layer, head, pos, hd);
                    cache
                        .try_store_key(layer, head, pos, &row)
                        .expect("store within the limit");
                    let neg: Vec<f32> = row.iter().map(|x| -x).collect();
                    cache
                        .try_store_value(layer, head, pos, &neg)
                        .expect("store within the limit");
                }
            }
            cache.set_seq_len(pos + 1);
        }
        assert!(cache.allocated_seq_len() > GROWTH_CHUNK_POSITIONS);
        assert!(cache.allocated_seq_len() <= cache.max_seq_len());
        assert_eq!(cache.max_seq_len(), 4096, "the limit never moves");
        for layer in 0..layers {
            for head in 0..heads {
                let keys = cache.keys_for(layer, head, last + 1);
                let values = cache.values_for(layer, head, last + 1);
                for pos in 0..=last {
                    let want = tagged_row(layer, head, pos, hd);
                    assert_eq!(
                        &keys[pos * hd..(pos + 1) * hd],
                        want.as_slice(),
                        "{backing:?} layer {layer} head {head} pos {pos}: key survives growth"
                    );
                    let want_v: Vec<f32> = want.iter().map(|x| -x).collect();
                    assert_eq!(
                        &values[pos * hd..(pos + 1) * hd],
                        want_v.as_slice(),
                        "{backing:?} layer {layer} head {head} pos {pos}: value survives growth"
                    );
                }
            }
        }
    }
}

#[test]
fn growth_mid_chunk_keeps_rows_the_cursor_has_not_reached_yet() {
    // A batched prefill stores a whole chunk before it advances the cursor.
    // A growth triggered by the chunk's tail must copy every allocated
    // position, not only `0..seq_len` (which is still 0 here).
    let mut cache =
        KvCache::try_new_lazy(KvCacheBacking::DenseF16, 1, 1, 4, 2048, 64).expect("lazy");
    for pos in 0..64 {
        cache
            .try_store_key(0, 0, pos, &tagged_row(0, 0, pos, 4))
            .expect("store");
    }
    assert_eq!(cache.seq_len(), 0);
    cache
        .try_store_key(0, 0, 900, &tagged_row(0, 0, 900, 4))
        .expect("store past the allocation grows it");
    assert!(cache.allocated_seq_len() > 900);
    let keys = cache.keys_for(0, 0, 901);
    for pos in (0..64).chain(std::iter::once(900)) {
        assert_eq!(
            &keys[pos * 4..(pos + 1) * 4],
            tagged_row(0, 0, pos, 4).as_slice(),
            "pos {pos}"
        );
    }
    assert!(keys[64 * 4..900 * 4].iter().all(|&x| x == 0.0));
}

#[test]
fn a_lazy_cache_refuses_positions_past_its_limit_and_stays_unchanged() {
    let mut cache =
        KvCache::try_new_lazy(KvCacheBacking::DenseF32, 1, 1, 4, 300, 256).expect("lazy");
    let err = cache
        .try_store_key(0, 0, 300, &[1.0; 4])
        .expect_err("pos == limit");
    assert!(matches!(
        err,
        ModelError::SequenceTooLong {
            seq_len: 301,
            max_ctx: 300
        }
    ));
    let err = cache
        .try_ensure_capacity(301)
        .expect_err("capacity past the limit");
    assert!(matches!(err, ModelError::SequenceTooLong { .. }));
    let err = cache.try_grow_to(1000).expect_err("grow past the limit");
    assert!(matches!(err, ModelError::SequenceTooLong { .. }));
    assert_eq!(
        cache.allocated_seq_len(),
        256,
        "a refused growth changes nothing"
    );
    // Growth inside the limit is clamped to it (not rounded to 512).
    cache.try_ensure_capacity(257).expect("inside the limit");
    assert_eq!(cache.allocated_seq_len(), 300);
    cache
        .try_store_key(0, 0, 299, &[1.0; 4])
        .expect("last position");
}

#[test]
fn clear_keeps_the_allocation_of_a_lazy_cache() {
    let mut cache =
        KvCache::try_new_lazy(KvCacheBacking::DenseF16, 1, 1, 4, 4096, 16).expect("lazy");
    cache.try_ensure_capacity(1000).expect("grow");
    let grown = cache.allocated_seq_len();
    cache.set_seq_len(1000);
    cache.clear();
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.allocated_seq_len(), grown);
}

#[test]
fn inject_block_grows_a_lazy_cache_to_cover_the_block() {
    let (heads, hd, block) = (2usize, 4usize, 16usize);
    let mut source = KvCache::new(1, heads, hd, 1024);
    for head in 0..heads {
        for pos in 600..600 + block {
            source
                .try_store_key(0, head, pos, &tagged_row(0, head, pos, hd))
                .expect("store");
            source
                .try_store_value(0, head, pos, &tagged_row(1, head, pos, hd))
                .expect("store");
        }
    }
    let (k, v) = source.extract_block(0, 600, block);
    let mut lazy =
        KvCache::try_new_lazy(KvCacheBacking::DenseF16, 1, heads, hd, 1024, 64).expect("lazy");
    lazy.try_inject_block(0, 600, block, &k, &v)
        .expect("injection grows the cache");
    assert!(lazy.allocated_seq_len() >= 600 + block);
    let (k2, v2) = lazy.extract_block(0, 600, block);
    assert_eq!(k2, k, "tagged rows are exactly representable in f16");
    assert_eq!(v2, v);
    // A wrongly sized block is refused, not sliced out of bounds.
    let err = lazy
        .try_inject_block(0, 0, block, &k[..4], &v)
        .expect_err("short block");
    assert!(matches!(err, ModelError::ShapeMismatch { .. }));
    let err = lazy
        .try_inject_block(9, 0, block, &k, &v)
        .expect_err("layer out of range");
    assert!(matches!(err, ModelError::ShapeMismatch { .. }));
}

// ── read-back accessors ─────────────────────────────────────────────────────

#[test]
fn keys_for_serves_both_element_types_and_follows_writes_and_growth() {
    let mut dense = KvCache::new(1, 1, 4, 8);
    dense
        .try_store_key(0, 0, 0, &[1.0, 2.0, 3.0, 4.0])
        .expect("store");
    assert_eq!(dense.keys_for(0, 0, 1), &[1.0f32, 2.0, 3.0, 4.0]);
    assert_eq!(dense.keys_for_owned(0, 0, 1), vec![1.0f32, 2.0, 3.0, 4.0]);

    let mut half =
        KvCache::try_new_lazy(KvCacheBacking::DenseF16, 1, 1, 4, 64, 8).expect("lazy f16");
    half.try_store_key(0, 0, 0, &[1.0, 2.0, 3.0, 4.0])
        .expect("store");
    assert_eq!(half.keys_for(0, 0, 1), &[1.0f32, 2.0, 3.0, 4.0]);
    // A store that grows the allocation invalidates the mirror too.
    half.try_store_key(0, 0, 20, &[9.0, 9.5, 10.0, 10.5])
        .expect("store past the first allocation");
    assert!(half.allocated_seq_len() > 20);
    let keys = half.keys_for(0, 0, 21);
    assert_eq!(&keys[..4], &[1.0f32, 2.0, 3.0, 4.0]);
    assert_eq!(&keys[80..84], &[9.0f32, 9.5, 10.0, 10.5]);
    assert_eq!(half.keys_for_owned(0, 0, 21), keys.to_vec());
    // Out-of-range coordinates yield empty rows, never a panic.
    assert!(half.keys_for(5, 0, 1).is_empty());
    assert!(half.values_for(0, 5, 1).is_empty());
}

// ── storage-aware attention ─────────────────────────────────────────────────

/// Reference: widen the whole history, then run the contiguous kernel per
/// head — what `attend_f16_group` must reproduce bit for bit.
fn reference_group(
    keys: &[f16],
    values: &[f16],
    seq_len: usize,
    head_dim: usize,
    queries: &[f32],
) -> Vec<f32> {
    let k: Vec<f32> = keys.iter().map(|h| h.to_f32()).collect();
    let v: Vec<f32> = values.iter().map(|h| h.to_f32()).collect();
    let mut out = vec![0.0f32; queries.len()];
    for (q, o) in queries
        .chunks_exact(head_dim)
        .zip(out.chunks_exact_mut(head_dim))
    {
        fused_attention_head_contiguous(q, &k, &v, o, seq_len, head_dim).expect("reference");
    }
    out
}

#[test]
fn f16_group_attention_is_bit_identical_to_dequantize_then_contiguous() {
    let mut rng = Rng(0x5EED_F16A_77E1_0001);
    // seq_len values straddle the 32-position block size; the growing key
    // scale makes the running max change across blocks, so the rescale path
    // is exercised; head_dim 256 is Bonsai 2's; groups 1..=6 cover every
    // shipped GQA ratio; 17 and 300 take the heap fallbacks.
    for &(seq_len, head_dim, group) in &[
        (1usize, 64usize, 1usize),
        (31, 64, 2),
        (32, 128, 4),
        (33, 128, 4),
        (77, 256, 6),
        (160, 64, 3),
        (45, 64, 17),
        (40, 300, 2),
    ] {
        let keys: Vec<f16> = (0..seq_len * head_dim)
            .map(|i| f16::from_f32(rng.next_f32() * (1.0 + (i / head_dim) as f32 * 0.05)))
            .collect();
        let values: Vec<f16> = (0..seq_len * head_dim)
            .map(|_| f16::from_f32(rng.next_f32()))
            .collect();
        let queries = rng.vec(group * head_dim, 2.0);
        let mut got = vec![f32::NAN; group * head_dim];
        attend_f16_group(&keys, &values, seq_len, head_dim, &queries, &mut got)
            .expect("f16 group attention");
        let want = reference_group(&keys, &values, seq_len, head_dim, &queries);
        let got_bits: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
        let want_bits: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            got_bits, want_bits,
            "seq_len {seq_len} head_dim {head_dim} group {group}: not bit-identical"
        );
    }
}

#[test]
fn f16_group_attention_handles_an_empty_history_and_rejects_bad_shapes() {
    let mut out = vec![7.0f32; 8];
    attend_f16_group(&[], &[], 0, 4, &[1.0; 8], &mut out).expect("empty history");
    assert!(
        out.iter().all(|&x| x == 0.0),
        "no history attends to nothing"
    );
    assert!(attend_f16_group(&[], &[], 0, 0, &[1.0; 8], &mut out).is_err());
    assert!(attend_f16_group(&[], &[], 0, 3, &[1.0; 8], &mut out).is_err());
    assert!(attend_f16_group(&[], &[], 0, 4, &[1.0; 8], &mut [0.0; 4]).is_err());
    let short = vec![f16::ZERO; 4];
    assert!(attend_f16_group(&short, &short, 2, 4, &[1.0; 4], &mut out).is_err());
}

#[test]
fn attend_group_matches_the_contiguous_kernel_on_both_element_types() {
    let (heads, hd, group, seq) = (2usize, 64usize, 3usize, 70usize);
    let mut rng = Rng(0xA77E_4D00_0000_0002);
    let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..heads * seq)
        .map(|_| (rng.vec(hd, 1.5), rng.vec(hd, 1.0)))
        .collect();
    let queries = rng.vec(group * hd, 1.0);
    for backing in [KvCacheBacking::DenseF32, KvCacheBacking::DenseF16] {
        let mut cache = KvCache::try_new_lazy(backing, 2, heads, hd, 512, 32).expect("lazy cache");
        for head in 0..heads {
            for pos in 0..seq {
                let (k, v) = &rows[head * seq + pos];
                cache.try_store_key(1, head, pos, k).expect("store");
                cache.try_store_value(1, head, pos, v).expect("store");
            }
        }
        for head in 0..heads {
            let mut got = vec![0.0f32; group * hd];
            cache
                .attend_group(1, head, seq, &queries, &mut got)
                .expect("attend");
            // Reference through the (dequantising) row accessor.
            let keys = cache.keys_for(1, head, seq);
            let values = cache.values_for(1, head, seq);
            let mut want = vec![0.0f32; group * hd];
            for (q, o) in queries.chunks_exact(hd).zip(want.chunks_exact_mut(hd)) {
                fused_attention_head_contiguous(q, keys, values, o, seq, hd).expect("ref");
            }
            assert_eq!(
                got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "{backing:?} head {head}"
            );
        }
    }
}

#[test]
fn attend_group_refuses_positions_that_were_never_allocated() {
    let cache = KvCache::try_new_lazy(KvCacheBacking::DenseF16, 1, 1, 4, 4096, 16).expect("lazy");
    let mut out = vec![0.0f32; 4];
    let err = cache
        .attend_group(0, 0, 17, &[1.0; 4], &mut out)
        .expect_err("past the allocation");
    assert!(matches!(err, ModelError::ShapeInvariant { .. }), "{err}");
    let err = cache
        .attend_group(0, 0, 5000, &[1.0; 4], &mut out)
        .expect_err("past the limit");
    assert!(matches!(err, ModelError::SequenceTooLong { .. }), "{err}");
    let err = cache
        .attend_group(3, 0, 1, &[1.0; 4], &mut out)
        .expect_err("layer out of range");
    assert!(matches!(err, ModelError::ShapeMismatch { .. }), "{err}");
}
