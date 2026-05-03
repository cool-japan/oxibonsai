//! KV Cache for autoregressive generation.
//!
//! Stores key and value tensors for each layer to avoid recomputation
//! during token-by-token generation. Provides both a standard contiguous
//! cache and a page-based cache for memory-efficient allocation.

/// Policy for KV cache storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCachePolicy {
    /// Standard FP32 cache (contiguous allocation).
    #[default]
    Standard,
    /// FP16 cache (half the memory of Standard).
    Fp16,
    /// Sliding window cache: only retain the most recent N positions.
    SlidingWindow(usize),
}

/// Per-layer KV cache storing FP32 key and value vectors.
#[derive(Debug)]
pub struct KvCache {
    /// Number of Transformer layers.
    num_layers: usize,
    /// Number of KV heads per layer.
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Current sequence length (number of tokens cached).
    seq_len: usize,
    /// Key cache: [num_layers × num_kv_heads × max_seq_len × head_dim].
    keys: Vec<f32>,
    /// Value cache: [num_layers × num_kv_heads × max_seq_len × head_dim].
    values: Vec<f32>,
}

impl KvCache {
    /// Create a new KV cache.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let total = num_layers * num_kv_heads * max_seq_len * head_dim;
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
            keys: vec![0.0; total],
            values: vec![0.0; total],
        }
    }

    /// Current number of cached tokens.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Store a key vector for a specific layer, head, and position.
    pub fn store_key(&mut self, layer: usize, head: usize, pos: usize, key: &[f32]) {
        debug_assert!(layer < self.num_layers);
        debug_assert!(head < self.num_kv_heads);
        debug_assert!(pos < self.max_seq_len);
        debug_assert_eq!(key.len(), self.head_dim);

        let offset = self.cache_offset(layer, head, pos);
        self.keys[offset..offset + self.head_dim].copy_from_slice(key);
    }

    /// Store a value vector for a specific layer, head, and position.
    pub fn store_value(&mut self, layer: usize, head: usize, pos: usize, value: &[f32]) {
        debug_assert!(layer < self.num_layers);
        debug_assert!(head < self.num_kv_heads);
        debug_assert!(pos < self.max_seq_len);
        debug_assert_eq!(value.len(), self.head_dim);

        let offset = self.cache_offset(layer, head, pos);
        self.values[offset..offset + self.head_dim].copy_from_slice(value);
    }

    /// Get all cached keys for a layer and head up to `seq_len`.
    ///
    /// Returns a slice of [seq_len × head_dim] in row-major order.
    pub fn keys_for(&self, layer: usize, head: usize, seq_len: usize) -> &[f32] {
        let start = self.cache_offset(layer, head, 0);
        let end = start + seq_len * self.head_dim;
        &self.keys[start..end]
    }

    /// Get all cached values for a layer and head up to `seq_len`.
    pub fn values_for(&self, layer: usize, head: usize, seq_len: usize) -> &[f32] {
        let start = self.cache_offset(layer, head, 0);
        let end = start + seq_len * self.head_dim;
        &self.values[start..end]
    }

    /// Advance the sequence position by one token.
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Reset the cache (clear all stored KV pairs).
    pub fn clear(&mut self) {
        self.seq_len = 0;
        // Optionally zero out for security, but not required for correctness
    }

    /// Compute flat offset into cache arrays.
    fn cache_offset(&self, layer: usize, head: usize, pos: usize) -> usize {
        ((layer * self.num_kv_heads + head) * self.max_seq_len + pos) * self.head_dim
    }

    /// Total memory used by this cache in bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * std::mem::size_of::<f32>()
    }

    /// Utilization ratio: fraction of cache capacity currently used.
    ///
    /// Returns a value in [0.0, 1.0].
    pub fn utilization_ratio(&self) -> f64 {
        if self.max_seq_len == 0 {
            return 0.0;
        }
        self.seq_len as f64 / self.max_seq_len as f64
    }

    /// Number of layers in this cache.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of KV heads per layer.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Manually set the cached sequence length.
    ///
    /// Used by the prefix-cache integration when restoring previously
    /// computed KV blocks: after [`inject_block`](Self::inject_block) writes
    /// the block contents, the consumer must call this to advertise the
    /// number of valid positions to subsequent attention computations.
    ///
    /// `n` is clamped to `max_seq_len`.
    pub fn set_seq_len(&mut self, n: usize) {
        self.seq_len = n.min(self.max_seq_len);
    }

    /// Extract one prefix-cache block worth of KV for a single layer.
    ///
    /// Reads `block_size` consecutive positions starting at `start_pos` for
    /// every KV head in `layer` and returns them in `[head][pos_in_block][dim]`
    /// order, packed as a flat `Vec<f32>` of length
    /// `num_kv_heads * block_size * head_dim`.
    ///
    /// Mirrors the layout used by [`crate::prefix_cache::CacheBlock`].
    ///
    /// Returns `(keys, values)`. If the requested range exceeds
    /// `max_seq_len`, the trailing positions are returned as zeros.
    pub fn extract_block(
        &self,
        layer: usize,
        start_pos: usize,
        block_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        debug_assert!(layer < self.num_layers);
        let per_layer = self.num_kv_heads * block_size * self.head_dim;
        let mut keys = vec![0.0f32; per_layer];
        let mut values = vec![0.0f32; per_layer];

        for head in 0..self.num_kv_heads {
            for off in 0..block_size {
                let pos = start_pos + off;
                if pos >= self.max_seq_len {
                    continue;
                }
                let src = self.cache_offset(layer, head, pos);
                let dst = (head * block_size + off) * self.head_dim;
                keys[dst..dst + self.head_dim]
                    .copy_from_slice(&self.keys[src..src + self.head_dim]);
                values[dst..dst + self.head_dim]
                    .copy_from_slice(&self.values[src..src + self.head_dim]);
            }
        }

        (keys, values)
    }

    /// Inject a previously extracted block back into the cache for a single layer.
    ///
    /// `keys` and `values` must have the same `[head][pos_in_block][dim]`
    /// layout produced by [`extract_block`](Self::extract_block); they are
    /// expected to be of length `num_kv_heads * block_size * head_dim`.
    /// Positions outside `max_seq_len` are silently skipped.
    pub fn inject_block(
        &mut self,
        layer: usize,
        start_pos: usize,
        block_size: usize,
        keys: &[f32],
        values: &[f32],
    ) {
        debug_assert!(layer < self.num_layers);
        let per_layer = self.num_kv_heads * block_size * self.head_dim;
        debug_assert_eq!(keys.len(), per_layer);
        debug_assert_eq!(values.len(), per_layer);

        for head in 0..self.num_kv_heads {
            for off in 0..block_size {
                let pos = start_pos + off;
                if pos >= self.max_seq_len {
                    continue;
                }
                let src = (head * block_size + off) * self.head_dim;
                let dst = self.cache_offset(layer, head, pos);
                self.keys[dst..dst + self.head_dim]
                    .copy_from_slice(&keys[src..src + self.head_dim]);
                self.values[dst..dst + self.head_dim]
                    .copy_from_slice(&values[src..src + self.head_dim]);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Paged KV Cache
// ──────────────────────────────────────────────────────────────────

/// Default number of positions per page.
const DEFAULT_PAGE_SIZE: usize = 256;

/// A single page in the paged KV cache.
///
/// Each page holds `page_size` positions worth of key and value data
/// for a single layer and head.
#[derive(Debug, Clone)]
struct KvPage {
    /// Key data: [page_size * head_dim] floats.
    keys: Vec<f32>,
    /// Value data: [page_size * head_dim] floats.
    values: Vec<f32>,
    /// Number of positions actually used in this page.
    used: usize,
}

impl KvPage {
    fn new(page_size: usize, head_dim: usize) -> Self {
        Self {
            keys: vec![0.0; page_size * head_dim],
            values: vec![0.0; page_size * head_dim],
            used: 0,
        }
    }
}

/// Page-based KV cache for memory-efficient allocation.
///
/// Instead of pre-allocating the full `max_seq_len` contiguously,
/// pages of `page_size` positions are allocated on demand. This is
/// beneficial when the actual sequence length is much shorter than
/// `max_seq_len`.
#[derive(Debug)]
pub struct PagedKvCache {
    /// Pages indexed as [layer][head][page_index].
    pages: Vec<Vec<Vec<KvPage>>>,
    /// Number of transformer layers.
    num_layers: usize,
    /// Number of KV heads per layer.
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Positions per page.
    page_size: usize,
    /// Maximum sequence length (total capacity).
    max_seq_len: usize,
    /// Current sequence length.
    seq_len: usize,
}

impl PagedKvCache {
    /// Create a new paged KV cache.
    ///
    /// Pages are allocated lazily as positions are stored.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self::with_page_size(
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            DEFAULT_PAGE_SIZE,
        )
    }

    /// Create a new paged KV cache with a custom page size.
    pub fn with_page_size(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        page_size: usize,
    ) -> Self {
        let pages = (0..num_layers)
            .map(|_| (0..num_kv_heads).map(|_| Vec::new()).collect())
            .collect();

        Self {
            pages,
            num_layers,
            num_kv_heads,
            head_dim,
            page_size,
            max_seq_len,
            seq_len: 0,
        }
    }

    /// Store a key vector for a specific layer, head, and position.
    pub fn store_key(&mut self, layer: usize, head: usize, pos: usize, key: &[f32]) {
        debug_assert!(layer < self.num_layers);
        debug_assert!(head < self.num_kv_heads);
        debug_assert!(pos < self.max_seq_len);
        debug_assert_eq!(key.len(), self.head_dim);

        let page_idx = pos / self.page_size;
        let offset_in_page = pos % self.page_size;

        self.ensure_page(layer, head, page_idx);

        let page = &mut self.pages[layer][head][page_idx];
        let start = offset_in_page * self.head_dim;
        page.keys[start..start + self.head_dim].copy_from_slice(key);
        if offset_in_page >= page.used {
            page.used = offset_in_page + 1;
        }
    }

    /// Store a value vector for a specific layer, head, and position.
    pub fn store_value(&mut self, layer: usize, head: usize, pos: usize, value: &[f32]) {
        debug_assert!(layer < self.num_layers);
        debug_assert!(head < self.num_kv_heads);
        debug_assert!(pos < self.max_seq_len);
        debug_assert_eq!(value.len(), self.head_dim);

        let page_idx = pos / self.page_size;
        let offset_in_page = pos % self.page_size;

        self.ensure_page(layer, head, page_idx);

        let page = &mut self.pages[layer][head][page_idx];
        let start = offset_in_page * self.head_dim;
        page.values[start..start + self.head_dim].copy_from_slice(value);
        if offset_in_page >= page.used {
            page.used = offset_in_page + 1;
        }
    }

    /// Get all cached keys for a layer and head up to `seq_len`, assembled into a contiguous buffer.
    pub fn keys_for(&self, layer: usize, head: usize, seq_len: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(seq_len * self.head_dim);
        let head_pages = &self.pages[layer][head];

        for pos in 0..seq_len {
            let page_idx = pos / self.page_size;
            let offset_in_page = pos % self.page_size;

            if page_idx < head_pages.len() {
                let page = &head_pages[page_idx];
                let start = offset_in_page * self.head_dim;
                result.extend_from_slice(&page.keys[start..start + self.head_dim]);
            } else {
                // Page not yet allocated; fill with zeros
                result.extend(std::iter::repeat_n(0.0f32, self.head_dim));
            }
        }

        result
    }

    /// Get all cached values for a layer and head up to `seq_len`.
    pub fn values_for(&self, layer: usize, head: usize, seq_len: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(seq_len * self.head_dim);
        let head_pages = &self.pages[layer][head];

        for pos in 0..seq_len {
            let page_idx = pos / self.page_size;
            let offset_in_page = pos % self.page_size;

            if page_idx < head_pages.len() {
                let page = &head_pages[page_idx];
                let start = offset_in_page * self.head_dim;
                result.extend_from_slice(&page.values[start..start + self.head_dim]);
            } else {
                result.extend(std::iter::repeat_n(0.0f32, self.head_dim));
            }
        }

        result
    }

    /// Current sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Advance the sequence position by one token.
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Reset the cache (deallocate all pages).
    pub fn clear(&mut self) {
        self.seq_len = 0;
        for layer_pages in &mut self.pages {
            for head_pages in layer_pages.iter_mut() {
                head_pages.clear();
            }
        }
    }

    /// Total memory currently allocated by this cache in bytes.
    ///
    /// Only counts allocated pages, not the full capacity.
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total_pages = 0usize;
        for layer_pages in &self.pages {
            for head_pages in layer_pages {
                total_pages += head_pages.len();
            }
        }
        // Each page has keys + values, each of page_size * head_dim floats
        total_pages * self.page_size * self.head_dim * std::mem::size_of::<f32>() * 2
    }

    /// Utilization ratio: fraction of allocated pages that are used.
    pub fn utilization_ratio(&self) -> f64 {
        let mut total_slots = 0usize;
        let mut used_slots = 0usize;
        for layer_pages in &self.pages {
            for head_pages in layer_pages {
                for page in head_pages {
                    total_slots += self.page_size;
                    used_slots += page.used;
                }
            }
        }
        if total_slots == 0 {
            return 0.0;
        }
        used_slots as f64 / total_slots as f64
    }

    /// Total number of pages allocated.
    pub fn total_pages(&self) -> usize {
        let mut count = 0usize;
        for layer_pages in &self.pages {
            for head_pages in layer_pages {
                count += head_pages.len();
            }
        }
        count
    }

    /// Page size (positions per page).
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Ensure a page exists at the given index, allocating it if needed.
    fn ensure_page(&mut self, layer: usize, head: usize, page_idx: usize) {
        let head_pages = &mut self.pages[layer][head];
        while head_pages.len() <= page_idx {
            head_pages.push(KvPage::new(self.page_size, self.head_dim));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_store_and_retrieve() {
        let mut cache = KvCache::new(2, 8, 128, 16);

        let key = vec![1.0f32; 128];
        let value = vec![2.0f32; 128];

        cache.store_key(0, 0, 0, &key);
        cache.store_value(0, 0, 0, &value);
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

        cache.store_key(0, 0, 0, &[1.0, 2.0, 3.0, 4.0]);
        cache.advance();
        cache.store_key(0, 0, 1, &[5.0, 6.0, 7.0, 8.0]);
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

    #[test]
    fn kv_cache_utilization() {
        let mut cache = KvCache::new(1, 1, 4, 10);
        assert!((cache.utilization_ratio() - 0.0).abs() < 1e-10);

        cache.advance();
        cache.advance();
        cache.advance();
        assert!((cache.utilization_ratio() - 0.3).abs() < 1e-10);
    }

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
                cache.store_key(1, head, pos, &key);
                cache.store_value(1, head, pos, &value);
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
            cache.store_key(0, 0, 4 + pos, &key);
            cache.store_value(0, 0, 4 + pos, &value);
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

    #[test]
    fn paged_kv_cache_utilization() {
        let mut cache = PagedKvCache::with_page_size(1, 1, 4, 16, 4);
        assert!((cache.utilization_ratio() - 0.0).abs() < 1e-10);

        cache.store_key(0, 0, 0, &[1.0; 4]);
        // 1 used out of 4 slots in 1 page = 0.25
        assert!((cache.utilization_ratio() - 0.25).abs() < 1e-10);
    }
}
