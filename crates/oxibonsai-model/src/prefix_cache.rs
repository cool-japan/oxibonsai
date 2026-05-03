//! Prefix KV-cache — share key/value tensors across requests with a common prefix.
//!
//! Architecture:
//!   - A `CacheBlock` holds the KV tensors for one "block" of tokens (block_size tokens).
//!   - Blocks are arranged in a trie keyed by token-id sequences.
//!   - A `PrefixCache` owns the trie and enforces a capacity limit (max_blocks).
//!   - Cache eviction uses LRU (Least Recently Used) policy tracked via a generation counter.

use std::collections::HashMap;

/// KV tensor pair for one block: (keys per layer, values per layer).
pub type KvBlockPair = (Vec<Vec<f32>>, Vec<Vec<f32>>);

// ──────────────────────────────────────────────────────────────────
// CacheBlock
// ──────────────────────────────────────────────────────────────────

/// One cache block: KV tensors for `block_size` tokens in every layer.
pub struct CacheBlock {
    /// key tensors: [num_layers][num_kv_heads * head_dim * block_size] f32
    pub keys: Vec<Vec<f32>>,
    /// value tensors: [num_layers][num_kv_heads * head_dim * block_size] f32
    pub values: Vec<Vec<f32>>,
    /// The exact token IDs this block covers.
    pub token_ids: Vec<u32>,
    /// LRU generation counter — higher means more recently used.
    pub last_used: u64,
    /// How many live requests are currently using this block.
    pub ref_count: usize,
}

impl CacheBlock {
    /// Allocate a new, zeroed cache block.
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, block_size: usize) -> Self {
        let per_layer = num_kv_heads * head_dim * block_size;
        let keys = (0..num_layers).map(|_| vec![0.0f32; per_layer]).collect();
        let values = (0..num_layers).map(|_| vec![0.0f32; per_layer]).collect();
        Self {
            keys,
            values,
            token_ids: Vec::new(),
            last_used: 0,
            ref_count: 0,
        }
    }

    /// Total memory consumed by this block's KV tensors in bytes.
    ///
    /// Formula: 2 (K+V) × num_layers × per_layer_elements × 4 bytes/f32.
    pub fn memory_bytes(&self) -> usize {
        let per_layer = self.keys.first().map(|v| v.len()).unwrap_or(0);
        // keys + values, each num_layers slices of per_layer f32s
        2 * self.keys.len() * per_layer * std::mem::size_of::<f32>()
    }
}

// ──────────────────────────────────────────────────────────────────
// Trie internals
// ──────────────────────────────────────────────────────────────────

/// A node in the prefix trie.
///
/// Uses a `Vec`-based arena (indices into `PrefixCache::nodes`) so that
/// all indices remain stable across insertions and evictions.
struct TrieNode {
    /// Maps token_id → child node index in the arena.
    children: HashMap<u32, usize>,
    /// Index into `PrefixCache::blocks` if this node holds a cached block.
    block_idx: Option<usize>,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            block_idx: None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// PrefixCache
// ──────────────────────────────────────────────────────────────────

/// Prefix KV-cache with trie-based lookup and LRU eviction.
///
/// The trie is keyed by complete blocks of `block_size` tokens.  Each
/// internal node in the trie corresponds to one block boundary; leaf
/// nodes that carry a `block_idx` have a fully populated `CacheBlock`.
pub struct PrefixCache {
    /// Arena of trie nodes.  Index 0 is always the root.
    nodes: Vec<TrieNode>,
    /// All allocated cache blocks (some may be logically free).
    blocks: Vec<CacheBlock>,
    /// Indices of blocks that are currently allocated (live).
    /// (We track occupied block indices; eviction removes from here.)
    occupied_blocks: Vec<usize>,
    /// Pool of block slots that have been evicted and can be reused.
    free_block_pool: Vec<usize>,
    /// Maximum number of simultaneously live blocks.
    max_blocks: usize,
    /// Tokens per block.
    block_size: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Monotonically increasing counter used for LRU tracking.
    generation: u64,
    /// Total cache hits since creation.
    pub hits: u64,
    /// Total cache misses since creation.
    pub misses: u64,
    /// Total blocks evicted since creation.
    pub evictions: u64,
}

impl PrefixCache {
    /// Create a new, empty prefix cache.
    pub fn new(
        max_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let root = TrieNode::new();
        Self {
            nodes: vec![root],
            blocks: Vec::new(),
            occupied_blocks: Vec::new(),
            free_block_pool: Vec::new(),
            max_blocks,
            block_size,
            num_layers,
            num_kv_heads,
            head_dim,
            generation: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    // ── public API ─────────────────────────────────────────────────

    /// Look up the longest cached prefix of `token_ids`.
    ///
    /// Walks the trie block-by-block.  For every complete block whose
    /// tokens match and whose trie node carries a cached block, the
    /// block's `last_used` stamp is refreshed and the block is returned.
    ///
    /// Returns `(matched_len, Vec<&CacheBlock>)`.
    pub fn lookup(&mut self, token_ids: &[u32]) -> (usize, Vec<&CacheBlock>) {
        let mut node_idx = 0usize; // root
        let mut matched_len = 0usize;
        let mut matched_block_indices: Vec<usize> = Vec::new();

        let full_blocks = token_ids.len() / self.block_size;

        for block_num in 0..full_blocks {
            let block_start = block_num * self.block_size;
            let block_end = block_start + self.block_size;
            let block_tokens = &token_ids[block_start..block_end];

            // All tokens in the block must follow the path in the trie.
            // We encode an entire block as a single edge keyed by the *first*
            // token of the block, using a compound key.  However, the trie is
            // keyed per-token — to support block-level granularity we use the
            // first token of the block as the edge key and store the full
            // `token_ids` list on the CacheBlock for validation.
            let edge_key = Self::block_edge_key(block_tokens);

            match self.nodes[node_idx].children.get(&edge_key).copied() {
                None => {
                    // No match at this block level — stop.
                    self.misses += 1;
                    break;
                }
                Some(child_node_idx) => {
                    // Check that the child node actually has a block.
                    let maybe_block_idx = self.nodes[child_node_idx].block_idx;
                    match maybe_block_idx {
                        None => {
                            self.misses += 1;
                            break;
                        }
                        Some(bidx) => {
                            // Validate token sequence matches exactly.
                            if self.blocks[bidx].token_ids != block_tokens {
                                self.misses += 1;
                                break;
                            }
                            // Hit — refresh timestamp and continue.
                            self.generation += 1;
                            self.blocks[bidx].last_used = self.generation;
                            self.blocks[bidx].ref_count += 1;
                            matched_len += self.block_size;
                            matched_block_indices.push(bidx);
                            self.hits += 1;
                            node_idx = child_node_idx;
                        }
                    }
                }
            }
        }

        // Collect immutable references (safe: we hold &mut self but return
        // shared refs to elements of self.blocks, which is fine for the caller).
        let block_refs: Vec<&CacheBlock> = matched_block_indices
            .iter()
            .map(|&bidx| &self.blocks[bidx])
            .collect();

        (matched_len, block_refs)
    }

    /// Insert a new block for `token_ids[block_start .. block_start + block_size]`.
    ///
    /// Evicts the LRU block if the cache is at capacity.
    /// Returns the index of the inserted block in `self.blocks`.
    pub fn insert(
        &mut self,
        token_ids: &[u32],
        block_start: usize,
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
    ) -> usize {
        // Evict if necessary.
        while self.occupied_blocks.len() >= self.max_blocks {
            if !self.evict_lru() {
                // Nothing evictable — cache is pinned; caller must wait.
                break;
            }
        }

        let block_end = block_start + self.block_size;
        let block_tokens = token_ids[block_start..block_end.min(token_ids.len())].to_vec();

        // Navigate/build the trie path up to this block.
        let mut node_idx = 0usize;
        let num_full_blocks_before = block_start / self.block_size;

        for blk in 0..num_full_blocks_before {
            let seg_start = blk * self.block_size;
            let seg_end = seg_start + self.block_size;
            let seg = &token_ids[seg_start..seg_end];
            let edge_key = Self::block_edge_key(seg);

            if let Some(&child) = self.nodes[node_idx].children.get(&edge_key) {
                node_idx = child;
            } else {
                // Intermediate node missing; create it (no block data).
                let new_node_idx = self.nodes.len();
                self.nodes.push(TrieNode::new());
                self.nodes[node_idx].children.insert(edge_key, new_node_idx);
                node_idx = new_node_idx;
            }
        }

        // Insert/update the leaf node for this block.
        let edge_key = Self::block_edge_key(&block_tokens);

        let leaf_node_idx = if let Some(&existing) = self.nodes[node_idx].children.get(&edge_key) {
            existing
        } else {
            let new_node_idx = self.nodes.len();
            self.nodes.push(TrieNode::new());
            self.nodes[node_idx].children.insert(edge_key, new_node_idx);
            new_node_idx
        };

        // Assign or reuse a block slot.
        self.generation += 1;
        let block_idx = if let Some(reuse_idx) = self.free_block_pool.pop() {
            // Reuse a previously evicted slot.
            let block = &mut self.blocks[reuse_idx];
            block.keys = keys;
            block.values = values;
            block.token_ids = block_tokens;
            block.last_used = self.generation;
            block.ref_count = 0;
            reuse_idx
        } else {
            // Allocate a new slot.
            let mut blk = CacheBlock::new(
                self.num_layers,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            );
            blk.keys = keys;
            blk.values = values;
            blk.token_ids = block_tokens;
            blk.last_used = self.generation;
            blk.ref_count = 0;
            let idx = self.blocks.len();
            self.blocks.push(blk);
            idx
        };

        self.nodes[leaf_node_idx].block_idx = Some(block_idx);
        self.occupied_blocks.push(block_idx);

        block_idx
    }

    /// Decrement the reference count of a block, making it eligible for eviction.
    pub fn release(&mut self, block_idx: usize) {
        if block_idx < self.blocks.len() && self.blocks[block_idx].ref_count > 0 {
            self.blocks[block_idx].ref_count -= 1;
        }
    }

    /// Number of currently live (occupied) blocks.
    pub fn len(&self) -> usize {
        self.occupied_blocks.len()
    }

    /// Returns `true` if the cache contains no live blocks.
    pub fn is_empty(&self) -> bool {
        self.occupied_blocks.is_empty()
    }

    /// Maximum number of blocks this cache can hold.
    pub fn capacity(&self) -> usize {
        self.max_blocks
    }

    /// Total memory consumed by all live blocks' KV tensors.
    pub fn memory_bytes(&self) -> usize {
        self.occupied_blocks
            .iter()
            .map(|&idx| self.blocks[idx].memory_bytes())
            .sum()
    }

    /// Cache hit rate in [0, 1].  Returns 0.0 if no lookups have been made.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    /// Tokens per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Borrow a cached block by its index in the underlying arena.
    ///
    /// Block indices are returned by [`CacheSession::block_indices`] when a
    /// session is prepared via
    /// [`PrefixAwarePrefill::prepare`](crate::prefix_cache::PrefixAwarePrefill::prepare).
    /// `None` means the index is out of bounds (e.g. the sentinel
    /// `usize::MAX` placed for trie path failures).
    pub fn get_block(&self, idx: usize) -> Option<&CacheBlock> {
        self.blocks.get(idx)
    }

    /// Remove all cached blocks, resetting the trie to an empty root.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(TrieNode::new());
        self.blocks.clear();
        self.occupied_blocks.clear();
        self.free_block_pool.clear();
        self.generation = 0;
        // Statistics are intentionally preserved across clear().
    }

    // ── private helpers ────────────────────────────────────────────

    /// Compute a single `u32` edge key that represents an entire block.
    ///
    /// We use a simple polynomial hash of the token IDs so that distinct
    /// token sequences produce distinct keys with very high probability.
    /// The trie node still stores the full `token_ids` in the `CacheBlock`
    /// for exact-match validation on lookup.
    fn block_edge_key(tokens: &[u32]) -> u32 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
        for &t in tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
        }
        // Fold to 32 bits.
        ((h >> 32) ^ (h & 0xffff_ffff)) as u32
    }

    /// Evict the least-recently-used block with `ref_count == 0`.
    ///
    /// Returns `true` if a block was evicted, `false` if all blocks are pinned.
    fn evict_lru(&mut self) -> bool {
        // Find the occupied block index with the smallest `last_used` that has ref_count == 0.
        let victim_pos = self
            .occupied_blocks
            .iter()
            .enumerate()
            .filter(|(_, &bidx)| self.blocks[bidx].ref_count == 0)
            .min_by_key(|(_, &bidx)| self.blocks[bidx].last_used)
            .map(|(pos, _)| pos);

        let Some(pos) = victim_pos else {
            return false;
        };

        let victim_block_idx = self.occupied_blocks.swap_remove(pos);

        // Remove the corresponding trie node → block association.
        // We search for the trie node whose block_idx == victim_block_idx.
        for node in &mut self.nodes {
            if node.block_idx == Some(victim_block_idx) {
                node.block_idx = None;
                break;
            }
        }

        // Return the slot to the free pool for reuse.
        self.free_block_pool.push(victim_block_idx);
        self.evictions += 1;

        true
    }
}

// ──────────────────────────────────────────────────────────────────
// CacheSession
// ──────────────────────────────────────────────────────────────────

/// A handle that tracks which cache blocks a specific request is using.
///
/// Holding a `CacheSession` keeps the referenced blocks' `ref_count`
/// elevated so they will not be evicted while the request is in flight.
pub struct CacheSession {
    /// Number of prefix tokens that were already cached.
    pub matched_prefix_len: usize,
    /// Indices of the cache blocks matched for this session (in order).
    pub block_indices: Vec<usize>,
}

impl CacheSession {
    /// Create a new session handle.
    pub fn new(matched_prefix_len: usize, block_indices: Vec<usize>) -> Self {
        Self {
            matched_prefix_len,
            block_indices,
        }
    }

    /// Number of tokens covered by cached blocks in this session.
    ///
    /// May differ slightly from `matched_prefix_len` if the prefix was not
    /// an exact multiple of `block_size`, but in practice they will be equal.
    pub fn cached_tokens(&self, block_size: usize) -> usize {
        self.block_indices.len() * block_size
    }

    /// Returns `true` if no prefix tokens were cached.
    pub fn is_empty(&self) -> bool {
        self.block_indices.is_empty()
    }
}

// ──────────────────────────────────────────────────────────────────
// PrefixAwarePrefill
// ──────────────────────────────────────────────────────────────────

/// Wraps a [`PrefixCache`] and exposes a higher-level prefill API.
///
/// The typical call pattern for one request is:
///
/// ```text
/// let (session, uncached_start) = prefill.prepare(&token_ids);
/// // run your model prefill on token_ids[uncached_start..]
/// prefill.store_blocks(&token_ids, uncached_start, new_kv_blocks);
/// prefill.release_session(session);
/// ```
pub struct PrefixAwarePrefill {
    /// The underlying prefix cache.
    pub cache: PrefixCache,
}

impl PrefixAwarePrefill {
    /// Wrap an existing `PrefixCache`.
    pub fn new(cache: PrefixCache) -> Self {
        Self { cache }
    }

    /// Determine how much of `token_ids` is already cached.
    ///
    /// Returns `(session, uncached_start)` where `uncached_start` is the
    /// index of the first token that must be processed by the model.
    pub fn prepare(&mut self, token_ids: &[u32]) -> (CacheSession, usize) {
        // Phase 1: lookup to get matched_len and number of matched blocks.
        // lookup() already increments ref_counts for matched blocks.
        let (matched_len, matched_blocks) = self.cache.lookup(token_ids);
        let num_matched = matched_blocks.len();
        // Drop the borrowed references immediately.
        drop(matched_blocks);

        // Phase 2: recover the block indices by walking the trie (no borrow conflict now).
        let block_indices: Vec<usize> = (0..num_matched)
            .map(|blk_num| {
                let block_start = blk_num * self.cache.block_size;
                let block_tokens = &token_ids[block_start..block_start + self.cache.block_size];
                let edge_key = PrefixCache::block_edge_key(block_tokens);
                self.find_block_idx_for_edge(blk_num, token_ids, edge_key)
            })
            .collect();

        let uncached_start = matched_len;
        let session = CacheSession::new(matched_len, block_indices);
        (session, uncached_start)
    }

    /// After prefill, store the newly computed KV blocks back into the cache.
    ///
    /// `keys_by_block` is a list of `(keys, values)` for each newly computed
    /// block, in order, starting from the block at `uncached_start`.
    pub fn store_blocks(
        &mut self,
        token_ids: &[u32],
        uncached_start: usize,
        keys_by_block: Vec<KvBlockPair>,
    ) {
        let block_size = self.cache.block_size;
        for (i, (keys, values)) in keys_by_block.into_iter().enumerate() {
            let block_start = uncached_start + i * block_size;
            let block_end = block_start + block_size;
            if block_end > token_ids.len() {
                // Incomplete final block — do not cache partial blocks.
                break;
            }
            self.cache.insert(token_ids, block_start, keys, values);
        }
    }

    /// Release all blocks held by a session (decrement their ref counts).
    pub fn release_session(&mut self, session: CacheSession) {
        for bidx in session.block_indices {
            self.cache.release(bidx);
        }
    }

    /// Snapshot of current cache statistics.
    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            hit_rate: self.cache.hit_rate(),
            cached_blocks: self.cache.len(),
            capacity_blocks: self.cache.capacity(),
            memory_bytes: self.cache.memory_bytes(),
            total_hits: self.cache.hits,
            total_misses: self.cache.misses,
            total_evictions: self.cache.evictions,
        }
    }

    // ── private helpers ────────────────────────────────────────────

    /// Walk the trie to find the block index for the block at position `blk_num`.
    fn find_block_idx_for_edge(&self, blk_num: usize, token_ids: &[u32], edge_key: u32) -> usize {
        // Navigate to the parent of the target node.
        let mut node_idx = 0usize;
        for blk in 0..blk_num {
            let seg_start = blk * self.cache.block_size;
            let seg_end = seg_start + self.cache.block_size;
            let seg = &token_ids[seg_start..seg_end];
            let parent_edge_key = PrefixCache::block_edge_key(seg);
            if let Some(&child) = self.cache.nodes[node_idx].children.get(&parent_edge_key) {
                node_idx = child;
            } else {
                // Trie path broken — return a sentinel.
                return usize::MAX;
            }
        }
        // Now look up the target child node.
        if let Some(&child_idx) = self.cache.nodes[node_idx].children.get(&edge_key) {
            self.cache.nodes[child_idx].block_idx.unwrap_or(usize::MAX)
        } else {
            usize::MAX
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// PrefixCacheStats
// ──────────────────────────────────────────────────────────────────

/// A snapshot of prefix-cache statistics for observability.
#[derive(Debug, serde::Serialize)]
pub struct PrefixCacheStats {
    /// Fraction of lookups that found a cached block, in [0, 1].
    pub hit_rate: f32,
    /// Number of blocks currently in the cache.
    pub cached_blocks: usize,
    /// Maximum number of blocks the cache can hold.
    pub capacity_blocks: usize,
    /// Total memory consumed by KV data in bytes.
    pub memory_bytes: usize,
    /// Cumulative cache hits.
    pub total_hits: u64,
    /// Cumulative cache misses.
    pub total_misses: u64,
    /// Cumulative evictions.
    pub total_evictions: u64,
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a CacheBlock with predictable data.
    fn make_block(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> CacheBlock {
        CacheBlock::new(num_layers, num_kv_heads, head_dim, block_size)
    }

    // Helper: build key/value layer tensors filled with a constant.
    fn make_kv(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        val: f32,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let per_layer = num_kv_heads * head_dim * block_size;
        let keys: Vec<Vec<f32>> = (0..num_layers).map(|_| vec![val; per_layer]).collect();
        let values: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![val + 1.0; per_layer])
            .collect();
        (keys, values)
    }

    #[test]
    fn test_cache_block_memory_bytes() {
        // 2 layers, 4 heads, head_dim=8, block_size=4
        // per_layer = 4 * 8 * 4 = 128 f32s
        // memory = 2 (K+V) * 2 layers * 128 * 4 bytes = 2048
        let block = make_block(2, 4, 8, 4);
        let expected = 2 * 2 * (4 * 8 * 4) * std::mem::size_of::<f32>();
        assert_eq!(block.memory_bytes(), expected);
    }

    #[test]
    fn test_prefix_cache_insert_and_lookup_hit() {
        let mut cache = PrefixCache::new(8, 4, 2, 2, 8);
        let token_ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let (keys, values) = make_kv(2, 2, 8, 4, 1.0);
        cache.insert(&token_ids, 0, keys, values);

        let (matched, blocks) = cache.lookup(&token_ids);
        assert_eq!(matched, 4, "should match one full block of 4 tokens");
        assert_eq!(blocks.len(), 1);
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn test_prefix_cache_lookup_miss() {
        let mut cache = PrefixCache::new(8, 4, 2, 2, 8);
        let token_ids: Vec<u32> = vec![10, 20, 30, 40];

        let (matched, blocks) = cache.lookup(&token_ids);
        assert_eq!(matched, 0);
        assert!(blocks.is_empty());
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn test_prefix_cache_partial_prefix_match() {
        let mut cache = PrefixCache::new(8, 4, 2, 2, 8);
        // Insert block 0 (tokens 0..4)
        let token_ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (keys0, values0) = make_kv(2, 2, 8, 4, 0.5);
        cache.insert(&token_ids, 0, keys0, values0);

        // Query with same first block but different second block.
        let query: Vec<u32> = vec![1, 2, 3, 4, 9, 10, 11, 12];
        let (matched, blocks) = cache.lookup(&query);
        // Should match first block (4 tokens) but miss on second.
        assert_eq!(matched, 4);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_prefix_cache_lru_eviction() {
        // max_blocks = 2 so inserting a third triggers eviction.
        let mut cache = PrefixCache::new(2, 4, 1, 1, 4);

        let tokens_a: Vec<u32> = vec![1, 2, 3, 4];
        let tokens_b: Vec<u32> = vec![5, 6, 7, 8];
        let tokens_c: Vec<u32> = vec![9, 10, 11, 12];

        let (ka, va) = make_kv(1, 1, 4, 4, 1.0);
        let (kb, vb) = make_kv(1, 1, 4, 4, 2.0);
        let (kc, vc) = make_kv(1, 1, 4, 4, 3.0);

        cache.insert(&tokens_a, 0, ka, va);
        cache.insert(&tokens_b, 0, kb, vb);
        // Access token_b to make it more recently used than token_a.
        let _ = cache.lookup(&tokens_b);
        // Now insert token_c — should evict token_a (LRU).
        cache.insert(&tokens_c, 0, kc, vc);

        assert_eq!(
            cache.len(),
            2,
            "should have exactly 2 blocks after eviction"
        );
        assert_eq!(cache.evictions, 1);

        // token_a should no longer be found.
        let (matched_a, _) = cache.lookup(&tokens_a);
        assert_eq!(matched_a, 0, "evicted block should not be found");
    }

    #[test]
    fn test_prefix_cache_ref_count_prevents_eviction() {
        let mut cache = PrefixCache::new(1, 4, 1, 1, 4);

        let tokens_a: Vec<u32> = vec![1, 2, 3, 4];
        let tokens_b: Vec<u32> = vec![5, 6, 7, 8];

        let (ka, va) = make_kv(1, 1, 4, 4, 1.0);
        let (kb, vb) = make_kv(1, 1, 4, 4, 2.0);

        let bidx_a = cache.insert(&tokens_a, 0, ka, va);
        // Pin block a by incrementing ref_count manually (simulates an active session).
        cache.blocks[bidx_a].ref_count += 1;

        // Inserting tokens_b when at capacity should fail to evict because bidx_a is pinned.
        cache.insert(&tokens_b, 0, kb, vb);

        // No evictions should have happened — the only eligible block was pinned.
        assert_eq!(cache.evictions, 0, "pinned block must not be evicted");

        // Release the manual pin.
        cache.release(bidx_a);
        assert_eq!(cache.blocks[bidx_a].ref_count, 0);
    }

    #[test]
    fn test_prefix_cache_hit_rate() {
        let mut cache = PrefixCache::new(8, 4, 1, 1, 4);
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let (k, v) = make_kv(1, 1, 4, 4, 1.0);
        cache.insert(&tokens, 0, k, v);

        // 1 hit
        let _ = cache.lookup(&tokens);
        // 1 miss
        let _ = cache.lookup(&[99, 100, 101, 102]);

        let rate = cache.hit_rate();
        assert!(
            (rate - 0.5).abs() < 1e-5,
            "hit rate should be 0.5, got {rate}"
        );
    }

    #[test]
    fn test_prefix_cache_clear() {
        let mut cache = PrefixCache::new(8, 4, 1, 1, 4);
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let (k, v) = make_kv(1, 1, 4, 4, 1.0);
        cache.insert(&tokens, 0, k, v);
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        // After clear, lookup should miss.
        let (matched, _) = cache.lookup(&tokens);
        assert_eq!(matched, 0);
    }

    #[test]
    fn test_cache_session_cached_tokens() {
        let session = CacheSession::new(8, vec![0, 1]);
        assert_eq!(session.cached_tokens(4), 8);
        assert!(!session.is_empty());

        let empty = CacheSession::new(0, vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.cached_tokens(4), 0);
    }

    #[test]
    fn test_prefix_aware_prefill_prepare() {
        let inner = PrefixCache::new(8, 4, 1, 1, 4);
        let mut prefill = PrefixAwarePrefill::new(inner);

        // Insert a block for the first 4 tokens.
        let token_ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (k, v) = make_kv(1, 1, 4, 4, 1.0);
        prefill.cache.insert(&token_ids, 0, k, v);

        let (session, uncached_start) = prefill.prepare(&token_ids);
        // First block (4 tokens) should be cached.
        assert_eq!(session.matched_prefix_len, 4);
        assert_eq!(uncached_start, 4);

        prefill.release_session(session);
    }

    #[test]
    fn test_prefix_cache_stats() {
        let inner = PrefixCache::new(8, 4, 1, 1, 4);
        let mut prefill = PrefixAwarePrefill::new(inner);

        let token_ids: Vec<u32> = vec![1, 2, 3, 4];
        let (k, v) = make_kv(1, 1, 4, 4, 1.0);
        prefill.cache.insert(&token_ids, 0, k, v);

        let _ = prefill.prepare(&token_ids);

        let stats = prefill.stats();
        assert!(stats.cached_blocks > 0 || stats.total_hits > 0 || stats.total_misses > 0);
        assert_eq!(stats.capacity_blocks, 8);
    }

    #[test]
    fn test_prefix_cache_capacity_enforcement() {
        let capacity = 4usize;
        let mut cache = PrefixCache::new(capacity, 4, 1, 1, 4);

        for i in 0..capacity + 2 {
            let tokens: Vec<u32> = (0..4).map(|j| (i * 4 + j) as u32).collect();
            let (k, v) = make_kv(1, 1, 4, 4, i as f32);
            cache.insert(&tokens, 0, k, v);
        }

        assert!(
            cache.len() <= capacity,
            "cache should not exceed max_blocks={capacity}, got {}",
            cache.len()
        );
        assert!(
            cache.evictions >= 2,
            "should have evicted at least 2 blocks"
        );
    }
}
