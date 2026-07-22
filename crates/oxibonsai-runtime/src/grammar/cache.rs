//! Memoization cache for Earley `allowed_tokens` results.
//!
//! Caches per-state token masks keyed by a 64-bit Earley chart hash **and**
//! the vocabulary size the mask was built against. Capacity is bounded;
//! eviction is LRU.
//!
//! # Design notes
//!
//! * The cache is keyed by `(state_hash, vocab_size)` — `state_hash` is a
//!   64-bit hash of the current chart set at `input_pos` from
//!   `EarleyRecognizer::state_hash()`; `vocab_size` is the mask length. The
//!   `vocab_size` component is required because `allowed_tokens` takes
//!   `vocab_size` as a per-call parameter (per the `TokenConstraint` trait):
//!   if the same `GrammarConstraint` were ever queried with two different
//!   `vocab_size` values while revisiting the same Earley state, keying on
//!   `state_hash` alone would return a cached mask of the wrong length for
//!   the second caller, silently under- or over-masking rather than
//!   producing a vector sized to what the caller asked for. Hash collisions
//!   *within* a fixed `(state_hash, vocab_size)` pair are theoretically
//!   possible but extremely unlikely in practice; the worst case there is a
//!   spurious cache hit returning a slightly wrong mask, which degrades
//!   output quality without crashing.
//!
//! * Eviction is strict LRU maintained by a `VecDeque<(u64, usize)>` (key
//!   order list). On a cache hit the key is moved to the back. On eviction
//!   the front is removed. This is O(n) for the hit path due to
//!   `VecDeque::position`, but grammar parse states are typically
//!   O(|grammar|²) bounded and capacity is small (default 256), so the scan
//!   cost is negligible.
//!
//! * All allocation is on the Rust heap; no unsafe code.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Cache key: an Earley chart-state hash paired with the vocabulary size the
/// mask was computed for. See the module doc for why `vocab_size` must be
/// part of the key rather than an incidental property of the stored value.
type CacheKey = (u64, usize);

// ─────────────────────────────────────────────────────────────────────────────
// Internal entry
// ─────────────────────────────────────────────────────────────────────────────

/// A single cached allowed-tokens mask for one Earley state.
///
/// The mask is stored as a reference-counted boxed slice so that callers can
/// cheaply clone the `Arc` rather than copying the entire `Vec<bool>`.
struct CachedMask {
    /// Shared, immutable token mask (true = allowed).
    mask: Arc<[bool]>,
}

// ─────────────────────────────────────────────────────────────────────────────
// AllowedTokensCache
// ─────────────────────────────────────────────────────────────────────────────

/// LRU cache mapping Earley state hashes → token masks.
///
/// Default capacity: 256 entries.  At 150 k tokens (Qwen3 vocab) × 1 byte each,
/// that is ≈38 MB worst-case; typical grammars cycle through far fewer states.
///
/// # Thread safety
///
/// `AllowedTokensCache` is **not** `Sync` on its own.  In `GrammarConstraint`
/// it is wrapped in a `std::sync::Mutex` to satisfy the `&self` signature of
/// `TokenConstraint::allowed_tokens`.
pub struct AllowedTokensCache {
    capacity: usize,
    inner: HashMap<CacheKey, CachedMask>,
    /// LRU order: front = least recently used, back = most recently used.
    lru: VecDeque<CacheKey>,
    /// Total cache hits (for testing / metrics).
    hits: u64,
    /// Total cache misses (for testing / metrics).
    misses: u64,
}

impl AllowedTokensCache {
    /// Create a cache with the given capacity.
    ///
    /// The capacity is clamped to a minimum of 1 to keep invariants simple.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            inner: HashMap::with_capacity(capacity),
            lru: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }

    /// Try to get a cached mask for the given `(state_hash, vocab_size)` pair.
    ///
    /// `vocab_size` is part of the key (not just `state_hash`) so that a mask
    /// built for one vocabulary size is never handed back to a caller that
    /// requested a different one — see the module doc for the rationale.
    ///
    /// On a hit the key is promoted to the back of the LRU queue (most recently
    /// used) and the hit counter is incremented.  On a miss the miss counter is
    /// incremented.
    pub fn get(&mut self, state_hash: u64, vocab_size: usize) -> Option<Arc<[bool]>> {
        let key: CacheKey = (state_hash, vocab_size);
        if let Some(entry) = self.inner.get(&key) {
            // Promote to back of LRU (most recently used).
            if let Some(pos) = self.lru.iter().position(|&k| k == key) {
                self.lru.remove(pos);
            }
            self.lru.push_back(key);
            self.hits += 1;
            Some(Arc::clone(&entry.mask))
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a mask for the given `(state_hash, vocab_size)` pair, evicting
    /// LRU if at capacity.
    ///
    /// If the key is already present the call is a no-op (the existing entry
    /// is kept; this is safe because a single-threaded `Mutex` prevents races).
    pub fn insert(&mut self, state_hash: u64, vocab_size: usize, mask: Vec<bool>) {
        let key: CacheKey = (state_hash, vocab_size);
        if self.inner.contains_key(&key) {
            return;
        }
        if self.inner.len() >= self.capacity {
            // Evict least-recently-used entry.
            if let Some(oldest) = self.lru.pop_front() {
                self.inner.remove(&oldest);
            }
        }
        let mask: Arc<[bool]> = Arc::from(mask.into_boxed_slice());
        self.inner.insert(key, CachedMask { mask });
        self.lru.push_back(key);
    }

    /// Number of cache hits since creation (for testing / observability).
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Number of cache misses since creation (for testing / observability).
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Number of entries currently held in the cache.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mask(v: &[bool]) -> Vec<bool> {
        v.to_vec()
    }

    #[test]
    fn cache_empty_initially() {
        let cache = AllowedTokensCache::with_capacity(4);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_miss_on_empty() {
        let mut cache = AllowedTokensCache::with_capacity(4);
        assert!(cache.get(42, 8).is_none());
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);
    }

    #[test]
    fn cache_insert_and_hit() {
        let mut cache = AllowedTokensCache::with_capacity(4);
        cache.insert(1, 3, make_mask(&[true, false, true]));
        let result = cache.get(1, 3).expect("should be present");
        assert_eq!(&*result, &[true, false, true]);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn cache_duplicate_insert_is_noop() {
        let mut cache = AllowedTokensCache::with_capacity(4);
        cache.insert(7, 1, make_mask(&[true]));
        cache.insert(7, 1, make_mask(&[false])); // should be ignored
        let result = cache.get(7, 1).expect("present");
        // Original value should survive.
        assert_eq!(&*result, &[true]);
    }

    #[test]
    fn cache_evicts_lru_at_capacity() {
        let mut cache = AllowedTokensCache::with_capacity(2);
        cache.insert(10, 1, make_mask(&[true]));
        cache.insert(20, 1, make_mask(&[true]));
        // Access 20 to make 10 the LRU.
        cache.get(20, 1);
        // Insert third entry — 10 should be evicted.
        cache.insert(30, 1, make_mask(&[true]));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(10, 1).is_none(), "10 should have been evicted");
        assert!(cache.get(20, 1).is_some(), "20 should still be present");
        assert!(cache.get(30, 1).is_some(), "30 should be present");
    }

    #[test]
    fn cache_capacity_one_always_evicts() {
        let mut cache = AllowedTokensCache::with_capacity(1);
        cache.insert(1, 1, make_mask(&[true]));
        cache.insert(2, 1, make_mask(&[false]));
        assert_eq!(cache.len(), 1);
        assert!(cache.get(1, 1).is_none());
        assert!(cache.get(2, 1).is_some());
    }

    #[test]
    fn cache_stats_track_correctly() {
        let mut cache = AllowedTokensCache::with_capacity(8);
        cache.get(99, 2); // miss
        cache.get(99, 2); // miss again
        cache.insert(99, 2, make_mask(&[true, true]));
        cache.get(99, 2); // hit
        cache.get(99, 2); // hit
        assert_eq!(cache.misses(), 2);
        assert_eq!(cache.hits(), 2);
    }

    #[test]
    fn cache_lru_promotes_on_hit() {
        // Insert A, B; hit A; insert C → B should be evicted (not A).
        let mut cache = AllowedTokensCache::with_capacity(2);
        cache.insert(1, 1, make_mask(&[true]));
        cache.insert(2, 1, make_mask(&[true]));
        cache.get(1, 1); // promote 1 to MRU
        cache.insert(3, 1, make_mask(&[true])); // should evict 2
        assert!(cache.get(1, 1).is_some(), "1 was promoted, should survive");
        assert!(cache.get(2, 1).is_none(), "2 was LRU, should be evicted");
        assert!(cache.get(3, 1).is_some(), "3 was just inserted");
    }

    // Regression test for issue #82: the same `state_hash` queried with two
    // different `vocab_size` values must not collide in the cache — each
    // `(state_hash, vocab_size)` pair is a distinct entry with its own mask.
    #[test]
    fn cache_same_state_hash_different_vocab_size_do_not_collide() {
        let mut cache = AllowedTokensCache::with_capacity(4);
        cache.insert(1, 3, make_mask(&[true, false, true]));
        cache.insert(1, 5, make_mask(&[false, false, false, true, false]));

        let small = cache.get(1, 3).expect("vocab_size=3 entry must be present");
        assert_eq!(&*small, &[true, false, true]);

        let large = cache.get(1, 5).expect("vocab_size=5 entry must be present");
        assert_eq!(&*large, &[false, false, false, true, false]);

        // Both entries coexist — the cache must not have merged or clobbered them.
        assert_eq!(cache.len(), 2);
    }

    // A miss for one vocab_size at a given state_hash must not be satisfied
    // by an entry cached under the same state_hash but a different vocab_size.
    #[test]
    fn cache_miss_when_vocab_size_differs_from_cached_entry() {
        let mut cache = AllowedTokensCache::with_capacity(4);
        cache.insert(1, 3, make_mask(&[true, false, true]));
        assert!(
            cache.get(1, 7).is_none(),
            "same state_hash but different vocab_size must be a cache miss"
        );
    }
}
