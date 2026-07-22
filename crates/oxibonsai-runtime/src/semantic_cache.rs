//! Semantic caching layer for LLM inference.
//!
//! Returns cached responses for semantically similar queries (above a cosine
//! similarity threshold), avoiding redundant model inference.  The cache uses
//! TF-IDF embeddings and cosine similarity for semantic matching, with LRU-style
//! eviction and TTL-based expiry.
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_runtime::semantic_cache::{CachedInference, SemanticCacheConfig};
//!
//! let config = SemanticCacheConfig::default();
//! let ci = CachedInference::new(config);
//!
//! let (response, was_hit) = ci.run_or_cache(
//!     "What is Rust programming language?",
//!     || "Rust is a systems programming language focused on safety.".to_string(),
//! );
//! assert!(!was_hit);
//!
//! let (response2, was_hit2) = ci.run_or_cache(
//!     "Tell me about the Rust language",
//!     || "Rust is a memory-safe systems language.".to_string(),
//! );
//! // May or may not be a hit depending on similarity
//! let _ = (response2, was_hit2);
//! ```

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use oxibonsai_rag::embedding::{Embedder, TfIdfEmbedder};
use oxibonsai_rag::vector_store::cosine_similarity;

/// Lock `mutex`, recovering from lock poisoning instead of panicking.
///
/// `std::sync::Mutex` poisons permanently once *any* thread panics while
/// holding it, which would otherwise turn one unrelated panic into a
/// process-lifetime outage for every future caller of this cache (a
/// server-reachable, shared component). None of the state guarded by the
/// mutexes in this module has an invariant that a mid-mutation panic could
/// leave "unsafe" to keep using — worst case is a stale/undercounted stat or
/// a partially-updated `Vec` — so recovering the inner value and logging a
/// warning is preferable to permanently wedging the cache.
fn lock_or_recover<'a, T>(mutex: &'a Mutex<T>, what: &'static str) -> MutexGuard<'a, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!(
            lock = what,
            "SemanticCache mutex was poisoned by a prior panic; recovering inner state instead of \
             propagating the panic to this call"
        );
        poisoned.into_inner()
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// SemanticCacheConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for semantic caching.
#[derive(Debug, Clone)]
pub struct SemanticCacheConfig {
    /// Minimum cosine similarity to consider a cache hit (default: 0.92).
    pub similarity_threshold: f32,
    /// Maximum number of cached entries — LRU eviction when exceeded (default: 1000).
    pub max_entries: usize,
    /// TTL for cached entries (default: 1 hour).
    pub ttl: Duration,
    /// Whether to cache streaming responses (default: false).
    pub cache_streaming: bool,
    /// Minimum prompt length in characters to cache; short prompts vary too
    /// much to benefit from semantic caching (default: 20).
    pub min_prompt_chars: usize,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.92,
            max_entries: 1000,
            ttl: Duration::from_secs(3600),
            cache_streaming: false,
            min_prompt_chars: 20,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CachedResponse
// ─────────────────────────────────────────────────────────────────────────────

/// A cached LLM response returned on a semantic cache hit.
#[derive(Debug, Clone)]
pub struct CachedResponse {
    /// The cached response text.
    pub response: String,
    /// The original prompt that produced this response.
    pub prompt: String,
    /// Cosine similarity between the lookup query and the stored prompt.
    pub similarity: f32,
    /// When this cache entry was created.
    pub created_at: Instant,
    /// How many times this entry has been returned as a cache hit.
    pub hit_count: u64,
}

impl CachedResponse {
    /// Returns `true` if this entry is older than `ttl`.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    /// Time elapsed since this entry was created.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CacheEntry (internal)
// ─────────────────────────────────────────────────────────────────────────────

/// Internal storage for a single cached prompt→response pair.
struct CacheEntry {
    prompt: String,
    response: String,
    /// L2-normalised TF-IDF embedding of `prompt`.
    vector: Vec<f32>,
    created_at: Instant,
    /// Monotonically increasing access counter used for LRU ordering.
    last_accessed: u64,
    hit_count: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// SemanticCacheStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics about the cache, suitable for monitoring and dashboards.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SemanticCacheStats {
    /// Total number of lookup attempts (hits + misses).
    pub total_requests: u64,
    /// Number of lookups that returned a cached response.
    pub cache_hits: u64,
    /// Number of lookups that did not find a matching entry.
    pub cache_misses: u64,
    /// Cache hit rate in `[0.0, 1.0]`.
    pub hit_rate: f32,
    /// Current number of entries in the cache.
    pub entries: usize,
    /// Number of LRU-based evictions (capacity exceeded).
    pub evictions: u64,
    /// Number of TTL-based evictions.
    pub expired_evictions: u64,
    /// Mean cosine similarity score across all cache hits.
    pub avg_similarity_on_hit: f32,
}

impl Default for SemanticCacheStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            hit_rate: 0.0,
            entries: 0,
            evictions: 0,
            expired_evictions: 0,
            avg_similarity_on_hit: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SemanticCache
// ─────────────────────────────────────────────────────────────────────────────

/// Semantic cache using TF-IDF embeddings and cosine similarity.
///
/// The cache embeds every incoming prompt with a refittable TF-IDF model and
/// performs a brute-force cosine search over stored entries.  When a result
/// above [`SemanticCacheConfig::similarity_threshold`] is found and has not
/// expired, the stored response is returned without running inference.
///
/// Thread-safety: all fields are guarded by `Mutex`.  The cache is `Send +
/// Sync` and can be shared across threads via `Arc<SemanticCache>`.
pub struct SemanticCache {
    config: SemanticCacheConfig,
    entries: Mutex<Vec<CacheEntry>>,
    embedder: Mutex<TfIdfEmbedder>,
    stats: Mutex<SemanticCacheStats>,
    /// All prompts ever inserted — used to refit the TF-IDF embedder.
    all_prompts: Mutex<Vec<String>>,
    /// Global access clock for LRU ordering.
    access_clock: Mutex<u64>,
    /// Sum of similarity scores across all hits (for computing the mean).
    similarity_sum: Mutex<f64>,
}

/// Embedding dimension used for the bootstrap TF-IDF model (before any prompts
/// have been inserted).  A small positive value avoids zero-dim panics.
const BOOTSTRAP_DIM: usize = 64;

/// Minimum number of new prompts that must accumulate before the embedder is
/// refitted.  Refitting is expensive, so we batch updates.
const REFIT_BATCH_SIZE: usize = 16;

impl SemanticCache {
    /// Create a new [`SemanticCache`] with the given configuration.
    ///
    /// The TF-IDF embedder is bootstrapped with synthetic vocabulary so that
    /// `lookup` calls before any `insert` return gracefully.
    pub fn new(mut config: SemanticCacheConfig) -> Self {
        // `max_entries == 0` has no sensible "unbounded" or "disabled"
        // reading and would otherwise make the very first `insert()` try to
        // evict from an empty `entries` vec (see `insert`'s eviction guard).
        // Clamp to the smallest usable capacity instead of panicking or
        // silently discarding every insert.
        if config.max_entries == 0 {
            tracing::warn!(
                "SemanticCacheConfig::max_entries was 0; clamping to 1 (0 has no valid \
                 'unbounded'/'disabled' semantics for this cache)"
            );
            config.max_entries = 1;
        }

        // Bootstrap embedder: fit on a tiny synthetic corpus so that dim > 0.
        let bootstrap_docs = [
            "hello world query prompt response cache",
            "semantic similarity cosine embedding language model",
            "retrieval augmented generation inference rust",
        ];
        let embedder = TfIdfEmbedder::fit(&bootstrap_docs, BOOTSTRAP_DIM);

        Self {
            config,
            entries: Mutex::new(Vec::new()),
            embedder: Mutex::new(embedder),
            stats: Mutex::new(SemanticCacheStats::default()),
            all_prompts: Mutex::new(Vec::new()),
            access_clock: Mutex::new(0),
            similarity_sum: Mutex::new(0.0),
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Check whether a semantically similar response is cached.
    ///
    /// Returns `None` on a miss, or when the best-matching entry has expired.
    /// On a hit, the entry's `hit_count` and the global access clock are updated.
    pub fn lookup(&self, prompt: &str) -> Option<CachedResponse> {
        if !self.is_cacheable(prompt) {
            let mut stats = lock_or_recover(&self.stats, "stats");
            stats.total_requests += 1;
            stats.cache_misses += 1;
            self.update_hit_rate(&mut stats);
            return None;
        }

        // Embed the query using the current embedder.
        let query_vec = {
            let embedder = lock_or_recover(&self.embedder, "embedder");
            match embedder.embed(prompt) {
                Ok(v) => v,
                Err(_) => {
                    let mut stats = lock_or_recover(&self.stats, "stats");
                    stats.total_requests += 1;
                    stats.cache_misses += 1;
                    self.update_hit_rate(&mut stats);
                    return None;
                }
            }
        };

        let mut entries = lock_or_recover(&self.entries, "entries");
        let ttl = self.config.ttl;
        let threshold = self.config.similarity_threshold;

        // Find the best non-expired match above the threshold.
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx: Option<usize> = None;

        for (idx, entry) in entries.iter().enumerate() {
            if entry.created_at.elapsed() > ttl {
                continue; // skip expired
            }
            if entry.vector.len() != query_vec.len() {
                continue; // dimension mismatch after a refit
            }
            let score = cosine_similarity(&query_vec, &entry.vector);
            if score >= threshold && score > best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }

        let mut stats = lock_or_recover(&self.stats, "stats");
        stats.total_requests += 1;

        match best_idx {
            Some(idx) => {
                // Advance access clock for LRU tracking.
                let clock = {
                    let mut c = lock_or_recover(&self.access_clock, "access_clock");
                    *c += 1;
                    *c
                };
                let entry = &mut entries[idx];
                entry.hit_count += 1;
                entry.last_accessed = clock;

                let response = CachedResponse {
                    response: entry.response.clone(),
                    prompt: entry.prompt.clone(),
                    similarity: best_score,
                    created_at: entry.created_at,
                    hit_count: entry.hit_count,
                };

                stats.cache_hits += 1;
                self.update_hit_rate(&mut stats);

                // Update rolling average similarity.
                {
                    let mut sim_sum = lock_or_recover(&self.similarity_sum, "similarity_sum");
                    *sim_sum += best_score as f64;
                    stats.avg_similarity_on_hit = (*sim_sum / stats.cache_hits as f64) as f32;
                }

                Some(response)
            }
            None => {
                stats.cache_misses += 1;
                self.update_hit_rate(&mut stats);
                None
            }
        }
    }

    /// Store a new `prompt`→`response` mapping in the cache.
    ///
    /// If the cache is at capacity, the least-recently-used entry is evicted.
    /// The TF-IDF embedder is refitted periodically as new prompts accumulate.
    pub fn insert(&self, prompt: &str, response: &str) {
        if !self.is_cacheable(prompt) {
            return;
        }

        // Add to the all_prompts list; refit if we've accumulated enough new ones.
        {
            let mut all_prompts = lock_or_recover(&self.all_prompts, "all_prompts");
            all_prompts.push(prompt.to_string());

            // Refit when: first insertion, or every REFIT_BATCH_SIZE new prompts.
            let should_refit = all_prompts.len() == 1 || all_prompts.len() % REFIT_BATCH_SIZE == 0;
            drop(all_prompts); // release before calling refit_embedder

            if should_refit {
                self.refit_embedder();
            }
        }

        // Embed with the (possibly just refitted) embedder.
        let vector = {
            let embedder = lock_or_recover(&self.embedder, "embedder");
            match embedder.embed(prompt) {
                Ok(v) => v,
                Err(_) => return, // silently skip unembed-able prompts
            }
        };

        let clock = {
            let mut c = lock_or_recover(&self.access_clock, "access_clock");
            *c += 1;
            *c
        };

        let mut entries = lock_or_recover(&self.entries, "entries");

        // Evict LRU entry if at capacity. The `!entries.is_empty()` guard
        // (rather than relying solely on `SemanticCacheConfig::new` clamping
        // `max_entries` away from 0) keeps this branch panic-free even if
        // `max_entries` is ever 0: `entries.len() >= 0` is trivially true on
        // an empty vec, and `min_by_key` over an empty iterator returns
        // `None`, which used to be force-unwrapped via `.expect(...)`.
        if !entries.is_empty() && entries.len() >= self.config.max_entries {
            if let Some(lru_idx) = entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(i, _)| i)
            {
                entries.swap_remove(lru_idx);

                let mut stats = lock_or_recover(&self.stats, "stats");
                stats.evictions += 1;
            }
        }

        entries.push(CacheEntry {
            prompt: prompt.to_string(),
            response: response.to_string(),
            vector,
            created_at: Instant::now(),
            last_accessed: clock,
            hit_count: 0,
        });

        let mut stats = lock_or_recover(&self.stats, "stats");
        stats.entries = entries.len();
    }

    /// Remove all expired entries from the cache.
    ///
    /// Returns the number of entries that were removed.
    pub fn evict_expired(&self) -> usize {
        let ttl = self.config.ttl;
        let mut entries = lock_or_recover(&self.entries, "entries");
        let before = entries.len();
        entries.retain(|e| e.created_at.elapsed() <= ttl);
        let removed = before - entries.len();

        let mut stats = lock_or_recover(&self.stats, "stats");
        stats.expired_evictions += removed as u64;
        stats.entries = entries.len();

        removed
    }

    /// Remove all entries and reset statistics.
    pub fn clear(&self) {
        lock_or_recover(&self.entries, "entries").clear();
        lock_or_recover(&self.all_prompts, "all_prompts").clear();
        *lock_or_recover(&self.similarity_sum, "similarity_sum") = 0.0;
        *lock_or_recover(&self.stats, "stats") = SemanticCacheStats::default();
    }

    /// Current number of entries in the cache.
    pub fn len(&self) -> usize {
        lock_or_recover(&self.entries, "entries").len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot of current cache statistics.
    pub fn stats(&self) -> SemanticCacheStats {
        lock_or_recover(&self.stats, "stats").clone()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Returns `true` if `prompt` is long enough to benefit from caching.
    fn is_cacheable(&self, prompt: &str) -> bool {
        prompt.len() >= self.config.min_prompt_chars
    }

    /// Refit the TF-IDF embedder using all prompts accumulated so far.
    ///
    /// After refitting, the output dimension generally changes (`fit`'s
    /// vocabulary — and therefore `max_features` — grows with the corpus).
    /// Every existing entry's stored vector is re-embedded with the freshly
    /// fit embedder *before* it is installed, so entries stay matchable at
    /// lookup time instead of becoming permanently unmatchable "zombie"
    /// entries that occupy a capacity slot forever (previously they were
    /// only ever removed by TTL expiry or an LRU race that could not
    /// reliably prioritize them, since a stale entry can no longer register
    /// hits to advance its `last_accessed`).
    fn refit_embedder(&self) {
        let all_prompts = lock_or_recover(&self.all_prompts, "all_prompts");
        if all_prompts.is_empty() {
            return;
        }

        // Determine a reasonable max_features: at least BOOTSTRAP_DIM, at most
        // 4× the number of prompts to avoid a huge sparse vocabulary.
        let max_features = BOOTSTRAP_DIM.max(all_prompts.len() * 4).min(4096);

        let doc_refs: Vec<&str> = all_prompts.iter().map(|s| s.as_str()).collect();
        let new_embedder = TfIdfEmbedder::fit(&doc_refs, max_features);
        drop(all_prompts);

        // Re-embed every existing entry against the new embedder. This is a
        // best-effort pass taken without holding the `embedder` mutex (the
        // freshly fit embedder is a local value, not the shared one yet), so
        // it cannot deadlock against `lookup`'s embedder-then-entries lock
        // order. Entries whose prompt can no longer be embedded (e.g. an
        // empty resulting vocabulary overlap) are dropped rather than left
        // behind with a stale vector.
        {
            let mut entries = lock_or_recover(&self.entries, "entries");
            let before = entries.len();
            entries.retain_mut(|entry| match new_embedder.embed(&entry.prompt) {
                Ok(v) => {
                    entry.vector = v;
                    true
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "semantic cache entry could not be re-embedded after refit; evicting"
                    );
                    false
                }
            });
            let dropped = before - entries.len();
            if dropped > 0 {
                let mut stats = lock_or_recover(&self.stats, "stats");
                stats.evictions += dropped as u64;
                stats.entries = entries.len();
            }
        }

        let mut embedder = lock_or_recover(&self.embedder, "embedder");
        *embedder = new_embedder;
    }

    /// Update the `hit_rate` field of `stats` from its raw counters.
    fn update_hit_rate(&self, stats: &mut SemanticCacheStats) {
        stats.hit_rate = if stats.total_requests == 0 {
            0.0
        } else {
            stats.cache_hits as f32 / stats.total_requests as f32
        };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CachedInference
// ─────────────────────────────────────────────────────────────────────────────

/// Middleware wrapper that checks the semantic cache before running inference.
///
/// ```rust
/// use oxibonsai_runtime::semantic_cache::{CachedInference, SemanticCacheConfig};
///
/// let ci = CachedInference::new(SemanticCacheConfig::default());
///
/// // First call: cache miss — closure runs.
/// let (resp, hit) = ci.run_or_cache(
///     "What is the capital of France?",
///     || "Paris is the capital of France.".to_string(),
/// );
/// assert!(!hit);
/// assert_eq!(resp, "Paris is the capital of France.");
/// ```
pub struct CachedInference {
    /// The underlying semantic cache.  Exposed so callers can inspect stats.
    pub cache: SemanticCache,
}

impl CachedInference {
    /// Create a new [`CachedInference`] backed by a freshly initialised cache.
    pub fn new(config: SemanticCacheConfig) -> Self {
        Self {
            cache: SemanticCache::new(config),
        }
    }

    /// Return a cached response if one exists, otherwise invoke `run_inference`
    /// and store its result.
    ///
    /// # Returns
    ///
    /// `(response, was_cache_hit)` — the response string and whether it came
    /// from the cache.
    pub fn run_or_cache<F>(&self, prompt: &str, run_inference: F) -> (String, bool)
    where
        F: FnOnce() -> String,
    {
        // Check cache first.
        if let Some(cached) = self.cache.lookup(prompt) {
            return (cached.response, true);
        }

        // Cache miss: run inference and store the result.
        let response = run_inference();
        self.cache.insert(prompt, &response);
        (response, false)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn short_ttl_config() -> SemanticCacheConfig {
        SemanticCacheConfig {
            ttl: Duration::from_millis(50),
            ..Default::default()
        }
    }

    fn low_threshold_config() -> SemanticCacheConfig {
        SemanticCacheConfig {
            similarity_threshold: 0.1,
            ..Default::default()
        }
    }

    // ── Basic miss / hit ──────────────────────────────────────────────────────

    #[test]
    fn test_semantic_cache_miss_on_empty() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());
        assert!(cache.lookup("What is the meaning of life?").is_none());
    }

    #[test]
    fn test_semantic_cache_exact_match() {
        let cache = SemanticCache::new(low_threshold_config());
        let prompt = "What is the capital of France and why is it important?";
        cache.insert(prompt, "Paris is the capital of France.");
        let result = cache.lookup(prompt);
        assert!(result.is_some(), "exact prompt should hit the cache");
        let cached = result.expect("just asserted Some");
        assert_eq!(cached.response, "Paris is the capital of France.");
        // Exact match should yield similarity ≈ 1.0
        assert!(cached.similarity > 0.9, "similarity={}", cached.similarity);
    }

    #[test]
    fn test_semantic_cache_insert_and_lookup() {
        let config = SemanticCacheConfig {
            similarity_threshold: 0.5,
            ..Default::default()
        };
        let cache = SemanticCache::new(config);
        let prompt = "Explain the concept of machine learning in detail";
        cache.insert(prompt, "Machine learning is a branch of AI.");
        assert_eq!(cache.len(), 1);
        let hit = cache.lookup(prompt);
        assert!(hit.is_some());
    }

    // ── TTL expiry ────────────────────────────────────────────────────────────

    #[test]
    fn test_semantic_cache_ttl_expiry() {
        let config = short_ttl_config();
        let cache = SemanticCache::new(config);
        let prompt = "Tell me everything about neural networks and deep learning";
        cache.insert(prompt, "Neural networks are computational graphs.");
        // Should be a hit immediately.
        assert!(
            cache.lookup(prompt).is_some(),
            "should hit before TTL expires"
        );
        // Wait for TTL to expire.
        std::thread::sleep(Duration::from_millis(100));
        // Should be a miss now.
        assert!(
            cache.lookup(prompt).is_none(),
            "should miss after TTL expires"
        );
    }

    // ── Min prompt length ─────────────────────────────────────────────────────

    #[test]
    fn test_semantic_cache_min_prompt_length() {
        let cache = SemanticCache::new(SemanticCacheConfig::default());
        // Default min_prompt_chars = 20
        let short = "Hi";
        cache.insert(short, "Hello!");
        assert_eq!(cache.len(), 0, "short prompt should not be cached");
        assert!(cache.lookup(short).is_none());
    }

    // ── Evict expired ─────────────────────────────────────────────────────────

    #[test]
    fn test_semantic_cache_evict_expired() {
        let config = short_ttl_config();
        let cache = SemanticCache::new(config);

        for i in 0..5 {
            let prompt = format!(
                "This is a sufficiently long prompt number {} for caching purposes",
                i
            );
            cache.insert(&prompt, "response");
        }
        assert_eq!(cache.len(), 5);

        std::thread::sleep(Duration::from_millis(100));
        let removed = cache.evict_expired();
        assert_eq!(removed, 5, "all entries should have expired");
        assert_eq!(cache.len(), 0);

        let stats = cache.stats();
        assert_eq!(stats.expired_evictions, 5);
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_semantic_cache_stats_hit_rate() {
        let config = low_threshold_config();
        let cache = SemanticCache::new(config);

        let prompt = "Describe the architecture of transformer neural networks in depth";
        cache.insert(prompt, "Transformers use attention mechanisms.");

        // 1 hit
        let _ = cache.lookup(prompt);
        // 1 miss (nothing similar)
        let _ = cache.lookup("Completely unrelated gibberish zzzzzzzz that matches nothing");

        let stats = cache.stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.total_requests, 2);
        assert!(
            (stats.hit_rate - 0.5).abs() < 1e-5,
            "hit_rate={}",
            stats.hit_rate
        );
    }

    // ── Clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_semantic_cache_clear() {
        let config = low_threshold_config();
        let cache = SemanticCache::new(config);

        for i in 0..10 {
            let prompt = format!(
                "This is prompt number {} that is long enough to be cached by the system",
                i
            );
            cache.insert(&prompt, "some response");
        }
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().total_requests, 0);
    }

    // ── CachedInference ───────────────────────────────────────────────────────

    #[test]
    fn test_cached_inference_returns_cached() {
        let config = low_threshold_config();
        let ci = CachedInference::new(config);

        let prompt = "What is Rust and why is it used for systems programming?";
        let (r1, hit1) = ci.run_or_cache(prompt, || "Rust is a systems language.".to_string());
        assert!(!hit1, "first call must be a miss");
        assert_eq!(r1, "Rust is a systems language.");

        let (r2, hit2) = ci.run_or_cache(prompt, || panic!("should not be called"));
        assert!(hit2, "second identical call must be a hit");
        assert_eq!(r2, "Rust is a systems language.");
    }

    #[test]
    fn test_cached_inference_calls_fn_on_miss() {
        let ci = CachedInference::new(SemanticCacheConfig::default());
        let mut called = false;
        let (resp, hit) = ci.run_or_cache(
            "Explain quantum entanglement in detail for a physics student",
            || {
                called = true;
                "Quantum entanglement is a phenomenon…".to_string()
            },
        );
        assert!(!hit);
        assert!(called);
        assert!(!resp.is_empty());
    }

    // ── Config defaults ───────────────────────────────────────────────────────

    #[test]
    fn test_cache_config_defaults() {
        let cfg = SemanticCacheConfig::default();
        assert!((cfg.similarity_threshold - 0.92).abs() < 1e-6);
        assert_eq!(cfg.max_entries, 1000);
        assert_eq!(cfg.ttl, Duration::from_secs(3600));
        assert!(!cfg.cache_streaming);
        assert_eq!(cfg.min_prompt_chars, 20);
    }

    // ── CachedResponse helpers ────────────────────────────────────────────────

    #[test]
    fn test_cached_response_is_expired() {
        let resp = CachedResponse {
            response: "answer".to_string(),
            prompt: "question".to_string(),
            similarity: 0.95,
            created_at: Instant::now(),
            hit_count: 1,
        };
        assert!(!resp.is_expired(Duration::from_secs(60)));
        // Simulate an old entry by checking with a zero duration.
        // Elapsed > 0 so even a zero TTL should be expired.
        std::thread::sleep(Duration::from_millis(1));
        assert!(resp.is_expired(Duration::ZERO));
    }

    // ── Refit re-embedding (finding #51) ──────────────────────────────────────

    /// Regression test: an entry inserted before a dimension-changing TF-IDF
    /// refit must remain matchable afterwards. Before the fix, `refit_embedder`
    /// replaced `self.embedder` without touching any already-stored entry
    /// vectors, so every entry inserted before a refit became a permanently
    /// unmatchable "zombie" the instant the embedder's output dimension
    /// changed (which happens on essentially every refit once distinct
    /// prompts accumulate).
    #[test]
    fn test_semantic_cache_survives_refit_stale_entries_remain_matchable() {
        let config = SemanticCacheConfig {
            similarity_threshold: 0.05,
            ..Default::default()
        };
        let cache = SemanticCache::new(config);

        let first_prompt = "The quick brown fox jumps over the lazy dog near the riverbank";
        cache.insert(first_prompt, "first response");
        // The very first insert always triggers a refit (all_prompts.len() == 1),
        // fit purely on `first_prompt`'s own small vocabulary.

        // Insert 15 more distinct prompts so all_prompts.len() reaches 16,
        // crossing the REFIT_BATCH_SIZE boundary and triggering a second
        // refit against a much larger corpus (almost certainly changing the
        // embedder's output dimension).
        for i in 0..15 {
            let prompt = format!(
                "Distinct filler prompt number {i} covering unrelated vocabulary about topic {i}"
            );
            cache.insert(&prompt, "filler response");
        }

        assert_eq!(cache.len(), 16, "all 16 distinct prompts should be cached");

        let hit = cache.lookup(first_prompt);
        assert!(
            hit.is_some(),
            "entry inserted before a dimension-changing refit must remain matchable afterwards"
        );
        assert_eq!(
            hit.expect("checked is_some above").response,
            "first response"
        );
    }

    // ── max_entries == 0 (finding #71) ────────────────────────────────────────

    /// Regression test: `max_entries: 0` used to make `insert`'s eviction
    /// guard (`entries.len() >= self.config.max_entries`) trivially true even
    /// on an empty `entries` vec, so `min_by_key(..).expect("entries is
    /// non-empty")` panicked on the very first `insert()` call. It must now
    /// either be clamped at construction or handled without panicking.
    #[test]
    fn test_semantic_cache_max_entries_zero_does_not_panic() {
        let config = SemanticCacheConfig {
            max_entries: 0,
            similarity_threshold: 0.05,
            ..Default::default()
        };
        let cache = SemanticCache::new(config);

        for i in 0..5 {
            let prompt =
                format!("Sufficiently long prompt number {i} for a zero max_entries test case");
            cache.insert(&prompt, "resp");
        }

        assert!(
            !cache.is_empty(),
            "max_entries=0 should be clamped to a usable capacity, not silently drop everything"
        );
        assert!(
            cache.len() <= 5,
            "cache should never exceed the number of unique inserts performed"
        );
    }

    // ── Poisoned-lock recovery (finding #70) ──────────────────────────────────

    /// Regression test: a panic on another thread while holding the
    /// `entries` mutex must not turn every subsequent `SemanticCache` call
    /// into a permanent panic. Before the fix, every internal lock site used
    /// `.lock().expect("... poisoned")`, so a single unrelated panic anywhere
    /// that held one of these locks would wedge the whole cache (a
    /// server-reachable, shared component) for the rest of the process
    /// lifetime.
    #[test]
    fn test_semantic_cache_recovers_from_poisoned_entries_lock() {
        let cache = std::sync::Arc::new(SemanticCache::new(low_threshold_config()));

        // Poison the `entries` mutex from a background thread that panics
        // while holding the lock.
        {
            let cache = std::sync::Arc::clone(&cache);
            let handle = std::thread::spawn(move || {
                let _guard = cache.entries.lock().expect("lock for poisoning");
                panic!("intentional panic to poison the entries mutex");
            });
            let result = handle.join();
            assert!(result.is_err(), "background thread should have panicked");
        }

        // The mutex is now poisoned. Operations that touch it must recover
        // instead of panicking.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let prompt = "This prompt is long enough to be cached after lock poisoning recovers";
            cache.insert(prompt, "recovered response");
            cache.lookup(prompt)
        }));

        assert!(
            outcome.is_ok(),
            "operations on a SemanticCache with a poisoned `entries` mutex must not panic"
        );
        let hit = outcome.expect("checked is_ok above");
        assert!(
            hit.is_some(),
            "insert+lookup should still work after poison recovery"
        );
    }
}
