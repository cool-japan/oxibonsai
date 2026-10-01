//! In-process model cache for GGUF files.
//!
//! Avoids reloading model weights for each request by keeping a bounded set of
//! [`ModelEntry`] values in a [`ModelCache`].  The cache uses LRU-like eviction
//! (evict the entry with the longest idle time) when the slot limit is reached.
//!
//! A companion [`ModelWarmup`] helper runs a small number of dummy inference
//! passes on a freshly-loaded engine so that internal caches and JIT paths are
//! primed before the first real request.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use oxibonsai_core::config::Qwen3Config;

use crate::engine::InferenceEngine;
use crate::sampling::SamplingParams;

// ─────────────────────────────────────────────────────────────────────────────
// ModelEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single cached model entry, storing metadata about a loaded model.
///
/// The entry does **not** own the actual weight tensors; those live in the
/// [`InferenceEngine`] that the cache manages externally.  The entry tracks
/// usage statistics so that the cache can decide which entries to evict.
pub struct ModelEntry {
    /// Model configuration extracted from GGUF metadata.
    pub config: Qwen3Config,
    /// Filesystem path to the GGUF file (if known).
    pub model_path: Option<String>,
    /// Wall-clock time at which this entry was first inserted.
    pub loaded_at: Instant,
    /// Wall-clock time of the most recent cache hit for this entry.
    pub last_used: Instant,
    /// Cumulative number of times this entry has been returned from the cache.
    pub use_count: u64,
    /// Estimated resident-memory footprint of the loaded model.
    pub memory_bytes: usize,
}

impl ModelEntry {
    /// Create a new entry stamped with the current time.
    pub fn new(config: Qwen3Config, model_path: Option<String>, memory_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            config,
            model_path,
            loaded_at: now,
            last_used: now,
            use_count: 0,
            memory_bytes,
        }
    }

    /// How long this entry has been in the cache.
    pub fn age(&self) -> Duration {
        self.loaded_at.elapsed()
    }

    /// How long since this entry was last accessed.
    pub fn idle_time(&self) -> Duration {
        self.last_used.elapsed()
    }

    /// Whether this entry has been idle for longer than `ttl`.
    pub fn is_stale(&self, ttl: Duration) -> bool {
        self.idle_time() >= ttl
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelCacheConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`ModelCache`].
#[derive(Debug, Clone)]
pub struct ModelCacheConfig {
    /// Maximum number of model entries to keep in the cache simultaneously.
    pub max_models: usize,
    /// Time-to-live: entries idle longer than this are eligible for eviction.
    pub ttl: Duration,
    /// When `true`, the cache will proactively evict entries when the total
    /// resident memory exceeds `memory_budget_bytes`.
    pub evict_on_memory_pressure: bool,
    /// Optional memory ceiling in bytes.  When the aggregate `memory_bytes` of
    /// all cached entries exceeds this value the least-recently-used entry is
    /// evicted before inserting a new one.
    pub memory_budget_bytes: Option<usize>,
}

impl Default for ModelCacheConfig {
    fn default() -> Self {
        Self {
            max_models: 4,
            ttl: Duration::from_secs(3600),
            evict_on_memory_pressure: true,
            memory_budget_bytes: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelCacheStats
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of cache utilisation metrics, suitable for serialisation to JSON.
#[derive(Debug, serde::Serialize)]
pub struct ModelCacheStats {
    /// Number of entries currently held in the cache.
    pub cached_models: usize,
    /// Cumulative cache hits since the cache was created.
    pub total_hits: u64,
    /// Cumulative cache misses since the cache was created.
    pub total_misses: u64,
    /// Hit rate as a fraction in `[0.0, 1.0]`.
    pub hit_rate: f32,
    /// Sum of `memory_bytes` across all cached entries.
    pub total_memory_bytes: usize,
    /// Age in seconds of the oldest entry, or `None` if the cache is empty.
    pub oldest_entry_age_secs: Option<u64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelCache
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe in-process model cache.
///
/// Uses a [`Mutex`]-guarded [`HashMap`] internally.  Eviction is based on
/// idle time (longest-idle entry is removed first) when the slot or memory
/// budget is exceeded.
pub struct ModelCache {
    entries: Mutex<HashMap<String, ModelEntry>>,
    config: ModelCacheConfig,
    /// Cumulative number of cache hits.
    pub hits: AtomicU64,
    /// Cumulative number of cache misses.
    pub misses: AtomicU64,
}

impl ModelCache {
    /// Create a new, empty cache with the given configuration.
    pub fn new(config: ModelCacheConfig) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            config,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Return a shared reference to the cached entry for `key`, or insert a
    /// new one produced by `loader` if none exists (or if the existing entry
    /// is stale).
    ///
    /// The returned [`Arc`] allows callers to hold a reference to the entry
    /// while the cache mutex is not held.
    pub fn get_or_insert<F>(&self, key: &str, loader: F) -> Arc<ModelEntry>
    where
        F: FnOnce() -> ModelEntry,
    {
        let mut entries = self
            .entries
            .lock()
            .expect("model cache mutex should not be poisoned");

        // Check for a live (non-stale) existing entry.
        if let Some(entry) = entries.get_mut(key) {
            if !entry.is_stale(self.config.ttl) {
                entry.last_used = Instant::now();
                entry.use_count += 1;
                self.hits.fetch_add(1, Ordering::Relaxed);
                // Clone the relevant fields into a new Arc — we cannot hand
                // out a reference into the HashMap while the mutex is held by
                // the caller.
                return Arc::new(ModelEntry {
                    config: entry.config.clone(),
                    model_path: entry.model_path.clone(),
                    loaded_at: entry.loaded_at,
                    last_used: entry.last_used,
                    use_count: entry.use_count,
                    memory_bytes: entry.memory_bytes,
                });
            }
            // Stale — remove and fall through to reload.
            entries.remove(key);
        }

        // Cache miss: invoke the loader.
        self.misses.fetch_add(1, Ordering::Relaxed);
        let new_entry = loader();

        // Evict if necessary before inserting.
        self.evict_if_needed_locked(&mut entries, new_entry.memory_bytes);

        let result = Arc::new(ModelEntry {
            config: new_entry.config.clone(),
            model_path: new_entry.model_path.clone(),
            loaded_at: new_entry.loaded_at,
            last_used: new_entry.last_used,
            use_count: new_entry.use_count,
            memory_bytes: new_entry.memory_bytes,
        });

        entries.insert(key.to_owned(), new_entry);
        result
    }

    /// Return `true` if a non-stale entry exists for `key`.
    pub fn contains(&self, key: &str) -> bool {
        let entries = self
            .entries
            .lock()
            .expect("model cache mutex should not be poisoned");
        entries
            .get(key)
            .map(|e| !e.is_stale(self.config.ttl))
            .unwrap_or(false)
    }

    /// Remove the entry for `key`.  Returns `true` if an entry was removed.
    pub fn evict(&self, key: &str) -> bool {
        let mut entries = self
            .entries
            .lock()
            .expect("model cache mutex should not be poisoned");
        entries.remove(key).is_some()
    }

    /// Remove all entries that have been idle longer than the configured TTL.
    /// Returns the number of entries removed.
    pub fn evict_stale(&self) -> usize {
        let mut entries = self
            .entries
            .lock()
            .expect("model cache mutex should not be poisoned");
        let ttl = self.config.ttl;
        let before = entries.len();
        entries.retain(|_, e| !e.is_stale(ttl));
        before - entries.len()
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .expect("model cache mutex should not be poisoned")
            .len()
    }

    /// `true` when the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cache hit rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no lookups have been performed yet.
    pub fn hit_rate(&self) -> f32 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }
        hits as f32 / total as f32
    }

    /// Sum of `memory_bytes` across all cached entries.
    pub fn total_memory_bytes(&self) -> usize {
        self.entries
            .lock()
            .expect("model cache mutex should not be poisoned")
            .values()
            .map(|e| e.memory_bytes)
            .sum()
    }

    /// Take a statistics snapshot of the current cache state.
    pub fn stats(&self) -> ModelCacheStats {
        let entries = self
            .entries
            .lock()
            .expect("model cache mutex should not be poisoned");
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            hits as f32 / total as f32
        };
        let total_memory_bytes: usize = entries.values().map(|e| e.memory_bytes).sum();
        let oldest_entry_age_secs = entries.values().map(|e| e.age().as_secs()).max();

        ModelCacheStats {
            cached_models: entries.len(),
            total_hits: hits,
            total_misses: misses,
            hit_rate,
            total_memory_bytes,
            oldest_entry_age_secs,
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Evict entries while over capacity or over memory budget.
    ///
    /// Must be called with the mutex already held.
    fn evict_if_needed_locked(
        &self,
        entries: &mut HashMap<String, ModelEntry>,
        incoming_bytes: usize,
    ) {
        // Slot limit.
        while entries.len() >= self.config.max_models {
            Self::evict_lru(entries);
        }

        // Memory budget.
        if self.config.evict_on_memory_pressure {
            if let Some(budget) = self.config.memory_budget_bytes {
                let current: usize = entries.values().map(|e| e.memory_bytes).sum();
                let projected = current.saturating_add(incoming_bytes);
                while projected > budget && !entries.is_empty() {
                    Self::evict_lru(entries);
                }
            }
        }
    }

    /// Remove the entry with the longest idle time (LRU eviction policy).
    fn evict_lru(entries: &mut HashMap<String, ModelEntry>) {
        if entries.is_empty() {
            return;
        }
        let lru_key = entries
            .iter()
            .max_by_key(|(_, e)| {
                // Convert to a comparable integer (microseconds since last use).
                e.idle_time().as_micros()
            })
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            entries.remove(&key);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelWarmup
// ─────────────────────────────────────────────────────────────────────────────

/// Runs a small number of dummy inference passes on a freshly-initialised
/// [`InferenceEngine`] to prime internal allocation caches and JIT paths
/// before the first real request arrives.
///
/// An empty [`warmup_prompt`](Self::warmup_prompt) warms up from a single
/// token resolved from the loaded tokenizer's own special-token map
/// ([`Self::with_tokenizer`], [`WarmupStartToken`]) — never a literal id,
/// which would name `<|im_start|>` in one vocabulary (`151644` in
/// Qwen3's) and an unrelated or out-of-range token in another (Bonsai 2's
/// specials sit at `248044..248076`).
pub struct ModelWarmup {
    /// Number of tokens to generate during the warmup pass.
    pub num_warmup_tokens: usize,
    /// Prompt text fed to the engine during warmup.
    pub warmup_prompt: String,
    /// The token an empty warm-up prompt starts from.
    start_token: WarmupStartToken,
}

/// The single token an empty warm-up prompt feeds, and where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupStartToken {
    /// The loaded tokenizer's `<|im_start|>` special token (the ChatML
    /// turn opener every Qwen-family chat prompt starts with).
    ChatStart(u32),
    /// The tokenizer's BOS id, when it has no `<|im_start|>` but its
    /// vocabulary marks the BOS token special.
    Bos(u32),
    /// The first vocabulary id, `0`, when neither resolves (no tokenizer
    /// was given, or its vocabulary has neither special token): present in
    /// every vocabulary.
    FirstVocabId,
}

impl WarmupStartToken {
    /// The ChatML turn opener this resolution looks up first.
    pub const CHAT_START: &'static str = "<|im_start|>";

    /// Resolve from `tokenizer`'s own special-token map: `<|im_start|>`,
    /// else a special BOS, else [`Self::FirstVocabId`].
    #[must_use]
    pub fn resolve(tokenizer: Option<&crate::tokenizer_bridge::TokenizerBridge>) -> Self {
        let Some(tokenizer) = tokenizer else {
            return Self::FirstVocabId;
        };
        if let Some(id) = tokenizer
            .inner()
            .token_to_id(Self::CHAT_START)
            .filter(|&id| tokenizer.is_special(id))
        {
            return Self::ChatStart(id);
        }
        match tokenizer
            .native()
            .map(oxibonsai_tokenizer::OxiTokenizer::bos_id)
            .filter(|&id| tokenizer.is_special(id))
        {
            Some(id) => Self::Bos(id),
            None => Self::FirstVocabId,
        }
    }

    /// The token id.
    #[must_use]
    pub const fn id(self) -> u32 {
        match self {
            Self::ChatStart(id) | Self::Bos(id) => id,
            Self::FirstVocabId => 0,
        }
    }
}

impl Default for ModelWarmup {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelWarmup {
    /// Create a warmup helper with sensible defaults (32 tokens, generic
    /// prompt, and — until [`Self::with_tokenizer`] resolves one — the first
    /// vocabulary id as the start token of an empty prompt).
    pub fn new() -> Self {
        Self {
            num_warmup_tokens: 32,
            warmup_prompt: "Warm up the inference engine.".to_owned(),
            start_token: WarmupStartToken::FirstVocabId,
        }
    }

    /// Resolve the start token of an empty warm-up prompt from the loaded
    /// tokenizer's special-token map ([`WarmupStartToken::resolve`]).
    pub fn with_tokenizer(mut self, tokenizer: &crate::tokenizer_bridge::TokenizerBridge) -> Self {
        self.start_token = WarmupStartToken::resolve(Some(tokenizer));
        self
    }

    /// The token an empty warm-up prompt starts from.
    pub fn start_token(&self) -> WarmupStartToken {
        self.start_token
    }

    /// The prompt [`Self::run`] feeds an engine whose vocabulary holds
    /// `vocab_size` ids: the warm-up text's first 16 bytes as ids (each
    /// reduced into the vocabulary), or — for an empty text — the single
    /// [`Self::start_token`].
    pub fn prompt_tokens(&self, vocab_size: usize) -> Vec<u32> {
        // Byte ids are already below 32000, the smallest real vocabulary
        // this runtime serves; a tiny synthetic vocabulary folds them in.
        let modulus = u32::try_from(vocab_size)
            .unwrap_or(u32::MAX)
            .clamp(1, 32_000);
        let dummy_tokens: Vec<u32> = self
            .warmup_prompt
            .bytes()
            .take(16)
            .map(|b| u32::from(b) % modulus)
            .collect();
        if dummy_tokens.is_empty() {
            vec![self.start_token.id()]
        } else {
            dummy_tokens
        }
    }

    /// Override the number of warmup tokens.
    pub fn with_tokens(mut self, n: usize) -> Self {
        self.num_warmup_tokens = n;
        self
    }

    /// Override the warmup prompt text.
    pub fn with_prompt(mut self, p: &str) -> Self {
        self.warmup_prompt = p.to_owned();
        self
    }

    /// Execute the warmup passes on `engine` using `params`.
    ///
    /// Generates up to [`ModelWarmup::num_warmup_tokens`] tokens from a small
    /// synthetic token sequence and discards the output.  Returns the elapsed
    /// wall-clock time in milliseconds.
    ///
    /// Errors from the engine are logged as warnings but do **not** propagate —
    /// warmup failure is non-fatal.
    pub fn run(&self, engine: &mut InferenceEngine<'_>, params: &SamplingParams) -> u64 {
        let start = Instant::now();

        // A minimal synthetic prompt: the warm-up text's bytes as ids, or
        // the resolved start token for an empty text.
        let prompt_tokens = self.prompt_tokens(engine.vocab_size());
        if self.warmup_prompt.is_empty() {
            match self.start_token {
                WarmupStartToken::ChatStart(id) => {
                    tracing::debug!(id, "warm-up starts from the tokenizer's <|im_start|>");
                }
                WarmupStartToken::Bos(id) => tracing::info!(
                    id,
                    "warm-up: the tokenizer has no <|im_start|> special token; starting from its \
                     BOS id"
                ),
                WarmupStartToken::FirstVocabId => tracing::info!(
                    "warm-up: no chat-start or BOS special token was resolved (no tokenizer \
                     given, or none in its vocabulary); starting from the first vocabulary id 0"
                ),
            }
        }

        // Temporarily swap in the caller-supplied params via generate_with_seed.
        match engine.generate_with_seed(&prompt_tokens, self.num_warmup_tokens, 0, params) {
            Ok(toks) => {
                tracing::debug!(generated = toks.len(), "warmup pass completed");
            }
            Err(e) => {
                tracing::warn!(error = %e, "warmup pass encountered an error (non-fatal)");
            }
        }

        // Reset state so the engine is clean for real requests.
        engine.reset();

        start.elapsed().as_millis() as u64
    }

    /// Heuristic: should this engine be warmed up?
    ///
    /// The current implementation always returns `true` — callers are
    /// responsible for deciding when to apply warmup (e.g. once after initial
    /// load, or after a cache miss).
    pub fn needs_warmup(_engine: &InferenceEngine<'_>) -> bool {
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::config::Qwen3Config;

    fn tiny_entry() -> ModelEntry {
        ModelEntry::new(
            Qwen3Config::tiny_test(),
            Some(std::env::temp_dir().join("tiny.gguf").display().to_string()),
            1024,
        )
    }

    // ── ModelEntry ────────────────────────────────────────────────────────

    #[test]
    fn test_model_entry_age() {
        let entry = tiny_entry();
        let age = entry.age();
        // Age should be very small (sub-second) immediately after creation.
        assert!(age < Duration::from_secs(1));
    }

    #[test]
    fn test_model_entry_is_stale() {
        let entry = tiny_entry();
        // With a 1-hour TTL the brand-new entry must not be stale.
        assert!(!entry.is_stale(Duration::from_secs(3600)));
        // With a zero TTL every entry is stale.
        assert!(entry.is_stale(Duration::from_nanos(0)));
    }

    // ── ModelCache — miss path ─────────────────────────────────────────────

    #[test]
    fn test_model_cache_miss_calls_loader() {
        let cache = ModelCache::new(ModelCacheConfig::default());
        let mut loader_called = false;

        let _entry = cache.get_or_insert("model-a", || {
            loader_called = true;
            tiny_entry()
        });

        assert!(loader_called, "loader should have been called on a miss");
        assert_eq!(cache.misses.load(Ordering::Relaxed), 1);
        assert_eq!(cache.hits.load(Ordering::Relaxed), 0);
        assert_eq!(cache.len(), 1);
    }

    // ── ModelCache — hit path ──────────────────────────────────────────────

    #[test]
    fn test_model_cache_hit_skips_loader() {
        let cache = ModelCache::new(ModelCacheConfig::default());

        // First call: miss.
        cache.get_or_insert("model-b", tiny_entry);

        // Second call: should be a hit.
        let mut second_loader_called = false;
        cache.get_or_insert("model-b", || {
            second_loader_called = true;
            tiny_entry()
        });

        assert!(!second_loader_called, "loader must not be called on a hit");
        assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
        assert_eq!(cache.misses.load(Ordering::Relaxed), 1);
    }

    // ── ModelCache — manual eviction ──────────────────────────────────────

    #[test]
    fn test_model_cache_evict() {
        let cache = ModelCache::new(ModelCacheConfig::default());
        cache.get_or_insert("model-c", tiny_entry);
        assert!(cache.contains("model-c"));

        let removed = cache.evict("model-c");
        assert!(removed);
        assert!(!cache.contains("model-c"));
        assert_eq!(cache.len(), 0);

        // Evicting a non-existent key returns false.
        assert!(!cache.evict("no-such-model"));
    }

    // ── ModelCache — stale eviction ──────────────────────────────────────

    #[test]
    fn test_model_cache_evict_stale() {
        // Use a zero TTL so every entry is immediately stale.
        let cfg = ModelCacheConfig {
            ttl: Duration::from_nanos(0),
            ..Default::default()
        };
        let cache = ModelCache::new(cfg);

        // Insert via get_or_insert so the entry lands in the map.
        {
            let mut entries = cache.entries.lock().expect("mutex should not be poisoned");
            entries.insert("model-d".to_owned(), tiny_entry());
        }

        assert_eq!(cache.len(), 1);
        let evicted = cache.evict_stale();
        assert_eq!(evicted, 1);
        assert_eq!(cache.len(), 0);
    }

    // ── ModelCache — hit rate ─────────────────────────────────────────────

    #[test]
    fn test_model_cache_hit_rate() {
        let cache = ModelCache::new(ModelCacheConfig::default());

        // No lookups yet → 0.0.
        assert!((cache.hit_rate() - 0.0).abs() < f32::EPSILON);

        cache.get_or_insert("rate-model", tiny_entry); // miss
        cache.get_or_insert("rate-model", tiny_entry); // hit
        cache.get_or_insert("rate-model", tiny_entry); // hit

        // 2 hits out of 3 total → ~0.667
        let rate = cache.hit_rate();
        assert!(rate > 0.6 && rate < 0.7, "expected ~0.667, got {rate}");
    }

    // ── ModelCache — stats snapshot ───────────────────────────────────────

    #[test]
    fn test_model_cache_stats() {
        let cache = ModelCache::new(ModelCacheConfig::default());
        cache.get_or_insert("stats-model", tiny_entry); // miss

        let stats = cache.stats();
        assert_eq!(stats.cached_models, 1);
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.total_hits, 0);
        assert_eq!(stats.total_memory_bytes, 1024);
        assert!(stats.oldest_entry_age_secs.is_some());
    }

    // ── ModelWarmup ───────────────────────────────────────────────────────

    #[test]
    fn test_warmup_runs_without_panic() {
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let mut engine = InferenceEngine::new(config, params.clone(), 42);

        let warmup = ModelWarmup::new().with_tokens(4).with_prompt("Hello");
        let elapsed_ms = warmup.run(&mut engine, &params);

        // Warmup must complete (even if it generates 0 tokens on a tiny model).
        // We just check it didn't panic and returned a sensible elapsed time.
        assert!(elapsed_ms < 60_000, "warmup should complete in under 60 s");
        assert!(ModelWarmup::needs_warmup(&engine));
    }

    /// A minimal byte-level `tokenizer.json` whose added tokens include
    /// `<|im_start|>` as a special token at id 12.
    const CHATML_TOKENIZER_JSON: &str = r##"{
        "model": {
            "type": "BPE",
            "vocab": { "a": 0, "b": 1, "c": 2, "Ġ": 3 },
            "merges": []
        },
        "added_tokens": [
            { "id": 12, "content": "<|im_start|>", "special": true },
            { "id": 13, "content": "<|im_end|>", "special": true }
        ],
        "pre_tokenizer": { "type": "ByteLevel" },
        "decoder": { "type": "ByteLevel" }
    }"##;

    /// An empty warm-up text starts from the loaded tokenizer's own
    /// `<|im_start|>` — whatever id its vocabulary gives it — never a
    /// literal from another vocabulary.
    #[test]
    fn an_empty_warmup_starts_from_the_tokenizers_chat_start_token() {
        let tokenizer =
            crate::tokenizer_bridge::TokenizerBridge::native_from_json_str(CHATML_TOKENIZER_JSON)
                .expect("tokenizer fixture loads");
        let warmup = ModelWarmup::new()
            .with_prompt("")
            .with_tokenizer(&tokenizer);
        assert_eq!(warmup.start_token(), WarmupStartToken::ChatStart(12));
        assert_eq!(warmup.prompt_tokens(64), vec![12]);

        // It drives a real engine without error (vocabulary 64 > id 12).
        let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy(), 42);
        let _ = warmup.run(&mut engine, &greedy());
        assert_eq!(
            engine.sequence_position(),
            0,
            "warm-up leaves the engine reset"
        );
    }

    /// Without a resolvable chat-start token the warm-up falls back to a
    /// special BOS when the vocabulary has one, else to the first
    /// vocabulary id — both present in the vocabulary the engine serves.
    #[test]
    fn an_empty_warmup_falls_back_to_the_bos_or_the_first_vocabulary_id() {
        use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

        // No tokenizer at all.
        assert_eq!(
            WarmupStartToken::resolve(None),
            WarmupStartToken::FirstVocabId
        );
        let bare = ModelWarmup::new().with_prompt("");
        assert_eq!(bare.start_token(), WarmupStartToken::FirstVocabId);
        assert_eq!(bare.prompt_tokens(32), vec![0]);

        // A vocabulary with no special tokens at all: its configured BOS id
        // (the default 1) is an ordinary token, so it is not used.
        let plain = crate::tokenizer_bridge::TokenizerBridge::native_from_json_str(
            r##"{
                "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "c": 2 }, "merges": [] },
                "added_tokens": [],
                "pre_tokenizer": { "type": "ByteLevel" },
                "decoder": { "type": "ByteLevel" }
            }"##,
        )
        .expect("plain tokenizer loads");
        assert_eq!(
            WarmupStartToken::resolve(Some(&plain)),
            WarmupStartToken::FirstVocabId
        );
        // The character-level stub declares a special `<bos>` at id 1.
        let stub = crate::tokenizer_bridge::TokenizerBridge::from_native_tokenizer(
            oxibonsai_tokenizer::OxiTokenizer::char_level_stub(32),
        );
        assert_eq!(
            WarmupStartToken::resolve(Some(&stub)),
            WarmupStartToken::Bos(1)
        );

        // A GGUF vocabulary whose BOS (`<|endoftext|>`, a control token) is
        // special but which has no `<|im_start|>`.
        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "tokenizer.ggml.model",
            MetadataWriteValue::Str("gpt2".to_string()),
        );
        writer.add_metadata(
            "tokenizer.ggml.tokens",
            MetadataWriteValue::ArrayStr(vec![
                "a".to_string(),
                "b".to_string(),
                "<|endoftext|>".to_string(),
            ]),
        );
        writer.add_metadata(
            "tokenizer.ggml.token_type",
            MetadataWriteValue::ArrayI32(vec![1, 1, 3]),
        );
        writer.add_metadata(
            "tokenizer.ggml.merges",
            MetadataWriteValue::ArrayStr(Vec::new()),
        );
        writer.add_metadata("tokenizer.ggml.bos_token_id", MetadataWriteValue::U32(2));
        writer.add_metadata("tokenizer.ggml.eos_token_id", MetadataWriteValue::U32(2));
        let bytes = writer.to_bytes().expect("serialize");
        let gguf =
            oxibonsai_core::gguf::reader::GgufFile::parse(&bytes).expect("tokenizer-only GGUF");
        let tokenizer =
            crate::tokenizer_bridge::TokenizerBridge::native_from_gguf_metadata(&gguf.metadata)
                .expect("GGUF vocabulary loads");
        let warmup = ModelWarmup::new()
            .with_prompt("")
            .with_tokenizer(&tokenizer);
        assert_eq!(warmup.start_token(), WarmupStartToken::Bos(2));
        assert_eq!(warmup.prompt_tokens(32), vec![2]);

        // A non-empty text never consults the start token, and its byte ids
        // fold into a tiny vocabulary.
        let text = ModelWarmup::new()
            .with_prompt("zz")
            .with_tokenizer(&tokenizer);
        assert_eq!(text.prompt_tokens(32), vec![u32::from(b'z') % 32; 2]);
        assert_eq!(text.prompt_tokens(151_936), vec![u32::from(b'z'); 2]);
    }

    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        }
    }
}
