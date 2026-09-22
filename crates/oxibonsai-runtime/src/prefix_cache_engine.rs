//! Prefix-cache-aware inference engine wrapper.
//!
//! [`PrefixCachedEngine`] wraps an [`InferenceEngine`] and transparently
//! intercepts the prefill phase: identical prompt prefixes (e.g. a shared
//! system prompt) are served from the KV-cache trie rather than being
//! re-processed by the model, cutting prefill cost to near-zero for cached
//! prefixes.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oxibonsai_core::config::Qwen3Config;
//! use oxibonsai_runtime::engine::InferenceEngine;
//! use oxibonsai_runtime::sampling::SamplingParams;
//! use oxibonsai_runtime::prefix_cache_engine::PrefixCachedEngine;
//!
//! let config = Qwen3Config::tiny_test();
//! let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
//! let mut cached = PrefixCachedEngine::new(engine, 64, 42);
//!
//! let tokens = cached.generate(&[1, 2, 3, 4], &SamplingParams::default());
//! let stats = cached.cache_stats();
//! println!("hit rate: {:.1}%", stats.hit_rate * 100.0);
//! ```
//!
//! ## Limitations (M-35)
//!
//! Real prefix-cache reuse is only correct when the engine's forward path
//! populates the CPU [`oxibonsai_model::KvCache`]. Two cases where it does
//! not are handled by refusing to use the cache-aware path at all, rather
//! than by probing after the fact and hoping the probe is representative:
//!
//! * **A GPU-resident KV tier**, detected by asking the engine's own kernel
//!   dispatcher directly
//!   ([`oxibonsai_kernels::traits::OneBitKernel::is_gpu_accelerated`] on
//!   [`InferenceEngine::kernel`]) rather than re-deriving the same
//!   feature-cfg condition locally — the two can then never silently drift
//!   apart. Metal/CUDA forward paths keep KV state on the device, separate
//!   from the CPU cache this trie reads from and writes into; injecting a
//!   cached prefix into the CPU cache would then be inert (the GPU forward
//!   never reads it), and extracting "new" blocks afterwards would capture
//!   zeros — silently and permanently poisoning the trie for every later
//!   request sharing that prefix. [`PrefixCachedEngine::generate`] detects
//!   this tier up front and falls back to a plain, uncached generation that
//!   is always correct.
//! * **A hybrid (recurrent-attention) architecture.** A model whose layers
//!   carry Gated-DeltaNet-style recurrent state (e.g. Bonsai 2 27B, GGUF
//!   `general.architecture = "qwen35"`) has no block-KV representation for
//!   that state at all — there is nothing to extract or inject, so prefix
//!   caching is refused unconditionally for such a model (to be recorded in
//!   TODO.md by B2-21-DOCS, the docs package; tracked upstream as B2-12
//!   wiring the real recurrent-state cache).
//!
//! Within the cache-aware path, every candidate block is checksummed against
//! its own live extracted content — not against an unrelated sample range —
//! before it is allowed into the trie; see
//! [`PrefixCachedEngine::store_new_blocks`].
//!
//! ## The "ideal fix" ([`cpu_kv_is_authoritative`], M-35 addendum)
//!
//! The checksum in [`block_has_real_content`] is a *fallback*: it infers
//! "was this really written by the CPU forward path" from the data itself
//! (all-zero is suspicious), which is reliable but, by construction, cannot
//! catch a forward path that writes *plausible-looking but stale* data.
//! [`cpu_kv_is_authoritative`] is the honest alternative the M-35 finding's
//! verifier correction asks for: it reads
//! [`BonsaiModel::gpu_path_active`](oxibonsai_model::model::BonsaiModel::gpu_path_active)
//! — already set truthfully by every forward path
//! (`forward_prefill`/`forward_into`/`mark_host_kv_rebuilt`/
//! `note_host_kv_written`, none of which this package owns, but all of
//! which already exist and are already `pub`) — combined with the same
//! hybrid-architecture check `generate()` uses. `store_new_blocks` consults
//! **both** this flag and the checksum; neither replaces the other.

use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_model::prefix_cache::{
    KvBlockPair, PrefixAwarePrefill, PrefixCache, PrefixCacheStats,
};

use crate::engine::InferenceEngine;
use crate::sampling::{Sampler, SamplingParams};

/// Whether `model`'s CPU-resident `KvCache` currently holds the true,
/// authoritative KV history for every position already processed (M-35's
/// "ideal fix" — see the module docs).
///
/// `false` in exactly the two cases [`PrefixCachedEngine::generate`]'s
/// up-front gate already refuses the cache-aware path for: a GPU-resident
/// forward path (`gpu_path_active()`, tracked honestly per forward call —
/// distinct from and more precise than that gate's own
/// [`OneBitKernel::is_gpu_accelerated`], which asks "can this kernel tier
/// run on the GPU at all" rather than "did the last forward call actually
/// use it") and a hybrid recurrent-attention architecture (no block-KV
/// representation exists to be authoritative about in the first place).
fn cpu_kv_is_authoritative(model: &BonsaiModel) -> bool {
    !model.gpu_path_active() && model.config().architecture != HYBRID_RECURRENT_ARCHITECTURE
}

/// Tokens per cache block — must divide evenly into most prompt lengths.
const BLOCK_SIZE: usize = 16;

/// GGUF `general.architecture` tag for the hybrid model family that mixes
/// recurrent linear-attention layers with full-attention layers (Bonsai 2
/// 27B). No `oxibonsai-model`/`oxibonsai-core` file this package owns
/// defines a shared constant for it yet, so it is spelled out here — see the
/// module docs' M-35 note on why prefix caching is refused unconditionally
/// for it (recurrent state has no block-KV representation to extract or
/// inject).
const HYBRID_RECURRENT_ARCHITECTURE: &str = "qwen35";

/// An [`InferenceEngine`] augmented with prefix KV-cache reuse.
///
/// On each [`generate`](PrefixCachedEngine::generate) call, unless the
/// engine's KV tier is GPU-resident or the model is a hybrid
/// recurrent-attention architecture (see the module docs — M-35), the
/// engine:
///
/// 1. Resets the model's KV cache (single-engine, sequential request model).
/// 2. Looks up the longest cached prefix in the trie.
/// 3. Injects the matched KV blocks back into the model's CPU cache.
/// 4. Runs prefill only on the uncached suffix at the correct `pos_start`.
/// 5. Checksums and stores any newly computed *and verified non-zero* full
///    blocks of KV state into the trie for subsequent requests.
/// 6. Sample-decodes new tokens up to `params.max_tokens` or EOS.
/// 7. Releases the session (decrements ref counts) when done.
///
/// In the refused cases, steps 2/3/5/7 (everything cache-specific) are
/// skipped entirely and step 4 prefills the whole prompt from position 0;
/// steps 1 and 6 still run.
pub struct PrefixCachedEngine<'a> {
    /// The underlying inference engine.
    pub inner: InferenceEngine<'a>,
    /// Prefix-cache-aware prefill helper with the block trie.
    pub prefix_cache: PrefixAwarePrefill,
    /// Per-request decode sampler, seeded once at construction and reused
    /// (state-advancing) across every `generate()` call — mirrors
    /// [`InferenceEngine::generate_with_params`]'s "swap params, keep RNG
    /// state" pattern rather than re-seeding on every call.
    sampler: Sampler,
}

impl<'a> PrefixCachedEngine<'a> {
    /// Wrap an existing [`InferenceEngine`] with a prefix cache.
    ///
    /// Derives `num_layers`, `num_kv_heads`, and `head_dim` directly from
    /// the engine's model configuration, so no manual wiring is required.
    ///
    /// # Parameters
    ///
    /// - `engine` — the inference engine to wrap.
    /// - `max_cache_blocks` — maximum number of simultaneously live cache
    ///   blocks.  Each block holds `BLOCK_SIZE` (16) tokens of KV data for
    ///   every layer; memory per block is approximately
    ///   `2 × num_layers × num_kv_heads × head_dim × 16 × 4` bytes.
    /// - `seed` — RNG seed for the wrapper's own decode sampler. `0` is
    ///   remapped to a fixed nonzero constant because the sampler's xorshift64
    ///   PRNG has `0` as a fixed point (it would otherwise generate the same
    ///   "random" value forever), matching the precedent in
    ///   `crate::speculative`.
    pub fn new(engine: InferenceEngine<'a>, max_cache_blocks: usize, seed: u64) -> Self {
        let cfg = engine.model().config();
        let cache = PrefixCache::new(
            max_cache_blocks,
            BLOCK_SIZE,
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
        );
        let prefix_cache = PrefixAwarePrefill::new(cache);
        let effective_seed = if seed == 0 { 0xdeadbeef_cafebabe } else { seed };
        let sampler = Sampler::new(SamplingParams::default(), effective_seed);
        Self {
            inner: engine,
            prefix_cache,
            sampler,
        }
    }

    /// Generate tokens from `prompt_tokens`, reusing any cached prefix when
    /// it is safe to do so (M-35 — see the module docs for the two refusal
    /// cases).
    ///
    /// Returns the generated token IDs (not including the prompt). On any
    /// internal error the method logs via `tracing::warn` and returns an
    /// empty vector — `generate` itself is infallible from the caller's
    /// perspective so it can be dropped into batch pipelines.
    pub fn generate(&mut self, prompt_tokens: &[u32], params: &SamplingParams) -> Vec<u32> {
        if prompt_tokens.is_empty() {
            return vec![];
        }

        // M-35: refuse the cache-aware path entirely rather than risk
        // poisoning the trie (GPU-resident KV) or attempting something that
        // has no block-KV representation at all (hybrid recurrent state).
        // See the module docs for both cases. GPU-residency is read straight
        // off the engine's own kernel dispatcher — the exact predicate
        // `BonsaiModel::forward`/`prefill` use to pick the Metal path —
        // rather than re-derived locally, so this refusal cannot silently
        // drift out of sync with the forward path it is guarding against.
        if self.inner.kernel().is_gpu_accelerated()
            || self.inner.model().config().architecture == HYBRID_RECURRENT_ARCHITECTURE
        {
            return self.generate_without_prefix_cache(prompt_tokens, params);
        }

        // ── Step 1: reset model KV cache ─────────────────────────────────────
        // We treat the wrapper as a single-engine, sequential request server.
        self.inner.model_mut().reset();

        // ── Step 2: query the prefix cache ───────────────────────────────────
        let (session, uncached_start) = self.prefix_cache.prepare(prompt_tokens);
        let block_size = self.prefix_cache.cache.block_size();
        let num_layers = self.inner.model().config().num_layers;

        // ── Step 3: restore cached blocks into the model's CPU KV cache ──────
        if uncached_start > 0 && !session.block_indices.is_empty() {
            for (block_num, &bidx) in session.block_indices.iter().enumerate() {
                if bidx == usize::MAX {
                    continue;
                }
                // Snapshot keys/values per layer before mutably borrowing model.
                let snapshots: Option<Vec<(Vec<f32>, Vec<f32>)>> =
                    self.prefix_cache.cache.get_block(bidx).map(|block| {
                        (0..num_layers)
                            .map(|l| (block.keys[l].clone(), block.values[l].clone()))
                            .collect()
                    });
                let snapshots = match snapshots {
                    Some(s) => s,
                    None => continue,
                };
                let block_start = block_num * block_size;
                let kv = self.inner.model_mut().kv_cache_mut();
                for (layer, (keys, values)) in snapshots.into_iter().enumerate() {
                    kv.inject_block(layer, block_start, block_size, &keys, &values);
                }
            }
            self.inner
                .model_mut()
                .kv_cache_mut()
                .set_seq_len(uncached_start);
        }

        // ── Step 4: prefill on the uncached suffix only ──────────────────────
        let last_logits = if uncached_start < prompt_tokens.len() {
            match self
                .inner
                .prefill_from_pos(&prompt_tokens[uncached_start..], uncached_start)
            {
                Ok(logits) => logits,
                Err(e) => {
                    tracing::warn!(error = %e, "prefix-cache prefill failed");
                    self.prefix_cache.release_session(session);
                    return vec![];
                }
            }
        } else {
            // Entire prompt was cached — re-run the final token to get logits
            // (we still need a fresh logits vector to drive the decode loop).
            let last_pos = prompt_tokens.len().saturating_sub(1);
            let last_tok = prompt_tokens[last_pos];
            match self.inner.decode_step(last_tok, last_pos) {
                Ok(logits) => logits,
                Err(e) => {
                    tracing::warn!(error = %e, "prefix-cache decode_step failed");
                    self.prefix_cache.release_session(session);
                    return vec![];
                }
            }
        };

        // ── Step 5/6: checksum-gated store of newly computed blocks ──────────
        self.store_new_blocks(prompt_tokens, uncached_start, block_size, num_layers);

        // ── Step 7: decode loop ──────────────────────────────────────────────
        let output = self.decode_loop(prompt_tokens.len(), last_logits, params);

        // ── Step 8: release session ──────────────────────────────────────────
        self.prefix_cache.release_session(session);
        output
    }

    /// Plain, uncached generation: full prefill from position 0, then the
    /// same seeded decode loop `generate()` uses. Used when the cache-aware
    /// path is refused outright (M-35 — GPU-resident KV tier, or a hybrid
    /// recurrent-attention model).
    fn generate_without_prefix_cache(
        &mut self,
        prompt_tokens: &[u32],
        params: &SamplingParams,
    ) -> Vec<u32> {
        self.inner.model_mut().reset();
        let last_logits = match self.inner.prefill_from_pos(prompt_tokens, 0) {
            Ok(logits) => logits,
            Err(e) => {
                tracing::warn!(error = %e, "prefix-cache bypass: plain prefill failed");
                return vec![];
            }
        };
        self.decode_loop(prompt_tokens.len(), last_logits, params)
    }

    /// Checksum every candidate new block against its own live extracted
    /// content and store only the ones that are genuinely non-zero (M-35).
    ///
    /// The previous implementation sampled `(layer 0, head 0)` over the
    /// **whole prompt** `0..prompt_len` to decide whether the CPU cache was
    /// "real", then used that single answer to gate storing a *different*
    /// range (`uncached_start..uncached_start + n*block_size`, the newly
    /// prefilled suffix). The two ranges coincide only when
    /// `uncached_start == 0`; whenever a request injects a cached prefix
    /// (making `0..uncached_start` genuinely non-zero) and the *suffix*
    /// prefill takes a path that leaves the CPU cache untouched, the old
    /// probe would see the injected prefix, pass, and store all-zero suffix
    /// blocks as if they were real KV — permanently, for the process
    /// lifetime.
    ///
    /// This checksums the *exact* keys/values about to be stored, for every
    /// layer of every candidate block, not a one-layer sample of an
    /// unrelated range — the near-zero probability of a genuinely computed
    /// RMSNorm-scaled key/value vector being all-zero makes "any nonzero
    /// value across the whole block, in every layer" a reliable real-content
    /// check. The first untrustworthy block stops the scan (blocks are
    /// stored as an unbroken run starting at `uncached_start`, so a later,
    /// individually-genuine block cannot be spliced in behind a gap without
    /// misrepresenting the trie's token alignment).
    fn store_new_blocks(
        &mut self,
        prompt_tokens: &[u32],
        uncached_start: usize,
        block_size: usize,
        num_layers: usize,
    ) {
        let new_blocks_count = prompt_tokens.len().saturating_sub(uncached_start) / block_size;
        if new_blocks_count == 0 {
            return;
        }

        // M-35 addendum: consult the honest, per-forward-path-tracked flag
        // *in addition to* the checksum below (never instead of it — the
        // checksum also catches a partially-populated block within an
        // otherwise-authoritative run, which this coarse, whole-model flag
        // cannot see). See `cpu_kv_is_authoritative`'s doc for why this is
        // additive rather than a replacement for `generate()`'s existing
        // up-front `is_gpu_accelerated()` gate.
        if !cpu_kv_is_authoritative(self.inner.model()) {
            tracing::warn!(
                uncached_start,
                "prefix-cache: skipping store — the model's CPU KV cache is not \
                 authoritative right now (GPU-resident forward path, or a hybrid \
                 recurrent-attention architecture)"
            );
            return;
        }

        let mut keys_by_block: Vec<KvBlockPair> = Vec::with_capacity(new_blocks_count);
        for blk in 0..new_blocks_count {
            let block_pos = uncached_start + blk * block_size;
            let mut layer_keys: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
            let mut layer_values: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
            for layer in 0..num_layers {
                let (k, v) = self
                    .inner
                    .model()
                    .kv_cache()
                    .extract_block(layer, block_pos, block_size);
                layer_keys.push(k);
                layer_values.push(v);
            }
            if !block_has_real_content(&layer_keys, &layer_values) {
                tracing::warn!(
                    block_pos,
                    "prefix-cache: skipping all-zero KV block (the CPU cache is not \
                     authoritative for this range, likely a GPU-resident forward path) \
                     instead of poisoning the trie"
                );
                break;
            }
            keys_by_block.push((layer_keys, layer_values));
        }

        if !keys_by_block.is_empty() {
            self.prefix_cache
                .store_blocks(prompt_tokens, uncached_start, keys_by_block);
        }
    }

    /// Shared decode loop used by both the cache-aware and the bypass path.
    ///
    /// Swaps `params` onto the wrapper's own seeded sampler (constructed
    /// once in `new()` from the caller-supplied seed) so that per-call
    /// sampling parameters are honoured while the PRNG state keeps
    /// advancing across calls — the same "swap params, keep RNG state"
    /// pattern [`InferenceEngine::generate_with_params`] uses, rather than
    /// discarding the configured seed and re-seeding at 0 every call.
    fn decode_loop(
        &mut self,
        prompt_len: usize,
        mut last_logits: Vec<f32>,
        params: &SamplingParams,
    ) -> Vec<u32> {
        let prev_params = self.sampler.params().clone();
        self.sampler.set_params(params.clone());
        let mut output = Vec::with_capacity(params.max_tokens);
        for (pos, _) in (prompt_len..).zip(0..params.max_tokens) {
            let next_token = match self.sampler.sample(&last_logits) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(error = %e, "prefix-cache sampler error");
                    break;
                }
            };
            if next_token == self.inner.eos_token_id() {
                break;
            }
            output.push(next_token);
            last_logits = match self.inner.decode_step(next_token, pos) {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!(error = %e, "prefix-cache decode loop error");
                    break;
                }
            };
        }
        self.sampler.set_params(prev_params);
        output
    }

    /// Return a snapshot of the current prefix-cache statistics.
    pub fn cache_stats(&self) -> PrefixCacheStats {
        self.prefix_cache.stats()
    }

    /// Clear all entries from the prefix cache.
    ///
    /// Does *not* reset the inner engine's KV cache.
    pub fn clear_cache(&mut self) {
        self.prefix_cache.cache.clear();
    }
}

/// Whether a candidate cache block's extracted keys/values look like real,
/// live-computed content rather than the zeros a non-CPU-authoritative
/// forward path would leave behind (M-35). Every layer's keys *and* values
/// must contain at least one nonzero value — a single all-zero layer among
/// several is enough to reject the whole block, since a partially-populated
/// block (some layers written by a CPU fallback mid-prompt, others left
/// zero by a GPU path) is exactly the "half the block is zeros" failure mode
/// the finding described.
fn block_has_real_content(layer_keys: &[Vec<f32>], layer_values: &[Vec<f32>]) -> bool {
    if layer_keys.is_empty() || layer_values.is_empty() {
        return false;
    }
    layer_keys.iter().all(|k| k.iter().any(|&x| x != 0.0))
        && layer_values.iter().all(|v| v.iter().any(|&x| x != 0.0))
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::config::{Qwen3Config, RopeScaling};
    use oxibonsai_model::model::BonsaiModel;

    fn make_engine_no_blocks(max_blocks: usize) -> PrefixCachedEngine<'static> {
        let config = Qwen3Config::tiny_test();
        let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        PrefixCachedEngine::new(engine, max_blocks, 42)
    }

    /// Build a config small enough to keep test runtimes tight while still
    /// satisfying the Q1_0_g128 constraint (in_features must be a multiple
    /// of 128).
    fn small_real_config() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 128,
            intermediate_size: 256,
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            value_length: 32,
            vocab_size: 256,
            max_context_length: 128,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10_000.0,
            rope_scaling: RopeScaling::None,
            sliding_window: None,
            architecture: "qwen3".to_string(),
            model_name: "PrefixCacheTest".to_string(),
        }
    }

    fn make_engine_with_real_blocks(max_blocks: usize) -> PrefixCachedEngine<'static> {
        use oxibonsai_kernels::{KernelDispatcher, KernelTier};
        let config = small_real_config();
        let model = BonsaiModel::new_for_testing_with_blocks(config);
        // Pin the engine to the Reference (CPU) tier so the CPU KV cache is
        // populated by the forward path. With auto_detect on a GPU host the
        // GPU shortcut would bypass the CPU cache entirely.
        let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
        let engine =
            InferenceEngine::from_model_with_kernel(model, kernel, SamplingParams::default(), 42);
        PrefixCachedEngine::new(engine, max_blocks, 42)
    }

    /// Same as [`make_engine_with_real_blocks`] but with
    /// `architecture = "qwen35"` (the hybrid recurrent-attention tag) — used
    /// to exercise M-35's second refusal case.
    fn make_hybrid_engine_with_real_blocks(max_blocks: usize) -> PrefixCachedEngine<'static> {
        use oxibonsai_kernels::{KernelDispatcher, KernelTier};
        let config = Qwen3Config {
            architecture: "qwen35".to_string(),
            ..small_real_config()
        };
        let model = BonsaiModel::new_for_testing_with_blocks(config);
        let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
        let engine =
            InferenceEngine::from_model_with_kernel(model, kernel, SamplingParams::default(), 42);
        PrefixCachedEngine::new(engine, max_blocks, 42)
    }

    #[test]
    fn prefix_cached_engine_construction() {
        let engine = make_engine_no_blocks(16);
        let stats = engine.cache_stats();
        assert_eq!(stats.cached_blocks, 0);
        assert_eq!(stats.capacity_blocks, 16);
    }

    #[test]
    fn prefix_cached_engine_generate_empty() {
        let mut engine = make_engine_no_blocks(16);
        let tokens = engine.generate(&[], &SamplingParams::default());
        assert!(tokens.is_empty());
    }

    #[test]
    fn prefix_cached_engine_clear_cache() {
        let mut engine = make_engine_no_blocks(16);
        // Run a generate so the cache might get some blocks.
        let prompt: Vec<u32> = (0..32).collect();
        let fast_params = SamplingParams {
            max_tokens: 4,
            top_k: 1,
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let _ = engine.generate(&prompt, &fast_params);
        engine.clear_cache();
        let stats = engine.cache_stats();
        assert_eq!(stats.cached_blocks, 0);
    }

    #[test]
    fn prefix_cached_engine_stats_structure() {
        let engine = make_engine_no_blocks(32);
        let stats = engine.cache_stats();
        assert_eq!(stats.capacity_blocks, 32);
        assert!((stats.hit_rate - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn prefix_cached_engine_repeated_prompt_builds_cache() {
        // Use a model with real blocks so the CPU KV cache is actually populated.
        let mut engine = make_engine_with_real_blocks(32);
        let prompt: Vec<u32> = (0..32).collect();
        let fast_params = SamplingParams {
            max_tokens: 1,
            top_k: 1,
            temperature: 0.0,
            ..SamplingParams::default()
        };

        // First call: cold cache.
        let _ = engine.generate(&prompt, &fast_params);
        let stats_after_first = engine.cache_stats();

        // Second call: same prompt; should record at least one hit and the
        // cache should contain entries.
        let _ = engine.generate(&prompt, &fast_params);
        let stats_after_second = engine.cache_stats();

        assert!(
            stats_after_first.cached_blocks > 0,
            "first call should have populated some cache blocks"
        );
        assert!(
            stats_after_second.total_hits > 0,
            "second call should record cache hits"
        );
    }

    /// Acceptance criterion #5 from issue #2: a repeated prompt must
    /// actually skip prefill work, not merely record bookkeeping hits.
    #[test]
    fn prefix_cached_engine_avoids_redundant_prefill_work() {
        let mut engine = make_engine_with_real_blocks(64);
        let prompt: Vec<u32> = (0..32).collect();
        let fast_params = SamplingParams {
            max_tokens: 2,
            top_k: 1,
            temperature: 0.0,
            ..SamplingParams::default()
        };

        let out1 = engine.generate(&prompt, &fast_params);
        let prefill_after_first = engine.inner.prefill_token_count();

        let out2 = engine.generate(&prompt, &fast_params);
        let prefill_after_second = engine.inner.prefill_token_count();

        let second_call_prefill = prefill_after_second - prefill_after_first;
        assert!(
            second_call_prefill < prompt.len() as u64,
            "second call prefilled {} tokens, expected < {} (prefix cache should have skipped some)",
            second_call_prefill,
            prompt.len()
        );
        assert!(
            engine.cache_stats().total_hits > 0,
            "cache should report hits"
        );
        // AC #3 from issue #2: cached path must produce identical output to
        // the cold-cache path. With temperature=0 and top_k=1 the sampler is
        // deterministic, so the two generations must match token-for-token.
        assert_eq!(
            out1, out2,
            "AC #3: cached path must produce identical output ({:?} vs {:?})",
            out1, out2
        );
    }

    fn make_engine_with_real_blocks_and_seed(
        max_blocks: usize,
        seed: u64,
    ) -> PrefixCachedEngine<'static> {
        use oxibonsai_kernels::{KernelDispatcher, KernelTier};
        let config = small_real_config();
        let model = BonsaiModel::new_for_testing_with_blocks(config);
        let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
        let engine =
            InferenceEngine::from_model_with_kernel(model, kernel, SamplingParams::default(), 42);
        PrefixCachedEngine::new(engine, max_blocks, seed)
    }

    /// Regression test for the "hardcoded seed=0" bug: `generate()` used to
    /// build a fresh `Sampler::new(params.clone(), 0)` on *every* call,
    /// discarding whatever seed the wrapper was constructed with. Because the
    /// sampler's xorshift64 PRNG has `0` as a fixed point, `rng_state=0`
    /// makes `next_u64()` return `0` forever, so `rand_val` is always `0.0`.
    /// With `top_k=0` (no truncation) and `top_p=1.0` (no re-sorting),
    /// `probs_buf` keeps its natural insertion order (token id 0 first), so
    /// the weighted-selection loop deterministically always picked token 0 —
    /// *regardless of the model's logits or the configured seed*. This test
    /// asserts the (fixed) real, model-driven behavior is not that constant
    /// all-token-0 sequence.
    #[test]
    fn prefix_cached_engine_does_not_collapse_to_seed_zero_fixed_point() {
        let mut engine = make_engine_with_real_blocks_and_seed(64, 0xC0FFEE);
        let prompt: Vec<u32> = (0..32).collect();
        let full_vocab_params = SamplingParams {
            max_tokens: 8,
            top_k: 0,
            top_p: 1.0,
            temperature: 2.0,
            repetition_penalty: 1.0,
        };

        let out = engine.generate(&prompt, &full_vocab_params);
        assert_eq!(out.len(), 8, "expected a full max_tokens decode run");
        assert!(
            out.iter().any(|&t| t != 0),
            "output collapsed to all-zero tokens {out:?}; this is the exact signature of the \
             seed=0 xorshift64 fixed-point bug (rand_val always 0.0 picks index 0 every step)",
        );
    }

    /// Regression test: two wrappers constructed with different explicit
    /// seeds, run against the identical prompt/model/params, must diverge.
    /// Before the fix every call built an ephemeral `Sampler::new(_, 0)`
    /// inside `generate()`, so the seed passed to `PrefixCachedEngine::new`
    /// was silently discarded and both wrappers would have produced
    /// byte-identical output.
    #[test]
    fn prefix_cached_engine_honours_configured_seed() {
        let prompt: Vec<u32> = (0..32).collect();
        let full_vocab_params = SamplingParams {
            max_tokens: 8,
            top_k: 0,
            top_p: 1.0,
            temperature: 2.0,
            repetition_penalty: 1.0,
        };

        let mut engine_a = make_engine_with_real_blocks_and_seed(64, 7);
        let mut engine_b = make_engine_with_real_blocks_and_seed(64, 424_242);

        let out_a = engine_a.generate(&prompt, &full_vocab_params);
        let out_b = engine_b.generate(&prompt, &full_vocab_params);

        assert_ne!(
            out_a, out_b,
            "two different configured seeds produced identical output; the wrapper is not \
             using the caller-supplied seed"
        );
    }

    // ── M-35: block_has_real_content ────────────────────────────────────────

    #[test]
    fn block_has_real_content_rejects_all_zero() {
        let keys = vec![vec![0.0f32; 8], vec![0.0f32; 8]];
        let values = vec![vec![0.0f32; 8], vec![0.0f32; 8]];
        assert!(!block_has_real_content(&keys, &values));
    }

    #[test]
    fn block_has_real_content_accepts_genuine_values() {
        let keys = vec![vec![0.1f32, 0.0, 0.0], vec![0.0, 0.2, 0.0]];
        let values = vec![vec![0.0f32, 0.0, 0.3], vec![0.4, 0.0, 0.0]];
        assert!(block_has_real_content(&keys, &values));
    }

    #[test]
    fn block_has_real_content_rejects_one_zero_layer_among_several() {
        // The exact failure mode the finding described: a transient GPU
        // fallback mid-prompt can leave *some* layers populated and others
        // still zero within the same block. Any single all-zero layer must
        // reject the whole block.
        let keys = vec![vec![0.1f32, 0.2], vec![0.0f32, 0.0]];
        let values = vec![vec![0.3f32, 0.4], vec![0.5f32, 0.6]];
        assert!(
            !block_has_real_content(&keys, &values),
            "one all-zero layer (keys[1]) must reject the whole block"
        );

        let keys2 = vec![vec![0.1f32, 0.2], vec![0.3f32, 0.4]];
        let values2 = vec![vec![0.5f32, 0.6], vec![0.0f32, 0.0]];
        assert!(
            !block_has_real_content(&keys2, &values2),
            "one all-zero layer (values[1]) must reject the whole block"
        );
    }

    #[test]
    fn block_has_real_content_rejects_empty() {
        assert!(!block_has_real_content(&[], &[]));
    }

    // ── M-35: GPU-resident KV tier refuses the cache-aware path ─────────────
    //
    // These exercise `OneBitKernel::is_gpu_accelerated` directly (the exact
    // accessor `generate()` now calls) rather than a locally re-derived
    // predicate, so a change to what counts as GPU-resident is caught here
    // instead of only silently diverging at the call site.

    #[test]
    fn cpu_reference_tier_is_not_gpu_accelerated() {
        use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
        let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
        assert!(!kernel.is_gpu_accelerated());
    }

    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    #[test]
    fn gpu_tier_is_gpu_accelerated() {
        use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
        let kernel = KernelDispatcher::with_tier(KernelTier::Gpu);
        assert!(kernel.is_gpu_accelerated());
    }

    // ── M-35: hybrid recurrent-attention architecture refuses caching ───────

    #[test]
    fn hybrid_architecture_never_populates_the_prefix_cache() {
        let mut engine = make_hybrid_engine_with_real_blocks(64);
        let prompt: Vec<u32> = (0..32).collect();
        let fast_params = SamplingParams {
            max_tokens: 2,
            top_k: 1,
            temperature: 0.0,
            ..SamplingParams::default()
        };

        let _ = engine.generate(&prompt, &fast_params);
        assert_eq!(
            engine.cache_stats().cached_blocks,
            0,
            "a hybrid (qwen35) architecture must never populate the prefix-cache trie \
             (it has recurrent state with no block-KV representation)"
        );

        // Repeating the identical prompt must still record zero hits — the
        // bypass path never even queries the trie.
        let _ = engine.generate(&prompt, &fast_params);
        assert_eq!(engine.cache_stats().total_hits, 0);
        assert_eq!(engine.cache_stats().cached_blocks, 0);
    }

    #[test]
    fn hybrid_architecture_bypass_still_generates_deterministically() {
        // The bypass path must still be a fully-functional generation, not
        // merely "refuses and returns empty".
        let fast_params = SamplingParams {
            max_tokens: 4,
            top_k: 1,
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let prompt: Vec<u32> = (0..32).collect();

        let mut engine_a = make_hybrid_engine_with_real_blocks(64);
        let mut engine_b = make_hybrid_engine_with_real_blocks(64);
        let out_a = engine_a.generate(&prompt, &fast_params);
        let out_b = engine_b.generate(&prompt, &fast_params);

        assert!(
            !out_a.is_empty(),
            "the bypass path must still generate tokens"
        );
        assert_eq!(
            out_a, out_b,
            "greedy decode from two freshly-constructed identical engines must match"
        );
    }

    // ── M-35 addendum: cpu_kv_is_authoritative ──────────────────────────────

    #[test]
    fn cpu_kv_is_authoritative_true_for_a_fresh_non_hybrid_model() {
        let model = BonsaiModel::new_for_testing_with_blocks(small_real_config());
        assert!(
            !model.gpu_path_active(),
            "a fresh model has not run the GPU path yet"
        );
        assert!(cpu_kv_is_authoritative(&model));
    }

    #[test]
    fn cpu_kv_is_authoritative_false_for_a_hybrid_qwen35_model() {
        let config = Qwen3Config {
            architecture: "qwen35".to_string(),
            ..small_real_config()
        };
        let model = BonsaiModel::new_for_testing_with_blocks(config);
        // A hybrid model refuses regardless of the GPU-path flag — it has no
        // block-KV representation to be authoritative about at all.
        assert!(!model.gpu_path_active());
        assert!(!cpu_kv_is_authoritative(&model));
    }

    #[test]
    fn store_new_blocks_is_a_no_op_when_cpu_kv_is_not_authoritative() {
        // Direct unit test of the M-35 addendum's belt-and-suspenders check
        // inside `store_new_blocks` itself, independent of `generate()`'s
        // own up-front gate (which a hybrid model never reaches this far
        // past anyway — see `hybrid_architecture_never_populates_the_prefix_cache`).
        let config = Qwen3Config {
            architecture: "qwen35".to_string(),
            ..small_real_config()
        };
        let model = BonsaiModel::new_for_testing_with_blocks(config);
        let kernel = oxibonsai_kernels::KernelDispatcher::with_tier(
            oxibonsai_kernels::KernelTier::Reference,
        );
        let engine =
            InferenceEngine::from_model_with_kernel(model, kernel, SamplingParams::default(), 42);
        let mut wrapped = PrefixCachedEngine::new(engine, 64, 42);

        let prompt: Vec<u32> = (0..32).collect();
        wrapped.store_new_blocks(&prompt, 0, BLOCK_SIZE, 1);
        assert_eq!(
            wrapped.cache_stats().cached_blocks,
            0,
            "store_new_blocks must refuse to populate the trie for a hybrid model \
             even when called directly"
        );
    }
}
