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
//! ## Limitations
//!
//! Real prefix-cache reuse is only effective when the engine's forward
//! path populates the CPU [`oxibonsai_model::KvCache`]. On Metal/CUDA tiers
//! the GPU keeps its own KV state separate from the CPU cache; in that
//! case the post-prefill extraction would yield all-zero tensors. This
//! engine detects that case (the `real_cpu_kv` check below) and falls back
//! to plain prefill without poisoning the trie. The session bookkeeping
//! (hit-rate stats) still runs.

use oxibonsai_model::prefix_cache::{
    KvBlockPair, PrefixAwarePrefill, PrefixCache, PrefixCacheStats,
};

use crate::engine::InferenceEngine;
use crate::sampling::{Sampler, SamplingParams};

/// Tokens per cache block — must divide evenly into most prompt lengths.
const BLOCK_SIZE: usize = 16;

/// An [`InferenceEngine`] augmented with prefix KV-cache reuse.
///
/// On each [`generate`](PrefixCachedEngine::generate) call the engine:
///
/// 1. Resets the model's KV cache (single-engine, sequential request model).
/// 2. Looks up the longest cached prefix in the trie.
/// 3. Injects the matched KV blocks back into the model's CPU cache.
/// 4. Runs prefill only on the uncached suffix at the correct `pos_start`.
/// 5. Extracts any newly produced full blocks of KV state and stores them
///    in the trie for subsequent requests (skipped on GPU tiers where the
///    CPU cache stays empty).
/// 6. Sample-decodes new tokens up to `params.max_tokens` or EOS.
/// 7. Releases the session (decrements ref counts) when done.
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

    /// Generate tokens from `prompt_tokens`, reusing any cached prefix.
    ///
    /// Returns the generated token IDs (not including the prompt). On any
    /// internal error the method logs via `tracing::warn` and returns an
    /// empty vector — `generate` itself is infallible from the caller's
    /// perspective so it can be dropped into batch pipelines.
    pub fn generate(&mut self, prompt_tokens: &[u32], params: &SamplingParams) -> Vec<u32> {
        if prompt_tokens.is_empty() {
            return vec![];
        }

        // ── Step 1: reset model KV cache ─────────────────────────────────────
        // We treat the wrapper as a single-engine, sequential request server.
        self.inner.model_mut().reset();

        // ── Step 2: query the prefix cache ───────────────────────────────────
        let (session, uncached_start) = self.prefix_cache.prepare(prompt_tokens);
        let block_size = self.prefix_cache.cache.block_size();
        let cfg = self.inner.model().config().clone();
        let num_layers = cfg.num_layers;

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
        let mut last_logits = if uncached_start < prompt_tokens.len() {
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

        // ── Step 5: detect whether the CPU KV cache was actually populated ──
        // GPU tiers (Metal/CUDA) maintain their own KV cache and leave the
        // CPU `KvCache` untouched; in that case any extraction yields zeros
        // which would silently corrupt the trie. We sample one layer/head/
        // range and skip the store_blocks step if everything is zero.
        let real_cpu_kv = {
            let kv = self.inner.model().kv_cache();
            let probe_len = prompt_tokens.len().min(kv.max_seq_len());
            kv.keys_for(0, 0, probe_len).iter().any(|&x| x != 0.0)
        };

        // ── Step 6: store newly computed blocks into the trie ────────────────
        if real_cpu_kv {
            let new_blocks_count = prompt_tokens.len().saturating_sub(uncached_start) / block_size;
            if new_blocks_count > 0 {
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
                    keys_by_block.push((layer_keys, layer_values));
                }
                self.prefix_cache
                    .store_blocks(prompt_tokens, uncached_start, keys_by_block);
            }
        }

        // ── Step 7: decode loop ──────────────────────────────────────────────
        // Swap `params` onto the wrapper's own seeded sampler (constructed
        // once in `new()` from the caller-supplied seed) so that per-call
        // sampling parameters are honoured while the PRNG state keeps
        // advancing across calls — the same "swap params, keep RNG state"
        // pattern `InferenceEngine::generate_with_params` uses, rather than
        // discarding the configured seed and re-seeding at 0 every call.
        let prev_params = self.sampler.params().clone();
        self.sampler.set_params(params.clone());
        let mut output = Vec::with_capacity(params.max_tokens);
        for (pos, _) in (prompt_tokens.len()..).zip(0..params.max_tokens) {
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

        // ── Step 8: release session ──────────────────────────────────────────
        self.prefix_cache.release_session(session);
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

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::config::Qwen3Config;
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
            vocab_size: 256,
            max_context_length: 128,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10_000.0,
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
}
