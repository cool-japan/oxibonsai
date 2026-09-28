//! Generation entry points of [`InferenceEngine`]: batched, plain,
//! tracked, request-id-tagged, seeded and per-call-parameter generation.
//!
//! A child module of [`crate::engine`] (declared there with `#[path]`), so
//! the methods keep direct access to the engine's private fields while
//! `engine.rs` stays under the workspace 2000-line ceiling. Every method is
//! an inherent method of [`InferenceEngine`]: the public paths
//! (`oxibonsai_runtime::engine::InferenceEngine::generate` and friends) are
//! the same as when they lived in `engine.rs`.
//!
//! Every entry point decodes under the engine's one decoding contract (see
//! the [`crate::engine`] module docs): a configured-greedy request on the
//! fused Metal route takes the GPU argmax, a sampled one on that route takes
//! the top-k candidate route when eligible (`crate::engine_greedy`), and
//! everything else runs the classic full-row loop with the sampler's
//! penalties applied before every draw.

use std::sync::atomic::Ordering;

use super::{InferenceEngine, MAX_PREALLOC_TOKENS};
use crate::batch_engine::{self, BatchResult};
use crate::error::RuntimeResult;
use crate::request_id::RequestId;
use crate::request_metrics::{RequestRateSnapshot, RequestRateTracker};
use crate::sampling::PenaltyParams;

impl InferenceEngine<'_> {
    /// Process a batch of prompts, delegating to [`batch_engine::batch_generate`].
    ///
    /// Resets the engine state between each prompt. Returns one result per prompt.
    pub fn batch_generate(
        &mut self,
        prompts: &[Vec<u32>],
        max_tokens: usize,
    ) -> Vec<RuntimeResult<BatchResult>> {
        self.stats.active_sessions.fetch_add(1, Ordering::Relaxed);

        let results = batch_engine::batch_generate(self, prompts, max_tokens);

        // Record stats for successful results
        for br in results.iter().flatten() {
            self.stats.record_request(br.generated_tokens.len());
        }

        self.stats.active_sessions.fetch_sub(1, Ordering::Relaxed);

        results
    }

    /// Generate tokens from a prompt.
    ///
    /// Runs prefill (process the entire prompt), then decodes
    /// token by token until `max_tokens` or EOS is reached.
    /// Returns the generated token IDs (not including the prompt).
    #[tracing::instrument(skip(self, prompt_tokens), fields(prompt_len = prompt_tokens.len()))]
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> RuntimeResult<Vec<u32>> {
        if prompt_tokens.is_empty() {
            return Ok(vec![]);
        }

        // `perf-11`: a configured-greedy request on the fused Metal route
        // decodes with the GPU argmax, downloading 4 bytes per token instead
        // of the whole f32 logit row (993 KB/token at Bonsai 2's 248 320
        // vocabulary). Eligibility — including "no penalties are configured"
        // — is decided by the one shared predicate, so this can never become
        // a second decoding contract.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if self.greedy_gpu_eligible(false) {
            return self.generate_greedy_gpu_unchecked(prompt_tokens, max_tokens, |_| true);
        }
        // `perf-11`, sampled half: a sampled request on the fused route
        // downloads only its top-k candidates per token when eligible.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(tokens) = self.try_sampled_topk_route(prompt_tokens, max_tokens, |_| true)? {
            return Ok(tokens);
        }

        // ═══════════════════════════════════════════════════════
        // 1. Prefill: batch process all prompt tokens
        // ═══════════════════════════════════════════════════════
        let Some(mut last_logits) = self.prefill_for_generate(prompt_tokens)? else {
            return Ok(vec![]);
        };

        // ═══════════════════════════════════════════════════════
        // 2. Decode: sample and generate
        // ═══════════════════════════════════════════════════════
        let decode_start = std::time::Instant::now();
        let mut output_tokens = Vec::with_capacity(max_tokens.min(MAX_PREALLOC_TOKENS));

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();

            // `SV-09`: cooperative cancellation, checked before the (costly)
            // forward of this step. Returns what has been generated so far.
            if self.is_cancelled() {
                tracing::debug!(pos, "generation cancelled");
                break;
            }

            // Sample next token, applying repetition/frequency/presence
            // penalties over the generated-token history so far.
            let next_token = self
                .sampler
                .sample_with_history(&last_logits, &output_tokens)?;

            // Check for EOS (any id in the model's terminator set, RT-18)
            if self.is_eos(next_token) {
                tracing::debug!(pos, "EOS token generated");
                break;
            }

            output_tokens.push(next_token);

            // Forward the generated token
            last_logits = self.forward_logits(next_token, pos)?;

            if let Some(m) = &self.metrics {
                m.decode_token_duration_seconds
                    .observe(step_start.elapsed().as_secs_f64());
            }
        }

        // Record tokens/sec and update memory gauge
        if let Some(m) = &self.metrics {
            let decode_elapsed = decode_start.elapsed().as_secs_f64();
            if decode_elapsed > 0.0 && !output_tokens.is_empty() {
                let tok_per_sec = output_tokens.len() as f64 / decode_elapsed;
                m.tokens_per_second.observe(tok_per_sec);
            }
            m.tokens_generated_total.inc_by(output_tokens.len() as u64);
            m.update_memory_from_rss();
        }

        // Record engine-level stats
        self.stats.record_request(output_tokens.len());

        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated = output_tokens.len(),
            "generation complete"
        );

        Ok(output_tokens)
    }

    /// Generate tokens from a prompt while populating a [`RequestRateTracker`].
    ///
    /// Behaves identically to [`InferenceEngine::generate`] but additionally:
    /// - records `record_admission()` immediately on entry,
    /// - records `record_first_token()` for the first sampled token,
    /// - records `record_token()` for every subsequent sampled token,
    /// - on success, pushes the resulting [`RequestRateSnapshot`] into the
    ///   engine's attached
    ///   [`RequestRateAggregator`](crate::request_metrics::RequestRateAggregator)
    ///   (if any).
    ///
    /// The tracker is borrowed mutably so callers can inspect intermediate
    /// state via [`RequestRateTracker::snapshot`] after the call returns.
    #[tracing::instrument(skip(self, prompt_tokens, tracker), fields(prompt_len = prompt_tokens.len()))]
    pub fn generate_tracked(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        tracker: &mut RequestRateTracker,
    ) -> RuntimeResult<Vec<u32>> {
        if prompt_tokens.is_empty() {
            return Ok(vec![]);
        }
        tracker.record_admission();

        // Greedy + fused Metal route → GPU argmax (`perf-11`), with the
        // per-token tracker events driven from the emit callback so the
        // recorded latency series is identical to the CPU path's.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if self.greedy_gpu_eligible(false) {
            let mut first_token_recorded = false;
            let tokens =
                self.generate_greedy_gpu_unchecked(prompt_tokens, max_tokens, |_token| {
                    if first_token_recorded {
                        tracker.record_token();
                    } else {
                        tracker.record_first_token();
                        first_token_recorded = true;
                    }
                    true
                })?;
            if let Some(agg) = &self.rate_aggregator {
                let snap: RequestRateSnapshot = tracker.snapshot();
                agg.record(snap);
            }
            return Ok(tokens);
        }
        // Sampled + fused route → top-k candidates (`perf-11`, sampled
        // half), with the same per-token tracker events.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let mut first_token_recorded = false;
            let routed = self.try_sampled_topk_route(prompt_tokens, max_tokens, |_token| {
                if first_token_recorded {
                    tracker.record_token();
                } else {
                    tracker.record_first_token();
                    first_token_recorded = true;
                }
                true
            })?;
            if let Some(tokens) = routed {
                if let Some(agg) = &self.rate_aggregator {
                    let snap: RequestRateSnapshot = tracker.snapshot();
                    agg.record(snap);
                }
                return Ok(tokens);
            }
        }

        let Some(mut last_logits) = self.prefill_for_generate(prompt_tokens)? else {
            return Ok(vec![]);
        };

        let decode_start = std::time::Instant::now();
        let mut output_tokens = Vec::with_capacity(max_tokens.min(MAX_PREALLOC_TOKENS));
        let mut first_token_recorded = false;

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();
            if self.is_cancelled() {
                tracing::debug!(pos, "tracked generation cancelled");
                break;
            }
            let next_token = self
                .sampler
                .sample_with_history(&last_logits, &output_tokens)?;
            if self.is_eos(next_token) {
                tracing::debug!(pos, "EOS token generated");
                break;
            }
            output_tokens.push(next_token);
            if !first_token_recorded {
                tracker.record_first_token();
                first_token_recorded = true;
            } else {
                tracker.record_token();
            }
            last_logits = self.forward_logits(next_token, pos)?;

            if let Some(m) = &self.metrics {
                m.decode_token_duration_seconds
                    .observe(step_start.elapsed().as_secs_f64());
            }
        }

        if let Some(m) = &self.metrics {
            let decode_elapsed = decode_start.elapsed().as_secs_f64();
            if decode_elapsed > 0.0 && !output_tokens.is_empty() {
                let tok_per_sec = output_tokens.len() as f64 / decode_elapsed;
                m.tokens_per_second.observe(tok_per_sec);
            }
            m.tokens_generated_total.inc_by(output_tokens.len() as u64);
            m.update_memory_from_rss();
        }
        self.stats.record_request(output_tokens.len());

        if let Some(agg) = &self.rate_aggregator {
            let snap: RequestRateSnapshot = tracker.snapshot();
            agg.record(snap);
        }

        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated = output_tokens.len(),
            "tracked generation complete"
        );

        Ok(output_tokens)
    }

    /// Generate tokens from a prompt with a [`RequestId`] tagging the
    /// surrounding tracing span and an internally-managed
    /// [`RequestRateTracker`].
    ///
    /// Returns both the generated tokens and the final tracker so callers
    /// can extract per-request metrics (e.g. queue-wait, p95 inter-token
    /// latency) for client-side observability.
    pub fn generate_with_request_id(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> RuntimeResult<(Vec<u32>, RequestRateTracker)> {
        let span = tracing::info_span!("generate_request", request_id = %request_id);
        let _enter = span.enter();
        let mut tracker = RequestRateTracker::new();
        let tokens = self.generate_tracked(prompt_tokens, max_tokens, &mut tracker)?;
        Ok((tokens, tracker))
    }

    /// Generate tokens from a prompt using a specific seed for this run.
    ///
    /// Temporarily overrides the sampler seed for deterministic multi-completion
    /// generation (`n > 1`). The sampler state is replaced for the duration of
    /// this call and then restored.
    ///
    /// The per-call sampler runs `params` with the given `seed` and carries
    /// over the engine sampler's configuration that is not part of
    /// [`SamplingParams`](crate::sampling::SamplingParams): the frequency /
    /// presence penalties and the min-p threshold
    /// ([`InferenceEngine::set_min_p`]), so a seeded request honours both
    /// exactly like an unseeded one.
    pub fn generate_with_seed(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        seed: u64,
        params: &crate::sampling::SamplingParams,
    ) -> RuntimeResult<Vec<u32>> {
        // Swap in a fresh sampler with the given seed, carrying over the
        // configured frequency/presence penalties and min-p threshold so
        // seeded multi-completion generation honours them just like the
        // primary path.
        let mut fresh = crate::sampling::Sampler::new(params.clone(), seed);
        fresh.set_penalties(*self.sampler.penalties());
        fresh.set_min_p(self.sampler.min_p());
        let old_sampler = std::mem::replace(&mut self.sampler, fresh);
        let result = self.generate(prompt_tokens, max_tokens);
        // Restore the original sampler
        self.sampler = old_sampler;
        result
    }

    /// Generate tokens from a prompt using caller-supplied sampling parameters
    /// for the duration of this call only.
    ///
    /// Swaps in `params` (temperature, top-k, top-p, repetition penalty) on the
    /// engine's existing sampler, runs [`InferenceEngine::generate`], then
    /// restores the previous parameters. Crucially, the sampler's PRNG state is
    /// **not** reset — only the parameters change — so the RNG sequence for the
    /// next request is identical to what it would have been had this call used
    /// the engine's default parameters. This makes the default-parameter case
    /// bit-identical to calling `generate` directly.
    pub fn generate_with_params(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        params: &crate::sampling::SamplingParams,
    ) -> RuntimeResult<Vec<u32>> {
        let prev_params = self.sampler.params().clone();
        self.sampler.set_params(params.clone());
        let result = self.generate(prompt_tokens, max_tokens);
        self.sampler.set_params(prev_params);
        result
    }

    /// Generate tokens using caller-supplied sampling parameters *and*
    /// frequency / presence penalties for the duration of this call only.
    ///
    /// Behaves like [`InferenceEngine::generate_with_params`] but additionally
    /// swaps in `penalties` (OpenAI `frequency_penalty` / `presence_penalty`),
    /// then restores both the previous parameters and penalties on return.
    /// This is the one-call seam intended for the OpenAI-compatible server:
    /// combined with `params.repetition_penalty`, it applies all three penalty
    /// families over the generated-token history. The PRNG state is preserved,
    /// so the all-default (no-penalty) case is bit-identical to
    /// [`InferenceEngine::generate`].
    pub fn generate_with_params_and_penalties(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        params: &crate::sampling::SamplingParams,
        penalties: &PenaltyParams,
    ) -> RuntimeResult<Vec<u32>> {
        let prev_params = self.sampler.params().clone();
        let prev_penalties = *self.sampler.penalties();
        self.sampler.set_params(params.clone());
        self.sampler.set_penalties(*penalties);
        let result = self.generate(prompt_tokens, max_tokens);
        self.sampler.set_params(prev_params);
        self.sampler.set_penalties(prev_penalties);
        result
    }
}
