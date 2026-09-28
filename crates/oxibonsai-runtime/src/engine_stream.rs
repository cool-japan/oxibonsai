//! Log-probability and streaming generation entry points of
//! [`InferenceEngine`].
//!
//! Split out of `engine.rs` to keep that file under the workspace 2000-line
//! ceiling; these are inherent methods of [`InferenceEngine`], so the files
//! form one API surface. Every decode step dispatches through the
//! [`crate::engine_seam`] seam, so both dense and hybrid engines stream.

use crate::engine::InferenceEngine;
#[cfg(feature = "server")]
use crate::engine::MAX_PREALLOC_TOKENS;
use crate::error::RuntimeResult;

impl InferenceEngine<'_> {
    /// Generate tokens while capturing per-step top-k log probabilities.
    ///
    /// Mirrors [`InferenceEngine::generate`] (prefill → token-by-token decode,
    /// penalties applied over the generated-token history), but for every
    /// emitted token it also records a
    /// [`LogprobsContent`](crate::api_types::LogprobsContent) computed from the
    /// model's raw output logits at that step: the chosen token's log
    /// probability plus the `top_k` highest-probability alternatives (OpenAI
    /// `top_logprobs`, clamped to 20).
    ///
    /// `id_to_token` maps a token id to its string form (typically the
    /// tokenizer's single-id decode); the engine has no tokenizer of its own,
    /// so the caller supplies it. The returned logprobs vector has exactly one
    /// entry per generated token, aligned with the returned token ids.
    ///
    /// Available only with the `server` feature, where the logprob types live.
    #[cfg(feature = "server")]
    pub fn generate_with_logprobs(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        top_k: usize,
        id_to_token: &dyn Fn(u32) -> String,
    ) -> RuntimeResult<(Vec<u32>, Vec<crate::api_types::LogprobsContent>)> {
        if prompt_tokens.is_empty() {
            return Ok((vec![], vec![]));
        }

        // OpenAI caps top_logprobs at 20.
        let top_k = top_k.min(20);

        // No GPU-argmax routing here, by construction: a 4-byte token
        // readback cannot produce `top_logprobs`, which is why
        // `greedy_gpu_eligible` takes a `needs_full_logits` argument at all
        // (`perf-11`'s correction names logprobs and logit_bias explicitly).
        // This path always decodes the full logit row.

        let Some(mut last_logits) = self.prefill_for_generate(prompt_tokens)? else {
            return Ok((vec![], vec![]));
        };
        let cap = max_tokens.min(MAX_PREALLOC_TOKENS);
        let mut output_tokens = Vec::with_capacity(cap);
        let mut logprobs: Vec<crate::api_types::LogprobsContent> = Vec::with_capacity(cap);

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            if self.is_cancelled() {
                tracing::debug!(pos, "logprobs generation cancelled");
                break;
            }
            let next_token = self
                .sampler
                .sample_with_history(&last_logits, &output_tokens)?;

            if self.is_eos(next_token) {
                tracing::debug!(pos, "EOS token generated (logprobs)");
                break;
            }

            // Capture logprobs from the model's raw (pre-penalty) output
            // distribution — the reported logprob is the model's, while the
            // chosen token already reflects any active penalties.
            logprobs.push(crate::api_types::compute_logprobs(
                &last_logits,
                next_token,
                top_k,
                id_to_token,
            ));
            output_tokens.push(next_token);

            last_logits = self.forward_logits(next_token, pos)?;
        }

        self.stats.record_request(output_tokens.len());

        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated = output_tokens.len(),
            "logprobs generation complete"
        );

        Ok((output_tokens, logprobs))
    }

    /// Generate tokens one at a time, sending each through the channel.
    /// Returns the total count of generated tokens.
    ///
    /// Not available on WASM targets (tokio channels not supported on wasm32-unknown-unknown).
    #[cfg(not(target_arch = "wasm32"))]
    #[tracing::instrument(skip(self, prompt_tokens, tx), fields(prompt_len = prompt_tokens.len()))]
    pub fn generate_streaming(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        tx: &tokio::sync::mpsc::UnboundedSender<u32>,
    ) -> RuntimeResult<usize> {
        if prompt_tokens.is_empty() {
            return Ok(0);
        }

        // `perf-11`: the server streams through this path, so routing it
        // through the GPU argmax is what actually removes the per-token
        // full-logit download from `serve`/`chat` — the CLI-only shortcut
        // was the whole finding.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if self.greedy_gpu_eligible(false) {
            let tokens =
                self.generate_greedy_gpu_unchecked(prompt_tokens, max_tokens, |token| {
                    // A send failure means the receiver was dropped (client
                    // disconnected): stop generating, exactly as below.
                    tx.send(token).is_ok()
                })?;
            return Ok(tokens.len());
        }
        // `perf-11`, sampled half: top-k candidates instead of full rows.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(tokens) =
            self.try_sampled_topk_route(prompt_tokens, max_tokens, |token| tx.send(token).is_ok())?
        {
            return Ok(tokens.len());
        }

        // Prefill: batch process all prompt tokens
        let Some(mut logits) = self.prefill_for_generate(prompt_tokens)? else {
            return Ok(0);
        };

        let decode_start = std::time::Instant::now();
        let mut generated = 0;
        // Generated-token history for repetition/frequency/presence penalties.
        let mut history: Vec<u32> = Vec::new();

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();
            if self.is_cancelled() {
                tracing::debug!(pos, "streaming generation cancelled");
                break;
            }
            let next_token = self.sampler.sample_with_history(&logits, &history)?;

            if self.is_eos(next_token) {
                tracing::debug!(pos, "EOS token generated (streaming)");
                break;
            }

            // Send token through channel; if receiver dropped, stop generating
            if tx.send(next_token).is_err() {
                tracing::debug!(pos, "receiver dropped, stopping generation");
                break;
            }
            history.push(next_token);

            logits = self.forward_logits(next_token, pos)?;
            generated += 1;

            if let Some(m) = &self.metrics {
                m.decode_token_duration_seconds
                    .observe(step_start.elapsed().as_secs_f64());
            }
        }

        // Record tokens/sec and update memory gauge
        if let Some(m) = &self.metrics {
            let decode_elapsed = decode_start.elapsed().as_secs_f64();
            if decode_elapsed > 0.0 && generated > 0 {
                let tok_per_sec = generated as f64 / decode_elapsed;
                m.tokens_per_second.observe(tok_per_sec);
            }
            m.tokens_generated_total.inc_by(generated as u64);
            m.update_memory_from_rss();
        }
        // Record engine-level stats. Kept symmetric with `generate` /
        // `generate_tracked`'s CPU tails and with the GPU-argmax path's
        // `generate_greedy_gpu_unchecked` (a wave-2 verifier finding: this
        // call was previously missing here, so `EngineStats::requests_completed`
        // / `tokens_generated` depended on which decode route a given
        // request happened to take).
        self.stats.record_request(generated);

        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated,
            "streaming generation complete"
        );

        Ok(generated)
    }

    /// Streaming generation using caller-supplied sampling parameters for the
    /// duration of this call only.
    ///
    /// Swaps in `params` on the engine's existing sampler, runs
    /// [`InferenceEngine::generate_streaming`], then restores the previous
    /// parameters. As with [`InferenceEngine::generate_with_params`], the
    /// sampler's PRNG state is preserved (only the parameters change), so the
    /// default-parameter case is bit-identical to calling
    /// `generate_streaming` directly.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn generate_streaming_with_params(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        params: &crate::sampling::SamplingParams,
        tx: &tokio::sync::mpsc::UnboundedSender<u32>,
    ) -> RuntimeResult<usize> {
        let prev_params = self.sampler.params().clone();
        self.sampler.set_params(params.clone());
        let result = self.generate_streaming(prompt_tokens, max_tokens, tx);
        self.sampler.set_params(prev_params);
        result
    }

    /// Streaming generation using a synchronous `std::sync::mpsc::Sender`.
    ///
    /// Each generated token is sent through the channel immediately, allowing
    /// the consumer to print tokens as they arrive without requiring a tokio runtime.
    #[tracing::instrument(skip(self, prompt_tokens, tx), fields(prompt_len = prompt_tokens.len()))]
    pub fn generate_streaming_sync(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        tx: &std::sync::mpsc::Sender<u32>,
    ) -> RuntimeResult<usize> {
        if prompt_tokens.is_empty() {
            return Ok(0);
        }

        // Greedy + fused Metal route → GPU argmax (`perf-11`). The CLI's
        // streaming path reaches this function.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if self.greedy_gpu_eligible(false) {
            let tokens =
                self.generate_greedy_gpu_unchecked(prompt_tokens, max_tokens, |token| {
                    tx.send(token).is_ok()
                })?;
            return Ok(tokens.len());
        }
        // `perf-11`, sampled half: top-k candidates instead of full rows.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(tokens) =
            self.try_sampled_topk_route(prompt_tokens, max_tokens, |token| tx.send(token).is_ok())?
        {
            return Ok(tokens.len());
        }

        // Prefill: batch process all prompt tokens
        let Some(mut logits) = self.prefill_for_generate(prompt_tokens)? else {
            return Ok(0);
        };

        let decode_start = std::time::Instant::now();
        let mut generated = 0;
        // Generated-token history for repetition/frequency/presence penalties.
        let mut history: Vec<u32> = Vec::new();

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();
            if self.is_cancelled() {
                tracing::debug!(pos, "streaming sync generation cancelled");
                break;
            }

            let next_token = self.sampler.sample_with_history(&logits, &history)?;

            if self.is_eos(next_token) {
                tracing::debug!(pos, "EOS token generated (streaming_sync)");
                break;
            }

            if tx.send(next_token).is_err() {
                tracing::debug!(pos, "receiver dropped, stopping generation");
                break;
            }
            history.push(next_token);

            logits = self.forward_logits(next_token, pos)?;
            generated += 1;

            if let Some(m) = &self.metrics {
                m.decode_token_duration_seconds
                    .observe(step_start.elapsed().as_secs_f64());
            }
        }

        if let Some(m) = &self.metrics {
            let decode_elapsed = decode_start.elapsed().as_secs_f64();
            if decode_elapsed > 0.0 && generated > 0 {
                let tok_per_sec = generated as f64 / decode_elapsed;
                m.tokens_per_second.observe(tok_per_sec);
            }
            m.tokens_generated_total.inc_by(generated as u64);
            m.update_memory_from_rss();
        }
        // See the identical comment in `generate_streaming`'s CPU tail
        // (wave-2 verifier finding): keeps `EngineStats` symmetric across
        // every decode route.
        self.stats.record_request(generated);

        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated,
            "streaming sync generation complete"
        );

        Ok(generated)
    }
}
