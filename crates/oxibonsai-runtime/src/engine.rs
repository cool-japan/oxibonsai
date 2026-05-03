//! Inference engine orchestrating model loading and generation.
//!
//! The [`InferenceEngine`] is the main entry point for running inference.
//! It owns the model, kernel dispatcher, and sampler, and provides both
//! blocking ([`InferenceEngine::generate`]) and streaming
//! ([`InferenceEngine::generate_streaming`]) generation APIs.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::KernelDispatcher;
use oxibonsai_model::model::BonsaiModel;

use crate::batch_engine::{self, BatchResult};
use crate::error::{RuntimeError, RuntimeResult};
use crate::metrics::InferenceMetrics;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::ngram_cache::NgramCache;
use crate::sampling::{Sampler, SamplingParams};

/// EOS token for Qwen3 models.
pub const EOS_TOKEN_ID: u32 = 151645;

/// Statistics about engine usage, accumulated over the engine's lifetime.
#[derive(Debug)]
pub struct EngineStats {
    /// Total number of tokens generated.
    pub total_tokens_generated: AtomicU64,
    /// Total number of inference requests completed.
    pub total_requests: AtomicU64,
    /// Number of currently active sessions.
    pub active_sessions: AtomicUsize,
    /// Engine start time.
    pub start_time: Instant,
}

impl EngineStats {
    /// Create new engine stats, recording the current time as start.
    pub fn new() -> Self {
        Self {
            total_tokens_generated: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            active_sessions: AtomicUsize::new(0),
            start_time: Instant::now(),
        }
    }

    /// Engine uptime in seconds.
    pub fn uptime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Record that a request completed with the given number of generated tokens.
    pub fn record_request(&self, tokens_generated: usize) {
        self.total_tokens_generated
            .fetch_add(tokens_generated as u64, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total tokens generated.
    pub fn tokens_generated(&self) -> u64 {
        self.total_tokens_generated.load(Ordering::Relaxed)
    }

    /// Get total requests completed.
    pub fn requests_completed(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Get number of active sessions.
    pub fn active_session_count(&self) -> usize {
        self.active_sessions.load(Ordering::Relaxed)
    }

    /// Average tokens per request (returns 0.0 if no requests).
    pub fn avg_tokens_per_request(&self) -> f64 {
        let reqs = self.requests_completed();
        if reqs == 0 {
            return 0.0;
        }
        self.tokens_generated() as f64 / reqs as f64
    }
}

impl Default for EngineStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Top-level inference engine.
pub struct InferenceEngine<'a> {
    model: BonsaiModel<'a>,
    kernel: KernelDispatcher,
    sampler: Sampler,
    metrics: Option<Arc<InferenceMetrics>>,
    stats: Arc<EngineStats>,
    /// Cumulative number of tokens that have been processed by
    /// [`InferenceEngine::prefill_from_pos`] across the engine's lifetime.
    ///
    /// Used by the prefix-cache integration to verify that cached prefixes
    /// actually reduce prefill work — the cached portion of a prompt is not
    /// re-fed into prefill, so a repeated prompt should increment this
    /// counter by strictly fewer tokens than its full length.
    prefill_token_count: u64,
}

impl<'a> InferenceEngine<'a> {
    /// Create a new inference engine from a configuration (no weights — for testing).
    pub fn new(config: Qwen3Config, sampling_params: SamplingParams, seed: u64) -> Self {
        let model = BonsaiModel::new(config);
        let kernel = KernelDispatcher::auto_detect();
        let sampler = Sampler::new(sampling_params, seed);

        tracing::info!(kernel = kernel.name(), "inference engine initialized");

        Self {
            model,
            kernel,
            sampler,
            metrics: None,
            stats: Arc::new(EngineStats::new()),
            prefill_token_count: 0,
        }
    }

    /// Wrap an already-constructed [`BonsaiModel`] in an inference engine.
    ///
    /// Lets tests (and future custom-model paths) build a model with
    /// non-trivial weights and then attach the standard sampler/kernel
    /// machinery without going through the GGUF loader.
    pub fn from_model(model: BonsaiModel<'a>, sampling_params: SamplingParams, seed: u64) -> Self {
        Self::from_model_with_kernel(
            model,
            KernelDispatcher::auto_detect(),
            sampling_params,
            seed,
        )
    }

    /// Wrap an already-constructed [`BonsaiModel`] using a caller-supplied
    /// kernel dispatcher.
    ///
    /// Use this when you need to pin the engine to a specific kernel tier
    /// (e.g. a CPU-only `KernelTier::Reference` for tests that exercise the
    /// CPU KV-cache path on a host that would otherwise auto-detect a GPU).
    pub fn from_model_with_kernel(
        model: BonsaiModel<'a>,
        kernel: KernelDispatcher,
        sampling_params: SamplingParams,
        seed: u64,
    ) -> Self {
        let sampler = Sampler::new(sampling_params, seed);
        Self {
            model,
            kernel,
            sampler,
            metrics: None,
            stats: Arc::new(EngineStats::new()),
            prefill_token_count: 0,
        }
    }

    /// Create a new inference engine from a loaded GGUF file.
    pub fn from_gguf(
        gguf: &'a GgufFile<'a>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Self> {
        let mut model = BonsaiModel::from_gguf(gguf, max_seq_len)?;
        let kernel = KernelDispatcher::auto_detect();

        // Upload all model weights to GPU memory once (no-op on CPU-only tiers).
        model.upload_weights_to_gpu(&kernel);

        // Pre-build GPU weight cache eagerly so it's outside the timing window.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            tracing::info!("pre-building GPU weight cache");
            model.get_or_create_gpu_cache().map_err(|e| {
                RuntimeError::Model(oxibonsai_model::error::ModelError::Internal(format!(
                    "GPU weight cache init: {e}"
                )))
            })?;
        }

        // Pre-warm both CUDA code paths so all first-call overhead (CUDA driver graph
        // capture, prefill kernel module loading, weight uploads) is paid during model
        // loading and NOT inside the benchmark timer.
        //
        // Two passes are required:
        //   1. Single-token decode via `model.forward` → captures the 36-layer CUDA
        //      driver graph (slow-path ~490ms becomes fast-path ~44ms thereafter).
        //   2. Two-token batch via `model.forward_prefill` → loads the prefill PTX
        //      module into GPU driver memory (`init_prefill_modules`) which takes
        //      ~100-200ms on first call and is not triggered by step 1.
        //
        // Both warmup K/V cache writes are at positions that real inference
        // overwrites immediately (K/V is written before attention reads it).
        // The CUDA KV cache is separate from the CPU-side `model.kv_cache`.
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            tracing::info!("CUDA warmup: pre-capturing driver graph + prefill modules");
            // Step 1: capture the 36-layer decode CUDA driver graph.
            let _ = model.forward(0, 0, &kernel);
            // Step 2: pre-load the batch-prefill PTX module into GPU driver memory
            // (`init_prefill_modules`) and pre-allocate the prefill KV cache,
            // single-token attention buffers, and activation buffers.
            // We use 17 tokens so the CUDA batch prefill code path is exercised
            // (prompts ≤ 16 tokens use the fast decode-graph path instead).
            // This ensures all one-time batch-prefill setup costs are paid before
            // the benchmark timer, covering longer prompts without a cold-start penalty.
            let _ = model.forward_prefill(&[0u32; 17], 0, &kernel);
            tracing::info!("CUDA warmup complete");
        }

        let sampler = Sampler::new(sampling_params, seed);

        tracing::info!(kernel = kernel.name(), "inference engine loaded from GGUF");

        Ok(Self {
            model,
            kernel,
            sampler,
            metrics: None,
            stats: Arc::new(EngineStats::new()),
            prefill_token_count: 0,
        })
    }

    /// Attach shared metrics to this engine for recording inference telemetry.
    pub fn set_metrics(&mut self, metrics: Arc<InferenceMetrics>) {
        self.metrics = Some(metrics);
    }

    /// Get a reference to the model.
    pub fn model(&self) -> &BonsaiModel<'a> {
        &self.model
    }

    /// Get a mutable reference to the model.
    ///
    /// Used by the prefix-cache integration to inject restored KV blocks
    /// before running the abbreviated prefill.
    pub fn model_mut(&mut self) -> &mut BonsaiModel<'a> {
        &mut self.model
    }

    /// Get a reference to the kernel dispatcher.
    pub fn kernel(&self) -> &KernelDispatcher {
        &self.kernel
    }

    /// Run prefill at a given KV-cache offset.
    ///
    /// Unlike [`InferenceEngine::generate`], this does **not** reset the
    /// model's KV cache before execution: callers (e.g. the prefix-cache
    /// engine) are expected to have prepared the cache state explicitly.
    ///
    /// Increments the [`prefill_token_count`](Self::prefill_token_count)
    /// counter by `prompt_tokens.len()` on success.
    pub fn prefill_from_pos(
        &mut self,
        prompt_tokens: &[u32],
        pos_start: usize,
    ) -> RuntimeResult<Vec<f32>> {
        let logits = self
            .model
            .forward_prefill(prompt_tokens, pos_start, &self.kernel)?;
        self.prefill_token_count = self
            .prefill_token_count
            .saturating_add(prompt_tokens.len() as u64);
        Ok(logits)
    }

    /// Forward one token at the given absolute position.
    pub fn decode_step(&mut self, token: u32, pos: usize) -> RuntimeResult<Vec<f32>> {
        Ok(self.model.forward(token, pos, &self.kernel)?)
    }

    /// Sample one token from `logits` using the engine's current sampler.
    pub fn sample(&mut self, logits: &[f32]) -> RuntimeResult<u32> {
        self.sampler.sample(logits)
    }

    /// Cumulative number of tokens that have been processed by
    /// [`InferenceEngine::prefill_from_pos`] over this engine's lifetime.
    pub fn prefill_token_count(&self) -> u64 {
        self.prefill_token_count
    }

    /// Reset the model state for a new conversation.
    pub fn reset(&mut self) {
        self.model.reset();
    }

    /// Get a shared reference to the engine statistics.
    pub fn stats(&self) -> &Arc<EngineStats> {
        &self.stats
    }

    /// Number of currently active sessions (tracked via stats).
    pub fn active_sessions(&self) -> usize {
        self.stats.active_session_count()
    }

    /// Total number of completed requests (tracked via stats).
    pub fn session_count(&self) -> u64 {
        self.stats.requests_completed()
    }

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

        // ═══════════════════════════════════════════════════════
        // 1. Prefill: batch process all prompt tokens
        // ═══════════════════════════════════════════════════════
        let prefill_start = std::time::Instant::now();
        let mut last_logits = self.model.forward_prefill(prompt_tokens, 0, &self.kernel)?;
        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(prefill_start.elapsed().as_secs_f64());
        }

        // ═══════════════════════════════════════════════════════
        // 2. Decode: sample and generate
        // ═══════════════════════════════════════════════════════
        let decode_start = std::time::Instant::now();
        let mut output_tokens = Vec::with_capacity(max_tokens);

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();

            // Sample next token
            let next_token = self.sampler.sample(&last_logits)?;

            // Check for EOS
            if next_token == EOS_TOKEN_ID {
                tracing::debug!(pos, "EOS token generated");
                break;
            }

            output_tokens.push(next_token);

            // Forward the generated token
            last_logits = self.model.forward(next_token, pos, &self.kernel)?;

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

    /// Generate tokens from a prompt using a specific seed for this run.
    ///
    /// Temporarily overrides the sampler seed for deterministic multi-completion
    /// generation (`n > 1`). The sampler state is replaced for the duration of
    /// this call and then restored.
    pub fn generate_with_seed(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        seed: u64,
        params: &crate::sampling::SamplingParams,
    ) -> RuntimeResult<Vec<u32>> {
        // Swap in a fresh sampler with the given seed
        let old_sampler = std::mem::replace(
            &mut self.sampler,
            crate::sampling::Sampler::new(params.clone(), seed),
        );
        let result = self.generate(prompt_tokens, max_tokens);
        // Restore the original sampler
        self.sampler = old_sampler;
        result
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

        // Prefill: batch process all prompt tokens
        let prefill_start = std::time::Instant::now();
        let mut logits = self.model.forward_prefill(prompt_tokens, 0, &self.kernel)?;
        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(prefill_start.elapsed().as_secs_f64());
        }

        let decode_start = std::time::Instant::now();
        let mut generated = 0;

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();
            let next_token = self.sampler.sample(&logits)?;

            if next_token == EOS_TOKEN_ID {
                tracing::debug!(pos, "EOS token generated (streaming)");
                break;
            }

            // Send token through channel; if receiver dropped, stop generating
            if tx.send(next_token).is_err() {
                tracing::debug!(pos, "receiver dropped, stopping generation");
                break;
            }

            logits = self.model.forward(next_token, pos, &self.kernel)?;
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

        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated,
            "streaming generation complete"
        );

        Ok(generated)
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

        // Prefill: batch process all prompt tokens
        let prefill_start = std::time::Instant::now();
        let mut logits = self.model.forward_prefill(prompt_tokens, 0, &self.kernel)?;
        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(prefill_start.elapsed().as_secs_f64());
        }

        let decode_start = std::time::Instant::now();
        let mut generated = 0;

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();

            let next_token = self.sampler.sample(&logits)?;

            if next_token == EOS_TOKEN_ID {
                tracing::debug!(pos, "EOS token generated (streaming_sync)");
                break;
            }

            if tx.send(next_token).is_err() {
                tracing::debug!(pos, "receiver dropped, stopping generation");
                break;
            }

            logits = self.model.forward(next_token, pos, &self.kernel)?;
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

        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated,
            "streaming sync generation complete"
        );

        Ok(generated)
    }

    /// Greedy generation entirely on GPU (temperature=0, argmax on Metal).
    ///
    /// Runs the full forward pass + argmax in a single GPU command buffer per
    /// token, downloading only the 4-byte token ID instead of the ~607KB logits
    /// vector. Falls back to the normal `generate` path if the GPU greedy path
    /// is not available.
    ///
    /// Returns the generated token IDs (not including the prompt).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[tracing::instrument(skip(self, prompt_tokens), fields(prompt_len = prompt_tokens.len()))]
    pub fn generate_greedy_gpu(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> RuntimeResult<Vec<u32>> {
        if prompt_tokens.is_empty() {
            return Ok(vec![]);
        }

        // ═══════════════════════════════════════════════════════
        // 1. Prefill: batch process all prompt tokens
        // ═══════════════════════════════════════════════════════
        let prefill_start = std::time::Instant::now();
        let last_logits = self.model.forward_prefill(prompt_tokens, 0, &self.kernel)?;
        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(prefill_start.elapsed().as_secs_f64());
        }

        // First decode token: argmax from prefill logits
        let first_token = {
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in last_logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            best_idx
        };

        // ═══════════════════════════════════════════════════════
        // 2. Decode: speculative greedy with n-gram drafting
        // ═══════════════════════════════════════════════════════
        let decode_start = std::time::Instant::now();
        let mut output_tokens = Vec::with_capacity(max_tokens);

        if first_token == EOS_TOKEN_ID {
            self.stats.record_request(0);
            return Ok(vec![]);
        }
        output_tokens.push(first_token);

        // N-gram cache for zero-cost draft generation
        let mut ngram_cache = NgramCache::new();
        ngram_cache.record(prompt_tokens);

        // Running context: prompt + generated tokens (for n-gram lookups)
        let mut context: Vec<u32> = prompt_tokens.to_vec();
        context.push(first_token);

        let speculation_k: usize = 4;
        let mut spec_attempts: u64 = 0;
        let mut spec_accepted_total: u64 = 0;
        let spec_enabled = std::env::var("OXIBONSAI_SPEC")
            .map(|v| v == "1")
            .unwrap_or(false);
        let spec_warmup = 15_usize; // build cache before speculating

        let mut next_token = first_token;
        let mut pos = prompt_tokens.len() + 1;
        let max_pos = prompt_tokens.len() + max_tokens;

        while pos < max_pos && output_tokens.len() < max_tokens {
            let step_start = std::time::Instant::now();
            let tokens_generated = output_tokens.len();

            // Try n-gram draft — skip warmup phase unless explicitly enabled
            let draft = if !spec_enabled || tokens_generated < spec_warmup {
                Vec::new()
            } else {
                ngram_cache.draft(&context, speculation_k)
            };

            // Adaptive: only speculate if recent accuracy > 60%
            // (batch of 5 costs ~4x single token, need high hit rate)
            let spec_ok = if spec_attempts >= 5 {
                let accuracy = spec_accepted_total as f64
                    / (spec_attempts as f64 * speculation_k as f64).max(1.0);
                accuracy > 0.6 || spec_attempts % 20 == 0
            } else {
                true // optimistic for first 5 attempts
            };

            if !draft.is_empty() && spec_ok {
                // ── Speculative path: batch verify ──────────────
                let mut batch = Vec::with_capacity(1 + draft.len());
                batch.push(next_token);
                batch.extend_from_slice(&draft);

                match self
                    .model
                    .forward_prefill_verify(&batch, pos - 1, &self.kernel)
                {
                    Ok(model_preds) => {
                        spec_attempts += 1;

                        // Verify draft against model predictions
                        let mut accepted: usize = 0;
                        for i in 0..draft.len() {
                            if i < model_preds.len() && draft[i] == model_preds[i] {
                                accepted += 1;
                            } else {
                                break;
                            }
                        }
                        spec_accepted_total += accepted as u64;

                        // Collect accepted draft tokens + bonus
                        let mut eos_seen = false;
                        for &token in draft.iter().take(accepted) {
                            if token == EOS_TOKEN_ID {
                                eos_seen = true;
                                break;
                            }
                            output_tokens.push(token);
                            context.push(token);
                        }

                        if !eos_seen {
                            // Bonus: model's prediction at the accept/reject boundary
                            let bonus = if accepted < model_preds.len() {
                                model_preds[accepted]
                            } else {
                                // All draft tokens matched, take the last prediction
                                match model_preds.last() {
                                    Some(&tok) => tok,
                                    None => break,
                                }
                            };

                            if bonus == EOS_TOKEN_ID {
                                tracing::debug!(pos, accepted, "EOS from speculative bonus");
                                break;
                            }

                            output_tokens.push(bonus);
                            context.push(bonus);
                            next_token = bonus;
                            pos += accepted + 1;

                            // Update n-gram cache with the newly accepted window
                            let window_start = context.len().saturating_sub(accepted + 4);
                            ngram_cache.record(&context[window_start..]);
                        } else {
                            tracing::debug!(pos, accepted, "EOS in draft tokens");
                            break;
                        }
                    }
                    Err(_e) => {
                        // Speculative verify failed — fall through to single-token decode
                        tracing::debug!("speculative verify failed, using single-token decode");
                        match self.model.forward_greedy_gpu(next_token, pos - 1) {
                            Ok(token_id) => {
                                if token_id == EOS_TOKEN_ID {
                                    tracing::debug!(pos, "EOS token generated (greedy GPU)");
                                    break;
                                }
                                output_tokens.push(token_id);
                                context.push(token_id);
                                let window_start = context.len().saturating_sub(3);
                                ngram_cache.record(&context[window_start..]);
                                next_token = token_id;
                                pos += 1;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e, pos,
                                    "greedy GPU path failed, falling back to normal forward"
                                );
                                let logits =
                                    self.model.forward(next_token, pos - 1, &self.kernel)?;
                                let mut best_idx = 0u32;
                                let mut best_val = f32::NEG_INFINITY;
                                for (i, &v) in logits.iter().enumerate() {
                                    if v > best_val {
                                        best_val = v;
                                        best_idx = i as u32;
                                    }
                                }
                                if best_idx == EOS_TOKEN_ID {
                                    tracing::debug!(pos, "EOS from CPU fallback");
                                    break;
                                }
                                output_tokens.push(best_idx);
                                context.push(best_idx);
                                let window_start = context.len().saturating_sub(3);
                                ngram_cache.record(&context[window_start..]);
                                next_token = best_idx;
                                pos += 1;
                            }
                        }
                    }
                }
            } else {
                // ── Single-token decode (no draft or accuracy too low) ──
                match self.model.forward_greedy_gpu(next_token, pos - 1) {
                    Ok(token_id) => {
                        if token_id == EOS_TOKEN_ID {
                            tracing::debug!(pos, "EOS token generated (greedy GPU)");
                            break;
                        }
                        output_tokens.push(token_id);
                        context.push(token_id);
                        let window_start = context.len().saturating_sub(3);
                        ngram_cache.record(&context[window_start..]);
                        next_token = token_id;
                        pos += 1;
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e, pos,
                            "greedy GPU path failed, falling back to normal forward"
                        );
                        let logits = self.model.forward(next_token, pos - 1, &self.kernel)?;
                        let mut best_idx = 0u32;
                        let mut best_val = f32::NEG_INFINITY;
                        for (i, &v) in logits.iter().enumerate() {
                            if v > best_val {
                                best_val = v;
                                best_idx = i as u32;
                            }
                        }
                        if best_idx == EOS_TOKEN_ID {
                            tracing::debug!(pos, "EOS from CPU fallback");
                            break;
                        }
                        output_tokens.push(best_idx);
                        context.push(best_idx);
                        let window_start = context.len().saturating_sub(3);
                        ngram_cache.record(&context[window_start..]);
                        next_token = best_idx;
                        pos += 1;
                    }
                }
            }

            if let Some(m) = &self.metrics {
                m.decode_token_duration_seconds
                    .observe(step_start.elapsed().as_secs_f64());
            }

            // Check for EOS from single-token path
            if output_tokens.last() == Some(&EOS_TOKEN_ID) {
                output_tokens.pop(); // Don't include EOS in output
                break;
            }
        }

        // Log speculative decode statistics
        if spec_attempts > 0 {
            let avg_accepted = spec_accepted_total as f64 / spec_attempts as f64;
            let accuracy =
                spec_accepted_total as f64 / (spec_attempts as f64 * speculation_k as f64).max(1.0);
            tracing::info!(
                spec_attempts,
                spec_accepted_total,
                avg_accepted = format!("{:.2}", avg_accepted),
                accuracy = format!("{:.1}%", accuracy * 100.0),
                "speculative decode stats"
            );
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

        self.stats.record_request(output_tokens.len());

        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated = output_tokens.len(),
            "greedy GPU generation complete"
        );

        Ok(output_tokens)
    }
}

impl InferenceEngine<'static> {
    /// Load an [`InferenceEngine`] directly from a path to a GGUF file.
    ///
    /// This is a convenience wrapper intended for server/CLI entry points that
    /// need an owned, `'static` engine.  It memory-maps the file, parses the
    /// GGUF container, and leaks both allocations so that the borrowed
    /// `GgufFile<'a>` lifetime can be promoted to `'static`.
    ///
    /// The leaked memory is intentional — the engine is expected to live for
    /// the process lifetime.  Do not call this in hot-paths.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::FileNotFound`] if `path` does not exist.  Other
    /// IO / parse / model-init errors propagate through [`RuntimeError`].
    pub fn from_gguf_path(
        path: impl AsRef<std::path::Path>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Self> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Err(RuntimeError::FileNotFound {
                path: path_ref.display().to_string(),
            });
        }

        // Memory-map and parse, then leak both so the resulting `GgufFile`
        // can live for `'static` without RAII concerns.
        let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(path_ref)?;
        let mmap: &'static memmap2::Mmap = Box::leak(Box::new(mmap));
        let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(mmap)?;
        let gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static> =
            Box::leak(Box::new(gguf));

        Self::from_gguf(gguf, sampling_params, seed, max_seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_creation() {
        let config = Qwen3Config::bonsai_8b();
        let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        assert_eq!(engine.model().config().num_layers, 36);
    }

    #[test]
    fn engine_stats_initial() {
        let config = Qwen3Config::bonsai_8b();
        let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        let stats = engine.stats();
        assert_eq!(stats.tokens_generated(), 0);
        assert_eq!(stats.requests_completed(), 0);
        assert_eq!(stats.active_session_count(), 0);
        assert!(stats.uptime_seconds() >= 0.0);
        assert!((stats.avg_tokens_per_request() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn engine_stats_record() {
        let stats = EngineStats::new();
        stats.record_request(10);
        stats.record_request(20);
        assert_eq!(stats.tokens_generated(), 30);
        assert_eq!(stats.requests_completed(), 2);
        assert!((stats.avg_tokens_per_request() - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn engine_session_tracking() {
        let config = Qwen3Config::bonsai_8b();
        let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        assert_eq!(engine.active_sessions(), 0);
        assert_eq!(engine.session_count(), 0);
    }

    #[test]
    fn engine_batch_generate_empty() {
        let config = Qwen3Config::bonsai_8b();
        let mut engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        let results = engine.batch_generate(&[], 10);
        assert!(results.is_empty());
        assert_eq!(engine.session_count(), 0);
    }

    #[test]
    fn engine_batch_generate_empty_prompts() {
        let config = Qwen3Config::bonsai_8b();
        let mut engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        let prompts = vec![vec![], vec![]];
        let results = engine.batch_generate(&prompts, 5);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.is_ok());
        }
        // Stats should reflect the completed requests
        assert_eq!(engine.stats().requests_completed(), 2);
    }

    #[test]
    fn engine_stats_default() {
        let stats = EngineStats::default();
        assert_eq!(stats.tokens_generated(), 0);
        assert_eq!(stats.requests_completed(), 0);
    }
}
