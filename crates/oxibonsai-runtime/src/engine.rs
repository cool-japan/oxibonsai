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
use oxibonsai_kernels::{KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;

use crate::batch_engine::{self, BatchResult};
use crate::error::{RuntimeError, RuntimeResult};
use crate::metrics::InferenceMetrics;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::ngram_cache::NgramCache;
use crate::request_id::RequestId;
use crate::request_metrics::{RequestRateAggregator, RequestRateSnapshot, RequestRateTracker};
use crate::sampling::{PenaltyParams, Sampler, SamplingParams};

/// Default EOS token id for Qwen3 / Bonsai models.
///
/// Used as a fallback when an engine is built without GGUF metadata (the
/// synthetic-config test paths) or when the loaded GGUF omits the
/// `tokenizer.ggml.eos_token_id` key. GGUF-loaded engines resolve their EOS
/// from that metadata key at construction time; see
/// [`InferenceEngine::eos_token_id`].
pub const EOS_TOKEN_ID: u32 = 151645;

/// Resolve the end-of-sequence token id from a loaded GGUF's tokenizer
/// metadata (`tokenizer.ggml.eos_token_id`), falling back to
/// [`EOS_TOKEN_ID`] when the key is absent.
fn resolve_eos_token_id(gguf: &GgufFile<'_>) -> u32 {
    gguf.metadata
        .get_u32(oxibonsai_core::gguf::tensor_info::keys::TOKENIZER_EOS_TOKEN_ID)
        .unwrap_or(EOS_TOKEN_ID)
}

/// Upper bound on the initial capacity reserved for a generation output buffer.
///
/// Generation still runs all the way to the caller's `max_tokens`; this only
/// caps the *up-front* `Vec::with_capacity` hint so that a hostile or
/// mistaken `max_tokens` (e.g. `usize::MAX`) cannot drive a multi-gigabyte
/// eager allocation before a single token has been produced. The buffer grows
/// on demand past this bound as real tokens are appended.
pub(crate) const MAX_PREALLOC_TOKENS: usize = 65_536;

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
    /// Optional workload-level rate aggregator. When attached, every
    /// `generate_tracked` call records its [`RequestRateSnapshot`] here on
    /// completion, allowing the operator to surface workload-level p50/p95
    /// inter-token latency, EWMA tokens-per-second, and queue-wait gauges
    /// (see [`InferenceMetrics::update_request_rate`]).
    rate_aggregator: Option<Arc<RequestRateAggregator>>,
    /// End-of-sequence token id this engine treats as a stop condition.
    ///
    /// Resolved from the loaded GGUF's `tokenizer.ggml.eos_token_id` metadata
    /// key at construction time, falling back to [`EOS_TOKEN_ID`] for the
    /// synthetic-config paths ([`InferenceEngine::new`], the `from_model*`
    /// constructors) and for GGUFs that omit the key.
    eos_token_id: u32,
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
            rate_aggregator: None,
            eos_token_id: EOS_TOKEN_ID,
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
            rate_aggregator: None,
            eos_token_id: EOS_TOKEN_ID,
        }
    }

    /// Wrap a [`BonsaiModel`], pinning the engine to a specific [`KernelTier`].
    ///
    /// Thin wrapper over [`from_model_with_kernel`](Self::from_model_with_kernel)
    /// using [`KernelDispatcher::with_tier`]. Used by the cross-backend
    /// determinism guard to build one engine on `KernelTier::Reference` (scalar
    /// CPU) and one on `KernelTier::Gpu` (Metal) from the same model bytes and
    /// assert byte-identical greedy output. `new`/auto-detect are left untouched.
    pub fn from_model_with_tier(
        model: BonsaiModel<'a>,
        tier: KernelTier,
        sampling_params: SamplingParams,
        seed: u64,
    ) -> Self {
        Self::from_model_with_kernel(
            model,
            KernelDispatcher::with_tier(tier),
            sampling_params,
            seed,
        )
    }

    /// Create a new inference engine from a loaded GGUF file.
    pub fn from_gguf(
        gguf: &'a GgufFile<'a>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Self> {
        let eos_token_id = resolve_eos_token_id(gguf);
        let model = BonsaiModel::from_gguf(gguf, max_seq_len)?;
        Self::from_model_with_gpu_warmup(model, sampling_params, seed, eos_token_id)
    }

    /// Create an engine from a loaded GGUF file, reusing a pre-loaded, shared
    /// token-embedding table.
    ///
    /// Identical to [`from_gguf`](Self::from_gguf) except the `token_embd`
    /// table is supplied by the caller (via
    /// [`BonsaiModel::from_gguf_with_embd`]) rather than re-dequantized from the
    /// GGUF. The engine pool uses this to share one `Arc<[f32]>` across all
    /// replicas (see [`build_pool_from_gguf`](crate::engine_pool::build_pool_from_gguf)).
    ///
    /// `token_embd` MUST be the dequantized `token_embd.weight` for this exact
    /// GGUF; see [`BonsaiModel::from_gguf_with_embd`] for the contract.
    pub fn from_gguf_with_embd(
        gguf: &'a GgufFile<'a>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
        token_embd: std::sync::Arc<[f32]>,
    ) -> RuntimeResult<Self> {
        let eos_token_id = resolve_eos_token_id(gguf);
        let model = BonsaiModel::from_gguf_with_embd(gguf, max_seq_len, token_embd)?;
        Self::from_model_with_gpu_warmup(model, sampling_params, seed, eos_token_id)
    }

    /// Shared core of [`from_gguf`](Self::from_gguf) and
    /// [`from_gguf_with_embd`](Self::from_gguf_with_embd): given an
    /// already-constructed [`BonsaiModel`], auto-detect the kernel, upload
    /// weights to GPU, run the per-tier warmups, and assemble the engine.
    ///
    /// Factored out so the (substantial) GPU/CUDA warmup logic has exactly one
    /// implementation regardless of how `token_embd` was obtained.
    fn from_model_with_gpu_warmup(
        mut model: BonsaiModel<'a>,
        sampling_params: SamplingParams,
        seed: u64,
        eos_token_id: u32,
    ) -> RuntimeResult<Self> {
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
            rate_aggregator: None,
            eos_token_id,
        })
    }

    /// Attach shared metrics to this engine for recording inference telemetry.
    pub fn set_metrics(&mut self, metrics: Arc<InferenceMetrics>) {
        self.metrics = Some(metrics);
    }

    /// Attach a workload-level [`RequestRateAggregator`] to this engine.
    ///
    /// Once attached, every call to [`InferenceEngine::generate_tracked`] (or
    /// [`InferenceEngine::generate_with_request_id`]) will push its
    /// per-request [`RequestRateSnapshot`] into the aggregator on completion.
    /// The aggregator is reference-counted, so the same instance can be shared
    /// with the Prometheus metrics layer or the admin endpoints.
    pub fn set_rate_aggregator(&mut self, aggregator: Arc<RequestRateAggregator>) {
        self.rate_aggregator = Some(aggregator);
    }

    /// Read-only access to the attached rate aggregator, if any.
    pub fn rate_aggregator(&self) -> Option<&Arc<RequestRateAggregator>> {
        self.rate_aggregator.as_ref()
    }

    /// Get a reference to the model.
    pub fn model(&self) -> &BonsaiModel<'a> {
        &self.model
    }

    /// Cheaply clone a handle to this engine's shared token-embedding table.
    ///
    /// Thin delegate to [`BonsaiModel::shared_token_embd`]. The engine pool
    /// calls this on replica `#1` to extract the one shared `Arc<[f32]>`, which
    /// it then hands to [`InferenceEngine::from_gguf_static_with_embd`] when
    /// building replicas `2..N` — so every replica's `token_embd` is a clone of
    /// the same allocation (one ~1.16 GiB table for the 1.7B, not N).
    pub fn model_token_embd(&self) -> std::sync::Arc<[f32]> {
        self.model.shared_token_embd()
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

    /// Kernel tier this engine dispatches to.
    ///
    /// Feature-agnostic convenience used by the engine pool to decide how many
    /// replicas may safely run in parallel (GPU tiers funnel through a
    /// process-global singleton and are pinned to a single replica).
    pub fn kernel_tier(&self) -> KernelTier {
        self.kernel.tier()
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

    /// Speculative-verify a batch of tokens starting at `pos_start`.
    ///
    /// Runs the tokens through the model in a single batched forward pass and
    /// returns the model's greedy (argmax) prediction for **every** position,
    /// i.e. `out[i]` is the token the model would generate after `tokens[i]`.
    /// This is the target-side scoring primitive used by the two-engine
    /// speculative decoder ([`crate::speculative::SpeculativeDecoder::generate_verified`]).
    ///
    /// Like [`prefill_from_pos`](Self::prefill_from_pos), this does **not**
    /// reset the KV cache: the caller manages committed positions and is
    /// responsible for having primed the cache up to `pos_start`.
    pub fn verify_batch(&mut self, tokens: &[u32], pos_start: usize) -> RuntimeResult<Vec<u32>> {
        Ok(self
            .model
            .forward_prefill_verify(tokens, pos_start, &self.kernel)?)
    }

    /// Roll the model's KV cache back to `committed_len`, discarding any
    /// speculative KV written beyond that position. Companion to
    /// `prefill_from_pos`: a speculative-decoding driver prefills a delta,
    /// drafts past it, then calls this to drop the rejected suffix so the
    /// next committed write targets `committed_len`.
    pub fn rewind_cache(&mut self, committed_len: usize) {
        self.model.kv_cache_mut().truncate(committed_len);
    }

    /// Sample one token from `logits` using the engine's current sampler.
    pub fn sample(&mut self, logits: &[f32]) -> RuntimeResult<u32> {
        self.sampler.sample(logits)
    }

    /// The end-of-sequence token id this engine treats as a stop condition.
    ///
    /// Resolved from the loaded GGUF's `tokenizer.ggml.eos_token_id` metadata
    /// key (falling back to [`EOS_TOKEN_ID`] when built from a synthetic config
    /// or when the key is absent).
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Current frequency / presence penalty parameters
    /// ([`crate::sampling::PenaltyParams`]).
    pub fn penalties(&self) -> PenaltyParams {
        *self.sampler.penalties()
    }

    /// Set the frequency / presence penalties applied during generation.
    ///
    /// Together with [`SamplingParams::repetition_penalty`] this is the seam
    /// through which the OpenAI `frequency_penalty` / `presence_penalty`
    /// request fields reach the decode loop. Penalties are applied over the
    /// generated-token history before sampling (see
    /// [`crate::sampling::Sampler::sample_with_history`]).
    pub fn set_penalties(&mut self, penalties: PenaltyParams) {
        self.sampler.set_penalties(penalties);
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
        let mut output_tokens = Vec::with_capacity(max_tokens.min(MAX_PREALLOC_TOKENS));

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();

            // Sample next token, applying repetition/frequency/presence
            // penalties over the generated-token history so far.
            let next_token = self
                .sampler
                .sample_with_history(&last_logits, &output_tokens)?;

            // Check for EOS
            if next_token == self.eos_token_id {
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

    /// Generate tokens from a prompt while populating a [`RequestRateTracker`].
    ///
    /// Behaves identically to [`InferenceEngine::generate`] but additionally:
    /// - records `record_admission()` immediately on entry,
    /// - records `record_first_token()` for the first sampled token,
    /// - records `record_token()` for every subsequent sampled token,
    /// - on success, pushes the resulting [`RequestRateSnapshot`] into the
    ///   engine's attached [`RequestRateAggregator`] (if any).
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

        let prefill_start = std::time::Instant::now();
        let mut last_logits = self.model.forward_prefill(prompt_tokens, 0, &self.kernel)?;
        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(prefill_start.elapsed().as_secs_f64());
        }

        let decode_start = std::time::Instant::now();
        let mut output_tokens = Vec::with_capacity(max_tokens.min(MAX_PREALLOC_TOKENS));
        let mut first_token_recorded = false;

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();
            let next_token = self
                .sampler
                .sample_with_history(&last_logits, &output_tokens)?;
            if next_token == self.eos_token_id {
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
            last_logits = self.model.forward(next_token, pos, &self.kernel)?;

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
    pub fn generate_with_seed(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        seed: u64,
        params: &crate::sampling::SamplingParams,
    ) -> RuntimeResult<Vec<u32>> {
        // Swap in a fresh sampler with the given seed, carrying over any
        // configured frequency/presence penalties so seeded multi-completion
        // generation honours them just like the primary path.
        let mut fresh = crate::sampling::Sampler::new(params.clone(), seed);
        fresh.set_penalties(*self.sampler.penalties());
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

        let mut last_logits = self.model.forward_prefill(prompt_tokens, 0, &self.kernel)?;
        let cap = max_tokens.min(MAX_PREALLOC_TOKENS);
        let mut output_tokens = Vec::with_capacity(cap);
        let mut logprobs: Vec<crate::api_types::LogprobsContent> = Vec::with_capacity(cap);

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let next_token = self
                .sampler
                .sample_with_history(&last_logits, &output_tokens)?;

            if next_token == self.eos_token_id {
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

            last_logits = self.model.forward(next_token, pos, &self.kernel)?;
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

        // Prefill: batch process all prompt tokens
        let prefill_start = std::time::Instant::now();
        let mut logits = self.model.forward_prefill(prompt_tokens, 0, &self.kernel)?;
        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(prefill_start.elapsed().as_secs_f64());
        }

        let decode_start = std::time::Instant::now();
        let mut generated = 0;
        // Generated-token history for repetition/frequency/presence penalties.
        let mut history: Vec<u32> = Vec::new();

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();
            let next_token = self.sampler.sample_with_history(&logits, &history)?;

            if next_token == self.eos_token_id {
                tracing::debug!(pos, "EOS token generated (streaming)");
                break;
            }

            // Send token through channel; if receiver dropped, stop generating
            if tx.send(next_token).is_err() {
                tracing::debug!(pos, "receiver dropped, stopping generation");
                break;
            }
            history.push(next_token);

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

        // Prefill: batch process all prompt tokens
        let prefill_start = std::time::Instant::now();
        let mut logits = self.model.forward_prefill(prompt_tokens, 0, &self.kernel)?;
        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(prefill_start.elapsed().as_secs_f64());
        }

        let decode_start = std::time::Instant::now();
        let mut generated = 0;
        // Generated-token history for repetition/frequency/presence penalties.
        let mut history: Vec<u32> = Vec::new();

        for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();

            let next_token = self.sampler.sample_with_history(&logits, &history)?;

            if next_token == self.eos_token_id {
                tracing::debug!(pos, "EOS token generated (streaming_sync)");
                break;
            }

            if tx.send(next_token).is_err() {
                tracing::debug!(pos, "receiver dropped, stopping generation");
                break;
            }
            history.push(next_token);

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

    /// Decode one greedy token, preferring the Metal GPU path and falling back
    /// to a **coherent** CPU forward when the GPU path fails (or is disabled via
    /// `force_cpu`).
    ///
    /// The Metal decode path ([`BonsaiModel::forward_greedy_gpu`]) maintains only
    /// the GPU-resident KV cache and never writes `self.model`'s CPU cache, so a
    /// naive CPU `forward()` after a GPU failure would attend over an all-zero
    /// cache and silently corrupt the continuation. The first time we fall through
    /// to the CPU, this rebuilds the CPU KV cache by replaying the committed token
    /// sequence (`committed`, covering positions `0..committed.len()`) through the
    /// scalar-CPU forward, then latches `cpu_fallback_active` so every subsequent
    /// token decodes on the CPU directly and never reads the now-stale GPU cache.
    ///
    /// `cpu_kernel` MUST be a non-GPU dispatcher: `self.kernel` may be
    /// GPU-accelerated, in which case `BonsaiModel::forward` would re-enter the
    /// Metal path and leave the CPU cache empty. A `KernelTier::Reference`
    /// dispatcher forces the CPU block path and is byte-identical to the canonical
    /// CPU reference (see the cross-backend determinism guard).
    ///
    /// Returns the greedy argmax token id.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn greedy_decode_token_with_fallback(
        &mut self,
        committed: &[u32],
        next_token: u32,
        pos: usize,
        cpu_kernel: &KernelDispatcher,
        cpu_fallback_active: &mut bool,
        force_cpu: bool,
    ) -> RuntimeResult<u32> {
        if !*cpu_fallback_active && !force_cpu {
            match self.model.forward_greedy_gpu(next_token, pos - 1) {
                Ok(token_id) => return Ok(token_id),
                Err(e) => {
                    tracing::warn!(
                        error = %e, pos,
                        "Metal greedy GPU decode failed; rebuilding the CPU KV cache from the \
                         committed sequence and continuing on the CPU"
                    );
                }
            }
        }
        let logits = if *cpu_fallback_active {
            // Cache already coherent from an earlier rebuild — normal CPU forward.
            self.model.forward(next_token, pos - 1, cpu_kernel)?
        } else {
            // First CPU fall-through: reconstruct the CPU KV cache from scratch by
            // replaying the committed tokens (positions 0..committed.len()) — this
            // reproduces exactly the cache a pure-CPU generation would have built.
            // The final replayed forward yields the logits for the next token.
            if force_cpu {
                tracing::warn!(
                    pos,
                    committed = committed.len(),
                    "forcing CPU greedy decode; rebuilding the CPU KV cache from the committed sequence"
                );
            }
            self.model.reset();
            let mut last = Vec::new();
            for (p, &tok) in committed.iter().enumerate() {
                last = self.model.forward(tok, p, cpu_kernel)?;
            }
            *cpu_fallback_active = true;
            last
        };
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        Ok(best_idx)
    }

    /// Greedy generation entirely on GPU (temperature=0, argmax on Metal).
    ///
    /// Runs the full forward pass + argmax in a single GPU command buffer per
    /// token, downloading only the 4-byte token ID instead of the ~607KB logits
    /// vector. On a Metal GPU dispatch failure mid-generation the decode
    /// transparently rebuilds the CPU KV cache and continues on the CPU (see
    /// `Self::greedy_decode_token_with_fallback`) rather than emitting a
    /// corrupted continuation.
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
        let mut output_tokens = Vec::with_capacity(max_tokens.min(MAX_PREALLOC_TOKENS));

        if first_token == self.eos_token_id {
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

        // Metal→CPU fallback machinery. `forward_greedy_gpu` maintains only the
        // GPU-resident KV cache; if it fails mid-generation we must NOT continue
        // on the CPU with the all-zero CPU cache (silent corruption). Instead we
        // rebuild the CPU cache from the committed sequence once, then decode the
        // remainder on the CPU. `cpu_fallback_kernel` is pinned to the scalar CPU
        // reference so `BonsaiModel::forward` takes the CPU block path (a
        // GPU-accelerated `self.kernel` would re-enter Metal and never populate
        // the CPU cache).
        let cpu_fallback_kernel = KernelDispatcher::with_tier(KernelTier::Reference);
        let mut cpu_fallback_active = false;
        // Optional debug / test seam: after this many committed tokens, force the
        // remainder of the decode onto the CPU path (exercises the KV-cache
        // rebuild without a real GPU fault). Read once; unset = never force.
        let force_cpu_after: Option<usize> = std::env::var("OXIBONSAI_FORCE_CPU_DECODE_AFTER")
            .ok()
            .and_then(|v| v.parse().ok());

        let mut next_token = first_token;
        let mut pos = prompt_tokens.len() + 1;
        let max_pos = prompt_tokens.len() + max_tokens;

        while pos < max_pos && output_tokens.len() < max_tokens {
            let step_start = std::time::Instant::now();
            let tokens_generated = output_tokens.len();
            let force_cpu = force_cpu_after.is_some_and(|k| tokens_generated >= k);

            // Try n-gram draft — skip warmup phase unless explicitly enabled.
            // Speculation relies on the GPU KV cache, so it is disabled once we
            // fall back to (or are forced onto) the CPU decode path.
            let draft = if !spec_enabled
                || cpu_fallback_active
                || force_cpu
                || tokens_generated < spec_warmup
            {
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
                            if token == self.eos_token_id {
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

                            if bonus == self.eos_token_id {
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
                        // Speculative verify failed — fall through to single-token
                        // decode (GPU, or a coherent CPU fallback).
                        tracing::debug!("speculative verify failed, using single-token decode");
                        let tok = self.greedy_decode_token_with_fallback(
                            &context,
                            next_token,
                            pos,
                            &cpu_fallback_kernel,
                            &mut cpu_fallback_active,
                            force_cpu,
                        )?;
                        if tok == self.eos_token_id {
                            tracing::debug!(pos, "EOS token generated");
                            break;
                        }
                        output_tokens.push(tok);
                        context.push(tok);
                        let window_start = context.len().saturating_sub(3);
                        ngram_cache.record(&context[window_start..]);
                        next_token = tok;
                        pos += 1;
                    }
                }
            } else {
                // ── Single-token decode (GPU, or a coherent CPU fallback) ──
                let tok = self.greedy_decode_token_with_fallback(
                    &context,
                    next_token,
                    pos,
                    &cpu_fallback_kernel,
                    &mut cpu_fallback_active,
                    force_cpu,
                )?;
                if tok == self.eos_token_id {
                    tracing::debug!(pos, "EOS token generated");
                    break;
                }
                output_tokens.push(tok);
                context.push(tok);
                let window_start = context.len().saturating_sub(3);
                ngram_cache.record(&context[window_start..]);
                next_token = tok;
                pos += 1;
            }

            if let Some(m) = &self.metrics {
                m.decode_token_duration_seconds
                    .observe(step_start.elapsed().as_secs_f64());
            }

            // Check for EOS from single-token path
            if output_tokens.last() == Some(&self.eos_token_id) {
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
    /// Build an engine from an already-`'static` [`GgufFile`].
    ///
    /// This is the shared core used both by [`from_gguf_path`](Self::from_gguf_path)
    /// (after it has leaked the mmap + parsed container to `'static`) and by the
    /// engine pool when constructing additional replicas off a single leaked
    /// GGUF — every replica borrows the *same* `&'static GgufFile` zero-copy, so
    /// only per-replica state (KV cache, light wrappers) is duplicated. The
    /// immutable `token_embd` table is shared across replicas via one
    /// `Arc<[f32]>` when the pool builder uses
    /// [`from_gguf_static_with_embd`](Self::from_gguf_static_with_embd).
    ///
    /// Performs no leaking itself; the caller owns the `'static` lifetime.
    ///
    /// # Errors
    ///
    /// Propagates model-init / GPU-cache errors through [`RuntimeError`].
    pub fn from_gguf_static(
        gguf: &'static GgufFile<'static>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Self> {
        // `from_gguf` is generic over the GGUF borrow lifetime; instantiating it
        // at `'static` yields an `InferenceEngine<'static>` directly.
        Self::from_gguf(gguf, sampling_params, seed, max_seq_len)
    }

    /// Build an engine from an already-`'static` [`GgufFile`], reusing a
    /// pre-loaded, shared token-embedding table.
    ///
    /// The `'static`-lifetime twin of
    /// [`from_gguf_with_embd`](Self::from_gguf_with_embd). The engine pool calls
    /// this for replicas `2..N`, passing the `Arc<[f32]>` extracted from replica
    /// `#1` (via [`InferenceEngine::model_token_embd`]) so every replica shares a
    /// single token-embedding allocation instead of re-dequantizing its own
    /// copy. KV caches and light wrappers remain per-replica.
    ///
    /// `token_embd` MUST be the dequantized `token_embd.weight` for this exact
    /// GGUF; see [`BonsaiModel::from_gguf_with_embd`] for the contract.
    pub fn from_gguf_static_with_embd(
        gguf: &'static GgufFile<'static>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
        token_embd: std::sync::Arc<[f32]>,
    ) -> RuntimeResult<Self> {
        Self::from_gguf_with_embd(gguf, sampling_params, seed, max_seq_len, token_embd)
    }

    /// Memory-map + parse a GGUF file and leak both allocations to `'static`,
    /// returning the constructed engine *and* the leaked `&'static GgufFile`.
    ///
    /// The leaked reference lets callers (e.g. the engine pool) build additional
    /// engine replicas off the *same* weights via [`from_gguf_static`](Self::from_gguf_static)
    /// without a second mmap or weight copy. The leaked memory is intentional —
    /// the GGUF is expected to live for the process lifetime.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::FileNotFound`] if `path` does not exist.  Other
    /// IO / parse / model-init errors propagate through [`RuntimeError`].
    pub fn from_gguf_path_leaked(
        path: impl AsRef<std::path::Path>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<(Self, &'static GgufFile<'static>)> {
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
        let gguf: &'static GgufFile<'static> = Box::leak(Box::new(gguf));

        let engine = Self::from_gguf_static(gguf, sampling_params, seed, max_seq_len)?;
        Ok((engine, gguf))
    }

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
        Self::from_gguf_path_leaked(path, sampling_params, seed, max_seq_len)
            .map(|(engine, _gguf)| engine)
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

    // ── EOS resolution ────────────────────────────────────────────────────

    #[test]
    fn resolve_eos_from_gguf_metadata() {
        use oxibonsai_core::gguf::reader::GgufFile;
        use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

        // A base Qwen3 tokenizer, for instance, assigns 151643 rather than the
        // instruct EOS 151645 — the engine must honour the model's own key.
        let mut w = GgufWriter::new();
        w.add_metadata(
            "tokenizer.ggml.eos_token_id",
            MetadataWriteValue::U32(151643),
        );
        let bytes = w.to_bytes().expect("write synthetic gguf");
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");
        assert_eq!(resolve_eos_token_id(&gguf), 151643);
        assert_ne!(resolve_eos_token_id(&gguf), EOS_TOKEN_ID);
    }

    #[test]
    fn resolve_eos_falls_back_when_absent() {
        use oxibonsai_core::gguf::reader::GgufFile;
        use oxibonsai_core::gguf::writer::GgufWriter;

        // No eos metadata key → fall back to the hardcoded Qwen3 default.
        let w = GgufWriter::new();
        let bytes = w.to_bytes().expect("write synthetic gguf");
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");
        assert_eq!(resolve_eos_token_id(&gguf), EOS_TOKEN_ID);
    }

    #[test]
    fn synthetic_engine_uses_default_eos() {
        let config = Qwen3Config::tiny_test();
        let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        assert_eq!(engine.eos_token_id(), EOS_TOKEN_ID);
    }

    // ── Penalty seam wiring ───────────────────────────────────────────────

    #[test]
    fn penalties_default_off_and_settable() {
        let config = Qwen3Config::tiny_test();
        let mut engine = InferenceEngine::new(config, SamplingParams::default(), 42);
        assert!(!engine.penalties().is_active(), "penalties default off");
        engine.set_penalties(PenaltyParams::new(0.5, 0.25));
        assert!(engine.penalties().is_active());
        assert_eq!(engine.penalties().frequency_penalty, 0.5);
        assert_eq!(engine.penalties().presence_penalty, 0.25);
    }

    #[test]
    fn generate_is_deterministic_with_penalties() {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.3,
            max_tokens: 128,
        };
        let prompt = vec![151644u32, 872, 1234];

        let mut e1 = InferenceEngine::new(Qwen3Config::tiny_test(), params.clone(), 7);
        e1.set_penalties(PenaltyParams::new(0.5, 0.5));
        let o1 = e1.generate(&prompt, 16).expect("gen1");

        let mut e2 = InferenceEngine::new(Qwen3Config::tiny_test(), params, 7);
        e2.set_penalties(PenaltyParams::new(0.5, 0.5));
        let o2 = e2.generate(&prompt, 16).expect("gen2");

        assert_eq!(
            o1, o2,
            "same seed + params + penalties must be deterministic"
        );
    }

    #[test]
    fn generate_with_params_and_penalties_restores_state() {
        let prompt = vec![151644u32, 872, 1234];
        let base = SamplingParams::default();
        let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), base.clone(), 11);

        let temp_params = SamplingParams {
            temperature: 0.5,
            repetition_penalty: 1.5,
            ..base.clone()
        };
        let penalties = PenaltyParams::new(0.8, 0.2);
        let _ = engine
            .generate_with_params_and_penalties(&prompt, 8, &temp_params, &penalties)
            .expect("scoped generate");

        // Both params and penalties are restored to their pre-call values.
        assert_eq!(engine.penalties(), PenaltyParams::default());
        assert!((engine.sampler.params().temperature - base.temperature).abs() < f32::EPSILON);
        assert!(
            (engine.sampler.params().repetition_penalty - base.repetition_penalty).abs()
                < f32::EPSILON
        );
    }

    #[cfg(feature = "server")]
    #[test]
    fn generate_with_logprobs_returns_sane_values() {
        let params = SamplingParams {
            temperature: 0.0, // greedy for a stable emitted token
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 128,
        };
        let prompt = vec![151644u32, 872, 1234];
        let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), params, 3);

        let (tokens, logprobs) = engine
            .generate_with_logprobs(&prompt, 6, 5, &|id| format!("tok{id}"))
            .expect("generate_with_logprobs");

        assert_eq!(
            tokens.len(),
            logprobs.len(),
            "one logprob entry per emitted token"
        );
        assert!(!tokens.is_empty(), "greedy tiny model should emit tokens");

        for lp in &logprobs {
            // A log-probability is always <= 0.
            assert!(
                lp.logprob <= 1e-4,
                "chosen-token logprob must be <= 0, got {}",
                lp.logprob
            );
            // top_logprobs is clamped to k (<= 20) and sorted descending.
            assert!(lp.top_logprobs.len() <= 5);
            assert!(!lp.top_logprobs.is_empty());
            for pair in lp.top_logprobs.windows(2) {
                assert!(
                    pair[0].logprob >= pair[1].logprob,
                    "top_logprobs must be sorted descending"
                );
            }
        }
    }

    #[cfg(feature = "server")]
    #[test]
    fn generate_with_logprobs_empty_prompt() {
        let mut engine =
            InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 1);
        let (tokens, logprobs) = engine
            .generate_with_logprobs(&[], 4, 3, &|id| format!("t{id}"))
            .expect("empty prompt ok");
        assert!(tokens.is_empty());
        assert!(logprobs.is_empty());
    }
}
