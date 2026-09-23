//! Inference engine orchestrating model loading and generation.
//!
//! The [`InferenceEngine`] is the main entry point for running inference.
//! It owns the model, kernel dispatcher, and sampler, and provides both
//! blocking ([`InferenceEngine::generate`]) and streaming
//! ([`InferenceEngine::generate_streaming`]) generation APIs.
//!
//! ## One decoding contract
//!
//! Every entry point decodes under the *same* contract: the engine's
//! configured penalties are applied before any argmax, or the call refuses
//! to run — never silently ignored. Concretely:
//!
//! * [`InferenceEngine::generate`] and friends apply
//!   [`Sampler::sample_with_history`] (repetition + frequency/presence
//!   penalties, then temperature/top-k/top-p) at every step.
//! * The GPU-argmax fast path (`crate::engine_greedy`) is taken **only**
//!   when the configured sampler is pure greedy with no penalties. `generate`
//!   / `generate_tracked` / the streaming pair all decide this through the
//!   single predicate [`InferenceEngine::greedy_gpu_eligible`], which also
//!   requires
//!   [`GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`](crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX)
//!   (a wave-2 verifier finding: the GPU argmax kernels were fixed to
//!   tie-break correctly and the gate is now open — see that constant's
//!   doc comment for the fix and how to revert it if a regression is found).
//!   [`InferenceEngine::generate_greedy_gpu`] is a separate, *direct* entry
//!   point that does not go through `greedy_gpu_eligible` — it checks the
//!   same gate and the same penalty condition itself. Either way, when
//!   penalties are configured, or the tie-break gate is closed, the call
//!   decodes the full logit row and applies penalties before the argmax
//!   instead.
//!
//! This closes `RT-24` and the CPU-vs-Metal greedy divergence: before it,
//! `generate_greedy_gpu` was pure argmax and consulted neither the sampler
//! nor its penalties, so the CLI's hardcoded `repetition_penalty: 1.1`
//! changed CPU output while leaving Metal output untouched.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::{KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;

use crate::batch_engine::{self, BatchResult};
use crate::engine_control::{
    gguf_fused_metal_route, resolve_eos_token_set, CancellationToken, EosTokenSet, FusedMetalRoute,
    RecurrentState, SpeculativeConfig, GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX,
};
use crate::error::{RuntimeError, RuntimeResult};
use crate::metrics::InferenceMetrics;
use crate::request_id::RequestId;
use crate::request_metrics::{RequestRateAggregator, RequestRateSnapshot, RequestRateTracker};
use crate::sampling::{PenaltyParams, Sampler, SamplingParams};

/// Default EOS token id for Qwen3 / Bonsai models.
///
/// Used as a fallback when an engine is built without GGUF metadata (the
/// synthetic-config test paths) or when the loaded GGUF omits the
/// `tokenizer.ggml.eos_token_id` key. GGUF-loaded engines resolve their EOS
/// set from that metadata at construction time; see
/// [`InferenceEngine::eos_token_ids`].
pub const EOS_TOKEN_ID: u32 = 151645;

/// Temperature below which sampling is exactly greedy (argmax).
///
/// Mirrors `Sampler::sample_core`'s own threshold; shared so the greedy
/// routing predicate and the sampler can never disagree about what "greedy"
/// means.
pub(crate) const GREEDY_TEMPERATURE_EPS: f32 = 1e-6;

/// Whether `kernel` dispatches to a GPU backend.
///
/// `KernelTier::Gpu` is only compiled in when a GPU backend feature is
/// enabled (the variant does not exist otherwise), so the comparison has to
/// be cfg-guarded — the same shape `engine_pool::resolve_pool_size` uses.
/// A build without `metal`/`native-cuda` — including the advertised
/// `wasm32-unknown-unknown --no-default-features` build — always answers
/// `false`.
fn kernel_is_gpu_tier(kernel: &KernelDispatcher) -> bool {
    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    {
        kernel.tier() == KernelTier::Gpu
    }
    #[cfg(not(any(feature = "metal", feature = "native-cuda")))]
    {
        let _ = kernel;
        false
    }
}

/// Resolve the end-of-sequence token id from a loaded GGUF's tokenizer
/// metadata (`tokenizer.ggml.eos_token_id`), falling back to
/// [`EOS_TOKEN_ID`] when the key is absent.
///
/// Thin wrapper over [`resolve_eos_token_set`] returning only the primary
/// id. The engine itself carries the whole set (`RT-18`); this exists for
/// the single-id assertions in this module's tests.
#[cfg(test)]
fn resolve_eos_token_id(gguf: &GgufFile<'_>) -> u32 {
    resolve_eos_token_set(gguf, EOS_TOKEN_ID).primary()
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
    pub(crate) model: BonsaiModel<'a>,
    pub(crate) kernel: KernelDispatcher,
    pub(crate) sampler: Sampler,
    pub(crate) metrics: Option<Arc<InferenceMetrics>>,
    pub(crate) stats: Arc<EngineStats>,
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
    /// End-of-sequence token ids this engine treats as stop conditions
    /// (`RT-18`).
    ///
    /// Resolved from the loaded GGUF's tokenizer metadata at construction
    /// time (`tokenizer.ggml.eos_token_id`, `tokenizer.ggml.eot_token_id`,
    /// plus any terminator token looked up by name in the file's own
    /// vocabulary — see [`resolve_eos_token_set`]), falling back to the
    /// single [`EOS_TOKEN_ID`] for the synthetic-config paths
    /// ([`InferenceEngine::new`], the `from_model*` constructors) and for
    /// GGUFs that omit the keys.
    pub(crate) eos: EosTokenSet,
    /// Cooperative cancellation flag, armed per request by the caller
    /// (`SV-09`). Checked once per decode step by every generation path and,
    /// when [`prefill_chunk_tokens`](Self::prefill_chunk_tokens) is set,
    /// once per prefill chunk.
    pub(crate) cancel: Option<CancellationToken>,
    /// Hybrid-model recurrent state, when one has been attached (`RT-28`).
    ///
    /// Cleared by [`InferenceEngine::reset_recurrent`], which
    /// [`InferenceEngine::reset`] — and therefore the server's per-request
    /// reset — always calls.
    recurrent: Option<Box<dyn RecurrentState>>,
    /// Speculative-decode configuration for the GPU greedy path
    /// (`RT-27` / `perf-16`).
    pub(crate) speculative: SpeculativeConfig,
    /// Prefill chunk size, or `None` (default) for a single batched prefill
    /// call.
    ///
    /// `Some(n)` splits a prompt into `n`-token prefill calls so an armed
    /// [`CancellationToken`] is observed *during* a long prefill rather than
    /// only after it. Left `None` by default so the shipping prefill path —
    /// including its batched GPU kernels — stays byte-for-byte the one the
    /// determinism gates measure.
    prefill_chunk_tokens: Option<usize>,
    /// Whether this engine's model decodes through the fused Metal graph
    /// (`MET-M1`), decided from the GGUF's quantization layout at load time.
    ///
    /// Two consumers: it suppresses the redundant `Scirs2Backend` weight
    /// upload at construction, and it is a precondition for routing greedy
    /// generation through the GPU argmax (`perf-11`). Always `false` for
    /// engines built from a synthetic config, which therefore keep exactly
    /// their previous behaviour.
    pub(crate) fused_gpu_decode: bool,
}

impl<'a> InferenceEngine<'a> {
    /// Create a new inference engine from a configuration (no weights — for testing).
    pub fn new(config: Qwen3Config, sampling_params: SamplingParams, seed: u64) -> Self {
        let model = BonsaiModel::new(config);
        let kernel = KernelDispatcher::auto_detect();
        let sampler = Sampler::new(sampling_params, seed);

        tracing::info!(kernel = kernel.name(), "inference engine initialized");

        Self::assemble(
            model,
            kernel,
            sampler,
            EosTokenSet::single(EOS_TOKEN_ID),
            false,
        )
    }

    /// Assemble an engine from its already-built parts.
    ///
    /// The single place the non-model fields get their initial values, so a
    /// new control-plane field cannot be silently forgotten by one of the
    /// six public constructors.
    fn assemble(
        model: BonsaiModel<'a>,
        kernel: KernelDispatcher,
        sampler: Sampler,
        eos: EosTokenSet,
        fused_gpu_decode: bool,
    ) -> Self {
        Self {
            model,
            kernel,
            sampler,
            metrics: None,
            stats: Arc::new(EngineStats::new()),
            prefill_token_count: 0,
            rate_aggregator: None,
            eos,
            cancel: None,
            recurrent: None,
            // `RT-27`: the documented default (off) with the legacy
            // `OXIBONSAI_SPEC` environment override still honoured, read
            // once here rather than per call inside the decode loop.
            speculative: SpeculativeConfig::default().with_env_override(),
            prefill_chunk_tokens: None,
            fused_gpu_decode,
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
        Self::assemble(
            model,
            kernel,
            sampler,
            EosTokenSet::single(EOS_TOKEN_ID),
            false,
        )
    }

    /// Wrap a [`BonsaiModel`], pinning the engine to a specific [`KernelTier`].
    ///
    /// Thin wrapper over [`from_model_with_kernel`](Self::from_model_with_kernel)
    /// using [`KernelDispatcher::with_tier`]. Used by the cross-backend
    /// determinism guard to build one engine on `KernelTier::Reference` (scalar
    /// CPU) and one on `KernelTier::Gpu` (Metal) from the same model bytes and
    /// assert byte-identical greedy output. `new`/auto-detect are left untouched.
    ///
    /// `K-03`/`sec-14`: `with_tier` re-validates the requested tier against
    /// the CPU's actual feature set and silently demotes an unsupported one
    /// (an `Avx512` request on a CPU without AVX-512 used to reach the
    /// `unsafe { #[target_feature] }` kernels and trap). A demotion is
    /// *silent* here by design — use
    /// [`try_from_model_with_tier`](Self::try_from_model_with_tier) to be
    /// told about it, and [`effective_tier_reason`](Self::effective_tier_reason)
    /// to surface what actually happened.
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

    /// Wrap a [`BonsaiModel`] on an explicitly-requested [`KernelTier`],
    /// failing when this machine cannot execute it.
    ///
    /// The checked sibling of [`from_model_with_tier`](Self::from_model_with_tier)
    /// (`K-03` follow-through): it routes through
    /// [`KernelDispatcher::try_with_tier`], so an `Avx2`/`Avx512` request on
    /// a CPU lacking those extensions — or a `Gpu` request with no
    /// accelerated backend, which would otherwise run every operation on the
    /// CPU fallback at roughly 1/7 the throughput (`perf-13`) — is an error
    /// instead of a quiet degradation. Prefer it wherever the tier comes
    /// from a user-facing flag or configuration file.
    ///
    /// # Errors
    ///
    /// [`RuntimeError::Kernel`] when the tier is unsupported on this host.
    pub fn try_from_model_with_tier(
        model: BonsaiModel<'a>,
        tier: KernelTier,
        sampling_params: SamplingParams,
        seed: u64,
    ) -> RuntimeResult<Self> {
        let kernel = KernelDispatcher::try_with_tier(tier)?;
        Ok(Self::from_model_with_kernel(
            model,
            kernel,
            sampling_params,
            seed,
        ))
    }

    /// Create a new inference engine from a loaded GGUF file.
    pub fn from_gguf(
        gguf: &'a GgufFile<'a>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Self> {
        let eos = resolve_eos_token_set(gguf, EOS_TOKEN_ID);
        let route = gguf_fused_metal_route(gguf);
        // A `qwen35` (PrismML Bonsai 2) file is structurally a different
        // stack -- 48 of its 64 layers have no `attn_q.weight` at all -- so
        // `BonsaiModel::from_gguf` would fail on a missing tensor and say
        // nothing about why. Name the seam instead (gatekeeper REQUIRED
        // #1(b)); `LoadedModel` is what routes the two kinds.
        if oxibonsai_model::hybrid::LoadedModel::is_hybrid_gguf(gguf) {
            return Err(RuntimeError::Config(format!(
                "this GGUF declares general.architecture = \"{}\", a hybrid stack the dense \
                 InferenceEngine cannot execute; load it through \
                 oxibonsai_model::hybrid::LoadedModel::from_gguf (or HybridModel::from_gguf), \
                 which selects the hybrid forward driver",
                oxibonsai_model::hybrid::LoadedModel::architecture_of(gguf)
            )));
        }
        let model = BonsaiModel::from_gguf(gguf, max_seq_len)?;
        Self::from_model_with_gpu_warmup(model, sampling_params, seed, eos, route)
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
        let eos = resolve_eos_token_set(gguf, EOS_TOKEN_ID);
        let route = gguf_fused_metal_route(gguf);
        let model = BonsaiModel::from_gguf_with_embd(gguf, max_seq_len, token_embd)?;
        Self::from_model_with_gpu_warmup(model, sampling_params, seed, eos, route)
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
        eos: EosTokenSet,
        route: FusedMetalRoute,
    ) -> RuntimeResult<Self> {
        let kernel = KernelDispatcher::auto_detect();

        // `MET-M1`: `upload_weights_to_gpu` fills `Scirs2Backend::weight_cache`
        // — a second, never-evicted, GPU-resident copy of every quantized
        // tensor (measured: 197 tensors / 435.69 MB on the ternary 1.7B, i.e.
        // the whole quantized model). On the **ternary** fused route nothing
        // in the shipping decode path ever reads those buffers, so the upload
        // is waste for the entire life of the process.
        //
        // It is skipped only when the GGUF's layout proves that route
        // (`FusedMetalRoute::gpu_weight_upload_redundant`). Every other model
        // uploads exactly as before — including the **one-bit** fused route,
        // whose `MetalGraph` cache is keyed on the very handles this call
        // produces, and **all** CUDA builds, where block-dispatch GEMV reads
        // these weights directly.
        let skip_redundant_upload = cfg!(all(feature = "metal", target_os = "macos"))
            && route.gpu_weight_upload_redundant()
            && kernel_is_gpu_tier(&kernel);
        if skip_redundant_upload {
            tracing::info!(
                "skipping the scirs2 GPU weight upload: this ternary model decodes through the \
                 fused Metal graph, which keeps its own weight cache (MET-M1)"
            );
        } else {
            // Upload all model weights to GPU memory once (no-op on CPU-only tiers).
            model.upload_weights_to_gpu(&kernel);
        }

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

        tracing::info!(
            kernel = kernel.name(),
            eos = ?eos.as_slice(),
            fused_route = ?route,
            "inference engine loaded from GGUF"
        );

        Ok(Self::assemble(
            model,
            kernel,
            sampler,
            eos,
            route.is_fused() && cfg!(all(feature = "metal", target_os = "macos")),
        ))
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
        if let Err(e) = self.try_rewind_cache(committed_len) {
            // Not a silent half-rollback: a caller using the infallible
            // spelling still gets a loud, coded record of the refusal.
            tracing::error!(
                error = %e,
                committed_len,
                "rewind_cache refused: this engine holds recurrent state that cannot be \
                 rolled back by moving a cursor -- use try_rewind_cache and handle the error, \
                 or snapshot/restore around the rollback window"
            );
        }
    }

    /// Fallible [`rewind_cache`](Self::rewind_cache) (RT-28, design SS3.6).
    ///
    /// A KV cache rolls back by moving a cursor: every stored key/value is
    /// still the same function of the same token, so truncating to
    /// `committed_len` is exact. A **recurrent** state has no such
    /// property -- `S` after `n + k` tokens does not determine `S` after
    /// `n` -- so on a hybrid (`qwen35`) engine, truncating only the KV
    /// cache leaves the Gated-DeltaNet state contaminated with tokens the
    /// caller believes it discarded, and every subsequent token is wrong
    /// with no signal. Speculative decoding and the pipeline are the two
    /// callers; both must either take a
    /// [`RecurrentCache::snapshot`](oxibonsai_model::hybrid::RecurrentCache::snapshot)
    /// at the draft start and restore it, or refuse hybrid models.
    ///
    /// # Errors
    ///
    /// [`ModelError::RecurrentRollbackUnsupported`] when recurrent state is
    /// attached to this engine.
    pub fn try_rewind_cache(&mut self, committed_len: usize) -> RuntimeResult<()> {
        if let Some(state) = self.recurrent.as_ref() {
            let bytes = state.recurrent_memory_bytes();
            return Err(RuntimeError::Model(
                oxibonsai_model::error::ModelError::RecurrentRollbackUnsupported {
                    pos: committed_len,
                    tokens: bytes,
                },
            ));
        }
        self.model.kv_cache_mut().truncate(committed_len);
        Ok(())
    }

    /// `true` when this engine holds recurrent state, i.e. when
    /// [`try_rewind_cache`](Self::try_rewind_cache) will refuse.
    ///
    /// Speculative decoding and the pipeline check this before starting a
    /// draft window rather than discovering it at rollback time.
    #[must_use]
    pub fn recurrent_rollback_supported(&self) -> bool {
        self.recurrent.is_none()
    }

    /// Sample one token from `logits` using the engine's current sampler.
    pub fn sample(&mut self, logits: &[f32]) -> RuntimeResult<u32> {
        self.sampler.sample(logits)
    }

    /// The **primary** end-of-sequence token id this engine treats as a stop
    /// condition.
    ///
    /// Resolved from the loaded GGUF's `tokenizer.ggml.eos_token_id` metadata
    /// key (falling back to [`EOS_TOKEN_ID`] when built from a synthetic config
    /// or when the key is absent). Generation stops on *any* id in
    /// [`eos_token_ids`](Self::eos_token_ids); this accessor exists for
    /// callers that can only carry one (e.g. `BeamSearchConfig`).
    pub fn eos_token_id(&self) -> u32 {
        self.eos.primary()
    }

    /// Every token id that terminates generation, primary first (`RT-18`).
    ///
    /// A single id is not enough: Bonsai 2 ends a turn on `<|im_end|>`
    /// (`248046`) but its chat contract also terminates on `<|endoftext|>`
    /// (`248044`), and models that distinguish "end of turn" from "end of
    /// text" publish both through `tokenizer.ggml.eot_token_id`. The set is
    /// resolved from the GGUF itself — ids are never hardcoded per model.
    pub fn eos_token_ids(&self) -> &[u32] {
        self.eos.as_slice()
    }

    /// Whether `token` terminates generation for this engine.
    ///
    /// The one predicate every decode loop uses, so widening the set can
    /// never leave a path comparing against a single id.
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(token)
    }

    /// Replace the end-of-sequence set.
    ///
    /// The first id becomes the primary. An empty iterator is ignored (the
    /// set is non-empty by construction), so a caller cannot accidentally
    /// build an engine that never stops.
    pub fn set_eos_token_ids<I: IntoIterator<Item = u32>>(&mut self, ids: I) {
        let mut iter = ids.into_iter();
        let Some(primary) = iter.next() else {
            tracing::warn!("ignoring an empty EOS token set; keeping the current one");
            return;
        };
        self.eos = EosTokenSet::with_extras(primary, iter);
    }

    /// Arm this engine with a cooperative cancellation token (`SV-09`).
    ///
    /// Every generation path checks the token once per decode step (and per
    /// prefill chunk when [`set_prefill_chunk_tokens`](Self::set_prefill_chunk_tokens)
    /// is configured) and stops early, returning the tokens produced so far
    /// — the same shape as hitting EOS. Cancellation is therefore never an
    /// error and never discards completed work.
    ///
    /// Intended for a server's per-request timeout / disconnect branch: hold
    /// one clone, hand the engine another. [`reset`](Self::reset) detaches
    /// the token, so a cancelled request cannot poison the next one to be
    /// served by the same pooled replica.
    pub fn set_cancellation_token(&mut self, token: CancellationToken) {
        self.cancel = Some(token);
    }

    /// The currently-armed cancellation token, if any.
    pub fn cancellation_token(&self) -> Option<&CancellationToken> {
        self.cancel.as_ref()
    }

    /// Detach the cancellation token without cancelling it.
    pub fn clear_cancellation_token(&mut self) {
        self.cancel = None;
    }

    /// Whether an armed token has been cancelled. `false` when none is armed.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.is_cancelled())
    }

    /// Attach the model's recurrent (linear-attention) state so the engine
    /// can reset it between requests (`RT-28`).
    ///
    /// See [`RecurrentState`] for why the hybrid model needs this and what
    /// `B2-10` has to implement.
    pub fn set_recurrent_state(&mut self, state: Box<dyn RecurrentState>) {
        tracing::debug!(
            name = state.recurrent_name(),
            bytes = state.recurrent_memory_bytes(),
            "recurrent state attached to engine"
        );
        self.recurrent = Some(state);
    }

    /// Mutable access to the attached recurrent state, if any.
    ///
    /// The `'static` bound is the `Box<dyn RecurrentState>` field's own: a
    /// `&mut` trait object is invariant in its lifetime, so the bound cannot
    /// be shortened to the borrow. `B2-10`'s `RecurrentCache` owns its
    /// buffers, so this costs it nothing.
    pub fn recurrent_state_mut(&mut self) -> Option<&mut (dyn RecurrentState + 'static)> {
        self.recurrent.as_deref_mut()
    }

    /// Detach the recurrent state, returning it to the caller.
    pub fn take_recurrent_state(&mut self) -> Option<Box<dyn RecurrentState>> {
        self.recurrent.take()
    }

    /// Bytes of recurrent state held by this engine (`0` when none is
    /// attached).
    pub fn recurrent_memory_bytes(&self) -> usize {
        self.recurrent
            .as_ref()
            .map_or(0, |s| s.recurrent_memory_bytes())
    }

    /// Clear the hybrid model's recurrent/conv state (`RT-28`).
    ///
    /// A no-op when no recurrent state is attached (every non-hybrid model
    /// today). Called by [`reset`](Self::reset), so the server's
    /// per-request reset (`RT-03`) already covers it — unlike a KV cache,
    /// recurrent state is not masked by position, so a stale `S` matrix
    /// would silently contaminate the next request rather than being
    /// overwritten.
    pub fn reset_recurrent(&mut self) {
        if let Some(state) = self.recurrent.as_deref_mut() {
            state.reset_recurrent();
        }
    }

    /// Current speculative-decode configuration (`RT-27` / `perf-16`).
    pub fn speculative(&self) -> SpeculativeConfig {
        self.speculative
    }

    /// Configure speculative decoding for the GPU greedy path.
    ///
    /// Replaces the undocumented `OXIBONSAI_SPEC` environment switch that
    /// used to be read inside the decode function. The environment variable
    /// still works as an override at construction time; an explicit call
    /// here always wins.
    pub fn set_speculative(&mut self, config: SpeculativeConfig) {
        self.speculative = config;
    }

    /// Prefill chunk size, or `None` for a single batched prefill call.
    pub fn prefill_chunk_tokens(&self) -> Option<usize> {
        self.prefill_chunk_tokens
    }

    /// Split prefill into chunks of at most `tokens` so an armed
    /// [`CancellationToken`] is observed *during* a long prompt ingest.
    ///
    /// `None` (the default) keeps the single-call prefill the determinism
    /// gates measure. `Some(0)` is treated as `None`.
    ///
    /// Intended for a server that also arms a cancellation token: a 2.5 k
    /// prompt can spend tens of seconds in prefill, during which a
    /// non-chunked engine cannot observe a deadline at all.
    pub fn set_prefill_chunk_tokens(&mut self, tokens: Option<usize>) {
        self.prefill_chunk_tokens = tokens.filter(|&n| n > 0);
    }

    /// Human-readable explanation of how this engine's effective kernel tier
    /// was chosen — e.g. `"gpu tier (backend=metal)"` or `"neon tier
    /// (requested avx2 not supported by this CPU, demoted)"`.
    ///
    /// `K-03`/`perf-13`: a requested tier can be demoted (an unsupported
    /// SIMD level) or silently degraded (a `Gpu` request with no accelerated
    /// backend runs everything on the CPU fallback). For the CLI build-info
    /// surface (`cli-19`) and `/admin/status`, so that degradation is
    /// observable rather than inferred from throughput.
    pub fn effective_tier_reason(&self) -> String {
        self.kernel.effective_tier_reason()
    }

    /// Whether this engine's model decodes through the fused Metal graph
    /// (`MET-M1` / `perf-11`).
    ///
    /// `true` only for a GGUF-loaded, all-ternary or all-1-bit model on a
    /// Metal build; it gates both the suppressed duplicate weight upload and
    /// the GPU-argmax greedy routing.
    pub fn uses_fused_gpu_decode(&self) -> bool {
        self.fused_gpu_decode && kernel_is_gpu_tier(&self.kernel)
    }

    /// The sampling parameters currently configured on this engine.
    ///
    /// Together with [`penalties`](Self::penalties) this is the complete
    /// decoding configuration a request will run under — the same thing
    /// [`greedy_gpu_eligible`](Self::greedy_gpu_eligible) reads, and what
    /// `/admin/status` should report rather than echoing the request.
    pub fn sampling_params(&self) -> &SamplingParams {
        self.sampler.params()
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
    ///
    /// Clears the KV cache and the hybrid recurrent/conv state
    /// ([`reset_recurrent`](Self::reset_recurrent), `RT-28`). This is the
    /// seam the server's per-request reset (`RT-03`) goes through, so a
    /// hybrid model's `S` matrix — which, unlike a KV cache, is not masked by
    /// position — cannot carry over between requests.
    ///
    /// It deliberately does **not** touch an armed
    /// [`CancellationToken`](crate::engine_control::CancellationToken):
    /// `run_blocking_generation` resets the engine *before* running the
    /// caller's closure, so detaching here would silently disarm a token the
    /// request had already armed. A token cannot leak into the *next*
    /// request either, because
    /// [`EngineLease`](crate::engine_pool::EngineLease) clears it when the
    /// replica returns to the pool; a caller owning an engine directly uses
    /// [`clear_cancellation_token`](Self::clear_cancellation_token).
    pub fn reset(&mut self) {
        self.model.reset();
        self.reset_recurrent();
    }

    /// Create a fresh [`CancellationToken`](crate::engine_control::CancellationToken),
    /// arm this engine with it, and return a handle to the caller.
    ///
    /// The one-line form of
    /// [`set_cancellation_token`](Self::set_cancellation_token) for a
    /// server's per-request path: keep the returned handle, cancel it from
    /// the timeout or disconnect branch, and the running generation stops at
    /// its next decode step and returns the tokens produced so far.
    pub fn arm_cancellation(&mut self) -> CancellationToken {
        let token = CancellationToken::new();
        self.cancel = Some(token.clone());
        token
    }

    /// The single predicate deciding whether a generation may take the
    /// GPU-argmax fast path (`RT-24` + `perf-11`).
    ///
    /// Every entry point asks this one function, because the original defect
    /// was exactly a *per-entry-point* decoding contract: the CLI's
    /// temperature-0 path reached `generate_greedy_gpu` (pure argmax) while
    /// the CPU path applied a repetition penalty before its argmax, so the
    /// same flags produced different tokens on the two backends.
    ///
    /// All of the following must hold:
    ///
    /// * [`GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`] is `true` — a wave-2
    ///   verifier finding, kept as an explicit dependency gate rather than
    ///   removing the route: the MSL/CUDA argmax kernels were fixed to
    ///   tie-break toward the global first index (`perf-11` / `FIX2-KERN`),
    ///   so routing here no longer regresses the temperature-0/no-penalty
    ///   cross-tier determinism invariant on any model whose logits contain
    ///   exact duplicates (routinely true of a ternary/1-bit quantized
    ///   model — the only family that can reach this predicate's `true`
    ///   branch at all). If a future kernel regression forces this constant
    ///   back to `false`, this condition alone closes the route again. See
    ///   that constant's doc comment for the verified example and the fix;
    /// * the model decodes through the fused Metal graph on a GPU tier —
    ///   `forward_greedy_gpu` maintains only the *GPU-resident* KV cache, so
    ///   a non-fused model would decode against an all-zero cache;
    /// * the caller does not need the full logit row (`needs_full_logits`):
    ///   log-probabilities cannot be reconstructed from a 4-byte argmax
    ///   readback;
    /// * the configured sampler is exactly greedy: temperature below
    ///   [`GREEDY_TEMPERATURE_EPS`], repetition penalty `1.0`, and no
    ///   frequency/presence penalty. Anything else has to see the logits.
    ///
    /// Note `SamplingParams::default()` carries `repetition_penalty: 1.1`,
    /// so "no penalties" is never the default — it is a deliberate caller
    /// choice, which is why this predicate reads the *sampler*, not the
    /// request.
    pub fn greedy_gpu_eligible(&self, needs_full_logits: bool) -> bool {
        if !GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX || needs_full_logits || !self.uses_fused_gpu_decode()
        {
            return false;
        }
        let params = self.sampler.params();
        params.temperature < GREEDY_TEMPERATURE_EPS
            && params.repetition_penalty == 1.0
            && !self.sampler.penalties().is_active()
    }

    /// Run prefill for a generation, honouring an armed cancellation token.
    ///
    /// Returns `Ok(None)` when cancellation was observed before the prompt
    /// was fully ingested (no logits exist yet, so the caller returns an
    /// empty completion). With [`prefill_chunk_tokens`](Self::prefill_chunk_tokens)
    /// unset — the default — this is exactly the single
    /// `forward_prefill(prompt, 0, kernel)` call it replaces.
    pub(crate) fn prefill_for_generate(
        &mut self,
        prompt_tokens: &[u32],
    ) -> RuntimeResult<Option<Vec<f32>>> {
        let prefill_start = std::time::Instant::now();
        if self.is_cancelled() {
            tracing::debug!("cancelled before prefill");
            return Ok(None);
        }

        let chunk = self
            .prefill_chunk_tokens
            .filter(|&n| n > 0 && n < prompt_tokens.len());

        let logits = match chunk {
            None => self.model.forward_prefill(prompt_tokens, 0, &self.kernel)?,
            Some(chunk) => {
                let mut last = Vec::new();
                for (i, window) in prompt_tokens.chunks(chunk).enumerate() {
                    if self.is_cancelled() {
                        tracing::debug!(
                            chunk_index = i,
                            "cancelled during chunked prefill; abandoning the prompt"
                        );
                        return Ok(None);
                    }
                    last = self
                        .model
                        .forward_prefill(window, i * chunk, &self.kernel)?;
                }
                last
            }
        };

        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(prefill_start.elapsed().as_secs_f64());
        }
        Ok(Some(logits))
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
#[path = "engine_tests.rs"]
mod tests;
