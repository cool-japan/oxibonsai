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
//!   penalties, then top-k/temperature/min-p/top-p) at every step.
//! * The GPU-argmax fast path (`crate::engine_greedy`) is taken **only**
//!   when the configured sampler is pure greedy with no penalties. `generate`
//!   / `generate_tracked` / the streaming pair all decide this through the
//!   single predicate [`InferenceEngine::greedy_gpu_eligible`], which also
//!   requires
//!   [`GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`]
//!   (the GPU argmax kernels break ties toward the first index, like the
//!   CPU samplers — see that constant's doc comment for how that is
//!   verified and how to close the gate if a regression is found).
//!   [`InferenceEngine::generate_greedy_gpu`] is a separate, *direct* entry
//!   point that does not go through `greedy_gpu_eligible` — it checks the
//!   same gate and the same penalty condition itself. Either way, when
//!   penalties are configured, or the tie-break gate is closed, the call
//!   decodes the full logit row and applies penalties before the argmax
//!   instead.
//! * A sampled request on the fused Metal route decodes the full logit row
//!   through the engine's sampler by default. With the opt-in sampled top-k
//!   route ([`InferenceEngine::set_sampled_topk`]; off by default — see
//!   [`SampledTopKConfig`]) and no penalty it instead draws its decode steps
//!   from the GPU's top-k candidates. The engine's own sampler makes the
//!   draw over a candidate sub-row that contains every top-k survivor, in the
//!   sampler's canonical survivor order, so the tokens are exactly the ones
//!   the full row gives.
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
use oxibonsai_kernels::gpu_backend::{GpuUploadScope, UploadStats, UNATTRIBUTED_MODEL_EPOCH};
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::{KernelDispatcher, KernelTier};
use oxibonsai_model::hybrid::{HybridModel, LoadedModel};
use oxibonsai_model::model::BonsaiModel;

use crate::engine_control::{
    gguf_fused_metal_route, resolve_eos_token_set, CancellationToken, EosTokenSet, FusedMetalRoute,
    RecurrentState, SpeculativeConfig, GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX,
};
use crate::engine_greedy::SampledTopKConfig;
pub use crate::engine_seam::{
    engine_error_code, tokenizer_from_gguf, Backend, EngineError, SequenceSnapshot,
};
use crate::error::{RuntimeError, RuntimeResult};
use crate::metrics::InferenceMetrics;
use crate::request_metrics::RequestRateAggregator;
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
    /// Sampled decode steps served from a GPU top-k candidate download
    /// instead of the full logit row (`perf-11`, sampled half).
    pub sampled_topk_steps: AtomicU64,
    /// Sampled decode steps on the top-k route that had to download the full
    /// logit row instead: unusable GPU candidates (a non-finite value inside
    /// the sampler's `top_k`, or a GPU argmax/top-k disagreement), a Metal
    /// failure recovered on the CPU, or the full-row reference mode.
    pub sampled_topk_full_row_steps: AtomicU64,
    /// Sampled requests on the fused GPU route that were not eligible for the
    /// top-k route at all (penalties configured, `top_k` of `0`, above the
    /// candidate count or at/above the vocabulary, or the route disabled —
    /// see [`SampledTopKConfig`]) and decoded on the full row.
    pub sampled_full_row_requests: AtomicU64,
}

impl EngineStats {
    /// Create new engine stats, recording the current time as start.
    pub fn new() -> Self {
        Self {
            total_tokens_generated: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            active_sessions: AtomicUsize::new(0),
            start_time: Instant::now(),
            sampled_topk_steps: AtomicU64::new(0),
            sampled_topk_full_row_steps: AtomicU64::new(0),
            sampled_full_row_requests: AtomicU64::new(0),
        }
    }

    /// Sampled decode steps served from GPU top-k candidates (`perf-11`).
    pub fn sampled_topk_steps(&self) -> u64 {
        self.sampled_topk_steps.load(Ordering::Relaxed)
    }

    /// Sampled top-k-route steps that fell back to the full logit row.
    pub fn sampled_topk_full_row_steps(&self) -> u64 {
        self.sampled_topk_full_row_steps.load(Ordering::Relaxed)
    }

    /// Sampled fused-route requests ineligible for the top-k route.
    pub fn sampled_full_row_requests(&self) -> u64 {
        self.sampled_full_row_requests.load(Ordering::Relaxed)
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
///
/// Holds either kind of model through [`LoadedModel`]: a dense Qwen3-family
/// [`BonsaiModel`] or a `qwen35` (PrismML Bonsai 2) [`HybridModel`]. Every
/// generation path dispatches through the seam in [`crate::engine_seam`];
/// see that module for what a hybrid engine does and the typed errors it
/// returns for what it does not.
pub struct InferenceEngine<'a> {
    pub(crate) model: LoadedModel<'a>,
    pub(crate) kernel: KernelDispatcher,
    pub(crate) sampler: Sampler,
    pub(crate) metrics: Option<Arc<InferenceMetrics>>,
    pub(crate) stats: Arc<EngineStats>,
    /// Cumulative number of tokens that have been processed by
    /// [`InferenceEngine::prefill_from_pos`] and
    /// [`InferenceEngine::prefill_multimodal`] (every row of a multimodal
    /// prompt, images expanded) across the engine's lifetime.
    ///
    /// Used by the prefix-cache integration to verify that cached prefixes
    /// actually reduce prefill work — the cached portion of a prompt is not
    /// re-fed into prefill, so a repeated prompt should increment this
    /// counter by strictly fewer tokens than its full length.
    prefill_token_count: u64,
    /// Optional workload-level rate aggregator. When attached, every
    /// `generate_tracked` call records its
    /// [`RequestRateSnapshot`](crate::request_metrics::RequestRateSnapshot) here on
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
    pub(crate) recurrent: Option<Box<dyn RecurrentState>>,
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
    /// Model epoch this engine's GPU weight uploads were attributed to
    /// (`MET-M1`), released — and only it — when the engine drops.
    /// [`UNATTRIBUTED_MODEL_EPOCH`] for engines that uploaded nothing under
    /// their own scope (every synthetic-config / `from_model*` engine, every
    /// hybrid engine).
    pub(crate) model_epoch: u64,
    /// What this engine's construction uploaded to the GPU weight cache under
    /// [`Self::model_epoch`]: fresh buffers versus buffers shared with a
    /// sibling replica that already held byte-identical weights (`MET-M1` /
    /// replica sharing). Empty for engines that uploaded nothing.
    pub(crate) gpu_uploads: UploadStats,
    /// The backend this engine was asked for ([`Backend::Auto`] for every
    /// constructor that takes none).
    pub(crate) backend: Backend,
    /// Identity of the current sequence: bumped by every reset, explicit or
    /// implicit (a prefill/decode restart at position 0). A
    /// [`SequenceSnapshot`] is only restorable onto the sequence it was
    /// taken from.
    pub(crate) sequence_id: u64,
    /// The restores that rolled the current sequence back, so a restore can
    /// refuse a snapshot whose positions were rewritten since it was taken
    /// (see [`crate::engine_seam::RewindLog`]).
    pub(crate) rewinds: crate::engine_seam::RewindLog,
    /// Sampled decode on the fused GPU route (`perf-11`, sampled half):
    /// whether to download top-k candidates instead of the full logit row
    /// (off by default — see [`SampledTopKConfig`]) and how many.
    pub(crate) sampled_topk: SampledTopKConfig,
    /// The Metal hybrid runner a hybrid engine decodes on, when one serves
    /// its model (see [`crate::engine_hybrid_gpu`]); `None` for a dense
    /// engine and for a hybrid engine on the CPU tier.
    pub(crate) hybrid_gpu: Option<crate::engine_hybrid_gpu::HybridGpu<'a>>,
    /// Test-only scripted generation (see [`ScriptedLogits`]); `None` — the
    /// only value outside `cfg(test)`, where the field does not exist — runs
    /// the model's real logits.
    #[cfg(test)]
    pub(crate) scripted_logits: Option<ScriptedLogits>,
}

/// Logit value every non-scripted token gets in a [`ScriptedLogits`] row:
/// finite (a log-probability of it stays JSON-serialisable) yet far enough
/// below the scripted token's `0.0` that no temperature up to the API's
/// `2.0` leaves it any probability mass in `f32`.
#[cfg(test)]
const SCRIPTED_LOGIT_FLOOR: f32 = -1.0e9;

/// A test-only generation script: make the engine's sampler yield a fixed
/// id sequence, whatever its temperature, `top_k`, `top_p` or seed.
///
/// Every logit row a generation reads — the prefill's last row
/// ([`InferenceEngine::prefill_for_generate`], once per generation however
/// the prefill is chunked) and every decode forward
/// (`InferenceEngine::forward_logits`) — is replaced by a row whose only
/// non-floor entry is the next scripted id, so any sampler configuration
/// draws exactly that id. The script restarts at every generation (each
/// request replays the same sequence) and, once exhausted, scripts the
/// engine's primary EOS id, so generation ends on its own with a natural
/// stop. It can instead fail the next logit row with a typed
/// [`EngineError`], to drive a handler's error mapping end to end; leave
/// every row an equal choice among a fixed id set, so the sampler's own
/// PRNG — and therefore its seed — decides each token (a seeded-draw test);
/// or make every row one fixed, **known** row — a handful of ids with given
/// logits, every other id the floor — so a test knows the exact
/// distribution each draw is made from (the min-p tests).
///
/// A script can also **hold** its generation once it has drawn a given
/// number of ids — the next decode row waits (bounded by
/// [`SCRIPTED_HOLD_LIMIT`]) for the engine's cancellation token — so a test
/// of "a stop sequence matched on the receiving side really cancels the
/// generation" does not race a tiny model that would otherwise finish all
/// its tokens before the receiver has seen the first one. The hold is
/// one-shot: later generations on the engine run unheld.
///
/// Compiled only under `cfg(test)`: production builds have no such field
/// and no hook, so their behaviour is untouched.
#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub(crate) struct ScriptedLogits {
    ids: Vec<u32>,
    cursor: usize,
    fail_with: Option<EngineError>,
    /// When non-empty, every row gives exactly these ids an equal logit
    /// (and every other id the floor) instead of following `ids`.
    uniform_over: Vec<u32>,
    /// When non-empty, every row gives each `(id, logit)` pair its logit
    /// (and every other id the floor) instead of following `ids`.
    known_row: Vec<(u32, f32)>,
    /// Hold the generation once this many ids have been drawn (see above).
    hold_after: Option<usize>,
}

/// Longest a held script waits for its generation to be cancelled before
/// carrying on as if it had not been held (a regression where nothing
/// cancels then shows up as a test failure, not a hang).
#[cfg(test)]
const SCRIPTED_HOLD_LIMIT: std::time::Duration = std::time::Duration::from_secs(5);

/// The scripts' setters. Every caller is an HTTP-level test, so they exist
/// only where those tests do (`cfg(test)` with the `server` feature).
#[cfg(all(test, feature = "server"))]
impl InferenceEngine<'_> {
    /// Script every generation on this engine to emit exactly `ids` (then
    /// the primary EOS id). See [`ScriptedLogits`].
    pub(crate) fn script_generation(&mut self, ids: Vec<u32>) {
        self.scripted_logits = Some(ScriptedLogits {
            ids,
            ..ScriptedLogits::default()
        });
    }

    /// Make every generation on this engine fail with `error` as soon as it
    /// reads a logit row. See [`ScriptedLogits`].
    pub(crate) fn script_failure(&mut self, error: EngineError) {
        self.scripted_logits = Some(ScriptedLogits {
            fail_with: Some(error),
            ..ScriptedLogits::default()
        });
    }

    /// Make every logit row an equal choice among `ids` (never EOS, so a
    /// generation runs to its token limit): which of them each step draws
    /// is up to the sampler's PRNG alone. See [`ScriptedLogits`].
    pub(crate) fn script_uniform_choice(&mut self, ids: Vec<u32>) {
        self.scripted_logits = Some(ScriptedLogits {
            uniform_over: ids,
            ..ScriptedLogits::default()
        });
    }

    /// [`Self::script_generation`], holding the first generation after
    /// `hold_after` ids until it is cancelled. See [`ScriptedLogits`].
    pub(crate) fn script_generation_held(&mut self, ids: Vec<u32>, hold_after: usize) {
        self.scripted_logits = Some(ScriptedLogits {
            ids,
            hold_after: Some(hold_after),
            ..ScriptedLogits::default()
        });
    }
}

#[cfg(test)]
impl InferenceEngine<'_> {
    /// Make every logit row the fixed row `entries` describes: each
    /// `(id, logit)` pair gets its logit, every other id the floor, so every
    /// draw of every generation is made from one known distribution. The
    /// engine-level min-p tests use it (in every feature configuration, so
    /// it is not `server`-gated like the setters above). See
    /// [`ScriptedLogits`].
    pub(crate) fn script_known_row(&mut self, entries: Vec<(u32, f32)>) {
        self.scripted_logits = Some(ScriptedLogits {
            known_row: entries,
            ..ScriptedLogits::default()
        });
    }

    /// Apply the script (if any) to one freshly computed logit row;
    /// `restart` rewinds the script to its first id (a new generation's
    /// prefill row).
    pub(crate) fn scripted_row(&mut self, row: Vec<f32>, restart: bool) -> RuntimeResult<Vec<f32>> {
        let eos = self.eos.primary();
        let Some(script) = self.scripted_logits.as_mut() else {
            return Ok(row);
        };
        if let Some(error) = script.fail_with.clone() {
            return Err(error.into());
        }
        let mut scripted = vec![SCRIPTED_LOGIT_FLOOR; row.len()];
        if !script.known_row.is_empty() {
            for &(id, logit) in &script.known_row {
                if let Some(slot) = scripted.get_mut(id as usize) {
                    *slot = logit;
                }
            }
            return Ok(scripted);
        }
        if !script.uniform_over.is_empty() {
            for &id in &script.uniform_over {
                if let Some(slot) = scripted.get_mut(id as usize) {
                    *slot = 0.0;
                }
            }
            return Ok(scripted);
        }
        if restart {
            script.cursor = 0;
        }
        // A decode row requested once `hold_after` ids were drawn: the last
        // of them has already been handed to the receiver, which is what
        // the hold waits on (one-shot).
        let hold = !restart
            && script
                .hold_after
                .is_some_and(|after| script.cursor >= after);
        if hold {
            script.hold_after = None;
        }
        let id = script.ids.get(script.cursor).copied().unwrap_or(eos);
        script.cursor = script.cursor.saturating_add(1);
        if let Some(slot) = scripted.get_mut(id as usize) {
            *slot = 0.0;
        }
        if hold {
            let start = std::time::Instant::now();
            while !self.is_cancelled() && start.elapsed() < SCRIPTED_HOLD_LIMIT {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        Ok(scripted)
    }
}

/// `MET-M1`, eviction half: an engine releases exactly its own model epoch
/// from the GPU weight cache when it drops — never the process-wide clear,
/// because a pool's replicas share one resident copy by reference and
/// dropping one replica must not pull weights out from under its siblings
/// (the backend frees a buffer only when its last epoch lets go).
impl Drop for InferenceEngine<'_> {
    fn drop(&mut self) {
        if self.model_epoch == UNATTRIBUTED_MODEL_EPOCH {
            return;
        }
        match oxibonsai_kernels::gpu_backend::release_model_weights(&self.kernel, self.model_epoch)
        {
            Ok(freed) => tracing::debug!(
                model_epoch = self.model_epoch,
                freed_buffers = freed,
                "engine dropped: released its GPU weight registrations"
            ),
            Err(e) => tracing::warn!(
                model_epoch = self.model_epoch,
                error = %e,
                "engine dropped: releasing its GPU weight registrations failed"
            ),
        }
    }
}

impl<'a> InferenceEngine<'a> {
    /// Create a new inference engine from a configuration (no weights — for testing).
    pub fn new(config: Qwen3Config, sampling_params: SamplingParams, seed: u64) -> Self {
        let model = BonsaiModel::new(config);
        let kernel = KernelDispatcher::auto_detect();
        let sampler = Sampler::new(sampling_params, seed);

        tracing::info!(kernel = kernel.name(), "inference engine initialized");

        Self::assemble(
            LoadedModel::Dense(Box::new(model)),
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
    /// public constructors.
    fn assemble(
        model: LoadedModel<'a>,
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
            model_epoch: UNATTRIBUTED_MODEL_EPOCH,
            gpu_uploads: UploadStats::default(),
            backend: Backend::Auto,
            sequence_id: 0,
            rewinds: crate::engine_seam::RewindLog::default(),
            sampled_topk: SampledTopKConfig::default(),
            hybrid_gpu: None,
            #[cfg(test)]
            scripted_logits: None,
        }
    }

    /// Wrap an already-loaded model of either kind with a caller-supplied
    /// dispatcher.
    ///
    /// For a [`LoadedModel::Hybrid`] the dispatcher is used for everything
    /// the engine itself dispatches (the hybrid model carries its own inside
    /// its layers); pass a CPU tier — the engine decodes the hybrid on its
    /// CPU layers (only the GGUF constructors build the Metal hybrid
    /// runner), and a GPU tier would make it report a GPU it never uses.
    pub fn from_loaded_model(
        model: LoadedModel<'a>,
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

    /// Wrap an already-constructed [`HybridModel`] (on the best CPU tier;
    /// the Metal hybrid runner needs the GGUF image the model was bound
    /// from, so only the GGUF constructors build one).
    ///
    /// The EOS set falls back to [`EOS_TOKEN_ID`] exactly as the dense
    /// `from_model*` constructors do; a GGUF-loaded hybrid engine resolves it
    /// from the file ([`InferenceEngine::from_gguf`]).
    pub fn from_hybrid_model(
        model: HybridModel<'a>,
        sampling_params: SamplingParams,
        seed: u64,
    ) -> Self {
        Self::from_loaded_model(
            LoadedModel::Hybrid(Box::new(model)),
            crate::engine_seam::cpu_dispatcher(),
            sampling_params,
            seed,
        )
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
            LoadedModel::Dense(Box::new(model)),
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

    /// Create a new inference engine from a loaded GGUF file, on
    /// [`Backend::Auto`].
    ///
    /// Either kind of model: a `qwen35` (PrismML Bonsai 2) file loads as a
    /// hybrid engine — on the Metal hybrid runner when one serves it on this
    /// host, else on the best CPU tier (see [`crate::engine_hybrid_gpu`]) —
    /// anything else as a dense one exactly as before.
    pub fn from_gguf(
        gguf: &'a GgufFile<'a>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Self> {
        Self::from_gguf_with_backend(gguf, sampling_params, seed, max_seq_len, Backend::Auto)
    }

    /// [`from_gguf`](Self::from_gguf) on an explicit [`Backend`].
    ///
    /// # Errors
    ///
    /// Model-load errors; [`EngineError::HybridGpuBackendUnsupported`] for
    /// [`Backend::Metal`] on a hybrid file the Metal hybrid runner cannot
    /// serve here (naming why); [`EngineError::BackendUnavailable`] for
    /// [`Backend::Metal`] on a dense file on a build or host without an
    /// accelerated Metal device.
    pub fn from_gguf_with_backend(
        gguf: &'a GgufFile<'a>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
        backend: Backend,
    ) -> RuntimeResult<Self> {
        Self::from_gguf_with_embd_and_backend(
            gguf,
            sampling_params,
            seed,
            max_seq_len,
            std::sync::Arc::from(Vec::new()),
            backend,
        )
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
    /// GGUF; see [`BonsaiModel::from_gguf_with_embd`] for the contract. A
    /// hybrid model decodes its embedding row-wise from the file, so for one
    /// `token_embd` must be empty (which is exactly what
    /// [`model_token_embd`](Self::model_token_embd) hands out for it).
    pub fn from_gguf_with_embd(
        gguf: &'a GgufFile<'a>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
        token_embd: std::sync::Arc<[f32]>,
    ) -> RuntimeResult<Self> {
        Self::from_gguf_with_embd_and_backend(
            gguf,
            sampling_params,
            seed,
            max_seq_len,
            token_embd,
            Backend::Auto,
        )
    }

    /// The one GGUF constructor every other one funnels into.
    ///
    /// # Errors
    ///
    /// As [`from_gguf_with_backend`](Self::from_gguf_with_backend), plus
    /// [`EngineError::SharedEmbeddingUnsupported`] for a non-empty
    /// `token_embd` on a hybrid file.
    pub fn from_gguf_with_embd_and_backend(
        gguf: &'a GgufFile<'a>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
        token_embd: std::sync::Arc<[f32]>,
        backend: Backend,
    ) -> RuntimeResult<Self> {
        let eos = resolve_eos_token_set(gguf, EOS_TOKEN_ID);

        // A `qwen35` (PrismML Bonsai 2) file is structurally a different
        // stack -- 48 of its 64 layers are Gated-DeltaNet recurrences with no
        // `attn_q.weight` at all -- so it loads through the hybrid driver.
        // `gguf_fused_metal_route` is deliberately NOT consulted for it: an
        // all-PQ2_0 file classifies as a "ternary fused route", and the fused
        // Metal graph cannot run a hybrid stack.
        if LoadedModel::is_hybrid_gguf(gguf) {
            if !token_embd.is_empty() {
                return Err(EngineError::SharedEmbeddingUnsupported {
                    architecture: crate::engine_seam::gguf_architecture(gguf),
                    len: token_embd.len(),
                }
                .into());
            }
            let loaded = crate::engine_seam::load_hybrid(gguf, max_seq_len, backend)?;
            let sampler = Sampler::new(sampling_params, seed);
            let mut engine = Self::assemble(
                LoadedModel::Hybrid(Box::new(loaded.model)),
                loaded.kernel,
                sampler,
                eos,
                false,
            );
            engine.backend = backend;
            engine.hybrid_gpu = loaded.gpu;
            // cli-16 + K-14: the effective executor, and how the INT8 tier
            // selector applies to it.
            tracing::info!(
                kernel = %engine.kernel_label(),
                tier_reason = %engine.effective_tier_reason(),
                eos = ?engine.eos.as_slice(),
                model = %engine.model_description(),
                "inference engine loaded from GGUF (hybrid)"
            );
            return Ok(engine);
        }

        let route = gguf_fused_metal_route(gguf);
        let kernel = crate::engine_seam::dense_dispatcher(backend)?;
        let model = {
            // `Backend::Cpu`: `BonsaiModel::from_gguf*` builds its layers'
            // dispatchers with `KernelDispatcher::auto_detect()` internally,
            // so pinning only the engine's dispatcher would leave every GEMV
            // on the GPU. The scope makes those internal auto-detects land on
            // the CPU tier too.
            let _cpu_only = (backend == Backend::Cpu)
                .then(oxibonsai_kernels::gpu_backend::CpuOnlyBackendScope::enter);
            BonsaiModel::from_gguf_with_embd(gguf, max_seq_len, token_embd)?
        };
        let mut engine =
            Self::from_model_with_gpu_warmup(model, kernel, sampling_params, seed, eos, route)?;
        engine.backend = backend;
        Ok(engine)
    }

    /// Shared core of the dense GGUF constructors: given an
    /// already-constructed [`BonsaiModel`] and the engine's dispatcher,
    /// upload weights to the GPU (attributed to a fresh model epoch), run the
    /// per-tier warmups, and assemble the engine.
    ///
    /// Factored out so the (substantial) GPU/CUDA warmup logic has exactly one
    /// implementation regardless of how `token_embd` was obtained.
    fn from_model_with_gpu_warmup(
        mut model: BonsaiModel<'a>,
        kernel: KernelDispatcher,
        sampling_params: SamplingParams,
        seed: u64,
        eos: EosTokenSet,
        route: FusedMetalRoute,
    ) -> RuntimeResult<Self> {
        // `MET-M1`: every GPU weight upload this engine makes is attributed to
        // its own epoch, so `Drop` can release exactly this engine's
        // registrations; a deduplicating backend (`Scirs2Backend`) shares a
        // byte-identical resident buffer with a sibling replica instead of
        // uploading a second copy.
        let model_epoch = oxibonsai_kernels::gpu_backend::next_gpu_model_epoch();
        let upload_scope = GpuUploadScope::enter(model_epoch);

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
        } else if kernel_is_gpu_tier(&kernel) {
            // Upload all model weights to GPU memory once. On a CPU tier
            // (including an explicit `Backend::Cpu`) there is nothing to
            // upload and the call is skipped outright.
            model.upload_weights_to_gpu(&kernel);
        }

        // Pre-build GPU weight cache eagerly so it's outside the timing window.
        // Only on a GPU tier: a CPU engine never reads it, and building it
        // would open the Metal device for nothing.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if kernel_is_gpu_tier(&kernel) {
            tracing::info!("pre-building GPU weight cache");
            model.get_or_create_gpu_cache().map_err(|e| {
                // `MET-M1`: this load already registered its uploads under
                // `model_epoch`, and no engine will exist to release them on
                // drop -- release them here, or a failed load pins its
                // weights on the device for the life of the process.
                if let Err(release) =
                    oxibonsai_kernels::gpu_backend::release_model_weights(&kernel, model_epoch)
                {
                    tracing::warn!(
                        error = %release,
                        model_epoch,
                        "failed to release a failed load's GPU weight registrations"
                    );
                }
                RuntimeError::Model(oxibonsai_model::error::ModelError::Internal(format!(
                    "GPU weight cache init: {e}"
                )))
            })?;
        }

        // Say what was uploaded and what was shared -- a replica that shares
        // a sibling's resident weights must not look like one that uploaded
        // nothing.
        let uploads = upload_scope.finish();
        if !uploads.is_empty() {
            tracing::info!(
                model_epoch,
                fresh_buffers = uploads.fresh_buffers,
                fresh_mib = uploads.fresh_bytes as f64 / (1024.0 * 1024.0),
                shared_buffers = uploads.shared_buffers,
                shared_mib = uploads.shared_bytes as f64 / (1024.0 * 1024.0),
                "GPU weight upload: fresh buffers were placed on the device, shared buffers \
                 reused a byte-identical copy another replica already holds"
            );
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

        // cli-16: name the resolved dominant tensor type together with the
        // effective tier (e.g. "TQ2_0_g128 GPU (accelerated)"), never a
        // hardcoded kernel family; K-14: say how the INT8 tier selector
        // applies to this dispatcher and format.
        let int8_tier = crate::engine_control::int8_tier_use_from_env(
            model.dominant_quant_type(),
            kernel.tier(),
            crate::engine_control::TierExecutor::Dense,
        );
        tracing::info!(
            kernel = %kernel.kernel_label(model.dominant_quant_type()),
            tier_reason = %kernel.effective_tier_reason(),
            int8_tier = %int8_tier,
            eos = ?eos.as_slice(),
            fused_route = ?route,
            "inference engine loaded from GGUF"
        );

        let mut engine = Self::assemble(
            LoadedModel::Dense(Box::new(model)),
            kernel,
            sampler,
            eos,
            route.is_fused() && cfg!(all(feature = "metal", target_os = "macos")),
        );
        engine.model_epoch = model_epoch;
        engine.gpu_uploads = uploads;
        Ok(engine)
    }

    /// Attach shared metrics to this engine for recording inference telemetry.
    pub fn set_metrics(&mut self, metrics: Arc<InferenceMetrics>) {
        self.metrics = Some(metrics);
    }

    /// Attach a workload-level [`RequestRateAggregator`] to this engine.
    ///
    /// Once attached, every call to [`InferenceEngine::generate_tracked`] (or
    /// [`InferenceEngine::generate_with_request_id`]) will push its
    /// per-request
    /// [`RequestRateSnapshot`](crate::request_metrics::RequestRateSnapshot)
    /// into the aggregator on completion.
    /// The aggregator is reference-counted, so the same instance can be shared
    /// with the Prometheus metrics layer or the admin endpoints.
    pub fn set_rate_aggregator(&mut self, aggregator: Arc<RequestRateAggregator>) {
        self.rate_aggregator = Some(aggregator);
    }

    /// Read-only access to the attached rate aggregator, if any.
    pub fn rate_aggregator(&self) -> Option<&Arc<RequestRateAggregator>> {
        self.rate_aggregator.as_ref()
    }

    /// Cheaply clone a handle to this engine's shared token-embedding table.
    ///
    /// Thin delegate to [`BonsaiModel::shared_token_embd`]. The engine pool
    /// calls this on replica `#1` to extract the one shared `Arc<[f32]>`, which
    /// it then hands to [`InferenceEngine::from_gguf_static_with_embd`] when
    /// building replicas `2..N` — so every replica's `token_embd` is a clone of
    /// the same allocation (one ~1.16 GiB table for the 1.7B, not N).
    ///
    /// A hybrid model has no dense table to share (its embedding is decoded
    /// row-wise straight out of the memory map, which every replica already
    /// shares), so it hands out the empty "load it from the GGUF" handle.
    pub fn model_token_embd(&self) -> std::sync::Arc<[f32]> {
        match &self.model {
            LoadedModel::Dense(model) => model.shared_token_embd(),
            LoadedModel::Hybrid(_) => std::sync::Arc::from(Vec::new()),
        }
    }

    /// Get a reference to the kernel dispatcher.
    pub fn kernel(&self) -> &KernelDispatcher {
        &self.kernel
    }

    /// Kernel tier this engine's decode runs on.
    ///
    /// Feature-agnostic convenience used by the engine pool to size itself:
    /// a GPU tier is bounded by `MetalGraph::max_sessions()`, a CPU tier runs
    /// replicas fully in parallel. A Metal-backed hybrid engine reports
    /// `KernelTier::Gpu` — its decode runs on the Metal hybrid runner — even
    /// though [`kernel`](Self::kernel) is the CPU dispatcher its CPU model
    /// (the embedding pass) keeps.
    pub fn kernel_tier(&self) -> KernelTier {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if self.hybrid_gpu.is_some() {
            return KernelTier::Gpu;
        }
        self.kernel.tier()
    }

    /// Run prefill at a given KV-cache offset.
    ///
    /// Unlike [`InferenceEngine::generate`], this does **not** reset the
    /// model's KV cache before execution: callers (e.g. the prefix-cache
    /// engine) are expected to have prepared the cache state explicitly.
    /// A hybrid model can only continue from its next position (or restart at
    /// `0`); any other `pos_start` is a typed error — see
    /// [`crate::engine_seam`].
    ///
    /// Increments the [`prefill_token_count`](Self::prefill_token_count)
    /// counter by `prompt_tokens.len()` on success.
    pub fn prefill_from_pos(
        &mut self,
        prompt_tokens: &[u32],
        pos_start: usize,
    ) -> RuntimeResult<Vec<f32>> {
        let logits = self.prefill_logits(prompt_tokens, pos_start)?;
        self.record_prefill_tokens(prompt_tokens.len());
        Ok(logits)
    }

    /// Forward one token at the given absolute position.
    pub fn decode_step(&mut self, token: u32, pos: usize) -> RuntimeResult<Vec<f32>> {
        self.forward_logits(token, pos)
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
    ///
    /// # Errors
    ///
    /// [`EngineError::RecurrentRollbackRequired`] on a hybrid engine:
    /// verification writes state for every draft position, and the rejected
    /// ones cannot be rolled back out of a recurrence.
    pub fn verify_batch(&mut self, tokens: &[u32], pos_start: usize) -> RuntimeResult<Vec<u32>> {
        match &mut self.model {
            LoadedModel::Dense(model) => {
                Ok(model.forward_prefill_verify(tokens, pos_start, &self.kernel)?)
            }
            LoadedModel::Hybrid(model) => Err(EngineError::RecurrentRollbackRequired {
                operation: "verify_batch (speculative verification)",
                architecture: model.config().base.architecture.clone(),
            }
            .into()),
        }
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
    /// Rolling back to the **current** position is the identity and always
    /// succeeds, recurrent state or not.
    ///
    /// # Errors
    ///
    /// [`ModelError::RecurrentRollbackUnsupported`](oxibonsai_model::error::ModelError::RecurrentRollbackUnsupported)
    /// when the sequence state includes a recurrence — a hybrid (`qwen35`)
    /// model, or an attached `RecurrentState` — and `committed_len` is not
    /// the current position. `tokens` reports how many tokens the state has
    /// consumed. Use [`snapshot_sequence`](Self::snapshot_sequence) /
    /// [`restore_sequence`](Self::restore_sequence) for an exact rollback
    /// point.
    pub fn try_rewind_cache(&mut self, committed_len: usize) -> RuntimeResult<()> {
        let consumed = self.sequence_position();
        if committed_len == consumed {
            return Ok(());
        }
        if !self.recurrent_rollback_supported() {
            return Err(RuntimeError::Model(
                oxibonsai_model::error::ModelError::RecurrentRollbackUnsupported {
                    pos: committed_len,
                    tokens: consumed,
                },
            ));
        }
        if let LoadedModel::Dense(model) = &mut self.model {
            model.kv_cache_mut().truncate(committed_len);
        }
        Ok(())
    }

    /// `true` when a KV-cursor rollback is exact for this engine, i.e. when
    /// [`try_rewind_cache`](Self::try_rewind_cache) can move backwards:
    /// `false` for a hybrid model and for an engine with an attached
    /// `RecurrentState`.
    ///
    /// Speculative decoding and the pipeline check this before starting a
    /// draft window rather than discovering it at rollback time.
    #[must_use]
    pub fn recurrent_rollback_supported(&self) -> bool {
        self.recurrent.is_none() && !self.model.is_hybrid()
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
    /// See [`RecurrentState`] for why the hybrid model needs this and what an
    /// attached state must implement.
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
    /// be shortened to the borrow. The hybrid model's `RecurrentCache` owns
    /// its buffers, so this costs it nothing.
    pub fn recurrent_state_mut(&mut self) -> Option<&mut (dyn RecurrentState + 'static)> {
        self.recurrent.as_deref_mut()
    }

    /// Detach the recurrent state, returning it to the caller.
    pub fn take_recurrent_state(&mut self) -> Option<Box<dyn RecurrentState>> {
        self.recurrent.take()
    }

    /// Bytes of recurrent state held by this engine: the hybrid model's own
    /// Gated-DeltaNet state (~157 MB for the 27B), the Metal runner's device
    /// copy (~150 MiB) on a Metal-backed engine, plus any attached
    /// `RecurrentState`; `0` for a dense engine with nothing attached.
    pub fn recurrent_memory_bytes(&self) -> usize {
        let attached = self
            .recurrent
            .as_ref()
            .map_or(0, |s| s.recurrent_memory_bytes());
        let own = self
            .model
            .as_hybrid()
            .map_or(0, |model| model.recurrent().memory_bytes());
        let runner = self.hybrid_gpu.as_ref().map_or(0, |gpu| {
            usize::try_from(gpu.recurrent_bytes()).unwrap_or(usize::MAX)
        });
        attached.saturating_add(own).saturating_add(runner)
    }

    /// Clear the recurrent/conv state (`RT-28`): the hybrid model's own, the
    /// Metal runner's on a Metal-backed engine, and any attached
    /// `RecurrentState`.
    ///
    /// A no-op for a dense engine with nothing attached. Called by
    /// [`reset`](Self::reset), so the server's per-request reset (`RT-03`)
    /// already covers it — unlike a KV cache, recurrent state is not masked
    /// by position, so a stale `S` matrix would silently contaminate the
    /// next request rather than being overwritten.
    pub fn reset_recurrent(&mut self) {
        if let Some(state) = self.recurrent.as_deref_mut() {
            state.reset_recurrent();
        }
        if let Some(model) = self.model.as_hybrid_mut() {
            model.recurrent_mut().reset();
        }
        if let Some(gpu) = self.hybrid_gpu.as_mut() {
            gpu.reset();
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
    ///
    /// A hybrid (`qwen35`) engine names its executor instead of the pinned
    /// CPU dispatcher's own "explicitly requested", which would misdescribe
    /// a [`Backend::Auto`] load: the Metal hybrid runner with its KV window
    /// and device ceiling, or the CPU tier.
    ///
    /// Every reason ends with how the `OXIBONSAI_KERNEL_TIER` INT8 selector
    /// applies to this engine (K-14, [`Int8TierUse`](crate::engine_control::Int8TierUse)):
    /// honoured on a CPU tier for the native formats, ignored by a GPU-tier
    /// decode of them, honoured on any tier for the PrismML formats' CPU
    /// GEMV (only the CPU model's, never the Metal runner's), or unset.
    pub fn effective_tier_reason(&self) -> String {
        use crate::engine_control::{int8_tier_use_from_env, TierExecutor};
        let format = self.dominant_quant_type();
        let (reason, executor) = match (self.model.is_hybrid(), &self.hybrid_gpu) {
            (false, _) => (self.kernel.effective_tier_reason(), TierExecutor::Dense),
            (true, Some(gpu)) => {
                let window = gpu.window();
                (
                    format!(
                        "{} (hybrid `{}` model, backend={}: decodes on the Metal hybrid runner; KV \
                         window {} positions, device ceiling {}; the CPU model stays loaded on the \
                         {} tier for the embedding pass)",
                        crate::engine_hybrid_gpu::HYBRID_RUNNER_LABEL,
                        self.architecture(),
                        self.backend,
                        window.window,
                        window.device_ceiling,
                        self.kernel.tier(),
                    ),
                    TierExecutor::HybridMetal,
                )
            }
            (true, None) => (
                format!(
                    "{} tier (hybrid `{}` model, backend={}: runs on the CPU model, on the best \
                     CPU tier)",
                    self.kernel.tier(),
                    self.architecture(),
                    self.backend
                ),
                TierExecutor::HybridCpu,
            ),
        };
        let int8 = int8_tier_use_from_env(format, self.kernel.tier(), executor);
        format!("{reason}; {int8}")
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

    /// Current min-p (probabilistic nucleus) threshold of this engine's
    /// sampler (`RT-23`); `0.0` means disabled.
    ///
    /// A pure forwarder to [`Sampler::min_p`] on the engine's own sampler —
    /// the one every generation entry point decodes with. See
    /// [`set_min_p`](Self::set_min_p) for what the value does and how long it
    /// lasts.
    pub fn min_p(&self) -> f32 {
        self.sampler.min_p()
    }

    /// Set the min-p (probabilistic nucleus) threshold applied to sampled
    /// generation (`RT-23`): after top-k, every candidate whose probability
    /// is below `min_p` times the most likely candidate's is dropped and the
    /// survivors are renormalised, before top-p.
    ///
    /// A pure forwarder to [`Sampler::set_min_p`], so this setter and a
    /// server's per-request override — which sets the value on the engine's
    /// sampler for one request and restores the previous one afterwards —
    /// always agree on what a given `min_p` does.
    ///
    /// * **Lifetime.** Sampler configuration, like the penalties of
    ///   [`set_penalties`](Self::set_penalties): it applies to every request
    ///   on this engine until changed. [`reset`](Self::reset) keeps it
    ///   (a reset clears sequence state — the KV cache and any recurrent
    ///   state — not configuration), and so does every generation entry
    ///   point: [`generate_with_params`](Self::generate_with_params) and
    ///   [`generate_with_params_and_penalties`](Self::generate_with_params_and_penalties)
    ///   swap only the [`SamplingParams`] and penalties, and
    ///   [`generate_with_seed`](Self::generate_with_seed) carries it into its
    ///   per-call sampler.
    /// * **Range.** The value is stored as given and clamped to `[0.0, 1.0]`
    ///   when a draw applies it: `0.0` or below (or `NaN`) disables the
    ///   filter, and above `1.0` behaves as `1.0` — only the candidates tied
    ///   with the most likely one survive, never none.
    /// * **Greedy.** A temperature-0 request is an argmax and ignores it.
    /// * **Fused GPU route.** The sampled top-k route draws with this same
    ///   sampler over candidates that contain every top-k survivor, so min-p
    ///   applies identically with the route on or off.
    pub fn set_min_p(&mut self, min_p: f32) {
        self.sampler.set_min_p(min_p);
    }

    /// Cumulative number of tokens that have been processed by
    /// [`InferenceEngine::prefill_from_pos`] and
    /// [`InferenceEngine::prefill_multimodal`] over this engine's lifetime.
    pub fn prefill_token_count(&self) -> u64 {
        self.prefill_token_count
    }

    /// Count `tokens` prefilled positions into
    /// [`prefill_token_count`](Self::prefill_token_count) — the one place
    /// every successful prefill (text or multimodal) records its length.
    pub(crate) fn record_prefill_tokens(&mut self, tokens: usize) {
        self.prefill_token_count = self.prefill_token_count.saturating_add(tokens as u64);
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
    /// [`CancellationToken`]:
    /// `run_blocking_generation` resets the engine *before* running the
    /// caller's closure, so detaching here would silently disarm a token the
    /// request had already armed. A token cannot leak into the *next*
    /// request either, because
    /// [`EngineLease`](crate::engine_pool::EngineLease) clears it when the
    /// replica returns to the pool; a caller owning an engine directly uses
    /// [`clear_cancellation_token`](Self::clear_cancellation_token).
    pub fn reset(&mut self) {
        // `LoadedModel::reset` clears the KV cursor on both arms and the
        // hybrid model's own recurrent state; the Metal runner's state (on a
        // Metal-backed hybrid engine) and an attached `RecurrentState` are
        // left for this engine to clear.
        self.model.reset();
        if let Some(gpu) = self.hybrid_gpu.as_mut() {
            gpu.reset();
        }
        if let Some(state) = self.recurrent.as_deref_mut() {
            state.reset_recurrent();
        }
        self.sequence_id = self.sequence_id.wrapping_add(1);
    }

    /// Create a fresh [`CancellationToken`],
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
    /// * [`GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`] is `true` — an explicit
    ///   dependency gate: the MSL/CUDA argmax kernels tie-break toward the
    ///   global first index (`perf-11`), so routing here keeps the
    ///   temperature-0/no-penalty cross-tier determinism invariant on any
    ///   model whose logits contain exact duplicates (routinely true of a
    ///   ternary/1-bit quantized model — the only family that can reach this
    ///   predicate's `true` branch at all). If a future kernel regression
    ///   forces this constant back to `false`, this condition alone closes
    ///   the route again. See that constant's doc comment for the verified
    ///   example and the kernel-level test;
    /// * the model decodes through the fused Metal graph on a GPU tier —
    ///   `forward_greedy_gpu` maintains only the *GPU-resident* KV cache, so
    ///   a non-fused model would decode against an all-zero cache;
    /// * the caller does not need the full logit row (`needs_full_logits`):
    ///   log-probabilities cannot be reconstructed from a 4-byte argmax
    ///   readback;
    /// * the configured sampler is exactly greedy: temperature below
    ///   `GREEDY_TEMPERATURE_EPS` (`1e-6`), repetition penalty `1.0`, and no
    ///   frequency/presence penalty. Anything else has to see the logits.
    ///
    /// `SamplingParams::default()` carries `repetition_penalty: 1.0` (no
    /// penalty, `RT-24`), but a caller can configure one on the engine's
    /// sampler at any time, which is why this predicate reads the
    /// *sampler*, not the request.
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
            None => self.prefill_logits(prompt_tokens, 0)?,
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
                    last = self.prefill_logits(window, i * chunk)?;
                }
                last
            }
        };

        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(prefill_start.elapsed().as_secs_f64());
        }
        // Test-only scripted generation: once per generation, whatever the
        // prefill chunking (see `ScriptedLogits`).
        #[cfg(test)]
        let logits = self.scripted_row(logits, true)?;
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
}

// The generation entry points (`batch_generate`, `generate`,
// `generate_tracked`, `generate_with_request_id`, `generate_with_seed`,
// `generate_with_params`, `generate_with_params_and_penalties`) live in a
// child module, so they keep access to this module's private fields while
// this file stays under the 2000-line ceiling.
#[path = "engine_generate.rs"]
mod generate;

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
