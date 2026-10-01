//! The `InferenceEngine` ↔ [`LoadedModel`] seam: one engine type for both the
//! dense (`qwen3`) and the hybrid (`qwen35`, Bonsai 2) model kinds.
//!
//! [`InferenceEngine`] used to hold a concrete [`BonsaiModel`], so a `qwen35`
//! (PrismML Bonsai 2) file — 48 of whose 64 layers are Gated-DeltaNet
//! recurrences with no `attn_q.weight` at all — could not be run through the
//! engine, the CLI or the server. The engine now holds a [`LoadedModel`] and
//! every path dispatches through the helpers in this module:
//!
//! * **forward / prefill** — `InferenceEngine::forward_logits` and
//!   `InferenceEngine::prefill_logits` (crate-internal) call the *exact* dense
//!   `BonsaiModel::forward` / `forward_prefill` the engine always called (so
//!   a dense engine is byte-identical to before); for a hybrid one, the
//!   Metal hybrid runner's `forward_into` / chunked `forward_prefill` when
//!   the engine was built with one ([`InferenceEngine::hybrid_backend`] is
//!   `Metal`), else the CPU model's `forward` / chunked `forward_prefill`;
//! * **introspection** — vocabulary, context, architecture, dominant quant
//!   type, cache geometry, memory, for both kinds;
//! * **dense-only access** — [`InferenceEngine::dense_model`] /
//!   [`InferenceEngine::dense_model_mut`] return `None` on a hybrid engine,
//!   and [`InferenceEngine::require_dense`] turns that into the typed
//!   [`EngineError::NotADenseModel`];
//! * **sequence state** — [`InferenceEngine::snapshot_sequence`] /
//!   [`InferenceEngine::restore_sequence`], the one exact rollback point a
//!   recurrent model supports, on either hybrid executor.
//!
//! # Which executor a hybrid engine decodes on
//!
//! `--backend auto` ([`Backend::Auto`]) builds the Metal hybrid runner when
//! this is a Metal build, a Metal device exists, the model's geometry and
//! weight formats are ones the runner serves and a KV window fits
//! ([`crate::engine_hybrid_gpu::hybrid_backend_plan`]); otherwise it runs
//! the CPU tier and logs why at `info`. [`Backend::Metal`] builds the runner
//! or refuses with [`EngineError::HybridGpuBackendUnsupported`] naming the
//! violated constraint. [`Backend::Cpu`] never touches Metal. The CPU model
//! stays loaded beside a runner (the runner reads its weight slices in place
//! from the same mapping; the embedding pass runs on the CPU model), so the
//! KV window is budgeted over both residents
//! ([`crate::engine_hybrid_gpu::plan_hybrid_metal_window`]).
//!
//! # What a hybrid engine does not do (typed refusals, never silent)
//!
//! | Operation | Dense | Hybrid |
//! |---|---|---|
//! | `generate*` / streaming / logprobs / cancellation / reset | yes | yes (Metal runner or CPU) |
//! | GPU-argmax greedy, fused-GPU sampled top-k | Metal fused route | no — full-row decode on the hybrid executor (the runner downloads the logit row) |
//! | `--backend metal` ([`Backend::Metal`]) | yes | yes (the Metal hybrid runner), or [`EngineError::HybridGpuBackendUnsupported`] naming why not |
//! | `rewind_cache` to an earlier position | KV cursor move | `ModelError::RecurrentRollbackUnsupported` |
//! | `verify_batch` / speculative decoding | yes | [`EngineError::RecurrentRollbackRequired`] |
//! | prefix-cache KV block restore | yes | [`EngineError::RecurrentRollbackRequired`] (`PrefixCachedEngine::try_new`) |
//! | embeddings (`embed`, `ModelEmbedder`) | yes (batched CPU prefill) | yes (CPU, `HybridModel::forward_hidden`, on either executor) |
//!
//! [`EngineError::NotADenseModel`] remains the refusal for what genuinely is
//! dense-only: [`InferenceEngine::require_dense`] hands it to any caller that
//! needs the dense block stack itself.
//!
//! # Snapshot semantics
//!
//! A [`SequenceSnapshot`] captures the sequence **at the current position**:
//! the KV cursor (every stored key/value below it is immutable until a
//! reset) plus, for a hybrid model, a deep copy of the whole recurrent state
//! — the CPU model's (`RecurrentCache::snapshot`, ~157 MB for the 27B) or
//! the Metal runner's device state (`HybridMetalRunner::snapshot_state`,
//! ~150 MiB; the device KV needs no copy, as the runner writes every
//! position before any query reads it). Restoring it is exact, and both
//! executors refuse the same misuse with the same error codes.
//! What is **not** snapshot-able: an arbitrary *past* position of a hybrid
//! model (a recurrence has no positional masking — that needs a checkpoint
//! ring, design §8.4 item 2), a KV-only block of a hybrid sequence (the
//! prefix cache), and an opaque [`RecurrentState`](crate::engine_control::RecurrentState)
//! attached through `set_recurrent_state` (the trait has no snapshot hook).
//! A snapshot is bound to the sequence it was taken from: any reset — explicit,
//! or the implicit restart of a prefill/decode at position 0 — invalidates it,
//! and so does an embedding pass (`InferenceEngine::embed`), which overwrites
//! the sequence's positions from 0.

use std::sync::Arc;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::GgufTensorType;
use oxibonsai_kernels::KernelDispatcher;
use oxibonsai_model::error::ModelError;
use oxibonsai_model::hybrid::{HybridModel, LoadedModel, RecurrentSnapshot};
use oxibonsai_model::model::BonsaiModel;

use crate::engine::InferenceEngine;
use crate::engine_hybrid_gpu::{
    hybrid_backend_plan, HybridBackend, HybridBackendPlan, HybridGpu, HybridGpuState,
    HYBRID_RUNNER_LABEL,
};
use crate::error::{RuntimeError, RuntimeResult};
use crate::tokenizer_bridge::TokenizerBridge;

// ─────────────────────────────────────────────────────────────────────────────
// Backend knob
// ─────────────────────────────────────────────────────────────────────────────

/// Which compute backend an engine is asked to run on (the engine-level knob
/// behind the CLI's `--backend`).
///
/// * [`Backend::Auto`] — the best available tier: the GPU when one is
///   accelerated. A **hybrid** model runs on the Metal hybrid runner when
///   one serves it on this host, and otherwise on the best CPU tier, with
///   one `info` line naming why.
/// * [`Backend::Cpu`] — the best CPU SIMD tier, all the way down: a dense
///   model is constructed inside a
///   [`CpuOnlyBackendScope`](oxibonsai_kernels::gpu_backend::CpuOnlyBackendScope),
///   so the per-layer dispatchers `oxibonsai-model` creates internally land
///   on the CPU as well, and nothing is uploaded to a GPU; a hybrid model
///   never builds the Metal runner.
/// * [`Backend::Metal`] — the Metal GPU, or a typed error: a dense model on a
///   build/host without it ([`EngineError::BackendUnavailable`]), or a hybrid
///   model the Metal runner cannot serve here
///   ([`EngineError::HybridGpuBackendUnsupported`], naming the constraint).
///
/// [`InferenceEngine::backend`] reports the knob an engine was built with;
/// [`InferenceEngine::hybrid_backend`] reports the executor a hybrid engine
/// actually decodes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Backend {
    /// Best available tier (GPU when accelerated; the Metal hybrid runner or
    /// the CPU for a hybrid model).
    #[default]
    Auto,
    /// Best CPU SIMD tier.
    Cpu,
    /// Metal GPU, or an error.
    Metal,
}

impl Backend {
    /// Stable lower-case name (`"auto"`, `"cpu"`, `"metal"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }

    /// Parse a user-supplied name, case-insensitively. `None` for anything
    /// else.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "metal" | "gpu" => Some(Self::Metal),
            _ => None,
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| format!("unknown backend `{s}` (expected auto, cpu or metal)"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RoPE-scaling knob (`--rope-scaling auto|on|off`)
// ─────────────────────────────────────────────────────────────────────────────

/// What `--rope-scaling` resolved to for one GGUF at load time: the scaling
/// the file declares, and the one the model is actually built with once the
/// override is applied (see
/// [`InferenceEngine::from_gguf_with_backend_and_rope`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RopeScalingAtLoad {
    /// The requested override.
    pub mode: oxibonsai_core::config::RopeScalingOverride,
    /// `<arch>.rope.scaling.*` exactly as the file declares it.
    pub declared: oxibonsai_core::config::RopeScaling,
    /// What the model's RoPE table is built from.
    pub effective: oxibonsai_core::config::RopeScaling,
}

/// Resolve — and log at INFO — which RoPE scaling strategy a load of `gguf`
/// under `mode` will use.
///
/// # Errors
///
/// A malformed `<arch>.rope.scaling.*` declaration, or
/// [`RopeScalingOverride::On`](oxibonsai_core::config::RopeScalingOverride::On)
/// on a file that declares no scaling — both as [`RuntimeError::Config`],
/// before any weight is touched.
pub fn resolve_rope_scaling_at_load(
    gguf: &GgufFile<'_>,
    mode: oxibonsai_core::config::RopeScalingOverride,
) -> RuntimeResult<RopeScalingAtLoad> {
    let arch = gguf_architecture(gguf);
    let declared = oxibonsai_core::config::RopeScaling::from_metadata(&gguf.metadata, &arch)
        .map_err(|e| RuntimeError::Config(format!("RoPE scaling metadata: {e}")))?;
    let effective = mode
        .apply(declared.clone(), &arch)
        .map_err(|e| RuntimeError::Config(format!("--rope-scaling {mode}: {e}")))?;
    tracing::info!(
        mode = %mode,
        declared = ?declared,
        effective = ?effective,
        architecture = %arch,
        "RoPE scaling strategy active at load"
    );
    Ok(RopeScalingAtLoad {
        mode,
        declared,
        effective,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Typed engine errors
// ─────────────────────────────────────────────────────────────────────────────

/// Why the engine refused an operation — the typed half of "never silently
/// fall back or pretend".
///
/// Carried to callers of [`RuntimeResult`]-returning APIs as the dedicated
/// [`RuntimeError::Engine`] variant (via `#[from]`), whose display keeps the
/// stable `[CODE]` from [`EngineError::error_code`]; [`engine_error_code`]
/// recovers the code from such an error. APIs that can return the enum
/// directly — [`InferenceEngine::require_dense`] — do.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// A dense-only operation was asked of an engine holding a hybrid model.
    #[error(
        "{operation} requires a dense (qwen3-family) model, but this engine holds a hybrid \
         `{architecture}` model: {reason}"
    )]
    NotADenseModel {
        /// The refused operation.
        operation: &'static str,
        /// `general.architecture` of the loaded model.
        architecture: String,
        /// Why the hybrid model cannot do it.
        reason: &'static str,
    },
    /// An operation that must roll rejected positions back was asked of an
    /// engine whose sequence state includes a recurrence.
    #[error(
        "{operation} needs to roll the sequence back to an earlier position, which a recurrent \
         (Gated-DeltaNet) state cannot do by moving a cursor; this engine holds a `{architecture}` \
         model"
    )]
    RecurrentRollbackRequired {
        /// The refused operation.
        operation: &'static str,
        /// `general.architecture` of the loaded model.
        architecture: String,
    },
    /// A GPU backend was explicitly requested for a hybrid model the Metal
    /// hybrid runner cannot serve on this build or host.
    #[error(
        "backend `{requested}` was explicitly requested for a hybrid `{architecture}` model, but \
         the Metal hybrid runner cannot serve it here: {reason}; use backend auto or cpu"
    )]
    HybridGpuBackendUnsupported {
        /// The backend that was asked for.
        requested: Backend,
        /// `general.architecture` of the model.
        architecture: String,
        /// The violated constraint: no Metal build or device, a geometry or
        /// weight format the runner does not serve, or no room for a KV
        /// window.
        reason: String,
    },
    /// The requested backend is not available in this build or on this host.
    #[error("backend `{requested}` is unavailable: {reason}")]
    BackendUnavailable {
        /// The backend that was asked for.
        requested: Backend,
        /// What is missing.
        reason: String,
    },
    /// A hybrid decode/prefill was asked to continue from a position other
    /// than the next one.
    #[error(
        "hybrid model positions must be contiguous: the recurrent state has consumed {expected} \
         tokens, so the next position is {expected}, not {got}"
    )]
    NonContiguousPosition {
        /// The only position the recurrent state can continue from.
        expected: usize,
        /// The position that was asked for.
        got: usize,
    },
    /// A caller-supplied dense token-embedding table was passed for a hybrid
    /// model.
    #[error(
        "a shared dense token-embedding table ({len} floats) was supplied for a hybrid \
         `{architecture}` model, whose embedding is decoded row-wise from the GGUF in the \
         rotated basis; pass an empty table"
    )]
    SharedEmbeddingUnsupported {
        /// `general.architecture` of the model.
        architecture: String,
        /// Length of the rejected table.
        len: usize,
    },
    /// The engine holds an opaque attached recurrent state with no snapshot
    /// hook.
    #[error(
        "the attached recurrent state `{name}` has no snapshot hook, so the sequence cannot be \
         snapshotted"
    )]
    RecurrentStateNotSnapshotable {
        /// `RecurrentState::recurrent_name` of the attached state.
        name: String,
    },
    /// A snapshot does not belong to this engine's current sequence.
    #[error("sequence snapshot cannot be restored: {detail}")]
    SnapshotMismatch {
        /// What does not match.
        detail: String,
    },
}

impl EngineError {
    /// Short, stable code for monitoring and for [`engine_error_code`].
    #[must_use]
    pub const fn error_code(&self) -> &'static str {
        match self {
            Self::NotADenseModel { .. } => "NOT_A_DENSE_MODEL",
            Self::RecurrentRollbackRequired { .. } => "RECURRENT_ROLLBACK_REQUIRED",
            Self::HybridGpuBackendUnsupported { .. } => "HYBRID_GPU_BACKEND_UNSUPPORTED",
            Self::BackendUnavailable { .. } => "BACKEND_UNAVAILABLE",
            Self::NonContiguousPosition { .. } => "NON_CONTIGUOUS_POSITION",
            Self::SharedEmbeddingUnsupported { .. } => "SHARED_EMBEDDING_UNSUPPORTED",
            Self::RecurrentStateNotSnapshotable { .. } => "RECURRENT_STATE_NOT_SNAPSHOTABLE",
            Self::SnapshotMismatch { .. } => "SNAPSHOT_MISMATCH",
        }
    }

    /// Every code [`Self::error_code`] can return.
    pub const ALL_CODES: [&'static str; 8] = [
        "NOT_A_DENSE_MODEL",
        "RECURRENT_ROLLBACK_REQUIRED",
        "HYBRID_GPU_BACKEND_UNSUPPORTED",
        "BACKEND_UNAVAILABLE",
        "NON_CONTIGUOUS_POSITION",
        "SHARED_EMBEDDING_UNSUPPORTED",
        "RECURRENT_STATE_NOT_SNAPSHOTABLE",
        "SNAPSHOT_MISMATCH",
    ];
}

#[cfg(test)]
impl EngineError {
    /// One refusal of every kind, in [`Self::ALL_CODES`] order — for tests
    /// that must cover every code (the error type's own, and the HTTP
    /// mapping's).
    pub(crate) fn one_of_each_kind() -> Vec<Self> {
        vec![
            Self::NotADenseModel {
                operation: "op",
                architecture: "qwen35".into(),
                reason: "r",
            },
            Self::RecurrentRollbackRequired {
                operation: "op",
                architecture: "qwen35".into(),
            },
            Self::HybridGpuBackendUnsupported {
                requested: Backend::Metal,
                architecture: "qwen35".into(),
                reason: "no Metal device was found on this host".into(),
            },
            Self::BackendUnavailable {
                requested: Backend::Metal,
                reason: "none".into(),
            },
            Self::NonContiguousPosition {
                expected: 3,
                got: 5,
            },
            Self::SharedEmbeddingUnsupported {
                architecture: "qwen35".into(),
                len: 7,
            },
            Self::RecurrentStateNotSnapshotable { name: "x".into() },
            Self::SnapshotMismatch { detail: "d".into() },
        ]
    }
}

/// The [`EngineError::error_code`] carried by `error`, if it is an engine
/// refusal ([`RuntimeError::Engine`]); `None` for every other error.
///
/// Also re-exported as `oxibonsai_runtime::engine::engine_error_code`; both
/// paths are this one function.
#[must_use]
pub fn engine_error_code(error: &RuntimeError) -> Option<&'static str> {
    match error {
        RuntimeError::Engine(refusal) => Some(refusal.error_code()),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequence snapshots
// ─────────────────────────────────────────────────────────────────────────────

/// An exact rollback point for an engine's current sequence (see the module
/// docs for what is and is not snapshot-able).
#[derive(Debug, Clone)]
pub struct SequenceSnapshot {
    /// Tokens consumed by the sequence when the snapshot was taken.
    position: usize,
    /// The engine sequence the snapshot belongs to.
    sequence_id: u64,
    /// What besides the KV cursor the snapshot restores.
    state: SnapshotState,
}

/// The per-sequence state a [`SequenceSnapshot`] carries besides the KV
/// cursor, by executor.
#[derive(Debug, Clone)]
enum SnapshotState {
    /// A dense engine: the KV cursor is the whole state.
    Dense,
    /// A hybrid engine on the CPU: a deep copy of the model's recurrent
    /// cache.
    HybridCpu(Box<RecurrentSnapshot>),
    /// A hybrid engine on the Metal runner: a copy of the device recurrent
    /// state (the device KV needs none — see the module docs).
    HybridMetal(Box<HybridGpuState>),
}

impl SnapshotState {
    /// The executor kind, for mismatch messages.
    fn kind(&self) -> &'static str {
        match self {
            Self::Dense => "a dense model",
            Self::HybridCpu(_) => "a hybrid model on the CPU",
            Self::HybridMetal(_) => "a hybrid model on the Metal runner",
        }
    }
}

impl SequenceSnapshot {
    /// Tokens the sequence had consumed at snapshot time.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether this snapshot carries a recurrent state (i.e. was taken from a
    /// hybrid engine, on either executor).
    #[must_use]
    pub fn has_recurrent_state(&self) -> bool {
        !matches!(self.state, SnapshotState::Dense)
    }

    /// The executor the snapshot was taken on: `None` for a dense engine.
    #[must_use]
    pub fn hybrid_backend(&self) -> Option<HybridBackend> {
        match self.state {
            SnapshotState::Dense => None,
            SnapshotState::HybridCpu(_) => Some(HybridBackend::Cpu),
            SnapshotState::HybridMetal(_) => Some(HybridBackend::Metal),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction helpers shared by the engine's constructors
// ─────────────────────────────────────────────────────────────────────────────

/// The best CPU SIMD dispatcher for this host.
pub(crate) fn cpu_dispatcher() -> KernelDispatcher {
    KernelDispatcher::with_tier(oxibonsai_kernels::cpu_kernel_tier())
}

/// `general.architecture` of a GGUF (`""` when absent).
pub(crate) fn gguf_architecture(gguf: &GgufFile<'_>) -> String {
    LoadedModel::architecture_of(gguf)
}

/// The hybrid half of an engine, as [`load_hybrid`] built it.
pub(crate) struct HybridLoad<'a> {
    /// The CPU model (bound at the engine's KV window).
    pub(crate) model: HybridModel<'a>,
    /// The CPU dispatcher the engine holds.
    pub(crate) kernel: KernelDispatcher,
    /// The Metal hybrid runner, when one serves the model.
    pub(crate) gpu: Option<HybridGpu<'a>>,
}

/// Build the hybrid half of an engine (see the module docs): the CPU model on
/// a pinned CPU dispatcher, its KV window clamped to the model's declared
/// context, and — for [`Backend::Metal`] and [`Backend::Auto`] — the Metal
/// hybrid runner when one serves the model on this host, with the window
/// budgeted for the CPU model and the runner together (the CPU model is
/// rebound at that window when the budget is the tighter limit).
///
/// # Errors
///
/// Model-load errors, and for [`Backend::Metal`] the typed
/// [`EngineError::HybridGpuBackendUnsupported`] naming why no runner serves
/// the model: no Metal build or device, a geometry or weight format the
/// runner does not serve, no room for a KV window, or a failed construction.
pub(crate) fn load_hybrid<'a>(
    gguf: &'a GgufFile<'a>,
    max_seq_len: usize,
    backend: Backend,
) -> RuntimeResult<HybridLoad<'a>> {
    let architecture = gguf_architecture(gguf);
    let kernel = cpu_dispatcher();
    let config = HybridModel::config_from_gguf(gguf)?;
    let declared = config.base.max_context_length.max(1);
    let cpu_window = max_seq_len.min(declared);
    let bind = |window: usize| -> RuntimeResult<HybridModel<'a>> {
        Ok(HybridModel::from_gguf_with(
            gguf,
            config.clone(),
            window,
            &Arc::new(cpu_dispatcher_clone(&kernel)),
        )?)
    };
    let refuse = |reason: String| -> RuntimeError {
        EngineError::HybridGpuBackendUnsupported {
            requested: backend,
            architecture: architecture.clone(),
            reason,
        }
        .into()
    };
    let fall_back = |reason: &str| {
        tracing::info!(
            architecture = %architecture,
            tier = %kernel.tier(),
            reason,
            "hybrid model: the Metal hybrid runner does not serve it here, so backend `auto` runs \
             it on the CPU tier"
        );
    };

    let model = bind(cpu_window)?;
    let (model, gpu) = match backend {
        Backend::Cpu => (model, None),
        Backend::Auto | Backend::Metal => match hybrid_backend_plan(gguf, &model, max_seq_len) {
            HybridBackendPlan::Cpu { reason } => {
                if backend == Backend::Metal {
                    return Err(refuse(reason));
                }
                fall_back(&reason);
                (model, None)
            }
            HybridBackendPlan::Metal { window, mapped } => {
                if window.clamped() {
                    tracing::warn!(
                        requested = window.requested,
                        window = window.window,
                        limits = ?window.limits_applied,
                        "hybrid model on the Metal runner: the requested KV window is clamped to \
                         {} positions ({})",
                        window.window,
                        window.describe_limits()
                    );
                }
                let model = if window.window == model.max_seq_len() {
                    model
                } else {
                    bind(window.window)?
                };
                let summary = window.summary();
                match HybridGpu::build(gguf, &model, window) {
                    Ok(gpu) => {
                        tracing::info!(
                            architecture = %architecture,
                            mapped,
                            "hybrid model: decoding on the Metal hybrid runner; {summary}"
                        );
                        (model, Some(gpu))
                    }
                    Err(reason) => {
                        if backend == Backend::Metal {
                            return Err(refuse(reason));
                        }
                        fall_back(&reason);
                        let model = if model.max_seq_len() == cpu_window {
                            model
                        } else {
                            bind(cpu_window)?
                        };
                        (model, None)
                    }
                }
            }
        },
    };
    if gpu.is_none() && cpu_window < max_seq_len {
        tracing::warn!(
            requested = max_seq_len,
            declared,
            "requested max_seq_len exceeds the hybrid model's declared context; clamped"
        );
    }
    Ok(HybridLoad { model, kernel, gpu })
}

/// A second dispatcher on the same CPU tier (a `KernelDispatcher` is not
/// `Clone`; the model's layers hold their own `Arc`).
fn cpu_dispatcher_clone(kernel: &KernelDispatcher) -> KernelDispatcher {
    KernelDispatcher::with_tier(kernel.tier())
}

/// Resolve the engine dispatcher for a **dense** model under `backend`.
pub(crate) fn dense_dispatcher(backend: Backend) -> RuntimeResult<KernelDispatcher> {
    match backend {
        Backend::Auto => Ok(KernelDispatcher::auto_detect()),
        Backend::Cpu => Ok(cpu_dispatcher()),
        Backend::Metal => metal_dispatcher(),
    }
}

/// A dispatcher on the Metal GPU tier, or a typed refusal.
fn metal_dispatcher() -> RuntimeResult<KernelDispatcher> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let kernel =
            KernelDispatcher::try_with_tier(oxibonsai_kernels::KernelTier::Gpu).map_err(|e| {
                RuntimeError::from(EngineError::BackendUnavailable {
                    requested: Backend::Metal,
                    reason: e.to_string(),
                })
            })?;
        Ok(kernel)
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        Err(EngineError::BackendUnavailable {
            requested: Backend::Metal,
            reason: "this build has no Metal backend compiled in (macOS with the `metal` feature \
                     is required)"
                .to_string(),
        }
        .into())
    }
}

/// Build a tokenizer from a GGUF's own embedded vocabulary
/// (`tokenizer.ggml.*` metadata, including the `tokenizer.ggml.pre`
/// pre-tokenizer — `qwen35` for Bonsai 2).
///
/// Bonsai 2 ships no `tokenizer.json`: its vocabulary, merges and special
/// tokens live only in the GGUF. This is the seam the CLI and server use when
/// no vocabulary-matching tokenizer file exists next to the model.
///
/// # Errors
///
/// [`RuntimeError::Tokenizer`] when the GGUF carries no usable embedded
/// tokenizer.
pub fn tokenizer_from_gguf(gguf: &GgufFile<'_>) -> RuntimeResult<TokenizerBridge> {
    let native = oxibonsai_tokenizer::OxiTokenizer::from_gguf_metadata(&gguf.metadata)
        .map_err(|e| RuntimeError::Tokenizer(format!("GGUF-embedded tokenizer: {e}")))?;
    Ok(TokenizerBridge::from_native_tokenizer(native))
}

// ─────────────────────────────────────────────────────────────────────────────
// The seam itself
// ─────────────────────────────────────────────────────────────────────────────

impl<'a> InferenceEngine<'a> {
    /// The loaded model, whichever kind it is.
    pub fn loaded_model(&self) -> &LoadedModel<'a> {
        &self.model
    }

    /// The loaded model, mutably.
    ///
    /// Resetting or advancing the model through this handle bypasses the
    /// engine's sequence tracking, so a [`SequenceSnapshot`] taken before
    /// such a direct mutation must not be restored after it.
    pub fn loaded_model_mut(&mut self) -> &mut LoadedModel<'a> {
        &mut self.model
    }

    /// The dense model, or `None` for a hybrid engine.
    pub fn dense_model(&self) -> Option<&BonsaiModel<'a>> {
        self.model.as_dense()
    }

    /// The dense model, mutably, or `None` for a hybrid engine.
    pub fn dense_model_mut(&mut self) -> Option<&mut BonsaiModel<'a>> {
        self.model.as_dense_mut()
    }

    /// The hybrid model, or `None` for a dense engine.
    pub fn hybrid_model(&self) -> Option<&HybridModel<'a>> {
        self.model.as_hybrid()
    }

    /// The hybrid model, mutably, or `None` for a dense engine.
    pub fn hybrid_model_mut(&mut self) -> Option<&mut HybridModel<'a>> {
        self.model.as_hybrid_mut()
    }

    /// The dense model, or the typed [`EngineError::NotADenseModel`] naming
    /// `operation`.
    ///
    /// # Errors
    ///
    /// [`EngineError::NotADenseModel`] on a hybrid engine.
    pub fn require_dense(&self, operation: &'static str) -> Result<&BonsaiModel<'a>, EngineError> {
        match &self.model {
            LoadedModel::Dense(model) => Ok(model),
            LoadedModel::Hybrid(model) => Err(EngineError::NotADenseModel {
                operation,
                architecture: model.config().base.architecture.clone(),
                reason: "the operation is implemented for the dense block stack only",
            }),
        }
    }

    /// `true` when this engine holds a hybrid (`qwen35`) model.
    pub fn is_hybrid(&self) -> bool {
        self.model.is_hybrid()
    }

    /// The executor a hybrid engine decodes on — the Metal hybrid runner or
    /// the CPU model — or `None` for a dense engine.
    ///
    /// Unlike [`backend`](Self::backend), which reports the knob the engine
    /// was built with (`auto` stays `auto`), this is what `auto` resolved to.
    pub fn hybrid_backend(&self) -> Option<HybridBackend> {
        if !self.model.is_hybrid() {
            return None;
        }
        Some(if self.hybrid_gpu.is_some() {
            HybridBackend::Metal
        } else {
            HybridBackend::Cpu
        })
    }

    /// The KV window of a Metal-backed hybrid engine and every limit that
    /// went into it; `None` for any other engine.
    pub fn hybrid_metal_window(&self) -> Option<&crate::engine_hybrid_gpu::HybridMetalWindow> {
        self.hybrid_gpu.as_ref().map(HybridGpu::window)
    }

    /// Whether a Metal-backed hybrid engine's runner reads the weights in
    /// place from the file mapping (`Some(false)`: it copied them); `None`
    /// for any other engine.
    pub fn hybrid_metal_weights_mapped(&self) -> Option<bool> {
        self.hybrid_gpu.as_ref().map(HybridGpu::is_mapped)
    }

    /// Bytes the Metal device reports as allocated by this process
    /// (`MTLDevice.currentAllocatedSize`), read through a Metal-backed
    /// hybrid engine's runner; `None` for any other engine.
    pub fn hybrid_metal_device_allocated_bytes(&self) -> Option<u64> {
        self.hybrid_gpu
            .as_ref()
            .map(HybridGpu::device_allocated_bytes)
    }

    /// `general.architecture` of the loaded model (`"qwen3"`, `"qwen35"`, …).
    pub fn architecture(&self) -> &str {
        match &self.model {
            LoadedModel::Dense(model) => &model.config().architecture,
            LoadedModel::Hybrid(model) => &model.config().base.architecture,
        }
    }

    /// The model's display name (`general.name`).
    pub fn model_name(&self) -> &str {
        match &self.model {
            LoadedModel::Dense(model) => &model.config().model_name,
            LoadedModel::Hybrid(model) => &model.config().base.model_name,
        }
    }

    /// Vocabulary width of the LM head.
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// Hidden width of the residual stream.
    pub fn hidden_size(&self) -> usize {
        self.model.hidden_size()
    }

    /// Layers in the stack.
    pub fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    /// The KV window this engine's model was built with.
    pub fn max_seq_len(&self) -> usize {
        self.model.max_seq_len()
    }

    /// Highest position + 1 this engine accepts: the dense model's effective
    /// context (`BonsaiModel::max_context`), the hybrid model's KV window.
    pub fn max_context(&self) -> usize {
        match &self.model {
            LoadedModel::Dense(model) => model.max_context(),
            LoadedModel::Hybrid(model) => model.max_seq_len(),
        }
    }

    /// The context length the model *declares* (`<arch>.context_length`) —
    /// what `/v1/models` reports, not what this engine allocated.
    pub fn context_length(&self) -> usize {
        match &self.model {
            LoadedModel::Dense(model) => model.context_length(),
            LoadedModel::Hybrid(model) => model.config().base.max_context_length,
        }
    }

    /// Tokens the current sequence has consumed (the KV cursor; for a hybrid
    /// model the recurrent state's token count — the Metal runner's when the
    /// engine decodes on it).
    pub fn sequence_position(&self) -> usize {
        if let Some(gpu) = &self.hybrid_gpu {
            return gpu.token_count();
        }
        match &self.model {
            LoadedModel::Dense(model) => model.kv_cache().seq_len(),
            LoadedModel::Hybrid(model) => model.recurrent().token_count(),
        }
    }

    /// `(layers, kv heads, head dim)` of the KV cache: every layer for a dense
    /// model, only the full-attention layers for a hybrid one.
    pub fn kv_cache_geometry(&self) -> (usize, usize, usize) {
        match &self.model {
            LoadedModel::Dense(model) => {
                let cache = model.kv_cache();
                (cache.num_layers(), cache.num_kv_heads(), cache.head_dim())
            }
            LoadedModel::Hybrid(model) => {
                let cache = model.kv_cache();
                (cache.num_layers(), cache.num_kv_heads(), cache.head_dim())
            }
        }
    }

    /// The resolved dominant weight quantization type (cli-16): what the
    /// load-time log line and `kernel_label` report.
    pub fn dominant_quant_type(&self) -> GgufTensorType {
        match &self.model {
            LoadedModel::Dense(model) => model.dominant_quant_type(),
            LoadedModel::Hybrid(model) => model.quant_type(),
        }
    }

    /// `"<dominant quant type> <kernel tier>"`, e.g. `"PQ2_0 NEON (128-bit)"`
    /// (cli-16) — the resolved quant family plus the effective tier, never a
    /// hardcoded kernel-family string. A Metal-backed hybrid engine reports
    /// `"<quant type> Metal (hybrid runner)"`: its decode runs the runner's
    /// own kernels, not the CPU dispatcher the engine also holds.
    pub fn kernel_label(&self) -> String {
        if self.hybrid_gpu.is_some() {
            return format!("{} {}", self.dominant_quant_type(), HYBRID_RUNNER_LABEL);
        }
        self.kernel.kernel_label(self.dominant_quant_type())
    }

    /// One-line model summary (`LoadedModel::describe`).
    pub fn model_description(&self) -> String {
        self.model.describe()
    }

    /// Bytes held by the KV cache — for a Metal-backed hybrid engine both
    /// residents': the CPU model's (grown only as far as its own passes
    /// reached) and the runner's (allocated for the whole window).
    pub fn kv_cache_memory_bytes(&self) -> usize {
        let runner = self.hybrid_gpu.as_ref().map_or(0, |gpu| {
            usize::try_from(gpu.kv_cache_bytes()).unwrap_or(usize::MAX)
        });
        let own = match &self.model {
            LoadedModel::Dense(model) => model.kv_cache_memory_bytes(),
            LoadedModel::Hybrid(model) => model.kv_cache().memory_bytes(),
        };
        own.saturating_add(runner)
    }

    /// Bytes of per-sequence state: the KV cache plus the hybrid model's
    /// recurrent state plus any attached `RecurrentState`.
    pub fn sequence_state_bytes(&self) -> usize {
        self.kv_cache_memory_bytes() + self.recurrent_memory_bytes()
    }

    /// The backend this engine was built for.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The epoch this engine's GPU weight uploads are attributed to
    /// (`MET-M1`); `UNATTRIBUTED_MODEL_EPOCH` for an engine that uploaded
    /// nothing under its own scope.
    pub fn model_epoch(&self) -> u64 {
        self.model_epoch
    }

    /// What this engine's construction placed in the GPU weight cache
    /// (`MET-M1` / replica sharing): `fresh_*` counts buffers it uploaded
    /// itself, `shared_*` the byte-identical buffers it found already
    /// resident — typically a sibling replica's — and registered a reference
    /// to instead of uploading a second copy. Empty for an engine that
    /// uploaded nothing (a CPU tier, the ternary fused route, a hybrid model).
    pub fn gpu_upload_stats(&self) -> oxibonsai_kernels::gpu_backend::UploadStats {
        self.gpu_uploads
    }

    /// Weight-cache registrations the GPU backend currently holds for this
    /// engine's [`model_epoch`](Self::model_epoch) — what the engine's `Drop`
    /// will release. `0` for an engine with no epoch of its own.
    pub fn gpu_weight_registrations(&self) -> usize {
        if self.model_epoch == oxibonsai_kernels::gpu_backend::UNATTRIBUTED_MODEL_EPOCH {
            return 0;
        }
        oxibonsai_kernels::gpu_backend::model_registration_count(&self.kernel, self.model_epoch)
    }

    // ── Forward dispatch ────────────────────────────────────────────────

    /// Enforce the hybrid recurrence's position contract before a forward at
    /// `pos`: the next position is the recurrent state's token count; `0`
    /// starts a new sequence (an implicit reset, exactly as a dense prefill at
    /// position 0 overwrites everything after it).
    ///
    /// The implicit reset clears everything an explicit
    /// [`reset`](InferenceEngine::reset) clears: the hybrid model's KV cursor
    /// and its own recurrent state, the Metal runner's state when the engine
    /// decodes on it, **and** an attached opaque
    /// [`RecurrentState`](crate::engine_control::RecurrentState) — it belongs
    /// to the sequence being abandoned, and unlike a KV cache it is not
    /// masked by position, so keeping it would contaminate the new sequence.
    ///
    /// On a Metal-backed engine the count is the runner's: its device state
    /// is the one every forward advances.
    fn prepare_hybrid_position(&mut self, pos: usize) -> RuntimeResult<()> {
        let LoadedModel::Hybrid(model) = &mut self.model else {
            return Ok(());
        };
        let expected = match &self.hybrid_gpu {
            Some(gpu) => gpu.token_count(),
            None => model.recurrent().token_count(),
        };
        if pos == expected {
            return Ok(());
        }
        if pos == 0 {
            tracing::debug!(
                consumed = expected,
                "hybrid forward at position 0: starting a new sequence (implicit reset)"
            );
            model.reset();
            if let Some(gpu) = self.hybrid_gpu.as_mut() {
                gpu.reset();
            }
            if let Some(state) = self.recurrent.as_deref_mut() {
                state.reset_recurrent();
            }
            self.sequence_id = self.sequence_id.wrapping_add(1);
            return Ok(());
        }
        if pos < expected {
            return Err(RuntimeError::Model(
                ModelError::RecurrentRollbackUnsupported {
                    pos,
                    tokens: expected,
                },
            ));
        }
        Err(EngineError::NonContiguousPosition { expected, got: pos }.into())
    }

    /// Single-token forward at `pos` on the engine's own dispatcher,
    /// returning the `[vocab]` logit row.
    ///
    /// Dense: exactly `BonsaiModel::forward(token, pos, &self.kernel)`.
    /// Hybrid: the Metal runner's `forward_into` on a Metal-backed engine,
    /// else `HybridModel::forward`, after the position contract.
    pub(crate) fn forward_logits(&mut self, token: u32, pos: usize) -> RuntimeResult<Vec<f32>> {
        if pos == 0 && !self.model.is_hybrid() {
            self.sequence_id = self.sequence_id.wrapping_add(1);
        }
        self.prepare_hybrid_position(pos)?;
        let row = match (&mut self.model, self.hybrid_gpu.as_mut()) {
            (LoadedModel::Dense(model), _) => model.forward(token, pos, &self.kernel)?,
            (LoadedModel::Hybrid(_), Some(gpu)) => gpu.forward_logits(token, pos)?,
            (LoadedModel::Hybrid(model), None) => model.forward_alloc(token, pos)?,
        };
        // Test-only scripted generation (`crate::engine::ScriptedLogits`).
        #[cfg(test)]
        let row = self.scripted_row(row, false)?;
        Ok(row)
    }

    /// Single-token forward at `pos` on an explicit dispatcher — the greedy
    /// path's coherent CPU-fallback replay. A hybrid model carries its
    /// dispatcher inside its layers (and a Metal-backed one decodes on its
    /// runner), so `kernel` only reaches the dense arm.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn forward_logits_on(
        &mut self,
        token: u32,
        pos: usize,
        kernel: &KernelDispatcher,
    ) -> RuntimeResult<Vec<f32>> {
        if pos == 0 && !self.model.is_hybrid() {
            self.sequence_id = self.sequence_id.wrapping_add(1);
        }
        self.prepare_hybrid_position(pos)?;
        match (&mut self.model, self.hybrid_gpu.as_mut()) {
            (LoadedModel::Dense(model), _) => Ok(model.forward(token, pos, kernel)?),
            (LoadedModel::Hybrid(_), Some(gpu)) => gpu.forward_logits(token, pos),
            (LoadedModel::Hybrid(model), None) => Ok(model.forward_alloc(token, pos)?),
        }
    }

    /// Batched prefill of `tokens` from `pos_start`, returning the **last**
    /// token's `[vocab]` logit row.
    ///
    /// Dense: exactly `BonsaiModel::forward_prefill(tokens, pos_start,
    /// &self.kernel)`. Hybrid: the prompt in `prefill_chunk`-token chunks
    /// (design §3.10) after the position contract — through the Metal
    /// runner's `forward_prefill` on a Metal-backed engine (each chunk also
    /// capped at the runner's own batch size, fixed when it was built), else
    /// `HybridModel::forward_prefill`. Either way the recurrent state
    /// advances once per token, in order, so the chunking never changes the
    /// result.
    pub(crate) fn prefill_logits(
        &mut self,
        tokens: &[u32],
        pos_start: usize,
    ) -> RuntimeResult<Vec<f32>> {
        if pos_start == 0 && !self.model.is_hybrid() {
            self.sequence_id = self.sequence_id.wrapping_add(1);
        }
        self.prepare_hybrid_position(pos_start)?;
        match (&mut self.model, self.hybrid_gpu.as_mut()) {
            (LoadedModel::Dense(model), _) => {
                Ok(model.forward_prefill(tokens, pos_start, &self.kernel)?)
            }
            (LoadedModel::Hybrid(model), gpu) => {
                if tokens.is_empty() {
                    return Err(RuntimeError::Model(ModelError::MissingTensor {
                        name: "forward_prefill: empty token_ids".into(),
                    }));
                }
                if let Some(gpu) = gpu {
                    return gpu.prefill_logits(tokens, pos_start, model.prefill_chunk());
                }
                let mut logits = vec![0.0f32; model.config().base.vocab_size];
                model.forward_prefill(tokens, pos_start, &mut logits)?;
                Ok(logits)
            }
        }
    }

    // ── Sequence snapshots ──────────────────────────────────────────────

    /// Take an exact rollback point at the current position (see the module
    /// docs) — on a Metal-backed hybrid engine, a copy of the runner's
    /// device recurrent state.
    ///
    /// # Errors
    ///
    /// [`EngineError::RecurrentStateNotSnapshotable`] when an opaque
    /// `RecurrentState` is attached (it has no snapshot hook).
    pub fn snapshot_sequence(&self) -> RuntimeResult<SequenceSnapshot> {
        if let Some(state) = self.recurrent.as_deref() {
            return Err(EngineError::RecurrentStateNotSnapshotable {
                name: state.recurrent_name().to_string(),
            }
            .into());
        }
        let state = match (&self.model, &self.hybrid_gpu) {
            (LoadedModel::Dense(_), _) => SnapshotState::Dense,
            (LoadedModel::Hybrid(_), Some(gpu)) => {
                SnapshotState::HybridMetal(Box::new(gpu.snapshot()?))
            }
            (LoadedModel::Hybrid(model), None) => {
                SnapshotState::HybridCpu(Box::new(model.recurrent().snapshot()))
            }
        };
        Ok(SequenceSnapshot {
            position: self.sequence_position(),
            sequence_id: self.sequence_id,
            state,
        })
    }

    /// Restore a [`SequenceSnapshot`] taken from this engine's **current**
    /// sequence: the KV cursor moves back to the snapshot's position (the
    /// entries below it were never overwritten) and a hybrid model's
    /// recurrent state is restored from the snapshot's deep copy — the CPU
    /// model's, or the Metal runner's device state.
    ///
    /// # Errors
    ///
    /// [`EngineError::SnapshotMismatch`] when the snapshot belongs to another
    /// sequence (any reset since it was taken), to the other model kind or
    /// the other hybrid executor, or lies beyond the current position.
    pub fn restore_sequence(&mut self, snapshot: &SequenceSnapshot) -> RuntimeResult<()> {
        if snapshot.sequence_id != self.sequence_id {
            return Err(EngineError::SnapshotMismatch {
                detail: format!(
                    "it was taken from sequence {} but the engine is on sequence {} (a reset or a \
                     restart at position 0 happened since)",
                    snapshot.sequence_id, self.sequence_id
                ),
            }
            .into());
        }
        let current = self.sequence_position();
        if snapshot.position > current {
            return Err(EngineError::SnapshotMismatch {
                detail: format!(
                    "it is at position {} but the sequence is only at {current}",
                    snapshot.position
                ),
            }
            .into());
        }
        match (&mut self.model, self.hybrid_gpu.as_mut(), &snapshot.state) {
            (LoadedModel::Dense(model), _, SnapshotState::Dense) => {
                model.kv_cache_mut().truncate(snapshot.position);
                Ok(())
            }
            (LoadedModel::Hybrid(model), None, SnapshotState::HybridCpu(recurrent)) => {
                model.recurrent_mut().restore(recurrent)?;
                model.kv_cache_mut().set_seq_len(snapshot.position);
                Ok(())
            }
            (LoadedModel::Hybrid(_), Some(gpu), SnapshotState::HybridMetal(state)) => {
                gpu.restore(state)
            }
            (model, gpu, state) => {
                let engine = match (model.is_hybrid(), gpu.is_some()) {
                    (false, _) => "a dense model",
                    (true, false) => "a hybrid model on the CPU",
                    (true, true) => "a hybrid model on the Metal runner",
                };
                Err(EngineError::SnapshotMismatch {
                    detail: format!(
                        "it was taken from {} but this engine holds {engine}",
                        state.kind()
                    ),
                }
                .into())
            }
        }
    }
}

#[cfg(test)]
#[path = "engine_seam_tests.rs"]
mod tests;
