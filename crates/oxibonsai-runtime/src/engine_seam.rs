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
//!   a dense engine is byte-identical to before), and the hybrid model's
//!   `forward` / chunked `forward_prefill` for a hybrid one;
//! * **introspection** — vocabulary, context, architecture, dominant quant
//!   type, cache geometry, memory, for both kinds;
//! * **dense-only access** — [`InferenceEngine::dense_model`] /
//!   [`InferenceEngine::dense_model_mut`] return `None` on a hybrid engine,
//!   and [`InferenceEngine::require_dense`] turns that into the typed
//!   [`EngineError::NotADenseModel`];
//! * **sequence state** — [`InferenceEngine::snapshot_sequence`] /
//!   [`InferenceEngine::restore_sequence`], the one exact rollback point a
//!   recurrent model supports.
//!
//! # What a hybrid engine does not do (typed refusals, never silent)
//!
//! | Operation | Dense | Hybrid |
//! |---|---|---|
//! | `generate*` / streaming / logprobs / cancellation / reset | yes | yes (CPU) |
//! | GPU-argmax greedy, fused-GPU sampled top-k | Metal fused route | no — CPU full-row path (no hybrid GPU encoder yet) |
//! | `--backend metal` ([`Backend::Metal`]) | yes | [`EngineError::HybridGpuBackendUnsupported`] |
//! | `rewind_cache` to an earlier position | KV cursor move | `ModelError::RecurrentRollbackUnsupported` |
//! | `verify_batch` / speculative decoding | yes | [`EngineError::RecurrentRollbackRequired`] |
//! | prefix-cache KV block restore | yes | [`EngineError::RecurrentRollbackRequired`] (`PrefixCachedEngine::try_new`) |
//! | embeddings (`embed`, `ModelEmbedder`) | yes (batched CPU prefill) | yes (CPU, `HybridModel::forward_hidden`) |
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
//! (`RecurrentCache::snapshot`, ~157 MB for the 27B). Restoring it is exact.
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
use crate::error::{RuntimeError, RuntimeResult};
use crate::tokenizer_bridge::TokenizerBridge;

// ─────────────────────────────────────────────────────────────────────────────
// Backend knob
// ─────────────────────────────────────────────────────────────────────────────

/// Which compute backend an engine is asked to run on (the engine-level knob
/// behind the CLI's `--backend`).
///
/// * [`Backend::Auto`] — today's behaviour: the best available tier (the GPU
///   when one is accelerated). A **hybrid** model has no GPU encoder yet, so
///   it logs that at `info` and runs on the best CPU tier.
/// * [`Backend::Cpu`] — the best CPU SIMD tier, all the way down: a dense
///   model is constructed inside a
///   [`CpuOnlyBackendScope`](oxibonsai_kernels::gpu_backend::CpuOnlyBackendScope),
///   so the per-layer dispatchers `oxibonsai-model` creates internally land
///   on the CPU as well, and nothing is uploaded to a GPU.
/// * [`Backend::Metal`] — the Metal GPU, or a typed error: unavailable on this
///   build/host ([`EngineError::BackendUnavailable`]), or a hybrid model
///   ([`EngineError::HybridGpuBackendUnsupported`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Backend {
    /// Best available tier (GPU when accelerated; CPU for a hybrid model).
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
    /// A GPU backend was explicitly requested for a hybrid model.
    #[error(
        "backend `{requested}` was explicitly requested for a hybrid `{architecture}` model, but \
         no hybrid GPU encoder exists yet (the Metal/CUDA hybrid decode path is future work); \
         use backend auto or cpu"
    )]
    HybridGpuBackendUnsupported {
        /// The backend that was asked for.
        requested: Backend,
        /// `general.architecture` of the model.
        architecture: String,
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
    /// Deep copy of the recurrent state (hybrid engines only).
    recurrent: Option<RecurrentSnapshot>,
}

impl SequenceSnapshot {
    /// Tokens the sequence had consumed at snapshot time.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether this snapshot carries a recurrent state (i.e. was taken from a
    /// hybrid engine).
    #[must_use]
    pub fn has_recurrent_state(&self) -> bool {
        self.recurrent.is_some()
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

/// Build the hybrid half of an engine: refuse an explicit GPU request, pin a
/// CPU dispatcher, clamp the KV window to the model's declared context and
/// bind the model.
///
/// Returns the model and the CPU dispatcher the engine should hold.
pub(crate) fn load_hybrid<'a>(
    gguf: &'a GgufFile<'a>,
    max_seq_len: usize,
    backend: Backend,
) -> RuntimeResult<(HybridModel<'a>, KernelDispatcher)> {
    let architecture = gguf_architecture(gguf);
    if backend == Backend::Metal {
        return Err(EngineError::HybridGpuBackendUnsupported {
            requested: backend,
            architecture,
        }
        .into());
    }
    let kernel = cpu_dispatcher();
    if backend == Backend::Auto {
        // Logged on every build, GPU-capable or not: `auto` never means a GPU
        // for a hybrid model today, and the operator should see why.
        tracing::info!(
            architecture = %architecture,
            tier = %kernel.tier(),
            "hybrid model: no hybrid GPU encoder exists yet, so backend `auto` runs this model on \
             the CPU tier (Metal/CUDA hybrid decode is future work)"
        );
    }
    let config = HybridModel::config_from_gguf(gguf)?;
    let declared = config.base.max_context_length.max(1);
    let window = max_seq_len.min(declared);
    if window < max_seq_len {
        tracing::warn!(
            requested = max_seq_len,
            declared,
            "requested max_seq_len exceeds the hybrid model's declared context; clamped"
        );
    }
    let model = HybridModel::from_gguf_with(
        gguf,
        config,
        window,
        &Arc::new(cpu_dispatcher_clone(&kernel)),
    )?;
    Ok((model, kernel))
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
    /// model also the recurrent state's token count).
    pub fn sequence_position(&self) -> usize {
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
    /// hardcoded kernel-family string.
    pub fn kernel_label(&self) -> String {
        self.kernel.kernel_label(self.dominant_quant_type())
    }

    /// One-line model summary (`LoadedModel::describe`).
    pub fn model_description(&self) -> String {
        self.model.describe()
    }

    /// Bytes held by the KV cache.
    pub fn kv_cache_memory_bytes(&self) -> usize {
        match &self.model {
            LoadedModel::Dense(model) => model.kv_cache_memory_bytes(),
            LoadedModel::Hybrid(model) => model.kv_cache().memory_bytes(),
        }
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
    /// and its own recurrent state, **and** an attached opaque
    /// [`RecurrentState`](crate::engine_control::RecurrentState) — it belongs
    /// to the sequence being abandoned, and unlike a KV cache it is not
    /// masked by position, so keeping it would contaminate the new sequence.
    fn prepare_hybrid_position(&mut self, pos: usize) -> RuntimeResult<()> {
        let LoadedModel::Hybrid(model) = &mut self.model else {
            return Ok(());
        };
        let expected = model.recurrent().token_count();
        if pos == expected {
            return Ok(());
        }
        if pos == 0 {
            tracing::debug!(
                consumed = expected,
                "hybrid forward at position 0: starting a new sequence (implicit reset)"
            );
            model.reset();
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
    /// Hybrid: `HybridModel::forward`, after the position contract.
    pub(crate) fn forward_logits(&mut self, token: u32, pos: usize) -> RuntimeResult<Vec<f32>> {
        if pos == 0 && !self.model.is_hybrid() {
            self.sequence_id = self.sequence_id.wrapping_add(1);
        }
        self.prepare_hybrid_position(pos)?;
        let row = match &mut self.model {
            LoadedModel::Dense(model) => model.forward(token, pos, &self.kernel)?,
            LoadedModel::Hybrid(model) => model.forward_alloc(token, pos)?,
        };
        // Test-only scripted generation (`crate::engine::ScriptedLogits`).
        #[cfg(test)]
        let row = self.scripted_row(row, false)?;
        Ok(row)
    }

    /// Single-token forward at `pos` on an explicit dispatcher — the greedy
    /// path's coherent CPU-fallback replay. A hybrid model carries its
    /// dispatcher inside its layers, so `kernel` only reaches the dense arm.
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
        match &mut self.model {
            LoadedModel::Dense(model) => Ok(model.forward(token, pos, kernel)?),
            LoadedModel::Hybrid(model) => Ok(model.forward_alloc(token, pos)?),
        }
    }

    /// Batched prefill of `tokens` from `pos_start`, returning the **last**
    /// token's `[vocab]` logit row.
    ///
    /// Dense: exactly `BonsaiModel::forward_prefill(tokens, pos_start,
    /// &self.kernel)`. Hybrid: `HybridModel::forward_prefill`, which runs the
    /// prompt in `prefill_chunk`-token chunks (design §3.10), after the
    /// position contract.
    pub(crate) fn prefill_logits(
        &mut self,
        tokens: &[u32],
        pos_start: usize,
    ) -> RuntimeResult<Vec<f32>> {
        if pos_start == 0 && !self.model.is_hybrid() {
            self.sequence_id = self.sequence_id.wrapping_add(1);
        }
        self.prepare_hybrid_position(pos_start)?;
        match &mut self.model {
            LoadedModel::Dense(model) => {
                Ok(model.forward_prefill(tokens, pos_start, &self.kernel)?)
            }
            LoadedModel::Hybrid(model) => {
                if tokens.is_empty() {
                    return Err(RuntimeError::Model(ModelError::MissingTensor {
                        name: "forward_prefill: empty token_ids".into(),
                    }));
                }
                let mut logits = vec![0.0f32; model.config().base.vocab_size];
                model.forward_prefill(tokens, pos_start, &mut logits)?;
                Ok(logits)
            }
        }
    }

    // ── Sequence snapshots ──────────────────────────────────────────────

    /// Take an exact rollback point at the current position (see the module
    /// docs).
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
        let recurrent = self
            .hybrid_model()
            .map(|model| model.recurrent().snapshot());
        Ok(SequenceSnapshot {
            position: self.sequence_position(),
            sequence_id: self.sequence_id,
            recurrent,
        })
    }

    /// Restore a [`SequenceSnapshot`] taken from this engine's **current**
    /// sequence: the KV cursor moves back to the snapshot's position (the
    /// entries below it were never overwritten) and a hybrid model's
    /// recurrent state is restored from the snapshot's deep copy.
    ///
    /// # Errors
    ///
    /// [`EngineError::SnapshotMismatch`] when the snapshot belongs to another
    /// sequence (any reset since it was taken), to the other model kind, or
    /// lies beyond the current position.
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
        match (&mut self.model, snapshot.recurrent.as_ref()) {
            (LoadedModel::Dense(model), None) => {
                model.kv_cache_mut().truncate(snapshot.position);
                Ok(())
            }
            (LoadedModel::Hybrid(model), Some(recurrent)) => {
                model.recurrent_mut().restore(recurrent)?;
                model.kv_cache_mut().set_seq_len(snapshot.position);
                Ok(())
            }
            (LoadedModel::Dense(_), Some(_)) => Err(EngineError::SnapshotMismatch {
                detail: "it carries a recurrent state but this engine holds a dense model".into(),
            }
            .into()),
            (LoadedModel::Hybrid(_), None) => Err(EngineError::SnapshotMismatch {
                detail: "it carries no recurrent state but this engine holds a hybrid model".into(),
            }
            .into()),
        }
    }
}

#[cfg(test)]
#[path = "engine_seam_tests.rs"]
mod tests;
