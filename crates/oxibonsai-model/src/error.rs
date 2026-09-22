//! Error types for the model crate.

use thiserror::Error;

/// Result type alias for model operations.
pub type ModelResult<T> = Result<T, ModelError>;

/// Errors that can occur during model construction and forward pass.
#[derive(Error, Debug)]
pub enum ModelError {
    /// A required tensor was not found during model loading.
    #[error("missing tensor: {name}")]
    MissingTensor { name: String },

    /// Tensor shape doesn't match expected dimensions.
    #[error("shape mismatch for '{name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Sequence length exceeds model's maximum context length.
    #[error("sequence length {seq_len} exceeds max context {max_ctx}")]
    SequenceTooLong { seq_len: usize, max_ctx: usize },

    /// A token position exceeds the bound of a precomputed per-position table
    /// (M-27, e.g. [`crate::layers::rope::RopeTable`]'s cos/sin rows).
    ///
    /// Deliberately **not** [`ModelError::SequenceTooLong`]: that variant is a
    /// recoverable "shorten the prompt" condition reported to the caller,
    /// whereas reaching a table bound means the model wired a table and a
    /// cache with inconsistent `max_seq_len` values — an internal invariant
    /// break that no amount of prompt trimming by the caller can fix.
    #[error("position {pos} out of range: max {max}")]
    PositionOutOfRange { pos: usize, max: usize },

    /// A fused GPU path maintained its **own** device-resident KV cache for
    /// this sequence, so the host [`KvCache`](crate::kv_cache::KvCache) holds
    /// no history for positions `0..pos` and a CPU block path must not attend
    /// over it (MET-05).
    ///
    /// The cure is to replay the committed prefix through a
    /// `KernelTier::Reference` dispatcher and call
    /// [`BonsaiModel::mark_host_kv_rebuilt`](crate::model::BonsaiModel::mark_host_kv_rebuilt)
    /// — exactly what `InferenceEngine::greedy_decode_token_with_fallback`
    /// already does for greedy decode. Construct and match it through
    /// [`gpu_fallback_requires_cache_rebuild`](crate::model::gpu_fallback_requires_cache_rebuild)
    /// / [`gpu_fallback_cache_rebuild_pos`](crate::model::gpu_fallback_cache_rebuild_pos)
    /// rather than naming the variant at a call site, so the runtime seam
    /// stays one function wide.
    #[error("GPU fallback at position {pos} requires a KV-cache rebuild")]
    GpuFallbackRequiresCacheRebuild {
        /// The position the CPU path was asked to run at.
        pos: usize,
    },

    /// A shape invariant that a bad configuration or a corrupted checkpoint
    /// could otherwise violate *silently* (M-12).
    ///
    /// Distinct from [`ModelError::ShapeMismatch`], which reports a tensor
    /// whose own dimensions disagree with what was asked for. This variant
    /// reports a **relationship** that must hold between several numbers —
    /// `num_attention_heads % num_kv_heads == 0`, `attn_q`'s output width
    /// against `head_dim * num_heads`, a layer's `(head_dim, num_kv_heads)`
    /// against the single KV-cache stride every layer shares — where the
    /// failure mode is not a bounds error but a cross-layer or cross-head
    /// aliasing read that returns plausible numbers.
    #[error("shape invariant violated for '{tensor}': expected {expected}, got {actual}")]
    ShapeInvariant {
        /// What was validated: a tensor name, or `layer <i>: <invariant>`.
        tensor: String,
        /// The value or relationship that was required.
        expected: String,
        /// What was actually found.
        actual: String,
    },

    /// A RoPE-scaling strategy (YaRN / linear / dynamic-NTK / LongRoPE) could
    /// not be turned into a frequency table (M-08).
    ///
    /// Raised at model load when `<arch>.rope.scaling.*` declares parameters
    /// this build cannot honour (e.g. a `factor < 1.0`, or an odd `head_dim`).
    /// Never silently downgraded to an unscaled table on a path that can
    /// report it: a model that asks for scaling and does not get it produces
    /// wrong output at every position, which must not look like success.
    #[error("rope scaling: {0}")]
    RopeScaling(#[from] crate::layers::rope_scaling::RopeScalingError),

    /// A tensor's *contents* are inconsistent with the type it is declared
    /// as, in a way no length or alignment check can catch (CQ-14).
    ///
    /// The motivating case: `PQ2_0` and the legacy `TQ2_0_g128` layout are
    /// both stored under ggml id 42 and are byte-compatible, but `PQ2_0`
    /// decodes the two-bit code `0b11` as `+2` while the `TQ2_0` family
    /// decodes it as `0`. A `PQ2_0` tensor mis-declared as `TQ2_0_g128` loads,
    /// aligns and runs — and silently zeroes every `+2` weight. The message
    /// carries the offending tensor's name.
    #[error("invalid tensor: {0}")]
    InvalidTensor(String),

    /// A Hadamard-folded `ssm_out` whose fold was computed in **tiled**
    /// v-head order on a model with more v-heads than k-heads (B2-10,
    /// design §3.3).
    ///
    /// `prism.hadamard.gdn_v_grouped = false` says the fold of
    /// `blk.N.ssm_out.weight` was computed with its columns in the GGUF's
    /// own tiled v-head order; a column permutation cannot be re-applied to
    /// an already-folded matrix, so the Gated-DeltaNet state would have to
    /// stay tiled while every other v-indexed activation is grouped. That
    /// combination has no reference implementation to validate against —
    /// PrismML's own `runtime.py` refuses it too — so it is refused here
    /// rather than executed with untested index math that would produce
    /// plausible-looking but wrong logits.
    ///
    /// Only reachable when `n_v_heads != n_k_heads`: with one v-head per
    /// k-head the two orders coincide and the flag is immaterial.
    #[error(
        "prism.hadamard.gdn_v_grouped = false with {n_v_heads} v-heads over {n_k_heads} k-heads: \
         a tiled-order fold of ssm_out is not supported (grouped folds only; \
         v_per_k must be 1 for an ungrouped fold)"
    )]
    UngroupedFoldedGdnOutput {
        /// `qwen35.ssm.group_count` — Gated-DeltaNet k-heads.
        n_k_heads: usize,
        /// `qwen35.ssm.time_step_rank` — Gated-DeltaNet v-heads.
        n_v_heads: usize,
    },

    /// A caller asked to roll a recurrent (Gated-DeltaNet) state back to an
    /// earlier position (M-05 / design §8.4 item 2).
    ///
    /// A KV cache rolls back by moving a cursor: every stored key/value is
    /// still the same function of the same token. A recurrence has no such
    /// property — `S` after `n` tokens is not recoverable from `S` after
    /// `n + k` tokens, so the only truthful implementations are "replay the
    /// prefix" or "restore a checkpoint taken at that position" (see
    /// [`crate::hybrid::RecurrentCache::snapshot`]). Silently leaving `S`
    /// untouched, which a cursor-style `truncate` would do, contaminates
    /// every subsequent token with state from tokens the caller believes it
    /// discarded.
    ///
    /// The checkpoint **ring** that would make an arbitrary rollback cheap
    /// is deferred (design §8.4 item 2); a single rollback point is already
    /// available through `snapshot`/`restore`.
    #[error(
        "cannot roll a recurrent state back to position {pos} ({tokens} tokens consumed): a \
         Gated-DeltaNet recurrence has no positional masking — use \
         RecurrentCache::snapshot/restore around the rollback window, or reset and replay"
    )]
    RecurrentRollbackUnsupported {
        /// The position the caller asked to roll back to.
        pos: usize,
        /// Tokens consumed into this state since the last reset.
        tokens: usize,
    },

    /// A hybrid (`qwen35`) GGUF has no `output.weight` (design §3.5).
    ///
    /// Every shipped Bonsai 2 file carries an explicit, Hadamard-folded LM
    /// head. Falling back to the tied token embedding would be wrong twice
    /// over for a folded model — the embedding is stored in the *rotated*
    /// basis and needs the inverse transform on lookup, and nothing says the
    /// two matrices are equal in the first place — so a tied head is refused
    /// rather than guessed at, exactly as PrismML's `runtime.py` refuses it.
    #[error(
        "hybrid model has no '{lm_head}': a tied LM head (reusing '{embedding}') is not \
         supported for qwen35 — the embedding is stored in the rotated basis"
    )]
    TiedLmHeadUnsupported {
        /// The LM-head tensor that is missing (`output.weight`).
        lm_head: String,
        /// The embedding that would have been reused (`token_embd.weight`).
        embedding: String,
    },

    /// Underlying core error.
    #[error("core: {0}")]
    Core(#[from] oxibonsai_core::error::BonsaiError),

    /// Underlying kernel error.
    #[error("kernel: {0}")]
    Kernel(#[from] oxibonsai_kernels::error::KernelError),

    /// Internal error (e.g. poisoned mutex).
    #[error("internal: {0}")]
    Internal(String),
}

impl ModelError {
    /// Return a short, stable error code string for monitoring and alerting.
    pub fn error_code(&self) -> &str {
        match self {
            Self::MissingTensor { .. } => "MISSING_TENSOR",
            Self::ShapeMismatch { .. } => "SHAPE_MISMATCH",
            Self::SequenceTooLong { .. } => "SEQUENCE_TOO_LONG",
            Self::PositionOutOfRange { .. } => "POSITION_OUT_OF_RANGE",
            Self::GpuFallbackRequiresCacheRebuild { .. } => "GPU_FALLBACK_REQUIRES_CACHE_REBUILD",
            Self::ShapeInvariant { .. } => "SHAPE_INVARIANT",
            Self::RopeScaling(_) => "ROPE_SCALING",
            Self::InvalidTensor(_) => "INVALID_TENSOR",
            Self::UngroupedFoldedGdnOutput { .. } => "UNGROUPED_FOLDED_GDN_OUTPUT",
            Self::RecurrentRollbackUnsupported { .. } => "RECURRENT_ROLLBACK_UNSUPPORTED",
            Self::TiedLmHeadUnsupported { .. } => "TIED_LM_HEAD_UNSUPPORTED",
            Self::Core(_) => "CORE_ERROR",
            Self::Kernel(_) => "KERNEL_ERROR",
            Self::Internal(_) => "INTERNAL_ERROR",
        }
    }
}
