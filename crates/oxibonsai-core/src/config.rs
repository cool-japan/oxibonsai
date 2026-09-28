//! Qwen3 model configuration extracted from GGUF metadata.
//!
//! The Bonsai-8B model uses the Qwen3-8B architecture. Configuration
//! values are read from GGUF metadata keys.
//!
//! # Required vs. defaulted fields (core-gguf-06 / M-03)
//!
//! Every structural field (layer/head counts, hidden size, vocab size, ...)
//! is genuinely **required**: [`Qwen3Config::from_metadata`] returns
//! [`BonsaiError::MissingConfigKey`]/[`BonsaiError::UnsupportedArchitecture`]
//! rather than silently substituting a Bonsai-8B number when a key is
//! absent. The hard-coded 8B/4B/1.7B numbers live **only** in the named
//! constructors ([`Qwen3Config::bonsai_8b`] and friends) — never as a
//! fallback inside `from_metadata`. This matters because a config built
//! from a defaulted `vocab_size` can silently size a softmax too small to
//! ever emit the real end-of-sequence token (verified on the PrismML
//! Bonsai 2 27B: metadata `vocab_size` is absent, the wrong 8B default is
//! 151936, but EOS is 248046 and the true vocabulary is 248320).
//!
//! Two entry points exist:
//!
//! - [`Qwen3Config::from_metadata`] — metadata only. Sufficient for every
//!   real GGUF this crate has been validated against: `vocab_size` falls
//!   back to the `tokenizer.ggml.tokens` array length, which every such
//!   file carries.
//! - [`Qwen3Config::from_metadata_and_tensors`] — metadata **and** the
//!   parsed [`TensorStore`], for callers that have both. `vocab_size`
//!   additionally consults `token_embd.weight`'s trailing shape dimension
//!   and, when both sources are available, hard-errors on disagreement
//!   instead of silently preferring one (a mismatch means the file itself
//!   is inconsistent).
//!
//! Legacy `llm.*` keys (an OxiBonsai converter artifact predating correct
//! `<general.architecture>.*` prefixing, still present in the shipped
//! `Ternary-Bonsai-{1.7B,8B}.gguf` files) remain a read-only fallback via
//! [`arch_key`] — see [`is_supported_architecture`] for the architecture
//! allowlist this crate can run inference for.

use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::metadata::MetadataStore;
use crate::gguf::tensor_info::{keys, tensor_names, TensorStore};

/// Architectures this build has a forward pass for.
///
/// `"qwen3"` is the dense Qwen3 architecture (Bonsai-{1.7B,4B,8B});
/// `"qwen35"` is the Qwen3.5 hybrid (full-attention + Gated-DeltaNet)
/// architecture used by the PrismML Bonsai 2 27B family — see
/// [`crate::config_hybrid::HybridConfig`].
pub const SUPPORTED_ARCHITECTURES: &[&str] = &["qwen3", "qwen35"];

/// Whether `arch` (a `general.architecture` value) is one this build can run
/// inference for.
///
/// Exposed as a small, stable public helper so callers such as the CLI's
/// `validate`/`info` commands can reject an unsupported architecture (e.g. a
/// `clip` vision-tower GGUF being reported as a dense Qwen3 model) without
/// duplicating this allowlist (cli-02).
pub fn is_supported_architecture(arch: &str) -> bool {
    SUPPORTED_ARCHITECTURES.contains(&arch)
}

/// Legacy architecture-independent key prefix.
///
/// An early OxiBonsai converter bug (core-gguf-02) wrote hyperparameters
/// under this generic prefix instead of `<general.architecture>.*`, as the
/// GGUF specification requires. The shipped `Ternary-Bonsai-1.7B.gguf` and
/// `Ternary-Bonsai-8B.gguf` files still use it. New code must never *write*
/// this namespace — only tolerate it on load, via [`arch_key`]'s two-step
/// chain.
const LEGACY_ARCH: &str = "llm";

/// Build the architecture-scoped GGUF metadata key `"<arch>.<suffix>"`.
///
/// Every structural hyperparameter is looked up first under this key, then
/// (only on a miss) under `arch_key(LEGACY_ARCH, suffix)` — the read-only
/// legacy fallback described on [`LEGACY_ARCH`]. Public because
/// [`crate::config_hybrid`] and [`crate::hadamard_config`] need the same
/// convention for their own `<arch>.*` keys.
pub fn arch_key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

/// Read a required `u32` hyperparameter, trying `<arch>.<suffix>` and then
/// the legacy `llm.<suffix>` key, before erroring on the architecture-scoped
/// key name (M-03: keep the two-step chain, make only its end fatal).
fn required_u32(metadata: &MetadataStore, arch: &str, suffix: &str) -> BonsaiResult<u32> {
    metadata
        .get_u32(&arch_key(arch, suffix))
        .or_else(|_| metadata.get_u32(&arch_key(LEGACY_ARCH, suffix)))
        .map_err(|_| BonsaiError::MissingConfigKey {
            key: arch_key(arch, suffix),
        })
}

/// Read a required `f32` hyperparameter with the same two-step chain as
/// [`required_u32`].
fn required_f32(metadata: &MetadataStore, arch: &str, suffix: &str) -> BonsaiResult<f32> {
    metadata
        .get_f32(&arch_key(arch, suffix))
        .or_else(|_| metadata.get_f32(&arch_key(LEGACY_ARCH, suffix)))
        .map_err(|_| BonsaiError::MissingConfigKey {
            key: arch_key(arch, suffix),
        })
}

/// Read an optional `u32` hyperparameter, trying `<arch>.<suffix>` then the
/// legacy `llm.<suffix>` key, returning `None` when neither resolves.
fn optional_u32(metadata: &MetadataStore, arch: &str, suffix: &str) -> Option<u32> {
    metadata
        .get_u32(&arch_key(arch, suffix))
        .or_else(|_| metadata.get_u32(&arch_key(LEGACY_ARCH, suffix)))
        .ok()
}

/// The GGUF key carrying the sliding-window attention span (M-17), read as
/// `<arch>.attention.sliding_window` with the usual legacy `llm.*` fallback.
const SLIDING_WINDOW_SUFFIX: &str = "attention.sliding_window";

/// Parse the optional `<arch>.attention.sliding_window` key (M-17).
///
/// Deliberately *not* routed through [`optional_u32`]: that helper collapses
/// "absent" and "present but unreadable" into the same `None`, which for this
/// key would turn a corrupt or out-of-range declaration into silent
/// fully-causal attention. Here an absent key is `Ok(None)` and a present one
/// must be a usable unsigned width or the file is rejected.
///
/// # Errors
///
/// [`BonsaiError::InvalidMetadata`] naming the `<arch>`-prefixed key when the
/// value is present but is not an unsigned integer, is zero, or does not fit
/// this platform's `usize`.
fn parse_sliding_window(metadata: &MetadataStore, arch: &str) -> BonsaiResult<Option<usize>> {
    let key = arch_key(arch, SLIDING_WINDOW_SUFFIX);
    let Some(value) = metadata
        .get(&key)
        .or_else(|| metadata.get(&arch_key(LEGACY_ARCH, SLIDING_WINDOW_SUFFIX)))
    else {
        return Ok(None);
    };
    let invalid = |reason: &str| BonsaiError::InvalidMetadata {
        key: key.clone(),
        reason: reason.to_string(),
    };
    let raw = value
        .as_u64()
        .ok_or_else(|| invalid("must be an unsigned integer number of key positions"))?;
    if raw == 0 {
        return Err(invalid(
            "must be >= 1 when present: a zero-width attention window admits no key positions \
             at all. Omit the key entirely for fully-causal attention.",
        ));
    }
    usize::try_from(raw)
        .map(Some)
        .map_err(|_| invalid("does not fit in this platform's usize"))
}

/// RoPE long-context scaling method declared in `<arch>.rope.scaling.*`
/// GGUF metadata (M-08).
///
/// Parsed here so it is visible to `oxibonsai info`/`validate` and survives
/// the GGUF round trip instead of being silently discarded — a shipped
/// model (`models/Bonsai-8B.gguf`) declares
/// `qwen3.rope.scaling.{type=yarn,factor=4.0,original_context_length=16384}`
/// today and the pre-fix loader dropped it entirely, producing wrong output
/// at every position (YaRN rescales `inv_freq` unconditionally, not only
/// past the original context). *Applying* this inside `RopeTable` /
/// `YarnFreqTable` is a separate package's job (MODEL-ROPE-CFG); this type
/// only carries the declaration through correctly.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RopeScaling {
    /// No `<arch>.rope.scaling.type` key is present: RoPE runs unscaled.
    #[default]
    None,
    /// YaRN long-context extrapolation (`rope.scaling.type == "yarn"`).
    Yarn {
        /// Context-length multiplier (`models/Bonsai-8B.gguf`: `4.0`).
        factor: f32,
        /// Pretraining context length YaRN extrapolates from.
        original_context_length: u32,
        /// Attention-temperature multiplier (`mscale`); `None` means "use
        /// the applying implementation's own default", commonly derived
        /// from `factor` when this key is absent.
        attn_factor: Option<f32>,
        /// Ramp-function low corrective-dimension boundary.
        beta_fast: Option<f32>,
        /// Ramp-function high corrective-dimension boundary.
        beta_slow: Option<f32>,
    },
    /// A declared scaling type this build does not implement, preserved
    /// verbatim (with whatever `factor`/`original_context_length` are
    /// present) rather than silently discarded.
    Other {
        scaling_type: String,
        factor: Option<f32>,
        original_context_length: Option<u32>,
    },
}

impl RopeScaling {
    /// Parse `<arch>.rope.scaling.*` from `metadata`.
    ///
    /// Returns `Ok(RopeScaling::None)` only when the `...rope.scaling.type`
    /// key is genuinely absent. A key that is present but malformed (wrong
    /// GGUF value type, or a `"yarn"` type missing `factor` /
    /// `original_context_length`) is a hard error — never a silent
    /// fall-through to `None`, since the file explicitly asked for scaling
    /// and this build failing to read it correctly must not look identical
    /// to "no scaling requested".
    pub fn from_metadata(metadata: &MetadataStore, arch: &str) -> BonsaiResult<Self> {
        let type_key = arch_key(arch, "rope.scaling.type");
        let Some(value) = metadata.get(&type_key) else {
            return Ok(RopeScaling::None);
        };
        let scaling_type = value.as_str().ok_or_else(|| BonsaiError::InvalidMetadata {
            key: type_key.clone(),
            reason: format!("expected a string, found {}", value.type_name()),
        })?;

        match scaling_type {
            "yarn" => {
                let factor = required_f32(metadata, arch, "rope.scaling.factor")?;
                let original_context_length =
                    required_u32(metadata, arch, "rope.scaling.original_context_length")?;
                let attn_factor = optional_f32(metadata, arch, "rope.scaling.attn_factor");
                let beta_fast = optional_f32(metadata, arch, "rope.scaling.beta_fast");
                let beta_slow = optional_f32(metadata, arch, "rope.scaling.beta_slow");
                Ok(RopeScaling::Yarn {
                    factor,
                    original_context_length,
                    attn_factor,
                    beta_fast,
                    beta_slow,
                })
            }
            other => {
                let factor = optional_f32(metadata, arch, "rope.scaling.factor");
                let original_context_length =
                    optional_u32(metadata, arch, "rope.scaling.original_context_length");
                Ok(RopeScaling::Other {
                    scaling_type: other.to_string(),
                    factor,
                    original_context_length,
                })
            }
        }
    }
}

// ─── `--rope-scaling auto|on|off` (wave-4b ruling R2, `RULING_bonsai8b_yarn.md`) ──

/// A caller's override of the RoPE scaling a GGUF declares
/// (`--rope-scaling auto|on|off`).
///
/// `Bonsai-8B.gguf` declares YaRN (factor 4, original context 16384);
/// honouring it (M-08) is what llama.cpp and the PrismML fork do, but it
/// changes that model's greedy text at every context length versus
/// OxiBonsai <= 0.2.4, so `Off` exists to reproduce the old, unscaled
/// behaviour on request, and `On` to assert that a file really declares
/// scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RopeScalingOverride {
    /// Honour whatever `<arch>.rope.scaling.*` the file declares. The
    /// default, and byte-identical to having no override at all.
    #[default]
    Auto,
    /// Force plain (unscaled) RoPE regardless of the declaration.
    Off,
    /// Require the file to declare a scaling strategy.
    On,
}

impl RopeScalingOverride {
    /// Stable lower-case name (`"auto"`, `"off"`, `"on"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::On => "on",
        }
    }

    /// Apply this override to the scaling a file of architecture `arch`
    /// declares.
    ///
    /// # Errors
    ///
    /// [`BonsaiError::InvalidMetadata`] naming `<arch>.rope.scaling.type`
    /// for [`RopeScalingOverride::On`] on a file that declares no scaling.
    pub fn apply(self, declared: RopeScaling, arch: &str) -> BonsaiResult<RopeScaling> {
        match self {
            Self::Auto => Ok(declared),
            Self::Off => Ok(RopeScaling::None),
            Self::On => match declared {
                RopeScaling::None => Err(BonsaiError::InvalidMetadata {
                    key: arch_key(arch, "rope.scaling.type"),
                    reason: format!(
                        "RoPE scaling was required (`--rope-scaling on`) but this file \
                         declares no {arch}.rope.scaling.* metadata"
                    ),
                }),
                other => Ok(other),
            },
        }
    }
}

impl std::fmt::Display for RopeScalingOverride {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

thread_local! {
    /// The override [`Qwen3Config::from_metadata`] applies on this thread;
    /// [`RopeScalingOverride::Auto`] (a no-op) unless a
    /// [`RopeScalingOverrideScope`] is alive on the same thread.
    static ROPE_SCALING_OVERRIDE: std::cell::Cell<RopeScalingOverride> =
        const { std::cell::Cell::new(RopeScalingOverride::Auto) };
}

/// RAII scope that makes every [`Qwen3Config::from_metadata`] /
/// [`Qwen3Config::from_metadata_and_tensors`] call **on the current thread**
/// apply `mode` to the file's declared RoPE scaling, until the guard drops.
///
/// This is how the override reaches model constructors that build their own
/// `Qwen3Config` from the GGUF internally (`BonsaiModel::from_gguf*`,
/// `HybridModel::from_gguf*`) without changing any of their signatures —
/// the same scoped-knob pattern as the kernels crate's
/// `CpuOnlyBackendScope`. The engine's additive constructor
/// (`InferenceEngine::from_gguf_with_backend_and_rope`) enters it around
/// model construction. Thread-local and `!Send` (so it can never be dropped
/// on a thread other than the one it set), and restores the previous value
/// on drop, so scopes nest and never leak into unrelated loads.
#[must_use = "the override only lasts while the scope guard is alive"]
#[derive(Debug)]
pub struct RopeScalingOverrideScope {
    previous: RopeScalingOverride,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl RopeScalingOverrideScope {
    /// Install `mode` for this thread until the returned guard drops.
    pub fn enter(mode: RopeScalingOverride) -> Self {
        let previous = ROPE_SCALING_OVERRIDE.with(|cell| cell.replace(mode));
        Self {
            previous,
            _not_send: std::marker::PhantomData,
        }
    }

    /// The override currently in force on this thread.
    #[must_use]
    pub fn active() -> RopeScalingOverride {
        ROPE_SCALING_OVERRIDE.with(std::cell::Cell::get)
    }
}

impl Drop for RopeScalingOverrideScope {
    fn drop(&mut self) {
        let previous = self.previous;
        ROPE_SCALING_OVERRIDE.with(|cell| cell.set(previous));
    }
}

/// Read an optional `f32` hyperparameter, trying `<arch>.<suffix>` then the
/// legacy `llm.<suffix>` key, returning `None` when neither resolves.
fn optional_f32(metadata: &MetadataStore, arch: &str, suffix: &str) -> Option<f32> {
    metadata
        .get_f32(&arch_key(arch, suffix))
        .or_else(|_| metadata.get_f32(&arch_key(LEGACY_ARCH, suffix)))
        .ok()
}

/// Resolve `vocab_size` with the fallback order verified against the real
/// PrismML Bonsai 2 27B header (core-gguf-06):
///
/// 1. `<arch>.vocab_size` (or the legacy `llm.vocab_size`);
/// 2. `token_embd.weight`'s trailing shape dimension, from `tensors` (only
///    when a [`TensorStore`] is supplied);
/// 3. `tokenizer.ggml.tokens`'s array length.
///
/// When both (2) and (3) are available and disagree, this is a hard error
/// (the file is internally inconsistent) rather than silently preferring
/// one source. When neither (1), (2) nor (3) resolves, this is
/// [`BonsaiError::MissingConfigKey`].
fn resolve_vocab_size(
    metadata: &MetadataStore,
    arch: &str,
    tensors: Option<&TensorStore>,
) -> BonsaiResult<usize> {
    if let Some(v) = optional_u32(metadata, arch, "vocab_size") {
        return Ok(v as usize);
    }

    let tensor_vocab = tensors
        .and_then(|t| t.get(tensor_names::TOKEN_EMBD))
        .and_then(|info| info.shape.last().copied());

    let token_list_vocab = metadata
        .get_string_array(keys::TOKENIZER_TOKENS)
        .ok()
        .map(|tokens| tokens.len() as u64);

    match (tensor_vocab, token_list_vocab) {
        (Some(t), Some(n)) if t != n => Err(BonsaiError::InvalidMetadata {
            key: keys::TOKENIZER_TOKENS.to_string(),
            reason: format!(
                "{}'s array length ({n}) disagrees with '{}' tensor rows ({t}); the file is \
                 internally inconsistent",
                keys::TOKENIZER_TOKENS,
                tensor_names::TOKEN_EMBD,
            ),
        }),
        (Some(t), _) => Ok(t as usize),
        (None, Some(n)) => Ok(n as usize),
        (None, None) => Err(BonsaiError::MissingConfigKey {
            key: arch_key(arch, "vocab_size"),
        }),
    }
}

/// Qwen3-8B model configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    /// Hidden size (embedding dimension). Default: 4096.
    pub hidden_size: usize,
    /// Intermediate size for SwiGLU MLP. Default: 14336.
    pub intermediate_size: usize,
    /// Number of Transformer layers. Default: 36.
    pub num_layers: usize,
    /// Number of attention heads (query). Default: 32.
    pub num_attention_heads: usize,
    /// Number of key-value heads (GQA). Default: 8.
    pub num_kv_heads: usize,
    /// Key head dimension (`attention.key_length`). Default: 128.
    pub head_dim: usize,
    /// Value head dimension (`attention.value_length`, M-30).
    ///
    /// Read **separately** from [`head_dim`](Self::head_dim): the two are
    /// equal in every model this crate has been validated against (128 for
    /// the 8B, 256 for the Bonsai 2 27B), but only by coincidence — nothing
    /// in the GGUF format guarantees it, and `qwen35`'s Gated-DeltaNet
    /// layers already have an asymmetric `nk=16`/`nv=48` head count. Falls
    /// back to [`head_dim`](Self::head_dim) when the key is absent, so the
    /// dense forward path (which assumes `head_dim` for both) stays
    /// bit-identical for every existing model.
    pub value_length: usize,
    /// Vocabulary size. Default: 151936.
    pub vocab_size: usize,
    /// Maximum context length. Default: 65536.
    pub max_context_length: usize,
    /// RMSNorm epsilon. Default: 1e-6.
    pub rms_norm_eps: f32,
    /// RoPE frequency base. Default: 1000000.0.
    pub rope_freq_base: f32,
    /// RoPE long-context scaling method (M-08). `RopeScaling::None` when the
    /// GGUF declares none.
    pub rope_scaling: RopeScaling,
    /// Sliding-window attention span, from `<arch>.attention.sliding_window`
    /// (M-17).
    ///
    /// `Some(w)` means every query position attends only to the `w` most
    /// recent key positions (itself included) — the local-attention window
    /// llama.cpp calls `n_swa`, as declared by Mistral-, Gemma- and
    /// Qwen2-family GGUFs. `None` means the key is absent and attention is
    /// fully causal over the whole context, which is the case for **every**
    /// model this crate ships against: neither `Bonsai-8B.gguf`,
    /// `Ternary-Bonsai-{1.7B,8B}.gguf` nor any of the six PrismML Bonsai 2
    /// 27B headers declares the key (verified against the real files), so
    /// every named constructor below carries `None` and the fully-causal
    /// path stays bit-identical.
    ///
    /// A **present-but-zero** value is rejected at parse time rather than
    /// folded into `None`: a zero window is not "no window", it is a window
    /// that admits no keys at all, and silently promoting it to full
    /// attention would turn a corrupt header into plausible-looking output.
    ///
    /// Consumed by `oxibonsai_model`'s `BonsaiModel::forward`, which routes
    /// every block through `TransformerBlock::forward_with_sliding_window`
    /// when this is `Some`.
    pub sliding_window: Option<usize>,
    /// Architecture name from GGUF metadata.
    pub architecture: String,
    /// Model name from GGUF metadata.
    pub model_name: String,
}

impl Qwen3Config {
    /// Extract configuration from GGUF metadata alone.
    ///
    /// Every structural field is required: an architecture outside
    /// [`SUPPORTED_ARCHITECTURES`] is [`BonsaiError::UnsupportedArchitecture`]
    /// and a missing hyperparameter (after the `<arch>.*` → legacy `llm.*`
    /// fallback chain) is [`BonsaiError::MissingConfigKey`] — this function
    /// never substitutes a hard-coded default the way the pre-fix version
    /// did.
    ///
    /// Prefer [`Qwen3Config::from_metadata_and_tensors`] when a
    /// [`TensorStore`] is available: it gives `vocab_size` a second,
    /// independent source (`token_embd.weight`'s shape) to cross-check
    /// against `tokenizer.ggml.tokens`. This entry point still resolves
    /// `vocab_size` correctly for every real GGUF validated so far, via the
    /// tokenizer token-list length alone.
    pub fn from_metadata(metadata: &MetadataStore) -> BonsaiResult<Self> {
        Self::build(metadata, None)
    }

    /// Extract configuration from GGUF metadata **and** the parsed tensor
    /// directory.
    ///
    /// Identical to [`Qwen3Config::from_metadata`] except that `vocab_size`
    /// additionally consults `tensors`'s `token_embd.weight` shape (see
    /// [`resolve_vocab_size`]) and hard-errors if it disagrees with
    /// `tokenizer.ggml.tokens`'s length instead of silently picking one.
    pub fn from_metadata_and_tensors(
        metadata: &MetadataStore,
        tensors: &TensorStore,
    ) -> BonsaiResult<Self> {
        Self::build(metadata, Some(tensors))
    }

    /// [`Qwen3Config::from_metadata`] with an explicit RoPE-scaling override
    /// (`--rope-scaling`), independent of any [`RopeScalingOverrideScope`]
    /// on this thread.
    ///
    /// # Errors
    ///
    /// As [`Qwen3Config::from_metadata`], plus
    /// [`BonsaiError::InvalidMetadata`] for [`RopeScalingOverride::On`] on a
    /// file that declares no scaling.
    pub fn from_metadata_with_rope_override(
        metadata: &MetadataStore,
        mode: RopeScalingOverride,
    ) -> BonsaiResult<Self> {
        Self::build_with(metadata, None, mode)
    }

    fn build(metadata: &MetadataStore, tensors: Option<&TensorStore>) -> BonsaiResult<Self> {
        Self::build_with(metadata, tensors, RopeScalingOverrideScope::active())
    }

    fn build_with(
        metadata: &MetadataStore,
        tensors: Option<&TensorStore>,
        rope_override: RopeScalingOverride,
    ) -> BonsaiResult<Self> {
        let architecture = metadata.get_string(keys::GENERAL_ARCHITECTURE)?.to_string();
        if !is_supported_architecture(&architecture) {
            return Err(BonsaiError::UnsupportedArchitecture { arch: architecture });
        }
        let arch = architecture.as_str();

        // Cosmetic only (never used for a structural decision): fall back to
        // the architecture name itself rather than inventing a specific
        // product name like the pre-fix "Bonsai-8B" default, which mislabels
        // any file that isn't actually that model (cli-02).
        let model_name = metadata
            .get_string(keys::GENERAL_NAME)
            .unwrap_or(arch)
            .to_string();

        let hidden_size = required_u32(metadata, arch, "embedding_length")? as usize;
        let num_layers = required_u32(metadata, arch, "block_count")? as usize;
        let num_attention_heads = required_u32(metadata, arch, "attention.head_count")? as usize;
        if num_attention_heads == 0 {
            return Err(BonsaiError::InvalidMetadata {
                key: arch_key(arch, "attention.head_count"),
                reason: "must be nonzero".to_string(),
            });
        }
        let num_kv_heads = required_u32(metadata, arch, "attention.head_count_kv")? as usize;
        let intermediate_size = required_u32(metadata, arch, "feed_forward_length")? as usize;
        let vocab_size = resolve_vocab_size(metadata, arch, tensors)?;
        let max_context_length = required_u32(metadata, arch, "context_length")? as usize;
        let rms_norm_eps = required_f32(metadata, arch, "attention.layer_norm_rms_epsilon")?;
        let rope_freq_base = required_f32(metadata, arch, "rope.freq_base")?;

        // head_dim/value_length are read explicitly from metadata when
        // present (some Qwen3 sizes, e.g. Qwen3-4B, have `head_dim !=
        // hidden_size / num_attention_heads`), falling back to the derived
        // value only when the key is genuinely absent — never to a
        // hard-coded constant.
        let head_dim = optional_u32(metadata, arch, "attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);
        let value_length = optional_u32(metadata, arch, "attention.value_length")
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        // `--rope-scaling` (wave-4b ruling R2): `Auto` returns the declared
        // value unchanged, so the default path is byte-identical.
        let rope_scaling =
            rope_override.apply(RopeScaling::from_metadata(metadata, arch)?, arch)?;

        let sliding_window = parse_sliding_window(metadata, arch)?;

        let config = Qwen3Config {
            hidden_size,
            intermediate_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            value_length,
            vocab_size,
            max_context_length,
            rms_norm_eps,
            rope_freq_base,
            rope_scaling,
            sliding_window,
            architecture,
            model_name,
        };
        // M-24/M-12 consolidation: every numeric invariant `ModelConfigBuilder::build`
        // (oxibonsai-model) already enforced is now shared, so a corrupt GGUF
        // declaring e.g. `qwen3.block_count = 0` is rejected here instead of
        // silently producing a `Qwen3Config` that only fails later, deep
        // inside a KV-cache allocation or a GQA head split.
        config.validate()?;
        Ok(config)
    }

    /// Validate the numeric invariants every [`Qwen3Config`] must satisfy,
    /// regardless of how it was built: from GGUF metadata
    /// ([`Self::from_metadata`] / [`Self::from_metadata_and_tensors`], which
    /// call this automatically) or via `oxibonsai_model`'s
    /// `ModelConfigBuilder::build` (which calls this explicitly, since it
    /// lives in a different crate and cannot run automatically the way
    /// `from_metadata` does).
    ///
    /// # What this deliberately does NOT check
    ///
    /// "`hidden_size` divisible by `num_attention_heads`" is NOT checked
    /// here. That constraint only makes sense while `head_dim` is still a
    /// *derived* value the caller has not overridden yet — an unmerged
    /// `Option`, which a fully-constructed `Qwen3Config` no longer
    /// expresses (by the time a `Qwen3Config` exists, `head_dim` is already
    /// whichever of the derived-or-explicit values won). `ModelConfigBuilder::build`
    /// keeps that one check as its own pre-check on the unmerged `Option`,
    /// before it ever constructs a `Qwen3Config` to hand to this function;
    /// every other constructor here (the named presets, `from_metadata`'s
    /// own explicit-or-derived `head_dim` resolution above) is exempt from
    /// it by construction rather than needing a runtime check.
    ///
    /// # Errors
    ///
    /// [`BonsaiError::InvalidMetadata`] naming the first offending field
    /// (checked in this order): `num_layers`, `hidden_size`,
    /// `num_attention_heads`, `num_kv_heads`, `head_dim`,
    /// `num_attention_heads % num_kv_heads == 0` (Grouped Query Attention),
    /// `intermediate_size`, `vocab_size >= 2`, `max_context_length`,
    /// `sliding_window` (M-17: `Some(0)` is rejected — a hand-built config
    /// must not be able to smuggle in the zero-width window
    /// [`from_metadata`](Self::from_metadata) already refuses),
    /// `rope_freq_base` (finite, positive), `rms_norm_eps` (finite,
    /// positive). Its `key` is the `Qwen3Config` field name, not a GGUF
    /// metadata key — this runs just as much for a config nothing built
    /// from metadata (a bare `ModelConfigBuilder`) as for one that did.
    pub fn validate(&self) -> BonsaiResult<()> {
        fn invalid(field: &str, reason: impl Into<String>) -> BonsaiError {
            BonsaiError::InvalidMetadata {
                key: field.to_string(),
                reason: reason.into(),
            }
        }
        fn finite_positive(field: &str, value: f32) -> BonsaiResult<()> {
            if value <= 0.0 || value.is_nan() || value.is_infinite() {
                return Err(invalid(
                    field,
                    format!("must be a finite positive number, got {value}"),
                ));
            }
            Ok(())
        }

        if self.num_layers == 0 {
            return Err(invalid("num_layers", "must be >= 1"));
        }
        if self.hidden_size == 0 {
            return Err(invalid("hidden_size", "must be >= 1"));
        }
        if self.num_attention_heads == 0 {
            return Err(invalid("num_attention_heads", "must be >= 1"));
        }
        if self.num_kv_heads == 0 {
            return Err(invalid("num_kv_heads", "must be >= 1"));
        }
        if self.head_dim == 0 {
            return Err(invalid("head_dim", "must be >= 1"));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_kv_heads) {
            return Err(invalid(
                "num_attention_heads",
                format!(
                    "({}) must be divisible by num_kv_heads ({}) for Grouped Query Attention",
                    self.num_attention_heads, self.num_kv_heads
                ),
            ));
        }
        if self.intermediate_size == 0 {
            return Err(invalid("intermediate_size", "must be >= 1"));
        }
        if self.vocab_size < 2 {
            return Err(invalid("vocab_size", "must be >= 2"));
        }
        if self.max_context_length == 0 {
            return Err(invalid("max_context_length", "must be >= 1"));
        }
        if self.sliding_window == Some(0) {
            return Err(invalid(
                "sliding_window",
                "must be >= 1 when present: a zero-width attention window admits no key \
                 positions at all. Use `None` for fully-causal attention.",
            ));
        }
        finite_positive("rope_freq_base", self.rope_freq_base)?;
        finite_positive("rms_norm_eps", self.rms_norm_eps)?;
        Ok(())
    }

    /// Create a tiny configuration suitable for fast unit tests.
    ///
    /// Uses minimal dimensions so that model forward passes complete
    /// in milliseconds instead of tens of seconds.
    pub fn tiny_test() -> Self {
        Qwen3Config {
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            value_length: 16,
            vocab_size: 151936, // must match real vocab for token IDs
            max_context_length: 512,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10_000.0,
            rope_scaling: RopeScaling::None,
            sliding_window: None,
            architecture: "qwen3".to_string(),
            model_name: "Bonsai-Tiny-Test".to_string(),
        }
    }

    /// Create a Bonsai-4B configuration.
    ///
    /// 24 layers, hidden=2560, intermediate=6912, heads=20, kv_heads=4.
    pub fn bonsai_4b() -> Self {
        Qwen3Config {
            hidden_size: 2560,
            intermediate_size: 6912,
            num_layers: 24,
            num_attention_heads: 20,
            num_kv_heads: 4,
            head_dim: 128,
            value_length: 128,
            vocab_size: 151936,
            max_context_length: 65536,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            rope_scaling: RopeScaling::None,
            sliding_window: None,
            architecture: "qwen3".to_string(),
            model_name: "Bonsai-4B".to_string(),
        }
    }

    /// Create a Bonsai-1.7B configuration.
    ///
    /// Corrected (M-34) against the real `models/Ternary-Bonsai-1.7B.gguf`
    /// header: `block_count 28, embedding_length 2048, feed_forward_length
    /// 6144, head_count 16, head_count_kv 8, vocab 151669,
    /// llm.context_length 32768`. The previous hard-coded
    /// 16/1536/4096/12/2/151936/65536 numbers matched no shipped file.
    pub fn bonsai_1_7b() -> Self {
        Qwen3Config {
            hidden_size: 2048,
            intermediate_size: 6144,
            num_layers: 28,
            num_attention_heads: 16,
            num_kv_heads: 8,
            head_dim: 128,
            value_length: 128,
            vocab_size: 151_669,
            max_context_length: 32768,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            rope_scaling: RopeScaling::None,
            sliding_window: None,
            architecture: "qwen3".to_string(),
            model_name: "Bonsai-1.7B".to_string(),
        }
    }

    /// Create a default Qwen3-8B / Bonsai-8B configuration.
    ///
    /// Corrected (M-34) against the real `models/Bonsai-8B.gguf` header:
    /// `feed_forward_length 12288` (was 14336) and `vocab 151669` (was
    /// 151936); `block_count 36`/`embedding_length 4096`/head counts/
    /// `head_dim 128`/`context_length 65536` already matched. The real file
    /// also declares `qwen3.rope.scaling.{type=yarn,factor=4.0,
    /// original_context_length=16384}` (M-08), carried here instead of
    /// silently dropped.
    pub fn bonsai_8b() -> Self {
        Qwen3Config {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            value_length: 128,
            vocab_size: 151_669,
            max_context_length: 65536,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            rope_scaling: RopeScaling::Yarn {
                factor: 4.0,
                original_context_length: 16384,
                attn_factor: None,
                beta_fast: None,
                beta_slow: None,
            },
            sliding_window: None,
            architecture: "qwen3".to_string(),
            model_name: "Bonsai-8B".to_string(),
        }
    }

    /// Create a Ternary-Bonsai-8B configuration.
    ///
    /// Same architecture as Bonsai-8B but with ternary ({-1,0,+1}) weights.
    ///
    /// Unlike [`Qwen3Config::bonsai_8b`], this does **not** inherit YaRN
    /// scaling (M-34 real-file correction): the real
    /// `models/Ternary-Bonsai-8B.gguf` header carries only 9 legacy `llm.*`
    /// hyperparameter keys (verified by parsing the file's KV block
    /// directly) and no `rope.scaling.*` key at all — the ternary converter
    /// dropped the upstream YaRN declaration when it wrote this file. Since
    /// [`RopeScaling::from_metadata`] correctly returns `RopeScaling::None`
    /// for that file, this named constructor must mirror it exactly rather
    /// than copying `bonsai_8b()`'s `Yarn` variant: MODEL-ROPE-CFG would
    /// otherwise apply a long-context rescale the shipped file never
    /// requested, producing wrong output at every position.
    pub fn ternary_bonsai_8b() -> Self {
        let mut cfg = Self::bonsai_8b();
        cfg.model_name = "Ternary-Bonsai-8B".to_string();
        cfg.rope_scaling = RopeScaling::None;
        cfg
    }

    /// Create a Ternary-Bonsai-4B configuration.
    pub fn ternary_bonsai_4b() -> Self {
        let mut cfg = Self::bonsai_4b();
        cfg.model_name = "Ternary-Bonsai-4B".to_string();
        cfg
    }

    /// Create a Ternary-Bonsai-1.7B configuration.
    pub fn ternary_bonsai_1_7b() -> Self {
        let mut cfg = Self::bonsai_1_7b();
        cfg.model_name = "Ternary-Bonsai-1.7B".to_string();
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::reader::GgufFile;
    use crate::gguf::writer::{GgufWriter, MetadataWriteValue};

    /// Build a [`MetadataStore`] from `(key, value)` pairs via the public
    /// writer + reader round trip (no direct field access — `MetadataStore`
    /// intentionally has none), mirroring the pattern used throughout this
    /// workspace's own integration tests (e.g. `ternary_integration.rs`).
    fn build_metadata_store(pairs: Vec<(&str, MetadataWriteValue)>) -> MetadataStore {
        let mut writer = GgufWriter::new();
        for (key, value) in pairs {
            writer.add_metadata(key, value);
        }
        let bytes = writer
            .to_bytes()
            .expect("synthetic metadata should serialise");
        GgufFile::parse(&bytes)
            .expect("synthetic metadata should parse back")
            .metadata
    }

    /// A complete, valid dense-Qwen3 metadata set (mirrors every fixture
    /// builder across the workspace, e.g. `ternary_integration.rs`).
    fn full_qwen3_pairs() -> Vec<(&'static str, MetadataWriteValue)> {
        vec![
            (
                "general.architecture",
                MetadataWriteValue::Str("qwen3".to_string()),
            ),
            (
                "general.name",
                MetadataWriteValue::Str("UnitTestModel".to_string()),
            ),
            ("qwen3.embedding_length", MetadataWriteValue::U32(128)),
            ("qwen3.block_count", MetadataWriteValue::U32(2)),
            ("qwen3.attention.head_count", MetadataWriteValue::U32(4)),
            ("qwen3.attention.head_count_kv", MetadataWriteValue::U32(2)),
            ("qwen3.feed_forward_length", MetadataWriteValue::U32(256)),
            ("qwen3.vocab_size", MetadataWriteValue::U32(32)),
            ("qwen3.context_length", MetadataWriteValue::U32(512)),
            (
                "qwen3.attention.layer_norm_rms_epsilon",
                MetadataWriteValue::F32(1e-6),
            ),
            ("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0)),
        ]
    }

    #[test]
    fn default_bonsai_8b_config() {
        let config = Qwen3Config::bonsai_8b();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 12288);
        assert_eq!(config.num_layers, 36);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.value_length, 128);
        assert_eq!(config.vocab_size, 151_669);
        assert_eq!(config.max_context_length, 65536);
        assert_eq!(
            config.rope_scaling,
            RopeScaling::Yarn {
                factor: 4.0,
                original_context_length: 16384,
                attn_factor: None,
                beta_fast: None,
                beta_slow: None,
            }
        );
    }

    #[test]
    fn bonsai_4b_config() {
        let config = Qwen3Config::bonsai_4b();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.intermediate_size, 6912);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_attention_heads, 20);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.value_length, 128);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.rope_scaling, RopeScaling::None);
    }

    #[test]
    fn bonsai_1_7b_config() {
        let config = Qwen3Config::bonsai_1_7b();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.value_length, 128);
        assert_eq!(config.vocab_size, 151_669);
        assert_eq!(config.max_context_length, 32768);
    }

    #[test]
    fn ternary_bonsai_8b_matches_spec() {
        let cfg = Qwen3Config::ternary_bonsai_8b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 12288);
        assert_eq!(cfg.num_layers, 36);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 151_669);
        assert_eq!(cfg.max_context_length, 65536);
        assert_eq!(cfg.model_name, "Ternary-Bonsai-8B");
        assert_eq!(cfg.architecture, "qwen3");
    }

    /// M-34 regression guard: the real `models/Ternary-Bonsai-8B.gguf`
    /// header has no `rope.scaling.*` key at all (verified by parsing the
    /// file directly), unlike `Bonsai-8B.gguf`'s YaRN declaration. Locks in
    /// the fix for the blocking failure this test package shipped:
    /// `ternary_bonsai_8b()` must NOT inherit `bonsai_8b()`'s
    /// `RopeScaling::Yarn`.
    #[test]
    fn ternary_bonsai_8b_has_no_rope_scaling_unlike_bonsai_8b() {
        assert_eq!(
            Qwen3Config::ternary_bonsai_8b().rope_scaling,
            RopeScaling::None
        );
        assert_ne!(
            Qwen3Config::bonsai_8b().rope_scaling,
            Qwen3Config::ternary_bonsai_8b().rope_scaling
        );
    }

    #[test]
    fn ternary_bonsai_name_distinct() {
        assert_ne!(
            Qwen3Config::bonsai_8b().model_name,
            Qwen3Config::ternary_bonsai_8b().model_name
        );
        assert_ne!(
            Qwen3Config::bonsai_4b().model_name,
            Qwen3Config::ternary_bonsai_4b().model_name
        );
        assert_ne!(
            Qwen3Config::bonsai_1_7b().model_name,
            Qwen3Config::ternary_bonsai_1_7b().model_name
        );
    }

    #[test]
    fn ternary_bonsai_4b_matches_spec() {
        let cfg = Qwen3Config::ternary_bonsai_4b();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.model_name, "Ternary-Bonsai-4B");
    }

    #[test]
    fn ternary_bonsai_1_7b_matches_spec() {
        let cfg = Qwen3Config::ternary_bonsai_1_7b();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.model_name, "Ternary-Bonsai-1.7B");
    }

    // ── from_metadata: required-key behaviour (core-gguf-06 / M-03) ───────

    /// Replaces `from_empty_metadata_uses_defaults`, which enshrined the
    /// defect this package fixes: empty metadata must be a hard error, not
    /// a silently-defaulted Bonsai-8B config.
    #[test]
    fn from_empty_metadata_is_a_hard_error() {
        let metadata = MetadataStore::new();
        let err = Qwen3Config::from_metadata(&metadata)
            .expect_err("empty metadata must not silently default");
        match err {
            BonsaiError::MissingConfigKey { key } => {
                assert_eq!(key, "general.architecture");
            }
            other => panic!("expected MissingConfigKey, got {other:?}"),
        }
    }

    #[test]
    fn full_valid_metadata_round_trips_exactly() {
        let metadata = build_metadata_store(full_qwen3_pairs());
        let config = Qwen3Config::from_metadata(&metadata).expect("full metadata should parse");
        assert_eq!(config.hidden_size, 128);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.intermediate_size, 256);
        assert_eq!(config.vocab_size, 32);
        assert_eq!(config.max_context_length, 512);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_freq_base, 10_000.0);
        assert_eq!(config.head_dim, 32); // derived: 128 / 4
        assert_eq!(config.value_length, 32); // falls back to head_dim
        assert_eq!(config.architecture, "qwen3");
        assert_eq!(config.model_name, "UnitTestModel");
        assert_eq!(config.rope_scaling, RopeScaling::None);
    }

    #[test]
    fn unsupported_architecture_is_rejected() {
        let metadata = build_metadata_store(vec![(
            "general.architecture",
            MetadataWriteValue::Str("clip".to_string()),
        )]);
        let err = Qwen3Config::from_metadata(&metadata)
            .expect_err("a clip vision-tower file must never be reported as Qwen3");
        match err {
            BonsaiError::UnsupportedArchitecture { arch } => assert_eq!(arch, "clip"),
            other => panic!("expected UnsupportedArchitecture, got {other:?}"),
        }
    }

    #[test]
    fn is_supported_architecture_allows_and_rejects() {
        assert!(is_supported_architecture("qwen3"));
        assert!(is_supported_architecture("qwen35"));
        assert!(!is_supported_architecture("clip"));
        assert!(!is_supported_architecture("llama"));
        assert!(!is_supported_architecture(""));
    }

    #[test]
    fn missing_required_field_after_architecture_is_hard_error() {
        let mut pairs = full_qwen3_pairs();
        pairs.retain(|(k, _)| *k != "qwen3.block_count");
        let metadata = build_metadata_store(pairs);
        let err = Qwen3Config::from_metadata(&metadata)
            .expect_err("a missing block_count must be a hard error");
        match err {
            BonsaiError::MissingConfigKey { key } => assert_eq!(key, "qwen3.block_count"),
            other => panic!("expected MissingConfigKey, got {other:?}"),
        }
    }

    /// M-03 correction: the two shipped ternary files
    /// (`Ternary-Bonsai-{1.7B,8B}.gguf`) carry every hyperparameter under
    /// the generic legacy `llm.*` prefix (an OxiBonsai converter bug,
    /// core-gguf-02), not `qwen3.*`. The two-step chain must keep loading
    /// them.
    #[test]
    fn legacy_llm_prefix_still_works() {
        let metadata = build_metadata_store(vec![
            (
                "general.architecture",
                MetadataWriteValue::Str("qwen3".to_string()),
            ),
            ("llm.embedding_length", MetadataWriteValue::U32(2048)),
            ("llm.block_count", MetadataWriteValue::U32(28)),
            ("llm.attention.head_count", MetadataWriteValue::U32(16)),
            ("llm.attention.head_count_kv", MetadataWriteValue::U32(8)),
            ("llm.feed_forward_length", MetadataWriteValue::U32(6144)),
            ("llm.vocab_size", MetadataWriteValue::U32(151_669)),
            ("llm.context_length", MetadataWriteValue::U32(32768)),
            (
                "llm.attention.layer_norm_rms_epsilon",
                MetadataWriteValue::F32(1e-6),
            ),
            ("llm.rope.freq_base", MetadataWriteValue::F32(1_000_000.0)),
        ]);
        let config =
            Qwen3Config::from_metadata(&metadata).expect("legacy llm.* metadata should parse");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.vocab_size, 151_669);
        assert_eq!(config.max_context_length, 32768);
    }

    // ── vocab_size fallback chain (core-gguf-06) ───────────────────────────

    #[test]
    fn vocab_size_falls_back_to_tokenizer_tokens_length_without_a_tensor_store() {
        let mut pairs = full_qwen3_pairs();
        pairs.retain(|(k, _)| *k != "qwen3.vocab_size");
        // A small array proves the fallback path (`from_metadata` reads the
        // array's *length*, never its content); the real 27B file's array
        // has 248320 entries, but building that many strings would only
        // slow the unit test down without exercising different code.
        pairs.push((
            "tokenizer.ggml.tokens",
            MetadataWriteValue::ArrayStr(vec!["a".into(), "b".into(), "c".into()]),
        ));
        let metadata = build_metadata_store(pairs);
        let config = Qwen3Config::from_metadata(&metadata)
            .expect("vocab_size should resolve from the tokenizer token list");
        assert_eq!(config.vocab_size, 3);
    }

    #[test]
    fn vocab_size_missing_everywhere_is_hard_error() {
        let mut pairs = full_qwen3_pairs();
        pairs.retain(|(k, _)| *k != "qwen3.vocab_size");
        let metadata = build_metadata_store(pairs);
        let err = Qwen3Config::from_metadata(&metadata)
            .expect_err("vocab_size must be required when no fallback resolves it");
        match err {
            BonsaiError::MissingConfigKey { key } => assert_eq!(key, "qwen3.vocab_size"),
            other => panic!("expected MissingConfigKey, got {other:?}"),
        }
    }

    #[test]
    fn from_metadata_and_tensors_resolves_vocab_from_tensor_shape() {
        let mut pairs = full_qwen3_pairs();
        pairs.retain(|(k, _)| *k != "qwen3.vocab_size");
        let metadata = build_metadata_store(pairs);

        let mut writer = GgufWriter::new();
        writer.add_tensor(crate::gguf::writer::TensorEntry {
            name: tensor_names::TOKEN_EMBD.to_string(),
            shape: vec![128, 500],
            tensor_type: crate::gguf::writer::TensorType::F32,
            data: vec![0u8; 128 * 500 * 4],
        });
        let bytes = writer
            .to_bytes()
            .expect("tensor-only file should serialise");
        let file = GgufFile::parse(&bytes).expect("tensor-only file should parse");

        let config = Qwen3Config::from_metadata_and_tensors(&metadata, &file.tensors)
            .expect("vocab_size should resolve from token_embd.weight's shape");
        assert_eq!(config.vocab_size, 500);
    }

    #[test]
    fn from_metadata_and_tensors_hard_errors_on_vocab_disagreement() {
        let mut pairs = full_qwen3_pairs();
        pairs.retain(|(k, _)| *k != "qwen3.vocab_size");
        pairs.push((
            "tokenizer.ggml.tokens",
            MetadataWriteValue::ArrayStr(vec!["a".into(), "b".into()]), // len 2
        ));
        let metadata = build_metadata_store(pairs);

        let mut writer = GgufWriter::new();
        writer.add_tensor(crate::gguf::writer::TensorEntry {
            name: tensor_names::TOKEN_EMBD.to_string(),
            shape: vec![128, 3], // disagrees: 3 rows vs. 2 tokens
            tensor_type: crate::gguf::writer::TensorType::F32,
            data: vec![0u8; 128 * 3 * 4],
        });
        let bytes = writer
            .to_bytes()
            .expect("tensor-only file should serialise");
        let file = GgufFile::parse(&bytes).expect("tensor-only file should parse");

        let err = Qwen3Config::from_metadata_and_tensors(&metadata, &file.tensors)
            .expect_err("disagreeing vocab sources must hard-error, not silently pick one");
        assert!(
            matches!(err, BonsaiError::InvalidMetadata { .. }),
            "{err:?}"
        );
    }

    // ── head_dim / value_length (M-30) ──────────────────────────────────────

    #[test]
    fn value_length_defaults_to_head_dim_when_absent() {
        let metadata = build_metadata_store(full_qwen3_pairs());
        let config = Qwen3Config::from_metadata(&metadata).expect("should parse");
        assert_eq!(config.value_length, config.head_dim);
    }

    #[test]
    fn value_length_read_independently_of_head_dim_when_present() {
        let mut pairs = full_qwen3_pairs();
        pairs.push(("qwen3.attention.key_length", MetadataWriteValue::U32(40)));
        pairs.push(("qwen3.attention.value_length", MetadataWriteValue::U32(64)));
        let metadata = build_metadata_store(pairs);
        let config = Qwen3Config::from_metadata(&metadata).expect("should parse");
        assert_eq!(config.head_dim, 40);
        assert_eq!(config.value_length, 64);
        assert_ne!(config.head_dim, config.value_length);
    }

    // ── RopeScalingOverride (`--rope-scaling`, wave-4b ruling R2) ──────────

    /// `full_qwen3_pairs()` plus the exact YaRN declaration
    /// `models/Bonsai-8B.gguf` carries.
    fn yarn_qwen3_pairs() -> Vec<(&'static str, MetadataWriteValue)> {
        let mut pairs = full_qwen3_pairs();
        pairs.push((
            "qwen3.rope.scaling.type",
            MetadataWriteValue::Str("yarn".to_string()),
        ));
        pairs.push(("qwen3.rope.scaling.factor", MetadataWriteValue::F32(4.0)));
        pairs.push((
            "qwen3.rope.scaling.original_context_length",
            MetadataWriteValue::U32(16384),
        ));
        pairs
    }

    fn bonsai_8b_yarn() -> RopeScaling {
        RopeScaling::Yarn {
            factor: 4.0,
            original_context_length: 16384,
            attn_factor: None,
            beta_fast: None,
            beta_slow: None,
        }
    }

    #[test]
    fn rope_override_auto_honours_the_declared_yarn() {
        let metadata = build_metadata_store(yarn_qwen3_pairs());
        let config =
            Qwen3Config::from_metadata_with_rope_override(&metadata, RopeScalingOverride::Auto)
                .expect("parse");
        assert_eq!(config.rope_scaling, bonsai_8b_yarn());
    }

    #[test]
    fn rope_override_off_forces_plain_rope_on_a_yarn_file() {
        let metadata = build_metadata_store(yarn_qwen3_pairs());
        let config =
            Qwen3Config::from_metadata_with_rope_override(&metadata, RopeScalingOverride::Off)
                .expect("parse");
        assert_eq!(config.rope_scaling, RopeScaling::None);
        // Every other field is exactly what `auto` resolves.
        let auto = Qwen3Config::from_metadata(&metadata).expect("parse");
        assert_eq!(
            Qwen3Config {
                rope_scaling: RopeScaling::None,
                ..auto
            },
            config
        );
    }

    #[test]
    fn rope_override_on_accepts_a_yarn_file_and_refuses_an_unscaled_one() {
        let yarn = build_metadata_store(yarn_qwen3_pairs());
        let config = Qwen3Config::from_metadata_with_rope_override(&yarn, RopeScalingOverride::On)
            .expect("a declared scaling satisfies `on`");
        assert_eq!(config.rope_scaling, bonsai_8b_yarn());

        let plain = build_metadata_store(full_qwen3_pairs());
        let err = Qwen3Config::from_metadata_with_rope_override(&plain, RopeScalingOverride::On)
            .expect_err("`on` on an unscaled file must fail");
        let msg = err.to_string();
        assert!(msg.contains("qwen3.rope.scaling.type"), "{msg}");
        assert!(msg.contains("--rope-scaling on"), "{msg}");
        assert!(
            msg.contains("declares no qwen3.rope.scaling.* metadata"),
            "the reason names the real architecture, never a placeholder: {msg}"
        );
    }

    #[test]
    fn rope_override_scope_applies_to_plain_from_metadata_and_restores() {
        let metadata = build_metadata_store(yarn_qwen3_pairs());
        assert_eq!(
            RopeScalingOverrideScope::active(),
            RopeScalingOverride::Auto
        );
        {
            let _off = RopeScalingOverrideScope::enter(RopeScalingOverride::Off);
            assert_eq!(
                Qwen3Config::from_metadata(&metadata)
                    .expect("parse")
                    .rope_scaling,
                RopeScaling::None,
                "the scope must reach callers that only use from_metadata"
            );
            {
                let _auto = RopeScalingOverrideScope::enter(RopeScalingOverride::Auto);
                assert_eq!(
                    Qwen3Config::from_metadata(&metadata)
                        .expect("parse")
                        .rope_scaling,
                    bonsai_8b_yarn(),
                    "an inner scope wins"
                );
            }
            assert_eq!(
                RopeScalingOverrideScope::active(),
                RopeScalingOverride::Off,
                "dropping the inner scope restores the outer one"
            );
        }
        assert_eq!(
            RopeScalingOverrideScope::active(),
            RopeScalingOverride::Auto
        );
        assert_eq!(
            Qwen3Config::from_metadata(&metadata)
                .expect("parse")
                .rope_scaling,
            bonsai_8b_yarn(),
            "no scope = the declaration, unchanged"
        );
    }

    #[test]
    fn rope_override_scope_is_thread_local() {
        let _off = RopeScalingOverrideScope::enter(RopeScalingOverride::Off);
        let seen_elsewhere = std::thread::spawn(RopeScalingOverrideScope::active)
            .join()
            .expect("join");
        assert_eq!(seen_elsewhere, RopeScalingOverride::Auto);
        assert_eq!(RopeScalingOverrideScope::active(), RopeScalingOverride::Off);
    }

    // ── RopeScaling (M-08) ──────────────────────────────────────────────────

    #[test]
    fn rope_scaling_none_when_absent() {
        let metadata = build_metadata_store(full_qwen3_pairs());
        let config = Qwen3Config::from_metadata(&metadata).expect("should parse");
        assert_eq!(config.rope_scaling, RopeScaling::None);
    }

    /// Mirrors the real `models/Bonsai-8B.gguf` declaration exactly
    /// (M-08's verified evidence).
    #[test]
    fn rope_scaling_yarn_parses_the_real_bonsai_8b_declaration() {
        let mut pairs = full_qwen3_pairs();
        pairs.push((
            "qwen3.rope.scaling.type",
            MetadataWriteValue::Str("yarn".to_string()),
        ));
        pairs.push(("qwen3.rope.scaling.factor", MetadataWriteValue::F32(4.0)));
        pairs.push((
            "qwen3.rope.scaling.original_context_length",
            MetadataWriteValue::U32(16384),
        ));
        let metadata = build_metadata_store(pairs);
        let config = Qwen3Config::from_metadata(&metadata).expect("should parse");
        assert_eq!(
            config.rope_scaling,
            RopeScaling::Yarn {
                factor: 4.0,
                original_context_length: 16384,
                attn_factor: None,
                beta_fast: None,
                beta_slow: None,
            }
        );
    }

    #[test]
    fn rope_scaling_yarn_missing_factor_is_hard_error() {
        let mut pairs = full_qwen3_pairs();
        pairs.push((
            "qwen3.rope.scaling.type",
            MetadataWriteValue::Str("yarn".to_string()),
        ));
        pairs.push((
            "qwen3.rope.scaling.original_context_length",
            MetadataWriteValue::U32(16384),
        ));
        let metadata = build_metadata_store(pairs);
        let err = Qwen3Config::from_metadata(&metadata)
            .expect_err("a yarn declaration missing factor must not silently become None");
        assert!(
            matches!(err, BonsaiError::MissingConfigKey { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rope_scaling_type_wrong_gguf_type_is_hard_error() {
        let mut pairs = full_qwen3_pairs();
        // Stored as an integer instead of a string.
        pairs.push(("qwen3.rope.scaling.type", MetadataWriteValue::U32(1)));
        let metadata = build_metadata_store(pairs);
        let err = Qwen3Config::from_metadata(&metadata)
            .expect_err("a malformed rope.scaling.type must not silently degrade to None");
        assert!(
            matches!(err, BonsaiError::InvalidMetadata { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rope_scaling_unknown_type_is_preserved_not_discarded() {
        let mut pairs = full_qwen3_pairs();
        pairs.push((
            "qwen3.rope.scaling.type",
            MetadataWriteValue::Str("linear".to_string()),
        ));
        pairs.push(("qwen3.rope.scaling.factor", MetadataWriteValue::F32(2.0)));
        let metadata = build_metadata_store(pairs);
        let config = Qwen3Config::from_metadata(&metadata).expect("should parse");
        match config.rope_scaling {
            RopeScaling::Other {
                scaling_type,
                factor,
                original_context_length,
            } => {
                assert_eq!(scaling_type, "linear");
                assert_eq!(factor, Some(2.0));
                assert_eq!(original_context_length, None);
            }
            other => panic!("expected RopeScaling::Other, got {other:?}"),
        }
    }

    // ── arch_key ────────────────────────────────────────────────────────────

    #[test]
    fn arch_key_formats_the_dot_joined_key() {
        assert_eq!(arch_key("qwen35", "block_count"), "qwen35.block_count");
        assert_eq!(arch_key("llm", "vocab_size"), "llm.vocab_size");
    }

    // ── M-17: `<arch>.attention.sliding_window` ─────────────────────────────

    #[test]
    fn sliding_window_absent_is_none() {
        let metadata = build_metadata_store(full_qwen3_pairs());
        let config = Qwen3Config::from_metadata(&metadata).expect("should parse");
        assert_eq!(
            config.sliding_window, None,
            "no shipped GGUF declares the key; absent must stay fully causal"
        );
    }

    #[test]
    fn sliding_window_u32_is_parsed() {
        let mut pairs = full_qwen3_pairs();
        pairs.push((
            "qwen3.attention.sliding_window",
            MetadataWriteValue::U32(4096),
        ));
        let metadata = build_metadata_store(pairs);
        let config = Qwen3Config::from_metadata(&metadata).expect("should parse");
        assert_eq!(config.sliding_window, Some(4096));
    }

    #[test]
    fn sliding_window_u64_is_parsed() {
        let mut pairs = full_qwen3_pairs();
        pairs.push((
            "qwen3.attention.sliding_window",
            MetadataWriteValue::U64(131_072),
        ));
        let metadata = build_metadata_store(pairs);
        let config = Qwen3Config::from_metadata(&metadata).expect("should parse");
        assert_eq!(config.sliding_window, Some(131_072));
    }

    #[test]
    fn sliding_window_legacy_llm_prefix_is_honoured() {
        let mut pairs = full_qwen3_pairs();
        pairs.push(("llm.attention.sliding_window", MetadataWriteValue::U32(512)));
        let metadata = build_metadata_store(pairs);
        let config = Qwen3Config::from_metadata(&metadata).expect("should parse");
        assert_eq!(config.sliding_window, Some(512));
    }

    #[test]
    fn sliding_window_zero_is_rejected_not_silently_none() {
        let mut pairs = full_qwen3_pairs();
        pairs.push(("qwen3.attention.sliding_window", MetadataWriteValue::U32(0)));
        let metadata = build_metadata_store(pairs);
        let err = Qwen3Config::from_metadata(&metadata)
            .expect_err("a zero-width window must not silently become fully-causal attention");
        match err {
            BonsaiError::InvalidMetadata { key, .. } => {
                assert_eq!(key, "qwen3.attention.sliding_window");
            }
            other => panic!("expected InvalidMetadata, got {other:?}"),
        }
    }

    #[test]
    fn sliding_window_non_integer_is_rejected() {
        let mut pairs = full_qwen3_pairs();
        pairs.push((
            "qwen3.attention.sliding_window",
            MetadataWriteValue::Str("4096".to_string()),
        ));
        let metadata = build_metadata_store(pairs);
        let err = Qwen3Config::from_metadata(&metadata)
            .expect_err("a non-integer window must be rejected, not ignored");
        assert!(
            matches!(err, BonsaiError::InvalidMetadata { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn validate_rejects_hand_built_zero_sliding_window() {
        let mut config = Qwen3Config::tiny_test();
        assert!(config.validate().is_ok());
        config.sliding_window = Some(0);
        match config.validate() {
            Err(BonsaiError::InvalidMetadata { key, .. }) => {
                assert_eq!(key, "sliding_window");
            }
            other => panic!("expected InvalidMetadata for sliding_window, got {other:?}"),
        }
        config.sliding_window = Some(1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn named_constructors_declare_no_sliding_window() {
        // Verified against the real headers: none of `Bonsai-8B.gguf`,
        // `Ternary-Bonsai-{1.7B,8B}.gguf` or the six Bonsai 2 27B GGUFs
        // carries `<arch>.attention.sliding_window`.
        for config in [
            Qwen3Config::tiny_test(),
            Qwen3Config::bonsai_4b(),
            Qwen3Config::bonsai_1_7b(),
            Qwen3Config::bonsai_8b(),
            Qwen3Config::ternary_bonsai_8b(),
            Qwen3Config::ternary_bonsai_4b(),
            Qwen3Config::ternary_bonsai_1_7b(),
        ] {
            assert_eq!(
                config.sliding_window, None,
                "{} must not declare a sliding window",
                config.model_name
            );
        }
    }
}
