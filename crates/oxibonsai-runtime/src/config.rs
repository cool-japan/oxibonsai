//! Layered configuration system for OxiBonsai.
//!
//! Loading order: defaults → TOML file → CLI argument overrides.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{RuntimeError, RuntimeResult};

/// Chat-template rendering types (B2-13's real Jinja subset), re-exported
/// here as part of the chat-contract configuration surface
/// (`--think`/`--no-think`, `--reasoning-effort`, `--tools`,
/// `[model].reasoning_effort`): a caller that depends only on
/// `oxibonsai-runtime` — the `oxibonsai` CLI in a default build, where
/// `oxibonsai-tokenizer` is not a direct dependency — can then render a
/// prompt through a model's own `tokenizer.chat_template`
/// ([`crate::TokenizerBridge::resolved_chat_template`]) with those options
/// applied, instead of being limited to raw-text encoding.
pub use oxibonsai_tokenizer::chat_templates::{RenderMessage, RenderOptions, ResolvedChatTemplate};

/// Top-level OxiBonsai configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OxiBonsaiConfig {
    /// Server configuration.
    pub server: ServerConfig,
    /// Sampling parameters.
    pub sampling: SamplingConfig,
    /// Model paths and limits.
    pub model: ModelConfig,
    /// Observability settings.
    pub observability: ObservabilityConfig,
    /// Text-to-image (Bonsai-Image) generation defaults — the `[imagen]`
    /// section. `oxibonsai image` / `oxibonsai repl` resolve their
    /// `--width`/`--height`/`--steps`/`--seed` flags against it (flag wins,
    /// then this section, then the CLI's own literal default), and
    /// `output_dir`/`model_path` fill in a relative `--out` and a missing
    /// `--dit` (B2-14).
    pub imagen: ImagenConfig,
}

/// HTTP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Bind host address.
    pub host: String,
    /// Bind port.
    pub port: u16,
}

/// Sampling parameters configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingConfig {
    /// Temperature for softmax scaling. 0.0 = greedy.
    pub temperature: f32,
    /// Top-k filtering (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) threshold (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
}

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Path to the GGUF model file.
    pub model_path: Option<String>,
    /// Path to tokenizer.json file.
    pub tokenizer_path: Option<String>,
    /// Maximum sequence length (prompt + generated).
    pub max_seq_len: usize,
    /// The loaded model's own architectural maximum context length in
    /// tokens (e.g. `262144` for Bonsai 2 27B), independent of the
    /// operator-configured [`ModelConfig::max_seq_len`] above. `None` until
    /// it is populated from the model's GGUF metadata
    /// (`<arch>.context_length`).
    ///
    /// This field only *carries* the value today; no forward/serving path
    /// reads it yet. **B2-12 implements the guard formula** —
    /// [`max_context_for_budget`] (Appendix A.3 exactly) and
    /// [`validate_requested_context`]/[`OxiBonsaiConfig::validate_context_budget`]
    /// (refuse a request naming both the model limit and the RAM-derived
    /// limit) — as pure, fully-tested functions; wiring a caller to fill
    /// this field from real GGUF metadata and to call them before serving a
    /// request is the CLI/model-load integration (B2-14 and B2-09/B2-11's
    /// territory, not owned by this package). Only the trivial non-zero
    /// sanity check below is enforced by [`OxiBonsaiConfig::validate`]
    /// itself.
    #[serde(default)]
    pub max_context: Option<usize>,
    /// A hard ceiling, in bytes, on the memory a single session's context
    /// window (KV cache, plus — for a hybrid architecture — recurrent
    /// linear-attention state) may occupy. `None` means "no explicit budget
    /// configured" — a caller should fall back to
    /// [`max_context_for_budget`] with an auto-detected
    /// [`total_ram_bytes`] in that case.
    ///
    /// See [`ModelConfig::max_context`] for how this and
    /// [`max_context_for_budget`] fit together; not read by any
    /// forward/serving path yet (same integration gap as `max_context`).
    #[serde(default)]
    pub ctx_budget_bytes: Option<u64>,
    /// RoPE long-context scaling override (wave-4b orchestrator addendum,
    /// `RULING_bonsai8b_yarn.md`): `auto` (default) honours the GGUF's own
    /// `<arch>.rope.scaling.*` metadata exactly; `off` forces plain RoPE;
    /// `on` requires the file to declare scaling. Applied at model load by
    /// [`crate::InferenceEngine::from_gguf_with_backend_and_rope`] (and the
    /// matching engine-pool builder), which scope
    /// [`oxibonsai_core::config::RopeScalingOverrideScope`] around the model
    /// constructor.
    #[serde(default)]
    pub rope_scaling: RopeScalingMode,
}

/// `--rope-scaling`/`[model].rope_scaling` control (wave-4b orchestrator
/// addendum, 2026-09-23; see `RULING_bonsai8b_yarn.md`).
///
/// Bonsai-8B's GGUF declares `qwen3.rope.scaling.{type=yarn,factor=4.0,
/// original_context_length=16384}`; OxiBonsai <= 0.2.4 ignored it (plain
/// RoPE at every position), while the wiring this session landed
/// (`oxibonsai_core::config::RopeScaling::from_metadata`, M-08) honours it
/// exactly like llama.cpp — which changes that model's decoded text at
/// every context length, not just past 16 384 tokens. This type exists so a
/// caller can opt back into the old behaviour (`Off`) or assert a model
/// declares scaling at all (`On`), instead of only ever getting `Auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RopeScalingMode {
    /// Honour the GGUF's own `<arch>.rope.scaling.*` metadata exactly, the
    /// same as llama.cpp. The shipped default.
    #[default]
    Auto,
    /// Force plain (unscaled) RoPE regardless of what the GGUF declares —
    /// reproduces OxiBonsai <= 0.2.4 behaviour on a model such as Bonsai-8B
    /// that declares YaRN.
    Off,
    /// Require the GGUF to declare a scaling strategy; an error (not a
    /// silent fallback to unscaled RoPE) when it does not.
    On,
}

impl std::fmt::Display for RopeScalingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::On => "on",
        })
    }
}

impl std::str::FromStr for RopeScalingMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            "on" => Ok(Self::On),
            other => Err(format!(
                "invalid --rope-scaling value '{other}': expected one of auto, on, off"
            )),
        }
    }
}

/// The serde-free core twin that the model constructors actually consult
/// (`oxibonsai-core` has no `serde` dependency, so the configuration-facing
/// type lives here and converts at the engine boundary).
impl From<RopeScalingMode> for oxibonsai_core::config::RopeScalingOverride {
    fn from(mode: RopeScalingMode) -> Self {
        match mode {
            RopeScalingMode::Auto => Self::Auto,
            RopeScalingMode::Off => Self::Off,
            RopeScalingMode::On => Self::On,
        }
    }
}

/// Text-to-image (Bonsai-Image / FLUX.2 Klein DiT) generation defaults.
///
/// CLI-CORE deviation (B2-12 addendum item 6): the config-file section is
/// the enabling half; reading it from `oxibonsai image` CLI flags is
/// B2-14's. `width`/`height` mirror `oxibonsai_image::pipeline`'s own
/// enforced bounds (multiples of 16, [`MIN_DIMENSION`, `MAX_DIMENSION`] —
/// not owned by this package, so checked here only informally by this doc,
/// not shared code). `steps`/`guidance_scale`'s defaults (4 / 3.5) are this
/// package's own choice, not a mirrored pipeline default — the pipeline
/// itself declares no default for either, only `validate_render_params`/
/// `validate_guidance` acceptance ranges (see [`OxiBonsaiConfig::validate`]'s
/// `guidance_scale` check, which matches `validate_guidance` exactly:
/// any finite value, including negative, is accepted).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImagenConfig {
    /// Path to the imagen model directory/checkpoint (DiT + VAE + text
    /// encoder weights). `None` means "not configured" — the CLI must
    /// require `--model`/a config value before running `oxibonsai image`.
    pub model_path: Option<String>,
    /// Default output image width in pixels. Must be a multiple of the
    /// DiT's patch size (16) for the pipeline to accept it without
    /// resizing; validated as non-zero and reasonably bounded here, with
    /// the patch-multiple check left to the pipeline itself, which already
    /// enforces it.
    pub width: u32,
    /// Default output image height in pixels. See [`ImagenConfig::width`].
    pub height: u32,
    /// Default number of denoising steps.
    pub steps: u32,
    /// Default classifier-free-guidance scale.
    pub guidance_scale: f32,
    /// Default RNG seed. `None` means "pick a fresh random seed per run";
    /// `Some(seed)` reproduces the same image for the same prompt (mirrors
    /// mflux/MLX-exact Threefry RNG behavior already implemented in the
    /// pipeline).
    pub seed: Option<u64>,
    /// Directory new images are written to when the CLI does not override
    /// it with an explicit `--out` path. `None` means "current directory".
    pub output_dir: Option<String>,
}

impl Default for ImagenConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            width: 1024,
            height: 1024,
            steps: 4,
            guidance_scale: 3.5,
            seed: None,
            output_dir: None,
        }
    }
}

/// Observability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ObservabilityConfig {
    /// Log level filter (e.g. "info", "debug", "warn").
    pub log_level: String,
    /// Whether to emit JSON-formatted logs.
    pub json_logs: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            // Gatekeeper REQUIRED #18 (waves 3+3.5 review): this used to be
            // `1.1`, one of the residual `repetition_penalty: 1.1` seeds
            // left over after `sampling.rs`'s own `SamplingParams::default()`
            // was corrected to `1.0` (RT-24 / gatekeeper REQUIRED #1(a)).
            // `1.0` (no-op) matches every other penalty default in this
            // struct and keeps `OxiBonsaiConfig::load`'s `#[serde(default)]`
            // path and `EngineBuilder::build` from silently reintroducing a
            // non-1.0 penalty this crate has otherwise standardised away.
            repetition_penalty: 1.0,
            max_tokens: 512,
        }
    }
}

impl SamplingConfig {
    /// Seed the defaults from a GGUF's own `general.sampling.*` metadata
    /// (design §5.5, RT-17): `temperature`/`top_p`/`top_k` take the model's
    /// declared values when present (Bonsai 2: 1.0 / 0.95 / 20) and fall
    /// back to [`SamplingConfig::default`] field by field otherwise. An
    /// explicit CLI flag, `--config` value or per-request API field is
    /// still applied on top of this by the caller — see
    /// [`crate::sampling::resolve_sampling_default_f32`].
    #[must_use]
    pub fn from_gguf_defaults(md: &oxibonsai_core::MetadataStore) -> Self {
        let declared = crate::sampling::GgufSamplingDefaults::from_metadata(md);
        let base = Self::default();
        Self {
            temperature: declared.temperature.unwrap_or(base.temperature),
            top_k: declared.top_k.unwrap_or(base.top_k),
            top_p: declared.top_p.unwrap_or(base.top_p),
            ..base
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            max_seq_len: 4096,
            max_context: None,
            ctx_budget_bytes: None,
            rope_scaling: RopeScalingMode::default(),
        }
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            json_logs: false,
        }
    }
}

/// Severity level for configuration warnings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningSeverity {
    /// Informational only.
    Info,
    /// May cause suboptimal behavior.
    Warning,
    /// Will likely cause failures.
    Error,
}

impl std::fmt::Display for WarningSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// A warning about a configuration value.
#[derive(Debug, Clone)]
pub struct ConfigWarning {
    /// Which configuration field this warning applies to.
    pub field: String,
    /// Human-readable warning message.
    pub message: String,
    /// Severity of this warning.
    pub severity: WarningSeverity,
}

impl std::fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.field, self.message)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// RAM-derived context guard (RT-29 / sec-11 / M-07, Appendix A.3)
// ─────────────────────────────────────────────────────────────────────────
//
// The KV cache is allocated eagerly at load time and never shrinks, so an
// operator-requested context length that does not fit in RAM alongside the
// model's weights is an out-of-memory kill, not a graceful error — unless
// something refuses it first. `max_context_for_budget` computes the largest
// context this host can actually afford; `validate_requested_context` (and
// `OxiBonsaiConfig::validate_context_budget`, above) refuse a request that
// exceeds it, naming both the model's own architectural limit and the
// RAM-derived one so an operator sees the whole picture at once.
//
// Reserves are named constants, not inline literals, so "tune OS_RESERVE
// once measured" (design doc Appendix A.3) is a one-line change:

/// Fixed memory set aside for forward-pass scratch buffers/activations,
/// independent of context length (Appendix A.3).
pub const ACTIVATION_RESERVE_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB

/// The OS/runtime reserve is `max(total_ram / OS_RESERVE_DIVISOR,
/// OS_RESERVE_MIN_BYTES)` — "25 % of total_ram, min 3 GiB" (Appendix A.3).
/// Expressed as an integer divisor rather than a floating-point fraction so
/// the whole formula stays exact, deterministic integer arithmetic.
pub const OS_RESERVE_DIVISOR: u64 = 4; // 1/4 = 25%
/// See [`OS_RESERVE_DIVISOR`].
pub const OS_RESERVE_MIN_BYTES: u64 = 3 * 1024 * 1024 * 1024; // 3 GiB

/// Bonsai 2 27B's recurrent (Gated-DeltaNet) state size per sequence, in
/// bytes, derived from `RecurrentCache::memory_bytes()`
/// (`crates/oxibonsai-model/src/hybrid/model.rs`, not owned by this
/// package): 48 linear-attention layers, each holding a conv1d state
/// (`[3][10240]` f32 = `122_880` B — 3 = `ssm.conv_kernel - 1` causal taps
/// over the concatenated qkv width) plus a per-v-head recurrent state `S`
/// (`[48][128][128]` f32 = `3_145_728` B — `ssm.time_step_rank=48` v-heads x
/// `ssm.state_size=128` x `head_v_dim=128`): `48 * (122_880 + 3_145_728) =
/// 156_893_184`. Gatekeeper REQUIRED #8 (waves 3+3.5 review) corrected this
/// from a stale `163_184_640` that did not match the real cache formula;
/// this is the CODE-side half of that fix (REQUIRED #19 covers the design
/// doc's own "~166 K" napkin-math text separately).
pub const BONSAI2_RECURRENT_BYTES: u64 = 156_893_184;

/// Bonsai 2 27B's KV cost per token at f16, from its Appendix A.3 geometry
/// (16 full-attention slots x 4 KV heads x 256 head_dim x 2 (K+V) x 2 bytes):
/// see [`kv_bytes_per_token`] for the general formula this is an instance of.
pub const BONSAI2_KV_BYTES_PER_TOKEN: u64 = 65_536;

/// Shipped operational default context length for Bonsai 2 (design doc
/// §3.7/Appendix A.3: "the default stays 8192 by policy"). Deliberately far
/// below the RAM-derived ceiling (178 176 tokens on a 24 GiB host with the
/// PQ2_0 weights, Appendix A.3 as corrected by gatekeeper fix #19): a
/// conservative out-of-the-box default that a caller can raise explicitly,
/// checked against [`max_context_for_budget`] via
/// [`validate_requested_context`].
///
/// This is **not** [`ModelConfig::max_seq_len`]'s `Default::default()`
/// value (which stays `4096` — a long-standing, model-agnostic default with
/// existing non-owned test coverage pinned to it; see this package's
/// recorded deviation). It is the value a Bonsai-2-aware caller (the CLI's
/// `--ctx`/model-registry defaults, B2-14's territory) should use.
pub const BONSAI2_DEFAULT_CONTEXT: usize = 8192;

/// `bytes_per_token = n_full_layers * n_kv_heads * head_dim * 2 (K+V) *
/// kv_elem_bytes` (Appendix A.3) — the per-token KV cost for a hybrid
/// model's full-attention layers (or every layer, for a non-hybrid model,
/// by passing the model's total layer count as `n_full_layers`).
///
/// Saturating throughout: a corrupted or adversarial GGUF's geometry fields
/// must not be able to wrap this into a small, plausible-looking number.
pub fn kv_bytes_per_token(
    n_full_layers: u64,
    n_kv_heads: u64,
    head_dim: u64,
    kv_elem_bytes: u64,
) -> u64 {
    n_full_layers
        .saturating_mul(n_kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(2)
        .saturating_mul(kv_elem_bytes)
}

/// The largest context length this host can afford, implementing Appendix
/// A.3 **exactly**:
///
/// ```text
/// usable   = total_ram - weight_bytes - recurrent_bytes
///                      - ACTIVATION_RESERVE_BYTES - OS_RESERVE
/// OS_RESERVE = max(total_ram / OS_RESERVE_DIVISOR, OS_RESERVE_MIN_BYTES)
/// max_ctx  = min(model_context_length, usable / bytes_per_token)
///            rounded down to 1024
/// ```
///
/// Every subtraction is saturating: weights (or weights + recurrent state)
/// larger than `total_ram` produce `usable = 0`, not a huge wrapped `u64` —
/// which would otherwise report an absurdly large max context on the exact
/// host least able to afford one. `bytes_per_token == 0` (a model with no
/// full-attention layers at all) is treated as "no KV-driven ceiling" and
/// returns `model_context_length` unclamped, rather than dividing by zero.
///
/// `model_context_length` is the model's own declared maximum (e.g.
/// `qwen35.context_length = 262144` for Bonsai 2); this function never
/// returns more than that regardless of how much RAM is available.
///
/// "Tune `OS_RESERVE` once measured; the formula, not the number, is the
/// contract" (Appendix A.3) — see [`OS_RESERVE_DIVISOR`]/
/// [`OS_RESERVE_MIN_BYTES`]/[`ACTIVATION_RESERVE_BYTES`].
pub fn max_context_for_budget(
    total_ram_bytes: u64,
    weight_bytes: u64,
    recurrent_bytes: u64,
    bytes_per_token: u64,
    model_context_length: usize,
) -> usize {
    let os_reserve = (total_ram_bytes / OS_RESERVE_DIVISOR).max(OS_RESERVE_MIN_BYTES);
    let usable = total_ram_bytes
        .saturating_sub(weight_bytes)
        .saturating_sub(recurrent_bytes)
        .saturating_sub(ACTIVATION_RESERVE_BYTES)
        .saturating_sub(os_reserve);

    // Appendix A.3's literal order: take `min(context_length,
    // usable/bytes_per_token)` FIRST, then round that combined bound down to
    // 1024 -- not the reverse (rounding `usable/bytes_per_token` alone
    // before the `.min`), which under-applies the "rounded down to 1024"
    // clause to `model_context_length` itself whenever the model's own
    // declared limit is the binding constraint and is not already a
    // multiple of 1024 (every real `<arch>.context_length` this build has
    // seen — 262144/131072/32768/8192/4096 — is, so the divergence was
    // previously unobservable; wave-3 review). `bytes_per_token == 0` (a
    // model with no full-attention layers at all) has no KV-driven ceiling
    // at all, so the pre-rounding bound is `model_context_length` itself
    // rather than a division — the final rounding step below still applies
    // uniformly, so this function's result is *always* a multiple of 1024,
    // never just "usually".
    // `checked_div` (rather than an explicit `bytes_per_token == 0` guard
    // followed by an unconditional `/`) both expresses "no KV-driven
    // ceiling" as `None` directly and keeps this crate's own
    // `clippy::manual_checked_ops` lint satisfied.
    let capped = match usable.checked_div(bytes_per_token) {
        None => model_context_length,
        Some(raw_max_ctx) => {
            // usize on every platform this crate targets is at least 32
            // bits, and `raw_max_ctx` is already bounded by `usable /
            // bytes_per_token`, which for any realistic (or
            // adversarial-but-finite) input fits comfortably; saturate
            // rather than risk a debug-mode panic on a 32-bit target.
            let raw_max_ctx = usize::try_from(raw_max_ctx).unwrap_or(usize::MAX);
            raw_max_ctx.min(model_context_length)
        }
    };
    (capped / 1024) * 1024
}

/// Refuse a requested context length that exceeds either the model's own
/// architectural limit or the RAM-derived limit from
/// [`max_context_for_budget`] (RT-29 / sec-11) — naming **both** bounds in
/// the error regardless of which one was actually violated, so an operator
/// sees the whole picture in one message instead of one number per retry.
///
/// # Errors
///
/// Returns [`RuntimeError::Config`] when `requested` exceeds
/// `model_context_length.min(ram_derived_max_ctx)`.
pub fn validate_requested_context(
    requested: usize,
    model_context_length: usize,
    ram_derived_max_ctx: usize,
) -> RuntimeResult<()> {
    let effective_limit = model_context_length.min(ram_derived_max_ctx);
    if requested > effective_limit {
        return Err(RuntimeError::Config(format!(
            "requested context {requested} exceeds this host's effective limit \
             ({effective_limit}): the model's own architectural limit is \
             {model_context_length} tokens, and the RAM-derived limit on this host is \
             {ram_derived_max_ctx} tokens"
        )));
    }
    Ok(())
}

/// Total physical RAM installed on this host, in bytes.
///
/// Returns `None` when the platform query fails or on a platform this
/// function does not know how to query (anything but Linux/macOS) — a
/// caller should treat that as "unknown, do not artificially constrain"
/// rather than fabricate a number ([`OxiBonsaiConfig::validate_context_budget`]
/// does exactly this, falling back to `u64::MAX`).
pub fn total_ram_bytes() -> Option<u64> {
    ram_platform::total_ram_bytes()
}

#[cfg(target_os = "macos")]
mod ram_platform {
    /// `sysctlbyname("hw.memsize")` — the standard macOS API for total
    /// physical memory (mirrors the pattern `crate::memory` already uses for
    /// per-process RSS via Mach `task_info`, just for hardware memory
    /// instead).
    pub(super) fn total_ram_bytes() -> Option<u64> {
        let name = std::ffi::CString::new("hw.memsize").ok()?;
        let mut mem: u64 = 0;
        let mut size: libc::size_t = std::mem::size_of::<u64>();
        // SAFETY: `name` is a valid, NUL-terminated C string for the
        // lifetime of this call; `mem`/`size` are valid, correctly-sized
        // buffers for the `oldp`/`oldlenp` out-parameters; `newp`/`newlen`
        // are null/0, requesting a read-only query (no write to sysctl).
        let ret = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut mem as *mut u64 as *mut core::ffi::c_void,
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 && mem > 0 {
            Some(mem)
        } else {
            None
        }
    }
}

#[cfg(target_os = "linux")]
mod ram_platform {
    /// Parse `MemTotal:` out of `/proc/meminfo` (kB, per `man proc`),
    /// mirroring the `/proc/self/statm` parsing `crate::memory` already
    /// uses for per-process RSS.
    pub(super) fn total_ram_bytes() -> Option<u64> {
        let content = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest.trim().split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod ram_platform {
    pub(super) fn total_ram_bytes() -> Option<u64> {
        None
    }
}

impl OxiBonsaiConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: &Path) -> RuntimeResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::Config(format!(
                "failed to read config file {}: {e}",
                path.display()
            ))
        })?;
        let config: Self = toml::from_str(&content).map_err(|e| {
            RuntimeError::Config(format!(
                "failed to parse config file {}: {e}",
                path.display()
            ))
        })?;
        Ok(config)
    }

    /// Load configuration from a TOML file if a path is given, otherwise return defaults.
    pub fn load_or_default(path: Option<&Path>) -> Self {
        match path {
            Some(p) => match Self::load(p) {
                Ok(cfg) => cfg,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to load config, using defaults");
                    Self::default()
                }
            },
            None => Self::default(),
        }
    }

    /// Validate this configuration, returning an error if any field is invalid.
    pub fn validate(&self) -> RuntimeResult<()> {
        if self.sampling.temperature < 0.0 {
            return Err(RuntimeError::Config(format!(
                "sampling.temperature must be >= 0.0, got {}",
                self.sampling.temperature
            )));
        }
        if self.sampling.top_p < 0.0 || self.sampling.top_p > 1.0 {
            return Err(RuntimeError::Config(format!(
                "sampling.top_p must be in [0.0, 1.0], got {}",
                self.sampling.top_p
            )));
        }
        if self.sampling.repetition_penalty < 1.0 {
            return Err(RuntimeError::Config(format!(
                "sampling.repetition_penalty must be >= 1.0, got {}",
                self.sampling.repetition_penalty
            )));
        }
        if self.sampling.max_tokens == 0 {
            return Err(RuntimeError::Config(
                "sampling.max_tokens must be > 0".to_string(),
            ));
        }
        if self.model.max_seq_len == 0 {
            return Err(RuntimeError::Config(
                "model.max_seq_len must be > 0".to_string(),
            ));
        }
        // `max_context` / `ctx_budget_bytes` feed B2-12's context-guard
        // formula (`max_context_for_budget` / `validate_context_budget`,
        // below), which needs the model's real weight/geometry numbers from
        // a loaded GGUF that `OxiBonsaiConfig` alone does not have — so only
        // the trivial non-zero sanity check is enforced here, not any
        // cross-field clamping; a caller with those numbers in hand should
        // call `validate_context_budget` directly.
        if self.model.max_context == Some(0) {
            return Err(RuntimeError::Config(
                "model.max_context, if set, must be > 0".to_string(),
            ));
        }
        if self.model.ctx_budget_bytes == Some(0) {
            return Err(RuntimeError::Config(
                "model.ctx_budget_bytes, if set, must be > 0".to_string(),
            ));
        }
        if self.imagen.width == 0 || self.imagen.height == 0 {
            return Err(RuntimeError::Config(
                "imagen.width and imagen.height must be > 0".to_string(),
            ));
        }
        if self.imagen.steps == 0 {
            return Err(RuntimeError::Config("imagen.steps must be > 0".to_string()));
        }
        if !self.imagen.guidance_scale.is_finite() {
            return Err(RuntimeError::Config(format!(
                "imagen.guidance_scale must be finite, got {}",
                self.imagen.guidance_scale
            )));
        }
        if self.server.host.is_empty() {
            return Err(RuntimeError::Config(
                "server.host must not be empty".to_string(),
            ));
        }
        // Port 0 is technically valid (OS assigns), so no check needed
        Ok(())
    }

    /// Validate the operator-requested [`ModelConfig::max_seq_len`] against
    /// both the model's own declared context limit and a RAM-derived
    /// ceiling (RT-29 / sec-11 / B2-12, Appendix A.3).
    ///
    /// Unlike [`OxiBonsaiConfig::validate`], this needs numbers
    /// `OxiBonsaiConfig` does not carry on its own — the model's on-disk
    /// weight size, its hybrid-architecture recurrent-state size (`0` for a
    /// non-hybrid model), the per-token KV cost for its geometry, and its
    /// own declared `<arch>.context_length` — so it takes them as
    /// parameters rather than reading `self.model.max_context` (which nothing
    /// populates from a real GGUF yet). Total RAM is auto-detected via
    /// [`total_ram_bytes`]; when detection fails (an unsupported platform),
    /// the RAM-derived bound is treated as unconstrained so an unrelated
    /// platform limitation cannot fail every request.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Config`] naming both `model_context_length`
    /// and the computed RAM-derived limit when
    /// `self.model.max_seq_len` exceeds either.
    pub fn validate_context_budget(
        &self,
        weight_bytes: u64,
        recurrent_bytes: u64,
        bytes_per_token: u64,
        model_context_length: usize,
    ) -> RuntimeResult<()> {
        let total_ram = total_ram_bytes().unwrap_or(u64::MAX);
        let ram_derived_max_ctx = max_context_for_budget(
            total_ram,
            weight_bytes,
            recurrent_bytes,
            bytes_per_token,
            model_context_length,
        );
        validate_requested_context(
            self.model.max_seq_len,
            model_context_length,
            ram_derived_max_ctx,
        )
    }

    /// [`OxiBonsaiConfig::validate_context_budget`], with `weight_bytes`
    /// derived from the real on-disk model at `model_path` instead of a
    /// caller-supplied number (sec-17 / M-Missed-4 addendum, "derive the
    /// guard from it" — this is the literal composition of
    /// [`oxibonsai_model::gguf_loader::estimate_memory_bytes`], this crate's
    /// existing `oxibonsai-model` dependency, into the RAM-derived context
    /// guard, closing the previously-unreported gap between the two).
    ///
    /// `recurrent_bytes`, `bytes_per_token`, and `model_context_length` are
    /// still taken as parameters: they come from the model's *architecture*
    /// (hybrid-state size, KV geometry, `<arch>.context_length`), not its
    /// on-disk weight bytes, and reading GGUF metadata for those is
    /// `oxibonsai-model`'s own loading path's job, not this function's.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Config`] when `model_path` cannot be read as
    /// a valid GGUF file (wrapping the underlying
    /// [`oxibonsai_model::gguf_loader::LoadError`]), or — via
    /// [`validate_context_budget`](Self::validate_context_budget) — when
    /// `self.model.max_seq_len` exceeds either the model's own limit or the
    /// RAM-derived one.
    pub fn validate_context_budget_for_model(
        &self,
        model_path: &Path,
        recurrent_bytes: u64,
        bytes_per_token: u64,
        model_context_length: usize,
    ) -> RuntimeResult<()> {
        let weight_bytes = oxibonsai_model::gguf_loader::estimate_memory_bytes(model_path)
            .map_err(|e| {
                RuntimeError::Config(format!(
                    "failed to estimate model memory footprint from {}: {e}",
                    model_path.display()
                ))
            })?;
        self.validate_context_budget(
            weight_bytes,
            recurrent_bytes,
            bytes_per_token,
            model_context_length,
        )
    }

    /// Run a dry-run check of this configuration.
    ///
    /// Returns warnings about potential issues without stopping execution.
    /// Checks for model file existence, tokenizer existence, and
    /// reasonable parameter values.
    pub fn dry_run_check(&self) -> Vec<ConfigWarning> {
        let mut warnings = Vec::new();

        // Check model file
        match &self.model.model_path {
            None => {
                warnings.push(ConfigWarning {
                    field: "model.model_path".to_string(),
                    message: "no model path configured".to_string(),
                    severity: WarningSeverity::Warning,
                });
            }
            Some(path) => {
                if !Path::new(path).exists() {
                    warnings.push(ConfigWarning {
                        field: "model.model_path".to_string(),
                        message: format!("model file does not exist: {}", path),
                        severity: WarningSeverity::Error,
                    });
                }
            }
        }

        // Check tokenizer file
        match &self.model.tokenizer_path {
            None => {
                warnings.push(ConfigWarning {
                    field: "model.tokenizer_path".to_string(),
                    message: "no tokenizer path configured; token IDs will be used".to_string(),
                    severity: WarningSeverity::Info,
                });
            }
            Some(path) => {
                if !Path::new(path).exists() {
                    warnings.push(ConfigWarning {
                        field: "model.tokenizer_path".to_string(),
                        message: format!("tokenizer file does not exist: {}", path),
                        severity: WarningSeverity::Error,
                    });
                }
            }
        }

        // Check sequence length
        if self.model.max_seq_len > 65536 {
            warnings.push(ConfigWarning {
                field: "model.max_seq_len".to_string(),
                message: format!(
                    "very large max_seq_len ({}); may require significant memory",
                    self.model.max_seq_len
                ),
                severity: WarningSeverity::Warning,
            });
        }

        // Check temperature
        if self.sampling.temperature > 2.0 {
            warnings.push(ConfigWarning {
                field: "sampling.temperature".to_string(),
                message: format!(
                    "high temperature ({}) may produce incoherent output",
                    self.sampling.temperature
                ),
                severity: WarningSeverity::Warning,
            });
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = OxiBonsaiConfig::default();
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.server.port, 8080);
        assert!((cfg.sampling.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.sampling.top_k, 40);
        assert!((cfg.sampling.top_p - 0.9).abs() < f32::EPSILON);
        // Gatekeeper REQUIRED #18: corrected from a stale `1.1` (see the
        // identical note on `SamplingConfig`'s `Default` impl above).
        assert!((cfg.sampling.repetition_penalty - 1.0).abs() < f32::EPSILON);
        assert_eq!(cfg.sampling.max_tokens, 512);
        assert_eq!(cfg.model.max_seq_len, 4096);
        assert!(cfg.model.model_path.is_none());
        assert!(cfg.model.tokenizer_path.is_none());
        assert!(
            cfg.model.max_context.is_none(),
            "max_context is unpopulated until B2-12 (wave 3) reads it from the GGUF"
        );
        assert!(
            cfg.model.ctx_budget_bytes.is_none(),
            "ctx_budget_bytes is unpopulated until B2-12 (wave 3) sets a default"
        );
        assert_eq!(cfg.observability.log_level, "info");
        assert!(!cfg.observability.json_logs);
        assert!(cfg.imagen.model_path.is_none());
        assert_eq!(cfg.imagen.width, 1024);
        assert_eq!(cfg.imagen.height, 1024);
        assert_eq!(cfg.imagen.steps, 4);
        assert!((cfg.imagen.guidance_scale - 3.5).abs() < f32::EPSILON);
        assert!(cfg.imagen.seed.is_none());
        assert!(cfg.imagen.output_dir.is_none());
    }

    #[test]
    fn toml_parsing() {
        let model_path = std::env::temp_dir().join("model.gguf");
        let tokenizer_path = std::env::temp_dir().join("tokenizer.json");
        let toml_str = format!(
            r#"
[server]
host = "127.0.0.1"
port = 3000

[sampling]
temperature = 0.5
top_k = 50
top_p = 0.95
repetition_penalty = 1.2
max_tokens = 1024

[model]
model_path = "{}"
tokenizer_path = "{}"
max_seq_len = 8192
max_context = 262144
ctx_budget_bytes = 17179869184

[observability]
log_level = "debug"
json_logs = true
"#,
            model_path.display(),
            tokenizer_path.display()
        );
        let cfg: OxiBonsaiConfig = toml::from_str(&toml_str).expect("should parse valid TOML");
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert_eq!(cfg.server.port, 3000);
        assert!((cfg.sampling.temperature - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.sampling.top_k, 50);
        assert_eq!(cfg.sampling.max_tokens, 1024);
        assert_eq!(
            cfg.model.model_path.as_deref(),
            Some(model_path.to_str().expect("path is valid UTF-8"))
        );
        assert_eq!(cfg.model.max_seq_len, 8192);
        assert_eq!(cfg.model.max_context, Some(262_144));
        assert_eq!(cfg.model.ctx_budget_bytes, Some(17_179_869_184));
        assert_eq!(cfg.observability.log_level, "debug");
        assert!(cfg.observability.json_logs);
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let toml_str = r#"
[server]
port = 9090
"#;
        let cfg: OxiBonsaiConfig = toml::from_str(toml_str).expect("should parse partial TOML");
        assert_eq!(cfg.server.port, 9090);
        // Rest should be defaults
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert!((cfg.sampling.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.model.max_seq_len, 4096);
    }

    #[test]
    fn missing_file_returns_default() {
        let path = std::env::temp_dir().join("nonexistent_oxibonsai_config_12345.toml");
        let cfg = OxiBonsaiConfig::load_or_default(Some(&path));
        assert_eq!(cfg.server.port, 8080);
    }

    #[test]
    fn load_or_default_none_returns_default() {
        let cfg = OxiBonsaiConfig::load_or_default(None);
        assert_eq!(cfg.server.host, "0.0.0.0");
    }

    // ── Validation tests ──

    #[test]
    fn validate_defaults_ok() {
        let cfg = OxiBonsaiConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_negative_temperature() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.sampling.temperature = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_top_p_out_of_range() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.sampling.top_p = 1.5;
        assert!(cfg.validate().is_err());

        cfg.sampling.top_p = -0.1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_repetition_penalty_too_low() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.sampling.repetition_penalty = 0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_max_tokens_zero() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.sampling.max_tokens = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_max_seq_len_zero() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.max_seq_len = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_max_context_unset_is_ok() {
        let cfg = OxiBonsaiConfig::default();
        assert!(cfg.model.max_context.is_none());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_max_context_zero_is_rejected() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.max_context = Some(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_max_context_positive_is_ok() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.max_context = Some(262_144);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_ctx_budget_bytes_zero_is_rejected() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.ctx_budget_bytes = Some(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_ctx_budget_bytes_positive_is_ok() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.ctx_budget_bytes = Some(8 * 1024 * 1024 * 1024);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_empty_host() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.server.host = String::new();
        assert!(cfg.validate().is_err());
    }

    // ── Dry-run check tests ──

    #[test]
    fn dry_run_no_model_path() {
        let cfg = OxiBonsaiConfig::default();
        let warnings = cfg.dry_run_check();
        assert!(warnings.iter().any(|w| w.field == "model.model_path"));
    }

    #[test]
    fn dry_run_nonexistent_model() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.model_path = Some(
            std::env::temp_dir()
                .join("nonexistent_oxibonsai_test_99999.gguf")
                .display()
                .to_string(),
        );
        let warnings = cfg.dry_run_check();
        let model_warning = warnings
            .iter()
            .find(|w| w.field == "model.model_path")
            .expect("should have model warning");
        assert_eq!(model_warning.severity, WarningSeverity::Error);
    }

    #[test]
    fn dry_run_high_temperature() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.sampling.temperature = 3.0;
        let warnings = cfg.dry_run_check();
        assert!(warnings.iter().any(|w| w.field == "sampling.temperature"));
    }

    #[test]
    fn dry_run_large_seq_len() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.max_seq_len = 100_000;
        let warnings = cfg.dry_run_check();
        assert!(warnings.iter().any(|w| w.field == "model.max_seq_len"));
    }

    #[test]
    fn warning_severity_display() {
        assert_eq!(format!("{}", WarningSeverity::Info), "info");
        assert_eq!(format!("{}", WarningSeverity::Warning), "warning");
        assert_eq!(format!("{}", WarningSeverity::Error), "error");
    }

    #[test]
    fn config_warning_display() {
        let w = ConfigWarning {
            field: "test.field".to_string(),
            message: "test message".to_string(),
            severity: WarningSeverity::Warning,
        };
        let s = format!("{}", w);
        assert!(s.contains("warning"));
        assert!(s.contains("test.field"));
        assert!(s.contains("test message"));
    }

    #[test]
    fn load_from_temp_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxibonsai_test_config.toml");
        std::fs::write(
            &path,
            r#"
[server]
host = "10.0.0.1"
port = 4444
"#,
        )
        .expect("write temp config");

        let cfg = OxiBonsaiConfig::load(&path).expect("should load temp config");
        assert_eq!(cfg.server.host, "10.0.0.1");
        assert_eq!(cfg.server.port, 4444);

        let _ = std::fs::remove_file(&path);
    }

    // ── [imagen] section (CLI-CORE deviation, addendum item 6) ─────────────

    #[test]
    fn imagen_toml_round_trip() {
        // CLAUDE.md: never hardcode absolute paths in tests, even as inert
        // config-value strings that touch no filesystem -- built from
        // `std::env::temp_dir()` instead (wave-3 review). TOML *literal*
        // strings (single-quoted) are used so a `\`-containing path (e.g. on
        // a platform whose temp dir uses backslashes) is never
        // escape-processed.
        let model_path = std::env::temp_dir().join("bonsai-image-model");
        let output_dir = std::env::temp_dir().join("bonsai-image-out");
        let toml_str = format!(
            "\n[imagen]\nmodel_path = '{model}'\nwidth = 512\nheight = 768\nsteps = 20\n\
             guidance_scale = 7.5\nseed = 42\noutput_dir = '{out}'\n",
            model = model_path.display(),
            out = output_dir.display(),
        );
        let cfg: OxiBonsaiConfig = toml::from_str(&toml_str).expect("should parse [imagen]");
        assert_eq!(
            cfg.imagen.model_path.as_deref(),
            Some(
                model_path
                    .to_str()
                    .expect("temp dir path must be valid UTF-8")
            )
        );
        assert_eq!(cfg.imagen.width, 512);
        assert_eq!(cfg.imagen.height, 768);
        assert_eq!(cfg.imagen.steps, 20);
        assert!((cfg.imagen.guidance_scale - 7.5).abs() < f32::EPSILON);
        assert_eq!(cfg.imagen.seed, Some(42));
        assert_eq!(
            cfg.imagen.output_dir.as_deref(),
            Some(
                output_dir
                    .to_str()
                    .expect("temp dir path must be valid UTF-8")
            )
        );
        // Every other section still gets its own defaults.
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.model.max_seq_len, 4096);
    }

    #[test]
    fn imagen_absent_section_uses_defaults() {
        let cfg: OxiBonsaiConfig = toml::from_str("[server]\nport = 1234\n").expect("parse");
        assert_eq!(cfg.imagen.width, 1024);
        assert_eq!(cfg.imagen.height, 1024);
        assert!(cfg.imagen.model_path.is_none());
    }

    #[test]
    fn validate_rejects_zero_imagen_dimensions() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.imagen.width = 0;
        assert!(cfg.validate().is_err());

        let mut cfg = OxiBonsaiConfig::default();
        cfg.imagen.height = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_imagen_steps() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.imagen.steps = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_negative_finite_guidance_but_rejects_non_finite() {
        // `oxibonsai_image::pipeline::validate_guidance` explicitly accepts
        // any finite value, including negative, as a real classifier-free-
        // guidance scale (its own doctest pins `validate_guidance(-3.5)` as
        // `Ok`) -- this config's `validate()` must match that contract
        // exactly rather than being stricter than the consumer it exists to
        // feed. Wave-3 review: a prior version of this test asserted the
        // opposite (that `-1.0` must be rejected), which was simply wrong
        // relative to the real pipeline it claimed to mirror.
        let mut cfg = OxiBonsaiConfig::default();
        cfg.imagen.guidance_scale = -1.0;
        assert!(
            cfg.validate().is_ok(),
            "a negative but finite guidance_scale must be accepted, matching validate_guidance"
        );

        cfg.imagen.guidance_scale = f32::NAN;
        assert!(
            cfg.validate().is_err(),
            "a NaN guidance_scale must be rejected"
        );

        cfg.imagen.guidance_scale = f32::INFINITY;
        assert!(
            cfg.validate().is_err(),
            "an infinite guidance_scale must be rejected"
        );
    }

    #[test]
    fn validate_accepts_reasonable_imagen_config() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.imagen.width = 2048;
        cfg.imagen.height = 2048;
        cfg.imagen.steps = 50;
        cfg.imagen.guidance_scale = 0.0; // 0.0 (no CFG) is a legitimate value
        assert!(cfg.validate().is_ok());
    }

    // ── RAM-derived context guard (RT-29 / sec-11 / M-07, Appendix A.3) ────

    #[test]
    fn kv_bytes_per_token_matches_bonsai2_27b_appendix_a3() {
        // 16 full-attention slots x 4 kv heads x 256 head_dim x 2 (K+V) x 2 bytes (f16).
        assert_eq!(kv_bytes_per_token(16, 4, 256, 2), 65_536);
        assert_eq!(
            kv_bytes_per_token(16, 4, 256, 2),
            BONSAI2_KV_BYTES_PER_TOKEN
        );
    }

    #[test]
    fn kv_bytes_per_token_saturates_instead_of_wrapping() {
        assert_eq!(
            kv_bytes_per_token(u64::MAX, u64::MAX, u64::MAX, u64::MAX),
            u64::MAX
        );
    }

    /// Appendix A.3's own worked example, computed **exactly** (not the
    /// document's illustrative "~166 K", which mixes GiB/GB units in its
    /// napkin math — "the formula, not the number, is the contract"). Using
    /// the *exact* PQ2_0 file size from Appendix A.4 (`7_206_168_928` B) and
    /// the corrected `BONSAI2_RECURRENT_BYTES`/`BONSAI2_KV_BYTES_PER_TOKEN`
    /// on a real 24 GiB host: `usable = 24 GiB - 7_206_168_928 -
    /// 156_893_184 - 256 MiB - 6 GiB(=25%) = 11_695_855_264`;
    /// `11_695_855_264 / 65_536 = 178_464`, rounded down to 1024 =
    /// `178_176` (unchanged from the pre-fix constant's `178_368 -> 178_176`
    /// rounding, since both fall in the same 1024-wide bucket) —
    /// comfortably below the model's own 262 144 limit, and comfortably
    /// above the 8192 shipped default.
    #[test]
    fn max_context_for_budget_bonsai2_27b_worked_example() {
        let total_ram = 24 * 1024 * 1024 * 1024u64;
        let result = max_context_for_budget(
            total_ram,
            7_206_168_928,
            BONSAI2_RECURRENT_BYTES,
            BONSAI2_KV_BYTES_PER_TOKEN,
            262_144,
        );
        assert_eq!(result, 178_176);
        assert!(result < 262_144, "must stay below the model's own limit");
        assert!(
            result > BONSAI2_DEFAULT_CONTEXT,
            "the RAM-derived ceiling must exceed the shipped conservative default"
        );
    }

    #[test]
    fn max_context_for_budget_result_is_always_a_multiple_of_1024() {
        // bytes_per_token = 1000 deliberately does not divide `usable`
        // evenly at a 1024 boundary, exercising the "rounded down to 1024"
        // clause specifically (not just an already-aligned coincidence).
        let result = max_context_for_budget(
            16 * 1024 * 1024 * 1024,
            1024 * 1024 * 1024,
            0,
            1000,
            20_000_000,
        );
        assert_eq!(result, 11_542_528);
        assert_eq!(result % 1024, 0);
    }

    #[test]
    fn max_context_for_budget_never_exceeds_the_model_declared_limit() {
        // Enormous RAM, tiny weights: the model's own architectural ceiling
        // must still win, exactly matching the shipped Bonsai 2 default.
        let result = max_context_for_budget(
            1024 * 1024 * 1024 * 1024,
            1024 * 1024 * 1024,
            0,
            65_536,
            8192,
        );
        assert_eq!(result, 8192);
    }

    #[test]
    fn max_context_for_budget_saturates_to_zero_when_weights_exceed_ram() {
        // 16 GiB of weights on an 8 GiB host: `usable` must saturate to 0,
        // not underflow to a huge number that reports an absurd max context
        // on exactly the host least able to afford one.
        let result = max_context_for_budget(
            8 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
            0,
            65_536,
            262_144,
        );
        assert_eq!(result, 0);
    }

    #[test]
    fn max_context_for_budget_zero_bytes_per_token_is_unclamped_by_ram() {
        // A model with no full-attention layers at all has no KV-driven
        // ceiling; this must not divide by zero, and must not artificially
        // clamp below the model's own declared limit -- but the function's
        // "always a multiple of 1024" contract (see
        // `max_context_for_budget_result_is_always_a_multiple_of_1024`)
        // still applies uniformly, so a non-1024-multiple `context_length`
        // of 5000 still rounds down to 4096 here, same as it would on the
        // RAM-derived path (wave-3 review: this branch previously bypassed
        // the rounding entirely and returned 5000 verbatim).
        let result = max_context_for_budget(1, 100, 100, 0, 5000);
        assert_eq!(result, 4096);
    }

    #[test]
    fn validate_requested_context_refuses_ctx_300000_naming_both_bounds() {
        // The literal acceptance scenario: `--ctx 300000` against Bonsai
        // 2's 262 144 model limit and a RAM-derived limit smaller still.
        let ram_derived = max_context_for_budget(
            24 * 1024 * 1024 * 1024,
            7_206_168_928,
            BONSAI2_RECURRENT_BYTES,
            BONSAI2_KV_BYTES_PER_TOKEN,
            262_144,
        );
        let err = validate_requested_context(300_000, 262_144, ram_derived)
            .expect_err("300000 must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("262144"),
            "error must name the model's own limit: {msg}"
        );
        assert!(
            msg.contains(&ram_derived.to_string()),
            "error must name the RAM-derived limit: {msg}"
        );
        assert!(
            msg.contains("300000"),
            "error must name the request itself: {msg}"
        );
    }

    #[test]
    fn validate_requested_context_accepts_the_shipped_default() {
        let ram_derived = max_context_for_budget(
            24 * 1024 * 1024 * 1024,
            7_206_168_928,
            BONSAI2_RECURRENT_BYTES,
            BONSAI2_KV_BYTES_PER_TOKEN,
            262_144,
        );
        assert!(validate_requested_context(BONSAI2_DEFAULT_CONTEXT, 262_144, ram_derived).is_ok());
    }

    #[test]
    fn validate_requested_context_reports_the_model_limit_when_that_is_the_binder() {
        // A request within the RAM budget but above the model's own limit
        // must still be refused, and still name both numbers.
        let err = validate_requested_context(300_000, 262_144, 400_000)
            .expect_err("300000 must be refused by the model's own 262144 limit");
        let msg = err.to_string();
        assert!(msg.contains("262144"));
        assert!(msg.contains("400000"));
    }

    #[test]
    fn total_ram_bytes_is_plausible_on_supported_platforms() {
        let ram = total_ram_bytes();
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            let ram = ram.expect("RAM detection should succeed on macOS/Linux");
            // Any real host has at least 512 MiB and less than 1 PiB.
            assert!(ram > 512 * 1024 * 1024, "implausibly small RAM: {ram}");
            assert!(ram < 1024u64.pow(5), "implausibly large RAM: {ram}");
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let _ = ram;
    }

    #[test]
    fn validate_context_budget_end_to_end_refuses_and_accepts() {
        let mut cfg = OxiBonsaiConfig::default();
        cfg.model.max_seq_len = 300_000;
        let err = cfg
            .validate_context_budget(
                7_206_168_928,
                BONSAI2_RECURRENT_BYTES,
                BONSAI2_KV_BYTES_PER_TOKEN,
                262_144,
            )
            .expect_err("300000 must be refused end-to-end through OxiBonsaiConfig");
        let msg = err.to_string();
        assert!(msg.contains("262144"));

        cfg.model.max_seq_len = BONSAI2_DEFAULT_CONTEXT;
        assert!(cfg
            .validate_context_budget(
                7_206_168_928,
                BONSAI2_RECURRENT_BYTES,
                BONSAI2_KV_BYTES_PER_TOKEN,
                262_144,
            )
            .is_ok());
    }

    /// A temp-file-backed synthetic GGUF with exactly one small tensor, so
    /// [`oxibonsai_model::gguf_loader::estimate_memory_bytes`] returns an
    /// exact, tiny, known value (its own on-disk size — this tensor is not
    /// named `token_embd.weight`, so no eager-dequant correction applies).
    /// Mirrors the pattern `oxibonsai_model::gguf_loader`'s own tests use
    /// (`std::env::temp_dir()`, never a hardcoded absolute path).
    fn write_tiny_synthetic_gguf(tag: &str) -> std::path::PathBuf {
        use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};

        let mut w = GgufWriter::new();
        w.add_tensor(TensorEntry {
            name: "output_norm.weight".to_string(),
            shape: vec![64],
            tensor_type: TensorType::F32,
            data: vec![0u8; 256], // 64 * 4 bytes
        });
        let bytes = w.to_bytes().expect("write synthetic gguf");

        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_validate_context_budget_for_model_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("synthetic.gguf");
        std::fs::write(&path, &bytes).expect("write temp gguf");
        path
    }

    #[test]
    fn validate_context_budget_for_model_derives_weight_bytes_from_the_real_gguf() {
        // sec-17 / M-Missed-4 addendum, "derive the guard from it": proves
        // `weight_bytes` really is read from the file on disk (not some
        // placeholder that always accepts) by refusing a request that is
        // only refusable once the model's own declared limit is applied --
        // the synthetic file's on-disk size (256 bytes) is negligible
        // against any real host's RAM, so only `model_context_length` can be
        // the binder here.
        let path = write_tiny_synthetic_gguf("accept_and_refuse");

        let cfg = OxiBonsaiConfig::default(); // max_seq_len = 4096
        assert!(
            cfg.validate_context_budget_for_model(&path, 0, 65_536, 262_144)
                .is_ok(),
            "the shipped default must be accepted against a negligible-weight model"
        );

        let mut cfg_over = cfg.clone();
        cfg_over.model.max_seq_len = 300_000;
        let err = cfg_over
            .validate_context_budget_for_model(&path, 0, 65_536, 262_144)
            .expect_err("300000 must be refused even with negligible weight_bytes");
        assert!(
            err.to_string().contains("262144"),
            "error must still name the model's own declared limit"
        );

        let _ = std::fs::remove_dir_all(path.parent().expect("has parent"));
    }

    #[test]
    fn validate_context_budget_for_model_reports_a_config_error_for_a_missing_file() {
        let missing = std::env::temp_dir().join(format!(
            "oxibonsai_validate_context_budget_for_model_missing_{}_{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let cfg = OxiBonsaiConfig::default();
        let err = cfg
            .validate_context_budget_for_model(&missing, 0, 65_536, 262_144)
            .expect_err("a missing model file must be a Config error, not a panic");
        assert!(matches!(err, RuntimeError::Config(_)));
    }

    // ── RopeScalingMode (wave-4b orchestrator addendum) ──

    #[test]
    fn rope_scaling_mode_defaults_to_auto() {
        assert_eq!(RopeScalingMode::default(), RopeScalingMode::Auto);
        assert_eq!(ModelConfig::default().rope_scaling, RopeScalingMode::Auto);
    }

    #[test]
    fn rope_scaling_mode_from_str_round_trips_all_variants() {
        for (s, mode) in [
            ("auto", RopeScalingMode::Auto),
            ("on", RopeScalingMode::On),
            ("off", RopeScalingMode::Off),
        ] {
            assert_eq!(s.parse::<RopeScalingMode>().expect("valid"), mode);
            assert_eq!(mode.to_string(), s);
        }
    }

    #[test]
    fn rope_scaling_mode_from_str_rejects_unknown_values() {
        let err = "yarn".parse::<RopeScalingMode>().expect_err("must reject");
        assert!(err.contains("yarn"));
        assert!(err.contains("auto"));
    }

    #[test]
    fn rope_scaling_mode_toml_round_trip() {
        let toml_str = r#"
[model]
rope_scaling = "off"
"#;
        let cfg: OxiBonsaiConfig = toml::from_str(toml_str).expect("should parse");
        assert_eq!(cfg.model.rope_scaling, RopeScalingMode::Off);

        let empty: OxiBonsaiConfig = toml::from_str("").expect("empty parses to defaults");
        assert_eq!(empty.model.rope_scaling, RopeScalingMode::Auto);
    }

    #[test]
    fn rope_scaling_mode_converts_to_the_core_override() {
        use oxibonsai_core::config::RopeScalingOverride;
        assert_eq!(
            RopeScalingOverride::from(RopeScalingMode::Auto),
            RopeScalingOverride::Auto
        );
        assert_eq!(
            RopeScalingOverride::from(RopeScalingMode::Off),
            RopeScalingOverride::Off
        );
        assert_eq!(
            RopeScalingOverride::from(RopeScalingMode::On),
            RopeScalingOverride::On
        );
    }

    /// The real Bonsai 2 27B `qwen35` hybrid geometry (design Appendix A.4),
    /// written out field by field so these cross-checks do not depend on a
    /// 7 GB file being present.
    fn bonsai2_27b_hybrid_config() -> oxibonsai_core::config_hybrid::HybridConfig {
        oxibonsai_core::config_hybrid::HybridConfig {
            base: oxibonsai_core::config::Qwen3Config {
                hidden_size: 5120,
                intermediate_size: 17408,
                num_layers: 64,
                num_attention_heads: 24,
                num_kv_heads: 4,
                head_dim: 256,
                value_length: 256,
                vocab_size: 248_320,
                max_context_length: 262_144,
                rms_norm_eps: 1e-6,
                rope_freq_base: 1.0e7,
                rope_scaling: oxibonsai_core::config::RopeScaling::None,
                sliding_window: None,
                architecture: "qwen35".to_string(),
                model_name: "Ternary-Bonsai-2-27B".to_string(),
            },
            full_attention_interval: 4,
            rope_dimension_count: 64,
            rope_sections: [11, 11, 10, 0],
            ssm_conv_kernel: 4,
            ssm_state_size: 128,
            ssm_group_count: 16,
            ssm_time_step_rank: 48,
            ssm_inner_size: 6144,
            nextn_predict_layers: 0,
            sampling_top_k: Some(20),
            sampling_top_p: Some(0.95),
            sampling_temperature: Some(1.0),
        }
    }

    /// Gatekeeper REQUIRED #8: `BONSAI2_RECURRENT_BYTES` must BE what the
    /// model's own recurrent cache allocates, not a hand-copied number —
    /// asserted against a real `RecurrentCache` built for the 27B geometry
    /// (zero-initialised, so the ~157 MB are lazily committed pages).
    #[test]
    fn bonsai2_recurrent_bytes_equals_the_real_recurrent_cache() {
        let cfg = bonsai2_27b_hybrid_config();
        cfg.validate().expect("the 27B geometry is valid");
        let cache = oxibonsai_model::hybrid::RecurrentCache::new(&cfg).expect("allocate cache");
        assert_eq!(cache.memory_bytes() as u64, BONSAI2_RECURRENT_BYTES);
        assert_eq!(BONSAI2_RECURRENT_BYTES, 156_893_184);
    }

    /// The companion constant: 16 full-attention layers x 4 KV heads x 256
    /// head dim x 2 (K+V) x 2 bytes (f16) = 64 KiB per token.
    #[test]
    fn bonsai2_kv_bytes_per_token_matches_the_hybrid_geometry() {
        let cfg = bonsai2_27b_hybrid_config();
        assert_eq!(cfg.num_full_layers(), 16);
        assert_eq!(cfg.num_linear_layers(), 48);
        let derived = kv_bytes_per_token(
            cfg.num_full_layers() as u64,
            cfg.base.num_kv_heads as u64,
            cfg.base.head_dim as u64,
            2,
        );
        assert_eq!(derived, BONSAI2_KV_BYTES_PER_TOKEN);
        assert_eq!(BONSAI2_KV_BYTES_PER_TOKEN, 65_536);
    }

    /// Design §5.5's `SamplingConfig::from_gguf_defaults`: the model's
    /// declared `general.sampling.{temp,top_p,top_k}` seed the defaults,
    /// every other field (and every undeclared one) keeps
    /// `SamplingConfig::default()`.
    #[test]
    fn sampling_config_from_gguf_defaults_reads_general_sampling() {
        use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};
        // `general.sampling.top_k` is written as GGUF INT32 (type 5) —
        // the type the real Bonsai 2 27B file stores it as.
        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen35".to_string()),
        );
        writer.add_metadata("general.sampling.temp", MetadataWriteValue::F32(1.0));
        writer.add_metadata("general.sampling.top_p", MetadataWriteValue::F32(0.95));
        writer.add_metadata("general.sampling.top_k", MetadataWriteValue::I32(20));
        let bytes = writer.to_bytes().expect("build fixture gguf");
        let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes).expect("parse fixture");

        let cfg = SamplingConfig::from_gguf_defaults(&gguf.metadata);
        assert!((cfg.temperature - 1.0).abs() < f32::EPSILON);
        assert!((cfg.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(cfg.top_k, 20);
        let base = SamplingConfig::default();
        assert!((cfg.repetition_penalty - base.repetition_penalty).abs() < f32::EPSILON);
        assert_eq!(cfg.max_tokens, base.max_tokens);
    }

    #[test]
    fn sampling_config_from_gguf_defaults_falls_back_when_undeclared() {
        let mut builder = oxibonsai_testkit::gguf_fixture::GgufFixtureBuilder::new();
        builder.metadata_str("general.architecture", "qwen3");
        let bytes = builder.build().expect("build fixture gguf");
        let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes).expect("parse fixture");

        let cfg = SamplingConfig::from_gguf_defaults(&gguf.metadata);
        let base = SamplingConfig::default();
        assert!((cfg.temperature - base.temperature).abs() < f32::EPSILON);
        assert!((cfg.top_p - base.top_p).abs() < f32::EPSILON);
        assert_eq!(cfg.top_k, base.top_k);
    }
}
