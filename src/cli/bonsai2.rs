//! Bonsai 2 (`qwen35` hybrid) specific CLI glue (design §5.6-§5.8).
//!
//! Everything here answers "does the loaded model belong to the Bonsai 2
//! family, and if so what must the CLI do differently":
//!
//! 1. [`is_qwen35_hybrid`] — model-family detection from
//!    `general.architecture`.
//! 2. [`default_max_seq_len`] — the default context policy (design §5.6):
//!    `BONSAI2_DEFAULT_CONTEXT` (8192) for a `qwen35` hybrid, the
//!    model-agnostic `4096` for every other architecture.
//! 3. [`HybridStateGeometry`] + [`check_context_budget`] +
//!    [`validate_context_for_model`] — the RAM + model-limit context guard
//!    (design §5.6 / Appendix A.3). The per-token KV cost and
//!    the per-sequence recurrent-state size are derived from the file's own
//!    [`HybridConfig`] (never a hardcoded 27B constant), and the refusal
//!    names the request, the model's own limit, the RAM-derived limit and the
//!    resulting GiB.
//! 4. [`default_enable_thinking`] — the `--think` default derived from the
//!    model's chat template: whether the template opens its generation
//!    prompt inside a `<think>` block when `enable_thinking` is left
//!    undefined.
//! 5. [`VisionRequest`] — the §5.7 vision flags (`--mmproj`, `--image`,
//!    `--image-max-tokens`): parsed and validated, then refused with the
//!    typed [`NOT_YET_SUPPORTED`] error until the vision tower lands.
//! 6. [`apply_prefill_chunk`] — `--prefill-chunk <N>`.
//!
//! Scope note: the context guard applies to the `qwen35` architecture. A
//! dense model already clamps an over-long request to its own declared
//! context at load (with a warning) and grows its `f16` KV cache lazily, so
//! it pre-commits no RAM a guard would need to protect.

use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_runtime::config::{
    RenderMessage, RenderOptions, ResolvedChatTemplate, BONSAI2_DEFAULT_CONTEXT,
};

/// The stable code of every "accepted, validated, but not implemented in
/// this release" refusal.
pub(crate) const NOT_YET_SUPPORTED: &str = "NOT_YET_SUPPORTED";

/// Build the typed [`NOT_YET_SUPPORTED`] error: `[NOT_YET_SUPPORTED]
/// <what>: <why>`.
pub(crate) fn not_yet_supported(what: &str, why: &str) -> anyhow::Error {
    anyhow::anyhow!("[{NOT_YET_SUPPORTED}] {what}: {why}")
}

/// Whether `arch` (a GGUF's `general.architecture` value) is the Bonsai 2 /
/// Qwen3.5 hybrid family (64-layer, 16-full/48-linear-attention GDN hybrid).
#[must_use]
pub(crate) fn is_qwen35_hybrid(arch: &str) -> bool {
    arch == "qwen35"
}

/// The `--max-seq-len`/`--ctx` default to apply when neither the CLI flag
/// nor `--config`'s `[model].max_seq_len` was given (design §5.6): `8192`
/// for a `qwen35` hybrid, the pre-existing model-agnostic `4096` otherwise.
#[must_use]
pub(crate) fn default_max_seq_len(arch: &str) -> usize {
    if is_qwen35_hybrid(arch) {
        BONSAI2_DEFAULT_CONTEXT
    } else {
        4096
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Per-sequence state geometry (design §3.6/§3.7/§5.6)
// ──────────────────────────────────────────────────────────────────────────

/// `f16` KV element size in bytes — the hybrid model's shipped KV precision.
const KV_ELEM_BYTES: u64 = 2;
/// `f32` recurrent-state element size in bytes.
const STATE_ELEM_BYTES: u64 = 4;

/// What one sequence of a `qwen35` hybrid costs in memory, derived from the
/// file's own [`HybridConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HybridStateGeometry {
    /// Total layers (`qwen35.block_count`).
    pub(crate) n_layers: usize,
    /// Full-attention (KV-cached) layers.
    pub(crate) n_full_layers: usize,
    /// Gated-DeltaNet (recurrent) layers.
    pub(crate) n_linear_layers: usize,
    /// KV bytes per token at `f16`: `n_full * n_kv_heads * (key_len +
    /// value_len) * 2` (27B: 65 536 = 64 KiB).
    pub(crate) kv_bytes_per_token: u64,
    /// Recurrent bytes per sequence: `n_linear * (conv_dim * (conv_kernel -
    /// 1) + n_v_heads * head_k_dim * head_v_dim) * 4` (27B: 156 893 184,
    /// what `RecurrentCache::memory_bytes()` allocates).
    pub(crate) recurrent_bytes: u64,
}

impl HybridStateGeometry {
    /// Derive the geometry from `cfg` (saturating, so a corrupt header can
    /// never wrap into a small, plausible number).
    pub(crate) fn from_config(cfg: &HybridConfig) -> Self {
        let n_full = cfg.num_full_layers();
        let n_linear = cfg.num_linear_layers();
        let per_token_kv_elems = (cfg.base.num_kv_heads as u64).saturating_mul(
            (cfg.base.head_dim as u64).saturating_add(cfg.base.value_length as u64),
        );
        let kv_bytes_per_token = (n_full as u64)
            .saturating_mul(per_token_kv_elems)
            .saturating_mul(KV_ELEM_BYTES);
        let conv_taps = cfg.ssm_conv_kernel.saturating_sub(1) as u64;
        let conv_elems = (cfg.conv_dim() as u64).saturating_mul(conv_taps);
        let ssm_elems = (cfg.n_v_heads() as u64)
            .saturating_mul(cfg.head_k_dim() as u64)
            .saturating_mul(cfg.head_v_dim() as u64);
        let recurrent_bytes = (n_linear as u64)
            .saturating_mul(conv_elems.saturating_add(ssm_elems))
            .saturating_mul(STATE_ELEM_BYTES);
        Self {
            n_layers: cfg.base.num_layers,
            n_full_layers: n_full,
            n_linear_layers: n_linear,
            kv_bytes_per_token,
            recurrent_bytes,
        }
    }

    /// KV bytes a context of `ctx` tokens costs.
    pub(crate) fn kv_bytes_at(&self, ctx: usize) -> u64 {
        (ctx as u64).saturating_mul(self.kv_bytes_per_token)
    }
}

/// Bytes rendered as GiB with two decimals (`"18.31 GiB"`).
pub(crate) fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

// ──────────────────────────────────────────────────────────────────────────
// Context guard (design §5.6 / Appendix A.3)
// ──────────────────────────────────────────────────────────────────────────

/// Every input of the context guard as plain numbers, so the decision (and
/// its message) is testable independently of the host it runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextBudgetInputs {
    /// The requested context (`--ctx`/`--max-seq-len`, or the default).
    pub(crate) requested: usize,
    /// The model's own declared limit (`qwen35.context_length`).
    pub(crate) model_context_length: usize,
    /// Physical RAM of the host (`u64::MAX` = unknown, do not constrain).
    pub(crate) total_ram_bytes: u64,
    /// Bytes the weights occupy (the GGUF file, or a transcoded image).
    pub(crate) weight_bytes: u64,
    /// Recurrent-state bytes per sequence.
    pub(crate) recurrent_bytes: u64,
    /// KV bytes per token.
    pub(crate) kv_bytes_per_token: u64,
}

/// What an accepted request resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextBudget {
    /// The RAM-derived maximum context (Appendix A.3's `max_ctx`).
    pub(crate) ram_limit: usize,
    /// `min(model limit, RAM-derived limit)`.
    pub(crate) effective_limit: usize,
}

/// Appendix A.3's usable budget: RAM minus weights, recurrent state, the
/// activation reserve and the OS reserve (saturating).
fn usable_budget_bytes(inputs: &ContextBudgetInputs) -> u64 {
    let os_reserve = (inputs.total_ram_bytes / oxibonsai_runtime::config::OS_RESERVE_DIVISOR)
        .max(oxibonsai_runtime::config::OS_RESERVE_MIN_BYTES);
    inputs
        .total_ram_bytes
        .saturating_sub(inputs.weight_bytes)
        .saturating_sub(inputs.recurrent_bytes)
        .saturating_sub(oxibonsai_runtime::config::ACTIVATION_RESERVE_BYTES)
        .saturating_sub(os_reserve)
}

/// Accept `inputs.requested` or refuse it with a message naming the
/// request, the model's own limit, the RAM-derived limit, and the resulting
/// GiB (design §5.6).
pub(crate) fn check_context_budget(inputs: &ContextBudgetInputs) -> Result<ContextBudget, String> {
    let ram_limit = oxibonsai_runtime::config::max_context_for_budget(
        inputs.total_ram_bytes,
        inputs.weight_bytes,
        inputs.recurrent_bytes,
        inputs.kv_bytes_per_token,
        inputs.model_context_length,
    );
    // `max_context_for_budget` also caps at the model limit; report the pure
    // RAM-derived bound separately so both numbers are visible.
    let usable = usable_budget_bytes(inputs);
    let pure_ram_limit = match usable.checked_div(inputs.kv_bytes_per_token) {
        Some(tokens) => (usize::try_from(tokens).unwrap_or(usize::MAX) / 1024) * 1024,
        None => usize::MAX,
    };
    let effective_limit = inputs.model_context_length.min(ram_limit);
    if inputs.requested <= effective_limit {
        return Ok(ContextBudget {
            ram_limit,
            effective_limit,
        });
    }
    let requested_kv = (inputs.requested as u64).saturating_mul(inputs.kv_bytes_per_token);
    // `u64::MAX` is the caller's "host RAM unknown" sentinel: the RAM bound
    // is then unbounded and only the model's own limit applies.
    let ram_text = if inputs.total_ram_bytes == u64::MAX || pure_ram_limit == usize::MAX {
        "unbounded (the host's RAM could not be determined, so only the model's own limit \
         applies)"
            .to_string()
    } else {
        format!(
            "{pure_ram_limit} tokens ({usable} usable for the KV cache = {total} RAM - {weights} \
             weights - {recurrent} recurrent state - reserves)",
            usable = gib(usable),
            total = gib(inputs.total_ram_bytes),
            weights = gib(inputs.weight_bytes),
            recurrent = gib(inputs.recurrent_bytes),
        )
    };
    Err(format!(
        "--ctx {requested} is refused: it exceeds this host's effective limit of {effective} \
         tokens. The model's own context limit is {model} tokens, and the RAM-derived limit is \
         {ram_text}. {requested} tokens would need {kv} of KV cache ({bpt} bytes/token) plus \
         {recurrent} of recurrent state. Pass --ctx <= {effective}.",
        requested = inputs.requested,
        effective = effective_limit,
        model = inputs.model_context_length,
        recurrent = gib(inputs.recurrent_bytes),
        kv = gib(requested_kv),
        bpt = inputs.kv_bytes_per_token,
    ))
}

/// Apply the context guard to a loaded GGUF (a no-op success for every
/// non-`qwen35` architecture; see the module scope note).
///
/// `weight_bytes` is what the weights occupy in RAM: the GGUF file size, or
/// a `--ptq1-transcode` image's size.
pub(crate) fn validate_context_for_model(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    arch: &str,
    weight_bytes: u64,
    requested_max_seq_len: usize,
) -> anyhow::Result<()> {
    if !is_qwen35_hybrid(arch) {
        return Ok(());
    }
    let cfg = HybridConfig::from_metadata(&gguf.metadata)
        .map_err(|e| anyhow::anyhow!("failed to read this qwen35 model's hybrid config: {e}"))?;
    let geometry = HybridStateGeometry::from_config(&cfg);
    let inputs = ContextBudgetInputs {
        requested: requested_max_seq_len,
        model_context_length: cfg.base.max_context_length,
        total_ram_bytes: oxibonsai_runtime::config::total_ram_bytes().unwrap_or(u64::MAX),
        weight_bytes,
        recurrent_bytes: geometry.recurrent_bytes,
        kv_bytes_per_token: geometry.kv_bytes_per_token,
    };
    let budget = check_context_budget(&inputs).map_err(anyhow::Error::msg)?;
    tracing::info!(
        requested = requested_max_seq_len,
        model_limit = cfg.base.max_context_length,
        ram_limit = budget.ram_limit,
        kv = %gib(geometry.kv_bytes_at(requested_max_seq_len)),
        recurrent = %gib(geometry.recurrent_bytes),
        "qwen35 context budget accepted"
    );
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// `--think` default derived from the template
// ──────────────────────────────────────────────────────────────────────────

/// Whether `rendered` (a fully rendered prompt) leaves the model inside an
/// open `<think>` block: the last `<think>` comes after the last
/// `</think>`. The Bonsai 2 template's generation prompt ends with
/// `<think>\n` when thinking is on, and with `<think>\n\n</think>\n\n` when
/// it is off.
pub(crate) fn prompt_opens_think_block(rendered: &str) -> bool {
    match (rendered.rfind("<think>"), rendered.rfind("</think>")) {
        (Some(open), Some(close)) => open > close,
        (Some(_), None) => true,
        _ => false,
    }
}

/// The template's own thinking default: render a one-message probe with
/// `add_generation_prompt` and `enable_thinking` left undefined, and report
/// whether the generation prompt opens inside `<think>`.
///
/// Callers keep passing `enable_thinking = None` to the template when the
/// user gave neither `--think` nor `--no-think` (the template's own
/// "undefined" branch is part of its byte-exact contract); this value tells
/// the CLI what that default *means* for the model (log line, reasoning
/// display).
///
/// # Errors
///
/// Returns the template's own render error.
pub(crate) fn default_enable_thinking(template: &ResolvedChatTemplate) -> anyhow::Result<bool> {
    let probe = [RenderMessage::new("user", "ping")];
    let opts = RenderOptions {
        add_generation_prompt: true,
        ..RenderOptions::default()
    };
    let rendered = template
        .render_with(&probe, &opts)
        .map_err(|e| anyhow::anyhow!("chat template failed to render a probe prompt: {e}"))?;
    Ok(prompt_opens_think_block(&rendered))
}

// ──────────────────────────────────────────────────────────────────────────
// §5.7 vision flags
// ──────────────────────────────────────────────────────────────────────────

/// The §5.7 vision inputs one command received.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct VisionRequest {
    /// `--mmproj <path>`.
    pub(crate) mmproj: Option<String>,
    /// `--image <path|url>` (repeatable).
    pub(crate) images: Vec<String>,
    /// `--image-max-tokens <N>` when passed explicitly.
    pub(crate) image_max_tokens: Option<usize>,
}

/// The reference demo's per-image token budget (design §5.7).
pub(crate) const DEFAULT_IMAGE_MAX_TOKENS: usize = 1024;

impl VisionRequest {
    /// `true` when no vision flag was passed at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.mmproj.is_none() && self.images.is_empty() && self.image_max_tokens.is_none()
    }

    /// The effective per-image budget (explicit value or the 1024 default).
    pub(crate) fn effective_image_max_tokens(&self) -> usize {
        self.image_max_tokens.unwrap_or(DEFAULT_IMAGE_MAX_TOKENS)
    }

    /// Validate every vision input, then refuse the request with the typed
    /// [`NOT_YET_SUPPORTED`] error (a no-op success when nothing was
    /// passed). A malformed input is reported as such first, so fixing the
    /// command line never hides behind the "not yet supported" answer.
    ///
    /// # Errors
    ///
    /// A validation failure (missing/unreadable mmproj, a projector GGUF
    /// whose `general.architecture` is not `clip`, a missing image file, a
    /// zero token budget), else the [`NOT_YET_SUPPORTED`] refusal.
    pub(crate) fn reject_until_supported(&self) -> anyhow::Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        if let Some(path) = &self.mmproj {
            validate_mmproj(path)?;
        }
        for image in &self.images {
            validate_image_ref(image)?;
        }
        if self.effective_image_max_tokens() == 0 {
            anyhow::bail!("--image-max-tokens must be >= 1");
        }
        Err(not_yet_supported(
            "vision input (--mmproj / --image / --image-max-tokens)",
            &format!(
                "validated ({} image(s), budget {} tokens/image{}), but the Qwen3-VL vision \
                 tower is not wired up yet; rerun without the vision flags for \
                 text-only inference",
                self.images.len(),
                self.effective_image_max_tokens(),
                self.mmproj
                    .as_deref()
                    .map(|p| format!(", projector {p}"))
                    .unwrap_or_default(),
            ),
        ))
    }
}

/// `--mmproj` must name a readable GGUF whose `general.architecture` is
/// `clip` (the vision projector, never a language model).
fn validate_mmproj(path: &str) -> anyhow::Result<()> {
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(path))
        .map_err(|e| anyhow::anyhow!("--mmproj {path}: cannot open: {e}"))?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)
        .map_err(|e| anyhow::anyhow!("--mmproj {path}: not a valid GGUF: {e}"))?;
    let arch = gguf
        .metadata
        .get_string(oxibonsai_core::gguf::tensor_info::keys::GENERAL_ARCHITECTURE)
        .unwrap_or("");
    if arch != "clip" {
        anyhow::bail!(
            "--mmproj {path}: general.architecture is '{arch}', expected 'clip' (a vision \
             projector such as Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf)"
        );
    }
    Ok(())
}

/// `--image` must be an `http(s)://` URL or an existing file.
fn validate_image_ref(image: &str) -> anyhow::Result<()> {
    if image.starts_with("http://") || image.starts_with("https://") {
        return Ok(());
    }
    let meta = std::fs::metadata(image)
        .map_err(|e| anyhow::anyhow!("--image {image}: cannot read: {e}"))?;
    if !meta.is_file() {
        anyhow::bail!("--image {image}: not a regular file");
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// `--prefill-chunk <N>` (design §5.7)
// ──────────────────────────────────────────────────────────────────────────

/// Apply `--prefill-chunk` to a loaded engine: the Gated-DeltaNet prefill
/// chunk of a hybrid model (`HybridModel::set_prefill_chunk`, default
/// 512), or the chunked-prefill plan of a dense one
/// (`BonsaiModel::set_prefill_chunk_tokens`). `None` keeps the model's own
/// default.
///
/// # Errors
///
/// The model's own refusal (a zero chunk — already rejected by the flag
/// parser, re-checked here).
pub(crate) fn apply_prefill_chunk(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prefill_chunk: Option<usize>,
) -> anyhow::Result<()> {
    let Some(chunk) = prefill_chunk else {
        return Ok(());
    };
    if let Some(hybrid) = engine.hybrid_model_mut() {
        hybrid
            .set_prefill_chunk(chunk)
            .map_err(|e| anyhow::anyhow!("--prefill-chunk {chunk}: {e}"))?;
        tracing::info!(chunk, "hybrid Gated-DeltaNet prefill chunk set");
    } else if let Some(dense) = engine.dense_model_mut() {
        anyhow::ensure!(chunk >= 1, "--prefill-chunk must be >= 1");
        dense.set_prefill_chunk_tokens(chunk);
        tracing::info!(chunk, "dense chunked-prefill size set");
    }
    Ok(())
}

#[cfg(test)]
#[path = "bonsai2_tests.rs"]
mod tests;
