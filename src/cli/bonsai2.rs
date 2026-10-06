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
//!    `--image-max-tokens`): validated before any model is loaded; the
//!    projector's header read for the options the engine is built with
//!    ([`VisionRequest::hybrid_load_options`]: a Metal-backed engine's KV
//!    window leaves room for the Metal tower); then the Qwen3-VL projector
//!    loaded once, for the executor the engine decodes on
//!    ([`VisionRequest::load_service_for`]), and every `--image` prepared
//!    ([`VisionRequest::prepare_images`]), checked against the context
//!    ([`prompt_rows`]) and encoded ([`VisionRequest::encode_prepared`]) for
//!    the multimodal prefill (design §6.2); which image references resolve
//!    comes from [`ImageSourceFlags`] (`--allow-image-url-fetch` with
//!    `--image-url-timeout-ms` / `--image-url-allow-host`, `serve
//!    --media-path`, and their `OXI_*` environment fallbacks; the opt-in
//!    installs the remote-image fetcher of [`super::image_fetch`]). Loading the
//!    projector also holds the ids the splice layer uses for
//!    `<|vision_start|>`, `<|vision_end|>` and `<|image_pad|>` against the
//!    model's own vocabulary ([`check_vision_markers`]) and refuses a model
//!    whose vocabulary disagrees.
//! 6. [`apply_prefill_chunk`] — `--prefill-chunk <N>`, reporting the chunk
//!    in effect once the engine is built ([`PrefillChunkOutcome`]): the
//!    model's chunk, or the call size the engine's executor takes for it.
//!
//! Scope note: the context guard applies to the `qwen35` architecture. A
//! dense model already clamps an over-long request to its own declared
//! context at load (with a warning) and grows its `f16` KV cache lazily, so
//! it pre-commits no RAM a guard would need to protect.

use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_runtime::config::{
    RenderMessage, RenderOptions, ResolvedChatTemplate, BONSAI2_DEFAULT_CONTEXT,
};

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
    /// `--image <path>` (repeatable; `run`/`chat`).
    pub(crate) images: Vec<String>,
    /// `--image-max-tokens <N>` when passed explicitly.
    pub(crate) image_max_tokens: Option<usize>,
}

/// The per-image merged-token budget unless `--image-max-tokens` says
/// otherwise (the Bonsai demo's default, design §5.7).
pub(crate) const DEFAULT_IMAGE_MAX_TOKENS: usize =
    oxibonsai_model::vision::DEFAULT_IMAGE_MAX_TOKENS;

/// Environment fallback for `--allow-image-url-fetch`: any of `1`, `true`,
/// `yes`, `on` (case-insensitive) opts in to remote (`http(s)`) image
/// references, which `run`, `chat` and `serve` then fetch through the
/// fetcher of [`super::image_fetch`] under its address policy (public
/// addresses only, plus the operator's allowlist). Without the opt-in a
/// remote reference is refused with `image_url_fetch_disabled` before any
/// connection or name lookup.
pub(crate) const ALLOW_IMAGE_URL_FETCH_ENV: &str = "OXI_ALLOW_IMAGE_URL_FETCH";

/// Environment fallback for `serve --media-path`: the directory `file://`
/// image references resolve inside (relative, no `..`, nothing outside it
/// once symlinks are resolved). Unset, a server accepts base64 `data:` URIs
/// only.
#[cfg(feature = "server")]
pub(crate) const MEDIA_PATH_ENV: &str = "OXI_MEDIA_PATH";

/// The image-source settings one command received: `--allow-image-url-fetch`,
/// `--image-url-timeout-ms` and `--image-url-allow-host` (`run`, `chat`,
/// `serve`) and `--media-path` (`serve`). They decide which image
/// *references* resolve, not what is encoded, so they travel beside
/// [`VisionRequest`] rather than in it.
///
/// Precedence: the flag, then its environment fallback
/// ([`ALLOW_IMAGE_URL_FETCH_ENV`], [`super::image_fetch::TIMEOUT_ENV`],
/// `MEDIA_PATH_ENV`), then the default (no opt-in; a 10 000 ms per-image
/// deadline; no media directory, so `data:` URIs only). A flag that is given
/// always wins, so an environment left over in a shell cannot widen or
/// redirect what a command line asked for — except the allowlist, whose
/// flag entries are *added* to [`super::image_fetch::ALLOW_HOSTS_ENV`]'s.
/// An allowlist (the flag or the environment) or an `--image-url-timeout-ms`
/// given without the opt-in is refused at start-up (it would configure a
/// fetch that never happens); the deadline's environment fallback is read
/// only once opted in.
///
/// Only a command that loads a projector (`--mmproj`) resolves these
/// settings — the flags need it, and the environment is read only while
/// building that command's image policy — so a stale variable in the shell
/// or in `.env` cannot stop a text-only command.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ImageSourceFlags {
    /// `--allow-image-url-fetch`.
    pub(crate) allow_image_url_fetch: bool,
    /// `--media-path <dir>` (`serve` only).
    pub(crate) media_path: Option<String>,
    /// `--image-url-timeout-ms <ms>`.
    pub(crate) image_url_timeout_ms: Option<u64>,
    /// `--image-url-allow-host <host[:port]>` (repeatable).
    pub(crate) image_url_allow_hosts: Vec<String>,
}

/// Where a media directory was named: the flag or its environment fallback,
/// for messages that must say which setting to fix.
#[cfg(feature = "server")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MediaRootSource {
    /// `--media-path`.
    Flag,
    /// The `OXI_MEDIA_PATH` environment variable.
    Env,
}

#[cfg(feature = "server")]
impl MediaRootSource {
    /// The setting's name as a user types it.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Flag => "--media-path",
            Self::Env => MEDIA_PATH_ENV,
        }
    }
}

/// A media directory as a command resolved it.
#[cfg(feature = "server")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MediaRoot {
    /// The directory, as the setting spelled it.
    pub(crate) path: std::path::PathBuf,
    /// Which setting named it.
    pub(crate) source: MediaRootSource,
}

/// `true` for the spellings every `OXI_*` on/off variable accepts.
fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Whether remote image references were opted in to: the flag, else the
/// environment value. (A flag can only turn the opt-in on: an absent flag is
/// "not given", so the environment still decides.)
fn resolve_url_fetch_opt_in(flag: bool, env: Option<&str>) -> bool {
    flag || env.is_some_and(is_truthy)
}

/// The media directory: the flag, else a non-blank environment value, else
/// none. `env` is passed in (not read here) so the rule is testable without
/// touching the process environment.
#[cfg(feature = "server")]
fn resolve_media_root(flag: Option<&str>, env: Option<&str>) -> Option<MediaRoot> {
    match (flag, env.map(str::trim).filter(|v| !v.is_empty())) {
        (Some(path), _) => Some(MediaRoot {
            path: path.into(),
            source: MediaRootSource::Flag,
        }),
        (None, Some(path)) => Some(MediaRoot {
            path: path.into(),
            source: MediaRootSource::Env,
        }),
        (None, None) => None,
    }
}

/// Which image references `run` / `chat` resolve when only the environment
/// speaks ([`ImageSourceFlags::cli_policy`] with no flag) — the tests' way
/// to get the policy a bare command would build.
#[cfg(test)]
pub(crate) fn cli_image_policy() -> oxibonsai_model::vision::ImageSourcePolicy {
    ImageSourceFlags::default()
        .cli_policy()
        .expect("the environment's image-source settings resolve")
}

/// The value of the environment variable `name`, if set.
fn env_value(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

impl ImageSourceFlags {
    /// `true` when no image-source flag was passed.
    pub(crate) fn is_empty(&self) -> bool {
        !self.allow_image_url_fetch
            && self.media_path.is_none()
            && self.image_url_timeout_ms.is_none()
            && self.image_url_allow_hosts.is_empty()
    }

    /// Whether remote references were opted in to, and by what: the flag,
    /// else the environment.
    fn url_fetch_opt_in(&self) -> Option<super::image_fetch::SettingSource> {
        if self.allow_image_url_fetch {
            Some(super::image_fetch::SettingSource::Flag)
        } else if resolve_url_fetch_opt_in(false, env_value(ALLOW_IMAGE_URL_FETCH_ENV).as_deref()) {
            Some(super::image_fetch::SettingSource::Env)
        } else {
            None
        }
    }

    /// What `run`, `chat` and `serve` do with remote image references under
    /// these flags and the environment: refuse them, or fetch them with the
    /// resolved deadline and allowlist.
    ///
    /// # Errors
    ///
    /// Without the opt-in: an allowlist (the flag or
    /// `OXI_IMAGE_URL_ALLOW_HOSTS`) or a `--image-url-timeout-ms`, which would
    /// configure nothing. With it: a deadline of 0 or not a number, or a
    /// malformed allowlist entry. Each names its setting.
    pub(crate) fn remote_fetch_report(
        &self,
    ) -> anyhow::Result<super::image_fetch::RemoteFetchReport> {
        use super::image_fetch::{
            ImageFetchSettings, RemoteFetchReport, ALLOW_HOSTS_ENV, TIMEOUT_ENV,
        };
        let hosts_env = env_value(ALLOW_HOSTS_ENV);
        let Some(from) = self.url_fetch_opt_in() else {
            if !self.image_url_allow_hosts.is_empty() {
                anyhow::bail!(
                    "--image-url-allow-host has no effect without --allow-image-url-fetch (or \
                     {ALLOW_IMAGE_URL_FETCH_ENV}=1): remote image URLs are refused, so there is \
                     nothing to exempt"
                );
            }
            if self.image_url_timeout_ms.is_some() {
                anyhow::bail!(
                    "--image-url-timeout-ms has no effect without --allow-image-url-fetch (or \
                     {ALLOW_IMAGE_URL_FETCH_ENV}=1): remote image URLs are refused, so nothing is \
                     fetched"
                );
            }
            if hosts_env.as_deref().is_some_and(|v| !v.trim().is_empty()) {
                anyhow::bail!(
                    "{ALLOW_HOSTS_ENV} is set, but remote image fetching is not enabled \
                     (--allow-image-url-fetch or {ALLOW_IMAGE_URL_FETCH_ENV}=1): an allowlist \
                     without the opt-in configures nothing; unset it, or opt in"
                );
            }
            return Ok(RemoteFetchReport::Disabled);
        };
        let settings = ImageFetchSettings::resolve(
            self.image_url_timeout_ms,
            env_value(TIMEOUT_ENV).as_deref(),
            &self.image_url_allow_hosts,
            hosts_env.as_deref(),
        )?;
        Ok(RemoteFetchReport::Enabled { from, settings })
    }

    /// The remote half of a policy: refused (no opt-in), or fetched through a
    /// new [`super::image_fetch::ImageUrlFetcher`] (its thread starts on its
    /// first fetch).
    fn remote_access(&self) -> anyhow::Result<oxibonsai_model::vision::RemoteImageAccess> {
        use super::image_fetch::{ImageUrlFetcher, RemoteFetchReport};
        use oxibonsai_model::vision::RemoteImageAccess;
        Ok(match self.remote_fetch_report()? {
            RemoteFetchReport::Disabled => RemoteImageAccess::Disabled,
            RemoteFetchReport::Enabled { settings, .. } => {
                RemoteImageAccess::Fetcher(ImageUrlFetcher::new(settings).into_shared())
            }
        })
    }

    /// Which image references `run` / `chat` resolve under these flags: the
    /// user's own local files and `data:` URIs, and remote references only
    /// with the opt-in (`--allow-image-url-fetch`, else the environment),
    /// through the fetcher of [`super::image_fetch`].
    ///
    /// # Errors
    ///
    /// As [`ImageSourceFlags::remote_fetch_report`].
    pub(crate) fn cli_policy(&self) -> anyhow::Result<oxibonsai_model::vision::ImageSourcePolicy> {
        Ok(oxibonsai_model::vision::ImageSourcePolicy::local_user()
            .with_remote_access(self.remote_access()?))
    }

    /// The directory `serve` resolves `file://` references inside:
    /// `--media-path`, else `OXI_MEDIA_PATH`, else none.
    #[cfg(feature = "server")]
    pub(crate) fn media_root(&self) -> Option<MediaRoot> {
        resolve_media_root(
            self.media_path.as_deref(),
            std::env::var(MEDIA_PATH_ENV).ok().as_deref(),
        )
    }

    /// Which image references `serve` resolves for a request: `data:` URIs,
    /// `file://` references inside the media directory when there is one,
    /// and remote references only with the opt-in, through the fetcher of
    /// [`super::image_fetch`]. The directory is canonicalised here, once, so
    /// requests are resolved against the real location whatever the process
    /// does to its working directory afterwards.
    ///
    /// # Errors
    ///
    /// A media directory that does not exist, is not a directory, or cannot
    /// be resolved — naming the setting that named it; and the errors of
    /// [`ImageSourceFlags::remote_fetch_report`].
    #[cfg(feature = "server")]
    pub(crate) fn server_policy(
        &self,
    ) -> anyhow::Result<oxibonsai_model::vision::ImageSourcePolicy> {
        let media_root = self
            .media_root()
            .map(|root| canonical_media_root(&root))
            .transpose()?;
        Ok(
            oxibonsai_model::vision::ImageSourcePolicy::server(media_root, false)
                .with_remote_access(self.remote_access()?),
        )
    }
}

/// `root` resolved to the directory it names: it must exist and be a
/// directory. Symlinks in the path itself are followed (the operator chose
/// this location); what a *request* may reach is bounded later, by the
/// resolver, on the resolved path.
///
/// # Errors
///
/// The setting's name, the path and why it is not a usable directory.
#[cfg(feature = "server")]
fn canonical_media_root(root: &MediaRoot) -> anyhow::Result<std::path::PathBuf> {
    let setting = root.source.label();
    if root.path.as_os_str().is_empty() {
        anyhow::bail!(
            "{setting} is empty: name the directory `file://` image references resolve inside"
        );
    }
    let canonical = root.path.canonicalize().map_err(|e| {
        anyhow::anyhow!("{setting} {}: cannot be resolved: {e}", root.path.display())
    })?;
    if !canonical.is_dir() {
        anyhow::bail!(
            "{setting} {}: not a directory (it must be the directory `file://` image references \
             resolve inside)",
            root.path.display()
        );
    }
    Ok(canonical)
}

// ──────────────────────────────────────────────────────────────────────────
// The image tokens against the model's vocabulary
// ──────────────────────────────────────────────────────────────────────────

/// `<|vision_start|>`, as a vocabulary spells it.
pub(crate) const VISION_START_TOKEN: &str = "<|vision_start|>";

/// `<|vision_end|>`, as a vocabulary spells it.
pub(crate) const VISION_END_TOKEN: &str = "<|vision_end|>";

/// `<|image_pad|>`, as a vocabulary spells it.
pub(crate) const IMAGE_PAD_TOKEN: &str = "<|image_pad|>";

/// The vocabulary of the language model a projector is loaded for: the
/// GGUF's `tokenizer.ggml.tokens` array (the string of every token id, in id
/// order), borrowed from the file. It is the model's own vocabulary — the one
/// its embedding table is indexed by — and what [`check_vision_markers`]
/// holds the splice layer's ids against.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelVocabulary<'a> {
    tokens: &'a [oxibonsai_core::MetadataValue],
}

impl<'a> ModelVocabulary<'a> {
    /// A vocabulary over `tokens` (entry `i` is the string of token id `i`).
    pub(crate) fn from_tokens(tokens: &'a [oxibonsai_core::MetadataValue]) -> Self {
        Self { tokens }
    }

    /// The vocabulary `gguf` declares; empty when the file carries no
    /// `tokenizer.ggml.tokens` array.
    pub(crate) fn of_gguf(gguf: &'a oxibonsai_core::gguf::reader::GgufFile<'_>) -> Self {
        Self::from_tokens(
            gguf.metadata
                .get_array(oxibonsai_core::gguf::tensor_info::keys::TOKENIZER_TOKENS)
                .unwrap_or(&[]),
        )
    }

    /// How many tokens the vocabulary holds (`0`: the model carries none).
    pub(crate) fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// The string of token `id`, when the vocabulary has such a token.
    fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(usize::try_from(id).ok()?)?.as_str()
    }

    /// The first id whose string is `token`, when the vocabulary has it.
    fn id_of(&self, token: &str) -> Option<u32> {
        self.tokens
            .iter()
            .position(|entry| entry.as_str() == Some(token))
            .and_then(|index| u32::try_from(index).ok())
    }
}

/// One image token that is not where the splice layer expects it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MarkerProblem {
    /// The token, as the vocabulary spells it.
    pub(crate) token: &'static str,
    /// The id the splice layer uses for it.
    pub(crate) splice_id: u32,
    /// The id the model's vocabulary gives it; `None` when the vocabulary has
    /// no such token at all.
    pub(crate) vocabulary_id: Option<u32>,
}

impl std::fmt::Display for MarkerProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.vocabulary_id {
            Some(found) => write!(
                f,
                "{} is id {found} in the model's vocabulary but the image splice uses id {}",
                self.token, self.splice_id
            ),
            None => write!(
                f,
                "{} is missing from the model's vocabulary (the image splice uses id {})",
                self.token, self.splice_id
            ),
        }
    }
}

/// The model's vocabulary does not give `<|vision_start|>`,
/// `<|vision_end|>` and `<|image_pad|>` the ids the splice layer uses, so a
/// prompt's image placeholders would be looked for under the wrong ids and
/// the image rows would replace the wrong tokens (or none). `--mmproj` is
/// refused rather than serving garbage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VisionMarkerError {
    /// The `--mmproj` the markers were required for.
    pub(crate) mmproj: String,
    /// How many tokens the model's vocabulary holds (`0`: the GGUF carries
    /// no vocabulary, so nothing could be checked).
    pub(crate) vocabulary_size: usize,
    /// Every marker that is missing or sits at another id (never empty).
    pub(crate) problems: Vec<MarkerProblem>,
}

impl VisionMarkerError {
    /// A short, stable code for monitoring and scripts, in the style of the
    /// `image_*` request codes.
    #[must_use]
    pub(crate) const fn code(&self) -> &'static str {
        "vision_vocabulary_mismatch"
    }
}

impl std::fmt::Display for VisionMarkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] --mmproj {}: ", self.code(), self.mmproj)?;
        if self.vocabulary_size == 0 {
            f.write_str(
                "the language model's GGUF carries no vocabulary (tokenizer.ggml.tokens), so the \
                 ids the image splice uses cannot be checked",
            )?;
        } else {
            write!(
                f,
                "the image splice does not match this model's vocabulary ({} tokens)",
                self.vocabulary_size
            )?;
        }
        f.write_str(": ")?;
        for (index, problem) in self.problems.iter().enumerate() {
            if index > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{problem}")?;
        }
        f.write_str(
            ". Images would not be spliced where the prompt marks them; use the Bonsai 2 \
             language model this projector belongs to, or drop --mmproj",
        )
    }
}

impl std::error::Error for VisionMarkerError {}

/// Hold the ids the splice layer uses for the three image tokens (`ids`, as
/// the loaded vision service reports them) against `vocabulary`: each token
/// must sit at exactly that id.
///
/// # Errors
///
/// [`VisionMarkerError`] listing EVERY token that is missing from the
/// vocabulary or sits at another id, each with both ids; an empty vocabulary
/// (a GGUF with no `tokenizer.ggml.tokens`) is refused too — nothing can be
/// verified against it.
pub(crate) fn check_vision_markers(
    mmproj: &str,
    ids: oxibonsai_model::vision::VisionTokenIds,
    vocabulary: &ModelVocabulary<'_>,
) -> Result<(), VisionMarkerError> {
    let expected = [
        (VISION_START_TOKEN, ids.vision_start),
        (VISION_END_TOKEN, ids.vision_end),
        (IMAGE_PAD_TOKEN, ids.image_pad),
    ];
    let problems: Vec<MarkerProblem> = expected
        .into_iter()
        .filter(|&(token, splice_id)| vocabulary.token(splice_id) != Some(token))
        .map(|(token, splice_id)| MarkerProblem {
            token,
            splice_id,
            vocabulary_id: vocabulary.id_of(token),
        })
        .collect();
    if problems.is_empty() {
        Ok(())
    } else {
        Err(VisionMarkerError {
            mmproj: mmproj.to_string(),
            vocabulary_size: vocabulary.token_count(),
            problems,
        })
    }
}

impl VisionRequest {
    /// `true` when no vision flag was passed at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.mmproj.is_none() && self.images.is_empty() && self.image_max_tokens.is_none()
    }

    /// The effective per-image budget (explicit value or the 1024 default).
    pub(crate) fn effective_image_max_tokens(&self) -> usize {
        self.image_max_tokens.unwrap_or(DEFAULT_IMAGE_MAX_TOKENS)
    }

    /// Check the vision flags before any model is loaded: an `--image`,
    /// `--image-max-tokens`, `--allow-image-url-fetch`,
    /// `--image-url-timeout-ms`, `--image-url-allow-host` or `--media-path`
    /// without `--mmproj` (flags that would silently do nothing), a budget
    /// out of range, an `--mmproj` that is not a `clip` projector GGUF, a
    /// media directory that is not one, the remote-image settings (an
    /// allowlist or an `--image-url-timeout-ms` without the opt-in; with it,
    /// a malformed entry or deadline), and every `--image` reference (a
    /// missing file; a remote URL
    /// that is refused — with its stable reason code — checked without any
    /// network activity: it is fetched once, when the image is prepared).
    /// A command with no vision flag and no image-source flag reads no image
    /// setting at all, the environment's included, so a stale
    /// `OXI_IMAGE_URL_*` value cannot stop a text-only command.
    /// `images_allowed` is `false` for `serve`, which takes images from
    /// requests (and is the only command with a media directory).
    ///
    /// # Errors
    ///
    /// The first problem found, naming the flag.
    pub(crate) fn validate(
        &self,
        images_allowed: bool,
        sources: &ImageSourceFlags,
    ) -> anyhow::Result<()> {
        if self.is_empty() && sources.is_empty() {
            return Ok(());
        }
        if !images_allowed && !self.images.is_empty() {
            anyhow::bail!(
                "--image is for `run` and `chat`; a server receives images as `image_url` content \
                 parts of each request"
            );
        }
        if images_allowed && sources.media_path.is_some() {
            anyhow::bail!(
                "--media-path is for `serve`: `run` and `chat` read the files you name with \
                 --image directly"
            );
        }
        if self.mmproj.is_none() {
            if !self.images.is_empty() {
                anyhow::bail!(
                    "--image needs the vision projector: pass --mmproj <mmproj GGUF> (e.g. \
                     Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf)"
                );
            }
            if sources.allow_image_url_fetch {
                anyhow::bail!(
                    "--allow-image-url-fetch has no effect without --mmproj <mmproj GGUF>"
                );
            }
            if sources.image_url_timeout_ms.is_some() {
                anyhow::bail!(
                    "--image-url-timeout-ms has no effect without --mmproj <mmproj GGUF>"
                );
            }
            if !sources.image_url_allow_hosts.is_empty() {
                anyhow::bail!(
                    "--image-url-allow-host has no effect without --mmproj <mmproj GGUF>"
                );
            }
            if sources.media_path.is_some() {
                anyhow::bail!("--media-path has no effect without --mmproj <mmproj GGUF>");
            }
            anyhow::bail!("--image-max-tokens has no effect without --mmproj <mmproj GGUF>");
        }
        let budget = self.effective_image_max_tokens();
        if !(1..=oxibonsai_model::vision::MAX_IMAGE_MAX_TOKENS).contains(&budget) {
            anyhow::bail!(
                "--image-max-tokens must be in 1..={}, got {budget}",
                oxibonsai_model::vision::MAX_IMAGE_MAX_TOKENS
            );
        }
        if let Some(path) = &self.mmproj {
            validate_mmproj(path)?;
        }
        #[cfg(feature = "server")]
        if !images_allowed {
            // The media directory, from the flag or the environment, fails
            // here — before any model is loaded — rather than as a refusal
            // of every request that names a file.
            sources.server_policy()?;
        }
        let policy = sources.cli_policy()?;
        for image in &self.images {
            validate_image_ref(image, &policy)?;
        }
        Ok(())
    }

    /// The projector check every command makes before any language-model
    /// weight is bound, and what it makes of the options a hybrid engine is
    /// built with (`HybridLoadScope`): the projector serves a `qwen35`
    /// (Bonsai 2) model only, and a Metal-backed engine's KV window leaves
    /// room for the projector's Metal tower (its resident bytes, read from
    /// the file's header and tensor types without building a tower), while
    /// the runner's calls are sized for `prefill_chunk`. Without `--mmproj`
    /// the options carry no tower. A wrong architecture or a projector the
    /// towers refuse fails here, fast; the model's image tokens are checked
    /// next ([`VisionRequest::check_image_tokens`]).
    ///
    /// # Errors
    ///
    /// A projector for anything but a `qwen35` language model — the typed
    /// `NOT_A_HYBRID_MODEL` refusal, before the model is loaded — or one the
    /// towers refuse.
    pub(crate) fn hybrid_load_options(
        &self,
        arch: &str,
        prefill_chunk: Option<usize>,
    ) -> anyhow::Result<oxibonsai_runtime::engine_hybrid_gpu::HybridLoadOptions> {
        let vision_resident_bytes = match &self.mmproj {
            None => 0,
            Some(path) => {
                check_projector_arch(path, arch)?;
                oxibonsai_runtime::vision_prefill::VisionService::metal_footprint(
                    std::path::Path::new(path),
                    self.effective_image_max_tokens(),
                )
                .map_err(anyhow::Error::from)?
            }
        };
        Ok(oxibonsai_runtime::engine_hybrid_gpu::HybridLoadOptions {
            vision_resident_bytes,
            prefill_chunk,
        })
    }

    /// The image-token check every command that loads a projector makes
    /// before any language-model weight is bound: the model's `vocabulary`
    /// holds `<|vision_start|>`, `<|vision_end|>` and `<|image_pad|>` at the
    /// ids every projector load splices with
    /// ([`oxibonsai_model::vision::VisionTokenIds::BONSAI2`], what
    /// `VisionService::token_ids` reports — held against the loaded service
    /// again by [`VisionRequest::load_service_for`]), so a model whose
    /// vocabulary disagrees is refused before a multi-GB load rather than
    /// after it. Nothing to check without `--mmproj`.
    ///
    /// # Errors
    ///
    /// A [`VisionMarkerError`] naming every token that is missing or sits at
    /// another id.
    pub(crate) fn check_image_tokens(
        &self,
        vocabulary: &ModelVocabulary<'_>,
    ) -> Result<(), VisionMarkerError> {
        match &self.mmproj {
            Some(path) => check_vision_markers(
                path,
                oxibonsai_model::vision::VisionTokenIds::BONSAI2,
                vocabulary,
            ),
            None => Ok(()),
        }
    }

    /// Load the vision projector for `engine` (a `qwen35` language model of
    /// architecture `arch` and vocabulary `vocabulary`), resolving request
    /// images under the policy `policy` builds: the tower of the executor the
    /// engine decodes on — the Metal tower for an engine on the Metal hybrid
    /// runner, the CPU tower otherwise, never both — with the engine and the
    /// projector checked together before any image is encoded
    /// (`VisionService::load_for_engine`: an engine that refuses image
    /// turns, or a projector whose rows are not as wide as the model's, is
    /// refused here, typed). `None` when no `--mmproj` was given.
    ///
    /// `policy` is called only when a projector is loaded: a text-only
    /// command never builds an image policy, so it never reads the
    /// remote-image settings (`OXI_ALLOW_IMAGE_URL_FETCH`,
    /// `OXI_IMAGE_URL_TIMEOUT_MS`, `OXI_IMAGE_URL_ALLOW_HOSTS`), and a stale
    /// or malformed value left in the shell or in `.env` cannot stop it. With
    /// `--mmproj` the same settings were already checked before the model
    /// loaded ([`VisionRequest::validate`]), so a bad one never costs a model
    /// load.
    ///
    /// The policy's decode budget follows `--image-max-tokens`
    /// ([`oxibonsai_model::vision::ImageSourcePolicy::with_token_budget`]): a
    /// source image with more pixels than the grid it is resized to can use
    /// (with the documented headroom) is refused as `image_too_large` from its
    /// header, before it is inflated — a small file can declare a huge image.
    /// A policy that already carries a budget keeps it.
    ///
    /// The ids the loaded service splices images in with
    /// (`VisionService::token_ids`) are compiled-in constants of the Bonsai 2
    /// vocabulary; they are held against `vocabulary` here
    /// ([`check_vision_markers`]), so a model whose vocabulary places
    /// `<|vision_start|>`, `<|vision_end|>` or `<|image_pad|>` elsewhere is
    /// refused at load instead of splicing images over the wrong tokens.
    ///
    /// # Errors
    ///
    /// The error `policy` returns; a projector for anything but a `qwen35`
    /// (Bonsai 2) language model (`NOT_A_HYBRID_MODEL`), the engine's own
    /// refusal of image turns, the projector's own load error (verbatim: the
    /// tower refuses a variant projector rather than guessing), a projector
    /// that does not fit the model (`projector_mismatch`), or a
    /// [`VisionMarkerError`] when the vocabulary disagrees with the splice
    /// layer's ids.
    pub(crate) fn load_service_for(
        &self,
        arch: &str,
        vocabulary: &ModelVocabulary<'_>,
        policy: impl FnOnce() -> anyhow::Result<oxibonsai_model::vision::ImageSourcePolicy>,
        engine: &oxibonsai_runtime::InferenceEngine<'_>,
    ) -> anyhow::Result<Option<std::sync::Arc<oxibonsai_runtime::vision_prefill::VisionService>>>
    {
        self.load_with(arch, vocabulary, policy, |path, budget, policy| {
            oxibonsai_runtime::vision_prefill::VisionService::load_for_engine(
                path, budget, policy, engine,
            )
        })
    }

    /// [`VisionRequest::load_service_for`] without an engine: the CPU tower,
    /// checked against the architecture and the vocabulary only — the form
    /// the unit tests drive the shared loader through.
    #[cfg(test)]
    pub(crate) fn load_service(
        &self,
        arch: &str,
        vocabulary: &ModelVocabulary<'_>,
        policy: oxibonsai_model::vision::ImageSourcePolicy,
    ) -> anyhow::Result<Option<std::sync::Arc<oxibonsai_runtime::vision_prefill::VisionService>>>
    {
        self.load_with(
            arch,
            vocabulary,
            move || Ok(policy),
            oxibonsai_runtime::vision_prefill::VisionService::load,
        )
    }

    /// The loader both forms share: the architecture check, the image
    /// policy (built here, only once a projector is known to be loaded), its
    /// token budget, the load, the vocabulary check and the log line naming
    /// the tower's executor and what it keeps resident.
    fn load_with(
        &self,
        arch: &str,
        vocabulary: &ModelVocabulary<'_>,
        policy: impl FnOnce() -> anyhow::Result<oxibonsai_model::vision::ImageSourcePolicy>,
        load: impl FnOnce(
            &std::path::Path,
            usize,
            oxibonsai_model::vision::ImageSourcePolicy,
        ) -> oxibonsai_runtime::error::RuntimeResult<
            oxibonsai_runtime::vision_prefill::VisionService,
        >,
    ) -> anyhow::Result<Option<std::sync::Arc<oxibonsai_runtime::vision_prefill::VisionService>>>
    {
        let Some(path) = &self.mmproj else {
            return Ok(None);
        };
        check_projector_arch(path, arch)?;
        let policy = policy()?;
        let policy = if policy.max_source_pixels.is_none() {
            policy.with_token_budget(self.effective_image_max_tokens())
        } else {
            policy
        };
        let started = std::time::Instant::now();
        let service = load(
            std::path::Path::new(path),
            self.effective_image_max_tokens(),
            policy,
        )
        .map_err(anyhow::Error::from)?;
        check_vision_markers(path, service.token_ids(), vocabulary)?;
        tracing::info!(
            mmproj = %path,
            backend = service.tower().backend_name(),
            blocks = service.tower().block_count(),
            resident_bytes = service.tower().resident_bytes(),
            image_max_tokens = self.effective_image_max_tokens(),
            max_source_pixels = ?service.policy().max_source_pixels,
            vocabulary_tokens = vocabulary.token_count(),
            seconds = started.elapsed().as_secs_f64(),
            "vision projector loaded on the {} tower; its image tokens match the model's \
             vocabulary",
            service.tower().backend_name()
        );
        Ok(Some(std::sync::Arc::new(service)))
    }

    /// Resolve, decode and preprocess every `--image` with `service`, in
    /// order: everything but the (expensive) tower encode, so the prompt's
    /// expanded length can be checked against the context window before
    /// any image is encoded.
    ///
    /// # Errors
    ///
    /// The first image's error, with its stable reason code:
    /// `[<code>] --image <ref>: <reason>`.
    pub(crate) fn prepare_images(
        &self,
        service: &oxibonsai_runtime::vision_prefill::VisionService,
    ) -> anyhow::Result<Vec<oxibonsai_model::vision::PreparedImage>> {
        self.images
            .iter()
            .enumerate()
            .map(|(i, image)| {
                service
                    .prepare_reference(i, image)
                    .map_err(|e| anyhow::anyhow!("[{}] --image {}: {e}", e.code(), shorten(image)))
            })
            .collect()
    }

    /// Encode the images [`VisionRequest::prepare_images`] prepared, in
    /// order, reporting each one's geometry and encode time on stderr.
    ///
    /// # Errors
    ///
    /// The first tower failure, with its stable reason code.
    pub(crate) fn encode_prepared(
        &self,
        service: &oxibonsai_runtime::vision_prefill::VisionService,
        prepared: &[oxibonsai_model::vision::PreparedImage],
    ) -> anyhow::Result<Vec<oxibonsai_runtime::vision_prefill::EncodedImage>> {
        prepared
            .iter()
            .enumerate()
            .map(|(i, image)| {
                let started = std::time::Instant::now();
                let reference = self.images.get(i).map_or("", String::as_str);
                let encoded = service.encode_prepared(i, image).map_err(|e| {
                    anyhow::anyhow!("[{}] --image {}: {e}", e.code(), shorten(reference))
                })?;
                eprintln!(
                    "[image {}: {} x {} -> {} x {} merged grid ({} image tokens) in {:.2}s]",
                    i + 1,
                    encoded.source.0,
                    encoded.source.1,
                    encoded.grid.h,
                    encoded.grid.w,
                    encoded.grid.n_tokens(),
                    started.elapsed().as_secs_f64()
                );
                Ok(encoded)
            })
            .collect()
    }
}

/// Sequence positions `tokens` occupies once every `<|image_pad|>` is
/// replaced by its image's rows (`grids`, in placeholder order): the splice
/// plan's row count, or the plain token count for a prompt that holds no
/// placeholder (a text-only turn, or a conversation whose image message was
/// dropped to fit the context).
///
/// # Errors
///
/// A prompt whose placeholders do not match the images
/// (`[<code>] <reason>`).
pub(crate) fn prompt_rows(
    tokens: &[u32],
    grids: &[oxibonsai_model::vision::GridSize],
    ids: oxibonsai_model::vision::VisionTokenIds,
) -> anyhow::Result<usize> {
    if !tokens.contains(&ids.image_pad) {
        return Ok(tokens.len());
    }
    oxibonsai_model::vision::plan_splice(tokens, grids, ids)
        .map(|plan| plan.total_rows())
        .map_err(|e| anyhow::anyhow!("[{}] {e}", e.code()))
}

/// An image reference shortened for a message (a data URI can be
/// megabytes).
fn shorten(reference: &str) -> String {
    const MAX: usize = 96;
    if reference.chars().count() <= MAX {
        reference.to_string()
    } else {
        let head: String = reference.chars().take(MAX).collect();
        format!("{head}...")
    }
}

/// A Bonsai 2 vision projector serves only a `qwen35` (Bonsai 2) language
/// model: its image rows enter the hybrid model's rows prefill, which a dense
/// model does not have. Any other architecture is refused with the engine's
/// own `NOT_A_HYBRID_MODEL` code — before the language model is loaded, so no
/// image is ever decoded for it.
///
/// # Errors
///
/// `[NOT_A_HYBRID_MODEL] --mmproj <path>: ...` naming both architectures.
fn check_projector_arch(path: &str, arch: &str) -> anyhow::Result<()> {
    if is_qwen35_hybrid(arch) {
        return Ok(());
    }
    // The engine's own code for the refusal, so the CLI and the server name
    // it the same way.
    let code = oxibonsai_runtime::engine_seam::EngineError::NotAHybridModel {
        operation: oxibonsai_runtime::vision_prefill::MULTIMODAL_PREFILL_OPERATION,
        architecture: arch.to_string(),
    }
    .error_code();
    anyhow::bail!(
        "[{code}] --mmproj {path}: the Bonsai 2 vision projector serves a `qwen35` (Bonsai 2) \
         language model, but this model's architecture is '{arch}': image input needs a \
         hybrid model's rows prefill"
    )
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

/// `--image` must be a `data:` URI, a readable local file, or — with the
/// opt-in (see [`ALLOW_IMAGE_URL_FETCH_ENV`]) — a remote URL the fetcher's
/// policy does not refuse. Nothing is fetched here: a remote URL is checked
/// for everything that needs no network (its syntax, a literal address,
/// `localhost`, the allowlist) and fetched once, when the image is prepared.
fn validate_image_ref(
    image: &str,
    policy: &oxibonsai_model::vision::ImageSourcePolicy,
) -> anyhow::Result<()> {
    use oxibonsai_model::vision::image_decode::{classify_image_source, ImageSource};
    let refused = |e: oxibonsai_model::vision::ImageInputError| {
        anyhow::anyhow!("[{}] --image {}: {e}", e.code(), shorten(image))
    };
    match classify_image_source(image).map_err(refused)? {
        ImageSource::Remote => match policy.remote.fetcher() {
            Some(fetcher) => fetcher.fetcher().preflight(image).map_err(refused),
            // Not fetched at all: resolving it produces the typed refusal
            // before anything is opened.
            None => oxibonsai_model::vision::load_image_bytes(image, policy)
                .map(drop)
                .map_err(refused),
        },
        ImageSource::DataUri => oxibonsai_model::vision::parse_data_uri(image)
            .map(|_| ())
            .map_err(refused),
        ImageSource::LocalFile(path) => {
            let meta = std::fs::metadata(&path)
                .map_err(|e| anyhow::anyhow!("--image {path}: cannot read: {e}"))?;
            if !meta.is_file() {
                anyhow::bail!("--image {path}: not a regular file");
            }
            Ok(())
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// `--prefill-chunk <N>` (design §5.7)
// ──────────────────────────────────────────────────────────────────────────

/// Which model's prefill chunk `--prefill-chunk` set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefillChunkKind {
    /// The Gated-DeltaNet prefill chunk of a hybrid (`qwen35`) model.
    Hybrid,
    /// The chunked-prefill size of a dense model.
    Dense,
}

/// What `--prefill-chunk` came to once the engine had it: the value that was
/// asked for, the value the model reports holding afterwards, and the most
/// tokens one prefill call of the engine's executor takes for it
/// (`InferenceEngine::prefill_chunk_in_effect`). The model holds a different
/// value when it clamps or rounds a request; the executor takes fewer tokens
/// per call than the model holds when its KV window's memory budget leaves
/// no room for larger calls (the Metal hybrid runner sizes its activation
/// scratch for one call). The log line reports the value in effect and names
/// the request beside it whenever the two differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillChunkOutcome {
    /// Which model's chunk it is.
    pub(crate) kind: PrefillChunkKind,
    /// `--prefill-chunk` as given.
    pub(crate) requested: usize,
    /// What the model holds after the set, read back from the model.
    pub(crate) honoured: usize,
    /// The most tokens one prefill call takes on the engine's executor.
    pub(crate) in_effect: usize,
}

impl PrefillChunkOutcome {
    /// Whether the executor takes fewer tokens per prefill call than the
    /// model holds: its KV window's memory budget capped the request.
    #[must_use]
    pub(crate) fn capped_by_executor(&self) -> bool {
        self.in_effect < self.honoured
    }

    /// The one line `--prefill-chunk` logs (at `WARN` when
    /// [`Self::capped_by_executor`], else `INFO`): the chunk in effect, and —
    /// only when it is not the request — the value that was asked for and
    /// what changed it.
    #[must_use]
    pub(crate) fn message(&self) -> String {
        let what = match self.kind {
            PrefillChunkKind::Hybrid => "hybrid Gated-DeltaNet prefill chunk",
            PrefillChunkKind::Dense => "dense chunked-prefill size",
        };
        if self.capped_by_executor() {
            format!(
                "{what} in effect is {} tokens per prefill call (--prefill-chunk asked for {}): \
                 the engine's executor takes calls of at most {} tokens, since larger calls \
                 would not leave room for its KV window in the memory budget",
                self.in_effect, self.requested, self.in_effect
            )
        } else if self.requested == self.in_effect {
            format!("{what} set to {} tokens", self.in_effect)
        } else {
            format!(
                "{what} is {} tokens (--prefill-chunk asked for {}; the model adjusted it)",
                self.in_effect, self.requested
            )
        }
    }
}

/// A model whose prefill chunk `--prefill-chunk` sets, and which reports what
/// it holds once it has been set.
trait PrefillChunkTarget {
    /// Which model this is.
    fn kind(&self) -> PrefillChunkKind;

    /// Ask the model to use `chunk` (it may clamp or round it).
    fn set_chunk(&mut self, chunk: usize) -> anyhow::Result<()>;

    /// The chunk the model holds now.
    fn held_chunk(&self) -> usize;
}

impl PrefillChunkTarget for oxibonsai_model::hybrid::HybridModel<'_> {
    fn kind(&self) -> PrefillChunkKind {
        PrefillChunkKind::Hybrid
    }

    fn set_chunk(&mut self, chunk: usize) -> anyhow::Result<()> {
        self.set_prefill_chunk(chunk)
            .map_err(|e| anyhow::anyhow!("--prefill-chunk {chunk}: {e}"))
    }

    fn held_chunk(&self) -> usize {
        self.prefill_chunk()
    }
}

impl PrefillChunkTarget for oxibonsai_model::model::BonsaiModel<'_> {
    fn kind(&self) -> PrefillChunkKind {
        PrefillChunkKind::Dense
    }

    fn set_chunk(&mut self, chunk: usize) -> anyhow::Result<()> {
        anyhow::ensure!(chunk >= 1, "--prefill-chunk must be >= 1");
        self.set_prefill_chunk_tokens(chunk);
        Ok(())
    }

    fn held_chunk(&self) -> usize {
        self.prefill_chunk_tokens()
    }
}

/// Set `requested` on `target`, then read back what it holds — the value
/// reported to the user is the model's, never an echo of the flag. Without an
/// executor of its own the model's chunk is the one in effect.
fn set_prefill_chunk_on<T: PrefillChunkTarget>(
    target: &mut T,
    requested: usize,
) -> anyhow::Result<PrefillChunkOutcome> {
    target.set_chunk(requested)?;
    let honoured = target.held_chunk();
    Ok(PrefillChunkOutcome {
        kind: target.kind(),
        requested,
        honoured,
        in_effect: honoured,
    })
}

/// Apply `--prefill-chunk` to a loaded engine: the Gated-DeltaNet prefill
/// chunk of a hybrid model (`HybridModel::set_prefill_chunk`, default
/// 512), or the chunked-prefill plan of a dense one
/// (`BonsaiModel::set_prefill_chunk_tokens`). `None` keeps the model's own
/// default (and returns `None`).
///
/// A hybrid engine is also built with the chunk (`HybridLoadOptions::
/// prefill_chunk`, see [`VisionRequest::hybrid_load_options`]), so a Metal
/// runner sizes its calls for it at construction; setting it on the model
/// again here is idempotent. Logs the chunk IN EFFECT — the most tokens one
/// prefill call of the engine's executor takes
/// (`InferenceEngine::prefill_chunk_in_effect`), never an echo of the flag
/// — with the requested value beside it when the two differ: at `WARN` when
/// the executor's KV-window memory budget capped it, at `INFO` otherwise.
/// Returns that outcome.
///
/// # Errors
///
/// The model's own refusal (a zero chunk — already rejected by the flag
/// parser, re-checked here).
pub(crate) fn apply_prefill_chunk(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prefill_chunk: Option<usize>,
) -> anyhow::Result<Option<PrefillChunkOutcome>> {
    let Some(requested) = prefill_chunk else {
        return Ok(None);
    };
    let outcome = if let Some(hybrid) = engine.hybrid_model_mut() {
        set_prefill_chunk_on(hybrid, requested)?
    } else if let Some(dense) = engine.dense_model_mut() {
        set_prefill_chunk_on(dense, requested)?
    } else {
        return Ok(None);
    };
    let outcome = PrefillChunkOutcome {
        in_effect: engine.prefill_chunk_in_effect(),
        ..outcome
    };
    if outcome.capped_by_executor() {
        tracing::warn!(
            requested = outcome.requested,
            honoured = outcome.honoured,
            in_effect = outcome.in_effect,
            "{}",
            outcome.message()
        );
    } else {
        tracing::info!(
            requested = outcome.requested,
            honoured = outcome.honoured,
            in_effect = outcome.in_effect,
            "{}",
            outcome.message()
        );
    }
    Ok(Some(outcome))
}

/// Also shared with the `run` / `chat` tests: the synthetic projector and
/// the golden fixture image.
#[cfg(test)]
#[path = "bonsai2_tests.rs"]
pub(crate) mod tests;

/// The image tokens against the model's vocabulary.
#[cfg(test)]
#[path = "bonsai2_markers_tests.rs"]
mod markers_tests;

/// `--prefill-chunk`.
#[cfg(test)]
#[path = "bonsai2_prefill_chunk_tests.rs"]
mod prefill_chunk_tests;
