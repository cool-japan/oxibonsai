//! Engine construction and prompt budgeting for commands that may carry
//! images (`run`, `chat` and `serve`, with `--mmproj`).
//!
//! # The backend a vision command runs on
//!
//! A hybrid (`qwen35`) engine decodes on the Metal hybrid runner or on the CPU
//! model, and the engine itself says whether its executor prefills image
//! rows ([`InferenceEngine::prefills_images`], from its pre-encode check
//! [`InferenceEngine::multimodal_executor`]) — both executors do, so an image
//! turn runs on the executor `--backend auto` resolved to, with the vision
//! tower of that executor. [`plan_vision_backend`] keeps the seam for an
//! executor that could not: build the engine as usual, and when vision is
//! requested under `--backend auto` and the predicate is `false`, rebuild it
//! on the CPU model ([`VisionBackendPlan::RebuildOnCpu`]) and say why.
//! Nothing here names a backend to decide that: the predicate is the single
//! seam.
//!
//! An **explicit** `--backend metal` (or `cpu`) is never overridden: an
//! engine that cannot serve image turns keeps its own typed refusal
//! ([`require_image_capable_engine`] raises it before any image is encoded,
//! instead of after).
//!
//! # The room a prompt leaves
//!
//! [`check_generation_room`] is the one check `run` applies to a prompt's
//! sequence positions against the context window, for a text prompt and for
//! a prompt with image rows alike (before any image is encoded).

use oxibonsai_runtime::engine_seam::Backend;
use oxibonsai_runtime::InferenceEngine;

#[cfg(feature = "server")]
use super::bonsai2;

/// The image policy a server resolves `image_url` parts under: `None` unless
/// `--mmproj` was given.
///
/// The policy (which resolves the media directory, from `--media-path` or the
/// `OXI_MEDIA_PATH` environment variable, and the remote-image settings, from
/// their flags or `OXI_ALLOW_IMAGE_URL_FETCH`, `OXI_IMAGE_URL_TIMEOUT_MS` and
/// `OXI_IMAGE_URL_ALLOW_HOSTS`) is built only for a server that loads a
/// projector: a text-only server never consults it, so a stale variable left
/// in a shell or in `.env` cannot stop a server that serves no images.
/// `serve` resolves it before any language-model weight is bound, so a media
/// directory that does not resolve fails fast.
///
/// # Errors
///
/// A media directory that does not resolve, naming the setting; a
/// remote-image setting `ImageSourceFlags::remote_fetch_report` refuses,
/// naming it.
#[cfg(feature = "server")]
pub(crate) fn serve_image_policy(
    vision: &bonsai2::VisionRequest,
    sources: &bonsai2::ImageSourceFlags,
) -> anyhow::Result<Option<oxibonsai_model::vision::ImageSourcePolicy>> {
    if vision.mmproj.is_none() {
        return Ok(None);
    }
    sources.server_policy().map(Some)
}

/// The server's projector loading without an engine: the CPU tower, under
/// the server policy ([`serve_image_policy`]), checked against the
/// architecture and the vocabulary — the form the unit tests drive it
/// through. `serve` itself loads the tower for its pool's executor
/// (`VisionRequest::load_service_for` on one replica).
#[cfg(all(test, feature = "server"))]
pub(crate) fn serve_vision_service(
    vision: &bonsai2::VisionRequest,
    sources: &bonsai2::ImageSourceFlags,
    arch: &str,
    vocabulary: &bonsai2::ModelVocabulary<'_>,
) -> anyhow::Result<Option<std::sync::Arc<oxibonsai_runtime::vision_prefill::VisionService>>> {
    match serve_image_policy(vision, sources)? {
        Some(policy) => vision.load_service(arch, vocabulary, policy),
        None => Ok(None),
    }
}

/// What the presence of vision does to the backend a command asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VisionBackendPlan {
    /// Keep the engine as built: no vision was requested, the backend was
    /// named explicitly, or the engine already prefills image rows.
    Keep,
    /// `--backend auto` resolved to an executor that cannot prefill image
    /// rows: rebuild the engine on the CPU model.
    RebuildOnCpu,
}

/// Decide whether a built engine must be rebuilt for a command that carries
/// vision.
///
/// * `requested` — the backend the command asked for (`--backend`, else
///   `auto`).
/// * `wants_vision` — the command loaded a projector (`--mmproj`) for a
///   hybrid engine.
/// * `engine_prefills_images` — the engine's own answer,
///   `InferenceEngine::prefills_images()`.
#[must_use]
pub(crate) fn plan_vision_backend(
    requested: Backend,
    wants_vision: bool,
    engine_prefills_images: bool,
) -> VisionBackendPlan {
    if wants_vision && requested == Backend::Auto && !engine_prefills_images {
        VisionBackendPlan::RebuildOnCpu
    } else {
        VisionBackendPlan::Keep
    }
}

/// Why the rebuild happens, for the one `INFO` line it logs: what happened and
/// what the user can do about it, in terms of flags and executor names. The
/// executor is the one the engine actually resolved to (`executor`), so the
/// line stays true for whichever runner cannot take image input.
#[must_use]
pub(crate) fn rebuild_reason(executor: &str) -> String {
    format!(
        "--mmproj is loaded and --backend auto chose this model's {executor} runner, which cannot \
         take image input yet; using the CPU engine for this session instead. Pass --backend \
         {executor} to keep that runner and have image turns refused, or --backend cpu to skip \
         building the {executor} engine first"
    )
}

/// The executor `engine` decodes on, as a word (`"cpu"`, `"metal"`).
#[must_use]
pub(crate) fn executor_name(engine: &InferenceEngine<'_>) -> String {
    engine
        .hybrid_backend()
        .map_or_else(|| engine.backend().to_string(), |b| b.to_string())
}

/// Refuse, before any image is encoded, an engine that cannot serve an image
/// turn — the engine's own typed refusal from its pre-encode check
/// ([`InferenceEngine::multimodal_executor`]: `NOT_A_HYBRID_MODEL` for a
/// dense engine, `BACKEND_UNAVAILABLE` naming the executor the engine decodes
/// on — never the backend it was asked for — for an executor without a rows
/// prefill), raised where the command still has the projector's encode ahead
/// of it rather than after it.
///
/// # Errors
///
/// The engine's typed refusal.
pub(crate) fn require_image_capable_engine(engine: &InferenceEngine<'_>) -> anyhow::Result<()> {
    engine
        .multimodal_executor()
        .map(drop)
        .map_err(anyhow::Error::from)
}

/// The same check as a `serve` start-up line: a server started with an
/// explicit backend that cannot prefill image rows would answer every image
/// request with the typed refusal, which the operator should hear once at
/// start-up instead of per request.
#[cfg(feature = "server")]
#[must_use]
pub(crate) fn startup_backend_warning(
    requested: Backend,
    engine_prefills_images: bool,
) -> Option<String> {
    (requested != Backend::Auto && !engine_prefills_images).then(|| {
        format!(
            "--mmproj is loaded but --backend {requested} decodes on a runner that cannot prefill \
             image rows: every image request will be refused with BACKEND_UNAVAILABLE. Serve \
             vision with --backend auto or --backend cpu"
        )
    })
}

/// A prompt of `prompt_rows` sequence positions against a context of
/// `max_context`, for a request of `max_tokens` new tokens: refuse the
/// prompts that leave nothing to generate into, before any encode or prefill
/// is spent on them. `images` is how many images the prompt's rows include
/// (`0` for a text prompt); it only changes the advice, which then names
/// `--image-max-tokens`.
///
/// * `prompt_rows > max_context` — the prompt alone does not fit (image rows
///   counted, placeholders expanded): a hard error naming both numbers, as
///   [`super::util::clamp_generation_budget`] reports it.
/// * `prompt_rows == max_context` with `max_tokens >= 1` — the prompt fills
///   the window exactly, so no position is left for even one generated
///   token; prefilling it (minutes, for an image prompt on the 27B) could
///   only end in an empty answer.
///
/// A prompt that fits but leaves less room than `max_tokens` is not refused:
/// the generation is clamped to the room left ([`super::util::clamp_generation_budget`]),
/// which is how the CLI treats an over-large `--max-tokens` everywhere.
///
/// # Errors
///
/// The two cases above, each naming the numbers.
pub(crate) fn check_generation_room(
    prompt_rows: usize,
    max_tokens: usize,
    max_context: usize,
    images: usize,
) -> anyhow::Result<()> {
    let (what, advice) = if images > 0 {
        (
            " (the prompt with its image rows)",
            "lower --image-max-tokens or raise --ctx",
        )
    } else {
        ("", "shorten it or raise --ctx")
    };
    if prompt_rows > max_context {
        anyhow::bail!(
            "sequence length {prompt_rows}{what} exceeds max context {max_context}: the prompt \
             alone does not fit; {advice}"
        );
    }
    if prompt_rows == max_context && max_tokens >= 1 {
        anyhow::bail!(
            "sequence length {prompt_rows}{what} fills max context {max_context} exactly: no \
             position is left to generate {max_tokens} token(s) into; {advice}"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "vision_tests.rs"]
mod tests;
