//! Multimodal (image + text) prefill and generation for a hybrid Bonsai 2
//! engine (bonsai2-design.md §6.2; SV-11).
//!
//! # The pipeline
//!
//! 1. The chat template renders each image part as
//!    `<|vision_start|><|image_pad|><|vision_end|>`; the prompt is tokenized
//!    as usual, so it holds one `<|image_pad|>` per image.
//! 2. [`VisionService`] turns every image reference into merged embedding
//!    rows: resolve (`data:` URI / local file / remote URL through the
//!    policy's fetcher, under an [`ImageSourcePolicy`]; a server hands each
//!    request its own view of the service whose fetcher reports to it,
//!    [`VisionService::with_fetch_observer`]), decode, preprocess exactly like the reference
//!    (smart resize to the 32-pixel grid within the `--image-max-tokens`
//!    budget), and encode with the Qwen3-VL tower.
//! 3. [`MultimodalPrompt`] checks the splice (one bracketed placeholder per
//!    image, in order) and cuts the prompt into [`PromptSegment`]s.
//! 4. [`InferenceEngine::prefill_multimodal`] runs the segments through the
//!    rows prefill of the executor the engine decodes on: text rows are
//!    looked up and inverse-Hadamard-transformed, image rows enter block 0
//!    verbatim, every row carries its 3-axis M-RoPE position.
//! 5. Decoding continues exactly like a text prompt's
//!    ([`InferenceEngine::generate_multimodal`] and its logprobs /
//!    streaming forms): same sampler, penalties, stop-on-EOS, cancellation
//!    and metrics as `generate` / `generate_with_logprobs` /
//!    `generate_streaming`, from the sequence position after the last
//!    prompt row (the model rotates later tokens at its M-RoPE offset).
//!
//! # Backends
//!
//! The rows prefill runs where the engine's hybrid decode runs, so one
//! sequence never spans two executors:
//!
//! * **CPU** (`--backend cpu`, or `auto` without a usable Metal runner):
//!   `HybridModel::forward_prefill_pieces`, which embeds, positions and
//!   prefills in one call.
//! * **Metal** (the hybrid runner): the CPU model assembles the rows
//!   (`HybridModel::assemble_prompt_at` — the same text embedding) at the
//!   runner's own rope start, and the runner's rows prefill
//!   (`HybridMetalRunner::forward_prefill_rows`) runs them and keeps the
//!   M-RoPE offset its later decode steps rotate by.
//!
//! [`VisionService`] follows the same split: it is built for the engine's
//! [`HybridBackend`] ([`VisionService::load_for_engine`] /
//! [`VisionService::load_for_backend`]), so a Metal engine loads only the
//! Metal tower (the file's `Q8_0` / `F16` weights as stored on the device,
//! 0.87 GiB resident for Bonsai 2) and never the CPU tower's 1.84 GB of
//! `f32` weights.
//!
//! # Refusals come before the encode
//!
//! Both hybrid executors run the rows prefill, so the only engine that
//! cannot serve an image turn is a dense one. [`InferenceEngine::multimodal_executor`]
//! answers which executor a multimodal prefill runs on — or the typed
//! [`EngineError::NotAHybridModel`] — without touching any state, so a
//! caller checks it before it spends a vision-tower encode on a request;
//! [`InferenceEngine::prefill_multimodal`] checks it first as well.
//! [`VisionService::check_engine`] adds the one pairing a projector and an
//! engine can get wrong (rows of another width than the model's residual
//! stream, [`MultimodalError::ProjectorMismatch`], naming the executor), and
//! [`VisionService::load_for_engine`] runs both checks when the projector
//! is loaded.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_model::hybrid::LoadedModel;
use oxibonsai_model::hybrid::PromptPiece;
use oxibonsai_model::vision::{
    load_image_source, plan_splice, prepare_image, GridSize, ImageInputError, ImageSourcePolicy,
    PreparedImage, PreprocessConfig, SharedRemoteImageFetcher, SpliceError, SpliceSegment,
    VisionConfig, VisionEncoder, VisionTokenIds, VisionTower,
};

use crate::engine::InferenceEngine;
use crate::engine_hybrid_gpu::HybridBackend;
use crate::engine_seam::{Backend, EngineError};
use crate::error::{RuntimeError, RuntimeResult};

mod remote_fetch;

use remote_fetch::ObservedFetcher;
pub use remote_fetch::{CurrentFetch, ImageFetchObserver};

/// The chat-template content types a multimodal user turn is built from
/// (an image part renders as the template's placeholder).
pub use oxibonsai_tokenizer::chat_templates::{RenderContent, RenderContentPart};

/// Images one request may carry: each is a full vision-tower encode (up to
/// tens of seconds on the CPU for a large image), so the count is bounded.
pub const MAX_IMAGES_PER_REQUEST: usize = 16;

/// The operation [`EngineError::NotAHybridModel`] names for a multimodal
/// request on a dense engine.
pub const MULTIMODAL_PREFILL_OPERATION: &str = "multimodal (image) prefill";

/// One stretch of a multimodal prompt, in prompt order.
#[derive(Debug, Clone, PartialEq)]
pub enum PromptSegment {
    /// Text token ids (no image placeholder among them).
    Text(Vec<u32>),
    /// One image's merged rows (`grid.h * grid.w` rows of the model's hidden
    /// width, row-major) in place of its `<|image_pad|>`.
    Image {
        /// `grid.n_tokens() * hidden` floats, the vision tower's output.
        rows: Vec<f32>,
        /// The merged grid the rows cover.
        grid: GridSize,
    },
}

impl PromptSegment {
    /// Sequence positions the segment occupies.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Text(tokens) => tokens.len(),
            Self::Image { grid, .. } => grid.n_tokens(),
        }
    }

    /// Whether the segment occupies no position.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One encoded image: the tower's merged rows and their grid.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedImage {
    /// `grid.n_tokens() * projection_dim` floats.
    pub rows: Vec<f32>,
    /// The merged grid (`height / 32 x width / 32` after preprocessing).
    pub grid: GridSize,
    /// The source image size, `(width, height)`.
    pub source: (usize, usize),
}

/// Why a multimodal request cannot run: every variant is a property of the
/// request or of how the server was started, with a stable
/// [`MultimodalError::code`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum MultimodalError {
    /// The request carries images but no vision projector is loaded.
    #[error(
        "this request contains image content, but no vision projector is loaded; start the \
         server (or command) with --mmproj <Bonsai 2 mmproj GGUF>"
    )]
    VisionUnavailable,
    /// The engine holds a dense model: image rows need the hybrid model's
    /// rows prefill.
    #[error(
        "multimodal prefill needs a hybrid `qwen35` (Bonsai 2) model, but this engine holds a \
         dense `{architecture}` model"
    )]
    NotAHybridModel {
        /// `general.architecture` of the loaded model.
        architecture: String,
    },
    /// Too many images in one request.
    #[error("the request carries {images} images; at most {limit} are accepted per request")]
    TooManyImages {
        /// Images in the request.
        images: usize,
        /// [`MAX_IMAGES_PER_REQUEST`].
        limit: usize,
    },
    /// An image reference could not become tower-ready pixels.
    #[error("image {index}: {source}")]
    Image {
        /// Image index in the request.
        index: usize,
        /// Why.
        #[source]
        source: ImageInputError,
    },
    /// The remote-image fetcher refused to fetch image `index` because it is
    /// at capacity (the most fetches it runs and queues at once): nothing is
    /// wrong with the request, and the same request can succeed shortly.
    /// Reported only through a request's view of the service
    /// ([`VisionService::with_fetch_observer`]); a server answers it with a
    /// retryable `503`.
    #[error(
        "image {index}: the remote-image fetcher is at capacity, so the image was not fetched; \
         retry after a short backoff ({source})"
    )]
    FetchOverloaded {
        /// Image index in the request.
        index: usize,
        /// The fetcher's refusal.
        #[source]
        source: ImageInputError,
    },
    /// The rendered prompt and the images do not splice.
    #[error(transparent)]
    Splice(#[from] SpliceError),
    /// The vision tower refused an image or failed.
    #[error("image {index}: the vision tower failed: {reason}")]
    Encode {
        /// Image index in the request.
        index: usize,
        /// The tower's error.
        reason: String,
    },
    /// The loaded projector's rows are not as wide as the model's residual
    /// stream, so no image it encodes can enter the model: a projector for
    /// another model was loaded. Raised by [`VisionService::check_engine`]
    /// before any image is encoded.
    #[error(
        "the vision projector emits {projector_width}-wide image rows, but the hybrid model \
         (decoding on the {executor}) reads {model_width}-wide rows: load the projector that \
         belongs to this model"
    )]
    ProjectorMismatch {
        /// `clip.vision.projection_dim` of the projector.
        projector_width: usize,
        /// The language model's hidden width.
        model_width: usize,
        /// The executor the engine decodes on ([`executor_label`]).
        executor: &'static str,
    },
}

impl MultimodalError {
    /// A short, stable code for monitoring and API error bodies.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::VisionUnavailable => "vision_unavailable",
            Self::NotAHybridModel { .. } => "NOT_A_HYBRID_MODEL",
            Self::TooManyImages { .. } => "too_many_images",
            Self::Image { source, .. } => source.code(),
            Self::FetchOverloaded { .. } => IMAGE_FETCH_OVERLOADED_CODE,
            Self::Splice(e) => e.code(),
            Self::Encode { .. } => "image_encode_failed",
            Self::ProjectorMismatch { .. } => "projector_mismatch",
        }
    }

    /// Whether the request itself is at fault (a `400`), rather than the
    /// server (a tower failure on an accepted image, or a server started with
    /// a projector that does not fit its model — a `500` — or a remote-image
    /// fetcher at capacity, [`Self::is_overload`]).
    #[must_use]
    pub fn is_client_error(&self) -> bool {
        !matches!(
            self,
            Self::Encode { .. } | Self::ProjectorMismatch { .. } | Self::FetchOverloaded { .. }
        )
    }

    /// Whether the server shed this request for load
    /// ([`Self::FetchOverloaded`]): retryable, answered as an overload (a
    /// `503` with `Retry-After`), never as a fault in the request.
    #[must_use]
    pub fn is_overload(&self) -> bool {
        matches!(self, Self::FetchOverloaded { .. })
    }
}

/// The `error.code` of a request whose remote image the fetcher refused
/// because it is at capacity ([`MultimodalError::FetchOverloaded`]): a `503`
/// with `Retry-After` and `type: overloaded_error` on a server.
pub const IMAGE_FETCH_OVERLOADED_CODE: &str = "image_fetch_overloaded";

/// How an executor is named in a refusal: the hybrid runner on the Metal
/// GPU, or the CPU model.
#[must_use]
pub fn executor_label(backend: HybridBackend) -> &'static str {
    match backend {
        HybridBackend::Cpu => "CPU hybrid model",
        HybridBackend::Metal => "Metal hybrid runner",
    }
}

/// Whether the hybrid executor `executor` runs the rows prefill itself —
/// both do: the CPU model (`HybridModel::forward_prefill_pieces`) and the
/// Metal runner (`HybridMetalRunner::forward_prefill_rows`), each keeping
/// its own M-RoPE offsets. The match is exhaustive on purpose: an executor
/// added later must decide here, at compile time, whether it serves image
/// rows, and one that does not is refused by
/// [`InferenceEngine::multimodal_executor`] with
/// [`multimodal_backend_refusal`]. (A dense engine has no hybrid executor
/// and is refused for its model kind instead,
/// [`EngineError::NotAHybridModel`].)
const fn executor_runs_rows(executor: HybridBackend) -> bool {
    match executor {
        HybridBackend::Cpu | HybridBackend::Metal => true,
    }
}

/// The typed refusal of a multimodal request by an engine whose executor
/// does not run the rows prefill — `BACKEND_UNAVAILABLE` through
/// [`crate::engine_seam::engine_error_code`], naming `backend` (pass the
/// executor the engine decodes on, not the backend it was asked for) and
/// the constraint.
///
/// No engine raises it today: every hybrid executor runs the rows prefill
/// ([`InferenceEngine::prefills_images`] is `true` for each), and the
/// complete pre-encode check is [`InferenceEngine::multimodal_executor`],
/// whose only refusal is then a dense engine's `NOT_A_HYBRID_MODEL`.
#[must_use]
pub fn multimodal_backend_refusal(backend: Backend, architecture: &str) -> RuntimeError {
    EngineError::BackendUnavailable {
        requested: backend,
        reason: format!(
            "vision (multimodal) prefill for the hybrid `{architecture}` model does not run on \
             the {backend} executor this engine decodes on, whose recurrent state and KV cache \
             would never see the image rows; the request is refused rather than split across \
             two executors (serve vision from a `--backend cpu` engine)"
        ),
    }
    .into()
}

impl From<MultimodalError> for RuntimeError {
    /// [`MultimodalError::NotAHybridModel`] becomes the engine's typed
    /// [`EngineError::NotAHybridModel`] (same `NOT_A_HYBRID_MODEL` code);
    /// every other kind is a request/configuration error carrying its
    /// `[code]`.
    fn from(error: MultimodalError) -> Self {
        match error {
            MultimodalError::NotAHybridModel { architecture } => EngineError::NotAHybridModel {
                operation: MULTIMODAL_PREFILL_OPERATION,
                architecture,
            }
            .into(),
            other => RuntimeError::Config(format!("[{}] {other}", other.code())),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The vision service: references -> encoded images
// ─────────────────────────────────────────────────────────────────────────────

/// What a vision projector keeps resident on each executor, read from its
/// header without building a tower — what `oxibonsai info` reports for a
/// projector file ([`VisionService::footprints`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionFootprints {
    /// ViT blocks.
    pub blocks: usize,
    /// The CPU tower: every weight as `f32`.
    pub cpu_bytes: u64,
    /// The Metal tower: its weights as the file stores them, its scratch
    /// for the budget's images and the host position grid; `None` on a
    /// build without the Metal backend.
    pub metal_bytes: Option<u64>,
    /// The per-image token budget the Metal figure is sized for.
    pub image_max_tokens: usize,
}

/// A loaded vision projector plus how images reach it: the preprocessing
/// (token budget) and which references are resolved. Built once at startup
/// (`--mmproj`) for the executor the engine decodes on (see the module docs,
/// "Backends") and shared read-only by every request. The tower is held
/// behind an `Arc`, so a request's own view of the service
/// ([`VisionService::with_fetch_observer`]) shares it instead of copying it.
#[derive(Debug)]
pub struct VisionService {
    tower: Arc<VisionEncoder>,
    preprocess: PreprocessConfig,
    policy: ImageSourcePolicy,
    ids: VisionTokenIds,
    /// A request's view only: set when its fetcher refused one of its
    /// fetches for capacity, so that image is reported as
    /// [`MultimodalError::FetchOverloaded`]. `None` on the shared service.
    fetch_overloaded: Option<Arc<AtomicBool>>,
}

/// Map and parse a projector GGUF and check it is one (`clip`), handing the
/// parsed file to `build` while the mapping lives.
fn with_projector<T>(
    path: &std::path::Path,
    build: impl FnOnce(&GgufFile<'_>, &str) -> RuntimeResult<T>,
) -> RuntimeResult<T> {
    let shown = path.display().to_string();
    let mmap = mmap_gguf_file(path)
        .map_err(|e| RuntimeError::Config(format!("--mmproj {shown}: cannot open: {e}")))?;
    let gguf = GgufFile::parse(&mmap)
        .map_err(|e| RuntimeError::Config(format!("--mmproj {shown}: not a valid GGUF: {e}")))?;
    let arch = LoadedModel::architecture_of(&gguf);
    if arch != "clip" {
        return Err(RuntimeError::Config(format!(
            "--mmproj {shown}: general.architecture is '{arch}', expected 'clip' (a vision \
             projector such as Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf)"
        )));
    }
    build(&gguf, &shown)
}

impl VisionService {
    /// Wrap a loaded CPU tower.
    ///
    /// # Errors
    ///
    /// [`RuntimeError::Config`] for an invalid budget
    /// (`--image-max-tokens` outside `1..=16384`).
    pub fn new(
        tower: VisionTower,
        image_max_tokens: usize,
        policy: ImageSourcePolicy,
    ) -> RuntimeResult<Self> {
        Self::with_encoder(VisionEncoder::Cpu(tower), image_max_tokens, policy)
    }

    /// Wrap a loaded tower on either executor.
    ///
    /// The policy's decode budget follows the token budget: a source image
    /// with more pixels than the grid it is resized to can use (with the
    /// documented headroom, [`PreprocessConfig::max_source_pixels`]) is
    /// refused as `image_too_large` from its header, before it is inflated —
    /// a small file can declare a huge image. A policy that already carries a
    /// tighter budget keeps it. Every constructor goes through here, so a
    /// library caller gets the same bound the CLI and the server apply.
    ///
    /// # Errors
    ///
    /// [`RuntimeError::Config`] for an invalid budget
    /// (`--image-max-tokens` outside `1..=16384`).
    pub fn with_encoder(
        tower: VisionEncoder,
        image_max_tokens: usize,
        policy: ImageSourcePolicy,
    ) -> RuntimeResult<Self> {
        let config = tower.config();
        let preprocess =
            PreprocessConfig::qwen_vl(config.patch_size, config.spatial_merge, image_max_tokens)
                .map_err(|e| RuntimeError::Config(e.to_string()))?;
        let mut policy = policy;
        let derived = preprocess.max_source_pixels();
        policy.max_source_pixels = Some(
            policy
                .max_source_pixels
                .map_or(derived, |own| own.min(derived)),
        );
        Ok(Self {
            tower: Arc::new(tower),
            preprocess,
            policy,
            ids: VisionTokenIds::BONSAI2,
            fetch_overloaded: None,
        })
    }

    /// Load the projector GGUF at `path` (`general.architecture = "clip"`)
    /// as a CPU tower and wrap it — [`VisionService::load_for_backend`]
    /// for an engine decoding on the CPU. The file is mapped only while its
    /// weights are dequantised; the tower owns everything afterwards.
    ///
    /// # Errors
    ///
    /// [`RuntimeError::Config`] naming the file for an unreadable / invalid
    /// GGUF, a non-`clip` architecture, or a projector the tower refuses
    /// (its error verbatim).
    pub fn load(
        path: &std::path::Path,
        image_max_tokens: usize,
        policy: ImageSourcePolicy,
    ) -> RuntimeResult<Self> {
        Self::load_for_backend(path, image_max_tokens, policy, None)
    }

    /// Load the projector GGUF at `path` for an engine whose hybrid decode
    /// runs on `backend` ([`InferenceEngine::hybrid_backend`]): the Metal
    /// tower (sized for `image_max_tokens`-token images) for
    /// [`HybridBackend::Metal`], the CPU tower otherwise — never both.
    ///
    /// # Errors
    ///
    /// As [`VisionService::load`], plus a Metal tower the device refuses (or
    /// a build without the Metal backend asked for one).
    pub fn load_for_backend(
        path: &std::path::Path,
        image_max_tokens: usize,
        policy: ImageSourcePolicy,
        backend: Option<HybridBackend>,
    ) -> RuntimeResult<Self> {
        let tower = with_projector(path, |gguf, shown| {
            let refused = |e: oxibonsai_model::error::ModelError| {
                RuntimeError::Config(format!("--mmproj {shown}: {e}"))
            };
            match backend {
                Some(HybridBackend::Metal) => metal_encoder(gguf, image_max_tokens, shown),
                Some(HybridBackend::Cpu) | None => VisionTower::from_mmproj(gguf)
                    .map(VisionEncoder::Cpu)
                    .map_err(refused),
            }
        })?;
        Self::with_encoder(tower, image_max_tokens, policy)
    }

    /// Load the projector GGUF at `path` for `engine`: refuse an engine that
    /// cannot serve image turns before the projector is even read
    /// ([`InferenceEngine::multimodal_executor`]), build the tower for the
    /// executor the engine decodes on ([`VisionService::load_for_backend`]),
    /// and check that its rows fit the model ([`VisionService::check_engine`])
    /// — so every refusal an engine and a projector can earn together is
    /// raised here, at load, and none after an image is encoded.
    ///
    /// # Errors
    ///
    /// [`EngineError::NotAHybridModel`] for a dense engine, everything
    /// [`VisionService::load_for_backend`] refuses, and
    /// [`MultimodalError::ProjectorMismatch`] for a projector of another
    /// width.
    pub fn load_for_engine(
        path: &std::path::Path,
        image_max_tokens: usize,
        policy: ImageSourcePolicy,
        engine: &InferenceEngine<'_>,
    ) -> RuntimeResult<Self> {
        let executor = engine.multimodal_executor()?;
        let service = Self::load_for_backend(path, image_max_tokens, policy, Some(executor))?;
        service.check_engine(engine)?;
        Ok(service)
    }

    /// Whether this projector's images can enter `engine`, checked without
    /// encoding anything: the engine prefills image rows
    /// ([`InferenceEngine::multimodal_executor`]) and the tower's rows are as
    /// wide as its model's residual stream. Returns the executor the
    /// multimodal prefill will run on — what a caller runs before it spends
    /// a vision-tower encode on a request.
    ///
    /// # Errors
    ///
    /// [`EngineError::NotAHybridModel`] for a dense engine;
    /// [`MultimodalError::ProjectorMismatch`] (as a [`RuntimeError`], code
    /// `projector_mismatch`) naming both widths and the executor.
    pub fn check_engine(&self, engine: &InferenceEngine<'_>) -> RuntimeResult<HybridBackend> {
        let executor = engine.multimodal_executor()?;
        let projector_width = self.tower.config().projection_dim;
        let model_width = engine.hidden_size();
        if projector_width != model_width {
            return Err(MultimodalError::ProjectorMismatch {
                projector_width,
                model_width,
                executor: executor_label(executor),
            }
            .into());
        }
        Ok(executor)
    }

    /// Bytes the Metal tower of the projector at `path` keeps resident when
    /// sized for `image_max_tokens`-token images (its weights as the file
    /// stores them, its scratch and the host position grid), read from the
    /// file's header and tensor types without building one — what a
    /// Metal-backed engine's KV window leaves room for
    /// ([`HybridLoadOptions::vision_resident_bytes`](crate::engine_hybrid_gpu::HybridLoadOptions)).
    /// `0` on a build without the Metal backend, where no Metal tower
    /// exists.
    ///
    /// # Errors
    ///
    /// As [`VisionService::load`] for the file, and an invalid budget.
    pub fn metal_footprint(path: &std::path::Path, image_max_tokens: usize) -> RuntimeResult<u64> {
        with_projector(path, |gguf, shown| {
            // The configuration first, so a projector the towers refuse is
            // named the same way on every build.
            VisionConfig::from_gguf(gguf)
                .map_err(|e| RuntimeError::Config(format!("--mmproj {shown}: {e}")))?;
            metal_footprint_of(gguf, image_max_tokens, shown)
        })
    }

    /// What the projector `gguf` keeps resident on each executor (see
    /// [`VisionFootprints`]), for `image_max_tokens`-token images.
    ///
    /// # Errors
    ///
    /// [`RuntimeError::Config`] for a projector the towers refuse (its
    /// configuration, or blocks of mixed storage) or an invalid budget.
    pub fn footprints(
        gguf: &GgufFile<'_>,
        image_max_tokens: usize,
    ) -> RuntimeResult<VisionFootprints> {
        let config =
            VisionConfig::from_gguf(gguf).map_err(|e| RuntimeError::Config(e.to_string()))?;
        let metal_bytes = if cfg!(all(feature = "metal", target_os = "macos")) {
            Some(metal_footprint_of(gguf, image_max_tokens, "projector")?)
        } else {
            None
        };
        Ok(VisionFootprints {
            blocks: config.blocks,
            cpu_bytes: VisionTower::footprint(&config),
            metal_bytes,
            image_max_tokens,
        })
    }

    /// Use `ids` as the vision marker ids (a vocabulary other than Bonsai
    /// 2's; the default is [`VisionTokenIds::BONSAI2`]).
    #[must_use]
    pub fn with_token_ids(mut self, ids: VisionTokenIds) -> Self {
        self.ids = ids;
        self
    }

    /// `service` as one request sees it: the same tower, preprocessing and
    /// marker ids, with a remote-image fetcher that reports every fetch it
    /// makes to `observer` (and starts none once the observer says the
    /// request was abandoned; see [`ImageFetchObserver`]). While the wrapped
    /// fetcher runs, the request is the calling thread's [`CurrentFetch`], so
    /// a fetcher that watches it drops a fetch in flight once the request is
    /// abandoned, and an image the fetcher refused because it is at capacity
    /// is reported as [`MultimodalError::FetchOverloaded`]. A server hands
    /// each request its own view so that the request's deadline can name the
    /// stage a fetch is in and a request whose client went away fetches
    /// nothing more. A policy without a fetcher installed is returned as it
    /// is (shared, not copied): there is nothing to observe.
    #[must_use]
    pub fn with_fetch_observer(
        service: &Arc<Self>,
        observer: Arc<dyn ImageFetchObserver>,
    ) -> Arc<Self> {
        match service.policy.remote.fetcher() {
            Some(fetcher) => {
                let overloaded = Arc::new(AtomicBool::new(false));
                Arc::new(Self {
                    tower: Arc::clone(&service.tower),
                    preprocess: service.preprocess,
                    policy: service.policy.clone().with_remote_fetcher(
                        SharedRemoteImageFetcher::new(Arc::new(ObservedFetcher::new(
                            fetcher.clone(),
                            observer,
                            Arc::clone(&overloaded),
                        ))),
                    ),
                    ids: service.ids,
                    fetch_overloaded: Some(overloaded),
                })
            }
            None => Arc::clone(service),
        }
    }

    /// The loaded tower, on the executor it was built for.
    #[must_use]
    pub fn tower(&self) -> &VisionEncoder {
        &self.tower
    }

    /// The preprocessing (token budget) every image goes through.
    #[must_use]
    pub fn preprocess(&self) -> &PreprocessConfig {
        &self.preprocess
    }

    /// Which image references are resolved.
    #[must_use]
    pub fn policy(&self) -> &ImageSourcePolicy {
        &self.policy
    }

    /// The vision marker ids the splice keys on.
    #[must_use]
    pub fn token_ids(&self) -> VisionTokenIds {
        self.ids
    }

    /// Resolve, decode and preprocess one image reference — everything but
    /// the (expensive) tower encode, so a request's geometry (and its
    /// context budget) is known before any image is encoded.
    ///
    /// # Errors
    ///
    /// [`MultimodalError::Image`] for a reference, decode or budget problem.
    pub fn prepare_reference(
        &self,
        index: usize,
        reference: &str,
    ) -> Result<PreparedImage, MultimodalError> {
        let image = load_image_source(reference, &self.policy)
            .map_err(|source| self.reference_error(index, source))?;
        prepare_image(&image, &self.preprocess)
            .map_err(|source| MultimodalError::Image { index, source })
    }

    /// The error of image `index`, whose reference did not resolve with
    /// `source`: [`MultimodalError::FetchOverloaded`] when this request's
    /// fetcher refused the fetch because it is at capacity (the images of a
    /// request are fetched one after another and the first failure ends the
    /// request, so the refusal is this image's), else
    /// [`MultimodalError::Image`].
    fn reference_error(&self, index: usize, source: ImageInputError) -> MultimodalError {
        let overloaded = self
            .fetch_overloaded
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire));
        if overloaded && matches!(source, ImageInputError::RemoteFetchFailed { .. }) {
            MultimodalError::FetchOverloaded { index, source }
        } else {
            MultimodalError::Image { index, source }
        }
    }

    /// [`VisionService::prepare_reference`] for every reference of one
    /// request, in order.
    ///
    /// # Errors
    ///
    /// [`MultimodalError::TooManyImages`] above
    /// [`MAX_IMAGES_PER_REQUEST`], else the first image's error.
    pub fn prepare_all(
        &self,
        references: &[String],
    ) -> Result<Vec<PreparedImage>, MultimodalError> {
        if references.len() > MAX_IMAGES_PER_REQUEST {
            return Err(MultimodalError::TooManyImages {
                images: references.len(),
                limit: MAX_IMAGES_PER_REQUEST,
            });
        }
        references
            .iter()
            .enumerate()
            .map(|(i, r)| self.prepare_reference(i, r))
            .collect()
    }

    /// Resolve, decode, preprocess and encode one image reference.
    ///
    /// # Errors
    ///
    /// [`MultimodalError::Image`] for a reference, decode or budget problem
    /// (a `400`), [`MultimodalError::Encode`] when the tower fails.
    pub fn encode_reference(
        &self,
        index: usize,
        reference: &str,
    ) -> Result<EncodedImage, MultimodalError> {
        let prepared = self.prepare_reference(index, reference)?;
        self.encode_prepared(index, &prepared)
    }

    /// Preprocess and encode already-decoded pixels.
    ///
    /// # Errors
    ///
    /// As [`VisionService::encode_reference`].
    pub fn encode_image(
        &self,
        index: usize,
        image: &oxibonsai_model::vision::ImageRgb8,
    ) -> Result<EncodedImage, MultimodalError> {
        let prepared = prepare_image(image, &self.preprocess)
            .map_err(|source| MultimodalError::Image { index, source })?;
        self.encode_prepared(index, &prepared)
    }

    /// Encode an image [`VisionService::prepare_reference`] (or
    /// [`prepare_image`]) already brought to the tower's geometry.
    ///
    /// # Errors
    ///
    /// [`MultimodalError::Encode`] when the tower fails.
    pub fn encode_prepared(
        &self,
        index: usize,
        prepared: &PreparedImage,
    ) -> Result<EncodedImage, MultimodalError> {
        let started = std::time::Instant::now();
        let (rows, grid) = self
            .tower
            .encode(&prepared.image, self.preprocess.max_tokens)
            .map_err(|e| MultimodalError::Encode {
                index,
                reason: e.to_string(),
            })?;
        tracing::info!(
            image = index,
            backend = self.tower.backend_name(),
            source_width = prepared.source.0,
            source_height = prepared.source.1,
            width = prepared.image.width,
            height = prepared.image.height,
            grid_h = grid.h,
            grid_w = grid.w,
            seconds = started.elapsed().as_secs_f64(),
            "image encoded by the vision tower"
        );
        Ok(EncodedImage {
            rows,
            grid,
            source: prepared.source,
        })
    }

    /// Encode every reference of one request, in order.
    ///
    /// # Errors
    ///
    /// [`MultimodalError::TooManyImages`] above
    /// [`MAX_IMAGES_PER_REQUEST`], else the first image's error.
    pub fn encode_all(&self, references: &[String]) -> Result<Vec<EncodedImage>, MultimodalError> {
        let prepared = self.prepare_all(references)?;
        self.encode_all_prepared(&prepared)
    }

    /// Encode every prepared image of one request, in order.
    ///
    /// # Errors
    ///
    /// The first image's [`MultimodalError::Encode`].
    pub fn encode_all_prepared(
        &self,
        prepared: &[PreparedImage],
    ) -> Result<Vec<EncodedImage>, MultimodalError> {
        prepared
            .iter()
            .enumerate()
            .map(|(i, p)| self.encode_prepared(i, p))
            .collect()
    }
}

/// The Metal tower of a projector, sized for `image_max_tokens`-token
/// images.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_encoder(
    gguf: &GgufFile<'_>,
    image_max_tokens: usize,
    shown: &str,
) -> RuntimeResult<VisionEncoder> {
    oxibonsai_model::vision::metal::VisionTowerMetal::from_mmproj(gguf, image_max_tokens)
        .map(VisionEncoder::Metal)
        .map_err(|e| RuntimeError::Config(format!("--mmproj {shown}: {e}")))
}

/// Without the Metal backend no Metal tower exists (and no engine decodes
/// on Metal to ask for one).
#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_encoder(
    _gguf: &GgufFile<'_>,
    _image_max_tokens: usize,
    shown: &str,
) -> RuntimeResult<VisionEncoder> {
    Err(RuntimeError::Config(format!(
        "--mmproj {shown}: a Metal vision tower needs a build with the Metal backend (macOS with \
         the `metal` feature)"
    )))
}

/// [`VisionService::metal_footprint`] of a parsed projector.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_footprint_of(
    gguf: &GgufFile<'_>,
    image_max_tokens: usize,
    shown: &str,
) -> RuntimeResult<u64> {
    oxibonsai_model::vision::metal::VisionTowerMetal::footprint(gguf, image_max_tokens)
        .map_err(|e| RuntimeError::Config(format!("--mmproj {shown}: {e}")))
}

/// No Metal tower exists without the Metal backend.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_footprint_of(
    _gguf: &GgufFile<'_>,
    _image_max_tokens: usize,
    _shown: &str,
) -> RuntimeResult<u64> {
    Ok(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Prompts
// ─────────────────────────────────────────────────────────────────────────────

/// A rendered, tokenized chat prompt with its encoded images: one
/// `<|image_pad|>` per image, checked at construction.
#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalPrompt {
    tokens: Vec<u32>,
    images: Vec<EncodedImage>,
    segments: Vec<SpliceSegment>,
    total_rows: usize,
}

impl MultimodalPrompt {
    /// Check that `tokens` holds exactly one bracketed placeholder per image
    /// (in order) and build the prompt.
    ///
    /// # Errors
    ///
    /// [`MultimodalError::Splice`] naming the mismatch.
    pub fn new(
        tokens: Vec<u32>,
        images: Vec<EncodedImage>,
        ids: VisionTokenIds,
    ) -> Result<Self, MultimodalError> {
        let grids: Vec<GridSize> = images.iter().map(|i| i.grid).collect();
        let plan = plan_splice(&tokens, &grids, ids)?;
        Ok(Self {
            total_rows: plan.total_rows(),
            segments: plan.segments().to_vec(),
            tokens,
            images,
        })
    }

    /// The rendered prompt's token ids (one placeholder per image).
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The encoded images, in placeholder order.
    #[must_use]
    pub fn images(&self) -> &[EncodedImage] {
        &self.images
    }

    /// Sequence positions the prompt occupies once every placeholder is
    /// replaced by its image's rows — what `usage.prompt_tokens` reports
    /// and the context budget counts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.total_rows
    }

    /// Whether the prompt is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_rows == 0
    }

    /// The prompt as [`PromptSegment`]s, text and images in order.
    #[must_use]
    pub fn segments(&self) -> Vec<PromptSegment> {
        self.segments
            .iter()
            .filter_map(|segment| match segment {
                SpliceSegment::Text(range) => self
                    .tokens
                    .get(range.clone())
                    .map(|t| PromptSegment::Text(t.to_vec())),
                SpliceSegment::Image(i) => self.images.get(*i).map(|img| PromptSegment::Image {
                    rows: img.rows.clone(),
                    grid: img.grid,
                }),
            })
            .collect()
    }
}

/// A chat prompt ready to generate from: text-only token ids, or a
/// multimodal prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatPrompt {
    /// Token ids (every text-only request — byte-identical to before).
    Text(Vec<u32>),
    /// Text with images spliced in.
    Multimodal(MultimodalPrompt),
}

impl ChatPrompt {
    /// Sequence positions the prompt occupies (images expanded).
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Text(tokens) => tokens.len(),
            Self::Multimodal(prompt) => prompt.len(),
        }
    }

    /// Whether the prompt is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The rendered prompt's token ids (placeholders unexpanded).
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        match self {
            Self::Text(tokens) => tokens,
            Self::Multimodal(prompt) => prompt.tokens(),
        }
    }

    /// Images in the prompt.
    #[must_use]
    pub fn image_count(&self) -> usize {
        match self {
            Self::Text(_) => 0,
            Self::Multimodal(prompt) => prompt.images().len(),
        }
    }

    /// Prefill the whole prompt as a new sequence (position 0) and return
    /// the last row's logits: `prefill_from_pos(tokens, 0)` for text,
    /// [`InferenceEngine::prefill_multimodal`] for a multimodal prompt. The
    /// next token then sits at sequence position [`ChatPrompt::len`] — for
    /// a caller that drives its own decode loop (a grammar-constrained or
    /// stop-sequence one).
    ///
    /// # Errors
    ///
    /// Whatever the selected prefill returns.
    pub fn prefill(&self, engine: &mut InferenceEngine<'_>) -> RuntimeResult<Vec<f32>> {
        match self {
            Self::Text(tokens) => engine.prefill_from_pos(tokens, 0),
            Self::Multimodal(prompt) => engine.prefill_multimodal(&prompt.segments(), 0),
        }
    }

    /// `engine.generate` for text, [`InferenceEngine::generate_multimodal`]
    /// for a multimodal prompt.
    ///
    /// # Errors
    ///
    /// Whatever the selected generation returns.
    pub fn generate(
        &self,
        engine: &mut InferenceEngine<'_>,
        max_tokens: usize,
    ) -> RuntimeResult<Vec<u32>> {
        match self {
            Self::Text(tokens) => engine.generate(tokens, max_tokens),
            Self::Multimodal(prompt) => engine.generate_multimodal(prompt, max_tokens),
        }
    }

    /// `engine.generate_with_logprobs` for text,
    /// [`InferenceEngine::generate_multimodal_with_logprobs`] for a
    /// multimodal prompt.
    ///
    /// # Errors
    ///
    /// Whatever the selected generation returns.
    #[cfg(feature = "server")]
    pub fn generate_with_logprobs(
        &self,
        engine: &mut InferenceEngine<'_>,
        max_tokens: usize,
        top_k: usize,
        id_to_token: &dyn Fn(u32) -> String,
    ) -> RuntimeResult<(Vec<u32>, Vec<crate::api_types::LogprobsContent>)> {
        match self {
            Self::Text(tokens) => {
                engine.generate_with_logprobs(tokens, max_tokens, top_k, id_to_token)
            }
            Self::Multimodal(prompt) => {
                engine.generate_multimodal_with_logprobs(prompt, max_tokens, top_k, id_to_token)
            }
        }
    }

    /// `engine.generate_streaming` for text,
    /// [`InferenceEngine::generate_multimodal_streaming`] for a multimodal
    /// prompt.
    ///
    /// # Errors
    ///
    /// Whatever the selected generation returns.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn generate_streaming(
        &self,
        engine: &mut InferenceEngine<'_>,
        max_tokens: usize,
        tx: &tokio::sync::mpsc::UnboundedSender<u32>,
    ) -> RuntimeResult<usize> {
        match self {
            Self::Text(tokens) => engine.generate_streaming(tokens, max_tokens, tx),
            Self::Multimodal(prompt) => {
                engine.generate_multimodal_streaming(prompt, max_tokens, tx)
            }
        }
    }

    /// `engine.generate_streaming_sync` for text,
    /// [`InferenceEngine::generate_multimodal_streaming_sync`] for a
    /// multimodal prompt.
    ///
    /// # Errors
    ///
    /// Whatever the selected generation returns.
    pub fn generate_streaming_sync(
        &self,
        engine: &mut InferenceEngine<'_>,
        max_tokens: usize,
        tx: &std::sync::mpsc::Sender<u32>,
    ) -> RuntimeResult<usize> {
        match self {
            Self::Text(tokens) => engine.generate_streaming_sync(tokens, max_tokens, tx),
            Self::Multimodal(prompt) => {
                engine.generate_multimodal_streaming_sync(prompt, max_tokens, tx)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The engine side
// ─────────────────────────────────────────────────────────────────────────────

/// What one decode step hands its observer: the token and the raw logits it
/// was sampled from.
type StepObserver<'o> = dyn FnMut(u32, &[f32]) -> bool + 'o;

impl<'a> InferenceEngine<'a> {
    /// The executor a multimodal prefill on this engine runs on — the CPU
    /// hybrid model or the Metal hybrid runner, whichever the engine decodes
    /// on ([`InferenceEngine::hybrid_backend`]): one sequence never spans
    /// two executors. Reads no state and changes none, so a caller runs it
    /// before it spends a vision-tower encode on a request (see the module
    /// docs, "Refusals come before the encode").
    ///
    /// # Errors
    ///
    /// [`EngineError::NotAHybridModel`] for a dense engine — the only
    /// engine that cannot serve an image turn, since both hybrid executors
    /// run the rows prefill.
    pub fn multimodal_executor(&self) -> RuntimeResult<HybridBackend> {
        match self.hybrid_backend() {
            Some(executor) if executor_runs_rows(executor) => Ok(executor),
            Some(executor) => Err(multimodal_backend_refusal(
                match executor {
                    HybridBackend::Cpu => Backend::Cpu,
                    HybridBackend::Metal => Backend::Metal,
                },
                self.architecture(),
            )),
            None => Err(EngineError::NotAHybridModel {
                operation: MULTIMODAL_PREFILL_OPERATION,
                architecture: self.architecture().to_string(),
            }
            .into()),
        }
    }

    /// Whether this engine serves image turns:
    /// [`InferenceEngine::multimodal_executor`] names an executor rather
    /// than refusing. `true` for a hybrid engine on the CPU model and for
    /// one on the Metal runner, `false` for a dense engine. The one question
    /// a caller asks to decide whether an image turn can run here — never
    /// which backend the engine was built with.
    #[must_use]
    pub fn prefills_images(&self) -> bool {
        self.multimodal_executor().is_ok()
    }

    /// Prefill a multimodal prompt from sequence position `start_pos` and
    /// return the last row's `[vocab]` logits.
    ///
    /// Text segments are embedded like any prompt (lookup + inverse
    /// Hadamard transform); image rows enter block 0 verbatim; every row
    /// rotates at its 3-axis M-RoPE position (design §6.2). The rows run on
    /// the executor the engine decodes on (see the module docs, "Backends"),
    /// and the engine's sequence bookkeeping is the text prefill's
    /// (`prepare_hybrid_position`): `start_pos` must be the sequence's next
    /// position, or `0` — which starts a new sequence (the hybrid model, the
    /// Metal runner, any attached recurrent state and the sequence id
    /// reset, exactly as a text prefill at 0 does). Every row counts into
    /// [`InferenceEngine::prefill_token_count`].
    ///
    /// # Errors
    ///
    /// [`InferenceEngine::multimodal_executor`]'s refusal first
    /// ([`EngineError::NotAHybridModel`] for a dense engine);
    /// `RecurrentRollbackUnsupported` / `NonContiguousPosition` for a
    /// `start_pos` the recurrence cannot continue from; an empty prompt; and
    /// anything the rows prefill returns.
    pub fn prefill_multimodal(
        &mut self,
        segments: &[PromptSegment],
        start_pos: usize,
    ) -> RuntimeResult<Vec<f32>> {
        let started = std::time::Instant::now();
        let backend = self.multimodal_executor()?;
        if segments.iter().all(PromptSegment::is_empty) {
            return Err(RuntimeError::Model(
                oxibonsai_model::error::ModelError::MissingTensor {
                    name: "prefill_multimodal: empty prompt".into(),
                },
            ));
        }
        self.prepare_hybrid_position(start_pos)?;
        let pieces: Vec<PromptPiece<'_>> = segments
            .iter()
            .filter(|s| !s.is_empty())
            .map(|segment| match segment {
                PromptSegment::Text(tokens) => PromptPiece::Tokens(tokens),
                PromptSegment::Image { rows, grid } => PromptPiece::Image { rows, grid: *grid },
            })
            .collect();
        let (logits, rows) = match (&mut self.model, self.hybrid_gpu.as_mut()) {
            (LoadedModel::Hybrid(model), Some(gpu)) => {
                let assembled =
                    model.assemble_prompt_at(&pieces, gpu.rope_start_for(start_pos)?)?;
                let logits = gpu.prefill_rows(
                    &assembled.rows,
                    &assembled.positions,
                    start_pos,
                    model.prefill_chunk(),
                )?;
                (logits, assembled.len())
            }
            (LoadedModel::Hybrid(model), None) => {
                let mut logits = vec![0.0f32; model.config().base.vocab_size];
                let rows = model.forward_prefill_pieces(&pieces, start_pos, Some(&mut logits))?;
                (logits, rows)
            }
            (LoadedModel::Dense(_), _) => {
                return Err(EngineError::NotAHybridModel {
                    operation: MULTIMODAL_PREFILL_OPERATION,
                    architecture: self.architecture().to_string(),
                }
                .into())
            }
        };
        self.record_prefill_tokens(rows);
        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(started.elapsed().as_secs_f64());
        }
        tracing::debug!(
            rows,
            start_pos,
            backend = backend.as_str(),
            rope_delta = self.rope_delta(),
            "multimodal prefill complete"
        );
        Ok(logits)
    }

    /// Prefill `prompt` for a generation (cancellation observed first, like
    /// the text prefill); `None` when cancelled before the prompt was
    /// ingested.
    fn prefill_multimodal_for_generate(
        &mut self,
        prompt: &MultimodalPrompt,
    ) -> RuntimeResult<Option<Vec<f32>>> {
        if self.is_cancelled() {
            tracing::debug!("cancelled before multimodal prefill");
            return Ok(None);
        }
        let logits = self.prefill_multimodal(&prompt.segments(), 0)?;
        #[cfg(test)]
        let logits = self.scripted_row(logits, true)?;
        Ok(Some(logits))
    }

    /// The decode loop every multimodal generation shares — the text
    /// paths' loop step for step: cancellation check, sample with the
    /// generated-token history (penalties), stop on EOS, hand the token and
    /// its logits to `observe` (which may stop the loop), forward the token
    /// at the next sequence position. Returns the generated ids.
    fn decode_multimodal(
        &mut self,
        mut logits: Vec<f32>,
        start: usize,
        max_tokens: usize,
        observe: &mut StepObserver<'_>,
    ) -> RuntimeResult<Vec<u32>> {
        let decode_start = std::time::Instant::now();
        let mut output: Vec<u32> = Vec::with_capacity(max_tokens.min(4096));
        for (pos, _) in (start..).zip(0..max_tokens) {
            let step_start = std::time::Instant::now();
            if self.is_cancelled() {
                tracing::debug!(pos, "multimodal generation cancelled");
                break;
            }
            let next = self.sampler.sample_with_history(&logits, &output)?;
            if self.is_eos(next) {
                tracing::debug!(pos, "EOS token generated (multimodal)");
                break;
            }
            if !observe(next, &logits) {
                tracing::debug!(pos, "receiver dropped, stopping multimodal generation");
                break;
            }
            output.push(next);
            logits = self.forward_logits(next, pos)?;
            if let Some(m) = &self.metrics {
                m.decode_token_duration_seconds
                    .observe(step_start.elapsed().as_secs_f64());
            }
        }
        if let Some(m) = &self.metrics {
            let elapsed = decode_start.elapsed().as_secs_f64();
            if elapsed > 0.0 && !output.is_empty() {
                m.tokens_per_second.observe(output.len() as f64 / elapsed);
            }
            m.tokens_generated_total.inc_by(output.len() as u64);
            m.update_memory_from_rss();
        }
        self.stats.record_request(output.len());
        tracing::info!(
            prompt_len = start,
            generated = output.len(),
            "multimodal generation complete"
        );
        Ok(output)
    }

    /// Generate from a multimodal prompt: [`InferenceEngine::prefill_multimodal`]
    /// from position 0, then the text path's decode loop (see the module
    /// docs). Returns the generated ids (not including the prompt).
    ///
    /// # Errors
    ///
    /// As [`InferenceEngine::prefill_multimodal`], plus sampling and decode
    /// errors.
    pub fn generate_multimodal(
        &mut self,
        prompt: &MultimodalPrompt,
        max_tokens: usize,
    ) -> RuntimeResult<Vec<u32>> {
        let Some(logits) = self.prefill_multimodal_for_generate(prompt)? else {
            return Ok(Vec::new());
        };
        self.decode_multimodal(logits, prompt.len(), max_tokens, &mut |_, _| true)
    }

    /// [`InferenceEngine::generate_multimodal`] that also records, per
    /// generated token, its log-probability and the `top_k` (at most 20)
    /// alternatives from the model's raw logits — `generate_with_logprobs`'
    /// contract.
    ///
    /// # Errors
    ///
    /// As [`InferenceEngine::generate_multimodal`].
    #[cfg(feature = "server")]
    pub fn generate_multimodal_with_logprobs(
        &mut self,
        prompt: &MultimodalPrompt,
        max_tokens: usize,
        top_k: usize,
        id_to_token: &dyn Fn(u32) -> String,
    ) -> RuntimeResult<(Vec<u32>, Vec<crate::api_types::LogprobsContent>)> {
        let top_k = top_k.min(20);
        let Some(logits) = self.prefill_multimodal_for_generate(prompt)? else {
            return Ok((Vec::new(), Vec::new()));
        };
        let mut logprobs = Vec::new();
        let tokens =
            self.decode_multimodal(logits, prompt.len(), max_tokens, &mut |token, row| {
                logprobs.push(crate::api_types::compute_logprobs(
                    row,
                    token,
                    top_k,
                    id_to_token,
                ));
                true
            })?;
        Ok((tokens, logprobs))
    }

    /// [`InferenceEngine::generate_multimodal`] sending each token through
    /// `tx` as it is generated (stops when the receiver is dropped);
    /// returns the number of tokens generated — `generate_streaming`'s
    /// contract.
    ///
    /// # Errors
    ///
    /// As [`InferenceEngine::generate_multimodal`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn generate_multimodal_streaming(
        &mut self,
        prompt: &MultimodalPrompt,
        max_tokens: usize,
        tx: &tokio::sync::mpsc::UnboundedSender<u32>,
    ) -> RuntimeResult<usize> {
        let Some(logits) = self.prefill_multimodal_for_generate(prompt)? else {
            return Ok(0);
        };
        let tokens =
            self.decode_multimodal(logits, prompt.len(), max_tokens, &mut |token, _| {
                tx.send(token).is_ok()
            })?;
        Ok(tokens.len())
    }

    /// [`InferenceEngine::generate_multimodal_streaming`] over a
    /// synchronous channel (the CLI's streaming printer) —
    /// `generate_streaming_sync`'s contract.
    ///
    /// # Errors
    ///
    /// As [`InferenceEngine::generate_multimodal`].
    pub fn generate_multimodal_streaming_sync(
        &mut self,
        prompt: &MultimodalPrompt,
        max_tokens: usize,
        tx: &std::sync::mpsc::Sender<u32>,
    ) -> RuntimeResult<usize> {
        let Some(logits) = self.prefill_multimodal_for_generate(prompt)? else {
            return Ok(0);
        };
        let tokens =
            self.decode_multimodal(logits, prompt.len(), max_tokens, &mut |token, _| {
                tx.send(token).is_ok()
            })?;
        Ok(tokens.len())
    }
}

/// [`VisionService::prepare_all`] on the blocking pool (decoding and
/// resampling are CPU work, and a remote image's fetch waits on its
/// fetcher; neither may run on an async worker thread). The references are
/// prepared one after another, so a request's remote images are fetched
/// sequentially.
///
/// # Errors
///
/// The first image's [`MultimodalError`], or
/// [`MultimodalError::Encode`] if the blocking task itself failed.
#[cfg(not(target_arch = "wasm32"))]
pub async fn prepare_references_blocking(
    service: Arc<VisionService>,
    references: Vec<String>,
) -> Result<Vec<PreparedImage>, MultimodalError> {
    tokio::task::spawn_blocking(move || service.prepare_all(&references))
        .await
        .unwrap_or_else(|e| {
            Err(MultimodalError::Encode {
                index: 0,
                reason: format!("the image preparation task failed: {e}"),
            })
        })
}

/// [`VisionService::encode_all_prepared`] on the blocking pool — each image
/// is a full vision-tower encode.
///
/// # Errors
///
/// The first image's [`MultimodalError::Encode`], including a failure of
/// the blocking task itself.
#[cfg(not(target_arch = "wasm32"))]
pub async fn encode_prepared_blocking(
    service: Arc<VisionService>,
    prepared: Vec<PreparedImage>,
) -> Result<Vec<EncodedImage>, MultimodalError> {
    tokio::task::spawn_blocking(move || service.encode_all_prepared(&prepared))
        .await
        .unwrap_or_else(|e| {
            Err(MultimodalError::Encode {
                index: 0,
                reason: format!("the encoding task failed: {e}"),
            })
        })
}

#[cfg(test)]
#[path = "vision_prefill_tests.rs"]
mod tests;
