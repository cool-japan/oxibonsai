//! Multimodal (image + text) prefill and generation for a hybrid Bonsai 2
//! engine (bonsai2-design.md §6.2; SV-11).
//!
//! # The pipeline
//!
//! 1. The chat template renders each image part as
//!    `<|vision_start|><|image_pad|><|vision_end|>`; the prompt is tokenized
//!    as usual, so it holds one `<|image_pad|>` per image.
//! 2. [`VisionService`] turns every image reference into merged embedding
//!    rows: resolve (`data:` URI / local file, under an
//!    [`ImageSourcePolicy`]), decode, preprocess exactly like the reference
//!    (smart resize to the 32-pixel grid within the `--image-max-tokens`
//!    budget), and encode with the Qwen3-VL tower.
//! 3. [`MultimodalPrompt`] checks the splice (one bracketed placeholder per
//!    image, in order) and cuts the prompt into [`PromptSegment`]s.
//! 4. [`InferenceEngine::prefill_multimodal`] runs the segments through the
//!    hybrid model's rows prefill (`HybridModel::forward_prefill_pieces`):
//!    text rows are looked up and inverse-Hadamard-transformed, image rows
//!    enter block 0 verbatim, every row carries its 3-axis M-RoPE position.
//! 5. Decoding continues exactly like a text prompt's
//!    ([`InferenceEngine::generate_multimodal`] and its logprobs /
//!    streaming forms): same sampler, penalties, stop-on-EOS, cancellation
//!    and metrics as `generate` / `generate_with_logprobs` /
//!    `generate_streaming`, from the sequence position after the last
//!    prompt row (the model rotates later tokens at its M-RoPE offset).
//!
//! # Backend
//!
//! Only the CPU `HybridModel` implements the rows prefill. The single
//! predicate [`InferenceEngine::multimodal_backend_is_cpu`] says whether
//! this engine's hybrid decode runs on that model; an engine that decodes on
//! the Metal hybrid runner must refuse multimodal requests
//! ([`multimodal_backend_refusal`]) instead of running the image turn on
//! the CPU model per request, because the runner keeps its own recurrent
//! state and KV cache and they would never see the image rows.

use std::sync::Arc;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_model::hybrid::LoadedModel;
use oxibonsai_model::hybrid::PromptPiece;
use oxibonsai_model::vision::{
    load_image_source, plan_splice, prepare_image, GridSize, ImageInputError, ImageSourcePolicy,
    PreparedImage, PreprocessConfig, SpliceError, SpliceSegment, VisionTokenIds, VisionTower,
};

use crate::engine::InferenceEngine;
use crate::engine_seam::{Backend, EngineError};
use crate::error::{RuntimeError, RuntimeResult};

/// The chat-template content types a multimodal user turn is built from
/// (an image part renders as the template's placeholder).
pub use oxibonsai_tokenizer::chat_templates::{RenderContent, RenderContentPart};

/// Images one request may carry: each is a full vision-tower encode (up to
/// tens of seconds on the CPU for a large image), so the count is bounded.
pub const MAX_IMAGES_PER_REQUEST: usize = 16;

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
            Self::Splice(e) => e.code(),
            Self::Encode { .. } => "image_encode_failed",
        }
    }

    /// Whether the request itself is at fault (a `400`), rather than the
    /// server (a `500` — only a tower failure on an accepted image).
    #[must_use]
    pub fn is_client_error(&self) -> bool {
        !matches!(self, Self::Encode { .. })
    }
}

impl From<MultimodalError> for RuntimeError {
    fn from(error: MultimodalError) -> Self {
        RuntimeError::Config(format!("[{}] {error}", error.code()))
    }
}

/// The typed refusal of a multimodal request by an engine whose hybrid
/// decode does not run on the CPU model (see the module docs, "Backend").
///
/// `BACKEND_UNAVAILABLE` through [`crate::engine_seam::engine_error_code`];
/// the message names the constraint.
#[must_use]
pub fn multimodal_backend_refusal(backend: Backend, architecture: &str) -> RuntimeError {
    EngineError::BackendUnavailable {
        requested: backend,
        reason: format!(
            "vision (multimodal) prefill for the hybrid `{architecture}` model runs on the CPU \
             model only, and this engine decodes on the {backend} hybrid runner, whose recurrent \
             state and KV cache would never see the image rows; the request is refused rather \
             than split across two backends (serve vision from a `--backend cpu` engine)"
        ),
    }
    .into()
}

// ─────────────────────────────────────────────────────────────────────────────
// The vision service: references -> encoded images
// ─────────────────────────────────────────────────────────────────────────────

/// A loaded vision projector plus how images reach it: the preprocessing
/// (token budget) and which references are resolved. Built once at startup
/// (`--mmproj`) and shared read-only by every request.
#[derive(Debug)]
pub struct VisionService {
    tower: VisionTower,
    preprocess: PreprocessConfig,
    policy: ImageSourcePolicy,
    ids: VisionTokenIds,
}

impl VisionService {
    /// Wrap a loaded tower.
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
        let preprocess = PreprocessConfig::for_tower(&tower, image_max_tokens)
            .map_err(|e| RuntimeError::Config(e.to_string()))?;
        Ok(Self {
            tower,
            preprocess,
            policy,
            ids: VisionTokenIds::BONSAI2,
        })
    }

    /// Load the projector GGUF at `path` (`general.architecture = "clip"`)
    /// and wrap it. The file is mapped only while its weights are
    /// dequantised; the tower owns everything afterwards.
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
        let shown = path.display();
        let mmap = mmap_gguf_file(path)
            .map_err(|e| RuntimeError::Config(format!("--mmproj {shown}: cannot open: {e}")))?;
        let gguf = GgufFile::parse(&mmap).map_err(|e| {
            RuntimeError::Config(format!("--mmproj {shown}: not a valid GGUF: {e}"))
        })?;
        let arch = LoadedModel::architecture_of(&gguf);
        if arch != "clip" {
            return Err(RuntimeError::Config(format!(
                "--mmproj {shown}: general.architecture is '{arch}', expected 'clip' (a vision \
                 projector such as Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf)"
            )));
        }
        let tower = VisionTower::from_mmproj(&gguf)
            .map_err(|e| RuntimeError::Config(format!("--mmproj {shown}: {e}")))?;
        Self::new(tower, image_max_tokens, policy)
    }

    /// Use `ids` as the vision marker ids (a vocabulary other than Bonsai
    /// 2's; the default is [`VisionTokenIds::BONSAI2`]).
    #[must_use]
    pub fn with_token_ids(mut self, ids: VisionTokenIds) -> Self {
        self.ids = ids;
        self
    }

    /// The loaded tower.
    #[must_use]
    pub fn tower(&self) -> &VisionTower {
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
            .map_err(|source| MultimodalError::Image { index, source })?;
        prepare_image(&image, &self.preprocess)
            .map_err(|source| MultimodalError::Image { index, source })
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
    /// Whether multimodal prefill runs on this engine's CPU hybrid model —
    /// the single seam deciding it (see the module docs, "Backend").
    ///
    /// `true` for every engine whose hybrid decode is the CPU `HybridModel`
    /// (`--backend cpu`, or `auto` on a host without a usable Metal runner).
    /// An engine whose hybrid decode runs on the Metal runner answers
    /// `false`, and [`InferenceEngine::prefill_multimodal`] then refuses with
    /// [`multimodal_backend_refusal`] rather than splitting one sequence
    /// across two backends (the runner's recurrent state and KV cache would
    /// never see the image rows); the predicate is the one line to change
    /// when the runner gains its own rows prefill.
    #[must_use]
    pub fn multimodal_backend_is_cpu(&self) -> bool {
        !matches!(
            self.hybrid_backend(),
            Some(crate::engine_hybrid_gpu::HybridBackend::Metal)
        )
    }

    /// [`InferenceEngine::multimodal_backend_is_cpu`] as a typed check.
    fn require_multimodal_backend(&self) -> RuntimeResult<()> {
        if self.multimodal_backend_is_cpu() {
            Ok(())
        } else {
            Err(multimodal_backend_refusal(
                self.backend(),
                self.architecture(),
            ))
        }
    }

    /// Prefill a multimodal prompt from sequence position `start_pos` and
    /// return the last row's `[vocab]` logits.
    ///
    /// Text segments are embedded like any prompt (lookup + inverse
    /// Hadamard transform); image rows enter block 0 verbatim; every row
    /// rotates at its 3-axis M-RoPE position (design §6.2). The engine's
    /// sequence bookkeeping is the text prefill's: `start_pos` must be the
    /// recurrent state's next position, or `0` — which starts a new
    /// sequence (the hybrid model, any attached recurrent state and the
    /// sequence id reset, exactly as a text prefill at 0 does).
    ///
    /// # Errors
    ///
    /// [`MultimodalError::NotAHybridModel`] (as a [`RuntimeError`]) for a
    /// dense engine; [`multimodal_backend_refusal`] when
    /// [`InferenceEngine::multimodal_backend_is_cpu`] is `false`;
    /// `RecurrentRollbackUnsupported` / `NonContiguousPosition` for a
    /// `start_pos` the recurrence cannot continue from; an empty prompt; and
    /// anything the model's prefill returns.
    pub fn prefill_multimodal(
        &mut self,
        segments: &[PromptSegment],
        start_pos: usize,
    ) -> RuntimeResult<Vec<f32>> {
        let started = std::time::Instant::now();
        let LoadedModel::Hybrid(_) = &self.model else {
            return Err(MultimodalError::NotAHybridModel {
                architecture: self.architecture().to_string(),
            }
            .into());
        };
        self.require_multimodal_backend()?;
        if segments.iter().all(PromptSegment::is_empty) {
            return Err(RuntimeError::Model(
                oxibonsai_model::error::ModelError::MissingTensor {
                    name: "prefill_multimodal: empty prompt".into(),
                },
            ));
        }
        self.continue_hybrid_sequence_at(start_pos)?;
        let LoadedModel::Hybrid(model) = &mut self.model else {
            return Err(MultimodalError::NotAHybridModel {
                architecture: String::new(),
            }
            .into());
        };
        let pieces: Vec<PromptPiece<'_>> = segments
            .iter()
            .filter(|s| !s.is_empty())
            .map(|segment| match segment {
                PromptSegment::Text(tokens) => PromptPiece::Tokens(tokens),
                PromptSegment::Image { rows, grid } => PromptPiece::Image { rows, grid: *grid },
            })
            .collect();
        let mut logits = vec![0.0f32; model.config().base.vocab_size];
        let rows = model.forward_prefill_pieces(&pieces, start_pos, Some(&mut logits))?;
        if let Some(m) = &self.metrics {
            m.prefill_duration_seconds
                .observe(started.elapsed().as_secs_f64());
        }
        tracing::debug!(rows, start_pos, "multimodal prefill complete");
        Ok(logits)
    }

    /// The hybrid position contract a text prefill applies
    /// (`engine_seam`'s `prepare_hybrid_position`), for the multimodal one:
    /// continue at the recurrent state's next position, or start a new
    /// sequence at `0` (the model, any attached recurrent state and the
    /// sequence id reset).
    fn continue_hybrid_sequence_at(&mut self, pos: usize) -> RuntimeResult<()> {
        let LoadedModel::Hybrid(model) = &mut self.model else {
            return Ok(());
        };
        let expected = model.recurrent().token_count();
        if pos == expected {
            return Ok(());
        }
        if pos == 0 {
            model.reset();
            if let Some(state) = self.recurrent.as_deref_mut() {
                state.reset_recurrent();
            }
            self.sequence_id = self.sequence_id.wrapping_add(1);
            return Ok(());
        }
        if pos < expected {
            return Err(RuntimeError::Model(
                oxibonsai_model::error::ModelError::RecurrentRollbackUnsupported {
                    pos,
                    tokens: expected,
                },
            ));
        }
        Err(EngineError::NonContiguousPosition { expected, got: pos }.into())
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
/// resampling are CPU work that must not run on an async worker thread).
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
mod tests {
    use super::*;
    use crate::engine_seam::engine_error_code;
    use crate::sampling::SamplingParams;
    use oxibonsai_testkit::qwen35_fixture::{synthetic_qwen35_gguf, HIDDEN, VOCAB};

    const MAX_SEQ: usize = 64;
    const IDS: VisionTokenIds = VisionTokenIds {
        vision_start: 500,
        vision_end: 501,
        image_pad: 502,
    };

    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        }
    }

    /// A hybrid engine decoding on the CPU model — the backend multimodal
    /// prefill runs on (see the module docs, "Backend"), chosen explicitly
    /// rather than left to `Backend::Auto`.
    fn cpu_engine<'a>(gguf: &'a GgufFile<'a>) -> InferenceEngine<'a> {
        InferenceEngine::from_gguf_with_backend(gguf, greedy(), 7, MAX_SEQ, Backend::Cpu)
            .expect("a CPU hybrid engine")
    }

    fn image(grid: GridSize, seed: u32) -> EncodedImage {
        let rows = (0..grid.n_tokens() * HIDDEN)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x >> 8) as f32 / 16_777_216.0) - 0.5
            })
            .collect();
        EncodedImage {
            rows,
            grid,
            source: (grid.w * 32, grid.h * 32),
        }
    }

    fn prompt(grid: GridSize) -> MultimodalPrompt {
        let tokens = vec![
            10,
            11,
            IDS.vision_start,
            IDS.image_pad,
            IDS.vision_end,
            12,
            13,
        ];
        MultimodalPrompt::new(tokens, vec![image(grid, 9)], IDS).expect("splices")
    }

    /// The seam's refusal shape: an engine error with a stable code whose
    /// message names the constraint.
    #[test]
    fn the_metal_backend_refusal_is_a_typed_engine_error_naming_the_constraint() {
        let err = multimodal_backend_refusal(Backend::Metal, "qwen35");
        assert_eq!(engine_error_code(&err), Some("BACKEND_UNAVAILABLE"));
        let text = err.to_string();
        assert!(text.contains("vision (multimodal) prefill"), "{text}");
        assert!(text.contains("CPU model only"), "{text}");
        assert!(text.contains("never see the image rows"), "{text}");
        assert!(text.contains("metal"), "{text}");
    }

    #[test]
    fn a_cpu_hybrid_engine_prefills_images_on_the_cpu_model() {
        let bytes = synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let engine = cpu_engine(&gguf);
        assert!(engine.multimodal_backend_is_cpu());
        assert!(engine.require_multimodal_backend().is_ok());
    }

    #[test]
    fn multimodal_prompts_expand_each_placeholder_to_its_grid() {
        let grid = GridSize { h: 2, w: 3 };
        let p = prompt(grid);
        assert_eq!(p.len(), 7 - 1 + 6);
        assert_eq!(p.images().len(), 1);
        let segments = p.segments();
        assert_eq!(segments.len(), 3);
        assert_eq!(
            segments[0],
            PromptSegment::Text(vec![10, 11, IDS.vision_start])
        );
        assert!(matches!(&segments[1], PromptSegment::Image { grid: g, .. } if *g == grid));
        assert_eq!(
            segments[2],
            PromptSegment::Text(vec![IDS.vision_end, 12, 13])
        );
        let chat = ChatPrompt::Multimodal(p);
        assert_eq!(chat.len(), 12);
        assert_eq!(chat.image_count(), 1);
        assert_eq!(chat.tokens().len(), 7);
        assert_eq!(ChatPrompt::Text(vec![1, 2]).len(), 2);

        let err = MultimodalPrompt::new(vec![1, IDS.image_pad], vec![image(grid, 1)], IDS)
            .expect_err("unbracketed");
        assert_eq!(err.code(), "image_placeholder_misplaced");
        let err = MultimodalPrompt::new(vec![1, 2], vec![image(grid, 1)], IDS).expect_err("none");
        assert_eq!(err.code(), "image_placeholder_count_mismatch");
        assert!(err.is_client_error());
    }

    #[test]
    fn a_dense_engine_refuses_multimodal_prefill_by_name() {
        let mut engine = InferenceEngine::new(
            oxibonsai_core::config::Qwen3Config::tiny_test(),
            greedy(),
            7,
        );
        let err = engine
            .prefill_multimodal(&prompt(GridSize { h: 1, w: 1 }).segments(), 0)
            .expect_err("dense");
        assert!(err.to_string().contains("[NOT_A_HYBRID_MODEL]"), "{err}");
    }

    /// The engine's multimodal prefill is the model's rows prefill (same
    /// logits, bit for bit), advances the sequence by the expanded length,
    /// and every generation form agrees with the plain one.
    #[test]
    fn hybrid_engine_prefills_and_generates_from_a_multimodal_prompt() {
        let bytes = synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let grid = GridSize { h: 2, w: 2 };
        let p = prompt(grid);

        let mut engine = cpu_engine(&gguf);
        let logits = engine
            .prefill_multimodal(&p.segments(), 0)
            .expect("prefill");
        assert_eq!(logits.len(), VOCAB);
        assert_eq!(engine.sequence_position(), p.len());
        let delta = engine.hybrid_model().map(|m| m.rope_delta());
        assert_eq!(delta, Some(grid.n_tokens() - 2));

        // The model-level rows prefill of the same pieces.
        let mut direct =
            oxibonsai_model::hybrid::HybridModel::from_gguf(&gguf, MAX_SEQ).expect("model");
        let segments = p.segments();
        let pieces: Vec<PromptPiece<'_>> = segments
            .iter()
            .map(|s| match s {
                PromptSegment::Text(t) => PromptPiece::Tokens(t),
                PromptSegment::Image { rows, grid } => PromptPiece::Image { rows, grid: *grid },
            })
            .collect();
        let mut want = vec![0.0f32; VOCAB];
        direct
            .forward_prefill_pieces(&pieces, 0, Some(&mut want))
            .expect("direct prefill");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&logits), bits(&want));

        // A continuation at a non-contiguous position is refused.
        let err = engine
            .prefill_multimodal(&p.segments(), p.len() + 3)
            .expect_err("gap");
        assert_eq!(engine_error_code(&err), Some("NON_CONTIGUOUS_POSITION"));

        // Generation: plain, logprobs and both streaming forms agree.
        let plain = engine.generate_multimodal(&p, 6).expect("generate");
        let (with_lp, logprobs) = engine
            .generate_multimodal_with_logprobs(&p, 6, 3, &|id| format!("<{id}>"))
            .expect("logprobs");
        assert_eq!(with_lp, plain);
        assert_eq!(logprobs.len(), plain.len());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let n = engine
            .generate_multimodal_streaming(&p, 6, &tx)
            .expect("stream");
        drop(tx);
        let mut streamed = Vec::new();
        while let Ok(t) = rx.try_recv() {
            streamed.push(t);
        }
        assert_eq!(n, plain.len());
        assert_eq!(streamed, plain);
        let (stx, srx) = std::sync::mpsc::channel();
        let n = engine
            .generate_multimodal_streaming_sync(&p, 6, &stx)
            .expect("stream sync");
        drop(stx);
        assert_eq!(srx.iter().collect::<Vec<_>>(), plain);
        assert_eq!(n, plain.len());
        // Through the prompt enum.
        let chat = ChatPrompt::Multimodal(p.clone());
        assert_eq!(chat.generate(&mut engine, 6).expect("enum"), plain);

        // The first generated token is the argmax of the prefill's logits.
        if let Some(&first) = plain.first() {
            let best = logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                    if v > b.1 {
                        (i, v)
                    } else {
                        b
                    }
                });
            assert_eq!(first as usize, best.0);
        }
    }

    /// A sequence snapshotted at `k`, continued by
    /// `prefill_multimodal(segments, k)` and then restored continues exactly
    /// like an engine that never saw the image: the model's M-RoPE offset
    /// follows the restored position (the tail and the decode step after it
    /// run past where the abandoned continuation ended).
    #[test]
    fn a_restored_sequence_continues_at_its_own_rope_offset() {
        let bytes = synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut engine = cpu_engine(&gguf);
        let mut reference = cpu_engine(&gguf);
        let lead = [10u32, 11, 12, 13];
        let tail = [14u32, 15, 16, 17, 18, 19, 20, 21, 22];
        let k = lead.len();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

        engine.prefill_from_pos(&lead, 0).expect("prefill");
        let snapshot = engine.snapshot_sequence().expect("snapshot");
        let grid = GridSize { h: 2, w: 3 };
        let continuation = [
            PromptSegment::Image {
                rows: image(grid, 21).rows,
                grid,
            },
            PromptSegment::Text(vec![30]),
        ];
        engine
            .prefill_multimodal(&continuation, k)
            .expect("image continuation");
        let delta = |e: &InferenceEngine<'_>| e.hybrid_model().map(|m| m.rope_delta());
        assert_eq!(delta(&engine), Some(grid.n_tokens() - 3));
        engine.restore_sequence(&snapshot).expect("restore");
        assert_eq!(engine.sequence_position(), k);
        assert_eq!(delta(&engine), Some(0), "the offset in force at k");
        let got = engine.prefill_from_pos(&tail, k).expect("continue");
        let got_next = engine.decode_step(23, k + tail.len()).expect("decode");

        reference.prefill_from_pos(&lead, 0).expect("prefill");
        let want = reference.prefill_from_pos(&tail, k).expect("continue");
        let want_next = reference.decode_step(23, k + tail.len()).expect("decode");
        assert_eq!(bits(&got), bits(&want));
        assert_eq!(bits(&got_next), bits(&want_next));
    }

    /// A text-only prompt through the enum is exactly `generate`.
    #[test]
    fn text_prompts_are_unchanged_through_the_prompt_enum() {
        let bytes = synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut engine = InferenceEngine::from_gguf(&gguf, greedy(), 7, MAX_SEQ).expect("engine");
        let tokens = vec![10u32, 11, 12, 13, 14];
        let direct = engine.generate(&tokens, 5).expect("generate");
        let via_enum = ChatPrompt::Text(tokens)
            .generate(&mut engine, 5)
            .expect("enum");
        assert_eq!(direct, via_enum);
    }

    #[test]
    fn error_codes_and_client_fault_are_stable() {
        let cases: Vec<(MultimodalError, &str, bool)> = vec![
            (
                MultimodalError::VisionUnavailable,
                "vision_unavailable",
                true,
            ),
            (
                MultimodalError::NotAHybridModel {
                    architecture: "qwen3".into(),
                },
                "NOT_A_HYBRID_MODEL",
                true,
            ),
            (
                MultimodalError::TooManyImages {
                    images: 17,
                    limit: MAX_IMAGES_PER_REQUEST,
                },
                "too_many_images",
                true,
            ),
            (
                MultimodalError::Image {
                    index: 0,
                    source: ImageInputError::Empty {
                        width: 0,
                        height: 1,
                    },
                },
                "image_empty",
                true,
            ),
            (
                MultimodalError::Encode {
                    index: 0,
                    reason: "x".into(),
                },
                "image_encode_failed",
                false,
            ),
        ];
        for (err, code, client) in cases {
            assert_eq!(err.code(), code);
            assert_eq!(err.is_client_error(), client, "{code}");
            let runtime: RuntimeError = err.into();
            assert!(
                runtime.to_string().contains(&format!("[{code}]")),
                "{runtime}"
            );
        }
    }
}
