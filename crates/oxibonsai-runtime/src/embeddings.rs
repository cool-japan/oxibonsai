//! OpenAI v1 embeddings endpoint.
//!
//! Implements `POST /v1/embeddings` — an OpenAI-compatible embedding API that
//! converts text (or token ID arrays) into dense float vectors.
//!
//! # Backends and determinism (`RT-08` / `SV-02` / `sec-19`)
//!
//! The [`EmbedderRegistry`] manages three backends, tried in priority order:
//!
//! 1. **A model-backed [`Embedder`]** — the production default (`RT-08` /
//!    `SV-02`). [`crate::embed_engine::ModelEmbedder`]
//!    tokenises the input, runs it through a loaded model and returns the
//!    mean-pooled, L2-normalised final hidden state
//!    (`BonsaiModel::forward_hidden`, taken *before* the LM head): a genuine
//!    semantic embedding, a pure function of `(model, text)` and of nothing
//!    else. Install one with [`EmbedderRegistry::with_model`] (which also
//!    wires the model-only refinements below) or, for any other
//!    [`Embedder`] implementation, [`EmbedderRegistry::with_model_embedder`]
//!    (which wires none of them — see that method's "Warning" and
//!    [`EmbedderRegistry::with_token_counter`]'s).
//!    * `usage.prompt_tokens` is then the **real** token count from the
//!      model's tokenizer ([`EmbeddingTokenCounter`]) rather than a
//!      whitespace-split word count.
//!    * `"input": [1, 2, 3]` is embedded as those token ids
//!      ([`TokenSequenceEmbedder`]) rather than as the *string* `"1 2 3"`.
//!    * An N-input batch takes the model's engine lock **once**, not N
//!      times ([`Embedder::embed_batch`], which
//!      [`crate::embed_engine::ModelEmbedder`] overrides).
//!    * Each text input is tokenized **once** per request: the ids answer
//!      the `context_length_exceeded` guard and `usage.prompt_tokens`, and
//!      are then embedded as ids ([`EmbeddingTokenCounter::token_ids`],
//!      [`EmbedderRegistry::with_token_aware_model`]) — not tokenized again
//!      to count, to bill and to embed.
//!
//!    When no model backend is installed,
//!    [`EmbedderRegistry::with_require_model_backend`] decides what happens.
//!    With it set — which is what the served router does — [`create_embeddings`]
//!    answers `501 Not Implemented` naming the missing backend, rather than
//!    ever handing back a non-semantic vector that merely *looks* like an
//!    embedding. Left unset (the default of the bare library constructors) the
//!    two lexical fallbacks below still answer, which is exactly what an
//!    embedded caller that has no model asked for.
//! 2. **[`TfIdfEmbedder`]** — a lexical bag-of-words backend. It is
//!    **stateless from the HTTP handler's point of view**: nothing in
//!    [`create_embeddings`] ever calls [`EmbedderRegistry::fit_tfidf`], so
//!    the vocabulary a registry starts with (`None`, i.e. inactive) is the
//!    vocabulary it keeps for the rest of the process — the exact opposite
//!    of the previous behaviour, where two texts submitted a request apart
//!    could land in different, client-request-dependent vector spaces. TF-IDF
//!    is therefore **opt-in**: a caller who wants it fits it once,
//!    administratively (e.g. at construction, with a known corpus), via
//!    [`EmbedderRegistry::fit_tfidf`] — the method itself remains a public,
//!    intentionally mutating operation for exactly that one-time use; it is
//!    simply no longer invoked implicitly from client-supplied request data.
//! 3. **[`IdentityEmbedder`]** — deterministic byte-hash fallback, always
//!    available, and **never the default on a served router** (`RT-08`): it is
//!    an explicitly opt-in test double, reachable only through the bare
//!    [`create_embeddings_router`] constructor, which library embedders and
//!    this crate's own router tests use. It is a pure function of the input
//!    bytes, so embedding the same text twice — in the same request, a
//!    different request, or a different process — always returns a
//!    bit-identical vector.
//!
//! The response's `model` field reports which backend actually answered
//! (`"bonsai-embeddings-model"` / `"-tfidf"` / `"-identity"`, see
//! [`EmbedderRegistry::backend_name`]) rather than echoing the client's
//! (arbitrary, unverified) `model` request field, so a caller can tell a
//! byte hash from a real embedding without reading this module's source.
//!
//! # Metrics (`SV-25`)
//!
//! Both branches of [`create_embeddings`] — the `501` refusal and the
//! computed response — record onto the **shared** [`InferenceMetrics`] the
//! router was built with ([`EmbeddingAppState::with_metrics`]):
//! `requests_total` on entry, `active_requests` through an RAII guard that
//! also fires when a client disconnects mid-request, `errors_total` on every
//! non-2xx answer, `prompt_tokens_total`, and `request_duration_seconds`.
//! `/v1/embeddings` carries its own [`EmbeddingAppState`] and is `merge`d
//! separately from the main `AppState`, so it escapes any `AppState`-based
//! instrumentation and has to be wired explicitly. A registry built without
//! metrics (the bare library constructors) records nothing — deliberately, so
//! this module never fabricates a second, unmounted registry whose counters
//! nobody scrapes.
//!
//! # Batch cap and per-item length cap (`sec-19`)
//!
//! A single request may embed at most
//! [`EmbedderRegistry::max_batch_size`] inputs (default
//! [`DEFAULT_MAX_EMBEDDING_BATCH_SIZE`], configurable via
//! [`EmbedderRegistry::with_max_batch_size`]); a larger batch is rejected
//! with `400 Bad Request` naming the limit, rather than accepted with no
//! bound on the CPU work a single client-controlled array can drive.
//! Independently, each individual input string is capped at
//! [`EmbedderRegistry::max_input_len`] characters (default
//! [`DEFAULT_MAX_EMBEDDING_INPUT_CHARS`], configurable via
//! [`EmbedderRegistry::with_max_input_len`]) — a single-element batch
//! carrying one multi-megabyte string previously passed the batch-size check
//! completely unbounded; a request with an over-length item is rejected with
//! `400` naming `input`. The embedding work itself (TF-IDF and any
//! model-backed computation, both CPU-bound) runs inside
//! [`tokio::task::spawn_blocking`] rather than directly on the async
//! handler's tokio worker thread, so a large batch cannot stall unrelated
//! requests sharing the same runtime (same defect class as `sec-03`).
//!
//! # Token-length truncation vs. `400 context_length_exceeded`
//!
//! A model backend additionally has its own, smaller per-input *token*
//! ceiling ([`EmbeddingTokenCounter::max_input_tokens`],
//! [`crate::embed_engine::DEFAULT_MAX_EMBEDDING_TOKENS`]) independent of the
//! *character* cap above. The two input shapes are handled differently:
//!
//! * **Text** (`"input": "..."` / `["...", "..."]`) that tokenizes past this
//!   ceiling is refused with `400 context_length_exceeded`, naming the real
//!   token count and the ceiling, rather than silently truncated and billed
//!   in full — the OpenAI contract (`usage.prompt_tokens` used to report
//!   the untruncated count while the model only ever saw the truncated one).
//! * **Raw token ids** (`"input": [1, 2, 3]`) deliberately keep truncating —
//!   a caller supplying ids already knows exactly how many it sent, so
//!   [`crate::embed_engine::ModelEmbedder::embed_tokens`]'s conventional
//!   embedding-endpoint truncation is kept for this path. `usage.prompt_tokens`
//!   is corrected instead to charge exactly what gets embedded
//!   (`min(ids.len(), max_input_tokens())`), not the full supplied length.
//!
//! See [`crate::embed_engine`]'s "Truncation" module-doc section for the
//! rationale in full, and [`EmbeddingTokenCounter::max_input_tokens`]'s docs.
//!
//! # Encoding formats
//!
//! - `"float"` (default, or the field omitted) — embedding returned as a
//!   JSON array of `f32` values.
//! - `"base64"` — embedding encoded as RFC 4648 base64 (standard alphabet,
//!   `=` padding), matching the OpenAI API contract: each `f32` is
//!   serialised as four little-endian bytes and the resulting byte string is
//!   base64-encoded. A real OpenAI-SDK client's `base64.b64decode(...)` call
//!   round-trips this correctly.
//! - Any other value is rejected with `400 Bad Request` naming
//!   `encoding_format` (`SV-02`), rather than silently falling back to
//!   `"float"` for a client that made a typo (e.g. `"binary"`) and would
//!   otherwise silently receive JSON when base64 was intended, or vice versa.
//!
//! # Dimensions and normalisation
//!
//! Every backend returns an L2-normalised (unit-length) vector **on
//! success**; a per-item embedding failure (a fully out-of-vocabulary TF-IDF
//! query under [`TfIdfEmbedder::with_strict_oov`]`(false)`, or a
//! model-backend error) is mapped to an all-zero vector instead of aborting
//! the whole batch — an all-zero vector has norm `0`, not `1`, so it is *not*
//! unit-length despite the general rule. The response's top-level
//! `normalized` field reports whether *every* item in this response actually
//! came back normalised (i.e. none of them is all-zero), so a caller does
//! not have to recompute norms itself to detect the fallback case; the
//! top-level `dimension` field reports the dimensionality every item in
//! `data` actually has, after any truncation below (`RT-08` fix item 1's
//! "document the dimension and the normalisation in the response").
//!
//! Setting `dimensions` in the request truncates each embedding vector to
//! that many leading dimensions and **re-normalises the truncated vector**
//! (Matryoshka truncation, matching the real OpenAI contract — a prior
//! version truncated without renormalising, silently handing back a
//! non-unit vector under `dimensions`). `dimensions: 0` is rejected with
//! `400` (it would truncate every embedding to an empty vector) rather than
//! silently accepted. If `dimensions` meets or exceeds the natural embedding
//! size, the full (already-normalised) vector is returned unmodified.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use oxibonsai_rag::embedding::{l2_normalize, Embedder, IdentityEmbedder, TfIdfEmbedder};

use crate::embed_engine::{EmbeddingTokenCounter, ModelEmbedder, TokenSequenceEmbedder};
use crate::metrics::InferenceMetrics;
use crate::server::ActiveRequestGuard;

/// Lock `mutex`, recovering from lock poisoning instead of panicking.
///
/// `EmbedderRegistry` backs the live, server-reachable `POST /v1/embeddings`
/// route. `std::sync::Mutex` poisons permanently once *any* thread panics
/// while holding it, so an `.expect(...)` here would turn one unrelated
/// panic into a process-lifetime outage for every subsequent embeddings
/// request. The guarded state (an `Option<TfIdfEmbedder>`) has no invariant
/// that a mid-mutation panic could leave unsafe to keep using, so recovering
/// the inner value and logging a warning is preferable to a permanent 500.
fn lock_or_recover<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!(
            "EmbedderRegistry mutex was poisoned by a prior panic; recovering inner state \
             instead of propagating the panic to this request"
        );
        poisoned.into_inner()
    })
}

// ─── Request / Response types ─────────────────────────────────────────────────

/// Input accepted by the embeddings endpoint.
///
/// All four variants are deserialized from untagged JSON, so the format is
/// inferred from the structure of the value supplied in the `"input"` field:
///
/// | JSON value | Variant |
/// |---|---|
/// | `"some text"` | `Single` |
/// | `["text one", "text two"]` | `Batch` |
/// | `[42, 1337]` | `TokenIds` |
/// | `[[42, 1337], [9, 99]]` | `BatchTokenIds` |
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// A single text string.
    Single(String),
    /// A batch of text strings.
    Batch(Vec<String>),
    /// A single token-ID sequence (converted to a space-joined string).
    TokenIds(Vec<u32>),
    /// A batch of token-ID sequences.
    BatchTokenIds(Vec<Vec<u32>>),
}

impl EmbeddingInput {
    /// Convert all inputs to `String` form for embedding.
    ///
    /// Token-ID sequences are rendered as space-separated decimal numbers so
    /// they can be passed through the text-based embedder.
    pub fn as_strings(&self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s.clone()],
            EmbeddingInput::Batch(v) => v.clone(),
            EmbeddingInput::TokenIds(ids) => {
                vec![ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")]
            }
            EmbeddingInput::BatchTokenIds(batch) => batch
                .iter()
                .map(|ids| {
                    ids.iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect(),
        }
    }

    /// The raw token-id batches, when the request supplied ids rather than
    /// text; `None` for the two text variants.
    ///
    /// [`create_embeddings`] routes these straight to a
    /// [`TokenSequenceEmbedder`] when one is installed, instead of going
    /// through [`as_strings`](Self::as_strings)'s decimal rendering — see that
    /// trait's docs for why `"1 2 3"` is not an approximation of `[1, 2, 3]`.
    pub fn as_token_batches(&self) -> Option<Vec<Vec<u32>>> {
        match self {
            EmbeddingInput::Single(_) | EmbeddingInput::Batch(_) => None,
            EmbeddingInput::TokenIds(ids) => Some(vec![ids.clone()]),
            EmbeddingInput::BatchTokenIds(batch) => Some(batch.clone()),
        }
    }

    /// Number of distinct inputs.
    pub fn len(&self) -> usize {
        match self {
            EmbeddingInput::Single(_) => 1,
            EmbeddingInput::Batch(v) => v.len(),
            EmbeddingInput::TokenIds(_) => 1,
            EmbeddingInput::BatchTokenIds(v) => v.len(),
        }
    }

    /// Whether the input contains no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `POST /v1/embeddings` request body.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    /// The model name (accepted but ignored — the registry selects the backend).
    pub model: Option<String>,
    /// The text(s) or token sequence(s) to embed.
    pub input: EmbeddingInput,
    /// Encoding format: `"float"` (default) or `"base64"`.
    pub encoding_format: Option<String>,
    /// If set, truncate each embedding to this many dimensions.
    pub dimensions: Option<usize>,
    /// Opaque caller identifier (not processed).
    pub user: Option<String>,
}

/// The serialised form of a single embedding.
///
/// When the request specifies `encoding_format = "base64"` the `Base64` variant
/// is used; otherwise `Float`.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingData {
    /// Embedding as a JSON array of `f32` values.
    Float(Vec<f32>),
    /// Embedding encoded as RFC 4648 base64 (see [`EmbedderRegistry::encode_base64`]).
    Base64(String),
}

/// A single embedding object in the response.
#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    /// Always `"embedding"`.
    pub object: String,
    /// The dense vector (or its encoded form).
    pub embedding: EmbeddingData,
    /// Zero-based position of this item among all inputs.
    pub index: usize,
}

/// Token usage reported in the embeddings response.
#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    /// Total tokens consumed by the prompt(s).
    pub prompt_tokens: usize,
    /// Same as `prompt_tokens` (there are no completion tokens for embeddings).
    pub total_tokens: usize,
}

/// `POST /v1/embeddings` response body.
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    /// Always `"list"`.
    pub object: String,
    /// One [`EmbeddingObject`] per input.
    pub data: Vec<EmbeddingObject>,
    /// The model / backend used.
    pub model: String,
    /// Token usage statistics.
    pub usage: EmbeddingUsage,
    /// Dimensionality every embedding vector in `data` actually has (after
    /// any `dimensions`-driven truncation) — see the module docs' "Dimensions
    /// and normalisation" section (`RT-08` fix item 1).
    pub dimension: usize,
    /// Whether every embedding vector in `data` is L2-normalised
    /// (unit-length). `false` when at least one item fell back to an
    /// all-zero vector (a per-item embedding failure or a fully
    /// out-of-vocabulary strict-mode TF-IDF query) rather than a genuine
    /// unit vector — see the module docs (`RT-08` fix item 1).
    pub normalized: bool,
}

// ─── EmbedderRegistry ─────────────────────────────────────────────────────────

/// Default ceiling on the number of inputs a single `POST /v1/embeddings`
/// request may embed (`sec-19`). Override with
/// [`EmbedderRegistry::with_max_batch_size`].
pub const DEFAULT_MAX_EMBEDDING_BATCH_SIZE: usize = 2048;

/// Default ceiling on the number of *characters* (not bytes) any single
/// input string in a `POST /v1/embeddings` request may contain (`sec-19`).
/// Override with [`EmbedderRegistry::with_max_input_len`]. Independent of
/// [`DEFAULT_MAX_EMBEDDING_BATCH_SIZE`], which only caps the *number* of
/// inputs: without this cap a single-element batch carrying one
/// multi-megabyte string passed the batch-size check completely unbounded.
pub const DEFAULT_MAX_EMBEDDING_INPUT_CHARS: usize = 8192;

/// Thread-safe registry that holds the active embedding backends.
///
/// Priority order (see the module docs for the full rationale): an
/// explicitly-installed [`EmbedderRegistry::with_model_embedder`] backend
/// always wins; otherwise a fitted TF-IDF vocabulary; otherwise the
/// deterministic `IdentityEmbedder` fallback. On creation neither the model
/// slot nor the TF-IDF slot is populated, so a fresh registry always serves
/// `IdentityEmbedder` until one of them is installed *administratively*
/// (never as a side effect of handling a client request — see
/// [`create_embeddings`]).
pub struct EmbedderRegistry {
    default_dim: usize,
    max_batch_size: usize,
    max_input_len: usize,
    /// When `true`, [`create_embeddings`] refuses to answer with the
    /// stateless TF-IDF/`IdentityEmbedder` fallback at all — see
    /// [`EmbedderRegistry::with_require_model_backend`].
    require_model_backend: bool,
    /// A model-backed embedder, when the caller has wired one in via
    /// [`EmbedderRegistry::with_model_embedder`]. Boxed behind `Arc<dyn
    /// Embedder>` (rather than a concrete type) because this crate has no
    /// dependency on `oxibonsai-model` and no way to name a concrete
    /// hidden-state embedder type; any `Embedder` implementation — including
    /// one built on `BonsaiModel` once it exists — can be installed here.
    model: Option<Arc<dyn Embedder>>,
    /// Exact `usage.prompt_tokens` source, when the installed backend owns a
    /// tokenizer (`SV-02`: the whitespace word count is not a token count).
    token_counter: Option<Arc<dyn EmbeddingTokenCounter>>,
    /// Native `"input": [1, 2, 3]` path, when the installed backend can
    /// consume token ids directly.
    token_embedder: Option<Arc<dyn TokenSequenceEmbedder>>,
    /// `true` only while `model`, `token_counter` and `token_embedder` are
    /// one object installed by [`EmbedderRegistry::with_token_aware_model`]
    /// (or [`EmbedderRegistry::with_model`]): only then are the ids the
    /// counter hands out ([`EmbeddingTokenCounter::token_ids`]) the ids the
    /// model would embed, so a text request may be tokenized once and
    /// embedded from those ids. Every builder that replaces one of the three
    /// independently clears it.
    reuse_text_tokens: bool,
    tfidf: std::sync::Mutex<Option<TfIdfEmbedder>>,
    identity: IdentityEmbedder,
    /// Why no model backend is installed, when the builder knows: named in
    /// the `501` body [`create_embeddings`] answers with — see
    /// [`EmbedderRegistry::with_unavailable_reason`].
    unavailable: Option<EmbedderUnavailable>,
}

/// `error.code` of the `/v1/embeddings` `501` when no more specific reason
/// was recorded ([`EmbedderRegistry::with_unavailable_reason`]).
pub const EMBEDDINGS_UNAVAILABLE_CODE: &str = "embeddings_unavailable";

/// `error.type` of the `/v1/embeddings` `501`.
pub const NOT_IMPLEMENTED_ERROR_TYPE: &str = "not_implemented_error";

/// Why a server has no model-backed embedder — e.g. the typed refusal a
/// model's embedder construction returned — carried into the `501` body of
/// `/v1/embeddings` so a client sees the reason, not only the operator's
/// log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedderUnavailable {
    /// A stable machine-readable code (an engine refusal's own code, such as
    /// `"NOT_A_DENSE_MODEL"`), or `None` for [`EMBEDDINGS_UNAVAILABLE_CODE`].
    pub code: Option<&'static str>,
    /// The human-readable reason, appended to the `501` message.
    pub message: String,
}

impl EmbedderUnavailable {
    /// A reason with an optional stable `code`.
    pub fn new(code: Option<&'static str>, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// The `error.code` this reason answers with.
    pub fn code(&self) -> &'static str {
        self.code.unwrap_or(EMBEDDINGS_UNAVAILABLE_CODE)
    }
}

impl EmbedderRegistry {
    /// Create a new registry.
    ///
    /// `default_dim` controls the dimensionality of the `IdentityEmbedder`
    /// fallback and is also used as `max_features` when fitting TF-IDF. No
    /// model backend is installed and the TF-IDF vocabulary starts empty —
    /// use [`EmbedderRegistry::with_model_embedder`] /
    /// [`EmbedderRegistry::fit_tfidf`] to opt into either.
    pub fn new(default_dim: usize) -> Self {
        let dim = default_dim.max(1);
        // `dim` is guaranteed ≥ 1 by the `.max(1)` clamp above, so
        // `IdentityEmbedder::new` cannot fail here.  We still handle the
        // `Err` branch explicitly to avoid `.expect()` in production code.
        let identity = match IdentityEmbedder::new(dim) {
            Ok(embedder) => embedder,
            Err(_) => unreachable!("dim ≥ 1 was guaranteed by max(1) above"),
        };
        Self {
            default_dim: dim,
            max_batch_size: DEFAULT_MAX_EMBEDDING_BATCH_SIZE,
            max_input_len: DEFAULT_MAX_EMBEDDING_INPUT_CHARS,
            require_model_backend: false,
            model: None,
            token_counter: None,
            token_embedder: None,
            reuse_text_tokens: false,
            tfidf: std::sync::Mutex::new(None),
            identity,
            unavailable: None,
        }
    }

    /// Install a real, model-backed embedder — the production backend
    /// (`RT-08` / `SV-02`).
    ///
    /// The one-call form of [`with_model_embedder`](Self::with_model_embedder)
    /// for [`ModelEmbedder`], which is more capable than a bare [`Embedder`]:
    /// this also installs it as the registry's
    /// [`EmbeddingTokenCounter`] (so `usage.prompt_tokens` is the model's own
    /// token count, not a whitespace word count), as its
    /// [`TokenSequenceEmbedder`] (so a `"input": [1, 2, 3]` request embeds
    /// those ids rather than the string `"1 2 3"`). An N-input text request
    /// takes the engine lock once, not N times, through `ModelEmbedder`'s
    /// [`Embedder::embed_batch`] override, and each text is tokenized once
    /// per request (see [`Self::with_token_aware_model`], which this is).
    /// Wiring all three by hand and forgetting one is the whole reason this
    /// exists.
    #[must_use]
    pub fn with_model(self, embedder: Arc<ModelEmbedder>) -> Self {
        self.with_token_aware_model(embedder)
    }

    /// Install one backend as the model [`Embedder`], the
    /// [`EmbeddingTokenCounter`] **and** the [`TokenSequenceEmbedder`]
    /// (builder) — the generic form of [`Self::with_model`], for any backend
    /// that owns a tokenizer and can embed ids.
    ///
    /// Because all three come from the same object, [`create_embeddings`]
    /// tokenizes each **text** input exactly once per request: the ids
    /// [`EmbeddingTokenCounter::token_ids`] returns answer the
    /// `context_length_exceeded` guard and `usage.prompt_tokens`, and are then
    /// embedded through [`TokenSequenceEmbedder::embed_token_batches`] rather
    /// than handed back to [`Embedder::embed_batch`] to be tokenized again.
    /// A request in which some input gets no ids from `token_ids` is counted
    /// with [`EmbeddingTokenCounter::count_tokens`] and embedded as text
    /// instead, so this is also safe for a backend that never hands ids out.
    ///
    /// # Contract
    ///
    /// Embedding `token_ids(text)` as ids must give the vector `embed(text)`
    /// gives — true of [`ModelEmbedder`] by construction (same tokenizer, same
    /// truncation, same engine).
    #[must_use]
    pub fn with_token_aware_model<E>(mut self, embedder: Arc<E>) -> Self
    where
        E: Embedder + EmbeddingTokenCounter + TokenSequenceEmbedder + 'static,
    {
        self.token_counter = Some(Arc::clone(&embedder) as Arc<dyn EmbeddingTokenCounter>);
        self.token_embedder = Some(Arc::clone(&embedder) as Arc<dyn TokenSequenceEmbedder>);
        self.model = Some(embedder as Arc<dyn Embedder>);
        self.reuse_text_tokens = true;
        self
    }

    /// Install an exact token counter for `usage.prompt_tokens` (builder).
    ///
    /// Independent of the embedding backend: a caller may want honest token
    /// accounting even behind a lexical backend.
    ///
    /// # Warning: pair this with a model backend
    ///
    /// Installing a counter with **no** [`Self::with_model_embedder`] (or
    /// [`Self::with_model`]) leaves [`Self::embedding_dim`] reporting
    /// `default_dim` (or the TF-IDF vocabulary size) while this counter may
    /// describe a *different* backend's tokens entirely — nothing ties the
    /// two together. That mismatch is caught in debug builds wherever a
    /// registry is finalised into an [`EmbeddingAppState`]
    /// ([`EmbeddingAppState::from_registry`]'s `debug_assert!`); prefer
    /// [`Self::with_model`], which wires the model, the counter and the
    /// token-id path from the same backend so they cannot disagree.
    #[must_use]
    pub fn with_token_counter(mut self, counter: Arc<dyn EmbeddingTokenCounter>) -> Self {
        self.token_counter = Some(counter);
        self.reuse_text_tokens = false;
        self
    }

    /// Install a native token-id embedding path (builder).
    ///
    /// # Warning
    ///
    /// See [`Self::with_token_counter`]'s "Warning" section — the same
    /// misuse (installed without a model backend) applies here, for the same
    /// reason: this embedder's own dimension can then disagree with
    /// [`Self::embedding_dim`].
    #[must_use]
    pub fn with_token_embedder(mut self, embedder: Arc<dyn TokenSequenceEmbedder>) -> Self {
        self.token_embedder = Some(embedder);
        self.reuse_text_tokens = false;
        self
    }

    /// Install a model-backed embedder as the highest-priority backend
    /// (builder).
    ///
    /// This is the seam a real neural embedding path (mean-pooled,
    /// L2-normalised hidden states from a loaded `BonsaiModel`) is meant to
    /// be wired through once it exists — see the module docs' "Backends and
    /// determinism" section for exactly what is still missing upstream (a
    /// `BonsaiModel::forward_hidden` seam and an `InferenceEngine` accessor,
    /// neither reachable from this crate today) and
    /// [`create_embeddings_router_with_model`] for the router constructor
    /// that takes one.
    #[must_use]
    pub fn with_model_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.model = Some(embedder);
        self.reuse_text_tokens = false;
        self
    }

    /// Override the maximum batch size (builder). Values are clamped to at
    /// least `1` so a misconfigured `0` cannot make the endpoint refuse every
    /// request, including a single-input one.
    #[must_use]
    pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size.max(1);
        self
    }

    /// The maximum number of inputs a single request may embed (`sec-19`).
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Override the maximum per-item input length, in characters (builder).
    /// Values are clamped to at least `1` for the same reason
    /// [`Self::with_max_batch_size`] clamps to `1`.
    #[must_use]
    pub fn with_max_input_len(mut self, max_input_len: usize) -> Self {
        self.max_input_len = max_input_len.max(1);
        self
    }

    /// The maximum number of characters any single input string may contain
    /// (`sec-19`).
    pub fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    /// Require a model-backed [`Embedder`] to answer every request (builder).
    ///
    /// When `true`, [`create_embeddings`] returns `501 Not Implemented`
    /// instead of ever falling back to TF-IDF or `IdentityEmbedder` when no
    /// [`Self::with_model_embedder`] backend is installed — the sanctioned
    /// interim substitute for a real model-backed default (`RT-08` / `SV-02`
    /// correction (b); see the module docs). Defaults to `false`, preserving
    /// the existing stateless-fallback behaviour for callers that have not
    /// opted in.
    #[must_use]
    pub fn with_require_model_backend(mut self, require: bool) -> Self {
        self.require_model_backend = require;
        self
    }

    /// Whether this registry is configured to require a model-backed
    /// [`Embedder`] (via [`Self::with_require_model_backend`]) and none is
    /// currently installed — the exact condition [`create_embeddings`]
    /// checks to answer `501` instead of computing a stateless-fallback
    /// vector. Kept as a method on the registry (checked inside this file's
    /// handler) rather than left for each call site to remember, so the gate
    /// cannot be silently bypassed by a caller that forgets to check first.
    pub fn model_backend_required_but_missing(&self) -> bool {
        self.require_model_backend && self.model.is_none()
    }

    /// Record why no model backend is installed (builder): the `501`
    /// [`create_embeddings`] answers while
    /// [`Self::model_backend_required_but_missing`] holds then carries
    /// `reason` — `error.code` is [`EmbedderUnavailable::code`] and the
    /// message ends with `": <reason.message>"`. Has no effect on a registry
    /// that serves a backend.
    #[must_use]
    pub fn with_unavailable_reason(mut self, reason: EmbedderUnavailable) -> Self {
        self.unavailable = Some(reason);
        self
    }

    /// The recorded reason no model backend is installed, if any.
    pub fn unavailable_reason(&self) -> Option<&EmbedderUnavailable> {
        self.unavailable.as_ref()
    }

    /// Name of the backend that would answer the *next* [`Self::embed_texts`]
    /// call, for the response's `model` field: `"model"`, `"tfidf"`, or
    /// `"identity"`. Reflects which backend actually produced a response
    /// instead of echoing the client's unverified `model` request field
    /// (`RT-08`).
    pub fn backend_name(&self) -> &'static str {
        if self.model.is_some() {
            return "model";
        }
        let guard = lock_or_recover(&self.tfidf);
        if guard.is_some() {
            "tfidf"
        } else {
            "identity"
        }
    }

    /// Embed a slice of text strings, returning one dense vector per input.
    ///
    /// Uses the model backend when one is installed; otherwise the TF-IDF
    /// backend once it has been fitted; otherwise `IdentityEmbedder`. Texts
    /// that fail to embed are silently replaced with a zero vector of the
    /// appropriate dimension.
    ///
    /// The model backend is reached through [`Embedder::embed_batch`]: the
    /// default is the per-item loop, and a [`ModelEmbedder`] overrides it to
    /// take its engine lock **once** for the whole slice instead of once per
    /// item — so the batched path is the one the HTTP handler's text
    /// requests actually reach.
    pub fn embed_texts(&self, texts: &[String]) -> Vec<Vec<f32>> {
        if let Some(ref model) = self.model {
            let dim = model.embedding_dim();
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            return model
                .embed_batch(&refs)
                .into_iter()
                .map(|r| r.unwrap_or_else(|_| vec![0.0; dim]))
                .collect();
        }
        let guard = lock_or_recover(&self.tfidf);
        if let Some(ref tfidf) = *guard {
            texts
                .iter()
                .map(|t| {
                    tfidf
                        .embed(t)
                        .unwrap_or_else(|_| vec![0.0; tfidf.embedding_dim()])
                })
                .collect()
        } else {
            texts
                .iter()
                .map(|t| {
                    self.identity
                        .embed(t)
                        .unwrap_or_else(|_| vec![0.0; self.default_dim])
                })
                .collect()
        }
    }

    /// Whether this registry can embed client-supplied token ids natively.
    pub fn has_token_embedder(&self) -> bool {
        self.token_embedder.is_some()
    }

    /// Embed token-id batches straight through the installed
    /// [`TokenSequenceEmbedder`], one vector per batch.
    ///
    /// Returns `None` when no such backend is installed, so the caller falls
    /// back to the decimal-string rendering. A per-batch failure degrades to a
    /// zero vector, exactly as [`embed_texts`](Self::embed_texts) does.
    ///
    /// Routes through [`TokenSequenceEmbedder::embed_token_batches`] (a
    /// [`ModelEmbedder`] overrides its default per-item loop to take its
    /// engine lock once for the whole set — for `"input": [[1, 2], [3, 4]]`
    /// requests as for plain text).
    pub fn embed_token_batches(&self, batches: &[Vec<u32>]) -> Option<Vec<Vec<f32>>> {
        let embedder = self.token_embedder.as_ref()?;
        let dim = self.embedding_dim();
        Some(
            embedder
                .embed_token_batches(batches)
                .into_iter()
                .map(|r| r.unwrap_or_else(|_| vec![0.0; dim]))
                .collect(),
        )
    }

    /// Total `usage.prompt_tokens` for `texts`.
    ///
    /// Uses the installed [`EmbeddingTokenCounter`] when there is one — the
    /// model's real tokenizer — and otherwise falls back to the historical
    /// whitespace-split word count, which is an approximation this module has
    /// no way to improve on without a tokenizer. Every input counts as at
    /// least one token either way, so a non-empty request never reports
    /// `prompt_tokens: 0`.
    ///
    /// [`create_embeddings`] does not call this: it tokenizes a request once
    /// and derives both the `context_length_exceeded` guard and this same
    /// total from that one pass.
    pub fn count_prompt_tokens(&self, texts: &[String]) -> usize {
        self.tokenize_texts(texts).prompt_tokens(texts)
    }

    /// One tokenization pass over a request's **text** inputs, whose result
    /// answers the `context_length_exceeded` guard, `usage.prompt_tokens` and
    /// — when the ids come from the embedding backend itself — the embedding
    /// pass, so [`create_embeddings`] tokenizes each text once rather than
    /// once to judge its length, once to bill it and once to embed it.
    fn tokenize_texts(&self, texts: &[String]) -> TextTokens {
        let Some(counter) = self.token_counter.as_ref() else {
            return TextTokens::Uncounted;
        };
        if !self.reuse_text_tokens {
            return TextTokens::Counted(texts.iter().map(|t| counter.count_tokens(t)).collect());
        }
        let ids: Vec<Option<Vec<u32>>> = texts.iter().map(|t| counter.token_ids(t)).collect();
        if ids.iter().all(Option::is_some) {
            return TextTokens::Tokenized(ids.into_iter().flatten().collect());
        }
        // Some input got no ids (the backend does not hand them out, or
        // tokenizing that input failed): count only what could not be
        // tokenized and let the text path embed the whole request, exactly
        // as it did before ids were reused.
        TextTokens::Counted(
            ids.into_iter()
                .zip(texts)
                .map(|(ids, text)| match ids {
                    Some(ids) => Some(ids.len()),
                    None => counter.count_tokens(text),
                })
                .collect(),
        )
    }

    /// The largest number of tokens a single **text** input may tokenize to
    /// before [`create_embeddings`] refuses it with `400
    /// context_length_exceeded`, or `None` when no
    /// installed backend reports a ceiling (see
    /// [`EmbeddingTokenCounter::max_input_tokens`]) — in which case no such
    /// guard applies at all.
    pub fn max_input_tokens(&self) -> Option<usize> {
        self.token_counter
            .as_ref()
            .and_then(|c| c.max_input_tokens())
    }

    /// Whether a token-aware path ([`Self::with_token_counter`] /
    /// [`Self::with_token_embedder`]) was installed without the model-backed
    /// [`Embedder`] that is supposed to own it ([`Self::with_model_embedder`],
    /// or [`Self::with_model`] which wires all of them together) —
    /// the misuse [`EmbeddingAppState::from_registry`]'s `debug_assert!`
    /// catches.
    fn has_orphaned_token_paths(&self) -> bool {
        self.model.is_none() && (self.token_counter.is_some() || self.token_embedder.is_some())
    }

    /// Fit the TF-IDF backend from `corpus`.
    ///
    /// After this call [`embed_texts`](Self::embed_texts) will use TF-IDF for
    /// all subsequent requests (unless a model backend is installed, which
    /// always takes priority). Subsequent calls replace the existing model.
    ///
    /// This is an intentionally mutating, explicit, administrative operation
    /// — call it once, e.g. right after constructing the registry with a
    /// known corpus. [`create_embeddings`] never calls this on a client's
    /// behalf (`RT-08` / `SV-02` / `sec-19`): doing so per-request made the
    /// vector space — and even the output dimension — depend on which other
    /// clients had queried the endpoint before, so two requests embedding the
    /// same text could receive different, incomparable vectors.
    pub fn fit_tfidf(&self, corpus: &[String]) {
        if corpus.is_empty() {
            return;
        }
        let refs: Vec<&str> = corpus.iter().map(String::as_str).collect();
        let fitted = TfIdfEmbedder::fit(&refs, self.default_dim);
        let mut guard = lock_or_recover(&self.tfidf);
        *guard = Some(fitted);
    }

    /// Return the current embedding dimension.
    ///
    /// Returns the model backend's dimension when one is installed,
    /// otherwise the TF-IDF vocabulary size when a fitted model is present,
    /// otherwise the configured `default_dim`.
    pub fn embedding_dim(&self) -> usize {
        if let Some(ref model) = self.model {
            return model.embedding_dim();
        }
        let guard = lock_or_recover(&self.tfidf);
        if let Some(ref tfidf) = *guard {
            tfidf.embedding_dim()
        } else {
            self.default_dim
        }
    }

    /// Encode an embedding vector as RFC 4648 base64 (pure Rust, no external
    /// deps).
    ///
    /// Each `f32` is serialised as four bytes in little-endian order; the
    /// resulting byte string is then base64-encoded with the standard
    /// alphabet and `=` padding, exactly as the OpenAI `encoding_format:
    /// "base64"` contract expects (a real client's `base64.b64decode(...)`
    /// round-trips this back to the original little-endian `f32` bytes).
    ///
    /// This previously emitted lowercase hex (a self-consistent but
    /// non-standard format that silently corrupts data for any real
    /// OpenAI-SDK client) — see finding `serve-api-08`.
    pub fn encode_base64(embedding: &[f32]) -> String {
        let mut bytes = Vec::with_capacity(embedding.len() * 4);
        for value in embedding {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        base64_encode_bytes(&bytes)
    }
}

/// What one pass of the registry's token counter learned about a request's
/// **text** inputs ([`EmbedderRegistry::tokenize_texts`]), reused for the
/// `context_length_exceeded` guard, for
/// `usage.prompt_tokens` and, when it holds ids, for the embedding pass.
#[derive(Debug)]
enum TextTokens {
    /// No token counter is installed: nothing was tokenized, no token-length
    /// guard applies, and usage falls back to the whitespace word count.
    Uncounted,
    /// One token count per input (`None` where the counter could not say),
    /// from a counter whose ids the embedding backend does not reuse.
    Counted(Vec<Option<usize>>),
    /// One id sequence per input, from the backend that also embeds them
    /// ([`EmbedderRegistry::with_token_aware_model`]).
    Tokenized(Vec<Vec<u32>>),
}

impl TextTokens {
    /// Token count of input `index`, when one is known.
    fn count(&self, index: usize) -> Option<usize> {
        match self {
            Self::Uncounted => None,
            Self::Counted(counts) => counts.get(index).copied().flatten(),
            Self::Tokenized(ids) => ids.get(index).map(Vec::len),
        }
    }

    /// Number of inputs this pass covered (`0` when nothing was counted).
    fn len(&self) -> usize {
        match self {
            Self::Uncounted => 0,
            Self::Counted(counts) => counts.len(),
            Self::Tokenized(ids) => ids.len(),
        }
    }

    /// The first input (by index) whose token count exceeds `max_tokens`,
    /// with that count. `None` when nothing does or nothing was counted (an
    /// input that cannot be counted cannot be judged over-length either).
    fn first_exceeding(&self, max_tokens: usize) -> Option<(usize, usize)> {
        (0..self.len()).find_map(|index| {
            let n_tokens = self.count(index)?;
            (n_tokens > max_tokens).then_some((index, n_tokens))
        })
    }

    /// `usage.prompt_tokens` for `texts` (the inputs this pass covered): the
    /// counted tokens where known, the whitespace word count otherwise, and
    /// at least one token per input either way, so a non-empty request never
    /// reports `prompt_tokens: 0`.
    fn prompt_tokens(&self, texts: &[String]) -> usize {
        texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                self.count(index)
                    .unwrap_or_else(|| text.split_whitespace().count())
                    .max(1)
            })
            .sum()
    }
}

/// RFC 4648 standard base64 alphabet.
const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode an arbitrary byte slice as RFC 4648 base64 (standard alphabet,
/// `=` padding). Pure Rust, no external dependencies.
fn base64_encode_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);

        let packed = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);

        out.push(BASE64_ALPHABET[((packed >> 18) & 0x3f) as usize] as char);
        out.push(BASE64_ALPHABET[((packed >> 12) & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 {
            BASE64_ALPHABET[((packed >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            BASE64_ALPHABET[(packed & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

// ─── App state ────────────────────────────────────────────────────────────────

/// Axum application state for the embeddings sub-router.
///
/// Its fields are private: a state is only ever built through
/// [`EmbeddingAppState::from_registry`], whose builder-misuse check a public
/// field would let a caller bypass by swapping
/// the registry afterwards. Read them through [`Self::registry`] and
/// [`Self::metrics`].
pub struct EmbeddingAppState {
    /// The active embedding registry.
    registry: EmbedderRegistry,
    /// The **shared** [`InferenceMetrics`] this route records onto (`SV-25`),
    /// or `None` for a router built without one (which then records nothing —
    /// see the module docs' "Metrics" section for why a private registry is
    /// not an acceptable substitute).
    metrics: Option<Arc<InferenceMetrics>>,
}

impl EmbeddingAppState {
    /// The active embedding registry.
    pub fn registry(&self) -> &EmbedderRegistry {
        &self.registry
    }

    /// The **shared** [`InferenceMetrics`] this route records onto (`SV-25`),
    /// or `None` for a router built without one.
    pub fn metrics(&self) -> Option<&Arc<InferenceMetrics>> {
        self.metrics.as_ref()
    }

    /// Create a new state with the given embedding dimensionality.
    pub fn new(dim: usize) -> Self {
        Self::from_registry(EmbedderRegistry::new(dim))
    }

    /// Create a state around an already-configured registry.
    ///
    /// This is the seam a server uses to install a model backend, a batch cap
    /// and a metrics handle without this module needing a constructor per
    /// combination:
    ///
    /// ```ignore
    /// let state = EmbeddingAppState::from_registry(
    ///         EmbedderRegistry::new(dim)
    ///             .with_require_model_backend(true)
    ///             .with_model(Arc::clone(&model_embedder)),
    ///     )
    ///     .with_metrics(Arc::clone(&metrics));
    /// let router = create_embeddings_router_from_state(state);
    /// ```
    pub fn from_registry(registry: EmbedderRegistry) -> Self {
        // Catch `with_token_counter`/`with_token_embedder`
        // installed without the model backend that is meant to own them.
        // Checked here — after the caller's whole builder chain has run,
        // regardless of the order its calls were made in — rather than
        // inside the individual `with_*` setters, which cannot see whether a
        // later call in the same chain will still install a model.
        debug_assert!(
            !registry.has_orphaned_token_paths(),
            "EmbedderRegistry: with_token_counter/with_token_embedder was installed without \
             with_model_embedder (or with_model, which wires all of them together) -- \
             embedding_dim() will report default_dim (or the tfidf dimension) while the \
             token-aware paths return a possibly different dimension, so the response's \
             `dimension` field can disagree with the actual vector length. See \
             EmbedderRegistry::with_token_counter's docs."
        );
        Self {
            registry,
            metrics: None,
        }
    }

    /// Attach the shared metrics registry (builder) — `SV-25`.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Arc<InferenceMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }
}

// ─── Handler ──────────────────────────────────────────────────────────────────
//
// `ActiveRequestGuard` (decrements `active_requests` when the request future
// is dropped, however it ends — normal return, error, or a client that
// disconnected mid-request; `SV-25`, the same RAII shape `SV-08` needs for
// the streaming tail) used to be a fourth local copy here, duplicating
// `server/chat.rs`'s, `completions.rs`'s and `api_extensions.rs`'s own.
// `server`'s copy is `pub(crate)`, so this file imports
// `crate::server::ActiveRequestGuard` instead of carrying its own.

/// Handler for `POST /v1/embeddings`.
///
/// Computes dense vector representations for all supplied inputs and returns
/// an OpenAI-compatible response.
#[tracing::instrument(skip(state))]
pub async fn create_embeddings(
    State(state): State<Arc<EmbeddingAppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Response, StatusCode> {
    // SV-25: instrument BOTH branches — the `501` refusal below is a request
    // and an error like any other, and a deployment that only ever refuses
    // must still be visible in `/metrics` as such. All of it is skipped when
    // the router was built without a shared metrics handle; see the module
    // docs' "Metrics" section.
    let started = std::time::Instant::now();
    // `ActiveRequestGuard` has no `enter`-style constructor (see
    // `completions.rs`'s identical call site): increment then wrap, so the
    // wrapped value's `Drop` is the only thing that ever decrements.
    let _active_guard = state.metrics().map(|metrics| {
        metrics.active_requests.inc();
        ActiveRequestGuard(Arc::clone(metrics))
    });
    if let Some(metrics) = state.metrics() {
        metrics.requests_total.inc();
    }
    let outcome = create_embeddings_inner(&state, req).await;
    if let Some(metrics) = state.metrics() {
        let failed = match &outcome {
            Ok(response) => !response.status().is_success(),
            Err(_) => true,
        };
        if failed {
            metrics.errors_total.inc();
        }
        metrics
            .request_duration_seconds
            .observe(started.elapsed().as_secs_f64());
    }
    outcome
}

/// The standard text of the `501` a model-only registry answers with when no
/// model backend is installed.
pub const MODEL_BACKEND_MISSING_MESSAGE: &str =
    "no model-backed embedding backend is installed on this server; this deployment was \
     configured with EmbedderRegistry::with_require_model_backend(true), which disables the \
     stateless TF-IDF/identity fallback rather than answering with a non-semantic vector";

/// The `501 Not Implemented` of a registry that requires a model backend and
/// has none: `{"error": {"message", "type": "not_implemented_error", "param":
/// null, "code"}}` — `code` and a `": <reason>"` message suffix from the
/// recorded [`EmbedderUnavailable`], else [`EMBEDDINGS_UNAVAILABLE_CODE`]
/// and [`MODEL_BACKEND_MISSING_MESSAGE`] alone.
fn model_backend_missing_response(reason: Option<&EmbedderUnavailable>) -> Response {
    let (message, code) = match reason {
        Some(reason) => (
            format!("{MODEL_BACKEND_MISSING_MESSAGE}: {}", reason.message),
            reason.code(),
        ),
        None => (
            MODEL_BACKEND_MISSING_MESSAGE.to_string(),
            EMBEDDINGS_UNAVAILABLE_CODE,
        ),
    };
    crate::server::api_error::ApiError::new(StatusCode::NOT_IMPLEMENTED, message)
        .with_type(NOT_IMPLEMENTED_ERROR_TYPE)
        .with_code(code)
        .into_response()
}

/// Body of [`create_embeddings`], with the metrics wrapper peeled off so
/// every `return` inside it is still covered by one exit point.
async fn create_embeddings_inner(
    state: &Arc<EmbeddingAppState>,
    req: EmbeddingRequest,
) -> Result<Response, StatusCode> {
    // RT-08 / SV-02 correction (b): when this registry was explicitly
    // configured to require a model-backed `Embedder`
    // (`with_require_model_backend(true)`) and none is installed, refuse the
    // request outright rather than ever answering with a non-semantic
    // byte-hash/TF-IDF vector. The served router sets it and installs a real
    // `ModelEmbedder` whenever the server has an engine and a tokenizer, so
    // this branch is reached only by a deployment that genuinely has no
    // model to embed with — and the body names why, when the builder
    // recorded a reason.
    if state.registry().model_backend_required_but_missing() {
        return Ok(model_backend_missing_response(
            state.registry().unavailable_reason(),
        ));
    }

    if req.input.is_empty() {
        // Return the shared OpenAI-style error envelope (with a JSON body)
        // instead of a body-less status code, so `/v1/embeddings` validation
        // errors are parseable in the same shape as every other route
        // (finding `serve-api-10`).
        return Ok(crate::http_error::error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "input must not be empty",
            Some("input"),
        ));
    }

    if req.dimensions == Some(0) {
        return Ok(crate::http_error::error_response(
            StatusCode::BAD_REQUEST,
            "dimensions must be at least 1 (0 would truncate every embedding to an empty \
             vector)",
            Some("dimensions"),
        ));
    }

    // SV-02: an `encoding_format` other than the two documented values is
    // rejected by name instead of silently defaulting to `"float"` — the
    // same accept-and-drop class the server already refuses for
    // `/v1/completions`'s unsupported fields.
    let use_base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Ok(crate::http_error::error_response(
                StatusCode::BAD_REQUEST,
                format!("encoding_format must be \"float\" or \"base64\", got {other:?}"),
                Some("encoding_format"),
            ));
        }
    };

    let texts = req.input.as_strings();

    // sec-19: cap the batch instead of doing unbounded work for however many
    // inputs a client cares to send. Checked before any embedding work runs.
    let max_batch_size = state.registry().max_batch_size();
    if texts.len() > max_batch_size {
        return Ok(crate::http_error::error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "batch of {} inputs exceeds the maximum of {max_batch_size}; split the \
                 request into smaller batches",
                texts.len()
            ),
            Some("input"),
        ));
    }

    // sec-19: independently of the batch-size cap above, no single input may
    // exceed `max_input_len` characters — otherwise a one-element batch
    // carrying a multi-megabyte string passes the batch check unbounded.
    let max_input_len = state.registry().max_input_len();
    if let Some(offending_index) = texts.iter().position(|t| t.chars().count() > max_input_len) {
        return Ok(crate::http_error::error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "input[{offending_index}] has {} characters, exceeding the maximum of \
                 {max_input_len}; split it into smaller inputs",
                texts[offending_index].chars().count()
            ),
            Some("input"),
        ));
    }

    // A request that supplied token ids (`"input": [1, 2, 3]`) is embedded as
    // those ids when the backend can consume them, instead of re-tokenising
    // the decimal string `as_strings` renders — see
    // `EmbeddingInput::as_token_batches`.
    let token_batches = if state.registry().has_token_embedder() {
        req.input.as_token_batches()
    } else {
        None
    };

    // A TEXT request is tokenized (or, behind a
    // counter whose ids the backend cannot reuse, counted) exactly once, and
    // that one pass answers the length guard, the usage block and — with a
    // token-aware model backend — the embedding pass below. A raw token-id
    // request needs none of it: its ids are already the tokens.
    let text_tokens = match token_batches {
        Some(_) => TextTokens::Uncounted,
        None => state.registry().tokenize_texts(&texts),
    };

    // A TEXT input that tokenizes past the installed backend's ceiling is
    // refused outright, OpenAI-style, naming the real token count — rather
    // than silently truncated and billed in full. This governs the TEXT
    // path only: a request that supplied raw token ids (`token_batches`
    // above is `Some`) is NOT covered by this guard and keeps truncating —
    // see `crate::embed_engine`'s "Truncation" module-doc section for why
    // the two paths are allowed to disagree.
    let max_input_tokens = state.registry().max_input_tokens();
    if token_batches.is_none() {
        if let Some(max_tokens) = max_input_tokens {
            if let Some((index, n_tokens)) = text_tokens.first_exceeding(max_tokens) {
                return Ok(crate::server::ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "input[{index}] has {n_tokens} tokens, which exceeds the embedding \
                         backend's limit of {max_tokens}; shorten it or split it into smaller \
                         inputs"
                    ),
                )
                .with_param("input")
                .with_code("context_length_exceeded")
                .with_field("n_tokens", n_tokens)
                .with_field("max_tokens", max_tokens)
                .into_response());
            }
        }
    }

    // `usage.prompt_tokens` must count what is ACTUALLY embedded. For a
    // token-id request taking the `token_batches` path above that is the ids
    // actually embedded: `EmbedderRegistry::embed_token_batches` (via
    // `TokenSequenceEmbedder`/`ModelEmbedder::embed_tokens`) truncates each
    // sequence to `max_input_tokens` when one is known, so charging the full
    // supplied length here — as a previous version did — would bill for ids
    // the model never saw whenever a sequence exceeds that ceiling (the
    // same over-billing class as the text path, but this path keeps truncating
    // by design rather than erroring, so it is the accounting, not the
    // truncation, that is fixed). Counting the decimal rendering instead of
    // the ids would ALSO be wrong regardless of truncation (`[10, 20, 30]`
    // is 3 tokens; `"10 20 30"` tokenises to 8 under the char-level
    // vocabulary, and to something else again under a real BPE). Otherwise
    // it is the model's own tokenizer — read off `text_tokens`, the one pass
    // above, not a second tokenization — or the historical whitespace word
    // count when no backend owns one (see
    // `EmbedderRegistry::count_prompt_tokens`). Computed from `&texts` /
    // `&token_batches` before either is moved into the blocking task below,
    // so this stays a borrow rather than a clone of the batch.
    let prompt_tokens: usize = match token_batches.as_ref() {
        Some(batches) => batches
            .iter()
            .map(|ids| match max_input_tokens {
                Some(max_tokens) => ids.len().min(max_tokens),
                None => ids.len(),
            })
            .sum(),
        None => text_tokens.prompt_tokens(&texts),
    };
    if let Some(metrics) = state.metrics() {
        metrics.prompt_tokens_total.inc_by(prompt_tokens as u64);
    }

    // RT-08 / SV-02 / sec-19: no fitting happens here. A previous version
    // fit TF-IDF on the fly from whatever texts.len() >= 2 the CURRENT client
    // happened to submit, which made the vector space (and even the output
    // dimension) a function of prior, unrelated requests — two identical
    // inputs embedded a request apart were not comparable, and a fresh
    // process answered the very first multi-text request differently from
    // every later one. `embed_texts` now always uses whichever backend was
    // installed administratively (see the module docs), so it is a pure
    // function of `texts` for the lifetime of this `EmbedderRegistry`.
    //
    // sec-19: this (TF-IDF and model-backed) computation is CPU-bound — a
    // model-backed batch is a full forward pass per input — and must not run
    // directly on the async handler's tokio worker thread; the same defect
    // class `sec-03` already fixes for `/v1/completions`. `state` is cloned
    // (cheap: `Arc`) and moved into the blocking task; `texts` /
    // `token_batches` / `text_tokens` are moved too (not cloned), since
    // nothing after this point needs the originals.
    //
    // Text whose ids the one pass above already produced from the model
    // backend itself (`TextTokens::Tokenized`) is embedded from those ids —
    // the same vectors `embed_texts` would compute (see
    // `EmbedderRegistry::with_token_aware_model`'s contract), without
    // tokenizing every input a second time.
    let state_for_embed = Arc::clone(state);
    let raw_embeddings = match tokio::task::spawn_blocking(move || {
        let registry = state_for_embed.registry();
        let ids = match (token_batches, text_tokens) {
            (Some(batches), _) | (None, TextTokens::Tokenized(batches)) => batches,
            (None, _) => return registry.embed_texts(&texts),
        };
        registry
            .embed_token_batches(&ids)
            // Both arms above only hand over ids when a token embedder is
            // installed (`has_token_embedder()` for request ids,
            // `with_token_aware_model` for tokenized text) and the registry
            // is immutable behind an `Arc`, so this `None` arm is
            // unreachable; falling back to the text path rather than
            // unwrapping keeps it panic-free anyway.
            .unwrap_or_else(|| registry.embed_texts(&texts))
    })
    .await
    {
        Ok(embeddings) => embeddings,
        Err(join_error) => {
            tracing::error!(error = %join_error, "embedding computation task panicked");
            return Ok(crate::http_error::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("embedding computation failed: {join_error}"),
                None,
            ));
        }
    };

    // Report which backend actually answered rather than echoing the
    // client's unverified `model` request field (RT-08's fix item 3: "make
    // the response `model` field report which backend answered").
    let model_name = format!("bonsai-embeddings-{}", state.registry().backend_name());

    // RT-08 fix item 1's trailing clause ("document the dimension and the
    // normalisation in the response"): `natural_dim` is uniform across the
    // whole batch (every backend has one fixed `embedding_dim()`), so the
    // post-truncation dimension every item in `data` will have can be
    // computed once, up front, rather than re-derived per item.
    let natural_dim = state.registry().embedding_dim();
    let final_dim = req
        .dimensions
        .filter(|&d| d < natural_dim)
        .unwrap_or(natural_dim);
    // Tracks whether any item fell back to an all-zero vector (a per-item
    // embedding failure), so `normalized` below is derived from what
    // actually happened rather than optimistically hardcoded to `true`.
    let mut any_zero_vector = false;

    let data: Vec<EmbeddingObject> = raw_embeddings
        .into_iter()
        .enumerate()
        .map(|(index, mut vec)| {
            // Optionally truncate to requested dimensions, re-normalising the
            // truncated vector (Matryoshka truncation) so it stays unit-length
            // — matching the OpenAI contract. A prior version truncated
            // without renormalising (SV-02 correction (a)), silently handing
            // back a non-unit vector whenever `dimensions` was set.
            if let Some(dim) = req.dimensions {
                if dim < vec.len() {
                    vec.truncate(dim);
                    l2_normalize(&mut vec);
                }
            }

            // SV-02 (blocking follow-up to correction (a)): this must run
            // AFTER truncation, not before. `any_zero_vector` has to describe
            // the vector this response actually ships, and truncating a
            // genuine unit vector down to a prefix that happens to be all
            // zero (every retained component was zero pre-truncation) leaves
            // it at all-zero -- `l2_normalize` is a no-op below its 1e-10
            // norm guard, so nothing re-checks after truncation unless this
            // does. Checking the pre-truncation vector here made `normalized`
            // stale relative to the shipped vector: a real TF-IDF query whose
            // only non-zero column landed past `dimensions` reported
            // `normalized: true` for a `[0.0, 0.0]` response.
            if vec.iter().all(|x| *x == 0.0) {
                any_zero_vector = true;
            }

            let embedding = if use_base64 {
                EmbeddingData::Base64(EmbedderRegistry::encode_base64(&vec))
            } else {
                EmbeddingData::Float(vec)
            };

            EmbeddingObject {
                object: "embedding".to_owned(),
                embedding,
                index,
            }
        })
        .collect();

    let response = EmbeddingResponse {
        object: "list".to_owned(),
        data,
        model: model_name,
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
        dimension: final_dim,
        normalized: !any_zero_vector,
    };

    Ok(Json(response).into_response())
}

// ─── Router factory ───────────────────────────────────────────────────────────

/// Build a standalone Axum router for the embeddings endpoint around an
/// already-configured [`EmbeddingAppState`].
///
/// This is the one constructor that can express every combination — model
/// backend, `require_model_backend`, batch caps, shared metrics — and the one
/// the served router uses. Mount it at the root with [`Router::merge`] or nest
/// it under a path prefix with [`Router::nest`]; it exposes a single route:
///
/// ```text
/// POST /v1/embeddings
/// ```
///
/// The three `create_embeddings_router*` helpers below are thin,
/// signature-stable wrappers over it, kept because they are `pub` and have
/// callers.
pub fn create_embeddings_router_from_state(state: EmbeddingAppState) -> Router {
    Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(Arc::new(state))
}

/// Build a standalone embeddings router serving the deterministic stateless
/// fallback backend (see the module docs' "Backends and determinism" section).
///
/// **Not what the served server mounts.** This is the library constructor: a
/// bare dimension, no engine, no model, no metrics, and therefore the
/// `IdentityEmbedder` byte-hash test double as the backend that actually
/// answers. Its signature is deliberately frozen — it is `pub`, reachable by
/// library embedders who have no model to hand it, and exercised by this
/// crate's own router tests. A server with a real model uses
/// [`create_embeddings_router_from_state`]; a caller with some other
/// [`Embedder`] uses [`create_embeddings_router_with_model`].
pub fn create_embeddings_router(dim: usize) -> Router {
    create_embeddings_router_from_state(EmbeddingAppState::new(dim))
}

/// Like [`create_embeddings_router`], but installs `model_embedder` as the
/// registry's highest-priority backend (see
/// [`EmbedderRegistry::with_model_embedder`]).
///
/// Takes any [`Embedder`], so a caller can plug in an embedding backend this
/// crate knows nothing about. For the project's own [`ModelEmbedder`], prefer
/// [`EmbedderRegistry::with_model`] through
/// [`create_embeddings_router_from_state`]: that additionally wires the exact
/// token counter and the native token-id path, which a bare `dyn Embedder`
/// cannot express.
pub fn create_embeddings_router_with_model(
    dim: usize,
    model_embedder: Arc<dyn Embedder>,
) -> Router {
    create_embeddings_router_from_state(EmbeddingAppState::from_registry(
        EmbedderRegistry::new(dim).with_model_embedder(model_embedder),
    ))
}

/// Like [`create_embeddings_router`], but configures the registry with
/// [`EmbedderRegistry::with_require_model_backend`] and installs no
/// model-backed [`Embedder`] — every request therefore gets an honest `501`
/// (`create_embeddings`'s `model_backend_required_but_missing` check)
/// instead of a silently-wrong, non-semantic `IdentityEmbedder` byte-hash
/// vector.
///
/// Answering with a byte-hash vector that merely *looks* like an embedding
/// is worse than refusing outright (`RT-08`). With
/// `BonsaiModel::forward_hidden` and [`ModelEmbedder`], a served deployment
/// normally *has* a model backend, and this constructor is the honest answer
/// only for one that does not (no engine loaded, or no tokenizer to encode
/// with). [`create_embeddings_router`] itself is unchanged, so this does not
/// silently alter its documented behavior for anyone already depending on the
/// stateless fallback.
pub fn create_embeddings_router_requiring_model(dim: usize) -> Router {
    create_embeddings_router_from_state(EmbeddingAppState::from_registry(
        EmbedderRegistry::new(dim).with_require_model_backend(true),
    ))
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "embeddings_tests.rs"]
mod tests;
