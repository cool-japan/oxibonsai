//! OpenAI v1 embeddings endpoint.
//!
//! Implements `POST /v1/embeddings` — an OpenAI-compatible embedding API that
//! converts text (or token ID arrays) into dense float vectors.
//!
//! # Backends and determinism (`RT-08` / `SV-02` / `sec-19`)
//!
//! The [`EmbedderRegistry`] manages three backends, tried in priority order:
//!
//! 1. **A model-backed [`Embedder`]** (see [`EmbedderRegistry::with_model_embedder`])
//!    — the intended production default once one is installed. **No live HTTP
//!    path installs one today**, because reaching a real model from this file
//!    needs three changes outside this crate's / this file's ownership: (a) a
//!    CPU-only `BonsaiModel::forward_hidden(&[u32]) -> ModelResult<Vec<f32>>`
//!    seam in `crates/oxibonsai-model/src/model/types/mod.rs` returning the
//!    post-`output_norm` hidden state instead of logits (the Metal fast path
//!    fuses the LM head into the same dispatch, so this needs either a
//!    non-fused variant or a CPU-only path); (b) an `InferenceEngine`
//!    accessor in `crates/oxibonsai-runtime/src/engine.rs` that calls it and
//!    mean-pools + L2-normalises the result; (c) `server.rs`'s
//!    `create_embeddings_router(512)` call site switched to
//!    [`create_embeddings_router_with_model`] once an `Embedder` wrapping
//!    that accessor exists. [`create_embeddings_router_with_model`] is the
//!    wiring point for whoever lands (a)–(c); it does not require any change
//!    to [`create_embeddings_router`]'s signature or its existing caller.
//!
//!    **Interim substitute while (a)–(c) are outstanding** (`SV-02`
//!    correction (b)): [`EmbedderRegistry::with_require_model_backend`] lets
//!    a caller who does *not* want the stateless fallback to ever answer
//!    disable it — [`create_embeddings`] then returns `501 Not Implemented`
//!    instead of a `200` carrying a byte-hash/TF-IDF vector whenever no
//!    model backend is installed. The check lives inside this file's own
//!    handler, not at any call site, so it cannot be silently bypassed by a
//!    caller that forgets to check first. **`server.rs`'s live call site now
//!    opts into it** (orchestrator decision D-1, wave 2.5): the running
//!    server calls [`create_embeddings_router_requiring_model`] rather than
//!    the bare [`create_embeddings_router`], so **`/v1/embeddings` on the
//!    running server honestly answers `501` today** (no model-backed
//!    `Embedder` is installed) instead of a silently-wrong `200` carrying an
//!    `IdentityEmbedder` byte-hash vector. `tests/embeddings_tests.rs`
//!    (outside this package's `owned_files`) still asserts `200` against
//!    the bare [`create_embeddings_router`] directly — that is intentional
//!    and unaffected, since that constructor's own documented behavior is
//!    unchanged; only the running server's call site moved to the stricter
//!    one. See [`create_embeddings_router_requiring_model`]'s own doc
//!    comment for the real-embedder follow-up (a)–(c) above still needs.
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
//!    available. This is what [`create_embeddings_router`] actually serves
//!    today for every request, since nothing pre-fits TF-IDF and no model is
//!    wired in: a pure function of the input bytes, so embedding the same
//!    text twice — in the same request, a different request, or a different
//!    process — always returns a bit-identical vector.
//!
//! The response's `model` field reports which backend actually answered
//! (`"bonsai-embeddings-model"` / `"-tfidf"` / `"-identity"`, see
//! [`EmbedderRegistry::backend_name`]) rather than echoing the client's
//! (arbitrary, unverified) `model` request field, so a caller can tell a
//! byte hash from a real embedding without reading this module's source.
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
    tfidf: std::sync::Mutex<Option<TfIdfEmbedder>>,
    identity: IdentityEmbedder,
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
            tfidf: std::sync::Mutex::new(None),
            identity,
        }
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
    pub fn embed_texts(&self, texts: &[String]) -> Vec<Vec<f32>> {
        if let Some(ref model) = self.model {
            let dim = model.embedding_dim();
            return texts
                .iter()
                .map(|t| model.embed(t).unwrap_or_else(|_| vec![0.0; dim]))
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
pub struct EmbeddingAppState {
    /// The active embedding registry.
    pub registry: EmbedderRegistry,
}

impl EmbeddingAppState {
    /// Create a new state with the given embedding dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            registry: EmbedderRegistry::new(dim),
        }
    }
}

// ─── Handler ──────────────────────────────────────────────────────────────────

/// Handler for `POST /v1/embeddings`.
///
/// Computes dense vector representations for all supplied inputs and returns
/// an OpenAI-compatible response.
#[tracing::instrument(skip(state))]
pub async fn create_embeddings(
    State(state): State<Arc<EmbeddingAppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Response, StatusCode> {
    // RT-08 / SV-02 correction (b): when this registry was explicitly
    // configured to require a model-backed `Embedder`
    // (`with_require_model_backend(true)`) and none is installed, refuse the
    // request outright rather than ever answering with a non-semantic
    // byte-hash/TF-IDF vector. This is the sanctioned interim substitute for
    // a live model-backed default; `server.rs`'s live call site opts into
    // it (see the module docs for the D-1 background and the real-embedder
    // follow-up still needed).
    if state.registry.model_backend_required_but_missing() {
        return Ok(crate::http_error::error_response(
            StatusCode::NOT_IMPLEMENTED,
            "no model-backed embedding backend is installed on this server; this deployment \
             was configured with EmbedderRegistry::with_require_model_backend(true), which \
             disables the stateless TF-IDF/identity fallback rather than answering with a \
             non-semantic vector",
            None,
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
    // same accept-and-drop class this package already refuses for
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
    let max_batch_size = state.registry.max_batch_size();
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
    let max_input_len = state.registry.max_input_len();
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

    // Count tokens for usage: approximate as whitespace-split word count.
    // Computed from `&texts` before `texts` is moved into the blocking task
    // below, so this stays a cheap borrow rather than a clone of the batch.
    let prompt_tokens: usize = texts
        .iter()
        .map(|t| t.split_whitespace().count().max(1))
        .sum();

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
    // sec-19: this (TF-IDF and, once wired, model-backed) computation is
    // CPU-bound and must not run directly on the async handler's tokio
    // worker thread — the same defect class `sec-03` already fixes for
    // `/v1/completions`. `state` is cloned (cheap: `Arc`) and moved into the
    // blocking task; `texts` is moved too (not cloned), since nothing after
    // this point needs the original `Vec<String>`.
    let state_for_embed = Arc::clone(&state);
    let raw_embeddings =
        match tokio::task::spawn_blocking(move || state_for_embed.registry.embed_texts(&texts))
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
    let model_name = format!("bonsai-embeddings-{}", state.registry.backend_name());

    // RT-08 fix item 1's trailing clause ("document the dimension and the
    // normalisation in the response"): `natural_dim` is uniform across the
    // whole batch (every backend has one fixed `embedding_dim()`), so the
    // post-truncation dimension every item in `data` will have can be
    // computed once, up front, rather than re-derived per item.
    let natural_dim = state.registry.embedding_dim();
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

/// Build a standalone Axum router for the embeddings endpoint, serving the
/// deterministic stateless default backend (see the module docs' "Backends
/// and determinism" section).
///
/// Mount this at the root with [`Router::merge`] or nest it under a path
/// prefix with [`Router::nest`].  The router exposes a single route:
///
/// ```text
/// POST /v1/embeddings
/// ```
///
/// This is the constructor `server.rs` calls; its signature is kept stable
/// (a bare dimension, no engine/model handle) both because it is `pub` and
/// reachable by library embedders who have no model to hand it, and because
/// changing it would require a matching change to that call site. A caller
/// that *does* have a model-backed [`Embedder`] should use
/// [`create_embeddings_router_with_model`] instead.
pub fn create_embeddings_router(dim: usize) -> Router {
    let state = Arc::new(EmbeddingAppState::new(dim));
    Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state)
}

/// Like [`create_embeddings_router`], but installs `model_embedder` as the
/// registry's highest-priority backend (see
/// [`EmbedderRegistry::with_model_embedder`]).
///
/// This is the wiring point for a real neural embedding path once one
/// exists: build an [`Embedder`] over a loaded model (e.g. mean-pooled,
/// L2-normalised hidden states from `BonsaiModel::forward_hidden`, once that
/// seam exists — see the module docs) and pass it here. No HTTP entry point
/// in this crate calls this constructor today.
pub fn create_embeddings_router_with_model(
    dim: usize,
    model_embedder: Arc<dyn Embedder>,
) -> Router {
    let state = Arc::new(EmbeddingAppState {
        registry: EmbedderRegistry::new(dim).with_model_embedder(model_embedder),
    });
    Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state)
}

/// Like [`create_embeddings_router`], but configures the registry with
/// [`EmbedderRegistry::with_require_model_backend`] and installs no
/// model-backed [`Embedder`] — every request therefore gets an honest `501`
/// (`create_embeddings`'s `model_backend_required_but_missing` check)
/// instead of a silently-wrong, non-semantic `IdentityEmbedder` byte-hash
/// vector.
///
/// This is orchestrator decision D-1 (wave 2.5, `RT-EMBEDDINGS` blocking 1):
/// a real model-backed embedder needs `BonsaiModel::forward_hidden` (a
/// non-fused, post-`output_norm` hidden-state accessor that does not yet
/// exist — outside `oxibonsai-runtime` entirely, in `oxibonsai-model`) plus
/// an `InferenceEngine::embed_hidden` seam, which the decision explicitly
/// scoped out of this wave as "a genuine feature, not a fix". Given that,
/// answering with a byte-hash vector that merely *looks* like an embedding
/// is worse than refusing outright, so `server.rs`'s `/v1/embeddings` call
/// site uses this constructor rather than [`create_embeddings_router`].
/// [`create_embeddings_router`] itself is unchanged (and still used by
/// library embedders and by this module's own test suite below) precisely
/// so this change does not silently alter its documented behavior for
/// anyone already depending on the stateless fallback.
pub fn create_embeddings_router_requiring_model(dim: usize) -> Router {
    let state = Arc::new(EmbeddingAppState {
        registry: EmbedderRegistry::new(dim).with_require_model_backend(true),
    });
    Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── EmbeddingInput ────────────────────────────────────────────────────────

    #[test]
    fn embedding_input_single_as_strings() {
        let input = EmbeddingInput::Single("hello world".to_string());
        assert_eq!(input.as_strings(), vec!["hello world"]);
        assert_eq!(input.len(), 1);
        assert!(!input.is_empty());
    }

    #[test]
    fn embedding_input_batch_as_strings() {
        let input = EmbeddingInput::Batch(vec!["foo".to_string(), "bar".to_string()]);
        let strings = input.as_strings();
        assert_eq!(strings.len(), 2);
        assert_eq!(strings[0], "foo");
        assert_eq!(strings[1], "bar");
        assert_eq!(input.len(), 2);
    }

    #[test]
    fn embedding_input_token_ids_as_strings() {
        let input = EmbeddingInput::TokenIds(vec![1u32, 2, 3]);
        let strings = input.as_strings();
        assert_eq!(strings.len(), 1);
        assert_eq!(strings[0], "1 2 3");
    }

    #[test]
    fn embedding_input_batch_token_ids_as_strings() {
        let input = EmbeddingInput::BatchTokenIds(vec![vec![10u32, 20], vec![30u32]]);
        let strings = input.as_strings();
        assert_eq!(strings.len(), 2);
        assert_eq!(strings[0], "10 20");
        assert_eq!(strings[1], "30");
    }

    #[test]
    fn embedding_input_empty_batch_is_empty() {
        let input = EmbeddingInput::Batch(vec![]);
        assert!(input.is_empty());
        assert_eq!(input.len(), 0);
    }

    // ── EmbedderRegistry ─────────────────────────────────────────────────────

    #[test]
    fn embedder_registry_basic_embed() {
        let registry = EmbedderRegistry::new(32);
        let texts = vec!["hello world".to_string(), "foo bar baz".to_string()];
        let embeddings = registry.embed_texts(&texts);
        assert_eq!(embeddings.len(), 2);
        // Each embedding must have exactly `default_dim` elements.
        for emb in &embeddings {
            assert_eq!(emb.len(), 32, "expected 32 dimensions, got {}", emb.len());
        }
    }

    #[test]
    fn embedder_registry_tfidf_fit_changes_dim() {
        let registry = EmbedderRegistry::new(64);
        let corpus: Vec<String> = (0..20)
            .map(|i| format!("document number {i} with some unique words term{i}"))
            .collect();
        registry.fit_tfidf(&corpus);
        // After fitting the dimension comes from the TF-IDF vocabulary.
        let dim = registry.embedding_dim();
        assert!(dim > 0, "expected positive dimension after fit");
    }

    #[test]
    fn embedder_registry_fit_empty_corpus_is_noop() {
        let registry = EmbedderRegistry::new(16);
        registry.fit_tfidf(&[]);
        // Should still use IdentityEmbedder (dim == default_dim).
        assert_eq!(registry.embedding_dim(), 16);
    }

    #[test]
    fn embedder_registry_embed_after_fit() {
        let registry = EmbedderRegistry::new(32);
        let corpus: Vec<String> = vec![
            "the quick brown fox".to_string(),
            "jumped over the lazy dog".to_string(),
            "the fox and the dog".to_string(),
        ];
        registry.fit_tfidf(&corpus);
        let embeddings = registry.embed_texts(&corpus);
        for emb in &embeddings {
            assert!(!emb.is_empty(), "embedding must not be empty after fit");
        }
    }

    // ── Poisoned-lock recovery (finding #70) ─────────────────────────────────

    /// Regression test: a panic on another thread while holding
    /// `EmbedderRegistry`'s internal `tfidf` mutex must not turn every
    /// subsequent `POST /v1/embeddings` request into a permanent panic.
    /// Before the fix, `embed_texts`/`fit_tfidf`/`embedding_dim` all used
    /// `.lock().expect("... poisoned")`, so a single unrelated panic while
    /// holding the lock would wedge this (server-reachable) route for the
    /// rest of the process lifetime.
    #[test]
    fn embedder_registry_recovers_from_poisoned_tfidf_lock() {
        let registry = Arc::new(EmbedderRegistry::new(16));

        // Poison the `tfidf` mutex from a background thread that panics
        // while holding the lock.
        {
            let registry = Arc::clone(&registry);
            let handle = std::thread::spawn(move || {
                let _guard = registry.tfidf.lock().expect("lock for poisoning");
                panic!("intentional panic to poison the tfidf mutex");
            });
            let result = handle.join();
            assert!(result.is_err(), "background thread should have panicked");
        }

        // The mutex is now poisoned. Operations that touch it must recover
        // instead of panicking.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let corpus: Vec<String> = vec![
                "the quick brown fox".to_string(),
                "jumped over the lazy dog".to_string(),
            ];
            registry.fit_tfidf(&corpus);
            let embeddings = registry.embed_texts(&corpus);
            let dim = registry.embedding_dim();
            (embeddings, dim)
        }));

        assert!(
            outcome.is_ok(),
            "operations on an EmbedderRegistry with a poisoned `tfidf` mutex must not panic"
        );
        let (embeddings, dim) = outcome.expect("checked is_ok above");
        assert_eq!(embeddings.len(), 2);
        assert!(
            dim > 0,
            "embedding_dim should still be usable after poison recovery"
        );
    }

    // ── encode_base64 (finding serve-api-08: must be real RFC 4648 base64,
    //    not hex) ──────────────────────────────────────────────────────────

    #[test]
    fn encode_base64_non_empty() {
        let vec = vec![1.0f32, 0.5f32, -1.0f32];
        let encoded = EmbedderRegistry::encode_base64(&vec);
        // 3 f32 values → 12 bytes → 12/3*4 = 16 base64 chars, no padding.
        assert_eq!(
            encoded.len(),
            16,
            "expected 16 base64 chars for 3 f32 values (12 bytes), got {}",
            encoded.len()
        );
        assert!(!encoded.is_empty());
        // Every character must be a valid RFC 4648 base64 alphabet character.
        assert!(encoded
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'='));
    }

    #[test]
    fn encode_base64_empty_input() {
        let encoded = EmbedderRegistry::encode_base64(&[]);
        assert!(encoded.is_empty());
    }

    #[test]
    fn encode_base64_deterministic() {
        let vec = vec![std::f32::consts::PI, 2.71f32];
        let a = EmbedderRegistry::encode_base64(&vec);
        let b = EmbedderRegistry::encode_base64(&vec);
        assert_eq!(a, b, "encoding must be deterministic");
    }

    /// Regression test for finding `serve-api-08`: the previous
    /// implementation emitted lowercase hex ("0000803f") under the
    /// `encoding_format: "base64"` contract. The expected string below was
    /// computed independently with Python's standard `base64` module
    /// (`base64.b64encode(struct.pack("<f", 1.0))` == `b"AACAPw=="`), so this
    /// test verifies interoperability with a real RFC 4648 base64 decoder,
    /// not just internal self-consistency.
    #[test]
    fn encode_base64_known_value_matches_real_base64_decoder() {
        // f32::to_le_bytes(1.0) == [0x00, 0x00, 0x80, 0x3f]
        let vec = vec![1.0f32];
        let encoded = EmbedderRegistry::encode_base64(&vec);
        assert_eq!(encoded, "AACAPw==");
    }

    /// Second independently-computed known vector: `base64.b64encode(
    /// struct.pack("<2f", 1.0, 0.5))` == `b"AACAPwAAAD8="`.
    #[test]
    fn encode_base64_known_value_two_floats() {
        let vec = vec![1.0f32, 0.5f32];
        let encoded = EmbedderRegistry::encode_base64(&vec);
        assert_eq!(encoded, "AACAPwAAAD8=");
    }

    /// Full round-trip: encode with the production encoder, decode with an
    /// independent, standard-conformant base64 decoder (implemented here
    /// for the test only), and confirm the reconstructed `f32` bytes match
    /// the originals exactly. This is the "real decoder" check the finding
    /// asked for: any RFC 4648-conformant decoder (including a real
    /// `base64.b64decode`) must be able to reverse our output.
    #[test]
    fn encode_base64_round_trips_through_independent_decoder() {
        let original = vec![1.0f32, -2.5f32, 0.0f32, std::f32::consts::PI, -999.125f32];
        let encoded = EmbedderRegistry::encode_base64(&original);
        let decoded_bytes = test_base64_decode(&encoded);

        let mut expected_bytes = Vec::with_capacity(original.len() * 4);
        for v in &original {
            expected_bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            decoded_bytes, expected_bytes,
            "round-trip through an independent base64 decoder must reproduce \
             the exact little-endian f32 byte sequence"
        );

        // Reinterpret the decoded bytes as f32 values and confirm they match.
        let decoded_floats: Vec<f32> = decoded_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded_floats, original);
    }

    /// Minimal standard-conformant RFC 4648 base64 decoder, used only to
    /// independently verify [`EmbedderRegistry::encode_base64`]'s output in
    /// tests (kept separate from the production encoder so the test does
    /// not just check the encoder against itself).
    fn test_base64_decode(s: &str) -> Vec<u8> {
        fn value_of(c: u8) -> u32 {
            match c {
                b'A'..=b'Z' => (c - b'A') as u32,
                b'a'..=b'z' => (c - b'a' + 26) as u32,
                b'0'..=b'9' => (c - b'0' + 52) as u32,
                b'+' => 62,
                b'/' => 63,
                _ => 0, // padding '=' contributes no bits
            }
        }
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
        for chunk in bytes.chunks(4) {
            let pad = chunk.iter().filter(|&&b| b == b'=').count();
            let c0 = value_of(chunk[0]);
            let c1 = value_of(*chunk.get(1).unwrap_or(&b'A'));
            let c2 = value_of(*chunk.get(2).unwrap_or(&b'A'));
            let c3 = value_of(*chunk.get(3).unwrap_or(&b'A'));
            let packed = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;
            let b0 = ((packed >> 16) & 0xff) as u8;
            let b1 = ((packed >> 8) & 0xff) as u8;
            let b2 = (packed & 0xff) as u8;
            match pad {
                0 => out.extend_from_slice(&[b0, b1, b2]),
                1 => out.extend_from_slice(&[b0, b1]),
                2 => out.push(b0),
                _ => {}
            }
        }
        out
    }

    // ── EmbeddingResponse serialisation ──────────────────────────────────────

    #[test]
    fn embedding_response_serialises_correctly() {
        let resp = EmbeddingResponse {
            object: "list".to_owned(),
            data: vec![EmbeddingObject {
                object: "embedding".to_owned(),
                embedding: EmbeddingData::Float(vec![0.1, 0.2]),
                index: 0,
            }],
            model: "bonsai-embeddings".to_owned(),
            usage: EmbeddingUsage {
                prompt_tokens: 3,
                total_tokens: 3,
            },
            dimension: 2,
            normalized: true,
        };
        let json = serde_json::to_string(&resp).expect("serialisation must succeed");
        assert!(json.contains("\"object\":\"list\""));
        assert!(json.contains("\"object\":\"embedding\""));
        assert!(json.contains("\"index\":0"));
        assert!(json.contains("\"dimension\":2"));
        assert!(json.contains("\"normalized\":true"));
    }

    // ── RT-08 / SV-02 / sec-19: statelessness, batch cap, backend naming ────

    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// POST `body` to `/v1/embeddings` on `app` and return (status, JSON).
    async fn post(app: Router, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let req = Request::post("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&body).expect("body serialisation"),
            ))
            .expect("request build");
        let resp = app.oneshot(req).await.expect("response");
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    /// The core acceptance property: embedding the same text twice across two
    /// *separate* requests on the *same* router returns a bit-identical
    /// vector. Before the fix, a request with >= 2 texts silently fit TF-IDF
    /// from its own input, so a second request's embedding of the same text
    /// (now scored against a vocabulary the first request installed) could
    /// differ in both value and dimension from the first.
    #[tokio::test]
    async fn embedding_same_text_twice_across_two_requests_is_bit_identical() {
        let app = create_embeddings_router(32);

        // First request: a multi-text batch that, under the old auto-fit
        // behaviour, would have installed a TF-IDF vocabulary derived from
        // *these specific texts* — poisoning every later request.
        let (status1, _) = post(
            app.clone(),
            serde_json::json!({ "input": ["alpha document one", "beta document two"] }),
        )
        .await;
        assert_eq!(status1, StatusCode::OK);

        // Second, unrelated request embeds a fixed probe text.
        let (status_a, json_a) =
            post(app.clone(), serde_json::json!({ "input": "probe text" })).await;
        assert_eq!(status_a, StatusCode::OK);

        // A third request, with yet another multi-text batch that would
        // previously have re-fit TF-IDF to a *different* vocabulary.
        let (status2, _) = post(
            app.clone(),
            serde_json::json!({ "input": ["gamma document three", "delta document four", "epsilon"] }),
        )
        .await;
        assert_eq!(status2, StatusCode::OK);

        // Re-embedding the exact same probe text must be bit-identical.
        let (status_b, json_b) = post(app, serde_json::json!({ "input": "probe text" })).await;
        assert_eq!(status_b, StatusCode::OK);

        assert_eq!(
            json_a["data"][0]["embedding"], json_b["data"][0]["embedding"],
            "the same text embedded a request apart must return a bit-identical vector"
        );
        assert_eq!(
            json_a["data"][0]["embedding"]
                .as_array()
                .expect("array")
                .len(),
            json_b["data"][0]["embedding"]
                .as_array()
                .expect("array")
                .len(),
            "the embedding dimension must not drift across requests either"
        );
    }

    /// Same property at the library level (no HTTP layer): two independently
    /// constructed registries (simulating two separate process lifetimes)
    /// must embed the same text identically, since neither one ever mutates
    /// from request content.
    #[test]
    fn two_independent_registries_embed_the_same_text_identically() {
        let a = EmbedderRegistry::new(24);
        let b = EmbedderRegistry::new(24);
        let out_a = a.embed_texts(&["consistent text".to_string()]);
        let out_b = b.embed_texts(&["consistent text".to_string()]);
        assert_eq!(out_a, out_b);
    }

    /// `fit_tfidf` remains available as an explicit, administrative
    /// operation (existing callers, including the sibling
    /// `tests/embeddings_tests.rs` integration suite, rely on this) — the fix
    /// removes the *automatic* per-request call from the handler, not the
    /// method itself.
    #[test]
    fn fit_tfidf_remains_an_explicit_public_operation() {
        let registry = EmbedderRegistry::new(50);
        assert_eq!(registry.backend_name(), "identity");
        let corpus: Vec<String> = (0..5).map(|i| format!("doc {i} content")).collect();
        registry.fit_tfidf(&corpus);
        assert_eq!(registry.backend_name(), "tfidf");
    }

    /// A batch over the (default) cap is rejected with `400`, not silently
    /// truncated or accepted at unbounded cost.
    #[tokio::test]
    async fn batch_over_the_default_cap_is_rejected_with_400() {
        let app = create_embeddings_router(16);
        let inputs: Vec<String> = (0..(DEFAULT_MAX_EMBEDDING_BATCH_SIZE + 1))
            .map(|i| format!("text {i}"))
            .collect();
        let (status, _) = post(app, serde_json::json!({ "input": inputs })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// A batch at or under the cap is accepted.
    #[tokio::test]
    async fn batch_at_the_cap_is_accepted() {
        let app = create_embeddings_router(16);
        let inputs: Vec<String> = (0..4).map(|i| format!("text {i}")).collect();
        let (status, json) = post(app, serde_json::json!({ "input": inputs })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["data"].as_array().expect("data array").len(), 4);
    }

    /// `with_max_batch_size` is honoured by the handler (via
    /// [`create_embeddings_router_with_model`], the only router constructor
    /// that exposes registry construction to this test without a model
    /// dependency — installing a trivial passthrough embedder here only to
    /// reach the registry builder, and asserting the cap check fires before
    /// any embedding happens).
    #[tokio::test]
    async fn configured_max_batch_size_is_enforced() {
        struct TinyEmbedder;
        impl Embedder for TinyEmbedder {
            fn embed(&self, _text: &str) -> Result<Vec<f32>, oxibonsai_rag::error::RagError> {
                Ok(vec![1.0])
            }
            fn embedding_dim(&self) -> usize {
                1
            }
        }
        let state = Arc::new(EmbeddingAppState {
            registry: EmbedderRegistry::new(4)
                .with_max_batch_size(2)
                .with_model_embedder(Arc::new(TinyEmbedder)),
        });
        let app = Router::new()
            .route("/v1/embeddings", axum::routing::post(create_embeddings))
            .with_state(state);

        let (status_ok, _) = post(app.clone(), serde_json::json!({ "input": ["a", "b"] })).await;
        assert_eq!(status_ok, StatusCode::OK);

        let (status_over, _) = post(app, serde_json::json!({ "input": ["a", "b", "c"] })).await;
        assert_eq!(status_over, StatusCode::BAD_REQUEST);
    }

    /// `with_max_batch_size(0)` clamps to `1` rather than making the endpoint
    /// refuse every request, including a single-input one.
    #[test]
    fn max_batch_size_zero_is_clamped_to_one() {
        let registry = EmbedderRegistry::new(8).with_max_batch_size(0);
        assert_eq!(registry.max_batch_size(), 1);
    }

    /// The response `model` field reports which backend answered, not the
    /// client's arbitrary `model` request field.
    #[tokio::test]
    async fn response_model_field_reports_the_real_backend_not_the_client_value() {
        let app = create_embeddings_router(16);
        let (status, json) = post(
            app,
            serde_json::json!({ "input": "hello", "model": "text-embedding-3-large" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json["model"].as_str().expect("model field"),
            "bonsai-embeddings-identity",
            "the client's claimed model name must not be echoed back verbatim"
        );
    }

    // ── Model-backed embedder seam ───────────────────────────────────────────

    /// A trivial deterministic test double standing in for a future
    /// `BonsaiModel`-backed embedder.
    struct FixedVectorEmbedder {
        dim: usize,
    }

    impl Embedder for FixedVectorEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, oxibonsai_rag::error::RagError> {
            // Deterministic function of the text length, just distinctive
            // enough to tell apart from Identity/TF-IDF output in a test.
            let mut v = vec![0.0f32; self.dim];
            v[0] = text.len() as f32;
            l2_normalize(&mut v);
            Ok(v)
        }

        fn embedding_dim(&self) -> usize {
            self.dim
        }
    }

    #[test]
    fn model_embedder_takes_priority_over_identity_and_tfidf() {
        let registry =
            EmbedderRegistry::new(8).with_model_embedder(Arc::new(FixedVectorEmbedder { dim: 4 }));
        // Even after fitting TF-IDF, the model backend must still win.
        registry.fit_tfidf(&["a document".to_string(), "another document".to_string()]);
        assert_eq!(registry.backend_name(), "model");
        assert_eq!(registry.embedding_dim(), 4);
        let out = registry.embed_texts(&["four".to_string()]);
        assert_eq!(out[0].len(), 4);
        assert!(
            out[0][0] > 0.0,
            "expected the model embedder's distinctive first component"
        );
    }

    #[tokio::test]
    async fn router_with_model_embedder_serves_the_model_backend() {
        let app = create_embeddings_router_with_model(8, Arc::new(FixedVectorEmbedder { dim: 4 }));
        let (status, json) = post(app, serde_json::json!({ "input": "hi" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json["model"].as_str().expect("model field"),
            "bonsai-embeddings-model"
        );
        assert_eq!(
            json["data"][0]["embedding"]
                .as_array()
                .expect("embedding array")
                .len(),
            4
        );
    }

    // ── D-1 / SV-02: create_embeddings_router_requiring_model ────────────────

    #[tokio::test]
    async fn router_requiring_model_refuses_with_501_when_none_is_installed() {
        let app = create_embeddings_router_requiring_model(8);
        let (status, json) = post(app, serde_json::json!({ "input": "hi" })).await;
        assert_eq!(
            status,
            StatusCode::NOT_IMPLEMENTED,
            "no model-backed embedder is installed, so this must refuse honestly rather than \
             answer with a byte-hash vector"
        );
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("model-backed"),
            "the error must explain why, not just carry a bare status: {json}"
        );
    }

    #[tokio::test]
    async fn router_requiring_model_is_independent_of_the_default_router() {
        // The default `create_embeddings_router` (used elsewhere in this
        // test suite, and by library embedders) must keep answering 200
        // with the stateless fallback -- this constructor is additive, not
        // a change to that one's documented behavior.
        let default_app = create_embeddings_router(8);
        let (status, _) = post(default_app, serde_json::json!({ "input": "hi" })).await;
        assert_eq!(status, StatusCode::OK);
    }

    // ── Dimension truncation renormalises (SV-02 correction (a)) ─────────────

    #[tokio::test]
    async fn dimensions_truncation_returns_a_unit_vector() {
        let app = create_embeddings_router(32);
        let (status, json) = post(
            app,
            serde_json::json!({ "input": "renormalisation test", "dimensions": 5 }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let vec: Vec<f32> = json["data"][0]["embedding"]
            .as_array()
            .expect("embedding array")
            .iter()
            .map(|v| v.as_f64().expect("f64") as f32)
            .collect();
        assert_eq!(vec.len(), 5);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "truncated embedding must be re-normalised to unit length, got norm {norm}"
        );
    }

    /// When `dimensions` is >= the natural size, the vector is returned
    /// unmodified (still unit-length, since every backend already normalises).
    #[tokio::test]
    async fn dimensions_at_or_above_natural_size_is_a_no_op() {
        let app = create_embeddings_router(8);
        let (status, json) = post(
            app,
            serde_json::json!({ "input": "no truncation needed", "dimensions": 999 }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json["data"][0]["embedding"]
                .as_array()
                .expect("embedding array")
                .len(),
            8
        );
    }

    /// `dimensions: 0` is rejected rather than silently truncating every
    /// embedding to an empty vector and returning `200`.
    #[tokio::test]
    async fn dimensions_zero_is_rejected_with_400() {
        let app = create_embeddings_router(16);
        let (status, json) = post(
            app,
            serde_json::json!({ "input": "hello", "dimensions": 0 }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["param"], "dimensions");
    }

    // ── dimension / normalized response fields (RT-08 fix item 1) ───────────

    #[tokio::test]
    async fn response_reports_the_natural_dimension() {
        let app = create_embeddings_router(16);
        let (status, json) = post(app, serde_json::json!({ "input": "hello world" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["dimension"].as_u64(), Some(16));
    }

    #[tokio::test]
    async fn response_dimension_field_reflects_truncation() {
        let app = create_embeddings_router(32);
        let (status, json) = post(
            app,
            serde_json::json!({ "input": "hello world", "dimensions": 5 }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["dimension"].as_u64(), Some(5));
    }

    #[tokio::test]
    async fn response_normalized_field_is_true_for_a_genuine_embedding() {
        let app = create_embeddings_router(16);
        let (status, json) = post(app, serde_json::json!({ "input": "hello world" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["normalized"].as_bool(), Some(true));
    }

    /// A fully out-of-vocabulary TF-IDF query (default, non-strict mode)
    /// falls back to an all-zero vector — the response must report
    /// `normalized: false` for it rather than unconditionally claiming
    /// `true` regardless of what actually happened (the exact defect class
    /// this fix's `dimension`/`normalized` fields exist to avoid
    /// reintroducing).
    #[tokio::test]
    async fn response_normalized_field_is_false_when_an_item_falls_back_to_zero_vector() {
        let registry = EmbedderRegistry::new(16);
        registry.fit_tfidf(&["alpha document".to_string(), "beta document".to_string()]);
        let state = Arc::new(EmbeddingAppState { registry });
        let app = Router::new()
            .route("/v1/embeddings", axum::routing::post(create_embeddings))
            .with_state(state);

        let (status, json) = post(app, serde_json::json!({ "input": "zzz" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            json["normalized"].as_bool(),
            Some(false),
            "an all-zero fallback vector must be reported as not normalised, got {json}"
        );
    }

    /// RT-EMBEDDINGS BLOCKING 2 / SV-02 follow-up: `normalized` must reflect
    /// the vector actually shipped, not the pre-truncation one. Reproduced
    /// with the real TF-IDF backend: fit on 12 documents that each pair a
    /// unique `termN` with the word every document shares (`shared`), so
    /// `shared` has the highest document frequency and sorts into vocabulary
    /// column 0, while every `termN` ties at document frequency 1 and is
    /// broken alphabetically -- putting `term5` at column 8, past a
    /// `dimensions: 2` truncation.
    ///
    /// Self-validating: the *untruncated* embedding is asserted non-zero
    /// first, proving `term5` really does embed to a genuine, non-degenerate
    /// vector, so the truncated all-zero result below is known to come from
    /// truncation discarding the signal rather than a broken fixture. Before
    /// the fix, `any_zero_vector` was computed on this same non-zero
    /// pre-truncation vector, so the truncated response below reported
    /// `normalized: true` for a shipped `[0.0, 0.0]`.
    #[tokio::test]
    async fn response_normalized_field_is_false_when_truncation_zeroes_every_component() {
        let corpus: Vec<String> = (0..12).map(|i| format!("term{i} shared")).collect();
        let registry = EmbedderRegistry::new(64);
        registry.fit_tfidf(&corpus);
        let state = Arc::new(EmbeddingAppState { registry });
        let app = Router::new()
            .route("/v1/embeddings", axum::routing::post(create_embeddings))
            .with_state(state);

        // No truncation: `term5` must embed to a genuine (non-degenerate)
        // unit vector -- the fixture's precondition.
        let (status, json) = post(app.clone(), serde_json::json!({ "input": "term5" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["normalized"].as_bool(), Some(true));
        let full = json["data"][0]["embedding"]
            .as_array()
            .expect("float embedding array")
            .clone();
        assert!(
            full.iter().any(|v| v.as_f64().unwrap_or(0.0) != 0.0),
            "fixture precondition: term5's untruncated embedding must have a \
             non-zero component, got {full:?}"
        );

        // Truncated to the first 2 columns: term5's only non-zero column
        // (8, "shared" occupies 0) is discarded, so the shipped vector is
        // exactly [0.0, 0.0].
        let (status, json) = post(
            app,
            serde_json::json!({ "input": "term5", "dimensions": 2 }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["dimension"].as_u64(), Some(2));
        let truncated = json["data"][0]["embedding"]
            .as_array()
            .expect("float embedding array");
        assert!(
            truncated.iter().all(|v| v.as_f64() == Some(0.0)),
            "fixture precondition: truncating to 2 columns must zero every \
             component, got {truncated:?}"
        );
        assert_eq!(
            json["normalized"].as_bool(),
            Some(false),
            "a vector that truncated to all-zero must not be reported as \
             normalized, got {json}"
        );
    }

    /// Sibling of the above: a truncation that *keeps* the surviving signal
    /// must still report `normalized: true` -- the fix moves the check to
    /// run after truncation, it does not make the field unconditionally
    /// `false` whenever `dimensions` is set. `shared` is the
    /// highest-document-frequency term in the same fixture and therefore
    /// sorts into vocabulary column 0, so a `dimensions: 2` truncation keeps
    /// it.
    #[tokio::test]
    async fn response_normalized_field_is_true_when_truncation_keeps_a_unit_vector() {
        let corpus: Vec<String> = (0..12).map(|i| format!("term{i} shared")).collect();
        let registry = EmbedderRegistry::new(64);
        registry.fit_tfidf(&corpus);
        let state = Arc::new(EmbeddingAppState { registry });
        let app = Router::new()
            .route("/v1/embeddings", axum::routing::post(create_embeddings))
            .with_state(state);

        let (status, json) = post(
            app,
            serde_json::json!({ "input": "shared", "dimensions": 2 }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["dimension"].as_u64(), Some(2));
        let embedding = json["data"][0]["embedding"]
            .as_array()
            .expect("float embedding array");
        let norm_sq: f64 = embedding
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0).powi(2))
            .sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-4,
            "fixture precondition: the truncated vector must still be a \
             (re-normalised) unit vector, got {json}"
        );
        assert_eq!(
            json["normalized"].as_bool(),
            Some(true),
            "a truncation that keeps a genuine unit vector must still report \
             normalized: true, got {json}"
        );
    }

    // ── encoding_format validation (SV-02) ───────────────────────────────────

    #[tokio::test]
    async fn unknown_encoding_format_is_rejected_with_400() {
        let app = create_embeddings_router(16);
        let (status, json) = post(
            app,
            serde_json::json!({ "input": "hello", "encoding_format": "binary" }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["param"], "encoding_format");
    }

    #[tokio::test]
    async fn encoding_format_float_is_accepted_explicitly() {
        let app = create_embeddings_router(16);
        let (status, json) = post(
            app,
            serde_json::json!({ "input": "hello", "encoding_format": "float" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(json["data"][0]["embedding"].is_array());
    }

    // ── Per-item input length cap (sec-19) ───────────────────────────────────

    #[tokio::test]
    async fn input_over_the_max_input_len_is_rejected_with_400() {
        let app = create_embeddings_router(16);
        let long_text = "a".repeat(DEFAULT_MAX_EMBEDDING_INPUT_CHARS + 1);
        let (status, json) = post(app, serde_json::json!({ "input": long_text })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["param"], "input");
    }

    #[tokio::test]
    async fn input_at_the_max_input_len_is_accepted() {
        let app = create_embeddings_router(16);
        let text = "a".repeat(DEFAULT_MAX_EMBEDDING_INPUT_CHARS);
        let (status, _json) = post(app, serde_json::json!({ "input": text })).await;
        assert_eq!(status, StatusCode::OK);
    }

    /// Only the offending item's index is named for a multi-input batch, so
    /// a caller can locate which entry needs shortening.
    #[tokio::test]
    async fn per_item_cap_names_the_offending_batch_index() {
        let app = create_embeddings_router(16);
        let long_text = "a".repeat(DEFAULT_MAX_EMBEDDING_INPUT_CHARS + 1);
        let (status, json) = post(app, serde_json::json!({ "input": ["short", long_text] })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let message = json["error"]["message"].as_str().expect("message");
        assert!(
            message.contains("input[1]"),
            "expected the message to name the offending index (1), got {message:?}"
        );
    }

    #[tokio::test]
    async fn configured_max_input_len_is_enforced() {
        let state = Arc::new(EmbeddingAppState {
            registry: EmbedderRegistry::new(4).with_max_input_len(5),
        });
        let app = Router::new()
            .route("/v1/embeddings", axum::routing::post(create_embeddings))
            .with_state(state);

        let (status_ok, _) = post(app.clone(), serde_json::json!({ "input": "hello" })).await;
        assert_eq!(status_ok, StatusCode::OK);

        let (status_over, json) = post(app, serde_json::json!({ "input": "hello world" })).await;
        assert_eq!(status_over, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["param"], "input");
    }

    #[test]
    fn max_input_len_zero_is_clamped_to_one() {
        let registry = EmbedderRegistry::new(8).with_max_input_len(0);
        assert_eq!(registry.max_input_len(), 1);
    }

    // ── require_model_backend gate (RT-08 / SV-02 correction (b)) ────────────

    #[tokio::test]
    async fn require_model_backend_returns_501_when_no_model_installed() {
        let state = Arc::new(EmbeddingAppState {
            registry: EmbedderRegistry::new(16).with_require_model_backend(true),
        });
        let app = Router::new()
            .route("/v1/embeddings", axum::routing::post(create_embeddings))
            .with_state(state);

        let (status, _json) = post(app, serde_json::json!({ "input": "hello" })).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    }

    #[tokio::test]
    async fn require_model_backend_still_serves_200_when_a_model_is_installed() {
        let state = Arc::new(EmbeddingAppState {
            registry: EmbedderRegistry::new(16)
                .with_require_model_backend(true)
                .with_model_embedder(Arc::new(FixedVectorEmbedder { dim: 4 })),
        });
        let app = Router::new()
            .route("/v1/embeddings", axum::routing::post(create_embeddings))
            .with_state(state);

        let (status, json) = post(app, serde_json::json!({ "input": "hello" })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["model"].as_str(), Some("bonsai-embeddings-model"));
    }

    #[test]
    fn model_backend_required_but_missing_reflects_the_gate() {
        let no_gate = EmbedderRegistry::new(8);
        assert!(!no_gate.model_backend_required_but_missing());

        let gated_no_model = EmbedderRegistry::new(8).with_require_model_backend(true);
        assert!(gated_no_model.model_backend_required_but_missing());

        let gated_with_model = EmbedderRegistry::new(8)
            .with_require_model_backend(true)
            .with_model_embedder(Arc::new(FixedVectorEmbedder { dim: 4 }));
        assert!(!gated_with_model.model_backend_required_but_missing());
    }

    /// The default router (the one `server.rs` actually calls, and the one
    /// the sibling `tests/embeddings_tests.rs` integration suite exercises
    /// directly) is unaffected by the gate: `with_require_model_backend`
    /// defaults to `false`, so it keeps serving the stateless fallback at
    /// `200` — see the module docs' "Backends and determinism" section for
    /// why the live `server.rs` call site does not opt into the gate today.
    #[tokio::test]
    async fn default_router_is_unaffected_by_the_require_model_backend_gate() {
        let app = create_embeddings_router(16);
        let (status, _json) = post(app, serde_json::json!({ "input": "hello" })).await;
        assert_eq!(status, StatusCode::OK);
    }
}
