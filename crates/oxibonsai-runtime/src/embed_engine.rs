//! Real, model-backed embeddings: the engine seam and the [`Embedder`]
//! implementation `/v1/embeddings` serves (`RT-08` / `SV-02`).
//!
//! Before this module the embeddings endpoint had exactly two backends, both
//! lexical: a TF-IDF bag of words and a byte-hash `IdentityEmbedder`. Neither
//! is a *semantic* embedding, and an OpenAI-SDK client had no way to tell.
//! This module computes a real one from the loaded model.
//!
//! Three layers, bottom up:
//!
//! 1. The model's hidden-state seam — mean-pooled, L2-normalised final hidden
//!    state, taken **before** the LM head:
//!    [`BonsaiModel::embed_mean_pooled`](oxibonsai_model::model::BonsaiModel::embed_mean_pooled)
//!    for a dense (`qwen3`) model, which runs the batched CPU prefill, and
//!    [`HybridModel::embed_mean_pooled`](oxibonsai_model::hybrid::HybridModel::embed_mean_pooled)
//!    for a hybrid (`qwen35`, Bonsai 2) one, which runs the hybrid's own
//!    chunked prefill driver with the LM head skipped. Both pool the same way.
//! 2. [`InferenceEngine::embed`] — the runtime seam: resets per-sequence state
//!    around the call and dispatches on the loaded model's kind.
//! 3. [`ModelEmbedder`] — an [`Embedder`] over an
//!    `Arc<Mutex<InferenceEngine>>` plus a [`TokenizerBridge`], which is what
//!    `crate::embeddings::EmbedderRegistry::with_model_embedder` takes.
//!
//! # Where the arithmetic runs
//!
//! A dense model's embedding runs on the **CPU** whatever tier its engine
//! decodes on: no head-free batched GPU prefill exists (every fused GPU
//! prefill folds in the LM head), and the batched CPU prefill is several
//! times faster than a per-token sweep on any tier. Only when that batched
//! pass declines a model (a sliding window, a quantization format without a
//! register-blocked GEMM) does the per-token fallback dispatch to the
//! engine's own tier. A hybrid model runs its CPU layers, as it does for
//! generation.
//!
//! # Why a `Mutex` over one dedicated engine, and not an `EnginePool` lease
//!
//! [`Embedder::embed`] is synchronous and takes `&self`, while
//! [`InferenceEngine::embed`] needs `&mut`; something has to provide interior
//! mutability. The obvious alternative — leasing a replica from
//! [`EnginePool`](crate::engine_pool::EnginePool) — does not fit: `acquire()`
//! is `async`, the embedding work already runs inside
//! `tokio::task::spawn_blocking`, and driving an async acquire from a blocking
//! thread deadlocks outright on a current-thread runtime (which is what
//! `#[tokio::test]` builds by default). So [`ModelEmbedder`] owns a dedicated
//! engine behind a `Mutex`: embedding requests **serialise** against each
//! other, and — more importantly — they never touch a replica that is serving
//! a chat completion, which matters because both models' hidden-state passes
//! rewrite the KV cache from position 0 (and a hybrid's also its recurrent
//! state).
//!
//! A GGUF-loaded replica shares its weights through the same memory map as
//! every other replica, so "a dedicated engine" costs a KV cache (plus, for a
//! hybrid, its recurrent state), not a second copy of the model.
//!
//! # Truncation: text refuses, raw token ids still truncate
//!
//! [`DEFAULT_MAX_EMBEDDING_TOKENS`] / [`ModelEmbedder::with_max_tokens`] set a
//! ceiling this module enforces by **truncating** — the conventional
//! behaviour for an embedding endpoint — with one layer above it refusing
//! instead:
//!
//! * A **text** input (`"input": "..."` / `["...", "..."]`) that tokenizes
//!   past the ceiling is refused one layer up, in
//!   `crate::embeddings::create_embeddings`, with `400
//!   context_length_exceeded` naming the real token count — *before* this
//!   module ever sees it — rather than silently truncated while
//!   `usage.prompt_tokens` bills the untruncated count.
//!   [`ModelEmbedder::embed_tokens`] itself still truncates — see its doc's
//!   "Truncation" section.
//! * A **raw token-id** input (`"input": [1, 2, 3]`,
//!   [`TokenSequenceEmbedder`]) deliberately keeps truncating rather than
//!   erroring: a caller supplying ids already knows exactly how many it
//!   sent. `crate::embeddings::EmbedderRegistry::count_prompt_tokens`'s
//!   caller charges `usage.prompt_tokens` for `min(ids.len(),
//!   max_input_tokens())` — what this module actually embeds — rather than
//!   the full supplied length.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use oxibonsai_model::error::ModelError;
use oxibonsai_rag::embedding::Embedder;
use oxibonsai_rag::error::RagError;

use crate::engine::InferenceEngine;
use crate::error::{RuntimeError, RuntimeResult};
use crate::tokenizer_bridge::TokenizerBridge;

/// Hard ceiling on how many tokens of one input a [`ModelEmbedder`] will feed
/// through the model, independent of the model's own context length.
///
/// Embedding is `O(n²)` in attention and every input is client-supplied, so a
/// single multi-megabyte string must not be able to buy an unbounded forward
/// pass (`sec-19`, the same defect class the HTTP layer's per-item character
/// cap addresses). Longer inputs are **truncated** — the conventional
/// behaviour for an embedding endpoint, and reported at `debug!` — rather than
/// rejected, so a caller embedding long documents still gets a usable vector.
/// Override with [`ModelEmbedder::with_max_tokens`].
pub const DEFAULT_MAX_EMBEDDING_TOKENS: usize = 2048;

/// Count the tokens an input really costs, for the `usage` block of an
/// embeddings response.
///
/// `crate::embeddings` otherwise approximates `prompt_tokens` as a
/// whitespace-split word count, which is wrong by a factor of two or more for
/// real text and wrong by an order of magnitude for CJK. A backend that owns a
/// tokenizer can answer exactly; one that does not returns `None` and the
/// caller keeps its approximation.
///
/// Separate from [`Embedder`] because that trait is `oxibonsai-rag`'s and has
/// no notion of tokens — a RAG embedder may be purely lexical.
pub trait EmbeddingTokenCounter: Send + Sync {
    /// Exact token count for `text`, or `None` when this backend cannot say.
    fn count_tokens(&self, text: &str) -> Option<usize>;

    /// The largest number of tokens this backend will actually embed before
    /// refusing rather than silently truncating.
    ///
    /// `crate::embeddings::create_embeddings` uses this to reject an
    /// over-length **text** input up front with `400
    /// context_length_exceeded`, naming the real token count, instead of
    /// silently truncating it and billing `usage.prompt_tokens` for more
    /// than the model actually saw. `None` — the default — means "no such
    /// ceiling is known", so no guard is applied and the backend's own
    /// behaviour applies unchanged.
    ///
    /// This governs the **text** input path only. The raw token-id path
    /// (`"input": [1, 2, 3]`, [`TokenSequenceEmbedder`]) deliberately keeps
    /// truncating rather than erroring — see that trait's docs and
    /// [`ModelEmbedder::embed_tokens`]'s "Truncation" section for why.
    fn max_input_tokens(&self) -> Option<usize> {
        None
    }

    /// The token ids `text` embeds as — the very ids [`Self::count_tokens`]
    /// counts — or `None` (the default) when this backend cannot hand them
    /// out.
    ///
    /// A backend that returns them **and** embeds ids directly
    /// ([`TokenSequenceEmbedder`], wired from the same object by
    /// [`EmbedderRegistry::with_model`](crate::embeddings::EmbedderRegistry::with_model))
    /// lets `crate::embeddings::create_embeddings` tokenize each text input
    /// exactly once per request: the ids answer the length guard and
    /// `usage.prompt_tokens`, then are embedded as ids — instead of
    /// tokenizing the same text again to count it, bill it and embed it.
    fn token_ids(&self, text: &str) -> Option<Vec<u32>> {
        let _ = text;
        None
    }
}

/// Embed a token-id sequence the client supplied directly.
///
/// An OpenAI embeddings request may carry `"input": [1, 2, 3]` — already
/// tokenised ids, not text. The [`Embedder`] trait has no way to express that,
/// so `crate::embeddings` used to render the ids back into the decimal
/// *string* `"1 2 3"` and embed that, which for a model backend is not a lossy
/// approximation but a different input entirely. A backend that can consume
/// ids directly implements this and the HTTP layer routes to it.
pub trait TokenSequenceEmbedder: Send + Sync {
    /// Embed `tokens` (already tokenised), returning an
    /// [`Embedder::embedding_dim`]-length vector.
    fn embed_token_ids(&self, tokens: &[u32]) -> Result<Vec<f32>, RagError>;

    /// Embed every id sequence in `batches`, one result per entry.
    ///
    /// The default loop calls [`embed_token_ids`](Self::embed_token_ids) once
    /// per sequence; [`ModelEmbedder`] overrides this to take its engine lock
    /// **once** for the whole batch instead of once per sequence — the same
    /// batched-lock treatment its [`Embedder::embed_batch`] override gives
    /// the text path, extended to `"input": [[1, 2], [3, 4]]` requests.
    fn embed_token_batches(&self, batches: &[Vec<u32>]) -> Vec<Result<Vec<f32>, RagError>> {
        batches
            .iter()
            .map(|ids| self.embed_token_ids(ids))
            .collect()
    }
}

/// Lock `mutex`, recovering from lock poisoning instead of propagating a panic.
///
/// Mirrors `crate::embeddings::lock_or_recover`'s policy: one unrelated panic
/// must not turn a live route into a process-lifetime outage. Recovery is safe
/// here even though the guarded value is a whole [`InferenceEngine`], whose KV
/// cache a panic mid-forward genuinely could leave half-written, because the
/// embedding path reads only per-sequence state that [`InferenceEngine::embed`]
/// clears *before* running the pass: the host KV cache, its coherence
/// watermark and the MET-05 device-KV latch of a dense model, the KV cursor of
/// a hybrid one, and the recurrent state of both. It reads nothing else —
/// notably, no step of either model's `forward_hidden` consults the engine's
/// [`CancellationToken`](crate::engine_control::CancellationToken), so a token
/// a panicking request left armed cannot leak into a later embedding (which is
/// what `EngineLease::drop` exists to prevent for pooled replicas). So a
/// recovered guard is as good as a clean one.
fn lock_engine_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!(
            "embedding engine mutex was poisoned by a prior panic; recovering the engine \
             instead of propagating the panic to this request"
        );
        poisoned.into_inner()
    })
}

/// The typed "you gave me no tokens" error, matching the model seam's own.
fn empty_input_error(who: &str) -> RuntimeError {
    RuntimeError::Model(ModelError::ShapeInvariant {
        tensor: who.to_string(),
        expected: "at least one token id".to_string(),
        actual: "0 token ids".to_string(),
    })
}

impl InferenceEngine<'_> {
    /// Embed an already-tokenised input: mean-pooled, L2-normalised final
    /// hidden state, `[hidden_size]` floats.
    ///
    /// Dispatches on the loaded model: a dense model runs
    /// [`BonsaiModel::embed_mean_pooled`](oxibonsai_model::model::BonsaiModel::embed_mean_pooled)
    /// — the batched CPU prefill, whatever tier this engine decodes on, with
    /// this engine's own dispatcher used only by the per-token fallback — and
    /// a hybrid (`qwen35`) model runs
    /// [`HybridModel::embed_mean_pooled`](oxibonsai_model::hybrid::HybridModel::embed_mean_pooled),
    /// whose rows are in the model's own (un-rotated) basis. Both pool and
    /// normalise identically.
    ///
    /// # This clobbers the engine's per-sequence state — and nothing else's
    ///
    /// Both hidden-state passes write KV positions `0..tokens.len()` (and a
    /// hybrid's advances its recurrent state), so this must not be
    /// interleaved with a generation in flight on the same engine. Each model
    /// clears its own per-sequence state on both sides of the pass; this
    /// method additionally clears the engine's attached
    /// [`RecurrentState`](crate::engine_control::RecurrentState), and ends the
    /// engine's current sequence, so a
    /// [`SequenceSnapshot`](crate::engine_seam::SequenceSnapshot) taken before
    /// the call can no longer be restored. [`ModelEmbedder`] additionally
    /// serialises callers behind a `Mutex` on a dedicated engine.
    ///
    /// **It deliberately does not call [`InferenceEngine::reset`].** For a
    /// dense model that would reach `BonsaiModel::reset`, which releases the
    /// **process-global** Metal device KV cache — wiping the state of a chat
    /// completion another engine may be decoding through the fused Metal path
    /// at that moment. Embedding never reads that cache, so it must not
    /// release it.
    ///
    /// # Errors
    ///
    /// * [`RuntimeError::Model`] wrapping [`ModelError::ShapeInvariant`] —
    ///   `text_tokens` is empty.
    /// * [`RuntimeError::Model`] wrapping [`ModelError::SequenceTooLong`] —
    ///   the input is longer than [`embedding_max_tokens`](Self::embedding_max_tokens).
    /// * Anything the forward pass returns (e.g. a token id past the
    ///   vocabulary).
    pub fn embed(&mut self, text_tokens: &[u32]) -> RuntimeResult<Vec<f32>> {
        if text_tokens.is_empty() {
            return Err(empty_input_error("InferenceEngine::embed"));
        }
        // The engine-level half of the reset: an attached `RecurrentState`
        // (RT-28) lives here, not on the model, so the models' own clears
        // cannot reach it. The model-level half happens inside each model's
        // `forward_hidden`, on both sides of the pass.
        self.reset_recurrent();
        // Disjoint field borrows: `&mut self.model` and `&self.kernel`, the
        // same shape `decode_step` already uses.
        let result = match &mut self.model {
            oxibonsai_model::hybrid::LoadedModel::Dense(model) => {
                model.embed_mean_pooled(text_tokens, &self.kernel)
            }
            oxibonsai_model::hybrid::LoadedModel::Hybrid(model) => {
                model.embed_mean_pooled(text_tokens)
            }
        };
        self.reset_recurrent();
        // Whatever sequence this engine held is gone (its KV positions were
        // overwritten and cleared), so no snapshot of it may be restored.
        self.sequence_id = self.sequence_id.wrapping_add(1);
        Ok(result?)
    }

    /// Dimensionality [`embed`](Self::embed) produces: the model's hidden size.
    pub fn embedding_dim(&self) -> usize {
        self.hidden_size()
    }

    /// Highest number of tokens [`embed`](Self::embed) can accept before the
    /// model refuses with [`ModelError::SequenceTooLong`]: a dense model's
    /// effective context, a hybrid model's KV window
    /// ([`HybridModel::max_seq_len`](oxibonsai_model::hybrid::HybridModel::max_seq_len)).
    pub fn embedding_max_tokens(&self) -> usize {
        self.max_context()
    }
}

/// An [`Embedder`] backed by a real loaded model.
///
/// Text is tokenised with `tokenizer`, truncated to
/// [`max_tokens`](Self::max_tokens), and run through
/// [`InferenceEngine::embed`]. The result is the mean-pooled, L2-normalised
/// final hidden state — a genuine semantic embedding, deterministic for a
/// given (model, text) pair and independent of every other request, which is
/// exactly what the TF-IDF backend `SV-02` condemned was not.
///
/// Install it with
/// `crate::embeddings::EmbedderRegistry::with_model_embedder(Arc::new(embedder))`.
pub struct ModelEmbedder {
    engine: Arc<Mutex<InferenceEngine<'static>>>,
    tokenizer: Arc<TokenizerBridge>,
    dim: usize,
    max_tokens: usize,
    /// Number of times this embedder has acquired `engine`'s lock. Test
    /// instrumentation only (the "one lock per batch" tests need an
    /// observable counter) — see [`lock_acquisitions`](Self::lock_acquisitions).
    lock_acquisitions: AtomicU64,
}

impl std::fmt::Debug for ModelEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `InferenceEngine` is not `Debug` and locking it here could block a
        // request just to format a log line, so report only the shape.
        f.debug_struct("ModelEmbedder")
            .field("dim", &self.dim)
            .field("max_tokens", &self.max_tokens)
            .field("tokenizer", &self.tokenizer.backend())
            .field("lock_acquisitions", &self.lock_acquisitions())
            .finish()
    }
}

impl ModelEmbedder {
    /// Build an embedder over a dedicated engine and a tokenizer.
    ///
    /// The dimension and the token ceiling are read from the engine once, here
    /// — `embedding_dim()` is on the hot path of every response and must not
    /// take the engine lock.
    ///
    /// `engine` should be an engine **nothing else generates on**: see the
    /// module docs.
    pub fn new(
        engine: Arc<Mutex<InferenceEngine<'static>>>,
        tokenizer: Arc<TokenizerBridge>,
    ) -> Self {
        let (dim, context) = {
            let guard = lock_engine_or_recover(&engine);
            (guard.embedding_dim(), guard.embedding_max_tokens())
        };
        Self {
            engine,
            tokenizer,
            dim,
            max_tokens: DEFAULT_MAX_EMBEDDING_TOKENS.min(context.max(1)),
            lock_acquisitions: AtomicU64::new(0),
        }
    }

    /// Load a dedicated embedding engine from a GGUF on disk and wrap it.
    ///
    /// The one-call constructor a server binary wants: it memory-maps and
    /// parses the GGUF (leaking both to `'static`, exactly as
    /// [`crate::engine_pool::build_pool_from_gguf`] does for its own replicas)
    /// and auto-detects the kernel tier.
    ///
    /// # Memory
    ///
    /// The extra `mmap` costs no extra resident weight memory: the OS shares
    /// the page cache with the pool's mapping of the same file. What this
    /// engine really adds is its own KV cache, which is why `max_seq_len`
    /// should be the *embedding* context (a few hundred tokens), not the chat
    /// context — see [`DEFAULT_MAX_EMBEDDING_TOKENS`].
    ///
    /// # Errors
    ///
    /// Anything [`InferenceEngine::from_gguf_path_leaked`] returns — a missing
    /// file, an unparseable header, an unsupported architecture.
    pub fn from_gguf_path(
        path: impl AsRef<std::path::Path>,
        tokenizer: Arc<TokenizerBridge>,
        sampling_params: crate::sampling::SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Arc<Self>> {
        let (engine, _gguf) =
            InferenceEngine::from_gguf_path_leaked(path, sampling_params, seed, max_seq_len)?;
        Self::from_engine(engine, tokenizer)
    }

    /// Wrap an already-built engine — dense or hybrid (`qwen35`); both
    /// kinds embed (see [`InferenceEngine::embed`]).
    ///
    /// # Errors
    ///
    /// None today; the `Result` keeps the constructor family uniform with
    /// [`from_gguf_path`](Self::from_gguf_path) and
    /// [`from_static_gguf`](Self::from_static_gguf), which can fail.
    pub fn from_engine(
        engine: InferenceEngine<'static>,
        tokenizer: Arc<TokenizerBridge>,
    ) -> RuntimeResult<Arc<Self>> {
        Ok(Arc::new(Self::new(Arc::new(Mutex::new(engine)), tokenizer)))
    }

    /// Wrap an engine built from an already-leaked, already-parsed GGUF,
    /// sharing the pool's `token_embd` table.
    ///
    /// The zero-extra-memory variant of [`from_gguf_path`](Self::from_gguf_path)
    /// for a caller that already holds the `&'static GgufFile` and the shared
    /// dequantized embedding handle (what
    /// [`crate::engine_pool::build_pool_from_gguf`] builds internally): the
    /// embedding engine then adds only its own KV cache, with no second mapping
    /// and no second copy of the `vocab × hidden` table. For a hybrid
    /// (`qwen35`) file the pool's handle is the empty "decode rows from the
    /// GGUF" one, and the engine adds its KV cache, recurrent state and
    /// activation scratch. A hybrid's embedding pass runs on its CPU model,
    /// so its embedding engine is pinned to the CPU backend: a Metal hybrid
    /// runner there would allocate its whole-window KV cache, recurrent
    /// state and gates and never run.
    ///
    /// # Errors
    ///
    /// Anything [`InferenceEngine::from_gguf_with_embd_and_backend`] returns
    /// — including a non-empty dense `token_embd` for a hybrid file, whose
    /// embedding is decoded row-wise in the rotated basis
    /// (`SHARED_EMBEDDING_UNSUPPORTED`).
    pub fn from_static_gguf(
        gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static>,
        token_embd: Arc<[f32]>,
        tokenizer: Arc<TokenizerBridge>,
        sampling_params: crate::sampling::SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Arc<Self>> {
        let backend = if oxibonsai_model::hybrid::LoadedModel::is_hybrid_gguf(gguf) {
            crate::engine_seam::Backend::Cpu
        } else {
            crate::engine_seam::Backend::Auto
        };
        let engine = InferenceEngine::from_gguf_with_embd_and_backend(
            gguf,
            sampling_params,
            seed,
            max_seq_len,
            token_embd,
            backend,
        )?;
        Self::from_engine(engine, tokenizer)
    }

    /// Override the per-input token ceiling (builder).
    ///
    /// Clamped to at least `1` — a `0` would truncate every input to nothing
    /// and turn the endpoint into a zero-vector generator — and to at most the
    /// model's effective context, past which the forward pass would refuse
    /// anyway.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        let context = {
            let guard = lock_engine_or_recover(&self.engine);
            guard.embedding_max_tokens()
        };
        self.max_tokens = max_tokens.clamp(1, context.max(1));
        self
    }

    /// Per-input token ceiling; longer inputs are truncated to this length.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Number of times this embedder has acquired its engine lock so far.
    ///
    /// Test instrumentation, not a production metric: it exists so a test
    /// can assert that an N-input batch — through
    /// [`embed_owned_batch`](Self::embed_owned_batch), the
    /// [`Embedder::embed_batch`] override,
    /// [`TokenSequenceEmbedder::embed_token_batches`], or
    /// [`crate::embeddings::EmbedderRegistry::embed_texts`] — takes this lock
    /// **once**, not once per item. Monotonically increasing, `Relaxed`
    /// ordering: callers compare a before/after delta, never an absolute
    /// value, so no stronger ordering is needed.
    pub fn lock_acquisitions(&self) -> u64 {
        self.lock_acquisitions.load(Ordering::Relaxed)
    }

    /// Dimensionality of every vector this embedder produces.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Tokenise `text` the way [`embed`](Embedder::embed) will, without
    /// truncation.
    ///
    /// This is the honest `prompt_tokens` for an embeddings response (see
    /// [`EmbeddingTokenCounter`]).
    pub fn tokenize(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        self.tokenizer.encode(text)
    }

    /// Embed an already-tokenised input, bypassing the tokenizer.
    ///
    /// The seam for `"input": [1, 2, 3]` — an OpenAI embeddings request may
    /// carry token ids directly, and rendering them back into decimal text
    /// just to re-tokenise them would be lossy nonsense. Truncates to
    /// [`max_tokens`](Self::max_tokens).
    ///
    /// # Truncation
    ///
    /// This method — reached from both [`Embedder::embed`]'s tokenized text
    /// and [`TokenSequenceEmbedder::embed_token_ids`]'s raw ids — always
    /// truncates silently rather than erroring; `crate::embeddings`'
    /// `context_length_exceeded` guard is a layer **above** this one and
    /// only ever applies to the **text** path (it checks the tokenized
    /// length before this method is ever called, so a text input that would
    /// need truncating here never reaches it). The raw token-id path keeps
    /// this method's truncate-not-reject behaviour deliberately: a caller
    /// supplying ids already knows their exact count, so refusing outright
    /// would be a worse experience than the conventional embedding-endpoint
    /// truncation this ceiling exists to apply.
    ///
    /// # Errors
    ///
    /// Empty input, or anything [`InferenceEngine::embed`] returns.
    pub fn embed_tokens(&self, tokens: &[u32]) -> RuntimeResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(empty_input_error("ModelEmbedder::embed_tokens"));
        }
        let used = if tokens.len() > self.max_tokens {
            tracing::debug!(
                supplied = tokens.len(),
                max_tokens = self.max_tokens,
                "embedding input truncated to the configured token ceiling"
            );
            &tokens[..self.max_tokens]
        } else {
            tokens
        };
        // A poisoned lock needs no extra recovery here — see
        // `lock_engine_or_recover`. Calling `reset()` on the recovered engine
        // would be actively wrong: it additionally releases the process-global
        // Metal device KV cache, which is exactly what an embedding request
        // must never do (see `InferenceEngine::embed`).
        self.lock_acquisitions.fetch_add(1, Ordering::Relaxed);
        let mut guard = lock_engine_or_recover(&self.engine);
        guard.embed(used)
    }

    /// Embed every owned text in `texts`, one per returned entry.
    ///
    /// A plain loop over the texts (each one is its own sequence, and each
    /// `embed` already batches that sequence's positions), but it takes the
    /// engine lock **once** for the whole batch instead of once per item.
    /// A per-item failure is reported in that item's slot; it never aborts the
    /// batch, matching how the HTTP layer already degrades a failed item to a
    /// zero vector.
    ///
    /// # This is a latency trade, not a free win
    ///
    /// Holding the lock across the whole batch makes a large batch a single
    /// long critical section: a concurrent one-input request waits behind all
    /// of it rather than interleaving after the first item. That is the right
    /// trade **here** because the alternative is not fairness but thrash — the
    /// per-item path would reacquire the lock N times while the one engine
    /// stays the bottleneck either way, and every `embed` resets the KV cache,
    /// so interleaving buys a waiting request nothing but a later start. The
    /// HTTP layer's `max_batch_size` is what actually bounds the wait; a
    /// deployment that wants tighter tail latency lowers it rather than
    /// splitting this lock.
    ///
    /// The same computation as the [`Embedder::embed_batch`] override (which
    /// takes `&[&str]`), for callers holding owned strings. Named apart from
    /// the trait method so it never shadows it on the concrete type:
    /// `.embed_batch(..)` on a `ModelEmbedder` resolves to the trait's
    /// `&[&str]` method wherever [`Embedder`] is in scope.
    pub fn embed_owned_batch(&self, texts: &[String]) -> Vec<Result<Vec<f32>, RagError>> {
        self.embed_texts_locked(texts)
    }

    /// The one batched text path behind [`Self::embed_owned_batch`] and the
    /// [`Embedder::embed_batch`] override: every text tokenized, truncated to
    /// [`Self::max_tokens`] and embedded under one engine-lock acquisition,
    /// one result per input in order.
    fn embed_texts_locked<S: AsRef<str>>(&self, texts: &[S]) -> Vec<Result<Vec<f32>, RagError>> {
        if texts.is_empty() {
            return Vec::new();
        }
        // See `embed_tokens` on why a poisoned lock needs no extra recovery.
        self.lock_acquisitions.fetch_add(1, Ordering::Relaxed);
        let mut guard = lock_engine_or_recover(&self.engine);
        texts
            .iter()
            .map(|text| {
                let tokens = self
                    .tokenizer
                    .encode(text.as_ref())
                    .map_err(|e| RagError::EmbeddingFailed(e.to_string()))?;
                if tokens.is_empty() {
                    return Err(RagError::EmptyDocument);
                }
                let used = if tokens.len() > self.max_tokens {
                    &tokens[..self.max_tokens]
                } else {
                    &tokens[..]
                };
                guard
                    .embed(used)
                    .map_err(|e| RagError::EmbeddingFailed(e.to_string()))
            })
            .collect()
    }

    /// Embed every token-id sequence in `batches`, one result per entry,
    /// taking the engine lock **once** for the whole set
    /// (`TokenSequenceEmbedder::embed_token_batches`'s override for
    /// `ModelEmbedder` — the text path's batched-lock treatment extended to
    /// `"input": [[1, 2], [3, 4]]` requests). A per-sequence failure is
    /// reported in that sequence's own slot.
    pub fn embed_token_id_batches(&self, batches: &[Vec<u32>]) -> Vec<Result<Vec<f32>, RagError>> {
        if batches.is_empty() {
            return Vec::new();
        }
        self.lock_acquisitions.fetch_add(1, Ordering::Relaxed);
        let mut guard = lock_engine_or_recover(&self.engine);
        batches
            .iter()
            .map(|ids| {
                if ids.is_empty() {
                    return Err(RagError::EmptyDocument);
                }
                let used = if ids.len() > self.max_tokens {
                    &ids[..self.max_tokens]
                } else {
                    &ids[..]
                };
                guard
                    .embed(used)
                    .map_err(|e| RagError::EmbeddingFailed(e.to_string()))
            })
            .collect()
    }
}

impl Embedder for ModelEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
        let tokens = self
            .tokenizer
            .encode(text)
            .map_err(|e| RagError::EmbeddingFailed(e.to_string()))?;
        if tokens.is_empty() {
            // An input that tokenises to nothing (the empty string, or pure
            // whitespace under some vocabularies) has no hidden state to pool.
            // Reported rather than silently answered with a zero vector.
            return Err(RagError::EmptyDocument);
        }
        self.embed_tokens(&tokens)
            .map_err(|e| RagError::EmbeddingFailed(e.to_string()))
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }

    /// One engine-lock acquisition for the whole batch instead of the
    /// default per-item loop's one per text — what
    /// [`crate::embeddings::EmbedderRegistry::embed_texts`] reaches through
    /// its `dyn Embedder`. A per-item failure stays in its own slot.
    fn embed_batch(&self, texts: &[&str]) -> Vec<Result<Vec<f32>, RagError>> {
        self.embed_texts_locked(texts)
    }
}

impl EmbeddingTokenCounter for ModelEmbedder {
    fn count_tokens(&self, text: &str) -> Option<usize> {
        self.tokenizer.encode(text).ok().map(|ids| ids.len())
    }

    fn max_input_tokens(&self) -> Option<usize> {
        Some(self.max_tokens)
    }

    /// The ids [`Embedder::embed`] would embed for `text` (same tokenizer,
    /// same call), so embedding them through
    /// [`TokenSequenceEmbedder::embed_token_batches`] gives the text path's
    /// exact vectors.
    fn token_ids(&self, text: &str) -> Option<Vec<u32>> {
        self.tokenizer.encode(text).ok()
    }
}

impl TokenSequenceEmbedder for ModelEmbedder {
    fn embed_token_ids(&self, tokens: &[u32]) -> Result<Vec<f32>, RagError> {
        if tokens.is_empty() {
            return Err(RagError::EmptyDocument);
        }
        self.embed_tokens(tokens)
            .map_err(|e| RagError::EmbeddingFailed(e.to_string()))
    }

    fn embed_token_batches(&self, batches: &[Vec<u32>]) -> Vec<Result<Vec<f32>, RagError>> {
        self.embed_token_id_batches(batches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_kernels::dispatch::KernelTier;
    use oxibonsai_tokenizer::OxiTokenizer;

    use crate::sampling::SamplingParams;

    fn greedy_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 8,
        }
    }

    fn weightless_engine() -> InferenceEngine<'static> {
        InferenceEngine::from_model_with_tier(
            oxibonsai_model::model::BonsaiModel::new(Qwen3Config::tiny_test()),
            KernelTier::Reference,
            greedy_params(),
            42,
        )
    }

    fn char_tokenizer() -> Arc<TokenizerBridge> {
        Arc::new(TokenizerBridge::from_native_tokenizer(
            OxiTokenizer::char_level_stub(256),
        ))
    }

    #[test]
    fn engine_embed_rejects_empty_token_input() {
        let mut engine = weightless_engine();
        let err = engine
            .embed(&[])
            .expect_err("an empty token slice must be refused");
        assert!(
            matches!(err, RuntimeError::Model(ModelError::ShapeInvariant { .. })),
            "empty input must surface as a typed model error, got {err:?}"
        );
    }

    #[test]
    fn engine_embed_returns_hidden_size_floats() {
        let mut engine = weightless_engine();
        let dim = engine.embedding_dim();
        let vector = engine.embed(&[1, 2, 3]).expect("embed should run");
        assert_eq!(vector.len(), dim);
    }

    #[test]
    fn model_embedder_reports_the_engines_hidden_size() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        assert_eq!(embedder.dimension(), Qwen3Config::tiny_test().hidden_size);
        assert_eq!(embedder.embedding_dim(), embedder.dimension());
    }

    #[test]
    fn model_embedder_refuses_input_that_tokenizes_to_nothing() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let err = embedder
            .embed("")
            .expect_err("the empty string has no hidden state to pool");
        assert!(matches!(err, RagError::EmptyDocument), "got {err:?}");
    }

    #[test]
    fn model_embedder_counts_tokens_with_the_real_tokenizer() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let counted = embedder
            .count_tokens("abcd")
            .expect("the char-level stub can always count");
        assert_eq!(
            counted, 4,
            "a char-level tokenizer must charge one token per character, not one per word"
        );
    }

    #[test]
    fn model_embedder_truncates_to_the_token_ceiling() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer())
                .with_max_tokens(2);
        assert_eq!(embedder.max_tokens(), 2);
        // 6 ids, ceiling 2 — must not raise SequenceTooLong, must not panic.
        let vector = embedder
            .embed_tokens(&[1, 2, 3, 4, 5, 6])
            .expect("an over-long input must be truncated, not refused");
        assert_eq!(vector.len(), embedder.dimension());
    }

    #[test]
    fn with_max_tokens_clamps_zero_to_one() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer())
                .with_max_tokens(0);
        assert_eq!(embedder.max_tokens(), 1);
    }

    #[test]
    fn embed_batch_returns_one_result_per_input() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let results =
            embedder.embed_owned_batch(&["ab".to_string(), String::new(), "cd".to_string()]);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(
            matches!(results[1], Err(RagError::EmptyDocument)),
            "an untokenisable item must fail in its own slot, not abort the batch"
        );
        assert!(results[2].is_ok());
    }

    #[test]
    fn embed_batch_of_nothing_is_empty() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        assert!(embedder.embed_owned_batch(&[]).is_empty());
    }

    /// The owned-string batch no longer shadows the trait method: on the
    /// concrete type, `.embed_batch(&[&str])` is the `Embedder` override and
    /// `.embed_owned_batch(&[String])` the inherent spelling, and the two
    /// agree slot for slot.
    #[test]
    fn embed_batch_on_the_concrete_type_is_the_trait_method() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let via_trait = embedder.embed_batch(&["ab", "", "cd"]);
        let via_owned =
            embedder.embed_owned_batch(&["ab".to_string(), String::new(), "cd".to_string()]);
        assert_eq!(via_trait.len(), 3);
        for (a, b) in via_trait.iter().zip(&via_owned) {
            assert_eq!(a.as_ref().ok(), b.as_ref().ok());
        }
        assert!(matches!(via_trait[1], Err(RagError::EmptyDocument)));
    }

    // ── The batched engine lock ────────────────────────────────────────────

    #[test]
    fn embed_tokens_takes_the_engine_lock_once_per_call() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let before = embedder.lock_acquisitions();
        embedder.embed_tokens(&[1, 2, 3]).expect("embed");
        embedder.embed_tokens(&[4, 5]).expect("embed");
        assert_eq!(
            embedder.lock_acquisitions() - before,
            2,
            "two individual embed_tokens calls must take the lock exactly twice"
        );
    }

    #[test]
    fn embed_batch_takes_the_engine_lock_once_for_the_whole_batch() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let before = embedder.lock_acquisitions();
        let texts = vec![
            "a".to_string(),
            "bb".to_string(),
            "ccc".to_string(),
            "dddd".to_string(),
        ];
        let results = embedder.embed_owned_batch(&texts);
        assert_eq!(results.len(), 4);
        assert_eq!(
            embedder.lock_acquisitions() - before,
            1,
            "a 4-input embed_owned_batch call must take the engine lock exactly once, not once \
             per item"
        );
    }

    /// The batched text path lives on the rag `Embedder` trait itself;
    /// `ModelEmbedder`'s override is reached through a `dyn Embedder` and
    /// still takes the engine lock once, with the same per-slot results as
    /// the inherent `&[String]` spelling.
    #[test]
    fn the_embedder_trait_batch_takes_the_engine_lock_once_for_the_whole_batch() {
        let embedder = Arc::new(ModelEmbedder::new(
            Arc::new(Mutex::new(weightless_engine())),
            char_tokenizer(),
        ));
        let as_dyn: Arc<dyn Embedder> = Arc::clone(&embedder) as _;
        let before = embedder.lock_acquisitions();
        let results = as_dyn.embed_batch(&["a", "bb", "", "ccc"]);
        assert_eq!(
            embedder.lock_acquisitions() - before,
            1,
            "a 4-input dyn Embedder::embed_batch must take the engine lock exactly once"
        );
        assert_eq!(results.len(), 4);
        assert!(matches!(results[2], Err(RagError::EmptyDocument)));
        let owned: Vec<String> = ["a", "bb", "", "ccc"].map(str::to_string).to_vec();
        for (via_trait, via_inherent) in results
            .iter()
            .zip(embedder.embed_owned_batch(&owned).iter())
        {
            assert_eq!(via_trait.as_ref().ok(), via_inherent.as_ref().ok());
        }
    }

    /// `token_ids` hands out exactly the ids `embed` embeds, so the
    /// single-tokenization HTTP path embeds the same vector.
    #[test]
    fn token_ids_are_the_ids_embed_uses() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let ids = embedder.token_ids("abc").expect("tokenizes");
        assert_eq!(Some(ids.clone()), embedder.tokenize("abc").ok());
        assert_eq!(Some(ids.len()), embedder.count_tokens("abc"));
        assert_eq!(
            embedder.embed_tokens(&ids).ok(),
            embedder.embed("abc").ok(),
            "embedding the handed-out ids must equal embedding the text"
        );
    }

    #[test]
    fn embed_batch_of_nothing_does_not_take_the_lock() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let before = embedder.lock_acquisitions();
        assert!(embedder.embed_owned_batch(&[]).is_empty());
        assert_eq!(embedder.lock_acquisitions(), before);
    }

    #[test]
    fn embed_token_id_batches_takes_the_engine_lock_once_for_the_whole_batch() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let before = embedder.lock_acquisitions();
        let batches = vec![vec![1u32, 2, 3], vec![4u32], vec![5u32, 6]];
        let results = embedder.embed_token_id_batches(&batches);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(Result::is_ok));
        assert_eq!(
            embedder.lock_acquisitions() - before,
            1,
            "a 3-batch embed_token_id_batches call must take the engine lock exactly once"
        );
    }

    #[test]
    fn embed_token_id_batches_reports_an_empty_sequence_in_its_own_slot() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let batches = vec![vec![1u32, 2], Vec::new(), vec![3u32]];
        let results = embedder.embed_token_id_batches(&batches);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(matches!(results[1], Err(RagError::EmptyDocument)));
        assert!(results[2].is_ok());
    }

    #[test]
    fn token_sequence_embedder_trait_object_routes_through_the_batched_override() {
        let embedder = Arc::new(ModelEmbedder::new(
            Arc::new(Mutex::new(weightless_engine())),
            char_tokenizer(),
        ));
        let before = embedder.lock_acquisitions();
        let as_trait_object: Arc<dyn TokenSequenceEmbedder> = Arc::clone(&embedder) as _;
        let batches = vec![vec![1u32, 2], vec![3u32, 4], vec![5u32, 6]];
        let results = as_trait_object.embed_token_batches(&batches);
        assert_eq!(results.len(), 3);
        assert_eq!(
            embedder.lock_acquisitions() - before,
            1,
            "the trait-object path must reach ModelEmbedder's batched override, not the \
             default per-item loop"
        );
    }

    #[test]
    fn max_input_tokens_matches_the_configured_ceiling() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer())
                .with_max_tokens(17);
        let counter: &dyn EmbeddingTokenCounter = &embedder;
        assert_eq!(counter.max_input_tokens(), Some(17));
    }

    // ── Hybrid (qwen35) engines embed ──────────────────────────────────────

    /// `InferenceEngine::embed` on the real Bonsai 2 27B returns a finite,
    /// unit-length, `hidden_size`-wide, input-dependent and deterministic
    /// vector, and `ModelEmbedder` wraps the same engine and serves it.
    ///
    /// The GGUF is located **only** through `OXI_BONSAI2_PTQ1_GGUF` or
    /// `OXI_BONSAI2_PQ2_GGUF` (checked in that order) — never through
    /// `OXIBONSAI_MODELS_DIR` or a repo-relative `models/` lookup, so a
    /// workspace test run with the models directory configured never maps a
    /// 6-7 GB model by accident, and never through `OXI_MODEL`, which in this
    /// repo names a dense model. Unset, it self-skips and records the skip.
    #[test]
    fn real_27b_hybrid_embed_returns_a_unit_vector() {
        use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

        const TEST_NAME: &str =
            "oxibonsai-runtime::lib::real_27b_hybrid_embed_returns_a_unit_vector";
        const MAX_SEQ: usize = 64;
        let path = std::env::var_os("OXI_BONSAI2_PTQ1_GGUF")
            .or_else(|| std::env::var_os("OXI_BONSAI2_PQ2_GGUF"))
            .filter(|p| !p.is_empty())
            .map(std::path::PathBuf::from);
        let Some(path) = path else {
            eprintln!(
                "{TEST_NAME}: neither OXI_BONSAI2_PTQ1_GGUF nor OXI_BONSAI2_PQ2_GGUF is set -- \
                 skipping"
            );
            record_skipped(Capability::Bonsai2Models, TEST_NAME);
            return;
        };
        let gate_start = std::time::Instant::now();
        let (mut engine, _gguf) =
            InferenceEngine::from_gguf_path_leaked(&path, greedy_params(), 42, MAX_SEQ)
                .unwrap_or_else(|e| {
                    panic!("the Bonsai 2 27B GGUF at {} must load: {e}", path.display())
                });
        assert!(
            engine.is_hybrid(),
            "a qwen35 GGUF must load as a hybrid engine"
        );
        let dim = engine.embedding_dim();
        assert_eq!(dim, engine.hidden_size());
        assert_eq!(engine.embedding_max_tokens(), MAX_SEQ);

        // Five in-vocabulary ids (a special token of the 248 320-entry
        // vocabulary's reserved range, then four ordinary ones): the
        // assertions are about the vector, not about any particular text.
        let tokens: [u32; 5] = [248_045, 846, 198, 9_707, 13];
        let first = engine.embed(&tokens).expect("the hybrid engine embeds");
        assert_eq!(first.len(), dim);
        assert!(first.iter().all(|v| v.is_finite()), "finite components");
        let norm = first.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("{TEST_NAME}: dim {dim}, norm {norm:.6}");
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "a real-model embedding must be unit-length; got {norm}"
        );
        let again = engine.embed(&tokens).expect("second embed");
        assert_eq!(first, again, "the same ids must embed identically");
        let other = engine.embed(&tokens[..3]).expect("a different input");
        assert_ne!(first, other, "a different input must embed differently");
        assert_eq!(engine.sequence_position(), 0, "embedding leaves no state");

        // The embedder the server installs wraps the same engine and serves
        // the same vector.
        let embedder = ModelEmbedder::from_engine(engine, char_tokenizer())
            .expect("a hybrid engine builds an embedder");
        assert_eq!(embedder.dimension(), dim);
        let served = embedder.embed_tokens(&tokens).expect("served embedding");
        assert_eq!(served, first);
        record_executed_timed(Capability::Bonsai2Models, TEST_NAME, gate_start.elapsed());
    }
}
