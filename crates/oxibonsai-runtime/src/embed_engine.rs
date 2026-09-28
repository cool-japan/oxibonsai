//! Real, model-backed embeddings: the engine seam and the [`Embedder`]
//! implementation `/v1/embeddings` serves (`RT-08` / `SV-02`, orchestrator
//! decision D-1).
//!
//! Before this module the embeddings endpoint had exactly two backends, both
//! lexical: a TF-IDF bag of words and a byte-hash `IdentityEmbedder`. Neither
//! is a *semantic* embedding, and an OpenAI-SDK client had no way to tell.
//! D-1 decided the product ships a real one; this module is the half that
//! computes it.
//!
//! Three layers, bottom up:
//!
//! 1. [`BonsaiModel::embed_mean_pooled`](oxibonsai_model::model::BonsaiModel::embed_mean_pooled)
//!    (in `oxibonsai-model`) — mean-pooled, L2-normalised final hidden state,
//!    taken **before** the LM head.
//! 2. [`InferenceEngine::embed`] — the runtime seam: resets per-sequence state
//!    around the call and hands the engine's own
//!    [`KernelDispatcher`](oxibonsai_kernels::KernelDispatcher) to the model,
//!    so an embedding runs on exactly the tier that engine decodes on.
//! 3. [`ModelEmbedder`] — an [`Embedder`] over an
//!    `Arc<Mutex<InferenceEngine>>` plus a [`TokenizerBridge`], which is what
//!    `crate::embeddings::EmbedderRegistry::with_model_embedder` takes.
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
//! a chat completion, which matters because
//! [`forward_hidden`](oxibonsai_model::model::BonsaiModel::forward_hidden)
//! rewrites the host KV cache from position 0.
//!
//! A GGUF-loaded replica shares its weights through the same memory map as
//! every other replica, so "a dedicated engine" costs a KV cache, not a second
//! copy of the model.
//!
//! # Truncation: text refuses, raw token ids still truncate (`EMBED-WIRE`)
//!
//! [`DEFAULT_MAX_EMBEDDING_TOKENS`] / [`ModelEmbedder::with_max_tokens`] set a
//! ceiling this module has always enforced by **truncating** — the
//! conventional behaviour for an embedding endpoint. What changed:
//!
//! * A **text** input (`"input": "..."` / `["...", "..."]`) that tokenizes
//!   past the ceiling is now refused one layer up, in
//!   `crate::embeddings::create_embeddings`, with `400
//!   context_length_exceeded` naming the real token count — *before* this
//!   module ever sees it — rather than silently truncated and billed in
//!   full (the defect the wave-4 verifier's minor[3] named:
//!   `usage.prompt_tokens` reported the untruncated count while the model
//!   only saw the truncated one). [`ModelEmbedder::embed_tokens`] itself is
//!   unchanged by this — see its doc's "Truncation" section.
//! * A **raw token-id** input (`"input": [1, 2, 3]`,
//!   [`TokenSequenceEmbedder`]) deliberately keeps truncating rather than
//!   erroring: a caller supplying ids already knows exactly how many it
//!   sent. `crate::embeddings::EmbedderRegistry::count_prompt_tokens`'s
//!   caller now charges `usage.prompt_tokens` for `min(ids.len(),
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
    /// refusing rather than silently truncating (EMBED-WIRE item 3).
    ///
    /// `crate::embeddings::create_embeddings` uses this to reject an
    /// over-length **text** input up front with `400
    /// context_length_exceeded`, naming the real token count, instead of
    /// silently truncating it and billing `usage.prompt_tokens` for more
    /// than the model actually saw (the defect the wave-4 verifier's
    /// minor[3] named). `None` — the default — means "no such ceiling is
    /// known", so no guard is applied and the historical behaviour
    /// (whatever the backend itself does) is unchanged.
    ///
    /// This governs the **text** input path only. The raw token-id path
    /// (`"input": [1, 2, 3]`, [`TokenSequenceEmbedder`]) deliberately keeps
    /// truncating rather than erroring — see that trait's docs and
    /// [`ModelEmbedder::embed_tokens`]'s "Truncation" section for why.
    fn max_input_tokens(&self) -> Option<usize> {
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
    /// **once** for the whole batch (EMBED-WIRE item 2) instead of once per
    /// sequence — the same batched-lock treatment
    /// [`BatchEmbedder::embed_batch`] gives the text path, extended to
    /// `"input": [[1, 2], [3, 4]]` requests.
    fn embed_token_batches(&self, batches: &[Vec<u32>]) -> Vec<Result<Vec<f32>, RagError>> {
        batches
            .iter()
            .map(|ids| self.embed_token_ids(ids))
            .collect()
    }
}

/// A batch-capable embedding backend: computes every text in a request under
/// **one** critical section instead of one per item (EMBED-WIRE item 2).
///
/// # Why this trait exists instead of a method on `Embedder`
///
/// The natural home for this is `oxibonsai_rag::embedding::Embedder` itself,
/// as a default-provided method (`fn embed_batch(&self, texts: &[&str]) ->
/// Result<Vec<Vec<f32>>, RagError> { texts.iter().map(|t|
/// self.embed(t)).collect() }`, backward-compatible with every existing
/// implementation since it would carry a default body). `Embedder` lives in
/// `crates/oxibonsai-rag/src/embedding.rs`, which is **not** in this
/// package's `owned_files` — recorded as a deviation rather than edited.
///
/// Until that lands, [`crate::embeddings::EmbedderRegistry`] detects an
/// installed backend that also implements this LOCAL trait the same way it
/// already detects [`EmbeddingTokenCounter`] / [`TokenSequenceEmbedder`] (a
/// second, optional `Arc`-typed field wired by
/// [`EmbedderRegistry::with_model`](crate::embeddings::EmbedderRegistry::with_model)),
/// and prefers it over the one-at-a-time [`Embedder::embed`] loop.
pub trait BatchEmbedder: Send + Sync {
    /// Embed every text in `texts` under one lock acquisition (or equivalent
    /// single critical section), one result per input in order. A per-item
    /// embedding failure is reported in that item's own slot — it never
    /// aborts the whole batch, matching
    /// [`EmbedderRegistry::embed_texts`](crate::embeddings::EmbedderRegistry::embed_texts)'s
    /// existing per-item degrade-to-zero-vector contract.
    fn embed_batch(&self, texts: &[String]) -> Vec<Result<Vec<f32>, RagError>>;
}

/// Lock `mutex`, recovering from lock poisoning instead of propagating a panic.
///
/// Mirrors `crate::embeddings::lock_or_recover`'s policy: one unrelated panic
/// must not turn a live route into a process-lifetime outage. Recovery is safe
/// here even though the guarded value is a whole [`InferenceEngine`], whose KV
/// cache a panic mid-forward genuinely could leave half-written, because the
/// embedding path reads exactly four pieces of engine/model state and
/// [`InferenceEngine::embed`] clears all four *before* running the pass: the
/// host KV cache, its coherence watermark, the MET-05 device-KV latch and the
/// recurrent state. It reads nothing else — notably, no step of
/// `forward_hidden` consults the engine's
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

/// The typed refusal a hybrid (`qwen35`) engine answers an embedding request
/// with: mean-pooled embeddings need `forward_hidden`, which only the dense
/// model implements today (EMBED-WIRE adds the hybrid seam).
fn hybrid_embedding_refusal(architecture: &str) -> RuntimeError {
    crate::engine_seam::EngineError::NotADenseModel {
        operation: "embeddings (InferenceEngine::embed / ModelEmbedder)",
        architecture: architecture.to_string(),
        reason: "mean-pooled embeddings need the model's pre-LM-head hidden states \
                 (`forward_hidden`), which the hybrid model does not expose yet",
    }
    .into()
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
    /// Runs on this engine's own kernel dispatcher, so an engine pinned to
    /// `KernelTier::Reference` embeds on the scalar CPU path and a Metal
    /// engine embeds through the Metal per-layer GEMVs.
    ///
    /// # This clobbers the engine's per-sequence state — and nothing else's
    ///
    /// The underlying
    /// [`forward_hidden`](oxibonsai_model::model::BonsaiModel::forward_hidden)
    /// writes host KV positions `0..tokens.len()`, so this must not be
    /// interleaved with a generation in flight on the same engine. It clears
    /// this model's host KV cache, MET-05 latch and recurrent state on both
    /// sides of the pass; [`ModelEmbedder`] additionally serialises callers
    /// behind a `Mutex` on a dedicated engine.
    ///
    /// **It deliberately does not call [`InferenceEngine::reset`].** That would
    /// reach `BonsaiModel::reset`, which releases the **process-global** Metal
    /// device KV cache — wiping the state of a chat completion another engine
    /// may be decoding through the fused Metal path at that moment. Embedding
    /// never reads that cache, so it must not release it. What `reset` would
    /// add over the narrower clear is exactly (a) that global release and (b)
    /// this engine's own attached
    /// [`RecurrentState`](crate::engine_control::RecurrentState), which is
    /// cleared here explicitly.
    ///
    /// # Errors
    ///
    /// * [`RuntimeError::Model`] wrapping [`ModelError::ShapeInvariant`] —
    ///   `text_tokens` is empty.
    /// * [`RuntimeError::Model`] wrapping [`ModelError::SequenceTooLong`] —
    ///   the input is longer than the model's effective context.
    /// * Anything the forward pass returns (e.g. a token id past the
    ///   vocabulary).
    pub fn embed(&mut self, text_tokens: &[u32]) -> RuntimeResult<Vec<f32>> {
        if text_tokens.is_empty() {
            return Err(empty_input_error("InferenceEngine::embed"));
        }
        // ENGINE-SEAM: only the dense model has `forward_hidden` today. A
        // hybrid engine refuses with the typed error rather than pooling
        // something else (EMBED-WIRE adds the hybrid hidden-state seam).
        if let oxibonsai_model::hybrid::LoadedModel::Hybrid(model) = &self.model {
            return Err(hybrid_embedding_refusal(&model.config().base.architecture));
        }
        // The engine-level half of the reset: an attached `RecurrentState`
        // (RT-28) lives here, not on the model, so `forward_hidden`'s own
        // narrow clear cannot reach it. The model-level half happens inside
        // `forward_hidden`, on both sides of the block loop.
        self.reset_recurrent();
        // Disjoint field borrows: `&mut self.model` and `&self.kernel`, the
        // same shape `decode_step` already uses.
        let result = match &mut self.model {
            oxibonsai_model::hybrid::LoadedModel::Dense(model) => {
                model.embed_mean_pooled(text_tokens, &self.kernel)
            }
            oxibonsai_model::hybrid::LoadedModel::Hybrid(model) => {
                return Err(hybrid_embedding_refusal(&model.config().base.architecture));
            }
        };
        self.reset_recurrent();
        Ok(result?)
    }

    /// Dimensionality [`embed`](Self::embed) produces: the model's hidden size.
    pub fn embedding_dim(&self) -> usize {
        self.hidden_size()
    }

    /// Highest number of tokens [`embed`](Self::embed) can accept before the
    /// model refuses with [`ModelError::SequenceTooLong`].
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
    /// instrumentation only (EMBED-WIRE item 2's "batched engine lock"
    /// acceptance test needs an observable counter) — see
    /// [`lock_acquisitions`](Self::lock_acquisitions).
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

    /// Wrap an already-built engine, refusing a hybrid one up front.
    ///
    /// # Errors
    ///
    /// The typed [`EngineError::NotADenseModel`](crate::engine_seam::EngineError::NotADenseModel)
    /// for a hybrid (`qwen35`) engine — every [`embed`](Embedder::embed) on it
    /// would refuse, so building the embedder at all would only advertise an
    /// endpoint that cannot answer.
    pub fn from_engine(
        engine: InferenceEngine<'static>,
        tokenizer: Arc<TokenizerBridge>,
    ) -> RuntimeResult<Arc<Self>> {
        if engine.is_hybrid() {
            return Err(hybrid_embedding_refusal(engine.architecture()));
        }
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
    /// and no second copy of the `vocab × hidden` table.
    ///
    /// # Errors
    ///
    /// Anything [`InferenceEngine::from_gguf_static_with_embd`] returns.
    pub fn from_static_gguf(
        gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static>,
        token_embd: Arc<[f32]>,
        tokenizer: Arc<TokenizerBridge>,
        sampling_params: crate::sampling::SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Arc<Self>> {
        // Refuse a hybrid file before paying for a second model instance
        // (its KV cache, recurrent state and scratch) that could never embed.
        if oxibonsai_model::hybrid::LoadedModel::is_hybrid_gguf(gguf) {
            return Err(hybrid_embedding_refusal(
                &oxibonsai_model::hybrid::LoadedModel::architecture_of(gguf),
            ));
        }
        let engine = InferenceEngine::from_gguf_static_with_embd(
            gguf,
            sampling_params,
            seed,
            max_seq_len,
            token_embd,
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
    /// can assert that an N-input batch — through [`embed_batch`](Self::embed_batch),
    /// [`TokenSequenceEmbedder::embed_token_batches`], or
    /// [`crate::embeddings::EmbedderRegistry::embed_texts`] once a
    /// [`BatchEmbedder`] backend is installed — takes this lock **once**,
    /// not once per item (EMBED-WIRE item 2). Monotonically increasing,
    /// `Relaxed` ordering: callers compare a before/after delta, never an
    /// absolute value, so no stronger ordering is needed.
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
    /// # Truncation (EMBED-WIRE item 3)
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

    /// Embed every text in `texts`, one per returned entry.
    ///
    /// A plain loop (the `Embedder` contract is one text at a time and the
    /// model seam has no batched hidden-state entry point), but it takes the
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
    pub fn embed_batch(&self, texts: &[String]) -> Vec<Result<Vec<f32>, RagError>> {
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
                    .encode(text)
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
    /// `ModelEmbedder` — EMBED-WIRE item 2's batched-lock treatment extended
    /// to `"input": [[1, 2], [3, 4]]` requests, not just plain text). A
    /// per-sequence failure is reported in that sequence's own slot.
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
}

impl EmbeddingTokenCounter for ModelEmbedder {
    fn count_tokens(&self, text: &str) -> Option<usize> {
        self.tokenizer.encode(text).ok().map(|ids| ids.len())
    }

    fn max_input_tokens(&self) -> Option<usize> {
        Some(self.max_tokens)
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

impl BatchEmbedder for ModelEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Vec<Result<Vec<f32>, RagError>> {
        // Resolves to the inherent `ModelEmbedder::embed_batch` above (Rust
        // prefers an inherent method over a trait method of the same name
        // even from inside that trait's own impl block), so this is a plain
        // delegation, not recursion.
        self.embed_batch(texts)
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
        let results = embedder.embed_batch(&["ab".to_string(), String::new(), "cd".to_string()]);
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
        assert!(embedder.embed_batch(&[]).is_empty());
    }

    // ── EMBED-WIRE item 2: the batched engine lock ────────────────────────

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
        let results = embedder.embed_batch(&texts);
        assert_eq!(results.len(), 4);
        assert_eq!(
            embedder.lock_acquisitions() - before,
            1,
            "a 4-input embed_batch call must take the engine lock exactly once, not once per \
             item"
        );
    }

    #[test]
    fn embed_batch_of_nothing_does_not_take_the_lock() {
        let embedder =
            ModelEmbedder::new(Arc::new(Mutex::new(weightless_engine())), char_tokenizer());
        let before = embedder.lock_acquisitions();
        assert!(embedder.embed_batch(&[]).is_empty());
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

    // ── EMBED-WIRE item 7: hybrid engines refuse embed() ───────────────────

    /// `InferenceEngine::embed` on a hybrid (`qwen35`) engine must return the
    /// typed `NOT_A_DENSE_MODEL` refusal, never silently pool something else
    /// or panic. Exercised on the real Bonsai 2 27B GGUF this workspace ships
    /// under `models/` — no synthetic hybrid fixture is reachable from this
    /// crate (`oxibonsai_model::hybrid::tests_support` is `pub(crate)` to
    /// `oxibonsai-model`; `oxibonsai-runtime`'s own `engine_seam_tests.rs`
    /// builds one but behind a private `mod tests` that only `engine_seam.rs`
    /// itself can name) — see `deviations`.
    ///
    /// Self-skips when no Bonsai 2 GGUF is found, honouring (in priority
    /// order) `OXI_BONSAI2_PTQ1_GGUF` / `OXI_BONSAI2_PQ2_GGUF` (this package
    /// family's established env vars,
    /// `crates/oxibonsai-model/tests/bonsai2_real/harness.rs`), then a
    /// repo-relative `models/` lookup via the testkit — never a hardcoded
    /// absolute path.
    ///
    /// Deliberately does **not** fall back to `OXI_MODEL`: in this repo that
    /// variable conventionally names the DENSE 1.7B/8B
    /// (`metal_concurrency_tests.rs`, ENGINE-SEAM's own verifier run), and a
    /// verifier or the between-wave gate that runs this crate's test suite
    /// with `OXI_MODEL` pointed at that dense model must not have THIS test
    /// try to load it and fail `assert!(engine.is_hybrid())`.
    #[test]
    fn embed_on_the_real_hybrid_27b_refuses_with_not_a_dense_model() {
        let path = std::env::var_os("OXI_BONSAI2_PTQ1_GGUF")
            .or_else(|| std::env::var_os("OXI_BONSAI2_PQ2_GGUF"))
            .map(std::path::PathBuf::from)
            .or_else(|| {
                oxibonsai_testkit::workspace::find_model("Ternary-Bonsai-2-27B-PTQ1_0.gguf")
            })
            .or_else(|| {
                oxibonsai_testkit::workspace::find_model("Ternary-Bonsai-2-27B-PQ2_0.gguf")
            });
        let Some(path) = path else {
            eprintln!(
                "embed_on_the_real_hybrid_27b_refuses_with_not_a_dense_model: no Bonsai 2 27B \
                 GGUF found (set OXI_BONSAI2_PTQ1_GGUF, OXI_BONSAI2_PQ2_GGUF, or \
                 OXIBONSAI_MODELS_DIR) -- skipping"
            );
            return;
        };
        let (mut engine, _gguf) =
            InferenceEngine::from_gguf_path_leaked(&path, greedy_params(), 42, 64)
                .expect("the real Bonsai 2 27B GGUF must load");
        assert!(
            engine.is_hybrid(),
            "a qwen35 GGUF must load as a hybrid engine"
        );

        let err = engine
            .embed(&[1, 2, 3])
            .expect_err("a hybrid engine must refuse embed(), not silently pool something else");
        assert_eq!(
            crate::engine_seam::engine_error_code(&err),
            Some("NOT_A_DENSE_MODEL"),
            "got {err:?}"
        );

        // `ModelEmbedder::from_engine` must refuse the same engine up front
        // too, rather than building an embedder that would fail on every
        // call — reuses the already-loaded engine instead of a second
        // multi-GB mmap.
        let from_engine_err = ModelEmbedder::from_engine(engine, char_tokenizer())
            .expect_err("ModelEmbedder::from_engine must refuse a hybrid engine up front");
        assert_eq!(
            crate::engine_seam::engine_error_code(&from_engine_err),
            Some("NOT_A_DENSE_MODEL"),
            "got {from_engine_err:?}"
        );
    }
}
