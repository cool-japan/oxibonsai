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
        // The engine-level half of the reset: an attached `RecurrentState`
        // (RT-28) lives here, not on the model, so `forward_hidden`'s own
        // narrow clear cannot reach it. The model-level half happens inside
        // `forward_hidden`, on both sides of the block loop.
        self.reset_recurrent();
        // Disjoint field borrows: `&mut self.model` and `&self.kernel`, the
        // same shape `decode_step` already uses.
        let result = self.model.embed_mean_pooled(text_tokens, &self.kernel);
        self.reset_recurrent();
        Ok(result?)
    }

    /// Dimensionality [`embed`](Self::embed) produces: the model's hidden size.
    pub fn embedding_dim(&self) -> usize {
        self.model.hidden_size()
    }

    /// Highest number of tokens [`embed`](Self::embed) can accept before the
    /// model refuses with [`ModelError::SequenceTooLong`].
    pub fn embedding_max_tokens(&self) -> usize {
        self.model.max_context()
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
}

impl std::fmt::Debug for ModelEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `InferenceEngine` is not `Debug` and locking it here could block a
        // request just to format a log line, so report only the shape.
        f.debug_struct("ModelEmbedder")
            .field("dim", &self.dim)
            .field("max_tokens", &self.max_tokens)
            .field("tokenizer", &self.tokenizer.backend())
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
        let engine = InferenceEngine::from_gguf_static_with_embd(
            gguf,
            sampling_params,
            seed,
            max_seq_len,
            token_embd,
        )?;
        Ok(Arc::new(Self::new(Arc::new(Mutex::new(engine)), tokenizer)))
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
}

impl TokenSequenceEmbedder for ModelEmbedder {
    fn embed_token_ids(&self, tokens: &[u32]) -> Result<Vec<f32>, RagError> {
        if tokens.is_empty() {
            return Err(RagError::EmptyDocument);
        }
        self.embed_tokens(tokens)
            .map_err(|e| RagError::EmbeddingFailed(e.to_string()))
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
}
