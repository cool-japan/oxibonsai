//! Embedding backends for the RAG pipeline.
//!
//! This module defines the [`Embedder`] trait and two built-in implementations:
//!
//! - [`IdentityEmbedder`] — deterministic byte-hash embedding for tests.
//! - [`TfIdfEmbedder`] — simple bag-of-words TF-IDF embedding with no external deps.
//!
//! It also defines [`EmbedderState`], an optional extension trait that lets
//! an embedder's fittable internal state (e.g. `TfIdfEmbedder`'s vocabulary
//! and IDF table) be persisted and restored alongside a
//! [`crate::retriever::Retriever`]'s index (RAG-EVAL-IMG-10).

use std::collections::HashMap;

use crate::error::RagError;

// ─────────────────────────────────────────────────────────────────────────────
// Embedder trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait that all embedding backends must implement.
///
/// Implementations must be `Send + Sync` so they can be shared across threads.
pub trait Embedder: Send + Sync {
    /// Embed a text string, returning a dense `f32` vector.
    ///
    /// The returned vector always has exactly [`Embedder::embedding_dim`] elements.
    fn embed(&self, text: &str) -> Result<Vec<f32>, RagError>;

    /// The fixed number of dimensions produced by this embedder.
    fn embedding_dim(&self) -> usize;

    /// Embed every text in `texts`, returning one result per input, in
    /// order.
    ///
    /// A failure is reported in that item's own slot and never aborts the
    /// rest of the batch, so a caller can degrade item by item (the HTTP
    /// embeddings endpoint answers a failed item with a zero vector). The
    /// default is a plain loop over [`Embedder::embed`]; a backend with a
    /// cheaper batched path — one lock acquisition, one scheduling unit —
    /// overrides it, and callers holding a `dyn Embedder` reach that
    /// override through this method.
    fn embed_batch(&self, texts: &[&str]) -> Vec<Result<Vec<f32>, RagError>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EmbedderState — optional persistence extension
// ─────────────────────────────────────────────────────────────────────────────

/// Optional extension for [`Embedder`] implementations that carry fittable
/// internal state (a vocabulary, IDF weights, learned weights, …), which
/// must be persisted alongside the vector store for
/// [`crate::retriever::Retriever::save`]/[`crate::retriever::Retriever::load_with_embedder`]
/// to round-trip an equivalent embedder (RAG-EVAL-IMG-10).
///
/// An embedder with nothing to persist beyond what the caller already
/// supplies to reconstruct it (like [`IdentityEmbedder`], whose only
/// parameter is `dim`) implements this trivially: `to_state` returns `None`,
/// and `from_state` builds a fresh instance from `dim_hint`.
pub trait EmbedderState: Embedder + Sized {
    /// Serialise this embedder's fittable state as opaque JSON, or `None` if
    /// there is nothing to persist.
    fn to_state(&self) -> Option<serde_json::Value>;

    /// Reconstruct an embedder from a previously-serialised state.
    ///
    /// `dim_hint` is the vector store's dimensionality (always available,
    /// even when `state` is `None`) — embedders whose only real parameter
    /// *is* the dimension (e.g. [`IdentityEmbedder`]) can reconstruct
    /// themselves from it alone.
    fn from_state(state: Option<&serde_json::Value>, dim_hint: usize) -> Result<Self, RagError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// IdentityEmbedder
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic, hash-based embedder intended for testing.
///
/// Converts a text string into a fixed-dimensional `f32` vector by iterating
/// over the bytes and accumulating them into bins using FNV-1a mixing.  The
/// output is L2-normalised so cosine similarity between identical texts is 1.0.
pub struct IdentityEmbedder {
    dim: usize,
}

impl IdentityEmbedder {
    /// Create an embedder that produces vectors of length `dim`.
    ///
    /// Returns [`RagError::DimensionMismatch`] if `dim` is zero.
    pub fn new(dim: usize) -> Result<Self, RagError> {
        if dim == 0 {
            return Err(RagError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        Ok(Self { dim })
    }

    /// Internal: hash bytes of `text` into a vector of `dim` floats.
    ///
    /// Uses a SplitMix64-inspired per-dimension mixing to produce a
    /// deterministic, well-distributed unit vector.  Each dimension `d` is
    /// seeded from the text bytes using an independent mixing step so that
    /// even very short texts produce a non-zero vector.
    fn hash_to_vec(&self, text: &str) -> Vec<f32> {
        let text_bytes = text.as_bytes();
        // Compute a 64-bit text fingerprint using FNV-1a
        let mut fingerprint: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        for &byte in text_bytes {
            fingerprint ^= byte as u64;
            fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01B3);
        }
        // Also fold in the text length so that "" != " "
        fingerprint ^= text_bytes.len() as u64;
        fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01B3);

        // For each dimension, derive a float using SplitMix64 stepping
        (0..self.dim)
            .map(|d| {
                // Mix dimension index with the fingerprint
                let mut z =
                    fingerprint.wrapping_add((d as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                // SplitMix64 finaliser
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                // Map to a float in (-1, 1) with guaranteed non-zero magnitude.
                // Interpret as signed and scale to float range.
                let signed = z as i64;
                // Divide by half of i64::MAX to get a value in [-2, 2], then
                // clamp to (-1, 1) range to stay inside unit-ball territory.
                let f = (signed as f64 / (i64::MAX as f64)).clamp(-1.0, 1.0) as f32;
                // Ensure non-zero: if exactly zero (astronomically unlikely),
                // substitute a small constant derived from the dimension.
                if f == 0.0 {
                    ((d + 1) as f32) * 1e-7
                } else {
                    f
                }
            })
            .collect()
    }
}

impl Embedder for IdentityEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
        let mut v = self.hash_to_vec(text);
        l2_normalize(&mut v);
        Ok(v)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

impl EmbedderState for IdentityEmbedder {
    fn to_state(&self) -> Option<serde_json::Value> {
        // Nothing to persist: `dim` is already captured by the vector
        // store's own dimensionality, which `from_state` receives as
        // `dim_hint`.
        None
    }

    fn from_state(_state: Option<&serde_json::Value>, dim_hint: usize) -> Result<Self, RagError> {
        IdentityEmbedder::new(dim_hint)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TfIdfEmbedder
// ─────────────────────────────────────────────────────────────────────────────

/// Options for [`TfIdfEmbedder::fit_with_options`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TfIdfFitOptions {
    /// Caps the vocabulary size; the terms surviving `min_df`/`max_df_ratio`
    /// filtering with the highest document frequency are retained.
    pub max_features: usize,
    /// Minimum document frequency (inclusive) for a term to be considered
    /// at all.  Terms appearing in fewer than `min_df` documents are
    /// dropped as noise (typos, one-off tokens). Default `1` (no filtering
    /// — every term appears in at least one document by construction).
    pub min_df: usize,
    /// Maximum document-frequency *ratio* (`df / n_docs`, inclusive) for a
    /// term to be considered.  Terms appearing in more than this fraction
    /// of documents are dropped as uninformative near-universal stop-words
    /// — this is the direct fix for the fact that plain `fit` used to keep
    /// the *highest*-document-frequency terms, i.e. it preferentially kept
    /// stop-words and discarded the rare, discriminative terms IDF
    /// weighting exists to emphasise (RAG-EVAL-IMG-21). Default `1.0` (no
    /// filtering — a document-frequency ratio can never exceed `1.0`).
    pub max_df_ratio: f32,
}

impl Default for TfIdfFitOptions {
    fn default() -> Self {
        Self {
            max_features: 4096,
            min_df: 1,
            max_df_ratio: 1.0,
        }
    }
}

/// Simple bag-of-words TF-IDF embedder with no external dependencies.
///
/// Build the vocabulary from a corpus with [`TfIdfEmbedder::fit`] (or
/// [`TfIdfEmbedder::fit_with_options`] for `min_df`/`max_df_ratio` control),
/// then embed new texts with [`Embedder::embed`].  The output dimensionality
/// equals the vocabulary size (capped at `max_features`).
pub struct TfIdfEmbedder {
    /// term → column index in the TF-IDF vector
    vocab: HashMap<String, usize>,
    /// Inverse document frequency for each vocabulary term (indexed by column)
    idf: Vec<f32>,
    /// Fixed output dimension == vocab size
    dim: usize,
    /// When `true`, [`Embedder::embed`] returns
    /// [`RagError::EmptyQueryVector`] for a query whose every token is
    /// out-of-vocabulary (an all-zero TF-IDF vector) instead of silently
    /// returning that all-zero vector. Default `false` — see
    /// [`TfIdfEmbedder::with_strict_oov`] and
    /// [`crate::retriever::Retriever::retrieve`], which independently
    /// rejects a zero-norm query against a similarity-metric store
    /// regardless of this flag.
    strict_oov: bool,
}

impl TfIdfEmbedder {
    /// Build vocabulary and IDF weights from a corpus.
    ///
    /// Tokenisation is whitespace + punctuation splitting, lowercased.
    /// Stop-words are not removed — the caller may pre-filter if desired,
    /// or use [`TfIdfEmbedder::fit_with_options`] with `max_df_ratio`.
    ///
    /// The `max_features` parameter caps the vocabulary size; the most
    /// frequent terms are retained (equivalent to
    /// [`TfIdfEmbedder::fit_with_options`] with `min_df: 1, max_df_ratio: 1.0`).
    pub fn fit(documents: &[&str], max_features: usize) -> Self {
        Self::fit_with_options(
            documents,
            TfIdfFitOptions {
                max_features,
                ..TfIdfFitOptions::default()
            },
        )
    }

    /// Like [`TfIdfEmbedder::fit`], with `min_df`/`max_df_ratio` filtering
    /// (RAG-EVAL-IMG-21).
    pub fn fit_with_options(documents: &[&str], options: TfIdfFitOptions) -> Self {
        let max_features = options.max_features.max(1);
        let n_docs = documents.len().max(1);

        // Count document frequency for each token
        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in documents {
            let tokens = tokenize(doc);
            let unique: std::collections::HashSet<String> = tokens.into_iter().collect();
            for tok in unique {
                *df.entry(tok).or_insert(0) += 1;
            }
        }

        // Drop terms outside [min_df, max_df_ratio * n_docs] *before*
        // truncating by max_features, so common-but-not-universal
        // stop-words don't crowd out discriminative terms.
        let max_df_ratio = options.max_df_ratio.clamp(0.0, 1.0);
        df.retain(|_, &mut doc_freq| {
            doc_freq >= options.min_df && (doc_freq as f32) <= max_df_ratio * n_docs as f32
        });

        // Sort by document frequency descending; take top max_features
        let mut df_vec: Vec<(String, usize)> = df.into_iter().collect();
        df_vec.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        df_vec.truncate(max_features);

        let dim = df_vec.len();
        let mut vocab = HashMap::with_capacity(dim);
        let mut idf = vec![0.0f32; dim];

        for (idx, (term, doc_freq)) in df_vec.into_iter().enumerate() {
            vocab.insert(term, idx);
            // Smooth IDF: log((1 + n_docs) / (1 + df)) + 1
            idf[idx] = ((1.0 + n_docs as f32) / (1.0 + doc_freq as f32)).ln() + 1.0;
        }

        Self {
            vocab,
            idf,
            dim,
            strict_oov: false,
        }
    }

    /// Reconstruct a `TfIdfEmbedder` directly from its parts (used by
    /// [`EmbedderState::from_state`] and available to callers who persist
    /// the vocabulary/IDF table themselves).
    ///
    /// Validates that `idf.len() == vocab.len()` and that `vocab`'s values
    /// are a permutation of `0..vocab.len()` — a duplicate or
    /// out-of-range column index would otherwise silently drop or corrupt
    /// a dimension on every subsequent `embed` call.
    pub fn from_parts(
        vocab: HashMap<String, usize>,
        idf: Vec<f32>,
        strict_oov: bool,
    ) -> Result<Self, RagError> {
        let dim = idf.len();
        if vocab.len() != dim {
            return Err(RagError::Persistence(format!(
                "TfIdfEmbedder::from_parts: vocab has {} terms but idf has {} entries",
                vocab.len(),
                dim
            )));
        }
        let mut seen = vec![false; dim];
        for &idx in vocab.values() {
            if idx >= dim || seen[idx] {
                return Err(RagError::Persistence(
                    "TfIdfEmbedder::from_parts: vocab column indices are not a valid \
                     permutation of 0..dim (duplicate or out-of-range index)"
                        .to_string(),
                ));
            }
            seen[idx] = true;
        }
        Ok(Self {
            vocab,
            idf,
            dim,
            strict_oov,
        })
    }

    /// Compute a raw term-frequency vector for `text` (no IDF weighting).
    ///
    /// Useful for inspecting term counts without the IDF transform.
    pub fn embed_bow(&self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);
        let n_tokens = tokens.len().max(1) as f32;
        let mut tf = vec![0.0f32; self.dim];
        for tok in &tokens {
            if let Some(&idx) = self.vocab.get(tok) {
                tf[idx] += 1.0;
            }
        }
        // Normalise by document length → term frequency
        for v in tf.iter_mut() {
            *v /= n_tokens;
        }
        tf
    }

    /// The vocabulary size (= output dimension).
    pub fn vocab_size(&self) -> usize {
        self.dim
    }

    /// Retrieve the vocabulary mapping (term → column index).
    pub fn vocab(&self) -> &HashMap<String, usize> {
        &self.vocab
    }

    /// Retrieve the per-column IDF weights.
    pub fn idf(&self) -> &[f32] {
        &self.idf
    }

    /// Whether this embedder rejects fully out-of-vocabulary queries with
    /// an error (see [`TfIdfEmbedder::with_strict_oov`]).
    pub fn strict_oov(&self) -> bool {
        self.strict_oov
    }

    /// Enable or disable strict out-of-vocabulary rejection (builder).
    ///
    /// When enabled, [`Embedder::embed`] returns
    /// [`RagError::EmptyQueryVector`] instead of an all-zero vector when
    /// none of `text`'s tokens are in the vocabulary (RAG-EVAL-IMG-21).
    /// Defaults to `false` so existing callers who rely on always getting a
    /// vector back (e.g. to embed it anyway and accept a meaningless score)
    /// keep that behaviour; new callers, and every query routed through
    /// [`crate::retriever::Retriever`], are protected either way — see
    /// [`crate::retriever::Retriever::retrieve`]'s independent zero-norm
    /// guard for similarity-metric stores.
    #[must_use]
    pub fn with_strict_oov(mut self, strict: bool) -> Self {
        self.strict_oov = strict;
        self
    }
}

impl Embedder for TfIdfEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
        if self.dim == 0 {
            return Err(RagError::EmbeddingFailed(
                "TfIdfEmbedder has an empty vocabulary".into(),
            ));
        }
        let mut tf = self.embed_bow(text);
        // Apply IDF weighting
        for (i, v) in tf.iter_mut().enumerate() {
            *v *= self.idf[i];
        }
        if self.strict_oov {
            let norm_sq: f32 = tf.iter().map(|x| x * x).sum();
            if norm_sq <= 1e-20 {
                // Typed as `EmptyQueryVector` (not `EmbeddingFailed`), matching
                // `retriever.rs::reject_degenerate_query`'s independent guard for
                // the same condition (RAG-21 / verifier REQUIRED #11): a
                // zero-norm embedding is a distinct, structurally-detectable
                // outcome from "the backend could not produce a vector at
                // all", and callers should be able to `matches!` on it without
                // string-sniffing a message. The diagnostic the message used to
                // carry (vocabulary size, query preview) is still available via
                // this trace event.
                tracing::debug!(
                    vocab_size = self.dim,
                    query_preview = %text.chars().take(80).collect::<String>(),
                    "TfIdfEmbedder::embed: query has no terms in the vocabulary (strict_oov)"
                );
                return Err(RagError::EmptyQueryVector);
            }
        }
        l2_normalize(&mut tf);
        Ok(tf)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

impl EmbedderState for TfIdfEmbedder {
    fn to_state(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "vocab": self.vocab,
            "idf": self.idf,
            "strict_oov": self.strict_oov,
        }))
    }

    fn from_state(state: Option<&serde_json::Value>, _dim_hint: usize) -> Result<Self, RagError> {
        let state = state.ok_or_else(|| {
            RagError::Persistence(
                "TfIdfEmbedder requires persisted state to reconstruct (vocabulary + IDF \
                 table); none was found in this snapshot"
                    .to_string(),
            )
        })?;
        let vocab: HashMap<String, usize> = state
            .get("vocab")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| RagError::Persistence(format!("invalid TfIdf vocab in snapshot: {e}")))?
            .unwrap_or_default();
        let idf: Vec<f32> = state
            .get("idf")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| RagError::Persistence(format!("invalid TfIdf idf in snapshot: {e}")))?
            .unwrap_or_default();
        let strict_oov = state
            .get("strict_oov")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        Self::from_parts(vocab, idf, strict_oov)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared math utilities
// ─────────────────────────────────────────────────────────────────────────────

/// L2-normalise `v` in place.  If the norm is zero (or very small), the vector
/// is left unchanged to avoid producing NaN.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Tokenise text into lowercase words, splitting on whitespace and common
/// punctuation.  Returns an empty `Vec` for empty input.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Embedder::embed_batch (EMBED-WIRE handover) ─────────────────────────

    /// An embedder that only implements the two required methods, counts
    /// `embed` calls and fails on one marker input.
    struct CountingEmbedder {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Embedder for CountingEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if text == "fail" {
                return Err(RagError::EmptyDocument);
            }
            Ok(vec![text.len() as f32, 1.0])
        }

        fn embedding_dim(&self) -> usize {
            2
        }
    }

    /// The default `embed_batch` is the per-item loop: one result per
    /// input, in order, a failure confined to its own slot, `embed` called
    /// exactly once per item — through a `dyn Embedder` too.
    #[test]
    fn default_embed_batch_is_a_per_item_loop_that_degrades_per_item() {
        let embedder = CountingEmbedder {
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let as_dyn: &dyn Embedder = &embedder;
        let results = as_dyn.embed_batch(&["a", "fail", "ccc"]);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().ok(), Some(&vec![1.0, 1.0]));
        assert!(matches!(results[1], Err(RagError::EmptyDocument)));
        assert_eq!(results[2].as_ref().ok(), Some(&vec![3.0, 1.0]));
        assert_eq!(
            embedder.calls.load(std::sync::atomic::Ordering::Relaxed),
            3,
            "the default embeds each item exactly once"
        );
        assert!(as_dyn.embed_batch(&[]).is_empty());
    }

    /// The built-in embedders inherit the default and agree with `embed`.
    #[test]
    fn built_in_embedders_batch_like_they_embed() {
        let identity = IdentityEmbedder::new(8).expect("dim 8");
        let texts = ["alpha", "beta"];
        let batched = identity.embed_batch(&texts);
        for (text, result) in texts.iter().zip(batched) {
            assert_eq!(result.ok(), identity.embed(text).ok());
        }
    }

    // ── RAG-EVAL-IMG-21: strict_oov ──────────────────────────────────────────

    #[test]
    fn default_embed_still_returns_zero_vector_for_oov() {
        // Existing behaviour (`test_tfidf_embedder_unknown_word_has_zero_weight`
        // in the crate's shared test module) is preserved by default: opt-in
        // is required for the strict error.
        let docs = ["cat sat mat", "bat rat hat"];
        let emb = TfIdfEmbedder::fit(&docs, 20);
        assert!(!emb.strict_oov());
        let v = emb.embed("zzz").expect("default behaviour must not error");
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn strict_oov_rejects_fully_out_of_vocabulary_query() {
        let docs = ["cat sat mat", "bat rat hat"];
        let emb = TfIdfEmbedder::fit(&docs, 20).with_strict_oov(true);
        let err = emb.embed("zzz");
        // Typed-error consistency (verifier REQUIRED #11): the strict-OOV
        // branch must construct the same `EmptyQueryVector` variant
        // `retriever.rs::reject_degenerate_query` already uses for a
        // zero-norm query, not the catch-all `EmbeddingFailed` (which stays
        // reserved for the *different*, configuration-error condition of an
        // empty vocabulary — see `tfidf_embed_empty_vocab_is_embedding_failed`
        // below).
        assert!(
            matches!(err, Err(RagError::EmptyQueryVector)),
            "expected EmptyQueryVector, got {err:?}"
        );
    }

    /// The empty-*vocabulary* case (`self.dim == 0`, a configuration error —
    /// nothing was ever fit) is a different condition from strict-OOV
    /// rejection and must stay `EmbeddingFailed`; the wave-1.5 addendum is
    /// explicit that this call site (`embed`'s very first check) is untouched
    /// by the `EmptyQueryVector` migration above.
    #[test]
    fn tfidf_embed_empty_vocab_is_embedding_failed() {
        let emb = TfIdfEmbedder::fit(&[], 20);
        assert_eq!(emb.vocab_size(), 0);
        let err = emb.embed("anything");
        assert!(
            matches!(err, Err(RagError::EmbeddingFailed(_))),
            "expected EmbeddingFailed for an empty vocabulary, got {err:?}"
        );
    }

    #[test]
    fn strict_oov_still_embeds_a_partially_known_query() {
        let docs = ["cat sat mat", "bat rat hat"];
        let emb = TfIdfEmbedder::fit(&docs, 20).with_strict_oov(true);
        let v = emb
            .embed("cat zzz")
            .expect("one known token must still embed");
        assert!(v.iter().any(|x| *x != 0.0));
    }

    // ── min_df / max_df_ratio ────────────────────────────────────────────────

    #[test]
    fn fit_default_matches_fit_with_default_options() {
        let docs = ["alpha beta", "beta gamma", "gamma delta"];
        let a = TfIdfEmbedder::fit(&docs, 10);
        let b = TfIdfEmbedder::fit_with_options(
            &docs,
            TfIdfFitOptions {
                max_features: 10,
                ..TfIdfFitOptions::default()
            },
        );
        assert_eq!(a.vocab_size(), b.vocab_size());
        let mut a_terms: Vec<&String> = a.vocab().keys().collect();
        let mut b_terms: Vec<&String> = b.vocab().keys().collect();
        a_terms.sort();
        b_terms.sort();
        assert_eq!(a_terms, b_terms);
    }

    #[test]
    fn max_df_ratio_excludes_near_universal_terms() {
        // "the" appears in every document; a tight max_df_ratio must drop it
        // even though it has the highest document frequency.
        let docs = [
            "the quick fox",
            "the lazy dog",
            "the rust language",
            "the go language",
        ];
        let filtered = TfIdfEmbedder::fit_with_options(
            &docs,
            TfIdfFitOptions {
                max_features: 10,
                min_df: 1,
                max_df_ratio: 0.5,
            },
        );
        assert!(
            !filtered.vocab().contains_key("the"),
            "a term in 100% of documents must be excluded at max_df_ratio=0.5"
        );
        assert!(filtered.vocab().contains_key("language"));
    }

    #[test]
    fn min_df_excludes_rare_terms() {
        let docs = ["alpha beta", "alpha gamma", "alpha delta"];
        let filtered = TfIdfEmbedder::fit_with_options(
            &docs,
            TfIdfFitOptions {
                max_features: 10,
                min_df: 2,
                max_df_ratio: 1.0,
            },
        );
        assert!(
            filtered.vocab().contains_key("alpha"),
            "df=3 must survive min_df=2"
        );
        assert!(
            !filtered.vocab().contains_key("beta"),
            "df=1 must be excluded by min_df=2"
        );
    }

    // ── EmbedderState ────────────────────────────────────────────────────────

    #[test]
    fn identity_embedder_state_roundtrips_via_dim_hint() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        assert!(emb.to_state().is_none());
        let restored = IdentityEmbedder::from_state(None, 16).expect("from_state");
        assert_eq!(restored.embedding_dim(), 16);
    }

    #[test]
    fn tfidf_embedder_state_roundtrips_vocab_and_idf() {
        let docs = ["rust is fast", "python is slow", "go is simple"];
        let emb = TfIdfEmbedder::fit(&docs, 50);
        let state = emb.to_state().expect("TfIdf must persist state");
        let restored = TfIdfEmbedder::from_state(Some(&state), emb.embedding_dim())
            .expect("from_state should succeed");
        assert_eq!(restored.vocab_size(), emb.vocab_size());
        assert_eq!(restored.vocab(), emb.vocab());
        assert_eq!(restored.idf(), emb.idf());
        // A restored embedder must embed identically to the original.
        let a = emb.embed("rust is great").expect("embed");
        let b = restored.embed("rust is great").expect("embed");
        assert_eq!(a, b);
    }

    #[test]
    fn tfidf_from_state_without_state_errors() {
        let err = TfIdfEmbedder::from_state(None, 8);
        assert!(matches!(err, Err(RagError::Persistence(_))));
    }

    #[test]
    fn tfidf_from_parts_rejects_length_mismatch() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        let err = TfIdfEmbedder::from_parts(vocab, vec![1.0, 2.0], false);
        assert!(matches!(err, Err(RagError::Persistence(_))));
    }

    #[test]
    fn tfidf_from_parts_rejects_duplicate_index() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 0); // duplicate column index
        let err = TfIdfEmbedder::from_parts(vocab, vec![1.0, 1.0], false);
        assert!(matches!(err, Err(RagError::Persistence(_))));
    }

    #[test]
    fn tfidf_from_parts_rejects_out_of_range_index() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 5); // out of range for dim=1
        let err = TfIdfEmbedder::from_parts(vocab, vec![1.0], false);
        assert!(matches!(err, Err(RagError::Persistence(_))));
    }

    // ── Pre-existing behaviour (regression guards) ──────────────────────────

    #[test]
    fn identity_embedder_produces_correct_dim_unchanged() {
        for dim in [8, 16, 32] {
            let emb = IdentityEmbedder::new(dim).expect("valid dim");
            let v = emb.embed("hello world").expect("embed");
            assert_eq!(v.len(), dim);
        }
    }

    #[test]
    fn tfidf_bow_sums_to_one_unchanged() {
        let docs = ["apple banana cherry", "cherry date elderberry"];
        let emb = TfIdfEmbedder::fit(&docs, 20);
        let bow = emb.embed_bow("apple cherry cherry");
        let total: f32 = bow.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }
}
