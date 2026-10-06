//! Retrieval pipeline: indexes documents and answers top-k queries.
//!
//! The [`Retriever`] is the core indexing and retrieval component.  It accepts
//! raw text documents, chunks them (with the configured [`ChunkConfig`] by
//! default, or a pluggable [`Chunker`] — see [`RetrieverBuilder`]), embeds
//! each chunk with an [`Embedder`] backend, and stores the resulting vectors
//! in a [`VectorStore`].  At query time it embeds the query string and
//! returns the most similar chunks, optionally narrowed by a
//! [`MetadataFilter`] (see [`Retriever::retrieve_filtered`]).
//!
//! Indexing a single document ([`Retriever::add_document`] and friends) is
//! transactional: every chunk is embedded and validated into a scratch
//! vector store first, and only merged into the live index if every chunk
//! succeeded, so a mid-document failure can never leave a partially-indexed
//! document behind.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::chunker::{chunk_document, Chunk, ChunkConfig, Chunker};
use crate::distance::Distance;
use crate::embedding::Embedder;
use crate::error::RagError;
use crate::metadata_filter::{MetadataFilter, MetadataValue};
use crate::vector_store::{SearchResult, VectorStore};

// ─────────────────────────────────────────────────────────────────────────────
// RetrieverConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration knobs for the [`Retriever`].
///
/// Marked `#[non_exhaustive]`; use [`RetrieverConfig::default`] plus the
/// `with_*` builders for forward-compatible construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RetrieverConfig {
    /// Maximum number of chunks to return per query.
    pub top_k: usize,
    /// Minimum acceptable score for a result to be returned; results below
    /// this threshold are discarded even if they fall within the top-k.
    ///
    /// **Sign convention**: this is compared
    /// against [`crate::vector_store::SearchResult::score`], i.e. the
    /// store's configured [`Distance`] *after* [`Distance::to_score`] —
    /// always "higher is better", but the achievable range depends on the
    /// metric:
    ///
    /// - Cosine: `[-1, 1]`; Angular: `[-1, 0]`; DotProduct: unbounded, sign
    ///   of the raw dot product.
    /// - Euclidean / Hamming: `(-inf, 0]` — the score is `-distance`, and a
    ///   distance is never negative, so **every** achievable score for
    ///   these metrics is at most zero.
    ///
    /// Because of that last point, this defaults to [`f32::MIN`] rather
    /// than the more ergonomic-looking `0.0`: a `0.0` default would
    /// silently discard *every* result for a Euclidean/Hamming/Angular
    /// store (only an exact zero-distance duplicate would ever pass).
    /// `f32::MIN` is also a convenient large *finite* sentinel to inspect
    /// by eye in a saved snapshot.
    ///
    /// **Non-finite values are safe to store here.** This is a `pub` field
    /// on a `#[non_exhaustive]` struct, so a non-finite value —
    /// `f32::NEG_INFINITY`/`f32::INFINITY`/`f32::NAN`; note that
    /// [`crate::vector_store::VectorStore::search`] itself uses
    /// `f32::NEG_INFINITY` as its own "no threshold" sentinel — can reach
    /// this field directly from outside the crate, bypassing
    /// [`RetrieverConfig::with_min_score`] entirely. Left to the derived
    /// `serde` behaviour, `serde_json` would serialise such a value as
    /// JSON `null`, which does not deserialise back into an `f32` — a
    /// `save`/`save_binary` call would return `Ok`, but the resulting file
    /// could then never be loaded again. To close that hole, this field is
    /// (de)serialised through `serialize_min_score`/
    /// `deserialize_min_score` below instead of the plain derive, which
    /// encode a non-finite value as the string `"-inf"`/`"inf"`/`"nan"` —
    /// so **every** `f32` value of this field, finite or not, round-trips
    /// through [`crate::retriever::Retriever::save`]/
    /// [`crate::retriever::Retriever::load`] and their binary counterparts
    /// (which embed this same JSON encoding — see the `binary` module in
    /// `persistence.rs`). Pick a metric-appropriate threshold with
    /// [`RetrieverConfig::with_min_score`] once you know which [`Distance`]
    /// the store uses.
    #[serde(
        serialize_with = "serialize_min_score",
        deserialize_with = "deserialize_min_score"
    )]
    pub min_score: f32,
    /// Whether to apply a secondary heuristic re-ranking pass after the
    /// initial similarity retrieval.  Currently the re-ranking pass
    /// boosts chunks that contain at least one query token (exact-match term
    /// overlap), breaking similarity ties in a more lexical direction.
    pub rerank: bool,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: f32::MIN,
            rerank: false,
        }
    }
}

impl RetrieverConfig {
    /// Set [`RetrieverConfig::top_k`] (builder).
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set [`RetrieverConfig::min_score`] (builder). See that field's
    /// documentation for the sign convention.
    #[must_use]
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Set [`RetrieverConfig::rerank`] (builder).
    #[must_use]
    pub fn with_rerank(mut self, rerank: bool) -> Self {
        self.rerank = rerank;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// `min_score` non-finite-safe (de)serialization
// ─────────────────────────────────────────────────────────────────────────────

/// `serde(serialize_with)` helper for [`RetrieverConfig::min_score`] — see
/// that field's documentation for why this exists.
///
/// Encodes a finite value exactly as the derived `Serialize` impl would (a
/// plain JSON number) and a non-finite value as one of the strings
/// `"-inf"`, `"inf"`, `"nan"` instead of the `null` `serde_json` would
/// otherwise produce for `NaN`/`±∞` — `null` does not deserialise back into
/// an `f32`, which is exactly the hazard this function closes.
fn serialize_min_score<S>(value: &f32, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if value.is_nan() {
        serializer.serialize_str("nan")
    } else if *value == f32::INFINITY {
        serializer.serialize_str("inf")
    } else if *value == f32::NEG_INFINITY {
        serializer.serialize_str("-inf")
    } else {
        serializer.serialize_f32(*value)
    }
}

/// `serde(deserialize_with)` counterpart of [`serialize_min_score`].
///
/// Accepts an ordinary JSON number (the finite case, in any numeric form a
/// `Deserializer` may hand back for it) or one of the sentinel strings
/// `"-inf"`/`"inf"`/`"nan"` written by [`serialize_min_score`] (plus a few
/// common spelling variants); any other string falls back to
/// `str::parse::<f32>`, so a hand-edited snapshot that quotes an ordinary
/// decimal value still loads.
///
/// This relies on [`serde::Deserializer::deserialize_any`], which requires
/// a self-describing format. Both snapshot formats this crate writes
/// qualify: plain JSON, and the compact binary format, which embeds
/// `RetrieverConfig` as a JSON blob inside an otherwise-binary envelope
/// (see the `binary` module in `persistence.rs`) precisely so it can reuse
/// this same `serde_json`-based (de)serialization. A future move of that
/// embedded blob to a non-self-describing codec would need a
/// format-specific decode path for this field instead of this function.
fn deserialize_min_score<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct MinScoreVisitor;

    impl serde::de::Visitor<'_> for MinScoreVisitor {
        type Value = f32;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a finite number, or one of the strings \"-inf\", \"inf\", \"nan\"")
        }

        fn visit_f64<E>(self, v: f64) -> Result<f32, E>
        where
            E: serde::de::Error,
        {
            Ok(v as f32)
        }

        fn visit_i64<E>(self, v: i64) -> Result<f32, E>
        where
            E: serde::de::Error,
        {
            Ok(v as f32)
        }

        fn visit_u64<E>(self, v: u64) -> Result<f32, E>
        where
            E: serde::de::Error,
        {
            Ok(v as f32)
        }

        fn visit_str<E>(self, v: &str) -> Result<f32, E>
        where
            E: serde::de::Error,
        {
            match v {
                "-inf" | "-infinity" | "-Infinity" => Ok(f32::NEG_INFINITY),
                "inf" | "+inf" | "infinity" | "Infinity" => Ok(f32::INFINITY),
                "nan" | "NaN" | "NAN" => Ok(f32::NAN),
                other => other
                    .parse::<f32>()
                    .map_err(|_| E::custom(format!("invalid min_score string {other:?}"))),
            }
        }
    }

    deserializer.deserialize_any(MinScoreVisitor)
}

// ─────────────────────────────────────────────────────────────────────────────
// Retriever
// ─────────────────────────────────────────────────────────────────────────────

/// Indexes chunked documents and retrieves the most relevant chunks for a query.
pub struct Retriever<E: Embedder> {
    store: VectorStore,
    embedder: E,
    config: RetrieverConfig,
    /// Total number of distinct documents indexed (each successful call to
    /// `add_document`/`add_document_with_metadata` increments this; a
    /// document that yields zero chunks does not, since nothing was
    /// actually indexed for it — see [`Retriever::produce_chunks`]).
    doc_count: usize,
    /// Optional pluggable chunking strategy; `None` uses [`chunk_document`]
    /// with the caller-supplied [`ChunkConfig`] (the historical default).
    /// See [`RetrieverBuilder::with_chunker`].
    chunker: Option<Box<dyn Chunker>>,
}

/// Reject a non-finite [`MetadataValue::Float`] before it can ever reach a
/// live chunk's metadata map.
///
/// This is the same hazard [`RetrieverConfig::min_score`] had before its
/// dedicated `serialize_min_score`/`deserialize_min_score` fix, reached
/// through a different, newly-added door: [`Retriever::add_document_with_metadata`]
/// and [`Retriever::add_chunks`] accept caller-supplied metadata directly,
/// so `MetadataValue::Float(f64::NAN)` (or `±∞`) can end up in the store —
/// `serde_json` serialises it as JSON `null`, and `MetadataValue`'s
/// untagged-enum `Deserialize` then fails to match *any* variant against
/// `null`, so `save`/`save_binary` return `Ok` for a snapshot that
/// `load`/`load_binary` can never read back. `MetadataValue` itself lives
/// in `metadata_filter.rs`, so unlike
/// `min_score` this cannot be closed with a `serde_with` attribute here;
/// rejecting it at the indexing boundary, mirroring
/// [`VectorStore::insert`]'s existing non-finite-vector guard
/// ([`RagError::NonFinite`]), closes it without touching that file.
fn reject_non_finite_metadata(metadata: &HashMap<String, MetadataValue>) -> Result<(), RagError> {
    for value in metadata.values() {
        if let MetadataValue::Float(f) = value {
            if !f.is_finite() {
                return Err(RagError::NonFinite);
            }
        }
    }
    Ok(())
}

impl<E: Embedder> Retriever<E> {
    /// Create a new retriever with `embedder` and `config`.
    ///
    /// The vector store is initialised with the dimensionality reported by
    /// `embedder`. Use [`RetrieverBuilder`] instead to also configure a
    /// custom [`Chunker`].
    pub fn new(embedder: E, config: RetrieverConfig) -> Self {
        let dim = embedder.embedding_dim();
        Self {
            store: VectorStore::new(dim),
            embedder,
            config,
            doc_count: 0,
            chunker: None,
        }
    }

    /// Configure a custom [`Chunker`] on an already-constructed retriever
    /// (builder-style; consumes and returns `self`).
    ///
    /// This is the same knob [`RetrieverBuilder::with_chunker`] sets during
    /// construction; it also exists as a method directly on `Retriever` so
    /// [`crate::pipeline::RagPipeline::with_chunker`] can apply it to the
    /// `Retriever` it already owns internally.
    #[must_use]
    pub fn with_chunker(mut self, chunker: Box<dyn Chunker>) -> Self {
        self.chunker = Some(chunker);
        self
    }

    /// Low-level constructor used by the persistence layer to reassemble a
    /// [`Retriever`] from its previously-persisted parts.
    ///
    /// Deliberately kept `#[doc(hidden)]`: unlike [`Retriever::add_chunks`],
    /// this replaces the store/config/doc_count *wholesale* and exists for
    /// the persistence layer's reconstruction path, not incremental
    /// indexing — [`Retriever::add_chunks`] and [`Retriever::store_mut`] are
    /// the supported public escape hatches for building an index by hand.
    #[doc(hidden)]
    pub fn from_parts(
        embedder: E,
        store: VectorStore,
        doc_count: usize,
        config: RetrieverConfig,
    ) -> Self {
        Self {
            store,
            embedder,
            config,
            doc_count,
            chunker: None,
        }
    }

    /// Split `text` into chunks using the configured [`Chunker`] (or
    /// [`chunk_document`] with `chunk_config` by default).
    ///
    /// Guarantees at least one chunk for any non-blank `text`: if the
    /// configured strategy produces none (e.g. the whole document is
    /// shorter than `chunk_config.min_chunk_size` — the default chunker's
    /// documented per-window behaviour), the whole (trimmed) document is
    /// indexed as a single chunk instead of being silently dropped from the
    /// corpus while `add_document` still reports success.
    fn produce_chunks(
        &self,
        text: &str,
        doc_id: usize,
        chunk_config: &ChunkConfig,
    ) -> Result<Vec<Chunk>, RagError> {
        let mut chunks = match &self.chunker {
            Some(chunker) => chunker.chunk(text, doc_id)?,
            None => {
                chunk_config
                    .validate()
                    .map_err(RagError::InvalidChunkConfig)?;
                chunk_document(text, doc_id, chunk_config)
            }
        };

        if chunks.is_empty() {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                debug!(
                    doc_id,
                    "chunker produced no chunks for a non-empty document; indexing it whole"
                );
                chunks.push(Chunk::new(trimmed.to_string(), doc_id, 0, 0));
            }
        }

        Ok(chunks)
    }

    /// Embed and insert `chunks` atomically: every chunk is embedded and
    /// validated into a scratch [`VectorStore`] first, and the whole batch
    /// is merged into the live store only if every chunk succeeded — a
    /// failure partway through (an embedder error, or a
    /// [`RagError::DimensionMismatch`]/[`RagError::NonFinite`] from
    /// [`VectorStore::insert`]) leaves the live store completely unchanged
    /// instead of half-updated.
    ///
    /// Also fills in each chunk's `doc_id`/`chunk_idx` metadata keys (in
    /// addition to the dedicated struct fields) *if not already present*,
    /// so `Exists`/`Equals` filters over them work out of the box even for
    /// chunks produced by the default chunker, which previously never
    /// attached any metadata at all.
    ///
    /// "If not already present" matters for
    /// [`Retriever::add_document_with_metadata`]: caller-supplied metadata
    /// is merged into each chunk *before* `stage_and_commit` runs, so a
    /// caller who deliberately sets their own `"doc_id"`/`"chunk_idx"`
    /// metadata value keeps it instead of having it silently overwritten by
    /// the automatic stamp (an unconditional `insert`
    /// here previously won regardless of merge order).
    fn stage_and_commit(&mut self, mut chunks: Vec<Chunk>) -> Result<usize, RagError> {
        for chunk in &mut chunks {
            // Computed before touching `chunk.metadata` so the two
            // `entry(..).or_insert(..)` calls below don't need a closure
            // that borrows `chunk` while `chunk.metadata` is itself
            // mutably borrowed.
            let doc_id_value = MetadataValue::Int(chunk.doc_id as i64);
            let chunk_idx_value = MetadataValue::Int(chunk.chunk_idx as i64);
            chunk
                .metadata
                .entry("doc_id".to_string())
                .or_insert(doc_id_value);
            chunk
                .metadata
                .entry("chunk_idx".to_string())
                .or_insert(chunk_idx_value);
        }

        let mut staged = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let vector = self.embedder.embed(&chunk.text)?;
            staged.push((vector, chunk));
        }

        let mut scratch = VectorStore::new_with_distance(self.store.dim(), self.store.distance());
        for (vector, chunk) in staged {
            scratch.insert(vector, chunk)?;
        }

        let indexed = scratch.len();
        self.store.merge_from(scratch);
        Ok(indexed)
    }

    /// Index a single document, attaching `metadata` to every chunk produced
    /// from it (in addition to the automatic `doc_id`/`chunk_idx` keys).
    ///
    /// The document is split with the configured [`Chunker`] (or
    /// `chunk_config` by default), each chunk is embedded, and the
    /// resulting vectors are inserted atomically (see
    /// `Retriever::stage_and_commit`). Returns the number of chunks that
    /// were indexed. This is the metadata-carrying entry point;
    /// [`Retriever::add_document`] is this
    /// method with an empty metadata map.
    ///
    /// `"doc_id"` and `"chunk_idx"` are reserved metadata keys: every
    /// indexed chunk gets them auto-stamped from its
    /// [`crate::chunker::Chunk::doc_id`]/[`crate::chunker::Chunk::chunk_idx`]
    /// fields (see `Retriever::stage_and_commit`), but only as a
    /// fallback — if `metadata` already sets either key, that caller value
    /// is kept instead of being overwritten by the automatic stamp.
    ///
    /// Returns [`RagError::NonFinite`] if `metadata` contains a
    /// [`MetadataValue::Float`] that is `NaN` or `±∞`, before touching any
    /// state — such a value would otherwise save without error but could
    /// never be loaded back (see `reject_non_finite_metadata`).
    pub fn add_document_with_metadata(
        &mut self,
        text: &str,
        chunk_config: &ChunkConfig,
        metadata: HashMap<String, MetadataValue>,
    ) -> Result<usize, RagError> {
        if text.trim().is_empty() {
            return Err(RagError::EmptyDocument);
        }
        reject_non_finite_metadata(&metadata)?;

        let doc_id = self.doc_count;
        // `produce_chunks` guarantees a non-empty result whenever `text` is
        // non-blank (its whole-document fallback runs after *both* the
        // custom-`Chunker` and default-chunker branches, on this same
        // `text`), and `text.trim().is_empty()` was already rejected above
        // -- so `chunks` can never be empty here.
        let mut chunks = self.produce_chunks(text, doc_id, chunk_config)?;

        for chunk in &mut chunks {
            for (key, value) in &metadata {
                chunk.metadata.insert(key.clone(), value.clone());
            }
        }

        let indexed = self.stage_and_commit(chunks)?;
        self.doc_count += 1;
        debug!(doc_id, indexed, "document indexed");
        Ok(indexed)
    }

    /// Index a single document.
    ///
    /// The document is split with `chunk_config`, each chunk is embedded,
    /// and the resulting vectors are inserted into the vector store.
    ///
    /// Returns the number of chunks that were successfully indexed.
    pub fn add_document(
        &mut self,
        text: &str,
        chunk_config: &ChunkConfig,
    ) -> Result<usize, RagError> {
        self.add_document_with_metadata(text, chunk_config, HashMap::new())
    }

    /// Index pre-built [`crate::chunker::Chunk`]s directly, bypassing the
    /// configured chunker entirely.
    ///
    /// This is the escape hatch for chunking strategies that don't fit the
    /// [`Chunker`] trait (e.g. converting [`crate::advanced_chunker::RichChunk`]s
    /// by hand with per-chunk metadata already attached), and for
    /// re-indexing chunks produced out-of-band. Callers own `doc_id`
    /// assignment on the chunks they pass in; this method does **not**
    /// advance [`Retriever::document_count`] (it has no way to know how
    /// many distinct documents `chunks` spans — it may be zero, one, or
    /// several), so callers doing document-level accounting should track it
    /// themselves. Embedding and insertion are still atomic, exactly like
    /// [`Retriever::add_document`].
    ///
    /// Returns [`RagError::NonFinite`] if any chunk's metadata contains a
    /// [`MetadataValue::Float`] that is `NaN` or `±∞`, checked before any
    /// chunk is embedded or inserted — see
    /// [`Retriever::add_document_with_metadata`]'s documentation and
    /// `reject_non_finite_metadata` for why.
    pub fn add_chunks(&mut self, chunks: Vec<Chunk>) -> Result<usize, RagError> {
        if chunks.is_empty() {
            return Ok(0);
        }
        for chunk in &chunks {
            reject_non_finite_metadata(&chunk.metadata)?;
        }
        self.stage_and_commit(chunks)
    }

    /// Index multiple documents, returning per-document chunk counts.
    ///
    /// Processing stops and the error is returned on the first failure.
    /// Because [`Retriever::add_document`] is itself atomic per document,
    /// any documents *before* the failing one remain
    /// indexed — only this call's own `Vec<usize>` of per-document counts is
    /// discarded on error; [`Retriever::document_count`] and
    /// [`Retriever::chunk_count`] still reflect what was actually indexed.
    pub fn add_documents(
        &mut self,
        texts: &[&str],
        chunk_config: &ChunkConfig,
    ) -> Result<Vec<usize>, RagError> {
        let mut counts = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            match self.add_document(text, chunk_config) {
                Ok(n) => counts.push(n),
                Err(e) => {
                    warn!(
                        failed_at = i,
                        documents_committed = counts.len(),
                        error = %e,
                        "add_documents: stopped at a failing document; the documents before it \
                         were already committed (add_document is atomic per document) even \
                         though this call now returns Err — see document_count()/chunk_count()"
                    );
                    return Err(e);
                }
            }
        }
        info!(
            documents = texts.len(),
            total_chunks = counts.iter().sum::<usize>(),
            "batch indexing complete"
        );
        Ok(counts)
    }

    /// Reject a query embedding that is degenerate for the store's metric,
    /// rather than silently scoring every entry identically and returning
    /// an insertion-order "ranking" with no error.
    ///
    /// A zero vector is only degenerate for similarity metrics whose score
    /// collapses to a constant at the origin — Cosine and Angular are
    /// defined to be `0`/`0.5` respectively for a zero input (see
    /// `distance.rs`), and DotProduct with a zero query is `0` for every
    /// entry by definition. Euclidean and Hamming treat the origin as an
    /// ordinary point and rank entries by their real distance from it, so
    /// they are excluded from this guard.
    fn reject_degenerate_query(&self, query_vec: &[f32]) -> Result<(), RagError> {
        let is_degenerate_at_zero = matches!(
            self.store.distance(),
            Distance::Cosine | Distance::Angular | Distance::DotProduct
        );
        if !is_degenerate_at_zero {
            return Ok(());
        }
        let norm_sq: f32 = query_vec.iter().map(|x| x * x).sum();
        if norm_sq <= 1e-20 {
            // RAG-21: a typed variant instead of a string-stuffed
            // `EmbeddingFailed`, so callers can match on the specific
            // condition (e.g. an out-of-vocabulary query against a
            // TF-IDF-style embedder) instead of parsing `Display` output.
            return Err(RagError::EmptyQueryVector);
        }
        Ok(())
    }

    /// Retrieve the top-k most relevant chunks for `query`, using
    /// [`RetrieverConfig::top_k`].
    ///
    /// Returns [`RagError::EmptyQuery`] if `query` is blank,
    /// [`RagError::NoDocumentsIndexed`] if the store is empty, and
    /// [`RagError::EmptyQueryVector`] if the embedded query is degenerate
    /// for the store's metric (see `Retriever::reject_degenerate_query`).
    pub fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, RagError> {
        self.retrieve_with_top_k(query, self.config.top_k)
    }

    /// Like [`Self::retrieve`], but uses an explicit `top_k` for this call
    /// instead of [`RetrieverConfig::top_k`].
    ///
    /// Exists for callers that must honour a *per-request* retrieval depth
    /// (e.g. an HTTP endpoint accepting a client-supplied `top_k`) while
    /// still going through the same [`RagError::EmptyQueryVector`] guard
    /// (RAG-21) and the retriever's own configured
    /// [`RetrieverConfig::min_score`] that [`Self::retrieve`] applies.
    /// Reconstructing the search call by hand against [`Self::store`]
    /// instead -- as `oxibonsai_runtime::rag_server::rag_query` used to --
    /// silently drops both.
    pub fn retrieve_with_top_k(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, RagError> {
        if query.trim().is_empty() {
            return Err(RagError::EmptyQuery);
        }
        if self.store.is_empty() {
            return Err(RagError::NoDocumentsIndexed);
        }

        let query_vec = self.embedder.embed(query)?;
        self.reject_degenerate_query(&query_vec)?;
        let mut results =
            self.store
                .search_with_threshold(&query_vec, top_k, self.config.min_score);

        if self.config.rerank {
            results = rerank(results, query);
        }

        debug!(
            query_len = query.len(),
            hits = results.len(),
            top_k,
            "retrieval complete"
        );
        Ok(results)
    }

    /// Retrieve the top-k chunks that pass a [`MetadataFilter`], honouring
    /// [`RetrieverConfig::min_score`] exactly like [`Retriever::retrieve`]
    /// (a prior version routed through
    /// [`VectorStore::search_filtered`] with the threshold hard-coded away,
    /// so adding a filter silently *widened* the result set).
    ///
    /// The filter is validated before any search work is performed.
    /// Returns [`RagError::EmptyQuery`] for blank queries,
    /// [`RagError::NoDocumentsIndexed`] if the store is empty, and
    /// [`RagError::InvalidFilter`] for malformed filters.
    pub fn retrieve_filtered(
        &self,
        query: &str,
        filter: &MetadataFilter,
    ) -> Result<Vec<SearchResult>, RagError> {
        if query.trim().is_empty() {
            return Err(RagError::EmptyQuery);
        }
        if self.store.is_empty() {
            return Err(RagError::NoDocumentsIndexed);
        }

        let query_vec = self.embedder.embed(query)?;
        self.reject_degenerate_query(&query_vec)?;
        let mut results = self.store.search_filtered(
            &query_vec,
            self.config.top_k,
            self.config.min_score,
            filter,
        )?;

        if self.config.rerank {
            results = rerank(results, query);
        }

        debug!(
            query_len = query.len(),
            hits = results.len(),
            "filtered retrieval complete"
        );
        Ok(results)
    }

    /// Like [`Self::retrieve`] but returns just the chunk text strings.
    pub fn retrieve_text(&self, query: &str) -> Result<Vec<String>, RagError> {
        Ok(self
            .retrieve(query)?
            .into_iter()
            .map(|r| r.chunk.text)
            .collect())
    }

    /// Number of distinct documents that have been indexed.
    pub fn document_count(&self) -> usize {
        self.doc_count
    }

    /// Total number of chunks currently in the vector store.
    pub fn chunk_count(&self) -> usize {
        self.store.len()
    }

    /// Borrow the underlying vector store.
    pub fn store(&self) -> &VectorStore {
        &self.store
    }

    /// Mutably borrow the underlying vector store — the supported way to
    /// reach [`VectorStore::delete`]/[`VectorStore::update`]/
    /// [`VectorStore::delete_by_doc_id`] through a `Retriever`.
    pub fn store_mut(&mut self) -> &mut VectorStore {
        &mut self.store
    }

    /// Borrow the embedder.
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Borrow the retriever's configuration.
    pub fn config(&self) -> &RetrieverConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RetrieverBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Fluent builder for [`Retriever`], for construction paths that need more
/// than [`Retriever::new`]'s `(embedder, config)` pair — currently, a
/// pluggable [`Chunker`]: previously, four of the
/// crate's five chunking strategies ([`crate::advanced_chunker::MarkdownChunker`],
/// [`crate::advanced_chunker::RecursiveCharSplitter`],
/// [`crate::advanced_chunker::SentenceChunker`],
/// [`crate::advanced_chunker::SlidingWindowChunker`], plus
/// [`crate::code_chunker::CodeChunker`]) had no way to reach `Retriever` at
/// all.
///
/// # Example
///
/// ```rust
/// use oxibonsai_rag::chunker::{Chunker, ParagraphChunker};
/// use oxibonsai_rag::embedding::IdentityEmbedder;
/// use oxibonsai_rag::retriever::{RetrieverBuilder, RetrieverConfig};
///
/// let embedder = IdentityEmbedder::new(32).expect("valid dim");
/// let chunker: Box<dyn Chunker> = Box::new(ParagraphChunker);
/// let mut retriever = RetrieverBuilder::new(embedder)
///     .with_config(RetrieverConfig::default().with_top_k(3))
///     .with_chunker(chunker)
///     .build();
///
/// retriever
///     .add_document("Paragraph one.\n\nParagraph two.", &Default::default())
///     .expect("index");
/// assert_eq!(retriever.chunk_count(), 2);
/// ```
pub struct RetrieverBuilder<E: Embedder> {
    embedder: E,
    config: RetrieverConfig,
    chunker: Option<Box<dyn Chunker>>,
}

impl<E: Embedder> RetrieverBuilder<E> {
    /// Start building a retriever around `embedder`, with
    /// [`RetrieverConfig::default`] and no custom chunker.
    pub fn new(embedder: E) -> Self {
        Self {
            embedder,
            config: RetrieverConfig::default(),
            chunker: None,
        }
    }

    /// Set the [`RetrieverConfig`] (builder).
    #[must_use]
    pub fn with_config(mut self, config: RetrieverConfig) -> Self {
        self.config = config;
        self
    }

    /// Set a custom [`Chunker`] (builder). Without this, the built
    /// `Retriever` chunks with [`chunk_document`], exactly like
    /// [`Retriever::new`].
    #[must_use]
    pub fn with_chunker(mut self, chunker: Box<dyn Chunker>) -> Self {
        self.chunker = Some(chunker);
        self
    }

    /// Build the configured [`Retriever`].
    pub fn build(self) -> Retriever<E> {
        let retriever = Retriever::new(self.embedder, self.config);
        match self.chunker {
            Some(chunker) => retriever.with_chunker(chunker),
            None => retriever,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-ranking helper
// ─────────────────────────────────────────────────────────────────────────────

/// Simple lexical re-ranker: bumps the score of chunks that contain at least
/// one query token.  This rewards exact-match overlap on top of the dense
/// similarity score, which helps short factual queries.
fn rerank(mut results: Vec<SearchResult>, query: &str) -> Vec<SearchResult> {
    let query_tokens: std::collections::HashSet<String> =
        crate::embedding::tokenize(query).into_iter().collect();

    for result in results.iter_mut() {
        let chunk_tokens: std::collections::HashSet<String> =
            crate::embedding::tokenize(&result.chunk.text)
                .into_iter()
                .collect();
        let overlap = query_tokens.intersection(&chunk_tokens).count();
        if overlap > 0 {
            // Small additive boost. Not clamped to <= 1.0 (RAG-08
            // interaction): that clamp was only ever harmless because
            // RAG-08 used to silently L2-normalise `DotProduct` scores
            // into [-1, 1] like Cosine; that normalisation was removed
            // (`vector_store.rs`'s `requires_normalized_inputs` excludes
            // `DotProduct`), so a `DotProduct` store's scores are now
            // unbounded, and clamping every *boosted* result to exactly
            // 1.0 destroyed ordering among them and inverted them relative
            // to any unboosted result already scoring above 1.
            // `SearchResult::score` has never documented an upper bound --
            // only "higher is better" -- so simply not capping it here is
            // correct for every metric, bounded or not.
            let boost = (overlap as f32 * 0.02).min(0.1);
            result.score += boost;
        }
    }

    // Re-sort after score adjustment
    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::{Chunk, FixedWindowChunker, ParagraphChunker};
    use crate::embedding::IdentityEmbedder;

    // ── RAG-EVAL-IMG-13: short documents are never dropped ──────────────────

    #[test]
    fn add_document_indexes_a_document_shorter_than_min_chunk_size() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut retriever = Retriever::new(emb, RetrieverConfig::default());
        // 3-word document, far under the default min_chunk_size (32 chars).
        let n = retriever
            .add_document("Tokyo is the capital.", &ChunkConfig::default())
            .expect("must index, not silently drop");
        assert_eq!(n, 1);
        assert_eq!(retriever.chunk_count(), 1);
        assert_eq!(retriever.document_count(), 1);
        let hits = retriever.retrieve("Tokyo").expect("retrieve");
        assert!(!hits.is_empty());
    }

    // ── RAG-EVAL-IMG-30: atomic indexing ─────────────────────────────────────

    #[test]
    fn add_document_rejects_invalid_chunk_config_before_touching_state() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        let bad_config = ChunkConfig {
            chunk_size: 16,
            overlap: 4,
            min_chunk_size: 32,
        };
        let result = ret.add_document("plenty of content to chunk here", &bad_config);
        assert!(matches!(result, Err(RagError::InvalidChunkConfig(_))));
        assert_eq!(ret.document_count(), 0);
        assert_eq!(ret.chunk_count(), 0);
    }

    struct FailingEmbedder {
        dim: usize,
        fail_on: &'static str,
    }
    impl Embedder for FailingEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
            if text.contains(self.fail_on) {
                return Err(RagError::EmbeddingFailed("intentional test failure".into()));
            }
            Ok(vec![1.0; self.dim])
        }
        fn embedding_dim(&self) -> usize {
            self.dim
        }
    }

    #[test]
    fn a_failing_chunk_leaves_no_partial_state_behind() {
        let embedder = FailingEmbedder {
            dim: 4,
            fail_on: "BOOM",
        };
        let mut ret = Retriever::new(embedder, RetrieverConfig::default());
        let chunk_cfg = ChunkConfig {
            chunk_size: 10,
            overlap: 0,
            min_chunk_size: 1,
        };
        // Constructed so the second window contains "BOOM" and fails.
        let text = "aaaaaaaaaa".to_string() + "BOOMBOOMBB";
        let result = ret.add_document(&text, &chunk_cfg);
        assert!(result.is_err());
        assert_eq!(
            ret.chunk_count(),
            0,
            "no chunk from the failing document may survive in the store"
        );
        assert_eq!(
            ret.document_count(),
            0,
            "doc_count must not advance for a document that failed"
        );
    }

    // ── RAG-EVAL-IMG-11: metadata reachable via the public API ──────────────

    #[test]
    fn add_document_with_metadata_is_queryable_via_metadata_filter() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        let mut meta = HashMap::new();
        meta.insert("tenant".to_string(), MetadataValue::from("acme"));
        ret.add_document_with_metadata(
            "Confidential quarterly numbers for Acme corp.",
            &ChunkConfig::default().with_min_chunk_size(1),
            meta,
        )
        .expect("index");

        let hits = ret
            .retrieve_filtered("quarterly", &MetadataFilter::eq("tenant", "acme"))
            .expect("retrieve_filtered");
        assert!(!hits.is_empty());
        let miss = ret
            .retrieve_filtered("quarterly", &MetadataFilter::eq("tenant", "other"))
            .expect("retrieve_filtered");
        assert!(miss.is_empty());
    }

    #[test]
    fn every_indexed_chunk_carries_doc_id_and_chunk_idx_metadata() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        ret.add_document(
            "some reasonably long content to make more than one chunk maybe not but anyway",
            &ChunkConfig::default().with_min_chunk_size(1),
        )
        .expect("index");
        let hits = ret
            .retrieve_filtered("content", &MetadataFilter::exists("doc_id"))
            .expect("retrieve_filtered");
        assert!(!hits.is_empty());
        assert_eq!(
            hits[0].chunk.metadata.get("doc_id"),
            Some(&MetadataValue::Int(0))
        );
    }

    // ── caller metadata must win over the auto-stamp ────────

    #[test]
    fn caller_supplied_doc_id_metadata_survives_the_automatic_stamp() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        let mut meta = HashMap::new();
        meta.insert(
            "doc_id".to_string(),
            MetadataValue::from("external-uuid-42"),
        );
        ret.add_document_with_metadata(
            "some content to index",
            &ChunkConfig::default().with_min_chunk_size(1),
            meta,
        )
        .expect("index");
        let hits = ret
            .retrieve_filtered("content", &MetadataFilter::exists("doc_id"))
            .expect("retrieve_filtered");
        assert_eq!(
            hits[0].chunk.metadata.get("doc_id"),
            Some(&MetadataValue::from("external-uuid-42")),
            "caller-supplied \"doc_id\" metadata must not be overwritten by the automatic stamp"
        );
    }

    // ── non-finite metadata must be rejected up front ──
    //
    // `add_document_with_metadata`/`add_chunks` are themselves new in this
    // diff (RAG-EVAL-IMG-11): before it, a `Retriever` caller had no way to
    // attach a `MetadataValue::Float` to a document at all, so this is a
    // silent-corruption path this diff would otherwise have opened, exactly
    // like `RetrieverConfig::min_score`'s -- `serde_json` serialises a
    // non-finite `f64` as JSON `null`, and `MetadataValue`'s untagged-enum
    // `Deserialize` then fails to match `null` against any variant.

    #[test]
    fn add_document_with_metadata_rejects_non_finite_float_before_touching_state() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut meta = HashMap::new();
            meta.insert("score".to_string(), MetadataValue::Float(bad));
            let err = ret.add_document_with_metadata(
                "some content to index",
                &ChunkConfig::default().with_min_chunk_size(1),
                meta,
            );
            assert!(
                matches!(err, Err(RagError::NonFinite)),
                "non-finite metadata Float {bad} must be rejected, got {err:?}"
            );
        }
        assert_eq!(
            ret.document_count(),
            0,
            "a rejected document must not be counted"
        );
        assert_eq!(
            ret.chunk_count(),
            0,
            "a rejected document must not be indexed"
        );
    }

    #[test]
    fn add_chunks_rejects_non_finite_float_metadata() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        let chunk = Chunk::new("hello there".to_string(), 0, 0, 0)
            .with_metadata("score", MetadataValue::Float(f64::NAN));
        let err = ret.add_chunks(vec![chunk]);
        assert!(matches!(err, Err(RagError::NonFinite)), "got {err:?}");
        assert_eq!(ret.chunk_count(), 0);
    }

    #[test]
    fn add_document_with_metadata_still_accepts_a_finite_float() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        let mut meta = HashMap::new();
        meta.insert("score".to_string(), MetadataValue::Float(0.75));
        ret.add_document_with_metadata(
            "some content to index",
            &ChunkConfig::default().with_min_chunk_size(1),
            meta,
        )
        .expect("a finite Float metadata value must still be accepted");
        assert_eq!(ret.chunk_count(), 1);
    }

    #[test]
    fn store_mut_allows_delete_after_indexing() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        ret.add_document(
            "hello there",
            &ChunkConfig::default().with_min_chunk_size(1),
        )
        .expect("index");
        let id = ret.store().entries()[0].id;
        assert!(ret.store_mut().delete(id).is_some());
        assert_eq!(ret.chunk_count(), 0);
    }

    #[test]
    fn add_chunks_bypasses_the_configured_chunker() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        let chunks = vec![
            Chunk::new("hand built one".into(), 7, 0, 0),
            Chunk::new("hand built two".into(), 7, 1, 0),
        ];
        let n = ret.add_chunks(chunks).expect("add_chunks");
        assert_eq!(n, 2);
        assert_eq!(ret.chunk_count(), 2);
        // add_chunks does not know how many *documents* it spans, so it
        // deliberately does not touch document_count().
        assert_eq!(ret.document_count(), 0);
    }

    // ── RAG-EVAL-IMG-12: RetrieverBuilder + Chunker ─────────────────────────

    #[test]
    fn retriever_builder_with_paragraph_chunker() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = RetrieverBuilder::new(emb)
            .with_chunker(Box::new(ParagraphChunker))
            .build();
        ret.add_document(
            "Para one.\n\nPara two.\n\nPara three.",
            &ChunkConfig::default(),
        )
        .expect("index");
        assert_eq!(ret.chunk_count(), 3);
    }

    #[test]
    fn retriever_with_chunker_matches_default_for_fixed_window() {
        let emb1 = IdentityEmbedder::new(16).expect("valid dim");
        let emb2 = IdentityEmbedder::new(16).expect("valid dim");
        let text = "a".repeat(600);
        let cfg = ChunkConfig::default();

        let mut default_ret = Retriever::new(emb1, RetrieverConfig::default());
        default_ret.add_document(&text, &cfg).expect("index");

        let mut explicit_ret = Retriever::new(emb2, RetrieverConfig::default())
            .with_chunker(Box::new(FixedWindowChunker(cfg.clone())));
        explicit_ret.add_document(&text, &cfg).expect("index");

        assert_eq!(default_ret.chunk_count(), explicit_ret.chunk_count());
    }

    // ── RAG-EVAL-IMG-21: degenerate (zero-norm) query rejection ─────────────

    struct ZeroEmbedder(usize);
    impl Embedder for ZeroEmbedder {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, RagError> {
            Ok(vec![0.0; self.0])
        }
        fn embedding_dim(&self) -> usize {
            self.0
        }
    }

    #[test]
    fn cosine_store_rejects_zero_norm_query() {
        let mut ret = Retriever::new(ZeroEmbedder(4), RetrieverConfig::default());
        // Insert with a *different*, non-degenerate embedder-independent
        // path: use add_chunks + a manual insert via store_mut so the
        // zero-embedder is only exercised by the *query*.
        ret.store_mut()
            .insert(vec![1.0, 0.0, 0.0, 0.0], Chunk::new("x".into(), 0, 0, 0))
            .expect("insert");
        let err = ret.retrieve("anything");
        // RAG-21: a typed `EmptyQueryVector`, not a string-stuffed
        // `EmbeddingFailed`, so a caller can match on the specific
        // condition instead of parsing the error's `Display` output.
        assert!(
            matches!(err, Err(RagError::EmptyQueryVector)),
            "a zero-norm query against a Cosine store must error with EmptyQueryVector, got {err:?}"
        );
    }

    // ── RAG-08 interaction: rerank must not clamp unbounded scores ──────────

    /// Test-only embedder that always returns the same fixed vector,
    /// regardless of the query text -- gives exact control over the raw
    /// DotProduct score a query would receive against a hand-inserted
    /// vector, which a real (hash- or corpus-derived) embedder cannot
    /// offer.
    struct FixedEmbedder(Vec<f32>);
    impl Embedder for FixedEmbedder {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, RagError> {
            Ok(self.0.clone())
        }
        fn embedding_dim(&self) -> usize {
            self.0.len()
        }
    }

    #[test]
    fn rerank_boost_does_not_collapse_unbounded_dot_product_scores() {
        // DotProduct has been unbounded since RAG-08's
        // silent L2 normalisation was removed (`vector_store.rs`'s
        // `requires_normalized_inputs` excludes DotProduct), so two real
        // entries can legitimately score > 1.0 *before* any rerank boost.
        // The old `rerank` clamped every *boosted* result to exactly 1.0,
        // which can rank it *below* an unboosted result that was never
        // clamped -- reproduced here with A (unboosted, raw score 2.0) and
        // B (boosted, raw score 1.95 -> 2.05), where the correct ranking
        // is B above A.
        let query_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let mut ret = Retriever::new(
            FixedEmbedder(query_vec),
            RetrieverConfig::default()
                .with_min_score(f32::MIN)
                .with_rerank(true),
        );
        *ret.store_mut() = VectorStore::new_with_distance(4, Distance::DotProduct);

        // Entry A: raw dot product 2.0, shares no tokens with the query --
        // rerank never touches its score.
        ret.store_mut()
            .insert(
                vec![2.0, 0.0, 0.0, 0.0],
                Chunk::new("no shared terms whatsoever".into(), 0, 0, 0),
            )
            .expect("insert");
        // Entry B: raw dot product 1.95, shares every query token -- boosted
        // by (5 * 0.02).min(0.1) = 0.1, landing at 2.05: correctly above A.
        ret.store_mut()
            .insert(
                vec![1.95, 0.0, 0.0, 0.0],
                Chunk::new("alpha beta gamma delta epsilon".into(), 0, 1, 0),
            )
            .expect("insert");

        let hits = ret
            .retrieve("alpha beta gamma delta epsilon")
            .expect("retrieve");
        assert_eq!(hits.len(), 2);
        assert!(
            hits[0].chunk.text.contains("alpha"),
            "the boosted entry (raw 1.95 + 0.1 = 2.05) must rank first, not be \
             clamped to 1.0 and buried below the unboosted 2.0 entry; got {:?}",
            hits.iter()
                .map(|h| (h.score, &h.chunk.text))
                .collect::<Vec<_>>()
        );
        assert!(
            hits[0].score > 1.0,
            "boosted DotProduct score must not be clamped to 1.0, got {}",
            hits[0].score
        );
        assert!(
            (hits[0].score - 2.05).abs() < 1e-5,
            "expected the boosted score to be exactly 1.95 + 0.1 = 2.05, got {}",
            hits[0].score
        );
    }

    #[test]
    fn euclidean_store_accepts_zero_norm_query() {
        let mut ret = Retriever::new(
            ZeroEmbedder(2),
            RetrieverConfig::default().with_min_score(f32::MIN),
        );
        *ret.store_mut() = VectorStore::new_with_distance(2, Distance::Euclidean);
        ret.store_mut()
            .insert(vec![3.0, 4.0], Chunk::new("point".into(), 0, 0, 0))
            .expect("insert");
        let hits = ret
            .retrieve("anything")
            .expect("euclidean must accept a zero query");
        assert_eq!(hits.len(), 1);
    }

    // ── RetrieverConfig serialization sanity (min_score default is finite) ──

    #[test]
    fn default_config_round_trips_through_json() {
        let cfg = RetrieverConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: RetrieverConfig = serde_json::from_str(&json).expect(
            "a default-constructed RetrieverConfig (min_score = f32::MIN) must round-trip \
             through JSON -- f32::NEG_INFINITY would not, since serde_json serialises it as \
             `null`",
        );
        assert_eq!(restored.top_k, cfg.top_k);
        assert_eq!(restored.min_score, cfg.min_score);
        assert_eq!(restored.rerank, cfg.rerank);
    }

    // ── Pre-existing behaviour (regression guards) ──────────────────────────

    #[test]
    fn add_and_retrieve_unchanged() {
        let emb = IdentityEmbedder::new(32).expect("valid dim");
        let config = RetrieverConfig {
            top_k: 3,
            ..Default::default()
        };
        let mut ret = Retriever::new(emb, config);
        let chunk_cfg = ChunkConfig {
            chunk_size: 128,
            overlap: 16,
            min_chunk_size: 10,
        };
        let n = ret
            .add_document(
                "Rust is a systems programming language focused on safety and performance.",
                &chunk_cfg,
            )
            .expect("add_document");
        assert!(n > 0);
        let results = ret.retrieve("Rust programming").expect("retrieve");
        assert!(!results.is_empty());
    }

    #[test]
    fn empty_document_and_query_errors_unchanged() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        assert!(matches!(
            ret.add_document("   ", &ChunkConfig::default()),
            Err(RagError::EmptyDocument)
        ));
        ret.add_document(
            "some content here for testing",
            &ChunkConfig {
                chunk_size: 50,
                overlap: 5,
                min_chunk_size: 5,
            },
        )
        .expect("add");
        assert!(matches!(ret.retrieve("  "), Err(RagError::EmptyQuery)));
    }

    #[test]
    fn no_documents_error_unchanged() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let ret = Retriever::new(emb, RetrieverConfig::default());
        assert!(matches!(
            ret.retrieve("anything"),
            Err(RagError::NoDocumentsIndexed)
        ));
    }
}
