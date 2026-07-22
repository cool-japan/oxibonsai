//! Retrieval pipeline: indexes documents and answers top-k queries.
//!
//! The [`Retriever`] is the core indexing and retrieval component.  It accepts
//! raw text documents, chunks them with the configured [`ChunkConfig`], embeds
//! each chunk with an [`Embedder`] backend, and stores the resulting vectors in
//! a [`VectorStore`].  At query time it embeds the query string and returns the
//! most similar chunks.

use tracing::{debug, info};

use crate::chunker::{chunk_document, ChunkConfig};
use crate::embedding::Embedder;
use crate::error::RagError;
use crate::metadata_filter::MetadataFilter;
use crate::vector_store::{SearchResult, VectorStore};

// ─────────────────────────────────────────────────────────────────────────────
// RetrieverConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration knobs for the [`Retriever`].
///
/// Marked `#[non_exhaustive]`; use [`RetrieverConfig::default`] plus the
/// `with_*` builders for forward-compatible construction.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RetrieverConfig {
    /// Maximum number of chunks to return per query.
    pub top_k: usize,
    /// Minimum cosine similarity score; chunks below this threshold are
    /// discarded even if they fall within the top-k.
    pub min_score: f32,
    /// Whether to apply a secondary heuristic re-ranking pass after the
    /// initial cosine-similarity retrieval.  Currently the re-ranking pass
    /// boosts chunks that contain at least one query token (exact-match term
    /// overlap), breaking cosine-similarity ties in a more lexical direction.
    pub rerank: bool,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.0,
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

    /// Set [`RetrieverConfig::min_score`] (builder).
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
// Retriever
// ─────────────────────────────────────────────────────────────────────────────

/// Indexes chunked documents and retrieves the most relevant chunks for a query.
pub struct Retriever<E: Embedder> {
    store: VectorStore,
    embedder: E,
    config: RetrieverConfig,
    /// Total number of distinct documents indexed (each call to
    /// `add_document` increments this, even if the document yields
    /// zero chunks because it is too short).
    doc_count: usize,
}

impl<E: Embedder> Retriever<E> {
    /// Create a new retriever with `embedder` and `config`.
    ///
    /// The vector store is initialised with the dimensionality reported by
    /// `embedder`.
    pub fn new(embedder: E, config: RetrieverConfig) -> Self {
        let dim = embedder.embedding_dim();
        Self {
            store: VectorStore::new(dim),
            embedder,
            config,
            doc_count: 0,
        }
    }

    /// Low-level constructor used by the persistence layer to reassemble a
    /// [`Retriever`] from its previously-persisted parts.
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
        }
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
        if text.trim().is_empty() {
            return Err(RagError::EmptyDocument);
        }
        chunk_config
            .validate()
            .map_err(RagError::InvalidChunkConfig)?;

        let doc_id = self.doc_count;
        self.doc_count += 1;

        let chunks = chunk_document(text, doc_id, chunk_config);
        let mut indexed = 0usize;

        for chunk in chunks {
            let vector = self.embedder.embed(&chunk.text)?;
            self.store.insert(vector, chunk)?;
            indexed += 1;
        }

        debug!(doc_id, indexed, "document indexed");
        Ok(indexed)
    }

    /// Index multiple documents, returning per-document chunk counts.
    ///
    /// Processing stops and the error is returned on the first failure.
    pub fn add_documents(
        &mut self,
        texts: &[&str],
        chunk_config: &ChunkConfig,
    ) -> Result<Vec<usize>, RagError> {
        let mut counts = Vec::with_capacity(texts.len());
        for text in texts {
            counts.push(self.add_document(text, chunk_config)?);
        }
        info!(
            documents = texts.len(),
            total_chunks = counts.iter().sum::<usize>(),
            "batch indexing complete"
        );
        Ok(counts)
    }

    /// Retrieve the top-k most relevant chunks for `query`.
    ///
    /// Returns [`RagError::EmptyQuery`] if `query` is blank, or
    /// [`RagError::NoDocumentsIndexed`] if the store is empty.
    pub fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, RagError> {
        if query.trim().is_empty() {
            return Err(RagError::EmptyQuery);
        }
        if self.store.is_empty() {
            return Err(RagError::NoDocumentsIndexed);
        }

        let query_vec = self.embedder.embed(query)?;
        let mut results =
            self.store
                .search_with_threshold(&query_vec, self.config.top_k, self.config.min_score);

        if self.config.rerank {
            results = rerank(results, query);
        }

        debug!(
            query_len = query.len(),
            hits = results.len(),
            "retrieval complete"
        );
        Ok(results)
    }

    /// Retrieve the top-k chunks that pass a [`MetadataFilter`].
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
        let mut results = self
            .store
            .search_filtered(&query_vec, self.config.top_k, filter)?;

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

    /// Borrow the embedder.
    pub fn embedder(&self) -> &E {
        &self.embedder
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-ranking helper
// ─────────────────────────────────────────────────────────────────────────────

/// Simple lexical re-ranker: bumps the score of chunks that contain at least
/// one query token.  This rewards exact-match overlap on top of the dense
/// cosine similarity, which helps short factual queries.
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
            // Small additive boost; capped to avoid overwhelming cosine score
            let boost = (overlap as f32 * 0.02).min(0.1);
            result.score = (result.score + boost).min(1.0);
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
