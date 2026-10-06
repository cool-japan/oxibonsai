//! The indexing stage of `POST /rag/index`: the heavy half of the request,
//! which runs on the blocking pool and stops when its request is abandoned.

use oxibonsai_rag::chunker::ChunkConfig;
use oxibonsai_rag::embedding::TfIdfEmbedder;
use oxibonsai_rag::error::RagError;
use oxibonsai_rag::pipeline::{RagConfig, RagPipeline};

use super::DEFAULT_MAX_FEATURES;
use crate::engine_control::CancellationToken;

/// Why [`build_index`] produced no pipeline.
pub(super) enum IndexError {
    /// The request was abandoned (its handler future was dropped).
    Cancelled,
    /// One document could not be indexed.
    Document { index: usize, source: RagError },
}

/// A freshly built pipeline and what indexing it reported.
pub(super) struct BuiltIndex {
    /// The pipeline, every document indexed.
    pub(super) pipeline: RagPipeline<TfIdfEmbedder>,
    /// The id of every indexed document (0-based, in request order).
    pub(super) document_ids: Vec<usize>,
    /// Chunks stored across all documents.
    pub(super) total_chunks: usize,
}

/// Fit a TF-IDF vocabulary on `documents` and index every one of them into a
/// fresh pipeline (so the vocabulary is always in-domain).
///
/// This is the heavy half of `POST /rag/index` — linear in the total size of
/// the documents — and runs on the blocking pool. `cancel` is checked before
/// each document, so an abandoned request stops after the document in
/// flight; a single document is indexed by one library call and cannot be
/// interrupted.
pub(super) fn build_index(
    documents: &[String],
    chunk_config: ChunkConfig,
    cancel: &CancellationToken,
) -> Result<BuiltIndex, IndexError> {
    if cancel.is_cancelled() {
        return Err(IndexError::Cancelled);
    }
    let doc_refs: Vec<&str> = documents.iter().map(String::as_str).collect();
    let embedder = TfIdfEmbedder::fit(&doc_refs, DEFAULT_MAX_FEATURES);
    let rag_config = RagConfig::default().with_chunk_config(chunk_config);
    let mut pipeline = RagPipeline::new(embedder, rag_config);

    let mut document_ids: Vec<usize> = Vec::with_capacity(documents.len());
    let mut total_chunks = 0usize;
    for (index, document) in documents.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(IndexError::Cancelled);
        }
        match pipeline.index_document(document) {
            Ok(chunk_count) => {
                document_ids.push(index);
                total_chunks += chunk_count;
            }
            Err(source) => return Err(IndexError::Document { index, source }),
        }
    }
    Ok(BuiltIndex {
        pipeline,
        document_ids,
        total_chunks,
    })
}
