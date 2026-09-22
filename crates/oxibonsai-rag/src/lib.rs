//! # oxibonsai-rag
//!
//! Pure Rust Retrieval-Augmented Generation (RAG) pipeline for OxiBonsai.
//!
//! This crate provides a self-contained, dependency-light RAG stack:
//!
//! - **[`vector_store`]** — in-memory flat index with cosine similarity search.
//! - **[`chunker`]** — split documents into overlapping character windows,
//!   sentence groups, or paragraphs.
//! - **[`embedding`]** — [`Embedder`] trait plus two built-in backends:
//!   [`IdentityEmbedder`] (deterministic hash, for tests) and
//!   [`TfIdfEmbedder`] (bag-of-words TF-IDF, no external deps).
//! - **[`retriever`]** — top-k chunk retrieval given a query string.
//! - **[`pipeline`]** — composes retrieval + prompt building for inference.
//!
//! ## Quick Start
//!
//! ```rust
//! use oxibonsai_rag::embedding::IdentityEmbedder;
//! use oxibonsai_rag::pipeline::{RagConfig, RagPipeline};
//!
//! let embedder = IdentityEmbedder::new(64).expect("valid dim");
//! let mut pipeline = RagPipeline::new(embedder, RagConfig::default());
//!
//! pipeline.index_document("Rust is a systems programming language.").expect("failed to index document");
//! let prompt = pipeline.build_prompt("What is Rust?").expect("failed to build prompt");
//! assert!(prompt.contains("Question: What is Rust?"));
//! ```

pub mod advanced_chunker;
pub mod chunker;
pub mod code_chunker;
pub mod distance;
pub mod embedding;
pub mod error;
pub mod metadata_filter;
pub mod persistence;
pub mod pipeline;
pub mod retriever;
pub mod semantic_chunker;
pub mod vector_store;

#[cfg(test)]
mod tests;

// ── Top-level re-exports ──────────────────────────────────────────────────────

pub use advanced_chunker::{
    ChunkStrategy, ChunkerRegistry, MarkdownChunker, RecursiveCharSplitter, RichChunk,
    SentenceChunker, SlidingWindowChunker,
};
pub use chunker::{chunk_by_paragraphs, chunk_by_sentences, chunk_document, Chunk, ChunkConfig};
pub use code_chunker::{CodeChunker, Language};
pub use distance::Distance;
pub use embedding::{Embedder, IdentityEmbedder, TfIdfEmbedder};
pub use error::RagError;
pub use metadata_filter::{MetadataFilter, MetadataValue};
pub use persistence::{IndexSnapshot, RetrieverSnapshot, SCHEMA_VERSION};
pub use pipeline::PromptParts;
pub use pipeline::{PipelineStats, RagConfig, RagPipeline};
pub use retriever::{Retriever, RetrieverConfig};
pub use semantic_chunker::SemanticChunker;
pub use vector_store::{cosine_similarity, dot_product, l2_normalize, SearchResult, VectorStore};
