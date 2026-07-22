//! Error types for the OxiBonsai RAG pipeline.

use std::io;

use thiserror::Error;

/// Errors that can occur in the RAG pipeline.
///
/// The enum is marked `#[non_exhaustive]` so that new variants can be added
/// in patch releases without breaking downstream pattern-matching.  Callers
/// should always include a wildcard (`_ => ...`) arm when exhaustively
/// matching on [`RagError`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RagError {
    /// Input document was empty (no text to index).
    #[error("document is empty")]
    EmptyDocument,

    /// Query string was empty.
    #[error("query is empty")]
    EmptyQuery,

    /// Retrieval was attempted before any documents were indexed.
    #[error("no documents have been indexed yet")]
    NoDocumentsIndexed,

    /// The embedding backend failed to produce a vector.
    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),

    /// A vector was inserted with a dimensionality that does not match the
    /// store's dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimensionality the store was configured with.
        expected: usize,
        /// The dimensionality of the offending vector.
        got: usize,
    },

    /// A persistence operation (save, load, schema-version check) failed.
    #[error("persistence error: {0}")]
    Persistence(String),

    /// A vector or scalar input contained `NaN` or `±∞`.  Distance metrics
    /// reject non-finite inputs eagerly rather than silently propagating
    /// poison values through downstream arithmetic.
    #[error("non-finite value in input (NaN or infinity)")]
    NonFinite,

    /// A metadata filter was ill-formed (e.g. empty key, empty `In` list).
    #[error("invalid metadata filter: {0}")]
    InvalidFilter(String),

    /// The chunker configuration was ill-formed (e.g. `min_chunk_size` larger
    /// than `chunk_size`, which would silently discard every produced chunk).
    #[error("invalid chunk config: {0}")]
    InvalidChunkConfig(String),

    /// I/O error (wraps [`std::io::Error`]).
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}
