//! Error type for the Pure-Rust Qwen3-4B text encoder.

/// Result alias for the text-encoder forward / weight loading.
pub type TeResult<T> = Result<T, TeError>;

/// Errors that can occur while loading the text-encoder weights or running the
/// Qwen3 encoder forward pass.
#[derive(Debug, thiserror::Error)]
pub enum TeError {
    /// An I/O error while reading an exported weight `.npy` file.
    #[error("I/O error for {path}: {source}")]
    Io {
        /// Path that triggered the error.
        path: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// A weight `.npy` file was malformed (bad magic, non-f32 dtype, …).
    #[error("invalid npy {path}: {reason}")]
    Npy {
        /// Path to the offending file.
        path: String,
        /// Human-readable reason the file was rejected.
        reason: String,
    },

    /// A required weight tensor was missing from the weights directory.
    #[error("missing weight tensor: {name}")]
    MissingWeight {
        /// Dotted tensor name that was not found.
        name: String,
    },

    /// A tensor had an unexpected shape for the operation that consumed it.
    #[error("shape error: {0}")]
    Shape(String),

    /// A tokenizer asset (`tokenizer.json`) was malformed or missing a field.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
}
