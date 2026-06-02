//! Error type for the Pure-Rust FLUX.2 SMALL VAE decoder.

/// Result alias for the VAE decoder.
pub type VaeResult<T> = Result<T, VaeError>;

/// Errors that can occur while loading weights or running the VAE decode path.
#[derive(Debug, thiserror::Error)]
pub enum VaeError {
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
}
