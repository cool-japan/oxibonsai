//! Error type for the `oxibonsai-image` DiT loader.

use oxibonsai_core::error::BonsaiError;

/// Result alias for the DiT loader.
pub type DitResult<T> = Result<T, DitError>;

/// Errors that can occur while loading or interrogating a FLUX.2 DiT
/// (`bonsai-image`) GGUF file.
#[derive(Debug, thiserror::Error)]
pub enum DitError {
    /// An underlying I/O error while opening or mapping the GGUF file.
    #[error("I/O error for {path}: {source}")]
    Io {
        /// Path that triggered the error.
        path: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// An error from the OxiBonsai core GGUF reader (parse, missing tensor,
    /// unsupported quant type, byte-slice validation, …).
    #[error("GGUF error: {0}")]
    Gguf(#[from] BonsaiError),

    /// The file's `general.architecture` is not `"bonsai-image"`.
    #[error("unexpected architecture {found:?} (expected {expected:?})")]
    WrongArchitecture {
        /// Architecture string actually found in the GGUF metadata.
        found: String,
        /// Architecture string that was expected.
        expected: String,
    },

    /// A required `bonsai-image.*` metadata key was missing.
    #[error("missing required metadata key: {key}")]
    MissingMetadata {
        /// The metadata key that was missing.
        key: String,
    },

    /// A metadata value had an unexpected type or shape.
    #[error("invalid metadata for {key}: {reason}")]
    InvalidMetadata {
        /// The metadata key whose value was invalid.
        key: String,
        /// Human-readable reason the value was rejected.
        reason: String,
    },

    /// A tensor was found but its stored GGUF type did not match the storage
    /// convention for its name space (quantized → `TQ2_0_g128`, plain → `BF16`).
    #[error("tensor {name} has type {found}, expected {expected}")]
    WrongTensorType {
        /// Tensor name.
        name: String,
        /// GGUF type actually found.
        found: String,
        /// GGUF type that was expected.
        expected: String,
    },

    /// A tensor's stored dimensionality was unexpected (e.g. a quantized linear
    /// that was not 2-D).
    #[error("tensor {name} has {found} dims, expected {expected}")]
    WrongRank {
        /// Tensor name.
        name: String,
        /// Number of dimensions actually found.
        found: usize,
        /// Number of dimensions that was expected.
        expected: usize,
    },

    /// A forward-pass shape invariant was violated (e.g. a weight whose
    /// `(out, in)` did not match the expected feature widths).
    #[error("forward-pass shape error: {0}")]
    Shape(String),

    /// An error from the OxiBonsai kernels crate (e.g. the ternary GEMM
    /// rejecting an unaligned contraction dimension).
    #[error("kernel error: {0}")]
    Kernel(String),
}

impl From<oxibonsai_kernels::KernelError> for DitError {
    fn from(source: oxibonsai_kernels::KernelError) -> Self {
        DitError::Kernel(source.to_string())
    }
}
