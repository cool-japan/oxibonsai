//! Error types for kernel operations.

use thiserror::Error;

/// Result type alias for kernel operations.
pub type KernelResult<T> = Result<T, KernelError>;

/// Errors that can occur during 1-bit kernel operations.
///
/// # Named vs unnamed length errors (K-02)
///
/// The length-contract variants come in two shapes. The unnamed
/// [`KernelError::BufferTooSmall`] / [`KernelError::DimensionMismatch`] report
/// only the numbers, which is all the original `debug_assert!`-replacement
/// sites had; a caller that passes four slices to one kernel then cannot tell
/// *which* one it got wrong. The named
/// [`KernelError::NamedBufferTooSmall`] / [`KernelError::NamedDimensionMismatch`]
/// additionally carry the buffer's role (`"output"`, `"weight"`, `"cos_row"`,
/// …), and [`KernelError::buffer_name`] reads it back uniformly.
///
/// **New code should construct the named form**, via
/// [`KernelError::buffer_too_small`] and [`KernelError::dimension_mismatch`].
/// The unnamed variants are kept because roughly 120 existing construction
/// sites across this crate's SIMD, GEMV, GEMM and packing modules still use
/// them; both forms report the same [`KernelError::error_code`], so a
/// monitoring consumer sees one stable code per condition regardless of which
/// shape produced it, and the remaining sites can migrate one file at a time
/// without a flag day.
#[derive(Error, Debug)]
pub enum KernelError {
    /// Matrix/vector dimension mismatch.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Matrix/vector dimension mismatch, naming the offending operand.
    #[error("dimension mismatch for '{name}': expected {expected}, got {got}")]
    NamedDimensionMismatch {
        /// Static role of the operand whose length is wrong.
        name: &'static str,
        /// Length the kernel requires.
        expected: usize,
        /// Length the caller supplied.
        got: usize,
    },

    /// Output buffer is too small.
    #[error("output buffer too small: need {needed} elements, have {available}")]
    BufferTooSmall { needed: usize, available: usize },

    /// A buffer is too small, naming which one.
    #[error("buffer '{name}' too small: need {needed} elements, have {available}")]
    NamedBufferTooSmall {
        /// Static role of the undersized buffer (e.g. `"output"`, `"weight"`).
        name: &'static str,
        /// Elements the kernel requires.
        needed: usize,
        /// Elements the caller supplied.
        available: usize,
    },

    /// Number of elements is not a multiple of the block size.
    #[error("{count} elements is not divisible by block size {block_size}")]
    NotBlockAligned { count: usize, block_size: usize },

    /// Underlying core error.
    #[error("core error: {0}")]
    Core(#[from] oxibonsai_core::error::BonsaiError),

    /// Operation is not supported by this kernel tier.
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// A GPU backend error propagated to the kernel layer.
    #[error("GPU error: {0}")]
    GpuError(String),
}

impl KernelError {
    /// Build a [`KernelError::NamedBufferTooSmall`].
    ///
    /// Preferred over constructing [`KernelError::BufferTooSmall`] directly:
    /// `name` is what lets a caller with several buffers in flight tell which
    /// one it sized wrong.
    #[must_use]
    pub const fn buffer_too_small(name: &'static str, needed: usize, available: usize) -> Self {
        Self::NamedBufferTooSmall {
            name,
            needed,
            available,
        }
    }

    /// Build a [`KernelError::NamedDimensionMismatch`].
    #[must_use]
    pub const fn dimension_mismatch(name: &'static str, expected: usize, got: usize) -> Self {
        Self::NamedDimensionMismatch {
            name,
            expected,
            got,
        }
    }

    /// The offending buffer's name, when this error carries one.
    ///
    /// `None` for the unnamed legacy variants and for every error that is not
    /// a length-contract violation, so a caller can log it unconditionally.
    #[must_use]
    pub const fn buffer_name(&self) -> Option<&'static str> {
        match self {
            Self::NamedBufferTooSmall { name, .. } | Self::NamedDimensionMismatch { name, .. } => {
                Some(name)
            }
            _ => None,
        }
    }

    /// Return a short, stable error code string for monitoring and alerting.
    ///
    /// The named and unnamed forms of one condition deliberately share a
    /// code: they describe the same failure, and a dashboard keyed on this
    /// string must not split when a call site migrates to the named variant.
    pub fn error_code(&self) -> &str {
        match self {
            Self::DimensionMismatch { .. } | Self::NamedDimensionMismatch { .. } => {
                "DIMENSION_MISMATCH"
            }
            Self::BufferTooSmall { .. } | Self::NamedBufferTooSmall { .. } => "BUFFER_TOO_SMALL",
            Self::NotBlockAligned { .. } => "NOT_BLOCK_ALIGNED",
            Self::Core(_) => "CORE_ERROR",
            Self::UnsupportedOperation(_) => "UNSUPPORTED_OPERATION",
            Self::GpuError(_) => "GPU_ERROR",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_buffer_too_small_reports_the_buffer_name() {
        let err = KernelError::buffer_too_small("output", 128, 64);
        assert_eq!(err.buffer_name(), Some("output"));
        assert_eq!(
            err.to_string(),
            "buffer 'output' too small: need 128 elements, have 64"
        );
    }

    #[test]
    fn named_dimension_mismatch_reports_the_operand_name() {
        let err = KernelError::dimension_mismatch("weight", 4096, 2048);
        assert_eq!(err.buffer_name(), Some("weight"));
        assert_eq!(
            err.to_string(),
            "dimension mismatch for 'weight': expected 4096, got 2048"
        );
    }

    #[test]
    fn unnamed_variants_have_no_buffer_name() {
        assert_eq!(
            KernelError::BufferTooSmall {
                needed: 8,
                available: 4
            }
            .buffer_name(),
            None
        );
        assert_eq!(
            KernelError::DimensionMismatch {
                expected: 8,
                got: 4
            }
            .buffer_name(),
            None
        );
        assert_eq!(
            KernelError::NotBlockAligned {
                count: 5,
                block_size: 4
            }
            .buffer_name(),
            None
        );
    }

    /// The named form must not change the monitoring code a dashboard keys
    /// on — otherwise migrating a call site would silently split a metric.
    #[test]
    fn named_and_unnamed_forms_share_one_error_code() {
        assert_eq!(
            KernelError::buffer_too_small("output", 128, 64).error_code(),
            KernelError::BufferTooSmall {
                needed: 128,
                available: 64
            }
            .error_code()
        );
        assert_eq!(
            KernelError::buffer_too_small("output", 128, 64).error_code(),
            "BUFFER_TOO_SMALL"
        );
        assert_eq!(
            KernelError::dimension_mismatch("q", 8, 4).error_code(),
            KernelError::DimensionMismatch {
                expected: 8,
                got: 4
            }
            .error_code()
        );
        assert_eq!(
            KernelError::dimension_mismatch("q", 8, 4).error_code(),
            "DIMENSION_MISMATCH"
        );
    }

    /// Every variant has a code; none falls through to a placeholder.
    #[test]
    fn every_variant_has_a_non_empty_error_code() {
        let errors = [
            KernelError::DimensionMismatch {
                expected: 1,
                got: 2,
            },
            KernelError::dimension_mismatch("a", 1, 2),
            KernelError::BufferTooSmall {
                needed: 1,
                available: 0,
            },
            KernelError::buffer_too_small("a", 1, 0),
            KernelError::NotBlockAligned {
                count: 3,
                block_size: 2,
            },
            KernelError::UnsupportedOperation("x".into()),
            KernelError::GpuError("x".into()),
        ];
        for e in &errors {
            assert!(!e.error_code().is_empty(), "{e}");
            assert!(!e.to_string().is_empty(), "{e:?}");
        }
    }
}
