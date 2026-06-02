//! Error type and GPU-resident weight handle for the Metal graph dispatch engine.

use metal::Buffer;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════════════

/// Errors raised by the Metal graph dispatch engine.
#[derive(Debug)]
pub enum MetalGraphError {
    /// No Metal-capable GPU device was found on the system.
    DeviceNotFound,
    /// MSL shader compilation failed.
    CompilationFailed(String),
    /// A GPU buffer could not be allocated.
    BufferCreationFailed,
    /// An encoding operation failed (pipeline not found, etc.).
    EncodingFailed(String),
    /// A command buffer execution failed or timed out.
    ExecutionFailed(String),
    /// Supplied dimensions or buffer lengths are inconsistent (e.g. `k` not a
    /// multiple of 128, or a slice length mismatching `m*k` / `m*n_rows`).
    InvalidDimensions(String),
}

impl fmt::Display for MetalGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound => write!(f, "no Metal-capable GPU device found"),
            Self::CompilationFailed(msg) => write!(f, "MSL compilation failed: {msg}"),
            Self::BufferCreationFailed => write!(f, "Metal buffer allocation failed"),
            Self::EncodingFailed(msg) => write!(f, "Metal encoding failed: {msg}"),
            Self::ExecutionFailed(msg) => write!(f, "Metal execution failed: {msg}"),
            Self::InvalidDimensions(msg) => write!(f, "Metal invalid dimensions: {msg}"),
        }
    }
}

impl std::error::Error for MetalGraphError {}

// ═══════════════════════════════════════════════════════════════════════════
// Weight handle
// ═══════════════════════════════════════════════════════════════════════════

/// Opaque handle to a weight buffer already resident on the GPU.
///
/// Stores the raw `metal::Buffer` directly so the graph can bind it
/// without going through any abstraction layer.
pub struct MetalWeightHandle {
    /// Raw Metal buffer containing packed weight data.
    pub(crate) buffer: Buffer,
    /// Size in bytes.
    pub(crate) byte_len: usize,
}

impl MetalWeightHandle {
    /// Size of the weight data in bytes.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

impl fmt::Debug for MetalWeightHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalWeightHandle")
            .field("byte_len", &self.byte_len)
            .finish()
    }
}
