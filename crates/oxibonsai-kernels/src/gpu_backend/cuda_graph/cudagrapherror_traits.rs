//! # CudaGraphError - Trait Implementations
//!
//! This module contains trait implementations for `CudaGraphError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::CudaGraphError;

impl fmt::Display for CudaGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound(s) => write!(f, "no CUDA device: {s}"),
            Self::CompilationFailed(s) => write!(f, "PTX compilation failed: {s}"),
            Self::DriverError(s) => write!(f, "CUDA driver error: {s}"),
            Self::WeightNotFound(id) => write!(f, "weight handle {id} not in cache"),
            Self::WeightLayoutError(s) => write!(f, "weight layout error: {s}"),
            Self::InvalidDimensions(s) => write!(f, "invalid dimensions: {s}"),
            Self::LockPoisoned => write!(f, "mutex lock poisoned"),
        }
    }
}

impl std::error::Error for CudaGraphError {}
