//! Runtime error types.

use oxibonsai_core::error::BonsaiError;
use oxibonsai_kernels::error::KernelError;
use oxibonsai_model::error::ModelError;

/// Runtime error type.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("core error: {0}")]
    Core(#[from] BonsaiError),

    #[error("kernel error: {0}")]
    Kernel(#[from] KernelError),

    #[error("model error: {0}")]
    Model(#[from] ModelError),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("generation stopped: {reason}")]
    GenerationStopped { reason: String },

    #[error("server error: {0}")]
    Server(String),

    #[error("GGUF file not found: {path}")]
    FileNotFound { path: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("operation timed out: {operation} after {duration_ms}ms")]
    Timeout {
        /// Description of the operation that timed out.
        operation: String,
        /// Duration in milliseconds before timeout.
        duration_ms: u64,
    },

    #[error("circuit breaker is open, requests are being rejected")]
    CircuitOpen,

    #[error("capacity exhausted: {resource}")]
    CapacityExhausted {
        /// Description of the exhausted resource (e.g. "kv_cache", "memory").
        resource: String,
    },

    #[error("batch error: {} sub-errors", .0.len())]
    BatchError(Vec<RuntimeError>),

    /// A typed engine refusal: an operation the loaded
    /// model or the requested backend cannot perform — a dense-only
    /// operation on a hybrid (`qwen35`) engine, a rollback a recurrent state
    /// cannot do, an unavailable backend, a non-contiguous position, a stale
    /// sequence snapshot, ….
    ///
    /// Displays as `engine error: [CODE] message`, where `CODE` is the
    /// stable [`EngineError::error_code`](crate::engine_seam::EngineError::error_code)
    /// (the bracketed code is what log scrapers and the
    /// [`engine_error_code`](crate::engine_seam::engine_error_code) contract
    /// key off). Before this variant existed an engine refusal travelled as
    /// [`Self::Config`] with a `[CODE]`-prefixed message; it is not a
    /// configuration error, so it no longer says so.
    #[error("engine error: [{code}] {refusal}", code = .0.error_code(), refusal = .0)]
    Engine(#[from] crate::engine_seam::EngineError),
}

impl RuntimeError {
    /// Return a short, stable error code string for monitoring and alerting.
    ///
    /// An [`Self::Engine`] refusal reports its own stable
    /// [`EngineError`](crate::engine_seam::EngineError) code (e.g.
    /// `NOT_A_DENSE_MODEL`) — the code the refusal was always identified by
    /// — rather than a generic `ENGINE_ERROR`.
    pub fn error_code(&self) -> &str {
        match self {
            Self::Core(_) => "CORE_ERROR",
            Self::Kernel(_) => "KERNEL_ERROR",
            Self::Model(_) => "MODEL_ERROR",
            Self::Tokenizer(_) => "TOKENIZER_ERROR",
            Self::GenerationStopped { .. } => "GENERATION_STOPPED",
            Self::Server(_) => "SERVER_ERROR",
            Self::FileNotFound { .. } => "FILE_NOT_FOUND",
            Self::Io(_) => "IO_ERROR",
            Self::Config(_) => "CONFIG_ERROR",
            Self::Timeout { .. } => "TIMEOUT",
            Self::CircuitOpen => "CIRCUIT_OPEN",
            Self::CapacityExhausted { .. } => "CAPACITY_EXHAUSTED",
            Self::BatchError(_) => "BATCH_ERROR",
            Self::Engine(error) => error.error_code(),
        }
    }

    /// Whether this error is potentially recoverable by retrying.
    ///
    /// An [`Self::Engine`] refusal is never retryable: every
    /// [`EngineError`](crate::engine_seam::EngineError) is a deterministic
    /// property of the loaded model, the requested backend or the caller's
    /// request (the same request fails the same way again), exactly as the
    /// [`Self::Config`] encoding it replaces was treated.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Io(_) => true,
            Self::Timeout { .. } => true,
            Self::Server(_) => true,
            Self::CircuitOpen => true,
            Self::CapacityExhausted { .. } => true,
            Self::BatchError(errors) => errors.iter().any(|e| e.is_retryable()),
            _ => false,
        }
    }
}

/// Result type alias.
pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_seam::EngineError;

    /// `RuntimeError::Engine` keeps the bracketed `[CODE]` in its display,
    /// reports the refusal's own stable code from `error_code()`, is never
    /// retryable, and is classified exactly like the `Config` encoding it
    /// replaced (abort / permanent).
    #[test]
    fn the_engine_variant_carries_the_stable_code_everywhere() {
        let errors = EngineError::one_of_each_kind();
        assert_eq!(errors.len(), EngineError::ALL_CODES.len());
        for (error, expected_code) in errors.into_iter().zip(EngineError::ALL_CODES) {
            let message = error.to_string();
            let runtime = RuntimeError::from(error);
            assert!(matches!(runtime, RuntimeError::Engine(_)));
            assert_eq!(runtime.error_code(), expected_code);
            assert_eq!(
                runtime.to_string(),
                format!("engine error: [{expected_code}] {message}")
            );
            assert!(!runtime.is_retryable(), "{expected_code}");
            assert!(
                matches!(
                    crate::recovery::recovery_strategy_for(&runtime),
                    crate::recovery::RecoveryStrategy::Abort
                ),
                "{expected_code}"
            );
            assert_eq!(
                crate::recovery::classify_error(&runtime),
                crate::recovery::ErrorClass::Permanent,
                "{expected_code}"
            );
        }
    }

    /// The typed variant is what the conversion produces — never the old
    /// `Config("[CODE] …")` string encoding.
    #[test]
    fn an_engine_refusal_is_not_a_configuration_error() {
        let runtime: RuntimeError = EngineError::SnapshotMismatch {
            detail: "stale".into(),
        }
        .into();
        assert!(!matches!(runtime, RuntimeError::Config(_)));
        assert!(!runtime.to_string().starts_with("configuration error"));
        assert_eq!(runtime.error_code(), "SNAPSHOT_MISMATCH");
    }
}
