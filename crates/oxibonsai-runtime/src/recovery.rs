//! Error recovery strategies for production resilience.
//!
//! Provides retry logic with exponential backoff, error classification,
//! and memory-aware batch sizing.

use std::time::Duration;

use crate::error::{RuntimeError, RuntimeResult};

/// Recovery strategy for different error types.
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation with backoff.
    Retry {
        /// Maximum number of retry attempts.
        max_attempts: usize,
        /// Base delay between retries (doubled each attempt).
        delay: Duration,
    },
    /// Fall back to an alternative approach.
    Fallback(String),
    /// Abort — the error is not recoverable.
    Abort,
}

impl std::fmt::Display for RecoveryStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Retry {
                max_attempts,
                delay,
            } => write!(
                f,
                "retry (max {} attempts, base delay {:?})",
                max_attempts, delay
            ),
            Self::Fallback(desc) => write!(f, "fallback: {}", desc),
            Self::Abort => write!(f, "abort"),
        }
    }
}

/// Determine the appropriate recovery strategy for a given error.
pub fn recovery_strategy_for(error: &RuntimeError) -> RecoveryStrategy {
    match error {
        // IO errors are often transient
        RuntimeError::Io(_) => RecoveryStrategy::Retry {
            max_attempts: 3,
            delay: Duration::from_millis(100),
        },
        // Timeout errors should be retried with longer timeout
        RuntimeError::Timeout { .. } => RecoveryStrategy::Retry {
            max_attempts: 2,
            delay: Duration::from_millis(500),
        },
        // Capacity errors: wait and retry
        RuntimeError::CapacityExhausted { .. } => RecoveryStrategy::Retry {
            max_attempts: 3,
            delay: Duration::from_millis(200),
        },
        // Circuit open: wait for recovery
        RuntimeError::CircuitOpen => RecoveryStrategy::Retry {
            max_attempts: 1,
            delay: Duration::from_secs(5),
        },
        // Config errors are permanent
        RuntimeError::Config(_) => RecoveryStrategy::Abort,
        // Typed engine refusals are deterministic (same as their former
        // `Config("[CODE] ..")` encoding)
        RuntimeError::Engine(_) => RecoveryStrategy::Abort,
        // File not found is permanent
        RuntimeError::FileNotFound { .. } => RecoveryStrategy::Abort,
        // Tokenizer errors may benefit from fallback
        RuntimeError::Tokenizer(_) => RecoveryStrategy::Fallback("use raw token IDs".to_string()),
        // Generation stopped is not really an error
        RuntimeError::GenerationStopped { .. } => RecoveryStrategy::Abort,
        // Server errors may be transient
        RuntimeError::Server(_) => RecoveryStrategy::Retry {
            max_attempts: 2,
            delay: Duration::from_millis(200),
        },
        // Core/kernel/model errors are generally permanent
        RuntimeError::Core(_) => RecoveryStrategy::Abort,
        RuntimeError::Kernel(_) => RecoveryStrategy::Abort,
        RuntimeError::Model(_) => RecoveryStrategy::Abort,
        // Batch errors: check individual errors
        RuntimeError::BatchError(_) => RecoveryStrategy::Retry {
            max_attempts: 1,
            delay: Duration::from_millis(100),
        },
    }
}

/// Retry a fallible operation with exponential backoff.
///
/// Calls `f` up to `max_attempts` times. If `f` succeeds, returns the result
/// immediately. If all attempts fail, returns the last error.
///
/// The delay between attempts doubles each time, starting from `base_delay`.
pub fn retry_with_backoff<F, T>(
    max_attempts: usize,
    base_delay: Duration,
    mut f: F,
) -> RuntimeResult<T>
where
    F: FnMut() -> RuntimeResult<T>,
{
    let attempts = max_attempts.max(1);
    let mut last_error = None;
    let mut delay = base_delay;

    for attempt in 0..attempts {
        match f() {
            Ok(val) => return Ok(val),
            Err(e) => {
                tracing::debug!(
                    attempt = attempt + 1,
                    max_attempts = attempts,
                    error = %e,
                    "retry attempt failed"
                );
                last_error = Some(e);

                if attempt + 1 < attempts {
                    std::thread::sleep(delay);
                    delay = delay.saturating_mul(2);
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        RuntimeError::Config("retry_with_backoff called with zero attempts".to_string())
    }))
}

/// Execute a closure with a synchronous timeout.
///
/// Spawns the closure on a separate thread and waits for it to complete
/// within the specified duration. Returns a timeout error if it takes too long.
pub fn with_timeout<F, T>(duration: Duration, f: F) -> RuntimeResult<T>
where
    F: FnOnce() -> RuntimeResult<T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = f();
        let _ = tx.send(result);
    });

    rx.recv_timeout(duration).unwrap_or_else(|e| match e {
        std::sync::mpsc::RecvTimeoutError::Timeout => Err(RuntimeError::Timeout {
            operation: "with_timeout".to_string(),
            duration_ms: duration.as_millis() as u64,
        }),
        std::sync::mpsc::RecvTimeoutError::Disconnected => Err(RuntimeError::Server(
            "timeout worker thread panicked".to_string(),
        )),
    })
}

/// Calculate recommended batch size based on available memory.
///
/// Returns the largest batch size that fits within available memory,
/// capped at `max_batch`.
pub fn recommended_batch_size(
    available_memory_bytes: u64,
    per_request_memory_bytes: u64,
    max_batch: usize,
) -> usize {
    if per_request_memory_bytes == 0 {
        return max_batch;
    }

    let fits = (available_memory_bytes / per_request_memory_bytes) as usize;
    fits.min(max_batch).max(1)
}

/// Classification of errors for monitoring and alerting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Retry may help (timeout, resource busy, transient IO).
    Transient,
    /// Won't recover without user intervention (invalid input, model error).
    Permanent,
    /// Memory/capacity related — may recover if load decreases.
    ResourceExhaustion,
}

impl std::fmt::Display for ErrorClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transient => write!(f, "transient"),
            Self::Permanent => write!(f, "permanent"),
            Self::ResourceExhaustion => write!(f, "resource_exhaustion"),
        }
    }
}

/// Classify an error for monitoring purposes.
pub fn classify_error(error: &RuntimeError) -> ErrorClass {
    match error {
        RuntimeError::Io(_) => ErrorClass::Transient,
        RuntimeError::Timeout { .. } => ErrorClass::Transient,
        RuntimeError::Server(_) => ErrorClass::Transient,
        RuntimeError::CircuitOpen => ErrorClass::Transient,
        RuntimeError::CapacityExhausted { .. } => ErrorClass::ResourceExhaustion,
        RuntimeError::Config(_) => ErrorClass::Permanent,
        RuntimeError::Engine(_) => ErrorClass::Permanent,
        RuntimeError::FileNotFound { .. } => ErrorClass::Permanent,
        RuntimeError::Tokenizer(_) => ErrorClass::Permanent,
        RuntimeError::GenerationStopped { .. } => ErrorClass::Permanent,
        RuntimeError::Core(_) => ErrorClass::Permanent,
        RuntimeError::Kernel(_) => ErrorClass::Permanent,
        RuntimeError::Model(_) => ErrorClass::Permanent,
        RuntimeError::BatchError(errors) => {
            // If any error is resource exhaustion, classify as such
            for e in errors {
                if classify_error(e) == ErrorClass::ResourceExhaustion {
                    return ErrorClass::ResourceExhaustion;
                }
            }
            // Otherwise check for transient
            for e in errors {
                if classify_error(e) == ErrorClass::Transient {
                    return ErrorClass::Transient;
                }
            }
            ErrorClass::Permanent
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_strategy_io_error() {
        let error = RuntimeError::Io(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "reset",
        ));
        let strategy = recovery_strategy_for(&error);
        matches!(strategy, RecoveryStrategy::Retry { .. });
    }

    #[test]
    fn recovery_strategy_config_error() {
        let error = RuntimeError::Config("bad config".to_string());
        let strategy = recovery_strategy_for(&error);
        assert!(matches!(strategy, RecoveryStrategy::Abort));
    }

    #[test]
    fn recovery_strategy_timeout() {
        let error = RuntimeError::Timeout {
            operation: "test".to_string(),
            duration_ms: 1000,
        };
        let strategy = recovery_strategy_for(&error);
        assert!(matches!(strategy, RecoveryStrategy::Retry { .. }));
    }

    #[test]
    fn recovery_strategy_capacity() {
        let error = RuntimeError::CapacityExhausted {
            resource: "kv_cache".to_string(),
        };
        let strategy = recovery_strategy_for(&error);
        assert!(matches!(strategy, RecoveryStrategy::Retry { .. }));
    }

    #[test]
    fn recovery_strategy_tokenizer() {
        let error = RuntimeError::Tokenizer("bad token".to_string());
        let strategy = recovery_strategy_for(&error);
        assert!(matches!(strategy, RecoveryStrategy::Fallback(_)));
    }

    #[test]
    fn recovery_strategy_display() {
        let retry = RecoveryStrategy::Retry {
            max_attempts: 3,
            delay: Duration::from_millis(100),
        };
        assert!(format!("{}", retry).contains("retry"));

        let fallback = RecoveryStrategy::Fallback("alt".to_string());
        assert!(format!("{}", fallback).contains("fallback"));

        assert_eq!(format!("{}", RecoveryStrategy::Abort), "abort");
    }

    #[test]
    fn retry_succeeds_first_attempt() {
        let mut count = 0;
        let result = retry_with_backoff(3, Duration::from_millis(1), || {
            count += 1;
            Ok(42)
        });
        assert_eq!(result.expect("should succeed"), 42);
        assert_eq!(count, 1);
    }

    #[test]
    fn retry_succeeds_second_attempt() {
        let mut count = 0;
        let result = retry_with_backoff(3, Duration::from_millis(1), || {
            count += 1;
            if count < 2 {
                Err(RuntimeError::Server("transient".to_string()))
            } else {
                Ok(42)
            }
        });
        assert_eq!(result.expect("should succeed"), 42);
        assert_eq!(count, 2);
    }

    #[test]
    fn retry_exhausts_attempts() {
        let mut count = 0;
        let result: RuntimeResult<i32> = retry_with_backoff(3, Duration::from_millis(1), || {
            count += 1;
            Err(RuntimeError::Server("fail".to_string()))
        });
        assert!(result.is_err());
        assert_eq!(count, 3);
    }

    #[test]
    fn retry_zero_attempts_treated_as_one() {
        let mut count = 0;
        let result: RuntimeResult<i32> = retry_with_backoff(0, Duration::from_millis(1), || {
            count += 1;
            Ok(99)
        });
        assert_eq!(result.expect("should succeed"), 99);
        assert_eq!(count, 1);
    }

    #[test]
    fn with_timeout_success() {
        let result = with_timeout(Duration::from_secs(5), || Ok(42));
        assert_eq!(result.expect("should succeed"), 42);
    }

    #[test]
    fn with_timeout_expires() {
        let result: RuntimeResult<i32> = with_timeout(Duration::from_millis(10), || {
            std::thread::sleep(Duration::from_secs(5));
            Ok(42)
        });
        assert!(result.is_err());
        let err = result.expect_err("should timeout");
        assert!(err.to_string().contains("timeout") || err.to_string().contains("Timeout"));
    }

    #[test]
    fn batch_size_normal() {
        assert_eq!(recommended_batch_size(1_000_000, 100_000, 16), 10);
    }

    #[test]
    fn batch_size_capped_at_max() {
        assert_eq!(recommended_batch_size(10_000_000, 100_000, 8), 8);
    }

    #[test]
    fn batch_size_minimum_one() {
        assert_eq!(recommended_batch_size(1, 1_000_000, 16), 1);
    }

    #[test]
    fn batch_size_zero_per_request() {
        assert_eq!(recommended_batch_size(1_000_000, 0, 16), 16);
    }

    #[test]
    fn classify_io_error() {
        let error = RuntimeError::Io(std::io::Error::other("test"));
        assert_eq!(classify_error(&error), ErrorClass::Transient);
    }

    #[test]
    fn classify_config_error() {
        let error = RuntimeError::Config("bad".to_string());
        assert_eq!(classify_error(&error), ErrorClass::Permanent);
    }

    #[test]
    fn classify_capacity_error() {
        let error = RuntimeError::CapacityExhausted {
            resource: "mem".to_string(),
        };
        assert_eq!(classify_error(&error), ErrorClass::ResourceExhaustion);
    }

    #[test]
    fn classify_timeout_error() {
        let error = RuntimeError::Timeout {
            operation: "gen".to_string(),
            duration_ms: 1000,
        };
        assert_eq!(classify_error(&error), ErrorClass::Transient);
    }

    #[test]
    fn classify_batch_error_resource() {
        let error = RuntimeError::BatchError(vec![RuntimeError::CapacityExhausted {
            resource: "mem".to_string(),
        }]);
        assert_eq!(classify_error(&error), ErrorClass::ResourceExhaustion);
    }

    #[test]
    fn classify_batch_error_transient() {
        let error = RuntimeError::BatchError(vec![RuntimeError::Server("err".to_string())]);
        assert_eq!(classify_error(&error), ErrorClass::Transient);
    }

    #[test]
    fn classify_batch_error_permanent() {
        let error = RuntimeError::BatchError(vec![RuntimeError::Config("bad".to_string())]);
        assert_eq!(classify_error(&error), ErrorClass::Permanent);
    }

    #[test]
    fn error_class_display() {
        assert_eq!(format!("{}", ErrorClass::Transient), "transient");
        assert_eq!(format!("{}", ErrorClass::Permanent), "permanent");
        assert_eq!(
            format!("{}", ErrorClass::ResourceExhaustion),
            "resource_exhaustion"
        );
    }
}
