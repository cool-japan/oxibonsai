//! Batched inference: process multiple prompts efficiently.
//!
//! Groups prompts into batches for prefill, then generates independently.
//! Provides a [`RequestQueue`] for continuous batching scenarios where
//! requests arrive over time and are drained in configurable batch sizes.

use std::time::Instant;

use crate::engine::InferenceEngine;
use crate::error::{RuntimeError, RuntimeResult};
use crate::sampling::SamplingParams;

// ─── Result types ──────────────────────────────────────────────────────

/// Result of a single batch element.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Number of prompt tokens processed.
    pub prompt_tokens: usize,
    /// Generated token IDs (not including the prompt).
    pub generated_tokens: Vec<u32>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Wall-clock time for this request in seconds.
    pub elapsed_seconds: f64,
}

/// Reason why token generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Reached the maximum token limit.
    MaxTokens,
    /// Generated the end-of-sequence token.
    Eos,
    /// An error occurred during generation.
    Error,
    /// Generation was stopped due to timeout.
    Timeout,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxTokens => write!(f, "max_tokens"),
            Self::Eos => write!(f, "eos"),
            Self::Error => write!(f, "error"),
            Self::Timeout => write!(f, "timeout"),
        }
    }
}

// ─── Batch configuration ───────────────────────────────────────────────

/// Batch inference configuration.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of prompts per batch.
    pub max_batch_size: usize,
    /// Maximum tokens to generate per request.
    pub max_tokens_per_request: usize,
    /// Optional timeout per request in milliseconds.
    pub timeout_per_request_ms: Option<u64>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_tokens_per_request: 512,
            timeout_per_request_ms: Some(30_000),
        }
    }
}

// ─── Batch generation ──────────────────────────────────────────────────

/// Process a batch of prompts sequentially (sharing the engine).
///
/// Each prompt is processed independently: the engine state is reset
/// between prompts. Returns one result per prompt.
pub fn batch_generate(
    engine: &mut InferenceEngine<'_>,
    prompts: &[Vec<u32>],
    max_tokens: usize,
) -> Vec<RuntimeResult<BatchResult>> {
    prompts
        .iter()
        .map(|prompt| {
            engine.reset();
            let start = Instant::now();

            match engine.generate(prompt, max_tokens) {
                Ok(tokens) => {
                    let finish_reason = if tokens.len() >= max_tokens {
                        FinishReason::MaxTokens
                    } else {
                        FinishReason::Eos
                    };
                    Ok(BatchResult {
                        prompt_tokens: prompt.len(),
                        generated_tokens: tokens,
                        finish_reason,
                        elapsed_seconds: start.elapsed().as_secs_f64(),
                    })
                }
                Err(e) => Err(e),
            }
        })
        .collect()
}

/// Process a batch with timeout per request.
///
/// Uses [`BatchConfig`] to control generation limits and timeouts.
/// If a request exceeds its timeout, it is terminated and marked
/// with [`FinishReason::Timeout`].
pub fn batch_generate_with_timeout(
    engine: &mut InferenceEngine<'_>,
    prompts: &[Vec<u32>],
    config: &BatchConfig,
) -> Vec<RuntimeResult<BatchResult>> {
    let effective_prompts = if prompts.len() > config.max_batch_size {
        &prompts[..config.max_batch_size]
    } else {
        prompts
    };

    effective_prompts
        .iter()
        .map(|prompt| {
            engine.reset();
            let start = Instant::now();
            let timeout = config
                .timeout_per_request_ms
                .map(std::time::Duration::from_millis);

            match engine.generate(prompt, config.max_tokens_per_request) {
                Ok(tokens) => {
                    let elapsed = start.elapsed();
                    let timed_out = timeout.is_some_and(|t| elapsed > t);

                    let finish_reason = if timed_out {
                        FinishReason::Timeout
                    } else if tokens.len() >= config.max_tokens_per_request {
                        FinishReason::MaxTokens
                    } else {
                        FinishReason::Eos
                    };

                    Ok(BatchResult {
                        prompt_tokens: prompt.len(),
                        generated_tokens: tokens,
                        finish_reason,
                        elapsed_seconds: elapsed.as_secs_f64(),
                    })
                }
                Err(e) => Err(e),
            }
        })
        .collect()
}

// ─── Request queue ─────────────────────────────────────────────────────

/// A single queued inference request.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Tokenized prompt.
    pub prompt_tokens: Vec<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling parameters for this request.
    pub params: SamplingParams,
}

/// Request queue for continuous batching.
///
/// Accumulates incoming requests and drains them in configurable
/// batch sizes for efficient processing.
pub struct RequestQueue {
    pending: Vec<BatchRequest>,
    max_size: usize,
}

impl RequestQueue {
    /// Create a new request queue with the given maximum capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            pending: Vec::with_capacity(max_size.min(1024)),
            max_size: max_size.max(1),
        }
    }

    /// Push a new request onto the queue.
    ///
    /// Returns an error if the queue is full.
    pub fn push(&mut self, request: BatchRequest) -> Result<(), RuntimeError> {
        if self.pending.len() >= self.max_size {
            return Err(RuntimeError::Server(format!(
                "request queue full (capacity: {})",
                self.max_size
            )));
        }
        self.pending.push(request);
        Ok(())
    }

    /// Drain up to `batch_size` requests from the front of the queue.
    ///
    /// Returns the drained requests in FIFO order.
    pub fn drain_batch(&mut self, batch_size: usize) -> Vec<BatchRequest> {
        let n = batch_size.min(self.pending.len());
        self.pending.drain(..n).collect()
    }

    /// Number of pending requests.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Whether the queue is at capacity.
    pub fn is_full(&self) -> bool {
        self.pending.len() >= self.max_size
    }

    /// Maximum queue capacity.
    pub fn capacity(&self) -> usize {
        self.max_size
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplingParams;
    use oxibonsai_core::config::Qwen3Config;

    // Test memory bound: `bonsai_8b()` would make `BonsaiModel::new` allocate
    // ~5 GB of token_embd + output_weight tables (plus a ~1.2 GB KV cache) for
    // every test using this helper; all callers here submit empty prompts and
    // never touch a dimension-dependent assertion, so the helper uses
    // `tiny_test()`, which exercises the identical batch-generation wiring
    // for a few tens of MB.
    fn make_engine() -> InferenceEngine<'static> {
        let config = Qwen3Config::tiny_test();
        InferenceEngine::new(config, SamplingParams::default(), 42)
    }

    #[test]
    fn batch_generate_empty_prompts() {
        let mut engine = make_engine();
        let results = batch_generate(&mut engine, &[], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn batch_generate_single_empty_prompt() {
        let mut engine = make_engine();
        let prompts = vec![vec![]];
        let results = batch_generate(&mut engine, &prompts, 10);
        assert_eq!(results.len(), 1);
        let result = results.into_iter().next().expect("should have one result");
        assert!(result.is_ok());
        let br = result.expect("should be ok");
        assert_eq!(br.prompt_tokens, 0);
        assert!(br.generated_tokens.is_empty());
        assert_eq!(br.finish_reason, FinishReason::Eos);
    }

    #[test]
    fn batch_generate_multiple_prompts() {
        let mut engine = make_engine();
        let prompts = vec![vec![], vec![], vec![]];
        let results = batch_generate(&mut engine, &prompts, 5);
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn batch_generate_with_timeout_respects_batch_size() {
        let mut engine = make_engine();
        let config = BatchConfig {
            max_batch_size: 2,
            max_tokens_per_request: 10,
            timeout_per_request_ms: Some(5_000),
        };
        // Provide 5 prompts but limit to 2
        let prompts = vec![vec![]; 5];
        let results = batch_generate_with_timeout(&mut engine, &prompts, &config);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn batch_config_default_values() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.max_tokens_per_request, 512);
        assert_eq!(config.timeout_per_request_ms, Some(30_000));
    }

    #[test]
    fn finish_reason_display() {
        assert_eq!(format!("{}", FinishReason::MaxTokens), "max_tokens");
        assert_eq!(format!("{}", FinishReason::Eos), "eos");
        assert_eq!(format!("{}", FinishReason::Error), "error");
        assert_eq!(format!("{}", FinishReason::Timeout), "timeout");
    }

    // ── RequestQueue tests ─────────────────────────────────────────────

    #[test]
    fn queue_new_empty() {
        let queue = RequestQueue::new(10);
        assert!(queue.is_empty());
        assert!(!queue.is_full());
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.capacity(), 10);
    }

    #[test]
    fn queue_min_capacity_is_one() {
        let queue = RequestQueue::new(0);
        assert_eq!(queue.capacity(), 1);
    }

    #[test]
    fn queue_push_and_drain() {
        let mut queue = RequestQueue::new(10);
        for i in 0..5 {
            let req = BatchRequest {
                prompt_tokens: vec![i as u32],
                max_tokens: 10,
                params: SamplingParams::default(),
            };
            queue.push(req).expect("should succeed");
        }
        assert_eq!(queue.len(), 5);
        assert!(!queue.is_full());

        let batch = queue.drain_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(queue.len(), 2);

        // Check FIFO order
        assert_eq!(batch[0].prompt_tokens, vec![0]);
        assert_eq!(batch[1].prompt_tokens, vec![1]);
        assert_eq!(batch[2].prompt_tokens, vec![2]);
    }

    #[test]
    fn queue_drain_more_than_available() {
        let mut queue = RequestQueue::new(10);
        let req = BatchRequest {
            prompt_tokens: vec![42],
            max_tokens: 10,
            params: SamplingParams::default(),
        };
        queue.push(req).expect("should succeed");

        let batch = queue.drain_batch(100);
        assert_eq!(batch.len(), 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn queue_full_rejects_push() {
        let mut queue = RequestQueue::new(2);
        let req1 = BatchRequest {
            prompt_tokens: vec![1],
            max_tokens: 10,
            params: SamplingParams::default(),
        };
        let req2 = BatchRequest {
            prompt_tokens: vec![2],
            max_tokens: 10,
            params: SamplingParams::default(),
        };
        let req3 = BatchRequest {
            prompt_tokens: vec![3],
            max_tokens: 10,
            params: SamplingParams::default(),
        };

        queue.push(req1).expect("should succeed");
        queue.push(req2).expect("should succeed");
        assert!(queue.is_full());

        let result = queue.push(req3);
        assert!(result.is_err());
    }

    #[test]
    fn queue_drain_empty() {
        let mut queue = RequestQueue::new(5);
        let batch = queue.drain_batch(3);
        assert!(batch.is_empty());
    }
}
