//! Continuous (iteration-level) batching for OxiBonsai.
//!
//! Continuous batching processes multiple inference requests simultaneously by
//! adding new requests to the active set as slots free up — unlike static
//! batching where every request in a batch must start and finish together.
//!
//! The [`ContinuousBatchScheduler`] maintains three queues:
//!
//! 1. **Waiting queue** — requests awaiting a slot.
//! 2. **Active set** — at most `max_concurrent` requests currently being decoded.
//! 3. **Completed list** — finished requests available for result retrieval.
//!
//! Each call to [`ContinuousBatchScheduler::step`] advances every active request
//! by exactly one token, promoting waiting requests when slots become available.

use std::collections::VecDeque;

use crate::engine::InferenceEngine;
use crate::sampling::SamplingParams;

// ─── Priority ──────────────────────────────────────────────────────────────

/// Priority level for request scheduling.
///
/// Higher-priority requests are promoted from the waiting queue before
/// lower-priority ones when slots become available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum RequestPriority {
    /// Lowest priority — background work.
    Low = 0,
    /// Default priority for most requests.
    #[default]
    Normal = 1,
    /// Elevated priority — user-facing interactive requests.
    High = 2,
    /// Highest priority — real-time / SLA-bound requests.
    Critical = 3,
}

// ─── State ─────────────────────────────────────────────────────────────────

/// Lifecycle state of a [`BatchRequest`].
#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    /// Sitting in the waiting queue.
    Waiting,
    /// Running the prompt prefill phase.
    Prefilling,
    /// Actively generating tokens one by one.
    Decoding,
    /// All tokens generated (or EOS hit).
    Completed,
    /// Generation failed with the enclosed message.
    Failed(String),
}

// ─── BatchRequest ──────────────────────────────────────────────────────────

/// A single inference request managed by the continuous-batch scheduler.
pub struct BatchRequest {
    /// Unique request identifier returned by [`ContinuousBatchScheduler::submit`].
    pub id: u64,
    /// Tokenised prompt.
    pub prompt_tokens: Vec<u32>,
    /// Sampling parameters for this request.
    pub params: SamplingParams,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Scheduling priority.
    pub priority: RequestPriority,
    /// Current lifecycle state.
    pub state: RequestState,
    /// Tokens generated so far (not including the prompt).
    pub generated_tokens: Vec<u32>,
    /// Wall-clock time at which the request was submitted.
    pub created_at: std::time::Instant,
    /// Wall-clock time at which the first token was generated (prefill complete).
    pub started_at: Option<std::time::Instant>,
    /// Wall-clock time at which generation finished.
    pub completed_at: Option<std::time::Instant>,
}

impl BatchRequest {
    /// Create a new request with `Normal` priority.
    pub fn new(
        id: u64,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        max_tokens: usize,
    ) -> Self {
        Self {
            id,
            prompt_tokens,
            params,
            max_tokens,
            priority: RequestPriority::Normal,
            state: RequestState::Waiting,
            generated_tokens: Vec::new(),
            created_at: std::time::Instant::now(),
            started_at: None,
            completed_at: None,
        }
    }

    /// Override the priority, returning `self` for builder-style chaining.
    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Elapsed time from submission to first generated token.
    ///
    /// Returns `None` if the first token has not yet been produced.
    pub fn time_to_first_token(&self) -> Option<std::time::Duration> {
        self.started_at.map(|s| s.duration_since(self.created_at))
    }

    /// Elapsed time from submission to completion.
    ///
    /// Returns `None` if the request has not yet completed.
    pub fn total_latency(&self) -> Option<std::time::Duration> {
        self.completed_at.map(|c| c.duration_since(self.created_at))
    }

    /// Number of tokens generated so far.
    pub fn tokens_generated(&self) -> usize {
        self.generated_tokens.len()
    }

    /// `true` when the request is in [`RequestState::Completed`] or
    /// [`RequestState::Failed`].
    pub fn is_finished(&self) -> bool {
        matches!(
            self.state,
            RequestState::Completed | RequestState::Failed(_)
        )
    }
}

// ─── Errors ────────────────────────────────────────────────────────────────

/// Errors returned by the continuous-batch scheduler.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// The waiting queue is at capacity.
    #[error("Queue full: {max_queue_size} requests waiting")]
    QueueFull {
        /// The configured maximum queue size.
        max_queue_size: usize,
    },
    /// No request with the given ID was found.
    #[error("Request {id} not found")]
    NotFound {
        /// The unknown request ID.
        id: u64,
    },
}

// ─── Stats ─────────────────────────────────────────────────────────────────

/// Throughput statistics snapshot.
#[derive(Debug, serde::Serialize)]
pub struct SchedulerStats {
    /// Total requests submitted since the scheduler was created.
    pub total_requests: u64,
    /// Total tokens generated across all completed requests.
    pub total_tokens_generated: u64,
    /// Current depth of the waiting queue.
    pub queue_depth: usize,
    /// Number of actively decoding requests.
    pub active_count: usize,
}

// ─── Scheduler ─────────────────────────────────────────────────────────────

/// Continuous-batch scheduler.
///
/// Manages request lifecycle from submission through generation to completion,
/// interleaving multiple requests at the token level.
pub struct ContinuousBatchScheduler {
    /// Maximum number of requests decoding simultaneously.
    pub max_concurrent: usize,
    /// Maximum number of requests that may wait in the queue.
    pub max_queue_size: usize,

    queue: VecDeque<BatchRequest>,
    active: Vec<BatchRequest>,
    completed: Vec<BatchRequest>,
    next_id: u64,
    total_requests: u64,
    total_tokens_generated: u64,
}

impl ContinuousBatchScheduler {
    /// Create a new scheduler.
    ///
    /// `max_concurrent` — at most this many requests decode in parallel.
    /// `max_queue_size` — queue rejects new submissions beyond this count.
    pub fn new(max_concurrent: usize, max_queue_size: usize) -> Self {
        Self {
            max_concurrent: max_concurrent.max(1),
            max_queue_size: max_queue_size.max(1),
            queue: VecDeque::new(),
            active: Vec::new(),
            completed: Vec::new(),
            next_id: 1,
            total_requests: 0,
            total_tokens_generated: 0,
        }
    }

    /// Submit a request with `Normal` priority.
    ///
    /// Returns the assigned request ID, or [`SchedulerError::QueueFull`] if the
    /// waiting queue is already at capacity.
    pub fn submit(
        &mut self,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        max_tokens: usize,
    ) -> Result<u64, SchedulerError> {
        self.submit_with_priority(prompt_tokens, params, max_tokens, RequestPriority::Normal)
    }

    /// Submit a request with an explicit priority.
    pub fn submit_with_priority(
        &mut self,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        max_tokens: usize,
        priority: RequestPriority,
    ) -> Result<u64, SchedulerError> {
        if self.queue.len() >= self.max_queue_size {
            return Err(SchedulerError::QueueFull {
                max_queue_size: self.max_queue_size,
            });
        }

        let id = self.next_id;
        self.next_id += 1;
        self.total_requests += 1;

        let request =
            BatchRequest::new(id, prompt_tokens, params, max_tokens).with_priority(priority);

        // Insert maintaining priority order (higher priority → closer to front)
        let pos = self
            .queue
            .iter()
            .position(|r| r.priority < priority)
            .unwrap_or(self.queue.len());
        self.queue.insert(pos, request);

        Ok(id)
    }

    /// Advance one iteration of the batch.
    ///
    /// 1. Promotes waiting requests into the active set until `max_concurrent`
    ///    slots are full (or the queue is drained).
    /// 2. Steps every active request by generating one token.
    /// 3. Moves finished requests to the completed list.
    pub fn step(&mut self, engine: &mut InferenceEngine<'_>) {
        // --- Promote waiting requests into the active set ---
        while self.active.len() < self.max_concurrent {
            match self.queue.pop_front() {
                Some(mut req) => {
                    req.state = RequestState::Prefilling;
                    self.active.push(req);
                }
                None => break,
            }
        }

        if self.active.is_empty() {
            return;
        }

        // --- Step each active request by one token ---
        let mut finished_indices: Vec<usize> = Vec::new();

        for (idx, req) in self.active.iter_mut().enumerate() {
            // Build the full context: prompt + already-generated tokens
            let context: Vec<u32> = req
                .prompt_tokens
                .iter()
                .chain(req.generated_tokens.iter())
                .copied()
                .collect();

            // Run the engine for a single token
            engine.reset();
            let generated = engine.generate(&context, 1);

            match generated {
                Ok(new_tokens) => {
                    if req.started_at.is_none() {
                        req.started_at = Some(std::time::Instant::now());
                        req.state = RequestState::Decoding;
                    }

                    if let Some(&token) = new_tokens.first() {
                        req.generated_tokens.push(token);
                    }

                    // Check stopping conditions
                    let hit_max = req.generated_tokens.len() >= req.max_tokens;
                    let hit_eos = new_tokens.is_empty(); // engine stopped at EOS

                    if hit_max || hit_eos {
                        req.state = RequestState::Completed;
                        req.completed_at = Some(std::time::Instant::now());
                        finished_indices.push(idx);
                    }
                }
                Err(e) => {
                    req.state = RequestState::Failed(e.to_string());
                    req.completed_at = Some(std::time::Instant::now());
                    finished_indices.push(idx);
                }
            }
        }

        // Move finished requests to completed list (iterate in reverse to
        // preserve index validity during removal)
        for &idx in finished_indices.iter().rev() {
            let req = self.active.remove(idx);
            self.total_tokens_generated += req.generated_tokens.len() as u64;
            self.completed.push(req);
        }
    }

    /// Run all pending and active requests to completion, blocking until the
    /// scheduler is idle.
    pub fn run_to_completion(&mut self, engine: &mut InferenceEngine<'_>) {
        while !self.is_idle() {
            self.step(engine);
        }
    }

    /// Look up a completed (or failed) request by ID.
    ///
    /// Returns `None` if the request is still waiting/decoding or does not exist.
    pub fn get_result(&self, id: u64) -> Option<&BatchRequest> {
        self.completed.iter().find(|r| r.id == id)
    }

    /// Number of requests currently waiting in the queue.
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Number of requests currently being decoded.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Number of completed (or failed) requests.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// `true` when there are no waiting or active requests.
    pub fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.active.is_empty()
    }

    /// Snapshot of current throughput statistics.
    pub fn throughput_stats(&self) -> SchedulerStats {
        SchedulerStats {
            total_requests: self.total_requests,
            total_tokens_generated: self.total_tokens_generated,
            queue_depth: self.queue.len(),
            active_count: self.active.len(),
        }
    }

    /// Remove and return all completed requests, clearing the completed list.
    pub fn drain_completed(&mut self) -> Vec<BatchRequest> {
        std::mem::take(&mut self.completed)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::config::Qwen3Config;

    // HOTFIX-TESTMEM: `bonsai_8b()` made `BonsaiModel::new` allocate ~5 GB of
    // token_embd + output_weight tables (plus a ~1.2 GB KV cache) for every
    // test using this helper, none of which exercise anything
    // dimension-dependent (both callers submit empty prompts). `tiny_test()`
    // exercises the identical scheduler <-> engine wiring for a few tens of
    // MB.
    fn make_engine() -> InferenceEngine<'static> {
        let config = Qwen3Config::tiny_test();
        InferenceEngine::new(config, SamplingParams::default(), 42)
    }

    fn default_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.0, // greedy for determinism
            ..Default::default()
        }
    }

    // ── Submit / queue tests ───────────────────────────────────────────────

    #[test]
    fn test_scheduler_submit_returns_id() {
        let mut sched = ContinuousBatchScheduler::new(4, 64);
        let id1 = sched
            .submit(vec![1, 2, 3], default_params(), 10)
            .expect("submit should succeed");
        let id2 = sched
            .submit(vec![4, 5, 6], default_params(), 10)
            .expect("submit should succeed");
        assert_ne!(id1, id2, "IDs must be unique");
        assert!(id1 > 0 && id2 > 0);
    }

    #[test]
    fn test_scheduler_queue_depth() {
        let mut sched = ContinuousBatchScheduler::new(1, 64);
        assert_eq!(sched.queue_depth(), 0);

        sched
            .submit(vec![1], default_params(), 5)
            .expect("submit should succeed");
        sched
            .submit(vec![2], default_params(), 5)
            .expect("submit should succeed");
        assert_eq!(sched.queue_depth(), 2);
    }

    #[test]
    fn test_scheduler_max_queue_enforced() {
        let mut sched = ContinuousBatchScheduler::new(8, 2);
        sched
            .submit(vec![1], default_params(), 5)
            .expect("first submit should succeed");
        sched
            .submit(vec![2], default_params(), 5)
            .expect("second submit should succeed");

        let err = sched
            .submit(vec![3], default_params(), 5)
            .expect_err("third submit should be rejected");

        assert!(
            matches!(err, SchedulerError::QueueFull { max_queue_size: 2 }),
            "unexpected error variant: {err}"
        );
    }

    // ── Priority tests ─────────────────────────────────────────────────────

    #[test]
    fn test_request_priority_ordering() {
        assert!(RequestPriority::Critical > RequestPriority::High);
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
    }

    #[test]
    fn test_priority_queue_ordering() {
        let mut sched = ContinuousBatchScheduler::new(1, 64);

        // Submit low priority first, then high priority
        sched
            .submit_with_priority(vec![1], default_params(), 5, RequestPriority::Low)
            .expect("submit low");
        sched
            .submit_with_priority(vec![2], default_params(), 5, RequestPriority::High)
            .expect("submit high");

        // The high-priority request should be at the front of the queue
        let front = sched.queue.front().expect("queue should not be empty");
        assert_eq!(front.priority, RequestPriority::High);
    }

    // ── State transition tests ─────────────────────────────────────────────

    #[test]
    fn test_request_state_transitions() {
        let req = BatchRequest::new(1, vec![10, 11], default_params(), 5);
        assert_eq!(req.state, RequestState::Waiting);
        assert!(!req.is_finished());

        let mut req = req;
        req.state = RequestState::Prefilling;
        assert!(!req.is_finished());

        req.state = RequestState::Decoding;
        assert!(!req.is_finished());

        req.state = RequestState::Completed;
        assert!(req.is_finished());

        req.state = RequestState::Failed("oops".into());
        assert!(req.is_finished());
    }

    // ── Latency measurement tests ──────────────────────────────────────────

    #[test]
    fn test_batch_request_time_to_first_token() {
        let mut req = BatchRequest::new(42, vec![1, 2, 3], default_params(), 10);
        assert!(req.time_to_first_token().is_none());
        assert!(req.total_latency().is_none());

        // Simulate first-token timing
        req.started_at = Some(req.created_at + std::time::Duration::from_millis(10));
        let ttft = req.time_to_first_token().expect("should have TTFT");
        assert!(ttft.as_millis() >= 10, "TTFT should be >= 10ms");

        req.completed_at = Some(req.created_at + std::time::Duration::from_millis(50));
        let lat = req.total_latency().expect("should have latency");
        assert!(lat.as_millis() >= 50, "latency should be >= 50ms");
    }

    // ── drain_completed tests ──────────────────────────────────────────────

    #[test]
    fn test_scheduler_drain_completed() {
        let mut sched = ContinuousBatchScheduler::new(4, 64);
        let mut engine = make_engine();

        let _id = sched
            .submit(vec![], default_params(), 2)
            .expect("submit should succeed");

        sched.run_to_completion(&mut engine);

        let drained = sched.drain_completed();
        assert!(
            !drained.is_empty(),
            "should have at least one completed request"
        );
        assert_eq!(
            sched.completed_count(),
            0,
            "completed list should be empty after drain"
        );
    }

    // ── Stats tests ────────────────────────────────────────────────────────

    #[test]
    fn test_scheduler_stats() {
        let mut sched = ContinuousBatchScheduler::new(4, 64);
        sched
            .submit(vec![1, 2], default_params(), 5)
            .expect("submit should succeed");
        sched
            .submit(vec![3, 4], default_params(), 5)
            .expect("submit should succeed");

        let stats = sched.throughput_stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.queue_depth, 2);
        assert_eq!(stats.active_count, 0);
        assert_eq!(stats.total_tokens_generated, 0);
    }

    // ── run_to_completion tests ────────────────────────────────────────────

    #[test]
    fn test_scheduler_run_to_completion() {
        let mut sched = ContinuousBatchScheduler::new(4, 64);
        let mut engine = make_engine();

        // Empty prompt — engine returns immediately with EOS
        let id = sched
            .submit(vec![], default_params(), 5)
            .expect("submit should succeed");

        sched.run_to_completion(&mut engine);

        assert!(sched.is_idle(), "scheduler should be idle after completion");

        let result = sched.get_result(id).expect("result should be available");
        assert!(
            result.is_finished(),
            "request should be finished, state={:?}",
            result.state
        );
    }

    #[test]
    fn test_scheduler_is_idle_initially() {
        let sched = ContinuousBatchScheduler::new(4, 64);
        assert!(sched.is_idle());
        assert_eq!(sched.active_count(), 0);
        assert_eq!(sched.queue_depth(), 0);
        assert_eq!(sched.completed_count(), 0);
    }
}
