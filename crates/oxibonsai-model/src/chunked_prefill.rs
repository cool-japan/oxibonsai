//! Chunked prefill: process long prompts in smaller chunks.
//!
//! Instead of processing the entire prompt in one forward pass,
//! chunked prefill splits it into chunks and processes each sequentially.
//! This reduces peak memory from O(seq_len²) to O(chunk_size²).
//!
//! [`create_prefill_chunks`] / [`PrefillScheduler`] are a chunk-layout
//! *decision* layer only — they do not call anything. [`run_chunked_prefill`]
//! (M-18) is the executor: it drives [`PrefillScheduler`] against
//! caller-supplied prefill/decode callbacks so this module actually gets
//! used end to end, and it always uses a non-overlapping chunk split for
//! that execution regardless of `ChunkedPrefillConfig::overlap` (see
//! [`run_chunked_prefill`]'s doc comment for why).

use crate::error::ModelResult;

/// Configuration for chunked prefill.
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// Maximum tokens per chunk (default: 512).
    pub chunk_size: usize,
    /// Extra tokens re-included at the start of each chunk after the first,
    /// for callers that manage their own attention/context handling per
    /// chunk (stride = `chunk_size - overlap`).
    ///
    /// **Not safe for KV-cache-backed causal execution (M-18).** Re-running
    /// a token whose position was already committed to a KV cache
    /// overwrites that position's cached key/value with one computed from a
    /// different (shorter, chunk-local) attention window than a fresh,
    /// first-time forward pass at that position would see — silent
    /// corruption of causal history, not a tuning knob. [`run_chunked_prefill`]
    /// therefore always ignores this field and forces non-overlapping
    /// chunks; it is honoured only by calling [`create_prefill_chunks`] /
    /// [`PrefillScheduler`] directly, which remains correct for non-KV-cache
    /// use cases (e.g. a caller that keeps its own sliding context window).
    pub overlap: usize,
    /// Priority of prefill vs decode in scheduling.
    pub priority: PrefillPriority,
}

/// Scheduling priority between prefill and decode phases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefillPriority {
    /// Prefill all chunks before any decode (default).
    PrefillFirst,
    /// Interleave prefill chunks with decode steps.
    Interleaved,
    /// Decode has priority; prefill when idle.
    DecodePriority,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 0,
            priority: PrefillPriority::PrefillFirst,
        }
    }
}

impl ChunkedPrefillConfig {
    /// Create a new config with the given chunk size.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            ..Default::default()
        }
    }

    /// Set the overlap between consecutive chunks.
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the scheduling priority.
    pub fn with_priority(mut self, priority: PrefillPriority) -> Self {
        self.priority = priority;
        self
    }
}

/// A chunk of the prompt to prefill.
#[derive(Debug, Clone)]
pub struct PrefillChunk {
    /// Token IDs in this chunk.
    pub tokens: Vec<u32>,
    /// Start position in the original sequence.
    pub start_pos: usize,
    /// End position (exclusive) in the original sequence.
    pub end_pos: usize,
    /// Zero-based index of this chunk.
    pub chunk_index: usize,
    /// Whether this is the last chunk.
    pub is_last: bool,
}

impl PrefillChunk {
    /// Number of tokens in this chunk.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether this chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Split a prompt into prefill chunks.
///
/// When `overlap > 0`, consecutive chunks share `overlap` tokens at the boundary
/// so the model has context continuity. The stride is `chunk_size - overlap`.
///
/// M-18: this function honours `overlap` exactly as configured — it has no
/// way to know whether the caller has a KV cache. [`run_chunked_prefill`] is
/// the KV-cache-safe entry point for real execution and always forces
/// `overlap = 0`; call this function directly only when the cache-corruption
/// caveat documented on [`ChunkedPrefillConfig::overlap`] does not apply to
/// your use case.
pub fn create_prefill_chunks(
    prompt_tokens: &[u32],
    config: &ChunkedPrefillConfig,
) -> Vec<PrefillChunk> {
    if prompt_tokens.is_empty() {
        return vec![];
    }

    let chunk_size = config.chunk_size.max(1);
    let overlap = config.overlap.min(chunk_size.saturating_sub(1));
    let stride = chunk_size - overlap;

    let mut chunks = Vec::new();
    let total = prompt_tokens.len();
    let mut start = 0usize;
    let mut index = 0usize;

    while start < total {
        let end = (start + chunk_size).min(total);
        let tokens = prompt_tokens[start..end].to_vec();

        chunks.push(PrefillChunk {
            tokens,
            start_pos: start,
            end_pos: end,
            chunk_index: index,
            is_last: false, // fixed up below
        });

        index += 1;

        // Advance by stride, but if stride would not make progress (e.g.
        // overlap >= chunk_size), force at least 1 token forward.
        let advance = stride.max(1);
        start += advance;
    }

    // Mark the last chunk.
    if let Some(last) = chunks.last_mut() {
        last.is_last = true;
    }

    chunks
}

/// Action returned by the prefill scheduler.
#[derive(Debug, Clone)]
pub enum PrefillAction {
    /// Process the next prefill chunk.
    Prefill(PrefillChunk),
    /// All prefill done, proceed with decode.
    StartDecode,
    /// Yield to decode for one step (interleaved mode).
    YieldToDecode,
}

/// Prefill scheduling: determines the order of chunks and decode steps.
pub struct PrefillScheduler {
    config: ChunkedPrefillConfig,
    chunks: Vec<PrefillChunk>,
    current_chunk: usize,
    prefill_complete: bool,
    /// Tracks whether the last action was a prefill (for interleaved mode).
    last_was_prefill: bool,
}

impl PrefillScheduler {
    /// Create a new scheduler for the given prompt.
    pub fn new(prompt_tokens: &[u32], config: ChunkedPrefillConfig) -> Self {
        let chunks = create_prefill_chunks(prompt_tokens, &config);
        Self {
            config,
            chunks,
            current_chunk: 0,
            prefill_complete: false,
            last_was_prefill: false,
        }
    }

    /// Get the next action to perform.
    pub fn next_action(&mut self) -> PrefillAction {
        if self.prefill_complete || self.current_chunk >= self.chunks.len() {
            self.prefill_complete = true;
            return PrefillAction::StartDecode;
        }

        match self.config.priority {
            PrefillPriority::PrefillFirst => {
                let chunk = self.chunks[self.current_chunk].clone();
                self.current_chunk += 1;
                if self.current_chunk >= self.chunks.len() {
                    self.prefill_complete = true;
                }
                self.last_was_prefill = true;
                PrefillAction::Prefill(chunk)
            }
            PrefillPriority::Interleaved => {
                if self.last_was_prefill && self.current_chunk < self.chunks.len() {
                    // Yield after each prefill chunk.
                    self.last_was_prefill = false;
                    PrefillAction::YieldToDecode
                } else {
                    let chunk = self.chunks[self.current_chunk].clone();
                    self.current_chunk += 1;
                    if self.current_chunk >= self.chunks.len() {
                        self.prefill_complete = true;
                    }
                    self.last_was_prefill = true;
                    PrefillAction::Prefill(chunk)
                }
            }
            PrefillPriority::DecodePriority => {
                // In decode-priority mode, we still prefill but always yield
                // between chunks to let decode run first.
                if self.last_was_prefill {
                    self.last_was_prefill = false;
                    PrefillAction::YieldToDecode
                } else {
                    let chunk = self.chunks[self.current_chunk].clone();
                    self.current_chunk += 1;
                    if self.current_chunk >= self.chunks.len() {
                        self.prefill_complete = true;
                    }
                    self.last_was_prefill = true;
                    PrefillAction::Prefill(chunk)
                }
            }
        }
    }

    /// Report that decode can be performed (for interleaved mode).
    pub fn decode_available(&self) -> bool {
        !self.prefill_complete && self.config.priority != PrefillPriority::PrefillFirst
    }

    /// Whether all prefill chunks have been processed.
    pub fn is_complete(&self) -> bool {
        self.prefill_complete
    }

    /// Progress as fraction (0.0 - 1.0).
    pub fn progress(&self) -> f32 {
        if self.chunks.is_empty() {
            return 1.0;
        }
        self.current_chunk as f32 / self.chunks.len() as f32
    }

    /// Total number of chunks.
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Estimate memory savings compared to full prefill.
    ///
    /// The dominant memory consumer in self-attention is the attention score
    /// matrix of shape `[num_heads, seq_len, seq_len]` (FP32). With chunked
    /// prefill the largest matrix is `[num_heads, chunk_size, chunk_size]`.
    pub fn memory_savings(&self, hidden_dim: usize) -> f32 {
        if self.chunks.is_empty() {
            return 0.0;
        }
        let total_tokens: usize = self.chunks.iter().map(|c| c.end_pos).max().unwrap_or(0);
        let chunk_size = self.config.chunk_size;
        if total_tokens == 0 || chunk_size == 0 {
            return 0.0;
        }
        let full = total_tokens as f64 * total_tokens as f64 * hidden_dim as f64;
        let chunked = chunk_size as f64 * chunk_size as f64 * hidden_dim as f64;
        if full == 0.0 {
            return 0.0;
        }
        1.0 - (chunked / full) as f32
    }
}

/// Estimate peak memory for chunked vs full prefill.
///
/// The estimate focuses on the attention score matrix which dominates memory
/// in transformer forward passes: `num_heads * seq_len * seq_len * 4` bytes
/// (FP32).
pub fn peak_memory_estimate(
    seq_len: usize,
    chunk_size: usize,
    _hidden_dim: usize,
    num_heads: usize,
) -> PrefillMemoryEstimate {
    let bytes_per_element = 4usize; // f32
    let full_prefill_bytes = num_heads * seq_len * seq_len * bytes_per_element;
    let effective_chunk = chunk_size.min(seq_len);
    let chunked_prefill_bytes = num_heads * effective_chunk * effective_chunk * bytes_per_element;

    let memory_savings_ratio = if full_prefill_bytes == 0 {
        0.0
    } else {
        1.0 - (chunked_prefill_bytes as f32 / full_prefill_bytes as f32)
    };

    let num_chunks = if chunk_size == 0 {
        0
    } else {
        seq_len.div_ceil(chunk_size)
    };

    PrefillMemoryEstimate {
        full_prefill_bytes,
        chunked_prefill_bytes,
        memory_savings_ratio,
        num_chunks,
    }
}

/// Memory estimate comparing full vs chunked prefill.
#[derive(Debug, Clone)]
pub struct PrefillMemoryEstimate {
    /// Peak memory for full (non-chunked) prefill in bytes.
    pub full_prefill_bytes: usize,
    /// Peak memory for chunked prefill in bytes.
    pub chunked_prefill_bytes: usize,
    /// Ratio of memory saved (0.0 = no savings, 1.0 = all saved).
    pub memory_savings_ratio: f32,
    /// Number of chunks needed.
    pub num_chunks: usize,
}

impl PrefillMemoryEstimate {
    /// Human-readable summary of the estimate.
    pub fn summary(&self) -> String {
        let full_mb = self.full_prefill_bytes as f64 / (1024.0 * 1024.0);
        let chunked_mb = self.chunked_prefill_bytes as f64 / (1024.0 * 1024.0);
        let pct = self.memory_savings_ratio * 100.0;
        format!(
            "Full prefill: {full_mb:.1} MB, Chunked: {chunked_mb:.1} MB \
             ({pct:.1}% savings, {n} chunks)",
            n = self.num_chunks,
        )
    }
}

// ─── Model-side chunking policy (M-18) ───────────────────────────────────────

/// Default prompt length above which [`crate::model::BonsaiModel::forward_prefill`]
/// splits a prompt into chunks (M-18).
///
/// **How this number was chosen.** The caveat carried from M-18's verdict is
/// real: on macOS `forward_prefill` attempts a fused Metal batch path *per
/// call*, so chunking changes GPU dispatch granularity for long prompts, and
/// a small default could cost more than the peak-memory bound it buys. That
/// trade-off needed a real model on real hardware to settle, and the wave
/// that first wrote this constant ran in a worktree whose `models/` was
/// empty — the measurement below (PERF-CPU-PREFILL verifier pass) is what
/// closed it. Independently of that measurement, 4096 was already a safe
/// starting point:
///
/// * The longest prompt any test in the workspace hands to `forward_prefill`
///   is 20 tokens (`cuda_synthetic_prefill_parity`), so at 4096 every test
///   takes exactly the single-shot path it took before, byte for byte.
/// * 4096 is also the historical `MAX_PREALLOC_CONTEXT`, i.e. the context most
///   deployments actually run with, so the default changes nothing for them
///   either.
///
/// Callers that *want* chunking set it explicitly through
/// [`crate::model::BonsaiModel::set_prefill_chunk_tokens`]; the runtime's own
/// `InferenceEngine::set_prefill_chunk_tokens` is the same knob one level up.
///
/// **What the batched CPU prefill changed, and what it did not (PERF-CPU-PREFILL).**
/// `BonsaiModel::forward_prefill_cpu` now sub-divides a prompt internally
/// into `CPU_PREFILL_MICRO_BATCH`-token passes (128 today), each attending
/// over every position the earlier passes committed. So on the **CPU path**
/// this constant no longer governs peak activation memory at all: the CPU
/// prefill's working set is `128 x intermediate_size` floats whatever
/// `chunk_size` says, and the register-blocked GEMM decodes each weight
/// block once per `MR` rows however the batch is split, so a smaller chunk
/// size can no longer buy memory *or* cost throughput there. Raising it
/// would not help either — the micro-batch, not the chunk, is the CPU's
/// unit of batching.
///
/// **The GPU half of M-18's caveat, measured (PERF-CPU-PREFILL verifier
/// pass, 2026-09-22).** `chunked_prefill::tests::real_model_metal_prefill_chunk_size_sweep`
/// runs the real, shipped `Ternary-Bonsai-1.7B.gguf` through the fused Metal
/// `forward_prefill` path, a 1200-token prompt, at
/// `chunk_size in {256, 512, 1024, 2048, 4096}` (1 to 5 dispatches). One real
/// run measured:
///
/// | chunk_size | n_chunks | elapsed |
/// |-----------:|---------:|--------:|
/// |        256 |        5 | 28.17 s |
/// |        512 |        3 | 28.13 s |
/// |       1024 |        2 | 29.42 s |
/// |       2048 |        1 | 29.96 s |
/// |       4096 |        1 | 29.95 s |
///
/// Total wall time is **flat within about 6 %** across a 16x range of chunk
/// sizes (5 dispatches down to 1) — if anything, the *smallest* chunk size
/// tried was fastest here, the opposite of "more dispatches cost more". At
/// this prompt length, this M3, and this run's machine load, there is no
/// measured GPU dispatch-overhead tax for chunking smaller, and therefore no
/// measured reason to move `DEFAULT_PREFILL_CHUNK_TOKENS` off 4096. This
/// closes the "not measured" caveat with a real number rather than removing
/// it: the ~6 % spread is within what shared, contended hardware can produce
/// on its own (see the similar caveat on the CPU-side measurement in
/// `crate::model::types::prefill_cpu`'s module doc), so read this as "no
/// effect detected at this scale", not "provably zero effect at every
/// prompt length" — a prompt spanning dozens of chunks was not tried here.
/// It therefore stays at 4096 — a deliberate no-op for every prompt the
/// workspace's tests and the historical 4096-token context produce, and now
/// also the value this measurement did not find a reason to change.
pub const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 4096;

/// Decide whether a prompt of `prompt_len` tokens should be chunked at
/// `chunk_tokens`, and with what configuration (M-18).
///
/// Returns `None` — "run it in one shot, exactly as before" — when chunking
/// would not actually split the prompt: an empty or single-token prompt, a
/// disabled (`0`) chunk size, or a prompt that already fits in one chunk.
/// That is what keeps the Metal fused batch path's dispatch granularity
/// unchanged for every short prompt.
pub fn prefill_chunk_plan(prompt_len: usize, chunk_tokens: usize) -> Option<ChunkedPrefillConfig> {
    if chunk_tokens == 0 || prompt_len <= chunk_tokens || prompt_len <= 1 {
        return None;
    }
    // `overlap` is forced to 0 by `run_chunked_prefill` anyway; set it here so
    // the returned config is honest on its own.
    Some(ChunkedPrefillConfig {
        chunk_size: chunk_tokens,
        overlap: 0,
        priority: PrefillPriority::PrefillFirst,
    })
}

// ─── Chunked-prefill executor (M-18) ─────────────────────────────────────────

/// Drive a full chunked prefill end-to-end, executing each decision from a
/// [`PrefillScheduler`] against caller-supplied callbacks.
///
/// This is the executor this module was missing: [`PrefillScheduler`] only
/// *decides* what to do next; nothing previously called it against a real
/// forward pass. `run_chunked_prefill` owns that loop —
/// `prefill_fn(chunk)` should invoke the real batched-prefill forward pass
/// (e.g. `BonsaiModel::forward_prefill` in `crate::model`) for exactly that
/// chunk's tokens and start position, and `decode_fn()` should perform one
/// decode step of any other in-flight work when the scheduler yields
/// (`PrefillPriority::Interleaved` / `DecodePriority`); with the default
/// `PrefillPriority::PrefillFirst`, `decode_fn` is never called.
///
/// **M-18 correctness fix: chunks are always non-overlapping**, regardless
/// of `config.overlap`. [`create_prefill_chunks`] and the plain
/// [`PrefillScheduler`] still honour `overlap` exactly as configured — see
/// [`ChunkedPrefillConfig::overlap`] for why that remains correct for their
/// non-KV-cache use case. But a caller that plugs a real KV-cache-backed
/// forward pass into `prefill_fn` through *this* function must not
/// reprocess an already-committed position: doing so would silently
/// overwrite that position's cached key/value with one computed from a
/// different (shorter, chunk-local) attention window than a fresh
/// first-time forward would see. `run_chunked_prefill` therefore always
/// rebuilds `config` with `overlap: 0` before scheduling, so it is safe to
/// call with any `ChunkedPrefillConfig` — including one reused from
/// elsewhere with `overlap > 0` set.
///
/// Returns the last chunk's `prefill_fn` output (mirroring
/// `BonsaiModel::forward_prefill`'s "return the last position's output"
/// contract), or `Ok(None)` for an empty prompt.
///
/// # Errors
///
/// Returns the first error either callback produces; no further chunks are
/// processed after that. The caller's KV cache (if any) is left with
/// whatever prefix was already committed by earlier chunks — the same
/// partial-progress contract a single unchunked `forward_prefill` call has
/// on a mid-prompt kernel failure.
pub fn run_chunked_prefill<T>(
    prompt_tokens: &[u32],
    config: ChunkedPrefillConfig,
    mut prefill_fn: impl FnMut(&PrefillChunk) -> ModelResult<T>,
    mut decode_fn: impl FnMut() -> ModelResult<()>,
) -> ModelResult<Option<T>> {
    if prompt_tokens.is_empty() {
        return Ok(None);
    }

    // M-18: force non-overlapping chunking for real execution — see the doc
    // comment above for why `overlap` cannot be honoured here.
    let config = ChunkedPrefillConfig {
        overlap: 0,
        ..config
    };
    let mut scheduler = PrefillScheduler::new(prompt_tokens, config);
    let mut last = None;

    loop {
        match scheduler.next_action() {
            PrefillAction::Prefill(chunk) => {
                tracing::debug!(
                    chunk_index = chunk.chunk_index,
                    start_pos = chunk.start_pos,
                    end_pos = chunk.end_pos,
                    "running chunked-prefill chunk"
                );
                last = Some(prefill_fn(&chunk)?);
            }
            PrefillAction::YieldToDecode => {
                decode_fn()?;
            }
            PrefillAction::StartDecode => break,
        }
    }

    Ok(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default() {
        let cfg = ChunkedPrefillConfig::default();
        assert_eq!(cfg.chunk_size, 512);
        assert_eq!(cfg.overlap, 0);
        assert_eq!(cfg.priority, PrefillPriority::PrefillFirst);
    }

    #[test]
    fn config_builder() {
        let cfg = ChunkedPrefillConfig::new(256)
            .with_overlap(32)
            .with_priority(PrefillPriority::Interleaved);
        assert_eq!(cfg.chunk_size, 256);
        assert_eq!(cfg.overlap, 32);
        assert_eq!(cfg.priority, PrefillPriority::Interleaved);
    }

    #[test]
    fn empty_prompt() {
        let chunks = create_prefill_chunks(&[], &ChunkedPrefillConfig::default());
        assert!(chunks.is_empty());
    }

    // ── M-18: model-side chunking policy ─────────────────────────────────────

    #[test]
    fn chunk_plan_is_a_no_op_for_prompts_that_fit() {
        assert!(prefill_chunk_plan(0, 4096).is_none());
        assert!(prefill_chunk_plan(1, 4096).is_none());
        assert!(
            prefill_chunk_plan(20, 4096).is_none(),
            "the longest prompt any test uses"
        );
        assert!(
            prefill_chunk_plan(4096, 4096).is_none(),
            "exactly one chunk"
        );
        assert!(
            prefill_chunk_plan(100_000, 0).is_none(),
            "chunk size 0 means chunking is disabled"
        );
    }

    #[test]
    fn chunk_plan_splits_a_prompt_that_does_not_fit() {
        let cfg = prefill_chunk_plan(4097, 4096).expect("should chunk");
        assert_eq!(cfg.chunk_size, 4096);
        assert_eq!(
            cfg.overlap, 0,
            "overlap is never safe for KV-cache execution"
        );
        assert_eq!(cfg.priority, PrefillPriority::PrefillFirst);
        let chunks = create_prefill_chunks(&vec![0u32; 4097], &cfg);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn default_chunk_size_is_a_no_op_for_the_historical_context() {
        assert_eq!(DEFAULT_PREFILL_CHUNK_TOKENS, 4096);
        assert!(
            prefill_chunk_plan(DEFAULT_PREFILL_CHUNK_TOKENS, DEFAULT_PREFILL_CHUNK_TOKENS)
                .is_none()
        );
    }

    /// The chunk *scheduler* must keep partitioning a prompt exactly, whatever
    /// chunk size a caller picks — the batched CPU prefill's own
    /// `CPU_PREFILL_MICRO_BATCH` sub-division happens strictly inside one
    /// chunk and must never make the scheduler drop or duplicate a position
    /// (see the `DEFAULT_PREFILL_CHUNK_TOKENS` note).
    #[test]
    fn every_chunk_size_partitions_the_prompt_exactly() {
        let prompt: Vec<u32> = (0..1000u32).collect();
        for chunk_size in [1usize, 7, 64, 128, 129, 333, 999, 1000, 1001, 4096] {
            let cfg = ChunkedPrefillConfig::new(chunk_size);
            let chunks = create_prefill_chunks(&prompt, &cfg);
            let mut next = 0usize;
            let mut seen = 0usize;
            for chunk in &chunks {
                assert_eq!(
                    chunk.start_pos, next,
                    "chunk_size={chunk_size}: chunks must be contiguous"
                );
                assert_eq!(
                    chunk.tokens.len(),
                    chunk.end_pos - chunk.start_pos,
                    "chunk_size={chunk_size}: token count must match the span"
                );
                assert_eq!(
                    chunk.tokens.as_slice(),
                    &prompt[chunk.start_pos..chunk.end_pos],
                    "chunk_size={chunk_size}: tokens must be the prompt's own"
                );
                next = chunk.end_pos;
                seen += chunk.tokens.len();
            }
            assert_eq!(
                seen,
                prompt.len(),
                "chunk_size={chunk_size}: every position exactly once"
            );
            assert_eq!(next, prompt.len(), "chunk_size={chunk_size}: full coverage");
        }
    }

    // ── M-18: run_chunked_prefill executor ───────────────────────────────────

    #[test]
    fn run_chunked_prefill_empty_prompt_returns_none() {
        let mut prefill_calls = 0usize;
        let result = run_chunked_prefill(
            &[],
            ChunkedPrefillConfig::default(),
            |_chunk| {
                prefill_calls += 1;
                Ok(())
            },
            || Ok(()),
        )
        .expect("empty prompt should not error");
        assert!(result.is_none());
        assert_eq!(prefill_calls, 0);
    }

    #[test]
    fn run_chunked_prefill_short_prompt_is_single_chunk() {
        let tokens: Vec<u32> = (0..100).collect();
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let result = run_chunked_prefill(
            &tokens,
            ChunkedPrefillConfig::new(512),
            |chunk| {
                seen.push((chunk.start_pos, chunk.end_pos));
                Ok(chunk.tokens.len())
            },
            || Ok(()),
        )
        .expect("should succeed");
        assert_eq!(seen, vec![(0, 100)]);
        assert_eq!(result, Some(100));
    }

    #[test]
    fn run_chunked_prefill_visits_non_overlapping_contiguous_chunks() {
        let tokens: Vec<u32> = (0..1500).collect();
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let _ = run_chunked_prefill(
            &tokens,
            ChunkedPrefillConfig::new(512),
            |chunk| {
                seen.push((chunk.start_pos, chunk.end_pos));
                Ok(())
            },
            || Ok(()),
        )
        .expect("should succeed");

        assert_eq!(seen, vec![(0, 512), (512, 1024), (1024, 1500)]);
        // Every position from 0..1500 is covered exactly once.
        for w in seen.windows(2) {
            assert_eq!(
                w[0].1, w[1].0,
                "chunks must be contiguous with no gap or overlap: {w:?}"
            );
        }
    }

    #[test]
    fn run_chunked_prefill_forces_non_overlapping_even_if_config_requests_overlap() {
        // M-18: even though this config asks for overlap=64, the executor
        // must never resubmit an already-committed position.
        let tokens: Vec<u32> = (0..1024).collect();
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let _ = run_chunked_prefill(
            &tokens,
            ChunkedPrefillConfig::new(512).with_overlap(64),
            |chunk| {
                seen.push((chunk.start_pos, chunk.end_pos));
                Ok(())
            },
            || Ok(()),
        )
        .expect("should succeed");

        for w in seen.windows(2) {
            assert!(
                w[1].0 >= w[0].1,
                "chunk {:?} must not overlap previous chunk {:?} (M-18)",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn run_chunked_prefill_returns_last_chunk_result() {
        let tokens: Vec<u32> = (0..1500).collect();
        let result = run_chunked_prefill(
            &tokens,
            ChunkedPrefillConfig::new(512),
            |chunk| Ok(chunk.chunk_index),
            || Ok(()),
        )
        .expect("should succeed");
        // 1500 tokens / 512 chunk_size = 3 chunks, indices 0, 1, 2.
        assert_eq!(result, Some(2));
    }

    #[test]
    fn run_chunked_prefill_propagates_prefill_error_and_stops() {
        let tokens: Vec<u32> = (0..1500).collect();
        let mut calls = 0usize;
        let result: ModelResult<Option<()>> = run_chunked_prefill(
            &tokens,
            ChunkedPrefillConfig::new(512),
            |_chunk| {
                calls += 1;
                if calls == 2 {
                    Err(crate::error::ModelError::Internal("boom".to_string()))
                } else {
                    Ok(())
                }
            },
            || Ok(()),
        );
        assert!(result.is_err(), "error from prefill_fn must propagate");
        assert_eq!(calls, 2, "must stop at the failing chunk, not continue");
    }

    #[test]
    fn run_chunked_prefill_propagates_decode_error() {
        let tokens: Vec<u32> = (0..1024).collect();
        let cfg = ChunkedPrefillConfig::new(512).with_priority(PrefillPriority::Interleaved);
        let result: ModelResult<Option<()>> = run_chunked_prefill(
            &tokens,
            cfg,
            |_chunk| Ok(()),
            || {
                Err(crate::error::ModelError::Internal(
                    "decode failed".to_string(),
                ))
            },
        );
        assert!(result.is_err(), "error from decode_fn must propagate");
    }

    #[test]
    fn run_chunked_prefill_interleaved_calls_decode_between_chunks() {
        let tokens: Vec<u32> = (0..1024).collect();
        let cfg = ChunkedPrefillConfig::new(512).with_priority(PrefillPriority::Interleaved);
        let mut prefill_calls = 0usize;
        let mut decode_calls = 0usize;
        let _ = run_chunked_prefill(
            &tokens,
            cfg,
            |_chunk| {
                prefill_calls += 1;
                Ok(())
            },
            || {
                decode_calls += 1;
                Ok(())
            },
        )
        .expect("should succeed");

        // 1024 / 512 = 2 chunks; PrefillScheduler's Interleaved mode yields
        // once after each prefill chunk (see the integration test
        // `scheduler_interleaved` in `tests/chunked_prefill_tests.rs`).
        assert_eq!(prefill_calls, 2);
        assert_eq!(decode_calls, 1);
    }

    #[test]
    fn run_chunked_prefill_prefill_first_never_calls_decode() {
        let tokens: Vec<u32> = (0..1024).collect();
        let cfg = ChunkedPrefillConfig::new(512); // default: PrefillFirst
        let mut decode_calls = 0usize;
        let _ = run_chunked_prefill(
            &tokens,
            cfg,
            |_chunk| Ok(()),
            || {
                decode_calls += 1;
                Ok(())
            },
        )
        .expect("should succeed");
        assert_eq!(decode_calls, 0);
    }

    // ── Wave-2.5 addendum (6) part 2: the GPU dispatch-granularity
    // measurement `DEFAULT_PREFILL_CHUNK_TOKENS`'s doc says is still open ──

    /// On macOS, `BonsaiModel::forward_prefill` tries the fused Metal batch
    /// path before anything else, so `chunk_size` is that path's per-call
    /// dispatch granularity for a long prompt — the measurement
    /// `DEFAULT_PREFILL_CHUNK_TOKENS`'s doc says "can only be read off a
    /// real model on real hardware" (PERF-CPU-PREFILL verifier pass: the
    /// prior "models/ is empty in the isolated worktree" excuse does not
    /// hold — the real GGUFs are readable at the primary worktree's
    /// `models/`). Ignored by default (needs a multi-hundred-MB real model
    /// + a dev Mac with Metal).
    ///
    /// Run with:
    /// ```text
    /// OXI_MODEL=/path/to/Ternary-Bonsai-1.7B.gguf \
    ///   cargo test -p oxibonsai-model --release --features metal --lib \
    ///   chunked_prefill::tests::real_model_metal_prefill_chunk_size_sweep \
    ///   -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires OXI_MODEL real ternary GGUF + Metal; run on dev Mac"]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn real_model_metal_prefill_chunk_size_sweep() {
        use crate::model::BonsaiModel;
        use oxibonsai_core::gguf::reader::GgufFile;
        use oxibonsai_kernels::{KernelDispatcher, KernelTier};
        use std::time::Instant;

        let Some(path) = std::env::var_os("OXI_MODEL") else {
            eprintln!(
                "real_model_metal_prefill_chunk_size_sweep: OXI_MODEL not set — skipping. \
                 Set OXI_MODEL=/path/to/Ternary-Bonsai-1.7B.gguf to run."
            );
            return;
        };
        let bytes = std::fs::read(&path).expect("read OXI_MODEL gguf");
        let gguf = GgufFile::parse(&bytes).expect("GgufFile::parse OXI_MODEL");
        let cfg = oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata)
            .expect("Qwen3Config::from_metadata OXI_MODEL");

        const MAX_SEQ: usize = 4096;
        const PROMPT_LEN: usize = 1200;
        let vocab = cfg.vocab_size as u32;
        assert!(vocab > 1, "real model must have a non-trivial vocabulary");
        let prompt: Vec<u32> = (0..PROMPT_LEN as u32)
            .map(|i| 1 + (i * 97) % (vocab - 1))
            .collect();

        // A fresh model per chunk size: `forward_prefill` writes the KV
        // cache, so each trial needs its own empty one to prefill from
        // `pos_start = 0` cleanly, exactly like the previous trial did.
        let kernel = KernelDispatcher::with_tier(KernelTier::Gpu);
        eprintln!("\nM-18 chunk-size sweep, Metal fused prefill, {PROMPT_LEN}-token prompt:");
        eprintln!(
            "{:>10}  {:>10}  {:>12}",
            "chunk_size", "n_chunks", "elapsed_ms"
        );
        for &chunk_size in &[256usize, 512, 1024, 2048, 4096] {
            let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
            model.set_prefill_chunk_tokens(chunk_size);
            let n_chunks = PROMPT_LEN.div_ceil(chunk_size.max(1));
            let t0 = Instant::now();
            model
                .forward_prefill(&prompt, 0, &kernel)
                .expect("metal fused prefill should succeed on the real model");
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            eprintln!("{chunk_size:>10}  {n_chunks:>10}  {elapsed_ms:>12.2}");
        }
    }
}
