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
/// On macOS `forward_prefill` runs the fused Metal batch prefill **per
/// chunk**, so this is that path's per-call granularity: every call pays its
/// own embedding gather, RoPE tables and weight binding, and inside a call
/// the kernels already run the prompt in bounded micro-batches
/// (`PREFILL_LOGITS_MICRO_BATCH` rows per command buffer), so a chunk no
/// longer bounds GPU memory or the length of a command buffer. The CPU path
/// is indifferent too: `BonsaiModel::forward_prefill_cpu` sub-divides every
/// call into `CPU_PREFILL_MICRO_BATCH`-row passes of its own.
///
/// **Measured** (`real_model_metal_prefill_chunk_size_sweep`: the real
/// GGUF loaded the way a Metal engine loads it — weights uploaded, fused
/// weight cache resident, the fused route pinned — Apple M3, release,
/// microseconds per prompt token; "sequential" is `forward` token by token,
/// the path the fused prefill replaces; load average 6-20 on a shared
/// host). Before the tiled Q1 prefill GEMM, the Bonsai-8B one-bit route was
/// slower than sequential decode at every size (the M-18 finding; its
/// row-wise GEMM re-read every input column once per weight row, so its
/// per-token cost grew with the call). "After" is the tiled Q1 / ternary
/// prefill GEMMs with the reworked flash attention:
///
/// | model, prompt | chunk 0 | 128 | 256 | 512 | 1024 | 4096 | sequential |
/// |---|---:|---:|---:|---:|---:|---:|---:|
/// | Bonsai-8B, 512 tokens, before | 147 414 | 60 054 | 80 330 | 147 414 | 147 414 | 147 414 | 48 009 |
/// | Bonsai-8B, 4096 tokens, after | 6 702 | 7 014 | 6 738 | 6 759 | 7 165 | 6 702 | 78 507 |
/// | Ternary-Bonsai-1.7B, 4096 tokens, before | 2 571 | 2 849 | 2 603 | 2 422 | 2 443 | 2 571 | 33 514 |
/// | Ternary-Bonsai-1.7B, 4096 tokens, after | 1 819 | 1 937 | 1 800 | 1 694 | 1 706 | 1 819 | 32 729 |
///
/// At 256 tokens in one call the fused prefill costs 5 635 µs/token on
/// Bonsai-8B against 46 245 for sequential decode (8.2x; before: 82 951
/// against 43 413, 0.52x) and 1 276 against 15 137 on Ternary-Bonsai-1.7B
/// (11.9x). A whole 4096-token prompt takes 27.5 s fused against 321.6 s
/// decoded token by token on Bonsai-8B (7.4 s against 134.1 s on the 1.7B),
/// and the per-token cost grows 1.19x (1.43x) from 256 to 4096 tokens — the
/// attention term, which is inherently quadratic.
///
/// Chunking moves the cost by less than 10 % either way — within the load
/// drift of a shared host, the arms running one after another — except that
/// 128-token chunks are dearer than one call on both models: per-call
/// overhead, repeated 32 times. So the default is the largest chunk that
/// still bounds a call's host-side staging (the `[chunk x hidden]` embedding
/// batch, 64 MiB on the 8B at 4096): **4096** — one call for every prompt of
/// the historical 4096-token context, 4096-token chunks beyond it. It is
/// ratified together with the prefill router in `forward_metal.rs`, which
/// keeps a model whose fused path measures slower than decode on the
/// sequential route and bounds every fused call by a deadline.
///
/// Callers that *want* smaller chunks set them through
/// [`crate::model::BonsaiModel::set_prefill_chunk_tokens`]; the runtime's own
/// `InferenceEngine::set_prefill_chunk_tokens` is the same knob one level up.
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

    // ── M-18: the chunk-size sweep on a real model through the fused Metal path ──

    /// Chunk sizes the sweep measures. `0` disables chunking (one fused call
    /// for the whole prompt); a size at or above the prompt length is the same
    /// dispatch as `0` and is measured once and reported for every such size.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const SWEEP_CHUNK_SIZES: [usize; 6] = [0, 128, 256, 512, 1024, 4096];

    /// Default sweep prompt length: the largest chunk size, so every entry of
    /// [`SWEEP_CHUNK_SIZES`] is a distinct dispatch pattern.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const SWEEP_DEFAULT_PROMPT_TOKENS: usize = 4096;

    /// Prompt length of the "short prompt" leg the per-token ratios are
    /// quoted at.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const SWEEP_SHORT_PROMPT_TOKENS: usize = 256;

    /// M-18 acceptance: fused prefill per-token speed-up over sequential
    /// decode on the short prompt.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const SWEEP_MIN_SHORT_SPEEDUP: f64 = 3.0;

    /// M-18 acceptance: fused per-token cost on the full prompt over the
    /// short prompt's (the attention term is the only super-linear part).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const SWEEP_MAX_GROWTH: f64 = 1.5;

    /// Whether `InferenceEngine::from_gguf` would skip the scirs2 weight
    /// upload for this file: an all-ternary file binds every fused Metal path
    /// through its own cached weight set, while a one-bit file's fused paths
    /// need the uploaded handles (mirrors `engine_control`'s
    /// `gpu_weight_upload_redundant`, which this crate cannot depend on).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn sweep_upload_redundant(gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>) -> bool {
        let ternary_head = gguf
            .tensors
            .get("output.weight")
            .is_some_and(|head| head.tensor_type.is_ternary());
        ternary_head
            && gguf.tensors.iter().all(|(name, info)| {
                let blocked_matrix = (name.starts_with("blk.") && name.ends_with(".weight"))
                    || name == "token_embd.weight";
                !blocked_matrix
                    || info.tensor_type.block_size() <= 1
                    || info.tensor_type.is_ternary()
            })
    }

    /// A model loaded and made GPU-resident exactly the way
    /// `InferenceEngine::from_gguf` prepares a Metal engine: the
    /// auto-detected dispatcher, the weight upload where the route needs it,
    /// and the eager fused-weight cache.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn sweep_gpu_model<'a>(
        gguf: &'a oxibonsai_core::gguf::reader::GgufFile<'a>,
        kernel: &oxibonsai_kernels::KernelDispatcher,
        max_seq: usize,
    ) -> crate::model::BonsaiModel<'a> {
        let mut model =
            crate::model::BonsaiModel::from_gguf(gguf, max_seq).expect("BonsaiModel::from_gguf");
        if !sweep_upload_redundant(gguf) {
            model.upload_weights_to_gpu(kernel);
        }
        model
            .get_or_create_gpu_cache()
            .unwrap_or_else(|e| panic!("fused GPU weight cache: {e}"));
        model
    }

    /// One timed `forward_prefill` of `prompt` at `chunk_tokens` on a fresh,
    /// warmed-up GPU model with the fused route pinned (the M-18 router would
    /// otherwise be free to measure the sequential path instead); returns the
    /// wall time and the number of fused batch-prefill calls that completed
    /// (zero means the fused path never ran).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn sweep_timed_prefill(
        gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
        kernel: &oxibonsai_kernels::KernelDispatcher,
        max_seq: usize,
        prompt: &[u32],
        chunk_tokens: usize,
    ) -> (std::time::Duration, u64) {
        use oxibonsai_kernels::gpu_backend::PrefillRoute;
        use oxibonsai_kernels::MetalGraph;
        let mut model = sweep_gpu_model(gguf, kernel, max_seq);
        model.force_metal_prefill_route(Some(PrefillRoute::Fused));
        // Warm-up: first-use weight residency and pipeline resolution are
        // paid here, not inside the timed call.
        model
            .forward_prefill(&prompt[..prompt.len().min(8)], 0, kernel)
            .expect("warm-up prefill");
        model.reset();
        model.set_prefill_chunk_tokens(chunk_tokens);
        let before = MetalGraph::prefill_fused_call_count();
        let started = std::time::Instant::now();
        let logits = model
            .forward_prefill(prompt, 0, kernel)
            .expect("fused Metal prefill on the real model");
        let elapsed = started.elapsed();
        let fused_calls = MetalGraph::prefill_fused_call_count() - before;
        model.force_metal_prefill_route(None);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "chunk_tokens={chunk_tokens}: non-finite prefill logits"
        );
        (elapsed, fused_calls)
    }

    /// The same prompt decoded one token at a time through `forward` on a
    /// fresh GPU model — the sequential path the fused prefill replaces.
    /// Returns the wall time after `short` positions and after all of them.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn sweep_sequential(
        gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
        kernel: &oxibonsai_kernels::KernelDispatcher,
        max_seq: usize,
        prompt: &[u32],
        short: usize,
    ) -> (std::time::Duration, std::time::Duration) {
        let mut model = sweep_gpu_model(gguf, kernel, max_seq);
        model
            .forward(prompt[0], 0, kernel)
            .expect("warm-up decode step");
        model.reset();
        let started = std::time::Instant::now();
        let mut at_short = None;
        for (pos, &token) in prompt.iter().enumerate() {
            let logits = model
                .forward(token, pos, kernel)
                .expect("sequential decode step");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "non-finite decode logits at {pos}"
            );
            if pos + 1 == short {
                at_short = Some(started.elapsed());
            }
        }
        let total = started.elapsed();
        (at_short.unwrap_or(total), total)
    }

    /// Microseconds per token.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn us_per_token(elapsed: std::time::Duration, tokens: usize) -> f64 {
        elapsed.as_secs_f64() * 1e6 / tokens.max(1) as f64
    }

    /// Best-effort 1/5/15-minute load average, printed beside every figure.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn sweep_load_average() -> String {
        match std::process::Command::new("uptime").output() {
            Ok(out) if out.status.success() => {
                let text = String::from_utf8_lossy(&out.stdout);
                match text.split_once("load average") {
                    Some((_, tail)) => tail.trim_start_matches([':', 's', ' ']).trim().to_string(),
                    None => text.trim().to_string(),
                }
            }
            _ => "unavailable".to_string(),
        }
    }

    /// M-18: the chunk-size sweep of the fused Metal prefill on a real model,
    /// against the sequential single-token decode it replaces.
    ///
    /// Loads `$OXI_MODEL` the way a production Metal engine does
    /// (auto-detected dispatcher, the weight upload where the route needs it,
    /// the eager fused weight cache) and, for every chunk size in
    /// [`SWEEP_CHUNK_SIZES`], prefills a `$OXIBONSAI_PREFILL_SWEEP_TOKENS`-token
    /// prompt (default [`SWEEP_DEFAULT_PROMPT_TOKENS`]) through
    /// `forward_prefill` on a fresh, warmed-up model with the fused route
    /// pinned, then decodes the same prompt one token at a time. Every arm must
    /// actually run the fused batch path (the fused-call counter moves by one
    /// per chunk); the per-token rates are printed for the
    /// `DEFAULT_PREFILL_CHUNK_TOKENS` table, and the M-18 acceptance is
    /// asserted: on the 256-token prompt the fused prefill is at least
    /// [`SWEEP_MIN_SHORT_SPEEDUP`]x faster per token than sequential decode,
    /// the whole prompt prefills faster fused than decoded, and the fused
    /// per-token cost grows at most [`SWEEP_MAX_GROWTH`]x from the short
    /// prompt to the full one.
    ///
    /// Self-skips — recording the skip — when `OXI_MODEL` is unset.
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn real_model_metal_prefill_chunk_size_sweep() {
        use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
        use oxibonsai_kernels::{KernelDispatcher, KernelTier};
        use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

        const TEST_NAME: &str = "oxibonsai-model::lib::real_model_metal_prefill_chunk_size_sweep";
        let Some(path) = std::env::var_os("OXI_MODEL").filter(|p| !p.is_empty()) else {
            eprintln!(
                "{TEST_NAME}: OXI_MODEL is not set -- skipping (set it to a real \
                 Ternary-Bonsai-1.7B / Bonsai-8B GGUF)"
            );
            record_skipped(Capability::LegacyModels, TEST_NAME);
            return;
        };
        let started = std::time::Instant::now();
        let prompt_len = std::env::var("OXIBONSAI_PREFILL_SWEEP_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n >= 2)
            .unwrap_or(SWEEP_DEFAULT_PROMPT_TOKENS);
        let mmap = mmap_gguf_file(std::path::Path::new(&path)).expect("mmap OXI_MODEL");
        let gguf = GgufFile::parse(&mmap).expect("GgufFile::parse OXI_MODEL");
        let cfg = oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata)
            .expect("Qwen3Config::from_metadata OXI_MODEL");
        let vocab = u32::try_from(cfg.vocab_size).expect("vocabulary fits u32");
        assert!(vocab > 1, "a real model has a non-trivial vocabulary");
        let n_layers = cfg.num_layers as u64;
        let max_seq = prompt_len + 64;
        let prompt: Vec<u32> = (0..prompt_len as u32)
            .map(|i| 1 + (i * 97) % (vocab - 1))
            .collect();
        let kernel = KernelDispatcher::auto_detect();
        assert_eq!(
            kernel.tier(),
            KernelTier::Gpu,
            "the sweep measures the Metal route: auto-detection must pick the GPU tier"
        );
        eprintln!(
            "M18_SWEEP model={path:?} layers={n_layers} prompt={prompt_len} load={}",
            sweep_load_average()
        );

        let mut measured: Vec<(usize, std::time::Duration)> = Vec::new();
        for &chunk in &SWEEP_CHUNK_SIZES {
            let effective = if chunk >= prompt_len { 0 } else { chunk };
            let calls = if effective == 0 {
                1
            } else {
                prompt_len.div_ceil(effective)
            };
            let elapsed = match measured.iter().find(|(c, _)| *c == effective) {
                Some(&(_, elapsed)) => elapsed,
                None => {
                    let (elapsed, fused_calls) =
                        sweep_timed_prefill(&gguf, &kernel, max_seq, &prompt, effective);
                    assert_eq!(
                        fused_calls, calls as u64,
                        "chunk_tokens={chunk}: the fused Metal batch path did not run for every \
                         call ({fused_calls} fused calls, expected {calls})"
                    );
                    measured.push((effective, elapsed));
                    elapsed
                }
            };
            eprintln!(
                "M18_SWEEP chunk_tokens={chunk} calls={calls} prefill_ms={:.1} \
                 prefill_us/token={:.1}",
                elapsed.as_secs_f64() * 1e3,
                us_per_token(elapsed, prompt_len)
            );
        }

        let short = SWEEP_SHORT_PROMPT_TOKENS.min(prompt_len);
        let (short_fused, short_calls) =
            sweep_timed_prefill(&gguf, &kernel, max_seq, &prompt[..short], 0);
        assert_eq!(short_calls, 1, "the {short}-token fused prefill ran");
        let (seq_short, seq_total) = sweep_sequential(&gguf, &kernel, max_seq, &prompt, short);
        let fused_short_us = us_per_token(short_fused, short);
        let seq_short_us = us_per_token(seq_short, short);
        let seq_total_us = us_per_token(seq_total, prompt_len);
        let fused_full = measured
            .iter()
            .find(|(c, _)| *c == 0)
            .map(|&(_, e)| e)
            .expect("the one-shot arm always runs");
        let fused_full_us = us_per_token(fused_full, prompt_len);
        eprintln!(
            "M18_SWEEP short prompt={short} fused_us/token={fused_short_us:.1} \
             sequential_us/token={seq_short_us:.1} ratio={:.2}x",
            seq_short_us / fused_short_us.max(f64::MIN_POSITIVE)
        );
        let growth = fused_full_us / fused_short_us.max(f64::MIN_POSITIVE);
        eprintln!(
            "M18_SWEEP full prompt={prompt_len} fused_ms={:.1} fused_us/token={fused_full_us:.1} \
             sequential_ms={:.1} sequential_us/token={seq_total_us:.1} \
             fused_growth_vs_short={growth:.2}x load={}",
            fused_full.as_secs_f64() * 1e3,
            seq_total.as_secs_f64() * 1e3,
            sweep_load_average()
        );
        // M-18 acceptance: the fused prefill beats sequential decode by 3x per
        // token on a short prompt, beats it outright on the whole prompt, and
        // stays near-linear (the attention term only) up to the full length.
        assert!(
            seq_short_us >= SWEEP_MIN_SHORT_SPEEDUP * fused_short_us,
            "{short} tokens: the fused prefill ({fused_short_us:.1} us/token) must be at least \
             {SWEEP_MIN_SHORT_SPEEDUP}x faster than sequential decode ({seq_short_us:.1} us/token)"
        );
        assert!(
            fused_full < seq_total,
            "{prompt_len} tokens: the fused prefill ({fused_full:?}) must beat sequential decode \
             ({seq_total:?})"
        );
        assert!(
            growth <= SWEEP_MAX_GROWTH,
            "the fused per-token cost grew {growth:.2}x from {short} to {prompt_len} tokens, above \
             {SWEEP_MAX_GROWTH}x"
        );
        record_executed_timed(Capability::LegacyModels, TEST_NAME, started.elapsed());
    }
}
