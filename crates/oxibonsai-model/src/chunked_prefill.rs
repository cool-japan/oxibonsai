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
//! [`run_chunked_prefill`]'s doc comment for why), cut by
//! [`prefill_chunk_windows`], which never leaves a short final chunk after
//! a batched one (F-3, [`PREFILL_PER_TOKEN_MAX_TOKENS`]).

use std::ops::Range;

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
        Self::from_chunks(config, chunks)
    }

    /// A scheduler over chunks the caller already cut (the
    /// [`run_chunked_prefill`] executor's [`prefill_chunk_windows`] split).
    fn from_chunks(config: ChunkedPrefillConfig, chunks: Vec<PrefillChunk>) -> Self {
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
/// the historical 4096-token context, 4096-token chunks beyond it (the last
/// call up to 4096 + [`PREFILL_PER_TOKEN_MAX_TOKENS`] tokens, see
/// [`prefill_chunk_windows`]; a 4097..=4112-token prompt is one call). It is
/// ratified together with the prefill router in `forward_metal.rs`, which
/// keeps a model whose fused path measures slower than decode on the
/// sequential route and bounds every fused call by a deadline.
///
/// Callers that *want* smaller chunks set them through
/// [`crate::model::BonsaiModel::set_prefill_chunk_tokens`]; the runtime's own
/// `InferenceEngine::set_prefill_chunk_tokens` is the same knob one level up.
pub const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 4096;

/// Longest prefill window [`crate::model::BonsaiModel::forward_prefill`] runs
/// on a GPU tier's per-token path rather than its batch kernels.
///
/// On a `native-cuda` build `forward_prefill_unchunked`
/// (`model/types/prefill_dispatch.rs`) hands a GPU window of at most this
/// many tokens to `forward_sequential`, a loop of per-token `forward` calls;
/// only a longer window reaches the CUDA batch prefill (which is also why
/// the runtime's CUDA warm-up prefills one token more than this). Before its
/// loop, `forward_sequential` requires the host KV cache to be coherent up
/// to the window's start (the MET-05 guard). The CUDA batch prefill of a Q1
/// or ternary model keeps the prompt's K/V in the device KV cache only, so a
/// 2..=16-token window that follows such a batch window is refused with
/// [`crate::error::ModelError::GpuFallbackRequiresCacheRebuild`] (F-3, found
/// for Q1 and ternary models in the 2026-10-07 RTX A4000 validation run).
/// [`prefill_chunk_windows`] therefore never ends a multi-window plan with a
/// window this short when the chunk is longer than this; the runtime's
/// `InferenceEngine` plans its own prefill windows the same way.
pub const PREFILL_PER_TOKEN_MAX_TOKENS: usize = 16;

/// Cut a `prompt_len`-token prompt into the contiguous, non-overlapping
/// windows a chunked prefill at `chunk_tokens` runs, one prefill call each.
///
/// The windows are those of `prompt.chunks(chunk_tokens)`, except that when
/// `chunk_tokens` exceeds [`PREFILL_PER_TOKEN_MAX_TOKENS`] a final window of
/// `1..=PREFILL_PER_TOKEN_MAX_TOKENS` tokens is merged into the window
/// before it, which then holds at most `chunk_tokens + 16` tokens. Every
/// earlier window took the batched path, so on a `native-cuda` build a
/// 2..=16-token tail would run on the per-token path, which reads the host
/// KV cache that a device-resident CUDA batch window of a Q1 or ternary
/// model never populated, and be refused (F-3; see
/// [`PREFILL_PER_TOKEN_MAX_TOKENS`]). A 1-token tail runs the decode-style
/// `forward` and would work, but merging it too keeps one invariant: with
/// more than one window, every window but the last holds exactly
/// `chunk_tokens` tokens and the last more than
/// [`PREFILL_PER_TOKEN_MAX_TOKENS`].
///
/// With `chunk_tokens <= PREFILL_PER_TOKEN_MAX_TOKENS` no window takes the
/// batched path, so no batched window precedes the tail and the plan is
/// exactly `chunks(chunk_tokens)`: merging there would instead create a
/// per-token to batched transition at a non-zero position.
///
/// `chunk_tokens == 0` (chunking disabled) or `chunk_tokens >= prompt_len`
/// gives the single window `0..prompt_len`, for an empty prompt too.
pub fn prefill_chunk_windows(prompt_len: usize, chunk_tokens: usize) -> Vec<Range<usize>> {
    if chunk_tokens == 0 || chunk_tokens >= prompt_len {
        return std::iter::once(0..prompt_len).collect();
    }
    let mut windows: Vec<Range<usize>> = (0..prompt_len)
        .step_by(chunk_tokens)
        .map(|start| start..(start + chunk_tokens).min(prompt_len))
        .collect();
    let short_tail = windows
        .last()
        .is_some_and(|tail| tail.len() <= PREFILL_PER_TOKEN_MAX_TOKENS);
    if chunk_tokens > PREFILL_PER_TOKEN_MAX_TOKENS && short_tail {
        if let Some(tail) = windows.pop() {
            if let Some(previous) = windows.last_mut() {
                previous.end = tail.end;
            }
        }
    }
    windows
}

/// Decide whether a prompt of `prompt_len` tokens should be chunked at
/// `chunk_tokens`, and with what configuration (M-18).
///
/// Returns `None` — "run it in one shot, exactly as before" — when chunking
/// would not actually split the prompt: an empty or single-token prompt, a
/// disabled (`0`) chunk size, or a prompt that [`prefill_chunk_windows`]
/// keeps in one window (at most `chunk_tokens` tokens, or at most
/// `chunk_tokens + PREFILL_PER_TOKEN_MAX_TOKENS` once the chunk exceeds
/// [`PREFILL_PER_TOKEN_MAX_TOKENS`], F-3). That is what keeps the Metal
/// fused batch path's dispatch granularity unchanged for every short prompt.
///
/// The returned config carries the plain chunk size; it does not encode the
/// short-tail merge. [`run_chunked_prefill`] applies the merge itself (it
/// cuts its chunks with [`prefill_chunk_windows`]), whereas
/// [`create_prefill_chunks`] / [`PrefillScheduler`] fed this config split
/// plainly.
pub fn prefill_chunk_plan(prompt_len: usize, chunk_tokens: usize) -> Option<ChunkedPrefillConfig> {
    if chunk_tokens == 0
        || prompt_len <= 1
        || prefill_chunk_windows(prompt_len, chunk_tokens).len() <= 1
    {
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
/// **F-3: no short final chunk after a batched one.** The chunks are the
/// windows of [`prefill_chunk_windows`] at `config.chunk_size` (at least
/// 1): when the chunk size exceeds [`PREFILL_PER_TOKEN_MAX_TOKENS`], a
/// final chunk of at most that many tokens is merged into the chunk before
/// it, so the last chunk can hold up to `chunk_size + 16` tokens. A
/// `native-cuda` Q1 or ternary model would otherwise run that tail on its
/// per-token path, which reads the host KV cache the device-resident batch
/// chunks never populated, and refuse it. Every other chunk is exactly
/// what [`create_prefill_chunks`] would cut with `overlap: 0`.
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
    // F-3: cut with the short-tail merge (see the doc comment above).
    let windows = prefill_chunk_windows(prompt_tokens.len(), config.chunk_size.max(1));
    let last_index = windows.len().saturating_sub(1);
    let chunks = windows
        .into_iter()
        .enumerate()
        .map(|(chunk_index, window)| PrefillChunk {
            tokens: prompt_tokens[window.clone()].to_vec(),
            start_pos: window.start,
            end_pos: window.end,
            chunk_index,
            is_last: chunk_index == last_index,
        })
        .collect();
    let mut scheduler = PrefillScheduler::from_chunks(config, chunks);
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
        for prompt_len in 4097..=4096 + PREFILL_PER_TOKEN_MAX_TOKENS {
            assert!(
                prefill_chunk_plan(prompt_len, 4096).is_none(),
                "F-3: a {prompt_len}-token prompt's short tail is merged into one window"
            );
        }
        assert!(
            prefill_chunk_plan(40, PREFILL_PER_TOKEN_MAX_TOKENS).is_some(),
            "a chunk of at most the per-token threshold merges nothing"
        );
    }

    #[test]
    fn chunk_plan_splits_a_prompt_that_does_not_fit() {
        let prompt_len = 4096 + PREFILL_PER_TOKEN_MAX_TOKENS + 1;
        let cfg = prefill_chunk_plan(prompt_len, 4096).expect("should chunk");
        assert_eq!(cfg.chunk_size, 4096);
        assert_eq!(
            cfg.overlap, 0,
            "overlap is never safe for KV-cache execution"
        );
        assert_eq!(cfg.priority, PrefillPriority::PrefillFirst);
        let chunks = create_prefill_chunks(&vec![0u32; prompt_len], &cfg);
        assert_eq!(chunks.len(), 2);
    }

    // ── F-3: no short final prefill window after a batched one ──────────────

    /// [`prefill_chunk_windows`] as `(start, end)` pairs.
    fn window_bounds(prompt_len: usize, chunk: usize) -> Vec<(usize, usize)> {
        prefill_chunk_windows(prompt_len, chunk)
            .into_iter()
            .map(|window| (window.start, window.end))
            .collect()
    }

    #[test]
    fn chunk_windows_fold_a_short_tail_into_the_previous_window() {
        let cases = [
            (512, 512, vec![(0, 512)]),
            (513, 512, vec![(0, 513)]),
            (528, 512, vec![(0, 528)]),
            (529, 512, vec![(0, 512), (512, 529)]),
            (1040, 512, vec![(0, 512), (512, 1040)]),
            (1041, 512, vec![(0, 512), (512, 1024), (1024, 1041)]),
            (4097, 4096, vec![(0, 4097)]),
            (4113, 4096, vec![(0, 4096), (4096, 4113)]),
            (100, 512, vec![(0, 100)]),
            (100, 0, vec![(0, 100)]),
            (100, 100, vec![(0, 100)]),
            (0, 512, vec![(0, 0)]),
            (1, 512, vec![(0, 1)]),
        ];
        for (prompt_len, chunk, want) in cases {
            assert_eq!(
                window_bounds(prompt_len, chunk),
                want,
                "prompt_len {prompt_len}, chunk {chunk}"
            );
        }
        for prompt_len in 4097..=4112 {
            assert_eq!(
                window_bounds(prompt_len, DEFAULT_PREFILL_CHUNK_TOKENS),
                vec![(0, prompt_len)],
                "the default chunk keeps a {prompt_len}-token prompt in one window"
            );
        }
    }

    #[test]
    fn chunk_windows_at_or_below_the_per_token_threshold_split_plainly() {
        assert_eq!(
            window_bounds(40, 2),
            (0..40).step_by(2).map(|s| (s, s + 2)).collect::<Vec<_>>()
        );
        assert_eq!(
            window_bounds(40, PREFILL_PER_TOKEN_MAX_TOKENS),
            vec![(0, 16), (16, 32), (32, 40)]
        );
    }

    #[test]
    fn chunk_windows_cover_the_prompt_and_keep_their_invariants() {
        for chunk in 1..=40usize {
            for prompt_len in 0..=200usize {
                let plan = prefill_chunk_windows(prompt_len, chunk);
                let context = format!("prompt_len {prompt_len}, chunk {chunk}: {plan:?}");
                // Contiguous, starting at 0 and ending at the prompt's end.
                assert_eq!(plan.first().map(|w| w.start), Some(0), "{context}");
                assert_eq!(plan.last().map(|w| w.end), Some(prompt_len), "{context}");
                assert!(
                    plan.windows(2).all(|pair| pair[0].end == pair[1].start),
                    "{context}"
                );
                let (last, rest) = match plan.split_last() {
                    Some(split) => split,
                    None => panic!("{context}: a plan always has a window"),
                };
                // No window but the last is shorter than the chunk.
                assert!(rest.iter().all(|w| w.len() == chunk), "{context}");
                assert_eq!(
                    plan.len() == 1,
                    prefill_chunk_plan(prompt_len, chunk).is_none(),
                    "{context}: prefill_chunk_plan must agree on single-window prompts"
                );
                if chunk > PREFILL_PER_TOKEN_MAX_TOKENS {
                    assert!(
                        last.len() <= chunk + PREFILL_PER_TOKEN_MAX_TOKENS,
                        "{context}"
                    );
                    if !rest.is_empty() {
                        assert!(last.len() > PREFILL_PER_TOKEN_MAX_TOKENS, "{context}");
                    }
                } else if prompt_len > chunk {
                    // No window takes the batched path: `chunks(chunk)`
                    // verbatim.
                    let plain: Vec<Range<usize>> = (0..prompt_len)
                        .step_by(chunk)
                        .map(|start| start..(start + chunk).min(prompt_len))
                        .collect();
                    assert_eq!(plan, plain, "{context}");
                }
            }
        }
    }

    #[test]
    fn run_chunked_prefill_folds_a_short_tail_chunk() {
        for (prompt_len, want) in [
            (513usize, vec![(0usize, 513usize)]),
            (529, vec![(0, 512), (512, 529)]),
            (1040, vec![(0, 512), (512, 1040)]),
        ] {
            let tokens: Vec<u32> = (0..prompt_len as u32).collect();
            let mut seen: Vec<(usize, usize, bool)> = Vec::new();
            let result = run_chunked_prefill(
                &tokens,
                ChunkedPrefillConfig::new(512),
                |chunk| {
                    assert_eq!(
                        chunk.tokens.as_slice(),
                        &tokens[chunk.start_pos..chunk.end_pos]
                    );
                    seen.push((chunk.start_pos, chunk.end_pos, chunk.is_last));
                    Ok(chunk.chunk_index)
                },
                || Ok(()),
            )
            .expect("should succeed");
            let bounds: Vec<(usize, usize)> = seen.iter().map(|&(s, e, _)| (s, e)).collect();
            assert_eq!(bounds, want, "prompt_len {prompt_len}");
            let last_flags: Vec<bool> = seen.iter().map(|&(_, _, last)| last).collect();
            let mut want_flags = vec![false; want.len()];
            if let Some(flag) = want_flags.last_mut() {
                *flag = true;
            }
            assert_eq!(last_flags, want_flags, "prompt_len {prompt_len}");
            assert_eq!(result, Some(want.len() - 1), "prompt_len {prompt_len}");
        }
        // A chunk at or below the threshold keeps its plain split.
        let tokens: Vec<u32> = (0..40).collect();
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let _ = run_chunked_prefill(
            &tokens,
            ChunkedPrefillConfig::new(PREFILL_PER_TOKEN_MAX_TOKENS),
            |chunk| {
                seen.push((chunk.start_pos, chunk.end_pos));
                Ok(())
            },
            || Ok(()),
        )
        .expect("should succeed");
        assert_eq!(seen, vec![(0, 16), (16, 32), (32, 40)]);
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
    /// short prompt's (the attention term is the only super-linear part), for
    /// a model of [`SWEEP_SMALL_MODEL_HIDDEN`] hidden units or more (see
    /// [`sweep_max_growth`]). Both sides of the ratio are best-of-
    /// [`SWEEP_RUNS`] minima. The Bonsai-8B measured 1.19-1.27x.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const SWEEP_MAX_GROWTH: f64 = 1.5;

    /// The growth bound for a model below [`SWEEP_SMALL_MODEL_HIDDEN`] hidden
    /// units. The attention term weighs more against the projections of a
    /// small model, so its growth is higher: the Ternary-Bonsai-1.7B measured
    /// 1.35x-1.43x over four sweeps at load averages from 6 to 163 (1.41x and
    /// 1.43x as single runs, 1.42x and 1.35x as best of 3). Best-of-N narrows
    /// the spread of one arm to ~2% (the prefill is GPU-bound) but not the
    /// spread between sweeps, which moves with whatever else the GPU is doing,
    /// so the bound is wider as well: 1.65x leaves ~15% over the worst figure
    /// measured and stays far under the 3.2x (84 to 271 ms/token from 256 to
    /// 4096 rows) of the row-wise kernel it guards against.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const SWEEP_MAX_GROWTH_SMALL_MODEL: f64 = 1.65;

    /// Hidden size below which a model counts as small for
    /// [`SWEEP_MAX_GROWTH_SMALL_MODEL`] (the 1.7B has 2048, the 8B 4096).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const SWEEP_SMALL_MODEL_HIDDEN: usize = 4096;

    /// The per-token growth bound for a model of `hidden_size`.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn sweep_max_growth(hidden_size: usize) -> f64 {
        if hidden_size < SWEEP_SMALL_MODEL_HIDDEN {
            SWEEP_MAX_GROWTH_SMALL_MODEL
        } else {
            SWEEP_MAX_GROWTH
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn sweep_growth_bound_follows_the_model_size() {
        assert_eq!(sweep_max_growth(2048), SWEEP_MAX_GROWTH_SMALL_MODEL);
        assert_eq!(sweep_max_growth(4095), SWEEP_MAX_GROWTH_SMALL_MODEL);
        assert_eq!(sweep_max_growth(4096), SWEEP_MAX_GROWTH);
        assert_eq!(sweep_max_growth(5120), SWEEP_MAX_GROWTH);
        const { assert!(SWEEP_MAX_GROWTH_SMALL_MODEL > SWEEP_MAX_GROWTH) };
    }

    /// Timed runs of every distinct prefill arm (each on a fresh, warmed-up
    /// model); every figure the sweep asserts on is the minimum of them — the
    /// sample closest to the machine's own floor on a host whose load average
    /// moved between 6 and 23 within one measured sweep.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    const SWEEP_RUNS: usize = 3;

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

    /// [`sweep_timed_prefill`] [`SWEEP_RUNS`] times: the fastest run, every
    /// run's wall time (in run order) and the number of fused batch-prefill
    /// calls each run completed — which must be the same every run.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn sweep_best_prefill(
        gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
        kernel: &oxibonsai_kernels::KernelDispatcher,
        max_seq: usize,
        prompt: &[u32],
        chunk_tokens: usize,
    ) -> (std::time::Duration, Vec<std::time::Duration>, u64) {
        let mut runs = Vec::with_capacity(SWEEP_RUNS);
        let mut fused_calls = None;
        for run in 0..SWEEP_RUNS {
            let (elapsed, calls) = sweep_timed_prefill(gguf, kernel, max_seq, prompt, chunk_tokens);
            if let Some(first) = fused_calls {
                assert_eq!(
                    calls, first,
                    "chunk_tokens={chunk_tokens}: run {run} completed {calls} fused calls, run 0 \
                     completed {first}"
                );
            }
            fused_calls = Some(calls);
            runs.push(elapsed);
        }
        let best = runs.iter().copied().min().unwrap_or_default();
        (best, runs, fused_calls.unwrap_or(0))
    }

    /// Wall times in milliseconds, comma-separated, for the sweep's log lines.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn sweep_runs_ms(runs: &[std::time::Duration]) -> String {
        runs.iter()
            .map(|d| format!("{:.1}", d.as_secs_f64() * 1e3))
            .collect::<Vec<_>>()
            .join(", ")
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
    /// per-token cost grows at most [`sweep_max_growth`]x (1.5x, 1.65x for a
    /// model under [`SWEEP_SMALL_MODEL_HIDDEN`] hidden units) from the short
    /// prompt to the full one.
    ///
    /// Every fused arm (each distinct chunk size and the short prompt) runs
    /// [`SWEEP_RUNS`] times and the figures are the minima, so a load spike
    /// during one run does not decide the verdict; the one sequential decode
    /// runs once (its margin over the fused arms is an order of magnitude).
    /// The load average is printed beside every arm and with the verdict, and
    /// repeated in every assertion message, so a failure reads against the
    /// load it was measured under.
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
        let max_growth = sweep_max_growth(cfg.hidden_size);
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
            // One fused call per planned window (F-3: a short final window
            // is merged into the one before it).
            let calls = prefill_chunk_windows(prompt_len, effective).len();
            let elapsed = match measured.iter().find(|(c, _)| *c == effective) {
                Some(&(_, elapsed)) => elapsed,
                None => {
                    let (elapsed, runs, fused_calls) =
                        sweep_best_prefill(&gguf, &kernel, max_seq, &prompt, effective);
                    assert_eq!(
                        fused_calls, calls as u64,
                        "chunk_tokens={chunk}: the fused Metal batch path did not run for every \
                         call ({fused_calls} fused calls, expected {calls})"
                    );
                    eprintln!(
                        "M18_SWEEP chunk_tokens={chunk} runs_ms=[{}] load={}",
                        sweep_runs_ms(&runs),
                        sweep_load_average()
                    );
                    measured.push((effective, elapsed));
                    elapsed
                }
            };
            eprintln!(
                "M18_SWEEP chunk_tokens={chunk} calls={calls} best_prefill_ms={:.1} \
                 best_prefill_us/token={:.1}",
                elapsed.as_secs_f64() * 1e3,
                us_per_token(elapsed, prompt_len)
            );
        }

        let short = SWEEP_SHORT_PROMPT_TOKENS.min(prompt_len);
        let (short_fused, short_runs, short_calls) =
            sweep_best_prefill(&gguf, &kernel, max_seq, &prompt[..short], 0);
        assert_eq!(short_calls, 1, "the {short}-token fused prefill ran");
        eprintln!(
            "M18_SWEEP short prompt={short} runs_ms=[{}] load={}",
            sweep_runs_ms(&short_runs),
            sweep_load_average()
        );
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
        let load = sweep_load_average();
        eprintln!(
            "M18_SWEEP full prompt={prompt_len} fused_ms={:.1} fused_us/token={fused_full_us:.1} \
             sequential_ms={:.1} sequential_us/token={seq_total_us:.1} \
             fused_growth_vs_short={growth:.2}x (best of {SWEEP_RUNS}, bound {max_growth}x for \
             hidden size {}) load={load}",
            fused_full.as_secs_f64() * 1e3,
            seq_total.as_secs_f64() * 1e3,
            cfg.hidden_size,
        );
        // M-18 acceptance: the fused prefill beats sequential decode by 3x per
        // token on a short prompt, beats it outright on the whole prompt, and
        // stays near-linear (the attention term only) up to the full length.
        assert!(
            seq_short_us >= SWEEP_MIN_SHORT_SPEEDUP * fused_short_us,
            "{short} tokens: the fused prefill ({fused_short_us:.1} us/token) must be at least \
             {SWEEP_MIN_SHORT_SPEEDUP}x faster than sequential decode ({seq_short_us:.1} \
             us/token); load average {load}"
        );
        assert!(
            fused_full < seq_total,
            "{prompt_len} tokens: the fused prefill ({fused_full:?}) must beat sequential decode \
             ({seq_total:?}); load average {load}"
        );
        assert!(
            growth <= max_growth,
            "the fused per-token cost grew {growth:.2}x (best of {SWEEP_RUNS}) from {short} to \
             {prompt_len} tokens, above {max_growth}x (hidden size {}); load average {load}",
            cfg.hidden_size
        );
        record_executed_timed(Capability::LegacyModels, TEST_NAME, started.elapsed());
    }
}
