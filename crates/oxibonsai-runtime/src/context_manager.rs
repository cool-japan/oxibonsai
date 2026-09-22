//! Context window management for multi-turn inference.
//!
//! Manages the token budget across conversation turns, supporting multiple
//! truncation strategies when context exceeds the model's maximum sequence length.
//!
//! ## Truncation Strategies
//!
//! - [`TruncationStrategy::TruncateLeft`] — drops the oldest conversation tokens (default).
//!   System prompt tokens are always preserved.
//! - [`TruncationStrategy::TruncateRight`] — drops the newest conversation tokens.
//! - [`TruncationStrategy::SlidingWindow`] — keeps system prompt plus a
//!   *fixed-size* recent window (see [`ContextWindow::set_sliding_window_size`]),
//!   batch-evicting down to that window instead of the bare minimum
//!   `TruncateLeft` removes. Unconfigured, it defaults to the same window
//!   size as the available capacity (byte-for-byte the same eviction count
//!   as `TruncateLeft`) — the knob, not the default, is what makes the two
//!   strategies genuinely distinct (RT-16).
//! - [`TruncationStrategy::Summarize`] — compresses the evicted span instead
//!   of discarding it: a real (if crude, by default) extractive compression
//!   keeps a lead+tail gist of what would otherwise be lost, and a caller
//!   with a model in hand can attach a real LLM-backed
//!   [`Summarizer`] via [`ContextWindow::set_summarizer`] for genuine
//!   recursive summarisation (RT-16 — this variant used to be a silent
//!   alias for `TruncateLeft`).
//!
//! ## Usage
//!
//! ```rust
//! use oxibonsai_runtime::context_manager::{ContextWindow, TruncationStrategy};
//!
//! let mut window = ContextWindow::new(2048, TruncationStrategy::TruncateLeft);
//! window.set_system_prompt(vec![1, 2, 3]).expect("system prompt fits");
//! window.append(&[10, 20, 30]);
//! let tokens = window.tokens();
//! assert!(tokens.len() <= 2048);
//! ```
//!
//! ### Wiring a real, model-backed summarizer
//!
//! ```rust
//! use oxibonsai_runtime::context_manager::{ContextWindow, TruncationStrategy};
//!
//! // In a real deployment this closure would decode `tokens` to text with
//! // the loaded tokenizer, prompt the loaded model with a fixed
//! // summarisation instruction, and re-encode the reply — "the engine
//! // needed to summarise is the one already in the process" (RT-16). This
//! // module has no tokenizer/model dependency, so it only defines the seam;
//! // the toy closure below just keeps the first token as a stand-in gist.
//! let mut window = ContextWindow::new(8, TruncationStrategy::Summarize);
//! window.set_summarizer(|tokens: &[u32], budget: usize| {
//!     Ok(tokens.iter().take(budget.max(1)).copied().collect())
//! });
//! window.append(&(0u32..20).collect::<Vec<_>>());
//! assert!(window.len() <= 8);
//! ```

use std::sync::Arc;

/// Minimum number of evicted tokens before `Summarize` bothers producing a
/// compressed replacement; below this a summary could not be meaningfully
/// shorter than the input, so the evicted span is simply dropped exactly as
/// `TruncateLeft` would for that call.
const MIN_TOKENS_TO_SUMMARIZE: usize = 4;

/// Target compression ratio for the built-in extractive summarizer: the
/// replacement is at most `evicted.len() / SUMMARY_DIVISOR` tokens (at least
/// one), guaranteeing real compression whenever summarisation triggers.
const SUMMARY_DIVISOR: usize = 4;

// ──────────────────────────────────────────────────────────────────
// Truncation strategy
// ──────────────────────────────────────────────────────────────────

/// Strategy for handling context that exceeds the maximum token budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Drop oldest conversation tokens (default). System prompt is never removed.
    TruncateLeft,
    /// Drop newest conversation tokens. System prompt is never removed.
    TruncateRight,
    /// Keep the system prompt plus a fixed-size recent window, batch-evicting
    /// down to that window (dropping the middle) rather than the bare
    /// minimum needed to fit. See [`ContextWindow::set_sliding_window_size`].
    SlidingWindow,
    /// Compress the evicted span instead of discarding it — see the module
    /// docs and [`ContextWindow::set_summarizer`].
    Summarize,
}

// ──────────────────────────────────────────────────────────────────
// Context error
// ──────────────────────────────────────────────────────────────────

/// Error type for context window operations.
#[derive(Debug)]
pub struct ContextError(String);

impl ContextError {
    /// Construct a `ContextError` with the given message.
    ///
    /// Public so that a caller-supplied [`Summarizer`] (implemented outside
    /// this crate, or in a higher layer of this crate that has a real
    /// tokenizer/model in hand) can report its own failures through the same
    /// error type this module uses internally.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContextError: {}", self.0)
    }
}

impl std::error::Error for ContextError {}

// ──────────────────────────────────────────────────────────────────
// Summarizer (RT-16)
// ──────────────────────────────────────────────────────────────────

/// A pluggable compressor for the token span [`TruncationStrategy::Summarize`]
/// is about to evict.
///
/// `oxibonsai_runtime::context_manager` works purely on token ids and has no
/// tokenizer or model dependency, so it cannot itself run "recursive
/// summarisation through the model" — that requires decoding ids to text,
/// prompting the loaded model with a summarisation instruction, and
/// re-encoding the reply, all of which live in higher layers. This trait is
/// the seam: a caller that *does* have a tokenizer and a model in hand
/// implements it (or, more simply, provides the equivalent closure — see the
/// blanket impl below) and attaches it with
/// [`ContextWindow::set_summarizer`]/[`ContextWindow::with_summarizer`] to
/// make `Summarize` genuinely LLM-backed.
///
/// Without one attached, `Summarize` still performs a real, deterministic,
/// non-verbatim compression (see the private `default_extractive_summary`)
/// — it never silently falls back to discarding the span the way
/// `TruncateLeft` would.
pub trait Summarizer: Send + Sync {
    /// Compress `tokens` (the conversation span about to be evicted) into a
    /// token sequence of at most `budget` tokens that still carries some of
    /// its information. Implementations may run a full model generation;
    /// they should return `Err` rather than panic on failure — a failure is
    /// still handled gracefully by [`ContextWindow::append`] (it falls back
    /// to the built-in extractive summary rather than losing the eviction
    /// entirely), but [`ContextWindow::try_append`] surfaces it to the
    /// caller.
    fn summarize(&self, tokens: &[u32], budget: usize) -> Result<Vec<u32>, ContextError>;
}

impl<F> Summarizer for F
where
    F: Fn(&[u32], usize) -> Result<Vec<u32>, ContextError> + Send + Sync,
{
    fn summarize(&self, tokens: &[u32], budget: usize) -> Result<Vec<u32>, ContextError> {
        self(tokens, budget)
    }
}

/// Deterministic, dependency-free default for [`TruncationStrategy::Summarize`]
/// when no [`Summarizer`] is attached.
///
/// Keeps a "lead + tail" gist of the evicted span: the first half of
/// `budget` tokens from the start of `tokens` and the second half from its
/// end, which are the two sub-spans most likely to carry topic-setting and
/// concluding information respectively. Each half is a genuine contiguous
/// sub-slice of the original (BPE-token-boundary-safe — no fabricated
/// tokens), unlike, say, uniformly downsampling every Nth token, which would
/// splice together unrelated token fragments into something that decodes as
/// noise rather than a lossy-but-real gist.
fn default_extractive_summary(tokens: &[u32], budget: usize) -> Vec<u32> {
    if budget == 0 || tokens.is_empty() {
        return Vec::new();
    }
    if tokens.len() <= budget {
        return tokens.to_vec();
    }
    let head_len = budget.div_ceil(2);
    let tail_len = budget - head_len;
    let mut out = Vec::with_capacity(budget);
    out.extend_from_slice(&tokens[..head_len]);
    if tail_len > 0 {
        out.extend_from_slice(&tokens[tokens.len() - tail_len..]);
    }
    out
}

// ──────────────────────────────────────────────────────────────────
// ContextWindow
// ──────────────────────────────────────────────────────────────────

/// A fixed-capacity token window with configurable truncation.
///
/// The window stores a protected *system prompt* (never truncated) and
/// a mutable *conversation* segment. Together they must fit within
/// `max_tokens`.
pub struct ContextWindow {
    /// Maximum total token count (system + conversation).
    pub max_tokens: usize,
    /// Tokens belonging to the system prompt — never truncated.
    pub system_tokens: Vec<u32>,
    /// Accumulated conversation tokens.
    pub conversation: Vec<u32>,
    /// How to truncate when the window is full.
    pub strategy: TruncationStrategy,
    /// [`TruncationStrategy::SlidingWindow`]'s fixed recent-window size, in
    /// tokens. `None` (the default) falls back to the full available
    /// capacity — i.e. the same eviction count as `TruncateLeft` — so an
    /// *unconfigured* `SlidingWindow` never surprises an existing caller
    /// with extra eviction; set it with
    /// [`ContextWindow::set_sliding_window_size`] to make the batching
    /// behaviour (evict down to a smaller fixed window, dropping the
    /// middle, instead of the bare minimum) actually kick in.
    sliding_window_size: Option<usize>,
    /// Optional real summarizer for [`TruncationStrategy::Summarize`]. See
    /// [`ContextWindow::set_summarizer`].
    summarizer: Option<Arc<dyn Summarizer>>,
}

impl ContextWindow {
    /// Create a new empty context window.
    pub fn new(max_tokens: usize, strategy: TruncationStrategy) -> Self {
        Self {
            max_tokens,
            system_tokens: Vec::new(),
            conversation: Vec::new(),
            strategy,
            sliding_window_size: None,
            summarizer: None,
        }
    }

    /// Configure [`TruncationStrategy::SlidingWindow`]'s fixed recent-window
    /// size, in tokens. Builder-style consuming setter.
    ///
    /// Once the conversation overflows, `SlidingWindow` batch-evicts down to
    /// this many tokens (capped to the available capacity) in one shot,
    /// rather than the bare minimum `TruncateLeft` removes on every call —
    /// this is the "drop the middle, keep a fixed window" behaviour that
    /// makes the strategy genuinely distinct (RT-16).
    pub fn with_sliding_window_size(mut self, size: usize) -> Self {
        self.sliding_window_size = Some(size);
        self
    }

    /// Same as [`Self::with_sliding_window_size`], for a window already in
    /// use (non-consuming).
    pub fn set_sliding_window_size(&mut self, size: usize) {
        self.sliding_window_size = Some(size);
    }

    /// Attach a real [`Summarizer`] for [`TruncationStrategy::Summarize`].
    /// Builder-style consuming setter. Accepts anything implementing
    /// [`Summarizer`], including a plain closure
    /// `Fn(&[u32], usize) -> Result<Vec<u32>, ContextError>`.
    pub fn with_summarizer(mut self, summarizer: impl Summarizer + 'static) -> Self {
        self.summarizer = Some(Arc::new(summarizer));
        self
    }

    /// Same as [`Self::with_summarizer`], for a window already in use
    /// (non-consuming).
    pub fn set_summarizer(&mut self, summarizer: impl Summarizer + 'static) {
        self.summarizer = Some(Arc::new(summarizer));
    }

    /// Detach any previously attached [`Summarizer`], reverting `Summarize`
    /// to the built-in extractive default.
    pub fn clear_summarizer(&mut self) {
        self.summarizer = None;
    }

    /// Whether a real [`Summarizer`] is currently attached.
    pub fn has_summarizer(&self) -> bool {
        self.summarizer.is_some()
    }

    /// Set the system prompt tokens.
    ///
    /// Returns an error if the system prompt alone exceeds `max_tokens`.
    pub fn set_system_prompt(&mut self, tokens: Vec<u32>) -> Result<(), ContextError> {
        if tokens.len() > self.max_tokens {
            return Err(ContextError::new(format!(
                "system prompt ({} tokens) exceeds max_tokens ({})",
                tokens.len(),
                self.max_tokens
            )));
        }
        self.system_tokens = tokens;
        // Truncate conversation if system prompt now leaves no room
        self.truncate_to_fit();
        Ok(())
    }

    /// Append tokens to the conversation.
    ///
    /// If the new tokens cause the window to overflow, truncation is applied
    /// *before* appending as much of the new tokens as will fit.
    ///
    /// Returns the number of tokens actually appended — i.e. how many of
    /// `tokens` are still present, verbatim, in [`Self::conversation`] after
    /// truncation. Under `Summarize`, when the pre-append conversation is
    /// shorter than the strategy's internal summary budget, a small number
    /// of the just-appended tokens can be folded into the generated summary
    /// rather than kept verbatim; this method still returns a safe count in
    /// that edge case (it does not double-count a summarized token as both
    /// "removed" and "kept"), but it is not the sharpest possible bound —
    /// use [`Self::try_append`] plus [`Self::len`] if that distinction
    /// matters to the caller.
    ///
    /// A [`Summarizer`] attached via [`Self::set_summarizer`] that returns
    /// `Err` is handled gracefully here: the built-in extractive summary is
    /// used for that call instead (the `Summarize` contract — a real,
    /// shorter, non-verbatim replacement for the evicted span — is still
    /// honoured), so this method never fails. Use [`Self::try_append`] to
    /// observe the underlying error instead.
    pub fn append(&mut self, tokens: &[u32]) -> usize {
        let old_len = self.conversation.len();
        self.conversation.extend_from_slice(tokens);
        let removed = self.truncate_to_fit_fallible(true).unwrap_or_else(|_| {
            // Unreachable when `allow_fallback = true` (see
            // `truncate_to_fit_fallible`'s doc), kept only so this method
            // stays infallible even if that invariant ever changes.
            self.conversation.len().saturating_sub(old_len)
        });
        Self::survivors(tokens.len(), old_len, removed, self.strategy)
    }

    /// Fallible counterpart of [`Self::append`]: identical behaviour, except
    /// a failing [`Summarizer`] (see [`Self::set_summarizer`]) is surfaced
    /// as an error instead of being masked by the built-in extractive
    /// fallback.
    pub fn try_append(&mut self, tokens: &[u32]) -> Result<usize, ContextError> {
        let old_len = self.conversation.len();
        self.conversation.extend_from_slice(tokens);
        let removed = self.truncate_to_fit_fallible(false)?;
        Ok(Self::survivors(
            tokens.len(),
            old_len,
            removed,
            self.strategy,
        ))
    }

    /// How many of the `new_len` just-appended tokens are still present,
    /// verbatim, after a truncation that removed `removed` tokens (by the
    /// generic `before_len - after_len` accounting `truncate_to_fit`
    /// reports) starting from a pre-append conversation of `old_len` tokens.
    ///
    /// `TruncateRight` evicts from the tail — i.e. from the just-appended
    /// tokens first — so survivors are simply `new_len - removed`. Every
    /// other strategy evicts from the front, so the just-appended tokens
    /// (which sit at indices `[old_len, old_len + new_len)`) are affected
    /// only once `removed` reaches past `old_len`.
    fn survivors(
        new_len: usize,
        old_len: usize,
        removed: usize,
        strategy: TruncationStrategy,
    ) -> usize {
        match strategy {
            TruncationStrategy::TruncateRight => new_len.saturating_sub(removed),
            TruncationStrategy::TruncateLeft
            | TruncationStrategy::SlidingWindow
            | TruncationStrategy::Summarize => {
                new_len.saturating_sub(removed.saturating_sub(old_len))
            }
        }
    }

    /// Truncate the conversation segment to make the total fit within `max_tokens`.
    ///
    /// Applies the configured [`TruncationStrategy`].
    /// Returns the number of tokens net-removed from the conversation
    /// (`before_len - after_len`) — for `Summarize`, this already accounts
    /// for the summary tokens spliced back in, so it is the *net* shrink,
    /// not the gross number of raw tokens evicted.
    pub fn truncate_to_fit(&mut self) -> usize {
        self.truncate_to_fit_fallible(true).unwrap_or(0)
    }

    /// Core truncation logic shared by [`Self::truncate_to_fit`],
    /// [`Self::append`] and [`Self::try_append`].
    ///
    /// `allow_fallback` controls what happens when `Summarize` has a
    /// [`Summarizer`] attached and it returns `Err`: `true` (used by the
    /// infallible callers) falls back to the built-in extractive summary and
    /// always returns `Ok`; `false` (used by [`Self::try_append`])
    /// propagates the error instead.
    fn truncate_to_fit_fallible(&mut self, allow_fallback: bool) -> Result<usize, ContextError> {
        let capacity_for_conv = self.max_tokens.saturating_sub(self.system_tokens.len());
        let before = self.conversation.len();
        if before <= capacity_for_conv {
            return Ok(0);
        }
        let excess = before - capacity_for_conv;

        match self.strategy {
            TruncationStrategy::TruncateLeft => {
                self.conversation.drain(0..excess);
            }
            TruncationStrategy::TruncateRight => {
                let new_len = before - excess;
                self.conversation.truncate(new_len);
            }
            TruncationStrategy::SlidingWindow => {
                // Batch-evict down to the configured fixed window (default:
                // the full available capacity, i.e. identical to
                // `TruncateLeft`) instead of the bare minimum `excess`.
                let window = self
                    .sliding_window_size
                    .unwrap_or(capacity_for_conv)
                    .min(capacity_for_conv);
                let drop = before.saturating_sub(window);
                self.conversation.drain(0..drop);
            }
            TruncationStrategy::Summarize => {
                self.apply_summarize(excess, capacity_for_conv, allow_fallback)?;
            }
        }

        Ok(before - self.conversation.len())
    }

    /// `Summarize` truncation: evict the oldest `excess` tokens plus a small
    /// extra `summary_budget` to make room, compress that whole evicted span
    /// (via the attached [`Summarizer`], or the built-in extractive default),
    /// and splice the (possibly clipped, to respect `max_tokens`) result back
    /// in ahead of what remains — instead of discarding the span outright.
    fn apply_summarize(
        &mut self,
        excess: usize,
        capacity_for_conv: usize,
        allow_fallback: bool,
    ) -> Result<(), ContextError> {
        // A span too small to usefully compress is simply dropped, exactly
        // like `TruncateLeft` would for that call — there is no shorter,
        // still-meaningful representation of one or two tokens.
        if excess < MIN_TOKENS_TO_SUMMARIZE {
            self.conversation.drain(0..excess);
            return Ok(());
        }

        let summary_budget = (excess / SUMMARY_DIVISOR).max(1);
        let to_evict = (excess + summary_budget).min(self.conversation.len());
        let evicted: Vec<u32> = self.conversation.drain(0..to_evict).collect();

        let summary = match &self.summarizer {
            Some(summarizer) => match summarizer.summarize(&evicted, summary_budget) {
                Ok(summary) => summary,
                Err(e) if allow_fallback => {
                    tracing::warn!(
                        error = %e,
                        "context_manager: attached Summarizer failed; falling back to the \
                         built-in extractive summary for this eviction"
                    );
                    default_extractive_summary(&evicted, summary_budget)
                }
                Err(e) => return Err(e),
            },
            None => default_extractive_summary(&evicted, summary_budget),
        };

        // A misbehaving custom summarizer must not be allowed to blow the
        // token budget back open; keep only the most recent part of an
        // over-long summary (mirrors the strategy's own bias toward
        // recency).
        let clipped_summary = if summary.len() > summary_budget {
            &summary[summary.len() - summary_budget..]
        } else {
            &summary[..]
        };

        let mut spliced = Vec::with_capacity(clipped_summary.len() + self.conversation.len());
        spliced.extend_from_slice(clipped_summary);
        spliced.extend_from_slice(&self.conversation);

        // Guard the invariant even if `summary_budget`'s bookkeeping above
        // is ever off by a token: never let the conversation end up over
        // capacity.
        if spliced.len() > capacity_for_conv {
            let overflow = spliced.len() - capacity_for_conv;
            spliced.drain(0..overflow);
        }

        self.conversation = spliced;
        Ok(())
    }

    /// Concatenate system tokens and conversation tokens into a single flat vector.
    ///
    /// The result is always within `max_tokens`.
    pub fn tokens(&self) -> Vec<u32> {
        let mut result = Vec::with_capacity(self.system_tokens.len() + self.conversation.len());
        result.extend_from_slice(&self.system_tokens);
        result.extend_from_slice(&self.conversation);
        result
    }

    /// Total token count (system + conversation).
    pub fn len(&self) -> usize {
        self.system_tokens.len() + self.conversation.len()
    }

    /// Returns `true` if both system and conversation are empty.
    pub fn is_empty(&self) -> bool {
        self.system_tokens.is_empty() && self.conversation.is_empty()
    }

    /// Number of additional tokens that can be appended before truncation.
    pub fn remaining_capacity(&self) -> usize {
        self.max_tokens.saturating_sub(self.len())
    }

    /// Returns `true` if the window is at or beyond its maximum capacity.
    pub fn is_at_limit(&self) -> bool {
        self.len() >= self.max_tokens
    }

    /// Clear all conversation tokens (system prompt is preserved).
    pub fn clear_conversation(&mut self) {
        self.conversation.clear();
    }

    /// Fraction of `max_tokens` currently in use: `len / max_tokens`.
    ///
    /// Returns 0.0 if `max_tokens` is zero.
    pub fn utilization(&self) -> f32 {
        if self.max_tokens == 0 {
            return 0.0;
        }
        self.len() as f32 / self.max_tokens as f32
    }
}

// ──────────────────────────────────────────────────────────────────
// ConversationTurn
// ──────────────────────────────────────────────────────────────────

/// A single turn in a multi-turn conversation.
pub struct ConversationTurn {
    /// Role identifier (e.g., `"user"`, `"assistant"`, `"system"`).
    pub role: String,
    /// Raw text content of this turn.
    pub content: String,
    /// Pre-tokenised representation of `content`.
    pub token_ids: Vec<u32>,
}

// ──────────────────────────────────────────────────────────────────
// ConversationContext
// ──────────────────────────────────────────────────────────────────

/// A multi-turn conversation with automatic context window management.
///
/// Each added turn is stored with its role, content, and token ids.
/// `build_tokens()` concatenates all turn token ids in order,
/// respecting the underlying [`ContextWindow`]'s token budget.
pub struct ConversationContext {
    window: ContextWindow,
    turns: Vec<ConversationTurn>,
}

impl ConversationContext {
    /// Create a new conversation context with the given maximum token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            window: ContextWindow::new(max_tokens, TruncationStrategy::TruncateLeft),
            turns: Vec::new(),
        }
    }

    /// Add a conversation turn.
    ///
    /// The turn's token ids are appended to the context window.
    pub fn add_turn(&mut self, role: &str, content: &str, token_ids: Vec<u32>) {
        self.window.append(&token_ids);
        self.turns.push(ConversationTurn {
            role: role.to_string(),
            content: content.to_string(),
            token_ids,
        });
    }

    /// Build a flat token sequence from all turns, respecting the window budget.
    ///
    /// Concatenates token ids in turn order. The result is always within
    /// `max_tokens` after truncation.
    pub fn build_tokens(&self) -> Vec<u32> {
        self.window.tokens()
    }

    /// Number of turns added to this conversation.
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Total token count across the current context window (after truncation).
    pub fn total_tokens(&self) -> usize {
        self.window.len()
    }

    /// Returns `true` if the context window is at its maximum capacity.
    pub fn is_full(&self) -> bool {
        self.window.is_at_limit()
    }

    /// Clear all turns and reset the context window.
    pub fn clear(&mut self) {
        self.turns.clear();
        self.window.clear_conversation();
    }

    /// Reference to the most recently added turn, if any.
    pub fn last_turn(&self) -> Option<&ConversationTurn> {
        self.turns.last()
    }

    /// Utilisation of the token budget: `total_tokens / max_tokens`.
    pub fn utilization(&self) -> f32 {
        self.window.utilization()
    }
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_window_append_within_limit() {
        let mut window = ContextWindow::new(100, TruncationStrategy::TruncateLeft);
        let appended = window.append(&[1, 2, 3, 4, 5]);
        assert!(appended > 0, "should append tokens when within limit");
        assert_eq!(window.conversation.len(), 5);
        assert_eq!(window.len(), 5);
    }

    #[test]
    fn test_context_window_truncate_left() {
        let mut window = ContextWindow::new(5, TruncationStrategy::TruncateLeft);
        // Fill to capacity
        window.append(&[1, 2, 3, 4, 5]);
        assert_eq!(window.conversation.len(), 5);

        // Append more — oldest should be dropped
        window.append(&[6, 7]);
        assert_eq!(
            window.conversation.len(),
            5,
            "should still be at max after truncation"
        );
        // The newest tokens (6, 7) should be at the end
        let last = *window.conversation.last().expect("must have tokens");
        assert_eq!(last, 7, "newest token should be 7");
        // The oldest tokens (1, 2) should be gone
        assert!(
            !window.conversation.contains(&1),
            "token 1 should have been truncated"
        );
    }

    #[test]
    fn test_context_window_truncate_right() {
        let mut window = ContextWindow::new(5, TruncationStrategy::TruncateRight);
        window.append(&[1, 2, 3, 4, 5]);
        window.append(&[6, 7]);
        // Newest tokens (6, 7) should be dropped, oldest retained
        assert_eq!(window.conversation.len(), 5);
        assert_eq!(
            window.conversation[0], 1,
            "token 1 should be preserved with TruncateRight"
        );
        assert!(
            !window.conversation.contains(&6),
            "token 6 should have been truncated"
        );
    }

    #[test]
    fn test_context_window_system_prompt_preserved() {
        let mut window = ContextWindow::new(10, TruncationStrategy::TruncateLeft);
        window
            .set_system_prompt(vec![100, 200, 300])
            .expect("system prompt should fit");

        // Fill remaining capacity (7 slots)
        window.append(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(window.len(), 10);

        // Add more — system tokens must survive
        window.append(&[8, 9]);
        let tokens = window.tokens();
        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens[0], 100, "system token 0 must be preserved");
        assert_eq!(tokens[1], 200, "system token 1 must be preserved");
        assert_eq!(tokens[2], 300, "system token 2 must be preserved");
    }

    #[test]
    fn test_context_window_remaining_capacity() {
        let mut window = ContextWindow::new(20, TruncationStrategy::TruncateLeft);
        assert_eq!(window.remaining_capacity(), 20);
        window.append(&[1, 2, 3]);
        assert_eq!(window.remaining_capacity(), 17);
        window.set_system_prompt(vec![10, 20]).expect("fits");
        // system (2) + conversation (3) = 5; remaining = 15
        assert_eq!(window.remaining_capacity(), 15);
    }

    #[test]
    fn test_context_window_system_prompt_too_large() {
        let mut window = ContextWindow::new(5, TruncationStrategy::TruncateLeft);
        let result = window.set_system_prompt(vec![1, 2, 3, 4, 5, 6]);
        assert!(
            result.is_err(),
            "system prompt larger than max_tokens should error"
        );
    }

    #[test]
    fn test_conversation_context_add_turn() {
        let mut ctx = ConversationContext::new(200);
        ctx.add_turn("user", "Hello!", vec![10, 20, 30]);
        ctx.add_turn("assistant", "Hi there!", vec![40, 50, 60, 70]);

        assert_eq!(ctx.turn_count(), 2);
        assert_eq!(ctx.total_tokens(), 7, "3 + 4 = 7 tokens total");

        let last = ctx.last_turn().expect("must have a last turn");
        assert_eq!(last.role, "assistant");
        assert_eq!(last.content, "Hi there!");
    }

    #[test]
    fn test_conversation_context_build_tokens() {
        let mut ctx = ConversationContext::new(100);
        ctx.add_turn("user", "A", vec![1, 2]);
        ctx.add_turn("assistant", "B", vec![3, 4, 5]);

        let tokens = ctx.build_tokens();
        assert_eq!(
            tokens,
            vec![1, 2, 3, 4, 5],
            "tokens should be in turn order"
        );
    }

    #[test]
    fn test_context_utilization() {
        let mut window = ContextWindow::new(100, TruncationStrategy::TruncateLeft);
        assert!(
            (window.utilization() - 0.0).abs() < f32::EPSILON,
            "empty window has 0.0 utilization"
        );
        window.append(&(0u32..50).collect::<Vec<_>>());
        assert!(
            (window.utilization() - 0.5).abs() < f32::EPSILON,
            "50/100 = 0.5 utilization"
        );
        window.append(&(0u32..50).collect::<Vec<_>>());
        assert!(
            (window.utilization() - 1.0).abs() < f32::EPSILON,
            "full window = 1.0 utilization"
        );
    }

    // ── RT-16: SlidingWindow is genuinely distinct from TruncateLeft ───────

    #[test]
    fn sliding_window_unconfigured_matches_truncate_left_exactly() {
        // Behaviour-preserving default: an existing `SlidingWindow` caller
        // that never calls `set_sliding_window_size` must see byte-identical
        // eviction to before this fix.
        let mut left = ContextWindow::new(10, TruncationStrategy::TruncateLeft);
        let mut sliding = ContextWindow::new(10, TruncationStrategy::SlidingWindow);
        for batch in [vec![1, 2, 3, 4, 5, 6, 7], vec![8, 9], vec![10, 11, 12]] {
            left.append(&batch);
            sliding.append(&batch);
            assert_eq!(left.conversation, sliding.conversation);
        }
    }

    #[test]
    fn sliding_window_with_configured_size_batch_evicts() {
        let mut window = ContextWindow::new(10, TruncationStrategy::SlidingWindow);
        window.set_sliding_window_size(4);
        window.append(&(0u32..10).collect::<Vec<_>>()); // fills to capacity: 0..10
        assert_eq!(window.conversation.len(), 10);

        // One more token should batch-evict down to the configured window
        // (4), not the bare minimum (which would leave 10 tokens).
        window.append(&[99]);
        assert_eq!(
            window.conversation.len(),
            4,
            "configured SlidingWindow must evict down to its fixed window, not the bare minimum"
        );
        assert_eq!(
            *window.conversation.last().expect("non-empty"),
            99,
            "the newest token must survive"
        );
    }

    #[test]
    fn sliding_window_and_truncate_left_now_diverge_when_configured() {
        // The literal RT-16 complaint: before the fix these two variants
        // shared one match arm and could never differ. Prove they can.
        let mut left = ContextWindow::new(10, TruncationStrategy::TruncateLeft);
        let mut sliding = ContextWindow::new(10, TruncationStrategy::SlidingWindow);
        sliding.set_sliding_window_size(3);

        left.append(&(0u32..10).collect::<Vec<_>>());
        sliding.append(&(0u32..10).collect::<Vec<_>>());
        left.append(&[100]);
        sliding.append(&[100]);

        assert_ne!(
            left.conversation.len(),
            sliding.conversation.len(),
            "a configured SlidingWindow must behave differently from TruncateLeft"
        );
    }

    // ── RT-16: Summarize is real, not a silent alias for TruncateLeft ──────

    #[test]
    fn summarize_retains_information_truncate_left_would_discard() {
        let mut summarize = ContextWindow::new(20, TruncationStrategy::Summarize);
        let mut truncate_left = ContextWindow::new(20, TruncationStrategy::TruncateLeft);

        let prompt: Vec<u32> = (1000..1040).collect(); // 40 tokens, well over budget
        summarize.append(&prompt);
        truncate_left.append(&prompt);

        // TruncateLeft keeps zero information about the dropped span by
        // construction — none of the earliest ids survive.
        assert!(!truncate_left.conversation.contains(&1000));

        // Summarize must retain *something* from the earliest part of the
        // conversation that TruncateLeft fully discarded — a non-verbatim,
        // shorter-than-original gist of the dropped span, not the raw span
        // itself and not nothing.
        assert!(
            summarize.conversation.contains(&1000),
            "Summarize should keep a lead token from the evicted span as part of its gist; \
             conversation = {:?}",
            summarize.conversation
        );
        assert!(
            summarize.conversation.len() <= 20,
            "must still respect max_tokens"
        );
        assert_ne!(
            summarize.conversation, truncate_left.conversation,
            "Summarize must not be a silent alias for TruncateLeft (RT-16)"
        );
    }

    #[test]
    fn summarize_never_exceeds_max_tokens() {
        let mut window = ContextWindow::new(16, TruncationStrategy::Summarize);
        window.append(&(0u32..500).collect::<Vec<_>>());
        assert!(window.len() <= 16);
        // Feed it again to exercise repeated summarize-on-summarize eviction.
        window.append(&(500u32..600).collect::<Vec<_>>());
        assert!(window.len() <= 16);
    }

    #[test]
    fn summarize_small_excess_degrades_to_plain_drop() {
        // Below MIN_TOKENS_TO_SUMMARIZE there is nothing meaningful to
        // compress; this must not panic and must still respect max_tokens.
        let mut window = ContextWindow::new(10, TruncationStrategy::Summarize);
        window.append(&(0u32..10).collect::<Vec<_>>());
        window.append(&[42]); // excess == 1, below the threshold
        assert!(window.len() <= 10);
        assert_eq!(*window.conversation.last().expect("non-empty"), 42);
    }

    #[test]
    fn custom_summarizer_is_invoked_and_its_output_is_used() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        let call_count = StdArc::new(AtomicUsize::new(0));
        let call_count_clone = StdArc::clone(&call_count);

        let mut window = ContextWindow::new(20, TruncationStrategy::Summarize);
        window.set_summarizer(move |tokens: &[u32], budget: usize| {
            call_count_clone.fetch_add(1, Ordering::Relaxed);
            // A distinctive marker sentinel proves *this* summarizer's
            // output, not the built-in default, ended up in the window.
            let mut out = vec![u32::MAX];
            out.extend(tokens.iter().take(budget.saturating_sub(1)).copied());
            Ok(out)
        });
        assert!(window.has_summarizer());

        window.append(&(0u32..40).collect::<Vec<_>>());

        assert!(
            call_count.load(Ordering::Relaxed) >= 1,
            "the attached Summarizer must actually be called"
        );
        assert!(
            window.conversation.contains(&u32::MAX),
            "the attached Summarizer's output (marked with the sentinel) must be used; got {:?}",
            window.conversation
        );
        assert!(window.len() <= 20);
    }

    #[test]
    fn failing_custom_summarizer_falls_back_to_extractive_default_on_append() {
        let mut window = ContextWindow::new(20, TruncationStrategy::Summarize);
        window.set_summarizer(|_tokens: &[u32], _budget: usize| {
            Err(ContextError::new("simulated summarizer failure"))
        });

        // `append` is infallible and must still honour the Summarize
        // contract (a real, shorter, non-verbatim replacement) via the
        // built-in fallback rather than surfacing the error or silently
        // reverting to bare truncation.
        let prompt: Vec<u32> = (1000..1040).collect();
        window.append(&prompt);

        assert!(window.len() <= 20);
        assert!(
            window.conversation.contains(&1000),
            "even on summarizer failure, append() must keep the fallback's lead-token gist"
        );
    }

    #[test]
    fn failing_custom_summarizer_surfaces_through_try_append() {
        let mut window = ContextWindow::new(20, TruncationStrategy::Summarize);
        window.set_summarizer(|_tokens: &[u32], _budget: usize| {
            Err(ContextError::new("simulated summarizer failure"))
        });

        let prompt: Vec<u32> = (1000..1040).collect();
        let result = window.try_append(&prompt);
        assert!(
            result.is_err(),
            "try_append must surface a failing Summarizer's error instead of masking it"
        );
    }

    #[test]
    fn oversized_custom_summarizer_output_is_clipped_not_overflowed() {
        // A summarizer that ignores `budget` and returns far too much must
        // never be allowed to push the window over `max_tokens`.
        let mut window = ContextWindow::new(20, TruncationStrategy::Summarize);
        window.set_summarizer(|tokens: &[u32], _budget: usize| Ok(tokens.to_vec()));

        window.append(&(0u32..40).collect::<Vec<_>>());
        assert!(
            window.len() <= 20,
            "an over-long Summarizer output must be clipped, never allowed to overflow max_tokens"
        );
    }

    #[test]
    fn clear_summarizer_reverts_to_the_extractive_default() {
        let mut window = ContextWindow::new(20, TruncationStrategy::Summarize);
        window.set_summarizer(|_tokens: &[u32], budget: usize| Ok(vec![u32::MAX; budget.max(1)]));
        assert!(window.has_summarizer());
        window.clear_summarizer();
        assert!(!window.has_summarizer());

        window.append(&(1000u32..1040).collect::<Vec<_>>());
        assert!(
            !window.conversation.contains(&u32::MAX),
            "after clear_summarizer, the custom summarizer must no longer run"
        );
    }

    // ── append()'s survivor count (regression: must not undercount when a
    //    strategy evicts more than the bare `excess`) ───────────────────────

    #[test]
    fn append_survivor_count_is_exact_for_truncate_left_incremental_growth() {
        let mut window = ContextWindow::new(5, TruncationStrategy::TruncateLeft);
        assert_eq!(window.append(&[1, 2, 3, 4, 5]), 5);
        // Appending 2 more into an already-full window of 5: excess=2,
        // both newly appended tokens fully survive (only old ones evicted).
        assert_eq!(
            window.append(&[6, 7]),
            2,
            "both newly appended tokens are present; append() must not undercount them"
        );
        assert_eq!(window.conversation, vec![3, 4, 5, 6, 7]);
    }

    #[test]
    fn append_survivor_count_is_exact_for_truncate_right() {
        let mut window = ContextWindow::new(5, TruncationStrategy::TruncateRight);
        assert_eq!(window.append(&[1, 2, 3, 4, 5]), 5);
        // TruncateRight drops from the tail, i.e. the just-appended tokens
        // are the first to go.
        assert_eq!(window.append(&[6, 7]), 0);
        assert_eq!(window.conversation, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn append_survivor_count_is_exact_for_configured_sliding_window() {
        let mut window = ContextWindow::new(10, TruncationStrategy::SlidingWindow);
        window.set_sliding_window_size(3);
        assert_eq!(window.append(&(0u32..10).collect::<Vec<_>>()), 10);

        // Eager eviction removes far more than the bare `excess` (5): with
        // 15 tokens total and a window of 3, positions 0..12 are dropped —
        // that overlaps 2 of the 5 just-appended tokens (100, 101 at
        // positions 10, 11), so only 3 of the 5 survive (102, 103, 104 at
        // positions 12, 13, 14). The point of this test is that the
        // survivor count matches *exactly* what is physically left in
        // `conversation`, not that it equals `tokens.len()`.
        let appended = window.append(&[100, 101, 102, 103, 104]);
        assert_eq!(window.conversation, vec![102, 103, 104]);
        assert_eq!(
            appended,
            window.conversation.len(),
            "every survivor here is a just-appended token (the window holds \
             nothing older), so the reported count must equal exactly what \
             is left; got {appended}"
        );
        assert_eq!(appended, 3);
    }

    // ── Summarizer trait blanket impl for plain closures ────────────────────

    #[test]
    fn summarizer_trait_object_works_through_with_summarizer_builder() {
        let window = ContextWindow::new(20, TruncationStrategy::Summarize).with_summarizer(
            |tokens: &[u32], budget: usize| Ok(tokens.iter().take(budget).copied().collect()),
        );
        assert!(window.has_summarizer());
    }
}
