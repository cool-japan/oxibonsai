//! [`LengthConstraint`] — enforce hard minimum and maximum generation lengths.

use super::error_trait::TokenConstraint;

// ─────────────────────────────────────────────────────────────────────────────
// LengthConstraint — enforce hard minimum and maximum generation lengths
// ─────────────────────────────────────────────────────────────────────────────

/// A constraint that enforces hard minimum and maximum token-count limits.
///
/// - While `count < min_len`: if a `stop_token` is configured it is excluded from
///   the mask (cannot stop early).
/// - While `count >= max_len`: if a `stop_token` is configured only that token is
///   allowed; otherwise an all-`false` mask is returned, signalling the caller to
///   halt generation externally.
/// - Between `min_len` and `max_len`: all tokens are allowed (`None`).
///
/// Completion is defined as either reaching `max_len` OR generating `min_len` or
/// more tokens followed by the `stop_token`.
///
/// # Example
/// ```rust
/// use oxibonsai_runtime::constrained_decoding::{LengthConstraint, TokenConstraint};
///
/// // Must generate at least 2 tokens, stop token is 1 (EOS), max 10.
/// let mut c = LengthConstraint::new(2, 10, Some(1));
/// // Before min_len: stop_token excluded
/// let mask = c.allowed_tokens(&[], 4).unwrap();
/// assert!(!mask[1]);  // stop token blocked
/// assert!(mask[0]);   // other tokens allowed
/// ```
pub struct LengthConstraint {
    /// Minimum number of tokens that must be generated before stopping.
    min_len: usize,
    /// Hard upper bound on generated token count.
    max_len: usize,
    /// Optional end-of-sequence token; treated specially for early-stop control.
    stop_token: Option<u32>,
    /// Number of tokens committed via `advance` so far.
    count: usize,
    /// True once the `stop_token` has been committed.
    stop_seen: bool,
}

impl LengthConstraint {
    /// Create a new `LengthConstraint`.
    ///
    /// `min_len` must be `<= max_len`.
    pub fn new(min_len: usize, max_len: usize, stop_token: Option<u32>) -> Self {
        Self {
            min_len,
            max_len,
            stop_token,
            count: 0,
            stop_seen: false,
        }
    }

    /// Current token count.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl TokenConstraint for LengthConstraint {
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        if self.count < self.min_len {
            // Cannot stop early — exclude stop_token if one is configured.
            if let Some(stop) = self.stop_token {
                let mut mask = vec![true; vocab_size];
                let stop_idx = stop as usize;
                if stop_idx < vocab_size {
                    mask[stop_idx] = false;
                }
                return Some(mask);
            }
            // No stop token: no restriction below min_len.
            return None;
        }

        if self.count >= self.max_len {
            // Must stop now.
            if let Some(stop) = self.stop_token {
                let mut mask = vec![false; vocab_size];
                let stop_idx = stop as usize;
                if stop_idx < vocab_size {
                    mask[stop_idx] = true;
                }
                return Some(mask);
            }
            // No stop token: emit an all-false mask to force external termination.
            return Some(vec![false; vocab_size]);
        }

        // Between min and max: unconstrained.
        None
    }

    /// Commits `token`, updating `count` and `stop_seen`.  Always returns `true`.
    fn advance(&mut self, token: u32) -> bool {
        if let Some(stop) = self.stop_token {
            if token == stop {
                self.stop_seen = true;
            }
        }
        self.count += 1;
        true
    }

    /// Returns `true` when at least `min_len` tokens have been generated AND either
    /// the `stop_token` was seen or `max_len` has been reached.
    fn is_complete(&self) -> bool {
        if self.count < self.min_len {
            return false;
        }
        self.count >= self.max_len || self.stop_seen
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.count = 0;
        self.stop_seen = false;
    }

    fn name(&self) -> &str {
        "LengthConstraint"
    }
}
