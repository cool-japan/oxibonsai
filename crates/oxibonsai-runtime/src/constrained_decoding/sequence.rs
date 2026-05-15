//! [`SequenceConstraint`] — force output to follow a specific token sequence exactly.

use super::error_trait::TokenConstraint;

// ─────────────────────────────────────────────────────────────────────────────
// SequenceConstraint — force output to follow a specific token sequence exactly
// ─────────────────────────────────────────────────────────────────────────────

/// A constraint that forces the generated output to reproduce a specific,
/// pre-determined token sequence.
///
/// While `position < target.len()`, only `target[position]` is allowed.
/// Once the target has been fully reproduced (`position >= target.len()`) the
/// constraint is satisfied and all tokens become allowed again (returns `None`).
///
/// # Example
/// ```rust
/// use oxibonsai_runtime::constrained_decoding::{SequenceConstraint, TokenConstraint};
///
/// let mut c = SequenceConstraint::new(vec![5, 6, 7]);
/// let mask = c.allowed_tokens(&[], 10).unwrap();
/// assert!(mask[5]);
/// assert!(!mask[6]);
/// assert_eq!(c.advance(5), true);
/// ```
pub struct SequenceConstraint {
    /// The token sequence that must be reproduced.
    target: Vec<u32>,
    /// Number of tokens consumed (next expected index into `target`).
    position: usize,
    /// Set to `true` if a mismatched token was ever committed.
    failed: bool,
}

impl SequenceConstraint {
    /// Create a new `SequenceConstraint` for the given target sequence.
    pub fn new(target: Vec<u32>) -> Self {
        Self {
            target,
            position: 0,
            failed: false,
        }
    }

    /// Whether the constraint has been violated (a wrong token was committed).
    pub fn is_failed(&self) -> bool {
        self.failed
    }
}

impl TokenConstraint for SequenceConstraint {
    /// Returns a bitmask allowing only the next expected token, or `None` once the
    /// full sequence has been reproduced.
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        if self.position >= self.target.len() {
            // Sequence fully consumed — no further restriction.
            return None;
        }
        let mut mask = vec![false; vocab_size];
        let next = self.target[self.position] as usize;
        if next < vocab_size {
            mask[next] = true;
        }
        Some(mask)
    }

    /// Commits `token`.  Returns `false` (and sets the failed flag) if `token`
    /// does not match the expected token at the current position.
    fn advance(&mut self, token: u32) -> bool {
        if self.position < self.target.len() && token != self.target[self.position] {
            self.failed = true;
            self.position += 1;
            return false;
        }
        self.position += 1;
        true
    }

    /// Returns `true` once all tokens in the target sequence have been consumed.
    fn is_complete(&self) -> bool {
        self.position >= self.target.len()
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.position = 0;
        self.failed = false;
    }

    fn name(&self) -> &str {
        "SequenceConstraint"
    }
}
