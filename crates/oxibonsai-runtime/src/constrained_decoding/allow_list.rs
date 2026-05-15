//! [`AllowListConstraint`] — force output to be one of a finite set of sequences.

use super::error_trait::TokenConstraint;

// ─────────────────────────────────────────────────────────────────────────────
// AllowListConstraint — force output to be one of a finite set of sequences
// ─────────────────────────────────────────────────────────────────────────────

/// A constraint that forces the generated token sequence to exactly match one
/// of a finite set of allowed token-id sequences (e.g., multiple-choice answers).
///
/// At each step only the union of next tokens across all still-active candidates
/// is permitted.  A candidate becomes inactive the moment any token in the prefix
/// fails to match.  Generation is considered complete once the full token sequence
/// of at least one candidate has been consumed.
///
/// # Example
/// ```rust
/// use oxibonsai_runtime::constrained_decoding::{AllowListConstraint, TokenConstraint};
///
/// // Two candidates: [10, 20] and [10, 30]
/// let mut c = AllowListConstraint::new(vec![vec![10, 20], vec![10, 30]]);
/// // First token: only 10 is allowed (shared prefix)
/// let mask = c.allowed_tokens(&[], 50).unwrap();
/// assert!(mask[10]);
/// assert!(!mask[20]);
/// assert!(!mask[30]);
/// ```
pub struct AllowListConstraint {
    /// All allowed sequences.
    candidates: Vec<Vec<u32>>,
    /// Which candidates still match the current prefix.
    active: Vec<bool>,
    /// Number of tokens consumed so far.
    position: usize,
}

impl AllowListConstraint {
    /// Create a new `AllowListConstraint` from a list of allowed token sequences.
    pub fn new(candidates: Vec<Vec<u32>>) -> Self {
        let n = candidates.len();
        Self {
            candidates,
            active: vec![true; n],
            position: 0,
        }
    }

    /// Returns the number of candidate sequences that are still active.
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }
}

impl TokenConstraint for AllowListConstraint {
    /// Returns a bitmask of tokens that are valid next tokens across all still-active
    /// candidates at the current position.  Always returns `Some` (never unconstrained).
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        let mut mask = vec![false; vocab_size];
        for (i, active) in self.active.iter().enumerate() {
            if !active {
                continue;
            }
            let seq = &self.candidates[i];
            if self.position < seq.len() {
                let tok = seq[self.position] as usize;
                if tok < vocab_size {
                    mask[tok] = true;
                }
            }
        }
        Some(mask)
    }

    /// Commits `token` at the current position.
    ///
    /// Any candidate where `candidates[i][position] != token` (or the candidate is
    /// already exhausted) is deactivated.  Returns `true` when at least one candidate
    /// remains active **or** a candidate was just completed at this position.
    fn advance(&mut self, token: u32) -> bool {
        let mut just_completed = false;
        for (i, active) in self.active.iter_mut().enumerate() {
            if !*active {
                continue;
            }
            let seq = &self.candidates[i];
            if self.position >= seq.len() {
                // Candidate was already completed; further tokens deactivate it.
                *active = false;
            } else if seq[self.position] == token {
                // Token matches; check if this completes the candidate.
                if self.position + 1 == seq.len() {
                    just_completed = true;
                }
                // Keep active — will be filtered by is_complete / future advance calls.
            } else {
                *active = false;
            }
        }
        self.position += 1;
        // Return true if at least one candidate is still active or one just completed.
        just_completed || self.active.iter().any(|&a| a)
    }

    /// Returns `true` when the consumed token sequence fully matches at least one
    /// candidate, i.e. `position == candidates[i].len()` for some active `i`.
    fn is_complete(&self) -> bool {
        self.candidates
            .iter()
            .enumerate()
            .any(|(i, seq)| self.active[i] && self.position == seq.len())
    }

    /// Reset to initial state (all candidates active, position zero).
    fn reset(&mut self) {
        self.active.fill(true);
        self.position = 0;
    }

    fn name(&self) -> &str {
        "AllowListConstraint"
    }
}
