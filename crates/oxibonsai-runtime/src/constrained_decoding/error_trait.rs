//! Core trait and error types for grammar-constrained decoding.
//!
//! This sub-module hosts:
//! - [`ConstraintError`]: errors that can arise when building or running a token constraint
//! - [`TokenConstraint`]: the trait implemented by every concrete constraint
//! - [`NoConstraint`]: a passthrough constraint that allows all tokens

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise when building or running a token constraint.
#[derive(Debug, thiserror::Error)]
pub enum ConstraintError {
    /// The supplied regex pattern was syntactically invalid.
    #[error("Invalid regex pattern: {0}")]
    InvalidPattern(String),

    /// The supplied JSON schema was invalid (reserved for future schema-based constraints).
    #[error("Invalid JSON schema: {0}")]
    InvalidSchema(String),

    /// The constraint was violated at a specific token.
    #[error("Constraint violated at token {token}: {reason}")]
    Violated { token: u32, reason: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A constraint that restricts which tokens are valid at each decoding step.
///
/// Implementors maintain internal state representing how far through the
/// constrained structure the generation has progressed.
pub trait TokenConstraint: Send + Sync {
    /// Given the tokens generated so far, return a bitmask of allowed next tokens.
    ///
    /// `vocab_size` is the total vocabulary size.  Returns `None` if all tokens
    /// are allowed (no active constraint).
    fn allowed_tokens(&self, generated: &[u32], vocab_size: usize) -> Option<Vec<bool>>;

    /// Called after a token is committed.
    ///
    /// Returns `false` if the constraint is now violated (generation should stop).
    fn advance(&mut self, token: u32) -> bool;

    /// Returns `true` if the current state is a valid terminal state.
    fn is_complete(&self) -> bool;

    /// Reset the constraint to its initial state.
    fn reset(&mut self);

    /// Human-readable name for debugging and logging.
    fn name(&self) -> &str;
}

// ─────────────────────────────────────────────────────────────────────────────
// NoConstraint — passthrough
// ─────────────────────────────────────────────────────────────────────────────

/// A passthrough constraint that places no restriction on the vocabulary.
pub struct NoConstraint;

impl TokenConstraint for NoConstraint {
    fn allowed_tokens(&self, _generated: &[u32], _vocab_size: usize) -> Option<Vec<bool>> {
        None
    }

    fn advance(&mut self, _token: u32) -> bool {
        true
    }

    fn is_complete(&self) -> bool {
        true
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "NoConstraint"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_constraint_allows_all() {
        let nc = NoConstraint;
        assert!(nc.allowed_tokens(&[], 10).is_none());
    }

    #[test]
    fn constraint_error_display() {
        let e = ConstraintError::InvalidPattern("bad".to_string());
        assert!(e.to_string().contains("bad"));
        let e2 = ConstraintError::Violated {
            token: 5,
            reason: "oops".to_string(),
        };
        assert!(e2.to_string().contains("5"));
        assert!(e2.to_string().contains("oops"));
    }
}
