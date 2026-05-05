//! [`GrammarConstraint`] — implements [`TokenConstraint`] using the Earley
//! chart-parser recognizer backed by a BNF context-free grammar.
//!
//! The `allowed_tokens` method speculatively feeds each token's byte sequence
//! through a **clone** of the current recognizer state and marks the token
//! allowed if and only if none of the bytes are rejected.
//!
//! This is O(vocab × max_token_len × grammar_factor) per decode step, which
//! is the correct reference implementation.  For large vocabularies consider
//! batching or incremental FIRST-set caching in a production system.

use std::sync::Arc;

use super::ast::Grammar;
use super::earley::EarleyRecognizer;
use crate::constrained_decoding::TokenConstraint;

// ─────────────────────────────────────────────────────────────────────────────
// GrammarConstraint
// ─────────────────────────────────────────────────────────────────────────────

/// A [`TokenConstraint`] that enforces a context-free grammar on the generated
/// byte stream, using the Earley chart-parser as the underlying recognizer.
///
/// # Construction
///
/// ```rust,no_run
/// use oxibonsai_runtime::grammar::{arithmetic_grammar, GrammarConstraint};
///
/// let grammar = arithmetic_grammar();
/// // Map each token id to its byte sequence; single-byte ASCII vocab here.
/// let decode_fn = |token_id: u32| -> Vec<u8> {
///     if token_id < 128 { vec![token_id as u8] } else { vec![] }
/// };
/// let constraint = GrammarConstraint::new(grammar, decode_fn, 128);
/// ```
///
/// # Token decode function
///
/// The `tokenizer_decode_fn` maps a token id to the **byte sequence** it
/// represents.  For an ASCII byte-level vocabulary it is simply
/// `|id| vec![id as u8]`.  For a real LLM tokenizer it should call into
/// `tokenizer.id_to_bytes(id)`.  Unknown / special tokens can return an empty
/// `Vec<u8>`; they will be allowed iff the current recognizer state is
/// accepting (which allows a graceful end-of-sequence).
pub struct GrammarConstraint {
    /// Original grammar (kept for resetting the recognizer).
    #[allow(dead_code)]
    grammar: Arc<Grammar>,
    /// Live Earley recognizer tracking the bytes generated so far.
    recognizer: EarleyRecognizer,
    /// Decodes a token id to its raw byte sequence.
    tokenizer_decode_fn: Arc<dyn Fn(u32) -> Vec<u8> + Send + Sync>,
    /// Total vocabulary size (used for mask allocation).
    #[allow(dead_code)]
    vocab_size: usize,
}

impl GrammarConstraint {
    /// Create a new `GrammarConstraint`.
    ///
    /// The `grammar` is normalised (multi-byte terminals split into chains)
    /// and wrapped in an `Arc` before being handed to the recognizer.
    ///
    /// # Parameters
    ///
    /// * `grammar`               — the context-free grammar to enforce
    /// * `tokenizer_decode_fn`   — maps token id → byte sequence
    /// * `vocab_size`            — total vocabulary size
    pub fn new(
        mut grammar: Grammar,
        tokenizer_decode_fn: impl Fn(u32) -> Vec<u8> + Send + Sync + 'static,
        vocab_size: usize,
    ) -> Self {
        grammar.normalise_terminals();
        let grammar = Arc::new(grammar);
        let recognizer = EarleyRecognizer::new(Arc::clone(&grammar));
        Self {
            grammar,
            recognizer,
            tokenizer_decode_fn: Arc::new(tokenizer_decode_fn),
            vocab_size,
        }
    }

    /// Return the current number of bytes consumed by the recognizer.
    pub fn bytes_consumed(&self) -> usize {
        self.recognizer.input_pos
    }

    /// Return `true` if the recognizer is still in a live (non-dead) state.
    pub fn is_live(&self) -> bool {
        self.recognizer.is_live()
    }

    /// Return the set of bytes valid as the next byte in the stream.
    ///
    /// This is a low-level utility; prefer `allowed_tokens` for normal use.
    pub fn next_byte_set(&self) -> std::collections::HashSet<u8> {
        self.recognizer.next_byte_set()
    }
}

impl TokenConstraint for GrammarConstraint {
    /// Compute a per-token mask.
    ///
    /// For each token id `t` in `0..vocab_size`:
    /// 1. Decode it to bytes via `tokenizer_decode_fn`.
    /// 2. Clone the current recognizer state.
    /// 3. Feed each byte — if any byte is rejected, mark the token `false`.
    /// 4. If all bytes feed successfully, mark the token `true`.
    ///
    /// Special case: a token with an empty byte sequence is allowed only when
    /// the recognizer is currently in an accepting state (end-of-sequence).
    ///
    /// Short-circuit optimisation: if `next_byte_set()` is empty and the
    /// recognizer is not in an accepting state, return all-false immediately.
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        // Early exit if the recognizer is completely dead.
        if !self.recognizer.is_live() {
            return Some(vec![false; vocab_size]);
        }

        let nbs = self.recognizer.next_byte_set();
        let currently_accepting = self.recognizer.is_accepting();

        // If the recognizer has no valid next bytes and is not accepting,
        // no token can advance the parse.
        if nbs.is_empty() && !currently_accepting {
            return Some(vec![false; vocab_size]);
        }

        let mask: Vec<bool> = (0..vocab_size)
            .map(|token_id| {
                let bytes = (self.tokenizer_decode_fn)(token_id as u32);

                if bytes.is_empty() {
                    // Empty-byte token: allowed only when we are currently accepting.
                    return currently_accepting;
                }

                // Fast pre-check: if the first byte of this token is not in the
                // next_byte_set, skip the expensive clone.
                if !nbs.contains(&bytes[0]) {
                    return false;
                }

                // Clone the recognizer and try feeding all bytes.
                let mut probe = self.recognizer.clone_state();
                for &b in &bytes {
                    if !probe.feed_byte(b) {
                        return false;
                    }
                }
                true
            })
            .collect();

        Some(mask)
    }

    /// Commit `token` to the recognizer by feeding its byte sequence.
    ///
    /// Returns `false` if any byte in the token's sequence is rejected by the
    /// grammar.
    fn advance(&mut self, token: u32) -> bool {
        let bytes = (self.tokenizer_decode_fn)(token);
        if bytes.is_empty() {
            // Empty token: only valid if we are currently accepting.
            return self.recognizer.is_accepting();
        }
        for &b in &bytes {
            if !self.recognizer.feed_byte(b) {
                return false;
            }
        }
        true
    }

    /// Returns `true` when the recognizer is in an accepting state.
    fn is_complete(&self) -> bool {
        self.recognizer.is_accepting()
    }

    /// Reset the recognizer to the initial state.
    fn reset(&mut self) {
        self.recognizer.reset();
    }

    fn name(&self) -> &str {
        "GrammarConstraint"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{arithmetic_grammar, simple_ab_grammar, csv_row_grammar};
    use crate::constrained_decoding::TokenConstraint;

    // ── Minimal ASCII byte-level vocab helper ───────────────────────────────

    /// Build a `GrammarConstraint` with a simple byte-level vocabulary
    /// where token id == ASCII code point (0..128).
    fn ascii_constraint(grammar: Grammar) -> GrammarConstraint {
        GrammarConstraint::new(grammar, |id| {
            if id < 128 { vec![id as u8] } else { vec![] }
        }, 128)
    }

    // ── Arithmetic grammar ──────────────────────────────────────────────────

    #[test]
    fn grammar_constraint_name() {
        let c = ascii_constraint(arithmetic_grammar());
        assert_eq!(c.name(), "GrammarConstraint");
    }

    #[test]
    fn grammar_constraint_not_complete_initially() {
        let c = ascii_constraint(arithmetic_grammar());
        assert!(!c.is_complete());
    }

    #[test]
    fn grammar_constraint_arithmetic_allows_digits_at_start() {
        let c = ascii_constraint(arithmetic_grammar());
        let mask = c.allowed_tokens(&[], 128).unwrap();
        for d in b'0'..=b'9' {
            assert!(mask[d as usize], "digit {d} should be allowed at start");
        }
        assert!(mask[b'(' as usize], "'(' should be allowed at start");
        assert!(!mask[b'+' as usize], "'+' should not be allowed at start");
    }

    #[test]
    fn grammar_constraint_advance_digit_and_operator() {
        let mut c = ascii_constraint(arithmetic_grammar());
        assert!(c.advance(b'1' as u32), "advancing '1' should succeed");
        assert!(c.advance(b'+' as u32), "advancing '+' after '1' should succeed");
    }

    #[test]
    fn grammar_constraint_advance_violation() {
        let mut c = ascii_constraint(arithmetic_grammar());
        let ok = c.advance(b'+' as u32);
        assert!(!ok, "'+' at start should be rejected");
    }

    #[test]
    fn grammar_constraint_complete_after_full_expression() {
        let mut c = ascii_constraint(arithmetic_grammar());
        c.advance(b'1' as u32);
        assert!(c.is_complete(), "single digit is a complete expression");
    }

    #[test]
    fn grammar_constraint_not_complete_after_operator() {
        let mut c = ascii_constraint(arithmetic_grammar());
        c.advance(b'1' as u32);
        c.advance(b'+' as u32);
        assert!(!c.is_complete(), "after '1+' the expression is incomplete");
    }

    #[test]
    fn grammar_constraint_reset() {
        let mut c = ascii_constraint(arithmetic_grammar());
        c.advance(b'5' as u32);
        assert!(c.is_complete());
        c.reset();
        assert!(!c.is_complete());
        assert_eq!(c.bytes_consumed(), 0);
    }

    #[test]
    fn grammar_constraint_full_sequence_1plus2() {
        let mut c = ascii_constraint(arithmetic_grammar());
        assert!(c.advance(b'1' as u32));
        assert!(c.is_complete());
        assert!(c.advance(b'+' as u32));
        assert!(!c.is_complete());
        assert!(c.advance(b'2' as u32));
        assert!(c.is_complete());
    }

    #[test]
    fn grammar_constraint_disallows_after_rejection() {
        let mut c = ascii_constraint(arithmetic_grammar());
        let ok = c.advance(b'+' as u32);
        // After a rejection the recognizer is dead.
        if !ok {
            let mask = c.allowed_tokens(&[], 128).unwrap();
            assert!(mask.iter().all(|&b| !b), "all tokens should be blocked after rejection");
        }
    }

    #[test]
    fn grammar_constraint_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GrammarConstraint>();
    }

    // ── Simple a^n b^n grammar ──────────────────────────────────────────────

    #[test]
    fn grammar_constraint_ab_sequence() {
        let mut c = ascii_constraint(simple_ab_grammar());
        // "ab" should be accepted.
        assert!(c.advance(b'a' as u32));
        assert!(!c.is_complete(), "after 'a' not yet complete");
        assert!(c.advance(b'b' as u32));
        assert!(c.is_complete(), "after 'ab' should be complete");
    }

    #[test]
    fn grammar_constraint_ab_sequence_longer() {
        let mut c = ascii_constraint(simple_ab_grammar());
        // "aabb" should be accepted.
        assert!(c.advance(b'a' as u32));
        assert!(c.advance(b'a' as u32));
        assert!(c.advance(b'b' as u32));
        assert!(c.advance(b'b' as u32));
        assert!(c.is_complete());
    }

    // ── CSV grammar ─────────────────────────────────────────────────────────

    #[test]
    fn grammar_constraint_csv_row() {
        let mut c = ascii_constraint(csv_row_grammar());
        // "a,b" is a valid two-field CSV row.
        for b in b"a,b" {
            assert!(c.advance(*b as u32), "byte {b} should be accepted");
        }
        assert!(c.is_complete());
    }

    #[test]
    fn grammar_constraint_csv_row_single_field() {
        let mut c = ascii_constraint(csv_row_grammar());
        for b in b"hello" {
            assert!(c.advance(*b as u32));
        }
        assert!(c.is_complete());
    }

    // ── Trait object safety ─────────────────────────────────────────────────

    #[test]
    fn grammar_constraint_implements_token_constraint_trait() {
        let c: Box<dyn TokenConstraint> = Box::new(ascii_constraint(arithmetic_grammar()));
        assert_eq!(c.name(), "GrammarConstraint");
        assert!(!c.is_complete());
    }

    // ── Empty byte token ────────────────────────────────────────────────────

    #[test]
    fn grammar_constraint_empty_token_only_when_accepting() {
        // Build a vocab where token 200 maps to empty bytes (special token).
        let g = arithmetic_grammar();
        let c = GrammarConstraint::new(g, |id| {
            if id < 128 { vec![id as u8] }
            else if id == 200 { vec![] }  // special EOS token
            else { vec![] }
        }, 201);

        // Initially not accepting, so token 200 should be blocked.
        let mask = c.allowed_tokens(&[], 201).unwrap();
        assert!(!mask[200], "EOS token should not be allowed when not accepting");
    }

    #[test]
    fn grammar_constraint_empty_token_allowed_when_accepting() {
        let g = arithmetic_grammar();
        let mut c = GrammarConstraint::new(g, |id| {
            if id < 128 { vec![id as u8] }
            else if id == 200 { vec![] }
            else { vec![] }
        }, 201);

        // After generating "9" (a complete expression) we are accepting.
        c.advance(b'9' as u32);
        assert!(c.is_complete());

        let mask = c.allowed_tokens(&[], 201).unwrap();
        assert!(mask[200], "EOS token should be allowed when accepting");
    }
}
