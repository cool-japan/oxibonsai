//! [`GrammarConstraint`] — implements [`TokenConstraint`] using the Earley
//! chart-parser recognizer backed by a BNF context-free grammar.
//!
//! The `allowed_tokens` method speculatively feeds each token's byte sequence
//! through a **clone** of the current recognizer state and marks the token
//! allowed if and only if none of the bytes are rejected.
//!
//! **Phase 16B optimization:** Token byte sequences are precomputed once during
//! construction via `tokenizer_decode_fn` and stored in `token_bytes: Vec<Vec<u8>>`.
//! A `first_byte_index: Box<[Vec<u32>; 256]>` maps each first byte to the list of
//! token IDs that start with that byte. During `allowed_tokens`, only tokens whose
//! first byte is in `next_byte_set` are probed — all others are skipped without
//! invoking the decode function or any recognizer cloning.
//!
//! This reduces the per-step work from O(vocab) decode calls + O(vocab) first-byte
//! checks + O(filtered_vocab × token_len) recognizer probes to just
//! O(|next_byte_set| × avg_matching_tokens × avg_token_len) recognizer probes.

use std::sync::{Arc, Mutex};

use super::ast::Grammar;
use super::cache::AllowedTokensCache;
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
///
/// # Phase 16B: Precomputed byte index
///
/// At construction time, `GrammarConstraint` eagerly calls `tokenizer_decode_fn`
/// for every token ID in `0..vocab_size`, storing the results in `token_bytes`.
/// Simultaneously, `first_byte_index[b]` accumulates the list of token IDs whose
/// first byte is `b`, and `empty_token_ids` collects IDs with empty byte sequences
/// (EOS, padding, special tokens).
///
/// This eliminates O(vocab) decode calls during each `allowed_tokens` call and
/// allows the inner loop to skip entire byte classes not present in
/// `next_byte_set` — often reducing the probed token count by 90–99 %.
pub struct GrammarConstraint {
    /// Original grammar (kept for potential future reset/inspection).
    #[allow(dead_code)]
    grammar: Arc<Grammar>,
    /// Live Earley recognizer tracking the bytes generated so far.
    recognizer: EarleyRecognizer,
    /// Decodes a token id to its raw byte sequence.
    ///
    /// Retained for potential out-of-range token handling or future callers that
    /// need to decode tokens not covered by the initial `0..vocab_size` range.
    #[allow(dead_code)]
    tokenizer_decode_fn: Arc<dyn Fn(u32) -> Vec<u8> + Send + Sync>,
    /// Total vocabulary size used to allocate the precomputed index.
    vocab_size: usize,
    /// LRU memoization cache for `allowed_tokens` results keyed by Earley state hash.
    ///
    /// Wrapped in `Mutex` because `TokenConstraint::allowed_tokens` takes `&self`,
    /// yet cache mutation requires `&mut`.  `Mutex::lock()` returning `PoisonError`
    /// on panic is handled gracefully: cache misses are silent (never panics).
    cache: Mutex<AllowedTokensCache>,

    // ── Phase 16B: Precomputed token index ──────────────────────────────────

    /// Precomputed byte sequences for every token in `0..vocab_size`.
    ///
    /// `token_bytes[id]` is the byte sequence for token `id`, precomputed once
    /// at construction time.  This is the primary data consumed by `allowed_tokens`
    /// and `advance`.
    token_bytes: Vec<Vec<u8>>,

    /// First-byte index: `first_byte_index[b]` is the list of token IDs
    /// (in `0..vocab_size`) whose first byte equals `b`.
    ///
    /// Boxed to avoid stack-allocating 256 `Vec<u32>`s (which may trigger a
    /// stack overflow for large vectors on some platforms).
    first_byte_index: Box<[Vec<u32>; 256]>,

    /// Token IDs (in `0..vocab_size`) whose byte sequence is empty.
    ///
    /// These represent EOS tokens, padding tokens, and other special tokens
    /// that do not contribute bytes to the grammar stream.  They are allowed
    /// only when the recognizer is in an accepting state.
    empty_token_ids: Vec<u32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private construction helper
// ─────────────────────────────────────────────────────────────────────────────

/// Type alias for the first-byte index (avoids clippy::type_complexity).
type FirstByteIndex = Box<[Vec<u32>; 256]>;

/// Aggregate result of `build_token_index`.
struct TokenIndex {
    token_bytes: Vec<Vec<u8>>,
    first_byte_index: FirstByteIndex,
    empty_token_ids: Vec<u32>,
}

/// Build the three precomputed structures from a decode function and vocab size.
fn build_token_index(decode_fn: &dyn Fn(u32) -> Vec<u8>, vocab_size: usize) -> TokenIndex {
    let mut token_bytes: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);

    // Use a Vec<Vec<u32>> of length 256 to avoid constructing 256 Vecs on the
    // stack before boxing — the std::array::from_fn approach would stack-allocate
    // [Vec<u32>; 256] = ~3 KB, which is fine, but building it element-by-element
    // via a Vec before converting avoids any platform-specific stack pressure.
    let mut raw_index: Vec<Vec<u32>> = (0..256_usize).map(|_| Vec::new()).collect();
    let mut empty_token_ids: Vec<u32> = Vec::new();

    for id in 0..vocab_size as u32 {
        let bytes = decode_fn(id);
        match bytes.first() {
            Some(&b) => raw_index[b as usize].push(id),
            None => empty_token_ids.push(id),
        }
        token_bytes.push(bytes);
    }

    // Convert Vec<Vec<u32>> (length 256) into Box<[Vec<u32>; 256]>.
    // We built `raw_index` with exactly 256 elements, so the try_into cannot fail.
    let first_byte_index: FirstByteIndex = raw_index
        .into_boxed_slice()
        .try_into()
        .expect("raw_index must have exactly 256 elements");

    TokenIndex {
        token_bytes,
        first_byte_index,
        empty_token_ids,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

impl GrammarConstraint {
    /// Create a new `GrammarConstraint`.
    ///
    /// The `grammar` is normalised (multi-byte terminals split into chains)
    /// and wrapped in an `Arc` before being handed to the recognizer.
    ///
    /// **Phase 16B:** This eagerly calls `tokenizer_decode_fn(id)` for every
    /// `id` in `0..vocab_size`, building `token_bytes` and `first_byte_index`.
    /// Construction cost is O(vocab_size × avg_decode_cost); subsequent
    /// `allowed_tokens` calls no longer call the decode function at all.
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
        let tokenizer_decode_fn: Arc<dyn Fn(u32) -> Vec<u8> + Send + Sync> =
            Arc::new(tokenizer_decode_fn);

        let idx = build_token_index(tokenizer_decode_fn.as_ref(), vocab_size);

        Self {
            grammar,
            recognizer,
            tokenizer_decode_fn,
            vocab_size,
            cache: Mutex::new(AllowedTokensCache::with_capacity(256)),
            token_bytes: idx.token_bytes,
            first_byte_index: idx.first_byte_index,
            empty_token_ids: idx.empty_token_ids,
        }
    }

    /// Create a new `GrammarConstraint` with a custom cache capacity.
    ///
    /// Identical to [`new`](Self::new) except that the LRU cache is initialised
    /// with `capacity` entries rather than the default 256.  Use a larger value
    /// when the grammar has many distinct parse states; use a smaller value to
    /// bound memory at the cost of more cache misses.
    ///
    /// **Phase 16B:** Same eager precomputation as [`new`](Self::new).
    ///
    /// # Parameters
    ///
    /// * `grammar`               — the context-free grammar to enforce
    /// * `tokenizer_decode_fn`   — maps token id → byte sequence
    /// * `vocab_size`            — total vocabulary size
    /// * `capacity`              — LRU cache capacity (clamped to ≥ 1)
    pub fn with_cache_capacity(
        mut grammar: Grammar,
        tokenizer_decode_fn: impl Fn(u32) -> Vec<u8> + Send + Sync + 'static,
        vocab_size: usize,
        capacity: usize,
    ) -> Self {
        grammar.normalise_terminals();
        let grammar = Arc::new(grammar);
        let recognizer = EarleyRecognizer::new(Arc::clone(&grammar));
        let tokenizer_decode_fn: Arc<dyn Fn(u32) -> Vec<u8> + Send + Sync> =
            Arc::new(tokenizer_decode_fn);

        let idx = build_token_index(tokenizer_decode_fn.as_ref(), vocab_size);

        Self {
            grammar,
            recognizer,
            tokenizer_decode_fn,
            vocab_size,
            cache: Mutex::new(AllowedTokensCache::with_capacity(capacity)),
            token_bytes: idx.token_bytes,
            first_byte_index: idx.first_byte_index,
            empty_token_ids: idx.empty_token_ids,
        }
    }

    /// Return cache hit/miss statistics as `(hits, misses)`.
    ///
    /// Useful for testing and for monitoring cache effectiveness in production.
    /// Returns `(0, 0)` if the internal `Mutex` has been poisoned (never panics).
    pub fn cache_stats(&self) -> (u64, u64) {
        self.cache
            .lock()
            .map(|c| (c.hits(), c.misses()))
            .unwrap_or((0, 0))
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

    /// Return the vocabulary size passed to the constructor.
    ///
    /// This equals `self.token_bytes.len()`.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Return an estimate of the heap memory (in bytes) occupied by the
    /// precomputed token index built during construction.
    ///
    /// The estimate accounts for:
    /// * `token_bytes`: 24-byte `Vec` header + inline byte storage per token.
    /// * `first_byte_index`: 24-byte `Vec` header + 4-byte u32 per entry,
    ///   for all 256 first-byte buckets.
    /// * `empty_token_ids`: 4 bytes per entry.
    ///
    /// This is a lower bound (does not include allocator overhead or padding).
    pub fn index_memory_bytes(&self) -> usize {
        // 24 = size_of::<Vec<u8>>() on 64-bit platforms (ptr + len + cap).
        let token_bytes_mem: usize = self.token_bytes.iter().map(|b| b.len() + 24).sum();
        // 24 = size_of::<Vec<u32>>(); 4 = size_of::<u32>().
        let index_mem: usize = self
            .first_byte_index
            .iter()
            .map(|v| v.len() * 4 + 24)
            .sum();
        token_bytes_mem + index_mem + self.empty_token_ids.len() * 4
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenConstraint implementation
// ─────────────────────────────────────────────────────────────────────────────

impl TokenConstraint for GrammarConstraint {
    /// Compute a per-token mask using the precomputed first-byte index.
    ///
    /// **Phase 16B algorithm:**
    ///
    /// 1. If the recognizer is dead, return all-false immediately.
    /// 2. Compute `next_byte_set` (NBS) and `is_accepting`.
    /// 3. If NBS is empty and not accepting, return all-false immediately.
    /// 4. Check the LRU cache keyed by `state_hash()`.
    /// 5. On cache miss: start with an all-false mask.
    ///    * For each `first_byte` in NBS, iterate `first_byte_index[first_byte]`
    ///      and probe only those tokens via `recognizer.clone_state()`.
    ///    * For empty-byte tokens (EOS/special), allow them iff `is_accepting`.
    /// 6. Insert the result into the LRU cache.
    ///
    /// The inner loop never calls `tokenizer_decode_fn` — it reads precomputed
    /// `token_bytes` instead.  Tokens whose first byte is NOT in NBS are never
    /// visited at all.
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        // ── Early exits ─────────────────────────────────────────────────────
        if !self.recognizer.is_live() {
            return Some(vec![false; vocab_size]);
        }

        let nbs = self.recognizer.next_byte_set();
        let currently_accepting = self.recognizer.is_accepting();

        if nbs.is_empty() && !currently_accepting {
            return Some(vec![false; vocab_size]);
        }

        // ── Cache lookup ─────────────────────────────────────────────────────
        let state_hash = self.recognizer.state_hash();
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(cached) = cache.get(state_hash) {
                return Some(cached.to_vec());
            }
        }

        // ── Cache miss: build mask using first-byte index ────────────────────
        let mut mask = vec![false; vocab_size];

        // Empty-byte tokens (EOS, special): allowed only when accepting.
        if currently_accepting {
            for &id in &self.empty_token_ids {
                if (id as usize) < vocab_size {
                    mask[id as usize] = true;
                }
            }
        }

        // Tokens grouped by first byte: iterate only over bytes that are in NBS.
        for &first_byte in &nbs {
            for &token_id in &self.first_byte_index[first_byte as usize] {
                let token_idx = token_id as usize;
                if token_idx >= vocab_size {
                    continue;
                }
                let bytes = &self.token_bytes[token_idx];
                if bytes.is_empty() {
                    // Should not happen (empties are in empty_token_ids), but
                    // handle defensively.
                    if currently_accepting {
                        mask[token_idx] = true;
                    }
                    continue;
                }
                // bytes[0] == first_byte by construction — no need to re-check.
                // Probe the remaining bytes via a cloned recognizer state.
                let mut probe = self.recognizer.clone_state();
                let mut ok = true;
                for &b in bytes {
                    if !probe.feed_byte(b) {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    mask[token_idx] = true;
                }
            }
        }

        // ── Store in cache ───────────────────────────────────────────────────
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(state_hash, mask.clone());
        }

        Some(mask)
    }

    /// Commit `token` to the recognizer by feeding its precomputed byte sequence.
    ///
    /// Uses the precomputed `token_bytes` slice instead of calling
    /// `tokenizer_decode_fn`, avoiding one decode call per accepted token.
    ///
    /// Returns `false` if any byte in the token's sequence is rejected by the
    /// grammar, or if the token ID is out of range for the precomputed index.
    fn advance(&mut self, token: u32) -> bool {
        let Some(bytes) = self.token_bytes.get(token as usize) else {
            // Token ID is beyond the precomputed vocab range.
            // Treat as empty → allowed only if currently accepting.
            return self.recognizer.is_accepting();
        };
        if bytes.is_empty() {
            return self.recognizer.is_accepting();
        }
        for &b in bytes {
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
    use crate::constrained_decoding::TokenConstraint;
    use crate::grammar::{arithmetic_grammar, csv_row_grammar, simple_ab_grammar};

    // ── Minimal ASCII byte-level vocab helper ───────────────────────────────

    /// Build a `GrammarConstraint` with a simple byte-level vocabulary
    /// where token id == ASCII code point (0..128).
    fn ascii_constraint(grammar: Grammar) -> GrammarConstraint {
        GrammarConstraint::new(
            grammar,
            |id| {
                if id < 128 {
                    vec![id as u8]
                } else {
                    vec![]
                }
            },
            128,
        )
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
        assert!(
            c.advance(b'+' as u32),
            "advancing '+' after '1' should succeed"
        );
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
            assert!(
                mask.iter().all(|&b| !b),
                "all tokens should be blocked after rejection"
            );
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
        let c = GrammarConstraint::new(
            g,
            |id| {
                if id < 128 {
                    vec![id as u8]
                } else {
                    vec![] // id == 200 is EOS; all non-ASCII ids map to empty
                }
            },
            201,
        );

        // Initially not accepting, so token 200 should be blocked.
        let mask = c.allowed_tokens(&[], 201).unwrap();
        assert!(
            !mask[200],
            "EOS token should not be allowed when not accepting"
        );
    }

    #[test]
    fn grammar_constraint_empty_token_allowed_when_accepting() {
        let g = arithmetic_grammar();
        let mut c = GrammarConstraint::new(
            g,
            |id| {
                if id < 128 {
                    vec![id as u8]
                } else {
                    vec![] // id == 200 is EOS; all non-ASCII ids map to empty
                }
            },
            201,
        );

        // After generating "9" (a complete expression) we are accepting.
        c.advance(b'9' as u32);
        assert!(c.is_complete());

        let mask = c.allowed_tokens(&[], 201).unwrap();
        assert!(mask[200], "EOS token should be allowed when accepting");
    }

    // ── Phase 16B: vocab_size accessor ──────────────────────────────────────

    #[test]
    fn grammar_constraint_vocab_size_accessor() {
        let c = ascii_constraint(arithmetic_grammar());
        assert_eq!(c.vocab_size(), 128);

        let c2 = GrammarConstraint::new(arithmetic_grammar(), |id| vec![id as u8], 512);
        assert_eq!(c2.vocab_size(), 512);
    }

    // ── Phase 16B: index_memory_bytes ───────────────────────────────────────

    #[test]
    fn grammar_constraint_index_memory_nonzero() {
        let c = ascii_constraint(arithmetic_grammar());
        assert!(
            c.index_memory_bytes() > 0,
            "index_memory_bytes must be > 0 for vocab_size > 0"
        );
    }

    #[test]
    fn grammar_constraint_index_memory_zero_vocab() {
        // vocab_size == 0 → token_bytes is empty, but first_byte_index still
        // holds 256 empty Vecs (each 24 bytes header).
        let c = GrammarConstraint::new(arithmetic_grammar(), |_id| vec![], 0);
        // 256 empty Vec<u32> × 24 bytes each = 6144 bytes minimum.
        assert_eq!(c.index_memory_bytes(), 256 * 24);
    }

    // ── Phase 16B: first-byte index correctness ──────────────────────────────

    #[test]
    fn grammar_constraint_digits_allowed_at_start_via_index() {
        // The arithmetic grammar starts with digits and '('.
        // Verify that the index path produces the same mask as the old path.
        let c = ascii_constraint(arithmetic_grammar());
        let mask = c.allowed_tokens(&[], 128).unwrap();

        for d in b'0'..=b'9' {
            assert!(
                mask[d as usize],
                "digit token {} should be allowed at start",
                d as char
            );
        }
        assert!(mask[b'(' as usize], "'(' should be allowed at start");
        // Non-first-byte tokens must be blocked.
        assert!(!mask[b'+' as usize], "'+' not valid at start");
        assert!(!mask[b' ' as usize], "space not valid at start");
        assert!(!mask[b'z' as usize], "'z' not valid at start");
    }

    #[test]
    fn grammar_constraint_advance_uses_cached_bytes() {
        // Verify that advance() via cached bytes works identically to the
        // old tokenizer_decode_fn path by checking recognizer state advancement.
        let mut c = ascii_constraint(arithmetic_grammar());

        // Feed "1+2" token by token.
        assert!(c.advance(b'1' as u32), "'1' should advance");
        assert!(c.is_complete(), "single digit is complete");
        assert!(c.advance(b'+' as u32), "'+' should advance after digit");
        assert!(!c.is_complete(), "incomplete after '+'");
        assert!(c.advance(b'2' as u32), "'2' should advance");
        assert!(c.is_complete(), "'1+2' is a complete expression");

        // Verify bytes_consumed reflects all bytes fed.
        assert_eq!(c.bytes_consumed(), 3, "3 bytes should have been consumed");
    }

    #[test]
    fn grammar_constraint_advance_out_of_range_token() {
        // Token ID beyond vocab_size (128) uses the "treat as accepting" fallback.
        let c = ascii_constraint(arithmetic_grammar());
        // At initial state, recognizer is NOT accepting → out-of-range token returns false.
        let mut c_mut = ascii_constraint(arithmetic_grammar());
        let ok = c_mut.advance(999); // well beyond vocab_size=128
        assert!(!ok, "out-of-range token should return false when not accepting");

        drop(c);

        // After advancing to an accepting state, out-of-range token returns true.
        let mut c2 = ascii_constraint(arithmetic_grammar());
        c2.advance(b'5' as u32); // now accepting
        assert!(c2.is_complete());
        let ok2 = c2.advance(999);
        assert!(ok2, "out-of-range token should return true when accepting");
    }

    // ── Phase 16B: precomputed bytes match decode fn ─────────────────────────

    #[test]
    fn grammar_constraint_precomputed_bytes_match_decode_fn() {
        // Verify token_bytes[id] == direct decode for all ids 0..128.
        let decode_fn = |id: u32| -> Vec<u8> {
            if id < 128 {
                vec![id as u8]
            } else {
                vec![]
            }
        };
        let c = GrammarConstraint::new(arithmetic_grammar(), decode_fn, 128);

        for id in 0u32..128 {
            let precomputed = &c.token_bytes[id as usize];
            let direct = if id < 128 { vec![id as u8] } else { vec![] };
            assert_eq!(
                precomputed, &direct,
                "precomputed bytes for token {id} must match direct decode"
            );
        }
    }
}
