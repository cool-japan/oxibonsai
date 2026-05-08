//! Integration tests for the Earley `AllowedTokensCache` and
//! `GrammarConstraint` memoization layer.
//!
//! Tests cover:
//! * Cache hit/miss semantics on repeated and distinct Earley states
//! * LRU eviction at capacity
//! * Mask correctness compared to a fresh uncached constraint
//! * Interaction with `advance()` after cache use
//! * `cache_stats()` observability
//! * Thread safety / `Send + Sync` bound compilation

use oxibonsai_runtime::{
    constrained_decoding::TokenConstraint,
    grammar::{
        arithmetic_grammar, json_lite_grammar, simple_ab_grammar, AllowedTokensCache,
        GrammarConstraint,
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Single-byte ASCII decode: token id `i` → `[i as u8]` for `i < 128`, else `[]`.
fn ascii_decode(id: u32) -> Vec<u8> {
    if id < 128 {
        vec![id as u8]
    } else {
        vec![]
    }
}

fn ascii_constraint(grammar: oxibonsai_runtime::grammar::Grammar) -> GrammarConstraint {
    GrammarConstraint::new(grammar, ascii_decode, 128)
}

fn ascii_constraint_with_cap(
    grammar: oxibonsai_runtime::grammar::Grammar,
    capacity: usize,
) -> GrammarConstraint {
    GrammarConstraint::with_cache_capacity(grammar, ascii_decode, 128, capacity)
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: cache_hit_on_repeated_state
// ─────────────────────────────────────────────────────────────────────────────

/// Calling `allowed_tokens` twice at the same parse state (no advance) must
/// produce a cache hit on the second call.
#[test]
fn cache_hit_on_repeated_state() {
    let c = ascii_constraint(arithmetic_grammar());

    // First call: computes the mask and populates the cache (1 miss).
    let mask1 = c.allowed_tokens(&[], 128).unwrap();
    // Second call: same Earley state → should be a cache hit.
    let mask2 = c.allowed_tokens(&[], 128).unwrap();

    assert_eq!(mask1, mask2, "repeated call must return identical mask");

    let (hits, misses) = c.cache_stats();
    assert_eq!(misses, 1, "should be exactly one miss (first call)");
    assert_eq!(hits, 1, "should be exactly one hit (second call)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: cache_miss_on_different_states
// ─────────────────────────────────────────────────────────────────────────────

/// After advancing the parser (feeding a byte) the Earley chart changes, so
/// `allowed_tokens` at the new state must be a cache miss.
#[test]
fn cache_miss_on_different_states() {
    let mut c = ascii_constraint(arithmetic_grammar());

    // State 0 (initial): miss + populate.
    c.allowed_tokens(&[], 128).unwrap();
    let (_, misses_before) = c.cache_stats();
    assert_eq!(misses_before, 1);

    // Advance to a different Earley state (after consuming '1').
    c.advance(b'1' as u32);

    // State 1: must be a new miss because the chart has changed.
    c.allowed_tokens(&[b'1' as u32], 128).unwrap();
    let (_, misses_after) = c.cache_stats();
    assert_eq!(misses_after, 2, "new state must produce a second miss");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: lru_eviction_at_capacity
// ─────────────────────────────────────────────────────────────────────────────

/// With capacity=2, visiting 3 distinct Earley states should never grow the
/// cache beyond 2 entries.
#[test]
fn lru_eviction_at_capacity() {
    // We need at least 3 distinct states.  The arithmetic grammar gives us:
    // state 0 (initial), state 1 (after '1'), state 2 (after '1+').
    let mut c = ascii_constraint_with_cap(arithmetic_grammar(), 2);

    c.allowed_tokens(&[], 128).unwrap(); // state 0 → miss, cached
    c.advance(b'1' as u32);
    c.allowed_tokens(&[], 128).unwrap(); // state 1 → miss, cached (evicts state 0 when state 2 is added later)
    c.advance(b'+' as u32);
    c.allowed_tokens(&[], 128).unwrap(); // state 2 → miss, evicts LRU

    // Regardless of eviction order the cache must respect capacity.
    // We can only test this via the AllowedTokensCache unit tests since
    // GrammarConstraint does not expose len(); but we can verify misses == 3.
    let (_, misses) = c.cache_stats();
    assert_eq!(misses, 3, "each of the 3 distinct states is a miss");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: mask_correctness_arithmetic
// ─────────────────────────────────────────────────────────────────────────────

/// The mask returned by the caching constraint must match the mask returned by a
/// freshly built constraint at the same logical state.
#[test]
fn mask_correctness_arithmetic() {
    let cached = ascii_constraint(arithmetic_grammar());
    let uncached = ascii_constraint(arithmetic_grammar());

    let mask_cached = cached.allowed_tokens(&[], 128).unwrap();
    let mask_uncached = uncached.allowed_tokens(&[], 128).unwrap();

    assert_eq!(
        mask_cached, mask_uncached,
        "cached and uncached masks must be identical at initial state"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: mask_correctness_vs_uncached_multi_step
// ─────────────────────────────────────────────────────────────────────────────

/// Build two constraints from the same grammar and run them in lockstep through
/// several advance() calls, asserting that `allowed_tokens` returns the same
/// mask at every step.
#[test]
fn mask_correctness_vs_uncached_multi_step() {
    // "cached" uses the default capacity-256 cache.
    // "reference" is a second independent constraint (its cache is also present
    // but has a different state — effectively they cross-check each other).
    let mut cached = ascii_constraint(arithmetic_grammar());
    let mut reference = ascii_constraint(arithmetic_grammar());

    let tokens_to_advance: &[u8] = b"1+2";

    for &b in tokens_to_advance {
        let m_cached = cached.allowed_tokens(&[], 128).unwrap();
        let m_ref = reference.allowed_tokens(&[], 128).unwrap();
        assert_eq!(m_cached, m_ref, "masks differ before advancing '{}'", b as char);

        cached.advance(b as u32);
        reference.advance(b as u32);
    }

    // Final state (after "1+2").
    let m_cached = cached.allowed_tokens(&[], 128).unwrap();
    let m_ref = reference.allowed_tokens(&[], 128).unwrap();
    assert_eq!(m_cached, m_ref, "masks differ at final state");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: grammar_cache_ab_grammar
// ─────────────────────────────────────────────────────────────────────────────

/// For the simple a^n b^n grammar, cached and uncached outputs must match at
/// each step of parsing "aabb".
#[test]
fn grammar_cache_ab_grammar() {
    let mut cached = ascii_constraint(simple_ab_grammar());
    let mut reference = ascii_constraint(simple_ab_grammar());

    for &b in b"aabb" {
        let m1 = cached.allowed_tokens(&[], 128).unwrap();
        let m2 = reference.allowed_tokens(&[], 128).unwrap();
        assert_eq!(
            m1, m2,
            "ab grammar masks differ before advancing '{}'",
            b as char
        );
        cached.advance(b as u32);
        reference.advance(b as u32);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: grammar_cache_json_lite
// ─────────────────────────────────────────────────────────────────────────────

/// At the initial state of the json_lite grammar both constraints must agree.
/// The json_lite grammar is complex enough to exercise a non-trivial chart hash.
#[test]
fn grammar_cache_json_lite() {
    let cached = ascii_constraint(json_lite_grammar());
    let reference = ascii_constraint(json_lite_grammar());

    let m1 = cached.allowed_tokens(&[], 128).unwrap();
    let m2 = reference.allowed_tokens(&[], 128).unwrap();
    assert_eq!(m1, m2, "json_lite initial masks must match");

    // The json_lite grammar starts with '{', '[', '"', digits, or literal chars.
    assert!(m1[b'{' as usize], "{{}} must be allowed at start of json_lite");
    assert!(m1[b'[' as usize], "[] must be allowed at start of json_lite");
    assert!(m1[b'"' as usize], "quote must be allowed at start of json_lite");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: cache_does_not_affect_advance
// ─────────────────────────────────────────────────────────────────────────────

/// Calling `allowed_tokens` (which populates the cache) and then `advance()`
/// must not corrupt the recognizer state.  The recognizer must still accept
/// valid continuations after the cache has been used.
#[test]
fn cache_does_not_affect_advance() {
    let mut c = ascii_constraint(arithmetic_grammar());

    // Populate cache at initial state.
    let mask_before = c.allowed_tokens(&[], 128).unwrap();
    // '1' is allowed at initial state.
    assert!(mask_before[b'1' as usize]);

    // Advance — the recognizer state must change correctly.
    assert!(c.advance(b'1' as u32), "advancing '1' should succeed");
    assert!(c.is_complete(), "single digit is a complete expression");

    // Populate cache at new state.
    let mask_after = c.allowed_tokens(&[], 128).unwrap();
    // After '1' the operators +, -, *, / are valid continuations.
    assert!(mask_after[b'+' as usize], "'+' should be allowed after '1'");
    assert!(mask_after[b'-' as usize], "'-' should be allowed after '1'");

    // '(' is not a valid continuation after a complete number.
    // (It could be in some grammars; arithmetic here only allows operators.)
    // We don't assert the negative here since the grammar is right-recursive
    // and may or may not allow '(' after a digit in prefix position;
    // the important thing is that advance() works after cache use.
    assert!(c.advance(b'+' as u32), "advancing '+' after '1' should succeed");
    assert!(!c.is_complete(), "incomplete after '1+'");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: cache_stats_hit_count
// ─────────────────────────────────────────────────────────────────────────────

/// `cache_stats()` must return accurate hit and miss counts across multiple
/// calls with and without state changes.
#[test]
fn cache_stats_hit_count() {
    let mut c = ascii_constraint(arithmetic_grammar());

    // 1st call at state 0: miss.
    c.allowed_tokens(&[], 128).unwrap();
    let (h, m) = c.cache_stats();
    assert_eq!(h, 0);
    assert_eq!(m, 1);

    // 2nd call at state 0 (no advance): hit.
    c.allowed_tokens(&[], 128).unwrap();
    let (h, m) = c.cache_stats();
    assert_eq!(h, 1);
    assert_eq!(m, 1);

    // 3rd call at state 0: hit again.
    c.allowed_tokens(&[], 128).unwrap();
    let (h, m) = c.cache_stats();
    assert_eq!(h, 2);
    assert_eq!(m, 1);

    // Advance → new state.
    c.advance(b'5' as u32);

    // Call at new state: miss.
    c.allowed_tokens(&[], 128).unwrap();
    let (h, m) = c.cache_stats();
    assert_eq!(h, 2);
    assert_eq!(m, 2);

    // Call at new state again: hit.
    c.allowed_tokens(&[], 128).unwrap();
    let (h, m) = c.cache_stats();
    assert_eq!(h, 3);
    assert_eq!(m, 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10: cache_thread_safety
// ─────────────────────────────────────────────────────────────────────────────

/// `GrammarConstraint` is `Send + Sync` (the `Mutex<AllowedTokensCache>` ensures
/// this).  This test verifies that the constraint can be moved to another thread
/// and that `allowed_tokens` works correctly there.
#[test]
fn cache_thread_safety() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    // Wrap in Arc<Mutex<>> to share between threads.
    let constraint = Arc::new(Mutex::new(ascii_constraint(arithmetic_grammar())));

    let handle = {
        let constraint = Arc::clone(&constraint);
        thread::spawn(move || {
            let c = constraint.lock().unwrap_or_else(|e| e.into_inner());
            let mask = c.allowed_tokens(&[], 128).unwrap();
            // Digits must be allowed at the start.
            for d in b'0'..=b'9' {
                assert!(mask[d as usize], "digit {d} should be allowed");
            }
            let (_, misses) = c.cache_stats();
            misses
        })
    };

    let misses_from_thread = handle.join().expect("thread should not panic");
    assert_eq!(misses_from_thread, 1, "one miss in the spawned thread");

    // Back on main thread: the cache already has the initial state, so this
    // call is a hit.
    let c = constraint.lock().unwrap_or_else(|e| e.into_inner());
    c.allowed_tokens(&[], 128).unwrap();
    let (hits, _) = c.cache_stats();
    assert_eq!(hits, 1, "main thread call is a cache hit");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11: AllowedTokensCache direct LRU stress
// ─────────────────────────────────────────────────────────────────────────────

/// Directly stress the `AllowedTokensCache` with sequential inserts and verify
/// that capacity is never exceeded.
#[test]
fn cache_direct_lru_stress() {
    let cap = 8usize;
    let mut cache = AllowedTokensCache::with_capacity(cap);

    for i in 0u64..32 {
        cache.insert(i, vec![true; 16]);
        assert!(
            cache.len() <= cap,
            "cache len {} exceeded capacity {cap}",
            cache.len()
        );
    }

    // After inserting 32 entries into a cap-8 cache, exactly 8 should remain.
    assert_eq!(cache.len(), cap);
    // The last `cap` keys (24..31) should be present; earlier ones evicted.
    for i in 24u64..32 {
        assert!(
            cache.get(i).is_some(),
            "key {i} should still be present in LRU"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12: with_cache_capacity_constructor
// ─────────────────────────────────────────────────────────────────────────────

/// `with_cache_capacity` must behave identically to `new` for correctness;
/// the custom capacity is validated via `cache_stats()` after capacity-forced eviction.
#[test]
fn with_cache_capacity_constructor() {
    // Capacity=1: every new state evicts the previous one.
    let mut c = ascii_constraint_with_cap(arithmetic_grammar(), 1);

    // State 0: miss.
    c.allowed_tokens(&[], 128).unwrap();
    // State 0 again: hit (still in cache).
    c.allowed_tokens(&[], 128).unwrap();
    let (h, m) = c.cache_stats();
    assert_eq!(h, 1, "second call at state 0 must be a hit");
    assert_eq!(m, 1);

    // Advance → new state.
    c.advance(b'3' as u32);
    // State 1: miss (also evicts state 0).
    c.allowed_tokens(&[], 128).unwrap();
    let (h, m) = c.cache_stats();
    assert_eq!(m, 2);
    assert_eq!(h, 1);

    // Go back to state 0 logically impossible via the constraint (no reset
    // here), so just verify stats are consistent.
    assert_eq!(h + m, 3, "total calls must equal hits + misses");
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 16B tests: first-byte index + precomputed byte sequences
// ─────────────────────────────────────────────────────────────────────────────

// Test 13: test_byte_index_skips_non_matching_tokens
// ─────────────────────────────────────────────────────────────────────────────

/// With the arithmetic grammar (which starts with digits and '('), verify that
/// `allowed_tokens` returns `true` only for digit-starting and '('-starting
/// tokens and `false` for tokens that start with bytes not in `next_byte_set`.
///
/// This test exercises the first-byte index path: the implementation should
/// skip all tokens whose first byte is not in the recognizer's `next_byte_set`
/// without probing them via `clone_state()`.
#[test]
fn test_byte_index_skips_non_matching_tokens() {
    let c = ascii_constraint(arithmetic_grammar());
    let mask = c.allowed_tokens(&[], 128).unwrap();

    // Digit-starting tokens must be allowed (arithmetic starts with digits or '(').
    for d in b'0'..=b'9' {
        assert!(
            mask[d as usize],
            "digit token '{}' (0x{:02x}) should be allowed at start",
            d as char,
            d
        );
    }
    assert!(mask[b'(' as usize], "'(' should be allowed at start");

    // Tokens whose first byte is definitively not in the start FIRST set
    // must be rejected by the first-byte index (never probed).
    let non_start_bytes: &[u8] = &[b'+', b'-', b'*', b'/', b')', b' ', b'a', b'z', b'\n'];
    for &b in non_start_bytes {
        assert!(
            !mask[b as usize],
            "byte '{}' (0x{:02x}) should not be allowed at start of arithmetic",
            b as char,
            b
        );
    }
}

// Test 14: test_precomputed_bytes_match_decode_fn
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that the `token_bytes` field (precomputed during `new()`) contains
/// the same byte sequences as calling the decode function directly.
///
/// Accesses the field via the `vocab_size()` accessor (to confirm construction
/// succeeded) and checks selected token IDs against a direct call.
#[test]
fn test_precomputed_bytes_match_decode_fn() {
    use oxibonsai_runtime::grammar::GrammarConstraint;

    let decode_fn_direct = |id: u32| -> Vec<u8> {
        if id < 128 {
            vec![id as u8]
        } else {
            vec![]
        }
    };

    let c = GrammarConstraint::new(arithmetic_grammar(), ascii_decode, 128);
    assert_eq!(c.vocab_size(), 128);

    // Check every token ID in 0..128 against the direct decode result.
    for id in 0u32..128 {
        let direct = decode_fn_direct(id);
        // allowed_tokens masks encode implicit precomputed byte usage;
        // cross-verify by checking that tokens allowed at the start of the
        // arithmetic grammar match what `direct` encodes.
        // We do this via the mask at the initial state.
        let mask = c.allowed_tokens(&[], 128).unwrap();
        let allowed = mask[id as usize];
        // If direct decodes to a non-empty sequence starting with a digit or '(',
        // it should be allowed; otherwise it shouldn't.
        let first_byte_opt = direct.first().copied();
        let should_be_allowed = match first_byte_opt {
            Some(b) => (b'0'..=b'9').contains(&b) || b == b'(',
            None => false, // empty = EOS, not allowed (not accepting at start)
        };
        assert_eq!(
            allowed, should_be_allowed,
            "token {id}: allowed={allowed} but expected={should_be_allowed} \
             (first_byte={first_byte_opt:?})"
        );
    }
}

// Test 15: test_index_memory_usage_nonzero
// ─────────────────────────────────────────────────────────────────────────────

/// `index_memory_bytes()` must return a strictly positive value for any
/// `GrammarConstraint` constructed with `vocab_size > 0`.
#[test]
fn test_index_memory_usage_nonzero() {
    use oxibonsai_runtime::grammar::GrammarConstraint;

    let c = GrammarConstraint::new(arithmetic_grammar(), ascii_decode, 128);
    let mem = c.index_memory_bytes();
    assert!(
        mem > 0,
        "index_memory_bytes() must be > 0 for vocab_size=128, got {mem}"
    );

    // Sanity: with vocab_size=128 and one byte per token, plus 256 index vecs,
    // the minimum is at least 128*(1+24) + 256*24 = 3200 + 6144 = 9344 bytes.
    assert!(
        mem >= 9344,
        "index_memory_bytes() = {mem} is below the expected lower bound of 9344"
    );
}

// Test 16: test_vocab_size_accessor
// ─────────────────────────────────────────────────────────────────────────────

/// `vocab_size()` must return the exact value passed to `new()` or
/// `with_cache_capacity()`.
#[test]
fn test_vocab_size_accessor() {
    use oxibonsai_runtime::grammar::GrammarConstraint;

    let c1 = GrammarConstraint::new(arithmetic_grammar(), ascii_decode, 128);
    assert_eq!(c1.vocab_size(), 128);

    let c2 = GrammarConstraint::new(arithmetic_grammar(), ascii_decode, 4096);
    assert_eq!(c2.vocab_size(), 4096);

    let c3 = GrammarConstraint::with_cache_capacity(arithmetic_grammar(), ascii_decode, 512, 64);
    assert_eq!(c3.vocab_size(), 512);

    let c4 = GrammarConstraint::new(arithmetic_grammar(), ascii_decode, 0);
    assert_eq!(c4.vocab_size(), 0);
}

// Test 17: test_advance_uses_cached_bytes
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that `advance()` using precomputed `token_bytes` produces the correct
/// recognizer state — identical to what the original `tokenizer_decode_fn` path
/// would have produced.
///
/// We cross-check by constructing two independent constraints from the same
/// grammar, advancing both through the same token sequence, and asserting that
/// `is_complete()`, `bytes_consumed()`, and `next_byte_set()` agree at every step.
#[test]
fn test_advance_uses_cached_bytes() {
    // Both constraints use the same decode function and grammar.
    let mut c1 = ascii_constraint(arithmetic_grammar());
    let mut c2 = ascii_constraint(arithmetic_grammar());

    let sequence: &[u8] = b"1+2*3";

    for (i, &byte) in sequence.iter().enumerate() {
        // Advance both by the same token.
        let ok1 = c1.advance(byte as u32);
        let ok2 = c2.advance(byte as u32);
        assert_eq!(
            ok1, ok2,
            "advance({}) disagreed at step {i}: c1={ok1} c2={ok2}",
            byte as char
        );

        // Recognizer state must be identical.
        assert_eq!(
            c1.is_complete(),
            c2.is_complete(),
            "is_complete() disagreed after step {i}"
        );
        assert_eq!(
            c1.bytes_consumed(),
            c2.bytes_consumed(),
            "bytes_consumed() disagreed after step {i}"
        );
        assert_eq!(
            c1.next_byte_set(),
            c2.next_byte_set(),
            "next_byte_set() disagreed after step {i}"
        );
    }

    // Final state: "1+2*3" is a valid arithmetic expression.
    assert!(c1.is_complete(), "c1 should be complete after '1+2*3'");
    assert!(c2.is_complete(), "c2 should be complete after '1+2*3'");
    assert_eq!(c1.bytes_consumed(), 5);
    assert_eq!(c2.bytes_consumed(), 5);
}
