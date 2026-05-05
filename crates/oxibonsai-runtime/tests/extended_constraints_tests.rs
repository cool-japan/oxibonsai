//! Extended token constraint integration tests.
//!
//! Covers [`AllowListConstraint`], [`SequenceConstraint`], and
//! [`LengthConstraint`] including trait-object and Send+Sync usage.

use oxibonsai_runtime::{
    AllowListConstraint, LengthConstraint, SequenceConstraint, TokenConstraint,
};

// ─────────────────────────────────────────────────────────────────────────────
// AllowListConstraint
// ─────────────────────────────────────────────────────────────────────────────

/// Helper: build a bool mask for a small vocab from a list of allowed token ids.
fn mask_allowed(tokens: &[u32], vocab_size: usize) -> Vec<bool> {
    let mut m = vec![false; vocab_size];
    for &t in tokens {
        m[t as usize] = true;
    }
    m
}

#[test]
fn allow_list_single_candidate_matches() {
    let mut c = AllowListConstraint::new(vec![vec![42u32]]);
    let mask = c.allowed_tokens(&[], 50).unwrap();
    assert_eq!(mask, mask_allowed(&[42], 50));
    assert!(!c.is_complete());
    let ok = c.advance(42);
    assert!(ok, "advance should return true on correct token");
    assert!(
        c.is_complete(),
        "should be complete after consuming the sole token"
    );
}

#[test]
fn allow_list_multi_candidate_correct_sequence() {
    // Candidates: [1, 2] and [1, 3]
    let mut c = AllowListConstraint::new(vec![vec![1u32, 2], vec![1, 3]]);
    // Position 0: both share token 1
    let mask0 = c.allowed_tokens(&[], 10).unwrap();
    assert!(mask0[1]);
    assert!(!mask0[2]);
    assert!(!mask0[3]);

    assert!(c.advance(1));

    // Position 1: candidate 0 wants 2, candidate 1 wants 3
    let mask1 = c.allowed_tokens(&[1], 10).unwrap();
    assert!(!mask1[1]);
    assert!(mask1[2]);
    assert!(mask1[3]);

    assert!(c.advance(2));
    assert!(c.is_complete(), "candidate [1,2] fully matched");
}

#[test]
fn allow_list_wrong_token_deactivates_candidate() {
    // Candidates: [10, 20] and [10, 30]
    let mut c = AllowListConstraint::new(vec![vec![10u32, 20], vec![10, 30]]);
    assert!(c.advance(10)); // both still alive

    // Only candidate with 30 at pos 1 remains after advancing with 30
    assert!(c.advance(30));
    assert!(c.is_complete());
    // Candidate [10, 20] should be deactivated since we advanced with 30
    assert_eq!(c.active_count(), 1, "only the 30-branch candidate remains");
}

#[test]
fn allow_list_all_deactivated_no_tokens() {
    // Single candidate [5, 6]
    let mut c = AllowListConstraint::new(vec![vec![5u32, 6]]);
    // Advance with wrong first token → deactivates the only candidate
    let ok = c.advance(99);
    // false because all candidates are deactivated and none just completed
    assert!(!ok, "no candidate survived — should return false");
    let mask = c.allowed_tokens(&[99], 100).unwrap();
    // All false since no active candidate has a valid next token
    assert!(mask.iter().all(|&b| !b), "all mask entries should be false");
}

#[test]
fn allow_list_empty_candidates() {
    let mut c = AllowListConstraint::new(vec![]);
    // No candidates → mask is all false
    let mask = c.allowed_tokens(&[], 10).unwrap();
    assert!(mask.iter().all(|&b| !b));
    assert!(!c.is_complete());
    let ok = c.advance(0);
    assert!(!ok);
}

#[test]
fn allow_list_reset_restores() {
    let mut c = AllowListConstraint::new(vec![vec![7u32, 8]]);
    c.advance(7);
    c.advance(8);
    assert!(c.is_complete());

    c.reset();
    assert!(!c.is_complete());
    assert_eq!(c.active_count(), 1);
    // After reset position is 0 again
    let mask = c.allowed_tokens(&[], 20).unwrap();
    assert!(mask[7]);
    assert!(!mask[8]);
}

#[test]
fn allow_list_is_complete_on_match() {
    // Two candidates of different length: [1] and [1, 2]
    let mut c = AllowListConstraint::new(vec![vec![1u32], vec![1, 2]]);
    // After advancing token 1: candidate [1] is complete (len=1, position=1)
    c.advance(1);
    assert!(c.is_complete(), "shortest candidate [1] fully matched");
}

#[test]
fn allow_list_prefix_overlap() {
    // Three candidates sharing the prefix [1, 2]:
    //   [1, 2, 3], [1, 2, 4], [1, 5]
    let mut c = AllowListConstraint::new(vec![vec![1u32, 2, 3], vec![1, 2, 4], vec![1, 5]]);
    // pos 0 → only 1 allowed
    let m0 = c.allowed_tokens(&[], 10).unwrap();
    assert!(m0[1]);
    assert!(!m0[2]);
    assert!(!m0[5]);

    c.advance(1); // all three still alive

    // pos 1 → 2 (from first two) and 5 (from third)
    let m1 = c.allowed_tokens(&[1], 10).unwrap();
    assert!(m1[2]);
    assert!(m1[5]);
    assert!(!m1[3]);

    c.advance(2); // third candidate [1,5] deactivated

    // pos 2 → 3 and 4
    let m2 = c.allowed_tokens(&[1, 2], 10).unwrap();
    assert!(m2[3]);
    assert!(m2[4]);
    assert!(!m2[5]);

    c.advance(4);
    assert!(c.is_complete()); // [1,2,4] fully matched
}

#[test]
fn allow_list_allowed_tokens_mask_correct() {
    // Candidates: [100, 200, 300] — position 1 should allow only 200
    let mut c = AllowListConstraint::new(vec![vec![100u32, 200, 300]]);
    c.advance(100);
    let mask = c.allowed_tokens(&[100], 400).unwrap();
    assert!(mask[200]);
    // Verify every other entry is false
    for (i, &b) in mask.iter().enumerate() {
        if i != 200 {
            assert!(!b, "token {i} should not be allowed at position 1");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SequenceConstraint
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn seq_constraint_forces_exact_sequence() {
    let mut c = SequenceConstraint::new(vec![10u32, 20, 30]);
    for (pos, &expected) in [10u32, 20, 30].iter().enumerate() {
        let mask = c.allowed_tokens(&[], 50).unwrap();
        // Only expected token allowed
        assert!(
            mask[expected as usize],
            "pos {pos}: expected token {expected}"
        );
        for (i, &b) in mask.iter().enumerate() {
            if i != expected as usize {
                assert!(!b, "pos {pos}: token {i} should be blocked");
            }
        }
        let ok = c.advance(expected);
        assert!(ok, "advance with correct token should succeed at pos {pos}");
    }
    assert!(c.is_complete());
    assert!(!c.is_failed());
}

#[test]
fn seq_constraint_violation_returns_false() {
    let mut c = SequenceConstraint::new(vec![5u32, 6, 7]);
    assert!(c.advance(5));
    // Feed wrong token
    let ok = c.advance(99);
    assert!(!ok, "wrong token must return false");
    assert!(c.is_failed());
}

#[test]
fn seq_constraint_empty_target_immediately_complete() {
    let c = SequenceConstraint::new(vec![]);
    assert!(c.is_complete(), "empty target is trivially complete");
    // allowed_tokens should return None (no restriction)
    assert!(c.allowed_tokens(&[], 10).is_none());
}

#[test]
fn seq_constraint_allows_all_after_sequence() {
    let mut c = SequenceConstraint::new(vec![3u32, 4]);
    c.advance(3);
    c.advance(4);
    assert!(c.is_complete());
    // After completion no restriction
    assert!(
        c.allowed_tokens(&[3, 4], 10).is_none(),
        "constraint is satisfied — should allow all tokens"
    );
}

#[test]
fn seq_constraint_reset() {
    let mut c = SequenceConstraint::new(vec![1u32, 2, 3]);
    c.advance(1);
    c.advance(99); // violation
    assert!(c.is_failed());

    c.reset();
    assert!(!c.is_failed());
    assert!(!c.is_complete());
    // Should enforce token 1 again
    let mask = c.allowed_tokens(&[], 10).unwrap();
    assert!(mask[1]);
    assert!(!mask[2]);
}

#[test]
fn seq_constraint_single_token() {
    let mut c = SequenceConstraint::new(vec![42u32]);
    assert!(!c.is_complete());
    let mask = c.allowed_tokens(&[], 100).unwrap();
    assert!(mask[42]);
    assert!(c.advance(42));
    assert!(c.is_complete());
}

#[test]
fn seq_constraint_is_complete_after_full_sequence() {
    let tokens: Vec<u32> = (0..8).collect();
    let mut c = SequenceConstraint::new(tokens.clone());
    for &t in &tokens {
        assert!(
            !c.is_complete(),
            "not yet complete before consuming all tokens"
        );
        c.advance(t);
    }
    assert!(c.is_complete(), "all tokens consumed → complete");
}

#[test]
fn seq_constraint_mask_token_out_of_vocab_ignored() {
    // If the target token exceeds vocab_size, the mask should have no true entries
    // (the token simply cannot appear in this vocab).
    let c = SequenceConstraint::new(vec![999u32]);
    let mask = c.allowed_tokens(&[], 10).unwrap();
    // 999 >= 10 → nothing in the mask is true
    assert!(mask.iter().all(|&b| !b));
}

// ─────────────────────────────────────────────────────────────────────────────
// LengthConstraint
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn length_constraint_allows_all_between_min_max() {
    // min=2, max=10, stop=1
    let mut c = LengthConstraint::new(2, 10, Some(1u32));
    // Advance past min_len
    c.advance(5);
    c.advance(5);
    assert_eq!(c.count(), 2);
    // Now between min and max → None (unconstrained)
    assert!(
        c.allowed_tokens(&[], 10).is_none(),
        "between min and max should return None"
    );
}

#[test]
fn length_constraint_prevents_stop_before_min() {
    let c = LengthConstraint::new(3, 10, Some(0u32));
    // count=0 < min=3 → stop token (0) should be blocked
    let mask = c.allowed_tokens(&[], 8).unwrap();
    assert!(!mask[0], "stop token must be blocked before min_len");
    for i in 1..8usize {
        assert!(mask[i], "non-stop token {i} should be allowed");
    }
}

#[test]
fn length_constraint_forces_stop_at_max() {
    let mut c = LengthConstraint::new(0, 3, Some(99u32));
    c.advance(1);
    c.advance(1);
    c.advance(1); // count == max_len == 3
                  // At max_len → only stop token allowed
    let mask = c.allowed_tokens(&[], 200).unwrap();
    assert!(
        mask[99],
        "stop token must be the only allowed token at max_len"
    );
    let non_stop_count = mask
        .iter()
        .enumerate()
        .filter(|&(i, &b)| i != 99 && b)
        .count();
    assert_eq!(
        non_stop_count, 0,
        "no other token should be allowed at max_len"
    );
}

#[test]
fn length_constraint_no_stop_token_no_limit_before_max() {
    // With no stop token and count below max, no restriction.
    let mut c = LengthConstraint::new(0, 100, None);
    c.advance(7);
    assert!(c.allowed_tokens(&[], 50).is_none());
}

#[test]
fn length_constraint_min_zero() {
    // min=0 means the stop token is allowed immediately
    let c = LengthConstraint::new(0, 10, Some(2u32));
    // count=0 >= min=0 and count=0 < max=10 → None (unconstrained)
    assert!(c.allowed_tokens(&[], 10).is_none());
}

#[test]
fn length_constraint_exact_min_eq_max() {
    // min==max==2: must generate exactly 2 tokens then stop immediately
    let mut c = LengthConstraint::new(2, 2, Some(0u32));
    // count=0 < min=2 → stop blocked
    let m0 = c.allowed_tokens(&[], 5).unwrap();
    assert!(!m0[0]);
    c.advance(3);
    // count=1 < min=2 → stop still blocked
    let m1 = c.allowed_tokens(&[3], 5).unwrap();
    assert!(!m1[0]);
    c.advance(3);
    // count=2 == max=2 → only stop allowed
    let m2 = c.allowed_tokens(&[3, 3], 5).unwrap();
    assert!(m2[0]);
    for i in 1..5usize {
        assert!(!m2[i]);
    }
}

#[test]
fn length_constraint_is_complete_with_stop_token() {
    let mut c = LengthConstraint::new(2, 20, Some(7u32));
    c.advance(1);
    c.advance(2);
    assert!(!c.is_complete(), "count == min_len but stop not yet seen");
    // Now see stop token
    c.advance(7);
    assert!(c.is_complete(), "stop token seen after min_len → complete");
}

#[test]
fn length_constraint_reset() {
    let mut c = LengthConstraint::new(1, 5, Some(9u32));
    c.advance(1);
    c.advance(9); // stop token
    assert!(c.is_complete());

    c.reset();
    assert_eq!(c.count(), 0);
    assert!(!c.is_complete());
    // After reset, stop token should be blocked again (count=0 < min=1)
    let mask = c.allowed_tokens(&[], 20).unwrap();
    assert!(
        !mask[9],
        "stop token blocked after reset because count < min_len"
    );
}

#[test]
fn length_constraint_no_stop_all_false_at_max() {
    // With no stop token, at max_len all tokens are blocked.
    let mut c = LengthConstraint::new(0, 2, None);
    c.advance(0);
    c.advance(0); // count == max
    let mask = c.allowed_tokens(&[], 10).unwrap();
    assert!(
        mask.iter().all(|&b| !b),
        "no stop token and count >= max → all blocked"
    );
    assert!(c.is_complete(), "max reached and min==0 → complete");
}

// ─────────────────────────────────────────────────────────────────────────────
// Box<dyn TokenConstraint> — trait object usage
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn constraint_boxed_trait_object_allow_list() {
    let mut c: Box<dyn TokenConstraint> =
        Box::new(AllowListConstraint::new(vec![vec![1u32, 2], vec![1, 3]]));
    assert_eq!(c.name(), "AllowListConstraint");
    assert!(c.allowed_tokens(&[], 10).is_some());
    assert!(c.advance(1));
    assert!(c.advance(2));
    assert!(c.is_complete());
    c.reset();
    assert!(!c.is_complete());
}

#[test]
fn constraint_boxed_trait_object_sequence() {
    let mut c: Box<dyn TokenConstraint> = Box::new(SequenceConstraint::new(vec![8u32, 9, 10]));
    assert_eq!(c.name(), "SequenceConstraint");
    assert!(!c.is_complete());
    c.advance(8);
    c.advance(9);
    c.advance(10);
    assert!(c.is_complete());
    c.reset();
    assert!(!c.is_complete());
}

#[test]
fn constraint_boxed_trait_object_length() {
    let mut c: Box<dyn TokenConstraint> = Box::new(LengthConstraint::new(1, 3, Some(0u32)));
    assert_eq!(c.name(), "LengthConstraint");
    assert!(!c.is_complete());
    c.advance(5); // count=1, min satisfied but stop not seen
    assert!(!c.is_complete());
    c.advance(0); // stop token — count=2 >= min=1
    assert!(c.is_complete());
    c.reset();
    assert!(!c.is_complete());
}

// ─────────────────────────────────────────────────────────────────────────────
// Send + Sync verification (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

/// Asserts that T: Send + Sync by constructing a Box<dyn TokenConstraint + Send + Sync>.
fn assert_send_sync<T: TokenConstraint + Send + Sync + 'static>(c: T) -> Box<dyn TokenConstraint> {
    Box::new(c)
}

#[test]
fn constraint_send_sync_allow_list() {
    let c = AllowListConstraint::new(vec![vec![1u32, 2]]);
    let boxed = assert_send_sync(c);
    assert_eq!(boxed.name(), "AllowListConstraint");
}

#[test]
fn constraint_send_sync_sequence() {
    let c = SequenceConstraint::new(vec![3u32, 4, 5]);
    let boxed = assert_send_sync(c);
    assert_eq!(boxed.name(), "SequenceConstraint");
}

#[test]
fn constraint_send_sync_length() {
    let c = LengthConstraint::new(2, 10, Some(1u32));
    let boxed = assert_send_sync(c);
    assert_eq!(boxed.name(), "LengthConstraint");
}
