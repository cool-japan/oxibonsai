//! Integration tests for the Regex → BNF Grammar compiler.
//!
//! Tests cover: all supported regex features, error cases, integration with
//! `GrammarConstraint`, and end-to-end acceptance/rejection semantics.

use std::sync::Arc;

use oxibonsai_runtime::constrained_decoding::TokenConstraint;
use oxibonsai_runtime::grammar::{EarleyRecognizer, GrammarConstraint};
use oxibonsai_runtime::{compile_regex, RegexCompileError};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compile pattern, normalise terminals, build an EarleyRecognizer.
fn make_recognizer(pattern: &str) -> EarleyRecognizer {
    let mut g = compile_regex(pattern).expect("pattern must compile");
    g.normalise_terminals();
    EarleyRecognizer::new(Arc::new(g))
}

/// Feed all bytes of `input` into the recognizer and return true iff all bytes
/// were accepted AND the recognizer is in an accepting state at the end.
fn recognizer_accepts(rec: &mut EarleyRecognizer, input: &str) -> bool {
    for b in input.bytes() {
        if !rec.feed_byte(b) {
            return false;
        }
    }
    rec.is_accepting()
}

/// Build a `GrammarConstraint` with byte-level ASCII vocabulary (token id == byte value,
/// for ids 0..256 where id < 256 maps to a single byte).
fn make_constraint(pattern: &str) -> GrammarConstraint {
    let grammar = compile_regex(pattern).expect("pattern must compile");
    GrammarConstraint::new(
        grammar,
        |id| {
            if id < 256 {
                vec![id as u8]
            } else {
                vec![]
            }
        },
        256,
    )
}

/// Feed `input` into a `GrammarConstraint` advancing byte-by-byte.
/// Returns true iff all bytes are accepted AND `is_complete()` at end.
fn constraint_accepts(c: &mut GrammarConstraint, input: &str) -> bool {
    for b in input.bytes() {
        if !c.advance(b as u32) {
            return false;
        }
    }
    c.is_complete()
}

/// Convenience function: compile pattern, create recognizer, check acceptance.
fn accepts(pattern: &str, input: &str) -> bool {
    let mut rec = make_recognizer(pattern);
    recognizer_accepts(&mut rec, input)
}

/// Convenience function: verify `accepts` is true.
fn assert_accepts(pattern: &str, input: &str) {
    assert!(
        accepts(pattern, input),
        "pattern /{pattern}/ should match {input:?} but did not"
    );
}

/// Convenience function: verify `accepts` is false.
fn assert_rejects(pattern: &str, input: &str) {
    assert!(
        !accepts(pattern, input),
        "pattern /{pattern}/ should NOT match {input:?} but it did"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: empty pattern returns EmptyPattern error
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_empty_pattern_error() {
    let result = compile_regex("");
    assert!(
        matches!(result, Err(RegexCompileError::EmptyPattern)),
        "empty pattern must return EmptyPattern error, got: {:?}",
        result.err()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: literal string "abc"
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_literal_abc() {
    assert_accepts("abc", "abc");
    assert_rejects("abc", "abcd");
    assert_rejects("abc", "ab");
    assert_rejects("abc", "xyz");
    assert_rejects("abc", "ABC");
    assert_rejects("abc", "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: alternation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_alternation() {
    assert_accepts("cat|dog", "cat");
    assert_accepts("cat|dog", "dog");
    assert_rejects("cat|dog", "car");
    assert_rejects("cat|dog", "bat");
    assert_rejects("cat|dog", "catdog");
    assert_rejects("cat|dog", "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Kleene star
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_kleene_star() {
    assert_accepts("a*", "");
    assert_accepts("a*", "a");
    assert_accepts("a*", "aaa");
    assert_accepts("a*", "aaaaaaaaaa");
    assert_rejects("a*", "b");
    assert_rejects("a*", "ab");
    assert_rejects("a*", "ba");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: plus quantifier
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_plus() {
    assert_accepts("a+", "a");
    assert_accepts("a+", "aaa");
    assert_rejects("a+", "");
    assert_rejects("a+", "b");
    assert_rejects("a+", "ba");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: optional quantifier
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_optional() {
    assert_accepts("colou?r", "color");
    assert_accepts("colou?r", "colour");
    assert_rejects("colou?r", "colouur");
    assert_rejects("colou?r", "colr");
    assert_rejects("colou?r", "col");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: dot (any byte except newline)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_dot_any() {
    assert_accepts("a.b", "axb");
    assert_accepts("a.b", "a1b");
    assert_accepts("a.b", "a b");
    assert_rejects("a.b", "ab");   // dot requires exactly one char
    assert_rejects("a.b", "axxb"); // too many chars
    // dot does NOT match newline
    assert_rejects("a.b", "a\nb");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: character class
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_char_class() {
    assert_accepts("[aeiou]+", "aei");
    assert_accepts("[aeiou]+", "a");
    assert_accepts("[aeiou]+", "uuuooa");
    assert_rejects("[aeiou]+", "xyz");
    assert_rejects("[aeiou]+", "");
    assert_rejects("[aeiou]+", "ab"); // 'b' is not a vowel
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: character class with range
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_char_class_range() {
    assert_accepts("[a-z]+", "hello");
    assert_accepts("[a-z]+", "z");
    assert_rejects("[a-z]+", "HELLO");
    assert_rejects("[a-z]+", "Hello");
    assert_rejects("[a-z]+", "");
    assert_rejects("[a-z]+", "123");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10: negated character class
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_negated_class() {
    assert_accepts("[^abc]+", "xyz");
    assert_accepts("[^abc]+", "123");
    assert_rejects("[^abc]+", "a");
    assert_rejects("[^abc]+", "b");
    assert_rejects("[^abc]+", "c");
    assert_rejects("[^abc]+", "xya");  // 'a' breaks it
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11: \d digit class
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_digit_class() {
    assert_accepts(r"\d+", "123");
    assert_accepts(r"\d+", "0");
    assert_rejects(r"\d+", "abc");
    assert_rejects(r"\d+", "");
    assert_rejects(r"\d+", "12a");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12: \w word class
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_word_class() {
    assert_accepts(r"\w+", "hello_world");
    assert_accepts(r"\w+", "abc123");
    assert_accepts(r"\w+", "_private");
    assert_rejects(r"\w+", "!@#");
    assert_rejects(r"\w+", "");
    assert_rejects(r"\w+", "hello world"); // space is not \w
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 13: \s whitespace class
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_whitespace_class() {
    assert_accepts(r"\s+", " \t");
    assert_accepts(r"\s+", " ");
    assert_accepts(r"\s+", "\t\n\r");
    assert_rejects(r"\s+", "a");
    assert_rejects(r"\s+", "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 14: counted exact quantifier {n}
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_counted_exact() {
    assert_accepts("a{3}", "aaa");
    assert_rejects("a{3}", "aa");
    assert_rejects("a{3}", "aaaa");
    assert_rejects("a{3}", "");
    assert_rejects("a{3}", "a");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 15: counted range quantifier {n,m}
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_counted_range() {
    assert_accepts("a{2,4}", "aa");
    assert_accepts("a{2,4}", "aaa");
    assert_accepts("a{2,4}", "aaaa");
    assert_rejects("a{2,4}", "a");
    assert_rejects("a{2,4}", "aaaaa");
    assert_rejects("a{2,4}", "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 16: grouping
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_grouping() {
    assert_accepts("(ab)+", "ab");
    assert_accepts("(ab)+", "abab");
    assert_accepts("(ab)+", "ababab");
    assert_rejects("(ab)+", "aba");
    assert_rejects("(ab)+", "a");
    assert_rejects("(ab)+", "");
    assert_rejects("(ab)+", "b");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 17: complex email-like pattern
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_complex_email_like() {
    let pattern = r"[a-z]+@[a-z]+\.[a-z]+";
    assert_accepts(pattern, "user@example.com");
    assert_accepts(pattern, "a@b.c");
    assert_rejects(pattern, "user@example");
    assert_rejects(pattern, "@example.com");
    assert_rejects(pattern, "user@.com");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 18: date format \d{4}-\d{2}-\d{2}
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_digit_word_sequence() {
    let pattern = r"\d{4}-\d{2}-\d{2}";
    assert_accepts(pattern, "2024-01-15");
    assert_accepts(pattern, "1999-12-31");
    assert_rejects(pattern, "24-01-15");
    assert_rejects(pattern, "2024-1-15");
    assert_rejects(pattern, "2024-01-5");
    assert_rejects(pattern, "2024/01/15");
    assert_rejects(pattern, "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 19: escape sequences — literal dot
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_escape_sequences() {
    assert_accepts(r"a\.b", "a.b");
    assert_rejects(r"a\.b", "axb");
    assert_rejects(r"a\.b", "ab");

    // Test other escapes
    assert_accepts(r"a\+b", "a+b");
    assert_accepts(r"\(x\)", "(x)");
    assert_accepts(r"a\\b", "a\\b");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 20: backreference \1 is unsupported
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_backreference_unsupported() {
    let result = compile_regex(r"(a)\1");
    assert!(
        matches!(result, Err(RegexCompileError::UnsupportedFeature(_))),
        "backreferences must return UnsupportedFeature, got: {:?}",
        result
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 21: lookahead is unsupported
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lookahead_unsupported() {
    let result = compile_regex("foo(?=bar)");
    assert!(
        matches!(result, Err(RegexCompileError::UnsupportedFeature(_))),
        "lookahead must return UnsupportedFeature, got: {:?}",
        result
    );

    let result2 = compile_regex("foo(?!bar)");
    assert!(
        matches!(result2, Err(RegexCompileError::UnsupportedFeature(_))),
        "negative lookahead must return UnsupportedFeature, got: {:?}",
        result2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 22: named groups are unsupported
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_named_group_unsupported() {
    let result = compile_regex("(?P<name>abc)");
    assert!(
        matches!(result, Err(RegexCompileError::UnsupportedFeature(_))),
        "Python-style named group must return UnsupportedFeature, got: {:?}",
        result
    );

    let result2 = compile_regex("(?<name>abc)");
    assert!(
        matches!(result2, Err(RegexCompileError::UnsupportedFeature(_))),
        "named group (?<name>...) must return UnsupportedFeature, got: {:?}",
        result2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 23: grammar has correct start symbol
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_dfa_grammar_start_symbol() {
    let grammar = compile_regex("abc").expect("should compile");
    // The start symbol must be a valid NT id (< nt_count).
    assert!(
        grammar.start < grammar.nt_count,
        "start symbol id {} must be < nt_count {}",
        grammar.start,
        grammar.nt_count
    );
    // The start NT name should contain "regex_s0".
    let name = grammar.nt_name(grammar.start);
    assert!(
        name.contains("regex_s"),
        "start NT name should contain 'regex_s', got: {name:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 24: integration with GrammarConstraint
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_grammar_integrates_with_constraint() {
    // This must not panic.
    let mut c = make_constraint(r"\d+");

    // Feed "42" byte-by-byte; should be accepted.
    assert!(constraint_accepts(&mut c, "42"));
    assert!(c.is_complete(), "should be accepting after '42'");

    // Reset and try a rejection.
    c.reset();
    // Feed 'a' — not a digit, should be rejected.
    let ok = c.advance(b'a' as u32);
    assert!(!ok, "'a' should not be accepted by \\d+");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 25: anchors ^ and $ are silently ignored
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_anchors_ignored() {
    // With anchors stripped, "^abc$" behaves like "abc".
    assert_accepts("^abc$", "abc");
    assert_rejects("^abc$", "xabc");
    assert_rejects("^abc$", "abcx");

    // Just an anchor alone compiles and matches the empty string.
    let g = compile_regex("^").expect("^ should compile");
    assert!(!g.rules.is_empty(), "anchors-only pattern should produce a grammar");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 26: nested groups
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_nested_groups() {
    // (a(b|c))+ matches one or more of: 'ab' or 'ac'
    let pattern = "(a(b|c))+";
    assert_accepts(pattern, "ab");
    assert_accepts(pattern, "ac");
    assert_accepts(pattern, "abac");
    assert_accepts(pattern, "acabac");  // ac + ab + ac
    assert_rejects(pattern, "a");
    assert_rejects(pattern, "b");
    assert_rejects(pattern, "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 27: large alternation does not hit limit
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_large_alternation() {
    // 10 alternation branches — should compile fine without hitting DFA state limit.
    let pattern = "a|b|c|d|e|f|g|h|i|j";
    let result = compile_regex(pattern);
    assert!(
        result.is_ok(),
        "10-way alternation must compile without hitting limit, got: {:?}",
        result.err()
    );
    let grammar = result.unwrap();
    assert!(!grammar.rules.is_empty());

    // Spot-check acceptance.
    assert_accepts(pattern, "a");
    assert_accepts(pattern, "j");
    assert_rejects(pattern, "k");
    assert_rejects(pattern, "ab");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 28: non-capturing group (?:...)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_non_capturing_group() {
    assert_accepts("(?:ab)+", "ab");
    assert_accepts("(?:ab)+", "abab");
    assert_rejects("(?:ab)+", "a");
    assert_rejects("(?:ab)+", "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 29: counted quantifier open-ended {n,}
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_counted_at_least() {
    // `a{2,}` means "two or more 'a'"
    assert_accepts("a{2,}", "aa");
    assert_accepts("a{2,}", "aaa");
    assert_accepts("a{2,}", "aaaaaaa");
    assert_rejects("a{2,}", "a");
    assert_rejects("a{2,}", "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 30: lookbehind is unsupported
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lookbehind_unsupported() {
    let result = compile_regex("(?<=foo)bar");
    assert!(
        matches!(result, Err(RegexCompileError::UnsupportedFeature(_))),
        "lookbehind must return UnsupportedFeature, got: {:?}",
        result
    );

    let result2 = compile_regex("(?<!foo)bar");
    assert!(
        matches!(result2, Err(RegexCompileError::UnsupportedFeature(_))),
        "negative lookbehind must return UnsupportedFeature, got: {:?}",
        result2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 31: compile gives grammar with rules
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_grammar_has_rules() {
    let grammar = compile_regex(r"[a-z]+").expect("should compile");
    assert!(!grammar.rules.is_empty(), "compiled grammar must have rules");
    assert!(grammar.nt_count > 0, "must allocate at least one NT");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 32: constraint allowed_tokens respects grammar
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_constraint_allowed_tokens_respects_grammar() {
    let mut c = make_constraint(r"[0-9]+");

    // At start, only digits should be allowed.
    let mask = c.allowed_tokens(&[], 256).expect("must return a mask");
    // Digit bytes (48–57) should be allowed.
    for d in b'0'..=b'9' {
        assert!(
            mask[d as usize],
            "digit '{}' should be allowed at start of \\d+",
            d as char
        );
    }
    // Letters should NOT be allowed at start.
    for d in b'a'..=b'z' {
        assert!(
            !mask[d as usize],
            "letter '{}' should NOT be allowed at start of [0-9]+",
            d as char
        );
    }

    // After feeding '5', still accepting (more digits possible), and digits still allowed.
    assert!(c.advance(b'5' as u32));
    assert!(c.is_complete());
    let mask2 = c.allowed_tokens(&[], 256).expect("must return a mask");
    for d in b'0'..=b'9' {
        assert!(
            mask2[d as usize],
            "digit '{}' should still be allowed after '5'",
            d as char
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 33: negated digit class \D
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_non_digit_class() {
    assert_accepts(r"\D+", "abc");
    assert_accepts(r"\D+", "!@#");
    assert_rejects(r"\D+", "123");
    assert_rejects(r"\D+", "abc5"); // '5' breaks it
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 34: single character literal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_single_char_literal() {
    assert_accepts("x", "x");
    assert_rejects("x", "y");
    assert_rejects("x", "xx");
    assert_rejects("x", "");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 35: mixed quantifiers in complex pattern
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mixed_quantifiers() {
    // `[A-Z][a-z]*` — a capital letter followed by zero or more lower-case
    let pattern = "[A-Z][a-z]*";
    assert_accepts(pattern, "Hello");
    assert_accepts(pattern, "A");
    assert_accepts(pattern, "World");
    assert_rejects(pattern, "hello");  // must start with capital
    assert_rejects(pattern, "");
    assert_rejects(pattern, "123");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 36: empty alternation branch (a|) — righthand branch is epsilon
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_alternation_with_empty_branch() {
    // `a|` has two branches: 'a' and empty → accepts "" or "a"
    let pattern = "a|";
    let result = compile_regex(pattern);
    // This should compile (an empty right branch is valid).
    assert!(result.is_ok(), "a| should compile, got: {:?}", result.err());
    let mut rec = make_recognizer(pattern);
    // The empty branch should be accepting at start.
    assert!(rec.is_accepting(), "a| should accept empty string");
    // But feeding 'a' should also eventually accept.
    assert!(rec.feed_byte(b'a'));
    assert!(rec.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 37: RegexCompileError implements Display and Error
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    let e = RegexCompileError::EmptyPattern;
    let s = e.to_string();
    assert!(!s.is_empty(), "Display must produce a non-empty string");

    let e2 = RegexCompileError::InvalidSyntax("oops".to_string());
    let s2 = e2.to_string();
    assert!(s2.contains("oops"), "Display should include the message");

    let e3 = RegexCompileError::DepthExceeded { limit: 2048 };
    let s3 = e3.to_string();
    assert!(s3.contains("2048"), "Display should include the limit");

    let e4 = RegexCompileError::UnsupportedFeature("lookahead".to_string());
    let s4 = e4.to_string();
    assert!(s4.contains("lookahead"), "Display should include the feature name");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 38: dot in alternation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_dot_in_alternation() {
    // `\d|\.` — a digit or a literal dot
    let pattern = r"\d|\.";
    assert_accepts(pattern, "5");
    assert_accepts(pattern, ".");
    assert_rejects(pattern, "a");
    assert_rejects(pattern, "");
}
