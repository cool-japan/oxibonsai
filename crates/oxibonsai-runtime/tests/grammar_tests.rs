//! Integration tests for the BNF Grammar Engine and GrammarConstraint.
//!
//! Covers:
//! * BNF parser (correct parsing, error cases)
//! * Earley recognizer (acceptance, rejection, next_byte_set, reset, clone)
//! * GrammarConstraint (allowed_tokens mask, advance, is_complete, reset, trait compliance)
//! * Pre-built example grammars

use std::sync::Arc;

use oxibonsai_runtime::grammar::{
    arithmetic_grammar, csv_row_grammar, simple_ab_grammar,
    parse_bnf, BnfParseError, Grammar, GrammarConstraint, EarleyRecognizer, Rule, Symbol,
};
use oxibonsai_runtime::constrained_decoding::TokenConstraint;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a normalised EarleyRecognizer from a BNF string.
fn recognizer_from_bnf(bnf: &str) -> EarleyRecognizer {
    let mut g = parse_bnf(bnf).expect("valid BNF");
    g.normalise_terminals();
    EarleyRecognizer::new(Arc::new(g))
}

/// Feed a str into the recognizer; returns false if any byte is rejected.
fn feed_str(rec: &mut EarleyRecognizer, s: &str) -> bool {
    for b in s.bytes() {
        if !rec.feed_byte(b) {
            return false;
        }
    }
    true
}

/// Build a GrammarConstraint with a simple byte-level vocab (token id == byte value, 0..128).
fn ascii_constraint(grammar: Grammar) -> GrammarConstraint {
    GrammarConstraint::new(
        grammar,
        |id| if id < 128 { vec![id as u8] } else { vec![] },
        128,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// BNF Parser tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bnf_parse_simple_rule() {
    let g = parse_bnf(r#"<S> ::= "hello""#).expect("valid");
    assert_eq!(g.rules.len(), 1);
    assert_eq!(g.rules[0].rhs[0], Symbol::Terminal(b"hello".to_vec()));
}

#[test]
fn bnf_parse_alternation() {
    let g = parse_bnf(r#"<S> ::= "a" | "b" | "c""#).expect("valid");
    assert_eq!(g.rules.len(), 3);
    // All rules share the same lhs.
    let lhs = g.rules[0].lhs;
    assert!(g.rules.iter().all(|r| r.lhs == lhs));
}

#[test]
fn bnf_parse_recursion() {
    let g = parse_bnf(r#"<S> ::= "a" <S> | "a""#).expect("valid");
    // Two rules: one recursive, one base case.
    assert_eq!(g.rules.len(), 2);
    // First rule has a NonTerminal in its rhs.
    assert!(g.rules[0].rhs.iter().any(|s| matches!(s, Symbol::NonTerminal(_))));
}

#[test]
fn bnf_parse_multi_line() {
    let g = parse_bnf(r#"
        <expr> ::= <term> "+" <expr>
        <expr> ::= <term>
        <term> ::= "x"
    "#).expect("valid multi-line BNF");
    // 3 separate rules
    assert_eq!(g.rules.len(), 3);
}

#[test]
fn bnf_parse_comments() {
    let g = parse_bnf(r#"
        # This is a comment
        <S> ::= "a" # another comment
    "#).expect("valid with comments");
    assert_eq!(g.rules.len(), 1);
    assert_eq!(g.rules[0].rhs[0], Symbol::Terminal(b"a".to_vec()));
}

#[test]
fn bnf_parse_empty_input_error() {
    assert!(matches!(parse_bnf(""), Err(BnfParseError::EmptyInput)));
    assert!(matches!(parse_bnf("  # only comment\n"), Err(BnfParseError::EmptyInput)));
}

#[test]
fn bnf_parse_undefined_nonterminal_error() {
    let err = parse_bnf(r#"<S> ::= <T>"#);
    assert!(
        matches!(err, Err(BnfParseError::UndefinedNonTerminal { ref name }) if name == "T"),
        "got: {err:?}"
    );
}

#[test]
fn bnf_parse_missing_separator_error() {
    let err = parse_bnf(r#"<S> = "a""#);
    assert!(
        matches!(err, Err(BnfParseError::MissingDefinitionSeparator { .. })),
        "got: {err:?}"
    );
}

#[test]
fn bnf_parse_unterminated_string_error() {
    let err = parse_bnf(r#"<S> ::= "unterminated"#);
    assert!(
        matches!(err, Err(BnfParseError::UnterminatedString { .. })),
        "got: {err:?}"
    );
}

#[test]
fn bnf_parse_string_escapes() {
    let g = parse_bnf(r#"<S> ::= "\n\r\t\"\\""#).expect("valid");
    assert_eq!(g.rules[0].rhs[0], Symbol::Terminal(b"\n\r\t\"\\".to_vec()));
}

#[test]
fn bnf_parse_start_symbol_is_first_lhs() {
    let g = parse_bnf(r#"
        <first> ::= "a"
        <second> ::= "b"
    "#).expect("valid");
    assert_eq!(g.nt_name(g.start()), "first");
}

#[test]
fn bnf_parse_epsilon_alternative() {
    // An empty alternative `""` is parsed as an epsilon terminal.
    let g = parse_bnf(r#"<S> ::= "a" | """#).expect("valid");
    assert_eq!(g.rules.len(), 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Earley recognizer tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn earley_accepts_arithmetic_1plus2() {
    let mut r = recognizer_from_bnf(r#"
        <expr>   ::= <term> "+" <expr> | <term>
        <term>   ::= <factor> "*" <term> | <factor>
        <factor> ::= "(" <expr> ")" | <number>
        <number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#);
    assert!(feed_str(&mut r, "1+2"));
    assert!(r.is_accepting());
}

#[test]
fn earley_accepts_arithmetic_9() {
    let mut r = recognizer_from_bnf(r#"
        <expr>   ::= <term> "+" <expr> | <term>
        <term>   ::= <factor> "*" <term> | <factor>
        <factor> ::= "(" <expr> ")" | <number>
        <number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#);
    assert!(feed_str(&mut r, "9"));
    assert!(r.is_accepting());
}

#[test]
fn earley_accepts_arithmetic_1times2plus3() {
    let mut r = recognizer_from_bnf(r#"
        <expr>   ::= <term> "+" <expr> | <term>
        <term>   ::= <factor> "*" <term> | <factor>
        <factor> ::= "(" <expr> ")" | <number>
        <number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#);
    assert!(feed_str(&mut r, "1*2+3"));
    assert!(r.is_accepting());
}

#[test]
fn earley_rejects_arithmetic_plus_at_start() {
    let mut r = recognizer_from_bnf(r#"
        <expr>   ::= <term> "+" <expr> | <term>
        <term>   ::= <factor> "*" <term> | <factor>
        <factor> ::= "(" <expr> ")" | <number>
        <number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#);
    let ok = feed_str(&mut r, "+1");
    // Either the first byte fails or the recognizer ends up not accepting.
    assert!(!ok || !r.is_accepting());
}

#[test]
fn earley_rejects_arithmetic_double_plus() {
    let mut r = recognizer_from_bnf(r#"
        <expr>   ::= <term> "+" <expr> | <term>
        <term>   ::= <factor> "*" <term> | <factor>
        <factor> ::= "(" <expr> ")" | <number>
        <number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#);
    let ok = feed_str(&mut r, "1++2");
    assert!(!ok || !r.is_accepting());
}

#[test]
fn earley_next_byte_set_at_start() {
    let mut g = parse_bnf(r#"
        <expr>   ::= <term> "+" <expr> | <term>
        <term>   ::= <factor> "*" <term> | <factor>
        <factor> ::= "(" <expr> ")" | <number>
        <number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#).expect("valid");
    g.normalise_terminals();
    let r = EarleyRecognizer::new(Arc::new(g));
    let nbs = r.next_byte_set();
    for d in b'0'..=b'9' {
        assert!(nbs.contains(&d), "digit {d} should be in next_byte_set at start");
    }
    assert!(nbs.contains(&b'('), "'(' should be in next_byte_set at start");
    assert!(!nbs.contains(&b'+'), "'+' should NOT be in next_byte_set at start");
}

#[test]
fn earley_next_byte_set_after_number() {
    let mut g = parse_bnf(r#"
        <expr>   ::= <term> "+" <expr> | <term>
        <term>   ::= <factor> "*" <term> | <factor>
        <factor> ::= "(" <expr> ")" | <number>
        <number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#).expect("valid");
    g.normalise_terminals();
    let mut r = EarleyRecognizer::new(Arc::new(g));
    r.feed_byte(b'1');
    let nbs = r.next_byte_set();
    // After a number, operators should be valid next bytes.
    assert!(nbs.contains(&b'+'), "'+' should be in next_byte_set after a digit");
    assert!(nbs.contains(&b'*'), "'*' should be in next_byte_set after a digit");
}

#[test]
fn earley_is_accepting_after_complete_input() {
    let mut r = recognizer_from_bnf(r#"<S> ::= "abc""#);
    assert!(feed_str(&mut r, "abc"));
    assert!(r.is_accepting());
}

#[test]
fn earley_not_accepting_mid_input() {
    let mut r = recognizer_from_bnf(r#"<S> ::= "abc""#);
    r.feed_byte(b'a');
    r.feed_byte(b'b');
    assert!(!r.is_accepting(), "should not accept mid-input");
}

#[test]
fn earley_reset_restores_initial_state() {
    let mut r = recognizer_from_bnf(r#"
        <expr>   ::= <term> "+" <expr> | <term>
        <term>   ::= "0" | "1" | "2"
    "#);
    feed_str(&mut r, "1+2");
    assert!(r.is_accepting());
    r.reset();
    assert_eq!(r.input_pos, 0);
    assert!(!r.is_accepting());
    // Fresh use after reset.
    feed_str(&mut r, "0");
    assert!(r.is_accepting());
}

#[test]
fn earley_accepts_ab_grammar() {
    let mut r = recognizer_from_bnf(r#"<S> ::= "a" <S> "b" | "ab""#);
    // "ab"
    assert!(feed_str(&mut r, "ab"));
    assert!(r.is_accepting());
    r.reset();
    // "aabb"
    assert!(feed_str(&mut r, "aabb"));
    assert!(r.is_accepting());
    r.reset();
    // "aaabbb"
    assert!(feed_str(&mut r, "aaabbb"));
    assert!(r.is_accepting());
}

#[test]
fn earley_rejects_ab_grammar_wrong() {
    let mut r = recognizer_from_bnf(r#"<S> ::= "a" <S> "b" | "ab""#);
    let ok = feed_str(&mut r, "ba");
    assert!(!ok || !r.is_accepting());
}

#[test]
fn earley_clone_state_is_independent() {
    let mut r = recognizer_from_bnf(r#"
        <expr>   ::= <term> "+" <expr> | <term>
        <term>   ::= "0" | "1" | "2"
    "#);
    feed_str(&mut r, "1");
    let pos_before = r.input_pos;
    let mut clone = r.clone_state();

    // Advance the original.
    feed_str(&mut r, "+2");
    assert!(r.is_accepting());
    assert_eq!(r.input_pos, 3);

    // Clone should still be at the snapshot position.
    assert_eq!(clone.input_pos, pos_before);
    // Clone can take its own path.
    feed_str(&mut clone, "+0");
    assert!(clone.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// GrammarConstraint integration tests
// ─────────────────────────────────────────────────────────────────────────────

// 8-token vocab used in several tests:
//   0='0', 1='1', 2='+', 3='*', 4='(', 5=')', 6='\n', 7=<eos (empty)>
fn make_mini_vocab_constraint(bnf: &str) -> GrammarConstraint {
    let g = parse_bnf(bnf).expect("valid BNF");
    GrammarConstraint::new(g, |id| match id {
        0 => vec![b'0'],
        1 => vec![b'1'],
        2 => vec![b'+'],
        3 => vec![b'*'],
        4 => vec![b'('],
        5 => vec![b')'],
        6 => vec![b'\n'],
        7 => vec![], // EOS
        _ => vec![],
    }, 8)
}

const MINI_ARITH_BNF: &str = r#"
    <expr>   ::= <term> "+" <expr> | <term>
    <term>   ::= <factor> "*" <term> | <factor>
    <factor> ::= "(" <expr> ")" | <number>
    <number> ::= "0" | "1"
"#;

#[test]
fn grammar_constraint_arithmetic_allows_correct_next() {
    let c = make_mini_vocab_constraint(MINI_ARITH_BNF);
    let mask = c.allowed_tokens(&[], 8).unwrap();
    // Digits (0='0', 1='1') and '(' (4) should be allowed at start.
    assert!(mask[0], "token 0 ('0') should be allowed at start");
    assert!(mask[1], "token 1 ('1') should be allowed at start");
    assert!(mask[4], "token 4 ('(') should be allowed at start");
    // Operators and newline should not.
    assert!(!mask[2], "token 2 ('+') should NOT be allowed at start");
    assert!(!mask[3], "token 3 ('*') should NOT be allowed at start");
    assert!(!mask[6], "token 6 ('\\n') should NOT be allowed at start");
}

#[test]
fn grammar_constraint_arithmetic_rejects_wrong_token() {
    let mut c = make_mini_vocab_constraint(MINI_ARITH_BNF);
    // Trying to advance with '+' (id=2) at start should fail.
    let ok = c.advance(2);
    assert!(!ok, "advancing '+' at start should return false");
}

#[test]
fn grammar_constraint_advance_ok() {
    let mut c = make_mini_vocab_constraint(MINI_ARITH_BNF);
    // Advance with '1' (id=1) — should succeed.
    assert!(c.advance(1), "advancing '1' should succeed");
    assert!(c.is_complete(), "'1' alone is a complete expression");
}

#[test]
fn grammar_constraint_advance_violation() {
    let mut c = make_mini_vocab_constraint(MINI_ARITH_BNF);
    // Advance with '+' at start — violation.
    assert!(!c.advance(2));
}

#[test]
fn grammar_constraint_complete_when_accepting() {
    let mut c = make_mini_vocab_constraint(MINI_ARITH_BNF);
    assert!(!c.is_complete(), "should not be complete initially");
    c.advance(0); // '0'
    assert!(c.is_complete(), "single digit should be a complete expression");
}

#[test]
fn grammar_constraint_reset() {
    let mut c = make_mini_vocab_constraint(MINI_ARITH_BNF);
    c.advance(1); // '1'
    assert!(c.is_complete());
    c.reset();
    assert!(!c.is_complete());
    assert_eq!(c.bytes_consumed(), 0);
}

#[test]
fn grammar_constraint_implements_token_constraint_trait() {
    let c: Box<dyn TokenConstraint> = Box::new(make_mini_vocab_constraint(MINI_ARITH_BNF));
    assert_eq!(c.name(), "GrammarConstraint");
}

#[test]
fn grammar_constraint_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GrammarConstraint>();
}

/// Generate "1+0" token by token, checking the allowed mask at each step.
#[test]
fn grammar_constraint_arithmetic_full_sequence() {
    let mut c = make_mini_vocab_constraint(MINI_ARITH_BNF);

    // Step 1: before any token, digits and '(' should be in the mask.
    {
        let mask = c.allowed_tokens(&[], 8).unwrap();
        assert!(mask[0], "'0' allowed at start");
        assert!(mask[1], "'1' allowed at start");
        assert!(mask[4], "'(' allowed at start");
        assert!(!mask[2], "'+' not allowed at start");
    }

    // Advance '1' (id=1).
    assert!(c.advance(1));
    assert!(c.is_complete(), "'1' is complete");

    // Step 2: after '1', operators should be in the mask.
    {
        let mask = c.allowed_tokens(&[1], 8).unwrap();
        assert!(mask[2], "'+' allowed after digit");
        assert!(mask[3], "'*' allowed after digit");
        // EOS (7) should be allowed since we are accepting.
        assert!(mask[7], "EOS allowed when accepting");
    }

    // Advance '+' (id=2).
    assert!(c.advance(2));
    assert!(!c.is_complete(), "incomplete after '1+'");

    // Step 3: after '1+', only digits and '(' should be in the mask.
    {
        let mask = c.allowed_tokens(&[1, 2], 8).unwrap();
        assert!(mask[0], "'0' allowed after '1+'");
        assert!(mask[1], "'1' allowed after '1+'");
        assert!(!mask[2], "'+' not allowed after '1+'");
    }

    // Advance '0' (id=0).
    assert!(c.advance(0));
    assert!(c.is_complete(), "'1+0' is a complete expression");
}

#[test]
fn grammar_constraint_csv_row() {
    let mut c = ascii_constraint(csv_row_grammar());
    // "a,b" is a valid two-field CSV row.
    for &b in b"a,b" {
        assert!(c.advance(b as u32), "byte {b} should be accepted");
    }
    assert!(c.is_complete());
}

// ─────────────────────────────────────────────────────────────────────────────
// Example grammar tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn examples_arithmetic_grammar_parses() {
    let g = arithmetic_grammar();
    assert!(g.rules.len() >= 10);
    assert_eq!(g.nt_name(g.start()), "expr");
}

#[test]
fn examples_simple_ab_grammar_parses() {
    let g = simple_ab_grammar();
    assert!(!g.rules.is_empty());
    assert_eq!(g.nt_name(g.start()), "S");
}

#[test]
fn examples_csv_row_grammar_parses() {
    let g = csv_row_grammar();
    assert!(!g.rules.is_empty());
    assert_eq!(g.nt_name(g.start()), "row");
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases and additional coverage
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn grammar_constraint_all_false_after_live_rejection() {
    let mut c = ascii_constraint(arithmetic_grammar());
    c.advance(b'+' as u32); // invalid first token
    let mask = c.allowed_tokens(&[], 128).unwrap();
    assert!(mask.iter().all(|&b| !b), "all tokens blocked after violation");
}

#[test]
fn grammar_constraint_reset_allows_fresh_use() {
    let mut c = ascii_constraint(arithmetic_grammar());
    c.advance(b'1' as u32);
    c.advance(b'+' as u32);
    c.reset();
    // After reset, should work again.
    let mask = c.allowed_tokens(&[], 128).unwrap();
    for d in b'0'..=b'9' {
        assert!(mask[d as usize], "digit {d} should be allowed after reset");
    }
}

#[test]
fn grammar_constraint_paren_expression() {
    let mut c = ascii_constraint(arithmetic_grammar());
    // "(1+2)*3" should be fully accepted.
    for &b in b"(1+2)*3" {
        assert!(c.advance(b as u32), "byte '{}' should be accepted", b as char);
    }
    assert!(c.is_complete());
}

#[test]
fn grammar_constraint_multi_char_terminal_normalisation() {
    // Use a grammar with a 2-char terminal to test normalisation.
    let g = parse_bnf(r#"<S> ::= "ab" | "cd""#).expect("valid");
    let mut c = GrammarConstraint::new(
        g,
        |id| if id < 128 { vec![id as u8] } else { vec![] },
        128,
    );
    // 'a' should be allowed.
    let mask = c.allowed_tokens(&[], 128).unwrap();
    assert!(mask[b'a' as usize]);
    assert!(mask[b'c' as usize]);
    assert!(!mask[b'b' as usize]); // 'b' not first byte of any terminal

    // Feed 'a' — now only 'b' should follow.
    c.advance(b'a' as u32);
    let mask2 = c.allowed_tokens(&[b'a' as u32], 128).unwrap();
    assert!(mask2[b'b' as usize]);
    assert!(!mask2[b'a' as usize]);
    c.advance(b'b' as u32);
    assert!(c.is_complete());
}

#[test]
fn grammar_constraint_left_recursive_language() {
    // E ::= E "+" "1" | "1"   (left-recursive)
    let g = parse_bnf(r#"<E> ::= <E> "+" "1" | "1""#).expect("valid");
    let mut c = GrammarConstraint::new(
        g,
        |id| if id < 128 { vec![id as u8] } else { vec![] },
        128,
    );
    // "1" is accepted.
    c.advance(b'1' as u32);
    assert!(c.is_complete());
    c.reset();
    // "1+1+1" is accepted.
    for &b in b"1+1+1" {
        assert!(c.advance(b as u32), "byte '{}' should be accepted", b as char);
    }
    assert!(c.is_complete());
}

#[test]
fn grammar_constraint_nullable_nt_in_sequence() {
    // S ::= <A> "x"
    // A ::= "" | "a"
    let g = parse_bnf(r#"
        <S> ::= <A> "x"
        <A> ::= "" | "a"
    "#).expect("valid");
    let mut c = GrammarConstraint::new(
        g,
        |id| if id < 128 { vec![id as u8] } else { vec![] },
        128,
    );
    // "x" should be accepted (A → ε).
    c.advance(b'x' as u32);
    assert!(c.is_complete());
    c.reset();
    // "ax" should be accepted.
    c.advance(b'a' as u32);
    c.advance(b'x' as u32);
    assert!(c.is_complete());
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional Rule/Symbol API coverage
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rule_epsilon_predicate() {
    let r = Rule::new(0, vec![]);
    assert!(r.is_epsilon());
    assert_eq!(r.rhs_len(), 0);

    let r2 = Rule::new(0, vec![Symbol::Terminal(vec![65])]);
    assert!(!r2.is_epsilon());
    assert_eq!(r2.rhs_len(), 1);
}

#[test]
fn symbol_accessors() {
    let t = Symbol::Terminal(vec![42]);
    assert!(t.is_terminal());
    assert!(!t.is_non_terminal());
    assert_eq!(t.terminal_bytes(), Some([42u8].as_ref()));
    assert_eq!(t.non_terminal_id(), None);

    let nt = Symbol::NonTerminal(7);
    assert!(!nt.is_terminal());
    assert!(nt.is_non_terminal());
    assert_eq!(nt.non_terminal_id(), Some(7));
    assert_eq!(nt.terminal_bytes(), None);
}

#[test]
fn grammar_rules_for_iterator() {
    let g = parse_bnf(r#"<S> ::= "a" | "b" | "c""#).expect("valid");
    let s_id = g.start();
    let rules: Vec<_> = g.rules_for(s_id).collect();
    assert_eq!(rules.len(), 3);
}

#[test]
fn grammar_nt_name_unknown_id() {
    let g = Grammar::new(0);
    assert_eq!(g.nt_name(9999), "<unknown>");
}
