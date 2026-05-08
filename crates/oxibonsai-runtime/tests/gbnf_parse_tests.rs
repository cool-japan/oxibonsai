//! Integration tests for the GBNF grammar parser.
//!
//! Tests cover:
//! * Error cases (empty, missing root, unknown rule)
//! * Literals, alternation, sequences
//! * Quantifiers: `*`, `+`, `?`
//! * Char classes: simple, range, negated, mixed, hex escapes
//! * Groups with quantifiers
//! * Rule references, recursive rules
//! * Comments, escape sequences
//! * Complex grammars
//! * Integration with GrammarConstraint

use oxibonsai_runtime::grammar::{parse_gbnf, GbnfParseError, GrammarConstraint};

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

/// Parse GBNF or panic with a descriptive message.
fn parse_ok(src: &str) -> oxibonsai_runtime::grammar::Grammar {
    parse_gbnf(src).unwrap_or_else(|e| panic!("parse_gbnf failed on:\n{src}\nError: {e}"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: empty grammar → EmptyGrammar
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_empty_grammar_error() {
    assert!(matches!(parse_gbnf(""), Err(GbnfParseError::EmptyGrammar)));
    assert!(matches!(
        parse_gbnf("   \n\n  \t\n"),
        Err(GbnfParseError::EmptyGrammar)
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: no `root` rule → MissingRootRule
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_missing_root_rule_error() {
    let err = parse_gbnf(r#"word ::= "hello""#);
    assert!(matches!(err, Err(GbnfParseError::MissingRootRule)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: simple string literal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_simple_literal() {
    let g = parse_ok(r#"root ::= "hello""#);
    assert!(g.start() < g.nt_count, "start symbol must be a valid NT id");
    // Root rule should have a non-empty RHS (each byte is a terminal).
    let root_rules: Vec<_> = g.rules_for(g.start()).collect();
    assert!(!root_rules.is_empty(), "root must have at least one rule");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: alternation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_alternation() {
    let g = parse_ok(r#"root ::= "cat" | "dog""#);
    let root_rules: Vec<_> = g.rules_for(g.start()).collect();
    // Two alternatives → two rules for root.
    assert_eq!(root_rules.len(), 2, "expected 2 root rules for alternation");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Kleene star
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_kleene_star() {
    let g = parse_ok(r#"root ::= "a"*"#);
    // Grammar must have rules (synthetic star NT adds ε and recursive rules).
    assert!(
        g.rules.len() >= 2,
        "star should create synthetic NT with at least 2 rules, got {}",
        g.rules.len()
    );
    // Verify we can find the synthetic star NT (ε rule).
    let has_epsilon = g.rules.iter().any(|r| r.is_epsilon());
    assert!(has_epsilon, "star must introduce an ε rule");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: plus quantifier
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_plus() {
    let g = parse_ok(r#"root ::= "a"+"#);
    // Plus creates star NT (2 rules) and plus NT (1 rule).
    assert!(
        g.rules.len() >= 3,
        "plus should create 3+ rules, got {}",
        g.rules.len()
    );
    // The `+` case must NOT have an ε rule for the plus NT itself.
    // (The star helper does, but not the plus NT.)
    let root_start = g.start();
    let root_rules: Vec<_> = g.rules_for(root_start).collect();
    // Root delegates to the plus_nt, so root has exactly 1 rule.
    assert_eq!(root_rules.len(), 1);
    // That rule's RHS is non-empty (not ε).
    assert!(!root_rules[0].1.is_epsilon());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: optional (`?`)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_optional() {
    let g = parse_ok(r#"root ::= "colo" "u"? "r""#);
    // The optional `u` expands to a synthetic opt NT with an ε rule.
    let has_epsilon = g.rules.iter().any(|r| r.is_epsilon());
    assert!(has_epsilon, "optional must introduce an ε rule");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: char class (simple)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_char_class_simple() {
    let g = parse_ok(r#"root ::= [abc]+"#);
    // The char class `[abc]` matches 3 bytes → 3 rules for the class NT.
    // Plus 2 rules for the star, 1 for plus, 1 for root = 7 rules minimum.
    assert!(
        g.rules.len() >= 6,
        "char class plus should produce 6+ rules, got {}",
        g.rules.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: char class with range
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_char_class_range() {
    let g = parse_ok(r#"root ::= [a-z]+"#);
    // `[a-z]` → 26 bytes. 26 rules for class_nt + 2 star + 1 plus + 1 root = 30+.
    assert!(
        g.rules.len() >= 29,
        "a-z class + plus expected >=29 rules, got {}",
        g.rules.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10: negated char class
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_char_class_negated() {
    let g = parse_ok(r#"root ::= [^abc]+"#);
    // `[^abc]` matches 256 - 3 = 253 bytes.
    // 253 class rules + 2 star + 1 plus + 1 root = 257+.
    assert!(
        g.rules.len() >= 256,
        "negated class [^abc] + plus expected >=256 rules, got {}",
        g.rules.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11: mixed char class
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_char_class_mixed() {
    let g = parse_ok(r#"root ::= [a-zA-Z0-9_]+"#);
    // a-z = 26, A-Z = 26, 0-9 = 10, _ = 1 → 63 distinct bytes.
    let class_nt_rules: Vec<_> = g
        .rules
        .iter()
        .filter(|r| {
            // Char class NT rules have a single 1-byte terminal RHS.
            r.rhs.len() == 1
                && r.rhs[0]
                    .terminal_bytes()
                    .map(|b| b.len() == 1)
                    .unwrap_or(false)
                && r.lhs != g.start()
        })
        .collect();
    // Should have exactly 63 such rules (one per matching byte in the class).
    assert_eq!(
        class_nt_rules.len(),
        63,
        "mixed class should have 63 terminal rules, got {}",
        class_nt_rules.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12: rule reference
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_rule_reference() {
    let g = parse_ok("root ::= word\nword ::= [a-z]+\n");
    // `root` rule should reference `word` NT.
    let root_rules: Vec<_> = g.rules_for(g.start()).collect();
    assert_eq!(root_rules.len(), 1);
    // The single root rule RHS should be a NonTerminal (reference to `word`).
    assert!(
        root_rules[0].1.rhs.iter().any(|s| s.is_non_terminal()),
        "root should reference word via NonTerminal"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 13: unknown rule reference → UnknownRule error
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_unknown_rule_reference_error() {
    // `ghost` is referenced but never defined.
    let err = parse_gbnf("root ::= ghost");
    assert!(
        matches!(err, Err(GbnfParseError::UnknownRule(ref n)) if n == "ghost"),
        "expected UnknownRule(\"ghost\"), got {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 14: comment is ignored
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_comment_ignored() {
    let g = parse_ok("# this is a comment\nroot ::= \"x\"");
    let root_rules: Vec<_> = g.rules_for(g.start()).collect();
    assert_eq!(root_rules.len(), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 15: string escape sequences
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_string_escape_sequences() {
    let g = parse_ok(r#"root ::= "\n\t\r\\""#);
    // Should parse without error; root rule has 4 byte terminals.
    let root_rules: Vec<_> = g.rules_for(g.start()).collect();
    assert!(!root_rules.is_empty());
    // RHS symbols should include the escaped bytes.
    let total_terminals: usize = root_rules
        .iter()
        .map(|(_, r)| r.rhs.iter().filter(|s| s.is_terminal()).count())
        .sum();
    assert_eq!(total_terminals, 4, "expected 4 terminal bytes (\\n, \\t, \\r, \\\\)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 16: hex escape in char class
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hex_escape_in_class() {
    // `[\x41-\x5A]` = A (0x41) to Z (0x5A) = 26 bytes.
    let g = parse_ok(r#"root ::= [\x41-\x5A]+"#);
    // Count single-byte terminal rules not in root or synthetic stars/plus.
    let class_rules: Vec<_> = g
        .rules
        .iter()
        .filter(|r| {
            r.rhs.len() == 1
                && r.rhs[0]
                    .terminal_bytes()
                    .map(|b| b.len() == 1)
                    .unwrap_or(false)
        })
        .collect();
    assert_eq!(
        class_rules.len(),
        26,
        "\\x41-\\x5A should produce 26 terminal rules, got {}",
        class_rules.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 17: group with quantifier
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_group_with_quantifier() {
    let g = parse_ok(r#"root ::= ("a" "b")+"#);
    // Group is a synthetic NT; plus wraps it with star.
    assert!(
        g.rules.len() >= 4,
        "group+ should produce 4+ rules, got {}",
        g.rules.len()
    );
    // ε rule must exist from the star.
    assert!(g.rules.iter().any(|r| r.is_epsilon()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 18: nested alternation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_nested_alternation() {
    let g = parse_ok(r#"root ::= "a" | ("b" | "c")"#);
    // Root has 2 alternatives: "a" and the group.
    let root_rules: Vec<_> = g.rules_for(g.start()).collect();
    assert_eq!(root_rules.len(), 2, "expected 2 root rules for nested alternation");
    // The group NT should have 2 alternatives.
    let group_nt_id = match &root_rules[1].1.rhs[..] {
        [oxibonsai_runtime::grammar::Symbol::NonTerminal(id)] => *id,
        _ => panic!("second alternative should be a NonTerminal group reference"),
    };
    let group_rules: Vec<_> = g.rules_for(group_nt_id).collect();
    assert_eq!(group_rules.len(), 2, "group should have 2 alternatives (b | c)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 19: recursive rule (self-referential)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_recursive_rule() {
    // root ::= "a" root?  — self-referential via optional
    let g = parse_ok(r#"root ::= "a" root?"#);
    // Should parse without infinite loop; root has 1 rule.
    let root_rules: Vec<_> = g.rules_for(g.start()).collect();
    assert_eq!(root_rules.len(), 1);
    // The RHS should have 2 items: terminal 'a' and a NonTerminal (opt wrapping root).
    assert_eq!(root_rules[0].1.rhs.len(), 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 20: multiword sequence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_multiword_sequence() {
    let g = parse_ok(r#"root ::= "hello" " " "world""#);
    let root_rules: Vec<_> = g.rules_for(g.start()).collect();
    assert_eq!(root_rules.len(), 1);
    // "hello" + " " + "world" = 11 bytes → 11 terminal symbols in the sequence.
    let terminal_count = root_rules[0].1.rhs.iter().filter(|s| s.is_terminal()).count();
    assert_eq!(terminal_count, 11, "expected 11 terminal bytes for \"hello\" \" \" \"world\"");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 21: complex JSON-like grammar
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_complex_json_like() {
    // A simplified JSON object grammar.
    let src = r#"
root    ::= "{" ws "}"
          | "{" ws members ws "}"
members ::= pair ("," ws pair)*
pair    ::= string ws ":" ws value
value   ::= string | number
string  ::= "\"" [^"]* "\""
number  ::= [0-9]+
ws      ::= [ \t\n]*
"#;
    // Must parse without error.
    let g = parse_gbnf(src).unwrap_or_else(|e| panic!("JSON-like parse failed: {e}"));
    assert!(!g.rules.is_empty());
    // `root` must be the start symbol.
    let root_name = g.nt_name(g.start()).to_string();
    assert_eq!(root_name, "root");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 22: grammar has correct start symbol
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_grammar_has_correct_start_symbol() {
    let g = parse_ok("root ::= \"x\"\nother ::= \"y\"");
    let start_name = g.nt_name(g.start()).to_string();
    assert_eq!(start_name, "root", "start symbol must be `root`");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 23: integration with GrammarConstraint
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_constraint_integration() {
    let g = parse_ok(r#"root ::= [0-9]+"#);
    // GrammarConstraint::new should not panic.
    let _constraint = GrammarConstraint::new(
        g,
        |id| {
            if id < 128 {
                vec![id as u8]
            } else {
                vec![]
            }
        },
        128,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 24: multiple rules all parse correctly
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_multiple_rules() {
    let src = "root ::= noun verb noun\nnoun ::= \"cat\" | \"dog\"\nverb ::= \"chases\" | \"sees\"\n";
    let g = parse_ok(src);
    // Should have 3 named NTs (root, noun, verb) + synthetic ones.
    assert!(g.nt_count >= 3, "expected at least 3 NTs, got {}", g.nt_count);
    // Root has 1 rule (the sequence noun verb noun).
    let root_rules: Vec<_> = g.rules_for(g.start()).collect();
    assert_eq!(root_rules.len(), 1);
    // noun and verb each have 2 rules.
    let noun_id = *g.nt_names.iter().find(|(_, n)| *n == "noun").unwrap().0;
    let verb_id = *g.nt_names.iter().find(|(_, n)| *n == "verb").unwrap().0;
    assert_eq!(g.rules_for(noun_id).count(), 2);
    assert_eq!(g.rules_for(verb_id).count(), 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 25: decimal number pattern
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_digit_word_pattern() {
    // root ::= [0-9]+ "." [0-9]+  — decimal number like "3.14"
    let g = parse_ok(r#"root ::= [0-9]+ "." [0-9]+"#);
    assert!(!g.rules.is_empty());
    // Grammar must have the root rule and digit class rules.
    let root_name = g.nt_name(g.start()).to_string();
    assert_eq!(root_name, "root");
    // Should contain ε rules (from the star helper inside plus).
    assert!(g.rules.iter().any(|r| r.is_epsilon()), "plus quantifier must introduce ε rule");
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional: GbnfParseError Display impl coverage
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_error_display_coverage() {
    let cases: &[GbnfParseError] = &[
        GbnfParseError::EmptyGrammar,
        GbnfParseError::MissingRootRule,
        GbnfParseError::UnknownRule("foo".to_string()),
        GbnfParseError::UnexpectedChar { line: 1, col: 2, ch: '!' },
        GbnfParseError::UnterminatedString,
        GbnfParseError::UnterminatedCharClass,
        GbnfParseError::InvalidEscape('z'),
        GbnfParseError::UnsupportedFeature("test".to_string()),
        GbnfParseError::RecursionLimit,
    ];
    for e in cases {
        assert!(
            !e.to_string().is_empty(),
            "Display for {e:?} must not be empty"
        );
    }
    // std::error::Error is implemented.
    let e: &dyn std::error::Error = &GbnfParseError::EmptyGrammar;
    assert!(e.source().is_none());
}
