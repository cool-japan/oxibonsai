//! Pre-canned grammars for use in tests and as examples.
//!
//! Each function returns a fully-parsed [`Grammar`] that has **not** yet been
//! normalised.  Call [`Grammar::normalise_terminals`] before passing the
//! grammar to [`super::earley::EarleyRecognizer`] or
//! [`super::constraint::GrammarConstraint`] (the constraint builder does this
//! automatically).

use super::ast::Grammar;
use super::bnf_parser::parse_bnf;

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic grammar
// ─────────────────────────────────────────────────────────────────────────────

/// An arithmetic expression grammar covering +, -, *, / and parentheses.
///
/// ```text
/// <expr>   ::= <term> "+" <expr> | <term> "-" <expr> | <term>
/// <term>   ::= <factor> "*" <term> | <factor> "/" <term> | <factor>
/// <factor> ::= "(" <expr> ")" | <number>
/// <number> ::= "0" | "1" | ... | "9"
/// ```
///
/// The grammar is right-recursive (as is typical for BNF) and handles standard
/// arithmetic precedence implicitly through the rule structure.
pub fn arithmetic_grammar() -> Grammar {
    parse_bnf(
        r#"
        <expr>   ::= <term> "+" <expr> | <term> "-" <expr> | <term>
        <term>   ::= <factor> "*" <term> | <factor> "/" <term> | <factor>
        <factor> ::= "(" <expr> ")" | <number>
        <number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#,
    )
    .expect("arithmetic grammar is valid BNF")
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple a^n b^n grammar
// ─────────────────────────────────────────────────────────────────────────────

/// The classic context-free language `{ a^n b^n | n ≥ 1 }`.
///
/// ```text
/// <S> ::= "a" <S> "b" | "ab"
/// ```
///
/// This grammar is right-recursive and tests balanced nesting of 'a's and 'b's.
/// It is **not** regular — it cannot be expressed as a regular expression.
pub fn simple_ab_grammar() -> Grammar {
    parse_bnf(
        r#"
        <S> ::= "a" <S> "b" | "ab"
    "#,
    )
    .expect("ab grammar is valid BNF")
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV row grammar
// ─────────────────────────────────────────────────────────────────────────────

/// A grammar for a single CSV row with lowercase-letter fields.
///
/// ```text
/// <row>   ::= <field> "," <row> | <field>
/// <field> ::= <char> <field> | <char>
/// <char>  ::= "a" | "b" | ... | "z"
/// ```
///
/// This matches rows like `"a,b,c"`, `"hello,world"`, etc.
pub fn csv_row_grammar() -> Grammar {
    parse_bnf(r#"
        <row>   ::= <field> "," <row> | <field>
        <field> ::= <char> <field> | <char>
        <char>  ::= "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
    "#)
    .expect("csv grammar is valid BNF")
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON-lite grammar (structural, not full JSON spec)
// ─────────────────────────────────────────────────────────────────────────────

/// A simplified JSON-like grammar covering objects, arrays, strings
/// (ASCII letters and digits only), numbers, and the boolean/null literals.
///
/// This is deliberately simplified for testing purposes — a production JSON
/// grammar would be much more elaborate.
pub fn json_lite_grammar() -> Grammar {
    parse_bnf(r#"
        <value>   ::= <object> | <array> | <string> | <number> | "true" | "false" | "null"
        <object>  ::= "{" "}" | "{" <members> "}"
        <members> ::= <pair> | <pair> "," <members>
        <pair>    ::= <string> ":" <value>
        <array>   ::= "[" "]" | "[" <elements> "]"
        <elements> ::= <value> | <value> "," <elements>
        <string>  ::= "\"" "\"" | "\"" <chars> "\""
        <chars>   ::= <char> | <char> <chars>
        <char>    ::= "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z" | "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        <number>  ::= <digit> | <digit> <number>
        <digit>   ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#)
    .expect("json_lite grammar is valid BNF")
}

// ─────────────────────────────────────────────────────────────────────────────
// Palindrome grammar
// ─────────────────────────────────────────────────────────────────────────────

/// Grammar for palindromes over the alphabet {a, b}.
///
/// ```text
/// <P> ::= "a" <P> "a" | "b" <P> "b" | "a" | "b" | ""
/// ```
///
/// This is an inherently ambiguous grammar (the empty string can always be
/// parsed as an epsilon palindrome), but Earley handles ambiguity correctly.
pub fn palindrome_grammar() -> Grammar {
    parse_bnf(
        r#"
        <P> ::= "a" <P> "a" | "b" <P> "b" | "a" | "b" | ""
    "#,
    )
    .expect("palindrome grammar is valid BNF")
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn examples_arithmetic_grammar_parses() {
        let g = arithmetic_grammar();
        assert!(
            g.rules.len() >= 10,
            "arithmetic grammar should have many rules"
        );
        let start_name = g.nt_name(g.start()).to_string();
        assert_eq!(start_name, "expr");
    }

    #[test]
    fn examples_simple_ab_grammar_parses() {
        let g = simple_ab_grammar();
        assert!(!g.rules.is_empty());
        let start_name = g.nt_name(g.start()).to_string();
        assert_eq!(start_name, "S");
    }

    #[test]
    fn examples_csv_row_grammar_parses() {
        let g = csv_row_grammar();
        assert!(!g.rules.is_empty());
        let start_name = g.nt_name(g.start()).to_string();
        assert_eq!(start_name, "row");
    }

    #[test]
    fn examples_json_lite_grammar_parses() {
        let g = json_lite_grammar();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn examples_palindrome_grammar_parses() {
        let g = palindrome_grammar();
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn examples_arithmetic_grammar_normalises() {
        let mut g = arithmetic_grammar();
        let count_before = g.rules.len();
        g.normalise_terminals();
        // No new rules should be added since all terminals are single chars.
        // (All terminals in the arithmetic grammar are single characters.)
        assert_eq!(
            g.rules.len(),
            count_before,
            "single-char terminals need no splitting"
        );
    }
}
