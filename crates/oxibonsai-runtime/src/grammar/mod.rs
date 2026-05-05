//! Context-free grammar engine with Earley-parser-based constrained decoding.
//!
//! This module provides:
//!
//! * A **BNF text parser** ([`parse_bnf`]) that converts a grammar string into
//!   an in-memory [`Grammar`] representation.
//! * An **Earley chart-parser recognizer** ([`EarleyRecognizer`]) that
//!   processes a byte stream against the grammar incrementally, supporting:
//!   - Left-recursive and right-recursive grammars
//!   - Nullable (ε) productions
//!   - Any context-free grammar (not just LL(k) or LR(k))
//! * A **[`GrammarConstraint`]** that wraps the recognizer and implements the
//!   [`crate::constrained_decoding::TokenConstraint`] trait so it can be
//!   dropped into any token-generation pipeline.
//! * Pre-built **[example grammars]** for testing and demonstration.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use oxibonsai_runtime::grammar::{parse_bnf, GrammarConstraint};
//! use oxibonsai_runtime::constrained_decoding::TokenConstraint;
//!
//! let grammar = parse_bnf(r#"
//!     <expr>   ::= <term> "+" <expr> | <term>
//!     <term>   ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
//! "#).expect("valid BNF");
//!
//! let mut constraint = GrammarConstraint::new(grammar, |id| {
//!     if id < 128 { vec![id as u8] } else { vec![] }
//! }, 128);
//!
//! // Check which tokens are allowed at the start.
//! let mask = constraint.allowed_tokens(&[], 128).unwrap();
//! assert!(mask[b'0' as usize]);
//! assert!(!mask[b'+' as usize]);
//! ```
//!
//! [example grammars]: examples

pub mod ast;
pub mod bnf_parser;
pub mod cache;
pub mod constraint;
pub mod earley;
pub mod examples;
pub mod json_schema_compiler;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use ast::{Grammar, NonTerminalId, Rule, RuleId, Symbol, NULL_NT};
pub use bnf_parser::{parse_bnf, BnfParseError};
pub use cache::AllowedTokensCache;
pub use constraint::GrammarConstraint;
pub use earley::EarleyRecognizer;
pub use examples::{
    arithmetic_grammar, csv_row_grammar, json_lite_grammar, palindrome_grammar, simple_ab_grammar,
};
pub use json_schema_compiler::{
    compile_json_schema, compile_json_schema_str, JsonSchemaCompileError,
};
