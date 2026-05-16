//! Grammar Abstract Syntax Tree types for context-free grammar definition.
//!
//! Terminals are represented as **byte sequences** (not Unicode strings),
//! making the engine tokenizer-agnostic. Any byte-level token stream can be
//! constrained without knowledge of the underlying text encoding.

use std::collections::HashMap;

/// Identifier for a non-terminal symbol (index into the grammar's rule list).
pub type NonTerminalId = usize;

/// Identifier for a specific grammar rule (index into `Grammar::rules`).
pub type RuleId = usize;

/// A sentinel `NonTerminalId` used to represent "no non-terminal" in contexts
/// where an optional NT id is needed.
pub const NULL_NT: NonTerminalId = usize::MAX;

// ─────────────────────────────────────────────────────────────────────────────
// Symbol
// ─────────────────────────────────────────────────────────────────────────────

/// One element on the right-hand side of a grammar rule.
///
/// # Design: byte-level terminals
///
/// Terminals are stored as raw byte sequences, not Unicode strings.  This
/// means the Earley engine can operate directly on byte streams and, crucially,
/// on the *subword byte sequences* that LLM tokenizers produce.  Multi-byte
/// terminals are pre-expanded into chains of single-byte rules before the
/// recognizer is constructed, ensuring each scan step advances by exactly one
/// byte.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Symbol {
    /// A literal byte sequence.  An empty `Vec` is a valid ε-terminal (epsilon).
    Terminal(Vec<u8>),
    /// A reference to another grammar rule identified by its [`NonTerminalId`].
    NonTerminal(NonTerminalId),
}

impl Symbol {
    /// Returns `true` if this is a terminal.
    #[inline]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Symbol::Terminal(_))
    }

    /// Returns `true` if this is a non-terminal.
    #[inline]
    pub fn is_non_terminal(&self) -> bool {
        matches!(self, Symbol::NonTerminal(_))
    }

    /// Returns the byte slice for a terminal, or `None` for a non-terminal.
    #[inline]
    pub fn terminal_bytes(&self) -> Option<&[u8]> {
        match self {
            Symbol::Terminal(b) => Some(b),
            Symbol::NonTerminal(_) => None,
        }
    }

    /// Returns the `NonTerminalId` for a non-terminal, or `None` for a terminal.
    #[inline]
    pub fn non_terminal_id(&self) -> Option<NonTerminalId> {
        match self {
            Symbol::NonTerminal(id) => Some(*id),
            Symbol::Terminal(_) => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule
// ─────────────────────────────────────────────────────────────────────────────

/// A single context-free production rule `lhs → rhs`.
///
/// An empty `rhs` represents an ε (epsilon / nullable) production.
#[derive(Debug, Clone)]
pub struct Rule {
    /// The non-terminal this rule expands.
    pub lhs: NonTerminalId,
    /// The sequence of symbols on the right-hand side.  Empty = epsilon.
    pub rhs: Vec<Symbol>,
}

impl Rule {
    /// Create a new rule.
    pub fn new(lhs: NonTerminalId, rhs: Vec<Symbol>) -> Self {
        Self { lhs, rhs }
    }

    /// Return the length of the right-hand side (0 = epsilon rule).
    pub fn rhs_len(&self) -> usize {
        self.rhs.len()
    }

    /// Returns `true` if this is an epsilon (empty) production.
    pub fn is_epsilon(&self) -> bool {
        self.rhs.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grammar
// ─────────────────────────────────────────────────────────────────────────────

/// A context-free grammar in BNF form.
///
/// Rules are indexed by their position in `rules`.  Non-terminals are
/// identified by a dense integer id (`0..nt_count`).  The start symbol is
/// recorded in `start`.
///
/// # Terminal normalisation
///
/// After construction from text, call [`Grammar::normalise_terminals`] to
/// split any multi-byte terminal into a chain of single-byte rules.  The
/// Earley recognizer requires this normalisation.
#[derive(Debug, Clone)]
pub struct Grammar {
    /// All production rules in insertion order.
    pub rules: Vec<Rule>,
    /// The start (root) non-terminal.
    pub start: NonTerminalId,
    /// Human-readable names for non-terminals (used for debugging / error messages).
    pub nt_names: HashMap<NonTerminalId, String>,
    /// Total number of distinct non-terminals allocated so far.
    pub nt_count: usize,
}

impl Grammar {
    /// Create an empty grammar with the given start symbol.
    ///
    /// `start` is the `NonTerminalId` of the axiom.  Callers must ensure that
    /// at least one rule with `lhs == start` is added before the grammar is
    /// used with the Earley recognizer.
    pub fn new(start: NonTerminalId) -> Self {
        Self {
            rules: Vec::new(),
            start,
            nt_names: HashMap::new(),
            nt_count: 0,
        }
    }

    /// Append a rule to the grammar.
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Iterate over all rules whose `lhs` matches `nt`.
    ///
    /// Returns pairs of `(RuleId, &Rule)` so callers can store rule indices.
    pub fn rules_for(&self, nt: NonTerminalId) -> impl Iterator<Item = (RuleId, &Rule)> + '_ {
        self.rules
            .iter()
            .enumerate()
            .filter(move |(_, r)| r.lhs == nt)
    }

    /// Return the start non-terminal id.
    pub fn start(&self) -> NonTerminalId {
        self.start
    }

    /// Return a human-readable name for a non-terminal, or `"<unknown>"`.
    pub fn nt_name(&self, id: NonTerminalId) -> &str {
        self.nt_names
            .get(&id)
            .map(|s| s.as_str())
            .unwrap_or("<unknown>")
    }

    /// Allocate a fresh `NonTerminalId` with the given debug name.
    ///
    /// This is used by [`Grammar::normalise_terminals`] and the BNF parser to
    /// create synthetic non-terminals for multi-byte terminal chains.
    pub fn alloc_nt(&mut self, name: impl Into<String>) -> NonTerminalId {
        let id = self.nt_count;
        self.nt_count += 1;
        self.nt_names.insert(id, name.into());
        id
    }

    /// Normalise all multi-byte terminals into chains of single-byte rules.
    ///
    /// The Earley recognizer scans one byte at a time.  Each terminal byte
    /// sequence `[b0, b1, ..., bk]` is therefore split into a fresh synthetic
    /// non-terminal whose rules chain the individual bytes:
    ///
    /// ```text
    /// <__T_b0b1...bk> ::= b0 <__T_b1...bk>
    /// <__T_bk>        ::= bk
    /// ```
    ///
    /// Single-byte terminals are left as-is.  Empty (ε) terminals are left
    /// as-is.
    ///
    /// This method is idempotent after one normalisation pass because all
    /// newly inserted terminals are guaranteed to be single bytes.
    pub fn normalise_terminals(&mut self) {
        // Cache already-seen multi-byte sequences so we don't add duplicate chains.
        let mut cache: HashMap<Vec<u8>, NonTerminalId> = HashMap::new();

        // We must iterate over a snapshot because we mutate self.rules.
        let rule_count = self.rules.len();
        for rule_idx in 0..rule_count {
            // We will rebuild the rhs in a scratch buffer and replace if changed.
            let mut new_rhs: Vec<Symbol> = Vec::new();
            let mut changed = false;

            // Clone the rhs to avoid borrow issues.
            let rhs: Vec<Symbol> = self.rules[rule_idx].rhs.clone();
            for symbol in rhs {
                match &symbol {
                    Symbol::Terminal(bytes) if bytes.len() > 1 => {
                        changed = true;
                        // Expand the multi-byte terminal into a chain NT.
                        let chain_nt = Self::intern_byte_chain(self, bytes.clone(), &mut cache);
                        new_rhs.push(Symbol::NonTerminal(chain_nt));
                    }
                    other => {
                        new_rhs.push(other.clone());
                    }
                }
            }

            if changed {
                self.rules[rule_idx].rhs = new_rhs;
            }
        }
    }

    /// Internal helper: ensure a chain NT exists for `bytes` and return its id.
    ///
    /// Recursively ensures that `bytes[1..]` also has a chain NT when `len > 2`.
    fn intern_byte_chain(
        grammar: &mut Grammar,
        bytes: Vec<u8>,
        cache: &mut HashMap<Vec<u8>, NonTerminalId>,
    ) -> NonTerminalId {
        if let Some(&id) = cache.get(&bytes) {
            return id;
        }

        // Build a descriptive name like `__T_61_62_63` for bytes [0x61, 0x62, 0x63].
        let name = {
            let hex: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
            format!("__T_{}", hex.join("_"))
        };

        let nt = grammar.alloc_nt(&name);
        cache.insert(bytes.clone(), nt);

        if bytes.len() == 1 {
            // Base case: single-byte chain rule `NT ::= [b0]`.
            grammar.rules.push(Rule {
                lhs: nt,
                rhs: vec![Symbol::Terminal(bytes)],
            });
        } else {
            // Recursive case: `NT ::= [b0] <rest_nt>`.
            let rest = bytes[1..].to_vec();
            let rest_nt = Self::intern_byte_chain(grammar, rest, cache);
            grammar.rules.push(Rule {
                lhs: nt,
                rhs: vec![
                    Symbol::Terminal(vec![bytes[0]]),
                    Symbol::NonTerminal(rest_nt),
                ],
            });
        }

        nt
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_grammar() -> Grammar {
        // S ::= "ab" | "c"
        let mut g = Grammar::new(0);
        g.alloc_nt("S"); // id = 0
        g.add_rule(Rule::new(0, vec![Symbol::Terminal(b"ab".to_vec())]));
        g.add_rule(Rule::new(0, vec![Symbol::Terminal(b"c".to_vec())]));
        g
    }

    #[test]
    fn symbol_is_terminal() {
        let t = Symbol::Terminal(vec![65]);
        assert!(t.is_terminal());
        assert!(!t.is_non_terminal());
        assert_eq!(t.terminal_bytes(), Some([65u8].as_ref()));
        assert_eq!(t.non_terminal_id(), None);
    }

    #[test]
    fn symbol_is_non_terminal() {
        let nt = Symbol::NonTerminal(3);
        assert!(nt.is_non_terminal());
        assert!(!nt.is_terminal());
        assert_eq!(nt.non_terminal_id(), Some(3));
        assert_eq!(nt.terminal_bytes(), None);
    }

    #[test]
    fn grammar_rules_for() {
        let g = make_simple_grammar();
        let rules: Vec<_> = g.rules_for(0).collect();
        assert_eq!(rules.len(), 2);
    }

    #[test]
    fn grammar_nt_name_unknown() {
        let g = Grammar::new(0);
        assert_eq!(g.nt_name(99), "<unknown>");
    }

    #[test]
    fn grammar_normalise_multi_byte_terminal() {
        let mut g = make_simple_grammar();
        let original_rule_count = g.rules.len();
        g.normalise_terminals();
        // The rule `S ::= "ab"` should have been rewritten to
        // `S ::= <chain>` where chain is a synthetic NT.
        // New rules should have been appended.
        assert!(g.rules.len() > original_rule_count);
        // The first rule's rhs should now contain a NonTerminal (the chain NT).
        let first_rhs = &g.rules[0].rhs;
        assert_eq!(first_rhs.len(), 1);
        assert!(first_rhs[0].is_non_terminal());
    }

    #[test]
    fn grammar_normalise_idempotent() {
        let mut g = make_simple_grammar();
        g.normalise_terminals();
        let count_after_first = g.rules.len();
        // Second pass must not add more rules.
        g.normalise_terminals();
        assert_eq!(g.rules.len(), count_after_first);
    }

    #[test]
    fn rule_is_epsilon() {
        let r = Rule::new(0, vec![]);
        assert!(r.is_epsilon());
        assert_eq!(r.rhs_len(), 0);

        let r2 = Rule::new(0, vec![Symbol::Terminal(vec![65])]);
        assert!(!r2.is_epsilon());
    }
}
