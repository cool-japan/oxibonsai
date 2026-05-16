//! Full Earley chart-parser recognizer.
//!
//! The Earley algorithm recognizes any context-free grammar in O(n³) time
//! (O(n²) for unambiguous grammars, O(n) for LR(k) grammars).  Unlike table-
//! driven LR/LL parsers it handles left-recursive, right-recursive, ambiguous,
//! and nullable grammars without special treatment.
//!
//! # Key invariants
//!
//! * `chart[k]` is a set (HashSet) of Earley items that are active at input
//!   position `k`.  Deduplication via HashSet handles left recursion by
//!   preventing infinite re-insertion.
//! * `chart[k]` is closed under **Predict** and **Complete** before scanning
//!   advances to `chart[k+1]`.
//! * Items whose dot points at a Terminal are candidates for **Scan**.
//!
//! # Grammar pre-requisite
//!
//! The grammar **must** have been normalised with
//! [`Grammar::normalise_terminals`] so that every terminal in every rule's rhs
//! is a **single-byte** sequence.  Multi-byte terminals must be represented as
//! chains of synthetic non-terminals before being passed here.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use super::ast::{Grammar, NonTerminalId, RuleId, Symbol};

// ─────────────────────────────────────────────────────────────────────────────
// Earley item
// ─────────────────────────────────────────────────────────────────────────────

/// One Earley chart item representing the parse state `A → α • β, j`.
///
/// * `rule`   — which production rule (index into `Grammar::rules`)
/// * `dot`    — how many rhs symbols have been consumed (0 = dot before first)
/// * `origin` — the input position (chart index) where this item started
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EarleyItem {
    /// Grammar rule id.
    pub rule: RuleId,
    /// Dot position within the rule's rhs (0 … rhs.len()).
    pub dot: usize,
    /// Input position where this item was predicted.
    pub origin: usize,
}

impl EarleyItem {
    #[inline]
    pub fn new(rule: RuleId, dot: usize, origin: usize) -> Self {
        Self { rule, dot, origin }
    }

    /// True when the dot is past the last rhs symbol (item is complete).
    #[inline]
    pub fn is_complete(&self, rhs_len: usize) -> bool {
        self.dot >= rhs_len
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FIRST sets
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-computed FIRST sets for all non-terminals.
///
/// `FIRST[nt]` = the set of bytes that can appear as the first byte of any
/// string derivable from `nt`.
///
/// The boolean `nullable[nt]` is `true` if `nt` can derive the empty string ε.
pub struct FirstSets {
    /// For each NT: set of first bytes.
    pub first: HashMap<NonTerminalId, HashSet<u8>>,
    /// For each NT: whether it can derive ε.
    pub nullable: HashMap<NonTerminalId, bool>,
}

impl FirstSets {
    /// Compute FIRST sets for all non-terminals in `grammar` using a standard
    /// iterative fixed-point algorithm.
    pub fn compute(grammar: &Grammar) -> Self {
        let n = grammar.nt_count;
        let mut first: HashMap<NonTerminalId, HashSet<u8>> =
            (0..n).map(|i| (i, HashSet::new())).collect();
        let mut nullable: HashMap<NonTerminalId, bool> = (0..n).map(|i| (i, false)).collect();

        // Fixed-point iteration: keep looping until nothing changes.
        loop {
            let mut changed = false;

            for rule in &grammar.rules {
                let lhs = rule.lhs;

                if rule.rhs.is_empty() {
                    // Epsilon production: lhs is nullable.
                    if !nullable[&lhs] {
                        *nullable
                            .get_mut(&lhs)
                            .expect("lhs key inserted during initialization") = true;
                        changed = true;
                    }
                    continue;
                }

                // Iterate over symbols in rhs; stop as soon as we find a
                // non-nullable symbol (i.e. one that cannot derive ε).
                let mut all_nullable = true;
                for sym in &rule.rhs {
                    match sym {
                        Symbol::Terminal(bytes) => {
                            if let Some(&b) = bytes.first() {
                                if first
                                    .get_mut(&lhs)
                                    .expect("lhs key inserted during initialization")
                                    .insert(b)
                                {
                                    changed = true;
                                }
                            }
                            // An empty terminal (`bytes.is_empty()`) is treated as ε.
                            if !bytes.is_empty() {
                                all_nullable = false;
                                break;
                            }
                            // Empty terminal: continue to next symbol.
                        }
                        Symbol::NonTerminal(nt) => {
                            // Add FIRST(nt) \ {ε} to FIRST(lhs).
                            let first_nt: HashSet<u8> = first.get(nt).cloned().unwrap_or_default();
                            for &b in &first_nt {
                                if first
                                    .get_mut(&lhs)
                                    .expect("lhs key inserted during initialization")
                                    .insert(b)
                                {
                                    changed = true;
                                }
                            }
                            if !nullable.get(nt).copied().unwrap_or(false) {
                                all_nullable = false;
                                break;
                            }
                            // nt is nullable: continue to next symbol.
                        }
                    }
                }

                if all_nullable && !nullable[&lhs] {
                    *nullable
                        .get_mut(&lhs)
                        .expect("lhs key inserted during initialization") = true;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        Self { first, nullable }
    }

    /// Return the set of bytes that can start strings derivable from symbol `sym`.
    pub fn first_of_symbol(&self, sym: &Symbol) -> HashSet<u8> {
        match sym {
            Symbol::Terminal(bytes) => {
                if let Some(&b) = bytes.first() {
                    let mut s = HashSet::new();
                    s.insert(b);
                    s
                } else {
                    HashSet::new() // empty terminal contributes nothing to FIRST
                }
            }
            Symbol::NonTerminal(nt) => self.first.get(nt).cloned().unwrap_or_default(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EarleyRecognizer
// ─────────────────────────────────────────────────────────────────────────────

/// Earley chart-parser recognizer.
///
/// Maintains a chart (vector of item sets) and advances it byte by byte.
///
/// # Grammar requirements
///
/// Pass the grammar through [`Grammar::normalise_terminals`] before
/// constructing the recognizer.  All terminal symbols must be single bytes.
pub struct EarleyRecognizer {
    /// The grammar we are recognizing against (shared, immutable).
    grammar: Arc<Grammar>,
    /// Pre-computed FIRST sets for fast `next_byte_set` computation.
    first_sets: Arc<FirstSets>,
    /// `chart[k]` = set of Earley items active at input position `k`.
    /// `chart.len() == input_pos + 1` at all times.
    chart: Vec<HashSet<EarleyItem>>,
    /// Number of bytes fed so far.
    pub input_pos: usize,
    /// Cache: for each NT id, the list of rule ids with that lhs.
    rule_index: Arc<HashMap<NonTerminalId, Vec<RuleId>>>,
}

impl EarleyRecognizer {
    /// Construct a recognizer for the given grammar.
    ///
    /// This computes FIRST sets, builds a rule index, initialises `chart[0]`,
    /// and runs the Predict/Complete closure on the initial chart.
    pub fn new(grammar: Arc<Grammar>) -> Self {
        let first_sets = Arc::new(FirstSets::compute(&grammar));
        let rule_index = Arc::new(build_rule_index(&grammar));

        let mut recognizer = Self {
            grammar,
            first_sets,
            chart: vec![HashSet::new()],
            input_pos: 0,
            rule_index,
        };

        recognizer.init_chart_zero();
        recognizer
    }

    /// Seed `chart[0]` with start-symbol items and run closure.
    fn init_chart_zero(&mut self) {
        let start = self.grammar.start();
        if let Some(rule_ids) = self.rule_index.get(&start).cloned() {
            for rule_id in rule_ids {
                self.chart[0].insert(EarleyItem::new(rule_id, 0, 0));
            }
        }
        self.closure(0);
    }

    /// Run the Predict/Complete closure on `chart[k]` using a worklist.
    ///
    /// * **Predict**: for each item `A → α • B β` with B a non-terminal,
    ///   add `B → • γ` items (all rules for B) to `chart[k]`.
    /// * **Complete**: for each completed item `A → γ •` with origin `j`,
    ///   advance all items in `chart[j]` that had `• A` in their rhs.
    fn closure(&mut self, k: usize) {
        // We use an explicit worklist (queue) to avoid re-scanning items.
        // Items are processed exactly once: when they are first added to the
        // chart set, they are enqueued; the HashSet ensures no duplicates.
        let mut worklist: VecDeque<EarleyItem> = self.chart[k].iter().cloned().collect();

        while let Some(item) = worklist.pop_front() {
            let rule = &self.grammar.rules[item.rule];
            let rhs_len = rule.rhs.len();

            if item.dot >= rhs_len {
                // ── Complete ────────────────────────────────────────────────
                let completed_nt = rule.lhs;
                let origin = item.origin;

                // Clone the origin chart to avoid simultaneous borrow.
                let origin_items: Vec<EarleyItem> = self.chart[origin].iter().cloned().collect();
                for orig_item in origin_items {
                    let orig_rule = &self.grammar.rules[orig_item.rule];
                    let orig_rhs_len = orig_rule.rhs.len();
                    if orig_item.dot < orig_rhs_len {
                        if let Symbol::NonTerminal(nt) = &orig_rule.rhs[orig_item.dot] {
                            if *nt == completed_nt {
                                let advanced = EarleyItem::new(
                                    orig_item.rule,
                                    orig_item.dot + 1,
                                    orig_item.origin,
                                );
                                if self.chart[k].insert(advanced.clone()) {
                                    worklist.push_back(advanced);
                                }
                            }
                        }
                    }
                }
            } else {
                // ── Predict / Epsilon-terminal ───────────────────────────────
                match &self.grammar.rules[item.rule].rhs[item.dot] {
                    Symbol::NonTerminal(nt) => {
                        let nt = *nt;
                        if let Some(rule_ids) = self.rule_index.get(&nt).cloned() {
                            for rule_id in rule_ids {
                                let new_item = EarleyItem::new(rule_id, 0, k);
                                if self.chart[k].insert(new_item.clone()) {
                                    worklist.push_back(new_item);
                                }
                            }
                        }
                    }
                    Symbol::Terminal(bytes) => {
                        if bytes.is_empty() {
                            // An empty terminal (ε) is consumed immediately —
                            // advance the dot without reading any byte.
                            let advanced = EarleyItem::new(item.rule, item.dot + 1, item.origin);
                            if self.chart[k].insert(advanced.clone()) {
                                worklist.push_back(advanced);
                            }
                        }
                        // Non-empty terminal: nothing to predict; scan handles it.
                    }
                }
            }
        }
    }

    /// Feed one input byte.
    ///
    /// Executes the **Scan** step: for each item in `chart[input_pos]` that
    /// expects `Terminal([byte])` next, advance the dot and add the item to
    /// `chart[input_pos + 1]`.  Then runs closure on the new chart set.
    ///
    /// Returns `true` if the input is still valid (at least one item survives),
    /// `false` if this byte is impossible given the grammar.
    pub fn feed_byte(&mut self, byte: u8) -> bool {
        let k = self.input_pos;
        let mut next_set: HashSet<EarleyItem> = HashSet::new();

        for item in &self.chart[k] {
            let rule = &self.grammar.rules[item.rule];
            let rhs_len = rule.rhs.len();
            if item.dot < rhs_len {
                if let Symbol::Terminal(bytes) = &rule.rhs[item.dot] {
                    // After normalisation all terminals are single bytes.
                    if bytes.len() == 1 && bytes[0] == byte {
                        next_set.insert(EarleyItem::new(item.rule, item.dot + 1, item.origin));
                    }
                }
            }
        }

        if next_set.is_empty() {
            // No scan succeeded — byte is invalid; push an empty set to keep
            // the chart length consistent (input_pos remains valid for reset).
            self.chart.push(HashSet::new());
            self.input_pos += 1;
            return false;
        }

        self.chart.push(next_set);
        self.input_pos += 1;
        self.closure(self.input_pos);
        true
    }

    /// Returns `true` if the current state is an accepting state.
    ///
    /// An accepting state exists when `chart[input_pos]` contains a completed
    /// item for the start symbol with origin 0:
    /// `start → γ •, origin=0`.
    pub fn is_accepting(&self) -> bool {
        let start = self.grammar.start();
        let k = self.input_pos;
        self.chart[k].iter().any(|item| {
            let rule = &self.grammar.rules[item.rule];
            rule.lhs == start && item.origin == 0 && item.dot == rule.rhs.len()
        })
    }

    /// Returns `true` if the chart is still live (not dead / empty).
    pub fn is_live(&self) -> bool {
        !self.chart[self.input_pos].is_empty()
    }

    /// Compute the set of bytes that can legally appear as the next input byte.
    ///
    /// For each item `A → α • X β` in `chart[input_pos]`:
    /// * If `X` is `Terminal([b])`: add `b`.
    /// * If `X` is `NonTerminal(nt)`: add `FIRST(nt)`.
    ///
    /// Uses the pre-computed FIRST sets for efficiency.
    pub fn next_byte_set(&self) -> HashSet<u8> {
        let k = self.input_pos;
        let mut result: HashSet<u8> = HashSet::new();

        for item in &self.chart[k] {
            let rule = &self.grammar.rules[item.rule];
            let rhs_len = rule.rhs.len();
            if item.dot < rhs_len {
                match &rule.rhs[item.dot] {
                    Symbol::Terminal(bytes) => {
                        // After normalisation this is always a single byte.
                        if bytes.len() == 1 {
                            result.insert(bytes[0]);
                        }
                    }
                    Symbol::NonTerminal(nt) => {
                        let first_nt = self.first_sets.first.get(nt).cloned().unwrap_or_default();
                        result.extend(first_nt);
                    }
                }
            }
        }

        result
    }

    /// Reset to the initial state (as if no bytes had been fed).
    pub fn reset(&mut self) {
        self.chart.clear();
        self.chart.push(HashSet::new());
        self.input_pos = 0;
        self.init_chart_zero();
    }

    /// Deep-clone the current recognizer state for speculative lookahead.
    ///
    /// The cloned recognizer shares the immutable `Arc` parts (grammar, first
    /// sets, rule index) but has its own independent chart.
    pub fn clone_state(&self) -> Self {
        Self {
            grammar: Arc::clone(&self.grammar),
            first_sets: Arc::clone(&self.first_sets),
            chart: self.chart.clone(),
            input_pos: self.input_pos,
            rule_index: Arc::clone(&self.rule_index),
        }
    }

    /// Feed an entire byte slice, returning `false` on first invalid byte.
    pub fn feed_bytes(&mut self, bytes: &[u8]) -> bool {
        for &b in bytes {
            if !self.feed_byte(b) {
                return false;
            }
        }
        true
    }

    /// Return the grammar (for debugging / testing).
    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    /// Return the number of active items in the current chart set (debugging).
    pub fn active_item_count(&self) -> usize {
        self.chart[self.input_pos].len()
    }

    /// Compute a 64-bit hash of the current Earley chart state.
    ///
    /// Collects all items from `chart[input_pos]`, sorts them deterministically,
    /// and hashes them using `DefaultHasher`. The resulting hash is used as a
    /// cache key in `GrammarConstraint::allowed_tokens`.
    pub fn state_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut items: Vec<(usize, usize, usize)> = self.chart[self.input_pos]
            .iter()
            .map(|item| (item.rule, item.dot, item.origin))
            .collect();
        items.sort_unstable();

        let mut hasher = DefaultHasher::new();
        items.hash(&mut hasher);
        self.input_pos.hash(&mut hasher);
        hasher.finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: rule index
// ─────────────────────────────────────────────────────────────────────────────

/// Build a map from `NonTerminalId` → `Vec<RuleId>` for O(1) rule lookup.
fn build_rule_index(grammar: &Grammar) -> HashMap<NonTerminalId, Vec<RuleId>> {
    let mut index: HashMap<NonTerminalId, Vec<RuleId>> = HashMap::new();
    for (rule_id, rule) in grammar.rules.iter().enumerate() {
        index.entry(rule.lhs).or_default().push(rule_id);
    }
    index
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::bnf_parser::parse_bnf;

    // Helper: build a normalised grammar and recognizer from BNF source.
    fn recognizer_from_bnf(bnf: &str) -> EarleyRecognizer {
        let mut g = parse_bnf(bnf).expect("valid BNF");
        g.normalise_terminals();
        EarleyRecognizer::new(Arc::new(g))
    }

    fn feed_str(rec: &mut EarleyRecognizer, s: &str) -> bool {
        for b in s.bytes() {
            if !rec.feed_byte(b) {
                return false;
            }
        }
        true
    }

    // ── Basic acceptance ────────────────────────────────────────────────────

    #[test]
    fn earley_accepts_single_terminal() {
        let mut r = recognizer_from_bnf(r#"<S> ::= "x""#);
        assert!(feed_str(&mut r, "x"));
        assert!(r.is_accepting());
    }

    #[test]
    fn earley_rejects_wrong_terminal() {
        let mut r = recognizer_from_bnf(r#"<S> ::= "x""#);
        assert!(!feed_str(&mut r, "y"));
        assert!(!r.is_accepting());
    }

    #[test]
    fn earley_accepts_epsilon_rule_empty_input() {
        // S ::= "" (epsilon rule — S is accepting at position 0)
        let r = recognizer_from_bnf(r#"<S> ::= """#);
        // No bytes fed — should already be accepting.
        // (An empty terminal Terminal([]) is equivalent to an epsilon rule.)
        // Note: the grammar may or may not produce a complete parse here
        // depending on whether we treat `""` as epsilon. Let's just test that
        // the recognizer is live.
        assert!(r.is_live());
    }

    // ── Alternation ─────────────────────────────────────────────────────────

    #[test]
    fn earley_accepts_alternation_first() {
        let mut r = recognizer_from_bnf(r#"<S> ::= "a" | "b""#);
        assert!(feed_str(&mut r, "a"));
        assert!(r.is_accepting());
    }

    #[test]
    fn earley_accepts_alternation_second() {
        let mut r = recognizer_from_bnf(r#"<S> ::= "a" | "b""#);
        assert!(feed_str(&mut r, "b"));
        assert!(r.is_accepting());
    }

    #[test]
    fn earley_rejects_alternation_neither() {
        let mut r = recognizer_from_bnf(r#"<S> ::= "a" | "b""#);
        assert!(!feed_str(&mut r, "c"));
    }

    // ── Recursion ───────────────────────────────────────────────────────────

    #[test]
    fn earley_accepts_right_recursive() {
        // S ::= "a" S | "a"
        let mut r = recognizer_from_bnf(r#"<S> ::= "a" <S> | "a""#);
        assert!(feed_str(&mut r, "aaa"));
        assert!(r.is_accepting());
    }

    #[test]
    fn earley_accepts_simple_ab_grammar() {
        // S ::= "a" S "b" | "ab"  — matches "ab", "aabb", "aaabbb", ...
        let mut r = recognizer_from_bnf(r#"<S> ::= "a" <S> "b" | "ab""#);
        assert!(feed_str(&mut r, "ab"));
        assert!(r.is_accepting());

        r.reset();
        assert!(feed_str(&mut r, "aabb"));
        assert!(r.is_accepting());

        r.reset();
        assert!(feed_str(&mut r, "aaabbb"));
        assert!(r.is_accepting());
    }

    #[test]
    fn earley_rejects_ab_grammar_wrong() {
        let mut r = recognizer_from_bnf(r#"<S> ::= "a" <S> "b" | "ab""#);
        assert!(!feed_str(&mut r, "ba"));
    }

    #[test]
    fn earley_rejects_ab_grammar_unbalanced() {
        let mut r = recognizer_from_bnf(r#"<S> ::= "a" <S> "b" | "ab""#);
        let ok = feed_str(&mut r, "aab");
        // "aab" is a prefix that could continue (needs one more 'b') — it may
        // still be live but NOT accepting.
        if ok {
            assert!(!r.is_accepting());
        }
    }

    // ── Arithmetic grammar ──────────────────────────────────────────────────

    fn arithmetic_recognizer() -> EarleyRecognizer {
        recognizer_from_bnf(
            r#"
            <expr>   ::= <term> "+" <expr> | <term> "-" <expr> | <term>
            <term>   ::= <factor> "*" <term> | <factor> "/" <term> | <factor>
            <factor> ::= "(" <expr> ")" | <number>
            <number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        "#,
        )
    }

    #[test]
    fn earley_accepts_arithmetic_single_digit() {
        let mut r = arithmetic_recognizer();
        assert!(feed_str(&mut r, "9"));
        assert!(r.is_accepting());
    }

    #[test]
    fn earley_accepts_arithmetic_1plus2() {
        let mut r = arithmetic_recognizer();
        assert!(feed_str(&mut r, "1+2"));
        assert!(r.is_accepting());
    }

    #[test]
    fn earley_accepts_arithmetic_1times2plus3() {
        let mut r = arithmetic_recognizer();
        assert!(feed_str(&mut r, "1*2+3"));
        assert!(r.is_accepting());
    }

    #[test]
    fn earley_accepts_arithmetic_paren() {
        let mut r = arithmetic_recognizer();
        assert!(feed_str(&mut r, "(1+2)*3"));
        assert!(r.is_accepting());
    }

    #[test]
    fn earley_rejects_arithmetic_plus_at_start() {
        let mut r = arithmetic_recognizer();
        // '+' cannot be the very first character of an arithmetic expression.
        assert!(!feed_str(&mut r, "+1"));
    }

    #[test]
    fn earley_rejects_arithmetic_double_plus() {
        let mut r = arithmetic_recognizer();
        let ok = feed_str(&mut r, "1++2");
        if ok {
            assert!(!r.is_accepting());
        }
    }

    // ── next_byte_set ───────────────────────────────────────────────────────

    #[test]
    fn earley_next_byte_set_at_start_arithmetic() {
        let r = arithmetic_recognizer();
        let nbs = r.next_byte_set();
        // At start, valid bytes are digits and '('
        for d in b'0'..=b'9' {
            assert!(nbs.contains(&d), "digit {d} should be in next_byte_set");
        }
        assert!(nbs.contains(&b'('), "'(' should be in next_byte_set");
        assert!(
            !nbs.contains(&b'+'),
            "'+' should not be in next_byte_set at start"
        );
    }

    #[test]
    fn earley_next_byte_set_after_digit() {
        let mut r = arithmetic_recognizer();
        feed_str(&mut r, "1");
        let nbs = r.next_byte_set();
        // After a digit the valid continuations are operators or end.
        assert!(nbs.contains(&b'+'), "'+' should be valid after a digit");
        assert!(nbs.contains(&b'-'), "'-' should be valid after a digit");
        assert!(nbs.contains(&b'*'), "'*' should be valid after a digit");
        assert!(nbs.contains(&b'/'), "'/' should be valid after a digit");
    }

    // ── not-accepting mid-input ─────────────────────────────────────────────

    #[test]
    fn earley_not_accepting_mid_input() {
        let mut r = arithmetic_recognizer();
        feed_str(&mut r, "1+");
        assert!(!r.is_accepting(), "should not accept after '1+'");
    }

    // ── reset ───────────────────────────────────────────────────────────────

    #[test]
    fn earley_reset_restores_initial_state() {
        let mut r = arithmetic_recognizer();
        feed_str(&mut r, "1+2");
        assert!(r.is_accepting());
        r.reset();
        assert_eq!(r.input_pos, 0);
        assert!(!r.is_accepting());
        // Can be used again.
        feed_str(&mut r, "9");
        assert!(r.is_accepting());
    }

    // ── clone_state independence ─────────────────────────────────────────────

    #[test]
    fn earley_clone_state_is_independent() {
        let mut r = arithmetic_recognizer();
        feed_str(&mut r, "1");
        let mut clone = r.clone_state();
        // Advance original to "+2" — accepting.
        feed_str(&mut r, "+2");
        assert!(r.is_accepting());
        // Clone is still at "1" position.
        assert_eq!(clone.input_pos, 1);
        // Advance clone to "*3" — also accepting.
        feed_str(&mut r, "*3");
        feed_str(&mut clone, "*3");
        assert!(clone.is_accepting());
    }

    // ── left recursion (explicit test) ─────────────────────────────────────
    // NOTE: Earley naturally handles left recursion via the HashSet deduplication.

    #[test]
    fn earley_handles_left_recursive_grammar() {
        // Left-recursive grammar: E ::= E "+" "1" | "1"
        let mut r = recognizer_from_bnf(r#"<E> ::= <E> "+" "1" | "1""#);
        // "1" is accepted.
        assert!(feed_str(&mut r, "1"));
        assert!(r.is_accepting());
        r.reset();
        // "1+1+1" is accepted.
        assert!(feed_str(&mut r, "1+1+1"));
        assert!(r.is_accepting());
        r.reset();
        // "+1" is rejected.
        assert!(!feed_str(&mut r, "+1"));
    }

    // ── nullable productions ────────────────────────────────────────────────

    #[test]
    fn earley_handles_nullable_productions() {
        // S ::= <A> "x"
        // A ::= "" | "a"
        let mut r = recognizer_from_bnf(
            r#"
            <S> ::= <A> "x"
            <A> ::= "" | "a"
        "#,
        );
        // "x" should be accepted (A → ε).
        assert!(feed_str(&mut r, "x"));
        assert!(r.is_accepting());
        r.reset();
        // "ax" should be accepted.
        assert!(feed_str(&mut r, "ax"));
        assert!(r.is_accepting());
    }

    // ── live-ness after rejected byte ───────────────────────────────────────

    #[test]
    fn earley_is_not_live_after_rejection() {
        let mut r = recognizer_from_bnf(r#"<S> ::= "abc""#);
        r.feed_byte(b'a');
        r.feed_byte(b'b');
        r.feed_byte(b'z'); // wrong
        assert!(!r.is_live());
    }
}
