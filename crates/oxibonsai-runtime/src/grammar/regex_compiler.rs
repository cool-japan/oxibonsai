//! Regex → BNF Grammar compiler.
//!
//! Compiles a regular expression pattern string into a [`Grammar`] that is
//! usable with [`super::constraint::GrammarConstraint`] for constrained decoding.
//!
//! # Algorithm
//!
//! 1. **Regex parser → Thompson NFA**: Builds a Nondeterministic Finite
//!    Automaton via Thompson construction from the parsed regex AST.
//! 2. **Subset DFA construction**: Converts the NFA to a DFA via powerset
//!    construction (ε-closure + transition computation).
//! 3. **DFA → Grammar**: Each DFA state becomes a non-terminal; transitions
//!    become single-byte terminal rules; accept states emit ε-productions.
//!
//! # Supported regex features
//!
//! - Literals: any byte literal
//! - `.` — any byte except `\n` (0x0A)
//! - `[abc]`, `[a-z]`, `[^abc]` — character classes
//! - `*`, `+`, `?` — greedy quantifiers
//! - `{n}`, `{n,}`, `{n,m}` — counted quantifiers
//! - `|` — alternation
//! - `(...)` — grouping (non-capturing)
//! - Anchors: `^` (start) and `$` (end) are silently ignored
//! - Escape sequences: `\d`, `\w`, `\s`, `\D`, `\W`, `\S`, `\n`, `\r`, `\t`,
//!   `\.`, `\\`, `\[`, `\]`, `\(`, `\)`, `\*`, `\+`, `\?`, `\{`, `\}`, `\|`
//!
//! # Unsupported (returns `RegexCompileError::UnsupportedFeature`)
//!
//! - Backreferences `\1`, `\2`, …
//! - Lookahead/lookbehind: `(?=...)`, `(?!...)`, `(?<=...)`, `(?<!...)`
//! - Named groups: `(?P<name>...)`, `(?<name>...)`
//! - Atomic groups, possessive quantifiers
//! - Unicode properties `\p{Letter}`

use std::collections::{BTreeSet, HashMap, VecDeque};

use super::ast::{Grammar, NonTerminalId, Rule, Symbol};

// ─────────────────────────────────────────────────────────────────────────────
// Public error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising from compiling a regex pattern into a Grammar.
#[derive(Debug, Clone, PartialEq)]
pub enum RegexCompileError {
    /// The regex pattern has a syntax error.
    InvalidSyntax(String),
    /// The pattern uses a feature not supported by this compiler.
    UnsupportedFeature(String),
    /// The DFA state count or NFA expansion exceeded an internal limit.
    DepthExceeded {
        /// The exceeded limit.
        limit: usize,
    },
    /// The pattern is the empty string.
    EmptyPattern,
    /// The pattern contains invalid UTF-8 byte sequences where UTF-8 is required.
    InvalidUtf8(String),
}

impl std::fmt::Display for RegexCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSyntax(msg) => write!(f, "regex syntax error: {msg}"),
            Self::UnsupportedFeature(feat) => {
                write!(f, "unsupported regex feature: {feat}")
            }
            Self::DepthExceeded { limit } => {
                write!(f, "regex complexity limit exceeded (limit: {limit})")
            }
            Self::EmptyPattern => write!(f, "regex pattern is empty"),
            Self::InvalidUtf8(msg) => write!(f, "invalid UTF-8 in regex pattern: {msg}"),
        }
    }
}

impl std::error::Error for RegexCompileError {}

// ─────────────────────────────────────────────────────────────────────────────
// ByteSet — 256-bit bitset for byte ranges
// ─────────────────────────────────────────────────────────────────────────────

/// A dense bitset covering all 256 possible byte values.
///
/// Stored as 4 × u64 words (256 bits total).  Bit `b` in word `b >> 6` at
/// position `b & 63`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct ByteSet([u64; 4]);

impl ByteSet {
    /// Empty set (no bytes).
    fn empty() -> Self {
        Self([0u64; 4])
    }

    /// Full set (all 256 bytes).
    fn full() -> Self {
        Self([u64::MAX; 4])
    }

    /// Set containing all bytes except `\n` (0x0A). Used for `.`.
    fn any_except_newline() -> Self {
        let mut s = Self::full();
        s.remove(b'\n');
        s
    }

    /// Set a single byte.
    fn insert(&mut self, b: u8) {
        let word = (b >> 6) as usize;
        let bit = b & 63;
        self.0[word] |= 1u64 << bit;
    }

    /// Remove a single byte.
    fn remove(&mut self, b: u8) {
        let word = (b >> 6) as usize;
        let bit = b & 63;
        self.0[word] &= !(1u64 << bit);
    }

    /// Test whether byte `b` is in this set.
    fn contains(&self, b: u8) -> bool {
        let word = (b >> 6) as usize;
        let bit = b & 63;
        self.0[word] & (1u64 << bit) != 0
    }

    /// Boolean complement of this set (all bytes NOT in self).
    fn complement(&self) -> Self {
        Self([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }

    /// Union of two byte sets.
    fn union(&self, other: &Self) -> Self {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    /// Iterate over all bytes in the set.
    fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0u8..=255u8).filter(|&b| self.contains(b))
    }

    /// Return true if the set is empty.
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.0 == [0u64; 4]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NFA representation (Thompson construction)
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of NFA states allowed (guards against pathological patterns).
const MAX_NFA_STATES: usize = 16_384;

/// Maximum number of DFA states allowed.
const MAX_DFA_STATES: usize = 2048;

/// Maximum repetition count expansion limit.
const MAX_REPETITION: usize = 64;

/// One state in the NFA.
#[derive(Debug, Clone)]
struct NfaState {
    /// Labeled (byte-set) transitions: (label, target_state_id).
    transitions: Vec<(ByteSet, usize)>,
    /// Epsilon transitions to other state ids.
    epsilon: Vec<usize>,
    /// Whether this is an accept state.
    is_accept: bool,
}

impl NfaState {
    fn new() -> Self {
        Self {
            transitions: Vec::new(),
            epsilon: Vec::new(),
            is_accept: false,
        }
    }
}

/// An NFA fragment returned by Thompson construction sub-routines.
/// `start` is the entry state id; `end` is the single accepting state id.
struct NfaFrag {
    start: usize,
    end: usize,
}

/// The full NFA builder — holds all states and a counter for fresh ids.
struct Nfa {
    states: Vec<NfaState>,
}

impl Nfa {
    fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Allocate a fresh NFA state and return its id.
    fn alloc(&mut self) -> Result<usize, RegexCompileError> {
        if self.states.len() >= MAX_NFA_STATES {
            return Err(RegexCompileError::DepthExceeded {
                limit: MAX_NFA_STATES,
            });
        }
        let id = self.states.len();
        self.states.push(NfaState::new());
        Ok(id)
    }

    /// Add an epsilon transition from `from` → `to`.
    fn add_epsilon(&mut self, from: usize, to: usize) {
        self.states[from].epsilon.push(to);
    }

    /// Add a labeled transition from `from` →[label]→ `to`.
    fn add_transition(&mut self, from: usize, label: ByteSet, to: usize) {
        self.states[from].transitions.push((label, to));
    }

    /// Compute the ε-closure of a set of NFA states.
    fn epsilon_closure(&self, seeds: impl IntoIterator<Item = usize>) -> BTreeSet<usize> {
        let mut closure: BTreeSet<usize> = BTreeSet::new();
        let mut worklist: VecDeque<usize> = VecDeque::new();

        for s in seeds {
            if closure.insert(s) {
                worklist.push_back(s);
            }
        }

        while let Some(state) = worklist.pop_front() {
            for &target in &self.states[state].epsilon {
                if closure.insert(target) {
                    worklist.push_back(target);
                }
            }
        }

        closure
    }

    /// Build Thompson fragment for a single ByteSet (character class or literal).
    fn build_byte_set(&mut self, label: ByteSet) -> Result<NfaFrag, RegexCompileError> {
        let start = self.alloc()?;
        let end = self.alloc()?;
        self.add_transition(start, label, end);
        Ok(NfaFrag { start, end })
    }

    /// Concatenate two fragments: `a` · `b`.
    fn build_concat(&mut self, a: NfaFrag, b: NfaFrag) -> NfaFrag {
        // Connect end(a) to start(b) via epsilon.
        self.add_epsilon(a.end, b.start);
        NfaFrag {
            start: a.start,
            end: b.end,
        }
    }

    /// Alternation: `a | b`.
    fn build_alternation(
        &mut self,
        a: NfaFrag,
        b: NfaFrag,
    ) -> Result<NfaFrag, RegexCompileError> {
        let start = self.alloc()?;
        let end = self.alloc()?;
        self.add_epsilon(start, a.start);
        self.add_epsilon(start, b.start);
        self.add_epsilon(a.end, end);
        self.add_epsilon(b.end, end);
        Ok(NfaFrag { start, end })
    }

    /// Kleene star: `a*`.
    fn build_star(&mut self, a: NfaFrag) -> Result<NfaFrag, RegexCompileError> {
        let start = self.alloc()?;
        let end = self.alloc()?;
        // start → a.start (enter loop)
        self.add_epsilon(start, a.start);
        // start → end (skip entirely)
        self.add_epsilon(start, end);
        // a.end → a.start (repeat)
        self.add_epsilon(a.end, a.start);
        // a.end → end (exit)
        self.add_epsilon(a.end, end);
        Ok(NfaFrag { start, end })
    }

    /// Plus: `a+` = `a · a*`.
    fn build_plus(&mut self, a: NfaFrag) -> Result<NfaFrag, RegexCompileError> {
        // We need two copies; just wire up manually: start→a.start, a.end loops.
        let loop_start = self.alloc()?;
        let loop_end = self.alloc()?;
        // After completing `a` once, we can repeat or exit.
        self.add_epsilon(a.end, loop_start);
        // loop_start → a.start (repeat)
        self.add_epsilon(loop_start, a.start);
        // loop_start → loop_end (exit)
        self.add_epsilon(loop_start, loop_end);
        Ok(NfaFrag {
            start: a.start,
            end: loop_end,
        })
    }

    /// Optional: `a?`.
    fn build_optional(&mut self, a: NfaFrag) -> Result<NfaFrag, RegexCompileError> {
        let start = self.alloc()?;
        let end = self.alloc()?;
        self.add_epsilon(start, a.start);
        self.add_epsilon(start, end);
        self.add_epsilon(a.end, end);
        Ok(NfaFrag { start, end })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Regex AST
// ─────────────────────────────────────────────────────────────────────────────

/// Internal regex AST node.
#[derive(Debug, Clone)]
enum RegexNode {
    /// Matches a set of bytes (single step).
    ByteClass(ByteSet),
    /// Concatenation of sub-expressions.
    Concat(Vec<RegexNode>),
    /// Alternation of sub-expressions.
    Alternation(Vec<RegexNode>),
    /// Kleene star: zero or more repetitions.
    Star(Box<RegexNode>),
    /// Plus: one or more repetitions.
    Plus(Box<RegexNode>),
    /// Optional: zero or one repetition.
    Optional(Box<RegexNode>),
    /// Counted exact: exactly `n` repetitions.
    CountedExact(Box<RegexNode>, usize),
    /// Counted range: `n` to `m` repetitions (or unbounded if m = None → `n,`).
    CountedRange(Box<RegexNode>, usize, Option<usize>),
    /// Empty: matches the empty string (useful as a base case).
    Empty,
}

// ─────────────────────────────────────────────────────────────────────────────
// Regex parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parser state for the regex string.
struct RegexParser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> RegexParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.input.get(self.pos).copied()?;
        self.pos += 1;
        Some(b)
    }

    fn expect(&mut self, expected: u8) -> Result<(), RegexCompileError> {
        match self.peek() {
            Some(b) if b == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(b) => Err(RegexCompileError::InvalidSyntax(format!(
                "expected '{}' at position {}, got '{}'",
                expected as char, self.pos, b as char
            ))),
            None => Err(RegexCompileError::InvalidSyntax(format!(
                "expected '{}' at position {} but got end of pattern",
                expected as char, self.pos
            ))),
        }
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.input.len()
    }

    /// Top-level: parse an alternation expression.
    fn parse_alternation(&mut self) -> Result<RegexNode, RegexCompileError> {
        let mut branches: Vec<RegexNode> = Vec::new();
        branches.push(self.parse_concat()?);
        while self.peek() == Some(b'|') {
            self.pos += 1; // consume '|'
            branches.push(self.parse_concat()?);
        }
        if branches.len() == 1 {
            Ok(branches.remove(0))
        } else {
            Ok(RegexNode::Alternation(branches))
        }
    }

    /// Parse a concatenation of atoms.
    fn parse_concat(&mut self) -> Result<RegexNode, RegexCompileError> {
        let mut atoms: Vec<RegexNode> = Vec::new();
        loop {
            match self.peek() {
                None | Some(b')') | Some(b'|') => break,
                _ => {
                    let atom = self.parse_quantified_atom()?;
                    match atom {
                        RegexNode::Empty => {}
                        other => atoms.push(other),
                    }
                }
            }
        }
        if atoms.is_empty() {
            Ok(RegexNode::Empty)
        } else if atoms.len() == 1 {
            Ok(atoms.remove(0))
        } else {
            Ok(RegexNode::Concat(atoms))
        }
    }

    /// Parse an atom followed by an optional quantifier.
    fn parse_quantified_atom(&mut self) -> Result<RegexNode, RegexCompileError> {
        let atom = self.parse_atom()?;
        match self.peek() {
            Some(b'*') => {
                self.pos += 1;
                Ok(RegexNode::Star(Box::new(atom)))
            }
            Some(b'+') => {
                self.pos += 1;
                Ok(RegexNode::Plus(Box::new(atom)))
            }
            Some(b'?') => {
                self.pos += 1;
                Ok(RegexNode::Optional(Box::new(atom)))
            }
            Some(b'{') => self.parse_counted_quantifier(atom),
            _ => Ok(atom),
        }
    }

    /// Parse `{n}`, `{n,}`, or `{n,m}` quantifier.
    fn parse_counted_quantifier(
        &mut self,
        atom: RegexNode,
    ) -> Result<RegexNode, RegexCompileError> {
        self.pos += 1; // consume '{'
        let n = self.parse_decimal_number()?;
        match self.peek() {
            Some(b'}') => {
                self.pos += 1;
                if n > MAX_REPETITION {
                    return Err(RegexCompileError::DepthExceeded {
                        limit: MAX_REPETITION,
                    });
                }
                Ok(RegexNode::CountedExact(Box::new(atom), n))
            }
            Some(b',') => {
                self.pos += 1; // consume ','
                match self.peek() {
                    Some(b'}') => {
                        self.pos += 1;
                        if n > MAX_REPETITION {
                            return Err(RegexCompileError::DepthExceeded {
                                limit: MAX_REPETITION,
                            });
                        }
                        Ok(RegexNode::CountedRange(Box::new(atom), n, None))
                    }
                    _ => {
                        let m = self.parse_decimal_number()?;
                        self.expect(b'}')?;
                        if n > m {
                            return Err(RegexCompileError::InvalidSyntax(format!(
                                "{{n,m}} quantifier has n={n} > m={m}"
                            )));
                        }
                        if m > MAX_REPETITION {
                            return Err(RegexCompileError::DepthExceeded {
                                limit: MAX_REPETITION,
                            });
                        }
                        Ok(RegexNode::CountedRange(Box::new(atom), n, Some(m)))
                    }
                }
            }
            Some(other) => Err(RegexCompileError::InvalidSyntax(format!(
                "unexpected '{other}' inside {{}} quantifier at position {}",
                self.pos
            ))),
            None => Err(RegexCompileError::InvalidSyntax(
                "unterminated '{' quantifier".to_string(),
            )),
        }
    }

    /// Parse a decimal integer.
    fn parse_decimal_number(&mut self) -> Result<usize, RegexCompileError> {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(RegexCompileError::InvalidSyntax(format!(
                "expected decimal number at position {start}"
            )));
        }
        let digits = &self.input[start..self.pos];
        // SAFETY: we verified all bytes are ASCII digits.
        let s = std::str::from_utf8(digits).map_err(|e| {
            RegexCompileError::InvalidUtf8(format!("non-UTF8 in decimal: {e}"))
        })?;
        s.parse::<usize>().map_err(|e| {
            RegexCompileError::InvalidSyntax(format!("overflow in decimal number: {e}"))
        })
    }

    /// Parse a single atom: literal, escape, class, group, or anchor.
    fn parse_atom(&mut self) -> Result<RegexNode, RegexCompileError> {
        match self.peek() {
            Some(b'^') => {
                // Anchor: silently ignore.
                self.pos += 1;
                Ok(RegexNode::Empty)
            }
            Some(b'$') => {
                // Anchor: silently ignore.
                self.pos += 1;
                Ok(RegexNode::Empty)
            }
            Some(b'.') => {
                self.pos += 1;
                Ok(RegexNode::ByteClass(ByteSet::any_except_newline()))
            }
            Some(b'[') => {
                self.pos += 1;
                self.parse_char_class()
            }
            Some(b'(') => {
                self.pos += 1;
                self.parse_group()
            }
            Some(b'\\') => {
                self.pos += 1;
                self.parse_escape()
            }
            Some(b) => {
                self.pos += 1;
                let mut set = ByteSet::empty();
                set.insert(b);
                Ok(RegexNode::ByteClass(set))
            }
            None => Err(RegexCompileError::InvalidSyntax(
                "unexpected end of pattern in atom".to_string(),
            )),
        }
    }

    /// Parse a character class `[...]`.
    fn parse_char_class(&mut self) -> Result<RegexNode, RegexCompileError> {
        let mut set = ByteSet::empty();
        let negated = if self.peek() == Some(b'^') {
            self.pos += 1;
            true
        } else {
            false
        };

        // First char can be `]` without closing (treated as literal `]`).
        let mut first = true;
        loop {
            match self.peek() {
                None => {
                    return Err(RegexCompileError::InvalidSyntax(
                        "unterminated character class '['".to_string(),
                    ));
                }
                Some(b']') if !first => {
                    self.pos += 1;
                    break;
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let escaped_set = self.parse_escape_to_set()?;
                    set = set.union(&escaped_set);
                }
                Some(b) => {
                    self.pos += 1;
                    // Check for range `x-y`.
                    if self.peek() == Some(b'-') && self.input.get(self.pos + 1) != Some(&b']') {
                        self.pos += 1; // consume '-'
                        match self.peek() {
                            Some(end_b) => {
                                self.pos += 1;
                                if end_b < b {
                                    return Err(RegexCompileError::InvalidSyntax(format!(
                                        "character class range end '{end_b}' < start '{b}'"
                                    )));
                                }
                                for c in b..=end_b {
                                    set.insert(c);
                                }
                            }
                            None => {
                                return Err(RegexCompileError::InvalidSyntax(
                                    "unterminated character class range".to_string(),
                                ));
                            }
                        }
                    } else {
                        set.insert(b);
                    }
                }
            }
            first = false;
        }

        if negated {
            set = set.complement();
        }

        Ok(RegexNode::ByteClass(set))
    }

    /// Parse a group `(...)`.
    fn parse_group(&mut self) -> Result<RegexNode, RegexCompileError> {
        // Check for special group prefixes.
        if self.peek() == Some(b'?') {
            // Look ahead to determine the kind.
            match self.input.get(self.pos + 1) {
                Some(b'=') | Some(b'!') => {
                    return Err(RegexCompileError::UnsupportedFeature(
                        "lookahead assertions (?=...) and (?!...) are not supported".to_string(),
                    ));
                }
                Some(b'<') => {
                    // Could be lookbehind (?<=...) / (?<!...) or named group (?<name>...).
                    match self.input.get(self.pos + 2) {
                        Some(b'=') | Some(b'!') => {
                            return Err(RegexCompileError::UnsupportedFeature(
                                "lookbehind assertions (?<=...) and (?<!...) are not supported"
                                    .to_string(),
                            ));
                        }
                        _ => {
                            return Err(RegexCompileError::UnsupportedFeature(
                                "named groups (?<name>...) are not supported".to_string(),
                            ));
                        }
                    }
                }
                Some(b'P') => {
                    return Err(RegexCompileError::UnsupportedFeature(
                        "named groups (?P<name>...) are not supported".to_string(),
                    ));
                }
                Some(b':') => {
                    // Non-capturing group `(?:...)` — consume the `?:` prefix and proceed.
                    self.pos += 2;
                }
                _ => {
                    return Err(RegexCompileError::UnsupportedFeature(format!(
                        "unsupported group type starting with '(?{}' at position {}",
                        self.input
                            .get(self.pos + 1)
                            .map(|&b| b as char)
                            .unwrap_or('?'),
                        self.pos
                    )));
                }
            }
        }

        let inner = self.parse_alternation()?;
        self.expect(b')')?;
        Ok(inner)
    }

    /// Parse an escape sequence at the current position (after consuming `\`).
    fn parse_escape(&mut self) -> Result<RegexNode, RegexCompileError> {
        let set = self.parse_escape_to_set()?;
        Ok(RegexNode::ByteClass(set))
    }

    /// Parse an escape sequence and return its ByteSet.
    fn parse_escape_to_set(&mut self) -> Result<ByteSet, RegexCompileError> {
        match self.advance() {
            None => Err(RegexCompileError::InvalidSyntax(
                "trailing backslash in pattern".to_string(),
            )),
            Some(b'd') => Ok(digit_set()),
            Some(b'D') => Ok(digit_set().complement()),
            Some(b'w') => Ok(word_set()),
            Some(b'W') => Ok(word_set().complement()),
            Some(b's') => Ok(space_set()),
            Some(b'S') => Ok(space_set().complement()),
            Some(b'n') => {
                let mut s = ByteSet::empty();
                s.insert(b'\n');
                Ok(s)
            }
            Some(b'r') => {
                let mut s = ByteSet::empty();
                s.insert(b'\r');
                Ok(s)
            }
            Some(b't') => {
                let mut s = ByteSet::empty();
                s.insert(b'\t');
                Ok(s)
            }
            Some(b) if is_meta_escapable(b) => {
                let mut s = ByteSet::empty();
                s.insert(b);
                Ok(s)
            }
            Some(b'1'..=b'9') => Err(RegexCompileError::UnsupportedFeature(
                "backreferences (\\1, \\2, ...) are not supported".to_string(),
            )),
            Some(b'p') => Err(RegexCompileError::UnsupportedFeature(
                "Unicode properties (\\p{...}) are not supported".to_string(),
            )),
            Some(other) => Err(RegexCompileError::InvalidSyntax(format!(
                "unknown escape sequence '\\{}'",
                other as char
            ))),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Character class helpers
// ─────────────────────────────────────────────────────────────────────────────

/// `\d`: ASCII decimal digits 0–9.
fn digit_set() -> ByteSet {
    let mut s = ByteSet::empty();
    for b in b'0'..=b'9' {
        s.insert(b);
    }
    s
}

/// `\w`: word characters: `[A-Za-z0-9_]`.
fn word_set() -> ByteSet {
    let mut s = ByteSet::empty();
    for b in b'A'..=b'Z' {
        s.insert(b);
    }
    for b in b'a'..=b'z' {
        s.insert(b);
    }
    for b in b'0'..=b'9' {
        s.insert(b);
    }
    s.insert(b'_');
    s
}

/// `\s`: whitespace characters: space, `\t`, `\n`, `\r`, `\x0B`, `\x0C`.
fn space_set() -> ByteSet {
    let mut s = ByteSet::empty();
    s.insert(b' ');
    s.insert(b'\t');
    s.insert(b'\n');
    s.insert(b'\r');
    s.insert(0x0B); // vertical tab
    s.insert(0x0C); // form feed
    s
}

/// Return true if `b` is a metacharacter that may be escaped with `\`.
fn is_meta_escapable(b: u8) -> bool {
    matches!(
        b,
        b'.' | b'\\' | b'[' | b']' | b'(' | b')' | b'*' | b'+' | b'?' | b'{' | b'}' | b'|'
            | b'^' | b'$' | b'0'
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Thompson NFA construction from RegexNode
// ─────────────────────────────────────────────────────────────────────────────

/// Recursively build an NFA fragment for the given [`RegexNode`].
fn build_nfa_frag(nfa: &mut Nfa, node: &RegexNode) -> Result<NfaFrag, RegexCompileError> {
    match node {
        RegexNode::Empty => {
            // Empty match: single state that is both start and end.
            let s = nfa.alloc()?;
            Ok(NfaFrag { start: s, end: s })
        }
        RegexNode::ByteClass(set) => nfa.build_byte_set(set.clone()),
        RegexNode::Concat(nodes) => {
            if nodes.is_empty() {
                let s = nfa.alloc()?;
                return Ok(NfaFrag { start: s, end: s });
            }
            let mut frag = build_nfa_frag(nfa, &nodes[0])?;
            for node in &nodes[1..] {
                let next = build_nfa_frag(nfa, node)?;
                frag = nfa.build_concat(frag, next);
            }
            Ok(frag)
        }
        RegexNode::Alternation(nodes) => {
            if nodes.is_empty() {
                let s = nfa.alloc()?;
                return Ok(NfaFrag { start: s, end: s });
            }
            let mut frag = build_nfa_frag(nfa, &nodes[0])?;
            for node in &nodes[1..] {
                let next = build_nfa_frag(nfa, node)?;
                frag = nfa.build_alternation(frag, next)?;
            }
            Ok(frag)
        }
        RegexNode::Star(inner) => {
            let inner_frag = build_nfa_frag(nfa, inner)?;
            nfa.build_star(inner_frag)
        }
        RegexNode::Plus(inner) => {
            let inner_frag = build_nfa_frag(nfa, inner)?;
            nfa.build_plus(inner_frag)
        }
        RegexNode::Optional(inner) => {
            let inner_frag = build_nfa_frag(nfa, inner)?;
            nfa.build_optional(inner_frag)
        }
        RegexNode::CountedExact(inner, n) => {
            // Expand to n concatenated copies.
            if *n == 0 {
                let s = nfa.alloc()?;
                return Ok(NfaFrag { start: s, end: s });
            }
            let first = build_nfa_frag(nfa, inner)?;
            let mut frag = first;
            for _ in 1..*n {
                let next = build_nfa_frag(nfa, inner)?;
                frag = nfa.build_concat(frag, next);
            }
            Ok(frag)
        }
        RegexNode::CountedRange(inner, n, m_opt) => {
            // Mandatory part: n copies.
            // Optional part: (m-n) optional copies (or unlimited if m = None).
            if let Some(m) = m_opt {
                // Build n mandatory copies.
                let mandatory = if *n == 0 {
                    let s = nfa.alloc()?;
                    NfaFrag { start: s, end: s }
                } else {
                    let first = build_nfa_frag(nfa, inner)?;
                    let mut frag = first;
                    for _ in 1..*n {
                        let next = build_nfa_frag(nfa, inner)?;
                        frag = nfa.build_concat(frag, next);
                    }
                    frag
                };

                // Build (m - n) optional copies.
                if *m == *n {
                    return Ok(mandatory);
                }
                let optional_count = m - n;
                let first_opt = build_nfa_frag(nfa, inner)?;
                let mut opt_frag = nfa.build_optional(first_opt)?;
                for _ in 1..optional_count {
                    let next = build_nfa_frag(nfa, inner)?;
                    let next_opt = nfa.build_optional(next)?;
                    opt_frag = nfa.build_concat(opt_frag, next_opt);
                }
                Ok(nfa.build_concat(mandatory, opt_frag))
            } else {
                // `{n,}`: n mandatory copies followed by `*`.
                let mandatory = if *n == 0 {
                    let s = nfa.alloc()?;
                    NfaFrag { start: s, end: s }
                } else {
                    let first = build_nfa_frag(nfa, inner)?;
                    let mut frag = first;
                    for _ in 1..*n {
                        let next = build_nfa_frag(nfa, inner)?;
                        frag = nfa.build_concat(frag, next);
                    }
                    frag
                };
                let star_inner = build_nfa_frag(nfa, inner)?;
                let star_frag = nfa.build_star(star_inner)?;
                Ok(nfa.build_concat(mandatory, star_frag))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Subset DFA construction
// ─────────────────────────────────────────────────────────────────────────────

/// One state in the DFA.
struct DfaState {
    /// Transitions: byte value → DFA state id.
    transitions: HashMap<u8, usize>,
    /// Whether this DFA state is an accepting state.
    is_accept: bool,
}

/// Construct the subset DFA from the NFA.
///
/// Returns `(dfa_states, start_state_id)`.
fn build_dfa(
    nfa: &Nfa,
    nfa_accept: usize,
    nfa_start: usize,
) -> Result<(Vec<DfaState>, usize), RegexCompileError> {
    // Powerset construction: each DFA state is a frozenset (BTreeSet) of NFA state ids.
    let start_closure = nfa.epsilon_closure([nfa_start]);
    let start_is_accept = start_closure.contains(&nfa_accept);

    let mut dfa_states: Vec<DfaState> = Vec::new();
    // Map from NFA state set → DFA state index.
    let mut set_to_dfa: HashMap<BTreeSet<usize>, usize> = HashMap::new();
    let mut worklist: VecDeque<(BTreeSet<usize>, usize)> = VecDeque::new();

    let start_idx = 0usize;
    dfa_states.push(DfaState {
        transitions: HashMap::new(),
        is_accept: start_is_accept,
    });
    set_to_dfa.insert(start_closure.clone(), start_idx);
    worklist.push_back((start_closure, start_idx));

    while let Some((nfa_set, dfa_id)) = worklist.pop_front() {
        // Collect all distinct byte transitions from this NFA state set.
        // Build a mapping: byte → set of target NFA states.
        let mut byte_targets: HashMap<u8, BTreeSet<usize>> = HashMap::new();

        for &nfa_state in &nfa_set {
            for (label, target) in &nfa.states[nfa_state].transitions {
                for b in label.iter() {
                    byte_targets.entry(b).or_default().insert(*target);
                }
            }
        }

        // For each unique byte b, compute ε-closure of the target set.
        for (b, targets) in byte_targets {
            let closure = nfa.epsilon_closure(targets);
            if closure.is_empty() {
                continue;
            }

            let next_dfa_id = if let Some(&existing) = set_to_dfa.get(&closure) {
                existing
            } else {
                // Allocate new DFA state.
                if dfa_states.len() >= MAX_DFA_STATES {
                    return Err(RegexCompileError::DepthExceeded {
                        limit: MAX_DFA_STATES,
                    });
                }
                let new_id = dfa_states.len();
                let is_accept = closure.contains(&nfa_accept);
                dfa_states.push(DfaState {
                    transitions: HashMap::new(),
                    is_accept,
                });
                set_to_dfa.insert(closure.clone(), new_id);
                worklist.push_back((closure, new_id));
                new_id
            };

            dfa_states[dfa_id].transitions.insert(b, next_dfa_id);
        }
    }

    Ok((dfa_states, start_idx))
}

// ─────────────────────────────────────────────────────────────────────────────
// DFA → Grammar
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a DFA into a [`Grammar`].
///
/// Each DFA state maps to a non-terminal `__regex_s{i}`.
/// For each transition `s →[b]→ t`, we emit:
///   `<__regex_s{s}> ::= Terminal([b]) <__regex_s{t}>`
/// For each accept state `s`, we emit an ε-production:
///   `<__regex_s{s}> ::=`
fn dfa_to_grammar(
    dfa_states: &[DfaState],
    start_idx: usize,
) -> Result<Grammar, RegexCompileError> {
    let num_states = dfa_states.len();

    // We must pre-allocate all NT ids first, then set the start.
    // Grammar::new(start) takes a start id, but we build the NTs via alloc_nt.
    // To work around this: create grammar with a placeholder start=0, then
    // alloc NTs (which assigns ids 0, 1, ..., num_states-1), then set start.
    let mut grammar = Grammar::new(0);

    // Allocate one NT per DFA state.
    let mut nt_ids: Vec<NonTerminalId> = Vec::with_capacity(num_states);
    for i in 0..num_states {
        let nt = grammar.alloc_nt(format!("__regex_s{i}"));
        nt_ids.push(nt);
    }

    // Set the actual start symbol.
    grammar.start = nt_ids[start_idx];

    // Emit rules.
    for (state_idx, dfa_state) in dfa_states.iter().enumerate() {
        let lhs_nt = nt_ids[state_idx];

        // ε-production for accept states.
        if dfa_state.is_accept {
            grammar.add_rule(Rule::new(lhs_nt, vec![]));
        }

        // Byte transition rules.
        // Group transitions by target state to consolidate, but since Grammar
        // only supports single-byte terminals, we emit one rule per byte.
        for (&byte_val, &target_idx) in &dfa_state.transitions {
            let target_nt = nt_ids[target_idx];
            grammar.add_rule(Rule::new(
                lhs_nt,
                vec![
                    Symbol::Terminal(vec![byte_val]),
                    Symbol::NonTerminal(target_nt),
                ],
            ));
        }
    }

    Ok(grammar)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compile a regex pattern string into a [`Grammar`].
///
/// The returned grammar is ready to be passed to
/// [`GrammarConstraint::new`](super::constraint::GrammarConstraint::new) for
/// constrained token generation with the given regex.
///
/// # Errors
///
/// - [`RegexCompileError::EmptyPattern`] — the pattern is the empty string
/// - [`RegexCompileError::InvalidSyntax`] — the regex has a syntax error
/// - [`RegexCompileError::UnsupportedFeature`] — a feature not supported by
///   this compiler was used (backreferences, lookahead, etc.)
/// - [`RegexCompileError::DepthExceeded`] — the DFA exceeded 2048 states or
///   a counted quantifier exceeded 64
/// - [`RegexCompileError::InvalidUtf8`] — internal (should not occur for
///   well-formed Rust strings)
///
/// # Example
///
/// ```rust
/// use oxibonsai_runtime::grammar::compile_regex;
///
/// let grammar = compile_regex(r"\d{4}-\d{2}-\d{2}").expect("valid regex");
/// assert!(!grammar.rules.is_empty());
/// ```
pub fn compile_regex(pattern: &str) -> Result<Grammar, RegexCompileError> {
    if pattern.is_empty() {
        return Err(RegexCompileError::EmptyPattern);
    }

    // ── Step 1: Parse regex → AST ────────────────────────────────────────────
    let mut parser = RegexParser::new(pattern);
    let ast = parser.parse_alternation()?;
    if !parser.is_at_end() {
        return Err(RegexCompileError::InvalidSyntax(format!(
            "unexpected character '{}' at position {} (unmatched ')'?)",
            parser.input[parser.pos] as char, parser.pos
        )));
    }

    // ── Step 2: Build Thompson NFA ────────────────────────────────────────────
    let mut nfa = Nfa::new();
    let frag = build_nfa_frag(&mut nfa, &ast)?;

    // Mark the NFA accept state.
    nfa.states[frag.end].is_accept = true;

    // ── Step 3: Subset DFA construction ──────────────────────────────────────
    let (dfa_states, start_idx) = build_dfa(&nfa, frag.end, frag.start)?;

    // ── Step 4: DFA → Grammar ────────────────────────────────────────────────
    let grammar = dfa_to_grammar(&dfa_states, start_idx)?;

    Ok(grammar)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_set_insert_contains() {
        let mut s = ByteSet::empty();
        s.insert(b'A');
        assert!(s.contains(b'A'));
        assert!(!s.contains(b'B'));
    }

    #[test]
    fn byte_set_complement() {
        let mut s = ByteSet::empty();
        s.insert(b'a');
        let c = s.complement();
        assert!(!c.contains(b'a'));
        assert!(c.contains(b'b'));
    }

    #[test]
    fn byte_set_union() {
        let mut a = ByteSet::empty();
        a.insert(b'x');
        let mut b = ByteSet::empty();
        b.insert(b'y');
        let u = a.union(&b);
        assert!(u.contains(b'x'));
        assert!(u.contains(b'y'));
        assert!(!u.contains(b'z'));
    }

    #[test]
    fn byte_set_any_except_newline_has_255_bytes() {
        let s = ByteSet::any_except_newline();
        let count = s.iter().count();
        assert_eq!(count, 255);
        assert!(!s.contains(b'\n'));
    }

    #[test]
    fn digit_set_is_ten_bytes() {
        let s = digit_set();
        let count = s.iter().count();
        assert_eq!(count, 10);
        for d in b'0'..=b'9' {
            assert!(s.contains(d));
        }
    }

    #[test]
    fn word_set_contains_alnum_underscore() {
        let s = word_set();
        assert!(s.contains(b'A'));
        assert!(s.contains(b'z'));
        assert!(s.contains(b'5'));
        assert!(s.contains(b'_'));
        assert!(!s.contains(b'!'));
        assert!(!s.contains(b' '));
    }

    #[test]
    fn space_set_contains_whitespace() {
        let s = space_set();
        assert!(s.contains(b' '));
        assert!(s.contains(b'\t'));
        assert!(s.contains(b'\n'));
        assert!(s.contains(b'\r'));
        assert!(!s.contains(b'a'));
    }

    #[test]
    fn parser_literal_parses() {
        let mut p = RegexParser::new("abc");
        let node = p.parse_alternation().unwrap();
        assert!(matches!(node, RegexNode::Concat(_)));
    }

    #[test]
    fn parser_alternation_parses() {
        let mut p = RegexParser::new("a|b");
        let node = p.parse_alternation().unwrap();
        assert!(matches!(node, RegexNode::Alternation(_)));
    }

    #[test]
    fn parser_counted_exact_parses() {
        let mut p = RegexParser::new("a{3}");
        let node = p.parse_alternation().unwrap();
        assert!(matches!(node, RegexNode::CountedExact(_, 3)));
    }

    #[test]
    fn parser_counted_range_parses() {
        let mut p = RegexParser::new("a{2,5}");
        let node = p.parse_alternation().unwrap();
        assert!(matches!(node, RegexNode::CountedRange(_, 2, Some(5))));
    }

    #[test]
    fn parser_unmatched_paren_fails() {
        let result = compile_regex("(abc");
        assert!(matches!(result, Err(RegexCompileError::InvalidSyntax(_))));
    }
}
