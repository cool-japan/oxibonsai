//! GBNF grammar parser — llama.cpp format → [`Grammar`].
//!
//! GBNF (Grammar BNF) is the grammar format used by llama.cpp for constrained
//! generation. It is EBNF-like:
//!
//! ```text
//! root  ::= item+
//! item  ::= [a-zA-Z]+ " "?
//! number ::= [0-9]+
//! ```
//!
//! # Supported syntax
//!
//! - **Rule definitions**: `name ::= body`
//! - **Alternation**: `a | b | c`
//! - **Sequences**: `a b c`
//! - **Groups**: `(a b c)` with optional quantifier after `)`
//! - **Quantifiers**: `*`, `+`, `?`
//! - **String literals**: `"abc"` → sequence of byte terminals
//! - **Char classes**: `[abc]`, `[a-z]`, `[^abc]`, `[a-zA-Z0-9_]`, `[\x41-\x5A]`
//! - **Rule references**: `rulename`
//! - **Comments**: `# comment to end of line`
//!
//! The `root` rule is the required start symbol.

use std::collections::HashMap;

use super::ast::{Grammar, NonTerminalId, Rule, Symbol};

// ─────────────────────────────────────────────────────────────────────────────
// Public error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising while parsing a GBNF grammar string.
#[derive(Debug, Clone, PartialEq)]
pub enum GbnfParseError {
    /// The grammar string contains no rules at all.
    EmptyGrammar,
    /// The grammar defines rules but none is named `root`.
    MissingRootRule,
    /// A rule reference in a rule body was never defined.
    UnknownRule(String),
    /// An unexpected character was encountered.
    UnexpectedChar { line: usize, col: usize, ch: char },
    /// A string literal was not terminated before end-of-input.
    UnterminatedString,
    /// A character class `[...]` was not terminated.
    UnterminatedCharClass,
    /// An escape sequence used an unsupported escape character.
    InvalidEscape(char),
    /// A grammar feature present in the input is not supported by this parser.
    UnsupportedFeature(String),
    /// The nesting depth of groups exceeded the compiled limit.
    RecursionLimit,
}

impl std::fmt::Display for GbnfParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGrammar => write!(f, "GBNF grammar is empty (no rules found)"),
            Self::MissingRootRule => write!(f, "GBNF grammar has no `root` rule"),
            Self::UnknownRule(name) => write!(f, "rule `{name}` referenced but never defined"),
            Self::UnexpectedChar { line, col, ch } => {
                write!(f, "line {line}:{col}: unexpected character `{ch}`")
            }
            Self::UnterminatedString => write!(f, "unterminated string literal"),
            Self::UnterminatedCharClass => write!(f, "unterminated character class `[...]`"),
            Self::InvalidEscape(ch) => write!(f, "invalid escape sequence `\\{ch}`"),
            Self::UnsupportedFeature(feat) => {
                write!(f, "unsupported GBNF feature: {feat}")
            }
            Self::RecursionLimit => write!(f, "group nesting depth exceeded limit"),
        }
    }
}

impl std::error::Error for GbnfParseError {}

// ─────────────────────────────────────────────────────────────────────────────
// Lexer tokens
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum recursion depth for group/body nesting.
const MAX_DEPTH: usize = 64;

/// Internal lexer token.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// An identifier (rule name).
    Ident(String),
    /// `::=`
    Assign,
    /// `|`
    Pipe,
    /// `*`
    Star,
    /// `+`
    Plus,
    /// `?`
    Question,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// A double-quoted string `"..."` with escape processing applied.
    StringLit(Vec<u8>),
    /// A character class `[...]` with expansion applied (list of matching bytes).
    CharClass(Vec<u8>),
    /// A logical newline (blank line or line break that is not inside a continuation).
    Newline,
    /// End of input.
    Eof,
}

// ─────────────────────────────────────────────────────────────────────────────
// Lexer
// ─────────────────────────────────────────────────────────────────────────────

/// Character-by-character lexer that produces a flat [`Vec<Token>`].
struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    fn new(src: &str) -> Self {
        Self {
            chars: src.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.chars.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    /// Skip spaces and tabs (but NOT newlines).
    fn skip_spaces(&mut self) {
        while matches!(self.peek(), Some(' ') | Some('\t') | Some('\r')) {
            self.advance();
        }
    }

    /// Parse an escape byte from a string literal or char class context.
    fn parse_escape_byte(&mut self) -> Result<u8, GbnfParseError> {
        match self.advance() {
            None => Err(GbnfParseError::UnterminatedString),
            Some('n') => Ok(b'\n'),
            Some('r') => Ok(b'\r'),
            Some('t') => Ok(b'\t'),
            Some('\\') => Ok(b'\\'),
            Some('"') => Ok(b'"'),
            Some('\'') => Ok(b'\''),
            Some('0') => Ok(0u8),
            Some('x') => self.parse_hex_escape(),
            Some(ch) => {
                // Reject unrecognised escapes strictly.
                Err(GbnfParseError::InvalidEscape(ch))
            }
        }
    }

    /// Parse `\xNN` hex escape, consuming the two hex digits.
    fn parse_hex_escape(&mut self) -> Result<u8, GbnfParseError> {
        let hi = self.advance().ok_or(GbnfParseError::UnterminatedString)?;
        let lo = self.advance().ok_or(GbnfParseError::UnterminatedString)?;
        let hex = format!("{hi}{lo}");
        u8::from_str_radix(&hex, 16).map_err(|_| GbnfParseError::UnexpectedChar {
            line: self.line,
            col: self.col,
            ch: hi,
        })
    }

    /// Parse a double-quoted string literal `"..."`.
    fn parse_string_lit(&mut self) -> Result<Vec<u8>, GbnfParseError> {
        // Opening `"` already consumed by caller.
        let mut bytes: Vec<u8> = Vec::new();
        loop {
            match self.peek() {
                None => return Err(GbnfParseError::UnterminatedString),
                Some('"') => {
                    self.advance();
                    break;
                }
                Some('\\') => {
                    self.advance(); // consume `\`
                    let b = self.parse_escape_byte()?;
                    bytes.push(b);
                }
                Some(ch) => {
                    // Emit the UTF-8 encoding of the character.
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf);
                    bytes.extend_from_slice(encoded.as_bytes());
                    self.advance();
                }
            }
        }
        Ok(bytes)
    }

    /// Parse a character class `[...]`, returning the sorted list of matching bytes.
    fn parse_char_class(&mut self) -> Result<Vec<u8>, GbnfParseError> {
        // Opening `[` already consumed by caller.
        // Determine if this is a negated class.
        let negated = if self.peek() == Some('^') {
            self.advance();
            true
        } else {
            false
        };

        let mut set = [false; 256];

        // Parse class body until `]`.
        loop {
            match self.peek() {
                None => return Err(GbnfParseError::UnterminatedCharClass),
                Some(']') => {
                    self.advance();
                    break;
                }
                Some('\\') => {
                    self.advance(); // consume `\`
                    let b = self.parse_escape_byte()?;
                    // Check for range: `\xNN-something`
                    if self.peek() == Some('-') && self.peek2() != Some(']') {
                        self.advance(); // consume `-`
                        let end_byte = self.parse_class_single_byte()?;
                        for byte in b..=end_byte {
                            set[byte as usize] = true;
                        }
                    } else {
                        set[b as usize] = true;
                    }
                }
                Some(ch) => {
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf);
                    // Only handle single-byte (ASCII) characters for char class members
                    // in non-range context; multi-byte Unicode is not supported in classes.
                    if encoded.len() > 1 {
                        return Err(GbnfParseError::UnsupportedFeature(
                            "non-ASCII character in character class".to_string(),
                        ));
                    }
                    let b = encoded.as_bytes()[0];
                    self.advance();

                    // Check for range: `a-z`
                    if self.peek() == Some('-') && self.peek2() != Some(']') {
                        self.advance(); // consume `-`
                        let end_byte = self.parse_class_single_byte()?;
                        for byte in b..=end_byte {
                            set[byte as usize] = true;
                        }
                    } else {
                        set[b as usize] = true;
                    }
                }
            }
        }

        // Apply negation.
        if negated {
            for b in &mut set {
                *b = !*b;
            }
        }

        let bytes: Vec<u8> = (0u8..=255u8).filter(|&b| set[b as usize]).collect();
        Ok(bytes)
    }

    /// Parse one byte for use as the start or end of a character class range.
    fn parse_class_single_byte(&mut self) -> Result<u8, GbnfParseError> {
        match self.peek() {
            None => Err(GbnfParseError::UnterminatedCharClass),
            Some(']') => Err(GbnfParseError::UnterminatedCharClass),
            Some('\\') => {
                self.advance(); // consume `\`
                self.parse_escape_byte()
            }
            Some(ch) => {
                let mut buf = [0u8; 4];
                let encoded = ch.encode_utf8(&mut buf);
                if encoded.len() > 1 {
                    return Err(GbnfParseError::UnsupportedFeature(
                        "non-ASCII character in character class range".to_string(),
                    ));
                }
                let b = encoded.as_bytes()[0];
                self.advance();
                Ok(b)
            }
        }
    }

    /// Tokenise the full input into a `Vec<Token>`.
    ///
    /// Newlines are emitted as `Token::Newline` only when they appear between
    /// rule bodies (not inside rule bodies, but the parser will handle merging).
    fn tokenise(&mut self) -> Result<Vec<Token>, GbnfParseError> {
        let mut tokens: Vec<Token> = Vec::new();

        loop {
            self.skip_spaces();

            match self.peek() {
                None => {
                    tokens.push(Token::Eof);
                    break;
                }
                // Comment: skip to end of line.
                Some('#') => {
                    while matches!(self.peek(), Some(ch) if ch != '\n') {
                        self.advance();
                    }
                }
                // Newline: emit a Newline token.
                Some('\n') => {
                    self.advance();
                    // Collapse multiple newlines into one token.
                    tokens.push(Token::Newline);
                }
                // `::=` assignment.
                Some(':') => {
                    let (line, col) = (self.line, self.col);
                    if self.chars.get(self.pos + 1).copied() == Some(':')
                        && self.chars.get(self.pos + 2).copied() == Some('=')
                    {
                        self.pos += 3;
                        self.col += 3;
                        tokens.push(Token::Assign);
                    } else {
                        return Err(GbnfParseError::UnexpectedChar { line, col, ch: ':' });
                    }
                }
                // `|` alternation.
                Some('|') => {
                    self.advance();
                    tokens.push(Token::Pipe);
                }
                // `*` quantifier.
                Some('*') => {
                    self.advance();
                    tokens.push(Token::Star);
                }
                // `+` quantifier.
                Some('+') => {
                    self.advance();
                    tokens.push(Token::Plus);
                }
                // `?` optional quantifier.
                Some('?') => {
                    self.advance();
                    tokens.push(Token::Question);
                }
                // `(` group open.
                Some('(') => {
                    self.advance();
                    tokens.push(Token::LParen);
                }
                // `)` group close.
                Some(')') => {
                    self.advance();
                    tokens.push(Token::RParen);
                }
                // String literal.
                Some('"') => {
                    self.advance(); // consume `"`
                    let bytes = self.parse_string_lit()?;
                    tokens.push(Token::StringLit(bytes));
                }
                // Character class.
                Some('[') => {
                    self.advance(); // consume `[`
                    let bytes = self.parse_char_class()?;
                    tokens.push(Token::CharClass(bytes));
                }
                // Identifier (rule name).
                Some(ch) if ch.is_ascii_alphabetic() || ch == '_' => {
                    let mut ident = String::new();
                    while let Some(c) = self.peek() {
                        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                            ident.push(c);
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    tokens.push(Token::Ident(ident));
                }
                Some(ch) => {
                    let (line, col) = (self.line, self.col);
                    return Err(GbnfParseError::UnexpectedChar { line, col, ch });
                }
            }
        }

        Ok(tokens)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pre-scan: collect rule names
// ─────────────────────────────────────────────────────────────────────────────

/// Scan the token stream to collect all rule names (every Ident followed by Assign).
fn collect_rule_names(tokens: &[Token]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut i = 0;
    while i < tokens.len() {
        if let Token::Ident(name) = &tokens[i] {
            // Skip trailing newlines between tokens.
            let mut j = i + 1;
            while j < tokens.len() {
                match &tokens[j] {
                    Token::Newline => j += 1,
                    _ => break,
                }
            }
            if matches!(tokens.get(j), Some(Token::Assign)) && !seen.contains(name) {
                seen.insert(name.clone());
                names.push(name.clone());
            }
        }
        i += 1;
    }
    names
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful recursive-descent parser over a flat token stream.
struct Parser<'g> {
    tokens: Vec<Token>,
    pos: usize,
    /// Maps rule name → pre-allocated NonTerminalId.
    nt_map: HashMap<String, NonTerminalId>,
    /// Which NTs have had at least one production rule added.
    defined: std::collections::HashSet<NonTerminalId>,
    /// The grammar being built.
    grammar: &'g mut Grammar,
    /// Counter for generating unique synthetic NT names.
    synthetic_counter: usize,
}

impl<'g> Parser<'g> {
    fn new(
        tokens: Vec<Token>,
        nt_map: HashMap<String, NonTerminalId>,
        grammar: &'g mut Grammar,
    ) -> Self {
        Self {
            tokens,
            pos: 0,
            nt_map,
            defined: std::collections::HashSet::new(),
            grammar,
            synthetic_counter: 0,
        }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> &Token {
        let tok = self.tokens.get(self.pos).unwrap_or(&Token::Eof);
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    /// Consume all consecutive `Token::Newline` tokens.
    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Token::Newline) {
            self.advance();
        }
    }

    /// Generate a fresh unique name suffix for a synthetic NT.
    fn next_synthetic_id(&mut self) -> usize {
        let id = self.synthetic_counter;
        self.synthetic_counter += 1;
        id
    }

    /// Allocate a synthetic NT in the grammar and return its id.
    fn alloc_synthetic(&mut self, prefix: &str) -> NonTerminalId {
        let id = self.next_synthetic_id();
        let name = format!("__{prefix}_{id}");
        self.grammar.alloc_nt(&name)
    }

    /// Resolve a rule name to its NT id, returning `UnknownRule` if not found.
    fn resolve_nt(&self, name: &str) -> Result<NonTerminalId, GbnfParseError> {
        self.nt_map
            .get(name)
            .copied()
            .ok_or_else(|| GbnfParseError::UnknownRule(name.to_string()))
    }

    /// Parse all rules from the token stream into `self.grammar`.
    fn parse_all(&mut self) -> Result<(), GbnfParseError> {
        self.skip_newlines();

        while !matches!(self.peek(), Token::Eof) {
            self.parse_one_rule()?;
            self.skip_newlines();
        }

        Ok(())
    }

    /// Parse one `name ::= body` rule definition.
    ///
    /// After parsing the body, rule(s) are added to `self.grammar` for `lhs_id`.
    fn parse_one_rule(&mut self) -> Result<(), GbnfParseError> {
        // Expect an identifier.
        let lhs_name = match self.advance().clone() {
            Token::Ident(name) => name,
            Token::Eof => return Ok(()),
            tok => {
                return Err(GbnfParseError::UnexpectedChar {
                    line: 0,
                    col: 0,
                    ch: token_to_char(&tok),
                });
            }
        };

        // Skip any newlines between name and `::=`.
        self.skip_newlines();

        // Expect `::=`.
        match self.advance().clone() {
            Token::Assign => {}
            tok => {
                return Err(GbnfParseError::UnexpectedChar {
                    line: 0,
                    col: 0,
                    ch: token_to_char(&tok),
                });
            }
        }

        let lhs_id = self.resolve_nt(&lhs_name)?;
        self.defined.insert(lhs_id);

        // Parse the body: one or more alternatives separated by `|`.
        // The body ends at a Newline followed by another `Ident ::=` pattern,
        // or at Eof.
        self.parse_body_into(lhs_id, 0)?;

        Ok(())
    }

    /// Parse `alternative ('|' alternative)*` and add rules for each alternative
    /// to `lhs_id`. This is the top-level body parser used for rule definitions.
    fn parse_body_into(
        &mut self,
        lhs_id: NonTerminalId,
        depth: usize,
    ) -> Result<(), GbnfParseError> {
        if depth > MAX_DEPTH {
            return Err(GbnfParseError::RecursionLimit);
        }

        loop {
            // Parse one alternative (possibly empty = ε production).
            let rhs = self.parse_alternative(depth)?;
            self.grammar.add_rule(Rule::new(lhs_id, rhs));

            // If next token is `|`, consume it and continue to next alternative.
            if matches!(self.peek(), Token::Pipe) {
                self.advance(); // consume `|`
                                // Skip newlines after `|` (continuation lines).
                self.skip_newlines();
                continue;
            }

            // Otherwise we're done with this body.
            break;
        }

        Ok(())
    }

    /// Parse `(atom quantifier?)*` — a sequence of atoms for one alternative.
    ///
    /// Stops at: `Eof`, `RParen`, `Pipe`, or a Newline that is NOT followed by `|`
    /// (the latter marks end of current rule body).
    fn parse_alternative(&mut self, depth: usize) -> Result<Vec<Symbol>, GbnfParseError> {
        let mut sequence: Vec<Symbol> = Vec::new();

        loop {
            match self.peek() {
                Token::Eof | Token::RParen => break,
                Token::Pipe => break,
                // A newline signals end of current alternative UNLESS the next
                // non-newline token is `|` (continuation of same rule).
                Token::Newline => {
                    // Look ahead past newlines to see if `|` follows.
                    let mut look = self.pos + 1;
                    while look < self.tokens.len() {
                        match &self.tokens[look] {
                            Token::Newline => look += 1,
                            Token::Pipe => {
                                // Continuation — keep parsing.
                                // Consume the newlines.
                                while matches!(self.peek(), Token::Newline) {
                                    self.advance();
                                }
                                // Now we see `|`, which the outer loop will handle.
                                return Ok(sequence);
                            }
                            _ => break,
                        }
                    }
                    // No `|` found after newlines: end of this rule.
                    break;
                }
                // Check if this is the start of a new rule definition.
                // Pattern: Ident followed (after optional newlines) by Assign.
                Token::Ident(_) if self.is_new_rule_start() => break,
                // Otherwise parse an atom.
                _ => {
                    let atom_sym = self.parse_atom(depth)?;
                    // Parse optional quantifier.
                    let sym = self.apply_quantifier(atom_sym, depth)?;
                    sequence.extend(sym);
                }
            }
        }

        Ok(sequence)
    }

    /// Returns true if the current position looks like `Ident ::=`
    /// (a new rule definition beginning), indicating we should stop parsing
    /// the current rule body.
    fn is_new_rule_start(&self) -> bool {
        if !matches!(self.peek(), Token::Ident(_)) {
            return false;
        }
        let mut j = self.pos + 1;
        while j < self.tokens.len() {
            match &self.tokens[j] {
                Token::Newline => j += 1,
                Token::Assign => return true,
                _ => return false,
            }
        }
        false
    }

    /// Parse one atom: string literal, char class, identifier (rule ref), or group.
    ///
    /// Returns a `Vec<Symbol>` because string literals expand inline into multiple
    /// byte terminals, while other atoms return a single symbol wrapped in a NT.
    fn parse_atom(&mut self, depth: usize) -> Result<AtomResult, GbnfParseError> {
        if depth > MAX_DEPTH {
            return Err(GbnfParseError::RecursionLimit);
        }

        match self.peek().clone() {
            Token::StringLit(bytes) => {
                self.advance();
                Ok(AtomResult::Inline(bytes))
            }
            Token::CharClass(bytes) => {
                self.advance();
                // Create a fresh synthetic NT for the char class.
                let class_nt = self.alloc_synthetic("gbnf_class");
                for b in &bytes {
                    self.grammar
                        .add_rule(Rule::new(class_nt, vec![Symbol::Terminal(vec![*b])]));
                }
                Ok(AtomResult::Single(Symbol::NonTerminal(class_nt)))
            }
            Token::Ident(name) => {
                self.advance();
                let nt_id = self.resolve_nt(&name)?;
                Ok(AtomResult::Single(Symbol::NonTerminal(nt_id)))
            }
            Token::LParen => {
                self.advance(); // consume `(`
                                // Parse body inside the group.
                let group_nt = self.alloc_synthetic("gbnf_group");
                self.parse_body_into(group_nt, depth + 1)?;
                // Expect `)`.
                match self.advance().clone() {
                    Token::RParen => {}
                    tok => {
                        return Err(GbnfParseError::UnexpectedChar {
                            line: 0,
                            col: 0,
                            ch: token_to_char(&tok),
                        });
                    }
                }
                Ok(AtomResult::Single(Symbol::NonTerminal(group_nt)))
            }
            tok => Err(GbnfParseError::UnexpectedChar {
                line: 0,
                col: 0,
                ch: token_to_char(&tok),
            }),
        }
    }

    /// Apply an optional quantifier (`*`, `+`, `?`) to an atom.
    ///
    /// Returns a `Vec<Symbol>` suitable for direct inclusion in an alternative's
    /// RHS sequence.
    fn apply_quantifier(
        &mut self,
        atom: AtomResult,
        depth: usize,
    ) -> Result<Vec<Symbol>, GbnfParseError> {
        if depth > MAX_DEPTH {
            return Err(GbnfParseError::RecursionLimit);
        }

        let quantifier = match self.peek() {
            Token::Star => {
                self.advance();
                Some('*')
            }
            Token::Plus => {
                self.advance();
                Some('+')
            }
            Token::Question => {
                self.advance();
                Some('?')
            }
            _ => None,
        };

        match quantifier {
            None => {
                // No quantifier: emit the atom as-is.
                Ok(atom_to_symbols(atom))
            }
            Some('?') => {
                // Optional: 0 or 1 repetitions.
                let opt_nt = self.alloc_synthetic("gbnf_opt");
                // opt_nt ::= ε
                self.grammar.add_rule(Rule::new(opt_nt, vec![]));
                // opt_nt ::= atom
                let rhs = atom_to_symbols(atom);
                self.grammar.add_rule(Rule::new(opt_nt, rhs));
                Ok(vec![Symbol::NonTerminal(opt_nt)])
            }
            Some('*') => {
                // Zero or more: Kleene star.
                let star_nt = self.alloc_synthetic("gbnf_star");
                let rhs = atom_to_symbols(atom);
                // star_nt ::= ε
                self.grammar.add_rule(Rule::new(star_nt, vec![]));
                // star_nt ::= atom star_nt
                let mut star_rhs = rhs;
                star_rhs.push(Symbol::NonTerminal(star_nt));
                self.grammar.add_rule(Rule::new(star_nt, star_rhs));
                Ok(vec![Symbol::NonTerminal(star_nt)])
            }
            Some('+') => {
                // One or more: plus.
                // plus_nt ::= atom star_nt
                // where star_nt allows zero or more additional occurrences.
                let star_nt = self.alloc_synthetic("gbnf_star");
                let atom_rhs = atom_to_symbols(atom);
                // star_nt ::= ε
                self.grammar.add_rule(Rule::new(star_nt, vec![]));
                // star_nt ::= atom star_nt
                let mut star_rhs = atom_rhs.clone();
                star_rhs.push(Symbol::NonTerminal(star_nt));
                self.grammar.add_rule(Rule::new(star_nt, star_rhs));

                let plus_nt = self.alloc_synthetic("gbnf_plus");
                // plus_nt ::= atom star_nt  (requires at least one)
                let mut plus_rhs = atom_rhs;
                plus_rhs.push(Symbol::NonTerminal(star_nt));
                self.grammar.add_rule(Rule::new(plus_nt, plus_rhs));
                Ok(vec![Symbol::NonTerminal(plus_nt)])
            }
            _ => unreachable!(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AtomResult helper
// ─────────────────────────────────────────────────────────────────────────────

/// The result of parsing one atom.
///
/// `Inline` is used for string literals which expand to multiple byte terminals
/// directly in the enclosing rule's RHS. `Single` wraps atoms that reduce to a
/// single `Symbol` (rule references, char classes, groups).
enum AtomResult {
    /// A string literal that expands to individual byte terminals.
    Inline(Vec<u8>),
    /// A single symbol (NonTerminal or a single Terminal).
    Single(Symbol),
}

/// Convert an `AtomResult` into a `Vec<Symbol>` for inclusion in a rule RHS.
fn atom_to_symbols(atom: AtomResult) -> Vec<Symbol> {
    match atom {
        AtomResult::Inline(bytes) => bytes
            .into_iter()
            .map(|b| Symbol::Terminal(vec![b]))
            .collect(),
        AtomResult::Single(sym) => vec![sym],
    }
}

/// Return a representative char for error reporting.
fn token_to_char(tok: &Token) -> char {
    match tok {
        Token::Assign => ':',
        Token::Pipe => '|',
        Token::Star => '*',
        Token::Plus => '+',
        Token::Question => '?',
        Token::LParen => '(',
        Token::RParen => ')',
        Token::Newline => '\n',
        Token::Eof => '\0',
        Token::Ident(_) => 'i',
        Token::StringLit(_) => '"',
        Token::CharClass(_) => '[',
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a GBNF grammar string and return a [`Grammar`].
///
/// The returned grammar has its start symbol set to the `root` rule NT.
///
/// # Errors
///
/// Returns a [`GbnfParseError`] if:
/// - The input is empty ([`GbnfParseError::EmptyGrammar`])
/// - No `root` rule is defined ([`GbnfParseError::MissingRootRule`])
/// - A rule body references an undefined rule name ([`GbnfParseError::UnknownRule`])
/// - The input contains syntax errors
pub fn parse_gbnf(src: &str) -> Result<Grammar, GbnfParseError> {
    // ── Lex ─────────────────────────────────────────────────────────────────
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenise()?;

    // Check for completely empty input.
    let has_content = tokens
        .iter()
        .any(|t| !matches!(t, Token::Newline | Token::Eof));
    if !has_content {
        return Err(GbnfParseError::EmptyGrammar);
    }

    // ── Pass 1: collect all defined rule names ───────────────────────────────
    let rule_names = collect_rule_names(&tokens);
    if rule_names.is_empty() {
        return Err(GbnfParseError::EmptyGrammar);
    }

    // Check for the mandatory `root` rule.
    if !rule_names.iter().any(|n| n == "root") {
        return Err(GbnfParseError::MissingRootRule);
    }

    // Pre-allocate a Grammar with a placeholder start (0 initially).
    let mut grammar = Grammar::new(0);

    // Allocate NT ids for all defined rule names.
    let mut nt_map: HashMap<String, NonTerminalId> = HashMap::new();
    for name in &rule_names {
        let id = grammar.alloc_nt(name);
        nt_map.insert(name.clone(), id);
    }

    // Set the grammar start to the `root` NT.
    let root_id = *nt_map.get("root").expect("root was in rule_names");
    grammar.start = root_id;

    // ── Pass 2: parse rule bodies ────────────────────────────────────────────
    let mut parser = Parser::new(tokens, nt_map.clone(), &mut grammar);
    parser.parse_all()?;

    // Extract defined set from parser before dropping the mutable borrow.
    let defined = parser.defined;

    // ── Validation: all referenced NTs must be defined ───────────────────────
    // Collect violations first to avoid borrow conflicts during iteration.
    let violations: Vec<String> = grammar
        .rules
        .iter()
        .flat_map(|rule| rule.rhs.iter())
        .filter_map(|sym| {
            if let Symbol::NonTerminal(id) = sym {
                let name = grammar.nt_names.get(id).cloned().unwrap_or_default();
                // Synthetic NTs (starting with `__`) are always defined by the parser.
                if !name.starts_with("__") && !defined.contains(id) {
                    return Some(name);
                }
            }
            None
        })
        .collect();

    if let Some(unknown) = violations.into_iter().next() {
        return Err(GbnfParseError::UnknownRule(unknown));
    }

    Ok(grammar)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(src: &str) -> Grammar {
        parse_gbnf(src).unwrap_or_else(|e| panic!("parse_gbnf failed: {e}"))
    }

    #[test]
    fn parse_simple_literal() {
        let g = parse_ok(r#"root ::= "hello""#);
        // Must have at least one rule for root.
        let root_rules: Vec<_> = g.rules_for(g.start()).collect();
        assert!(!root_rules.is_empty());
    }

    #[test]
    fn parse_alternation_internal() {
        let g = parse_ok(r#"root ::= "cat" | "dog""#);
        let root_rules: Vec<_> = g.rules_for(g.start()).collect();
        assert_eq!(root_rules.len(), 2);
    }

    #[test]
    fn error_empty_grammar_internal() {
        assert!(matches!(parse_gbnf(""), Err(GbnfParseError::EmptyGrammar)));
    }

    #[test]
    fn error_missing_root_internal() {
        let err = parse_gbnf("word ::= \"x\"");
        assert!(matches!(err, Err(GbnfParseError::MissingRootRule)));
    }

    #[test]
    fn parse_star_quantifier_internal() {
        let g = parse_ok(r#"root ::= "a"*"#);
        assert!(!g.rules.is_empty());
        // Start NT must appear as the LHS of at least one rule.
        let start_has_rule = g.rules.iter().any(|r| r.lhs == g.start);
        assert!(start_has_rule, "root NT must have at least one rule");
    }

    #[test]
    fn error_display_non_empty() {
        let e = GbnfParseError::EmptyGrammar;
        assert!(!e.to_string().is_empty());
        let e2 = GbnfParseError::UnknownRule("foo".to_string());
        assert!(e2.to_string().contains("foo"));
    }
}
