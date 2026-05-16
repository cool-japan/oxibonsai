//! Hand-rolled BNF text parser.
//!
//! Supported syntax:
//! ```text
//! <expr> ::= <term> "+" <expr> | <term>
//! <term> ::= <factor> "*" <term> | <factor>
//! <factor> ::= "(" <expr> ")" | <num>
//! <num> ::= "0" | "1" | ... | "9"
//! ```
//!
//! Rules:
//! - Non-terminals: `<name>` (angle brackets required)
//! - Terminals: `"..."` (double-quoted; supports `\\`, `\"`, `\n`, `\r`, `\t` escapes)
//! - Separator: `::=` between lhs and rhs
//! - Alternation: `|` separates alternatives
//! - Comments: `#` to end of line
//! - Whitespace: ignored between tokens (but not inside terminals)
//! - Multiple rules for the same non-terminal are merged
//! - Lines can be continued with `\` at the very end (before the `\n`)

use std::collections::HashMap;

use super::ast::{Grammar, NonTerminalId, Rule, Symbol};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise while parsing a BNF grammar string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BnfParseError {
    /// An unexpected character was encountered.
    UnexpectedChar {
        line: usize,
        col: usize,
        got: char,
        expected: String,
    },
    /// A string literal was not closed before end-of-input.
    UnterminatedString { line: usize, col: usize },
    /// `::=` was not found after the lhs non-terminal.
    MissingDefinitionSeparator { line: usize, col: usize },
    /// A non-terminal name was empty (`<>`).
    EmptyNonTerminalName { line: usize, col: usize },
    /// A non-terminal was used in a rule rhs but never defined.
    UndefinedNonTerminal { name: String },
    /// The input contained no rules at all.
    EmptyInput,
}

impl std::fmt::Display for BnfParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedChar {
                line,
                col,
                got,
                expected,
            } => write!(
                f,
                "line {line}:{col}: unexpected `{got}`, expected {expected}"
            ),
            Self::UnterminatedString { line, col } => {
                write!(f, "line {line}:{col}: unterminated string literal")
            }
            Self::MissingDefinitionSeparator { line, col } => write!(
                f,
                "line {line}:{col}: expected `::=` after non-terminal name"
            ),
            Self::EmptyNonTerminalName { line, col } => {
                write!(f, "line {line}:{col}: empty non-terminal name `<>`")
            }
            Self::UndefinedNonTerminal { name } => write!(f, "undefined non-terminal `{name}`"),
            Self::EmptyInput => write!(f, "grammar input is empty (no rules found)"),
        }
    }
}

impl std::error::Error for BnfParseError {}

// ─────────────────────────────────────────────────────────────────────────────
// Token type (internal)
// ─────────────────────────────────────────────────────────────────────────────

/// Lexer token with position information.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// `<name>`
    NonTerminal {
        name: String,
        line: usize,
        col: usize,
    },
    /// `"bytes"` — the bytes are the actual UTF-8 encoding of the escape-processed string.
    Terminal {
        bytes: Vec<u8>,
        line: usize,
        col: usize,
    },
    /// `::=`
    Assign { line: usize, col: usize },
    /// `|`
    Pipe { line: usize },
    /// End of the token stream.
    Eof,
}

impl Token {
    fn position(&self) -> (usize, usize) {
        match self {
            Token::NonTerminal { line, col, .. } => (*line, *col),
            Token::Terminal { line, col, .. } => (*line, *col),
            Token::Assign { line, col } => (*line, *col),
            Token::Pipe { line } => (*line, 0),
            Token::Eof => (0, 0),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lexer
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful character-by-character lexer.
struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
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

    /// Skip whitespace (space, tab, CR, LF) and comments (`# … \n`).
    ///
    /// Also handles line continuation: a `\` immediately before `\n` is treated
    /// as whitespace and the newline is consumed.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(' ') | Some('\t') | Some('\r') | Some('\n') => {
                    self.advance();
                }
                // Line continuation: backslash followed immediately by newline.
                Some('\\') if self.peek2() == Some('\n') => {
                    self.advance(); // consume '\'
                    self.advance(); // consume '\n'
                }
                Some('#') => {
                    // Comment: skip until newline.
                    while let Some(ch) = self.peek() {
                        if ch == '\n' {
                            break;
                        }
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    /// Parse one non-terminal `<name>`.
    fn parse_non_terminal(&mut self) -> Result<Token, BnfParseError> {
        let (line, col) = (self.line, self.col);
        // Consume '<'.
        self.advance();
        let mut name = String::new();
        loop {
            match self.peek() {
                Some('>') => {
                    self.advance();
                    break;
                }
                Some(ch) if ch != '\n' => {
                    name.push(ch);
                    self.advance();
                }
                _ => {
                    return Err(BnfParseError::UnexpectedChar {
                        line,
                        col,
                        got: self.peek().unwrap_or('\0'),
                        expected: "`>` to close non-terminal name".to_string(),
                    });
                }
            }
        }
        if name.is_empty() {
            return Err(BnfParseError::EmptyNonTerminalName { line, col });
        }
        Ok(Token::NonTerminal { name, line, col })
    }

    /// Parse one terminal `"..."` with escape processing.
    fn parse_terminal(&mut self) -> Result<Token, BnfParseError> {
        let (line, col) = (self.line, self.col);
        // Consume opening `"`.
        self.advance();
        let mut bytes: Vec<u8> = Vec::new();
        loop {
            match self.peek() {
                None | Some('\n') => {
                    return Err(BnfParseError::UnterminatedString { line, col });
                }
                Some('"') => {
                    self.advance();
                    break;
                }
                Some('\\') => {
                    self.advance(); // consume '\'
                    match self.peek() {
                        None => {
                            return Err(BnfParseError::UnterminatedString { line, col });
                        }
                        Some('n') => {
                            bytes.push(b'\n');
                            self.advance();
                        }
                        Some('r') => {
                            bytes.push(b'\r');
                            self.advance();
                        }
                        Some('t') => {
                            bytes.push(b'\t');
                            self.advance();
                        }
                        Some('"') => {
                            bytes.push(b'"');
                            self.advance();
                        }
                        Some('\\') => {
                            bytes.push(b'\\');
                            self.advance();
                        }
                        Some('0') => {
                            bytes.push(0u8);
                            self.advance();
                        }
                        Some('x') => {
                            // Hex escape \xNN
                            self.advance(); // consume 'x'
                            let mut hex = String::new();
                            for _ in 0..2 {
                                match self.peek() {
                                    Some(c) if c.is_ascii_hexdigit() => {
                                        hex.push(c);
                                        self.advance();
                                    }
                                    other => {
                                        return Err(BnfParseError::UnexpectedChar {
                                            line: self.line,
                                            col: self.col,
                                            got: other.unwrap_or('\0'),
                                            expected: "hex digit after \\x".to_string(),
                                        });
                                    }
                                }
                            }
                            let byte_val = u8::from_str_radix(&hex, 16).unwrap_or(0);
                            bytes.push(byte_val);
                        }
                        Some(c) => {
                            // Unknown escape — treat as literal.
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                            self.advance();
                        }
                    }
                }
                Some(ch) => {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    self.advance();
                }
            }
        }
        Ok(Token::Terminal { bytes, line, col })
    }

    /// Attempt to consume `::=`.  Returns `Err` if not present.
    fn parse_assign(&mut self) -> Result<Token, BnfParseError> {
        let (line, col) = (self.line, self.col);
        // Expect three specific characters.
        if self.chars.get(self.pos).copied() == Some(':')
            && self.chars.get(self.pos + 1).copied() == Some(':')
            && self.chars.get(self.pos + 2).copied() == Some('=')
        {
            self.pos += 3;
            self.col += 3;
            Ok(Token::Assign { line, col })
        } else {
            Err(BnfParseError::MissingDefinitionSeparator { line, col })
        }
    }

    /// Tokenise the entire input into a flat `Vec<Token>`.
    fn tokenise(&mut self) -> Result<Vec<Token>, BnfParseError> {
        let mut tokens: Vec<Token> = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            match self.peek() {
                None => {
                    tokens.push(Token::Eof);
                    break;
                }
                Some('<') => {
                    tokens.push(self.parse_non_terminal()?);
                }
                Some('"') => {
                    tokens.push(self.parse_terminal()?);
                }
                Some(':') => {
                    tokens.push(self.parse_assign()?);
                }
                Some('|') => {
                    let line = self.line;
                    self.advance();
                    tokens.push(Token::Pipe { line });
                }
                Some('=') => {
                    // A bare `=` suggests the user wrote `<S> = ...` instead of
                    // `<S> ::= ...`.  Emit a targeted error.
                    return Err(BnfParseError::MissingDefinitionSeparator {
                        line: self.line,
                        col: self.col,
                    });
                }
                Some(ch) => {
                    return Err(BnfParseError::UnexpectedChar {
                        line: self.line,
                        col: self.col,
                        got: ch,
                        expected: "`<`, `\"`, `::=`, `|` or `#`".to_string(),
                    });
                }
            }
        }
        Ok(tokens)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful parser over a flat token stream.
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn is_eof(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    /// Parse all rules and return a completed `Grammar`.
    ///
    /// Pass 1: scan all non-terminal names and assign ids.
    /// Pass 2: parse each rule's alternatives.
    fn parse(&mut self) -> Result<Grammar, BnfParseError> {
        if self.tokens.is_empty() || matches!(self.tokens[0], Token::Eof) {
            return Err(BnfParseError::EmptyInput);
        }

        // ── Pass 1: collect all non-terminal names ────────────────────────
        let mut nt_to_id: HashMap<String, NonTerminalId> = HashMap::new();
        let mut nt_order: Vec<String> = Vec::new(); // insertion order

        // First pass over the token stream: collect all lhs names
        // (every NT that appears directly before `::=`).
        for i in 0..self.tokens.len() {
            if let Token::NonTerminal { name, .. } = &self.tokens[i] {
                // Check if the next non-EOF, non-pipe, non-terminal, non-`::=` token
                // after this is `::=`.
                let j = i + 1;
                if matches!(self.tokens.get(j), Some(Token::Assign { .. }))
                    && !nt_to_id.contains_key(name)
                {
                    let id = nt_to_id.len();
                    nt_to_id.insert(name.clone(), id);
                    nt_order.push(name.clone());
                }
            }
        }

        // Also collect NTs that only appear on the rhs (forward references) by
        // scanning all NT tokens and registering them if not already known.
        // We need to give them an id even if they have no explicit definition so
        // that we can report an UndefinedNonTerminal error later.
        for tok in &self.tokens {
            if let Token::NonTerminal { name, .. } = tok {
                if !nt_to_id.contains_key(name) {
                    let id = nt_to_id.len();
                    nt_to_id.insert(name.clone(), id);
                    nt_order.push(name.clone());
                }
            }
        }

        if nt_order.is_empty() {
            return Err(BnfParseError::EmptyInput);
        }

        // Determine the start symbol: LHS of the first rule.
        let start_name = {
            let mut found: Option<String> = None;
            for i in 0..self.tokens.len() {
                if let Token::NonTerminal { name, .. } = &self.tokens[i] {
                    if matches!(self.tokens.get(i + 1), Some(Token::Assign { .. })) {
                        found = Some(name.clone());
                        break;
                    }
                }
            }
            found.ok_or(BnfParseError::EmptyInput)?
        };
        let start_id = *nt_to_id.get(&start_name).expect("start must be in map");

        // Build grammar skeleton with nt_count set.
        let nt_count = nt_to_id.len();
        let mut grammar = Grammar::new(start_id);
        grammar.nt_count = nt_count;
        for (name, id) in &nt_to_id {
            grammar.nt_names.insert(*id, name.clone());
        }

        // ── Pass 2: parse rules ───────────────────────────────────────────
        // Track which NTs have been defined with at least one rule.
        let mut defined_nts: std::collections::HashSet<NonTerminalId> =
            std::collections::HashSet::new();

        while !self.is_eof() {
            // Expect: <lhs> ::= alternative ( | alternative )*
            let (lhs_name, _line, _col) = match self.advance() {
                Token::NonTerminal { name, line, col } => (name.clone(), *line, *col),
                Token::Eof => break,
                tok => {
                    let (l, c) = tok.position();
                    return Err(BnfParseError::UnexpectedChar {
                        line: l,
                        col: c,
                        got: '?',
                        expected: "non-terminal name like `<rule>`".to_string(),
                    });
                }
            };

            let lhs_id = *nt_to_id.get(&lhs_name).expect("registered in pass 1");

            // Consume `::=`.
            match self.peek() {
                Token::Assign { .. } => {
                    self.advance();
                }
                tok => {
                    let (l, c) = tok.position();
                    return Err(BnfParseError::MissingDefinitionSeparator { line: l, col: c });
                }
            }

            defined_nts.insert(lhs_id);

            // Parse one or more alternatives separated by `|`.
            loop {
                let mut rhs: Vec<Symbol> = Vec::new();

                // Collect symbols until Pipe, Eof, or next LHS-followed-by-assign pattern.
                loop {
                    match self.peek() {
                        Token::Pipe { .. } => break,
                        Token::Eof => break,
                        Token::NonTerminal { name, line, col } => {
                            // Check if this NT is followed by `::=` — if so it is a
                            // new rule definition, not a symbol in the current rhs.
                            let name = name.clone();
                            let (l, c) = (*line, *col);
                            let next_is_assign =
                                matches!(self.tokens.get(self.pos + 1), Some(Token::Assign { .. }));
                            if next_is_assign && !rhs.is_empty() {
                                // We've completed the rhs and are now looking at a new rule.
                                break;
                            } else if next_is_assign && rhs.is_empty() {
                                // Empty alternative before a new rule — treat as epsilon.
                                break;
                            } else {
                                let nt_id = match nt_to_id.get(&name) {
                                    Some(&id) => id,
                                    None => {
                                        return Err(BnfParseError::UndefinedNonTerminal { name });
                                    }
                                };
                                // Consume the token.
                                self.advance();
                                let _ = (l, c);
                                rhs.push(Symbol::NonTerminal(nt_id));
                            }
                        }
                        Token::Terminal { bytes, .. } => {
                            let bytes = bytes.clone();
                            self.advance();
                            rhs.push(Symbol::Terminal(bytes));
                        }
                        Token::Assign { line, col } => {
                            let (l, c) = (*line, *col);
                            return Err(BnfParseError::UnexpectedChar {
                                line: l,
                                col: c,
                                got: ':',
                                expected: "a symbol or `|`".to_string(),
                            });
                        }
                    }
                }

                grammar.add_rule(Rule::new(lhs_id, rhs));

                match self.peek() {
                    Token::Pipe { .. } => {
                        self.advance(); // consume `|`
                                        // Continue parsing alternatives for the same lhs.
                    }
                    _ => break,
                }
            }
        }

        // Validate: every NT used on any rhs must have been defined.
        for rule in &grammar.rules {
            for symbol in &rule.rhs {
                if let Symbol::NonTerminal(id) = symbol {
                    if !defined_nts.contains(id) {
                        let name = grammar.nt_name(*id).to_string();
                        return Err(BnfParseError::UndefinedNonTerminal { name });
                    }
                }
            }
        }

        Ok(grammar)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a BNF grammar string and return a [`Grammar`].
///
/// The grammar must contain at least one rule.  The start symbol is the LHS
/// of the **first** rule encountered in the input.
///
/// # Errors
///
/// Returns a [`BnfParseError`] if the input is syntactically invalid or if a
/// non-terminal is referenced but never defined.
pub fn parse_bnf(input: &str) -> Result<Grammar, BnfParseError> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenise()?;
    let mut parser = Parser::new(tokens);
    parser.parse()
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Happy-path parsing ──────────────────────────────────────────────────

    #[test]
    fn parse_single_terminal_rule() {
        let g = parse_bnf(r#"<S> ::= "hello""#).expect("valid");
        assert_eq!(g.rules.len(), 1);
        assert_eq!(g.rules[0].rhs.len(), 1);
        assert_eq!(g.rules[0].rhs[0], Symbol::Terminal(b"hello".to_vec()));
    }

    #[test]
    fn parse_alternation() {
        let g = parse_bnf(r#"<S> ::= "a" | "b" | "c""#).expect("valid");
        assert_eq!(g.rules.len(), 3);
    }

    #[test]
    fn parse_sequential_symbols() {
        let g = parse_bnf(r#"<S> ::= "(" <S> ")""#).expect("valid");
        assert_eq!(g.rules.len(), 1);
        // rhs should be: Terminal("("), NonTerminal(S), Terminal(")")
        assert_eq!(g.rules[0].rhs.len(), 3);
        assert!(matches!(g.rules[0].rhs[0], Symbol::Terminal(ref b) if b == b"("));
        assert!(matches!(g.rules[0].rhs[1], Symbol::NonTerminal(_)));
        assert!(matches!(g.rules[0].rhs[2], Symbol::Terminal(ref b) if b == b")"));
    }

    #[test]
    fn parse_multi_rule_same_nt() {
        let g = parse_bnf(
            r#"
            <S> ::= "x" <S>
            <S> ::= "y"
        "#,
        )
        .expect("valid");
        assert_eq!(g.rules.len(), 2);
        assert_eq!(g.rules[0].lhs, g.rules[1].lhs);
    }

    #[test]
    fn parse_comments() {
        let g = parse_bnf(
            r#"
            # This is a comment
            <S> ::= "a" # trailing comment
        "#,
        )
        .expect("valid");
        assert_eq!(g.rules.len(), 1);
    }

    #[test]
    fn parse_escape_sequences() {
        let g = parse_bnf(r#"<S> ::= "\n\r\t\"\\""#).expect("valid");
        assert_eq!(g.rules[0].rhs[0], Symbol::Terminal(b"\n\r\t\"\\".to_vec()));
    }

    #[test]
    fn parse_multi_rule_alternation_and_sequence() {
        // expr  ::= term "+" expr | term
        // term  ::= "0" | "1"
        let g = parse_bnf(
            r#"
            <expr> ::= <term> "+" <expr> | <term>
            <term> ::= "0" | "1"
        "#,
        )
        .expect("valid");

        // 4 rules total: 2 for expr, 2 for term
        assert_eq!(g.rules.len(), 4);
    }

    #[test]
    fn parse_start_symbol_is_first_rule_lhs() {
        let g = parse_bnf(
            r#"
            <first> ::= "a"
            <second> ::= "b"
        "#,
        )
        .expect("valid");
        let start_name = g.nt_name(g.start()).to_string();
        assert_eq!(start_name, "first");
    }

    // ── Error cases ─────────────────────────────────────────────────────────

    #[test]
    fn error_empty_input() {
        assert!(matches!(parse_bnf(""), Err(BnfParseError::EmptyInput)));
        assert!(matches!(
            parse_bnf("   # only a comment\n"),
            Err(BnfParseError::EmptyInput)
        ));
    }

    #[test]
    fn error_missing_separator() {
        // `<S>` present but no `::=`
        let err = parse_bnf(r#"<S> = "a""#);
        assert!(
            matches!(err, Err(BnfParseError::MissingDefinitionSeparator { .. })),
            "expected MissingDefinitionSeparator, got {err:?}"
        );
    }

    #[test]
    fn error_unterminated_string() {
        let err = parse_bnf(r#"<S> ::= "unterminated"#);
        assert!(
            matches!(err, Err(BnfParseError::UnterminatedString { .. })),
            "expected UnterminatedString, got {err:?}"
        );
    }

    #[test]
    fn error_empty_nonterminal_name() {
        let err = parse_bnf(r#"<> ::= "a""#);
        assert!(
            matches!(err, Err(BnfParseError::EmptyNonTerminalName { .. })),
            "expected EmptyNonTerminalName, got {err:?}"
        );
    }

    #[test]
    fn error_undefined_nonterminal() {
        let err = parse_bnf(r#"<S> ::= <T>"#);
        assert!(
            matches!(err, Err(BnfParseError::UndefinedNonTerminal { .. })),
            "expected UndefinedNonTerminal, got {err:?}"
        );
    }

    #[test]
    fn error_display_implementations() {
        let e = BnfParseError::EmptyInput;
        assert!(!e.to_string().is_empty());

        let e2 = BnfParseError::UnterminatedString { line: 1, col: 5 };
        let s = e2.to_string();
        assert!(s.contains('1') && s.contains('5'));

        let e3 = BnfParseError::UndefinedNonTerminal { name: "foo".into() };
        assert!(e3.to_string().contains("foo"));
    }

    #[test]
    fn parse_hex_escape() {
        // "\x41" is ASCII 'A' (0x41).
        let g = parse_bnf(r#"<S> ::= "\x41""#).expect("valid");
        assert_eq!(g.rules[0].rhs[0], Symbol::Terminal(vec![0x41]));
    }
}
