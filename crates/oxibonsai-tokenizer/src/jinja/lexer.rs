//! Lexer for the Jinja subset: `{{ … }}`, `{% … %}` and `{# … #}` with the
//! `-` / `+` whitespace-control markers.
//!
//! The lexer is **total**: every input either produces a token stream or a
//! [`JinjaError::Syntax`] carrying a 1-based line and column.  It never
//! panics and never silently drops malformed input.
//!
//! Whitespace handling reproduces the environment HuggingFace `transformers`
//! and the llama.cpp/minja chat-template renderer both use:
//! `trim_blocks = true`, `lstrip_blocks = true`, `keep_trailing_newline =
//! false`, and all newlines normalised to `\n`.

use super::{JinjaError, JinjaOptions};

/// A lexical token.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// Literal template text (already whitespace-adjusted).
    Text(String),
    /// `{{`
    VarStart,
    /// `}}`
    VarEnd,
    /// `{%`
    BlockStart,
    /// `%}`
    BlockEnd,
    /// An identifier or keyword.
    Name(String),
    /// A string literal (escapes already decoded).
    Str(String),
    /// An integer literal.
    Int(i64),
    /// A float literal.
    Float(f64),
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `**`
    Power,
    /// `/`
    Slash,
    /// `//`
    DoubleSlash,
    /// `%`
    Percent,
    /// `~`
    Tilde,
    /// `==`
    EqEq,
    /// `!=`
    NotEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
    /// `=`
    Assign,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `,`
    Comma,
    /// `:`
    Colon,
    /// `.`
    Dot,
    /// `|`
    Pipe,
    /// End of input.
    Eof,
}

impl TokenKind {
    /// A human-readable spelling for error messages.
    pub fn describe(&self) -> String {
        match self {
            TokenKind::Text(_) => "template text".to_string(),
            TokenKind::VarStart => "'{{'".to_string(),
            TokenKind::VarEnd => "'}}'".to_string(),
            TokenKind::BlockStart => "'{%'".to_string(),
            TokenKind::BlockEnd => "'%}'".to_string(),
            TokenKind::Name(n) => format!("'{n}'"),
            TokenKind::Str(_) => "a string literal".to_string(),
            TokenKind::Int(i) => format!("'{i}'"),
            TokenKind::Float(f) => format!("'{f}'"),
            TokenKind::Eof => "end of template".to_string(),
            other => format!("'{}'", other.punctuation().unwrap_or("?")),
        }
    }

    fn punctuation(&self) -> Option<&'static str> {
        Some(match self {
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Power => "**",
            TokenKind::Slash => "/",
            TokenKind::DoubleSlash => "//",
            TokenKind::Percent => "%",
            TokenKind::Tilde => "~",
            TokenKind::EqEq => "==",
            TokenKind::NotEq => "!=",
            TokenKind::Lt => "<",
            TokenKind::LtEq => "<=",
            TokenKind::Gt => ">",
            TokenKind::GtEq => ">=",
            TokenKind::Assign => "=",
            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::Comma => ",",
            TokenKind::Colon => ":",
            TokenKind::Dot => ".",
            TokenKind::Pipe => "|",
            _ => return None,
        })
    }
}

/// A token together with its 1-based source position.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// What was lexed.
    pub kind: TokenKind,
    /// 1-based line.
    pub line: usize,
    /// 1-based column.
    pub column: usize,
}

/// Which delimiter pair the lexer is currently inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagKind {
    Variable,
    Block,
}

impl TagKind {
    fn closing(self) -> (char, char) {
        match self {
            TagKind::Variable => ('}', '}'),
            TagKind::Block => ('%', '}'),
        }
    }

    fn opening(self) -> &'static str {
        match self {
            TagKind::Variable => "{{",
            TagKind::Block => "{%",
        }
    }
}

/// The lexer state.
pub struct Lexer<'a> {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
    opts: &'a JinjaOptions,
}

impl<'a> Lexer<'a> {
    /// Create a lexer over `source`.
    ///
    /// Newlines are normalised to `\n` and — unless
    /// [`JinjaOptions::keep_trailing_newline`] is set — a single trailing
    /// newline is removed, exactly like Jinja's own lexer.
    pub fn new(source: &str, opts: &'a JinjaOptions) -> Self {
        let mut normalised = String::with_capacity(source.len());
        let mut it = source.chars().peekable();
        while let Some(c) = it.next() {
            if c == '\r' {
                if it.peek() == Some(&'\n') {
                    it.next();
                }
                normalised.push('\n');
            } else {
                normalised.push(c);
            }
        }
        if !opts.keep_trailing_newline && normalised.ends_with('\n') {
            normalised.pop();
        }
        Self {
            chars: normalised.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
            opts,
        }
    }

    /// Tokenize the whole template.
    pub fn tokenize(mut self) -> Result<Vec<Token>, JinjaError> {
        let mut tokens: Vec<Token> = Vec::new();
        let mut text = String::new();
        let mut text_line = self.line;
        let mut text_column = self.column;
        // Whether the pending text begins at the start of a physical line —
        // `lstrip_blocks` only strips indentation when nothing else precedes
        // the tag on its line.
        let mut text_at_line_start = true;
        // Set after a tag closed with `-`; strips the leading whitespace of
        // the text that follows.
        let mut strip_leading_ws = false;
        // Set after a block/comment tag closed without `+`; with
        // `trim_blocks` it swallows the single newline right after the tag.
        let mut trim_leading_newline = false;

        while let Some(c) = self.peek(0) {
            if c == '{' {
                if let Some(next) = self.peek(1) {
                    if next == '{' || next == '%' || next == '#' {
                        let marker = self.peek(2);
                        apply_left_trim(&mut text, marker, next, self.opts, text_at_line_start);
                        if !text.is_empty() {
                            tokens.push(Token {
                                kind: TokenKind::Text(std::mem::take(&mut text)),
                                line: text_line,
                                column: text_column,
                            });
                        }
                        text.clear();

                        let (tag_line, tag_column) = (self.line, self.column);
                        self.bump();
                        self.bump();
                        if marker == Some('-') || marker == Some('+') {
                            self.bump();
                        }
                        let trims = if next == '#' {
                            self.lex_comment(tag_line, tag_column)?
                        } else {
                            let kind = if next == '{' {
                                TagKind::Variable
                            } else {
                                TagKind::Block
                            };
                            self.lex_tag(kind, tag_line, tag_column, &mut tokens)?
                        };
                        strip_leading_ws = trims.strip_ws;
                        trim_leading_newline = trims.trim_newline;
                        text_line = self.line;
                        text_column = self.column;
                        text_at_line_start = self.column == 1;
                        continue;
                    }
                }
            }

            if trim_leading_newline {
                trim_leading_newline = false;
                if c == '\n' {
                    self.bump();
                    text_line = self.line;
                    text_column = self.column;
                    text_at_line_start = self.column == 1;
                    continue;
                }
            }
            if strip_leading_ws {
                if c.is_whitespace() {
                    self.bump();
                    text_line = self.line;
                    text_column = self.column;
                    text_at_line_start = self.column == 1;
                    continue;
                }
                strip_leading_ws = false;
            }

            text.push(c);
            self.bump();
        }

        if !text.is_empty() {
            tokens.push(Token {
                kind: TokenKind::Text(text),
                line: text_line,
                column: text_column,
            });
        }
        tokens.push(Token {
            kind: TokenKind::Eof,
            line: self.line,
            column: self.column,
        });
        Ok(tokens)
    }

    fn peek(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if let Some(c) = c {
            self.pos += 1;
            if c == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
        c
    }

    fn syntax(&self, message: impl Into<String>) -> JinjaError {
        JinjaError::Syntax {
            line: self.line,
            column: self.column,
            message: message.into(),
        }
    }

    fn syntax_at(&self, line: usize, column: usize, message: impl Into<String>) -> JinjaError {
        JinjaError::Syntax {
            line,
            column,
            message: message.into(),
        }
    }

    /// Consume `{# … #}`; returns the trailing whitespace behaviour.
    fn lex_comment(&mut self, line: usize, column: usize) -> Result<TrimSpec, JinjaError> {
        loop {
            let Some(c) = self.peek(0) else {
                return Err(self.syntax_at(line, column, "unterminated comment: missing '#}'"));
            };
            if c == '#' && self.peek(1) == Some('}') {
                self.bump();
                self.bump();
                return Ok(TrimSpec {
                    strip_ws: false,
                    trim_newline: self.opts.trim_blocks,
                });
            }
            if (c == '-' || c == '+') && self.peek(1) == Some('#') && self.peek(2) == Some('}') {
                let minus = c == '-';
                self.bump();
                self.bump();
                self.bump();
                return Ok(TrimSpec {
                    strip_ws: minus,
                    trim_newline: !minus && self.opts.trim_blocks,
                });
            }
            self.bump();
        }
    }

    /// Consume the body of a `{{ … }}` or `{% … %}` tag, pushing tokens.
    fn lex_tag(
        &mut self,
        kind: TagKind,
        line: usize,
        column: usize,
        tokens: &mut Vec<Token>,
    ) -> Result<TrimSpec, JinjaError> {
        tokens.push(Token {
            kind: match kind {
                TagKind::Variable => TokenKind::VarStart,
                TagKind::Block => TokenKind::BlockStart,
            },
            line,
            column,
        });
        let (c0, c1) = kind.closing();
        let mut depth: usize = 0;

        loop {
            while self.peek(0).map(char::is_whitespace).unwrap_or(false) {
                self.bump();
            }
            let Some(c) = self.peek(0) else {
                return Err(self.syntax_at(
                    line,
                    column,
                    format!("unterminated '{}' tag", kind.opening()),
                ));
            };

            if depth == 0 {
                if c == c0 && self.peek(1) == Some(c1) {
                    let (l, col) = (self.line, self.column);
                    self.bump();
                    self.bump();
                    tokens.push(Token {
                        kind: match kind {
                            TagKind::Variable => TokenKind::VarEnd,
                            TagKind::Block => TokenKind::BlockEnd,
                        },
                        line: l,
                        column: col,
                    });
                    return Ok(TrimSpec {
                        strip_ws: false,
                        trim_newline: kind == TagKind::Block && self.opts.trim_blocks,
                    });
                }
                if (c == '-' || c == '+') && self.peek(1) == Some(c0) && self.peek(2) == Some(c1) {
                    let minus = c == '-';
                    let (l, col) = (self.line, self.column);
                    self.bump();
                    self.bump();
                    self.bump();
                    tokens.push(Token {
                        kind: match kind {
                            TagKind::Variable => TokenKind::VarEnd,
                            TagKind::Block => TokenKind::BlockEnd,
                        },
                        line: l,
                        column: col,
                    });
                    return Ok(TrimSpec {
                        strip_ws: minus,
                        trim_newline: !minus && kind == TagKind::Block && self.opts.trim_blocks,
                    });
                }
            }

            let token = self.lex_token(&mut depth)?;
            tokens.push(token);
        }
    }

    fn lex_token(&mut self, depth: &mut usize) -> Result<Token, JinjaError> {
        let (line, column) = (self.line, self.column);
        let Some(c) = self.peek(0) else {
            return Err(self.syntax("unexpected end of template"));
        };

        let kind = if c == '\'' || c == '"' {
            TokenKind::Str(self.lex_string(c)?)
        } else if c.is_ascii_digit() {
            self.lex_number()?
        } else if c.is_alphabetic() || c == '_' {
            let mut name = String::new();
            while let Some(c) = self.peek(0) {
                if c.is_alphanumeric() || c == '_' {
                    name.push(c);
                    self.bump();
                } else {
                    break;
                }
            }
            TokenKind::Name(name)
        } else {
            self.lex_punctuation(depth)?
        };

        Ok(Token { kind, line, column })
    }

    fn lex_punctuation(&mut self, depth: &mut usize) -> Result<TokenKind, JinjaError> {
        let (line, column) = (self.line, self.column);
        let Some(c) = self.bump() else {
            return Err(self.syntax("unexpected end of template"));
        };
        let two = self.peek(0);
        let kind = match (c, two) {
            ('*', Some('*')) => {
                self.bump();
                TokenKind::Power
            }
            ('/', Some('/')) => {
                self.bump();
                TokenKind::DoubleSlash
            }
            ('=', Some('=')) => {
                self.bump();
                TokenKind::EqEq
            }
            ('!', Some('=')) => {
                self.bump();
                TokenKind::NotEq
            }
            ('<', Some('=')) => {
                self.bump();
                TokenKind::LtEq
            }
            ('>', Some('=')) => {
                self.bump();
                TokenKind::GtEq
            }
            ('+', _) => TokenKind::Plus,
            ('-', _) => TokenKind::Minus,
            ('*', _) => TokenKind::Star,
            ('/', _) => TokenKind::Slash,
            ('%', _) => TokenKind::Percent,
            ('~', _) => TokenKind::Tilde,
            ('<', _) => TokenKind::Lt,
            ('>', _) => TokenKind::Gt,
            ('=', _) => TokenKind::Assign,
            (',', _) => TokenKind::Comma,
            (':', _) => TokenKind::Colon,
            ('.', _) => TokenKind::Dot,
            ('|', _) => TokenKind::Pipe,
            ('(', _) => {
                *depth += 1;
                TokenKind::LParen
            }
            ('[', _) => {
                *depth += 1;
                TokenKind::LBracket
            }
            ('{', _) => {
                *depth += 1;
                TokenKind::LBrace
            }
            (')', _) => {
                *depth = depth.saturating_sub(1);
                TokenKind::RParen
            }
            (']', _) => {
                *depth = depth.saturating_sub(1);
                TokenKind::RBracket
            }
            ('}', _) => {
                *depth = depth.saturating_sub(1);
                TokenKind::RBrace
            }
            (other, _) => {
                return Err(self.syntax_at(line, column, format!("unexpected character {other:?}")))
            }
        };
        Ok(kind)
    }

    fn lex_number(&mut self) -> Result<TokenKind, JinjaError> {
        let (line, column) = (self.line, self.column);
        let mut digits = String::new();
        let mut is_float = false;
        while let Some(c) = self.peek(0) {
            if c.is_ascii_digit() {
                digits.push(c);
                self.bump();
            } else if c == '_' {
                self.bump();
            } else {
                break;
            }
        }
        if self.peek(0) == Some('.') && self.peek(1).map(|c| c.is_ascii_digit()).unwrap_or(false) {
            is_float = true;
            digits.push('.');
            self.bump();
            while let Some(c) = self.peek(0) {
                if c.is_ascii_digit() {
                    digits.push(c);
                    self.bump();
                } else if c == '_' {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        if matches!(self.peek(0), Some('e') | Some('E')) {
            let sign_offset = usize::from(matches!(self.peek(1), Some('+') | Some('-')));
            if self
                .peek(1 + sign_offset)
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
            {
                is_float = true;
                digits.push('e');
                self.bump();
                if sign_offset == 1 {
                    if let Some(sign) = self.bump() {
                        digits.push(sign);
                    }
                }
                while let Some(c) = self.peek(0) {
                    if c.is_ascii_digit() {
                        digits.push(c);
                        self.bump();
                    } else {
                        break;
                    }
                }
            }
        }

        if is_float {
            digits
                .parse::<f64>()
                .map(TokenKind::Float)
                .map_err(|e| self.syntax_at(line, column, format!("invalid float literal: {e}")))
        } else {
            match digits.parse::<i64>() {
                Ok(i) => Ok(TokenKind::Int(i)),
                // Python has arbitrary-precision integers; a literal too large
                // for i64 degrades to a float rather than failing the render.
                Err(_) => digits.parse::<f64>().map(TokenKind::Float).map_err(|e| {
                    self.syntax_at(line, column, format!("invalid integer literal: {e}"))
                }),
            }
        }
    }

    fn lex_string(&mut self, quote: char) -> Result<String, JinjaError> {
        let (line, column) = (self.line, self.column);
        self.bump();
        let mut out = String::new();
        loop {
            let Some(c) = self.bump() else {
                return Err(self.syntax_at(line, column, "unterminated string literal"));
            };
            if c == quote {
                return Ok(out);
            }
            if c != '\\' {
                out.push(c);
                continue;
            }
            let Some(esc) = self.bump() else {
                return Err(self.syntax_at(line, column, "unterminated string literal"));
            };
            match esc {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                '0' => out.push('\0'),
                'a' => out.push('\u{7}'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'v' => out.push('\u{b}'),
                '\\' => out.push('\\'),
                '\'' => out.push('\''),
                '"' => out.push('"'),
                '\n' => {}
                'x' => self.push_hex_escape(&mut out, 2, line, column)?,
                'u' => self.push_hex_escape(&mut out, 4, line, column)?,
                'U' => self.push_hex_escape(&mut out, 8, line, column)?,
                // Python keeps an unrecognised escape verbatim.
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
        }
    }

    fn push_hex_escape(
        &mut self,
        out: &mut String,
        width: usize,
        line: usize,
        column: usize,
    ) -> Result<(), JinjaError> {
        let mut hex = String::with_capacity(width);
        for _ in 0..width {
            let Some(c) = self.peek(0) else {
                return Err(self.syntax_at(line, column, "truncated escape sequence"));
            };
            if !c.is_ascii_hexdigit() {
                return Err(self.syntax_at(
                    line,
                    column,
                    format!("invalid hex digit {c:?} in escape sequence"),
                ));
            }
            hex.push(c);
            self.bump();
        }
        let code = u32::from_str_radix(&hex, 16)
            .map_err(|e| self.syntax_at(line, column, format!("invalid escape sequence: {e}")))?;
        let ch = char::from_u32(code).ok_or_else(|| {
            self.syntax_at(line, column, format!("invalid code point U+{code:04X}"))
        })?;
        out.push(ch);
        Ok(())
    }
}

/// How the text after a tag must be trimmed.
struct TrimSpec {
    strip_ws: bool,
    trim_newline: bool,
}

/// Apply the left-hand whitespace rules to the text preceding a tag.
///
/// `at_line_start` says whether the pending text itself began at column 1;
/// `lstrip_blocks` may only strip indentation when the tag is the first
/// non-whitespace thing on its line, so `{{ a }}   {% if %}` keeps its spaces.
fn apply_left_trim(
    text: &mut String,
    marker: Option<char>,
    tag: char,
    opts: &JinjaOptions,
    at_line_start: bool,
) {
    if marker == Some('-') {
        let trimmed = text.trim_end().len();
        text.truncate(trimmed);
        return;
    }
    if marker == Some('+') {
        return;
    }
    if opts.lstrip_blocks && tag != '{' {
        let stripped = text.trim_end_matches([' ', '\t']);
        if stripped.ends_with('\n') || (stripped.is_empty() && at_line_start) {
            let len = stripped.len();
            text.truncate(len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Result<Vec<TokenKind>, JinjaError> {
        let opts = JinjaOptions::default();
        Ok(Lexer::new(src, &opts)
            .tokenize()?
            .into_iter()
            .map(|t| t.kind)
            .collect())
    }

    #[test]
    fn plain_text_is_one_token() {
        assert_eq!(
            lex("hello").expect("lex"),
            vec![TokenKind::Text("hello".into()), TokenKind::Eof]
        );
    }

    #[test]
    fn trailing_newline_is_dropped_once() {
        assert_eq!(
            lex("line\n\n").expect("lex"),
            vec![TokenKind::Text("line\n".into()), TokenKind::Eof]
        );
    }

    #[test]
    fn variable_tag_tokens() {
        assert_eq!(
            lex("{{ a.b }}").expect("lex"),
            vec![
                TokenKind::VarStart,
                TokenKind::Name("a".into()),
                TokenKind::Dot,
                TokenKind::Name("b".into()),
                TokenKind::VarEnd,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn dict_literal_does_not_close_the_tag_early() {
        let kinds = lex("{{ {'a': 1} }}").expect("lex");
        assert_eq!(kinds.first(), Some(&TokenKind::VarStart));
        assert_eq!(kinds[kinds.len() - 2], TokenKind::VarEnd);
        assert!(kinds.contains(&TokenKind::LBrace));
        assert!(kinds.contains(&TokenKind::RBrace));
    }

    #[test]
    fn percent_inside_variable_tag_is_an_operator() {
        assert_eq!(
            lex("{{ 5 % 2 }}").expect("lex"),
            vec![
                TokenKind::VarStart,
                TokenKind::Int(5),
                TokenKind::Percent,
                TokenKind::Int(2),
                TokenKind::VarEnd,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn comments_are_dropped() {
        assert_eq!(
            lex("a{# x\ny #}b").expect("lex"),
            vec![
                TokenKind::Text("a".into()),
                TokenKind::Text("b".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn whitespace_control_markers() {
        assert_eq!(
            lex("a   {{- v -}}   b").expect("lex"),
            vec![
                TokenKind::Text("a".into()),
                TokenKind::VarStart,
                TokenKind::Name("v".into()),
                TokenKind::VarEnd,
                TokenKind::Text("b".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn trim_blocks_swallows_the_newline_after_a_block_tag() {
        assert_eq!(
            lex("{% if a %}\nX{% endif %}\n").expect("lex"),
            vec![
                TokenKind::BlockStart,
                TokenKind::Name("if".into()),
                TokenKind::Name("a".into()),
                TokenKind::BlockEnd,
                TokenKind::Text("X".into()),
                TokenKind::BlockStart,
                TokenKind::Name("endif".into()),
                TokenKind::BlockEnd,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn lstrip_blocks_removes_indentation_before_a_block_tag() {
        assert_eq!(
            lex("x\n   {% if a %}y{% endif %}").expect("lex"),
            vec![
                TokenKind::Text("x\n".into()),
                TokenKind::BlockStart,
                TokenKind::Name("if".into()),
                TokenKind::Name("a".into()),
                TokenKind::BlockEnd,
                TokenKind::Text("y".into()),
                TokenKind::BlockStart,
                TokenKind::Name("endif".into()),
                TokenKind::BlockEnd,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn lstrip_blocks_does_not_apply_to_variable_tags() {
        let kinds = lex("x\n   {{ a }}").expect("lex");
        assert_eq!(kinds.first(), Some(&TokenKind::Text("x\n   ".into())));
    }

    #[test]
    fn string_escapes_are_decoded() {
        assert_eq!(
            lex(r#"{{ 'a\nbé\x41\\' }}"#).expect("lex")[1],
            TokenKind::Str("a\nbéA\\".into())
        );
    }

    #[test]
    fn numbers() {
        assert_eq!(
            lex("{{ 1 1.5 1e3 1_000 }}").expect("lex")[1..5],
            [
                TokenKind::Int(1),
                TokenKind::Float(1.5),
                TokenKind::Float(1000.0),
                TokenKind::Int(1000)
            ]
        );
    }

    #[test]
    fn unterminated_constructs_error_with_position() {
        for src in ["{{ a", "{% if a", "{# c", "{{ 'abc }}", "{{ @ }}"] {
            let err = lex(src).expect_err("must not lex");
            match err {
                JinjaError::Syntax { line, column, .. } => {
                    assert!(line >= 1 && column >= 1, "bad position for {src:?}");
                }
                other => panic!("expected a syntax error for {src:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn crlf_is_normalised() {
        assert_eq!(
            lex("a\r\nb").expect("lex"),
            vec![TokenKind::Text("a\nb".into()), TokenKind::Eof]
        );
    }
}
