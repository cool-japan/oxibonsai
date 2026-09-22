//! Recursive-descent parser for the Jinja subset.
//!
//! The expression grammar mirrors Jinja's own precedence chain exactly, which
//! matters for real chat templates:
//!
//! ```text
//! condexpr := or ['if' or ['else' condexpr]]
//! or       := and ('or' and)*
//! and      := not ('and' not)*
//! not      := 'not' not | compare
//! compare  := math1 (('=='|'!='|'<'|'<='|'>'|'>='|'in'|'not in') math1)*
//! math1    := concat (('+'|'-') concat)*
//! concat   := math2 ('~' math2)*
//! math2    := pow (('*'|'/'|'//'|'%') pow)*
//! pow      := unary ('**' pow)*
//! unary    := ('-'|'+') unary | primary postfix* (filter | test)*
//! ```
//!
//! So `a | string if a is string else a | tojson | safe` binds as
//! `(a|string) if (a is string) else (a|tojson|safe)` — the line Bonsai 2's
//! template uses to serialise tool-call arguments.
//!
//! Every construct outside the subset — `{% include %}`, `{% extends %}`,
//! `{% raw %}`, an unknown filter, an unknown test, an unknown statement —
//! is rejected here with a [`JinjaError::Syntax`] carrying a line and column,
//! so it can never reach the evaluator and silently render as nothing.

use std::sync::Arc;

use super::ast::{
    BinOp, CallBlock, CmpOp, Expr, ForLoop, MacroDef, MacroParam, Node, SetTarget, UnaryOp,
};
use super::lexer::{Token, TokenKind};
use super::{JinjaError, JinjaOptions};

/// Filters the engine implements.  Anything else is a syntax error.
pub const SUPPORTED_FILTERS: &[&str] = &[
    "default",
    "first",
    "int",
    "items",
    "join",
    "last",
    "length",
    "list",
    "lower",
    "map",
    "replace",
    "safe",
    "selectattr",
    "string",
    "tojson",
    "trim",
    "upper",
];

/// Tests the engine implements.  Anything else is a syntax error.
pub const SUPPORTED_TESTS: &[&str] = &[
    "boolean",
    "defined",
    "false",
    "integer",
    "iterable",
    "mapping",
    "none",
    "number",
    "sequence",
    "string",
    "true",
    "undefined",
];

/// Statements that exist in Jinja but are deliberately out of scope; naming
/// them explicitly produces a better diagnostic than "unknown statement".
const UNSUPPORTED_STATEMENTS: &[&str] = &[
    "autoescape",
    "block",
    "do",
    "extends",
    "filter",
    "from",
    "import",
    "include",
    "raw",
    "trans",
    "with",
];

/// The parser.
pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    depth: usize,
    loop_depth: usize,
    max_depth: usize,
}

impl<'a> Parser<'a> {
    /// Create a parser over a token stream.
    pub fn new(tokens: &'a [Token], opts: &JinjaOptions) -> Self {
        Self {
            tokens,
            pos: 0,
            depth: 0,
            loop_depth: 0,
            max_depth: opts.max_parse_depth,
        }
    }

    /// Parse a whole template.
    pub fn parse_template(mut self) -> Result<Vec<Node>, JinjaError> {
        let (nodes, terminator) = self.parse_nodes(&[])?;
        match terminator {
            None => Ok(nodes),
            Some(name) => Err(self.error_here(format!("unexpected '{name}' block"))),
        }
    }

    // ── token helpers ────────────────────────────────────────────────────

    fn peek(&self) -> &TokenKind {
        self.tokens
            .get(self.pos)
            .map(|t| &t.kind)
            .unwrap_or(&TokenKind::Eof)
    }

    fn peek_at(&self, offset: usize) -> &TokenKind {
        self.tokens
            .get(self.pos + offset)
            .map(|t| &t.kind)
            .unwrap_or(&TokenKind::Eof)
    }

    fn position(&self) -> (usize, usize) {
        match self.tokens.get(self.pos).or_else(|| self.tokens.last()) {
            Some(t) => (t.line, t.column),
            None => (1, 1),
        }
    }

    fn error_here(&self, message: impl Into<String>) -> JinjaError {
        let (line, column) = self.position();
        JinjaError::Syntax {
            line,
            column,
            message: message.into(),
        }
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<(), JinjaError> {
        if self.peek() == kind {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error_here(format!(
                "expected {}, found {}",
                kind.describe(),
                self.peek().describe()
            )))
        }
    }

    fn expect_name(&mut self) -> Result<String, JinjaError> {
        match self.peek().clone() {
            TokenKind::Name(n) => {
                self.pos += 1;
                Ok(n)
            }
            other => Err(self.error_here(format!("expected a name, found {}", other.describe()))),
        }
    }

    fn eat_keyword(&mut self, keyword: &str) -> bool {
        if matches!(self.peek(), TokenKind::Name(n) if n == keyword) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn at_keyword(&self, keyword: &str) -> bool {
        matches!(self.peek(), TokenKind::Name(n) if n == keyword)
    }

    fn enter(&mut self) -> Result<(), JinjaError> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(self.error_here(format!(
                "template nested deeper than {} levels",
                self.max_depth
            )));
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    // ── statements ───────────────────────────────────────────────────────

    /// Parse nodes until one of `terminators` (or EOF).
    ///
    /// On a terminator the `{%` and the tag name are consumed, and the name is
    /// returned so the caller can continue with the tag's own arguments.
    fn parse_nodes(
        &mut self,
        terminators: &[&str],
    ) -> Result<(Vec<Node>, Option<String>), JinjaError> {
        self.enter()?;
        let mut nodes = Vec::new();
        loop {
            match self.peek().clone() {
                TokenKind::Eof => {
                    self.leave();
                    return Ok((nodes, None));
                }
                TokenKind::Text(text) => {
                    self.pos += 1;
                    nodes.push(Node::Text(text));
                }
                TokenKind::VarStart => {
                    self.pos += 1;
                    let expr = self.parse_expression()?;
                    self.expect(&TokenKind::VarEnd)?;
                    nodes.push(Node::Output(expr));
                }
                TokenKind::BlockStart => {
                    let name = match self.peek_at(1) {
                        TokenKind::Name(n) => n.clone(),
                        other => {
                            let described = other.describe();
                            self.pos += 1;
                            return Err(self.error_here(format!(
                                "expected a statement name, found {described}"
                            )));
                        }
                    };
                    if terminators.contains(&name.as_str()) {
                        self.pos += 2;
                        self.leave();
                        return Ok((nodes, Some(name)));
                    }
                    self.pos += 2;
                    let node = self.parse_statement(&name)?;
                    nodes.push(node);
                }
                other => {
                    return Err(self
                        .error_here(format!("unexpected {} outside of a tag", other.describe())))
                }
            }
        }
    }

    fn parse_statement(&mut self, name: &str) -> Result<Node, JinjaError> {
        match name {
            "if" => self.parse_if(),
            "for" => self.parse_for(),
            "set" => self.parse_set(),
            "macro" => self.parse_macro(),
            "call" => self.parse_call_block(),
            "generation" => {
                self.expect(&TokenKind::BlockEnd)?;
                let (body, term) = self.parse_nodes(&["endgeneration"])?;
                self.close_block(term, "generation", "endgeneration")?;
                Ok(Node::Generation(body))
            }
            "break" | "continue" => {
                if self.loop_depth == 0 {
                    return Err(self.error_here(format!("'{name}' outside of a loop")));
                }
                self.expect(&TokenKind::BlockEnd)?;
                Ok(if name == "break" {
                    Node::Break
                } else {
                    Node::Continue
                })
            }
            other if UNSUPPORTED_STATEMENTS.contains(&other) => {
                Err(self.error_here(format!("'{other}' is not supported by this Jinja subset")))
            }
            other => Err(self.error_here(format!("unknown statement '{other}'"))),
        }
    }

    fn close_block(
        &mut self,
        terminator: Option<String>,
        opened: &str,
        expected: &str,
    ) -> Result<(), JinjaError> {
        match terminator {
            Some(t) if t == expected => self.expect(&TokenKind::BlockEnd),
            Some(t) => {
                Err(self.error_here(format!("unexpected '{t}' while looking for '{expected}'")))
            }
            None => Err(self.error_here(format!(
                "unterminated '{opened}' block: missing '{{% {expected} %}}'"
            ))),
        }
    }

    fn parse_if(&mut self) -> Result<Node, JinjaError> {
        let mut branches = Vec::new();
        let mut otherwise = Vec::new();
        loop {
            let cond = self.parse_expression()?;
            self.expect(&TokenKind::BlockEnd)?;
            let (body, term) = self.parse_nodes(&["elif", "else", "endif"])?;
            branches.push((cond, body));
            match term.as_deref() {
                Some("elif") => continue,
                Some("else") => {
                    self.expect(&TokenKind::BlockEnd)?;
                    let (body, term) = self.parse_nodes(&["endif"])?;
                    otherwise = body;
                    self.close_block(term, "if", "endif")?;
                    break;
                }
                Some("endif") => {
                    self.expect(&TokenKind::BlockEnd)?;
                    break;
                }
                Some(other) => {
                    return Err(
                        self.error_here(format!("unexpected '{other}' while looking for 'endif'"))
                    )
                }
                None => {
                    return Err(self.error_here("unterminated 'if' block: missing '{% endif %}'"))
                }
            }
        }
        Ok(Node::If {
            branches,
            otherwise,
        })
    }

    fn parse_for(&mut self) -> Result<Node, JinjaError> {
        let targets = self.parse_target_names()?;
        if !self.eat_keyword("in") {
            return Err(self.error_here("expected 'in' in a for statement"));
        }
        let iter = self.parse_iterable_expression()?;
        let cond = if self.eat_keyword("if") {
            Some(self.parse_expression_no_cond()?)
        } else {
            None
        };
        if self.at_keyword("recursive") {
            return Err(self.error_here("recursive loops are not supported by this Jinja subset"));
        }
        self.expect(&TokenKind::BlockEnd)?;
        self.loop_depth += 1;
        let (body, term) = self.parse_nodes(&["else", "endfor"])?;
        let mut otherwise = Vec::new();
        match term.as_deref() {
            Some("else") => {
                self.expect(&TokenKind::BlockEnd)?;
                let (else_body, term) = self.parse_nodes(&["endfor"])?;
                otherwise = else_body;
                self.loop_depth = self.loop_depth.saturating_sub(1);
                self.close_block(term, "for", "endfor")?;
            }
            Some("endfor") => {
                self.loop_depth = self.loop_depth.saturating_sub(1);
                self.expect(&TokenKind::BlockEnd)?;
            }
            other => {
                self.loop_depth = self.loop_depth.saturating_sub(1);
                return Err(match other {
                    Some(t) => {
                        self.error_here(format!("unexpected '{t}' while looking for 'endfor'"))
                    }
                    None => self.error_here("unterminated 'for' block: missing '{% endfor %}'"),
                });
            }
        }
        Ok(Node::For(Box::new(ForLoop {
            targets,
            iter,
            cond,
            body,
            otherwise,
        })))
    }

    fn parse_set(&mut self) -> Result<Node, JinjaError> {
        let first = self.expect_name()?;
        let target = if self.peek() == &TokenKind::Dot {
            self.pos += 1;
            let attr = self.expect_name()?;
            SetTarget::Attr { base: first, attr }
        } else if self.peek() == &TokenKind::Comma {
            let mut names = vec![first];
            while self.peek() == &TokenKind::Comma {
                self.pos += 1;
                names.push(self.expect_name()?);
            }
            SetTarget::Names(names)
        } else {
            SetTarget::Name(first)
        };

        if self.peek() == &TokenKind::Assign {
            self.pos += 1;
            let value = self.parse_tuple_expression()?;
            self.expect(&TokenKind::BlockEnd)?;
            return Ok(Node::Set { target, value });
        }

        if matches!(target, SetTarget::Names(_)) {
            return Err(self.error_here("a block '{% set %}' cannot unpack a tuple"));
        }
        self.expect(&TokenKind::BlockEnd)?;
        let (body, term) = self.parse_nodes(&["endset"])?;
        self.close_block(term, "set", "endset")?;
        Ok(Node::SetBlock { target, body })
    }

    fn parse_macro(&mut self) -> Result<Node, JinjaError> {
        let name = self.expect_name()?;
        let params = self.parse_macro_params()?;
        self.expect(&TokenKind::BlockEnd)?;
        let (body, term) = self.parse_nodes(&["endmacro"])?;
        self.close_block(term, "macro", "endmacro")?;
        Ok(Node::Macro(Arc::new(MacroDef { name, params, body })))
    }

    fn parse_call_block(&mut self) -> Result<Node, JinjaError> {
        let params = if self.peek() == &TokenKind::LParen {
            self.parse_macro_params()?
        } else {
            Vec::new()
        };
        let callee = self.parse_expression()?;
        if !matches!(callee, Expr::Call { .. }) {
            return Err(self.error_here("'{% call %}' requires a macro invocation"));
        }
        self.expect(&TokenKind::BlockEnd)?;
        let (body, term) = self.parse_nodes(&["endcall"])?;
        self.close_block(term, "call", "endcall")?;
        Ok(Node::CallBlock(Box::new(CallBlock {
            callee,
            params,
            body,
        })))
    }

    fn parse_macro_params(&mut self) -> Result<Vec<MacroParam>, JinjaError> {
        self.expect(&TokenKind::LParen)?;
        let mut params = Vec::new();
        while self.peek() != &TokenKind::RParen {
            let name = self.expect_name()?;
            let default = if self.peek() == &TokenKind::Assign {
                self.pos += 1;
                Some(self.parse_expression()?)
            } else {
                None
            };
            params.push(MacroParam { name, default });
            if self.peek() == &TokenKind::Comma {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok(params)
    }

    fn parse_target_names(&mut self) -> Result<Vec<String>, JinjaError> {
        let mut names = vec![self.expect_name()?];
        while self.peek() == &TokenKind::Comma {
            self.pos += 1;
            names.push(self.expect_name()?);
        }
        Ok(names)
    }

    // ── expressions ──────────────────────────────────────────────────────

    /// Parse a full expression, including the inline conditional.
    pub fn parse_expression(&mut self) -> Result<Expr, JinjaError> {
        self.enter()?;
        let expr = self.parse_cond_expr()?;
        self.leave();
        Ok(expr)
    }

    /// Parse an expression that stops before a trailing `if` — used for a
    /// `{% for x in xs if cond %}` iterable and for the loop filter itself.
    fn parse_expression_no_cond(&mut self) -> Result<Expr, JinjaError> {
        self.enter()?;
        let expr = self.parse_or()?;
        self.leave();
        Ok(expr)
    }

    /// Parse `a, b, c` into a tuple (a single expression stays scalar).
    fn parse_tuple_expression(&mut self) -> Result<Expr, JinjaError> {
        let first = self.parse_expression()?;
        if self.peek() != &TokenKind::Comma {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.peek() == &TokenKind::Comma {
            self.pos += 1;
            if self.peek() == &TokenKind::BlockEnd {
                break;
            }
            items.push(self.parse_expression()?);
        }
        Ok(Expr::Tuple(items))
    }

    fn parse_iterable_expression(&mut self) -> Result<Expr, JinjaError> {
        let first = self.parse_expression_no_cond()?;
        if self.peek() != &TokenKind::Comma {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.peek() == &TokenKind::Comma {
            self.pos += 1;
            items.push(self.parse_expression_no_cond()?);
        }
        Ok(Expr::Tuple(items))
    }

    fn parse_cond_expr(&mut self) -> Result<Expr, JinjaError> {
        let expr = self.parse_or()?;
        if !self.at_keyword("if") {
            return Ok(expr);
        }
        self.pos += 1;
        let cond = self.parse_or()?;
        let otherwise = if self.eat_keyword("else") {
            Some(Box::new(self.parse_cond_expr()?))
        } else {
            None
        };
        Ok(Expr::Cond {
            cond: Box::new(cond),
            then: Box::new(expr),
            otherwise,
        })
    }

    fn parse_or(&mut self) -> Result<Expr, JinjaError> {
        self.enter()?;
        let mut left = self.parse_and()?;
        while self.eat_keyword("or") {
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        self.leave();
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, JinjaError> {
        let mut left = self.parse_not()?;
        while self.eat_keyword("and") {
            let right = self.parse_not()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, JinjaError> {
        if self.at_keyword("not") {
            self.pos += 1;
            self.enter()?;
            let operand = self.parse_not()?;
            self.leave();
            return Ok(Expr::Unary {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            });
        }
        self.parse_compare()
    }

    fn parse_compare(&mut self) -> Result<Expr, JinjaError> {
        let lhs = self.parse_math1()?;
        let mut rest = Vec::new();
        loop {
            let op = match self.peek() {
                TokenKind::EqEq => CmpOp::Eq,
                TokenKind::NotEq => CmpOp::Ne,
                TokenKind::Lt => CmpOp::Lt,
                TokenKind::LtEq => CmpOp::Le,
                TokenKind::Gt => CmpOp::Gt,
                TokenKind::GtEq => CmpOp::Ge,
                TokenKind::Name(n) if n == "in" => CmpOp::In,
                TokenKind::Name(n) if n == "not" => {
                    if matches!(self.peek_at(1), TokenKind::Name(m) if m == "in") {
                        self.pos += 1;
                        CmpOp::NotIn
                    } else {
                        break;
                    }
                }
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_math1()?;
            rest.push((op, rhs));
        }
        if rest.is_empty() {
            Ok(lhs)
        } else {
            Ok(Expr::Compare {
                lhs: Box::new(lhs),
                rest,
            })
        }
    }

    fn parse_math1(&mut self) -> Result<Expr, JinjaError> {
        let mut left = self.parse_concat()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let right = self.parse_concat()?;
            left = Expr::Binary {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_concat(&mut self) -> Result<Expr, JinjaError> {
        let mut left = self.parse_math2()?;
        while self.peek() == &TokenKind::Tilde {
            self.pos += 1;
            let right = self.parse_math2()?;
            left = Expr::Binary {
                op: BinOp::Concat,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_math2(&mut self) -> Result<Expr, JinjaError> {
        let mut left = self.parse_pow()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::DoubleSlash => BinOp::FloorDiv,
                TokenKind::Percent => BinOp::Mod,
                _ => break,
            };
            self.pos += 1;
            let right = self.parse_pow()?;
            left = Expr::Binary {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_pow(&mut self) -> Result<Expr, JinjaError> {
        let left = self.parse_unary(true)?;
        if self.peek() == &TokenKind::Power {
            self.pos += 1;
            self.enter()?;
            let right = self.parse_pow()?;
            self.leave();
            return Ok(Expr::Binary {
                op: BinOp::Pow,
                lhs: Box::new(left),
                rhs: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_unary(&mut self, with_filter: bool) -> Result<Expr, JinjaError> {
        self.enter()?;
        let mut node = match self.peek() {
            TokenKind::Minus => {
                self.pos += 1;
                Expr::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(self.parse_unary(false)?),
                }
            }
            TokenKind::Plus => {
                self.pos += 1;
                Expr::Unary {
                    op: UnaryOp::Pos,
                    operand: Box::new(self.parse_unary(false)?),
                }
            }
            _ => self.parse_primary()?,
        };
        node = self.parse_postfix(node)?;
        if with_filter {
            node = self.parse_filter_expr(node)?;
        }
        self.leave();
        Ok(node)
    }

    fn parse_postfix(&mut self, mut node: Expr) -> Result<Expr, JinjaError> {
        loop {
            match self.peek() {
                TokenKind::Dot => {
                    self.pos += 1;
                    let name = self.expect_name()?;
                    node = Expr::GetAttr {
                        obj: Box::new(node),
                        name,
                    };
                }
                TokenKind::LBracket => {
                    self.pos += 1;
                    node = self.parse_subscript(node)?;
                }
                TokenKind::LParen => {
                    let (args, kwargs) = self.parse_call_args()?;
                    node = Expr::Call {
                        callee: Box::new(node),
                        args,
                        kwargs,
                    };
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn parse_filter_expr(&mut self, mut node: Expr) -> Result<Expr, JinjaError> {
        loop {
            match self.peek() {
                TokenKind::Pipe => {
                    self.pos += 1;
                    let (line, column) = self.position();
                    let name = self.expect_name()?;
                    if !SUPPORTED_FILTERS.contains(&name.as_str()) {
                        return Err(JinjaError::Syntax {
                            line,
                            column,
                            message: format!("unknown filter '{name}'"),
                        });
                    }
                    let (args, kwargs) = if self.peek() == &TokenKind::LParen {
                        self.parse_call_args()?
                    } else {
                        (Vec::new(), Vec::new())
                    };
                    node = Expr::Filter {
                        value: Box::new(node),
                        name,
                        args,
                        kwargs,
                    };
                }
                TokenKind::Name(n) if n == "is" => {
                    self.pos += 1;
                    let negated = self.eat_keyword("not");
                    let (line, column) = self.position();
                    let name = self.expect_name()?;
                    if !SUPPORTED_TESTS.contains(&name.as_str()) {
                        return Err(JinjaError::Syntax {
                            line,
                            column,
                            message: format!("unknown test '{name}'"),
                        });
                    }
                    let args = if self.peek() == &TokenKind::LParen {
                        let (args, kwargs) = self.parse_call_args()?;
                        if !kwargs.is_empty() {
                            return Err(self.error_here("tests do not accept keyword arguments"));
                        }
                        args
                    } else {
                        Vec::new()
                    };
                    node = Expr::Test {
                        value: Box::new(node),
                        name,
                        args,
                        negated,
                    };
                }
                TokenKind::LParen => {
                    let (args, kwargs) = self.parse_call_args()?;
                    node = Expr::Call {
                        callee: Box::new(node),
                        args,
                        kwargs,
                    };
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn parse_subscript(&mut self, obj: Expr) -> Result<Expr, JinjaError> {
        // `a[:2]`, `a[::-1]`, `a[1:2:3]`, `a[x]`
        let mut parts: Vec<Option<Expr>> = Vec::new();
        let mut is_slice = false;
        let mut current: Option<Expr> = None;
        loop {
            match self.peek() {
                TokenKind::RBracket => {
                    self.pos += 1;
                    parts.push(current.take());
                    if parts.len() > 3 {
                        return Err(self.error_here("too many ':' in a slice"));
                    }
                    break;
                }
                TokenKind::Colon => {
                    self.pos += 1;
                    is_slice = true;
                    parts.push(current.take());
                    if parts.len() > 3 {
                        return Err(self.error_here("too many ':' in a slice"));
                    }
                }
                TokenKind::Comma => {
                    return Err(self.error_here("tuple subscripts are not supported"))
                }
                TokenKind::Eof => return Err(self.error_here("unterminated subscript")),
                _ => {
                    if current.is_some() {
                        return Err(self.error_here("expected ':' or ']' in a subscript"));
                    }
                    current = Some(self.parse_expression()?);
                }
            }
        }
        if !is_slice {
            let index = parts
                .pop()
                .flatten()
                .ok_or_else(|| self.error_here("empty subscript"))?;
            return Ok(Expr::GetItem {
                obj: Box::new(obj),
                index: Box::new(index),
            });
        }
        let mut it = parts.into_iter();
        let start = it.next().flatten().map(Box::new);
        let stop = it.next().flatten().map(Box::new);
        let step = it.next().flatten().map(Box::new);
        Ok(Expr::Slice {
            obj: Box::new(obj),
            start,
            stop,
            step,
        })
    }

    #[allow(clippy::type_complexity)]
    fn parse_call_args(&mut self) -> Result<(Vec<Expr>, Vec<(String, Expr)>), JinjaError> {
        self.expect(&TokenKind::LParen)?;
        let mut args = Vec::new();
        let mut kwargs = Vec::new();
        while self.peek() != &TokenKind::RParen {
            if matches!(self.peek(), TokenKind::Star | TokenKind::Power) {
                return Err(self.error_here("argument unpacking is not supported"));
            }
            let is_kwarg =
                matches!(self.peek(), TokenKind::Name(_)) && self.peek_at(1) == &TokenKind::Assign;
            if is_kwarg {
                let name = self.expect_name()?;
                self.expect(&TokenKind::Assign)?;
                let value = self.parse_expression()?;
                kwargs.push((name, value));
            } else {
                if !kwargs.is_empty() {
                    return Err(self.error_here("positional argument after a keyword argument"));
                }
                args.push(self.parse_expression()?);
            }
            if self.peek() == &TokenKind::Comma {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok((args, kwargs))
    }

    fn parse_primary(&mut self) -> Result<Expr, JinjaError> {
        self.enter()?;
        let expr = match self.peek().clone() {
            TokenKind::Str(s) => {
                self.pos += 1;
                // Adjacent string literals concatenate, like Python.
                let mut buf = s;
                while let TokenKind::Str(next) = self.peek().clone() {
                    self.pos += 1;
                    buf.push_str(&next);
                }
                Expr::Str(Arc::from(buf.as_str()))
            }
            TokenKind::Int(i) => {
                self.pos += 1;
                Expr::Int(i)
            }
            TokenKind::Float(f) => {
                self.pos += 1;
                Expr::Float(f)
            }
            TokenKind::Name(name) => {
                self.pos += 1;
                match name.as_str() {
                    "true" | "True" => Expr::Bool(true),
                    "false" | "False" => Expr::Bool(false),
                    "none" | "None" => Expr::NoneLit,
                    _ => Expr::Name(name),
                }
            }
            TokenKind::LParen => {
                self.pos += 1;
                self.parse_paren_group()?
            }
            TokenKind::LBracket => {
                self.pos += 1;
                let mut items = Vec::new();
                while self.peek() != &TokenKind::RBracket {
                    items.push(self.parse_expression()?);
                    if self.peek() == &TokenKind::Comma {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                self.expect(&TokenKind::RBracket)?;
                Expr::List(items)
            }
            TokenKind::LBrace => {
                self.pos += 1;
                let mut items = Vec::new();
                while self.peek() != &TokenKind::RBrace {
                    let key = self.parse_expression()?;
                    self.expect(&TokenKind::Colon)?;
                    let value = self.parse_expression()?;
                    items.push((key, value));
                    if self.peek() == &TokenKind::Comma {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                self.expect(&TokenKind::RBrace)?;
                Expr::Dict(items)
            }
            other => {
                return Err(self.error_here(format!(
                    "expected an expression, found {}",
                    other.describe()
                )))
            }
        };
        self.leave();
        Ok(expr)
    }

    fn parse_paren_group(&mut self) -> Result<Expr, JinjaError> {
        if self.peek() == &TokenKind::RParen {
            self.pos += 1;
            return Ok(Expr::Tuple(Vec::new()));
        }
        let first = self.parse_expression()?;
        if self.peek() != &TokenKind::Comma {
            self.expect(&TokenKind::RParen)?;
            return Ok(first);
        }
        let mut items = vec![first];
        while self.peek() == &TokenKind::Comma {
            self.pos += 1;
            if self.peek() == &TokenKind::RParen {
                break;
            }
            items.push(self.parse_expression()?);
        }
        self.expect(&TokenKind::RParen)?;
        Ok(Expr::Tuple(items))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jinja::lexer::Lexer;

    fn parse(src: &str) -> Result<Vec<Node>, JinjaError> {
        let opts = JinjaOptions::default();
        let tokens = Lexer::new(src, &opts).tokenize()?;
        Parser::new(&tokens, &opts).parse_template()
    }

    fn parse_expr(src: &str) -> Result<Expr, JinjaError> {
        let nodes = parse(&format!("{{{{ {src} }}}}"))?;
        match nodes.into_iter().next() {
            Some(Node::Output(e)) => Ok(e),
            other => panic!("expected an output node, got {other:?}"),
        }
    }

    #[test]
    fn filter_binds_tighter_than_the_inline_conditional() {
        let expr = parse_expr("v | string if v is string else v | tojson | safe").expect("parse");
        match expr {
            Expr::Cond {
                cond,
                then,
                otherwise,
            } => {
                assert!(matches!(*cond, Expr::Test { ref name, .. } if name == "string"));
                assert!(matches!(*then, Expr::Filter { ref name, .. } if name == "string"));
                match otherwise {
                    Some(b) => {
                        assert!(matches!(*b, Expr::Filter { ref name, .. } if name == "safe"))
                    }
                    None => panic!("expected an else branch"),
                }
            }
            other => panic!("expected a conditional, got {other:?}"),
        }
    }

    #[test]
    fn concat_binds_tighter_than_addition() {
        let expr = parse_expr("'a' ~ 'b' + 'c' ~ 'd'").expect("parse");
        match expr {
            Expr::Binary { op, lhs, rhs } => {
                assert_eq!(op, BinOp::Add);
                assert!(matches!(
                    *lhs,
                    Expr::Binary {
                        op: BinOp::Concat,
                        ..
                    }
                ));
                assert!(matches!(
                    *rhs,
                    Expr::Binary {
                        op: BinOp::Concat,
                        ..
                    }
                ));
            }
            other => panic!("expected an addition, got {other:?}"),
        }
    }

    #[test]
    fn slices_parse_in_every_shape() {
        for src in ["xs[::-1]", "xs[1:]", "xs[:2]", "xs[1:2:3]", "xs[:]"] {
            assert!(
                matches!(parse_expr(src).expect("parse"), Expr::Slice { .. }),
                "{src} should parse as a slice"
            );
        }
        assert!(matches!(
            parse_expr("xs[0]").expect("parse"),
            Expr::GetItem { .. }
        ));
    }

    #[test]
    fn for_loop_filter_is_not_an_inline_conditional() {
        let nodes = parse("{% for x in xs if x %}{{ x }}{% endfor %}").expect("parse");
        match nodes.into_iter().next() {
            Some(Node::For(f)) => {
                assert!(f.cond.is_some());
                assert!(matches!(f.iter, Expr::Name(ref n) if n == "xs"));
            }
            other => panic!("expected a for loop, got {other:?}"),
        }
    }

    #[test]
    fn unknown_filters_and_tests_are_rejected_with_a_position() {
        for src in ["{{ a | nosuch }}", "{{ a is nosuch }}"] {
            match parse(src).expect_err("must not parse") {
                JinjaError::Syntax { line, column, .. } => {
                    assert!(line >= 1 && column >= 1);
                }
                other => panic!("expected a syntax error, got {other:?}"),
            }
        }
    }

    #[test]
    fn unsupported_statements_are_rejected() {
        for src in [
            "{% include 'x' %}",
            "{% extends 'x' %}",
            "{% raw %}x{% endraw %}",
            "{% block a %}{% endblock %}",
            "{% filter upper %}x{% endfilter %}",
            "{% with a = 1 %}{% endwith %}",
            "{% nosuchtag %}",
        ] {
            assert!(
                matches!(parse(src), Err(JinjaError::Syntax { .. })),
                "{src} must be rejected"
            );
        }
    }

    #[test]
    fn unterminated_blocks_are_rejected() {
        for src in [
            "{% if a %}x",
            "{% for x in y %}x",
            "{% macro m() %}x",
            "{% set x %}y",
            "{% if a %}x{% endfor %}",
            "{% endif %}",
        ] {
            assert!(
                matches!(parse(src), Err(JinjaError::Syntax { .. })),
                "{src} must be rejected"
            );
        }
    }

    #[test]
    fn loop_controls_require_a_loop() {
        assert!(parse("{% break %}").is_err());
        assert!(parse("{% continue %}").is_err());
        assert!(parse("{% for x in y %}{% break %}{% endfor %}").is_ok());
    }

    #[test]
    fn deeply_nested_expressions_are_bounded() {
        let src = format!("{{{{ {}1{} }}}}", "(".repeat(512), ")".repeat(512));
        match parse(&src) {
            Err(JinjaError::Syntax { message, .. }) => {
                assert!(message.contains("nested"), "unexpected message: {message}");
            }
            other => panic!("expected a depth error, got {other:?}"),
        }
    }

    #[test]
    fn deeply_nested_blocks_are_bounded() {
        let src = format!("{}x{}", "{% if a %}".repeat(512), "{% endif %}".repeat(512));
        assert!(matches!(parse(&src), Err(JinjaError::Syntax { .. })));
    }
}
