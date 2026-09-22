//! Abstract syntax tree for the supported Jinja subset.
//!
//! The tree is produced by [`crate::jinja::parser`] and consumed by
//! [`crate::jinja::eval`].  It is deliberately *closed*: there is no
//! "unknown node" escape hatch, so a construct outside the supported subset
//! cannot reach the evaluator — the parser rejects it with a
//! [`crate::jinja::JinjaError::Syntax`] carrying a line and column.

use std::sync::Arc;

/// Binary arithmetic / concatenation operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/` (Python true division — always yields a float)
    Div,
    /// `//` (floor division)
    FloorDiv,
    /// `%` (modulo)
    Mod,
    /// `**` (exponentiation)
    Pow,
    /// `~` (string concatenation, stringifying both sides)
    Concat,
}

impl BinOp {
    /// The operator's source spelling, used in error messages.
    pub fn spelling(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::FloorDiv => "//",
            BinOp::Mod => "%",
            BinOp::Pow => "**",
            BinOp::Concat => "~",
        }
    }
}

/// Comparison / membership operators (Python-style chaining is supported).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `in`
    In,
    /// `not in`
    NotIn,
}

/// Prefix operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `-`
    Neg,
    /// `+`
    Pos,
    /// `not`
    Not,
}

/// An expression node.
#[derive(Debug, Clone)]
pub enum Expr {
    /// String literal.
    Str(Arc<str>),
    /// Integer literal.
    Int(i64),
    /// Floating-point literal.
    Float(f64),
    /// `true` / `false`.
    Bool(bool),
    /// `none` / `null`.
    NoneLit,
    /// A variable reference.
    Name(String),
    /// `[a, b, c]`.
    List(Vec<Expr>),
    /// `(a, b, c)`.
    Tuple(Vec<Expr>),
    /// `{'k': v}`.
    Dict(Vec<(Expr, Expr)>),
    /// `-a`, `+a`, `not a`.
    Unary {
        /// Which prefix operator.
        op: UnaryOp,
        /// The operand.
        operand: Box<Expr>,
    },
    /// `a + b`, `a ~ b`, ...
    Binary {
        /// Which operator.
        op: BinOp,
        /// Left-hand side.
        lhs: Box<Expr>,
        /// Right-hand side.
        rhs: Box<Expr>,
    },
    /// `a < b <= c` — a chained comparison, evaluated left to right with
    /// short-circuiting exactly like Python.
    Compare {
        /// The left-most operand.
        lhs: Box<Expr>,
        /// `(operator, right-hand operand)` for each link of the chain.
        rest: Vec<(CmpOp, Expr)>,
    },
    /// `a and b` (short-circuiting).
    And(Box<Expr>, Box<Expr>),
    /// `a or b` (short-circuiting).
    Or(Box<Expr>, Box<Expr>),
    /// `then if cond else otherwise` — `otherwise` is `None` for the
    /// `x if c` form, which yields `Undefined` when the condition is false.
    Cond {
        /// The condition.
        cond: Box<Expr>,
        /// Value when the condition is truthy.
        then: Box<Expr>,
        /// Value when the condition is falsy (absent ⇒ `Undefined`).
        otherwise: Option<Box<Expr>>,
    },
    /// `obj.name`.
    GetAttr {
        /// The object.
        obj: Box<Expr>,
        /// The attribute name.
        name: String,
    },
    /// `obj[index]`.
    GetItem {
        /// The object.
        obj: Box<Expr>,
        /// The subscript.
        index: Box<Expr>,
    },
    /// `obj[start:stop:step]` with Python slice semantics.
    Slice {
        /// The object.
        obj: Box<Expr>,
        /// Optional start bound.
        start: Option<Box<Expr>>,
        /// Optional stop bound.
        stop: Option<Box<Expr>>,
        /// Optional step.
        step: Option<Box<Expr>>,
    },
    /// `callee(args, k=v)`.
    Call {
        /// The callable expression.
        callee: Box<Expr>,
        /// Positional arguments.
        args: Vec<Expr>,
        /// Keyword arguments, in source order.
        kwargs: Vec<(String, Expr)>,
    },
    /// `value | name(args)`.
    Filter {
        /// The filtered value.
        value: Box<Expr>,
        /// Filter name (validated against the supported set at parse time).
        name: String,
        /// Positional arguments.
        args: Vec<Expr>,
        /// Keyword arguments.
        kwargs: Vec<(String, Expr)>,
    },
    /// `value is [not] name(args)`.
    Test {
        /// The tested value.
        value: Box<Expr>,
        /// Test name (validated against the supported set at parse time).
        name: String,
        /// Positional arguments.
        args: Vec<Expr>,
        /// Whether the test was written as `is not`.
        negated: bool,
    },
}

/// The target of a `{% set %}` statement.
#[derive(Debug, Clone)]
pub enum SetTarget {
    /// `{% set x = ... %}`.
    Name(String),
    /// `{% set a, b = ... %}` — tuple unpacking.
    Names(Vec<String>),
    /// `{% set ns.field = ... %}` — mutation of a `namespace()` object.
    Attr {
        /// The namespace variable.
        base: String,
        /// The field being assigned.
        attr: String,
    },
}

/// One declared macro parameter.
#[derive(Debug, Clone)]
pub struct MacroParam {
    /// Parameter name.
    pub name: String,
    /// Default value expression, evaluated at call time when the argument is
    /// not supplied.
    pub default: Option<Expr>,
}

/// A `{% macro %}` definition.
#[derive(Debug, Clone)]
pub struct MacroDef {
    /// Macro name (`"caller"` for the implicit block of `{% call %}`).
    pub name: String,
    /// Declared parameters, in order.
    pub params: Vec<MacroParam>,
    /// Macro body.
    pub body: Vec<Node>,
}

/// A `{% call %}` block.
#[derive(Debug, Clone)]
pub struct CallBlock {
    /// The macro invocation the block decorates.
    pub callee: Expr,
    /// Parameters of the implicit `caller()` macro.
    pub params: Vec<MacroParam>,
    /// The block body, exposed to the macro as `caller()`.
    pub body: Vec<Node>,
}

/// A `{% for %}` loop.
#[derive(Debug, Clone)]
pub struct ForLoop {
    /// Loop target names (more than one ⇒ tuple unpacking).
    pub targets: Vec<String>,
    /// The iterable expression.
    pub iter: Expr,
    /// Optional `{% for x in xs if cond %}` filter, applied *before* the
    /// `loop` object is built (so `loop.index` counts filtered items only).
    pub cond: Option<Expr>,
    /// Loop body.
    pub body: Vec<Node>,
    /// `{% else %}` body, rendered when the loop ran zero times.
    pub otherwise: Vec<Node>,
}

/// A statement / template node.
#[derive(Debug, Clone)]
pub enum Node {
    /// Literal template text.
    Text(String),
    /// `{{ expr }}`.
    Output(Expr),
    /// `{% if %}` / `{% elif %}` / `{% else %}`.
    If {
        /// `(condition, body)` for the `if` and every `elif`.
        branches: Vec<(Expr, Vec<Node>)>,
        /// The `{% else %}` body (empty when absent).
        otherwise: Vec<Node>,
    },
    /// `{% for %}`.
    For(Box<ForLoop>),
    /// `{% set x = expr %}`.
    Set {
        /// Assignment target.
        target: SetTarget,
        /// Assigned expression.
        value: Expr,
    },
    /// `{% set x %}…{% endset %}`.
    SetBlock {
        /// Assignment target.
        target: SetTarget,
        /// The body whose rendering becomes the value.
        body: Vec<Node>,
    },
    /// `{% macro %}` definition.
    Macro(Arc<MacroDef>),
    /// `{% call %}` block.
    CallBlock(Box<CallBlock>),
    /// `{% generation %}…{% endgeneration %}` — the HuggingFace assistant-mask
    /// extension.  The body is rendered verbatim (pass-through); the engine
    /// does not emit any marker of its own.
    Generation(Vec<Node>),
    /// `{% break %}` (loop-controls extension).
    Break,
    /// `{% continue %}` (loop-controls extension).
    Continue,
}
