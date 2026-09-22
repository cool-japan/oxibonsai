//! Runtime values for the Jinja subset engine.
//!
//! The value model mirrors the Python objects a real Jinja environment would
//! see when rendering a chat template, because byte-identical rendering of
//! `tokenizer.chat_template` requires Python's *observable* semantics:
//!
//! * `{{ true }}` renders `True`, `{{ none }}` renders `None` (Python `str()`);
//! * `{{ [1, 'a'] }}` renders `[1, 'a']` and `{{ (1, 2) }}` renders `(1, 2)`
//!   (Python `repr()`), so lists and tuples are distinct types;
//! * an undefined value is falsy, renders as the empty string, has length 0
//!   and iterates empty, but **errors** on attribute access — which is exactly
//!   why `messages[0].role` on an empty `messages` list is a hard error;
//! * mappings keep **insertion order**, because `tojson` must reproduce
//!   Python's `json.dumps` defaults (which do *not* sort keys).

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;

use super::ast::MacroDef;
use super::JinjaError;

/// Maximum structural nesting inspected by the recursive value helpers
/// (`repr`, `tojson`, equality, ...).  Namespaces can form reference cycles,
/// so every recursive walk is bounded instead of trusting the data.
pub const MAX_VALUE_DEPTH: usize = 64;

/// An insertion-ordered string-keyed map.
///
/// `std::collections::HashMap` and `BTreeMap` both destroy the key order that
/// `tojson` has to reproduce, so mappings carry their own ordered storage.
#[derive(Debug, Clone, Default)]
pub struct ValueMap {
    entries: Vec<(String, Value)>,
    index: HashMap<String, usize>,
}

impl ValueMap {
    /// Create an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty map with room for `n` entries.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            entries: Vec::with_capacity(n),
            index: HashMap::with_capacity(n),
        }
    }

    /// Insert or overwrite `key`.  An overwrite keeps the original position,
    /// matching Python's `dict` semantics.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        let key = key.into();
        match self.index.get(&key) {
            Some(&i) => self.entries[i].1 = value,
            None => {
                self.index.insert(key.clone(), self.entries.len());
                self.entries.push((key, value));
            }
        }
    }

    /// Look a key up.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.index
            .get(key)
            .and_then(|&i| self.entries.get(i))
            .map(|e| &e.1)
    }

    /// Whether the key is present.
    pub fn contains_key(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate the entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    /// The keys, in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.iter().map(|(k, _)| k)
    }
}

impl FromIterator<(String, Value)> for ValueMap {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        let mut map = ValueMap::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

/// The built-in global functions a template may call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Builtin {
    /// `raise_exception(msg)` — surfaces as [`JinjaError::TemplateRaise`].
    RaiseException,
    /// `namespace(**kwargs)` — creates a mutable namespace object.
    Namespace,
    /// `range(stop)` / `range(start, stop[, step])`.
    Range,
    /// `dict(**kwargs)`.
    Dict,
}

impl Builtin {
    /// The name the function is bound to in the global scope.
    pub fn name(self) -> &'static str {
        match self {
            Builtin::RaiseException => "raise_exception",
            Builtin::Namespace => "namespace",
            Builtin::Range => "range",
            Builtin::Dict => "dict",
        }
    }
}

/// A callable macro, optionally carrying the frames captured at the point a
/// `{% call %}` block created it.
#[derive(Debug)]
pub struct MacroValue {
    /// The macro definition.
    pub def: Arc<MacroDef>,
    /// Frames captured for a `caller()` closure; `None` for a plain macro,
    /// which resolves free names against the template's root frame.
    pub closure: Option<Rc<Vec<HashMap<String, Value>>>>,
}

/// A runtime value.
#[derive(Debug, Clone)]
pub enum Value {
    /// A name that was never defined, or a missing mapping key / list index.
    Undefined,
    /// Python `None`.
    None,
    /// Python `bool`.
    Bool(bool),
    /// Python `int`.
    Int(i64),
    /// Python `float`.
    Float(f64),
    /// Python `str`.
    ///
    /// `Arc` rather than `Rc` so that evaluating a string literal can share
    /// the compiled template's storage — the AST is `Send + Sync`.
    Str(Arc<str>),
    /// Python `list`.
    List(Rc<Vec<Value>>),
    /// Python `tuple` — distinct from a list only in `repr()`.
    Tuple(Rc<Vec<Value>>),
    /// Python `dict`, insertion-ordered.
    Map(Rc<ValueMap>),
    /// A Jinja `namespace()` object: a mapping with *mutable* fields shared by
    /// every reference to it, which is how templates carry state out of a loop.
    Namespace(Rc<RefCell<ValueMap>>),
    /// A user-defined `{% macro %}` (or the implicit `caller()`).
    Macro(Rc<MacroValue>),
    /// A built-in global function.
    Builtin(Builtin),
}

impl Value {
    /// Build a string value.
    pub fn str(s: impl AsRef<str>) -> Value {
        Value::Str(Arc::from(s.as_ref()))
    }

    /// Build a list value.
    pub fn list(items: Vec<Value>) -> Value {
        Value::List(Rc::new(items))
    }

    /// Build a map value.
    pub fn map(map: ValueMap) -> Value {
        Value::Map(Rc::new(map))
    }

    /// The Python type name, used verbatim in error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Undefined => "undefined",
            Value::None => "none",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Tuple(_) => "tuple",
            Value::Map(_) => "dict",
            Value::Namespace(_) => "namespace",
            Value::Macro(_) => "macro",
            Value::Builtin(_) => "function",
        }
    }

    /// Python truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Undefined | Value::None => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(l) | Value::Tuple(l) => !l.is_empty(),
            Value::Map(m) => !m.is_empty(),
            Value::Namespace(_) | Value::Macro(_) | Value::Builtin(_) => true,
        }
    }

    /// Borrow the string payload of a `Str`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Borrow the items of a list *or* a tuple.
    pub fn as_seq(&self) -> Option<&[Value]> {
        match self {
            Value::List(l) | Value::Tuple(l) => Some(l),
            _ => None,
        }
    }

    /// Numeric view used by the arithmetic and comparison helpers.
    pub fn as_number(&self) -> Option<Num> {
        match self {
            Value::Bool(b) => Some(Num::Int(i64::from(*b))),
            Value::Int(i) => Some(Num::Int(*i)),
            Value::Float(f) => Some(Num::Float(*f)),
            _ => None,
        }
    }

    /// Python `str()` — what `{{ … }}` writes into the output.
    pub fn to_display_string(&self) -> Result<String, JinjaError> {
        let mut out = String::new();
        self.write_display(&mut out, 0)?;
        Ok(out)
    }

    /// Python `repr()` — how a value is rendered *inside* a container.
    pub fn to_repr_string(&self) -> Result<String, JinjaError> {
        let mut out = String::new();
        self.write_repr(&mut out, 0)?;
        Ok(out)
    }

    fn write_display(&self, out: &mut String, depth: usize) -> Result<(), JinjaError> {
        check_depth(depth)?;
        match self {
            Value::Undefined => Ok(()),
            Value::None => {
                out.push_str("None");
                Ok(())
            }
            Value::Bool(b) => {
                out.push_str(if *b { "True" } else { "False" });
                Ok(())
            }
            Value::Int(i) => {
                out.push_str(&i.to_string());
                Ok(())
            }
            Value::Float(f) => {
                out.push_str(&format_float(*f));
                Ok(())
            }
            Value::Str(s) => {
                out.push_str(s);
                Ok(())
            }
            _ => self.write_repr(out, depth),
        }
    }

    fn write_repr(&self, out: &mut String, depth: usize) -> Result<(), JinjaError> {
        check_depth(depth)?;
        match self {
            Value::Undefined => {
                out.push_str("Undefined");
                Ok(())
            }
            Value::Str(s) => {
                out.push_str(&py_repr_str(s));
                Ok(())
            }
            Value::List(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_repr(out, depth + 1)?;
                }
                out.push(']');
                Ok(())
            }
            Value::Tuple(items) => {
                out.push('(');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_repr(out, depth + 1)?;
                }
                if items.len() == 1 {
                    out.push(',');
                }
                out.push(')');
                Ok(())
            }
            Value::Map(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&py_repr_str(k));
                    out.push_str(": ");
                    v.write_repr(out, depth + 1)?;
                }
                out.push('}');
                Ok(())
            }
            Value::Namespace(ns) => {
                let borrowed = ns.try_borrow().map_err(|_| {
                    JinjaError::runtime("namespace is already borrowed (reference cycle)")
                })?;
                out.push_str("<Namespace ");
                out.push('{');
                for (i, (k, v)) in borrowed.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&py_repr_str(k));
                    out.push_str(": ");
                    v.write_repr(out, depth + 1)?;
                }
                out.push('}');
                out.push('>');
                Ok(())
            }
            Value::Macro(m) => {
                out.push_str(&format!("<macro {}>", m.def.name));
                Ok(())
            }
            Value::Builtin(b) => {
                out.push_str(&format!("<function {}>", b.name()));
                Ok(())
            }
            _ => self.write_display(out, depth),
        }
    }

    /// Serialize as JSON using Python's `json.dumps` **defaults**: separators
    /// `", "` / `": "`, `ensure_ascii=True`, and **no** key sorting.
    pub fn to_json_string(&self) -> Result<String, JinjaError> {
        let mut out = String::new();
        self.write_json(&mut out, 0)?;
        Ok(out)
    }

    fn write_json(&self, out: &mut String, depth: usize) -> Result<(), JinjaError> {
        check_depth(depth)?;
        match self {
            Value::None => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Int(i) => out.push_str(&i.to_string()),
            Value::Float(f) => out.push_str(&format_json_float(*f)),
            Value::Str(s) => json_escape_into(s, out),
            Value::List(items) | Value::Tuple(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_json(out, depth + 1)?;
                }
                out.push(']');
            }
            Value::Map(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    json_escape_into(k, out);
                    out.push_str(": ");
                    v.write_json(out, depth + 1)?;
                }
                out.push('}');
            }
            Value::Undefined | Value::Namespace(_) | Value::Macro(_) | Value::Builtin(_) => {
                return Err(JinjaError::runtime(format!(
                    "object of type '{}' is not JSON serializable",
                    self.type_name()
                )));
            }
        }
        Ok(())
    }

    /// Python `==` semantics, bounded against reference cycles.
    pub fn equals(&self, other: &Value) -> bool {
        self.equals_depth(other, 0)
    }

    fn equals_depth(&self, other: &Value, depth: usize) -> bool {
        if depth > MAX_VALUE_DEPTH {
            return false;
        }
        match (self, other) {
            (Value::Undefined, Value::Undefined) => true,
            (Value::Undefined, _) | (_, Value::Undefined) => false,
            (Value::None, Value::None) => true,
            (Value::None, _) | (_, Value::None) => false,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Str(_), _) | (_, Value::Str(_)) => false,
            (Value::List(a), Value::List(b)) | (Value::Tuple(a), Value::Tuple(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x.equals_depth(y, depth + 1))
            }
            (Value::Map(a), Value::Map(b)) => {
                a.len() == b.len()
                    && a.iter().all(|(k, v)| {
                        b.get(k)
                            .map(|other| v.equals_depth(other, depth + 1))
                            .unwrap_or(false)
                    })
            }
            (Value::Namespace(a), Value::Namespace(b)) => Rc::ptr_eq(a, b),
            (Value::Macro(a), Value::Macro(b)) => Rc::ptr_eq(a, b),
            (Value::Builtin(a), Value::Builtin(b)) => a == b,
            _ => match (self.as_number(), other.as_number()) {
                (Some(a), Some(b)) => a.eq_num(b),
                _ => false,
            },
        }
    }

    /// Python `<`, `<=`, `>`, `>=` — a type mismatch is an error, never a
    /// silent `false`.
    pub fn compare(&self, other: &Value) -> Result<Ordering, JinjaError> {
        if let (Some(a), Some(b)) = (self.as_number(), other.as_number()) {
            return a
                .cmp_num(b)
                .ok_or_else(|| JinjaError::runtime("cannot order NaN".to_string()));
        }
        match (self, other) {
            (Value::Str(a), Value::Str(b)) => Ok(a.as_ref().cmp(b.as_ref())),
            _ => Err(JinjaError::runtime(format!(
                "'<' not supported between instances of '{}' and '{}'",
                self.type_name(),
                other.type_name()
            ))),
        }
    }

    /// Python `in`.
    ///
    /// Membership in an undefined value is `false` rather than an error:
    /// Jinja's `Undefined` has no `__contains__`, so Python falls back to its
    /// (empty) `__iter__`.
    pub fn contains(&self, needle: &Value) -> Result<bool, JinjaError> {
        match self {
            Value::Undefined => Ok(false),
            Value::Str(hay) => match needle.as_str() {
                Some(n) => Ok(hay.contains(n)),
                None => Err(JinjaError::runtime(format!(
                    "'in <string>' requires string as left operand, not '{}'",
                    needle.type_name()
                ))),
            },
            Value::List(items) | Value::Tuple(items) => Ok(items.iter().any(|v| v.equals(needle))),
            Value::Map(m) => Ok(needle.as_str().map(|k| m.contains_key(k)).unwrap_or(false)),
            Value::Namespace(ns) => {
                let borrowed = ns
                    .try_borrow()
                    .map_err(|_| JinjaError::runtime("namespace is already borrowed"))?;
                Ok(needle
                    .as_str()
                    .map(|k| borrowed.contains_key(k))
                    .unwrap_or(false))
            }
            _ => Err(JinjaError::runtime(format!(
                "argument of type '{}' is not a container or iterable",
                self.type_name()
            ))),
        }
    }

    /// Python `len()`.  An undefined value has length 0, exactly like Jinja's
    /// `Undefined.__len__`.
    pub fn length(&self) -> Result<usize, JinjaError> {
        match self {
            Value::Undefined => Ok(0),
            Value::Str(s) => Ok(s.chars().count()),
            Value::List(l) | Value::Tuple(l) => Ok(l.len()),
            Value::Map(m) => Ok(m.len()),
            Value::Namespace(ns) => ns
                .try_borrow()
                .map(|b| b.len())
                .map_err(|_| JinjaError::runtime("namespace is already borrowed")),
            _ => Err(JinjaError::runtime(format!(
                "object of type '{}' has no len()",
                self.type_name()
            ))),
        }
    }

    /// Materialise the value as a sequence, Python-style: strings iterate by
    /// character, mappings by key, and an undefined value iterates empty
    /// (Jinja's `Undefined.__iter__`).
    pub fn iterate(&self) -> Result<Vec<Value>, JinjaError> {
        match self {
            Value::Undefined => Ok(Vec::new()),
            Value::List(l) | Value::Tuple(l) => Ok(l.as_ref().clone()),
            Value::Str(s) => Ok(s.chars().map(|c| Value::str(c.to_string())).collect()),
            Value::Map(m) => Ok(m.keys().map(Value::str).collect()),
            Value::Namespace(ns) => {
                let borrowed = ns
                    .try_borrow()
                    .map_err(|_| JinjaError::runtime("namespace is already borrowed"))?;
                Ok(borrowed.keys().map(Value::str).collect())
            }
            _ => Err(JinjaError::runtime(format!(
                "'{}' object is not iterable",
                self.type_name()
            ))),
        }
    }

    /// Attribute access (`obj.name`).
    ///
    /// A missing attribute yields [`Value::Undefined`]; accessing an attribute
    /// **of** an undefined value is a hard error, which is what makes
    /// `messages[0].role` fail on an empty message list instead of silently
    /// rendering nothing.
    pub fn get_attr(&self, name: &str) -> Result<Value, JinjaError> {
        match self {
            Value::Undefined => Err(JinjaError::runtime(format!(
                "cannot read attribute '{name}' of an undefined value"
            ))),
            Value::Map(m) => Ok(m.get(name).cloned().unwrap_or(Value::Undefined)),
            Value::Namespace(ns) => {
                let borrowed = ns
                    .try_borrow()
                    .map_err(|_| JinjaError::runtime("namespace is already borrowed"))?;
                Ok(borrowed.get(name).cloned().unwrap_or(Value::Undefined))
            }
            _ => Ok(Value::Undefined),
        }
    }

    /// Subscript access (`obj[index]`).
    pub fn get_item(&self, index: &Value) -> Result<Value, JinjaError> {
        match self {
            Value::Undefined => Err(JinjaError::runtime(
                "cannot subscript an undefined value".to_string(),
            )),
            Value::Map(_) | Value::Namespace(_) => match index.as_str() {
                Some(k) => self.get_attr(k),
                None => Ok(Value::Undefined),
            },
            Value::List(items) | Value::Tuple(items) => match index.as_number() {
                Some(Num::Int(i)) => Ok(resolve_index(i, items.len())
                    .and_then(|i| items.get(i).cloned())
                    .unwrap_or(Value::Undefined)),
                _ => Ok(Value::Undefined),
            },
            Value::Str(s) => match index.as_number() {
                Some(Num::Int(i)) => {
                    let chars: Vec<char> = s.chars().collect();
                    Ok(resolve_index(i, chars.len())
                        .and_then(|i| chars.get(i))
                        .map(|c| Value::str(c.to_string()))
                        .unwrap_or(Value::Undefined))
                }
                _ => Ok(Value::Undefined),
            },
            _ => Ok(Value::Undefined),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_display_string() {
            Ok(s) => f.write_str(&s),
            Err(_) => f.write_str("<unrenderable>"),
        }
    }
}

/// A numeric view of a value, preserving the int/float distinction that
/// Python's `repr` (and therefore `tojson`) depends on.
#[derive(Debug, Clone, Copy)]
pub enum Num {
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
}

impl Num {
    /// Python `==` between two numbers.
    pub fn eq_num(self, other: Num) -> bool {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) => a == b,
            _ => self.as_f64() == other.as_f64(),
        }
    }

    /// Python ordering between two numbers (`None` when either is NaN).
    pub fn cmp_num(self, other: Num) -> Option<Ordering> {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) => Some(a.cmp(&b)),
            _ => self.as_f64().partial_cmp(&other.as_f64()),
        }
    }

    /// Widen to `f64`.
    pub fn as_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
}

fn check_depth(depth: usize) -> Result<(), JinjaError> {
    if depth > MAX_VALUE_DEPTH {
        Err(JinjaError::limit(format!(
            "value nesting deeper than {MAX_VALUE_DEPTH}"
        )))
    } else {
        Ok(())
    }
}

/// Normalise a possibly negative Python index against a length.
pub fn resolve_index(i: i64, len: usize) -> Option<usize> {
    let len_i = i64::try_from(len).ok()?;
    let idx = if i < 0 { i.checked_add(len_i)? } else { i };
    if idx < 0 || idx >= len_i {
        None
    } else {
        usize::try_from(idx).ok()
    }
}

/// Python `repr()` of a string: single quotes unless the string contains a
/// single quote but no double quote.
pub fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `repr()` of a float: always distinguishable from an int (`1.0`, not
/// `1`), switching to exponential notation on the same thresholds as CPython.
pub fn format_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    let exp_form = format!("{x:e}");
    let exponent = exp_form
        .rsplit_once('e')
        .and_then(|(_, e)| e.parse::<i32>().ok())
        .unwrap_or(0);
    if !(-4..16).contains(&exponent) {
        let (mantissa, _) = exp_form
            .rsplit_once('e')
            .unwrap_or((exp_form.as_str(), "0"));
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{}e{}{:02}", mantissa, sign, exponent.abs());
    }
    let plain = format!("{x}");
    if plain.contains(['.', 'e', 'E', 'n', 'i']) {
        plain
    } else {
        format!("{plain}.0")
    }
}

/// `json.dumps` rendering of a float — same as [`format_float`] except that
/// the non-finite spellings follow Python's JSON extension.
pub fn format_json_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    format_float(x)
}

/// Escape a string exactly like `json.dumps(..., ensure_ascii=True)`: every
/// code point outside printable ASCII becomes a `\uXXXX` escape (astral planes
/// become a surrogate pair), and `<`, `>`, `&`, `'` are **not** escaped.
pub fn json_escape_into(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (' '..='~').contains(&c) => out.push(c),
            c => {
                let cp = c as u32;
                if cp <= 0xFFFF {
                    out.push_str(&format!("\\u{cp:04x}"));
                } else {
                    let v = cp - 0x1_0000;
                    let high = 0xD800 + (v >> 10);
                    let low = 0xDC00 + (v & 0x3FF);
                    out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                }
            }
        }
    }
    out.push('"');
}

// ── JSON interop ─────────────────────────────────────────────────────────────

impl Value {
    /// Parse a JSON document into a value, **preserving object key order**.
    ///
    /// This is the constructor template contexts should be built with: going
    /// through [`serde_json::Value`] would re-sort object keys (its `Map` is a
    /// `BTreeMap` unless the `preserve_order` feature is on) and `tojson`
    /// would then emit a different byte string than the reference runtime.
    pub fn from_json_str(json: &str) -> Result<Value, JinjaError> {
        serde_json::from_str::<Value>(json)
            .map_err(|e| JinjaError::runtime(format!("invalid JSON context: {e}")))
    }
}

impl From<&serde_json::Value> for Value {
    /// Convert from `serde_json`.
    ///
    /// **Key order caveat.**  Object key order follows `serde_json::Map`'s
    /// iteration order, which is *sorted* unless that crate's
    /// `preserve_order` feature is enabled (it is **not** enabled in this
    /// workspace).  A chat template that pipes a tools object through
    /// `tojson` would then emit `{"description": …, "name": …}` where the
    /// reference runtime emits `{"name": …, "description": …}`, changing the
    /// prompt byte for byte.  The order is lost when `serde_json` parses, so
    /// it cannot be recovered here: build the context with
    /// [`Value::from_json_str`] from the original JSON *text* whenever the
    /// rendering has to match a reference renderer.
    fn from(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::Null => Value::None,
            serde_json::Value::Bool(b) => Value::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(u) = n.as_u64() {
                    i64::try_from(u)
                        .map(Value::Int)
                        .unwrap_or(Value::Float(u as f64))
                } else {
                    Value::Float(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            serde_json::Value::String(s) => Value::str(s),
            serde_json::Value::Array(a) => Value::list(a.iter().map(Value::from).collect()),
            serde_json::Value::Object(o) => {
                let mut map = ValueMap::with_capacity(o.len());
                for (k, v) in o {
                    map.insert(k.clone(), Value::from(v));
                }
                Value::map(map)
            }
        }
    }
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Int(v))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Value, E> {
        Ok(i64::try_from(v)
            .map(Value::Int)
            .unwrap_or(Value::Float(v as f64)))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        Ok(Value::Float(v))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Value, E> {
        Ok(Value::str(v))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::None)
    }

    fn visit_none<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::None)
    }

    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        d.deserialize_any(ValueVisitor)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(v) = seq.next_element::<Value>()? {
            items.push(v);
        }
        Ok(Value::list(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut out = ValueMap::new();
        while let Some((k, v)) = map.next_entry::<String, Value>()? {
            out.insert(k, v);
        }
        Ok(Value::map(out))
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(ValueVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_map_preserves_insertion_order_on_overwrite() {
        let mut m = ValueMap::new();
        m.insert("z", Value::Int(1));
        m.insert("a", Value::Int(2));
        m.insert("z", Value::Int(3));
        let keys: Vec<&str> = m.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["z", "a"]);
        assert_eq!(m.get("z"), Some(&Value::Int(3)));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn json_round_trip_keeps_key_order() {
        let v = Value::from_json_str(r#"{"z":1,"m":2,"a":3}"#).expect("parse");
        assert_eq!(
            v.to_json_string().expect("json"),
            r#"{"z": 1, "m": 2, "a": 3}"#
        );
    }

    #[test]
    fn json_escapes_match_python_defaults() {
        let v = Value::str("< > & ' \" \\ \n \t \u{7f} \u{e9} \u{1f600}");
        assert_eq!(
            v.to_json_string().expect("json"),
            "\"< > & ' \\\" \\\\ \\n \\t \\u007f \\u00e9 \\ud83d\\ude00\""
        );
    }

    #[test]
    fn float_formatting_matches_python_repr() {
        assert_eq!(format_float(1.0), "1.0");
        assert_eq!(format_float(2.5), "2.5");
        assert_eq!(format_float(-0.0), "-0.0");
        assert_eq!(format_float(0.1), "0.1");
        assert_eq!(format_float(1e16), "1e+16");
        assert_eq!(format_float(1e-5), "1e-05");
        assert_eq!(format_float(1e15), "1000000000000000.0");
        assert_eq!(format_json_float(f64::INFINITY), "Infinity");
        assert_eq!(format_json_float(f64::NAN), "NaN");
    }

    #[test]
    fn undefined_semantics() {
        let u = Value::Undefined;
        assert!(!u.truthy());
        assert_eq!(u.to_display_string().expect("display"), "");
        assert_eq!(u.length().expect("len"), 0);
        assert!(u.iterate().expect("iter").is_empty());
        assert!(u.get_attr("x").is_err());
        assert!(u.to_json_string().is_err());
        assert!(u.equals(&Value::Undefined));
        assert!(!u.equals(&Value::None));
    }

    #[test]
    fn repr_matches_python() {
        let list = Value::list(vec![
            Value::Int(1),
            Value::str("a"),
            Value::Bool(true),
            Value::None,
        ]);
        assert_eq!(list.to_repr_string().expect("repr"), "[1, 'a', True, None]");
        let tup = Value::Tuple(Rc::new(vec![Value::Int(1)]));
        assert_eq!(tup.to_repr_string().expect("repr"), "(1,)");
        assert_eq!(py_repr_str("it's"), "\"it's\"");
    }

    #[test]
    fn negative_index_resolution() {
        assert_eq!(resolve_index(-1, 3), Some(2));
        assert_eq!(resolve_index(0, 3), Some(0));
        assert_eq!(resolve_index(-4, 3), None);
        assert_eq!(resolve_index(3, 3), None);
    }

    #[test]
    fn numeric_equality_follows_python() {
        assert!(Value::Bool(true).equals(&Value::Int(1)));
        assert!(Value::Int(1).equals(&Value::Float(1.0)));
        assert!(!Value::Int(1).equals(&Value::str("1")));
    }
}
