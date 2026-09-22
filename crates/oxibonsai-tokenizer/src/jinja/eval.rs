//! Interpreter for the Jinja subset: statements, expressions, filters, tests
//! and the handful of globals a real chat template needs.
//!
//! Semantics follow CPython/Jinja closely enough to render
//! `tokenizer.chat_template` byte-identically:
//!
//! * `{% if %}` does **not** open a scope (so `{% set %}` inside it is visible
//!   afterwards) while `{% for %}`, `{% macro %}` and `{% call %}` do — which
//!   is exactly why templates carry loop state in a `namespace()`;
//! * a macro resolves free names against the template's root frame, so a macro
//!   defined before a `{% set %}` still sees the value at call time;
//! * arithmetic is checked: an overflow, a division by zero or a type mismatch
//!   is a [`JinjaError::Runtime`], never a panic and never a wrong number.
//!
//! Every loop, every recursion and the output itself are bounded by
//! [`JinjaOptions`], because this engine renders attacker-influenced data on
//! the serve path.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use super::ast::{BinOp, CallBlock, CmpOp, Expr, ForLoop, MacroDef, Node, SetTarget, UnaryOp};
use super::value::{Builtin, MacroValue, Num, Value, ValueMap};
use super::{JinjaError, JinjaOptions};

/// How a block finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Normal,
    Break,
    Continue,
}

/// Render `nodes` against `context`.
pub fn render(nodes: &[Node], context: &Value, opts: &JinjaOptions) -> Result<String, JinjaError> {
    let mut root: HashMap<String, Value> = HashMap::new();
    match context {
        Value::Map(m) => {
            for (k, v) in m.iter() {
                root.insert(k.clone(), v.clone());
            }
        }
        Value::Namespace(ns) => {
            let borrowed = ns
                .try_borrow()
                .map_err(|_| JinjaError::runtime("context namespace is already borrowed"))?;
            for (k, v) in borrowed.iter() {
                root.insert(k.clone(), v.clone());
            }
        }
        Value::Undefined | Value::None => {}
        other => {
            return Err(JinjaError::runtime(format!(
                "template context must be a mapping, got '{}'",
                other.type_name()
            )))
        }
    }
    for builtin in [
        Builtin::RaiseException,
        Builtin::Namespace,
        Builtin::Range,
        Builtin::Dict,
    ] {
        root.entry(builtin.name().to_string())
            .or_insert(Value::Builtin(builtin));
    }

    let mut interp = Interpreter {
        opts,
        frames: vec![root],
        barrier: Vec::new(),
        out: String::new(),
        emitted: 0,
        iterations: 0,
        depth: 0,
    };
    interp.exec_nodes(nodes)?;
    Ok(interp.out)
}

/// The interpreter state.
struct Interpreter<'a> {
    opts: &'a JinjaOptions,
    frames: Vec<HashMap<String, Value>>,
    /// Frame indices below which name resolution jumps straight to the root
    /// frame — pushed by every macro invocation.
    barrier: Vec<usize>,
    out: String,
    emitted: usize,
    iterations: usize,
    depth: usize,
}

impl Interpreter<'_> {
    // ── scope handling ───────────────────────────────────────────────────

    fn lookup(&self, name: &str) -> Value {
        let floor = self.barrier.last().copied().unwrap_or(0);
        for frame in self.frames[floor..].iter().rev() {
            if let Some(v) = frame.get(name) {
                return v.clone();
            }
        }
        if floor > 0 {
            if let Some(v) = self.frames[0].get(name) {
                return v.clone();
            }
        }
        Value::Undefined
    }

    fn assign(&mut self, name: &str, value: Value) {
        if let Some(frame) = self.frames.last_mut() {
            frame.insert(name.to_string(), value);
        }
    }

    fn emit(&mut self, text: &str) -> Result<(), JinjaError> {
        self.emitted = self.emitted.saturating_add(text.len());
        if self.emitted > self.opts.max_output_bytes {
            return Err(JinjaError::limit(format!(
                "rendered output exceeded {} bytes",
                self.opts.max_output_bytes
            )));
        }
        self.out.push_str(text);
        Ok(())
    }

    fn enter(&mut self) -> Result<(), JinjaError> {
        self.depth += 1;
        if self.depth > self.opts.max_render_depth {
            return Err(JinjaError::limit(format!(
                "render recursion deeper than {}",
                self.opts.max_render_depth
            )));
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    fn count_iteration(&mut self, n: usize) -> Result<(), JinjaError> {
        self.iterations = self.iterations.saturating_add(n);
        if self.iterations > self.opts.max_loop_iterations {
            return Err(JinjaError::limit(format!(
                "loop iterations exceeded {}",
                self.opts.max_loop_iterations
            )));
        }
        Ok(())
    }

    // ── statements ───────────────────────────────────────────────────────

    fn exec_nodes(&mut self, nodes: &[Node]) -> Result<Flow, JinjaError> {
        self.enter()?;
        for node in nodes {
            match self.exec_node(node)? {
                Flow::Normal => {}
                other => {
                    self.leave();
                    return Ok(other);
                }
            }
        }
        self.leave();
        Ok(Flow::Normal)
    }

    fn exec_node(&mut self, node: &Node) -> Result<Flow, JinjaError> {
        match node {
            Node::Text(text) => {
                self.emit(text)?;
                Ok(Flow::Normal)
            }
            Node::Output(expr) => {
                let value = self.eval(expr)?;
                let text = value.to_display_string()?;
                self.emit(&text)?;
                Ok(Flow::Normal)
            }
            Node::If {
                branches,
                otherwise,
            } => {
                for (cond, body) in branches {
                    if self.eval(cond)?.truthy() {
                        return self.exec_nodes(body);
                    }
                }
                self.exec_nodes(otherwise)
            }
            Node::For(loop_node) => self.exec_for(loop_node),
            Node::Set { target, value } => {
                let value = self.eval(value)?;
                self.assign_target(target, value)?;
                Ok(Flow::Normal)
            }
            Node::SetBlock { target, body } => {
                let rendered = self.render_to_string(body)?;
                self.assign_target(target, Value::str(rendered))?;
                Ok(Flow::Normal)
            }
            Node::Macro(def) => {
                let value = Value::Macro(Rc::new(MacroValue {
                    def: Arc::clone(def),
                    closure: None,
                }));
                self.assign(&def.name, value);
                Ok(Flow::Normal)
            }
            Node::CallBlock(block) => {
                self.exec_call_block(block)?;
                Ok(Flow::Normal)
            }
            Node::Generation(body) => self.exec_nodes(body),
            Node::Break => Ok(Flow::Break),
            Node::Continue => Ok(Flow::Continue),
        }
    }

    fn render_to_string(&mut self, nodes: &[Node]) -> Result<String, JinjaError> {
        let saved = std::mem::take(&mut self.out);
        let result = self.exec_nodes(nodes);
        let rendered = std::mem::replace(&mut self.out, saved);
        result?;
        Ok(rendered)
    }

    fn exec_for(&mut self, loop_node: &ForLoop) -> Result<Flow, JinjaError> {
        let iterable = self.eval(&loop_node.iter)?;
        let mut items = iterable.iterate()?;
        self.count_iteration(items.len())?;

        if let Some(cond) = &loop_node.cond {
            let mut kept = Vec::with_capacity(items.len());
            self.frames.push(HashMap::new());
            let filtered = (|| -> Result<Vec<Value>, JinjaError> {
                for item in items.drain(..) {
                    self.bind_targets(&loop_node.targets, &item)?;
                    if self.eval(cond)?.truthy() {
                        kept.push(item);
                    }
                }
                Ok(kept)
            })();
            self.frames.pop();
            items = filtered?;
        }

        if items.is_empty() {
            return self.exec_nodes(&loop_node.otherwise);
        }

        let length = items.len();
        let mut flow = Flow::Normal;
        for (index, item) in items.iter().enumerate() {
            self.count_iteration(1)?;
            self.frames.push(HashMap::new());
            let step = (|| -> Result<Flow, JinjaError> {
                self.bind_targets(&loop_node.targets, item)?;
                let loop_object = build_loop_object(&items, index, length);
                self.assign("loop", loop_object);
                self.exec_nodes(&loop_node.body)
            })();
            self.frames.pop();
            match step? {
                Flow::Break => {
                    flow = Flow::Normal;
                    break;
                }
                _ => flow = Flow::Normal,
            }
        }
        Ok(flow)
    }

    fn bind_targets(&mut self, targets: &[String], item: &Value) -> Result<(), JinjaError> {
        if targets.len() == 1 {
            self.assign(&targets[0], item.clone());
            return Ok(());
        }
        let parts = item.iterate()?;
        if parts.len() != targets.len() {
            return Err(JinjaError::runtime(format!(
                "cannot unpack {} value(s) into {} target(s)",
                parts.len(),
                targets.len()
            )));
        }
        for (name, value) in targets.iter().zip(parts) {
            self.assign(name, value);
        }
        Ok(())
    }

    fn assign_target(&mut self, target: &SetTarget, value: Value) -> Result<(), JinjaError> {
        match target {
            SetTarget::Name(name) => {
                self.assign(name, value);
                Ok(())
            }
            SetTarget::Names(names) => {
                let parts = value.iterate()?;
                if parts.len() != names.len() {
                    return Err(JinjaError::runtime(format!(
                        "cannot unpack {} value(s) into {} target(s)",
                        parts.len(),
                        names.len()
                    )));
                }
                for (name, part) in names.iter().zip(parts) {
                    self.assign(name, part);
                }
                Ok(())
            }
            SetTarget::Attr { base, attr } => match self.lookup(base) {
                Value::Namespace(ns) => {
                    let mut borrowed = ns.try_borrow_mut().map_err(|_| {
                        JinjaError::runtime("namespace is already borrowed".to_string())
                    })?;
                    borrowed.insert(attr.clone(), value);
                    Ok(())
                }
                other => Err(JinjaError::runtime(format!(
                    "cannot assign to '{base}.{attr}': '{}' is not a namespace",
                    other.type_name()
                ))),
            },
        }
    }

    fn exec_call_block(&mut self, block: &CallBlock) -> Result<(), JinjaError> {
        let Expr::Call {
            callee,
            args,
            kwargs,
        } = &block.callee
        else {
            return Err(JinjaError::runtime(
                "'{% call %}' requires a macro invocation".to_string(),
            ));
        };
        let caller = Value::Macro(Rc::new(MacroValue {
            def: Arc::new(MacroDef {
                name: "caller".to_string(),
                params: block.params.clone(),
                body: block.body.clone(),
            }),
            closure: Some(Rc::new(self.frames.clone())),
        }));
        let rendered = self.eval_call(callee, args, kwargs, Some(caller))?;
        let text = rendered.to_display_string()?;
        self.emit(&text)
    }

    // ── expressions ──────────────────────────────────────────────────────

    fn eval(&mut self, expr: &Expr) -> Result<Value, JinjaError> {
        self.enter()?;
        let value = self.eval_inner(expr);
        self.leave();
        value
    }

    fn eval_inner(&mut self, expr: &Expr) -> Result<Value, JinjaError> {
        match expr {
            Expr::Str(s) => Ok(Value::Str(Arc::clone(s))),
            Expr::Int(i) => Ok(Value::Int(*i)),
            Expr::Float(f) => Ok(Value::Float(*f)),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::NoneLit => Ok(Value::None),
            Expr::Name(name) => Ok(self.lookup(name)),
            Expr::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.eval(item)?);
                }
                Ok(Value::list(out))
            }
            Expr::Tuple(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.eval(item)?);
                }
                Ok(Value::Tuple(Rc::new(out)))
            }
            Expr::Dict(pairs) => {
                let mut map = ValueMap::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let key = self.eval(k)?.to_display_string()?;
                    let value = self.eval(v)?;
                    map.insert(key, value);
                }
                Ok(Value::map(map))
            }
            Expr::Unary { op, operand } => {
                let value = self.eval(operand)?;
                unary_op(*op, &value)
            }
            Expr::Binary { op, lhs, rhs } => {
                let l = self.eval(lhs)?;
                let r = self.eval(rhs)?;
                binary_op(*op, &l, &r)
            }
            Expr::Compare { lhs, rest } => {
                let mut left = self.eval(lhs)?;
                for (op, rhs) in rest {
                    let right = self.eval(rhs)?;
                    if !compare_op(*op, &left, &right)? {
                        return Ok(Value::Bool(false));
                    }
                    left = right;
                }
                Ok(Value::Bool(true))
            }
            Expr::And(lhs, rhs) => {
                let left = self.eval(lhs)?;
                if left.truthy() {
                    self.eval(rhs)
                } else {
                    Ok(left)
                }
            }
            Expr::Or(lhs, rhs) => {
                let left = self.eval(lhs)?;
                if left.truthy() {
                    Ok(left)
                } else {
                    self.eval(rhs)
                }
            }
            Expr::Cond {
                cond,
                then,
                otherwise,
            } => {
                if self.eval(cond)?.truthy() {
                    self.eval(then)
                } else {
                    match otherwise {
                        Some(e) => self.eval(e),
                        None => Ok(Value::Undefined),
                    }
                }
            }
            Expr::GetAttr { obj, name } => {
                let target = self.eval(obj)?;
                target.get_attr(name)
            }
            Expr::GetItem { obj, index } => {
                let target = self.eval(obj)?;
                let key = self.eval(index)?;
                target.get_item(&key)
            }
            Expr::Slice {
                obj,
                start,
                stop,
                step,
            } => {
                let target = self.eval(obj)?;
                let start = self.eval_opt_index(start)?;
                let stop = self.eval_opt_index(stop)?;
                let step = self.eval_opt_index(step)?;
                slice_value(&target, start, stop, step)
            }
            Expr::Call {
                callee,
                args,
                kwargs,
            } => self.eval_call(callee, args, kwargs, None),
            Expr::Filter {
                value,
                name,
                args,
                kwargs,
            } => {
                let target = self.eval(value)?;
                let mut positional = Vec::with_capacity(args.len());
                for arg in args {
                    positional.push(self.eval(arg)?);
                }
                let mut named = Vec::with_capacity(kwargs.len());
                for (k, v) in kwargs {
                    named.push((k.clone(), self.eval(v)?));
                }
                self.apply_filter(name, target, &positional, &named)
            }
            Expr::Test {
                value,
                name,
                args,
                negated,
            } => {
                let target = self.eval(value)?;
                let mut positional = Vec::with_capacity(args.len());
                for arg in args {
                    positional.push(self.eval(arg)?);
                }
                let result = apply_test(name, &target, &positional)?;
                Ok(Value::Bool(result != *negated))
            }
        }
    }

    fn eval_opt_index(&mut self, expr: &Option<Box<Expr>>) -> Result<Option<i64>, JinjaError> {
        match expr {
            None => Ok(None),
            Some(e) => {
                let value = self.eval(e)?;
                match value.as_number() {
                    Some(Num::Int(i)) => Ok(Some(i)),
                    Some(Num::Float(f)) => Ok(Some(f as i64)),
                    None => Err(JinjaError::runtime(format!(
                        "slice indices must be integers, got '{}'",
                        value.type_name()
                    ))),
                }
            }
        }
    }

    fn eval_call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        kwargs: &[(String, Expr)],
        caller: Option<Value>,
    ) -> Result<Value, JinjaError> {
        let mut positional = Vec::with_capacity(args.len());
        for arg in args {
            positional.push(self.eval(arg)?);
        }
        let mut named = Vec::with_capacity(kwargs.len());
        for (k, v) in kwargs {
            named.push((k.clone(), self.eval(v)?));
        }

        if let Expr::GetAttr { obj, name } = callee {
            let target = self.eval(obj)?;
            if let Some(result) = call_method(&target, name, &positional, &named)? {
                return Ok(result);
            }
            let attribute = target.get_attr(name)?;
            if matches!(attribute, Value::Undefined) {
                return Err(JinjaError::runtime(format!(
                    "'{}' object has no attribute '{}'",
                    target.type_name(),
                    name
                )));
            }
            return self.call_value(attribute, positional, named, caller);
        }

        let function = self.eval(callee)?;
        self.call_value(function, positional, named, caller)
    }

    fn call_value(
        &mut self,
        function: Value,
        args: Vec<Value>,
        kwargs: Vec<(String, Value)>,
        caller: Option<Value>,
    ) -> Result<Value, JinjaError> {
        match function {
            Value::Macro(m) => self.call_macro(&m, args, kwargs, caller),
            Value::Builtin(Builtin::RaiseException) => {
                let message = match args.first() {
                    Some(v) => v.to_display_string()?,
                    None => String::new(),
                };
                Err(JinjaError::TemplateRaise(message))
            }
            Value::Builtin(Builtin::Namespace) => {
                let mut map = ValueMap::with_capacity(kwargs.len());
                for arg in &args {
                    match arg {
                        Value::Map(m) => {
                            for (k, v) in m.iter() {
                                map.insert(k.clone(), v.clone());
                            }
                        }
                        other => {
                            return Err(JinjaError::runtime(format!(
                                "namespace() takes mappings or keyword arguments, got '{}'",
                                other.type_name()
                            )))
                        }
                    }
                }
                for (k, v) in kwargs {
                    map.insert(k, v);
                }
                Ok(Value::Namespace(Rc::new(RefCell::new(map))))
            }
            Value::Builtin(Builtin::Dict) => {
                let mut map = ValueMap::with_capacity(kwargs.len());
                for (k, v) in kwargs {
                    map.insert(k, v);
                }
                Ok(Value::map(map))
            }
            Value::Builtin(Builtin::Range) => self.builtin_range(&args),
            other => Err(JinjaError::runtime(format!(
                "'{}' object is not callable",
                other.type_name()
            ))),
        }
    }

    fn builtin_range(&mut self, args: &[Value]) -> Result<Value, JinjaError> {
        let ints: Result<Vec<i64>, JinjaError> = args
            .iter()
            .map(|v| match v.as_number() {
                Some(Num::Int(i)) => Ok(i),
                _ => Err(JinjaError::runtime(format!(
                    "range() expects integers, got '{}'",
                    v.type_name()
                ))),
            })
            .collect();
        let ints = ints?;
        let (start, stop, step) = match ints.len() {
            1 => (0, ints[0], 1),
            2 => (ints[0], ints[1], 1),
            3 => (ints[0], ints[1], ints[2]),
            n => {
                return Err(JinjaError::runtime(format!(
                    "range() takes 1 to 3 arguments, got {n}"
                )))
            }
        };
        if step == 0 {
            return Err(JinjaError::runtime("range() step must not be zero"));
        }
        let mut out = Vec::new();
        let mut current = start;
        while (step > 0 && current < stop) || (step < 0 && current > stop) {
            out.push(Value::Int(current));
            self.count_iteration(1)?;
            current = current
                .checked_add(step)
                .ok_or_else(|| JinjaError::runtime("range() overflowed"))?;
        }
        Ok(Value::list(out))
    }

    fn call_macro(
        &mut self,
        macro_value: &Rc<MacroValue>,
        args: Vec<Value>,
        kwargs: Vec<(String, Value)>,
        caller: Option<Value>,
    ) -> Result<Value, JinjaError> {
        let def = Arc::clone(&macro_value.def);
        if args.len() > def.params.len() {
            return Err(JinjaError::runtime(format!(
                "macro '{}' takes not more than {} argument(s), got {}",
                def.name,
                def.params.len(),
                args.len()
            )));
        }

        let mut frame: HashMap<String, Value> = HashMap::with_capacity(def.params.len() + 1);
        for (param, value) in def.params.iter().zip(args.iter()) {
            frame.insert(param.name.clone(), value.clone());
        }
        for (name, value) in kwargs {
            if !def.params.iter().any(|p| p.name == name) {
                return Err(JinjaError::runtime(format!(
                    "macro '{}' takes no keyword argument '{}'",
                    def.name, name
                )));
            }
            if frame.contains_key(&name) {
                return Err(JinjaError::runtime(format!(
                    "macro '{}' got multiple values for argument '{}'",
                    def.name, name
                )));
            }
            frame.insert(name, value);
        }
        for param in &def.params {
            if frame.contains_key(&param.name) {
                continue;
            }
            let value = match &param.default {
                Some(expr) => self.eval(expr)?,
                None => Value::Undefined,
            };
            frame.insert(param.name.clone(), value);
        }
        if let Some(caller) = caller {
            frame.insert("caller".to_string(), caller);
        }

        self.enter()?;
        let base = self.frames.len();
        if let Some(closure) = &macro_value.closure {
            for captured in closure.iter() {
                self.frames.push(captured.clone());
            }
        }
        self.frames.push(frame);
        self.barrier.push(base);
        let saved = std::mem::take(&mut self.out);
        let result = self.exec_nodes(&def.body);
        let rendered = std::mem::replace(&mut self.out, saved);
        self.barrier.pop();
        self.frames.truncate(base);
        self.leave();
        result?;
        Ok(Value::str(rendered))
    }

    // ── filters ──────────────────────────────────────────────────────────

    fn apply_filter(
        &mut self,
        name: &str,
        value: Value,
        args: &[Value],
        kwargs: &[(String, Value)],
    ) -> Result<Value, JinjaError> {
        match name {
            "safe" => {
                no_args(name, args, kwargs)?;
                Ok(value)
            }
            "tojson" => {
                no_args(name, args, kwargs)?;
                Ok(Value::str(value.to_json_string()?))
            }
            "string" => {
                no_args(name, args, kwargs)?;
                Ok(Value::str(value.to_display_string()?))
            }
            "length" => {
                no_args(name, args, kwargs)?;
                Ok(Value::Int(
                    i64::try_from(value.length()?).unwrap_or(i64::MAX),
                ))
            }
            "trim" => {
                let text = value.to_display_string()?;
                match args.first() {
                    None => Ok(Value::str(text.trim())),
                    Some(chars) => {
                        let set: Vec<char> = chars
                            .as_str()
                            .ok_or_else(|| {
                                JinjaError::runtime("trim() expects a string of characters")
                            })?
                            .chars()
                            .collect();
                        Ok(Value::str(text.trim_matches(|c| set.contains(&c))))
                    }
                }
            }
            "lower" => {
                no_args(name, args, kwargs)?;
                Ok(Value::str(value.to_display_string()?.to_lowercase()))
            }
            "upper" => {
                no_args(name, args, kwargs)?;
                Ok(Value::str(value.to_display_string()?.to_uppercase()))
            }
            "int" => {
                // Jinja's `int` filter raises on an undefined value (its
                // `__int__` fails) instead of falling back to the default.
                if matches!(value, Value::Undefined) {
                    return Err(JinjaError::runtime(
                        "cannot convert an undefined value to int",
                    ));
                }
                let default = args.first().cloned().unwrap_or(Value::Int(0));
                Ok(to_int(&value).unwrap_or(default))
            }
            "list" => {
                no_args(name, args, kwargs)?;
                Ok(Value::list(value.iterate()?))
            }
            "first" => {
                no_args(name, args, kwargs)?;
                Ok(value
                    .iterate()?
                    .first()
                    .cloned()
                    .unwrap_or(Value::Undefined))
            }
            "last" => {
                no_args(name, args, kwargs)?;
                Ok(value.iterate()?.last().cloned().unwrap_or(Value::Undefined))
            }
            "items" => {
                no_args(name, args, kwargs)?;
                filter_items(&value)
            }
            "default" => {
                let fallback = args.first().cloned().unwrap_or_else(|| Value::str(""));
                let boolean = args
                    .get(1)
                    .cloned()
                    .or_else(|| {
                        kwargs
                            .iter()
                            .find(|(k, _)| k == "boolean")
                            .map(|(_, v)| v.clone())
                    })
                    .map(|v| v.truthy())
                    .unwrap_or(false);
                let replace = if boolean {
                    !value.truthy()
                } else {
                    matches!(value, Value::Undefined)
                };
                Ok(if replace { fallback } else { value })
            }
            "replace" => {
                let text = value.to_display_string()?;
                let old = args
                    .first()
                    .ok_or_else(|| JinjaError::runtime("replace() needs an old value"))?
                    .to_display_string()?;
                let new = args
                    .get(1)
                    .ok_or_else(|| JinjaError::runtime("replace() needs a new value"))?
                    .to_display_string()?;
                match args.get(2) {
                    None => Ok(Value::str(text.replace(&old, &new))),
                    Some(count) => {
                        let n = match to_int(count) {
                            Some(Value::Int(n)) if n >= 0 => usize::try_from(n).unwrap_or(0),
                            _ => {
                                return Err(JinjaError::runtime(
                                    "replace() count must be a non-negative integer",
                                ))
                            }
                        };
                        Ok(Value::str(text.replacen(&old, &new, n)))
                    }
                }
            }
            "join" => {
                let separator = match args.first() {
                    Some(v) => v.to_display_string()?,
                    None => String::new(),
                };
                let attribute = kwargs
                    .iter()
                    .find(|(k, _)| k == "attribute")
                    .map(|(_, v)| v.clone());
                let mut parts = Vec::new();
                for item in value.iterate()? {
                    let item = match &attribute {
                        Some(attr) => {
                            let key = attr.to_display_string()?;
                            item.get_attr(&key)?
                        }
                        None => item,
                    };
                    parts.push(item.to_display_string()?);
                }
                Ok(Value::str(parts.join(&separator)))
            }
            "selectattr" => {
                let attribute = args
                    .first()
                    .ok_or_else(|| JinjaError::runtime("selectattr() needs an attribute name"))?
                    .to_display_string()?;
                let test_name = match args.get(1) {
                    Some(v) => v.to_display_string()?,
                    None => String::new(),
                };
                let test_args: Vec<Value> = args.iter().skip(2).cloned().collect();
                let mut kept = Vec::new();
                for item in value.iterate()? {
                    let attr = item.get_attr(&attribute)?;
                    let keep = if test_name.is_empty() {
                        attr.truthy()
                    } else {
                        apply_test(&test_name, &attr, &test_args)?
                    };
                    if keep {
                        kept.push(item);
                    }
                }
                Ok(Value::list(kept))
            }
            "map" => {
                let attribute = kwargs
                    .iter()
                    .find(|(k, _)| k == "attribute")
                    .map(|(_, v)| v.clone());
                let mut mapped = Vec::new();
                match attribute {
                    Some(attr) => {
                        let key = attr.to_display_string()?;
                        for item in value.iterate()? {
                            mapped.push(item.get_attr(&key)?);
                        }
                    }
                    None => {
                        let filter_name = args
                            .first()
                            .ok_or_else(|| {
                                JinjaError::runtime("map() needs a filter name or attribute=")
                            })?
                            .to_display_string()?;
                        let rest: Vec<Value> = args.iter().skip(1).cloned().collect();
                        for item in value.iterate()? {
                            mapped.push(self.apply_filter(&filter_name, item, &rest, &[])?);
                        }
                    }
                }
                Ok(Value::list(mapped))
            }
            other => Err(JinjaError::runtime(format!("unknown filter '{other}'"))),
        }
    }
}

// ── free helpers ─────────────────────────────────────────────────────────────

fn no_args(name: &str, args: &[Value], kwargs: &[(String, Value)]) -> Result<(), JinjaError> {
    if args.is_empty() && kwargs.is_empty() {
        Ok(())
    } else {
        Err(JinjaError::runtime(format!(
            "filter '{name}' takes no arguments"
        )))
    }
}

fn filter_items(value: &Value) -> Result<Value, JinjaError> {
    let pairs: Vec<Value> = match value {
        Value::Map(m) => m
            .iter()
            .map(|(k, v)| Value::Tuple(Rc::new(vec![Value::str(k), v.clone()])))
            .collect(),
        Value::Namespace(ns) => {
            let borrowed = ns
                .try_borrow()
                .map_err(|_| JinjaError::runtime("namespace is already borrowed"))?;
            borrowed
                .iter()
                .map(|(k, v)| Value::Tuple(Rc::new(vec![Value::str(k), v.clone()])))
                .collect()
        }
        Value::Undefined => Vec::new(),
        other => {
            return Err(JinjaError::runtime(format!(
                "'{}' object has no items()",
                other.type_name()
            )))
        }
    };
    Ok(Value::list(pairs))
}

fn to_int(value: &Value) -> Option<Value> {
    match value {
        Value::Int(i) => Some(Value::Int(*i)),
        Value::Bool(b) => Some(Value::Int(i64::from(*b))),
        Value::Float(f) => {
            if f.is_finite() {
                Some(Value::Int(*f as i64))
            } else {
                None
            }
        }
        Value::Str(s) => {
            let text = s.trim();
            if let Ok(i) = text.parse::<i64>() {
                return Some(Value::Int(i));
            }
            text.parse::<f64>()
                .ok()
                .filter(|f| f.is_finite())
                .map(|f| Value::Int(f as i64))
        }
        _ => None,
    }
}

/// Build the `loop` object for iteration `index` of a loop of `length` items.
fn build_loop_object(items: &[Value], index: usize, length: usize) -> Value {
    let mut map = ValueMap::with_capacity(9);
    let as_i64 = |n: usize| i64::try_from(n).unwrap_or(i64::MAX);
    map.insert("index0", Value::Int(as_i64(index)));
    map.insert("index", Value::Int(as_i64(index + 1)));
    map.insert("revindex", Value::Int(as_i64(length - index)));
    map.insert("revindex0", Value::Int(as_i64(length - index - 1)));
    map.insert("first", Value::Bool(index == 0));
    map.insert("last", Value::Bool(index + 1 == length));
    map.insert("length", Value::Int(as_i64(length)));
    map.insert(
        "previtem",
        index
            .checked_sub(1)
            .and_then(|i| items.get(i))
            .cloned()
            .unwrap_or(Value::Undefined),
    );
    map.insert(
        "nextitem",
        items.get(index + 1).cloned().unwrap_or(Value::Undefined),
    );
    Value::map(map)
}

/// Dispatch a method call such as `s.startswith('x')` or `d.get('k')`.
///
/// Returns `Ok(None)` when the receiver has no such built-in method, letting
/// the caller fall back to an attribute that holds a callable.
fn call_method(
    target: &Value,
    name: &str,
    args: &[Value],
    kwargs: &[(String, Value)],
) -> Result<Option<Value>, JinjaError> {
    if !kwargs.is_empty() {
        return Ok(None);
    }
    match target {
        Value::Str(text) => {
            let result = match name {
                "startswith" | "endswith" => {
                    let needles = match args.first() {
                        Some(Value::Tuple(items)) | Some(Value::List(items)) => items
                            .iter()
                            .map(|v| v.to_display_string())
                            .collect::<Result<Vec<_>, _>>()?,
                        Some(other) => vec![other.to_display_string()?],
                        None => {
                            return Err(JinjaError::runtime(format!("{name}() needs an argument")))
                        }
                    };
                    let hit = needles.iter().any(|n| {
                        if name == "startswith" {
                            text.starts_with(n.as_str())
                        } else {
                            text.ends_with(n.as_str())
                        }
                    });
                    Value::Bool(hit)
                }
                "strip" => Value::str(text.trim()),
                "lstrip" => Value::str(text.trim_start()),
                "rstrip" => Value::str(text.trim_end()),
                "upper" => Value::str(text.to_uppercase()),
                "lower" => Value::str(text.to_lowercase()),
                "replace" => {
                    let old = args
                        .first()
                        .ok_or_else(|| JinjaError::runtime("replace() needs an old value"))?
                        .to_display_string()?;
                    let new = args
                        .get(1)
                        .ok_or_else(|| JinjaError::runtime("replace() needs a new value"))?
                        .to_display_string()?;
                    Value::str(text.replace(&old, &new))
                }
                "split" => {
                    let parts: Vec<Value> = match args.first() {
                        None => text.split_whitespace().map(Value::str).collect(),
                        Some(sep) => {
                            let sep = sep.to_display_string()?;
                            if sep.is_empty() {
                                return Err(JinjaError::runtime("empty separator in split()"));
                            }
                            text.split(sep.as_str()).map(Value::str).collect()
                        }
                    };
                    Value::list(parts)
                }
                "join" => {
                    let mut parts = Vec::new();
                    for item in args
                        .first()
                        .ok_or_else(|| JinjaError::runtime("join() needs an iterable"))?
                        .iterate()?
                    {
                        parts.push(item.to_display_string()?);
                    }
                    Value::str(parts.join(text.as_ref()))
                }
                _ => return Ok(None),
            };
            Ok(Some(result))
        }
        Value::Map(_) | Value::Namespace(_) => {
            let result = match name {
                "get" => {
                    let key = args
                        .first()
                        .ok_or_else(|| JinjaError::runtime("get() needs a key"))?
                        .to_display_string()?;
                    let fallback = args.get(1).cloned().unwrap_or(Value::None);
                    match target.get_attr(&key)? {
                        Value::Undefined => fallback,
                        found => found,
                    }
                }
                "keys" => Value::list(target.iterate()?),
                "values" => {
                    let items = filter_items(target)?;
                    let mut values = Vec::new();
                    for pair in items.iterate()? {
                        values.push(pair.get_item(&Value::Int(1))?);
                    }
                    Value::list(values)
                }
                "items" => filter_items(target)?,
                _ => return Ok(None),
            };
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}

/// Evaluate `value is name(args)`.
pub fn apply_test(name: &str, value: &Value, args: &[Value]) -> Result<bool, JinjaError> {
    let result = match name {
        "defined" => !matches!(value, Value::Undefined),
        "undefined" => matches!(value, Value::Undefined),
        "none" => matches!(value, Value::None),
        "string" => matches!(value, Value::Str(_)),
        "mapping" => matches!(value, Value::Map(_) | Value::Namespace(_)),
        "iterable" => matches!(
            value,
            Value::Str(_) | Value::List(_) | Value::Tuple(_) | Value::Map(_)
        ),
        "sequence" => matches!(
            value,
            Value::Str(_) | Value::List(_) | Value::Tuple(_) | Value::Map(_)
        ),
        "number" => matches!(value, Value::Int(_) | Value::Float(_) | Value::Bool(_)),
        "integer" => matches!(value, Value::Int(_) | Value::Bool(_)),
        "boolean" => matches!(value, Value::Bool(_)),
        "true" => matches!(value, Value::Bool(true)),
        "false" => matches!(value, Value::Bool(false)),
        "equalto" | "eq" | "==" => {
            let other = args
                .first()
                .ok_or_else(|| JinjaError::runtime("test 'equalto' needs an argument"))?;
            value.equals(other)
        }
        "ne" | "!=" => {
            let other = args
                .first()
                .ok_or_else(|| JinjaError::runtime("test 'ne' needs an argument"))?;
            !value.equals(other)
        }
        "in" => {
            let container = args
                .first()
                .ok_or_else(|| JinjaError::runtime("test 'in' needs an argument"))?;
            container.contains(value)?
        }
        other => return Err(JinjaError::runtime(format!("unknown test '{other}'"))),
    };
    Ok(result)
}

fn unary_op(op: UnaryOp, value: &Value) -> Result<Value, JinjaError> {
    match op {
        UnaryOp::Not => Ok(Value::Bool(!value.truthy())),
        UnaryOp::Pos | UnaryOp::Neg => match value.as_number() {
            Some(Num::Int(i)) => {
                if op == UnaryOp::Pos {
                    Ok(Value::Int(i))
                } else {
                    i.checked_neg()
                        .map(Value::Int)
                        .ok_or_else(|| JinjaError::runtime("integer overflow in unary '-'"))
                }
            }
            Some(Num::Float(f)) => Ok(Value::Float(if op == UnaryOp::Pos { f } else { -f })),
            None => Err(JinjaError::runtime(format!(
                "bad operand type for unary '{}': '{}'",
                if op == UnaryOp::Pos { "+" } else { "-" },
                value.type_name()
            ))),
        },
    }
}

fn binary_op(op: BinOp, lhs: &Value, rhs: &Value) -> Result<Value, JinjaError> {
    if op == BinOp::Concat {
        let mut text = lhs.to_display_string()?;
        text.push_str(&rhs.to_display_string()?);
        return Ok(Value::str(text));
    }
    if op == BinOp::Add {
        if let (Value::Str(a), Value::Str(b)) = (lhs, rhs) {
            let mut text = String::with_capacity(a.len() + b.len());
            text.push_str(a);
            text.push_str(b);
            return Ok(Value::str(text));
        }
        if let (Some(a), Some(b)) = (lhs.as_seq(), rhs.as_seq()) {
            let mut items = a.to_vec();
            items.extend_from_slice(b);
            return Ok(match (lhs, rhs) {
                (Value::Tuple(_), Value::Tuple(_)) => Value::Tuple(Rc::new(items)),
                _ => Value::list(items),
            });
        }
    }
    if op == BinOp::Mul {
        if let (Value::Str(a), Some(Num::Int(n))) = (lhs, rhs.as_number()) {
            return repeat_string(a, n);
        }
        if let (Some(Num::Int(n)), Value::Str(b)) = (lhs.as_number(), rhs) {
            return repeat_string(b, n);
        }
        if let (Some(items), Some(Num::Int(n))) = (lhs.as_seq(), rhs.as_number()) {
            return repeat_seq(items, n);
        }
    }

    let (a, b) = match (lhs.as_number(), rhs.as_number()) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            return Err(JinjaError::runtime(format!(
                "unsupported operand type(s) for {}: '{}' and '{}'",
                op.spelling(),
                lhs.type_name(),
                rhs.type_name()
            )))
        }
    };

    match (a, b) {
        (Num::Int(x), Num::Int(y)) => int_arithmetic(op, x, y),
        _ => float_arithmetic(op, a.as_f64(), b.as_f64()),
    }
}

fn repeat_string(text: &Arc<str>, times: i64) -> Result<Value, JinjaError> {
    if times <= 0 {
        return Ok(Value::str(""));
    }
    let n = usize::try_from(times).unwrap_or(usize::MAX);
    let total = text
        .len()
        .checked_mul(n)
        .ok_or_else(|| JinjaError::runtime("string repetition overflowed"))?;
    if total > MAX_REPEAT_BYTES {
        return Err(JinjaError::limit(format!(
            "string repetition longer than {MAX_REPEAT_BYTES} bytes"
        )));
    }
    Ok(Value::str(text.repeat(n)))
}

fn repeat_seq(items: &[Value], times: i64) -> Result<Value, JinjaError> {
    if times <= 0 {
        return Ok(Value::list(Vec::new()));
    }
    let n = usize::try_from(times).unwrap_or(usize::MAX);
    let total = items
        .len()
        .checked_mul(n)
        .ok_or_else(|| JinjaError::runtime("sequence repetition overflowed"))?;
    if total > MAX_REPEAT_ITEMS {
        return Err(JinjaError::limit(format!(
            "sequence repetition longer than {MAX_REPEAT_ITEMS} items"
        )));
    }
    let mut out = Vec::with_capacity(total);
    for _ in 0..n {
        out.extend_from_slice(items);
    }
    Ok(Value::list(out))
}

/// Upper bound on a single `'x' * n` expansion.
const MAX_REPEAT_BYTES: usize = 8 * 1024 * 1024;
/// Upper bound on a single `[x] * n` expansion.
const MAX_REPEAT_ITEMS: usize = 1_000_000;

fn int_arithmetic(op: BinOp, x: i64, y: i64) -> Result<Value, JinjaError> {
    let overflow = || JinjaError::runtime(format!("integer overflow in '{}'", op.spelling()));
    match op {
        BinOp::Add => x.checked_add(y).map(Value::Int).ok_or_else(overflow),
        BinOp::Sub => x.checked_sub(y).map(Value::Int).ok_or_else(overflow),
        BinOp::Mul => x.checked_mul(y).map(Value::Int).ok_or_else(overflow),
        BinOp::Div => {
            if y == 0 {
                Err(JinjaError::runtime("division by zero"))
            } else {
                Ok(Value::Float(x as f64 / y as f64))
            }
        }
        BinOp::FloorDiv => {
            if y == 0 {
                return Err(JinjaError::runtime("integer division or modulo by zero"));
            }
            let quotient = x.checked_div(y).ok_or_else(overflow)?;
            let remainder = x.checked_rem(y).ok_or_else(overflow)?;
            if remainder != 0 && ((remainder < 0) != (y < 0)) {
                quotient.checked_sub(1).map(Value::Int).ok_or_else(overflow)
            } else {
                Ok(Value::Int(quotient))
            }
        }
        BinOp::Mod => {
            if y == 0 {
                return Err(JinjaError::runtime("integer division or modulo by zero"));
            }
            let remainder = x.checked_rem(y).ok_or_else(overflow)?;
            if remainder != 0 && ((remainder < 0) != (y < 0)) {
                remainder
                    .checked_add(y)
                    .map(Value::Int)
                    .ok_or_else(overflow)
            } else {
                Ok(Value::Int(remainder))
            }
        }
        BinOp::Pow => {
            if y < 0 {
                return Ok(Value::Float((x as f64).powf(y as f64)));
            }
            let exponent = u32::try_from(y).map_err(|_| overflow())?;
            x.checked_pow(exponent).map(Value::Int).ok_or_else(overflow)
        }
        // `~` is handled before the numeric dispatch; this arm returns an
        // error rather than panicking should that invariant ever be lost.
        BinOp::Concat => Err(JinjaError::runtime(
            "internal error: '~' reached the numeric dispatch",
        )),
    }
}

fn float_arithmetic(op: BinOp, x: f64, y: f64) -> Result<Value, JinjaError> {
    let value = match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        BinOp::Div => {
            if y == 0.0 {
                return Err(JinjaError::runtime("float division by zero"));
            }
            x / y
        }
        BinOp::FloorDiv => {
            if y == 0.0 {
                return Err(JinjaError::runtime("float floor division by zero"));
            }
            (x / y).floor()
        }
        BinOp::Mod => {
            if y == 0.0 {
                return Err(JinjaError::runtime("float modulo by zero"));
            }
            x - y * (x / y).floor()
        }
        BinOp::Pow => x.powf(y),
        BinOp::Concat => {
            return Err(JinjaError::runtime(
                "internal error: '~' reached the numeric dispatch",
            ))
        }
    };
    Ok(Value::Float(value))
}

fn compare_op(op: CmpOp, lhs: &Value, rhs: &Value) -> Result<bool, JinjaError> {
    match op {
        CmpOp::Eq => Ok(lhs.equals(rhs)),
        CmpOp::Ne => Ok(!lhs.equals(rhs)),
        CmpOp::In => rhs.contains(lhs),
        CmpOp::NotIn => Ok(!rhs.contains(lhs)?),
        CmpOp::Lt => Ok(lhs.compare(rhs)?.is_lt()),
        CmpOp::Le => Ok(lhs.compare(rhs)?.is_le()),
        CmpOp::Gt => Ok(lhs.compare(rhs)?.is_gt()),
        CmpOp::Ge => Ok(lhs.compare(rhs)?.is_ge()),
    }
}

/// Python slice semantics (`a[start:stop:step]`), including negative bounds
/// and a negative step.
fn slice_value(
    target: &Value,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Result<Value, JinjaError> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(JinjaError::runtime("slice step cannot be zero"));
    }

    let chars: Vec<Value>;
    let items: &[Value] = match target {
        Value::List(l) | Value::Tuple(l) => l,
        Value::Str(s) => {
            chars = s.chars().map(|c| Value::str(c.to_string())).collect();
            &chars
        }
        Value::Undefined => return Ok(Value::Undefined),
        other => {
            return Err(JinjaError::runtime(format!(
                "'{}' object is not subscriptable",
                other.type_name()
            )))
        }
    };

    let len = i64::try_from(items.len()).unwrap_or(i64::MAX);
    let clamp_forward = |v: i64| {
        let v = if v < 0 { v.saturating_add(len) } else { v };
        v.clamp(0, len)
    };
    let clamp_backward = |v: i64| {
        let v = if v < 0 { v.saturating_add(len) } else { v };
        v.clamp(-1, (len - 1).max(-1))
    };
    let (mut cursor, end) = if step > 0 {
        (
            start.map(clamp_forward).unwrap_or(0),
            stop.map(clamp_forward).unwrap_or(len),
        )
    } else {
        (
            start.map(clamp_backward).unwrap_or(len - 1),
            stop.map(clamp_backward).unwrap_or(-1),
        )
    };

    let mut selected = Vec::new();
    while (step > 0 && cursor < end) || (step < 0 && cursor > end) {
        match usize::try_from(cursor).ok().and_then(|i| items.get(i)) {
            Some(v) => selected.push(v.clone()),
            None => break,
        }
        cursor = match cursor.checked_add(step) {
            Some(next) => next,
            None => break,
        };
    }

    Ok(match target {
        Value::Str(_) => {
            let mut text = String::new();
            for part in &selected {
                text.push_str(part.as_str().unwrap_or(""));
            }
            Value::str(text)
        }
        Value::Tuple(_) => Value::Tuple(Rc::new(selected)),
        _ => Value::list(selected),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jinja::JinjaTemplate;

    fn render_str(src: &str, ctx_json: &str) -> Result<String, JinjaError> {
        let template = JinjaTemplate::compile(src)?;
        let ctx = Value::from_json_str(ctx_json)?;
        template.render(&ctx)
    }

    #[test]
    fn arithmetic_is_checked_not_panicking() {
        assert!(render_str("{{ 1 / 0 }}", "{}").is_err());
        assert!(render_str("{{ 1 % 0 }}", "{}").is_err());
        assert!(render_str("{{ 1 // 0 }}", "{}").is_err());
        assert!(render_str("{{ 9223372036854775807 + 1 }}", "{}").is_err());
        assert!(render_str("{{ -9223372036854775807 - 2 }}", "{}").is_err());
        assert!(render_str("{{ 99999999999 * 99999999999 }}", "{}").is_err());
        assert!(render_str("{{ 2 ** 4000 }}", "{}").is_err());
    }

    #[test]
    fn python_floor_division_and_modulo() {
        assert_eq!(render_str("{{ -7 // 2 }}", "{}").expect("render"), "-4");
        assert_eq!(render_str("{{ -7 % 2 }}", "{}").expect("render"), "1");
        assert_eq!(render_str("{{ 7 // -2 }}", "{}").expect("render"), "-4");
        assert_eq!(render_str("{{ 7 % -2 }}", "{}").expect("render"), "-1");
    }

    #[test]
    fn loop_iterations_are_bounded() {
        let src = "{% for a in xs %}{% for b in xs %}{% for c in xs %}.{% endfor %}{% endfor %}{% endfor %}";
        let xs: Vec<String> = (0..200).map(|i| i.to_string()).collect();
        let ctx = format!("{{\"xs\": [{}]}}", xs.join(","));
        match render_str(src, &ctx) {
            Err(JinjaError::Limit(_)) => {}
            other => panic!("expected a limit error, got {other:?}"),
        }
    }

    #[test]
    fn output_size_is_bounded() {
        let template = JinjaTemplate::compile_with(
            "{% for i in xs %}0123456789{% endfor %}",
            JinjaOptions {
                max_output_bytes: 64,
                ..JinjaOptions::default()
            },
        )
        .expect("compile");
        let ctx = Value::from_json_str(&format!(
            "{{\"xs\": [{}]}}",
            (0..100)
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ))
        .expect("ctx");
        assert!(matches!(template.render(&ctx), Err(JinjaError::Limit(_))));
    }

    #[test]
    fn raise_exception_surfaces_as_template_raise() {
        match render_str("{{ raise_exception('boom') }}", "{}") {
            Err(JinjaError::TemplateRaise(msg)) => assert_eq!(msg, "boom"),
            other => panic!("expected a TemplateRaise, got {other:?}"),
        }
    }

    #[test]
    fn undefined_attribute_access_is_an_error() {
        assert!(render_str("{{ messages[0].role }}", r#"{"messages": []}"#).is_err());
        assert_eq!(
            render_str("[{{ d.missing }}]", r#"{"d": {}}"#).expect("render"),
            "[]"
        );
    }

    #[test]
    fn namespace_survives_a_loop_but_a_plain_set_does_not() {
        assert_eq!(
            render_str(
                "{% set ns = namespace(n=0) %}{% for i in xs %}{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}",
                r#"{"xs": [1, 2, 3]}"#
            )
            .expect("render"),
            "6"
        );
        assert_eq!(
            render_str(
                "{% set n = 'a' %}{% for i in xs %}{% set n = 'b' %}{% endfor %}{{ n }}",
                r#"{"xs": [1]}"#
            )
            .expect("render"),
            "a"
        );
    }

    #[test]
    fn set_inside_if_is_visible_afterwards() {
        assert_eq!(
            render_str(
                "{% set x = 'a' %}{% if true %}{% set x = 'b' %}{% endif %}{{ x }}",
                "{}"
            )
            .expect("render"),
            "b"
        );
    }

    #[test]
    fn assigning_to_a_non_namespace_attribute_errors() {
        assert!(render_str("{% set d = 1 %}{% set d.x = 2 %}", "{}").is_err());
    }

    #[test]
    fn slices_follow_python() {
        assert_eq!(
            render_str("{{ xs[::-1] | join(',') }}", r#"{"xs": ["a","b","c"]}"#).expect("render"),
            "c,b,a"
        );
        assert_eq!(
            render_str("{{ xs[-2:] | join(',') }}", r#"{"xs": ["a","b","c"]}"#).expect("render"),
            "b,c"
        );
        assert_eq!(
            render_str("{{ xs[10:20] | join(',') }}", r#"{"xs": ["a"]}"#).expect("render"),
            ""
        );
        assert!(render_str("{{ xs[::0] }}", r#"{"xs": []}"#).is_err());
    }

    #[test]
    fn macro_arity_is_enforced() {
        assert!(render_str("{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, 2) }}", "{}").is_err());
        assert!(render_str("{% macro m(a) %}{{ a }}{% endmacro %}{{ m(b=1) }}", "{}").is_err());
        assert_eq!(
            render_str(
                "{% macro m(a, b) %}[{{ a }}{{ b }}]{% endmacro %}{{ m(1) }}",
                "{}"
            )
            .expect("render"),
            "[1]"
        );
    }

    #[test]
    fn generation_block_is_transparent() {
        assert_eq!(
            render_str("a{% generation %}b{% endgeneration %}c", "{}").expect("render"),
            "abc"
        );
    }

    #[test]
    fn context_must_be_a_mapping() {
        let template = JinjaTemplate::compile("x").expect("compile");
        assert!(template.render(&Value::Int(1)).is_err());
        assert_eq!(template.render(&Value::Undefined).expect("render"), "x");
    }

    #[test]
    fn calling_a_missing_method_errors() {
        assert!(render_str("{{ s.nosuch() }}", r#"{"s": "a"}"#).is_err());
        assert!(render_str("{{ nosuchfn('x') }}", "{}").is_err());
    }
}
