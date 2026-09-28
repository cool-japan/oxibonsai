//! High-level tool-calling orchestration for OxiBonsai.
//!
//! This module sits on top of the low-level `api_types` helpers and provides a
//! complete tool-use pipeline:
//!
//! 1. **Schema → grammar**: `build_tool_constraint` compiles a list of
//!    [`ToolDefinition`]s into a BNF [`Grammar`] that constrains generation to
//!    valid JSON tool invocations.
//! 2. **Output → call**: `select_tool` parses raw model output and extracts the
//!    first [`ToolCall`] it finds, matching against a provided registry.
//! 3. **Convenience constructors**: `make_tool_call` and `new_tool_call_id`
//!    expose the low-level helpers under module-level names.
//! 4. **XML tool calls** (B2-13/RT-11, bonsai2-design.md §5.4): Bonsai 2's
//!    chat template teaches the model
//!    `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n
//!    </function>\n</tool_call>`, not the JSON payload `select_tool`/
//!    `parse_tool_call` above expect. [`parse_xml_tool_calls`] parses that
//!    shape (zero or more calls per message); [`parse_tool_calls`] is the
//!    single entry point server/extended handlers should call — it tries
//!    the XML shape first and falls back to the legacy JSON shape, so
//!    neither format regresses. A truncated `<tool_call>` (still open at
//!    end of text — the model was cut off, or is still streaming) yields
//!    [`ToolParseError::Truncated`], never a panic and never a half-formed
//!    call.

use std::collections::HashMap;

use crate::api_types::{FunctionCallResult, ToolCall, ToolDefinition};
use crate::grammar::{compile_json_schema, Grammar, Rule, Symbol};

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors produced by the tool-calling layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallError {
    /// The model output contained no tool call.
    NoToolCallFound,
    /// The extracted function name does not match any registered tool.
    UnknownTool { name: String },
    /// The argument JSON in the tool call could not be parsed.
    MalformedArguments { reason: String },
    /// The grammar for a tool definition could not be compiled.
    GrammarCompileError { reason: String },
    /// The provided tool list is empty (nothing to constrain against).
    EmptyToolList,
}

impl std::fmt::Display for ToolCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolCallError::NoToolCallFound => write!(f, "no tool call found in model output"),
            ToolCallError::UnknownTool { name } => write!(f, "unknown tool: '{name}'"),
            ToolCallError::MalformedArguments { reason } => {
                write!(f, "malformed tool arguments: {reason}")
            }
            ToolCallError::GrammarCompileError { reason } => {
                write!(f, "grammar compile error: {reason}")
            }
            ToolCallError::EmptyToolList => write!(f, "tool list is empty"),
        }
    }
}

impl std::error::Error for ToolCallError {}

// ── ID generation ─────────────────────────────────────────────────────────────

/// Generate a unique tool-call identifier with the `call_` prefix.
///
/// Delegates to [`crate::api_types::generate_tool_call_id`] and is exposed
/// here for ergonomic use alongside the rest of the tool-calling API.
pub fn new_tool_call_id() -> String {
    crate::api_types::generate_tool_call_id()
}

// ── Tool call construction ────────────────────────────────────────────────────

/// Construct a [`ToolCall`] from its constituent parts.
///
/// `id` should be produced by [`new_tool_call_id`]. The `arguments` string must
/// be a JSON object serialised to a `String` (the OpenAI wire format).
pub fn make_tool_call(id: String, name: String, arguments: String) -> ToolCall {
    ToolCall {
        id,
        tool_type: "function".to_string(),
        function: FunctionCallResult { name, arguments },
    }
}

// ── Tool selection ────────────────────────────────────────────────────────────

/// Parse raw model output and extract the first tool call.
///
/// The parser looks for the `<tool_call>…</tool_call>` pattern emitted by the
/// model and validates:
///
/// 1. That a `name` field is present.
/// 2. That the name appears in `tools` (when `tools` is non-empty).
/// 3. That the `arguments` value satisfies the matched tool's parameter
///    schema — i.e. it parses as a JSON object and every property listed in
///    the schema's `required` array is present (via
///    [`validate_tool_arguments`]). When `tools` is empty there is no schema
///    to validate against, so only a bare JSON-parse check is performed.
///
/// On success the returned [`ToolCall`] carries a freshly generated ID.
///
/// # Errors
///
/// - [`ToolCallError::NoToolCallFound`] — no `<tool_call>` tag found.
/// - [`ToolCallError::UnknownTool`]    — name not in `tools` registry.
/// - [`ToolCallError::MalformedArguments`] — argument payload is not valid
///   JSON, is not a JSON object, or is missing a required property.
pub fn select_tool(output: &str, tools: &[ToolDefinition]) -> Result<ToolCall, ToolCallError> {
    let call_id = new_tool_call_id();

    // Use the low-level parser from api_types.
    let tool_call = crate::api_types::parse_tool_call(output, &call_id)
        .ok_or(ToolCallError::NoToolCallFound)?;

    // Validate the name against the registered tools (if any), keeping a
    // reference to the matched tool so its parameter schema can be used to
    // validate the arguments below.
    let matched_tool = if tools.is_empty() {
        None
    } else {
        let matched = tools
            .iter()
            .find(|t| t.function.name == tool_call.function.name);
        match matched {
            Some(tool) => Some(tool),
            None => {
                return Err(ToolCallError::UnknownTool {
                    name: tool_call.function.name.clone(),
                });
            }
        }
    };

    match matched_tool {
        // A registered tool was matched: enforce its full parameter schema
        // (JSON object + all required properties present), not just
        // "parses as JSON".
        Some(tool) => {
            validate_tool_arguments(&tool_call.function.arguments, tool)?;
        }
        // No registry to validate against — fall back to a bare JSON parse.
        None => {
            let _parsed: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
                .map_err(|e| ToolCallError::MalformedArguments {
                    reason: e.to_string(),
                })?;
        }
    }

    Ok(tool_call)
}

// ── Grammar constraint construction ──────────────────────────────────────────

/// Compile a list of tool definitions into a BNF grammar that constrains model
/// output to valid JSON tool invocations.
///
/// The generated grammar produces outputs of the form:
///
/// ```text
/// <tool_call>{"name": "<fn_name>", "arguments": <ARGS_SCHEMA>}</tool_call>
/// ```
///
/// where `<ARGS_SCHEMA>` is constrained by the JSON Schema of each function's
/// `parameters` field. When multiple tools are provided the grammar accepts any
/// one of them (union of alternatives).
///
/// # Errors
///
/// Returns [`ToolCallError::EmptyToolList`] when `tools` is empty, or
/// [`ToolCallError::GrammarCompileError`] if any schema fails to compile.
pub fn build_tool_constraint(tools: &[ToolDefinition]) -> Result<Grammar, ToolCallError> {
    if tools.is_empty() {
        return Err(ToolCallError::EmptyToolList);
    }

    // Compile one grammar per tool, then merge into a union.
    let mut args_grammars: Vec<Grammar> = Vec::with_capacity(tools.len());
    for tool in tools {
        let g = compile_json_schema(&tool.function.parameters).map_err(|e| {
            ToolCallError::GrammarCompileError {
                reason: format!("{e}"),
            }
        })?;
        args_grammars.push(g);
    }

    merge_tool_grammars(tools, args_grammars)
}

/// Merge per-tool parameter grammars into a single root grammar that accepts
/// any valid `<tool_call>…</tool_call>` invocation.
///
/// All NT IDs from each arg grammar are remapped to fresh IDs in the merged
/// grammar so there are no collisions. The root NT (id=0) has one rule per tool;
/// each rule is a terminal prefix + the remapped arg-grammar start NT + suffix.
fn merge_tool_grammars(
    tools: &[ToolDefinition],
    args_grammars: Vec<Grammar>,
) -> Result<Grammar, ToolCallError> {
    // Root NT gets id=0. Grammar::new(0) sets start=0.
    let mut merged = Grammar::new(0);
    let root_nt = merged.alloc_nt("tool_call_root"); // id=0
    debug_assert_eq!(root_nt, 0, "root_nt must be 0 to match start");

    // next_nt tracks how many NTs we have allocated so far (root = 1).
    let mut next_nt: usize = 1;

    for (tool_idx, (tool, arg_grammar)) in tools.iter().zip(args_grammars.iter()).enumerate() {
        // Determine the NT count of arg_grammar by finding the maximum NT id
        // referenced across all rules (lhs and rhs), then +1.
        let arg_nt_count = arg_grammar
            .rules
            .iter()
            .flat_map(|r| {
                std::iter::once(r.lhs).chain(r.rhs.iter().filter_map(|s| s.non_terminal_id()))
            })
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let nt_offset = next_nt;

        // Allocate arg_nt_count fresh NTs in the merged grammar.
        for nt_j in 0..arg_nt_count {
            merged.alloc_nt(format!("t{tool_idx}_nt{nt_j}"));
        }
        next_nt += arg_nt_count;

        // Copy rules with remapped NT IDs.
        for rule in &arg_grammar.rules {
            let new_lhs = rule.lhs + nt_offset;
            let new_rhs: Vec<Symbol> = rule
                .rhs
                .iter()
                .map(|sym| match sym {
                    Symbol::NonTerminal(id) => Symbol::NonTerminal(id + nt_offset),
                    Symbol::Terminal(bytes) => Symbol::Terminal(bytes.clone()),
                })
                .collect();
            merged.add_rule(Rule::new(new_lhs, new_rhs));
        }

        // The start NT of the arg grammar, offset into merged scope.
        let args_start = arg_grammar.start + nt_offset;

        // Root rule: root → Terminal(prefix) NonTerminal(args_start) Terminal(suffix)
        let prefix = format!(
            "<tool_call>{{\"name\":\"{}\",\"arguments\":",
            tool.function.name
        );
        let suffix = "}</tool_call>".to_string();

        merged.add_rule(Rule::new(
            root_nt,
            vec![
                Symbol::Terminal(prefix.into_bytes()),
                Symbol::NonTerminal(args_start),
                Symbol::Terminal(suffix.into_bytes()),
            ],
        ));
    }

    Ok(merged)
}

// ── Tool registry helper ──────────────────────────────────────────────────────

/// A lightweight registry of tools keyed by function name for O(1) lookup.
///
/// Build it once from a `&[ToolDefinition]` slice; query it with
/// [`ToolRegistry::get`].
pub struct ToolRegistry<'a> {
    map: HashMap<&'a str, &'a ToolDefinition>,
}

impl<'a> ToolRegistry<'a> {
    /// Build a registry from a slice of tool definitions.
    pub fn new(tools: &'a [ToolDefinition]) -> Self {
        let map = tools
            .iter()
            .map(|t| (t.function.name.as_str(), t))
            .collect();
        Self { map }
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.map.get(name).copied()
    }

    /// Return all registered tool names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().copied()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// `true` if the registry contains no tools.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// ── Argument validation ───────────────────────────────────────────────────────

/// Validate that a JSON arguments string satisfies a tool's parameter schema.
///
/// This is a structural check: confirms the arguments parse as a JSON object
/// and that every required property listed in the schema is present.
///
/// Returns `Ok(serde_json::Value)` on success (the parsed arguments object).
pub fn validate_tool_arguments(
    arguments: &str,
    tool: &ToolDefinition,
) -> Result<serde_json::Value, ToolCallError> {
    let parsed: serde_json::Value =
        serde_json::from_str(arguments).map_err(|e| ToolCallError::MalformedArguments {
            reason: e.to_string(),
        })?;

    if !parsed.is_object() {
        return Err(ToolCallError::MalformedArguments {
            reason: "tool arguments must be a JSON object".to_string(),
        });
    }

    // Validate required properties if defined in the schema.
    if let Some(required) = tool.function.parameters.get("required") {
        if let Some(req_arr) = required.as_array() {
            let obj = parsed.as_object().expect("parsed is_object checked above");
            for req_field in req_arr {
                if let Some(field_name) = req_field.as_str() {
                    if !obj.contains_key(field_name) {
                        return Err(ToolCallError::MalformedArguments {
                            reason: format!("missing required field '{field_name}'"),
                        });
                    }
                }
            }
        }
    }

    Ok(parsed)
}

// ═══════════════════════════════════════════════════════════════════════════
// XML tool-call parsing (B2-13/RT-11, bonsai2-design.md §5.4)
// ═══════════════════════════════════════════════════════════════════════════

/// Errors from [`parse_xml_tool_calls`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParseError {
    /// A `<tool_call>` block was opened but never closed before the text
    /// ended — the model was cut off (max tokens / stop) or, on the
    /// streaming path, has simply not finished emitting it yet. Never
    /// treat this as "no tool call": the caller should either wait for more
    /// tokens (streaming) or report a failed/incomplete generation
    /// (non-streaming) — never silently drop or half-apply the call.
    Truncated,
    /// A `<tool_call>` block closed, but its interior does not match the
    /// `<function=NAME>` / `<parameter=KEY>...</parameter>` shape.
    Malformed(String),
}

impl std::fmt::Display for ToolParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolParseError::Truncated => {
                write!(f, "a <tool_call> block was never closed")
            }
            ToolParseError::Malformed(reason) => write!(f, "malformed <tool_call> XML: {reason}"),
        }
    }
}

impl std::error::Error for ToolParseError {}

/// One parsed `<tool_call><function=NAME>...</function></tool_call>` block.
#[derive(Debug, Clone, PartialEq)]
pub struct XmlToolCall {
    /// The function name from `<function=NAME>`.
    pub name: String,
    /// `<parameter=KEY>VALUE</parameter>` pairs, in the order they appeared.
    /// A `VALUE` that parses as JSON (a number, boolean, array, object, or
    /// `null`) is stored as that parsed value; otherwise it is stored as a
    /// JSON string — mirroring the template's own convention of emitting
    /// `|tojson` for non-string arguments and raw text for string ones
    /// (bonsai2-design.md §5.4).
    pub arguments: serde_json::Map<String, serde_json::Value>,
}

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN_PREFIX: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN_PREFIX: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

/// Parse zero or more `<tool_call>` blocks out of an assistant message.
///
/// Returns `(leading_natural_language, calls)`: text **before** the first
/// `<tool_call>` is kept verbatim (the template's own `<IMPORTANT>` block
/// allows reasoning before a call, never after). Text **after** the last
/// `</tool_call>` that is not itself another `<tool_call>` block is a
/// protocol violation per that same `<IMPORTANT>` block — it is logged via
/// `tracing::warn!` and dropped rather than silently concatenated onto
/// `leading_natural_language` or a call's arguments.
///
/// Every byte offset this function slices at comes from `str::find`-ing a
/// fixed ASCII literal (or is `0`/`text.len()`), which is always a valid
/// UTF-8 char boundary — this function never panics on any input, verified
/// by `parse_xml_tool_calls_never_panics*` below.
///
/// # Errors
///
/// - [`ToolParseError::Truncated`] — a `<tool_call>` was opened but the
///   matching `</tool_call>` never appears.
/// - [`ToolParseError::Malformed`] — a closed `<tool_call>...</tool_call>`
///   block's interior does not match the expected shape (no
///   `<function=NAME>`, no `</function>`, or a `<parameter=KEY>` with no
///   matching `</parameter>`).
pub fn parse_xml_tool_calls(text: &str) -> Result<(String, Vec<XmlToolCall>), ToolParseError> {
    let Some(first_open) = text.find(TOOL_CALL_OPEN) else {
        return Ok((text.to_string(), Vec::new()));
    };
    let leading = text[..first_open].to_string();
    let mut rest = &text[first_open..];
    let mut calls = Vec::new();

    while let Some(inner_and_after) = rest.strip_prefix(TOOL_CALL_OPEN) {
        let Some(close_rel) = inner_and_after.find(TOOL_CALL_CLOSE) else {
            return Err(ToolParseError::Truncated);
        };
        let inner = &inner_and_after[..close_rel];
        calls.push(parse_function_block(inner)?);
        rest = &inner_and_after[close_rel + TOOL_CALL_CLOSE.len()..];

        // Consecutive tool-call blocks may be separated by nothing or by
        // whitespace (the reference template emits them back-to-back); skip
        // that separator before checking for the next block.
        let after_whitespace = rest.trim_start_matches(['\n', '\r', ' ', '\t']);
        if after_whitespace.starts_with(TOOL_CALL_OPEN) {
            rest = after_whitespace;
            continue;
        }
        if !rest.is_empty() {
            // Trailing content after the last real `<tool_call>` block that
            // is not itself another block: the `<IMPORTANT>` block the
            // template teaches the model states reasoning must come
            // *before* a call, never after — this is a protocol violation,
            // not natural language to append anywhere. Warn and drop it.
            tracing::warn!(
                target: "oxibonsai_runtime::tool_calling",
                dropped_bytes = rest.len(),
                "text after the final </tool_call> is a protocol violation \
                 (reasoning must precede a tool call, never follow it); dropping it"
            );
        }
        break;
    }

    Ok((leading, calls))
}

/// Parse one `<tool_call>` block's interior (everything between
/// `<tool_call>` and `</tool_call>`, exclusive).
fn parse_function_block(inner: &str) -> Result<XmlToolCall, ToolParseError> {
    let after_fn_prefix = inner
        .find(FUNCTION_OPEN_PREFIX)
        .map(|i| &inner[i + FUNCTION_OPEN_PREFIX.len()..])
        .ok_or_else(|| ToolParseError::Malformed("missing <function=NAME> tag".to_string()))?;

    let name_end = after_fn_prefix
        .find('>')
        .ok_or_else(|| ToolParseError::Malformed("unterminated <function=NAME> tag".to_string()))?;
    let name = after_fn_prefix[..name_end].trim().to_string();
    if name.is_empty() {
        return Err(ToolParseError::Malformed("empty function name".to_string()));
    }

    let body_and_after = &after_fn_prefix[name_end + 1..];
    let function_close_rel = body_and_after
        .find(FUNCTION_CLOSE)
        .ok_or_else(|| ToolParseError::Malformed("missing </function> tag".to_string()))?;
    let body = &body_and_after[..function_close_rel];

    let mut arguments = serde_json::Map::new();
    let mut cursor = body;
    while let Some(open_rel) = cursor.find(PARAMETER_OPEN_PREFIX) {
        let after_open = &cursor[open_rel + PARAMETER_OPEN_PREFIX.len()..];
        let Some(key_end) = after_open.find('>') else {
            return Err(ToolParseError::Malformed(
                "unterminated <parameter=KEY> tag".to_string(),
            ));
        };
        let key = after_open[..key_end].trim().to_string();
        let value_and_after = &after_open[key_end + 1..];
        let Some(close_rel) = value_and_after.find(PARAMETER_CLOSE) else {
            return Err(ToolParseError::Malformed(
                "missing </parameter> tag".to_string(),
            ));
        };
        let raw_value = value_and_after[..close_rel].trim();
        // "Parsed as JSON when they parse, else kept as strings" (§5.4):
        // a bare number/bool/array/object/null round-trips through
        // `serde_json`; anything else (including a bare word with no
        // quotes, which the template never JSON-quotes for a string
        // argument) becomes a JSON string holding the literal text.
        let value = serde_json::from_str::<serde_json::Value>(raw_value)
            .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
        if !key.is_empty() {
            arguments.insert(key, value);
        }
        cursor = &value_and_after[close_rel + PARAMETER_CLOSE.len()..];
    }

    Ok(XmlToolCall { name, arguments })
}

/// Convert parsed XML tool calls into OpenAI-shaped [`ToolCall`]s, minting a
/// fresh id for each (see [`new_tool_call_id`]).
pub fn xml_tool_calls_to_openai(calls: Vec<XmlToolCall>) -> Vec<ToolCall> {
    calls
        .into_iter()
        .map(|call| {
            let arguments =
                serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string());
            make_tool_call(new_tool_call_id(), call.name, arguments)
        })
        .collect()
}

/// The outcome of [`parse_tool_calls`] — the single entry point server /
/// extended-endpoint handlers should call once generation has finished.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolCallParseOutcome {
    /// No tool-call markup of any kind (XML or legacy JSON) was found; the
    /// whole text is ordinary content.
    None,
    /// One or more complete tool calls were found.
    Found {
        /// Text before the first tool call — the model's natural-language
        /// preamble, if any.
        leading_text: String,
        /// The parsed calls, OpenAI-shaped and each carrying a fresh id.
        calls: Vec<ToolCall>,
    },
    /// A `<tool_call>` was opened but never closed. The caller must not
    /// treat this as "no tool call" (which would leak the partial XML as
    /// visible assistant content) — see [`ToolParseError::Truncated`].
    Truncated {
        /// Text before the truncated `<tool_call>` block — the model's
        /// natural-language preamble, if any. The partial `<tool_call>...`
        /// markup itself is never included here: a caller that reports
        /// this text as `message.content` must never surface a half-formed
        /// tool-call tag as if it were prose.
        leading_text: String,
    },
}

/// Parse tool calls out of a finished assistant message: tries the Bonsai 2
/// XML shape first ([`parse_xml_tool_calls`]), falling back to the legacy
/// JSON payload shape ([`crate::api_types::parse_tool_call`]) so a model
/// that emits the older `<tool_call>{"name":...}</tool_call>` form is not
/// regressed by adding XML support.
pub fn parse_tool_calls(text: &str) -> ToolCallParseOutcome {
    match parse_xml_tool_calls(text) {
        Ok((leading, calls)) if !calls.is_empty() => ToolCallParseOutcome::Found {
            leading_text: leading,
            calls: xml_tool_calls_to_openai(calls),
        },
        Err(ToolParseError::Truncated) => ToolCallParseOutcome::Truncated {
            leading_text: leading_text_before_tool_call(text).to_string(),
        },
        // No `<tool_call>` tag at all, or one that closed but whose
        // interior isn't the XML shape: try the legacy JSON payload before
        // concluding there is no tool call.
        Ok(_) | Err(ToolParseError::Malformed(_)) => legacy_json_tool_call(text),
    }
}

/// Text before the first `<tool_call>` marker, or the whole string when
/// there is none. Used to recover [`ToolCallParseOutcome::Truncated`]'s
/// `leading_text` — [`ToolParseError::Truncated`] itself stays a plain unit
/// variant (its own extensive `assert_eq!` coverage below pins that shape),
/// so the higher-level [`parse_tool_calls`] re-derives the preamble here
/// rather than threading it through the lower-level error type.
fn leading_text_before_tool_call(text: &str) -> &str {
    text.find(TOOL_CALL_OPEN)
        .map(|i| &text[..i])
        .unwrap_or(text)
}

/// The legacy `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
/// fallback [`parse_tool_calls`] uses when the XML shape does not match.
fn legacy_json_tool_call(text: &str) -> ToolCallParseOutcome {
    match crate::api_types::parse_tool_call(text, &new_tool_call_id()) {
        Some(call) => {
            let split_at = text.find(TOOL_CALL_OPEN).unwrap_or(text.len());
            ToolCallParseOutcome::Found {
                leading_text: text[..split_at].to_string(),
                calls: vec![call],
            }
        }
        None => ToolCallParseOutcome::None,
    }
}

/// Incremental wrapper around [`parse_xml_tool_calls`] for the SSE
/// streaming path: feed it the growing decoded-so-far assistant text and it
/// reports only the calls newly *completed* since the last call, so a
/// caller can stream `tool_calls` deltas without re-emitting the same call
/// twice or emitting a partially-formed one.
#[derive(Debug, Clone, Default)]
pub struct XmlToolCallStreamParser {
    emitted: usize,
}

impl XmlToolCallStreamParser {
    /// Build a fresh parser with nothing emitted yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Re-parse the full text generated so far; returns only the tool calls
    /// newly completed since the last call to this method. Returns an empty
    /// `Vec` while a call is still open ([`ToolParseError::Truncated`]) or
    /// its interior does not (yet) match the expected shape — both are
    /// "not yet", not errors, since more tokens may still complete it; only
    /// [`Self::finish`] turns a still-open block into a hard error.
    pub fn feed(&mut self, text_so_far: &str) -> Vec<XmlToolCall> {
        match parse_xml_tool_calls(text_so_far) {
            Ok((_, calls)) if calls.len() > self.emitted => {
                let new_calls = calls[self.emitted..].to_vec();
                self.emitted = calls.len();
                new_calls
            }
            _ => Vec::new(),
        }
    }

    /// Number of calls already reported by [`Self::feed`].
    pub fn emitted_count(&self) -> usize {
        self.emitted
    }

    /// Call once generation has ended (EOS / stop / max tokens) with the
    /// complete final text. `Ok(())` when nothing was left open;
    /// `Err` — always [`ToolParseError`], never a panic — when a
    /// `<tool_call>` is still unclosed or malformed at end of stream.
    pub fn finish(&self, final_text: &str) -> Result<(), ToolParseError> {
        parse_xml_tool_calls(final_text).map(|_| ())
    }
}

#[cfg(test)]
mod xml_tool_call_tests {
    use super::*;

    // ── The template's own worked example (bonsai2-design.md §5.4) ────────

    const TEMPLATE_EXAMPLE: &str = concat!(
        "<tool_call>\n",
        "<function=example_function_name>\n",
        "<parameter=example_parameter_1>\n",
        "value_1\n",
        "</parameter>\n",
        "<parameter=example_parameter_2>\n",
        "This is the value for the second parameter\n",
        "that can span\n",
        "multiple lines\n",
        "</parameter>\n",
        "</function>\n",
        "</tool_call>",
    );

    #[test]
    fn round_trips_the_templates_own_example() {
        let (leading, calls) = parse_xml_tool_calls(TEMPLATE_EXAMPLE).expect("must parse");
        assert_eq!(leading, "");
        assert_eq!(calls.len(), 1);
        let call = &calls[0];
        assert_eq!(call.name, "example_function_name");
        assert_eq!(
            call.arguments.get("example_parameter_1"),
            Some(&serde_json::Value::String("value_1".to_string()))
        );
        assert_eq!(
            call.arguments.get("example_parameter_2"),
            Some(&serde_json::Value::String(
                "This is the value for the second parameter\nthat can span\nmultiple lines"
                    .to_string()
            ))
        );
    }

    #[test]
    fn leading_natural_language_is_preserved() {
        let text = format!("Let me check that for you.\n{TEMPLATE_EXAMPLE}");
        let (leading, calls) = parse_xml_tool_calls(&text).expect("must parse");
        assert_eq!(leading, "Let me check that for you.\n");
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn no_tool_call_tag_returns_empty_with_full_text_as_leading() {
        let (leading, calls) = parse_xml_tool_calls("just a normal answer").expect("must parse");
        assert_eq!(leading, "just a normal answer");
        assert!(calls.is_empty());
    }

    #[test]
    fn numeric_and_boolean_arguments_parse_as_json_not_strings() {
        let text = concat!(
            "<tool_call>\n<function=set_count>\n",
            "<parameter=count>\n42\n</parameter>\n",
            "<parameter=enabled>\ntrue\n</parameter>\n",
            "</function>\n</tool_call>",
        );
        let (_, calls) = parse_xml_tool_calls(text).expect("must parse");
        assert_eq!(
            calls[0].arguments.get("count"),
            Some(&serde_json::Value::Number(42.into()))
        );
        assert_eq!(
            calls[0].arguments.get("enabled"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[test]
    fn multiple_tool_calls_all_parsed() {
        let text = concat!(
            "<tool_call>\n<function=a>\n</function>\n</tool_call>\n",
            "<tool_call>\n<function=b>\n</function>\n</tool_call>",
        );
        let (_, calls) = parse_xml_tool_calls(text).expect("must parse");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn function_with_no_parameters() {
        let text = "<tool_call>\n<function=ping>\n</function>\n</tool_call>";
        let (_, calls) = parse_xml_tool_calls(text).expect("must parse");
        assert_eq!(calls[0].name, "ping");
        assert!(calls[0].arguments.is_empty());
    }

    // ── Truncation (RT-11) ─────────────────────────────────────────────────

    #[test]
    fn truncated_after_open_tag_is_truncated_not_panic() {
        let err = parse_xml_tool_calls("<tool_call>\n<function=foo>\n<parameter=x>\nbar")
            .expect_err("must error");
        assert_eq!(err, ToolParseError::Truncated);
    }

    #[test]
    fn truncated_bare_open_tag_is_truncated() {
        let err = parse_xml_tool_calls("thinking...<tool_call>").expect_err("must error");
        assert_eq!(err, ToolParseError::Truncated);
    }

    #[test]
    fn first_call_complete_second_call_truncated_is_truncated_not_partial_success() {
        let text = format!("{TEMPLATE_EXAMPLE}\n<tool_call>\n<function=incomplete");
        let err = parse_xml_tool_calls(&text).expect_err("must error, never a half-formed call");
        assert_eq!(err, ToolParseError::Truncated);
    }

    // ── Malformed (closed, but wrong interior shape) ───────────────────────

    #[test]
    fn closed_but_no_function_tag_is_malformed() {
        let err = parse_xml_tool_calls("<tool_call>\nnot a function\n</tool_call>")
            .expect_err("must error");
        assert!(matches!(err, ToolParseError::Malformed(_)));
    }

    #[test]
    fn closed_but_unterminated_function_name_is_malformed() {
        let err = parse_xml_tool_calls("<tool_call>\n<function=foo\n</tool_call>")
            .expect_err("must error");
        assert!(matches!(err, ToolParseError::Malformed(_)));
    }

    #[test]
    fn closed_but_missing_function_close_is_malformed() {
        let err = parse_xml_tool_calls("<tool_call>\n<function=foo>\n</tool_call>")
            .expect_err("must error");
        assert!(matches!(err, ToolParseError::Malformed(_)));
    }

    #[test]
    fn empty_function_name_is_malformed() {
        let err = parse_xml_tool_calls("<tool_call>\n<function=>\n</function>\n</tool_call>")
            .expect_err("must error");
        assert!(matches!(err, ToolParseError::Malformed(_)));
    }

    // ── xml_tool_calls_to_openai ───────────────────────────────────────────

    #[test]
    fn xml_tool_calls_to_openai_shape() {
        let (_, calls) = parse_xml_tool_calls(TEMPLATE_EXAMPLE).expect("must parse");
        let openai = xml_tool_calls_to_openai(calls);
        assert_eq!(openai.len(), 1);
        assert_eq!(openai[0].function.name, "example_function_name");
        assert!(openai[0].id.starts_with("call_"));
        let args: serde_json::Value =
            serde_json::from_str(&openai[0].function.arguments).expect("valid JSON");
        assert_eq!(args["example_parameter_1"], "value_1");
    }

    #[test]
    fn xml_tool_calls_get_distinct_ids() {
        let text = concat!(
            "<tool_call>\n<function=a>\n</function>\n</tool_call>\n",
            "<tool_call>\n<function=b>\n</function>\n</tool_call>",
        );
        let (_, calls) = parse_xml_tool_calls(text).expect("must parse");
        let openai = xml_tool_calls_to_openai(calls);
        assert_ne!(openai[0].id, openai[1].id);
    }

    // ── parse_tool_calls: XML-first, JSON-fallback dispatcher ──────────────

    #[test]
    fn parse_tool_calls_prefers_xml() {
        let outcome = parse_tool_calls(TEMPLATE_EXAMPLE);
        match outcome {
            ToolCallParseOutcome::Found { calls, .. } => {
                assert_eq!(calls[0].function.name, "example_function_name");
            }
            other => panic!("expected Found, got {other:?}"),
        }
    }

    #[test]
    fn parse_tool_calls_falls_back_to_legacy_json() {
        let text = r#"<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>"#;
        let outcome = parse_tool_calls(text);
        match outcome {
            ToolCallParseOutcome::Found { calls, .. } => {
                assert_eq!(calls[0].function.name, "get_weather");
            }
            other => panic!("expected Found (legacy JSON fallback), got {other:?}"),
        }
    }

    #[test]
    fn parse_tool_calls_none_when_no_markup() {
        assert_eq!(parse_tool_calls("plain answer"), ToolCallParseOutcome::None);
    }

    #[test]
    fn parse_tool_calls_truncated_propagates() {
        assert_eq!(
            parse_tool_calls("<tool_call>\n<function=foo"),
            ToolCallParseOutcome::Truncated {
                leading_text: String::new()
            }
        );
    }

    #[test]
    fn parse_tool_calls_truncated_preserves_the_leading_preamble() {
        // A truncated call must still report whatever natural-language text
        // came before it, not just an empty string -- the caller uses this
        // as `message.content` instead of the half-formed XML itself.
        let outcome = parse_tool_calls("Let me check.\n<tool_call>\n<function=foo");
        assert_eq!(
            outcome,
            ToolCallParseOutcome::Truncated {
                leading_text: "Let me check.\n".to_string()
            }
        );
    }

    // ── XmlToolCallStreamParser ─────────────────────────────────────────────

    #[test]
    fn stream_parser_emits_only_newly_completed_calls() {
        let mut parser = XmlToolCallStreamParser::new();
        let prefix = "Let me check.\n<tool_call>\n<function=a>\n</function>\n</tool_call>";
        assert_eq!(parser.feed(prefix).len(), 1);
        assert_eq!(parser.emitted_count(), 1);
        // Feeding the same text again reports nothing new.
        assert!(parser.feed(prefix).is_empty());
        let extended = format!("{prefix}\n<tool_call>\n<function=b>\n</function>\n</tool_call>");
        let newly = parser.feed(&extended);
        assert_eq!(newly.len(), 1);
        assert_eq!(newly[0].name, "b");
        assert_eq!(parser.emitted_count(), 2);
    }

    #[test]
    fn stream_parser_feed_is_empty_while_call_is_still_open() {
        let mut parser = XmlToolCallStreamParser::new();
        assert!(parser
            .feed("<tool_call>\n<function=a>\n<parameter=x>\nun")
            .is_empty());
    }

    #[test]
    fn stream_parser_finish_ok_when_nothing_open() {
        let parser = XmlToolCallStreamParser::new();
        assert!(parser.finish(TEMPLATE_EXAMPLE).is_ok());
        assert!(parser.finish("plain text, no tool call").is_ok());
    }

    #[test]
    fn stream_parser_finish_errors_when_still_open() {
        let parser = XmlToolCallStreamParser::new();
        let err = parser
            .finish("<tool_call>\n<function=a>\nstill going")
            .expect_err("must error");
        assert_eq!(err, ToolParseError::Truncated);
    }

    // ── ToolParseError::Display ─────────────────────────────────────────────

    #[test]
    fn tool_parse_error_display_not_empty() {
        assert!(!ToolParseError::Truncated.to_string().is_empty());
        assert!(!ToolParseError::Malformed("x".to_string())
            .to_string()
            .is_empty());
    }

    // ── Fuzz: never panics on arbitrary input ──────────────────────────────

    mod fuzz {
        use super::*;
        use proptest::prelude::*;

        /// A strategy biased toward the tag fragments this parser looks
        /// for, so truncation/malformed-interior edge cases are actually
        /// exercised — uniform random Unicode text almost never contains
        /// `<tool_call>` and would rarely reach past the first `.find`.
        fn tag_fragment_text() -> impl Strategy<Value = String> {
            prop::collection::vec(
                prop_oneof![
                    Just("<tool_call>".to_string()),
                    Just("</tool_call>".to_string()),
                    Just("<function=".to_string()),
                    Just(">".to_string()),
                    Just("</function>".to_string()),
                    Just("<parameter=".to_string()),
                    Just("</parameter>".to_string()),
                    Just("\n".to_string()),
                    Just("{\"a\":1}".to_string()),
                    "[a-zA-Z0-9_]{0,6}".prop_map(|s| s),
                ],
                0..16,
            )
            .prop_map(|parts| parts.concat())
        }

        proptest! {
            #[test]
            fn parse_xml_tool_calls_never_panics_on_arbitrary_text(s in ".{0,300}") {
                let _ = parse_xml_tool_calls(&s);
            }

            #[test]
            fn parse_xml_tool_calls_never_panics_on_tag_fragments(s in tag_fragment_text()) {
                let _ = parse_xml_tool_calls(&s);
            }

            #[test]
            fn parse_tool_calls_never_panics(s in tag_fragment_text()) {
                let _ = parse_tool_calls(&s);
            }

            #[test]
            fn stream_parser_never_panics_incrementally(chunks in prop::collection::vec(tag_fragment_text(), 0..8)) {
                let mut parser = XmlToolCallStreamParser::new();
                let mut acc = String::new();
                for chunk in chunks {
                    acc.push_str(&chunk);
                    let _ = parser.feed(&acc);
                }
                let _ = parser.finish(&acc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_tool() -> ToolDefinition {
        ToolDefinition::function(
            "get_weather",
            Some("Get current weather".to_string()),
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"}
                },
                "required": ["location"]
            }),
        )
    }

    fn calc_tool() -> ToolDefinition {
        ToolDefinition::function(
            "calculate",
            Some("Perform a calculation".to_string()),
            json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }),
        )
    }

    // ── new_tool_call_id ──────────────────────────────────────────────────────

    #[test]
    fn tool_call_id_has_call_prefix() {
        let id = new_tool_call_id();
        assert!(id.starts_with("call_"), "id={id}");
    }

    #[test]
    fn tool_call_ids_are_generated_repeatedly() {
        let ids: Vec<_> = (0..5).map(|_| new_tool_call_id()).collect();
        for id in &ids {
            assert!(id.starts_with("call_"));
        }
    }

    // ── make_tool_call ────────────────────────────────────────────────────────

    #[test]
    fn make_tool_call_round_trips_fields() {
        let tc = make_tool_call(
            "call_abc123".to_string(),
            "get_weather".to_string(),
            r#"{"location":"Paris"}"#.to_string(),
        );
        assert_eq!(tc.id, "call_abc123");
        assert_eq!(tc.tool_type, "function");
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.function.arguments, r#"{"location":"Paris"}"#);
    }

    // ── select_tool ───────────────────────────────────────────────────────────

    #[test]
    fn select_tool_parses_xml_wrapper() {
        let output =
            r#"<tool_call>{"name":"get_weather","arguments":{"location":"Tokyo"}}</tool_call>"#;
        let tools = vec![weather_tool()];
        let tc = select_tool(output, &tools).expect("should parse");
        assert_eq!(tc.function.name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&tc.function.arguments).expect("valid json");
        assert_eq!(args["location"], "Tokyo");
    }

    #[test]
    fn select_tool_no_tag_returns_not_found() {
        let output = "I will now get the weather for Paris.";
        let tools = vec![weather_tool()];
        assert!(matches!(
            select_tool(output, &tools),
            Err(ToolCallError::NoToolCallFound)
        ));
    }

    #[test]
    fn select_tool_unknown_name_returns_error() {
        let output = r#"<tool_call>{"name":"unknown_fn","arguments":{}}</tool_call>"#;
        let tools = vec![weather_tool()];
        assert!(matches!(
            select_tool(output, &tools),
            Err(ToolCallError::UnknownTool { .. })
        ));
    }

    #[test]
    fn select_tool_empty_tools_skips_name_check() {
        let output = r#"<tool_call>{"name":"any_function","arguments":{}}</tool_call>"#;
        let tc = select_tool(output, &[]).expect("should accept any tool");
        assert_eq!(tc.function.name, "any_function");
    }

    // Regression test for issue #56: select_tool must reject a tool call whose
    // arguments are missing a required property, not just check "is valid JSON".
    #[test]
    fn select_tool_missing_required_argument_returns_malformed_arguments() {
        let output = r#"<tool_call>{"name":"get_weather","arguments":{}}</tool_call>"#;
        let tools = vec![weather_tool()];
        assert!(matches!(
            select_tool(output, &tools),
            Err(ToolCallError::MalformedArguments { .. })
        ));
    }

    #[test]
    fn select_tool_non_object_arguments_returns_malformed_arguments() {
        let output = r#"<tool_call>{"name":"get_weather","arguments":"not-an-object"}</tool_call>"#;
        let tools = vec![weather_tool()];
        assert!(matches!(
            select_tool(output, &tools),
            Err(ToolCallError::MalformedArguments { .. })
        ));
    }

    // ── validate_tool_arguments ───────────────────────────────────────────────

    #[test]
    fn validate_tool_args_all_required_present() {
        let tool = weather_tool();
        let args = r#"{"location":"Berlin","unit":"celsius"}"#;
        assert!(validate_tool_arguments(args, &tool).is_ok());
    }

    #[test]
    fn validate_tool_args_missing_required_returns_error() {
        let tool = weather_tool();
        let args = r#"{"unit":"fahrenheit"}"#;
        assert!(matches!(
            validate_tool_arguments(args, &tool),
            Err(ToolCallError::MalformedArguments { .. })
        ));
    }

    #[test]
    fn validate_tool_args_invalid_json_returns_error() {
        let tool = weather_tool();
        assert!(matches!(
            validate_tool_arguments("{bad json}", &tool),
            Err(ToolCallError::MalformedArguments { .. })
        ));
    }

    // ── build_tool_constraint ─────────────────────────────────────────────────

    #[test]
    fn build_tool_constraint_empty_tools_returns_error() {
        assert!(matches!(
            build_tool_constraint(&[]),
            Err(ToolCallError::EmptyToolList)
        ));
    }

    #[test]
    fn build_tool_constraint_single_tool_returns_grammar() {
        let tools = vec![weather_tool()];
        let g = build_tool_constraint(&tools).expect("should build grammar");
        assert!(!g.rules.is_empty(), "grammar must have rules");
    }

    #[test]
    fn build_tool_constraint_multi_tool_root_has_one_rule_per_tool() {
        let tools = vec![weather_tool(), calc_tool()];
        let g = build_tool_constraint(&tools).expect("should build grammar");
        let root_rules: Vec<_> = g.rules.iter().filter(|r| r.lhs == g.start).collect();
        assert_eq!(root_rules.len(), 2, "one rule per tool in root NT");
    }

    // ── ToolRegistry ──────────────────────────────────────────────────────────

    #[test]
    fn tool_registry_lookup_by_name() {
        let tools = vec![weather_tool(), calc_tool()];
        let reg = ToolRegistry::new(&tools);
        assert!(reg.get("get_weather").is_some());
        assert!(reg.get("calculate").is_some());
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn tool_registry_len_and_is_empty() {
        let tools = vec![weather_tool()];
        let reg = ToolRegistry::new(&tools);
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        let empty: Vec<ToolDefinition> = vec![];
        let er = ToolRegistry::new(&empty);
        assert!(er.is_empty());
    }

    // ── ToolCallError display ─────────────────────────────────────────────────

    #[test]
    fn tool_call_error_display_not_empty() {
        let errors = [
            ToolCallError::NoToolCallFound,
            ToolCallError::UnknownTool { name: "foo".into() },
            ToolCallError::MalformedArguments {
                reason: "bad".into(),
            },
            ToolCallError::GrammarCompileError {
                reason: "oops".into(),
            },
            ToolCallError::EmptyToolList,
        ];
        for e in &errors {
            assert!(!e.to_string().is_empty(), "error {e:?} has empty Display");
        }
    }
}
