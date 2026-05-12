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
/// 3. That the `arguments` value, if present, is a valid JSON object.
///
/// On success the returned [`ToolCall`] carries a freshly generated ID.
///
/// # Errors
///
/// - [`ToolCallError::NoToolCallFound`] — no `<tool_call>` tag found.
/// - [`ToolCallError::UnknownTool`]    — name not in `tools` registry.
/// - [`ToolCallError::MalformedArguments`] — argument payload is not valid JSON.
pub fn select_tool(output: &str, tools: &[ToolDefinition]) -> Result<ToolCall, ToolCallError> {
    let call_id = new_tool_call_id();

    // Use the low-level parser from api_types.
    let tool_call = crate::api_types::parse_tool_call(output, &call_id)
        .ok_or(ToolCallError::NoToolCallFound)?;

    // Validate the name against the registered tools (if any).
    if !tools.is_empty() {
        let known = tools
            .iter()
            .any(|t| t.function.name == tool_call.function.name);
        if !known {
            return Err(ToolCallError::UnknownTool {
                name: tool_call.function.name.clone(),
            });
        }
    }

    // Validate that the arguments string is valid JSON.
    let _parsed: serde_json::Value =
        serde_json::from_str(&tool_call.function.arguments).map_err(|e| {
            ToolCallError::MalformedArguments {
                reason: e.to_string(),
            }
        })?;

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
