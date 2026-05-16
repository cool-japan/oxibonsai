use oxibonsai_runtime::api_types::{ToolCall, ToolDefinition};
use oxibonsai_runtime::tool_calling::{
    build_tool_constraint, make_tool_call, new_tool_call_id, select_tool, validate_tool_arguments,
    ToolCallError, ToolRegistry,
};
use serde_json::json;

// ── Fixtures ──────────────────────────────────────────────────────────────────

fn weather_tool() -> ToolDefinition {
    ToolDefinition::function(
        "get_weather",
        Some("Retrieve current weather for a location".to_string()),
        json!({
            "type": "object",
            "properties": {
                "location": { "type": "string" },
                "unit":     { "type": "string", "enum": ["celsius", "fahrenheit"] }
            },
            "required": ["location"],
            "additionalProperties": false
        }),
    )
}

fn search_tool() -> ToolDefinition {
    ToolDefinition::function(
        "web_search",
        Some("Search the internet".to_string()),
        json!({
            "type": "object",
            "properties": {
                "query":    { "type": "string" },
                "max_results": { "type": "integer" }
            },
            "required": ["query"],
            "additionalProperties": false
        }),
    )
}

fn calc_tool() -> ToolDefinition {
    ToolDefinition::function(
        "calculate",
        None,
        json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string" }
            },
            "required": ["expression"],
            "additionalProperties": false
        }),
    )
}

fn wrap_tool_call(name: &str, args: &serde_json::Value) -> String {
    format!(
        r#"<tool_call>{{"name":"{}","arguments":{}}}</tool_call>"#,
        name,
        serde_json::to_string(args).unwrap()
    )
}

// ── new_tool_call_id ──────────────────────────────────────────────────────────

#[test]
fn tool_call_id_starts_with_call_prefix() {
    let id = new_tool_call_id();
    assert!(
        id.starts_with("call_"),
        "expected 'call_' prefix, got '{id}'"
    );
}

#[test]
fn tool_call_id_non_empty_suffix() {
    let id = new_tool_call_id();
    assert!(id.len() > 5, "id too short: '{id}'");
}

#[test]
fn multiple_ids_generated_without_panic() {
    let ids: Vec<String> = (0..20).map(|_| new_tool_call_id()).collect();
    assert_eq!(ids.len(), 20);
    for id in &ids {
        assert!(id.starts_with("call_"));
    }
}

// ── make_tool_call ─────────────────────────────────────────────────────────────

#[test]
fn make_tool_call_preserves_all_fields() {
    let tc: ToolCall = make_tool_call(
        "call_0001".to_string(),
        "get_weather".to_string(),
        r#"{"location":"Paris"}"#.to_string(),
    );
    assert_eq!(tc.id, "call_0001");
    assert_eq!(tc.tool_type, "function");
    assert_eq!(tc.function.name, "get_weather");
    assert_eq!(tc.function.arguments, r#"{"location":"Paris"}"#);
}

#[test]
fn make_tool_call_arguments_can_be_empty_object() {
    let tc = make_tool_call(
        "call_x".to_string(),
        "no_args".to_string(),
        "{}".to_string(),
    );
    assert_eq!(tc.function.arguments, "{}");
}

#[test]
fn make_tool_call_type_is_always_function() {
    let tc = make_tool_call("id".to_string(), "fn".to_string(), "{}".to_string());
    assert_eq!(tc.tool_type, "function");
}

// ── select_tool ───────────────────────────────────────────────────────────────

#[test]
fn select_tool_simple_weather_call() {
    let output = wrap_tool_call("get_weather", &json!({"location": "London"}));
    let tools = vec![weather_tool()];
    let tc = select_tool(&output, &tools).expect("should parse");
    assert_eq!(tc.function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
    assert_eq!(args["location"], "London");
}

#[test]
fn select_tool_with_surrounding_text() {
    // Model sometimes emits prose before/after the tag.
    let output = format!(
        "Let me look that up for you.\n{}\nThat's all.",
        wrap_tool_call("get_weather", &json!({"location": "Berlin"}))
    );
    let tc = select_tool(&output, &[weather_tool()]).expect("should parse");
    assert_eq!(tc.function.name, "get_weather");
}

#[test]
fn select_tool_no_tag_returns_no_tool_call_found() {
    let err = select_tool(
        "I'll get the weather for you right away!",
        &[weather_tool()],
    )
    .expect_err("should fail");
    assert!(matches!(err, ToolCallError::NoToolCallFound), "{err}");
}

#[test]
fn select_tool_empty_tag_returns_error() {
    // Empty inner JSON — tool_call tag present but no name field.
    let output = "<tool_call>{}</tool_call>";
    // parse_tool_call returns None when "name" is absent → NoToolCallFound.
    let result = select_tool(output, &[weather_tool()]);
    assert!(result.is_err());
}

#[test]
fn select_tool_unknown_function_name_returns_unknown_tool() {
    let output = wrap_tool_call("delete_everything", &json!({}));
    let err = select_tool(&output, &[weather_tool(), calc_tool()]).expect_err("should fail");
    assert!(matches!(err, ToolCallError::UnknownTool { name } if name == "delete_everything"));
}

#[test]
fn select_tool_empty_registry_accepts_any_name() {
    let output = wrap_tool_call("arbitrary_function", &json!({"x": 1}));
    let tc = select_tool(&output, &[]).expect("empty registry should accept any name");
    assert_eq!(tc.function.name, "arbitrary_function");
}

#[test]
fn select_tool_multi_registry_accepts_first_match() {
    let tools = vec![weather_tool(), search_tool(), calc_tool()];
    let output = wrap_tool_call("web_search", &json!({"query": "Rust async"}));
    let tc = select_tool(&output, &tools).expect("should parse");
    assert_eq!(tc.function.name, "web_search");
}

#[test]
fn select_tool_assigned_id_starts_with_call_prefix() {
    let output = wrap_tool_call("get_weather", &json!({"location": "NYC"}));
    let tc = select_tool(&output, &[weather_tool()]).unwrap();
    assert!(tc.id.starts_with("call_"), "id={}", tc.id);
}

#[test]
fn select_tool_returns_valid_arguments_json() {
    let output = wrap_tool_call(
        "get_weather",
        &json!({"location": "Tokyo", "unit": "celsius"}),
    );
    let tc = select_tool(&output, &[weather_tool()]).unwrap();
    let args: serde_json::Value =
        serde_json::from_str(&tc.function.arguments).expect("arguments should be valid JSON");
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["unit"], "celsius");
}

// ── validate_tool_arguments ───────────────────────────────────────────────────

#[test]
fn validate_accepts_all_required_fields_present() {
    let result = validate_tool_arguments(r#"{"location":"Paris"}"#, &weather_tool());
    assert!(result.is_ok());
}

#[test]
fn validate_accepts_required_plus_optional() {
    let result =
        validate_tool_arguments(r#"{"location":"Madrid","unit":"celsius"}"#, &weather_tool());
    assert!(result.is_ok());
}

#[test]
fn validate_rejects_missing_required_field() {
    let err = validate_tool_arguments(r#"{"unit":"fahrenheit"}"#, &weather_tool())
        .expect_err("should fail: 'location' missing");
    assert!(
        matches!(err, ToolCallError::MalformedArguments { .. }),
        "{err}"
    );
}

#[test]
fn validate_rejects_invalid_json() {
    let err = validate_tool_arguments("not json", &weather_tool()).expect_err("should fail");
    assert!(matches!(err, ToolCallError::MalformedArguments { .. }));
}

#[test]
fn validate_rejects_json_array_not_object() {
    let err = validate_tool_arguments("[1,2,3]", &weather_tool()).expect_err("should fail");
    assert!(matches!(err, ToolCallError::MalformedArguments { .. }));
}

#[test]
fn validate_accepts_empty_object_for_no_required_fields() {
    let no_required =
        ToolDefinition::function("ping", None, json!({"type": "object", "properties": {}}));
    let result = validate_tool_arguments("{}", &no_required);
    assert!(result.is_ok());
}

#[test]
fn validate_returns_parsed_value_on_success() {
    let parsed = validate_tool_arguments(r#"{"query":"hello","max_results":5}"#, &search_tool())
        .expect("should succeed");
    assert!(parsed.is_object());
    assert_eq!(parsed["query"], "hello");
    assert_eq!(parsed["max_results"], 5);
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

#[test]
fn registry_get_returns_correct_definition() {
    let tools = vec![weather_tool(), search_tool(), calc_tool()];
    let reg = ToolRegistry::new(&tools);
    let def = reg.get("web_search").expect("should find web_search");
    assert_eq!(def.function.name, "web_search");
}

#[test]
fn registry_get_missing_returns_none() {
    let tools = vec![weather_tool()];
    let reg = ToolRegistry::new(&tools);
    assert!(reg.get("nonexistent").is_none());
}

#[test]
fn registry_len_matches_tool_count() {
    let tools = vec![weather_tool(), calc_tool()];
    let reg = ToolRegistry::new(&tools);
    assert_eq!(reg.len(), 2);
}

#[test]
fn registry_is_empty_for_empty_slice() {
    let tools: Vec<ToolDefinition> = vec![];
    let reg = ToolRegistry::new(&tools);
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
}

#[test]
fn registry_names_all_present() {
    let tools = vec![weather_tool(), search_tool(), calc_tool()];
    let reg = ToolRegistry::new(&tools);
    let mut names: Vec<&str> = reg.names().collect();
    names.sort_unstable();
    assert_eq!(names, &["calculate", "get_weather", "web_search"]);
}

// ── build_tool_constraint ─────────────────────────────────────────────────────

#[test]
fn build_constraint_empty_tools_returns_error() {
    let err = build_tool_constraint(&[]).expect_err("should fail for empty list");
    assert!(matches!(err, ToolCallError::EmptyToolList), "{err}");
}

#[test]
fn build_constraint_single_tool_produces_grammar_with_rules() {
    let g = build_tool_constraint(&[weather_tool()]).expect("should build");
    assert!(!g.rules.is_empty());
}

#[test]
fn build_constraint_multi_tool_root_has_one_rule_per_tool() {
    let tools = vec![weather_tool(), search_tool(), calc_tool()];
    let g = build_tool_constraint(&tools).expect("should build");
    let root_rules: Vec<_> = g.rules.iter().filter(|r| r.lhs == g.start).collect();
    assert_eq!(root_rules.len(), 3, "one root rule per tool");
}

#[test]
fn build_constraint_grammar_has_valid_start_nt() {
    let g = build_tool_constraint(&[weather_tool()]).expect("should build");
    // Start NT must be referenced by at least one rule.
    let start_in_rules = g.rules.iter().any(|r| r.lhs == g.start);
    assert!(start_in_rules, "start NT must have at least one rule");
}

// ── ToolCallError ─────────────────────────────────────────────────────────────

#[test]
fn tool_call_error_display_variants_non_empty() {
    let variants: Vec<Box<dyn std::error::Error>> = vec![
        Box::new(ToolCallError::NoToolCallFound),
        Box::new(ToolCallError::UnknownTool { name: "fn".into() }),
        Box::new(ToolCallError::MalformedArguments {
            reason: "bad".into(),
        }),
        Box::new(ToolCallError::GrammarCompileError {
            reason: "oops".into(),
        }),
        Box::new(ToolCallError::EmptyToolList),
    ];
    for e in &variants {
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn tool_call_error_implements_std_error() {
    let e: &dyn std::error::Error = &ToolCallError::NoToolCallFound;
    let _ = e.to_string();
}
