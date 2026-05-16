//! Integration tests for the JSON Schema → BNF Grammar compiler.
//!
//! Tests cover: all primitive types, composite types, `enum`, `anyOf`,
//! `oneOf`, `allOf`, `$ref` / `$defs`, depth limit, unsupported keywords,
//! error cases, and end-to-end round-trips via `GrammarConstraint`.

use std::sync::Arc;

use oxibonsai_runtime::constrained_decoding::TokenConstraint;
use oxibonsai_runtime::grammar::{
    compile_json_schema, compile_json_schema_str, EarleyRecognizer, GrammarConstraint,
    JsonSchemaCompileError,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a `GrammarConstraint` with a byte-level ASCII vocab (token id == byte value).
fn ascii_constraint_from_schema(schema_json: &str) -> GrammarConstraint {
    let grammar = compile_json_schema_str(schema_json).expect("schema must compile");
    GrammarConstraint::new(
        grammar,
        |id| if id < 128 { vec![id as u8] } else { vec![] },
        128,
    )
}

/// Feed an entire string into a `GrammarConstraint` advancing token-by-token.
/// Returns `true` iff all tokens were accepted.
fn feed_str_constraint(c: &mut GrammarConstraint, s: &str) -> bool {
    for b in s.bytes() {
        if !c.advance(b as u32) {
            return false;
        }
    }
    true
}

/// Build a normalised `EarleyRecognizer` from a compiled JSON schema.
fn recognizer_from_schema(schema_json: &str) -> EarleyRecognizer {
    let mut g = compile_json_schema_str(schema_json).expect("schema must compile");
    g.normalise_terminals();
    EarleyRecognizer::new(Arc::new(g))
}

/// Feed bytes into an `EarleyRecognizer`; returns false if any byte is rejected.
fn feed_str_recognizer(rec: &mut EarleyRecognizer, s: &str) -> bool {
    for b in s.bytes() {
        if !rec.feed_byte(b) {
            return false;
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: compile_empty_object_schema
//
// An empty schema `{}` has no "type" — we treat it as "any JSON value".
// This produces a grammar with at least one rule (it succeeds, not an error).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_empty_object_schema() {
    // `{}` has no "type"; compiler treats it as any-value (non-error).
    let result = compile_json_schema_str("{}");
    assert!(
        result.is_ok(),
        "empty schema should compile successfully, got: {:?}",
        result.err()
    );
    let g = result.unwrap();
    assert!(!g.rules.is_empty(), "compiled grammar must have rules");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: compile_string_type
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_string_type() {
    let g = compile_json_schema_str(r#"{"type":"string"}"#).expect("should compile");
    assert!(!g.rules.is_empty());
    // Start NT must have rules.
    let start_rules: Vec<_> = g.rules_for(g.start()).collect();
    assert!(
        !start_rules.is_empty(),
        "start NT must have at least one rule"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: compile_integer_type
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_integer_type() {
    let mut rec = recognizer_from_schema(r#"{"type":"integer"}"#);
    assert!(
        feed_str_recognizer(&mut rec, "42"),
        "integer 42 should be accepted"
    );
    assert!(rec.is_accepting(), "42 is a complete integer");

    // Negative integer.
    let mut rec2 = recognizer_from_schema(r#"{"type":"integer"}"#);
    assert!(feed_str_recognizer(&mut rec2, "-7"));
    assert!(rec2.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: compile_number_type
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_number_type() {
    let mut rec = recognizer_from_schema(r#"{"type":"number"}"#);
    assert!(
        feed_str_recognizer(&mut rec, "3"),
        "whole number should be accepted"
    );
    assert!(rec.is_accepting());

    // Float.
    let mut rec2 = recognizer_from_schema(r#"{"type":"number"}"#);
    assert!(feed_str_recognizer(&mut rec2, "3.14"));
    assert!(rec2.is_accepting(), "3.14 must be accepted as a number");

    // Negative float.
    let mut rec3 = recognizer_from_schema(r#"{"type":"number"}"#);
    assert!(feed_str_recognizer(&mut rec3, "-0.5"));
    assert!(rec3.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: compile_boolean_type
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_boolean_type() {
    let mut rec_true = recognizer_from_schema(r#"{"type":"boolean"}"#);
    assert!(feed_str_recognizer(&mut rec_true, "true"));
    assert!(rec_true.is_accepting());

    let mut rec_false = recognizer_from_schema(r#"{"type":"boolean"}"#);
    assert!(feed_str_recognizer(&mut rec_false, "false"));
    assert!(rec_false.is_accepting());

    // Invalid.
    let mut rec_bad = recognizer_from_schema(r#"{"type":"boolean"}"#);
    let ok = feed_str_recognizer(&mut rec_bad, "yes");
    assert!(!ok || !rec_bad.is_accepting(), "\"yes\" is not a boolean");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: compile_null_type
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_null_type() {
    let mut rec = recognizer_from_schema(r#"{"type":"null"}"#);
    assert!(feed_str_recognizer(&mut rec, "null"));
    assert!(rec.is_accepting());

    let mut rec_bad = recognizer_from_schema(r#"{"type":"null"}"#);
    let ok = feed_str_recognizer(&mut rec_bad, "undefined");
    assert!(!ok || !rec_bad.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: compile_object_with_required_props
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_object_with_required_props() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "string"}
        },
        "required": ["x", "y"]
    }"#;
    let g = compile_json_schema_str(schema).expect("should compile");

    // Grammar must reference NTs for integer and string.
    // At minimum: start NT + object NT + integer NT + string NT + digit NT etc.
    assert!(
        g.nt_count >= 3,
        "must have multiple NTs for object with properties"
    );

    // Check grammar has at least 2 rules (object body + value NTs).
    assert!(g.rules.len() >= 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: compile_array_of_integers
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_array_of_integers() {
    let schema = r#"{"type":"array","items":{"type":"integer"}}"#;
    let mut rec = recognizer_from_schema(schema);

    // Empty array.
    assert!(feed_str_recognizer(&mut rec, "[]"));
    assert!(rec.is_accepting());

    // Single-element array.
    let mut rec2 = recognizer_from_schema(schema);
    assert!(feed_str_recognizer(&mut rec2, "[42]"));
    assert!(rec2.is_accepting());

    // Multi-element array.
    let mut rec3 = recognizer_from_schema(schema);
    assert!(feed_str_recognizer(&mut rec3, "[1,2,3]"));
    assert!(rec3.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: compile_nested_object_array (3 levels deep)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_nested_object_array() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"}
                    },
                    "required": ["id"]
                }
            }
        },
        "required": ["items"]
    }"#;
    let g = compile_json_schema_str(schema).expect("nested schema should compile");
    // 3 levels → object, array, inner object, integer NTs all present.
    assert!(g.nt_count >= 4, "expected multiple NTs for 3-level nesting");
    assert!(!g.rules.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10: compile_enum_strings
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_enum_strings() {
    let schema = r#"{"enum":["foo","bar","baz"]}"#;
    let mut c = ascii_constraint_from_schema(schema);

    // '"' starts a string literal — should be the only allowed first byte.
    let mask = c.allowed_tokens(&[], 128).unwrap();
    assert!(
        mask[b'"' as usize],
        "quote should be allowed at start of enum string"
    );
    assert!(
        !mask[b'x' as usize],
        "'x' not a valid start for any enum value"
    );

    // Feed '"foo"' — must be accepted.
    assert!(feed_str_constraint(&mut c, r#""foo""#));
    assert!(c.is_complete());

    // '"qux"' should be rejected (not in enum).
    let mut c2 = ascii_constraint_from_schema(schema);
    let ok = feed_str_constraint(&mut c2, r#""qux""#);
    // Either rejected mid-stream or not accepting at end.
    assert!(!ok || !c2.is_complete());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11: compile_enum_integers
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_enum_integers() {
    let schema = r#"{"enum":[1,2,3]}"#;
    let mut rec = recognizer_from_schema(schema);

    assert!(feed_str_recognizer(&mut rec, "2"));
    assert!(rec.is_accepting());

    // 4 is not in the enum.
    let mut rec2 = recognizer_from_schema(schema);
    let ok = feed_str_recognizer(&mut rec2, "4");
    assert!(!ok || !rec2.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12: compile_any_of
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_any_of() {
    let schema = r#"{"anyOf":[{"type":"string"},{"type":"integer"}]}"#;
    let mut rec_str = recognizer_from_schema(schema);

    // String value.
    assert!(feed_str_recognizer(&mut rec_str, r#""hello""#));
    assert!(rec_str.is_accepting());

    // Integer value.
    let mut rec_int = recognizer_from_schema(schema);
    assert!(feed_str_recognizer(&mut rec_int, "99"));
    assert!(rec_int.is_accepting());

    // Boolean — not in anyOf.
    let mut rec_bad = recognizer_from_schema(schema);
    let ok = feed_str_recognizer(&mut rec_bad, "true");
    assert!(!ok || !rec_bad.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 13: compile_one_of (identical behaviour to anyOf)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_one_of() {
    let schema = r#"{"oneOf":[{"type":"boolean"},{"type":"null"}]}"#;
    let mut rec_true = recognizer_from_schema(schema);
    assert!(feed_str_recognizer(&mut rec_true, "true"));
    assert!(rec_true.is_accepting());

    let mut rec_null = recognizer_from_schema(schema);
    assert!(feed_str_recognizer(&mut rec_null, "null"));
    assert!(rec_null.is_accepting());

    // Integer — not in oneOf.
    let mut rec_int = recognizer_from_schema(schema);
    let ok = feed_str_recognizer(&mut rec_int, "42");
    assert!(!ok || !rec_int.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 14: compile_ref_to_defs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_ref_to_defs() {
    let schema = r##"{
        "$defs": {
            "MyString": {"type": "string"}
        },
        "$ref": "#/$defs/MyString"
    }"##;
    let g = compile_json_schema_str(schema).expect("$ref to $defs must compile");
    assert!(!g.rules.is_empty());
    // The compiled grammar should resolve the reference.
    let mut rec = EarleyRecognizer::new({
        let mut g2 = g;
        g2.normalise_terminals();
        Arc::new(g2)
    });
    assert!(feed_str_recognizer(&mut rec, r#""hello""#));
    assert!(rec.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 15: compile_recursive_ref
//
// A linked-list style schema:
// Node: { "value": integer, "next": { "$ref": "#/$defs/Node" } | null }
//
// We model this as:
//   $defs.Node: object with required ["value","next"]
//   where "next" is anyOf [$ref Node, null]
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_recursive_ref() {
    // Self-referential: Node has a required "value" integer field.
    // The compiler must survive this without infinite recursion.
    let schema = r##"{
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "next": {
                        "anyOf": [
                            {"$ref": "#/$defs/Node"},
                            {"type": "null"}
                        ]
                    }
                },
                "required": ["value", "next"]
            }
        },
        "$ref": "#/$defs/Node"
    }"##;
    // Must compile without panic or DepthExceeded.
    let result = compile_json_schema_str(schema);
    // The recursive $ref is handled by the pre-allocation in pass 1 — the
    // body references the already-allocated NT id for "Node", so there is
    // no infinite expansion during compilation.  The runtime Earley parser
    // handles the recursive grammar correctly.
    assert!(
        result.is_ok(),
        "recursive $ref schema must compile, got: {:?}",
        result.err()
    );
    let g = result.unwrap();
    assert!(!g.rules.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 16: unsupported_keyword_not
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unsupported_keyword_not() {
    let schema = r#"{"not": {"type": "string"}}"#;
    let err = compile_json_schema_str(schema).expect_err("'not' should not be supported");
    assert!(
        matches!(err, JsonSchemaCompileError::UnsupportedKeyword(ref kw) if kw == "not"),
        "expected UnsupportedKeyword(\"not\"), got: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 17: unsupported_keyword_if_then
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unsupported_keyword_if_then() {
    let schema = r#"{"if": {"type": "string"}, "then": {"type": "integer"}}"#;
    let err = compile_json_schema_str(schema).expect_err("'if'/'then' should not be supported");
    assert!(
        matches!(err, JsonSchemaCompileError::UnsupportedKeyword(_)),
        "expected UnsupportedKeyword, got: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 18: invalid_json_str
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn invalid_json_str() {
    let err = compile_json_schema_str("not json at all }{").expect_err("invalid JSON must fail");
    assert!(
        matches!(err, JsonSchemaCompileError::InvalidJson(_)),
        "expected InvalidJson, got: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 19: dangling_ref
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dangling_ref() {
    // $ref to a definition that doesn't exist in $defs.
    let schema = r##"{"$ref": "#/$defs/Missing"}"##;
    let err = compile_json_schema_str(schema).expect_err("dangling $ref must fail");
    assert!(
        matches!(err, JsonSchemaCompileError::DanglingRef(_)),
        "expected DanglingRef, got: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 20: depth_exceeded
//
// Build a schema nested 33 levels deep — one more than the limit of 32.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn depth_exceeded() {
    // Build a schema with 33 levels of anyOf nesting.
    let mut schema = serde_json::json!({"type": "integer"});
    for _ in 0..33 {
        schema = serde_json::json!({"anyOf": [schema]});
    }
    let schema_str = serde_json::to_string(&schema).unwrap();
    let err =
        compile_json_schema_str(&schema_str).expect_err("depth > 32 must return DepthExceeded");
    assert!(
        matches!(err, JsonSchemaCompileError::DepthExceeded { limit: 32 }),
        "expected DepthExceeded {{ limit: 32 }}, got: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 21: all_of_merges_objects
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn all_of_merges_objects() {
    let schema = r#"{
        "allOf": [
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"]
            },
            {
                "type": "object",
                "properties": {"y": {"type": "string"}},
                "required": ["y"]
            }
        ]
    }"#;
    let g = compile_json_schema_str(schema).expect("allOf of objects must compile");
    assert!(!g.rules.is_empty(), "merged object must produce rules");

    // The merged object schema should include rules for both x (integer) and y (string).
    // We verify by checking that the NT names include both integer and string NTs.
    let has_integer_nt = g.nt_names.values().any(|n| n.contains("integer"));
    let has_string_nt = g.nt_names.values().any(|n| n.contains("string"));
    assert!(has_integer_nt, "merged schema should contain integer NT");
    assert!(has_string_nt, "merged schema should contain string NT");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 22: end_to_end_grammar_constraint_string_type
//
// Compile string schema → GrammarConstraint → verify allowed_tokens:
//   - '"' (0x22) must be allowed as first byte.
//   - digit bytes ('0'..'9') must NOT be allowed as first byte.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn end_to_end_grammar_constraint_string_type() {
    let schema = r#"{"type":"string"}"#;
    let grammar = compile_json_schema_str(schema).expect("string schema must compile");

    // Byte-level vocab where token id == byte value (0..128).
    let c = GrammarConstraint::new(
        grammar,
        |id| if id < 128 { vec![id as u8] } else { vec![] },
        128,
    );

    let mask = c
        .allowed_tokens(&[], 128)
        .expect("allowed_tokens must return Some");

    // '"' (0x22 = 34) must be allowed — it starts any string value.
    assert!(
        mask[b'"' as usize],
        "quote character (0x22) must be allowed at start of string type"
    );

    // Digit tokens must NOT be allowed (digits don't start a JSON string).
    for d in b'0'..=b'9' {
        assert!(
            !mask[d as usize],
            "digit '{}' should NOT be allowed at start of string type",
            d as char
        );
    }

    // Boolean and null starters must also be rejected.
    assert!(!mask[b't' as usize], "'t' should not start a string");
    assert!(!mask[b'f' as usize], "'f' should not start a string");
    assert!(!mask[b'n' as usize], "'n' should not start a string");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 23: end_to_end_integer_constraint_advance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn end_to_end_integer_constraint_advance() {
    let schema = r#"{"type":"integer"}"#;
    let mut c = ascii_constraint_from_schema(schema);

    // Digits must be allowed at start.
    let mask = c.allowed_tokens(&[], 128).unwrap();
    for d in b'0'..=b'9' {
        assert!(
            mask[d as usize],
            "digit '{}' must be allowed at start",
            d as char
        );
    }
    // '-' is also valid at start (negative integer).
    assert!(
        mask[b'-' as usize],
        "'-' must be allowed at start of integer"
    );

    // Feed "123" and check we are accepting.
    assert!(feed_str_constraint(&mut c, "123"));
    assert!(c.is_complete(), "123 is a complete integer");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 24: enum_bool_null values
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_enum_bool_null() {
    let schema = r#"{"enum":[true,false,null]}"#;
    let mut rec_true = recognizer_from_schema(schema);
    assert!(feed_str_recognizer(&mut rec_true, "true"));
    assert!(rec_true.is_accepting());

    let mut rec_false = recognizer_from_schema(schema);
    assert!(feed_str_recognizer(&mut rec_false, "false"));
    assert!(rec_false.is_accepting());

    let mut rec_null = recognizer_from_schema(schema);
    assert!(feed_str_recognizer(&mut rec_null, "null"));
    assert!(rec_null.is_accepting());

    // "maybe" is not in the enum.
    let mut rec_bad = recognizer_from_schema(schema);
    let ok = feed_str_recognizer(&mut rec_bad, "maybe");
    assert!(!ok || !rec_bad.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 25: object_empty_required
//
// An object with no required properties emits "{}" grammar.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn object_empty_required() {
    let schema = r#"{"type":"object","properties":{"x":{"type":"integer"}}}"#;
    let mut rec = recognizer_from_schema(schema);
    // Only empty object is valid (no required properties → only `{}` rule).
    assert!(feed_str_recognizer(&mut rec, "{}"));
    assert!(rec.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 26: compile_definitions_alias
//
// Use `definitions` (older JSON Schema draft) instead of `$defs`.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_definitions_alias() {
    let schema = r##"{
        "definitions": {
            "Num": {"type": "integer"}
        },
        "$ref": "#/definitions/Num"
    }"##;
    let g = compile_json_schema_str(schema).expect("'definitions' alias must compile");
    assert!(!g.rules.is_empty());

    let mut rec = EarleyRecognizer::new({
        let mut g2 = g;
        g2.normalise_terminals();
        Arc::new(g2)
    });
    assert!(feed_str_recognizer(&mut rec, "7"));
    assert!(rec.is_accepting());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 27: allOf_rejects_non_object
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn all_of_rejects_non_object() {
    let schema = r#"{"allOf":[{"type":"string"},{"type":"integer"}]}"#;
    let err = compile_json_schema_str(schema).expect_err("allOf of non-objects must fail");
    assert!(
        matches!(err, JsonSchemaCompileError::UnsupportedKeyword(_)),
        "expected UnsupportedKeyword for allOf of non-objects, got: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 28: unsupported_keyword_pattern
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unsupported_keyword_pattern() {
    let schema = r#"{"type":"string","pattern":"^[a-z]+"}"#;
    let err = compile_json_schema_str(schema).expect_err("'pattern' should not be supported");
    assert!(
        matches!(err, JsonSchemaCompileError::UnsupportedKeyword(ref kw) if kw == "pattern"),
        "expected UnsupportedKeyword(\"pattern\"), got: {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 29: grammar_has_correct_start_nt
//
// Verify that the start NT returned by compile_json_schema is consistently
// the root of the compiled language.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn grammar_has_correct_start_nt() {
    let g = compile_json_schema_str(r#"{"type":"boolean"}"#).expect("must compile");
    let start = g.start();
    // Start NT must have at least one rule.
    let start_rules: Vec<_> = g.rules_for(start).collect();
    assert!(
        !start_rules.is_empty(),
        "start NT {start} must have at least one rule, got 0"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 30: compile_json_schema_value_api
//
// Test the `compile_json_schema(&Value)` API directly (not the str variant).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compile_json_schema_value_api() {
    let schema_val = serde_json::json!({"type": "null"});
    let g = compile_json_schema(&schema_val).expect("must compile from Value");
    assert!(!g.rules.is_empty());

    let mut rec = EarleyRecognizer::new({
        let mut g2 = g;
        g2.normalise_terminals();
        Arc::new(g2)
    });
    assert!(feed_str_recognizer(&mut rec, "null"));
    assert!(rec.is_accepting());
}
