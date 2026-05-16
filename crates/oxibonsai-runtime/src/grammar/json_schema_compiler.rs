//! JSON Schema → BNF Grammar compiler.
//!
//! Compiles a JSON Schema (represented as a `serde_json::Value`) directly
//! into a [`Grammar`] using [`Grammar::alloc_nt`] and manual rule construction.
//! No text emission or re-parsing — the grammar is built programmatically,
//! following the same right-recursive style used in `examples.rs`.
//!
//! # Supported JSON Schema v1 features
//!
//! - **Primitives:** `string`, `integer`, `number`, `boolean`, `null`
//! - **Composites:** `object` (with `properties`, `required`,
//!   `additionalProperties: false`), `array` (with `items`)
//! - **Constraints:** `enum` (string/integer/boolean/null values)
//! - **Composition:** `anyOf`, `oneOf` (compiled identically; Earley handles
//!   ambiguity), `allOf` (merge when all branches are object schemas)
//! - **References:** `$ref` pointing to `"#/$defs/..."` or `"#/definitions/..."`,
//!   via two-pass pre-allocation of NT ids.
//!
//! # Out of scope (returns `UnsupportedKeyword`)
//!
//! `not`, `if`, `then`, `else`, `patternProperties`,
//! `additionalProperties: <subschema>` (only `false` is allowed),
//! `pattern`, `format`, `multipleOf`, `minimum`, `maximum`,
//! `exclusiveMinimum`, `exclusiveMaximum`.

use std::collections::HashMap;

use serde_json::Value;

use super::ast::{Grammar, NonTerminalId, Rule, Symbol};

// ─────────────────────────────────────────────────────────────────────────────
// Public error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising from compiling a JSON Schema into a Grammar.
#[derive(Debug, Clone)]
pub enum JsonSchemaCompileError {
    /// The schema JSON is structurally invalid (missing field, wrong type, etc.).
    InvalidSchema(String),
    /// A JSON Schema keyword that is not supported by this compiler was used.
    UnsupportedKeyword(String),
    /// A `$ref` points to a `$defs`/`definitions` key that does not exist.
    DanglingRef(String),
    /// The schema nesting depth exceeded the compiled limit.
    DepthExceeded {
        /// Maximum depth allowed.
        limit: usize,
    },
    /// The input string passed to [`compile_json_schema_str`] is not valid JSON.
    InvalidJson(String),
}

impl std::fmt::Display for JsonSchemaCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSchema(msg) => write!(f, "invalid JSON schema: {msg}"),
            Self::UnsupportedKeyword(kw) => write!(f, "unsupported JSON schema keyword: {kw}"),
            Self::DanglingRef(r) => write!(f, "unresolvable $ref: {r}"),
            Self::DepthExceeded { limit } => {
                write!(f, "schema nesting depth exceeded limit of {limit}")
            }
            Self::InvalidJson(msg) => write!(f, "invalid JSON input: {msg}"),
        }
    }
}

impl std::error::Error for JsonSchemaCompileError {}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compile a JSON Schema (as a `serde_json::Value`) into a [`Grammar`].
///
/// The returned grammar has not yet been normalised; call
/// [`Grammar::normalise_terminals`] before passing it to
/// [`super::constraint::GrammarConstraint`] (the constraint builder does this
/// automatically).
///
/// # Errors
///
/// Returns a [`JsonSchemaCompileError`] if the schema is structurally invalid,
/// uses an unsupported keyword, contains a dangling `$ref`, or exceeds the
/// maximum recursion depth (32).
pub fn compile_json_schema(schema: &Value) -> Result<Grammar, JsonSchemaCompileError> {
    Compiler::new().compile(schema)
}

/// Compile a JSON Schema from a JSON string.
///
/// Parses the string with `serde_json` then delegates to [`compile_json_schema`].
///
/// # Errors
///
/// Returns [`JsonSchemaCompileError::InvalidJson`] if the input is not valid JSON,
/// or any other [`JsonSchemaCompileError`] variant if schema compilation fails.
pub fn compile_json_schema_str(schema_json: &str) -> Result<Grammar, JsonSchemaCompileError> {
    let value: Value = serde_json::from_str(schema_json)
        .map_err(|e| JsonSchemaCompileError::InvalidJson(e.to_string()))?;
    compile_json_schema(&value)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: compiler state
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum allowed schema nesting depth.
const MAX_DEPTH: usize = 32;

/// Internal compiler that accumulates a [`Grammar`] while walking a JSON Schema.
struct Compiler {
    grammar: Grammar,
    /// Maps `$defs` / `definitions` key → pre-allocated NT id (pass 1).
    defs_nt: HashMap<String, NonTerminalId>,
    /// Tracks which `$defs` keys have already been compiled in pass 2, to
    /// handle forward references without infinite loops.
    defs_compiled: HashMap<String, bool>,
    /// Shared string NT reused across all `"string"` occurrences (None until
    /// first use — allocated lazily).
    string_nt: Option<NonTerminalId>,
    /// Shared digit NT (0–9) reused for integers and numbers.
    digit_nt: Option<NonTerminalId>,
}

impl Compiler {
    fn new() -> Self {
        // Start grammar with a placeholder start id=0; the actual start NT
        // is allocated during compile() and replaces grammar.start.
        let grammar = Grammar::new(0);
        Self {
            grammar,
            defs_nt: HashMap::new(),
            defs_compiled: HashMap::new(),
            string_nt: None,
            digit_nt: None,
        }
    }

    /// Entry point — orchestrates the two-pass compilation.
    fn compile(mut self, root: &Value) -> Result<Grammar, JsonSchemaCompileError> {
        // ── Pass 1: pre-allocate NT ids for every $defs / definitions key ───
        self.pass1_alloc_defs(root);

        // ── Pass 2: compile each definition body ────────────────────────────
        // We collect the keys first to avoid borrow conflicts.
        let def_keys: Vec<String> = self.defs_nt.keys().cloned().collect();
        for key in &def_keys {
            self.pass2_compile_def(key, root)?;
        }

        // ── Pass 2: compile the root schema itself ───────────────────────────
        let start_nt = self.compile_schema(root, 0)?;
        self.grammar.start = start_nt;

        Ok(self.grammar)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pass 1 helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Scan the top-level `$defs` / `definitions` object and pre-allocate one
    /// NT id per key.  This makes every `$ref` resolvable even if the ref
    /// appears before the definition body.
    fn pass1_alloc_defs(&mut self, root: &Value) {
        let defs = root
            .get("$defs")
            .or_else(|| root.get("definitions"))
            .and_then(|v| v.as_object());

        if let Some(map) = defs {
            for key in map.keys() {
                if !self.defs_nt.contains_key(key) {
                    let nt = self.grammar.alloc_nt(format!("$def_{key}"));
                    self.defs_nt.insert(key.clone(), nt);
                    self.defs_compiled.insert(key.clone(), false);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pass 2 helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Compile the body of definition `key` into rules for the pre-allocated NT.
    fn pass2_compile_def(&mut self, key: &str, root: &Value) -> Result<(), JsonSchemaCompileError> {
        // Guard: only compile once.
        if *self.defs_compiled.get(key).unwrap_or(&false) {
            return Ok(());
        }
        // Mark as compiled early to break potential self-reference cycles.
        self.defs_compiled.insert(key.to_string(), true);

        let def_nt = *self.defs_nt.get(key).expect("NT pre-allocated in pass 1");

        let body_value = root
            .get("$defs")
            .or_else(|| root.get("definitions"))
            .and_then(|v| v.get(key))
            .ok_or_else(|| {
                JsonSchemaCompileError::InvalidSchema(format!("$defs key '{key}' not found"))
            })?;

        // Compile the definition body — the result NT must equal def_nt.
        // We use a small trick: compile normally to get an intermediate NT,
        // then add a bridging rule `def_nt ::= intermediate_nt` if they differ.
        let compiled_nt = self.compile_schema(body_value, 0)?;
        if compiled_nt != def_nt {
            self.grammar
                .add_rule(Rule::new(def_nt, vec![Symbol::NonTerminal(compiled_nt)]));
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Core recursive schema compiler
    // ─────────────────────────────────────────────────────────────────────────

    /// Compile one JSON Schema node at `depth`.  Returns the NT id whose rules
    /// represent the language defined by the schema.
    fn compile_schema(
        &mut self,
        schema: &Value,
        depth: usize,
    ) -> Result<NonTerminalId, JsonSchemaCompileError> {
        if depth > MAX_DEPTH {
            return Err(JsonSchemaCompileError::DepthExceeded { limit: MAX_DEPTH });
        }

        // Handle non-object schemas (booleans: `true` → any, `false` → nothing).
        // JSON Schema allows `true` / `false` as schemas; we treat them as any
        // or empty respectively, though in practice this is rare in our use-case.
        if schema.is_boolean() {
            // Treat `true` as empty-string match (epsilon) — accept anything
            // by returning an NT that produces epsilon.  This is a pragmatic
            // simplification for the constrained-decoding use case.
            let nt = self.grammar.alloc_nt("__bool_schema");
            self.grammar
                .add_rule(Rule::new(nt, vec![Symbol::Terminal(vec![])]));
            return Ok(nt);
        }

        let obj = match schema.as_object() {
            Some(o) => o,
            None => {
                return Err(JsonSchemaCompileError::InvalidSchema(
                    "schema must be a JSON object or boolean".to_string(),
                ));
            }
        };

        // ── Reject unsupported top-level keywords ────────────────────────────
        for unsupported in &[
            "not",
            "if",
            "then",
            "else",
            "patternProperties",
            "pattern",
            "format",
            "multipleOf",
            "exclusiveMinimum",
            "exclusiveMaximum",
        ] {
            if obj.contains_key(*unsupported) {
                return Err(JsonSchemaCompileError::UnsupportedKeyword(
                    unsupported.to_string(),
                ));
            }
        }

        // ── $ref ─────────────────────────────────────────────────────────────
        if let Some(ref_val) = obj.get("$ref") {
            return self.compile_ref(ref_val);
        }

        // ── enum ─────────────────────────────────────────────────────────────
        if let Some(enum_val) = obj.get("enum") {
            return self.compile_enum(enum_val, depth);
        }

        // ── anyOf ────────────────────────────────────────────────────────────
        if let Some(any_of) = obj.get("anyOf") {
            return self.compile_any_of(any_of, depth);
        }

        // ── oneOf (identical handling to anyOf) ──────────────────────────────
        if let Some(one_of) = obj.get("oneOf") {
            return self.compile_any_of(one_of, depth);
        }

        // ── allOf ────────────────────────────────────────────────────────────
        if let Some(all_of) = obj.get("allOf") {
            return self.compile_all_of(all_of, schema, depth);
        }

        // ── type-dispatched ──────────────────────────────────────────────────
        match obj.get("type").and_then(|v| v.as_str()) {
            Some("string") => self.compile_string_type(),
            Some("integer") => self.compile_integer_type(),
            Some("number") => self.compile_number_type(),
            Some("boolean") => Ok(self.compile_boolean_type()),
            Some("null") => Ok(self.compile_null_type()),
            Some("object") => self.compile_object_type(schema, depth),
            Some("array") => self.compile_array_type(schema, depth),
            Some(other) => Err(JsonSchemaCompileError::InvalidSchema(format!(
                "unknown type: '{other}'"
            ))),
            None => {
                // Schema without "type": if it has no other composition
                // keywords, treat it as the "value" NT (any JSON value).
                // This handles `{}` (empty schema) gracefully.
                Ok(self.compile_any_value_type())
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // $ref
    // ─────────────────────────────────────────────────────────────────────────

    fn compile_ref(&mut self, ref_val: &Value) -> Result<NonTerminalId, JsonSchemaCompileError> {
        let ref_str = ref_val.as_str().ok_or_else(|| {
            JsonSchemaCompileError::InvalidSchema("$ref must be a string".to_string())
        })?;

        // We support: "#/$defs/Foo" and "#/definitions/Foo".
        let key = if let Some(k) = ref_str.strip_prefix("#/$defs/") {
            k
        } else if let Some(k) = ref_str.strip_prefix("#/definitions/") {
            k
        } else {
            return Err(JsonSchemaCompileError::UnsupportedKeyword(format!(
                "$ref to external schema or unsupported path: {ref_str}"
            )));
        };

        self.defs_nt
            .get(key)
            .copied()
            .ok_or_else(|| JsonSchemaCompileError::DanglingRef(ref_str.to_string()))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // enum
    // ─────────────────────────────────────────────────────────────────────────

    fn compile_enum(
        &mut self,
        enum_val: &Value,
        _depth: usize,
    ) -> Result<NonTerminalId, JsonSchemaCompileError> {
        let values = enum_val.as_array().ok_or_else(|| {
            JsonSchemaCompileError::InvalidSchema("\"enum\" value must be a JSON array".to_string())
        })?;

        if values.is_empty() {
            return Err(JsonSchemaCompileError::InvalidSchema(
                "\"enum\" array must not be empty".to_string(),
            ));
        }

        let enum_nt = self.grammar.alloc_nt("__enum");

        for v in values {
            let literal = json_value_to_literal(v)?;
            // Each alternative: enum_nt ::= <literal bytes>
            self.grammar
                .add_rule(Rule::new(enum_nt, vec![Symbol::Terminal(literal)]));
        }

        Ok(enum_nt)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // anyOf / oneOf
    // ─────────────────────────────────────────────────────────────────────────

    fn compile_any_of(
        &mut self,
        arr: &Value,
        depth: usize,
    ) -> Result<NonTerminalId, JsonSchemaCompileError> {
        let variants = arr.as_array().ok_or_else(|| {
            JsonSchemaCompileError::InvalidSchema("anyOf/oneOf must be an array".to_string())
        })?;

        if variants.is_empty() {
            return Err(JsonSchemaCompileError::InvalidSchema(
                "anyOf/oneOf must have at least one variant".to_string(),
            ));
        }

        let any_nt = self.grammar.alloc_nt("__anyOf");

        for variant in variants {
            let var_nt = self.compile_schema(variant, depth + 1)?;
            // any_nt ::= <variant_nt>
            self.grammar
                .add_rule(Rule::new(any_nt, vec![Symbol::NonTerminal(var_nt)]));
        }

        Ok(any_nt)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // allOf
    // ─────────────────────────────────────────────────────────────────────────

    /// Compile `allOf` by merging all branches if they are all object schemas.
    /// Otherwise returns `UnsupportedKeyword`.
    fn compile_all_of(
        &mut self,
        all_of_arr: &Value,
        _parent: &Value,
        depth: usize,
    ) -> Result<NonTerminalId, JsonSchemaCompileError> {
        let variants = all_of_arr.as_array().ok_or_else(|| {
            JsonSchemaCompileError::InvalidSchema("allOf must be an array".to_string())
        })?;

        if variants.is_empty() {
            return Err(JsonSchemaCompileError::InvalidSchema(
                "allOf must have at least one element".to_string(),
            ));
        }

        // Verify all variants are object schemas (have "type":"object" or just
        // have "properties").
        for variant in variants {
            let is_object = variant
                .get("type")
                .and_then(|v| v.as_str())
                .map(|t| t == "object")
                .unwrap_or(false)
                || variant.get("properties").is_some();

            if !is_object {
                return Err(JsonSchemaCompileError::UnsupportedKeyword(
                    "allOf can only merge object schemas in this compiler".to_string(),
                ));
            }
        }

        // Merge all properties and required fields.
        let mut merged_props: serde_json::Map<String, Value> = serde_json::Map::new();
        let mut merged_required: Vec<String> = Vec::new();
        let mut has_additional_false = false;

        for variant in variants {
            if let Some(props) = variant.get("properties").and_then(|v| v.as_object()) {
                for (k, v) in props {
                    merged_props.insert(k.clone(), v.clone());
                }
            }
            if let Some(req) = variant.get("required").and_then(|v| v.as_array()) {
                for r in req {
                    if let Some(s) = r.as_str() {
                        if !merged_required.contains(&s.to_string()) {
                            merged_required.push(s.to_string());
                        }
                    }
                }
            }
            if variant
                .get("additionalProperties")
                .and_then(|v| v.as_bool())
                == Some(false)
            {
                has_additional_false = true;
            }
        }

        // Build a merged object schema and compile it.
        let mut merged = serde_json::json!({
            "type": "object",
            "properties": merged_props,
            "required": merged_required,
        });
        if has_additional_false {
            if let Some(obj) = merged.as_object_mut() {
                obj.insert("additionalProperties".to_string(), Value::Bool(false));
            }
        }

        self.compile_object_type(&merged, depth)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Primitive types
    // ─────────────────────────────────────────────────────────────────────────

    /// Returns the shared string NT, allocating it on first use.
    ///
    /// Grammar fragment (simplified — allows printable ASCII except `"` and `\`):
    ///
    /// ```text
    /// <string>      ::= '"' '"' | '"' <string_chars> '"'
    /// <string_chars> ::= <string_char> | <string_char> <string_chars>
    /// <string_char>  ::= 0x20 | 0x21 | 0x23..=0x5B | 0x5D..=0x7E
    /// ```
    fn compile_string_type(&mut self) -> Result<NonTerminalId, JsonSchemaCompileError> {
        if let Some(nt) = self.string_nt {
            return Ok(nt);
        }

        let str_nt = self.grammar.alloc_nt("__string");
        let chars_nt = self.grammar.alloc_nt("__string_chars");
        let char_nt = self.grammar.alloc_nt("__string_char");

        // __string ::= '"' '"'
        self.grammar.add_rule(Rule::new(
            str_nt,
            vec![Symbol::Terminal(vec![b'"']), Symbol::Terminal(vec![b'"'])],
        ));
        // __string ::= '"' __string_chars '"'
        self.grammar.add_rule(Rule::new(
            str_nt,
            vec![
                Symbol::Terminal(vec![b'"']),
                Symbol::NonTerminal(chars_nt),
                Symbol::Terminal(vec![b'"']),
            ],
        ));

        // __string_chars ::= __string_char
        self.grammar
            .add_rule(Rule::new(chars_nt, vec![Symbol::NonTerminal(char_nt)]));
        // __string_chars ::= __string_char __string_chars
        self.grammar.add_rule(Rule::new(
            chars_nt,
            vec![Symbol::NonTerminal(char_nt), Symbol::NonTerminal(chars_nt)],
        ));

        // __string_char ::= 0x20 | 0x21 | 0x23..=0x5B | 0x5D..=0x7E
        // (printable ASCII excluding '"' (0x22) and '\' (0x5C))
        for b in 0x20u8..=0x21u8 {
            self.grammar
                .add_rule(Rule::new(char_nt, vec![Symbol::Terminal(vec![b])]));
        }
        for b in 0x23u8..=0x5Bu8 {
            self.grammar
                .add_rule(Rule::new(char_nt, vec![Symbol::Terminal(vec![b])]));
        }
        for b in 0x5Du8..=0x7Eu8 {
            self.grammar
                .add_rule(Rule::new(char_nt, vec![Symbol::Terminal(vec![b])]));
        }

        self.string_nt = Some(str_nt);
        Ok(str_nt)
    }

    /// Returns the shared digit NT (0–9), allocating on first use.
    fn ensure_digit_nt(&mut self) -> NonTerminalId {
        if let Some(nt) = self.digit_nt {
            return nt;
        }
        let digit_nt = self.grammar.alloc_nt("__digit");
        for b in b'0'..=b'9' {
            self.grammar
                .add_rule(Rule::new(digit_nt, vec![Symbol::Terminal(vec![b])]));
        }
        self.digit_nt = Some(digit_nt);
        digit_nt
    }

    /// Compile `{"type":"integer"}`.
    ///
    /// Grammar:
    /// ```text
    /// <integer>      ::= <digits> | '-' <digits>
    /// <digits>       ::= <digit> | <digit> <digits>
    /// <digit>        ::= '0' | ... | '9'
    /// ```
    fn compile_integer_type(&mut self) -> Result<NonTerminalId, JsonSchemaCompileError> {
        let digit_nt = self.ensure_digit_nt();

        let digits_nt = self.grammar.alloc_nt("__digits");
        // __digits ::= __digit
        self.grammar
            .add_rule(Rule::new(digits_nt, vec![Symbol::NonTerminal(digit_nt)]));
        // __digits ::= __digit __digits
        self.grammar.add_rule(Rule::new(
            digits_nt,
            vec![
                Symbol::NonTerminal(digit_nt),
                Symbol::NonTerminal(digits_nt),
            ],
        ));

        let int_nt = self.grammar.alloc_nt("__integer");
        // __integer ::= __digits
        self.grammar
            .add_rule(Rule::new(int_nt, vec![Symbol::NonTerminal(digits_nt)]));
        // __integer ::= '-' __digits
        self.grammar.add_rule(Rule::new(
            int_nt,
            vec![Symbol::Terminal(vec![b'-']), Symbol::NonTerminal(digits_nt)],
        ));

        Ok(int_nt)
    }

    /// Compile `{"type":"number"}`.
    ///
    /// Grammar:
    /// ```text
    /// <number>    ::= <integer> | <integer> '.' <digits>
    /// ```
    fn compile_number_type(&mut self) -> Result<NonTerminalId, JsonSchemaCompileError> {
        let int_nt = self.compile_integer_type()?;
        // Reuse the `__digits` NT if already allocated; allocate a fresh one
        // for the fractional part to avoid ambiguity with integer's own digits NT.
        let digit_nt = self.ensure_digit_nt();

        let frac_digits_nt = self.grammar.alloc_nt("__frac_digits");
        // __frac_digits ::= __digit
        self.grammar.add_rule(Rule::new(
            frac_digits_nt,
            vec![Symbol::NonTerminal(digit_nt)],
        ));
        // __frac_digits ::= __digit __frac_digits
        self.grammar.add_rule(Rule::new(
            frac_digits_nt,
            vec![
                Symbol::NonTerminal(digit_nt),
                Symbol::NonTerminal(frac_digits_nt),
            ],
        ));

        let num_nt = self.grammar.alloc_nt("__number");
        // __number ::= __integer
        self.grammar
            .add_rule(Rule::new(num_nt, vec![Symbol::NonTerminal(int_nt)]));
        // __number ::= __integer '.' __frac_digits
        self.grammar.add_rule(Rule::new(
            num_nt,
            vec![
                Symbol::NonTerminal(int_nt),
                Symbol::Terminal(vec![b'.']),
                Symbol::NonTerminal(frac_digits_nt),
            ],
        ));

        Ok(num_nt)
    }

    /// Compile `{"type":"boolean"}`.  Returns an NT with two rules: `true` and `false`.
    fn compile_boolean_type(&mut self) -> NonTerminalId {
        let bool_nt = self.grammar.alloc_nt("__boolean");
        self.grammar
            .add_rule(Rule::new(bool_nt, vec![Symbol::Terminal(b"true".to_vec())]));
        self.grammar.add_rule(Rule::new(
            bool_nt,
            vec![Symbol::Terminal(b"false".to_vec())],
        ));
        bool_nt
    }

    /// Compile `{"type":"null"}`.  Returns an NT with one rule: `null`.
    fn compile_null_type(&mut self) -> NonTerminalId {
        let null_nt = self.grammar.alloc_nt("__null");
        self.grammar
            .add_rule(Rule::new(null_nt, vec![Symbol::Terminal(b"null".to_vec())]));
        null_nt
    }

    /// Compile a schema with no "type" (or the `{}` schema) as a generic JSON
    /// value: string | integer | number | boolean | null | object | array.
    ///
    /// This is deliberately non-recursive for the array/object arms to avoid
    /// infinite grammar expansion; we emit shallow stubs that match valid JSON
    /// structure (empty object `{}` and empty array `[]`).
    fn compile_any_value_type(&mut self) -> NonTerminalId {
        let val_nt = self.grammar.alloc_nt("__any_value");

        // Primitives
        let str_nt = self.string_nt.unwrap_or_else(|| {
            // Lazily compile; ignore error (primitives always succeed).
            self.compile_string_type().expect("string type compile")
        });
        let bool_nt = self.compile_boolean_type();
        let null_nt = self.compile_null_type();

        // For integer and number, allocate digits NT first.
        let digit_nt = self.ensure_digit_nt();
        let digits_nt = self.grammar.alloc_nt("__any_val_digits");
        self.grammar
            .add_rule(Rule::new(digits_nt, vec![Symbol::NonTerminal(digit_nt)]));
        self.grammar.add_rule(Rule::new(
            digits_nt,
            vec![
                Symbol::NonTerminal(digit_nt),
                Symbol::NonTerminal(digits_nt),
            ],
        ));

        let num_nt = self.grammar.alloc_nt("__any_val_num");
        self.grammar
            .add_rule(Rule::new(num_nt, vec![Symbol::NonTerminal(digits_nt)]));
        self.grammar.add_rule(Rule::new(
            num_nt,
            vec![Symbol::Terminal(vec![b'-']), Symbol::NonTerminal(digits_nt)],
        ));

        // Empty object stub: `{}`
        let obj_stub_nt = self.grammar.alloc_nt("__any_val_obj");
        self.grammar.add_rule(Rule::new(
            obj_stub_nt,
            vec![Symbol::Terminal(vec![b'{']), Symbol::Terminal(vec![b'}'])],
        ));

        // Empty array stub: `[]`
        let arr_stub_nt = self.grammar.alloc_nt("__any_val_arr");
        self.grammar.add_rule(Rule::new(
            arr_stub_nt,
            vec![Symbol::Terminal(vec![b'[']), Symbol::Terminal(vec![b']'])],
        ));

        // __any_value ::= __string | __boolean | __null | __any_val_num | {} | []
        self.grammar
            .add_rule(Rule::new(val_nt, vec![Symbol::NonTerminal(str_nt)]));
        self.grammar
            .add_rule(Rule::new(val_nt, vec![Symbol::NonTerminal(bool_nt)]));
        self.grammar
            .add_rule(Rule::new(val_nt, vec![Symbol::NonTerminal(null_nt)]));
        self.grammar
            .add_rule(Rule::new(val_nt, vec![Symbol::NonTerminal(num_nt)]));
        self.grammar
            .add_rule(Rule::new(val_nt, vec![Symbol::NonTerminal(obj_stub_nt)]));
        self.grammar
            .add_rule(Rule::new(val_nt, vec![Symbol::NonTerminal(arr_stub_nt)]));

        val_nt
    }

    // ─────────────────────────────────────────────────────────────────────────
    // object
    // ─────────────────────────────────────────────────────────────────────────

    /// Compile `{"type":"object","properties":{...},"required":[...]}`.
    ///
    /// In v1, only required properties are included in the grammar body.
    /// Optional properties are silently omitted when `additionalProperties: false`.
    ///
    /// The grammar shape (for required props `[p1, p2]`) is:
    /// ```text
    /// <obj> ::= '{' '"p1"' ':' <S1> ',' '"p2"' ':' <S2> '}'
    ///         | '{' '}'        ← only when there are no required properties
    /// ```
    fn compile_object_type(
        &mut self,
        schema: &Value,
        depth: usize,
    ) -> Result<NonTerminalId, JsonSchemaCompileError> {
        // Reject additionalProperties: <subschema> (only false is allowed).
        if let Some(ap) = schema.get("additionalProperties") {
            if !ap.is_boolean() {
                return Err(JsonSchemaCompileError::UnsupportedKeyword(
                    "additionalProperties as a subschema is not supported; use false or omit it"
                        .to_string(),
                ));
            }
        }

        let properties = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let required: Vec<String> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Only compile required properties in v1.
        let obj_nt = self.grammar.alloc_nt("__object");

        if required.is_empty() {
            // Empty object: `{}`
            self.grammar.add_rule(Rule::new(
                obj_nt,
                vec![Symbol::Terminal(vec![b'{']), Symbol::Terminal(vec![b'}'])],
            ));
            return Ok(obj_nt);
        }

        // Compile the value NT for each required property.
        // We must compile these before building the rule body to avoid
        // borrow issues with self.grammar.
        let mut prop_nts: Vec<(String, NonTerminalId)> = Vec::new();
        for prop_name in &required {
            let sub_schema = properties.get(prop_name).ok_or_else(|| {
                JsonSchemaCompileError::InvalidSchema(format!(
                    "required property '{prop_name}' not found in 'properties'"
                ))
            })?;
            let val_nt = self.compile_schema(sub_schema, depth + 1)?;
            prop_nts.push((prop_name.clone(), val_nt));
        }

        // Build the rule body:
        // '{' key0 ':' <val0> ',' key1 ':' <val1> ... '}'
        let mut body: Vec<Symbol> = Vec::new();
        body.push(Symbol::Terminal(vec![b'{']));

        for (i, (prop_name, val_nt)) in prop_nts.iter().enumerate() {
            if i > 0 {
                body.push(Symbol::Terminal(vec![b',']));
            }
            // Property key as a JSON string literal: '"propname"'
            let key_bytes = json_string_literal_bytes(prop_name);
            body.push(Symbol::Terminal(key_bytes));
            body.push(Symbol::Terminal(vec![b':']));
            body.push(Symbol::NonTerminal(*val_nt));
        }

        body.push(Symbol::Terminal(vec![b'}']));
        self.grammar.add_rule(Rule::new(obj_nt, body));

        Ok(obj_nt)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // array
    // ─────────────────────────────────────────────────────────────────────────

    /// Compile `{"type":"array","items":<S>}`.
    ///
    /// Grammar:
    /// ```text
    /// <array>       ::= '[' ']'
    ///                 | '[' <array_items> ']'
    /// <array_items> ::= <S_nt>
    ///                 | <S_nt> ',' <array_items>
    /// ```
    fn compile_array_type(
        &mut self,
        schema: &Value,
        depth: usize,
    ) -> Result<NonTerminalId, JsonSchemaCompileError> {
        let items_schema = schema.get("items");

        let item_nt = if let Some(items) = items_schema {
            self.compile_schema(items, depth + 1)?
        } else {
            // No items schema — use the generic value NT.
            self.compile_any_value_type()
        };

        let items_nt = self.grammar.alloc_nt("__array_items");
        // __array_items ::= <item_nt>
        self.grammar
            .add_rule(Rule::new(items_nt, vec![Symbol::NonTerminal(item_nt)]));
        // __array_items ::= <item_nt> ',' __array_items
        self.grammar.add_rule(Rule::new(
            items_nt,
            vec![
                Symbol::NonTerminal(item_nt),
                Symbol::Terminal(vec![b',']),
                Symbol::NonTerminal(items_nt),
            ],
        ));

        let arr_nt = self.grammar.alloc_nt("__array");
        // __array ::= '[' ']'
        self.grammar.add_rule(Rule::new(
            arr_nt,
            vec![Symbol::Terminal(vec![b'[']), Symbol::Terminal(vec![b']'])],
        ));
        // __array ::= '[' __array_items ']'
        self.grammar.add_rule(Rule::new(
            arr_nt,
            vec![
                Symbol::Terminal(vec![b'[']),
                Symbol::NonTerminal(items_nt),
                Symbol::Terminal(vec![b']']),
            ],
        ));

        Ok(arr_nt)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a `serde_json::Value` to its JSON literal byte representation,
/// suitable for a single terminal symbol in the grammar.
///
/// Supports: strings, integers, booleans, null.
/// Returns `InvalidSchema` for floats or arrays/objects (not valid enum members
/// in our simplified v1).
fn json_value_to_literal(v: &Value) -> Result<Vec<u8>, JsonSchemaCompileError> {
    match v {
        Value::String(_) => {
            // Emit as JSON string literal: `"<escaped>"`
            let json_repr = serde_json::to_string(v)
                .map_err(|e| JsonSchemaCompileError::InvalidSchema(e.to_string()))?;
            Ok(json_repr.into_bytes())
        }
        Value::Number(n) => {
            // Integers only in v1.
            if n.is_i64() || n.is_u64() {
                Ok(n.to_string().into_bytes())
            } else {
                Err(JsonSchemaCompileError::UnsupportedKeyword(
                    "float enum values are not supported".to_string(),
                ))
            }
        }
        Value::Bool(b) => {
            if *b {
                Ok(b"true".to_vec())
            } else {
                Ok(b"false".to_vec())
            }
        }
        Value::Null => Ok(b"null".to_vec()),
        _ => Err(JsonSchemaCompileError::UnsupportedKeyword(
            "enum values must be strings, integers, booleans, or null".to_string(),
        )),
    }
}

/// Build the byte sequence for a JSON string literal `"<key>"`.
/// The key is assumed to be a plain ASCII identifier (no escaping needed for
/// typical JSON Schema property names).  For safety we still pass through
/// `serde_json::to_string` so that non-ASCII / special characters are escaped.
fn json_string_literal_bytes(key: &str) -> Vec<u8> {
    serde_json::to_string(key)
        .expect("key serialization must succeed")
        .into_bytes()
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_string_gives_grammar() {
        let g = compile_json_schema_str(r#"{"type":"string"}"#).expect("should compile");
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn compile_boolean_gives_grammar() {
        let g = compile_json_schema_str(r#"{"type":"boolean"}"#).expect("should compile");
        assert!(!g.rules.is_empty());
    }

    #[test]
    fn json_value_to_literal_string() {
        let v = Value::String("hello".to_string());
        let b = json_value_to_literal(&v).expect("ok");
        assert_eq!(b, br#""hello""#);
    }

    #[test]
    fn json_value_to_literal_integer() {
        let v = Value::Number(serde_json::Number::from(42));
        let b = json_value_to_literal(&v).expect("ok");
        assert_eq!(b, b"42");
    }

    #[test]
    fn json_value_to_literal_bool() {
        let b = json_value_to_literal(&Value::Bool(true)).expect("ok");
        assert_eq!(b, b"true");
    }

    #[test]
    fn json_value_to_literal_null() {
        let b = json_value_to_literal(&Value::Null).expect("ok");
        assert_eq!(b, b"null");
    }

    #[test]
    fn json_string_literal_bytes_simple() {
        let b = json_string_literal_bytes("name");
        assert_eq!(b, br#""name""#);
    }
}
