//! Extended OpenAI-compatible API types.
//!
//! Provides request/response types for full OpenAI API compatibility including
//! function calling (tools), logprobs, JSON mode, and multi-completion support.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ── Phase 19: Tool calling types ──────────────────────────────────────────────

/// A function definition for tool use.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolFunction {
    /// The name of the function.
    pub name: String,
    /// An optional description of the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema object describing the function parameters.
    pub parameters: serde_json::Value,
}

/// A tool available to the model (OpenAI-compatible format).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    /// Must be `"function"`.
    #[serde(rename = "type")]
    pub r#type: String,
    /// The function definition.
    pub function: ToolFunction,
}

impl ToolDefinition {
    /// Convenience constructor for a function-type tool.
    pub fn function(
        name: impl Into<String>,
        description: Option<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: name.into(),
                description,
                parameters,
            },
        }
    }
}

/// A function call made by the model (name + serialised arguments).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolFunctionCall {
    /// Name of the function invoked.
    pub name: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
}

/// A tool call produced by the model in a chat completion response.
///
/// Uses `r#type` (serialised as `"type"`) to avoid the reserved keyword.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallResult {
    /// Unique identifier for this tool call (prefix `call_`).
    pub id: String,
    /// Type of tool call — always `"function"`.
    #[serde(rename = "type")]
    pub r#type: String,
    /// The function invoked.
    pub function: ToolFunctionCall,
}

impl ToolCallResult {
    /// Construct a `ToolCallResult` for a function call.
    pub fn new_function(id: String, name: String, arguments: String) -> Self {
        Self {
            id,
            r#type: "function".to_string(),
            function: ToolFunctionCall { name, arguments },
        }
    }
}

// ── Function calling ──────────────────────────────────────────────────────────

/// A function that can be called by the model.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct FunctionDefinition {
    /// The name of the function.
    pub name: String,
    /// A description of what the function does.
    pub description: Option<String>,
    /// The parameters the function accepts (JSON Schema object).
    pub parameters: Option<serde_json::Value>,
}

/// A tool that can be used during generation.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Tool {
    /// The type of tool. Currently only `"function"` is supported.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition.
    pub function: FunctionDefinition,
}

/// Controls which tool (if any) is called by the model.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// A string value: `"none"`, `"auto"`, or `"required"`.
    String(String),
    /// A specific named tool to call.
    Named(NamedToolChoice),
}

/// A specific tool choice identifying a function by name.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct NamedToolChoice {
    /// The type of the tool (e.g. `"function"`).
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function to call.
    pub function: FunctionName,
}

/// A function identified by name only.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct FunctionName {
    /// The name of the function.
    pub name: String,
}

/// A tool call made by the model in the response.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolCall {
    /// A unique ID for this tool call.
    pub id: String,
    /// The type of tool call (always `"function"`).
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function that was called.
    pub function: FunctionCallResult,
}

/// The result of a function call — name and serialized arguments.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FunctionCallResult {
    /// The name of the function called.
    pub name: String,
    /// The arguments to the function as a JSON string.
    pub arguments: String,
}

// ── Logprobs ─────────────────────────────────────────────────────────────────

/// Log probability information for a single generated token.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LogprobsContent {
    /// The token text.
    pub token: String,
    /// The log probability of this token.
    pub logprob: f32,
    /// The UTF-8 bytes of the token, if representable.
    pub bytes: Option<Vec<u8>>,
    /// The top alternative tokens at this position.
    pub top_logprobs: Vec<TopLogprob>,
}

/// A top-k alternative token and its log probability.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TopLogprob {
    /// The token text.
    pub token: String,
    /// The log probability of this token.
    pub logprob: f32,
    /// The UTF-8 bytes of the token, if representable.
    pub bytes: Option<Vec<u8>>,
}

/// Logprob information attached to a choice.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChoiceLogprobs {
    /// Per-token log probability content for the choice.
    pub content: Option<Vec<LogprobsContent>>,
}

// ── Response format ───────────────────────────────────────────────────────────

/// The format in which the model should return its response.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ResponseFormat {
    /// `"text"`, `"json_object"`, or `"json_schema"`.
    #[serde(rename = "type")]
    pub format_type: String,
    /// JSON schema definition (only used when `format_type == "json_schema"`).
    pub json_schema: Option<JsonSchemaFormat>,
}

/// A named JSON schema that the model output must conform to.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct JsonSchemaFormat {
    /// A human-readable name for the schema.
    pub name: String,
    /// The JSON Schema object.
    pub schema: serde_json::Value,
    /// Whether the model must strictly follow the schema.
    pub strict: Option<bool>,
}

// ── Stop sequences ────────────────────────────────────────────────────────────

/// One or more stop sequences that terminate generation.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum StopSequences {
    /// A single stop sequence string.
    Single(String),
    /// Multiple stop sequence strings.
    Multiple(Vec<String>),
}

impl StopSequences {
    /// Return a slice of stop sequence strings.
    pub fn as_slice(&self) -> &[String] {
        match self {
            StopSequences::Single(s) => std::slice::from_ref(s),
            StopSequences::Multiple(v) => v.as_slice(),
        }
    }

    /// Consume and return all stop sequences as a `Vec<String>`.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequences::Single(s) => vec![s],
            StopSequences::Multiple(v) => v,
        }
    }
}

// ── Usage info (public alias used by ExtendedChatResponse) ───────────────────

/// Token usage information for a completion request.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UsageInfo {
    /// Tokens consumed by the prompt.
    pub prompt_tokens: usize,
    /// Tokens generated in the completion.
    pub completion_tokens: usize,
    /// Total tokens (prompt + completion).
    pub total_tokens: usize,
}

// ── Extended chat completion request ─────────────────────────────────────────

/// A full OpenAI-compatible chat completion request including all optional fields.
#[derive(Debug, serde::Deserialize)]
pub struct ExtendedChatRequest {
    /// The conversation messages.
    pub messages: Vec<crate::server::ChatMessage>,
    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy).
    pub temperature: Option<f32>,
    /// Nucleus sampling probability threshold.
    pub top_p: Option<f32>,
    /// Whether to stream the response as SSE.
    pub stream: Option<bool>,
    /// Sequences that stop generation.
    pub stop: Option<StopSequences>,
    /// Tools available to the model.
    pub tools: Option<Vec<Tool>>,
    /// Controls which tool is called, if any.
    pub tool_choice: Option<ToolChoice>,
    /// Whether to return log probabilities for generated tokens.
    pub logprobs: Option<bool>,
    /// Number of top alternative tokens to include in logprobs (0–20).
    pub top_logprobs: Option<usize>,
    /// Format constraint for the response.
    pub response_format: Option<ResponseFormat>,
    /// Seed for deterministic generation.
    pub seed: Option<u64>,
    /// Number of independent completions to generate (default 1, max 4).
    pub n: Option<usize>,
    /// Penalty applied for tokens that are present in the context.
    pub presence_penalty: Option<f32>,
    /// Penalty applied proportional to a token's frequency in the context.
    pub frequency_penalty: Option<f32>,
    /// Repetition penalty (`1.0` = disabled). When omitted, the handler seeds
    /// from the engine's own startup `SamplingParams` rather than a
    /// hardcoded literal or `SamplingParams::default()` (gatekeeper
    /// `REQUIRED #1`): the previous hardcoded `1.1` permanently disqualified
    /// every extended-endpoint request from the GPU-argmax greedy path
    /// (`InferenceEngine::greedy_gpu_eligible` requires exactly `1.0`), even
    /// a `temperature: 0` request against a server started with no
    /// repetition penalty configured at all.
    pub repetition_penalty: Option<f32>,
    /// An optional identifier for the end user.
    pub user: Option<String>,
}

fn default_max_tokens() -> usize {
    256
}

// ── Multimodal content parts (SV-11 — prepare only) ──────────────────────────
//
// `ChatMessage.content` (`crate::server::ChatMessage`) is `Option<String>`,
// so a request whose message content is an array of content parts (the
// OpenAI vision shape, `content: string | ContentPart[]`) is rejected at the
// type level with a bare deserialization error before any handler code runs
// — the type-level block Bonsai 2 vision needs removed (mmproj / Qwen3-VL
// merger, `<|image_pad|>` token expansion; see `CONTEXT.md`'s Bonsai 2
// section). `ChatMessage` itself lives in `server.rs`, which this package
// does not own, so this cannot be wired into it directly this wave (see
// `deviations`); what *is* fully implemented here, end to end, is the
// value-level machinery B2-20 (wave 6) will plug straight into
// `ChatMessage.content: Option<MessageContent>` — deserialization, the
// text-only extraction used by every prompt builder today, and an honest
// rejection of `image_url` parts (never a silent drop, never a stub image
// path: "do not accept `image_url` yet, and say so in the 400 message").

/// One part of a multipart chat message `content` array (OpenAI vision
/// shape: `{"type": "text", "text": "..."}` /
/// `{"type": "image_url", "image_url": {"url": "...", ...}}`).
///
/// `ImageUrl` is parsed structurally — never silently dropped or merged into
/// an "unknown variant" deserialization error — precisely so that
/// [`MessageContent::into_text_only`] can recognize it and produce a clear,
/// specific rejection message instead of an opaque schema error. Parsing an
/// `image_url` part is not the same as supporting it: nothing here decodes,
/// fetches, or otherwise acts on the URL (`B2-20`'s job).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A plain text segment.
    Text {
        /// The text content.
        text: String,
    },
    /// An image reference. Structurally accepted, semantically rejected —
    /// see [`MessageContent::into_text_only`].
    ImageUrl {
        /// The image reference payload.
        image_url: ImageUrlPart,
    },
}

/// The `image_url` object of an [`ContentPart::ImageUrl`] part.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ImageUrlPart {
    /// The image URL (`http(s)://...` or a `data:` URI).
    pub url: String,
    /// OpenAI's optional resolution hint (`"auto"` / `"low"` / `"high"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A chat message's `content`: either a plain string (the common case, and
/// the only shape `crate::server::ChatMessage` accepts today) or an array of
/// [`ContentPart`]s (the OpenAI multimodal shape).
///
/// `#[serde(untagged)]` tries each variant in declaration order, so a bare
/// JSON string deserializes as [`MessageContent::Text`] and a JSON array as
/// [`MessageContent::Parts`] — matching the wire format exactly, including
/// on the way back out: serializing `Text(s)` re-emits the bare string `s`,
/// not `{"Text": s}`, so a future `ChatMessage.content: Option<MessageContent>`
/// stays byte-identical on responses (which only ever construct `Text`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Multipart content.
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Flatten this content into a single string for the text-only prompt
    /// builders every endpoint uses today.
    ///
    /// A bare string passes through unchanged. A parts array concatenates
    /// every [`ContentPart::Text`] segment (in order, with no separator —
    /// matching the single-text-part case OpenAI examples typically show).
    /// An `image_url` part is never silently dropped: its presence is an
    /// `Err` naming the field, so a caller can surface a `400` rather than
    /// send the model a prompt silently missing the image the client asked
    /// about.
    pub fn into_text_only(self) -> Result<String, String> {
        match self {
            MessageContent::Text(s) => Ok(s),
            MessageContent::Parts(parts) => {
                if parts
                    .iter()
                    .any(|p| matches!(p, ContentPart::ImageUrl { .. }))
                {
                    return Err("image_url content parts are not supported yet; only text \
                         content parts are accepted (image input support is planned)"
                        .to_string());
                }
                Ok(parts
                    .into_iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text),
                        ContentPart::ImageUrl { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""))
            }
        }
    }
}

// ── Extended choice with logprobs ─────────────────────────────────────────────

/// A single completion choice that may include logprobs and tool calls.
#[derive(Debug, serde::Serialize)]
pub struct ExtendedChoice {
    /// Zero-based index of this choice among all returned completions.
    pub index: usize,
    /// The generated assistant message.
    pub message: crate::server::ChatMessage,
    /// Why generation stopped (`"stop"`, `"length"`, `"tool_calls"`, etc.).
    pub finish_reason: String,
    /// Log probability information (present only when `logprobs` was requested).
    pub logprobs: Option<ChoiceLogprobs>,
    /// Tool calls made by the model, if any.
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ── Extended completion response ──────────────────────────────────────────────

/// A full OpenAI-compatible chat completion response.
#[derive(Debug, serde::Serialize)]
pub struct ExtendedChatResponse {
    /// Unique identifier for this completion.
    pub id: String,
    /// Object type: always `"chat.completion"`.
    pub object: String,
    /// Unix timestamp of creation.
    pub created: u64,
    /// The model that generated this completion.
    pub model: String,
    /// One or more completion choices.
    pub choices: Vec<ExtendedChoice>,
    /// Token usage statistics.
    pub usage: UsageInfo,
    /// A fingerprint of the model/backend configuration for reproducibility.
    pub system_fingerprint: Option<String>,
}

// ── Utility functions ─────────────────────────────────────────────────────────

/// Compute logprob information for the chosen token, including top-k alternatives.
///
/// `logits` is the raw (pre-softmax) logit vector from the model.
/// `chosen_token` is the index of the token that was actually sampled.
/// `top_k` is the number of alternatives to include (clamped to `logits.len()`).
/// `id_to_token` maps a token ID to its string representation.
pub fn compute_logprobs(
    logits: &[f32],
    chosen_token: u32,
    top_k: usize,
    id_to_token: &dyn Fn(u32) -> String,
) -> LogprobsContent {
    if logits.is_empty() {
        return LogprobsContent {
            token: id_to_token(chosen_token),
            logprob: 0.0,
            bytes: token_bytes(id_to_token(chosen_token).as_str()),
            top_logprobs: vec![],
        };
    }

    // Compute log-softmax over the full logit vector.
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let log_sum_exp = sum_exp.ln() + max_logit;

    // Build sorted list of (token_id, logprob) for top-k.
    let effective_k = top_k.clamp(1, logits.len());
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l - log_sum_exp))
        .collect();
    // Partial sort: bring top-k to the front.
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(effective_k);

    let chosen_logprob = logits
        .get(chosen_token as usize)
        .copied()
        .unwrap_or(f32::NEG_INFINITY)
        - log_sum_exp;

    let chosen_text = id_to_token(chosen_token);
    let chosen_bytes = token_bytes(&chosen_text);

    let top_logprobs: Vec<TopLogprob> = indexed
        .iter()
        .map(|&(tid, lp)| {
            let text = id_to_token(tid);
            let bytes = token_bytes(&text);
            TopLogprob {
                token: text,
                logprob: lp,
                bytes,
            }
        })
        .collect();

    LogprobsContent {
        token: chosen_text,
        logprob: chosen_logprob,
        bytes: chosen_bytes,
        top_logprobs,
    }
}

/// Return the UTF-8 bytes of a token string, or `None` if empty.
fn token_bytes(token: &str) -> Option<Vec<u8>> {
    if token.is_empty() {
        None
    } else {
        Some(token.as_bytes().to_vec())
    }
}

/// Return `true` if `text` is valid JSON (object or array).
pub fn is_valid_json(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
}

/// Attempt to parse a tool call from generated text.
///
/// The model is expected to emit tool calls in the form:
/// ```text
/// <tool_call>{"name": "fn_name", "arguments": {...}}</tool_call>
/// ```
///
/// Returns `Some(ToolCall)` on success, `None` if the pattern is not found
/// or the inner JSON cannot be parsed.
pub fn parse_tool_call(text: &str, call_id: &str) -> Option<ToolCall> {
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    let start = text.find(start_tag)?;
    let inner_start = start + start_tag.len();
    let end = text[inner_start..].find(end_tag).map(|e| inner_start + e)?;

    let inner = text[inner_start..end].trim();
    let value: serde_json::Value = serde_json::from_str(inner).ok()?;

    let name = value.get("name")?.as_str()?.to_string();
    let arguments = match value.get("arguments") {
        Some(args) => serde_json::to_string(args).ok()?,
        None => "{}".to_string(),
    };

    Some(ToolCall {
        id: call_id.to_string(),
        tool_type: "function".to_string(),
        function: FunctionCallResult { name, arguments },
    })
}

/// Generate a unique tool call identifier with the `call_` prefix.
///
/// Uses a timestamp-derived hash to produce 8 hex characters, yielding
/// identifiers such as `call_1a2b3c4d`.
pub fn generate_tool_call_id() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut hasher = DefaultHasher::new();
    ts.hash(&mut hasher);
    let hash = hasher.finish();
    format!("call_{:08x}", hash & 0xFFFF_FFFF)
}

/// Compute a stable hex fingerprint from a model configuration value.
///
/// Used to populate `system_fingerprint` in responses, giving clients a way
/// to detect backend configuration changes between requests.
pub fn fingerprint_from_config(config_hash_input: &str) -> String {
    let mut hasher = DefaultHasher::new();
    config_hash_input.hash(&mut hasher);
    format!("fp_{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_sequences_single_as_slice() {
        let s = StopSequences::Single("stop".to_string());
        assert_eq!(s.as_slice(), &["stop"]);
    }

    #[test]
    fn stop_sequences_multiple_as_slice() {
        let s = StopSequences::Multiple(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(s.as_slice(), &["a", "b"]);
    }

    #[test]
    fn stop_sequences_single_into_vec() {
        let s = StopSequences::Single("x".to_string());
        assert_eq!(s.into_vec(), vec!["x"]);
    }

    #[test]
    fn stop_sequences_multiple_into_vec() {
        let s = StopSequences::Multiple(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(s.into_vec(), vec!["a", "b"]);
    }

    #[test]
    fn is_valid_json_object() {
        assert!(is_valid_json(r#"{"key": "value"}"#));
    }

    #[test]
    fn is_valid_json_array() {
        assert!(is_valid_json(r#"[1, 2, 3]"#));
    }

    #[test]
    fn is_valid_json_invalid() {
        assert!(!is_valid_json("not json"));
        assert!(!is_valid_json(""));
    }

    #[test]
    fn parse_tool_call_valid() {
        let text = r#"<tool_call>{"name":"get_weather","arguments":{"city":"London"}}</tool_call>"#;
        let tc = parse_tool_call(text, "call_abc123").expect("should parse");
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.id, "call_abc123");
        assert_eq!(tc.tool_type, "function");
    }

    #[test]
    fn parse_tool_call_invalid() {
        let text = "No tool call here";
        assert!(parse_tool_call(text, "call_x").is_none());
    }

    #[test]
    fn generate_tool_call_id_prefix() {
        let id = generate_tool_call_id();
        assert!(id.starts_with("call_"), "expected call_ prefix, got: {id}");
        assert_eq!(id.len(), 13, "expected 13 chars, got: {id}");
    }

    #[test]
    fn fingerprint_from_config_stable() {
        let fp1 = fingerprint_from_config("bonsai-8b");
        let fp2 = fingerprint_from_config("bonsai-8b");
        assert_eq!(fp1, fp2);
        assert!(fp1.starts_with("fp_"));
    }

    #[test]
    fn compute_logprobs_top_tokens() {
        let logits = vec![1.0f32, 3.0, 2.0, 0.5, 1.5];
        let lp = compute_logprobs(&logits, 1, 3, &|id| format!("tok{id}"));
        assert_eq!(lp.token, "tok1");
        assert!(
            lp.logprob <= 0.0,
            "logprob should be <= 0 (log probability)"
        );
        assert_eq!(lp.top_logprobs.len(), 3);
        // The highest logit (index 1) should be the first top logprob
        assert_eq!(lp.top_logprobs[0].token, "tok1");
    }

    // ── MessageContent / ContentPart (SV-11 prepare-only) ────────────────────

    #[test]
    fn message_content_deserializes_bare_string() {
        let mc: MessageContent = serde_json::from_str(r#""hello world""#).expect("string form");
        assert_eq!(mc, MessageContent::Text("hello world".to_string()));
        assert_eq!(mc.into_text_only().expect("text only"), "hello world");
    }

    #[test]
    fn message_content_deserializes_text_parts_array() {
        let json = serde_json::json!([
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]);
        let mc: MessageContent = serde_json::from_value(json).expect("parts form");
        assert_eq!(
            mc,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "hello ".to_string()
                },
                ContentPart::Text {
                    text: "world".to_string()
                },
            ])
        );
        assert_eq!(mc.into_text_only().expect("text only"), "hello world");
    }

    #[test]
    fn message_content_single_text_part_round_trips() {
        let json = serde_json::json!([{"type": "text", "text": "hi"}]);
        let mc: MessageContent = serde_json::from_value(json).expect("parts form");
        assert_eq!(mc.into_text_only().expect("text only"), "hi");
    }

    #[test]
    fn content_part_image_url_deserializes_structurally() {
        let json = serde_json::json!({
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.png", "detail": "high"}
        });
        let part: ContentPart = serde_json::from_value(json).expect("image_url part");
        match part {
            ContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/cat.png");
                assert_eq!(image_url.detail.as_deref(), Some("high"));
            }
            ContentPart::Text { .. } => panic!("expected ImageUrl variant"),
        }
    }

    #[test]
    fn message_content_rejects_image_url_honestly() {
        let json = serde_json::json!([
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]);
        let mc: MessageContent = serde_json::from_value(json).expect("parts form");
        let err = mc.into_text_only().expect_err("image_url must be rejected");
        assert!(
            err.contains("image_url"),
            "error must name the unsupported field, got: {err}"
        );
        assert!(
            !err.to_lowercase().contains("todo") && !err.to_lowercase().contains("stub"),
            "rejection message must be a real, honest error, not a stub marker"
        );
    }

    #[test]
    fn message_content_text_serializes_as_bare_string() {
        // Round-tripping `Text` must stay byte-identical to a plain string —
        // this is what keeps a future `ChatMessage.content:
        // Option<MessageContent>` from changing today's response wire shape.
        let mc = MessageContent::Text("hi".to_string());
        let json = serde_json::to_string(&mc).expect("serialize");
        assert_eq!(json, r#""hi""#);
    }
}
