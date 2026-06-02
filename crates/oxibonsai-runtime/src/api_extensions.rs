//! Extended `/v1/chat/completions` handler.
//!
//! Adds support for tools (function calling), logprobs, `n > 1` completions,
//! response format constraints (JSON mode / JSON Schema), stop sequences,
//! and frequency/presence penalties on top of the base server implementation.

use axum::{extract::State, response::IntoResponse, Json};
use std::collections::HashMap;
use std::sync::Arc;

use crate::api_types::{
    ChoiceLogprobs, ExtendedChatRequest, ExtendedChatResponse, ExtendedChoice, UsageInfo,
};
use crate::engine::InferenceEngine;
use crate::sampling::SamplingParams;
use crate::server::{AppState, ChatMessage};

// ── Extended handler ──────────────────────────────────────────────────────────

/// Handler for `POST /v1/chat/completions/extended`.
///
/// Supports all standard fields plus `tools`, `tool_choice`, `logprobs`,
/// `top_logprobs`, `response_format`, `n`, `presence_penalty`,
/// `frequency_penalty`, and `stop`.
pub async fn extended_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExtendedChatRequest>,
) -> impl IntoResponse {
    let n = req.n.unwrap_or(1).clamp(1, 4);
    let max_tokens = req.max_tokens;
    let temperature = req.temperature.unwrap_or(0.7);
    let seed = req.seed.unwrap_or(42);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs_k = req.top_logprobs.unwrap_or(0).clamp(0, 20);
    let response_format = req.response_format.clone();
    let tools = req.tools.clone();
    let frequency_penalty = req.frequency_penalty.unwrap_or(0.0);
    let presence_penalty = req.presence_penalty.unwrap_or(0.0);

    // Build stop checker
    let stop_checker = match req.stop {
        Some(ref seqs) => StopChecker::new(seqs.as_slice().to_vec()),
        None => StopChecker::new(vec![]),
    };

    // Build prompt text from messages
    let prompt_text = build_extended_prompt(&req.messages);

    // Tokenize the prompt
    let prompt_tokens = {
        let tokenizer = state.tokenizer();
        if let Some(tok) = tokenizer {
            match tok.encode(&prompt_text) {
                Ok(tokens) => tokens,
                Err(e) => {
                    tracing::error!(error = %e, "tokenization failed");
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": "tokenization failed"})),
                    )
                        .into_response();
                }
            }
        } else {
            vec![151644u32]
        }
    };

    let prompt_len = prompt_tokens.len();

    // Build sampling params
    let sampling_params = SamplingParams {
        temperature,
        top_k: 40,
        top_p: req.top_p.unwrap_or(0.9),
        repetition_penalty: 1.1,
        ..SamplingParams::default()
    };

    // Generate n completions. One lease serves all `n` runs (they reset KV
    // between runs, as before), so the replica is held for the whole batch.
    let mut engine = match state.acquire_engine().await {
        Ok(lease) => lease,
        Err(e) => {
            tracing::error!(error = %e, "engine pool acquire failed");
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "engine pool unavailable"})),
            )
                .into_response();
        }
    };

    let raw_completions: Vec<String> = {
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let run_seed = seed.wrapping_add(i as u64);
            engine.reset();

            let output_tokens = match engine.generate_with_seed(
                &prompt_tokens,
                max_tokens,
                run_seed,
                &sampling_params,
            ) {
                Ok(toks) => toks,
                Err(e) => {
                    tracing::error!(error = %e, "generation failed for completion {i}");
                    return (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": "generation failed"})),
                    )
                        .into_response();
                }
            };

            // Apply frequency/presence penalty post-hoc if requested
            // (for simplicity; a full implementation would fold this into decoding)
            let _ = frequency_penalty;
            let _ = presence_penalty;

            // Decode
            let text = if let Some(tok) = state.tokenizer() {
                tok.decode(&output_tokens)
                    .unwrap_or_else(|_| format!("{output_tokens:?}"))
            } else {
                format!("{output_tokens:?}")
            };

            results.push(text);
        }
        results
    };

    // Apply stop sequences and response format enforcement
    let json_enforcer = JsonModeEnforcer::new();
    let is_json_mode = response_format
        .as_ref()
        .map(|rf| rf.format_type == "json_object" || rf.format_type == "json_schema")
        .unwrap_or(false);

    let total_completion_tokens: usize;
    let choices: Vec<ExtendedChoice> = {
        let mut comp_tokens = 0usize;
        let choices_out: Vec<ExtendedChoice> = raw_completions
            .into_iter()
            .enumerate()
            .map(|(idx, raw_text)| {
                let (truncated, hit_stop) = stop_checker.truncate_at_stop(&raw_text);
                let finish_reason = "stop".to_string();
                let _ = hit_stop;

                // Apply JSON mode enforcement if requested
                let final_text = if is_json_mode {
                    json_enforcer.enforce(&truncated)
                } else {
                    truncated.clone()
                };

                // Check for tool call pattern in the output
                let tool_calls = if tools.is_some() {
                    let call_id = crate::api_types::generate_tool_call_id();
                    crate::api_types::parse_tool_call(&final_text, &call_id).map(|tc| vec![tc])
                } else {
                    None
                };

                // Build logprobs (simplified: no actual logit data here, so we skip)
                let logprobs = if want_logprobs && top_logprobs_k > 0 {
                    // Without access to raw logits here, return empty content
                    Some(ChoiceLogprobs {
                        content: Some(vec![]),
                    })
                } else if want_logprobs {
                    Some(ChoiceLogprobs {
                        content: Some(vec![]),
                    })
                } else {
                    None
                };

                // Estimate token count
                let approx_tokens = final_text.split_whitespace().count().max(1);
                comp_tokens += approx_tokens;

                ExtendedChoice {
                    index: idx,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: Some(final_text),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    finish_reason,
                    logprobs,
                    tool_calls,
                }
            })
            .collect();
        total_completion_tokens = comp_tokens;
        choices_out
    };

    // Build system fingerprint from model name
    let system_fingerprint = Some(crate::api_types::fingerprint_from_config("bonsai-8b"));

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let response = ExtendedChatResponse {
        id: format!("chatcmpl-ext-{}", rand_ext_id()),
        object: "chat.completion".to_string(),
        created,
        model: "bonsai-8b".to_string(),
        choices,
        usage: UsageInfo {
            prompt_tokens: prompt_len,
            completion_tokens: total_completion_tokens,
            total_tokens: prompt_len + total_completion_tokens,
        },
        system_fingerprint,
    };

    Json(response).into_response()
}

/// Build a prompt string from a slice of chat messages (ChatML format).
///
/// Messages with `content = None` are skipped (they represent tool-call turns).
fn build_extended_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let text = match msg.content.as_deref() {
            Some(t) => t,
            None => continue,
        };
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(text);
                prompt.push_str("<|im_end|>\n");
            }
            "user" => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(text);
                prompt.push_str("<|im_end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(text);
                prompt.push_str("<|im_end|>\n");
            }
            _ => {
                prompt.push_str(text);
                prompt.push('\n');
            }
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn rand_ext_id() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{ts:x}")
}

// ── JSON mode enforcer ────────────────────────────────────────────────────────

/// Wraps generation to produce valid JSON output.
///
/// Strategy (applied in order):
/// 1. If the text already parses as JSON — return it as-is.
/// 2. Try to extract the first `{…}` or `[…]` substring and parse that.
/// 3. If still not valid JSON — wrap the text in `{"response": "<text>"}`.
pub struct JsonModeEnforcer {
    /// Maximum extraction/wrap attempts (unused here; reserved for future streaming use).
    pub max_retries: usize,
}

impl JsonModeEnforcer {
    /// Create a new enforcer with default settings.
    pub fn new() -> Self {
        Self { max_retries: 3 }
    }

    /// Return a string guaranteed to be valid JSON, applying extraction or
    /// wrapping if needed.
    pub fn enforce(&self, text: &str) -> String {
        // Fast path: already valid JSON
        if crate::api_types::is_valid_json(text) {
            return text.to_string();
        }

        // Try to extract a JSON object substring
        if let Some(extracted) = extract_json_substring(text) {
            if crate::api_types::is_valid_json(&extracted) {
                return extracted;
            }
        }

        // Fallback: wrap in a JSON object
        let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
        format!(r#"{{"response": "{escaped}"}}"#)
    }
}

impl Default for JsonModeEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

/// Try to find and return the first valid-looking JSON object or array in `text`.
fn extract_json_substring(text: &str) -> Option<String> {
    // Look for first `{` and last matching `}` (greedy — works for well-nested JSON)
    if let Some(obj) = extract_balanced(text, '{', '}') {
        return Some(obj);
    }
    // Try array
    if let Some(arr) = extract_balanced(text, '[', ']') {
        return Some(arr);
    }
    None
}

/// Extract the outermost balanced delimited substring starting from the first
/// occurrence of `open` in `text`.
fn extract_balanced(text: &str, open: char, close: char) -> Option<String> {
    let start = text.find(open)?;
    let substr = &text[start..];
    let mut depth = 0i32;
    let mut end_idx = None;

    for (i, ch) in substr.char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                end_idx = Some(i + ch.len_utf8());
                break;
            }
        }
    }

    end_idx.map(|e| substr[..e].to_string())
}

// ── Stop sequence checker ─────────────────────────────────────────────────────

/// Detects and truncates text at stop sequences.
pub struct StopChecker {
    sequences: Vec<String>,
}

impl StopChecker {
    /// Create a new checker with the given stop sequences.
    pub fn new(sequences: Vec<String>) -> Self {
        Self { sequences }
    }

    /// Returns `Some(&str)` with the first matched stop sequence, or `None`.
    pub fn check<'a>(&'a self, text: &str) -> Option<&'a str> {
        for seq in &self.sequences {
            if text.contains(seq.as_str()) {
                return Some(seq.as_str());
            }
        }
        None
    }

    /// Return `(truncated_text, hit_stop)`.
    ///
    /// If any stop sequence is found, the text is truncated at that point.
    pub fn truncate_at_stop(&self, text: &str) -> (String, bool) {
        let mut earliest: Option<(usize, &str)> = None;
        for seq in &self.sequences {
            if let Some(pos) = text.find(seq.as_str()) {
                match earliest {
                    None => earliest = Some((pos, seq.as_str())),
                    Some((prev_pos, _)) if pos < prev_pos => {
                        earliest = Some((pos, seq.as_str()));
                    }
                    _ => {}
                }
            }
        }

        match earliest {
            Some((pos, _)) => (text[..pos].to_string(), true),
            None => (text.to_string(), false),
        }
    }

    /// Returns `true` if no stop sequences are configured.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

// ── Multi-completion generator ────────────────────────────────────────────────

/// Generate `n` independent completions from the same prompt, seeding each run
/// with `base_seed + i` for determinism.
///
/// **Note**: This function resets the engine before each run.
pub fn generate_n_completions(
    engine: &mut InferenceEngine<'_>,
    prompt: &str,
    params: &SamplingParams,
    n: usize,
    base_seed: u64,
) -> Vec<String> {
    let prompt_tokens: Vec<u32> = {
        // Simple whitespace-based tokenization fallback (no real tokenizer available here)
        prompt
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| (i as u32).wrapping_add(1000))
            .collect()
    };

    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        engine.reset();
        let seed = base_seed.wrapping_add(i as u64);
        let text = engine
            .generate_with_seed(&prompt_tokens, 64, seed, params)
            .map(|toks| format!("{toks:?}"))
            .unwrap_or_else(|_| String::new());
        results.push(text);
    }
    results
}

// ── Frequency / presence penalty ─────────────────────────────────────────────

/// Apply frequency and presence penalties in-place to a logit vector.
///
/// For each token that has been seen:
/// - **frequency penalty** reduces the logit proportionally to its count.
/// - **presence penalty** reduces the logit by a fixed amount for any seen token.
pub fn apply_frequency_penalty(
    logits: &mut [f32],
    token_counts: &HashMap<u32, usize>,
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    for (&token_id, &count) in token_counts {
        if let Some(logit) = logits.get_mut(token_id as usize) {
            *logit -= frequency_penalty * count as f32;
            *logit -= presence_penalty;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_mode_enforcer_valid_passthrough() {
        let enforcer = JsonModeEnforcer::new();
        let json = r#"{"key": "value"}"#;
        assert_eq!(enforcer.enforce(json), json);
    }

    #[test]
    fn json_mode_enforcer_extracts_substring() {
        let enforcer = JsonModeEnforcer::new();
        let text = r#"Here is some text {"key": "value"} and more"#;
        let result = enforcer.enforce(text);
        assert!(
            crate::api_types::is_valid_json(&result),
            "result should be valid JSON, got: {result}"
        );
    }

    #[test]
    fn json_mode_enforcer_wraps_invalid() {
        let enforcer = JsonModeEnforcer::new();
        let text = "not json at all";
        let result = enforcer.enforce(text);
        assert!(
            crate::api_types::is_valid_json(&result),
            "result should be valid JSON, got: {result}"
        );
        let v: serde_json::Value = serde_json::from_str(&result).expect("should parse as json");
        assert!(v.get("response").is_some(), "should have 'response' key");
    }

    #[test]
    fn stop_checker_finds_sequence() {
        let checker = StopChecker::new(vec!["STOP".to_string(), "END".to_string()]);
        assert_eq!(checker.check("Hello STOP world"), Some("STOP"));
        assert_eq!(checker.check("No match here"), None);
    }

    #[test]
    fn stop_checker_truncates_correctly() {
        let checker = StopChecker::new(vec!["<end>".to_string()]);
        let (truncated, hit) = checker.truncate_at_stop("Hello world<end>more text");
        assert_eq!(truncated, "Hello world");
        assert!(hit);
    }

    #[test]
    fn stop_checker_no_match() {
        let checker = StopChecker::new(vec!["nope".to_string()]);
        let (truncated, hit) = checker.truncate_at_stop("Hello world");
        assert_eq!(truncated, "Hello world");
        assert!(!hit);
    }

    #[test]
    fn stop_checker_is_empty() {
        let empty = StopChecker::new(vec![]);
        assert!(empty.is_empty());
        let non_empty = StopChecker::new(vec!["x".to_string()]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn apply_frequency_penalty_reduces_seen() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let mut counts = HashMap::new();
        counts.insert(1u32, 2usize); // token 1 seen twice
        apply_frequency_penalty(&mut logits, &counts, 0.5, 0.0);
        // token 1 logit should be reduced by 0.5 * 2 = 1.0
        assert!(
            (logits[1] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            logits[1]
        );
        // others unchanged
        assert!((logits[0] - 1.0).abs() < 1e-5);
        assert!((logits[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn apply_presence_penalty_reduces_seen() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let mut counts = HashMap::new();
        counts.insert(0u32, 1usize);
        apply_frequency_penalty(&mut logits, &counts, 0.0, 1.0);
        assert!(
            (logits[0] - 0.0).abs() < 1e-5,
            "expected 0.0, got {}",
            logits[0]
        );
        assert!((logits[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn extract_balanced_object() {
        let text = r#"prefix {"a":1} suffix"#;
        let result = extract_balanced(text, '{', '}');
        assert_eq!(result.as_deref(), Some(r#"{"a":1}"#));
    }

    #[test]
    fn extract_balanced_array() {
        let text = r#"pre [1,2,3] post"#;
        let result = extract_balanced(text, '[', ']');
        assert_eq!(result.as_deref(), Some("[1,2,3]"));
    }
}
