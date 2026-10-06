//! WASM-compatible inference API.
//!
//! Provides a JSON-in / JSON-out interface for WASM hosts (wasmtime, wasmer,
//! browser environments, etc.). No `wasm-bindgen` required — this module works
//! with any WASM runtime that can call exported functions with string arguments.
//!
//! ## Text prompts (TOK-11)
//!
//! The crate advertises a "WASM-safe, zero-FFI" Pure-Rust tokenizer
//! (the native backend of [`crate::tokenizer_bridge::TokenizerBridge`], backed by
//! [`oxibonsai_tokenizer`]) that compiles for `wasm32-unknown-unknown` with
//! no C dependency, but this module used to accept only raw token ids —
//! nothing here ever wired the advertised tokenizer up. `prompt_text` /
//! `tokenizer_json` below is that wiring: a WASM host with no filesystem can
//! pass the *contents* of a `tokenizer.json` alongside plain text and get
//! decoded text back, going through
//! [`TokenizerBridge::native_from_json_str`] — the Pure-Rust backend that is
//! always available, including on `wasm32-unknown-unknown`, regardless of
//! whether the `hf-tokenizer` feature is compiled in.
//!
//! ## Request format
//!
//! Token-id prompt (unchanged from before this fix):
//!
//! ```json
//! {
//!   "hidden_size": 4096,
//!   "num_layers": 32,
//!   "num_attention_heads": 32,
//!   "num_kv_heads": 8,
//!   "intermediate_size": 14336,
//!   "vocab_size": 151936,
//!   "max_context_length": 32768,
//!   "rms_norm_eps": 1e-6,
//!   "rope_theta": 1000000.0,
//!   "head_dim": 128,
//!   "prompt_tokens": [151644, 872, 151645],
//!   "max_tokens": 32,
//!   "temperature": 0.7,
//!   "top_k": 40,
//!   "top_p": 0.9,
//!   "seed": 42
//! }
//! ```
//!
//! Text prompt (TOK-11 — `prompt_tokens` above is replaced by `prompt_text`
//! plus `tokenizer_json`; every other field is the same):
//!
//! ```json
//! {
//!   "hidden_size": 256, "num_layers": 2, "num_attention_heads": 4,
//!   "num_kv_heads": 2, "intermediate_size": 512, "vocab_size": 1024,
//!   "max_context_length": 512, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
//!   "head_dim": 64,
//!   "prompt_text": "Hello!",
//!   "tokenizer_json": "{ ...contents of a tokenizer.json... }",
//!   "max_tokens": 5
//! }
//! ```
//!
//! Exactly one of `prompt_tokens` / `prompt_text` must be present — a
//! request with neither, or both, is rejected with an error rather than
//! guessing which one was meant.
//!
//! ## Response format (success)
//!
//! Token-id prompt (unchanged): `text` is omitted entirely.
//!
//! ```json
//! { "tokens": [1234, 5678, ...], "error": null }
//! ```
//!
//! Text prompt, or any request that supplied `tokenizer_json` (even
//! alongside `prompt_tokens`, to get the generated ids detokenized too):
//! `text` holds the decoded completion.
//!
//! ```json
//! { "tokens": [1234, 5678, ...], "text": "some completion", "error": null }
//! ```
//!
//! ## Response format (error)
//!
//! ```json
//! { "tokens": [], "error": "description of the error" }
//! ```

use crate::engine::InferenceEngine;
use crate::sampling::SamplingParams;
use crate::tokenizer_bridge::TokenizerBridge;
use oxibonsai_core::config::{Qwen3Config, RopeScaling};

// ─── Request / Response types ────────────────────────────────────────────────

/// JSON request for WASM inference.
#[derive(serde::Deserialize, Debug)]
struct WasmInferenceRequest {
    // ── Model architecture (matches Qwen3Config fields) ──────────────────
    hidden_size: usize,
    num_layers: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_context_length: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    head_dim: usize,

    // ── Inference parameters ──────────────────────────────────────────────
    /// Prompt as a list of token IDs. Mutually exclusive with `prompt_text`
    /// (TOK-11) — exactly one of the two must be present.
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    /// Prompt as raw text, tokenized with the embedded native BPE backend
    /// (TOK-11). Requires `tokenizer_json` to also be present. Mutually
    /// exclusive with `prompt_tokens`.
    #[serde(default)]
    prompt_text: Option<String>,
    /// Contents (**not** a path — this API does no file I/O, so it works on
    /// a `wasm32-unknown-unknown` host with no filesystem) of a
    /// HuggingFace-format `tokenizer.json`. Used to tokenize `prompt_text`
    /// when present, and — whenever this field is present at all, even
    /// alongside a plain `prompt_tokens` request — to detokenize the
    /// generated ids into the response's `text` field.
    #[serde(default)]
    tokenizer_json: Option<String>,
    /// Maximum number of tokens to generate.
    max_tokens: usize,

    // ── Sampling parameters (all optional with sensible defaults) ─────────
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_seed")]
    seed: u64,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_top_k() -> usize {
    40
}
fn default_top_p() -> f32 {
    0.9
}
fn default_seed() -> u64 {
    42
}

/// JSON response from WASM inference.
#[derive(serde::Serialize, Debug)]
struct WasmInferenceResponse {
    /// Generated token IDs (empty on error).
    tokens: Vec<u32>,
    /// Decoded completion text, when a `tokenizer_json` was supplied
    /// (TOK-11). Omitted entirely (not `null`) for a plain token-id request,
    /// so that request shape's response JSON is byte-for-byte unchanged from
    /// before this fix.
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    /// Error message, or `null` on success.
    error: Option<String>,
}

impl WasmInferenceResponse {
    fn success(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            text: None,
            error: None,
        }
    }

    fn success_with_text(tokens: Vec<u32>, text: String) -> Self {
        Self {
            tokens,
            text: Some(text),
            error: None,
        }
    }

    fn error(msg: impl Into<String>) -> Self {
        Self {
            tokens: vec![],
            text: None,
            error: Some(msg.into()),
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Run inference from a JSON request string, returning a JSON response string.
///
/// This is the primary entry point for WASM hosts. It is synchronous and
/// self-contained: creates an engine, runs generation, and returns results.
///
/// All configuration is passed via JSON — no file I/O or mmap required,
/// making this fully compatible with wasm32-unknown-unknown.
///
/// # Example (Rust)
///
/// ```no_run
/// let req = r#"{
///   "hidden_size": 256, "num_layers": 2, "num_attention_heads": 4,
///   "num_kv_heads": 2, "intermediate_size": 512, "vocab_size": 1024,
///   "max_context_length": 512, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
///   "head_dim": 64, "prompt_tokens": [1, 2, 3], "max_tokens": 5
/// }"#;
/// let resp = oxibonsai_runtime::wasm_api::generate_json(req);
/// // resp is a JSON string like {"tokens":[...],"error":null}
/// ```
pub fn generate_json(request_json: &str) -> String {
    let response = match run_inference(request_json) {
        Ok((tokens, Some(text))) => WasmInferenceResponse::success_with_text(tokens, text),
        Ok((tokens, None)) => WasmInferenceResponse::success(tokens),
        Err(e) => WasmInferenceResponse::error(e),
    };

    match serde_json::to_string(&response) {
        Ok(s) => s,
        Err(e) => format!(r#"{{"tokens":[],"error":"failed to serialize response: {e}"}}"#),
    }
}

/// Run inference, returning `(generated token ids, decoded text if a
/// tokenizer was supplied)` or an error description.
fn run_inference(request_json: &str) -> Result<(Vec<u32>, Option<String>), String> {
    let req: WasmInferenceRequest =
        serde_json::from_str(request_json).map_err(|e| format!("invalid request JSON: {e}"))?;

    // TOK-11: load the tokenizer (if any) up front so both the prompt-source
    // resolution below and the post-generation detokenize step can use it.
    let tokenizer = match &req.tokenizer_json {
        Some(json) => Some(
            TokenizerBridge::native_from_json_str(json)
                .map_err(|e| format!("invalid tokenizer_json: {e}"))?,
        ),
        None => None,
    };

    // Exactly one of `prompt_tokens` / `prompt_text` — never silently prefer
    // one over the other, and never silently treat "neither supplied" as an
    // empty prompt (that would turn a request that used to fail
    // deserialization outright into a quiet success).
    let prompt_tokens = match (&req.prompt_tokens, &req.prompt_text) {
        (Some(_), Some(_)) => {
            return Err(
                "specify exactly one of `prompt_tokens` or `prompt_text`, not both".to_string(),
            );
        }
        (None, None) => {
            return Err(
                "request must supply exactly one of `prompt_tokens` or `prompt_text`".to_string(),
            );
        }
        (Some(ids), None) => ids.clone(),
        (None, Some(text)) => match &tokenizer {
            Some(tok) => tok
                .encode(text)
                .map_err(|e| format!("failed to encode prompt_text: {e}"))?,
            None => {
                return Err(
                    "prompt_text requires tokenizer_json (the contents of a tokenizer.json) to \
                     be supplied alongside it"
                        .to_string(),
                );
            }
        },
    };

    let config = Qwen3Config {
        hidden_size: req.hidden_size,
        num_layers: req.num_layers,
        num_attention_heads: req.num_attention_heads,
        num_kv_heads: req.num_kv_heads,
        intermediate_size: req.intermediate_size,
        vocab_size: req.vocab_size,
        max_context_length: req.max_context_length,
        rms_norm_eps: req.rms_norm_eps,
        rope_freq_base: req.rope_theta,
        head_dim: req.head_dim,
        value_length: req.head_dim,
        rope_scaling: RopeScaling::None,
        sliding_window: None,
        architecture: "qwen3".to_string(),
        model_name: "bonsai".to_string(),
    };

    let sampling = SamplingParams {
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        // The `1.0` no-op default, standardised workspace-wide (RT-24).
        // `WasmInferenceRequest` has no `repetition_penalty` field of its
        // own yet (a caller cannot opt into a non-default penalty through
        // this API at all today).
        repetition_penalty: 1.0,
        ..SamplingParams::default()
    };

    let mut engine = InferenceEngine::new(config, sampling, req.seed);

    let generated = engine
        .generate(&prompt_tokens, req.max_tokens)
        .map_err(|e| format!("inference error: {e}"))?;

    let text = match &tokenizer {
        Some(tok) => Some(
            tok.decode(&generated)
                .map_err(|e| format!("failed to decode generated tokens: {e}"))?,
        ),
        None => None,
    };

    Ok((generated, text))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config_json(prompt_tokens: &[u32], max_tokens: usize) -> String {
        let tokens_json = serde_json::to_string(prompt_tokens).expect("serialize tokens");
        format!(
            r#"{{
              "hidden_size": 256,
              "num_layers": 2,
              "num_attention_heads": 4,
              "num_kv_heads": 2,
              "intermediate_size": 512,
              "vocab_size": 1024,
              "max_context_length": 128,
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0,
              "head_dim": 64,
              "prompt_tokens": {tokens_json},
              "max_tokens": {max_tokens}
            }}"#
        )
    }

    /// A minimal HuggingFace-format `tokenizer.json`: byte-level BPE with no
    /// merges (every ASCII char is its own token id), covering just enough
    /// vocabulary to encode/decode "Hi" (TOK-11 fixture).
    const TINY_TOKENIZER_JSON: &str = r##"{
        "model": {
            "type": "BPE",
            "vocab": { "H": 0, "i": 1 },
            "merges": []
        },
        "added_tokens": [],
        "pre_tokenizer": { "type": "ByteLevel" },
        "decoder": { "type": "ByteLevel" }
    }"##;

    /// Build a request body using `prompt_text` + `tokenizer_json` (TOK-11)
    /// instead of raw `prompt_tokens`. `max_tokens: 0` keeps every such test
    /// deterministic — it exercises the encode/decode wiring without
    /// depending on what an untrained/randomly-initialized model happens to
    /// generate.
    fn tiny_text_config_json(prompt_text: &str, tokenizer_json: &str, max_tokens: usize) -> String {
        let prompt_text_json = serde_json::to_string(prompt_text).expect("serialize prompt_text");
        let tokenizer_json_json =
            serde_json::to_string(tokenizer_json).expect("serialize tokenizer_json");
        format!(
            r#"{{
              "hidden_size": 256,
              "num_layers": 2,
              "num_attention_heads": 4,
              "num_kv_heads": 2,
              "intermediate_size": 512,
              "vocab_size": 1024,
              "max_context_length": 128,
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0,
              "head_dim": 64,
              "prompt_text": {prompt_text_json},
              "tokenizer_json": {tokenizer_json_json},
              "max_tokens": {max_tokens}
            }}"#
        )
    }

    #[test]
    fn generate_json_empty_prompt_returns_empty_tokens() {
        let req = tiny_config_json(&[], 5);
        let resp_str = generate_json(&req);
        let resp: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON response");
        assert!(resp["error"].is_null(), "expected no error, got: {resp}");
        let tokens = resp["tokens"].as_array().expect("tokens array");
        assert!(tokens.is_empty(), "empty prompt should yield no tokens");
    }

    #[test]
    fn generate_json_invalid_json_returns_error() {
        let resp_str = generate_json("this is not json");
        let resp: serde_json::Value =
            serde_json::from_str(&resp_str).expect("response should be valid JSON");
        assert!(
            !resp["error"].is_null(),
            "invalid input should produce an error"
        );
    }

    #[test]
    fn generate_json_missing_required_field_returns_error() {
        // Missing hidden_size
        let req = r#"{"num_layers": 2, "prompt_tokens": [1], "max_tokens": 1}"#;
        let resp_str = generate_json(req);
        let resp: serde_json::Value =
            serde_json::from_str(&resp_str).expect("response should be valid JSON");
        assert!(
            !resp["error"].is_null(),
            "missing fields should produce an error"
        );
    }

    #[test]
    fn response_serialization_success() {
        let r = WasmInferenceResponse::success(vec![1, 2, 3]);
        let s = serde_json::to_string(&r).expect("serialize");
        assert!(s.contains("\"tokens\":[1,2,3]"));
        assert!(s.contains("\"error\":null"));
    }

    #[test]
    fn response_serialization_error() {
        let r = WasmInferenceResponse::error("something went wrong");
        let s = serde_json::to_string(&r).expect("serialize");
        assert!(s.contains("\"tokens\":[]"));
        assert!(s.contains("\"error\":\"something went wrong\""));
    }

    #[test]
    fn response_serialization_with_text() {
        let r = WasmInferenceResponse::success_with_text(vec![1, 2], "hi".to_string());
        let s = serde_json::to_string(&r).expect("serialize");
        assert!(s.contains("\"tokens\":[1,2]"));
        assert!(s.contains("\"text\":\"hi\""));
        assert!(s.contains("\"error\":null"));
    }

    // ── TOK-11: text prompt / tokenizer_json wiring ─────────────────────────

    #[test]
    fn plain_token_request_response_has_no_text_field_at_all() {
        // Byte-for-byte compatibility with pre-TOK-11 callers: a
        // `prompt_tokens` request with no `tokenizer_json` must not gain a
        // `"text": null` key.
        let req = tiny_config_json(&[], 0);
        let resp_str = generate_json(&req);
        let resp: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON");
        assert!(
            resp.get("text").is_none(),
            "a plain token-id request must omit `text` entirely; got {resp}"
        );
    }

    #[test]
    fn prompt_text_with_tokenizer_json_succeeds_and_returns_text_field() {
        let req = tiny_text_config_json("Hi", TINY_TOKENIZER_JSON, 0);
        let resp_str = generate_json(&req);
        let resp: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON");
        assert!(resp["error"].is_null(), "expected no error, got: {resp}");
        assert_eq!(
            resp["tokens"].as_array().expect("tokens array").len(),
            0,
            "max_tokens: 0 must generate nothing"
        );
        assert_eq!(
            resp["text"].as_str(),
            Some(""),
            "decode of zero generated tokens must be the empty string, not null/absent; got {resp}"
        );
    }

    #[test]
    fn prompt_tokens_with_tokenizer_json_also_decodes_text() {
        // `tokenizer_json` detokenizes the *output* regardless of which
        // prompt source was used for the input.
        let req = format!(
            r#"{{
              "hidden_size": 256, "num_layers": 2, "num_attention_heads": 4,
              "num_kv_heads": 2, "intermediate_size": 512, "vocab_size": 1024,
              "max_context_length": 128, "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0, "head_dim": 64,
              "prompt_tokens": [0, 1],
              "tokenizer_json": {},
              "max_tokens": 0
            }}"#,
            serde_json::to_string(TINY_TOKENIZER_JSON).expect("serialize")
        );
        let resp_str = generate_json(&req);
        let resp: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON");
        assert!(resp["error"].is_null(), "expected no error, got: {resp}");
        assert_eq!(resp["text"].as_str(), Some(""));
    }

    #[test]
    fn prompt_text_without_tokenizer_json_is_a_clear_error() {
        let req = r#"{
              "hidden_size": 256, "num_layers": 2, "num_attention_heads": 4,
              "num_kv_heads": 2, "intermediate_size": 512, "vocab_size": 1024,
              "max_context_length": 128, "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0, "head_dim": 64,
              "prompt_text": "Hi",
              "max_tokens": 1
            }"#;
        let resp_str = generate_json(req);
        let resp: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON");
        let error = resp["error"].as_str().expect("must be an error response");
        assert!(
            error.contains("tokenizer_json"),
            "error should explain the missing tokenizer_json; got {error:?}"
        );
    }

    #[test]
    fn both_prompt_tokens_and_prompt_text_is_rejected_not_silently_resolved() {
        let req = format!(
            r#"{{
              "hidden_size": 256, "num_layers": 2, "num_attention_heads": 4,
              "num_kv_heads": 2, "intermediate_size": 512, "vocab_size": 1024,
              "max_context_length": 128, "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0, "head_dim": 64,
              "prompt_tokens": [1],
              "prompt_text": "Hi",
              "tokenizer_json": {},
              "max_tokens": 1
            }}"#,
            serde_json::to_string(TINY_TOKENIZER_JSON).expect("serialize")
        );
        let resp_str = generate_json(&req);
        let resp: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON");
        let error = resp["error"].as_str().expect("must be an error response");
        assert!(
            error.contains("exactly one"),
            "supplying both prompt sources must be a clear error, not a silent precedence \
             choice; got {error:?}"
        );
    }

    #[test]
    fn neither_prompt_tokens_nor_prompt_text_is_rejected() {
        // A request omitting `prompt_tokens` entirely must still be an
        // error, not a quietly-accepted empty prompt — `prompt_tokens`
        // becoming `Option` for TOK-11 must not turn "missing" into
        // "valid and empty".
        let req = r#"{
              "hidden_size": 256, "num_layers": 2, "num_attention_heads": 4,
              "num_kv_heads": 2, "intermediate_size": 512, "vocab_size": 1024,
              "max_context_length": 128, "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0, "head_dim": 64,
              "max_tokens": 1
            }"#;
        let resp_str = generate_json(req);
        let resp: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON");
        let error = resp["error"]
            .as_str()
            .expect("omitting both prompt sources must be an error, not a silent success");
        assert!(error.contains("exactly one"));
    }

    #[test]
    fn invalid_tokenizer_json_is_a_clear_error() {
        let req = tiny_text_config_json("Hi", "this is not valid tokenizer json", 1);
        let resp_str = generate_json(&req);
        let resp: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON");
        let error = resp["error"].as_str().expect("must be an error response");
        assert!(error.contains("tokenizer_json"));
    }

    #[test]
    fn prompt_text_encode_failure_is_reported_not_swallowed() {
        // `tokenizer_json` with a model type this crate's native backend
        // cannot build at all (rather than merely an unusual vocabulary)
        // must surface as an error through the same `run_inference` path
        // `prompt_text` uses, proving encode failures are propagated rather
        // than silently producing an empty/default prompt.
        const UNSUPPORTED_MODEL_JSON: &str = r##"{
            "model": { "type": "NotARealModelType" },
            "added_tokens": [],
            "pre_tokenizer": { "type": "ByteLevel" },
            "decoder": { "type": "ByteLevel" }
        }"##;
        let req = tiny_text_config_json("Hi", UNSUPPORTED_MODEL_JSON, 1);
        let resp_str = generate_json(&req);
        let resp: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON");
        assert!(
            !resp["error"].is_null(),
            "an unloadable tokenizer_json must be reported as an error, not silently \
             degrade to an empty prompt; got {resp}"
        );
    }
}
