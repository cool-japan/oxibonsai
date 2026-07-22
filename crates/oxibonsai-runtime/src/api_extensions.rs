//! Extended `/v1/chat/completions` handler.
//!
//! Adds support for tools (function calling), `n > 1` completions, response
//! format constraints (JSON mode / JSON Schema), stop sequences, and real SSE
//! streaming on top of the base server implementation.
//!
//! OpenAI `frequency_penalty` / `presence_penalty` and `logprobs` are now
//! honored for real (they were previously rejected with `400` because no
//! sampler/logits seam existed):
//!
//! - `frequency_penalty` / `presence_penalty` — validated to `[-2.0, 2.0]` and
//!   applied over the generated-token history via
//!   [`crate::sampling::PenaltyParams`] /
//!   [`crate::sampling::Sampler::sample_with_history`]. An all-zero pair is a
//!   no-op.
//! - `logprobs` — when requested, generation runs through
//!   [`InferenceEngine::generate_with_logprobs`], which captures the per-step
//!   logits and returns real per-token log probabilities plus `top_logprobs`
//!   alternatives. Because that variant has no per-call seed or params seam, a
//!   `logprobs` request honors the frequency/presence penalties but samples
//!   with the engine's ambient `temperature` / `top_p` and is not
//!   seed-reproducible (the seeded, params-honoring, non-logprobs path via
//!   [`InferenceEngine::generate_with_seed`] still is).
//!
//! `stream: true` is implemented as real token-by-token SSE (see
//! `extended_chat_completions_stream`), reusing the same
//! `generate_streaming_with_params` machinery the base `/v1/chat/completions`
//! endpoint uses in `server.rs`. It is only supported for the plain-text,
//! single-choice case: combining it with `tools`, `n > 1`, or a JSON-mode
//! `response_format` is rejected with `400 Bad Request` rather than silently
//! ignored, because each of those features needs the complete generated text
//! before it can be applied (tool-call parsing, multi-choice interleaving,
//! and JSON extraction/wrapping all operate on a finished string, not a
//! partial one).

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};

use crate::api_types::{
    ChoiceLogprobs, ExtendedChatRequest, ExtendedChatResponse, ExtendedChoice, UsageInfo,
};
use crate::engine::InferenceEngine;
use crate::sampling::SamplingParams;
use crate::server::{AppState, ChatMessage, MAX_OUTPUT_TOKENS};

// ── Extended handler ──────────────────────────────────────────────────────────

/// Maximum number of independent completions (`n`) the extended endpoint can
/// produce in a single request. Requests asking for more are rejected with
/// `400 Bad Request` rather than silently clamped.
pub const MAX_EXTENDED_N_CHOICES: usize = 4;

/// Build an OpenAI-compatible `400 Bad Request` JSON error response.
///
/// Delegates to the shared [`crate::http_error`] envelope so every route on the
/// server emits the identical `{"error": {message, type, param, code}}` shape.
fn bad_request(message: String, param: &str) -> axum::response::Response {
    crate::http_error::bad_request(message, param)
}

/// Handler for `POST /v1/chat/completions/extended`.
///
/// Supports all standard fields plus `tools`, `tool_choice`, `logprobs`,
/// `top_logprobs`, `response_format`, `n`, and `stop`. `frequency_penalty` /
/// `presence_penalty` are validated but rejected with `400` when non-zero
/// (see module docs); `n` is capped at [`MAX_EXTENDED_N_CHOICES`] and any
/// larger value is rejected with `400` rather than silently clamped.
pub async fn extended_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExtendedChatRequest>,
) -> impl IntoResponse {
    if req.max_tokens < 1 {
        return bad_request("max_tokens must be at least 1".to_string(), "max_tokens");
    }
    if req.max_tokens > MAX_OUTPUT_TOKENS {
        return bad_request(
            format!(
                "max_tokens {} exceeds the maximum of {MAX_OUTPUT_TOKENS}",
                req.max_tokens
            ),
            "max_tokens",
        );
    }
    let requested_n = req.n.unwrap_or(1);
    if !(1..=MAX_EXTENDED_N_CHOICES).contains(&requested_n) {
        return bad_request(
            format!("n must be between 1 and {MAX_EXTENDED_N_CHOICES}, got {requested_n}"),
            "n",
        );
    }
    let n = requested_n;
    // OpenAI frequency/presence penalties are now applied for real over the
    // generated-token history via the sampler's penalty seam (previously they
    // were rejected with `400` because no seam existed). They are validated to
    // the OpenAI `[-2.0, 2.0]` range and forwarded to the engine; an all-zero
    // `PenaltyParams` is a no-op.
    let frequency_penalty = req.frequency_penalty.unwrap_or(0.0);
    let presence_penalty = req.presence_penalty.unwrap_or(0.0);
    if !frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&frequency_penalty) {
        return bad_request(
            "frequency_penalty must be a finite number in the range [-2.0, 2.0]".to_string(),
            "frequency_penalty",
        );
    }
    if !presence_penalty.is_finite() || !(-2.0..=2.0).contains(&presence_penalty) {
        return bad_request(
            "presence_penalty must be a finite number in the range [-2.0, 2.0]".to_string(),
            "presence_penalty",
        );
    }
    let penalties = crate::sampling::PenaltyParams::new(frequency_penalty, presence_penalty);
    let max_tokens = req.max_tokens;
    let temperature = req.temperature.unwrap_or(0.7);
    let seed = req.seed.unwrap_or(42);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs_k = req.top_logprobs.unwrap_or(0).clamp(0, 20);
    let response_format = req.response_format.clone();
    let tools = req.tools.clone();
    let is_json_mode = response_format
        .as_ref()
        .map(|rf| rf.format_type == "json_object" || rf.format_type == "json_schema")
        .unwrap_or(false);

    // `stream: true` is only implemented for the plain-text, single-choice
    // case (see module docs): tool-call parsing, multi-choice interleaving,
    // and JSON-mode extraction/wrapping all need the complete generated text,
    // which isn't available mid-stream. Reject those combinations honestly
    // with `400` rather than silently ignoring `stream` (the previous
    // behavior) or silently ignoring the incompatible field.
    let stream = req.stream.unwrap_or(false);
    if stream {
        if tools.is_some() {
            return bad_request(
                "stream: true is not supported together with tools: tool-call parsing \
                 looks for a complete <tool_call>...</tool_call> block, which isn't \
                 available until streaming finishes; omit tools or set stream to false"
                    .to_string(),
                "stream",
            );
        }
        if n > 1 {
            return bad_request(
                format!(
                    "stream: true only supports n = 1: interleaving {n} streamed choices \
                     is not implemented; omit n (or set it to 1) or set stream to false"
                ),
                "stream",
            );
        }
        if is_json_mode {
            return bad_request(
                "stream: true is not supported together with a json_object/json_schema \
                 response_format: JSON-mode extraction/wrapping needs the complete \
                 generated text, which isn't available until streaming finishes; omit \
                 response_format or set stream to false"
                    .to_string(),
                "response_format",
            );
        }
    }

    // Build stop checker
    let stop_checker = match req.stop {
        Some(ref seqs) => StopChecker::new(seqs.as_slice().to_vec()),
        None => StopChecker::new(vec![]),
    };

    // Build prompt text from messages, neutralizing special-token markers in
    // user/system content when sanitization is enabled (finding `security-03`).
    let prompt_text = build_extended_prompt(&req.messages, state.sanitize_prompt());

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

    // Resolve the real loaded-model id once, for both the response `model`
    // field and the fingerprint input, instead of the previous hardcoded
    // "bonsai-8b" literal (mirrors the same fix already applied to the base
    // /v1/chat/completions and /v1/models handlers in server.rs).
    let model_id = state.model_info().descriptor().await.id;

    if stream {
        let stop_sequences = stop_checker.sequences.clone();
        return extended_chat_completions_stream(
            state,
            prompt_tokens,
            max_tokens,
            sampling_params,
            penalties,
            stop_sequences,
            model_id,
        )
        .await;
    }

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

    // Apply frequency/presence penalties for the whole batch, restoring the
    // engine's previous penalties before the lease is returned to the pool so
    // they cannot leak into the next request served by this replica.
    let prev_penalties = engine.penalties();
    engine.set_penalties(penalties);

    // Each entry is `(decoded_text, output_token_count, logprobs)`; the token
    // count is retained (rather than re-derived from whitespace-splitting the
    // decoded text later) so the finish-reason computation below can tell a
    // length-truncated run (`output_token_count == max_tokens`) apart from one
    // that stopped early on EOS or a stop sequence. `logprobs` is `Some` only
    // when the client requested them.
    type RawCompletion = (
        String,
        usize,
        Option<Vec<crate::api_types::LogprobsContent>>,
    );
    let raw_completions: Vec<RawCompletion> = {
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            engine.reset();

            // Logprobs and seeded generation are mutually exclusive at this
            // layer: the logits-capturing variant honors the engine's penalties
            // (set above) but has no per-call seed seam, so a `logprobs`
            // request is not seed-reproducible (documented on the function).
            // The far more common non-logprobs path stays fully seeded.
            let (output_tokens, logprobs) = if want_logprobs {
                let id_to_token = |id: u32| -> String {
                    match state.tokenizer() {
                        Some(tok) => tok.decode(&[id]).unwrap_or_else(|_| format!("<{id}>")),
                        None => format!("<{id}>"),
                    }
                };
                match engine.generate_with_logprobs(
                    &prompt_tokens,
                    max_tokens,
                    top_logprobs_k,
                    &id_to_token,
                ) {
                    Ok((toks, lp)) => (toks, Some(lp)),
                    Err(e) => {
                        tracing::error!(error = %e, "logprobs generation failed for completion {i}");
                        engine.set_penalties(prev_penalties);
                        return (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": "generation failed"})),
                        )
                            .into_response();
                    }
                }
            } else {
                let run_seed = seed.wrapping_add(i as u64);
                match engine.generate_with_seed(
                    &prompt_tokens,
                    max_tokens,
                    run_seed,
                    &sampling_params,
                ) {
                    Ok(toks) => (toks, None),
                    Err(e) => {
                        tracing::error!(error = %e, "generation failed for completion {i}");
                        engine.set_penalties(prev_penalties);
                        return (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": "generation failed"})),
                        )
                            .into_response();
                    }
                }
            };
            let output_len = output_tokens.len();

            // Decode
            let text = if let Some(tok) = state.tokenizer() {
                tok.decode(&output_tokens)
                    .unwrap_or_else(|_| format!("{output_tokens:?}"))
            } else {
                format!("{output_tokens:?}")
            };

            results.push((text, output_len, logprobs));
        }
        results
    };

    // Restore the engine's penalties now that the batch is complete.
    engine.set_penalties(prev_penalties);

    // Apply stop sequences and response format enforcement (`is_json_mode`
    // was already computed above, before the stream/tools/n compatibility
    // checks).
    let json_enforcer = JsonModeEnforcer::new();

    let total_completion_tokens: usize;
    let choices: Vec<ExtendedChoice> = {
        let mut comp_tokens = 0usize;
        let choices_out: Vec<ExtendedChoice> = raw_completions
            .into_iter()
            .enumerate()
            .map(|(idx, (raw_text, output_len, run_logprobs))| {
                let (truncated, hit_stop) = stop_checker.truncate_at_stop(&raw_text);

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

                let finish_reason = determine_extended_finish_reason(
                    tool_calls.is_some(),
                    hit_stop,
                    output_len,
                    max_tokens,
                );

                // Real per-token logprobs, captured during generation by the
                // engine's logits-capturing variant when the client requested
                // them (`logprobs: true`). `content: Some([...])` carries one
                // entry per generated token, each with the chosen token's log
                // probability and its `top_logprobs` alternatives.
                let logprobs: Option<ChoiceLogprobs> = run_logprobs.map(|content| ChoiceLogprobs {
                    content: Some(content),
                });

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

    // Build system fingerprint from the real loaded-model id (resolved above).
    let system_fingerprint = Some(crate::api_types::fingerprint_from_config(&model_id));

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let response = ExtendedChatResponse {
        id: format!("chatcmpl-ext-{}", rand_ext_id()),
        object: "chat.completion".to_string(),
        created,
        model: model_id,
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

// ── Streaming (SSE) ───────────────────────────────────────────────────────────

/// One chunk of an extended-endpoint SSE stream (OpenAI `chat.completion.chunk`
/// shape). Structurally identical to `server.rs`'s private `ChatCompletionChunk`
/// but kept as its own type here since that one isn't `pub`.
#[derive(Debug, serde::Serialize)]
struct ExtendedChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ExtendedChunkChoice>,
}

#[derive(Debug, serde::Serialize)]
struct ExtendedChunkChoice {
    index: usize,
    delta: ExtendedChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct ExtendedChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

fn extended_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    delta: ExtendedChunkDelta,
    finish_reason: Option<String>,
) -> String {
    let chunk = ExtendedChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ExtendedChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// Real SSE streaming for `POST /v1/chat/completions/extended`.
///
/// Only reachable for the plain-text, single-choice, non-JSON-mode case (see
/// [`extended_chat_completions`]'s compatibility checks); `tools`, `n > 1`,
/// and JSON-mode `response_format` are all rejected with `400` before this
/// function is ever called.
///
/// Reuses [`InferenceEngine::generate_streaming_with_params`] — the same
/// primitive the base `/v1/chat/completions` endpoint uses for streaming in
/// `server.rs` — and additionally applies client-supplied `stop` sequences by
/// watching the accumulating decoded text and suppressing further chunks once
/// a stop sequence is seen. **Known limitation**: because SSE chunks already
/// sent to the client cannot be retracted, a stop sequence that straddles a
/// chunk boundary (part of it decoded in an earlier chunk, the rest in a
/// later one) may leak the leading fragment before generation is recognized
/// as stopped; this mirrors the inherent limits of incremental streaming
/// truncation and does not affect the non-streaming endpoint, which truncates
/// the complete text after the fact.
///
/// **Known limitation**: unlike the non-streaming path (which seeds a fresh
/// `Sampler` per run via `generate_with_seed`), `generate_streaming_with_params`
/// has no seam to apply a request-supplied `seed` — it continues the engine's
/// ambient PRNG state, same as the base (non-extended) streaming endpoint in
/// `server.rs`. `stream: true` requests are therefore not seed-reproducible;
/// only non-streaming requests are.
#[allow(clippy::too_many_arguments)]
async fn extended_chat_completions_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling_params: SamplingParams,
    penalties: crate::sampling::PenaltyParams,
    stop_sequences: Vec<String>,
    model_id: String,
) -> axum::response::Response {
    let completion_id = format!("chatcmpl-ext-{}", rand_ext_id());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut lease = match state.acquire_engine().await {
        Ok(lease) => lease,
        Err(e) => {
            tracing::error!(error = %e, "engine pool acquire failed");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "engine pool unavailable"})),
            )
                .into_response();
        }
    };

    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
    let (finish_tx, finish_rx) = tokio::sync::mpsc::unbounded_channel::<usize>();

    // Run generation on a blocking thread, exactly like the base endpoint's
    // `chat_completions_stream` in server.rs. The lease (and thus `token_tx`)
    // drops at the end of the closure, which both returns the engine to the
    // pool and closes the token channel so `content_stream` below terminates.
    tokio::task::spawn_blocking(move || {
        lease.reset();
        // Apply frequency/presence penalties for this run, restoring the
        // engine's previous penalties before the lease drops so they don't leak
        // to the next request served by this pool replica.
        let prev_penalties = lease.penalties();
        lease.set_penalties(penalties);
        let result = lease.generate_streaming_with_params(
            &prompt_tokens,
            max_tokens,
            &sampling_params,
            &token_tx,
        );
        lease.set_penalties(prev_penalties);
        // Send the real generated-token count for finish_reason, even on
        // error (0 generated is an honest "stop" rather than "length").
        let _ = finish_tx.send(result.unwrap_or(0));
    });

    let hit_stop = Arc::new(AtomicBool::new(false));
    let hit_stop_for_content = Arc::clone(&hit_stop);

    let mut decode_state = state.tokenizer().map(|t| t.new_decode_stream(true));
    let state_for_content = Arc::clone(&state);
    let mut accumulated = String::new();

    let id_for_content = completion_id.clone();
    let model_for_content = model_id.clone();
    let token_stream = UnboundedReceiverStream::new(token_rx);

    let content_stream = token_stream.filter_map(move |token_id| {
        if hit_stop_for_content.load(Ordering::Relaxed) {
            return None;
        }

        let text = match (state_for_content.tokenizer(), decode_state.as_mut()) {
            (Some(tok), Some(dec_state)) => match tok.step_decode(dec_state, token_id) {
                Ok(Some(txt)) => txt,
                Ok(None) => return None,
                Err(_) => format!("[{token_id}]"),
            },
            _ => format!("[{token_id}]"),
        };
        if text.is_empty() {
            return None;
        }

        let chunk_start = accumulated.len();
        accumulated.push_str(&text);

        let mut stop_pos: Option<usize> = None;
        for seq in &stop_sequences {
            if seq.is_empty() {
                continue;
            }
            if let Some(pos) = accumulated.find(seq.as_str()) {
                stop_pos = Some(stop_pos.map_or(pos, |prev| prev.min(pos)));
            }
        }

        let visible_text = match stop_pos {
            Some(pos) => {
                hit_stop_for_content.store(true, Ordering::Relaxed);
                let visible_end = pos.max(chunk_start).min(accumulated.len());
                accumulated[chunk_start..visible_end].to_string()
            }
            None => text,
        };

        if visible_text.is_empty() {
            return None;
        }

        Some(extended_chunk_json(
            &id_for_content,
            created,
            &model_for_content,
            ExtendedChunkDelta {
                role: None,
                content: Some(visible_text),
            },
            None,
        ))
    });

    let hit_stop_for_finish = Arc::clone(&hit_stop);
    let id_for_finish = completion_id.clone();
    let model_for_finish = model_id.clone();
    let finish_stream = UnboundedReceiverStream::new(finish_rx).map(move |generated| {
        // Mirrors `determine_extended_finish_reason`'s precedence for the
        // (tool-call-free) streaming case: an explicit stop-sequence match
        // wins over a length-based truncation.
        let finish_reason = if hit_stop_for_finish.load(Ordering::Relaxed) || generated < max_tokens
        {
            "stop"
        } else {
            "length"
        };
        extended_chunk_json(
            &id_for_finish,
            created,
            &model_for_finish,
            ExtendedChunkDelta {
                role: None,
                content: None,
            },
            Some(finish_reason.to_string()),
        )
    });

    let role_event = extended_chunk_json(
        &completion_id,
        created,
        &model_id,
        ExtendedChunkDelta {
            role: Some("assistant".to_string()),
            content: None,
        },
        None,
    );

    let full_stream = tokio_stream::once(role_event)
        .chain(content_stream)
        .chain(finish_stream)
        .map(|json_str| -> Result<Event, Infallible> { Ok(Event::default().data(json_str)) })
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));

    Sse::new(full_stream).into_response()
}

/// Build a prompt string from a slice of chat messages (ChatML format).
///
/// Messages with `content = None` are skipped (they represent tool-call turns).
/// When `sanitize` is `true`, each message's content is passed through
/// [`crate::server::neutralize_special_markers`] before concatenation so
/// client-supplied text cannot forge fake role/turn boundaries once the merged
/// prompt is tokenized (finding `security-03`).
fn build_extended_prompt(messages: &[ChatMessage], sanitize: bool) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let raw = match msg.content.as_deref() {
            Some(t) => t,
            None => continue,
        };
        let text = if sanitize {
            crate::server::neutralize_special_markers(raw)
        } else {
            raw.to_string()
        };
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(&text);
                prompt.push_str("<|im_end|>\n");
            }
            "user" => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&text);
                prompt.push_str("<|im_end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(&text);
                prompt.push_str("<|im_end|>\n");
            }
            _ => {
                prompt.push_str(&text);
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

// ── Finish reason ─────────────────────────────────────────────────────────────

/// Determine the OpenAI-compatible `finish_reason` for one extended-chat
/// choice.
///
/// Priority order (matches OpenAI semantics):
/// 1. `"tool_calls"` — the model's output was parsed as a tool call.
/// 2. `"stop"` — an explicit client-supplied stop sequence was matched.
/// 3. `"length"` — generation exhausted `max_tokens` without hitting EOS or a
///    stop sequence. This relies on `engine::generate`'s decode loop, which
///    only ever returns exactly `max_tokens` output tokens when it never
///    broke early on `EOS_TOKEN_ID` (see `engine.rs::generate`), so
///    `output_len >= max_tokens` reliably signals truncation. Mirrors
///    `completions.rs::determine_finish_reason`.
/// 4. `"stop"` — otherwise the run ended naturally on EOS.
fn determine_extended_finish_reason(
    has_tool_calls: bool,
    hit_stop: bool,
    output_len: usize,
    max_tokens: usize,
) -> String {
    if has_tool_calls {
        "tool_calls".to_string()
    } else if hit_stop {
        "stop".to_string()
    } else if output_len >= max_tokens {
        "length".to_string()
    } else {
        "stop".to_string()
    }
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

    // ── determine_extended_finish_reason (finding 28 regression) ─────────────

    #[test]
    fn finish_reason_tool_calls_takes_priority() {
        // Even if the run also happened to exhaust max_tokens, a parsed tool
        // call must win.
        assert_eq!(
            determine_extended_finish_reason(true, false, 10, 10),
            "tool_calls"
        );
        assert_eq!(
            determine_extended_finish_reason(true, true, 10, 10),
            "tool_calls"
        );
    }

    #[test]
    fn finish_reason_stop_sequence_wins_over_length() {
        // A stop sequence match must report "stop" even when output_len
        // happens to equal max_tokens.
        assert_eq!(determine_extended_finish_reason(false, true, 8, 8), "stop");
    }

    #[test]
    fn finish_reason_length_when_truncated() {
        // Regression for finding 28: previously this always returned "stop"
        // regardless of truncation.
        assert_eq!(
            determine_extended_finish_reason(false, false, 8, 8),
            "length"
        );
        assert_eq!(
            determine_extended_finish_reason(false, false, 10, 8),
            "length"
        );
    }

    #[test]
    fn finish_reason_stop_on_natural_eos() {
        assert_eq!(determine_extended_finish_reason(false, false, 3, 8), "stop");
    }
}
