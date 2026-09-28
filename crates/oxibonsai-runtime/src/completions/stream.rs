//! `POST /v1/completions` SSE streaming (`stream: true`).
//!
//! The legacy endpoint streams real SSE — `text_completion` chunks, one
//! `finish_reason` per prompt, optional `logprobs`, a `usage` chunk when
//! `stream_options.include_usage` is set, then `data: [DONE]` — through the
//! chat endpoint's SSE machinery ([`crate::server::sse::sse_response`]).
//!
//! Split out of `completions.rs` (`mod stream;`, `use super::*;`) to keep
//! both files under the workspace's 2000-line ceiling.
//!
//! # Batches
//!
//! A batched prompt streams too (OpenAI semantics): one SSE stream carries
//! every prompt's chunks, each with `choices[0].index` = its prompt's
//! index. The prompts are generated in order on the one leased replica
//! (reset between prompts): prompt `i`'s echo chunk, text chunks and finish
//! chunk all precede prompt `i + 1`'s first chunk. The usage chunk that
//! follows the last finish chunk aggregates the whole batch. Prompt `i` is
//! seeded with `seed + i`, exactly like the non-streaming path, so each
//! index's streamed text equals that index's non-streamed text.
//!
//! # Per prompt
//!
//! * **Sampling** — the request's resolved params, penalties, `min_p` and
//!   seed run each generation ([`RequestSampling`]); the replica's own
//!   sampler comes back afterwards.
//! * **Stop sequences** — a byte-accurate hold-back window
//!   ([`crate::pipeline::StopSequenceMatcher`]) so a match split across two
//!   decoded chunks never leaks a prefix of it; a match also CANCELS that
//!   prompt's generation, and the prompt reports `finish_reason: "stop"`.
//! * **`logprobs`** — a dedicated per-token decode + logit-capture loop
//!   ([`generate_streaming_logprobs_chunks`]) over the engine's own
//!   `prefill_for_generate` / `sampler` / `decode_step` seams, holding back
//!   whole (piece, logprobs) pairs so each chunk's parallel arrays describe
//!   exactly its text.
//! * A token the decoder cannot render is never shown as id syntax
//!   ([`PieceDecoder`]); without a tokenizer nothing is shown.
//!
//! Every stream answers with an `x-request-id` header and observes
//! `request_duration_seconds` when it ends (completion, a client
//! disconnect, or the SSE deadline). [`StreamCancelGuard`] cancels the
//! running generation — and every prompt not yet started — however the
//! stream ends; cancellation is a no-op once generation already finished.

use super::*;

use crate::engine::InferenceEngine;
use crate::engine_control::CancellationToken;
use crate::pipeline::{StopMatch, StopSequenceMatcher};
use crate::server::sse::sse_response;
use crate::tokenizer_bridge::chat_render::{DecodedPiece, PieceDecoder};
use crate::tokenizer_bridge::TokenizerBridge;

// ── Cancellation and lifetime guards ────────────────────────────────────────

/// The cancellation handle of whichever prompt's generation is running,
/// shared between the blocking generation loop (which arms one per prompt)
/// and the SSE stream (which cancels it when the stream goes away).
#[derive(Clone, Default)]
struct GenerationSlot(Arc<std::sync::Mutex<SlotState>>);

#[derive(Default)]
struct SlotState {
    /// The running prompt's token.
    current: Option<CancellationToken>,
    /// The stream is gone: no further prompt may start.
    abandoned: bool,
}

impl GenerationSlot {
    /// Install the next prompt's token. `false` (with the token cancelled)
    /// once the stream has been abandoned.
    fn start(&self, token: &CancellationToken) -> bool {
        let Ok(mut state) = self.0.lock() else {
            token.cancel();
            return false;
        };
        if state.abandoned {
            token.cancel();
            return false;
        }
        state.current = Some(token.clone());
        true
    }

    /// The stream is gone: cancel the running prompt and start no other.
    fn abandon(&self) {
        if let Ok(mut state) = self.0.lock() {
            state.abandoned = true;
            if let Some(token) = state.current.as_ref() {
                token.cancel();
            }
        }
    }
}

/// Abandons the [`GenerationSlot`] on drop: an early client disconnect, or
/// `sse::sse_response`'s own per-request deadline, drops the stream without
/// the generation having finished — this stops it (and every prompt after
/// it) instead of letting it decode to `max_tokens` on a lease the client
/// can no longer see. Harmless after a normal finish.
struct AbandonOnDrop(GenerationSlot);

impl Drop for AbandonOnDrop {
    fn drop(&mut self) {
        self.0.abandon();
    }
}

/// Observes `request_duration_seconds` when dropped — at the true end of a
/// streaming response (completion, a client disconnect, or the SSE
/// deadline), the streaming counterpart of the non-streaming path's
/// observation at the end of `create_completion`.
struct ObserveDurationOnDrop {
    metrics: Arc<crate::metrics::InferenceMetrics>,
    request_start: std::time::Instant,
}

impl Drop for ObserveDurationOnDrop {
    fn drop(&mut self) {
        self.metrics
            .request_duration_seconds
            .observe(self.request_start.elapsed().as_secs_f64());
    }
}

/// Wraps the assembled SSE stream so the running generation is cancelled
/// the instant the stream itself is dropped (see [`AbandonOnDrop`]) and the
/// request's duration is observed at that same moment.
struct StreamCancelGuard<S> {
    inner: S,
    _abandon: AbandonOnDrop,
    _duration: ObserveDurationOnDrop,
}

impl<S: tokio_stream::Stream + Unpin> tokio_stream::Stream for StreamCancelGuard<S> {
    type Item = S::Item;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        std::pin::Pin::new(&mut this.inner).poll_next(cx)
    }
}

// ── Stop-sequence hold-back ─────────────────────────────────────────────────

/// Per-prompt stop-sequence hold-back state: `push_and_release` returns
/// what is safe to send, `take_remaining` the rest once generation ended
/// without a match.
#[derive(Default)]
struct CompletionStopTracker {
    accumulated: String,
    released_len: usize,
    stopped: bool,
}

impl CompletionStopTracker {
    /// Append `piece` and return whatever portion is now safe to release:
    /// everything up to (but not including) a stop-sequence match, or —
    /// when no match is found — everything except a trailing hold-back
    /// window that could still grow into one.
    fn push_and_release(&mut self, piece: &str, matcher: &StopSequenceMatcher) -> String {
        if self.stopped {
            return String::new();
        }
        self.accumulated.push_str(piece);
        if matcher.is_empty() {
            let safe_len = self.accumulated.len();
            if safe_len > self.released_len {
                let out = self.accumulated[self.released_len..safe_len].to_string();
                self.released_len = safe_len;
                return out;
            }
            return String::new();
        }
        match matcher.check(&self.accumulated) {
            StopMatch::Found { start, .. } => {
                self.stopped = true;
                if start > self.released_len {
                    let out = self.accumulated[self.released_len..start].to_string();
                    self.released_len = self.accumulated.len();
                    out
                } else {
                    self.released_len = self.accumulated.len();
                    String::new()
                }
            }
            StopMatch::None => {
                let hold = matcher.hold_back_len(&self.accumulated);
                let safe_len = self.accumulated.len().saturating_sub(hold);
                if safe_len > self.released_len {
                    let out = self.accumulated[self.released_len..safe_len].to_string();
                    self.released_len = safe_len;
                    out
                } else {
                    String::new()
                }
            }
        }
    }

    /// Release everything still held back once generation has ended
    /// naturally (EOS / `max_tokens`) without ever matching a stop
    /// sequence. A no-op once a match already occurred or nothing remains.
    fn take_remaining(&mut self) -> String {
        if self.stopped || self.released_len >= self.accumulated.len() {
            return String::new();
        }
        let out = self.accumulated[self.released_len..].to_string();
        self.released_len = self.accumulated.len();
        out
    }
}

/// Per-prompt stop-sequence hold-back state for the `logprobs` + `stream`
/// combination: the same byte-level "never emit a byte that could still be
/// the unfinished start of a match" guarantee [`CompletionStopTracker`]
/// gives the plain streaming path, but holding back whole pending
/// (piece, logprobs-entries) pairs rather than raw bytes, so a released
/// chunk's `tokens`/`token_logprobs`/`top_logprobs`/`text_offset` arrays
/// stay genuinely parallel to the text it carries — a token's own entry is
/// only ever released together with that token's own complete text, never
/// split.
#[derive(Default)]
struct LogprobsStopTracker {
    /// Exactly the concatenation of every `(piece, _)` in `pending`,
    /// maintained as an invariant by [`Self::release_up_to`]'s trim.
    accumulated: String,
    /// Not-yet-released `(decoded piece, logprobs entries)` pairs, oldest
    /// first. More than one entry per piece happens when a piece only
    /// became decodable after several UTF-8-incomplete tokens were
    /// buffered via [`Self::buffer_pending_entry`].
    pending: Vec<(String, Vec<crate::api_types::LogprobsContent>)>,
    /// Logprobs entries for tokens that have been sampled (and thus have a
    /// real entry) but whose own text is not yet decodable on its own
    /// (`step_decode` returned `Ok(None)`) — attached to the next piece
    /// that DOES complete.
    utf8_pending_entries: Vec<crate::api_types::LogprobsContent>,
    stopped: bool,
}

impl LogprobsStopTracker {
    /// Remember a logprobs entry whose token contributed to a still-
    /// incomplete UTF-8 sequence. See the field doc.
    fn buffer_pending_entry(&mut self, entry: crate::api_types::LogprobsContent) {
        self.utf8_pending_entries.push(entry);
    }

    /// Feed one COMPLETE decoded piece and the logprobs entry for the
    /// token that just completed it (plus every entry buffered by
    /// [`Self::buffer_pending_entry`] since the last completed piece).
    /// Returns the `(text, entries)` now safe to release as one SSE
    /// chunk, if any.
    ///
    /// An empty `piece` is a token with no text to show at all (no
    /// tokenizer is attached): it contributes nothing to the text, so it
    /// can never complete a stop sequence, but its logprobs entry is still
    /// released — in order, as an empty-text chunk — rather than lost.
    fn push(
        &mut self,
        piece: String,
        entry: crate::api_types::LogprobsContent,
        matcher: &StopSequenceMatcher,
    ) -> Option<(String, Vec<crate::api_types::LogprobsContent>)> {
        if self.stopped {
            return None;
        }
        let mut entries = std::mem::take(&mut self.utf8_pending_entries);
        entries.push(entry);

        self.accumulated.push_str(&piece);
        self.pending.push((piece, entries));

        if matcher.is_empty() {
            return self.release_up_to(self.accumulated.len());
        }
        let safe_len = match matcher.check(&self.accumulated) {
            StopMatch::Found { start, .. } => {
                self.stopped = true;
                start
            }
            StopMatch::None => {
                let hold = matcher.hold_back_len(&self.accumulated);
                self.accumulated.len().saturating_sub(hold)
            }
        };
        self.release_up_to(safe_len)
    }

    /// Release every whole pending piece whose contribution ends at or
    /// before the byte offset `safe_len` (measured into `self.accumulated`
    /// as it stood at the start of the call that computed `safe_len`).
    fn release_up_to(
        &mut self,
        safe_len: usize,
    ) -> Option<(String, Vec<crate::api_types::LogprobsContent>)> {
        let mut cursor = 0usize;
        let mut split_at = 0usize;
        for (piece, _) in &self.pending {
            let end = cursor + piece.len();
            if end > safe_len {
                break;
            }
            cursor = end;
            split_at += 1;
        }
        if split_at == 0 {
            return None;
        }
        let released: Vec<(String, Vec<crate::api_types::LogprobsContent>)> =
            self.pending.drain(..split_at).collect();
        let mut text = String::with_capacity(cursor);
        let mut entries = Vec::new();
        for (piece, piece_entries) in released {
            text.push_str(&piece);
            entries.extend(piece_entries);
        }
        self.accumulated = self.accumulated[cursor..].to_string();
        Some((text, entries))
    }

    /// Release everything still pending once generation ends naturally
    /// (EOS / `max_tokens` / cancellation) without ever matching a stop
    /// sequence. A no-op once a match already occurred or nothing remains.
    fn take_remaining(&mut self) -> Option<(String, Vec<crate::api_types::LogprobsContent>)> {
        if self.stopped || self.pending.is_empty() {
            return None;
        }
        self.release_up_to(self.accumulated.len())
    }
}

/// One `(text, logprobs entries)` pair of the `logprobs` streaming loop.
type LogprobsChunk = (String, Vec<crate::api_types::LogprobsContent>);

/// Runs entirely on the blocking pool: prefill, then per step — cancel
/// check, sample, EOS check, capture logprobs, UTF-8-safe piece decode,
/// stop-sequence hold-back — sending each safe-to-release `(text, entries)`
/// pair through `tx` as it becomes available.
///
/// Reimplements `InferenceEngine::generate_with_logprobs`'s exact per-step
/// loop (prefill_for_generate → sample_with_history → EOS check →
/// compute_logprobs → decode_step) using the same engine seams its
/// non-streaming counterpart uses, adding the streaming decode +
/// stop-sequence pairing the collect-everything method has no equivalent
/// for.
///
/// Returns the number of tokens actually generated and whether a stop
/// sequence matched.
fn generate_streaming_logprobs_chunks(
    engine: &mut InferenceEngine<'_>,
    tokenizer: Option<&TokenizerBridge>,
    prompt_tokens: &[u32],
    max_tokens: usize,
    top_k: usize,
    stop_matcher: &StopSequenceMatcher,
    tx: &tokio::sync::mpsc::UnboundedSender<LogprobsChunk>,
) -> RuntimeResult<(usize, bool)> {
    if prompt_tokens.is_empty() {
        return Ok((0, false));
    }
    let top_k = top_k.min(20);
    let Some(mut last_logits) = engine.prefill_for_generate(prompt_tokens)? else {
        return Ok((0, false));
    };
    let id_to_token = |id: u32| -> String {
        match tokenizer {
            Some(tok) => tok.decode(&[id]).unwrap_or_else(|_| format!("<{id}>")),
            None => format!("<{id}>"),
        }
    };
    let mut piece_decoder = PieceDecoder::new(tokenizer);
    let mut output_tokens: Vec<u32> = Vec::new();
    let mut tracker = LogprobsStopTracker::default();

    for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
        if engine.is_cancelled() {
            tracing::debug!(pos, "streaming logprobs generation cancelled");
            break;
        }
        let next_token = engine
            .sampler
            .sample_with_history(&last_logits, &output_tokens)?;
        if engine.is_eos(next_token) {
            tracing::debug!(pos, "EOS token generated (streaming logprobs)");
            break;
        }
        let entry =
            crate::api_types::compute_logprobs(&last_logits, next_token, top_k, &id_to_token);
        output_tokens.push(next_token);

        // A token with no text of its own yet (a partial UTF-8 sequence)
        // hands its logprobs entry to the next completed piece; a token with
        // no text ever (no tokenizer) releases its entry with empty text —
        // never `"[<id>]"` (`PieceDecoder`).
        let piece = match piece_decoder.next(tokenizer, next_token) {
            DecodedPiece::Text(text) => Some(text),
            DecodedPiece::Pending => None,
            DecodedPiece::Omitted => Some(String::new()),
        };

        match piece {
            Some(piece) => {
                if let Some((text, entries)) = tracker.push(piece, entry, stop_matcher) {
                    if tx.send((text, entries)).is_err() {
                        tracing::debug!(pos, "receiver dropped, stopping generation");
                        break;
                    }
                }
            }
            None => tracker.buffer_pending_entry(entry),
        }

        if tracker.stopped {
            break;
        }

        last_logits = engine.decode_step(next_token, pos)?;
    }
    if let Some((text, entries)) = tracker.take_remaining() {
        let _ = tx.send((text, entries));
    }
    Ok((output_tokens.len(), tracker.stopped))
}

// ── Chunk shapes ────────────────────────────────────────────────────────────

/// A `text_completion` SSE chunk (OpenAI-compatible).
#[derive(Debug, Serialize)]
struct CompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChunkChoice>,
    /// `None` → omitted entirely (a request that did not ask for usage);
    /// `Some(Value::Null)` → `"usage": null` on every non-final chunk once
    /// `stream_options.include_usage` is set; `Some(object)` → the final
    /// usage-carrying chunk. The chat endpoint's chunks follow the same
    /// contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<serde_json::Value>,
}

/// One choice in a [`CompletionChunk`]; `index` is the prompt's position in
/// the request's batch (`0` for a single prompt).
#[derive(Debug, Serialize)]
struct CompletionChunkChoice {
    text: String,
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<serde_json::Value>,
    finish_reason: Option<String>,
}

/// The identity every chunk of one stream shares.
struct ChunkHeader {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
}

impl ChunkHeader {
    fn chunk(&self, choice: CompletionChunkChoice) -> String {
        let chunk = CompletionChunk {
            id: self.id.clone(),
            object: "text_completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![choice],
            usage: self.include_usage.then_some(serde_json::Value::Null),
        };
        serde_json::to_string(&chunk).unwrap_or_default()
    }

    /// A content chunk for prompt `index`, `finish_reason: null`.
    fn text(&self, index: usize, text: String) -> String {
        self.chunk(CompletionChunkChoice {
            text,
            index,
            logprobs: None,
            finish_reason: None,
        })
    }

    /// Prompt `index`'s final chunk: empty `text`, a real `finish_reason`.
    fn finish(&self, index: usize, finish_reason: &str) -> String {
        self.chunk(CompletionChunkChoice {
            text: String::new(),
            index,
            logprobs: None,
            finish_reason: Some(finish_reason.to_string()),
        })
    }

    /// A chunk carrying `text` and its own `logprobs` object — `entries`'
    /// parallel arrays describe exactly `text` (never more, never less; see
    /// [`LogprobsStopTracker`]).
    fn logprobs(
        &self,
        index: usize,
        text: String,
        entries: &[crate::api_types::LogprobsContent],
        base_offset: usize,
    ) -> String {
        // Every entry's own text is fully included (`LogprobsStopTracker`
        // only ever releases whole pieces), so a cursor of `usize::MAX`
        // guarantees `build_completion_logprobs` never drops one.
        let logprobs = build_completion_logprobs(entries, usize::MAX, base_offset);
        self.chunk(CompletionChunkChoice {
            text,
            index,
            logprobs: serde_json::to_value(logprobs).ok(),
            finish_reason: None,
        })
    }

    /// The final usage-carrying chunk (only when `include_usage` was set),
    /// aggregated over every prompt of the request.
    fn usage(&self, prompt_tokens: usize, completion_tokens: usize) -> String {
        let chunk = CompletionChunk {
            id: self.id.clone(),
            object: "text_completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: Vec::new(),
            usage: Some(serde_json::json!({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            })),
        };
        serde_json::to_string(&chunk).unwrap_or_default()
    }
}

// ── The stream ───────────────────────────────────────────────────────────────

/// A validated streaming request, tokenized by [`create_completion`].
pub(super) struct StreamRequest {
    pub(super) validated: ValidatedRequest,
    /// One token list per prompt, in batch order.
    pub(super) prompt_token_batches: Vec<Vec<u32>>,
    /// Echoed as the stream's `x-request-id` header.
    pub(super) request_id: RequestId,
    /// The whole request's start instant: `request_duration_seconds` is
    /// observed against it when the stream ends.
    pub(super) request_start: std::time::Instant,
    /// Moved into the generation task, so the `active_requests` gauge
    /// decrements when the generation actually finishes, not the moment
    /// this module returns the initial SSE response object.
    pub(super) active_guard: ActiveRequestGuard,
}

/// What one prompt's generation streams.
enum PromptBody {
    /// Generated token ids (the plain path decodes and stop-tracks them).
    Tokens(tokio::sync::mpsc::UnboundedReceiver<u32>),
    /// Already stop-tracked `(text, logprobs)` pairs (the `logprobs` path).
    Logprobs(tokio::sync::mpsc::UnboundedReceiver<LogprobsChunk>),
}

/// One prompt's generation, handed from the blocking loop to the stream as
/// it starts.
struct PromptStream {
    index: usize,
    prompt_tokens: usize,
    /// Cancels this prompt's generation (a stop-sequence match).
    cancel: CancellationToken,
    body: PromptBody,
    /// `Ok((generated, stopped))` — `stopped` as the `logprobs` loop saw it
    /// (the plain path decides it on the receiving side) — or the error's
    /// message.
    outcome: tokio::sync::oneshot::Receiver<Result<(usize, bool), String>>,
}

/// Handle a validated `stream: true` request — one prompt or a batch — end
/// to end.
pub(super) async fn stream_completion(
    state: Arc<AppState>,
    req: StreamRequest,
) -> Result<Response, ApiError> {
    let StreamRequest {
        validated,
        prompt_token_batches,
        request_id,
        request_start,
        active_guard,
    } = req;
    let duration = ObserveDurationOnDrop {
        metrics: Arc::clone(state.metrics()),
        request_start,
    };

    // Resolved BEFORE acquiring the engine lease below: on a pool with no
    // replica free besides the one this request is about to hold,
    // `ServedModelInfo::descriptor` acquires its own, *separate* lease from
    // the same pool on its first (cache-filling) call — calling it while
    // already holding this request's lease would self-deadlock a
    // single-replica pool.
    let model_id = state.model_info().descriptor().await.id;

    let lease = state.acquire_engine().await.map_err(|e| {
        tracing::error!(error = %e, "engine pool acquire failed");
        state.metrics().errors_total.inc();
        ApiError::service_unavailable(format!("no inference engine replica is available: {e}"))
            .with_code("engine_unavailable")
    })?;

    // Prompt `i`'s sampling configuration, resolved against the replica's
    // own ambient parameters.
    let samplings: Vec<RequestSampling> = (0..prompt_token_batches.len())
        .map(|index| validated.sampling(lease.sampling_params(), index))
        .collect();
    let max_tokens = validated.max_tokens;
    let logprobs_top_k = validated.logprobs_top_k;
    let stop_matcher = Arc::new(StopSequenceMatcher::new(validated.stop_checker.sequences()));

    let slot = GenerationSlot::default();
    let (prompt_tx, prompt_rx) = tokio::sync::mpsc::unbounded_channel::<PromptStream>();
    let slot_for_task = slot.clone();
    let matcher_for_task = Arc::clone(&stop_matcher);
    let state_for_task = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let _active_guard = active_guard;
        let mut lease = lease;
        for (index, (prompt_tokens, sampling)) in
            prompt_token_batches.iter().zip(&samplings).enumerate()
        {
            lease.reset();
            let cancel = lease.arm_cancellation();
            if !slot_for_task.start(&cancel) {
                break;
            }
            let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel();
            let result = match logprobs_top_k {
                None => {
                    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
                    let prompt = PromptStream {
                        index,
                        prompt_tokens: prompt_tokens.len(),
                        cancel,
                        body: PromptBody::Tokens(token_rx),
                        outcome: outcome_rx,
                    };
                    if prompt_tx.send(prompt).is_err() {
                        break;
                    }
                    sampling
                        .run(&mut lease, |engine| {
                            engine.generate_streaming(prompt_tokens, max_tokens, &token_tx)
                        })
                        .map(|generated| (generated, false))
                }
                Some(top_k) => {
                    let (chunk_tx, chunk_rx) =
                        tokio::sync::mpsc::unbounded_channel::<LogprobsChunk>();
                    let prompt = PromptStream {
                        index,
                        prompt_tokens: prompt_tokens.len(),
                        cancel,
                        body: PromptBody::Logprobs(chunk_rx),
                        outcome: outcome_rx,
                    };
                    if prompt_tx.send(prompt).is_err() {
                        break;
                    }
                    let tokenizer = state_for_task.tokenizer();
                    sampling.run(&mut lease, |engine| {
                        generate_streaming_logprobs_chunks(
                            engine,
                            tokenizer,
                            prompt_tokens,
                            max_tokens,
                            top_k,
                            &matcher_for_task,
                            &chunk_tx,
                        )
                    })
                }
            };
            match &result {
                Ok((generated, _)) => state_for_task
                    .metrics()
                    .tokens_generated_total
                    .inc_by(*generated as u64),
                Err(_) => state_for_task.metrics().errors_total.inc(),
            }
            let failed = result.is_err();
            // Each prompt's channel closes before its outcome is reported,
            // so the stream drains every chunk first. A dropped receiver is
            // harmless (the client went away).
            let _ = outcome_tx.send(result.map_err(|e| e.to_string()));
            if failed {
                break;
            }
        }
        // `lease` drops here: the engine returns to the pool.
    });

    let header = ChunkHeader {
        id: format!("cmpl-{}", completion_id_from_nanos()),
        created: unix_timestamp_secs(),
        model: model_id,
        include_usage: validated.include_usage,
    };
    let (payload_tx, payload_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(drive_prompts(PromptDriver {
        state: Arc::clone(&state),
        header,
        prompts: validated.prompts,
        echo: validated.echo,
        max_tokens,
        stop_matcher,
        prompt_rx,
        payload_tx,
    }));

    let guarded_stream = StreamCancelGuard {
        inner: tokio_stream::wrappers::UnboundedReceiverStream::new(payload_rx),
        _abandon: AbandonOnDrop(slot),
        _duration: duration,
    };

    Ok(sse_response(
        guarded_stream,
        state.limits(),
        request_id_header_map(request_id),
    ))
}

/// Everything [`drive_prompts`] needs.
struct PromptDriver {
    state: Arc<AppState>,
    header: ChunkHeader,
    /// The prompt texts, for `echo` and the `text_offset` base.
    prompts: Vec<String>,
    echo: bool,
    max_tokens: usize,
    stop_matcher: Arc<StopSequenceMatcher>,
    prompt_rx: tokio::sync::mpsc::UnboundedReceiver<PromptStream>,
    payload_tx: tokio::sync::mpsc::UnboundedSender<String>,
}

/// Turn each prompt's generation into its SSE chunks, in prompt order, then
/// the aggregated usage chunk. Ends early — cancelling the running
/// generation — when the client goes away, and after a failed prompt (whose
/// error object ends the stream).
async fn drive_prompts(driver: PromptDriver) {
    let PromptDriver {
        state,
        header,
        prompts,
        echo,
        max_tokens,
        stop_matcher,
        mut prompt_rx,
        payload_tx,
    } = driver;
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;
    while let Some(prompt) = prompt_rx.recv().await {
        let prompt_text = prompts.get(prompt.index).map(String::as_str).unwrap_or("");
        if echo
            && !prompt_text.is_empty()
            && payload_tx
                .send(header.text(prompt.index, prompt_text.to_string()))
                .is_err()
        {
            prompt.cancel.cancel();
            return;
        }
        let stopped_on_this_side = match prompt.body {
            PromptBody::Tokens(mut token_rx) => {
                let mut decoder = PieceDecoder::new(state.tokenizer());
                let mut stop = CompletionStopTracker::default();
                while let Some(id) = token_rx.recv().await {
                    if payload_tx.is_closed() {
                        prompt.cancel.cancel();
                        return;
                    }
                    if stop.stopped {
                        continue;
                    }
                    let DecodedPiece::Text(piece) = decoder.next(state.tokenizer(), id) else {
                        continue;
                    };
                    let released = stop.push_and_release(&piece, &stop_matcher);
                    // A match cancels this prompt's generation, not only
                    // what the stream shows.
                    if stop.stopped {
                        prompt.cancel.cancel();
                    }
                    if !released.is_empty()
                        && payload_tx
                            .send(header.text(prompt.index, released))
                            .is_err()
                    {
                        prompt.cancel.cancel();
                        return;
                    }
                }
                let rest = stop.take_remaining();
                if !rest.is_empty() && payload_tx.send(header.text(prompt.index, rest)).is_err() {
                    return;
                }
                stop.stopped
            }
            PromptBody::Logprobs(mut chunk_rx) => {
                // `text_offset` runs past the echoed prompt when `echo` is
                // set, like the non-streaming `build_completion_logprobs`.
                let mut offset = if echo { prompt_text.chars().count() } else { 0 };
                while let Some((text, entries)) = chunk_rx.recv().await {
                    let chars = text.chars().count();
                    let chunk = header.logprobs(prompt.index, text, &entries, offset);
                    offset += chars;
                    if payload_tx.send(chunk).is_err() {
                        prompt.cancel.cancel();
                        return;
                    }
                }
                false
            }
        };
        let outcome = prompt.outcome.await.unwrap_or_else(|_| {
            Err("the generation task ended without reporting an outcome".to_string())
        });
        match outcome {
            Ok((generated, stopped_in_loop)) => {
                total_prompt_tokens += prompt.prompt_tokens;
                total_completion_tokens += generated;
                let stopped = stopped_on_this_side || stopped_in_loop;
                let finish_reason = if !stopped && generated >= max_tokens {
                    "length"
                } else {
                    "stop"
                };
                if payload_tx
                    .send(header.finish(prompt.index, finish_reason))
                    .is_err()
                {
                    return;
                }
            }
            Err(message) => {
                tracing::error!(error = %message, "streaming generation failed mid-stream");
                let _ = payload_tx.send(ApiError::internal(message).to_json().to_string());
                return;
            }
        }
    }
    state
        .metrics()
        .prompt_tokens_total
        .inc_by(total_prompt_tokens as u64);
    if header.include_usage {
        let _ = payload_tx.send(header.usage(total_prompt_tokens, total_completion_tokens));
    }
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
