//! The body of `POST /rag/query`: the retrieval stage, the generation and the
//! response, which the handler runs under the per-request deadline.
//!
//! Retrieval, prompt assembly and tokenisation are one stage on the blocking
//! pool ([`RagState::prepare_query`]); the generation goes through the chat
//! routes' blocking-pool seam with the same abandonment guard and the same
//! stage record, so a deadline that expires says where it caught the request.

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use oxibonsai_rag::error::RagError;
use oxibonsai_rag::pipeline::RagConfig;

use super::{
    abandoned, api_error, rough_token_count, RagQueryRequest, RagQueryResponse, RagState, RagUsage,
};
use crate::engine_control::CancellationToken;
use crate::sampling::SamplingParams;
use crate::server::api_error::ApiError;
use crate::server::blocking::{run_blocking_generation, CancelOnAbandon};
use crate::server::deadline::CancelSlot;
use crate::server::phase::{self, Phase};
use crate::server::sampling_scope::RequestSampling;

/// Hard upper bound on the number of tokens a single `/rag/query` request may
/// ask the engine to generate, mirroring the chat server's output ceiling so an
/// oversized `max_tokens` cannot drive an unbounded allocation.
const MAX_RAG_OUTPUT_TOKENS: usize = 8192;

/// Inclusive bounds on the client-supplied `top_k` for `/rag/query`. Values
/// outside this range are rejected with `400 Bad Request` rather than
/// silently clamped, so callers get honest feedback instead of a
/// mismatch between the requested and actual retrieval depth (finding
/// `serve-api-06`/`rag-eval-01`).
const MIN_RAG_TOP_K: usize = 1;
const MAX_RAG_TOP_K: usize = 50;

/// The answer of a server that has no tokenizer: retrieval succeeded and the
/// prompt was built, but the model cannot be run on raw text. Transparent
/// instead of returning numeric token IDs dressed up as an answer.
const NO_TOKENIZER_ANSWER: &str = "[no tokenizer configured: retrieval succeeded and the \
                          prompt was built, but text generation is unavailable on \
                          this server]";

/// What [`RagState::prepare_query`] hands the generation step.
pub(super) struct PreparedQuery {
    /// The prompt built from the retrieved context and the query.
    prompt: String,
    /// The retrieved context chunks, in rank order.
    retrieved_chunks: Vec<String>,
    /// Documents in the index at query time.
    documents_searched: usize,
    /// Chunks the retriever returned.
    chunks_retrieved: usize,
    /// The encoded prompt; empty on a server without a tokenizer.
    input_tokens: Vec<u32>,
    /// The whitespace-token estimate reported on a server without a
    /// tokenizer; `0` otherwise.
    rough_prompt_tokens: usize,
}

impl RagState {
    /// The query-side work that runs before generation: retrieval (embedding
    /// the query, searching the vector store), prompt assembly and — with a
    /// tokenizer — encoding the prompt. All of it is linear in the size of
    /// the index, the retrieved context or the (client-supplied) query, so it
    /// runs on the blocking pool as one stage.
    pub(super) fn prepare_query(
        &self,
        query: &str,
        top_k: usize,
        cancel: &CancellationToken,
    ) -> Result<PreparedQuery, ApiError> {
        // A request abandoned while this stage waited for a blocking thread
        // starts no work.
        if cancel.is_cancelled() {
            return Err(abandoned("retrieval"));
        }

        // The pipeline's own `Retriever` is always constructed with a *fixed*
        // `RetrieverConfig` (top_k baked in at pipeline-construction time), so
        // `RagPipeline::build_prompt` cannot be asked to use a different top_k
        // per request. To make the client-supplied `top_k` genuinely drive both
        // the reported chunk count *and* the generation prompt (finding
        // `serve-api-06`/`rag-eval-01`), we perform retrieval ourselves against
        // the pipeline's embedder + vector store directly, then assemble the
        // context/prompt using the exact same defaults `RagState` constructs its
        // pipelines with (`RagConfig::default()`), and finally feed that same
        // context into the generation step.
        let pipeline = self.snapshot();
        let retriever = pipeline.retriever();
        let documents_searched = retriever.document_count();

        // Route through the retriever's own public retrieval API (RAG,
        // unowned sibling) instead of reconstructing the search call by
        // hand against `retriever.store()` directly, which used to bypass
        // both the zero-norm query guard (RAG-21 /
        // `Retriever::reject_degenerate_query`) and the retriever's own
        // *configured* `min_score` (a freshly-built
        // `RetrieverConfig::default()` was used instead of the pipeline's
        // real config). `retrieve_with_top_k` exists specifically so this
        // per-request client-supplied `top_k` can still override
        // `RetrieverConfig::top_k` while every other guard/config applies
        // exactly like `Retriever::retrieve`.
        let results = match retriever.retrieve_with_top_k(query, top_k) {
            Ok(results) => results,
            // An empty index is not a client error for this endpoint --
            // the pipeline still answers, just with no retrieved context
            // (matches `/rag/query`'s long-standing documented/tested
            // behaviour against a fresh server before any `/rag/index`
            // call).
            Err(RagError::NoDocumentsIndexed) => Vec::new(),
            Err(RagError::EmptyQueryVector) => {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    "query embedding has zero norm; refusing to return an arbitrary ranking",
                ));
            }
            Err(e) => {
                return Err(api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("query embedding failed: {e}"),
                ));
            }
        };

        let retrieved_chunks: Vec<String> = results.iter().map(|r| r.chunk.text.clone()).collect();
        let chunks_retrieved = retrieved_chunks.len();

        // Assemble the context block the same way
        // `RagPipeline::retrieve_context` does: concatenate chunk texts with
        // the configured separator, dropping chunks that would exceed
        // `max_context_chars`.
        let rag_defaults = RagConfig::default();
        let sep = &rag_defaults.context_separator;
        let mut parts: Vec<&str> = Vec::with_capacity(results.len());
        let mut total_chars = 0usize;
        for result in &results {
            let text_len = result.chunk.text.len();
            let sep_len = if parts.is_empty() { 0 } else { sep.len() };
            if total_chars + sep_len + text_len > rag_defaults.max_context_chars
                && !parts.is_empty()
            {
                break;
            }
            total_chars += sep_len + text_len;
            parts.push(&result.chunk.text);
        }
        let context = parts.join(sep.as_str());

        let prompt = rag_defaults
            .prompt_template
            .replace("{context}", &context)
            .replace("{query}", query);

        // When a tokenizer is configured we encode the *actual* context+query
        // prompt, so the answer genuinely depends on the retrieved context and
        // the query. Without a tokenizer we cannot map the prompt text onto the
        // model's vocabulary, so generation is skipped and said so honestly
        // rather than fabricating an answer from a fixed start token.
        let (input_tokens, rough_prompt_tokens) = match &self.tokenizer {
            Some(tokenizer) => {
                let tokens = match tokenizer.encode(&prompt) {
                    Ok(tokens) if !tokens.is_empty() => tokens,
                    Ok(_) => {
                        return Err(api_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "prompt encoded to an empty token sequence",
                        ));
                    }
                    Err(e) => {
                        return Err(api_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("prompt tokenisation failed: {e}"),
                        ));
                    }
                };
                (tokens, 0)
            }
            None => (Vec::new(), rough_token_count(&prompt)),
        };

        Ok(PreparedQuery {
            prompt,
            retrieved_chunks,
            documents_searched,
            chunks_retrieved,
            input_tokens,
            rough_prompt_tokens,
        })
    }
}

/// The generation step of a query: queue for a replica, arm its
/// cancellation, run the generation on the blocking pool, and return the
/// generated token ids.
///
/// The request's stage record follows the generation — queued, prefilling,
/// decoding, with the tokens generated so far — so a deadline that expires
/// says where it caught the request. The sampler runs exactly as
/// `InferenceEngine::generate_with_params` would run it (the request's
/// parameters swapped onto the replica's sampler for this one generation,
/// its PRNG untouched).
async fn generate_tokens(
    state: &Arc<RagState>,
    slot: &CancelSlot,
    input_tokens: Vec<u32>,
    max_tokens: usize,
    params: SamplingParams,
) -> Result<Vec<u32>, ApiError> {
    let phase = slot.phase();
    phase.enter(Phase::WaitingForEngine);
    let mut lease = state.engines.acquire().await.map_err(|e| {
        api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("engine pool acquire failed: {e}"),
        )
    })?;
    phase.enter(Phase::Prefill);

    // Arm cancellation now that generation is about to start: the token is
    // recorded in the slot the deadline cancels (with the prefill chunked, so
    // a long prompt's ingest observes it), and the guard holds another handle
    // to it across the blocking generation, so a handler future that is
    // dropped before the answer is in hand — the client went away, or a layer
    // outside the handler gave up on it — cancels the generation instead of
    // leaving the replica decoding to `max_tokens` for nobody.
    let abandon = CancelOnAbandon::new(slot.arm_lease(&mut lease));

    let sampling = RequestSampling {
        params,
        penalties: None,
        min_p: None,
        seed: None,
    };
    let generation_phase = phase.clone();
    #[cfg(test)]
    let probe = Arc::clone(state);
    // Generation is synchronous, CPU-bound work and must not run on a tokio
    // worker thread: the lease moves onto the blocking pool, which resets the
    // engine first and returns the replica to the pool when this finishes.
    let generated = run_blocking_generation(lease, move |lease| {
        #[cfg(test)]
        probe.note_stage("generation");
        sampling.run(lease, |engine| {
            phase::observe_generation(&generation_phase, |tx| {
                engine.generate_streaming_sync(&input_tokens, max_tokens, tx)
            })
        })
    })
    .await;
    // The answer (or the task's failure) is in hand: nothing is abandoned.
    abandon.disarm();
    generated?.map_err(|e| {
        tracing::error!(error = %e, "RAG generation failed");
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("generation failed: {e}"),
        )
    })
}

/// The handler proper, split out of `rag_query` so it runs under the
/// per-request deadline.
pub(super) async fn rag_query_inner(
    state: Arc<RagState>,
    req: RagQueryRequest,
    slot: CancelSlot,
) -> Result<Response, ApiError> {
    let RagQueryRequest {
        query,
        max_tokens,
        top_k,
        temperature,
        include_context,
    } = req;

    if query.trim().is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "query must not be empty",
        ));
    }

    let max_tokens = max_tokens.unwrap_or(256).clamp(1, MAX_RAG_OUTPUT_TOKENS);
    let top_k = match top_k {
        None => 3,
        Some(k) if (MIN_RAG_TOP_K..=MAX_RAG_TOP_K).contains(&k) => k,
        Some(k) => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "top_k ({k}) must be between {MIN_RAG_TOP_K} and {MAX_RAG_TOP_K} inclusive"
                ),
            ));
        }
    };
    let include_context = include_context.unwrap_or(false);

    // ── 1. Retrieve, build the prompt and encode it ─────────────────────────
    let cancel = CancellationToken::new();
    let prepared = {
        let worker = Arc::clone(&state);
        state
            .run_stage("retrieval", &cancel, move |cancel| {
                worker.prepare_query(&query, top_k, cancel)
            })
            .await??
    };
    let PreparedQuery {
        prompt,
        retrieved_chunks,
        documents_searched,
        chunks_retrieved,
        input_tokens,
        rough_prompt_tokens,
    } = prepared;

    // ── 2. Generate an answer from the real RAG prompt ───────────────────────
    // Without a tokenizer we cannot map the prompt text onto the model's
    // vocabulary, so we skip generation and say so honestly rather than
    // fabricating an answer from a fixed start token.
    let (answer, completion_tokens, prompt_tokens_count) = match &state.tokenizer {
        Some(tokenizer) => {
            let prompt_tokens_count = input_tokens.len();
            // The stage record learns the prompt's size.
            slot.phase().set_workload(prompt_tokens_count, 0);

            // Honor the request temperature when it is a valid value.
            let mut params = SamplingParams::default();
            if let Some(temperature) = temperature {
                if temperature.is_finite() && (0.0..=2.0).contains(&temperature) {
                    params.temperature = temperature;
                }
            }

            let output_tokens =
                generate_tokens(&state, &slot, input_tokens, max_tokens, params).await?;

            let completion_tokens = output_tokens.len();
            let answer = tokenizer.decode(&output_tokens).map_err(|e| {
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("decoding generated tokens failed: {e}"),
                )
            })?;
            (answer, completion_tokens, prompt_tokens_count)
        }
        None => {
            // No tokenizer: retrieval succeeded and the prompt was built, but
            // we cannot run the model on raw text. Be transparent instead of
            // returning numeric token IDs dressed up as an answer.
            (NO_TOKENIZER_ANSWER.to_string(), 0usize, rough_prompt_tokens)
        }
    };

    // ── 3. Build response ────────────────────────────────────────────────────
    let resp = RagQueryResponse {
        answer,
        retrieved_chunks: if include_context {
            Some(retrieved_chunks)
        } else {
            None
        },
        prompt_used: prompt,
        usage: RagUsage {
            documents_searched,
            chunks_retrieved,
            prompt_tokens: prompt_tokens_count,
            completion_tokens,
        },
    };

    Ok((StatusCode::OK, Json(resp)).into_response())
}
