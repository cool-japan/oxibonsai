//! The model-backed embedder the standalone server answers `/v1/embeddings`
//! from (`HANDOVER-RT` item 13).
//!
//! Until this module the `oxibonsai-serve` binary built its engine pool
//! without keeping the parts an embedder needs and never handed the router
//! one, so `/v1/embeddings` on the standalone server always answered the
//! route's honest `501`, while `oxibonsai serve` (the CLI) served real
//! embeddings from the very same model. This is the CLI's construction,
//! reachable from the binary and from integration tests:
//!
//! * the pool is built with
//!   [`build_pool_from_gguf_parts`](oxibonsai_runtime::engine_pool::build_pool_from_gguf_parts),
//!   which keeps the leaked GGUF and the shared token-embedding table
//!   ([`PoolBuild`]);
//! * [`build_embedder`] wraps a **dedicated** embedding engine built off those
//!   same parts ([`ModelEmbedder::from_static_gguf`] — it shares the weights
//!   through the one memory map, so it costs a KV cache, not a second model),
//!   with its KV window bounded by the embedder's input ceiling
//!   ([`embedding_window`]);
//! * the result goes to the router through
//!   [`RouterOptions::with_embedder`](oxibonsai_runtime::server::RouterOptions::with_embedder).
//!
//! When there can be no embedder — no tokenizer to encode text with, or a
//! hybrid (`qwen35`) model, whose pre-LM-head hidden states are not exposed
//! yet (the typed `NOT_A_DENSE_MODEL` refusal) — [`build_embedder`] logs why
//! and returns `None`, and the route keeps answering `501`: a missing
//! embedder is never fatal to serving chat.

use std::sync::Arc;

use oxibonsai_runtime::embed_engine::{ModelEmbedder, DEFAULT_MAX_EMBEDDING_TOKENS};
use oxibonsai_runtime::engine::engine_error_code;
use oxibonsai_runtime::engine_pool::PoolBuild;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;

/// The KV window of the dedicated embedding engine for a server whose
/// engines run with a `max_seq_len` context: that context, capped at the
/// embedder's own input ceiling ([`DEFAULT_MAX_EMBEDDING_TOKENS`]) — an
/// embedding never feeds more tokens than that, so a larger cache would be
/// memory no request can use — and never zero.
pub fn embedding_window(max_seq_len: usize) -> usize {
    max_seq_len.clamp(1, DEFAULT_MAX_EMBEDDING_TOKENS)
}

/// The model-backed embedder `/v1/embeddings` is served from, or `None` —
/// the route's honest `501` — when there can be none (see the module docs).
///
/// Mirrors `oxibonsai serve`'s own construction exactly: `tokenizer` becomes
/// the embedder's (a `TokenizerBridge` is not `Clone`, so the caller loads a
/// second one for the router), `params`/`seed` configure the dedicated
/// engine's sampler (never used to embed, but required to build an engine),
/// and `max_seq_len` bounds its KV window through [`embedding_window`].
pub fn build_embedder(
    built: &PoolBuild,
    tokenizer: Option<TokenizerBridge>,
    params: SamplingParams,
    seed: u64,
    max_seq_len: usize,
) -> Option<Arc<ModelEmbedder>> {
    let Some(tokenizer) = tokenizer else {
        tracing::info!("no tokenizer: /v1/embeddings answers 501 (an embedder needs one)");
        return None;
    };
    let window = embedding_window(max_seq_len);
    match ModelEmbedder::from_static_gguf(
        built.gguf,
        Arc::clone(&built.shared_token_embd),
        Arc::new(tokenizer),
        params,
        seed,
        window,
    ) {
        Ok(embedder) => {
            tracing::info!(
                window,
                "serving /v1/embeddings from a dedicated embedding engine"
            );
            Some(embedder)
        }
        Err(e) if engine_error_code(&e) == Some("NOT_A_DENSE_MODEL") => {
            tracing::info!(
                error = %e,
                "embeddings are not supported for this model yet (known limitation); \
                 /v1/embeddings answers 501"
            );
            None
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "failed to build the embedding engine; /v1/embeddings answers 501"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedding_window_is_the_context_capped_at_the_embedder_ceiling() {
        assert_eq!(embedding_window(0), 1, "never a zero-length KV window");
        assert_eq!(embedding_window(1), 1);
        assert_eq!(embedding_window(512), 512);
        assert_eq!(
            embedding_window(DEFAULT_MAX_EMBEDDING_TOKENS),
            DEFAULT_MAX_EMBEDDING_TOKENS
        );
        assert_eq!(
            embedding_window(DEFAULT_MAX_EMBEDDING_TOKENS * 8),
            DEFAULT_MAX_EMBEDDING_TOKENS,
            "a long serving context must not size the embedding cache past what an \
             embedding can use"
        );
    }
}
