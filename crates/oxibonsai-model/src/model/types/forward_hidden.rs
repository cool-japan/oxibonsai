//! Hidden-state forward pass: the pre-LM-head seam a real embedding backend
//! needs (`RT-08` / `SV-02`, orchestrator decision D-1).
//!
//! [`BonsaiModel::forward`] and every fused GPU entry point return **logits**:
//! they run the Transformer blocks, apply `output_norm`, and then project
//! through the LM head. An embedding endpoint wants the state one step
//! earlier — the final-normed hidden state, `[n_tokens × hidden_size]` — which
//! no existing entry point exposes.
//!
//! # Why this is the host-KV block loop, and not a fused GPU dispatch
//!
//! The work order allows reusing a fused GPU **prefill** *if one exists that
//! skips the LM head*. None does. Every batched fused entry point —
//! `forward_metal.rs`'s `try_metal_prefill_with_lm_head` (and its `_ternary`
//! twin), `try_metal_full_forward_with_lm_head`, `forward_cuda.rs`'s
//! `try_cuda_full_forward_with_lm_head` — fuses the head into the same dispatch
//! and hands back `[vocab_size]` logits, with the pre-head hidden state never
//! leaving device memory.
//!
//! (The CUDA side is the same shape: every `forward_cuda/*.rs` prefill is
//! either `try_cuda_prefill_with_lm_head*` — logits — or
//! `try_cuda_prefill_verify*`, which returns argmax token ids and so is also
//! head-fused.)
//!
//! Head-free fused entry points *do* exist — `try_metal_full_forward_inner`,
//! `try_metal_full_forward_ternary_inner` and `try_cuda_full_forward_inner` —
//! but they are **single-token decode** paths, not prefills, and they read and
//! write the **process-global** device KV cache. Driving them from an embedding
//! request would interleave that request's positions with whatever a concurrent
//! chat completion is decoding through the same singleton. They are therefore
//! deliberately not used here, for correctness rather than convenience.
//!
//! So [`BonsaiModel::forward_hidden`] runs the **per-block host-KV loop** —
//! exactly the path [`forward_core`](super::BonsaiModel::forward) falls through
//! to — and says so at `debug!` rather than silently "falling back". Per-layer
//! GEMVs still dispatch to whatever tier the supplied
//! [`KernelDispatcher`](oxibonsai_kernels::KernelDispatcher) selects (including
//! Metal/CUDA, whose kernels are stateless and mutex-guarded), so this is not a
//! scalar-only path; it is only the *whole-model* fusion that is unavailable.
//!
//! # Per-sequence state: this call clobbers **this model's**, and nothing else's
//!
//! [`forward_hidden`](BonsaiModel::forward_hidden) writes positions
//! `0..tokens.len()` of the host KV cache, so it **cannot** be interleaved with
//! an ongoing generation on the same model. It therefore clears this model's
//! per-sequence state on both sides of the block loop: once before, so it never
//! attends over a previous sequence's history (and so the MET-05 device-KV latch
//! is cleared, making the host-KV block path legal again), and once after, so it
//! leaves nothing behind for whoever runs next.
//!
//! It deliberately does **not** go through [`BonsaiModel::reset`], even though
//! that is the obvious way to say the same thing. `reset` additionally calls
//! `MetalGraph::clear_global_kv_cache_if_present()`, and that cache is
//! **process-global**: a `reset` issued by an embedding request would wipe the
//! device-resident KV cache of a chat completion that a *different* engine is
//! decoding through the fused Metal path at that moment, silently corrupting
//! its output. Since embedding never reads the device KV cache (it runs the
//! per-block host-KV loop), it has no business releasing it, so
//! [`reset_embedding_sequence_state`](BonsaiModel::forward_hidden) clears only
//! the host cache, the coherence watermark, the MET-05 latch and the recurrent
//! state.
//!
//! A caller that owns a live conversation on *this* model must still not call
//! this mid-conversation; the runtime seam
//! ([`InferenceEngine::embed`](../../../../oxibonsai_runtime/embed_engine/index.html))
//! serialises embedding requests behind a `Mutex` on a dedicated engine for
//! exactly that reason.

use super::{run_blocks, BonsaiModel};
use crate::error::{ModelError, ModelResult};
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::KernelDispatcher;

/// Smallest L2 norm that is still normalised rather than treated as the zero
/// vector.
///
/// Matches `oxibonsai_rag::embedding::l2_normalize`'s own guard, so a pooled
/// vector that this crate declines to normalise is the same one the RAG side
/// would decline to normalise — the HTTP layer's `normalized: false` signal
/// stays consistent across both.
const MIN_NORMALIZABLE_NORM: f32 = 1e-10;

impl BonsaiModel<'_> {
    /// Final-normed hidden state of every token, **before** the LM head.
    ///
    /// Returns a row-major `[n_tokens × hidden_size]` buffer: row `i` is the
    /// output of `output_norm` for `tokens[i]` at position `i`, i.e. exactly
    /// what [`forward_core`](Self::forward) would have fed into
    /// `apply_lm_head`.
    ///
    /// Resets per-sequence state before *and* after the pass — see the module
    /// docs. Positions are assigned `0..tokens.len()`, so the caller does not
    /// supply a `pos_start`: an embedding is a property of the text alone.
    ///
    /// # Errors
    ///
    /// * [`ModelError::ShapeInvariant`] — `tokens` is empty, or the model has
    ///   a zero `hidden_size`.
    /// * [`ModelError::SequenceTooLong`] — the prompt is longer than the
    ///   model's effective context (sec-11).
    /// * [`ModelError::MissingTensor`] — a token id past the vocabulary: the
    ///   embedding table has no row for it (the error the row lookup
    ///   returns, naming the id and the vocabulary size).
    /// * Anything the blocks or `output_norm` return.
    pub fn forward_hidden(
        &mut self,
        tokens: &[u32],
        kernel: &KernelDispatcher,
    ) -> ModelResult<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        if tokens.is_empty() {
            return Err(empty_token_error("BonsaiModel::forward_hidden"));
        }
        if hidden_size == 0 {
            return Err(ModelError::ShapeInvariant {
                tensor: "BonsaiModel::forward_hidden".to_string(),
                expected: "hidden_size >= 1".to_string(),
                actual: "hidden_size = 0".to_string(),
            });
        }
        if tokens.len() > self.max_context {
            return Err(ModelError::SequenceTooLong {
                seq_len: tokens.len(),
                max_ctx: self.max_context,
            });
        }

        // The spec's "no silent fallback" clause: say *why* this is the CPU
        // block loop rather than a fused dispatch, once per call, at debug.
        tracing::debug!(
            tokens = tokens.len(),
            kernel = kernel.name(),
            tier = ?kernel.tier(),
            "forward_hidden: no head-free fused GPU PREFILL exists (every batched fused \
             entry point fuses the LM head, and the head-free fused paths are single-token \
             decode paths against the process-global device KV cache), so this runs the \
             per-block host-KV loop; per-layer GEMVs still dispatch to the supplied tier"
        );

        // Clear first: a stale host KV cache — or a set MET-05 device-KV latch
        // left by a fused GPU decode — would make position 0 attend over
        // another sequence's history.
        self.reset_embedding_sequence_state();
        let result = self.forward_hidden_inner(tokens, kernel, hidden_size);
        // Clear afterwards too, so an embedding pass never leaves its own
        // positions visible to whatever runs on this model next. Runs even on
        // the error path: a half-populated cache is exactly what must not
        // survive.
        self.reset_embedding_sequence_state();
        result
    }

    /// Clear everything [`BonsaiModel::reset`] clears **except** the
    /// process-global device KV cache — see the module docs for why that
    /// exception is the whole point.
    fn reset_embedding_sequence_state(&mut self) {
        self.kv_cache.clear();
        self.host_kv_written = 0;
        // Drop the MET-05 backend latch: this pass starts at position 0, where
        // the host-KV block path needs no history and is allowed again.
        self.set_gpu_path_active(false);
        // A hybrid model's recurrent `S` matrix is not masked by position, so
        // it must be cleared like the KV cache (RT-28 / M-Missed-3).
        self.reset_recurrent();
    }

    /// Body of [`forward_hidden`](Self::forward_hidden), between the two resets.
    fn forward_hidden_inner(
        &mut self,
        tokens: &[u32],
        kernel: &KernelDispatcher,
        hidden_size: usize,
    ) -> ModelResult<Vec<f32>> {
        let mut out = vec![0.0f32; tokens.len().saturating_mul(hidden_size)];
        let mut hidden = vec![0.0f32; hidden_size];
        let mut normed = vec![0.0f32; hidden_size];
        // Read once, up front: the `self.blocks` loop below needs it while
        // `self.kv_cache` is mutably borrowed (M-17, same reason
        // `forward_core` hoists it).
        let sliding_window = self.config.sliding_window;

        for (pos, &token_id) in tokens.iter().enumerate() {
            self.ensure_context_capacity(pos)?;
            self.token_embd.copy_row(token_id, &mut hidden)?;
            run_blocks(
                &self.blocks,
                sliding_window,
                &mut hidden,
                pos,
                &mut self.kv_cache,
                &self.rope,
                kernel,
            )?;
            // This is a host-KV path: record the watermark so a subsequent
            // position in this same loop passes MET-05's coherence check.
            self.note_host_kv_written(pos);
            self.output_norm.forward(&hidden, &mut normed)?;
            let row = pos * hidden_size;
            out[row..row + hidden_size].copy_from_slice(&normed);
        }
        Ok(out)
    }

    /// Mean-pooled, L2-normalised sentence embedding for `tokens`.
    ///
    /// The standard decoder-LM embedding recipe: run
    /// [`forward_hidden`](Self::forward_hidden), average the per-token final
    /// hidden states, then scale to unit length. The result has
    /// [`hidden_size`](Self::hidden_size) elements.
    ///
    /// A degenerate model (all-zero weights, e.g. the weight-less
    /// [`BonsaiModel::new`] test constructor) pools to the zero vector, whose
    /// norm is `0`; it is returned **as the zero vector** rather than divided
    /// by ~0 into `NaN`s. That is the same convention
    /// `oxibonsai_rag::embedding::l2_normalize` uses, and the HTTP layer
    /// reports it honestly through `EmbeddingResponse::normalized`.
    ///
    /// # Errors
    ///
    /// Everything [`forward_hidden`](Self::forward_hidden) can return; in
    /// particular [`ModelError::ShapeInvariant`] for empty input.
    pub fn embed_mean_pooled(
        &mut self,
        tokens: &[u32],
        kernel: &KernelDispatcher,
    ) -> ModelResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(empty_token_error("BonsaiModel::embed_mean_pooled"));
        }
        let hidden_size = self.config.hidden_size;
        let states = self.forward_hidden(tokens, kernel)?;
        let mut pooled = vec![0.0f32; hidden_size];
        let mut rows = 0usize;
        for row in states.chunks_exact(hidden_size) {
            for (acc, value) in pooled.iter_mut().zip(row) {
                *acc += *value;
            }
            rows += 1;
        }
        if rows == 0 {
            // Unreachable: `forward_hidden` rejects an empty `tokens` and
            // always writes exactly `tokens.len()` full rows. Handled as a
            // typed error anyway rather than dividing by zero.
            return Err(ModelError::ShapeInvariant {
                tensor: "BonsaiModel::embed_mean_pooled".to_string(),
                expected: format!("{} hidden rows", tokens.len()),
                actual: "0 hidden rows".to_string(),
            });
        }
        let inv_rows = 1.0f32 / rows as f32;
        for value in pooled.iter_mut() {
            *value *= inv_rows;
        }
        l2_normalize_in_place(&mut pooled);
        Ok(pooled)
    }
}

/// The typed "you gave me no tokens" error both public entry points return.
fn empty_token_error(who: &str) -> ModelError {
    ModelError::ShapeInvariant {
        tensor: who.to_string(),
        expected: "at least one token id".to_string(),
        actual: "0 token ids".to_string(),
    }
}

/// Scale `v` to unit L2 length in place, leaving a (near-)zero vector
/// untouched.
///
/// A local copy of `oxibonsai_rag::embedding::l2_normalize`'s body:
/// `oxibonsai-model` does not depend on `oxibonsai-rag` (and must not — the
/// dependency runs the other way), and a non-finite guard is added so a model
/// that produced `NaN`/`inf` hidden states reports a zero vector rather than
/// propagating `NaN` into a cosine similarity, where it would silently poison
/// every ranking it takes part in.
fn l2_normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() {
        for x in v.iter_mut() {
            *x = 0.0;
        }
        return;
    }
    if norm > MIN_NORMALIZABLE_NORM {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_kernels::dispatch::KernelTier;

    fn reference_kernel() -> KernelDispatcher {
        KernelDispatcher::with_tier(KernelTier::Reference)
    }

    #[test]
    fn forward_hidden_rejects_empty_input_with_a_typed_error() {
        let mut model = BonsaiModel::new(Qwen3Config::tiny_test());
        let err = model
            .forward_hidden(&[], &reference_kernel())
            .expect_err("empty token slice must be refused");
        assert!(
            matches!(err, ModelError::ShapeInvariant { .. }),
            "empty input must be a typed ShapeInvariant, got {err:?}"
        );
    }

    #[test]
    fn embed_mean_pooled_rejects_empty_input_with_a_typed_error() {
        let mut model = BonsaiModel::new(Qwen3Config::tiny_test());
        let err = model
            .embed_mean_pooled(&[], &reference_kernel())
            .expect_err("empty token slice must be refused");
        assert!(
            matches!(err, ModelError::ShapeInvariant { .. }),
            "empty input must be a typed ShapeInvariant, got {err:?}"
        );
    }

    #[test]
    fn forward_hidden_returns_one_row_per_token() {
        let config = Qwen3Config::tiny_test();
        let hidden_size = config.hidden_size;
        let mut model = BonsaiModel::new(config);
        let states = model
            .forward_hidden(&[1, 2, 3], &reference_kernel())
            .expect("weight-less forward_hidden should still run");
        assert_eq!(
            states.len(),
            3 * hidden_size,
            "forward_hidden must return [n_tokens x hidden_size]"
        );
    }

    #[test]
    fn forward_hidden_refuses_a_prompt_past_the_effective_context() {
        let mut config = Qwen3Config::tiny_test();
        config.max_context_length = 4;
        let mut model = BonsaiModel::new(config);
        let tokens: Vec<u32> = (0..5).collect();
        let err = model
            .forward_hidden(&tokens, &reference_kernel())
            .expect_err("a prompt past the effective context must be refused");
        assert!(
            matches!(err, ModelError::SequenceTooLong { .. }),
            "over-long prompt must be SequenceTooLong, got {err:?}"
        );
    }

    /// Pins the documented `# Errors` entry: a token past the vocabulary is
    /// `MissingTensor` (the embedding has no such row), not a shape error —
    /// and the pass still leaves no KV history behind.
    #[test]
    fn forward_hidden_refuses_a_token_past_the_vocabulary_as_missing_tensor() {
        let config = Qwen3Config::tiny_test();
        let vocab = u32::try_from(config.vocab_size).expect("tiny vocab fits u32");
        let mut model = BonsaiModel::new(config);
        let err = model
            .forward_hidden(&[1, vocab], &reference_kernel())
            .expect_err("a token past the vocabulary must be refused");
        match &err {
            ModelError::MissingTensor { name } => assert!(
                name.contains(&format!("token_id {vocab}")),
                "the error must name the offending id: {name}"
            ),
            other => panic!("a token past the vocabulary must be MissingTensor, got {other:?}"),
        }
        assert_eq!(model.kv_cache().seq_len(), 0);
    }

    #[test]
    fn forward_hidden_leaves_no_kv_history_behind() {
        let mut model = BonsaiModel::new(Qwen3Config::tiny_test());
        let _ = model
            .forward_hidden(&[7, 8, 9], &reference_kernel())
            .expect("forward_hidden should run");
        assert_eq!(
            model.kv_cache().seq_len(),
            0,
            "forward_hidden must reset per-sequence state on the way out"
        );
    }

    #[test]
    fn embed_mean_pooled_has_hidden_size_elements() {
        let config = Qwen3Config::tiny_test();
        let hidden_size = config.hidden_size;
        let mut model = BonsaiModel::new(config);
        let pooled = model
            .embed_mean_pooled(&[4, 5], &reference_kernel())
            .expect("weight-less pooling should still run");
        assert_eq!(pooled.len(), hidden_size);
    }

    #[test]
    fn l2_normalize_in_place_leaves_the_zero_vector_alone() {
        let mut v = vec![0.0f32; 8];
        l2_normalize_in_place(&mut v);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn l2_normalize_in_place_scales_to_unit_length() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize_in_place(&mut v);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
    }

    #[test]
    fn l2_normalize_in_place_zeroes_a_non_finite_vector() {
        let mut v = vec![f32::INFINITY, 1.0];
        l2_normalize_in_place(&mut v);
        assert!(
            v.iter().all(|x| *x == 0.0),
            "a non-finite vector must collapse to zeros, not propagate NaN"
        );
    }
}
