//! Hidden-state forward pass: the pre-LM-head seam a real embedding backend
//! needs (`RT-08` / `SV-02`).
//!
//! [`BonsaiModel::forward`] and every fused GPU entry point return **logits**:
//! they run the Transformer blocks, apply `output_norm`, and then project
//! through the LM head. An embedding endpoint wants the state one step
//! earlier — the final-normed hidden state, `[n_tokens × hidden_size]` — which
//! [`BonsaiModel::forward_hidden`] returns.
//!
//! # Three routes, one contract
//!
//! [`forward_hidden`](BonsaiModel::forward_hidden) tries, in order:
//!
//! 1. **The head-free Metal prefill** (`forward_metal_hidden.rs`, Metal
//!    builds only), when the caller's dispatcher is a GPU tier and the
//!    model's fused weight cache is resident: every layer as batched GEMMs
//!    and flash attention on the GPU over 128-row micro-batches, the
//!    `output_norm` of every row read back, no LM head. It runs in a
//!    request-scoped Metal session with its own device KV cache, so it never
//!    touches the KV cache of a generation the process is decoding (MET-05).
//!    It declines (a CPU dispatcher, a sliding window, a non-Q1/TQ2 head, no
//!    resident cache) without doing anything, and any error it returns sends
//!    the call on to the next route with one `debug` line.
//! 2. **The batched CPU prefill** (`prefill_cpu.rs`): every projection is a
//!    register-blocked GEMM over a micro-batch of prompt rows instead of one
//!    GEMV sweep over the weights per token, and the pass hands each
//!    micro-batch's post-block rows to a sink that applies `output_norm` to
//!    every row. Those rows are exactly what the per-token path would have
//!    fed to the LM head, computed by the same pass the generation prefill
//!    uses, so the two cannot disagree about a hidden state. It runs its
//!    GEMMs on the best **CPU** tier whatever dispatcher the caller supplies.
//! 3. When the batched pass declines too — a single token, a model that
//!    declares a sliding attention window, a layer outside `Q1_0_g128` /
//!    `TQ2_0_g128`, a geometry it does not handle; every decline is decided
//!    before anything is written —
//!    [`forward_hidden_sequential`](BonsaiModel::forward_hidden_sequential):
//!    the per-token host-KV block loop, whose per-layer GEMVs dispatch to the
//!    supplied tier. That function is public because it is also the parity
//!    reference both batched routes are measured against, numerically (tests
//!    bound every row and the pooled vector) and in time (the runtime's
//!    embedding benchmark).
//!
//! # Why a request-scoped Metal session, and not the decode entry points
//!
//! Every other batched fused entry point — `forward_metal.rs`'s
//! `try_metal_prefill_with_lm_head` (and its `_ternary` twin),
//! `try_metal_full_forward_with_lm_head`, and on CUDA every
//! `try_cuda_prefill_with_lm_head*` (logits) or `try_cuda_prefill_verify*`
//! (argmax ids) — fuses the LM head into the same dispatch, with the pre-head
//! hidden state never leaving device memory, and the head-free fused entry
//! points (`try_metal_full_forward_inner`,
//! `try_metal_full_forward_ternary_inner`, `try_cuda_full_forward_inner`) are
//! **single-token decode** paths that read and write the device KV cache of
//! the session the calling thread dispatches in. Driving any of them from an
//! embedding request would interleave its positions with whatever a
//! concurrent chat completion is decoding through that same cache. The Metal
//! route instead runs the prefill kernels in a sibling session of its own and
//! drops it on return. CUDA has no head-free batched route; there the batched
//! CPU pass stays the embedding path.
//!
//! # Measured (real `Ternary-Bonsai-1.7B.gguf`, Apple M3, release)
//!
//! The runtime's `embed_bench_short_and_long` (`OXI_MODEL` or
//! `OXIBONSAI_EMBED_BENCH=1`) times the production `InferenceEngine::embed`
//! on a Metal engine — the head-free Metal prefill — against the batched CPU
//! pass (`forward_hidden` on a CPU-tier dispatcher, the production path
//! before the Metal route existed) and the per-token loop both replaced
//! (`forward_hidden_sequential` on a dispatcher built exactly as the
//! engine's own), minimum of three runs per leg (a single run for the two
//! slow legs at 2000 tokens), load average 10-12:
//!
//! | tokens | per-token loop | batched CPU | Metal | Metal vs per-token |
//! |---|---|---|---|---|
//! | 10 | 1.521 s | 0.202 s | **0.084 s** | 18.1x |
//! | 200 | 42.32 s | 3.245 s | **0.330 s** | 128.2x |
//! | 2000 | 352.5 s (one run) | 45.40 s (one run) | **3.250 s** | 108.5x |
//!
//! The pooled vectors of all three routes agree to cosine >= 0.999999 at
//! every length; the real-model gate (`metal_hidden_parity_tests`) holds the
//! Metal rows of all three dense models to per-row cosine >= 0.999 and
//! pooled >= 0.9999 against the per-token reference at 10, 300 and 2000
//! tokens (measured worst row 0.99995, on Ternary-Bonsai-8B at 2000 tokens).
//! Before the Metal route, the batched CPU pass needed 48.2 s for the
//! 2000-token input at load ~21 and 95.7 s at load ~96 — over the HTTP
//! layer's 60 s request budget under load; the Metal pass stays an order of
//! magnitude inside it. The per-token loop is this slow on a GPU engine
//! because each of its per-layer GEMVs is a separate Metal dispatch with its
//! own wait.
//!
//! # Per-sequence state: this call clobbers **this model's**, and nothing else's
//!
//! Both CPU routes write positions `0..tokens.len()` of the host KV cache, so
//! they **cannot** be interleaved with an ongoing generation on the same
//! model (the Metal route writes only its own request-scoped device KV).
//! `forward_hidden` therefore clears this model's per-sequence state on both
//! sides of the pass, whichever route runs: once before, so it never attends
//! over a previous sequence's history (and so the MET-05 device-KV latch is
//! cleared, making the host-KV path legal again), and once after, so it
//! leaves nothing behind for whoever runs next.
//!
//! None goes through [`BonsaiModel::reset`], even though that is the
//! obvious way to say the same thing. `reset` additionally calls
//! `MetalGraph::clear_global_kv_cache_if_present()`, and that cache is
//! **process-global**: a `reset` issued by an embedding request would wipe the
//! device-resident KV cache of a chat completion that a *different* engine is
//! decoding through the fused Metal path at that moment, silently corrupting
//! its output. Since embedding never reads the device KV cache, it has no
//! business releasing it, so `reset_embedding_sequence_state` clears only the
//! host cache, the coherence watermark, the MET-05 latch and the recurrent
//! state.
//!
//! A caller that owns a live conversation on *this* model must still not call
//! this mid-conversation; the runtime seam (`InferenceEngine::embed` in
//! `oxibonsai-runtime`) serialises embedding requests behind a `Mutex` on a
//! dedicated engine for exactly that reason.

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
    /// Runs the head-free Metal prefill when `kernel` is a GPU tier and the
    /// model's fused weight cache is resident, else the batched CPU prefill
    /// (see the module docs), whose GEMMs use the best CPU tier regardless of
    /// `kernel`; `kernel` is also what the per-token fallback dispatches to
    /// when the batched CPU pass declines this model or input.
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
        Self::check_embedding_input(
            tokens,
            hidden_size,
            self.max_context,
            "BonsaiModel::forward_hidden",
        )?;
        // Clear first: a stale host KV cache — or a set MET-05 device-KV latch
        // left by a fused GPU decode — would make position 0 attend over
        // another sequence's history.
        self.reset_embedding_sequence_state();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        match self.try_metal_forward_hidden(tokens, kernel) {
            Ok(Some(rows)) => {
                // The Metal pass wrote nothing of this model's state; the
                // reset keeps the "nothing survives an embedding" contract
                // uniform across the routes.
                self.reset_embedding_sequence_state();
                return Ok(rows);
            }
            Ok(None) => {}
            Err(e) => tracing::debug!(
                error = %e,
                tokens = tokens.len(),
                "forward_hidden: the head-free Metal prefill failed; running the batched CPU pass"
            ),
        }
        let result = match self.forward_hidden_batched(tokens, hidden_size) {
            Ok(Some(rows)) => Ok(rows),
            Ok(None) => {
                tracing::debug!(
                    tokens = tokens.len(),
                    kernel = kernel.name(),
                    tier = ?kernel.tier(),
                    "forward_hidden: the batched CPU prefill declined this model or input \
                     (one token, a sliding window, a layer outside Q1_0_g128/TQ2_0_g128, or \
                     an unhandled geometry); running the per-token host-KV loop on the \
                     supplied tier"
                );
                // The decline wrote nothing, so the state is still clean.
                self.forward_hidden_sequential_inner(tokens, kernel, hidden_size)
            }
            Err(e) => Err(e),
        };
        // Clear afterwards too, so an embedding pass never leaves its own
        // positions visible to whatever runs on this model next. Runs even on
        // the error path: a half-populated cache is exactly what must not
        // survive.
        self.reset_embedding_sequence_state();
        result
    }

    /// [`forward_hidden`](Self::forward_hidden) computed by the per-token
    /// host-KV block loop: one sweep of every block per token, dispatching
    /// every per-layer GEMV to `kernel`.
    ///
    /// The path `forward_hidden` falls back to when the batched pass
    /// declines, and the reference it is measured against: the same
    /// contract, the same errors and the same per-sequence resets on both
    /// sides, but one full sweep over the weights per token — the batched
    /// path is several times faster, and more so the shorter the input.
    ///
    /// # Errors
    ///
    /// As [`forward_hidden`](Self::forward_hidden).
    pub fn forward_hidden_sequential(
        &mut self,
        tokens: &[u32],
        kernel: &KernelDispatcher,
    ) -> ModelResult<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        Self::check_embedding_input(
            tokens,
            hidden_size,
            self.max_context,
            "BonsaiModel::forward_hidden_sequential",
        )?;
        self.reset_embedding_sequence_state();
        let result = self.forward_hidden_sequential_inner(tokens, kernel, hidden_size);
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

    /// The batched half of [`forward_hidden`](Self::forward_hidden):
    /// `Ok(None)` — having written nothing — when the batched CPU prefill
    /// declines this model or input.
    fn forward_hidden_batched(
        &mut self,
        tokens: &[u32],
        hidden_size: usize,
    ) -> ModelResult<Option<Vec<f32>>> {
        let len =
            tokens
                .len()
                .checked_mul(hidden_size)
                .ok_or_else(|| ModelError::ShapeInvariant {
                    tensor: "BonsaiModel::forward_hidden".to_string(),
                    expected: "n_tokens * hidden_size representable as usize".to_string(),
                    actual: format!("{} x {hidden_size}", tokens.len()),
                })?;
        let mut out = vec![0.0f32; len];
        let ran = self.run_prefill_cpu(tokens, 0, |output_norm, rows| {
            let start = rows.first * hidden_size;
            let end = start + rows.count * hidden_size;
            let dst = out
                .get_mut(start..end)
                .ok_or_else(|| ModelError::ShapeInvariant {
                    tensor: "BonsaiModel::forward_hidden rows".to_string(),
                    expected: format!("rows {start}..{end} of {len} floats"),
                    actual: "out of range".to_string(),
                })?;
            for (src, dst) in rows
                .data
                .chunks_exact(hidden_size)
                .zip(dst.chunks_exact_mut(hidden_size))
            {
                output_norm.forward(src, dst)?;
            }
            Ok(())
        })?;
        Ok(ran.then_some(out))
    }

    /// Body of the per-token path, between the two resets.
    fn forward_hidden_sequential_inner(
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
        Self::mean_pool_normalized(
            &states,
            hidden_size,
            tokens.len(),
            "BonsaiModel::embed_mean_pooled",
        )
    }

    /// Refuse an embedding input every model kind must refuse, with the same
    /// typed errors whichever model is asked (the hybrid stack's
    /// `forward_hidden` calls this too).
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] for an empty input or a zero hidden
    /// width; [`ModelError::SequenceTooLong`] for more than `max_tokens`
    /// tokens.
    pub(crate) fn check_embedding_input(
        tokens: &[u32],
        hidden_size: usize,
        max_tokens: usize,
        who: &str,
    ) -> ModelResult<()> {
        if tokens.is_empty() {
            return Err(empty_token_error(who));
        }
        if hidden_size == 0 {
            return Err(ModelError::ShapeInvariant {
                tensor: who.to_string(),
                expected: "hidden_size >= 1".to_string(),
                actual: "hidden_size = 0".to_string(),
            });
        }
        if tokens.len() > max_tokens {
            return Err(ModelError::SequenceTooLong {
                seq_len: tokens.len(),
                max_ctx: max_tokens,
            });
        }
        Ok(())
    }

    /// Average `n_tokens` rows of `states` (`[n_tokens × hidden_size]`) and
    /// scale the result to unit L2 length — the pooling step every model
    /// kind's `embed_mean_pooled` shares, so the dense and hybrid stacks
    /// cannot pool differently.
    ///
    /// A (near-)zero mean stays the zero vector, and a non-finite one becomes
    /// the zero vector, rather than being divided into `NaN`s (see
    /// [`embed_mean_pooled`](Self::embed_mean_pooled)).
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] when `states` does not hold exactly
    /// `n_tokens` full rows, or either count is zero.
    pub(crate) fn mean_pool_normalized(
        states: &[f32],
        hidden_size: usize,
        n_tokens: usize,
        who: &str,
    ) -> ModelResult<Vec<f32>> {
        if hidden_size == 0 || n_tokens == 0 || states.len() != n_tokens * hidden_size {
            return Err(ModelError::ShapeInvariant {
                tensor: who.to_string(),
                expected: format!("{n_tokens} hidden rows of {hidden_size} floats"),
                actual: format!("{} floats", states.len()),
            });
        }
        let mut pooled = vec![0.0f32; hidden_size];
        for row in states.chunks_exact(hidden_size) {
            for (acc, value) in pooled.iter_mut().zip(row) {
                *acc += *value;
            }
        }
        let inv_rows = 1.0f32 / n_tokens as f32;
        for value in pooled.iter_mut() {
            *value *= inv_rows;
        }
        l2_normalize_in_place(&mut pooled);
        Ok(pooled)
    }
}

/// The typed "you gave me no tokens" error every embedding entry point
/// returns.
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
    use crate::model::types::prefill_cpu::tests::{
        cosine, fixture, max_scaled_diff, ternary_fixture, tiny_config,
    };
    use crate::model::types::prefill_cpu::CPU_PREFILL_MICRO_BATCH;
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

    // ── The batched path against its sequential reference ──────────────────

    /// Every row of the batched result against the sequential reference,
    /// under the bound the batched prefill's own logit parity test uses
    /// (`batched_prefill_matches_the_sequential_reference`: cosine >= 0.9999
    /// and every element within 1e-4 of the vector's scale), plus the pooled
    /// embeddings at cosine >= 0.99999.
    fn assert_rows_match(batched: &[f32], sequential: &[f32], hidden_size: usize, what: &str) {
        assert_eq!(batched.len(), sequential.len(), "{what}: row buffer length");
        for (row, (got, want)) in batched
            .chunks_exact(hidden_size)
            .zip(sequential.chunks_exact(hidden_size))
            .enumerate()
        {
            let cos = cosine(got, want);
            assert!(cos >= 0.9999, "{what}: row {row} cos {cos}");
            let (rel, idx) = max_scaled_diff(got, want);
            assert!(
                rel <= 1e-4,
                "{what}: row {row} element {idx} diverged by {rel} (sequential {}, batched {})",
                want[idx],
                got[idx]
            );
        }
        let n = batched.len() / hidden_size;
        let pooled_batched = BonsaiModel::mean_pool_normalized(batched, hidden_size, n, what)
            .expect("pool the batched rows");
        let pooled_sequential = BonsaiModel::mean_pool_normalized(sequential, hidden_size, n, what)
            .expect("pool the sequential rows");
        let cos = cosine(&pooled_batched, &pooled_sequential);
        assert!(cos >= 0.99999, "{what}: pooled cos {cos}");
    }

    fn prompt(len: u32, mul: u32, add: u32, vocab: u32) -> Vec<u32> {
        (0..len).map(|i| (i * mul + add) % vocab).collect()
    }

    #[test]
    fn forward_hidden_batched_matches_the_sequential_reference() {
        let cfg = tiny_config(2);
        let hidden = cfg.hidden_size;
        let tokens = prompt(12, 5, 0, 96);
        let mut model = fixture(cfg);
        let batched = model
            .forward_hidden(&tokens, &reference_kernel())
            .expect("batched forward_hidden");
        let sequential = model
            .forward_hidden_sequential(&tokens, &reference_kernel())
            .expect("sequential reference");
        assert_rows_match(&batched, &sequential, hidden, "Q1_0_g128");
    }

    #[test]
    fn forward_hidden_batched_matches_the_sequential_reference_ternary() {
        let cfg = tiny_config(2);
        let hidden = cfg.hidden_size;
        let tokens = prompt(12, 5, 0, 96);
        let mut model = ternary_fixture(cfg);
        let batched = model
            .forward_hidden(&tokens, &reference_kernel())
            .expect("batched forward_hidden");
        let sequential = model
            .forward_hidden_sequential(&tokens, &reference_kernel())
            .expect("sequential reference");
        assert_rows_match(&batched, &sequential, hidden, "TQ2_0_g128");
    }

    /// A prompt longer than one micro-batch: every micro-batch's rows must
    /// land at their own offset, so the boundary is invisible row by row.
    #[test]
    fn forward_hidden_micro_batch_boundary_is_invisible() {
        let cfg = tiny_config(1);
        let hidden = cfg.hidden_size;
        let len = u32::try_from(CPU_PREFILL_MICRO_BATCH + 5).expect("fits");
        let tokens = prompt(len, 3, 1, 96);
        for (label, mut model) in [
            ("Q1_0_g128", fixture(cfg.clone())),
            ("TQ2_0_g128", ternary_fixture(cfg.clone())),
        ] {
            let batched = model
                .forward_hidden(&tokens, &reference_kernel())
                .expect("batched forward_hidden");
            let sequential = model
                .forward_hidden_sequential(&tokens, &reference_kernel())
                .expect("sequential reference");
            assert_rows_match(&batched, &sequential, hidden, label);
        }
    }

    /// The batched half really runs for an in-scope model and input, and
    /// really declines — writing nothing — for every out-of-scope one.
    #[test]
    fn forward_hidden_takes_the_batched_path_exactly_when_it_is_in_scope() {
        let hidden = tiny_config(2).hidden_size;
        let mut model = fixture(tiny_config(2));
        let rows = model
            .forward_hidden_batched(&[3, 4, 5], hidden)
            .expect("in scope")
            .expect("an in-scope model and input takes the batched path");
        assert_eq!(rows.len(), 3 * hidden);
        model.reset_embedding_sequence_state();

        assert!(
            model
                .forward_hidden_batched(&[3], hidden)
                .expect("declining is not an error")
                .is_none(),
            "one token is the decode path"
        );
        assert_eq!(model.kv_cache().seq_len(), 0, "a decline writes nothing");

        let mut windowed_cfg = tiny_config(2);
        windowed_cfg.sliding_window = Some(4);
        let mut windowed = fixture(windowed_cfg);
        assert!(windowed
            .forward_hidden_batched(&[3, 4, 5], hidden)
            .expect("declining is not an error")
            .is_none());

        let mut blockless = BonsaiModel::new(tiny_config(2));
        assert!(blockless
            .forward_hidden_batched(&[3, 4, 5], hidden)
            .expect("declining is not an error")
            .is_none());
    }

    /// When the batched pass declines, `forward_hidden` *is* the sequential
    /// path — bit for bit, not merely within tolerance.
    #[test]
    fn a_declined_input_falls_back_to_the_sequential_path_bit_for_bit() {
        let mut windowed_cfg = tiny_config(2);
        windowed_cfg.sliding_window = Some(4);
        let tokens = prompt(9, 7, 2, 96);
        let mut model = fixture(windowed_cfg);
        let via_fallback = model
            .forward_hidden(&tokens, &reference_kernel())
            .expect("forward_hidden");
        let reference = model
            .forward_hidden_sequential(&tokens, &reference_kernel())
            .expect("sequential");
        assert_eq!(
            via_fallback.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let mut model = fixture(tiny_config(2));
        let one = model
            .forward_hidden(&[17], &reference_kernel())
            .expect("one token");
        let one_ref = model
            .forward_hidden_sequential(&[17], &reference_kernel())
            .expect("one token, sequential");
        assert_eq!(one, one_ref);
    }

    /// The batched path clears per-sequence state on both sides: nothing
    /// survives it, and a previous sequence's history cannot leak into it.
    #[test]
    fn forward_hidden_batched_leaves_no_kv_history_and_ignores_a_previous_sequence() {
        let kernel = reference_kernel();
        let tokens = prompt(10, 11, 3, 96);
        let mut clean = fixture(tiny_config(2));
        let expected = clean.forward_hidden(&tokens, &kernel).expect("clean");
        assert_eq!(clean.kv_cache().seq_len(), 0);
        assert_eq!(clean.host_kv_written, 0);

        // A model mid-conversation: five positions of unrelated history.
        let mut busy = fixture(tiny_config(2));
        for (pos, tok) in [9u32, 8, 7, 6, 5].into_iter().enumerate() {
            busy.forward(tok, pos, &kernel).expect("decode");
        }
        assert_eq!(busy.kv_cache().seq_len(), 5);
        let got = busy.forward_hidden(&tokens, &kernel).expect("busy");
        assert_eq!(
            got, expected,
            "an embedding must not depend on what the model decoded before"
        );
        assert_eq!(busy.kv_cache().seq_len(), 0);
        assert_eq!(busy.host_kv_written, 0);
    }

    /// A token past the vocabulary is refused by the batched path with the
    /// same typed error the sequential path gives, and nothing survives.
    #[test]
    fn forward_hidden_batched_refuses_a_token_past_the_vocabulary_as_missing_tensor() {
        let cfg = tiny_config(2);
        let vocab = u32::try_from(cfg.vocab_size).expect("fits u32");
        let mut model = fixture(cfg);
        let err = model
            .forward_hidden(&[1, 2, vocab], &reference_kernel())
            .expect_err("a token past the vocabulary must be refused");
        assert!(
            matches!(err, ModelError::MissingTensor { .. }),
            "got {err:?}"
        );
        assert_eq!(model.kv_cache().seq_len(), 0);
    }

    #[test]
    fn forward_hidden_batched_is_deterministic() {
        let tokens = prompt(20, 13, 5, 96);
        let mut model = ternary_fixture(tiny_config(2));
        let a = model
            .forward_hidden(&tokens, &reference_kernel())
            .expect("first");
        let b = model
            .forward_hidden(&tokens, &reference_kernel())
            .expect("second");
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn embed_mean_pooled_on_the_batched_path_is_a_unit_vector_matching_the_reference() {
        let cfg = tiny_config(2);
        let hidden = cfg.hidden_size;
        let tokens = prompt(15, 7, 1, 96);
        let mut model = fixture(cfg);
        let pooled = model
            .embed_mean_pooled(&tokens, &reference_kernel())
            .expect("pooled");
        assert_eq!(pooled.len(), hidden);
        let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "unit norm, got {norm}");
        let rows = model
            .forward_hidden_sequential(&tokens, &reference_kernel())
            .expect("reference rows");
        let reference = BonsaiModel::mean_pool_normalized(&rows, hidden, tokens.len(), "ref")
            .expect("reference pooling");
        let cos = cosine(&pooled, &reference);
        assert!(cos >= 0.99999, "pooled cos {cos}");
    }

    #[test]
    fn mean_pool_normalized_averages_rows_and_scales_to_unit_length() {
        let states = [1.0f32, 0.0, 3.0, 0.0];
        let pooled = BonsaiModel::mean_pool_normalized(&states, 2, 2, "test").expect("well formed");
        assert_eq!(pooled, vec![1.0, 0.0]);
        let zero = BonsaiModel::mean_pool_normalized(&[0.0; 6], 3, 2, "test").expect("zeros");
        assert!(zero.iter().all(|x| *x == 0.0));
        for (states, hidden, n) in [(&[1.0f32; 5][..], 2usize, 2usize), (&[][..], 2, 0)] {
            let err = BonsaiModel::mean_pool_normalized(states, hidden, n, "test")
                .expect_err("malformed input");
            assert!(matches!(err, ModelError::ShapeInvariant { .. }), "{err:?}");
        }
    }

    #[test]
    fn check_embedding_input_gives_the_shared_typed_errors() {
        assert!(matches!(
            BonsaiModel::check_embedding_input(&[], 8, 4, "t"),
            Err(ModelError::ShapeInvariant { .. })
        ));
        assert!(matches!(
            BonsaiModel::check_embedding_input(&[1], 0, 4, "t"),
            Err(ModelError::ShapeInvariant { .. })
        ));
        assert!(matches!(
            BonsaiModel::check_embedding_input(&[1; 5], 8, 4, "t"),
            Err(ModelError::SequenceTooLong {
                seq_len: 5,
                max_ctx: 4
            })
        ));
        assert!(BonsaiModel::check_embedding_input(&[1; 4], 8, 4, "t").is_ok());
    }

    // ── Real models ──────────────────────────────────────────────────────

    /// `"The quick brown fox jumps over the lazy dog."` (Qwen3 tokenizer).
    const TEXT_FOX: [u32; 10] = [785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13];

    /// `"Embeddings map a piece of text to a vector, so that passages with
    /// similar meanings land near each other."`
    const TEXT_EMBEDDINGS: [u32; 22] = [
        25486, 24602, 2415, 264, 6573, 315, 1467, 311, 264, 4621, 11, 773, 429, 46769, 448, 4428,
        49700, 4268, 3143, 1817, 1008, 13,
    ];

    /// `"A mixture of experts routes every token to a few specialised
    /// feed-forward networks, while a dense model sends every token through
    /// all of its weights. …"` (56 tokens).
    const TEXT_EXPERTS: [u32; 56] = [
        32, 20980, 315, 11647, 11291, 1449, 3950, 311, 264, 2421, 87904, 5395, 44804, 14155, 11,
        1393, 264, 27950, 1614, 21308, 1449, 3950, 1526, 678, 315, 1181, 14324, 13, 11733, 525,
        16176, 835, 311, 835, 11, 323, 2176, 525, 25070, 389, 279, 1852, 62019, 11, 714, 862, 4938,
        323, 12564, 20872, 1745, 45373, 518, 44378, 882, 13,
    ];

    /// `"Rivers carve valleys over millions of years. …"` (104 tokens).
    const TEXT_RIVERS: [u32; 104] = [
        49, 1945, 79637, 85397, 916, 11728, 315, 1635, 13, 9959, 51207, 304, 279, 1550, 8166, 11,
        85681, 4628, 438, 432, 6560, 1412, 11, 323, 23377, 9278, 323, 9798, 429, 39236, 279, 14796,
        2721, 19117, 448, 1449, 17726, 13, 10967, 279, 4268, 51039, 724, 11, 279, 1482, 69170, 323,
        21025, 1181, 2795, 11, 4752, 69125, 77366, 323, 90587, 429, 614, 22313, 9720, 2474, 279,
        1156, 23429, 82, 13, 48696, 1431, 1936, 82525, 323, 55489, 288, 311, 81823, 429, 4802, 11,
        11133, 279, 2310, 10775, 315, 17726, 323, 42801, 369, 24020, 3015, 323, 17728, 11, 323,
        279, 35517, 4226, 553, 7218, 862, 58032, 14696, 770, 13,
    ];

    /// Length of the long text: several micro-batches, and attention over
    /// a history two orders of magnitude longer than the short texts'.
    const LONG_TEXT_TOKENS: usize = 1000;

    /// The four texts: three sentences/paragraphs and a ~1000-token document
    /// made of the four passages above, repeated in order.
    fn real_texts() -> Vec<(&'static str, Vec<u32>)> {
        let passage: Vec<u32> = TEXT_FOX
            .iter()
            .chain(TEXT_EMBEDDINGS.iter())
            .chain(TEXT_EXPERTS.iter())
            .chain(TEXT_RIVERS.iter())
            .copied()
            .collect();
        let long: Vec<u32> = passage
            .iter()
            .copied()
            .cycle()
            .take(LONG_TEXT_TOKENS)
            .collect();
        vec![
            ("fox (10 tokens)", TEXT_FOX.to_vec()),
            ("embeddings (22 tokens)", TEXT_EMBEDDINGS.to_vec()),
            ("rivers (104 tokens)", TEXT_RIVERS.to_vec()),
            ("document (1000 tokens)", long),
        ]
    }

    /// The batched hidden rows against the per-token reference on the real
    /// shipped models: `Ternary-Bonsai-1.7B` (`TQ2_0_g128`) and `Bonsai-8B`
    /// (`Q1_0_g128`), four texts each including a ~1000-token one, pooled
    /// cosine >= 0.9999 per text. Both legs run on the same CPU tier (the one
    /// the batched pass pins), so the comparison isolates the batched pass
    /// from any cross-tier arithmetic difference.
    ///
    /// Model files are located only through `OXIBONSAI_MODELS_DIR`; the test
    /// self-skips — recording the skip — when it is unset or a file is
    /// absent, unless `OXI_REQUIRE_MODEL_FILES=1` demands the files.
    #[test]
    fn real_model_batched_hidden_matches_sequential() {
        use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
        use oxibonsai_core::GgufTensorType;
        use oxibonsai_testkit::capability::{record_executed, record_skipped, Capability};

        const TEST_NAME: &str =
            "oxibonsai-model::lib::real_model_batched_hidden_matches_sequential";
        const MAX_SEQ: usize = 2048;
        let require = std::env::var("OXI_REQUIRE_MODEL_FILES").is_ok_and(|v| v == "1");
        let dir_set = std::env::var_os("OXIBONSAI_MODELS_DIR").is_some_and(|d| !d.is_empty());
        if !dir_set {
            assert!(
                !require,
                "OXI_REQUIRE_MODEL_FILES=1: set OXIBONSAI_MODELS_DIR to the real-model directory"
            );
            eprintln!("{TEST_NAME}: OXIBONSAI_MODELS_DIR is not set -- skipping");
            record_skipped(Capability::LegacyModels, TEST_NAME);
            return;
        }

        let kernel = KernelDispatcher::with_tier(oxibonsai_kernels::cpu_kernel_tier());
        let mut executed = 0usize;
        let models = [
            ("Ternary-Bonsai-1.7B.gguf", GgufTensorType::TQ2_0_g128),
            ("Bonsai-8B.gguf", GgufTensorType::Q1_0_g128),
        ];
        for &(file, expected_quant) in &models {
            let Some(path) = oxibonsai_testkit::workspace::find_model(file) else {
                assert!(
                    !require,
                    "OXI_REQUIRE_MODEL_FILES=1: {file} must be present"
                );
                eprintln!(
                    "{TEST_NAME}: {file} not found under OXIBONSAI_MODELS_DIR -- skipping it"
                );
                continue;
            };
            let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {file}: {e}"));
            let gguf = GgufFile::parse(&mmap).unwrap_or_else(|e| panic!("parse {file}: {e}"));
            let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ)
                .unwrap_or_else(|e| panic!("load {file}: {e}"));
            assert_eq!(model.dominant_quant_type(), expected_quant, "{file}");
            let hidden = model.hidden_size();
            for (label, tokens) in real_texts() {
                let started = std::time::Instant::now();
                let batched = model
                    .forward_hidden(&tokens, &kernel)
                    .unwrap_or_else(|e| panic!("{file} {label}: batched: {e}"));
                let batched_secs = started.elapsed().as_secs_f64();
                let started = std::time::Instant::now();
                let sequential = model
                    .forward_hidden_sequential(&tokens, &kernel)
                    .unwrap_or_else(|e| panic!("{file} {label}: sequential: {e}"));
                let sequential_secs = started.elapsed().as_secs_f64();
                assert_eq!(batched.len(), tokens.len() * hidden);
                assert_eq!(sequential.len(), batched.len());
                let worst_row = batched
                    .chunks_exact(hidden)
                    .zip(sequential.chunks_exact(hidden))
                    .map(|(a, b)| cosine(a, b))
                    .fold(f64::INFINITY, f64::min);
                let pooled_batched =
                    BonsaiModel::mean_pool_normalized(&batched, hidden, tokens.len(), label)
                        .expect("pool batched");
                let pooled_sequential =
                    BonsaiModel::mean_pool_normalized(&sequential, hidden, tokens.len(), label)
                        .expect("pool sequential");
                let pooled_cos = cosine(&pooled_batched, &pooled_sequential);
                eprintln!(
                    "{file} {label}: pooled cos = {pooled_cos:.12}, worst row cos = \
                     {worst_row:.12}; batched {batched_secs:.3} s, sequential \
                     {sequential_secs:.3} s ({:.1}x)",
                    sequential_secs / batched_secs.max(f64::MIN_POSITIVE)
                );
                assert!(
                    pooled_cos >= 0.9999,
                    "{file} {label}: batched pooled embedding diverged from the per-token \
                     reference: cos = {pooled_cos}"
                );
                assert!(
                    batched.iter().all(|v| v.is_finite()),
                    "{file} {label}: non-finite hidden state"
                );
            }
            executed += 1;
        }
        if executed == models.len() {
            record_executed(Capability::LegacyModels, TEST_NAME);
        } else {
            record_skipped(Capability::LegacyModels, TEST_NAME);
        }
    }
}
