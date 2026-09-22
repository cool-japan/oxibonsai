//! Greedy decoding: the GPU-argmax fast path and its penalty-honouring
//! sibling.
//!
//! Split out of [`crate::engine`] to keep that file under the workspace
//! 2000-line ceiling. Everything here is an inherent method of
//! [`InferenceEngine`], so the two files form one API surface.
//!
//! ## The decoding contract (`RT-24`, and the CPU-vs-Metal divergence)
//!
//! [`InferenceEngine::generate_greedy_gpu`] used to be pure argmax: it never
//! consulted `self.sampler`, so a caller that had configured
//! `repetition_penalty: 1.1` (which `SamplingParams::default()` does, and
//! which the CLI hardcoded) got *penalised* greedy on the CPU path and
//! *unpenalised* greedy on the Metal path — measurably different text on 7
//! of 9 real-model runs, at a top-1/top-2 margin ~5000× the cross-backend
//! numerical delta, i.e. an algorithm mismatch, not floating-point noise.
//!
//! It now honours the configured penalties instead of ignoring them:
//!
//! * no penalties configured → GPU argmax, 4 bytes read back per token;
//! * penalties configured → the full logit row is downloaded and the
//!   penalty stage runs *before* a first-index argmax, which is exactly what
//!   the CPU path does.
//!
//! Both arms stop on any id in the engine's EOS set (`RT-18`) and observe the
//! armed [`CancellationToken`](crate::engine_control::CancellationToken)
//! every step (`SV-09`). The CPU-side arms — [`argmax_first`] here and
//! `sampling.rs::argmax` — break ties toward the **first** index (`RT-22`).
//!
//! ## The GPU kernel now shares that tie-break too (`perf-11` / `FIX2-KERN`)
//!
//! An earlier version of this doc claimed the MSL argmax kernel's
//! lowest-`tid` rule was equivalent to first-index tie-breaking. A wave-2
//! verifier review traced the kernel directly and found that was false: it
//! tied toward the lowest *thread id*, and a payload's thread id is not its
//! original array index, so two exactly-equal maxima could resolve to the
//! *higher* index (the traced example: indices `1000` and `2000` under a
//! 1024-wide threadgroup used to resolve to `2000`). Both the MSL kernel
//! (`kernel_sources/utility.rs`'s `argmax`) and its CUDA twin
//! (`cuda_kernels.rs`'s `argmax_f32`) now compare the payload's *original
//! index* on a value-tie, verified by a dedicated kernel-level harness
//! (`crates/oxibonsai-kernels/tests/gpu_argmax_tiebreak.rs`, ≥ 300
//! randomized multi-way ties plus the worked example, all resolving to the
//! minimal tied index). See
//! [`crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`], the gate
//! this module's routing depends on, for the full trace and the fix; it is
//! now `true`, so every entry point in this module that can reach
//! `forward_greedy_gpu` — [`InferenceEngine::greedy_gpu_eligible`]'s four
//! routed callers *and* [`InferenceEngine::generate_greedy_gpu`]'s own
//! direct, unconditional entry point (the CLI's `--temperature 0` fast
//! path, which does not go through `greedy_gpu_eligible` at all) — may take
//! the GPU-argmax fast path, and when a model is not fused for Metal decode
//! (or a Metal dispatch fails mid-generation), the per-token CPU fallback in
//! [`InferenceEngine::greedy_decode_token_with_fallback`] still applies the
//! same first-index rule the GPU kernel now also implements, so the two
//! tiers agree on ties either way.

#[cfg(all(feature = "metal", target_os = "macos"))]
use oxibonsai_kernels::{KernelDispatcher, KernelTier};

use crate::engine::{InferenceEngine, MAX_PREALLOC_TOKENS};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX;
use crate::error::RuntimeResult;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::ngram_cache::NgramCache;

/// Index of the maximum value, ties broken toward the **first** index.
///
/// `RT-22` / §5.4 of the CPU-vs-Metal divergence report: multiple tie-break
/// conventions existed on the CPU side alone (sampler `max_by` → last,
/// engine loops → first). First-index is the one convention; this helper is
/// the engine's single implementation of it, and every CPU decode loop in
/// this module routes through it. The GPU argmax kernel now implements this
/// same convention too (`perf-11` / `FIX2-KERN`; see
/// [`crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`] for the fix
/// and the kernel-level test that verifies it), so this function and the
/// GPU kernel agree on ties rather than the gate existing to route around a
/// permanent mismatch.
///
/// `NaN` never wins: `v > best` is false for `NaN`, so an all-`NaN` slice
/// yields index `0` rather than a panic.
pub fn argmax_first(values: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

impl<'a> InferenceEngine<'a> {
    /// Greedy generation on the GPU (temperature 0, argmax on Metal).
    ///
    /// Runs the full forward pass + argmax in a single GPU command buffer per
    /// token, downloading only the 4-byte token id instead of the full f32
    /// logits vector. On a Metal dispatch failure mid-generation the decode
    /// transparently rebuilds the CPU KV cache and continues on the CPU (see
    /// [`Self::greedy_decode_token_with_fallback`]) rather than emitting a
    /// corrupted continuation.
    ///
    /// ## Penalties are honoured, never ignored (`RT-24`)
    ///
    /// When the engine's sampler has a repetition penalty ≠ 1.0 or an active
    /// frequency/presence penalty, the GPU argmax cannot express the request
    /// — penalties need the whole logit row. This call then decodes through
    /// [`Self::generate_greedy_penalised`], which downloads the full row and
    /// applies the same penalty stage the CPU path uses, rather than
    /// silently dropping the penalties (the defect) or failing the request
    /// (which would break every caller that leaves
    /// `SamplingParams::default()`'s `repetition_penalty: 1.1` in place).
    ///
    /// ## The tie-break gate applies here too (wave-2 verifier follow-up)
    ///
    /// Unlike `generate`/`generate_tracked`/the streaming pair, this is a
    /// **direct, unconditional** entry point: the CLI's `--temperature 0`
    /// fast path (`cmd_run.rs`) calls it without going through
    /// [`InferenceEngine::greedy_gpu_eligible`] at all. It shares the exact
    /// same underlying risk those four routed callers do — a mid-generation
    /// [`Self::greedy_decode_token_with_fallback`] call returns whatever
    /// `forward_greedy_gpu` produces for the *second* and later tokens
    /// without re-deriving it on the CPU — so it consults
    /// [`crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`] itself
    /// rather than relying on a caller to have checked it. If this gate is
    /// ever reverted to `false`, this behaves exactly like the penalty
    /// branch below: the full logit row is decoded on the CPU every step,
    /// which costs the whole point of this method (the 4-byte GPU readback)
    /// but keeps the temperature-0 determinism invariant intact.
    ///
    /// Temperature is deliberately *not* part of the predicate: this method
    /// is the explicit "decode greedily" entry point and callers invoke it to
    /// force greedy regardless of the configured temperature. Path selection
    /// by temperature belongs to the caller (the CLI does it) and to
    /// [`InferenceEngine::greedy_gpu_eligible`] for the automatic routing.
    ///
    /// Returns the generated token ids (prompt excluded, EOS excluded).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[tracing::instrument(skip(self, prompt_tokens), fields(prompt_len = prompt_tokens.len()))]
    pub fn generate_greedy_gpu(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> RuntimeResult<Vec<u32>> {
        if !GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX {
            tracing::info!(
                "generate_greedy_gpu: GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX is false \
                 (the tie-break gate has been reverted, see engine_control.rs), \
                 decoding the full logit row on the CPU every step instead of the \
                 4-byte GPU argmax readback -- see that constant's doc comment"
            );
            return self.generate_greedy_penalised(prompt_tokens, max_tokens, |_| true);
        }
        if self.greedy_penalties_active() {
            let params = self.sampler.params().clone();
            let penalties = *self.sampler.penalties();
            tracing::info!(
                repetition_penalty = params.repetition_penalty,
                frequency_penalty = penalties.frequency_penalty,
                presence_penalty = penalties.presence_penalty,
                "generate_greedy_gpu: penalties are configured, decoding the full logit row \
                 and applying them before the argmax (the GPU argmax cannot honour them)"
            );
            return self.generate_greedy_penalised(prompt_tokens, max_tokens, |_| true);
        }
        self.generate_greedy_gpu_unchecked(prompt_tokens, max_tokens, |_| true)
    }

    /// Whether the configured sampler carries any penalty that a bare argmax
    /// would silently discard.
    pub fn greedy_penalties_active(&self) -> bool {
        self.sampler.params().repetition_penalty != 1.0 || self.sampler.penalties().is_active()
    }

    /// Greedy decode over the **full** logit row with the engine's penalty
    /// stage applied before the argmax.
    ///
    /// The penalised arm of the decoding contract, shared by
    /// [`Self::generate_greedy_gpu`] and available to any caller that must
    /// decode greedily while honouring penalties. Penalties are applied by
    /// [`Sampler::sample_with_history`](crate::sampling::Sampler::sample_with_history)
    /// — the *same* function the CPU streaming path uses — with the sampler
    /// temporarily pinned to temperature 0 so the selection is a first-index
    /// argmax and never consumes RNG. Parameters are restored on exit,
    /// including on the error path.
    ///
    /// `emit` is called for every committed token and returns `false` to
    /// stop (a dropped stream receiver).
    pub fn generate_greedy_penalised<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        mut emit: F,
    ) -> RuntimeResult<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Ok(vec![]);
        }

        let Some(mut logits) = self.prefill_for_generate(prompt_tokens)? else {
            return Ok(vec![]);
        };

        let prev_params = self.sampler.params().clone();
        let mut greedy_params = prev_params.clone();
        greedy_params.temperature = 0.0;
        self.sampler.set_params(greedy_params);

        let decode_start = std::time::Instant::now();
        let mut output_tokens = Vec::with_capacity(max_tokens.min(MAX_PREALLOC_TOKENS));
        let result = (|| -> RuntimeResult<()> {
            for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
                let step_start = std::time::Instant::now();
                if self.is_cancelled() {
                    tracing::debug!(pos, "penalised greedy generation cancelled");
                    break;
                }
                let next_token = self.sampler.sample_with_history(&logits, &output_tokens)?;
                if self.is_eos(next_token) {
                    tracing::debug!(pos, "EOS token generated (penalised greedy)");
                    break;
                }
                if !emit(next_token) {
                    tracing::debug!(pos, "receiver dropped, stopping generation");
                    break;
                }
                output_tokens.push(next_token);
                logits = self.model.forward(next_token, pos, &self.kernel)?;
                if let Some(m) = &self.metrics {
                    m.decode_token_duration_seconds
                        .observe(step_start.elapsed().as_secs_f64());
                }
            }
            Ok(())
        })();
        self.sampler.set_params(prev_params);
        result?;

        self.record_decode_metrics(decode_start, output_tokens.len());
        self.stats.record_request(output_tokens.len());
        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated = output_tokens.len(),
            "penalised greedy generation complete"
        );
        Ok(output_tokens)
    }

    /// Record the per-generation tokens/second, token-count and memory
    /// gauges, shared by every decode loop in this module.
    pub(crate) fn record_decode_metrics(&self, decode_start: std::time::Instant, generated: usize) {
        if let Some(m) = &self.metrics {
            let decode_elapsed = decode_start.elapsed().as_secs_f64();
            if decode_elapsed > 0.0 && generated > 0 {
                m.tokens_per_second
                    .observe(generated as f64 / decode_elapsed);
            }
            m.tokens_generated_total.inc_by(generated as u64);
            m.update_memory_from_rss();
        }
    }

    /// Decode one greedy token, preferring the Metal GPU path and falling back
    /// to a **coherent** CPU forward when the GPU path fails (or is disabled via
    /// `force_cpu`).
    ///
    /// The Metal decode path (`BonsaiModel::forward_greedy_gpu`) maintains only
    /// the GPU-resident KV cache and never writes `self.model`'s CPU cache, so a
    /// naive CPU `forward()` after a GPU failure would attend over an all-zero
    /// cache and silently corrupt the continuation. The first time we fall through
    /// to the CPU, this rebuilds the CPU KV cache by replaying the committed token
    /// sequence (`committed`, covering positions `0..committed.len()`) through the
    /// scalar-CPU forward, then latches `cpu_fallback_active` so every subsequent
    /// token decodes on the CPU directly and never reads the now-stale GPU cache.
    ///
    /// `cpu_kernel` MUST be a non-GPU dispatcher: `self.kernel` may be
    /// GPU-accelerated, in which case `BonsaiModel::forward` would re-enter the
    /// Metal path and leave the CPU cache empty. A `KernelTier::Reference`
    /// dispatcher forces the CPU block path and is byte-identical to the canonical
    /// CPU reference (see the cross-backend determinism guard).
    ///
    /// Returns the greedy argmax token id. On the CPU fallback branch this is
    /// [`argmax_first`]'s first-index tie-break; on the GPU branch it is
    /// whatever `forward_greedy_gpu` returns, which now agrees with that same
    /// first-index rule on a tie (`perf-11` / `FIX2-KERN` fixed both the MSL
    /// and CUDA argmax kernels; see
    /// [`crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`] for the
    /// fix and the kernel-level test that verifies it) — so a caller reached
    /// through either [`InferenceEngine::greedy_gpu_eligible`]'s routed
    /// entry points or `generate_greedy_gpu`'s direct one gets the same
    /// tie-break regardless of which branch below actually answers.
    ///
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn greedy_decode_token_with_fallback(
        &mut self,
        committed: &[u32],
        next_token: u32,
        pos: usize,
        cpu_kernel: &KernelDispatcher,
        cpu_fallback_active: &mut bool,
        force_cpu: bool,
    ) -> RuntimeResult<u32> {
        if !*cpu_fallback_active && !force_cpu {
            match self.model.forward_greedy_gpu(next_token, pos - 1) {
                Ok(token_id) => return Ok(token_id),
                Err(e) => {
                    tracing::warn!(
                        error = %e, pos,
                        "Metal greedy GPU decode failed; rebuilding the CPU KV cache from the \
                         committed sequence and continuing on the CPU"
                    );
                }
            }
        }
        let logits = if *cpu_fallback_active {
            // Cache already coherent from an earlier rebuild — normal CPU forward.
            self.model.forward(next_token, pos - 1, cpu_kernel)?
        } else {
            // First CPU fall-through: reconstruct the CPU KV cache from scratch by
            // replaying the committed tokens (positions 0..committed.len()) — this
            // reproduces exactly the cache a pure-CPU generation would have built.
            // The final replayed forward yields the logits for the next token.
            if force_cpu {
                tracing::warn!(
                    pos,
                    committed = committed.len(),
                    "forcing CPU greedy decode; rebuilding the CPU KV cache from the committed sequence"
                );
            }
            self.model.reset();
            let mut last = Vec::new();
            for (p, &tok) in committed.iter().enumerate() {
                last = self.model.forward(tok, p, cpu_kernel)?;
            }
            *cpu_fallback_active = true;
            last
        };
        Ok(argmax_first(&logits))
    }

    /// The GPU-argmax greedy decode loop.
    ///
    /// `unchecked` in the sense that it does **not** re-derive eligibility:
    /// callers have either asked for greedy explicitly
    /// ([`Self::generate_greedy_gpu`], after routing penalised requests away)
    /// or gone through [`InferenceEngine::greedy_gpu_eligible`]. It is
    /// `pub(crate)` so `generate`, `generate_tracked` and both streaming
    /// entry points share this one implementation (`perf-11`) instead of
    /// growing their own.
    ///
    /// `emit` is invoked for every committed token *before* it is appended;
    /// returning `false` stops generation (a dropped stream receiver). The
    /// returned vector holds exactly the emitted tokens.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn generate_greedy_gpu_unchecked<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        mut emit: F,
    ) -> RuntimeResult<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Ok(vec![]);
        }

        // ═══════════════════════════════════════════════════════
        // 1. Prefill: batch process all prompt tokens
        // ═══════════════════════════════════════════════════════
        let Some(last_logits) = self.prefill_for_generate(prompt_tokens)? else {
            return Ok(vec![]);
        };

        // First decode token: argmax from prefill logits (first-index ties).
        let first_token = argmax_first(&last_logits);

        // ═══════════════════════════════════════════════════════
        // 2. Decode: greedy, optionally with n-gram drafting
        // ═══════════════════════════════════════════════════════
        let decode_start = std::time::Instant::now();
        let mut output_tokens = Vec::with_capacity(max_tokens.min(MAX_PREALLOC_TOKENS));

        if max_tokens == 0 || self.is_eos(first_token) {
            self.stats.record_request(0);
            return Ok(vec![]);
        }
        if !emit(first_token) {
            self.stats.record_request(0);
            return Ok(vec![]);
        }
        output_tokens.push(first_token);

        // N-gram cache for zero-cost draft generation
        let mut ngram_cache = NgramCache::new();
        ngram_cache.record(prompt_tokens);

        // Running context: prompt + generated tokens (for n-gram lookups)
        let mut context: Vec<u32> = prompt_tokens.to_vec();
        context.push(first_token);

        // `RT-27`: speculation is a documented engine configuration read once
        // here, not an undocumented environment variable read inside the loop.
        let spec = self.speculative;
        let speculation_k = spec.draft_len;
        let mut spec_attempts: u64 = 0;
        let mut spec_accepted_total: u64 = 0;

        // Metal→CPU fallback machinery. `forward_greedy_gpu` maintains only the
        // GPU-resident KV cache; if it fails mid-generation we must NOT continue
        // on the CPU with the all-zero CPU cache (silent corruption). Instead we
        // rebuild the CPU cache from the committed sequence once, then decode the
        // remainder on the CPU. `cpu_fallback_kernel` is pinned to the scalar CPU
        // reference so `BonsaiModel::forward` takes the CPU block path (a
        // GPU-accelerated `self.kernel` would re-enter Metal and never populate
        // the CPU cache).
        let cpu_fallback_kernel = KernelDispatcher::with_tier(KernelTier::Reference);
        let mut cpu_fallback_active = false;
        // Optional debug / test seam: after this many committed tokens, force the
        // remainder of the decode onto the CPU path (exercises the KV-cache
        // rebuild without a real GPU fault). Resolved once (already applied
        // from the legacy `OXIBONSAI_FORCE_CPU_DECODE_AFTER` env var at
        // construction, via `SpeculativeConfig::with_env_override` — a
        // wave-2 verifier finding folded this out of a raw per-call env read
        // and into the documented config alongside `RT-27`'s own variable).
        let force_cpu_after: Option<usize> = spec.force_cpu_decode_after;

        let mut next_token = first_token;
        let mut pos = prompt_tokens.len() + 1;
        let max_pos = prompt_tokens.len() + max_tokens;
        let mut stopped = false;

        while pos < max_pos && output_tokens.len() < max_tokens && !stopped {
            let step_start = std::time::Instant::now();
            // `SV-09`: cooperative cancellation, once per decode step.
            if self.is_cancelled() {
                tracing::debug!(pos, "greedy GPU generation cancelled");
                break;
            }
            let tokens_generated = output_tokens.len();
            let force_cpu = force_cpu_after.is_some_and(|k| tokens_generated >= k);

            // Try n-gram draft — skip the warmup phase unless explicitly enabled.
            // Speculation relies on the GPU KV cache, so it is disabled once we
            // fall back to (or are forced onto) the CPU decode path.
            let draft = if !spec.is_enabled()
                || cpu_fallback_active
                || force_cpu
                || tokens_generated < spec.warmup_tokens
            {
                Vec::new()
            } else {
                ngram_cache.draft(&context, speculation_k)
            };

            // Adaptive: only speculate while the measured accept rate justifies
            // the batched verify (a draft of k costs ~k× a single token).
            let spec_ok = spec.should_attempt(spec_attempts, spec_accepted_total);

            if !draft.is_empty() && spec_ok {
                // ── Speculative path: batch verify ──────────────
                let mut batch = Vec::with_capacity(1 + draft.len());
                batch.push(next_token);
                batch.extend_from_slice(&draft);

                match self
                    .model
                    .forward_prefill_verify(&batch, pos - 1, &self.kernel)
                {
                    Ok(model_preds) => {
                        spec_attempts += 1;

                        // Verify draft against model predictions
                        let mut accepted: usize = 0;
                        for i in 0..draft.len() {
                            if i < model_preds.len() && draft[i] == model_preds[i] {
                                accepted += 1;
                            } else {
                                break;
                            }
                        }
                        spec_accepted_total += accepted as u64;

                        // Collect accepted draft tokens + bonus
                        let mut eos_seen = false;
                        for &token in draft.iter().take(accepted) {
                            if self.is_eos(token) {
                                eos_seen = true;
                                break;
                            }
                            if !emit(token) {
                                stopped = true;
                                break;
                            }
                            output_tokens.push(token);
                            context.push(token);
                        }

                        if stopped {
                            break;
                        }

                        if !eos_seen {
                            // Bonus: model's prediction at the accept/reject boundary
                            let bonus = if accepted < model_preds.len() {
                                model_preds[accepted]
                            } else {
                                // All draft tokens matched, take the last prediction
                                match model_preds.last() {
                                    Some(&tok) => tok,
                                    None => break,
                                }
                            };

                            if self.is_eos(bonus) {
                                tracing::debug!(pos, accepted, "EOS from speculative bonus");
                                break;
                            }

                            if !emit(bonus) {
                                break;
                            }
                            output_tokens.push(bonus);
                            context.push(bonus);
                            next_token = bonus;
                            pos += accepted + 1;

                            // Update n-gram cache with the newly accepted window
                            let window_start = context.len().saturating_sub(accepted + 4);
                            ngram_cache.record(&context[window_start..]);
                        } else {
                            tracing::debug!(pos, accepted, "EOS in draft tokens");
                            break;
                        }
                    }
                    Err(_e) => {
                        // Speculative verify failed — fall through to single-token
                        // decode (GPU, or a coherent CPU fallback).
                        tracing::debug!("speculative verify failed, using single-token decode");
                        let tok = self.greedy_decode_token_with_fallback(
                            &context,
                            next_token,
                            pos,
                            &cpu_fallback_kernel,
                            &mut cpu_fallback_active,
                            force_cpu,
                        )?;
                        if self.is_eos(tok) {
                            tracing::debug!(pos, "EOS token generated");
                            break;
                        }
                        if !emit(tok) {
                            break;
                        }
                        output_tokens.push(tok);
                        context.push(tok);
                        let window_start = context.len().saturating_sub(3);
                        ngram_cache.record(&context[window_start..]);
                        next_token = tok;
                        pos += 1;
                    }
                }
            } else {
                // ── Single-token decode (GPU, or a coherent CPU fallback) ──
                let tok = self.greedy_decode_token_with_fallback(
                    &context,
                    next_token,
                    pos,
                    &cpu_fallback_kernel,
                    &mut cpu_fallback_active,
                    force_cpu,
                )?;
                if self.is_eos(tok) {
                    tracing::debug!(pos, "EOS token generated");
                    break;
                }
                if !emit(tok) {
                    break;
                }
                output_tokens.push(tok);
                context.push(tok);
                let window_start = context.len().saturating_sub(3);
                ngram_cache.record(&context[window_start..]);
                next_token = tok;
                pos += 1;
            }

            if let Some(m) = &self.metrics {
                m.decode_token_duration_seconds
                    .observe(step_start.elapsed().as_secs_f64());
            }
        }

        // Log speculative decode statistics
        if spec_attempts > 0 {
            let avg_accepted = spec_accepted_total as f64 / spec_attempts as f64;
            let accuracy =
                spec_accepted_total as f64 / (spec_attempts as f64 * speculation_k as f64).max(1.0);
            tracing::info!(
                spec_attempts,
                spec_accepted_total,
                avg_accepted = format!("{:.2}", avg_accepted),
                accuracy = format!("{:.1}%", accuracy * 100.0),
                "speculative decode stats"
            );
        }

        self.record_decode_metrics(decode_start, output_tokens.len());
        self.stats.record_request(output_tokens.len());

        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated = output_tokens.len(),
            "greedy GPU generation complete"
        );

        Ok(output_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::argmax_first;
    use crate::engine::InferenceEngine;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX;
    use crate::sampling::{PenaltyParams, SamplingParams};
    use oxibonsai_core::config::Qwen3Config;

    fn greedy_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 16,
        }
    }

    /// §5.4 of the divergence report: every *CPU-side* argmax in the engine
    /// must break ties toward the **first** index — the sampler's convention
    /// (`RT-22`). The MSL/CUDA kernels now share this convention too
    /// (`GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`, `perf-11` / `FIX2-KERN`), and
    /// this helper is the CPU-side reference both tiers agree with.
    #[test]
    fn argmax_first_breaks_ties_toward_the_first_index() {
        assert_eq!(argmax_first(&[1.0, 1.0, 1.0]), 0);
        assert_eq!(argmax_first(&[0.0, 5.0, 5.0, 5.0, 1.0]), 1);
        assert_eq!(argmax_first(&[-1.0, -1.0]), 0);
    }

    #[test]
    fn argmax_first_agrees_with_the_sampler_on_ties() {
        // The sampler is the reference implementation of the convention; a
        // temperature-0 sample is a pure argmax.
        let logits = vec![0.5, 9.0, 9.0, 2.0, 9.0];
        let mut sampler = crate::sampling::Sampler::new(greedy_params(), 7);
        let sampled = sampler.sample(&logits).expect("greedy sample");
        assert_eq!(
            argmax_first(&logits),
            sampled,
            "engine argmax and sampler argmax must agree on a tie"
        );
    }

    #[test]
    fn argmax_first_is_nan_safe_and_never_panics() {
        assert_eq!(argmax_first(&[]), 0);
        assert_eq!(argmax_first(&[f32::NAN, f32::NAN]), 0);
        // A NaN after the true maximum must not displace it.
        assert_eq!(argmax_first(&[1.0, 4.0, f32::NAN, 2.0]), 1);
        assert_eq!(argmax_first(&[f32::NEG_INFINITY; 3]), 0);
    }

    /// The `RT-24` predicate: a configured penalty must make the engine
    /// refuse the argmax-only route. Checked on a synthetic engine, which is
    /// never fused-GPU-capable, so the predicate's penalty terms are the
    /// interesting part — see `greedy_gpu_eligible`'s doc for the full set.
    #[test]
    fn greedy_routing_is_refused_when_penalties_are_configured() {
        let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
        // Synthetic engines never take the fused route.
        assert!(!engine.uses_fused_gpu_decode());
        assert!(!engine.greedy_gpu_eligible(false));

        // Even with penalties off, logprobs must never take it.
        assert!(!engine.greedy_gpu_eligible(true));

        engine.set_penalties(PenaltyParams::new(0.5, 0.0));
        assert!(!engine.greedy_gpu_eligible(false));
    }

    /// `perf-11` / `FIX2-KERN` follow-up: `generate_greedy_gpu` is a
    /// *direct*, unconditional entry point -- unlike
    /// `generate`/`generate_tracked`/the streaming pair, it never consults
    /// `greedy_gpu_eligible`, so it checks
    /// [`GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`] itself. Now that the gate is
    /// `true`, calling it on a *synthetic* (non-fused) engine no longer
    /// short-circuits straight to [`InferenceEngine::generate_greedy_penalised`]:
    /// it proceeds into [`InferenceEngine::generate_greedy_gpu_unchecked`],
    /// which attempts [`BonsaiModel::forward_greedy_gpu`] every step through
    /// [`Self::greedy_decode_token_with_fallback`]. A `tiny_test()` model
    /// built via `BonsaiModel::new` has no fused Metal graph (its LM head is
    /// the unquantized `OutputWeight::Fp32` variant, which
    /// `forward_greedy_gpu` explicitly refuses), so that call errors on
    /// *every* step and the fallback machinery transparently rebuilds the
    /// CPU KV cache and decodes on `KernelTier::Reference` from then on --
    /// this must still be token-identical to calling the explicit CPU path
    /// directly, which is exactly the regression this test guards now that
    /// the "closed gate" shortcut is gone. Both engines are pinned to
    /// `KernelTier::Reference` (rather than `InferenceEngine::new`'s
    /// `auto_detect`) so the comparison is deterministic on any host,
    /// Metal-equipped or not: `generate_greedy_gpu`'s own prefill and its
    /// internal CPU fallback would otherwise use two independently-chosen
    /// kernel tiers whose floating-point reductions are not proven bit-exact
    /// against each other (only exact here because a weightless model's
    /// logits are identically `0.0`, which pinning removes any dependence
    /// on).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn generate_greedy_gpu_falls_back_to_cpu_and_matches_the_explicit_cpu_path_when_unfused() {
        use oxibonsai_kernels::KernelTier;
        use oxibonsai_model::model::BonsaiModel;

        // Compile-time, not runtime (clippy: `assert!` on a `const` is
        // pointless as a test): this test's premise -- see the constant's
        // doc comment before touching either.
        const {
            assert!(GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX);
        }
        let prompt = [1u32, 2, 3];
        let mut via_gpu_entry = InferenceEngine::from_model_with_tier(
            BonsaiModel::new(Qwen3Config::tiny_test()),
            KernelTier::Reference,
            greedy_params(),
            42,
        );
        let mut via_cpu_entry = InferenceEngine::from_model_with_tier(
            BonsaiModel::new(Qwen3Config::tiny_test()),
            KernelTier::Reference,
            greedy_params(),
            42,
        );
        assert!(
            !via_gpu_entry.uses_fused_gpu_decode(),
            "test premise: a synthetic model must not present as fused, so \
             generate_greedy_gpu_unchecked's internal forward_greedy_gpu call \
             is exercised on every step and must fail over to the CPU path"
        );

        let gpu_entry_tokens = via_gpu_entry
            .generate_greedy_gpu(&prompt, 8)
            .expect("generate_greedy_gpu");
        let cpu_entry_tokens = via_cpu_entry
            .generate_greedy_penalised(&prompt, 8, |_| true)
            .expect("generate_greedy_penalised");

        assert_eq!(
            gpu_entry_tokens, cpu_entry_tokens,
            "with the tie-break gate open but the model unfused, \
             generate_greedy_gpu's per-token CPU fallback must still match \
             the explicit CPU-argmax path token-for-token"
        );
    }

    /// Sibling of the above: a configured penalty must still force the
    /// penalised CPU path even with the tie-break gate open -- the gate and
    /// the penalty check are independent conditions, and opening one must
    /// not silently bypass the other. This is the "flipping \[the gate\]
    /// off still routes correctly" guarantee from the other direction: the
    /// penalised branch `generate_greedy_gpu` falls back to when
    /// [`InferenceEngine::greedy_penalties_active`] is `true` is the exact
    /// same code path it used unconditionally while the gate was closed, so
    /// proving it still runs correctly here is proving that a future revert
    /// of the gate would still have a working fallback to land on.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn generate_greedy_gpu_still_honours_penalties_with_the_gate_open() {
        const {
            assert!(GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX);
        }
        let prompt = [1u32, 2, 3];
        let params = greedy_params();
        let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), params, 42);
        engine.set_penalties(PenaltyParams::new(0.5, 0.0));
        assert!(engine.greedy_penalties_active());

        let via_penalised_entry = engine
            .generate_greedy_gpu(&prompt, 8)
            .expect("generate_greedy_gpu");
        let mut reference = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
        reference.set_penalties(PenaltyParams::new(0.5, 0.0));
        let via_explicit_entry = reference
            .generate_greedy_penalised(&prompt, 8, |_| true)
            .expect("generate_greedy_penalised");

        assert_eq!(
            via_penalised_entry, via_explicit_entry,
            "a configured penalty must route generate_greedy_gpu through the \
             penalised CPU path regardless of the tie-break gate's value"
        );
    }
}
