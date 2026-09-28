//! Decoding on the fused Metal route: greedy (the GPU-argmax fast path and
//! its penalty-honouring sibling) and sampled (the top-k candidate route).
//!
//! Split out of [`crate::engine`] to keep that file under the workspace
//! 2000-line ceiling. Everything here is an inherent method of
//! [`InferenceEngine`], so the two files form one API surface.
//!
//! ## The decoding contract (`RT-24`, and the CPU-vs-Metal divergence)
//!
//! [`InferenceEngine::generate_greedy_gpu`] used to be pure argmax: it never
//! consulted `self.sampler`, so a caller that had configured
//! `repetition_penalty: 1.1` (as `SamplingParams::default()` and the CLI both
//! did at the time) got *penalised* greedy on the CPU path and
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
//! ## The GPU kernel shares that tie-break (`perf-11`)
//!
//! The MSL argmax kernel once tied toward the lowest *thread id*, and a
//! payload's thread id is not its original array index, so two
//! exactly-equal maxima could resolve to the *higher* index (the traced
//! example: indices `1000` and `2000` under a 1024-wide threadgroup
//! resolved to `2000`). Both the MSL kernel (`kernel_sources/utility.rs`'s
//! `argmax`) and its CUDA twin (`cuda_kernels.rs`'s `argmax_f32`) compare
//! the payload's *original index* on a value-tie, verified by a dedicated
//! kernel-level harness (`crates/oxibonsai-kernels/tests/gpu_argmax_tiebreak.rs`,
//! ≥ 300 randomized multi-way ties plus the worked example, all resolving
//! to the minimal tied index). See
//! [`crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`], the gate
//! this module's routing depends on, for the full trace; it is `true`, so
//! every entry point in this module that can reach `forward_greedy_gpu` —
//! [`InferenceEngine::greedy_gpu_eligible`]'s four routed callers *and*
//! [`InferenceEngine::generate_greedy_gpu`]'s own direct, unconditional
//! entry point (the CLI's `--temperature 0` fast path, which does not go
//! through `greedy_gpu_eligible` at all) — may take the GPU-argmax fast
//! path, and when a model is not fused for Metal decode (or a Metal
//! dispatch fails mid-generation), the per-token CPU fallback in
//! `InferenceEngine::greedy_decode_token_with_fallback` applies the same
//! first-index rule the GPU kernel implements, so the two tiers agree on
//! ties either way.
//!
//! ## The sampled top-k route (`perf-11`, sampled half)
//!
//! A sampled request on the fused route downloads only the top-`k`
//! `(id, logit)` candidates of each decode step's resident logit row
//! instead of the whole row — on by default ([`SampledTopKConfig`]). The
//! engine's own sampler draws over the candidate sub-row; because the
//! sampler ranks and walks its top-k survivors in a canonical order that
//! depends only on the survivor set (`crate::sampling`'s module docs), and
//! the candidates contain every survivor in that order, a seeded draw picks
//! exactly the token the full-row draw would. Requests the candidates
//! cannot serve exactly (a penalty, `top_k` of `0` or above the candidate
//! count, log-probabilities) decode the full row, and every such request
//! or step is counted ([`crate::engine::EngineStats`] and, when attached,
//! the Prometheus counters).

#[cfg(all(feature = "metal", target_os = "macos"))]
use oxibonsai_kernels::{KernelDispatcher, KernelTier};
#[cfg(all(feature = "metal", target_os = "macos"))]
use oxibonsai_model::hybrid::LoadedModel;

use crate::engine::{InferenceEngine, GREEDY_TEMPERATURE_EPS, MAX_PREALLOC_TOKENS};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX;
use crate::error::RuntimeResult;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::ngram_cache::NgramCache;

/// Default number of top-k candidates a sampled request on the fused GPU
/// route downloads per token instead of the full logit row (`perf-11`).
pub const DEFAULT_SAMPLED_TOPK_CANDIDATES: usize = 64;

/// Where sampled decode on the fused GPU route gets its candidates from
/// (`perf-11`, sampled half).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SampledTopKMode {
    /// Route disabled: a sampled request decodes every step's full logit row
    /// through the classic sampler (`Sampler::sample_with_history`). Its
    /// output is byte-identical to [`Self::GpuCandidates`]'s; it only costs
    /// the full download.
    Off,
    /// **The default**: download only the top-`k` `(id, logit)` pairs of each
    /// decode step's logit row (the GPU `topk_f32` kernel over the resident
    /// logits) and draw the step's token from them — byte-identical to the
    /// classic full-row draw (see [`SampledTopKConfig`]).
    #[default]
    GpuCandidates,
    /// Download the full row and extract the very same top-`k` on the CPU,
    /// then draw from them exactly as [`Self::GpuCandidates`] does: the
    /// **reference** the GPU candidate download is proven bit-for-bit equal
    /// to. Costs the full download; for verification.
    FullRowCandidates,
}

/// Configuration of the sampled top-k route (`perf-11`, sampled half).
///
/// # Eligibility
///
/// With the route enabled, a sampled request (temperature above zero) on the
/// fused GPU route takes it when no penalty is configured (a penalty must see
/// every logit), the caller does not need the full row (log-probabilities),
/// and the sampler's `top_k` is in `1..=candidates` **and** below the
/// vocabulary. Then the sampler's whole support lies inside the downloaded
/// candidates — temperature, `top_k`, then `min_p`/`top_p` over the
/// `top_k`-renormalised distribution need no tail mass — and both the full
/// row and the candidate sub-row are narrowed by a real top-`k` selection.
/// Anything else decodes the full row and is counted in
/// [`EngineStats::sampled_full_row_requests`](crate::engine::EngineStats::sampled_full_row_requests):
/// a `top_k` of `0` (top-p over the whole vocabulary, whose softmax needs
/// every logit — a GPU tail-mass sum could not be bit-identical to the host
/// sampler's sequential `f32` sum, so the route does not try), a `top_k`
/// above the candidate count or at/above the vocabulary, a penalty, or the
/// route disabled.
///
/// # Default: [`SampledTopKMode::GpuCandidates`] — byte-identical to the classic sampler
///
/// `Sampler::sample_core` selects its `top_k` survivors and walks them — the
/// softmax sum, min-p/top-p, the weighted draw — in a canonical order: raw
/// logit descending, `NaN` last, an exact tie to the lower index (top-p as a
/// prefix scan of that order). That order depends only on the survivor set,
/// and the GPU candidates are the top `candidates` logits in exactly that
/// order with `candidates >= top_k`, so they contain the survivors: a draw
/// over the candidate sub-row consumes the same single random draw and picks
/// the same token as the classic draw over the full row — min-p included,
/// which the engine's sampler applies within the survivors
/// ([`InferenceEngine::set_min_p`]). A seeded sampled request therefore
/// produces the same tokens with the route on or off (tested on the ternary
/// and the 1-bit fused fixtures across `top_k`, `top_p`, min-p, temperature
/// and seed, and on the real 1.7B, against the route switched off and an
/// independently spelled-out classic loop), so switching it off
/// ([`SampledTopKMode::Off`] through [`InferenceEngine::set_sampled_topk`])
/// only changes what each decode step downloads, never the output.
///
/// # What the route guarantees
///
/// * Every step whose full row is on the host anyway — the prefill row, and
///   every fallback step — is drawn by the classic sampler over that full
///   row, exactly as under [`SampledTopKMode::Off`].
/// * A candidate step draws with the engine's own sampler over the candidate
///   sub-row (descending logit, an exact tie to the lower id), consuming
///   exactly one random draw as the classic sampler does, and picking the
///   same token. The GPU download is bit-identical to the CPU extraction of
///   the same row
///   ([`SampledTopKMode::FullRowCandidates`]; tested on a fused fixture and
///   on the real 1.7B).
/// * Candidates that cannot stand in for the row — a non-finite value inside
///   the sampler's `top_k`, a GPU top-k / fused-argmax disagreement, a failed
///   download — fall back to the resident full row, and a failed fused
///   forward to a coherent CPU replay; both are counted in
///   [`EngineStats::sampled_topk_full_row_steps`](crate::engine::EngineStats::sampled_topk_full_row_steps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampledTopKConfig {
    /// Where the candidates come from.
    pub mode: SampledTopKMode,
    /// How many candidates to download per token (clamped to
    /// `1..=MAX_RESIDENT_TOPK` and to the vocabulary).
    pub candidates: usize,
}

impl Default for SampledTopKConfig {
    /// The route on ([`SampledTopKMode::GpuCandidates`]) with
    /// [`DEFAULT_SAMPLED_TOPK_CANDIDATES`] candidates per decode step.
    fn default() -> Self {
        Self {
            mode: SampledTopKMode::GpuCandidates,
            candidates: DEFAULT_SAMPLED_TOPK_CANDIDATES,
        }
    }
}

impl SampledTopKConfig {
    /// [`SampledTopKMode::GpuCandidates`] with
    /// [`DEFAULT_SAMPLED_TOPK_CANDIDATES`] candidates per decode step (the
    /// default configuration, spelled out).
    #[must_use]
    pub fn gpu_candidates() -> Self {
        Self {
            mode: SampledTopKMode::GpuCandidates,
            ..Self::default()
        }
    }

    /// The candidate count actually used for a `vocab`-wide logit row.
    #[must_use]
    pub fn effective_candidates(&self, vocab: usize) -> usize {
        self.candidates
            .clamp(1, oxibonsai_kernels::gpu_backend::MAX_RESIDENT_TOPK)
            .min(vocab.max(1))
    }
}

/// The top-`k` `(id, logit)` candidates of a full logit row, in exactly the
/// order the GPU `topk_f32` kernel produces them: descending logit, an exact
/// tie to the **smaller** id, `NaN` and `-inf` never selected, unfilled slots
/// padded with `(0, -inf)`.
///
/// This is the CPU half of the sampled top-k route's byte-identity: sampling
/// this sub-row and sampling the GPU's downloaded one are the same
/// computation on the same numbers.
pub fn top_k_candidates(row: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut ranked: Vec<(u32, f32)> = row
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > f32::NEG_INFINITY)
        .map(|(i, v)| (u32::try_from(i).unwrap_or(u32::MAX), *v))
        .collect();
    let order = |a: &(u32, f32), b: &(u32, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    };
    if k < ranked.len() {
        ranked.select_nth_unstable_by(k, order);
        ranked.truncate(k);
    }
    ranked.sort_unstable_by(order);
    let mut ids = Vec::with_capacity(k);
    let mut values = Vec::with_capacity(k);
    for (id, value) in ranked.into_iter().take(k) {
        ids.push(id);
        values.push(value);
    }
    while ids.len() < k {
        ids.push(0);
        values.push(f32::NEG_INFINITY);
    }
    (ids, values)
}

/// Index of the maximum value, ties broken toward the **first** index.
///
/// `RT-22` / §5.4 of the CPU-vs-Metal divergence report: multiple tie-break
/// conventions existed on the CPU side alone (sampler `max_by` → last,
/// engine loops → first). First-index is the one convention; this helper is
/// the engine's single implementation of it, and every CPU decode loop in
/// this module routes through it. The GPU argmax kernel implements this
/// same convention (`perf-11`; see
/// [`crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`] for the
/// kernel-level test that verifies it), so this function and the GPU kernel
/// agree on ties rather than the gate existing to route around a permanent
/// mismatch.
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
    /// `Self::greedy_decode_token_with_fallback`) rather than emitting a
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
    /// (which would break every caller that configures a repetition
    /// penalty and asks for greedy output).
    ///
    /// ## The tie-break gate applies here too
    ///
    /// Unlike `generate`/`generate_tracked`/the streaming pair, this is a
    /// **direct, unconditional** entry point: the CLI's `--temperature 0`
    /// fast path (`cmd_run.rs`) calls it without going through
    /// [`InferenceEngine::greedy_gpu_eligible`] at all. It shares the exact
    /// same underlying risk those four routed callers do — a mid-generation
    /// `Self::greedy_decode_token_with_fallback` call returns whatever
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
        if self.is_hybrid() {
            // No hybrid GPU encoder exists yet (waves 5+): the explicit
            // greedy entry point decodes the full logit row on the CPU with
            // a first-index argmax -- the same answer, not a pretend GPU run.
            tracing::info!(
                "generate_greedy_gpu: hybrid model has no fused GPU decode path; decoding \
                 greedily on the CPU"
            );
            return self.generate_greedy_penalised(prompt_tokens, max_tokens, |_| true);
        }
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
                logits = self.forward_logits(next_token, pos)?;
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
    /// whatever `forward_greedy_gpu` returns, which agrees with that same
    /// first-index rule on a tie (`perf-11`: both the MSL and CUDA argmax
    /// kernels compare original indices on a value tie; see
    /// [`crate::engine_control::GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`] for the
    /// kernel-level test that verifies it) — so a caller reached
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
        // Only a dense model has a fused GPU greedy decode; a hybrid engine
        // (never routed here -- see `generate_greedy_gpu`) goes straight to
        // the CPU arm below.
        let gpu_attempt = match &self.model {
            LoadedModel::Dense(model) if !*cpu_fallback_active && !force_cpu => {
                Some(model.forward_greedy_gpu(next_token, pos - 1))
            }
            _ => None,
        };
        if let Some(attempt) = gpu_attempt {
            match attempt {
                Ok(token_id) => return Ok(token_id),
                // The DESIGNED signal (MET-05): the fused GPU path kept its
                // own device KV cache, so the host cache holds no history
                // and the CPU must replay the committed prefix first. This
                // is expected control flow, not a failure.
                Err(e)
                    if e.downcast_ref::<oxibonsai_model::error::ModelError>()
                        .and_then(oxibonsai_model::model::gpu_fallback_cache_rebuild_pos)
                        .is_some() =>
                {
                    let rebuild_pos = e
                        .downcast_ref::<oxibonsai_model::error::ModelError>()
                        .and_then(oxibonsai_model::model::gpu_fallback_cache_rebuild_pos);
                    tracing::warn!(
                        error = %e, pos, ?rebuild_pos,
                        "GPU decode requires a host KV-cache rebuild (MET-05); replaying the \
                         committed sequence on the CPU"
                    );
                }
                // Anything else is a GENUINE Metal failure. It is still
                // recovered from -- dropping the request would be worse --
                // but at `error` level with its stable code, so it is
                // distinguishable from the designed signal above instead of
                // being swallowed into the same `warn!`.
                Err(e) => {
                    let code = e
                        .downcast_ref::<oxibonsai_model::error::ModelError>()
                        .map_or(
                            "NON_MODEL_ERROR",
                            oxibonsai_model::error::ModelError::error_code,
                        );
                    tracing::error!(
                        error = %e,
                        code,
                        pos,
                        "Metal greedy GPU decode FAILED (not the MET-05 cache-rebuild signal); \
                         rebuilding the CPU KV cache and continuing on the CPU -- this is a \
                         real GPU error, not expected control flow"
                    );
                }
            }
        }
        let logits = if *cpu_fallback_active {
            // Cache already coherent from an earlier rebuild — normal CPU forward.
            self.forward_logits_on(next_token, pos - 1, cpu_kernel)?
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
                last = self.forward_logits_on(tok, p, cpu_kernel)?;
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
        // rebuild without a real GPU fault). Resolved once: the legacy
        // `OXIBONSAI_FORCE_CPU_DECODE_AFTER` env var is applied at
        // construction, via `SpeculativeConfig::with_env_override`, into the
        // documented config alongside `RT-27`'s own variable — never read
        // per call inside the loop.
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

                match self.verify_batch(&batch, pos - 1) {
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

// ═════════════════════════════════════════════════════════════════════════
//  Sampled decode on the fused GPU route: top-k candidates, not full rows
//  (`perf-11`, sampled half)
// ═════════════════════════════════════════════════════════════════════════

impl InferenceEngine<'_> {
    /// The sampled top-k route's configuration (`perf-11`).
    pub fn sampled_topk(&self) -> SampledTopKConfig {
        self.sampled_topk
    }

    /// Configure the sampled top-k route: the candidate count, or the
    /// full-row reference / disabled modes.
    pub fn set_sampled_topk(&mut self, config: SampledTopKConfig) {
        self.sampled_topk = config;
    }

    /// Whether the configured sampler actually samples (temperature at or
    /// above [`GREEDY_TEMPERATURE_EPS`]).
    fn sampler_is_sampled(&self) -> bool {
        self.sampler.params().temperature >= GREEDY_TEMPERATURE_EPS
    }

    /// The single predicate deciding whether a sampled generation may take
    /// the top-k route (`perf-11`, sampled half): the fused GPU route, a
    /// truly sampled request, no penalty (a penalty must see every logit),
    /// no need for the full row (`needs_full_logits`: logprobs), the route
    /// enabled, and the sampler's `top_k` within
    /// `1..=candidates` and below the vocabulary — see [`SampledTopKConfig`]
    /// for why those last two conditions are what make the candidate sub-row
    /// an exact stand-in for the full row.
    pub fn sampled_topk_eligible(&self, needs_full_logits: bool) -> bool {
        if needs_full_logits
            || !self.uses_fused_gpu_decode()
            || self.sampled_topk.mode == SampledTopKMode::Off
            || !self.sampler_is_sampled()
            || self.greedy_penalties_active()
        {
            return false;
        }
        let top_k = self.sampler.params().top_k;
        let vocab = self.vocab_size();
        // `top_k < vocab`: at `top_k >= vocab` the full-row sampler performs
        // no top-k selection at all (it walks the row in index order) while
        // the candidate sub-row is walked in rank order, so the two could
        // not share one realisation even with a canonical sampler.
        top_k >= 1 && top_k < vocab && top_k <= self.sampled_topk.effective_candidates(vocab)
    }

    /// Count a sampled request on the fused GPU route that is about to decode
    /// the full logit row because it is not eligible for the top-k route —
    /// in [`EngineStats`](crate::engine::EngineStats) and, when attached, in
    /// the Prometheus `oxibonsai_sampled_full_row_requests_total` counter.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn note_sampled_full_row_request(&self) {
        if self.uses_fused_gpu_decode() && self.sampler_is_sampled() {
            self.stats
                .sampled_full_row_requests
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(m) = &self.metrics {
                m.sampled_full_row_requests_total.inc();
            }
        }
    }

    /// Count one decode step served from GPU top-k candidates
    /// ([`EngineStats`](crate::engine::EngineStats) and, when attached, the
    /// Prometheus `oxibonsai_sampled_topk_steps_total` counter).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn note_sampled_topk_step(&self) {
        self.stats
            .sampled_topk_steps
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(m) = &self.metrics {
            m.sampled_topk_steps_total.inc();
        }
    }

    /// Count one top-k-route decode step that downloaded the full logit row
    /// instead ([`EngineStats`](crate::engine::EngineStats) and, when
    /// attached, the Prometheus `oxibonsai_sampled_topk_full_row_steps_total`
    /// counter).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn note_sampled_topk_full_row_step(&self) {
        self.stats
            .sampled_topk_full_row_steps
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(m) = &self.metrics {
            m.sampled_topk_full_row_steps_total.inc();
        }
    }

    /// Sample one token from a candidate sub-row (`ids`/`values` in the
    /// kernel's order): the engine's own sampler runs temperature, `top_k`,
    /// `min_p`, `top_p` and the weighted draw over the `values`, consuming
    /// exactly one random draw, and the winning index maps back to its id.
    #[cfg(any(test, all(feature = "metal", target_os = "macos")))]
    pub(crate) fn sample_candidates(&mut self, ids: &[u32], values: &[f32]) -> RuntimeResult<u32> {
        let index = self.sampler.sample(values)? as usize;
        ids.get(index).copied().ok_or_else(|| {
            crate::error::RuntimeError::Config(format!(
                "sampled top-k: sampler returned candidate {index} of {}",
                ids.len()
            ))
        })
    }

    /// Sample one token from the top-`k` candidates of a full host logit row:
    /// the CPU extraction ([`top_k_candidates`]) followed by
    /// [`Self::sample_candidates`] — the computation a GPU candidate step
    /// performs, and what [`SampledTopKMode::FullRowCandidates`] runs as its
    /// bit-exact reference.
    ///
    /// Because `Sampler::sample_core` walks its top-k survivors in the
    /// canonical order (see [`SampledTopKConfig`]), a seeded draw through
    /// this function and a seeded `Sampler::sample` over the same full row
    /// pick the same token whenever `k >= top_k` (tested draw for draw on
    /// tie-heavy rows).
    #[cfg(test)]
    pub(crate) fn sample_row_candidates(&mut self, row: &[f32], k: usize) -> RuntimeResult<u32> {
        let (ids, values) = top_k_candidates(row, k);
        self.sample_candidates(&ids, &values)
    }

    /// Sample one token from a full host logit row with the classic sampler
    /// — exactly the draw [`SampledTopKMode::Off`] makes at every step. The
    /// route uses it for every step whose full row is on the host anyway
    /// (the prefill row and every fallback), so those steps can never differ
    /// from the route switched off.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn sample_full_row(&mut self, row: &[f32]) -> RuntimeResult<u32> {
        self.sampler.sample(row)
    }

    /// Whether top-`k` candidates (`ids`/`values`, descending) can stand in
    /// for the full row this step: their winner agrees with the fused
    /// forward's own argmax, and every value the sampler's `top_k` will read
    /// is finite (a `-inf` pad inside it means fewer finite logits than
    /// `top_k`, and a `NaN` is never selected).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn candidates_usable(&self, ids: &[u32], values: &[f32], argmax_id: u32) -> bool {
        let top_k = self.sampler.params().top_k;
        ids.first() == Some(&argmax_id)
            && values.len() >= top_k
            && values[..top_k].iter().all(|v| v.is_finite())
    }

    /// Download the fused forward's resident `[vocab]` logit row.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn download_resident_row(vocab: usize) -> RuntimeResult<Vec<f32>> {
        oxibonsai_kernels::gpu_backend::metal_resident_logits_download(vocab).map_err(|e| {
            crate::error::RuntimeError::Kernel(oxibonsai_kernels::error::KernelError::GpuError(
                e.to_string(),
            ))
        })
    }

    /// One sampled decode step on the fused route: forward `token` at `pos`
    /// on the GPU (logits stay resident), then draw the step's token.
    ///
    /// * [`SampledTopKMode::GpuCandidates`]: download only the top-`k`
    ///   candidates and draw from them; when they cannot stand in for the
    ///   row, download the resident row and draw with the classic sampler.
    /// * [`SampledTopKMode::FullRowCandidates`]: download the resident row and
    ///   run the very same candidate draw on its CPU extraction — the
    ///   bit-exact reference of the GPU download — with the same classic
    ///   fallback.
    ///
    /// A fused forward that fails (or `force_cpu`) falls back to a coherent
    /// CPU replay (exactly the greedy path's MET-05 machinery) and the
    /// classic sampler. Every step not served by GPU candidates is counted in
    /// `sampled_topk_full_row_steps`.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[allow(clippy::too_many_arguments)]
    fn sampled_topk_step(
        &mut self,
        committed: &[u32],
        token: u32,
        pos: usize,
        k: usize,
        cpu_kernel: &KernelDispatcher,
        cpu_fallback_active: &mut bool,
        force_cpu: bool,
    ) -> RuntimeResult<u32> {
        let gpu_attempt = match &self.model {
            LoadedModel::Dense(model) if !*cpu_fallback_active && !force_cpu => {
                Some(model.forward_greedy_gpu(token, pos))
            }
            _ => None,
        };
        match gpu_attempt {
            Some(Ok(argmax_id)) => {
                let vocab = self.vocab_size();
                if self.sampled_topk.mode == SampledTopKMode::GpuCandidates {
                    match oxibonsai_kernels::gpu_backend::metal_resident_logits_topk(vocab, k) {
                        Ok(candidates)
                            if self.candidates_usable(
                                &candidates.ids,
                                &candidates.values,
                                argmax_id,
                            ) =>
                        {
                            self.note_sampled_topk_step();
                            return self.sample_candidates(&candidates.ids, &candidates.values);
                        }
                        Ok(candidates) => tracing::debug!(
                            pos,
                            argmax_id,
                            gpu_first = ?candidates.ids.first(),
                            "sampled top-k: GPU candidates cannot stand in for the full row this \
                             step; downloading the resident row"
                        ),
                        Err(e) => tracing::warn!(
                            error = %e,
                            pos,
                            "sampled top-k: GPU candidate download failed; downloading the \
                             resident row"
                        ),
                    }
                }
                // The logits are still resident: no second forward needed.
                self.note_sampled_topk_full_row_step();
                let row = Self::download_resident_row(vocab)?;
                if self.sampled_topk.mode == SampledTopKMode::FullRowCandidates {
                    let (ids, values) = top_k_candidates(&row, k);
                    if self.candidates_usable(&ids, &values, argmax_id) {
                        return self.sample_candidates(&ids, &values);
                    }
                }
                return self.sample_full_row(&row);
            }
            Some(Err(e)) => {
                let rebuild = e
                    .downcast_ref::<oxibonsai_model::error::ModelError>()
                    .and_then(oxibonsai_model::model::gpu_fallback_cache_rebuild_pos);
                if rebuild.is_some() {
                    tracing::warn!(
                        error = %e, pos, ?rebuild,
                        "sampled GPU decode requires a host KV-cache rebuild (MET-05); replaying \
                         the committed sequence on the CPU"
                    );
                } else {
                    tracing::error!(
                        error = %e,
                        pos,
                        "sampled Metal GPU decode FAILED; rebuilding the CPU KV cache and \
                         continuing on the CPU -- a real GPU error, not expected control flow"
                    );
                }
            }
            None => {}
        }

        self.note_sampled_topk_full_row_step();
        let logits = if *cpu_fallback_active {
            self.forward_logits_on(token, pos, cpu_kernel)?
        } else {
            // Rebuild the host KV cache coherently: replay every committed
            // token (the last of which is `token`, at `pos`).
            self.model.reset();
            let mut last = Vec::new();
            for (p, &tok) in committed.iter().enumerate() {
                last = self.forward_logits_on(tok, p, cpu_kernel)?;
            }
            *cpu_fallback_active = true;
            last
        };
        self.sample_full_row(&logits)
    }

    /// The sampled top-k decode loop (`perf-11`, sampled half).
    ///
    /// `unchecked` like [`Self::generate_greedy_gpu_unchecked`]: the caller
    /// has gone through [`Self::sampled_topk_eligible`]. The first token is
    /// drawn from the prefill row — already on the host — by the classic
    /// sampler, exactly as [`SampledTopKMode::Off`] draws it; only the decode
    /// steps after it read GPU candidates.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn generate_sampled_gpu_topk_unchecked<F>(
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
        let Some(prefill_row) = self.prefill_for_generate(prompt_tokens)? else {
            return Ok(vec![]);
        };
        let k = self.sampled_topk.effective_candidates(self.vocab_size());
        let decode_start = std::time::Instant::now();
        let mut output_tokens = Vec::with_capacity(max_tokens.min(MAX_PREALLOC_TOKENS));
        let mut context: Vec<u32> = prompt_tokens.to_vec();
        let cpu_kernel = KernelDispatcher::with_tier(KernelTier::Reference);
        let mut cpu_fallback_active = false;
        let force_cpu_after = self.speculative.force_cpu_decode_after;
        let mut pos = prompt_tokens.len();

        for step in 0..max_tokens {
            let step_start = std::time::Instant::now();
            if self.is_cancelled() {
                tracing::debug!(pos, "sampled top-k generation cancelled");
                break;
            }
            let next = if step == 0 {
                self.sample_full_row(&prefill_row)?
            } else {
                let fed = *context.last().ok_or_else(|| {
                    crate::error::RuntimeError::Config(
                        "sampled top-k: empty decode context".to_string(),
                    )
                })?;
                let force_cpu = force_cpu_after.is_some_and(|n| output_tokens.len() >= n);
                let token = self.sampled_topk_step(
                    &context,
                    fed,
                    pos,
                    k,
                    &cpu_kernel,
                    &mut cpu_fallback_active,
                    force_cpu,
                )?;
                pos += 1;
                token
            };
            if self.is_eos(next) {
                tracing::debug!(pos, "EOS token generated (sampled top-k)");
                break;
            }
            if !emit(next) {
                tracing::debug!(pos, "receiver dropped, stopping generation");
                break;
            }
            output_tokens.push(next);
            context.push(next);
            if let Some(m) = &self.metrics {
                m.decode_token_duration_seconds
                    .observe(step_start.elapsed().as_secs_f64());
            }
        }

        self.record_decode_metrics(decode_start, output_tokens.len());
        self.stats.record_request(output_tokens.len());
        tracing::info!(
            prompt_len = prompt_tokens.len(),
            generated = output_tokens.len(),
            candidates = k,
            "sampled top-k generation complete"
        );
        Ok(output_tokens)
    }

    /// Route a sampled request that reached `generate`/`generate_tracked`/the
    /// streaming pair: through the top-k route when eligible, else count it
    /// as a full-row request and let the caller run its classic loop.
    ///
    /// Returns `Some(tokens)` when the top-k route ran.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn try_sampled_topk_route<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        emit: F,
    ) -> RuntimeResult<Option<Vec<u32>>>
    where
        F: FnMut(u32) -> bool,
    {
        if self.sampled_topk_eligible(false) {
            return self
                .generate_sampled_gpu_topk_unchecked(prompt_tokens, max_tokens, emit)
                .map(Some);
        }
        self.note_sampled_full_row_request();
        Ok(None)
    }
}

#[cfg(test)]
#[path = "engine_topk_tests.rs"]
mod topk_tests;

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
    /// (`RT-22`). The MSL/CUDA kernels share this convention
    /// (`GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`, `perf-11`), and this helper is
    /// the CPU-side reference both tiers agree with.
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

    /// `perf-11`: `generate_greedy_gpu` is a
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
