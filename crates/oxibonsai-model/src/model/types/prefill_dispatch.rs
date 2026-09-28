//! Batched prefill / speculative-verify entry points of `BonsaiModel` and
//! their GPU → CPU dispatch ladder, split out of `model/types/mod.rs`
//! (B2-11-FIX).

use super::BonsaiModel;
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    )
))]
use super::OutputWeight;
use crate::error::{ModelError, ModelResult};
use oxibonsai_kernels::traits::OneBitKernel;

impl BonsaiModel<'_> {
    /// Process multiple prompt tokens in a single batch forward pass on GPU.
    ///
    /// Uses GEMM instead of GEMV for projections (processing all tokens at once),
    /// with sequential per-token attention. Only the last token's logits are
    /// returned (for generation to start). The GPU KV cache is populated for
    /// all positions.
    ///
    /// Falls back to sequential single-token forward if the GPU batch path
    /// is unavailable.
    ///
    /// A prompt longer than [`prefill_chunk_tokens`](Self::prefill_chunk_tokens)
    /// is split into non-overlapping chunks and driven through
    /// [`crate::chunked_prefill::run_chunked_prefill`] (M-18); anything that
    /// fits in one chunk takes the single-shot path unchanged, so the fused
    /// Metal batch path's dispatch granularity is unaffected for short prompts.
    pub fn forward_prefill(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<f32>> {
        let Some(config) =
            crate::chunked_prefill::prefill_chunk_plan(token_ids.len(), self.prefill_chunk_tokens)
        else {
            return self.forward_prefill_unchunked(token_ids, pos_start, kernel);
        };
        crate::chunked_prefill::run_chunked_prefill(
            token_ids,
            config,
            |chunk| {
                self.forward_prefill_unchunked(&chunk.tokens, pos_start + chunk.start_pos, kernel)
            },
            // `PrefillFirst` never yields, so this is never called; the
            // executor also serves the interleaved priorities, where a caller
            // supplies real decode work.
            || Ok(()),
        )?
        .ok_or_else(|| {
            ModelError::Internal(
                "forward_prefill: the chunked executor produced no logits for a non-empty prompt"
                    .to_string(),
            )
        })
    }

    /// Single-shot batched prefill: the whole prompt in one call.
    ///
    /// This is exactly what [`forward_prefill`](Self::forward_prefill) was
    /// before M-18 wired the chunk scheduler in, and it is what each chunk
    /// runs through.
    fn forward_prefill_unchunked(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(ModelError::MissingTensor {
                name: "forward_prefill: empty token_ids".into(),
            });
        }
        if token_ids.len() == 1 {
            return self.forward(token_ids[0], pos_start, kernel);
        }
        // M-17: every batched prefill path below — GPU and the batched CPU one
        // — computes full causal attention. A model that declared
        // `<arch>.attention.sliding_window` must not silently get it, so it
        // falls through to `forward_sequential`, whose per-token
        // `forward_into` honours the window. See `forward_into`.
        let windowed = self.config.sliding_window.is_some();
        let _gpu_kernel = kernel.is_gpu_accelerated() && !self.force_cpu_at(pos_start) && !windowed;
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel && token_ids.len() <= 16 {
            return self.forward_sequential(token_ids, pos_start, kernel);
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if _gpu_kernel
            && !matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
        {
            // Fused Metal prefill supports OneBit + Ternary today. It maintains
            // the DEVICE KV cache only (MET-05).
            match self.try_metal_prefill_with_lm_head(token_ids, pos_start) {
                Ok(logits) => {
                    self.note_device_kv_used();
                    return Ok(logits);
                }
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "metal batch prefill failed, falling back to sequential"
                    );
                }
            }
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
        {
            // FP8 hybrid batch prefill (Phase 28.B): batched FP8 GEMM projections
            // on the GPU, attention + K/V store on the CPU against `self.kv_cache`
            // — the same cache the per-token FP8 decode path reads. Correct by
            // construction (no split KV cache), and a HOST-KV path for MET-05.
            let is_e4m3 = matches!(&self.output_weight, OutputWeight::FP8E4M3(_));
            match self.try_metal_prefill_with_lm_head_fp8(token_ids, pos_start, is_e4m3) {
                Ok(logits) => {
                    self.note_host_kv_written(pos_start + token_ids.len() - 1);
                    return Ok(logits);
                }
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "metal FP8 batch prefill failed, falling back to sequential"
                    );
                }
            }
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // FP8 batch GEMM prefill (Phase 26).
            let is_e4m3 = matches!(&self.output_weight, OutputWeight::FP8E4M3(_));
            match self.try_cuda_prefill_with_lm_head_fp8(token_ids, pos_start, is_e4m3) {
                Ok(logits) => {
                    self.note_device_kv_used();
                    return Ok(logits);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda FP8 batch prefill failed, falling back to sequential",
                ),
            }
            // Fallback: sequential token-by-token CUDA GEMV.
            return self.forward_sequential(token_ids, pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::Q4_0(_) | OutputWeight::Q8_0(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            let q4_0 = matches!(&self.output_weight, OutputWeight::Q4_0(_));
            match self.try_cuda_prefill_with_lm_head_q_std(token_ids, pos_start, q4_0) {
                Ok(logits) => {
                    self.note_device_kv_used();
                    return Ok(logits);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda Q4_0/Q8_0 batch prefill failed, falling back to sequential",
                ),
            }
            // Fallback: sequential token-by-token
            return self.forward_sequential(token_ids, pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::Q2K(_)
                    | OutputWeight::Q3K(_)
                    | OutputWeight::Q4K(_)
                    | OutputWeight::Q5K(_)
                    | OutputWeight::Q6K(_)
                    | OutputWeight::Q8K(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // K-quant batch GEMM prefill (Phase 25).
            let fmt = self.output_weight.k_quant_format().ok_or_else(|| {
                ModelError::Internal(format!(
                    "forward_prefill: K-quant branch entered with a {} LM head",
                    self.output_weight.kind()
                ))
            })?;
            match self.try_cuda_prefill_with_lm_head_k_quant(token_ids, pos_start, fmt) {
                Ok(logits) => {
                    self.note_device_kv_used();
                    return Ok(logits);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda K-quant batch prefill failed, falling back to sequential",
                ),
            }
            // Fallback: sequential token-by-token forward using CUDA GEMV.
            return self.forward_sequential(token_ids, pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel {
            match self.try_cuda_prefill_with_lm_head(token_ids, pos_start) {
                Ok(logits) => {
                    self.note_device_kv_used();
                    return Ok(logits);
                }
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("LM head not supported on CUDA prefill path") {
                        tracing::debug!(
                            error = % e,
                            "cuda batch prefill skipped (LM head dtype not supported), using sequential"
                        );
                    } else {
                        tracing::warn!(
                            error = % e,
                            "cuda batch prefill failed, falling back to sequential"
                        );
                    }
                }
            }
        }
        // perf-M2: try the batched CPU prefill before falling back to the
        // per-token loop -- this is the hook that makes `prefill_cpu` reachable.
        // Skipped for a windowed model (M-17): `forward_prefill_cpu`'s
        // `gqa_attention_row` is unconditionally full-causal.
        if !windowed {
            if let Some(logits) = self.forward_prefill_cpu(token_ids, pos_start)? {
                return Ok(logits);
            }
        }
        self.forward_sequential(token_ids, pos_start, kernel)
    }

    /// Sequential per-token prefill fallback shared by every batched path.
    ///
    /// Guards the MET-05 transition once, up front: looping into
    /// [`forward`](Self::forward) from a sequence whose history lives in the
    /// device KV cache would attend over an all-zero host cache.
    fn forward_sequential(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<f32>> {
        self.require_host_kv_coherent(pos_start)?;
        let mut last_logits = Vec::new();
        for (i, &token_id) in token_ids.iter().enumerate() {
            last_logits = self.forward(token_id, pos_start + i, kernel)?;
        }
        Ok(last_logits)
    }

    /// Sequential per-token verify fallback (argmax at every position).
    fn forward_sequential_verify(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<u32>> {
        self.require_host_kv_coherent(pos_start)?;
        let mut token_ids_out = Vec::with_capacity(token_ids.len());
        for (i, &token_id) in token_ids.iter().enumerate() {
            let logits = self.forward(token_id, pos_start + i, kernel)?;
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (j, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = j as u32;
                }
            }
            token_ids_out.push(best_idx);
        }
        Ok(token_ids_out)
    }

    /// Forward pass for speculative decode verification.
    ///
    /// Processes multiple tokens in batch via GPU prefill, then runs the LM head
    /// and argmax on ALL positions (not just the last). Returns the greedy
    /// argmax token ID for each input position.
    ///
    /// If GPU batch path is unavailable, falls back to sequential CPU forward
    /// with argmax at each position.
    pub fn forward_prefill_verify(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<u32>> {
        if token_ids.is_empty() {
            return Ok(vec![]);
        }
        // M-17: as in `forward_prefill` — every batched verify path is
        // full-causal, so a windowed model takes the sequential one.
        let _gpu_kernel = kernel.is_gpu_accelerated()
            && !self.force_cpu_at(pos_start)
            && self.config.sliding_window.is_none();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if _gpu_kernel
            && !matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
        {
            // Fused Metal prefill verify supports OneBit + Ternary today; FP8
            // falls through to per-token sequential, which dispatches through
            // `KernelDispatcher::gemv_fp8_*` (Metal GPU via Phase 27).
            match self.try_metal_prefill_verify(token_ids, pos_start) {
                Ok(ids) => {
                    self.note_device_kv_used();
                    return Ok(ids);
                }
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "metal batch prefill verify failed, falling back to sequential"
                    );
                }
            }
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // FP8 batch GEMM prefill verify (Phase 26).
            let is_e4m3 = matches!(&self.output_weight, OutputWeight::FP8E4M3(_));
            match self.try_cuda_prefill_verify_fp8(token_ids, pos_start, is_e4m3) {
                Ok(ids) => {
                    self.note_device_kv_used();
                    return Ok(ids);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda FP8 batch prefill verify failed, falling back to sequential",
                ),
            }
            return self.forward_sequential_verify(token_ids, pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::Q2K(_)
                    | OutputWeight::Q3K(_)
                    | OutputWeight::Q4K(_)
                    | OutputWeight::Q5K(_)
                    | OutputWeight::Q6K(_)
                    | OutputWeight::Q8K(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // K-quant batch GEMM prefill verify (Phase 25).
            let fmt = self.output_weight.k_quant_format().ok_or_else(|| {
                ModelError::Internal(format!(
                    "forward_prefill_verify: K-quant branch entered with a {} LM head",
                    self.output_weight.kind()
                ))
            })?;
            match self.try_cuda_prefill_verify_k_quant(token_ids, pos_start, fmt) {
                Ok(ids) => {
                    self.note_device_kv_used();
                    return Ok(ids);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda K-quant batch prefill verify failed, falling back to sequential",
                ),
            }
            // Fallback: sequential token-by-token with argmax.
            return self.forward_sequential_verify(token_ids, pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel {
            match self.try_cuda_prefill_verify(token_ids, pos_start) {
                Ok(ids) => {
                    self.note_device_kv_used();
                    return Ok(ids);
                }
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "cuda batch prefill verify failed, falling back to sequential"
                    );
                }
            }
        }
        self.forward_sequential_verify(token_ids, pos_start, kernel)
    }
}

/// Log one CUDA batch-prefill fallback at the level it deserves (**F15.4**).
///
/// The Q4_0/Q8_0, K-quant and FP8 CUDA batch-prefill families are disabled by
/// construction — `forward_cuda::cuda_split_prefill_allowed()` is an
/// unconditional `false` while no kernels-side entry point can hand the
/// prompt's K/V back to the host — so all six entry points return the same
/// refusal on **every** prompt. Reporting that at `warn!` filled logs with a
/// line the operator can do nothing about and which does not describe a
/// failure: the bit-correct sequential prefill runs and the answer is right.
///
/// This keeps the message text unchanged and changes only its level: the
/// by-construction refusal is `debug!`, anything else — a real dispatch
/// failure, once the guard is lifted — stays `warn!`. Matching on the
/// refusal's own wording mirrors the idiom already used for the "LM head not
/// supported on CUDA prefill path" skip a few branches below.
#[cfg(all(
    feature = "native-cuda",
    not(all(feature = "metal", target_os = "macos")),
    any(target_os = "linux", target_os = "windows")
))]
fn cuda_prefill_fallback_log(error: &str, message: &'static str) {
    // The distinguishing clause of `forward_cuda::cuda_split_prefill_disabled`.
    const DISABLED_BY_CONSTRUCTION: &str = "writes a GPU-private KV cache";
    if error.contains(DISABLED_BY_CONSTRUCTION) {
        tracing::debug!(error, "{message}");
    } else {
        tracing::warn!(error, "{message}");
    }
}
