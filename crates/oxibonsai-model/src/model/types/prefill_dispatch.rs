//! Batched prefill / speculative-verify entry points of `BonsaiModel` and
//! their GPU → CPU dispatch ladder, split out of `model/types/mod.rs`.

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
                    self.note_cuda_prefill_kv(pos_start, token_ids.len());
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
            // F6: on success the prompt's device K/V has already been read back
            // into `self.kv_cache`, so this is a HOST-KV write.
            match self.try_cuda_prefill_with_lm_head_q_std(token_ids, pos_start, q4_0) {
                Ok(logits) => {
                    self.note_cuda_prefill_kv(pos_start, token_ids.len());
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
                    self.note_cuda_prefill_kv(pos_start, token_ids.len());
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
                    self.note_cuda_prefill_kv(pos_start, token_ids.len());
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
                    self.note_cuda_prefill_kv(pos_start, token_ids.len());
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
                OutputWeight::Q4_0(_) | OutputWeight::Q8_0(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // Q4_0/Q8_0 batch GEMM prefill verify (Phase 24B). F6: on success
            // every verified position's device K/V is already in
            // `self.kv_cache`, so this is a HOST-KV write.
            let q4_0 = matches!(&self.output_weight, OutputWeight::Q4_0(_));
            match self.try_cuda_prefill_verify_q_std(token_ids, pos_start, q4_0) {
                Ok(ids) => {
                    self.note_cuda_prefill_kv(pos_start, token_ids.len());
                    return Ok(ids);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda Q4_0/Q8_0 batch prefill verify failed, falling back to sequential",
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
                    self.note_cuda_prefill_kv(pos_start, token_ids.len());
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
                    self.note_cuda_prefill_kv(pos_start, token_ids.len());
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

    /// MET-05 bookkeeping after a CUDA batch prefill / verify wrote
    /// `n_tokens` positions starting at `pos_start`.
    ///
    /// The Q4_0/Q8_0 family reads its GPU-private device K/V back into
    /// `self.kv_cache` before returning (finding **F6**), and its decode
    /// attends over that host cache, so its prefill is a host-KV write. The
    /// Q1 and ternary paths keep the prompt's K/V only in the device cache
    /// their fused decode shares, so they latch the device-KV state; so do
    /// the K-quant and FP8 families, whose prefill is refused today (their
    /// callers store no read-back) and so never reaches this.
    #[cfg(all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    ))]
    fn note_cuda_prefill_kv(&mut self, pos_start: usize, n_tokens: usize) {
        if matches!(
            &self.output_weight,
            OutputWeight::Q4_0(_) | OutputWeight::Q8_0(_)
        ) {
            if let Some(last) = (pos_start + n_tokens).checked_sub(1) {
                self.note_host_kv_written(last);
            }
        } else {
            self.note_device_kv_used();
        }
    }
}

/// Write one CUDA batch prefill's device K/V read-back (finding **F6**) into
/// the host `KvCache` at `[pos_start, pos_start + batch_size)`, then advance
/// the cache cursor over that window (never backwards).
///
/// `readback[l]` is device layer `l`'s `(keys, values)`, each `[n_kv_heads *
/// batch_size * head_dim]` in `[head][position][dim]` order — exactly
/// [`KvCache::try_inject_block`](crate::kv_cache::KvCache::try_inject_block)'s
/// block layout — and `layer_indices[l]` is the host-cache layer it belongs
/// to (the block's own `layer_index()`, the index the host-KV attention path
/// stores under).
///
/// Every layer index and every layer's length is checked before anything is
/// written, and a window past `max_seq_len` is refused rather than handed to
/// `try_inject_block`, which would silently skip the positions at or past
/// the limit.
///
/// # Errors
/// [`ModelError::ShapeMismatch`] for a layer-count, layer-index or
/// per-layer length mismatch; [`ModelError::SequenceTooLong`] for a window
/// past the cache's limit; what `try_inject_block` reports for a failed
/// growth.
#[cfg(any(
    test,
    all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    )
))]
pub(super) fn store_kv_readback(
    kv_cache: &mut crate::kv_cache::KvCache,
    layer_indices: &[usize],
    readback: &[(Vec<f32>, Vec<f32>)],
    pos_start: usize,
    batch_size: usize,
) -> ModelResult<()> {
    if readback.len() != layer_indices.len() {
        return Err(ModelError::ShapeMismatch {
            name: "cuda kv read-back layer count".to_string(),
            expected: vec![layer_indices.len()],
            actual: vec![readback.len()],
        });
    }
    let max_ctx = kv_cache.max_seq_len();
    let end = pos_start
        .checked_add(batch_size)
        .filter(|&end| end <= max_ctx)
        .ok_or_else(|| ModelError::SequenceTooLong {
            seq_len: pos_start.saturating_add(batch_size),
            max_ctx,
        })?;
    let per_layer = kv_cache
        .num_kv_heads()
        .checked_mul(batch_size)
        .and_then(|n| n.checked_mul(kv_cache.head_dim()))
        .ok_or_else(|| {
            ModelError::Internal("cuda kv read-back: window size overflows".to_string())
        })?;
    for (&layer, (keys, values)) in layer_indices.iter().zip(readback) {
        if layer >= kv_cache.num_layers() {
            return Err(ModelError::ShapeMismatch {
                name: "cuda kv read-back layer index".to_string(),
                expected: vec![kv_cache.num_layers()],
                actual: vec![layer],
            });
        }
        if keys.len() != per_layer || values.len() != per_layer {
            return Err(ModelError::ShapeMismatch {
                name: "cuda kv read-back layer window".to_string(),
                expected: vec![per_layer],
                actual: vec![keys.len(), values.len()],
            });
        }
    }
    for (&layer, (keys, values)) in layer_indices.iter().zip(readback) {
        kv_cache.try_inject_block(layer, pos_start, batch_size, keys, values)?;
    }
    if kv_cache.seq_len() < end {
        kv_cache.set_seq_len(end);
    }
    Ok(())
}

/// Log one CUDA batch-prefill fallback at the level it deserves (**F15.4**).
///
/// The K-quant and FP8 CUDA batch-prefill families are refused by
/// construction (`forward_cuda::cuda_split_prefill_allowed()` is `false`:
/// their callers store no K/V read-back into the host cache), and the
/// Q4_0/Q8_0 family is refused for every prompt that does not start at
/// position 0 (`forward_cuda::cuda_split_prefill_allowed_with_readback`: its
/// GPU-private device cache holds no history before `pos_start`). Those
/// refusals come back on every such prompt, and reporting them at `warn!`
/// filled logs with a line the operator can do nothing about and which does
/// not describe a failure: the bit-correct sequential prefill runs and the
/// answer is right.
///
/// This keeps the message text unchanged and changes only its level: every
/// by-construction refusal carries the clause below and is `debug!`,
/// anything else — a real dispatch failure — stays `warn!`. Matching on the
/// refusal's own wording mirrors the idiom already used for the "LM head not
/// supported on CUDA prefill path" skip a few branches below.
#[cfg(all(
    feature = "native-cuda",
    not(all(feature = "metal", target_os = "macos")),
    any(target_os = "linux", target_os = "windows")
))]
fn cuda_prefill_fallback_log(error: &str, message: &'static str) {
    // The distinguishing clause of `forward_cuda::cuda_split_prefill_disabled`
    // and `forward_cuda::cuda_split_prefill_needs_history`.
    const DISABLED_BY_CONSTRUCTION: &str = "writes a GPU-private KV cache";
    if error.contains(DISABLED_BY_CONSTRUCTION) {
        tracing::debug!(error, "{message}");
    } else {
        tracing::warn!(error, "{message}");
    }
}

#[cfg(test)]
mod tests {
    use super::store_kv_readback;
    use crate::error::ModelError;
    use crate::kv_cache::KvCache;

    const HEADS: usize = 2;
    const HEAD_DIM: usize = 4;

    /// A `[head][position][dim]` window whose every element is distinct and
    /// names its layer, head, position and dim.
    fn window(layer: usize, batch: usize, value_bias: f32) -> Vec<f32> {
        let mut w = Vec::with_capacity(HEADS * batch * HEAD_DIM);
        for h in 0..HEADS {
            for t in 0..batch {
                for d in 0..HEAD_DIM {
                    w.push(value_bias + (layer * 1000 + h * 100 + t * 10 + d) as f32);
                }
            }
        }
        w
    }

    fn readback(layers: usize, batch: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        (0..layers)
            .map(|l| (window(l, batch, 0.0), window(l, batch, 0.5)))
            .collect()
    }

    #[test]
    fn store_kv_readback_writes_each_layer_window_where_the_host_path_reads_it() {
        let mut kv = KvCache::new(2, HEADS, HEAD_DIM, 16);
        let rb = readback(2, 3);
        store_kv_readback(&mut kv, &[0, 1], &rb, 5, 3).expect("store");
        for (layer, (keys, values)) in rb.iter().enumerate() {
            let (got_k, got_v) = kv.extract_block(layer, 5, 3);
            assert_eq!(&got_k, keys, "layer {layer} keys");
            assert_eq!(&got_v, values, "layer {layer} values");
        }
        // Position 5 + 1 of head 1 in layer 1, read the way attention does.
        let keys = kv.keys_for(1, 1, 8);
        let at = |pos: usize| &keys[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
        assert_eq!(at(6), &[1110.0, 1111.0, 1112.0, 1113.0]);
        assert_eq!(
            at(4),
            &[0.0; HEAD_DIM],
            "positions outside the window stay untouched"
        );
        assert_eq!(kv.seq_len(), 8, "cursor advanced over the window");
    }

    #[test]
    fn store_kv_readback_follows_the_block_layer_indices() {
        let mut kv = KvCache::new(2, HEADS, HEAD_DIM, 16);
        let rb = readback(2, 2);
        // Device layer 0 belongs to host layer 1 and vice versa.
        store_kv_readback(&mut kv, &[1, 0], &rb, 0, 2).expect("store");
        assert_eq!(kv.extract_block(1, 0, 2).0, rb[0].0);
        assert_eq!(kv.extract_block(0, 0, 2).0, rb[1].0);
    }

    #[test]
    fn store_kv_readback_refuses_a_window_past_the_limit_without_writing() {
        let mut kv = KvCache::new(1, HEADS, HEAD_DIM, 8);
        let rb = readback(1, 3);
        let err = store_kv_readback(&mut kv, &[0], &rb, 6, 3).expect_err("[6, 9) > 8");
        assert!(matches!(
            err,
            ModelError::SequenceTooLong {
                seq_len: 9,
                max_ctx: 8
            }
        ));
        assert_eq!(kv.extract_block(0, 6, 2).0, vec![0.0; HEADS * 2 * HEAD_DIM]);
        assert_eq!(kv.seq_len(), 0);
    }

    #[test]
    fn store_kv_readback_validates_every_layer_before_writing_any() {
        let mut kv = KvCache::new(2, HEADS, HEAD_DIM, 8);
        let mut rb = readback(2, 2);
        rb[1].1.pop(); // layer 1's values one element short
        let err = store_kv_readback(&mut kv, &[0, 1], &rb, 0, 2).expect_err("short layer");
        assert!(matches!(err, ModelError::ShapeMismatch { .. }));
        assert_eq!(
            kv.extract_block(0, 0, 2).0,
            vec![0.0; HEADS * 2 * HEAD_DIM],
            "layer 0 must not be written when layer 1 is malformed"
        );

        let err = store_kv_readback(&mut kv, &[0], &readback(2, 2), 0, 2)
            .expect_err("layer count mismatch");
        assert!(matches!(err, ModelError::ShapeMismatch { .. }));
        let err = store_kv_readback(&mut kv, &[0, 2], &readback(2, 2), 0, 2)
            .expect_err("layer index past the cache");
        assert!(matches!(err, ModelError::ShapeMismatch { .. }));
    }

    #[test]
    fn store_kv_readback_never_moves_the_cursor_backwards() {
        let mut kv = KvCache::new(1, HEADS, HEAD_DIM, 16);
        kv.set_seq_len(10);
        store_kv_readback(&mut kv, &[0], &readback(1, 3), 0, 3).expect("store");
        assert_eq!(kv.seq_len(), 10);
    }
}
