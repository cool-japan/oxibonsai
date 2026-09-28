//! Single-token decode (`forward` / `forward_into`) and the per-block
//! host-KV loop, split out of `model/types/mod.rs` (B2-11-FIX).

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    )
))]
use super::lm_head::copy_logits;
#[cfg(all(
    feature = "native-cuda",
    not(all(feature = "metal", target_os = "macos")),
    any(target_os = "linux", target_os = "windows")
))]
use super::OutputWeight;
use super::{BonsaiModel, ModelScratch};
use crate::block::TransformerBlock;
use crate::error::{ModelError, ModelResult};
use crate::kv_cache::KvCache;
use crate::layers::rope::RopeTable;
use oxibonsai_kernels::traits::OneBitKernel;

impl BonsaiModel<'_> {
    /// Forward pass for a single token at position `pos`.
    ///
    /// Returns logits over the vocabulary `[vocab_size]`. Allocates the result
    /// vector; [`forward_into`](Self::forward_into) writes into a caller-owned
    /// buffer and allocates nothing at all.
    #[tracing::instrument(skip(self, kernel), fields(token_id, pos))]
    pub fn forward(
        &mut self,
        token_id: u32,
        pos: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<f32>> {
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.forward_into(token_id, pos, kernel, &mut logits)?;
        Ok(logits)
    }

    /// Forward pass for a single token at position `pos`, writing
    /// `[vocab_size]` logits into `logits`.
    ///
    /// Allocation-free after the first call: every intermediate lives in
    /// [`ModelScratch`] (M-22).
    ///
    /// # Errors
    ///
    /// * [`ModelError::SequenceTooLong`] — `pos` is beyond the effective
    ///   context (sec-11).
    /// * [`gpu_fallback_requires_cache_rebuild`] — the GPU decode path has been
    ///   maintaining a device KV cache for this sequence and cannot fall back
    ///   to the CPU without the host cache being rebuilt first (MET-05).
    /// * [`ModelError::ShapeMismatch`] — `logits` is shorter than `vocab_size`.
    pub fn forward_into(
        &mut self,
        token_id: u32,
        pos: usize,
        kernel: &dyn OneBitKernel,
        logits: &mut [f32],
    ) -> ModelResult<()> {
        let vocab = self.config.vocab_size;
        if logits.len() < vocab {
            return Err(ModelError::ShapeMismatch {
                name: "forward logits".to_string(),
                expected: vec![vocab],
                actual: vec![logits.len()],
            });
        }
        self.ensure_context_capacity(pos)?;
        // Move the scratch out so the `&self` GPU entry points can be called
        // while its buffers are borrowed mutably. `take` moves the `Vec`s (no
        // allocation) and the buffers are put back on every exit path.
        let mut scratch = std::mem::take(&mut self.scratch);
        let result = self.forward_core(token_id, pos, kernel, &mut scratch, logits);
        self.scratch = scratch;
        result
    }

    /// Body of [`forward_into`](Self::forward_into), with the scratch detached.
    fn forward_core(
        &mut self,
        token_id: u32,
        pos: usize,
        kernel: &dyn OneBitKernel,
        scratch: &mut ModelScratch,
        logits: &mut [f32],
    ) -> ModelResult<()> {
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        if scratch.hidden.len() != h {
            scratch.hidden.resize(h, 0.0);
        }
        if scratch.normed.len() != h {
            scratch.normed.resize(h, 0.0);
        }
        self.token_embd.copy_row(token_id, &mut scratch.hidden)?;
        let t_blocks_start = std::time::Instant::now();
        // M-17: read once, up front. Every `self.blocks` loop below needs it
        // while `self.kv_cache` is mutably borrowed, so it must already be a
        // plain `Option<usize>` (it is `Copy`) rather than a field access.
        let sliding_window = self.config.sliding_window;
        // M-17: the fused whole-layer GPU paths below all compute FULL causal
        // attention — none of them takes a window argument. Claiming them for
        // a model that declared `<arch>.attention.sliding_window` would return
        // confident output that ignores the very constraint the file declares,
        // so a windowed model routes through the per-block path (which honours
        // it) instead. `None` — every shipped model — leaves this expression
        // exactly as it was.
        //
        // NOT covered here: `forward_greedy_gpu` (`forward_metal.rs`, a
        // different package's file this wave) is a *second* fused-Metal decode
        // entry point that `engine_greedy.rs` calls directly, never through
        // `forward_into`, so this gate cannot reach it. It needs the same
        // refusal at its own head — returning `Err` is enough, because its one
        // production caller already falls back to the CPU path on `Err`. The
        // exact change is recorded in this package's `deviations`.
        let _gpu_kernel =
            kernel.is_gpu_accelerated() && !self.force_cpu_at(pos) && sliding_window.is_none();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if _gpu_kernel {
            if scratch.gpu_logits.len() != vocab {
                scratch.gpu_logits.resize(vocab, 0.0);
            }
            match self.try_metal_full_forward_with_lm_head(
                &mut scratch.hidden,
                pos,
                &mut scratch.gpu_logits,
            ) {
                Ok(()) => {
                    self.note_device_kv_used();
                    // `try_metal_full_forward_with_lm_head` owns the `Vec` and
                    // may resize it; copy what it produced rather than indexing
                    // a length it did not promise (M-29: no panics here).
                    copy_logits(&scratch.gpu_logits, &mut logits[..vocab]);
                    let t_elapsed = t_blocks_start.elapsed();
                    tracing::debug!(
                        target : "fwd_profile",
                        "pos={pos} fused_gpu={:.1}ms (metal layers+norm+lm_head)", t_elapsed
                        .as_secs_f64() * 1000.0,
                    );
                    return Ok(());
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        pos,
                        "fused Metal forward+LM head unavailable, trying the next path"
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
            // FP8 models use CUDA-accelerated GEMV via KernelTier::Gpu block dispatch.
            // Skip the Q1/TQ2 fused CUDA graph paths (they only handle 1-bit/ternary
            // weights). This is a HOST-KV path: the blocks read and write
            // `self.kv_cache`.
            self.require_host_kv_coherent(pos)?;
            run_blocks(
                &self.blocks,
                sliding_window,
                &mut scratch.hidden,
                pos,
                &mut self.kv_cache,
                &self.rope,
                kernel,
            )?;
            self.note_host_kv_written(pos);
            let t_blocks_elapsed = t_blocks_start.elapsed();
            tracing::debug!(
                target: "fwd_profile",
                "pos={pos} fp8_cuda_dispatch={:.1}ms (cuda gemv via block dispatch)",
                t_blocks_elapsed.as_secs_f64() * 1000.0,
            );
            let t_norm_start = std::time::Instant::now();
            self.output_norm
                .forward(&scratch.hidden, &mut scratch.normed)?;
            let t_norm_elapsed = t_norm_start.elapsed();
            let t_lm_start = std::time::Instant::now();
            self.apply_lm_head(&scratch.normed, &mut logits[..vocab])?;
            let t_lm_elapsed = t_lm_start.elapsed();
            tracing::debug!(
                target: "fwd_profile",
                "pos={pos} norm={:.2}ms lm_head={:.2}ms",
                t_norm_elapsed.as_secs_f64() * 1000.0,
                t_lm_elapsed.as_secs_f64() * 1000.0,
            );
            return Ok(());
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::Q4_0(_)
                    | OutputWeight::Q8_0(_)
                    | OutputWeight::Q2K(_)
                    | OutputWeight::Q3K(_)
                    | OutputWeight::Q4K(_)
                    | OutputWeight::Q5K(_)
                    | OutputWeight::Q6K(_)
                    | OutputWeight::Q8K(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // Q4_0/Q8_0 and K-quant models: skip the Q1 fused CUDA graph path.
            // Each layer GEMV dispatches to CUDA via LinearQ*::forward(), against
            // the HOST KV cache.
            self.require_host_kv_coherent(pos)?;
            run_blocks(
                &self.blocks,
                sliding_window,
                &mut scratch.hidden,
                pos,
                &mut self.kv_cache,
                &self.rope,
                kernel,
            )?;
            self.note_host_kv_written(pos);
            let t_blocks_elapsed = t_blocks_start.elapsed();
            tracing::debug!(
                target: "fwd_profile",
                "pos={pos} quant_cuda_dispatch={:.1}ms (cuda gemv via block dispatch)",
                t_blocks_elapsed.as_secs_f64() * 1000.0,
            );
            let t_norm_start = std::time::Instant::now();
            self.output_norm
                .forward(&scratch.hidden, &mut scratch.normed)?;
            let t_norm_elapsed = t_norm_start.elapsed();
            let t_lm_start = std::time::Instant::now();
            self.apply_lm_head(&scratch.normed, &mut logits[..vocab])?;
            let t_lm_elapsed = t_lm_start.elapsed();
            tracing::debug!(
                target: "fwd_profile",
                "pos={pos} norm={:.2}ms lm_head={:.2}ms",
                t_norm_elapsed.as_secs_f64() * 1000.0,
                t_lm_elapsed.as_secs_f64() * 1000.0,
            );
            return Ok(());
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel {
            match self.try_cuda_full_forward_with_lm_head(&scratch.hidden, pos) {
                Ok(fused_logits) => {
                    self.note_device_kv_used();
                    copy_logits(&fused_logits, &mut logits[..vocab]);
                    return Ok(());
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        pos,
                        "fused CUDA forward+LM head unavailable, trying the next path"
                    );
                }
            }
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let did_full_forward = if _gpu_kernel {
            match self.try_metal_full_forward_inner(&mut scratch.hidden, pos) {
                Ok(()) => true,
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        pos,
                        "fused Metal Q1 full-layer forward unavailable, trying ternary"
                    );
                    match self.try_metal_full_forward_ternary_inner(&mut scratch.hidden, pos) {
                        Ok(()) => true,
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                pos,
                                "fused Metal ternary full-layer forward unavailable, \
                                 falling back to per-block CPU/GPU dispatch"
                            );
                            false
                        }
                    }
                }
            }
        } else {
            false
        };
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        let did_full_forward = if _gpu_kernel {
            match self.try_cuda_full_forward_inner(&scratch.hidden, pos) {
                Ok(new_hidden) => {
                    let n = new_hidden.len().min(scratch.hidden.len());
                    scratch.hidden[..n].copy_from_slice(&new_hidden[..n]);
                    true
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        pos,
                        "fused CUDA full-layer forward unavailable, falling back to \
                         per-block CPU/GPU dispatch"
                    );
                    false
                }
            }
        } else {
            false
        };
        #[cfg(not(any(
            all(feature = "metal", target_os = "macos"),
            all(
                feature = "native-cuda",
                not(all(feature = "metal", target_os = "macos")),
                any(target_os = "linux", target_os = "windows")
            )
        )))]
        let did_full_forward = false;
        if did_full_forward {
            // The fused GPU layer path maintains its own device KV cache.
            self.note_device_kv_used();
        } else {
            // Host-KV path: every block reads the history out of `self.kv_cache`,
            // so it must actually contain that history (MET-05).
            self.require_host_kv_coherent(pos)?;
            run_blocks(
                &self.blocks,
                sliding_window,
                &mut scratch.hidden,
                pos,
                &mut self.kv_cache,
                &self.rope,
                kernel,
            )?;
            self.note_host_kv_written(pos);
        }
        let t_blocks_elapsed = t_blocks_start.elapsed();
        let t_norm_start = std::time::Instant::now();
        self.output_norm
            .forward(&scratch.hidden, &mut scratch.normed)?;
        let t_norm_elapsed = t_norm_start.elapsed();
        let t_lm_start = std::time::Instant::now();
        self.apply_lm_head(&scratch.normed, &mut logits[..vocab])?;
        let t_lm_elapsed = t_lm_start.elapsed();
        tracing::debug!(
            target : "fwd_profile",
            "pos={pos} blocks={:.1}ms norm={:.1}ms lm_head={:.1}ms gpu={}",
            t_blocks_elapsed.as_secs_f64() * 1000.0, t_norm_elapsed.as_secs_f64() *
            1000.0, t_lm_elapsed.as_secs_f64() * 1000.0, did_full_forward,
        );
        Ok(())
    }
}

/// Run every Transformer block over `hidden`, honouring the configured
/// sliding-window span (M-17).
///
/// A free function rather than a method because all three call sites hold
/// disjoint borrows of `self` at once (`&self.blocks`, `&mut self.kv_cache`,
/// `&self.rope`), which only stays legal while they are separate arguments.
///
/// `sliding_window` is `Some(w)` exactly when the GGUF declared
/// `<arch>.attention.sliding_window = w`. In that case each block runs
/// [`TransformerBlock::forward_with_sliding_window`] over a local window of
/// the `w` most recent key positions; otherwise it runs the unchanged
/// [`TransformerBlock::forward`], so every model shipped today (none declares
/// the key) keeps a byte-identical forward pass.
///
/// # Why `sink_tokens: 0`
///
/// [`SlidingWindowConfig`] can also pin the first *N* positions into the
/// window as attention sinks. The GGUF key carries a width only — it is
/// llama.cpp's `n_swa`, a pure causal local window with no sinks — so
/// declaring any sink here would silently attend to positions the reference
/// implementation does not. Sinks stay reachable through
/// [`crate::layers::sliding_window`] for callers that genuinely want them.
pub(super) fn run_blocks(
    blocks: &[TransformerBlock<'_>],
    sliding_window: Option<usize>,
    hidden: &mut [f32],
    pos: usize,
    kv_cache: &mut KvCache,
    rope: &RopeTable,
    kernel: &dyn OneBitKernel,
) -> ModelResult<()> {
    match sliding_window {
        None => {
            for block in blocks {
                block.forward(hidden, pos, kv_cache, rope, kernel)?;
            }
        }
        Some(window) => {
            let sw = crate::layers::sliding_window::SlidingWindowConfig::new(window, 0);
            for block in blocks {
                block.forward_with_sliding_window(
                    hidden,
                    pos,
                    kv_cache,
                    rope,
                    kernel,
                    Some(&sw),
                )?;
            }
        }
    }
    Ok(())
}
