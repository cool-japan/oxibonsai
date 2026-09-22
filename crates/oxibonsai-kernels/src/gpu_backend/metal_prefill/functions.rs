//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use metal::{MTLResourceOptions, MTLSize};
use std::sync::Arc;

use super::super::metal_full_layer::GpuKvCache;
use super::super::metal_graph::{
    alloc_buf, commit_and_wait, div_ceil, download_f32, set_scalar, upload_f32, MetalGraph,
    MetalGraphError, MetalWeightHandle,
};

use super::attention::{
    batched_attention_supported, prefill_batch_capacity, PrefillAttnDims, PrefillAttnPipelines,
    PrefillBufferCache,
};
use super::types::{LayerConfig, LayerWeightRefs, PrefillBuffers};

/// Smallest prefill batch routed through the `v10` simdgroup ternary GEMM
/// (perf-02).
///
/// `v10`'s output tile is `64x64` columns/rows; a batch below one tile pays
/// full staging cost for mostly-padding columns, where `v7`'s 8-column chunked
/// grid is faster. 64 is also exactly the boundary that leaves the speculative
/// verify path (batch = 4) and any short prompt on the kernel they have always
/// used.
const PREFILL_TQ2_V10_MIN_BATCH: usize = 64;

impl MetalGraph {
    /// Acquire the prefill buffer set, allocating if needed.
    ///
    /// **Bucketed and grow-only** (perf-M3). The set used to be cached
    /// exact-match on `batch_size`, so every distinct prompt length threw away
    /// and reallocated ~172 MB of Metal buffers. It is now allocated for a
    /// capacity rounded up to [`prefill_batch_capacity`] and reused by any
    /// smaller batch at the same model dimensions; when it must grow, the new
    /// capacity is the larger of the resident one and the requested bucket, so
    /// a sweep of prompt lengths never oscillates. The column-major layout
    /// makes an oversized set exactly equivalent to a tight one — a batch of
    /// `n` reads and writes only the first `n` columns.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn acquire_prefill_buffers(
        &self,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<PrefillBufferCache>>, MetalGraphError> {
        let mut guard = self.prefill_buffers.lock().map_err(|_| {
            MetalGraphError::ExecutionFailed("prefill_buffers lock poisoned".into())
        })?;
        // `None` ⇒ the resident set already serves this call; `Some(cap)` ⇒
        // allocate at `cap`, never below what is already resident for these
        // same dimensions.
        let needed_capacity = match guard.as_ref() {
            Some(cache)
                if cache.serves(
                    batch_size,
                    hidden_size,
                    intermediate_size,
                    nq,
                    nkv,
                    head_dim,
                    max_seq,
                ) =>
            {
                None
            }
            Some(cache) => Some(prefill_batch_capacity(batch_size).max(cache.growth_floor(
                hidden_size,
                intermediate_size,
                nq,
                nkv,
                head_dim,
                max_seq,
            ))),
            None => Some(prefill_batch_capacity(batch_size)),
        };
        if let Some(capacity) = needed_capacity {
            *guard = Some(PrefillBufferCache::allocate(
                &self.device,
                capacity,
                hidden_size,
                intermediate_size,
                nq,
                nkv,
                head_dim,
                max_seq,
            )?);
        }
        Ok(guard)
    }
    /// Encode ALL transformer layers for batch prefill in a SINGLE command buffer.
    ///
    /// Like `encode_full_forward`, except:
    /// - `hidden_batch` is `batch_size × hidden_size` (all prompt tokens)
    /// - `cos_table`/`sin_table` are `batch_size × half_dim` (all positions' RoPE)
    /// - Each layer uses GEMM (not GEMV) for batched projections
    /// - Attention is batch-wide: two dispatches per layer (perf-01),
    ///   falling back to the per-token loop only for geometries the
    ///   batched kernels cannot serve
    /// - After all layers, only the LAST token feeds into final norm + LM head
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn encode_full_forward_prefill(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[(
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
        )],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        logits_out: Option<&mut Vec<f32>>,
        greedy_token_id_out: Option<&mut u32>,
    ) -> Result<(), MetalGraphError> {
        let half_dim = head_dim / 2;
        let f = std::mem::size_of::<f32>();
        if hidden_batch.len() < batch_size * hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden_batch too short: need {}, got {}",
                batch_size * hidden_size,
                hidden_batch.len()
            )));
        }
        if cos_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "cos_table too short: need {}, got {}",
                batch_size * half_dim,
                cos_table.len()
            )));
        }
        if sin_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "sin_table too short: need {}, got {}",
                batch_size * half_dim,
                sin_table.len()
            )));
        }
        if layer_weights.len() != n_layers {
            return Err(MetalGraphError::EncodingFailed(format!(
                "layer_weights length mismatch: need {n_layers}, got {}",
                layer_weights.len()
            )));
        }
        // Compile (once per process) the batched attention pipelines BEFORE
        // taking the prefill-buffer and KV-cache locks: a first-run MSL
        // compile under those locks would serialise every other GPU caller.
        // `None` ⇒ unsupported geometry or no pipelines ⇒ per-token fallback.
        let attn = if batched_attention_supported(nq, nkv, head_dim) {
            self.prefill_attn_pipelines()
        } else {
            None
        };
        let pb_guard = self.acquire_prefill_buffers(
            batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = &pb_guard
            .as_ref()
            .ok_or_else(|| {
                MetalGraphError::ExecutionFailed("prefill_buffers not allocated".into())
            })?
            .bufs;
        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden_batch[..batch_size * hidden_size]);
            upload_f32(&bufs.cos_buf, &cos_table[..batch_size * half_dim]);
            upload_f32(&bufs.sin_buf, &sin_table[..batch_size * half_dim]);
        }
        let config = LayerConfig {
            hidden_size,
            intermediate_size,
            n_q_heads: nq,
            n_kv_heads: nkv,
            head_dim,
            eps,
            max_seq_len,
        };
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        for (layer_idx, weights) in layer_weights.iter().enumerate() {
            let layer_refs = LayerWeightRefs {
                attn_norm: &weights.0.buffer,
                qkv: &weights.1.buffer,
                q_norm: &weights.2.buffer,
                k_norm: &weights.3.buffer,
                output_proj: &weights.4.buffer,
                ffn_norm: &weights.5.buffer,
                gate_up: &weights.6.buffer,
                down: &weights.7.buffer,
            };
            self.encode_layer_prefill(
                encoder,
                bufs,
                kv,
                &layer_refs,
                layer_idx,
                batch_size,
                pos_start,
                &config,
                attn,
            )?;
        }
        match (final_norm_w, lm_head_w) {
            (Some(fnorm_w), Some(lm_w)) if lm_head_out_features > 0 => {
                let h = hidden_size as u32;
                let last_token_offset = ((batch_size - 1) * hidden_size * f) as u64;
                encoder.set_compute_pipeline_state(&self.pipelines.rmsnorm_weighted_v2);
                encoder.set_buffer(0, Some(&bufs.hidden_buf), last_token_offset);
                encoder.set_buffer(1, Some(&fnorm_w.buffer), 0);
                encoder.set_buffer(2, Some(&bufs.normed_buf), 0);
                unsafe {
                    set_scalar(encoder, 3, &final_norm_eps);
                    set_scalar(encoder, 4, &h);
                }
                encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
                let mut lg = self.logits_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into())
                })?;
                let needed_bytes = (lm_head_out_features * f) as u64;
                if lg.as_ref().is_none_or(|b| b.length() < needed_bytes) {
                    *lg = Some(alloc_buf(
                        &self.device,
                        needed_bytes,
                        MTLResourceOptions::StorageModeShared,
                    )?);
                }
                let logits_buf = lg.as_ref().ok_or(MetalGraphError::BufferCreationFailed)?;
                self.dispatch_gemv_q1(
                    encoder,
                    &lm_w.buffer,
                    &bufs.normed_buf,
                    logits_buf,
                    lm_head_out_features as u32,
                    h,
                );
                if greedy_token_id_out.is_some() {
                    let mut tid_guard = self.token_id_buf.lock().map_err(|_| {
                        MetalGraphError::ExecutionFailed("token_id_buf lock poisoned".into())
                    })?;
                    let needed = std::mem::size_of::<u32>() as u64;
                    if tid_guard.as_ref().is_none_or(|b| b.length() < needed) {
                        *tid_guard = Some(alloc_buf(
                            &self.device,
                            needed,
                            MTLResourceOptions::StorageModeShared,
                        )?);
                    }
                    let token_id_buf_ref = tid_guard
                        .as_ref()
                        .ok_or(MetalGraphError::BufferCreationFailed)?;
                    self.dispatch_argmax(
                        encoder,
                        logits_buf,
                        token_id_buf_ref,
                        lm_head_out_features as u32,
                    );
                    encoder.end_encoding();
                    commit_and_wait(cmd_buf, "prefill_q1_greedy")?;
                    let token_id = unsafe { *(token_id_buf_ref.contents() as *const u32) };
                    if let Some(out) = greedy_token_id_out {
                        *out = token_id;
                    }
                } else {
                    encoder.end_encoding();
                    commit_and_wait(cmd_buf, "prefill_q1_logits")?;
                    if let Some(out) = logits_out {
                        out.resize(lm_head_out_features, 0.0);
                        unsafe { download_f32(logits_buf, out) };
                    }
                }
            }
            _ => {
                encoder.end_encoding();
                commit_and_wait(cmd_buf, "prefill_q1_no_lm_head")?;
            }
        }
        Ok(())
    }
    /// Encode full-forward prefill for **verification** (speculative decoding).
    ///
    /// Identical to `encode_full_forward_prefill` for the layer loop, but the
    /// tail runs final RMSNorm + LM-head GEMM on **all** batch positions and
    /// returns per-position argmax token IDs instead of single-token logits.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn encode_full_forward_prefill_verify(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[(
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
        )],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        batch_token_ids_out: &mut Vec<u32>,
    ) -> Result<(), MetalGraphError> {
        let half_dim = head_dim / 2;
        let f = std::mem::size_of::<f32>();
        if hidden_batch.len() < batch_size * hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden_batch too short: need {}, got {}",
                batch_size * hidden_size,
                hidden_batch.len()
            )));
        }
        if cos_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "cos_table too short: need {}, got {}",
                batch_size * half_dim,
                cos_table.len()
            )));
        }
        if sin_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "sin_table too short: need {}, got {}",
                batch_size * half_dim,
                sin_table.len()
            )));
        }
        if layer_weights.len() != n_layers {
            return Err(MetalGraphError::EncodingFailed(format!(
                "layer_weights length mismatch: need {n_layers}, got {}",
                layer_weights.len()
            )));
        }
        // Compile (once per process) the batched attention pipelines BEFORE
        // taking the prefill-buffer and KV-cache locks: a first-run MSL
        // compile under those locks would serialise every other GPU caller.
        // `None` ⇒ unsupported geometry or no pipelines ⇒ per-token fallback.
        let attn = if batched_attention_supported(nq, nkv, head_dim) {
            self.prefill_attn_pipelines()
        } else {
            None
        };
        let pb_guard = self.acquire_prefill_buffers(
            batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = &pb_guard
            .as_ref()
            .ok_or_else(|| {
                MetalGraphError::ExecutionFailed("prefill_buffers not allocated".into())
            })?
            .bufs;
        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden_batch[..batch_size * hidden_size]);
            upload_f32(&bufs.cos_buf, &cos_table[..batch_size * half_dim]);
            upload_f32(&bufs.sin_buf, &sin_table[..batch_size * half_dim]);
        }
        let config = LayerConfig {
            hidden_size,
            intermediate_size,
            n_q_heads: nq,
            n_kv_heads: nkv,
            head_dim,
            eps,
            max_seq_len,
        };
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        for (layer_idx, weights) in layer_weights.iter().enumerate() {
            let layer_refs = LayerWeightRefs {
                attn_norm: &weights.0.buffer,
                qkv: &weights.1.buffer,
                q_norm: &weights.2.buffer,
                k_norm: &weights.3.buffer,
                output_proj: &weights.4.buffer,
                ffn_norm: &weights.5.buffer,
                gate_up: &weights.6.buffer,
                down: &weights.7.buffer,
            };
            self.encode_layer_prefill(
                encoder,
                bufs,
                kv,
                &layer_refs,
                layer_idx,
                batch_size,
                pos_start,
                &config,
                attn,
            )?;
        }
        match (final_norm_w, lm_head_w) {
            (Some(fnorm_w), Some(lm_w)) if lm_head_out_features > 0 => {
                let h = hidden_size as u32;
                self.dispatch_batched_rmsnorm(
                    encoder,
                    &bufs.hidden_buf,
                    &fnorm_w.buffer,
                    &bufs.normed_buf,
                    final_norm_eps,
                    h,
                    batch_size as u32,
                );
                let mut lg = self.logits_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into())
                })?;
                let needed_bytes = (batch_size * lm_head_out_features * f) as u64;
                if lg.as_ref().is_none_or(|b| b.length() < needed_bytes) {
                    *lg = Some(alloc_buf(
                        &self.device,
                        needed_bytes,
                        MTLResourceOptions::StorageModeShared,
                    )?);
                }
                let logits_buf = lg.as_ref().ok_or(MetalGraphError::BufferCreationFailed)?;
                self.dispatch_gemm_q1_v7(
                    encoder,
                    &lm_w.buffer,
                    &bufs.normed_buf,
                    logits_buf,
                    lm_head_out_features as u32,
                    h,
                    batch_size as u32,
                );
                let mut tid_guard = self.token_id_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("token_id_buf lock poisoned".into())
                })?;
                let needed = (batch_size * std::mem::size_of::<u32>()) as u64;
                if tid_guard.as_ref().is_none_or(|b| b.length() < needed) {
                    *tid_guard = Some(alloc_buf(
                        &self.device,
                        needed,
                        MTLResourceOptions::StorageModeShared,
                    )?);
                }
                let token_id_buf_ref = tid_guard
                    .as_ref()
                    .ok_or(MetalGraphError::BufferCreationFailed)?;
                let vocab = lm_head_out_features as u32;
                let f32_size = std::mem::size_of::<f32>() as u64;
                let u32_size = std::mem::size_of::<u32>() as u64;
                for col in 0..batch_size {
                    let logit_offset = col as u64 * vocab as u64 * f32_size;
                    let tid_offset = col as u64 * u32_size;
                    encoder.set_compute_pipeline_state(&self.pipelines.argmax);
                    encoder.set_buffer(0, Some(logits_buf), logit_offset);
                    encoder.set_buffer(1, Some(token_id_buf_ref), tid_offset);
                    unsafe {
                        set_scalar(encoder, 2, &vocab);
                    }
                    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
                }
                encoder.end_encoding();
                commit_and_wait(cmd_buf, "prefill_q1_verify")?;
                batch_token_ids_out.clear();
                batch_token_ids_out.reserve(batch_size);
                unsafe {
                    let ptr = token_id_buf_ref.contents() as *const u32;
                    for col in 0..batch_size {
                        batch_token_ids_out.push(*ptr.add(col));
                    }
                }
            }
            _ => {
                encoder.end_encoding();
                commit_and_wait(cmd_buf, "prefill_q1_verify_no_lm_head")?;
            }
        }
        Ok(())
    }
    /// Encode a single transformer layer for batch prefill (multiple tokens).
    ///
    /// Non-attention operations (RMSNorm, QKV projection, FFN) use batched GEMM
    /// kernels to process all tokens in parallel. Attention is batch-wide too
    /// since perf-01: `prefill_qkv_prepare` + `prefill_flash_attention`, two
    /// dispatches for the whole prompt, with the causal mask applied in-kernel.
    /// [`Self::encode_attention_per_token`] remains the fallback for geometries
    /// the batched kernels cannot serve.
    ///
    /// The hidden state is read from and written to `bufs.hidden_buf` in-place
    /// via residual-add GEMM variants.
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_prefill(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &PrefillBuffers,
        kv: &GpuKvCache,
        layer_weights: &LayerWeightRefs<'_>,
        layer_idx: usize,
        batch_size: usize,
        pos_start: usize,
        config: &LayerConfig,
        attn: Option<&PrefillAttnPipelines>,
    ) -> Result<(), MetalGraphError> {
        let h = config.hidden_size;
        let nq = config.n_q_heads;
        let nkv = config.n_kv_heads;
        let hd = config.head_dim;
        let inter = config.intermediate_size;
        let qkv_out = (nq + 2 * nkv) * hd;
        let bs = batch_size as u32;
        let inv_sqrt_hd = 1.0f32 / (hd as f32).sqrt();
        let heads_per_group = (nq / nkv) as u32;
        let cache_layer_offset = kv.layer_offset_elements(layer_idx);
        self.dispatch_batched_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            layer_weights.attn_norm,
            &bufs.normed_buf,
            config.eps,
            h as u32,
            bs,
        );
        self.dispatch_gemm_q1_v7(
            encoder,
            layer_weights.qkv,
            &bufs.normed_buf,
            &bufs.qkv_buf,
            qkv_out as u32,
            h as u32,
            bs,
        );
        // ── Attention phase ───────────────────────────────────────────────
        // perf-01: the whole prompt in TWO batch-wide dispatches, replacing
        // the `6 x batch` tiny dispatches this used to encode. The per-token
        // fallback is the historical path and runs only when the batched
        // kernels are unavailable for this geometry or device.
        let attn_dims = PrefillAttnDims {
            nq: nq as u32,
            nkv: nkv as u32,
            heads_per_group,
            head_dim: hd as u32,
            eps: config.eps,
            max_seq: config.max_seq_len as u32,
            pos_start: pos_start as u32,
            batch_size: bs,
            scale: inv_sqrt_hd,
            layer_offset: cache_layer_offset,
        };
        match attn {
            Some(pipelines) => self.encode_attention_batched(
                encoder,
                bufs,
                kv,
                layer_weights,
                pipelines,
                &attn_dims,
            ),
            None => self.encode_attention_per_token(
                encoder,
                bufs,
                kv,
                layer_weights,
                batch_size,
                pos_start,
                config,
                cache_layer_offset,
            ),
        }
        self.dispatch_gemm_q1_v7_residual(
            encoder,
            layer_weights.output_proj,
            &bufs.attn_out_buf,
            &bufs.hidden_buf,
            h as u32,
            (nq * hd) as u32,
            bs,
            &bufs.hidden_buf,
        );
        self.dispatch_batched_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            layer_weights.ffn_norm,
            &bufs.normed_buf,
            config.eps,
            h as u32,
            bs,
        );
        self.dispatch_fused_gate_up_swiglu_gemm(
            encoder,
            layer_weights.gate_up,
            &bufs.normed_buf,
            &bufs.swiglu_buf,
            inter as u32,
            h as u32,
            bs,
        );
        self.dispatch_gemm_q1_v7_residual(
            encoder,
            layer_weights.down,
            &bufs.swiglu_buf,
            &bufs.hidden_buf,
            h as u32,
            inter as u32,
            bs,
            &bufs.hidden_buf,
        );
        Ok(())
    }
    /// Encode a single transformer layer for batch prefill — ternary
    /// (TQ2_0_g128) variant.
    ///
    /// Mirrors [`Self::encode_layer_prefill`] step-for-step: same RMSNorm,
    /// same batch-wide attention phase, same residual structure. The only
    /// difference is that every weight GEMM dispatches through
    /// [`Self::dispatch_gemm_tq2_prefill`] (`v10` at prompt batch sizes,
    /// `v7` below the tile) instead of `dispatch_gemm_q1_v7`, and
    /// because TQ2 has no fused residual / fused gate+up+SwiGLU GEMM kernel
    /// the corresponding sites expand to two dispatches each:
    ///
    /// - Q1 `gemm_q1_v7_residual(W, x, hidden)` →
    ///   `gemm_tq2(W, x, normed_buf)` + `residual_add(hidden, normed_buf)`.
    /// - Q1 `fused_gate_up_swiglu_gemm_q1(W, x, swiglu_buf)` →
    ///   `gemm_tq2(W, x, gate_up_buf [n_rows = 2·inter])`
    ///   + `batched_swiglu(gate_up_buf, swiglu_buf)`.
    ///
    /// `normed_buf` is reused as the scratch destination for both residual
    /// adds — it is overwritten by the next layer's RMSNorm anyway, and
    /// during the final layer the residual_add is the last write so its
    /// staleness does not matter.
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_prefill_ternary(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &PrefillBuffers,
        kv: &GpuKvCache,
        layer_weights: &LayerWeightRefs<'_>,
        layer_idx: usize,
        batch_size: usize,
        pos_start: usize,
        config: &LayerConfig,
        attn: Option<&PrefillAttnPipelines>,
    ) -> Result<(), MetalGraphError> {
        let h = config.hidden_size;
        let nq = config.n_q_heads;
        let nkv = config.n_kv_heads;
        let hd = config.head_dim;
        let inter = config.intermediate_size;
        let qkv_out = (nq + 2 * nkv) * hd;
        let bs = batch_size as u32;
        let inv_sqrt_hd = 1.0f32 / (hd as f32).sqrt();
        let heads_per_group = (nq / nkv) as u32;
        let cache_layer_offset = kv.layer_offset_elements(layer_idx);
        self.dispatch_batched_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            layer_weights.attn_norm,
            &bufs.normed_buf,
            config.eps,
            h as u32,
            bs,
        );
        self.dispatch_gemm_tq2_prefill(
            encoder,
            layer_weights.qkv,
            &bufs.normed_buf,
            &bufs.qkv_buf,
            qkv_out as u32,
            h as u32,
            bs,
        );
        // ── Attention phase ───────────────────────────────────────────────
        // perf-01: the whole prompt in TWO batch-wide dispatches, replacing
        // the `6 x batch` tiny dispatches this used to encode. The per-token
        // fallback is the historical path and runs only when the batched
        // kernels are unavailable for this geometry or device.
        let attn_dims = PrefillAttnDims {
            nq: nq as u32,
            nkv: nkv as u32,
            heads_per_group,
            head_dim: hd as u32,
            eps: config.eps,
            max_seq: config.max_seq_len as u32,
            pos_start: pos_start as u32,
            batch_size: bs,
            scale: inv_sqrt_hd,
            layer_offset: cache_layer_offset,
        };
        match attn {
            Some(pipelines) => self.encode_attention_batched(
                encoder,
                bufs,
                kv,
                layer_weights,
                pipelines,
                &attn_dims,
            ),
            None => self.encode_attention_per_token(
                encoder,
                bufs,
                kv,
                layer_weights,
                batch_size,
                pos_start,
                config,
                cache_layer_offset,
            ),
        }
        self.dispatch_gemm_tq2_prefill(
            encoder,
            layer_weights.output_proj,
            &bufs.attn_out_buf,
            &bufs.normed_buf,
            h as u32,
            (nq * hd) as u32,
            bs,
        );
        self.dispatch_residual_add(
            encoder,
            &bufs.hidden_buf,
            &bufs.normed_buf,
            (batch_size * h) as u32,
        );
        self.dispatch_batched_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            layer_weights.ffn_norm,
            &bufs.normed_buf,
            config.eps,
            h as u32,
            bs,
        );
        self.dispatch_gemm_tq2_prefill(
            encoder,
            layer_weights.gate_up,
            &bufs.normed_buf,
            &bufs.gate_up_buf,
            (2 * inter) as u32,
            h as u32,
            bs,
        );
        self.dispatch_batched_swiglu(
            encoder,
            &bufs.gate_up_buf,
            &bufs.swiglu_buf,
            inter as u32,
            bs,
        );
        self.dispatch_gemm_tq2_prefill(
            encoder,
            layer_weights.down,
            &bufs.swiglu_buf,
            &bufs.normed_buf,
            h as u32,
            inter as u32,
            bs,
        );
        self.dispatch_residual_add(
            encoder,
            &bufs.hidden_buf,
            &bufs.normed_buf,
            (batch_size * h) as u32,
        );
        Ok(())
    }
    /// Encode ALL transformer layers for batch prefill in a SINGLE command
    /// buffer — ternary (TQ2_0_g128) variant.
    ///
    /// Mirror of [`Self::encode_full_forward_prefill`] but with weight GEMMs
    /// routed through the TQ2 batched kernels. The final-norm + LM-head tail
    /// also uses TQ2 GEMV (`dispatch_gemv_tq2`) for the LM head.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn encode_full_forward_prefill_ternary(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[(
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
        )],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        logits_out: Option<&mut Vec<f32>>,
        greedy_token_id_out: Option<&mut u32>,
    ) -> Result<(), MetalGraphError> {
        let half_dim = head_dim / 2;
        let f = std::mem::size_of::<f32>();
        if hidden_batch.len() < batch_size * hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden_batch too short: need {}, got {}",
                batch_size * hidden_size,
                hidden_batch.len()
            )));
        }
        if cos_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "cos_table too short: need {}, got {}",
                batch_size * half_dim,
                cos_table.len()
            )));
        }
        if sin_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "sin_table too short: need {}, got {}",
                batch_size * half_dim,
                sin_table.len()
            )));
        }
        if layer_weights.len() != n_layers {
            return Err(MetalGraphError::EncodingFailed(format!(
                "layer_weights length mismatch: need {n_layers}, got {}",
                layer_weights.len()
            )));
        }
        // Compile (once per process) the batched attention pipelines BEFORE
        // taking the prefill-buffer and KV-cache locks: a first-run MSL
        // compile under those locks would serialise every other GPU caller.
        // `None` ⇒ unsupported geometry or no pipelines ⇒ per-token fallback.
        let attn = if batched_attention_supported(nq, nkv, head_dim) {
            self.prefill_attn_pipelines()
        } else {
            None
        };
        let pb_guard = self.acquire_prefill_buffers(
            batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = &pb_guard
            .as_ref()
            .ok_or_else(|| {
                MetalGraphError::ExecutionFailed("prefill_buffers not allocated".into())
            })?
            .bufs;
        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden_batch[..batch_size * hidden_size]);
            upload_f32(&bufs.cos_buf, &cos_table[..batch_size * half_dim]);
            upload_f32(&bufs.sin_buf, &sin_table[..batch_size * half_dim]);
        }
        let config = LayerConfig {
            hidden_size,
            intermediate_size,
            n_q_heads: nq,
            n_kv_heads: nkv,
            head_dim,
            eps,
            max_seq_len,
        };
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        for (layer_idx, weights) in layer_weights.iter().enumerate() {
            let layer_refs = LayerWeightRefs {
                attn_norm: &weights.0.buffer,
                qkv: &weights.1.buffer,
                q_norm: &weights.2.buffer,
                k_norm: &weights.3.buffer,
                output_proj: &weights.4.buffer,
                ffn_norm: &weights.5.buffer,
                gate_up: &weights.6.buffer,
                down: &weights.7.buffer,
            };
            self.encode_layer_prefill_ternary(
                encoder,
                bufs,
                kv,
                &layer_refs,
                layer_idx,
                batch_size,
                pos_start,
                &config,
                attn,
            )?;
        }
        match (final_norm_w, lm_head_w) {
            (Some(fnorm_w), Some(lm_w)) if lm_head_out_features > 0 => {
                let h = hidden_size as u32;
                let last_token_offset = ((batch_size - 1) * hidden_size * f) as u64;
                encoder.set_compute_pipeline_state(&self.pipelines.rmsnorm_weighted_v2);
                encoder.set_buffer(0, Some(&bufs.hidden_buf), last_token_offset);
                encoder.set_buffer(1, Some(&fnorm_w.buffer), 0);
                encoder.set_buffer(2, Some(&bufs.normed_buf), 0);
                unsafe {
                    set_scalar(encoder, 3, &final_norm_eps);
                    set_scalar(encoder, 4, &h);
                }
                encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
                let mut lg = self.logits_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into())
                })?;
                let needed_bytes = (lm_head_out_features * f) as u64;
                if lg.as_ref().is_none_or(|b| b.length() < needed_bytes) {
                    *lg = Some(alloc_buf(
                        &self.device,
                        needed_bytes,
                        MTLResourceOptions::StorageModeShared,
                    )?);
                }
                let logits_buf = lg.as_ref().ok_or(MetalGraphError::BufferCreationFailed)?;
                self.dispatch_gemv_tq2(
                    encoder,
                    &lm_w.buffer,
                    &bufs.normed_buf,
                    logits_buf,
                    lm_head_out_features as u32,
                    h,
                );
                if greedy_token_id_out.is_some() {
                    let mut tid_guard = self.token_id_buf.lock().map_err(|_| {
                        MetalGraphError::ExecutionFailed("token_id_buf lock poisoned".into())
                    })?;
                    let needed = std::mem::size_of::<u32>() as u64;
                    if tid_guard.as_ref().is_none_or(|b| b.length() < needed) {
                        *tid_guard = Some(alloc_buf(
                            &self.device,
                            needed,
                            MTLResourceOptions::StorageModeShared,
                        )?);
                    }
                    let token_id_buf_ref = tid_guard
                        .as_ref()
                        .ok_or(MetalGraphError::BufferCreationFailed)?;
                    self.dispatch_argmax(
                        encoder,
                        logits_buf,
                        token_id_buf_ref,
                        lm_head_out_features as u32,
                    );
                    encoder.end_encoding();
                    commit_and_wait(cmd_buf, "prefill_ternary_greedy")?;
                    let token_id = unsafe { *(token_id_buf_ref.contents() as *const u32) };
                    if let Some(out) = greedy_token_id_out {
                        *out = token_id;
                    }
                } else {
                    encoder.end_encoding();
                    commit_and_wait(cmd_buf, "prefill_ternary_logits")?;
                    if let Some(out) = logits_out {
                        out.resize(lm_head_out_features, 0.0);
                        unsafe { download_f32(logits_buf, out) };
                    }
                }
            }
            _ => {
                encoder.end_encoding();
                commit_and_wait(cmd_buf, "prefill_ternary_no_lm_head")?;
            }
        }
        Ok(())
    }
    /// Encode ALL transformer layers for batch prefill — **verify** variant
    /// (speculative decoding), ternary (TQ2_0_g128) weights.
    ///
    /// Mirror of [`Self::encode_full_forward_prefill_verify`] but with TQ2
    /// GEMM for projections and TQ2 GEMV for the per-position LM head.
    /// The LM head is dispatched per column because the existing TQ2 GEMV
    /// is not batched; this matches what the per-position fused tail does
    /// internally.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn encode_full_forward_prefill_verify_ternary(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[(
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
        )],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        batch_token_ids_out: &mut Vec<u32>,
    ) -> Result<(), MetalGraphError> {
        let half_dim = head_dim / 2;
        let f = std::mem::size_of::<f32>();
        if hidden_batch.len() < batch_size * hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden_batch too short: need {}, got {}",
                batch_size * hidden_size,
                hidden_batch.len()
            )));
        }
        if cos_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "cos_table too short: need {}, got {}",
                batch_size * half_dim,
                cos_table.len()
            )));
        }
        if sin_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "sin_table too short: need {}, got {}",
                batch_size * half_dim,
                sin_table.len()
            )));
        }
        if layer_weights.len() != n_layers {
            return Err(MetalGraphError::EncodingFailed(format!(
                "layer_weights length mismatch: need {n_layers}, got {}",
                layer_weights.len()
            )));
        }
        // Compile (once per process) the batched attention pipelines BEFORE
        // taking the prefill-buffer and KV-cache locks: a first-run MSL
        // compile under those locks would serialise every other GPU caller.
        // `None` ⇒ unsupported geometry or no pipelines ⇒ per-token fallback.
        let attn = if batched_attention_supported(nq, nkv, head_dim) {
            self.prefill_attn_pipelines()
        } else {
            None
        };
        let pb_guard = self.acquire_prefill_buffers(
            batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = &pb_guard
            .as_ref()
            .ok_or_else(|| {
                MetalGraphError::ExecutionFailed("prefill_buffers not allocated".into())
            })?
            .bufs;
        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden_batch[..batch_size * hidden_size]);
            upload_f32(&bufs.cos_buf, &cos_table[..batch_size * half_dim]);
            upload_f32(&bufs.sin_buf, &sin_table[..batch_size * half_dim]);
        }
        let config = LayerConfig {
            hidden_size,
            intermediate_size,
            n_q_heads: nq,
            n_kv_heads: nkv,
            head_dim,
            eps,
            max_seq_len,
        };
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        for (layer_idx, weights) in layer_weights.iter().enumerate() {
            let layer_refs = LayerWeightRefs {
                attn_norm: &weights.0.buffer,
                qkv: &weights.1.buffer,
                q_norm: &weights.2.buffer,
                k_norm: &weights.3.buffer,
                output_proj: &weights.4.buffer,
                ffn_norm: &weights.5.buffer,
                gate_up: &weights.6.buffer,
                down: &weights.7.buffer,
            };
            self.encode_layer_prefill_ternary(
                encoder,
                bufs,
                kv,
                &layer_refs,
                layer_idx,
                batch_size,
                pos_start,
                &config,
                attn,
            )?;
        }
        match (final_norm_w, lm_head_w) {
            (Some(fnorm_w), Some(lm_w)) if lm_head_out_features > 0 => {
                let h = hidden_size as u32;
                self.dispatch_batched_rmsnorm(
                    encoder,
                    &bufs.hidden_buf,
                    &fnorm_w.buffer,
                    &bufs.normed_buf,
                    final_norm_eps,
                    h,
                    batch_size as u32,
                );
                let mut lg = self.logits_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into())
                })?;
                let needed_bytes = (batch_size * lm_head_out_features * f) as u64;
                if lg.as_ref().is_none_or(|b| b.length() < needed_bytes) {
                    *lg = Some(alloc_buf(
                        &self.device,
                        needed_bytes,
                        MTLResourceOptions::StorageModeShared,
                    )?);
                }
                let logits_buf = lg.as_ref().ok_or(MetalGraphError::BufferCreationFailed)?;
                let f32_size = std::mem::size_of::<f32>() as u64;
                for col in 0..batch_size {
                    let normed_offset = col as u64 * h as u64 * f32_size;
                    let logit_offset = col as u64 * lm_head_out_features as u64 * f32_size;
                    encoder.set_compute_pipeline_state(&self.pipelines.gemv_tq2_g128_v1);
                    encoder.set_buffer(0, Some(&lm_w.buffer), 0);
                    encoder.set_buffer(1, Some(&bufs.normed_buf), normed_offset);
                    encoder.set_buffer(2, Some(logits_buf), logit_offset);
                    unsafe {
                        set_scalar(encoder, 3, &(lm_head_out_features as u32));
                        set_scalar(encoder, 4, &h);
                    }
                    let tg_count = div_ceil(lm_head_out_features, 8);
                    encoder.dispatch_thread_groups(
                        MTLSize::new(tg_count as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
                let mut tid_guard = self.token_id_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("token_id_buf lock poisoned".into())
                })?;
                let needed = (batch_size * std::mem::size_of::<u32>()) as u64;
                if tid_guard.as_ref().is_none_or(|b| b.length() < needed) {
                    *tid_guard = Some(alloc_buf(
                        &self.device,
                        needed,
                        MTLResourceOptions::StorageModeShared,
                    )?);
                }
                let token_id_buf_ref = tid_guard
                    .as_ref()
                    .ok_or(MetalGraphError::BufferCreationFailed)?;
                let vocab = lm_head_out_features as u32;
                let u32_size = std::mem::size_of::<u32>() as u64;
                for col in 0..batch_size {
                    let logit_offset = col as u64 * vocab as u64 * f32_size;
                    let tid_offset = col as u64 * u32_size;
                    encoder.set_compute_pipeline_state(&self.pipelines.argmax);
                    encoder.set_buffer(0, Some(logits_buf), logit_offset);
                    encoder.set_buffer(1, Some(token_id_buf_ref), tid_offset);
                    unsafe {
                        set_scalar(encoder, 2, &vocab);
                    }
                    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
                }
                encoder.end_encoding();
                commit_and_wait(cmd_buf, "prefill_ternary_verify")?;
                batch_token_ids_out.clear();
                batch_token_ids_out.reserve(batch_size);
                unsafe {
                    let ptr = token_id_buf_ref.contents() as *const u32;
                    for col in 0..batch_size {
                        batch_token_ids_out.push(*ptr.add(col));
                    }
                }
            }
            _ => {
                encoder.end_encoding();
                commit_and_wait(cmd_buf, "prefill_ternary_verify_no_lm_head")?;
            }
        }
        Ok(())
    }
    /// Ternary (TQ2_0_g128) GEMM for the prefill path: `v10` for real prompt
    /// batches, `v7` below the tile (perf-02).
    ///
    /// `gemm_tq2_g128_v10_simdgroup` is the `simdgroup_matrix` kernel the DiT
    /// path already ships (parity-validated, ~3.8x `v9` on large `M`). It
    /// reads the **same** SoA weight buffer and the same column-major
    /// `inputs[col*k + elem]` / `outputs[col*n_rows + row]` layout as `v7`, so
    /// the swap is argument-for-argument — but its output tile is `64x64`, so
    /// below [`PREFILL_TQ2_V10_MIN_BATCH`] columns most of the tile is padding
    /// and `v7`'s 8-column chunking wins. That threshold also keeps the two
    /// small-batch callers on `v7` by construction: single-token decode (a
    /// GEMV, not this path at all) and speculative verify (batch = 4).
    ///
    /// `v10` stages the dequantized weight as `half` — exact for ternary
    /// `code x scale` — and accumulates in f32 in a different order from `v7`,
    /// so results are numerically equivalent (cos >= 0.999, max-abs ~1e-5) but
    /// not bit-identical.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_gemm_tq2_prefill(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        blocks: &metal::Buffer,
        inputs: &metal::Buffer,
        outputs: &metal::Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
    ) {
        if batch_size as usize >= PREFILL_TQ2_V10_MIN_BATCH {
            self.dispatch_gemm_tq2_v10(encoder, blocks, inputs, outputs, n_rows, k, batch_size);
        } else {
            self.dispatch_gemm_tq2_v7(encoder, blocks, inputs, outputs, n_rows, k, batch_size);
        }
    }

    /// Batched attention phase for one prefill layer — **two dispatches for
    /// the whole prompt** (perf-01).
    ///
    /// 1. `prefill_qkv_prepare` RMSNorms and rotates every token's Q and K
    ///    heads and writes K/V straight into the KV cache. Q is rewritten **in
    ///    place** over the Q section of `qkv_buf`, which is why this path needs
    ///    no buffer the per-token path did not already have.
    /// 2. `prefill_flash_attention` computes causal GQA attention for every
    ///    `(token, head)` with an online softmax, so the `batch x n_ctx` score
    ///    matrix is never materialised, and writes `attn_out_buf`.
    ///
    /// Both dispatches go into the caller's encoder. Metal serialises
    /// dispatches within one compute encoder (the default
    /// `MTLDispatchTypeSerial`), so the flash kernel is guaranteed to see the
    /// keys and values the prepare kernel just stored — the same ordering
    /// guarantee the per-token path relied on between `fused_kv_store` and
    /// `batched_attention_scores_v2`.
    fn encode_attention_batched(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &PrefillBuffers,
        kv: &GpuKvCache,
        layer_weights: &LayerWeightRefs<'_>,
        pipelines: &PrefillAttnPipelines,
        dims: &PrefillAttnDims,
    ) {
        self.dispatch_prefill_qkv_prepare(
            pipelines,
            encoder,
            &bufs.qkv_buf,
            layer_weights.q_norm,
            layer_weights.k_norm,
            &bufs.cos_buf,
            &bufs.sin_buf,
            &kv.k_cache,
            &kv.v_cache,
            dims,
        );
        self.dispatch_prefill_flash_attention(
            pipelines,
            encoder,
            &bufs.qkv_buf,
            &kv.k_cache,
            &kv.v_cache,
            &bufs.attn_out_buf,
            dims,
        );
        Self::note_batched_attention_layer();
    }

    /// Historical per-token attention phase: six single-token dispatches per
    /// prompt token (`fused_qk_norm`, `fused_qk_rope`, `fused_kv_store`,
    /// `batched_attention_scores_v2`, `batched_softmax`,
    /// `batched_attention_weighted_sum`).
    ///
    /// Retained as the fallback for geometries the batched kernels cannot
    /// serve ([`batched_attention_supported`]) and for devices/toolchains
    /// where they fail to compile. It is the reference the batched path is
    /// parity-tested against, so it must stay byte-for-byte what it was.
    #[allow(clippy::too_many_arguments)]
    fn encode_attention_per_token(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &PrefillBuffers,
        kv: &GpuKvCache,
        layer_weights: &LayerWeightRefs<'_>,
        batch_size: usize,
        pos_start: usize,
        config: &LayerConfig,
        cache_layer_offset: u64,
    ) {
        let nq = config.n_q_heads;
        let nkv = config.n_kv_heads;
        let hd = config.head_dim;
        let qkv_out = (nq + 2 * nkv) * hd;
        let half_dim = hd / 2;
        let inv_sqrt_hd = 1.0f32 / (hd as f32).sqrt();
        let heads_per_group = (nq / nkv) as u32;
        let f = std::mem::size_of::<f32>();
        for t in 0..batch_size {
            let pos = pos_start + t;
            let seq_len = (pos + 1) as u32;
            let qkv_col_byte_offset = (t * qkv_out * f) as u64;
            let q_byte_offset = qkv_col_byte_offset;
            let k_byte_offset = qkv_col_byte_offset + (nq * hd * f) as u64;
            let v_byte_offset = qkv_col_byte_offset + ((nq + nkv) * hd * f) as u64;
            self.dispatch_fused_qk_norm(
                encoder,
                &bufs.qkv_buf,
                q_byte_offset,
                &bufs.qkv_buf,
                k_byte_offset,
                &bufs.q_normed_buf,
                &bufs.k_normed_buf,
                layer_weights.q_norm,
                layer_weights.k_norm,
                nq as u32,
                nkv as u32,
                hd as u32,
                config.eps,
            );
            let rope_byte_offset = (t * half_dim * f) as u64;
            {
                encoder.set_compute_pipeline_state(&self.pipelines.fused_qk_rope);
                encoder.set_buffer(0, Some(&bufs.q_normed_buf), 0);
                encoder.set_buffer(1, Some(&bufs.k_normed_buf), 0);
                encoder.set_buffer(2, Some(&bufs.q_rope_buf), 0);
                encoder.set_buffer(3, Some(&bufs.k_rope_buf), 0);
                encoder.set_buffer(4, Some(&bufs.cos_buf), rope_byte_offset);
                encoder.set_buffer(5, Some(&bufs.sin_buf), rope_byte_offset);
                unsafe {
                    set_scalar(encoder, 6, &(nq as u32));
                    set_scalar(encoder, 7, &(nkv as u32));
                    set_scalar(encoder, 8, &(half_dim as u32));
                }
                let tg_x = div_ceil(half_dim, 64) as u64;
                encoder.dispatch_thread_groups(
                    MTLSize::new(tg_x, (nq + nkv) as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            }
            self.dispatch_fused_kv_store(
                encoder,
                &bufs.k_rope_buf,
                &bufs.qkv_buf,
                v_byte_offset,
                &kv.k_cache,
                &kv.v_cache,
                nkv as u32,
                hd as u32,
                config.max_seq_len as u32,
                pos as u32,
                cache_layer_offset,
            );
            {
                self.dispatch_attention_scores_v2(
                    encoder,
                    &bufs.q_rope_buf,
                    &kv.k_cache,
                    &bufs.scores_buf,
                    hd as u32,
                    nq as u32,
                    nkv as u32,
                    heads_per_group,
                    config.max_seq_len as u32,
                    seq_len,
                    inv_sqrt_hd,
                    cache_layer_offset,
                );
            }
            {
                encoder.set_compute_pipeline_state(&self.pipelines.batched_softmax);
                encoder.set_buffer(0, Some(&bufs.scores_buf), 0);
                unsafe {
                    set_scalar(encoder, 1, &(nq as u32));
                    set_scalar(encoder, 2, &(config.max_seq_len as u32));
                    set_scalar(encoder, 3, &seq_len);
                }
                encoder
                    .dispatch_thread_groups(MTLSize::new(nq as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            {
                let attn_col_byte_offset = (t * nq * hd * f) as u64;
                encoder.set_compute_pipeline_state(&self.pipelines.batched_attention_weighted_sum);
                encoder.set_buffer(0, Some(&bufs.scores_buf), 0);
                encoder.set_buffer(1, Some(&kv.v_cache), 0);
                encoder.set_buffer(2, Some(&bufs.attn_out_buf), attn_col_byte_offset);
                unsafe {
                    set_scalar(encoder, 3, &(hd as u32));
                    set_scalar(encoder, 4, &(nq as u32));
                    set_scalar(encoder, 5, &(nkv as u32));
                    set_scalar(encoder, 6, &heads_per_group);
                    set_scalar(encoder, 7, &(config.max_seq_len as u32));
                    set_scalar(encoder, 8, &seq_len);
                    set_scalar(encoder, 9, &cache_layer_offset);
                }
                let tg_x = div_ceil(hd, 64) as u64;
                encoder.dispatch_thread_groups(
                    MTLSize::new(tg_x, nq as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            }
        }
    }
}
