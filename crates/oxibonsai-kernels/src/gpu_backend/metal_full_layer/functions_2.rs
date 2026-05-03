//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::metal_graph::{
    alloc_buf, div_ceil, download_f32, set_scalar, upload_f32, MetalGraph, MetalGraphError,
    MetalWeightHandle,
};
use super::functions::gpu_profile;
use super::types::{FullLayerBuffers, GpuKvCache};
use metal::{MTLResourceOptions, MTLSize};
use std::sync::Arc;

impl MetalGraph {
    /// Get a cached weight or upload f32 data (e.g. norm weights) and cache it.
    ///
    /// Unlike `get_or_upload_weight` which uploads raw bytes (Q1 blocks),
    /// this reinterprets `&[f32]` as `&[u8]` for the Metal buffer.
    pub fn get_or_upload_f32_weight(
        &self,
        key: u64,
        data: &[f32],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.get_or_upload_weight(key, byte_slice)
    }
    /// Acquire the full-layer buffer set, allocating if needed.
    fn acquire_full_layer_buffers(
        &self,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<FullLayerBuffers>>, MetalGraphError> {
        let mut guard = self.full_layer_buffers.lock().map_err(|_| {
            MetalGraphError::ExecutionFailed("full_layer_buffers lock poisoned".into())
        })?;
        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.matches(hidden_size, intermediate_size, nq, nkv, head_dim, max_seq),
            None => true,
        };
        if needs_alloc {
            *guard = Some(FullLayerBuffers::allocate(
                &self.device,
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
    /// Acquire the KV cache, allocating if needed.
    pub(crate) fn acquire_kv_cache(
        &self,
        n_layers: usize,
        n_kv: usize,
        max_seq: usize,
        head_dim: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<GpuKvCache>>, MetalGraphError> {
        let mut guard = self
            .kv_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("kv_cache lock poisoned".into()))?;
        let needs_alloc = match guard.as_ref() {
            Some(c) => !c.matches(n_layers, n_kv, max_seq, head_dim),
            None => true,
        };
        if needs_alloc {
            *guard = Some(GpuKvCache::allocate(
                &self.device,
                n_layers,
                n_kv,
                max_seq,
                head_dim,
            )?);
        }
        Ok(guard)
    }
    /// Encode a single transformer layer's dispatches into an existing encoder.
    ///
    /// This is the core encoding logic extracted from `encode_full_layer`.
    /// It encodes all 18 GPU dispatches (attention + FFN) for one layer
    /// without creating/committing a command buffer. The hidden state is
    /// read from and written to `bufs.hidden_buf` in-place (via residual adds).
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_into(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &FullLayerBuffers,
        kv: &GpuKvCache,
        layer_idx: usize,
        pos: usize,
        attn_norm_w: &MetalWeightHandle,
        fused_qkv_w: &MetalWeightHandle,
        q_norm_w: &MetalWeightHandle,
        k_norm_w: &MetalWeightHandle,
        attn_proj_w: &MetalWeightHandle,
        ffn_norm_w: &MetalWeightHandle,
        gate_up_w: &MetalWeightHandle,
        down_w: &MetalWeightHandle,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<(), MetalGraphError> {
        let seq_len = (pos + 1) as u32;
        let inv_sqrt_hd = 1.0f32 / (head_dim as f32).sqrt();
        let heads_per_group = (nq / nkv) as u32;
        let h = hidden_size as u32;
        let inter = intermediate_size as u32;
        let qkv_total_rows = (nq * head_dim + 2 * nkv * head_dim) as u32;
        let cache_layer_offset = kv.layer_offset_elements(layer_idx);
        self.dispatch_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            &attn_norm_w.buffer,
            &bufs.normed_buf,
            eps,
            h,
        );
        self.dispatch_gemv_q1(
            encoder,
            &fused_qkv_w.buffer,
            &bufs.normed_buf,
            &bufs.qkv_buf,
            qkv_total_rows,
            h,
        );
        {
            let q_offset: u64 = 0;
            let k_offset = (nq * head_dim * std::mem::size_of::<f32>()) as u64;
            self.dispatch_fused_qk_norm_rope(
                encoder,
                &bufs.qkv_buf,
                q_offset,
                &bufs.qkv_buf,
                k_offset,
                &bufs.q_rope_buf,
                &bufs.k_rope_buf,
                &q_norm_w.buffer,
                &k_norm_w.buffer,
                &bufs.cos_buf,
                &bufs.sin_buf,
                nq as u32,
                nkv as u32,
                head_dim as u32,
                eps,
            );
        }
        {
            let v_offset = ((nq * head_dim + nkv * head_dim) * std::mem::size_of::<f32>()) as u64;
            self.dispatch_fused_kv_store(
                encoder,
                &bufs.k_rope_buf,
                &bufs.qkv_buf,
                v_offset,
                &kv.k_cache,
                &kv.v_cache,
                nkv as u32,
                head_dim as u32,
                max_seq_len as u32,
                pos as u32,
                cache_layer_offset,
            );
        }
        {
            {
                self.dispatch_attention_scores_v2(
                    encoder,
                    &bufs.q_rope_buf,
                    &kv.k_cache,
                    &bufs.scores_buf,
                    head_dim as u32,
                    nq as u32,
                    nkv as u32,
                    heads_per_group,
                    max_seq_len as u32,
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
                    set_scalar(encoder, 2, &(max_seq_len as u32));
                    set_scalar(encoder, 3, &seq_len);
                }
                encoder
                    .dispatch_thread_groups(MTLSize::new(nq as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            {
                encoder.set_compute_pipeline_state(&self.pipelines.batched_attention_weighted_sum);
                encoder.set_buffer(0, Some(&bufs.scores_buf), 0);
                encoder.set_buffer(1, Some(&kv.v_cache), 0);
                encoder.set_buffer(2, Some(&bufs.attn_out_buf), 0);
                unsafe {
                    set_scalar(encoder, 3, &(head_dim as u32));
                    set_scalar(encoder, 4, &(nq as u32));
                    set_scalar(encoder, 5, &(nkv as u32));
                    set_scalar(encoder, 6, &heads_per_group);
                    set_scalar(encoder, 7, &(max_seq_len as u32));
                    set_scalar(encoder, 8, &seq_len);
                    set_scalar(encoder, 9, &cache_layer_offset);
                }
                let tg_x = div_ceil(head_dim, 64) as u64;
                encoder.dispatch_thread_groups(
                    MTLSize::new(tg_x, nq as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            }
        }
        self.dispatch_gemv_q1_residual(
            encoder,
            &attn_proj_w.buffer,
            &bufs.attn_out_buf,
            &bufs.hidden_buf,
            h,
            (nq * head_dim) as u32,
            &bufs.hidden_buf,
        );
        self.dispatch_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            &ffn_norm_w.buffer,
            &bufs.normed_buf,
            eps,
            h,
        );
        self.dispatch_fused_gate_up_swiglu(
            encoder,
            &gate_up_w.buffer,
            &bufs.normed_buf,
            &bufs.swiglu_buf,
            inter,
            h,
        );
        self.dispatch_gemv_q1_residual(
            encoder,
            &down_w.buffer,
            &bufs.swiglu_buf,
            &bufs.hidden_buf,
            h,
            inter,
            &bufs.hidden_buf,
        );
        Ok(())
    }
    /// Ternary (TQ2_0_g128) twin of [`Self::encode_layer_into`].
    ///
    /// Byte-for-byte identical to the 1-bit encoder, but every matrix
    /// multiplication is routed through `dispatch_gemv_tq2` instead of
    /// `dispatch_gemv_q1`. Because TQ2 lacks fused residual-add and fused
    /// gate+up+SwiGLU kernels, those sites expand to two dispatches each:
    ///
    /// - Q1 `gemv_q1_residual(W, x, hidden)` → `gemv_tq2(W, x, tmp)` + `residual_add(hidden, tmp)`.
    /// - Q1 `fused_gate_up_swiglu(W, x, swiglu_buf)` → `gemv_tq2(W, x, gate_up_buf[2·inter])`
    ///   + `swiglu_single(gate_up_buf, swiglu_buf, inter)`.
    ///
    /// `attn_out_buf` is re-used as the scratch destination for the attention
    /// projection GEMV (it is no longer needed after the attention sublayer).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_into_ternary(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &FullLayerBuffers,
        kv: &GpuKvCache,
        layer_idx: usize,
        pos: usize,
        attn_norm_w: &MetalWeightHandle,
        fused_qkv_w: &MetalWeightHandle,
        q_norm_w: &MetalWeightHandle,
        k_norm_w: &MetalWeightHandle,
        attn_proj_w: &MetalWeightHandle,
        ffn_norm_w: &MetalWeightHandle,
        gate_up_w: &MetalWeightHandle,
        down_w: &MetalWeightHandle,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<(), MetalGraphError> {
        let seq_len = (pos + 1) as u32;
        let inv_sqrt_hd = 1.0f32 / (head_dim as f32).sqrt();
        let heads_per_group = (nq / nkv) as u32;
        let h = hidden_size as u32;
        let inter = intermediate_size as u32;
        let qkv_total_rows = (nq * head_dim + 2 * nkv * head_dim) as u32;
        let cache_layer_offset = kv.layer_offset_elements(layer_idx);
        self.dispatch_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            &attn_norm_w.buffer,
            &bufs.normed_buf,
            eps,
            h,
        );
        self.dispatch_gemv_tq2(
            encoder,
            &fused_qkv_w.buffer,
            &bufs.normed_buf,
            &bufs.qkv_buf,
            qkv_total_rows,
            h,
        );
        {
            let q_offset: u64 = 0;
            let k_offset = (nq * head_dim * std::mem::size_of::<f32>()) as u64;
            self.dispatch_fused_qk_norm_rope(
                encoder,
                &bufs.qkv_buf,
                q_offset,
                &bufs.qkv_buf,
                k_offset,
                &bufs.q_rope_buf,
                &bufs.k_rope_buf,
                &q_norm_w.buffer,
                &k_norm_w.buffer,
                &bufs.cos_buf,
                &bufs.sin_buf,
                nq as u32,
                nkv as u32,
                head_dim as u32,
                eps,
            );
        }
        {
            let v_offset = ((nq * head_dim + nkv * head_dim) * std::mem::size_of::<f32>()) as u64;
            self.dispatch_fused_kv_store(
                encoder,
                &bufs.k_rope_buf,
                &bufs.qkv_buf,
                v_offset,
                &kv.k_cache,
                &kv.v_cache,
                nkv as u32,
                head_dim as u32,
                max_seq_len as u32,
                pos as u32,
                cache_layer_offset,
            );
        }
        {
            {
                self.dispatch_attention_scores_v2(
                    encoder,
                    &bufs.q_rope_buf,
                    &kv.k_cache,
                    &bufs.scores_buf,
                    head_dim as u32,
                    nq as u32,
                    nkv as u32,
                    heads_per_group,
                    max_seq_len as u32,
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
                    set_scalar(encoder, 2, &(max_seq_len as u32));
                    set_scalar(encoder, 3, &seq_len);
                }
                encoder
                    .dispatch_thread_groups(MTLSize::new(nq as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            {
                encoder.set_compute_pipeline_state(&self.pipelines.batched_attention_weighted_sum);
                encoder.set_buffer(0, Some(&bufs.scores_buf), 0);
                encoder.set_buffer(1, Some(&kv.v_cache), 0);
                encoder.set_buffer(2, Some(&bufs.attn_out_buf), 0);
                unsafe {
                    set_scalar(encoder, 3, &(head_dim as u32));
                    set_scalar(encoder, 4, &(nq as u32));
                    set_scalar(encoder, 5, &(nkv as u32));
                    set_scalar(encoder, 6, &heads_per_group);
                    set_scalar(encoder, 7, &(max_seq_len as u32));
                    set_scalar(encoder, 8, &seq_len);
                    set_scalar(encoder, 9, &cache_layer_offset);
                }
                let tg_x = div_ceil(head_dim, 64) as u64;
                encoder.dispatch_thread_groups(
                    MTLSize::new(tg_x, nq as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            }
        }
        self.dispatch_gemv_tq2(
            encoder,
            &attn_proj_w.buffer,
            &bufs.attn_out_buf,
            &bufs.normed_buf,
            h,
            (nq * head_dim) as u32,
        );
        self.dispatch_residual_add(encoder, &bufs.hidden_buf, &bufs.normed_buf, h);
        self.dispatch_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            &ffn_norm_w.buffer,
            &bufs.normed_buf,
            eps,
            h,
        );
        self.dispatch_gemv_tq2(
            encoder,
            &gate_up_w.buffer,
            &bufs.normed_buf,
            &bufs.gate_up_buf,
            2 * inter,
            h,
        );
        self.dispatch_swiglu_single(encoder, &bufs.gate_up_buf, &bufs.swiglu_buf, inter);
        self.dispatch_gemv_tq2(
            encoder,
            &down_w.buffer,
            &bufs.swiglu_buf,
            &bufs.normed_buf,
            h,
            inter,
        );
        self.dispatch_residual_add(encoder, &bufs.hidden_buf, &bufs.normed_buf, h);
        Ok(())
    }
    /// Encode a complete transformer layer (attention + FFN) in one command buffer.
    ///
    /// All GPU dispatches share the same command buffer and encoder. Metal's
    /// automatic hazard tracking on shared-mode buffers ensures correct
    /// read-after-write dependencies between dispatches.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_full_layer(
        &self,
        hidden: &mut [f32],
        pos: usize,
        layer_idx: usize,
        attn_norm_w: &MetalWeightHandle,
        fused_qkv_w: &MetalWeightHandle,
        q_norm_w: &MetalWeightHandle,
        k_norm_w: &MetalWeightHandle,
        attn_proj_w: &MetalWeightHandle,
        ffn_norm_w: &MetalWeightHandle,
        gate_up_w: &MetalWeightHandle,
        down_w: &MetalWeightHandle,
        rope_cos: &[f32],
        rope_sin: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        n_layers: usize,
    ) -> Result<(), MetalGraphError> {
        let half_dim = head_dim / 2;
        if hidden.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden too short: need {hidden_size}, got {}",
                hidden.len()
            )));
        }
        if rope_cos.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_cos too short: need {half_dim}, got {}",
                rope_cos.len()
            )));
        }
        if rope_sin.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_sin too short: need {half_dim}, got {}",
                rope_sin.len()
            )));
        }
        let fl_guard = self.acquire_full_layer_buffers(
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = fl_guard.as_ref().ok_or_else(|| {
            MetalGraphError::ExecutionFailed("full_layer_buffers not allocated".into())
        })?;
        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden[..hidden_size]);
            upload_f32(&bufs.cos_buf, &rope_cos[..half_dim]);
            upload_f32(&bufs.sin_buf, &rope_sin[..half_dim]);
        }
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        self.encode_layer_into(
            encoder,
            bufs,
            kv,
            layer_idx,
            pos,
            attn_norm_w,
            fused_qkv_w,
            q_norm_w,
            k_norm_w,
            attn_proj_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            eps,
            max_seq_len,
        )?;
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe {
            download_f32(&bufs.hidden_buf, &mut hidden[..hidden_size]);
        }
        Ok(())
    }
    /// Ternary (TQ2_0_g128) twin of `Self::encode_full_layer`.
    ///
    /// Identical control flow, but defers all GEMV work to the ternary
    /// encoder `Self::encode_layer_into_ternary`. Intended for models whose
    /// attention/FFN projection weights are stored as TQ2_0_g128 blocks.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_full_layer_ternary(
        &self,
        hidden: &mut [f32],
        pos: usize,
        layer_idx: usize,
        attn_norm_w: &MetalWeightHandle,
        fused_qkv_w: &MetalWeightHandle,
        q_norm_w: &MetalWeightHandle,
        k_norm_w: &MetalWeightHandle,
        attn_proj_w: &MetalWeightHandle,
        ffn_norm_w: &MetalWeightHandle,
        gate_up_w: &MetalWeightHandle,
        down_w: &MetalWeightHandle,
        rope_cos: &[f32],
        rope_sin: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        n_layers: usize,
    ) -> Result<(), MetalGraphError> {
        let half_dim = head_dim / 2;
        if hidden.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden too short: need {hidden_size}, got {}",
                hidden.len()
            )));
        }
        if rope_cos.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_cos too short: need {half_dim}, got {}",
                rope_cos.len()
            )));
        }
        if rope_sin.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_sin too short: need {half_dim}, got {}",
                rope_sin.len()
            )));
        }
        let fl_guard = self.acquire_full_layer_buffers(
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = fl_guard.as_ref().ok_or_else(|| {
            MetalGraphError::ExecutionFailed("full_layer_buffers not allocated".into())
        })?;
        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden[..hidden_size]);
            upload_f32(&bufs.cos_buf, &rope_cos[..half_dim]);
            upload_f32(&bufs.sin_buf, &rope_sin[..half_dim]);
        }
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        self.encode_layer_into_ternary(
            encoder,
            bufs,
            kv,
            layer_idx,
            pos,
            attn_norm_w,
            fused_qkv_w,
            q_norm_w,
            k_norm_w,
            attn_proj_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            eps,
            max_seq_len,
        )?;
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe {
            download_f32(&bufs.hidden_buf, &mut hidden[..hidden_size]);
        }
        Ok(())
    }
    /// Encode ALL transformer layers into a SINGLE Metal command buffer.
    ///
    /// This eliminates N-1 command buffer submissions (one per layer), reducing
    /// GPU scheduling overhead. The hidden state persists in `hidden_buf` across
    /// layers — each layer reads and writes it in-place via residual_add dispatches.
    ///
    /// Metal's automatic hazard tracking on shared-mode buffers with a
    /// non-concurrent compute encoder guarantees correct sequential ordering.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn encode_full_forward(
        &self,
        hidden: &mut [f32],
        pos: usize,
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
        rope_cos: &[f32],
        rope_sin: &[f32],
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
        if hidden.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden too short: need {hidden_size}, got {}",
                hidden.len()
            )));
        }
        if rope_cos.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_cos too short: need {half_dim}, got {}",
                rope_cos.len()
            )));
        }
        if rope_sin.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_sin too short: need {half_dim}, got {}",
                rope_sin.len()
            )));
        }
        if layer_weights.len() != n_layers {
            return Err(MetalGraphError::EncodingFailed(format!(
                "layer_weights length mismatch: need {n_layers}, got {}",
                layer_weights.len()
            )));
        }
        let fl_guard = self.acquire_full_layer_buffers(
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = fl_guard.as_ref().ok_or_else(|| {
            MetalGraphError::ExecutionFailed("full_layer_buffers not allocated".into())
        })?;
        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden[..hidden_size]);
            upload_f32(&bufs.cos_buf, &rope_cos[..half_dim]);
            upload_f32(&bufs.sin_buf, &rope_sin[..half_dim]);
        }
        let profiling = std::env::var("OXIBONSAI_PROFILE").is_ok();
        let gpu_profiling = gpu_profile::is_enabled();
        if profiling {
            let mut layer_times = Vec::with_capacity(n_layers);
            for (layer_idx, weights) in layer_weights.iter().enumerate() {
                let layer_cmd = self.command_queue.new_command_buffer();
                let layer_enc = layer_cmd.new_compute_command_encoder();
                self.encode_layer_into(
                    layer_enc,
                    bufs,
                    kv,
                    layer_idx,
                    pos,
                    weights.0,
                    weights.1,
                    weights.2,
                    weights.3,
                    weights.4,
                    weights.5,
                    weights.6,
                    weights.7,
                    hidden_size,
                    intermediate_size,
                    nq,
                    nkv,
                    head_dim,
                    eps,
                    max_seq_len,
                )?;
                layer_enc.end_encoding();
                layer_cmd.commit();
                let t = std::time::Instant::now();
                layer_cmd.wait_until_completed();
                let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
                layer_times.push(elapsed_ms);
                eprintln!("[profile] layer {:2} = {:.3}ms", layer_idx, elapsed_ms);
            }
            let sum: f64 = layer_times.iter().sum();
            let min = layer_times.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = layer_times
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "[profile] layers total={:.3}ms  avg={:.3}ms  min={:.3}ms  max={:.3}ms",
                sum,
                sum / n_layers as f64,
                min,
                max,
            );
            let tail_cmd = self.command_queue.new_command_buffer();
            let tail_enc = tail_cmd.new_compute_command_encoder();
            self.encode_tail_and_commit(
                tail_enc,
                tail_cmd,
                bufs,
                hidden,
                hidden_size,
                final_norm_w,
                final_norm_eps,
                lm_head_w,
                lm_head_out_features,
                logits_out,
                greedy_token_id_out,
                true,
                None,
            )?;
        } else {
            let wall_start = if gpu_profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let cmd_buf = self.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            for (layer_idx, weights) in layer_weights.iter().enumerate() {
                self.encode_layer_into(
                    encoder,
                    bufs,
                    kv,
                    layer_idx,
                    pos,
                    weights.0,
                    weights.1,
                    weights.2,
                    weights.3,
                    weights.4,
                    weights.5,
                    weights.6,
                    weights.7,
                    hidden_size,
                    intermediate_size,
                    nq,
                    nkv,
                    head_dim,
                    eps,
                    max_seq_len,
                )?;
            }
            self.encode_tail_and_commit(
                encoder,
                cmd_buf,
                bufs,
                hidden,
                hidden_size,
                final_norm_w,
                final_norm_eps,
                lm_head_w,
                lm_head_out_features,
                logits_out,
                greedy_token_id_out,
                false,
                wall_start,
            )?;
        }
        Ok(())
    }
    /// Ternary (TQ2_0_g128) twin of `Self::encode_full_forward`.
    ///
    /// Transformer layers dispatch through `Self::encode_layer_into_ternary`,
    /// so every attention/FFN projection is a TQ2 GEMV. The trailing
    /// `final_norm + TQ2 LM head + (optional argmax)` block is forwarded to
    /// `Self::encode_tail_and_commit_ternary` which dispatches the LM head via
    /// `dispatch_gemv_tq2`. Pass `None` for both `final_norm_w` and `lm_head_w`
    /// to skip the tail (the caller then runs its own final-norm + LM-head
    /// path on CPU).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn encode_full_forward_ternary(
        &self,
        hidden: &mut [f32],
        pos: usize,
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
        rope_cos: &[f32],
        rope_sin: &[f32],
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
        if hidden.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden too short: need {hidden_size}, got {}",
                hidden.len()
            )));
        }
        if rope_cos.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_cos too short: need {half_dim}, got {}",
                rope_cos.len()
            )));
        }
        if rope_sin.len() < half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "rope_sin too short: need {half_dim}, got {}",
                rope_sin.len()
            )));
        }
        if layer_weights.len() != n_layers {
            return Err(MetalGraphError::EncodingFailed(format!(
                "layer_weights length mismatch: need {n_layers}, got {}",
                layer_weights.len()
            )));
        }
        let fl_guard = self.acquire_full_layer_buffers(
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = fl_guard.as_ref().ok_or_else(|| {
            MetalGraphError::ExecutionFailed("full_layer_buffers not allocated".into())
        })?;
        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden[..hidden_size]);
            upload_f32(&bufs.cos_buf, &rope_cos[..half_dim]);
            upload_f32(&bufs.sin_buf, &rope_sin[..half_dim]);
        }
        let profiling = std::env::var("OXIBONSAI_PROFILE").is_ok();
        let gpu_profiling = gpu_profile::is_enabled();
        if profiling {
            let mut layer_times = Vec::with_capacity(n_layers);
            for (layer_idx, weights) in layer_weights.iter().enumerate() {
                let layer_cmd = self.command_queue.new_command_buffer();
                let layer_enc = layer_cmd.new_compute_command_encoder();
                self.encode_layer_into_ternary(
                    layer_enc,
                    bufs,
                    kv,
                    layer_idx,
                    pos,
                    weights.0,
                    weights.1,
                    weights.2,
                    weights.3,
                    weights.4,
                    weights.5,
                    weights.6,
                    weights.7,
                    hidden_size,
                    intermediate_size,
                    nq,
                    nkv,
                    head_dim,
                    eps,
                    max_seq_len,
                )?;
                layer_enc.end_encoding();
                layer_cmd.commit();
                let t = std::time::Instant::now();
                layer_cmd.wait_until_completed();
                let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
                layer_times.push(elapsed_ms);
                eprintln!("[profile] layer {:2} = {:.3}ms", layer_idx, elapsed_ms);
            }
            let sum: f64 = layer_times.iter().sum();
            let min = layer_times.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = layer_times
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "[profile] layers total={:.3}ms  avg={:.3}ms  min={:.3}ms  max={:.3}ms",
                sum,
                sum / n_layers as f64,
                min,
                max,
            );
            let tail_cmd = self.command_queue.new_command_buffer();
            let tail_enc = tail_cmd.new_compute_command_encoder();
            self.encode_tail_and_commit_ternary(
                tail_enc,
                tail_cmd,
                bufs,
                hidden,
                hidden_size,
                final_norm_w,
                final_norm_eps,
                lm_head_w,
                lm_head_out_features,
                logits_out,
                greedy_token_id_out,
                true,
                None,
            )?;
        } else {
            let wall_start = if gpu_profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let cmd_buf = self.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            for (layer_idx, weights) in layer_weights.iter().enumerate() {
                self.encode_layer_into_ternary(
                    encoder,
                    bufs,
                    kv,
                    layer_idx,
                    pos,
                    weights.0,
                    weights.1,
                    weights.2,
                    weights.3,
                    weights.4,
                    weights.5,
                    weights.6,
                    weights.7,
                    hidden_size,
                    intermediate_size,
                    nq,
                    nkv,
                    head_dim,
                    eps,
                    max_seq_len,
                )?;
            }
            self.encode_tail_and_commit_ternary(
                encoder,
                cmd_buf,
                bufs,
                hidden,
                hidden_size,
                final_norm_w,
                final_norm_eps,
                lm_head_w,
                lm_head_out_features,
                logits_out,
                greedy_token_id_out,
                false,
                wall_start,
            )?;
        }
        Ok(())
    }
    /// Shared tail: final RMSNorm + LM head + argmax, then commit + wait + download.
    ///
    /// When `profiling` is true, prints timing for the tail section.
    /// When `gpu_profile_wall_start` is Some, captures GPU timing breakdown.
    #[allow(clippy::too_many_arguments)]
    fn encode_tail_and_commit(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        cmd_buf: &metal::CommandBufferRef,
        bufs: &FullLayerBuffers,
        hidden: &mut [f32],
        hidden_size: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        logits_out: Option<&mut Vec<f32>>,
        greedy_token_id_out: Option<&mut u32>,
        profiling: bool,
        gpu_profile_wall_start: Option<std::time::Instant>,
    ) -> Result<(), MetalGraphError> {
        match (final_norm_w, lm_head_w) {
            (Some(fnorm_w), Some(lm_w)) if lm_head_out_features > 0 => {
                let h = hidden_size as u32;
                self.dispatch_rmsnorm(
                    encoder,
                    &bufs.hidden_buf,
                    &fnorm_w.buffer,
                    &bufs.normed_buf,
                    final_norm_eps,
                    h,
                );
                let mut lg = self.logits_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into())
                })?;
                let needed_bytes = (lm_head_out_features * std::mem::size_of::<f32>()) as u64;
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
                    let encode_end = std::time::Instant::now();
                    encoder.end_encoding();
                    cmd_buf.commit();
                    let t = std::time::Instant::now();
                    cmd_buf.wait_until_completed();
                    if profiling {
                        eprintln!(
                            "[profile] tail (norm+lmhead+argmax) = {:.3}ms",
                            t.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    if let Some(ws) = gpu_profile_wall_start {
                        let (gs, ge) = unsafe { gpu_profile::gpu_cmd_times(cmd_buf) };
                        gpu_profile::record_and_print(ws, encode_end, gs, ge);
                    }
                    let token_id = unsafe { *(token_id_buf_ref.contents() as *const u32) };
                    if let Some(out) = greedy_token_id_out {
                        *out = token_id;
                    }
                } else {
                    let encode_end = std::time::Instant::now();
                    encoder.end_encoding();
                    cmd_buf.commit();
                    let t = std::time::Instant::now();
                    cmd_buf.wait_until_completed();
                    if profiling {
                        eprintln!(
                            "[profile] tail (norm+lmhead) = {:.3}ms",
                            t.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    if let Some(ws) = gpu_profile_wall_start {
                        let (gs, ge) = unsafe { gpu_profile::gpu_cmd_times(cmd_buf) };
                        gpu_profile::record_and_print(ws, encode_end, gs, ge);
                    }
                    if let Some(out) = logits_out {
                        out.resize(lm_head_out_features, 0.0);
                        unsafe { download_f32(logits_buf, out) };
                    }
                }
            }
            _ => {
                let encode_end = std::time::Instant::now();
                encoder.end_encoding();
                cmd_buf.commit();
                let t = std::time::Instant::now();
                cmd_buf.wait_until_completed();
                if profiling {
                    eprintln!(
                        "[profile] tail (no lmhead) = {:.3}ms",
                        t.elapsed().as_secs_f64() * 1000.0
                    );
                }
                if let Some(ws) = gpu_profile_wall_start {
                    let (gs, ge) = unsafe { gpu_profile::gpu_cmd_times(cmd_buf) };
                    gpu_profile::record_and_print(ws, encode_end, gs, ge);
                }
                unsafe {
                    download_f32(&bufs.hidden_buf, &mut hidden[..hidden_size]);
                }
            }
        }
        Ok(())
    }
    /// Ternary tail: final RMSNorm + TQ2 LM head + optional argmax, then commit + wait + download.
    ///
    /// Identical to `encode_tail_and_commit` in control flow and all steps
    /// except the LM-head GEMV, which is dispatched via `dispatch_gemv_tq2`
    /// instead of `dispatch_gemv_q1`. Use this when the LM head is stored as
    /// a TQ2_0_g128 quantised weight (ternary models).
    ///
    /// When `profiling` is true, prints timing for the tail section.
    /// When `gpu_profile_wall_start` is Some, captures GPU timing breakdown.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_tail_and_commit_ternary(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        cmd_buf: &metal::CommandBufferRef,
        bufs: &FullLayerBuffers,
        hidden: &mut [f32],
        hidden_size: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        logits_out: Option<&mut Vec<f32>>,
        greedy_token_id_out: Option<&mut u32>,
        profiling: bool,
        gpu_profile_wall_start: Option<std::time::Instant>,
    ) -> Result<(), MetalGraphError> {
        match (final_norm_w, lm_head_w) {
            (Some(fnorm_w), Some(lm_w)) if lm_head_out_features > 0 => {
                let h = hidden_size as u32;
                self.dispatch_rmsnorm(
                    encoder,
                    &bufs.hidden_buf,
                    &fnorm_w.buffer,
                    &bufs.normed_buf,
                    final_norm_eps,
                    h,
                );
                let mut lg = self.logits_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into())
                })?;
                let needed_bytes = (lm_head_out_features * std::mem::size_of::<f32>()) as u64;
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
                    let encode_end = std::time::Instant::now();
                    encoder.end_encoding();
                    cmd_buf.commit();
                    let t = std::time::Instant::now();
                    cmd_buf.wait_until_completed();
                    if profiling {
                        eprintln!(
                            "[profile] tail ternary (norm+lmhead+argmax) = {:.3}ms",
                            t.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    if let Some(ws) = gpu_profile_wall_start {
                        let (gs, ge) = unsafe { gpu_profile::gpu_cmd_times(cmd_buf) };
                        gpu_profile::record_and_print(ws, encode_end, gs, ge);
                    }
                    let token_id = unsafe { *(token_id_buf_ref.contents() as *const u32) };
                    if let Some(out) = greedy_token_id_out {
                        *out = token_id;
                    }
                } else {
                    let encode_end = std::time::Instant::now();
                    encoder.end_encoding();
                    cmd_buf.commit();
                    let t = std::time::Instant::now();
                    cmd_buf.wait_until_completed();
                    if profiling {
                        eprintln!(
                            "[profile] tail ternary (norm+lmhead) = {:.3}ms",
                            t.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    if let Some(ws) = gpu_profile_wall_start {
                        let (gs, ge) = unsafe { gpu_profile::gpu_cmd_times(cmd_buf) };
                        gpu_profile::record_and_print(ws, encode_end, gs, ge);
                    }
                    if let Some(out) = logits_out {
                        out.resize(lm_head_out_features, 0.0);
                        unsafe { download_f32(logits_buf, out) };
                    }
                }
            }
            _ => {
                let encode_end = std::time::Instant::now();
                encoder.end_encoding();
                cmd_buf.commit();
                let t = std::time::Instant::now();
                cmd_buf.wait_until_completed();
                if profiling {
                    eprintln!(
                        "[profile] tail ternary (no lmhead) = {:.3}ms",
                        t.elapsed().as_secs_f64() * 1000.0
                    );
                }
                if let Some(ws) = gpu_profile_wall_start {
                    let (gs, ge) = unsafe { gpu_profile::gpu_cmd_times(cmd_buf) };
                    gpu_profile::record_and_print(ws, encode_end, gs, ge);
                }
                unsafe {
                    download_f32(&bufs.hidden_buf, &mut hidden[..hidden_size]);
                }
            }
        }
        Ok(())
    }
}
