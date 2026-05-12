//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{ModelError, ModelResult};
use crate::kv_cache::KvCache;
use crate::layers::attention_fused::fused_attention_head_contiguous;
use crate::layers::linear::LinearLayer;
use crate::layers::rms_norm::RmsNorm;
use crate::layers::rope::RopeTable;
use crate::layers::sliding_window::SlidingWindowConfig;
use crate::layers::swiglu::swiglu as swiglu_fn;
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::GpuWeightHandle;
use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;

#[cfg(any(
    feature = "metal",
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
use super::functions::blocks_as_bytes;
use super::functions::{compute_gqa_attention, PAR_HEAD_MIN_HEADS};

/// Statistics collected during a single layer's forward pass.
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Layer index.
    pub layer_idx: usize,
    /// Time spent on attention norm + Q/K/V projection.
    pub projection_us: u64,
    /// Time spent on RoPE application.
    pub rope_us: u64,
    /// Time spent on attention computation (GQA).
    pub attention_us: u64,
    /// Time spent on FFN (MLP) sublayer.
    pub ffn_us: u64,
    /// Total forward time for this layer.
    pub total_us: u64,
}
impl LayerStats {
    /// Create empty stats for a given layer.
    pub(crate) fn new(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            projection_us: 0,
            rope_us: 0,
            attention_us: 0,
            ffn_us: 0,
            total_us: 0,
        }
    }
    /// Fraction of time spent in attention (vs total).
    pub fn attention_fraction(&self) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        self.attention_us as f64 / self.total_us as f64
    }
    /// Fraction of time spent in FFN (vs total).
    pub fn ffn_fraction(&self) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        self.ffn_us as f64 / self.total_us as f64
    }
}
/// A single Qwen3 Transformer block.
///
/// Holds references to weight data (zero-copy from GGUF mmap).
pub struct TransformerBlock<'a> {
    /// Layer index (0-based).
    layer_idx: usize,
    /// Pre-attention RMSNorm.
    attn_norm: RmsNorm,
    /// Q projection: [hidden_size → num_heads * head_dim].
    attn_q: LinearLayer<'a>,
    /// K projection: [hidden_size → num_kv_heads * head_dim].
    attn_k: LinearLayer<'a>,
    /// V projection: [hidden_size → num_kv_heads * head_dim].
    attn_v: LinearLayer<'a>,
    /// Output projection: [num_heads * head_dim → hidden_size].
    attn_output: LinearLayer<'a>,
    /// Per-head QK-norm on Q vectors (shape=[head_dim], shared across all Q heads).
    attn_q_norm: RmsNorm,
    /// Per-head QK-norm on K vectors (shape=[head_dim], shared across all KV heads).
    attn_k_norm: RmsNorm,
    /// Pre-FFN RMSNorm.
    ffn_norm: RmsNorm,
    /// Gate projection: [hidden_size → intermediate_size].
    ffn_gate: LinearLayer<'a>,
    /// Up projection: [hidden_size → intermediate_size].
    ffn_up: LinearLayer<'a>,
    /// Down projection: [intermediate_size → hidden_size].
    ffn_down: LinearLayer<'a>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    /// Fused Q+K+V weight handle (single GPU dispatch).
    fused_qkv_handle: Option<GpuWeightHandle>,
    /// Fused gate+up weight handle (single GPU dispatch).
    fused_gate_up_handle: Option<GpuWeightHandle>,
    /// Pre-allocated scratch buffers (Mutex for Sync safety; uncontended in practice).
    scratch: Mutex<ScratchBuffers>,
}
impl<'a> TransformerBlock<'a> {
    /// Create a new Transformer block from loaded weights.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layer_idx: usize,
        attn_norm: RmsNorm,
        attn_q: LinearLayer<'a>,
        attn_k: LinearLayer<'a>,
        attn_v: LinearLayer<'a>,
        attn_output: LinearLayer<'a>,
        attn_q_norm: RmsNorm,
        attn_k_norm: RmsNorm,
        ffn_norm: RmsNorm,
        ffn_gate: LinearLayer<'a>,
        ffn_up: LinearLayer<'a>,
        ffn_down: LinearLayer<'a>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_size: usize,
    ) -> Self {
        let inter = ffn_gate.out_features();
        let scratch = Mutex::new(ScratchBuffers::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            inter,
        ));
        Self {
            layer_idx,
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_q_norm,
            attn_k_norm,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            fused_qkv_handle: None,
            fused_gate_up_handle: None,
            scratch,
        }
    }
    /// Upload all weight matrices in this block to GPU memory.
    ///
    /// After calling this, all GEMV operations in [`forward`](Self::forward)
    /// will use GPU-resident weight buffers, eliminating per-call
    /// host→device copies.
    pub fn upload_to_gpu(&mut self, kernel: &dyn OneBitKernel) {
        self.attn_q.upload_to_gpu();
        self.attn_k.upload_to_gpu();
        self.attn_v.upload_to_gpu();
        self.attn_output.upload_to_gpu();
        self.ffn_gate.upload_to_gpu();
        self.ffn_up.upload_to_gpu();
        self.ffn_down.upload_to_gpu();
        if let (Some(q_blk), Some(k_blk), Some(v_blk)) = (
            self.attn_q.blocks_1bit(),
            self.attn_k.blocks_1bit(),
            self.attn_v.blocks_1bit(),
        ) {
            let mut qkv_blocks = Vec::with_capacity(q_blk.len() + k_blk.len() + v_blk.len());
            qkv_blocks.extend_from_slice(q_blk);
            qkv_blocks.extend_from_slice(k_blk);
            qkv_blocks.extend_from_slice(v_blk);
            self.fused_qkv_handle = kernel.upload_weights(&qkv_blocks);
        }
        if let (Some(gate_blk), Some(up_blk)) =
            (self.ffn_gate.blocks_1bit(), self.ffn_up.blocks_1bit())
        {
            let mut gate_up_blocks = Vec::with_capacity(gate_blk.len() + up_blk.len());
            gate_up_blocks.extend_from_slice(gate_blk);
            gate_up_blocks.extend_from_slice(up_blk);
            self.fused_gate_up_handle = kernel.upload_weights(&gate_up_blocks);
        }
    }
    /// Forward pass for a single token at position `pos`.
    ///
    /// - `hidden`: Input/output hidden state `[hidden_size]`. Modified in-place.
    /// - `pos`: Current token position in the sequence.
    /// - `kv_cache`: KV cache to store/retrieve K and V vectors.
    /// - `rope`: Precomputed RoPE table.
    /// - `kernel`: 1-bit kernel dispatcher.
    #[allow(clippy::needless_late_init)]
    #[tracing::instrument(skip_all, fields(layer = self.layer_idx))]
    pub fn forward(
        &self,
        hidden: &mut [f32],
        pos: usize,
        kv_cache: &mut KvCache,
        rope: &RopeTable,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<()> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if let Some(Ok(())) = self.try_full_layer_gpu(hidden, pos, rope, kv_cache) {
                return Ok(());
            }
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            if let Some(Ok(())) = self.try_full_layer_cuda(hidden, pos, rope, kv_cache) {
                return Ok(());
            }
        }
        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nq / nkv;
        let total_start = Instant::now();
        let mut scratch = self.scratch.lock().map_err(|e| {
            crate::error::ModelError::Internal(format!("scratch lock poisoned: {e}"))
        })?;
        scratch.clear();
        let ScratchBuffers {
            normed,
            q_all,
            k_all,
            v_all,
            q_normed,
            k_normed,
            q_rope,
            k_rope,
            attn_out,
            attn_proj,
            gate_out,
            up_out,
            swiglu_out,
            down_out,
            fused_qkv,
            fused_gate_up,
        } = &mut *scratch;
        let norm_us: u128;
        let qkv_us: u128;
        let qknorm_us: u128;
        let rope_us: u128;
        let cache_us: u128;
        let attn_us: u128;
        let ffn_us: u128;
        {
            let norm_start = Instant::now();
            self.attn_norm.forward(hidden, normed)?;
            norm_us = norm_start.elapsed().as_micros();
            let qkv_start = Instant::now();
            if let Some(fused_handle) = self.fused_qkv_handle {
                let q_rows = nq * hd;
                let k_rows = nkv * hd;
                let total_rows = q_rows + k_rows + k_rows;
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let metal_ok = {
                    if let (Some(q_blk), Some(k_blk), Some(v_blk)) = (
                        self.attn_q.blocks_1bit(),
                        self.attn_k.blocks_1bit(),
                        self.attn_v.blocks_1bit(),
                    ) {
                        let q_bytes = blocks_as_bytes(q_blk);
                        let k_bytes = blocks_as_bytes(k_blk);
                        let v_bytes = blocks_as_bytes(v_blk);
                        oxibonsai_kernels::try_metal_qkv(
                            normed,
                            fused_qkv,
                            fused_handle.id(),
                            q_bytes,
                            k_bytes,
                            v_bytes,
                            total_rows,
                            h,
                        )
                        .is_ok()
                    } else {
                        false
                    }
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let metal_ok = false;
                #[cfg(all(
                    feature = "native-cuda",
                    not(all(feature = "metal", target_os = "macos")),
                    any(target_os = "linux", target_os = "windows")
                ))]
                let cuda_ok = if !metal_ok {
                    if let (Some(q_blk), Some(k_blk), Some(v_blk)) = (
                        self.attn_q.blocks_1bit(),
                        self.attn_k.blocks_1bit(),
                        self.attn_v.blocks_1bit(),
                    ) {
                        let q_bytes = blocks_as_bytes(q_blk);
                        let k_bytes = blocks_as_bytes(k_blk);
                        let v_bytes = blocks_as_bytes(v_blk);
                        oxibonsai_kernels::try_cuda_qkv(
                            normed,
                            fused_qkv,
                            fused_handle.id(),
                            q_bytes,
                            k_bytes,
                            v_bytes,
                            total_rows,
                            h,
                        )
                        .is_ok()
                    } else {
                        false
                    }
                } else {
                    false
                };
                #[cfg(not(all(
                    feature = "native-cuda",
                    not(all(feature = "metal", target_os = "macos")),
                    any(target_os = "linux", target_os = "windows")
                )))]
                let cuda_ok = false;
                if !metal_ok && !cuda_ok {
                    kernel.gemv_cached(fused_handle, normed, fused_qkv, total_rows, h)?;
                }
                q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                k_all[..k_rows].copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                v_all[..k_rows].copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
            } else {
                self.attn_q.forward_vec(normed, q_all)?;
                self.attn_k.forward_vec(normed, k_all)?;
                self.attn_v.forward_vec(normed, v_all)?;
            }
            qkv_us = qkv_start.elapsed().as_micros();
        }
        let qknorm_start = Instant::now();
        for head in 0..nq {
            let start = head * hd;
            self.attn_q_norm
                .forward(&q_all[start..start + hd], &mut q_normed[start..start + hd])?;
        }
        for head in 0..nkv {
            let start = head * hd;
            self.attn_k_norm
                .forward(&k_all[start..start + hd], &mut k_normed[start..start + hd])?;
        }
        qknorm_us = qknorm_start.elapsed().as_micros();
        let rope_start = Instant::now();
        for head in 0..nq {
            let start = head * hd;
            rope.apply(
                &q_normed[start..start + hd],
                &mut q_rope[start..start + hd],
                pos,
            )?;
        }
        for head in 0..nkv {
            let start = head * hd;
            rope.apply(
                &k_normed[start..start + hd],
                &mut k_rope[start..start + hd],
                pos,
            )?;
        }
        rope_us = rope_start.elapsed().as_micros();
        let cache_start = Instant::now();
        for head in 0..nkv {
            let start = head * hd;
            kv_cache.store_key(self.layer_idx, head, pos, &k_rope[start..start + hd]);
            kv_cache.store_value(self.layer_idx, head, pos, &v_all[start..start + hd]);
        }
        cache_us = cache_start.elapsed().as_micros();
        let seq_len = pos + 1;
        let attn_start = Instant::now();
        compute_gqa_attention(
            q_rope,
            attn_out,
            kv_cache,
            self.layer_idx,
            nq,
            heads_per_group,
            hd,
            seq_len,
        )?;
        attn_us = attn_start.elapsed().as_micros();
        let ffn_start = Instant::now();
        let did_batch_ffn =
            if let (Some(attn_proj_handle), Some(gate_up_handle), Some(down_handle)) = (
                self.attn_output.gpu_handle(),
                self.fused_gate_up_handle,
                self.ffn_down.gpu_handle(),
            ) {
                let inter = self.ffn_gate.out_features();
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let (Some(attn_proj_blk), Some(gate_blk), Some(up_blk), Some(down_blk)) = (
                        self.attn_output.blocks_1bit(),
                        self.ffn_gate.blocks_1bit(),
                        self.ffn_up.blocks_1bit(),
                        self.ffn_down.blocks_1bit(),
                    ) {
                        let attn_proj_bytes = blocks_as_bytes(attn_proj_blk);
                        let gate_bytes = blocks_as_bytes(gate_blk);
                        let up_bytes = blocks_as_bytes(up_blk);
                        let down_bytes = blocks_as_bytes(down_blk);
                        let metal_result = oxibonsai_kernels::try_metal_ffn(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle.id(),
                            attn_proj_bytes,
                            gate_up_handle.id(),
                            gate_bytes,
                            up_bytes,
                            down_handle.id(),
                            down_bytes,
                            h,
                            inter,
                        );
                        if metal_result.is_ok() {
                            true
                        } else {
                            tracing::warn!(
                                error = ? metal_result.err(),
                                "MetalGraph FFN failed, falling back"
                            );
                            kernel.batch_ffn_phase(
                                hidden,
                                attn_out,
                                self.ffn_norm.weight(),
                                self.ffn_norm.eps(),
                                attn_proj_handle,
                                gate_up_handle,
                                down_handle,
                                h,
                                inter,
                                nq * hd,
                            )?
                        }
                    } else {
                        kernel.batch_ffn_phase(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle,
                            gate_up_handle,
                            down_handle,
                            h,
                            inter,
                            nq * hd,
                        )?
                    }
                }
                #[cfg(all(
                    feature = "native-cuda",
                    not(all(feature = "metal", target_os = "macos")),
                    any(target_os = "linux", target_os = "windows")
                ))]
                {
                    if let (Some(attn_proj_blk), Some(gate_blk), Some(up_blk), Some(down_blk)) = (
                        self.attn_output.blocks_1bit(),
                        self.ffn_gate.blocks_1bit(),
                        self.ffn_up.blocks_1bit(),
                        self.ffn_down.blocks_1bit(),
                    ) {
                        let attn_proj_bytes = blocks_as_bytes(attn_proj_blk);
                        let gate_bytes = blocks_as_bytes(gate_blk);
                        let up_bytes = blocks_as_bytes(up_blk);
                        let down_bytes = blocks_as_bytes(down_blk);
                        let cuda_result = oxibonsai_kernels::try_cuda_ffn(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle.id(),
                            attn_proj_bytes,
                            gate_up_handle.id(),
                            gate_bytes,
                            up_bytes,
                            down_handle.id(),
                            down_bytes,
                            h,
                            inter,
                        );
                        if cuda_result.is_ok() {
                            true
                        } else {
                            tracing::warn!(
                                error = ? cuda_result.err(),
                                "CudaGraph FFN failed, falling back"
                            );
                            kernel.batch_ffn_phase(
                                hidden,
                                attn_out,
                                self.ffn_norm.weight(),
                                self.ffn_norm.eps(),
                                attn_proj_handle,
                                gate_up_handle,
                                down_handle,
                                h,
                                inter,
                                nq * hd,
                            )?
                        }
                    } else {
                        kernel.batch_ffn_phase(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle,
                            gate_up_handle,
                            down_handle,
                            h,
                            inter,
                            nq * hd,
                        )?
                    }
                }
                #[cfg(not(any(
                    all(feature = "metal", target_os = "macos"),
                    all(
                        feature = "native-cuda",
                        any(target_os = "linux", target_os = "windows")
                    )
                )))]
                {
                    kernel.batch_ffn_phase(
                        hidden,
                        attn_out,
                        self.ffn_norm.weight(),
                        self.ffn_norm.eps(),
                        attn_proj_handle,
                        gate_up_handle,
                        down_handle,
                        h,
                        inter,
                        nq * hd,
                    )?
                }
            } else {
                false
            };
        if !did_batch_ffn {
            self.attn_output.forward_vec(attn_out, attn_proj)?;
            for i in 0..h {
                hidden[i] += attn_proj[i];
            }
            self.ffn_norm.forward(hidden, normed)?;
            if let Some(fused_handle) = self.fused_gate_up_handle {
                let inter = gate_out.len();
                let total_rows = inter * 2;
                kernel.gemv_cached(fused_handle, normed, fused_gate_up, total_rows, h)?;
                gate_out[..inter].copy_from_slice(&fused_gate_up[..inter]);
                up_out[..inter].copy_from_slice(&fused_gate_up[inter..total_rows]);
            } else {
                self.ffn_gate.forward_vec(normed, gate_out)?;
                self.ffn_up.forward_vec(normed, up_out)?;
            }
            swiglu_fn(gate_out, up_out, swiglu_out);
            self.ffn_down.forward_vec(swiglu_out, down_out)?;
            for i in 0..h {
                hidden[i] += down_out[i];
            }
        }
        ffn_us = ffn_start.elapsed().as_micros();
        let total_us = total_start.elapsed().as_micros();
        tracing::debug!(
            target : "block_profile",
            "L{layer}: norm={norm_us}µs qkv={qkv_us}µs qknorm={qknorm_us}µs rope={rope_us}µs cache={cache_us}µs attn={attn_us}µs ffn={ffn_us}µs total={total_us}µs",
            layer = self.layer_idx,
        );
        Ok(())
    }
    /// Forward pass with timing statistics collection.
    ///
    /// Same computation as `forward`, but records per-phase timing.
    #[tracing::instrument(skip_all, fields(layer = self.layer_idx))]
    pub fn forward_with_stats(
        &self,
        hidden: &mut [f32],
        pos: usize,
        kv_cache: &mut KvCache,
        rope: &RopeTable,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<LayerStats> {
        let total_start = Instant::now();
        let mut stats = LayerStats::new(self.layer_idx);
        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nq / nkv;
        let mut scratch = self.scratch.lock().map_err(|e| {
            crate::error::ModelError::Internal(format!("scratch lock poisoned: {e}"))
        })?;
        scratch.clear();
        let ScratchBuffers {
            normed,
            q_all,
            k_all,
            v_all,
            q_normed,
            k_normed,
            q_rope,
            k_rope,
            attn_out,
            attn_proj,
            gate_out,
            up_out,
            swiglu_out,
            down_out,
            fused_qkv,
            fused_gate_up,
        } = &mut *scratch;
        let proj_start = Instant::now();
        let batch_qkv = if let Some(fused_handle) = self.fused_qkv_handle {
            kernel.batch_attn_phase(
                hidden,
                self.attn_norm.weight(),
                self.attn_norm.eps(),
                fused_handle,
                nq * hd,
                nkv * hd,
                h,
            )?
        } else {
            None
        };
        if let Some((q_data, k_data, v_data)) = batch_qkv {
            q_all[..nq * hd].copy_from_slice(&q_data);
            k_all[..nkv * hd].copy_from_slice(&k_data);
            v_all[..nkv * hd].copy_from_slice(&v_data);
        } else {
            self.attn_norm.forward(hidden, normed)?;
            if let Some(fused_handle) = self.fused_qkv_handle {
                let q_rows = nq * hd;
                let k_rows = nkv * hd;
                let total_rows = q_rows + k_rows + k_rows;
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let metal_ok = {
                    if let (Some(q_blk), Some(k_blk), Some(v_blk)) = (
                        self.attn_q.blocks_1bit(),
                        self.attn_k.blocks_1bit(),
                        self.attn_v.blocks_1bit(),
                    ) {
                        let q_bytes = blocks_as_bytes(q_blk);
                        let k_bytes = blocks_as_bytes(k_blk);
                        let v_bytes = blocks_as_bytes(v_blk);
                        oxibonsai_kernels::try_metal_qkv(
                            normed,
                            fused_qkv,
                            fused_handle.id(),
                            q_bytes,
                            k_bytes,
                            v_bytes,
                            total_rows,
                            h,
                        )
                        .is_ok()
                    } else {
                        false
                    }
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let metal_ok = false;
                if !metal_ok {
                    kernel.gemv_cached(fused_handle, normed, fused_qkv, total_rows, h)?;
                }
                q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                k_all[..k_rows].copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                v_all[..k_rows].copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
            } else {
                self.attn_q.forward_vec(normed, q_all)?;
                self.attn_k.forward_vec(normed, k_all)?;
                self.attn_v.forward_vec(normed, v_all)?;
            }
        }
        for head in 0..nq {
            let start = head * hd;
            self.attn_q_norm
                .forward(&q_all[start..start + hd], &mut q_normed[start..start + hd])?;
        }
        for head in 0..nkv {
            let start = head * hd;
            self.attn_k_norm
                .forward(&k_all[start..start + hd], &mut k_normed[start..start + hd])?;
        }
        stats.projection_us = proj_start.elapsed().as_micros() as u64;
        let rope_start = Instant::now();
        for head in 0..nq {
            let start = head * hd;
            rope.apply(
                &q_normed[start..start + hd],
                &mut q_rope[start..start + hd],
                pos,
            )?;
        }
        for head in 0..nkv {
            let start = head * hd;
            rope.apply(
                &k_normed[start..start + hd],
                &mut k_rope[start..start + hd],
                pos,
            )?;
        }
        stats.rope_us = rope_start.elapsed().as_micros() as u64;
        let attn_start = Instant::now();
        for head in 0..nkv {
            let start = head * hd;
            kv_cache.store_key(self.layer_idx, head, pos, &k_rope[start..start + hd]);
            kv_cache.store_value(self.layer_idx, head, pos, &v_all[start..start + hd]);
        }
        let seq_len = pos + 1;
        compute_gqa_attention(
            q_rope,
            attn_out,
            kv_cache,
            self.layer_idx,
            nq,
            heads_per_group,
            hd,
            seq_len,
        )?;
        let did_batch_ffn =
            if let (Some(attn_proj_handle), Some(gate_up_handle), Some(down_handle)) = (
                self.attn_output.gpu_handle(),
                self.fused_gate_up_handle,
                self.ffn_down.gpu_handle(),
            ) {
                let inter = self.ffn_gate.out_features();
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let (Some(attn_proj_blk), Some(gate_blk), Some(up_blk), Some(down_blk)) = (
                        self.attn_output.blocks_1bit(),
                        self.ffn_gate.blocks_1bit(),
                        self.ffn_up.blocks_1bit(),
                        self.ffn_down.blocks_1bit(),
                    ) {
                        let attn_proj_bytes = blocks_as_bytes(attn_proj_blk);
                        let gate_bytes = blocks_as_bytes(gate_blk);
                        let up_bytes = blocks_as_bytes(up_blk);
                        let down_bytes = blocks_as_bytes(down_blk);
                        let metal_result = oxibonsai_kernels::try_metal_ffn(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle.id(),
                            attn_proj_bytes,
                            gate_up_handle.id(),
                            gate_bytes,
                            up_bytes,
                            down_handle.id(),
                            down_bytes,
                            h,
                            inter,
                        );
                        if metal_result.is_ok() {
                            true
                        } else {
                            kernel.batch_ffn_phase(
                                hidden,
                                attn_out,
                                self.ffn_norm.weight(),
                                self.ffn_norm.eps(),
                                attn_proj_handle,
                                gate_up_handle,
                                down_handle,
                                h,
                                inter,
                                nq * hd,
                            )?
                        }
                    } else {
                        kernel.batch_ffn_phase(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle,
                            gate_up_handle,
                            down_handle,
                            h,
                            inter,
                            nq * hd,
                        )?
                    }
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    kernel.batch_ffn_phase(
                        hidden,
                        attn_out,
                        self.ffn_norm.weight(),
                        self.ffn_norm.eps(),
                        attn_proj_handle,
                        gate_up_handle,
                        down_handle,
                        h,
                        inter,
                        nq * hd,
                    )?
                }
            } else {
                false
            };
        if !did_batch_ffn {
            self.attn_output.forward_vec(attn_out, attn_proj)?;
            for i in 0..h {
                hidden[i] += attn_proj[i];
            }
        }
        stats.attention_us = attn_start.elapsed().as_micros() as u64;
        let ffn_start = Instant::now();
        if !did_batch_ffn {
            self.ffn_norm.forward(hidden, normed)?;
            if let Some(fused_handle) = self.fused_gate_up_handle {
                let inter = gate_out.len();
                let total_rows = inter * 2;
                kernel.gemv_cached(fused_handle, normed, fused_gate_up, total_rows, h)?;
                gate_out[..inter].copy_from_slice(&fused_gate_up[..inter]);
                up_out[..inter].copy_from_slice(&fused_gate_up[inter..total_rows]);
            } else {
                self.ffn_gate.forward_vec(normed, gate_out)?;
                self.ffn_up.forward_vec(normed, up_out)?;
            }
            swiglu_fn(gate_out, up_out, swiglu_out);
            self.ffn_down.forward_vec(swiglu_out, down_out)?;
            for i in 0..h {
                hidden[i] += down_out[i];
            }
        }
        stats.ffn_us = ffn_start.elapsed().as_micros() as u64;
        stats.total_us = total_start.elapsed().as_micros() as u64;
        Ok(stats)
    }
    /// Forward pass with optional sliding window attention.
    ///
    /// When `sliding_window` is `Some`, attention is restricted to positions
    /// within the window, reducing compute for long sequences.
    #[tracing::instrument(skip_all, fields(layer = self.layer_idx))]
    pub fn forward_with_sliding_window(
        &self,
        hidden: &mut [f32],
        pos: usize,
        kv_cache: &mut KvCache,
        rope: &RopeTable,
        kernel: &dyn OneBitKernel,
        sliding_window: Option<&SlidingWindowConfig>,
    ) -> ModelResult<()> {
        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nq / nkv;
        let mut scratch = self.scratch.lock().map_err(|e| {
            crate::error::ModelError::Internal(format!("scratch lock poisoned: {e}"))
        })?;
        scratch.clear();
        let ScratchBuffers {
            normed,
            q_all,
            k_all,
            v_all,
            q_normed,
            k_normed,
            q_rope,
            k_rope,
            attn_out,
            attn_proj,
            gate_out,
            up_out,
            swiglu_out,
            down_out,
            fused_qkv,
            fused_gate_up,
        } = &mut *scratch;
        let batch_qkv = if let Some(fused_handle) = self.fused_qkv_handle {
            kernel.batch_attn_phase(
                hidden,
                self.attn_norm.weight(),
                self.attn_norm.eps(),
                fused_handle,
                nq * hd,
                nkv * hd,
                h,
            )?
        } else {
            None
        };
        if let Some((q_data, k_data, v_data)) = batch_qkv {
            q_all[..nq * hd].copy_from_slice(&q_data);
            k_all[..nkv * hd].copy_from_slice(&k_data);
            v_all[..nkv * hd].copy_from_slice(&v_data);
        } else {
            self.attn_norm.forward(hidden, normed)?;
            if let Some(fused_handle) = self.fused_qkv_handle {
                let q_rows = nq * hd;
                let k_rows = nkv * hd;
                let total_rows = q_rows + k_rows + k_rows;
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let metal_ok = {
                    if let (Some(q_blk), Some(k_blk), Some(v_blk)) = (
                        self.attn_q.blocks_1bit(),
                        self.attn_k.blocks_1bit(),
                        self.attn_v.blocks_1bit(),
                    ) {
                        let q_bytes = blocks_as_bytes(q_blk);
                        let k_bytes = blocks_as_bytes(k_blk);
                        let v_bytes = blocks_as_bytes(v_blk);
                        oxibonsai_kernels::try_metal_qkv(
                            normed,
                            fused_qkv,
                            fused_handle.id(),
                            q_bytes,
                            k_bytes,
                            v_bytes,
                            total_rows,
                            h,
                        )
                        .is_ok()
                    } else {
                        false
                    }
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let metal_ok = false;
                if !metal_ok {
                    kernel.gemv_cached(fused_handle, normed, fused_qkv, total_rows, h)?;
                }
                q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                k_all[..k_rows].copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                v_all[..k_rows].copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
            } else {
                self.attn_q.forward_vec(normed, q_all)?;
                self.attn_k.forward_vec(normed, k_all)?;
                self.attn_v.forward_vec(normed, v_all)?;
            }
        }
        for head in 0..nq {
            let start = head * hd;
            self.attn_q_norm
                .forward(&q_all[start..start + hd], &mut q_normed[start..start + hd])?;
        }
        for head in 0..nkv {
            let start = head * hd;
            self.attn_k_norm
                .forward(&k_all[start..start + hd], &mut k_normed[start..start + hd])?;
        }
        for head in 0..nq {
            let start = head * hd;
            rope.apply(
                &q_normed[start..start + hd],
                &mut q_rope[start..start + hd],
                pos,
            )?;
        }
        for head in 0..nkv {
            let start = head * hd;
            rope.apply(
                &k_normed[start..start + hd],
                &mut k_rope[start..start + hd],
                pos,
            )?;
        }
        for head in 0..nkv {
            let start = head * hd;
            kv_cache.store_key(self.layer_idx, head, pos, &k_rope[start..start + hd]);
            kv_cache.store_value(self.layer_idx, head, pos, &v_all[start..start + hd]);
        }
        let full_seq_len = pos + 1;
        if let Some(sw_config) = sliding_window {
            let (positions, _count) =
                crate::layers::sliding_window::attention_range(pos, full_seq_len, sw_config);
            let num_kv_heads = nq / heads_per_group;
            let windowed_len = positions.len();
            let windowed_kv: Vec<(Vec<f32>, Vec<f32>)> = (0..num_kv_heads)
                .map(|kv_h| {
                    let all_keys = kv_cache.keys_for(self.layer_idx, kv_h, full_seq_len);
                    let all_values = kv_cache.values_for(self.layer_idx, kv_h, full_seq_len);
                    let wk: Vec<f32> = positions
                        .iter()
                        .flat_map(|&p| all_keys[p * hd..(p + 1) * hd].iter().copied())
                        .collect();
                    let wv: Vec<f32> = positions
                        .iter()
                        .flat_map(|&p| all_values[p * hd..(p + 1) * hd].iter().copied())
                        .collect();
                    (wk, wv)
                })
                .collect();
            if nq >= PAR_HEAD_MIN_HEADS {
                attn_out.par_chunks_mut(hd).enumerate().try_for_each(
                    |(q_head, out_slice)| -> ModelResult<()> {
                        let kv_head = q_head / heads_per_group;
                        let q_start = q_head * hd;
                        let (wk, wv) = &windowed_kv[kv_head];
                        fused_attention_head_contiguous(
                            &q_rope[q_start..q_start + hd],
                            wk,
                            wv,
                            out_slice,
                            windowed_len,
                            hd,
                        )
                        .map_err(|e| {
                            ModelError::Internal(format!(
                                "parallel sliding-window head {q_head}: {e}"
                            ))
                        })
                    },
                )?;
            } else {
                for q_head in 0..nq {
                    let kv_head = q_head / heads_per_group;
                    let q_start = q_head * hd;
                    let (wk, wv) = &windowed_kv[kv_head];
                    fused_attention_head_contiguous(
                        &q_rope[q_start..q_start + hd],
                        wk,
                        wv,
                        &mut attn_out[q_start..q_start + hd],
                        windowed_len,
                        hd,
                    )?;
                }
            }
        } else {
            compute_gqa_attention(
                q_rope,
                attn_out,
                kv_cache,
                self.layer_idx,
                nq,
                heads_per_group,
                hd,
                full_seq_len,
            )?;
        }
        let did_batch_ffn =
            if let (Some(attn_proj_handle), Some(gate_up_handle), Some(down_handle)) = (
                self.attn_output.gpu_handle(),
                self.fused_gate_up_handle,
                self.ffn_down.gpu_handle(),
            ) {
                let inter = self.ffn_gate.out_features();
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let (Some(attn_proj_blk), Some(gate_blk), Some(up_blk), Some(down_blk)) = (
                        self.attn_output.blocks_1bit(),
                        self.ffn_gate.blocks_1bit(),
                        self.ffn_up.blocks_1bit(),
                        self.ffn_down.blocks_1bit(),
                    ) {
                        let attn_proj_bytes = blocks_as_bytes(attn_proj_blk);
                        let gate_bytes = blocks_as_bytes(gate_blk);
                        let up_bytes = blocks_as_bytes(up_blk);
                        let down_bytes = blocks_as_bytes(down_blk);
                        let metal_result = oxibonsai_kernels::try_metal_ffn(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle.id(),
                            attn_proj_bytes,
                            gate_up_handle.id(),
                            gate_bytes,
                            up_bytes,
                            down_handle.id(),
                            down_bytes,
                            h,
                            inter,
                        );
                        if metal_result.is_ok() {
                            true
                        } else {
                            kernel.batch_ffn_phase(
                                hidden,
                                attn_out,
                                self.ffn_norm.weight(),
                                self.ffn_norm.eps(),
                                attn_proj_handle,
                                gate_up_handle,
                                down_handle,
                                h,
                                inter,
                                nq * hd,
                            )?
                        }
                    } else {
                        kernel.batch_ffn_phase(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle,
                            gate_up_handle,
                            down_handle,
                            h,
                            inter,
                            nq * hd,
                        )?
                    }
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    kernel.batch_ffn_phase(
                        hidden,
                        attn_out,
                        self.ffn_norm.weight(),
                        self.ffn_norm.eps(),
                        attn_proj_handle,
                        gate_up_handle,
                        down_handle,
                        h,
                        inter,
                        nq * hd,
                    )?
                }
            } else {
                false
            };
        if !did_batch_ffn {
            self.attn_output.forward_vec(attn_out, attn_proj)?;
            for i in 0..h {
                hidden[i] += attn_proj[i];
            }
            self.ffn_norm.forward(hidden, normed)?;
            if let Some(fused_handle) = self.fused_gate_up_handle {
                let inter = gate_out.len();
                let total_rows = inter * 2;
                kernel.gemv_cached(fused_handle, normed, fused_gate_up, total_rows, h)?;
                gate_out[..inter].copy_from_slice(&fused_gate_up[..inter]);
                up_out[..inter].copy_from_slice(&fused_gate_up[inter..total_rows]);
            } else {
                self.ffn_gate.forward_vec(normed, gate_out)?;
                self.ffn_up.forward_vec(normed, up_out)?;
            }
            swiglu_fn(gate_out, up_out, swiglu_out);
            self.ffn_down.forward_vec(swiglu_out, down_out)?;
            for i in 0..h {
                hidden[i] += down_out[i];
            }
        }
        Ok(())
    }
    /// Get the layer index.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }
    /// Attempt full-layer GPU dispatch (attention + FFN in a single command buffer).
    ///
    /// Returns:
    /// - `Some(Ok(()))` if the full layer was successfully computed on GPU.
    /// - `Some(Err(..))` if GPU dispatch was attempted but failed.
    /// - `None` if preconditions are not met (handles not available).
    ///
    /// On `Some(Ok(()))`, `hidden` is modified in-place and the caller should
    /// return early, skipping the CPU path entirely. The GPU manages its own
    /// KV cache internally.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn try_full_layer_gpu(
        &self,
        hidden: &mut [f32],
        pos: usize,
        rope: &RopeTable,
        kv_cache: &KvCache,
    ) -> Option<ModelResult<()>> {
        let fused_qkv_handle = self.fused_qkv_handle?;
        let attn_proj_handle = self.attn_output.gpu_handle()?;
        let fused_gate_up_handle = self.fused_gate_up_handle?;
        let down_handle = self.ffn_down.gpu_handle()?;
        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let inter = self.ffn_gate.out_features();
        let eps = self.attn_norm.eps();
        let n_layers = kv_cache.num_layers();
        let max_seq_len = kv_cache.max_seq_len();
        let norm_handle_base = 1_000_000u64 + (self.layer_idx as u64) * 10;
        let attn_norm_handle_id = norm_handle_base;
        let q_norm_handle_id = norm_handle_base + 1;
        let k_norm_handle_id = norm_handle_base + 2;
        let ffn_norm_handle_id = norm_handle_base + 3;
        let (q_blk, k_blk, v_blk, out_blk, gate_blk, up_blk, dn_blk) = match (
            self.attn_q.blocks_1bit(),
            self.attn_k.blocks_1bit(),
            self.attn_v.blocks_1bit(),
            self.attn_output.blocks_1bit(),
            self.ffn_gate.blocks_1bit(),
            self.ffn_up.blocks_1bit(),
            self.ffn_down.blocks_1bit(),
        ) {
            (Some(q), Some(k), Some(v), Some(o), Some(g), Some(u), Some(d)) => {
                (q, k, v, o, g, u, d)
            }
            _ => return None,
        };
        let fused_qkv_bytes = blocks_as_bytes(q_blk);
        let fused_qkv_k_bytes = blocks_as_bytes(k_blk);
        let fused_qkv_v_bytes = blocks_as_bytes(v_blk);
        let mut qkv_concat = Vec::with_capacity(
            fused_qkv_bytes.len() + fused_qkv_k_bytes.len() + fused_qkv_v_bytes.len(),
        );
        qkv_concat.extend_from_slice(fused_qkv_bytes);
        qkv_concat.extend_from_slice(fused_qkv_k_bytes);
        qkv_concat.extend_from_slice(fused_qkv_v_bytes);
        let attn_proj_bytes = blocks_as_bytes(out_blk);
        let gate_bytes = blocks_as_bytes(gate_blk);
        let up_bytes = blocks_as_bytes(up_blk);
        let down_bytes = blocks_as_bytes(dn_blk);
        let rope_cos = rope.cos_at(pos);
        let rope_sin = rope.sin_at(pos);
        let result = oxibonsai_kernels::try_metal_full_layer(
            hidden,
            pos,
            self.layer_idx,
            attn_norm_handle_id,
            self.attn_norm.weight(),
            fused_qkv_handle.id(),
            &qkv_concat,
            q_norm_handle_id,
            self.attn_q_norm.weight(),
            k_norm_handle_id,
            self.attn_k_norm.weight(),
            attn_proj_handle.id(),
            attn_proj_bytes,
            ffn_norm_handle_id,
            self.ffn_norm.weight(),
            fused_gate_up_handle.id(),
            gate_bytes,
            up_bytes,
            down_handle.id(),
            down_bytes,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            eps,
            max_seq_len,
            n_layers,
        );
        match result {
            Ok(()) => {
                tracing::debug!(
                    target : "block_profile", "L{layer}: full-layer GPU dispatch OK",
                    layer = self.layer_idx,
                );
                Some(Ok(()))
            }
            Err(e) => {
                tracing::warn!(
                    layer = self.layer_idx, error = % e,
                    "full-layer GPU dispatch failed, falling back to CPU path"
                );
                Some(Err(crate::error::ModelError::Internal(format!(
                    "Metal full-layer dispatch failed: {e}"
                ))))
            }
        }
    }
    /// Attempt full-layer CUDA GPU dispatch (attention + FFN, no intermediate
    /// CPU round-trips between the two sublayers).
    ///
    /// Returns:
    /// - `Some(Ok(()))` if the full layer was successfully computed on GPU.
    /// - `Some(Err(..))` if GPU dispatch was attempted but failed.
    /// - `None` if preconditions are not met (handles not available).
    #[cfg(all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    ))]
    fn try_full_layer_cuda(
        &self,
        hidden: &mut [f32],
        pos: usize,
        rope: &RopeTable,
        kv_cache: &KvCache,
    ) -> Option<ModelResult<()>> {
        let fused_qkv_handle = self.fused_qkv_handle?;
        let attn_proj_handle = self.attn_output.gpu_handle()?;
        let fused_gate_up_handle = self.fused_gate_up_handle?;
        let down_handle = self.ffn_down.gpu_handle()?;
        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nq / nkv;
        let inter = self.ffn_gate.out_features();
        let eps = self.attn_norm.eps();
        let n_layers = kv_cache.num_layers();
        let max_seq_len = kv_cache.max_seq_len();
        let norm_handle_base = 2_000_000u64 + (self.layer_idx as u64) * 10;
        let attn_norm_handle_id = norm_handle_base;
        let q_norm_handle_id = norm_handle_base + 1;
        let k_norm_handle_id = norm_handle_base + 2;
        let ffn_norm_handle_id = norm_handle_base + 3;
        let (q_blk, k_blk, v_blk, out_blk, gate_blk, up_blk, dn_blk) = match (
            self.attn_q.blocks_1bit(),
            self.attn_k.blocks_1bit(),
            self.attn_v.blocks_1bit(),
            self.attn_output.blocks_1bit(),
            self.ffn_gate.blocks_1bit(),
            self.ffn_up.blocks_1bit(),
            self.ffn_down.blocks_1bit(),
        ) {
            (Some(q), Some(k), Some(v), Some(o), Some(g), Some(u), Some(d)) => {
                (q, k, v, o, g, u, d)
            }
            _ => return None,
        };
        let fused_qkv_bytes = blocks_as_bytes(q_blk);
        let fused_qkv_k_bytes = blocks_as_bytes(k_blk);
        let fused_qkv_v_bytes = blocks_as_bytes(v_blk);
        let mut qkv_concat = Vec::with_capacity(
            fused_qkv_bytes.len() + fused_qkv_k_bytes.len() + fused_qkv_v_bytes.len(),
        );
        qkv_concat.extend_from_slice(fused_qkv_bytes);
        qkv_concat.extend_from_slice(fused_qkv_k_bytes);
        qkv_concat.extend_from_slice(fused_qkv_v_bytes);
        let attn_proj_bytes = blocks_as_bytes(out_blk);
        let gate_bytes = blocks_as_bytes(gate_blk);
        let up_bytes = blocks_as_bytes(up_blk);
        let down_bytes = blocks_as_bytes(dn_blk);
        let rope_cos = rope.cos_at(pos);
        let rope_sin = rope.sin_at(pos);
        let result = oxibonsai_kernels::try_cuda_full_layer(
            hidden,
            pos,
            self.layer_idx,
            attn_norm_handle_id,
            self.attn_norm.weight(),
            fused_qkv_handle.id(),
            &qkv_concat,
            attn_proj_handle.id(),
            attn_proj_bytes,
            q_norm_handle_id,
            self.attn_q_norm.weight(),
            k_norm_handle_id,
            self.attn_k_norm.weight(),
            ffn_norm_handle_id,
            self.ffn_norm.weight(),
            fused_gate_up_handle.id(),
            gate_bytes,
            up_bytes,
            down_handle.id(),
            down_bytes,
            rope_cos,
            rope_sin,
            h,
            inter,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            max_seq_len,
            n_layers,
        );
        match result {
            Ok(()) => {
                tracing::debug!(
                    target : "block_profile", "L{layer}: CUDA full-layer dispatch OK",
                    layer = self.layer_idx,
                );
                Some(Ok(()))
            }
            Err(e) => {
                tracing::warn!(
                    layer = self.layer_idx, error = % e,
                    "CUDA full-layer dispatch failed, falling back to CPU path"
                );
                Some(Err(crate::error::ModelError::Internal(format!(
                    "CUDA full-layer dispatch failed: {e}"
                ))))
            }
        }
    }
    /// Attention norm weight slice.
    pub fn attn_norm_weight(&self) -> &[f32] {
        self.attn_norm.weight()
    }
    /// Attention norm epsilon.
    pub fn attn_norm_eps(&self) -> f32 {
        self.attn_norm.eps()
    }
    /// Q-norm weight slice.
    pub fn q_norm_weight(&self) -> &[f32] {
        self.attn_q_norm.weight()
    }
    /// K-norm weight slice.
    pub fn k_norm_weight(&self) -> &[f32] {
        self.attn_k_norm.weight()
    }
    /// FFN norm weight slice.
    pub fn ffn_norm_weight(&self) -> &[f32] {
        self.ffn_norm.weight()
    }
    /// Layer index.
    pub fn layer_index(&self) -> usize {
        self.layer_idx
    }
    /// Fused QKV GPU handle (if uploaded).
    pub fn fused_qkv_gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.fused_qkv_handle
    }
    /// Attention output projection GPU handle (if uploaded).
    pub fn attn_output_gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.attn_output.gpu_handle()
    }
    /// Fused gate+up GPU handle (if uploaded).
    pub fn fused_gate_up_gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.fused_gate_up_handle
    }
    /// FFN down projection GPU handle (if uploaded).
    pub fn ffn_down_gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.ffn_down.gpu_handle()
    }
    /// Q projection block slice — `None` for ternary layers.
    pub fn attn_q_blocks(&self) -> Option<&[oxibonsai_core::BlockQ1_0G128]> {
        self.attn_q.blocks_1bit()
    }
    /// K projection block slice — `None` for ternary layers.
    pub fn attn_k_blocks(&self) -> Option<&[oxibonsai_core::BlockQ1_0G128]> {
        self.attn_k.blocks_1bit()
    }
    /// V projection block slice — `None` for ternary layers.
    pub fn attn_v_blocks(&self) -> Option<&[oxibonsai_core::BlockQ1_0G128]> {
        self.attn_v.blocks_1bit()
    }
    /// Output projection block slice — `None` for ternary layers.
    pub fn attn_output_blocks(&self) -> Option<&[oxibonsai_core::BlockQ1_0G128]> {
        self.attn_output.blocks_1bit()
    }
    /// FFN gate block slice — `None` for ternary layers.
    pub fn ffn_gate_blocks(&self) -> Option<&[oxibonsai_core::BlockQ1_0G128]> {
        self.ffn_gate.blocks_1bit()
    }
    /// FFN up block slice — `None` for ternary layers.
    pub fn ffn_up_blocks(&self) -> Option<&[oxibonsai_core::BlockQ1_0G128]> {
        self.ffn_up.blocks_1bit()
    }
    /// FFN down block slice — `None` for ternary layers.
    pub fn ffn_down_blocks(&self) -> Option<&[oxibonsai_core::BlockQ1_0G128]> {
        self.ffn_down.blocks_1bit()
    }
    /// FFN gate output features (intermediate_size).
    pub fn ffn_gate_out_features(&self) -> usize {
        self.ffn_gate.out_features()
    }
    /// Q projection block slice — `None` for 1-bit layers.
    pub fn attn_q_blocks_ternary(&self) -> Option<&[oxibonsai_core::BlockTQ2_0_g128]> {
        self.attn_q.blocks_ternary()
    }
    /// K projection block slice — `None` for 1-bit layers.
    pub fn attn_k_blocks_ternary(&self) -> Option<&[oxibonsai_core::BlockTQ2_0_g128]> {
        self.attn_k.blocks_ternary()
    }
    /// V projection block slice — `None` for 1-bit layers.
    pub fn attn_v_blocks_ternary(&self) -> Option<&[oxibonsai_core::BlockTQ2_0_g128]> {
        self.attn_v.blocks_ternary()
    }
    /// Output projection block slice — `None` for 1-bit layers.
    pub fn attn_output_blocks_ternary(&self) -> Option<&[oxibonsai_core::BlockTQ2_0_g128]> {
        self.attn_output.blocks_ternary()
    }
    /// FFN gate block slice — `None` for 1-bit layers.
    pub fn ffn_gate_blocks_ternary(&self) -> Option<&[oxibonsai_core::BlockTQ2_0_g128]> {
        self.ffn_gate.blocks_ternary()
    }
    /// FFN up block slice — `None` for 1-bit layers.
    pub fn ffn_up_blocks_ternary(&self) -> Option<&[oxibonsai_core::BlockTQ2_0_g128]> {
        self.ffn_up.blocks_ternary()
    }
    /// FFN down block slice — `None` for 1-bit layers.
    pub fn ffn_down_blocks_ternary(&self) -> Option<&[oxibonsai_core::BlockTQ2_0_g128]> {
        self.ffn_down.blocks_ternary()
    }
    /// Q projection FP8 E4M3 block slice — `None` for non-FP8-E4M3 layers.
    pub fn attn_q_blocks_fp8e4m3(&self) -> Option<&[oxibonsai_core::BlockFP8E4M3]> {
        self.attn_q.blocks_fp8_e4m3()
    }
    /// K projection FP8 E4M3 block slice — `None` for non-FP8-E4M3 layers.
    pub fn attn_k_blocks_fp8e4m3(&self) -> Option<&[oxibonsai_core::BlockFP8E4M3]> {
        self.attn_k.blocks_fp8_e4m3()
    }
    /// V projection FP8 E4M3 block slice — `None` for non-FP8-E4M3 layers.
    pub fn attn_v_blocks_fp8e4m3(&self) -> Option<&[oxibonsai_core::BlockFP8E4M3]> {
        self.attn_v.blocks_fp8_e4m3()
    }
    /// Output projection FP8 E4M3 block slice — `None` for non-FP8-E4M3 layers.
    pub fn attn_output_blocks_fp8e4m3(&self) -> Option<&[oxibonsai_core::BlockFP8E4M3]> {
        self.attn_output.blocks_fp8_e4m3()
    }
    /// FFN gate FP8 E4M3 block slice — `None` for non-FP8-E4M3 layers.
    pub fn ffn_gate_blocks_fp8e4m3(&self) -> Option<&[oxibonsai_core::BlockFP8E4M3]> {
        self.ffn_gate.blocks_fp8_e4m3()
    }
    /// FFN up FP8 E4M3 block slice — `None` for non-FP8-E4M3 layers.
    pub fn ffn_up_blocks_fp8e4m3(&self) -> Option<&[oxibonsai_core::BlockFP8E4M3]> {
        self.ffn_up.blocks_fp8_e4m3()
    }
    /// FFN down FP8 E4M3 block slice — `None` for non-FP8-E4M3 layers.
    pub fn ffn_down_blocks_fp8e4m3(&self) -> Option<&[oxibonsai_core::BlockFP8E4M3]> {
        self.ffn_down.blocks_fp8_e4m3()
    }
    /// Q projection FP8 E5M2 block slice — `None` for non-FP8-E5M2 layers.
    pub fn attn_q_blocks_fp8e5m2(&self) -> Option<&[oxibonsai_core::BlockFP8E5M2]> {
        self.attn_q.blocks_fp8_e5m2()
    }
    /// K projection FP8 E5M2 block slice — `None` for non-FP8-E5M2 layers.
    pub fn attn_k_blocks_fp8e5m2(&self) -> Option<&[oxibonsai_core::BlockFP8E5M2]> {
        self.attn_k.blocks_fp8_e5m2()
    }
    /// V projection FP8 E5M2 block slice — `None` for non-FP8-E5M2 layers.
    pub fn attn_v_blocks_fp8e5m2(&self) -> Option<&[oxibonsai_core::BlockFP8E5M2]> {
        self.attn_v.blocks_fp8_e5m2()
    }
    /// Output projection FP8 E5M2 block slice — `None` for non-FP8-E5M2 layers.
    pub fn attn_output_blocks_fp8e5m2(&self) -> Option<&[oxibonsai_core::BlockFP8E5M2]> {
        self.attn_output.blocks_fp8_e5m2()
    }
    /// FFN gate FP8 E5M2 block slice — `None` for non-FP8-E5M2 layers.
    pub fn ffn_gate_blocks_fp8e5m2(&self) -> Option<&[oxibonsai_core::BlockFP8E5M2]> {
        self.ffn_gate.blocks_fp8_e5m2()
    }
    /// FFN up FP8 E5M2 block slice — `None` for non-FP8-E5M2 layers.
    pub fn ffn_up_blocks_fp8e5m2(&self) -> Option<&[oxibonsai_core::BlockFP8E5M2]> {
        self.ffn_up.blocks_fp8_e5m2()
    }
    /// FFN down FP8 E5M2 block slice — `None` for non-FP8-E5M2 layers.
    pub fn ffn_down_blocks_fp8e5m2(&self) -> Option<&[oxibonsai_core::BlockFP8E5M2]> {
        self.ffn_down.blocks_fp8_e5m2()
    }
    /// Q projection Q4_0 block slice — `None` for non-Q4_0 layers.
    pub fn attn_q_blocks_q4_0(&self) -> Option<&[oxibonsai_core::BlockQ4_0]> {
        self.attn_q.blocks_q4_0()
    }
    /// K projection Q4_0 block slice — `None` for non-Q4_0 layers.
    pub fn attn_k_blocks_q4_0(&self) -> Option<&[oxibonsai_core::BlockQ4_0]> {
        self.attn_k.blocks_q4_0()
    }
    /// V projection Q4_0 block slice — `None` for non-Q4_0 layers.
    pub fn attn_v_blocks_q4_0(&self) -> Option<&[oxibonsai_core::BlockQ4_0]> {
        self.attn_v.blocks_q4_0()
    }
    /// Output projection Q4_0 block slice — `None` for non-Q4_0 layers.
    pub fn attn_output_blocks_q4_0(&self) -> Option<&[oxibonsai_core::BlockQ4_0]> {
        self.attn_output.blocks_q4_0()
    }
    /// FFN gate Q4_0 block slice — `None` for non-Q4_0 layers.
    pub fn ffn_gate_blocks_q4_0(&self) -> Option<&[oxibonsai_core::BlockQ4_0]> {
        self.ffn_gate.blocks_q4_0()
    }
    /// FFN up Q4_0 block slice — `None` for non-Q4_0 layers.
    pub fn ffn_up_blocks_q4_0(&self) -> Option<&[oxibonsai_core::BlockQ4_0]> {
        self.ffn_up.blocks_q4_0()
    }
    /// FFN down Q4_0 block slice — `None` for non-Q4_0 layers.
    pub fn ffn_down_blocks_q4_0(&self) -> Option<&[oxibonsai_core::BlockQ4_0]> {
        self.ffn_down.blocks_q4_0()
    }
    /// Q projection Q8_0 block slice — `None` for non-Q8_0 layers.
    pub fn attn_q_blocks_q8_0(&self) -> Option<&[oxibonsai_core::BlockQ8_0]> {
        self.attn_q.blocks_q8_0()
    }
    /// K projection Q8_0 block slice — `None` for non-Q8_0 layers.
    pub fn attn_k_blocks_q8_0(&self) -> Option<&[oxibonsai_core::BlockQ8_0]> {
        self.attn_k.blocks_q8_0()
    }
    /// V projection Q8_0 block slice — `None` for non-Q8_0 layers.
    pub fn attn_v_blocks_q8_0(&self) -> Option<&[oxibonsai_core::BlockQ8_0]> {
        self.attn_v.blocks_q8_0()
    }
    /// Output projection Q8_0 block slice — `None` for non-Q8_0 layers.
    pub fn attn_output_blocks_q8_0(&self) -> Option<&[oxibonsai_core::BlockQ8_0]> {
        self.attn_output.blocks_q8_0()
    }
    /// FFN gate Q8_0 block slice — `None` for non-Q8_0 layers.
    pub fn ffn_gate_blocks_q8_0(&self) -> Option<&[oxibonsai_core::BlockQ8_0]> {
        self.ffn_gate.blocks_q8_0()
    }
    /// FFN up Q8_0 block slice — `None` for non-Q8_0 layers.
    pub fn ffn_up_blocks_q8_0(&self) -> Option<&[oxibonsai_core::BlockQ8_0]> {
        self.ffn_up.blocks_q8_0()
    }
    /// FFN down Q8_0 block slice — `None` for non-Q8_0 layers.
    pub fn ffn_down_blocks_q8_0(&self) -> Option<&[oxibonsai_core::BlockQ8_0]> {
        self.ffn_down.blocks_q8_0()
    }

    // ── Q2_K block accessors (BlockQ2K, 84 bytes/block) ──────────────────────
    /// Q projection Q2_K block slice — `None` for non-Q2_K layers.
    pub fn attn_q_blocks_q2k(&self) -> Option<&[oxibonsai_core::BlockQ2K]> {
        self.attn_q.blocks_q2k()
    }
    /// K projection Q2_K block slice — `None` for non-Q2_K layers.
    pub fn attn_k_blocks_q2k(&self) -> Option<&[oxibonsai_core::BlockQ2K]> {
        self.attn_k.blocks_q2k()
    }
    /// V projection Q2_K block slice — `None` for non-Q2_K layers.
    pub fn attn_v_blocks_q2k(&self) -> Option<&[oxibonsai_core::BlockQ2K]> {
        self.attn_v.blocks_q2k()
    }
    /// Output projection Q2_K block slice — `None` for non-Q2_K layers.
    pub fn attn_output_blocks_q2k(&self) -> Option<&[oxibonsai_core::BlockQ2K]> {
        self.attn_output.blocks_q2k()
    }
    /// FFN gate Q2_K block slice — `None` for non-Q2_K layers.
    pub fn ffn_gate_blocks_q2k(&self) -> Option<&[oxibonsai_core::BlockQ2K]> {
        self.ffn_gate.blocks_q2k()
    }
    /// FFN up Q2_K block slice — `None` for non-Q2_K layers.
    pub fn ffn_up_blocks_q2k(&self) -> Option<&[oxibonsai_core::BlockQ2K]> {
        self.ffn_up.blocks_q2k()
    }
    /// FFN down Q2_K block slice — `None` for non-Q2_K layers.
    pub fn ffn_down_blocks_q2k(&self) -> Option<&[oxibonsai_core::BlockQ2K]> {
        self.ffn_down.blocks_q2k()
    }

    // ── Q3_K block accessors (BlockQ3K, 110 bytes/block) ─────────────────────
    /// Q projection Q3_K block slice — `None` for non-Q3_K layers.
    pub fn attn_q_blocks_q3k(&self) -> Option<&[oxibonsai_core::BlockQ3K]> {
        self.attn_q.blocks_q3k()
    }
    /// K projection Q3_K block slice — `None` for non-Q3_K layers.
    pub fn attn_k_blocks_q3k(&self) -> Option<&[oxibonsai_core::BlockQ3K]> {
        self.attn_k.blocks_q3k()
    }
    /// V projection Q3_K block slice — `None` for non-Q3_K layers.
    pub fn attn_v_blocks_q3k(&self) -> Option<&[oxibonsai_core::BlockQ3K]> {
        self.attn_v.blocks_q3k()
    }
    /// Output projection Q3_K block slice — `None` for non-Q3_K layers.
    pub fn attn_output_blocks_q3k(&self) -> Option<&[oxibonsai_core::BlockQ3K]> {
        self.attn_output.blocks_q3k()
    }
    /// FFN gate Q3_K block slice — `None` for non-Q3_K layers.
    pub fn ffn_gate_blocks_q3k(&self) -> Option<&[oxibonsai_core::BlockQ3K]> {
        self.ffn_gate.blocks_q3k()
    }
    /// FFN up Q3_K block slice — `None` for non-Q3_K layers.
    pub fn ffn_up_blocks_q3k(&self) -> Option<&[oxibonsai_core::BlockQ3K]> {
        self.ffn_up.blocks_q3k()
    }
    /// FFN down Q3_K block slice — `None` for non-Q3_K layers.
    pub fn ffn_down_blocks_q3k(&self) -> Option<&[oxibonsai_core::BlockQ3K]> {
        self.ffn_down.blocks_q3k()
    }

    // ── Q4_K block accessors (BlockQ4K, 144 bytes/block) ─────────────────────
    /// Q projection Q4_K block slice — `None` for non-Q4_K layers.
    pub fn attn_q_blocks_q4k(&self) -> Option<&[oxibonsai_core::BlockQ4K]> {
        self.attn_q.blocks_q4k()
    }
    /// K projection Q4_K block slice — `None` for non-Q4_K layers.
    pub fn attn_k_blocks_q4k(&self) -> Option<&[oxibonsai_core::BlockQ4K]> {
        self.attn_k.blocks_q4k()
    }
    /// V projection Q4_K block slice — `None` for non-Q4_K layers.
    pub fn attn_v_blocks_q4k(&self) -> Option<&[oxibonsai_core::BlockQ4K]> {
        self.attn_v.blocks_q4k()
    }
    /// Output projection Q4_K block slice — `None` for non-Q4_K layers.
    pub fn attn_output_blocks_q4k(&self) -> Option<&[oxibonsai_core::BlockQ4K]> {
        self.attn_output.blocks_q4k()
    }
    /// FFN gate Q4_K block slice — `None` for non-Q4_K layers.
    pub fn ffn_gate_blocks_q4k(&self) -> Option<&[oxibonsai_core::BlockQ4K]> {
        self.ffn_gate.blocks_q4k()
    }
    /// FFN up Q4_K block slice — `None` for non-Q4_K layers.
    pub fn ffn_up_blocks_q4k(&self) -> Option<&[oxibonsai_core::BlockQ4K]> {
        self.ffn_up.blocks_q4k()
    }
    /// FFN down Q4_K block slice — `None` for non-Q4_K layers.
    pub fn ffn_down_blocks_q4k(&self) -> Option<&[oxibonsai_core::BlockQ4K]> {
        self.ffn_down.blocks_q4k()
    }

    // ── Q5_K block accessors (BlockQ5K, 176 bytes/block) ─────────────────────
    /// Q projection Q5_K block slice — `None` for non-Q5_K layers.
    pub fn attn_q_blocks_q5k(&self) -> Option<&[oxibonsai_core::BlockQ5K]> {
        self.attn_q.blocks_q5k()
    }
    /// K projection Q5_K block slice — `None` for non-Q5_K layers.
    pub fn attn_k_blocks_q5k(&self) -> Option<&[oxibonsai_core::BlockQ5K]> {
        self.attn_k.blocks_q5k()
    }
    /// V projection Q5_K block slice — `None` for non-Q5_K layers.
    pub fn attn_v_blocks_q5k(&self) -> Option<&[oxibonsai_core::BlockQ5K]> {
        self.attn_v.blocks_q5k()
    }
    /// Output projection Q5_K block slice — `None` for non-Q5_K layers.
    pub fn attn_output_blocks_q5k(&self) -> Option<&[oxibonsai_core::BlockQ5K]> {
        self.attn_output.blocks_q5k()
    }
    /// FFN gate Q5_K block slice — `None` for non-Q5_K layers.
    pub fn ffn_gate_blocks_q5k(&self) -> Option<&[oxibonsai_core::BlockQ5K]> {
        self.ffn_gate.blocks_q5k()
    }
    /// FFN up Q5_K block slice — `None` for non-Q5_K layers.
    pub fn ffn_up_blocks_q5k(&self) -> Option<&[oxibonsai_core::BlockQ5K]> {
        self.ffn_up.blocks_q5k()
    }
    /// FFN down Q5_K block slice — `None` for non-Q5_K layers.
    pub fn ffn_down_blocks_q5k(&self) -> Option<&[oxibonsai_core::BlockQ5K]> {
        self.ffn_down.blocks_q5k()
    }

    // ── Q6_K block accessors (BlockQ6K, 210 bytes/block) ─────────────────────
    /// Q projection Q6_K block slice — `None` for non-Q6_K layers.
    pub fn attn_q_blocks_q6k(&self) -> Option<&[oxibonsai_core::BlockQ6K]> {
        self.attn_q.blocks_q6k()
    }
    /// K projection Q6_K block slice — `None` for non-Q6_K layers.
    pub fn attn_k_blocks_q6k(&self) -> Option<&[oxibonsai_core::BlockQ6K]> {
        self.attn_k.blocks_q6k()
    }
    /// V projection Q6_K block slice — `None` for non-Q6_K layers.
    pub fn attn_v_blocks_q6k(&self) -> Option<&[oxibonsai_core::BlockQ6K]> {
        self.attn_v.blocks_q6k()
    }
    /// Output projection Q6_K block slice — `None` for non-Q6_K layers.
    pub fn attn_output_blocks_q6k(&self) -> Option<&[oxibonsai_core::BlockQ6K]> {
        self.attn_output.blocks_q6k()
    }
    /// FFN gate Q6_K block slice — `None` for non-Q6_K layers.
    pub fn ffn_gate_blocks_q6k(&self) -> Option<&[oxibonsai_core::BlockQ6K]> {
        self.ffn_gate.blocks_q6k()
    }
    /// FFN up Q6_K block slice — `None` for non-Q6_K layers.
    pub fn ffn_up_blocks_q6k(&self) -> Option<&[oxibonsai_core::BlockQ6K]> {
        self.ffn_up.blocks_q6k()
    }
    /// FFN down Q6_K block slice — `None` for non-Q6_K layers.
    pub fn ffn_down_blocks_q6k(&self) -> Option<&[oxibonsai_core::BlockQ6K]> {
        self.ffn_down.blocks_q6k()
    }

    // ── Q8_K block accessors (BlockQ8K, 292 bytes/block) ─────────────────────
    /// Q projection Q8_K block slice — `None` for non-Q8_K layers.
    pub fn attn_q_blocks_q8k(&self) -> Option<&[oxibonsai_core::BlockQ8K]> {
        self.attn_q.blocks_q8k()
    }
    /// K projection Q8_K block slice — `None` for non-Q8_K layers.
    pub fn attn_k_blocks_q8k(&self) -> Option<&[oxibonsai_core::BlockQ8K]> {
        self.attn_k.blocks_q8k()
    }
    /// V projection Q8_K block slice — `None` for non-Q8_K layers.
    pub fn attn_v_blocks_q8k(&self) -> Option<&[oxibonsai_core::BlockQ8K]> {
        self.attn_v.blocks_q8k()
    }
    /// Output projection Q8_K block slice — `None` for non-Q8_K layers.
    pub fn attn_output_blocks_q8k(&self) -> Option<&[oxibonsai_core::BlockQ8K]> {
        self.attn_output.blocks_q8k()
    }
    /// FFN gate Q8_K block slice — `None` for non-Q8_K layers.
    pub fn ffn_gate_blocks_q8k(&self) -> Option<&[oxibonsai_core::BlockQ8K]> {
        self.ffn_gate.blocks_q8k()
    }
    /// FFN up Q8_K block slice — `None` for non-Q8_K layers.
    pub fn ffn_up_blocks_q8k(&self) -> Option<&[oxibonsai_core::BlockQ8K]> {
        self.ffn_up.blocks_q8k()
    }
    /// FFN down Q8_K block slice — `None` for non-Q8_K layers.
    pub fn ffn_down_blocks_q8k(&self) -> Option<&[oxibonsai_core::BlockQ8K]> {
        self.ffn_down.blocks_q8k()
    }
}
/// Pre-allocated scratch buffers for a single TransformerBlock's forward pass.
/// Eliminates per-token heap allocations in the hot path.
struct ScratchBuffers {
    normed: Vec<f32>,
    q_all: Vec<f32>,
    k_all: Vec<f32>,
    v_all: Vec<f32>,
    q_normed: Vec<f32>,
    k_normed: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    attn_out: Vec<f32>,
    attn_proj: Vec<f32>,
    gate_out: Vec<f32>,
    up_out: Vec<f32>,
    swiglu_out: Vec<f32>,
    down_out: Vec<f32>,
    fused_qkv: Vec<f32>,
    fused_gate_up: Vec<f32>,
}
impl ScratchBuffers {
    fn new(h: usize, nq: usize, nkv: usize, hd: usize, inter: usize) -> Self {
        Self {
            normed: vec![0.0; h],
            q_all: vec![0.0; nq * hd],
            k_all: vec![0.0; nkv * hd],
            v_all: vec![0.0; nkv * hd],
            q_normed: vec![0.0; nq * hd],
            k_normed: vec![0.0; nkv * hd],
            q_rope: vec![0.0; nq * hd],
            k_rope: vec![0.0; nkv * hd],
            attn_out: vec![0.0; nq * hd],
            attn_proj: vec![0.0; h],
            gate_out: vec![0.0; inter],
            up_out: vec![0.0; inter],
            swiglu_out: vec![0.0; inter],
            down_out: vec![0.0; h],
            fused_qkv: vec![0.0; nq * hd + nkv * hd + nkv * hd],
            fused_gate_up: vec![0.0; inter * 2],
        }
    }
    /// Zero all buffers before reuse.
    fn clear(&mut self) {
        self.normed.fill(0.0);
        self.q_all.fill(0.0);
        self.k_all.fill(0.0);
        self.v_all.fill(0.0);
        self.q_normed.fill(0.0);
        self.k_normed.fill(0.0);
        self.q_rope.fill(0.0);
        self.k_rope.fill(0.0);
        self.attn_out.fill(0.0);
        self.attn_proj.fill(0.0);
        self.gate_out.fill(0.0);
        self.up_out.fill(0.0);
        self.swiglu_out.fill(0.0);
        self.down_out.fill(0.0);
        self.fused_qkv.fill(0.0);
        self.fused_gate_up.fill(0.0);
    }
}
