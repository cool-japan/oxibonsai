//! Single-token forward pass with timing statistics
//! (`TransformerBlock::forward_with_stats`).

use crate::error::ModelResult;
use crate::kv_cache::KvCache;
use crate::layers::rope::RopeTable;
use crate::layers::swiglu::try_swiglu;
use oxibonsai_kernels::traits::OneBitKernel;
use std::time::Instant;

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::block::functions::blocks_as_bytes;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::block::functions::blocks_as_bytes_ternary;
use crate::block::functions::compute_gqa_attention;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::block::functions::try_metal_gemv_ternary_fused;
use crate::block::functions::{advance_kv_cache_to, validate_shapes};

use super::block_def::TransformerBlock;
use super::layer_stats::LayerStats;
use super::scratch::ScratchBuffers;

impl<'a> TransformerBlock<'a> {
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
        // M-12 / M-19: same validation + cursor bookkeeping as `forward()`;
        // see its comments for the rationale.
        validate_shapes(
            self.layer_idx,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.attn_q.out_features(),
            kv_cache,
            pos,
        )?;
        advance_kv_cache_to(kv_cache, pos);
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
                    // M-21 hardening: see `forward.rs` — `gemv_cached`
                    // always decodes `Q1_0_g128`, so a `fused_qkv_handle`
                    // built over any other format must never reach it.
                    if self.attn_q.blocks_1bit().is_some() {
                        kernel.gemv_cached(fused_handle, normed, fused_qkv, total_rows, h)?;
                    } else {
                        self.attn_q.forward_vec(normed, q_all)?;
                        self.attn_k.forward_vec(normed, k_all)?;
                        self.attn_v.forward_vec(normed, v_all)?;
                    }
                }
                if metal_ok || self.attn_q.blocks_1bit().is_some() {
                    q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                    k_all[..k_rows].copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                    v_all[..k_rows].copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
                }
            } else {
                // M-21: ternary fused-QKV Metal fast path; see `forward.rs`
                // for the full rationale (no `fused_qkv_handle` exists for
                // ternary, so key on `blocks_ternary()` + the ternary GPU
                // handle's own `.id()` instead, gated on
                // `attn_q.gpu_handle()`). The id — not the weight's mmap
                // pointer — comes from the same process-global monotonic
                // counter 1-bit handles use, so it is never reused across
                // model loads.
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let ternary_metal_ok = {
                    if let (Some(q_blk), Some(k_blk), Some(v_blk)) = (
                        self.attn_q.blocks_ternary(),
                        self.attn_k.blocks_ternary(),
                        self.attn_v.blocks_ternary(),
                    ) {
                        if let Some(hnd) = self.attn_q.gpu_handle() {
                            let q_rows = nq * hd;
                            let k_rows = nkv * hd;
                            let total_rows = q_rows + k_rows + k_rows;
                            let q_bytes = blocks_as_bytes_ternary(q_blk);
                            let k_bytes = blocks_as_bytes_ternary(k_blk);
                            let v_bytes = blocks_as_bytes_ternary(v_blk);
                            let slot = hnd.id();
                            if try_metal_gemv_ternary_fused(
                                normed,
                                fused_qkv,
                                slot,
                                &[q_bytes, k_bytes, v_bytes],
                                total_rows,
                                h,
                            )
                            .is_ok()
                            {
                                q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                                k_all[..k_rows]
                                    .copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                                v_all[..k_rows]
                                    .copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let ternary_metal_ok = false;
                if !ternary_metal_ok {
                    self.attn_q.forward_vec(normed, q_all)?;
                    self.attn_k.forward_vec(normed, k_all)?;
                    self.attn_v.forward_vec(normed, v_all)?;
                }
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
            kv_cache.try_store_key(self.layer_idx, head, pos, &k_rope[start..start + hd])?;
            kv_cache.try_store_value(self.layer_idx, head, pos, &v_all[start..start + hd])?;
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
        let did_batch_ffn = if let (
            Some(attn_proj_handle),
            Some(gate_up_handle),
            Some(down_handle),
        ) = (
            self.attn_output.gpu_handle(),
            self.fused_gate_up_handle,
            self.ffn_down.gpu_handle(),
        ) {
            // M-21 hardening: see `forward.rs` for the full rationale —
            // `batch_ffn_phase`/`try_metal_ffn` always decode
            // `Q1_0_g128`, so this whole-layer batched path must
            // confirm every matrix is genuinely 1-bit before entering.
            if self.attn_output.blocks_1bit().is_some()
                && self.ffn_gate.blocks_1bit().is_some()
                && self.ffn_up.blocks_1bit().is_some()
                && self.ffn_down.blocks_1bit().is_some()
            {
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
                // M-21 hardening: mirrors the QKV-side guard above.
                if self.ffn_gate.blocks_1bit().is_some() {
                    let inter = gate_out.len();
                    let total_rows = inter * 2;
                    kernel.gemv_cached(fused_handle, normed, fused_gate_up, total_rows, h)?;
                    gate_out[..inter].copy_from_slice(&fused_gate_up[..inter]);
                    up_out[..inter].copy_from_slice(&fused_gate_up[inter..total_rows]);
                } else {
                    self.ffn_gate.forward_vec(normed, gate_out)?;
                    self.ffn_up.forward_vec(normed, up_out)?;
                }
            } else {
                self.ffn_gate.forward_vec(normed, gate_out)?;
                self.ffn_up.forward_vec(normed, up_out)?;
            }
            try_swiglu(gate_out, up_out, swiglu_out)?;
            self.ffn_down.forward_vec(swiglu_out, down_out)?;
            for i in 0..h {
                hidden[i] += down_out[i];
            }
        }
        stats.ffn_us = ffn_start.elapsed().as_micros() as u64;
        stats.total_us = total_start.elapsed().as_micros() as u64;
        Ok(stats)
    }
}
