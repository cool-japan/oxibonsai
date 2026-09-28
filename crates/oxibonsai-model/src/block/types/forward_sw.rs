//! Sliding-window forward pass (`TransformerBlock::forward_with_sliding_window`).

use crate::error::{ModelError, ModelResult};
use crate::kv_cache::KvCache;
use crate::layers::attention_fused::fused_attention_head_contiguous;
use crate::layers::rope::RopeTable;
use crate::layers::sliding_window::SlidingWindowConfig;
use crate::layers::swiglu::try_swiglu;
use oxibonsai_kernels::traits::OneBitKernel;
use rayon::prelude::*;

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::block::functions::blocks_as_bytes;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::block::functions::blocks_as_bytes_ternary;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::block::functions::try_metal_gemv_ternary_fused;
use crate::block::functions::{advance_kv_cache_to, validate_shapes};
use crate::block::functions::{compute_gqa_attention, PAR_HEAD_MIN_HEADS};

use super::block_def::TransformerBlock;
use super::scratch::ScratchBuffers;

impl<'a> TransformerBlock<'a> {
    /// Forward pass with optional sliding window attention.
    ///
    /// When `sliding_window` is `Some`, attention is restricted to positions
    /// within the window, reducing compute for long sequences.
    ///
    /// # M-17
    ///
    /// This entry point is not yet called from `BonsaiModel`'s per-token
    /// loop: `Qwen3Config` has no `sliding_window` field to route on, and
    /// wiring that requires a change to `model/types/mod.rs` (owned by a
    /// different package this wave — see this package's `deviations`). It
    /// is kept rather than deleted because `OneBitKernel::batch_attn_phase`
    /// (`traits.rs`) names this file as its documented revival point; the
    /// parity test below (`sliding_window_with_window_ge_seq_len_matches_full_attention`
    /// in `functions.rs`) verifies it is correct today, so the only
    /// remaining gap is the model-level call site.
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
                // for the full rationale. The slot is the ternary GPU
                // handle's own `.id()`, not the weight's mmap pointer — ids
                // come from the same process-global monotonic counter
                // 1-bit handles use, so they are never reused across model
                // loads.
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
            kv_cache.try_store_key(self.layer_idx, head, pos, &k_rope[start..start + hd])?;
            kv_cache.try_store_value(self.layer_idx, head, pos, &v_all[start..start + hd])?;
        }
        let full_seq_len = pos + 1;
        if let Some(sw_config) = sliding_window {
            let (positions, _count) =
                crate::layers::sliding_window::attention_range(pos, full_seq_len, sw_config);
            let num_kv_heads = nq / heads_per_group;
            let windowed_len = positions.len();
            // Read only the window's rows (widened from `f16` when the
            // cache stores `f16`), never the whole `0..full_seq_len` history.
            let windowed_kv: Vec<(Vec<f32>, Vec<f32>)> = (0..num_kv_heads)
                .map(|kv_h| {
                    (
                        kv_cache.gather_keys(self.layer_idx, kv_h, &positions),
                        kv_cache.gather_values(self.layer_idx, kv_h, &positions),
                    )
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
        Ok(())
    }
}
