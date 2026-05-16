//! Single-token forward pass (`TransformerBlock::forward`).

use crate::error::ModelResult;
use crate::kv_cache::KvCache;
use crate::layers::rope::RopeTable;
use crate::layers::swiglu::swiglu as swiglu_fn;
use oxibonsai_kernels::traits::OneBitKernel;
use std::time::Instant;

#[cfg(any(
    feature = "metal",
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
use crate::block::functions::blocks_as_bytes;
use crate::block::functions::compute_gqa_attention;

use super::block_def::TransformerBlock;
use super::scratch::ScratchBuffers;

impl<'a> TransformerBlock<'a> {
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
}
