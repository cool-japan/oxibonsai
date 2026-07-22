//! Metal GPU **batch prefill** for FP8 (E4M3 / E5M2) models (Phase 28.B).
//!
//! # Why this exists — and why it is correct
//!
//! The FP8 per-token decode path (`BonsaiModel::forward` → `TransformerBlock::forward`)
//! runs its attention over the **CPU** [`crate::kv_cache::KvCache`] owned by the
//! model (`self.kv_cache`) and dispatches its linear projections through the FP8
//! GEMV kernels. Any batch prefill for FP8 must therefore leave K/V for every
//! prompt position in *that same* cache, or decode will attend over all-zero
//! prompt state and silently corrupt generation.
//!
//! The CUDA FP8 batch prefill (`forward_cuda_fp8.rs`) writes a **GPU-private** KV
//! cache that is never read back, so it is disabled by default (the documented
//! split-KV-cache trap). This module avoids that trap **by construction** with a
//! hybrid split:
//!
//! - The heavy linear projections — fused QKV, attention output, fused gate/up
//!   SwiGLU, and FFN down — run as **batched FP8 GEMMs on the GPU** via
//!   [`oxibonsai_kernels::metal_gemm_fp8_e4m3`] & friends (Phase 28 kernels),
//!   processing every prompt position in one dispatch instead of the per-token
//!   sequential GEMV loop.
//! - Attention and the K/V store run on the **CPU** against `self.kv_cache`,
//!   using the exact same [`fused_attention_head_contiguous`] the per-token path
//!   uses. Decode then reads real prompt K/V — no split cache, no read-back.
//!
//! This is still a large speedup over the per-token path (one GEMM per weight
//! matrix per layer instead of `batch` sequential GEMVs) while remaining
//! bit-faithful to the sequential FP8 forward within FP32 noise.

#![cfg(all(feature = "metal", target_os = "macos"))]

use super::{BonsaiModel, OutputWeight};
use crate::layers::attention_fused::fused_attention_head_contiguous;
use crate::layers::rms_norm::RmsNorm;
use oxibonsai_kernels::MetalGraphError;

// ── FP8 block → raw-byte reinterpretation ────────────────────────────────────
//
// `BlockFP8E4M3` / `BlockFP8E5M2` are `#[repr(C)]` with a compile-time
// `size_of == BLOCK_FP8_BYTES` (34) assertion in `oxibonsai_core`, so a flat
// byte view is exactly the on-disk / on-GPU block layout the kernels expect.

fn fp8_e4m3_as_bytes(blocks: &[oxibonsai_core::BlockFP8E4M3]) -> &[u8] {
    // SAFETY: `BlockFP8E4M3` is `#[repr(C)]`, `size_of == BLOCK_FP8_BYTES`, and
    // the slice is valid for `blocks.len() * BLOCK_FP8_BYTES` initialized bytes.
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
        )
    }
}

fn fp8_e5m2_as_bytes(blocks: &[oxibonsai_core::BlockFP8E5M2]) -> &[u8] {
    // SAFETY: see `fp8_e4m3_as_bytes`.
    unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
        )
    }
}

// ── Variant-dispatched GEMM wrappers ─────────────────────────────────────────

/// Batched FP8 GEMM (accumulate): `outputs += W · inputs`. Column/token-major.
fn gemm_fp8(
    is_e4m3: bool,
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), MetalGraphError> {
    if is_e4m3 {
        oxibonsai_kernels::metal_gemm_fp8_e4m3(blocks, inputs, outputs, n_rows, k, batch_size)
    } else {
        oxibonsai_kernels::metal_gemm_fp8_e5m2(blocks, inputs, outputs, n_rows, k, batch_size)
    }
}

/// Batched FP8 GEMM with fused residual: `outputs = residual + W · inputs`.
#[allow(clippy::too_many_arguments)]
fn gemm_fp8_residual(
    is_e4m3: bool,
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    residual: &[f32],
    n_rows: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), MetalGraphError> {
    if is_e4m3 {
        oxibonsai_kernels::metal_gemm_fp8_e4m3_residual(
            blocks, inputs, outputs, residual, n_rows, k, batch_size,
        )
    } else {
        oxibonsai_kernels::metal_gemm_fp8_e5m2_residual(
            blocks, inputs, outputs, residual, n_rows, k, batch_size,
        )
    }
}

/// Fused gate+up FP8 GEMM with SwiGLU epilogue: `out = SiLU(gate·x) * (up·x)`.
fn fused_gate_up_swiglu_fp8(
    is_e4m3: bool,
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_ffn_rows: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), MetalGraphError> {
    if is_e4m3 {
        oxibonsai_kernels::metal_fused_gate_up_swiglu_fp8_e4m3(
            blocks, inputs, outputs, n_ffn_rows, k, batch_size,
        )
    } else {
        oxibonsai_kernels::metal_fused_gate_up_swiglu_fp8_e5m2(
            blocks, inputs, outputs, n_ffn_rows, k, batch_size,
        )
    }
}

/// Owned per-layer weight buffers + norm layers for an FP8 batch prefill.
///
/// Everything here is **owned** (byte copies / cloned norm weights) so the
/// prefill loop can mutably borrow `self.kv_cache` while iterating without
/// keeping any borrow of `self.blocks` alive.
struct Fp8LayerBufs {
    layer_idx: usize,
    attn_norm: RmsNorm,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    ffn_norm: RmsNorm,
    /// Fused Q|K|V FP8 block bytes (rows = `nq*hd + 2*nkv*hd`, k = `hidden`).
    qkv_bytes: Vec<u8>,
    /// Attention-output projection FP8 block bytes (rows = `hidden`, k = `nq*hd`).
    attn_proj_bytes: Vec<u8>,
    /// Fused gate|up FP8 block bytes (rows = `2*inter`, k = `hidden`).
    gate_up_bytes: Vec<u8>,
    /// FFN down projection FP8 block bytes (rows = `hidden`, k = `inter`).
    down_bytes: Vec<u8>,
}

impl<'a> BonsaiModel<'a> {
    /// Collect owned per-layer FP8 weight buffers + norm layers.
    ///
    /// `eps` is the shared RMSNorm epsilon (`config.rms_norm_eps`), which every
    /// per-block norm — attn / q / k / ffn — is constructed with at load time.
    fn build_fp8_layer_bufs(
        &self,
        is_e4m3: bool,
        eps: f32,
    ) -> Result<Vec<Fp8LayerBufs>, Box<dyn std::error::Error>> {
        let mut out = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (qkv_bytes, attn_proj_bytes, gate_up_bytes, down_bytes) = if is_e4m3 {
                let qb = fp8_e4m3_as_bytes(
                    block
                        .attn_q_blocks_fp8e4m3()
                        .ok_or("attn_q: not FP8 E4M3")?,
                );
                let kb = fp8_e4m3_as_bytes(
                    block
                        .attn_k_blocks_fp8e4m3()
                        .ok_or("attn_k: not FP8 E4M3")?,
                );
                let vb = fp8_e4m3_as_bytes(
                    block
                        .attn_v_blocks_fp8e4m3()
                        .ok_or("attn_v: not FP8 E4M3")?,
                );
                let apb = fp8_e4m3_as_bytes(
                    block
                        .attn_output_blocks_fp8e4m3()
                        .ok_or("attn_output: not FP8 E4M3")?,
                );
                let gb = fp8_e4m3_as_bytes(
                    block
                        .ffn_gate_blocks_fp8e4m3()
                        .ok_or("ffn_gate: not FP8 E4M3")?,
                );
                let ub = fp8_e4m3_as_bytes(
                    block
                        .ffn_up_blocks_fp8e4m3()
                        .ok_or("ffn_up: not FP8 E4M3")?,
                );
                let db = fp8_e4m3_as_bytes(
                    block
                        .ffn_down_blocks_fp8e4m3()
                        .ok_or("ffn_down: not FP8 E4M3")?,
                );
                (
                    concat_bytes(&[qb, kb, vb]),
                    apb.to_vec(),
                    concat_bytes(&[gb, ub]),
                    db.to_vec(),
                )
            } else {
                let qb = fp8_e5m2_as_bytes(
                    block
                        .attn_q_blocks_fp8e5m2()
                        .ok_or("attn_q: not FP8 E5M2")?,
                );
                let kb = fp8_e5m2_as_bytes(
                    block
                        .attn_k_blocks_fp8e5m2()
                        .ok_or("attn_k: not FP8 E5M2")?,
                );
                let vb = fp8_e5m2_as_bytes(
                    block
                        .attn_v_blocks_fp8e5m2()
                        .ok_or("attn_v: not FP8 E5M2")?,
                );
                let apb = fp8_e5m2_as_bytes(
                    block
                        .attn_output_blocks_fp8e5m2()
                        .ok_or("attn_output: not FP8 E5M2")?,
                );
                let gb = fp8_e5m2_as_bytes(
                    block
                        .ffn_gate_blocks_fp8e5m2()
                        .ok_or("ffn_gate: not FP8 E5M2")?,
                );
                let ub = fp8_e5m2_as_bytes(
                    block
                        .ffn_up_blocks_fp8e5m2()
                        .ok_or("ffn_up: not FP8 E5M2")?,
                );
                let db = fp8_e5m2_as_bytes(
                    block
                        .ffn_down_blocks_fp8e5m2()
                        .ok_or("ffn_down: not FP8 E5M2")?,
                );
                (
                    concat_bytes(&[qb, kb, vb]),
                    apb.to_vec(),
                    concat_bytes(&[gb, ub]),
                    db.to_vec(),
                )
            };
            out.push(Fp8LayerBufs {
                layer_idx: block.layer_index(),
                attn_norm: RmsNorm::new(block.attn_norm_weight().to_vec(), eps),
                q_norm: RmsNorm::new(block.q_norm_weight().to_vec(), eps),
                k_norm: RmsNorm::new(block.k_norm_weight().to_vec(), eps),
                ffn_norm: RmsNorm::new(block.ffn_norm_weight().to_vec(), eps),
                qkv_bytes,
                attn_proj_bytes,
                gate_up_bytes,
                down_bytes,
            });
        }
        Ok(out)
    }

    /// Metal FP8 (E4M3 / E5M2) batch prefill: all layers + final norm + LM head.
    ///
    /// Runs the linear projections as batched FP8 GEMMs on the GPU, while
    /// applying q/k-norm, RoPE, the K/V store, and attention on the CPU against
    /// `self.kv_cache` — the same cache the per-token FP8 decode path reads.
    /// Returns the **last** prompt position's logits (for generation to start);
    /// every prompt position's K/V is left in `self.kv_cache`.
    ///
    /// Marked `pub` so parity tests can drive this **strict** path directly,
    /// bypassing the silent sequential fallback in
    /// [`Self::forward_prefill`](super::BonsaiModel::forward_prefill).
    ///
    /// Returns `Err` (caller falls back to the sequential per-token path) when
    /// the Metal device is unavailable, a weight is not the expected FP8
    /// variant, the LM head is not FP8, or the prompt overruns the context.
    pub fn try_metal_prefill_with_lm_head_fp8(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        is_e4m3: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        if batch_size == 0 {
            return Err("FP8 prefill: empty token_ids".into());
        }
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        // Context-length guard: the batched RoPE gather below indexes
        // `self.rope.{cos,sin}_at(pos_start + t)` with no bound, and `RopeTable`
        // is sized to exactly `max_seq_len` rows — a prompt that overflows the
        // context would slice out of bounds and panic inside the request task.
        // Returning Err makes `forward_prefill` fall back to the sequential
        // path, whose per-token `forward()` returns a clean `SequenceTooLong`.
        if pos_start + batch_size > self.kv_cache.max_seq_len() {
            return Err(format!(
                "FP8 prefill sequence too long: {batch_size} tokens at pos {pos_start} exceeds max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }

        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let heads_per_group = nq.checked_div(nkv).unwrap_or(1);
        let eps = self.blocks[0].attn_norm_eps();
        let q_rows = nq * hd;
        let kv_rows = nkv * hd;
        let total_qkv_rows = q_rows + 2 * kv_rows;

        // ── Extract owned LM head + final norm (variant must match the model) ─
        let (lm_head_bytes, lm_head_out_features): (Vec<u8>, usize) = match &self.output_weight {
            OutputWeight::FP8E4M3(lm) if is_e4m3 => {
                (fp8_e4m3_as_bytes(lm.blocks()).to_vec(), lm.out_features())
            }
            OutputWeight::FP8E5M2(lm) if !is_e4m3 => {
                (fp8_e5m2_as_bytes(lm.blocks()).to_vec(), lm.out_features())
            }
            _ => return Err("FP8 LM head expected (variant mismatch)".into()),
        };
        let final_norm = RmsNorm::new(self.output_norm.weight().to_vec(), self.output_norm.eps());

        // ── Extract owned per-layer weight buffers ───────────────────────────
        let layer_bufs = self.build_fp8_layer_bufs(is_e4m3, eps)?;

        // ── Embed prompt tokens into `[batch × hidden]` (token-major) ────────
        let mut hidden = vec![0.0f32; batch_size * h];
        for (t, &token_id) in token_ids.iter().enumerate() {
            let embd_start = token_id as usize * h;
            let embd_end = embd_start + h;
            if embd_end > self.token_embd.len() {
                return Err(format!(
                    "token_id {} out of range (vocab={})",
                    token_id,
                    self.token_embd.len() / h
                )
                .into());
            }
            hidden[t * h..(t + 1) * h].copy_from_slice(&self.token_embd[embd_start..embd_end]);
        }

        // From here on we only touch owned buffers + `self.rope` (shared) and
        // `self.kv_cache` (mutable) — disjoint fields, so both coexist.

        // Reusable scratch buffers.
        let mut normed = vec![0.0f32; batch_size * h];
        let mut qkv = vec![0.0f32; batch_size * total_qkv_rows];
        let mut q_rope_batch = vec![0.0f32; batch_size * q_rows];
        let mut attn_out = vec![0.0f32; batch_size * q_rows];
        let mut next_hidden = vec![0.0f32; batch_size * h];
        let mut swiglu_batch = vec![0.0f32; batch_size * inter];
        let mut head_norm = vec![0.0f32; hd];
        let mut head_rope = vec![0.0f32; hd];

        for lb in &layer_bufs {
            let layer_idx = lb.layer_idx;

            // 1. Pre-attention RMSNorm (per position).
            for t in 0..batch_size {
                lb.attn_norm
                    .forward(&hidden[t * h..(t + 1) * h], &mut normed[t * h..(t + 1) * h])?;
            }

            // 2. Fused QKV projection — one batched GEMM over all positions.
            qkv.fill(0.0);
            gemm_fp8(
                is_e4m3,
                &lb.qkv_bytes,
                &normed,
                &mut qkv,
                total_qkv_rows,
                h,
                batch_size,
            )
            .map_err(box_err)?;

            // 3. Per-position q/k-norm + RoPE + K/V store into `self.kv_cache`.
            for t in 0..batch_size {
                let pos = pos_start + t;
                let base = t * total_qkv_rows;
                // Q heads: q-norm then RoPE into the batched query buffer.
                for head in 0..nq {
                    let s = base + head * hd;
                    lb.q_norm.forward(&qkv[s..s + hd], &mut head_norm)?;
                    let out_base = t * q_rows + head * hd;
                    self.rope
                        .apply(&head_norm, &mut q_rope_batch[out_base..out_base + hd], pos)?;
                }
                // K/V heads: k-norm + RoPE for K, raw V, both stored per head.
                for head in 0..nkv {
                    let k_s = base + q_rows + head * hd;
                    let v_s = base + q_rows + kv_rows + head * hd;
                    lb.k_norm.forward(&qkv[k_s..k_s + hd], &mut head_norm)?;
                    self.rope.apply(&head_norm, &mut head_rope, pos)?;
                    self.kv_cache.store_key(layer_idx, head, pos, &head_rope);
                    self.kv_cache
                        .store_value(layer_idx, head, pos, &qkv[v_s..v_s + hd]);
                }
            }

            // 4. GQA attention (per position) against `self.kv_cache` — causal
            //    via `seq_len = pos + 1`. Identical math to the per-token path.
            for t in 0..batch_size {
                let seq_len = pos_start + t + 1;
                let q_base = t * q_rows;
                for q_head in 0..nq {
                    let kv_head = q_head / heads_per_group;
                    let qh = q_base + q_head * hd;
                    let keys = self.kv_cache.keys_for(layer_idx, kv_head, seq_len);
                    let values = self.kv_cache.values_for(layer_idx, kv_head, seq_len);
                    fused_attention_head_contiguous(
                        &q_rope_batch[qh..qh + hd],
                        keys,
                        values,
                        &mut attn_out[qh..qh + hd],
                        seq_len,
                        hd,
                    )?;
                }
            }

            // 5. Attention-output projection + residual — one batched GEMM.
            //    next_hidden = hidden + W_o · attn_out.
            gemm_fp8_residual(
                is_e4m3,
                &lb.attn_proj_bytes,
                &attn_out,
                &mut next_hidden,
                &hidden,
                h,
                q_rows,
                batch_size,
            )
            .map_err(box_err)?;
            hidden.copy_from_slice(&next_hidden);

            // 6. Pre-FFN RMSNorm (per position).
            for t in 0..batch_size {
                lb.ffn_norm
                    .forward(&hidden[t * h..(t + 1) * h], &mut normed[t * h..(t + 1) * h])?;
            }

            // 7. Fused gate+up + SwiGLU — one batched GEMM over all positions.
            fused_gate_up_swiglu_fp8(
                is_e4m3,
                &lb.gate_up_bytes,
                &normed,
                &mut swiglu_batch,
                inter,
                h,
                batch_size,
            )
            .map_err(box_err)?;

            // 8. FFN down projection + residual — one batched GEMM.
            //    next_hidden = hidden + W_down · swiglu.
            gemm_fp8_residual(
                is_e4m3,
                &lb.down_bytes,
                &swiglu_batch,
                &mut next_hidden,
                &hidden,
                h,
                inter,
                batch_size,
            )
            .map_err(box_err)?;
            hidden.copy_from_slice(&next_hidden);
        }

        // ── Final RMSNorm + LM head on the LAST position only ────────────────
        let last = batch_size - 1;
        let mut normed_last = vec![0.0f32; h];
        final_norm.forward(&hidden[last * h..(last + 1) * h], &mut normed_last)?;
        let mut logits = vec![0.0f32; lm_head_out_features];
        gemm_fp8(
            is_e4m3,
            &lm_head_bytes,
            &normed_last,
            &mut logits,
            lm_head_out_features,
            h,
            1,
        )
        .map_err(box_err)?;
        Ok(logits)
    }
}

/// Concatenate several byte slices into one owned `Vec<u8>`.
fn concat_bytes(parts: &[&[u8]]) -> Vec<u8> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

/// Map a [`MetalGraphError`] into a boxed error, logging a fallback warning.
fn box_err(e: MetalGraphError) -> Box<dyn std::error::Error> {
    tracing::warn!(error = %e, "metal FP8 batch prefill GEMM failed");
    Box::new(e) as Box<dyn std::error::Error>
}
