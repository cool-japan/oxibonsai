//! Q4_0 / Q8_0 standard-quant CUDA batch-prefill helpers and entry points.
//!
//! This family's batch prefill drives a GPU-private device KV cache while its
//! decode attends over the host `self.kv_cache`, so every entry point here
//! reads the prompt's device K/V back and writes it into the host cache
//! (finding **F6**) — and refuses a call at `pos_start > 0`, whose earlier
//! positions the device cache does not hold
//! ([`super::cuda_split_prefill_allowed_with_readback`]).
//!
//! Run on hardware (RTX A4000, CUDA 12.0, 2026-10-07): CUDA-P11 passed on
//! real Q4_0 / Q8_0 weights (`crates/oxibonsai-model/tests/cuda_p11_q_std_kv_readback.rs`).

use oxibonsai_kernels::gpu_backend::cuda_full_layer::KvReadback;

use super::super::q1_slots::SlotNamespace;
use super::super::{BonsaiModel, OutputWeight};
use super::byte_helpers::{blocks_q4_0_as_bytes, blocks_q8_0_as_bytes};

/// Process-wide number of Q4_0/Q8_0 CUDA batch prefills that completed on the
/// device and stored their K/V read-back into the host cache (finding **F6**);
/// read through [`BonsaiModel::cuda_q_std_prefill_count`].
static CUDA_Q_STD_PREFILLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl BonsaiModel<'_> {
    /// Process-wide number of Q4_0/Q8_0 CUDA batch prefills — the
    /// `forward_prefill` entry point, not the verify one — that ran on the
    /// device **and** stored their K/V read-back into the host cache (finding
    /// **F6**). Monotonic, never reset, and counted over every model in the
    /// process.
    ///
    /// Observability only: nothing reads it to make a decision. A silent
    /// fallback of `forward_prefill` to a host-KV path leaves the same host
    /// K/V behind as the batch prefill does, so a caller that must know which
    /// path answered (the CUDA-P11 hardware harness) compares this count
    /// around the call.
    pub fn cuda_q_std_prefill_count() -> u64 {
        CUDA_Q_STD_PREFILLS.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl<'a> BonsaiModel<'a> {
    /// Convert a Q4_0 block slice to raw bytes (zero-copy, file-level fn below).
    ///
    /// GPU batch prefill for Q4_0/Q8_0 models (CUDA): all layers + final norm + LM head.
    ///
    /// Returns the last token's logits.  Uses `try_cuda_prefill_q_std` (Phase 24A).
    pub(super) fn build_cuda_q_std_qkv_concats(
        &self,
        q4_0: bool,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let n_layers = self.blocks.len();
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let (q_bytes, k_bytes, v_bytes) = if q4_0 {
                (
                    blocks_q4_0_as_bytes(block.attn_q_blocks_q4_0().ok_or("attn_q: not Q4_0")?),
                    blocks_q4_0_as_bytes(block.attn_k_blocks_q4_0().ok_or("attn_k: not Q4_0")?),
                    blocks_q4_0_as_bytes(block.attn_v_blocks_q4_0().ok_or("attn_v: not Q4_0")?),
                )
            } else {
                (
                    blocks_q8_0_as_bytes(block.attn_q_blocks_q8_0().ok_or("attn_q: not Q8_0")?),
                    blocks_q8_0_as_bytes(block.attn_k_blocks_q8_0().ok_or("attn_k: not Q8_0")?),
                    blocks_q8_0_as_bytes(block.attn_v_blocks_q8_0().ok_or("attn_v: not Q8_0")?),
                )
            };
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        Ok(qkv_concats)
    }

    /// This model's CUDA Q4_0/Q8_0 slot namespace: the composition over its
    /// `cuda_model_epoch`, shared with the Q1 and ternary namespaces (see
    /// [`BonsaiModel::cuda_q1_slots`](super::super::q1::BonsaiModel::cuda_q1_slots)).
    fn cuda_q_std_slots(&self) -> SlotNamespace {
        SlotNamespace::new(self.cuda_model_epoch)
    }

    /// Build per-layer `CudaQStdPrefillLayerParams` for the Q4_0/Q8_0 CUDA path.
    ///
    /// Handle namespaces: every norm / weight handle is composed over this
    /// model's `cuda_model_epoch` via [`SlotNamespace::q_std_norm_base`] /
    /// [`SlotNamespace::q_std_weight_base`], distinct per format (`q4_0`
    /// selects Q4_0 vs Q8_0's range) and from
    /// every other family's and every other model's namespace.
    ///
    /// Per-layer offsets: +0=attn_norm, +1=q_norm, +2=k_norm, +3=ffn_norm
    ///                    +0=fused_qkv, +1=attn_proj, +2=gate_up, +3=down
    pub(super) fn build_cuda_q_std_layer_params<'b>(
        &'b self,
        qkv_concats: &'b [Vec<u8>],
        q4_0: bool,
    ) -> Result<Vec<oxibonsai_kernels::CudaQStdPrefillLayerParams<'b>>, Box<dyn std::error::Error>>
    {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        SlotNamespace::check_layer_count(n_layers)?;
        let slots = self.cuda_q_std_slots();
        let mut layer_params: Vec<oxibonsai_kernels::CudaQStdPrefillLayerParams<'b>> =
            Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = slots.q_std_norm_base(block.layer_index(), q4_0);
            let weight_handle_base = slots.q_std_weight_base(block.layer_index(), q4_0);
            let (attn_proj_bytes, gate_bytes, up_bytes, down_bytes) = if q4_0 {
                (
                    blocks_q4_0_as_bytes(
                        block
                            .attn_output_blocks_q4_0()
                            .ok_or("attn_output: not Q4_0")?,
                    ),
                    blocks_q4_0_as_bytes(block.ffn_gate_blocks_q4_0().ok_or("ffn_gate: not Q4_0")?),
                    blocks_q4_0_as_bytes(block.ffn_up_blocks_q4_0().ok_or("ffn_up: not Q4_0")?),
                    blocks_q4_0_as_bytes(block.ffn_down_blocks_q4_0().ok_or("ffn_down: not Q4_0")?),
                )
            } else {
                (
                    blocks_q8_0_as_bytes(
                        block
                            .attn_output_blocks_q8_0()
                            .ok_or("attn_output: not Q8_0")?,
                    ),
                    blocks_q8_0_as_bytes(block.ffn_gate_blocks_q8_0().ok_or("ffn_gate: not Q8_0")?),
                    blocks_q8_0_as_bytes(block.ffn_up_blocks_q8_0().ok_or("ffn_up: not Q8_0")?),
                    blocks_q8_0_as_bytes(block.ffn_down_blocks_q8_0().ok_or("ffn_down: not Q8_0")?),
                )
            };
            layer_params.push(oxibonsai_kernels::CudaQStdPrefillLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: weight_handle_base,
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: weight_handle_base + 1,
                attn_proj_bytes,
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: weight_handle_base + 2,
                gate_bytes,
                up_bytes,
                down_handle: weight_handle_base + 3,
                down_bytes,
                q4_0,
            });
        }
        Ok(layer_params)
    }

    /// GPU batch prefill for Q4_0/Q8_0 models (CUDA): all layers, the final
    /// norm and the LM head, with the prompt's K/V written into
    /// `self.kv_cache` (finding **F6**).
    ///
    /// Returns the last token's logits.  Dispatches to `try_cuda_prefill_q_std` (Phase 24A).
    ///
    /// # Errors
    /// A `pos_start > 0` call (refused by construction, see the module doc),
    /// any dispatch failure, or a failed host-cache store; the caller falls
    /// back to the sequential path on every one of them.
    pub(in super::super) fn try_cuda_prefill_with_lm_head_q_std(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        q4_0: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // F6 SPLIT-CACHE GUARD: see `cuda_split_prefill_allowed_with_readback`
        // for why only `pos_start == 0` is grounded, and
        // `cuda_split_prefill_disabled` for why the
        // `OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL` override no longer overrides.
        if !super::cuda_split_prefill_allowed_with_readback(pos_start) {
            return Err(super::cuda_split_prefill_needs_history(
                "Q4_0/Q8_0",
                false,
                pos_start,
            ));
        }
        let (logits, readback) = self.run_cuda_prefill_q_std(token_ids, pos_start, q4_0)?;
        self.store_cuda_kv_readback(&readback, pos_start, token_ids.len())?;
        CUDA_Q_STD_PREFILLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(logits)
    }

    /// The device half of [`Self::try_cuda_prefill_with_lm_head_q_std`]:
    /// runs the batch prefill and returns the last token's logits plus the
    /// prompt window's K/V read-back, leaving `self` untouched.
    fn run_cuda_prefill_q_std(
        &self,
        token_ids: &[u32],
        pos_start: usize,
        q4_0: bool,
    ) -> Result<(Vec<f32>, KvReadback), Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        if pos_start + batch_size > self.kv_cache.max_seq_len() {
            return Err(format!(
                "prefill sequence too long: {batch_size} tokens at pos {pos_start} exceeds \
                 max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let half_dim = hd / 2;
        let heads_per_group = nq.checked_div(nkv).unwrap_or(1);
        let max_seq_len = self.kv_cache.max_seq_len();

        // Build flattened hidden_batch from token embeddings
        let mut hidden_batch = vec![0.0f32; batch_size * h];
        // M-02: gather only the rows this batch references, decoding each
        // straight out of the quantized table. The deleted dense
        // `Index<Range<usize>>` hatch materialized the whole
        // `vocab x hidden` FP32 table here (1.16 / 2.31 / 4.74 GiB for the
        // 1.7B / 8B / 27B) on the first multi-token prompt.
        self.token_embd.copy_rows(token_ids, &mut hidden_batch)?;

        // Build RoPE tables for the full batch
        let mut cos_table = vec![0.0f32; batch_size * half_dim];
        let mut sin_table = vec![0.0f32; batch_size * half_dim];
        for t in 0..batch_size {
            let pos = pos_start + t;
            let cos_vals = self.rope.cos_at_checked(pos)?;
            let sin_vals = self.rope.sin_at_checked(pos)?;
            cos_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(cos_vals);
            sin_table[t * half_dim..(t + 1) * half_dim].copy_from_slice(sin_vals);
        }

        // Handle namespaces for final norm + LM head
        let final_norm_handle = self.cuda_q_std_slots().q_std_final_norm(q4_0);
        let lm_head_handle = self.cuda_q_std_slots().q_std_lm_head(q4_0);
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();

        let (lm_head_bytes, lm_head_out_features) = match &self.output_weight {
            OutputWeight::Q4_0(ref linear) if q4_0 => {
                (blocks_q4_0_as_bytes(linear.blocks()), linear.out_features())
            }
            OutputWeight::Q8_0(ref linear) if !q4_0 => {
                (blocks_q8_0_as_bytes(linear.blocks()), linear.out_features())
            }
            _ => {
                return Err(format!(
                    "try_cuda_prefill_with_lm_head_q_std: LM head quant mismatch (q4_0={})",
                    q4_0
                )
                .into())
            }
        };

        let qkv_concats = self.build_cuda_q_std_qkv_concats(q4_0)?;
        let layer_params = self.build_cuda_q_std_layer_params(&qkv_concats, q4_0)?;

        let mut logits = vec![0.0f32; lm_head_out_features];
        let mut readback = KvReadback::new();
        oxibonsai_kernels::try_cuda_prefill_q_std(
            &hidden_batch,
            batch_size,
            pos_start,
            n_layers,
            &layer_params,
            &cos_table,
            &sin_table,
            h,
            inter,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            max_seq_len,
            Some(final_norm_handle),
            Some(final_norm_bytes),
            final_norm_eps,
            Some(lm_head_handle),
            Some(lm_head_bytes),
            lm_head_out_features,
            q4_0,
            Some(&mut logits),
            None,
            Some(&mut readback),
        )
        .map_err(|e| {
            tracing::warn!(error = %e, "CUDA Q4_0/Q8_0 batch prefill dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok((logits, readback))
    }

    /// GPU batch prefill verify for Q4_0/Q8_0 models (CUDA): greedy argmax
    /// per position, with every verified position's K/V written into
    /// `self.kv_cache` (finding **F6**).
    ///
    /// Returns the greedy argmax token ID for each input position.
    /// Dispatches to `try_cuda_prefill_q_std` with `greedy_token_id_out` set (Phase 24A).
    ///
    /// # Errors
    /// As [`Self::try_cuda_prefill_with_lm_head_q_std`].
    pub(in super::super) fn try_cuda_prefill_verify_q_std(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        q4_0: bool,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        // F6 SPLIT-CACHE GUARD: as in `try_cuda_prefill_with_lm_head_q_std`.
        // The per-token calls below each read positions `[0, pos]`, all of
        // which this loop itself wrote when it starts at 0.
        if !super::cuda_split_prefill_allowed_with_readback(pos_start) {
            return Err(super::cuda_split_prefill_needs_history(
                "Q4_0/Q8_0",
                true,
                pos_start,
            ));
        }
        let (token_ids_out, readbacks) =
            self.run_cuda_prefill_verify_q_std(token_ids, pos_start, q4_0)?;
        // One single-position read-back per verified token, stored in order.
        for (t, readback) in readbacks.iter().enumerate() {
            self.store_cuda_kv_readback(readback, pos_start + t, 1)?;
        }
        Ok(token_ids_out)
    }

    /// The device half of [`Self::try_cuda_prefill_verify_q_std`]: one
    /// batch-1 prefill per token, returning the argmax ids plus each
    /// token's single-position K/V read-back, leaving `self` untouched.
    fn run_cuda_prefill_verify_q_std(
        &self,
        token_ids: &[u32],
        pos_start: usize,
        q4_0: bool,
    ) -> Result<(Vec<u32>, Vec<KvReadback>), Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        if pos_start + batch_size > self.kv_cache.max_seq_len() {
            return Err(format!(
                "prefill-verify sequence too long: {batch_size} tokens at pos {pos_start} \
                 exceeds max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let heads_per_group = nq.checked_div(nkv).unwrap_or(1);
        let max_seq_len = self.kv_cache.max_seq_len();

        let final_norm_handle = self.cuda_q_std_slots().q_std_final_norm(q4_0);
        let lm_head_handle = self.cuda_q_std_slots().q_std_lm_head(q4_0);
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();

        let (lm_head_bytes, lm_head_out_features) = match &self.output_weight {
            OutputWeight::Q4_0(ref linear) if q4_0 => {
                (blocks_q4_0_as_bytes(linear.blocks()), linear.out_features())
            }
            OutputWeight::Q8_0(ref linear) if !q4_0 => {
                (blocks_q8_0_as_bytes(linear.blocks()), linear.out_features())
            }
            _ => {
                return Err(format!(
                    "try_cuda_prefill_verify_q_std: LM head quant mismatch (q4_0={})",
                    q4_0
                )
                .into())
            }
        };

        let qkv_concats = self.build_cuda_q_std_qkv_concats(q4_0)?;
        let layer_params = self.build_cuda_q_std_layer_params(&qkv_concats, q4_0)?;

        let mut token_ids_out: Vec<u32> = Vec::with_capacity(batch_size);
        let mut readbacks: Vec<KvReadback> = Vec::with_capacity(batch_size);
        for (t, &tok_id) in token_ids.iter().enumerate() {
            // M-02: one row, decoded from the quantized table.
            let mut single_hidden = vec![0.0f32; h];
            self.token_embd.copy_row(tok_id, &mut single_hidden)?;
            let pos = pos_start + t;
            let cos_single = self.rope.cos_at_checked(pos)?;
            let sin_single = self.rope.sin_at_checked(pos)?;

            let mut greedy_id: u32 = 0;
            let mut readback = KvReadback::new();
            oxibonsai_kernels::try_cuda_prefill_q_std(
                &single_hidden,
                1,
                pos,
                n_layers,
                &layer_params,
                cos_single,
                sin_single,
                h,
                inter,
                nq,
                nkv,
                hd,
                heads_per_group,
                eps,
                max_seq_len,
                Some(final_norm_handle),
                Some(final_norm_bytes),
                final_norm_eps,
                Some(lm_head_handle),
                Some(lm_head_bytes),
                lm_head_out_features,
                q4_0,
                None,
                Some(&mut greedy_id),
                Some(&mut readback),
            )
            .map_err(|e| {
                tracing::warn!(
                    error = %e,
                    "CUDA Q4_0/Q8_0 prefill verify dispatch failed at pos {pos}"
                );
                Box::new(e) as Box<dyn std::error::Error>
            })?;
            token_ids_out.push(greedy_id);
            readbacks.push(readback);
        }
        Ok((token_ids_out, readbacks))
    }
}
