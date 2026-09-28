//! Public weight / handle / block accessors for `TransformerBlock`.
//!
//! Methods are grouped by quant family (separated by comment dividers).  All
//! accessors are `pub fn` — they form part of the crate's public API via
//! `block/mod.rs:23`'s `pub use types::*;`.

use oxibonsai_kernels::GpuWeightHandle;

use super::block_def::TransformerBlock;

impl<'a> TransformerBlock<'a> {
    // ── Norm weights & layer index ───────────────────────────────────────────

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

    // ── GPU weight handles ───────────────────────────────────────────────────

    /// Fused QKV GPU handle (if uploaded) — the **1-bit** one.
    ///
    /// Every caller decodes this handle's buffer as `Q1_0_g128` (the Q1 GPU
    /// weight cache, the Q1 fused-forward and CUDA paths), so it is gated on
    /// the block's format (`M-21`): a ternary block always answers `None`
    /// here, even if a handle were ever stored in the 1-bit field, and exposes
    /// its own through [`Self::fused_qkv_gpu_handle_ternary`].
    pub fn fused_qkv_gpu_handle(&self) -> Option<GpuWeightHandle> {
        if self.attn_q.blocks_ternary().is_some() {
            return None;
        }
        self.fused_qkv_handle
    }
    /// Attention output projection GPU handle (if uploaded).
    pub fn attn_output_gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.attn_output.gpu_handle()
    }
    /// Fused gate+up GPU handle (if uploaded) — the **1-bit** one, gated on
    /// the block's format exactly like [`Self::fused_qkv_gpu_handle`]
    /// (`M-21`); see [`Self::fused_gate_up_gpu_handle_ternary`] for ternary
    /// blocks.
    pub fn fused_gate_up_gpu_handle(&self) -> Option<GpuWeightHandle> {
        if self.ffn_gate.blocks_ternary().is_some() {
            return None;
        }
        self.fused_gate_up_handle
    }
    /// Fused Q‖K‖V GPU handle of a **ternary** block (`M-21`), if
    /// `upload_to_gpu` ran on a GPU-tier kernel.
    pub fn fused_qkv_gpu_handle_ternary(&self) -> Option<GpuWeightHandle> {
        self.fused_qkv_handle_ternary
    }
    /// Fused gate‖up GPU handle of a **ternary** block (`M-21`), if
    /// `upload_to_gpu` ran on a GPU-tier kernel.
    pub fn fused_gate_up_gpu_handle_ternary(&self) -> Option<GpuWeightHandle> {
        self.fused_gate_up_handle_ternary
    }
    /// GPU weight-cache slot of a ternary block's fused Q‖K‖V buffer
    /// (`M-21`): the dedicated fused handle's id when `upload_to_gpu` built
    /// one, else the Q projection's handle id (the slot the fused arm keyed
    /// on before the dedicated handle existed); `None` when the block was
    /// never uploaded, i.e. the fused arm must not engage.
    ///
    /// Every forward path that runs the ternary fused-QKV GEMV must derive its
    /// slot here so they all bind **one** GPU buffer — `forward` does;
    /// `forward_with_sliding_window` / `forward_with_stats` still key on
    /// [`Self::legacy_ternary_fused_qkv_slot`] until their owner switches them
    /// (a recorded one-line change each).
    pub fn ternary_fused_qkv_slot(&self) -> Option<u64> {
        self.fused_qkv_handle_ternary
            .or_else(|| self.attn_q.gpu_handle())
            .map(|handle| handle.id())
    }
    /// The Q projection's handle id — the fused-QKV slot every ternary Metal
    /// forward path keyed on before `M-21` gave the concatenation its own
    /// handle. Kept so the paths not yet switched to
    /// [`Self::ternary_fused_qkv_slot`] have a named source for it.
    pub fn legacy_ternary_fused_qkv_slot(&self) -> Option<u64> {
        self.attn_q.gpu_handle().map(|handle| handle.id())
    }
    /// GPU weight-cache slot of a ternary block's fused gate‖up buffer
    /// (`M-21`): the dedicated fused handle's id when present, else the gate
    /// projection's handle id; `None` when the block was never uploaded.
    pub fn ternary_fused_gate_up_slot(&self) -> Option<u64> {
        self.fused_gate_up_handle_ternary
            .or_else(|| self.ffn_gate.gpu_handle())
            .map(|handle| handle.id())
    }
    /// FFN down projection GPU handle (if uploaded).
    pub fn ffn_down_gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.ffn_down.gpu_handle()
    }

    // ── Q1 (1-bit, BlockQ1_0G128) block accessors ────────────────────────────

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

    // ── Ternary (TQ2, BlockTQ2_0_g128) block accessors ───────────────────────

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

    // ── FP8 E4M3 (BlockFP8E4M3) block accessors ──────────────────────────────

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

    // ── FP8 E5M2 (BlockFP8E5M2) block accessors ──────────────────────────────

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

    // ── Q4_0 / Q8_0 (standard-quant) block accessors ─────────────────────────

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
