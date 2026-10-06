//! GPU weight upload for `TransformerBlock`.

// Only `FusedKernel` is imported: `upload_weights` and
// `upload_weights_ternary` reach this file through its supertrait bounds, so
// `OneBitKernel`/`TernaryKernel` do not additionally need to be in scope.
use oxibonsai_kernels::traits::FusedKernel;

use super::block_def::TransformerBlock;

/// Whether this build uploads a **ternary** block's Q‖K‖V and gate‖up
/// concatenations to the kernel's own weight cache (`M-21`).
///
/// Every GPU build does except Metal. There, the fused ternary arms of the
/// block forwards run on `MetalGraph`, whose weight cache is a *separate*
/// store: they key their own copy of each concatenation on the block's
/// mapping namespace (the mapped tensors' addresses under the model's mapping
/// epoch — the very slots the model's full-forward ternary cache uses), and
/// nothing on Metal ever reads the kernel-side copy. Uploading it anyway cost
/// ~8.9 MB per layer (≈250 MB on Ternary-Bonsai-1.7B) of never-read GPU
/// memory for every Metal caller of `BonsaiModel::upload_weights_to_gpu`
/// (`MET-M1` / the `M-21` residue). On CUDA the concatenation this uploads
/// **is** the buffer the fused GEMVs bind, so it stays byte-for-byte as it
/// was.
const UPLOAD_TERNARY_CONCATENATIONS: bool = !cfg!(all(feature = "metal", target_os = "macos"));

impl<'a> TransformerBlock<'a> {
    /// Upload all weight matrices in this block to GPU memory.
    ///
    /// After calling this, all GEMV operations in [`forward`](Self::forward)
    /// will use GPU-resident weight buffers, eliminating per-call
    /// host→device copies.
    ///
    /// # Fused handles, per format (`M-21`)
    ///
    /// Besides the seven per-matrix uploads, the two row-wise concatenations
    /// the fused dispatches consume are uploaded once:
    ///
    /// - **1-bit** blocks: Q‖K‖V into `fused_qkv_handle` and gate‖up into
    ///   `fused_gate_up_handle` via `OneBitKernel::upload_weights`. Every
    ///   consumer of those two fields decodes `Q1_0_g128`.
    /// - **ternary** blocks, every GPU build but Metal: the same two
    ///   concatenations into the separate `fused_qkv_handle_ternary` /
    ///   `fused_gate_up_handle_ternary` fields via
    ///   `TernaryKernel::upload_weights_ternary` — reachable here because this
    ///   takes a `&dyn FusedKernel` (`FusedKernel: OneBitKernel +
    ///   TernaryKernel`). On CUDA the concatenation stored is the very buffer
    ///   the fused ternary GEMVs bind.
    /// - **ternary** blocks on **Metal**: nothing beyond the seven per-matrix
    ///   uploads (see `UPLOAD_TERNARY_CONCATENATIONS`). The fused arms build
    ///   each concatenation once in `MetalGraph`'s own cache, keyed on the
    ///   block's mapping namespace (`TransformerBlock::ternary_fused_qkv_slot`
    ///   / `ternary_fused_gate_up_slot`), which is the same buffer the model's
    ///   full-forward ternary cache holds — so running both costs one copy,
    ///   and every replica of the GGUF mapping shares it.
    ///
    /// The ternary handles are deliberately **not** stored in the 1-bit
    /// fields: `forward` branches on `fused_qkv_handle` first, and a ternary
    /// handle there would divert the block into the 1-bit fused branch —
    /// whose `Q1_0_g128` guard then falls back to three separate projections —
    /// and away from the working ternary fused-QKV arm.
    ///
    /// A CPU-tier kernel returns `None` from every upload call, so a CPU-only
    /// block keeps every fused field `None` and runs the per-matrix CPU path
    /// exactly as before; nothing here is a CPU-tier behaviour change.
    ///
    /// Cost: like the 1-bit path always has, a GPU-tier ternary block on a
    /// non-Metal build holds its Q‖K‖V and gate‖up bytes once more in the
    /// kernel's weight cache (8.9 MB per layer on Ternary-Bonsai-1.7B: a
    /// 4096-row Q‖K‖V and a 12 288-row gate‖up at `k = 2048`, 34 bytes per
    /// 128 weights). The engine never calls this on the fused ternary Metal
    /// route (`FusedMetalRoute::gpu_weight_upload_redundant`).
    pub fn upload_to_gpu(&mut self, kernel: &dyn FusedKernel) {
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
        } else if UPLOAD_TERNARY_CONCATENATIONS {
            if let (Some(q_blk), Some(k_blk), Some(v_blk)) = (
                self.attn_q.blocks_ternary(),
                self.attn_k.blocks_ternary(),
                self.attn_v.blocks_ternary(),
            ) {
                let mut qkv_blocks = Vec::with_capacity(q_blk.len() + k_blk.len() + v_blk.len());
                qkv_blocks.extend_from_slice(q_blk);
                qkv_blocks.extend_from_slice(k_blk);
                qkv_blocks.extend_from_slice(v_blk);
                self.fused_qkv_handle_ternary = kernel.upload_weights_ternary(&qkv_blocks);
            }
        }
        if let (Some(gate_blk), Some(up_blk)) =
            (self.ffn_gate.blocks_1bit(), self.ffn_up.blocks_1bit())
        {
            let mut gate_up_blocks = Vec::with_capacity(gate_blk.len() + up_blk.len());
            gate_up_blocks.extend_from_slice(gate_blk);
            gate_up_blocks.extend_from_slice(up_blk);
            self.fused_gate_up_handle = kernel.upload_weights(&gate_up_blocks);
        } else if UPLOAD_TERNARY_CONCATENATIONS {
            if let (Some(gate_blk), Some(up_blk)) =
                (self.ffn_gate.blocks_ternary(), self.ffn_up.blocks_ternary())
            {
                let mut gate_up_blocks = Vec::with_capacity(gate_blk.len() + up_blk.len());
                gate_up_blocks.extend_from_slice(gate_blk);
                gate_up_blocks.extend_from_slice(up_blk);
                self.fused_gate_up_handle_ternary = kernel.upload_weights_ternary(&gate_up_blocks);
            }
        }
    }
}
