//! GPU weight upload for `TransformerBlock`.

use oxibonsai_kernels::traits::OneBitKernel;

use super::block_def::TransformerBlock;

impl<'a> TransformerBlock<'a> {
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
}
