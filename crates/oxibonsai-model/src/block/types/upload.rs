//! GPU weight upload for `TransformerBlock`.

// Only `FusedKernel` is imported: `upload_weights` and
// `upload_weights_ternary` reach this file through its supertrait bounds, so
// `OneBitKernel`/`TernaryKernel` do not additionally need to be in scope.
use oxibonsai_kernels::traits::FusedKernel;

use super::block_def::TransformerBlock;

impl<'a> TransformerBlock<'a> {
    /// Upload all weight matrices in this block to GPU memory.
    ///
    /// After calling this, all GEMV operations in [`forward`](Self::forward)
    /// will use GPU-resident weight buffers, eliminating per-call
    /// host→device copies.
    ///
    /// # M-21, first half: the signature is now `&dyn FusedKernel`
    ///
    /// This parameter used to be `&dyn OneBitKernel`, and that erasure — not
    /// any missing kernel — was what pinned the fused QKV / gate-up handles
    /// to the 1-bit path. `TernaryKernel::upload_weights_ternary` (the exact
    /// mirror of `OneBitKernel::upload_weights`) has existed all along
    /// (`traits.rs:233`, `dispatch.rs`, `gpu_backend/mod.rs`); it was simply
    /// unreachable from a trait object that only promised `OneBitKernel`.
    ///
    /// `FusedKernel: OneBitKernel + TernaryKernel` plus its blanket impl
    /// (`oxibonsai_kernels::traits`) makes both halves reachable from one
    /// `dyn` reference. Every caller in the tree already passes the concrete
    /// `KernelDispatcher` (`engine.rs`, `BonsaiModel::upload_weights_to_gpu`,
    /// the Metal parity suites, `block/functions.rs`), which satisfies the
    /// blanket impl, so nothing downstream had to change.
    ///
    /// # M-21, second half: why the ternary fused handles are still not built
    ///
    /// With the signature widened, the obvious next step is an `else if let
    /// (Some(q), Some(k), Some(v)) = (self.attn_q.blocks_ternary(), ...)` arm
    /// that stores `kernel.upload_weights_ternary(&combined)` into
    /// `fused_qkv_handle`. That is **deliberately not done here**, because
    /// storing it in *that* field would be a regression rather than a win:
    /// `forward.rs:205` branches on `if let Some(fused_handle) =
    /// self.fused_qkv_handle`, and its 1-bit guard at `:282` then falls
    /// through to three separate `forward_vec` calls for a non-1-bit block —
    /// bypassing the working ternary Metal fused-QKV fast path that lives in
    /// that `if`'s `else` arm (`:303`, keyed on `blocks_ternary()` and
    /// `attn_q.gpu_handle().id()`). Populating the field would therefore turn
    /// the flagship ternary Metal decode path off.
    ///
    /// Landing the win needs `block_def.rs` to gain *separate*
    /// `fused_qkv_handle_ternary` / `fused_gate_up_handle_ternary` fields and
    /// `forward.rs` / `forward_stats.rs` / `forward_sw.rs` to consume them —
    /// all files another package owns this wave. The exact change is recorded
    /// in this package's `deviations`; the widening here is its precondition
    /// and is what the finding's ownership-blocked half asked for.
    ///
    /// Meanwhile no ternary fusion is actually lost: every ternary matrix is
    /// already GPU-resident through the unconditional per-matrix
    /// `upload_to_gpu()` calls below (which route to
    /// `TernaryKernel::upload_weights_ternary` via `layers/linear.rs`), and
    /// the fused Metal path keys on `attn_q.gpu_handle()`'s own `.id()` —
    /// a value from the same process-global monotonic counter 1-bit handles
    /// use, so it is never reused across model loads the way a freed mmap
    /// address can be.
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
