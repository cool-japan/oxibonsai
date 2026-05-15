//! Pre-allocated scratch buffers for a single `TransformerBlock`'s forward
//! pass.  Eliminates per-token heap allocations in the hot path.
//!
//! Visibility: `pub(super)` so the sibling forward modules can destructure
//! the struct without going through accessor methods.

pub(super) struct ScratchBuffers {
    pub(super) normed: Vec<f32>,
    pub(super) q_all: Vec<f32>,
    pub(super) k_all: Vec<f32>,
    pub(super) v_all: Vec<f32>,
    pub(super) q_normed: Vec<f32>,
    pub(super) k_normed: Vec<f32>,
    pub(super) q_rope: Vec<f32>,
    pub(super) k_rope: Vec<f32>,
    pub(super) attn_out: Vec<f32>,
    pub(super) attn_proj: Vec<f32>,
    pub(super) gate_out: Vec<f32>,
    pub(super) up_out: Vec<f32>,
    pub(super) swiglu_out: Vec<f32>,
    pub(super) down_out: Vec<f32>,
    pub(super) fused_qkv: Vec<f32>,
    pub(super) fused_gate_up: Vec<f32>,
}

impl ScratchBuffers {
    pub(super) fn new(h: usize, nq: usize, nkv: usize, hd: usize, inter: usize) -> Self {
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
    pub(super) fn clear(&mut self) {
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
