//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::{CpuBackend, GpuError};
use cudarc::driver::{CudaFunction, CudaSlice};
use std::sync::Arc;

use super::cudagraph_type::CudaGraph;

/// Process-wide serialization guard for the GPU-singleton parity tests.
///
/// Every GPU parity test drives the one process-wide [`CudaGraph`] singleton,
/// which owns a single CUDA stream. Running the tests in parallel races on that
/// stream — an async error from one test poisons the shared context for the
/// others (surfacing as spurious `alloc` / `STREAM_CAPTURE_INVALIDATED` driver
/// errors). Each GPU parity test holds the guard returned here so they run one
/// at a time regardless of the harness `--test-threads`. Production drives the
/// singleton from a single thread, so it needs no such lock.
///
/// Poison-tolerant: a panicking test (a parity assert failure) must not wedge
/// the remaining GPU tests, so a poisoned lock is recovered via `into_inner`.
#[cfg(test)]
pub(crate) fn gpu_parity_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU_PARITY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU_PARITY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// Pre-allocated GPU buffers for the LM-head GEMV.
pub(crate) struct LmHeadBuffers {
    pub(crate) d_input: CudaSlice<f32>,
    pub(crate) d_output: CudaSlice<f32>,
    pub(crate) hidden_capacity: usize,
    pub(crate) vocab_capacity: usize,
}
impl LmHeadBuffers {
    pub(crate) fn fits(&self, hidden: usize, vocab: usize) -> bool {
        self.hidden_capacity >= hidden && self.vocab_capacity >= vocab
    }
}
/// Errors from the CUDA graph dispatch engine.
#[derive(Debug)]
pub enum CudaGraphError {
    /// No CUDA-capable GPU found or driver not present.
    DeviceNotFound(String),
    /// NVRTC PTX compilation failed.
    CompilationFailed(String),
    /// CUDA driver API error.
    DriverError(String),
    /// A requested weight handle was not found in the cache.
    WeightNotFound(u64),
    /// The internal weight layout conversion failed (malformed bytes).
    WeightLayoutError(String),
    /// A mutex was poisoned.
    LockPoisoned,
}
/// Pre-allocated GPU activation buffers for a single FFN forward pass.
///
/// `d_scratch` is reused for both the attn_proj GEMV output (size h) and the
/// down GEMV output (size h) — they are never needed simultaneously.  This
/// saves one GPU buffer vs. keeping separate `d_proj` and `d_down`.
///
/// Resized lazily when `hidden_size` or `intermediate_size` changes.
#[allow(dead_code)]
pub(crate) struct CudaActivationBuffers {
    pub(crate) d_hidden: CudaSlice<f32>,
    pub(crate) d_attn_out: CudaSlice<f32>,
    pub(crate) d_norm_weight: CudaSlice<f32>,
    pub(crate) d_scratch: CudaSlice<f32>,
    pub(crate) d_normed: CudaSlice<f32>,
    /// Intermediate gate+up GEMV output (2 × inter). Retained as a pre-allocated fallback
    /// buffer; not used in the fused `fused_gate_up_swiglu_q1` path.
    #[allow(dead_code)]
    pub(crate) d_gate_up: CudaSlice<f32>,
    pub(crate) d_swiglu: CudaSlice<f32>,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
}
impl CudaActivationBuffers {
    pub(crate) fn matches(&self, h: usize, inter: usize) -> bool {
        self.hidden_size == h && self.intermediate_size == inter
    }
}
/// Pre-allocated GPU buffers for the QKV projection.
///
/// Eliminates per-call `cuMemAlloc`/`cuMemFree` in `encode_qkv_phase`.
pub(crate) struct QkvBuffers {
    pub(crate) d_input: CudaSlice<f32>,
    pub(crate) d_output: CudaSlice<f32>,
    pub(crate) input_capacity: usize,
    pub(crate) output_capacity: usize,
}
impl QkvBuffers {
    pub(crate) fn fits(&self, input_len: usize, output_len: usize) -> bool {
        self.input_capacity >= input_len && self.output_capacity >= output_len
    }
}
/// Thin `GpuBackendTrait` wrapper around `CudaGraph`.
///
/// Returned by [`select_backend`](super::super::select_backend) when a CUDA device
/// is present and the `native-cuda` feature is enabled.
pub struct NativeCudaBackend {
    pub(super) graph: Arc<CudaGraph>,
    pub(super) cpu_fallback: CpuBackend,
}
impl NativeCudaBackend {
    /// Initialise the backend (may fail if no CUDA device).
    pub fn new() -> Result<Self, GpuError> {
        let graph = CudaGraph::global().map_err(|e| GpuError::NotAvailable(e.to_string()))?;
        Ok(Self {
            graph,
            cpu_fallback: CpuBackend::new(),
        })
    }
}
/// Handles to all compiled CUDA kernel functions used by `CudaGraph`.
#[allow(dead_code)]
pub(crate) struct CudaModules {
    pub(crate) gemv_q1_g128_v7: CudaFunction,
    pub(crate) gemv_q1_g128_v7_residual: CudaFunction,
    pub(crate) gemv_q1_g128_v8: CudaFunction,
    pub(crate) gemv_q1_g128_v8_residual: CudaFunction,
    pub(crate) gemv_q1_g128_v9: CudaFunction,
    pub(crate) gemv_q1_g128_v9_residual: CudaFunction,
    pub(crate) rmsnorm_weighted_v2: CudaFunction,
    pub(crate) residual_add: CudaFunction,
    pub(crate) swiglu_fused: CudaFunction,
    /// Fused gate+up Q1 GEMV with SwiGLU epilogue — halves dispatch count for FFN step 5+6.
    pub(crate) fused_gate_up_swiglu: CudaFunction,
    pub(crate) argmax_f32: CudaFunction,
    /// Ternary (TQ2_0_g128) GEMV — SoA weight layout, 8 rows per CTA.
    pub(crate) gemv_tq2_g128_v1: CudaFunction,
    // ── Image-generation (FLUX.2 DiT/VAE) prototype kernels ──
    /// Dense f32 GEMM for the DiT linear projections / VAE matmuls.
    pub(crate) gemm_f32: CudaFunction,
    /// Ternary (TQ2_0_g128) GEMM (large-M) for the DiT ternary linears.
    pub(crate) gemm_tq2: CudaFunction,
    /// DiT joint (txt+img) flash-attention, scalar-FP32 online softmax
    /// (FA_DMAX=128 build: 32 KiB shared, no >48 KiB opt-in → full L1).
    pub(crate) joint_attention_flash_f32: CudaFunction,
    /// Wide variant of [`Self::joint_attention_flash_f32`] compiled with
    /// FA_DMAX=384 for the VAE mid-block self-attention (head_dim up to 384,
    /// 96 KiB dynamic shared). Separate build so the small DiT launches are not
    /// penalised by the large shared-mem opt-in.
    pub(crate) joint_attention_flash_f32_large: CudaFunction,
    /// VAE im2col expansion for the implicit-GEMM convolutions.
    pub(crate) imagen_vae_im2col: CudaFunction,
    /// VAE group normalization.
    pub(crate) imagen_vae_groupnorm: CudaFunction,
    /// VAE SiLU activation.
    pub(crate) imagen_vae_silu: CudaFunction,
    /// VAE nearest-neighbour upsample.
    pub(crate) imagen_vae_upsample_nearest: CudaFunction,
    // ── DiT per-block glue kernels (for the resident DiT forward) ──
    /// `y = (1 + scale) * x + shift`.
    pub(crate) dit_modulate: CudaFunction,
    /// `h += gate * delta`.
    pub(crate) dit_gated_residual_add: CudaFunction,
    /// LayerNorm (affine = false), one block per row.
    pub(crate) dit_layer_norm: CudaFunction,
    /// Per-head QK-RMSNorm, one block per `head_dim` chunk.
    pub(crate) dit_rms_norm_heads: CudaFunction,
    /// SwiGLU `silu(gate) * up`.
    pub(crate) dit_swiglu: CudaFunction,
    /// Interleaved (adjacent-pair) RoPE, in place.
    pub(crate) dit_rope_interleaved: CudaFunction,
    /// Reshape: gather a token-major slice into head-major `[heads, seq, head_dim]`.
    pub(crate) dit_tokens_to_heads: CudaFunction,
    /// Reshape: strided per-row slice copy (mlp-extract / attn‖gated concat).
    pub(crate) dit_strided_row_copy: CudaFunction,
}
/// Reusable input/output device buffers for `encode_gemv_tq2_cached`.
///
/// Grows monotonically to fit the largest GEMV seen so far. Eliminates the
/// per-call `cuMemAlloc`/`cuMemFree` round-trip that otherwise dominates
/// dispatch overhead for short kernels.
pub(crate) struct TernaryGemvBuffers {
    pub(crate) d_input: CudaSlice<f32>,
    pub(crate) d_output: CudaSlice<f32>,
    pub(crate) input_capacity: usize,
    pub(crate) output_capacity: usize,
}
impl TernaryGemvBuffers {
    pub(crate) fn fits(&self, input_len: usize, output_len: usize) -> bool {
        self.input_capacity >= input_len && self.output_capacity >= output_len
    }
}
