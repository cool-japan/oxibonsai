//! GPU kernel source strings for Metal (MSL) and CUDA backends.
//!
//! Each kernel is provided as a compile-time string constant that is
//! compiled at runtime via scirs2-core's `GpuCompiler`.
//!
//! # Module organisation
//!
//! | Module       | Contents |
//! |--------------|----------|
//! | `decode`     | V7 GEMV + residual + fused gate/up/SwiGLU (single-token)     |
//! | `prefill`    | V7 GEMM + residual + fused GEMM SwiGLU + batched ops |
//! | `attention`  | Fused QK-norm/RoPE/KV + batched attention score/softmax/sum |
//! | `utility`    | RMSNorm, SwiGLU, softmax, ReLU, SiLU, residual-add, argmax |
//! | `archive`    | Historical V1-V6, V8-V10 kernels and CUDA stubs |
//!
//! # scirs2-core buffer naming convention
//!
//! scirs2-core's Metal backend binds buffers in a fixed order based on
//! their *names*.  The standard buffer order is:
//!
//! | Name       | Metal buffer index (if set) |
//! |------------|----------------------------|
//! | `"x"`      | next index                 |
//! | `"y"`      | next index                 |
//! | `"a"`      | next index                 |
//! | `"b"`      | next index                 |
//! | `"result"` | next index                 |
//! | `"output"` | next index                 |
//!
//! Only buffers that are actually set contribute an index.
//!
//! Scalar parameters are appended after all buffers, in order:
//! `"alpha"`, `"beta"`, `"n"`, `"m"`, `"k"`.
//!
//! All MSL kernels in this module use `[[buffer(N)]]` annotations that
//! match this assignment.

mod archive;
mod attention;
mod decode;
mod decode_ternary;
mod prefill;
mod utility;

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub use archive::*;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use attention::*;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use decode::*;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use decode_ternary::*;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use prefill::*;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use utility::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    #[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
    use super::*;

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn metal_kernels_contain_entry_points() {
        assert!(MSL_GEMV_Q1_G128.contains("kernel void gemv_q1_g128"));
        assert!(MSL_GEMV_Q1_G128_V2.contains("kernel void gemv_q1_g128_v2"));
        assert!(MSL_GEMV_Q1_G128_V3.contains("kernel void gemv_q1_g128_v3"));
        assert!(MSL_GEMM_Q1_G128.contains("kernel void gemm_q1_g128"));
        assert!(MSL_SOFTMAX.contains("kernel void softmax"));
        assert!(MSL_RELU.contains("kernel void relu"));
        assert!(MSL_RMSNORM.contains("kernel void rmsnorm"));
        assert!(MSL_SILU.contains("kernel void silu"));
        assert!(MSL_MATVEC_F32.contains("kernel void matvec_f32"));
        assert!(MSL_SWIGLU.contains("kernel void swiglu"));
        assert!(MSL_SWIGLU_FUSED.contains("kernel void swiglu_fused"));
        assert!(MSL_RESIDUAL_ADD.contains("kernel void residual_add"));
        assert!(MSL_RMSNORM_WEIGHTED.contains("kernel void rmsnorm_weighted"));
        assert!(MSL_RMSNORM_WEIGHTED_V2.contains("kernel void rmsnorm_weighted_v2"));
        // Batched attention kernels
        assert!(MSL_BATCHED_RMSNORM_V2.contains("kernel void batched_rmsnorm_v2"));
        assert!(MSL_BATCHED_ATTENTION_SCORES.contains("kernel void batched_attention_scores"));
        assert!(MSL_BATCHED_ATTENTION_SCORES_V2.contains("kernel void batched_attention_scores_v2"));
        assert!(MSL_BATCHED_SOFTMAX.contains("kernel void batched_softmax"));
        assert!(MSL_BATCHED_ATTENTION_WEIGHTED_SUM
            .contains("kernel void batched_attention_weighted_sum"));
        // Fused kernels
        assert!(MSL_FUSED_QK_NORM.contains("kernel void fused_qk_norm"));
        assert!(MSL_FUSED_QK_ROPE.contains("kernel void fused_qk_rope"));
        assert!(MSL_FUSED_KV_STORE.contains("kernel void fused_kv_store"));
        assert!(MSL_GEMV_Q1_G128_RESIDUAL.contains("kernel void gemv_q1_g128_residual"));
        assert!(MSL_GEMV_Q1_G128_V7.contains("kernel void gemv_q1_g128_v7"));
        assert!(MSL_GEMV_Q1_G128_V7_RESIDUAL.contains("kernel void gemv_q1_g128_v7_residual"));
        assert!(MSL_GEMV_Q1_G128_V8.contains("kernel void gemv_q1_g128_v8"));
        assert!(MSL_GEMV_Q1_G128_V8_RESIDUAL.contains("kernel void gemv_q1_g128_v8_residual"));
        assert!(MSL_GEMV_Q1_G128_V9.contains("kernel void gemv_q1_g128_v9"));
        assert!(MSL_GEMV_Q1_G128_V9_RESIDUAL.contains("kernel void gemv_q1_g128_v9_residual"));
        assert!(MSL_GEMV_Q1_G128_V10.contains("kernel void gemv_q1_g128_v10"));
        assert!(MSL_GEMV_Q1_G128_V10_RESIDUAL.contains("kernel void gemv_q1_g128_v10_residual"));

        assert!(MSL_ARGMAX.contains("kernel void argmax"));
        assert!(MSL_FUSED_GATE_UP_SWIGLU_Q1.contains("kernel void fused_gate_up_swiglu_q1"));
        // Ternary GEMV kernel
        assert!(MSL_GEMV_TQ2_G128_V1.contains("kernel void gemv_tq2_g128_v1"));

        // V7-based GEMM batch prefill kernels
        assert!(MSL_GEMM_Q1_G128_V7.contains("kernel void gemm_q1_g128_v7"));
        assert!(MSL_GEMM_Q1_G128_V7_RESIDUAL.contains("kernel void gemm_q1_g128_v7_residual"));
        assert!(
            MSL_FUSED_GATE_UP_SWIGLU_GEMM_Q1.contains("kernel void fused_gate_up_swiglu_gemm_q1")
        );
        // Ternary V7-based GEMM batch prefill kernel
        assert!(MSL_GEMM_TQ2_G128_V7.contains("kernel void gemm_tq2_g128_v7"));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_kernels_contain_entry_points() {
        assert!(CUDA_GEMV_Q1_G128.contains("gemv_q1_g128"));
        assert!(CUDA_GEMM_Q1_G128.contains("gemm_q1_g128"));
        assert!(CUDA_SOFTMAX.contains("softmax"));
        assert!(CUDA_RELU.contains("relu"));
        assert!(CUDA_RMSNORM.contains("rmsnorm"));
        assert!(CUDA_SILU.contains("silu"));
        assert!(CUDA_MATVEC_F32.contains("matvec_f32"));
        assert!(CUDA_SWIGLU.contains("swiglu"));
        assert!(CUDA_SWIGLU_FUSED.contains("swiglu_fused"));
        assert!(CUDA_RESIDUAL_ADD.contains("residual_add"));
        assert!(CUDA_RMSNORM_WEIGHTED.contains("rmsnorm_weighted"));
    }
}
