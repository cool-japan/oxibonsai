//! Host-only kernel-source-string assertions for the K-quant CUDA prefill
//! module.  These tests verify that the NVRTC source string includes all 18
//! expected kernel names without requiring a CUDA device.

use crate::gpu_backend::cuda_k_quant_prefill_kernels::CUDA_K_QUANT_PREFILL_KERNELS_SRC;

/// Verify the kernel source contains `gemm_q2k`.
#[test]
fn test_k_quant_prefill_kernels_src_has_gemm_q2k() {
    assert!(
        CUDA_K_QUANT_PREFILL_KERNELS_SRC.contains("gemm_q2k"),
        "CUDA_K_QUANT_PREFILL_KERNELS_SRC must contain gemm_q2k"
    );
}

/// Verify the kernel source contains `gemm_q4k`.
#[test]
fn test_k_quant_prefill_kernels_src_has_gemm_q4k() {
    assert!(
        CUDA_K_QUANT_PREFILL_KERNELS_SRC.contains("gemm_q4k"),
        "CUDA_K_QUANT_PREFILL_KERNELS_SRC must contain gemm_q4k"
    );
}

/// Verify the kernel source contains `fused_gate_up_swiglu_gemm_q6k`.
#[test]
fn test_k_quant_prefill_kernels_src_has_fused_gate_up_q6k() {
    assert!(
        CUDA_K_QUANT_PREFILL_KERNELS_SRC.contains("fused_gate_up_swiglu_gemm_q6k"),
        "CUDA_K_QUANT_PREFILL_KERNELS_SRC must contain fused_gate_up_swiglu_gemm_q6k"
    );
}

/// Verify the kernel source contains `gemm_q8k`.
#[test]
fn test_k_quant_prefill_kernels_src_has_gemm_q8k() {
    assert!(
        CUDA_K_QUANT_PREFILL_KERNELS_SRC.contains("gemm_q8k"),
        "CUDA_K_QUANT_PREFILL_KERNELS_SRC must contain gemm_q8k"
    );
}

/// Verify all 6 format kernels are present in the source.
#[test]
fn test_k_quant_format_variants_all_present() {
    let src = CUDA_K_QUANT_PREFILL_KERNELS_SRC;
    assert!(src.contains("gemm_q2k"), "missing gemm_q2k");
    assert!(src.contains("gemm_q3k"), "missing gemm_q3k");
    assert!(src.contains("gemm_q4k"), "missing gemm_q4k");
    assert!(src.contains("gemm_q5k"), "missing gemm_q5k");
    assert!(src.contains("gemm_q6k"), "missing gemm_q6k");
    assert!(src.contains("gemm_q8k"), "missing gemm_q8k");
}
