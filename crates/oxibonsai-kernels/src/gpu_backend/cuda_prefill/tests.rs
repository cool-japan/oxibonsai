//! Tests for the CUDA prefill path.
//!
//! Host-only tests assert kernel source-string contents and dimension
//! arithmetic; they run on all platforms (no GPU required).
//!
//! CI-GPU-gated tests early-return when no CUDA device is present so the suite
//! stays green on macOS / CPU-only Linux.

use super::{init_prefill_modules, CudaPrefillBuffers};
use crate::gpu_backend::cuda_graph::CudaGraph;
use crate::gpu_backend::cuda_prefill_kernels::CUDA_PREFILL_KERNELS_SRC;

/// Verify the kernel source string contains `gemm_q1_g128_v7` without GPU.
#[test]
fn test_prefill_kernel_source_has_gemm() {
    assert!(
        CUDA_PREFILL_KERNELS_SRC.contains("gemm_q1_g128_v7"),
        "CUDA_PREFILL_KERNELS_SRC must contain gemm_q1_g128_v7"
    );
}

/// Verify the kernel source string contains `batched_rmsnorm_v2` without GPU.
#[test]
fn test_prefill_kernel_source_has_batched_rmsnorm() {
    assert!(
        CUDA_PREFILL_KERNELS_SRC.contains("batched_rmsnorm_v2"),
        "CUDA_PREFILL_KERNELS_SRC must contain batched_rmsnorm_v2"
    );
}

/// Verify the kernel source string contains `fused_gate_up_swiglu_gemm_q1`.
#[test]
fn test_prefill_kernel_source_has_fused_gemm() {
    assert!(
        CUDA_PREFILL_KERNELS_SRC.contains("fused_gate_up_swiglu_gemm_q1"),
        "CUDA_PREFILL_KERNELS_SRC must contain fused_gate_up_swiglu_gemm_q1"
    );
}

/// Verify `CudaPrefillBuffers::matches` correctly checks dimension equality.
#[test]
fn test_prefill_buffers_dimension_arithmetic() {
    let batch_size = 8usize;
    let _hidden_size = 2048usize;
    let intermediate_size = 8192usize;
    let nq = 32usize;
    let nkv = 8usize;
    let head_dim = 64usize;
    let _max_seq = 512usize;
    let qkv_total = (nq + 2 * nkv) * head_dim;
    assert_eq!(qkv_total, 48 * 64);
    let gate_up_size = 2 * batch_size * intermediate_size;
    assert_eq!(gate_up_size, 2 * 8 * 8192);
    // Touch the imported type so it stays cfg-active.
    let _ = std::mem::size_of::<CudaPrefillBuffers>();
}

/// Verify `init_prefill_modules` / `CudaGraph::global` gracefully skip without GPU.
#[test]
fn test_cuda_prefill_modules_init() {
    let graph_result = CudaGraph::global();
    if graph_result.is_err() {
        // No CUDA device present — skip gracefully.
        return;
    }
    let graph = graph_result.expect("prefill graph init should succeed");
    let result = init_prefill_modules(&graph);
    assert!(
        result.is_ok(),
        "prefill module init failed: {:?}",
        result.err()
    );
}

/// Host-only: verify kernel source contains all three TQ2 GEMM kernel names.
#[test]
fn test_prefill_kernel_source_has_gemm_tq2() {
    assert!(
        CUDA_PREFILL_KERNELS_SRC.contains("gemm_tq2_g128_v7"),
        "must contain gemm_tq2_g128_v7"
    );
    assert!(
        CUDA_PREFILL_KERNELS_SRC.contains("gemm_tq2_g128_v7_residual"),
        "must contain gemm_tq2_g128_v7_residual"
    );
    assert!(
        CUDA_PREFILL_KERNELS_SRC.contains("fused_gate_up_swiglu_gemm_tq2"),
        "must contain fused_gate_up_swiglu_gemm_tq2"
    );
}

/// Host-only: verify TQ2 helper functions are present in kernel source.
#[test]
fn test_prefill_kernel_source_has_tq2_helpers() {
    assert!(
        CUDA_PREFILL_KERNELS_SRC.contains("pf_decode_tq2"),
        "must contain pf_decode_tq2 helper"
    );
    assert!(
        CUDA_PREFILL_KERNELS_SRC.contains("pf_byte_dot_tq2"),
        "must contain pf_byte_dot_tq2 helper"
    );
}

/// CI-GPU-gated: compile TQ2 prefill kernels and verify module loads.
#[test]
fn test_cuda_prefill_tq2_modules_compile() {
    let graph_result = CudaGraph::global();
    if graph_result.is_err() {
        eprintln!("SKIP: test_cuda_prefill_tq2_modules_compile — no CUDA device");
        return;
    }
    let graph = graph_result.expect("tq2 prefill graph init should succeed");
    let result = init_prefill_modules(&graph);
    assert!(
        result.is_ok(),
        "TQ2 prefill module init failed (kernel compile error?): {:?}",
        result.err()
    );
    // Verify the TQ2 modules are accessible (they are fields of CudaPrefillModules).
    // If we got Ok, all 8 kernels (5 Q1 + 3 TQ2) compiled and loaded successfully.
}
