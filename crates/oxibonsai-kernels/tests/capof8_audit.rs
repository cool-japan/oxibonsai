//! Cap-of-8 batch-kernel regression audit (host-only, no GPU required).
//!
//! # Background
//!
//! Every batch-GEMM kernel in OxiBonsai's CUDA (NVRTC) and Metal (MSL)
//! prefill paths uses a fixed warp-per-row + 8-column tile layout:
//!
//! ```text
//!   for (uint col_base = 0; col_base < batch_size; col_base += 8u) {
//!       // process up to 8 output columns at once for this row
//!   }
//! ```
//!
//! Without the outer `col_base` loop, the kernel silently truncates any
//! batch beyond the first 8 columns — the bug pattern documented in the
//! `kernel_pattern_capof8` audit memory which originally surfaced as
//! Phase 13.x prefill regressions.
//!
//! This test enumerates every batch-GEMM kernel by name and asserts that
//! (a) the kernel entry-point appears in the corresponding source string
//! and (b) the source contains the `col_base += 8` outer-loop pattern.
//! It runs on all platforms (host-only) — the only feature-gating is to
//! match the cfg-gates of the kernel-source files themselves.
//!
//! When adding a new batch kernel, add it to the matching list below and
//! verify the source contains the cap-of-8 loop *before* shipping.
//! Single-token GEMV kernels (where `batch_size = 1` by construction) are
//! deliberately omitted; only kernels that take a runtime `batch_size`
//! argument need the loop.

/// Assert that `source` contains the kernel entry-point `kernel_name`
/// AND the `col_base += 8` outer-loop pattern.
///
/// The CUDA pattern is `col_base += 8` (plain) and the MSL pattern is
/// `col_base += 8u` — accept either spelling.
#[track_caller]
fn assert_capof8(source: &str, kernel_name: &str) {
    assert!(
        source.contains(kernel_name),
        "audit: kernel `{kernel_name}` entry-point not found in its source string"
    );
    assert!(
        source.contains("col_base += 8u") || source.contains("col_base += 8 "),
        "audit: kernel `{kernel_name}` is missing the cap-of-8 outer-loop pattern \
         (`col_base += 8` / `col_base += 8u`). Without it, any batch_size > 8 silently \
         truncates — see kernel_pattern_capof8 memory."
    );
}

// =============================================================================
// CUDA batch-GEMM kernels (native-cuda on Linux/Windows)
// =============================================================================

/// Audit the 6 batch kernels in `CUDA_PREFILL_KERNELS_SRC`
/// (Q1 × 3 variants + TQ2 × 3 variants).
#[test]
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
fn cuda_prefill_kernels_use_capof8() {
    use oxibonsai_kernels::gpu_backend::cuda_prefill_kernels::CUDA_PREFILL_KERNELS_SRC;

    let src = CUDA_PREFILL_KERNELS_SRC;
    // Q1
    assert_capof8(src, "gemm_q1_g128_v7");
    assert_capof8(src, "gemm_q1_g128_v7_residual");
    assert_capof8(src, "fused_gate_up_swiglu_gemm_q1");
    // TQ2
    assert_capof8(src, "gemm_tq2_g128_v7");
    assert_capof8(src, "gemm_tq2_g128_v7_residual");
    assert_capof8(src, "fused_gate_up_swiglu_gemm_tq2");
}

/// Audit the 18 batch kernels in `CUDA_K_QUANT_PREFILL_KERNELS_SRC`
/// (3 variants × 6 K-quant formats).
#[test]
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
fn cuda_k_quant_prefill_kernels_use_capof8() {
    use oxibonsai_kernels::gpu_backend::cuda_k_quant_prefill_kernels::CUDA_K_QUANT_PREFILL_KERNELS_SRC;

    let src = CUDA_K_QUANT_PREFILL_KERNELS_SRC;
    for fmt in ["q2k", "q3k", "q4k", "q5k", "q6k", "q8k"] {
        assert_capof8(src, &format!("gemm_{fmt}"));
        assert_capof8(src, &format!("gemm_{fmt}_residual"));
        assert_capof8(src, &format!("fused_gate_up_swiglu_gemm_{fmt}"));
    }
}

/// Audit the 6 batch kernels in `CUDA_FP8_PREFILL_KERNELS_SRC`
/// (3 variants × 2 FP8 formats — `gemv_pf_*` are per-token, not batch,
/// and are deliberately not audited).
#[test]
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
fn cuda_fp8_prefill_kernels_use_capof8() {
    use oxibonsai_kernels::gpu_backend::cuda_fp8_prefill_kernels::CUDA_FP8_PREFILL_KERNELS_SRC;

    let src = CUDA_FP8_PREFILL_KERNELS_SRC;
    for fmt in ["fp8_e4m3", "fp8_e5m2"] {
        assert_capof8(src, &format!("gemm_{fmt}"));
        assert_capof8(src, &format!("gemm_{fmt}_residual"));
        assert_capof8(src, &format!("fused_gate_up_swiglu_gemm_{fmt}"));
    }
}

/// Audit the 6 batch kernels in `CUDA_Q_STD_PREFILL_KERNELS_SRC`
/// (3 variants × 2 Q-std formats: Q4_0 and Q8_0).
#[test]
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
fn cuda_q_std_prefill_kernels_use_capof8() {
    use oxibonsai_kernels::gpu_backend::cuda_q_std_prefill_kernels::CUDA_Q_STD_PREFILL_KERNELS_SRC;

    let src = CUDA_Q_STD_PREFILL_KERNELS_SRC;
    for fmt in ["q4_0", "q8_0"] {
        assert_capof8(src, &format!("gemm_{fmt}"));
        assert_capof8(src, &format!("gemm_{fmt}_residual"));
        assert_capof8(src, &format!("fused_gate_up_swiglu_gemm_{fmt}"));
    }
}

// =============================================================================
// Metal batch-GEMM kernels (MSL, macOS only)
// =============================================================================

/// Audit the 4 batch kernels in `kernel_sources::prefill`
/// (Q1 × 3 variants + TQ2 × 1 variant).
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_prefill_kernels_use_capof8() {
    use oxibonsai_kernels::gpu_backend::kernel_sources::{
        MSL_FUSED_GATE_UP_SWIGLU_GEMM_Q1, MSL_GEMM_Q1_G128_V7, MSL_GEMM_Q1_G128_V7_RESIDUAL,
        MSL_GEMM_TQ2_G128_V7,
    };

    assert_capof8(MSL_GEMM_Q1_G128_V7, "gemm_q1_g128_v7");
    assert_capof8(MSL_GEMM_Q1_G128_V7_RESIDUAL, "gemm_q1_g128_v7_residual");
    assert_capof8(
        MSL_FUSED_GATE_UP_SWIGLU_GEMM_Q1,
        "fused_gate_up_swiglu_gemm_q1",
    );
    assert_capof8(MSL_GEMM_TQ2_G128_V7, "gemm_tq2_g128_v7");
}

/// Audit the 6 batch FP8 kernels in `kernel_sources::fp8_prefill`
/// (3 variants × 2 FP8 formats — `gemv_*_pf` are per-token).
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_fp8_prefill_kernels_use_capof8() {
    use oxibonsai_kernels::gpu_backend::kernel_sources::{
        MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1, MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1,
        MSL_GEMM_FP8_E4M3_RESIDUAL_V1, MSL_GEMM_FP8_E4M3_V1, MSL_GEMM_FP8_E5M2_RESIDUAL_V1,
        MSL_GEMM_FP8_E5M2_V1,
    };

    // E4M3
    assert_capof8(MSL_GEMM_FP8_E4M3_V1, "gemm_fp8_e4m3");
    assert_capof8(MSL_GEMM_FP8_E4M3_RESIDUAL_V1, "gemm_fp8_e4m3_residual");
    assert_capof8(
        MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1,
        "fused_gate_up_swiglu_gemm_fp8_e4m3",
    );
    // E5M2
    assert_capof8(MSL_GEMM_FP8_E5M2_V1, "gemm_fp8_e5m2");
    assert_capof8(MSL_GEMM_FP8_E5M2_RESIDUAL_V1, "gemm_fp8_e5m2_residual");
    assert_capof8(
        MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1,
        "fused_gate_up_swiglu_gemm_fp8_e5m2",
    );
}

// =============================================================================
// Host-only sanity check: assert the helper rejects a non-cap-of-8 source.
// =============================================================================

/// Defensive: a source missing the cap-of-8 pattern must trigger an assertion.
/// We call the helper via `std::panic::catch_unwind` so we can detect the panic.
#[test]
fn helper_rejects_source_without_capof8() {
    let bad_source = "extern \"C\" __global__ void gemm_bad(...) { /* no cap-of-8 loop */ }";
    let result = std::panic::catch_unwind(|| {
        assert_capof8(bad_source, "gemm_bad");
    });
    assert!(
        result.is_err(),
        "helper should panic on missing cap-of-8 pattern"
    );
}

/// Defensive: the helper must also reject a source missing the kernel name.
#[test]
fn helper_rejects_source_without_kernel_name() {
    let some_source = "for (uint col_base = 0u; col_base < batch_size; col_base += 8u) {}";
    let result = std::panic::catch_unwind(|| {
        assert_capof8(some_source, "nonexistent_kernel");
    });
    assert!(
        result.is_err(),
        "helper should panic on missing kernel-name pattern"
    );
}
