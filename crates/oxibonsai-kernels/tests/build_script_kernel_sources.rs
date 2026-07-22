//! Regression tests for `build.rs`'s embedded-metallib source discovery.
//!
//! `build.rs` pre-compiles a whitelisted subset of the MSL kernels living
//! under `src/gpu_backend/kernel_sources/` into a build-time metallib on
//! macOS. That directory used to be a single file
//! (`src/gpu_backend/kernel_sources.rs`); when it was split into a module
//! directory, `build.rs` kept pointing at the deleted single-file path, so
//! `std::fs::read_to_string` always failed and the embedded-metallib fast
//! path silently produced an empty stub on every macOS build.
//!
//! These tests run on every platform (they only touch the filesystem, not
//! Metal/xcrun) and assert the two invariants that regressed:
//!   1. `kernel_sources/` is a directory of `.rs` files (not a single file),
//!      so a path pointed at the old single-file location would fail.
//!   2. Every kernel constant `build.rs`'s `ACTIVE_KERNELS` whitelist expects
//!      (mirrored here) is actually declared somewhere under that directory,
//!      so a rename/typo/whitelist drift is caught in CI instead of
//!      silently degrading to the empty-metallib fallback at macOS build
//!      time.

use std::fs;
use std::path::PathBuf;

/// Mirrors `ACTIVE_KERNELS` in `build.rs`, which must in turn mirror the
/// kernel list pushed by `build_combined_msl()` in
/// `src/gpu_backend/metal_graph/pipelines.rs`. Kept here as an independent
/// copy (rather than importing build.rs, which isn't a normal compilation
/// target) so a future desync between `build.rs` and this test is itself a
/// signal that something needs reconciling in three places at once.
const ACTIVE_KERNELS: &[&str] = &[
    "MSL_GEMV_Q1_G128_V7",
    "MSL_GEMV_Q1_G128_V7_RESIDUAL",
    "MSL_RMSNORM_WEIGHTED_V2",
    "MSL_RESIDUAL_ADD",
    "MSL_FUSED_QK_NORM",
    "MSL_FUSED_QK_ROPE",
    "MSL_FUSED_QK_NORM_ROPE",
    "MSL_FUSED_KV_STORE",
    "MSL_FUSED_GATE_UP_SWIGLU_Q1",
    "MSL_BATCHED_ATTENTION_SCORES_V2",
    "MSL_BATCHED_SOFTMAX",
    "MSL_BATCHED_ATTENTION_WEIGHTED_SUM",
    "MSL_ARGMAX",
    "MSL_BATCHED_RMSNORM_V2",
    "MSL_BATCHED_SWIGLU",
    "MSL_GEMM_Q1_G128_V7",
    "MSL_GEMM_Q1_G128_V7_RESIDUAL",
    "MSL_FUSED_GATE_UP_SWIGLU_GEMM_Q1",
    "MSL_GEMV_TQ2_G128_V1",
    "MSL_GEMM_TQ2_G128_V7",
    "MSL_GEMM_TQ2_G128_V8_TILED",
    "MSL_GEMM_TQ2_G128_V9_SIMDGROUP",
    "MSL_GEMM_TQ2_G128_V10_SIMDGROUP",
    "MSL_GEMM_F32_SIMDGROUP",
    "MSL_IM2COL_F32",
    "MSL_GROUPNORM_F32",
    "MSL_SILU_F32",
    "MSL_UPSAMPLE_NEAREST_F32",
    "MSL_CONV2D_F32_IMPLICIT",
    "MSL_DIT_JOINT_ATTENTION_FLASH",
];

fn kernel_sources_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/gpu_backend/kernel_sources")
}

/// The old (deleted) single-file location `build.rs` used to point at. This
/// path must NOT exist — its presence (or the directory's absence) is
/// exactly the regression that made the embedded-metallib fast path dead.
fn legacy_single_file_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/gpu_backend/kernel_sources.rs")
}

#[test]
fn kernel_sources_is_a_directory_of_rs_files_not_a_single_file() {
    let dir = kernel_sources_dir();
    assert!(
        dir.is_dir(),
        "expected {} to be a directory (kernel_sources module split); \
         build.rs's source reader depends on this being a directory of .rs files",
        dir.display()
    );

    let legacy = legacy_single_file_path();
    assert!(
        !legacy.is_file(),
        "found a stray {} — this is the old single-file kernel_sources.rs path \
         that build.rs used to (incorrectly) target after the module split; \
         its presence would mask the regression this test guards against",
        legacy.display()
    );

    let rs_file_count = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .flatten()
        .filter(|entry| entry.path().extension().and_then(|e| e.to_str()) == Some("rs"))
        .count();
    assert!(
        rs_file_count >= 10,
        "expected kernel_sources/ to contain at least 10 .rs module files, found {rs_file_count}"
    );
}

/// Concatenate every `.rs` file under `kernel_sources/`, mirroring
/// `build.rs`'s `read_kernel_sources()` helper, and confirm every
/// `ACTIVE_KERNELS` entry is declared somewhere in the concatenation. This
/// is the same lookup `build.rs::extract_and_combine_msl` performs, so a
/// failure here means the macOS build-time metallib step would panic (by
/// design, since a silent skip previously produced an incomplete metallib
/// with no per-function runtime fallback).
#[test]
fn all_active_kernels_are_declared_under_kernel_sources_dir() {
    let dir = kernel_sources_dir();
    let mut combined = String::new();
    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("rs"))
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no .rs files found under {}",
        dir.display()
    );

    for path in &paths {
        let content = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        combined.push_str(&content);
        combined.push('\n');
    }

    let mut missing = Vec::new();
    for kernel_name in ACTIVE_KERNELS {
        let pattern = format!("pub const {kernel_name}: &str = r#\"");
        if !combined.contains(&pattern) {
            missing.push(*kernel_name);
        }
    }

    assert!(
        missing.is_empty(),
        "ACTIVE_KERNELS whitelist (mirrored from build.rs) references kernel constants \
         that are not declared under {}: {missing:?}. Either the whitelist is stale or the \
         kernel was renamed/removed — build.rs would panic building this crate on macOS.",
        dir.display()
    );
}
