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
//! Metal/xcrun) and assert the invariants that regressed:
//!   1. `kernel_sources/` is a directory of `.rs` files (not a single file),
//!      so a path pointed at the old single-file location would fail.
//!   2. Every kernel constant `build.rs`'s `ACTIVE_KERNELS` whitelist expects
//!      (mirrored here) is actually declared somewhere under that directory,
//!      so a rename/typo/whitelist drift is caught in CI instead of
//!      silently degrading to the empty-metallib fallback at macOS build
//!      time.
//!   3. (MET-12) The set of kernels `build_combined_msl()` in `pipelines.rs`
//!      actually pushes into the combined MSL matches `ACTIVE_KERNELS`
//!      (mirrored here) exactly — the *reverse* direction from #2. Before
//!      this test existed, a kernel added to `build_combined_msl` (and
//!      `pipeline_for`) but never added to `ACTIVE_KERNELS` would compile a
//!      combined MSL library the embedded metallib doesn't fully cover:
//!      `MetalPipelines::compile()` would then fail on that kernel's
//!      `pipeline_for(&library, device, "...")` with no per-function
//!      fallback, `MetalGraph::global()` would fail, and every caller's
//!      `.is_ok()` would silently degrade the whole Metal backend to CPU.
//!      `pipelines.rs::load_or_compile_library` closes the same gap at
//!      *runtime* (falling through to disk-cache/xcrun/runtime compilation
//!      instead of trusting an embedded metallib that turns out to be
//!      incomplete); this test closes it at *build/CI* time by catching the
//!      whitelist desync itself, before it ever reaches a shipped binary.
//!   4. (wave-3 re-review) #3 above only compared this file's own
//!      `ACTIVE_KERNELS` *mirror* against `pipelines.rs`'s real pushes —
//!      nothing compared `build.rs`'s own real `ACTIVE_KERNELS` (the one
//!      list that actually drives `extract_and_combine_msl` and therefore
//!      the embedded metallib) against anything. Two hand-typed mirrors that
//!      happen to agree with each other say nothing about whether either one
//!      still agrees with the array that ships. `combined_msl_pushes_exactly_
//!      match_active_kernels` below now parses `build.rs`'s literal
//!      `ACTIVE_KERNELS` array textually — the same "read the real file as
//!      text" technique as `all_active_kernels_are_declared_under_kernel_
//!      sources_dir` above — and cross-checks it against this file's mirror
//!      too, making the invariant a genuine three-way tie (mirror ==
//!      `build_combined_msl()` pushes == `build.rs`'s real `ACTIVE_KERNELS`)
//!      instead of a two-way one with an unchecked third party.

use std::collections::BTreeSet;
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
    // GPU top-k (perf-11 sampled-path partial reduction, beside MSL_ARGMAX)
    "MSL_TOPK_F32",
    // K-quant GEMV (MET-10)
    "MSL_GEMV_Q2K_V1",
    "MSL_GEMV_Q3K_V1",
    "MSL_GEMV_Q4K_V1",
    "MSL_GEMV_Q5K_V1",
    "MSL_GEMV_Q6K_V1",
    "MSL_GEMV_Q8K_V1",
    // Standard GGUF Q4_0 / Q8_0 GEMV (MET-10)
    "MSL_GEMV_Q4_0_V1",
    "MSL_GEMV_Q8_0_V1",
    // FP8 single-token GEMV (MET-10)
    "MSL_GEMV_FP8_E4M3_V1",
    "MSL_GEMV_FP8_E5M2_V1",
    // FP8 batch prefill GEMM / fused / gemv-pf (MET-10)
    "MSL_GEMM_FP8_E4M3_V1",
    "MSL_GEMM_FP8_E4M3_RESIDUAL_V1",
    "MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1",
    "MSL_GEMV_FP8_E4M3_PF_V1",
    "MSL_GEMM_FP8_E5M2_V1",
    "MSL_GEMM_FP8_E5M2_RESIDUAL_V1",
    "MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1",
    "MSL_GEMV_FP8_E5M2_PF_V1",
    // Batched prefill attention (perf-01), in the combined library
    "MSL_PREFILL_QKV_PREPARE",
    "MSL_PREFILL_FLASH_ATTENTION",
    // Qwen3.5 / Bonsai 2 hybrid stack (MET-09); the common prelude first
    "MSL_QWEN35_COMMON",
    "MSL_QWEN35_ROTATE",
    "MSL_QWEN35_GEMV",
    "MSL_QWEN35_SSM",
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

/// Absolute path to `src/gpu_backend/metal_graph/pipelines.rs`.
fn pipelines_rs_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/gpu_backend/metal_graph/pipelines.rs")
}

/// Parse every `MSL_*` name `build_combined_msl()` pushes out of
/// `pipelines.rs`'s source text.
///
/// Every push in that function has the fixed textual shape
/// `src.push_str(kernel_sources::MSL_XXX);` — this scans for exactly that
/// literal prefix (not a general "does this file mention MSL_XXX anywhere"
/// scan), so it cannot pick up an unrelated reference such as the bf16
/// sidecar kernel (`load_bf16_library`'s `let src = kernel_sources::
/// MSL_GEMM_BF16_SIMDGROUP;`, a different code shape entirely — that kernel
/// is intentionally excluded from the combined library, see its own doc).
/// This mirrors the existing `all_active_kernels_are_declared_under_kernel_sources_dir`
/// test's philosophy: a precise textual scan over real Rust parsing.
fn parse_build_combined_msl_pushes(pipelines_src: &str) -> BTreeSet<String> {
    const NEEDLE: &str = "src.push_str(kernel_sources::";
    let mut names = BTreeSet::new();
    let mut rest = pipelines_src;
    while let Some(pos) = rest.find(NEEDLE) {
        let after = &rest[pos + NEEDLE.len()..];
        let end = after.find(')').unwrap_or(after.len());
        names.insert(after[..end].to_string());
        rest = &after[end..];
    }
    names
}

/// Absolute path to `build.rs`.
fn build_rs_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("build.rs")
}

/// Parse the literal `&[&str]` entries out of `build.rs`'s REAL
/// `const ACTIVE_KERNELS: &[&str] = &[ ... ];` array — the array that
/// actually drives `extract_and_combine_msl` and therefore the embedded
/// metallib `try_load_embedded_metallib` loads — as opposed to either mirror
/// already cross-checked by [`combined_msl_pushes_exactly_match_active_kernels`]
/// (this file's own `ACTIVE_KERNELS` constant, and `build_combined_msl()`'s
/// pushes).
///
/// `build.rs` is a Cargo build script, not an importable module, so a
/// textual parse (mirroring this file's other checks' philosophy) is the
/// only way an integration test can read it. `build.rs`'s own
/// `#[cfg(target_os = "macos")]` gate on the real declaration is irrelevant
/// here — this reads the file as plain text, not compiled code, so the
/// parse runs identically on every host platform (matching every other test
/// in this file, which the module doc already notes "run on every
/// platform"). Strips `//`-to-end-of-line comments first so a future comment
/// mentioning a quoted word can never be mistaken for an array entry.
fn parse_build_rs_active_kernels(build_rs_src: &str) -> BTreeSet<String> {
    let uncommented: String = build_rs_src
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");

    const NEEDLE: &str = "const ACTIVE_KERNELS: &[&str] = &[";
    let start = uncommented.find(NEEDLE).unwrap_or_else(|| {
        panic!(
            "build.rs: could not find `{NEEDLE}` — has the ACTIVE_KERNELS declaration's exact \
             text changed? Update parse_build_rs_active_kernels to match."
        )
    });
    let after = &uncommented[start + NEEDLE.len()..];
    let end = after.find("];").unwrap_or_else(|| {
        panic!("build.rs: the ACTIVE_KERNELS array is unterminated (no closing `];` found)")
    });
    let block = &after[..end];

    let mut names = BTreeSet::new();
    let mut rest = block;
    while let Some(q1) = rest.find('"') {
        let after_q1 = &rest[q1 + 1..];
        let q2 = after_q1.find('"').unwrap_or_else(|| {
            panic!("build.rs: unterminated string literal inside the ACTIVE_KERNELS array")
        });
        names.insert(after_q1[..q2].to_string());
        rest = &after_q1[q2 + 1..];
    }
    names
}

/// MET-12: the *reverse* direction from
/// `all_active_kernels_are_declared_under_kernel_sources_dir` — every name
/// `build_combined_msl()` pushes must be in `ACTIVE_KERNELS` (so `build.rs`
/// actually embeds it), and every name in `ACTIVE_KERNELS` must be pushed
/// (so the whitelist carries no dead entries). See the module doc for why
/// only checking one direction left this gap.
///
/// (wave-3 re-review, module doc point 4) Also parses `build.rs`'s own real
/// `ACTIVE_KERNELS` array and cross-checks it against this file's mirror,
/// so the three lists that must all agree — this file's `ACTIVE_KERNELS`
/// mirror, `build_combined_msl()`'s real pushes, and `build.rs`'s real
/// `ACTIVE_KERNELS` — are actually all compared to each other, not just the
/// first two to each other while the one that drives the real build goes
/// unchecked.
#[test]
fn combined_msl_pushes_exactly_match_active_kernels() {
    let path = pipelines_rs_path();
    let src = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    let pushed = parse_build_combined_msl_pushes(&src);
    assert!(
        !pushed.is_empty(),
        "found zero `src.push_str(kernel_sources::...)` calls in {} — the textual scan itself \
         is broken (build_combined_msl's shape changed) or the function was emptied out",
        path.display()
    );
    let whitelist: BTreeSet<String> = ACTIVE_KERNELS.iter().map(|s| s.to_string()).collect();

    let pushed_but_not_whitelisted: Vec<&String> = pushed.difference(&whitelist).collect();
    let whitelisted_but_not_pushed: Vec<&String> = whitelist.difference(&pushed).collect();

    assert!(
        pushed_but_not_whitelisted.is_empty(),
        "build_combined_msl() in {} pushes {pushed_but_not_whitelisted:?}, which \
         ACTIVE_KERNELS (mirrored here, and in build.rs) does not list — build.rs would embed \
         a metallib missing this entry point, silently degrading the whole Metal backend to CPU \
         (MET-12). Add it to ACTIVE_KERNELS in both build.rs and this file.",
        path.display()
    );
    assert!(
        whitelisted_but_not_pushed.is_empty(),
        "ACTIVE_KERNELS (mirrored here, and in build.rs) lists {whitelisted_but_not_pushed:?}, \
         which build_combined_msl() in {} never pushes — either build_combined_msl is missing \
         the push (the kernel is compiled into build.rs's embedded metallib but never actually \
         used by MetalPipelines::compile/pipeline_for) or the whitelist entry is dead and should \
         be removed from both build.rs and this file.",
        path.display()
    );

    // Third leg (wave-3 re-review): cross-check this file's `ACTIVE_KERNELS`
    // mirror against build.rs's REAL `ACTIVE_KERNELS` array — the list that
    // actually drives `extract_and_combine_msl` and therefore the embedded
    // metallib. The two asserts above only prove this file's mirror agrees
    // with `pipelines.rs`; without this leg, build.rs's real list could
    // silently drift from both and nothing here would notice.
    let build_rs = build_rs_path();
    let build_rs_src = fs::read_to_string(&build_rs)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", build_rs.display()));
    let build_rs_active_kernels = parse_build_rs_active_kernels(&build_rs_src);
    assert!(
        !build_rs_active_kernels.is_empty(),
        "found zero string literals inside build.rs's ACTIVE_KERNELS array at {} — the textual \
         parse itself is broken (the declaration's shape changed) or the array was emptied out",
        build_rs.display()
    );

    let build_rs_missing_from_mirror: Vec<&String> =
        build_rs_active_kernels.difference(&whitelist).collect();
    let mirror_missing_from_build_rs: Vec<&String> =
        whitelist.difference(&build_rs_active_kernels).collect();

    assert!(
        build_rs_missing_from_mirror.is_empty(),
        "build.rs's real ACTIVE_KERNELS array at {} lists {build_rs_missing_from_mirror:?}, \
         which this file's own ACTIVE_KERNELS mirror does not — the two whitelists have \
         desynced; update this file's mirror to match build.rs (the array that actually drives \
         the embedded metallib).",
        build_rs.display()
    );
    assert!(
        mirror_missing_from_build_rs.is_empty(),
        "this file's ACTIVE_KERNELS mirror lists {mirror_missing_from_build_rs:?}, which \
         build.rs's real ACTIVE_KERNELS array at {} does not — the two whitelists have \
         desynced; update build.rs's array to match (or remove the stale mirror entry here if \
         it is no longer needed).",
        build_rs.display()
    );
}
