/// Build script for oxibonsai-kernels: pre-compile Metal shaders into a metallib.
///
/// On macOS with the Metal Toolchain installed:
///   1. Reads every `.rs` file under `src/gpu_backend/kernel_sources/` and
///      extracts the actively-used MSL raw string constants
///   2. Concatenates them into a single .metal file
///   3. Compiles via `xcrun -sdk macosx metal` → AIR → `xcrun metallib`
///   4. Writes the metallib to OUT_DIR for `include_bytes!()` consumption
///
/// On non-macOS or without the Metal Toolchain: writes an empty metallib stub,
/// and the Rust code falls back to runtime MSL compilation.
use std::path::Path;
#[cfg(target_os = "macos")]
use std::path::PathBuf;

/// Directory (relative to the crate root) holding the `kernel_sources` module
/// split (`mod.rs` plus one file per kernel family). Kept as a single
/// constant so the rerun-if-changed wiring and the source reader can never
/// drift apart from each other.
const KERNEL_SOURCES_DIR: &str = "src/gpu_backend/kernel_sources";

fn main() {
    emit_kernel_sources_rerun_if_changed();

    // Detect a nightly (or dev) compiler so the AArch64 software-prefetch
    // intrinsic (`core::arch::aarch64::_prefetch`, gated behind the
    // `stdarch_aarch64_prefetch` nightly feature) can be enabled. On stable
    // the feature attribute would error (E0554), so we gracefully degrade the
    // prefetch to a no-op — it is a pure perf hint and never affects results.
    //
    // Always declare the cfg via `rustc-check-cfg` so the `unexpected_cfgs`
    // lint stays quiet (required on current Rust); only *set* it on nightly.
    detect_nightly_aarch64_prefetch();

    let out_dir = match std::env::var("OUT_DIR") {
        Ok(d) => d,
        Err(_) => return,
    };

    let metallib_path = Path::new(&out_dir).join("combined.metallib");

    #[cfg(target_os = "macos")]
    {
        if try_compile_metal_shaders(&out_dir) {
            return;
        }
    }

    // Write empty stub if compilation was not attempted or failed
    let _ = std::fs::write(&metallib_path, b"");
}

/// Emit `cargo:rerun-if-changed` for the `kernel_sources/` module directory.
///
/// Cargo (>= 1.50) recursively stats a directory passed to
/// `rerun-if-changed`, but we additionally enumerate every `.rs` file inside
/// it and emit a per-file directive too — belt-and-suspenders against older
/// Cargo versions/toolchains that only stat the exact path given, and so a
/// `cargo:rerun-if-changed` failure mode (silently never re-running) fails
/// loud rather than quiet if the directory-level watch is ever unsupported.
fn emit_kernel_sources_rerun_if_changed() {
    println!("cargo:rerun-if-changed={KERNEL_SOURCES_DIR}");
    if let Ok(entries) = std::fs::read_dir(KERNEL_SOURCES_DIR) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

/// Detect whether the active compiler is a nightly/dev build and, if so, emit
/// the `nightly_aarch64_prefetch` cfg so `lib.rs` may enable the
/// `stdarch_aarch64_prefetch` feature and the prefetch intrinsic stays active.
///
/// Detection uses `$RUSTC -vV` (falling back to `rustc`) and inspects the
/// `release:` line: a nightly toolchain reports e.g. `release: 1.96.0-nightly`,
/// while dev builds report `-dev`. No external crates are required.
fn detect_nightly_aarch64_prefetch() {
    // Declare the cfg unconditionally so `unexpected_cfgs` never fires, even on
    // stable where the cfg is never set.
    println!("cargo:rustc-check-cfg=cfg(nightly_aarch64_prefetch)");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = match std::process::Command::new(&rustc).arg("-vV").output() {
        Ok(o) if o.status.success() => o,
        _ => return,
    };
    let version_info = String::from_utf8_lossy(&output.stdout);

    let is_nightly = version_info.lines().any(|line| {
        line.strip_prefix("release:")
            .map(|rest| {
                let rest = rest.trim();
                rest.contains("nightly") || rest.contains("dev")
            })
            .unwrap_or(false)
    });

    if is_nightly {
        println!("cargo:rustc-cfg=nightly_aarch64_prefetch");
    }
}

/// Read and concatenate every `.rs` file under `kernel_sources/` (sorted by
/// filename for deterministic output) into a single string that
/// `extract_and_combine_msl` can scan for `pub const MSL_XXX: &str = r#"..."#`
/// declarations, regardless of which sub-module a given kernel now lives in.
#[cfg(target_os = "macos")]
fn read_kernel_sources() -> Option<String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(KERNEL_SOURCES_DIR)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("rs"))
        .collect();
    if paths.is_empty() {
        return None;
    }
    paths.sort();

    let mut combined = String::new();
    for path in paths {
        let content = std::fs::read_to_string(&path).ok()?;
        combined.push_str(&content);
        combined.push('\n');
    }
    Some(combined)
}

#[cfg(target_os = "macos")]
fn try_compile_metal_shaders(out_dir: &str) -> bool {
    let ks_content = match read_kernel_sources() {
        Some(c) => c,
        None => return false,
    };

    let combined_msl = extract_and_combine_msl(&ks_content);
    if combined_msl.is_empty() {
        return false;
    }

    let metal_path = Path::new(out_dir).join("combined.metal");
    let air_path = Path::new(out_dir).join("combined.air");
    let metallib_path = Path::new(out_dir).join("combined.metallib");

    if std::fs::write(&metal_path, &combined_msl).is_err() {
        return false;
    }

    // Step 1: MSL → AIR
    let metal_src = match metal_path.to_str() {
        Some(s) => s,
        None => return false,
    };
    let air_dst = match air_path.to_str() {
        Some(s) => s,
        None => return false,
    };
    let result = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c", metal_src, "-o", air_dst])
        .output();
    match result {
        Ok(ref output) if output.status.success() => {}
        _ => return false,
    }

    // Step 2: AIR → metallib
    let metallib_dst = match metallib_path.to_str() {
        Some(s) => s,
        None => return false,
    };
    let result = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", air_dst, "-o", metallib_dst])
        .output();
    match result {
        Ok(ref output) if output.status.success() => {}
        _ => return false,
    }

    // Clean up intermediate files
    let _ = std::fs::remove_file(&metal_path);
    let _ = std::fs::remove_file(&air_path);

    true
}

/// MSL constant names that are actively used in the dispatch pipeline.
///
/// This list MUST mirror the constants pushed by `build_combined_msl()` in
/// `src/gpu_backend/metal_graph/pipelines.rs` exactly — that function is the
/// single source of truth for "which kernels does the runtime pipeline
/// actually load". If the two lists diverge, `try_load_embedded_metallib`
/// either (a) misses an entry point `MetalPipelines::compile()` needs, which
/// is a hard failure with no per-function fallback once the embedded library
/// loads, or (b) ships extra unused shader text, which only costs a little
/// extra build-time compilation. Prefer keeping this in exact sync with
/// `build_combined_msl()` whenever kernels are added, removed, or renamed
/// there; `extract_and_combine_msl` below panics loudly (failing the build)
/// if any of these names cannot be found in `kernel_sources/`, so a rename
/// or deletion is caught immediately instead of silently degrading to the
/// empty-metallib fallback.
#[cfg(target_os = "macos")]
const ACTIVE_KERNELS: &[&str] = &[
    // Decode path (single-token)
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
    // Prefill path (batch)
    "MSL_BATCHED_RMSNORM_V2",
    "MSL_BATCHED_SWIGLU",
    "MSL_GEMM_Q1_G128_V7",
    "MSL_GEMM_Q1_G128_V7_RESIDUAL",
    "MSL_FUSED_GATE_UP_SWIGLU_GEMM_Q1",
    // Ternary (TQ2_0_g128)
    "MSL_GEMV_TQ2_G128_V1",
    "MSL_GEMM_TQ2_G128_V7",
    "MSL_GEMM_TQ2_G128_V8_TILED",
    "MSL_GEMM_TQ2_G128_V9_SIMDGROUP",
    "MSL_GEMM_TQ2_G128_V10_SIMDGROUP",
    // f32-exact (text encoder)
    "MSL_GEMM_F32_SIMDGROUP",
    // FLUX.2 VAE decoder per-op f32 primitives
    "MSL_IM2COL_F32",
    "MSL_GROUPNORM_F32",
    "MSL_SILU_F32",
    "MSL_UPSAMPLE_NEAREST_F32",
    "MSL_CONV2D_F32_IMPLICIT",
    // FLUX.2 DiT joint attention (flash-attention simdgroup_matrix)
    "MSL_DIT_JOINT_ATTENTION_FLASH",
];

/// Extract actively-used MSL raw string literals from the concatenated
/// `kernel_sources/` module content.
///
/// Only includes constants in the [`ACTIVE_KERNELS`] whitelist, matching the
/// kernels used by `build_combined_msl()` in `metal_graph/pipelines.rs`.
/// Historical/experimental kernels (in `archive.rs`, `fp8.rs`,
/// `fp8_prefill.rs`, etc.) are kept in source for documentation but excluded
/// here to reduce shader compilation time.
///
/// # Panics
///
/// Panics (failing the build with a clear message) if any name in
/// [`ACTIVE_KERNELS`] is not found as a `pub const NAME: &str = r#"..."#`
/// declaration anywhere in `source`. This is intentional: a silent miss here
/// previously produced an incomplete metallib that the runtime treats as
/// broken (see the module doc), so drift must be a hard build error, not a
/// quiet degradation.
#[cfg(target_os = "macos")]
fn extract_and_combine_msl(source: &str) -> String {
    let mut combined = String::with_capacity(source.len() / 2);

    for kernel_name in ACTIVE_KERNELS {
        // Find `pub const MSL_XXX: &str = r#"`
        let pattern = format!("pub const {kernel_name}: &str = r#\"");
        let start_idx = source.find(&pattern).unwrap_or_else(|| {
            panic!(
                "oxibonsai-kernels build.rs: ACTIVE_KERNELS entry `{kernel_name}` was not \
                 found in any file under `{KERNEL_SOURCES_DIR}/`. This whitelist must stay in \
                 sync with `build_combined_msl()` in \
                 `src/gpu_backend/metal_graph/pipelines.rs` — update one or both."
            )
        });
        let content_start = start_idx + pattern.len();
        // Find the closing `"#`
        let end_offset = source[content_start..].find("\"#").unwrap_or_else(|| {
            panic!(
                "oxibonsai-kernels build.rs: unterminated raw string literal for \
                 `{kernel_name}` (no closing `\"#` found)."
            )
        });
        let content_end = content_start + end_offset;
        combined.push_str(&source[content_start..content_end]);
        combined.push('\n');
    }

    combined
}
