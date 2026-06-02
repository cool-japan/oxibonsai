/// Build script for oxibonsai-kernels: pre-compile Metal shaders into a metallib.
///
/// On macOS with the Metal Toolchain installed:
///   1. Reads kernel_sources.rs and extracts all MSL raw string constants
///   2. Concatenates them into a single .metal file
///   3. Compiles via `xcrun -sdk macosx metal` → AIR → `xcrun metallib`
///   4. Writes the metallib to OUT_DIR for `include_bytes!()` consumption
///
/// On non-macOS or without the Metal Toolchain: writes an empty metallib stub,
/// and the Rust code falls back to runtime MSL compilation.
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/gpu_backend/kernel_sources.rs");

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

#[cfg(target_os = "macos")]
fn try_compile_metal_shaders(out_dir: &str) -> bool {
    let ks_path = Path::new("src/gpu_backend/kernel_sources.rs");
    let ks_content = match std::fs::read_to_string(ks_path) {
        Ok(c) => c,
        Err(_) => return false,
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

/// Extract actively-used MSL raw string literals from kernel_sources.rs.
///
/// Only includes constants in the ACTIVE_KERNELS whitelist, matching the
/// kernels used by `build_combined_msl()` in `metal_graph.rs`.
/// Historical/experimental kernels are kept in source for documentation
/// but excluded to halve shader compilation time.
#[cfg(target_os = "macos")]
fn extract_and_combine_msl(source: &str) -> String {
    /// MSL constant names that are actively used in the dispatch pipeline.
    const ACTIVE_KERNELS: &[&str] = &[
        // Decode path
        "MSL_GEMV_Q1_G128_V7",
        "MSL_GEMV_Q1_G128_V7_RESIDUAL",
        "MSL_RMSNORM_WEIGHTED_V2",
        "MSL_SWIGLU_FUSED",
        "MSL_RESIDUAL_ADD",
        "MSL_FUSED_QK_NORM",
        "MSL_FUSED_QK_ROPE",
        "MSL_FUSED_KV_STORE",
        "MSL_FUSED_GATE_UP_SWIGLU_Q1",
        "MSL_BATCHED_ATTENTION_SCORES",
        "MSL_BATCHED_SOFTMAX",
        "MSL_BATCHED_ATTENTION_WEIGHTED_SUM",
        "MSL_ARGMAX",
        // Prefill path
        "MSL_BATCHED_RMSNORM_V2",
        "MSL_BATCHED_SWIGLU",
        "MSL_GEMM_Q1_G128_V7",
        "MSL_GEMM_Q1_G128_V7_RESIDUAL",
        "MSL_FUSED_GATE_UP_SWIGLU_GEMM_Q1",
    ];

    let mut combined = String::with_capacity(source.len() / 2);

    for kernel_name in ACTIVE_KERNELS {
        // Find `pub const MSL_XXX: &str = r#"`
        let pattern = format!("pub const {kernel_name}: &str = r#\"");
        if let Some(start_idx) = source.find(&pattern) {
            let content_start = start_idx + pattern.len();
            // Find the closing `"#`
            if let Some(end_offset) = source[content_start..].find("\"#") {
                let content_end = content_start + end_offset;
                combined.push_str(&source[content_start..content_end]);
                combined.push('\n');
            }
        }
    }

    combined
}
