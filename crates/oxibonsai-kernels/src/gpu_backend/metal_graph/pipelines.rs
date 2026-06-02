//! Compile-time MSL kernel pipeline construction and caching.
//!
//! Concatenates the actively-used MSL kernel sources into a single library,
//! compiles them via `xcrun metal` (or runtime fallback), caches the resulting
//! `.metallib` on disk, and extracts each compute pipeline by entry-point name.

use metal::{CompileOptions, ComputePipelineState, Device, Library};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use crate::gpu_backend::kernel_sources;

use super::error::MetalGraphError;

// ═══════════════════════════════════════════════════════════════════════════
// Pre-compiled pipeline states
// ═══════════════════════════════════════════════════════════════════════════

/// All kernel pipeline states compiled from a single MSL library.
///
/// Only actively-used kernels are compiled.  Historical/experimental
/// kernel MSL constants are kept in `kernel_sources.rs` for reference
/// but excluded from the combined MSL to halve shader compilation time.
pub(crate) struct MetalPipelines {
    // ── Decode path (single-token) ──────────────────────────────────
    // V7: fully unrolled inner loop (current active)
    pub(crate) gemv_q1_g128_v7: ComputePipelineState,
    pub(crate) gemv_q1_g128_v7_residual: ComputePipelineState,

    // Activation / norm
    pub(crate) rmsnorm_weighted_v2: ComputePipelineState,
    pub(crate) residual_add: ComputePipelineState,
    // Fused kernels (dispatch reduction)
    pub(crate) fused_qk_norm: ComputePipelineState,
    pub(crate) fused_qk_rope: ComputePipelineState,
    pub(crate) fused_qk_norm_rope: ComputePipelineState,
    pub(crate) fused_kv_store: ComputePipelineState,
    pub(crate) fused_gate_up_swiglu_q1: ComputePipelineState,
    // Batched attention kernels (multi-head, GQA-aware)
    pub(crate) batched_attention_scores_v2: ComputePipelineState,
    pub(crate) batched_softmax: ComputePipelineState,
    pub(crate) batched_attention_weighted_sum: ComputePipelineState,

    // GPU argmax for greedy decoding
    pub(crate) argmax: ComputePipelineState,
    // ── Prefill path (batch) ────────────────────────────────────────
    pub(crate) batched_rmsnorm_v2: ComputePipelineState,
    pub(crate) batched_swiglu: ComputePipelineState,
    pub(crate) gemm_q1_g128_v7: ComputePipelineState,
    pub(crate) gemm_q1_g128_v7_residual: ComputePipelineState,
    pub(crate) fused_gate_up_swiglu_gemm_q1: ComputePipelineState,

    // ── Ternary (TQ2_0_g128) ────────────────────────────────────────
    pub(crate) gemv_tq2_g128_v1: ComputePipelineState,
    /// Batched ternary GEMM (prefill path).  Mirrors `gemm_q1_g128_v7`'s
    /// dispatch shape but decodes TQ2_0_g128 weights and supports arbitrary
    /// batch sizes (Q1's V7 silently caps at 8 columns).
    pub(crate) gemm_tq2_g128_v7: ComputePipelineState,
    /// Tiled batched ternary GEMM for the **large-M** path (DiT, `M` up to
    /// 1536).  2-D grid `[ceil(N/8), ceil(M/32)]` with register-blocked
    /// `TN×TM` micro-tiles; parallelizes `M` and decodes each weight block
    /// once per M-tile (vs `v7`'s serial `M/8` re-decodes).  Numerically
    /// equivalent to `gemm_tq2_g128_v7`.  Retained as a fallback now that
    /// `encode_gemm_tq2` dispatches the faster `simdgroup_matrix`
    /// `gemm_tq2_g128_v9_simdgroup`; still compiled (and exercised by the v8/v9
    /// benches), hence `#[allow(dead_code)]`.
    #[allow(dead_code)]
    pub(crate) gemm_tq2_g128_v8_tiled: ComputePipelineState,
    /// `simdgroup_matrix` (8×8×8 HW MAC) batched ternary GEMM for the
    /// **large-M** path (DiT).  Same op / SoA weight buffer as `v8`, but
    /// dequantizes the weight transposed into threadgroup memory and drives
    /// Apple's `simdgroup_float8x8` matrix units (f32 accumulate, f16 staged
    /// operands).  Numerically equivalent to `v7` / `v8` within the
    /// `dit_parity` cosine gate; used only by `encode_gemm_tq2`.
    pub(crate) gemm_tq2_g128_v9_simdgroup: ComputePipelineState,
    /// Staging-optimized `simdgroup_matrix` batched ternary GEMM (`v10`) for the
    /// **large-M** path (DiT).  Same op / SoA weight buffer / decode bits as
    /// `v9`, but stages the dequantized weight as `half` (exact for ternary
    /// `code×scale`), vectorizes the dequant-scatter across all 128 threads, and
    /// double-buffers the K-tile staging so the matrix units overlap the staging
    /// latency.  Numerically equivalent to `v7`/`v8`/`v9` within the `dit_parity`
    /// cosine gate; dispatched by `encode_gemm_tq2` only when it both passes
    /// parity and beats `v9` on the back-to-back ratio.
    pub(crate) gemm_tq2_g128_v10_simdgroup: ComputePipelineState,

    // ── f32-exact (text encoder, Qwen3-4B) ──────────────────────────
    /// f32-exact `simdgroup_matrix` GEMM for the large-M **text-encoder** path.
    /// Computes `out[M,N] = A[M,K] · W[N,K]ᵀ` over pure-f32 weights (the FLUX.2
    /// TE has no quantized format), reusing `v9`'s tile geometry / A-staging /
    /// matrix-MAC structure but loading the f32 weight tile directly (no
    /// dequant). Numerically equivalent to the CPU `gemm_abt` (cos ≈ 1.0);
    /// dispatched by `encode_gemm_f32` only.
    pub(crate) gemm_f32_simdgroup: ComputePipelineState,

    // ── FLUX.2 VAE decoder per-op f32 primitives ────────────────────
    /// im2col patch extraction `[rows, kH·kW·C_in]` in `(kH,kW,C_in)` order
    /// (feeds `gemm_f32_simdgroup` for the k≥3 VAE convs). Dispatched by
    /// `encode_conv2d_f32`.
    pub(crate) im2col_f32: ComputePipelineState,
    /// PyTorch-compatible GroupNorm (32 groups, eps 1e-6, per-channel affine,
    /// Kahan-compensated f32 reduction). Dispatched by `encode_groupnorm_f32`.
    pub(crate) groupnorm_f32: ComputePipelineState,
    /// Element-wise SiLU. Dispatched by `encode_silu_f32`.
    pub(crate) silu_f32: ComputePipelineState,
    /// Nearest ×2 upsample `[C,H,W] → [C,2H,2W]`. Dispatched by
    /// `encode_upsample_nearest_f32`.
    pub(crate) upsample_nearest_f32: ComputePipelineState,
    /// **im2col-free implicit-GEMM** Conv2d (`k=3, pad=1, stride=1`): gathers conv
    /// patches on-the-fly into threadgroup memory and drives `simdgroup_float8x8`
    /// HW MACs (f32 accumulate), so the im2col patch matrix never hits global
    /// memory. Dispatched by `encode_conv2d_f32` for the high-res VAE convs;
    /// numerically equivalent to the `im2col_f32` + `gemm_f32_simdgroup` path
    /// (reassociated sums only), which is retained as a fallback.
    pub(crate) conv2d_f32_implicit: ComputePipelineState,

    // ── FLUX.2 DiT joint attention ──────────────────────────────────
    /// Flash-attention (online-softmax) DiT joint (txt+img) attention driving
    /// `simdgroup_float8x8` HW matrix units for both attention matmuls (`Q·Kᵀ`
    /// and `P·V`), f32 accumulate, writing the token-major transposed output.
    /// Built to beat the rayon+NEON CPU at the DiT shape; the shipping path.
    /// Dispatched by `dispatch_joint_attention_flash` (`encode_joint_attention_flash`
    /// / `encode_joint_attention_flash_pooled`).
    pub(crate) joint_attention_flash_f32: ComputePipelineState,
}

impl MetalPipelines {
    /// Compile the combined MSL source and extract individual pipelines.
    ///
    /// Tries to load a cached `.metallib` from `~/.cache/oxibonsai/` first.
    /// If no cache is found, compiles MSL via `xcrun metal` + `xcrun metallib`
    /// to produce a binary metallib (cached for next run).  Falls back to
    /// runtime `new_library_with_source()` if `xcrun` is unavailable.
    pub(super) fn compile(device: &Device) -> Result<Self, MetalGraphError> {
        // Concatenate all kernel sources into a single MSL string.
        let combined_src = build_combined_msl();

        let library = load_or_compile_library(device, &combined_src)?;

        // Decode path
        let gemv_q1_g128_v7 = pipeline_for(&library, device, "gemv_q1_g128_v7")?;
        let gemv_q1_g128_v7_residual = pipeline_for(&library, device, "gemv_q1_g128_v7_residual")?;
        let rmsnorm_weighted_v2 = pipeline_for(&library, device, "rmsnorm_weighted_v2")?;
        let residual_add = pipeline_for(&library, device, "residual_add")?;
        let fused_qk_norm = pipeline_for(&library, device, "fused_qk_norm")?;
        let fused_qk_rope = pipeline_for(&library, device, "fused_qk_rope")?;
        let fused_qk_norm_rope = pipeline_for(&library, device, "fused_qk_norm_rope")?;
        let fused_kv_store = pipeline_for(&library, device, "fused_kv_store")?;
        let fused_gate_up_swiglu_q1 = pipeline_for(&library, device, "fused_gate_up_swiglu_q1")?;
        let batched_attention_scores_v2 =
            pipeline_for(&library, device, "batched_attention_scores_v2")?;
        let batched_softmax = pipeline_for(&library, device, "batched_softmax")?;
        let batched_attention_weighted_sum =
            pipeline_for(&library, device, "batched_attention_weighted_sum")?;
        let argmax = pipeline_for(&library, device, "argmax")?;
        // Prefill path
        let batched_rmsnorm_v2 = pipeline_for(&library, device, "batched_rmsnorm_v2")?;
        let batched_swiglu = pipeline_for(&library, device, "batched_swiglu")?;
        let gemm_q1_g128_v7 = pipeline_for(&library, device, "gemm_q1_g128_v7")?;
        let gemm_q1_g128_v7_residual = pipeline_for(&library, device, "gemm_q1_g128_v7_residual")?;
        let fused_gate_up_swiglu_gemm_q1 =
            pipeline_for(&library, device, "fused_gate_up_swiglu_gemm_q1")?;
        let gemv_tq2_g128_v1 = pipeline_for(&library, device, "gemv_tq2_g128_v1")?;
        let gemm_tq2_g128_v7 = pipeline_for(&library, device, "gemm_tq2_g128_v7")?;
        let gemm_tq2_g128_v8_tiled = pipeline_for(&library, device, "gemm_tq2_g128_v8_tiled")?;
        let gemm_tq2_g128_v9_simdgroup =
            pipeline_for(&library, device, "gemm_tq2_g128_v9_simdgroup")?;
        let gemm_tq2_g128_v10_simdgroup =
            pipeline_for(&library, device, "gemm_tq2_g128_v10_simdgroup")?;
        let gemm_f32_simdgroup = pipeline_for(&library, device, "gemm_f32_simdgroup")?;
        // VAE decoder per-op f32 primitives
        let im2col_f32 = pipeline_for(&library, device, "im2col_f32")?;
        let groupnorm_f32 = pipeline_for(&library, device, "groupnorm_f32")?;
        let silu_f32 = pipeline_for(&library, device, "silu_f32")?;
        let upsample_nearest_f32 = pipeline_for(&library, device, "upsample_nearest_f32")?;
        let conv2d_f32_implicit = pipeline_for(&library, device, "conv2d_f32_implicit")?;
        // DiT joint attention (flash-attention simdgroup_matrix — shipping path)
        let joint_attention_flash_f32 =
            pipeline_for(&library, device, "joint_attention_flash_f32")?;

        Ok(Self {
            gemv_q1_g128_v7,
            gemv_q1_g128_v7_residual,
            rmsnorm_weighted_v2,
            residual_add,
            fused_qk_norm,
            fused_qk_rope,
            fused_qk_norm_rope,
            fused_kv_store,
            fused_gate_up_swiglu_q1,
            batched_attention_scores_v2,
            batched_softmax,
            batched_attention_weighted_sum,
            argmax,
            batched_rmsnorm_v2,
            batched_swiglu,
            gemm_q1_g128_v7,
            gemm_q1_g128_v7_residual,
            fused_gate_up_swiglu_gemm_q1,
            gemv_tq2_g128_v1,
            gemm_tq2_g128_v7,
            gemm_tq2_g128_v8_tiled,
            gemm_tq2_g128_v9_simdgroup,
            gemm_tq2_g128_v10_simdgroup,
            gemm_f32_simdgroup,
            im2col_f32,
            groupnorm_f32,
            silu_f32,
            upsample_nearest_f32,
            conv2d_f32_implicit,
            joint_attention_flash_f32,
        })
    }
}

/// Extract a named compute pipeline from a compiled library.
fn pipeline_for(
    library: &Library,
    device: &Device,
    name: &str,
) -> Result<ComputePipelineState, MetalGraphError> {
    let func = library
        .get_function(name, None)
        .map_err(|e| MetalGraphError::EncodingFailed(format!("function '{name}': {e}")))?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| MetalGraphError::CompilationFailed(format!("pipeline '{name}': {e}")))
}

/// Build a single MSL string containing only the actively-used kernels.
///
/// Historical/experimental kernel constants (V1–V6, V8–V10, old GEMM, etc.)
/// are kept in `kernel_sources.rs` for documentation but excluded here
/// to reduce shader compilation time (~4000 → ~2000 MSL lines).
fn build_combined_msl() -> String {
    let mut src = String::with_capacity(16384);
    // ── Decode path (single-token) ──────────────────────────────────────
    src.push_str(kernel_sources::MSL_GEMV_Q1_G128_V7);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMV_Q1_G128_V7_RESIDUAL);
    src.push('\n');

    src.push_str(kernel_sources::MSL_RMSNORM_WEIGHTED_V2);
    src.push('\n');
    src.push_str(kernel_sources::MSL_RESIDUAL_ADD);
    src.push('\n');
    src.push_str(kernel_sources::MSL_FUSED_QK_NORM);
    src.push('\n');
    src.push_str(kernel_sources::MSL_FUSED_QK_ROPE);
    src.push('\n');
    src.push_str(kernel_sources::MSL_FUSED_QK_NORM_ROPE);
    src.push('\n');
    src.push_str(kernel_sources::MSL_FUSED_KV_STORE);
    src.push('\n');
    src.push_str(kernel_sources::MSL_FUSED_GATE_UP_SWIGLU_Q1);
    src.push('\n');

    src.push_str(kernel_sources::MSL_BATCHED_ATTENTION_SCORES_V2);
    src.push('\n');
    src.push_str(kernel_sources::MSL_BATCHED_SOFTMAX);
    src.push('\n');
    src.push_str(kernel_sources::MSL_BATCHED_ATTENTION_WEIGHTED_SUM);
    src.push('\n');
    src.push_str(kernel_sources::MSL_ARGMAX);
    src.push('\n');
    // ── Prefill path (batch) ────────────────────────────────────────────
    src.push_str(kernel_sources::MSL_BATCHED_RMSNORM_V2);
    src.push('\n');
    src.push_str(kernel_sources::MSL_BATCHED_SWIGLU);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMM_Q1_G128_V7);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMM_Q1_G128_V7_RESIDUAL);
    src.push('\n');
    src.push_str(kernel_sources::MSL_FUSED_GATE_UP_SWIGLU_GEMM_Q1);
    src.push('\n');
    // ── Ternary (TQ2_0_g128) ────────────────────────────────────────────
    src.push_str(kernel_sources::MSL_GEMV_TQ2_G128_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMM_TQ2_G128_V7);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMM_TQ2_G128_V8_TILED);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMM_TQ2_G128_V9_SIMDGROUP);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMM_TQ2_G128_V10_SIMDGROUP);
    src.push('\n');
    // ── f32-exact (text encoder) ────────────────────────────────────────
    src.push_str(kernel_sources::MSL_GEMM_F32_SIMDGROUP);
    src.push('\n');
    // ── FLUX.2 VAE decoder per-op f32 primitives ─────────────────────────
    src.push_str(kernel_sources::MSL_IM2COL_F32);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GROUPNORM_F32);
    src.push('\n');
    src.push_str(kernel_sources::MSL_SILU_F32);
    src.push('\n');
    src.push_str(kernel_sources::MSL_UPSAMPLE_NEAREST_F32);
    src.push('\n');
    src.push_str(kernel_sources::MSL_CONV2D_F32_IMPLICIT);
    src.push('\n');
    // ── FLUX.2 DiT joint attention (flash-attention simdgroup_matrix) ─────
    src.push_str(kernel_sources::MSL_DIT_JOINT_ATTENTION_FLASH);
    src.push('\n');
    src
}

// ═══════════════════════════════════════════════════════════════════════════
// Pre-compiled metallib caching
// ═══════════════════════════════════════════════════════════════════════════

/// Compute a 64-bit hash of the combined MSL source for cache keying.
fn msl_hash(msl_source: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    msl_source.hash(&mut hasher);
    hasher.finish()
}

/// Return the cache directory for pre-compiled metallibs: `~/.cache/oxibonsai/`.
fn metallib_cache_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".cache").join("oxibonsai"))
}

/// Try to load a cached `.metallib` from disk.
fn try_load_cached_metallib(device: &Device, cache_path: &std::path::Path) -> Option<Library> {
    let data = std::fs::read(cache_path).ok()?;
    tracing::debug!(
        "loading cached metallib ({} bytes) from {}",
        data.len(),
        cache_path.display()
    );
    device.new_library_with_data(&data).ok()
}

/// Compile MSL source to a `.metallib` binary via `xcrun metal` + `xcrun metallib`,
/// cache the result to `cache_path`, and load the library.
fn compile_msl_via_xcrun(
    device: &Device,
    msl_source: &str,
    cache_path: &std::path::Path,
) -> Option<Library> {
    let tmp_dir = std::env::temp_dir().join("oxibonsai_metal_build");
    if std::fs::create_dir_all(&tmp_dir).is_err() {
        return None;
    }

    let metal_path = tmp_dir.join("combined.metal");
    let air_path = tmp_dir.join("combined.air");
    let metallib_path = tmp_dir.join("combined.metallib");

    if std::fs::write(&metal_path, msl_source).is_err() {
        return None;
    }

    // Step 1: MSL → AIR (Apple Intermediate Representation)
    let metal_src_str = metal_path.to_str()?;
    let air_str = air_path.to_str()?;
    let output = std::process::Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            metal_src_str,
            "-o",
            air_str,
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::debug!(
            "xcrun metal compilation failed: {}",
            &stderr[..stderr.len().min(500)]
        );
        return None;
    }

    // Step 2: AIR → metallib
    let metallib_str = metallib_path.to_str()?;
    let output = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", air_str, "-o", metallib_str])
        .output()
        .ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::debug!("xcrun metallib linking failed: {stderr}");
        return None;
    }

    let metallib_data = std::fs::read(&metallib_path).ok()?;
    tracing::info!(
        "compiled metallib via xcrun ({} bytes), caching to {}",
        metallib_data.len(),
        cache_path.display()
    );

    // Cache for future runs
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(cache_path, &metallib_data);

    // Clean up temp files
    let _ = std::fs::remove_file(&metal_path);
    let _ = std::fs::remove_file(&air_path);
    let _ = std::fs::remove_file(&metallib_path);
    let _ = std::fs::remove_dir(&tmp_dir);

    device.new_library_with_data(&metallib_data).ok()
}

/// Compile MSL source at runtime using `device.new_library_with_source()`.
fn compile_msl_runtime(device: &Device, msl_source: &str) -> Result<Library, MetalGraphError> {
    tracing::debug!("falling back to runtime MSL compilation");
    let options = CompileOptions::new();
    device
        .new_library_with_source(msl_source, &options)
        .map_err(MetalGraphError::CompilationFailed)
}

/// Pre-compiled metallib bytes embedded at build time.
///
/// If the Metal Toolchain is available during `cargo build`, `build.rs`
/// compiles all MSL kernels into a `.metallib` and this constant contains
/// the binary data.  Otherwise it is an empty slice and the runtime
/// falls back to MSL compilation.
static PRECOMPILED_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/combined.metallib"));

/// Try loading the build-time pre-compiled metallib.
fn try_load_embedded_metallib(device: &Device) -> Option<Library> {
    if PRECOMPILED_METALLIB.is_empty() {
        return None;
    }
    tracing::info!(
        "loading build-time pre-compiled metallib ({} bytes)",
        PRECOMPILED_METALLIB.len()
    );
    device.new_library_with_data(PRECOMPILED_METALLIB).ok()
}

/// Load a Metal library: embedded metallib → cached metallib → xcrun → runtime compilation.
fn load_or_compile_library(device: &Device, msl_source: &str) -> Result<Library, MetalGraphError> {
    // 1. Try build-time embedded metallib (fastest: no I/O, no compilation)
    if let Some(lib) = try_load_embedded_metallib(device) {
        return Ok(lib);
    }

    let hash = msl_hash(msl_source);
    let cache_filename = format!("kernels_{hash:016x}.metallib");

    // 2. Try disk-cached metallib from a previous xcrun run
    if let Some(cache_dir) = metallib_cache_dir() {
        let cache_path = cache_dir.join(&cache_filename);

        if let Some(lib) = try_load_cached_metallib(device, &cache_path) {
            tracing::info!("loaded pre-compiled metallib from cache (hash={hash:016x})");
            return Ok(lib);
        }

        // 3. Try xcrun offline compilation + caching
        if let Some(lib) = compile_msl_via_xcrun(device, msl_source, &cache_path) {
            return Ok(lib);
        }
    }

    // 4. Final fallback: runtime compilation (no caching possible)
    compile_msl_runtime(device, msl_source)
}
