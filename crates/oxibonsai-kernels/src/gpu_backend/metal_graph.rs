//! Direct Metal dispatch engine for OxiBonsai FFN pipeline.
//!
//! Bypasses scirs2-core's abstraction layer and encodes all FFN operations
//! into a single command buffer with a single compute encoder, following the
//! llama.cpp architecture pattern.
//!
//! # Architecture
//!
//! - Single `metal::Device` (system default, singleton)
//! - Dedicated `metal::CommandQueue` per graph
//! - Pre-compiled compute pipeline states from concatenated MSL sources
//! - Lazily pre-allocated intermediate GPU buffers (shared mode + hazard tracking)
//!
//! # Buffer hazard tracking
//!
//! All CPU-accessible buffers use `MTLResourceOptions::StorageModeShared` with default
//! (tracked) hazard tracking mode.  With a non-concurrent compute encoder,
//! Metal automatically inserts memory barriers for read-after-write
//! dependencies, so explicit `memory_barrier_with_resources` calls are
//! not required.

#![cfg(feature = "metal")]

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions,
};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::kernel_sources;

// ═══════════════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════════════

/// Errors raised by the Metal graph dispatch engine.
#[derive(Debug)]
pub enum MetalGraphError {
    /// No Metal-capable GPU device was found on the system.
    DeviceNotFound,
    /// MSL shader compilation failed.
    CompilationFailed(String),
    /// A GPU buffer could not be allocated.
    BufferCreationFailed,
    /// An encoding operation failed (pipeline not found, etc.).
    EncodingFailed(String),
    /// A command buffer execution failed or timed out.
    ExecutionFailed(String),
}

impl fmt::Display for MetalGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound => write!(f, "no Metal-capable GPU device found"),
            Self::CompilationFailed(msg) => write!(f, "MSL compilation failed: {msg}"),
            Self::BufferCreationFailed => write!(f, "Metal buffer allocation failed"),
            Self::EncodingFailed(msg) => write!(f, "Metal encoding failed: {msg}"),
            Self::ExecutionFailed(msg) => write!(f, "Metal execution failed: {msg}"),
        }
    }
}

impl std::error::Error for MetalGraphError {}

// ═══════════════════════════════════════════════════════════════════════════
// Weight handle
// ═══════════════════════════════════════════════════════════════════════════

/// Opaque handle to a weight buffer already resident on the GPU.
///
/// Stores the raw `metal::Buffer` directly so the graph can bind it
/// without going through any abstraction layer.
pub struct MetalWeightHandle {
    /// Raw Metal buffer containing packed weight data.
    pub(crate) buffer: Buffer,
    /// Size in bytes.
    pub(crate) byte_len: usize,
}

impl MetalWeightHandle {
    /// Size of the weight data in bytes.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

impl fmt::Debug for MetalWeightHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalWeightHandle")
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q1 AoS → SoA reformatter
// ═══════════════════════════════════════════════════════════════════════════

/// Reformat Q1_0_g128 weight bytes from AoS to SoA layout.
///
/// AoS (input):  [scale₀|data₀][scale₁|data₁]...[scaleₙ|dataₙ]
///   Each block: 2 bytes FP16 scale + 16 bytes sign data = 18 bytes
///
/// SoA (output): [scale₀|scale₁|...|scaleₙ][data₀|data₁|...|dataₙ]
///   Scales section: N × 2 bytes (sequential, perfectly coalesced)
///   Data section:   N × 16 bytes (16-byte aligned, uint4 loads)
///
/// Total size is unchanged: N × 18 bytes.
/// Returns `None` if the input length is not a multiple of 18.
fn reformat_q1_aos_to_soa(aos_bytes: &[u8]) -> Option<Vec<u8>> {
    const BLOCK_SIZE: usize = 18;
    const SCALE_SIZE: usize = 2;
    const DATA_SIZE: usize = 16;

    if aos_bytes.is_empty() || aos_bytes.len() % BLOCK_SIZE != 0 {
        return None;
    }

    let n_blocks = aos_bytes.len() / BLOCK_SIZE;
    let mut soa = vec![0u8; n_blocks * BLOCK_SIZE];

    let (scales_section, data_section) = soa.split_at_mut(n_blocks * SCALE_SIZE);

    for i in 0..n_blocks {
        let block_start = i * BLOCK_SIZE;
        // Copy scale (2 bytes) to scales section
        scales_section[i * SCALE_SIZE..i * SCALE_SIZE + SCALE_SIZE]
            .copy_from_slice(&aos_bytes[block_start..block_start + SCALE_SIZE]);
        // Copy data (16 bytes) to data section
        data_section[i * DATA_SIZE..i * DATA_SIZE + DATA_SIZE]
            .copy_from_slice(&aos_bytes[block_start + SCALE_SIZE..block_start + BLOCK_SIZE]);
    }

    Some(soa)
}

/// Reformat TQ2_0_g128 AoS → SoA for the Metal ternary GEMV kernel.
///
/// AoS (input) block layout: `{ qs: [u8; 32], d: f16 }` = 34 bytes
/// SoA (output) layout: `[all d: N × 2 bytes FP16 LE][all qs: N × 32 bytes]`
/// Note: the ternary MSL kernel consumes this layout directly — scales first,
/// then qs data, matching the convention in `scirs2_backend::upload_weights_ternary`.
///
/// Total size is unchanged: N × 34 bytes.
/// Returns `None` if the input length is not a multiple of 34.
fn reformat_tq2_aos_to_soa(aos_bytes: &[u8]) -> Option<Vec<u8>> {
    const BLOCK_SIZE: usize = 34;
    const SCALE_SIZE: usize = 2;
    const DATA_SIZE: usize = 32;

    if aos_bytes.is_empty() || aos_bytes.len() % BLOCK_SIZE != 0 {
        return None;
    }

    let n_blocks = aos_bytes.len() / BLOCK_SIZE;
    let mut soa = vec![0u8; n_blocks * BLOCK_SIZE];

    let (scales_section, data_section) = soa.split_at_mut(n_blocks * SCALE_SIZE);

    for i in 0..n_blocks {
        let block_start = i * BLOCK_SIZE;
        // TQ2 AoS: first 32 bytes are qs, last 2 are scale. SoA: scales first, then qs.
        data_section[i * DATA_SIZE..i * DATA_SIZE + DATA_SIZE]
            .copy_from_slice(&aos_bytes[block_start..block_start + DATA_SIZE]);
        scales_section[i * SCALE_SIZE..i * SCALE_SIZE + SCALE_SIZE]
            .copy_from_slice(&aos_bytes[block_start + DATA_SIZE..block_start + BLOCK_SIZE]);
    }

    Some(soa)
}

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
}

impl MetalPipelines {
    /// Compile the combined MSL source and extract individual pipelines.
    ///
    /// Tries to load a cached `.metallib` from `~/.cache/oxibonsai/` first.
    /// If no cache is found, compiles MSL via `xcrun metal` + `xcrun metallib`
    /// to produce a binary metallib (cached for next run).  Falls back to
    /// runtime `new_library_with_source()` if `xcrun` is unavailable.
    fn compile(device: &Device) -> Result<Self, MetalGraphError> {
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

// ═══════════════════════════════════════════════════════════════════════════
// Pre-allocated GPU buffers for the FFN pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Lazily allocated intermediate buffers used by `encode_ffn_phase`.
struct MetalBuffers {
    hidden_buf: Buffer,
    attn_out_buf: Buffer,
    norm_weight_buf: Buffer,
    proj_buf: Buffer,
    normed_buf: Buffer,
    swiglu_buf: Buffer,
    down_buf: Buffer,
    /// Hidden dimension these buffers were allocated for.
    hidden_size: usize,
    /// Intermediate dimension (gate/up half size).
    intermediate_size: usize,
}

impl MetalBuffers {
    /// Allocate all intermediate buffers for the given dimensions.
    fn allocate(
        device: &Device,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, MetalGraphError> {
        let h_bytes = (hidden_size * std::mem::size_of::<f32>()) as u64;
        let inter_bytes = (intermediate_size * std::mem::size_of::<f32>()) as u64;
        let shared = MTLResourceOptions::StorageModeShared;
        let private = MTLResourceOptions::StorageModePrivate;

        Ok(Self {
            hidden_buf: alloc_buf(device, h_bytes, shared)?, // CPU upload/download
            attn_out_buf: alloc_buf(device, h_bytes, shared)?, // CPU upload
            norm_weight_buf: alloc_buf(device, h_bytes, shared)?, // CPU upload
            proj_buf: alloc_buf(device, h_bytes, private)?,  // GPU-only intermediate
            normed_buf: alloc_buf(device, h_bytes, private)?, // GPU-only intermediate
            swiglu_buf: alloc_buf(device, inter_bytes, private)?, // GPU-only intermediate

            down_buf: alloc_buf(device, h_bytes, private)?, // GPU-only intermediate
            hidden_size,
            intermediate_size,
        })
    }

    /// Check whether existing buffers match the requested dimensions.
    fn matches(&self, hidden_size: usize, intermediate_size: usize) -> bool {
        self.hidden_size == hidden_size && self.intermediate_size == intermediate_size
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pre-allocated GPU buffers for the attention pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: allocate a Metal buffer, converting a null pointer into an error.
pub(crate) fn alloc_buf(
    device: &Device,
    byte_len: u64,
    opts: MTLResourceOptions,
) -> Result<Buffer, MetalGraphError> {
    if byte_len == 0 {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    let buf = device.new_buffer(byte_len, opts);
    // StorageModePrivate buffers have contents() == null by design
    if opts.contains(MTLResourceOptions::StorageModePrivate) {
        // For private buffers, just check length as a sanity proxy
        if buf.length() < byte_len {
            return Err(MetalGraphError::BufferCreationFailed);
        }
    } else if buf.contents().is_null() {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    Ok(buf)
}

// ═══════════════════════════════════════════════════════════════════════════
// Upload / download helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Copy a host `f32` slice into a shared Metal buffer.
///
/// # Safety
///
/// The buffer must have been allocated with `StorageModeShared` and must be
/// large enough to hold `data.len()` floats.
pub(crate) unsafe fn upload_f32(buf: &Buffer, data: &[f32]) {
    std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len());
}

/// Copy from a shared Metal buffer into a host `f32` slice.
///
/// # Safety
///
/// The buffer must have been allocated with `StorageModeShared` and must
/// contain at least `out.len()` floats of valid data.
pub(crate) unsafe fn download_f32(buf: &Buffer, out: &mut [f32]) {
    std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), out.len());
}

/// Upload raw bytes (weight data) into a GPU-accessible Metal buffer.
///
/// Uses `StorageModeShared` so the CPU can write directly and the GPU
/// can read without an explicit blit copy.
fn upload_bytes(device: &Device, data: &[u8]) -> Result<Buffer, MetalGraphError> {
    if data.is_empty() {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    let opts = MTLResourceOptions::StorageModeShared;
    let buf = device.new_buffer(data.len() as u64, opts);
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut u8, data.len());
    }
    Ok(buf)
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute threadgroup count: `ceil(n / divisor)`, guaranteed >= 1.
#[inline]
pub(crate) fn div_ceil(n: usize, divisor: usize) -> usize {
    n.div_ceil(divisor)
}

/// Convenience: `set_bytes` for a single scalar value at a given buffer index.
///
/// # Safety
///
/// The encoder must be in a valid state and `index` must not collide with
/// any buffer binding.
pub(crate) unsafe fn set_scalar<T: Copy>(
    encoder: &metal::ComputeCommandEncoderRef,
    index: u64,
    value: &T,
) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        value as *const T as *const c_void,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// MetalGraph
// ═══════════════════════════════════════════════════════════════════════════

/// Process-wide singleton for `MetalGraph`.
static GLOBAL_METAL_GRAPH: OnceLock<Mutex<Option<Arc<MetalGraph>>>> = OnceLock::new();

/// Direct Metal dispatch engine for the FFN pipeline.
///
/// Holds a Metal device, command queue, pre-compiled pipeline states, and
/// lazily allocated intermediate buffers.  All FFN operations are encoded
/// into a single command buffer with a single compute encoder, then
/// committed and synchronously waited upon.
pub struct MetalGraph {
    pub(crate) device: Device,
    pub(crate) command_queue: CommandQueue,
    pub(crate) pipelines: MetalPipelines,
    /// Lazily allocated intermediate buffers, protected by a mutex for
    /// interior mutability (buffer contents are mutated on each dispatch).
    buffers: Mutex<Option<MetalBuffers>>,
    /// Lazy cache of GPU-resident weight buffers, keyed by `GpuWeightHandle` id.
    weight_cache: Mutex<HashMap<u64, Arc<MetalWeightHandle>>>,
    /// Lazily allocated KV cache for all layers.
    pub(crate) kv_cache: Mutex<Option<super::metal_full_layer::GpuKvCache>>,
    /// Lazily allocated full-layer intermediate buffers.
    pub(crate) full_layer_buffers: Mutex<Option<super::metal_full_layer::FullLayerBuffers>>,
    /// Lazily allocated logits output buffer for fused LM head dispatch.
    pub(crate) logits_buf: Mutex<Option<Buffer>>,
    /// Persistent 4-byte buffer for GPU argmax token ID output (greedy decoding).
    pub(crate) token_id_buf: Mutex<Option<Buffer>>,
    /// Lazily allocated prefill buffers for batch processing.
    pub(crate) prefill_buffers: Mutex<Option<super::metal_prefill::PrefillBuffers>>,
}

// Metal objects (Device, CommandQueue, etc.) are Send+Sync in the metal crate.
unsafe impl Send for MetalGraph {}
unsafe impl Sync for MetalGraph {}

impl MetalGraph {
    // ─────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────

    /// Create a new `MetalGraph` bound to the system default Metal device.
    ///
    /// Compiles all MSL kernels into pipeline states.  This is an expensive
    /// operation — prefer `global()` for repeated use.
    pub fn new() -> Result<Self, MetalGraphError> {
        let device = Device::system_default().ok_or(MetalGraphError::DeviceNotFound)?;
        let command_queue = device.new_command_queue();
        let pipelines = MetalPipelines::compile(&device)?;

        Ok(Self {
            device,
            command_queue,
            pipelines,
            buffers: Mutex::new(None),
            weight_cache: Mutex::new(HashMap::new()),
            kv_cache: Mutex::new(None),
            full_layer_buffers: Mutex::new(None),
            logits_buf: Mutex::new(None),
            token_id_buf: Mutex::new(None),
            prefill_buffers: Mutex::new(None),
        })
    }

    /// Get or create the process-wide `MetalGraph` singleton.
    pub fn global() -> Result<Arc<Self>, MetalGraphError> {
        let mutex = GLOBAL_METAL_GRAPH.get_or_init(|| Mutex::new(None));
        let mut guard = mutex
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("MetalGraph lock poisoned".into()))?;
        if let Some(ref cached) = *guard {
            return Ok(Arc::clone(cached));
        }
        let graph = Arc::new(Self::new()?);
        *guard = Some(Arc::clone(&graph));
        Ok(graph)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Weight management
    // ─────────────────────────────────────────────────────────────────────

    /// Upload raw packed weight bytes to a GPU-resident Metal buffer.
    ///
    /// The returned handle can be passed to `encode_ffn_phase` or
    /// `encode_gemv` without further copies.
    pub fn upload_weight(&self, data: &[u8]) -> Result<MetalWeightHandle, MetalGraphError> {
        let buffer = upload_bytes(&self.device, data)?;
        Ok(MetalWeightHandle {
            byte_len: data.len(),
            buffer,
        })
    }

    /// Get a cached `MetalWeightHandle` or upload raw bytes and cache it.
    ///
    /// `key` is typically the `GpuWeightHandle`'s `u64` ID.
    pub fn get_or_upload_weight(
        &self,
        key: u64,
        raw_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let handle = Arc::new(self.upload_weight(raw_bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Like `get_or_upload_weight`, but accepts a closure that produces the bytes.
    ///
    /// This avoids unnecessary allocation when the weight is already cached.
    pub fn get_or_upload_weight_lazy(
        &self,
        key: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let bytes = data_fn();
        let handle = Arc::new(self.upload_weight(&bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Upload Q1_0_g128 weight bytes in SoA layout for optimal GPU coalescing.
    ///
    /// Automatically reformats AoS → SoA during upload. The returned handle
    /// contains weights in SoA format ready for V7 kernels.
    pub fn upload_q1_weight_soa(
        &self,
        aos_data: &[u8],
    ) -> Result<MetalWeightHandle, MetalGraphError> {
        let soa_data = reformat_q1_aos_to_soa(aos_data).ok_or_else(|| {
            MetalGraphError::ExecutionFailed(format!(
                "Q1 SoA reformat failed: input length {} is not a multiple of 18",
                aos_data.len()
            ))
        })?;
        let buffer = upload_bytes(&self.device, &soa_data)?;
        Ok(MetalWeightHandle {
            byte_len: soa_data.len(),
            buffer,
        })
    }

    /// Get a cached SoA weight handle or reformat AoS→SoA and upload.
    pub fn get_or_upload_q1_weight_soa(
        &self,
        key: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let handle = Arc::new(self.upload_q1_weight_soa(aos_bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Like `get_or_upload_q1_weight_soa`, but accepts a closure that produces AoS bytes.
    pub fn get_or_upload_q1_weight_soa_lazy(
        &self,
        key: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let aos_bytes = data_fn();
        let handle = Arc::new(self.upload_q1_weight_soa(&aos_bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Upload TQ2_0_g128 (ternary) weight bytes in SoA layout.
    ///
    /// Reformats 34-byte AoS blocks `{ qs:[u8;32], d:f16 }` into SoA
    /// `[N × 2B scales][N × 32B qs]` ready for `gemv_tq2_g128_v1`.
    pub fn upload_tq2_weight_soa(
        &self,
        aos_data: &[u8],
    ) -> Result<MetalWeightHandle, MetalGraphError> {
        let soa_data = reformat_tq2_aos_to_soa(aos_data).ok_or_else(|| {
            MetalGraphError::ExecutionFailed(format!(
                "TQ2 SoA reformat failed: input length {} is not a multiple of 34",
                aos_data.len()
            ))
        })?;
        let buffer = upload_bytes(&self.device, &soa_data)?;
        Ok(MetalWeightHandle {
            byte_len: soa_data.len(),
            buffer,
        })
    }

    /// Get a cached TQ2 SoA weight handle or reformat AoS→SoA and upload.
    pub fn get_or_upload_tq2_weight_soa(
        &self,
        key: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let handle = Arc::new(self.upload_tq2_weight_soa(aos_bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Like `get_or_upload_tq2_weight_soa`, but accepts a closure that produces AoS bytes.
    pub fn get_or_upload_tq2_weight_soa_lazy(
        &self,
        key: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let aos_bytes = data_fn();
        let handle = Arc::new(self.upload_tq2_weight_soa(&aos_bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Single GEMV dispatch
    // ─────────────────────────────────────────────────────────────────────

    /// Execute a single Q1_0_g128 GEMV: `output = weight × input`.
    ///
    /// `weight` must have been uploaded via `upload_weight`.
    /// `input` and `output` are CPU-side f32 slices.
    ///
    /// - `n_rows`: number of output rows (weight matrix rows)
    /// - `k`: number of input elements (weight matrix columns, must be multiple of 128)
    pub fn encode_gemv(
        &self,
        weight: &MetalWeightHandle,
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> Result<(), MetalGraphError> {
        if input.len() < k {
            return Err(MetalGraphError::EncodingFailed(format!(
                "input too short: need {k}, got {}",
                input.len()
            )));
        }
        if output.len() < n_rows {
            return Err(MetalGraphError::EncodingFailed(format!(
                "output too short: need {n_rows}, got {}",
                output.len()
            )));
        }

        let opts = MTLResourceOptions::StorageModeShared;
        let input_bytes = std::mem::size_of_val(input) as u64;
        let output_bytes = (n_rows * std::mem::size_of::<f32>()) as u64;

        let input_buf = alloc_buf(&self.device, input_bytes, opts)?;
        let output_buf = alloc_buf(&self.device, output_bytes, opts)?;

        unsafe { upload_f32(&input_buf, input) };

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        self.dispatch_gemv_q1(
            encoder,
            &weight.buffer,
            &input_buf,
            &output_buf,
            n_rows as u32,
            k as u32,
        );

        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe { download_f32(&output_buf, &mut output[..n_rows]) };

        Ok(())
    }

    /// Execute a single TQ2_0_g128 (ternary) GEMV: `output = weight × input`.
    ///
    /// Mirror of `encode_gemv` for ternary weights. `weight` must have been
    /// uploaded via [`upload_tq2_weight_soa`](Self::upload_tq2_weight_soa).
    pub fn encode_gemv_tq2(
        &self,
        weight: &MetalWeightHandle,
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> Result<(), MetalGraphError> {
        if input.len() < k {
            return Err(MetalGraphError::EncodingFailed(format!(
                "input too short: need {k}, got {}",
                input.len()
            )));
        }
        if output.len() < n_rows {
            return Err(MetalGraphError::EncodingFailed(format!(
                "output too short: need {n_rows}, got {}",
                output.len()
            )));
        }

        let opts = MTLResourceOptions::StorageModeShared;
        let input_bytes = std::mem::size_of_val(input) as u64;
        let output_bytes = (n_rows * std::mem::size_of::<f32>()) as u64;

        let input_buf = alloc_buf(&self.device, input_bytes, opts)?;
        let output_buf = alloc_buf(&self.device, output_bytes, opts)?;

        unsafe { upload_f32(&input_buf, input) };

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        self.dispatch_gemv_tq2(
            encoder,
            &weight.buffer,
            &input_buf,
            &output_buf,
            n_rows as u32,
            k as u32,
        );

        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe { download_f32(&output_buf, &mut output[..n_rows]) };

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // FFN phase dispatch (7 operations, 1 encoder)
    // ─────────────────────────────────────────────────────────────────────

    /// Encode the full FFN phase as 7 sequential operations in one encoder.
    ///
    /// # Data flow
    ///
    /// 1. Upload `hidden`, `attn_out`, `norm_weight` to GPU
    /// 2. GEMV(attn_proj_weight, attn_out) → proj_buf
    /// 3. residual_add(hidden_buf, proj_buf)
    /// 4. rmsnorm_weighted(hidden_buf, norm_weight_buf) → normed_buf
    /// 5. GEMV(gate_up_weight, normed_buf) → gate_up_buf
    /// 6. swiglu_fused(gate_up_buf) → swiglu_buf
    /// 7. GEMV(down_weight, swiglu_buf) → down_buf
    /// 8. residual_add(hidden_buf, down_buf)
    /// 9. Read hidden_buf back to `hidden`
    ///
    /// All operations share one command buffer and one encoder.  Metal's
    /// automatic hazard tracking on shared-mode buffers ensures correct
    /// ordering of read-after-write dependencies.
    ///
    /// # Parameters
    ///
    /// - `hidden`: hidden state (read + written in-place), length = `hidden_size`
    /// - `attn_out`: attention output, length = `hidden_size`
    /// - `norm_weight`: RMSNorm weight vector, length = `hidden_size`
    /// - `attn_proj_weight`: pre-uploaded Q1 weight handle (hidden×hidden)
    /// - `gate_up_weight`: pre-uploaded Q1 weight handle ((intermediate*2)×hidden)
    /// - `down_weight`: pre-uploaded Q1 weight handle (hidden×intermediate)
    /// - `hidden_size`: dimension of the hidden state
    /// - `intermediate_size`: dimension of the MLP intermediate layer
    /// - `eps`: RMSNorm epsilon (typically 1e-6)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ffn_phase(
        &self,
        hidden: &mut [f32],
        attn_out: &[f32],
        norm_weight: &[f32],
        attn_proj_weight: &MetalWeightHandle,
        gate_up_weight: &MetalWeightHandle,
        down_weight: &MetalWeightHandle,
        hidden_size: usize,
        intermediate_size: usize,
        eps: f32,
    ) -> Result<(), MetalGraphError> {
        static FFN_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
        let call_num = FFN_CALL_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        let t_total = Instant::now();

        // ── Validate inputs ──────────────────────────────────────────────
        if hidden.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden too short: need {hidden_size}, got {}",
                hidden.len()
            )));
        }
        if attn_out.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "attn_out too short: need {hidden_size}, got {}",
                attn_out.len()
            )));
        }
        if norm_weight.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "norm_weight too short: need {hidden_size}, got {}",
                norm_weight.len()
            )));
        }

        // ── Ensure intermediate buffers ──────────────────────────────────
        let t0 = Instant::now();
        let guard = self.acquire_buffers(hidden_size, intermediate_size)?;
        let bufs = guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("buffers not allocated".into()))?;
        let dt_acquire = t0.elapsed();

        // ── Step 1: Upload CPU → GPU ─────────────────────────────────────
        let t1 = Instant::now();
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden[..hidden_size]);
            upload_f32(&bufs.attn_out_buf, &attn_out[..hidden_size]);
            upload_f32(&bufs.norm_weight_buf, &norm_weight[..hidden_size]);
        }
        let dt_upload = t1.elapsed();

        // ── Create command buffer + single encoder ───────────────────────
        let t2 = Instant::now();
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        let dt_encode_setup = t2.elapsed();

        let h = hidden_size as u32;
        let inter = intermediate_size as u32;

        // ── Step 2: GEMV(attn_proj, attn_out) → proj_buf ────────────────
        // n_rows = hidden_size, k = hidden_size
        self.dispatch_gemv_q1(
            encoder,
            &attn_proj_weight.buffer,
            &bufs.attn_out_buf,
            &bufs.proj_buf,
            h,
            h,
        );

        // ── Step 3: residual_add(hidden_buf, proj_buf) ───────────────────
        self.dispatch_residual_add(encoder, &bufs.hidden_buf, &bufs.proj_buf, h);

        // ── Step 4: rmsnorm_weighted(hidden_buf, norm_weight_buf) → normed_buf
        self.dispatch_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            &bufs.norm_weight_buf,
            &bufs.normed_buf,
            eps,
            h,
        );

        // ── Step 5: Fused gate+up+SwiGLU → swiglu_buf ──────────────────
        self.dispatch_fused_gate_up_swiglu(
            encoder,
            &gate_up_weight.buffer,
            &bufs.normed_buf,
            &bufs.swiglu_buf,
            inter,
            h,
        );

        // ── Step 7: GEMV(down, swiglu) → down_buf ───────────────────────
        // n_rows = hidden_size, k = intermediate_size
        self.dispatch_gemv_q1(
            encoder,
            &down_weight.buffer,
            &bufs.swiglu_buf,
            &bufs.down_buf,
            h,
            inter,
        );

        // ── Step 8: residual_add(hidden_buf, down_buf) ──────────────────
        self.dispatch_residual_add(encoder, &bufs.hidden_buf, &bufs.down_buf, h);

        // ── Commit and wait ──────────────────────────────────────────────
        encoder.end_encoding();
        cmd_buf.commit();
        let t3 = Instant::now();
        cmd_buf.wait_until_completed();
        let dt_gpu_wait = t3.elapsed();

        // ── Step 9: Read back ────────────────────────────────────────────
        let t4 = Instant::now();
        unsafe {
            download_f32(&bufs.hidden_buf, &mut hidden[..hidden_size]);
        }
        let dt_download = t4.elapsed();

        let dt_total = t_total.elapsed();
        if call_num % 36 == 0 {
            tracing::debug!(
                "MetalGraph FFN #{}: acquire={}µs upload={}µs encode={}µs gpu_wait={}µs download={}µs total={}µs",
                call_num,
                dt_acquire.as_micros(),
                dt_upload.as_micros(),
                dt_encode_setup.as_micros(),
                dt_gpu_wait.as_micros(),
                dt_download.as_micros(),
                dt_total.as_micros(),
            );
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // QKV phase dispatch (single GEMV, 1 encoder)
    // ─────────────────────────────────────────────────────────────────────

    /// Encode a fused QKV projection as a single GEMV dispatch.
    ///
    /// This is a thin wrapper around [`encode_gemv`](Self::encode_gemv) that
    /// provides a named entry point specifically for the Q/K/V projection
    /// hot-path in `block.rs`.
    ///
    /// - `input`: normed hidden state (length ≥ `k`)
    /// - `output`: fused QKV output (length ≥ `n_rows`)
    /// - `weight`: pre-uploaded fused Q+K+V weight handle
    /// - `n_rows`: total output rows (q_rows + k_rows + v_rows)
    /// - `k`: input dimension (hidden_size)
    pub fn encode_qkv_phase(
        &self,
        input: &[f32],
        output: &mut [f32],
        weight: &MetalWeightHandle,
        n_rows: usize,
        k: usize,
    ) -> Result<(), MetalGraphError> {
        self.encode_gemv(weight, input, output, n_rows, k)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal: buffer management
    // ─────────────────────────────────────────────────────────────────────

    /// Acquire the intermediate buffer set, allocating or re-allocating as
    /// needed.  Returns a mutex guard whose inner `Option` is guaranteed to
    /// be `Some`.
    fn acquire_buffers(
        &self,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<MetalBuffers>>, MetalGraphError> {
        let mut guard = self
            .buffers
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("buffer lock poisoned".into()))?;

        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.matches(hidden_size, intermediate_size),
            None => true,
        };

        if needs_alloc {
            *guard = Some(MetalBuffers::allocate(
                &self.device,
                hidden_size,
                intermediate_size,
            )?);
        }

        Ok(guard)
    }

    /// Expose the device reference for external buffer allocation.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: verify MetalGraph can be created on a Metal-capable system.
    #[test]
    fn test_metal_graph_creation() {
        // Skip if no Metal device (e.g. CI linux).
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new();
        assert!(graph.is_ok(), "MetalGraph::new() failed: {:?}", graph.err());
    }

    /// Test weight upload round-trip.
    #[test]
    fn test_weight_upload() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("failed to create MetalGraph");
        let data = vec![0u8; 1024];
        let handle = graph.upload_weight(&data);
        assert!(handle.is_ok());
        let handle = handle.expect("upload_weight failed");
        assert_eq!(handle.byte_len(), 1024);
    }

    /// Test the singleton accessor.
    #[test]
    fn test_global_singleton() {
        if Device::system_default().is_none() {
            return;
        }
        let g1 = MetalGraph::global();
        assert!(g1.is_ok());
        let g2 = MetalGraph::global();
        assert!(g2.is_ok());
        // Both should point to the same allocation.
        let g1 = g1.expect("global failed");
        let g2 = g2.expect("global failed");
        assert!(Arc::ptr_eq(&g1, &g2));
    }

    /// Test residual_add via a minimal single-op dispatch.
    #[test]
    fn test_residual_add_single() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("failed to create MetalGraph");
        let n = 256usize;
        let opts = MTLResourceOptions::StorageModeShared;

        let a_buf = alloc_buf(&graph.device, (n * 4) as u64, opts).expect("alloc a_buf");
        let b_buf = alloc_buf(&graph.device, (n * 4) as u64, opts).expect("alloc b_buf");

        // a = [1.0; 256], b = [2.0; 256]
        let a_data: Vec<f32> = vec![1.0; n];
        let b_data: Vec<f32> = vec![2.0; n];
        unsafe {
            upload_f32(&a_buf, &a_data);
            upload_f32(&b_buf, &b_data);
        }

        let cmd_buf = graph.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        graph.dispatch_residual_add(encoder, &a_buf, &b_buf, n as u32);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let mut result = vec![0.0f32; n];
        unsafe { download_f32(&a_buf, &mut result) };

        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 1e-6,
                "residual_add mismatch at index {i}: expected 3.0, got {v}"
            );
        }
    }

    /// Test GEMM Q1 dispatch: batch_size=4, verifies output matches expected values.
    ///
    /// Uses a trivial weight matrix (all-ones signs, scale=1.0h) so that
    /// each output element equals the sum of the corresponding input column.
    #[test]
    fn test_gemm_q1_batch4() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("failed to create MetalGraph");
        let opts = MTLResourceOptions::StorageModeShared;

        let n_rows: u32 = 8;
        let k: u32 = 128;
        let batch_size: u32 = 4;
        let blocks_per_row = (k / 128) as usize;

        // Build Q1_g128 weight buffer in SoA layout:
        // [scales: total_blocks × 2B][data: total_blocks × 16B]
        // All signs = 1 (bit set), scale = 1.0h = 0x3C00 LE → [0x00, 0x3C]
        let total_blocks = n_rows as usize * blocks_per_row;
        let total_weight_bytes = total_blocks * 2 + total_blocks * 16;
        let data_section = total_blocks * 2;
        let mut weight_data = vec![0u8; total_weight_bytes];
        for row in 0..n_rows as usize {
            for b in 0..blocks_per_row {
                let block_idx = row * blocks_per_row + b;
                // f16 1.0 in little-endian at scale position
                weight_data[block_idx * 2] = 0x00;
                weight_data[block_idx * 2 + 1] = 0x3C;
                // All 128 sign bits = 1 (all +1) at data position
                let d = data_section + block_idx * 16;
                for j in 0..16 {
                    weight_data[d + j] = 0xFF;
                }
            }
        }

        let weight_buf =
            alloc_buf(&graph.device, total_weight_bytes as u64, opts).expect("alloc weight_buf");
        unsafe {
            std::ptr::copy_nonoverlapping(
                weight_data.as_ptr(),
                weight_buf.contents() as *mut u8,
                total_weight_bytes,
            );
        }

        // Input: 4 columns of k=128 floats (column-major: input[col * k + i])
        // col 0 = all 1.0, col 1 = all 2.0, col 2 = all 0.5, col 3 = all -1.0
        let col_values = [1.0f32, 2.0, 0.5, -1.0];
        let input_floats = batch_size as usize * k as usize;
        let mut input_data = vec![0.0f32; input_floats];
        for col in 0..batch_size as usize {
            for i in 0..k as usize {
                input_data[col * k as usize + i] = col_values[col];
            }
        }

        let input_buf =
            alloc_buf(&graph.device, (input_floats * 4) as u64, opts).expect("alloc input_buf");
        unsafe {
            upload_f32(&input_buf, &input_data);
        }

        // Output: batch_size columns of n_rows floats (column-major)
        let output_floats = batch_size as usize * n_rows as usize;
        let output_buf =
            alloc_buf(&graph.device, (output_floats * 4) as u64, opts).expect("alloc output_buf");

        // Dispatch GEMM
        let cmd_buf = graph.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        graph.dispatch_gemm_q1_v7(
            encoder,
            &weight_buf,
            &input_buf,
            &output_buf,
            n_rows,
            k,
            batch_size,
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read back and verify
        let mut result = vec![0.0f32; output_floats];
        unsafe {
            download_f32(&output_buf, &mut result);
        }

        // Expected: each row for col c = scale * (sum of 128 signs * col_value)
        // With all signs = +1 and scale = 1.0:
        //   col 0: 1.0 * 128 * 1.0 = 128.0
        //   col 1: 1.0 * 128 * 2.0 = 256.0
        //   col 2: 1.0 * 128 * 0.5 = 64.0
        //   col 3: 1.0 * 128 * -1.0 = -128.0
        let expected_col_sums = [128.0f32, 256.0, 64.0, -128.0];
        for (col, expected) in expected_col_sums.iter().enumerate() {
            for row in 0..n_rows as usize {
                let idx = col * n_rows as usize + row;
                assert!(
                    (result[idx] - expected).abs() < 0.1,
                    "GEMM mismatch at col={col} row={row}: expected {expected}, got {}",
                    result[idx]
                );
            }
        }
    }

    /// Test GEMM Q1 vs independent GEMVs: batch GEMM should match individual GEMV results.
    #[test]
    fn test_gemm_matches_gemv() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("failed to create MetalGraph");
        let opts = MTLResourceOptions::StorageModeShared;

        let n_rows: u32 = 16;
        let k: u32 = 256; // 2 blocks per row
        let batch_size: u32 = 4;
        let blocks_per_row = (k / 128) as usize;
        let total_blocks = n_rows as usize * blocks_per_row;
        let total_weight_bytes = total_blocks * 2 + total_blocks * 16;

        // Build weight in SoA layout with alternating signs: block 0 all +1, block 1 all -1
        let data_section = total_blocks * 2;
        let mut weight_data = vec![0u8; total_weight_bytes];
        for row in 0..n_rows as usize {
            for b in 0..blocks_per_row {
                let block_idx = row * blocks_per_row + b;
                weight_data[block_idx * 2] = 0x00; // f16 1.0 LE
                weight_data[block_idx * 2 + 1] = 0x3C;
                let fill = if b % 2 == 0 { 0xFF } else { 0x00 }; // +1 or -1
                let d = data_section + block_idx * 16;
                for j in 0..16 {
                    weight_data[d + j] = fill;
                }
            }
        }

        let weight_buf =
            alloc_buf(&graph.device, total_weight_bytes as u64, opts).expect("alloc weight_buf");
        unsafe {
            std::ptr::copy_nonoverlapping(
                weight_data.as_ptr(),
                weight_buf.contents() as *mut u8,
                total_weight_bytes,
            );
        }

        // Input: varied per column
        let input_floats = batch_size as usize * k as usize;
        let mut input_data = vec![0.0f32; input_floats];
        for col in 0..batch_size as usize {
            for i in 0..k as usize {
                input_data[col * k as usize + i] = (col as f32 + 1.0) * 0.1;
            }
        }

        let input_buf =
            alloc_buf(&graph.device, (input_floats * 4) as u64, opts).expect("alloc input_buf");
        unsafe {
            upload_f32(&input_buf, &input_data);
        }

        // GEMM output
        let output_floats = batch_size as usize * n_rows as usize;
        let gemm_out_buf =
            alloc_buf(&graph.device, (output_floats * 4) as u64, opts).expect("alloc gemm_out_buf");

        {
            let cmd = graph.command_queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            graph.dispatch_gemm_q1_v7(
                enc,
                &weight_buf,
                &input_buf,
                &gemm_out_buf,
                n_rows,
                k,
                batch_size,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let mut gemm_result = vec![0.0f32; output_floats];
        unsafe {
            download_f32(&gemm_out_buf, &mut gemm_result);
        }

        // Now run individual GEMVs and compare
        for col in 0..batch_size as usize {
            // Upload this column's input into a separate buffer
            let col_input = &input_data[col * k as usize..(col + 1) * k as usize];
            let col_in_buf =
                alloc_buf(&graph.device, (k as usize * 4) as u64, opts).expect("alloc col_in_buf");
            unsafe {
                upload_f32(&col_in_buf, col_input);
            }

            let col_out_buf = alloc_buf(&graph.device, (n_rows as usize * 4) as u64, opts)
                .expect("alloc col_out_buf");

            let cmd = graph.command_queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            graph.dispatch_gemv_q1(enc, &weight_buf, &col_in_buf, &col_out_buf, n_rows, k);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            let mut gemv_result = vec![0.0f32; n_rows as usize];
            unsafe {
                download_f32(&col_out_buf, &mut gemv_result);
            }

            for row in 0..n_rows as usize {
                let gemm_val = gemm_result[col * n_rows as usize + row];
                let gemv_val = gemv_result[row];
                assert!(
                    (gemm_val - gemv_val).abs() < 1e-3,
                    "GEMM/GEMV mismatch col={col} row={row}: gemm={gemm_val}, gemv={gemv_val}"
                );
            }
        }
    }

    /// Test batched SwiGLU dispatch.
    #[test]
    fn test_batched_swiglu() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("failed to create MetalGraph");
        let opts = MTLResourceOptions::StorageModeShared;

        let inter: u32 = 64;
        let batch_size: u32 = 3;

        // gate_up: [batch_size × inter × 2] floats
        // For each batch b and element e:
        //   gate = gate_up[b * inter * 2 + e]
        //   up   = gate_up[b * inter * 2 + inter + e]
        let gate_up_len = batch_size as usize * inter as usize * 2;
        let mut gate_up_data = vec![0.0f32; gate_up_len];
        for b in 0..batch_size as usize {
            for e in 0..inter as usize {
                let base = b * inter as usize * 2;
                gate_up_data[base + e] = (b as f32 + 1.0) * 0.5; // gate
                gate_up_data[base + inter as usize + e] = (e as f32) * 0.1; // up
            }
        }

        let gate_up_buf =
            alloc_buf(&graph.device, (gate_up_len * 4) as u64, opts).expect("alloc gate_up_buf");
        unsafe {
            upload_f32(&gate_up_buf, &gate_up_data);
        }

        let output_len = batch_size as usize * inter as usize;
        let output_buf =
            alloc_buf(&graph.device, (output_len * 4) as u64, opts).expect("alloc output_buf");

        let cmd = graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        graph.dispatch_batched_swiglu(enc, &gate_up_buf, &output_buf, inter, batch_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut result = vec![0.0f32; output_len];
        unsafe {
            download_f32(&output_buf, &mut result);
        }

        // Verify: output[b * inter + e] = silu(gate) * up
        for b in 0..batch_size as usize {
            for e in 0..inter as usize {
                let g = (b as f32 + 1.0) * 0.5;
                let u = (e as f32) * 0.1;
                let silu_g = g / (1.0 + (-g).exp());
                let expected = silu_g * u;
                let actual = result[b * inter as usize + e];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "batched_swiglu mismatch b={b} e={e}: expected {expected}, got {actual}"
                );
            }
        }
    }

    /// Test batched RMSNorm dispatch.
    #[test]
    fn test_batched_rmsnorm() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("failed to create MetalGraph");
        let opts = MTLResourceOptions::StorageModeShared;

        let dim: u32 = 64;
        let batch_size: u32 = 3;
        let eps: f32 = 1e-5;

        // Input: batch_size vectors of dim floats
        let input_len = batch_size as usize * dim as usize;
        let mut input_data = vec![0.0f32; input_len];
        for b in 0..batch_size as usize {
            for i in 0..dim as usize {
                input_data[b * dim as usize + i] = (b as f32 + 1.0) * (i as f32 + 1.0) * 0.01;
            }
        }

        // Weight: all 1.0 (identity scaling)
        let weight_data = vec![1.0f32; dim as usize];

        let input_buf =
            alloc_buf(&graph.device, (input_len * 4) as u64, opts).expect("alloc input_buf");
        let weight_buf =
            alloc_buf(&graph.device, (dim as usize * 4) as u64, opts).expect("alloc weight_buf");
        let output_buf =
            alloc_buf(&graph.device, (input_len * 4) as u64, opts).expect("alloc output_buf");

        unsafe {
            upload_f32(&input_buf, &input_data);
            upload_f32(&weight_buf, &weight_data);
        }

        let cmd = graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        graph.dispatch_batched_rmsnorm(
            enc,
            &input_buf,
            &weight_buf,
            &output_buf,
            eps,
            dim,
            batch_size,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut result = vec![0.0f32; input_len];
        unsafe {
            download_f32(&output_buf, &mut result);
        }

        // Verify: for each batch, check RMS normalization
        for b in 0..batch_size as usize {
            let offset = b * dim as usize;
            let slice = &input_data[offset..offset + dim as usize];

            // Compute expected RMS
            let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
            let rms_inv = 1.0 / (sq_sum / dim as f32 + eps).sqrt();

            for i in 0..dim as usize {
                let expected = slice[i] * rms_inv; // weight = 1.0
                let actual = result[offset + i];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "batched_rmsnorm mismatch b={b} i={i}: expected {expected}, got {actual}"
                );
            }
        }
    }

    /// Correctness: Metal TQ2 GEMV must match the scalar reference within tolerance.
    ///
    /// Uses small shapes so the test runs fast but exercises:
    /// - SoA AoS→SoA reformat
    /// - MSL `gemv_tq2_g128_v1` (SIMD-group-per-row, 8 rows/threadgroup)
    /// - The 2-bit ternary encoding `0b00→-1, 0b01→0, 0b10→+1, 0b11→0`
    #[test]
    fn test_encode_gemv_tq2_matches_reference() {
        if Device::system_default().is_none() {
            return;
        }
        use half::f16;
        use oxibonsai_core::BlockTQ2_0_g128;

        let graph = MetalGraph::new().expect("failed to create MetalGraph");

        let n_rows = 16usize;
        let k = 256usize; // 2 blocks per row
        let blocks_per_row = k / 128;

        // Build a deterministic set of ternary blocks covering every 2-bit code.
        let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(n_rows * blocks_per_row);
        for row in 0..n_rows {
            for bk in 0..blocks_per_row {
                let mut qs = [0u8; 32];
                for (byte_idx, b) in qs.iter_mut().enumerate() {
                    let seed = row * 31 + bk * 17 + byte_idx;
                    let c0 = (seed % 3) as u8;
                    let c1 = ((seed / 3) % 3) as u8;
                    let c2 = ((seed / 9) % 3) as u8;
                    let c3 = ((seed / 27) % 3) as u8;
                    *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
                }
                blocks.push(BlockTQ2_0_g128 {
                    qs,
                    d: f16::from_f32(0.125 + 0.03125 * row as f32),
                });
            }
        }

        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.5).collect();

        // Reference via scalar kernel.
        let mut expected = vec![0f32; n_rows];
        crate::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut expected, n_rows, k)
            .expect("scalar reference GEMV failed");

        // Upload via Metal SoA path.
        let aos_bytes = {
            let ptr = blocks.as_ptr() as *const u8;
            let len = std::mem::size_of_val(blocks.as_slice());
            unsafe { std::slice::from_raw_parts(ptr, len) }
        };
        let handle = graph
            .upload_tq2_weight_soa(aos_bytes)
            .expect("upload_tq2_weight_soa failed");

        let mut got = vec![0f32; n_rows];
        graph
            .encode_gemv_tq2(&handle, &input, &mut got, n_rows, k)
            .expect("encode_gemv_tq2 failed");

        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "row {i}: expected {a}, got {b} (|Δ|={})",
                (a - b).abs()
            );
        }
    }

    /// Correctness test: `encode_tail_and_commit_ternary` must produce the same
    /// greedy token ID as the CPU scalar reference (RMSNorm → TQ2 GEMV → argmax).
    ///
    /// Synthetic geometry: hidden_size = 128, vocab_size = 256.
    /// The LM head has 256 rows × 128 columns (2 TQ2_0_g128 blocks per row).
    #[test]
    fn test_encode_tail_ternary_matches_reference() {
        if Device::system_default().is_none() {
            return;
        }
        use super::super::metal_full_layer::FullLayerBuffers;
        use half::f16;
        use oxibonsai_core::BlockTQ2_0_g128;

        let hidden_size = 128usize;
        let vocab_size = 256usize;
        let blocks_per_row = hidden_size / 128; // = 1

        let graph = MetalGraph::new().expect("MetalGraph::new failed");

        // ── Build deterministic hidden vector ──────────────────────────────
        let hidden_vec: Vec<f32> = (0..hidden_size)
            .map(|i| (i as f32) * 0.005 - 0.32)
            .collect();

        // ── Build deterministic RMSNorm weight (all ones for simplicity) ──
        let norm_weight: Vec<f32> = vec![1.0f32; hidden_size];
        let norm_eps = 1e-5f32;

        // ── Build deterministic TQ2 LM-head weight blocks ─────────────────
        let total_blocks = vocab_size * blocks_per_row;
        let mut lm_blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(total_blocks);
        for row in 0..vocab_size {
            for bk in 0..blocks_per_row {
                let mut qs = [0u8; 32];
                for (byte_idx, byte) in qs.iter_mut().enumerate() {
                    let seed = row * 37 + bk * 13 + byte_idx;
                    let c0 = (seed % 3) as u8;
                    let c1 = ((seed / 3) % 3) as u8;
                    let c2 = ((seed / 9) % 3) as u8;
                    let c3 = ((seed / 27) % 3) as u8;
                    *byte = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
                }
                lm_blocks.push(BlockTQ2_0_g128 {
                    qs,
                    d: f16::from_f32(0.0625 + 0.015625 * (row as f32 * 0.5 + bk as f32)),
                });
            }
        }

        // ── CPU reference: RMSNorm → scalar TQ2 GEMV → argmax ─────────────
        // RMSNorm: out_i = (x_i / sqrt(mean(x^2) + eps)) * weight_i
        let sq_mean: f32 = hidden_vec.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;
        let rms_scale = 1.0 / (sq_mean + norm_eps).sqrt();
        let normed_ref: Vec<f32> = hidden_vec
            .iter()
            .zip(norm_weight.iter())
            .map(|(&x, &w)| x * rms_scale * w)
            .collect();

        let mut logits_ref = vec![0.0f32; vocab_size];
        crate::gemv_ternary::gemv_tq2_0_g128(
            &lm_blocks,
            &normed_ref,
            &mut logits_ref,
            vocab_size,
            hidden_size,
        )
        .expect("scalar gemv_tq2_0_g128 failed");

        let expected_token: u32 = logits_ref
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .expect("logits_ref is empty");

        // ── Upload norm weight and LM head to GPU ─────────────────────────
        let norm_handle = graph
            .get_or_upload_f32_weight(9_000_001u64, &norm_weight)
            .expect("upload norm_weight failed");

        let lm_aos_bytes: &[u8] = {
            let ptr = lm_blocks.as_ptr() as *const u8;
            let len = std::mem::size_of_val(lm_blocks.as_slice());
            unsafe { std::slice::from_raw_parts(ptr, len) }
        };
        let lm_handle = graph
            .get_or_upload_tq2_weight_soa(9_000_002u64, lm_aos_bytes)
            .expect("upload_tq2_weight_soa failed");

        // ── Allocate FullLayerBuffers with minimal dimensions ──────────────
        // We only need hidden_buf and normed_buf for the tail; set the other
        // dimensions to their minimum viable values.
        let bufs = FullLayerBuffers::allocate(
            &graph.device,
            hidden_size,
            hidden_size, // intermediate_size — any positive value
            1,           // nq
            1,           // nkv
            64,          // head_dim — must be even
            1,           // max_seq
        )
        .expect("FullLayerBuffers::allocate failed");

        // Upload hidden vector into hidden_buf.
        unsafe { upload_f32(&bufs.hidden_buf, &hidden_vec) };

        // ── Run the GPU ternary tail ───────────────────────────────────────
        let mut got_token: u32 = u32::MAX;
        let mut hidden_mut = hidden_vec.clone();

        let cmd_buf = graph.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        graph
            .encode_tail_and_commit_ternary(
                encoder,
                cmd_buf,
                &bufs,
                &mut hidden_mut,
                hidden_size,
                Some(&norm_handle),
                norm_eps,
                Some(&lm_handle),
                vocab_size,
                None,
                Some(&mut got_token),
                false,
                None,
            )
            .expect("encode_tail_and_commit_ternary failed");

        // ── Verify token IDs match bit-exactly ────────────────────────────
        assert_eq!(
            got_token, expected_token,
            "GPU greedy token {got_token} != CPU reference token {expected_token}"
        );
    }
}
