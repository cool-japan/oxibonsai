//! GPU (CUDA) backend for the DiT ternary (`TQ2_0_g128`) matmuls.
//!
//! CUDA sibling of `crate::gpu` (the Metal backend), authored as a
//! line-for-line mirror. It routes the 100 ternary Linears of the FLUX.2 Klein
//! DiT forward onto the CUDA TQ2 GEMM kernel (`CudaGraph::encode_gemm_tq2` in
//! `oxibonsai-kernels`), keeping the per-weight 2-bit codes resident on the GPU
//! and only crossing the bus with the (small) f32 activations per matmul, plus
//! the DiT joint flash-attention (`CudaGraph::encode_joint_attention_flash_pooled`).
//!
//! The whole module is gated on `cfg(all(feature = "native-cuda", any(target_os
//! = "linux", target_os = "windows")))` — the same gate under which
//! `oxibonsai-kernels` re-exports `CudaGraph` — and is `target_os`-DISJOINT
//! from the Metal gate (macOS), so at most one of the two GPU backends ever
//! compiles and the default Pure-Rust CPU path is entirely unaffected. The same
//! env vars as the Metal path are reused (`OXI_DIT_GPU` / `OXI_DIT_ATTN_GPU`):
//! Metal and CUDA are mutually exclusive at build by `target_os`, so one flag
//! per subsystem is unambiguous.
//!
//! Correctness contract (verified by the Phase-1 CUDA kernel unit tests): the
//! kernel computes, for the DiT's row-major `input[M, K]`, out-major weight
//! blocks `[N, K]`, and `out[M, N]`, `out[m, n] = Σ_k input[m, k] ·
//! dequant(W)[n, k]` — i.e. exactly the same `A·Bᵀ` contraction as the CPU
//! [`crate::gemm::gemm_abt`], with no transpose and the identical AoS block
//! layout the kernel's reformat expects.
//!
//! On *any* error this module returns a `CudaGpuMatmulError`; the caller
//! ([`crate::math::ternary_matmul`]) swallows it and falls back to the CPU path,
//! so a GPU failure can never break a forward pass (no `unwrap`/`expect`/`panic!`).
//!
//! That fallback used to be completely silent (RAG-EVAL-IMG-18) on the Metal
//! sibling (`crate::gpu` (Metal builds)) before it was fixed there; every function below
//! now gets the same one-time-latch treatment (an `AtomicBool` per function,
//! `tracing::warn!` naming the op and the underlying error the first time it
//! fails, silent thereafter for the life of the process). Unlike
//! `crate::gpu` — which still uses `eprintln!` because it predates this
//! crate's `tracing` dependency and is out of this fix's scope to touch — this
//! (owned) file uses `tracing::warn!` directly.
//!
//! **No CUDA hardware on the reference macOS/Apple-Silicon development
//! machine**, so this module has never *run*, and this crate's own gate
//! commands never compile it (`target_os`-gated to Linux/Windows). It has,
//! however, been type-checked for real: `cargo check`/`cargo clippy -- -D
//! warnings -p oxibonsai-image --target x86_64-unknown-linux-gnu --features
//! native-cuda` (both `--all-features` and `native-cuda`-only) succeed
//! cleanly, including this module's `#[cfg(test)]` code — cross-compiling
//! resolves every type and runs every lint without needing a CUDA toolkit,
//! since `cargo check` only type-checks and never links. What is *not*
//! verified is linking or execution (this workstation has no Linux
//! cross-linker configured and no CUDA runtime/device either way), so a
//! logic bug that only manifests at the kernel-call boundary (e.g. an actual
//! `CudaGraph`/`CudaGraphError` behavioural mismatch) would not be caught
//! here. The two functions with a direct Metal analogue
//! (`ternary_matmul_gpu`, `joint_attention_gpu`) mirror `crate::gpu`'s
//! already-compiling, already-tested per-step `.inspect_err(..)` placement
//! exactly; the CUDA-only fused-resident functions below them
//! (`dense_matmul_gpu`, `single_block_gpu`, `single_blocks_gpu`,
//! `double_block_gpu`, which have no Metal-file counterpart to mirror)
//! instead wrap their unchanged original body in a thin `_impl` + outer
//! `.inspect_err(..)`, which is equivalent for a function's-worth of internal
//! `?`s without threading a closure through each one of their many individual
//! fallible steps by hand.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;
use oxibonsai_kernels::{CudaGraph, CudaGraphError, DitSingleBlockWeights};

use crate::blocks::{DoubleMod, ModTriple};
use crate::forward::QkvNorm;
use crate::math::RopeTables;
use crate::weights::{DitWeights, QuantizedLinear};

/// An error from the GPU ternary-matmul path. The caller converts this into a
/// silent CPU fallback, so it never propagates out of a forward pass.
#[derive(Debug, thiserror::Error)]
pub enum CudaGpuMatmulError {
    /// The process-wide CUDA graph singleton could not be obtained (e.g. no
    /// CUDA device, or the device failed to initialise).
    #[error("CUDA graph unavailable: {0}")]
    GraphUnavailable(String),
    /// The CUDA TQ2 GEMM (weight upload / encode / dispatch) failed.
    #[error("CUDA TQ2 GEMM failed: {0}")]
    Cuda(#[from] CudaGraphError),
    /// A DiT weight needed by the fused single-block path could not be resolved
    /// (missing tensor / wrong quant type). Triggers the per-op CPU/GPU fallback.
    #[error("DiT weight lookup failed: {0}")]
    Weights(String),
}

/// Reinterpret the packed ternary blocks as their raw little-endian AoS bytes.
///
/// `BlockTQ2_0_g128` is `#[repr(C)]` and exactly 34 bytes (`qs[32] ‖ d:f16`),
/// which *is* the AoS layout the kernel's `reformat_tq2_aos_to_soa` consumes, so
/// this byte view is a valid weight upload with no conversion.
fn blocks_as_bytes(blocks: &[BlockTQ2_0_g128]) -> &[u8] {
    debug_assert_eq!(std::mem::size_of::<BlockTQ2_0_g128>(), 34);
    let len = std::mem::size_of_val(blocks);
    // SAFETY: `blocks` is a live `&[BlockTQ2_0_g128]`; `BlockTQ2_0_g128` is
    // `#[repr(C)]` with no padding (size_of == 34, the sum of its fields) and no
    // invalid byte patterns (all-bytes-valid POD: `[u8;32]` + `f16`). The
    // resulting `&[u8]` borrows `blocks` for the same lifetime and is read-only,
    // and `len` is the exact byte size of the slice.
    unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, len) }
}

/// One-time confirmation that the GPU path actually executed at least once
/// (used by the parity example to PROVE the GPU ran, not a silent CPU fallback).
static GPU_USED: AtomicBool = AtomicBool::new(false);

/// Returns `true` once any [`ternary_matmul_gpu`] call has succeeded.
///
/// Lock-free and cheap; intended for diagnostics / parity assertions.
pub fn gpu_was_used() -> bool {
    GPU_USED.load(Ordering::Relaxed)
}

/// Cached runtime toggle for the GPU DiT path. Default ON when the `native-cuda`
/// feature is compiled; set env `OXI_DIT_GPU=0` to force the CPU path (for A/B
/// parity testing without recompiling).
static GPU_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether the DiT should use the GPU ternary path.
///
/// `true` unless the environment variable `OXI_DIT_GPU` is set to `0`. The env
/// read is cached in a `OnceLock` on first call.
pub fn dit_gpu_enabled() -> bool {
    *GPU_ENABLED.get_or_init(|| !matches!(std::env::var("OXI_DIT_GPU").ok().as_deref(), Some("0")))
}

/// One-time confirmation that the GPU **joint-attention** path actually executed
/// at least once (used by the parity example to PROVE the GPU flash-attention
/// kernel ran, not a silent CPU fallback). Tracked separately from [`GPU_USED`]
/// so the attention contribution can be A/B'd independently of the ternary path.
static DIT_ATTN_GPU_USED: AtomicBool = AtomicBool::new(false);

/// Returns `true` once any [`joint_attention_gpu`] call has succeeded.
///
/// Lock-free and cheap; intended for diagnostics / parity assertions.
pub fn dit_attn_gpu_was_used() -> bool {
    DIT_ATTN_GPU_USED.load(Ordering::Relaxed)
}

/// Cached runtime toggle for the GPU DiT **joint-attention** path. Default
/// **ON** when the `native-cuda` feature is compiled (this flash-attention path
/// is a parity-proven win — gated cos ≥ 0.999); set env `OXI_DIT_ATTN_GPU=0` to
/// force the CPU path (for A/B parity testing without recompiling).
static ATTN_GPU_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether the DiT should use the GPU flash-attention path for joint attention.
///
/// `true` unless the environment variable `OXI_DIT_ATTN_GPU` is set to `0`. The
/// env read is cached in a `OnceLock` on first call. Kept a *separate* toggle
/// from [`dit_gpu_enabled`] (which gates the ternary matmuls) so the attention
/// contribution can be measured independently for A/B parity + timing. The
/// per-op CPU fallback in [`crate::math::joint_attention`] (a silent fall-through
/// on any GPU `Err`) makes default-on safe.
pub fn dit_attn_gpu_enabled() -> bool {
    *ATTN_GPU_ENABLED
        .get_or_init(|| !matches!(std::env::var("OXI_DIT_ATTN_GPU").ok().as_deref(), Some("0")))
}

/// Latch for [`ternary_matmul_gpu`]'s one-time fallback warning.
static TERNARY_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
/// Latch for [`joint_attention_gpu`]'s one-time fallback warning.
static ATTN_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
/// Latch for [`dense_matmul_gpu`]'s one-time fallback warning.
static DENSE_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
/// Latch for [`single_block_gpu`]'s one-time fallback warning.
static SINGLE_BLOCK_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
/// Latch for [`single_blocks_gpu`]'s one-time fallback warning.
static SINGLE_BLOCKS_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
/// Latch for [`double_block_gpu`]'s one-time fallback warning.
static DOUBLE_BLOCK_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

/// Emit a one-time `tracing::warn!` the first time `flag`'s GPU op fails and
/// the caller falls back to the CPU reference path.
///
/// `flag.swap(true, ..)` returns the *previous* value, so the message prints
/// exactly once per process per call site — every subsequent failure (e.g. a
/// GPU that stays unavailable for the life of the process) is silently
/// dropped, matching a "warn once" contract without spamming the log on the
/// DiT's ~100 per-step matmul calls. Mirrors `crate::gpu`'s already-shipped
/// latch pattern (see the module docs for why this uses `tracing::warn!`
/// where that one still uses `eprintln!`).
fn warn_gpu_fallback_once(flag: &'static AtomicBool, op: &str, reason: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            op,
            reason,
            "oxibonsai-image: GPU op failed, falling back to CPU; further \
             occurrences in this process are not logged"
        );
    }
}

/// Compute FLUX.2 DiT joint multi-head scaled-dot-product attention on the GPU
/// via the CUDA flash-attention kernel.
///
/// Mirrors the CPU reference [`crate::math::joint_attention`] exactly: `q`, `k`,
/// `v` are head-major `[num_heads, seq, head_dim]` f32 (RoPE already applied to
/// q,k upstream), and the returned `Vec<f32>` is the token-major attention output
/// `[seq, num_heads * head_dim]` (heads concatenated along the feature axis), with
/// `scale = 1/sqrt(head_dim)` and a non-causal softmax over keys.
///
/// Uses the **pooled** flash entry point
/// (`CudaGraph::encode_joint_attention_flash_pooled`). The CPU↔GPU transfers
/// are negligible at the DiT shape, so this captures the full flash-kernel win
/// over the rayon+NEON CPU attention without needing q/k/v residency.
///
/// # Errors
/// Returns `CudaGpuMatmulError` if the CUDA graph is unavailable or the kernel
/// encode fails (e.g. `head_dim` not a multiple of 8, `head_dim > 128`, or
/// `seq` over the kernel's compile-time cap). The caller
/// ([`crate::math::joint_attention`]) falls back to the CPU path on any error.
pub fn joint_attention_gpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Result<Vec<f32>, CudaGpuMatmulError> {
    let graph = CudaGraph::global()
        .map_err(|e| CudaGpuMatmulError::GraphUnavailable(e.to_string()))
        .inspect_err(|e| {
            warn_gpu_fallback_once(&ATTN_FALLBACK_WARNED, "joint attention", &e.to_string())
        })?;
    // The kernel writes the token-major transposed result `[seq, num_heads*head_dim]`.
    let mut out = vec![0.0f32; seq * num_heads * head_dim];
    graph
        .encode_joint_attention_flash_pooled(q, k, v, &mut out, num_heads, seq, head_dim)
        .inspect_err(|e| {
            warn_gpu_fallback_once(&ATTN_FALLBACK_WARNED, "joint attention", &e.to_string())
        })?;
    DIT_ATTN_GPU_USED.store(true, Ordering::Relaxed);
    Ok(out)
}

/// Compute `out[m, n] = input[m, k] · dequant(blocks)[n, k]ᵀ` on the GPU.
///
/// - `blocks`: out-major ternary blocks (`n * (k / 128)` of them), borrowed from
///   the long-lived mmap'd GGUF in `DitWeights`.
/// - `input`: row-major `[m, k]`.
/// - `out`: row-major `[m, n]` (written in full).
///
/// The weight is uploaded to the GPU **once** and cached by its (mmap-stable,
/// per-weight-unique) pointer key, so subsequent steps reuse the resident SoA
/// buffer and only the activations cross the bus.
///
/// # Errors
/// Returns `CudaGpuMatmulError` if the CUDA graph is unavailable or the kernel
/// upload/encode fails (incl. `k % 128 != 0` or a length mismatch). The caller
/// falls back to the CPU path on any error.
pub fn ternary_matmul_gpu(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaGpuMatmulError> {
    let graph = CudaGraph::global()
        .map_err(|e| CudaGpuMatmulError::GraphUnavailable(e.to_string()))
        .inspect_err(|e| {
            warn_gpu_fallback_once(&TERNARY_FALLBACK_WARNED, "ternary matmul", &e.to_string())
        })?;
    // `blocks` is borrowed from the run-long mmap; its base address is stable and
    // unique per weight, so it doubles as a cache key with no per-Linear bookkeeping.
    // Pointer addresses are huge and won't collide with the LLM's small key space.
    let key = blocks.as_ptr() as u64;
    let handle = graph
        .get_or_upload_weight_tq2_soa_lazy(key, || blocks_as_bytes(blocks).to_vec())
        .inspect_err(|e| {
            warn_gpu_fallback_once(&TERNARY_FALLBACK_WARNED, "ternary matmul", &e.to_string())
        })?;
    graph
        .encode_gemm_tq2(&handle, input, out, m, n, k)
        .inspect_err(|e| {
            warn_gpu_fallback_once(&TERNARY_FALLBACK_WARNED, "ternary matmul", &e.to_string())
        })?;
    GPU_USED.store(true, Ordering::Relaxed);
    Ok(())
}

/// Dense **f32** matmul `out[m,n] = Σ_k input[m,k]·weight[n,k]` on the GPU via
/// `CudaGraph::encode_gemm_f32` — for the DiT's bf16-decoded *dense* Linears
/// (`x_embedder`, `context_embedder`, `proj_out`, `norm_out`). The stage0
/// `context_embedder` `[512×7680→3072]` dominates the DiT wall on the CPU
/// (~2.8 s/step), so routing it here is the largest stage0 win.
///
/// `weight` is the run-fresh `to_f32_vec()` of a bf16 tensor — a transient
/// buffer whose address is recycled across calls — so, exactly as in the TE
/// path, the device-weight cache entry is evicted right after the GEMM to avoid
/// a stale-pointer collision; each call therefore re-uploads its weight.
///
/// # Errors
/// `CudaGpuMatmulError` if the CUDA graph is unavailable or the upload / encode
/// fails. The caller ([`crate::math::dense_matmul`]) falls back to the CPU SIMD
/// `gemm_abt` on any `Err`.
pub fn dense_matmul_gpu(
    weight: &[f32],
    input: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaGpuMatmulError> {
    let graph = CudaGraph::global()
        .map_err(|e| CudaGpuMatmulError::GraphUnavailable(e.to_string()))
        .inspect_err(|e| {
            warn_gpu_fallback_once(&DENSE_FALLBACK_WARNED, "dense matmul", &e.to_string())
        })?;
    let key = weight.as_ptr() as u64;
    let handle = graph
        .get_or_upload_f32_weight(key, weight)
        .inspect_err(|e| {
            warn_gpu_fallback_once(&DENSE_FALLBACK_WARNED, "dense matmul", &e.to_string())
        })?;
    graph
        .encode_gemm_f32(&handle, input, out, m, n, k)
        .inspect_err(|e| {
            warn_gpu_fallback_once(&DENSE_FALLBACK_WARNED, "dense matmul", &e.to_string())
        })?;
    graph.evict_f32_weight(key).inspect_err(|e| {
        warn_gpu_fallback_once(&DENSE_FALLBACK_WARNED, "dense matmul", &e.to_string())
    })?;
    GPU_USED.store(true, Ordering::Relaxed);
    Ok(())
}

/// One-time confirmation that the **fused resident** single-block GPU path ran.
///
/// Distinct from [`GPU_USED`]: a fused-block failure falls back to the per-op
/// block, which itself sets [`GPU_USED`] via [`ternary_matmul_gpu`] — so this
/// separate flag is what PROVES the fused path (one upload / one download)
/// actually executed, not a per-op fallback.
static DIT_FUSED_USED: AtomicBool = AtomicBool::new(false);

/// Returns `true` once any [`single_block_gpu`] call has succeeded.
///
/// Lock-free and cheap; intended for diagnostics / parity assertions.
pub fn dit_fused_block_was_used() -> bool {
    DIT_FUSED_USED.load(Ordering::Relaxed)
}

/// Cached runtime toggle for the **fused resident** DiT block forward (both
/// single- and dual-stream blocks). Default **ON**: set env `OXI_DIT_FUSED=0` to
/// force the per-op GPU path (for A/B parity / timing without recompiling). The
/// fused path keeps each block's activations device-resident across the whole
/// block (one upload / one download), validated cos-identical to the per-op path
/// (`dit_parity` 51/51) and ~1.56× faster end-to-end. Gated together with
/// [`dit_gpu_enabled`] at the call site, so `OXI_DIT_GPU=0` (no GPU ternary) also
/// disables fusion — i.e. it still forces the full CPU path as before.
static DIT_FUSED_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether DiT blocks should use the fused resident GPU forward.
///
/// `true` unless the environment variable `OXI_DIT_FUSED` is set to `0`. The env
/// read is cached in a `OnceLock` on first call.
pub fn dit_fused_enabled() -> bool {
    *DIT_FUSED_ENABLED
        .get_or_init(|| !matches!(std::env::var("OXI_DIT_FUSED").ok().as_deref(), Some("0")))
}

/// Run one FLUX.2 DiT single-stream block fully GPU-resident.
///
/// Mirrors [`crate::blocks::SingleBlock::forward`] op-for-op (LayerNorm →
/// modulate → fused qkv-mlp ternary GEMM → q/k/v reshape → QK-RMSNorm → RoPE →
/// joint flash-attention → SwiGLU → `[attn ‖ gated]` concat → `to_out` ternary
/// GEMM → gated residual add), but uploads `h` once and downloads it once,
/// collapsing the ~11 per-op host round-trips of the unfused path to a single
/// one. `h` is row-major `[seq, hidden_size]`, modified in place.
///
/// On any `Err`, `h` is left **untouched** (the device→host copy is the final
/// step, reached only after every launch succeeds), so the caller falls back to
/// the per-op CPU/GPU block with no corruption.
///
/// # Errors
/// `CudaGpuMatmulError` if the CUDA graph is unavailable, a required weight
/// (`to_qkv_mlp_proj` / `to_out`) is missing or the wrong quant type, or the
/// resident encode fails. On failure, a one-time `tracing::warn!` fires first
/// — see `warn_gpu_fallback_once`.
#[allow(clippy::too_many_arguments)]
pub fn single_block_gpu(
    weights: &DitWeights,
    index: u32,
    h: &mut [f32],
    seq: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    ffn_inner: usize,
    eps: f32,
    rope: &RopeTables,
    mod_single: &ModTriple,
    norms: &QkvNorm,
) -> Result<(), CudaGpuMatmulError> {
    single_block_gpu_impl(
        weights,
        index,
        h,
        seq,
        hidden_size,
        num_heads,
        head_dim,
        ffn_inner,
        eps,
        rope,
        mod_single,
        norms,
    )
    .inspect_err(|e| {
        warn_gpu_fallback_once(
            &SINGLE_BLOCK_FALLBACK_WARNED,
            "single block (fused)",
            &e.to_string(),
        )
    })
}

/// The unwarned body of [`single_block_gpu`], factored out so the public
/// function's only job is dispatching to it and attaching the one-time
/// fallback warning — see this file's module docs for why the CUDA-only fused
/// functions use this `_impl` + outer `.inspect_err(..)` shape instead of
/// threading the warning through each individual internal `?`.
#[allow(clippy::too_many_arguments)]
fn single_block_gpu_impl(
    weights: &DitWeights,
    index: u32,
    h: &mut [f32],
    seq: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    ffn_inner: usize,
    eps: f32,
    rope: &RopeTables,
    mod_single: &ModTriple,
    norms: &QkvNorm,
) -> Result<(), CudaGpuMatmulError> {
    let graph =
        CudaGraph::global().map_err(|e| CudaGpuMatmulError::GraphUnavailable(e.to_string()))?;
    let p = format!("single_transformer_blocks.{index}");
    let proj = weights
        .quantized_linear(&format!("{p}.attn.to_qkv_mlp_proj"))
        .map_err(|e| CudaGpuMatmulError::Weights(e.to_string()))?;
    let to_out = weights
        .quantized_linear(&format!("{p}.attn.to_out"))
        .map_err(|e| CudaGpuMatmulError::Weights(e.to_string()))?;
    // The block base pointers double as stable per-weight upload cache keys
    // (same run-long-mmap convention as `ternary_matmul_gpu`).
    let proj_key = proj.blocks.as_ptr() as u64;
    let out_key = to_out.blocks.as_ptr() as u64;
    graph.encode_dit_single_block(
        h,
        proj_key,
        blocks_as_bytes(proj.blocks),
        proj.out_features as usize,
        out_key,
        blocks_as_bytes(to_out.blocks),
        &norms.q,
        &norms.k,
        &mod_single.shift,
        &mod_single.scale,
        &mod_single.gate,
        &rope.cos,
        &rope.sin,
        seq,
        hidden_size,
        num_heads,
        head_dim,
        ffn_inner,
        eps,
    )?;
    GPU_USED.store(true, Ordering::Relaxed);
    DIT_FUSED_USED.store(true, Ordering::Relaxed);
    Ok(())
}

/// Run the WHOLE single-stream block stack GPU-resident in one call.
///
/// Mirrors calling [`single_block_gpu`] for every `j in 0..num_single`, but the
/// residual stream `joint` and the SHARED modulation/RoPE are uploaded once, the
/// scratch is allocated once, and `joint` is downloaded once — eliminating the
/// per-block PCIe round-trip + device sync that dominate the DiT wall on a
/// discrete GPU. The per-block ternary weights are cached by their mmap pointer;
/// only the tiny QK-norm vectors are re-staged. On any `Err` the caller falls
/// back to the per-block path (and `joint` is written back only on success). A
/// one-time `tracing::warn!` fires first — see `warn_gpu_fallback_once`.
#[allow(clippy::too_many_arguments)]
pub fn single_blocks_gpu(
    weights: &DitWeights,
    num_single: usize,
    joint: &mut [f32],
    seq: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    ffn_inner: usize,
    eps: f32,
    rope: &RopeTables,
    mod_single: &ModTriple,
    norms: &[QkvNorm],
) -> Result<(), CudaGpuMatmulError> {
    single_blocks_gpu_impl(
        weights,
        num_single,
        joint,
        seq,
        hidden_size,
        num_heads,
        head_dim,
        ffn_inner,
        eps,
        rope,
        mod_single,
        norms,
    )
    .inspect_err(|e| {
        warn_gpu_fallback_once(
            &SINGLE_BLOCKS_FALLBACK_WARNED,
            "single blocks (fused, whole stack)",
            &e.to_string(),
        )
    })
}

/// The unwarned body of [`single_blocks_gpu`] — see this file's module docs
/// for why the CUDA-only fused functions use this `_impl` + outer
/// `.inspect_err(..)` shape.
#[allow(clippy::too_many_arguments)]
fn single_blocks_gpu_impl(
    weights: &DitWeights,
    num_single: usize,
    joint: &mut [f32],
    seq: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    ffn_inner: usize,
    eps: f32,
    rope: &RopeTables,
    mod_single: &ModTriple,
    norms: &[QkvNorm],
) -> Result<(), CudaGpuMatmulError> {
    let graph =
        CudaGraph::global().map_err(|e| CudaGpuMatmulError::GraphUnavailable(e.to_string()))?;
    if norms.len() != num_single {
        return Err(CudaGpuMatmulError::Weights(format!(
            "single_blocks_gpu: norms {} != num_single {num_single}",
            norms.len()
        )));
    }
    // Resolve every block's ternary weights up front; `QuantizedLinear` borrows
    // the run-long mmap, so the `(handle, bytes)` views below stay valid across
    // the resident encode.
    let mut lins: Vec<(QuantizedLinear, QuantizedLinear)> = Vec::with_capacity(num_single);
    let mut proj_out = 0usize;
    for j in 0..num_single {
        let p = format!("single_transformer_blocks.{j}");
        let proj = weights
            .quantized_linear(&format!("{p}.attn.to_qkv_mlp_proj"))
            .map_err(|e| CudaGpuMatmulError::Weights(e.to_string()))?;
        let to_out = weights
            .quantized_linear(&format!("{p}.attn.to_out"))
            .map_err(|e| CudaGpuMatmulError::Weights(e.to_string()))?;
        if j == 0 {
            proj_out = proj.out_features as usize;
        }
        lins.push((proj, to_out));
    }
    let block_params: Vec<DitSingleBlockWeights> = lins
        .iter()
        .enumerate()
        .map(|(j, (proj, to_out))| DitSingleBlockWeights {
            proj_handle: proj.blocks.as_ptr() as u64,
            proj_bytes: blocks_as_bytes(proj.blocks),
            out_handle: to_out.blocks.as_ptr() as u64,
            out_bytes: blocks_as_bytes(to_out.blocks),
            norm_q: &norms[j].q,
            norm_k: &norms[j].k,
        })
        .collect();
    graph.encode_dit_single_blocks(
        joint,
        &block_params,
        proj_out,
        &mod_single.shift,
        &mod_single.scale,
        &mod_single.gate,
        &rope.cos,
        &rope.sin,
        seq,
        hidden_size,
        num_heads,
        head_dim,
        ffn_inner,
        eps,
    )?;
    GPU_USED.store(true, Ordering::Relaxed);
    DIT_FUSED_USED.store(true, Ordering::Relaxed);
    Ok(())
}

/// Run one FLUX.2 DiT dual-stream (double) block fully GPU-resident.
///
/// Mirrors [`crate::blocks::DoubleBlock::forward`] op-for-op across both streams
/// (image `hidden` `[seq_img, hidden]`, text `enc` `[seq_txt, hidden]`): modulated
/// LayerNorm → 6 separate q/k/v ternary projections → per-stream QK-RMSNorm →
/// txt‖img head-major concat → RoPE → one joint flash-attention → split → 2
/// `to_out` projections → gated residual; then per-stream LayerNorm+modulate →
/// SwiGLU feed-forward → gated residual. Each stream is uploaded once and
/// downloaded once.
///
/// On any `Err`, both streams are left **untouched** (the device→host copies are
/// the final step), so the caller falls back to the per-op block with no
/// corruption.
///
/// # Errors
/// `CudaGpuMatmulError` if the CUDA graph is unavailable, a required weight is
/// missing / the wrong quant type, or the resident encode fails. A one-time
/// `tracing::warn!` fires first — see `warn_gpu_fallback_once`.
#[allow(clippy::too_many_arguments)]
pub fn double_block_gpu(
    weights: &DitWeights,
    index: u32,
    hidden: &mut [f32],
    enc: &mut [f32],
    seq_img: usize,
    seq_txt: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    ffn_inner: usize,
    eps: f32,
    rope: &RopeTables,
    mod_img: &DoubleMod,
    mod_txt: &DoubleMod,
) -> Result<(), CudaGpuMatmulError> {
    double_block_gpu_impl(
        weights,
        index,
        hidden,
        enc,
        seq_img,
        seq_txt,
        hidden_size,
        num_heads,
        head_dim,
        ffn_inner,
        eps,
        rope,
        mod_img,
        mod_txt,
    )
    .inspect_err(|e| {
        warn_gpu_fallback_once(
            &DOUBLE_BLOCK_FALLBACK_WARNED,
            "double block (fused)",
            &e.to_string(),
        )
    })
}

/// The unwarned body of [`double_block_gpu`] — see this file's module docs for
/// why the CUDA-only fused functions use this `_impl` + outer
/// `.inspect_err(..)` shape.
#[allow(clippy::too_many_arguments)]
fn double_block_gpu_impl(
    weights: &DitWeights,
    index: u32,
    hidden: &mut [f32],
    enc: &mut [f32],
    seq_img: usize,
    seq_txt: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    ffn_inner: usize,
    eps: f32,
    rope: &RopeTables,
    mod_img: &DoubleMod,
    mod_txt: &DoubleMod,
) -> Result<(), CudaGpuMatmulError> {
    let graph =
        CudaGraph::global().map_err(|e| CudaGpuMatmulError::GraphUnavailable(e.to_string()))?;
    let p = format!("transformer_blocks.{index}");
    let ql = |suffix: &str| -> Result<QuantizedLinear<'_>, CudaGpuMatmulError> {
        weights
            .quantized_linear(&format!("{p}.{suffix}"))
            .map_err(|e| CudaGpuMatmulError::Weights(e.to_string()))
    };
    let nrm = |suffix: &str| -> Result<Vec<f32>, CudaGpuMatmulError> {
        Ok(weights
            .bf16_tensor(&format!("{p}.{suffix}"))
            .map_err(|e| CudaGpuMatmulError::Weights(e.to_string()))?
            .to_f32_vec())
    };
    // (handle, AoS bytes) for a ternary weight; the block base ptr is the stable
    // per-weight upload cache key (same convention as `ternary_matmul_gpu`). The
    // returned bytes borrow the run-long mmap (lifetime independent of `&q`).
    fn tw<'a>(q: &QuantizedLinear<'a>) -> (u64, &'a [u8]) {
        (q.blocks.as_ptr() as u64, blocks_as_bytes(q.blocks))
    }
    let to_q = ql("attn.to_q")?;
    let to_k = ql("attn.to_k")?;
    let to_v = ql("attn.to_v")?;
    let add_q = ql("attn.add_q_proj")?;
    let add_k = ql("attn.add_k_proj")?;
    let add_v = ql("attn.add_v_proj")?;
    let to_out = ql("attn.to_out.0")?;
    let to_add_out = ql("attn.to_add_out")?;
    let ff_in = ql("ff.linear_in")?;
    let ff_out = ql("ff.linear_out")?;
    let ffc_in = ql("ff_context.linear_in")?;
    let ffc_out = ql("ff_context.linear_out")?;
    let norm_q_img = nrm("attn.norm_q.weight")?;
    let norm_k_img = nrm("attn.norm_k.weight")?;
    let norm_q_txt = nrm("attn.norm_added_q.weight")?;
    let norm_k_txt = nrm("attn.norm_added_k.weight")?;
    graph.encode_dit_double_block(
        hidden,
        enc,
        tw(&to_q),
        tw(&to_k),
        tw(&to_v),
        tw(&add_q),
        tw(&add_k),
        tw(&add_v),
        tw(&to_out),
        tw(&to_add_out),
        tw(&ff_in),
        tw(&ff_out),
        tw(&ffc_in),
        tw(&ffc_out),
        &norm_q_img,
        &norm_k_img,
        &norm_q_txt,
        &norm_k_txt,
        (&mod_img.msa.shift, &mod_img.msa.scale, &mod_img.msa.gate),
        (&mod_img.mlp.shift, &mod_img.mlp.scale, &mod_img.mlp.gate),
        (&mod_txt.msa.shift, &mod_txt.msa.scale, &mod_txt.msa.gate),
        (&mod_txt.mlp.shift, &mod_txt.mlp.scale, &mod_txt.mlp.gate),
        &rope.cos,
        &rope.sin,
        seq_img,
        seq_txt,
        hidden_size,
        num_heads,
        head_dim,
        ffn_inner,
        eps,
    )?;
    GPU_USED.store(true, Ordering::Relaxed);
    DIT_FUSED_USED.store(true, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_is_thirty_four_bytes() {
        assert_eq!(std::mem::size_of::<BlockTQ2_0_g128>(), 34);
    }

    #[test]
    fn blocks_as_bytes_length_is_34_per_block() {
        // Two all-zero blocks → 68 bytes, and the view aliases the source.
        let blocks = vec![
            BlockTQ2_0_g128 {
                qs: [0u8; 32],
                d: half::f16::ZERO,
            };
            2
        ];
        let bytes = blocks_as_bytes(&blocks);
        assert_eq!(bytes.len(), 68);
        assert_eq!(bytes.as_ptr() as usize, blocks.as_ptr() as usize);
    }

    #[test]
    fn gpu_enabled_defaults_on_when_env_unset() {
        // Note: OnceLock caches the first read; this asserts the default policy
        // (env `OXI_DIT_GPU` unset → enabled). It does not mutate the env.
        if std::env::var("OXI_DIT_GPU").is_err() {
            assert!(dit_gpu_enabled());
        }
    }

    #[test]
    fn attn_gpu_enabled_defaults_on_when_env_unset() {
        // Note: OnceLock caches the first read; this asserts the default policy
        // (env `OXI_DIT_ATTN_GPU` unset → enabled). It does not mutate the env.
        if std::env::var("OXI_DIT_ATTN_GPU").is_err() {
            assert!(dit_attn_gpu_enabled());
        }
    }

    #[test]
    fn fused_enabled_defaults_on_when_env_unset() {
        // Note: OnceLock caches the first read; this asserts the default policy
        // (env `OXI_DIT_FUSED` unset → enabled). It does not mutate the env.
        if std::env::var("OXI_DIT_FUSED").is_err() {
            assert!(dit_fused_enabled());
        }
    }

    #[test]
    fn warn_gpu_fallback_once_fires_exactly_once_per_flag() {
        // A private, test-local latch (never touched by any other test or by
        // the real GPU call sites), so this is deterministic regardless of
        // process/thread scheduling — mirrors `crate::gpu`'s Metal sibling
        // test of the same name.
        static LOCAL_WARNED: AtomicBool = AtomicBool::new(false);
        assert!(
            !LOCAL_WARNED.load(Ordering::Relaxed),
            "fresh static must start false"
        );
        warn_gpu_fallback_once(&LOCAL_WARNED, "test op", "first reason");
        assert!(
            LOCAL_WARNED.load(Ordering::Relaxed),
            "the first call must latch the flag"
        );
        warn_gpu_fallback_once(&LOCAL_WARNED, "test op", "second reason");
        warn_gpu_fallback_once(&LOCAL_WARNED, "test op", "third reason");
        assert!(LOCAL_WARNED.load(Ordering::Relaxed));
    }
}
