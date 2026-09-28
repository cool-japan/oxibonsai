//! Compile-time MSL kernel pipeline construction and caching.
//!
//! Concatenates the actively-used MSL kernel sources into a single library,
//! compiles them via `xcrun metal` (or runtime fallback), caches the resulting
//! `.metallib` on disk, and extracts each compute pipeline by entry-point name.

use metal::{CompileOptions, ComputePipelineState, Device, Library};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Mutex;

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
    /// GPU partial top-k (`perf-11` sampled-decode path): `k` highest
    /// `(id, value)` pairs from a logits row, so a sampled request no longer
    /// downloads the full row (993 KB/token at Bonsai 2's 248 320 vocab) just
    /// to run `top_k`/`top_p` on the CPU. Dispatched by
    /// `metal_dispatch.rs::dispatch_topk_f32`; wiring it into the engine's
    /// sampled decode arm (`engine_greedy.rs`, which the engine-side package
    /// owns — the item moved there with that file in wave 4) has not landed,
    /// so nothing in the non-test build calls `dispatch_topk_f32` yet — same
    /// situation as `gemm_tq2_g128_v8_tiled` above, hence
    /// `#[allow(dead_code)]`. `metal_dispatch.rs`'s own tests dispatch this
    /// kernel for real and check its output against a CPU oracle.
    #[allow(dead_code)]
    pub(crate) topk_f32: ComputePipelineState,
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

    /// bf16-input / f32-accumulate sibling of `gemm_f32_simdgroup` for the TE.
    /// Same shape and buffers; stages the operands as `bfloat` (M3+/Metal 3.1)
    /// for ~2× throughput at the model's native precision. Dispatched by
    /// `encode_gemm_bf16`; the GPU TE path uses it by default (cos ≈ 1.0), with
    /// `OXI_TE_GEMM_F32=1` to force the exact f32 kernel.
    ///
    /// `None` when the device/toolchain lacks `bfloat` simdgroup_matrix support
    /// (M1/M2 or macOS < 14): the kernel is compiled best-effort in its **own**
    /// library, kept OUT of the combined metallib so the rest of the backend
    /// (DiT/VAE/TE-f32, which only need M1-class half/f32 simdgroup) always
    /// compiles. When `None` the TE GEMM transparently uses `gemm_f32_simdgroup`.
    pub(crate) gemm_bf16_simdgroup: Option<ComputePipelineState>,

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

    /// The compiled Metal library backing every pipeline above, retained so
    /// [`MetalPipelines::pipeline_for`] can resolve *additional* entry points
    /// on demand.
    ///
    /// MET-10: the K-quant (`metal_k_quant_kernels.rs`), standard-quant
    /// (`metal_q_std_kernels.rs`), FP8 GEMV (`metal_fp8_kernels.rs`) and FP8
    /// batch-prefill (`metal_fp8_prefill.rs`) families used to compile their
    /// own per-kernel `MTLLibrary` from source on first use — 16 separate,
    /// uncached compilations on private devices and queues. Their MSL rides
    /// this combined, embedded/disk-cached library (`build_combined_msl`
    /// below), and all four families now resolve their 16 entry points by
    /// name through `pipeline_for` against this retained library instead of
    /// growing 16 named fields here that only one family each would read.
    library: Library,
    /// Lazily-populated cache backing [`MetalPipelines::pipeline_for`], so a
    /// repeat lookup by name is an `Arc`-free pointer-retain instead of a
    /// fresh `get_function` + `new_compute_pipeline_state_with_function`.
    by_name: Mutex<HashMap<String, ComputePipelineState>>,
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
        let topk_f32 = pipeline_for(&library, device, "topk_f32")?;
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
        // Optional bf16 TE GEMM: compiled best-effort in its OWN library so a
        // device/toolchain without `bfloat` simdgroup support (M1/M2, macOS<14)
        // can never fail `compile()`. `None` ⇒ the TE GEMM uses the f32 kernel.
        let gemm_bf16_simdgroup = try_compile_bf16_pipeline(device);
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
            topk_f32,
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
            gemm_bf16_simdgroup,
            im2col_f32,
            groupnorm_f32,
            silu_f32,
            upsample_nearest_f32,
            conv2d_f32_implicit,
            joint_attention_flash_f32,
            library,
            by_name: Mutex::new(HashMap::new()),
        })
    }

    /// Resolve (and cache) a compute pipeline for `name` from the combined
    /// embedded metallib, compiling it into a pipeline state on first request
    /// and cloning the cached state on every later one.
    ///
    /// MET-10: this is how the K-quant, standard-quant, FP8 GEMV and FP8
    /// batch-prefill kernel families (`metal_k_quant_kernels.rs`,
    /// `metal_q_std_kernels.rs`, `metal_fp8_kernels.rs`,
    /// `metal_fp8_prefill.rs`) obtain their pipelines — against this shared,
    /// disk-cached, embedded metallib and the shared `MetalGraph`
    /// device/command queue, instead of compiling a private `MTLLibrary` per
    /// kernel from source — without this struct growing a field for every
    /// one of the 16 entry points those families use (see the `library`
    /// field doc).
    ///
    /// Returns [`MetalGraphError::EncodingFailed`] if `name` is not a
    /// `kernel void` anywhere in the combined MSL, or if the cache mutex is
    /// poisoned by an earlier panic — for any name actually listed in
    /// `build.rs::ACTIVE_KERNELS` this cannot happen in a working build
    /// (`combined_msl_matches_active_kernels_exactly` in
    /// `tests/build_script_kernel_sources.rs` and the runtime entry-point
    /// verification in [`load_or_compile_library`] both guard that).
    ///
    /// Not a strict single-compile guarantee under concurrent callers: the
    /// mutex only serialises the cache read/insert, not the whole
    /// lookup-or-compile sequence, so two threads racing on the same
    /// still-uncached `name` can each miss, each compile their own
    /// `ComputePipelineState` for that entry point, and each insert —
    /// the loser's insert simply overwrites the winner's in the map. Both
    /// pipeline states are equally valid (same function, same library), so
    /// this is a redundant compile on a cold name under contention, never a
    /// correctness issue; every caller still gets back a working pipeline
    /// for `name`, including the one whose insert lost the race.
    pub(crate) fn pipeline_for(
        &self,
        device: &Device,
        name: &str,
    ) -> Result<ComputePipelineState, MetalGraphError> {
        let mut cache = self.by_name.lock().map_err(|_| {
            MetalGraphError::EncodingFailed(format!(
                "pipeline_for('{name}'): by-name pipeline cache mutex poisoned"
            ))
        })?;
        if let Some(pso) = cache.get(name) {
            return Ok(pso.clone());
        }
        let pso = pipeline_for(&self.library, device, name)?;
        cache.insert(name.to_string(), pso.clone());
        Ok(pso)
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
    src.push_str(kernel_sources::MSL_TOPK_F32);
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
    // ── K-quant GEMV (MET-10) ─────────────────────────────────────────────
    // Not extracted into a named `MetalPipelines` field (see the `library`
    // field doc) — resolved on demand through `MetalPipelines::pipeline_for`.
    src.push_str(kernel_sources::MSL_GEMV_Q2K_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMV_Q3K_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMV_Q4K_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMV_Q5K_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMV_Q6K_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMV_Q8K_V1);
    src.push('\n');
    // ── Standard GGUF Q4_0 / Q8_0 GEMV (MET-10) ────────────────────────────
    src.push_str(kernel_sources::MSL_GEMV_Q4_0_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMV_Q8_0_V1);
    src.push('\n');
    // ── FP8 single-token GEMV (MET-10) ─────────────────────────────────────
    src.push_str(kernel_sources::MSL_GEMV_FP8_E4M3_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMV_FP8_E5M2_V1);
    src.push('\n');
    // ── FP8 batch prefill GEMM / fused / gemv-pf (MET-10) ──────────────────
    src.push_str(kernel_sources::MSL_GEMM_FP8_E4M3_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMM_FP8_E4M3_RESIDUAL_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMV_FP8_E4M3_PF_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMM_FP8_E5M2_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMM_FP8_E5M2_RESIDUAL_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1);
    src.push('\n');
    src.push_str(kernel_sources::MSL_GEMV_FP8_E5M2_PF_V1);
    src.push('\n');
    src
}

/// Scan `msl_source` (the exact string [`build_combined_msl`] just produced)
/// for every `kernel void <name>(` entry-point declaration, line-anchored so
/// a mention inside a comment can never match.
///
/// This is deliberately **derived from the MSL text itself** rather than a
/// hand-maintained `const` list of entry-point names: `ACTIVE_KERNELS`
/// (`build.rs`) and the textual mirror in
/// `tests/build_script_kernel_sources.rs` already give this whitelist two
/// independent copies (by `MSL_*` Rust-constant name); a third, hand-typed
/// list of the underlying Metal *entry-point* names — a different
/// namespace — would be a fourth place to fall out of sync (MET-12 review
/// note) instead of closing the gap. Deriving it from `msl_source` cannot
/// desync by construction: whatever `build_combined_msl()` pushes is exactly
/// what this scans.
fn required_entry_points(msl_source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for line in msl_source.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("kernel void ") {
            if let Some(paren) = rest.find('(') {
                let name = rest[..paren].trim();
                if !name.is_empty() {
                    names.insert(name.to_string());
                }
            }
        }
    }
    names
}

/// Verify every entry point [`required_entry_points`] finds in `msl_source`
/// actually resolves in `library` (MET-12).
///
/// `library.get_function` compiles nothing — it only looks up an already
/// linked-in symbol — so this is cheap even for the ~49-kernel combined
/// library and safe to run on every embedded-metallib load.
fn embedded_library_has_all_required_entry_points(library: &Library, msl_source: &str) -> bool {
    required_entry_points(msl_source)
        .iter()
        .all(|name| library.get_function(name, None).is_ok())
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

/// Process-local counter making [`compile_msl_via_xcrun`]'s temp build
/// directory unique across the (up to two) calls one process makes: the
/// primary combined library and the optional bf16 sidecar
/// ([`load_bf16_library`]) both route through this same function.
static TEMP_BUILD_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// RAII guard for [`compile_msl_via_xcrun`]'s per-call temp build directory:
/// removes it (recursively) on drop, so every exit path — the successful
/// tail expression, or any of the function's many early `?`/`return None`s —
/// cleans up instead of leaking one throwaway directory per xcrun
/// invocation.
struct TempBuildDir(PathBuf);

impl TempBuildDir {
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempBuildDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Compile MSL source to a `.metallib` binary via `xcrun metal` + `xcrun metallib`,
/// cache the result to `cache_path`, and load the library.
///
/// # MET-M5: a per-process, per-call unique build directory
///
/// This used to build into a **fixed, shared** temp directory
/// (`$TMPDIR/oxibonsai_metal_build`) with fixed file names
/// (`combined.{metal,air,metallib}`) — on the only hardware-validated
/// backend. Two processes compiling concurrently (this project routinely
/// runs multi-session builds) would interleave writes to the same files, and
/// a symlink planted at the fixed metallib path could redirect the write to
/// an attacker-chosen location. Every call now gets its own directory named
/// with both the process id and a process-local sequence number (this
/// function runs up to twice per process — combined library, then the bf16
/// sidecar), created with `create_dir` (fails if the name already exists,
/// unlike `create_dir_all`) rather than assumed-fresh, and removed on every
/// exit path by the [`TempBuildDir`] guard.
fn compile_msl_via_xcrun(
    device: &Device,
    msl_source: &str,
    cache_path: &std::path::Path,
) -> Option<Library> {
    let seq = TEMP_BUILD_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_dir_path = std::env::temp_dir().join(format!(
        "oxibonsai_metal_build_{}_{seq}",
        std::process::id()
    ));
    if std::fs::create_dir(&tmp_dir_path).is_err() {
        return None;
    }
    let tmp_dir = TempBuildDir(tmp_dir_path);
    let dir = tmp_dir.path();

    let metal_path = dir.join("combined.metal");
    let air_path = dir.join("combined.air");
    let metallib_path = dir.join("combined.metallib");

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

    // Cache for future runs (MET-13: atomic publish — see
    // `write_metallib_cache_atomically`).
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    write_metallib_cache_atomically(cache_path, &metallib_data);

    // `tmp_dir`'s `Drop` removes the whole per-call build directory (and
    // every file still in it) once this function returns.
    device.new_library_with_data(&metallib_data).ok()
}

/// Publish `data` to `cache_path` atomically: write to a pid-suffixed
/// temporary file **in the same directory** as `cache_path` — required for
/// `std::fs::rename` to be atomic, since a rename across directories/
/// filesystems is not — then `rename` it into place. `cache_path.
/// with_extension(..)` only rewrites the file-name component, so the
/// temporary file is guaranteed to land next to `cache_path` regardless of
/// what directory that is.
///
/// Shared by both cache slots this module writes: the primary combined
/// metallib and the bf16 sidecar (`load_bf16_library` reaches this through
/// `compile_msl_via_xcrun`), so fixing it here fixes both.
///
/// # MET-13
///
/// The previous `std::fs::write(cache_path, &metallib_data)` was a single
/// non-atomic write. Two concurrent processes racing to populate the same
/// cache slot (same MSL source hash) could each observe a partial write from
/// the other, leaving a truncated `.metallib` on disk.
/// `try_load_cached_metallib`'s `new_library_with_data(...).ok()?` already
/// turns a corrupt file into a miss rather than a wrong load, so this was a
/// cold-start performance bug, not a correctness one — but every later
/// process would pay a full `xcrun` recompile until some process happened to
/// win the write race cleanly. `rename` within one directory is atomic on
/// every platform this project targets (POSIX `rename(2)`, Windows
/// `MoveFileEx` via `std::fs::rename`), so a reader can only ever observe the
/// old file or the fully-written new one, never a partial one.
fn write_metallib_cache_atomically(cache_path: &std::path::Path, data: &[u8]) {
    let tmp_path = cache_path.with_extension(format!("metallib.tmp.{}", std::process::id()));
    if let Err(e) = std::fs::write(&tmp_path, data) {
        tracing::debug!(
            "failed to write metallib cache temp file {}: {e}",
            tmp_path.display()
        );
        let _ = std::fs::remove_file(&tmp_path);
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, cache_path) {
        tracing::debug!(
            "failed to publish metallib cache {} (from {}): {e}",
            cache_path.display(),
            tmp_path.display()
        );
        let _ = std::fs::remove_file(&tmp_path);
    }
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
///
/// # MET-12: the embedded metallib is verified, not trusted blindly
///
/// `build.rs`'s `ACTIVE_KERNELS` whitelist and `build_combined_msl()` below
/// are two independently maintained lists that must name the exact same set
/// of kernels; nothing enforced the direction "everything
/// `build_combined_msl()` pushes was actually embedded" until now. Before
/// this fix, a kernel added to `build_combined_msl()` (and `pipeline_for`)
/// but forgotten in `ACTIVE_KERNELS` produced an embedded metallib silently
/// missing that entry point: `try_load_embedded_metallib` would still
/// succeed (the *library* loads fine — it is just missing one function), so
/// this function returned `Ok` immediately, `MetalPipelines::compile()`
/// then failed on the missing `pipeline_for(&library, device, "...")` call
/// with no per-function fallback, `MetalGraph::global()` failed, and every
/// caller's `.is_ok()` silently degraded the *entire* Metal backend to CPU.
///
/// Now, after loading the embedded library, every entry point
/// [`required_entry_points`] finds in the freshly computed `msl_source` is
/// checked with a cheap (non-compiling) `library.get_function` lookup; any
/// miss logs a warning and falls through to the disk-cache/xcrun/runtime
/// cascade below, which compiles fresh from `msl_source` and therefore
/// cannot itself be missing anything `build_combined_msl()` just produced —
/// turning the former hard failure into the cascade the module doc already
/// promised.
fn load_or_compile_library(device: &Device, msl_source: &str) -> Result<Library, MetalGraphError> {
    // 1. Try build-time embedded metallib (fastest: no I/O, no compilation) —
    //    but only trust it if every kernel `msl_source` actually calls for is
    //    present (MET-12).
    if let Some(lib) = try_load_embedded_metallib(device) {
        if embedded_library_has_all_required_entry_points(&lib, msl_source) {
            return Ok(lib);
        }
        tracing::warn!(
            "embedded metallib is missing one or more entry points required by the current \
             kernel_sources/ (ACTIVE_KERNELS in build.rs has drifted from build_combined_msl()); \
             falling back to disk-cache/xcrun/runtime compilation instead of shipping an \
             incomplete library"
        );
    }

    // The disk cache below is keyed by a hash of `msl_source` itself
    // (`kernels_<hash>.metallib`), so unlike the embedded metallib it cannot
    // suffer this same drift: any change to `msl_source` changes the cache
    // filename, making a stale/incomplete cache entry an ordinary miss
    // rather than a silently-served wrong library.
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

/// Compile the **optional** bf16 TE GEMM kernel into its own small library and
/// extract its pipeline, returning `None` on any failure.
///
/// `gemm_bf16_simdgroup` uses `simdgroup_matrix<bfloat,8,8>`, which needs
/// Apple-GPU `bfloat` simdgroup support (M3+/Metal 3.1). Keeping it OUT of the
/// combined metallib means the rest of the backend (DiT/VAE/TE-f32 — only
/// M1-class `half`/f32 simdgroup) always compiles; here a missing `bfloat`
/// feature simply yields `None` (the source fails to compile on an old SDK, or
/// pipeline creation fails on an old GPU) and the TE GEMM falls back to the
/// exact f32 kernel. The failure is therefore never propagated out of
/// [`MetalPipelines::compile`].
fn try_compile_bf16_pipeline(device: &Device) -> Option<ComputePipelineState> {
    let library = load_bf16_library(device)?;
    let func = library.get_function("gemm_bf16_simdgroup", None).ok()?;
    match device.new_compute_pipeline_state_with_function(&func) {
        Ok(pso) => Some(pso),
        Err(e) => {
            tracing::info!("bf16 TE GEMM unavailable on this device ({e}); using the f32 GEMM");
            None
        }
    }
}

/// Load the standalone bf16-kernel library: disk cache → `xcrun` → runtime MSL
/// compilation. Mirrors [`load_or_compile_library`] but (a) omits the embedded
/// (combined) metallib, and (b) returns `Option` because the bf16 kernel is
/// optional — any failure (e.g. no `bfloat` support in the toolchain) is a
/// silent `None`, not an error.
fn load_bf16_library(device: &Device) -> Option<Library> {
    let src = kernel_sources::MSL_GEMM_BF16_SIMDGROUP;
    let hash = msl_hash(src);
    let cache_filename = format!("bf16_{hash:016x}.metallib");

    if let Some(cache_dir) = metallib_cache_dir() {
        let cache_path = cache_dir.join(&cache_filename);
        if let Some(lib) = try_load_cached_metallib(device, &cache_path) {
            return Some(lib);
        }
        if let Some(lib) = compile_msl_via_xcrun(device, src, &cache_path) {
            return Some(lib);
        }
    }

    // Runtime fallback (no caching). `new_library_with_source` fails on a
    // toolchain without `bfloat` simdgroup support → `None`.
    let options = CompileOptions::new();
    device.new_library_with_source(src, &options).ok()
}

#[cfg(test)]
mod combined_msl_entry_point_tests {
    use super::*;

    /// MET-12 unit-level guard for [`required_entry_points`] itself.
    ///
    /// `tests/build_script_kernel_sources.rs` proves `build_combined_msl()`'s
    /// pushes exactly match `ACTIVE_KERNELS` (in both `build.rs` and its own
    /// mirror) — a *whitelist*-level check. Nothing previously exercised the
    /// *runtime* verifier this module actually ships — [`required_entry_points`]'s
    /// line-anchored `kernel void <name>(` scan — against the real combined
    /// MSL text it runs on in [`load_or_compile_library`]. A future MSL
    /// reformat (e.g. a declaration wrapping onto a second line before the
    /// `(`, such as `kernel void\nfoo(...)`) would silently make that scan
    /// under-count, with nothing catching it: `required_entry_points` would
    /// just return one name short, and `embedded_library_has_all_required_
    /// entry_points` would verify (and pass) a smaller-than-real set.
    ///
    /// This compares [`required_entry_points`]'s line-anchored count against
    /// a second, independent, **not** line-anchored count of the same marker
    /// text — deliberately not a hardcoded kernel count, so it never needs
    /// updating when a kernel is legitimately added or removed, but it DOES
    /// fire the moment the two extraction methods disagree, which is exactly
    /// the reformat hazard above (the naive substring count still finds a
    /// wrapped declaration; the line-anchored scan does not).
    #[test]
    fn required_entry_points_matches_a_naive_occurrence_count() {
        let msl = build_combined_msl();
        let scanned = required_entry_points(&msl);
        let naive_occurrences = msl.matches("kernel void ").count();

        assert_eq!(
            scanned.len(),
            naive_occurrences,
            "required_entry_points() (line-anchored) found {} entry point(s) but a naive \
             substring count of \"kernel void \" found {naive_occurrences} in the same text \
             ({scanned:?}); this means some `kernel void <name>(` declaration is no longer at \
             the start of its line (e.g. wrapped onto a second line before the `(`), so the \
             MET-12 runtime verifier in `load_or_compile_library` would silently under-check \
             the embedded metallib",
            scanned.len(),
        );

        // Non-triviality floor: catches `build_combined_msl()` being
        // accidentally emptied out, which would otherwise make both counts
        // agree at 0 and pass the equality check above vacuously.
        assert!(
            scanned.len() >= 40,
            "expected at least 40 Metal entry points in the combined MSL source, found {} — \
             build_combined_msl() looks broken or emptied out, not just missing one kernel",
            scanned.len()
        );
    }
}
