//! Runtime kernel dispatch with CPU feature detection.
//!
//! Uses SciRS2-Core's SIMD capability detection to select the best available
//! kernel implementation at runtime. Falls back to scalar reference
//! when no SIMD acceleration is available.
//!
//! The selection hierarchy (highest priority first):
//! 1. AVX-512F (x86-64 only)
//! 2. AVX2 + FMA (x86-64 only)
//! 3. NEON (AArch64 only)
//! 4. Reference (scalar — always available)
//!
//! ## The opt-in INT8 tier on the native formats (K-14)
//!
//! `OneBitKernel::{gemv, gemm}` (`Q1_0_g128`) and
//! `TernaryKernel::{gemv_ternary_g128, gemm_ternary_g128}` (`TQ2_0_g128`)
//! first ask [`KernelDispatcher::native_int8_tier`] whether
//! `OXIBONSAI_KERNEL_TIER` names an [`Int8Tier`]; when it does (and this is
//! a CPU-tier dispatcher) the call runs the matching
//! [`crate::dispatch_int8`] kernel instead. Unset — the default — every
//! call runs exactly the per-tier body it always ran, now held by the
//! `*_on_tier` methods below, so the default stays bit-identical. The
//! `parallel` / `parallel_tiled` drivers make the same check once at their
//! own entry and then call the `*_on_tier` methods from every tile, so the
//! environment is read once per call, never per tile.

use crate::dequant;
use crate::dispatch_int8::{self, Int8Tier};
use crate::error::KernelResult;
use crate::gemm;
use crate::gemv;
use crate::tier::TierReason;
use crate::traits::{Fp8Kernel, OneBitKernel, TernaryKernel};
use crate::weight_cache::GpuWeightHandle;
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, BlockTQ2_0_g128};
#[cfg(feature = "gpu")]
use std::sync::Arc;

// `KernelTier` (the tier enum + its `Display` impl), `TierReason`, and the
// pure CPU-feature-detection tier selection (`clamp_tier_to_cpu`,
// `select_tier`, `warn_gpu_unavailable_once`, `cpu_kernel_tier`) live in the
// sibling `tier` module, which keeps this file inside its 2000-line budget.
// Re-exported here so every existing `crate::dispatch::KernelTier` /
// `oxibonsai_kernels::KernelTier` path (and `lib.rs`'s own `pub use
// dispatch::{cpu_kernel_tier, KernelDispatcher, KernelTier};`) keeps
// resolving unchanged.
pub use crate::tier::{cpu_kernel_tier, KernelTier};

/// Dispatches kernel calls to the best available implementation.
///
/// Uses [`scirs2_core::simd::detect::CpuFeatures`] for CPU feature
/// detection, ensuring consistent SIMD dispatch across the COOLJAPAN ecosystem.
pub struct KernelDispatcher {
    tier: KernelTier,
    /// GPU backend handle, available when `gpu` feature is enabled and a
    /// hardware-accelerated backend was detected at construction time.
    #[cfg(feature = "gpu")]
    gpu_backend: Option<Arc<dyn crate::gpu_backend::GpuBackendTrait>>,
    tier_reason: TierReason,
}

impl std::fmt::Debug for KernelDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // GPU backend is a trait object without Debug; show only the tier.
        f.debug_struct("KernelDispatcher")
            .field("tier", &self.tier)
            .field("tier_reason", &self.tier_reason)
            .finish_non_exhaustive()
    }
}

impl KernelDispatcher {
    /// Create a dispatcher that auto-detects the best available kernel tier.
    ///
    /// Queries SciRS2-Core's cached `CpuFeatures` to determine the
    /// optimal tier for the current CPU.
    ///
    /// If the `gpu` feature is compiled in but no accelerated backend is
    /// found, this degrades to the best CPU SIMD tier (perf-13: ~1/7 the
    /// throughput) and emits a one-time-per-process `tracing::warn!` naming
    /// the reason, rather than only the `INFO`-level "selected kernel tier"
    /// line that gives no hint the degradation happened. Use
    /// [`Self::try_with_tier`]`(KernelTier::Gpu)` instead of this method
    /// when a missing GPU should be a hard error.
    ///
    /// Under an active
    /// [`CpuOnlyBackendScope`](crate::gpu_backend::CpuOnlyBackendScope) (the
    /// engine's explicit `Backend::Cpu`) the GPU is neither probed nor
    /// reported missing: the best CPU tier is chosen with the reason "CPU
    /// requested explicitly", because an explicit CPU request is not a
    /// degradation and the one-time "GPU unavailable" warning would
    /// misdescribe it.
    pub fn auto_detect() -> Self {
        if crate::gpu_backend::cpu_only_backend_active() {
            let caps = scirs2_core::simd::detect::get_cpu_features();
            let tier = Self::select_tier(caps);
            tracing::info!(tier = %tier, "selected kernel tier (CPU requested explicitly)");
            return Self {
                tier,
                #[cfg(feature = "gpu")]
                gpu_backend: None,
                tier_reason: TierReason::CpuRequestedByScope,
            };
        }

        // Try GPU first when the feature is compiled in.
        #[cfg(feature = "gpu")]
        {
            #[cfg(test)]
            tests::note_gpu_probe();
            let backend = crate::gpu_backend::select_backend();
            if backend.is_accelerated() {
                let name = backend.name();
                tracing::info!(backend = name, "GPU backend available");
                return Self {
                    tier: KernelTier::Gpu,
                    tier_reason: TierReason::GpuBackend(name),
                    gpu_backend: Some(Arc::from(backend)),
                };
            }
            Self::warn_gpu_unavailable_once(backend.name());
        }

        let caps = scirs2_core::simd::detect::get_cpu_features();
        let tier = Self::select_tier(caps);
        tracing::info!(tier = %tier, "selected kernel tier");

        #[cfg(feature = "gpu")]
        let tier_reason = TierReason::CpuAutoDetectGpuUnavailable;
        #[cfg(not(feature = "gpu"))]
        let tier_reason = TierReason::CpuAutoDetectNoGpuFeature;

        Self {
            tier,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
            tier_reason,
        }
    }

    /// Create a dispatcher with a specific tier (for testing/benchmarks).
    ///
    /// The requested tier is re-validated against the CPU's actual feature
    /// set via `Self::clamp_tier_to_cpu` and silently demoted (with a
    /// `tracing::warn!`) if unsupported (K-03/sec-14), so this constructor
    /// can never hand back a dispatcher that would later hit an
    /// illegal-instruction trap in the `unsafe { #[target_feature] }`
    /// AVX2/AVX-512 kernels. A no-op for a tier the CPU already supports —
    /// e.g. `with_tier(cpu_kernel_tier())` — and on aarch64 (where NEON is
    /// the ISA baseline) always a no-op.
    ///
    /// Use [`Self::try_with_tier`] to be told about a demotion via `Err`
    /// instead of silently accepting it, or the `unsafe`
    /// [`Self::with_tier_unchecked`] to bypass validation entirely
    /// (benchmarks only).
    pub fn with_tier(tier: KernelTier) -> Self {
        let clamped = Self::clamp_tier_to_cpu(tier);
        let tier_reason = if clamped == tier {
            TierReason::Requested
        } else {
            TierReason::Demoted(tier)
        };
        Self {
            tier: clamped,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
            tier_reason,
        }
    }

    /// Construct a dispatcher with an explicitly-requested tier, rejecting
    /// one the current hardware cannot execute instead of silently
    /// constructing a [`Self::with_tier`] that demotes quietly (K-03/sec-14)
    /// or, for `KernelTier::Gpu`, silently running every op on the CPU
    /// fallback path (perf-13).
    ///
    /// # Errors
    ///
    /// - For `tier == KernelTier::Gpu` (when the `gpu` feature is compiled
    ///   in): [`crate::error::KernelError::UnsupportedOperation`] naming the
    ///   backend if no accelerated GPU backend is available.
    /// - For `Avx2`/`Avx512` on x86-64: the same error naming the best tier
    ///   this CPU actually supports, if the requested one is unsupported.
    /// - `Reference` and (on aarch64) `Neon` always succeed.
    pub fn try_with_tier(tier: KernelTier) -> KernelResult<Self> {
        #[cfg(feature = "gpu")]
        if tier == KernelTier::Gpu {
            let backend = crate::gpu_backend::select_backend();
            if !backend.is_accelerated() {
                return Err(crate::error::KernelError::UnsupportedOperation(format!(
                    "GPU kernel tier was explicitly requested but no accelerated GPU \
                     backend is available (selected backend '{}' reports not accelerated)",
                    backend.name()
                )));
            }
            let name = backend.name();
            return Ok(Self {
                tier: KernelTier::Gpu,
                tier_reason: TierReason::GpuBackend(name),
                gpu_backend: Some(Arc::from(backend)),
            });
        }

        let clamped = Self::clamp_tier_to_cpu(tier);
        if clamped != tier {
            return Err(crate::error::KernelError::UnsupportedOperation(format!(
                "kernel tier {tier} was explicitly requested but is not supported by this \
                 CPU (best available: {clamped})"
            )));
        }
        Ok(Self {
            tier,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
            tier_reason: TierReason::Requested,
        })
    }

    /// Construct a dispatcher with the given tier **without** re-validating
    /// it against the running CPU's feature set.
    ///
    /// For benchmarks that pin a specific tier for comparison purposes on
    /// hardware already known to support it (`benches/kernel_benchmarks.rs`).
    /// Prefer [`Self::try_with_tier`] or the clamped [`Self::with_tier`]
    /// everywhere else.
    ///
    /// # Safety
    ///
    /// The caller must ensure `tier` is actually executable on the current
    /// CPU: requesting `Avx512`/`Avx2` on hardware lacking those extensions
    /// makes any subsequent kernel call that reaches the corresponding
    /// `unsafe { #[target_feature] }` function undefined behavior (illegal
    /// instruction / miscompiled SIMD).
    pub unsafe fn with_tier_unchecked(tier: KernelTier) -> Self {
        Self {
            tier,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
            tier_reason: TierReason::Unchecked,
        }
    }

    /// Create a dispatcher that uses the GPU backend with the given handle.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(backend: Arc<dyn crate::gpu_backend::GpuBackendTrait>) -> Self {
        let tier_reason = TierReason::GpuHandle(backend.name());
        Self {
            tier: KernelTier::Gpu,
            gpu_backend: Some(backend),
            tier_reason,
        }
    }

    /// Get the selected kernel tier.
    pub fn tier(&self) -> KernelTier {
        self.tier
    }

    /// The GPU backend handle, if this dispatcher was constructed with one
    /// (`auto_detect`/`try_with_tier(Gpu)`/`with_gpu` on a machine with an
    /// accelerated backend) — `None` on every CPU tier and on a `Gpu`-tier
    /// dispatcher built without the `gpu` feature.
    ///
    /// This is how `impl Drop for InferenceEngine`
    /// (`oxibonsai-runtime/src/engine.rs`) reaches the backend to evict its
    /// GPU weight cache (`Scirs2Backend::clear_weight_cache`) on drop
    /// (MET-M1).
    #[cfg(feature = "gpu")]
    pub fn gpu_backend(&self) -> Option<&Arc<dyn crate::gpu_backend::GpuBackendTrait>> {
        self.gpu_backend.as_ref()
    }

    /// Tier-only portion of [`OneBitKernel::name`] / [`Self::kernel_label`]
    /// (cli-16) — no quant-family prefix, since a dispatcher instance is
    /// shared across every quant-kernel trait and cannot know on its own
    /// which one a given caller is using.
    fn tier_label(&self) -> &'static str {
        match self.tier {
            KernelTier::Reference => "reference (scalar)",
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => "AVX2+FMA (256-bit)",
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => "AVX-512 (512-bit)",
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => "NEON (128-bit)",
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => "GPU (accelerated)",
        }
    }

    /// A human-readable label combining a model's resolved dominant quant
    /// type with this dispatcher's kernel tier, e.g.
    /// `"TQ2_0_g128 NEON (128-bit)"` (cli-16).
    ///
    /// [`OneBitKernel::name`] cannot include the quant family for the reason
    /// `Self::tier_label` documents; a caller that already knows which
    /// type it resolved a model to (a model tracking its own
    /// `dominant_quant_type`) calls this instead — e.g.
    /// `oxibonsai-runtime/src/engine.rs`'s `"inference engine loaded from
    /// GGUF kernel=..."` log lines and `oxibonsai-runtime/src/engine_seam.rs`'s
    /// `InferenceEngine::kernel_label`.
    pub fn kernel_label(&self, dominant_type: oxibonsai_core::GgufTensorType) -> String {
        format!("{dominant_type} {}", self.tier_label())
    }

    /// Human-readable explanation of how the effective kernel tier was
    /// chosen — e.g. `"gpu tier (backend=metal)"` or `"neon tier (requested
    /// avx2 not supported by this CPU, demoted)"`. For the CLI build-info
    /// surface (cli-19) and `/admin/status` so a GPU-unavailable degradation
    /// to a much slower CPU tier is observable instead of silent (perf-13).
    pub fn effective_tier_reason(&self) -> String {
        match self.tier_reason {
            #[cfg(feature = "gpu")]
            TierReason::GpuBackend(name) => format!("gpu tier (backend={name})"),
            #[cfg(feature = "gpu")]
            TierReason::GpuHandle(name) => {
                format!("gpu tier (backend={name}, pre-built handle)")
            }
            #[cfg(not(feature = "gpu"))]
            TierReason::CpuAutoDetectNoGpuFeature => {
                format!(
                    "{} tier (auto-detected; gpu feature not compiled in)",
                    self.tier
                )
            }
            #[cfg(feature = "gpu")]
            TierReason::CpuAutoDetectGpuUnavailable => format!(
                "{} tier (auto-detected; GPU backend not accelerated, fell back)",
                self.tier
            ),
            TierReason::CpuRequestedByScope => format!(
                "{} tier (CPU requested explicitly, backend=cpu; no GPU probe)",
                self.tier
            ),
            TierReason::Requested => format!("{} tier (explicitly requested)", self.tier),
            TierReason::Demoted(requested) => format!(
                "{} tier (requested {requested} not supported by this CPU, demoted)",
                self.tier
            ),
            TierReason::Unchecked => {
                format!("{} tier (unchecked, caller-validated)", self.tier)
            }
        }
    }
}

// `clamp_tier_to_cpu`, `select_tier` (both used by the constructors above via
// `Self::`) and the free function `cpu_kernel_tier` now live in `crate::tier`
// (re-exported at the top of this file) — see that module's doc comment.

/// Minimum number of rows before the GPU path is worthwhile.
///
/// Below this threshold the overhead of host-to-device transfer exceeds the
/// compute savings, so we fall back to the best SIMD tier.
#[cfg(feature = "gpu")]
const GPU_MIN_ROWS: usize = 1024;

impl KernelDispatcher {
    /// Dense FP32 GEMV — the LM-head projection (K-12 / M-23).
    ///
    /// `out[..out_features] = weights[out_features × in_features] · input[..in_features]`,
    /// row-major, with an empty `weights` denoting the all-zero projection of
    /// the weightless config-only model constructors (M-33).
    ///
    /// This is the dispatcher seam the FP32 LM head previously did not have:
    /// `oxibonsai-model`'s `apply_lm_head` used to call a copy of the kernel
    /// private to `model/types/lm_head.rs`, so the GPU tiers could never claim
    /// the single largest GEMV in the model (`248 320 × 5120` on Bonsai 2 27B).
    /// Every tier now arrives here.
    ///
    /// # Tier behaviour
    ///
    /// Every tier — including [`KernelTier::Gpu`] — currently executes
    /// [`crate::gemv_f32::gemv_f32`], whose NEON-`fmla` / AVX2-`vfmadd` lane
    /// structure already covers the CPU tiers without a per-tier body. `Gpu`
    /// deliberately runs the same CPU kernel rather than degrading silently
    /// to *something else*: neither
    /// [`GpuBackendTrait`](crate::gpu_backend::GpuBackendTrait) nor the Metal
    /// / CUDA kernel-source sets carries a dense-FP32 GEMV entry point yet, and
    /// the parity gate requires this projection to stay byte-identical
    /// to the pre-hoist body on a real model. When a GPU FP32 GEMV lands, this
    /// method is the one place that routes to it, and the change becomes
    /// visible to `apply_lm_head` without touching `oxibonsai-model` at all.
    ///
    /// # Errors
    ///
    /// Propagates [`crate::gemv_f32::gemv_f32`]'s named shape errors.
    pub fn gemv_f32(
        &self,
        weights: &[f32],
        input: &[f32],
        out: &mut [f32],
        out_features: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        // Single body for every tier today — deliberately not a `match` on
        // `self.tier`, which would be five arms doing the same thing. The
        // dispatcher seam is what matters: this is the one call site a future
        // GPU (or per-tier CPU) FP32 GEMV has to be wired into, and every
        // caller already arrives through it.
        crate::gemv_f32::gemv_f32(weights, input, out, out_features, in_features)
    }

    /// Return the best CPU-only tier for use as GPU fallback.
    ///
    /// Uses Rust's built-in `is_x86_feature_detected!` macros directly
    /// for reliable detection, bypassing scirs2_core which may have issues
    /// on certain platforms.
    ///
    /// `pub(crate)` (not private) so the sibling dispatch-split files
    /// (`dispatch_std_quant.rs`, `dispatch_prism.rs`) can call it from their
    /// own `cpu_*_fallback` methods.
    #[cfg(feature = "gpu")]
    pub(crate) fn cpu_tier() -> KernelTier {
        #[cfg(target_arch = "x86_64")]
        {
            let has_avx512f = is_x86_feature_detected!("avx512f");
            let has_avx512bw = is_x86_feature_detected!("avx512bw");
            let has_avx512vl = is_x86_feature_detected!("avx512vl");
            let has_avx2 = is_x86_feature_detected!("avx2");
            let has_fma = is_x86_feature_detected!("fma");

            if has_avx512f && has_avx512bw && has_avx512vl {
                return KernelTier::Avx512;
            }
            if has_avx2 && has_fma {
                return KernelTier::Avx2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            return KernelTier::Neon;
        }

        #[allow(unreachable_code)]
        KernelTier::Reference
    }

    /// Dispatch a `dequant` call using the best *CPU* tier.
    #[cfg(feature = "gpu")]
    fn cpu_dequant(blocks: &[BlockQ1_0G128], output: &mut [f32]) -> KernelResult<()> {
        match Self::cpu_tier() {
            KernelTier::Reference => dequant::dequant_1bit_g128(blocks, output),
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe { crate::simd_avx2::dequant_1bit_g128_avx2(blocks, output) },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::dequant_1bit_g128_avx512(blocks, output)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe { crate::simd_neon::dequant_1bit_g128_neon(blocks, output) },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => dequant::dequant_1bit_g128(blocks, output),
        }
    }

    /// Dispatch a `gemv` call using the best *CPU* tier.
    #[cfg(feature = "gpu")]
    fn cpu_gemv(
        blocks: &[BlockQ1_0G128],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match Self::cpu_tier() {
            KernelTier::Reference => gemv::gemv_1bit_g128(blocks, input, output, n_rows, k),
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_avx2::gemv_1bit_g128_avx2_prefetch(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::gemv_1bit_g128_avx512_auto(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_neon::gemv_1bit_g128_neon_prefetch(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => gemv::gemv_1bit_g128(blocks, input, output, n_rows, k),
        }
    }

    /// Dispatch a `gemm` call using the best *CPU* tier.
    #[cfg(feature = "gpu")]
    fn cpu_gemm(
        blocks: &[BlockQ1_0G128],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match Self::cpu_tier() {
            KernelTier::Reference => gemm::gemm_1bit_g128(blocks, input, output, m, n_rows, k),
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_avx2::gemm_1bit_g128_avx2_prefetch(blocks, input, output, m, n_rows, k)
            },
            // No prefetch variant for AVX-512 GEMM — keep non-prefetch.
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::gemm_1bit_g128_avx512(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_neon::gemm_1bit_g128_neon_prefetch(blocks, input, output, m, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => gemm::gemm_1bit_g128(blocks, input, output, m, n_rows, k),
        }
    }

    /// Dispatch a `dequant_ternary` call using the best *CPU* tier.
    #[cfg(feature = "gpu")]
    fn cpu_dequant_ternary(
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
        output: &mut [f32],
    ) -> KernelResult<()> {
        match Self::cpu_tier() {
            KernelTier::Reference => crate::dequant_ternary::dequant_tq2_0_g128(blocks, output),
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_avx2::dequant_tq2_0_g128_avx2(blocks, output)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::dequant_tq2_0_g128_avx512(blocks, output)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_neon::dequant_tq2_0_g128_neon(blocks, output)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => crate::dequant_ternary::dequant_tq2_0_g128(blocks, output),
        }
    }

    /// Dispatch a `gemv_ternary` call using the best *CPU* tier.
    #[cfg(feature = "gpu")]
    fn cpu_gemv_ternary(
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match Self::cpu_tier() {
            KernelTier::Reference => {
                crate::gemv_ternary::gemv_tq2_0_g128(blocks, input, output, n_rows, k)
            }
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_avx2::gemv_tq2_0_g128_avx2_prefetch(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::gemv_tq2_0_g128_avx512_prefetch(
                    blocks, input, output, n_rows, k,
                )
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_neon::gemv_tq2_0_g128_neon_prefetch(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                crate::gemv_ternary::gemv_tq2_0_g128(blocks, input, output, n_rows, k)
            }
        }
    }

    /// Dispatch a `gemm_ternary` call using the best *CPU* tier.
    #[cfg(feature = "gpu")]
    fn cpu_gemm_ternary(
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match Self::cpu_tier() {
            KernelTier::Reference => {
                crate::gemm_ternary::gemm_tq2_0_g128(blocks, input, output, m, n_rows, k)
            }
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_avx2::gemm_tq2_0_g128_avx2(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::gemm_tq2_0_g128_avx512(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_neon::gemm_tq2_0_g128_neon(blocks, input, output, m, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                crate::gemm_ternary::gemm_tq2_0_g128(blocks, input, output, m, n_rows, k)
            }
        }
    }

    // `cpu_gemv_q4_0_fallback` / `cpu_gemv_q8_0_fallback` (K-17) now live in
    // `dispatch_std_quant.rs`, alongside the `StandardQuantKernel` impl that
    // calls them (a single trait impl cannot itself be split across files,
    // so it made sense to move its fallbacks with it).

    /// Reinterpret a slice of `BlockQ1_0G128` as raw bytes (zero-copy).
    ///
    /// # Safety
    /// `BlockQ1_0G128` is `#[repr(C)]` with a well-defined 18-byte layout,
    /// so this transmute is safe.
    #[cfg(feature = "gpu")]
    fn blocks_as_bytes(blocks: &[BlockQ1_0G128]) -> &[u8] {
        let ptr = blocks.as_ptr() as *const u8;
        let len = std::mem::size_of_val(blocks);
        // SAFETY: BlockQ1_0G128 is repr(C), POD-like, with no padding.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

// ─── Native-format GEMV/GEMM: INT8 selection + per-tier bodies ────────────

impl KernelDispatcher {
    /// The opt-in INT8 tier (K-14) this dispatcher's native-format
    /// (`Q1_0_g128` / `TQ2_0_g128`) GEMV and GEMM calls run on, if any.
    ///
    /// `Some` only when `OXIBONSAI_KERNEL_TIER` names an [`Int8Tier`]
    /// ([`Int8Tier::from_env`], read fresh on every call, clamped to what
    /// this CPU supports) **and** this is a CPU-tier dispatcher. A
    /// `KernelTier::Gpu` dispatcher is never diverted — neither its GPU
    /// kernels nor its CPU fallbacks — so a GPU engine's own forward pass
    /// and the CPU-vs-Metal determinism guard stay independent of the
    /// selector. `None`, the default, means every native GEMV/GEMM runs
    /// exactly the per-tier kernel it ran before the INT8 tier existed.
    ///
    /// This is per-*dispatcher*, not per-engine: `oxibonsai-model`'s
    /// batched CPU prefill (`prefill_cpu::prefill_dispatcher`) and its
    /// batched embedding pass always build their own CPU-tier dispatcher and
    /// call through it, so a GPU engine whose fused GPU prefill is declined
    /// or fails, or a batched embedding call on any engine, still runs that
    /// CPU-side work on the INT8 tier when the variable is set — the
    /// sentence above is only about *this* dispatcher's own tier. The
    /// per-token embedding fallback (`forward_hidden_sequential`, taken when
    /// the batched pass declines) is not one of these: it runs on the
    /// *caller's* dispatcher, so it is diverted only when that dispatcher is
    /// itself CPU-tier with the variable set, exactly like any other
    /// per-token forward call.
    ///
    /// Every native entry point asks this once, at entry:
    /// `OneBitKernel::{gemv, gemm}`, `TernaryKernel::{gemv_ternary_g128,
    /// gemm_ternary_g128}`, and the `parallel` / `parallel_tiled` drivers.
    #[must_use]
    pub fn native_int8_tier(&self) -> Option<Int8Tier> {
        #[cfg(feature = "gpu")]
        if self.tier == KernelTier::Gpu {
            return None;
        }
        Int8Tier::from_env()
    }

    /// `Q1_0_g128` GEMV on this dispatcher's own [`KernelTier`], never the
    /// INT8 tier — the body `OneBitKernel::gemv` runs when
    /// [`Self::native_int8_tier`] is `None`, and what every f32 tile and
    /// chunk of the `parallel` / `parallel_tiled` drivers calls after their
    /// single entry-point check (so no tile re-reads the environment).
    pub(crate) fn gemv_1bit_on_tier(
        &self,
        blocks: &[BlockQ1_0G128],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match self.tier {
            KernelTier::Reference => gemv::gemv_1bit_g128(blocks, input, output, n_rows, k),
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_avx2::gemv_1bit_g128_avx2_prefetch(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::gemv_1bit_g128_avx512_auto(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_neon::gemv_1bit_g128_neon_prefetch(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                if n_rows < GPU_MIN_ROWS {
                    return Self::cpu_gemv(blocks, input, output, n_rows, k);
                }
                if let Some(ref backend) = self.gpu_backend {
                    let bytes = Self::blocks_as_bytes(blocks);
                    match backend.gemv_q1_g128(bytes, input, n_rows, k) {
                        Ok(result) => {
                            let copy_len = output.len().min(result.len());
                            output[..copy_len].copy_from_slice(&result[..copy_len]);
                            return Ok(());
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "GPU gemv failed, falling back to CPU");
                            return Self::cpu_gemv(blocks, input, output, n_rows, k);
                        }
                    }
                }
                Self::cpu_gemv(blocks, input, output, n_rows, k)
            }
        }
    }

    /// `Q1_0_g128` GEMM on this dispatcher's own [`KernelTier`], never the
    /// INT8 tier — see [`Self::gemv_1bit_on_tier`].
    pub(crate) fn gemm_1bit_on_tier(
        &self,
        blocks: &[BlockQ1_0G128],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match self.tier {
            KernelTier::Reference => gemm::gemm_1bit_g128(blocks, input, output, m, n_rows, k),
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_avx2::gemm_1bit_g128_avx2_prefetch(blocks, input, output, m, n_rows, k)
            },
            // No prefetch variant for AVX-512 GEMM — keep non-prefetch.
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::gemm_1bit_g128_avx512(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_neon::gemm_1bit_g128_neon_prefetch(blocks, input, output, m, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                if n_rows < GPU_MIN_ROWS {
                    return Self::cpu_gemm(blocks, input, output, m, n_rows, k);
                }
                if let Some(ref backend) = self.gpu_backend {
                    let bytes = Self::blocks_as_bytes(blocks);
                    match backend.gemm_q1_g128(bytes, input, m, n_rows, k) {
                        Ok(result) => {
                            let copy_len = output.len().min(result.len());
                            output[..copy_len].copy_from_slice(&result[..copy_len]);
                            return Ok(());
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "GPU gemm failed, falling back to CPU");
                            return Self::cpu_gemm(blocks, input, output, m, n_rows, k);
                        }
                    }
                }
                Self::cpu_gemm(blocks, input, output, m, n_rows, k)
            }
        }
    }

    /// `TQ2_0_g128` GEMV on this dispatcher's own [`KernelTier`], never the
    /// INT8 tier — see [`Self::gemv_1bit_on_tier`].
    pub(crate) fn gemv_ternary_on_tier(
        &self,
        blocks: &[BlockTQ2_0_g128],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match self.tier {
            KernelTier::Reference => {
                crate::gemv_ternary::gemv_tq2_0_g128(blocks, input, output, n_rows, k)
            }
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_avx2::gemv_tq2_0_g128_avx2_prefetch(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::gemv_tq2_0_g128_avx512_prefetch(
                    blocks, input, output, n_rows, k,
                )
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_neon::gemv_tq2_0_g128_neon_prefetch(blocks, input, output, n_rows, k)
            },
            // No ternary GPU kernels — fall back to best CPU SIMD tier.
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => Self::cpu_gemv_ternary(blocks, input, output, n_rows, k),
        }
    }

    /// `TQ2_0_g128` GEMM on this dispatcher's own [`KernelTier`], never the
    /// INT8 tier — see [`Self::gemv_1bit_on_tier`].
    pub(crate) fn gemm_ternary_on_tier(
        &self,
        blocks: &[BlockTQ2_0_g128],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match self.tier {
            KernelTier::Reference => {
                crate::gemm_ternary::gemm_tq2_0_g128(blocks, input, output, m, n_rows, k)
            }
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_avx2::gemm_tq2_0_g128_avx2(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::gemm_tq2_0_g128_avx512(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_neon::gemm_tq2_0_g128_neon(blocks, input, output, m, n_rows, k)
            },
            // No ternary GPU kernels — fall back to best CPU SIMD tier.
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => Self::cpu_gemm_ternary(blocks, input, output, m, n_rows, k),
        }
    }
}

impl OneBitKernel for KernelDispatcher {
    fn dequant(&self, blocks: &[BlockQ1_0G128], output: &mut [f32]) -> KernelResult<()> {
        match self.tier {
            KernelTier::Reference => dequant::dequant_1bit_g128(blocks, output),
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe { crate::simd_avx2::dequant_1bit_g128_avx2(blocks, output) },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::dequant_1bit_g128_avx512(blocks, output)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe { crate::simd_neon::dequant_1bit_g128_neon(blocks, output) },
            // GPU dequant is not worth the transfer cost — use best CPU path.
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => Self::cpu_dequant(blocks, output),
        }
    }

    /// The opt-in INT8 tier when [`KernelDispatcher::native_int8_tier`]
    /// selects one, else `KernelDispatcher::gemv_1bit_on_tier` — today's
    /// per-tier body, unchanged.
    fn gemv(
        &self,
        blocks: &[BlockQ1_0G128],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        if let Some(tier) = self.native_int8_tier() {
            return dispatch_int8::gemv_1bit_g128_int8(tier, blocks, input, output, n_rows, k);
        }
        self.gemv_1bit_on_tier(blocks, input, output, n_rows, k)
    }

    /// The opt-in INT8 tier when [`KernelDispatcher::native_int8_tier`]
    /// selects one, else `KernelDispatcher::gemm_1bit_on_tier`.
    fn gemm(
        &self,
        blocks: &[BlockQ1_0G128],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        if let Some(tier) = self.native_int8_tier() {
            return dispatch_int8::gemm_1bit_g128_int8(tier, blocks, input, output, m, n_rows, k);
        }
        self.gemm_1bit_on_tier(blocks, input, output, m, n_rows, k)
    }

    /// cli-16: this used to hardcode a `"Q1_0_g128 "` prefix regardless of
    /// which quant family the caller was actually running — one
    /// `KernelDispatcher` backs `OneBitKernel`/`TernaryKernel`/
    /// `StandardQuantKernel`/`Fp8Kernel`/`PrismKernel` alike, so it has no
    /// way to know which one a given caller cares about. The tier portion
    /// was always correct; only the quant-family guess was wrong. Fixed by
    /// dropping the guess (see `Self::tier_label`) rather than by
    /// rewriting the tier text. [`Self::kernel_label`] is the parameterized
    /// replacement for a caller that *does* know the model's resolved quant
    /// type.
    fn name(&self) -> &'static str {
        self.tier_label()
    }

    fn is_gpu_accelerated(&self) -> bool {
        #[cfg(feature = "gpu")]
        let answer = self.tier == KernelTier::Gpu;
        #[cfg(not(feature = "gpu"))]
        let answer = false;
        answer
    }

    fn upload_weights(&self, blocks: &[BlockQ1_0G128]) -> Option<GpuWeightHandle> {
        #[cfg(feature = "gpu")]
        {
            if let (KernelTier::Gpu, Some(ref backend)) = (self.tier, &self.gpu_backend) {
                let bytes = Self::blocks_as_bytes(blocks);
                match backend.upload_weights_raw(bytes) {
                    Ok(handle) => return Some(handle),
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to upload weights to GPU");
                    }
                }
            }
        }
        let _ = blocks;
        None
    }

    fn gemv_cached(
        &self,
        handle: GpuWeightHandle,
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        #[cfg(feature = "gpu")]
        {
            if let (KernelTier::Gpu, Some(ref backend)) = (self.tier, &self.gpu_backend) {
                match backend.gemv_q1_g128_cached(handle, input, n_rows, k) {
                    Ok(result) => {
                        let len = output.len().min(result.len());
                        output[..len].copy_from_slice(&result[..len]);
                        return Ok(());
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "cached GPU gemv failed, cannot fallback without blocks");
                        return Err(crate::error::KernelError::GpuError(e.to_string()));
                    }
                }
            }
        }
        let _ = (handle, input, output, n_rows, k);
        Err(crate::error::KernelError::UnsupportedOperation(
            "gemv_cached requires GPU tier".into(),
        ))
    }

    fn batch_attn_phase(
        &self,
        hidden: &[f32],
        norm_weight: &[f32],
        norm_eps: f32,
        qkv_handle: GpuWeightHandle,
        q_rows: usize,
        k_rows: usize,
        h: usize,
    ) -> KernelResult<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
        // Disabled: CPU RMSNorm + single fused GEMV is faster than
        // GPU batch (dispatch_no_wait + dispatch) for only 2 operations.
        // The GPU batch creates 4 new Metal buffers per call; the fallback
        // reuses pre-allocated io_input/output buffers.
        //
        // K-20: this is a measurement-driven `Ok(None)`, not a missing
        // feature — do not "finish" this by wiring in a call. The real,
        // working Metal implementation this permanently skips lives at
        // `gpu_backend/scirs2_backend.rs`'s `batch_attn_phase` (wired into
        // `GpuBackendTrait` at `gpu_backend/mod.rs`); it stays reachable via
        // that trait impl for a future caller (the `forward_sw.rs` /
        // `forward_stats.rs` revival point) that changes the underlying
        // cost model, e.g. batching many more than 2 ops together. See the
        // `#[doc(hidden)]` note on `OneBitKernel::batch_attn_phase` in
        // `traits.rs`.
        let _ = (hidden, norm_weight, norm_eps, qkv_handle, q_rows, k_rows, h);
        Ok(None)
    }

    fn batch_ffn_phase(
        &self,
        hidden: &mut [f32],
        attn_out: &[f32],
        norm_weight: &[f32],
        norm_eps: f32,
        attn_proj_handle: GpuWeightHandle,
        gate_up_handle: GpuWeightHandle,
        down_handle: GpuWeightHandle,
        h: usize,
        intermediate: usize,
        attn_proj_k: usize,
    ) -> KernelResult<bool> {
        #[cfg(feature = "gpu")]
        {
            if let (KernelTier::Gpu, Some(ref backend)) = (self.tier, &self.gpu_backend) {
                match backend.batch_ffn_phase(
                    hidden,
                    attn_out,
                    norm_weight,
                    norm_eps,
                    attn_proj_handle,
                    gate_up_handle,
                    down_handle,
                    h,
                    intermediate,
                    attn_proj_k,
                ) {
                    Ok(true) => return Ok(true),
                    Ok(false) => return Ok(false),
                    Err(e) => {
                        tracing::warn!(error = %e, "batch FFN phase failed, falling back");
                        return Ok(false);
                    }
                }
            }
        }
        let _ = (
            hidden,
            attn_out,
            norm_weight,
            norm_eps,
            attn_proj_handle,
            gate_up_handle,
            down_handle,
            h,
            intermediate,
            attn_proj_k,
        );
        Ok(false)
    }
}

impl TernaryKernel for KernelDispatcher {
    fn dequant_ternary_g128(
        &self,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
        output: &mut [f32],
    ) -> KernelResult<()> {
        match self.tier {
            KernelTier::Reference => crate::dequant_ternary::dequant_tq2_0_g128(blocks, output),
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_avx2::dequant_tq2_0_g128_avx2(blocks, output)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_avx512::dequant_tq2_0_g128_avx512(blocks, output)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_neon::dequant_tq2_0_g128_neon(blocks, output)
            },
            // No ternary GPU kernels — fall back to best CPU SIMD tier.
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => Self::cpu_dequant_ternary(blocks, output),
        }
    }

    /// The opt-in INT8 tier (the legacy `{-1, 0, +1, 0}` table, K-01) when
    /// [`KernelDispatcher::native_int8_tier`] selects one, else
    /// `KernelDispatcher::gemv_ternary_on_tier`.
    fn gemv_ternary_g128(
        &self,
        blocks: &[BlockTQ2_0_g128],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        if let Some(tier) = self.native_int8_tier() {
            return dispatch_int8::gemv_two_bit_int8(tier, blocks, input, output, n_rows, k);
        }
        self.gemv_ternary_on_tier(blocks, input, output, n_rows, k)
    }

    /// The opt-in INT8 tier when [`KernelDispatcher::native_int8_tier`]
    /// selects one, else `KernelDispatcher::gemm_ternary_on_tier`.
    fn gemm_ternary_g128(
        &self,
        blocks: &[BlockTQ2_0_g128],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        if let Some(tier) = self.native_int8_tier() {
            return dispatch_int8::gemm_two_bit_int8(tier, blocks, input, output, m, n_rows, k);
        }
        self.gemm_ternary_on_tier(blocks, input, output, m, n_rows, k)
    }

    fn upload_weights_ternary(
        &self,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    ) -> Option<GpuWeightHandle> {
        #[cfg(feature = "gpu")]
        {
            if let (KernelTier::Gpu, Some(ref backend)) = (self.tier, &self.gpu_backend) {
                match backend.upload_weights_ternary(blocks) {
                    Ok(handle) => return Some(handle),
                    Err(e) => {
                        // Some backends (e.g. NativeCudaBackend without a TQ2 kernel)
                        // legitimately don't support ternary uploads. We get called
                        // once per ternary weight tensor at model load — log just
                        // once to avoid hundreds of identical warnings.
                        use std::sync::atomic::{AtomicBool, Ordering};
                        static WARNED: AtomicBool = AtomicBool::new(false);
                        if !WARNED.swap(true, Ordering::Relaxed) {
                            tracing::warn!(
                                error = %e,
                                backend = backend.name(),
                                "ternary weight GPU upload not supported by backend; \
                                 falling back to CPU SIMD for ternary GEMV (this message \
                                 is shown once per process)"
                            );
                        }
                    }
                }
            }
        }
        let _ = blocks;
        None
    }

    fn gemv_ternary_g128_cached(
        &self,
        handle: GpuWeightHandle,
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        #[cfg(feature = "gpu")]
        {
            if let (KernelTier::Gpu, Some(ref backend)) = (self.tier, &self.gpu_backend) {
                match backend.gemv_tq2_g128_cached(handle, input, n_rows, k) {
                    Ok(result) => {
                        let len = output.len().min(result.len());
                        output[..len].copy_from_slice(&result[..len]);
                        return Ok(());
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "cached GPU ternary gemv failed, cannot fallback without blocks");
                        return Err(crate::error::KernelError::GpuError(e.to_string()));
                    }
                }
            }
        }
        let _ = (handle, input, output, n_rows, k);
        Err(crate::error::KernelError::UnsupportedOperation(
            "gemv_ternary_g128_cached requires GPU tier".into(),
        ))
    }
}

// Compile-time size checks: BlockFP8E4M3/E5M2 must be exactly BLOCK_FP8_BYTES (34).
// This ensures the raw-pointer cast in the CUDA GPU dispatch is sound.
const _: () =
    assert!(std::mem::size_of::<oxibonsai_core::BlockFP8E4M3>() == oxibonsai_core::BLOCK_FP8_BYTES);
const _: () =
    assert!(std::mem::size_of::<oxibonsai_core::BlockFP8E5M2>() == oxibonsai_core::BLOCK_FP8_BYTES);

impl Fp8Kernel for KernelDispatcher {
    /// Dequantize FP8 E4M3FN blocks — tier-aware SIMD dispatch.
    fn dequant_fp8_e4m3(&self, blocks: &[BlockFP8E4M3], output: &mut [f32]) -> KernelResult<()> {
        match self.tier {
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::dequant_fp8_e4m3_avx512(blocks, output)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::dequant_fp8_e4m3_avx2(blocks, output)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::dequant_fp8_e4m3_neon(blocks, output)
            },
            _ => crate::dequant_fp8::dequant_fp8_e4m3(blocks, output),
        }
    }

    /// Dequantize FP8 E5M2 blocks — tier-aware SIMD dispatch.
    fn dequant_fp8_e5m2(&self, blocks: &[BlockFP8E5M2], output: &mut [f32]) -> KernelResult<()> {
        match self.tier {
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::dequant_fp8_e5m2_avx512(blocks, output)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::dequant_fp8_e5m2_avx2(blocks, output)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::dequant_fp8_e5m2_neon(blocks, output)
            },
            _ => crate::dequant_fp8::dequant_fp8_e5m2(blocks, output),
        }
    }

    /// FP8 E4M3FN GEMV — tier-aware SIMD dispatch with optional GPU acceleration.
    ///
    /// Dispatch priority on the `KernelTier::Gpu` path:
    /// 1. Metal (macOS + `metal` feature) — `metal_gemv_fp8_e4m3`.
    /// 2. CUDA (Linux/Windows + `native-cuda` feature) — `cuda_gemv_fp8_e4m3`.
    /// 3. CPU SIMD fallback (AVX-512 / AVX2 / NEON / scalar).
    ///
    /// The raw-byte cast of `blocks` to `*const u8` is sound because
    /// `BlockFP8E4M3` is `#[repr(C)]` with size `BLOCK_FP8_BYTES = 34`.
    fn gemv_fp8_e4m3(
        &self,
        blocks: &[BlockFP8E4M3],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        // GPU dispatch via Metal — macOS only, `metal` feature.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            // SAFETY: BlockFP8E4M3 is repr(C) with size BLOCK_FP8_BYTES (= 34).
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    blocks.as_ptr().cast::<u8>(),
                    blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
                )
            };
            match crate::gpu_backend::metal_gemv_fp8_e4m3(bytes, input, output, n_rows, k) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    // No Metal device or compile failure: fall through to CPU SIMD path.
                    let msg = e.to_string();
                    if !msg.contains("no Metal-capable GPU device") {
                        tracing::warn!(
                            error = %e,
                            "Metal FP8 E4M3 GEMV failed, falling back to CPU SIMD"
                        );
                    }
                }
            }
        }

        // GPU dispatch via CUDA NVRTC — Linux/Windows only, native-cuda feature.
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            // SAFETY: BlockFP8E4M3 is repr(C) with size BLOCK_FP8_BYTES (= 34),
            // validated by the compile-time assert above.  The slice lifetime is
            // tied to `blocks` which outlives this call.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    blocks.as_ptr().cast::<u8>(),
                    blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
                )
            };
            match crate::gpu_backend::cuda_gemv_fp8_e4m3(bytes, input, output, n_rows, k) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    // No CUDA device — fall through to CPU SIMD path silently.
                    // Any other error is logged at warn level and we fall through.
                    let msg = e.to_string();
                    if !msg.contains("no CUDA device") {
                        tracing::warn!(
                            error = %e,
                            "CUDA FP8 E4M3 GEMV failed, falling back to CPU SIMD"
                        );
                    }
                }
            }
        }

        match self.tier {
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::gemv_fp8_e4m3_avx512(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::gemv_fp8_e4m3_avx2(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::gemv_fp8_e4m3_neon(blocks, input, output, n_rows, k)
            },
            _ => crate::gemv_fp8::gemv_fp8_e4m3(blocks, input, output, n_rows, k),
        }
    }

    /// FP8 E5M2 GEMV — tier-aware SIMD dispatch with optional GPU acceleration.
    ///
    /// Mirrors [`gemv_fp8_e4m3`](Self::gemv_fp8_e4m3): Metal → CUDA → CPU SIMD.
    /// The raw-byte cast is sound because `BlockFP8E5M2` is `#[repr(C)]` with
    /// size `BLOCK_FP8_BYTES = 34`.
    fn gemv_fp8_e5m2(
        &self,
        blocks: &[BlockFP8E5M2],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        // GPU dispatch via Metal — macOS only, `metal` feature.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            // SAFETY: BlockFP8E5M2 is repr(C) with size BLOCK_FP8_BYTES (= 34).
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    blocks.as_ptr().cast::<u8>(),
                    blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
                )
            };
            match crate::gpu_backend::metal_gemv_fp8_e5m2(bytes, input, output, n_rows, k) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let msg = e.to_string();
                    if !msg.contains("no Metal-capable GPU device") {
                        tracing::warn!(
                            error = %e,
                            "Metal FP8 E5M2 GEMV failed, falling back to CPU SIMD"
                        );
                    }
                }
            }
        }

        // GPU dispatch via CUDA NVRTC — Linux/Windows only, native-cuda feature.
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            // SAFETY: BlockFP8E5M2 is repr(C) with size BLOCK_FP8_BYTES (= 34),
            // validated by the compile-time assert above.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    blocks.as_ptr().cast::<u8>(),
                    blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
                )
            };
            match crate::gpu_backend::cuda_gemv_fp8_e5m2(bytes, input, output, n_rows, k) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let msg = e.to_string();
                    if !msg.contains("no CUDA device") {
                        tracing::warn!(
                            error = %e,
                            "CUDA FP8 E5M2 GEMV failed, falling back to CPU SIMD"
                        );
                    }
                }
            }
        }

        match self.tier {
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::gemv_fp8_e5m2_avx512(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::gemv_fp8_e5m2_avx2(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::gemv_fp8_e5m2_neon(blocks, input, output, n_rows, k)
            },
            _ => crate::gemv_fp8::gemv_fp8_e5m2(blocks, input, output, n_rows, k),
        }
    }

    /// FP8 E4M3FN GEMM — tier-aware SIMD dispatch.
    fn gemm_fp8_e4m3(
        &self,
        blocks: &[BlockFP8E4M3],
        inputs: &[f32],
        outputs: &mut [f32],
        n_rows: usize,
        k: usize,
        batch: usize,
    ) -> KernelResult<()> {
        match self.tier {
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::gemm_fp8_e4m3_avx512(
                    blocks, inputs, outputs, n_rows, k, batch,
                )
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::gemm_fp8_e4m3_avx2(blocks, inputs, outputs, n_rows, k, batch)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::gemm_fp8_e4m3_neon(blocks, inputs, outputs, n_rows, k, batch)
            },
            _ => crate::gemm_fp8::gemm_fp8_e4m3(blocks, inputs, outputs, n_rows, k, batch),
        }
    }

    /// FP8 E5M2 GEMM — tier-aware SIMD dispatch.
    fn gemm_fp8_e5m2(
        &self,
        blocks: &[BlockFP8E5M2],
        inputs: &[f32],
        outputs: &mut [f32],
        n_rows: usize,
        k: usize,
        batch: usize,
    ) -> KernelResult<()> {
        match self.tier {
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::gemm_fp8_e5m2_avx512(
                    blocks, inputs, outputs, n_rows, k, batch,
                )
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::gemm_fp8_e5m2_avx2(blocks, inputs, outputs, n_rows, k, batch)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::gemm_fp8_e5m2_neon(blocks, inputs, outputs, n_rows, k, batch)
            },
            _ => crate::gemm_fp8::gemm_fp8_e5m2(blocks, inputs, outputs, n_rows, k, batch),
        }
    }

    fn name_fp8(&self) -> &'static str {
        match self.tier {
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => "fp8_avx512",
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => "fp8_avx2",
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => "fp8_neon",
            _ => "fp8_reference",
        }
    }
}

// `impl StandardQuantKernel for KernelDispatcher` (Q4_0, Q8_0, and the six
// K-quant formats), its Metal `quant_gpu_bytes`/`warn_metal_gemv_fallback`
// helpers, and the `Gpu`-arm CPU fallbacks now live in
// `dispatch_std_quant.rs` — see that module's doc comment.

#[cfg(all(test, feature = "gpu"))]
thread_local! {
    /// Records which CPU tier a `cpu_gemv_*_fallback` (this file's Q1_0/
    /// ternary fallbacks, plus `dispatch_std_quant.rs`'s Q4_0/Q8_0/K-quant
    /// ones and `dispatch_prism.rs`'s prism ones) most recently routed onto,
    /// so a K-17 regression test can observe the *routing decision* directly
    /// instead of only the numeric result — the scalar/AVX2/AVX-512/NEON
    /// kernels are numerically equivalent (mod float rounding), so comparing
    /// outputs alone cannot tell "always hardcoded scalar" (the K-17 bug)
    /// apart from "correctly tiered, and this machine's best tier happens to
    /// be Reference". A thread-local, not a process-global atomic, because
    /// `cargo test` runs each test function on its own thread by default,
    /// giving every test its own counter with no cross-test interference.
    /// `pub(crate)` (not private) so the sibling dispatch-split files can
    /// record/read it too.
    pub(crate) static LAST_GPU_FALLBACK_TIER: std::cell::Cell<Option<KernelTier>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(all(test, feature = "gpu"))]
pub(crate) fn record_gpu_fallback_tier(tier: KernelTier) {
    LAST_GPU_FALLBACK_TIER.with(|cell| cell.set(Some(tier)));
}

#[cfg(test)]
mod tests {
    use super::*;
    // `StandardQuantKernel`'s impl moved to `dispatch_std_quant.rs`, but the
    // K-17 regression tests below still call `dispatcher.gemv_q4_0`/
    // `gemv_q8_0` through it, so the trait must be in scope here too. Both
    // call sites (`gpu_tier_gemv_q4_0_routes_through_cpu_simd_fallback` and
    // `gpu_tier_gemv_q8_0_routes_through_cpu_simd_fallback`) are
    // `#[cfg(feature = "gpu")]`-only, so the import itself must be gated the
    // same way or a non-gpu build warns about an unused import.
    #[cfg(feature = "gpu")]
    use crate::traits::StandardQuantKernel;

    thread_local! {
        /// GPU backend probes [`KernelDispatcher::auto_detect`] made on this
        /// thread — the only path to its one-time "GPU unavailable" warning,
        /// so "no probe" proves "no warning" without capturing log output.
        /// Thread-local, so parallel tests cannot disturb each other's count.
        static GPU_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Record one GPU backend probe (called from `auto_detect`).
    #[cfg(feature = "gpu")]
    pub(super) fn note_gpu_probe() {
        GPU_PROBES.with(|probes| probes.set(probes.get() + 1));
    }

    fn gpu_probes() -> usize {
        GPU_PROBES.with(std::cell::Cell::get)
    }

    #[test]
    fn auto_detect_creates_dispatcher() {
        let dispatcher = KernelDispatcher::auto_detect();
        // On x86-64 with AVX2, it should pick Avx2; otherwise Reference
        let _tier = dispatcher.tier();
        let _name = dispatcher.name();
    }

    /// Under an explicit CPU request
    /// (`CpuOnlyBackendScope`, the engine's `Backend::Cpu`) `auto_detect`
    /// must neither probe the GPU nor record the "GPU unavailable" fallback
    /// reason — it records that the CPU was requested explicitly.
    #[test]
    fn auto_detect_under_an_explicit_cpu_scope_skips_the_gpu_probe_and_says_why() {
        let before = gpu_probes();
        let dispatcher = {
            let _scope = crate::gpu_backend::CpuOnlyBackendScope::enter();
            KernelDispatcher::auto_detect()
        };
        assert_eq!(
            gpu_probes(),
            before,
            "an explicit CPU request must not probe (or warn about) the GPU"
        );
        assert_eq!(dispatcher.tier_reason, TierReason::CpuRequestedByScope);
        assert_eq!(dispatcher.tier(), cpu_kernel_tier());
        let reason = dispatcher.effective_tier_reason();
        assert!(
            reason.contains("requested explicitly"),
            "the reason must describe an explicit request, not a fallback: {reason}"
        );
        assert!(
            !reason.contains("fell back") && !reason.contains("not accelerated"),
            "{reason}"
        );
    }

    /// The other arm: with no scope active, `auto_detect` keeps its
    /// auto-detection behaviour — it probes the GPU when the feature is
    /// compiled in (whatever that probe finds on this host), and never
    /// claims an explicit CPU request.
    #[test]
    fn auto_detect_without_the_scope_keeps_auto_detecting() {
        assert!(
            !crate::gpu_backend::cpu_only_backend_active(),
            "precondition: no CPU-only scope on this test thread"
        );
        let before = gpu_probes();
        let dispatcher = KernelDispatcher::auto_detect();
        assert_ne!(dispatcher.tier_reason, TierReason::CpuRequestedByScope);
        #[cfg(feature = "gpu")]
        {
            assert_eq!(gpu_probes(), before + 1, "the GPU must be probed once");
            assert!(matches!(
                dispatcher.tier_reason,
                TierReason::GpuBackend(_) | TierReason::CpuAutoDetectGpuUnavailable
            ));
        }
        #[cfg(not(feature = "gpu"))]
        {
            assert_eq!(gpu_probes(), before, "no GPU feature, nothing to probe");
            assert_eq!(
                dispatcher.tier_reason,
                TierReason::CpuAutoDetectNoGpuFeature
            );
        }
    }

    /// K-12/M-23: every tier's `gemv_f32` must be byte-identical to the
    /// free-function kernel — the LM-head logits the parity gate
    /// measures come out of exactly this path.
    #[test]
    fn gemv_f32_is_byte_identical_across_every_tier() {
        let (out_features, in_features) = (300usize, 64usize);
        let weights: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.011)
            .collect();
        let input: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.003 - 0.1).collect();

        let mut reference = vec![0.0f32; out_features];
        crate::gemv_f32::gemv_f32(&weights, &input, &mut reference, out_features, in_features)
            .expect("free-function kernel");

        let mut tiers = vec![KernelTier::Reference];
        #[cfg(target_arch = "x86_64")]
        {
            tiers.push(KernelTier::Avx2);
            tiers.push(KernelTier::Avx512);
        }
        #[cfg(target_arch = "aarch64")]
        tiers.push(KernelTier::Neon);
        #[cfg(feature = "gpu")]
        tiers.push(KernelTier::Gpu);

        for tier in tiers {
            // `with_tier` clamps an unsupported tier down to one this CPU can
            // actually run, so this stays correct on every machine.
            let dispatcher = KernelDispatcher::with_tier(tier);
            let mut got = vec![0.0f32; out_features];
            dispatcher
                .gemv_f32(&weights, &input, &mut got, out_features, in_features)
                .expect("dispatcher gemv_f32");
            for (row, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
                assert_eq!(g.to_bits(), r.to_bits(), "tier {tier:?}, row {row}");
            }
        }
    }

    #[test]
    fn gemv_f32_empty_weights_zero_the_logits() {
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        let mut out = vec![3.0f32; 8];
        dispatcher
            .gemv_f32(&[], &[1.0; 4], &mut out, 8, 4)
            .expect("zero LM head");
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_backend_accessor_agrees_with_tier() {
        // A `Gpu`-tier dispatcher must expose a backend handle; a CPU-tier
        // one must not.
        let cpu = KernelDispatcher::with_tier(KernelTier::Reference);
        assert!(cpu.gpu_backend().is_none());

        let auto = KernelDispatcher::auto_detect();
        if auto.tier() == KernelTier::Gpu {
            assert!(
                auto.gpu_backend().is_some(),
                "a Gpu-tier dispatcher must have a backend handle"
            );
        } else {
            assert!(auto.gpu_backend().is_none());
        }
    }

    /// Verify that CPU feature detection uses std's is_x86_feature_detected!
    /// and not scirs2_core, which may have issues on some platforms.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn cpu_feature_detection_uses_std() {
        // This test verifies the fix for GitHub issue #4:
        // Token generation hangs at 100% CPU on Windows AMD CPUs.
        //
        // The issue was that scirs2_core might incorrectly detect CPU features,
        // causing the wrong kernel tier to be selected.

        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_fma = is_x86_feature_detected!("fma");

        let dispatcher = KernelDispatcher::auto_detect();
        let tier = dispatcher.tier();

        // If std detects AVX2+FMA, tier should be at least Avx2 (or GPU if available).
        // The original bug (#4) was the dispatcher falling back to Reference on
        // AVX2+FMA hardware; GPU is acceptable since it's strictly faster than AVX2
        // for the workloads where dispatch matters.
        if has_avx2 && has_fma {
            #[cfg(feature = "gpu")]
            let acceptable = matches!(
                tier,
                KernelTier::Avx2 | KernelTier::Avx512 | KernelTier::Gpu
            );
            #[cfg(not(feature = "gpu"))]
            let acceptable = matches!(tier, KernelTier::Avx2 | KernelTier::Avx512);
            assert!(
                acceptable,
                "Expected AVX2/AVX-512/GPU tier when AVX2+FMA detected, got {:?}",
                tier
            );
        }
    }

    #[test]
    fn reference_tier_works() {
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        assert_eq!(dispatcher.tier(), KernelTier::Reference);
        // cli-16: `name()` no longer guesses a quant family.
        assert_eq!(dispatcher.name(), "reference (scalar)");
        assert_eq!(
            dispatcher.kernel_label(oxibonsai_core::GgufTensorType::TQ2_0_g128),
            "TQ2_0_g128 reference (scalar)"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_tier_name() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Avx2);
        assert_eq!(dispatcher.tier(), KernelTier::Avx2);
        assert_eq!(dispatcher.name(), "AVX2+FMA (256-bit)");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_tier_name() {
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Neon);
        assert_eq!(dispatcher.tier(), KernelTier::Neon);
        assert_eq!(dispatcher.name(), "NEON (128-bit)");
        assert_eq!(
            dispatcher.kernel_label(oxibonsai_core::GgufTensorType::Q1_0_g128),
            "Q1_0_g128 NEON (128-bit)"
        );
    }

    #[test]
    fn dispatcher_exposes_ternary_gemv() {
        use crate::TernaryKernel;
        use half::f16;
        use oxibonsai_core::BlockTQ2_0_g128;

        let dispatcher = KernelDispatcher::auto_detect();

        // row 0: all +1 (qs=0xAA = 0b10101010 → four 0b10 codes per byte → +1)
        // row 1: all -1 (qs=0x00 = 0b00000000 → four 0b00 codes per byte → -1)
        let block_pos = BlockTQ2_0_g128 {
            qs: [0xAA; 32],
            d: f16::from_f32(1.0),
        };
        let block_neg = BlockTQ2_0_g128 {
            qs: [0x00; 32],
            d: f16::from_f32(1.0),
        };
        let blocks = vec![block_pos, block_neg];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 2];

        dispatcher
            .gemv_ternary_g128(&blocks, &input, &mut output, 2, 128)
            .expect("gemv_ternary_g128 should succeed");
        assert!(
            (output[0] - 128.0).abs() < 1.0,
            "row0 expected ~128.0, got {}",
            output[0]
        );
        assert!(
            (output[1] + 128.0).abs() < 1.0,
            "row1 expected ~-128.0, got {}",
            output[1]
        );
    }

    #[test]
    fn dispatcher_ternary_reference_tier() {
        use crate::TernaryKernel;
        use half::f16;
        use oxibonsai_core::BlockTQ2_0_g128;

        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        let blocks = vec![BlockTQ2_0_g128 {
            qs: [0xAA; 32],
            d: f16::from_f32(1.0),
        }];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];

        dispatcher
            .gemv_ternary_g128(&blocks, &input, &mut output, 1, 128)
            .expect("gemv_ternary_g128 should succeed");
        assert!((output[0] - 128.0).abs() < 1.0);
    }

    #[test]
    fn ternary_upload_non_gpu_returns_none() {
        use crate::TernaryKernel;
        use half::f16;
        use oxibonsai_core::BlockTQ2_0_g128;

        // Reference tier has no GPU — upload_weights_ternary must return None.
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        let block = BlockTQ2_0_g128 {
            qs: [0xAAu8; 32],
            d: f16::from_f32(1.0),
        };
        let handle = dispatcher.upload_weights_ternary(&[block]);
        assert!(
            handle.is_none(),
            "expected None for non-GPU tier, got {:?}",
            handle
        );
    }

    // `with_tier`/`try_with_tier` clamping (K-03/sec-14) and
    // `effective_tier_reason` are now tested alongside `clamp_tier_to_cpu`/
    // `select_tier` themselves in `crate::tier`'s own test module.

    // ── K-17: GPU tier falls back to the CPU SIMD tier, not scalar ──────

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_gemv_q4_0_routes_through_cpu_simd_fallback() {
        use oxibonsai_core::BlockQ4_0;

        LAST_GPU_FALLBACK_TIER.with(|c| c.set(None));

        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let blocks: Vec<BlockQ4_0> = Vec::new();
        // `in_features = 1` is not a multiple of QK_STD (32), so
        // `q_std_gpu_bytes` rejects the shape before any real Metal call is
        // attempted — this makes the test deterministic on every machine/CI
        // runner regardless of whether a real GPU device is present.
        let input = vec![0.0f32; 1];
        let mut output = vec![0.0f32; 1];

        // The Ok/Err outcome is not interesting here — a degenerate 1-wide
        // shape legitimately errors on every tier, scalar included — only
        // whether routing went through `cpu_gemv_q4_0_fallback` is.
        let _ = dispatcher.gemv_q4_0(&blocks, &input, &mut output, 1, 1);

        let routed = LAST_GPU_FALLBACK_TIER.with(|c| c.get());
        assert_eq!(
            routed,
            Some(KernelDispatcher::cpu_tier()),
            "KernelTier::Gpu arm of gemv_q4_0 must route through \
             cpu_gemv_q4_0_fallback (K-17), not call gemv_q4_0_scalar directly"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_gemv_q8_0_routes_through_cpu_simd_fallback() {
        use oxibonsai_core::BlockQ8_0;

        LAST_GPU_FALLBACK_TIER.with(|c| c.set(None));

        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let blocks: Vec<BlockQ8_0> = Vec::new();
        let input = vec![0.0f32; 1];
        let mut output = vec![0.0f32; 1];

        let _ = dispatcher.gemv_q8_0(&blocks, &input, &mut output, 1, 1);

        let routed = LAST_GPU_FALLBACK_TIER.with(|c| c.get());
        assert_eq!(
            routed,
            Some(KernelDispatcher::cpu_tier()),
            "KernelTier::Gpu arm of gemv_q8_0 must route through \
             cpu_gemv_q8_0_fallback (K-17), not call gemv_q8_0_scalar directly"
        );
    }
}
