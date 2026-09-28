//! [`KernelTier`] selection: the enum itself, why a [`KernelDispatcher`]
//! ended up on its current tier (`TierReason`), and the pure CPU-feature
//! detection logic (no [`KernelDispatcher`] field access) that picks one.
//!
//! Split out of `dispatch.rs` purely for file size (wave-1 / wave-1.5
//! addenda: `dispatch.rs` was 1997/2000 lines with `LinearLayer` and
//! `ModelVariant` dispatch still to add). Every item here is re-exported or
//! used from `dispatch.rs`, so callers keep addressing them as
//! `oxibonsai_kernels::{KernelTier, cpu_kernel_tier}` / `crate::dispatch::*`
//! exactly as before the split.

use crate::dispatch::KernelDispatcher;

/// Kernel implementation tier, ordered from slowest to fastest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelTier {
    /// Pure scalar Rust — correctness reference.
    Reference,
    /// AVX2 + FMA (256-bit SIMD, x86-64).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512F + AVX-512BW + AVX-512VL (512-bit SIMD, x86-64).
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// NEON (128-bit SIMD, AArch64).
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// GPU-accelerated (Metal / CUDA via scirs2-core).
    #[cfg(feature = "gpu")]
    Gpu,
}

impl std::fmt::Display for KernelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reference => write!(f, "reference"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => write!(f, "avx2+fma"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => write!(f, "avx512f+bw+vl"),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => write!(f, "neon"),
            #[cfg(feature = "gpu")]
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

/// Why a [`KernelDispatcher`] ended up on its current [`KernelTier`] — the
/// data behind [`KernelDispatcher::effective_tier_reason`] (perf-13).
///
/// Kept as a small `Copy` enum rather than a pre-formatted `String` because
/// dispatcher construction is on a per-call hot path for some entry points:
/// `gemv_q4_0`/`gemv_q8_0` build a fresh `KernelDispatcher::with_tier(cpu_kernel_tier())`
/// for every weight matrix on every token. The human-readable text is only
/// assembled on demand, in [`KernelDispatcher::effective_tier_reason`].
///
/// `pub(crate)`, not private: constructed and matched from `dispatch.rs`,
/// a sibling module rather than a descendant, which enum-variant visibility
/// (unlike struct-field visibility) allows once the enum itself is at least
/// as visible as its user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TierReason {
    /// GPU tier with a real, accelerated backend (`backend.name()`).
    #[cfg(feature = "gpu")]
    GpuBackend(&'static str),
    /// GPU tier from a pre-built handle via [`KernelDispatcher::with_gpu`].
    #[cfg(feature = "gpu")]
    GpuHandle(&'static str),
    /// CPU tier chosen by [`KernelDispatcher::auto_detect`]'s CPU feature
    /// probe because the `gpu` feature was not compiled in (GPU was never
    /// attempted).
    #[cfg(not(feature = "gpu"))]
    CpuAutoDetectNoGpuFeature,
    /// CPU tier chosen by [`KernelDispatcher::auto_detect`] after a GPU
    /// backend was tried but was not accelerated (perf-13).
    #[cfg(feature = "gpu")]
    CpuAutoDetectGpuUnavailable,
    /// CPU tier chosen by [`KernelDispatcher::auto_detect`] because a
    /// [`CpuOnlyBackendScope`](crate::gpu_backend::CpuOnlyBackendScope) is
    /// active on this thread — the CPU was requested explicitly (the engine's
    /// `Backend::Cpu`), so no GPU backend was probed and no
    /// "GPU unavailable" warning applies.
    CpuRequestedByScope,
    /// Tier explicitly requested via `with_tier`/`try_with_tier` and valid
    /// on this CPU (no demotion).
    Requested,
    /// Tier explicitly requested via `with_tier` but not supported by this
    /// CPU; demoted (K-03/sec-14). Carries the tier that was requested,
    /// before demotion.
    Demoted(KernelTier),
    /// Constructed via the `unsafe` [`KernelDispatcher::with_tier_unchecked`]
    /// (benchmarks only) — not re-validated against the CPU's feature set.
    Unchecked,
}

impl KernelDispatcher {
    /// Emit a one-time (per process) `WARN` that GPU auto-detection fell
    /// back to a CPU tier, which perf-13 measured at roughly 1/7 the GPU
    /// tier's throughput with, previously, no diagnostic above `INFO`.
    #[cfg(feature = "gpu")]
    pub(crate) fn warn_gpu_unavailable_once(backend_name: &str) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                backend = backend_name,
                "GPU tier unavailable (backend not accelerated); auto-detect fell back to a \
                 CPU SIMD tier, which is substantially slower (perf-13). Call \
                 KernelDispatcher::try_with_tier(KernelTier::Gpu) to fail hard instead of \
                 silently degrading."
            );
        }
    }

    /// Re-validate a requested tier against the CPU's actual feature set,
    /// demoting to a tier the hardware genuinely supports (K-03/sec-14).
    ///
    /// Only ever demotes on x86-64, where `Avx512`/`Avx2` are re-checked
    /// with `is_x86_feature_detected!`; on aarch64, NEON is the ISA baseline
    /// so the tier is returned unchanged regardless of what a scirs2
    /// `has_neon` probe reports. `Reference` and (when compiled in) `Gpu`
    /// always pass through unchanged. A no-op for a tier the hardware
    /// already supports, so `with_tier(cpu_kernel_tier())`
    /// (`gemv_q4_0.rs`/`gemv_q8_0.rs`) keeps behaving exactly as before.
    pub(crate) fn clamp_tier_to_cpu(tier: KernelTier) -> KernelTier {
        #[cfg(target_arch = "x86_64")]
        {
            match tier {
                KernelTier::Avx512 => {
                    let ok = is_x86_feature_detected!("avx512f")
                        && is_x86_feature_detected!("avx512bw")
                        && is_x86_feature_detected!("avx512vl");
                    if !ok {
                        let demoted = if is_x86_feature_detected!("avx2")
                            && is_x86_feature_detected!("fma")
                        {
                            KernelTier::Avx2
                        } else {
                            KernelTier::Reference
                        };
                        tracing::warn!(
                            requested = %tier,
                            demoted_to = %demoted,
                            "requested kernel tier not supported by this CPU, demoting"
                        );
                        return demoted;
                    }
                }
                KernelTier::Avx2 => {
                    let ok = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
                    if !ok {
                        tracing::warn!(
                            requested = %tier,
                            demoted_to = %KernelTier::Reference,
                            "requested kernel tier not supported by this CPU, demoting"
                        );
                        return KernelTier::Reference;
                    }
                }
                _ => {}
            }
        }
        tier
    }

    /// Select best tier based on detected capabilities.
    ///
    /// On x86_64, validates scirs2_core detection against Rust's built-in
    /// `is_x86_feature_detected!` macros to ensure correct tier selection.
    /// This prevents issues where scirs2_core might incorrectly detect
    /// CPU features on certain platforms (e.g., Windows AMD CPUs).
    pub(crate) fn select_tier(caps: &scirs2_core::simd::detect::CpuFeatures) -> KernelTier {
        #[cfg(target_arch = "x86_64")]
        {
            // Use Rust's built-in feature detection as the source of truth.
            // scirs2_core detection may be unreliable on some platforms.
            let has_avx512f = is_x86_feature_detected!("avx512f");
            let has_avx512bw = is_x86_feature_detected!("avx512bw");
            let has_avx512vl = is_x86_feature_detected!("avx512vl");
            let has_avx2 = is_x86_feature_detected!("avx2");
            let has_fma = is_x86_feature_detected!("fma");

            // Log if there's a mismatch between scirs2_core and std detection
            if caps.has_avx512f != has_avx512f {
                tracing::warn!(
                    scirs2_avx512f = caps.has_avx512f,
                    std_avx512f = has_avx512f,
                    "CPU feature detection mismatch for AVX-512F, using std detection"
                );
            }
            if caps.has_avx2 != has_avx2 || caps.has_fma != has_fma {
                tracing::warn!(
                    scirs2_avx2 = caps.has_avx2,
                    scirs2_fma = caps.has_fma,
                    std_avx2 = has_avx2,
                    std_fma = has_fma,
                    "CPU feature detection mismatch for AVX2/FMA, using std detection"
                );
            }

            // AVX-512 requires all three: avx512f, avx512bw, and avx512vl
            if has_avx512f && has_avx512bw && has_avx512vl {
                tracing::debug!("AVX-512 (F+BW+VL) detected, selecting AVX-512 tier");
                return KernelTier::Avx512;
            }
            if has_avx2 && has_fma {
                tracing::debug!("AVX2 + FMA detected, selecting AVX2 tier");
                return KernelTier::Avx2;
            }

            // Log fallback to reference tier
            tracing::warn!(
                has_avx512f,
                has_avx512bw,
                has_avx512vl,
                has_avx2,
                has_fma,
                "No SIMD acceleration available, falling back to reference tier (this will be slow)"
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            if caps.has_neon {
                return KernelTier::Neon;
            }
        }

        // Suppress unused-variable warning on architectures with no SIMD paths
        let _ = caps;
        KernelTier::Reference
    }
}

/// Best **CPU** kernel tier for the current machine, detected once and cached.
///
/// Unlike [`KernelDispatcher::auto_detect`], this never selects the GPU tier and
/// never logs, so it is cheap enough to call on every kernel invocation. It is
/// used by the standard-quant (Q4_0 / Q8_0) free-function entry points to build
/// a lightweight CPU dispatcher per call without repeating feature detection.
pub fn cpu_kernel_tier() -> KernelTier {
    static TIER: std::sync::OnceLock<KernelTier> = std::sync::OnceLock::new();
    *TIER.get_or_init(|| {
        let caps = scirs2_core::simd::detect::get_cpu_features();
        KernelDispatcher::select_tier(caps)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_tier_reference_is_a_no_op() {
        // `Reference` is always valid — clamp must never touch it.
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        assert_eq!(dispatcher.tier(), KernelTier::Reference);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn with_tier_neon_is_never_demoted_on_aarch64() {
        // K-03/sec-14: clamp_tier_to_cpu must only ever demote on x86-64 — a
        // scirs2 `has_neon = false` report must not downgrade the baseline
        // aarch64 tier.
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Neon);
        assert_eq!(dispatcher.tier(), KernelTier::Neon);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn with_tier_is_a_no_op_for_a_supported_tier() {
        // gemv_q4_0.rs / gemv_q8_0.rs call `with_tier(cpu_kernel_tier())`
        // every GEMV — the clamp must never demote an already-valid tier.
        let tier = cpu_kernel_tier();
        let dispatcher = KernelDispatcher::with_tier(tier);
        assert_eq!(
            dispatcher.tier(),
            tier,
            "with_tier must be a no-op for a tier this CPU already supports"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn with_tier_demotes_avx512_when_unsupported() {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            return; // this host genuinely supports AVX-512; nothing to demote
        }
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Avx512);
        assert_ne!(
            dispatcher.tier(),
            KernelTier::Avx512,
            "with_tier must demote a tier this CPU cannot execute"
        );
    }

    #[test]
    fn try_with_tier_reference_always_succeeds() {
        let dispatcher =
            KernelDispatcher::try_with_tier(KernelTier::Reference).expect("Reference is universal");
        assert_eq!(dispatcher.tier(), KernelTier::Reference);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn try_with_tier_neon_always_succeeds_on_aarch64() {
        let dispatcher = KernelDispatcher::try_with_tier(KernelTier::Neon)
            .expect("NEON is the aarch64 baseline");
        assert_eq!(dispatcher.tier(), KernelTier::Neon);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn try_with_tier_rejects_unsupported_avx512() {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            return; // this host genuinely supports AVX-512
        }
        let result = KernelDispatcher::try_with_tier(KernelTier::Avx512);
        assert!(
            result.is_err(),
            "try_with_tier must hard-fail for a tier this CPU cannot execute"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn try_with_tier_gpu_errs_or_succeeds_consistently_with_auto_detect() {
        // Whatever `auto_detect` finds for the GPU tier, `try_with_tier`
        // must agree: hard error when unavailable, a real Gpu-tier
        // dispatcher when available (perf-13).
        let auto = KernelDispatcher::auto_detect();
        let result = KernelDispatcher::try_with_tier(KernelTier::Gpu);
        if auto.tier() == KernelTier::Gpu {
            assert_eq!(
                result
                    .expect("auto_detect found an accelerated GPU backend")
                    .tier(),
                KernelTier::Gpu
            );
        } else {
            assert!(
                result.is_err(),
                "try_with_tier(Gpu) must hard-error when no accelerated backend exists"
            );
        }
    }

    #[test]
    fn with_tier_unchecked_preserves_the_requested_tier() {
        // SAFETY: Reference is always executable.
        let dispatcher = unsafe { KernelDispatcher::with_tier_unchecked(KernelTier::Reference) };
        assert_eq!(dispatcher.tier(), KernelTier::Reference);
    }

    #[test]
    fn effective_tier_reason_mentions_the_tier() {
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        let reason = dispatcher.effective_tier_reason();
        assert!(
            reason.contains("reference"),
            "effective_tier_reason() = {reason:?} should name the tier"
        );
    }
}
