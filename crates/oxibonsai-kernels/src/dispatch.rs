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

use crate::dequant;
use crate::error::KernelResult;
use crate::gemm;
use crate::gemv;
use crate::traits::{OneBitKernel, TernaryKernel};
use crate::weight_cache::GpuWeightHandle;
use oxibonsai_core::tensor::BlockQ1_0G128;
#[cfg(feature = "gpu")]
use std::sync::Arc;

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
}

impl std::fmt::Debug for KernelDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // GPU backend is a trait object without Debug; show only the tier.
        f.debug_struct("KernelDispatcher")
            .field("tier", &self.tier)
            .finish_non_exhaustive()
    }
}

impl KernelDispatcher {
    /// Create a dispatcher that auto-detects the best available kernel tier.
    ///
    /// Queries SciRS2-Core's cached `CpuFeatures` to determine the
    /// optimal tier for the current CPU.
    pub fn auto_detect() -> Self {
        // Try GPU first when the feature is compiled in.
        #[cfg(feature = "gpu")]
        {
            let backend = crate::gpu_backend::select_backend();
            if backend.is_accelerated() {
                tracing::info!(backend = backend.name(), "GPU backend available");
                return Self {
                    tier: KernelTier::Gpu,
                    gpu_backend: Some(Arc::from(backend)),
                };
            }
        }

        let caps = scirs2_core::simd::detect::get_cpu_features();
        let tier = Self::select_tier(caps);
        tracing::info!(tier = %tier, "selected kernel tier");
        Self {
            tier,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
        }
    }

    /// Create a dispatcher with a specific tier (for testing/benchmarks).
    pub fn with_tier(tier: KernelTier) -> Self {
        Self {
            tier,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
        }
    }

    /// Create a dispatcher that uses the GPU backend with the given handle.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(backend: Arc<dyn crate::gpu_backend::GpuBackendTrait>) -> Self {
        Self {
            tier: KernelTier::Gpu,
            gpu_backend: Some(backend),
        }
    }

    /// Get the selected kernel tier.
    pub fn tier(&self) -> KernelTier {
        self.tier
    }

    /// Select best tier based on detected capabilities.
    ///
    /// On x86_64, validates scirs2_core detection against Rust's built-in
    /// `is_x86_feature_detected!` macros to ensure correct tier selection.
    /// This prevents issues where scirs2_core might incorrectly detect
    /// CPU features on certain platforms (e.g., Windows AMD CPUs).
    fn select_tier(caps: &scirs2_core::simd::detect::CpuFeatures) -> KernelTier {
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

/// Minimum number of rows before the GPU path is worthwhile.
///
/// Below this threshold the overhead of host-to-device transfer exceeds the
/// compute savings, so we fall back to the best SIMD tier.
#[cfg(feature = "gpu")]
const GPU_MIN_ROWS: usize = 1024;

impl KernelDispatcher {
    /// Return the best CPU-only tier for use as GPU fallback.
    ///
    /// Uses Rust's built-in `is_x86_feature_detected!` macros directly
    /// for reliable detection, bypassing scirs2_core which may have issues
    /// on certain platforms.
    #[cfg(feature = "gpu")]
    fn cpu_tier() -> KernelTier {
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
                crate::simd_avx512::gemv_1bit_g128_avx512_prefetch(blocks, input, output, n_rows, k)
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

    fn gemv(
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
                crate::simd_avx512::gemv_1bit_g128_avx512_prefetch(blocks, input, output, n_rows, k)
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

    fn gemm(
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

    fn name(&self) -> &'static str {
        match self.tier {
            KernelTier::Reference => "Q1_0_g128 reference (scalar)",
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => "Q1_0_g128 AVX2+FMA (256-bit)",
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => "Q1_0_g128 AVX-512 (512-bit)",
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => "Q1_0_g128 NEON (128-bit)",
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => "Q1_0_g128 GPU (accelerated)",
        }
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

    fn gemv_ternary_g128(
        &self,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
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

    fn gemm_ternary_g128(
        &self,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_detect_creates_dispatcher() {
        let dispatcher = KernelDispatcher::auto_detect();
        // On x86-64 with AVX2, it should pick Avx2; otherwise Reference
        let _tier = dispatcher.tier();
        let _name = dispatcher.name();
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
        assert_eq!(dispatcher.name(), "Q1_0_g128 reference (scalar)");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_tier_name() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Avx2);
        assert_eq!(dispatcher.tier(), KernelTier::Avx2);
        assert_eq!(dispatcher.name(), "Q1_0_g128 AVX2+FMA (256-bit)");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_tier_name() {
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Neon);
        assert_eq!(dispatcher.tier(), KernelTier::Neon);
        assert_eq!(dispatcher.name(), "Q1_0_g128 NEON (128-bit)");
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
}
