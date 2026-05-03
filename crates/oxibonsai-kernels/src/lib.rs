#![cfg_attr(target_arch = "aarch64", feature(stdarch_aarch64_prefetch))]

//! # oxibonsai-kernels
//!
//! 1-bit Q1\_0\_g128 compute kernels for OxiBonsai.
//!
//! Provides dequantization and fused matrix-multiply operations optimized
//! for the PrismML 1-bit weight format. The kernels are organised in a
//! tiered dispatch architecture that auto-selects the fastest implementation
//! available on the current CPU:
//!
//! | Tier | Feature gate | Instruction set |
//! |------|-------------|-----------------|
//! | **Reference** | always | Pure scalar Rust (correctness baseline) |
//! | **AVX2+FMA** | `simd-avx2` | 256-bit SIMD (x86-64) |
//! | **AVX-512** | `simd-avx512` | 512-bit SIMD (x86-64) |
//! | **NEON** | `simd-neon` | 128-bit SIMD (AArch64) |
//!
//! Runtime dispatch is handled by [`KernelDispatcher`] which queries
//! SciRS2-Core's SIMD capability cache on construction.
//!
//! ## Key Kernels
//!
//! | Kernel | Description |
//! |--------|-------------|
//! | [`dequant::dequant_1bit_g128`] | Unpack 128 sign bits + FP16 scale → FP32 |
//! | [`gemv::gemv_1bit_g128`] | 1-bit weight matrix × FP32 vector (single-token decode) |
//! | [`gemm::gemm_1bit_g128`] | 1-bit weight matrix × FP32 matrix (multi-token prefill) |
//!
//! ## Trait
//!
//! All tiers implement [`OneBitKernel`] so callers are agnostic to the
//! underlying SIMD level.

#[cfg(all(feature = "metal", target_os = "macos"))]
#[macro_use]
extern crate objc;

pub mod gpu_backend;
#[cfg(feature = "gpu")]
pub use gpu_backend::Scirs2Backend;
pub use gpu_backend::{
    gpu_gemv_1bit, gpu_matmul, select_backend, CpuBackend, DeviceBuffer, GpuBackend,
    GpuBackendTrait, GpuError, LaunchConfig,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use gpu_backend::{
    build_cached_weights, build_cached_weights_ternary_only, print_gpu_profile_summary,
    try_metal_ffn, try_metal_forward_greedy_ternary, try_metal_full_forward,
    try_metal_full_forward_cached, try_metal_full_forward_prefill,
    try_metal_full_forward_prefill_verify, try_metal_full_forward_ternary, try_metal_full_layer,
    try_metal_prefill_ternary, try_metal_prefill_verify_ternary, try_metal_qkv, CachedLayerWeights,
    CachedModelWeights, FullForwardLayerParams, FullForwardLayerParamsTernary, MetalGraph,
    MetalGraphError, MetalWeightHandle,
};

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub use gpu_backend::{
    try_cuda_ffn, try_cuda_full_forward, try_cuda_full_forward_ternary,
    try_cuda_full_forward_ternary_with_gpu_lm_head, try_cuda_full_forward_with_gpu_lm_head,
    try_cuda_full_layer, try_cuda_prefill, try_cuda_qkv, CudaCachedLayerWeights,
    CudaFullForwardLayerParams, CudaFullForwardLayerParamsTernary, CudaGraph, CudaGraphError,
    NativeCudaBackend,
};

pub mod dequant;
pub mod dequant_ternary;
pub mod dispatch;
pub mod error;
pub mod gemm;
pub mod gemm_ternary;
pub mod gemv;
pub mod gemv_ternary;
pub mod packing;
pub mod parallel;
pub mod parallel_tiled;
#[cfg(target_arch = "x86_64")]
pub mod simd_avx2;
#[cfg(target_arch = "x86_64")]
pub mod simd_avx512;
#[cfg(target_arch = "aarch64")]
pub mod simd_neon;
pub mod tiled;
pub mod traits;
pub mod weight_cache;

pub mod aligned;
pub mod prefetch;
pub mod simd_float_ops;
pub mod tuning;

pub use aligned::{AlignedBlocks, AlignedBuffer};
pub use dispatch::{KernelDispatcher, KernelTier};
pub use error::{KernelError, KernelResult};
pub use parallel::{gemm_ternary_g128_par, gemv_ternary_g128_par};
pub use parallel_tiled::{gemm_adaptive_ternary, gemv_adaptive, gemv_adaptive_ternary};
pub use prefetch::{PrefetchConfig, PrefetchLocality, PrefetchStrategy};
pub use simd_float_ops::{rms_norm_simd, rope_apply_simd, silu_simd, softmax_simd, swiglu_simd};
pub use traits::{OneBitKernel, TernaryKernel};
pub use tuning::{PlatformProfile, TunedThresholds, TuningSummary};
pub use weight_cache::GpuWeightHandle;
