# oxibonsai-kernels TODO

> 1-bit + ternary quantized compute kernels with SIMD dispatch, parallelism, and GPU backends
> Version 0.1.3 — 291 tests passing
> Last updated: 2026-05-03

## Status: Stable (mature, complete)

NEON, AVX-512, tiled GEMM, packing, property tests, cross-tier correctness,
performance tuning, cache-line alignment, prefetch hints, SIMD ternary,
fused Metal TQ2 full-forward, and native CUDA NVRTC paths all shipped.

## Done

- [x] Reference scalar dequant (`dequant_1bit_g128`)
- [x] Reference scalar GEMV (`gemv_1bit_g128`)
- [x] Reference scalar GEMM (`gemm_1bit_g128`)
- [x] `OneBitKernel` trait abstraction
- [x] `KernelDispatcher` with `KernelTier` enum
- [x] AVX2+FMA SIMD kernels — `simd_avx2.rs` (dequant, gemv, gemm, 426 lines)
- [x] SciRS2-Core-powered runtime SIMD detection (`scirs2_core::simd::detect::get_cpu_features()`)
- [x] Rayon parallel GEMV (row-parallel, threshold 64 rows) (`parallel.rs`)
- [x] Rayon parallel GEMM (batch-parallel, threshold 4) (`parallel.rs`)
- [x] Criterion benchmark suite (dequant, gemv, gemm — reference vs AVX2 vs parallel)
- [x] **AVX-512 kernels** — `simd_avx512.rs` (~701 lines), 512-bit operations, `KernelTier::Avx512` in dispatcher
- [x] **NEON (ARM) kernels** — `simd_neon.rs` (~768 lines), NEON intrinsics, `KernelTier::Neon` in dispatcher
- [x] **Tiled GEMM** — Cache-blocking tiled matrix multiply (`tiled.rs`, 622 lines)
- [x] **Packing** — Panel packing for cache efficiency (`packing.rs`, 452 lines)
- [x] **Parallel tiled GEMM** — Rayon-parallel tiled paths (`parallel_tiled.rs`)
- [x] **Property tests** — `proptest` roundtrip: dequant → requant identity, GEMV distributivity, GEMM associativity (`proptest_kernels.rs`)
- [x] **Cross-tier correctness** — Automated reference vs AVX2 vs AVX-512 output comparison (`cross_tier.rs`)
- [x] **Parallel GEMV tuning** — `tuning.rs` with PlatformProfile, TunedThresholds, auto-detection of optimal thresholds per platform
- [x] **Cache-line alignment** — `aligned.rs` with AlignedBuffer/AlignedBlocks, 64-byte aligned allocations for SIMD loads
- [x] **Prefetch hints** — `prefetch.rs` with PrefetchConfig, software prefetch via x86 _mm_prefetch / ARM _prefetch, platform-specific dispatch

## Ternary Bonsai

- [x] Scalar ternary kernels: `dequant_ternary.rs`, `gemv_ternary.rs`, `gemm_ternary.rs`
- [x] **Phase 10 — SIMD ternary kernels (NEON / AVX2 / AVX-512)**: `gemv_tq2_0_g128_*`, `dequant_tq2_0_g128_*`, `gemm_tq2_0_g128_*`
- [x] `TernaryKernel` trait in `traits.rs` + `impl TernaryKernel for KernelDispatcher` in `dispatch.rs`

## GPU Backends

- [x] `GpuBackendTrait` abstraction + `CpuBackend` baseline
- [x] `scirs2_backend::Scirs2Backend` — portable CUDA/Metal via scirs2-core
- [x] **Phase 11 — Metal TQ2 GEMV** fused kernels (`metal_graph.rs`, `metal_prefill.rs`)
- [x] **Phase 12 — Native CUDA NVRTC backend** (`cuda_full_layer.rs`, `cuda_graph/`, `cuda_kernels.rs`, `cuda_prefill*.rs`, `cuda_attn_kernels.rs`) with CUDA Graph execution
- [x] **Phase 13.x — Fused Metal TQ2 full-forward** (`metal_full_layer/`) — single command buffer, ~50 tok/s on 1.7B ternary (~13× speedup)
- [x] Runtime NVRTC kernel sources (`kernel_sources/`: attention, decode, decode_ternary, prefill, utility, archive)
