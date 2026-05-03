# oxibonsai-kernels

[![Version](https://img.shields.io/badge/version-0.1.3-blue.svg)](https://crates.io/crates/oxibonsai-kernels)

Q1_0_g128 (1-bit) and TQ2_0_g128 (ternary) compute kernels for OxiBonsai — dequantization, GEMV, GEMM, fused full-forward.

Implements the full compute stack for 1-bit and ternary inference: scalar
reference kernels, SIMD-accelerated tiers (AVX2+FMA, AVX-512, NEON), tiled
cache-blocked GEMM, parallel Rayon dispatch, and production GPU backends
(Metal fused full-forward, native CUDA via NVRTC, plus scirs2-core backend).

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

**Status:** Stable (mature, complete) — 291 tests passing.

## Features

- `dequant_1bit_g128` / `dequant_tq2_0_g128` — dequantize Q1_0_g128 / TQ2_0_g128 blocks to f32
- `gemv_1bit_g128` / `gemv_tq2_0_g128` — fused 1-bit / ternary GEMV (matrix-vector multiply)
- `gemm_1bit_g128` / `gemm_tq2_0_g128` — fused 1-bit / ternary GEMM (batched matrix multiply)
- `KernelDispatcher::auto_detect()` — selects the best SIMD tier at runtime
- Tiled GEMM with cache-line alignment and software prefetch hints
- Parallel dispatch via Rayon (`gemv_*_par`, `gemm_*_par`, tiled parallel paths)
- Platform tuning: `PlatformProfile`, `TunedThresholds`
- `OneBitKernel` and `TernaryKernel` traits unified through `KernelDispatcher`
- GPU backend trait (`GpuBackendTrait`) with three concrete paths:
  - **Metal**: fused full-forward TQ2 path (single command buffer) — ~50 tok/s on 1.7B ternary (~13× speedup)
  - **Native CUDA**: NVRTC-compiled kernels with CUDA Graph execution (multi-encoding pass); prefill path with dedicated attention kernels for KV-cache population
  - **scirs2-core backend**: portable CUDA/Metal via `scirs2-core::gpu`

## SIMD Tiers

| Tier | Feature Flag | Width | Platform |
|------|-------------|-------|----------|
| Reference (scalar) | *(default)* | N/A | All |
| AVX2+FMA | `simd-avx2` | 256-bit | x86-64 |
| AVX-512 | `simd-avx512` | 512-bit | x86-64 |
| NEON | `simd-neon` | 128-bit | AArch64 |

## Cargo Features

| Feature | Purpose |
|---------|---------|
| `simd-avx2` | Enable AVX2+FMA SIMD kernels (x86-64) |
| `avx2` | Alias for `simd-avx2` (Cargo shorthand) |
| `simd-avx512` | Enable AVX-512 SIMD kernels (x86-64) |
| `simd-neon` | Enable NEON SIMD kernels (AArch64) |
| `neon` | Alias for `simd-neon` (Cargo shorthand) |
| `metal` | Metal GPU backend + fused full-forward (macOS only) |
| `native-cuda` | Native CUDA NVRTC backend via `cudarc` (Linux/Windows) |
| `cuda` | scirs2-core CUDA backend (implies `gpu`) |
| `gpu` | Enable `scirs2-core/gpu` baseline GPU trait support |
| `wasm` | WebAssembly target adjustments |

## Usage

```toml
[dependencies]
# Auto-detect at runtime:
oxibonsai-kernels = { version = "0.1.3", features = ["simd-avx2"] }
```

```rust
use oxibonsai_kernels::KernelDispatcher;

let dispatcher = KernelDispatcher::auto_detect();
// dispatcher selects AVX2, AVX-512, NEON, or scalar automatically
```

## License

Apache-2.0 — COOLJAPAN OU
