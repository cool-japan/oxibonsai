# oxibonsai-kernels

[![Version](https://img.shields.io/badge/version-0.2.4-blue.svg)](https://crates.io/crates/oxibonsai-kernels)

Q1_0_g128 (1-bit), TQ2_0_g128 (ternary) and Bonsai 2 (PQ2_0 / PTQ1_0 / group-64 Q2_0) compute kernels for OxiBonsai — dequantization, GEMV, GEMM, fused full-forward — plus the blockwise Hadamard transform and Gated-DeltaNet kernels of the 27B hybrid, and GEMV kernels for standard GGUF quant (Q4_0/Q8_0) and K-quant (Q2_K–Q8_K).

Implements the full compute stack for sub-2-bit inference: scalar
reference kernels, SIMD-accelerated tiers (AVX2+FMA, AVX-512, NEON), tiled
cache-blocked GEMM, parallel Rayon dispatch, and production GPU backends
(Metal fused full-forward and the hybrid runner, native CUDA via NVRTC — run on one RTX A4000 (CUDA 12.0, x86_64 Linux) in this release; the Bonsai 2 hybrid kernels have no CUDA forward and were not run, apart from a partial PQ2_0 GEMV check (CUDA-P09) — plus scirs2-core backend).

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

**Status:** Stable (mature, complete) — 1,171 tests (`cargo nextest list -p oxibonsai-kernels --all-features`).

## Features

- `dequant_1bit_g128` / `dequant_tq2_0_g128` — dequantize Q1_0_g128 / TQ2_0_g128 blocks to f32
- `gemv_1bit_g128` / `gemv_tq2_0_g128` — fused 1-bit / ternary GEMV (matrix-vector multiply)
- `gemm_1bit_g128` / `gemm_tq2_0_g128` — fused 1-bit / ternary GEMM (batched matrix multiply)
- `gemv_q4_0` / `gemv_q8_0` — standard GGUF quant GEMV (Q4_0 4-bit / Q8_0 8-bit): scalar reference + AVX2 + AVX-512 SIMD tiers, NEON (`simd_q_std_neon`), and Rayon row-parallel (`gemv_q4_0_par` / `gemv_q8_0_par`)
- `gemv_q2k` / `gemv_q3k` / `gemv_q4k` / `gemv_q5k` / `gemv_q6k` / `gemv_q8k` — K-quant (Q2_K–Q8_K) GEMV kernels sharing one Rayon row-parallel driver
- `KernelDispatcher::auto_detect()` — selects the best SIMD tier at runtime
- Tiled GEMM with cache-line alignment and software prefetch hints
- Parallel dispatch via Rayon (`gemv_*_par`, `gemm_*_par`, tiled parallel paths)
- Platform tuning: `PlatformProfile`, `TunedThresholds`
- `OneBitKernel` and `TernaryKernel` traits unified through `KernelDispatcher`
- GPU backend trait (`GpuBackendTrait`) with three concrete paths:
  - **Metal**: fused full-forward TQ2 path (single command buffer) — ~50 tok/s on 1.7B ternary (~13× speedup); standalone GEMV kernels for Q4_0/Q8_0 and K-quant (Q2_K–Q8_K), dispatched per layer by `oxibonsai-model`'s `Linear*::forward()` before it falls back to CPU
  - **Native CUDA**: NVRTC-compiled kernels with CUDA Graph execution (multi-encoding pass); prefill path with dedicated attention kernels for KV-cache population; per-position GEMV kernels also cover Q4_0/Q8_0, K-quant, and FP8 — batched-prefill kernels exist for these formats; the Q4_0/Q8_0 batch prefill runs for prompts of more than 16 tokens at position 0, with a K/V read-back into the host cache (CUDA-P11, RTX A4000), while K-quant and FP8 prefill stay on the sequential per-position path. The Q4_0/Q8_0/K-quant/FP8 decode GEMVs upload the weight matrix on every call, so they can be slower than an AVX-512 CPU
  - **scirs2-core backend**: portable CUDA/Metal via `scirs2-core::gpu`

## SIMD Tiers

| Tier | Feature Flag | Width | Platform |
|------|-------------|-------|----------|
| Reference (scalar) | *(default)* | N/A | All |
| AVX2+FMA | `simd-avx2` | 256-bit | x86-64 |
| AVX-512 | `simd-avx512` | 512-bit | x86-64 |
| NEON | `simd-neon` | 128-bit | AArch64 |

> **Tiers are selected at runtime, not at build time.** `KernelDispatcher` uses `is_x86_feature_detected!` to pick AVX-512 only when AVX-512F+BW+VL are all present, otherwise AVX2+FMA, otherwise the scalar reference. Every SIMD function carries a per-function `#[target_feature(...)]` attribute, so all tiers are always compiled into a single x86-64 binary that is safe on every x86-64 CPU and falls back automatically (AVX-512 → AVX-2 → scalar) with no SIGILL. The `Feature Flag` column above is therefore informational: the `simd-avx2` / `simd-avx512` / `simd-neon` features do **not** gate tier selection (see below).
>
> AVX-512 is absent from Intel *consumer* CPUs since Alder Lake (Raptor Lake / Meteor Lake / Arrow Lake / Lunar Lake have none) and mainly benefits Xeon / HEDT and AMD Zen 4+; consumer hardware auto-selects the AVX-2 tier.
>
> **Opt-in INT8 dot-product tier.** Setting `OXIBONSAI_KERNEL_TIER` to `int8-scalar`, `neon-int8`, `neon-dot`, `neon-i8mm` or `avx512-vnni` runs the CPU GEMV/GEMM of the native `TQ2_0_g128` / `Q1_0_g128` formats (all seven kernel entry points) and the PrismML `PQ2_0` / group-64 `Q2_0` formats on INT8 kernels (`SDOT` on `neon-dot`, `SMMLA` for batched GEMM on `neon-i8mm`, `VPDPBUSD` on `avx512-vnni`). The activation is quantized to INT8 once per call (one scale per weight block) and every block accumulates exactly in `i32`, so the tier is lossy against the f32 default (cosine against f32 0.999991–0.999993 on real weights; `neon-i8mm` decodes the 1.7B on the CPU 8.76× faster) and **never selected by default** — unset, the f32 kernels run bit-identically. The GPU tier is never diverted while a GPU prefill actually runs; when one is declined or fails, the CPU batched prefill (and the batched CPU embedding pass on any engine) runs on the selected INT8 tier. `PTQ1_0` has no INT8 kernel, and `avx512-vnni` compiles but has never run on VNNI hardware.

## Cargo Features

| Feature | Purpose |
|---------|---------|
| `simd-avx2` | No-op, accepted for compatibility — the AVX2+FMA tier is always compiled and auto-selected at runtime (does not gate the tier) |
| `avx2` | Alias for `simd-avx2` (Cargo shorthand) |
| `simd-avx512` | No-op, accepted for compatibility — the AVX-512 tier is always compiled and auto-selected at runtime when AVX-512F+BW+VL are present (does not gate the tier) |
| `simd-neon` | No-op, accepted for compatibility — the NEON tier is always compiled and auto-selected at runtime on AArch64 (does not gate the tier) |
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
oxibonsai-kernels = { version = "0.2.4", features = ["simd-avx2"] }
```

```rust
use oxibonsai_kernels::KernelDispatcher;

let dispatcher = KernelDispatcher::auto_detect();
// dispatcher selects AVX2, AVX-512, NEON, or scalar automatically
```

## License

Apache-2.0 — COOLJAPAN OU
