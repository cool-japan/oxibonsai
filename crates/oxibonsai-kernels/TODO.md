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

## Deferred (CUDA hardware required)

- [~] **HIGH PRIORITY — CUDA Q1 batched-prefill cap-of-8 silent correctness bug** (discovered /cont audit 2026-05-03; deferred — CUDA hardware required)
  - **Goal:** Fix the silent correctness bug in three CUDA NVRTC kernels that mirrors the Metal Q1 cap-of-8 bug fixed on the MSL side in /ultra slice #1 (2026-05-03). Without this fix, NVIDIA users running batched prefill with any prompt > 8 tokens receive silently corrupted logits — output columns 8..N are zeroed by `simd_sum(0)`.
  - **Bug detail:** `crates/oxibonsai-kernels/src/gpu_backend/cuda_prefill_kernels.rs` has `const unsigned int cols = batch_size < 8u ? batch_size : 8u;` at three sites:
    - `gemm_q1_g128_v7` (line ~87)
    - `gemm_q1_g128_v7_residual` (line ~161)
    - `fused_gate_up_swiglu_gemm_q1` (line ~241)
  - **Fix (mechanical, mirrors MSL fix):** Wrap each kernel's inner column-processing in `for (unsigned int col_base = 0u; col_base < batch_size; col_base += 8u) { const unsigned int cols_remaining = batch_size - col_base; const unsigned int cols = cols_remaining < 8u ? cols_remaining : 8u; ... }`. Reset accumulators (`col_sums[8]`, `gate_sums[8]`, `up_sums[8]`) at the top of each outer iteration. Replace `inputs+col*k` with `inputs+(col_base+cc)*k`. Final write-back uses `outputs[(col_base+cc)*n_rows+row]` (and `residual[(col_base+cc)*n_rows+row]` for the residual variant; `outputs[(col_base+cc)*inter_size+pos]` for the fused gate+up). Reference: see `MSL_GEMM_Q1_G128_V7` lines ~59–61 in `kernel_sources/prefill.rs` after the slice #1 fix landed — the CUDA fix is the C-syntax mirror of the same code.
  - **Files:** `crates/oxibonsai-kernels/src/gpu_backend/cuda_prefill_kernels.rs` (3 kernels, ~150–250 LoC delta total). Dispatch geometry preserved (no changes to dispatchers in `metal_dispatch.rs` or `cuda_dispatch*.rs`). Stale comment to clean up: `metal_dispatch.rs:273-274` mentions "unlike `dispatch_gemm_q1_v7` whose kernel silently caps at 8 columns" — that comment is post-MSL-fix stale.
  - **Tests:** Mirror `crates/oxibonsai-model/tests/metal_prefill_q1_parity_tests.rs::test_batched_q1_prefill_matches_per_position_batch12` on the CUDA side. Build a 2-layer synthetic Q1 GGUF in `std::env::temp_dir()`, run a 12-token prompt through CUDA batched prefill vs CUDA per-position fused, assert logits match within `1e-3` (fp32 noise) at every position. Test must be `#[cfg(all(feature = "native-cuda", any(target_os = "linux", target_os = "windows")))]` gated. Discriminating-test verification: temporarily reintroduce the cap on one kernel and confirm test fails; revert and confirm it passes — same protocol the MSL slice followed.
  - **Risk:** Cannot validate on macOS dev machines. The fix is mechanical and mirrors the proven MSL fix exactly, so risk is low *as a fix*; but landing untested CUDA kernels into production violates the project's `// TODO(CI-GPU)` discipline. Recommend running on a CUDA host (or CI-GPU runner) before merging.
  - **Status (2026-05-03):** Mechanical fix written in `cuda_prefill_kernels.rs`, mirroring the proven MSL fix (`MSL_GEMM_Q1_G128_V7`, `prefill.rs` lines 59–86). All three kernels now wrap column processing in a `for col_base` outer loop; accumulators reset inside the loop; all `col * k` / `col * n_rows + row` indices updated to `(col_base + col) * ...`. **NOT VALIDATED on CUDA hardware.** Before merging, must run on a CUDA host: (1) confirm NVRTC compiles all 3 kernels without PTX error; (2) run a 12-token prompt through CUDA batched Q1 prefill vs CUDA per-position fused; assert logits match within 1e-3 at every position; (3) confirm discriminating test `batch_size=12` passes (would fail on original cap-of-8). Mirror test: `metal_prefill_q1_parity_tests.rs::test_batched_q1_prefill_matches_per_position_batch12`.
  - **Severity:** **1 (CRITICAL)** — silent correctness bug shipped in 0.1.3+. Higher priority than Slices B+C below (those are optimization gaps with correct fallbacks; this is wrong outputs in shipped production).
  - **Discovered:** /cont audit 2026-05-03, paired with the MSL fix from /ultra 2026-05-03 slice #1. Memory entry: `kernel_pattern_capof8.md`.

- [ ] CUDA batched TQ2 prefill (deferred — CUDA hardware required for parity validation)
  - **Goal:** Mirror the new Metal batched TQ2 prefill (`MSL_GEMM_TQ2_G128_V7`, `encode_full_forward_prefill_ternary`, `try_metal_full_forward_prefill_ternary`) on the CUDA NVRTC backend. Replace the Err-guard at `crates/oxibonsai-model/src/model/types/forward_cuda.rs:418` with a real call into a new `try_cuda_prefill_ternary` that processes all prompt tokens in batched dispatches per layer.
  - **Design:** New `cuda_prefill_ternary` module mirroring the structure of `cuda_prefill.rs::try_cuda_prefill` for Q1. New PTX/NVRTC kernel `cuda_gemm_tq2_g128_v7` that decodes TQ2_0_g128 blocks identically to the existing per-position TQ2 GEMV (find decode in `cuda_full_layer/encode_ternary.rs`). Reuse format-agnostic dispatchers: batched RMSNorm, per-token attention loop, batched FFN structure, KV cache management, RoPE.
  - **Files:** `crates/oxibonsai-kernels/src/gpu_backend/cuda_prefill.rs` (new entry `try_cuda_prefill_ternary` + new `encode_prefill_layer_ternary`), `crates/oxibonsai-kernels/src/gpu_backend/cuda_prefill_kernels.rs` (new PTX `cuda_gemm_tq2_g128_v7`), `crates/oxibonsai-kernels/src/gpu_backend/cuda_full_layer/mod.rs` (export), `crates/oxibonsai-model/src/model/types/forward_cuda.rs:418` (replace Err-guard).
  - **Prerequisites:** none beyond existing CUDA Q1 batched prefill infrastructure.
  - **Tests:** Bit-exact parity test (CUDA batched ternary prefill vs CUDA per-position fused ternary) using a 2-layer synthetic ternary fixture (mirror `crates/oxibonsai-model/tests/metal_prefill_ternary_parity_tests.rs`). Gated `#[cfg(all(feature = "native-cuda", any(target_os = "linux", target_os = "windows")))]`.
  - **Risk:** Cannot validate on macOS dev machines — must run on CUDA hardware. Recommend CI-GPU runner before merging. Estimated ~810–1010 LoC including new PTX kernel.
  - **Discovered:** /stub-check 2026-05-03 (paired with Slice A which shipped Metal batched TQ2 prefill).

- [ ] CUDA batched TQ2 prefill-verify (depends on Slice B above)
  - **Goal:** Add the speculative-decode verification entry point `try_cuda_prefill_verify_ternary` that runs the batched ternary prefill + per-position GPU argmax (LM-head GEMV + token-ID download), replacing the Err-guard at `crates/oxibonsai-model/src/model/types/forward_cuda.rs:524`.
  - **Design:** Thin wrapper on top of Slice B that dispatches the LM-head + argmax stage per position, returning a Vec<u32> of greedy token IDs. Mirrors the Metal `try_metal_full_forward_prefill_verify_ternary` shape.
  - **Files:** `crates/oxibonsai-kernels/src/gpu_backend/cuda_prefill.rs` (new `try_cuda_prefill_verify_ternary`), `crates/oxibonsai-model/src/model/types/forward_cuda.rs:524` (replace Err-guard).
  - **Prerequisites:** Slice B implementation.
  - **Tests:** Greedy-match parity test (CUDA batched verify vs CUDA per-position sequential verify). Same gating as Slice B.
  - **Risk:** Same hardware-validation gap as Slice B. Estimated ~150–200 LoC on top of Slice B.
  - **Discovered:** /stub-check 2026-05-03.
